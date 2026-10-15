"""AP-style backbone training and target-side rows for the Hyser 1dof protocol H-A.

Thin wrapper around ``APBackbone`` (per-session K x 256 read-in tables + shared
Mamba-3 trunk + linear head).  The 000688 pretraining loop is tied to NWB
records, so this module has its own trainer but reuses ``APBackbone``,
``drift_augment`` and the same loss (trunk MSE + lam_head * head MSE +
lam_white * whitening).  No shared module is edited.

Conventions
-----------
* Trial index in a session: ``(finger-1)*3 + (sample-1)`` (15 trials of 1250 bins).
* Source split: sample 3 of every finger is held out for source validation.
* Input z-score: per-channel mean/std over the given trials; source sessions use
  their own training trials, targets use the B calibration trials (neural data only).
* Labels: z-scored with pooled source-training stats; R2 is always computed after
  mapping back to physical units.  Every trial is decoded as an independent
  sequence from zero state (the trials are separate recordings).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ap_backbone import APBackbone
from .hyser_data import N_CH, N_FINGER, load_session
from .session_frontend import drift_augment

SOURCE_VAL_SAMPLE = 3


# ----------------------------------------------------------------- stats ----

def input_stats(x: np.ndarray, floor: float = 1e-3):
    """Per-channel mean/std over [trials, T, C]; std floored at floor * mean std."""
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.maximum(sd, floor * max(sd.mean(), 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32)


def r2_scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Macro (mean of per-channel) and pooled R2 over [N, 5] arrays in physical units."""
    ss_res = ((truth - pred) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0)) ** 2).sum(0)
    per = 1.0 - ss_res / np.maximum(ss_tot, 1e-18)
    pooled = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-18)
    ratio = float(np.std(pred) / max(np.std(truth), 1e-18))
    return {"r2": float(per.mean()), "r2_pooled": float(pooled), "r2_per_channel": per.tolist(),
            "unit_ratio": ratio}


# ---------------------------------------------------------------- sources ---

class SourceSet:
    """Session1 of all subjects: raw envelopes, z-scored tensors, label stats."""

    def __init__(self, subjects=range(1, 21), cache_dir=None, session=1):
        kw = {} if cache_dir is None else {"cache_dir": cache_dir}
        self.subjects = list(subjects)
        self.raw, self.y, self.sample, self.finger = [], [], [], []
        for s in self.subjects:
            blob = load_session(s, session, **kw)
            self.raw.append(blob["x"]); self.y.append(blob["y"])
            self.sample.append(blob["sample"]); self.finger.append(blob["finger"])
        self.train_idx = [np.where(s != SOURCE_VAL_SAMPLE)[0] for s in self.sample]
        self.val_idx = [np.where(s == SOURCE_VAL_SAMPLE)[0] for s in self.sample]
        self.stats = [input_stats(x[i]) for x, i in zip(self.raw, self.train_idx)]
        ytr = np.concatenate([y[i].reshape(-1, N_FINGER) for y, i in zip(self.y, self.train_idx)])
        self.y_mean = ytr.mean(0).astype(np.float32)
        self.y_std = ytr.std(0).astype(np.float32)
        self.xz = [zscore(x, mu, sd) for x, (mu, sd) in zip(self.raw, self.stats)]
        self.yz = [((y - self.y_mean) / self.y_std).astype(np.float32) for y in self.y]

    def label_stats(self):
        return {"y_mean": self.y_mean, "y_std": self.y_std}


# ------------------------------------------------------------------ model ---

def build_model(n_sessions, *, k, width, layers, state_size, dropout, device):
    return APBackbone([N_CH] * n_sessions, N_FINGER, k=k, width=width, layers=layers,
                      state_size=state_size, dropout=dropout).to(device)


def trunk_hash(model) -> str:
    h = hashlib.sha256()
    for p in model.base.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def decode_trials(model, x_z: torch.Tensor, table: torch.Tensor, chunk=32) -> torch.Tensor:
    """x_z [n, T, 256], table [K, 256] -> normalized prediction [n, T, 5] (no grad)."""
    outs = []
    with torch.inference_mode():
        for lo in range(0, x_z.shape[0], chunk):
            outs.append(model.base(x_z[lo:lo + chunk] @ table.T))
    return torch.cat(outs)


def to_phys(pred_norm, y_mean, y_std):
    return pred_norm * y_std + y_mean


def score_prediction(pred_norm: np.ndarray, y_phys: np.ndarray, y_mean, y_std) -> dict:
    p = to_phys(pred_norm.reshape(-1, N_FINGER), y_mean, y_std)
    return r2_scores(y_phys.reshape(-1, N_FINGER), p)


@torch.inference_mode()
def validate(model, src: SourceSet, xz_t, yz_t, device):
    """Macro mean over source sessions of query R2 (physical units) on held-out sample-3 trials."""
    model.eval()
    scores, head_scores = [], []
    for s in range(len(src.subjects)):
        idx = torch.as_tensor(src.val_idx[s], device=device)
        x = xz_t[s][idx]
        pred = decode_trials(model, x, model.tables[s]).cpu().numpy()
        truth = src.y[s][src.val_idx[s]]
        scores.append(score_prediction(pred, truth, src.y_mean, src.y_std)["r2"])
        head = (x @ model.tables[s].T) @ model.head.weight.T + model.head.bias
        head_scores.append(score_prediction(head.cpu().numpy(), truth, src.y_mean, src.y_std)["r2"])
    return {"val_r2": float(np.mean(scores)), "head_r2": float(np.mean(head_scores)),
            "per_session": [float(v) for v in scores]}


def train_backbone(src: SourceSet, out_dir, *, k=16, width=128, layers=2, state_size=32, dropout=0.1,
                   lr=1e-3, weight_decay=1e-2, batch=16, max_steps=4000, val_every=50, patience=12,
                   lam_head=0.5, lam_white=0.1, gain_sigma=0.3, offset_sigma=0.1, channel_dropout=0.05,
                   table_init="supervised", seed=0, device="cuda:0", log=print):
    """Train on the source train trials; early-stop on source validation R2 (target never loaded)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, lr=lr,
               weight_decay=weight_decay, batch=batch, max_steps=max_steps, val_every=val_every,
               patience=patience, lam_head=lam_head, lam_white=lam_white, gain_sigma=gain_sigma,
               offset_sigma=offset_sigma, channel_dropout=channel_dropout, table_init=table_init, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    n = len(src.subjects)
    model = build_model(n, k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, device=dev)
    xz_t = [torch.from_numpy(x).to(dev) for x in src.xz]
    yz_t = [torch.from_numpy(y).to(dev) for y in src.yz]
    if table_init == "supervised":
        wins = [x[i].reshape(-1, N_CH) for x, i in zip(src.xz, src.train_idx)]
        labs = [y[i].reshape(-1, N_FINGER) for y, i in zip(src.yz, src.train_idx)]
        model.init_from_supervised([torch.from_numpy(w) for w in wins], [torch.from_numpy(l) for l in labs])
    elif table_init != "random":
        raise ValueError(table_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pairs = [(s, int(t)) for s in range(n) for t in src.train_idx[s]]
    best, best_step, bad, history = -np.inf, 0, 0, []
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        model.train()
        pick = rng.choice(len(pairs), size=batch, replace=False)
        xs = torch.stack([xz_t[pairs[i][0]][pairs[i][1]] for i in pick])
        ys = torch.stack([yz_t[pairs[i][0]][pairs[i][1]] for i in pick])
        sess = torch.as_tensor([pairs[i][0] for i in pick], device=dev)
        xs = drift_augment(xs, gain_sigma, offset_sigma, channel_dropout)
        tables = torch.stack(list(model.tables))                          # [S, K, N]
        z = torch.einsum("btn,bkn->btk", xs, tables[sess])
        dec, head = model.base(z), model.head(z)
        loss_dec = (dec - ys).square().mean()
        loss = loss_dec + lam_head * (head - ys).square().mean()
        mask = torch.ones(z.shape[:2], dtype=torch.bool, device=dev)
        loss = loss + lam_white * model.whitening_penalty(z, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % val_every == 0 or step == 1:
            m = validate(model, src, xz_t, yz_t, dev)
            ev = {"step": step, "loss": float(loss_dec), "elapsed": time.perf_counter() - t0,
                  "val_r2": m["val_r2"], "head_r2": m["head_r2"]}
            history.append(ev)
            log(json.dumps(ev))
            if m["val_r2"] > best:
                best, best_step, bad = m["val_r2"], step, 0
                torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_step": step,
                            "val_r2": best, "val_per_session": m["per_session"],
                            "subjects": src.subjects, "y_mean": src.y_mean, "y_std": src.y_std,
                            "stats": src.stats}, out / "best.pt")
            else:
                bad += 1
                if bad >= patience:
                    break
    summary = {"cfg": cfg, "best_step": best_step, "best_val_r2": best, "steps_run": step,
               "stopped_early": bool(bad >= patience), "budget_rule_ok": bool(best_step <= 0.8 * max_steps),
               "seconds": time.perf_counter() - t0,
               "n_params_trunk": int(sum(p.numel() for p in model.base.parameters())),
               "n_params_tables": int(sum(t.numel() for t in model.tables))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "history.json").write_text(json.dumps(history))
    return summary


def load_backbone(path, device="cuda:0"):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    model = build_model(len(blob["subjects"]), k=c["k"], width=c["width"], layers=c["layers"],
                        state_size=c["state_size"], dropout=0.0, device=torch.device(device))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob


# ---------------------------------------------------- target-side routines --

def fit_tables(model, x_cal, y_cal, init_tables, *, steps=300, lr=3e-3, chunk=64):
    """Gradient fit of read-in tables only through the frozen trunk (ui_prior style).

    x_cal [D, B, T, N] z-scored inputs, y_cal [D, B, T, 5] normalized labels,
    init_tables [D, K, N].  The D problems are independent (separate Adam
    elements, loss is a sum over problems); full-batch over the B trials.
    Returns tables [D, K, N] and the final per-problem loss.
    """
    for p in model.base.parameters():
        if p.requires_grad:
            raise RuntimeError("trunk must be frozen")
    table = torch.nn.Parameter(init_tables.clone())
    opt = torch.optim.Adam([table], lr=lr)
    d, b = x_cal.shape[:2]
    last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_each = torch.zeros(d, device=table.device)
        for lo in range(0, d, max(1, chunk // b)):
            hi = min(d, lo + max(1, chunk // b))
            z = torch.einsum("dbtn,dkn->dbtk", x_cal[lo:hi], table[lo:hi])
            dec = model.base(z.reshape(-1, z.shape[2], z.shape[3])).reshape(hi - lo, b, z.shape[2], -1)
            le = (dec - y_cal[lo:hi]).square().mean((1, 2, 3))
            le.sum().backward()
            loss_each[lo:hi] = le.detach()
        opt.step()
        last = loss_each
    return table.detach(), last.cpu().numpy()


def causal_ema(x: np.ndarray, beta: float) -> np.ndarray:
    """Causal exponential smoothing along axis -2 (time): y[t] = beta*y[t-1] + (1-beta)*x[t]."""
    if beta <= 0:
        return x
    from scipy.signal import lfilter
    return lfilter([1 - beta], [1.0, -beta], x, axis=-2).astype(np.float32)


def ridge_fit(xz, yz, lam, ema=0.0):
    """Closed-form ridge 256 -> 5 on z-scored (optionally EMA-smoothed) features; returns W, b."""
    f = causal_ema(xz, ema).reshape(-1, xz.shape[-1]).astype(np.float64)
    y = yz.reshape(-1, yz.shape[-1]).astype(np.float64)
    fm, ym = f.mean(0), y.mean(0)
    fc = f - fm
    a = fc.T @ fc + lam * len(fc) / 1000.0 * np.eye(fc.shape[1])
    w = np.linalg.solve(a, fc.T @ (y - ym))
    return w, ym - fm @ w


def ridge_predict(xz, w, b, ema=0.0):
    f = causal_ema(xz, ema)
    return f @ w + b
