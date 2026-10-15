"""AP-style backbone training and target-side rows for the NinaPro DB6 protocol D-A.

Thin wrapper around ``APBackbone`` (per-session K x 14 read-in tables + shared
Mamba-3 trunk + linear head), mirroring ``hyser_backbone.py`` but for per-bin
classification of 8 classes (rest + 7 grasps) with cross-entropy.

Conventions
-----------
* A trial is one ``rerepetition`` run (rest gap + one grasp repetition), see
  ``db6_data``.  Every trial is decoded as an independent sequence from zero
  state; trials are right-padded with zeros to ``LMAX`` bins and masked in losses
  and metrics (the model is causal, so padding never changes earlier outputs).
* Source split: repetitions in ``SOURCE_VAL_REPS`` of every grasp are held out
  for source validation (3 of 12 per grasp, 21 of 84 trials per session).
* Input z-score: per-channel mean/std over the valid bins of the given trials;
  source sessions use their own training trials, targets use the B calibration
  trials (neural data only, no labels).
* Metrics: per-bin balanced accuracy and macro-F1 over the 8 classes, and the
  per-trial majority-vote grasp accuracy (majority over bins predicted non-rest).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ap_backbone import APBackbone
from .db6_data import DEFAULT_CACHE, DEFAULT_TAU_MS, N_CH, N_CLASS, load_session, trial_slices
from .session_frontend import drift_augment

LMAX = 704                      # >= longest trial (700 bins) over all 100 sessions
SOURCE_VAL_REPS = (4, 8, 12)
IGNORE = -100


# ------------------------------------------------------------- data tensors --

class Sess:
    """One session's trials, right-padded: x [n,LMAX,N] raw envelope, y [n,LMAX] (IGNORE on padding)."""

    def __init__(self, key, blob):
        sl = trial_slices(blob)
        n = len(sl)
        self.key = tuple(key)
        self.x = np.zeros((n, LMAX, N_CH), np.float32)
        self.y = np.full((n, LMAX), IGNORE, np.int64)
        self.lens = np.array([s.stop - s.start for s in sl])
        if self.lens.max() > LMAX:
            raise ValueError(f"{key}: trial of {self.lens.max()} bins exceeds LMAX={LMAX}")
        for i, s in enumerate(sl):
            self.x[i, :self.lens[i]] = blob["x"][s]
            self.y[i, :self.lens[i]] = blob["y"][s]
        self.grasp = blob["trial_grasp"].astype(np.int64)
        self.rep = blob["trial_rep"].astype(np.int64)

    @property
    def n(self):
        return self.x.shape[0]


def load_sess(key, tau_ms=DEFAULT_TAU_MS, cache_dir=DEFAULT_CACHE) -> Sess:
    return Sess(key, load_session(*key, tau_ms=tau_ms, cache_dir=cache_dir))


def valid_mask(lens, length=LMAX):
    return (np.arange(length)[None, :] < np.asarray(lens)[:, None])


def input_stats(x, lens, floor: float = 1e-3):
    """Per-channel mean/std over the valid bins of [n,L,C] trials; std floored at floor * mean std."""
    flat = x[valid_mask(lens, x.shape[1])].astype(np.float64)
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.maximum(sd, floor * max(sd.mean(), 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore(x, mu, sd, lens):
    """z-score and re-zero the padding."""
    m = valid_mask(lens, x.shape[1])[..., None]
    return (((x - mu) / sd) * m).astype(np.float32)


def draw_calibration(b: int, rng, grasp, rep):
    """Class-balanced draw of b trials: round-robin over a random grasp order, random unused repetition.

    Rest is present in every trial.  Returns sorted trial indices; the grasp coverage is
    ``np.bincount(grasp[idx], minlength=8)[1:]`` (counts differ by at most 1 across grasps).
    """
    order = list(rng.permutation(7) + 1)
    chosen, used = [], {g: set() for g in order}
    i = 0
    while len(chosen) < b:
        g = int(order[i % 7]); i += 1
        free = [t for t in np.where(grasp == g)[0] if t not in used[g]]
        if not free:
            continue
        t = int(rng.choice(free)); used[g].add(t); chosen.append(t)
    return sorted(chosen)


def coverage(idx, grasp):
    """Per-grasp counts (7,) of a calibration set."""
    return np.bincount(grasp[np.asarray(idx)], minlength=N_CLASS)[1:].tolist()


# ----------------------------------------------------------------- metrics ---

def confusion(truth, pred, n=N_CLASS):
    return np.bincount(truth * n + pred, minlength=n * n).reshape(n, n)


def class_scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Balanced accuracy (mean per-class recall) and macro-F1 over the classes present in truth."""
    cm = confusion(truth, pred)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    present = support > 0
    recall = tp[present] / support[present]
    prec = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
    rec_all = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    f1 = np.where(prec + rec_all > 0, 2 * prec * rec_all / np.maximum(prec + rec_all, 1e-12), 0.0)
    return {"bal_acc": float(recall.mean()), "macro_f1": float(f1[present].mean()),
            "acc": float(tp.sum() / cm.sum())}


def vote_grasp(pred_trial: np.ndarray) -> int:
    """Majority class among bins predicted non-rest; 0 (rest) if no bin is predicted non-rest."""
    nz = pred_trial[pred_trial > 0]
    return int(np.bincount(nz, minlength=N_CLASS).argmax()) if nz.size else 0


def score_logits(logits: np.ndarray, y: np.ndarray, lens, grasp) -> dict:
    """logits [n,L,8], y [n,L] (IGNORE on padding), per-trial grasp [n] -> metric dict."""
    pred = logits.argmax(-1)
    m = y != IGNORE
    out = class_scores(y[m], pred[m])
    votes = np.array([vote_grasp(pred[i, :lens[i]]) for i in range(len(lens))])
    out["vote_acc"] = float((votes == grasp).mean())
    return out


# ---------------------------------------------------------------- sources ---

class SourceSet:
    """Source sessions (days 1-2): raw trials, per-session train/val split, z-scored tensors."""

    def __init__(self, keys, tau_ms=DEFAULT_TAU_MS, cache_dir=DEFAULT_CACHE, val_reps=SOURCE_VAL_REPS):
        self.keys = [tuple(k) for k in keys]
        self.tau_ms = tau_ms
        self.sess = [load_sess(k, tau_ms, cache_dir) for k in self.keys]
        self.train_idx = [np.where(~np.isin(s.rep, val_reps))[0] for s in self.sess]
        self.val_idx = [np.where(np.isin(s.rep, val_reps))[0] for s in self.sess]
        self.stats = [input_stats(s.x[i], s.lens[i]) for s, i in zip(self.sess, self.train_idx)]
        self.xz = [zscore(s.x, mu, sd, s.lens) for s, (mu, sd) in zip(self.sess, self.stats)]

    def subject_of(self, i):
        return self.keys[i][0]


# ------------------------------------------------------------------ model ---

def build_model(n_sessions, *, k, width, layers, state_size, dropout, device):
    return APBackbone([N_CH] * n_sessions, N_CLASS, k=k, width=width, layers=layers,
                      state_size=state_size, dropout=dropout).to(device)


def trunk_hash(model) -> str:
    h = hashlib.sha256()
    for p in model.base.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def decode_trials(model, x_z: torch.Tensor, table: torch.Tensor, chunk=32) -> torch.Tensor:
    """x_z [n,L,N], table [K,N] -> logits [n,L,8] (no grad)."""
    outs = []
    with torch.inference_mode():
        for lo in range(0, x_z.shape[0], chunk):
            outs.append(model.base(x_z[lo:lo + chunk] @ table.T))
    return torch.cat(outs)


def masked_ce(logits, y):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=IGNORE)


@torch.inference_mode()
def validate(model, src: SourceSet, xz_t, y_t, device):
    """Mean over source sessions of balanced accuracy / macro-F1 on the held-out repetitions."""
    model.eval()
    bal, f1, head_bal = [], [], []
    for s in range(len(src.keys)):
        vi = src.val_idx[s]
        idx = torch.as_tensor(vi, device=device)
        x = xz_t[s][idx]
        logits = decode_trials(model, x, model.tables[s]).cpu().numpy()
        sc = score_logits(logits, src.sess[s].y[vi], src.sess[s].lens[vi], src.sess[s].grasp[vi])
        bal.append(sc["bal_acc"]); f1.append(sc["macro_f1"])
        head = ((x @ model.tables[s].T) @ model.head.weight.T + model.head.bias).cpu().numpy()
        head_bal.append(score_logits(head, src.sess[s].y[vi], src.sess[s].lens[vi], src.sess[s].grasp[vi])["bal_acc"])
    return {"val_bal_acc": float(np.mean(bal)), "val_macro_f1": float(np.mean(f1)),
            "head_bal_acc": float(np.mean(head_bal)), "per_session": [float(v) for v in bal]}


def train_backbone(src: SourceSet, out_dir, *, k=8, width=128, layers=2, state_size=32, dropout=0.1,
                   lr=1e-3, weight_decay=1e-2, batch=16, max_steps=6000, val_every=100, patience=12,
                   lam_head=0.5, lam_white=0.1, gain_sigma=0.3, offset_sigma=0.1, channel_dropout=0.05,
                   table_init="supervised", seed=0, device="cuda:0", log=print):
    """Train on the source train trials; early-stop on source-validation balanced accuracy."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, lr=lr,
               weight_decay=weight_decay, batch=batch, max_steps=max_steps, val_every=val_every,
               patience=patience, lam_head=lam_head, lam_white=lam_white, gain_sigma=gain_sigma,
               offset_sigma=offset_sigma, channel_dropout=channel_dropout, table_init=table_init,
               seed=seed, tau_ms=src.tau_ms)
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    n = len(src.keys)
    model = build_model(n, k=k, width=width, layers=layers, state_size=state_size, dropout=dropout, device=dev)
    xz_t = [torch.from_numpy(x).to(dev) for x in src.xz]
    y_t = [torch.from_numpy(s.y).to(dev) for s in src.sess]
    len_t = [torch.from_numpy(s.lens).to(dev) for s in src.sess]
    if table_init == "supervised":
        wins, labs = [], []
        for s in range(n):
            tr = src.train_idx[s]
            m = valid_mask(src.sess[s].lens[tr])
            f = src.xz[s][tr][m]
            lab = np.eye(N_CLASS, dtype=np.float32)[src.sess[s].y[tr][m]]
            wins.append(torch.from_numpy(f)); labs.append(torch.from_numpy(lab - lab.mean(0)))
        model.init_from_supervised(wins, labs)
    elif table_init != "random":
        raise ValueError(table_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pairs = [(s, int(t)) for s in range(n) for t in src.train_idx[s]]
    best, best_step, bad, history = -np.inf, 0, 0, []
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        model.train()
        pick = rng.choice(len(pairs), size=batch, replace=False)
        lens = torch.stack([len_t[pairs[i][0]][pairs[i][1]] for i in pick])
        top = int(lens.max())
        xs = torch.stack([xz_t[pairs[i][0]][pairs[i][1], :top] for i in pick])
        ys = torch.stack([y_t[pairs[i][0]][pairs[i][1], :top] for i in pick])
        sess = torch.as_tensor([pairs[i][0] for i in pick], device=dev)
        mask = ys != IGNORE
        xs = drift_augment(xs, gain_sigma, offset_sigma, channel_dropout) * mask.unsqueeze(-1)
        tables = torch.stack(list(model.tables))                          # [S, K, N]
        z = torch.einsum("btn,bkn->btk", xs, tables[sess])
        dec, head = model.base(z), model.head(z)
        loss_dec = masked_ce(dec, ys)
        loss = loss_dec + lam_head * masked_ce(head, ys)
        loss = loss + lam_white * model.whitening_penalty(z, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % val_every == 0 or step == 1:
            m = validate(model, src, xz_t, y_t, dev)
            ev = {"step": step, "loss": float(loss_dec), "elapsed": time.perf_counter() - t0,
                  "val_bal_acc": m["val_bal_acc"], "val_macro_f1": m["val_macro_f1"], "head_bal_acc": m["head_bal_acc"]}
            history.append(ev)
            log(json.dumps(ev))
            if m["val_bal_acc"] > best:
                best, best_step, bad = m["val_bal_acc"], step, 0
                torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_step": step,
                            "val_bal_acc": best, "val_macro_f1": m["val_macro_f1"],
                            "val_per_session": m["per_session"], "keys": src.keys,
                            "stats": src.stats}, out / "best.pt")
            else:
                bad += 1
                if bad >= patience:
                    break
    summary = {"cfg": cfg, "best_step": best_step, "best_val_bal_acc": best, "steps_run": step,
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
    model = build_model(len(blob["keys"]), k=c["k"], width=c["width"], layers=c["layers"],
                        state_size=c["state_size"], dropout=0.0, device=torch.device(device))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob


# ---------------------------------------------------- target-side routines --

def fit_tables(model, x_cal, y_cal, init_tables, *, steps=300, lr=3e-3, chunk=64):
    """Gradient fit of read-in tables only through the frozen trunk (ui_prior style).

    x_cal [D,B,T,N] z-scored inputs (padding zeroed), y_cal [D,B,T] int labels (IGNORE on padding),
    init_tables [D,K,N].  The D problems are independent (separate Adam elements, loss is a
    sum over problems); full batch over the B trials.  Returns tables [D,K,N] and the final
    per-problem loss.
    """
    for p in model.base.parameters():
        if p.requires_grad:
            raise RuntimeError("trunk must be frozen")
    table = torch.nn.Parameter(init_tables.clone())
    opt = torch.optim.Adam([table], lr=lr)
    d, b = x_cal.shape[:2]
    last = None
    step_d = max(1, chunk // b)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_each = torch.zeros(d, device=table.device)
        for lo in range(0, d, step_d):
            hi = min(d, lo + step_d)
            z = torch.einsum("dbtn,dkn->dbtk", x_cal[lo:hi], table[lo:hi])
            dec = model.base(z.reshape(-1, z.shape[2], z.shape[3])).reshape(hi - lo, b, z.shape[2], -1)
            le = torch.stack([masked_ce(dec[j], y_cal[lo + j]) for j in range(hi - lo)])
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


def onehot(y):
    return np.eye(N_CLASS, dtype=np.float64)[y]


def ridge_fit(xz, y, lens, lam, ema=0.0):
    """Closed-form ridge (N -> 8 one-hot) on z-scored, causally smoothed features of the valid bins.

    xz [n,L,N], y [n,L] labels (IGNORE on padding); returns W [N,8], b [8].  A linear
    least-squares classifier on one-hot targets (LDA-style, closed form).
    """
    f = causal_ema(xz, ema)[valid_mask(lens, xz.shape[1])].astype(np.float64)
    t = onehot(y[valid_mask(lens, xz.shape[1])])
    fm, tm = f.mean(0), t.mean(0)
    fc = f - fm
    a = fc.T @ fc + lam * len(fc) / 1000.0 * np.eye(fc.shape[1])
    w = np.linalg.solve(a, fc.T @ (t - tm))
    return w, tm - fm @ w


def ridge_logits(xz, w, b, ema=0.0):
    return causal_ema(xz, ema) @ w + b


# ---------------------------------------------------------- protocol keys ---

# S2_D2_T2 is a truncated recording (103 s, a single grasp, 12 trials): excluded from the source set.
EXCLUDED_SOURCE = ((2, 2, 2),)


def source_keys(subjects=range(1, 11), days=(1, 2)):
    return [(s, d, t) for s in subjects for d in days for t in (1, 2) if (s, d, t) not in EXCLUDED_SOURCE]


def target_keys(subjects=range(1, 11), days=(3, 4, 5)):
    return [(s, d, t) for s in subjects for d in days for t in (1, 2)]


def latest_source_index(keys, subject):
    """Index of the subject's chronologically latest source session in ``keys``."""
    idx = [i for i, k in enumerate(keys) if k[0] == subject]
    return max(idx, key=lambda i: (keys[i][1], keys[i][2]))
