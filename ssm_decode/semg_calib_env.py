"""Episodic training / evaluation environments for sEMG calibrators (Hyser H-A, DB6 D-A).

An *episode* simulates the target-side calibration inside a SOURCE session: draw B
calibration trials, stream their sums, run the calibrator, decode held-out query trials
of the same source session with the frozen trunk and backprop into the CALIBRATOR only
(the trunk is frozen, ``requires_grad=False``).  Target sessions (Hyser session 2, DB6
days 3-5) are never loaded by the training code.

Priors (identical construction in training and evaluation)
* ``mean``  channel-aligned mean of the source tables, excluding the target subject.
* ``own``   the subject's earlier-session table.  Hyser has one source session per subject,
            so training uses a *stale surrogate* of the own table (blend with the population
            mean + noise) inside subject-fold calibrators that never see the fold's subjects;
            DB6 uses a real earlier session of the same subject.
* ``free``  identity withheld (no prior, no prior columns).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import hyser_backbone as hysb
from .semg_calib import (SemgCalibrator, SemgSpec, SemgStats, TrialBank, build_prior, describe,
                         mean_prior, segment_sums)

HYSER_LABEL_LAGS = (0, 2, 4, 6)
HYSER_B_TRAIN_MAX = 10
STALE_VAL = ((0.3, 0.2), (0.5, 0.4), (0.7, 0.6))     # (blend with population mean, noise sd / table rms)


# ------------------------------------------------------------ session store ----

class SessStore:
    """GPU copy of one session's trials + a lazily filled bank of per-trial sums.

    ``stats(idx)`` only ever computes sums for the requested (calibration) trials; the
    bank therefore never holds data of query trials.
    """

    def __init__(self, x, label_fn, lens, spec: SemgSpec, device):
        self.x = x                       # [n, T, N] float32 on device
        self.label_fn = label_fn         # (i, L) -> [L, D] float64 labels of trial i
        self.lens = np.asarray(lens)
        self.spec = spec
        self.bank = TrialBank(spec)
        self.device = device

    def stats(self, idx) -> SemgStats:
        for i in idx:
            if not self.bank.has(i):
                length = int(self.lens[i])
                xi = self.x[i, :length].double()
                yi = self.label_fn(i, length)
                self.bank.add(i, xi, yi, torch.ones(length, dtype=torch.bool, device=self.device))
        return SemgStats.from_bank(self.bank, idx)


# ----------------------------------------------------------------- helpers ----

def table_rms(tables: torch.Tensor) -> float:
    return float(tables.pow(2).mean().sqrt())


def forward_table(calib: SemgCalibrator, stats: SemgStats, prior, *, grad: bool):
    """Descriptors under no_grad (no learnable part), generator with/without grad."""
    with torch.no_grad():
        d, r, mean, std = describe(stats, prior, calib.spec)
    pr = torch.zeros(calib.spec.k, d.shape[0], device=d.device) if prior is None else prior.to(torch.float32)
    if grad:
        return calib(d, r, pr), mean.float(), std.float()
    with torch.no_grad():
        return calib(d, r, pr), mean.float(), std.float()


def draw_hyser_cal(b, rng, fingers, samples, allowed=(1, 2, 3)):
    """Balanced draw (identical to scripts/run_hyser_eval.draw_calibration when allowed=(1,2,3))."""
    order = list(rng.permutation(5) + 1)
    used = {f: [] for f in order}
    chosen = []
    i = 0
    while len(chosen) < b:
        f = order[i % 5]; i += 1
        free = [s for s in allowed if s not in used[f]]
        s = int(rng.choice(free)); used[f].append(s)
        chosen.append(int(np.where((fingers == f) & (samples == s))[0][0]))
    return sorted(chosen)


# ------------------------------------------------------------------- Hyser ----

class HyserEnv:
    kind = "hyser"
    metric = "r2"

    def __init__(self, ckpt, device="cuda:0"):
        self.dev = torch.device(device)
        self.model, blob = hysb.load_backbone(ckpt, device=device)
        self.blob = blob
        self.src = hysb.SourceSet()
        self.tables = torch.stack([t.detach() for t in self.model.tables]).to(self.dev)   # [20,K,256]
        self.k = self.model.k
        self.spec = SemgSpec("hyser", 256, self.k, 5, HYSER_LABEL_LAGS)
        self.y_mean, self.y_std = blob["y_mean"], blob["y_std"]
        self.out_scale = table_rms(self.tables)
        self.n_sub = len(self.src.subjects)
        self.sess = []
        for s in range(self.n_sub):
            x = torch.from_numpy(self.src.raw[s]).to(self.dev)
            yz = torch.from_numpy(self.src.yz[s]).to(self.dev)
            self.sess.append(SessStore(x, (lambda i, L, yz=yz: yz[i, :L].double()), [x.shape[1]] * x.shape[0],
                                       self.spec, self.dev))
        self.yz = [torch.from_numpy(y).to(self.dev) for y in self.src.yz]
        self.fingers = [np.asarray(f) for f in self.src.finger]
        self.samples = [np.asarray(s) for s in self.src.sample]

    # -- sampling ---------------------------------------------------------------
    def split_subjects(self, fold: int, n_folds: int = 5, n_val: int = 3, seed: int = 0):
        """(train, val, heldout) subject indices.  fold<0: no held-out subjects."""
        allsub = list(range(self.n_sub))
        held = [s for s in allsub if fold >= 0 and s % n_folds == fold]
        rest = [s for s in allsub if s not in held]
        rng = np.random.default_rng([4242, seed, max(fold, -1) + 1])
        val = sorted(rng.choice(rest, size=n_val, replace=False).tolist())
        return [s for s in rest if s not in val], val, held

    def prior_for(self, source, s, pool, rng=None, stale=None):
        """Training / validation prior of source session s (never uses s's own table in 'mean')."""
        others = [j for j in pool if j != s]
        base = mean_prior(self.tables[others]) if others else torch.zeros_like(self.tables[0])
        if source == "mean":
            return base
        if source == "own":
            rho, sig = stale if stale is not None else (float(rng.uniform(0, 0.7)), float(rng.uniform(0, 0.6)))
            noise = torch.from_numpy(rng.standard_normal(base.shape).astype(np.float32)).to(self.dev) \
                if rng is not None else torch.zeros_like(base)
            return (1 - rho) * self.tables[s] + rho * base + sig * self.out_scale * noise
        if source == "free":
            return None
        raise ValueError(source)

    def sample_episode(self, rng, train_subs, source, b_max=HYSER_B_TRAIN_MAX):
        s = int(rng.choice(train_subs))
        b = int(min(b_max, np.floor(np.exp(rng.uniform(0, np.log(b_max + 1))))))
        cal = draw_hyser_cal(b, rng, self.fingers[s], self.samples[s], allowed=(1, 2))
        q_pool = self.src.val_idx[s]
        q = sorted(rng.choice(q_pool, size=min(4, len(q_pool)), replace=False).tolist())
        prior = self.prior_for(source, s, train_subs, rng)
        return {"s": s, "cal": cal, "q": q, "prior": prior}

    def val_episodes(self, val_subs, train_subs, source, budgets=(1, 2, 5, 10), draws=3):
        eps = []
        for s in val_subs:
            for b in budgets:
                for d in range(draws):
                    rng = np.random.default_rng([555, s, b, d])
                    cal = draw_hyser_cal(b, rng, self.fingers[s], self.samples[s], allowed=(1, 2))
                    rho, sig = STALE_VAL[d % len(STALE_VAL)]
                    g = torch.Generator(device="cpu").manual_seed(int(1000 * s + 10 * b + d))
                    prior = self.prior_for(source, s, train_subs, None, stale=(rho, 0.0))
                    if source == "own":          # fixed-seed staleness noise: deterministic validation
                        noise = torch.randn(prior.shape, generator=g).to(self.dev)
                        prior = prior + sig * self.out_scale * noise
                    eps.append({"s": s, "cal": cal, "q": [int(i) for i in self.src.val_idx[s]],
                                "prior": prior, "b": b})
        return eps

    # -- loss / metric -----------------------------------------------------------
    def run_episodes(self, calib, eps, *, grad: bool, want_pred=False):
        zs, ys, sizes = [], [], []
        for e in eps:
            st = self.sess[e["s"]].stats(e["cal"])
            table, mean, std = forward_table(calib, st, e["prior"], grad=grad)
            xq = self.sess[e["s"]].x[e["q"]]
            z = ((xq - mean) / std) @ table.T
            zs.append(z); ys.append(self.yz[e["s"]][e["q"]]); sizes.append(z.shape[0])
        z = torch.cat(zs); y = torch.cat(ys)
        if grad:
            dec = self.model.base(z)
        else:
            with torch.inference_mode():
                dec = self.model.base(z)
        return dec, y, sizes

    def loss(self, calib, eps):
        dec, y, _ = self.run_episodes(calib, eps, grad=True)
        return (dec - y).square().mean()

    @torch.no_grad()
    def val_score(self, calib, eps):
        scores = []
        for lo in range(0, len(eps), 12):
            chunk = eps[lo:lo + 12]
            dec, y, sizes = self.run_episodes(calib, chunk, grad=False)
            off = 0
            for e, n in zip(chunk, sizes):
                p = dec[off:off + n].cpu().numpy(); off += n
                truth = self.src.y[e["s"]][e["q"]]
                scores.append(hysb.score_prediction(p, truth, self.y_mean, self.y_std)["r2"])
        return float(np.mean(scores))


# ------------------------------------------------------------------ training ----

def fit_normalizers(env, calib, eps_fn, n_eps=200):
    ds, rs = [], []
    for _ in range(n_eps):
        e = eps_fn()
        st = env.sess[e["s"]].stats(e["cal"])
        with torch.no_grad():
            d, r, _, _ = describe(st, e["prior"], calib.spec)
        ds.append(d); rs.append(r)
    calib.set_normalizers(torch.cat(ds), torch.cat(rs))


def train_calibrator(env, variant, source, *, seed, train_pool, val_eps, steps_cap=4000, min_steps=400,
                     val_every=100, patience=6, batch=8, lr=1e-3, weight_decay=1e-2, m=8, log=print):
    """Episodic training on SOURCE sessions only; early stop on source validation.

    ``train_pool``: Hyser subject indices or DB6 session indices usable as training targets.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng([8128, seed])
    mode = "free" if source == "free" else "aligned"
    calib = SemgCalibrator(variant, env.spec, m=m, prior_mode=mode, out_scale=env.out_scale).to(env.dev)
    for p in env.model.parameters():
        assert not p.requires_grad, "trunk must be frozen"
    if env.kind == "hyser":
        draw = lambda: env.sample_episode(rng, train_pool, source)  # noqa: E731
    else:
        draw = lambda: env.sample_episode(rng, train_pool, source)  # noqa: E731
    fit_normalizers(env, calib, draw)
    opt = torch.optim.AdamW(calib.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_step, bad, hist = -np.inf, 0, 0, []
    best_state = copy.deepcopy(calib.state_dict())
    t0 = time.perf_counter()
    base = env.val_score(calib, val_eps)
    hist.append({"step": 0, "val": base})
    if base > best:
        best, best_step = base, 0
    for step in range(1, steps_cap + 1):
        calib.train()
        eps = [draw() for _ in range(batch)]
        loss = env.loss(calib, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calib.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % val_every == 0:
            calib.eval()
            v = env.val_score(calib, val_eps)
            hist.append({"step": step, "loss": float(loss), "val": v, "elapsed": time.perf_counter() - t0})
            log(json.dumps(hist[-1]))
            if v > best:
                best, best_step, bad = v, step, 0
                best_state = copy.deepcopy(calib.state_dict())
            else:
                bad += 1
                if bad >= patience and step >= min_steps:
                    break
    calib.load_state_dict(best_state)
    calib.eval()
    info = {"variant": variant, "source": source, "best_val": best, "val_step0": base, "best_step": best_step,
            "steps_run": step, "steps_cap": steps_cap, "budget_rule_ok": bool(best_step <= 0.8 * steps_cap),
            "seconds": time.perf_counter() - t0, "n_params": int(sum(p.numel() for p in calib.parameters())),
            "history": hist}
    return calib, info
