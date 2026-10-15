#!/usr/bin/env python
"""Per-add fixed-point accumulation on continuous (non-integer) HD-sEMG features.

CPU only, no decoding: measures the relative error of the per-add (round every
increment to the register step, saturating add) Gram / sx / syx(lag 0) sums against
float64, on the cached Hyser 1-DoF envelope features (N = 256, 50 Hz, 1250 bins per trial).
Two range modes: "source" = steps fitted on SOURCE subjects (power of two, max |sum| over all
15 trials of a session = budget B_max = 15) and applied to TARGET subjects at B = 1, 8, 15
trials (what a source-only design would do); "own" = steps from the target session's own
15-trial sums (ideal auto-gain; isolates the pure rounding error).  Counters (n) are exact.
The cached features are continuous (leaky-integrator envelope in mV, NOT integers), so
products are not exactly representable: every add rounds.  Also reports the register width
that would make an int8-coded input (x -> round(x / max * 127)) accumulate exactly.  Output: results/population_calib_p2_20261015/fix/hyser_peradd.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/population_calib_p2_20261015/fix"

from ssm_decode.calib_quant import _sat_cumsum, fixed_step, qmax            # noqa: E402
from ssm_decode.hyser_data import load_session                              # noqa: E402

SRC = (4, 5, 6)
TGT = (1, 2, 3)
WIDTHS = (16, 20, 24, 28, 32)
BUDGETS = (1, 8, 15)
CHUNK = 125


def trial_sums(x, y):
    """Exact float64 sums of one trial: sx [N], gram [N, N], syx0 [D, N]."""
    return x.sum(0), x.T @ x, y.T @ x


def peradd_trial(state, x, y, steps, q):
    """Advance the three registers (integer multiples of their step) through one trial."""
    ev = 0
    for t0 in range(0, x.shape[0], CHUNK):
        xc, yc = x[t0:t0 + CHUNK], y[t0:t0 + CHUNK]
        incs = {"sx": xc, "gram": torch.einsum("tn,tm->tnm", xc, xc),
                "syx": torch.einsum("td,tn->tdn", yc, xc)}
        for k, inc in incs.items():
            state[k], e = _sat_cumsum(state[k], torch.round(inc / steps[k]), q)
            ev += e
    return ev


def corr_of(sx, gram, n):
    mean = sx / n
    cov = gram / n - torch.outer(mean, mean)
    std = cov.diagonal().clamp_min(1e-30).sqrt()
    return (cov / torch.outer(std, std)).clamp(-1, 1), std


def session_arrays(subject, session=1):
    s = load_session(subject, session)
    x = torch.from_numpy(s["x"].astype(np.float64))          # [15, 1250, 256]
    y = torch.from_numpy(s["y"].astype(np.float64))
    y = (y - y.mean((0, 1))) / y.std((0, 1)).clamp_min(1e-12)
    return x, y


def session_maxabs(sub):
    x, y = session_arrays(sub)
    tot = [0, 0, 0]
    for j in range(x.shape[0]):
        s = trial_sums(x[j], y[j])
        tot = [s[i] + tot[i] for i in range(3)]
    return {k: float(v.abs().max()) for k, v in zip(("sx", "gram", "syx"), tot)}


def main():
    src = [session_maxabs(sub) for sub in SRC]
    maxabs_src = {k: max(m[k] for m in src) for k in src[0]}
    own = {sub: session_maxabs(sub) for sub in TGT}
    rows = []
    for mode, bits in [(m, b) for m in ("source", "own") for b in WIDTHS]:
        q = qmax(bits)
        for sub in TGT:
            maxabs = maxabs_src if mode == "source" else own[sub]
            steps = {k: fixed_step(v, bits) for k, v in maxabs.items()}
            x, y = session_arrays(sub)
            for B in BUDGETS:
                idx = list(range(B)) if B == 15 else list(np.random.default_rng(sub * 100 + B).choice(15, B, replace=False))
                state = {"sx": torch.zeros(256, dtype=torch.float64),
                         "gram": torch.zeros(256, 256, dtype=torch.float64),
                         "syx": torch.zeros(5, 256, dtype=torch.float64)}
                ev = 0
                ex = [torch.zeros(256, dtype=torch.float64), torch.zeros(256, 256, dtype=torch.float64),
                      torch.zeros(5, 256, dtype=torch.float64)]
                for j in idx:
                    ev += peradd_trial(state, x[j], y[j], steps, q)
                    s = trial_sums(x[j], y[j])
                    ex = [ex[i] + s[i] for i in range(3)]
                n = float(B * x.shape[1])
                sx, gram, syx = (state[k] * steps[k] for k in ("sx", "gram", "syx"))
                std_q = (gram.diagonal() / n - (sx / n) ** 2).clamp_min(0).sqrt()
                c_q, _ = corr_of(sx, gram, n)
                c_e, std_e = corr_of(ex[0], ex[1], n)
                rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))
                var_bad = int((c_q.diagonal() <= 0).sum() + (std_q <= 1e-6).sum())
                rows.append({"mode": mode, "bits": bits, "subject": sub, "var_nonpositive": var_bad, "B": B, "sat_events": ev,
                             "relerr_gram": rel(gram, ex[1]), "relerr_sx": rel(sx, ex[0]),
                             "relerr_syx": rel(syx, ex[2]), "relerr_corr": rel(c_q, c_e),
                             "max_abs_corr_err": float((c_q - c_e).abs().max()),
                             "relerr_std": rel(std_q, std_e),
                             "step_gram": steps["gram"], "step_sx": steps["sx"], "step_syx": steps["syx"],
                             "gram_max_over_step": maxabs["gram"] / steps["gram"]})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hyser_peradd.csv", index=False)
    # register width for exact accumulation of int8-coded inputs (global scale per session, 15 trials)
    w = []
    for sub in TGT:
        x, y = session_arrays(sub)
        code = torch.round(x / x.max() * 127)
        g = max(float((code[j].T @ code[j]).abs().max()) for j in range(15))
        gram_tot = float(sum(code[j].T @ code[j] for j in range(15)).abs().max())
        w.append({"subject": sub, "gram_max_codes": gram_tot, "signed_bits_exact": int(np.ceil(np.log2(gram_tot + 1))) + 1})
    pd.DataFrame(w).to_csv(OUT / "hyser_int8_exact_width.csv", index=False)
    print(pd.DataFrame(w).to_string())
    g = df.groupby(["mode", "bits", "B"])[["relerr_gram", "relerr_sx", "relerr_syx", "relerr_corr",
                                   "max_abs_corr_err", "relerr_std", "sat_events", "var_nonpositive"]].mean()
    print(g.to_string())
    g.to_csv(OUT / "hyser_peradd_summary.csv")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
