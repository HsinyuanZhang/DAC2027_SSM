#!/usr/bin/env python
"""Per-B source-only selection of the closed-form ridge (lam, ema) for DB6 (v2, contiguous streaming).

For every source session and budget B, simulate a B-trial calibration (3 draws from the session's training
trials, B-trial z-score with the given std floor), fit the ridge on the calibration segments (EMA from zero
state inside each segment), decode the WHOLE session stream with one causal EMA pass and score the
balanced accuracy on the held-out repetitions.  The grid is open-ended (lam 1e-6..1e4, ema 0..0.9995);
a best point on the grid edge is reported (``edge`` field).  No target data are used.

--order shuffled (default) scores on streams whose trial order is permuted: DB6 records each grasp as a block of 12
consecutive repetitions, so a long EMA (time constant of a few trials) would otherwise gain from "same grasp as the
previous trial" rather than from the signal.  --order original keeps the recorded order (reported for comparison).

usage: select_db6_ridge.py <out.json> --floor F [--tau 160] [--order shuffled|original]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_backbone as hb  # noqa: E402

BUDGETS = [1, 2, 4, 8, 16, 42]
LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
EMAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.9995]
N_DRAWS = 3


def one_session(args):
    key, tau, floor, order = args
    S = hb.load_stream(key, tau)
    if order == "shuffled":
        S = hb.reorder_stream(S, np.random.default_rng([777, key[0], key[1] * 10 + key[2]]).permutation(S.n))
    tr = np.where(~np.isin(S.rep, hb.SOURCE_VAL_REPS))[0]
    va = np.where(np.isin(S.rep, hb.SOURCE_VAL_REPS))[0]
    out = {}
    for b in BUDGETS:
        for d in range(N_DRAWS):
            rng = np.random.default_rng([104729, key[0], key[1] * 10 + key[2], b, d])
            cal = tr[hb.draw_calibration(min(b, len(tr)), rng, S.grasp[tr], S.rep[tr])]
            xz = hb.zscore(S.x, *hb.input_stats(S, cal, floor))
            segs, ys = hb.segment_arrays(S, xz, cal)
            for ema in EMAS:
                feat = hb.causal_ema(xz, ema)
                for lam in LAMS:
                    w, bias = hb.ridge_fit(segs, ys, lam, ema)
                    sc = hb.score_stream(feat @ w + bias, S, va)["bal_acc"]
                    out.setdefault(f"{b}|{lam}|{ema}", []).append(sc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--floor", type=float, required=True)
    ap.add_argument("--tau", type=float, default=160.0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--order", default="shuffled", choices=["shuffled", "original"])
    a = ap.parse_args()
    keys = hb.source_keys()
    with Pool(a.procs) as pool:
        parts = pool.map(one_session, [(k, a.tau, a.floor, a.order) for k in keys])
    agg = {}
    for p in parts:
        for key, v in p.items():
            agg.setdefault(key, []).extend(v)
    res = {}
    for b in BUDGETS:
        mean = {k.split("|", 1)[1]: float(np.mean(v)) for k, v in agg.items() if k.split("|")[0] == str(b)}
        best = max(mean, key=mean.get)
        lam, ema = (float(v) for v in best.split("|"))
        res[b] = {"B": b, "lam": lam, "ema": ema, "best_source_sim_bal_acc": mean[best], "floor": a.floor, "order": a.order,
                  "edge": {"lam": lam in (LAMS[0], LAMS[-1]), "ema": ema == EMAS[-1]}, "scores": mean}
        print(b, lam, ema, round(mean[best], 4), res[b]["edge"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
