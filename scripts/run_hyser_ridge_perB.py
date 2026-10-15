#!/usr/bin/env python
"""Deployable closed-form baseline with per-B hyperparameters chosen on SOURCE only.

Row-norm (z-score from the B calibration trials) + causal EMA + ridge 256 -> 5.
For each B, (lam, ema) is picked by simulating B-trial calibration inside every source
session (balanced draws from its 10 training trials, scored on its 5 held-out trials).
Target data are used only at fit time (B labeled trials).  CPU only; no trunk involved.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import hyser_backbone as hb  # noqa: E402
from ssm_decode.hyser_data import N_CH, load_session  # noqa: E402

spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/run_hyser_eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
LAMS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
EMAS = [0.0, 0.5, 0.8, 0.9, 0.95]
DRAWS = 8
OUT = ROOT / "results/hyser_20261015/eval_closed_form"
OUT.mkdir(parents=True, exist_ok=True)

src = hb.SourceSet()
select = {}
for b in ev.BUDGETS:
    scores = {f"{lam}|{ema}": [] for lam in LAMS for ema in EMAS}
    for s in range(20):
        tr = src.train_idx[s]; va = src.val_idx[s]
        fing = src.finger[s][tr]; smp = src.sample[s][tr]
        for d in range(4):
            rng = np.random.default_rng([31, s, b, d])
            # balanced draw restricted to training trials (samples 1,2 of each finger)
            order = list(rng.permutation(5) + 1); used = {f: [] for f in order}; chosen = []; i = 0
            while len(chosen) < b:
                f = order[i % 5]; i += 1
                free = [x for x in (1, 2) if x not in used[f]]
                x_ = int(rng.choice(free)); used[f].append(x_)
                chosen.append(int(np.where((src.finger[s] == f) & (src.sample[s] == x_))[0][0]))
            raw = src.raw[s]
            mu, sd = hb.input_stats(raw[chosen])
            xc = hb.zscore(raw[chosen], mu, sd); yc = src.yz[s][chosen]
            xv = hb.zscore(raw[va], mu, sd)
            for lam in LAMS:
                for ema in EMAS:
                    w, bias = hb.ridge_fit(xc, yc, lam, ema)
                    p = hb.ridge_predict(xv, w, bias, ema)
                    scores[f"{lam}|{ema}"].append(hb.score_prediction(p, src.y[s][va], src.y_mean, src.y_std)["r2"])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(mean, key=mean.get)
    select[b] = {"best": best, "best_source_r2": mean[best], "scores": mean}
    print("B", b, best, mean[best], flush=True)
(OUT / "ridge_perB_select.json").write_text(json.dumps(select, indent=1))

with (OUT / "rows.jsonl").open("w") as fh:
    for t in range(1, 21):
        tgt = load_session(t, 2); x, y = tgt["x"], tgt["y"]
        yz = ((y - src.y_mean) / src.y_std).astype(np.float32)
        for b in ev.BUDGETS:
            lam, ema = (float(v) for v in select[b]["best"].split("|"))
            for d in range(8):
                rng = np.random.default_rng([7919, t, b, d])
                cal = ev.draw_calibration(b, rng, tgt["finger"], tgt["sample"])
                q = [i for i in range(15) if i not in cal]
                mu, sd = hb.input_stats(x[cal])
                w, bias = hb.ridge_fit(hb.zscore(x[cal], mu, sd), yz[cal], lam, ema)
                p = hb.ridge_predict(hb.zscore(x[q], mu, sd), w, bias, ema)
                sc = hb.score_prediction(p, y[q], src.y_mean, src.y_std)
                fh.write(json.dumps({"target": t, "B": b, "draw": d, "method": "rownorm_ridge_perB", "seed": -1,
                                     "cal_trials": cal, "n_query": len(q), "role": "deployable_closed_form",
                                     "h2_backprop_steps": 0, "h1_trunk_unchanged": True, "h5_pass": True,
                                     "ridge_lam": lam, "ridge_ema": ema,
                                     "param_bytes_fp16": (N_CH * 5 + 2 * N_CH + 5) * 2, **sc}) + "\n")
print("done")
