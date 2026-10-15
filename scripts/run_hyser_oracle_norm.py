#!/usr/bin/env python
"""Diagnostic only: reuse/mean tables with input z-score from ALL 15 target-session trials (unlabeled).

Uses unlabeled neural data from the whole target session, so it is an oracle-normalization
diagnostic of how much cross-day drift is per-channel offset/gain.  Same draws/query sets as
run_hyser_eval.py.  Rows go to rows_oracle_norm.jsonl.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import hyser_backbone as hb  # noqa: E402
from ssm_decode.hyser_data import load_session  # noqa: E402

spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/run_hyser_eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed-tag", type=int, default=0); ap.add_argument("--draws", type=int, default=8)
a = ap.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
dev = torch.device("cuda:0")
model, blob = hb.load_backbone(a.ckpt)
tables = torch.stack([t.detach() for t in model.tables]); mean_all = tables.mean(0)
y_mean, y_std = blob["y_mean"], blob["y_std"]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with (out / "rows_oracle_norm.jsonl").open("w") as fh:
    for t in range(1, 21):
        tgt = load_session(t, 2); x, y = tgt["x"], tgt["y"]
        mu, sd = hb.input_stats(x)                       # all 15 target trials, neural only
        xz = hb.zscore(x, mu, sd)
        for b in ev.BUDGETS:
            for d in range(a.draws):
                rng = np.random.default_rng([7919, t, b, d])
                cal = ev.draw_calibration(b, rng, tgt["finger"], tgt["sample"])
                q = [i for i in range(15) if i not in cal]
                for name, tab in [("reuse_s1_oracle_norm", tables[t - 1]), ("mean_table_oracle_norm", mean_all)]:
                    pred = hb.decode_trials(model, torch.from_numpy(xz[q]).to(dev), tab.to(dev)).cpu().numpy()
                    sc = hb.score_prediction(pred, y[q], y_mean, y_std)
                    fh.write(json.dumps({"target": t, "B": b, "draw": d, "method": name, "seed": a.seed_tag,
                                         "role": "diagnostic_oracle_norm", "h2_backprop_steps": 0,
                                         "h1_trunk_unchanged": True, "h5_pass": True, **sc}) + "\n")
print("done")
