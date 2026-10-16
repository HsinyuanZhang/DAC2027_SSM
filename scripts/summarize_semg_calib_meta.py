#!/usr/bin/env python
"""Training summary (source validation) and H1-H5 flags of the Q-D rows -> compact csv/json."""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "semg_calib_20261016"
OUT = ROOT / "results" / "analysis" / "semg_calib_20261016"
OUT.mkdir(parents=True, exist_ok=True)

WEAK_VAL = 0.05      # descriptive flag: source-val best below this (identity-free prior-only is about -0.04)
rows = []
for f in sorted(glob.glob(str(RES / "*" / "seed*" / "*" / "info.json")) + glob.glob(str(RES / "fix" / "*" / "seed*" / "*" / "info.json"))):
    p = Path(f)
    i = json.load(open(f))
    rows.append({"run": "fix" if p.parts[-5] == "fix" else "orig", "dataset": p.parts[-4], "seed": i["seed"], "config": p.parts[-2], "variant": i["variant"],
                 "source": i["source"], "fold": i.get("fold", -1), "val_prior_only": i["val_step0"],
                 "val_best": i["best_val"], "best_step": i["best_step"], "steps_run": i["steps_run"],
                 "steps_cap": i["steps_cap"], "budget_rule_ok": i["budget_rule_ok"], "n_params": i["n_params"],
                 "seconds": i["seconds"], "collapsed_step0": i["best_step"] == 0,
                 "weak_val": bool(i["best_step"] > 0 and i["best_val"] < WEAK_VAL)})
tr = pd.DataFrame(rows)
tr.to_csv(OUT / "training_summary.csv", index=False)
print(tr.groupby(["run", "dataset", "source", "variant"])[["val_prior_only", "val_best", "best_step"]].mean().round(3).to_string())

flags = {}
for f in sorted(glob.glob(str(RES / "*" / "seed*" / "rows*.jsonl")) + glob.glob(str(RES / "fix" / "*" / "seed*" / "rows*.jsonl"))):
    p = Path(f)
    pre = "fix/" if p.parts[-4] == "fix" else ""
    n = h1 = h4 = h5 = 0
    h2 = 0
    for ln in open(f):
        r = json.loads(ln)
        if not r["method"].startswith("cal_"):
            continue
        n += 1
        h1 += bool(r["h1_trunk_unchanged"]); h5 += bool(r["h5_pass"])
        h2 += (r["h2_backprop_steps"] == 0 and not r["h2_grad_enabled_inside"])
        h4 += (r["h4_trials_used"] == len(r["cal_trials"]) == r["B"])
    flags[f"{pre}{p.parts[-3]}/{p.parts[-2]}/{p.name}"] = {"rows": n, "h1_pass": h1, "h2_pass": h2, "h4_pass": h4, "h5_pass": h5}
flags["_notes"] = {
    "h1": "hash of the trunk held by the evaluating process; on the DB6 CPU rows this is the hash of the CPU copy "
          "of the weights (the GPU model is not touched), so it cannot detect a change of the GPU trunk.",
    "h2": "backprop_steps is a constant 0 written by the eval script plus the no_grad guard; it is a code-path "
          "property, not a measurement.",
    "h4": "only checks that the number of calibration trials equals B (len(cal_trials) == B); the unit test "
          "(non-calibration trials replaced by noise give bit-identical stats) is the real content check.",
    "h5": "covers the output table only (K x N fp16 <= 32 KB). The descriptor accumulators (Gram etc.) are NOT "
          "included: they exceed 32 KB for Hyser (N=256) and fit for DB6 (N=14) only without any per-trial copy; "
          "see cost_model.json. H5 is therefore NOT satisfied for the whole calibration path on Hyser.",
    "accumulator_simulation": "accumulators were simulated in float64; fp32 per-bin streaming was checked by the "
                              "reviewer (rel err ~2e-7, |dR2| <= 1e-4); fixed-point per-add rounding/saturation "
                              "was NOT simulated (feasibility rule 8 open item)."}
(OUT / "h_flags.json").write_text(json.dumps(flags, indent=1))
print(json.dumps(flags, indent=1))
