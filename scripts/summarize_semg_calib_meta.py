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

rows = []
for f in sorted(glob.glob(str(RES / "*" / "seed*" / "*" / "info.json"))):
    p = Path(f)
    i = json.load(open(f))
    rows.append({"dataset": p.parts[-4], "seed": i["seed"], "config": p.parts[-2], "variant": i["variant"],
                 "source": i["source"], "fold": i.get("fold", -1), "val_prior_only": i["val_step0"],
                 "val_best": i["best_val"], "best_step": i["best_step"], "steps_run": i["steps_run"],
                 "steps_cap": i["steps_cap"], "budget_rule_ok": i["budget_rule_ok"], "n_params": i["n_params"],
                 "seconds": i["seconds"]})
tr = pd.DataFrame(rows)
tr.to_csv(OUT / "training_summary.csv", index=False)
print(tr.groupby(["dataset", "source", "variant"])[["val_prior_only", "val_best", "best_step"]].mean().round(3).to_string())

flags = {}
for f in sorted(glob.glob(str(RES / "*" / "seed*" / "rows*.jsonl"))):
    p = Path(f)
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
    flags[f"{p.parts[-3]}/{p.parts[-2]}/{p.name}"] = {"rows": n, "h1_pass": h1, "h2_pass": h2, "h4_pass": h4, "h5_pass": h5}
(OUT / "h_flags.json").write_text(json.dumps(flags, indent=1))
print(json.dumps(flags, indent=1))
