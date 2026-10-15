#!/usr/bin/env python
"""Collect backbone-selection runs (source validation only) into one csv."""
import csv
import glob
import json
import os

rows = []
for stage in ("grid", "grid2", "final"):
    for f in sorted(glob.glob(f"results/hyser_20261015/{stage}/*/summary.json")):
        s = json.load(open(f)); c = s["cfg"]
        rows.append({"stage": stage, "name": os.path.basename(os.path.dirname(f)), "k": c["k"], "width": c["width"],
                     "layers": c["layers"], "state": c["state_size"], "dropout": c["dropout"], "lr": c["lr"],
                     "wd": c["weight_decay"], "ch_dropout": c["channel_dropout"], "seed": c["seed"],
                     "max_steps": c["max_steps"], "best_step": s["best_step"], "steps_run": s["steps_run"],
                     "budget_rule_ok": s["budget_rule_ok"], "source_val_r2": round(s["best_val_r2"], 4),
                     "trunk_params": s["n_params_trunk"], "table_params": s["n_params_tables"],
                     "seconds": round(s["seconds"])})
with open("results/analysis/hyser_20261015/backbone_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "runs")
