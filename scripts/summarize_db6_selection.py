#!/usr/bin/env python
"""Collect all DB6 v2 backbone runs (contiguous source validation only) into backbone_selection.csv.

stages: 0 = trained on original-order streams (block-structure ablation), 1 = architecture / tau / window search
(trial-order-shuffled training), 2 = std-floor candidates, 3 = final seeds (final/seed0 is a copy of the stage-1
winner).  best_val_shuf = order-shuffled validation (selection metric), best_val_orig = recorded order.

usage: summarize_db6_selection.py <out.csv> [<results root, default results/db6_v2_20261016>]
"""
import csv
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results" / "db6_v2_20261016"
out = Path(sys.argv[1])
rows = []
for stage, pat in (("0", "blockorder/*/summary.json"), ("1", "select/*/summary.json"),
                   ("2", "floor/*/summary.json"), ("3", "final/*/summary.json")):
    for f in sorted(glob.glob(str(root / pat))):
        s = json.load(open(f)); c = s["cfg"]
        rows.append({"stage": stage, "run": Path(f).parent.name, "k": c["k"], "width": c["width"], "layers": c["layers"],
                     "state_size": c["state_size"], "dropout": c["dropout"], "tau_ms": c["tau_ms"],
                     "win_min": c.get("win_min"), "win_max": c.get("win_max"),
                     "shuffle_trials": c.get("shuffle_trials", False), "std_floor_train": c.get("std_floor_train", 1e-3),
                     "seed": c["seed"], "max_steps": c["max_steps"], "best_step": s["best_step"], "steps_run": s["steps_run"],
                     "best_val_orig": round(s["best_val_bal_acc"], 4),
                     "best_val_shuf": round(s["best_val_shuf_bal_acc"], 4) if "best_val_shuf_bal_acc" in s else "",
                     "budget_rule_ok": s["budget_rule_ok"], "n_params_trunk": s["n_params_trunk"],
                     "n_params_tables": s["n_params_tables"]})
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["stage"], r["run"], r["best_val_orig"], r["best_val_shuf"], r["best_step"], r["max_steps"], r["budget_rule_ok"])
