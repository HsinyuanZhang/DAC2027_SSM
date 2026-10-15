#!/usr/bin/env python
"""Collect all DB6 backbone-selection runs (source validation only) into backbone_selection.csv."""
import csv
import glob
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results" / "db6_20261015"
out = Path(sys.argv[1])
rows = []
for stage, pat in (("1", "grid/*/summary.json"), ("2", "grid2/*/summary.json"), ("3", "grid3/*/summary.json")):
    for f in sorted(glob.glob(str(root / pat))):
        s = json.load(open(f)); c = s["cfg"]
        rows.append({"stage": stage, "run": Path(f).parent.name, "k": c["k"], "width": c["width"], "layers": c["layers"],
                     "state_size": c["state_size"], "dropout": c["dropout"], "tau_ms": c["tau_ms"],
                     "weight_decay": c["weight_decay"], "channel_dropout": c["channel_dropout"], "seed": c["seed"],
                     "max_steps": c["max_steps"], "best_step": s["best_step"], "steps_run": s["steps_run"],
                     "best_val_bal_acc": round(s["best_val_bal_acc"], 4), "budget_rule_ok": s["budget_rule_ok"],
                     "n_params_trunk": s["n_params_trunk"], "n_params_tables": s["n_params_tables"]})
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["stage"], r["run"], r["best_val_bal_acc"], r["best_step"], r["max_steps"], r["budget_rule_ok"])
