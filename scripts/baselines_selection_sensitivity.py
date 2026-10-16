#!/usr/bin/env python
"""Sensitivity of the published-baseline rows to the second-review d-handling fix in the source-only selection.

Re-runs the target rows whose source-selected configuration changes when the selection is repeated with the fixed
d handling (results/baselines_20261016_selsens/select/, produced by ``BL_OUT=... run_baselines_readin.py select``).
Reporting only: the tables use the original source-selected configurations.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_baselines_readin as rb                                   # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                        # noqa: E402
from scripts.run_population_calib import World, draw_trials                 # noqa: E402

OLD = ROOT / "results/baselines_20261016"
NEW = ROOT / "results/baselines_20261016_selsens"
CASES = {"days": ["cca_sqrt_a"], "month": ["cca_sqrt_a_ridge", "stab_pool_a"], "year": ["cca_sqrt_a_ridge", "stab_pool_a"]}


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    out = NEW / "sensitivity_eval.jsonl"
    for st, names in CASES.items():
        sel_stratum = "days" if st == "days" else "month"            # month and year share trunk, sources and selection
        rb.OUT = OLD
        old_sel = rb.selected_cfgs(st)
        rb.OUT = NEW
        new_sel = rb.selected_cfgs(sel_stratum)
        world = World(st, "O", dev, load_sessions=False)
        refdata = rb.RefData(world)
        for target in D1_POOLS[st]["targets"]:
            sd, _ = world.target(target)
            for name in names:
                kind, inp, tr, mode, part = rb.EVAL_METHODS[name]
                for tag, sel in (("orig_sel", old_sel), ("fixed_sel", new_sel)):
                    cfg = rb.method_cfg(name, sel)
                    for B in (8, 33):
                        vals = []
                        for draw in range(rb.N_DRAWS):
                            idx = draw_trials(target, B, draw, sd.bank.n_trials)
                            t, c, mean, std, info = rb.run_method(kind, cfg, refdata, sd, idx)
                            r2, _ = rb.score(world, sd, t, c, mean, std, idx, cfg.get("transform", "raw"), pool=inp == "pooled")
                            vals.append(r2)
                        row = {"stratum": st, "target": target, "method": name, "selection": tag, "cfg": cfg, "B": B,
                               "r2_mean": float(np.mean(vals)), "r2_draws": vals}
                        with out.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(st, name, tag, B, round(float(np.mean(vals)), 4), flush=True)


if __name__ == "__main__":
    main()
