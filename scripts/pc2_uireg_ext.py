#!/usr/bin/env python
"""A'2 follow-up (review item 6): extend the ui_reg step grid at B = 33 to {600, 1000}.

The original selection (run_pc2_uireg.py) hit the upper step edge (300) at B = 33 for days
and month and the source R2 was still rising.  This script re-selects (lam, lr, steps) at
B = 33 ONLY, on SOURCE sessions only, with steps (10, 30, 100, 300, 600, 1000) and a reduced
(lam in {0,1e-4,1e-3,1e-2} x lr in {3e-3,1e-2}, 1 draw per source session) grid, then re-evaluates ui_reg (start = electrode-aligned prior) on the targets at
B = 33.  Outputs: results/population_calib_p2_20261015/fix/uireg/.  Non-deployable reference.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P2 = ROOT / "results/population_calib_p2_20261015"
os.environ["PC_OUT"] = str(P2)
FIX = P2 / "fix" / "uireg"

from scripts import run_population_calib as rpc                           # noqa: E402
from scripts import run_pc2_uireg as ru                                   # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402

STEPS = (10, 30, 100, 300, 600, 1000)
LAMS = (0.0, 1e-4, 1e-3, 1e-2)
LRS = (3e-3, 1e-2)                 # lr=1e-3 was never selected at B=33 (days/month: 3e-3)
SEL_DRAWS = 1                      # reduced from 2 for run time (GPU shared); 7 source episodes
B = 33


def cmd_select(args):
    device = torch.device("cuda:0")
    world = rpc.World(args.stratum, "O", device, load_sessions=True)
    FIX.mkdir(parents=True, exist_ok=True)
    path = FIX / f"select_ext_{args.stratum}.jsonl"
    seen = ru.jsonl_keys(path, ("session", "draw", "lam", "lr"))
    combos_all = list(itertools.product(LAMS, LRS))
    t0 = time.perf_counter()
    for name in world.ids:
        sd, prior = world.sessions[name], world.priors[name]
        for draw in range(SEL_DRAWS):
            combos = [c for c in combos_all if (name, draw, c[0], c[1]) not in seen]
            if not combos:
                continue
            idx = rpc.draw_trials(f"uireg-sel-{name}", B, draw, sd.bank.n_trials)
            with path.open("a") as f:
                for lam, lr, step, r2 in ru.trajectories(world, sd, idx, prior.cpu().numpy(), combos,
                                                         snapshots=STEPS):
                    f.write(json.dumps({"stratum": args.stratum, "session": name, "B": B, "draw": draw,
                                        "lam": lam, "lr": lr, "steps": step, "r2": r2}) + "\n")
            print(f"[select-ext {args.stratum}] {name[-8:]} draw={draw} t={time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines()])
    df = df[df.lr.isin(LRS) & df.lam.isin(LAMS) & df.steps.isin(STEPS) & (df.draw < SEL_DRAWS)]
    g = df.groupby(["lam", "lr", "steps"]).r2.agg(["mean", "count"]).reset_index()
    g = g[g["count"] == g["count"].max()]                 # only fully evaluated combos
    best = g.loc[g["mean"].idxmax()]
    sel = {"lam": float(best.lam), "lr": float(best.lr), "steps": int(best.steps),
           "source_r2": float(best["mean"]), "n_episodes": int(best["count"])}
    (FIX / f"selection_ext_{args.stratum}.json").write_text(json.dumps(sel, indent=1))
    g.to_csv(FIX / f"select_grid_ext_{args.stratum}.csv", index=False)
    print(sel, flush=True)


def cmd_eval(args):
    device = torch.device("cuda:0")
    sel = json.loads((FIX / f"selection_ext_{args.sel_from or args.stratum}.json").read_text())
    world = rpc.World(args.stratum, "O", device, load_sessions=False)
    path = FIX / f"eval_ext_{args.stratum}.jsonl"
    seen = ru.jsonl_keys(path, ("target", "draw"))
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for draw in range(args.draws):
            if (target, draw) in seen:
                continue
            idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
            r2 = [r for _, _, _, r in ru.trajectories(world, sd, idx, prior.cpu().numpy(),
                                                     [(sel["lam"], sel["lr"])], snapshots=(sel["steps"],))][0]
            row = {"stratum": args.stratum, "pool": "O", "variant": "ui_reg_ext", "seed": 0, "target": target,
                   "B": B, "draw": draw, "r2": r2, **{k: sel[k] for k in ("lam", "lr", "steps")},
                   "note": "gradient through trunk; non-deployable reference; step grid extended to 1000"}
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("target", "B", "draw", "r2", "steps")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    e.add_argument("--draws", type=int, default=8)
    e.add_argument("--sel-from", default=None)
    args = p.parse_args()
    {"select": cmd_select, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
