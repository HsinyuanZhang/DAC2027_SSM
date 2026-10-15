#!/usr/bin/env python
"""A'2: fair regularised gradient reference ``ui_reg`` and ``p0_ui_reg`` (GPU0).

  select   choose (lam, steps, lr) per B by simulating B-trial calibration on
           SOURCE sessions only (leave-one-out electrode prior as start/center,
           query = the other trials of the same source session)
  eval     target rows: ui_reg (start = electrode-aligned prior) and
           p0_ui_reg (start = P0 table), both with the source-selected rule

Both are labeled non-deployable references (gradient through the frozen
trunk).  Target data is touched only through the B chosen labeled trials.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("PC_OUT", str(ROOT / "results/population_calib_p2_20261015")))
os.environ["PC_OUT"] = str(OUT)

from scripts import run_population_calib as rpc                           # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402
from ssm_decode.population_calib import PopStats, norm_stats             # noqa: E402
from ssm_decode.ui_reg import ui_reg_trajectory                           # noqa: E402

LAMS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LRS = (1e-3, 3e-3, 1e-2)
STEPS = (10, 30, 100, 300)
SEL_DRAWS = 2


def prep(world, sd, idx):
    """B-trial normalisation inputs for the gradient fit (same as the P rows)."""
    stats = PopStats.from_trials(sd.bank, idx)
    mean, std = norm_stats(stats)
    xz = ((sd.x32.double() - mean) / std).cpu().numpy()
    rec = sd.record
    y_norm = (rec.behavior.astype(np.float64) - world.label_stats[0]) / world.label_stats[1]
    valid = rec.eval_mask & np.isfinite(rec.behavior).all(1)
    windows = [sd.trials[i] for i in idx]
    return mean, std, xz, y_norm, valid, windows


def trajectories(world, sd, idx, init, combos, snapshots=STEPS):
    """Yield (lam, lr, step, r2) for each combo; init is also the penalty center."""
    mean, std, xz, y_norm, valid, windows = prep(world, sd, idx)
    q = sd.query_mask(idx)
    for lam, lr in combos:
        traj = ui_reg_trajectory(world.adapter.model, xz, y_norm, windows, init, init,
                                 lam=lam, lr=lr, snapshots=snapshots, device=world.device,
                                 seed=0, valid_mask=valid)
        for step, tab in traj.items():
            pred = rpc.decode_full(world, sd, torch.from_numpy(tab).float().to(world.device),
                                   mean, std)
            r2, _ = rpc.r2_physical(world, sd, pred, q)
            yield lam, lr, step, r2


def jsonl_keys(path, fields):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add(tuple(r[f] for f in fields))
    return keys


def cmd_select(args):
    device = torch.device("cuda:0")
    world = rpc.World(args.stratum, "O", device, load_sessions=True)
    path = OUT / "uireg" / f"select_{args.stratum}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = jsonl_keys(path, ("session", "B", "draw", "lam", "lr"))
    combos_all = list(itertools.product(LAMS, LRS))
    t0 = time.perf_counter()
    for name in world.ids:
        sd, prior = world.sessions[name], world.priors[name]
        for B in rpc.B_GRID:
            for draw in range(SEL_DRAWS):
                combos = [c for c in combos_all if (name, B, draw, c[0], c[1]) not in seen]
                if not combos:
                    continue
                idx = rpc.draw_trials(f"uireg-sel-{name}", B, draw, sd.bank.n_trials)
                with path.open("a") as f:
                    for lam, lr, step, r2 in trajectories(world, sd, idx, prior.cpu().numpy(), combos):
                        f.write(json.dumps({"stratum": args.stratum, "session": name, "B": B,
                                            "draw": draw, "lam": lam, "lr": lr, "steps": step,
                                            "r2": r2}) + "\n")
                print(f"[select {args.stratum}] {name[-8:]} B={B} draw={draw} "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)
    write_selection(args.stratum)


def write_selection(stratum):
    import pandas as pd
    path = OUT / "uireg" / f"select_{stratum}.jsonl"
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines()])
    g = df.groupby(["B", "lam", "lr", "steps"]).r2.agg(["mean", "count"]).reset_index()
    sel = {}
    for B, sub in g.groupby("B"):
        best = sub.loc[sub["mean"].idxmax()]
        sel[str(B)] = {"lam": float(best.lam), "lr": float(best.lr), "steps": int(best.steps),
                       "source_r2": float(best["mean"]), "n_episodes": int(best["count"])}
    (OUT / "uireg" / f"selection_{stratum}.json").write_text(json.dumps(sel, indent=1))
    g.to_csv(OUT / "uireg" / f"select_grid_{stratum}.csv", index=False)
    return sel


def cmd_eval(args):
    device = torch.device("cuda:0")
    sel_stratum = args.sel_from or args.stratum
    sel = json.loads((OUT / "uireg" / f"selection_{sel_stratum}.json").read_text())
    world = rpc.World(args.stratum, "O", device, load_sessions=False)
    path = OUT / "uireg" / f"eval_{args.stratum}.jsonl"
    seen = jsonl_keys(path, ("variant", "seed", "target", "B", "draw"))
    p0 = {}
    for seed in (0, 1):
        run_dir = rpc.P1_OUT / "runs" / f"{args.stratum}_O" / "p0" / f"seed{seed}"
        p0[seed] = rpc.load_run(run_dir, world)[0]
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for B in rpc.B_GRID:
            s = sel[str(B)]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                starts = []
                if ("ui_reg", 0, target, B, draw) not in seen:
                    starts.append(("ui_reg", 0, prior.cpu().numpy()))
                for seed in (0, 1):
                    if ("p0_ui_reg", seed, target, B, draw) not in seen:
                        res = p0[seed].calibrate(PopStats.from_trials(sd.bank, idx), prior)
                        starts.append(("p0_ui_reg", seed, res.table.cpu().numpy().astype(np.float64)))
                for variant, seed, init in starts:
                    r2 = [r for _, _, st, r in trajectories(world, sd, idx, init, [(s["lam"], s["lr"])],
                                                           snapshots=(s["steps"],))][0]
                    row = {"stratum": args.stratum, "pool": "O", "variant": variant, "seed": seed,
                           "target": target, "B": B, "draw": draw, "r2": r2, "lam": s["lam"],
                           "lr": s["lr"], "steps": s["steps"],
                           "note": "gradient through trunk; non-deployable reference; rule selected on source"}
                    with path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row[k] for k in ("variant", "seed", "target", "B", "draw", "r2")}),
                          flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    e.add_argument("--draws", type=int, default=8)
    e.add_argument("--sel-from", default=None, help="reuse the selection of another stratum")
    args = p.parse_args()
    {"select": cmd_select, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
