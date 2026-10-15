#!/usr/bin/env python
"""P3 review fix (item 1): like-for-like re-selection of the NON-DEPLOYABLE reference P0+ui_reg.

The P2 reference rule (lambda, lr, steps) was selected on the six trunk-source sessions starting
from the LOO electrode prior and then reused from the P0 start.  The P3 candidates (R1/R2/R3) for
month/year use rules selected on out-of-sample (OOS) source sessions from the P0 start.  Here the
reference rule is selected on the SAME simulation episodes as the candidates: same sessions, same
seeds/draws (``run_pc3_refine.draw_sim``), same P0 calibrators, same scoring window, start AND
shrinkage centre = the P0 table.  Selection is source-only (the target is never touched).

  select  --stratum S --set is|oos [--sessions a,b]   append episode rows (resumable, sharded)
  write   --stratum S --tag is0|oos|oosc               aggregate rows -> selection json (+grid csv)
  eval    --stratum S --tag T [--rule-stratum S2]       target rows p0_ui_reg_<T> (8 draws x 2 seeds)

tags: ``is0`` = IS sessions (P0 start); ``oos`` = the four OOS sessions; ``oosc`` = causal OOS
(only sessions dated before the earliest target of the stratum; month: 0307 only).  month and year
share trunk, sources, P0 calibrators and OOS sessions (bitwise-identical P0 weights, checked in the
P3 run), so the year evaluation reuses the month simulation (``--rule-stratum month``).

Step grid up to 1000 (review item 8), pruned for run time: steps <= 300 for B <= 4 and for lr = 1e-2
(P2 never selected more than 100 steps there / the larger lr diverges), up to 1000 otherwise.
Backprop through the frozen trunk: reference only, never a candidate.
"""
from __future__ import annotations

import argparse
import glob
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

from scripts import run_pc3_refine as pc                                  # noqa: E402
from scripts import run_population_calib as rpc                           # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402
from ssm_decode.ui_reg import ui_reg_trajectory                           # noqa: E402

LAMS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
LRS = (1e-3, 3e-3, 1e-2)
STEPS = (10, 30, 100, 300, 600, 1000)


def max_steps(B, lr):
    return 300 if (B <= 4 or lr >= 1e-2) else 1000


def uidir():
    d = pc.OUT / "uiref"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_traj(world, sd, ep, lam, lr, snapshots):
    """ui_reg trajectory from the episode's P0 table (start = shrinkage centre); {step: table}."""
    init = ep.table0.cpu().numpy()
    return ui_reg_trajectory(world.adapter.model, ep.xz_all.cpu().numpy(), sd.y_norm.cpu().numpy(),
                             [sd.trials[i] for i in ep.idx], init, init, lam=lam, lr=lr,
                             snapshots=snapshots, device=world.device, seed=0,
                             valid_mask=sd.valid.cpu().numpy())


def cmd_select(args):
    world, names = pc.sim_world(args.stratum, args.set)
    if args.sessions:
        keep = args.sessions.split(",")
        names = [n for n in names if n[-8:] in keep]
    p0 = pc.load_p0(world, args.stratum)
    shard = "-".join(n[-8:] for n in names)
    path = uidir() / f"select_{args.stratum}_{args.set}_{shard}.jsonl"
    seen = pc.seen_keys(path) if path.exists() else set()
    t0 = time.perf_counter()
    for name in names:
        sd, prior = world.sessions[name], world.priors[name]
        for seed in pc.SEEDS:
            for B in rpc.B_GRID:
                for draw in range(pc.SEL_DRAWS):
                    if (name, seed, B, draw) in seen:
                        continue
                    idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                    ep = pc.p0_episode(world, sd, prior, p0[seed], idx, pc.SIM_MAX_BINS)
                    rows = [{"stratum": args.stratum, "set": args.set, "session": name, "seed": seed,
                             "B": B, "draw": draw, "lam": None, "lr": None, "steps": 0,
                             "r2": ep.r2(ep.table0)}]
                    for lam, lr in itertools.product(LAMS, LRS):
                        snaps = tuple(s for s in STEPS if s <= max_steps(B, lr))
                        for step, tab in run_traj(world, sd, ep, lam, lr, snaps).items():
                            rows.append({"stratum": args.stratum, "set": args.set, "session": name,
                                         "seed": seed, "B": B, "draw": draw, "lam": lam, "lr": lr,
                                         "steps": step,
                                         "r2": ep.r2(torch.from_numpy(tab).float().to(world.device))})
                    with path.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[uiref select {args.stratum}/{args.set}] {name[-8:]} s{seed} B={B} d={draw} "
                          f"p0={rows[0]['r2']:.3f} t={time.perf_counter() - t0:.0f}s", flush=True)


def tag_sessions(stratum, tag):
    """Session names (suffix dates) a tag may use, and the shard set it reads."""
    if tag == "is0":
        return "is", None
    first = min(t[-8:] for t in D1_POOLS[stratum]["targets"])
    if tag == "oos":
        return "oos", None
    if tag == "oosc":
        return "oos", first              # causal: strictly before the earliest target
    raise ValueError(tag)


def cmd_write(args):
    sim_set, before = tag_sessions(args.stratum, args.tag)
    rows = []
    for f in sorted(glob.glob(str(uidir() / f"select_{args.stratum}_{sim_set}_*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    p0 = df[df.lam.isna()]
    df = df[df.lam.notna()]
    if before is not None:
        df = df[df.session.str[-8:] < before]
        p0 = p0[p0.session.str[-8:] < before]
    df = df.assign(r2=df.r2.fillna(-1.0))
    g = df.groupby(["B", "lam", "lr", "steps"]).r2.agg(["mean", "count"]).reset_index()
    sel = {}
    for B, sub in g.groupby("B"):
        full = sub[sub["count"] == sub["count"].max()]       # only fully evaluated episodes
        best = full.loc[full["mean"].idxmax()]
        sel[str(B)] = {"lam": float(best.lam), "lr": float(best.lr), "steps": int(best.steps),
                       "source_r2": float(best["mean"]), "n_episodes": int(best["count"]),
                       "p0_source_r2": float(p0[p0.B == B].r2.mean()),
                       "sessions": sorted(set(df.session.str[-8:]))}
    (uidir() / f"selection_{args.stratum}_{args.tag}.json").write_text(json.dumps(sel, indent=1))
    g.to_csv(uidir() / f"select_grid_{args.stratum}_{args.tag}.csv", index=False)
    print(json.dumps({B: {k: v for k, v in s.items() if k != "sessions"} for B, s in sel.items()}, indent=1))


def cmd_eval(args):
    rule_stratum = args.rule_stratum or args.stratum
    if args.rule_file:
        sel = json.loads(Path(args.rule_file).read_text())
    else:
        sel = json.loads((uidir() / f"selection_{rule_stratum}_{args.tag}.json").read_text())
    world = pc.load_world(args.stratum, False)
    p0 = pc.load_p0(world, args.stratum)
    path = uidir() / f"eval_{args.stratum}_{args.tag}.jsonl"
    seen = pc.jsonl_keyset(path, ("seed", "target", "B", "draw"))
    t0 = time.perf_counter()
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for B in rpc.B_GRID:
            s = sel[str(B)]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in pc.SEEDS:
                    if (seed, target, B, draw) in seen:
                        continue
                    ep = pc.p0_episode(world, sd, prior, p0[seed], idx)
                    tab = run_traj(world, sd, ep, s["lam"], s["lr"], (s["steps"],))[s["steps"]]
                    row = {"stratum": args.stratum, "variant": f"p0_ui_reg_{args.tag}", "seed": seed,
                           "target": target, "B": B, "draw": draw,
                           "r2": ep.r2(torch.from_numpy(tab).float().to(world.device)),
                           "lam": s["lam"], "lr": s["lr"], "steps": s["steps"],
                           "rule_from": rule_stratum,
                           "note": "gradient through trunk; non-deployable reference; rule selected on source"}
                    with path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[uiref eval {args.stratum}/{args.tag}] {target[-4:]} B={B} d={draw} s{seed} "
                          f"r2={row['r2']:.3f} t={time.perf_counter() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    s.add_argument("--set", required=True, choices=["is", "oos"])
    s.add_argument("--sessions", default=None, help="comma list of YYYYMMDD suffixes (shard)")
    w = sub.add_parser("write")
    w.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    w.add_argument("--tag", required=True, choices=["is0", "oos", "oosc"])
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    e.add_argument("--tag", required=True)
    e.add_argument("--rule-stratum", default=None)
    e.add_argument("--rule-file", default=None, help="explicit selection json (e.g. the P2 rule)")
    e.add_argument("--draws", type=int, default=rpc.N_DRAWS)
    args = p.parse_args()
    {"select": cmd_select, "write": cmd_write, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
