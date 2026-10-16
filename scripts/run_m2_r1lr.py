#!/usr/bin/env python
"""CPU-only budget-compliant R1-stream variants for FALCON M2 (review item 1).

Full R1-stream accumulates a 257 x 257 fp32 Gram (264 KB), over the SRAM budget.  Compact variants
project the 257-d augmented final feature on a FIXED rank-r source basis (top-r eigenvectors of the
source feature second moment, SOURCE sessions only) and keep an r x r Gram and a D x r cross term:
``p0_r1s_lr16`` / ``p0_r1s_lr32``.  Rule (lambda, burn) per (variant, B) is selected on the SOURCE
simulation exactly as for the full variant (``off`` is a candidate).  Targets never enter selection.

Subcommands (CPU only; uses the plain-export CPU Mamba-3 executor, |dR2| <= 7e-5 vs the GPU rows):
  select    source selection for one (seed, pool, cseed) -> select_r1lr.jsonl / selection_r1lr.json
  evaluate  target rows: p0_r1s_lr16, p0_r1s_lr32 plus parity rows p0_cpu / p0_r1s_cpu (the GPU-selected
            full rule re-run on CPU), all on the same draws as the GPU rows.
Output: $M2_OUT/r1lr_cpu/seed*/pool*/c*/eval_rows.jsonl
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

from ssm_decode import fwd_refine as fr                                   # noqa: E402
from ssm_decode import m2_calib as m2                                     # noqa: E402
from ssm_decode.population_calib import PopStats, norm_stats              # noqa: E402

OUT = Path(os.environ.get("M2_OUT", str(ROOT / "results/m2_calib_20261016")))
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BURNS = (0, 10, 20)
RANKS = (16, 32)
SEL_DRAWS = 2
DEVICE = torch.device("cpu")


def gpu_run_dir(seed, pool, cseed):
    return OUT / "runs" / f"seed{seed}" / pool / f"c{cseed}"


def rd_of(args):
    return OUT / "r1lr_cpu" / f"seed{args.seed}" / args.pool / f"c{args.cseed}"


def load_calib(args, world):
    blob = torch.load(gpu_run_dir(args.seed, args.pool, args.cseed) / "best.pt", map_location="cpu",
                      weights_only=False)
    a = blob["args"]
    calib = m2.build_calibrator(world, rank=a["rank"], hidden=a["hidden"], gain=not a["no_gain"],
                                label_free=a.get("label_free", False))
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    for p in calib.parameters():
        p.requires_grad_(False)
    return calib, json.loads((gpu_run_dir(args.seed, args.pool, args.cseed) / "done.json").read_text())


def get_basis(world, rd_pool: Path):
    """[H+1, max(RANKS)] fixed source basis (cached per (seed, pool); calibrator independent)."""
    f = rd_pool / "basis_source.pt"
    if f.exists():
        return torch.load(f)
    g = m2.source_feature_moment(world)
    basis = fr.feature_basis(g, max(RANKS)).float()
    rd_pool.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(basis, tmp)
    os.replace(tmp, f)
    return basis


def pick_rule(df, rank, B):
    """argmax of mean SOURCE R2 over (lambda in LAMS + off, burn) for (rank, B)."""
    sub = df[(df["rank"] == rank) & (df.B == B)].drop_duplicates(
        subset=["session", "draw", "lam_key", "burn"], keep="last")
    g = sub.groupby(["lam_key", "burn"]).r2.agg(["mean", "count"]).reset_index()
    best = g.loc[g["mean"].idxmax()]
    off = g[g.lam_key == "off"]["mean"].iloc[0]
    return {"lam": None if best.lam_key == "off" else float(best.lam_key), "burn": int(best.burn),
            "kind": "lowrank", "rank": int(rank), "source_r2": float(best["mean"]),
            "source_r2_off": float(off), "n_episodes": int(best["count"])}


def cmd_select(args):
    import pandas as pd
    torch.set_num_threads(args.threads)
    world = m2.M2World(args.seed, args.pool, DEVICE)
    calib, _ = load_calib(args, world)
    rd = rd_of(args)
    rd.mkdir(parents=True, exist_ok=True)
    basis = get_basis(world, rd.parent)
    path = rd / "select_r1lr.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["session"], r["B"], r["draw"]))
    trunk, t0 = world.trunk, time.perf_counter()
    for name in world.source_ids:
        sd, prior = world.sessions[name], world.priors[name]
        tr = world.train_trials(name)
        qmask = world.val_mask(name)
        for B in args.budgets:
            for draw in range(SEL_DRAWS):
                if (name, B, draw) in seen:
                    continue
                idx = tr[m2.draw_trials(f"m2sel-{name}", B, draw, len(tr))]
                stats = PopStats.from_trials(sd.bank, idx)
                with torch.no_grad():
                    res = calib.calibrate(stats, prior[0])
                    y0, h = m2.decode_full(trunk, sd, res.table, prior[1], res.mean, res.std, want_h=True)
                    hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                    sts = m2.stream_readout_stats(trunk, sd, prior, calib, idx, burn=BURNS, use_calib=True)
                    rows = [{"rank": r, "lam": None, "burn": 0,
                             "r2": m2.r2_physical(world.label_stats, sd, y0, qmask)[0]} for r in RANKS]
                    for burn, rank, lam in itertools.product(BURNS, RANKS, LAMS):
                        v = m2.ridge_v(trunk, sts[burn], lam, "lowrank", basis=basis[:, :rank])
                        y = hb @ v.to(hb.dtype).T
                        rows.append({"rank": rank, "lam": lam, "burn": burn,
                                     "r2": m2.r2_physical(world.label_stats, sd, y, qmask)[0]})
                with path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps({"session": name, "B": B, "draw": draw, **r}) + "\n")
                print(f"[select-r1lr s{args.seed} {args.pool} c{args.cseed}] {name} B={B} d={draw} "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines()])
    df["lam_key"] = df["lam"].map(lambda x: "off" if x is None or x != x else repr(float(x)))
    sel = {str(r): {str(B): pick_rule(df, r, B) for B in sorted(df.B.unique())} for r in RANKS}
    (rd / "selection_r1lr.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel))


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add((r["variant"], r["target"], r["B"], r["draw"]))
    return keys


def cmd_evaluate(args):
    torch.set_num_threads(args.threads)
    world = m2.M2World(args.seed, args.pool, DEVICE)
    trunk = world.trunk
    calib, done = load_calib(args, world)
    rd = rd_of(args)
    basis = get_basis(world, rd.parent)
    sel = json.loads((rd / "selection_r1lr.json").read_text())
    sel_full = json.loads((gpu_run_dir(args.seed, args.pool, args.cseed) / "selection_r1.json").read_text())
    path = rd / "eval_rows.jsonl"
    seen = done_keys(path)
    hash0 = trunk.trunk_hash
    fp32 = m2.table_bytes(world.k, m2.N_CHANNELS)["fp32_bytes"]
    r1_bytes = {f"lr{r}": m2.r1_stream_bytes("lowrank", r) for r in RANKS}
    r1_bytes["full"] = m2.r1_stream_bytes("full")
    t0 = time.perf_counter()
    names = ["p0_cpu", "p0_r1s_cpu"] + [f"p0_r1s_lr{r}" for r in RANKS]
    for target in world.targets:
        sd, prior = world.target(target)
        n_tr = sd.bank.n_trials
        for B in args.budgets:
            for draw in range(args.draws):
                if all((n, target, B, draw) in seen for n in names):
                    continue
                idx = m2.draw_trials(target, B, draw, n_tr)
                qmask = sd.query_mask(idx)
                stats = PopStats.from_trials(sd.bank, idx)
                base = {"seed": args.seed, "pool": args.pool, "cseed": args.cseed, "target": target, "B": B,
                        "draw": draw, "n_trials": n_tr, "tag": "p0", "backend": "cpu_plain_export",
                        "trunk_dir": trunk.pool_dir.name, "h4_budget": int(B),
                        "h4_query_disjoint": bool(not (_used(sd, idx) & qmask.cpu().numpy()).any()),
                        "h2_backward_calls": 0, "h2_zero_backprop": True}
                rule_full = sel_full["p0"][str(B)]
                rules = {r: sel[str(r)][str(B)] for r in RANKS}
                burns = sorted({rule_full["burn"]} | {rl["burn"] for rl in rules.values()})
                rows = []
                with torch.no_grad():
                    res = calib.calibrate(stats, prior[0])
                    y0, h = m2.decode_full(trunk, sd, res.table, prior[1], res.mean, res.std, want_h=True)
                    hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                    sts = m2.stream_readout_stats(trunk, sd, prior, calib, idx, burn=burns, use_calib=True)

                    def emit(variant, y, **extra):
                        r2, unit = m2.r2_physical(world.label_stats, sd, y, qmask)
                        rows.append({**base, "variant": variant, "r2": r2, "unit_ratio": unit["unit_ratio"],
                                     "unit_check_passed": unit["passed"], "h5_bytes": fp32,
                                     "h5_within_32kb": fp32 <= 32 * 1024, **extra})

                    emit("p0_cpu", y0)
                    if rule_full["lam"] is None:
                        emit("p0_r1s_cpu", y0, rule=rule_full, r1_sram=r1_bytes["full"])
                    else:
                        v = m2.ridge_v(trunk, sts[rule_full["burn"]], rule_full["lam"], rule_full["kind"])
                        emit("p0_r1s_cpu", hb @ v.to(hb.dtype).T, rule=rule_full, r1_sram=r1_bytes["full"])
                    for r, rule in rules.items():
                        if rule["lam"] is None:
                            emit(f"p0_r1s_lr{r}", y0, rule=rule, r1_sram=r1_bytes[f"lr{r}"])
                        else:
                            v = m2.ridge_v(trunk, sts[rule["burn"]], rule["lam"], "lowrank",
                                           basis=basis[:, :r])
                            emit(f"p0_r1s_lr{r}", hb @ v.to(hb.dtype).T, rule=rule, r1_sram=r1_bytes[f"lr{r}"])
                assert trunk._hash() == hash0, "trunk changed"
                rows = [{**r, "h1_trunk_unchanged": True, "h1_trunk_hash": hash0[:16]} for r in rows
                        if (r["variant"], target, B, draw) not in seen]
                with path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"[eval-r1lr s{args.seed} {args.pool} c{args.cseed}] {target[-8:]} B={B} d={draw} "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)


def _used(sd, idx):
    used = np.zeros(sd.n_bins, dtype=bool)
    for i in idx:
        a, b = sd.trials[i]
        used[a:b] = True
    return used


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("select", cmd_select), ("evaluate", cmd_evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--pool", required=True)
        p.add_argument("--cseed", type=int, default=0)
        p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 33])
        p.add_argument("--draws", type=int, default=8)
        p.add_argument("--threads", type=int, default=8)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
