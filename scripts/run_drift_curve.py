#!/usr/bin/env python
"""Drift-vs-gap curve on DANDI 000688 sub-M: many targets, one frozen month AP backbone.

Re-uses, WITHOUT re-selection, the existing month artefacts:
  * frozen month AP trunk (bit-identical to the year trunk, checked by ``check-trunk``),
  * P1 P0 calibrators (pool O, seeds 0/1),
  * causal source-selected rules of P3 section R.12 (``selection_month_oosc.json``; source set = {20140307}):
      R1-stream (lambda, readout kind by best source score), R2 (F7-J), R3 (F7-J + readout ridge),
  * the like-for-like non-deployable reference P0+ui_reg (``uiref/selection_month_oosc.json``).

Rows per (target, B, draw, seed): p_prior (electrode-aligned mean prior, seed-free, stored with seed 0),
p0, r1s (R1-stream), r2, r3, p0_ui_reg.  All candidate paths run on B labelled trials only, no backprop
through the trunk (counted); ui_reg backpropagates through the frozen trunk and is a reference only.
Decoding is one contiguous causal pass over the whole session; only query bins are scored.

  python scripts/run_drift_curve.py check-targets
  python scripts/run_drift_curve.py check-trunk
  python scripts/run_drift_curve.py eval --targets 20140626,20140627 [--draws 8]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P3_FIX = ROOT / "results/population_calib_p3_20261016/fix"
OUT = ROOT / "results/drift_curve_20261016"


def _use_p3_fix_dir():
    """Jacobian / basis / selection files live in the P3 review-fix dir (must precede run_pc3_refine import)."""
    os.environ["PC_OUT"] = str(P3_FIX)

STRATUM = "month"
LATEST_SOURCE = "20140306"
SELECTION_SESSIONS = ("20140307",)           # causal OOS source set of the month rules (README P3 R.12)
FINAL_DATES = ("20150623", "20150625", "20150626")
TARGET_DATES = ("20140626", "20140627", "20140929", "20141203", "20150511", "20150512",
                "20150610", "20150611", "20150612", "20150615", "20150616", "20150617")
B_LIST = (1, 2, 4, 8, 16, 33)
PREFIX = "sub-M_ses-CO-"
MIN_TRIALS_FACTOR = 33 + 20                  # need B=33 calibration trials plus >= 20 query trials


def days_from_source(date: str) -> int:
    f = lambda s: dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))        # noqa: E731
    return (f(date) - f(LATEST_SOURCE)).days


def validate_targets(dates, source_dates, selection_dates=SELECTION_SESSIONS, final_dates=FINAL_DATES):
    """Reject final sessions, any source / selection session, duplicates and non-later sessions."""
    dates = [str(d) for d in dates]
    if len(set(dates)) != len(dates):
        raise ValueError(f"duplicate targets: {dates}")
    latest = max(source_dates)
    for d in dates:
        if len(d) != 8 or not d.isdigit():
            raise ValueError(f"target must be YYYYMMDD, got {d!r}")
        if d in final_dates:
            raise ValueError(f"{d} is a FINAL session; never use it")
        if d in selection_dates:
            raise ValueError(f"{d} is a source-selection session of the causal rules")
        if d in source_dates:
            raise ValueError(f"{d} is a trunk-source session")
        if d <= latest:
            raise ValueError(f"{d} is not later than the latest source session {latest}")
    return dates


def check_selection_sets(pc_sel, ui_sel):
    """The rules must have been selected on the causal set {20140307} only (no re-selection here)."""
    for B in map(str, B_LIST):
        got = tuple(ui_sel[B]["sessions"])
        if got != SELECTION_SESSIONS:
            raise ValueError(f"ui_reg rule for B={B} selected on {got}, expected {SELECTION_SESSIONS}")
    for key in ("r2", "r1s_full", "r3_full"):
        if set(pc_sel[key]) != set(map(str, B_LIST)):
            raise ValueError(f"selection {key} lacks some B: {sorted(pc_sel[key])}")


def check_mapping(sd, prior, orig_elec, name):
    """Structural unit / electrode-mapping checks (fail loudly).  Returns a record for the README."""
    el = np.asarray(sd.electrodes)
    n = sd.n_units
    if el.ndim != 1 or el.shape[0] != n:
        raise ValueError(f"{name}: unit_to_electrode has shape {el.shape} for {n} units")
    if not np.issubdtype(el.dtype, np.integer):
        raise ValueError(f"{name}: unit_to_electrode is not integer ({el.dtype})")
    if el.min() < 0 or el.max() >= 96:
        raise ValueError(f"{name}: electrode index out of [0, 96): {el.min()}..{el.max()}")
    if tuple(prior.shape) != (10, n):
        raise ValueError(f"{name}: prior shape {tuple(prior.shape)} != (K=10, N={n})")
    if not bool(np.isfinite(prior.detach().cpu().numpy()).all()):
        raise ValueError(f"{name}: non-finite prior")
    cnt = np.zeros(96, dtype=int)
    for e in orig_elec:
        for x in e:
            cnt[x] += 1
    covered = cnt[el] > 0
    n_trials = len(sd.trials)
    if n_trials < MIN_TRIALS_FACTOR:
        raise ValueError(f"{name}: only {n_trials} usable trials (< {MIN_TRIALS_FACTOR})")
    zero_prior = float((prior.detach().abs().sum(0) == 0).float().mean())
    return {"target": name[-8:], "n_units": int(n), "n_distinct_electrodes": int(len(np.unique(el))),
            "frac_units_electrode_in_sources": float(covered.mean()),
            "frac_units_zero_prior_column": zero_prior, "n_usable_trials": int(n_trials),
            "n_bins": int(sd.n_bins), "mean_count_per_bin": float(sd.x32.mean())}


def pick_kind(sel, prefix, B):
    _use_p3_fix_dir()
    from scripts.analyze_pc3 import KINDS
    cand = {k: sel[f"{prefix}_{k}"][str(B)]["source_r2"] for k in KINDS}
    return max(cand, key=cand.get)               # same source-only choice as analyze_pc3.best_kind


def cmd_check_trunk(args):
    import torch
    from scripts.run_labeled_calibration import D1_POOLS, APBackboneAdapter
    sd = {}
    for st in ("month", "year"):
        a = APBackboneAdapter(ROOT / D1_POOLS[st]["ap"], torch.device("cpu"))
        sd[st] = {k: v.clone() for k, v in a.model.state_dict().items()}
        sd[st + "_hash"] = a.trunk_hash
    same = sd["month"].keys() == sd["year"].keys() and all(torch.equal(sd["month"][k], sd["year"][k]) for k in sd["month"])
    print(json.dumps({"month_hash": sd["month_hash"][:16], "year_hash": sd["year_hash"][:16],
                      "state_dicts_bitwise_equal": bool(same)}))
    if not same:
        raise SystemExit("month and year trunks differ")


def cmd_check_targets(args):
    _use_p3_fix_dir()
    import torch
    from scripts import run_pc3_refine as pc
    from scripts.run_labeled_calibration import D1_POOLS
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(TARGET_DATES, src)
    world = pc.load_world(STRATUM, False)
    recs = []
    for d in dates:
        sd, prior = world.target(PREFIX + d)
        r = check_mapping(sd, prior, world.orig_elec, PREFIX + d)
        r["days_from_latest_source"] = days_from_source(d)
        recs.append(r)
        print(json.dumps(r), flush=True)
        del sd, prior
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "target_inventory.json").write_text(json.dumps(recs, indent=1))


def cmd_eval(args):
    _use_p3_fix_dir()
    import torch
    from scripts import run_pc3_refine as pc
    from scripts import run_pc3_uiref as ui
    from scripts import run_population_calib as rpc
    from scripts.run_labeled_calibration import D1_POOLS

    pc.set_variant("base")
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(args.targets.split(","), src)
    sel = json.loads(pc.selection_file(STRATUM, "oosc").read_text())
    ui_sel = json.loads((ui.uidir() / f"selection_{STRATUM}_oosc.json").read_text())
    check_selection_sets(sel, ui_sel)
    og, ob = pc._patch_autograd_counter()
    world = pc.load_world(STRATUM, False)
    p0 = pc.load_p0(world, STRATUM)
    basis = pc.load_basis(STRATUM)
    hash0 = world.adapter._hash()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rows_{'-'.join(d[2:] for d in dates)}.jsonl"
    seen = pc.jsonl_keyset(path, ("method", "seed", "target", "B", "draw"))
    t0 = time.perf_counter()
    for d in dates:
        target = PREFIX + d
        sd, prior = world.target(target)
        info = check_mapping(sd, prior, world.orig_elec, target)
        print(f"[drift] {d} N={info['n_units']} cov={info['frac_units_electrode_in_sources']:.2f} "
              f"trials={info['n_usable_trials']}", flush=True)
        for B in B_LIST:
            rule = sel["r2"][str(B)]["cfg"]
            jb = pc.jbar_for(STRATUM, rule["L"])
            k1, k3 = pick_kind(sel, "r1s", B), pick_kind(sel, "r3", B)
            lam1, lam3 = sel[f"r1s_{k1}"][str(B)]["cfg"]["lam"], sel[f"r3_{k3}"][str(B)]["cfg"]["lam"]
            u = ui_sel[str(B)]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in pc.SEEDS:
                    need = [m for m in ("p0", "p_prior", "r1s", "r2", "r3", "p0_ui_reg")
                            if (m, seed, target, B, draw) not in seen and not (m == "p_prior" and seed != 0)]
                    if not need:
                        continue
                    ep = pc.p0_episode(world, sd, prior, p0[seed], idx)
                    used = np.zeros(sd.n_bins, dtype=bool)
                    for i in idx:
                        a, b = sd.trials[i]
                        used[a:b] = True
                    base = {"target": target, "date": d, "days": days_from_source(d), "n_units": info["n_units"],
                            "B": B, "draw": draw, "seed": seed, "h4_budget": int(B),
                            "h4_query_disjoint": bool(not (used & ep.q.cpu().numpy()).any()),
                            "n_query_bins": int(ep.q.sum()), "replay_bins": ep.replay_bins,
                            "max_trial_len": max(ep.trial_lens)}
                    rows = []
                    pc._BWD["n"] = 0

                    def emit(method, r2, cfg, **extra):
                        rows.append({**base, "method": method, "r2": r2, "cfg": cfg,
                                     "h2_backward_calls": pc._BWD["n"], **extra})
                    if "p0" in need:
                        emit("p0", ep.r2(ep.table0), {})
                    if "p_prior" in need:
                        epp = pc.prior_episode(world, sd, prior, idx)
                        emit("p_prior", epp.r2(epp.table0), {})
                    if "r1s" in need:
                        sts = pc.stream_stats(world, sd, prior, p0[seed], idx, None)
                        emit("r1s", ep.r2(ep.table0, ep.v_from(sts, lam1, k1, basis)), {"kind": k1, "lam": lam1})
                    if "r2" in need or "r3" in need:
                        tab = ep.refine(jb, rule["eta"], rule["rho"], [rule["S"]])[rule["S"]]
                        if "r2" in need:
                            emit("r2", ep.r2(tab), dict(rule))
                        if "r3" in need:
                            st3 = ep.stats(tab)
                            emit("r3", ep.r2(tab, ep.v_from(st3, lam3, k3, basis)),
                                 {**rule, "kind": k3, "lam": lam3})
                    bwd_cand = pc._BWD["n"]
                    if "p0_ui_reg" in need:      # reference: gradient through the frozen trunk (not a candidate)
                        tab = ui.run_traj(world, sd, ep, u["lam"], u["lr"], (u["steps"],))[u["steps"]]
                        rows.append({**base, "method": "p0_ui_reg",
                                     "r2": ep.r2(torch.from_numpy(tab).float().to(world.device)),
                                     "cfg": {"lam": u["lam"], "lr": u["lr"], "steps": u["steps"]},
                                     "h2_backward_calls": None, "note": "non-deployable reference"})
                    for r in rows:
                        if r["method"] != "p0_ui_reg":
                            r["h2_backward_calls"] = bwd_cand
                    with path.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[drift] {d} B={B} d={draw} s{seed} t={time.perf_counter() - t0:.0f}s", flush=True)
        del sd, prior
    assert world.adapter._hash() == hash0, "trunk changed"
    torch_ = __import__("torch")
    torch_.autograd.grad, torch_.autograd.backward = og, ob


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-targets")
    sub.add_parser("check-trunk")
    e = sub.add_parser("eval")
    e.add_argument("--targets", required=True, help="comma list of YYYYMMDD")
    e.add_argument("--draws", type=int, default=8)
    args = p.parse_args()
    {"check-targets": cmd_check_targets, "check-trunk": cmd_check_trunk, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
