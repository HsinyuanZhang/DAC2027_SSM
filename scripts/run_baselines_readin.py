#!/usr/bin/env python
"""Published read-in-alignment baselines (Stabilizer, CCA/PCA alignment) on the frozen AP backbone.

Subcommands
  select   source-only hyper-parameter selection (pseudo-targets = the other source sessions)
  eval     target rows on the same (target, B, draw) calibration subsets as the P1/P3 rows
  sanity   bridge check on the baseline session itself (no target data)

Protocol identical to scripts/run_population_calib.py: draws via ``draw_trials`` (same RNG),
contiguous causal decoding of the whole recording, query R2 in physical units on usable trials
that are not calibration trials.  Never touches FINAL sessions or results/official_evalai/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_labeled_calibration import D1_POOLS                       # noqa: E402
from scripts.run_population_calib import World, draw_trials, r2_physical   # noqa: E402
from ssm_decode import published_baselines as pb                           # noqa: E402
from ssm_decode.dandi688_data import dandi688_subm_root                    # noqa: E402

OUT = Path(os.environ.get("BL_OUT", str(ROOT / "results/baselines_20261016")))
B_GRID = (1, 4, 8, 16, 33)
N_DRAWS = 8
K = 10
_ANGLE_CACHE: dict = {}


# ---------------------------------------------------------------- data ----

def trial_angles(session: str, record) -> dict:
    """(start_bin, stop_bin) -> reach-direction angle (rad) from the NWB trials table (task metadata)."""
    if session in _ANGLE_CACHE:
        return _ANGLE_CACHE[session]
    from pynwb import NWBHDF5IO
    path = dandi688_subm_root() / f"{session}_behavior+ecephys.nwb"
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        df = io.read().intervals["trials"].to_dataframe()
    edges = np.asarray(record.bin_edges)
    n_bins = len(edges) - 1
    out = {}
    for _, row in df.iterrows():
        a = max(0, int(np.searchsorted(edges, float(row.start_time))))
        b = min(n_bins, int(np.searchsorted(edges, float(row.stop_time))))
        if b <= a:
            continue
        ang = row.get("target_dir")
        out[(a, b)] = float(ang) if ang is not None and np.isfinite(ang) else float("nan")
    assert set(record.trial_bounds) <= set(out), "trial table / bounds mismatch"
    _ANGLE_CACHE[session] = out
    return out


def fit_bins(sd, idx) -> np.ndarray:
    """Raw counts of the eval-valid bins of trials ``idx`` (calibration neural data)."""
    rows = []
    for i in idx:
        a, b = sd.trials[i]
        rows.append(sd.record.neural[a:b][sd.valid_np[a:b]])
    return np.concatenate(rows).astype(np.float64)


def trial_blocks(sd, idx):
    """Per-trial raw counts [T_i, N] (full interval) and reach-direction labels."""
    ang = trial_angles(sd.name, sd.record)
    xs, cs = [], []
    for i in idx:
        a, b = sd.trials[i]
        xs.append(sd.record.neural[a:b].astype(np.float64))
        cs.append(pb.condition_label(ang[(a, b)]))
    return xs, cs


def decode_affine(world, sd, t, c, mean, std, transform="raw"):
    dev = world.device
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    c_t = torch.as_tensor(c, dtype=torch.float32, device=dev)
    x_in = torch.sqrt(torch.clamp(sd.x32, min=0.0)) if transform == "sqrt" else sd.x32
    xz = (x_in - mean_t) / std_t
    with torch.inference_mode():
        return world.adapter.model.base((xz @ t_t.T + c_t).unsqueeze(0)).squeeze(0)


def build_ref(world, fa_min_priv, seed=0, transform="raw"):
    """Baseline day = the most recent source session of the backbone's pool."""
    from ssm_decode.population_calib import SessionData
    name = world.orig_ids[-1]
    from ssm_decode.dandi688_data import load_dandi688_recording
    rec = load_dandi688_recording(name)
    sd = SessionData(rec, world.label_stats, world.device, name=name)
    stats = world.adapter.blob["source_statistics"][len(world.orig_ids) - 1]
    ref_mean, ref_std = np.asarray(stats[0], np.float64), np.asarray(stats[1], np.float64)
    w_ref = world.orig_tables[-1]                                   # [K, N]
    idx = np.arange(len(sd.trials))
    x_fit = fit_bins(sd, idx)
    z_ref = ((x_fit - ref_mean) / ref_std) @ w_ref.T
    xs, cs = trial_blocks(sd, idx)
    ref = pb.build_ref_day(name=name, x_fit=x_fit, z_ref=z_ref, electrodes=sd.electrodes,
                           ref_mean=ref_mean, ref_std=ref_std, x_ref_trials=xs, ref_conds=cs,
                           k=K, fa_min_priv=fa_min_priv, fa_seed=seed, transform=transform)
    return ref, sd


# ------------------------------------------------------------- methods ----

def run_method(method, cfg, ref, sd, idx_cal, seed=0):
    """Return (table T, offset c, mean, std, info) for one calibration subset.

    ``stab_*`` / ``cca_*``; suffix ``_b`` = all target-session trials (paper regime, H4 violation),
    otherwise the B calibration trials only.
    """
    use_all = method.endswith("_b")
    idx_fit = np.arange(len(sd.trials)) if use_all else idx_cal
    if method.startswith("stab"):
        return pb.stabilizer_readin(ref, fit_bins(sd, idx_fit), sd.electrodes, k=K, align_frac=cfg["frac"],
                                    fa_min_priv=cfg["floor"], seed=seed, align="noalign" not in method)
    if method.startswith("cca"):
        xs, cs = trial_blocks(sd, idx_fit)
        return pb.cca_readin(ref, fit_bins(sd, idx_fit), xs, cs, k=K, ridge=cfg["ridge"],
                             transform=cfg.get("transform", "raw"))
    raise ValueError(method)


METHOD_META = {
    "stab_a": dict(target_data="B calibration trials, neural only (no labels)", backprop=False, h4=True),
    "stab_a_noalign": dict(target_data="B calibration trials, neural only (no labels); control without rotation",
                           backprop=False, h4=True),
    "stab_b": dict(target_data="ALL target-session usable trials, neural only (no labels; includes query bins)",
                   backprop=False, h4=False),
    "cca_a": dict(target_data="B calibration trials: neural + reach-direction labels", backprop=False, h4=True),
    "cca_b": dict(target_data="ALL target-session usable trials: neural + reach-direction labels (includes query trials)",
                  backprop=False, h4=False),
}


def native_query_mask(sd, idx):
    """Query bins whose 50-bin causal window lies fully inside one trial (APST's own query rule)."""
    start = torch.full((sd.n_bins,), -10 ** 9, dtype=torch.long, device=sd.device)
    for a, b in sd.trials:
        start[a:b] = a
    pos = torch.arange(sd.n_bins, device=sd.device) - start
    return sd.query_mask(idx) & (pos >= 49)


def score(world, sd, t, c, mean, std, idx, transform="raw", native=False):
    pred = decode_affine(world, sd, t, c, mean, std, transform)
    r2, unit = r2_physical(world, sd, pred, sd.query_mask(idx))
    if native:
        r2n, _ = r2_physical(world, sd, pred, native_query_mask(sd, idx))
        return float(r2), unit, float(r2n)
    return float(r2), unit


# -------------------------------------------------------------- select ----

def cmd_select(args):
    device = torch.device(args.device)
    world = World(args.stratum, "O", device, load_sessions=True)
    sel_path = OUT / "select" / f"select_{args.stratum}.jsonl"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if sel_path.exists():
        for line in sel_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["method"], json.dumps({"transform": "raw", **r["cfg"]}, sort_keys=True), r["pseudo"], r["B"], r["draw"]))
    refs = {(fl, "raw"): build_ref(world, fl)[0] for fl in (0.01, 0.1)}
    refs[(0.1, "sqrt")] = build_ref(world, 0.1, transform="sqrt")[0]
    ref_sd = None
    pseudo = world.orig_ids[:-1]
    grid = []
    for fl in (0.01, 0.1):
        for fr in (0.5, 0.75, 1.0):
            grid.append(("stab_a", {"frac": fr, "floor": fl}))
    grid.append(("stab_a_noalign", {"frac": 1.0, "floor": 0.1}))
    for tr in ("raw", "sqrt"):
        grid.append(("cca_a_plain", {"ridge": 0.0, "floor": 0.1, "transform": tr}))
        for rg in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
            grid.append(("cca_a", {"ridge": rg, "floor": 0.1, "transform": tr}))
    for method, cfg in grid:
        ref = refs[(cfg["floor"], cfg.get("transform", "raw"))]
        for name in pseudo:
            sd, _ = world.sessions[name], None
            for B in (4, 16):
                for draw in range(2):
                    key = (method, json.dumps({"transform": "raw", **cfg}, sort_keys=True), name, B, draw)
                    if key in done:
                        continue
                    idx = draw_trials(f"val-{name}", B, draw + 1234, sd.bank.n_trials)
                    res = run_method(method, cfg, ref, sd, idx)
                    if res is None:
                        r2 = float("nan")
                        info = {}
                    else:
                        t, c, mean, std, info = res
                        r2, _ = score(world, sd, t, c, mean, std, idx, cfg.get("transform", "raw"))
                    row = {"stratum": args.stratum, "method": method, "cfg": {"transform": "raw", **cfg}, "pseudo": name,
                           "B": B, "draw": draw, "r2": r2, "info": info}
                    with sel_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row[k] for k in ("method", "cfg", "pseudo", "B", "r2")}), flush=True)


def selected_cfg(stratum):
    """Best config per method family by mean pseudo-target R2 (source sessions only)."""
    import collections
    rows = [json.loads(l) for l in (OUT / "select" / f"select_{stratum}.jsonl").read_text().splitlines()]
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["method"], json.dumps(r["cfg"], sort_keys=True))].append(r["r2"])
    best = {}
    for (m, cfg), v in agg.items():
        score_ = float(np.nanmean(v))
        if m not in best or score_ > best[m][0]:
            best[m] = (score_, json.loads(cfg))
    return {m: c for m, (s, c) in best.items()}, {m: s for m, (s, c) in best.items()}


def best_ridge_cfg(stratum, transform):
    """Best ridge for CCA within one input transform (source pseudo-targets only)."""
    import collections
    rows = [json.loads(l) for l in (OUT / "select" / f"select_{stratum}.jsonl").read_text().splitlines()]
    agg = collections.defaultdict(list)
    for r in rows:
        if r["method"] == "cca_a" and r["cfg"].get("transform", "raw") == transform:
            agg[json.dumps(r["cfg"], sort_keys=True)].append(r["r2"])
    best = max(agg, key=lambda k: float(np.nanmean(agg[k])))
    return json.loads(best)


# ----------------------------------------------------------------- eval ----

def cmd_eval(args):
    device = torch.device(args.device)
    cfgs, _ = selected_cfg(args.stratum)
    world = World(args.stratum, "O", device, load_sessions=False)
    floor = cfgs["stab_a"]["floor"]
    refs = {"raw": build_ref(world, floor)[0], "sqrt": build_ref(world, 0.1, transform="sqrt")[0]}
    rows_path = OUT / "eval" / f"eval_{args.stratum}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["method"], r["target"], r["B"], r["draw"]))
    cr = {tr: best_ridge_cfg(args.stratum, tr) for tr in ("raw", "sqrt")}
    plan = [("stab_a", cfgs["stab_a"]), ("stab_a_noalign", cfgs["stab_a_noalign"]), ("stab_b", cfgs["stab_a"])]
    for tr, tag in (("raw", "cca"), ("sqrt", "cca_sqrt")):
        for part in ("a", "b"):
            plan.append((f"{tag}_{part}", {"ridge": 0.0, "floor": 0.1, "transform": tr}))
            plan.append((f"{tag}_{part}_ridge", cr[tr]))
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, _ = world.target(target)
        cache_b = {}
        for name, cfg in plan:
            base = name.removesuffix("_ridge")
            meta = METHOD_META[base.replace("cca_sqrt", "cca")]
            tr = cfg.get("transform", "raw")
            for B in B_GRID:
                for draw in range(N_DRAWS):
                    if (name, target, B, draw) in seen:
                        continue
                    idx = draw_trials(target, B, draw, sd.bank.n_trials)
                    t0 = time.perf_counter()
                    if base.endswith("_b"):                      # alignment independent of B/draw
                        if name not in cache_b:
                            cache_b[name] = run_method(base, cfg, refs[tr], sd, idx)
                        res = cache_b[name]
                    else:
                        res = run_method(base, cfg, refs[tr], sd, idx)
                    row = {"stratum": args.stratum, "method": name, "target": target, "B": B,
                           "draw": draw, "cfg": cfg, "target_data": meta["target_data"],
                           "target_backprop": meta["backprop"], "h4_compliant": meta["h4"],
                           "ref_session": refs[tr].name}
                    if res is None:
                        row.update(r2=None, status="NO_COMMON_CONDITION")
                    else:
                        t, c, mean, std, info = res
                        r2, unit, r2n = score(world, sd, t, c, mean, std, idx, tr, native=True)
                        row.update(r2=r2, r2_native=r2n, status="OK", unit_ratio=unit["unit_ratio"],
                                   unit_check_passed=unit["passed"], info=info,
                                   seconds=round(time.perf_counter() - t0, 3))
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row.get(k) for k in ("stratum", "method", "target", "B", "draw", "r2")}), flush=True)


# --------------------------------------------------------------- sanity ----

def cmd_sanity(args):
    """Bridge check: use the baseline session itself as the 'new day' (all its trials)."""
    device = torch.device(args.device)
    world = World(args.stratum, "O", device, load_sessions=True)
    ref, sd = build_ref(world, 0.1)
    idx_all = np.arange(len(sd.trials))
    own = world.sessions[ref.name]
    # source table decode (what the trunk was trained with)
    tab = torch.as_tensor(world.orig_tables[-1], dtype=torch.float32, device=device)
    mean, std = ref.mean, ref.std
    pred = decode_affine(world, own, tab.cpu().numpy(), np.zeros(K), mean, std)
    q = own.valid & (own.trial_id >= 0)
    r2_src, _ = r2_physical(world, own, pred, q)
    out = {"ref": ref.name, "r2_source_table": float(r2_src)}
    for m, cfg in (("stab_b", {"frac": 1.0, "floor": 0.1}), ("cca_b", {"ridge": 0.0, "floor": 0.1})):
        t, c, mean2, std2, info = run_method(m, cfg, ref, own, idx_all)
        pred = decode_affine(world, own, t, c, mean2, std2)
        r2, _ = r2_physical(world, own, pred, q)
        out[m] = float(r2)
        out[m + "_info"] = info
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["select", "eval", "sanity"])
    p.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    {"select": cmd_select, "eval": cmd_eval, "sanity": cmd_sanity}[args.cmd](args)


if __name__ == "__main__":
    main()
