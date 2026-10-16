#!/usr/bin/env python
"""Published read-in-alignment baselines (Stabilizer, PCA+CCA) on the frozen AP backbone.

Subcommands
  select   source-only hyper-parameter selection (pseudo-targets = the other source sessions), whole grid fixed
           before any target row is evaluated (stage 1: latent dimension d, Stabilizer floor/fraction, CCA ridge;
           stage 2: CCA number of phases and trial alignment at the stage-1 optimum)
  eval     target rows on the same (target, B, draw) calibration subsets as the P1/P3 rows
  sanity   bridge ceiling (baseline session as its own 'new day') for the configurations used in the table

Input types: ``units`` (sorted units; the Stabilizer then needs the ad-hoc same-electrode unit matching) and
``pooled`` (counts summed per electrode into the fixed 96 M1 channels; no unit matching, columns are
electrode-consistent by construction).

Protocol identical to scripts/run_population_calib.py: draws via ``draw_trials`` (same RNG),
contiguous causal decoding of the whole recording, query R2 in physical units on usable trials
that are not calibration trials.  Never touches FINAL sessions or results/official_evalai/.
"""
from __future__ import annotations

import argparse
import collections
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

from scripts.run_labeled_calibration import D1_POOLS                       # noqa: E402
from scripts.run_population_calib import World, draw_trials, r2_physical   # noqa: E402
from ssm_decode import published_baselines as pb                           # noqa: E402
from ssm_decode.dandi688_data import dandi688_subm_root                    # noqa: E402
from ssm_decode.electrode_align import N_ELECTRODES, pool_counts, pool_matrix   # noqa: E402

OUT = Path(os.environ.get("BL_OUT", str(ROOT / "results/baselines_20261016")))
B_GRID = (1, 4, 8, 16, 33)
N_DRAWS = 8
K = 10                                   # trunk (AP latent) dimension
D_GRID = (3, 5, 8, 10, "N")              # FA / PCA latent dimension candidates ("N" = all input dimensions)
GO_WINDOW = 50                           # bins after the go cue used by the go-aligned CCA trajectories
_META_CACHE: dict = {}


# ---------------------------------------------------------------- data ----

def trial_meta(session: str, record) -> dict:
    """(start_bin, stop_bin) -> (reach-direction angle (rad), go-cue bin) from the NWB trials table (task metadata)."""
    if session in _META_CACHE:
        return _META_CACHE[session]
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
        go = row.get("go_cue_time")
        out[(a, b)] = (float(ang) if ang is not None and np.isfinite(ang) else float("nan"),
                       int(np.searchsorted(edges, float(go))) if go is not None and np.isfinite(go) else -1)
    assert set(record.trial_bounds) <= set(out), "trial table / bounds mismatch"
    _META_CACHE[session] = out
    return out


def fit_bins(sd, idx) -> np.ndarray:
    """Raw unit counts of the eval-valid bins of trials ``idx`` (calibration neural data)."""
    rows = []
    for i in idx:
        a, b = sd.trials[i]
        rows.append(sd.record.neural[a:b][sd.valid_np[a:b]])
    return np.concatenate(rows).astype(np.float64)


def featurize(x, sd, inp):
    """Unit counts -> method input: sorted units as they are, or counts pooled per electrode (96 channels)."""
    return x if inp == "units" else pool_counts(x, sd.electrodes)


def feature_electrodes(sd, inp):
    return np.asarray(sd.electrodes) if inp == "units" else np.arange(N_ELECTRODES)


def trial_blocks(sd, idx, inp="units", align="trial"):
    """Per-trial counts [T_i, D] and reach-direction labels.

    ``align='trial'``: the whole trial (time-normalised later); ``align='go'``: the ``GO_WINDOW`` bins after the
    go cue (trials without a go cue or too short get label -1 and are skipped by the CCA).
    """
    meta = trial_meta(sd.name, sd.record)
    xs, cs = [], []
    for i in idx:
        a, b = sd.trials[i]
        ang, go = meta[(a, b)]
        cond = pb.condition_label(ang)
        if align == "go":
            if go < a or go + GO_WINDOW > b:
                cond = -1
                seg = sd.record.neural[a:b]
            else:
                seg = sd.record.neural[go:go + GO_WINDOW]
        else:
            seg = sd.record.neural[a:b]
        xs.append(featurize(seg.astype(np.float64), sd, inp))
        cs.append(cond)
    return xs, cs


def decode_affine(world, sd, t, c, mean, std, transform="raw", pool=False):
    dev = world.device
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    c_t = torch.as_tensor(c, dtype=torch.float32, device=dev)
    x_in = sd.x32
    if pool:                                              # per-electrode sums, fixed 96-channel order
        x_in = x_in @ torch.as_tensor(pool_matrix(sd.electrodes).T, dtype=torch.float32, device=dev)
    if transform == "sqrt":
        x_in = torch.sqrt(torch.clamp(x_in, min=0.0))
    xz = (x_in - mean_t) / std_t
    with torch.inference_mode():
        return world.adapter.model.base((xz @ t_t.T + c_t).unsqueeze(0)).squeeze(0)


class RefData:
    """Baseline-day (most recent source session) data and everything that does not depend on the method config."""

    def __init__(self, world, load_session=None):
        from ssm_decode.population_calib import SessionData
        from ssm_decode.dandi688_data import load_dandi688_recording
        self.name = world.orig_ids[-1]
        self.sd = load_session or SessionData(load_dandi688_recording(self.name), world.label_stats,
                                              world.device, name=self.name)
        stats = world.adapter.blob["source_statistics"][len(world.orig_ids) - 1]
        self.mean, self.std = np.asarray(stats[0], np.float64), np.asarray(stats[1], np.float64)
        w_ref = world.orig_tables[-1]                                  # [K, N]
        idx = np.arange(len(self.sd.trials))
        self.idx = idx
        self.x_units = fit_bins(self.sd, idx)
        self.z_ref = ((self.x_units - self.mean) / self.std) @ w_ref.T
        self._cache: dict = {}

    def n_active(self, inp):
        x = featurize(self.x_units, self.sd, inp)
        return int((x.std(0) > 1e-6).sum())

    def ref(self, inp, floor, d, transform, phases, align, kind="cca"):
        key = (inp, floor, d, transform, phases, align, kind)
        if key not in self._cache:
            xs, cs = trial_blocks(self.sd, self.idx, inp, align) if kind == "cca" else ([], [])
            self._cache[key] = pb.build_ref_day(
                name=self.name, x_fit=featurize(self.x_units, self.sd, inp), z_ref=self.z_ref,
                electrodes=feature_electrodes(self.sd, inp), ref_mean=self.mean, ref_std=self.std,
                x_ref_trials=xs, ref_conds=cs, k=d, fa_min_priv=floor, transform=transform, t_phase=phases,
                with_fa=kind == "stab", with_pca=kind == "cca")
        return self._cache[key]


# ------------------------------------------------------------- methods ----

def resolve_d(cfg, refdata, x_fit_feat):
    """Latent dimension actually used: ``cfg['d']`` ('N' = all active input dimensions of the baseline day),
    capped by the number of active input dimensions of the target fit data (FA / PCA need d <= dimensions)."""
    d = refdata.n_active(cfg["inp"]) if cfg["d"] == "N" else int(cfg["d"])
    return int(min(d, int((x_fit_feat.std(0) > 1e-6).sum())))


def run_method(kind, cfg, refdata, sd, idx_cal, *, use_all=False, align=True, seed=0):
    """Return (table T, offset c, mean, std, info) for one calibration subset.

    ``kind`` = ``stab`` / ``cca``; ``use_all`` = all target-session trials (paper regime, H4 violation), otherwise
    the B calibration trials only.  The table acts on the (pooled) feature columns.
    """
    inp = cfg["inp"]
    idx_fit = np.arange(len(sd.trials)) if use_all else idx_cal
    x_feat = featurize(fit_bins(sd, idx_fit), sd, inp)
    d = resolve_d(cfg, refdata, x_feat)
    if kind == "stab":
        ref = refdata.ref(inp, cfg["floor"], d, "raw", 20, "trial", "stab")
        res = pb.stabilizer_readin(ref, x_feat, feature_electrodes(sd, inp), k=d, align_frac=cfg["frac"],
                                   fa_min_priv=cfg["floor"], seed=seed, align=align)
    else:
        ref = refdata.ref(inp, 0.1, d, cfg["transform"], cfg["phases"], cfg["align"], "cca")
        xs, cs = trial_blocks(sd, idx_fit, inp, cfg["align"])
        res = pb.cca_readin(ref, x_feat, xs, cs, k=d, ridge=cfg["ridge"], t_phase=cfg["phases"],
                            transform=cfg["transform"])
    if res is not None:
        res[4]["d_eff"] = d
    return res


# Method registry: eval row name -> (kind, input, transform, CCA mode, part)
EVAL_METHODS = {}
for _inp, _pre in (("units", ""), ("pooled", "pool_")):
    EVAL_METHODS[f"stab_{_pre}a"] = ("stab", _inp, "raw", None, "a")
    EVAL_METHODS[f"stab_{_pre}a_noalign"] = ("stab", _inp, "raw", None, "noalign")
    EVAL_METHODS[f"stab_{_pre}b"] = ("stab", _inp, "raw", None, "b")
    for _tr in (("raw", "sqrt") if _inp == "units" else ("sqrt",)):
        _tag = f"cca_{_pre}{'sqrt_' if _tr == 'sqrt' else ''}"
        for _part in ("a", "b"):
            EVAL_METHODS[f"{_tag}{_part}"] = ("cca", _inp, _tr, "plain", _part)
            EVAL_METHODS[f"{_tag}{_part}_ridge"] = ("cca", _inp, _tr, "ridge", _part)

METHOD_META = {
    "a": dict(target_data="B calibration trials, neural only (no labels)", backprop=False, h4=True),
    "noalign": dict(target_data="B calibration trials, neural only (no labels); control without rotation",
                    backprop=False, h4=True),
    "b": dict(target_data="ALL target-session usable trials, neural only (no labels; includes query bins)",
              backprop=False, h4=False),
}
CCA_META = {
    "a": dict(target_data="B calibration trials: neural + reach-direction labels", backprop=False, h4=True),
    "b": dict(target_data="ALL target-session usable trials: neural + reach-direction labels (includes query trials)",
              backprop=False, h4=False),
}


def native_query_mask(sd, idx):
    """Query bins whose 50-bin causal window lies fully inside one trial (APST's own query rule)."""
    start = torch.full((sd.n_bins,), -10 ** 9, dtype=torch.long, device=sd.device)
    for a, b in sd.trials:
        start[a:b] = a
    pos = torch.arange(sd.n_bins, device=sd.device) - start
    return sd.query_mask(idx) & (pos >= 49)


def score(world, sd, t, c, mean, std, idx, transform="raw", native=False, pool=False):
    pred = decode_affine(world, sd, t, c, mean, std, transform, pool)
    r2, unit = r2_physical(world, sd, pred, sd.query_mask(idx))
    if native:
        r2n, _ = r2_physical(world, sd, pred, native_query_mask(sd, idx))
        return float(r2), unit, float(r2n)
    return float(r2), unit


# -------------------------------------------------------------- select ----

STAB_FLOORS = (0.01, 0.1)
STAB_FRACS = (0.5, 0.75, 1.0)
CCA_RIDGES = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
PHASES = (10, 20, 40)
ALIGNS = ("trial", "go")


def sel_keys():
    """(family key, kind, input, transform, mode) of every independently selected configuration."""
    keys = []
    for inp in ("units", "pooled"):
        keys.append((f"stab/{inp}", "stab", inp, "raw", None))
        for tr in (("raw", "sqrt") if inp == "units" else ("sqrt",)):
            for mode in ("plain", "ridge"):
                keys.append((f"cca/{inp}/{tr}/{mode}", "cca", inp, tr, mode))
    return keys


def stage1_cfgs(kind, inp, tr, mode):
    out = []
    for d in D_GRID:
        if kind == "stab":
            out += [{"inp": inp, "d": d, "floor": fl, "frac": fr} for fl in STAB_FLOORS for fr in STAB_FRACS]
        else:
            for rg in ((0.0,) if mode == "plain" else CCA_RIDGES):
                out.append({"inp": inp, "d": d, "transform": tr, "ridge": rg, "phases": 20, "align": "trial"})
    return out


def cfg_json(cfg):
    return json.dumps(cfg, sort_keys=True)


def read_select(stratum):
    f = OUT / "select" / f"select_{stratum}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def best_cfg(rows, key, stages=(1, 2)):
    """Best configuration of a selection family by mean pseudo-target R2 (source sessions only)."""
    agg = collections.defaultdict(list)
    for r in rows:
        if r["key"] == key and r["stage"] in stages:
            agg[cfg_json(r["cfg"])].append(r["r2"])
    if not agg:
        raise KeyError(key)
    best = max(agg, key=lambda k: float(np.nanmean(agg[k])))
    return json.loads(best), float(np.nanmean(agg[best]))


def cmd_select(args):
    device = torch.device(args.device)
    world = World(args.stratum, "O", device, load_sessions=True)
    sel_path = OUT / "select" / f"select_{args.stratum}.jsonl"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["key"], r["stage"], cfg_json(r["cfg"]), r["pseudo"], r["B"], r["draw"]) for r in read_select(args.stratum)}
    refdata = RefData(world, load_session=world.sessions[world.orig_ids[-1]])
    pseudo = world.orig_ids[:-1]

    def run_stage(stage, plan):
        for key, kind, cfgs in plan:
            for cfg in cfgs:
                for name in pseudo:
                    sd = world.sessions[name]
                    for B in (4, 16):
                        for draw in range(2):
                            if (key, stage, cfg_json(cfg), name, B, draw) in done:
                                continue
                            idx = draw_trials(f"val-{name}", B, draw + 1234, sd.bank.n_trials)
                            res = run_method(kind, cfg, refdata, sd, idx)
                            if res is None:
                                r2, info = float("nan"), {}
                            else:
                                t, c, mean, std, info = res
                                r2, _ = score(world, sd, t, c, mean, std, idx, cfg.get("transform", "raw"),
                                              pool=cfg["inp"] == "pooled")
                            row = {"stratum": args.stratum, "key": key, "stage": stage, "cfg": cfg, "pseudo": name,
                                   "B": B, "draw": draw, "r2": r2, "info": info}
                            with sel_path.open("a") as f:
                                f.write(json.dumps(row) + "\n")
                            done.add((key, stage, cfg_json(cfg), name, B, draw))
            print(json.dumps({"stage": stage, "key": key, "done": True}), flush=True)

    run_stage(1, [(k, kind, stage1_cfgs(kind, inp, tr, mode)) for k, kind, inp, tr, mode in sel_keys()])
    rows = read_select(args.stratum)
    plan2 = []
    for k, kind, inp, tr, mode in sel_keys():
        if kind != "cca":
            continue
        b, _ = best_cfg(rows, k, stages=(1,))
        plan2.append((k, kind, [{**b, "phases": p, "align": al} for p in PHASES for al in ALIGNS
                                if (p, al) != (b["phases"], b["align"])]))
    run_stage(2, plan2)
    rows = read_select(args.stratum)
    for k, kind, inp, tr, mode in sel_keys():
        b, s_ = best_cfg(rows, k)
        print(json.dumps({"selected": k, "cfg": b, "mean_pseudo_r2": round(s_, 4)}), flush=True)


def selected_cfgs(stratum):
    rows = read_select(stratum)
    return {k: best_cfg(rows, k) for k, *_ in sel_keys()}


def method_cfg(name, sel):
    """Selected configuration (and score) used by an eval row name."""
    kind, inp, tr, mode, part = EVAL_METHODS[name]
    key = f"stab/{inp}" if kind == "stab" else f"cca/{inp}/{tr}/{mode}"
    return sel[key][0]


# ----------------------------------------------------------------- eval ----

def cmd_eval(args):
    device = torch.device(args.device)
    sel = selected_cfgs(args.stratum)
    world = World(args.stratum, "O", device, load_sessions=False)
    refdata = RefData(world)
    rows_path = OUT / "eval" / f"eval_{args.stratum}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["method"], r["target"], r["B"], r["draw"]))
    (OUT / "select" / f"selected_{args.stratum}.json").write_text(
        json.dumps({k: {"cfg": c, "mean_pseudo_r2": s} for k, (c, s) in sel.items()}, indent=1))
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, _ = world.target(target)
        cache_b = {}
        for name, (kind, inp, tr, mode, part) in EVAL_METHODS.items():
            cfg = method_cfg(name, sel)
            meta = (CCA_META if kind == "cca" else METHOD_META)[part]
            for B in B_GRID:
                for draw in range(N_DRAWS):
                    if (name, target, B, draw) in seen:
                        continue
                    idx = draw_trials(target, B, draw, sd.bank.n_trials)
                    t0 = time.perf_counter()
                    if part == "b":                              # alignment independent of B/draw
                        if name not in cache_b:
                            cache_b[name] = run_method(kind, cfg, refdata, sd, idx, use_all=True)
                        res = cache_b[name]
                    else:
                        res = run_method(kind, cfg, refdata, sd, idx, align=part != "noalign")
                    row = {"stratum": args.stratum, "method": name, "target": target, "B": B,
                           "draw": draw, "cfg": cfg, "target_data": meta["target_data"],
                           "target_backprop": meta["backprop"], "h4_compliant": meta["h4"],
                           "ref_session": refdata.name}
                    if res is None:
                        row.update(r2=None, status="NO_COMMON_CONDITION")
                    else:
                        t, c, mean, std, info = res
                        r2, unit, r2n = score(world, sd, t, c, mean, std, idx, cfg.get("transform", "raw"),
                                              native=True, pool=inp == "pooled")
                        row.update(r2=r2, r2_native=r2n, status="OK", unit_ratio=unit["unit_ratio"],
                                   unit_check_passed=unit["passed"], info=info,
                                   seconds=round(time.perf_counter() - t0, 3))
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row.get(k) for k in ("stratum", "method", "target", "B", "draw", "r2")}), flush=True)


# --------------------------------------------------------------- sanity ----

def cmd_sanity(args):
    """Bridge ceiling: the baseline session itself as the 'new day' (all its trials), for every selected config."""
    device = torch.device(args.device)
    sel = selected_cfgs(args.stratum)
    world = World(args.stratum, "O", device, load_sessions=True)
    refdata = RefData(world, load_session=world.sessions[world.orig_ids[-1]])
    own = refdata.sd
    idx_all = np.arange(len(own.trials))
    tab = np.asarray(world.orig_tables[-1])
    pred = decode_affine(world, own, tab, np.zeros(K), refdata.mean, refdata.std)
    q = own.valid & (own.trial_id >= 0)
    r2_src, _ = r2_physical(world, own, pred, q)
    out = {"ref": refdata.name, "r2_source_table": float(r2_src), "rows": []}
    for name, (kind, inp, tr, mode, part) in EVAL_METHODS.items():
        if part != "b":
            continue
        cfg = method_cfg(name, sel)
        res = run_method(kind, cfg, refdata, own, idx_all, use_all=True)
        t, c, mean, std, info = res
        pred = decode_affine(world, own, t, c, mean, std, cfg.get("transform", "raw"), pool=inp == "pooled")
        r2, _ = r2_physical(world, own, pred, q)
        out["rows"].append({"method": name, "cfg": cfg, "r2_bridge_ceiling": float(r2), "info": info})
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
