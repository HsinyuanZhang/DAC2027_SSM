#!/usr/bin/env python
"""H4: S1 residuals from the live contiguous forward (L1) vs per-trial zero state (L0).

L0 is ``rebased_feedback_loop`` (unchanged). L1 carries trunk state and the FIR ring
across the whole calibration prefix. L2 carries state but clears the ring at each
calibration-trial start.

Selection of (eta, rho) is redone per variant on the same source episodes (protocol (a),
AND gate). The conservative (>=2 positive sessions) gate is recorded too. A secondary
row runs L1 at the stored L0 pick.

  python scripts/run_h4_live.py select --group mid
  python scripts/run_h4_live.py eval --group mid
  python scripts/run_h4_live.py aggregate

Resumable: a (variant, tag, session, seed, B) or eval row already on disk is skipped.
Local cuda:0 only. Does not push.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PC_OUT"] = str(ROOT / "results/population_calib_p3_20261016/fix")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_h4")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch

from scripts import run_causal_main as rcm
from scripts import run_pc3_refine as pc
from scripts.run_p0solve import score_tables
from ssm_decode import expansion_causal as ec
from ssm_decode import expansion_protocol as ep
from ssm_decode.atomic_jsonl import append_jsonl, read_jsonl
from ssm_decode.causal_protocol import (
    calibration_indices,
    measured_read_bin,
)
from ssm_decode.live_residual import GAP_RING, live_rebased_many
from ssm_decode.population_calib import PopStats, norm_stats

OUT = ROOT / "results/h4_live_20261021"
ROUND2 = ROOT / "results/expansion_round2/x/expansion_out"
LEGACY = ROOT / "results/causal_main_20261019/select"
BS = (1, 4, 8, 16, 33)
SEEDS = (0, 1)
LATENCIES = (0, 1)
ETAS = rcm.ETAS
RHOS = rcm.RHOS
CONFIGS = [(float(e), float(r)) for e in ETAS for r in RHOS]
FROZEN_TAGS = ("days", "month", "year")
ANAL = ROOT / "results/analysis/h4_live_20261021"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a") as f:
        f.write(line + "\n")


def p0_cache(sd, prior, calibs, k_max: int):
    """Prefix P0 and norm only. CCA is not used by S1 and is not computed."""
    spans = [(int(a), int(b)) for a, b in sd.trials]
    norm, tables, read_bin = {}, {s: {} for s in calibs}, {}
    for k in range(1, int(k_max) + 1):
        idx = calibration_indices(sd.bank.n_trials, k)
        rcm.assert_prefix_idx(idx, k)
        read_bin[k] = measured_read_bin(spans, idx)
        stats = PopStats.from_trials(sd.bank, idx)
        mean_t, std_t = norm_stats(stats)
        norm[k] = (rcm.to_cpu(mean_t), rcm.to_cpu(std_t))
        for seed, calib in calibs.items():
            res = calib.calibrate(stats, prior)
            if bool(res.grad_enabled_inside):
                raise AssertionError("P0 calibrate ran with grad enabled")
            tables[seed][k] = rcm.to_cpu(res.table)
    return {"norm": norm, "p0": tables, "cca": {}, "read_bin": read_bin}


def _bundles():
    return ec.load_episodes(ROUND2, LEGACY)


def _targets(group: str) -> list[dict]:
    """Round-2 causal rows, one dict per (bid, date) in the requested trunk-gap tier."""
    rows = []
    root = ROUND2 / "causal_rows"
    for path in sorted(root.glob("*/*.jsonl")):
        for row in read_jsonl(path):
            if row.get("date") == "20140307":
                continue
            if str(row.get("date", "")).startswith("2016"):
                continue
            rows.append(row)
            break  # header identity is repeated; one row is enough to identify the target
    # The break above keeps only the first row. Re-read dates properly.
    seen = {}
    for path in sorted(root.glob("*/*.jsonl")):
        bid = path.parent.name
        date = path.stem
        if date == "20140307" or date.startswith("2016"):
            continue
        sample = None
        for row in read_jsonl(path):
            if row.get("method") == "p0" and int(row.get("B", 0)) == 8:
                sample = row
                break
        if sample is None:
            sample = read_jsonl(path)[0]
        gap = int(sample["trunk_gap_days"])
        if group == "mid" and not (31 <= gap <= 180):
            continue
        if group == "short" and gap > 30:
            continue
        if group == "long" and gap <= 180:
            continue
        if group not in ("mid", "short", "long"):
            raise SystemExit(f"unknown group {group}")
        seen[(bid, date)] = {
            "bid": bid, "date": date, "target": sample["target"],
            "trunk_gap_days": gap, "gap_days": int(sample["gap_days"]),
            "stratum": sample.get("stratum"), "tier_label": sample.get("tier"),
        }
    return [seen[k] for k in sorted(seen)]


def _l_of_frozen(tag: str, B: int) -> int:
    sim = {"days": "oos", "month": "oosc", "year": "oos"}[tag]
    blob = json.loads(pc.selection_file(tag, sim).read_text())
    return int(blob["r2"][str(int(B))]["cfg"]["L"])


def _l_of_expansion(bid: str) -> dict[int, int]:
    blob = json.loads((ROUND2 / "select" / f"{bid}.json").read_text())
    return {int(b): int(blob["b"][b]["r2"]["L"]) for b in blob["b"] if b.isdigit()}


def _source_units(group: str) -> list[dict]:
    """(tag, session, seed) that can vote for some target in this group, with the B list."""
    reported = ec.reported_target_dates()
    bundles = _bundles()["s1"]
    targets = _targets(group)
    want = set()
    for t in targets:
        for B in BS:
            pool = ec.selection_pool(bundles, t["date"], "expanding", reported, B=B)
            for ep_row in pool:
                want.add((str(ep_row["tag"]), str(ep_row["session"]), int(ep_row["seed"])))
    # B list per (tag, session): budgets that exist on the stored episode and that we report.
    stored = defaultdict(set)
    meta = {}
    for ep_row in bundles:
        key = (str(ep_row["tag"]), str(ep_row["session"]), int(ep_row["seed"]))
        if key not in want:
            continue
        if int(ep_row["B"]) not in BS:
            continue
        stored[key].add(int(ep_row["B"]))
        meta[key] = {"eta_r2": float(ep_row.get("eta_r2", 1.0)),
                     "gap_days": ep_row.get("gap_days")}
    units = []
    for key, bs in sorted(stored.items()):
        tag, session, seed = key
        units.append({"tag": tag, "session": session, "seed": seed, "Bs": tuple(sorted(bs)),
                      "eta_r2": meta[key]["eta_r2"], "gap_days": meta[key]["gap_days"]})
    return units


class _Frozen:
    def __init__(self, tag: str):
        self.tag = tag
        self.world = pc.load_world(tag, False)
        self.p0 = pc.load_p0(self.world, tag)
        self.base = self.world.adapter.model.base
        self._jbar = {}
        self.L = {B: _l_of_frozen(tag, B) for B in BS}

    def jbar(self, L: int):
        if L not in self._jbar:
            self._jbar[L] = pc.jbar_for(self.tag, int(L))
        return self._jbar[L]

    def session(self, name: str):
        if name in self.world.sessions:
            return self.world.sessions[name], self.world.priors[name]
        return self.world.target(name)


class _Expansion:
    def __init__(self, bid: str):
        from scripts.run_expansion import Ctx, Store, register
        self.tag = bid
        bbs = [b for b in ep.build_backbones() if b.bid == bid]
        if not bbs:
            raise SystemExit(f"unknown expansion backbone {bid}")
        st = Store(ROUND2, None)
        register(bbs, st)
        self.ctx = Ctx(bbs[0], st, argparse.Namespace(device="cuda:0"), load_sessions=False)
        self.world = self.ctx.world
        self.p0 = self.ctx.p0
        self.base = self.world.adapter.model.base
        self.L = _l_of_expansion(bid)

    def jbar(self, L: int):
        return self.ctx.jbar(int(L))

    def session(self, name: str):
        return self.world.target(name)


def _open_trunk(tag: str):
    if tag in FROZEN_TAGS:
        return _Frozen(tag)
    return _Expansion(tag)


def _score_configs(world, sd, cache, seed, B, tables, biases):
    idx = calibration_indices(sd.bank.n_trials, B)
    mean, std = cache["norm"][B]
    q = sd.query_mask(np.asarray(idx))
    q = q.clone()
    q[rcm.SIM_MAX:] = False
    if int(q.sum()) < 50:
        return None
    return score_tables(world, sd, mean, std, tables, biases, q, rcm.SIM_MAX)


def _live_group(trunk, sd, prior, cache, seed, jbar, B_max, Bs, ring_mode: str,
                latency: int = 0, drop_trial0: bool = False):
    """One live pass at the longest B of this kernel. Snapshots cover every B in ``Bs``."""
    device = sd.x32.device
    prepare = rcm._prepare_from(cache, seed, "p0", 0.0, device)
    spans = [(int(a), int(b)) for a, b in sd.trials]

    def _run(configs):
        try:
            return live_rebased_many(
                trunk.base, sd.x32, sd.y_norm, sd.valid.to(dtype=sd.x32.dtype), spans,
                prior.float(), configs, prepare, jbar, int(B_max), ring_mode=ring_mode,
                latency=int(latency), drop_trial0=bool(drop_trial0))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(configs) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(configs) // 2
            _log(f"OOM at {len(configs)} configs, splitting")
            return _run(configs[:mid]) + _run(configs[mid:])

    many = _run(CONFIGS)
    torch.cuda.empty_cache()
    by_b = {}
    for B in Bs:
        tables, biases = [], []
        for ans in many:
            snap = ans["history"][int(B) - 1]
            tables.append(rcm.to_cpu(snap["table"]))
            biases.append(rcm.to_cpu(snap["bias"]))
        scores = _score_configs(trunk.world, sd, cache, seed, int(B), tables, biases)
        by_b[int(B)] = scores
    return by_b


def _select_path(variant: str, latency: int = 0) -> Path:
    if variant == "l1dev":
        return OUT / f"select_l1dev_lat{int(latency)}.jsonl"
    return OUT / f"select_{variant}.jsonl"


def _done_select(variant: str, latency: int = 0) -> set:
    path = _select_path(variant, latency)
    if not path.exists():
        return set()
    return {(r["tag"], r["session"], int(r["seed"]), int(r["B"])) for r in read_jsonl(path)}


def _parity_l0(trunk, sd, prior, cache, seed, B, jbar, stored_rebase):
    """One stored (eta, rho) through the unchanged L0 loop. Returns (eta, rho, stored, fresh)."""
    hit = stored_rebase[len(stored_rebase) // 2]
    eta, rho = float(hit["eta"]), float(hit["rho"])
    device = sd.x32.device
    prepare = rcm._prepare_from(cache, seed, "p0", 0.0, device)
    idx = calibration_indices(sd.bank.n_trials, B)
    raw = [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype), sd.valid[a:b].to(dtype=sd.x32.dtype))
           for a, b in (sd.trials[int(i)] for i in idx)]
    from ssm_decode.causal_protocol import run_rebased_many
    ans = run_rebased_many(trunk.base, raw, prior.float(), [(eta, rho)], prepare, jbar=jbar)[0]
    scores = _score_configs(trunk.world, sd, cache, seed, B,
                            [rcm.to_cpu(ans["table"])], [rcm.to_cpu(ans["bias"])])
    fresh = None if scores is None else float(scores[0])
    return eta, rho, float(hit["r2"]), fresh


def cmd_select(args):
    if not torch.cuda.is_available():
        raise SystemExit("cuda:0 is not available")
    units = _source_units(args.group)
    by_tag = defaultdict(list)
    for u in units:
        by_tag[u["tag"]].append(u)
    _log(f"select {args.group}: {len(units)} (tag, session, seed) across {sorted(by_tag)}")
    # Rough cost, replaced by a measured per-pass time on the first unit.
    _log(f"GPU-time estimate before measurement: {len(units)} units x 2 variants. "
         f"A unit is one live grid pass per kernel length plus query scores. "
         f"Printed again after the first pass.")
    # l1dev is simulated at both install latencies; the table depends on latency.
    if args.variant == "both":
        jobs = (("l1", 0), ("l2", 0))
    elif args.variant == "l1dev":
        jobs = (("l1dev", 0), ("l1dev", 1))
    else:
        jobs = ((args.variant, 0),)
    measured = []
    for tag, group in by_tag.items():
        _log(f"load trunk {tag} ({len(group)} units)")
        t_load = time.perf_counter()
        trunk = _open_trunk(tag)
        _log(f"trunk {tag} ready in {time.perf_counter() - t_load:.0f}s  L={ {b: trunk.L.get(b) for b in BS} }")
        # Sessions may repeat across seeds. Cache the recording per session.
        loaded = {}
        for unit in group:
            if unit["session"] not in loaded:
                _log(f"session {unit['session']}")
                sd, prior = trunk.session(unit["session"])
                seeds_here = sorted({u["seed"] for u in group if u["session"] == unit["session"]})
                k_max = max(b for u in group if u["session"] == unit["session"]
                            for b in u["Bs"] if b <= sd.bank.n_trials)
                calibs = {s: trunk.p0[s] for s in seeds_here}
                cache = p0_cache(sd, prior, calibs, k_max)
                loaded[unit["session"]] = (sd, prior, cache, k_max)
            sd, prior, cache, k_max = loaded[unit["session"]]
            Bs = tuple(b for b in unit["Bs"] if b <= sd.bank.n_trials and b <= k_max)
            if not Bs:
                continue
            # L0 parity once per trunk, on the first unit.
            if tag not in getattr(cmd_select, "_parity_done", set()):
                stored = _stored_rebase(unit["tag"], unit["session"], unit["seed"], Bs[-1])
                if stored is not None:
                    jb = trunk.jbar(trunk.L[Bs[-1]])
                    eta, rho, old, fresh = _parity_l0(trunk, sd, prior, cache, unit["seed"], Bs[-1], jb, stored)
                    diff = None if fresh is None else abs(fresh - old)
                    _log(f"L0 parity sample {tag} {unit['session'][-8:]} B={Bs[-1]} "
                         f"eta={eta} rho={rho} stored={old:.4f} fresh={fresh} |diff|={diff}")
                    if diff is not None and diff > 0.05:
                        raise SystemExit(f"L0 parity sample off by {diff:.3f}; stopping")
                cmd_select._parity_done = getattr(cmd_select, "_parity_done", set()) | {tag}
            by_L = defaultdict(list)
            for B in Bs:
                by_L[int(trunk.L[B])].append(int(B))
            for variant, lat in jobs:
                done = _done_select(variant, lat)
                pending_L = {}
                for L, lbs in by_L.items():
                    need = [b for b in lbs if (tag, unit["session"], unit["seed"], b) not in done]
                    if need:
                        pending_L[L] = need
                if not pending_L:
                    continue
                ring = "reset" if variant == "l2" else "carry"
                drop0 = variant == "l1dev"
                rows = []
                t1 = time.perf_counter()
                for L, lbs in pending_L.items():
                    jb = trunk.jbar(L)
                    scored = _live_group(trunk, sd, prior, cache, unit["seed"], jb, max(lbs), lbs, ring,
                                         latency=lat, drop_trial0=drop0)
                    for B, scores in scored.items():
                        if scores is None:
                            _log(f"skip {variant} {tag} {unit['session'][-8:]} B={B} query<50")
                            continue
                        # p0 reference under the same scorer (eta=rho=0 must match scores of that config).
                        w0 = cache["p0"][unit["seed"]][B]
                        zeros = np.zeros(w0.shape[0])
                        p0_scores = _score_configs(trunk.world, sd, cache, unit["seed"], B, [w0], [zeros])
                        p0_r2 = None if p0_scores is None else float(p0_scores[0])
                        rebase = [{"eta": float(e), "rho": float(r), "r2": float(sc)}
                                  for (e, r), sc in zip(CONFIGS, scores)]
                        zero = next((s["r2"] for s in rebase if s["eta"] == 0.0 and s["rho"] == 0.0), None)
                        if zero is not None and p0_r2 is not None and abs(zero - p0_r2) > 1e-3:
                            _log(f"WARN {variant} eta=0 {zero:.4f} != p0 {p0_r2:.4f} "
                                 f"{unit['session'][-8:]} B={B}")
                        rows.append({
                            "tag": tag, "session": unit["session"], "seed": int(unit["seed"]),
                            "B": int(B), "protocol": "causal_first_B", "variant": variant,
                            "ring_mode": ring, "p0": p0_r2, "base": p0_r2, "alpha": 0.0,
                            "rebase": rebase, "eta_r2": float(unit["eta_r2"]),
                            "gap_days": unit["gap_days"], "L": int(L),
                            "n_bins": int(sd.n_bins), "n_trials": int(sd.bank.n_trials),
                            "latency": int(lat), "drop_trial0": bool(drop0),
                        })
                if rows:
                    append_jsonl(_select_path(variant, lat), rows)
                    dt = time.perf_counter() - t1
                    measured.append(dt)
                    _log(f"select {variant} lat{lat} {tag} {unit['session'][-8:]} s{unit['seed']} "
                         f"B={sorted(b for b in Bs)} {dt:.1f}s")
                    if len(measured) == 1:
                        per = measured[0]
                        _log(f"GPU-time estimate: first select pass {per:.0f}s. "
                             f"Upper bound {len(units)} units x {len(jobs)} passes "
                             f"x {per:.0f}s = {len(units) * len(jobs) * per / 3600:.2f} h "
                             f"(passes already on disk are skipped; kernel groups share a pass).")
        del loaded, trunk
        torch.cuda.empty_cache()
    _log(f"select {args.group} done")


def _stored_rebase(tag, session, seed, B):
    for ep_row in _bundles()["s1"]:
        if (str(ep_row["tag"]) == tag and str(ep_row["session"]) == session
                and int(ep_row["seed"]) == seed and int(ep_row["B"]) == B and ep_row.get("rebase")):
            return ep_row["rebase"]
    return None


def _eta_r2_target(bid: str, B: int) -> float:
    """Tie-break eta is the target backbone's own R2 eta, matching causal-eval."""
    blob = json.loads((ROUND2 / "select" / f"{bid}.json").read_text())
    return float(blob["b"][str(int(B))]["r2"]["eta"])


def _picks(bid: str, date: str, B: int):
    """AND and conservative picks for L1 and L2. L0 pick is the stored causal-row cfg."""
    reported = ec.reported_target_dates()
    eta_r2 = _eta_r2_target(bid, B)
    out = {}
    for variant in ("l1", "l2"):
        path = _select_path(variant)
        episodes = read_jsonl(path) if path.exists() else []
        pool = ec.selection_pool(episodes, date, "expanding", reported, B=B)
        if not pool:
            raise RuntimeError(f"no {variant} selection episodes for {bid} {date} B={B}")
        out[variant] = ec.gate_choices(pool, eta_r2, "p0")
        out[variant]["eta_r2"] = eta_r2
        out[variant]["n_pool"] = len(pool)
    return out


def _maps_for(base, sd, prior, cache, seed, spans, B, eta, rho, jbar, ring_mode,
              latency: int = 0, drop_trial0: bool = False):
    cold, initial = rcm.cold_and_initial(rcm.to_cpu(prior), cache, spans)
    if float(eta) == 0.0 and float(rho) == 0.0:
        maps, reads = rcm.p0_commits(cache, seed, spans, B, cold, initial)
        return maps, reads, {"eta": 0.0, "rho": 0.0, "fallback_base": True, "n_commits": 0,
                             "ring_mode": ring_mode, "latency": int(latency),
                             "drop_trial0": bool(drop_trial0)}
    if ring_mode == "l0":
        body, reads, info = rcm.loop_commits(
            base, sd, prior, cache, seed, spans, B, eta=float(eta), rho=float(rho),
            jbar=jbar, alpha=0.0, kind="p0")
        maps, reads = rcm.install_maps(body, cold, initial, B)
        return maps, reads, {**info, "ring_mode": "l0"}
    mode = "reset" if ring_mode == "l2" else "carry"
    device = sd.x32.device
    prepare = rcm._prepare_from(cache, seed, "p0", 0.0, device)
    ans = live_rebased_many(
        base, sd.x32, sd.y_norm, sd.valid.to(dtype=sd.x32.dtype), spans, prior.float(),
        [(float(eta), float(rho))], prepare, jbar, int(B), ring_mode=mode,
        latency=int(latency), drop_trial0=bool(drop_trial0))[0]
    hist = ans["history"]
    if len(hist) != int(B):
        raise AssertionError(f"expected {B} commits, got {len(hist)}")
    body = {}
    for item in hist:
        k = int(item["k"]) + 1
        body[k] = rcm.np_map(rcm.to_cpu(item["table"]), rcm.to_cpu(item["bias"]),
                             rcm.to_cpu(item["mean"]), rcm.to_cpu(item["std"]), rcm._end(cache, k))
    maps, reads = rcm.install_maps(body, cold, initial, B)
    return maps, reads, {"eta": float(eta), "rho": float(rho), "fallback_base": False,
                         "n_commits": int(B), "ring_mode": ring_mode, "latency": int(latency),
                         "drop_trial0": bool(drop_trial0)}


def _stored_l0_pick(bid: str, date: str, B: int, seed: int):
    path = ROUND2 / "causal_rows" / bid / f"{date}.jsonl"
    for row in read_jsonl(path):
        if (row.get("method") == "s1_causal" and int(row["B"]) == int(B)
                and int(row["seed"]) == int(seed) and int(row["latency"]) == 0):
            cfg = row.get("cfg") or {}
            return {"eta": float(cfg.get("eta", 0.0)), "rho": float(cfg.get("rho", 0.0)),
                    "fallback_base": bool(cfg.get("fallback_base", False)),
                    "stored_r2": row.get("r2"), "source_gain": cfg.get("source_gain")}
    return None


def _stored_p0(bid, date, B, seed, latency):
    path = ROUND2 / "causal_rows" / bid / f"{date}.jsonl"
    for row in read_jsonl(path):
        if (row.get("method") == "p0" and int(row["B"]) == int(B) and int(row["seed"]) == int(seed)
                and int(row["latency"]) == int(latency)):
            return row.get("r2")
    return None


def _eval_key(row):
    return (row["method"], row["bid"], row["date"], int(row["B"]), int(row["seed"]), int(row["latency"]))


def cmd_eval(args):
    if not torch.cuda.is_available():
        raise SystemExit("cuda:0 is not available")
    targets = _targets(args.group)
    path = OUT / f"eval_{args.group}.jsonl"
    done = {_eval_key(r) for r in read_jsonl(path)} if path.exists() else set()
    by_bid = defaultdict(list)
    for t in targets:
        by_bid[t["bid"]].append(t)
    n_cells = len(targets) * len(BS) * len(SEEDS) * len(LATENCIES)
    _log(f"eval {args.group}: {len(targets)} targets, up to {n_cells} latency cells. "
         f"GPU-time estimate: causal-eval was ~1-2 min/target for every frozen method; "
         f"H4 is P0+L0+L1+L2 only, about 1 min/target if picks alias, "
         f"~{len(targets)}-{2 * len(targets)} min for this group before pick-dependent repeats.")
    for bid, group in by_bid.items():
        _log(f"eval trunk {bid} targets={[t['date'] for t in group]}")
        trunk = _open_trunk(bid)
        # Kernel length is the backbone's R2 L, same as causal-eval, not the source trunk's.
        for t in group:
            sd, prior = trunk.session(t["target"])
            spans = [(int(a), int(b)) for a, b in sd.trials]
            Bs = tuple(b for b in BS if b <= len(spans))
            k_max = max(Bs)
            cache = p0_cache(sd, prior, {s: trunk.p0[s] for s in SEEDS}, k_max)
            behavior = sd.record.behavior
            label_stats = trunk.world.label_stats
            dev = sd.x32.device
            for B in Bs:
                picks = _picks(bid, t["date"], B)
                jb = trunk.jbar(int(trunk.L[B]))
                for seed in SEEDS:
                    need_lat = [lat for lat in LATENCIES
                                if any((m, bid, t["date"], B, seed, lat) not in done
                                       for m in ("p0", "s1_l0", "s1_l1", "s1_l2",
                                                 "s1_l1_transfer", "s1_l1_cons", "s1_l2_cons"))]
                    if not need_lat:
                        continue
                    l0_pick = _stored_l0_pick(bid, t["date"], B, seed)
                    if l0_pick is None:
                        _log(f"no stored L0 pick {bid} {t['date']} B={B} s{seed}")
                        continue
                    specs = {}
                    specs["p0"] = (0.0, 0.0, "l0")  # short-circuits to p0_commits
                    specs["s1_l0"] = (l0_pick["eta"], l0_pick["rho"], "l0")
                    for variant, method in (("l1", "s1_l1"), ("l2", "s1_l2")):
                        choice = (picks[variant].get("and") or {})
                        specs[method] = (float(choice.get("eta", 0.0)), float(choice.get("rho", 0.0)), variant)
                        cons = picks[variant].get("conservative") or {}
                        specs[method + "_cons"] = (float(cons.get("eta", 0.0)), float(cons.get("rho", 0.0)), variant)
                    specs["s1_l1_transfer"] = (l0_pick["eta"], l0_pick["rho"], "l1")
                    # Build each distinct (eta, rho, ring) once.
                    built = {}
                    t1 = time.perf_counter()
                    for method, (eta, rho, ring) in specs.items():
                        key = (float(eta), float(rho), ring)
                        if key not in built:
                            built[key] = _maps_for(trunk.base, sd, prior, cache, seed, spans, B,
                                                   eta, rho, jb, ring)
                    jobs, meta = [], []
                    q = sd.query_mask(calibration_indices(sd.bank.n_trials, B)).detach().cpu().numpy()
                    for method, (eta, rho, ring) in specs.items():
                        maps, reads, info = built[(float(eta), float(rho), ring)]
                        for lat in need_lat:
                            if (method, bid, t["date"], B, seed, lat) in done:
                                continue
                            segs = rcm.segments_for(spans, B, sd.n_bins, lat, "anytime")
                            rcm.assert_segment_causal(segs, spans, reads, staged_trial0=True)
                            packed = {k: rcm._pack(maps[k], dev) for _a, _b, k in segs}
                            jobs.append({"segments": segs, "packed": packed, "qmask": q, "want_h": False})
                            meta.append((method, lat, info))
                    scores = rcm.decode_many(trunk.base, sd.x32, jobs, label_stats, behavior) if jobs else []
                    rows = []
                    for (method, lat, info), r2 in zip(meta, scores):
                        stored_p0 = _stored_p0(bid, t["date"], B, seed, lat)
                        stored_l0 = l0_pick["stored_r2"] if lat == 0 else None
                        if lat != 0:
                            # stored L0 at this latency, for the parity column
                            for row in read_jsonl(ROUND2 / "causal_rows" / bid / f"{t['date']}.jsonl"):
                                if (row.get("method") == "s1_causal" and int(row["B"]) == int(B)
                                        and int(row["seed"]) == int(seed) and int(row["latency"]) == int(lat)):
                                    stored_l0 = row.get("r2")
                        choice = None
                        if method.startswith("s1_l1") and "transfer" not in method:
                            which = "conservative" if method.endswith("cons") else "and"
                            choice = (picks["l1"].get(which) or {})
                        elif method.startswith("s1_l2"):
                            which = "conservative" if method.endswith("cons") else "and"
                            choice = (picks["l2"].get(which) or {})
                        rows.append({
                            "method": method, "bid": bid, "target": t["target"], "date": t["date"],
                            "B": int(B), "seed": int(seed), "latency": int(lat), "r2": float(r2),
                            "trunk_gap_days": int(t["trunk_gap_days"]), "gap_days": int(t["gap_days"]),
                            "stratum": t["stratum"], "eta": info.get("eta"), "rho": info.get("rho"),
                            "fallback_base": info.get("fallback_base"), "ring_mode": info.get("ring_mode"),
                            "stored_p0": stored_p0, "stored_l0": stored_l0,
                            "l0_eta": l0_pick["eta"], "l0_rho": l0_pick["rho"],
                            "pick_source_gain": None if not choice else choice.get("source_gain"),
                            "pick_fallback": None if not choice else choice.get("fallback_base"),
                            "n_bins": int(sd.n_bins), "group": args.group,
                        })
                    if rows:
                        append_jsonl(path, rows)
                        done.update(_eval_key(r) for r in rows)
                        # Parity line for L0.
                        for r in rows:
                            if r["method"] == "s1_l0" and r["stored_l0"] is not None:
                                diff = abs(float(r["r2"]) - float(r["stored_l0"]))
                                if diff > 1e-3:
                                    _log(f"L0 |diff|={diff:.4f} {bid} {t['date']} B={B} "
                                         f"s{seed} lat={r['latency']}")
                        _log(f"eval {bid} {t['date']} B={B} s{seed} n={len(rows)} "
                             f"{time.perf_counter() - t1:.1f}s")
            del sd, prior, cache
            torch.cuda.empty_cache()
        del trunk
        torch.cuda.empty_cache()
    _log(f"eval {args.group} done")


def _boot(x, rng, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan, int(x[0] > 0)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), int((x > 0).sum())


def cmd_aggregate(args):
    """Session-clustered means. Writes summary csv. README prose is written by the caller."""
    ANAL.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(OUT.glob("eval_*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        _log("aggregate: no eval rows")
        return
    # mean over seeds, then over backbones of the same target session
    cell = defaultdict(list)
    for r in rows:
        if r.get("r2") is None:
            continue
        key = (r["group"], r["method"], r["target"], int(r["B"]), int(r["latency"]), r["bid"])
        cell[key].append(float(r["r2"]))
    bid_mean = []
    for key, vals in cell.items():
        group, method, target, B, lat, bid = key
        gap = next(int(r["trunk_gap_days"]) for r in rows
                   if r["target"] == target and r["bid"] == bid and r["group"] == group)
        bid_mean.append({"group": group, "method": method, "target": target, "bid": bid,
                         "B": B, "latency": lat, "r2": float(np.mean(vals)), "trunk_gap_days": gap})
    sess = defaultdict(list)
    for r in bid_mean:
        sess[(r["group"], r["method"], r["target"], r["B"], r["latency"])].append(r["r2"])
    flat = []
    for key, vals in sess.items():
        group, method, target, B, lat = key
        flat.append({"group": group, "method": method, "target": target, "B": B,
                     "latency": lat, "r2": float(np.mean(vals))})
    rng = np.random.default_rng(0)
    methods = ("p0", "s1_l0", "s1_l1", "s1_l2", "s1_l1_transfer", "s1_l1_cons", "s1_l2_cons",
               "s1_l1_dev", "s1_l1_dev_transfer", "s1_l1_dev_cons")
    summary = []
    for group in sorted({r["group"] for r in flat}):
        for lat in (0, 1):
            for B in list(BS) + ["B4-33"]:
                for method in methods:
                    xs = []
                    sessions = []
                    if B == "B4-33":
                        by_t = defaultdict(list)
                        for r in flat:
                            if r["group"] == group and r["latency"] == lat and r["method"] == method and r["B"] in (4, 8, 16, 33):
                                by_t[r["target"]].append(r["r2"])
                        for t, vals in sorted(by_t.items()):
                            if len(vals) == 4:
                                xs.append(float(np.mean(vals)))
                                sessions.append(t)
                    else:
                        for r in flat:
                            if r["group"] == group and r["latency"] == lat and r["method"] == method and r["B"] == B:
                                xs.append(r["r2"])
                                sessions.append(r["target"])
                    mean, lo, hi, npos = _boot(xs, rng)
                    summary.append({"group": group, "latency": lat, "B": B, "method": method,
                                    "n": len(xs), "mean": mean, "ci_lo": lo, "ci_hi": hi,
                                    "n_pos_vs_zero": npos})
            # paired
            for base_m, other in (("s1_l0", "s1_l1"), ("p0", "s1_l1"), ("p0", "s1_l0"),
                                  ("p0", "s1_l2"), ("s1_l0", "s1_l2"), ("s1_l0", "s1_l1_transfer"),
                                  ("p0", "s1_l1_cons"), ("p0", "s1_l1_dev"), ("s1_l0", "s1_l1_dev"),
                                  ("p0", "s1_l1_dev_transfer"), ("s1_l0", "s1_l1_dev_transfer"),
                                  ("p0", "s1_l1_dev_cons")):
                for B in list(BS) + ["B4-33"]:
                    def _vec(method, B=B, group=group, lat=lat):
                        out = {}
                        if B == "B4-33":
                            by_t = defaultdict(list)
                            for r in flat:
                                if (r["group"] == group and r["latency"] == lat and r["method"] == method
                                        and r["B"] in (4, 8, 16, 33)):
                                    by_t[r["target"]].append(r["r2"])
                            for t, vals in by_t.items():
                                if len(vals) == 4:
                                    out[t] = float(np.mean(vals))
                        else:
                            for r in flat:
                                if r["group"] == group and r["latency"] == lat and r["method"] == method and r["B"] == B:
                                    out[r["target"]] = r["r2"]
                        return out
                    a, b = _vec(other), _vec(base_m)
                    common = sorted(set(a) & set(b))
                    diff = [a[t] - b[t] for t in common]
                    mean, lo, hi, npos = _boot(diff, rng)
                    summary.append({"group": group, "latency": lat, "B": B,
                                    "method": f"{other}-minus-{base_m}", "n": len(diff),
                                    "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pos_vs_zero": npos})
    # pick changes at latency 0, seed-averaged identity (picks do not depend on seed)
    picks = {}
    for r in rows:
        if int(r["latency"]) != 0 or int(r["seed"]) != 0:
            continue
        if r["method"] not in ("s1_l0", "s1_l1", "s1_l2"):
            continue
        picks.setdefault((r["group"], r["bid"], r["date"], int(r["B"])), {})[r["method"]] = (
            r.get("eta"), r.get("rho"), r.get("fallback_base"))
    n_change = defaultdict(int)
    n_pick = defaultdict(int)
    for key, block in picks.items():
        group, _bid, _date, B = key
        if "s1_l0" in block and "s1_l1" in block:
            n_pick[(group, B)] += 1
            if block["s1_l0"][:2] != block["s1_l1"][:2]:
                n_change[(group, B)] += 1
    dest = ANAL / "summary.csv"
    header = ["group", "latency", "B", "method", "n", "mean", "ci_lo", "ci_hi", "n_pos"]
    with dest.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in summary:
            vals = [r["group"], str(r["latency"]), str(r["B"]), r["method"], str(r["n"]),
                    "" if r["mean"] != r["mean"] else f"{r['mean']:.6f}",
                    "" if r["ci_lo"] != r["ci_lo"] else f"{r['ci_lo']:.6f}",
                    "" if r["ci_hi"] != r["ci_hi"] else f"{r['ci_hi']:.6f}",
                    str(r["n_pos_vs_zero"])]
            f.write(",".join(vals) + "\n")
    # compact per-target csv
    per = ANAL / "per_target.csv"
    with per.open("w") as f:
        f.write("group,bid,date,target,B,seed,latency,method,r2,eta,rho,fallback,stored_p0,stored_l0,trunk_gap_days\n")
        for r in rows:
            def fmt(v):
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.6f}"
                return str(v)
            f.write(",".join([
                str(r.get("group", "")), r["bid"], r["date"], r["target"], str(r["B"]), str(r["seed"]),
                str(r["latency"]), r["method"], fmt(r.get("r2")), fmt(r.get("eta")), fmt(r.get("rho")),
                fmt(r.get("fallback_base")), fmt(r.get("stored_p0")), fmt(r.get("stored_l0")),
                str(r.get("trunk_gap_days", "")),
            ]) + "\n")
    pick_path = ANAL / "pick_changes.csv"
    with pick_path.open("w") as f:
        f.write("group,bid,date,B,l0_eta,l0_rho,l1_eta,l1_rho,l2_eta,l2_rho,l1_changed\n")
        for key, block in sorted(picks.items()):
            group, bid, date, B = key
            def pair(m):
                if m not in block:
                    return "", ""
                return block[m][0], block[m][1]
            e0, r0 = pair("s1_l0")
            e1, r1 = pair("s1_l1")
            e2, r2 = pair("s1_l2")
            changed = int((e0, r0) != (e1, r1)) if e0 != "" and e1 != "" else ""
            f.write(f"{group},{bid},{date},{B},{e0},{r0},{e1},{r1},{e2},{r2},{changed}\n")
    _write_cluster_tables(bid_mean)
    _log(f"aggregate wrote {dest} pick changes {dict(n_change)} / {dict(n_pick)}")
    _log("GAP_RING " + GAP_RING)


def _mean_over(records, method, lat, bs, key):
    """Mean of ``method`` at one latency. ``key`` is target (session cluster) or bid (backbone)."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["method"] != method or int(r["latency"]) != int(lat) or int(r["B"]) not in bs:
            continue
        buckets[r[key]][int(r["B"])].append(float(r["r2"]))
    out = {}
    for k, byb in buckets.items():
        if any(b not in byb for b in bs):
            continue
        out[k] = float(np.mean([np.mean(byb[b]) for b in bs]))
    return out


def _write_cluster_tables(bid_mean):
    """Session-clustered and backbone-clustered paired diffs, plus subject and per-backbone rows."""
    rng = np.random.default_rng(0)
    methods = ("p0", "s1_l0", "s1_l1", "s1_l1_dev", "s1_l1_dev_transfer")
    pairs = (("p0", "s1_l0"), ("p0", "s1_l1"), ("p0", "s1_l1_dev"), ("s1_l0", "s1_l1_dev"),
             ("p0", "s1_l1_dev_transfer"), ("s1_l0", "s1_l1_dev_transfer"), ("s1_l1", "s1_l1_dev"))
    subsets = []
    groups = sorted({r["group"] for r in bid_mean})
    for g in groups:
        subsets.append((g, [r for r in bid_mean if r["group"] == g]))
    mid = [r for r in bid_mean if r["group"] == "mid"]
    if mid:
        subsets.append(("mid_subM", [r for r in mid if str(r["target"]).startswith("sub-M")]))
        subsets.append(("mid_subC", [r for r in mid if str(r["target"]).startswith("sub-C")]))
        for bid in sorted({r["bid"] for r in mid}):
            subsets.append((f"mid:{bid}", [r for r in mid if r["bid"] == bid]))
    rows = []
    for name, recs in subsets:
        if not recs:
            continue
        for cluster, key in (("session", "target"), ("backbone", "bid")):
            for lat in (0, 1):
                for label, bs in (("B4-33", (4, 8, 16, 33)),) + tuple((str(b), (b,)) for b in BS):
                    for method in methods:
                        vals = list(_mean_over(recs, method, lat, bs, key).values())
                        mean, lo, hi, npos = _boot(vals, rng)
                        rows.append({"group": name, "cluster": cluster, "latency": lat, "B": label,
                                     "method": method, "n": len(vals), "mean": mean, "ci_lo": lo,
                                     "ci_hi": hi, "n_pos": npos})
                    for base_m, other in pairs:
                        a = _mean_over(recs, other, lat, bs, key)
                        b = _mean_over(recs, base_m, lat, bs, key)
                        common = sorted(set(a) & set(b))
                        diff = [a[k] - b[k] for k in common]
                        mean, lo, hi, npos = _boot(diff, rng)
                        rows.append({"group": name, "cluster": cluster, "latency": lat, "B": label,
                                     "method": f"{other}-minus-{base_m}", "n": len(diff),
                                     "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pos": npos})
    dest = ANAL / "summary_clusters.csv"
    with dest.open("w") as f:
        f.write("group,cluster,latency,B,method,n,mean,ci_lo,ci_hi,n_pos\n")
        for r in rows:
            def fmt(v):
                return "" if v != v else f"{v:.6f}"
            f.write(",".join([
                r["group"], r["cluster"], str(r["latency"]), str(r["B"]), r["method"], str(r["n"]),
                fmt(r["mean"]), fmt(r["ci_lo"]), fmt(r["ci_hi"]), str(r["n_pos"]),
            ]) + "\n")
    _log(f"clusters wrote {dest} ({len(rows)} rows)")


DRIFT_DATES = ("20140626", "20140627", "20140929", "20141203", "20150511", "20150512",
                "20150610", "20150611", "20150612", "20150615", "20150616", "20150617")


_FROZEN_ROWS = None


def _stored_frozen(stratum, date, B, seed, latency, method):
    global _FROZEN_ROWS
    if _FROZEN_ROWS is None:
        _FROZEN_ROWS = {}
        for row in read_jsonl(ROOT / "results/causal_main_20261019/rows.jsonl"):
            if row.get("method") not in ("p0", "s1_causal_and"):
                continue
            key = (row.get("stratum"), str(row.get("date")), int(row["B"]), int(row["seed"]),
                   int(row["latency"]), row.get("method"))
            _FROZEN_ROWS[key] = row
    return _FROZEN_ROWS.get((stratum, str(date), int(B), int(seed), int(latency), method))


def _picks_frozen(tag: str, date: str, B: int):
    """Protocol (a) AND / conservative, tie-break = that frozen trunk's R2 eta."""
    reported = ec.reported_target_dates()
    sim = {"days": "oos", "month": "oosc", "year": "oos"}[tag]
    blob = json.loads(pc.selection_file(tag, sim).read_text())
    eta_r2 = float(blob["r2"][str(int(B))]["cfg"]["eta"])
    out = {}
    for variant in ("l1", "l2"):
        episodes = read_jsonl(_select_path(variant))
        pool = ec.selection_pool(episodes, date, "expanding", reported, B=B)
        if not pool:
            out[variant] = {"and": {"eta": 0.0, "rho": 0.0, "fallback_base": True, "source_gain": 0.0,
                                    "empty_pool": True},
                            "conservative": {"eta": 0.0, "rho": 0.0, "fallback_base": True},
                            "eta_r2": eta_r2, "n_pool": 0}
            continue
        choice = ec.gate_choices(pool, eta_r2, "p0")
        choice["eta_r2"] = eta_r2
        choice["n_pool"] = len(pool)
        out[variant] = choice
    return out


def cmd_eval_frozen(args):
    """Frozen days trunk (2 sessions) and month trunk on the 12 drift dates.

    L0 uses the stored causal-main AND pick (``s1_causal_and``). L1/L2 re-pick with
    protocol (a) on the live-residual episodes.
    """
    if not torch.cuda.is_available():
        raise SystemExit("cuda:0 is not available")
    jobs = []
    if args.which in ("days", "both"):
        jobs.append(("days", "days", ("20150616", "20150617")))
    if args.which in ("drift", "both"):
        jobs.append(("drift", "month", DRIFT_DATES))
    path = OUT / "eval_frozen.jsonl"
    done = {_eval_key(r) for r in read_jsonl(path)} if path.exists() else set()
    _log(f"eval frozen {args.which}: " + ", ".join(f"{s}:{len(ds)}" for s, _t, ds in jobs))
    for stratum, tag, dates in jobs:
        trunk = _open_trunk(tag)
        for date in dates:
            name = "sub-M_ses-CO-" + date
            sd, prior = trunk.session(name)
            spans = [(int(a), int(b)) for a, b in sd.trials]
            Bs = tuple(b for b in BS if b <= len(spans))
            cache = p0_cache(sd, prior, {s: trunk.p0[s] for s in SEEDS}, max(Bs))
            behavior = sd.record.behavior
            label_stats = trunk.world.label_stats
            dev = sd.x32.device
            from ssm_decode.causal_protocol import TRUNK_LAST
            import datetime as dt
            last = dt.datetime.strptime(TRUNK_LAST[tag], "%Y%m%d").date()
            cur = dt.datetime.strptime(date, "%Y%m%d").date()
            trunk_gap = (cur - last).days
            for B in Bs:
                picks = _picks_frozen(tag, date, B)
                jb = trunk.jbar(int(trunk.L[B]))
                for seed in SEEDS:
                    need_lat = [lat for lat in LATENCIES
                                if any((m, tag, date, B, seed, lat) not in done
                                       for m in ("p0", "s1_l0", "s1_l1", "s1_l2",
                                                 "s1_l1_transfer", "s1_l1_cons", "s1_l2_cons"))]
                    if not need_lat:
                        continue
                    stored = _stored_frozen(stratum, date, B, seed, 0, "s1_causal_and")
                    if stored is None:
                        _log(f"no stored AND row {stratum} {date} B={B}")
                        continue
                    cfg = stored.get("cfg") or {}
                    l0_eta, l0_rho = float(cfg.get("eta", 0.0)), float(cfg.get("rho", 0.0))
                    specs = {
                        "p0": (0.0, 0.0, "l0"),
                        "s1_l0": (l0_eta, l0_rho, "l0"),
                    }
                    for variant, method in (("l1", "s1_l1"), ("l2", "s1_l2")):
                        choice = picks[variant].get("and") or {}
                        specs[method] = (float(choice.get("eta", 0.0)), float(choice.get("rho", 0.0)), variant)
                        cons = picks[variant].get("conservative") or {}
                        specs[method + "_cons"] = (float(cons.get("eta", 0.0)),
                                                   float(cons.get("rho", 0.0)), variant)
                    specs["s1_l1_transfer"] = (l0_eta, l0_rho, "l1")
                    built = {}
                    t1 = time.perf_counter()
                    for method, (eta, rho, ring) in specs.items():
                        key = (float(eta), float(rho), ring)
                        if key not in built:
                            built[key] = _maps_for(trunk.base, sd, prior, cache, seed, spans, B,
                                                   eta, rho, jb, ring)
                    q = sd.query_mask(calibration_indices(sd.bank.n_trials, B)).detach().cpu().numpy()
                    jobs_dec, meta = [], []
                    for method, (eta, rho, ring) in specs.items():
                        maps, reads, info = built[(float(eta), float(rho), ring)]
                        for lat in need_lat:
                            if (method, tag, date, B, seed, lat) in done:
                                continue
                            segs = rcm.segments_for(spans, B, sd.n_bins, lat, "anytime")
                            rcm.assert_segment_causal(segs, spans, reads, staged_trial0=True)
                            packed = {k: rcm._pack(maps[k], dev) for _a, _b, k in segs}
                            jobs_dec.append({"segments": segs, "packed": packed, "qmask": q, "want_h": False})
                            meta.append((method, lat, info))
                    scores = rcm.decode_many(trunk.base, sd.x32, jobs_dec, label_stats, behavior) if jobs_dec else []
                    rows = []
                    for (method, lat, info), r2 in zip(meta, scores):
                        st_p0 = _stored_frozen(stratum, date, B, seed, lat, "p0")
                        st_l0 = _stored_frozen(stratum, date, B, seed, lat, "s1_causal_and")
                        rows.append({
                            "method": method, "bid": tag, "target": name, "date": date,
                            "B": int(B), "seed": int(seed), "latency": int(lat), "r2": float(r2),
                            "trunk_gap_days": int(trunk_gap), "gap_days": int(stored.get("gap_days") or trunk_gap),
                            "stratum": stratum, "eta": info.get("eta"), "rho": info.get("rho"),
                            "fallback_base": info.get("fallback_base"), "ring_mode": info.get("ring_mode"),
                            "stored_p0": None if st_p0 is None else st_p0.get("r2"),
                            "stored_l0": None if st_l0 is None else st_l0.get("r2"),
                            "l0_eta": l0_eta, "l0_rho": l0_rho,
                            "n_pool": picks["l1"].get("n_pool"), "group": stratum,
                        })
                    if rows:
                        append_jsonl(path, rows)
                        done.update(_eval_key(r) for r in rows)
                        for r in rows:
                            if r["method"] == "s1_l0" and r["stored_l0"] is not None:
                                diff = abs(float(r["r2"]) - float(r["stored_l0"]))
                                if diff > 1e-3:
                                    _log(f"L0 |diff|={diff:.4f} {stratum} {date} B={B} s{seed} lat={r['latency']}")
                        _log(f"frozen {stratum} {date} B={B} s{seed} pool={picks['l1'].get('n_pool')} "
                             f"{time.perf_counter() - t1:.1f}s")
            del sd, prior, cache
            torch.cuda.empty_cache()
        del trunk
        torch.cuda.empty_cache()
    _log("eval frozen done")


def _choice_eta(choice):
    if not choice:
        return 0.0, 0.0
    return float(choice.get("eta", 0.0)), float(choice.get("rho", 0.0))


def _picks_l1dev(bid: str, date: str, B: int, latency: int, frozen_tag: str | None):
    reported = ec.reported_target_dates()
    if frozen_tag:
        sim = {"days": "oos", "month": "oosc", "year": "oos"}[frozen_tag]
        blob = json.loads(pc.selection_file(frozen_tag, sim).read_text())
        eta_r2 = float(blob["r2"][str(int(B))]["cfg"]["eta"])
    else:
        eta_r2 = _eta_r2_target(bid, B)
    episodes = read_jsonl(_select_path("l1dev", latency)) if _select_path("l1dev", latency).exists() else []
    pool = ec.selection_pool(episodes, date, "expanding", reported, B=B)
    if not pool:
        empty = {"eta": 0.0, "rho": 0.0, "fallback_base": True, "source_gain": 0.0, "empty_pool": True}
        return {"and": empty, "conservative": dict(empty), "eta_r2": eta_r2, "n_pool": 0}
    choice = ec.gate_choices(pool, eta_r2, "p0")
    choice["eta_r2"] = eta_r2
    choice["n_pool"] = len(pool)
    return choice


def cmd_eval_dev(args):
    """L1-dev (no trial-0 S1 step, commit schedule at this latency) for every target group.

    P0 / L0 / L1-staged rows already on disk are not recomputed. L0's (eta, rho) is the
    stored pick: round-2 causal rows, or causal-main ``s1_causal_and`` on days/drift.
    """
    if not torch.cuda.is_available():
        raise SystemExit("cuda:0 is not available")
    which = args.group
    plan = []
    if which in ("mid", "short", "long", "round2"):
        groups = ("mid", "short", "long") if which == "round2" else (which,)
        for g in groups:
            by = defaultdict(list)
            for t in _targets(g):
                by[t["bid"]].append(t)
            plan.append((g, None, by))
    if which in ("days", "drift", "frozen"):
        if which in ("days", "frozen"):
            plan.append(("days", "days", {"days": [
                {"bid": "days", "date": d, "target": "sub-M_ses-CO-" + d, "trunk_gap_days": None,
                 "gap_days": None, "stratum": "days"} for d in ("20150616", "20150617")] }))
        if which in ("drift", "frozen"):
            plan.append(("drift", "month", {"month": [
                {"bid": "month", "date": d, "target": "sub-M_ses-CO-" + d, "trunk_gap_days": None,
                 "gap_days": None, "stratum": "drift"} for d in DRIFT_DATES] }))
    _log(f"eval L1-dev {which}: " + ", ".join(g for g, _t, _b in plan))
    dev_methods = ("s1_l1_dev", "s1_l1_dev_transfer", "s1_l1_dev_cons")
    for group, frozen_tag, by_bid in plan:
        path = OUT / f"eval_{group}_dev.jsonl"
        done = {_eval_key(r) for r in read_jsonl(path)} if path.exists() else set()
        for bid, targets in by_bid.items():
            trunk = _open_trunk(bid if frozen_tag is None else frozen_tag)
            for t in targets:
                sd, prior = trunk.session(t["target"])
                spans = [(int(a), int(b)) for a, b in sd.trials]
                Bs = tuple(b for b in BS if b <= len(spans))
                if not Bs:
                    del sd, prior
                    continue
                cache = p0_cache(sd, prior, {s: trunk.p0[s] for s in SEEDS}, max(Bs))
                behavior = sd.record.behavior
                label_stats = trunk.world.label_stats
                dev = sd.x32.device
                if frozen_tag:
                    import datetime as dt
                    from ssm_decode.causal_protocol import TRUNK_LAST
                    last = dt.datetime.strptime(TRUNK_LAST[frozen_tag], "%Y%m%d").date()
                    cur = dt.datetime.strptime(t["date"], "%Y%m%d").date()
                    trunk_gap = (cur - last).days
                else:
                    trunk_gap = int(t["trunk_gap_days"])
                for B in Bs:
                    for seed in SEEDS:
                        if frozen_tag:
                            stored = _stored_frozen("days" if frozen_tag == "days" else "drift",
                                                    t["date"], B, seed, 0, "s1_causal_and")
                            if stored is None:
                                _log(f"no stored AND {group} {t['date']} B={B}")
                                continue
                            cfg = stored.get("cfg") or {}
                            l0_eta, l0_rho = float(cfg.get("eta", 0.0)), float(cfg.get("rho", 0.0))
                        else:
                            l0 = _stored_l0_pick(bid, t["date"], B, seed)
                            if l0 is None:
                                _log(f"no stored L0 {bid} {t['date']} B={B}")
                                continue
                            l0_eta, l0_rho = l0["eta"], l0["rho"]
                        for lat in LATENCIES:
                            if all((m, bid, t["date"], B, seed, lat) in done for m in dev_methods):
                                continue
                            picks = _picks_l1dev(bid, t["date"], B, lat, frozen_tag)
                            e_and, r_and = _choice_eta(picks.get("and"))
                            e_cons, r_cons = _choice_eta(picks.get("conservative"))
                            specs = {
                                "s1_l1_dev": (e_and, r_and),
                                "s1_l1_dev_transfer": (l0_eta, l0_rho),
                                "s1_l1_dev_cons": (e_cons, r_cons),
                            }
                            built = {}
                            t1 = time.perf_counter()
                            jb = trunk.jbar(int(trunk.L[B]))
                            for method, (eta, rho) in specs.items():
                                key = (float(eta), float(rho))
                                if key not in built:
                                    built[key] = _maps_for(
                                        trunk.base, sd, prior, cache, seed, spans, B, eta, rho, jb,
                                        "l1", latency=lat, drop_trial0=True)
                            q = sd.query_mask(calibration_indices(sd.bank.n_trials, B)).detach().cpu().numpy()
                            jobs, meta = [], []
                            for method, (eta, rho) in specs.items():
                                if (method, bid, t["date"], B, seed, lat) in done:
                                    continue
                                maps, reads, info = built[(float(eta), float(rho))]
                                segs = rcm.segments_for(spans, B, sd.n_bins, lat, "anytime")
                                rcm.assert_segment_causal(segs, spans, reads, staged_trial0=True)
                                packed = {k: rcm._pack(maps[k], dev) for _a, _b, k in segs}
                                jobs.append({"segments": segs, "packed": packed, "qmask": q, "want_h": False})
                                meta.append((method, info))
                            scores = rcm.decode_many(trunk.base, sd.x32, jobs, label_stats, behavior) if jobs else []
                            rows = []
                            for (method, info), r2 in zip(meta, scores):
                                rows.append({
                                    "method": method, "bid": bid, "target": t["target"], "date": t["date"],
                                    "B": int(B), "seed": int(seed), "latency": int(lat), "r2": float(r2),
                                    "trunk_gap_days": int(trunk_gap),
                                    "gap_days": t.get("gap_days") if t.get("gap_days") is not None else int(trunk_gap),
                                    "stratum": t.get("stratum"), "eta": info.get("eta"), "rho": info.get("rho"),
                                    "fallback_base": info.get("fallback_base"), "ring_mode": "l1dev",
                                    "l0_eta": l0_eta, "l0_rho": l0_rho,
                                    "n_pool": picks.get("n_pool"), "group": group,
                                    "drop_trial0": True,
                                })
                            if rows:
                                append_jsonl(path, rows)
                                done.update(_eval_key(r) for r in rows)
                            _log(f"dev {group} {bid} {t['date']} B={B} s{seed} lat={lat} "
                                 f"pool={picks.get('n_pool')} {time.perf_counter() - t1:.1f}s")
                del sd, prior, cache
                torch.cuda.empty_cache()
            del trunk
            torch.cuda.empty_cache()
    _log(f"eval L1-dev {which} done")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--group", required=True, choices=("mid", "short", "long"))
    s.add_argument("--variant", default="both", choices=("both", "l1", "l2", "l1dev"))
    s.set_defaults(fn=cmd_select)
    e = sub.add_parser("eval")
    e.add_argument("--group", required=True, choices=("mid", "short", "long"))
    e.set_defaults(fn=cmd_eval)
    f = sub.add_parser("eval-frozen")
    f.add_argument("--which", default="both", choices=("days", "drift", "both"))
    f.set_defaults(fn=cmd_eval_frozen)
    d = sub.add_parser("eval-dev")
    d.add_argument("--group", required=True,
                   choices=("mid", "short", "long", "round2", "days", "drift", "frozen"))
    d.set_defaults(fn=cmd_eval_dev)
    a = sub.add_parser("aggregate")
    a.set_defaults(fn=cmd_aggregate)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
