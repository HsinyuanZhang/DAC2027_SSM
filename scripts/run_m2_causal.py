#!/usr/bin/env python
"""Causal-first-B on FALCON M2.  The schedule and the gates live in
``ssm_decode.causal_protocol`` / ``stream_f7``; this driver only adapts M2.

  python scripts/run_m2_causal.py all
  python scripts/run_m2_causal.py select --what cca
  python scripts/run_m2_causal.py eval --pools pool1 --seeds 0 --budgets 8

Source episodes use the first B usable trials (recorded order) and are scored on
the trunk's own validation tail (last 20% of raw trials).  That tail was never
a gradient target.  The trials in between were: they are not a legal score.
Targets are scored on every usable trial after the first B.
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
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_m2causal")

from ssm_decode import fwd_refine as fr                         # noqa: E402
from ssm_decode import m2_calib as m2                           # noqa: E402
from ssm_decode import m2_causal as mc                          # noqa: E402
from ssm_decode.population_calib import PopStats, norm_stats    # noqa: E402

OUT = mc.OUT
SEL = OUT / "select"


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("GPU disappeared; stopping")
    return torch.device("cuda:0")


def _filt(args, name, default):
    raw = getattr(args, name, None)
    if not raw:
        return list(default)
    cast = type(default[0])
    return [cast(x) for x in str(raw).split(",") if x]


def log(msg):
    print(msg, flush=True)


def load_json(path: Path, default=None):
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mc._jsonable(obj), indent=1))


def seen_set(path: Path, fields: tuple[str, ...]) -> set:
    out = set()
    for r in mc.read_jsonl(path):
        out.add(tuple(r.get(k) for k in fields))
    return out


def nobias_caller(model):
    return mc.LatentTrunk(model, None)


def bias_caller(model, bias):
    dev = next(model.parameters()).device
    return mc.LatentTrunk(model, torch.as_tensor(np.asarray(bias, np.float32), device=dev))


def open_world(seed, pool, device, *, sources=True):
    world = m2.M2World(seed, pool, device, load_sources=sources)
    err = float(np.max(np.abs(world.trunk.prior_bias - world.trunk.folded_bias)))
    if err > 1e-5:
        raise AssertionError(f"mean fold bias != root in_proj bias ({err})")
    err_w = float(np.max(np.abs(world.trunk.prior_table - world.trunk.folded_table)))
    if err_w > 1e-5:
        raise AssertionError(f"mean fold table != root in_proj weight ({err_w})")
    return world


def conds_for(sd) -> np.ndarray:
    return mc.usable_conditions(sd, mc.load_trial_conditions(sd.name))


def anchor_path(seed, pool, phases) -> Path:
    return OUT / "anchor" / f"seed{seed}_{pool}_p{int(phases)}.pkl"


def get_anchor(world, phases: int):
    path = anchor_path(world.seed, world.pool, phases)
    if path.exists():
        return mc.load_pickle(path)
    name = world.trunk.source_ids[-1]
    sd = world.sessions[name]
    ref = mc.fit_anchor(world.trunk, sd, conds_for(sd), phases=int(phases))
    mc.save_pickle(path, ref)
    log(f"[anchor] s{world.seed} {world.pool} phases={phases} d={ref.m2_active} session={name}")
    return ref


def jbar_path(seed, pool) -> Path:
    return OUT / "jbar" / f"seed{seed}_{pool}.pt"


def get_jbar_blob(world):
    path = jbar_path(world.seed, world.pool)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    log(f"[jbar] s{world.seed} {world.pool}")
    blob = mc.jacobian_sums(world.trunk, world.sessions)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return blob


def basis_path(seed, pool) -> Path:
    return OUT / "basis" / f"seed{seed}_{pool}.pt"


def get_basis(world):
    path = basis_path(world.seed, world.pool)
    if path.exists():
        return torch.load(path, map_location=world.device, weights_only=False)
    log(f"[basis] s{world.seed} {world.pool}")
    moment = m2.source_feature_moment(world)
    basis = fr.feature_basis(moment, mc.R1_RANK).float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, path)
    return basis.to(world.device)


def cca_cfg(pool: str) -> dict:
    return load_json(SEL / "cca_config.json")[pool]


def protocols() -> dict:
    return load_json(SEL / "alpha_protocols.json")


def rule_file(what: str, pool: str) -> Path:
    return SEL / f"{what}_{pool}.json"


# ---------------------------------------------------------------- select ----

def _score_final_cca(caller, world, sd, cache, B, prior_b, qmask):
    got = cache["cca"][B]
    if not isinstance(got, tuple):
        return None
    w, b_extra, info = got
    mean, std = cache["norm"][B]
    bias = np.asarray(prior_b, np.float64) + np.asarray(b_extra, np.float64)
    r2 = mc.score_static(caller, sd, world.label_stats, w, bias, mean, std, qmask)
    return float(r2), info


def cmd_prepare(args):
    device = require_gpu()
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            world = open_world(seed, pool, device, sources=True)
            get_jbar_blob(world)
            get_basis(world)
            # Anchors for every candidate phase.  select-cca scores them.
            for phases in mc.CCA_PHASES:
                get_anchor(world, phases)
            del world
            torch.cuda.empty_cache()
    log("[prepare] done")


def cmd_select_cca(args):
    """One (phases, ridge) per pool.  Source val R² of the final CCA map, B in {8,16,33}."""
    device = require_gpu()
    path = SEL / "cca_episodes.jsonl"
    seen = seen_set(path, ("seed", "pool", "session", "phases", "ridge", "B"))
    budgets = (8, 16, 33)
    caller_cache = {}
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            world = open_world(seed, pool, device, sources=True)
            caller = nobias_caller(world.trunk.model)
            for phases in mc.CCA_PHASES:
                ref = get_anchor(world, phases)
                for name in world.source_ids:
                    sd = world.sessions[name]
                    prior_t, prior_b_t = world.priors[name]
                    prior_b = mc._as_np(prior_b_t)
                    conds = conds_for(sd)
                    # One cache per ridge would repeat P0.  CCA is the only ridge-dependent
                    # piece; rebuild the CCA entries per ridge, reuse nothing across ridges
                    # because the anchor phase is already fixed in ``ref``.
                    for ridge in mc.CCA_RIDGES:
                        need = [B for B in budgets
                                if (seed, pool, name, int(phases), float(ridge), int(B)) not in seen
                                and B <= sd.bank.n_trials]
                        if not need:
                            continue
                        k_max = max(need)
                        cache = mc.prefix_cache(sd, prior_t, prior_b, {}, k_max, conds, ref,
                                                ridge=float(ridge), phases=int(phases))
                        qmask = world.val_mask(name)
                        rows = []
                        for B in need:
                            scored = _score_final_cca(caller, world, sd, cache, B, prior_b, qmask)
                            rows.append({"seed": int(seed), "pool": pool, "session": name,
                                         "date": mc.ymd(name), "phases": int(phases), "ridge": float(ridge),
                                         "B": int(B), "r2": None if scored is None else scored[0],
                                         "cca_error": None if scored is not None else cache["cca"][B]})
                        mc.append_jsonl(path, rows)
                        for B in need:
                            seen.add((seed, pool, name, int(phases), float(ridge), int(B)))
                        log(f"[cca] s{seed} {pool} {name} p={phases} ridge={ridge}")
            del world, caller
            torch.cuda.empty_cache()
    _write_cca_config()


def _write_cca_config():
    rows = [r for r in mc.read_jsonl(SEL / "cca_episodes.jsonl") if r.get("r2") is not None]
    if not rows:
        raise SystemExit("no CCA episodes scored")
    cfg = {}
    for pool in mc.POOLS:
        sub = [r for r in rows if r["pool"] == pool]
        if not sub:
            continue
        groups = {}
        for r in sub:
            groups.setdefault((int(r["phases"]), float(r["ridge"])), []).append(float(r["r2"]))
        # Exact ties: smaller phase, then smaller ridge (do not buy a longer average for nothing).
        best = sorted(groups, key=lambda k: (float(np.mean(groups[k])), -k[0], -k[1]), reverse=True)[0]
        cfg[pool] = {"phases": best[0], "ridge": best[1], "source_r2": float(np.mean(groups[best])),
                     "n": len(groups[best]),
                     "grid": {f"p{a}_r{b:g}": float(np.mean(v)) for (a, b), v in sorted(groups.items())},
                     "align": "trial", "transform": "raw",
                     "note": "align=trial because held-in M2 has no go cue. "
                             "Scored on source validation tails at B in {8,16,33}, all backbone seeds."}
    write_json(SEL / "cca_config.json", cfg)
    log(f"[cca] config {json.dumps({p: {k: cfg[p][k] for k in ('phases', 'ridge', 'source_r2')} for p in cfg})}")


def _p0_r2(caller, world, sd, cache, cseed, B, prior_b, qmask) -> float:
    w = cache["p0"][cseed][B]
    mean, std = cache["norm"][B]
    return float(mc.score_static(caller, sd, world.label_stats, w, prior_b, mean, std, qmask))


def cmd_select_closed(args):
    """Alpha, R1-lr16 and R2 grids.  Offline final maps on the source val tail."""
    device = require_gpu()
    alpha_path = SEL / "alpha_episodes.jsonl"
    r1_path = SEL / "r1_episodes.jsonl"
    r2_path = SEL / "r2_episodes.jsonl"
    seen_a = seen_set(alpha_path, ("seed", "pool", "session", "cseed", "B"))
    seen_r1 = seen_set(r1_path, ("seed", "pool", "session", "cseed", "B"))
    seen_r2 = seen_set(r2_path, ("seed", "pool", "session", "cseed", "B"))
    budgets = _filt(args, "budgets", mc.B_LIST)
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            spec = cca_cfg(pool)
            world = open_world(seed, pool, device, sources=True)
            calibs = mc.load_calibrators(world, seed, pool)
            caller = nobias_caller(world.trunk.model)
            biased = bias_caller(world.trunk.model, world.trunk.prior_bias)  # replaced per LOO below
            jblob = get_jbar_blob(world)
            ref = get_anchor(world, spec["phases"])
            basis = get_basis(world)
            v0 = world.trunk.head()
            for name in world.source_ids:
                sd = world.sessions[name]
                prior_t, prior_b_t = world.priors[name]
                prior_b = mc._as_np(prior_b_t)
                biased = bias_caller(world.trunk.model, prior_b)
                conds = conds_for(sd)
                k_max = max(b for b in budgets if b <= sd.bank.n_trials)
                cache = mc.prefix_cache(sd, prior_t, prior_b, calibs, k_max, conds, ref,
                                        ridge=float(spec["ridge"]), phases=int(spec["phases"]))
                qmask = world.val_mask(name)
                gap = mc.days_between(name, mc._date_session(mc.POOL_TRUNK[pool]["last"]))
                for cseed in mc.CSEEDS:
                    for B in budgets:
                        if B > sd.bank.n_trials:
                            continue
                        idx = mc.source_train_idx(world, name, B)
                        if (seed, pool, name, cseed, int(B)) not in seen_a:
                            p0 = _p0_r2(caller, world, sd, cache, cseed, B, prior_b, qmask)
                            alphas = []
                            weights, biases = [], []
                            order = []
                            for alpha in mc.ALPHAS:
                                got = mc.w_alpha_parts(cache, cseed, B, float(alpha))
                                if got is None:
                                    alphas.append({"alpha": float(alpha), "r2": None})
                                    continue
                                w, b_shrink, mean, std, _ = got
                                weights.append(w)
                                biases.append(np.asarray(prior_b) + b_shrink)
                                order.append(float(alpha))
                            if weights:
                                scores = mc.score_many(caller, sd, world.label_stats, mean, std, weights, biases,
                                                       qmask)
                                by_a = dict(zip(order, scores))
                                alphas = [{"alpha": float(a), "r2": (None if a not in by_a else float(by_a[a]))}
                                          for a in mc.ALPHAS]
                                if abs(float(by_a.get(0.0, p0)) - p0) > 1e-3:
                                    log(f"[alpha] WARN s{seed} {name} B={B} c{cseed} "
                                        f"alpha0 {by_a.get(0.0)} != p0 {p0}")
                            mc.append_jsonl(alpha_path, [{"seed": int(seed), "pool": pool, "session": name,
                                                          "date": mc.ymd(name), "cseed": int(cseed), "B": int(B),
                                                          "gap_days": int(gap), "protocol": "causal_first_B",
                                                          "p0": p0, "alphas": alphas}])
                            seen_a.add((seed, pool, name, cseed, int(B)))
                        if (seed, pool, name, cseed, int(B)) not in seen_r1:
                            _select_r1_episode(world, sd, name, prior_t, prior_b_t, calibs[cseed], idx, qmask,
                                               basis, B, seed, pool, cseed, r1_path)
                            seen_r1.add((seed, pool, name, cseed, int(B)))
                        if (seed, pool, name, cseed, int(B)) not in seen_r2:
                            _select_r2_episode(biased, caller, world, sd, cache, cseed, B, prior_b, prior_t,
                                               qmask, jblob, name, seed, pool, r2_path)
                            seen_r2.add((seed, pool, name, cseed, int(B)))
                        log(f"[closed] s{seed} {pool} {name} c{cseed} B={B}")
            del world
            torch.cuda.empty_cache()
    _write_alpha_protocols()
    _write_r1_rules()
    _write_r2_rules()


def _select_r1_episode(world, sd, name, prior, prior_b_t, calib, idx, qmask, basis, B, seed, pool, cseed, path):
    """Compact R1 on the offline final P0.  ``off`` is a candidate.  Stats are the zero-state stream."""
    stats = PopStats.from_trials(sd.bank, idx)
    with torch.no_grad():
        res = calib.calibrate(stats, prior)
        y0, h = m2.decode_full(world.trunk, sd, res.table, prior_b_t, res.mean, res.std, want_h=True)
    off = m2.r2_physical(world.label_stats, sd, y0, qmask)[0]
    sts = m2.stream_readout_stats(world.trunk, sd, (prior, prior_b_t), calib, idx, burn=mc.R1_BURNS, use_calib=True)
    rows_score = [{"lam": None, "burn": 0, "r2": float(off)}]
    hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
    for burn in mc.R1_BURNS:
        for lam in mc.R1_LAMS:
            v = m2.ridge_v(world.trunk, sts[burn], lam, "lowrank", basis=basis[:, :mc.R1_RANK])
            y = hb @ v.to(hb.dtype).T
            rows_score.append({"lam": float(lam), "burn": int(burn),
                               "r2": float(m2.r2_physical(world.label_stats, sd, y, qmask)[0])})
    mc.append_jsonl(path, [{"seed": int(seed), "pool": pool, "session": name, "cseed": int(cseed),
                            "B": int(B), "protocol": "causal_first_B", "scores": rows_score}])


def _select_r2_episode(biased, caller, world, sd, cache, cseed, B, prior_b, prior_t, qmask, jblob, name,
                       seed, pool, path):
    idx = mc.calibration_indices(sd.bank.n_trials, B)
    mean, std = cache["norm"][B]
    table0 = torch.as_tensor(cache["p0"][cseed][B], dtype=torch.float32, device=sd.device)
    scores = []
    X, Y, M = mc.replay_batch(sd, idx, mean, std)
    for L in mc.R2_LS:
        jb = mc.jbar_from(jblob, L, exclude=name, device=sd.device)
        for eta in mc.R2_ETAS:
            for rho in mc.R2_RHOS:
                snaps = fr.fj_refine(biased, X, Y, M, table0, jb, float(eta), float(rho), mc.R2_STEPS)
                weights = [mc._as_np(snaps[s]) for s in mc.R2_STEPS]
                biases = [prior_b for _ in mc.R2_STEPS]
                got = mc.score_many(caller, sd, world.label_stats, mean, std, weights, biases, qmask)
                for S, r2 in zip(mc.R2_STEPS, got):
                    scores.append({"L": int(L), "eta": float(eta), "rho": float(rho), "S": int(S), "r2": float(r2)})
    mc.append_jsonl(path, [{"seed": int(seed), "pool": pool, "session": name, "cseed": int(cseed),
                            "B": int(B), "protocol": "causal_first_B", "scores": scores}])


def _write_alpha_protocols():
    rows = mc.read_jsonl(SEL / "alpha_episodes.jsonl")
    out = {"reported_dates": sorted(mc.REPORTED_DATES), "protocols": {}}
    for variant in ("expanding", "strict"):
        block = {}
        for pool, target in mc.POOL_OF_TARGET.items():
            pass
        for pool in mc.POOLS:
            block[pool] = {}
            for target, owner in mc.POOL_OF_TARGET.items():
                if owner != pool:
                    continue
                block[pool][target] = {}
                for B in mc.B_LIST:
                    block[pool][target][str(B)] = mc.select_alpha_m2(rows, pool, target, B, variant)
        out["protocols"][variant] = block
    write_json(SEL / "alpha_protocols.json", out)
    brief = {pool: {t: {b: spec["alpha"] for b, spec in per.items()}
                    for t, per in out["protocols"]["expanding"][pool].items()}
             for pool in out["protocols"]["expanding"]}
    log(f"[alpha] expanding {json.dumps(brief)}")


def _simple_token(v):
    """Smaller is simpler.  None (the off switch) beats any positive setting on a tie."""
    if v is None:
        return (0, 0.0)
    if isinstance(v, str):
        order = {"off": 0, "diag": 1, "lr16": 2, "lowrank": 2, "full": 3}
        return (1, float(order.get(v, 9)))
    return (1, float(v))


def _best_score_row(rows, fields):
    """Max mean r2.  Exact ties prefer the simpler key."""
    groups = {}
    for r in rows:
        key = tuple(r[f] for f in fields)
        groups.setdefault(key, []).append(float(r["r2"]))

    def sort_key(k):
        tokens = tuple(_simple_token(v) for v in k)
        # Negate so max() prefers the simpler (smaller) token after the mean.
        return (float(np.mean(groups[k])), tuple((-a, -b) for a, b in tokens))

    best = max(groups, key=sort_key)
    return best, float(np.mean(groups[best])), len(groups[best])


def _write_r1_rules():
    eps = mc.read_jsonl(SEL / "r1_episodes.jsonl")
    for pool in mc.POOLS:
        sub = [r for r in eps if r["pool"] == pool]
        if not sub:
            continue
        by_b = {}
        flat_by_b = {}
        for ep in sub:
            flat_by_b.setdefault(int(ep["B"]), []).extend(
                [{"lam": s["lam"], "burn": s["burn"], "r2": s["r2"]} for s in ep["scores"]])
        rules = {"pool": pool, "protocol": "causal_first_B", "rank": mc.R1_RANK, "b": {}}
        for B, rows in sorted(flat_by_b.items()):
            # lam None is off.  _best_score_row treats None as simplest.
            key, mean, n = _best_score_row(rows, ("lam", "burn"))
            rules["b"][str(B)] = {"lam": key[0], "burn": int(key[1]), "kind": "lowrank",
                                  "rank": mc.R1_RANK, "source_r2": mean, "n_scores": n,
                                  "n_episodes": len(sub) // max(len(flat_by_b), 1)}
        write_json(rule_file("r1", pool), rules)
        log(f"[r1] {pool} " + " ".join(f"B{b}:{rules['b'][b]['lam']}/{rules['b'][b]['burn']}" for b in rules["b"]))


def _write_r2_rules():
    eps = mc.read_jsonl(SEL / "r2_episodes.jsonl")
    for pool in mc.POOLS:
        sub = [r for r in eps if r["pool"] == pool]
        if not sub:
            continue
        flat_by_b = {}
        for ep in sub:
            flat_by_b.setdefault(int(ep["B"]), []).extend(ep["scores"])
        rules = {"pool": pool, "protocol": "causal_first_B", "b": {}}
        for B, rows in sorted(flat_by_b.items()):
            key, mean, n = _best_score_row(rows, ("L", "eta", "rho", "S"))
            rules["b"][str(B)] = {"L": int(key[0]), "eta": float(key[1]), "rho": float(key[2]),
                                  "S": int(key[3]), "source_r2": mean, "n_scores": n}
        write_json(rule_file("r2", pool), rules)
        log(f"[r2] {pool} " + " ".join(
            f"B{b}:L{rules['b'][b]['L']}/e{rules['b'][b]['eta']}/s{rules['b'][b]['S']}" for b in rules["b"]))


def cmd_select_refine(args):
    """S1, COMBO, CCA+S1 and R3.  Needs alpha protocols and the R2 rule (kernel length, tie-break)."""
    device = require_gpu()
    proto = protocols()
    paths = {w: SEL / f"{w}_episodes.jsonl" for w in ("s1", "combo", "cca_s1", "r3")}
    seen = {w: seen_set(p, ("seed", "pool", "session", "cseed", "B")) for w, p in paths.items()}
    budgets = _filt(args, "budgets", mc.B_LIST)
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            spec = cca_cfg(pool)
            r2_rules = load_json(rule_file("r2", pool))
            world = open_world(seed, pool, device, sources=True)
            calibs = mc.load_calibrators(world, seed, pool)
            caller = nobias_caller(world.trunk.model)
            jblob = get_jbar_blob(world)
            ref = get_anchor(world, spec["phases"])
            basis = get_basis(world)
            v0 = world.trunk.head()
            # Alpha is constant across this pool's targets (one gap bin).  Use the first target.
            target0 = [t for t, p in mc.POOL_OF_TARGET.items() if p == pool][0]
            for name in world.source_ids:
                sd = world.sessions[name]
                prior_t, prior_b_t = world.priors[name]
                prior_b = mc._as_np(prior_b_t)
                conds = conds_for(sd)
                k_max = max(b for b in budgets if b <= sd.bank.n_trials)
                cache = mc.prefix_cache(sd, prior_t, prior_b, calibs, k_max, conds, ref,
                                        ridge=float(spec["ridge"]), phases=int(spec["phases"]))
                qmask = world.val_mask(name)
                base_loop = nobias_caller(world.trunk.model)
                for cseed in mc.CSEEDS:
                    for B in budgets:
                        if B > sd.bank.n_trials:
                            continue
                        alpha = proto["protocols"]["expanding"][pool][target0][str(B)]["alpha"]
                        if alpha is None:
                            alpha = 0.0
                        r2b = r2_rules["b"][str(B)]
                        jb = mc.jbar_from(jblob, int(r2b["L"]), exclude=name, device=device)
                        p0 = _p0_r2(caller, world, sd, cache, cseed, B, prior_b, qmask)
                        idx = mc.calibration_indices(sd.bank.n_trials, B)
                        raw = _raw_trials(sd, idx)
                        if (seed, pool, name, cseed, int(B)) not in seen["s1"]:
                            _grid_rebase(base_loop, raw, prior_t, cache, cseed, B, prior_b, 0.0, "p0", jb,
                                         caller, sd, world, qmask, p0, p0, float(r2b["eta"]),
                                         paths["s1"], seed, pool, name, cseed)
                            seen["s1"].add((seed, pool, name, cseed, int(B)))
                        if (seed, pool, name, cseed, int(B)) not in seen["combo"]:
                            kind = "p0" if float(alpha) == 0.0 else "w_alpha"
                            base_r = _closed_base_r2(caller, world, sd, cache, cseed, B, prior_b, qmask,
                                                     kind, float(alpha))
                            if base_r is None:
                                log(f"[combo] skip {name} B={B} CCA missing")
                            else:
                                _grid_rebase(base_loop, raw, prior_t, cache, cseed, B, prior_b, float(alpha),
                                             kind, jb, caller, sd, world, qmask, p0, base_r, float(r2b["eta"]),
                                             paths["combo"], seed, pool, name, cseed)
                                seen["combo"].add((seed, pool, name, cseed, int(B)))
                        if (seed, pool, name, cseed, int(B)) not in seen["cca_s1"]:
                            base_r = _closed_base_r2(caller, world, sd, cache, cseed, B, prior_b, qmask, "cca", 1.0)
                            if base_r is None:
                                log(f"[cca+s1] skip {name} B={B} CCA missing")
                            else:
                                _grid_rebase(base_loop, raw, prior_t, cache, cseed, B, prior_b, 1.0, "cca", jb,
                                             caller, sd, world, qmask, p0, base_r, float(r2b["eta"]),
                                             paths["cca_s1"], seed, pool, name, cseed)
                                seen["cca_s1"].add((seed, pool, name, cseed, int(B)))
                        if (seed, pool, name, cseed, int(B)) not in seen["r3"]:
                            _select_r3_episode(world, sd, cache, cseed, B, prior_b, prior_t, qmask, jblob,
                                               basis, v0, name, r2b, seed, pool, paths["r3"])
                            seen["r3"].add((seed, pool, name, cseed, int(B)))
                        log(f"[refine] s{seed} {pool} {name} c{cseed} B={B}")
            del world
            torch.cuda.empty_cache()
    _write_margin("s1", "p0")
    _write_margin("combo", "base")
    _write_margin("cca_s1", "base")
    _write_r3_rules()


def _raw_trials(sd, idx):
    return [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype), sd.valid[a:b].to(dtype=sd.x32.dtype))
            for a, b in (sd.trials[int(i)] for i in idx)]


def _closed_base_r2(caller, world, sd, cache, cseed, B, prior_b, qmask, kind, alpha):
    if kind == "p0":
        return _p0_r2(caller, world, sd, cache, cseed, B, prior_b, qmask)
    if kind == "cca":
        got = cache["cca"][B]
        if not isinstance(got, tuple):
            return None
        w, b_extra = got[0], got[1]
        mean, std = cache["norm"][B]
        bias = np.asarray(prior_b) + np.asarray(b_extra)
        return float(mc.score_static(caller, sd, world.label_stats, w, bias, mean, std, qmask))
    got = mc.w_alpha_parts(cache, cseed, B, float(alpha))
    if got is None:
        return None
    w, b_shrink, mean, std, _ = got
    return float(mc.score_static(caller, sd, world.label_stats, w, np.asarray(prior_b) + b_shrink,
                                 mean, std, qmask))


def _grid_rebase(base_loop, raw, prior_t, cache, cseed, B, prior_b, alpha, kind, jbar, caller, sd, world,
                 qmask, p0, base_r, eta_r2, path, seed, pool, name, cseed_):
    from ssm_decode.causal_protocol import run_rebased_many
    configs = [(e, r) for e in mc.S1_ETAS for r in mc.S1_RHOS]
    prepare = mc._prepare(cache, cseed, kind, alpha, prior_t.device)
    bias_fn = mc._loop_base_bias(cache, cseed, kind, alpha, prior_b, prior_t.device)
    many = run_rebased_many(base_loop, raw, prior_t.float(), configs, prepare, jbar=jbar, base_bias=bias_fn)
    mean, std = cache["norm"][B]
    weights, biases = [], []
    for ans in many:
        bias = mc._as_np(ans["bias"]) + np.asarray(prior_b, np.float64)
        if kind == "cca":
            bias = bias + np.asarray(cache["cca"][B][1], np.float64)
        elif kind == "w_alpha" and float(alpha) != 0.0:
            bias = bias + np.asarray(mc.w_alpha_parts(cache, cseed, B, alpha)[1], np.float64)
        weights.append(mc._as_np(ans["table"]))
        biases.append(bias)
    scores = mc.score_many(caller, sd, world.label_stats, mean, std, weights, biases, qmask)
    rebase = [{"eta": float(e), "rho": float(r), "r2": float(sc)} for (e, r), sc in zip(configs, scores)]
    zero = next(s["r2"] for s in rebase if s["eta"] == 0.0 and s["rho"] == 0.0)
    if abs(zero - float(base_r)) > 1e-3:
        log(f"[rebase] WARN eta0 {zero:.4f} != base {base_r:.4f} on {name} B={B} {kind}")
    gap = mc.days_between(name, mc._date_session(mc.POOL_TRUNK[pool]["last"]))
    mc.append_jsonl(path, [{"seed": int(seed), "pool": pool, "session": name, "date": mc.ymd(name),
                            "cseed": int(cseed_), "B": int(B), "protocol": "causal_first_B",
                            "gap_days": int(gap), "p0": float(p0), "base": float(base_r),
                            "alpha": float(alpha), "kind": kind, "eta_r2": float(eta_r2), "rebase": rebase}])


def _select_r3_episode(world, sd, cache, cseed, B, prior_b, prior_t, qmask, jblob, basis, v0, name, r2b,
                       seed, pool, path):
    idx = mc.calibration_indices(sd.bank.n_trials, B)
    mean, std = cache["norm"][B]
    caller_b = bias_caller(world.trunk.model, prior_b)
    table0 = torch.as_tensor(cache["p0"][cseed][B], dtype=torch.float32, device=sd.device)
    jb = mc.jbar_from(jblob, int(r2b["L"]), exclude=name, device=sd.device)
    tab = mc.r2_table(caller_b, sd, idx, table0, mean, std, jb, r2b["eta"], r2b["rho"], r2b["S"])
    # Contiguous features of the refined table; the Gram is the zero-state replay (frozen R3).
    y, h = _forward_yh(nobias_caller(world.trunk.model), sd, mc._as_np(tab), prior_b, mean, std)
    st_caller = caller_b
    X, Y, M = mc.replay_batch(sd, idx, mean, std)
    st = fr.readout_stats(st_caller, X, Y, M, tab.float())
    lams = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0)
    scores = []
    base = float(m2.r2_physical(world.label_stats, sd, y, qmask)[0])
    scores.append({"kind": "off", "lam": None, "r2": base})
    for kind in mc.R3_KINDS:
        for lam in lams:
            if kind == "lr16":
                dv = fr.ridge_delta(st, float(lam), "lowrank", basis=basis[:, :mc.R1_RANK])
            else:
                dv = fr.ridge_delta(st, float(lam), kind)
            vv = (v0.double() + dv).to(dtype=h.dtype)
            hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
            yy = hb @ vv.T
            scores.append({"kind": kind, "lam": float(lam),
                           "r2": float(m2.r2_physical(world.label_stats, sd, yy, qmask)[0])})
    mc.append_jsonl(path, [{"seed": int(seed), "pool": pool, "session": name, "cseed": int(cseed),
                            "B": int(B), "protocol": "causal_first_B", "scores": scores}])


def _forward_yh(caller, sd, weight, bias, mean, std):
    dev = sd.device
    w = torch.as_tensor(np.asarray(weight, np.float32), device=dev)
    b = torch.as_tensor(np.asarray(bias, np.float32), device=dev)
    mean_t = torch.as_tensor(np.asarray(mean, np.float32), device=dev)
    std_t = torch.as_tensor(np.asarray(std, np.float32), device=dev)
    with torch.inference_mode():
        xz = (sd.x32 - mean_t) / std_t.clamp_min(1e-8)
        z = (xz @ w.T + b).unsqueeze(0)
        store = {}
        handle = caller.out_proj.register_forward_pre_hook(lambda m, inp: store.__setitem__("h", inp[0]))
        try:
            y = caller(z)
        finally:
            handle.remove()
    return y[0], store["h"][0]


def _write_margin(what: str, base_key: str):
    for pool in mc.POOLS:
        eps = [r for r in mc.read_jsonl(SEL / f"{what}_episodes.jsonl") if r["pool"] == pool and r.get("rebase")]
        if not eps:
            log(f"[margin] {what} {pool} empty")
            continue
        rules = mc.margin_rules(eps, base_key)
        rules["pool"] = pool
        rules["what"] = what
        write_json(rule_file(what, pool), rules)
        brief = {b: {"and": rules["b"][b]["and"]["eta"], "fb": rules["b"][b]["and"]["fallback_base"],
                     "sessions": rules["b"][b]["and"]["session_gate"]["n_positive_sessions"]}
                 for b in rules["b"]}
        log(f"[margin] {what} {pool} {json.dumps(brief)}")


def _write_r3_rules():
    eps = mc.read_jsonl(SEL / "r3_episodes.jsonl")
    for pool in mc.POOLS:
        sub = [r for r in eps if r["pool"] == pool]
        if not sub:
            continue
        flat = {}
        for ep in sub:
            flat.setdefault(int(ep["B"]), []).extend(ep["scores"])
        rules = {"pool": pool, "protocol": "causal_first_B", "b": {}}
        for B, rows in sorted(flat.items()):
            key, mean, n = _best_score_row(rows, ("kind", "lam"))
            rules["b"][str(B)] = {"kind": key[0], "lam": key[1], "source_r2": mean, "n_scores": n}
        write_json(rule_file("r3", pool), rules)
        log(f"[r3] {pool} " + " ".join(f"B{b}:{rules['b'][b]['kind']}/{rules['b'][b]['lam']}" for b in rules["b"]))


def cmd_select_uireg(args):
    """P0+ui_reg.  Start = shrinkage centre = the prefix P0 table (000688 run_traj), not the prior."""
    device = require_gpu()
    path = SEL / "uireg_episodes.jsonl"
    seen = seen_set(path, ("seed", "pool", "session", "cseed", "B"))
    budgets = _filt(args, "budgets", mc.B_LIST)
    # Two most recent source sessions, matching the M2 ui_reg budget.  Both calibrators,
    # because the trajectory starts at that calibrator's P0.
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            spec = cca_cfg(pool)
            world = open_world(seed, pool, device, sources=True)
            calibs = mc.load_calibrators(world, seed, pool)
            ref = get_anchor(world, spec["phases"])
            names = list(world.source_ids)[-2:]
            for name in names:
                sd = world.sessions[name]
                prior_t, prior_b_t = world.priors[name]
                prior_b = mc._as_np(prior_b_t)
                conds = conds_for(sd)
                k_max = max(b for b in budgets if b <= sd.bank.n_trials)
                cache = mc.prefix_cache(sd, prior_t, prior_b, calibs, k_max, conds, ref,
                                        ridge=float(spec["ridge"]), phases=int(spec["phases"]))
                qmask = world.val_mask(name)
                for cseed in mc.CSEEDS:
                    for B in budgets:
                        if B > sd.bank.n_trials or (seed, pool, name, cseed, int(B)) in seen:
                            continue
                        t1 = time.perf_counter()
                        idx = mc.source_train_idx(world, name, B)
                        mean, std = cache["norm"][B]
                        tab0 = cache["p0"][cseed][B]
                        scores = _uireg_grid(world, sd, idx, tab0, prior_b, np.asarray(mean), np.asarray(std), qmask)
                        mc.append_jsonl(path, [{"seed": int(seed), "pool": pool, "session": name,
                                                "cseed": int(cseed), "B": int(B),
                                                "protocol": "causal_first_B", "init": "p0", "scores": scores}])
                        seen.add((seed, pool, name, cseed, int(B)))
                        log(f"[uireg] s{seed} {pool} {name} c{cseed} B={B} {time.perf_counter()-t1:.0f}s")
            del world
            torch.cuda.empty_cache()
    _write_uireg_rules()


def _uireg_grid(world, sd, idx, table0, prior_b, mean, std, qmask):
    from ssm_decode.population_calib import SessionData  # noqa: F401  (kept local to document the prep)
    prep_mean = torch.as_tensor(mean, device=sd.device)
    prep_std = torch.as_tensor(std, device=sd.device)
    xz = ((sd.x32 - prep_mean) / prep_std.clamp_min(1e-8)).detach().cpu().numpy()
    y_norm = sd.record.behavior  # replaced below with normalised labels
    ym, ys = world.label_stats
    y_norm = (sd.record.behavior - ym) / ys
    valid = sd.valid_np
    windows = [sd.trials[int(i)] for i in idx]
    rows = []
    # steps=0 is the unfitted P0, so the grid can fall back to "do nothing".
    y = m2.decode_full(world.trunk, sd, torch.as_tensor(table0, dtype=torch.float32, device=sd.device),
                       torch.as_tensor(prior_b, dtype=torch.float32, device=sd.device),
                       prep_mean, prep_std)
    rows.append({"lam": None, "lr": None, "steps": 0,
                 "r2": float(m2.r2_physical(world.label_stats, sd, y, qmask)[0])})
    for lam in mc.UI_LAMS:
        for lr in mc.UI_LRS:
            traj = m2.ui_reg_trajectory_m2(
                world.trunk, xz, y_norm, windows, table0, prior_b, table0, lam=lam, lr=lr,
                snapshots=mc.UI_STEPS, device=world.device, seed=0, valid_mask=valid)
            for step, tab in traj.items():
                y = m2.decode_full(world.trunk, sd, torch.as_tensor(tab, dtype=torch.float32, device=sd.device),
                                   torch.as_tensor(prior_b, dtype=torch.float32, device=sd.device),
                                   prep_mean, prep_std)
                rows.append({"lam": float(lam), "lr": float(lr), "steps": int(step),
                             "r2": float(m2.r2_physical(world.label_stats, sd, y, qmask)[0])})
    return rows


def _write_uireg_rules():
    eps = mc.read_jsonl(SEL / "uireg_episodes.jsonl")
    for pool in mc.POOLS:
        sub = [r for r in eps if r["pool"] == pool]
        if not sub:
            continue
        flat = {}
        for ep in sub:
            flat.setdefault(int(ep["B"]), []).extend(ep["scores"])
        rules = {"pool": pool, "protocol": "causal_first_B", "init": "p0", "center": "p0", "b": {}}
        for B, rows in sorted(flat.items()):
            key, mean, n = _best_score_row(rows, ("lam", "lr", "steps"))
            rules["b"][str(B)] = {"lam": key[0], "lr": key[1], "steps": int(key[2]),
                                  "source_r2": mean, "n_scores": n}
        write_json(rule_file("uireg", pool), rules)
        log(f"[uireg] {pool} " + " ".join(
            f"B{b}:{rules['b'][b]['steps']}" for b in rules["b"]))


# ---------------------------------------------------------------- eval ----

_BWD = {"n": 0}


def _patch_bwd():
    orig = torch.autograd.backward

    def counting(*a, **k):
        _BWD["n"] += 1
        return orig(*a, **k)
    torch.autograd.backward = counting


def cmd_eval(args):
    device = require_gpu()
    _patch_bwd()
    proto = protocols()
    path = OUT / "rows.jsonl"
    seen = seen_set(path, ("method", "pool", "seed", "cseed", "target", "B", "latency"))
    budgets = _filt(args, "budgets", mc.B_LIST)
    latencies = _filt(args, "latency", (0, 1))
    for seed in _filt(args, "seeds", mc.SEEDS):
        for pool in _filt(args, "pools", mc.POOLS):
            if not require_gpu():
                return
            spec = cca_cfg(pool)
            world = open_world(seed, pool, device, sources=True)
            calibs = mc.load_calibrators(world, seed, pool)
            caller = nobias_caller(world.trunk.model)
            jblob = get_jbar_blob(world)
            ref = get_anchor(world, spec["phases"])
            basis = get_basis(world)
            v0 = world.trunk.head()
            rules = {w: load_json(rule_file(w, pool)) for w in ("s1", "combo", "cca_s1", "r1", "r2", "r3", "uireg")}
            legacy_ui = load_json(ROOT / "results/m2_calib_20261016/uireg" / f"seed{seed}" / pool / "selection_uireg.json")
            hash0 = world.trunk.trunk_hash
            targets = [t for t, p in mc.POOL_OF_TARGET.items() if p == pool]
            if args.targets:
                want = set(args.targets.split(","))
                targets = [t for t in targets if t in want]
            for target in targets:
                sd, prior = world.target(target)
                prior_t, prior_b_t = prior
                prior_w = mc._as_np(prior_t)
                prior_b = mc._as_np(prior_b_t)
                conds = conds_for(sd)
                k_max = max(b for b in budgets if b <= sd.bank.n_trials)
                log(f"[eval] s{seed} {pool} {target} trials={sd.bank.n_trials}")
                cache = mc.prefix_cache(sd, prior_t, prior_b, calibs, k_max, conds, ref,
                                        ridge=float(spec["ridge"]), phases=int(spec["phases"]))
                spans = [(int(a), int(b)) for a, b in sd.trials]
                gaps = mc.target_gaps(pool, target)
                cold, initial = mc.cold_initial(prior_w, prior_b, cache)
                jb_full = {L: mc.jbar_from(jblob, L, exclude=None, device=device) for L in mc.R2_LS}
                for B in budgets:
                    if B > len(spans):
                        log(f"[skip] {target} B={B}")
                        continue
                    _eval_one_b(world, sd, caller, cache, calibs, cold, initial, spans, prior_t, prior_w, prior_b,
                                rules, proto, legacy_ui, jb_full, basis, v0, gaps, spec, hash0,
                                seed, pool, target, B, latencies, seen, path)
            del world
            torch.cuda.empty_cache()
    log("[eval] done")


def _eval_one_b(world, sd, caller, cache, calibs, cold, initial, spans, prior_t, prior_w, prior_b,
                rules, proto, legacy_ui, jb_full, basis, v0, gaps, spec, hash0,
                seed, pool, target, B, latencies, seen, path):
    label = world.label_stats
    q = sd.query_mask(mc.calibration_indices(sd.bank.n_trials, B))
    alpha_a = proto["protocols"]["expanding"][pool][target][str(B)]
    alpha_b = proto["protocols"]["strict"][pool][target][str(B)]
    r2b = rules["r2"]["b"][str(B)]
    jb = jb_full[int(r2b["L"])]
    rows = []
    bwd0 = _BWD["n"]
    # Calibrator-free maps once.  cseed -1.
    free = {}
    free["p_prior"] = (mc.prior_maps(prior_w, prior_b, cache, B, cold, initial), "anytime",
                       {"bias": "folded_in_proj"})
    cca_maps, cca_info = mc.cca_final_maps(cache, B, prior_b, cold, initial)
    if cca_maps is None:
        free["ridge_cca"] = None
        free_note = {"ridge_cca": cca_info}
    else:
        free["ridge_cca"] = (cca_maps, "final_only", cca_info)
        free_note = {}
    for method, payload in list(free.items()):
        if payload is None:
            continue
        maps, install, cfg = payload
        _emit_decode(caller, sd, label, maps, install, spans, B, q, latencies, method, -1,
                     seed, pool, target, gaps, "causal_first_B", False, cfg, bwd0, hash0, rows, seen)
    for method, info in free_note.items():
        _emit_unavailable(method, -1, seed, pool, target, B, latencies, gaps, info, rows, seen)

    for cseed in mc.CSEEDS:
        bwd_enter = _BWD["n"]
        maps_p0 = mc.p0_maps(cache, cseed, B, prior_b, cold, initial)
        built = {"p0": (maps_p0, "anytime", {})}
        notes = {}
        a_alpha = alpha_a.get("alpha")
        b_alpha = alpha_b.get("alpha")
        if a_alpha is None:
            notes["w_alpha"] = {"unavailable": True, "reason": "no expanding alpha"}
        else:
            got, info = mc.alpha_maps(cache, cseed, B, float(a_alpha), prior_b, cold, initial)
            if got is None:
                notes["w_alpha"] = info
            else:
                built["w_alpha"] = (got, "anytime", {**(info or {}), "protocol": "expanding",
                                                     "alpha_detail": alpha_a})
        if b_alpha is None:
            notes["w_alpha_b"] = {"unavailable": True, "reason": "strict pool empty", "detail": alpha_b}
        else:
            got, info = mc.alpha_maps(cache, cseed, B, float(b_alpha), prior_b, cold, initial)
            if got is None:
                notes["w_alpha_b"] = info
            else:
                built["w_alpha_b"] = (got, "anytime", {**(info or {}), "protocol": "strict",
                                                       "alpha_detail": alpha_b})
        s1 = rules["s1"]["b"][str(B)]["and"]
        combo = rules["combo"]["b"][str(B)]["and"]
        cca_s = rules["cca_s1"]["b"][str(B)]["and"]
        _put_loop(built, notes, "s1", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                  float(s1["eta"]), float(s1["rho"]), 0.0, "p0", s1, cold, initial)
        _put_loop(built, notes, "s1_cons", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                  *_cons_eta(s1), 0.0, "p0", s1, cold, initial)
        if a_alpha is None:
            notes["combo"] = {"unavailable": True, "reason": "no expanding alpha"}
            notes["combo_cons"] = notes["combo"]
        else:
            kind_a = "p0" if float(a_alpha) == 0.0 else "w_alpha"
            _put_loop(built, notes, "combo", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                      float(combo["eta"]), float(combo["rho"]), float(a_alpha), kind_a, combo,
                      cold, initial)
            _put_loop(built, notes, "combo_cons", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                      *_cons_eta(combo), float(a_alpha), kind_a, combo, cold, initial)
        if b_alpha is None:
            notes["combo_b"] = {"unavailable": True, "reason": "strict pool empty"}
        else:
            kind_b = "p0" if float(b_alpha) == 0.0 else "w_alpha"
            _put_loop(built, notes, "combo_b", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                      float(combo["eta"]), float(combo["rho"]), float(b_alpha), kind_b, combo,
                      cold, initial)
        _put_loop(built, notes, "cca_s1", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                  float(cca_s["eta"]), float(cca_s["rho"]), 1.0, "cca", cca_s, cold, initial)
        _put_loop(built, notes, "cca_s1_cons", caller, sd, prior_t, cache, cseed, B, prior_b, jb,
                  *_cons_eta(cca_s), 1.0, "cca", cca_s, cold, initial)
        r1 = rules["r1"]["b"][str(B)]
        r1m, r1info = mc.r1_maps(world.trunk.model, sd, cache, cseed, B, prior_b, prior_t,
                                 r1["lam"], int(r1["burn"]), basis, v0)
        built["r1_lr16"] = ({mc.COLD: cold, mc.INITIAL: initial, **r1m}, "anytime", r1info)
        # References.  Candidate construction (including R1) must not have backpropped.
        bwd_candidates = _BWD["n"]
        if bwd_candidates != bwd_enter:
            raise AssertionError(f"candidate path backpropped ({bwd_candidates - bwd_enter})")
        tab_r2 = mc.r2_table(bias_caller(world.trunk.model, prior_b), sd,
                             mc.calibration_indices(sd.bank.n_trials, B),
                             torch.as_tensor(cache["p0"][cseed][B], dtype=torch.float32, device=sd.device),
                             *cache["norm"][B], jb, r2b["eta"], r2b["rho"], r2b["S"])
        mean_b, std_b = cache["norm"][B]
        final = mc.np_map(mc._as_np(tab_r2), prior_b, mean_b, std_b, mc._end(cache, B))
        built["r2"] = ({mc.COLD: cold, mc.INITIAL: initial, B: final}, "final_only", dict(r2b))
        r3 = rules["r3"]["b"][str(B)]
        if r3["kind"] == "off" or r3["lam"] is None:
            built["r3"] = (built["r2"][0], "final_only", {**r3, "alias_of": "r2"})
        else:
            v = mc.r3_readout(bias_caller(world.trunk.model, prior_b), sd,
                              mc.calibration_indices(sd.bank.n_trials, B), tab_r2, mean_b, std_b,
                              r3["lam"], r3["kind"], basis, v0)
            final3 = mc.np_map(mc._as_np(tab_r2), prior_b, mean_b, std_b, mc._end(cache, B), readout=v)
            built["r3"] = ({mc.COLD: cold, mc.INITIAL: initial, B: final3}, "final_only", dict(r3))
        ui = rules["uireg"]["b"][str(B)]
        if int(ui["steps"]) == 0 or ui["lam"] is None:
            built["p0_ui_reg"] = ({mc.COLD: cold, mc.INITIAL: initial, B: mc.np_map(
                cache["p0"][cseed][B], prior_b, mean_b, std_b, mc._end(cache, B))}, "final_only",
                                  {**ui, "init": "p0", "skipped_fit": True})
        else:
            tab_ui = _run_uireg(world, sd, mc.calibration_indices(sd.bank.n_trials, B),
                                cache["p0"][cseed][B], prior_b, mean_b, std_b, ui["lam"], ui["lr"], ui["steps"])
            built["p0_ui_reg"] = ({mc.COLD: cold, mc.INITIAL: initial, B: mc.np_map(
                tab_ui, prior_b, mean_b, std_b, mc._end(cache, B))}, "final_only",
                                  {**ui, "init": "p0", "center": "p0"})
        for method, payload in built.items():
            maps, install, cfg = payload
            ref_flag = method in ("r2", "r3", "p0_ui_reg")
            proto_name = "causal_first_B"
            _emit_decode(caller, sd, label, maps, install, spans, B, q, latencies, method, cseed,
                         seed, pool, target, gaps, proto_name, ref_flag, cfg, bwd_candidates, hash0,
                         rows, seen)
        for method, info in notes.items():
            _emit_unavailable(method, cseed, seed, pool, target, B, latencies, gaps, info, rows, seen)
    # Legacy prior-init ui_reg, stored random-draw hyperparameters, causal first-B eval.
    # One row per backbone seed (no calibrator).
    leg = legacy_ui[str(B)]
    if all(("ui_reg_legacy", pool, seed, -1, target, int(B), int(lat)) not in seen for lat in latencies):
        mean_b, std_b = cache["norm"][B]
        tab = _run_uireg(world, sd, mc.calibration_indices(sd.bank.n_trials, B), prior_w, prior_b,
                         mean_b, std_b, leg["lam"], leg["lr"], leg["steps"], center=prior_w)
        maps = {mc.COLD: cold, mc.INITIAL: initial, B: mc.np_map(tab, prior_b, mean_b, std_b, mc._end(cache, B))}
        _emit_decode(caller, sd, label, maps, "final_only", spans, B, q, latencies, "ui_reg_legacy", -1,
                     seed, pool, target, gaps, "random_draw_legacy", True,
                     {"lam": leg["lam"], "lr": leg["lr"], "steps": leg["steps"], "init": "prior",
                      "center": "prior"}, _BWD["n"], hash0, rows, seen)
    if rows:
        mc.append_jsonl(path, rows)
    if world.trunk.trunk_hash != hash0:
        raise AssertionError("trunk hash changed")
    log(f"[eval] s{seed} {pool} {target} B={B} rows={len(rows)}")


def _cons_eta(spec):
    gate = spec.get("session_gate") or {}
    if gate.get("deploy"):
        return float(spec["eta"]), float(spec["rho"])
    return 0.0, 0.0


def _wrap(body, cold, initial, B):
    maps = {mc.COLD: cold, mc.INITIAL: initial, **body}
    missing = [k for k in range(1, B + 1) if k not in maps]
    if missing:
        raise AssertionError(f"missing commits {missing[:6]}")
    return maps


def _put_loop(built, notes, label, caller, sd, prior_t, cache, cseed, B, prior_b, jbar,
              eta, rho, alpha, kind, spec, cold, initial):
    try:
        body, info = mc.rebased_maps(caller, sd, prior_t, cache, cseed, B, eta=eta, rho=rho, jbar=jbar,
                                     alpha=alpha, kind=kind, prior_b=prior_b)
    except RuntimeError as exc:
        notes[label] = {"unavailable": True, "reason": str(exc)}
        return
    built[label] = (_wrap(body, cold, initial, B), "anytime",
                    {**info, "alpha": float(alpha), "kind": kind,
                     "gate": spec.get("mode"),
                     "fallback_base": spec.get("fallback_base"),
                     "session_gate_deploy": (spec.get("session_gate") or {}).get("deploy"),
                     "n_positive_sessions": (spec.get("session_gate") or {}).get("n_positive_sessions")})


def _emit_decode(caller, sd, label, maps, install, spans, B, q, latencies, method, cseed,
                 seed, pool, target, gaps, proto, reference, cfg, bwd, hash0, rows, seen):
    # Loops return commits 1..B only.  Wrap with the cold/initial already stored on p_prior
    # by looking at maps: if COLD is missing, the caller must have included it.
    if mc.COLD not in maps:
        raise AssertionError(f"{method} maps missing the cold key; wrap before emit")
    jobs = []
    metas = []
    for lat in latencies:
        key = (method, pool, int(seed), int(cseed), target, int(B), int(lat))
        if key in seen:
            continue
        segs = mc.install_check(maps, spans, B, sd.n_bins, lat, install)
        on_final, n_q = mc.query_bins_on_commit(segs, q.detach().cpu().numpy(), B)
        jobs.append({"segments": segs, "maps": maps, "qmask": q})
        metas.append((int(lat), int(on_final), int(n_q)))
    if not jobs:
        return
    scores = mc.decode_jobs(caller, sd, label, jobs)
    for (lat, on_final, n_q), r2 in zip(metas, scores):
        rows.append({"method": method, "pool": pool, "seed": int(seed), "cseed": int(cseed),
                     "target": target, "date": mc.ymd(target), "B": int(B), "latency": int(lat),
                     "r2": float(r2), "unavailable": False, "n_query_bins": n_q,
                     "query_bins_on_final": on_final, "n_usable": len(spans),
                     "gap_days_trunk": gaps["gap_days_trunk"], "gap_days_source": gaps["gap_days_source"],
                     "gap_bin": gaps["gap_bin"], "latest_source": gaps["latest_source"],
                     "install": install, "reference": bool(reference), "selection_protocol": proto,
                     "h2_backward_calls_before_row": int(bwd), "h1_trunk_hash": hash0[:16],
                     "cfg": mc._jsonable(cfg)})
        seen.add((method, pool, int(seed), int(cseed), target, int(B), int(lat)))


def _emit_unavailable(method, cseed, seed, pool, target, B, latencies, gaps, info, rows, seen):
    for lat in latencies:
        key = (method, pool, int(seed), int(cseed), target, int(B), int(lat))
        if key in seen:
            continue
        rows.append({"method": method, "pool": pool, "seed": int(seed), "cseed": int(cseed),
                     "target": target, "date": mc.ymd(target), "B": int(B), "latency": int(lat),
                     "r2": None, "unavailable": True, "gap_days_trunk": gaps["gap_days_trunk"],
                     "gap_days_source": gaps["gap_days_source"], "gap_bin": gaps["gap_bin"],
                     "selection_protocol": "unavailable", "cfg": mc._jsonable(info)})
        seen.add(key)


def _run_uireg(world, sd, idx, table0, prior_b, mean, std, lam, lr, steps, center=None):
    mean_t = torch.as_tensor(np.asarray(mean, np.float32), device=sd.device)
    std_t = torch.as_tensor(np.asarray(std, np.float32), device=sd.device)
    xz = ((sd.x32 - mean_t) / std_t.clamp_min(1e-8)).detach().cpu().numpy()
    ym, ys = world.label_stats
    y_norm = (sd.record.behavior - ym) / ys
    windows = [sd.trials[int(i)] for i in idx]
    centre = np.asarray(table0 if center is None else center, np.float64)
    traj = m2.ui_reg_trajectory_m2(
        world.trunk, xz, y_norm, windows, np.asarray(table0, np.float64), np.asarray(prior_b, np.float64),
        centre, lam=float(lam), lr=float(lr), snapshots=(int(steps),), device=world.device, seed=0,
        valid_mask=sd.valid_np)
    return traj[int(steps)]


def cmd_select(args):
    what = args.what
    if what == "cca":
        cmd_select_cca(args)
    elif what == "closed":
        cmd_select_closed(args)
    elif what == "refine":
        cmd_select_refine(args)
    elif what == "uireg":
        cmd_select_uireg(args)
    elif what == "all":
        cmd_select_cca(args)
        cmd_select_closed(args)
        cmd_select_refine(args)
        cmd_select_uireg(args)
    else:
        raise SystemExit(what)


def cmd_all(args):
    cmd_prepare(args)
    cmd_select_cca(args)
    cmd_select_closed(args)
    cmd_select_refine(args)
    cmd_select_uireg(args)
    # Fast fail before the long eval, when the unit tests can already run.
    import subprocess
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{ROOT}/.tools/mamba_deps:{ROOT}"
    env.setdefault("TRITON_CACHE_DIR", "/tmp/triton_m2causal")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_m2_causal.py"], cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"tests failed ({proc.returncode}); not starting eval")
    cmd_eval(args)
    cmd_analyze(args)


def cmd_analyze(_args):
    from scripts.analyze_m2_causal import main as analyze_main
    analyze_main()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pools", default=None)
    common.add_argument("--seeds", default=None)
    common.add_argument("--budgets", default=None)
    common.add_argument("--targets", default=None)
    common.add_argument("--latency", default=None)
    sub.add_parser("prepare", parents=[common])
    s = sub.add_parser("select", parents=[common])
    s.add_argument("--what", required=True, choices=("cca", "closed", "refine", "uireg", "all"))
    sub.add_parser("eval", parents=[common])
    sub.add_parser("analyze", parents=[common])
    sub.add_parser("all", parents=[common])
    args = p.parse_args()
    {"prepare": cmd_prepare, "select": cmd_select, "eval": cmd_eval,
     "analyze": cmd_analyze, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
