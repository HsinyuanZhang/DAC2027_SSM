#!/usr/bin/env python
"""Streaming fixed-feedback refinement (S1-FIR and eligibility traces) on the P3 protocol.

Subcommands (resumable; each finished key is skipped):
  exactness     algebraic reordering on one real session (fixed trajectory)
  fit           exponential fit of the source Jacobian, per backbone / lag
  select        eta (and the S1+R1s ridge) on the causal source set only
  eval-drift    12 sub-M drift targets, month causal rules, month P0
  eval-year     20150511 with the year OOS rules and the year P0
  eval-subc     sub-C month targets of sc_0629 and sc_0708
  cost          analytic MAC / SRAM table (no target data)

Target-side candidate paths call stream_f7.stream_refine, which runs under
torch.no_grad.  Stored P0 / R2 / R3 rows are not recomputed; the analyzer joins them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P3_FIX = ROOT / "results/population_calib_p3_20261016/fix"
# Do not export PC_OUT here.  run_pc3_refine binds its own OUT at import; we point
# that attribute at the fix directory after import and log the override.

from scripts import run_drift_curve as drift                      # noqa: E402
from scripts import run_pc3_refine as pc                          # noqa: E402
from scripts import run_population_calib as rpc                   # noqa: E402

_pc_out_default = pc.OUT
pc.OUT = P3_FIX
print(f"[stream_f7] pc.OUT {_pc_out_default} -> {pc.OUT}", flush=True)
from ssm_decode import fwd_refine as fr                           # noqa: E402
from ssm_decode import stream_f7 as sf                            # noqa: E402
from ssm_decode.atomic_jsonl import append_jsonl, read_jsonl      # noqa: E402

OUT = ROOT / "results/stream_f7_20261019"
EXP_OUT = ROOT / "results/expansion_round1/x/expansion_out"
FINAL_DATES = {"20150623", "20150625", "20150626"}
YEAR_CORE = "20150511"
MODES = (4, 8)


def gpu_ok():
    if not torch.cuda.is_available():
        raise SystemExit("cuda:0 is not available; stopping (do not fall back to another GPU)")
    return torch.device("cuda:0")


def trials_from_episode(ep):
    y, valid = ep.sd.y_norm, ep.sd.valid
    out = []
    for a, b in ep.windows:
        out.append((ep.xz_all[a:b], y[a:b].to(dtype=ep.xz_all.dtype), valid[a:b].to(dtype=ep.xz_all.dtype)))
    return out


def decode_prefix(ep, table, want_h=False):
    with torch.no_grad():
        if want_h:
            y, h = fr.run_trunk(ep.base, ep.xz.unsqueeze(0), table, want_h=True)
            return y[0], h[0]
        return fr.run_trunk(ep.base, ep.xz.unsqueeze(0), table)[0], None


def r2_of_pred(ep, pred) -> float:
    full = torch.zeros(ep.sd.n_bins, pred.shape[1], dtype=pred.dtype, device=pred.device)
    full[:ep.nb] = pred
    return float(rpc.r2_physical(ep.world, ep.sd, full, ep.q)[0])


def apply_readout(h, v_aug):
    hb = torch.cat([h, torch.ones_like(h[:, :1])], -1)
    return hb @ v_aug.to(dtype=hb.dtype, device=hb.device).transpose(0, 1)


def score_table(ep, table, readout=None, kind=None, lam=None, basis=None) -> float:
    if table is None or not torch.isfinite(table).all():
        return -1.0
    if readout is None:
        pred, _ = decode_prefix(ep, table, want_h=False)
    else:
        _, h = decode_prefix(ep, table, want_h=True)
        v = ep.v_from(readout, lam, kind, basis)
        pred = apply_readout(h, v)
    if pred is None or not torch.isfinite(pred).all():
        return -1.0
    r2 = r2_of_pred(ep, pred)
    return r2 if math.isfinite(r2) else -1.0


def disjoint(ep) -> bool:
    used = torch.zeros(ep.sd.n_bins, dtype=torch.bool)
    for a, b in ep.windows:
        used[a:b] = True
    return bool(not (used & ep.q.cpu()).any())


def load_fit(path: Path):
    blob = torch.load(path, weights_only=False)
    return blob


def fit_one(jbar_by_lag: dict, tag: str, path: Path):
    """jbar_by_lag: {L_rule: tensor [L+1, D, K]} source kernel, already averaged."""
    if path.exists():
        print(f"[fit] {tag} exists", flush=True)
        return load_fit(path)
    out = {"tag": tag, "modes": {}, "causality": sf.CAUSALITY}
    for L, jb in sorted(jbar_by_lag.items()):
        out["modes"][str(L)] = {}
        for m in MODES:
            t0 = time.perf_counter()
            fit = sf.fit_exponential_kernel(jb, m, seed=L * 10 + m)
            out["modes"][str(L)][str(m)] = {
                "rel_frobenius": fit["rel_frobenius"],
                "rho": [float(v) for v in fit["rho"].tolist()],
                "rho_abs_max": fit["rho_abs_max"],
                "n_lags": fit["n_lags"],
                "init": fit["init"],
                "C": fit["C"].cpu().contiguous(),
            }
            print(f"[fit] {tag} L={L} M={m} rel={fit['rel_frobenius']:.4f} "
                  f"t={time.perf_counter() - t0:.1f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def poles_of(fit_blob, L, m):
    block = fit_blob["modes"][str(L)][str(m)]
    rho = torch.tensor(block["rho"], dtype=torch.float64)
    return rho, block["C"], float(block["rel_frobenius"])


def r2_rules(stratum, sim_set):
    pc.set_variant("base")
    return json.loads(pc.selection_file(stratum, sim_set).read_text())["r2"]


def cmd_exactness(_args):
    gpu_ok()
    pc.set_variant("base")
    world = pc.load_world("month", True)
    p0 = pc.load_p0(world, "month")
    name = world.ids[0]
    sd, prior = world.sessions[name], world.priors[name]
    idx = rpc.draw_trials(name, 4, 0, sd.bank.n_trials)
    ep = pc.p0_episode(world, sd, prior, p0[0], idx)
    jb = pc.jbar_for("month", 32)
    g, q, n, r = fr.fj_gradient(ep.base, ep.X, ep.Y, ep.M, ep.table0, jb)
    # Same stored trajectory: float64 online sum versus the batch convolution.
    u64 = sf.fir_data_term(ep.X.double(), r.double(), jb.double())
    batch64 = sf.batch_data_term(ep.X.double(), r.double(), jb.double())
    diff64 = float((u64 - batch64).abs().max())
    # Same residual, sliced to each true window (drop end padding only).  This is the
    # trial-boundary identity; it does not re-run the trunk.
    u_slice = torch.zeros_like(u64)
    y_pad = None
    with torch.no_grad():
        y_pad = fr.run_trunk(ep.base, ep.X, ep.table0)
    fwd_pad = 0.0
    fwd_batch = 0.0
    for i, (a, b) in enumerate(ep.windows):
        ln = b - a
        u_slice = u_slice + sf.fir_data_term(ep.X[i, :ln].double(), r[i, :ln].double(), jb.double())
        X = ep.xz_all[a:b]
        with torch.no_grad():
            y_short = fr.run_trunk(ep.base, X.unsqueeze(0), ep.table0)[0]
            y_alone = fr.run_trunk(ep.base, ep.X[i:i + 1], ep.table0)[0]
        fwd_pad = max(fwd_pad, float((y_short - y_alone[:ln]).abs().max()))
        fwd_batch = max(fwd_batch, float((y_pad[i, :ln] - y_alone[:ln]).abs().max()))
    diff_slice = float((u_slice - u64).abs().max())
    # Single-trial forwards are not bit-identical to the batched kernel R2 uses.
    # Record the one-step table gap; it is a numeric diagnostic, not the algebra check.
    trials = trials_from_episode(ep)
    eta, rho = 3.0, 0.1
    frozen = sf.frozen_fir_update(ep.base, trials, ep.table0, jb, eta, rho)
    one = fr.fj_refine(ep.base, ep.X, ep.Y, ep.M, ep.table0, jb, eta, rho / eta, [1])[1]
    diff_step = float((frozen - one).abs().max())
    passed = bool(diff64 < 1e-8 and diff_slice < 1e-8 and fwd_pad < 1e-8)
    rec = {"session": name, "B": 4, "draw": 0, "seed": 0, "L": 32, "n_valid": float(n),
           "max_abs_fir_vs_batch_f64": diff64,
           "max_abs_sliced_window_vs_padded_batch_f64": diff_slice,
           "max_abs_unpadded_forward_vs_padded_prefix": fwd_pad,
           "max_abs_single_vs_batched_forward": fwd_batch,
           "max_abs_fj_gradient_f32_vs_f64_reorder": float((u64 - (g * n).double()).abs().max()),
           "max_abs_q": float((q - sf.valid_sum_x2(ep.X, ep.M)[1] / n).abs().max()),
           "max_abs_frozen_single_forward_vs_batched_fj_step": diff_step,
           "passed": passed,
           "note": (
               "Algebraic reordering is judged on one stored trajectory (f64). "
               "End-padding does not change the causal prefix. "
               "A single-trial forward is not bit-identical to the batched R2 kernel; "
               "that gap is reported separately and is not an algebraic failure."
           )}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exactness_real.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)
    if not passed:
        raise SystemExit("real-data reordering check failed")


def _jbar_map_subm(stratum, lags):
    pc.set_variant("base")
    return {L: pc.jbar_for(stratum, L).detach().cpu() for L in lags}


def cmd_fit(args):
    gpu_ok()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.which in ("subm", "all"):
        # month and year share the trunk and the six source sessions; fit once.
        jb = _jbar_map_subm("month", (8, 16, 32))
        fit_one(jb, "subm", OUT / "fit_subm.pt")
    if args.which in ("subc", "all"):
        _fit_subc()


def _subc_contexts(bids):
    from scripts import run_expansion as ex
    st = ex.Store(str(EXP_OUT), None)
    ns = SimpleNamespace(bid=",".join(bids), tier="primary", exclude_apst_dev=False, device="cuda:0")
    bbs = ex.select_backbones(ns)
    ex.register(bbs, st)
    ctxs = {}
    for bb in bbs:
        ex.guard_backbone(bb)
        ctxs[bb.bid] = (bb, ex.Ctx(bb, st, ns, load_sessions=True))
    return ctxs


def _fit_subc():
    from scripts.run_expansion import average_jbar
    for bid in ("sc_0629", "sc_0708"):
        sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
        lags = sorted({int(sel["b"][str(B)]["r2"]["L"]) for B in (1, 4, 8, 16, 33)})
        blob = torch.load(EXP_OUT / f"jacobian/{bid}.pt", weights_only=False)
        jmap = {L: average_jbar(blob, L).cpu() for L in lags}
        fit_one(jmap, bid, OUT / f"fit_{bid}.pt")


def _score_grid(ep, trials, kind, eta_rho_pairs, jbar=None, poles=None, coeff=None, basis=None,
                with_readout=False):
    """One streaming pass per eta.  Readout lambdas share that pass (no extra forward)."""
    s1, r1 = [], []
    for eta, rho in eta_rho_pairs:
        ans = sf.stream_refine(ep.base, trials, ep.table0, eta, rho, kind=kind, jbar=jbar,
                               poles=poles, coeff=coeff, collect_readout=with_readout)
        pred, h = decode_prefix(ep, ans["table"], want_h=with_readout)
        s1.append({"eta": float(eta), "rho": float(rho), "r2": r2_of_pred(ep, pred)})
        if with_readout:
            for kind_r in pc.READOUT_KINDS:
                for lam in pc.RIDGE_LAMS:
                    v = ep.v_from(ans["readout"], lam, kind_r, basis)
                    r1.append({"eta": float(eta), "rho": float(rho), "kind": kind_r, "lam": float(lam),
                               "r2": r2_of_pred(ep, apply_readout(h, v))})
    return s1, r1


def _run_select(tag, episode_iter, path_jsonl, fit_blob, rule_of_B, n_expected):
    """episode_iter yields (session, seed, B, draw, ep, jbar_full_cache dict L->tensor)."""
    seen = {(r["session"], r["seed"], r["B"], r["draw"]) for r in read_jsonl(path_jsonl)}
    t0 = time.perf_counter()
    for session, seed, B, draw, ep, jb_cache in episode_iter():
        if (session, seed, B, draw) in seen:
            continue
        rule = rule_of_B(B)
        L, rho = int(rule["L"]), float(rule["rho"])
        jb = jb_cache[L]
        trials = trials_from_episode(ep)
        pairs = [(eta, rho) for eta in pc.ETAS]
        s1, r1 = _score_grid(ep, trials, "fir", pairs, jbar=jb, basis=ep._basis, with_readout=True)
        tr = {}
        for m in MODES:
            poles, coeff, rel = poles_of(fit_blob, L, m)
            rows, _ = _score_grid(ep, trials, "trace", pairs, poles=poles, coeff=coeff, with_readout=False)
            tr[str(m)] = rows
            tr[str(m) + "_rel"] = rel
        rec = {"tag": tag, "session": session, "seed": int(seed), "B": int(B), "draw": int(draw),
               "L": L, "rho": rho, "eta_r2": float(rule["eta"]), "S_r2": int(rule["S"]),
               "p0": float(ep.r2(ep.table0)), "s1": s1, "s1_r1s": r1, "trace": {str(m): tr[str(m)] for m in MODES},
               "fit_rel": {str(m): tr[str(m) + "_rel"] for m in MODES}}
        append_jsonl(path_jsonl, [rec])
        seen.add((session, seed, B, draw))
        print(f"[select {tag}] {session[-8:]} B={B} s{seed} d{draw} p0={rec['p0']:.3f} "
              f"t={time.perf_counter() - t0:.0f}s", flush=True)
    rows = read_jsonl(path_jsonl)
    _write_selection(tag, rows, path_jsonl.with_name(f"select_{tag}.json"), n_expected)


def _mean_groups(items, keyfn):
    acc = {}
    for it in items:
        acc.setdefault(keyfn(it), []).append(float(it["r2"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}, {k: len(v) for k, v in acc.items()}


def _best(means: dict):
    return max(sorted(means), key=lambda k: means[k])


def _write_selection(tag, rows, path: Path, n_expected):
    byB = {}
    for r in rows:
        byB.setdefault(int(r["B"]), []).append(r)
    out = {"tag": tag, "n_episodes": {str(B): len(v) for B, v in sorted(byB.items())}, "b": {}}
    for B, group in sorted(byB.items()):
        if n_expected is not None and len(group) != n_expected:
            raise SystemExit(f"{tag} B={B}: {len(group)} episodes, expected {n_expected}")
        s1_items = [it for r in group for it in r["s1"]]
        means, counts = _mean_groups(s1_items, lambda it: float(it["eta"]))
        eta = float(_best(means))
        r1_items = [it for r in group for it in r["s1_r1s"]]
        m1, _ = _mean_groups(r1_items, lambda it: (float(it["eta"]), it["kind"], float(it["lam"])))
        eta1, kind, lam = _best(m1)
        traces = {}
        for m in MODES:
            items = [it for r in group for it in r["trace"][str(m)]]
            mt, _ = _mean_groups(items, lambda it: float(it["eta"]))
            traces[str(m)] = {"eta": float(_best(mt)), "source_r2": mt[float(_best(mt))],
                              "fit_rel": float(sum(r["fit_rel"][str(m)] for r in group) / len(group)),
                              "grid": {str(k): v for k, v in sorted(mt.items())}}
        # source score of the R2 eta (same rho), for the "same eta" comparison
        eta_r2 = float(group[0]["eta_r2"])
        out["b"][str(B)] = {
            "L": int(group[0]["L"]), "rho": float(group[0]["rho"]), "S_r2": int(group[0]["S_r2"]),
            "eta_r2": eta_r2,
            "s1": {"eta": eta, "source_r2": means[eta], "grid": {str(k): v for k, v in sorted(means.items())},
                   "n": counts[eta]},
            "s1_r1s": {"eta": float(eta1), "kind": kind, "lam": float(lam), "source_r2": m1[(eta1, kind, lam)]},
            "trace": traces,
            "p0_source_r2": float(sum(r["p0"] for r in group) / len(group)),
            "sessions": sorted({r["session"] for r in group}),
        }
    path.write_text(json.dumps(out, indent=1))
    print(f"[select {tag}] wrote {path}", flush=True)


def _basis_on(ep, basis):
    ep._basis = basis
    return ep


def cmd_select(args):
    gpu_ok()
    pc.set_variant("base")
    OUT.mkdir(parents=True, exist_ok=True)
    fit_blob = load_fit(OUT / "fit_subm.pt")
    which = args.which
    if which in ("month", "all"):
        rules = r2_rules("month", "oosc")
        world, names = pc.sim_world("month", "oosc")
        p0 = pc.load_p0(world, "month")
        basis = pc.load_basis("month")
        jb_cache = {L: pc.jbar_for("month", L) for L in (8, 16, 32)}

        def episodes():
            for name in names:
                sd, prior = world.sessions[name], world.priors[name]
                for seed in pc.SEEDS:
                    for B in rpc.B_GRID:
                        for draw in range(pc.SEL_DRAWS):
                            idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                            ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx, pc.SIM_MAX_BINS), basis)
                            yield name, seed, B, draw, ep, jb_cache

        n = len(names) * len(pc.SEEDS) * pc.SEL_DRAWS
        _run_select("month", episodes, OUT / "select_month_episodes.jsonl", fit_blob,
                    lambda B: rules[str(B)]["cfg"], n)
    if which in ("year", "all"):
        rules = r2_rules("year", "oos")
        world, names = pc.sim_world("year", "oos")
        p0 = pc.load_p0(world, "year")
        basis = pc.load_basis("year")
        jb_cache = {L: pc.jbar_for("year", L) for L in (8, 16, 32)}

        def episodes_y():
            for name in names:
                sd, prior = world.sessions[name], world.priors[name]
                for seed in pc.SEEDS:
                    for B in rpc.B_GRID:
                        for draw in range(pc.SEL_DRAWS):
                            idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                            ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx, pc.SIM_MAX_BINS), basis)
                            yield name, seed, B, draw, ep, jb_cache

        n = len(names) * len(pc.SEEDS) * pc.SEL_DRAWS
        _run_select("year", episodes_y, OUT / "select_year_episodes.jsonl", fit_blob,
                    lambda B: rules[str(B)]["cfg"], n)


def cmd_select_subc(_args):
    gpu_ok()
    from scripts.run_expansion import average_jbar
    OUT.mkdir(parents=True, exist_ok=True)
    for bid in ("sc_0629", "sc_0708"):
        ctxs = _subc_contexts([bid])
        bb, ctx = ctxs[bid]
        sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
        assert sel["selection_session"] == bb.val_session
        fit_blob = load_fit(OUT / f"fit_{bid}.pt")
        basis = torch.load(EXP_OUT / f"basis/{bid}.pt", weights_only=False)["U"].to(ctx.device)
        blob = torch.load(EXP_OUT / f"jacobian/{bid}.pt", weights_only=False)
        lags = sorted({int(sel["b"][str(B)]["r2"]["L"]) for B in sel["b"]})
        jb_cache = {L: average_jbar(blob, L).to(ctx.device) for L in lags}
        name = bb.val_session
        sd, prior = ctx.world.sessions[name], ctx.world.priors[name]
        # 4 draws x 2 seeds, matching the round-1 selection episodes (tag expsel).
        n_draw = int(sel["grid"]["sel_draws"])

        def episodes(name=name, sd=sd, prior=prior, ctx=ctx, basis=basis, jb_cache=jb_cache, n_draw=n_draw):
            for seed in (0, 1):
                for B in (1, 4, 8, 16, 33):
                    for draw in range(n_draw):
                        idx = pc.draw_sim(name, f"expsel-s{seed}", B, draw, sd)
                        ep = _basis_on(pc.p0_episode(ctx.world, sd, prior, ctx.p0[seed], idx, pc.SIM_MAX_BINS), basis)
                        yield name, seed, B, draw, ep, jb_cache

        def rule_of(B, sel=sel):
            r = sel["b"][str(B)]["r2"]
            return {"L": r["L"], "eta": r["eta"], "rho": r["rho"], "S": r["S"]}

        _run_select(bid, episodes, OUT / f"select_{bid}_episodes.jsonl", fit_blob, rule_of, n_draw * 2)


def _methods_for(ep, rule, chosen, fit_blob, basis):
    """Build the five streaming specs from the R2 rule and the source-selected etas."""
    L, rho = int(rule["L"]), float(rule["rho"])
    jb = chosen["jbar"]
    specs = [
        {"method": "s1_r2eta", "kind": "fir", "eta": float(rule["eta"]), "rho": rho, "jbar": jb,
         "cfg": {"L": L, "eta": float(rule["eta"]), "rho": rho, "S_r2": int(rule["S"]), "selected": "r2"}},
        {"method": "s1", "kind": "fir", "eta": float(chosen["s1"]["eta"]), "rho": rho, "jbar": jb,
         "cfg": {"L": L, "eta": float(chosen["s1"]["eta"]), "rho": rho, "source_r2": chosen["s1"]["source_r2"],
                 "selected": "eta_on_source"}},
    ]
    r1 = chosen["s1_r1s"]
    specs.append({"method": "s1_r1s", "kind": "fir", "eta": float(r1["eta"]), "rho": rho, "jbar": jb,
                  "readout": True, "lam": float(r1["lam"]), "readout_kind": r1["kind"], "basis": basis,
                  "cfg": {"L": L, "eta": float(r1["eta"]), "rho": rho, "kind": r1["kind"], "lam": float(r1["lam"]),
                          "source_r2": r1["source_r2"], "selected": "eta_kind_lam_on_source"}})
    for m in MODES:
        poles, coeff, rel = poles_of(fit_blob, L, m)
        eta = float(chosen["trace"][str(m)]["eta"])
        specs.append({"method": f"tr{m}", "kind": "trace", "eta": eta, "rho": rho, "poles": poles, "coeff": coeff,
                      "cfg": {"L": L, "M": m, "eta": eta, "rho": rho, "fit_rel": rel,
                              "source_r2": chosen["trace"][str(m)]["source_r2"], "selected": "eta_on_source"}})
    return specs


def _eval_episode(ep, specs, bwd):
    trials = trials_from_episode(ep)
    bwd["n"] = 0
    rows = []
    for spec in specs:
        ans = sf.stream_refine(ep.base, trials, ep.table0, spec["eta"], spec["rho"], kind=spec["kind"],
                               jbar=spec.get("jbar"), poles=spec.get("poles"), coeff=spec.get("coeff"),
                               collect_readout=bool(spec.get("readout")))
        if spec.get("readout"):
            r2 = score_table(ep, ans["table"], ans["readout"], spec["readout_kind"], spec["lam"], spec["basis"])
        else:
            r2 = score_table(ep, ans["table"])
        rows.append({"method": spec["method"], "r2": r2, "cfg": spec["cfg"],
                     "n_commits": ans["n_commits"], "h2_backward_calls": bwd["n"]})
    if bwd["n"] != 0:
        raise RuntimeError(f"target-side backward calls: {bwd['n']}")
    return rows


def _eval_driver(path: Path, jobs, bwd_box):
    """jobs is an iterator of (key_tuple, meta_dict, ep, specs)."""
    seen = {(r["method"], r["target"], r["B"], r["draw"], r["seed"]) for r in read_jsonl(path)}
    t0 = time.perf_counter()
    n = 0
    for key, meta, ep, specs in jobs:
        need = [s for s in specs if (s["method"], meta["target"], meta["B"], meta["draw"], meta["seed"]) not in seen]
        if not need:
            continue
        base = {"h4_budget": int(meta["B"]), "h4_query_disjoint": disjoint(ep),
                "n_query_bins": int(ep.q.sum()), "replay_bins": int(ep.replay_bins)}
        for row in _eval_episode(ep, need, bwd_box):
            rec = {**meta, **base, **row}
            append_jsonl(path, [rec])
            seen.add((rec["method"], rec["target"], rec["B"], rec["draw"], rec["seed"]))
        n += 1
        if n % 4 == 0:
            print(f"[eval] {meta['target'][-8:]} B={meta['B']} d={meta['draw']} s{meta['seed']} "
                  f"t={time.perf_counter() - t0:.0f}s", flush=True)


def _patch():
    return pc._patch_autograd_counter()


def cmd_eval_drift(args):
    gpu_ok()
    pc.set_variant("base")
    dates = drift.validate_targets(drift.TARGET_DATES, ["20140203", "20140217", "20140218", "20140303",
                                                        "20140304", "20140306"])
    if args.targets:
        want = set(args.targets.split(","))
        dates = [d for d in dates if d in want]
    sel = json.loads(pc.selection_file("month", "oosc").read_text())
    chosen = json.loads((OUT / "select_month.json").read_text())["b"]
    fit_blob = load_fit(OUT / "fit_subm.pt")
    world = pc.load_world("month", False)
    p0 = pc.load_p0(world, "month")
    basis = pc.load_basis("month")
    jb_cache = {}
    og, ob = _patch()
    try:
        path = OUT / "rows_drift.jsonl"

        def jobs():
            for d in dates:
                target = drift.PREFIX + d
                if d in FINAL_DATES or d == "20140307":
                    raise SystemExit(f"refusing target {d}")
                sd, prior = world.target(target)
                for B in rpc.B_GRID:
                    rule = sel["r2"][str(B)]["cfg"]
                    L = int(rule["L"])
                    if L not in jb_cache:
                        jb_cache[L] = pc.jbar_for("month", L)
                    block = dict(chosen[str(B)])
                    block["jbar"] = jb_cache[L]
                    for draw in range(args.draws):
                        idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                        for seed in pc.SEEDS:
                            ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx), basis)
                            meta = {"stratum": "subm_drift", "bid": "month", "target": target, "date": d,
                                    "B": B, "draw": draw, "seed": seed, "rule_set": "month_oosc"}
                            yield (target, B, draw, seed), meta, ep, _methods_for(ep, rule, block, fit_blob, basis)
                del sd, prior

        _eval_driver(path, jobs(), pc._BWD)
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob


def cmd_eval_year(args):
    gpu_ok()
    pc.set_variant("base")
    if YEAR_CORE in FINAL_DATES:
        raise SystemExit("year core is a final session")
    sel = json.loads(pc.selection_file("year", "oos").read_text())
    chosen = json.loads((OUT / "select_year.json").read_text())["b"]
    fit_blob = load_fit(OUT / "fit_subm.pt")
    world = pc.load_world("year", False)
    p0 = pc.load_p0(world, "year")
    basis = pc.load_basis("year")
    target = "sub-M_ses-CO-" + YEAR_CORE
    sd, prior = world.target(target)
    og, ob = _patch()
    try:
        path = OUT / "rows_year.jsonl"
        jb_cache = {}

        def jobs():
            for B in rpc.B_GRID:
                rule = sel["r2"][str(B)]["cfg"]
                L = int(rule["L"])
                if L not in jb_cache:
                    jb_cache[L] = pc.jbar_for("year", L)
                block = dict(chosen[str(B)])
                block["jbar"] = jb_cache[L]
                for draw in range(args.draws):
                    idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                    for seed in pc.SEEDS:
                        ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx), basis)
                        meta = {"stratum": "subm_year", "bid": "year", "target": target, "date": YEAR_CORE,
                                "B": B, "draw": draw, "seed": seed, "rule_set": "year_oos"}
                        yield (target, B, draw, seed), meta, ep, _methods_for(ep, rule, block, fit_blob, basis)

        _eval_driver(path, jobs(), pc._BWD)
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob


def cmd_eval_subc(args):
    gpu_ok()
    from scripts.run_expansion import average_jbar
    og, ob = _patch()
    try:
        for bid in ("sc_0629", "sc_0708"):
            ctxs = _subc_contexts([bid])
            bb, ctx = ctxs[bid]
            sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
            chosen_all = json.loads((OUT / f"select_{bid}.json").read_text())["b"]
            fit_blob = load_fit(OUT / f"fit_{bid}.pt")
            basis = torch.load(EXP_OUT / f"basis/{bid}.pt", weights_only=False)["U"].to(ctx.device)
            blob = torch.load(EXP_OUT / f"jacobian/{bid}.pt", weights_only=False)
            jb_cache = {}
            targets = [t for t in bb.targets if t.stratum == "month"]
            if not targets:
                raise SystemExit(f"{bid}: no month targets")
            path = OUT / f"rows_{bid}.jsonl"

            def jobs(bid=bid, ctx=ctx, sel=sel, chosen_all=chosen_all, fit_blob=fit_blob, basis=basis,
                     blob=blob, targets=targets):
                for t in targets:
                    date = t.session[-8:]
                    if date in FINAL_DATES or date == "20140307":
                        raise SystemExit(f"refusing target {t.session}")
                    sd, prior = ctx.world.target(t.session)
                    for B in (1, 4, 8, 16, 33):
                        r = sel["b"][str(B)]["r2"]
                        rule = {"L": r["L"], "eta": r["eta"], "rho": r["rho"], "S": r["S"]}
                        L = int(rule["L"])
                        if L not in jb_cache:
                            jb_cache[L] = average_jbar(blob, L).to(ctx.device)
                        block = dict(chosen_all[str(B)])
                        block["jbar"] = jb_cache[L]
                        for draw in range(args.draws):
                            idx = rpc.draw_trials(t.session, B, draw, sd.bank.n_trials)
                            for seed in (0, 1):
                                ep = _basis_on(pc.p0_episode(ctx.world, sd, prior, ctx.p0[seed], idx), basis)
                                meta = {"stratum": "subc_month", "bid": bid, "target": t.session, "date": date,
                                        "B": B, "draw": draw, "seed": seed,
                                        "rule_set": sel["selection_session"], "gap_days": int(t.gap_days)}
                                yield (t.session, B, draw, seed), meta, ep, _methods_for(ep, rule, block, fit_blob, basis)
                    del sd, prior

            _eval_driver(path, jobs(), pc._BWD)
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob


def cmd_cost(_args):
    """Analytic cost.  Trunk MACs from an instantiated K=10 trunk; N enters only the read-in and the rings."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ssm_decode.mamba3_official import build_mamba3_official
    from ssm_decode.refine_cost import refinement_cost
    trunk = fr.trunk_macs_per_bin(build_mamba3_official(10, 2, 128, 2, 32, 0.0).eval())
    rows = []
    # Headline month rule at B=8 is L=32 (33 taps), S=10.  T=200 is a point estimate for the
    # replay buffer only; per-bin MAC rows do not use it.
    taps, S, B, T, K, D = 33, 10, 8, 200, 10, 2
    for N in (26, 256):
        case = {"name": f"N{N}", "N": N, "K": K, "D": D, "H": 128, "T": T, "bx": 3, "by": 16,
                "trunk": trunk, "bin_ms": 20, "state_bits": 16}
        r2 = refinement_cost(case, B, "r2", steps=S, lags=taps - 1, replay_bins=B * T, staging_bins=T)
        fir = sf.fir_bin_macs(taps, D, K, N)
        for M in (4, 8):
            tr = sf.trace_bin_macs(M, D, K, N)
            sram_fir = sf.stream_sram_bytes(n=N, k=K, n_lags=taps, n_modes=M, include_fir=True,
                                            include_trace=False, staging_bins=T, d=D)
            sram_tr = sf.stream_sram_bytes(n=N, k=K, n_lags=taps, n_modes=M, include_fir=False,
                                           include_trace=True, staging_bins=T, d=D)
            sram_r1 = sf.stream_sram_bytes(n=N, k=K, n_lags=taps, n_modes=M, include_fir=True,
                                           include_trace=False, staging_bins=T, d=D, readout_h=128)
            rows.append({
                "N": N, "K": K, "D": D, "L_taps": taps, "M": M, "S_r2": S, "B": B, "T_point": T,
                "fir_weight_x_act_per_bin": fir["weight_x_act"],
                "fir_act_x_act_per_bin": fir["act_x_act"],
                "fir_total_per_bin": fir["total"],
                "trace_weight_x_act_per_bin": tr["weight_x_act"],
                "trace_act_x_act_per_bin": tr["act_x_act"],
                "trace_total_per_bin": tr["total"],
                "live_trunk_weight_x_act_per_bin": trunk["fixed_weight_macs"] + K * N,
                "live_trunk_act_x_act_per_bin": trunk["act_x_act_macs"],
                "r2_extra_trunk_forwards": S,
                "r2_macs_total_for_BT": r2["macs_total"],
                "r2_replay_buffer_bytes": r2["replay_buffer_bytes"],
                "r2_sram_total_bytes": r2["sram_total"],
                "r2_macs_by_class": r2["macs_by_class"],
                "s1_fir_sram_bytes": sram_fir["total"],
                "s1_fir_sram_detail": {k: v for k, v in sram_fir.items() if k != "total"},
                "trace_sram_bytes": sram_tr["total"],
                "trace_sram_detail": {k: v for k, v in sram_tr.items() if k != "total"},
                "s1_r1s_sram_bytes": sram_r1["total"],
                "s1_r1s_sram_detail": {k: v for k, v in sram_r1.items() if k != "total"},
                "note": (
                    "S1 pays the live trunk once (the calibration trial is the inference forward) plus the "
                    "FIR/trace accumulator.  R2 pays the trunk, the feedback convolution and one outer product "
                    "S times, and keeps the raw replay buffer of all B trials.  "
                    "FIR weight_x_act is Jbar (ROM) times the residual; FIR act_x_act is the lagged outer "
                    "product.  Trace weight_x_act is C_m^T e plus rho*s; trace act_x_act is the outer product "
                    "with the trace.  The one-trial staging buffer is the current trial only."
                ),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    like = []
    for N in (26, 256):
        mem = sf.like_for_like_memory(n=N, k=K, d=D, n_lags=taps, n_modes=4, replay_bins=B * T,
                                      staging_bins=T, state_elems=trunk["state_elems"])
        mem["N"] = N
        like.append(mem)
    (OUT / "cost.json").write_text(json.dumps({"trunk": trunk, "rows": rows, "like_for_like": like}, indent=1))
    print(f"[cost] wrote {OUT / 'cost.json'}", flush=True)


ETAS_J = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
RHOS_J = (0.0, 0.01, 0.03, 0.1, 0.3)


def _raw_trials(sd, idx):
    idx = np.sort(np.asarray(idx))
    return [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype),
             sd.valid[a:b].to(dtype=sd.x32.dtype)) for a, b in (sd.trials[int(i)] for i in idx)]


def _prepare_causal(sd, prior, calib, idx):
    """stream_stats convention: trial 0 uses its own norm; later trials use the prefix."""
    from ssm_decode.population_calib import PopStats, norm_stats
    idx = np.sort(np.asarray(idx))
    cache = {}

    def _calib(nprefix):
        if nprefix not in cache:
            cache[nprefix] = calib.calibrate(PopStats.from_trials(sd.bank, idx[:nprefix]), prior)
        return cache[nprefix]

    def prepare(k, closed):
        if not closed and k == 0:
            mean, std = norm_stats(PopStats.from_trials(sd.bank, idx[:1]))
            return mean, std, None
        res = _calib(k + 1 if closed else k)
        return res.mean, res.std, res.table

    return prepare


def _forward_normed(ep, table, mean, std):
    if table is None or mean is None or not torch.isfinite(table).all():
        return None, None
    xz = (ep.sd.x32[:ep.nb] - mean.to(dtype=ep.sd.x32.dtype)) / std.to(dtype=ep.sd.x32.dtype).clamp_min(1e-8)
    with torch.no_grad():
        y, h = fr.run_trunk(ep.base, xz.unsqueeze(0), table, want_h=True)
    return y[0], h[0]


def _r2_or_neg(ep, pred) -> float:
    if pred is None or not torch.isfinite(pred).all():
        return -1.0
    r2 = r2_of_pred(ep, pred)
    return r2 if math.isfinite(r2) else -1.0


def _score_normed(ep, table, mean, std, readout=None, kind=None, lam=None, basis=None) -> float:
    pred, h = _forward_normed(ep, table, mean, std)
    if readout is not None and h is not None:
        pred = apply_readout(h, ep.v_from(readout, lam, kind, basis))
    return _r2_or_neg(ep, pred)


def _best_eta_rho(rows):
    means, _ = _mean_groups(rows, lambda it: (float(it["eta"]), float(it["rho"])))
    return _best(means), means


def _fallback(best_r2, p0_r2):
    return bool(best_r2 < p0_r2)


def _run_select_fix(tag, episode_iter, path_jsonl, fit_blob, rule_of_B, n_expected):
    seen = {(r["session"], r["seed"], r["B"], r["draw"]) for r in read_jsonl(path_jsonl)}
    t0 = time.perf_counter()
    for session, seed, B, draw, ep, prior, calib, jb_cache in episode_iter():
        if (session, seed, B, draw) in seen:
            continue
        rule = rule_of_B(B)
        L, rho_r2 = int(rule["L"]), float(rule["rho"])
        jb = jb_cache[L]
        trials = trials_from_episode(ep)
        raw = _raw_trials(ep.sd, ep.idx)
        prepare = _prepare_causal(ep.sd, prior, calib, ep.idx)
        p0 = float(ep.r2(ep.table0))
        s1 = []
        for eta in ETAS_J:
            for rho in RHOS_J:
                ans = sf.stream_refine(ep.base, trials, ep.table0, eta, rho, kind="fir", jbar=jb)
                s1.append({"eta": float(eta), "rho": float(rho), "r2": score_table(ep, ans["table"])})
        s1c, ridge = [], []
        for eta in ETAS_J:
            for rho in RHOS_J:
                ans = sf.causal_feedback_loop(ep.base, raw, prior.float(), eta, rho, prepare, kind="fir",
                                              jbar=jb, collect_readout=True)
                pred, h = _forward_normed(ep, ans["table"], ans["mean"], ans["std"])
                s1c.append({"eta": float(eta), "rho": float(rho), "r2": _r2_or_neg(ep, pred)})
                for kind_r in pc.READOUT_KINDS:
                    for lam in pc.RIDGE_LAMS:
                        v = ep.v_from(ans["readout"], lam, kind_r, ep._basis)
                        ridge.append({"eta": float(eta), "rho": float(rho), "kind": kind_r, "lam": float(lam),
                                      "r2": _r2_or_neg(ep, apply_readout(h, v))})
        rec = {"tag": tag, "session": session, "seed": int(seed), "B": int(B), "draw": int(draw),
               "L": L, "rho_r2": rho_r2, "eta_r2": float(rule["eta"]), "p0": p0,
               "s1_joint": s1, "s1c": s1c, "s1c_ridge": ridge}
        append_jsonl(path_jsonl, [rec])
        seen.add((session, seed, B, draw))
        print(f"[select-fix {tag}] {str(session)[-8:]} B={B} s{seed} d{draw} p0={p0:.3f} "
              f"t={time.perf_counter() - t0:.0f}s", flush=True)
    _write_selection_fix(tag, read_jsonl(path_jsonl), path_jsonl.with_name(f"select_{tag}_fix.json"), n_expected)


def _write_selection_fix(tag, rows, path, n_expected):
    byB = {}
    for r in rows:
        byB.setdefault(int(r["B"]), []).append(r)
    out = {"tag": tag, "grid": {"eta": list(ETAS_J), "rho": list(RHOS_J)},
           "n_episodes": {str(B): len(v) for B, v in sorted(byB.items())}, "b": {}}
    for B, group in sorted(byB.items()):
        if n_expected is not None and len(group) != n_expected:
            raise SystemExit(f"{tag} B={B}: {len(group)} episodes, expected {n_expected}")
        p0 = float(sum(r["p0"] for r in group) / len(group))

        def pack(key):
            items = [it for r in group for it in r[key]]
            (eta, rho), means = _best_eta_rho(items)
            src = means[(float(eta), float(rho))]
            return {"eta": float(eta), "rho": float(rho), "source_r2": src,
                    "fallback_p0": _fallback(src, p0),
                    "grid": {f"{a}|{b}": v for (a, b), v in sorted(means.items())}}

        ridge_items = [it for r in group for it in r["s1c_ridge"]]
        m1, _ = _mean_groups(ridge_items, lambda it: (float(it["eta"]), float(it["rho"]), it["kind"], float(it["lam"])))
        eta, rho, kind, lam = _best(m1)
        src = m1[(float(eta), float(rho), kind, float(lam))]
        out["b"][str(B)] = {
            "L": int(group[0]["L"]), "eta_r2": float(group[0]["eta_r2"]), "rho_r2": float(group[0]["rho_r2"]),
            "p0_source_r2": p0, "sessions": sorted({r["session"] for r in group}),
            "s1_joint": pack("s1_joint"), "s1c": pack("s1c"),
            "s1c_ridge": {"eta": float(eta), "rho": float(rho), "kind": kind, "lam": float(lam),
                          "source_r2": src, "fallback_p0": _fallback(src, p0)},
        }
    path.write_text(json.dumps(out, indent=1))
    print(f"[select-fix {tag}] wrote {path}", flush=True)


def _need_np():
    import numpy as np
    return np


def cmd_select_fix(args):
    gpu_ok()
    pc.set_variant("base")
    OUT.mkdir(parents=True, exist_ok=True)
    which = args.which
    if which in ("month", "all"):
        _select_fix_stratum("month", "oosc")
    if which in ("year", "all"):
        _select_fix_stratum("year", "oos")


def _select_fix_stratum(stratum, sim_set):
    rules = r2_rules(stratum, sim_set)
    world, names = pc.sim_world(stratum, sim_set)
    p0 = pc.load_p0(world, stratum)
    basis = pc.load_basis(stratum)
    jb_cache = {L: pc.jbar_for(stratum, L) for L in (8, 16, 32)}
    # Causal selection uses the same P0 calibrator the non-causal path starts from.
    def episodes(world=world, names=names, p0=p0, basis=basis, jb_cache=jb_cache):
        for name in names:
            sd, prior = world.sessions[name], world.priors[name]
            for seed in pc.SEEDS:
                for B in rpc.B_GRID:
                    for draw in range(pc.SEL_DRAWS):
                        idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                        ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx, pc.SIM_MAX_BINS), basis)
                        yield name, seed, B, draw, ep, prior, p0[seed], jb_cache
    n = len(names) * len(pc.SEEDS) * pc.SEL_DRAWS
    _run_select_fix(stratum, episodes, OUT / f"select_{stratum}_fix_episodes.jsonl", None,
                    lambda B, rules=rules: rules[str(B)]["cfg"], n)


def cmd_select_subc_fix(_args):
    gpu_ok()
    from scripts.run_expansion import average_jbar
    OUT.mkdir(parents=True, exist_ok=True)
    for bid in ("sc_0629", "sc_0708"):
        ctxs = _subc_contexts([bid])
        bb, ctx = ctxs[bid]
        sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
        basis = torch.load(EXP_OUT / f"basis/{bid}.pt", weights_only=False)["U"].to(ctx.device)
        blob = torch.load(EXP_OUT / f"jacobian/{bid}.pt", weights_only=False)
        lags = sorted({int(sel["b"][str(B)]["r2"]["L"]) for B in sel["b"]})
        jb_cache = {L: average_jbar(blob, L).to(ctx.device) for L in lags}
        name = bb.val_session
        sd, prior = ctx.world.sessions[name], ctx.world.priors[name]
        n_draw = int(sel["grid"]["sel_draws"])

        def episodes(name=name, sd=sd, prior=prior, ctx=ctx, basis=basis, jb_cache=jb_cache, n_draw=n_draw):
            for seed in (0, 1):
                for B in (1, 4, 8, 16, 33):
                    for draw in range(n_draw):
                        idx = pc.draw_sim(name, f"expsel-s{seed}", B, draw, sd)
                        ep = _basis_on(pc.p0_episode(ctx.world, sd, prior, ctx.p0[seed], idx, pc.SIM_MAX_BINS), basis)
                        yield name, seed, B, draw, ep, prior, ctx.p0[seed], jb_cache

        def rule_of(B, sel=sel):
            r = sel["b"][str(B)]["r2"]
            return {"L": r["L"], "eta": r["eta"], "rho": r["rho"], "S": r["S"]}

        _run_select_fix(bid, episodes, OUT / f"select_{bid}_fix_episodes.jsonl", None, rule_of, n_draw * 2)


def _eval_fix_episode(ep, prior, calib, specs):
    """specs: method, mode in {noncausal, causal}, eta, rho, kind, jbar/poles, fallback."""
    trials = trials_from_episode(ep)
    raw = _raw_trials(ep.sd, ep.idx)
    prepare = _prepare_causal(ep.sd, prior, calib, ep.idx)
    rows = []
    for spec in specs:
        if spec.get("fallback_p0"):
            rows.append({"method": spec["method"], "r2": score_table(ep, ep.table0), "cfg": spec["cfg"],
                         "n_commits": 0})
            continue
        if spec["mode"] == "noncausal":
            ans = sf.stream_refine(ep.base, trials, ep.table0, spec["eta"], spec["rho"], kind=spec["kind"],
                                   jbar=spec.get("jbar"), poles=spec.get("poles"), coeff=spec.get("coeff"),
                                   n_lags=spec.get("n_lags"))
            r2 = score_table(ep, ans["table"])
        else:
            ans = sf.causal_feedback_loop(ep.base, raw, prior.float(), spec["eta"], spec["rho"], prepare,
                                         kind=spec["kind"], jbar=spec.get("jbar"), poles=spec.get("poles"),
                                         coeff=spec.get("coeff"), n_lags=spec.get("n_lags"),
                                         collect_readout=bool(spec.get("readout")))
            if spec.get("readout"):
                r2 = _score_normed(ep, ans["table"], ans["mean"], ans["std"], ans["readout"],
                                   spec["readout_kind"], spec["lam"], spec["basis"])
            else:
                r2 = _score_normed(ep, ans["table"], ans["mean"], ans["std"])
        rows.append({"method": spec["method"], "r2": r2, "cfg": spec["cfg"], "n_commits": len(ep.idx)})
    return rows


def _specs_fix(ep, rule, chosen, fit_blob, basis):
    L = int(rule["L"])
    jb = chosen["jbar"]
    specs = []
    for key, method, mode in (("s1_joint", "s1_joint", "noncausal"), ("s1c", "s1c", "causal")):
        block = chosen[key]
        specs.append({"method": method, "mode": mode, "kind": "fir", "eta": block["eta"], "rho": block["rho"],
                      "jbar": jb, "fallback_p0": block["fallback_p0"],
                      "cfg": {"L": L, "eta": block["eta"], "rho": block["rho"], "fallback_p0": block["fallback_p0"],
                              "source_r2": block["source_r2"], "protocol": mode, "selected": "eta_rho_joint"}})
    ridge = chosen["s1c_ridge"]
    specs.append({"method": "s1c_ridge", "mode": "causal", "kind": "fir", "eta": ridge["eta"], "rho": ridge["rho"],
                  "jbar": jb, "readout": True, "readout_kind": ridge["kind"], "lam": ridge["lam"], "basis": basis,
                  "fallback_p0": ridge["fallback_p0"],
                  "cfg": {"L": L, "eta": ridge["eta"], "rho": ridge["rho"], "kind": ridge["kind"],
                          "lam": ridge["lam"], "fallback_p0": ridge["fallback_p0"],
                          "source_r2": ridge["source_r2"], "protocol": "causal", "selected": "eta_rho_kind_lam"}})
    # Truncated traces share the FIR (eta, rho): the truncated kernel targets the same U.
    for m, mode, method, src in ((4, "noncausal", "tr4_trunc", "s1_joint"), (8, "noncausal", "tr8_trunc", "s1_joint"),
                                 (4, "causal", "tr4c", "s1c"), (8, "causal", "tr8c", "s1c")):
        poles, coeff, rel = poles_of(fit_blob, L, m)
        block = chosen[src]
        specs.append({"method": method, "mode": mode, "kind": "trace", "eta": block["eta"], "rho": block["rho"],
                      "poles": poles, "coeff": coeff, "n_lags": int(jb.shape[0]),
                      "fallback_p0": block["fallback_p0"],
                      "cfg": {"L": L, "M": m, "eta": block["eta"], "rho": block["rho"], "fit_rel": rel,
                              "n_lags": int(jb.shape[0]), "truncated": True, "fallback_p0": block["fallback_p0"],
                              "protocol": mode, "selected": "shared_with_" + src}})
    return specs


def _eval_fix_driver(path, jobs):
    seen = {(r["method"], r["target"], r["B"], r["draw"], r["seed"]) for r in read_jsonl(path)}
    t0 = time.perf_counter()
    n = 0
    for meta, ep, prior, calib, specs in jobs:
        need = [s for s in specs if (s["method"], meta["target"], meta["B"], meta["draw"], meta["seed"]) not in seen]
        if not need:
            continue
        base = {"h4_budget": int(meta["B"]), "h4_query_disjoint": disjoint(ep),
                "n_query_bins": int(ep.q.sum()), "h2_backward_calls": 0}
        for row in _eval_fix_episode(ep, prior, calib, need):
            rec = {**meta, **base, **row}
            append_jsonl(path, [rec])
            seen.add((rec["method"], rec["target"], rec["B"], rec["draw"], rec["seed"]))
        n += 1
        if n % 4 == 0:
            print(f"[eval-fix] {meta['target'][-8:]} B={meta['B']} d={meta['draw']} s{meta['seed']} "
                  f"t={time.perf_counter() - t0:.0f}s", flush=True)


def cmd_eval_drift_fix(args):
    gpu_ok()
    pc.set_variant("base")
    dates = drift.validate_targets(drift.TARGET_DATES, ["20140203", "20140217", "20140218", "20140303",
                                                        "20140304", "20140306"])
    sel = json.loads(pc.selection_file("month", "oosc").read_text())
    chosen = json.loads((OUT / "select_month_fix.json").read_text())["b"]
    fit_blob = load_fit(OUT / "fit_subm.pt")
    world = pc.load_world("month", False)
    p0 = pc.load_p0(world, "month")
    basis = pc.load_basis("month")
    jb_cache = {}
    path = OUT / "rows_drift_fix.jsonl"

    def jobs():
        for d in dates:
            if d in FINAL_DATES or d == "20140307":
                raise SystemExit(d)
            target = drift.PREFIX + d
            sd, prior = world.target(target)
            for B in rpc.B_GRID:
                rule = sel["r2"][str(B)]["cfg"]
                L = int(rule["L"])
                if L not in jb_cache:
                    jb_cache[L] = pc.jbar_for("month", L)
                block = dict(chosen[str(B)])
                block["jbar"] = jb_cache[L]
                for draw in range(args.draws):
                    idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                    for seed in pc.SEEDS:
                        ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx), basis)
                        meta = {"stratum": "subm_drift", "bid": "month", "target": target, "date": d,
                                "B": B, "draw": draw, "seed": seed, "rule_set": "month_oosc", "fix": "causal_v2"}
                        yield meta, ep, prior, p0[seed], _specs_fix(ep, rule, block, fit_blob, basis)
            del sd, prior

    _eval_fix_driver(path, jobs())


def cmd_eval_year_fix(args):
    gpu_ok()
    pc.set_variant("base")
    sel = json.loads(pc.selection_file("year", "oos").read_text())
    chosen = json.loads((OUT / "select_year_fix.json").read_text())["b"]
    fit_blob = load_fit(OUT / "fit_subm.pt")
    world = pc.load_world("year", False)
    p0 = pc.load_p0(world, "year")
    basis = pc.load_basis("year")
    target = "sub-M_ses-CO-" + YEAR_CORE
    sd, prior = world.target(target)
    jb_cache = {}
    path = OUT / "rows_year_fix.jsonl"

    def jobs():
        for B in rpc.B_GRID:
            rule = sel["r2"][str(B)]["cfg"]
            L = int(rule["L"])
            if L not in jb_cache:
                jb_cache[L] = pc.jbar_for("year", L)
            block = dict(chosen[str(B)])
            block["jbar"] = jb_cache[L]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in pc.SEEDS:
                    ep = _basis_on(pc.p0_episode(world, sd, prior, p0[seed], idx), basis)
                    meta = {"stratum": "subm_year", "bid": "year", "target": target, "date": YEAR_CORE,
                            "B": B, "draw": draw, "seed": seed, "rule_set": "year_oos", "fix": "causal_v2"}
                    yield meta, ep, prior, p0[seed], _specs_fix(ep, rule, block, fit_blob, basis)

    _eval_fix_driver(path, jobs())


def cmd_eval_subc_fix(args):
    gpu_ok()
    from scripts.run_expansion import average_jbar
    for bid in ("sc_0629", "sc_0708"):
        ctxs = _subc_contexts([bid])
        bb, ctx = ctxs[bid]
        sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
        chosen_all = json.loads((OUT / f"select_{bid}_fix.json").read_text())["b"]
        fit_blob = load_fit(OUT / f"fit_{bid}.pt")
        basis = torch.load(EXP_OUT / f"basis/{bid}.pt", weights_only=False)["U"].to(ctx.device)
        blob = torch.load(EXP_OUT / f"jacobian/{bid}.pt", weights_only=False)
        jb_cache = {}
        targets = [t for t in bb.targets if t.stratum == "month"]
        path = OUT / f"rows_{bid}_fix.jsonl"

        def jobs(bid=bid, ctx=ctx, sel=sel, chosen_all=chosen_all, fit_blob=fit_blob, basis=basis,
                 blob=blob, targets=targets):
            for t in targets:
                date = t.session[-8:]
                if date in FINAL_DATES or date == "20140307":
                    raise SystemExit(date)
                sd, prior = ctx.world.target(t.session)
                for B in (1, 4, 8, 16, 33):
                    r = sel["b"][str(B)]["r2"]
                    rule = {"L": r["L"], "eta": r["eta"], "rho": r["rho"], "S": r["S"]}
                    L = int(rule["L"])
                    if L not in jb_cache:
                        jb_cache[L] = average_jbar(blob, L).to(ctx.device)
                    block = dict(chosen_all[str(B)])
                    block["jbar"] = jb_cache[L]
                    for draw in range(args.draws):
                        idx = rpc.draw_trials(t.session, B, draw, sd.bank.n_trials)
                        for seed in (0, 1):
                            ep = _basis_on(pc.p0_episode(ctx.world, sd, prior, ctx.p0[seed], idx), basis)
                            meta = {"stratum": "subc_month", "bid": bid, "target": t.session, "date": date,
                                    "B": B, "draw": draw, "seed": seed, "rule_set": sel["selection_session"],
                                    "fix": "causal_v2"}
                            yield meta, ep, prior, ctx.p0[seed], _specs_fix(ep, rule, block, fit_blob, basis)
                del sd, prior

        _eval_fix_driver(path, jobs())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exactness")
    f = sub.add_parser("fit")
    f.add_argument("--which", default="all", choices=("subm", "subc", "all"))
    s = sub.add_parser("select")
    s.add_argument("--which", default="all", choices=("month", "year", "all"))
    sub.add_parser("select-subc")
    e = sub.add_parser("eval-drift")
    e.add_argument("--draws", type=int, default=8)
    e.add_argument("--targets", default=None)
    y = sub.add_parser("eval-year")
    y.add_argument("--draws", type=int, default=8)
    c = sub.add_parser("eval-subc")
    c.add_argument("--draws", type=int, default=8)
    sub.add_parser("cost")
    sf = sub.add_parser("select-fix")
    sf.add_argument("--which", default="all", choices=("month", "year", "all"))
    sub.add_parser("select-subc-fix")
    for name in ("eval-drift-fix", "eval-year-fix", "eval-subc-fix"):
        e = sub.add_parser(name)
        e.add_argument("--draws", type=int, default=8)
    args = p.parse_args()
    {"exactness": cmd_exactness, "fit": cmd_fit, "select": cmd_select, "select-subc": cmd_select_subc,
     "eval-drift": cmd_eval_drift, "eval-year": cmd_eval_year, "eval-subc": cmd_eval_subc,
     "cost": cmd_cost, "select-fix": cmd_select_fix, "select-subc-fix": cmd_select_subc_fix,
     "eval-drift-fix": cmd_eval_drift_fix, "eval-year-fix": cmd_eval_year_fix,
     "eval-subc-fix": cmd_eval_subc_fix}[args.cmd](args)


if __name__ == "__main__":
    main()
