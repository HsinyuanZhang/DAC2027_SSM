#!/usr/bin/env python
"""Causal-first-B stage for the expansion queue. Does not change the earlier stages.

``causal-source`` scores each OOS backbone's held-out pool[-1] (alpha, S1, COMBO at each alpha, CCA+S1).
``sm_0307`` is IS and is skipped. ``causal-eval`` scores every target with the frozen methods, selecting
alpha and (eta, rho) only from episodes protocol (a) or (b) allows. R2 and R3 replay the first B trials
with the backbone's existing source rule. Latency 0 is primary; latency 1 is the sensitivity.

Rows go to ``causal_rows/`` (not ``rows/``). Selection episodes go to ``causal_select/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ssm_decode import expansion_causal as ec
from ssm_decode import expansion_protocol as ep
from ssm_decode.atomic_jsonl import append_jsonl, read_jsonl, repair_jsonl
from ssm_decode.causal_protocol import COLD, INITIAL


def _ints(text, default):
    return tuple(int(x) for x in text.split(",")) if text else tuple(default)


def _floats(text, default):
    return tuple(float(x) for x in text.split(",")) if text else tuple(default)


def _artifact_store(args):
    from scripts.run_expansion import Store
    return Store(args.artifacts or args.out, None)


def _out_store(args):
    from scripts.run_expansion import Store
    return Store(args.out, args.work)


def _grids(args):
    from scripts.run_causal_main import ETAS, RHOS
    from ssm_decode.p0solve import ALPHAS
    return {"etas": _floats(getattr(args, "etas", None), ETAS),
            "rhos": _floats(getattr(args, "rhos", None), RHOS),
            "alphas": _floats(getattr(args, "combo_alphas", None), ALPHAS),
            "Bs": _ints(getattr(args, "budgets", None), ep.B_LIST),
            "seeds": _ints(getattr(args, "seeds", None), ep.SEEDS),
            "latencies": _ints(getattr(args, "latency", None) or "0,1", (0, 1))}


def assert_fixed_cca_path(session: str) -> None:
    """The sub-C ridge-CCA path bug opened every NWB under sub-M. Refuse that layout."""
    from scripts.run_baselines_readin import nwb_path_for_session
    from ssm_decode.dandi688_data import session_subject
    path = nwb_path_for_session(session)
    subject = session_subject(session)
    if path.parts[-2] != subject:
        raise RuntimeError(f"ridge-CCA NWB for {session} resolved to {path}, not under {subject}")


def _jsonable_choice(choice):
    if choice is None:
        return None
    keep = ("eta", "rho", "fallback_base", "mode", "source_gain", "source_se", "base_key",
            "n_viable", "conservative_rejected", "rejected_eta", "rejected_rho")
    return {k: choice.get(k) for k in keep if k in choice or True}


def _alpha_vote(alpha: dict) -> dict:
    """Sessions that actually vote in the gap bin used for alpha, not the whole allowed pool."""
    return {"n_sessions": int(alpha.get("n_sessions") or 0),
            "sessions": list(alpha.get("sessions") or []),
            "tags": list(alpha.get("tags") or []),
            "bin": alpha.get("bin"), "bin_used": alpha.get("bin_used"),
            "fallback_from": alpha.get("fallback_from")}


def cmd_causal_source(args):
    """Score held-out pool[-1] episodes. Idempotent: a unit already on disk is not redone."""
    import torch
    from scripts.run_expansion import Ctx, guard_backbone, register, select_backbones
    from scripts import run_causal_main as rcm
    from scripts import run_pc3_refine as pc
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    st, art = _out_store(args), _artifact_store(args)
    bbs = select_backbones(args)
    register(bbs, art)
    grids = _grids(args)
    pc.set_variant("base")
    pc._patch_autograd_counter()
    legacy = Path(args.legacy_select) if getattr(args, "legacy_select", None) else None
    for bb in bbs:
        guard_backbone(bb)
        if not bb.oos_selection:
            marker = f"causal_select/{bb.bid}/done.json"
            from scripts.run_expansion import jwrite
            jwrite(st.w(marker), {"bid": bb.bid, "skipped": "IS",
                                  "reason": "sm_0307 selection is in-sample and is not part of the pool"})
            st.push(marker)
            print(f"[causal-source] {bb.bid}: IS, not a selection contributor", flush=True)
            continue
        _score_heldout(bb, st, art, args, grids, legacy, pc, rcm)
    print("[causal-source] done", flush=True)


def _heldout_rows(st, legacy, bid, family):
    local_path = st.o(f"causal_select/{bid}/{family}.jsonl")
    # Read the scratch copy if this process already pulled it; fall back to out.
    work_path = st.w(f"causal_select/{bid}/{family}.jsonl")
    rows = read_jsonl(work_path) if work_path.exists() else read_jsonl(local_path)
    if legacy is not None and bid in ec.LEGACY_TAGS:
        rows = rows + ec.load_jsonl(ec.legacy_family_path(legacy, family, bid))
    return rows


def _score_heldout(bb, st, art, args, grids, legacy, pc, rcm):
    import torch
    from scripts.run_expansion import Ctx, jwrite
    sel_path = art.o(f"select/{bb.bid}.json")
    if not sel_path.exists():
        raise SystemExit(f"{bb.bid}: run select before causal-source (missing {sel_path})")
    if not art.exists(f"jacobian/{bb.bid}.pt") and not (art.work / f"jacobian/{bb.bid}.pt").exists():
        if not art.o(f"jacobian/{bb.bid}.pt").exists():
            raise SystemExit(f"{bb.bid}: run jacobian before causal-source")
    sel = json.loads(sel_path.read_text())
    session = bb.val_session
    assert_fixed_cca_path(session)
    for src in bb.trunk_sources:
        assert_fixed_cca_path(src)
    families = ec.FAMILIES
    pending = {}
    for fam in families:
        rel = f"causal_select/{bb.bid}/{fam}.jsonl"
        st.pull(rel)
        repair_jsonl(st.w(rel))
        rows = _heldout_rows(st, legacy, bb.bid, fam)
        pending[fam] = ec.pending_units(rows, session, grids["seeds"], grids["Bs"], grids["alphas"], fam)
    if not any(pending.values()):
        _write_source_done(st, jwrite, bb, grids, skipped=False)
        print(f"[causal-source] {bb.bid}: already complete", flush=True)
        return
    pc._BWD["n"] = 0
    ctx = Ctx(bb, art, args, load_sessions=True)
    sd, prior = ctx.world.sessions[session], ctx.world.priors[session]
    k_max = max(grids["Bs"])
    if k_max > sd.bank.n_trials:
        raise SystemExit(f"{bb.bid}: held-out {session} has only {sd.bank.n_trials} trials")
    print(f"[causal-source] cache {bb.bid} {session[-8:]} k={k_max}", flush=True)
    cache = rcm.prefix_cache(sd, prior, ctx.p0, k_max, ctx.world, "month")
    base = ctx.world.adapter.model.base
    gap_days = int(bb.trunk_gap_days(session))
    configs = [(e, r) for e in grids["etas"] for r in grids["rhos"]]
    device = ctx.device
    for fam in families:
        rel = f"causal_select/{bb.bid}/{fam}.jsonl"
        path = st.w(rel)
        for session_id, seed, B, alpha_key in pending[fam]:
            if int(pc._BWD["n"]) != 0:
                raise SystemExit(f"{bb.bid}: backward call during causal-source")
            L = int(sel["b"][str(B)]["r2"]["L"])
            eta_r2 = float(sel["b"][str(B)]["r2"]["eta"])
            row = _score_unit(fam, ctx.world, sd, prior, base, cache, ctx.jbar(L), device,
                              session_id, int(seed), int(B), alpha_key, configs, eta_r2, gap_days,
                              bb.bid, rcm)
            append_jsonl(path, [row])
            st.push(rel)
            print(f"[causal-source] {bb.bid} {fam} B={B} s{seed} a={alpha_key} "
                  f"skipped={row.get('skipped')}", flush=True)
    if int(pc._BWD["n"]) != 0:
        raise SystemExit(f"{bb.bid}: backward call during causal-source")
    _write_source_done(st, jwrite, bb, grids, skipped=False)


def _write_source_done(st, jwrite, bb, grids, skipped: bool):
    marker = f"causal_select/{bb.bid}/done.json"
    jwrite(st.w(marker), {"bid": bb.bid, "session": bb.val_session, "skipped": skipped,
                          "oos": True, "Bs": list(grids["Bs"]), "seeds": list(grids["seeds"]),
                          "alphas": list(grids["alphas"]), "rule_version": ec.RULE_VERSION})
    st.push(marker)


def _rebase_scores(world, base, sd, prior, cache, seed, idx, kind, alpha, jbar, device, configs,
                   mean, std, rcm):
    from ssm_decode.causal_protocol import run_rebased_many
    prepare = rcm._prepare_from(cache, seed, kind, alpha, device)
    bias_fn = None
    if kind == "cca" or (kind == "w_alpha" and float(alpha) != 0.0):
        bias_fn = rcm._base_bias_fn(cache, seed, alpha, device, "cca" if kind == "cca" else "w_alpha")
    raw = [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype), sd.valid[a:b].to(dtype=sd.x32.dtype))
           for a, b in (sd.trials[int(i)] for i in idx)]
    many = run_rebased_many(base, raw, prior.float(), list(configs), prepare, jbar=jbar, base_bias=bias_fn)
    tables, biases = [], []
    for ans in many:
        bias = rcm.to_cpu(ans["bias"])
        if kind == "w_alpha" and float(alpha) != 0.0:
            bias = bias + np.asarray(rcm.w_alpha_of(cache, seed, len(idx), alpha)[1], np.float64)
        elif kind == "cca":
            bias = bias + np.asarray(cache["cca"][len(idx)][1], np.float64)
        tables.append(rcm.to_cpu(ans["table"]))
        biases.append(bias)
    scores = rcm._score_np(world, sd, mean, std, tables, biases, idx)
    if scores is None:
        raise RuntimeError("rebase query shorter than 50 bins")
    return [{"eta": float(e), "rho": float(r), "r2": float(sc)} for (e, r), sc in zip(configs, scores)]


def _score_unit(fam, world, sd, prior, base, cache, jbar, device, session, seed, B, alpha_key,
                configs, eta_r2, gap_days, tag, rcm):
    idx = rcm.calibration_indices(sd.bank.n_trials, B)
    rcm.assert_prefix_idx(idx, B)
    mean, std = cache["norm"][B]
    w0 = cache["p0"][seed][B]
    zeros = np.zeros(w0.shape[0])
    head = {"tag": tag, "session": session, "seed": int(seed), "B": int(B),
            "protocol": "causal_first_B", "gap_days": int(gap_days), "eta_r2": float(eta_r2)}
    p0_scores = rcm._score_np(world, sd, mean, std, [w0], [zeros], idx)
    if p0_scores is None:
        return {**head, "skipped": "query_short"}
    p0_r2 = float(p0_scores[0])
    if fam == "alpha":
        rows_a = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            got = rcm.w_alpha_of(cache, seed, B, alpha)
            if got is None:
                rows_a.append({"alpha": alpha, "r2": -1.0})
                continue
            w, b, mean_a, std_a, _info = got
            sc = rcm._score_np(world, sd, mean_a, std_a, [w], [b], idx)
            rows_a.append({"alpha": alpha, "r2": None if sc is None else float(sc[0])})
        return {**head, "p0": p0_r2, "alphas": rows_a}
    if fam == "combo":
        alpha = float(alpha_key)
        kind = "p0" if alpha == 0.0 else "w_alpha"
        got = rcm.w_alpha_of(cache, seed, B, alpha)
        if got is None:
            return {**head, "alpha": alpha, "skipped": "cca_missing", "p0": p0_r2}
        w, b, mean_a, std_a, _info = got
        scb = rcm._score_np(world, sd, mean_a, std_a, [w], [b], idx)
        if scb is None:
            return {**head, "alpha": alpha, "skipped": "query_short", "p0": p0_r2}
        rebase = _rebase_scores(world, base, sd, prior, cache, seed, idx, kind, alpha, jbar, device,
                                configs, mean, std, rcm)
        return {**head, "p0": p0_r2, "base": float(scb[0]), "alpha": alpha, "rebase": rebase}
    if fam == "cca_s1":
        got = cache["cca"][B]
        if not isinstance(got, tuple):
            return {**head, "alpha": 1.0, "skipped": "cca_missing", "p0": p0_r2}
        scb = rcm._score_np(world, sd, mean, std, [got[0]], [got[1]], idx)
        if scb is None:
            return {**head, "alpha": 1.0, "skipped": "query_short", "p0": p0_r2}
        rebase = _rebase_scores(world, base, sd, prior, cache, seed, idx, "cca", 1.0, jbar, device,
                                configs, mean, std, rcm)
        return {**head, "p0": p0_r2, "base": float(scb[0]), "alpha": 1.0, "rebase": rebase}
    rebase = _rebase_scores(world, base, sd, prior, cache, seed, idx, "p0", 0.0, jbar, device,
                            configs, mean, std, rcm)
    return {**head, "p0": p0_r2, "base": p0_r2, "alpha": 0.0, "rebase": rebase}


def calibration_index(n_trials: int, B: int):
    """The calibration set is the first B usable trials. Not a draw."""
    from ssm_decode.causal_protocol import calibration_indices
    return calibration_indices(n_trials, B)


def _same_step(a, b) -> bool:
    if not a or not b:
        return False
    return (float(a["eta"]), float(a["rho"])) == (float(b["eta"]), float(b["rho"]))


def build_method_maps(base, sd, prior, cache, seed, spans, B, *, alpha, eta, rho, jbar, kind):
    """Commit maps for one frozen refinement. Reads the first B usable trials and nothing after them."""
    from scripts import run_causal_main as rcm
    cold, initial = rcm.cold_and_initial(rcm.to_cpu(prior), cache, spans)
    if kind == "w_alpha":
        maps, reads, info = rcm.alpha_commits(cache, seed, spans, B, float(alpha), cold, initial)
        return maps, reads, info
    body, reads, info = rcm.loop_commits(
        base, sd, prior, cache, seed, spans, B, eta=float(eta), rho=float(rho), jbar=jbar,
        alpha=0.0 if kind == "p0" else float(alpha), kind=kind)
    maps, reads = rcm.install_maps(body, cold, initial, B)
    return maps, reads, info


def first_b_maps(neural: np.ndarray, *, B: int = 8, alpha: float = 0.75, eta: float = 1.0,
                 rho: float = 0.1):
    """CPU stand-in used by the perturbation test. Statistics and the loop see only trials ``0 .. k-1``."""
    import torch
    from scripts import run_causal_main as rcm
    from ssm_decode.causal_protocol import measured_read_bin
    neural = np.asarray(neural, dtype=np.float64)
    n = neural.shape[1]
    spans = [(i * 5, i * 5 + 4) for i in range(B)]
    kdim, dout = 2, 1
    torch.manual_seed(0)                                 # same trunk on every call; only ``neural`` changes

    class _Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Linear(kdim, dout)

        def forward(self, z):
            return self.out_proj(z)

    trunk = _Trunk().eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    prior = torch.zeros(kdim, n)
    y = torch.zeros(neural.shape[0], dout)
    valid = torch.ones(neural.shape[0])

    class _Bank:
        n_trials = B

    class _SD:
        pass

    sd = _SD()
    sd.bank = _Bank()
    sd.trials = spans
    sd.x32 = torch.as_tensor(neural, dtype=torch.float32)
    sd.y_norm = y
    sd.valid = valid
    sd.name = "synthetic"
    cache = {"norm": {}, "p0": {0: {}}, "cca": {}, "read_bin": {}}
    for k in range(1, B + 1):
        idx = calibration_index(B, k)
        rcm.assert_prefix_idx(idx, k)
        read = measured_read_bin(spans, idx)
        chunk = np.concatenate([neural[a:b] for a, b in (spans[int(i)] for i in idx)], axis=0)
        mean = chunk.mean(axis=0)
        std = np.ones(n)
        weight = np.stack([mean, 0.5 * mean], axis=0)          # [K, N], depends only on the prefix
        cache["norm"][k] = (mean, std)
        cache["p0"][0][k] = weight
        cache["cca"][k] = (0.3 * weight, np.array([mean[0], -mean[1]]), {"family": "synthetic"})
        cache["read_bin"][k] = read
    jbar = torch.zeros(2, dout, kdim)
    jbar[0, 0, :] = 0.1
    out = {}
    out["w_alpha"] = build_method_maps(
        trunk, sd, prior, cache, 0, spans, B, alpha=alpha, eta=0.0, rho=0.0, jbar=jbar, kind="w_alpha")[0]
    out["combo"] = build_method_maps(
        trunk, sd, prior, cache, 0, spans, B, alpha=alpha, eta=eta, rho=rho, jbar=jbar,
        kind="p0" if alpha == 0.0 else "w_alpha")[0]
    out["s1"] = build_method_maps(
        trunk, sd, prior, cache, 0, spans, B, alpha=0.0, eta=eta, rho=rho, jbar=jbar, kind="p0")[0]
    return out, spans


def cmd_causal_eval(args):
    """Frozen-method rows under causal-first-B. Resumes from ``causal_rows`` keys already on disk."""
    import torch
    from scripts.run_expansion import (Ctx, guard_backbone, jwrite, register, select_backbones)
    from scripts import run_causal_main as rcm
    from scripts import run_pc3_refine as pc
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    st, art = _out_store(args), _artifact_store(args)
    bbs = select_backbones(args)
    register(bbs, art)
    grids = _grids(args)
    pc.set_variant("base")
    pc._patch_autograd_counter()
    legacy = Path(args.legacy_select) if getattr(args, "legacy_select", None) else None
    if (st.out / "causal_select").exists():
        st.pull("causal_select")
    root = st.work if (st.work / "causal_select").exists() else st.out
    bundles = ec.load_episodes(root, legacy)
    reported = ec.reported_target_dates()
    allow_incomplete = bool(getattr(args, "allow_incomplete_pool", False))
    for bb in bbs:
        guard_backbone(bb)
        _eval_backbone(bb, st, art, args, grids, bundles, reported, allow_incomplete, pc, rcm, jwrite)
    print("[causal-eval] done", flush=True)


def _eval_backbone(bb, st, art, args, grids, bundles, reported, allow_incomplete, pc, rcm, jwrite):
    import torch
    from scripts.run_expansion import Ctx
    sel_path = art.o(f"select/{bb.bid}.json")
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else None
    if sel is None:
        raise SystemExit(f"{bb.bid}: select/*.json is required before causal-eval")
    if not art.o(f"jacobian/{bb.bid}.pt").exists():
        raise SystemExit(f"{bb.bid}: jacobian is required before causal-eval")
    ctx = Ctx(bb, art, args, load_sessions=False)
    basis_path = art.o(f"basis/{bb.bid}.pt")
    basis = torch.load(basis_path, weights_only=False)["U"].to(ctx.device) if basis_path.exists() else None
    hash0 = ctx.world.adapter._hash()
    want_dates = None if not getattr(args, "targets", None) else set(args.targets.split(","))
    for t in bb.targets:
        date = ep.date_of(t.session)
        if want_dates is not None and date not in want_dates:
            continue
        if date == "20140307" and bb.bid != "sm_0307":
            raise SystemExit("20140307 is not a target of this round except sm_0307")
        if t.session in ep.ALL_FORBIDDEN:
            raise PermissionError(t.session)
        missing = ec.missing_contributors(date, bundles.get("alpha", []))
        if missing and not allow_incomplete:
            raise SystemExit(f"{bb.bid} {date}: selection pool incomplete, missing {missing}. "
                             "Run causal-source on those backbones first, or pass --allow-incomplete-pool.")
        _eval_one_target(bb, t, ctx, st, sel, basis, grids, bundles, reported, missing, hash0, pc, rcm, jwrite)


def _eval_one_target(bb, t, ctx, st, sel, basis, grids, bundles, reported, missing, hash0, pc, rcm, jwrite):
    import torch
    date = ep.date_of(t.session)
    rel = f"causal_rows/{bb.bid}/{date}.jsonl"
    st.pull(rel)
    path = st.w(rel)
    repair_jsonl(path)
    seen_rows = read_jsonl(path)
    if not ec.pending_eval(seen_rows, ec.EVAL_METHODS, grids["seeds"], grids["Bs"], grids["latencies"]):
        print(f"[causal-eval] {bb.bid} {date}: already complete", flush=True)
        return
    assert_fixed_cca_path(t.session)
    for src in list(bb.trunk_sources) + list(bb.heldout):
        assert_fixed_cca_path(src)
    sd, prior = ctx.world.target(t.session)
    spans = [(int(a), int(b)) for a, b in sd.trials]
    Bs = tuple(b for b in grids["Bs"] if b <= len(spans))
    if not Bs:
        print(f"[causal-eval] {bb.bid} {date}: no B fits {len(spans)} trials", flush=True)
        return
    rules_by_b = {B: ec.target_rules(bundles, date, bb.trunk_gap_days(t.session), B, reported,
                                    float(sel["b"][str(B)]["r2"]["eta"]) if sel and str(B) in sel["b"] else 1.0)
                  for B in Bs}
    comp_rel = f"causal_select/composition/{bb.bid}_{date}.json"
    jwrite(st.w(comp_rel), {"bid": bb.bid, "target": t.session, "date": date,
                            "missing_contributors": missing, "pool_complete": not missing,
                            "selection_oos_backbone": bb.oos_selection,
                            "by_B": {str(B): {"a": rules_by_b[B]["a"]["composition"],
                                             "b": rules_by_b[B]["b"]["composition"],
                                             "s1_a": rules_by_b[B]["a"]["s1_composition"],
                                             "s1_b": rules_by_b[B]["b"]["s1_composition"],
                                             "combo_a": rules_by_b[B]["a"]["combo_composition"],
                                             "combo_b": rules_by_b[B]["b"]["combo_composition"],
                                             "cca_a": rules_by_b[B]["a"]["cca_composition"],
                                             "alpha_a": rules_by_b[B]["a"]["alpha"].get("alpha"),
                                             "alpha_b": rules_by_b[B]["b"]["alpha"].get("alpha"),
                                             "alpha_vote_a": _alpha_vote(rules_by_b[B]["a"]["alpha"]),
                                             "alpha_vote_b": _alpha_vote(rules_by_b[B]["b"]["alpha"]),
                                             "n_sessions_a": rules_by_b[B]["a"]["composition"]["n_sessions"],
                                             "n_sessions_b": rules_by_b[B]["b"]["composition"]["n_sessions"]}
                                    for B in Bs},
                            "rule_version": ec.RULE_VERSION})
    st.push(comp_rel)
    print(f"[causal-eval] {bb.bid} {date} pool sessions "
          + ", ".join(f"B{B}:a={rules_by_b[B]['a']['composition']['n_sessions']}/"
                       f"b={rules_by_b[B]['b']['composition']['n_sessions']}" for B in Bs), flush=True)
    k_max = max(Bs)
    # Month CCA_CFG for every backbone, including days. Causal-main days used CCA_CFG["days"].
    # Not re-selected here; the short-gap ridge-CCA gap is reported with that disclosure.
    cache = rcm.prefix_cache(sd, prior, ctx.p0, k_max, ctx.world, "month")
    base = ctx.world.adapter.model.base
    behavior = sd.record.behavior
    label_stats = ctx.world.label_stats
    dev = sd.x32.device
    t0 = time.perf_counter()
    for B in Bs:
        rules = rules_by_b[B]
        r2_cfg = sel["b"][str(B)]["r2"] if sel and str(B) in sel["b"] else None
        r3_cfg = sel["b"][str(B)]["r3"] if sel and str(B) in sel["b"] else None
        jb = ctx.jbar(int(r2_cfg["L"])) if r2_cfg else None
        for seed in grids["seeds"]:
            pending = ec.pending_eval(seen_rows, ec.EVAL_METHODS, (seed,), (B,), grids["latencies"])
            if not pending:
                continue
            pc._BWD["n"] = 0
            built, notes = _build_rows_maps(
                base, ctx, sd, prior, cache, spans, B, int(seed), rules, jb, r2_cfg, r3_cfg, basis, pc, rcm)
            if int(pc._BWD["n"]) != 0:
                raise SystemExit(f"{bb.bid} {date}: backward call in a candidate")
            q = sd.query_mask(calibration_index(sd.bank.n_trials, B)).detach().cpu().numpy()
            cal = np.zeros(sd.n_bins, dtype=bool)
            for i in calibration_index(sd.bank.n_trials, B):
                a, b = spans[int(i)]
                cal[a:b] = True
            need = {(m, lat) for m, _s, _b, lat in pending}
            jobs, meta = [], []
            for method, lat in sorted(need):
                if method not in built:
                    continue
                maps, reads, install, _cfg = built[method]
                segs = rcm.segments_for(spans, B, sd.n_bins, lat, install)
                rcm.assert_segment_causal(segs, spans, reads, staged_trial0=True)
                on_final, n_q = rcm.query_bins_on_commit(segs, q, B)
                packed = {k: rcm._pack(maps[k], dev) for _a, _b, k in segs}
                want_h = any(packed[k]["readout"] is not None for _a, _b, k in segs)
                jobs.append({"segments": segs, "packed": packed, "qmask": q, "want_h": want_h})
                meta.append((method, lat, on_final, n_q, install))
            scores = rcm.decode_many(base, sd.x32, jobs, label_stats, behavior) if jobs else []
            rows = []
            common = {"bid": bb.bid, "subject": bb.subject, "target": t.session, "date": date,
                      "stratum": t.stratum, "gap_days": t.gap_days,
                      "trunk_gap_days": bb.trunk_gap_days(t.session), "tier": t.tier,
                      "apst_dev": t.apst_dev, "B": int(B), "seed": int(seed),
                      "protocol": "causal_first_B", "rule_version": ec.RULE_VERSION,
                      "selection_oos_backbone": bb.oos_selection,
                      "h1_trunk_hash": hash0[:16], "h1_trunk_unchanged": ctx.world.adapter._hash() == hash0,
                      "h2_backward_calls": int(pc._BWD["n"]),
                      "h4_budget": int(B), "h4_query_disjoint": bool(not (cal & q).any()),
                      "n_query_bins": int(q.sum()), "cca_cfg": "month",
                      "cca_path": "subject-directory"}
            for (method, lat, on_final, n_q, install), r2 in zip(meta, scores):
                _maps, _reads, _install, cfg = built[method]
                rows.append({**common, "method": method, "latency": int(lat), "r2": float(r2),
                             "install": install, "replay": method in ("r2", "r3"),
                             "reference": method in ("r2", "r3"),
                             "query_bins_on_final": int(on_final), "cfg": cfg})
            for method, info in notes.items():
                for lat in grids["latencies"]:
                    if (method, lat) not in need:
                        continue
                    rows.append({**common, "method": method, "latency": int(lat), "r2": None,
                                 "unavailable": True, "install": None, "replay": method in ("r2", "r3"),
                                 "reference": method in ("r2", "r3"), "cfg": info})
            if int(seed) == min(grids["seeds"]):
                for row in list(rows):
                    if row["method"] not in ec.SEED_FREE:
                        continue
                    for other in grids["seeds"]:
                        if int(other) == int(seed):
                            continue
                        if (row["method"], int(other), int(B), int(row["latency"])) in {
                                ec.eval_row_key(r) for r in seen_rows}:
                            continue
                        clone = dict(row)
                        clone["seed"] = int(other)
                        rows.append(clone)
            append_jsonl(path, rows)
            seen_rows.extend(rows)
            st.push(rel)
            print(f"[causal-eval] {bb.bid} {date} B={B} s{seed} n={len(rows)} "
                  f"t={time.perf_counter() - t0:.0f}s", flush=True)
    del sd, prior


def _choice_pair(block, which):
    """``block`` is gate_choices output. ``which`` is 'and' or 'conservative'."""
    if not block:
        return None
    return block.get(which)


def _build_rows_maps(base, ctx, sd, prior, cache, spans, B, seed, rules, jb, r2_cfg, r3_cfg, basis, pc, rcm):
    """Install every method for one (target, B, seed). Aliases a gate that picked the same step."""
    cold, initial = rcm.cold_and_initial(rcm.to_cpu(prior), cache, spans)
    prior_np = rcm.to_cpu(prior)
    built, notes = {}, {}
    maps, reads = rcm.prior_commits(prior_np, cache, spans, B, cold, initial)
    built["p_prior"] = (maps, reads, "anytime", {})
    maps, reads = rcm.p0_commits(cache, seed, spans, B, cold, initial)
    built["p0"] = (maps, reads, "anytime", {})
    for label, side in (("w_alpha_a", "a"), ("w_alpha_b", "b")):
        alpha = rules[side]["alpha"].get("alpha")
        comp = rules[side]["composition"]
        if alpha is None:
            notes[label] = {"unavailable": True, "reason": "no causal alpha", "n_sessions": comp["n_sessions"]}
            continue
        maps, reads, info = rcm.alpha_commits(cache, seed, spans, B, float(alpha), cold, initial)
        if maps is None:
            notes[label] = {**(info or {}), "n_sessions": comp["n_sessions"]}
        else:
            built[label] = (maps, reads, "anytime", {**(info or {}), "protocol": side,
                                                     "n_sessions": comp["n_sessions"],
                                                     "sessions": comp["sessions"]})
    maps, reads, info = rcm.cca_final_only(cache, spans, B, cold, initial)
    if maps is None:
        notes["ridge_cca"] = info or {"unavailable": True}
    else:
        built["ridge_cca"] = (maps, reads, "final_only", {**(info or {}), "cca_cfg": "month",
                                                          "path": "nwb_path_for_session"})

    def refine(label, choice, kind, alpha, comp):
        if choice is None or (kind != "p0" and alpha is None):
            notes[label] = {"unavailable": True, "reason": "no selection", "n_sessions": comp["n_sessions"]}
            return
        if float(choice["eta"]) != 0.0 and jb is None:
            notes[label] = {"unavailable": True, "reason": "no jacobian for this B"}
            return
        use_kind = "p0" if (kind == "w_alpha" and float(alpha or 0.0) == 0.0) else kind
        body, reads, info = rcm.loop_commits(
            base, sd, prior, cache, seed, spans, B, eta=float(choice["eta"]), rho=float(choice["rho"]),
            jbar=jb, alpha=0.0 if use_kind == "p0" else float(alpha or 0.0), kind=use_kind)
        maps, reads = rcm.install_maps(body, cold, initial, B)
        built[label] = (maps, reads, "anytime", {**_jsonable_choice(choice), **(info or {}),
                                                 "n_sessions": comp["n_sessions"],
                                                 "sessions": comp["sessions"], "gate": choice.get("mode", "and")})

    def cons_alias(and_label, cons_label, block, comp, kind, alpha):
        if _same_step(_choice_pair(block, "and"), _choice_pair(block, "conservative")) and and_label in built:
            m, r, inst, cfg = built[and_label]
            built[cons_label] = (m, r, inst, {**cfg, "gate": "conservative", "alias_of": and_label})
        else:
            refine(cons_label, _choice_pair(block, "conservative"), kind, alpha, comp)

    s1a = rules["a"]["s1"]
    refine("s1_causal", _choice_pair(s1a, "and"), "p0", 0.0, rules["a"]["s1_composition"])
    cons_alias("s1_causal", "s1_causal_cons", s1a, rules["a"]["s1_composition"], "p0", 0.0)
    s1b = rules["b"]["s1"]
    if _same_step(_choice_pair(s1a, "and"), _choice_pair(s1b, "and")) and "s1_causal" in built:
        m, r, inst, cfg = built["s1_causal"]
        built["s1_causal_b"] = (m, r, inst, {**cfg, "protocol": "b", "alias_of": "s1_causal"})
    else:
        refine("s1_causal_b", _choice_pair(s1b, "and"), "p0", 0.0, rules["b"]["s1_composition"])
    cons_alias("s1_causal_b", "s1_causal_b_cons", s1b, rules["b"]["s1_composition"], "p0", 0.0)
    for side, label, cons in (("a", "combo_a", "combo_a_cons"), ("b", "combo_b", "combo_b_cons")):
        alpha = rules[side]["alpha"].get("alpha")
        block = rules[side]["combo"]
        comp = rules[side]["combo_composition"]
        kind = "p0" if alpha in (None, 0.0) else "w_alpha"
        refine(label, _choice_pair(block, "and"), kind, alpha, comp)
        if _same_step(_choice_pair(block, "and"), _choice_pair(block, "conservative")) and label in built:
            m, r, inst, cfg = built[label]
            built[cons] = (m, r, inst, {**cfg, "gate": "conservative", "alias_of": label})
        else:
            refine(cons, _choice_pair(block, "conservative"), kind, alpha, comp)
    cca = rules["a"]["cca_s1"]
    refine("cca_s1", _choice_pair(cca, "and"), "cca", 1.0, rules["a"]["cca_composition"])
    cons_alias("cca_s1", "cca_s1_cons", cca, rules["a"]["cca_composition"], "cca", 1.0)
    ccab = rules["b"]["cca_s1"]
    if _same_step(_choice_pair(cca, "and"), _choice_pair(ccab, "and")) and "cca_s1" in built:
        m, r, inst, cfg = built["cca_s1"]
        built["cca_s1_b"] = (m, r, inst, {**cfg, "protocol": "b", "alias_of": "cca_s1"})
    else:
        refine("cca_s1_b", _choice_pair(ccab, "and"), "cca", 1.0, rules["b"]["cca_composition"])
    cons_alias("cca_s1_b", "cca_s1_b_cons", ccab, rules["b"]["cca_composition"], "cca", 1.0)
    if r2_cfg is None or jb is None:
        notes["r2"] = {"unavailable": True, "reason": "no R2 source rule"}
        notes["r3"] = {"unavailable": True, "reason": "no R3 source rule"}
    else:
        idx = calibration_index(sd.bank.n_trials, B)
        rcm.assert_prefix_idx(idx, B)
        episode = pc.p0_episode(ctx.world, sd, prior, ctx.p0[seed], idx)
        tab = episode.refine(jb, r2_cfg["eta"], r2_cfg["rho"], [r2_cfg["S"]])[r2_cfg["S"]]
        mean_b, std_b = rcm.to_cpu(episode.mean), rcm.to_cpu(episode.std)
        final = rcm.final_reference_map(rcm.to_cpu(tab), None, mean_b, std_b, rcm._end(cache, B))
        maps = {COLD: cold, INITIAL: initial, B: final}
        reads = {k: m.max_bin_read for k, m in maps.items()}
        built["r2"] = (maps, reads, "final_only", {k: r2_cfg[k] for k in ("L", "eta", "rho", "S")})
        if r3_cfg is None or basis is None:
            notes["r3"] = {"unavailable": True, "reason": "no R3 rule or PCA basis"}
        else:
            st3 = episode.stats(tab)
            v = episode.v_from(st3, r3_cfg["lam"], r3_cfg["kind"], basis).detach().cpu().numpy()
            final = rcm.final_reference_map(rcm.to_cpu(tab), None, mean_b, std_b, rcm._end(cache, B),
                                           readout=v)
            maps = {COLD: cold, INITIAL: initial, B: final}
            reads = {k: m.max_bin_read for k, m in maps.items()}
            built["r3"] = (maps, reads, "final_only", {"kind": r3_cfg["kind"], "lam": r3_cfg["lam"],
                                                       **{k: r2_cfg[k] for k in ("L", "eta", "rho", "S")}})
    return built, notes
