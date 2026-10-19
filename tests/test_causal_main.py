"""Causal-first-B schedule: first-B set, commit times, staging, latency, final==offline."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode.causal_protocol import (COLD, INITIAL, AffineMap, assert_prefix_idx,
                                        assert_segment_causal, calibration_indices,
                                        calibration_spans, commit_segments, latent_from_maps,
                                        margin_pick, measured_read_bin, query_bins_on_commit,
                                        query_indices, run_rebased_many)
from ssm_decode.stream_f7 import rebased_feedback_loop


USABLE = [(10, 20), (30, 40), (50, 60), (70, 80)]
N_BINS = 100


def test_calibration_set_is_the_first_b_usable_trials():
    idx = calibration_indices(len(USABLE), 2)
    assert list(idx) == [0, 1]
    assert calibration_spans(USABLE, 3) == USABLE[:3]
    assert list(query_indices(len(USABLE), 2)) == [2, 3]
    with pytest.raises(ValueError):
        calibration_indices(len(USABLE), 5)
    shuffled = [(30, 40), (10, 20)]
    with pytest.raises(ValueError):
        calibration_spans(shuffled, 1)


def test_commits_switch_at_documented_boundaries():
    # latency 0: commit k turns on at the start of usable trial k.
    segs = commit_segments(USABLE, 2, N_BINS, latency=0)
    assert segs == [(0, 10, COLD), (10, 30, INITIAL), (30, 50, 1), (50, 100, 2)]
    # latency 1 trial: one usable trial later.
    segs_l = commit_segments(USABLE, 2, N_BINS, latency=1)
    assert segs_l == [(0, 10, COLD), (10, 50, INITIAL), (50, 70, 1), (70, 100, 2)]
    # No query trial: the final map turns on at the end of the last usable trial.
    only = USABLE[:2]
    segs_b = commit_segments(only, 2, N_BINS, latency=0)
    assert segs_b[-1] == (40, 100, 2)
    assert (30, 40, 1) in segs_b


def test_final_only_installs_one_commit_after_the_block():
    segs = commit_segments(USABLE, 2, N_BINS, latency=0, install="final_only")
    assert segs == [(0, 10, COLD), (10, 50, INITIAL), (50, 100, 2)]
    assert all(k in (COLD, INITIAL, 2) for _, _, k in segs)


def test_no_future_bins_except_trial0_staging():
    segs = commit_segments(USABLE, 2, N_BINS, latency=0)
    reads = {COLD: -1, INITIAL: USABLE[0][1] - 1, 1: USABLE[0][1] - 1, 2: USABLE[1][1] - 1}
    assert_segment_causal(segs, USABLE, reads, staged_trial0=True)
    # A commit that peeks past its prefix is rejected.
    leaked = dict(reads)
    leaked[1] = USABLE[1][1]          # read into the next trial
    with pytest.raises(AssertionError):
        assert_segment_causal(segs, USABLE, leaked, staged_trial0=True)
    # Trial-0 staging may not read past trial 0.
    over = dict(reads)
    over[INITIAL] = USABLE[0][1]
    with pytest.raises(AssertionError):
        assert_segment_causal(segs, USABLE, over, staged_trial0=True)
    # Cold map reads nothing.
    cold = dict(reads)
    cold[COLD] = 0
    with pytest.raises(AssertionError):
        assert_segment_causal(segs, USABLE, cold, staged_trial0=True)
    # Latency 1 still keeps commit 1 (prefix of trial 0 only) strictly before its segment.
    segs_l = commit_segments(USABLE, 2, N_BINS, latency=1)
    assert_segment_causal(segs_l, USABLE, reads, staged_trial0=True)


def test_final_commit_equals_offline_on_the_first_b_trials():
    def build(k):
        # Offline-equivalent stand-in: the table is a function of the first k trials only.
        w = np.full((1, 2), k, dtype=np.float64)
        return AffineMap(w, np.zeros(1), np.zeros(2), np.ones(2), max_bin_read=USABLE[k - 1][1] - 1)

    B = 2
    commits = {k: build(k) for k in range(1, B + 1)}
    offline = build(B)
    assert np.array_equal(commits[B].weight, offline.weight)
    assert not np.array_equal(commits[1].weight, offline.weight)
    # The installed final map is that offline map, and at latency 0 every query bin sees it.
    segs = commit_segments(USABLE, B, N_BINS, latency=0)
    q = np.zeros(N_BINS, dtype=bool)
    for a, b in USABLE[B:]:
        q[a:b] = True
    on, total = query_bins_on_commit(segs, q, B)
    assert on == total == 20
    # Latency 1 leaves the first query trial on the previous commit.
    segs_l = commit_segments(USABLE, B, N_BINS, latency=1)
    on_l, total_l = query_bins_on_commit(segs_l, q, B)
    assert total_l == 20 and on_l == 10


def test_latent_uses_the_map_of_its_segment_only():
    segs = [(0, 2, 0), (2, 4, 1)]
    maps = {
        0: AffineMap(np.array([[1.0, 0.0]]), np.zeros(1), np.zeros(2), np.ones(2)),
        1: AffineMap(np.array([[0.0, 1.0]]), np.array([3.0]), np.zeros(2), np.ones(2)),
    }
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    z = latent_from_maps(x, segs, maps)
    assert np.allclose(z[:, 0], [1.0, 2.0, 33.0, 43.0])


def test_margin_gate_or_versus_and():
    def ep(rows, p0=0.0):
        return {"p0": p0, "rebase": [{"eta": e, "rho": r, "r2": v} for e, r, v in rows]}

    # Gain 0.001, SE tiny: clears 2 SE but not 0.005. OR deploys, AND falls back.
    small = [ep([(1.0, 0.0, 0.001), (3.0, 0.0, 0.0)]) for _ in range(8)]
    # identical episodes -> SE 0
    picked_or = margin_pick(small, "rebase", eta_r2=3.0, mode="or")
    picked_and = margin_pick(small, "rebase", eta_r2=3.0, mode="and")
    assert picked_or["fallback_base"] is False and picked_or["eta"] == 1.0
    assert picked_and["fallback_base"] is True and picked_and["eta"] == 0.0
    # Gain 0.01 but SE larger than gain/2: OR (margin) deploys, AND does not.
    noisy = []
    for sign in (1.0, -1.0, 1.0, -1.0):
        noisy.append(ep([(1.0, 0.0, 0.01 + 0.05 * sign)]))
    # mean gain 0.01, spread large
    or_n = margin_pick(noisy, "rebase", eta_r2=1.0, mode="or")
    and_n = margin_pick(noisy, "rebase", eta_r2=1.0, mode="and")
    assert or_n["source_gain"] == pytest.approx(0.01)
    assert or_n["fallback_base"] is False
    assert and_n["fallback_base"] is True
    # Both gates agree when the gain clears the margin and 2 SE.
    strong = [ep([(1.0, 0.0, 0.02), (3.0, 0.1, 0.02)]) for _ in range(6)]
    both = margin_pick(strong, "rebase", eta_r2=3.0, mode="and")
    assert both["fallback_base"] is False
    # Tie on gain: prefer eta closest to the R2 rule.
    assert both["eta"] == 3.0


class _TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(2, 1)

    def forward(self, z):
        return self.out_proj(z)


def _prepare_factory(prior):
    def prepare(k, closed):
        n = k + 1 if closed else max(k, 1)
        mean = prior.new_full((prior.shape[1],), float(n) * 0.01)
        std = prior.new_ones(prior.shape[1])
        table = prior + float(n) * 0.05
        return mean, std, table
    return prepare


def test_rebased_history_matches_return_and_eta0_is_prefix_p0():
    torch.manual_seed(0)
    base = _TinyTrunk()
    prior = torch.zeros(2, 3)
    raw = []
    for _ in range(3):
        x = torch.randn(5, 3)
        y = torch.randn(5, 1)
        m = torch.ones(5)
        raw.append((x, y, m))
    prepare = _prepare_factory(prior)
    jbar = torch.randn(2, 1, 2) * 0.01
    hist = []
    ans = rebased_feedback_loop(base, raw, prior, 0.0, 0.0, prepare, mode="rebase",
                                kind="fir", jbar=jbar, history=hist)
    assert len(hist) == 3
    assert torch.allclose(ans["table"], hist[-1]["table"])
    assert torch.allclose(ans["bias"], hist[-1]["bias"])
    # eta = rho = 0: the installed table is exactly the prefix base (P0 stand-in).
    assert torch.allclose(hist[-1]["table"], hist[-1]["base"], atol=1e-6)
    assert torch.allclose(hist[-1]["delta"], torch.zeros_like(hist[-1]["delta"]), atol=1e-6)


def test_measured_read_bin_follows_the_indices_not_the_budget_name():
    spans = [(0, 10), (20, 40), (50, 70)]
    assert measured_read_bin(spans, [0]) == 9
    assert measured_read_bin(spans, [0, 1]) == 39
    # A caller that names a later trial cannot hide behind "this is commit 1".
    assert measured_read_bin(spans, [0, 2]) == 69
    with pytest.raises(AssertionError):
        assert_prefix_idx([0, 2], 1)
    assert_prefix_idx([0, 1], 2)


def test_gate_gain_is_against_the_methods_own_base():
    # P0 is weak, the closed-form base is already strong. A small step over the base
    # must not pass just because it beats P0.
    def ep(step):
        return {"p0": 0.10, "base": 0.50,
                "rebase": [{"eta": 0.0, "rho": 0.0, "r2": 0.50},
                           {"eta": 1.0, "rho": 0.0, "r2": 0.50 + step}]}
    rows = [ep(0.001) for _ in range(8)]
    vs_p0 = margin_pick(rows, "rebase", eta_r2=1.0, mode="and", base_key="p0")
    vs_base = margin_pick(rows, "rebase", eta_r2=1.0, mode="and", base_key="base")
    assert vs_p0["fallback_base"] is False and vs_p0["eta"] == 1.0
    assert vs_base["fallback_base"] is True and vs_base["eta"] == 0.0


def test_query_and_prefix_perturbation_of_a_measured_prefix():
    """Same contract as the reviewer's leak.py / leak_prefix.py, on a pure prefix statistic.

    The statistic of commit k is the mean of the neural bins of trials 0..k-1, plus a
    bias that is zero unless a caller illegally adds a future-bin feature. Perturbing
    bins after trial J-1 leaves commits k<=J unchanged and can change later commits.
    """
    spans = [(0, 4), (6, 10), (12, 16), (20, 24)]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 2))

    def commits(arr):
        out = {}
        for k in range(1, 5):
            idx = list(range(k))
            assert_prefix_idx(idx, k)
            read = measured_read_bin(spans, idx)
            acc = np.zeros(2)
            n = 0
            for i in idx:
                a, b = spans[i]
                acc += arr[a:b].sum(0)
                n += b - a
            # Bias path: a constant plus nothing from bins after `read`.
            bias = np.array([0.25, -0.1])
            out[k] = (acc / n, bias, read)
        return out

    base = commits(x)
    # Query perturbation: everything after the calibration block (B=2).
    q = x.copy()
    q[spans[1][1]:] = 99.0
    got = commits(q)
    for k in (1, 2):
        assert np.allclose(base[k][0], got[k][0])
        assert np.allclose(base[k][1], got[k][1])
    # In-block prefix perturbation at J=2: commits 1 stays, commit 2 moves, commit 3 may move.
    mid = x.copy()
    mid[spans[1][0]:spans[1][1]] += 5.0
    got2 = commits(mid)
    assert np.allclose(base[1][0], got2[1][0])
    assert not np.allclose(base[2][0], got2[2][0])
    assert base[1][2] == spans[0][1] - 1
    assert base[2][2] == spans[1][1] - 1


def test_subc_nwb_path_is_not_under_subm():
    from scripts.run_baselines_readin import nwb_path_for_session
    path = nwb_path_for_session("sub-C_ses-CO-20150319")
    assert path.parts[-2] == "sub-C"
    assert path.is_file()
    path_m = nwb_path_for_session("sub-M_ses-CO-20140626")
    assert path_m.parts[-2] == "sub-M"
    assert path_m.is_file()


def test_batched_rebased_matches_scalar_loop():
    torch.manual_seed(1)
    base = _TinyTrunk()
    prior = torch.randn(2, 3) * 0.1
    raw = []
    for _ in range(2):
        raw.append((torch.randn(6, 3), torch.randn(6, 1), torch.ones(6)))
    prepare = _prepare_factory(prior)
    jbar = torch.randn(3, 1, 2) * 0.02
    configs = [(0.0, 0.0), (1.0, 0.0), (0.3, 0.1)]
    many = run_rebased_many(base, raw, prior, configs, prepare, jbar=jbar)
    for (eta, rho), got in zip(configs, many):
        ans = rebased_feedback_loop(base, raw, prior, eta, rho, prepare, mode="rebase",
                                    kind="fir", jbar=jbar)
        assert torch.allclose(got["table"], ans["table"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(got["bias"], ans["bias"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(got["mean"], ans["mean"], atol=1e-6)


def _map_delta(a, b, fields=("weight", "bias", "mean", "std")):
    worst = 0.0
    for f in fields:
        worst = max(worst, float(np.max(np.abs(getattr(a, f) - getattr(b, f)))))
    return worst


@pytest.mark.skipif(__import__("os").environ.get("CAUSAL_LEAK") != "1", reason="set CAUSAL_LEAK=1")
def test_real_query_and_prefix_perturbation():
    """Reviewer leak.py / leak_prefix.py on one short target, including the COMBO bias path.

    Covers P0, W_alpha, S1, COMBO, CCA+S1, R1, R2, R3 and ui_reg. Query bins and, separately,
    bins after an in-block prefix must not move commits that were not allowed to read them.
    """
    import dataclasses
    import os
    os.environ.setdefault("PC_OUT", "/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016/fix")
    from scripts import run_causal_main as rcm
    from scripts import run_pc3_refine as pc
    from scripts import run_population_calib as rpc
    from ssm_decode.population_calib import SessionData

    pc.set_variant("base")
    dev = torch.device("cuda:0")
    world = rpc.World("days", "O", dev, load_sessions=False)
    p0 = pc.load_p0(world, "days")
    p3 = rcm.load_p3("days")
    B, J = 2, 1
    L = int(p3["r2"][str(B)]["cfg"]["L"])
    jb = pc.jbar_for("days", L)
    fix = rcm.load_fix_rules("days")
    rules = rcm.load_s1_rules("days")[0]
    s1 = rules["b"][str(B)]["and"]
    date = "20150616"
    alphas = rcm.alpha_bundle(fix["trunk"], fix["pooled"], date, B)
    sd, prior = world.target(rcm.PREFIX + date)
    spans = [(int(a), int(b)) for a, b in sd.trials]
    basis = pc.load_basis("days")
    k1 = __import__("scripts.run_drift_curve", fromlist=["pick_kind"]).pick_kind(p3, "r1s", B)
    lam1 = p3[f"r1s_{k1}"][str(B)]["cfg"]["lam"]
    r2_cfg = p3["r2"][str(B)]["cfg"]
    k3 = __import__("scripts.run_drift_curve", fromlist=["pick_kind"]).pick_kind(p3, "r3", B)
    lam3 = p3[f"r3_{k3}"][str(B)]["cfg"]["lam"]
    ui_block, _ui_spec = rcm.load_ui("days")
    ui = ui_block[str(B)]

    def build(s):
        cache = rcm.prefix_cache(s, prior, p0, B, world, "days")
        cold, initial = rcm.cold_and_initial(rcm.to_cpu(prior), cache, spans)
        base = world.adapter.model.base
        out = {}
        out["p0"], _ = rcm.p0_commits(cache, 0, spans, B, cold, initial)
        out["wa"], _, _ = rcm.alpha_commits(cache, 0, spans, B, alphas["pooled"], cold, initial)
        out["s1"], _, _ = rcm.loop_commits(base, s, prior, cache, 0, spans, B, eta=s1["eta"], rho=s1["rho"],
                                           jbar=jb, alpha=0.0, kind="p0")
        out["combo"], _, _ = rcm.loop_commits(
            base, s, prior, cache, 0, spans, B, eta=float(fix["combo"][0]["b"][str(B)]["and"]["eta"]),
            rho=float(fix["combo"][0]["b"][str(B)]["and"]["rho"]), jbar=jb,
            alpha=alphas["pooled"], kind="p0" if alphas["pooled"] == 0 else "w_alpha")
        out["cca_s1"], _, _ = rcm.loop_commits(
            base, s, prior, cache, 0, spans, B, eta=float(fix["cca_s1"][0]["b"][str(B)]["and"]["eta"]),
            rho=float(fix["cca_s1"][0]["b"][str(B)]["and"]["rho"]), jbar=jb, alpha=1.0, kind="cca")
        body, reads, _ = rcm.r1_commits(world, s, prior, p0[0], cache, 0, spans, B, lam1, k1, basis)
        out["r1"] = body
        idx = rcm.calibration_indices(s.bank.n_trials, B)
        ep = pc.p0_episode(world, s, prior, p0[0], idx)
        tab = ep.refine(jb, r2_cfg["eta"], r2_cfg["rho"], [r2_cfg["S"]])[r2_cfg["S"]]
        out["r2"] = {B: rcm.final_reference_map(rcm.to_cpu(tab), None, rcm.to_cpu(ep.mean), rcm.to_cpu(ep.std),
                                                rcm._end(cache, B))}
        st3 = ep.stats(tab)
        v = ep.v_from(st3, lam3, k3, basis).detach().cpu().numpy()
        out["r3"] = {B: rcm.final_reference_map(rcm.to_cpu(tab), None, rcm.to_cpu(ep.mean), rcm.to_cpu(ep.std),
                                                rcm._end(cache, B), readout=v)}
        from scripts import run_pc3_uiref as uiref
        ut = uiref.run_traj(world, s, ep, ui["lam"], ui["lr"], (ui["steps"],))[ui["steps"]]
        out["ui"] = {B: rcm.final_reference_map(ut, None, rcm.to_cpu(ep.mean), rcm.to_cpu(ep.std), rcm._end(cache, B))}
        # COMBO bias path: eta=0 must reproduce the W_alpha bias on the final commit.
        pure, _, _ = rcm.loop_commits(base, s, prior, cache, 0, spans, B, eta=0.0, rho=0.0, jbar=jb,
                                      alpha=alphas["pooled"], kind="p0" if alphas["pooled"] == 0 else "w_alpha")
        out["combo_eta0"] = pure
        return out

    def corrupt(s, start):
        rng = np.random.default_rng(1)
        neu = np.array(s.record.neural, copy=True)
        beh = np.array(s.record.behavior, copy=True)
        neu[start:] = rng.poisson(neu[start:].mean(0, keepdims=True) * 3 + 0.5, size=neu[start:].shape)
        beh[start:] = rng.normal(size=beh[start:].shape) * 50
        rec = dataclasses.replace(s.record, neural=neu, behavior=beh)
        return SessionData(rec, world.label_stats, dev, name=s.name, usable=s.trials)

    orig = build(sd)
    # eta=0 COMBO final bias matches the closed-form base (the bias path is not dropped).
    assert np.allclose(orig["combo_eta0"][B].weight, orig["wa"][B].weight, atol=1e-5)
    assert np.allclose(orig["combo_eta0"][B].bias, orig["wa"][B].bias, atol=1e-5)
    query = build(corrupt(sd, spans[B - 1][1]))
    for name in ("p0", "wa", "s1", "combo", "cca_s1", "r1", "r2", "r3", "ui"):
        for k, amap in orig[name].items():
            if k in (COLD, INITIAL):
                continue
            assert _map_delta(amap, query[name][k]) < 1e-6, name
    prefix = build(corrupt(sd, spans[J - 1][1]))
    for name in ("p0", "wa", "s1", "combo", "cca_s1", "r1"):
        for k, amap in orig[name].items():
            if k in (COLD, INITIAL) or k > J:
                continue
            assert _map_delta(amap, prefix[name][k]) < 1e-6, (name, k)
