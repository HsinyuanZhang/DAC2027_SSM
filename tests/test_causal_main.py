"""Causal-first-B schedule: first-B set, commit times, staging, latency, final==offline."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode.causal_protocol import (COLD, INITIAL, AffineMap, assert_segment_causal,
                                        calibration_indices, calibration_spans, commit_segments,
                                        latent_from_maps, margin_pick, query_bins_on_commit,
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
