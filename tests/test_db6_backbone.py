"""CPU unit checks for the DB6 D-A helpers (v2: contiguous streaming decode; no Mamba kernels needed
except the optional GPU test enabled with DB6_TEST_CUDA=<device>)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from ssm_decode import db6_backbone as hb
from ssm_decode import db6_data as dd

GRASP = np.repeat(np.arange(1, 8), 12)
REP = np.tile(np.arange(1, 13), 7)


def make_stream(n_trials=6, seed=0, lens=(40, 55, 33, 60, 47, 52), gap=7):
    """Synthetic continuous stream of adjacent trials (+ a trailing unlabeled gap)."""
    rng = np.random.default_rng(seed)
    starts, ends, pos = [], [], 0
    for ln in lens[:n_trials]:
        starts.append(pos); pos += ln; ends.append(pos)
    t = pos + gap
    blob = {"x": rng.gamma(2.0, 3.0, size=(t, dd.N_CH)).astype(np.float32),
            "y": np.zeros(t, np.int8), "trial_start": np.array(starts), "trial_end": np.array(ends),
            "trial_grasp": (np.arange(n_trials) % 7 + 1).astype(np.int8),
            "trial_rep": (np.arange(n_trials) // 7 + 1).astype(np.int16)}
    for i in range(n_trials):                      # movement occupies the second half of each trial
        a, b = starts[i], ends[i]
        blob["y"][(a + b) // 2:b] = blob["trial_grasp"][i]
    return hb.Stream((9, 9, 9), blob)


class ToyBase(torch.nn.Module):
    """Causal stateful stand-in for the Mamba trunk: GRU over the latent stream -> 8 logits."""

    def __init__(self, k):
        super().__init__()
        torch.manual_seed(0)
        self.rnn = torch.nn.GRU(k, 16, batch_first=True)
        self.out = torch.nn.Linear(16, dd.N_CLASS)

    def forward(self, z):
        return self.out(self.rnn(z)[0])


def test_calibration_draws_are_balanced_distinct_and_cover_rest():
    for b in (1, 2, 4, 8, 16, 42):
        for seed in range(30):
            cal = hb.draw_calibration(b, np.random.default_rng(seed), GRASP, REP)
            assert len(cal) == b == len(set(cal)) and cal == sorted(cal)
            cov = np.array(hb.coverage(cal, GRASP))
            assert cov.sum() == b and cov.max() - cov.min() <= 1
            assert (cov > 0).sum() == min(b, 7)


def test_source_protocol_keys():
    sk = hb.source_keys()
    assert len(sk) == 10 * 4 - 1 and (2, 2, 2) not in sk and all(k[1] in (1, 2) for k in sk)
    tk = hb.target_keys()
    assert len(tk) == 60 and all(k[1] in (3, 4, 5) for k in tk) and not set(tk) & set(sk)
    assert sk[hb.latest_source_index(sk, 1)] == (1, 2, 2)
    assert sk[hb.latest_source_index(sk, 2)] == (2, 2, 1)       # truncated D2_T2 is not a source


def test_stats_floor_and_zscore():
    s = make_stream()
    mu, sd = hb.input_stats(s, range(s.n), floor=0.0)
    z = hb.zscore(s.x, mu, sd)[s.bin_mask(range(s.n))]
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-3)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-3)
    # a near-dead channel (std 1e-4 of the others) is not amplified to unit variance when floored
    dead = hb.Stream(s.key, {"x": s.x.copy(), "y": s.y, "trial_start": s.tstart, "trial_end": s.tend,
                             "trial_grasp": s.grasp, "trial_rep": s.rep})
    dead.x[:, 13] *= 1e-4
    _, sd0 = hb.input_stats(dead, range(dead.n), floor=0.0)
    _, sd1 = hb.input_stats(dead, range(dead.n), floor=0.05)
    assert sd0[13] < 1e-3 * np.median(sd0) and sd1[13] == pytest.approx(0.05 * np.median(sd1), rel=1e-5)
    assert np.allclose(np.delete(sd0, 13), np.delete(sd1, 13))      # live channels untouched


def test_metrics_perfect_chance_and_vote():
    y = np.repeat(np.arange(8), 10)
    s = hb.class_scores(y, y)
    assert s["bal_acc"] == 1.0 and s["macro_f1"] == 1.0
    assert hb.class_scores(y, np.zeros_like(y))["bal_acc"] == pytest.approx(1 / 8)
    t = np.array([0] * 90 + [1] * 10); p = np.zeros(100, int)
    assert hb.class_scores(t, p)["bal_acc"] == pytest.approx(0.5)
    assert hb.vote_grasp(np.array([0, 0, 3, 3, 5, 0])) == 3 and hb.vote_grasp(np.zeros(5, int)) == 0


def test_score_stream_scores_only_requested_trials():
    s = make_stream()
    logits = np.zeros((s.T, dd.N_CLASS))
    logits[np.arange(s.T), s.y] = 1.0                       # perfect everywhere
    for i in range(s.n):                                    # corrupt trial 2 completely
        if i == 2:
            logits[s.tstart[i]:s.tend[i]] = 0
            logits[s.tstart[i]:s.tend[i], 7] = 1
    good = [0, 1, 3, 4, 5]
    assert hb.score_stream(logits, s, good)["bal_acc"] == 1.0           # calibration/other trials excluded
    assert hb.score_stream(logits, s, [2])["acc"] == 0.0
    # bins outside every trial (trailing gap) never enter the score
    logits[s.tend[-1]:] = 0; logits[s.tend[-1]:, 5] = 1
    assert hb.score_stream(logits, s, good)["bal_acc"] == 1.0


def test_cal_segments_merge_adjacent_trials_only():
    s = make_stream()
    assert hb.cal_segments(s, [3, 1, 2]) == [(int(s.tstart[1]), int(s.tend[3]))]
    assert hb.cal_segments(s, [0, 2, 3, 5]) == [(int(s.tstart[0]), int(s.tend[0])),
                                                 (int(s.tstart[2]), int(s.tend[3])),
                                                 (int(s.tstart[5]), int(s.tend[5]))]
    segs, ys = hb.segment_arrays(s, s.x, [0, 2])
    assert [len(v) for v in segs] == [s.lens[0], s.lens[2]] and [len(v) for v in ys] == [s.lens[0], s.lens[2]]


def test_stream_decode_carries_state_across_trial_boundaries():
    """Output at the first bin of trial k must depend on the inputs of trial k-1 (no state reset at the boundary)."""
    s = make_stream()
    base = ToyBase(dd.N_CH).eval()
    table = torch.randn(dd.N_CH, dd.N_CH) * 0.3
    xz = torch.from_numpy(hb.zscore(s.x, *hb.input_stats(s, range(s.n), 0.0)))
    ref = hb.decode_stream(base, xz, table)[0]
    for k in range(1, s.n):
        x2 = xz.clone()
        x2[s.tstart[k - 1]:s.tend[k - 1]] += 2.0                 # perturb ONLY trial k-1
        out = hb.decode_stream(base, x2, table)[0]
        assert np.abs(out[s.tstart[k]] - ref[s.tstart[k]]).max() > 1e-4, f"state was reset at the start of trial {k}"
        if k > 1:
            assert np.abs(out[:s.tstart[k - 1]] - ref[:s.tstart[k - 1]]).max() < 1e-6    # causality: past unchanged


def test_stream_decode_equals_one_pass_not_per_trial():
    s = make_stream()
    base = ToyBase(dd.N_CH).eval()
    table = torch.randn(dd.N_CH, dd.N_CH) * 0.3
    xz = torch.from_numpy(hb.zscore(s.x, *hb.input_stats(s, range(s.n), 0.0)))
    full = hb.decode_stream(base, xz, table)[0]
    per_trial = np.concatenate([hb.decode_stream(base, xz[a:b], table)[0] for a, b in zip(s.tstart, s.tend)])
    inside = np.concatenate([full[a:b] for a, b in zip(s.tstart, s.tend)])
    assert np.abs(full[:s.tend[0]] - per_trial[:s.lens[0]]).max() < 1e-5          # first trial: identical
    assert np.abs(inside - per_trial).max() > 1e-3                                # later trials differ: state is carried


def test_query_scoring_excludes_calibration_bins_and_ignores_their_content():
    s = make_stream()
    base = ToyBase(dd.N_CH).eval()
    table = torch.randn(dd.N_CH, dd.N_CH) * 0.3
    xz = torch.from_numpy(hb.zscore(s.x, *hb.input_stats(s, range(s.n), 0.0)))
    cal, query = [1, 4], [0, 2, 3, 5]
    lg = hb.decode_stream(base, xz, table)[0]
    a = hb.score_stream(lg, s, query)
    # corrupting the model outputs on the calibration trials' bins does not change the query score
    lg2 = lg.copy()
    for i in cal:
        lg2[s.tstart[i]:s.tend[i]] = np.random.default_rng(0).normal(size=(s.lens[i], dd.N_CLASS)) * 50
    assert hb.score_stream(lg2, s, query) == a
    mask = s.bin_mask(query)
    assert not (mask & s.bin_mask(cal)).any() and mask.sum() == s.lens[query].sum()


def test_ridge_closed_form_separates_synthetic_classes_and_is_causal_over_the_stream():
    rng = np.random.default_rng(1)
    L = 400
    y = np.repeat(rng.integers(0, 8, size=L // 10), 10)
    centers = rng.normal(size=(8, dd.N_CH)) * 2
    x = (centers[y] + rng.normal(size=(L, dd.N_CH)) * 0.3).astype(np.float32)
    w, b = hb.ridge_fit([x[:200], x[250:]], [y[:200], y[250:]], 0.1, 0.0)
    assert (hb.ridge_logits(x, w, b).argmax(-1) == y).mean() > 0.9
    x2 = x.copy(); x2[250:] += 5
    a, c = hb.ridge_logits(x, w, b, 0.9), hb.ridge_logits(x2, w, b, 0.9)
    np.testing.assert_allclose(a[:250], c[:250], atol=1e-5)
    assert np.abs(a[250] - c[250]).max() > 1e-3                      # one EMA pass over the whole stream
    # EMA at the start of a segment does not look at data outside the segment
    w1, _ = hb.ridge_fit([x[250:]], [y[250:]], 0.1, 0.9)
    x3 = x.copy(); x3[:250] += 9
    w2, _ = hb.ridge_fit([x3[250:]], [y[250:]], 0.1, 0.9)
    np.testing.assert_allclose(w1, w2)


def test_fit_tables_uses_only_given_segments_and_improves_loss():
    s = make_stream()
    base = ToyBase(dd.N_CH).eval()
    for p in base.parameters():
        p.requires_grad_(False)

    class M:                                                     # minimal stand-in exposing .base
        pass
    m = M(); m.base = base
    xz = hb.zscore(s.x, *hb.input_stats(s, range(s.n), 0.0))
    segs, ys = hb.segment_arrays(s, xz, [1, 2, 4])
    t0 = torch.randn(1, dd.N_CH, dd.N_CH) * 0.2
    xs = [torch.from_numpy(v) for v in segs]; yl = [torch.from_numpy(v) for v in ys]
    _, l_a = hb.fit_tables(m, xs, yl, [0] * len(xs), t0, steps=1, lr=1e-9)
    _, l_b = hb.fit_tables(m, xs, yl, [0] * len(xs), t0, steps=60, lr=3e-2)
    assert l_b[0] < l_a[0]


@pytest.mark.skipif("DB6_TEST_CUDA" not in os.environ, reason="set DB6_TEST_CUDA=<cuda device> to run the Mamba test")
def test_mamba_trunk_state_is_not_reset_between_trials():
    dev = os.environ["DB6_TEST_CUDA"]
    model = hb.build_model(1, k=8, width=64, layers=1, state_size=32, dropout=0.0, device=torch.device(dev)).eval()
    s = make_stream()
    xz = torch.from_numpy(hb.zscore(s.x, *hb.input_stats(s, range(s.n), 0.0))).to(dev)
    ref = hb.decode_stream(model.base, xz, model.tables[0].detach())[0]
    x2 = xz.clone(); x2[s.tstart[1]:s.tend[1]] += 2.0
    out = hb.decode_stream(model.base, x2, model.tables[0].detach())[0]
    assert np.abs(out[s.tstart[2]] - ref[s.tstart[2]]).max() > 1e-4


def test_reorder_stream_keeps_trial_content_and_labels():
    s = make_stream()
    order = [3, 0, 5, 1, 4, 2]
    r = hb.reorder_stream(s, order)
    assert r.T == s.lens.sum() and r.n == s.n
    for new, old in enumerate(order):
        np.testing.assert_array_equal(r.x[r.tstart[new]:r.tend[new]], s.x[s.tstart[old]:s.tend[old]])
        np.testing.assert_array_equal(r.y[r.tstart[new]:r.tend[new]], s.y[s.tstart[old]:s.tend[old]])
        assert r.grasp[new] == s.grasp[old] and r.rep[new] == s.rep[old]
    assert (r.tstart[1:] == r.tend[:-1]).all()                      # adjacent trials, no trailing gap


def test_shuffled_decoding_of_a_trial_depends_on_its_new_predecessor():
    """A decoder with state sees a different history after reordering (what the order-shuffle diagnostic probes)."""
    s = make_stream()
    base = ToyBase(dd.N_CH).eval()
    table = torch.randn(dd.N_CH, dd.N_CH) * 0.3
    z = lambda st: torch.from_numpy(hb.zscore(st.x, *hb.input_stats(s, range(s.n), 0.0)))  # noqa: E731
    a = hb.decode_stream(base, z(s), table)[0]
    r = hb.reorder_stream(s, [0, 2, 1, 3, 4, 5])
    b = hb.decode_stream(base, z(r), table)[0]
    # trial 2 comes after trial 0 instead of trial 1: the output at its first bin differs
    assert np.abs(b[r.tstart[1]] - a[s.tstart[2]]).max() > 1e-4
