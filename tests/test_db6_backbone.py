"""CPU unit checks for the DB6 D-A helpers (no Mamba kernels needed)."""
from __future__ import annotations

import numpy as np
import pytest

from ssm_decode import db6_backbone as hb
from ssm_decode import db6_data as dd

GRASP = np.repeat(np.arange(1, 8), 12)
REP = np.tile(np.arange(1, 13), 7)


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


def test_stats_and_zscore_ignore_padding():
    rng = np.random.default_rng(0)
    lens = np.array([30, 50, 40])
    x = np.zeros((3, 64, dd.N_CH), np.float32)
    for i, n in enumerate(lens):
        x[i, :n] = rng.gamma(2.0, 3.0, size=(n, dd.N_CH))
    mu, sd = hb.input_stats(x, lens)
    z = hb.zscore(x, mu, sd, lens)
    m = hb.valid_mask(lens, 64)
    np.testing.assert_allclose(z[m].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z[m].std(0), 1, atol=1e-3)
    assert (z[~m] == 0).all()


def test_metrics_perfect_chance_and_vote():
    y = np.repeat(np.arange(8), 10)
    s = hb.class_scores(y, y)
    assert s["bal_acc"] == 1.0 and s["macro_f1"] == 1.0
    assert hb.class_scores(y, np.zeros_like(y))["bal_acc"] == pytest.approx(1 / 8)
    # balanced accuracy ignores class frequency: 90 rest bins right, 10 grasp bins wrong
    t = np.array([0] * 90 + [1] * 10); p = np.zeros(100, int)
    assert hb.class_scores(t, p)["bal_acc"] == pytest.approx(0.5)
    assert hb.vote_grasp(np.array([0, 0, 3, 3, 5, 0])) == 3 and hb.vote_grasp(np.zeros(5, int)) == 0
    logits = np.zeros((2, 6, 8)); logits[0, :, 2] = 1; logits[1, :, 0] = 1
    y2 = np.full((2, 6), hb.IGNORE); y2[0, :4] = 2; y2[1, :4] = 0
    sc = hb.score_logits(logits, y2, np.array([4, 4]), np.array([2, 5]))
    assert sc["bal_acc"] == 1.0 and sc["vote_acc"] == 0.5


def test_ridge_closed_form_separates_synthetic_classes_and_is_causal():
    rng = np.random.default_rng(1)
    n, L = 12, 40
    y = rng.integers(0, 8, size=(n, L)); lens = np.full(n, L)
    centers = rng.normal(size=(8, dd.N_CH)) * 2
    x = (centers[y] + rng.normal(size=(n, L, dd.N_CH)) * 0.3).astype(np.float32)
    w, b = hb.ridge_fit(x, y, lens, 0.1, 0.0)
    acc = (hb.ridge_logits(x, w, b).argmax(-1) == y).mean()
    assert acc > 0.9
    x2 = x.copy(); x2[:, 25:] += 5
    a, c = hb.ridge_logits(x, w, b, 0.9), hb.ridge_logits(x2, w, b, 0.9)
    np.testing.assert_allclose(a[:, :25], c[:, :25], atol=1e-5)
