"""CPU unit checks for the Hyser H-A helpers (no Mamba kernels needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from ssm_decode import hyser_backbone as hb

_spec = importlib.util.spec_from_file_location(
    "run_hyser_eval", Path(__file__).resolve().parents[1] / "scripts" / "run_hyser_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

FINGERS = np.repeat(np.arange(1, 6), 3)
SAMPLES = np.tile(np.arange(1, 4), 5)


def test_calibration_draws_are_balanced_and_disjoint_from_query():
    for b, expect_max in [(1, 1), (2, 1), (5, 1), (10, 2)]:
        for seed in range(20):
            cal = ev.draw_calibration(b, np.random.default_rng(seed), FINGERS, SAMPLES)
            assert len(cal) == b == len(set(cal))
            counts = np.bincount(FINGERS[cal], minlength=6)[1:]
            assert counts.max() == expect_max and counts.max() - counts[counts > 0].min() <= 0 or b == 10
            if b == 10:
                assert (counts == 2).all()
            if b == 5:
                assert (counts == 1).all()


def test_input_stats_and_zscore():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 3.0, size=(3, 50, 7)).astype(np.float32)
    mu, sd = hb.input_stats(x)
    z = hb.zscore(x, mu, sd)
    np.testing.assert_allclose(z.reshape(-1, 7).mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.reshape(-1, 7).std(0), 1, atol=1e-3)


def test_r2_is_one_for_perfect_and_unit_invariant():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((100, 5))
    assert hb.r2_scores(t, t)["r2"] == 1.0
    p = t + 0.3 * rng.standard_normal(t.shape)
    a = hb.r2_scores(t, p)["r2"]
    scale = np.array([1, 10, 100, 1e-2, 5.0])
    assert abs(hb.r2_scores(t * scale, p * scale)["r2"] - a) < 1e-9


def test_fit_tables_moves_only_tables_and_reduces_loss():
    torch.manual_seed(0)
    trunk = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 5))
    for p in trunk.parameters():
        p.requires_grad_(False)
    model = hb.APBackbone([12], 5, k=4, base=trunk)
    before = [p.detach().clone() for p in trunk.parameters()]
    x = torch.randn(2, 3, 20, 12)
    true_table = torch.randn(2, 4, 12)
    y = trunk(torch.einsum("dbtn,dkn->dbtk", x, true_table)).detach()
    init = torch.randn(2, 4, 12) * 0.1
    start = (trunk(torch.einsum("dbtn,dkn->dbtk", x, init)) - y).square().mean().item()
    fit, loss = hb.fit_tables(model, x, y, init, steps=200, lr=3e-2)
    assert fit.shape == init.shape and loss.mean() < 0.5 * start
    for a, b in zip(before, trunk.parameters()):
        assert torch.equal(a, b)
        assert b.grad is None or float(b.grad.abs().sum()) == 0.0


def test_ridge_closed_form_recovers_linear_map_and_ema_is_causal():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 200, 6)).astype(np.float32)
    w_true = rng.standard_normal((6, 5))
    y = x @ w_true + 0.5
    w, b = hb.ridge_fit(x, y, lam=1e-6)
    np.testing.assert_allclose(w, w_true, atol=1e-3)
    np.testing.assert_allclose(b, 0.5, atol=1e-3)
    x2 = x.copy(); x2[:, 100:] = 9.0
    np.testing.assert_array_equal(hb.causal_ema(x, 0.9)[:, :100], hb.causal_ema(x2, 0.9)[:, :100])
