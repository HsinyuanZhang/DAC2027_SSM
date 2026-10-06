"""Tests for stage-2' pieces: set-attention fold, descriptors, prior correction."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ssm_decode import association as assoc
from ssm_decode.set_attention import SetAttentionReadIn, unit_descriptors


def test_fold_table_is_permutation_equivariant():
    torch.manual_seed(0)
    read_in = SetAttentionReadIn(desc_dim=7, k=4)
    desc = torch.randn(13, 7)
    table = read_in.fold_table(desc)
    perm = torch.randperm(13)
    table_perm = read_in.fold_table(desc[perm])
    assert torch.allclose(table[:, perm], table_perm, atol=1e-6)


def test_fold_table_shapes_and_softmax():
    torch.manual_seed(1)
    read_in = SetAttentionReadIn(desc_dim=5, k=6, tau=0.5)
    desc = torch.randn(9, 5)
    table = read_in.fold_table(desc)
    assert table.shape == (6, 9)


def _fake_record(t=4000, n=9, seed=3):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, 4))
    phi = rng.standard_normal((t, 4))
    neural = (phi @ h.T + rng.standard_normal((t, n)) * 0.5)
    neural = np.clip(np.round(np.exp(neural * 0.3) - 0.7), 0, 5).astype(np.float32)
    behavior = phi[:, :2] * 0.8
    return SimpleNamespace(neural=neural, behavior=behavior,
                           eval_mask=np.ones(t, dtype=bool),
                           trial_bounds=[(0, t // 2), (t // 2, t)])


def test_unit_descriptors_layout_and_label_free():
    from ssm_decode.set_attention import raw_unit_stats
    record = _fake_record()
    bins = np.arange(1500)
    desc, info = unit_descriptors(record, bins, lags=2, with_profile=True)
    stats_width = raw_unit_stats(record, bins).shape[1]
    assert desc.shape[1] == 2 * 3 + stats_width
    assert info["k0"] == 6 and info["stats"] == stats_width
    assert np.isfinite(desc).all()
    free, _ = unit_descriptors(record, bins, lags=2, with_profile=False)
    # Label-free descriptor keeps activity stats identical, profile zeroed.
    np.testing.assert_allclose(desc[:, 6:], free[:, 6:])
    assert np.all(free[:, :6] == 0.0)
    # Too few finite behavior bins falls back to a zero profile instead of raising.
    broken = SimpleNamespace(neural=record.neural, behavior=np.full_like(record.behavior, np.nan),
                             eval_mask=record.eval_mask, trial_bounds=record.trial_bounds)
    desc_b, _ = unit_descriptors(broken, bins, lags=2, with_profile=True)
    assert np.all(desc_b[:, :6] == 0.0)


def test_prior_correction_recovers_channel_gain():
    rng = np.random.default_rng(5)
    n, k, d, m = 24, 6, 2, 8000
    w0 = rng.standard_normal((k, n))
    g_true = np.exp(rng.normal(0, 0.25, n))
    b_src = rng.standard_normal((d, k))
    phi = rng.standard_normal((m, k))
    w_true = w0 * g_true[None, :]
    x = phi @ w_true + rng.standard_normal((m, n)) * 0.2
    y = (x @ w_true.T) @ b_src.T + rng.standard_normal((m, d)) * 0.05
    out = assoc.prior_correction_table(b_src, w0, x[:400], y[:400], lam=1.0)
    assert np.abs(out["channel_gains"] / g_true - 1).max() < 0.15
    # The corrected table must decode better than the raw prior on held-out
    # bins through the head path.
    def r2(table):
        pred = (x[400:] @ table.T) @ b_src.T
        truth = y[400:]
        return 1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean(0)) ** 2).sum()
    assert r2(out["table"]) > r2(w0)


def test_read_in_gradient_flows():
    torch.manual_seed(2)
    read_in = SetAttentionReadIn(desc_dim=3, k=2)
    desc = torch.randn(5, 3, requires_grad=True)
    table = read_in.fold_table(desc)
    table.sum().backward()
    for name, param in read_in.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert desc.grad is not None
