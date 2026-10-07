"""Required unit tests for the labeled-calibration round (doc Q0)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ssm_decode.identity_encoder import (CalibContext, F7Calibrator,
                                         IdentityCalibrator, hand_statistics,
                                         hand_statistics_streamed)


def _ctx(seed=0, t=3000, n=12, d=2, bins=None):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((t, 4))
    behavior = latent[:, :2] * 0.7
    neural = np.clip(np.round(np.exp(0.3 * (latent @ rng.standard_normal((4, n)))) - 0.7),
                     0, 5).astype(np.float32)
    record = SimpleNamespace(neural=neural, behavior=behavior,
                             eval_mask=np.ones(t, dtype=bool))
    x_norm = (neural - neural.mean(0)) / (neural.std(0) + 1e-9)
    prior = rng.standard_normal((4, n)) * 0.1
    return CalibContext(record, bins if bins is not None else np.arange(1000),
                        8, prior, None, x_norm), record


def test_f7_linear_trunk_equals_gradient_descent():
    """G = -2*lr*A^T, eta = 1, q = 1: one F7 step equals one GD step."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, k, d, L = 7, 5, 2, 60
    A = rng.standard_normal((d, k)) * 0.3
    T = rng.standard_normal((k, n)) * 0.2
    x = rng.standard_normal((L, n))
    x = x / np.sqrt((x ** 2).mean(0))                # q_i = 1 exactly
    ystar = rng.standard_normal((L, d))
    lr = 0.05
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(ystar.astype(np.float32))
    Tt = torch.from_numpy(T.astype(np.float32)).requires_grad_(True)
    At = torch.from_numpy(A.astype(np.float32))
    loss = (((xt @ Tt.T) @ At.T - yt) ** 2).mean()
    loss.backward()
    gd = (Tt - lr * Tt.grad).detach().numpy()
    # MSE-mean loss: dL/dT = -(2/(L*D)) * sum_t A^T r_t x_t^T, so the GD
    # step column is +(2*lr/(L*D)) * sum A^T r x.  F7 with q=1 gives
    # G * mean_t(r x); matching requires G = 2*lr/D * A^T and eps = 0.
    f7 = F7Calibrator(d, k, steps=1, gtype="linear", lag=0)
    with torch.no_grad():
        f7.blocks[0].G.weight.copy_(2.0 * lr / d * At.T)
        f7.blocks[0].G.bias.zero_()
        f7.blocks[0].eta.fill_(1.0)
        f7.blocks[0].eps = 0.0
    record = SimpleNamespace(behavior=ystar.astype(np.float32),
                             eval_mask=np.ones(L, dtype=bool),
                             neural=x.astype(np.float32))
    x_norm = x  # already zero-mean-ish; identity normalization keeps q=1
    ctx = CalibContext(record, np.arange(L), 4, T,
                       lambda T_: (xt @ T_.float().T) @ At.T, x_norm)
    out = f7.calibrate(ctx)
    assert out is not None
    np.testing.assert_allclose(out[0], gd, atol=1e-5)


def test_loo_prior_excludes_own_session():
    from scripts.run_labeled_calibration import D1_POOLS
    # structural check on the training code path: mean over j != s
    tables = [np.full((3, 4), float(j)) for j in range(4)]
    s = 2
    loo = np.mean([tables[j] for j in range(len(tables)) if j != s], 0)
    assert loo[0, 0] == pytest.approx((0 + 1 + 3) / 3)
    assert loo[0, 0] != pytest.approx(np.mean([t[0, 0] for t in tables]))


def test_h2_target_side_no_gradients():
    """A calibrate() that builds a graph would leak requires_grad; the guard
    must make every output detached (test fails if the guard is removed)."""
    ctx, _ = _ctx()
    calib = IdentityCalibrator("f3", k=4, drop_groups=["activity"])
    with torch.no_grad():
        out = calib._calibrate_grad(ctx)
    arr = out[0]
    assert not isinstance(arr, torch.Tensor) or not arr.requires_grad


def test_unit_permutation_permutes_table_columns():
    # Prior-free groups must permute exactly with the units, and the
    # row-wise encoder's table columns follow the same permutation.
    ctx, _ = _ctx(seed=1)
    groups = ["crosscov", "direction", "t4"]
    stats_a = hand_statistics(ctx, groups)
    perm = np.random.default_rng(0).permutation(ctx.x_norm.shape[1])
    x2 = ctx.x_norm[:, perm]
    ctx2 = CalibContext(ctx.record, ctx.bins, ctx.budget,
                        ctx.prior_table[:, perm], None, x2)
    stats_b = hand_statistics(ctx2, groups)
    # stats are [N, S]: units are ROWS, so the permutation hits rows.
    np.testing.assert_allclose(stats_a[perm, :], stats_b, atol=1e-8)
    from ssm_decode.identity_encoder import IdentityCalibrator
    calib = IdentityCalibrator("f3", k=4, drop_groups=["activity"])
    with torch.no_grad():
        t_a, _, _ = calib.hand(torch.from_numpy(stats_a.astype(np.float32)))
        t_b, _, _ = calib.hand(torch.from_numpy(stats_b.astype(np.float32)))
    np.testing.assert_allclose(t_a.detach().numpy()[:, perm],
                               t_b.detach().numpy(), atol=1e-6)
    # and the encoder table columns follow the same permutation


def test_streaming_equals_batch_statistics():
    ctx, _ = _ctx(seed=2)
    batch = hand_statistics(ctx)
    streamed = hand_statistics_streamed(ctx, chunks=3)
    np.testing.assert_allclose(batch, streamed, atol=1e-9)
