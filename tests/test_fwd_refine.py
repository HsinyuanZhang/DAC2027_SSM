"""Tests for the P3 forward-only refinements (F7-J read-in rule, readout ridge)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode import fwd_refine as fr

K, H, D, N, LAGS = 4, 6, 2, 7, 5


class LinTrunk(nn.Module):
    """Linear causal trunk: h_t = sum_l z_{t-l} C_l, y = out_proj(h).  Exact Jacobians."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.C = nn.Parameter(torch.randn(LAGS, K, H, generator=g, dtype=torch.float64) * 0.5)
        self.out_proj = nn.Linear(H, D).double()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z):
        t = z.shape[1]
        h = torch.zeros(z.shape[0], t, H, dtype=z.dtype)
        for l in range(LAGS):
            h[:, l:] = h[:, l:] + z[:, :t - l] @ self.C[l]
        return self.out_proj(h)

    def exact_jbar(self, lmax):
        j = torch.zeros(lmax + 1, D, K, dtype=torch.float64)
        for l in range(min(LAGS, lmax + 1)):
            j[l] = self.out_proj.weight @ self.C[l].T
        return j


def _data(seed=0, lens=(30, 22, 41)):
    g = torch.Generator().manual_seed(seed)
    xz = torch.randn(sum(lens) + 10, N, generator=g, dtype=torch.float64)
    y = torch.randn(xz.shape[0], D, generator=g, dtype=torch.float64)
    valid = torch.rand(xz.shape[0], generator=g) > 0.15
    wins, a = [], 3
    for ln in lens:
        wins.append((a, a + ln))
        a += ln
    X, Y, M = fr.pad_trials(xz, y, valid, wins)
    T = torch.randn(K, N, generator=g, dtype=torch.float64) * 0.3
    return X, Y, M, T


def test_fj_gradient_equals_negative_true_gradient_elementwise():
    base = LinTrunk()
    X, Y, M, T = _data()
    jbar = base.exact_jbar(LAGS + 3)
    g, q, n, _ = fr.fj_gradient(base, X, Y, M, T, jbar)
    true = fr.diagnostic_true_neg_gradient(base, X, Y, M, T)
    torch.testing.assert_close(g, true, rtol=1e-9, atol=1e-10)
    assert fr.cosine(g, true) == pytest.approx(1.0, abs=1e-12)
    assert float(n) == float(M.sum())


def test_one_fj_step_equals_one_gradient_step_elementwise():
    base = LinTrunk(1)
    X, Y, M, T = _data(1)
    jbar = base.exact_jbar(8)
    eta, lam, eps = 0.37, 0.2, 1e-3
    center = T + 0.1
    out = fr.fj_refine(base, X, Y, M, T, jbar, eta, lam, [1], center=center, eps=eps)[1]
    true = fr.diagnostic_true_neg_gradient(base, X, Y, M, T)
    n = M.sum()
    q = ((X * M.unsqueeze(-1)) * X).sum((0, 1)) / n
    expect = T + eta * true / (q + eps)[None, :] - eta * lam * (T - center)
    torch.testing.assert_close(out, expect, rtol=1e-9, atol=1e-10)
    # without preconditioning and shrinkage: plain gradient descent step
    plain = fr.fj_refine(base, X, Y, M, T, jbar, eta, 0.0, [1], precondition=False)[1]
    torch.testing.assert_close(plain, T + eta * true, rtol=1e-9, atol=1e-10)


def test_short_feedback_window_is_a_strict_truncation():
    base = LinTrunk()
    X, Y, M, T = _data()
    full = base.exact_jbar(8)
    g_full, *_ = fr.fj_gradient(base, X, Y, M, T, full)
    g_short, *_ = fr.fj_gradient(base, X, Y, M, T, full[:2])
    assert fr.cosine(g_full, g_short) < 0.999            # LAGS=5 > 2: truncation changes it
    g_exact, *_ = fr.fj_gradient(base, X, Y, M, T, full[:LAGS])
    torch.testing.assert_close(g_full, g_exact)


def test_temporal_feedback_matches_loop():
    g = torch.Generator().manual_seed(3)
    r = torch.randn(2, 9, D, generator=g, dtype=torch.float64)
    j = torch.randn(4, D, K, generator=g, dtype=torch.float64)
    e = fr.temporal_feedback(r, j)
    for b in range(2):
        for t in range(9):
            ref = sum(r[b, t + l] @ j[l] for l in range(4) if t + l < 9)
            if isinstance(ref, int):
                ref = torch.zeros(K, dtype=torch.float64)
            torch.testing.assert_close(e[b, t], ref)


def test_lag_jacobians_recover_analytic_kernel_and_zero_deviation():
    base = LinTrunk(2)
    g = torch.Generator().manual_seed(4)
    z = torch.randn(5, 40, K, generator=g, dtype=torch.float64)
    j = fr.lag_jacobians(base, z, 10)
    exact = base.exact_jbar(10)
    for m in range(5):
        torch.testing.assert_close(j[m], exact)
    xz = torch.randn(300, N, generator=g, dtype=torch.float64)
    sums = fr.average_jacobian_session(base, xz, torch.randn(K, N, generator=g, dtype=torch.float64),
                                       range(60, 300, 7), lmax=10, width=50, chunk=8)
    jb = sums["sum"] / sums["n"]
    torch.testing.assert_close(jb, exact)
    assert np.allclose(fr.jacobian_representativeness(sums, sums["n"]), 0.0, atol=1e-6)


def test_refinement_paths_use_no_backward():
    base = LinTrunk()
    X, Y, M, T = _data()
    calls = {"n": 0}
    og, ob = torch.autograd.grad, torch.autograd.backward
    torch.autograd.grad = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), og(*a, **k))[1]
    torch.autograd.backward = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), ob(*a, **k))[1]
    try:
        out = fr.fj_refine(base, X, Y, M, T, base.exact_jbar(6), 0.1, 0.1, [1, 3])
        st = fr.readout_stats(base, X, Y, M, out[3])
        fr.ridge_delta(st, 0.1, "full")
        fr.ridge_delta(st, 0.1, "diag")
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob
    assert calls["n"] == 0
    assert not out[3].requires_grad


def test_fj_decreases_loss_on_linear_trunk():
    base = LinTrunk(5)
    X, Y, M, T = _data(5)

    def loss(tab):
        r = (Y - base(X @ tab.T)) * M.unsqueeze(-1)
        return float((r ** 2).sum() / (2 * M.sum()))

    out = fr.fj_refine(base, X, Y, M, T, base.exact_jbar(6), 0.05, 0.0, [1, 5, 10], precondition=False)
    assert loss(out[1]) < loss(T) and loss(out[10]) < loss(out[5]) < loss(out[1])


def test_readout_stats_additive_and_ridge_limits():
    base = LinTrunk(6)
    X, Y, M, T = _data(6)
    st = fr.readout_stats(base, X, Y, M, T)
    parts = [fr.readout_stats(base, X[i:i + 1], Y[i:i + 1], M[i:i + 1], T) for i in range(3)]
    tot = parts[0]
    for p in parts[1:]:
        tot = fr.add_readout_stats(tot, p)
    torch.testing.assert_close(tot["G"], st["G"])
    torch.testing.assert_close(tot["R"], st["R"])
    assert tot["n"] == st["n"]
    # lam -> 0: V solves least squares for the features on the valid bins
    _, h = fr.run_trunk(base, X, T, want_h=True)
    hb = torch.cat([h, torch.ones_like(h[..., :1])], -1)
    sel = M.bool()
    A, Yv = hb[sel], Y[sel]
    v_ls = torch.linalg.lstsq(A, Yv).solution.T
    v = fr.head_matrix(base) + fr.ridge_delta(st, 1e-12, "full")
    torch.testing.assert_close(v, v_ls, rtol=1e-6, atol=1e-7)
    # lam -> inf: no change
    assert fr.ridge_delta(st, 1e12, "full").abs().max() < 1e-9
    # full-rank orthonormal basis == full ridge; diag differs from full
    u = torch.linalg.qr(torch.randn(H + 1, H + 1, dtype=torch.float64)).Q
    torch.testing.assert_close(fr.ridge_delta(st, 0.3, "lowrank", basis=u),
                               fr.ridge_delta(st, 0.3, "full"), rtol=1e-8, atol=1e-9)
    assert (fr.ridge_delta(st, 0.3, "diag") - fr.ridge_delta(st, 0.3, "full")).abs().max() > 1e-6
    # rank-r basis: delta lives in the span of the basis
    d_r = fr.ridge_delta(st, 0.3, "lowrank", basis=u[:, :2])
    torch.testing.assert_close(d_r @ (torch.eye(H + 1, dtype=torch.float64) - u[:, :2] @ u[:, :2].T),
                               torch.zeros(D, H + 1, dtype=torch.float64), atol=1e-10, rtol=0)


def test_decode_with_readout_identity_and_replacement():
    base = LinTrunk(7)
    X, Y, M, T = _data(7)
    xf = X[0]
    y0 = fr.decode_with_readout(base, xf, T)
    torch.testing.assert_close(y0, base(xf.unsqueeze(0) @ T.T).squeeze(0))
    y1 = fr.decode_with_readout(base, xf, T, fr.head_matrix(base))
    torch.testing.assert_close(y0, y1)
    v2 = fr.head_matrix(base) * 2.0
    torch.testing.assert_close(fr.decode_with_readout(base, xf, T, v2),
                               2.0 * y0, rtol=1e-9, atol=1e-9)


def test_pad_trials_padding_does_not_change_valid_outputs():
    base = LinTrunk(8)
    X, Y, M, T = _data(8)
    solo = base(X[1:2, :22] @ T.T)
    pad = base(X[1:2] @ T.T)[:, :22]
    torch.testing.assert_close(solo, pad)
