"""Reordering exactness, trace identity, causality and the target-side no-grad guard."""
from __future__ import annotations

import torch
from torch import nn

from ssm_decode import fwd_refine as fr
from ssm_decode import stream_f7 as sf

K, H, D, N, LAGS = 4, 6, 2, 7, 5


class LinTrunk(nn.Module):
    """Linear causal trunk with an exact Jacobian (same construction as test_fwd_refine)."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.C = nn.Parameter(torch.randn(LAGS, K, H, generator=g, dtype=torch.float64) * 0.3)
        self.out_proj = nn.Linear(H, D).double()
        for p in self.parameters():
            p.requires_grad_(False)
        self.grad_enabled_during_forward = []

    def forward(self, z):
        self.grad_enabled_during_forward.append(bool(torch.is_grad_enabled()))
        t = z.shape[1]
        h = torch.zeros(z.shape[0], t, H, dtype=z.dtype)
        for lag in range(LAGS):
            h[:, lag:] = h[:, lag:] + z[:, : t - lag] @ self.C[lag]
        return self.out_proj(h)


def _episode(seed=0, lens=(18, 11, 25)):
    g = torch.Generator().manual_seed(seed)
    xz = torch.randn(sum(lens) + 8, N, generator=g, dtype=torch.float64)
    y = torch.randn(xz.shape[0], D, generator=g, dtype=torch.float64)
    valid = torch.rand(xz.shape[0], generator=g) > 0.2
    wins, a = [], 2
    for ln in lens:
        wins.append((a, a + ln))
        a += ln + 1
    X, Y, M = fr.pad_trials(xz, y, valid, wins)
    table = torch.randn(K, N, generator=g, dtype=torch.float64) * 0.2
    jbar = torch.randn(6, D, K, generator=g, dtype=torch.float64)
    return xz, y, valid, wins, X, Y, M, table, jbar


def _unpadded(xz, y, valid, wins):
    return [(xz[a:b], y[a:b], valid[a:b].to(xz.dtype)) for a, b in wins]


def test_reordering_matches_temporal_feedback_to_1e_9():
    _, _, _, _, X, Y, M, _, jbar = _episode()
    r = (Y - torch.randn_like(Y) * 0.1) * M.unsqueeze(-1)
    online = sf.fir_data_term(X, r, jbar)
    batch = sf.batch_data_term(X, r, jbar)
    torch.testing.assert_close(online, batch, rtol=1e-9, atol=1e-9)


def test_reordering_matches_fj_gradient_including_mask_and_padding():
    base = LinTrunk()
    _, _, _, wins, X, Y, M, table, jbar = _episode(1)
    # Kernel longer than every trial, so end-padding of the residual matters.
    g, q, n, r = fr.fj_gradient(base, X, Y, M, table, jbar)
    U = sf.fir_data_term(X, r, jbar)
    torch.testing.assert_close(U, g * n, rtol=1e-9, atol=1e-9)
    n2, x2 = sf.valid_sum_x2(X, M)
    torch.testing.assert_close(x2 / n2, q, rtol=1e-9, atol=1e-9)
    # Unpadded trials (the streaming boundary) equal the padded batch.
    xz = X.new_zeros(wins[-1][1] + 5, N)
    # Rebuild from the padded rows directly: padding is a zero mask at the end, and a
    # causal trunk's prefix does not depend on it.  Compare row-wise data terms.
    parts = []
    for i in range(X.shape[0]):
        # drop only the trailing padded bins (mask stays zero there; internal gaps stay)
        length = int(M[i].sum().item())  # NOT the window length when internal bins are invalid
        # use the true padded width minus trailing zeros of the mask
        nz = int((M[i] > 0).nonzero()[-1]) + 1 if int(M[i].sum()) else 0
        parts.append(sf.fir_data_term(X[i, :nz], r[i, :nz], jbar))
    torch.testing.assert_close(sum(parts), U, rtol=1e-9, atol=1e-9)
    assert wins  # windows are part of the fixture; the padded path above is the one R2 uses


def test_frozen_one_commit_matches_one_fj_step():
    base = LinTrunk(2)
    xz, y, valid, wins, X, Y, M, table, jbar = _episode(2, lens=(14, 17))
    eta, rho, eps = 0.37, 0.1, 1e-3
    center = table + 0.05
    got = sf.frozen_fir_update(base, _unpadded(xz, y, valid, wins), table, jbar, eta, rho,
                               center=center, eps=eps)
    expect = fr.fj_refine(base, X, Y, M, table, jbar, eta, rho / eta, [1], center=center, eps=eps)[1]
    torch.testing.assert_close(got, expect, rtol=1e-9, atol=1e-9)


def test_iir_matches_scipy_and_trace_matches_exponential_fir():
    from scipy.signal import lfilter

    g = torch.Generator().manual_seed(3)
    x = torch.randn(40, N, generator=g, dtype=torch.float64)
    for rho in (0.0, 0.4, -0.35, 0.9):
        y = sf.iir_causal(x, rho)
        ref = lfilter([1.0], [1.0, -rho], x.numpy(), axis=0)
        torch.testing.assert_close(y, torch.from_numpy(ref.copy()), rtol=1e-12, atol=1e-12)

    rho = torch.tensor([0.45, -0.3, 0.15], dtype=torch.float64)
    C = torch.randn(3, D, K, generator=g, dtype=torch.float64)
    r = torch.randn(x.shape[0], D, generator=g, dtype=torch.float64)
    # Every lag that can touch x_{t-l} with l < T is inside the kernel, so the
    # truncated FIR equals the infinite exponential sum under x_<0 = 0.
    jbar = sf.exponential_jbar(rho, C, x.shape[0])
    torch.testing.assert_close(sf.trace_data_term(x, r, rho, C), sf.fir_data_term(x, r, jbar),
                               rtol=1e-9, atol=1e-9)
    rel, _, _ = sf.reconstruct_exponentials(rho, jbar)
    assert float(rel) < 1e-8


def test_fit_stays_stable_and_recovers_a_sum_of_exponentials():
    g = torch.Generator().manual_seed(4)
    rho = torch.tensor([0.55, -0.25, 0.8], dtype=torch.float64)
    C = torch.randn(3, D, K, generator=g, dtype=torch.float64) * 0.2
    jbar = sf.exponential_jbar(rho, C, 18)
    fit = sf.fit_exponential_kernel(jbar, 3, steps=4, seed=4)
    assert fit["rho_abs_max"] < 0.95
    # Separated poles: a nonconvex fit, not an interpolant.  2e-3 is recovery, not noise.
    assert fit["rel_frobenius"] < 2e-3
    # The recursive trace of the fitted kernel matches the FIR of that same kernel.
    rh = fit["rho"].double()
    cc = fit["C"].double()
    x = torch.randn(18, N, generator=g, dtype=torch.float64)
    r = torch.randn(18, D, generator=g, dtype=torch.float64)
    jb = sf.exponential_jbar(rh, cc, 18)
    torch.testing.assert_close(sf.trace_data_term(x, r, rh, cc), sf.fir_data_term(x, r, jb),
                               rtol=1e-9, atol=1e-9)


def test_per_bin_contribution_ignores_future_residual_and_input():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, N, generator=g, dtype=torch.float64)
    r = torch.randn(16, D, generator=g, dtype=torch.float64)
    jbar = torch.randn(5, D, K, generator=g, dtype=torch.float64)
    per = sf.fir_per_time(x, r, jbar)
    torch.testing.assert_close(per.sum(0), sf.fir_data_term(x, r, jbar), rtol=1e-9, atol=1e-9)
    t = 7
    r2, x2 = r.clone(), x.clone()
    r2[t + 1:] = torch.randn_like(r2[t + 1:])
    x2[t + 1:] = torch.randn_like(x2[t + 1:])
    per2 = sf.fir_per_time(x2, r2, jbar)
    torch.testing.assert_close(per2[: t + 1], per[: t + 1], rtol=1e-9, atol=1e-9)
    # The trace contribution at t is likewise invariant to future inputs and residuals.
    rho = torch.tensor([0.4, -0.2], dtype=torch.float64)
    C = torch.randn(2, D, K, generator=g, dtype=torch.float64)
    # Prefix equality: run the accumulator on the prefix only.
    torch.testing.assert_close(sf.trace_data_term(x[: t + 1], r[: t + 1], rho, C),
                               sf.trace_data_term(x2[: t + 1], r2[: t + 1], rho, C),
                               rtol=1e-9, atol=1e-9)


def test_streaming_commit_does_not_read_later_trials_and_runs_under_no_grad():
    base = LinTrunk(6)
    xz, y, valid, wins, _, _, _, table, jbar = _episode(6, lens=(9, 13, 8))
    trials = _unpadded(xz, y, valid, wins)
    seen = []

    def hook(step, tab):
        seen.append(tab.clone())

    sf.stream_refine(base, trials, table, eta=0.5, rho=0.1, kind="fir", jbar=jbar, on_commit=hook)
    assert base.grad_enabled_during_forward and not any(base.grad_enabled_during_forward)
    assert len(seen) == 3

    def _blow(*_a, **_k):
        raise AssertionError("target-side streaming called autograd")

    orig_g, orig_b = torch.autograd.grad, torch.autograd.backward
    torch.autograd.grad, torch.autograd.backward = _blow, _blow
    try:
        sf.stream_refine(base, trials, table, eta=0.5, rho=0.1, kind="fir", jbar=jbar)
        rho = torch.tensor([0.2, -0.1], dtype=torch.float64)
        C = jbar[:2].transpose(0, 1).transpose(1, 2).contiguous()  # not used as a fit; shape [2,D,K] needed
        # jbar is [L,D,K]; build a legal coefficient instead
        C = torch.randn(2, D, K, dtype=torch.float64)
        sf.stream_refine(base, trials, table, eta=0.4, rho=0.0, kind="trace", poles=rho, coeff=C)
    finally:
        torch.autograd.grad, torch.autograd.backward = orig_g, orig_b

    # Scrambling a later trial does not change the commit of an earlier one.
    later = [(trials[0][0], trials[0][1], trials[0][2]),
             (trials[1][0], trials[1][1] * 3, trials[1][2]),
             (torch.randn_like(trials[2][0]), torch.randn_like(trials[2][1]), trials[2][2])]
    seen2 = []
    sf.stream_refine(base, later, table, eta=0.5, rho=0.1, kind="fir", jbar=jbar,
                     on_commit=lambda s, tab: seen2.append(tab.clone()))
    torch.testing.assert_close(seen2[0], seen[0], rtol=1e-9, atol=1e-9)
    assert not torch.allclose(seen2[1], seen[1])


def test_causality_statements_and_cost_identity():
    for key in ("r2_replay", "fir_reorder", "s1_fir", "trace", "s1_r1s"):
        text = sf.CAUSALITY[key]
        assert "future" in text or "Future" in text
    # Design note: N=256, K=10, D=2, M=4 -> 11,344 MACs / bin.
    mac = sf.trace_bin_macs(4, 2, 10, 256)
    assert mac["total"] == 11344
    assert mac["weight_x_act"] == 4 * 2 * 10 + 4 * 256
    assert mac["act_x_act"] == 4 * 10 * 256
    fir = sf.fir_bin_macs(33, 2, 10, 26)
    assert fir["weight_x_act"] == 33 * 2 * 10
    assert fir["act_x_act"] == 33 * 10 * 26
    sram = sf.stream_sram_bytes(n=26, k=10, n_lags=33, n_modes=4, include_fir=True, include_trace=False,
                                staging_bins=200)
    assert sram["FIR ring LxN"] == 33 * 26 * 4
    assert sram["U accumulator KxN"] == 10 * 26 * 4
    assert "one-trial staging (inputs+labels+mask)" in sram
    assert "total" in sram and sram["total"] > sram["FIR ring LxN"]
