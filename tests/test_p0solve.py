"""Closed-form checks for P0-Solve and the W_alpha coordinate map."""
import numpy as np
import pytest
import torch

from ssm_decode.p0solve import (
    accumulate_raw, add_moments, apply_readin, batch_hg, cost_model, direct_ls_solve,
    fit_anchor, map_cca_into_w0_coordinates, mean_std_from_raw, moments_to_normalized,
    phi_features, project_update, proxy_objective, shrink_tables, solve_p0,
    target_side_solve,
)


def _rand(rng, *shape):
    return rng.normal(size=shape)


def test_closed_form_matches_direct_least_squares():
    rng = np.random.default_rng(0)
    k, n, p = 3, 7, 4
    w0 = _rand(rng, k, n)
    f = _rand(rng, k, p)
    h = _rand(rng, n, p)
    g = h.T @ h + 0.1 * np.eye(p)          # SPD-ish Gram
    d = rng.uniform(0.3, 2.0, size=n)
    for lam, eps in ((0.3, 1e-3), (2.0, 0.1), (1e-2, 1e-6)):
        closed = solve_p0(w0, f, h, g, lam, eps, d)
        direct = direct_ls_solve(w0, f, h, g, lam, eps, d)
        np.testing.assert_allclose(closed, direct, rtol=1e-8, atol=1e-8)
        # the stationary point beats the prior and a random perturbation
        base = proxy_objective(w0, w0, f, h, g, lam, eps, d)
        got = proxy_objective(closed, w0, f, h, g, lam, eps, d)
        pert = proxy_objective(closed + 0.05 * _rand(rng, k, n), w0, f, h, g, lam, eps, d)
        assert got <= base + 1e-8
        assert got <= pert + 1e-6


def test_large_lambda_returns_w0():
    rng = np.random.default_rng(1)
    k, n, p = 4, 6, 3
    w0 = _rand(rng, k, n)
    f = _rand(rng, k, p)
    h = _rand(rng, n, p)
    g = _rand(rng, p, p)
    g = g.T @ g
    w = solve_p0(w0, f, h, g, lam=1e16, eps=1e-4, d=None)
    np.testing.assert_allclose(w, w0, rtol=1e-8, atol=1e-8)


def test_raw_accumulation_matches_normalized_batch():
    rng = np.random.default_rng(2)
    t, n, p = 40, 5, 8
    r = rng.normal(size=(t, n)) + rng.normal(size=(n,))
    cond = rng.integers(-1, 8, size=t)
    phase = np.linspace(-1, 1, t)
    phi = phi_features(cond, phase, p)
    mask = rng.random(t) > 0.25
    raw = accumulate_raw(r, phi, mask)
    # a second chunk must add exactly
    mid = t // 2
    raw2 = add_moments(accumulate_raw(r[:mid], phi[:mid], mask[:mid]),
                       accumulate_raw(r[mid:], phi[mid:], mask[mid:]))
    np.testing.assert_allclose(raw.sum_rphi, raw2.sum_rphi)
    np.testing.assert_allclose(raw.sum_phiphi, raw2.sum_phiphi)
    mean, std = mean_std_from_raw(raw)
    h, g = moments_to_normalized(raw, mean, std)
    x = (r - mean) / std
    hb, gb = batch_hg(x, phi, mask)
    np.testing.assert_allclose(h, hb, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g, gb, rtol=1e-10, atol=1e-10)
    # commit-time norm can differ from the accumulator's own norm; conversion still matches
    mean2 = mean + rng.normal(scale=0.2, size=n)
    std2 = np.abs(std * rng.uniform(0.5, 1.5, size=n)) + 0.1
    h2, g2 = moments_to_normalized(raw, mean2, std2)
    hb2, gb2 = batch_hg((r - mean2) / std2, phi, mask)
    np.testing.assert_allclose(h2, hb2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g2, gb2, rtol=1e-10, atol=1e-10)


def test_cca_and_w0_share_normalized_coordinates():
    rng = np.random.default_rng(3)
    k, n, t = 4, 9, 15
    t_cca = _rand(rng, k, n)
    c = _rand(rng, k)
    mean_c = _rand(rng, n)
    std_c = rng.uniform(0.2, 3.0, size=n)
    mean_p = _rand(rng, n)
    std_p = rng.uniform(0.2, 3.0, size=n)
    w, b = map_cca_into_w0_coordinates(t_cca, c, mean_c, std_c, mean_p, std_p)
    r = rng.normal(size=(t, n)) * 2 + 0.5
    via_cca = ((r - mean_c) / std_c) @ t_cca.T + c
    via_w0 = apply_readin(r, w, mean_p, std_p, b)
    np.testing.assert_allclose(via_w0, via_cca, rtol=1e-10, atol=1e-10)
    # shrinkage endpoints: alpha=0 is W0 (bias 0), alpha=1 is the CCA map
    w0 = _rand(rng, k, n)
    w_a, b_a = shrink_tables(w0, w, b, 0.0)
    np.testing.assert_allclose(w_a, w0)
    np.testing.assert_allclose(b_a, 0.0)
    w_b, b_b = shrink_tables(w0, w, b, 1.0)
    np.testing.assert_allclose(w_b, w)
    np.testing.assert_allclose(b_b, b)
    # and the mixed map is the same affine mix in these coordinates
    w_m, b_m = shrink_tables(w0, w, b, 0.25)
    mixed = apply_readin(r, w_m, mean_p, std_p, b_m)
    pure0 = apply_readin(r, w0, mean_p, std_p, None)
    pure1 = apply_readin(r, w, mean_p, std_p, b)
    np.testing.assert_allclose(mixed, 0.75 * pure0 + 0.25 * pure1, rtol=1e-10, atol=1e-10)


def test_projection_is_the_row_space_projector():
    rng = np.random.default_rng(4)
    k, n, p = 3, 8, 2
    h = _rand(rng, n, p)
    d = rng.uniform(0.5, 1.5, size=n)
    m = (h * d[:, None]).T                              # row space, [p, N]
    # a matrix already in the row space is unchanged
    coef = _rand(rng, k, p)
    inside = coef @ m
    np.testing.assert_allclose(project_update(inside, h, d), inside, rtol=1e-8, atol=1e-8)
    # a matrix orthogonal to every row of H^T D projects to 0
    _, _, vt = np.linalg.svd(m, full_matrices=True)
    null = vt[p:].T                                     # [N, N-p]
    outside = _rand(rng, k, n - p) @ null.T
    np.testing.assert_allclose(project_update(outside, h, d), 0.0, atol=1e-8)


def test_phi_reuses_eight_direction_labels():
    cond = np.array([0, 3, -1, 7])
    phase = np.array([-1.0, 0.0, 0.5, 1.0])
    p8 = phi_features(cond, phase, 8)
    assert p8.shape == (4, 8)
    np.testing.assert_allclose(p8.sum(1), [1, 1, 0, 1])
    assert p8[0, 0] == 1 and p8[1, 3] == 1 and p8[2].sum() == 0
    p16 = phi_features(cond, phase, 16)
    np.testing.assert_allclose(p16[:, :8], p8)
    np.testing.assert_allclose(p16[:, 8:], p8 * phase[:, None])


def test_anchor_recovers_a_linear_map():
    rng = np.random.default_rng(5)
    f_true = _rand(rng, 3, 4)
    phi = _rand(rng, 50, 4)
    u = phi @ f_true.T
    got = fit_anchor(u, phi, ridge=0.0)
    np.testing.assert_allclose(got, f_true, rtol=1e-8, atol=1e-8)


def test_target_side_no_grad_guard():
    rng = np.random.default_rng(6)
    k, n, t, p = 2, 4, 12, 8
    r = rng.normal(size=(t, n))
    phi = phi_features(rng.integers(0, 8, size=t), np.linspace(-1, 1, t), p)
    raw = accumulate_raw(r, phi, np.ones(t, dtype=bool))
    mean, std = mean_std_from_raw(raw)
    w0 = _rand(rng, k, n)
    f = _rand(rng, k, p)
    probe = []
    called = {"backward": 0}
    real = torch.autograd.backward

    def boom(*a, **k):
        called["backward"] += 1
        raise AssertionError("target side must not backprop")

    torch.autograd.backward = boom
    try:
        with torch.enable_grad():                    # caller has grad ON
            # a live tensor must not leak into the solve
            live = torch.nn.Parameter(torch.zeros(1))
            assert live.requires_grad
            table = target_side_solve(w0, f, raw, mean, std, lam=0.5, eps=1e-3, probe=probe)
    finally:
        torch.autograd.backward = real
    assert probe == [False]
    assert called["backward"] == 0
    assert isinstance(table, np.ndarray)
    assert table.shape == w0.shape
    ref = solve_p0(w0, f, *moments_to_normalized(raw, mean, std), 0.5, 1e-3)
    np.testing.assert_allclose(table, ref)


def test_cost_model_matches_the_accumulator_layout():
    c = cost_model(26, k=10, p=16)
    # sum r phi^T, sum phi phi^T, sum r, sum r^2, sum phi, count
    assert c["accumulator_floats"] == 26 * 16 + 16 * 16 + 2 * 26 + 16 + 1
    assert c["macs_per_bin"]["sum_rphi"] == 26 * 16
    assert c["solve_matrix"] == "16x16"
    c256 = cost_model(256, k=10, p=16)
    assert c256["macs_per_bin"]["sum_rphi"] == 256 * 16
    assert c256["macs_per_bin"]["total_full_G"] == 256 * 16 + 256 + 256


def test_shrink_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        shrink_tables(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), 1.5)
