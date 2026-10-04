"""Synthetic checks for the Track A association math."""
from __future__ import annotations

import numpy as np
import pytest

from ssm_decode import association as assoc


def _make_fa_data(m=40000, n=40, k=5, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    h_true = rng.standard_normal((n, k))
    phi = rng.standard_normal((m, k))
    eps = rng.standard_normal((m, n)) * noise
    x = phi @ h_true.T + eps
    return x, h_true


def _principal_angles(a, b):
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def test_fa_recovers_subspace_and_noise():
    x, h_true = _make_fa_data()
    fa = assoc.factor_analysis_em(x, 5, iters=500, seed=0)
    angles = _principal_angles(h_true, fa["H"])
    assert angles.max() < 0.15
    assert np.allclose(fa["psi"], 0.3**2, rtol=0.35)


def test_posterior_readin_denoises():
    x, h_true = _make_fa_data(m=20000, noise=0.5)
    fa = assoc.factor_analysis_em(x, 5, iters=500, seed=1)
    beta = assoc.posterior_readin(fa["H"], fa["psi"])
    z_hat = (x - fa["mean"]) @ beta.T
    phi_true = np.linalg.lstsq(h_true, (x - x.mean(0)).T, rcond=None)[0].T
    # The FA frame is arbitrary, so check subspace recovery by linear map fit.
    ridge = np.linalg.solve(z_hat.T @ z_hat + np.eye(5), z_hat.T @ phi_true)
    residual = phi_true - z_hat @ ridge
    r2 = 1 - residual.var(0) / phi_true.var(0)
    assert r2.min() > 0.7


def test_procrustes_recovers_known_rotation():
    rng = np.random.default_rng(2)
    k = 6
    q0, _ = np.linalg.qr(rng.standard_normal((k, k)))
    b = rng.standard_normal((k, k))
    m_fit = b @ q0
    t = assoc.procrustes_frame(b, m_fit)
    assert np.allclose(t @ t.T, np.eye(k), atol=1e-10)
    assert np.allclose(t, q0.T, atol=1e-8) or np.allclose(b @ t - m_fit, 0, atol=1e-8)


def test_procrustes_partial_rank_is_orthogonal():
    rng = np.random.default_rng(3)
    b = rng.standard_normal((2, 6))
    m_fit = rng.standard_normal((2, 6))
    t = assoc.procrustes_frame(b, m_fit)
    assert np.allclose(t @ t.T, np.eye(6), atol=1e-10)
    # The label-visible rows must match the best possible orthogonal gain.
    u, s, vt = np.linalg.svd(b.T @ m_fit)
    assert np.allclose((b @ t - m_fit).sum(), (b @ (u @ vt) - m_fit).sum(), atol=1e-8)


def test_behavior_features_lags_and_nan():
    y = np.arange(12, dtype=float).reshape(6, 2)
    phi = assoc.behavior_features(y, lags=2)
    assert phi.shape == (6, 6)
    assert np.allclose(phi[2], [4, 5, 2, 3, 0, 1])
    assert np.isnan(phi[1, 4:]).all()
    y_nan = y.copy()
    y_nan[3, 0] = np.nan
    phi_nan = assoc.behavior_features(y_nan, lags=1)
    assert np.isnan(phi_nan[3, 0]) and not np.isnan(phi_nan[3, 1])
    assert np.isnan(phi_nan[4, 2]) and not np.isnan(phi_nan[4, 3])


def test_supervised_profiles_recovers_linear_map():
    rng = np.random.default_rng(4)
    m, n, k0 = 30000, 12, 6
    phi = rng.standard_normal((m, k0))
    h_true = rng.standard_normal((k0, n)) * 0.5
    x = phi @ h_true + rng.standard_normal((m, n)) * 0.1
    out = assoc.supervised_profiles(phi, x, lam=1.0)
    assert out["H"].shape == (n, k0)
    assert np.abs(out["H"].T - h_true).max() < 0.05


def test_am1_table_recovers_frame_and_decode():
    rng = np.random.default_rng(5)
    n, k, d, m = 30, 5, 2, 30000
    h_true = rng.standard_normal((n, k))
    psi_true = rng.uniform(0.2, 0.6, n)
    phi_all = rng.standard_normal((m + 400, k))
    x = phi_all @ h_true.T + rng.standard_normal((m + 400, n)) * np.sqrt(psi_true)
    # Labels exist only on the first 400 bins.
    head_true = rng.standard_normal((d, k))
    y = phi_all[:400] @ head_true.T
    # The source head is the target head in a rotated frame.
    q0, _ = np.linalg.qr(rng.standard_normal((k, k)))
    b_src = head_true @ q0
    out = assoc.am1_table(x[:400], x, y, b_src, k=k, lam=1.0, head_lam=1.0)
    assert out["table"].shape == (k, n)
    # Latents under the fitted table must decode held-out labels through the
    # source head, which is the deployment anchor for the frame.
    z_query = (x[400:] - out["fa"]["mean"]) @ out["table"].T
    y_query = phi_all[400:] @ head_true.T
    pred = z_query @ b_src.T
    r2 = 1 - ((y_query - pred) ** 2).sum() / ((y_query - y_query.mean(0)) ** 2).sum()
    assert r2 > 0.9
    # The frame must align the source head rows onto the label-fitted head.
    assert np.abs(b_src @ out["T"] - out["M"]).max() < 0.5


def test_shrinkage_pulls_toward_prior():
    rng = np.random.default_rng(6)
    phi = rng.standard_normal((40, 4))
    h_true = rng.standard_normal((4, 10))
    x = phi @ h_true
    prior = {"mean": np.zeros(4), "var": np.ones(4) * 4.0}
    strong = assoc.shrinkage_profiles(phi, x, prior, lam=1e-6)
    weak = assoc.shrinkage_profiles(phi, x, prior, lam=1e6)
    assert np.abs(strong["H"].T - h_true).max() < 1e-6
    assert np.abs(weak["H"]).max() < np.abs(strong["H"]).max()


def test_shared_profile_map_ridge():
    rng = np.random.default_rng(7)
    g_true = rng.standard_normal((6, 4))
    profiles = []
    tables = []
    for _ in range(3):
        p = rng.standard_normal((50, 6))
        profiles.append(p)
        tables.append((p @ g_true).T)
    out = assoc.fit_shared_profile_map(profiles, tables, lam=1e-3)
    assert np.abs(out["g"] - g_true).max() < 1e-3


def test_latent_covariance():
    z = np.random.default_rng(8).standard_normal((5000, 3)) * [1.0, 2.0, 0.5]
    cov = assoc.latent_covariance(z)
    assert np.allclose(np.diag(cov), [1.0, 4.0, 0.25], rtol=0.1)
