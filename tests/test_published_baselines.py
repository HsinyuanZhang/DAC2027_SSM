"""Unit tests for the published read-in-alignment baselines (numpy only)."""
import numpy as np

from ssm_decode import published_baselines as pb


def _rot(k, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((k, k)))
    return q


def test_orthonormal_transform_recovers_rotation():
    rng = np.random.default_rng(0)
    m1 = rng.standard_normal((20, 5))
    r = _rot(5, 1)
    m2 = m1 @ r                       # m1 = m2 @ T.T  ->  T = r.T
    t = pb.orthonormal_transform(m1, m2)
    assert np.allclose(m2 @ t.T, m1, atol=1e-9)


def test_identify_stable_rows_drops_corrupted_rows():
    rng = np.random.default_rng(2)
    m1 = rng.standard_normal((30, 4))
    m2 = m1 @ _rot(4, 3)
    bad = [3, 11, 17]
    m2[bad] += 5.0 * rng.standard_normal((3, 4))
    mask = pb.identify_stable_rows(m1, m2, 27, 0.01)
    assert mask.sum() == 27
    assert not mask[bad].any()


def test_fa_recovers_loading_subspace():
    rng = np.random.default_rng(4)
    n, k, m = 12, 3, 4000
    c = rng.standard_normal((n, k))
    z = rng.standard_normal((m, k))
    x = z @ c.T + 0.3 * rng.standard_normal((m, n))
    fa = pb.fa_em(x, k, min_priv=0.01, restarts=2, iters=300)
    q1, _ = np.linalg.qr(c)
    q2, _ = np.linalg.qr(fa["C"])
    assert np.linalg.svd(q1.T @ q2, compute_uv=False).min() > 0.98


def test_cca_aligns_rotated_views():
    rng = np.random.default_rng(5)
    base = rng.standard_normal((200, 4)) @ np.diag([3, 2, 1.5, 1])
    r = _rot(4, 6)
    lt = base @ r + 0.01 * rng.standard_normal(base.shape)
    mr, mt, m_r, m_t, corr = pb.cca(base, lt, 0.0)
    assert corr.min() > 0.99
    trans = mt @ np.linalg.inv(mr)
    aligned = (lt - m_t) @ trans + m_r
    assert np.abs(aligned - base).max() < 0.2


def test_match_units_by_electrode_rank_order():
    ea, eb = np.array([5, 5, 9]), np.array([9, 5])
    ia, ib = pb.match_units_by_electrode(ea, np.array([3.0, 1.0, 2.0]), eb, np.array([4.0, 5.0]))
    pairs = dict(zip(ia.tolist(), ib.tolist()))
    assert pairs == {0: 1, 2: 0}      # highest-rate unit on electrode 5 pairs with the only unit there


def test_fa_fit_silent_columns_have_zero_loadings():
    rng = np.random.default_rng(7)
    z = rng.standard_normal((2000, 3))
    c = rng.standard_normal((6, 3))
    x = np.zeros((2000, 10))
    x[:, [1, 2, 4, 5, 8, 9]] = z @ c.T + 0.3 * rng.standard_normal((2000, 6))
    fa = pb.fa_fit(x, 3, min_priv=0.01, restarts=2)
    assert fa["C"].shape == (10, 3)
    assert np.abs(fa["C"][[0, 3, 6, 7]]).max() == 0.0
    beta, o = pb.fa_beta(fa)
    assert beta.shape == (3, 10) and np.isfinite(beta).all() and np.isfinite(o).all()


def test_latent_dim_is_free_and_bridge_exact_when_d_equals_n():
    """K x d bridge: with d = N (full-rank input) the bridge reproduces the known AP latent exactly."""
    rng = np.random.default_rng(8)
    n, k, m = 7, 3, 3000
    x = rng.poisson(3.0, size=(m, n)).astype(float) * rng.uniform(0.5, 2.0, n)
    w = rng.standard_normal((k, n))
    z = (x - x.mean(0)) / x.std(0) @ w.T
    ref = pb.build_ref_day(name="r", x_fit=x, z_ref=z, electrodes=np.arange(n), ref_mean=x.mean(0), ref_std=x.std(0),
                           x_ref_trials=[], ref_conds=[], k=n, fa_min_priv=0.01)
    xz = (x - ref.zmean) / ref.zstd
    lat_p = (xz - xz.mean(0)) @ ref.pca_p
    assert ref.a_pca.shape == (k, n)
    assert np.abs(lat_p @ ref.a_pca.T + ref.c_pca - z).max() < 1e-6
    ref3 = pb.build_ref_day(name="r", x_fit=x, z_ref=z, electrodes=np.arange(n), ref_mean=x.mean(0), ref_std=x.std(0),
                            x_ref_trials=[], ref_conds=[], k=3, fa_min_priv=0.01)
    assert ref3.a_fa.shape == (k, 3)


def test_stabilizer_on_pooled_channels_ignores_silent_channels():
    """Electrode-pooled input: channel c is electrode c in every session; silent channels form no stable pair."""
    rng = np.random.default_rng(9)
    n, k, m = 20, 3, 4000
    active = np.array([2, 3, 5, 8, 9, 12, 13, 17])
    c = rng.standard_normal((len(active), k))
    x_ref = np.zeros((m, n))
    x_ref[:, active] = 5 + np.abs(rng.standard_normal((m, k)) @ c.T + 0.3 * rng.standard_normal((m, len(active))))
    ref = pb.build_ref_day(name="r", x_fit=x_ref, z_ref=rng.standard_normal((m, 2)), electrodes=np.arange(n),
                           ref_mean=x_ref.mean(0), ref_std=x_ref.std(0) + 1e-9, x_ref_trials=[], ref_conds=[],
                           k=k, fa_min_priv=0.01)
    x_new = x_ref.copy()
    t, cvec, mean, std, info = pb.stabilizer_readin(ref, x_new, np.arange(n), k=k, align_frac=1.0, fa_min_priv=0.01)
    assert info["n_matched"] == len(active)               # silent channels dropped from the pairing
    assert t.shape == (2, n) and np.abs(t[:, [0, 1, 4, 6]]).max() < 1e-9
