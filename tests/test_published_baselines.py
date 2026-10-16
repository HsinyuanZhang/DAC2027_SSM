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


# ---- reference-dimension handling (second review: the target must not shrink the baseline-day latent / bridge) ----

def _lowrank_day(seed, n=8, k=3, m=3000, scale=None):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n, k))
    lat = rng.standard_normal((m, k))
    x = 10.0 + lat @ c.T + 0.2 * rng.standard_normal((m, n))
    return x, c, lat


def test_fa_bridge_exact_at_reference_dimension():
    x, _, _ = _lowrank_day(10)
    n = x.shape[1]
    rng = np.random.default_rng(11)
    z = (x - x.mean(0)) / x.std(0) @ rng.standard_normal((3, n)).T
    ref = pb.build_ref_day(name="r", x_fit=x, z_ref=z, electrodes=np.arange(n), ref_mean=x.mean(0), ref_std=x.std(0),
                           x_ref_trials=[], ref_conds=[], k=n, fa_min_priv=0.01, with_pca=False)
    xz = (x - ref.zmean) / ref.zstd
    lat = xz @ ref.beta.T + ref.o
    assert ref.a_fa.shape == (3, n)
    assert np.abs(lat @ ref.a_fa.T + ref.c_fa - z).max() < 1e-5


def test_bridge_below_reference_dimension_is_lossy():
    """The loss that the earlier cap of d at the target's dimension introduced on the baseline day."""
    n = 8
    x = 10.0 + np.random.default_rng(12).standard_normal((3000, n))        # isotropic: every dropped PC costs variance
    z = (x - x.mean(0)) / x.std(0) @ np.random.default_rng(13).standard_normal((3, n)).T

    def r2(k):
        ref = pb.build_ref_day(name="r", x_fit=x, z_ref=z, electrodes=np.arange(n), ref_mean=x.mean(0),
                               ref_std=x.std(0), x_ref_trials=[], ref_conds=[], k=k, fa_min_priv=0.01, with_fa=False)
        xz = (x - ref.zmean) / ref.zstd
        zh = (xz - xz.mean(0)) @ ref.pca_p @ ref.a_pca.T + ref.c_pca
        return 1 - ((z - zh) ** 2).sum() / ((z - z.mean(0)) ** 2).sum()
    assert r2(n) > 1 - 1e-9 and r2(5) < 0.95


class _Ref:
    def __init__(self, n):
        self.n = n

    def n_active(self, inp):
        return self.n


def test_resolve_d_is_reference_dimension_not_capped_by_target():
    import scripts.run_baselines_readin as rb
    ref = _Ref(43)
    target_fit = np.zeros((50, 60))
    target_fit[:, :26] = np.random.default_rng(0).standard_normal((50, 26))      # only 26 active target columns
    assert rb.resolve_d({"inp": "units", "d": "N"}, ref, target_fit) == 43
    assert rb.resolve_d({"inp": "units", "d": 10}, ref, target_fit) == 10
    assert rb.resolve_d({"inp": "units", "d": 5}, ref, target_fit[:, :3]) == 5
    assert rb.resolve_d({"inp": "units", "d": 10}, _Ref(7), target_fit) == 7        # never above the reference dimension


def test_stabilizer_target_with_fewer_active_dims_maps_into_reference_space():
    """Reference d = N_ref = 8; the target has 3 silent units.  The target latent is rank-deficient (5 active
    dimensions) and is mapped into the 8-d reference latent space; the read-in stays finite, has the reference-bridge
    shape and decodes the shared low-rank structure."""
    x_ref, c, lat = _lowrank_day(14, n=8, k=3)
    n = 8
    w = np.random.default_rng(15).standard_normal((2, n))
    z_ref = (x_ref - x_ref.mean(0)) / x_ref.std(0) @ w.T
    ref = pb.build_ref_day(name="r", x_fit=x_ref, z_ref=z_ref, electrodes=np.arange(n), ref_mean=x_ref.mean(0),
                           ref_std=x_ref.std(0), x_ref_trials=[], ref_conds=[], k=n, fa_min_priv=0.01, with_pca=False)
    x_new = x_ref.copy()
    x_new[:, [1, 4, 6]] = 0.0                                   # three units lost on the new day
    t, cvec, mean, std, info = pb.stabilizer_readin(ref, x_new, np.arange(n), k=n, align_frac=1.0, fa_min_priv=0.01)
    assert ref.a_fa.shape == (2, n)                             # bridge stays at the reference dimension
    assert t.shape == (2, n) and np.isfinite(t).all() and np.isfinite(cvec).all()
    assert info["n_matched"] == 5
    pred = ((x_new - mean) / std) @ t.T + cvec
    zc = z_ref - z_ref.mean(0)
    r2 = 1 - ((z_ref - pred) ** 2).sum() / (zc ** 2).sum()
    assert r2 > 0.3                                             # shared structure survives; exactness is not claimed
    # identical day at the reference dimension is reproduced exactly
    t0, c0, m0, s0, _ = pb.stabilizer_readin(ref, x_ref, np.arange(n), k=n, align_frac=1.0, fa_min_priv=0.01)
    pred0 = ((x_ref - m0) / s0) @ t0.T + c0
    assert np.abs(pred0 - z_ref).max() < 1e-4


def test_cca_target_with_fewer_active_dims_maps_into_reference_space():
    rng = np.random.default_rng(16)
    n, k, t_ph = 8, 8, 10
    x_ref, _, _ = _lowrank_day(17, n=n, k=3, m=2000)
    z_ref = (x_ref - x_ref.mean(0)) / x_ref.std(0) @ rng.standard_normal((2, n)).T
    conds = list(range(8))
    # per-condition trials: condition mean pattern + noise, same on both days
    pat = rng.standard_normal((8, n))
    trials = [10 + pat[c_] + 0.1 * rng.standard_normal((40, n)) for c_ in conds for _ in range(3)]
    labels = [c_ for c_ in conds for _ in range(3)]
    ref = pb.build_ref_day(name="r", x_fit=x_ref, z_ref=z_ref, electrodes=np.arange(n), ref_mean=x_ref.mean(0),
                           ref_std=x_ref.std(0), x_ref_trials=trials, ref_conds=labels, k=k, fa_min_priv=0.01,
                           t_phase=t_ph, with_fa=False)
    keep = [0, 2, 3, 5, 7]
    tr_new = [np.where(np.isin(np.arange(n), keep), t_, 0.0) for t_ in trials]
    x_new = np.concatenate(tr_new)
    res = pb.cca_readin(ref, x_new, tr_new, labels, k=k, ridge=1.0, t_phase=t_ph)
    assert res is not None
    t, c, mean, std, info = res
    assert t.shape == (2, n) and np.isfinite(t).all() and np.isfinite(c).all()
    assert np.abs(t[:, [1, 4, 6]]).max() < 1e-9                  # silent target columns carry no weight
