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
