"""Electrode pooling and the NoMAD column-ordering guard (numpy only)."""
import numpy as np
import pytest

from ssm_decode import electrode_align as ea


def test_pool_counts_sums_per_electrode_in_fixed_channel_order():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 2.0]])
    el = np.array([7, 3, 7, 95])
    p = ea.pool_counts(x, el)
    assert p.shape == (2, 96)
    assert p[0, 7] == 4.0 and p[0, 3] == 2.0 and p[0, 95] == 4.0 and p[1, 7] == 0.0 and p[1, 95] == 2.0
    assert p.sum() == x.sum()
    assert np.allclose(p, x @ ea.pool_matrix(el).T)


def test_pooled_input_is_invariant_to_unit_order():
    """Re-sorting (permuting) units must not change the pooled input: columns are electrodes, not units."""
    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(50, 12)).astype(float)
    el = rng.integers(0, 96, size=12)
    perm = rng.permutation(12)
    assert np.array_equal(ea.pool_counts(x, el), ea.pool_counts(x[:, perm], el[perm]))


def test_electrode_matched_order_aligns_columns():
    ref_el, tgt_el = np.array([5, 5, 9, 11]), np.array([9, 5, 20])
    order, col = ea.electrode_matched_order(ref_el, np.array([3.0, 1.0, 2.0, 4.0]), tgt_el, np.array([4.0, 5.0, 1.0]))
    assert order[0] == 1 and col[0] == 5           # highest-rate unit on electrode 5
    assert order[2] == 0 and col[2] == 9
    assert col[1] == -1 and col[3] == -1            # unmatched columns carry no electrode label
    assert order[1] == 2 or order[3] == 2           # left-over unit (electrode 20) fills a left-over column
    assert (col[col >= 0] == ref_el[col >= 0]).all()


def test_guard_fails_when_target_columns_are_not_electrode_aligned():
    ref_col = np.array([4, 10, 33, 50, 71])
    own_order = np.array([10, 33, 4, 71, 50])       # target units sorted in a different order
    with pytest.raises(ea.ColumnOrderError):
        ea.assert_columns_electrode_aligned(ref_col, own_order, init_from_day0=True)
    # shuffled baseline day control: same units, permuted columns -> must fail
    with pytest.raises(ea.ColumnOrderError):
        ea.assert_columns_electrode_aligned(ref_col, ref_col[np.random.default_rng(1).permutation(5)])
    # a target wider than the baseline cannot be placed
    with pytest.raises(ea.ColumnOrderError):
        ea.assert_columns_electrode_aligned(ref_col, np.concatenate([ref_col, [90]]))


def test_guard_passes_for_aligned_layouts_and_without_day0_init():
    ref_col = np.array([4, 10, 33, 50, 71])
    ea.assert_columns_electrode_aligned(ref_col, ref_col)
    ea.assert_columns_electrode_aligned(ref_col, np.array([4, -1, 33, -1, 71]))   # padding / arbitrary fill allowed
    ea.assert_columns_electrode_aligned(np.arange(96), np.arange(96))              # pooled layout
    ea.assert_columns_electrode_aligned(ref_col, ref_col[::-1], init_from_day0=False)


def test_guard_consistent_with_electrode_matched_order():
    rng = np.random.default_rng(3)
    ref_el = np.sort(rng.choice(96, 17, replace=False))
    tgt_el = rng.permutation(np.sort(rng.choice(96, 15, replace=False)))
    order, col = ea.electrode_matched_order(ref_el, rng.random(17), tgt_el, rng.random(15))
    ea.assert_columns_electrode_aligned(ref_el, col)                  # reordered layout passes
    with pytest.raises(ea.ColumnOrderError):
        ea.assert_columns_electrode_aligned(ref_el, np.pad(tgt_el, (0, 2), constant_values=-1))   # raw own order fails
