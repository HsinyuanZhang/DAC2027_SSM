"""Electrode-level input handling for cross-session baselines (numpy only).

000688 sub-M re-sorts units every day, so a session's neural columns are NOT in a fixed order.  Published
methods that assume a fixed channel set (Stabilizer, NoMAD with day-0 read-in initialisation) need an
explicit, electrode-consistent column layout:

* ``pool_matrix`` / ``pool_counts``: sum the unit counts per electrode into the fixed 96 M1 channels
  (the multi-unit / threshold-crossing style input of these papers).  Column ``e`` is always electrode ``e``.
* ``electrode_matched_order``: re-order a session's sorted-unit columns to the baseline day's columns by
  shared electrode (rate-rank within an electrode), the layout used by the reviewer's NoMAD check.
* ``assert_columns_electrode_aligned``: guard that raises when target columns are not electrode-aligned to the
  baseline-day columns while the algorithm initialises the target read-in from the baseline day.

This module must stay importable without torch (it is loaded by path from the TF environment).
"""
from __future__ import annotations

import numpy as np

N_ELECTRODES = 96


class ColumnOrderError(ValueError):
    """Target columns are not electrode-aligned to the baseline-day columns."""


def pool_matrix(electrodes, n_electrodes: int = N_ELECTRODES) -> np.ndarray:
    """0/1 matrix ``P`` [n_electrodes, n_units] with ``P[e, u] = 1`` iff unit ``u`` sits on electrode ``e``."""
    el = np.asarray(electrodes, dtype=np.int64)
    if el.size and (el.min() < 0 or el.max() >= n_electrodes):
        raise ValueError(f"electrode index outside [0, {n_electrodes})")
    p = np.zeros((n_electrodes, len(el)))
    p[el, np.arange(len(el))] = 1.0
    return p


def pool_counts(x: np.ndarray, electrodes, n_electrodes: int = N_ELECTRODES) -> np.ndarray:
    """Sum unit counts ``x`` [T, n_units] per electrode -> [T, n_electrodes] (fixed channel order)."""
    out = np.zeros((x.shape[0], n_electrodes), dtype=x.dtype if x.dtype.kind == "f" else np.float64)
    np.add.at(out.T, np.asarray(electrodes, dtype=np.int64), x.T)
    return out


def electrode_matched_order(ref_elec, ref_rate, tgt_elec, tgt_rate, *, fill_unmatched: bool = True):
    """Column order that places target units into the baseline day's columns by shared electrode.

    Units on the same electrode are paired in descending mean-rate rank.  Returns ``(order, col_elec)``:
    ``order[c]`` is the target unit placed into baseline column ``c`` (``-1`` = zero-filled column) and
    ``col_elec[c]`` is the electrode of that unit when it was placed by electrode matching, ``-1`` otherwise.
    With ``fill_unmatched`` the left-over target units fill the left-over baseline columns in index order
    (arbitrary placement, flagged ``col_elec = -1``); extra target units beyond the baseline width are dropped.
    """
    ref_elec = np.asarray(ref_elec)
    tgt_elec = np.asarray(tgt_elec)
    ref_rate = np.asarray(ref_rate)
    tgt_rate = np.asarray(tgt_rate)
    order = -np.ones(len(ref_elec), dtype=np.int64)
    col_elec = -np.ones(len(ref_elec), dtype=np.int64)
    used = set()
    for e in np.intersect1d(ref_elec, tgt_elec):
        ua = np.flatnonzero(ref_elec == e)
        ub = np.flatnonzero(tgt_elec == e)
        ua = ua[np.argsort(-ref_rate[ua], kind="stable")]
        ub = ub[np.argsort(-tgt_rate[ub], kind="stable")]
        for p, q in zip(ua, ub):
            order[p] = q
            col_elec[p] = e
            used.add(int(q))
    if fill_unmatched:
        free = [c for c in range(len(ref_elec)) if order[c] < 0]
        rest = [u for u in range(len(tgt_elec)) if u not in used]
        for c, u in zip(free, rest):
            order[c] = u
    return order, col_elec


def assert_columns_electrode_aligned(ref_col_elec, tgt_col_elec, *, init_from_day0: bool = True,
                                     max_mismatch_frac: float = 0.0) -> dict:
    """Guard for algorithms that initialise the target read-in from the baseline day (NoMAD ``INIT_READIN_FROM_DAY0``).

    ``ref_col_elec`` / ``tgt_col_elec``: electrode label of each input column (``-1`` = zero padding or an
    arbitrary-fill column without a meaningful electrode).  Raises :class:`ColumnOrderError` when a labelled
    target column does not sit on the same electrode as the baseline column with the same index (or lies beyond
    the baseline width), unless the read-in is not initialised from day 0.
    """
    ref = np.asarray(ref_col_elec)
    tgt = np.asarray(tgt_col_elec)
    n = min(len(ref), len(tgt))
    labelled = tgt >= 0
    mism = np.zeros(len(tgt), dtype=bool)
    mism[:n] = labelled[:n] & (tgt[:n] != ref[:n])
    mism[n:] = labelled[n:]
    info = {"n_labelled": int(labelled.sum()), "n_mismatch": int(mism.sum()),
            "n_aligned": int((labelled & ~mism).sum())}
    if init_from_day0 and mism.sum() > max_mismatch_frac * max(labelled.sum(), 1):
        raise ColumnOrderError(
            f"{info['n_mismatch']}/{info['n_labelled']} target columns are not on the same electrode as the "
            "baseline-day column with the same index; day-0 read-in initialisation would map them to other units")
    return info
