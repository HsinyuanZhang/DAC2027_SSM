"""CPU tests for the DANDI 000688 sub-M CO adapter (two small session loads)."""
from __future__ import annotations

import numpy as np
import pytest

from ssm_decode import dandi688_data as d
from ssm_decode.data import Recording, split_support_query

SMALLEST_SESSION = "sub-M_ses-CO-20150612"  # smallest CO file, 15 M1 units
SECOND_SESSION = "sub-M_ses-CO-20150616"    # first dev session, 17 M1 units
SPINT_TRAIN = (
    "sub-M_ses-CO-20150511", "sub-M_ses-CO-20150512", "sub-M_ses-CO-20150610",
    "sub-M_ses-CO-20150611", "sub-M_ses-CO-20150612", "sub-M_ses-CO-20150615",
)
SPINT_DEV = ("sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617")
SPINT_FINAL = ("sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626")


@pytest.fixture(scope="module")
def metadata() -> dict:
    rows = d.list_dandi688_subm_co_sessions(sessions=[SMALLEST_SESSION, SECOND_SESSION])
    return {row["session"]: row for row in rows}


@pytest.fixture(scope="module")
def smallest() -> d.Dandi688Recording:
    return d.load_dandi688_recording(SMALLEST_SESSION)


@pytest.fixture(scope="module")
def second() -> d.Dandi688Recording:
    return d.load_dandi688_recording(SECOND_SESSION)


def test_recording_is_pilot_recording(smallest):
    assert isinstance(smallest, Recording)
    assert (smallest.task, smallest.split) == ("dandi688", "co")
    assert smallest.session == SMALLEST_SESSION
    assert smallest.path.is_file()


def test_shapes_match_session_metadata(smallest, second, metadata):
    for record in (smallest, second):
        row = metadata[record.session]
        assert record.neural.shape == (row["n_bins"], row["n_units"])
        assert record.behavior.shape == (row["n_bins"], 2)
        assert record.bin_edges.shape == (row["n_bins"] + 1,)
        assert len(record.trial_bounds) <= row["n_trials"]
        assert 0 < len(record.trial_bounds)
        total = float(record.neural.sum())
        expected = row["mean_rate_hz"] * row["n_units"] * row["duration_s"]
        assert total == pytest.approx(expected, rel=1e-3)


def test_neural_counts_and_behavior_on_eval_bins(smallest, second):
    for record in (smallest, second):
        x, y = record.neural, record.behavior
        assert np.isfinite(x).all() and (x >= 0).all()
        assert np.array_equal(x, np.round(x))  # integer spike counts
        assert np.isfinite(y[record.eval_mask]).all()
        assert np.linalg.norm(y[record.eval_mask], axis=1).mean() > 0.1
        assert record.eval_mask.any() and not record.eval_mask.all()


def test_trial_bounds_change_and_eval_consistency(smallest, second):
    for record in (smallest, second):
        n_bins = record.neural.shape[0]
        assert record.trial_change.shape == (n_bins,)
        assert record.eval_mask.shape == (n_bins,)
        bounds = np.asarray(record.trial_bounds)
        assert bounds.shape[1] == 2
        assert (bounds[:, 0] >= 0).all() and (bounds[:, 1] <= n_bins).all()
        assert (bounds[:, 1] > bounds[:, 0]).all()
        assert int(record.trial_change.sum()) == len(record.trial_bounds)
        for a, b in record.trial_bounds:
            assert record.trial_change[a]
        # eval bins sit inside trial bounds only, never in inter-trial gaps
        inside = np.zeros(n_bins, dtype=bool)
        for a, b in record.trial_bounds:
            inside[a:b] = True
        assert (record.eval_mask & ~inside).sum() == 0


def test_canonical_electrode_map(smallest, second):
    for record in (smallest, second):
        n_units = record.neural.shape[1]
        assert record.unit_to_electrode.shape == (n_units,)
        assert record.unit_to_electrode.dtype == np.int64
        assert ((record.unit_to_electrode >= 0) & (record.unit_to_electrode < d.CANONICAL_ELECTRODES)).all()
        assert record.observed_mask.shape == (d.CANONICAL_ELECTRODES,)
        assert record.observed_mask[record.unit_to_electrode].all()
        assert int(record.observed_mask.sum()) <= n_units  # multi-unit electrodes collapse


def test_variable_unit_counts_across_sessions(smallest, second):
    assert smallest.neural.shape[1] != second.neural.shape[1]
    assert smallest.neural.shape[0] != second.neural.shape[0]


def test_pilot_support_query_helper_still_works(smallest):
    support, query = split_support_query(smallest, support_trials=3, sequence=50)
    assert support.size and query.size
    assert not np.intersect1d(support, query).size


def test_default_plan_matches_spint_2015_protocol():
    plan = d.dandi688_plan()
    assert tuple(plan["source_sessions"]) == SPINT_TRAIN
    assert tuple(plan["target_sessions"]) == SPINT_DEV
    assert tuple(plan["final_sessions"]) == SPINT_FINAL
    assert plan["final_sessions_withheld"] and not plan["final_accessed"]
    assert not set(plan["source_sessions"]) & set(plan["target_sessions"])
    assert plan["protocol_schema"] == "dandi688_subm_2015_co_v1"


def test_plan_rejects_overlap_and_unknown_sessions():
    with pytest.raises(ValueError, match="overlap"):
        d.dandi688_plan(sessions=["sub-M_ses-CO-20140203"], targets=["sub-M_ses-CO-20140203"])
    with pytest.raises(KeyError):
        d.dandi688_plan(sessions=["sub-M_ses-CO-19999999"], targets=SPINT_DEV)


def test_plan_withholds_final_sessions_behind_flag():
    with pytest.raises(PermissionError, match="withheld"):
        d.dandi688_plan(sessions=SPINT_TRAIN, targets=["sub-M_ses-CO-20150623"])
    plan = d.dandi688_plan(sessions=SPINT_TRAIN, targets=["sub-M_ses-CO-20150623"], include_final=True)
    assert plan["target_sessions"] == ["sub-M_ses-CO-20150623"]
    assert not plan["final_sessions_withheld"]


def test_root_resolution_and_env_override(monkeypatch):
    assert d.dandi688_subm_root().name == "sub-M"
    monkeypatch.setenv(d.ROOT_ENV, str(d.DEFAULT_ROOT))
    assert d.dandi688_subm_root() == (d.DEFAULT_ROOT / "sub-M").resolve()
    monkeypatch.setenv(d.ROOT_ENV, str(d.DEFAULT_ROOT / "sub-M"))
    assert d.dandi688_subm_root() == (d.DEFAULT_ROOT / "sub-M").resolve()
    with pytest.raises(FileNotFoundError):
        d.dandi688_subm_root(root="/nonexistent/dandi688_root")


def test_loader_rejects_non_co_sessions():
    with pytest.raises(ValueError, match="sub-M CO"):
        d.load_dandi688_recording("sub-M_ses-RT-20140606")
    with pytest.raises(KeyError):
        d.load_dandi688_recording("sub-M_ses-CO-19990101")
