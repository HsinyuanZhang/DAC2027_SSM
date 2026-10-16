"""Guard tests for scripts/run_drift_curve.py (no GPU, no recordings)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_drift_curve as dc                                  # noqa: E402

SOURCES = ["20140203", "20140217", "20140218", "20140303", "20140304", "20140306"]


def test_default_target_list_is_valid():
    assert dc.validate_targets(dc.TARGET_DATES, SOURCES) == list(dc.TARGET_DATES)
    assert len(dc.TARGET_DATES) == 12


@pytest.mark.parametrize("bad", ["20150623", "20150625", "20150626", "20140307", "20140306", "20140203",
                                 "20140101", "2014062"])
def test_rejects_final_selection_source_early_or_malformed(bad):
    with pytest.raises(ValueError):
        dc.validate_targets(["20140626", bad], SOURCES)


def test_rejects_duplicates():
    with pytest.raises(ValueError):
        dc.validate_targets(["20140626", "20140626"], SOURCES)


def test_days_from_source():
    assert dc.days_from_source("20140306") == 0
    assert dc.days_from_source("20140626") == 112
    assert dc.days_from_source("20150617") == 468


def test_selection_set_check_rejects_noncausal_rule():
    ok = {str(B): {"sessions": ["20140307"]} for B in dc.B_LIST}
    pc_sel = {k: {str(B): {} for B in dc.B_LIST} for k in ("r2", "r1s_full", "r3_full")}
    dc.check_selection_sets(pc_sel, ok)
    bad = {**ok, "8": {"sessions": ["20140307", "20140627"]}}
    with pytest.raises(ValueError):
        dc.check_selection_sets(pc_sel, bad)
