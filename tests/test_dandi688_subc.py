"""sub-C loader extension, forbidden-session guard, and sub-M bit-identity (CPU; 3 small session loads)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ssm_decode import dandi688_data as d

SUBC_SMALL = "sub-C_ses-CO-20131022"       # smallest sub-C CO file (26 MB, 41 units)
APST_FINAL = (
    "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
)
# sha256 of (neural, behavior, trial_change, eval_mask, trial_bounds, unit_to_electrode, observed_mask,
# bin_edges) produced by the loader BEFORE the sub-C extension (2026-10-16, commit 7b80f69 + working tree).
SUBM_GOLDEN = {
    "sub-M_ses-CO-20150612": "204aab47eb5ba1310ccc189ee356cdaae228deb98329d43faa5c84bc5376dbe8",
    "sub-M_ses-CO-20140306": "bede9c2a38a5d1d9f485df20d27767f00b24c75145d4014352c5856407c0abd0",
}


def _digest(r) -> str:
    h = hashlib.sha256()
    for a in (r.neural, r.behavior, r.trial_change, r.eval_mask, np.asarray(r.trial_bounds),
              r.unit_to_electrode, r.observed_mask, r.bin_edges):
        a = np.ascontiguousarray(a)
        h.update(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("session", sorted(SUBM_GOLDEN))
def test_subm_loader_is_bit_identical_to_pre_extension(session):
    assert _digest(d.load_dandi688_recording(session)) == SUBM_GOLDEN[session]


@pytest.fixture(scope="module")
def subc():
    return d.load_dandi688_recording(SUBC_SMALL)


def test_subc_recording_contract(subc):
    assert subc.session == SUBC_SMALL and (subc.task, subc.split) == ("dandi688", "co")
    t, n = subc.neural.shape
    assert subc.behavior.shape == (t, 2) and subc.bin_edges.shape == (t + 1,)
    assert subc.unit_to_electrode.shape == (n,) and n == 41
    assert 0 <= subc.unit_to_electrode.min() and subc.unit_to_electrode.max() < d.CANONICAL_ELECTRODES
    assert np.array_equal(subc.observed_mask, np.isin(np.arange(96), subc.unit_to_electrode))
    assert np.isfinite(subc.neural).all() and (subc.neural >= 0).all()
    assert subc.eval_mask.any() and np.isfinite(subc.behavior[subc.eval_mask]).all()
    assert int(subc.trial_change.sum()) == len(subc.trial_bounds)


def test_subc_inventory_row_matches_loader(subc):
    (row,) = d.list_dandi688_co_sessions("sub-C", sessions=[SUBC_SMALL])
    assert (row["subject"], row["session"], row["date"]) == ("sub-C", SUBC_SMALL, "20131022")
    assert (row["n_bins"], row["n_units"]) == subc.neural.shape


def test_forbidden_set_equals_apst_final_split():
    assert set(d.SUBC_FINAL_SESSIONS) == set(APST_FINAL) == set(d.FORBIDDEN_SESSIONS)
    for path in (Path(d.APST_SRC) / "apst/legacy_dandi/protocol.json",
                 Path(d.DEFAULT_ROOT).parents[2] / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"):
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        final = blob.get("final_sessions") or blob.get("session_splits", {}).get("test")
        assert set(final) == set(d.FORBIDDEN_SESSIONS), path


@pytest.fixture
def no_nwb_access(monkeypatch):
    opened = []

    def boom(path):
        opened.append(str(path))
        raise AssertionError(f"NWB opened: {path}")
    monkeypatch.setattr(d, "_scan_session", boom)
    return opened


@pytest.mark.parametrize("session", APST_FINAL)
def test_loader_refuses_forbidden_before_opening_any_file(session, no_nwb_access):
    with pytest.raises(PermissionError):
        d.load_dandi688_recording(session)
    assert no_nwb_access == []


@pytest.mark.parametrize("session", APST_FINAL)
def test_inventory_refuses_forbidden_when_requested(session, no_nwb_access):
    with pytest.raises(PermissionError):
        d.list_dandi688_co_sessions("sub-C", sessions=[SUBC_SMALL, session])
    assert no_nwb_access == []


def test_whole_subject_inventory_never_opens_forbidden(monkeypatch):
    seen = []

    def stub(path):
        seen.append(Path(path).name)
        edges = np.arange(0.0, 1.0, 0.02)
        return {"edges": edges, "spikes": [np.array([0.1])], "trials": [], "canonical_keys": list(range(96))}
    monkeypatch.setattr(d, "_scan_session", stub)
    rows = d.list_dandi688_co_sessions("sub-C")
    names = {r["session"] for r in rows}
    assert len(rows) == 47 and not names & set(APST_FINAL)
    assert not any(any(f in n for f in APST_FINAL) for n in seen)


def test_rt_and_foreign_sessions_rejected():
    for bad in ("sub-C_ses-RT-20131009", "sub-J_ses-CO-20160405", "sub-M_ses-RT-20140606"):
        with pytest.raises(ValueError):
            d.load_dandi688_recording(bad)
    with pytest.raises(ValueError):
        d.dandi688_subject_root("sub-J")


def test_subm_final_sessions_stay_withheld_in_plan():
    with pytest.raises(PermissionError):
        d.dandi688_plan(targets=["sub-M_ses-CO-20150623"])


# ---- review fixes 2026-10-16: sub-M finals refused by the loader, _scan_session guarded --------------------
SUBM_FINAL = ("sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626")


@pytest.mark.parametrize("session", SUBM_FINAL)
def test_loader_refuses_subm_final_before_opening_any_file(session, no_nwb_access):
    with pytest.raises(PermissionError, match="sub-M final"):
        d.load_dandi688_recording(session)
    assert no_nwb_access == []


@pytest.mark.parametrize("session", SUBM_FINAL)
def test_inventory_refuses_subm_final_when_requested(session, no_nwb_access):
    with pytest.raises(PermissionError):
        d.list_dandi688_co_sessions("sub-M", sessions=[session])
    assert no_nwb_access == []


@pytest.mark.parametrize("session", APST_FINAL + SUBM_FINAL)
def test_scan_session_itself_refuses_final_sessions(session, monkeypatch):
    """The low-level scanner is guarded too: no caller can reach a held-out file through it (pynwb is not even
    imported: the guard runs first, so a non-existent path is enough)."""
    import sys
    monkeypatch.setitem(sys.modules, "pynwb", None)          # an import attempt would raise ImportError instead
    with pytest.raises(PermissionError):
        d._scan_session(Path("/nonexistent") / f"{session}{d.NWB_SUFFIX}")


def test_scan_session_guard_does_not_block_allowed_sessions(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pynwb", None)
    with pytest.raises(ImportError):                         # got past the guard, failed at the (blocked) import
        d._scan_session(Path("/nonexistent") / f"{SUBC_SMALL}{d.NWB_SUFFIX}")
    with pytest.raises(ImportError):
        d._scan_session(Path("/nonexistent") / "sub-M_ses-CO-20150612_behavior+ecephys.nwb")


def test_whole_subject_inventory_skips_subm_finals(monkeypatch):
    seen = []

    def stub(path):
        seen.append(Path(path).name)
        edges = np.arange(0.0, 1.0, 0.02)
        return {"edges": edges, "spikes": [np.array([0.1])], "trials": [], "canonical_keys": list(range(96))}
    monkeypatch.setattr(d, "_scan_session", stub)
    rows = d.list_dandi688_co_sessions("sub-M")
    assert rows and not {r["session"] for r in rows} & set(SUBM_FINAL)
    assert not any(any(f in n for f in SUBM_FINAL) for n in seen)


def test_all_refused_set_is_the_union():
    assert d.ALL_REFUSED_SESSIONS == set(SUBM_FINAL) | set(APST_FINAL)
    for s in d.TRAIN_SESSIONS + d.DEV_SESSIONS:
        d.assert_session_allowed(s)                           # allowed sub-M sessions pass


def test_plan_with_include_final_still_plans_paths_but_never_reads():
    plan = d.dandi688_plan(sessions=d.TRAIN_SESSIONS, targets=["sub-M_ses-CO-20150623"], include_final=True)
    assert plan["target_sessions"] == ["sub-M_ses-CO-20150623"] and plan["final_accessed"] is False
    with pytest.raises(PermissionError):
        d.dandi688_plan(sessions=d.TRAIN_SESSIONS, targets=["sub-M_ses-CO-20150623"], include_final=True,
                        with_metadata=True)
