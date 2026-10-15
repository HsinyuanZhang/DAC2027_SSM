"""Unit checks for the NinaPro DB6 reader, causal front end, labels and trial segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from ssm_decode import db6_data as d

REAL = d.mat_path(1, 1, 1)
needs_data = pytest.mark.skipif(not REAL.exists(), reason="DB6 data not mounted")


@pytest.fixture(scope="module")
def raw():
    return d.read_mat(REAL)


@needs_data
def test_real_file_parse(raw):
    emg = raw["emg"]
    assert emg.shape[1] == 16 and emg.shape[0] > 1_000_000          # ~12 min at 2 kHz
    assert emg.shape[0] / d.EMG_FS == pytest.approx(emg.shape[0] / 2000)
    assert (raw["subj"], raw["daytesting"], raw["time"]) == (1, 1, 1)
    # columns 8, 9 are identically zero; every active column carries signal
    sd = d.channel_std(emg)
    assert sd[8] == 0 and sd[9] == 0
    assert (sd[list(d.ACTIVE_COLUMNS)] > 0).all() and len(d.ACTIVE_COLUMNS) == d.N_CH == 14
    assert set(np.unique(raw["restimulus"])) == {0, *d.GRASP_CODES}
    assert raw["rerepetition"].min() >= 0 and raw["rerepetition"].max() == d.N_REP


@needs_data
def test_session_arrays_shapes_and_trials(raw):
    a = d.session_arrays(raw, 80.0)
    n_bins = raw["emg"].shape[0] // d.BIN
    assert a["x"].shape == (n_bins, 14) and a["y"].shape == (n_bins,)
    assert np.isfinite(a["x"]).all() and (a["x"] >= 0).all()
    assert not a["bad_runs"]
    # 7 grasps x 12 repetitions, each exactly once
    pairs = list(zip(a["trial_grasp"].tolist(), a["trial_rep"].tolist()))
    assert len(pairs) == 84 and len(set(pairs)) == 84
    assert set(pairs) == {(g, r) for g in range(1, 8) for r in range(1, 13)}
    # trials are disjoint, ordered, and tile the labeled part of the stream
    s, e = a["trial_start"], a["trial_end"]
    assert (e > s).all() and (s[1:] >= e[:-1]).all()
    covered = np.zeros(n_bins, bool)
    for i, j in zip(s, e):
        assert not covered[i:j].any()
        covered[i:j] = True
    assert (covered == (a["rerep"] > 0)).all()
    # every trial holds exactly one grasp class (plus rest), and it is the recorded one
    for i, j, g in zip(s, e, a["trial_grasp"]):
        present = set(np.unique(a["y"][i:j]).tolist())
        assert present == {0, int(g)}


def test_envelope_causal_prefix_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8000, 14))
    full = d.envelope(x, 80.0)
    cut = 5013
    np.testing.assert_allclose(d.envelope(x[:cut], 80.0), full[:cut], atol=1e-12)
    changed = x.copy()
    changed[cut:] += 10.0
    np.testing.assert_allclose(d.envelope(changed, 80.0)[:cut], full[:cut], atol=1e-12)


def test_envelope_step_response_and_bin_sampling():
    tau = 80.0
    x = np.ones((4000, 1))
    env = d.envelope(x, tau)
    a = 1.0 - np.exp(-1.0 / (d.EMG_FS * tau * 1e-3))
    assert env[0, 0] == pytest.approx(a)
    assert env[int(0.080 * d.EMG_FS) - 1, 0] == pytest.approx(1 - np.exp(-1), abs=0.01)
    idx = d.bin_end_index(100)
    assert idx[0] == 39 and idx[1] == 79 and idx[-1] == 3999
    assert (np.diff(idx) == d.BIN).all()
    # rectification: sign flips do not matter
    np.testing.assert_allclose(d.envelope(-x, tau), env)


def test_label_bin_alignment_synthetic():
    n = 40 * 100
    rest = np.zeros(n, np.int8)
    rerep = np.zeros(n, np.int8)
    rest[40 * 10 + 20:40 * 30 + 5] = 6                      # movement begins mid-bin 10, ends mid-bin 30
    rerep[:40 * 40] = 1
    y, rr = d.bin_labels(rest, rerep)
    assert y.shape == (100,)
    assert y[9] == 0 and y[10] == 4                         # bin 10 ends at sample 439 >= 420 => labeled by its end sample
    assert y[10] == d.class_of_code(rest[[40 * 11 - 1]])[0]
    assert y[11] == 4 and y[29] == 4                        # code 6 is the 4th grasp
    assert y[30] == d.class_of_code(rest[[40 * 31 - 1]])[0] == 0
    assert (rr[:40] == 1).all() and (rr[40:] == 0).all()
    with pytest.raises(ValueError):
        d.class_of_code(np.array([0, 2]))


def test_segment_trials_synthetic():
    # [rest|g1 r1] [rest|g1 r2] [rest|g3 r1] trailing rest with rerep 0
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 2, 2, 0, 0], np.int8)
    rr = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0], np.int16)
    s, e, g, r, bad = d.segment_trials(y, rr)
    assert not bad
    assert s.tolist() == [0, 4, 7] and e.tolist() == [4, 7, 10] and g.tolist() == [1, 1, 2] and r.tolist() == [1, 2, 1]
    # a run with two grasps or no grasp is rejected, not silently kept
    s, e, g, r, bad = d.segment_trials(np.array([0, 1, 2, 0, 0], np.int8), np.array([1, 1, 1, 2, 2], np.int16))
    assert len(s) == 0 and len(bad) == 2


@needs_data
def test_cache_roundtrip_and_trial_arrays(raw, tmp_path):
    outs = d.build_session(1, 1, 1, cache_dir=tmp_path, taus=(40.0, 80.0))
    assert len(outs) == 2 and all(p.endswith(".npz") for p in outs)
    blob = d.load_session(1, 1, 1, 80.0, cache_dir=tmp_path)
    a = d.session_arrays(raw, 80.0)
    np.testing.assert_array_equal(blob["x"], a["x"])
    np.testing.assert_array_equal(blob["y"], a["y"])
    np.testing.assert_array_equal(blob["trial_start"], a["trial_start"])
    np.testing.assert_array_equal(blob["trial_rep"], a["trial_rep"])
    assert blob["meta"]["tau_ms"] == 80.0 and blob["meta"]["subject"] == 1
    assert blob["x"].dtype == np.float32 and blob["y"].dtype == np.int8
    tr = d.trial_arrays(blob)
    assert len(tr) == 84 and all(x.shape[0] == y.shape[0] for x, y in tr)
    # second call with existing files is a no-op, tau differs between the two caches
    assert d.build_session(1, 1, 1, cache_dir=tmp_path, taus=(40.0, 80.0)) == outs
    other = d.load_session(1, 1, 1, 40.0, cache_dir=tmp_path)
    assert not np.allclose(other["x"], blob["x"])


@needs_data
def test_file_name_header_consistency_check(tmp_path):
    import shutil
    wrong = tmp_path / "S2_D1_T1.mat"
    shutil.copy(REAL, wrong)                                # subject field says 1, name says 2
    with pytest.raises(ValueError, match="disagree"):
        d.read_mat(wrong)


@pytest.mark.skipif(not d.mat_path(2, 2, 2).exists(), reason="DB6 data not mounted")
def test_known_truncated_session_is_detected():
    a = d.session_arrays(d.read_mat(d.mat_path(2, 2, 2)), 80.0)
    assert len(a["trial_start"]) == 12 and set(a["trial_grasp"].tolist()) == {1}   # one grasp only
