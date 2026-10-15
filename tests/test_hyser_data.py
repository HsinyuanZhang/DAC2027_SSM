"""Unit checks for the Hyser WFDB reader and the causal streaming front end."""
from __future__ import annotations

import numpy as np
import pytest

from ssm_decode import hyser_data as h

FOLDER = h.session_dir(1, 1)
EMG = FOLDER / "1dof_preprocess_finger1_sample1"
FORCE = FOLDER / "1dof_force_finger1_sample1"
pytestmark = pytest.mark.skipif(not EMG.with_suffix(".hea").exists(), reason="Hyser data not mounted")


def test_header_parsing_real_file():
    hd = h.read_header(str(EMG) + ".hea")
    assert hd["n_sig"] == 256 and hd["fs"] == 2048 and hd["n_samples"] == 51200
    first = hd["signals"][0]
    assert first["description"] == "ED-8-8" and first["units"] == "V"
    assert first["gain"] == pytest.approx(99597.0777) and first["baseline"] == -2339
    assert first["init_value"] == -837 and first["checksum"] == -1526
    hf = h.read_header(str(FORCE) + ".hea")
    assert hf["n_sig"] == 5 and hf["fs"] == 100 and hf["n_samples"] == 2500
    assert [s["description"] for s in hf["signals"]] == ["thumb", "index", "middle", "ring", "little"]


def test_sample_counts_match_header_and_dat_size():
    for stem, rate in ((EMG, 2048), (FORCE, 100)):
        phys, hd = h.read_record(stem)           # verifies size, init value, checksum
        assert phys.shape == (hd["n_samples"], hd["n_sig"])
        assert hd["n_samples"] / rate == pytest.approx(25.0)
        assert (stem.with_suffix(".dat").stat().st_size == 2 * hd["n_samples"] * hd["n_sig"])


def test_checksum_verification_can_fail(tmp_path):
    for ext in (".hea", ".dat"):
        data = (str(FORCE) + ext)
        (tmp_path / ("r" + ext)).write_bytes(open(data, "rb").read())
    raw = bytearray((tmp_path / "r.dat").read_bytes())
    raw[10] ^= 0x55
    (tmp_path / "r.dat").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum|initial"):
        h.read_record(tmp_path / "r")


def test_physical_unit_conversion():
    phys, hd = h.read_record(EMG)
    raw = np.fromfile(str(EMG) + ".dat", dtype="<i2").reshape(-1, 256)
    s = hd["signals"][3]
    expect = (raw[:, 3].astype(np.float64) - s["baseline"]) / s["gain"]
    np.testing.assert_allclose(phys[:, 3], expect)
    # The header baseline is the channel mean of the preprocessed (band-passed) signal.
    assert abs(raw[:, 3].mean() - s["baseline"]) < 5
    # Amplifier-output scale: tens of mV rms at most, nonzero.
    assert 1e-4 < phys.std() < 1.0
    force, _ = h.read_record(FORCE)
    assert np.abs(force).max() < 10.0


def test_decimation_shapes():
    assert h.bin_end_index(1250)[-1] == 51199 and h.bin_end_index(1250)[0] == 39
    x = h.decimate_emg(np.zeros((51200, 3)))
    assert x.shape == (1250, 3)
    assert h.decimate_force(np.zeros((2500, 5))).shape == (1250, 5)


def test_envelope_is_causal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, 4))
    full = h.envelope(x)
    cut = 2000
    x2 = x.copy()
    x2[cut:] = rng.standard_normal((4096 - cut, 4)) * 50          # change the future only
    alt = h.envelope(x2)
    np.testing.assert_array_equal(full[:cut], alt[:cut])
    assert not np.allclose(full[cut:], alt[cut:])
    # decimated output at bin k depends only on inputs up to the end of bin k
    d1, d2 = h.decimate_emg(full), h.decimate_emg(alt)
    last_safe = int(np.searchsorted(h.bin_end_index(d1.shape[0]), cut, side="left"))
    np.testing.assert_array_equal(d1[:last_safe], d2[:last_safe])


def test_envelope_step_response_matches_time_constant():
    step = np.ones((4096, 1))
    y = h.envelope(step, tau_ms=80.0)[:, 0]
    n = int(0.080 * h.EMG_FS)
    assert y[n] == pytest.approx(1 - np.exp(-1), abs=0.01)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)


def test_force_decimation_is_causal_mean():
    f = np.arange(20, dtype=np.float64)[:, None]
    d = h.decimate_force(f)
    np.testing.assert_allclose(d[:, 0], [0.5 + 2 * k for k in range(10)])


def test_cache_round_trip(tmp_path):
    # Build one real session into a temp cache and verify values against a direct computation.
    h.build_session(1, 1, cache_dir=tmp_path)
    blob = h.load_session(1, 1, cache_dir=tmp_path)
    assert blob["x"].shape == (15, 1250, 256) and blob["x"].dtype == np.float32
    assert blob["y"].shape == (15, 1250, 5)
    assert list(blob["finger"][:4]) == [1, 1, 1, 2] and list(blob["sample"][:4]) == [1, 2, 3, 1]
    np.testing.assert_array_equal(blob["x"][0], h.emg_features(EMG))
    np.testing.assert_array_equal(blob["y"][0], h.force_features(FORCE))
    assert blob["meta"]["tau_ms"] == h.DEFAULT_TAU_MS
    assert np.isfinite(blob["x"]).all() and (blob["x"] >= 0).all()
