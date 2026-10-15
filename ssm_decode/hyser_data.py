"""Hyser HD-sEMG (PhysioNet hd-semg 2.0.0) reader and causal streaming front end.

Layout of ``1dof_dataset/subjectXX_sessionY``: 5 fingers x 3 samples (trials),
25 s each.  ``1dof_preprocess_fingerF_sampleS`` is 256-channel EMG at 2048 Hz
(51200 samples) and ``1dof_force_fingerF_sampleS`` is 5-channel force at 100 Hz
(2500 samples).  Both are WFDB format 16 (int16 little-endian, multiplexed).

``ndof_dataset/subjectXX_sessionY`` has the same file format with
``ndof_{preprocess,force,raw}_combinationC_sampleS`` for C in 1..15 and
S in 1..2 (25 s each, force is 5-channel at 100 Hz); ``random_dataset`` has
``random_{preprocess,force,raw}_sampleS`` for S in 1..5.  Only 1dof is wired
into the cache builder; ``ndof_trial_names`` lists the ndof file stems.

Front end (strictly causal, streaming-implementable):
  1. physical = (digital - baseline) / gain  (volts), converted to millivolts (amplifier-output scale; header units are V after the acquisition gain);
  2. per-channel rectification |x|;
  3. first-order leaky integrator ``y[n] = y[n-1] + a (|x[n]| - y[n-1])`` with
     ``a = 1 - exp(-1 / (fs * tau))`` (a one-pole low-pass, i.e. a first-order SSM);
  4. decimation to 50 Hz by sampling the integrator at the end of each 20 ms bin
     (bin k reads sample ``(k+1)*2048//50 - 1``, 2048/50 is not an integer);
  5. force: mean of the two 100 Hz samples that end bin k (indices 2k, 2k+1).
Output at bin k therefore depends only on inputs up to the end of bin k.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DEFAULT_ROOT = Path("/mnt/data/datasets/hd-semg")
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "results" / "data_cache" / "hyser_1dof"
EMG_FS = 2048
FORCE_FS = 100
OUT_FS = 50
N_CH = 256
N_FINGER = 5
N_SAMPLE = 3
N_BINS = 1250                       # 25 s at 50 Hz
DEFAULT_TAU_MS = 80.0

_GAIN = re.compile(r"^([-+0-9.eE]+)(?:\(([-+0-9]+)\))?(?:/(\S+))?$")


# ------------------------------------------------------------------ WFDB ----

def read_header(path) -> dict:
    """Parse a WFDB ``.hea`` file (single-segment, one file per record)."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    rec = lines[0].split()
    name, n_sig, fs, n_samp = rec[0], int(rec[1]), float(rec[2]), int(rec[3])
    signals = []
    for ln in lines[1:1 + n_sig]:
        f = ln.split()
        gain_match = _GAIN.match(f[2])
        if gain_match is None:
            raise ValueError(f"cannot parse gain field {f[2]!r} in {path}")
        gain = float(gain_match.group(1))
        adc_zero = int(f[4]) if len(f) > 4 else 0
        baseline = int(gain_match.group(2)) if gain_match.group(2) is not None else adc_zero
        signals.append({"file": f[0], "format": int(f[1]), "gain": gain, "baseline": baseline,
                        "units": gain_match.group(3) or "mV", "adc_res": int(f[3]),
                        "adc_zero": adc_zero, "init_value": int(f[5]), "checksum": int(f[6]),
                        "description": f[-1]})
    if len(signals) != n_sig:
        raise ValueError(f"{path}: header lists {len(signals)} signals, expected {n_sig}")
    if any(s["format"] != 16 for s in signals):
        raise ValueError("only WFDB format 16 is supported")
    if len({s["file"] for s in signals}) != 1:
        raise ValueError("multi-file records are not supported")
    return {"name": name, "n_sig": n_sig, "fs": fs, "n_samples": n_samp, "signals": signals}


def read_record(stem, *, verify=True):
    """Read ``stem + '.hea'/'.dat'``; return (physical [T, C] float64, header).

    Physical value = (digital - baseline) / gain.  With ``verify`` the sample
    count, initial value and 16-bit checksum of every signal are checked.
    """
    stem = str(stem)
    header = read_header(stem + ".hea")
    n_sig, n_samp = header["n_sig"], header["n_samples"]
    raw = np.fromfile(stem + ".dat", dtype="<i2")
    if raw.size != n_sig * n_samp:
        raise ValueError(f"{stem}: {raw.size} values in .dat, header says {n_sig}x{n_samp}")
    digital = raw.reshape(n_samp, n_sig)
    if verify:
        for j, s in enumerate(header["signals"]):
            if int(digital[0, j]) != s["init_value"]:
                raise ValueError(f"{stem}: initial value mismatch on channel {j}")
            total = int(digital[:, j].astype(np.int64).sum())
            wrapped = ((total + 2**15) % 2**16) - 2**15
            if wrapped != ((s["checksum"] + 2**15) % 2**16) - 2**15:
                raise ValueError(f"{stem}: checksum mismatch on channel {j}")
    gain = np.array([s["gain"] for s in header["signals"]])
    base = np.array([s["baseline"] for s in header["signals"]], dtype=np.float64)
    physical = (digital.astype(np.float64) - base) / gain
    return physical, header


# --------------------------------------------------------------- front end ---

def leaky_alpha(tau_ms: float, fs: float = EMG_FS) -> float:
    return float(1.0 - np.exp(-1.0 / (fs * tau_ms * 1e-3)))


def envelope(emg: np.ndarray, tau_ms: float = DEFAULT_TAU_MS, fs: float = EMG_FS) -> np.ndarray:
    """Causal rectify + one-pole low-pass along axis 0 (initial state 0)."""
    a = leaky_alpha(tau_ms, fs)
    return lfilter([a], [1.0, -(1.0 - a)], np.abs(emg), axis=0)


def bin_end_index(n_bins: int, fs: int = EMG_FS, out_fs: int = OUT_FS) -> np.ndarray:
    return ((np.arange(n_bins) + 1) * fs) // out_fs - 1


def decimate_emg(env: np.ndarray, out_fs: int = OUT_FS, fs: int = EMG_FS) -> np.ndarray:
    n_bins = env.shape[0] * out_fs // fs
    return env[bin_end_index(n_bins, fs, out_fs)]


def decimate_force(force: np.ndarray, fs: int = FORCE_FS, out_fs: int = OUT_FS) -> np.ndarray:
    """Causal decimation: mean of the ``fs/out_fs`` samples ending each bin."""
    step = fs // out_fs
    n_bins = force.shape[0] // step
    return force[:n_bins * step].reshape(n_bins, step, -1).mean(1)


def emg_features(stem, tau_ms: float = DEFAULT_TAU_MS) -> np.ndarray:
    """EMG record -> [50 Hz bins, 256] envelope in millivolts (amplifier-output scale) (float32)."""
    physical, _ = read_record(stem)
    return decimate_emg(envelope(physical * 1e3, tau_ms)).astype(np.float32)


def force_features(stem) -> np.ndarray:
    physical, _ = read_record(stem)
    return decimate_force(physical).astype(np.float32)


# ------------------------------------------------------------------ cache ----

def session_dir(subject: int, session: int, root=DEFAULT_ROOT, dataset="1dof_dataset") -> Path:
    return Path(root) / dataset / f"subject{subject:02d}_session{session}"


def trial_names_1dof():
    """Trial order used everywhere: index = (finger-1)*3 + (sample-1)."""
    return [(f, s) for f in range(1, N_FINGER + 1) for s in range(1, N_SAMPLE + 1)]


def ndof_trial_names():
    return [(c, s) for c in range(1, 16) for s in (1, 2)]


def _one_trial(args):
    folder, finger, sample, tau_ms, kind = args
    stem = Path(folder) / f"1dof_{kind}_finger{finger}_sample{sample}"
    if kind == "preprocess":
        return emg_features(stem, tau_ms)
    return force_features(stem)


def cache_path(subject: int, session: int, cache_dir=DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / f"s{subject:02d}_ses{session}.npz"


def build_session(subject: int, session: int, *, root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE,
                  tau_ms: float = DEFAULT_TAU_MS, overwrite=False) -> Path:
    out = cache_path(subject, session, cache_dir)
    if out.exists() and not overwrite:
        return out
    folder = session_dir(subject, session, root)
    xs, ys = [], []
    for finger, sample in trial_names_1dof():
        x = _one_trial((folder, finger, sample, tau_ms, "preprocess"))
        y = _one_trial((folder, finger, sample, tau_ms, "force"))
        if x.shape != (N_BINS, N_CH) or y.shape != (N_BINS, N_FINGER):
            raise ValueError(f"{folder} finger{finger} sample{sample}: shapes {x.shape} {y.shape}")
        xs.append(x)
        ys.append(y)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    meta = {"subject": subject, "session": session, "tau_ms": tau_ms, "bins": N_BINS,
            "emg_units": "millivolt (amplifier output)", "force_units": "volt (sensor amplifier output)"}
    np.savez(tmp, x=np.stack(xs), y=np.stack(ys),
             finger=np.array([f for f, _ in trial_names_1dof()]),
             sample=np.array([s for _, s in trial_names_1dof()]),
             meta=json.dumps(meta))
    tmp.replace(out)
    return out


def _build_job(job):
    return str(build_session(*job[:2], root=job[2], cache_dir=job[3], tau_ms=job[4], overwrite=job[5]))


def build_cache(subjects=range(1, 21), sessions=(1, 2), *, root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE,
                tau_ms=DEFAULT_TAU_MS, workers=6, overwrite=False):
    jobs = [(s, k, root, cache_dir, tau_ms, overwrite) for s in subjects for k in sessions]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_build_job, jobs))


def load_session(subject: int, session: int, cache_dir=DEFAULT_CACHE) -> dict:
    """Return {'x': [15,1250,256] f32, 'y': [15,1250,5] f32, 'finger': [15], 'sample': [15], 'meta': dict}."""
    with np.load(cache_path(subject, session, cache_dir)) as z:
        return {"x": z["x"], "y": z["y"], "finger": z["finger"], "sample": z["sample"],
                "meta": json.loads(str(z["meta"]))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tau-ms", type=float, default=DEFAULT_TAU_MS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for p in build_cache(workers=a.workers, tau_ms=a.tau_ms, overwrite=a.overwrite):
        print(p, flush=True)
