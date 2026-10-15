"""NinaPro DB6 reader and causal streaming front end (grasp classification).

Files: ``S{s}_D{d}_T{t}.mat`` (subject 1..10, day 1..5, T1 = morning, T2 = afternoon).
Each is a continuous recording of one session (MATLAB v5, readable with scipy):
``emg`` [T, 16] float32 volts at 2000 Hz (14 Delsys Trigno electrodes; two
columns are all-zero, see ``ACTIVE_COLUMNS``), ``stimulus`` / ``restimulus``
[T, 1] (movement id as shown / relabeled to the measured onset), ``repetition`` /
``rerepetition`` (same, for the repetition counter), ``daytesting``, ``time``,
``subj``, ``acc`` (48 columns, unused), ``object`` / ``reobject`` (unused).

Grasp codes seen in ``restimulus`` are {1, 3, 4, 6, 9, 10, 11} (7 grasps) plus 0 (rest);
they are mapped to classes 1..7 and rest = 0 (``GRASP_CODES``).  A session holds
12 repetitions per grasp, recorded as 7 consecutive blocks of 12 repetitions.

Labels use the *relabeled* ``restimulus`` / ``rerepetition`` (aligned with the
measured movement onset, as recommended by the dataset authors), not the raw
``stimulus`` / ``repetition`` (which follow the on-screen cue).

Trial definition
----------------
``rerepetition`` partitions the stream into runs ``[rest preceding the movement |
movement]``: the counter increments when the previous movement ends, so each run
holds exactly the rest gap that leads into repetition ``r`` of a grasp plus that
repetition.  One *trial* = one maximal run of constant ``rerepetition > 0`` (taken at
the bin level), and it must contain exactly one non-zero ``restimulus`` code
(otherwise it is recorded in ``meta['bad_runs']`` and excluded).  The ~1 s of rest
after the last movement carries ``rerepetition == 0`` and belongs to no trial.
Trials are disjoint and cover every labeled repetition exactly once.  Trials only select which
bins are trained on / scored; decoding itself is one causal pass over the whole session stream
(model state is NEVER reset at trial boundaries, see ``db6_backbone``).

Front end (strictly causal, streaming-implementable):
  1. volts -> millivolts, per-channel rectification |x|;
  2. first-order leaky integrator ``y[n] = y[n-1] + a (|x[n]| - y[n-1])``
     (a = 1 - exp(-1 / (fs tau)); a one-pole low-pass, i.e. a first-order SSM),
     run over the whole session stream (state carried across trials);
  3. 50 Hz bins by sampling the integrator at the last sample of each 20 ms bin
     (bin k reads sample ``(k+1)*40 - 1``, 2000/50 = 40 exactly);
  4. the class label of bin k is ``restimulus`` at that same bin-end sample; its trial
     membership is the ``rerepetition`` run of that sample.
Output at bin k depends only on inputs up to the end of bin k.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

from .hyser_data import envelope as _envelope

DEFAULT_ROOT = Path("/mnt/data/datasets/ninapro_db6/extracted")
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "results" / "data_cache" / "db6"
EMG_FS = 2000
OUT_FS = 50
BIN = EMG_FS // OUT_FS                      # 40 samples per 20 ms bin
DEFAULT_TAU_MS = 80.0
N_SUBJECT, N_DAY = 10, 5
N_CLASS = 8                                 # rest + 7 grasps
GRASP_CODES = (1, 3, 4, 6, 9, 10, 11)       # restimulus codes -> classes 1..7
N_REP = 12
# 16 columns are stored, 14 are Delsys electrodes.  Columns 8 and 9 (0-based) are
# identically zero in every session (verified by ``scan_channels`` over all 100 files).
ACTIVE_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15)
N_CH = len(ACTIVE_COLUMNS)

_NAME = re.compile(r"S(\d+)_D(\d+)_T(\d+)\.mat$")


# ------------------------------------------------------------------ files ----

def mat_path(subject: int, day: int, tod: int, root=DEFAULT_ROOT) -> Path:
    """Path of ``S{subject}_D{day}_T{tod}.mat`` (tod 1 = morning, 2 = afternoon)."""
    group = "a" if day <= 3 else "b"
    return Path(root) / f"DB6_s{subject}_{group}" / f"S{subject}_D{day}_T{tod}.mat"


def session_keys(subjects=range(1, N_SUBJECT + 1), days=range(1, N_DAY + 1), tods=(1, 2)):
    return [(s, d, t) for s in subjects for d in days for t in tods]


def read_mat(path) -> dict:
    """Load one session file; returns numpy arrays with squeezed label vectors."""
    m = _NAME.search(str(path))
    d = sio.loadmat(str(path))
    out = {k: np.asarray(d[k]).ravel() for k in
           ("restimulus", "rerepetition", "stimulus", "repetition")}
    out["emg"] = np.asarray(d["emg"])
    out["daytesting"], out["time"], out["subj"] = (int(np.asarray(d[k]).ravel()[0])
                                                   for k in ("daytesting", "time", "subj"))
    if m is not None:
        s, dd, tt = (int(v) for v in m.groups())
        if (out["subj"], out["daytesting"], out["time"]) != (s, dd, tt):
            raise ValueError(f"{path}: header fields {(out['subj'], out['daytesting'], out['time'])} "
                             f"disagree with file name {(s, dd, tt)}")
    n = out["emg"].shape[0]
    for k in ("restimulus", "rerepetition", "stimulus", "repetition"):
        if out[k].shape[0] != n:
            raise ValueError(f"{path}: {k} length {out[k].shape[0]} != emg length {n}")
    if out["emg"].shape[1] != 16:
        raise ValueError(f"{path}: expected 16 emg columns, got {out['emg'].shape[1]}")
    return out


def channel_std(emg: np.ndarray) -> np.ndarray:
    return emg.astype(np.float64).std(0)


# --------------------------------------------------------------- front end ---

def envelope(emg_mv: np.ndarray, tau_ms: float = DEFAULT_TAU_MS, fs: float = EMG_FS) -> np.ndarray:
    """Causal rectify + one-pole low-pass along axis 0 (initial state 0)."""
    return _envelope(emg_mv, tau_ms, fs)


def bin_end_index(n_bins: int, fs: int = EMG_FS, out_fs: int = OUT_FS) -> np.ndarray:
    return ((np.arange(n_bins) + 1) * fs) // out_fs - 1


def class_of_code(code: np.ndarray) -> np.ndarray:
    """restimulus code -> class id (0 rest, 1..7 grasps); unknown codes raise."""
    lut = np.full(int(max(code.max(), max(GRASP_CODES))) + 1, -1, dtype=np.int8)
    lut[0] = 0
    for i, c in enumerate(GRASP_CODES, start=1):
        lut[c] = i
    out = lut[code]
    if (out < 0).any():
        raise ValueError(f"unexpected restimulus codes {np.unique(code[out < 0])}")
    return out


def bin_labels(restimulus: np.ndarray, rerepetition: np.ndarray):
    """Per-bin (class, rerepetition) sampled at each bin-end sample (label/bin alignment)."""
    n_bins = restimulus.shape[0] // BIN
    idx = bin_end_index(n_bins)
    return class_of_code(restimulus[idx]), rerepetition[idx].astype(np.int16)


def segment_trials(y: np.ndarray, rerep: np.ndarray):
    """Split bins into trials = maximal runs of constant rerepetition > 0 with one grasp.

    Returns (start, end, grasp, rep) int arrays (``end`` exclusive) and a list of
    rejected runs ``(start, end, rerep, grasps_in_run)``.
    """
    n = y.shape[0]
    cut = np.flatnonzero(np.diff(rerep)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, n]
    keep, bad = [], []
    for a, b in zip(starts, ends):
        r = int(rerep[a])
        if r <= 0:
            continue
        g = np.unique(y[a:b][y[a:b] > 0])
        if g.size != 1:
            bad.append((int(a), int(b), r, [int(v) for v in g]))
            continue
        keep.append((int(a), int(b), int(g[0]), r))
    arr = np.array(keep, dtype=np.int64).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], bad


def session_arrays(raw: dict, tau_ms: float = DEFAULT_TAU_MS) -> dict:
    """Raw session dict -> per-bin envelope ``x`` [Tb, 14] (mV, float32), labels, trial table."""
    emg_mv = raw["emg"][:, ACTIVE_COLUMNS].astype(np.float64) * 1e3
    env = _envelope(emg_mv, tau_ms, EMG_FS)
    n_bins = env.shape[0] // BIN
    x = env[bin_end_index(n_bins)].astype(np.float32)
    y, rerep = bin_labels(raw["restimulus"], raw["rerepetition"])
    start, end, grasp, rep, bad = segment_trials(y, rerep)
    return {"x": x, "y": y, "rerep": rerep, "trial_start": start, "trial_end": end,
            "trial_grasp": grasp.astype(np.int8), "trial_rep": rep.astype(np.int16), "bad_runs": bad}


# ------------------------------------------------------------------ cache ----

def tau_tag(tau_ms: float) -> str:
    return f"tau{int(round(tau_ms)):03d}"


def cache_path(subject: int, day: int, tod: int, tau_ms: float = DEFAULT_TAU_MS,
               cache_dir=DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / f"s{subject:02d}_d{day}_t{tod}_{tau_tag(tau_ms)}.npz"


def build_session(subject: int, day: int, tod: int, *, root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE,
                  taus=(DEFAULT_TAU_MS,), overwrite=False) -> list:
    outs = [cache_path(subject, day, tod, t, cache_dir) for t in taus]
    if all(o.exists() for o in outs) and not overwrite:
        return [str(o) for o in outs]
    raw = read_mat(mat_path(subject, day, tod, root))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ch_std = channel_std(raw["emg"])
    for tau, out in zip(taus, outs):
        a = session_arrays(raw, tau)
        meta = {"subject": subject, "day": day, "tod": tod, "tau_ms": tau, "fs": EMG_FS, "out_fs": OUT_FS,
                "emg_units": "millivolt", "active_columns": list(ACTIVE_COLUMNS),
                "n_samples": int(raw["emg"].shape[0]), "bad_runs": a["bad_runs"],
                "channel_std_volt": ch_std.tolist(), "label_source": "restimulus/rerepetition"}
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, x=a["x"], y=a["y"], rerep=a["rerep"], trial_start=a["trial_start"],
                 trial_end=a["trial_end"], trial_grasp=a["trial_grasp"], trial_rep=a["trial_rep"],
                 meta=json.dumps(meta))
        tmp.replace(out)
    return [str(o) for o in outs]


def _build_job(job):
    s, d, t, root, cache_dir, taus, overwrite = job
    return build_session(s, d, t, root=root, cache_dir=cache_dir, taus=taus, overwrite=overwrite)


def build_cache(keys=None, *, root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE, taus=(DEFAULT_TAU_MS,),
                workers=3, overwrite=False):
    keys = session_keys() if keys is None else keys
    jobs = [(s, d, t, root, cache_dir, tuple(taus), overwrite) for s, d, t in keys]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [p for r in pool.map(_build_job, jobs) for p in r]


def load_session(subject: int, day: int, tod: int, tau_ms: float = DEFAULT_TAU_MS,
                 cache_dir=DEFAULT_CACHE) -> dict:
    """Return the cached session: x [Tb,14] f32, y [Tb] i8, rerep, trial table, meta."""
    with np.load(cache_path(subject, day, tod, tau_ms, cache_dir)) as z:
        out = {k: z[k] for k in ("x", "y", "rerep", "trial_start", "trial_end", "trial_grasp", "trial_rep")}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def trial_slices(blob: dict):
    return [slice(int(a), int(b)) for a, b in zip(blob["trial_start"], blob["trial_end"])]


def trial_arrays(blob: dict):
    """List of (x_trial [L,14], y_trial [L]) in trial order."""
    return [(blob["x"][s], blob["y"][s]) for s in trial_slices(blob)]


def scan_channels(keys=None, root=DEFAULT_ROOT) -> dict:
    """Per-file channel std (volts) for all 16 columns; used to document the dead columns."""
    out = {}
    for s, d, t in (session_keys() if keys is None else keys):
        out[f"S{s}_D{d}_T{t}"] = channel_std(read_mat(mat_path(s, d, t, root))["emg"]).tolist()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--taus", type=float, nargs="+", default=[20.0, 40.0, 80.0, 160.0])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for p in build_cache(workers=a.workers, taus=a.taus, overwrite=a.overwrite):
        print(p, flush=True)
