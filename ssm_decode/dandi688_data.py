"""DANDI 000688 subject-M center-out (CO) data access for the SSM pilot.

Raw NWB files live on the local SSD by default.  Set ``SSM_DANDI688_ROOT``
to redirect all access to another tree, for example a merged CO+RT root.
The six sub-M RT sessions currently live on HDD under
``/mnt/data/work_host/datasets/DANDI_000688_missing/000688`` and are out of
scope, but the root resolution already accepts a merged tree without change.

Neural columns are M1 sorted single units, so the unit count varies between
sessions.  ``unit_to_electrode`` indexes each unit into the 96-electrode
canonical M1 map (bank A/B/C x pin 1..32, sorted by hardware key), matching
the frozen SPINT sub-M 2015 CO protocol.  The APST loader
(``apst.data.dandi688``) is reused for session enumeration but not for array
loading: it exposes no electrode map and no evaluation mask, so this adapter
re-implements its verified binning (spike-range 20 ms edges, per-unit
histograms, cursor velocity interpolated at bin centers) in one NWB pass.

eval_mask rule (the raw NWB carries no mask, unlike FALCON files):
a bin is eval-eligible iff (1) it belongs to a trials-table interval whose
``result`` is ``"R"`` (rewarded), and (2) its bin center lies inside the
``cursor_vel`` timestamp support, so no extrapolated zero velocity produced
by the edge clamping of the interpolation is ever scored.  Inter-trial gaps,
aborted/failed/incomplete trials, and velocity-free head/tail segments are
excluded.  ``trial_bounds`` carries one half-open bin range per trials-table
row (any result, table order, clipped to the binned spike support).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ssm_decode.data import Recording

APST_SRC = Path(__file__).resolve().parents[2] / "APST" / "src"
if str(APST_SRC) not in sys.path:
    sys.path.insert(0, str(APST_SRC))
from apst.data.dandi688 import list_dandi688_sessions  # noqa: E402

ROOT_ENV = "SSM_DANDI688_ROOT"
DEFAULT_ROOT = Path("/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688")
SUBJECT = "sub-M"
TASK_SURFACE = "CO"
BIN_SECONDS = 0.020
BIN_SIZE_MS = 20
CANONICAL_ELECTRODES = 96
NWB_SUFFIX = "_behavior+ecephys.nwb"

SPINT_PROTOCOL_SCHEMA = "dandi688_subm_2015_co_v1"
SPINT_PROTOCOL_REFERENCE = (
    "/mnt/data/work_host/SPINT/apst_final_week_20260916/"
    "dandi688_subm_2015_20260921/src/protocol.py"
)
TRAIN_SESSIONS = (
    "sub-M_ses-CO-20150511", "sub-M_ses-CO-20150512", "sub-M_ses-CO-20150610",
    "sub-M_ses-CO-20150611", "sub-M_ses-CO-20150612", "sub-M_ses-CO-20150615",
)
DEV_SESSIONS = ("sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617")
FINAL_SESSIONS = ("sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626")


@dataclass(frozen=True)
class Dandi688Recording(Recording):
    """A sub-M CO recording on per-session M1 sorted units.

    ``unit_to_electrode`` maps each neural column to its index in the
    96-electrode canonical M1 map, ``observed_mask`` marks the occupied
    electrodes of that map, and ``bin_edges`` are the ``[T + 1]`` physical
    spike-time edges behind ``neural`` and ``behavior``.
    """

    unit_to_electrode: np.ndarray
    observed_mask: np.ndarray
    bin_edges: np.ndarray


def dandi688_subm_root(root: str | Path | None = None) -> Path:
    """Resolve the tree holding ``sub-M`` NWB files (env overridable)."""
    base = Path(root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT).expanduser().resolve()
    if base.name == SUBJECT and base.is_dir():
        return base
    for candidate in (base, base / "000688", base / "dandi_000688"):
        if (candidate / SUBJECT).is_dir():
            return candidate / SUBJECT
    raise FileNotFoundError(
        f"no {SUBJECT} directory under {base}; set {ROOT_ENV} or pass root="
    )


def _scan_session(path: Path) -> dict:
    """Open one NWB once and gather M1 units, spike edges, velocity and trials."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        electrodes = nwb.electrodes.to_dataframe()
        required = {"group_name", "location", "bank", "pin", "label"}
        if not required <= set(electrodes.columns):
            raise ValueError(f"{path.name}: electrode table lacks {sorted(required - set(electrodes.columns))}")
        key_by_id: dict[int, tuple[str, str, int, str]] = {}
        for electrode_id, row in electrodes.iterrows():
            if str(row.group_name) == "electrode_group_M1" and str(row.location) == "Primary Motor Cortex":
                key_by_id[int(electrode_id)] = (str(row.group_name), str(row.bank), int(row.pin), str(row.label))
        canonical = sorted(key_by_id.values())
        if len(canonical) != CANONICAL_ELECTRODES or len(set(canonical)) != len(canonical):
            raise ValueError(f"{path.name}: expected {CANONICAL_ELECTRODES} unique M1 electrode keys, got {len(canonical)}")
        canonical_index = {key: index for index, key in enumerate(canonical)}
        unit_rows: list[int] = []
        unit_to_electrode: list[int] = []
        for unit_row, region in enumerate(units["electrodes"]):
            indices = getattr(region, "index", None)
            if indices is None or len(indices) != 1:
                continue
            electrode_id = int(indices[0])
            if electrode_id not in key_by_id:
                continue
            unit_rows.append(unit_row)
            unit_to_electrode.append(canonical_index[key_by_id[electrode_id]])
        if not unit_rows:
            raise ValueError(f"{path.name}: no M1 sorted units")
        spikes = [np.asarray(units.iloc[row]["spike_times"], dtype=np.float64) for row in unit_rows]
        if any(spike.size == 0 for spike in spikes):
            raise ValueError(f"{path.name}: an M1 unit has no spike times")
        all_spikes = np.concatenate(spikes)
        edges = np.arange(float(all_spikes.min()), float(all_spikes.max()) + BIN_SECONDS, BIN_SECONDS)
        series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        vel_times = np.asarray(series.timestamps[:], dtype=np.float64)
        vel_data = np.asarray(series.data[:], dtype=np.float64)
        trials_df = nwb.intervals["trials"].to_dataframe()
        trials = []
        for raw_row, row in trials_df.iterrows():
            result = row.get("result")
            trials.append({
                "raw_trial_row": int(raw_row),
                "start_time": float(row.start_time),
                "stop_time": float(row.stop_time),
                "result": None if result is None else str(result),
            })
        starts = [t["start_time"] for t in trials]
        if any(b < a for a, b in zip(starts, starts[1:])):
            raise ValueError(f"{path.name}: trials table is not ordered by start time")
    return {
        "path": path, "canonical_keys": canonical,
        "unit_rows": np.asarray(unit_rows, dtype=np.int64),
        "unit_to_electrode": np.asarray(unit_to_electrode, dtype=np.int64),
        "spikes": spikes, "edges": edges.astype(np.float64),
        "vel_times": vel_times, "vel_data": vel_data, "trials": trials,
    }


def list_dandi688_subm_co_sessions(*, root: str | Path | None = None,
                                   sessions: Iterable[str] | None = None) -> list[dict]:
    """Inventory the sub-M CO sessions with per-session load metadata.

    Each row reports date, M1 unit count, bin count, trial counts and the
    mean per-unit firing rate.  Pass ``sessions`` to scan a subset.
    """
    subject_dir = dandi688_subm_root(root)
    surface = f"_ses-{TASK_SURFACE}-"
    rows = [row for row in list_dandi688_sessions(SUBJECT, root=subject_dir.parent)
            if row["subject"] == SUBJECT and surface in row["session"]]
    if sessions is not None:
        wanted = {str(session) for session in sessions}
        rows = [row for row in rows if row["session"] in wanted]
        missing = sorted(wanted - {row["session"] for row in rows})
        if missing:
            raise KeyError(f"sub-M CO session(s) not found under {subject_dir}: {missing}")
    result = []
    for row in rows:
        scan = _scan_session(Path(row["path"]))
        edges = scan["edges"]
        duration = float(edges[-1] - edges[0])
        n_units = len(scan["spikes"])
        total_spikes = sum(int(spike.size) for spike in scan["spikes"])
        result.append({
            "dataset": "dandi688", "subject": SUBJECT, "task": TASK_SURFACE,
            "session": str(row["session"]), "date": str(row["session"]).split(surface)[1],
            "path": str(Path(row["path"]).resolve()),
            "n_units": n_units, "n_bins": int(edges.size - 1), "bin_size_ms": BIN_SIZE_MS,
            "duration_s": round(duration, 3),
            "n_trials": len(scan["trials"]),
            "n_rewarded": sum(1 for trial in scan["trials"] if trial["result"] == "R"),
            "n_m1_electrodes": len(scan["canonical_keys"]),
            "mean_rate_hz": round(total_spikes / max(n_units * duration, 1e-9), 3),
        })
    return result


def load_dandi688_recording(session: str, *, root: str | Path | None = None) -> Dandi688Recording:
    """Load one ``sub-M_ses-CO-YYYYMMDD`` session as a pilot ``Recording``.

    ``neural`` is [T, N_s] non-negative 20 ms spike counts over the session's
    own M1 units, ``behavior`` is [T, 2] cursor velocity at bin centers, and
    ``trial_change`` marks the first bin of each trials-table interval.  See
    the module docstring for the derived ``eval_mask`` rule.
    """
    prefix = f"{SUBJECT}_ses-{TASK_SURFACE}-"
    if not session.startswith(prefix):
        raise ValueError(f"this adapter serves sub-M CO sessions only (RT tree is HDD), got {session!r}")
    subject_dir = dandi688_subm_root(root)
    path = subject_dir / f"{session}{NWB_SUFFIX}"
    if not path.is_file():
        raise KeyError(f"{session}: no NWB under {subject_dir}")
    scan = _scan_session(path)
    edges = scan["edges"]
    n_bins = int(edges.size - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    neural = np.stack([np.histogram(spike, bins=edges)[0] for spike in scan["spikes"]], axis=1).astype(np.float32)
    vel_times, vel_data = scan["vel_times"], scan["vel_data"]
    behavior = np.stack(
        [np.interp(centers, vel_times, vel_data[:, axis], left=0.0, right=0.0) for axis in range(vel_data.shape[1])],
        axis=1,
    ).astype(np.float32)
    trial_change = np.zeros(n_bins, dtype=bool)
    eval_mask = np.zeros(n_bins, dtype=bool)
    covered = (centers >= vel_times[0]) & (centers <= vel_times[-1])
    bounds: list[tuple[int, int]] = []
    for trial in scan["trials"]:
        start_bin = max(0, int(np.searchsorted(edges, trial["start_time"])))
        stop_bin = min(n_bins, int(np.searchsorted(edges, trial["stop_time"])))
        if stop_bin <= start_bin:
            continue
        trial_change[start_bin] = True
        bounds.append((start_bin, stop_bin))
        if trial["result"] == "R":
            eval_mask[start_bin:stop_bin] |= covered[start_bin:stop_bin]
    if neural.ndim != 2 or behavior.ndim != 2 or neural.shape[0] != behavior.shape[0]:
        raise ValueError(f"invalid array geometry neural={neural.shape}, behavior={behavior.shape}")
    if behavior.shape[1] != 2:
        raise ValueError(f"expected 2-dim cursor velocity, got {behavior.shape[1]}")
    if eval_mask.size != neural.shape[0] or not eval_mask.any():
        raise ValueError(f"{session}: eval mask empty or length mismatch")
    if not np.isfinite(neural).all() or not np.isfinite(behavior).all():
        raise ValueError(f"{session}: non-finite neural or behavior values")
    unit_map = scan["unit_to_electrode"]
    observed = np.zeros(CANONICAL_ELECTRODES, dtype=bool)
    observed[unit_map] = True
    return Dandi688Recording(
        task="dandi688", split="co", session=session, path=path.resolve(),
        neural=neural, behavior=behavior, trial_change=trial_change, eval_mask=eval_mask,
        trial_bounds=tuple(bounds),
        unit_to_electrode=unit_map, observed_mask=observed, bin_edges=edges,
    )


def dandi688_plan(sessions: Iterable[str] | None = None, targets: Iterable[str] | None = None, *,
                  include_final: bool = False, with_metadata: bool = False,
                  root: str | Path | None = None) -> dict:
    """Produce a deterministic sub-M CO source/target plan without overlap.

    The default reproduces the frozen SPINT 2015 protocol: the six train
    sessions form the source pool, the two dev sessions are the first
    targets, and the three final sessions stay withheld and available on
    request via ``include_final=True``.  Source and target sets must never
    overlap, and every requested session must exist on disk.
    """
    subject_dir = dandi688_subm_root(root)
    source = list(TRAIN_SESSIONS) if sessions is None else [str(s) for s in sessions]
    target = list(DEV_SESSIONS) if targets is None else [str(s) for s in targets]
    if not source or not target:
        raise ValueError("plan needs at least one source and one target session")
    overlap = sorted(set(source) & set(target))
    if overlap:
        raise ValueError(f"source and target sessions must not overlap: {overlap}")
    requested_final = sorted((set(source) | set(target)) & set(FINAL_SESSIONS))
    if requested_final and not include_final:
        raise PermissionError(f"final sessions are withheld: {requested_final}; pass include_final=True")
    available = {p.stem.removesuffix("_behavior+ecephys") for p in subject_dir.glob(f"{SUBJECT}_ses-{TASK_SURFACE}-*{NWB_SUFFIX}")}
    missing = sorted((set(source) | set(target)) - available)
    if missing:
        raise KeyError(f"session(s) not found under {subject_dir}: {missing}")
    plan = {
        "task": "dandi688", "subject": SUBJECT, "surface": TASK_SURFACE,
        "data_root": str(subject_dir), "root_env": ROOT_ENV,
        "bin_seconds": BIN_SECONDS, "canonical_electrodes": CANONICAL_ELECTRODES,
        "protocol_schema": SPINT_PROTOCOL_SCHEMA, "protocol_reference": SPINT_PROTOCOL_REFERENCE,
        "source_sessions": source, "target_sessions": target,
        "final_sessions": list(FINAL_SESSIONS),
        "final_sessions_withheld": not include_final, "final_accessed": False,
        "held_out_accessed": False,
    }
    if with_metadata:
        roster = sorted(set(source) | set(target) | (set(FINAL_SESSIONS) if include_final else set()))
        rows = list_dandi688_subm_co_sessions(root=root, sessions=roster)
        plan["sessions"] = {row["session"]: row for row in rows}
    return plan
