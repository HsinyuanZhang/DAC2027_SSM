"""Multi-target rotation plans and an SSD cache for FALCON recordings.

The audit found that every prior result used one target session.  This module
builds leave-one-run-out rotations over all held-in sessions and caches the
binned recordings on the SSD so that rotations do not re-read NWB from HDD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import DEFAULT_ROOT, Recording, _bounds, load_recording, sha256_file, visible_sessions

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "results" / "data_cache"


def rotation_plans(task: str, *, root: Path = DEFAULT_ROOT) -> list[dict]:
    """One plan per held-in session as target; the rest are sources."""
    held_ids = [r["session"] for r in visible_sessions(task, root=root)["held_in"]]
    if len(held_ids) < 3:
        raise ValueError(f"{task}: rotation needs >=3 held-in sessions, got {len(held_ids)}")
    plans = []
    for target in held_ids:
        plans.append({
            "task": task,
            "target_session": target,
            "source_sessions": [s for s in held_ids if s != target],
            "protocol": "leave_one_run_out",
        })
    return plans


def cached_recording(task: str, split: str, session: str, *, root: Path = DEFAULT_ROOT,
                     cache_root: Path = DEFAULT_CACHE) -> Recording:
    """Load a recording through an SSD npz cache keyed by the NWB sha256.

    The NWB stays on HDD.  A cache hit never touches it.  A cache entry whose
    stored sha256 disagrees with the current NWB is rebuilt.
    """
    cache_dir = Path(cache_root) / task / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{session}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        nwb = Path(str(blob["path"]))
        if nwb.exists() and sha256_file(nwb) == str(blob["sha256"]):
            return Recording(task, split, session, nwb,
                             blob["neural"], blob["behavior"], blob["trial_change"],
                             blob["eval_mask"],
                             tuple((int(a), int(b)) for a, b in blob["trial_bounds"]))
    record = load_recording(task, split, session, root=root)
    np.savez(path, neural=record.neural, behavior=record.behavior,
             trial_change=np.asarray(record.trial_change), eval_mask=record.eval_mask,
             trial_bounds=np.asarray(record.trial_bounds, dtype=np.int64),
             path=str(record.path), sha256=sha256_file(record.path))
    return record
