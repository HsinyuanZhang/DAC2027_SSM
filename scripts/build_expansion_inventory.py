#!/usr/bin/env python
"""Scan the sub-M / sub-C CO sessions once and store a small per-session inventory (CPU only).

Output (tracked in git, ~100 KB): ``results/analysis/expansion_20261016/session_inventory.json`` with, per session,
unit count, bin/trial counts, usable-trial count, electrode map, file size.  The expansion protocol
(``ssm_decode/expansion_protocol.py``) is built from this file, so the protocol, the pool tests and the
Colab bundle list never need the NWB files.

Never opens a forbidden session: sub-M final (20150623/25/26) and the six APST-held-out sub-C sessions
are skipped here and rejected by the loader.

  python scripts/build_expansion_inventory.py [--workers 8] [--out results/analysis/expansion_20261016/session_inventory.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import dandi688_data as dd                             # noqa: E402
from ssm_decode.track_a_experiment import usable_trials               # noqa: E402


def candidate_sessions() -> list[tuple[str, str]]:
    """(session id, path) of every allowed CO session of both subjects (forbidden ones excluded)."""
    out = []
    for subject in dd.SUBJECTS:
        folder = dd.dandi688_subject_root(subject)
        for p in sorted(folder.glob(f"{subject}_ses-{dd.TASK_SURFACE}-*{dd.NWB_SUFFIX}")):
            sid = p.name[: -len(dd.NWB_SUFFIX)]
            if sid in dd.FORBIDDEN_SESSIONS or sid in dd.FINAL_SESSIONS:
                continue
            out.append((sid, str(p)))
    return out


def scan_one(item: tuple[str, str]) -> dict:
    sid, path = item
    rec = dd.load_dandi688_recording(sid)
    usable = usable_trials(rec)
    return {
        "session": sid, "subject": sid.split("_")[0], "date": sid.rsplit("-", 1)[1],
        "n_units": int(rec.neural.shape[1]), "n_bins": int(rec.neural.shape[0]),
        "n_trials": len(rec.trial_bounds), "n_usable_trials": len(usable),
        "unit_to_electrode": [int(x) for x in rec.unit_to_electrode],
        "size_bytes": os.path.getsize(path), "file": Path(path).name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results/analysis/expansion_20261016/session_inventory.json"))
    args = ap.parse_args()
    items = candidate_sessions()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(scan_one, items))
    rows.sort(key=lambda r: (r["subject"], r["date"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=0))
    print(f"wrote {len(rows)} sessions -> {args.out}")


if __name__ == "__main__":
    main()
