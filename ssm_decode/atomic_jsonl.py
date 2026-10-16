"""Crash-safe jsonl helpers for resumable queues (Colab sessions can die at any moment).

* :func:`append_jsonl` never appends in place.  It copies the existing valid lines plus the new rows to a
  temporary file in the same directory and renames it over the target (``os.replace`` is atomic on POSIX), so a
  reader sees either the old or the new file, never a torn one.
* :func:`read_jsonl` tolerates a truncated LAST line (a writer killed in the middle of an in-place append, e.g.
  by an older version of a driver) and drops it; a corrupt line anywhere else is a real error.
* :func:`repair_jsonl` rewrites the file without the truncated last line.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _split(path: Path) -> tuple[list[str], bool]:
    """Valid lines of ``path`` and whether a truncated last line was dropped."""
    if not path.exists():
        return [], False
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    dropped = False
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}: corrupt jsonl line {i + 1} (not the last line)")
            lines = lines[:i]
            dropped = True
            break
    return lines, dropped


def read_jsonl(path) -> list[dict]:
    lines, _ = _split(Path(path))
    return [json.loads(line) for line in lines]


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write("".join(line + "\n" for line in lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def repair_jsonl(path) -> bool:
    """Drop a truncated last line in place (atomically).  Returns True when something was repaired."""
    path = Path(path)
    lines, dropped = _split(path)
    if dropped:
        _write_atomic(path, lines)
    return dropped


def append_jsonl(path, rows) -> None:
    """Atomically append ``rows`` (dicts) to ``path`` (existing valid lines are kept)."""
    path = Path(path)
    lines, _ = _split(path)
    lines = lines + [json.dumps(r) for r in rows]
    _write_atomic(path, lines)
