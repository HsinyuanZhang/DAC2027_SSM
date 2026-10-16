"""Crash-safe jsonl appends: atomic replace, torn-last-line recovery (no torch, no NWB)."""
from __future__ import annotations

import json
import os

import pytest

from ssm_decode import atomic_jsonl as aj


def test_append_creates_and_extends(tmp_path):
    p = tmp_path / "d" / "rows.jsonl"
    aj.append_jsonl(p, [{"a": 1}])
    aj.append_jsonl(p, [{"a": 2}, {"a": 3}])
    assert [r["a"] for r in aj.read_jsonl(p)] == [1, 2, 3]
    assert not list(tmp_path.rglob("*.tmp"))


def test_append_is_atomic_when_the_replace_never_happens(tmp_path, monkeypatch):
    """A crash between 'write temp' and 'replace' leaves the previous file untouched (never a partial row)."""
    p = tmp_path / "rows.jsonl"
    aj.append_jsonl(p, [{"a": 1}])
    before = p.read_bytes()

    def boom(src, dst):
        raise KeyboardInterrupt("killed before os.replace")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        aj.append_jsonl(p, [{"a": 2}])
    monkeypatch.undo()
    assert p.read_bytes() == before
    assert [r["a"] for r in aj.read_jsonl(p)] == [1]
    aj.append_jsonl(p, [{"a": 2}])                  # the stale .tmp is simply overwritten on the next try
    assert [r["a"] for r in aj.read_jsonl(p)] == [1, 2]


def test_torn_last_line_is_tolerated_and_repaired(tmp_path):
    """A file written in place by an older driver and killed mid-line: resume must not crash or double count."""
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + '{"a": 3, "b": "tru')
    assert [r["a"] for r in aj.read_jsonl(p)] == [1, 2]          # read tolerates it
    assert aj.repair_jsonl(p) is True
    assert p.read_text().endswith("\n") and len(p.read_text().splitlines()) == 2
    assert aj.repair_jsonl(p) is False                           # idempotent
    p.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2')
    aj.append_jsonl(p, [{"a": 9}])                               # append also drops the torn tail
    assert [r["a"] for r in aj.read_jsonl(p)] == [1, 9]


def test_corrupt_middle_line_is_an_error(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\nnot json\n" + json.dumps({"a": 3}) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        aj.read_jsonl(p)


def test_missing_and_empty_files(tmp_path):
    assert aj.read_jsonl(tmp_path / "nope.jsonl") == []
    assert aj.repair_jsonl(tmp_path / "nope.jsonl") is False
    (tmp_path / "e.jsonl").write_text("")
    assert aj.read_jsonl(tmp_path / "e.jsonl") == []
