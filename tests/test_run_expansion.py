"""Driver plumbing of scripts/run_expansion.py: storage split (Drive vs scratch), selection, queue script (no GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_expansion as rx                          # noqa: E402


def args(**k):
    base = dict(bid="all", tier="primary", exclude_apst_dev=False)
    base.update(k)
    return SimpleNamespace(**base)


def test_store_push_pull_roundtrip(tmp_path):
    st = rx.Store(tmp_path / "drive", tmp_path / "scratch")
    assert st.split
    f = st.w("rows/x/a.jsonl")
    f.write_text("1\n")
    d = st.w("runs/b/seed0")
    d.mkdir(parents=True)
    (d / "done.json").write_text("{}")
    assert not st.exists("rows/x/a.jsonl")
    st.push("rows/x/a.jsonl")
    st.push("runs/b/seed0")
    assert st.exists("rows/x/a.jsonl") and st.exists("runs/b/seed0/done.json")
    # a fresh scratch (new Colab VM) pulls the persistent state back
    st2 = rx.Store(tmp_path / "drive", tmp_path / "scratch2")
    st2.pull("rows/x/a.jsonl")
    assert (tmp_path / "scratch2/rows/x/a.jsonl").read_text() == "1\n"
    # overwrite is atomic and leaves no .tmp file behind
    f.write_text("1\n2\n")
    st.push("rows/x/a.jsonl")
    assert (tmp_path / "drive/rows/x/a.jsonl").read_text() == "1\n2\n"
    assert not list((tmp_path / "drive").rglob("*.tmp"))


def test_store_without_scratch_is_noop(tmp_path):
    st = rx.Store(tmp_path)
    assert not st.split
    st.w("a/b.txt").write_text("x")
    st.push("a/b.txt")
    st.pull("a/b.txt")
    assert (tmp_path / "a/b.txt").read_text() == "x"


def test_select_backbones_filters_and_validates():
    allb = rx.select_backbones(args())
    assert [b.bid for b in allb].count("sc_0708") == 1
    assert len(allb) == 8
    one = rx.select_backbones(args(bid="sm_0611,sc_0713"))
    assert {b.bid for b in one} == {"sm_0611", "sc_0713"}
    with pytest.raises(SystemExit):
        rx.select_backbones(args(bid="nope"))
    nodev = rx.select_backbones(args(bid="sc_0708", exclude_apst_dev=True))[0]
    assert not any(t.apst_dev for t in nodev.targets) and len(nodev.targets) == 5
    full = rx.select_backbones(args(bid="sc_0708", tier="all"))[0]
    assert len(full.targets) == 17


def test_reused_backbone_points_to_existing_p1_artefacts(tmp_path):
    bb = rx.select_backbones(args(bid="sm_0307"))[0]
    st = rx.Store(tmp_path)
    assert bb.reuse and rx.ap_path(bb, st) == ROOT / "results/stage2_intervals_20261006/month_pool_k10_x8/best.pt"
    assert rx.p0_dir(bb, st, 1) == ROOT / "results/population_calib_20261015/runs/month_O/p0/seed1"
    new = rx.select_backbones(args(bid="sc_0713"))[0]
    assert rx.ap_path(new, st) == tmp_path / "backbones/sc_0713/final/best.pt"
    assert rx.p0_dir(new, st, 0) == tmp_path / "runs/sc_0713_O/p0/seed0"


def test_ap_command_uses_pool_in_order_and_k(tmp_path):
    bb = rx.select_backbones(args(bid="sc_0629"))[0]
    cmd = rx.ap_cmd(bb, tmp_path / "x", 2000, SimpleNamespace(device="cuda:0", ap_val_interval=100))
    assert cmd[cmd.index("--dandi688-sources") + 1].split(",") == list(bb.pool)
    assert cmd[cmd.index("--k") + 1] == "10" and cmd[cmd.index("--steps") + 1] == "2000"
    assert cmd[cmd.index("--target-session") + 1] == bb.targets[0].session


def test_queue_script_syntax_and_dry_run(tmp_path):
    script = ROOT / "scripts/run_expansion_queue.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    res = subprocess.run([str(script), "--dry-run", "--out", str(tmp_path / "o"), "--bids", "sm_0307,sm_0611"],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PY": sys.executable,
                                                              "PYTHONPATH": str(ROOT)}, timeout=100)
    assert res.returncode == 0, res.stderr[-500:]
    assert "sm_0307" in res.stdout and "sm_0611" in res.stdout and "sc_0708" not in res.stdout.split("backbones:")[1].split("|")[0]
