"""Tests for scripts/colab_orch.py (VM orchestrator) and scripts/colab_exec_retry.sh (host retry wrapper)."""
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_orch", ROOT / "scripts" / "colab_orch.py")
orch = importlib.util.module_from_spec(spec)
sys.modules["colab_orch"] = orch
spec.loader.exec_module(orch)


def make(tmp_path, chains, **kw):
    q = tmp_path / "q.jsonl"
    q.write_text("".join(json.dumps(c) + "\n" for c in chains))
    d = tmp_path / "custom_done.txt"                     # NOT /content/queue_done.txt
    o = orch.Orchestrator(2, str(q), str(d), repo=str(tmp_path), log_dir=str(tmp_path / "logs"), grace=0.0,
                          stop_file=str(tmp_path / "stop"), finished_file=str(tmp_path / "finished.json"),
                          poll=0.05, env="", **kw)
    return o, q, d


def test_done_file_comes_from_argv_and_single_quotes_are_allowed(tmp_path):
    o, q, d = make(tmp_path, [{"name": "a", "cmds": ["!echo 'it works' > out_a.txt"]},
                              {"name": "b", "cmds": ["!echo hi", "!test -f out_a.txt || sleep 0.3"]}])
    o.run()
    assert set(d.read_text().split()) == {"a", "b"}
    assert (tmp_path / "out_a.txt").read_text().strip() == "it works"
    fin = json.loads((tmp_path / "finished.json").read_text())
    assert fin["failed"] == {} and fin["done"] == ["a", "b"]


def test_done_chains_are_skipped_and_exit_when_nothing_left(tmp_path):
    o, q, d = make(tmp_path, [{"name": "a", "cmds": ["!touch ran_a"]}])
    d.write_text("a\n")
    o.run()                                              # returns immediately: all chains done
    assert not (tmp_path / "ran_a").exists()


def test_failed_chain_is_logged_not_retried_and_does_not_block_exit(tmp_path):
    o, q, d = make(tmp_path, [{"name": "bad", "cmds": ["!echo x >> tries.txt", "!exit 7"]},
                              {"name": "good", "cmds": ["!true"]}])
    o.run()
    assert d.read_text().split() == ["good"]
    failed = (tmp_path / "custom_done.txt.failed").read_text()
    assert failed.startswith("bad\trc=7")
    assert (tmp_path / "tries.txt").read_text().count("x") == 1          # never re-run
    assert json.loads((tmp_path / "finished.json").read_text())["failed"] == {"bad": 7}


def test_stop_file_and_hook_stop_launching(tmp_path):
    o, q, d = make(tmp_path, [{"name": "a", "cmds": ["!touch ran_a"]}])
    (tmp_path / "stop").write_text("")
    o.run()
    assert not (tmp_path / "ran_a").exists()
    (tmp_path / "stop").unlink()
    o2, q2, d2 = make(tmp_path, [{"name": "h", "cmds": ["!touch ran_h"]}], hook_cmd="exit 1", hook_interval=0.0)
    o2.t_hook = -1e9
    o2.run()
    assert o2.hook_stop and not (tmp_path / "ran_h").exists()


def test_on_exit_command_runs_once_and_marker_is_written(tmp_path):
    o, q, d = make(tmp_path, [{"name": "a", "cmds": ["!true"]}], on_exit=f"echo bye >> {tmp_path}/exit.log")
    o.run()
    assert (tmp_path / "exit.log").read_text() == "bye\n" and (tmp_path / "finished.json").exists()


def test_max_hours_stops_launching(tmp_path):
    t = [0.0]
    o, q, d = make(tmp_path, [{"name": "a", "cmds": ["!touch ran_a"]}], max_hours=1.0)
    o.clock = lambda: t[0]
    o.t_start = o.t_activity = o.t_hook = 0.0
    t[0] = 3601.0
    assert o.step() is True and not (tmp_path / "ran_a").exists()


# ---------------------------------------------------------------- retry wrapper ----

def fake_colab(tmp_path, behaviour):
    """A stand-in `colab` that counts calls; behaviour(call_no) -> (rc, text)."""
    script = tmp_path / "fake_colab.sh"
    lines = ["#!/bin/bash", f'n=$(cat {tmp_path}/calls 2>/dev/null || echo 0); n=$((n+1)); echo $n > {tmp_path}/calls']
    for no, (rc, text) in behaviour.items():
        lines.append(f'if [ "$n" = "{no}" ]; then echo "{text}"; exit {rc}; fi')
    lines.append('echo ok; exit 0')
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def run_retry(tmp_path, fake, body="print(1)\n"):
    f = tmp_path / "x.py"
    f.write_text(body)
    env = dict(os.environ, COLAB_EXEC_CMD=str(fake), COLAB_EXEC_SLEEP="0", COLAB_EXEC_TRIES="6")
    p = subprocess.run(["bash", str(ROOT / "scripts/colab_exec_retry.sh"), "sess", str(f), "5"],
                       capture_output=True, text=True, env=env, timeout=60)
    calls = int((tmp_path / "calls").read_text()) if (tmp_path / "calls").exists() else 0
    return p, calls


def test_retry_does_not_rerun_on_rc0_even_if_output_reports_a_lost_connection(tmp_path):
    p, calls = run_retry(tmp_path, fake_colab(tmp_path, {1: (0, "RuntimeError: Connection was lost")}))
    assert calls == 1 and p.returncode == 3 and "NOT re-running" in p.stderr


def test_retry_clean_success_runs_once(tmp_path):
    p, calls = run_retry(tmp_path, fake_colab(tmp_path, {}))
    assert calls == 1 and p.returncode == 0 and "ok" in p.stdout


def test_retry_on_nonzero_rc_until_success(tmp_path):
    p, calls = run_retry(tmp_path, fake_colab(tmp_path, {1: (1, "boom"), 2: (1, "TimeoutError")}))
    assert calls == 3 and p.returncode == 0


def test_nonidempotent_launch_file_gets_a_single_attempt(tmp_path):
    p, calls = run_retry(tmp_path, fake_colab(tmp_path, {1: (1, "boom")}),
                         body="# NONIDEMPOTENT\nimport subprocess\n")
    assert calls == 1 and p.returncode == 1


def test_retry_gives_up_after_the_configured_tries(tmp_path):
    p, calls = run_retry(tmp_path, fake_colab(tmp_path, {i: (1, "boom") for i in range(1, 10)}))
    assert calls == 6 and p.returncode == 1
