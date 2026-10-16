#!/usr/bin/env python
"""VM-side chain orchestrator (Colab).

usage: python colab_orch.py N [QUEUE.jsonl [DONE.txt]] [options]

Queue lines: {"name": ..., "cmds": ["<args of scripts/run_m2_calib.py>", "!<raw shell command>"]}.
Runs up to N chains concurrently (a chain = its commands run sequentially with &&), re-reads the queue
file every ``--poll`` s so it can be appended to, records finished chains in DONE.txt (chains must be
idempotent/resumable) and chains that exited non-zero in DONE.txt + ".failed" (name, rc, time; a failed
chain is NOT retried in this process).  Commands may contain single quotes (they are passed to
``bash -c`` as one argv element, no nested shell quoting).

Lifetime (never leave the VM running silently):
  * The orchestrator EXITS when every queue entry is done or failed and nothing is running, after a
    ``--grace`` s window in which an appended queue line is still picked up (default 60 s).
  * ``--idle-exit S``: same exit rule applied from the last time a chain was started/finished (default
    off; ``--grace`` already covers the common case).
  * ``--max-hours H``: stop LAUNCHING new chains after H hours (running chains finish, then exit).
  * ``--stop-file F`` (default /content/orch.stop): if the file exists no new chain is launched; the
    orchestrator exits when the running ones are done.  The host can create it with ``colab exec``.
  * ``--hook-cmd CMD --hook-interval S``: run CMD every S s (e.g. a script that reads a host-written
    balance file); a non-zero exit stops launching like the stop file (CU-check hook, see runbook).
  * ``--on-exit CMD``: run once on exit (e.g. the runtime-unassign call, see docs/colab_runbook.md);
    ``/content/orch.finished`` is always written on exit so that the host can poll and call
    ``colab stop``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

REPO = os.environ.get("ORCH_REPO", "/home/xinyuan/Work_host/SSM")
ENV = f"PYTHONNOUSERSITE=1 PYTHONPATH={REPO}/.tools/mamba_deps:{REPO}"


def read_done(path):
    return set(l.strip() for l in open(path) if l.strip()) if os.path.exists(path) else set()


def load_queue(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def build_script(chain, done_file, repo=REPO, env=ENV):
    """One bash script: commands joined by &&, then CHAIN_OK and the DONE marker (DONE file from argv)."""
    parts = [f"{env} python scripts/run_m2_calib.py {x}" if not x.startswith("!") else x[1:]
             for x in chain["cmds"]]
    parts += [f"echo CHAIN_OK {shlex.quote(chain['name'])}",
              f"echo {shlex.quote(chain['name'])} >> {shlex.quote(done_file)}"]
    return " && ".join(parts)


class Orchestrator:
    def __init__(self, n, queue, done, *, repo=REPO, log_dir="/content/logs", grace=60.0, idle_exit=0.0,
                 max_hours=0.0, stop_file="/content/orch.stop", hook_cmd=None, hook_interval=600.0,
                 on_exit=None, finished_file="/content/orch.finished", poll=10.0, env=None, clock=time.time):
        self.n, self.queue, self.done_file = n, queue, done
        self.failed_file = done + ".failed"
        self.repo, self.log_dir, self.grace, self.idle_exit = repo, log_dir, grace, idle_exit
        self.max_hours, self.stop_file, self.hook_cmd, self.hook_interval = max_hours, stop_file, hook_cmd, hook_interval
        self.on_exit, self.finished_file, self.poll = on_exit, finished_file, poll
        self.env = env if env is not None else f"PYTHONNOUSERSITE=1 PYTHONPATH={repo}/.tools/mamba_deps:{repo}"
        self.clock = clock
        self.started, self.running, self.failed = set(), {}, {}
        self.t_start = self.t_activity = self.t_hook = clock()
        self.hook_stop = False
        os.makedirs(log_dir, exist_ok=True)

    def _reap(self):
        for name, p in list(self.running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del self.running[name]
            self.t_activity = self.clock()
            if rc != 0 and name not in read_done(self.done_file):
                self.failed[name] = rc
                with open(self.failed_file, "a") as f:
                    f.write(f"{name}\trc={rc}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"[orch] chain {name} FAILED rc={rc} (log {self.log_dir}/{name}.log)", flush=True)

    def _may_launch(self):
        if os.path.exists(self.stop_file) or self.hook_stop:
            return False
        if self.max_hours and (self.clock() - self.t_start) > self.max_hours * 3600:
            return False
        return True

    def _run_hook(self):
        if not self.hook_cmd or self.clock() - self.t_hook < self.hook_interval:
            return
        self.t_hook = self.clock()
        rc = subprocess.call(self.hook_cmd, shell=True)
        if rc != 0:
            self.hook_stop = True
            print(f"[orch] hook returned {rc}: no further chains will be launched", flush=True)

    def step(self):
        """One scheduling pass.  Returns True when the orchestrator should exit."""
        self._reap()
        self._run_hook()
        done = read_done(self.done_file)
        q = load_queue(self.queue)
        launching = self._may_launch()
        for c in q:
            if not launching or len(self.running) >= self.n:
                break
            name = c["name"]
            if name in self.started or name in done:
                continue
            script = build_script(c, self.done_file, self.repo, self.env)
            log = open(os.path.join(self.log_dir, f"{name}.log"), "a")
            self.running[name] = subprocess.Popen(["bash", "-c", script], cwd=self.repo, stdout=log,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
            self.started.add(name)
            self.t_activity = self.clock()
        if self.running:
            return False
        pending = [c for c in q if c["name"] not in self.started and c["name"] not in done]
        if pending and launching:
            return False
        idle = self.clock() - self.t_activity
        if not launching:
            return True                      # stop requested / time cap: nothing running -> exit now
        return idle >= max(self.grace, self.idle_exit)

    def run(self):
        try:
            while not self.step():
                time.sleep(self.poll)
        finally:
            self.finish()

    def finish(self):
        done = read_done(self.done_file)
        msg = {"done": sorted(done), "failed": self.failed, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.finished_file, "w") as f:
            json.dump(msg, f)
        print(f"[orch] exit: {len(done)} chains done, {len(self.failed)} failed", flush=True)
        if self.on_exit:
            subprocess.call(self.on_exit, shell=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=4)
    ap.add_argument("queue", nargs="?", default="/content/queue.jsonl")
    ap.add_argument("done", nargs="?", default="/content/queue_done.txt")
    ap.add_argument("--grace", type=float, default=60.0)
    ap.add_argument("--idle-exit", type=float, default=0.0)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--stop-file", default="/content/orch.stop")
    ap.add_argument("--hook-cmd")
    ap.add_argument("--hook-interval", type=float, default=600.0)
    ap.add_argument("--on-exit")
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args(argv)
    Orchestrator(a.n, a.queue, a.done, grace=a.grace, idle_exit=a.idle_exit, max_hours=a.max_hours,
                 stop_file=a.stop_file, hook_cmd=a.hook_cmd, hook_interval=a.hook_interval,
                 on_exit=a.on_exit, poll=a.poll).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
