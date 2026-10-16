#!/usr/bin/env python
"""VM-side chain orchestrator (Colab).  usage: python colab_orch.py N [QUEUE.jsonl [DONE.txt]]
Queue lines: {"name": ..., "cmds": ["<args of scripts/run_m2_calib.py>", "!<raw shell command>"]}.
Runs up to N chains concurrently (each chain = its commands run sequentially with &&), re-reads the queue file
every 10 s so it can be appended to, and records finished chains in DONE.txt (chains are idempotent/resumable).
Commands must not contain single quotes."""
import json, os, subprocess, sys, time
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Q = sys.argv[2] if len(sys.argv) > 2 else "/content/queue.jsonl"
D = sys.argv[3] if len(sys.argv) > 3 else "/content/queue_done.txt"
ENV = "PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM"
os.makedirs("/content/logs", exist_ok=True)
started, running = set(), {}
def finished():
    return set(l.strip() for l in open(D)) if os.path.exists(D) else set()
while True:
    done = finished()
    for n, p in list(running.items()):
        if p.poll() is not None:
            del running[n]
    q = [json.loads(l) for l in open(Q) if l.strip()] if os.path.exists(Q) else []
    for c in q:
        if len(running) >= N: break
        if c["name"] in started or c["name"] in done: continue
        script = " && ".join(f"{ENV} python scripts/run_m2_calib.py {x}" if not x.startswith("!") else x[1:] for x in c["cmds"])
        script += f" && echo CHAIN_OK {c['name']} && echo {c['name']} >> /content/queue_done.txt"
        p = subprocess.Popen(f"cd /home/xinyuan/Work_host/SSM && bash -c '{script}' >> /content/logs/{c['name']}.log 2>&1", shell=True)
        running[c["name"]] = p; started.add(c["name"])
    if not running and all(c["name"] in started or c["name"] in done for c in q) and os.path.exists("/content/orch.stop"):
        break
    time.sleep(10)
