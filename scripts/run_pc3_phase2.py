#!/usr/bin/env python
"""P3 phase 2 scheduler (GPU0, <=3 concurrent run_pc3_refine jobs).

Year shares the trunk, the six sources, the P0 calibrators (identical weights, verified) and the
OOS sessions with month, so its source-simulation files are exact copies of month's with the
stratum label rewritten (no recomputation).  Order: R3 source simulation, then target eval.
"""
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/home/xinyuan/Work_host/SSM")
P3 = ROOT / "results/population_calib_p3_20261016"
SEL = P3 / "select"
LOG = P3 / "logs"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"
ENV = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0", PC_OUT=str(P3),
           PYTHONPATH=f"{ROOT}/.tools/mamba_deps:{ROOT}", TRITON_CACHE_DIR="/tmp/triton_pc3_gpu0")


def nrun():
    out = subprocess.run(["pgrep", "-fc", "python scripts/run_pc3_refine.py"],
                         capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def copy_year(stage, sset):
    src = SEL / f"{stage}_month_{sset}.jsonl"
    dst = SEL / f"{stage}_year_{sset}.jsonl"
    dst.write_text(src.read_text().replace('"stratum": "month"', '"stratum": "year"'))


def launch(cmd, name):
    f = (LOG / f"{name}.log").open("a")
    return subprocess.Popen([PY, "scripts/run_pc3_refine.py", *cmd], stdout=f, stderr=subprocess.STDOUT,
                            env=ENV, cwd=ROOT)


def wait_file(p):
    while not Path(p).exists():
        time.sleep(15)


def wait_slot():
    while nrun() >= 3:
        time.sleep(15)


LOCK = threading.Lock()


def run_job(cmd, name):
    """Launch when a slot is free (<=3 jobs in total), block until the job ends."""
    with LOCK:
        wait_slot()
        proc = launch(cmd, name)
        time.sleep(25)                       # let the process appear in pgrep before the next launch
    proc.wait()
    if proc.returncode != 0:
        print(f"FAILED {name} rc={proc.returncode}", flush=True)


def chain(sset, order):
    """R3 simulation for month/days, copy year from month, then target eval of all strata."""
    threads = []

    def r3(stratum):
        wait_file(SEL / f"selection_{stratum}_{sset}.json")
        run_job(["select-r3", "--stratum", stratum, "--set", sset], f"select-r3_{stratum}_{sset}")
    for stratum in order:
        t = threading.Thread(target=r3, args=(stratum,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for stage in ("r12", "r3"):
        copy_year(stage, sset)
    shutil.copy(SEL / f"selection_month_{sset}.json", SEL / f"selection_year_{sset}.json")
    threads = []
    for stratum in ("days", "month", "year"):
        t = threading.Thread(target=run_job, args=(["eval", "--stratum", stratum, "--set", sset],
                                                    f"eval_{stratum}_{sset}"))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    chains = [threading.Thread(target=chain, args=("oos", ["month", "days"])),
              threading.Thread(target=chain, args=("is", ["month", "days"]))]
    for c in chains:
        c.start()
        time.sleep(30)
    for c in chains:
        c.join()
    print("phase2 done", flush=True)
