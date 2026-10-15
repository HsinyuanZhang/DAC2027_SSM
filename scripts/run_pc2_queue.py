#!/usr/bin/env python
"""Stage queue for the P2 iteration (A'1 identity-free regime), GPU0 only.

Usage: run_pc2_queue.py --stage noalign [--workers 3] [--modes globalmean zero]
Idempotent: finished training runs are skipped, evaluation rows resume.
Outputs go to results/population_calib_p2_20261015/ (PC_OUT).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/population_calib_p2_20261015"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"
DRIVER = str(ROOT / "scripts/run_population_calib.py")
SUFFIX = {"globalmean": "_na_gm", "zero": "_na_zero"}
ENV = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0", PC_OUT=str(OUT),
           PYTHONPATH=f"{ROOT}/.tools/mamba_deps:{ROOT}", TRITON_CACHE_DIR="/tmp/triton_pc2_gpu0",
           OMP_NUM_THREADS="4")


def job(stratum, variant, seed, mode, m=8):
    tag = variant + (f"_m{m}" if variant in ("p3", "p4") else "") + SUFFIX[mode]
    common = ["--stratum", stratum, "--pool", "O", "--variant", variant, "--seed", str(seed),
              "--m", str(m), "--tag", tag, "--prior-mode", mode]
    return f"{stratum}_O_{tag}_s{seed}", [[PY, DRIVER, "train", *common],
                                           [PY, DRIVER, "evaluate", *common]]


def qat_job(stratum, variant, seed, m=8):
    """Ternary-QAT fine-tune of a P1 checkpoint on SOURCE sessions only (STE)."""
    base = variant + (f"_m{m}" if variant in ("p3", "p4") else "")
    tag = base + "_qat_tern"
    init = ROOT / "results/population_calib_20261015/runs" / f"{stratum}_O" / base / f"seed{seed}" / "best.pt"
    common = ["--stratum", stratum, "--pool", "O", "--variant", variant, "--seed", str(seed),
              "--m", str(m), "--tag", tag]
    train = [PY, DRIVER, "train", *common, "--init-from", str(init), "--qat", "tern",
             "--lr", "3e-4", "--steps", "600", "--min-steps", "300", "--patience", "4"]
    return f"{stratum}_O_{tag}_s{seed}", [train]


def run_job(item):
    name, cmds = item
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for cmd in cmds:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
            if rc != 0:
                print(f"[pc2] FAILED {name}: {' '.join(cmd[2:4])} rc={rc}", flush=True)
                return name, rc
    print(f"[pc2] ok {name}", flush=True)
    return name, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--modes", nargs="*", default=["globalmean", "zero"])
    p.add_argument("--variants", nargs="*", default=["p0", "p2", "p3", "p4"])
    a = p.parse_args()
    items = []
    if a.stage == "noalign":
        for mode in a.modes:
            for variant in a.variants:
                for st in ("days", "month", "year"):
                    for s in (0, 1):
                        items.append(job(st, variant, s, mode))
        for mode in a.modes:      # P-prior rows with the same identity-free prior
            for st in ("days", "month", "year"):
                items.append((f"ref_{st}{SUFFIX[mode]}",
                              [[PY, DRIVER, "reference", "--stratum", st, "--prior-only",
                                "--prior-mode", mode]]))
    elif a.stage == "qat":
        for variant in ("p0", "p4"):
            for st in ("days", "month", "year"):
                for s in (0, 1):
                    items.append(qat_job(st, variant, s))
    else:
        raise SystemExit("unknown stage")
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_job, items))
    bad = [n for n, rc in res if rc]
    print(f"[pc2] stage done: {len(res) - len(bad)} ok, {len(bad)} failed {bad}", flush=True)


if __name__ == "__main__":
    main()
