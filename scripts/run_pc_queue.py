#!/usr/bin/env python
"""Stage queue for the population-calibration ablation chain (GPU0 only).

Usage: run_pc_queue.py --stage refs|A|B|C1|C2|D|E [--workers 4]
Every job is idempotent (train skips finished runs, evaluate resumes rows),
so a stage can simply be re-launched after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/population_calib_20261015"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"
DRIVER = str(ROOT / "scripts/run_population_calib.py")
STRATA = ("days", "month", "year")
POOLS = ("O", "E")
SEEDS = (0, 1)
ENV = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0",
           PYTHONPATH=f"{ROOT}/.tools/mamba_deps:{ROOT}", TRITON_CACHE_DIR="/tmp/triton_pc_gpu0",
           OMP_NUM_THREADS="4")


def tag_of(variant, m=8, output="residual", label_free=False):
    return variant + (f"_m{m}" if variant in ("p3", "p4") else "") + \
        ("" if output == "residual" else "_abs") + ("_nolab" if label_free else "")


def job(stratum, pool, variant, seed, *, m=8, output="residual", label_free=False,
        evaluate=True, oracle=False, shuffle=False):
    tag = tag_of(variant, m, output, label_free)
    common = ["--stratum", stratum, "--pool", pool, "--variant", variant, "--seed", str(seed),
              "--m", str(m), "--output", output, "--tag", tag] + \
        (["--label-free"] if label_free else [])
    cmds = [[PY, DRIVER, "train", *common]]
    if evaluate:
        cmds.append([PY, DRIVER, "evaluate", *common])
    if shuffle:
        cmds.append([PY, DRIVER, "evaluate", *common, "--shuffle-labels"])
    if oracle:
        cmds.append([PY, DRIVER, "oracle", *common])
    return f"{stratum}_{pool}_{tag}_s{seed}", cmds


def run_job(item):
    name, cmds = item
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for cmd in cmds:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
            if rc != 0:
                print(f"[pc] FAILED {name}: {' '.join(cmd[2:5])} rc={rc}", flush=True)
                return name, rc
    print(f"[pc] ok {name}", flush=True)
    return name, 0


def run_all(items, workers):
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run_job, items))
    bad = [n for n, rc in results if rc]
    print(f"[pc] stage done: {len(results) - len(bad)} ok, {len(bad)} failed {bad}", flush=True)


def val_of(stratum, pool, tag, seed):
    p = OUT / "runs" / f"{stratum}_{pool}" / tag / f"seed{seed}" / "done.json"
    return json.loads(p.read_text())["best_val_r2"] if p.exists() else float("nan")


def select_m(variant, pool):
    """Pick m for p3/p4 on DAYS source validation only (mean over seeds)."""
    best, best_v = None, -1e9
    for m in (4, 8, 16):
        v = [val_of("days", pool, tag_of(variant, m), s) for s in SEEDS]
        v = [x for x in v if x == x]
        if v and sum(v) / len(v) > best_v:
            best, best_v = m, sum(v) / len(v)
    return best


def selection_table():
    rows = {}
    for variant in ("p0", "p1", "p2", "p3", "p4"):
        for pool in POOLS:
            for stratum in STRATA:
                m = select_m(variant, pool) if variant in ("p3", "p4") else 8
                if m is None:
                    continue
                v = [val_of(stratum, pool, tag_of(variant, m), s) for s in SEEDS]
                rows[f"{variant}|{pool}|{stratum}"] = {"m": m, "val": v}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    items = []
    if a.stage == "refs":
        for st in STRATA:
            items.append((f"ref_{st}", [[PY, DRIVER, "reference", "--stratum", st]]))
    elif a.stage == "A":      # P0 rows + residual/absolute ablation (source-val only)
        for st in STRATA:
            for pool in POOLS:
                for s in SEEDS:
                    items.append(job(st, pool, "p0", s))
        for pool in POOLS:
            for s in SEEDS:
                items.append(job("days", pool, "p0", s, output="absolute", evaluate=False))
    elif a.stage == "B":      # P1, P2
        for variant in ("p1", "p2"):
            for st in STRATA:
                for pool in POOLS:
                    for s in SEEDS:
                        items.append(job(st, pool, variant, s))
    elif a.stage == "C1":     # PMA seed sweep on days only
        for variant in ("p3", "p4"):
            for m in (4, 8, 16):
                for pool in POOLS:
                    for s in SEEDS:
                        items.append(job("days", pool, variant, s, m=m))
    elif a.stage == "C2":     # transfer the days-selected m to month/year
        for variant in ("p3", "p4"):
            for pool in POOLS:
                m = select_m(variant, pool)
                print(f"[pc] selected m for {variant}/{pool}: {m}", flush=True)
                for st in ("month", "year"):
                    for s in SEEDS:
                        items.append(job(st, pool, variant, s, m=m))
    elif a.stage == "select":
        print(json.dumps(selection_table(), indent=1))
        return
    elif a.stage.startswith("D:"):   # oracle for a given variant, e.g. D:p2 or D:p4
        variant = a.stage[2:]
        for st in STRATA:
            for pool in POOLS:
                m = select_m(variant, pool) if variant in ("p3", "p4") else 8
                for s in SEEDS:
                    name, cmds = job(st, pool, variant, s, m=m, evaluate=False, oracle=True)
                    items.append((name + "_oracle", cmds[1:]))
    elif a.stage.startswith("E:"):   # shuffled-label control for a variant
        variant = a.stage[2:]
        for st in STRATA:
            for pool in POOLS:
                m = select_m(variant, pool) if variant in ("p3", "p4") else 8
                for s in SEEDS:
                    name, cmds = job(st, pool, variant, s, m=m, evaluate=False, shuffle=True)
                    items.append((name + "_shufY", cmds[1:]))
    elif a.stage.startswith("N:"):   # label-free (trained) control for a variant
        variant = a.stage[2:]
        for st in STRATA:
            for pool in POOLS:
                m = select_m(variant, pool) if variant in ("p3", "p4") else 8
                for s in SEEDS:
                    items.append(job(st, pool, variant, s, m=m, label_free=True))
    else:
        sys.exit("unknown stage")
    run_all(items, a.workers)


if __name__ == "__main__":
    main()
