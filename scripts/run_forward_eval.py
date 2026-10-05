#!/usr/bin/env python
"""Drive the multi-target time-forward protocol (revision 4.1) for one task.

Phases per (task, capacity):
1. Pretrain one source pool per chronological pool key.  The budget-extension
   rule retrains with double steps when best_step lands in the last 20
   percent, at most twice.
2. Fit adaptation methods per (target, method, seed, lr) with deferred query.
   LR selection per (target, method) uses the mean prefix-validation R2 over
   seeds; the query runs only for the selected lr.
3. Aggregate: multi-target mean, per-target paired difference to a reference
   method, per-target seed SD.

Usage:
  python scripts/run_forward_eval.py --task m2 --capacity wide \
      --methods none,lora --root results/forward_eval_m2_wide_20261005 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode.forward_eval import CAPACITIES, SUPPORT_TRIALS, aggregate, forward_plans  # noqa: E402


def capacity_of(args):
    """Named capacity or an explicit width,layers,state override."""
    if args.custom_capacity:
        width, layers, state = (int(x) for x in args.custom_capacity.split(","))
        return {"width": width, "layers": layers, "state_size": state}
    return CAPACITIES[args.capacity]

PY = str(Path(sys.executable).resolve())


def child_env(gpu: int, tag: str) -> dict:
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1",
               PYTHONPATH=f"{ROOT}/.tools/mamba_deps:{ROOT}",
               TRITON_CACHE_DIR=f"/tmp/triton_fe_{tag}",
               CUDA_VISIBLE_DEVICES=str(gpu))
    return env


def run_quiet(cmd, env, log_path):
    with open(log_path, "w") as stream:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    return result.returncode == 0


def _capacity_of(args):
    if args.width:
        return {"width": args.width, "layers": args.layers,
                "state_size": args.state_size, "ffn_mult": args.ffn_mult if args.ffn_mult else 2}
    if args.capacity not in CAPACITIES:
        raise SystemExit(f"unknown capacity {args.capacity!r}; pass --width/--layers/--state-size")
    return CAPACITIES[args.capacity]


def pretrain_pool(args, pool_key, sources, pool_dir):
    capacity = _capacity_of(args)
    pool_dir.parent.mkdir(parents=True, exist_ok=True)
    steps = args.pretrain_steps
    log = pool_dir.parent / f"{pool_key}.train.log"
    for attempt in range(3):
        final_dir = pool_dir if attempt == 0 else pool_dir.parent / f"{pool_dir.name}_x{2 ** attempt}"
        if not (final_dir / "manifest.json").exists():
            if final_dir.exists():
                subprocess.run(["rm", "-rf", str(final_dir)], check=True)
            cmd = [PY, "-m", "ssm_decode.session_pretraining",
                   "--task", args.task,
                   "--source-sessions", ",".join(sources),
                   "--output", str(final_dir),
                   "--device", "cuda:0",
                   "--width", str(capacity["width"]),
                   "--layers", str(capacity["layers"]),
                   "--state-size", str(capacity["state_size"]),
                   "--ffn-mult", str(capacity.get("ffn_mult") or 2),
                   "--steps", str(steps)] + list(args.pretrain_extra or [])
            print(f"[pool] {pool_key}: pretrain {len(sources)} sources, steps {steps}", flush=True)
            if not run_quiet(cmd, child_env(args.gpu, f"pool_{pool_key}"), log):
                raise RuntimeError(f"pool pretraining failed: {pool_key} (see {log})")
        blob = json.loads((final_dir / "manifest.json").read_text())
        if blob.get("status") != "completed":
            raise RuntimeError(f"pool {pool_key}: training status {blob.get('status')!r}")
        best_step = int(blob.get("best_step", 0))
        if best_step > 0.8 * steps and attempt < 2:
            print(f"[pool] {pool_key}: best_step {best_step} in last 20% of {steps}; "
                  f"retraining at {steps * 2}", flush=True)
            steps *= 2
            continue
        print(f"[pool] {pool_key} ready (best_step {best_step}, budget {steps})", flush=True)
        return final_dir / "best.pt"
    raise RuntimeError(f"pool {pool_key}: extension budget exhausted")


def fit_target(args, checkpoint, target, method, seed, lr, fit_dir):
    if (fit_dir / "fit_result.json").exists():
        return fit_dir
    fit_dir.parent.mkdir(parents=True, exist_ok=True)
    if fit_dir.exists():
        subprocess.run(["rm", "-rf", str(fit_dir)], check=True)
    support = SUPPORT_TRIALS[args.task]
    cmd = [PY, "-m", "ssm_decode.cross_session_iteration",
           "--task", args.task,
           "--pretrained", str(checkpoint),
           "--target-session", target,
           "--method", method,
           "--seed", str(seed),
           "--lr", repr(lr),
           "--steps", str(args.adapt_steps),
           "--support-trials", str(support),
           "--query-cutoff-trials", str(support),
           "--normalization", "support",
           "--policy", "recording_causal_fixed_window",
           "--defer-query",
           "--output", str(fit_dir),
           "--device", "cuda:0"]
    tag = fit_dir.relative_to(fit_dir.parents[3])
    if not run_quiet(cmd, child_env(args.gpu, re.sub(r"[^a-zA-Z0-9]", "_", str(tag))),
                     fit_dir.parent / f"{fit_dir.name}.log"):
        raise RuntimeError(f"fit failed: {fit_dir} (see {fit_dir.name}.log)")
    return fit_dir


def evaluate_fit(args, fit_dir):
    metrics = fit_dir / "query_evaluation" / "metrics.json"
    if metrics.exists():
        return
    cmd = [PY, "-m", "ssm_decode.cross_session_iteration",
           "--evaluate-checkpoint", str(fit_dir / "best.pt"),
           "--output", str(fit_dir / "query_evaluation"),
           "--device", "cuda:0"]
    tag = re.sub(r"[^a-zA-Z0-9]", "_", str(fit_dir.relative_to(fit_dir.parents[3])))
    if not run_quiet(cmd, child_env(args.gpu, f"eval_{tag}"), fit_dir.parent / f"{fit_dir.name}.eval.log"):
        raise RuntimeError(f"query evaluation failed: {fit_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["m1", "m2"])
    parser.add_argument("--capacity", required=True,
                        help="named capacity or label for a --width/--layers/--state-size override")
    parser.add_argument("--methods", default="none,lora")
    parser.add_argument("--mode", default="forward", choices=["forward", "leave_one_out"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--none-seeds", default="0")
    parser.add_argument("--lrs", default="0.0003,0.001", help="lr candidates for trained methods")
    parser.add_argument("--adapt-steps", type=int, default=1000)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--pretrain-extra", action="append", default=None,
                        help="repeatable; each entry is one flag or value passed to session_pretraining")
    for name in ("width", "layers", "state-size", "ffn-mult"):
        parser.add_argument("--" + name, type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    if not args.aggregate_only:
        root.mkdir(parents=True, exist_ok=True)
        plans = forward_plans(args.task, mode=args.mode)
        (root / "plan.json").write_text(json.dumps(plans, indent=2) + "\n")
        seeds = [int(s) for s in args.seeds.split(",")]
        none_seeds = [int(s) for s in args.none_seeds.split(",")]
        lrs = [float(x) for x in args.lrs.split(",")]
        for pool_key, pool in plans["pools"].items():
            checkpoint = pretrain_pool(args, pool_key, pool["sources"], root / "pools" / pool_key)
            for target in pool["targets"]:
                for method in args.methods.split(","):
                    method_seeds = none_seeds if method == "none" else seeds
                    method_lrs = [None] if method == "none" else lrs
                    fit_dirs = {}
                    for lr in method_lrs:
                        scores = []
                        for seed in method_seeds:
                            name = "seed%d" % seed if lr is None else "seed%d_lr%g" % (seed, lr)
                            fit_dir = root / "targets" / target / method / name
                            fit_target(args, checkpoint, target, method, seed, lr or 1e-3, fit_dir)
                            result = json.loads((fit_dir / "fit_result.json").read_text())
                            scores.append(float(result["best_prefix_val_r2"]))
                            fit_dirs[(lr, seed)] = fit_dir
                        print(f"[fit] {target}/{method} lr={lr}: prefix-val mean "
                              f"{sum(scores) / len(scores):.4f}", flush=True)
                    # Select lr by the mean prefix-validation over seeds, then
                    # evaluate the query for that lr only.
                    selected_lr = None
                    if method != "none":
                        means = {}
                        for lr in lrs:
                            vals = [json.loads((fit_dirs[(lr, s)] / "fit_result.json").read_text())
                                    ["best_prefix_val_r2"] for s in method_seeds]
                            means[lr] = sum(vals) / len(vals)
                        selected_lr = max(means, key=means.get)
                        print(f"[select] {target}/{method}: lr={selected_lr} "
                              f"(means {means})", flush=True)
                    for seed in method_seeds:
                        evaluate_fit(args, fit_dirs[(selected_lr, seed)])
    summary = aggregate(root)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["methods"], indent=2))
    if "paired_diffs_vs_reference" in summary:
        print("paired diffs vs", summary["reference"], ":",
              json.dumps(summary["paired_diffs_vs_reference"], indent=2))


if __name__ == "__main__":
    main()
