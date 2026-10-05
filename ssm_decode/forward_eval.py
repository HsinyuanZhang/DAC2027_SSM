"""Multi-target time-forward evaluation protocol (revision 4.1).

Plans
-----
``forward_plans`` groups the held-in sessions of one task by recording day.
Each target session uses every session from strictly earlier days as its
source pool.  Targets that share a day share one pool, so the pool is
pretrained once and every checkpoint consumer reuses it.  ``leave-one-out``
mode keeps the earlier rotation as a control.

Rules
-----
1. Query labels never enter selection.  Adaptation fits select the best step
   on prefix validation; the query evaluation runs afterwards.
2. A pretraining whose best step lands in the last 20 percent of the budget
   is retrained with double the budget, at most twice.
3. Aggregation reports the multi-target mean, per-target paired differences
   against a reference method, and the per-target seed standard deviation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import DEFAULT_ROOT, visible_sessions

CAPACITIES = {
    "small": {"width": 128, "layers": 2, "state_size": 32},
    "wide": {"width": 256, "layers": 4, "state_size": 32},
}
SUPPORT_TRIALS = {"m1": 10, "m2": 33}


def _day(session: str) -> str:
    return session.split("-Run")[0] if "-Run" in session else session


def forward_plans(task: str, *, mode: str = "forward", root: Path = DEFAULT_ROOT) -> dict:
    """Build the rotation plan for one task.

    Returns ``{'pools': {pool_key: {'sources', 'targets'}}, 'mode': ...}``.
    ``pool_key`` orders pools chronologically so runners can queue them.
    """
    held = [r["session"] for r in visible_sessions(task, root=root)["held_in"]]
    if len(held) < 3:
        raise ValueError(f"{task}: forward protocol needs >=3 held-in sessions")
    if mode == "forward":
        days = sorted({_day(s) for s in held})
        pools: dict[str, dict] = {}
        for index, day in enumerate(days):
            if index == 0:
                continue
            sources = [s for s in held if _day(s) in days[:index]]
            targets = [s for s in held if _day(s) == day]
            pools[f"pool{index}_{days[index - 1]}_to_{day}"] = {
                "sources": sources, "targets": targets}
        if not pools:
            raise ValueError(f"{task}: no forward pool exists")
        return {"task": task, "mode": mode, "pools": pools}
    if mode == "leave_one_out":
        pools = {f"loo_{target}": {"sources": [s for s in held if s != target],
                                   "targets": [target]} for target in held}
        return {"task": task, "mode": mode, "pools": pools}
    raise ValueError(f"unknown mode {mode!r}")


def pool_key_for_sources(plans: dict, sources: list[str]) -> str | None:
    wanted = set(sources)
    for key, pool in plans["pools"].items():
        if set(pool["sources"]) == wanted:
            return key
    return None


def load_metrics(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("targets/*/*/seed*/query_evaluation/metrics.json")):
        blob = json.loads(path.read_text())
        import re

        parts = path.parts
        target = parts[-5]
        method = parts[-4]
        match = re.match(r"seed(\d+)", parts[-3])
        if not match:
            continue
        seed = int(match.group(1))
        rows.append({"target": target, "method": method, "seed": seed,
                     "metrics": blob, "path": str(path)})
    return rows


def aggregate(root: Path, *, reference: str = "none") -> dict:
    """Multi-target mean, per-target paired differences, seed variance."""
    import numpy as np

    rows = load_metrics(root)
    if not rows:
        raise ValueError(f"no query metrics under {root}")
    def r2(row):
        metrics = row["metrics"]
        # Primary cohort: direct decode, all-valid query bins.
        try:
            return float(metrics["final"]["zero"]["all_valid"]["r2_variance_weighted"])
        except (KeyError, TypeError):
            pass
        try:
            return float(metrics["final"]["zero"]["r2_variance_weighted"])
        except (KeyError, TypeError):
            raise KeyError(f"no R2 key in metrics structure {list(metrics)}") from None
    targets = sorted({r["target"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    summary = {"targets": targets, "reference": reference, "methods": {}}
    for method in methods:
        per_target = {}
        for target in targets:
            values = [r2(r) for r in rows if r["method"] == method and r["target"] == target]
            if values:
                per_target[target] = {"mean": float(np.mean(values)),
                                      "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                      "seeds": len(values)}
        if per_target:
            grand = [v["mean"] for v in per_target.values()]
            summary["methods"][method] = {
                "multi_target_mean": float(np.mean(grand)),
                "per_target": per_target}
    if reference in summary["methods"]:
        diffs = {}
        for method in methods:
            if method == reference:
                continue
            pairs = []
            for target in targets:
                a = summary["methods"].get(method, {}).get("per_target", {}).get(target)
                b = summary["methods"][reference]["per_target"].get(target)
                if a and b:
                    pairs.append(a["mean"] - b["mean"])
            if pairs:
                diffs[method] = {"mean": float(np.mean(pairs)),
                                 "wins": int(sum(p > 0 for p in pairs)),
                                 "n": len(pairs), "per_target": pairs}
        summary["paired_diffs_vs_reference"] = diffs
    return summary
