"""Shared evaluation checks (review iteration-3, items A2/A3/A5/C1).

Both evaluation entry points use the same functions and the same bands.
Checks RECORD violations into result rows instead of aborting the whole
evaluation, so one weak method cannot destroy every other method's numbers.
Each check has a unit test that fails when the check is broken.
"""
from __future__ import annotations

import numpy as np

# Unified physical-unit band.  Catches order-of-magnitude unit errors
# (review A1 of iteration 2) while tolerating amplitude shrinkage of weak
# tables, which is model quality rather than a unit error.
UNIT_RATIO_BAND = (0.1, 10.0)


def physical_unit_ratio(pred: np.ndarray, truth: np.ndarray) -> float:
    """Standard-deviation ratio of prediction to truth."""
    return float(np.std(pred) / max(np.std(truth), 1e-9))


def check_physical_units(pred: np.ndarray, truth: np.ndarray) -> dict:
    ratio = physical_unit_ratio(pred, truth)
    lo, hi = UNIT_RATIO_BAND
    return {"check": "physical_units", "unit_ratio": ratio,
            "passed": bool(lo <= ratio <= hi),
            "detail": f"ratio {ratio:.3f} outside [{lo}, {hi}]" if not (lo <= ratio <= hi) else ""}


def check_source_gate(blob: dict, minimum: float | None) -> dict:
    """Compare the checkpoint's source validation score against a fixed bar.

    ``minimum`` must come from an external reference (for example the AP
    backbone on the same pool), never from the evaluated checkpoint itself
    (review A2 of iteration 3).
    """
    score = float(blob.get("best_source_val_trunk_r2", float("nan")))
    if minimum is None:
        return {"check": "source_gate", "source_val_r2": score, "passed": True,
                "detail": "no gate configured"}
    passed = bool(np.isfinite(score) and score >= minimum)
    return {"check": "source_gate", "source_val_r2": score, "minimum": float(minimum),
            "passed": passed,
            "detail": "" if passed else f"source val {score} below gate {minimum}"}


def check_budget_rule(blob: dict) -> dict:
    """best_step must land outside the last 20 percent of the trained budget.

    A best step of 0 is legal for a distillation-only selection (review B1
    of iteration 3).
    """
    args = blob.get("args", {})
    steps, best = int(args.get("steps", 0)), int(blob.get("best_step", -1))
    if steps <= 0 or best < 0:
        return {"check": "budget_rule", "steps": steps, "best_step": best,
                "passed": False, "detail": "missing steps or best_step"}
    passed = bool(best <= 0.8 * steps)
    return {"check": "budget_rule", "steps": steps, "best_step": best,
            "passed": passed,
            "detail": "" if passed else f"best_step {best} in last 20% of {steps}"}


def flatten_checks(checks: list[dict]) -> dict:
    """Merge check records into one row-friendly dict."""
    flat = {}
    for record in checks:
        name = record.pop("check")
        for key, value in record.items():
            flat[f"{name}_{key}"] = value
        flat[f"{name}_passed"] = record.get("passed", None)
    return flat
