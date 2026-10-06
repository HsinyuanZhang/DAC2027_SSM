"""Tests for the shared evaluation checks (review iteration-3, 4.4).

Each test fails when its check is broken: the checks must be able to fail.
"""
from __future__ import annotations

import numpy as np

from ssm_decode.eval_checks import (check_budget_rule, check_physical_units,
                                    check_source_gate, flatten_checks,
                                    physical_unit_ratio)


def test_physical_units_passes_on_matching_scale():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((500, 2))
    pred = truth * 0.8
    record = check_physical_units(pred, truth)
    assert record["passed"] is True
    assert 0.7 < record["unit_ratio"] < 0.9


def test_physical_units_fails_on_unit_error():
    rng = np.random.default_rng(1)
    truth = rng.standard_normal((500, 2)) * 20.0     # raw units
    pred = rng.standard_normal((500, 2))            # normalized units
    record = check_physical_units(pred, truth)
    assert record["passed"] is False
    assert record["unit_ratio"] < 0.1


def test_physical_units_tolerates_weak_table_shrinkage():
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((500, 2))
    pred = truth * 0.2 + 0.05 * rng.standard_normal((500, 2))
    assert check_physical_units(pred, truth)["passed"] is True


def test_source_gate_fails_below_external_minimum():
    blob = {"best_source_val_trunk_r2": 0.607}
    gate = check_source_gate(blob, 0.665)
    assert gate["passed"] is False
    assert gate["minimum"] == 0.665


def test_source_gate_passes_at_minimum():
    gate = check_source_gate({"best_source_val_trunk_r2": 0.70}, 0.665)
    assert gate["passed"] is True


def test_source_gate_fails_on_missing_score():
    gate = check_source_gate({}, 0.5)
    assert gate["passed"] is False


def test_budget_rule_fails_in_last_twenty_percent():
    blob = {"args": {"steps": 16000}, "best_step": 13700}
    assert check_budget_rule(blob)["passed"] is False


def test_budget_rule_passes_when_converged():
    blob = {"args": {"steps": 16000}, "best_step": 11500}
    assert check_budget_rule(blob)["passed"] is True


def test_budget_rule_allows_distill_only_selection():
    # best_step 0 = distillation-only selection stayed best (review B1).
    blob = {"args": {"steps": 4000}, "best_step": 0}
    assert check_budget_rule(blob)["passed"] is True


def test_budget_rule_fails_on_missing_fields():
    assert check_budget_rule({})["passed"] is False


def test_flatten_checks_merges_records():
    flat = flatten_checks([{"check": "physical_units", "unit_ratio": 1.2, "passed": True,
                            "detail": ""},
                           {"check": "budget_rule", "steps": 100, "best_step": 50,
                            "passed": True, "detail": ""}])
    assert flat["physical_units_unit_ratio"] == 1.2
    assert flat["physical_units_passed"] is True
    assert flat["budget_rule_steps"] == 100
    assert flat["budget_rule_passed"] is True


def test_ratio_helper_is_pure():
    rng = np.random.default_rng(3)
    truth = rng.standard_normal((10, 1))
    pred = truth * 2.0
    assert abs(physical_unit_ratio(pred, truth) - 2.0) < 1e-6
    # Constant arrays degenerate to zero std; the helper must not raise.
    assert physical_unit_ratio(np.ones((5, 1)), np.ones((5, 1))) >= 0.0
