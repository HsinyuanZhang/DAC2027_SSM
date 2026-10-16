"""Source-only rule selection (R1-stream / R2 / R3) on synthetic episode logs (no torch)."""
from __future__ import annotations

import pytest

from ssm_decode import expansion_rules as er


def r12_row(B, p0, best_kind="lr16", best_lam=0.1, best_cfg=(16, 10.0, 0.1, 3), bump=0.05):
    r2 = {"p0": p0}
    for k in er.READOUT_KINDS:
        for lam in er.RIDGE_LAMS:
            r2[f"{k}|{lam}"] = p0 + (bump if (k, lam) == (best_kind, best_lam) else 0.0) - 0.001 * er.RIDGE_LAMS.index(lam)
    cfg = {}
    for L in (8, 16, 32):
        for S in (1, 3):
            cfg[er.cfg_key(L, 10.0, 0.1, S)] = p0 + 0.01
    cfg[er.cfg_key(*best_cfg)] = p0 + 0.2
    return {"B": B, "r2": r2, "r2cfg": cfg}


def r3_row(B, best_kind="full", best_lam=0.01):
    r3 = {f"{k}|{lam}": 0.5 - 0.1 * (k != best_kind) - 0.01 * er.RIDGE_LAMS.index(lam) + (0.2 if (k, lam) == (best_kind, best_lam) else 0)
          for k in er.READOUT_KINDS for lam in er.RIDGE_LAMS}
    return {"B": B, "r2sel": 0.6, "r3": r3}


def test_r1_picks_best_kind_and_lambda_by_mean_source_score():
    sel = er.select_r1([r12_row(4, 0.5), r12_row(4, 0.6)])
    assert (sel["kind"], sel["lam"]) == ("lr16", 0.1) and sel["n_episodes"] == 2
    assert sel["p0_source_r2"] == pytest.approx(0.55) and set(sel["per_kind"]) == set(er.READOUT_KINDS)


def test_r2_picks_best_feedback_rule_and_round_trips_the_key():
    sel = er.select_r2([r12_row(4, 0.5), r12_row(4, 0.7)])
    assert (sel["L"], sel["eta"], sel["rho"], sel["S"]) == (16, 10.0, 0.1, 3)
    assert er.parse_cfg_key(er.cfg_key(32, 1000.0, 0.03, 10)) == {"L": 32, "eta": 1000.0, "rho": 0.03, "S": 10}


def test_r3_selects_kind_and_lambda_on_the_refined_tables():
    sel = er.select_r3([r3_row(4), r3_row(4)])
    assert (sel["kind"], sel["lam"]) == ("full", 0.01) and sel["r2_source_r2"] == pytest.approx(0.6)


def test_ties_are_deterministic_first_sorted_key():
    row = {"B": 1, "r2cfg": {er.cfg_key(16, 10.0, 0.1, 3): 0.5, er.cfg_key(8, 10.0, 0.1, 3): 0.5}}
    assert er.select_r2([row])["L"] == 16          # sorted string order: "16|..." < "8|..."
    assert er.select_r2([row, row])["L"] == 16


def test_assemble_requires_every_budget_and_records_scope():
    bs = (1, 4)
    r12 = [r12_row(B, 0.5) for B in bs for _ in range(2)]
    r3 = [r3_row(B) for B in bs for _ in range(2)]
    out = er.assemble_selection("sc_x", "sub-C_ses-CO-20150707", True, "OOS", bs, r12, r3)
    assert out["selection_oos"] is True and out["rule_version"] == "expansion_v2"
    assert set(out["b"]) == {"1", "4"} and set(out["b"]["4"]) == {"r1s", "r2", "r3"}
    with pytest.raises(ValueError, match="no selection episodes"):
        er.assemble_selection("sc_x", "s", True, "OOS", (1, 4, 8), r12, r3)
