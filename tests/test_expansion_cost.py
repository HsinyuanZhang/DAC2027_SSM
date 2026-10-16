"""Cost model sanity (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import expansion_cost as ec                         # noqa: E402
from ssm_decode import expansion_protocol as ep                  # noqa: E402


def costs(f=2.0, draws=4):
    inv = ep.load_inventory()
    return {b.bid: ec.backbone_cost(b, inv, draws, f) for b in ep.select_tier(ep.build_backbones(), tier="primary")}


def test_upper_bound_dominates_expected_and_scales_with_l4_factor():
    a, b = costs(2.0), costs(2.5)
    for bid in a:
        assert a[bid]["upper"]["wall_s"] > a[bid]["expected"]["wall_s"] > 0
        assert b[bid]["upper"]["wall_s"] > a[bid]["upper"]["wall_s"]
    assert costs(2.0, 2)["sc_0708"]["upper"]["wall_s"] < a["sc_0708"]["upper"]["wall_s"]


def test_reused_backbone_skips_backbone_and_p0_training():
    c = costs()
    assert c["sm_0307"]["upper"]["gpu_3090_s"] < c["sm_0627"]["upper"]["gpu_3090_s"]


def test_full_queue_fits_the_cu_cap_and_the_two_round_split_fits_a_session():
    c = costs()
    total_cu = sum(v["upper"]["wall_s"] for v in c.values()) / 3600 * ec.CU_PER_HOUR
    assert total_cu < ec.CU_CAP
    plan = [["sc_0708", "sc_0713", "sc_0629"], ["sc_0309", "sm_0307", "sm_0627", "sm_0512", "sm_0611"]]
    assert sorted(b for r in plan for b in r) == sorted(c)
    for r in plan:
        assert sum(c[b]["upper"]["wall_s"] for b in r) / 3600 < 10.5          # --deadline-hours default
    rounds = ec.split_rounds(c, 2, "upper")
    assert sorted(b for r in rounds for b in r) == sorted(c)
