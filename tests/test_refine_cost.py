"""Analytic cost model of the P3 refinements and the trunk MAC counter."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from ssm_decode import fwd_refine as fr
from ssm_decode.refine_cost import _readout_macs, _readout_stat_entries, p0_macs, refinement_cost


class _FakeSSM(nn.Module):
    d_state, headdim, d_inner = 8, 4, 16

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(5, 7)


class _FakeTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 5)
        self.ssm = _FakeSSM()
        self.out_proj = nn.Linear(5, 2)


def test_trunk_macs_counter():
    m = fr.trunk_macs_per_bin(_FakeTrunk())
    assert m["linear_macs"] == 3 * 5 + 5 * 7 + 5 * 2
    assert m["scan_macs"] == 2 * 16 * 8
    assert m["state_elems"] == 16 * 8
    assert m["total"] == m["linear_macs"] + m["scan_macs"]


CASE = {"name": "t", "N": 6, "K": 3, "D": 2, "H": 10, "T": 20, "bx": 4, "by": 16,
        "trunk": {"total": 1000, "state_elems": 100}}


def test_readout_entry_and_mac_formulas():
    assert _readout_stat_entries(10, 2, "full") == 11 * 12 // 2 + 2 * 11
    assert _readout_stat_entries(10, 2, "diag") == 11 + 22
    assert _readout_stat_entries(10, 2, "lr4") == 4 * 5 // 2 + 2 * 4
    acc, solve = _readout_macs(10, 2, 7, "full")
    assert acc == 7 * (66 + 22) and solve == 11 ** 3 // 6 + 2 * 2 * 121


def test_r1_cost_is_one_replay_pass_and_buffer_matches_bits():
    r = refinement_cost(CASE, 4, "r1_full")
    nb = 4 * 20
    assert r["replay_bins"] == nb
    assert r["macs"]["replay forward (trunk+read-in)"] == nb * (1000 + 3 * 6)
    assert r["replay_buffer_bytes"] == nb * (6 * 4 + 2 * 16 + 1) / 8


def test_f7j_cost_scales_with_steps_and_r3_adds_one_pass():
    r2a = refinement_cost(CASE, 4, "r2", steps=2, lags=8)
    r2b = refinement_cost(CASE, 4, "r2", steps=4, lags=8)
    assert r2b["macs"]["replay forward (trunk+read-in) x S"] == 2 * r2a["macs"]["replay forward (trunk+read-in) x S"]
    r3 = refinement_cost(CASE, 4, "r3_full", steps=4, lags=8)
    assert r3["macs"]["replay forward for readout pass"] == r2b["macs"]["replay forward (trunk+read-in) x S"] // 4
    assert r3["macs_total"] > r2b["macs_total"]


def test_streaming_variant_has_no_replay_buffer_but_same_forward_macs():
    r1 = refinement_cost(CASE, 5, "r1_lr4")
    rs = refinement_cost(CASE, 5, "r1s_lr4")
    assert "raw replay buffer" not in rs["sram_bytes"] and rs["replay_buffer_bytes"] == 0.0
    assert rs["macs"]["streaming forward (trunk+read-in), once per arriving bin"] == \
        r1["macs"]["replay forward (trunk+read-in)"]
    assert rs["sram_total"] < r1["sram_total"]


def test_p0_baseline_macs_formula():
    m = p0_macs(6, 3, 2, 100, 4)
    assert m["accumulate"] == 100 * (36 + 16 * 6 + 9) + 4 * (6 + 24 + 2)
    assert m["finalize"] == 7 * 36 + 30 * 6 + 27


# ---------------------------------------------------- review-fix additions ----

def test_actual_replay_bins_and_staging_override_the_mean_T_sizing():
    base = refinement_cost(CASE, 4, "r1_lr4")
    big = refinement_cost(CASE, 4, "r1_lr4", replay_bins=1000)
    assert base["replay_bins"] == 80 and big["replay_bins"] == 1000
    assert big["replay_buffer_bytes"] == 1000 * (6 * 4 + 2 * 16 + 1) / 8
    assert big["macs"]["replay forward (trunk+read-in)"] == 1000 * (1000 + 3 * 6)
    s0 = refinement_cost(CASE, 4, "r1s_lr4")
    s1 = refinement_cost(CASE, 4, "r1s_lr4", staging_bins=500)
    k = "one-trial staging buffer (trial 0 normalisation)"
    assert s0["sram_bytes"][k] == 20 * (6 * 4 + 2 * 16 + 1) / 8
    assert s1["sram_bytes"][k] == 500 * (6 * 4 + 2 * 16 + 1) / 8


def test_mac_classes_partition_the_total_and_scan_is_not_fixed_weight():
    case = {**CASE, "trunk": {"total": 1000, "fixed_weight_macs": 900, "act_x_act_macs": 100, "state_elems": 100}}
    for method, kw in (("r1_full", {}), ("r1s_lr4", {}), ("r2", {"steps": 3, "lags": 4}), ("r3_lr4", {"steps": 3, "lags": 4})):
        r = refinement_cost(case, 5, method, **kw)
        by = r["macs_by_class"]
        assert sum(by.values()) == r["macs_total"]
        assert by["fixed_weight_rram"] < r["macs_total"]               # scan, Gram, outer product are not RRAM weights
        assert by["act_x_act"] > 0
    r = refinement_cost(case, 5, "r1_full")
    assert r["macs_by_class"]["fixed_weight_rram"] == 5 * 20 * 900
    assert r["macs_by_class"]["act_x_act"] >= 5 * 20 * 100
    assert r["macs_by_class"]["sram_table_x_act"] == 5 * 20 * 3 * 6
    assert r["fixed_weight_fraction"] < 0.9
