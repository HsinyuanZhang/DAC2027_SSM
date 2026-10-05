"""C10 state budget anchor checks."""
import pytest

from ssm_decode.state_budget import (BUDGET_BYTES, C10_VARIANTS, state_bytes,
                                     state_report, state_values)


def test_values_formula():
    # d_state=32: 64*32 + 8 + 32 + 64 = 2152 carried values per head-layer.
    assert state_values(32) == 2152
    assert state_values(8) == 64 * 8 + 2 + 8 + 64


def test_wide_anchor_matches_roadmap():
    # Roadmap: 256x4, N=32 needs approximately 202 KB at INT24.
    assert abs(state_bytes(256, 4, 32) / 1024 - 201.9) < 0.2


def test_small_reference_under_budget():
    assert state_bytes(128, 2, 32) / 1024 < 128


def test_c10_variants_within_budget_except_wide():
    for name, cfg in C10_VARIANTS.items():
        report = state_report(**cfg)
        if name == "wide_ref":
            assert not report["within_budget"]
        else:
            assert report["within_budget"], f"{name} exceeds budget: {report}"
