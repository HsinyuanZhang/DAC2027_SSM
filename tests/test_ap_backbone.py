"""Unit checks for the C8 AP backbone and multi-target rotation plans."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from ssm_decode.ap_backbone import APBackbone


def _tiny_model(n=12, k=4, seed=0):
    torch.manual_seed(seed)
    trunk = torch.nn.Sequential(torch.nn.Linear(k, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    return APBackbone([n, n], 2, k=k, base=trunk)


def test_forward_shapes_and_paths():
    model = _tiny_model()
    x = torch.randn(3, 10, 12)
    decode, head = model(x, 0)
    assert decode.shape == (3, 10, 2) and head.shape == (3, 10, 2)


def test_decode_with_table_matches_latent_path():
    model = _tiny_model(seed=1)
    x = torch.randn(2, 8, 12)
    table = model.tables[0].detach()
    direct = model.decode_with_table(x, table)
    decode, _ = model(x, 0)
    assert torch.allclose(direct, decode, atol=1e-6)


def test_whitening_penalty_orders_scales():
    model = _tiny_model()
    white = torch.randn(500, 4)
    scaled = white * 3.0
    mask = torch.ones(500, dtype=torch.bool)
    p_white = model.whitening_penalty(white, mask)
    p_scaled = model.whitening_penalty(scaled, mask)
    assert p_scaled > 10 * max(float(p_white), 1e-12)


def test_init_from_fa_sets_table_scale():
    model = _tiny_model()
    rng = np.random.default_rng(3)
    h = rng.standard_normal((12, 4))
    counts = rng.standard_normal((4000, 4))
    x = torch.from_numpy((counts @ h.T + rng.standard_normal((4000, 12)) * 0.2).astype(np.float32))
    diag = model.init_from_fa([x, x], em_iters=50)
    assert set(diag) == {0, 1}
    assert all(np.isfinite(v["psi_min"]) for v in diag.values())
    assert model.tables[0].shape == (4, 12)
    assert torch.isfinite(model.tables[0]).all()


@pytest.mark.skipif(not os.environ.get("SSM_INTEGRATION"),
                    reason="reads the real FALCON catalog from disk")
def test_rotation_plans_and_cache():
    from ssm_decode.multi_target import cached_recording, rotation_plans
    plans = rotation_plans("m2")
    assert len(plans) == 7
    targets = {p["target_session"] for p in plans}
    assert len(targets) == 7
    for plan in plans:
        assert plan["target_session"] not in plan["source_sessions"]
        assert len(plan["source_sessions"]) == 6
    # Cache round trip on the smallest surface: one minival session.
    from ssm_decode.data import visible_sessions
    mini = visible_sessions("m2")["minival"][0]["session"]
    first = cached_recording("m2", "minival", mini)
    second = cached_recording("m2", "minival", mini)
    assert first.session == second.session == mini
    assert np.allclose(first.neural, second.neural)
