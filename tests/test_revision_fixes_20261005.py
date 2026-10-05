"""Tests for the 2026-10-05 revision fixes (roadmap items 2, 3, 7, 8, 9, 10, 11)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode import association as assoc


# --- item 2: mixed-channel latent routing -------------------------------

def test_latent_batch_routes_mixed_sessions():
    from ssm_decode.ap_backbone import APBackbone
    from ssm_decode.ap_pretraining import latent_batch
    trunk = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    model = APBackbone([6, 6], 2, k=4, base=trunk)
    x = torch.randn(5, 6, 6)
    sessions = torch.tensor([0, 1, 0, 1, 1])
    z = latent_batch(model, x, sessions)
    assert z.shape == (5, 6, 4)
    for session in (0, 1):
        rows = sessions == session
        assert torch.allclose(z[rows], model.latent(x[rows], session), atol=1e-6)
    # Item 2 regression: a mixed batch must never collapse onto tables[0].
    assert not torch.allclose(z[sessions == 1], model.latent(x[sessions == 1], 0))
    # Unequal channel counts cannot form a dense batch; the grouped path
    # raises an informative error instead of stacking mismatched tables.
    mixed = APBackbone([5, 7], 2, k=4, base=trunk)
    with pytest.raises(RuntimeError):
        latent_batch(mixed, torch.zeros(2, 6, 7), torch.tensor([0, 1]))


# --- item 3: core_step with batch != nheads ------------------------------

def test_core_step_batch_not_equal_heads():
    from ssm_decode import mamba3_streaming as ms
    try:
        Mamba3 = ms._import_mamba3()
    except RuntimeError as exc:
        pytest.skip(f"official Mamba-3 unavailable: {exc}")
    core = Mamba3(d_model=128, d_state=16, headdim=64, chunk_size=64, is_mimo=False)
    core.eval()
    states_b3 = ms.core_initial_states(core, 3)
    states_singles = [ms.core_initial_states(core, 1) for _ in range(3)]
    torch.manual_seed(0)
    outs = []
    with torch.inference_mode():
        for step in range(4):
            u = torch.randn(3, 128) * 0.05
            y, states_b3 = ms.core_step(core, u, states_b3)
            for i in range(3):
                yi, states_singles[i] = ms.core_step(core, u[i:i + 1], states_singles[i])
                assert torch.allclose(y[i], yi[0], atol=1e-4), \
                    f"batched step diverges from single step at t={step}, row={i}"
            outs.append(y)
    assert torch.isfinite(torch.stack(outs)).all()


# --- item 7: behavior features from the continuous recording -------------

def test_behavior_features_selected_from_continuous_recording():
    # Labeled bins may be non-contiguous.  Features must come from the
    # continuous recording, so a lag at a segment start sees the true
    # predecessor bin, not the previous labeled segment.
    y = np.arange(24, dtype=float).reshape(12, 2)
    phi_full = assoc.behavior_features(y, lags=1)
    labeled = np.array([5, 6, 10])
    selected = phi_full[labeled]
    # lag column of selected rows must equal y[4], y[5], y[9].
    assert np.allclose(selected[:, 2:4], y[[4, 5, 9]])
    # Contrast: features of the concatenated selection would use y[6] and y[9]-as-y[5].
    concatenated = assoc.behavior_features(y[labeled], lags=1)
    assert not np.allclose(selected[1:, 2:4], concatenated[1:, 2:4])


# --- items 8, 9: single-pass decode and frozen-base ui_grad ----------------

class _CountingBase(nn.Module):
    """Causal stand-in that counts forward calls and carries state."""

    def __init__(self, k=4, d=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.out = nn.Linear(k, d)
        self.calls = 0

    def forward(self, z):
        self.calls += 1
        return torch.cumsum(z, dim=1) * self.scale @ self.out.weight.T + self.out.bias


def _fake_model(k=4, n=6):
    from ssm_decode.ap_backbone import APBackbone
    model = APBackbone([n], 2, k=k, base=_CountingBase(k, 2))
    return model


def _fake_record(neural, behavior, good):
    return SimpleNamespace(neural=neural, behavior=behavior, eval_mask=good,
                           trial_bounds=[(0, len(neural))], session="fake")


def test_decode_r2_single_contiguous_pass():
    from ssm_decode.track_a_experiment import decode_r2_record
    model = _fake_model()
    rng = np.random.default_rng(1)
    x_z = rng.standard_normal((50, 6))
    table = rng.standard_normal((4, 6))
    record = _fake_record(x_z, np.stack([np.arange(50, dtype=float)] * 2, 1),
                          np.ones(50, dtype=bool))
    stats = (np.zeros(2), np.ones(2))
    decode_r2_record(model, x_z, table, np.arange(10, 40), record, stats,
                     device="cpu", latent_offset=np.zeros(4))
    # One recording, one forward: chunked resets would need several calls.
    assert model.base.calls == 1


def test_ui_grad_freezes_base_and_trains_table():
    from ssm_decode.track_a_experiment import ui_grad_table
    model = _fake_model()
    before = {n: p.detach().clone() for n, p in model.base.named_parameters()}
    rng = np.random.default_rng(2)
    x_z = rng.standard_normal((40, 6)).astype(np.float64)
    y = rng.standard_normal((40, 2))
    init = rng.standard_normal((4, 6))
    table, loss = ui_grad_table(model, x_z, y, [(0, 40)], init, device="cpu",
                                steps=5, lr=1e-2, seed=0,
                                valid_mask=np.ones(40, dtype=bool))
    for name, p in model.base.named_parameters():
        assert torch.equal(before[name], p.detach()), f"base parameter {name} changed"
    assert not np.allclose(table, init)


# --- item 10: checkpoint target must match the argument -------------------

def test_checkpoint_target_mismatch_raises(tmp_path):
    from ssm_decode.track_a_experiment import run
    ckpt = tmp_path / "best.pt"
    torch.save({"target_session": "sessionA", "source_sessions": ["s1"]}, ckpt)
    args = SimpleNamespace(task="m2", target_session="sessionB", checkpoint=str(ckpt),
                           output=str(tmp_path / "out"), device="cpu",
                           data_root="unused", cache_root="unused",
                           budgets="0", oracle=False, lags=2, fa_iters=5,
                           ui_steps=5, seed=0, lam=1.0, head_lam=1.0, ui_lr=3e-3)
    with pytest.raises(ValueError, match="checkpoint target"):
        run(args)


# --- item 11: atomic cache write and root key -----------------------------

def test_cache_root_key_and_atomic_write(tmp_path, monkeypatch):
    from ssm_decode import multi_target as mt
    built = {"count": 0}

    def fake_load(task, split, session, *, root):
        built["count"] += 1
        nwb = tmp_path / f"fake_{session}_{root.name}.nwb"
        nwb.write_bytes(b"nwb")
        return mt.Recording(task, split, session, nwb,
                            np.zeros((4, 3), dtype=np.float32),
                            np.zeros((4, 2), dtype=np.float32),
                            np.zeros(4, dtype=bool), np.ones(4, dtype=bool),
                            ((0, 2), (2, 4)))

    monkeypatch.setattr(mt, "load_recording", fake_load)
    monkeypatch.setattr(mt, "sha256_file", lambda p: "fixedhash")
    cache = tmp_path / "cache"
    a = mt.cached_recording("m2", "held_in", "s1", root=Path("/rootA"), cache_root=cache)
    b = mt.cached_recording("m2", "held_in", "s1", root=Path("/rootA"), cache_root=cache)
    assert built["count"] == 1 and a.session == b.session
    # Different data root: the entry must not be reused.
    mt.cached_recording("m2", "held_in", "s1", root=Path("/rootB"), cache_root=cache)
    assert built["count"] == 2
    # No half-written leftovers.
    assert not list(cache.rglob("*.tmp"))
