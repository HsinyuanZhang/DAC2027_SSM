"""Tests for the P2 iteration: identity-free priors, quantisers, LUT softmax,
ui_reg and the analytic cost model."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode import calib_cost, calib_quant as cq
from ssm_decode.population_calib import (PopStats, PopulationCalibrator, corr_columns,
                                         desc_dim, describe, identity_free_prior,
                                         prior_columns)
from ssm_decode.ui_reg import ui_reg_trajectory
from test_population_calib import K, _model, _session

# ------------------------------------------------------- identity-free prior ----


def test_identity_free_prior_is_unit_blind():
    rng = np.random.default_rng(0)
    tabs = [rng.standard_normal((K, n)) for n in (9, 12, 7)]
    gm = identity_free_prior(tabs, 9, "globalmean")
    assert gm.shape == (K, 9)
    assert np.allclose(gm, gm[:, :1])                       # same column for every unit
    np.testing.assert_allclose(gm[:, 0], np.concatenate(tabs, 1).mean(1))
    # leave-one-out drops exactly one session's columns
    loo = identity_free_prior(tabs, 5, "globalmean", exclude=1)
    np.testing.assert_allclose(loo[:, 0], np.concatenate([tabs[0], tabs[2]], 1).mean(1))
    assert not np.allclose(loo[:, 0], gm[:, 0])
    assert identity_free_prior(tabs, 4, "zero").shape == (K, 4)
    assert not identity_free_prior(tabs, 4, "zero").any()
    with pytest.raises(ValueError):
        identity_free_prior(tabs, 4, "aligned")


@pytest.mark.parametrize("mode", ["globalmean", "zero"])
@pytest.mark.parametrize("variant", ["p0", "p2", "p4"])
def test_noalign_calibrator_ignores_prior_column_and_unit_order(mode, variant):
    sd, _ = _session(12)
    fd = desc_dim(K)
    rng = np.random.default_rng(1)
    tabs = [rng.standard_normal((K, 9)) for _ in range(3)]
    prior = torch.from_numpy(identity_free_prior(tabs, sd.n_units, mode))
    idx = np.array([0, 2, 5, 7])
    stats = PopStats.from_trials(sd.bank, idx)
    model = _model(variant, fd, m=4, prior_mode=mode)
    d, r, _, _ = describe(stats, prior)
    with torch.no_grad():
        base = model(d, r, prior.float())
        d2 = d.clone()
        d2[:, prior_columns(K)] += 5.0                      # prior column feature is masked
        np.testing.assert_allclose(model(d2, r, prior.float()).numpy(), base.numpy(), atol=1e-6)
        if mode == "zero":
            d3 = d.clone()
            d3[:, corr_columns(K)] += 5.0                   # corr features void under zero prior
            np.testing.assert_allclose(model(d3, r, prior.float()).numpy(), base.numpy(), atol=1e-6)
        # unit order cannot matter: no electrode/identity input exists
        perm = np.random.default_rng(2).permutation(sd.n_units)
        dp, rp, _, _ = describe(stats.select_units(perm), prior[:, perm])
        np.testing.assert_allclose(model(dp, rp, prior[:, perm].float()).numpy(),
                                   base[:, perm].numpy(), atol=1e-5, rtol=1e-4)


def test_aligned_model_does_read_the_prior_column():
    """Non-vacuity: the default (aligned) model is NOT blind to the prior column."""
    sd, prior = _session(13)
    stats = PopStats.from_trials(sd.bank, np.array([1, 3, 4]))
    model = _model("p0", desc_dim(K))
    d, r, _, _ = describe(stats, prior)
    with torch.no_grad():
        a = model(d, r, prior.float())
        d2 = d.clone()
        d2[:, prior_columns(K)] += 5.0
        b = model(d2, r, prior.float())
    assert (a - b).abs().max() > 1e-4


def test_globalmean_corr_feature_uses_population_latent():
    """With a constant-column prior the K latents are copies of the population sum."""
    sd, _ = _session(14)
    tabs = [np.random.default_rng(3).standard_normal((K, 9))]
    prior = torch.from_numpy(identity_free_prior(tabs, sd.n_units, "globalmean"))
    d, _, _, _ = describe(PopStats.from_trials(sd.bank, np.array([0, 1, 2, 3])), prior)
    corr = d[:, corr_columns(K)]
    assert torch.allclose(corr.abs(), corr[:, :1].abs().expand_as(corr), atol=1e-5)
    dz, _, _, _ = describe(PopStats.from_trials(sd.bank, np.array([0, 1, 2, 3])), torch.zeros(K, 9))
    assert float(dz[:, corr_columns(K)].abs().max()) < 1e-3


# ------------------------------------------------------------- quantisers ----


@pytest.mark.parametrize("bits", [8, 4])
def test_sym_row_quant_roundtrip_and_idempotence(bits):
    torch.manual_seed(0)
    w = torch.randn(7, 33, dtype=torch.float64)
    q = cq.quant_sym_rows(w, bits)
    scale = w.abs().amax(1, keepdim=True) / cq.qmax(bits)
    assert (q - w).abs().max() <= scale.max() / 2 + 1e-12      # half-step error bound
    levels = torch.round(q / scale)
    assert levels.abs().max() <= cq.qmax(bits)
    torch.testing.assert_close(levels, levels.round())          # integer codes
    for r in range(w.shape[0]):
        assert len(torch.unique(levels[r])) <= 2 ** bits - 1
    torch.testing.assert_close(cq.quant_sym_rows(q, bits), q, atol=1e-12, rtol=1e-12)


def test_ternary_twn_values_threshold_and_idempotence():
    torch.manual_seed(1)
    w = torch.randn(6, 64, dtype=torch.float64)
    q = cq.quant_ternary_rows(w)
    for r in range(6):
        vals = torch.unique(q[r])
        assert len(vals) <= 3
        nz = vals[vals != 0]
        assert len(nz) == 2 and torch.isclose(nz[0], -nz[1])     # +-alpha
        delta = 0.7 * w[r].abs().mean()
        assert ((q[r] != 0) == (w[r].abs() > delta)).all()
        alpha = w[r].abs()[w[r].abs() > delta].mean()
        assert torch.isclose(nz.abs().max(), alpha)
    torch.testing.assert_close(cq.quant_ternary_rows(q), q, atol=1e-12, rtol=1e-12)


def test_quantize_calibrator_copy_semantics():
    model = _model("p4", desc_dim(K), m=4)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    q = cq.quantize_calibrator(model, "w4")
    for k, v in model.state_dict().items():                       # original untouched
        assert torch.equal(v, ref[k])
    for name, mod in q.named_modules():
        if isinstance(mod, nn.Linear):
            for r in range(mod.weight.shape[0]):
                assert len(torch.unique(mod.weight[r])) <= 15
            torch.testing.assert_close(mod.bias, dict(model.named_modules())[name].bias)
    assert (q.u != model.u).any()
    t8 = cq.quantize_calibrator(model, "w8")
    d = torch.randn(9, desc_dim(K))
    r = torch.randn(9, 6)
    pr = torch.zeros(K, 9)
    with torch.no_grad():
        fp, e8, e4 = model(d, r, pr), t8(d, r, pr), q(d, r, pr)
    assert (e8 - fp).abs().max() < (e4 - fp).abs().max() + 1e-9   # 8 bit closer than 4 bit


def test_fixed_point_roundtrip_saturation_and_steps():
    step = cq.fixed_step(100.0, 12)
    assert step == 2.0 ** math.ceil(math.log2(100.0 / 2047))
    assert 100.0 <= step * cq.qmax(12)
    x = torch.linspace(-150, 150, 1001, dtype=torch.float64)
    q = cq.to_fixed(x, step, 12)
    assert q.abs().max() <= step * cq.qmax(12) + 1e-12            # saturates
    inside = x.abs() <= 100.0
    assert (q[inside] - x[inside]).abs().max() <= step / 2 + 1e-12
    torch.testing.assert_close(cq.to_fixed(q, step, 12), q)       # idempotent
    t = torch.randn(5, 11, dtype=torch.float64)
    tq = cq.quant_table(t, 4)
    torch.testing.assert_close(cq.quant_table(tq, 4), tq, atol=1e-12, rtol=1e-12)
    for k in range(5):
        assert len(torch.unique(tq[k])) <= 15                      # per-row scale, 4 bit


def test_quantized_stats_exact_when_wide_and_on_grid_when_narrow():
    sd, prior = _session(15)
    idx = np.array([0, 2, 3, 6, 8])
    exact = PopStats.from_trials(sd.bank, idx)
    steps_w = cq.acc_steps([exact], 31)
    wide = cq.quantized_stats(sd.bank, idx, steps_w, 31)
    for k in exact.d:
        torch.testing.assert_close(wide.d[k], exact.d[k], atol=1e-5 * float(exact.d[k].abs().max()) + 1e-9,
                                   rtol=1e-4)
    steps8 = cq.acc_steps([exact], 8)
    narrow = cq.quantized_stats(sd.bank, idx, steps8, 8)
    for k, v in narrow.d.items():
        grid = v / steps8[k]
        torch.testing.assert_close(grid, grid.round())
        assert float(v.abs().max()) <= steps8[k] * 127 + 1e-9
    assert (narrow.d["gram"] - exact.d["gram"]).abs().max() > 0 or steps8["gram"] <= 1


# ------------------------------------------------------------ exp2 LUT softmax ----


def test_exp2_lut_error_bound():
    lut = cq.Exp2LUT()
    assert lut.codes.max() < 2 ** 16 and lut.codes.min() >= 2 ** 15   # 16-bit unsigned codes
    assert len(lut.codes) == 256
    x = -torch.rand(20000, dtype=torch.float64) * 12
    err = (lut.exp(x) - torch.exp(x)).abs()
    rel_bound = 2 ** (0.5 / 256) - 1 + 2.0 ** -16                   # mid-point LUT + 15-bit code rounding
    assert (err <= rel_bound * torch.exp(x) + 2.0 ** -17 + 1e-12).all()
    assert float(lut.exp(torch.zeros(1, dtype=torch.float64))) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("recip", ["exact", "lut"])
def test_lut_softmax_error_bound(recip):
    torch.manual_seed(0)
    sm = cq.make_lut_softmax(recip)
    rel = (2 ** (0.5 / 256) - 1 + 2.0 ** -16) * 2          # numerator and normaliser errors
    if recip == "lut":
        rel += 0.0021                                      # reciprocal LUT bound (see below)
    for n, m in ((26, 8), (256, 8), (5, 16)):
        x = torch.randn(n, m, dtype=torch.float64) * 3
        for dim in (0, 1):
            p, q = torch.softmax(x, dim), sm(x, dim)
            n_red = x.shape[dim]
            bound = rel * p + 2.0 ** -17 * (1 + n_red * p) * (1.003 if recip == "exact" else 1.01)
            assert ((q - p).abs() <= bound).all()
            assert abs(float(q.sum(dim).mean()) - 1.0) < rel + 1e-3


def test_lut_reciprocal_bound():
    s = torch.rand(5000, dtype=torch.float64) * 300 + 1.0
    rel = (cq.lut_reciprocal(s) * s - 1).abs()
    assert rel.max() < 0.0021


def test_p4_forward_with_lut_softmax_close_to_fp():
    sd, prior = _session(16)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1, 2, 5]))
    model = _model("p4", desc_dim(K), m=8)
    d, r, _, _ = describe(stats, prior)
    with torch.no_grad():
        fp = model(d, r, prior.float())
        model.softmax_fn = cq.make_lut_softmax("lut")
        lut = model(d, r, prior.float())
    assert model.softmax_fn is not torch.softmax
    rel = (lut - fp).norm() / (fp - prior.float()).norm().clamp_min(1e-9)
    assert float(rel) < 0.05


# ------------------------------------------------------------------ ui_reg ----


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.rnn = nn.GRU(2, 2, batch_first=True)

    def forward(self, x):
        h, _ = self.rnn(torch.tanh(self.lin(x)))
        return h


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.base = _Base()


def _ui_problem():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 5))
    y = rng.standard_normal((120, 2))
    windows = [(0, 30), (40, 70), (80, 110)]
    t0 = rng.standard_normal((3, 5)) * 0.3
    return x, y, windows, t0


def test_ui_reg_lambda_zero_equals_unregularised_reference():
    from ssm_decode.track_a_experiment import ui_grad_table
    x, y, windows, t0 = _ui_problem()
    model = _Model()
    ref, _ = ui_grad_table(model, x, y, windows, t0, device="cpu", steps=25, lr=3e-3, seed=0)
    out = ui_reg_trajectory(model, x, y, windows, t0, t0, lam=0.0, lr=3e-3, snapshots=(10, 25),
                            device="cpu", seed=0)
    np.testing.assert_allclose(out[25], ref, atol=1e-6)
    assert not np.allclose(out[10], out[25])


def test_ui_reg_penalty_pulls_toward_center_and_leaves_trunk_untouched():
    x, y, windows, t0 = _ui_problem()
    model = _Model()
    before = [p.detach().clone() for p in model.base.parameters()]
    dist = []
    for lam in (0.0, 1e-2, 10.0):
        out = ui_reg_trajectory(model, x, y, windows, t0, t0, lam=lam, lr=1e-2, snapshots=(60,),
                                device="cpu", seed=0)
        dist.append(float(np.linalg.norm(out[60] - t0)))
    assert dist[0] > dist[1] > dist[2]
    assert all(torch.equal(a, b) for a, b in zip(before, model.base.parameters()))
    assert all(p.requires_grad for p in model.base.parameters())    # restored


# ----------------------------------------------------------------- cost model ----


def test_cost_model_matches_actual_tensor_shapes():
    sd, prior = _session(17, n=9)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1, 2]))
    actual = {k: v.numel() for k, v in stats.d.items()}
    assert actual == calib_cost.acc_elements(9, 2)
    assert calib_cost.desc_dim_general(K, 2) == desc_dim(K)
    d, r, _, _ = describe(stats, prior)
    assert d.shape[1] == calib_cost.desc_dim_general(K, 2)


@pytest.mark.parametrize("variant", ["p0", "p1", "p2", "p3", "p4"])
def test_forward_macs_match_hook_count(variant):
    sd, prior = _session(18, n=9)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1, 2]))
    d, r, _, _ = describe(stats, prior)
    model = _model(variant, desc_dim(K), m=8)
    macs = {"n": 0}

    def hook(mod, inp, out):
        macs["n"] += inp[0].shape[0] * mod.in_features * mod.out_features
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.register_forward_hook(hook)
    with torch.no_grad():
        model(d, r, prior.float())
    rep = calib_cost.forward_macs(variant, 9, K, 2, m=8)
    lin = macs["n"]
    extra = 0
    if variant in ("p3", "p4"):                  # non-Linear matmuls counted analytically
        extra = 9 * 8 * 64 * 2 + 9 * 8 * 32 + 9 * 8 * 32
    if variant == "p2":
        extra = 9 * 32
    assert rep["forward_macs"] == lin + extra


def test_cost_report_is_consistent():
    rep = calib_cost.full_report(26, 10, 2, "p4", 8, 100)
    assert rep["total_mac_one_calibration"] == 8 * rep["anytime_stream_mac_per_trial"] + rep["anytime_refresh_mac"]
    assert rep["acc_mac_per_bin"] >= 26 * 26                       # Gram dominates
    big = calib_cost.full_report(256, 16, 5, "p4", 8, 1250)
    assert big["acc_gram_bytes"] == 256 * 256 * 4
    assert big["alt_latent_acc_mac_per_bin"] < big["acc_mac_per_bin"]


def test_qat_forward_equals_ptq_and_gradients_flow():
    fd = desc_dim(K)
    model = _model("p4", fd, m=4)
    ptq = cq.quantize_calibrator(model, "tern")
    qat = _model("p4", fd, m=4)
    qat.load_state_dict(model.state_dict())
    cq.enable_qat(qat, "tern")
    d, r, pr = torch.randn(9, fd), torch.randn(9, 6), torch.zeros(K, 9)
    out_q = qat(d, r, pr)
    with torch.no_grad():
        torch.testing.assert_close(out_q, ptq(d, r, pr), atol=1e-5, rtol=1e-4)
    out_q.square().sum().backward()
    for name, p in qat.named_parameters():
        if "gen.4" in name or name in ("u", "a") or "val" in name:
            assert p.grad is not None, name
    assert qat.gen[0].weight.grad.abs().sum() > 0              # STE passes gradients to FP latents
    assert set(qat.state_dict()) == set(model.state_dict())     # checkpoint layout unchanged
