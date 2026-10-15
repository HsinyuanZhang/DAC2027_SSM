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
        a, b = wide.d[k], exact.d[k]
        if k == "sphix":                                          # row 0 is shared with sx
            a, b = a[1:], b[1:]
        torch.testing.assert_close(a, b, atol=1e-5 * float(exact.d[k].abs().max()) + 1e-9, rtol=1e-4)
    steps8 = cq.acc_steps([exact], 8)
    narrow = cq.quantized_stats(sd.bank, idx, steps8, 8)
    for k, v in narrow.d.items():
        grid = v / steps8[k]
        torch.testing.assert_close(grid, grid.round())
        assert float(v.abs().max()) <= steps8[k] * 127 + 1e-9
    assert (narrow.d["gram"] - exact.d["gram"]).abs().max() > 0 or steps8["gram"] <= 1


# --------------------------------------------- per-add (per-bin) accumulator model ----


def _naive_sat_acc(sd, idx, key, step, bits):
    """Slow reference: python loop over bins, round each increment, saturating add."""
    from ssm_decode.population_calib import segment_sums
    ang = cq.ang_bins_of(sd)
    q = cq.qmax(bits)
    acc = None
    for j in idx:
        a, b = sd.trials[int(j)]
        prev = None
        for t in range(a + 1, b + 1):
            cur = segment_sums(sd.x[a:t], sd.y_norm[a:t], ang[a:t], sd.valid[a:t])[key]
            inc = cur if prev is None else cur - prev
            prev = cur
            r = torch.round(inc / step)
            acc = r if acc is None else (acc + r).clamp(-q, q)
    return acc * step


def test_bin_increments_sum_to_segment_sums():
    from ssm_decode.population_calib import segment_sums
    sd, _ = _session(21)
    ang = cq.ang_bins_of(sd)
    for j in (0, 3, 6):                                     # trial 3 has an invalid gap
        a, b = sd.trials[j]
        ref = segment_sums(sd.x[a:b], sd.y_norm[a:b], ang[a:b], sd.valid[a:b])
        tot = None
        for t0 in range(0, b - a, 17):                      # chunking must not matter
            inc = cq.bin_increments(sd.x[a:b], sd.y_norm[a:b], ang[a:b], sd.valid[a:b],
                                    t0, min(t0 + 17, b - a))
            part = {k: v.sum(0) for k, v in inc.items()}
            tot = part if tot is None else {k: tot[k] + part[k] for k in tot}
        assert set(tot) == set(ref)
        for k in ref:
            torch.testing.assert_close(tot[k], ref[k], atol=1e-9, rtol=1e-9, msg=k)


def test_perbin_equals_exact_sum_when_step_one_and_no_saturation():
    sd, _ = _session(22)
    idx = np.array([0, 2, 3, 6, 8])
    exact = PopStats.from_trials(sd.bank, idx)
    int_keys = ("sx", "gram", "ac_xx", "ac_a", "ac_b", "sdir")
    steps = {k: 1.0 for k in exact.d}
    st, info = cq.perbin_stats(sd, idx, steps, 31, int_keys, return_info=True)
    for k in int_keys:
        torch.testing.assert_close(st.d[k], exact.d[k], atol=0, rtol=0, msg=k)
    assert sum(info["sat_events"].values()) == 0
    # a very fine step reproduces every data key to within half a step per bin
    fine = {k: 2.0 ** -30 for k in exact.d}
    st2 = cq.perbin_stats(sd, idx, fine, 60)
    n_bins = float(exact.d["n"])
    for k in cq.ALL_DATA_KEYS:
        tol = 1e-5 if k.startswith("q_") else n_bins * 2.0 ** -30
        torch.testing.assert_close(st2.d[k], exact.d[k], atol=tol + 1e-9, rtol=1e-6, msg=k)


def test_perbin_matches_naive_loop_and_rounds_every_add():
    sd, _ = _session(23)
    idx = np.array([1, 2, 4])
    exact = PopStats.from_trials(sd.bank, idx)
    for key in ("sx", "gram", "ac_xx", "syx", "sdir"):
        step = 4.0 if key in ("sx", "gram", "ac_xx", "sdir") else 0.5   # step > increments: lossy
        st = cq.perbin_stats(sd, idx, {k: step for k in exact.d}, 31, (key,))
        ref = _naive_sat_acc(sd, idx, key, step, 31)
        torch.testing.assert_close(st.d[key], ref, atol=1e-9, rtol=0, msg=key)
    # per-add rounding is NOT the same as rounding the per-trial sums (the old model)
    step = {k: 4.0 for k in exact.d}
    per_add = cq.perbin_stats(sd, idx, step, 31, ("sx",)).d["sx"]
    two = cq.twolevel_stats(sd.bank, idx, step, 31, ("sx",)).d["sx"]
    assert (per_add - exact.d["sx"]).abs().sum() > (two - exact.d["sx"]).abs().sum()


def test_perbin_saturation_clamps_each_add_and_counts_events():
    sd, _ = _session(24)
    idx = np.array([0, 1, 2, 3])
    exact = PopStats.from_trials(sd.bank, idx)
    bits, step = 6, 1.0                                     # qmax = 31 << per-unit sums
    steps = {k: step for k in exact.d}
    st, info = cq.perbin_stats(sd, idx, steps, bits, ("sx", "gram"), return_info=True)
    q = cq.qmax(bits) * step
    assert float(st.d["sx"].abs().max()) <= q and float(st.d["gram"].abs().max()) <= q
    assert float(exact.d["sx"].max()) > q                   # the test is non-vacuous
    assert info["sat_events"]["sx"] > 0
    torch.testing.assert_close(st.d["sx"], _naive_sat_acc(sd, idx, "sx", step, bits), atol=0, rtol=0)
    assert float(st.d["sx"].max()) == q                     # stuck at the rail, no wrap-around


def test_counters_are_never_quantised():
    sd, _ = _session(25)
    idx = np.array([0, 2, 3, 5])
    exact = PopStats.from_trials(sd.bank, idx)
    steps = cq.acc_steps([exact], 8)
    for mode in ("perbin", "twolevel"):
        for keys in (None, cq.ACC_GROUPS["norm"] + cq.ACC_GROUPS["ac"] + cq.ACC_GROUPS["cross"]
                     + cq.ACC_GROUPS["dir"]):             # legacy groups list counters: ignored
            st = (cq.perbin_stats(sd, idx, steps, 8, keys) if mode == "perbin"
                  else cq.twolevel_stats(sd.bank, idx, steps, 8, keys))
            for k in cq.COUNTER_KEYS:
                torch.testing.assert_close(st.d[k], exact.d[k], atol=0, rtol=0, msg=k)
            assert float(st.d["sphi"][0]) == float(exact.d["n"])
            assert float(st.d["sphiphi"][0, 0]) == float(exact.d["n"])
            assert float(st.d["n"]) > 1
    # the descriptor of an 8-bit accumulator is finite (no n=0 clamp, no NaN correlation)
    _, prior = _session(25)
    st = cq.perbin_stats(sd, idx, steps, 8)
    d, r, _, _ = describe(st, prior)
    assert torch.isfinite(d).all() and torch.isfinite(r).all()
    assert cq.counter_bits(33, 2741) == 17 and cq.counter_bits(33, 1250) == 16


def test_derived_keys_equal_exact_sums_and_get_no_accumulator():
    sd, _ = _session(28)
    idx = np.array([0, 2, 3, 6])
    exact = PopStats.from_trials(sd.bank, idx)
    d = {k: v.clone() for k, v in exact.d.items()}
    for k in ("sphi", "sphiphi"):
        d[k] = torch.zeros_like(d[k])
    d["sphix"] = exact.d["sphix"].clone()
    d["sphix"][0] = 0
    cq._set_derived(d)
    for k in ("sphi", "sphiphi", "sphix"):
        torch.testing.assert_close(d[k], exact.d[k], atol=1e-9, rtol=1e-9, msg=k)
    assert "sphi" not in cq.ALL_DATA_KEYS and "sphiphi" not in cq.ALL_DATA_KEYS
    steps = cq.acc_steps([exact], 8)
    assert steps["sphix"] == cq.fixed_step(float(exact.d["sphix"][1:].abs().max()), 8)
    st, info = cq.perbin_stats(sd, idx, steps, 8, return_info=True)
    assert "sphi" not in info["sat_events"] and "sphiphi" not in info["sat_events"]
    torch.testing.assert_close(st.d["sphix"][0], st.d["sx"])           # shared with the sx register


def test_twolevel_equals_legacy_per_trial_rounding():
    sd, _ = _session(26)
    idx = np.array([0, 1, 4, 7])
    exact = PopStats.from_trials(sd.bank, idx)
    steps = cq.acc_steps([exact], 10)
    keys = cq.ALL_DATA_KEYS
    new = cq.twolevel_stats(sd.bank, idx, steps, 10, keys, alias_counters=False)
    old = cq.quantized_stats(sd.bank, idx, steps, 10, keys)
    for k in keys:
        torch.testing.assert_close(new.d[k], old.d[k], atol=0, rtol=0, msg=k)
    for k in cq.COUNTER_KEYS:
        torch.testing.assert_close(new.d[k], exact.d[k], atol=0, rtol=0)


def test_multi_config_shares_increments_without_changing_results():
    sd, _ = _session(27)
    idx = np.array([0, 3, 5])
    exact = PopStats.from_trials(sd.bank, idx)
    cfgs = {"a": (cq.acc_steps([exact], 12), 12, None), "b": (cq.acc_steps([exact], 16), 16, ("gram", "q_sx2"))}
    multi = cq.perbin_stats_multi(sd, idx, cfgs)
    for name, (steps, bits, keys) in cfgs.items():
        single = cq.perbin_stats(sd, idx, steps, bits, keys)
        for k in single.d:
            torch.testing.assert_close(multi[name][0].d[k], single.d[k], atol=0, rtol=0, msg=(name, k))


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


# ------------------------------------------------------- review-fix: cost model ----


def test_cost_model_review_terms():
    base = calib_cost.full_report(26, 10, 2, "p4", 8, 220, acc_bits=16, desc_bits=16, table_bits=8)
    new = calib_cost.full_report(26, 10, 2, "p4", 8, 220, acc_bits=16, desc_bits=16, table_bits=8,
                                 cnt_bits=17, stage_bits=32, x_bits=8, y_bits=16)
    legacy_keys = ("acc_bytes", "descriptor_bytes", "table_bytes_double", "total_mac_one_calibration")
    # counters at 17 bit instead of 16 -> only acc_bytes differs, by 17 counter elements
    assert calib_cost.counter_elements(26, 2) == 1 + 3 + 5 + 8
    assert new["acc_bytes"] - base["acc_bytes"] == calib_cost.counter_elements(26, 2) * (17 - 16) / 8
    assert new["total_mac_one_calibration"] == base["total_mac_one_calibration"]
    assert base["staging_bytes"] == 0 and new["staging_bytes"] > 0
    # staging copy: data keys without q_* plus counters
    el = calib_cost.acc_elements(26, 2)
    data = sum(v for k, v in el.items() if k not in cq.COUNTER_KEYS and not k.startswith("q_"))
    assert new["staging_bytes"] == (data * 32 + calib_cost.counter_elements(26, 2) * 17) / 8
    # scratch for q_*: sx, syx lag 0, sy lag 0 (+ n)
    assert new["scratch_elements"] == 26 + 2 * 26 + 2 + 1
    # delay lines: 5 x vectors (largest AC lag), 2 y vectors (negative cross lags)
    assert new["delay_x_elements"] == 5 * 26 and new["delay_y_elements"] == 2 * 2
    assert new["total_sram_bytes"] > base["acc_bytes"] + base["descriptor_bytes"]
    # P4 second pass: N x H buffer or recompute phi (Fd H + 2 H^2 MACs per unit)
    fd = calib_cost.desc_dim_general(10, 2)
    assert new["phi_recompute_mac_per_unit"] == fd * 64 + 2 * 64 * 64
    assert new["phi_recompute_extra_mac"] == 26 * new["phi_recompute_mac_per_unit"]
    assert new["phi_buffer_bytes"] == 26 * 64 * 16 / 8
    p0 = calib_cost.full_report(26, 10, 2, "p0", 8, 220)
    assert p0["phi_buffer_bytes"] == 0 and p0["phi_recompute_extra_mac"] == 0
    # per-group widths are accepted
    g = calib_cost.sram_bytes(26, 10, 2, "p0", acc_bits={"norm": 20, "ac": 12, "cross": 12, "dir": 16, "quad": 12})
    assert g["acc_bytes"] > calib_cost.sram_bytes(26, 10, 2, "p0", acc_bits=12)["acc_bytes"]
    del legacy_keys


def test_cost_model_derived_dir_removes_redundant_accumulators():
    base = calib_cost.acc_per_bin(26, 2)
    der = calib_cost.acc_per_bin(26, 2, derived_dir=True)
    assert base["mac"] - der["mac"] == 9 + 26          # phi phi^T and the constant row of phi x^T
    assert base["add"] - der["add"] == 3               # sphi
    a = calib_cost.sram_bytes(26, 10, 2, "p0", acc_bits=16, cnt_bits=17)
    b = calib_cost.sram_bytes(26, 10, 2, "p0", acc_bits=16, cnt_bits=17, derived_dir=True)
    assert a["acc_elements"] - b["acc_elements"] == 3 + 9 + 26
    # the cost model's counter set is the one the quantiser keeps exact
    assert set(cq.COUNTER_KEYS) <= set(calib_cost.acc_elements(26, 2))


@pytest.mark.parametrize("variant", ["p0", "p2", "p4"])
def test_fixed_weight_share_excludes_activation_products(variant):
    rep = calib_cost.forward_macs(variant, 26, 10, 2, m=8)
    if variant == "p0":
        assert rep["actact_macs"] == 0 and rep["fixed_weight_macs"] == rep["forward_macs"]
    if variant == "p4":                       # beta @ z (m dctx per unit) + softmax-weighted pooling
        assert rep["actact_macs"] == 26 * 8 * 32 * 2
    assert rep["fixed_weight_macs"] + rep["actact_macs"] == rep["forward_macs"]
    full = calib_cost.full_report(26, 10, 2, variant, 8, 220)
    assert full["fixed_weight_fraction"] <= full["forward_share_incl_actact"] + 1e-12
    assert full["fixed_weight_macs"] == rep["fixed_weight_macs"]


def test_exp2_lut_output_range_matches_documentation():
    lut = cq.Exp2LUT(256)
    top = float(lut.exp(torch.zeros(1, dtype=torch.float64)))
    assert 1.0 < top <= 2 ** (0.5 / 256) + 2 ** -16        # exp(0) -> 1.00135, not 1
    assert "1.00135" in cq.Exp2LUT.__doc__


def test_prior_mode_assertion_is_backward_compatible():
    from types import SimpleNamespace
    from scripts import run_population_calib as rpc
    aligned, gm = SimpleNamespace(prior_mode="aligned"), SimpleNamespace(prior_mode="globalmean")
    assert rpc.check_prior_mode(aligned, {"variant": "p0"})            # old checkpoint: no key == aligned
    assert rpc.check_prior_mode(gm, {"prior_mode": "globalmean"})
    with pytest.raises(AssertionError):
        rpc.check_prior_mode(aligned, {"prior_mode": "globalmean"})    # identity-free model, aligned prior
    with pytest.raises(AssertionError):
        rpc.check_prior_mode(gm, {"variant": "p0"})                    # aligned model, identity-free prior
    assert not rpc.check_prior_mode(aligned, {"prior_mode": "zero"}, strict=False)
