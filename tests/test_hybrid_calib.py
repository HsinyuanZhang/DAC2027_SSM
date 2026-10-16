"""Tests for the hybrid-identity calibrator (covered flag, uncovered simulation, pooling, guards)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ssm_decode.hybrid_calib import (KINDS, HybridCalibrator, covered_flags, electrode_counts,
                                     mask_prior, match_width, norm_entropy, pair_corr,
                                     simulate_uncovered)
from ssm_decode.population_calib import corr_columns
from ssm_decode.population_calib import (PopStats, SessionData, desc_dim, describe,
                                         electrode_mean_table)

K = 4


def _session(seed=0, n=9, n_trials=10, trial_len=80, electrodes=None):
    rng = np.random.default_rng(seed)
    t = n_trials * trial_len + 30
    latent = rng.standard_normal((t, 3))
    behavior = (latent[:, :2] * 0.7).astype(np.float32)
    rate = np.exp(0.4 * latent @ rng.standard_normal((3, n)) - 0.5)
    neural = rng.poisson(rate).astype(np.float32)
    bounds = tuple((15 + i * trial_len, 15 + (i + 1) * trial_len) for i in range(n_trials))
    mask = np.zeros(t, dtype=bool)
    for a, b in bounds:
        mask[a:b] = True
    el = np.arange(n) if electrodes is None else np.asarray(electrodes)
    rec = SimpleNamespace(neural=neural, behavior=behavior, eval_mask=mask, trial_bounds=bounds,
                          unit_to_electrode=el, session=f"syn{seed}")
    sd = SessionData(rec, (np.zeros(2), np.ones(2)), torch.device("cpu"))
    prior = torch.from_numpy(rng.standard_normal((K, n)) * 0.1)
    return sd, prior


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def _model(kind, fd, **kw):
    return _randomize(HybridCalibrator(kind, K, fd, **kw))


def _inputs(seed=1, n=9):
    sd, prior = _session(seed, n=n)
    cov = np.array([True, True, False, True, False, True, True, False, True][:n])
    prior = mask_prior(prior, cov)
    stats = PopStats.from_trials(sd.bank, np.array([0, 2, 3, 5, 7]))
    d, r, mean, std = describe(stats, prior)
    return sd, prior, cov, stats, d, r


def _pair(stats, perm=None):
    c, n = pair_corr(stats)
    if perm is not None:
        c = c[perm][:, perm]
    return c, n


# ------------------------------------------------------- covered flag ----

def test_covered_flag_uses_source_electrode_metadata_only():
    src = [np.array([0, 1, 2, 5]), np.array([1, 2, 7])]
    el = np.array([0, 3, 5, 7, 7, 9])
    np.testing.assert_array_equal(covered_flags(src, el), [True, False, True, True, True, False])
    # leave-one-out: dropping session 1 uncovers electrode 7
    np.testing.assert_array_equal(covered_flags(src, el, exclude=1), [True, False, True, False, False, False])
    # the function signature has no access to neural data / labels / target query
    import inspect
    assert list(inspect.signature(covered_flags).parameters) == ["source_electrodes", "unit_electrodes",
                                                                  "exclude", "width"]


def test_covered_flag_equals_zero_prior_column_of_electrode_aligned_prior():
    rng = np.random.default_rng(0)
    el_src = [rng.integers(0, 20, size=8) for _ in range(4)]
    tables = [rng.standard_normal((K, 8)) + 3.0 for _ in range(4)]       # nonzero columns
    for exclude in (None, 2):
        avg = electrode_mean_table(tables, el_src, width=20, exclude=exclude)
        el_t = rng.integers(0, 20, size=15)
        zero_col = np.abs(avg[:, el_t]).sum(0) == 0
        np.testing.assert_array_equal(covered_flags(el_src, el_t, exclude=exclude, width=20), ~zero_col)
    assert electrode_counts(el_src, 20).sum() == 32


# ------------------------------------------------ uncovered simulation ----

def test_simulate_uncovered_masks_whole_electrodes_and_reaches_fraction():
    rng = np.random.default_rng(3)
    el = np.repeat(np.arange(10), 3)                      # 30 units, 10 electrodes x 3 units
    cov = np.ones(30, dtype=bool)
    fr = []
    for f in (0.0, 0.1, 0.3, 0.5):
        out = simulate_uncovered(rng, el, cov, f)
        unc = ~out
        fr.append(unc.mean())
        assert unc.mean() >= f - 1e-9 and unc.mean() <= f + 3 / 30 + 1e-9
        for e in np.unique(el[unc]):                      # electrode-level masking
            assert unc[el == e].all()
        assert out.any()
    assert fr[0] == 0.0 and fr == sorted(fr)


def test_simulate_uncovered_keeps_natural_uncovered_and_is_never_all_masked():
    rng = np.random.default_rng(0)
    el = np.array([0, 0, 1, 2, 3])
    cov = np.array([False, False, True, True, True])
    out = simulate_uncovered(rng, el, cov, 0.2)           # natural uncovered 0.4 > 0.2 -> unchanged
    np.testing.assert_array_equal(out, cov)
    out = simulate_uncovered(rng, el, cov, 1.0)
    assert out.sum() >= 1 and not out[:2].any()


def test_simulated_fraction_distribution_in_training_range():
    from scripts.run_hybrid_identity import FRAC_MAX
    rng = np.random.default_rng(1)
    el = np.repeat(np.arange(15), 2)
    fr = [(~simulate_uncovered(rng, el, np.ones(30, bool), rng.uniform(0, FRAC_MAX))).mean()
          for _ in range(300)]
    assert min(fr) <= 2 / 30 and max(fr) <= FRAC_MAX + 2 / 30 + 1e-9 and 0.15 < np.mean(fr) < 0.35


def test_mask_prior_zeroes_uncovered_columns_only():
    prior = torch.ones(K, 6)
    cov = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    out = mask_prior(prior, cov)
    assert (out[:, ~cov] == 0).all() and (out[:, cov] == 1).all()


def test_calibrate_rejects_nonzero_prior_on_uncovered_units():
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model("hyb_ds", desc_dim(K))
    bad = prior.clone()
    bad[:, ~cov] = 1.0
    with pytest.raises(ValueError):
        cal.calibrate(stats, bad, cov)


# ------------------------------------------------- permutation / pooling ----

@pytest.mark.parametrize("kind", KINDS)
def test_permutation_equivariance(kind):
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model(kind, desc_dim(K))
    perm = torch.randperm(d.shape[0], generator=torch.Generator().manual_seed(4))
    t1 = cal(d, r, prior.float(), cov, _pair(stats))
    t2 = cal(d[perm], r[perm], prior[:, perm].float(), cov[perm.numpy()], _pair(stats, perm))
    torch.testing.assert_close(t1[:, perm], t2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_network_cross_unit_flow_only_for_pooled_nonlocal(kind):
    """Network-level only: d is perturbed AFTER the descriptors exist, so descriptor-level coupling
    (see test_descriptor_level_coupling_*) is invisible to this test.  Do not read it as 'the -loc /
    wide controls carry no cross-unit information'."""
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model(kind, desc_dim(K))
    pair = _pair(stats)
    base = cal(d, r, prior.float(), cov, pair)
    d2 = d.clone()
    d2[0] += 3.0                                           # perturb unit 0 only
    pert = cal(d2, r, prior.float(), cov, pair)
    others = torch.arange(1, d.shape[0])
    leaks = not torch.allclose(base[:, others], pert[:, others], atol=1e-6)
    pooled_nonlocal = kind != "flat" and not kind.endswith("_loc")
    assert leaks == pooled_nonlocal


def test_hybrid_covered_path_has_no_pooling_context():
    """In the hybrid variants covered units are produced by g_cov only: perturbing an UNCOVERED
    unit's descriptor must not change covered units' columns, in any hybrid variant."""
    sd, prior, cov, stats, d, r = _inputs()
    for kind in ("hyb_ds", "hyb_pma"):
        cal = _model(kind, desc_dim(K))
        base = cal(d, r, prior.float(), cov)
        d2 = d.clone()
        d2[2] += 5.0                                       # unit 2 is uncovered
        pert = cal(d2, r, prior.float(), cov)
        c = torch.as_tensor(cov)
        torch.testing.assert_close(base[:, c], pert[:, c], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(base[:, ~c], pert[:, ~c])


def test_all_variants_condition_covered_units_on_the_population():
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model("all_ds", desc_dim(K))
    base = cal(d, r, prior.float(), cov)
    d2 = d.clone()
    d2[2] += 5.0
    pert = cal(d2, r, prior.float(), cov)
    c = torch.as_tensor(cov)
    assert not torch.allclose(base[:, c], pert[:, c])


def test_zero_init_means_untrained_table_equals_masked_prior():
    sd, prior, cov, stats, d, r = _inputs()
    for kind in KINDS:
        cal = HybridCalibrator(kind, K, desc_dim(K))
        torch.testing.assert_close(cal(d, r, prior.float(), cov, _pair(stats)), prior.float())


def test_flag_changes_output_only_when_used():
    sd, prior, cov, stats, d, r = _inputs()
    with_flag = _model("flat", desc_dim(K), use_flag=True)
    no_flag = _model("flat", desc_dim(K), use_flag=False)
    flipped = ~cov
    assert not torch.allclose(with_flag(d, r, prior.float(), cov), with_flag(d, r, prior.float(), flipped))
    torch.testing.assert_close(no_flag(d, r, prior.float(), cov), no_flag(d, r, prior.float(), flipped))


# ------------------------------------------------------ capacity control ----

def test_local_controls_have_identical_parameter_counts():
    fd = desc_dim(10)
    for a, b in (("hyb_ds", "hyb_ds_loc"), ("hyb_pma", "hyb_pma_loc"),
                 ("all_ds", "all_ds_loc"), ("all_pma", "all_pma_loc")):
        assert HybridCalibrator(a, 10, fd).n_params == HybridCalibrator(b, 10, fd).n_params


def test_wide_controls_match_pooled_parameter_counts_within_one_percent():
    from scripts.run_hybrid_identity import MATCHED_WIDE, VARIANTS, build
    fd = desc_dim(10)
    for name, wide in MATCHED_WIDE.items():
        n_p = build(name, 10, fd).n_params
        n_w = build(wide, 10, fd).n_params
        assert abs(n_p - n_w) / n_p < 0.01, (name, wide, n_p, n_w)
        assert VARIANTS[wide]["kind"] == "flat"
    assert match_width(build("h2d", 10, fd).n_params, 10, fd) == VARIANTS["h1w138"]["hidden"]


# ----------------------------------------------------- no-grad guard ----

@pytest.mark.parametrize("kind", ("flat", "hyb_ds", "hyb_pma", "all_pma"))
def test_target_calibrate_runs_without_grad_and_params_unchanged(kind):
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model(kind, desc_dim(K))
    for p in cal.parameters():
        p.requires_grad_(False)
    before = [p.clone() for p in cal.parameters()]
    n_calls = {"n": 0}
    og, ob = torch.autograd.grad, torch.autograd.backward
    torch.autograd.grad = lambda *a, **k: (n_calls.__setitem__("n", n_calls["n"] + 1), og(*a, **k))[1]
    torch.autograd.backward = lambda *a, **k: (n_calls.__setitem__("n", n_calls["n"] + 1), ob(*a, **k))[1]
    try:
        with torch.enable_grad():                          # even if grad mode is on outside
            res = cal.calibrate(stats, prior, cov)
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob
    assert n_calls["n"] == 0 and not res.table.requires_grad and not res.diagnostic_only
    for a, b in zip(before, cal.parameters()):
        assert torch.equal(a, b)


def test_table_uses_only_the_supplied_trial_statistics():
    """Stats built from B chosen trials only: changing an un-chosen trial must not change the table."""
    sd, prior = _session(5)
    cov = np.ones(sd.n_units, dtype=bool)
    cal = _model("hyb_pma", desc_dim(K))
    idx = np.array([1, 4, 6])
    t1 = cal.calibrate(PopStats.from_trials(sd.bank, idx), prior, cov).table
    sd.bank.t["sx"][8] += 100.0                            # a trial outside the budget
    t2 = cal.calibrate(PopStats.from_trials(sd.bank, idx), prior, cov).table
    torch.testing.assert_close(t1, t2)


# ============================================================ v2 additions ====

def _perturb_gram(stats, i, j, delta=40.0):
    """Copy of ``stats`` with the Gram entries (i, j) and (j, i) changed (diagonal untouched)."""
    d = {k: v.clone() for k, v in stats.d.items()}
    d["gram"][i, j] += delta
    d["gram"][j, i] += delta
    return PopStats(d, stats.n_trials)


def _tables(cal, stats, prior, cov):
    return cal.calibrate(stats, prior, cov).table


def test_descriptor_level_coupling_exists_in_describe():
    """describe(): correlating x_i with z = prior . x (a sum over covered units) couples an uncovered
    unit's descriptor to covered units' second moments.  Perturbing a covered unit's Gram row must
    change an uncovered unit's latent-correlation features (and nothing else about it)."""
    sd, prior, cov, stats, d, r = _inputs()
    iu = int(np.flatnonzero(~cov)[0])
    jc = int(np.flatnonzero(cov)[0])
    d2, r2, _, _ = describe(_perturb_gram(stats, iu, jc), prior)
    cc = corr_columns(K)
    assert (d2[iu, cc] - d[iu, cc]).abs().max() > 1e-3
    other = [c for c in range(d.shape[1]) if c not in cc]
    torch.testing.assert_close(d2[iu][other], d[iu][other])
    # perturbing a COVERED-covered second moment also moves the uncovered unit's corr (through z)
    jc2 = int(np.flatnonzero(cov)[1])
    d3, _, _, _ = describe(_perturb_gram(stats, jc, jc2), prior)
    assert (d3[iu, cc] - d[iu, cc]).abs().max() > 1e-4


@pytest.mark.parametrize("kind", ("flat", "hyb_ds_loc", "hyb_pma_loc"))
def test_loc_and_flat_controls_receive_descriptor_coupling_unless_cut(kind):
    """End to end (stats -> table): without ``cut_coupling`` the uncovered columns of the 'no cross-unit
    flow' controls DO depend on covered units' Gram entries; with it they do not."""
    sd, prior, cov, stats, d, r = _inputs()
    iu = int(np.flatnonzero(~cov)[0])
    jc = int(np.flatnonzero(cov)[0])
    st2 = _perturb_gram(stats, iu, jc)
    unc = torch.as_tensor(~cov)
    for cut in (False, True):
        cal = _model(kind, desc_dim(K), cut_coupling=cut)
        t1, t2 = _tables(cal, stats, prior, cov), _tables(cal, st2, prior, cov)
        changed = not torch.allclose(t1[:, unc], t2[:, unc], atol=1e-6)
        assert changed == (not cut), (kind, cut)
    # the in-network-only test cannot see it: perturbing d AFTER describe leaves the -loc columns alone
    cal = _model(kind, desc_dim(K))
    d2 = d.clone()
    d2[int(np.flatnonzero(cov)[0])] += 3.0
    others = [i for i in range(d.shape[0]) if i != int(np.flatnonzero(cov)[0])]
    t1 = cal(d, r, prior.float(), cov)
    t2 = cal(d2, r, prior.float(), cov)
    assert torch.allclose(t1[:, others], t2[:, others], atol=1e-6)


def test_cut_coupling_zeroes_exactly_the_uncovered_corr_features():
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model("flat", desc_dim(K), cut_coupling=True)
    cc = corr_columns(K)
    _, dtf, _ = cal._features(d, r, cov)
    unc = torch.as_tensor(~cov)
    assert (dtf[unc][:, cc] == 0).all()                     # uncovered: corr features exactly zero
    assert (dtf[~unc][:, cc] != 0).any()                    # covered: untouched
    other = [c for c in range(d.shape[1]) if c not in cc]
    nocut = _model("flat", desc_dim(K), cut_coupling=False)
    _, dtf0, _ = nocut._features(d, r, cov)
    torch.testing.assert_close(dtf[:, other], dtf0[:, other])          # all other columns identical
    torch.testing.assert_close(dtf[~unc], dtf0[~unc])                   # covered rows identical
    # the output of an uncovered unit is invariant to its own raw corr features
    d2 = d.clone()
    d2[np.flatnonzero(~cov)[0], cc[0]] += 5.0
    torch.testing.assert_close(cal(d, r, prior.float(), cov), cal(d2, r, prior.float(), cov))
    # prior column of uncovered units is already 0 (zero column) -> the cut leaves no population channel
    assert (prior[:, unc] == 0).all()


def test_set_normalizers_uses_covered_rows_for_corr_columns_when_cut():
    sd, prior, cov, stats, d, r = _inputs()
    cal = HybridCalibrator("flat", K, desc_dim(K), cut_coupling=True)
    dz = d.clone()
    dz[torch.as_tensor(~cov)[:, None] & cal.corr_mask[None, :]] = 0.0
    cal.set_normalizers(dz, r, cov)
    cc = corr_columns(K)
    torch.testing.assert_close(cal.d_mu[cc], dz[torch.as_tensor(cov)][:, cc].mean(0))


def test_xatt_permutation_equivariance_over_covered_units():
    """Permuting only the covered units permutes their columns and leaves every uncovered column
    unchanged (softmax over covered keys is a set operation)."""
    sd, prior, cov, stats, d, r = _inputs()
    for kind in ("hyb_xatt", "hyb_xatt_np"):
        cal = _model(kind, desc_dim(K), cut_coupling=True)
        ic = np.flatnonzero(cov)
        perm = np.arange(len(cov))
        perm[ic] = ic[::-1]
        perm_t = torch.as_tensor(perm)
        t1 = cal(d, r, prior.float(), cov, _pair(stats))
        t2 = cal(d[perm_t], r[perm_t], prior[:, perm_t].float(), cov[perm], _pair(stats, perm_t))
        torch.testing.assert_close(t1[:, perm_t], t2, atol=1e-5, rtol=1e-5)
        unc = torch.as_tensor(~cov)
        torch.testing.assert_close(t1[:, unc], t2[:, unc], atol=1e-5, rtol=1e-5)


def test_xatt_control_has_no_pairwise_term_and_xatt_does():
    """Perturb ONLY the (uncovered i, covered j) Gram entries: hyb_xatt_np (cut) is unchanged,
    hyb_xatt changes.  Without cut, the np control would change through the descriptor corr features
    (checked for completeness)."""
    sd, prior, cov, stats, d, r = _inputs()
    iu = int(np.flatnonzero(~cov)[0])
    jc = int(np.flatnonzero(cov)[0])
    st2 = _perturb_gram(stats, iu, jc, delta=60.0)
    assert HybridCalibrator("hyb_xatt", K, desc_dim(K)).n_params == \
        HybridCalibrator("hyb_xatt_np", K, desc_dim(K)).n_params
    for kind, expect_change in (("hyb_xatt", True), ("hyb_xatt_np", False)):
        cal = _model(kind, desc_dim(K), cut_coupling=True)
        t1, t2 = _tables(cal, stats, prior, cov), _tables(cal, st2, prior, cov)
        assert (not torch.allclose(t1, t2, atol=1e-6)) == expect_change, kind
    cal = _model("hyb_xatt_np", desc_dim(K), cut_coupling=False)
    assert not torch.allclose(_tables(cal, stats, prior, cov), _tables(cal, st2, prior, cov), atol=1e-6)
    # the attention input is exactly the key scalar in the control: changing corr_uk is a no-op
    cal = _model("hyb_xatt_np", desc_dim(K), cut_coupling=True)
    c, n = pair_corr(stats)
    c2 = c.clone()
    c2[iu, :] = 0.9
    t1 = cal(d, r, prior.float(), cov, (c, n))
    t2 = cal(d, r, prior.float(), cov, (c2, n))
    torch.testing.assert_close(t1, t2)


def test_xatt_softmax_is_over_covered_units_and_matches_online_softmax():
    from ssm_decode.population_calib import online_softmax_pool
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model("hyb_xatt", desc_dim(K), cut_coupling=True)
    _, dtf, _ = cal._features(d, r, cov)
    iu, ic = torch.nonzero(torch.as_tensor(~cov)).squeeze(-1), torch.nonzero(torch.as_tensor(cov)).squeeze(-1)
    c, n = pair_corr(stats)
    ctx, abar, attn = cal.attn(dtf[iu], dtf[ic], prior.T[ic].float(), c[iu][:, ic], torch.zeros(len(ic)), n)
    torch.testing.assert_close(attn.sum(1), torch.ones(len(iu), cal.attn.heads))
    # chunked online softmax over covered keys reproduces the weighted prior average for head 0, unit 0
    pri = prior.T[ic].float()
    sc = torch.log(attn[0, :, 0])[:, None]                  # log weights as scores
    pooled = online_softmax_pool(sc, pri, chunk=2)[0]
    torch.testing.assert_close(pooled, (attn[0, :, 0][:, None] * pri).sum(0), atol=1e-5, rtol=1e-5)


def test_norm_entropy_and_pma_collapse_monitor():
    uni = torch.full((5, 8), 1 / 8)
    one = torch.zeros(5, 8)
    one[:, 0] = 1.0
    assert abs(float(norm_entropy(uni, 1).mean()) - 1.0) < 1e-6
    assert float(norm_entropy(one, 1).mean()) < 1e-3
    sd, prior, cov, stats, d, r = _inputs()
    n = d.shape[0]
    # collapsed PMA (u = a = 0): uniform pooling and routing -> entropies 1, max weight x N = 1
    cal = _model("hyb_pma", desc_dim(K), pma_fix=True, cut_coupling=True)
    with torch.no_grad():
        cal.u.zero_(), cal.a.zero_()
        cal.rel[-1].bias.fill_(50.0)                         # lam -> 1 so the log-lam score is flat
    dg = cal.diagnostics(d, r, prior.float(), cov)
    assert dg["beta_entropy"] > 0.999 and dg["pool_entropy"] > 0.999
    assert abs(dg["pool_maxN"] - 1.0) < 1e-2
    # sharp seeds: low entropy and a max weight well above uniform
    with torch.no_grad():
        cal.u.copy_(torch.randn_like(cal.u) * 40.0)
        cal.a.copy_(torch.randn_like(cal.a) * 40.0)
    dg2 = cal.diagnostics(d, r, prior.float(), cov)
    assert dg2["beta_entropy"] < 0.9 and dg2["pool_maxN"] > 1.5 and dg2["ctx_std"] >= 0
    # local control and non-PMA kinds report nothing for PMA; xatt reports attention statistics
    assert _model("hyb_pma_loc", desc_dim(K), pma_fix=True).diagnostics(d, r, prior.float(), cov) == {}
    dx = _model("hyb_xatt", desc_dim(K), cut_coupling=True).diagnostics(d, r, prior.float(), cov, _pair(stats))
    assert 0.0 < dx["attn_entropy"] <= 1.0 and dx["attn_maxN"] >= 1.0 - 1e-6


def test_pma_fix_changes_init_scale_and_adds_temperatures_only():
    a = HybridCalibrator("hyb_pma", K, desc_dim(K), pma_fix=False)
    b = HybridCalibrator("hyb_pma", K, desc_dim(K), pma_fix=True)
    assert b.n_params == a.n_params + 2
    assert HybridCalibrator("hyb_pma_loc", K, desc_dim(K), pma_fix=True).n_params == b.n_params
    assert b.u.std() > 5 * a.u.std() * 0.5                   # unit-scale seeds vs 0.1-scale seeds
    from ssm_decode.hybrid_calib import no_decay_names
    names = [n for n, _ in b.named_parameters() if n in no_decay_names()]
    assert sorted(names) == ["a", "log_tb", "log_ts", "u"]


def test_pair_norm_variants_keep_the_control_property_and_equivariance():
    sd, prior, cov, stats, d, r = _inputs()
    iu = int(np.flatnonzero(~cov)[0])
    jc = int(np.flatnonzero(cov)[0])
    st2 = _perturb_gram(stats, iu, jc, delta=60.0)
    assert HybridCalibrator("hyb_xatt", K, desc_dim(K), pair_norm=True).n_params == \
        HybridCalibrator("hyb_xatt_np", K, desc_dim(K), pair_norm=True).n_params == \
        HybridCalibrator("hyb_xatt", K, desc_dim(K)).n_params
    for kind, expect_change in (("hyb_xatt", True), ("hyb_xatt_np", False)):
        cal = _model(kind, desc_dim(K), cut_coupling=True, pair_norm=True)
        t1, t2 = _tables(cal, stats, prior, cov), _tables(cal, st2, prior, cov)
        assert (not torch.allclose(t1, t2, atol=1e-6)) == expect_change, kind
        perm = torch.randperm(d.shape[0], generator=torch.Generator().manual_seed(7))
        a = cal(d, r, prior.float(), cov, _pair(stats))
        b = cal(d[perm], r[perm], prior[:, perm].float(), cov[perm.numpy()], _pair(stats, perm))
        torch.testing.assert_close(a[:, perm], b, atol=1e-5, rtol=1e-5)
