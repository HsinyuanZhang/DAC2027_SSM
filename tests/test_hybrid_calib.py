"""Tests for the hybrid-identity calibrator (covered flag, uncovered simulation, pooling, guards)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ssm_decode.hybrid_calib import (KINDS, HybridCalibrator, covered_flags, electrode_counts,
                                     mask_prior, match_width, simulate_uncovered)
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
    t1 = cal(d, r, prior.float(), cov)
    t2 = cal(d[perm], r[perm], prior[:, perm].float(), cov[perm.numpy()])
    torch.testing.assert_close(t1[:, perm], t2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_cross_unit_dependence_only_for_pooled_nonlocal(kind):
    sd, prior, cov, stats, d, r = _inputs()
    cal = _model(kind, desc_dim(K))
    base = cal(d, r, prior.float(), cov)
    d2 = d.clone()
    d2[0] += 3.0                                           # perturb unit 0 only
    pert = cal(d2, r, prior.float(), cov)
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
        torch.testing.assert_close(cal(d, r, prior.float(), cov), prior.float())


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
