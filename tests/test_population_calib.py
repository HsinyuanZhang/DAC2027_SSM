"""Tests for population-conditioned static calibration (feasibility P1, line A)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ssm_decode.population_calib import (PopStats, PopulationCalibrator, SessionData,
                                         VARIANTS, assert_candidate, describe,
                                         desc_dim, electrode_mean_table, norm_stats,
                                         online_softmax_pool, select_candidates)

K = 4


def _session(seed=0, n=9, n_trials=10, trial_len=80):
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
    mask[bounds[3][0]:bounds[3][0] + 10] = False        # interior invalid gap
    rec = SimpleNamespace(neural=neural, behavior=behavior, eval_mask=mask,
                          trial_bounds=bounds, unit_to_electrode=np.arange(n),
                          session=f"syn{seed}")
    sd = SessionData(rec, (np.zeros(2), np.ones(2)), torch.device("cpu"))
    prior = torch.from_numpy(rng.standard_normal((K, n)) * 0.1)
    return sd, prior


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def _model(variant, fd, **kw):
    return _randomize(PopulationCalibrator(variant, K, fd, **kw))


def test_streamed_equals_batch_descriptors():
    sd, prior = _session(1)
    idx = np.array([0, 2, 3, 5, 7, 8])
    batch = PopStats.from_trials(sd.bank, idx)
    streamed = PopStats.from_trials(sd.bank, idx[:1])
    for lo in range(1, len(idx), 2):                     # chunked arrival
        streamed = streamed + PopStats.from_trials(sd.bank, idx[lo:lo + 2])
    for k in batch.d:
        torch.testing.assert_close(batch.d[k], streamed.d[k], atol=1e-9, rtol=1e-9)
    d1, r1, m1, s1 = describe(batch, prior)
    d2, r2, m2, s2 = describe(streamed, prior)
    torch.testing.assert_close(d1, d2)
    torch.testing.assert_close(r1, r2)
    torch.testing.assert_close(m1, m2)


def test_descriptor_matches_direct_numpy_reference():
    """Independent computation from concatenated labeled bins."""
    sd, prior = _session(2)
    idx = np.array([1, 4, 6])
    stats = PopStats.from_trials(sd.bank, idx)
    d, r, mean, std = describe(stats, prior)
    bins = np.concatenate([np.arange(*sd.trials[i])[sd.valid_np[slice(*sd.trials[i])]]
                           for i in idx])
    x = sd.record.neural[bins].astype(np.float64)
    np.testing.assert_allclose(mean.numpy(), x.mean(0), atol=1e-10)
    np.testing.assert_allclose(std.numpy(), x.std(0), atol=1e-10)
    # log_rate feature (column 0) and lag-1 autocorrelation (column 2)
    np.testing.assert_allclose(d[:, 0].numpy(), np.log(x.mean(0) + 1e-2), atol=1e-5)
    # lag-1 autocorrelation inside trials, valid pairs only
    num, cnt = np.zeros(x.shape[1]), 0
    m, sdv = x.mean(0), x.std(0)
    for i in idx:
        a, b = sd.trials[i]
        v = sd.valid_np[a:b]
        xs = sd.record.neural[a:b].astype(np.float64)
        pair = v[1:] & v[:-1]
        num += (((xs[1:] - m) * (xs[:-1] - m))[pair]).sum(0)
        cnt += pair.sum()
    np.testing.assert_allclose(d[:, 2].numpy(), np.clip(num / cnt / sdv ** 2, -1, 1), atol=1e-5)
    assert float(stats.d["n"]) == len(bins)
    assert d.shape[1] == desc_dim(K)


def test_streamed_descriptors_do_not_use_other_trials():
    """Trials outside the chosen set cannot influence the descriptors (H4)."""
    sd, prior = _session(3)
    idx = np.array([2, 5])
    d1 = describe(PopStats.from_trials(sd.bank, idx), prior)[0]
    sd.bank.t["sx"][7] += 1e3                             # corrupt an unused trial
    d2 = describe(PopStats.from_trials(sd.bank, idx), prior)[0]
    torch.testing.assert_close(d1, d2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_unit_permutation_equivariance(variant):
    sd, prior = _session(4)
    idx = np.array([0, 3, 4, 9])
    stats = PopStats.from_trials(sd.bank, idx)
    model = _model(variant, desc_dim(K), m=4)
    perm = np.random.default_rng(0).permutation(sd.n_units)
    with torch.no_grad():
        d, r, _, _ = describe(stats, prior)
        t_a = model(d, r, prior.float())
        stats_p = stats.select_units(perm)
        d_p, r_p, _, _ = describe(stats_p, prior[:, perm])
        t_b = model(d_p, r_p, prior[:, perm].float())
    assert t_a.abs().sum() > 0
    torch.testing.assert_close(t_a[:, perm], t_b, atol=1e-5, rtol=1e-4)
    # table columns must differ across units, otherwise the test is vacuous
    assert (t_a[:, 0] - t_a[:, 1]).abs().sum() > 1e-4


@pytest.mark.parametrize("variant", VARIANTS)
def test_target_side_no_grad_guard(variant):
    sd, prior = _session(5)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1, 2]))
    model = _model(variant, desc_dim(K), m=4)
    seen = []
    model.register_forward_hook(lambda mod, i, o: seen.append(torch.is_grad_enabled()))
    with torch.enable_grad():                             # caller has grad ON
        res = model.calibrate(stats, prior)
    assert seen == [False]                                # forward ran without autograd
    assert not res.table.requires_grad and not res.grad_enabled_inside
    assert all(p.grad is None for p in model.parameters())
    # the training path DOES build a graph (so the guard above can fail)
    d, r, _, _ = describe(stats, prior)
    train_out = model(d, r, prior.float())
    assert train_out.requires_grad


def test_shrinkage_reduces_to_p0_when_lambda_is_one():
    sd, prior = _session(6)
    stats = PopStats.from_trials(sd.bank, np.array([0, 4, 6]))
    fd = desc_dim(K)
    p0 = _model("p0", fd)
    p1 = _model("p1", fd, )
    p1.gen.load_state_dict(p0.gen.state_dict())
    for name in ("d_mu", "d_sd", "r_mu", "r_sd"):
        getattr(p1, name).copy_(getattr(p0, name))
    p1.lam_override = 1.0
    d, r, _, _ = describe(stats, prior)
    with torch.no_grad():
        a = p0(d, r, prior.float())
        b = p1(d, r, prior.float())
        torch.testing.assert_close(a, b)
        p1.lam_override = 0.5                             # and it is not vacuous
        c = p1(d, r, prior.float())
    assert (a - c).abs().max() > 1e-4


def test_shrinkage_lambda_zero_ignores_unit_descriptor():
    sd, prior = _session(7)
    stats = PopStats.from_trials(sd.bank, np.array([1, 2]))
    p1 = _model("p1", desc_dim(K))
    p1.lam_override = 0.0
    d, r, _, _ = describe(stats, prior)
    with torch.no_grad():
        out = p1(d, r, prior.float())
    # every unit sees the same shrunk descriptor mu0 -> identical residual columns
    resid = out - prior.float()
    torch.testing.assert_close(resid[:, 0], resid[:, 1])


def test_oracle_cannot_be_selected_as_candidate():
    sd, prior = _session(8)
    model = _model("p2", desc_dim(K))
    stats = PopStats.from_trials(sd.bank, np.arange(sd.bank.n_trials))
    res = model.calibrate(stats, prior, oracle=True)
    assert res.diagnostic_only
    with pytest.raises(ValueError):
        assert_candidate(res)
    ok = model.calibrate(stats, prior)
    assert assert_candidate(ok) is ok
    rows = [{"variant": "p_oracle", "val_score": 0.99, "diagnostic_only": True},
            {"variant": "p_oracle_half", "val_score": 0.98},
            {"variant": "p_prior", "val_score": 0.97},
            {"variant": "p2", "val_score": 0.30}, {"variant": "p1", "val_score": 0.20}]
    assert select_candidates(rows)["variant"] == "p2"
    with pytest.raises(ValueError):
        select_candidates(rows[:3])


def test_pma_online_softmax_equals_batch_pooling():
    torch.manual_seed(0)
    scores = torch.randn(11, 4, dtype=torch.float64) * 3
    values = torch.randn(11, 5, dtype=torch.float64)
    batch = torch.softmax(scores, dim=0).T @ values
    for chunk in (1, 3, 11):
        torch.testing.assert_close(online_softmax_pool(scores, values, chunk), batch)


def test_p4_log_reliability_shifts_pooling_toward_reliable_units():
    sd, prior = _session(9)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1, 2]))
    d, r, _, _ = describe(stats, prior)
    p3, p4 = _model("p3", desc_dim(K), m=4), _model("p4", desc_dim(K), m=4)
    p4.load_state_dict(p3.state_dict())
    with torch.no_grad():
        _, a3 = p3(d, r, prior.float(), return_aux=True)
        _, a4 = p4(d, r, prior.float(), return_aux=True)
    shift = a4["scores"] - a3["scores"]
    torch.testing.assert_close(shift, torch.log(a4["lam"].clamp_min(1e-6))[:, None].expand_as(shift))


def test_electrode_mean_matches_loo_prior_from_round1():
    from scripts.run_labeled_calibration import loo_prior
    rng = np.random.default_rng(0)
    elec = [np.sort(rng.choice(96, 12, replace=False)) for _ in range(4)]
    tabs = [rng.standard_normal((3, 12)) for _ in range(4)]
    recs = [SimpleNamespace(unit_to_electrode=e) for e in elec]
    for s in range(4):
        ref = loo_prior(s, recs, tabs)
        mine = electrode_mean_table(tabs, elec, exclude=s)[:, elec[s]]
        np.testing.assert_allclose(ref, mine)
    full = electrode_mean_table(tabs, elec)
    loo = electrode_mean_table(tabs, elec, exclude=0)
    assert not np.allclose(full, loo)


def test_norm_stats_match_support_stats_rule():
    sd, _ = _session(10)
    stats = PopStats.from_trials(sd.bank, np.array([0, 1]))
    mean, std = norm_stats(stats)
    assert (std >= 1e-6).all()


def test_label_free_control_ignores_all_label_statistics():
    """Shuffling behavior must not move a label-free calibrator's table."""
    sd, prior = _session(11)
    rec2 = SimpleNamespace(**{k: getattr(sd.record, k) for k in
                              ("neural", "eval_mask", "trial_bounds", "unit_to_electrode", "session")})
    rec2.behavior = np.roll(sd.record.behavior, 333, axis=0)
    sd2 = SessionData(rec2, (np.zeros(2), np.ones(2)), torch.device("cpu"), usable=sd.trials)
    idx = np.array([0, 2, 5, 7])
    lf = _model("p2", desc_dim(K), label_free=True)
    full = _model("p2", desc_dim(K))
    full.load_state_dict(lf.state_dict())
    full.d_keep.fill_(1.0), full.r_keep.fill_(1.0)
    with torch.no_grad():
        outs = []
        for model in (lf, full):
            for s in (sd, sd2):
                d, r, _, _ = describe(PopStats.from_trials(s.bank, idx), prior)
                outs.append(model(d, r, prior.float()))
    torch.testing.assert_close(outs[0], outs[1])           # label-free: identical
    assert (outs[2] - outs[3]).abs().max() > 1e-5          # full model does read labels
