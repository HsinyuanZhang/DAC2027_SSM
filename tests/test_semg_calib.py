"""Tests for the sEMG fixed-weight forward calibrators (Q-D)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssm_decode import hyser_backbone as hysb
from ssm_decode.semg_calib import (SemgCalibrator, SemgSpec, SemgStats, StreamAcc, TrialBank, VARIANTS,
                                   build_prior, cost_model, describe, mean_prior, norm_stats, segment_sums)
from ssm_decode.semg_calib_db6 import Db6StreamEnv
from ssm_decode.semg_calib_env import HyserEnv, SessStore, draw_hyser_cal

K = 4
N = 7
SPEC_H = SemgSpec("hyser_like", N, K, 3, (0, 2, 4))
SPEC_D = SemgSpec("db6_like", N, K, 5, (0,))
ROOT = Path(__file__).resolve().parents[1]


def _trials(spec, n_trials=4, seed=0, lens=(60, 45, 80, 50)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_trials):
        t = lens[i % len(lens)]
        x = torch.from_numpy(np.abs(rng.standard_normal((t, spec.n_ch))) + 0.2).double()
        y = torch.from_numpy(rng.standard_normal((t, spec.d_label))).double()
        v = torch.ones(t, dtype=torch.bool)
        v[7:11] = False                                   # interior invalid gap
        out.append((x, y, v))
    return out


def _stats(spec, trials):
    bank = TrialBank(spec)
    for i, (x, y, v) in enumerate(trials):
        bank.add(i, x, y, v)
    return SemgStats.from_bank(bank, list(range(len(trials))))


def _rand_model(variant, spec, prior_mode="aligned", seed=0):
    m = SemgCalibrator(variant, spec, prior_mode=prior_mode, out_scale=0.3)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


def test_streamed_chunks_equal_batch_sums_and_descriptors():
    trials = _trials(SPEC_H)
    batch = _stats(SPEC_H, trials)
    rng = np.random.default_rng(1)
    acc = StreamAcc(SPEC_H)
    for x, y, v in trials:
        cuts = sorted(rng.choice(np.arange(1, x.shape[0]), size=5, replace=False).tolist())
        for a, b in zip([0] + cuts, cuts + [x.shape[0]]):
            acc.update(x[a:b], y[a:b], v[a:b])
        acc.end_trial()
    streamed = acc.result()
    assert streamed.n_trials == batch.n_trials
    for k in batch.d:
        assert torch.allclose(streamed.d[k], batch.d[k], atol=1e-9, rtol=1e-9), k
    prior = torch.randn(K, N, dtype=torch.float64) * 0.1
    d0, r0, m0, s0 = describe(batch, prior, SPEC_H)
    d1, r1, m1, s1 = describe(streamed, prior, SPEC_H)
    assert torch.allclose(d0, d1, atol=1e-6) and torch.allclose(r0, r1, atol=1e-6)


def test_trial_order_and_chunk_merge_are_exact():
    trials = _trials(SPEC_D, seed=3)
    a = _stats(SPEC_D, trials)
    b = _stats(SPEC_D, trials[::-1])
    for k in a.d:
        assert torch.allclose(a.d[k], b.d[k], atol=1e-9), k
    s1 = _stats(SPEC_D, trials[:2])
    bank = TrialBank(SPEC_D)
    for i, (x, y, v) in enumerate(trials):
        bank.add(i, x, y, v)
    s2 = SemgStats.from_bank(bank, [2, 3])
    merged = s1 + s2
    for k in ("n", "sx", "gram", "syx", "q_sx2"):
        assert torch.allclose(merged.d[k], a.d[k], atol=1e-9), k


def test_norm_stats_match_backbone_input_stats():
    trials = _trials(SPEC_H, seed=5)
    xs = [x[v] for x, _, v in trials]
    st = _stats(SPEC_H, trials)
    mu, sd = norm_stats(st)
    flat = torch.cat(xs).numpy()
    ref_mu, ref_sd = hysb.input_stats(flat[None])           # [1, T, C] -> same flat statistics
    assert np.allclose(mu.numpy(), ref_mu, rtol=1e-5, atol=1e-6)
    assert np.allclose(sd.numpy(), ref_sd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("prior_mode", ["aligned", "free"])
def test_permutation_equivariance_over_channels(variant, prior_mode):
    trials = _trials(SPEC_H, seed=7)
    perm = torch.as_tensor(np.random.default_rng(2).permutation(N))
    prior = torch.randn(K, N, dtype=torch.float64) * 0.1 if prior_mode == "aligned" else None
    model = _rand_model(variant, SPEC_H, prior_mode).eval()
    t0 = model.calibrate(_stats(SPEC_H, trials), prior).table
    trials_p = [(x[:, perm], y, v) for x, y, v in trials]
    prior_p = prior[:, perm] if prior is not None else None
    res = model.calibrate(_stats(SPEC_H, trials_p), prior_p)
    assert torch.allclose(res.table, t0[:, perm], atol=2e-4)
    assert torch.allclose(res.mean, model.calibrate(_stats(SPEC_H, trials), prior).mean[perm])


def test_identity_free_ignores_prior_columns():
    trials = _trials(SPEC_D, seed=9)
    st = _stats(SPEC_D, trials)
    model = _rand_model("p2", SPEC_D, "free").eval()
    d0, r0, _, _ = describe(st, None, SPEC_D)
    d1, r1, _, _ = describe(st, torch.randn(K, N, dtype=torch.float64), SPEC_D)
    g = SPEC_D.groups
    cols = g["prior"] + g["corr_prior"]
    other = [c for c in range(SPEC_D.desc_dim) if c not in cols]
    assert torch.allclose(d0[:, other], d1[:, other])
    assert float(d0[:, cols].abs().sum()) == 0.0
    # the generator output does not depend on the (zeroed) prior-dependent columns
    pr = torch.zeros(K, N)
    t_a = model(d0, r0, pr)
    t_b = model(d1, r1, pr)
    assert torch.allclose(t_a, t_b, atol=1e-6)


def test_no_grad_target_guard_even_when_called_with_grad_enabled():
    trials = _trials(SPEC_H, seed=11)
    prior = torch.randn(K, N, dtype=torch.float64) * 0.1
    model = _rand_model("p4", SPEC_H).eval()
    assert torch.is_grad_enabled()
    res = model.calibrate(_stats(SPEC_H, trials), prior)
    assert not res.table.requires_grad
    assert res.grad_enabled_inside is False
    assert all(p.grad is None for p in model.parameters())


def test_calibrator_starts_at_the_prior():
    trials = _trials(SPEC_H, seed=13)
    prior = torch.randn(K, N, dtype=torch.float64) * 0.1
    m = SemgCalibrator("p4", SPEC_H, out_scale=0.5)
    out = m.calibrate(_stats(SPEC_H, trials), prior).table
    assert torch.allclose(out, prior.float(), atol=1e-6)


def test_descriptors_use_only_requested_trials():
    """Replacing the non-calibration trials by noise must not change stats / descriptors."""
    trials = _trials(SPEC_H, n_trials=6, seed=15, lens=(50,))
    x_all = torch.stack([t[0] for t in trials]).float()
    y_all = torch.stack([t[1] for t in trials])
    store = SessStore(x_all, lambda i, L: y_all[i, :L], [50] * 6, SPEC_H, torch.device("cpu"))
    cal = [1, 4]
    st1 = store.stats(cal)
    assert sorted(store.bank.rows) == cal                 # nothing else was ever summed
    x2 = x_all.clone()
    x2[[0, 2, 3, 5]] = torch.randn_like(x2[[0, 2, 3, 5]]) * 50
    store2 = SessStore(x2, lambda i, L: y_all[i, :L], [50] * 6, SPEC_H, torch.device("cpu"))
    st2 = store2.stats(cal)
    for k in st1.d:
        assert torch.equal(st1.d[k], st2.d[k]), k


def test_prior_construction_excludes_target_data():
    tables = torch.randn(6, K, N)
    p = mean_prior(tables, exclude=[1, 4])
    t2 = tables.clone()
    t2[[1, 4]] = 1e6
    assert torch.allclose(mean_prior(t2, exclude=[1, 4]), p)
    assert build_prior("free", tables, source="mean") is None
    own = tables[2]
    assert torch.equal(build_prior("aligned", tables, source="own", own=own), own)
    assert build_prior("aligned", tables, source="own", own=own) is not own
    with pytest.raises(ValueError):
        build_prior("aligned", tables, source="own")


def _stub_db6_env():
    env = object.__new__(Db6StreamEnv)
    env.keys = [(s, d, t) for s in (1, 2, 3) for d in (1, 2) for t in (1, 2)]
    env.subject = [k[0] for k in env.keys]
    env.tables = torch.randn(len(env.keys), K, N)
    env.val_sess = [i for i, k in enumerate(env.keys) if (k[1], k[2]) == (2, 2)]
    return env


def test_db6_priors_never_use_target_subject_or_later_sessions():
    env = _stub_db6_env()
    j = env.keys.index((2, 2, 1))
    prior = env.prior_for("mean", j, exclude_val=False)
    t2 = env.tables.clone()
    for i, k in enumerate(env.keys):
        if k[0] == 2:
            t2[i] = 1e6                                   # the target subject's tables
    env2 = _stub_db6_env(); env2.tables = t2
    assert torch.allclose(env2.prior_for("mean", j, exclude_val=False), prior)
    # 'own' prior = strictly earlier session of the same subject; never the target session itself
    for tgt in [(2, 1, 2), (2, 2, 1), (2, 2, 2)]:
        jj = env.keys.index(tgt)
        prev = env.earlier(jj)
        assert all((env.keys[i][1], env.keys[i][2]) < (tgt[1], tgt[2]) and env.keys[i][0] == 2 for i in prev)
        latest = env.prior_for("own", jj, latest=True)
        assert torch.equal(latest, env.tables[max(prev, key=lambda i: env.keys[i][1:])])
    with pytest.raises(ValueError):
        env.prior_for("own", env.keys.index((2, 1, 1)))
    # training sessions never include the validation sessions
    assert not set(env.train_sessions("mean")) & set(env.val_sess)
    assert not set(env.train_sessions("own")) & set(env.val_sess)


@pytest.mark.skipif(not (ROOT / "results/analysis/hyser_20261015/raw/rows_seed0.jsonl.gz").exists(),
                    reason="reference rows missing")
def test_hyser_draws_reproduce_reference_calibration_sets():
    fingers = np.repeat(np.arange(1, 6), 3)
    samples = np.tile(np.arange(1, 4), 5)
    seen = 0
    with gzip.open(ROOT / "results/analysis/hyser_20261015/raw/rows_seed0.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["method"] != "reuse_s1_calnorm" or r["target"] > 4:
                continue
            rng = np.random.default_rng([7919, r["target"], r["B"], r["draw"]])
            assert draw_hyser_cal(r["B"], rng, fingers, samples) == r["cal_trials"]
            seen += 1
    assert seen >= 100


def test_cost_model_is_self_consistent():
    spec = SemgSpec("hyser", 256, 16, 5, (0, 2, 4, 6))
    c = cost_model(spec, "p4", 1250, 10)
    assert c["accumulate_macs"] == 1250 * 10 * c["accumulate_macs_per_bin"]
    assert c["accumulate_macs_per_bin"] >= 256 * 256
    assert c["sram_bytes"]["gram_upper_triangle"] < c["sram_bytes"]["gram_full"]
    assert c["total_macs"] == c["accumulate_macs"] + c["finalize_macs"] + c["calibrator_fixed_macs"]


def _stub_hyser_env():
    env = object.__new__(HyserEnv)
    env.n_sub = 20
    env.tables = torch.randn(20, K, N)
    env.out_scale = 0.1
    env.dev = torch.device("cpu")
    return env


def test_hyser_folds_never_leak_heldout_subjects_into_training_or_priors():
    env = _stub_hyser_env()
    seen_held = set()
    for fold in range(5):
        train, val, held = env.split_subjects(fold, 5)
        assert not (set(train) & set(held)) and not (set(val) & set(held)) and not (set(train) & set(val))
        assert sorted(train + val + held) == list(range(20))
        seen_held |= set(held)
        # training priors are built from training subjects only and never use the target's own table
        s = train[0]
        base = env.prior_for("mean", s, train)
        t2 = env.tables.clone()
        t2[held + val + [s]] = 1e6
        env2 = _stub_hyser_env(); env2.tables = t2
        assert torch.allclose(env2.prior_for("mean", s, train), base)
        rng = np.random.default_rng(0)
        own = env.prior_for("own", s, train, rng)
        assert own.shape == base.shape and not torch.allclose(own, env.tables[s])   # stale surrogate, not the raw table
    assert seen_held == set(range(20))                  # every subject is held out exactly once


def _random_trunk_state_dict(width=128, layers=2, state=32, k_in=8, out=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g) * 0.1  # noqa: E731
    h = 2 * width
    heads = h // 64
    sd = {"base.in_proj.weight": r(width, k_in), "base.in_proj.bias": r(width),
          "base.final_norm.weight": torch.ones(width), "base.final_norm.bias": torch.zeros(width),
          "base.out_proj.weight": r(out, width), "base.out_proj.bias": r(out)}
    for i in range(layers):
        p = f"base.blocks.{i}."
        ip_rows = 2 * h + 2 * state + 3 * heads + state // 4
        sd.update({p + "norm1.weight": torch.ones(width), p + "norm1.bias": torch.zeros(width),
                   p + "ssm.in_proj.weight": r(ip_rows, width), p + "ssm.dt_bias": r(heads),
                   p + "ssm.B_bias": r(heads, 1, state), p + "ssm.C_bias": r(heads, 1, state), p + "ssm.D": r(heads),
                   p + "ssm.B_norm.weight": torch.ones(state), p + "ssm.C_norm.weight": torch.ones(state),
                   p + "ssm.out_proj.weight": r(width, h), p + "norm2.weight": torch.ones(width),
                   p + "norm2.bias": torch.zeros(width), p + "ffn.0.weight": r(2 * width, width),
                   p + "ffn.0.bias": r(2 * width), p + "ffn.2.weight": r(width, 2 * width), p + "ffn.2.bias": r(width)})
    return sd


def test_cpu_trunk_decodes_long_streams_causally_and_carries_state():
    from ssm_decode.semg_calib_db6 import TrainableCpuTrunk, _LongCpuDecoder
    sd = _random_trunk_state_dict()
    trunk = _LongCpuDecoder(sd)
    x = torch.randn(2, 400, 8)
    full = trunk(x)
    prefix = trunk(x[:, :260])
    assert torch.allclose(full[:, :260], prefix, atol=1e-4)          # causal beyond the 128-bin guard of the port
    other = x.clone(); other[:, 300:] += 5.0
    assert torch.allclose(trunk(other)[:, :300], full[:, :300], atol=1e-4)
    assert not torch.allclose(trunk(other)[:, 320:], full[:, 320:], atol=1e-3)
    # the differentiable variant gives the same values and a gradient w.r.t. the input
    tr = TrainableCpuTrunk(sd)
    xg = x[:, :96].clone().requires_grad_(True)
    out = tr(xg)
    assert torch.allclose(out, trunk(x[:, :96]), atol=1e-5)
    out.square().mean().backward()
    assert xg.grad is not None and float(xg.grad.abs().sum()) > 0
