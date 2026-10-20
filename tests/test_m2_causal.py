"""M2 causal-first-B adapter.  Map tests run on CPU.  Parity and the S1 loop need CUDA."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ssm_decode import m2_calib as m2
from ssm_decode import m2_causal as mc
from ssm_decode.causal_protocol import COLD, INITIAL, calibration_indices
from ssm_decode.p0solve import shrink_tables


def test_relative_target_direction_is_an_8bin_condition():
    labels = mc.load_trial_conditions("2020-10-20-Run1")
    assert labels.shape[0] == 248
    counts = np.bincount(labels, minlength=8)
    # Absolute tgt_loc angle collapses into one quadrant.  The reach direction does not.
    assert counts.min() >= 10
    assert counts.max() <= 80
    # Every held-in session we score or select on loads, and none of them is minival/held-out.
    for session in ("2020-10-19-Run1", "2020-10-19-Run2", "2020-10-20-Run1", "2020-10-20-Run2",
                    "2020-10-27-Run1", "2020-10-27-Run2", "2020-10-28-Run1"):
        lab = mc.load_trial_conditions(session)
        assert lab.min() >= 0 and lab.max() <= 7
        assert "held-in-calib" in str(mc.nwb_path(session))


def test_folded_bias_is_the_mean_of_the_source_folds():
    pool = m2.resolve_pool_dir(0, "pool1")
    blob = torch.load(pool / "best.pt", map_location="cpu", weights_only=False)
    bank = torch.load(pool / "session_bank_best.pt", map_location="cpu", weights_only=False)
    tables, biases = m2.fold_bank(bank["state_dict"])
    assert np.allclose(np.mean(biases, 0), blob["state_dict"]["in_proj.bias"].numpy(), atol=1e-6)
    assert np.allclose(np.mean(tables, 0), blob["state_dict"]["in_proj.weight"].numpy(), atol=1e-6)


def test_shrink_keeps_the_frozen_bias_outside_alpha():
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=(6, 4))
    w_cca = rng.normal(size=(6, 4))
    prior = rng.normal(size=(6,))
    b_full = rng.normal(size=(6,))
    b_extra = b_full - prior
    alpha = 0.25
    w, b_shrink = shrink_tables(w0, w_cca, b_extra, alpha)
    installed = prior + b_shrink
    assert np.allclose(w, (1 - alpha) * w0 + alpha * w_cca)
    assert np.allclose(installed, (1 - alpha) * prior + alpha * b_full)
    # alpha = 0 is the prior, bias included, weight untouched.
    w, b_shrink = shrink_tables(w0, w_cca, b_extra, 0.0)
    assert np.allclose(w, w0) and np.allclose(prior + b_shrink, prior)


def test_strict_protocol_drops_a_trunk_trained_on_a_reported_target():
    def ep(session, gap, r2s):
        return {"pool": "pool2", "session": session, "date": mc.ymd(session), "B": 8,
                "gap_days": gap, "alphas": [{"alpha": a, "r2": r} for a, r in zip(mc.ALPHAS, r2s)]}
    rows = [ep("2020-10-19-Run1", -1, [0.1, 0.2, 0.3, 0.4, 0.5]),
            ep("2020-10-20-Run1", 0, [0.1, 0.2, 0.3, 0.4, 0.5])]
    strict = mc.select_alpha_m2(rows, "pool2", "2020-10-27-Run1", 8, "strict")
    expanding = mc.select_alpha_m2(rows, "pool2", "2020-10-27-Run1", 8, "expanding")
    assert strict["alpha"] is None
    assert expanding["alpha"] == 1.0
    assert expanding["bin"] == "le14"
    # pool1's trunk was not fit on a reported date, so (a) and (b) agree.
    rows1 = [dict(rows[0], pool="pool1")]
    a = mc.select_alpha_m2(rows1, "pool1", "2020-10-20-Run1", 8, "expanding")
    b = mc.select_alpha_m2(rows1, "pool1", "2020-10-20-Run1", 8, "strict")
    assert a["alpha"] == 1.0 and b["alpha"] == 1.0


def _cpu_world():
    return m2.M2World(0, "pool1", torch.device("cpu"), load_sources=True)


def test_first_b_maps_keep_the_bias_and_ignore_the_future():
    """Prefix causality at B=8 for P-prior, P0, ridge-CCA and W_alpha.  No trunk forward."""
    world = _cpu_world()
    err = float(np.max(np.abs(world.trunk.prior_bias - world.trunk.folded_bias)))
    assert err < 1e-5
    calibs = mc.load_calibrators(world, 0, "pool1")
    ref = mc.fit_anchor(world.trunk, world.sessions[world.trunk.source_ids[-1]],
                        mc.usable_conditions(world.sessions[world.trunk.source_ids[-1]],
                                             mc.load_trial_conditions(world.trunk.source_ids[-1])),
                        phases=10)
    ref.m2_active = int(ref.m2_active)
    sd, prior = world.target("2020-10-20-Run1")
    prior_t, prior_b_t = prior
    prior_b = mc._as_np(prior_b_t)
    prior_w = mc._as_np(prior_t)
    conds = mc.usable_conditions(sd, mc.load_trial_conditions(sd.name))
    B, J = 8, 4
    assert sd.bank.n_trials >= B
    idx = calibration_indices(sd.bank.n_trials, B)
    assert list(idx) == list(range(B))
    q = sd.query_mask(idx).numpy()
    used = np.zeros(sd.n_bins, dtype=bool)
    spans = [(int(a), int(b)) for a, b in sd.trials]
    for i in idx:
        a, b = spans[int(i)]
        used[a:b] = True
    assert not np.any(used & q)

    def bundle(session):
        cache = mc.prefix_cache(session, prior_t, prior_b, calibs, B, conds, ref, ridge=10.0, phases=10)
        cold, initial = mc.cold_initial(prior_w, prior_b, cache)
        out = {
            "p_prior": mc.prior_maps(prior_w, prior_b, cache, B, cold, initial),
            "p0": mc.p0_maps(cache, 0, B, prior_b, cold, initial),
        }
        cca, info = mc.cca_final_maps(cache, B, prior_b, cold, initial)
        assert cca is not None, info
        out["ridge_cca"] = cca
        wa, info = mc.alpha_maps(cache, 0, B, 0.5, prior_b, cold, initial)
        assert wa is not None, info
        out["w_alpha"] = wa
        # alpha 0 is P0, including the frozen bias.
        w0, info = mc.alpha_maps(cache, 0, B, 0.0, prior_b, cold, initial)
        assert np.allclose(w0[B].weight, out["p0"][B].weight)
        assert np.allclose(w0[B].bias, prior_b)
        return out

    # The offline P0 on these exact trials matches a direct calibrate.
    stats_idx = calibration_indices(sd.bank.n_trials, B)
    direct = calibs[0].calibrate(__import__("ssm_decode.population_calib", fromlist=["PopStats"]).PopStats.from_trials(
        sd.bank, stats_idx), prior_t)
    orig = bundle(sd)
    assert np.allclose(orig["p0"][B].weight, mc._as_np(direct.table))
    assert np.allclose(orig["p0"][B].bias, prior_b)
    assert np.allclose(orig["p_prior"][COLD].bias, prior_b)
    assert orig["p_prior"][COLD].max_bin_read == -1
    # Query perturbation: nothing the first B commits read.
    query = bundle(mc.corrupt_session(sd, world.label_stats, spans[B - 1][1]))
    for name in ("p_prior", "p0", "ridge_cca", "w_alpha"):
        for k, amap in orig[name].items():
            if k == COLD:
                continue
            assert mc.map_delta(amap, query[name][k]) < 1e-8, name
    # In-block perturbation at J=4.  Commits that have not reached trial J stay put.
    prefix = bundle(mc.corrupt_session(sd, world.label_stats, spans[J - 1][1], seed=2))
    for name in ("p_prior", "p0", "w_alpha"):
        for k, amap in orig[name].items():
            if k in (COLD, INITIAL) or (isinstance(k, int) and k > J):
                continue
            assert mc.map_delta(amap, prefix[name][k]) < 1e-8, (name, k)
        assert mc.map_delta(orig[name][B], prefix[name][B]) > 1e-6, name
    # ridge-CCA is final-only: its one commit reads all B trials, so it must move.
    assert mc.map_delta(orig["ridge_cca"][B], prefix["ridge_cca"][B]) > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_static_decode_matches_stored_random_draw_p0():
    """Same trials as a stored random-draw row.  The adapter's static decode, bias included."""
    dev = torch.device("cuda:0")
    world = m2.M2World(0, "pool1", dev, load_sources=False)
    calibs = mc.load_calibrators(world, 0, "pool1")
    sd, prior = world.target("2020-10-20-Run1")
    assert sd.bank.n_trials == 230
    B, draw = 8, 0
    idx = m2.draw_trials("2020-10-20-Run1", B, draw, sd.bank.n_trials)
    from ssm_decode.population_calib import PopStats
    res = calibs[0].calibrate(PopStats.from_trials(sd.bank, idx), prior[0])
    q = sd.query_mask(idx)
    caller = mc.LatentTrunk(world.trunk.model, None)
    got = mc.score_static(caller, sd, world.label_stats, mc._as_np(res.table), mc._as_np(prior[1]),
                          mc._as_np(res.mean), mc._as_np(res.std), q)
    y = m2.decode_full(world.trunk, sd, res.table, prior[1], res.mean, res.std)
    ref = m2.r2_physical(world.label_stats, sd, y, q)[0]
    assert abs(got - ref) < 1e-5
    stored = 0.189595466175422
    assert abs(got - stored) < 1e-3, got


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_s1_loop_prefix_causality_at_b8():
    """The rebased loop's commits do not read past the prefix.  B=8, J=4, eta>0."""
    dev = torch.device("cuda:0")
    world = m2.M2World(0, "pool1", dev, load_sources=True)
    calibs = mc.load_calibrators(world, 0, "pool1")
    # A short real Jacobian so the kernel is the trunk's, not a placeholder.
    blob = mc.jacobian_sums(world.trunk, world.sessions, lmax=8, width=160, n_windows=4, seed=0)
    jbar = mc.jbar_from(blob, 8, device=dev)
    sd, prior = world.target("2020-10-20-Run1")
    prior_t, prior_b = prior[0], mc._as_np(prior[1])
    # S1 on P0 does not need CCA.  An empty ref is avoided by fitting the real anchor at phases 10
    # only if we want W_alpha; here kind='p0' never reads it.  Pass a dummy cache CCA of errors.
    conds = mc.usable_conditions(sd, mc.load_trial_conditions(sd.name))
    ref = mc.fit_anchor(world.trunk, world.sessions[world.source_ids[-1] if False else world.trunk.source_ids[-1]],
                        mc.usable_conditions(world.sessions[world.trunk.source_ids[-1]],
                                             mc.load_trial_conditions(world.trunk.source_ids[-1])),
                        phases=10)
    B, J = 8, 4
    caller = mc.LatentTrunk(world.trunk.model, None)

    def maps_of(session):
        cache = mc.prefix_cache(session, prior_t, prior_b, calibs, B, conds, ref, ridge=10.0, phases=10)
        cold, initial = mc.cold_initial(mc._as_np(prior_t), prior_b, cache)
        body, info = mc.rebased_maps(caller, session, prior_t, cache, 0, B, eta=1.0, rho=0.0, jbar=jbar,
                                     alpha=0.0, kind="p0", prior_b=prior_b)
        assert info["fallback_base"] is False
        body[COLD], body[INITIAL] = cold, initial
        # eta = 0 reproduces P0, bias included.
        base, info0 = mc.rebased_maps(caller, session, prior_t, cache, 0, B, eta=0.0, rho=0.0, jbar=jbar,
                                      alpha=0.0, kind="p0", prior_b=prior_b)
        p0 = mc.p0_maps(cache, 0, B, prior_b, cold, initial)
        assert np.allclose(base[B].weight, p0[B].weight, atol=1e-5)
        assert np.allclose(base[B].bias, p0[B].bias, atol=1e-5)
        return body

    spans = [(int(a), int(b)) for a, b in sd.trials]
    orig = maps_of(sd)
    query = maps_of(mc.corrupt_session(sd, world.label_stats, spans[B - 1][1]))
    for k, amap in orig.items():
        if k == COLD:
            continue
        assert mc.map_delta(amap, query[k]) < 1e-5, k
    prefix = maps_of(mc.corrupt_session(sd, world.label_stats, spans[J - 1][1], seed=3))
    for k, amap in orig.items():
        if k in (COLD, INITIAL) or (isinstance(k, int) and k > J):
            continue
        assert mc.map_delta(amap, prefix[k]) < 1e-5, k
    assert mc.map_delta(orig[B], prefix[B]) > 1e-5
