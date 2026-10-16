"""CPU tests for ssm_decode/m2_calib.py (no Mamba-3 / GPU needed: a tiny causal GRU stands in)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ssm_decode import m2_calib as m2
from ssm_decode.population_calib import PopStats, SessionData, norm_stats

N, K = 12, 24
DEV = torch.device("cpu")


class FakeTrunk:
    """Causal stand-in with the interface used by m2_calib: run(z, want_h) and head()."""

    def __init__(self, k=K, h=16, d=2, seed=0):
        torch.manual_seed(seed)
        self.rnn = nn.GRU(k, h, batch_first=True)
        self.out_proj = nn.Linear(h, d)
        for p in list(self.rnn.parameters()) + list(self.out_proj.parameters()):
            p.requires_grad_(False)
        self.model = SimpleNamespace(out_proj=self.out_proj)
        self.k = k

    def run(self, z, want_h=False):
        h, _ = self.rnn(z)
        y = self.out_proj(h)
        return (y, h) if want_h else y

    def head(self):
        return torch.cat([self.out_proj.weight, self.out_proj.bias[:, None]], 1).detach()


def synth_record(seed=0, n_trials=40, n=N, tlen=60):
    rng = np.random.default_rng(seed)
    bounds, t = [], 0
    for _ in range(n_trials):
        L = tlen + int(rng.integers(0, 20))
        bounds.append((t, t + L))
        t += L
    x = rng.poisson(0.5, size=(t, n)).astype(np.float64)
    ang = rng.uniform(0, 2 * np.pi, size=t)
    y = 0.01 * np.stack([np.cos(ang), np.sin(ang)], 1)
    return SimpleNamespace(neural=x, behavior=y, eval_mask=np.ones(t, bool), trial_bounds=tuple(bounds),
                           session=f"s{seed}", unit_to_electrode=np.arange(n))


def synth_session(seed=0):
    rec = synth_record(seed)
    return SessionData(rec, (np.zeros(2), np.full(2, 0.01)), DEV, name=f"s{seed}")


def fake_prior(seed=0, n=N):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, n, generator=g) * 0.1, torch.randn(K, generator=g) * 0.1


def test_fold_bank_mean_equals_folded_in_proj():
    torch.manual_seed(0)
    sdict = {"base.in_proj.weight": torch.randn(K, N), "base.in_proj.bias": torch.randn(K),
             "bank.gain": torch.exp(0.3 * torch.randn(4, N)), "bank.bias": 0.1 * torch.randn(4, N),
             "bank.embedding": 0.1 * torch.randn(4, K)}
    tabs, bias = m2.fold_bank(sdict)
    # reference: the repo's own fold of the MEAN gain / offset / embedding
    from ssm_decode.session_frontend import SessionFrontendBank, fold_session_input
    bank = SessionFrontendBank(4, N, K)
    with torch.no_grad():
        bank.gain.copy_(sdict["bank.gain"]), bank.bias.copy_(sdict["bank.bias"])
        bank.embedding.copy_(sdict["bank.embedding"])
    lin = nn.Linear(N, K)
    with torch.no_grad():
        lin.weight.copy_(sdict["base.in_proj.weight"]), lin.bias.copy_(sdict["base.in_proj.bias"])
    folded = fold_session_input(lin, bank)
    assert np.abs(np.mean(tabs, 0) - folded.weight.detach().double().numpy()).max() < 1e-5
    assert np.abs(np.mean(bias, 0) - folded.bias.detach().double().numpy()).max() < 1e-5


def test_loo_prior_excludes_own_session():
    tr = object.__new__(m2.M2Trunk)
    tr.src_tables = [np.full((K, N), float(j)) for j in range(4)]
    tr.src_bias = [np.full(K, 10.0 * j) for j in range(4)]
    tab, bias = tr.loo_prior(2)
    assert np.allclose(tab, (0 + 1 + 3) / 3) and np.allclose(bias, 10 * (0 + 1 + 3) / 3)
    tab, _ = tr.loo_prior(None)
    assert np.allclose(tab, 1.5)


def test_resolve_pool_dir_extension_rule(tmp_path):
    base = tmp_path / m2.BACKBONE_ROOTS[1] / m2.POOL_DIRS["pool3"]
    x2 = base.with_name(base.name + "_x2")
    base.mkdir(parents=True), x2.mkdir(parents=True)
    (base / "manifest.json").write_text(json.dumps({"best_step": 1800, "args": {"steps": 2000}}))
    assert m2.resolve_pool_dir(1, "pool3", root=tmp_path) == x2          # 1800 > 0.8 * 2000
    (base / "manifest.json").write_text(json.dumps({"best_step": 1400, "args": {"steps": 2000}}))
    assert m2.resolve_pool_dir(1, "pool3", root=tmp_path) == base
    assert m2.resolve_pool_dir(1, "pool1", root=tmp_path) == tmp_path / m2.BACKBONE_ROOTS[1] / m2.POOL_DIRS["pool1"]


def test_draw_trials_matches_p1():
    pytest.importorskip("scripts.run_population_calib")
    from scripts import run_population_calib as rpc
    for B in (1, 8, 33):
        assert (m2.draw_trials("x", B, 3, 200) == rpc.draw_trials("x", B, 3, 200)).all()


def test_untrained_calibrator_reproduces_prior_table():
    sd = synth_session()
    prior_tab, _ = fake_prior()
    world = SimpleNamespace(k=K, device=DEV, u_full=torch.linalg.svd(prior_tab, full_matrices=False)[0])
    calib = m2.build_calibrator(world, rank=4, hidden=8)
    idx = np.arange(5)
    stats = PopStats.from_trials(sd.bank, idx)
    d, r, _, _ = m2.describe_for(calib, stats, prior_tab)
    calib.set_normalizers(d, r)
    res = calib.calibrate(stats, prior_tab)
    assert torch.allclose(res.table, prior_tab.float(), atol=1e-7)
    assert not res.table.requires_grad and res.mode == "candidate"
    assert res.table.shape == (K, N)


def test_calibrator_is_per_channel_independent():
    """P0 has no population pooling: permuting channels permutes table columns (same stats)."""
    sd = synth_session(1)
    prior_tab, _ = fake_prior(1)
    world = SimpleNamespace(k=K, device=DEV, u_full=torch.linalg.svd(prior_tab, full_matrices=False)[0])
    calib = m2.build_calibrator(world, rank=4, hidden=8)
    with torch.no_grad():
        for p in calib.gen.parameters():
            p.add_(0.05 * torch.randn_like(p))
    idx = np.arange(6)
    stats = PopStats.from_trials(sd.bank, idx)
    d, r, _, _ = m2.describe_for(calib, stats, prior_tab)
    calib.set_normalizers(d, r)
    t0 = calib.calibrate(stats, prior_tab).table
    perm = torch.randperm(N)
    stats_p = stats.select_units(perm.numpy())
    t1 = calib.calibrate(stats_p, prior_tab[:, perm]).table
    assert torch.allclose(t0[:, perm], t1, atol=1e-5)


def _run_stream(trunk, sd, prior, calib, idx, burn=0, use_calib=True):
    return m2.stream_readout_stats(trunk, sd, prior, calib, idx, burn=burn, use_calib=use_calib)


class SpyCalib:
    """Records how many trials each calibration saw; returns the prior with an index-dependent scale."""

    def __init__(self):
        self.seen = []

    def calibrate(self, stats, prior):
        self.seen.append(stats.n_trials)
        mean, std = norm_stats(stats)
        return SimpleNamespace(table=prior.float() * (1.0 + 0.1 * stats.n_trials), mean=mean, std=std)


def test_stream_uses_only_earlier_trials_and_each_trial_once():
    sd = synth_session(2)
    trunk = FakeTrunk()
    prior = fake_prior(2)
    spy = SpyCalib()
    idx = np.array([7, 3, 12, 20])
    st = _run_stream(trunk, sd, prior, spy, idx)
    assert spy.seen == [1, 2, 3]                      # trial j (chronological) sees trials < j
    # reference: per-trial separate forwards, accumulated manually
    G = torch.zeros(K * 0 + 17, 17, dtype=torch.float64)
    R = torch.zeros(2, 17, dtype=torch.float64)
    n = 0.0
    order = np.sort(idx)
    for j, tr in enumerate(order):
        if j == 0:
            mean, std = norm_stats(PopStats.from_trials(sd.bank, order[:1]))
            table = prior[0].float()
        else:
            res = SpyCalib().calibrate(PopStats.from_trials(sd.bank, order[:j]), prior[0])
            mean, std, table = res.mean, res.std, res.table
        a, b = sd.trials[tr]
        z = m2.latent(sd, table, prior[1], mean, std, a, b)[None]
        y, h = trunk.run(z, want_h=True)
        hb = torch.cat([h, torch.ones_like(h[..., :1])], -1)[0]
        r0 = sd.y_norm.float()[a:b] - y[0]
        G += (hb.T @ hb).double()
        R += (r0.T @ hb).double()
        n += b - a
    assert abs(st["n"] - n) < 1e-6
    assert torch.allclose(st["G"], G, atol=1e-4) and torch.allclose(st["R"], R, atol=1e-4)


def test_stream_burn_excludes_leading_bins():
    sd = synth_session(3)
    trunk = FakeTrunk()
    prior = fake_prior(3)
    idx = np.array([1, 5])
    s0 = _run_stream(trunk, sd, prior, SpyCalib(), idx, burn=0)
    s10 = _run_stream(trunk, sd, prior, SpyCalib(), idx, burn=10)
    assert s0["n"] - s10["n"] == 20


def test_ridge_noop_and_shrinks_with_lambda():
    sd = synth_session(4)
    trunk = FakeTrunk()
    prior = fake_prior(4)
    st = _run_stream(trunk, sd, prior, SpyCalib(), np.arange(8))
    v0 = trunk.head().double()
    assert torch.equal(m2.ridge_v(trunk, st, None), v0)
    d_small = (m2.ridge_v(trunk, st, 1e-3) - v0).norm()
    d_big = (m2.ridge_v(trunk, st, 1e6) - v0).norm()
    assert d_big < 1e-4 * max(float(d_small), 1e-9) + 1e-8 and d_small > d_big


def test_ridge_recovers_a_shifted_readout():
    """With enough clean data, R1 moves V toward the true readout (sanity of the sign conventions)."""
    torch.manual_seed(0)
    sd = synth_session(5)
    trunk = FakeTrunk()
    prior = fake_prior(5)
    # re-define the labels as an exact linear function of the features with a shifted readout
    spy = SpyCalib()
    idx = np.arange(20)
    st0 = _run_stream(trunk, sd, prior, spy, idx)
    v_true = trunk.head().double() + 0.3 * torch.randn(2, 17, dtype=torch.double)
    # build y = hb @ v_true for every bin of every trial by monkeypatching y_norm
    order = np.sort(idx)
    y_new = sd.y_norm.clone()
    for j, tr in enumerate(order):
        if j == 0:
            mean, std = norm_stats(PopStats.from_trials(sd.bank, order[:1]))
            table = prior[0].float()
        else:
            res = SpyCalib().calibrate(PopStats.from_trials(sd.bank, order[:j]), prior[0])
            mean, std, table = res.mean, res.std, res.table
        a, b = sd.trials[tr]
        _, h = trunk.run(m2.latent(sd, table, prior[1], mean, std, a, b)[None], want_h=True)
        hb = torch.cat([h[0], torch.ones_like(h[0][:, :1])], 1)
        y_new[a:b] = (hb.double() @ v_true.T).float()
    sd.y_norm = y_new
    st = _run_stream(trunk, sd, prior, SpyCalib(), idx)
    v = m2.ridge_v(trunk, st, 1e-6)
    assert (v - v_true).norm() < 1e-2 * v_true.norm()


def test_no_grad_in_candidate_paths():
    sd = synth_session(6)
    trunk = FakeTrunk()
    prior = fake_prior(6)
    st = _run_stream(trunk, sd, prior, SpyCalib(), np.arange(4))
    assert not st["G"].requires_grad and not st["R"].requires_grad
    y = m2.decode_full(trunk, sd, prior[0], prior[1], *norm_stats(PopStats.from_trials(sd.bank, np.arange(4))))
    assert not y.requires_grad


def test_decode_is_causal_and_contiguous():
    sd = synth_session(7)
    trunk = FakeTrunk()
    prior = fake_prior(7)
    mean, std = norm_stats(PopStats.from_trials(sd.bank, np.arange(4)))
    y = m2.decode_full(trunk, sd, prior[0], prior[1], mean, std)
    sd2 = synth_session(7)
    sd2.x32 = sd2.x32.clone()
    sd2.x32[300:] += 5.0                       # perturb the future only
    y2 = m2.decode_full(trunk, sd2, prior[0], prior[1], mean, std)
    assert torch.allclose(y[:300], y2[:300], atol=1e-6)
    assert not torch.allclose(y[300:], y2[300:])


def test_train_val_regions_are_disjoint():
    rec = synth_record(8, n_trials=50)
    vs = m2.val_start_bin(rec)
    w = object.__new__(m2.M2World)
    sd = SessionData(rec, (np.zeros(2), np.full(2, 0.01)), DEV, name="s8")
    w.sessions, w.val_start = {"s8": sd}, {"s8": vs}
    tr = w.train_trials("s8")
    assert all(sd.trials[i][1] <= vs for i in tr)
    vm = w.val_mask("s8")
    assert vm[:vs].sum() == 0 and vm[vs:].any()
    for i in tr:                                 # no train trial overlaps the val query bins
        a, b = sd.trials[i]
        assert not vm[a:b].any()


def test_int8_quant_and_table_size():
    t = torch.randn(K, N)
    tq, nb = m2.quantize_table_int8(t)
    assert nb == K * N + 4 * K
    assert (tq - t).abs().max() <= (t.abs().amax(1).max() / 127.0) * 0.5 + 1e-6
    tb = m2.table_bytes(256, 96)
    assert tb["fp32_bytes"] == 98304 and tb["int8_bytes"] == 24576
    assert not tb["fp32_fits"] and tb["int8_fits"]


def test_ui_reg_m2_penalty_pulls_to_center_and_trunk_unchanged():
    sd = synth_session(9)
    trunk = FakeTrunk()
    prior = fake_prior(9)
    mean, std = norm_stats(PopStats.from_trials(sd.bank, np.arange(4)))
    xz = ((sd.x32.double() - mean) / std).numpy()
    y = sd.y_norm.double().numpy()
    wins = [sd.trials[i] for i in range(4)]
    before = [p.clone() for p in trunk.rnn.parameters()]
    free = m2.ui_reg_trajectory_m2(trunk, xz, y, wins, prior[0].numpy(), prior[1].numpy(), prior[0].numpy(),
                                   lam=0.0, lr=1e-2, snapshots=[20], device=DEV)[20]
    pen = m2.ui_reg_trajectory_m2(trunk, xz, y, wins, prior[0].numpy(), prior[1].numpy(), prior[0].numpy(),
                                  lam=1e3, lr=1e-2, snapshots=[20], device=DEV)[20]
    c = prior[0].numpy()
    assert np.linalg.norm(pen - c) < np.linalg.norm(free - c)
    for a, b in zip(before, trunk.rnn.parameters()):
        assert torch.equal(a, b)


def test_real_pool_files_consistency_if_present():
    pool = m2.ROOT / m2.BACKBONE_ROOTS[0] / m2.POOL_DIRS["pool1"]
    if not (pool / "session_bank_best.pt").exists():
        pytest.skip("pool files not present")
    best = torch.load(pool / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    bank = torch.load(pool / "session_bank_best.pt", map_location="cpu", weights_only=False)
    tabs, bias = m2.fold_bank(bank["state_dict"])
    assert np.abs(np.mean(tabs, 0) - best["in_proj.weight"].double().numpy()).max() < 1e-5
    assert np.abs(np.mean(bias, 0) - best["in_proj.bias"].double().numpy()).max() < 1e-5
    rec = m2.load_m2_recording("2020-10-19-Run1")
    assert rec.neural.shape[1] == m2.N_CHANNELS and len(rec.trial_bounds) > 100


def test_ui_reg_padding_does_not_change_the_trajectory():
    sd = synth_session(10)
    trunk = FakeTrunk()
    prior = fake_prior(10)
    mean, std = norm_stats(PopStats.from_trials(sd.bank, np.arange(4)))
    xz = ((sd.x32.double() - mean) / std).numpy()
    y = sd.y_norm.double().numpy()
    wins = [sd.trials[i] for i in range(4)]
    kw = dict(lam=1e-3, lr=1e-2, snapshots=[15], device=DEV, seed=3)
    a = m2.ui_reg_trajectory_m2(trunk, xz, y, wins, prior[0].numpy(), prior[1].numpy(), prior[0].numpy(),
                                pad_multiple=1, **kw)[15]
    b = m2.ui_reg_trajectory_m2(trunk, xz, y, wins, prior[0].numpy(), prior[1].numpy(), prior[0].numpy(),
                                pad_multiple=64, **kw)[15]
    assert np.abs(a - b).max() < 1e-6


# ------------------------------------------------- review fixes (2026-10-16) ----

def _write_npz(dirpath, session="2020-10-19-Run1", n=50, **override):
    """Synthetic cache entry with the metadata cached_recording writes."""
    rng = np.random.default_rng(0)
    d = dict(neural=rng.poisson(1.0, (n, m2.N_CHANNELS)).astype(np.float32),
             behavior=rng.normal(size=(n, 2)).astype(np.float32), trial_change=np.zeros(n, bool),
             eval_mask=np.ones(n, bool), trial_bounds=np.array([[0, 20], [20, 50]], np.int64),
             path=f"/data/000953/sub-MonkeyN-held-in-calib/sub-MonkeyN-held-in-calib_ses-{session}_behavior+ecephys.nwb",
             sha256="a" * 64, data_root="/data")
    d.update(override)
    dirpath.mkdir(parents=True, exist_ok=True)
    np.savez(dirpath / f"{session}.npz", **d)
    return d


def test_loader_integrity_ok_and_rejects_tampering(tmp_path):
    _write_npz(tmp_path)
    man = m2.build_m2_manifest(tmp_path)
    rec = m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    assert rec.neural.shape == (50, m2.N_CHANNELS) and len(rec.trial_bounds) == 2
    # array tampering (same metadata) -> digest mismatch
    d = _write_npz(tmp_path)
    d["neural"][0, 0] += 1.0
    np.savez(tmp_path / "2020-10-19-Run1.npz", **d)
    with pytest.raises(ValueError, match="array_sha256"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    # NWB hash / data root changed in the cache metadata -> mismatch against the pinned manifest
    _write_npz(tmp_path, sha256="b" * 64)
    with pytest.raises(ValueError, match="nwb_sha256"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    _write_npz(tmp_path, data_root="/elsewhere")
    with pytest.raises(ValueError, match="data_root"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)


def test_loader_rejects_missing_metadata_wrong_dataset_and_unpinned(tmp_path):
    _write_npz(tmp_path)
    man = m2.build_m2_manifest(tmp_path)
    # missing metadata keys
    blob = dict(np.load(tmp_path / "2020-10-19-Run1.npz"))
    for k in ("sha256", "data_root"):
        blob.pop(k)
    np.savez(tmp_path / "2020-10-19-Run1.npz", **blob)
    with pytest.raises(ValueError, match="lacks integrity metadata"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    # DANDI 000954 (HumanPitt) path must be refused
    _write_npz(tmp_path, path="/data/000954/sub-HumanPitt-held-in-calib/x_ses-2020-10-19-Run1_a.nwb")
    with pytest.raises(ValueError, match="000953"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    # malformed sha
    _write_npz(tmp_path, sha256="xyz")
    with pytest.raises(ValueError, match="malformed"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)
    # session not pinned / manifest file missing
    _write_npz(tmp_path)
    with pytest.raises(ValueError, match="not in the pinned manifest"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest={})
    # inconsistent shapes
    _write_npz(tmp_path, behavior=np.zeros((7, 2), np.float32))
    with pytest.raises(ValueError, match="shapes"):
        m2.load_m2_recording("2020-10-19-Run1", tmp_path, manifest=man)


def test_loader_verify_nwb_rehashes_the_file(tmp_path):
    import hashlib
    nwb_dir = tmp_path / "000953" / "sub-MonkeyN-held-in-calib"
    nwb_dir.mkdir(parents=True)
    nwb = nwb_dir / "sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb"
    nwb.write_bytes(b"fake nwb")
    sha = hashlib.sha256(b"fake nwb").hexdigest()
    cache = tmp_path / "cache"
    _write_npz(cache, path=str(nwb), sha256=sha)
    man = m2.build_m2_manifest(cache)
    assert m2.load_m2_recording("2020-10-19-Run1", cache, manifest=man, verify_nwb=True).session
    nwb.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="NWB"):
        m2.load_m2_recording("2020-10-19-Run1", cache, manifest=man, verify_nwb=True)
    nwb.unlink()                                       # no NWB (Colab VM): hash check requires the file
    with pytest.raises(ValueError, match="NWB"):
        m2.load_m2_recording("2020-10-19-Run1", cache, manifest=man, verify_nwb=True)
    assert m2.load_m2_recording("2020-10-19-Run1", cache, manifest=man).session   # default needs no NWB


def test_pinned_manifest_matches_real_cache_if_present():
    if not (m2.M2_CACHE / "2020-10-19-Run1.npz").exists() or not m2.M2_MANIFEST.exists():
        pytest.skip("cache/manifest not present")
    for f in sorted(m2.M2_CACHE.glob("*.npz")):
        assert m2.load_m2_recording(f.stem).session == f.stem


def test_stream_multi_burn_equals_single_burn_calls():
    sd = synth_session(21)
    trunk = FakeTrunk()
    prior = fake_prior(21)
    idx = np.array([2, 6, 9])
    multi = m2.stream_readout_stats(trunk, sd, prior, SpyCalib(), idx, burn=(0, 10, 20))
    for b in (0, 10, 20):
        one = m2.stream_readout_stats(trunk, sd, prior, SpyCalib(), idx, burn=b)
        assert multi[b]["n"] == one["n"]
        assert torch.allclose(multi[b]["G"], one["G"], atol=1e-6) and torch.allclose(multi[b]["R"], one["R"], atol=1e-6)


def test_lowrank_ridge_full_basis_equals_full_and_is_a_projection():
    sd = synth_session(22)
    trunk = FakeTrunk()
    prior = fake_prior(22)
    st = m2.stream_readout_stats(trunk, sd, prior, SpyCalib(), np.arange(10))
    eye = torch.eye(17, dtype=torch.float64)
    v_full = m2.ridge_v(trunk, st, 0.1, "full")
    assert torch.allclose(m2.ridge_v(trunk, st, 0.1, "lowrank", basis=eye), v_full, atol=1e-9)
    # rank-r: the correction lives in the span of the basis
    q, _ = torch.linalg.qr(torch.randn(17, 4, dtype=torch.float64))
    dv = m2.ridge_v(trunk, st, 0.1, "lowrank", basis=q) - trunk.head().double()
    assert torch.allclose(dv @ (eye - q @ q.T), torch.zeros_like(dv), atol=1e-9)


def test_source_feature_moment_is_symmetric_psd_and_source_only():
    sd = synth_session(23)
    trunk = FakeTrunk()
    world = SimpleNamespace(sessions={"a": sd}, trunk=trunk, priors={"a": fake_prior(23)},
                            train_trials=lambda name: np.arange(0, 30))
    g = m2.source_feature_moment(world)
    assert g.shape == (17, 17) and torch.allclose(g, g.T, atol=1e-9)
    assert float(torch.linalg.eigvalsh(g).min()) > -1e-9
    assert abs(float(g[-1, -1]) - 1.0) < 1e-6          # the constant (bias) feature has unit second moment


def test_r1_stream_sram_accounting():
    full = m2.r1_stream_bytes("full")
    assert full["gram_bytes"] == 257 * 257 * 4 == 264196 and not full["fits_32kb_alone"]
    lr16, lr32 = m2.r1_stream_bytes("lowrank", 16), m2.r1_stream_bytes("lowrank", 32)
    assert lr16["gram_bytes"] == 1024 and lr32["gram_bytes"] == 4096
    assert lr16["sram_total_bytes"] < lr32["sram_total_bytes"] < 6 * 1024
    assert lr16["fixed_basis_fp32_bytes"] == 257 * 16 * 4
    # int8 table (96 x 256 + row scales) plus the compact state stays within the 32 KB budget
    assert m2.table_bytes(256, 96)["int8_plus_row_scales_bytes"] + lr32["sram_total_bytes"] <= 32 * 1024


def test_analyzer_dedups_every_row_file(tmp_path):
    import json as _json
    import sys
    sys.path.insert(0, str(m2.ROOT))
    from scripts import analyze_m2_calib as an

    def row(v, r2, **kw):
        return {"variant": v, "seed": 0, "pool": "pool1", "cseed": 0, "tag": "p0", "target": "t", "B": 8,
                "draw": 0, "r2": r2, **kw}
    runs = tmp_path / "runs/seed0/pool1/c0"
    runs.mkdir(parents=True)
    (runs / "eval_rows.jsonl").write_text("\n".join(_json.dumps(r) for r in
                                                     [row("p0", 0.1), row("p0", 0.5), row("p_prior", 0.2)]) + "\n")
    ui = tmp_path / "uireg/seed0/pool1"
    ui.mkdir(parents=True)
    (ui / "eval_uireg.jsonl").write_text("\n".join(_json.dumps(
        {"seed": 0, "pool": "pool1", "target": "t", "B": 8, "draw": 0, "variant": "ui_reg", "r2": r})
        for r in (0.3, 0.4, 0.45)) + "\n")
    lr = tmp_path / "r1lr_cpu/seed0/pool1/c0"
    lr.mkdir(parents=True)
    (lr / "eval_rows.jsonl").write_text("\n".join(_json.dumps(row("p0_r1s_lr16", r)) for r in (0.7, 0.6)) + "\n")
    df = an.load_rows(tmp_path)
    got = {v: df[df.variant == v].r2.tolist() for v in ("p0", "p_prior", "ui_reg", "p0_r1s_lr16")}
    assert got == {"p0": [0.5], "p_prior": [0.2], "ui_reg": [0.45], "p0_r1s_lr16": [0.6]}


def test_selection_writers_dedup_resumed_episodes(tmp_path):
    import json as _json
    import sys
    sys.path.insert(0, str(m2.ROOT))
    from scripts import run_m2_calib as rc
    base = {"base": "p0", "session": "s", "B": 8, "draw": 0}
    rows = [{**base, "lam": None, "kind": "full", "burn": 0, "r2": 0.5},
            {**base, "lam": 1.0, "kind": "full", "burn": 0, "r2": 0.4},
            {**base, "lam": 1.0, "kind": "full", "burn": 0, "r2": 0.9},      # relaunched chain re-appended
            {**base, "lam": 1.0, "kind": "full", "burn": 0, "r2": 0.9}]
    (tmp_path / "select_r1.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    sel = rc.write_r1_selection(tmp_path)
    assert sel["p0"]["8"]["n_episodes"] == 1 and sel["p0"]["8"]["lam"] == 1.0
    assert abs(sel["p0"]["8"]["source_r2"] - 0.9) < 1e-12
    urows = [{"session": "s", "B": 8, "draw": 0, "lam": 0.0, "lr": 0.01, "steps": 10, "r2": r} for r in (0.1, 0.2, 0.2)]
    (tmp_path / "select_uireg.jsonl").write_text("\n".join(_json.dumps(r) for r in urows) + "\n")
    usel = rc.write_uireg_selection(tmp_path)
    assert usel["8"]["n_episodes"] == 1 and abs(usel["8"]["source_r2"] - 0.2) < 1e-12


def test_analyzer_paired_markdown_header_matches_rows(tmp_path):
    import sys
    import pandas as pd
    sys.path.insert(0, str(m2.ROOT))
    from scripts import analyze_m2_calib as an
    recs = []
    for seed in (0, 1):
        for pool, targets in (("pool1", ("a", "b")), ("pool2", ("c",))):
            for t in targets:
                for draw in range(2):
                    recs.append(dict(variant="x", seed=seed, pool=pool, target=t, B=8, draw=draw,
                                     r2=0.5 + 0.1 * seed + 0.01 * draw))
                    recs.append(dict(variant="y", seed=seed, pool=pool, target=t, B=8, draw=draw, r2=0.4))
    p = an.paired(pd.DataFrame(recs), "x", "y", [0, 1])
    md = an.md_paired(p).splitlines()
    ncols = [l.count("|") for l in md]
    assert len(set(ncols)) == 1, ncols                       # header, separator and rows have equal width
    assert abs(p.iloc[0]["seed_sd_paired"] - np.std([0.105, 0.205], ddof=1)) < 1e-9
    assert p.iloc[0]["n_pools"] == 2 and p.iloc[0]["n_seeds"] == 2
