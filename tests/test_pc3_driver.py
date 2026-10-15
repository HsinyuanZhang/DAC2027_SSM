"""Driver-level tests for scripts/run_pc3_refine.py (no GPU, no recordings)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pc3_refine as pc                                  # noqa: E402
from ssm_decode import fwd_refine as fr                                   # noqa: E402


def test_draw_sim_uses_only_trials_inside_window():
    sd = SimpleNamespace(trials=[(i * 100, i * 100 + 90) for i in range(900)])
    n_in = sum(1 for a, b in sd.trials if b <= pc.SIM_MAX_BINS)
    for B in (1, 8, 33):
        idx = pc.draw_sim("s", "t", B, 0, sd)
        assert len(idx) == B and idx.max() < n_in
        assert all(sd.trials[i][1] <= pc.SIM_MAX_BINS for i in idx)


def test_write_selection_picks_best_config_per_method_and_B(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "OUT", tmp_path)
    p = pc.sel_path("days", "oos", "r12")
    p.parent.mkdir(parents=True)
    rows = []
    for ses in ("a", "b"):
        for B in (1, 8):
            for lam, r2 in ((0.1, 0.5 + 0.1 * (B == 8)), (1.0, 0.6), (10.0, 0.4)):
                rows.append({"stratum": "days", "set": "oos", "session": ses, "seed": 0, "B": B, "draw": 0,
                             "method": "r1_full", "cfg": {"lam": lam}, "r2": r2})
            rows.append({"stratum": "days", "set": "oos", "session": ses, "seed": 0, "B": B, "draw": 0,
                         "method": "r2", "cfg": {"L": 8, "eta": 1.0, "rho": 0.0, "S": 1}, "r2": None})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    sel = pc.write_selection("days", "oos", "r12")
    assert sel["r1_full"]["1"]["cfg"]["lam"] == 1.0 and sel["r1_full"]["8"]["cfg"]["lam"] == 0.1 \
        or sel["r1_full"]["8"]["cfg"]["lam"] == 1.0
    assert sel["r1_full"]["1"]["n_episodes"] == 2
    # non-finite episodes (None) are scored -1, never silently dropped
    assert sel["r2"]["1"]["source_r2"] == -1.0


def test_stream_stats_matches_manual_prefix_construction():
    """Streaming accumulation == per-trial forward with the table/norm of the previous trials."""
    from test_fwd_refine import LinTrunk, K, N, D                         # noqa: F401
    base = LinTrunk(3)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(120, N, generator=g, dtype=torch.float64) * 3
    y = torch.randn(120, D, generator=g, dtype=torch.float64)
    valid = torch.ones(120, dtype=torch.bool)
    trials = [(0, 40), (40, 80), (80, 120)]
    xz = (x - x.mean(0)) / x.std(0)
    X, Y, M = fr.pad_trials(xz, y, valid, trials)
    T = torch.randn(K, N, generator=g, dtype=torch.float64) * 0.2
    whole = fr.readout_stats(base, X, Y, M, T)
    parts = [fr.readout_stats(base, X[i:i + 1], Y[i:i + 1], M[i:i + 1], T) for i in range(3)]
    tot = parts[0]
    for q in parts[1:]:
        tot = fr.add_readout_stats(tot, q)
    torch.testing.assert_close(tot["G"], whole["G"])
    torch.testing.assert_close(tot["R"], whole["R"])


def test_stream_stats_uses_only_previous_trials_for_each_trial_and_one_forward_each(monkeypatch):
    """Trial j is forwarded once with the table/normalisation built from trials < j."""
    from test_population_calib import _session
    from ssm_decode.population_calib import norm_stats
    sd, prior = _session(2, n=7, n_trials=6, trial_len=60)
    K = prior.shape[0]
    base = SimpleNamespace(out_proj=None)
    world = SimpleNamespace(adapter=SimpleNamespace(model=SimpleNamespace(base=base)))
    seen_prefix = []

    class FakeCalib:
        def calibrate(self, stats, prior):
            seen_prefix.append(stats.n_trials)
            n = float(stats.d["n"])
            mean, std = norm_stats(stats)
            return SimpleNamespace(table=prior + 0.001 * n, mean=mean, std=std)
    calls = []

    def fake_readout_stats(b, X, Y, M, table):
        calls.append((X.clone(), table.clone()))
        return {"G": torch.zeros(2, 2, dtype=torch.float64), "R": torch.zeros(1, 2, dtype=torch.float64), "n": float(M.sum())}
    monkeypatch.setattr(pc.fr, "readout_stats", fake_readout_stats)
    idx = np.array([4, 1, 2, 0])
    st = pc.stream_stats(world, sd, prior, FakeCalib(), idx)
    order = np.sort(idx)
    assert len(calls) == 4                                          # exactly one forward per trial
    assert seen_prefix == [1, 2, 3]                                 # table for trial j from trials < j only
    n_bins = [float(sd.bank.t["n"][i]) for i in order]
    assert st["n"] == pytest.approx(sum(n_bins))                    # every trial counted exactly once
    for j in range(1, 4):                                           # table of trial j depends on prefix j only
        n_prefix = float(sum(n_bins[:j]))
        torch.testing.assert_close(calls[j][1], (prior + 0.001 * n_prefix).float())
    torch.testing.assert_close(calls[0][1], prior.float())          # trial 0: prior table


# ---------------------------------------------------- review-fix additions ----

def _fake_world(stratum):
    from scripts.run_labeled_calibration import D1_POOLS
    spec = D1_POOLS[stratum]
    names = [f"sub-M_ses-CO-{d}" for d in ("20140307", "20140627", "20140929", "20141203",
                                            "20140203", "20140217", "20140218", "20140303",
                                            "20140304", "20140306")]
    return SimpleNamespace(orig_ids=spec["sources"].split(","), ids=spec["sources"].split(","),
                           sessions={n: None for n in names})


def test_causal_oos_set_for_month_is_only_sessions_before_the_target():
    w = _fake_world("month")
    full = pc.sim_sessions(w, "month", "oos")
    causal = pc.sim_sessions(w, "month", "oosc")
    assert sorted(n[-8:] for n in full) == ["20140307", "20140627", "20140929", "20141203"]
    assert [n[-8:] for n in causal] == ["20140307"]                 # 0627 is the day AFTER the target
    wy = _fake_world("year")
    assert pc.sim_sessions(wy, "year", "oosc") == pc.sim_sessions(wy, "year", "oos")   # already causal


def test_variant_switch_changes_file_names_and_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "OUT", tmp_path)
    pc.set_variant("base")
    assert pc.sel_path("month", "oosc", "r12").name == "r12_month_oosc.jsonl"
    pc.set_variant("cap256", "base")
    try:
        assert pc.sel_path("month", "oosc", "r12").name == "r12_month_oosc_cap256.jsonl"
        assert pc.CFG["cap"] == pc.CAP_BINS and not pc.CFG["contig"]
        assert pc.selection_file("month", "oosc", pc.CFG["sel_variant"]).name == "selection_month_oosc.json"
        pc.set_variant("contig")
        assert pc.CFG["contig"] and pc.CFG["cap"] is None
        pc.set_variant("ext")
        assert pc.CFG["ext"] and pc.selection_file("year", "oos").name == "selection_year_oos_ext.json"
    finally:
        pc.set_variant("base")


def test_ep_replay_modes_cap_and_contig(monkeypatch):
    from test_fwd_refine import LinTrunk, K, N, D
    base = LinTrunk(1)
    g = torch.Generator().manual_seed(0)
    nb = 400
    sd = SimpleNamespace(x32=torch.rand(nb, N, generator=g, dtype=torch.float64), y_norm=torch.randn(nb, D, generator=g, dtype=torch.float64),
                         valid=torch.ones(nb, dtype=torch.bool), n_bins=nb, name="s",
                         trials=[(0, 50), (60, 90), (100, 300), (320, 360)],
                         query_mask=lambda idx: torch.ones(nb, dtype=torch.bool))
    world = SimpleNamespace(adapter=SimpleNamespace(model=SimpleNamespace(base=base)))
    T0 = torch.zeros(K, N, dtype=torch.float64)
    mean, std = torch.zeros(N, dtype=torch.float64), torch.ones(N, dtype=torch.float64)
    idx = np.array([0, 2, 3])
    ep = pc.Ep(world, sd, idx, T0, mean, std, cap=None, contig=False)
    assert ep.X.shape[:2] == (3, 200) and ep.replay_bins == 50 + 200 + 40
    ep = pc.Ep(world, sd, idx, T0, mean, std, cap=100, contig=False)
    assert ep.X.shape[:2] == (3, 100) and ep.replay_bins == 50 + 100 + 40 and ep.replay_bins_uncapped == 290
    assert ep.trial_lens == [50, 200, 40]
    ep = pc.Ep(world, sd, idx, T0, mean, std, cap=None, contig=True)
    assert ep.X.shape == (1, 290, N) and ep.replay_bins == 290


def test_uiref_helpers(tmp_path, monkeypatch):
    from scripts import run_pc3_uiref as ur
    assert ur.max_steps(33, 1e-3) == 1000 and ur.max_steps(33, 1e-2) == 300 and ur.max_steps(4, 1e-3) == 300
    assert ur.tag_sessions("month", "oosc") == ("oos", "20140626")
    assert ur.tag_sessions("year", "is0") == ("is", None)
    monkeypatch.setattr(pc, "OUT", tmp_path)
    rows = []
    for ses, r in (("sub-M_ses-CO-20140307", 0.5), ("sub-M_ses-CO-20140627", 0.9)):
        for B in (1, 8):
            rows.append({"stratum": "month", "set": "oos", "session": ses, "seed": 0, "B": B, "draw": 0,
                         "lam": None, "lr": None, "steps": 0, "r2": 0.3})
            for lam, rr in ((0.0, r), (1.0, 0.1)):
                rows.append({"stratum": "month", "set": "oos", "session": ses, "seed": 0, "B": B, "draw": 0,
                             "lam": lam, "lr": 1e-3, "steps": 10, "r2": rr})
    (ur.uidir() / "select_month_oos_x.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ur.cmd_write(SimpleNamespace(stratum="month", tag="oosc"))
    sel = json.loads((ur.uidir() / "selection_month_oosc.json").read_text())
    assert sel["8"]["lam"] == 0.0 and sel["8"]["n_episodes"] == 1 and sel["8"]["sessions"] == ["20140307"]
    ur.cmd_write(SimpleNamespace(stratum="month", tag="oos"))
    sel = json.loads((ur.uidir() / "selection_month_oos.json").read_text())
    assert sel["8"]["n_episodes"] == 2 and sel["8"]["source_r2"] == pytest.approx(0.7)


def test_uiref_run_traj_moves_the_table_and_respects_snapshots():
    """ui_reg from the P0 table on a toy trunk: start = centre, snapshots returned, lam=large pins it."""
    from scripts import run_pc3_uiref as ur
    from test_fwd_refine import LinTrunk, K, N, D
    base = LinTrunk(1).float()
    g = torch.Generator().manual_seed(0)
    nb = 120
    sd = SimpleNamespace(x32=torch.randn(nb, N, generator=g), y_norm=torch.randn(nb, D, generator=g),
                         valid=torch.ones(nb, dtype=torch.bool), n_bins=nb, name="s",
                         trials=[(0, 40), (40, 80), (80, 120)], query_mask=lambda idx: torch.ones(nb, dtype=torch.bool))
    world = SimpleNamespace(adapter=SimpleNamespace(model=SimpleNamespace(base=base)), device="cpu")
    T0 = torch.randn(K, N, generator=g) * 0.2
    ep = pc.Ep(world, sd, np.array([0, 1, 2]), T0, torch.zeros(N), torch.ones(N), cap=None, contig=False)
    tabs = ur.run_traj(world, sd, ep, 0.0, 1e-2, (3, 10))
    assert sorted(tabs) == [3, 10]
    assert np.abs(tabs[10] - T0.numpy()).max() > 1e-4
    pinned = ur.run_traj(world, sd, ep, 1e6, 1e-6, (10,))[10]
    assert np.abs(pinned - T0.numpy()).max() < 1e-3
