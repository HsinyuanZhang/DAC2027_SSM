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
