#!/usr/bin/env python
"""P3 review fix, CPU-only analyses (no trunk, no GPU).

  1. Per-draw replay statistics of the REAL evaluation draws (items 3, 7): replayed bins, longest
     trial, first-arriving trial length, for every (stratum, target, B, draw) of the P3 protocol, plus
     a Monte-Carlo over 2000 random draws and the worst case (the B longest usable trials).
  2. The per-unit preconditioner check (item 5): q_i = mean over valid replayed bins of x_i^2 after
     z-scoring on the same B trials; reports max_i |q_i - 1| over all draws.
  3. Item 7: for every eval cell, the share of the replay bins supplied by the longest trial and the
     R2 / R3 - P0 delta of the same draw, to show whether the collapses coincide with outlier trials.

Writes results/analysis/population_calib_p3_20261016/{replay_draw_stats.csv, replay_mc_summary.csv,
precond_check.json, long_trial_vs_delta.csv}.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402
from ssm_decode.dandi688_data import load_dandi688_recording              # noqa: E402
from ssm_decode.population_calib import PopStats, SessionData, norm_stats  # noqa: E402
from ssm_decode.track_a_experiment import usable_trials                    # noqa: E402

OUTD = ROOT / "results/analysis/population_calib_p3_20261016"
P3 = ROOT / "results/population_calib_p3_20261016"
BS = (1, 2, 4, 8, 16, 33)
N_DRAWS = 8
MC = 2000
CAP = 256                  # fixed-length replay design (run_pc3_refine.CAP_BINS)


def draw_trials(name, B, draw, n_trials):          # identical to run_population_calib.draw_trials
    seed = zlib.crc32(f"{name}-{B}-{draw}".encode())
    return np.sort(np.random.default_rng(seed).choice(n_trials, size=B, replace=False))


def label_stats(stratum):
    blob = torch.load(ROOT / D1_POOLS[stratum]["ap"], map_location="cpu", weights_only=False)
    return tuple(np.asarray(s, dtype=np.float64) for s in blob["source_statistics"][0][2:4])


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, mc_rows, qdev = [], [], {}
    for stratum in ("days", "month", "year"):
        ls = label_stats(stratum)
        for target in D1_POOLS[stratum]["targets"]:
            rec = load_dandi688_recording(target)
            sd = SessionData(rec, ls, torch.device("cpu"), name=target)
            lens = np.array([b - a for a, b in sd.trials])
            n_units = sd.n_units
            for B in BS:
                for draw in range(N_DRAWS):
                    idx = draw_trials(target, B, draw, len(sd.trials))
                    mean, std = norm_stats(PopStats.from_trials(sd.bank, idx))
                    # q_i over the valid bins of the replayed trials
                    sel = torch.zeros(sd.n_bins, dtype=torch.bool)
                    for i in idx:
                        a, b = sd.trials[i]
                        sel[a:b] = True
                    sel &= sd.valid
                    xz = (sd.x32[sel] - mean.float()) / std.float()
                    q = (xz ** 2).mean(0)
                    # units at the std floor (std set to 1: silent units in these B trials) have q = var
                    s_raw = (sd.x32[sel].double().std(0, unbiased=False) < 1e-6)
                    dev = float((q - 1).abs()[~s_raw].max())
                    qdev.setdefault(stratum, []).append(dev)
                    L = lens[idx]
                    rows.append({"stratum": stratum, "target": target, "N": n_units, "B": B, "draw": draw,
                                 "replay_bins": int(L.sum()), "mean_trial_bins": float(L.mean()),
                                 "max_trial_bins": int(L.max()), "first_trial_bins": int(L[0]),
                                 "replay_bins_cap": int(np.minimum(L, CAP).sum()),
                                 "n_trials_over_cap": int((L > CAP).sum()),
                                 "share_longest": float(L.max() / L.sum()),
                                 "median_len_session": float(np.median(lens)),
                                 "longest_over_median": float(L.max() / np.median(lens)),
                                 "q_dev_max": dev, "n_silent_units": int(s_raw.sum())})
                # Monte-Carlo over random draws (same sampler family, different seeds)
                rng = np.random.default_rng(zlib.crc32(f"mc-{target}-{B}".encode()))
                tot, first, totc = [], [], []
                for _ in range(MC):
                    idx = np.sort(rng.choice(len(lens), size=B, replace=False))
                    tot.append(lens[idx].sum())
                    totc.append(np.minimum(lens[idx], CAP).sum())
                    first.append(lens[idx[0]])
                tot, first, totc = np.array(tot), np.array(first), np.array(totc)
                mc_rows.append({"stratum": stratum, "target": target, "N": n_units, "B": B,
                                "n_usable_trials": len(lens), "median_trial_bins": float(np.median(lens)),
                                "mean_trial_bins": float(lens.mean()),
                                "max_trial_bins_session": int(lens.max()),
                                "replay_bins_mc_median": float(np.median(tot)),
                                "replay_bins_mc_p95": float(np.percentile(tot, 95)),
                                "replay_bins_mc_max": int(tot.max()),
                                "replay_bins_worst_case": int(np.sort(lens)[-B:].sum()),
                                "replay_bins_cap_mc_median": float(np.median(totc)),
                                "replay_bins_cap_mc_max": int(totc.max()),
                                "replay_bins_cap_worst_case": int(np.minimum(np.sort(lens)[-B:], CAP).sum()),
                                "frac_trials_over_cap": float((lens > CAP).mean()),
                                "replay_bins_mean_T": float(B * lens.mean()),
                                "first_trial_mc_p95": float(np.percentile(first, 95)),
                                "first_trial_mc_max": int(first.max())})
            print(f"[cpu] {stratum} {target[-8:]} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUTD / "replay_draw_stats.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(OUTD / "replay_mc_summary.csv", index=False)
    (OUTD / "precond_check.json").write_text(json.dumps(
        {s: {"n_draw_cells": len(v), "max_abs_q_minus_1": float(np.max(v)),
             "median_max_abs_q_minus_1": float(np.median(v))} for s, v in qdev.items()}, indent=1))
    # item 7: longest-trial share vs per-draw R2 / R3 delta (headline sets)
    head = {"days": "is", "month": "oos", "year": "oos"}
    ev = []
    for s, st in head.items():
        f = P3 / f"eval/eval_{s}_{st}.jsonl"
        e = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
        e = e[e.method.isin(["p0", "r2", "r3_full", "r3_lr16"])]
        e = e.groupby(["stratum", "target", "B", "draw", "method"]).r2.mean().unstack("method").reset_index()
        e["d_r2"] = e.r2 - e.p0
        ev.append(e)
    ev = pd.concat(ev)
    m = ev.merge(d, on=["stratum", "target", "B", "draw"])
    m[["stratum", "target", "B", "draw", "replay_bins", "max_trial_bins", "share_longest", "longest_over_median",
       "p0", "r2", "d_r2"]].to_csv(OUTD / "long_trial_vs_delta.csv", index=False)
    print(json.loads((OUTD / "precond_check.json").read_text()))


if __name__ == "__main__":
    main()
