#!/usr/bin/env python
"""Tables for the M2 causal-first-B run.  Reads results/m2_causal_20261020/rows.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import m2_causal as mc                       # noqa: E402

OUT = ROOT / "results" / "analysis" / "m2_causal_20261020"
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 8: 2.306, 14: 2.145}

# Primary rows.  Sensitivities are reported beside them, not instead of them.
PRIMARY = ("p_prior", "p0", "ridge_cca", "w_alpha", "s1", "combo", "cca_s1",
           "r1_lr16", "r2", "r3", "p0_ui_reg")
GATES = (("s1", "s1_cons"), ("combo", "combo_cons"), ("cca_s1", "cca_s1_cons"))


def _ci(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    n = len(vals)
    if n == 0:
        return dict(mean=np.nan, ci=np.nan, n=0, sd=np.nan)
    mean = float(vals.mean())
    if n == 1:
        return dict(mean=mean, ci=np.nan, n=1, sd=0.0)
    sd = float(vals.std(ddof=1))
    se = sd / np.sqrt(n)
    t = TCRIT.get(n - 1, 1.96)
    return dict(mean=mean, ci=float(t * se), n=n, sd=sd)


def _load() -> pd.DataFrame:
    path = mc.OUT / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    return df


def _cell(df, method, B, latency=0):
    """One number per (seed, target): mean over calibrator seeds.  cseed < 0 is already one row."""
    sub = df[(df.method == method) & (df.B == B) & (df.latency == latency) & ~df.unavailable.fillna(False)]
    if sub.empty:
        return sub
    rows = []
    for (seed, target), g in sub.groupby(["seed", "target"]):
        rows.append({"seed": int(seed), "target": target, "pool": g["pool"].iloc[0], "r2": float(g.r2.mean())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = _load()
    lat0 = df[df.latency == 0]
    # Headline: target mean, target-cluster CI, pool-cluster CI.
    summary = []
    per_target = []
    per_pool = []
    for method in sorted(lat0.method.unique()):
        for B in sorted(lat0.B.unique()):
            cell = _cell(lat0, method, int(B), 0)
            if cell is None or cell.empty:
                continue
            by_target = cell.groupby("target").r2.mean()
            # Pool value: mean over its targets, then mean over seeds.  CI across the 3 pools.
            pool_vals = []
            for pool, g in cell.groupby("pool"):
                # seed means inside the pool, then the pool mean
                seed_means = g.groupby("seed").r2.mean()
                pool_mean = float(g.groupby("target").r2.mean().mean())
                pool_vals.append(pool_mean)
                per_pool.append({"method": method, "B": int(B), "pool": pool, "r2": pool_mean,
                                 "n_seeds": int(seed_means.size), "n_targets": int(g.target.nunique())})
            for target, r2 in by_target.items():
                per_target.append({"method": method, "B": int(B), "target": target,
                                   "pool": mc.POOL_OF_TARGET[target], "r2": float(r2)})
            tc, pc = _ci(by_target.values), _ci(pool_vals)
            summary.append({"method": method, "B": int(B), "latency": 0,
                            "target_mean": tc["mean"], "target_ci95": tc["ci"], "n_targets": tc["n"],
                            "pool_mean": pc["mean"], "pool_ci95": pc["ci"], "n_pools": pc["n"]})
    summary_df = pd.DataFrame(summary).sort_values(["method", "B"])
    summary_df.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_target).to_csv(OUT / "per_target.csv", index=False)
    pd.DataFrame(per_pool).to_csv(OUT / "per_pool.csv", index=False)

    # Gate sensitivity on the target means.
    gate_rows = []
    pt = pd.DataFrame(per_target)
    for base, cons in GATES:
        for B in sorted(pt.B.unique()):
            a = pt[(pt.method == base) & (pt.B == B)].set_index("target").r2
            b = pt[(pt.method == cons) & (pt.B == B)].set_index("target").r2
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            delta = (b.loc[common] - a.loc[common]).astype(float)
            gate_rows.append({"base": base, "cons": cons, "B": int(B),
                              "mean_delta": float(delta.mean()),
                              "n_neg": int((delta < -1e-6).sum()), "n": int(len(delta)),
                              "worst": float(delta.min()), "worst_target": str(delta.idxmin())})
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_sensitivity.csv", index=False)

    # Protocol-change deltas against the stored random-draw means.
    rnd = pd.read_csv(ROOT / "results/analysis/m2_calib_20261016/all_rows.csv.gz")
    # Map stored variant -> causal method.  ui_reg (prior init) compares to ui_reg_legacy.
    mapping = {"p_prior": "p_prior", "p0": "p0", "p0_r1s_lr16": "r1_lr16", "ui_reg": "ui_reg_legacy"}
    deltas = []
    for variant, method in mapping.items():
        sub = rnd[rnd.variant == variant]
        for B, g in sub.groupby("B"):
            # draw/cseed mean inside (seed, target), then target mean of seed means
            cell = g.groupby(["seed", "target"]).r2.mean().reset_index()
            by_t = cell.groupby("target").r2.mean()
            causal = pt[(pt.method == method) & (pt.B == int(B))].set_index("target").r2
            common = by_t.index.intersection(causal.index)
            if len(common) == 0:
                continue
            d = causal.loc[common] - by_t.loc[common]
            deltas.append({"variant": variant, "method": method, "B": int(B),
                           "random_mean": float(by_t.loc[common].mean()),
                           "causal_mean": float(causal.loc[common].mean()),
                           "delta": float(d.mean()), "n_targets": int(len(common)),
                           "n_seeds_random": int(cell.seed.nunique())})
    pd.DataFrame(deltas).to_csv(OUT / "protocol_deltas.csv", index=False)

    # Latency 1 minus latency 0, target means.
    lat_rows = []
    for method in PRIMARY:
        for B in sorted(df.B.unique()):
            a = _cell(df, method, int(B), 0)
            b = _cell(df, method, int(B), 1)
            if a.empty or b.empty:
                continue
            aa = a.groupby("target").r2.mean()
            bb = b.groupby("target").r2.mean()
            common = aa.index.intersection(bb.index)
            if len(common) == 0:
                continue
            lat_rows.append({"method": method, "B": int(B),
                             "latency1_minus_0": float((bb.loc[common] - aa.loc[common]).mean())})
    pd.DataFrame(lat_rows).to_csv(OUT / "latency.csv", index=False)
    print(f"[analyze] {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
