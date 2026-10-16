#!/usr/bin/env python
"""Aggregate hybrid-identity v2 rows into csv tables and figures.

Unit of analysis = target session (12 targets share one trunk): every cell is first averaged over
draws and seeds per (key, frac, target, B); across-target intervals are t intervals (df = n_targets-1).
They do not cover trunk / calibrator-training variance; the per-seed across-target means (3 seeds) are
reported next to each interval as the seed spread.

key = "cond/variant" for the v2 calibrators (cond in cut / coupled); the reused P0 is "P0".
P0 reference value of a cell = mean of its two seeds (P1 pool-O seeds 0/1; no third seed exists).

Metrics of a key v (per (frac, target, B, draw, seed)):
  total  R2(v) - R2(P0)                                   (P0: mean of its two seeds)
  unc    R2(covered cols = P0 seed0, uncovered cols = v) - R2(P0 seed0)
  cov    R2(covered cols = v, uncovered cols = P0 seed0) - R2(P0 seed0)
  attr   R2(v) - R2(covered cols = v, uncovered cols = P0 seed0)   (in-context attribution of v's
         uncovered columns; the 'swap_cov' complement)
  value  R2(v) - R2(v with uncovered columns zeroed)
For a pair (a, b) every metric is metric(a) - metric(b) per row (so for unc/attr the shared P0 base
cancels).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/hybrid_identity_v2_20261017"
OUT = ROOT / "results/analysis/hybrid_identity_v2_20261017"
B_LIST = (1, 2, 4, 8, 16, 33)
METRICS = ("total", "unc", "cov", "attr", "value")
LABEL = {"P0": "P0 (reused, no masked training)",
         "h0m": "P0 + masked training", "h1": "P0 + masked + flag", "h1w137": "wide+flag (137, ~XATT)",
         "h1w153": "wide+flag (153, ~PMA)", "h2p": "PMA v1 init", "h2pf": "PMA fixed",
         "h2pf_loc": "PMA fixed, per-unit control", "hx": "XATT (pairwise)",
         "hx_np": "XATT control (no pairwise)", "hxs": "XATT, standardised pair input",
         "hxs_np": "XATT-std control (no pairwise)"}

PAIRS = [   # (a, b, role)   a, b are "cond/variant" or "P0"
    ("cut/h0m", "P0", "masked training, cut"), ("cut/h1", "P0", "flag+masked, cut"),
    ("cut/h1", "cut/h0m", "flag alone"),
    ("cut/hx", "P0", "XATT vs P0"), ("cut/hx_np", "P0", "XATT-control vs P0"),
    ("cut/h2pf", "P0", "PMA-fixed vs P0"), ("cut/h2pf_loc", "P0", "PMA-fixed-control vs P0"),
    ("cut/h2p", "P0", "PMA-v1 vs P0"), ("cut/h1w137", "P0", "wide137 vs P0"), ("cut/h1w153", "P0", "wide153 vs P0"),
    ("cut/hx", "cut/hx_np", "XATT vs capacity control (no pairwise)"),
    ("cut/hx", "cut/h1w137", "XATT vs wide"), ("cut/hx_np", "cut/h1w137", "XATT-control vs wide"),
    ("cut/hx", "cut/h1", "XATT vs flag-only"), ("cut/hx", "cut/h0m", "XATT vs like-for-like P0"),
    ("cut/hxs", "P0", "XATT-std vs P0"), ("cut/hxs_np", "P0", "XATT-std-control vs P0"),
    ("cut/hxs", "cut/hxs_np", "XATT-std vs capacity control (no pairwise)"),
    ("cut/hxs", "cut/h1w137", "XATT-std vs wide"), ("cut/hxs", "cut/h1", "XATT-std vs flag-only"),
    ("cut/hxs", "cut/hx", "standardised vs raw pair input"),
    ("cut/h2pf", "cut/h2pf_loc", "PMA-fixed vs per-unit control"),
    ("cut/h2pf", "cut/h1w153", "PMA-fixed vs wide"), ("cut/h2pf", "cut/h1", "PMA-fixed vs flag-only"),
    ("cut/h2pf", "cut/h2p", "fixed vs v1 PMA init"),
    ("cut/h2p", "cut/h2pf_loc", "PMA-v1 vs per-unit control"),
    ("cut/h2p", "cut/h1w153", "PMA-v1 vs wide"), ("cut/h2p", "cut/h1", "PMA-v1 vs flag-only"),
    ("coupled/h1", "P0", "flag+masked, coupled"),
    ("coupled/hx", "coupled/hx_np", "XATT vs control (coupled)"),
    ("coupled/h2pf", "coupled/h2pf_loc", "PMA-fixed vs control (coupled)"),
    ("coupled/hx", "coupled/h1", "XATT vs flag-only (coupled)"),
    ("coupled/h2pf", "coupled/h1", "PMA-fixed vs flag-only (coupled)"),
    ("coupled/h1", "cut/h1", "value of descriptor coupling (flag-only)"),
    ("coupled/hx", "cut/hx", "value of descriptor coupling (XATT)"),
    ("coupled/hx_np", "cut/hx_np", "value of descriptor coupling (XATT control)"),
    ("coupled/h2pf", "cut/h2pf", "value of descriptor coupling (PMA fixed)"),
    ("coupled/h2pf_loc", "cut/h2pf_loc", "value of descriptor coupling (PMA control)"),
]
MAIN_KEYS = ["P0", "cut/h0m", "cut/h1", "cut/h1w137", "cut/h1w153", "cut/hx_np", "cut/hx", "cut/hxs_np",
             "cut/hxs", "cut/h2pf_loc", "cut/h2pf", "cut/h2p"]


def load_rows():
    rows = []
    for f in sorted((RAW / "eval").glob("rows_*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["frac"] = df["frac"].map(lambda x: "nat" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:g}")
    df["key"] = np.where(df.variant == "h0", "P0", df.cond + "/" + df.variant)
    kk = ["key", "seed", "target", "frac", "B", "draw"]
    dup = df.duplicated(kk, keep="first")
    # P0 rows are re-computed by every eval invocation: identical by construction, check and drop
    first = df[~dup].set_index(kk)
    for _, r in df[dup].iterrows():
        assert r.key == "P0", r
        assert abs(first.loc[tuple(r[k] for k in kk)].r2 - r.r2) < 1e-4
    return df[~dup].reset_index(drop=True)


def ci_t(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def wide(df):
    """Model rows with metric columns (m), P0 rows with metric columns (p), and the per-cell P0 base."""
    idx = ["frac", "target", "unc_frac", "B", "draw"]
    p0 = df[df.key == "P0"]
    p0s0 = p0[p0.seed == 0].set_index(idx)
    base = pd.concat([p0.groupby(idx).r2.mean().rename("r2_p0mean"), p0s0["r2"].rename("r2_base"),
                      p0s0["r2_zero_unc"].rename("r2_p0_zero"), p0s0["r2_prior"].rename("r2_prior")],
                     axis=1).reset_index()
    m = df[df.key != "P0"].drop(columns=["r2_base"]).merge(base, on=idx, how="left")
    m["total"] = m.r2 - m.r2_p0mean
    m["unc"] = m.r2_swap_unc - m.r2_base
    m["cov"] = m.r2_swap_cov - m.r2_base
    m["attr"] = m.r2 - m.r2_swap_cov
    m["value"] = m.r2 - m.r2_zero_unc
    p = p0[idx + ["seed", "r2"]].merge(base, on=idx, how="left")
    p["total"] = p.r2 - p.r2_p0mean
    for c in ("unc", "cov", "attr"):
        p[c] = 0.0
    p["value"] = np.where(p.seed == 0, p.r2 - p.r2_p0_zero, np.nan)
    return m, p, base


def pair_series(m, p, a, b, metric):
    """Per (frac, target, B, draw, seed) difference series for the pair (a, b)."""
    idx = ["frac", "target", "unc_frac", "B", "draw"]

    def get(k):
        if k == "P0":
            d = p[p.seed == 0] if metric != "total" else p
            if metric == "total":                       # seed-averaged P0 -> 0 by construction
                d = p.groupby(idx).agg(**{metric: (metric, "mean")}).reset_index()
                d["seed"] = -1
            return d[idx + ["seed", metric]]
        d = m[m.key == k]
        return d[idx + ["seed", metric]]
    ga, gb = get(a), get(b)
    if a == "P0" or b == "P0":
        j = ga.merge(gb.drop(columns="seed"), on=idx, suffixes=("_a", "_b")) if b == "P0" else \
            ga.drop(columns="seed").merge(gb, on=idx, suffixes=("_a", "_b"))
    else:
        j = ga.merge(gb, on=idx + ["seed"], suffixes=("_a", "_b"))
    j["d"] = j[f"{metric}_a"] - j[f"{metric}_b"]
    return j


def paired_table(m, p):
    out = []
    for a, b, role in PAIRS:
        if (a != "P0" and a not in set(m.key)) or (b != "P0" and b not in set(m.key)):
            continue
        for metric in METRICS:
            if metric == "value" and (a == "P0" or b == "P0"):
                pass
            j = pair_series(m, p, a, b, metric)
            if j.empty:
                continue
            for frac, jf in j.groupby("frac"):
                slices = {f"B={B}": jf[jf.B == B] for B in sorted(jf.B.unique())}
                bs = sorted(jf.B.unique())
                if len(bs) > 1:
                    slices["B>=2"] = jf[jf.B >= 2]
                    slices["Bavg"] = jf
                for sl, js in slices.items():
                    if js.empty:
                        continue
                    per_t = js.groupby("target").d.mean()
                    mean, lo, hi = ci_t(per_t.values)
                    rec = {"frac": frac, "slice": sl, "a": a, "b": b, "role": role, "metric": metric,
                           "mean": mean, "lo": lo, "hi": hi, "n_pos": int((per_t > 0).sum()),
                           "n_targets": len(per_t),
                           "unc_frac_mean": float(js.groupby("target").unc_frac.first().mean())}
                    ps = js.groupby(["seed", "target"]).d.mean().groupby("seed").mean()
                    for s, v in ps.items():
                        if s >= 0:
                            rec[f"seed{s}"] = float(v)
                    if len(ps) > 1 and (ps.index >= 0).all():
                        rec["seed_min"], rec["seed_max"] = float(ps.min()), float(ps.max())
                    out.append(rec)
    return pd.DataFrame(out)


def cells_table(df):
    d = df.groupby(["key", "frac", "target", "B"]).agg(r2=("r2", "mean"), unc_frac=("unc_frac", "first")).reset_index()
    g = d.groupby(["key", "frac", "B"]).agg(r2_mean=("r2", "mean"), n_targets=("target", "nunique"),
                                            unc_frac=("unc_frac", "mean")).reset_index()
    lo_hi = d.groupby(["key", "frac", "B"]).r2.apply(lambda x: pd.Series(ci_t(x.values), index=["m", "lo", "hi"]))
    lo_hi = lo_hi.unstack().reset_index()
    g = g.merge(lo_hi[["key", "frac", "B", "lo", "hi"]], on=["key", "frac", "B"])
    return g


def headroom_table(m, p, base, pattern="oracle_*.jsonl"):
    path = sorted((RAW / "eval").glob(pattern))
    if not path:
        return None, None
    orc = pd.DataFrame([json.loads(x) for f in path for x in f.read_text().splitlines() if x.strip()])
    orc["frac"] = orc["frac"].map(lambda x: "nat" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:g}")
    idx = ["frac", "target", "B", "draw"]
    swap = m.groupby(["key"] + idx).agg(swap_unc=("r2_swap_unc", "mean"), r2=("r2", "mean"),
                                        unc_frac=("unc_frac", "first")).reset_index()
    j = swap.merge(orc[idx + ["r2_p0", "r2_zero_unc", "r2_oracle"]], on=idx)
    j["gain_v"] = j.swap_unc - j.r2_p0
    j["gain_or"] = j.r2_oracle - j.r2_p0
    j["val_or"] = j.r2_oracle - j.r2_zero_unc
    j["val_v"] = j.swap_unc - j.r2_zero_unc
    rows = []
    for (key, frac, B), g in j.groupby(["key", "frac", "B"]):
        pt = g.groupby("target")[["gain_v", "gain_or", "val_v", "val_or"]].mean()
        if len(pt) < 2:
            continue
        rows.append({"key": key, "frac": frac, "B": B, "n_targets": len(pt),
                     "gain_v": pt.gain_v.mean(), "gain_oracle": pt.gain_or.mean(),
                     "headroom_vs_P0_cols": pt.gain_or.mean(),
                     "captured_vs_P0": pt.gain_v.mean() / pt.gain_or.mean() if pt.gain_or.mean() > 1e-6 else np.nan,
                     "value_v": pt.val_v.mean(), "value_oracle": pt.val_or.mean(),
                     "captured_of_value": pt.val_v.mean() / pt.val_or.mean() if pt.val_or.mean() > 1e-6 else np.nan,
                     "ci_gain_v": "[%.3f, %.3f]" % tuple(ci_t(pt.gain_v.values)[1:])})
    return pd.DataFrame(rows), orc


def training_diag():
    rows, curves = [], []
    for done in sorted((RAW / "runs").glob("*/*/seed*/done.json")):
        rd = done.parent
        cond, name, seed = rd.parent.parent.name, rd.parent.name, int(rd.name[4:])
        d = json.loads(done.read_text())
        logs = [json.loads(x) for x in (rd / "train_log.jsonl").read_text().splitlines() if x.strip()]
        best = [r for r in logs if r["step"] == d["best_step"]]
        rec = {"cond": cond, "variant": name, "seed": seed, "n_params": d["n_params"],
               "best_step": d["best_step"], "best_val_r2": d["best_val_r2"], "converged": d["converged"],
               "last_step": d["last_step"], "seconds": d["seconds"]}
        if best and best[0].get("diag"):
            rec.update({f"best_{k}": v for k, v in best[0]["diag"].items()})
        if logs and logs[0].get("diag"):
            rec.update({f"first_{k}": v for k, v in logs[0]["diag"].items()})
        if logs and logs[-1].get("diag"):
            rec.update({f"last_{k}": v for k, v in logs[-1]["diag"].items()})
        rows.append(rec)
        for r in logs:
            curves.append({"cond": cond, "variant": name, "seed": seed, "step": r["step"], "val_r2": r["val_r2"],
                           **{k: v for k, v in (r.get("diag") or {}).items()}})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def eval_diag(df):
    d = df[df.diag.notna()].copy()
    if d.empty:
        return None
    recs = []
    for _, r in d.iterrows():
        recs.append({"key": r.key, "frac": r.frac, "B": r.B, "seed": r.seed, **r.diag})
    t = pd.DataFrame(recs)
    return t.groupby(["key", "frac", "B"]).mean(numeric_only=True).drop(columns="seed").reset_index()


def make_figures(cells, paired, hr, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"P0": "#222222", "cut/h0m": "#8c8c8c", "cut/h1": "#1b9e77", "cut/hx": "#d95f02", "cut/hx_np": "#e6ab02",
           "cut/h2pf": "#7570b3", "cut/h2pf_loc": "#66a61e", "cut/h1w137": "#a6761d", "cut/h2p": "#e7298a",
           "cut/hxs": "#b2182b", "cut/hxs_np": "#6a3d9a"}
    xs = ["nat", "0.5", "0.7", "0.9"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for B, a in zip((2, 8, 33), ax):
        for k, c in pal.items():
            ys = []
            for f in xs:
                s = cells[(cells.key == k) & (cells.frac == f) & (cells.B == B)]
                ys.append(s.r2_mean.iloc[0] if len(s) else np.nan)
            a.plot(range(len(xs)), ys, "-o", color=c, label=LABEL.get(k.split("/")[-1], k), lw=1.6, ms=4)
        a.set_xticks(range(len(xs)))
        a.set_xticklabels(["natural\n(0.08-0.50)", "0.5", "0.7", "0.9"])
        a.set_title(f"B={B}")
        a.set_xlabel("target uncovered fraction (simulated at eval)")
        a.set_ylabel("query R2 (12-target mean)")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "r2_vs_uncovered_fraction.png", dpi=150)
    fig.savefig(OUT / "r2_vs_uncovered_fraction.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sets = [("cut/hx", "cut/hx_np", "XATT - control (no pairwise)", "#d95f02"),
            ("cut/hx", "cut/h1w137", "XATT - wide", "#a6761d"),
            ("cut/h2pf", "cut/h2pf_loc", "PMA-fixed - per-unit", "#7570b3"),
            ("cut/h2pf", "cut/h1w153", "PMA-fixed - wide", "#66a61e"),
            ("cut/hx", "cut/h1", "XATT - flag-only", "#1b9e77")]
    for a, (met, ttl) in zip(ax, (("total", "total R2 difference (Bavg)"), ("unc", "uncovered-column effect (Bavg)"),
                                  ("attr", "in-context attribution (Bavg)"))):
        for i, (ka, kb, lab, c) in enumerate(sets):
            ys, lo, hi = [], [], []
            for f in xs:
                s = paired[(paired.a == ka) & (paired.b == kb) & (paired.metric == met) & (paired.frac == f)
                           & (paired.slice == "Bavg")]
                ys.append(s["mean"].iloc[0] if len(s) else np.nan)
                lo.append(s["lo"].iloc[0] if len(s) else np.nan)
                hi.append(s["hi"].iloc[0] if len(s) else np.nan)
            x = np.arange(len(xs)) + (i - 2) * 0.07
            ys, lo, hi = map(np.array, (ys, lo, hi))
            a.errorbar(x, ys, yerr=[ys - lo, hi - ys], fmt="-o", color=c, label=lab, lw=1.3, ms=4, capsize=2)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(range(len(xs)))
        a.set_xticklabels(["natural", "0.5", "0.7", "0.9"])
        a.set_title(ttl)
        a.set_xlabel("target uncovered fraction")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_ylabel("paired difference in R2 (across-target 95% t-CI)")
    fig.tight_layout()
    fig.savefig(OUT / "paired_vs_controls.png", dpi=150)
    fig.savefig(OUT / "paired_vs_controls.pdf")
    plt.close(fig)

    if len(curves):
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for v, c in (("h2p", "#e7298a"), ("h2pf", "#7570b3")):
            s = curves[(curves.cond == "cut") & (curves.variant == v)]
            for seed, g in s.groupby("seed"):
                if "beta_entropy" in g:
                    ax[0].plot(g.step, g.beta_entropy, color=c, alpha=0.7, label=LABEL[v] if seed == 0 else None)
                    ax[1].plot(g.step, g.pool_maxN, color=c, alpha=0.7)
                    ax[2].plot(g.step, g.ctx_std, color=c, alpha=0.7)
        s = curves[(curves.cond == "cut") & (curves.variant == "hx")]
        for seed, g in s.groupby("seed"):
            if "attn_entropy" in g:
                ax[0].plot(g.step, g.attn_entropy, color="#d95f02", alpha=0.7, ls="--",
                           label="XATT attention entropy" if seed == 0 else None)
        ax[0].set_title("normalised entropy (1 = uniform / collapsed)")
        ax[1].set_title("max pool weight x N (1 = uniform)")
        ax[2].set_title("context std across units")
        for a in ax:
            a.set_xlabel("training step")
        ax[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "collapse_monitors.png", dpi=150)
        fig.savefig(OUT / "collapse_monitors.pdf")
        plt.close(fig)
    if hr is not None and len(hr):
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
        for a, f in zip(ax, xs):
            s = hr[(hr.frac == f) & (hr.B == 8)]
            s = s[s.key.isin(MAIN_KEYS)]
            a.bar(range(len(s)), s.captured_of_value, color=[pal.get(k, "#999") for k in s.key])
            a.set_xticks(range(len(s)))
            a.set_xticklabels([k.split("/")[-1] for k in s.key], rotation=60, fontsize=7)
            a.set_title(f"uncovered {f}, B=8")
        ax[0].set_ylabel("fraction of oracle value captured")
        fig.tight_layout()
        fig.savefig(OUT / "headroom_captured.png", dpi=150)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    m, p, base = wide(df)
    cells = cells_table(df)
    cells.to_csv(OUT / "cells_r2.csv", index=False)
    paired = paired_table(m, p)
    paired.to_csv(OUT / "paired_across_targets.csv", index=False)
    per_t = m.groupby(["key", "frac", "target", "B"]).agg(total=("total", "mean"), unc=("unc", "mean"),
                                                         cov=("cov", "mean"), attr=("attr", "mean"),
                                                         value=("value", "mean"), r2=("r2", "mean"),
                                                         unc_frac=("unc_frac", "first")).reset_index()
    per_t.to_csv(OUT / "per_target.csv", index=False)
    hr, orc = headroom_table(m, p, base)
    if hr is not None:
        hr.to_csv(OUT / "headroom.csv", index=False)
        orc.to_csv(OUT / "oracle_rows.csv", index=False)
    hr2, orc2 = headroom_table(m, p, base, "oraclecal_*.jsonl")
    if hr2 is not None:
        hr2.to_csv(OUT / "headroom_calibfit.csv", index=False)
        orc2.to_csv(OUT / "oracle_calibfit_rows.csv", index=False)
    if orc is not None:
        keep = ["r2_p0", "r2_zero_unc", "r2_oracle", "n_uncovered", "unc_frac"]
        a = orc.groupby(["frac", "B"])[keep + ["xfit_p0", "xfit_oracle_insample", "xfit_oracle_heldout"]].mean()
        if orc2 is not None:
            a = a.join(orc2.groupby(["frac", "B"]).r2_oracle.mean().rename("r2_fit_on_B_trials"))
        a.reset_index().to_csv(OUT / "oracle_summary.csv", index=False)
    td, curves = training_diag()
    td.to_csv(OUT / "train_summary.csv", index=False)
    curves.to_csv(OUT / "train_curves.csv", index=False)
    ed = eval_diag(df)
    if ed is not None:
        ed.to_csv(OUT / "eval_diagnostics.csv", index=False)
    vs = RAW / "valsrc.json"
    if vs.exists():
        pd.DataFrame(json.loads(vs.read_text())).to_csv(OUT / "source_val.csv", index=False)
    # structural counts for the README
    n = df.groupby(["key", "frac"]).size().unstack()
    n.to_csv(OUT / "row_counts.csv")
    info = {"autograd_calls_max": int(df.autograd_calls.max()), "h4_query_disjoint_all": bool(df.h4_query_disjoint.all()),
            "rows": int(len(df))}
    (OUT / "audit.json").write_text(json.dumps(info, indent=1))
    make_figures(cells, paired, hr, curves)
    print(info)


if __name__ == "__main__":
    main()
