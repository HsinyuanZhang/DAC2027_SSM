#!/usr/bin/env python
"""Aggregate hybrid-identity rows (run_hybrid_identity.py eval) into csv tables and the figure.

Cell = mean over draws x seeds per (variant, target, B).  Per-target CI = cluster bootstrap over draws
(both seeds of a draw stay together).  Across-target CI = t interval (df = n_targets - 1) on target
means (clustered by target session; all targets share one trunk / one calibrator set, so this does not
cover trunk / calibrator-training variance; seed spread is reported next to it).

Metrics of a variant v (all paired per (target, B, draw, seed) against the reused P0 = "h0"):
  total      R2(v) - R2(h0)
  unc        R2(covered cols from h0, uncovered cols from v) - R2(h0)     (effect on uncovered units)
  cov        R2(covered cols from v, uncovered cols from h0) - R2(h0)     (effect on covered units)
For a pair (a, b) the metric is metric(a) - metric(b); for unc this is R2(cov=h0, unc=a) - R2(cov=h0, unc=b).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/hybrid_identity_20261016"
OUT = ROOT / "results/analysis/hybrid_identity_20261016"
B_LIST = (1, 2, 4, 8, 16, 33)
RNG = np.random.default_rng(0)
VARS = ["h0", "h0m", "h1", "h1w113", "h1w130", "h1w138", "h1w153", "h2d", "h2d_loc", "h2p", "h2p_loc",
        "h3d", "h3d_loc", "h3p", "h3p_loc"]
LABEL = {"h0": "H0 = P0 (reused)", "h0m": "P0 + masked training (no flag)", "h1": "H1 = P0 + covered flag",
         "h1w113": "P0-wide+flag (113, ~H3d)", "h1w130": "P0-wide+flag (130, ~H3p)",
         "h1w138": "P0-wide+flag (138, ~H2d)", "h1w153": "P0-wide+flag (153, ~H2p)",
         "h2d": "H2-DeepSets (hybrid)", "h2d_loc": "H2-DeepSets-local", "h2p": "H2-PMA m=8 (hybrid)",
         "h2p_loc": "H2-PMA-local", "h3d": "H3-DeepSets (all units)", "h3d_loc": "H3-DeepSets-local",
         "h3p": "H3-PMA (all units)", "h3p_loc": "H3-PMA-local"}
COMPARE = [   # (a, b, role)
    ("h0m", "h0", "masked-training aug only"), ("h1", "h0", "flag + aug"), ("h1", "h0m", "flag alone"),
    ("h2d", "h0", "H2 DeepSets"), ("h2p", "h0", "H2 PMA"), ("h3d", "h0", "H3 DeepSets"),
    ("h3p", "h0", "H3 PMA"),
    ("h1w113", "h0", "wide control"), ("h1w130", "h0", "wide control"), ("h1w138", "h0", "wide control"),
    ("h1w153", "h0", "wide control"),
    ("h2d_loc", "h0", "local control"), ("h2p_loc", "h0", "local control"),
    ("h3d_loc", "h0", "local control"), ("h3p_loc", "h0", "local control"),
    ("h2d", "h2d_loc", "pooling vs local"), ("h2p", "h2p_loc", "pooling vs local"),
    ("h3d", "h3d_loc", "pooling vs local"), ("h3p", "h3p_loc", "pooling vs local"),
    ("h2d", "h1w138", "pooling vs wide"), ("h2p", "h1w153", "pooling vs wide"),
    ("h3d", "h1w113", "pooling vs wide"), ("h3p", "h1w130", "pooling vs wide"),
    ("h2d", "h1", "hybrid vs flag"), ("h2p", "h1", "hybrid vs flag"),
    ("h3d", "h2d", "condition covered too"), ("h3p", "h2p", "condition covered too"),
    ("h2p", "h2d", "PMA vs DeepSets"),
]
METRICS = ("total", "unc", "cov")


def load():
    rows = []
    for f in sorted(glob.glob(str(RAW / "eval/rows_*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def wide(df):
    """index (target,B,draw,seed) -> columns r2__v, unc__v, cov__v."""
    idx = ["target", "date", "days", "n_units", "unc_frac", "n_uncovered", "B", "draw", "seed"]
    base = df[df.variant == "h0"].set_index(idx)["r2"].rename("r2__h0").to_frame()
    out = base.copy()
    out["zero__h0"] = df[df.variant == "h0"].set_index(idx)["r2_zero_unc"]
    for v in sorted(set(df.variant) - {"h0"}):
        d = df[df.variant == v].set_index(idx)
        out[f"r2__{v}"] = d["r2"]
        out[f"zero__{v}"] = d["r2_zero_unc"]
        out[f"unc__{v}"] = d["r2_swap_unc"] - out["r2__h0"]
        out[f"cov__{v}"] = d["r2_swap_cov"] - out["r2__h0"]
        out[f"swapunc__{v}"] = d["r2_swap_unc"]
    out["unc__h0"] = 0.0
    out["cov__h0"] = 0.0
    out["total__h0"] = 0.0
    for v in sorted(set(df.variant) - {"h0"}):
        out[f"total__{v}"] = out[f"r2__{v}"] - out["r2__h0"]
    return out.reset_index()


def metric(w, v, m):
    return w[f"{m}__{v}"]


def diff_series(w, a, b, m):
    if m == "unc":     # R2(cov=h0, unc=a) - R2(cov=h0, unc=b)
        return w[f"unc__{a}"] - w[f"unc__{b}"]
    return w[f"{m}__{a}"] - w[f"{m}__{b}"]


def boot_ci(x_draw):
    """cluster bootstrap over draws; x_draw: [n_draw] (seed-averaged)."""
    n = len(x_draw)
    bs = x_draw[RNG.integers(0, n, size=(2000, n))].mean(1)
    return np.percentile(bs, [2.5, 97.5])


def build_paired(w):
    per_t, across = [], []
    targets = sorted(w.target.unique())
    unc_frac = w.groupby("target").unc_frac.first()
    days = w.groupby("target").days.first()
    for a, b, role in COMPARE:
        if f"r2__{a}" not in w or f"r2__{b}" not in w:
            continue
        for m in METRICS:
            if m == "unc" and (a == "h0" or b == "h0") and False:
                continue
            w2 = w.assign(_d=diff_series(w, a, b, m))
            for Bsel in list(B_LIST) + ["B>=2", "all"]:
                if Bsel == "B>=2":
                    sub = w2[w2.B >= 2]
                elif Bsel == "all":
                    sub = w2
                else:
                    sub = w2[w2.B == Bsel]
                # per (target, seed): mean over draws (and B for pooled selections)
                tm = sub.groupby("target")._d.mean().reindex(targets)
                for s in (0, 1):
                    pass
                seed_means = [sub[sub.seed == s].groupby("target")._d.mean().reindex(targets) for s in (0, 1)]
                for t in targets:
                    dd = sub[sub.target == t].groupby(["B", "draw"])._d.mean().values
                    lo, hi = boot_ci(dd)
                    per_t.append(dict(a=a, b=b, role=role, metric=m, B=Bsel, target=t,
                                      unc_frac=unc_frac[t], days=days[t], diff=tm[t], ci_lo=lo, ci_hi=hi))
                x = tm.values
                n = len(x)
                se = x.std(ddof=1) / np.sqrt(n)
                tcrit = stats.t.ppf(0.975, n - 1)
                uf = unc_frac.reindex(targets).values
                rho, rp = stats.spearmanr(uf, x) if np.std(x) > 0 else (np.nan, np.nan)
                lr = stats.linregress(uf, x) if np.std(x) > 0 else None
                across.append(dict(a=a, b=b, role=role, metric=m, B=Bsel, n_targets=n, mean=x.mean(),
                                   ci_lo=x.mean() - tcrit * se, ci_hi=x.mean() + tcrit * se,
                                   n_pos=int((x > 0).sum()), seed0=seed_means[0].mean(), seed1=seed_means[1].mean(),
                                   spearman_vs_unc_frac=rho, spearman_p=rp,
                                   slope_per_unit_unc_frac=(lr.slope if lr else np.nan),
                                   slope_p=(lr.pvalue if lr else np.nan)))
    return pd.DataFrame(per_t), pd.DataFrame(across)


def build_cells(df):
    rows = []
    for (v, t, B), g in df.groupby(["variant", "target", "B"]):
        pd_ = g.groupby("draw").r2.mean().values
        lo, hi = boot_ci(pd_)
        rows.append(dict(variant=v, target=t, B=B, unc_frac=g.unc_frac.iloc[0], n_units=g.n_units.iloc[0],
                         r2=g.r2.mean(), ci_lo=lo, ci_hi=hi, r2_seed0=g[g.seed == 0].r2.mean(),
                         r2_seed1=g[g.seed == 1].r2.mean(), r2_zero_unc=g.r2_zero_unc.mean(),
                         r2_swap_unc=g.r2_swap_unc.mean() if "r2_swap_unc" in g else np.nan,
                         r2_swap_cov=g.r2_swap_cov.mean() if "r2_swap_cov" in g else np.nan))
    return pd.DataFrame(rows)


def contrib_table(w, variants):
    """Value of the uncovered units' columns: R2(full) - R2(uncovered columns zeroed), per variant."""
    rows = []
    for v in variants:
        if f"r2__{v}" not in w:
            continue
        c = (w[f"r2__{v}"] - w[f"zero__{v}"])
        for Bsel in list(B_LIST) + ["B>=2"]:
            sub = w[(w.B >= 2) if Bsel == "B>=2" else (w.B == Bsel)].assign(_c=c)
            tm = sub.groupby("target")._c.mean()
            n = len(tm)
            se = tm.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            rows.append(dict(variant=v, B=Bsel, mean_contrib_uncovered_cols=tm.mean(),
                             ci_lo=tm.mean() - tcrit * se, ci_hi=tm.mean() + tcrit * se))
    return pd.DataFrame(rows)


def train_summary():
    rows = []
    for f in sorted(glob.glob(str(RAW / "runs/*/seed*/done.json"))):
        d = json.loads(Path(f).read_text())
        rows.append({k: d[k] for k in ("name", "seed", "n_params", "best_step", "best_val_r2", "converged",
                                       "final_budget", "stopped_early", "seconds")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure ----
OI = {"h1": "#E69F00", "h2d": "#56B4E9", "h2p": "#009E73", "h3d": "#0072B2", "h3p": "#D55E00",
      "ctrl": "#7F7F7F"}
MK = {"h1": "s", "h2d": "o", "h2p": "^", "h3d": "D", "h3p": "v"}


def figure(per_t, across, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))

    def scat(a, metr, a_, b_, color, marker, label, hollow=False, dx=0.0):
        s = per_t[(per_t.a == a_) & (per_t.b == b_) & (per_t.metric == metr) & (per_t.B == "B>=2")]
        if s.empty:
            return
        a.errorbar(s.unc_frac + dx, s["diff"], yerr=[s["diff"] - s.ci_lo, s.ci_hi - s["diff"]], fmt="none",
                   ecolor=color, elinewidth=0.6, alpha=0.5)
        a.scatter(s.unc_frac + dx, s["diff"], s=26, color=("white" if hollow else color), edgecolor=color,
                  marker=marker, linewidth=1.0, label=label, zorder=3)

    for k, (a_, lab) in enumerate([("h1", "H1 flag"), ("h2d", "H2 DeepSets"), ("h2p", "H2 PMA"),
                                   ("h3d", "H3 DeepSets"), ("h3p", "H3 PMA")]):
        scat(ax[0, 0], "total", a_, "h0", OI[a_], MK[a_], lab, dx=(k - 2) * 0.004)
        scat(ax[0, 1], "unc", a_, "h0", OI[a_], MK[a_], lab, dx=(k - 2) * 0.004)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axhline(0, color="black", lw=0.8)
    ax[0, 0].set_title("(a) total R2 gain over P0 (B>=2 mean)")
    ax[0, 1].set_title("(b) effect of uncovered columns (covered = P0)")
    ax[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    for a_, b_, c, mk, hol, lab in [("h2d", "h2d_loc", OI["h2d"], "o", False, "H2-DS: pooled - local"),
                                    ("h2d", "h1w138", OI["h2d"], "o", True, "H2-DS: pooled - wide"),
                                    ("h2p", "h2p_loc", OI["h2p"], "^", False, "H2-PMA: pooled - local"),
                                    ("h2p", "h1w153", OI["h2p"], "^", True, "H2-PMA: pooled - wide")]:
        scat(ax[1, 0], "unc", a_, b_, c, mk, lab, hollow=hol)
    ax[1, 0].set_title("(c) pooling attribution on uncovered columns (B>=2 mean)")
    ax[1, 0].legend(fontsize=7, frameon=False)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("uncovered unit fraction of the target")
    ax[0, 0].set_ylabel("paired diff in query R2")
    # (d) across-target mean +- t CI vs B for total gain
    for k_, (a_, b_, c, mk, ls, lab) in enumerate([("h2d", "h0", OI["h2d"], "o", "-", "H2-DS - P0"),
                                   ("h2p", "h0", OI["h2p"], "^", "-", "H2-PMA - P0"),
                                   ("h1", "h0", OI["h1"], "s", "-", "H1 - P0"),
                                   ("h2d_loc", "h0", OI["h2d"], "o", ":", "H2-DS-local - P0"),
                                   ("h1w138", "h0", OI["ctrl"], "x", ":", "wide(138) - P0")]):
        s = across[(across.a == a_) & (across.b == b_) & (across.metric == "total") & across.B.isin(B_LIST)]
        if s.empty:
            continue
        x = np.arange(len(B_LIST))
        s = s.set_index("B").reindex(B_LIST)
        ax[1, 1].errorbar(x + (k_ - 2) * 0.05, s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]],
                          color=c, marker=mk, ls=ls, lw=1.2, ms=4, capsize=2, label=lab)
    ax[1, 1].axhline(0, color="black", lw=0.8)
    ax[1, 1].set_xticks(range(len(B_LIST)))
    ax[1, 1].set_xticklabels(B_LIST)
    ax[1, 1].set_xlabel("B (labeled trials)")
    ax[1, 1].set_title("(d) mean total gain, 95% t-CI over 12 targets")
    ax[1, 1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    targets = sorted(df.target.unique())
    exp = {"h0": len(targets) * len(B_LIST) * 8 * 2}
    print(df.groupby("variant").size().to_dict(), "targets:", len(targets))
    for v in df.variant.unique():
        assert (df.groupby("variant").size()[v] == exp["h0"]), f"incomplete variant {v}"
    assert df.autograd_calls.max() == 0, "autograd called on target side"
    assert df.h4_query_disjoint.all()
    w = wide(df)
    cells = build_cells(df)
    cells.to_csv(OUT / "cells_r2.csv", index=False, float_format="%.5f")
    per_t, across = build_paired(w)
    per_t.to_csv(OUT / "paired_per_target.csv", index=False, float_format="%.5f")
    across.to_csv(OUT / "paired_across_targets.csv", index=False, float_format="%.5f")
    contrib_table(w, [v for v in VARS if f"r2__{v}" in w]).to_csv(OUT / "uncovered_columns_contribution.csv",
                                                                   index=False, float_format="%.5f")
    vs = RAW / "valsrc.json"
    if vs.exists():
        pd.DataFrame(json.loads(vs.read_text())).to_csv(OUT / "source_val.csv", index=False, float_format="%.5f")
    train_summary().to_csv(OUT / "train_summary.csv", index=False)
    inv = df.groupby("target")[["days", "n_units", "n_uncovered", "unc_frac"]].first().reset_index()
    inv.to_csv(OUT / "target_coverage.csv", index=False)
    flags = {"rows": int(len(df)), "max_autograd_calls_on_target_side": int(df.autograd_calls.max()),
             "all_query_disjoint": bool(df.h4_query_disjoint.all()), "variants": sorted(df.variant.unique()),
             "targets": targets}
    (OUT / "flags.json").write_text(json.dumps(flags, indent=1))
    figure(per_t, across, OUT / "hybrid_gain_vs_uncovered_fraction")
    print("done")


if __name__ == "__main__":
    main()
