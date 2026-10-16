#!/usr/bin/env python
"""Aggregate the drift-curve rows (run_drift_curve.py) into csv tables and the figure.

Per-target cell = mean over draws x seeds (8 x 2); 95% CI = cluster bootstrap over draws (both seeds of a
draw stay together; P-prior is seed-free and broadcast to both seeds).  Paired differences are formed per
(draw, seed) and averaged per target; across-target CI = t interval (df = n_targets - 1) on target means.
All 12 targets share ONE trunk / calibrator / rule set, so the across-target CI is not an independent-sample
CI of the method; seed spread is reported next to it.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/drift_curve_20261016"
OUT = ROOT / "results/analysis/drift_curve_20261016"
P3 = ROOT / "results/population_calib_p3_20261016/fix"
B_LIST = (1, 2, 4, 8, 16, 33)
METHODS = ("p_prior", "p0", "r1s", "r2", "r3", "p0_ui_reg")
LABEL = {"p_prior": "P-prior", "p0": "P0", "r1s": "P0 + R1-stream", "r2": "R2 (F7-J)",
         "r3": "R3 (F7-J + readout ridge)", "p0_ui_reg": "P0 + ui_reg (non-deployable)"}
DIFFS = (("p0", "p_prior"), ("r1s", "p0"), ("r3", "p0"), ("p0_ui_reg", "p0"), ("r2", "p0"),
         ("p0_ui_reg", "r1s"), ("p0_ui_reg", "r3"))
RNG = np.random.default_rng(0)


def load():
    rows = []
    for f in sorted(glob.glob(str(RAW / "rows_*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["cfg"] = df.cfg.map(json.dumps)
    return df


def pivot(df):
    """[target, B, draw, seed] x method; P-prior broadcast to both seeds (seed-free)."""
    prior = df[df.method == "p_prior"].drop(columns="seed")
    other = df[df.method != "p_prior"]
    idx = ["target", "date", "days", "n_units", "B", "draw", "seed"]
    pv = other.pivot_table(index=idx, columns="method", values="r2").reset_index()
    pr = prior[["target", "B", "draw", "r2"]].rename(columns={"r2": "p_prior"})
    pv = pv.merge(pr, on=["target", "B", "draw"], how="left")
    return pv


def check_complete(pv, targets):
    exp = len(targets) * len(B_LIST) * 8 * 2
    assert len(pv) == exp, f"expected {exp} (target,B,draw,seed) cells, got {len(pv)}"
    for m in METHODS:
        assert pv[m].notna().all(), f"missing values for {m}"
        assert np.isfinite(pv[m]).all(), f"non-finite {m}"


def boot_ci(mat, n=4000):
    """mat [draws, seeds]; cluster bootstrap over draws -> (lo, hi)."""
    d = mat.shape[0]
    ix = RNG.integers(0, d, size=(n, d))
    means = mat[ix].mean(axis=(1, 2))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grid(sub, col):
    return sub.pivot(index="draw", columns="seed", values=col).values


def cells(pv):
    out = []
    for (t, B), sub in pv.groupby(["target", "B"]):
        for m in METHODS:
            mat = grid(sub, m)
            lo, hi = boot_ci(mat)
            out.append({"target": t[-8:], "days": int(sub.days.iloc[0]), "n_units": int(sub.n_units.iloc[0]),
                        "B": B, "method": m, "r2_mean": mat.mean(), "ci_lo": lo, "ci_hi": hi,
                        "r2_seed0": mat[:, 0].mean(), "r2_seed1": mat[:, 1].mean(), "r2_draw_sd": mat.mean(1).std(ddof=1)})
    return pd.DataFrame(out)


def paired_per_target(pv):
    out = []
    for (t, B), sub in pv.groupby(["target", "B"]):
        for a, b in DIFFS:
            mat = grid(sub.assign(d=sub[a] - sub[b]), "d")
            lo, hi = boot_ci(mat)
            out.append({"target": t[-8:], "days": int(sub.days.iloc[0]), "B": B, "diff": f"{a}-{b}",
                        "mean": mat.mean(), "ci_lo": lo, "ci_hi": hi, "seed0": mat[:, 0].mean(), "seed1": mat[:, 1].mean()})
    return pd.DataFrame(out)


def across_targets(pt):
    """t interval over target means; pooled B>=4 = mean over B in {4,8,16,33}."""
    out = []
    pt = pt.copy()
    pooled = pt[pt.B >= 4].groupby(["target", "days", "diff"], as_index=False)[["mean", "seed0", "seed1"]].mean()
    pooled["B"] = "4-33"
    pt["B"] = pt.B.astype(str)
    allp = pd.concat([pt[["target", "days", "B", "diff", "mean", "seed0", "seed1"]], pooled], ignore_index=True)
    for (B, diff), g in allp.groupby(["B", "diff"]):
        v = g["mean"].values
        n = len(v)
        half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        rho, p = stats.spearmanr(g.days, v)
        sl = stats.linregress(g.days, v)
        out.append({"B": B, "diff": diff, "n_targets": n, "mean": v.mean(), "ci_lo": v.mean() - half,
                    "ci_hi": v.mean() + half, "n_pos": int((v > 0).sum()),
                    "mean_seed0": g.seed0.mean(), "mean_seed1": g.seed1.mean(),
                    "sd_across_targets": v.std(ddof=1), "spearman_days": rho, "spearman_p": p,
                    "slope_per_100d": sl.slope * 100, "slope_p": sl.pvalue})
    return pd.DataFrame(out)


def gap_groups(pt):
    g = pt.copy()
    g["group"] = pd.cut(g.days, [0, 150, 250, 300, 440, 500],
                        labels=["~112 d (0626,0627)", "207 d (0929)", "272 d (1203)", "431-432 d (0511,0512)",
                                "461-468 d (0610-0617)"])
    pooled = g[g.B >= 4].groupby(["target", "group", "diff"], observed=True, as_index=False)["mean"].mean()
    pooled["B"] = "4-33"
    b8 = g[g.B.isin([8, 33])].assign(B=lambda d: d.B.astype(str))[["target", "group", "diff", "mean", "B"]]
    allg = pd.concat([pooled, b8])
    return allg.groupby(["group", "B", "diff"], observed=True)["mean"].agg(["mean", "std", "count"]).reset_index()


def consistency(pv):
    """P0 / R3 / ui_reg for the two existing targets against the earlier P3 run (same code path)."""
    rows = []
    ev = pd.DataFrame([json.loads(l) for l in (P3 / "eval/eval_month_oosc.jsonl").read_text().splitlines()])
    ev_y = pd.DataFrame([json.loads(l) for l in (P3.parent / "eval/eval_year_oos.jsonl").read_text().splitlines()])
    uim = pd.DataFrame([json.loads(l) for l in (P3 / "uiref/eval_month_oosc.jsonl").read_text().splitlines()])
    sel = json.loads((P3 / "select/selection_month_oosc.json").read_text())
    kinds = ("full", "diag", "lr8", "lr16", "lr32")
    for B in B_LIST:
        k3 = max(kinds, key=lambda k: sel[f"r3_{k}"][str(B)]["source_r2"])
        k1 = max(kinds, key=lambda k: sel[f"r1s_{k}"][str(B)]["source_r2"])
        for tag, evx, tdate in (("month", ev, "20140626"), ("year", ev_y, "20150511")):
            new = pv[(pv.date == tdate) & (pv.B == B)].set_index(["draw", "seed"])
            old = evx[(evx.B == B)].copy()
            # year used its own (OOS) rules, so only P0 (rule-free) is comparable for 20150511
            for meth, col, old_m in (("p0", "p0", "p0"),) + \
                    ((("r2", "r2", "r2"), ("r3", "r3", f"r3_{k3}"), ("r1s", "r1s", f"r1s_{k1}")) if tag == "month" else ()):
                o = old[old.method == old_m].set_index(["draw", "seed"]).r2
                j = new[col].to_frame().join(o.rename("old"), how="inner")
                rows.append({"stratum_of_old_run": tag, "date": tdate, "B": B, "method": meth,
                             "max_abs_diff": float((j[col] - j.old).abs().max()), "n": len(j)})
        n = pv[(pv.date == "20140626") & (pv.B == B)].set_index(["draw", "seed"])
        o = uim[uim.B == B].set_index(["draw", "seed"]).r2
        j = n["p0_ui_reg"].to_frame().join(o.rename("old"), how="inner")
        rows.append({"stratum_of_old_run": "month", "date": "20140626", "B": B, "method": "p0_ui_reg",
                     "max_abs_diff": float((j.p0_ui_reg - j.old).abs().max()), "n": len(j)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure ----

STYLE = {  # Okabe-Ito (colorblind safe) + distinct marker / line style as secondary encoding
    "p_prior": dict(color="#999999", marker="s", ls=":", filled=True),
    "p0": dict(color="#0072B2", marker="o", ls="-", filled=True),
    "r1s": dict(color="#009E73", marker="^", ls="-", filled=True),
    "r3": dict(color="#D55E00", marker="D", ls="--", filled=True),
    "p0_ui_reg": dict(color="#CC79A7", marker="v", ls="-.", filled=False),
}
OFFS = {"p_prior": -4.5, "p0": -2.25, "r1s": 0.0, "r3": 2.25, "p0_ui_reg": 4.5}


def figure(ce, path_stub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, B in zip(axes.ravel(), B_LIST):
        for m, st in STYLE.items():
            s = ce[(ce.B == B) & (ce.method == m)].sort_values("days")
            x = s.days.values + OFFS[m]
            ax.errorbar(x, s.r2_mean, yerr=[s.r2_mean - s.ci_lo, s.ci_hi - s.r2_mean], color=st["color"],
                        marker=st["marker"], ls="none", ms=4.5, capsize=1.5, elinewidth=0.8,
                        mfc=st["color"] if st["filled"] else "white", label=LABEL[m] if m != "r3" else LABEL[m] + " [optional tier]")
        ax.set_title(f"B = {B} labeled trials", fontsize=10)
        ax.set_ylim(-0.02, 0.95)
        ax.set_xlim(90, 490)
    fig.supxlabel("Days between latest source session (2014-03-06) and target session", fontsize=9.5, y=0.075)
    for ax in axes[:, 0]:
        ax.set_ylabel("Query $R^2$ (physical units)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("DANDI 000688 sub-M: query $R^2$ vs. drift (12 targets, one frozen month trunk; "
                 "mean and 95% CI over 8 draws x 2 seeds per target; one marker = one target session)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stub}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_diff(pt, path_stub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    specs = [("p0-p_prior", "P0 $-$ P-prior", "#0072B2", "o", "-"),
             ("r1s-p0", "(P0+R1-stream) $-$ P0", "#009E73", "^", "-"),
             ("r3-p0", "R3 $-$ P0 [optional tier]", "#D55E00", "D", "--"),
             ("p0_ui_reg-p0", "ui_reg $-$ P0 (non-deployable)", "#CC79A7", "v", "-.")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, B in zip(axes, (8, 33)):
        for k, (d, lab, c, mk, ls) in enumerate(specs):
            s = pt[(pt.B == B) & (pt["diff"] == d)].sort_values("days")
            x = s.days.values + (k - 1.5) * 2.0
            ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]], color=c, marker=mk, ls="none",
                        ms=4.5, capsize=1.5, elinewidth=0.8, label=lab, mfc=c if d != "p0_ui_reg-p0" else "white")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"B = {B} labeled trials", fontsize=10)
        ax.set_xlabel("Days between latest source session (2014-03-06) and target")
        ax.set_xlim(90, 490)
    axes[0].set_ylabel("Paired difference in query $R^2$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stub}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    pv = pivot(df)
    targets = sorted(pv.target.unique())
    check_complete(pv, targets)
    chk = df[df.method != "p0_ui_reg"]
    flags = {"n_rows": int(len(df)), "n_targets": len(targets),
             "all_candidate_rows_zero_backprop": bool((chk.h2_backward_calls == 0).all()),
             "all_query_disjoint": bool(df.h4_query_disjoint.all()),
             "B_equals_budget": bool((df.h4_budget == df.B).all())}
    (OUT / "flags.json").write_text(json.dumps(flags, indent=1))
    print(flags)
    ce = cells(pv)
    ce.to_csv(OUT / "cells_r2.csv", index=False, float_format="%.5f")
    pt = paired_per_target(pv)
    pt.to_csv(OUT / "paired_per_target.csv", index=False, float_format="%.5f")
    at = across_targets(pt)
    at.to_csv(OUT / "paired_across_targets.csv", index=False, float_format="%.5f")
    gg = gap_groups(pt)
    gg.to_csv(OUT / "paired_by_gap_group.csv", index=False, float_format="%.5f")
    cons = consistency(pv)
    cons.to_csv(OUT / "consistency_vs_p3.csv", index=False, float_format="%.6g")
    pv.drop(columns=["n_units"]).to_csv(OUT / "rows_compact.csv.gz", index=False, float_format="%.5f")
    figure(ce, str(OUT / "drift_curve_r2_vs_gap"))
    figure_diff(pt, str(OUT / "drift_curve_paired_diffs"))
    print(cons.groupby(["stratum_of_old_run", "method"]).max_abs_diff.max())


if __name__ == "__main__":
    main()
