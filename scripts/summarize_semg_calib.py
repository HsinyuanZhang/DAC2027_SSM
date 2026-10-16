#!/usr/bin/env python
"""Aggregate Q-D rows (sEMG calibrators + reused reference rows) into compact csv/json.

Hyser: R2 per (target, backbone seed) (mean over draws).  PRIMARY CI = seed x fold cluster-robust (the 4 targets of a
       subject fold share one calibrator per backbone seed; 10 clusters, t with 9 df).  Also reported: fold-only clusters
       (5 clusters, t with 4 df; the two backbone seeds of a target share the target), two-way seed x fold + target
       (Cameron-Gelbach-Miller, 9 df; ``*_ci95_2way``) and the legacy target-level t-CI
       over 20 targets (anticonservative: ignores the shared calibrator), columns ``*_ci95_target``.
DB6 : balanced accuracy / macro-F1 per subject (draws -> sessions of a day -> seeds), t-CI over 10 subjects
       (one calibrator per backbone seed is shared by all subjects, see the per-seed tables for seed dependence).
"""
import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "semg_calib_20261016"
ANA = ROOT / "results" / "analysis"

HYSER_REF = ["reuse_s1_srcnorm", "reuse_s1_calnorm", "mean_table_calnorm", "mean_loso_calnorm",
             "rownorm_ridge_perB", "ui_prior_s1_calnorm", "ui_prior_mean_calnorm"]
HYSER_DEPLOYABLE = ["reuse_s1_calnorm", "mean_loso_calnorm", "mean_table_calnorm", "rownorm_ridge_perB"]


def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def tci(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(sps.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))


def cluster_ci(vals, clusters):
    """Mean and 95% half-width with cluster-robust (CR1) variance and t(G-1) quantile."""
    v = np.asarray(vals, float)
    c = np.asarray(clusters)
    ids = np.unique(c)
    g = len(ids)
    mu = v.mean()
    if g < 2:
        return float(mu), float("nan")
    score = np.array([(v[c == i] - mu).sum() for i in ids])
    var = g / (g - 1) * (score ** 2).sum() / len(v) ** 2
    return float(mu), float(sps.t.ppf(0.975, g - 1) * np.sqrt(var))


def _cr_var(v, c):
    ids = np.unique(c)
    g = len(ids)
    if g < 2:
        return float("nan"), g
    mu = v.mean()
    score = np.array([(v[c == i] - mu).sum() for i in ids])
    return g / (g - 1) * (score ** 2).sum() / len(v) ** 2, g


def cluster_ci2(vals, c1, c2, c12):
    """Two-way cluster-robust CI (Cameron-Gelbach-Miller): V = V(c1) + V(c2) - V(c1 x c2); df = min(G1, G2) - 1.

    c1 = seed x fold (shared calibrator), c2 = target (the two backbone seeds see the same target),
    c12 = the (seed, target) unit itself.  Falls back to V(c1) if the combination is not positive."""
    v = np.asarray(vals, float)
    v1, g1 = _cr_var(v, np.asarray(c1))
    v2, g2 = _cr_var(v, np.asarray(c2))
    v12, _ = _cr_var(v, np.asarray(c12))
    var = v1 + v2 - v12
    if not np.isfinite(var) or var <= 0:
        var = v1
    return float(v.mean()), float(sps.t.ppf(0.975, min(g1, g2) - 1) * np.sqrt(var))


N_FOLDS = 5
WEAK_VAL = 0.05          # descriptive flag only: source-val best below this (prior-only is about -0.04)


def fold_of(t):
    return (np.asarray(t) - 1) % N_FOLDS


def load_hyser(seeds=(0, 1)):
    ref = []
    for s in seeds:
        ref += [r for r in read_jsonl(ANA / f"hyser_20261015/raw/rows_seed{s}.jsonl.gz") if r["method"] in HYSER_REF]
    # post-review rerun (the committed raw gz holds the pre-review lambda grid)
    for r in read_jsonl(ROOT / "results/hyser_20261015/eval_closed_form/rows.jsonl"):
        if r["method"] == "rownorm_ridge_perB":
            for s in seeds:                       # backbone-free row: replicated per seed for pairing
                ref.append({**r, "seed": s})
    cal = []
    for s in seeds:
        for sub in ("", "fix"):                   # "fix" = post-review capacity controls (same protocol, same draws)
            p = RES / sub / "hyser" / f"seed{s}" / "rows.jsonl"
            if p.exists():
                cal += [r for r in read_jsonl(p) if r["method"].startswith("cal_")]
    keep = ["target", "B", "draw", "seed", "method", "r2", "r2_pooled", "unit_ratio"]
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in ref + cal])
    df["fold"] = fold_of(df["target"])
    df["cluster"] = df["seed"] * N_FOLDS + df["fold"]
    return df


def per_unit(df, metric):
    """One value per (method, B, target, backbone seed): mean over the 8 draws."""
    g = df.groupby(["method", "B", "target", "seed"])[metric].mean().reset_index()
    g["fold"] = fold_of(g["target"])
    g["cluster"] = g["seed"] * N_FOLDS + g["fold"]
    return g


def per_target(df, metric):
    """Legacy: mean over draws AND seeds, one value per target."""
    return (df.groupby(["method", "B", "target"])[metric].mean().reset_index())


def _cis(vals, seed_fold_cluster, fold_cluster, target_ids):
    """(mean, seed x fold CI [primary], fold-only CI, legacy target-level CI, two-way seed x fold + target CI)."""
    vals = np.asarray(vals, float)
    tid = np.asarray(target_ids)
    mu, ci = cluster_ci(vals, seed_fold_cluster)
    _, ci_fold = cluster_ci(vals, fold_cluster)
    tl = pd.Series(vals).groupby(tid).mean().values
    _, ci_tgt = tci(tl)
    unit = np.arange(len(vals))
    _, ci_2 = cluster_ci2(vals, seed_fold_cluster, tid, unit)
    return mu, ci, ci_fold, ci_tgt, ci_2


def table(df, metric="r2"):
    pu = per_unit(df, metric)
    rows = []
    for (m, b), g in pu.groupby(["method", "B"]):
        mu, ci, cif, cit, ci2 = _cis(g[metric].values, g["cluster"].values, g["fold"].values, g["target"].values)
        rows.append({"method": m, "B": b, "n_targets": g["target"].nunique(), "n_clusters": g["cluster"].nunique(),
                     f"{metric}_mean": mu, f"{metric}_ci95": ci, f"{metric}_ci95_fold": cif,
                     f"{metric}_ci95_2way": ci2, f"{metric}_ci95_target": cit})
    return pd.DataFrame(rows)


def seed_table(df, metric="r2"):
    g = df.groupby(["method", "B", "seed", "target"])[metric].mean().reset_index()
    g = g.groupby(["method", "B", "seed"])[metric].mean().unstack("seed")
    g.columns = [f"seed{c}" for c in g.columns]
    g["seed_spread"] = g.max(axis=1) - g.min(axis=1)
    return g.reset_index()


def _pivot_units(df, metric):
    pu = per_unit(df, metric)
    return pu.pivot_table(index=["B", "target", "seed", "fold", "cluster"], columns="method", values=metric)


def paired_from_pivot(pt, a, b, drop_clusters=()):
    """Paired differences a-b per (B, target, seed) with clustered CIs; rows of ``drop_clusters`` removed."""
    out = []
    if a not in pt or b not in pt:
        return out
    for B, g in pt.groupby(level="B"):
        gg = g[[a, b]].dropna()
        if len(drop_clusters):
            gg = gg[~gg.index.get_level_values("cluster").isin(list(drop_clusters))]
        if len(gg) == 0:
            continue
        d = (gg[a] - gg[b]).values
        tid = gg.index.get_level_values("target").values
        tgt = pd.Series(d, index=tid).groupby(level=0).mean().values
        mu, ci, cif, cit, ci2 = _cis(d, gg.index.get_level_values("cluster").values,
                                     gg.index.get_level_values("fold").values, tid)
        out.append({"a": a, "b": b, "B": B, "diff_mean": mu, "diff_ci95": ci, "diff_ci95_fold": cif,
                    "diff_ci95_2way": ci2, "diff_ci95_target": cit,
                    "excludes_zero": bool(abs(mu) > ci) if np.isfinite(ci) else False,
                    "excludes_zero_fold": bool(abs(mu) > cif) if np.isfinite(cif) else False,
                    "excludes_zero_2way": bool(abs(mu) > ci2) if np.isfinite(ci2) else False,
                    "n_clusters": int(gg.index.get_level_values("cluster").nunique()), "n": len(tgt),
                    "frac_targets_a_better": float((tgt > 0).mean())})
    return out


def paired(df, a, b, metric="r2"):
    return paired_from_pivot(_pivot_units(df, metric), a, b)


CONTROL_PAIRS = [("p2", "p0w"), ("p4", "p0w"), ("p4", "p0w4"), ("p2", "p2l"), ("p2", "p2c"), ("p4", "p2l"),
                 ("p0w", "p0"), ("p2l", "p0"), ("p2c", "p0"), ("p2l", "p0w"), ("p2c", "p0w"), ("p0w4", "p0w")]
BASE_PAIRS = [("p1", "p0"), ("p2", "p0"), ("p4", "p0"), ("p2", "p1"), ("p4", "p1"), ("p4", "p2")]


def training_info(sources=("free",)):
    """(seed, fold, variant) -> info for every identity-free Hyser calibrator (original + fix runs)."""
    out = {}
    for f in sorted(glob.glob(str(RES / "**" / "hyser" / "seed*" / "free_*_f*" / "info.json"), recursive=True)):
        p = Path(f)
        i = json.load(open(f))
        out[(i["seed"], i["fold"], i["variant"])] = {"best_step": i["best_step"], "val_best": i["best_val"],
                                                     "val0": i["val_step0"], "n_params": i["n_params"],
                                                     "steps_run": i["steps_run"]}
    return out


def collapse_sensitivity(df, info):
    """Free-mode comparisons with and without the cells that kept the step-0 table (ORIGINAL selection rule kept).

    collapsed = best_step == 0 (the table never left its initialisation); weak = source-val best < WEAK_VAL.
    A cluster (seed x fold) is dropped for a pair (a, b) when a or b is collapsed (resp. collapsed/weak) in it.
    Numbers under the original rule are the reported ones; this is a disclosure/sensitivity analysis only.
    """
    pt = _pivot_units(df, "r2")
    rows = []
    cells = []
    for (sd, fo, v), i in sorted(info.items()):
        cells.append({"seed": sd, "fold": fo, "variant": v, **i, "collapsed": i["best_step"] == 0,
                      "weak": (i["best_step"] > 0 and i["val_best"] < WEAK_VAL)})
    cells = pd.DataFrame(cells)
    pairs = BASE_PAIRS + CONTROL_PAIRS
    for a_, b_ in pairs:
        ma, mb = f"cal_{a_}_free", f"cal_{b_}_free"
        if ma not in pt or mb not in pt:
            continue
        sub = cells[cells.variant.isin([a_, b_])]
        col = set(sd * N_FOLDS + fo for sd, fo in sub[sub.collapsed][["seed", "fold"]].itertuples(index=False))
        weak = col | set(sd * N_FOLDS + fo for sd, fo in sub[sub.weak][["seed", "fold"]].itertuples(index=False))
        for tag, drop in (("all", ()), ("drop_collapsed", sorted(col)), ("drop_collapsed_or_weak", sorted(weak))):
            for r in paired_from_pivot(pt, ma, mb, drop):
                rows.append({**r, "subset": tag, "dropped_clusters": str(list(drop)), "pair": f"{a_}-{b_}"})
    lev = []
    for v in ("p0", "p1", "p2", "p4", "p0w", "p0w4", "p2l", "p2c"):
        m = f"cal_{v}_free"
        if m not in pt:
            continue
        sub = cells[cells.variant == v]
        col = set(sd * N_FOLDS + fo for sd, fo in sub[sub.collapsed][["seed", "fold"]].itertuples(index=False))
        weak = col | set(sd * N_FOLDS + fo for sd, fo in sub[sub.weak][["seed", "fold"]].itertuples(index=False))
        for tag, drop in (("all", ()), ("drop_collapsed", sorted(col)), ("drop_collapsed_or_weak", sorted(weak))):
            g = pt[m].dropna().reset_index()
            g = g[~g.cluster.isin(list(drop))]
            for B, gg in g.groupby("B"):
                mu, ci, cif, cit, ci2 = _cis(gg[m].values, gg.cluster.values, gg.fold.values, gg.target.values)
                lev.append({"variant": v, "subset": tag, "B": B, "r2_mean": mu, "r2_ci95": ci, "r2_ci95_fold": cif,
                            "r2_ci95_2way": ci2,
                            "n_clusters": gg.cluster.nunique(), "dropped_clusters": str(list(drop))})
    return pd.DataFrame(rows), pd.DataFrame(lev), cells


def hyser_main():
    df = load_hyser()
    tab = table(df)
    tab.to_csv(RES / "hyser_table.csv", index=False)
    st = seed_table(df)
    st.to_csv(RES / "hyser_seed_table.csv", index=False)
    methods = sorted(m for m in df.method.unique() if m.startswith("cal_"))
    pt = _pivot_units(df, "r2")
    pr = []
    base = {B: max(tab[(tab.method.isin(HYSER_DEPLOYABLE)) & (tab.B == B)].set_index("method")["r2_mean"].items(),
                   key=lambda kv: kv[1]) for B in sorted(df.B.unique())}
    for m in methods:
        for ref in HYSER_DEPLOYABLE + ["ui_prior_s1_calnorm", "ui_prior_mean_calnorm"]:
            pr += paired_from_pivot(pt, m, ref)
        # vs best deployable baseline per B (selected post hoc on target means: favours baselines)
        for B in sorted(df.B.unique()):
            bn = base[B][0]
            gg = pt.loc[B][[m, bn]].dropna()
            d = (gg[m] - gg[bn]).values
            tid = gg.index.get_level_values("target").values
            tgt = pd.Series(d, index=tid).groupby(level=0).mean().values
            mu, ci, cif, cit, ci2 = _cis(d, gg.index.get_level_values("cluster").values,
                                         gg.index.get_level_values("fold").values, tid)
            pr.append({"a": m, "b": f"best_deployable({bn})", "B": B, "diff_mean": mu, "diff_ci95": ci,
                       "diff_ci95_fold": cif, "diff_ci95_2way": ci2, "diff_ci95_target": cit,
                       "excludes_zero": bool(abs(mu) > ci), "excludes_zero_fold": bool(abs(mu) > cif),
                       "excludes_zero_2way": bool(abs(mu) > ci2), "n_clusters": int(gg.index.get_level_values("cluster").nunique()),
                       "n": len(tgt), "frac_targets_a_better": float((tgt > 0).mean())})
    # variant-vs-variant paired differences inside each prior source (pooling question + capacity controls)
    vv = []
    for src in ("mean", "own", "free"):
        for a_, b_ in BASE_PAIRS + (CONTROL_PAIRS if src == "free" else []):
            vv += paired_from_pivot(pt, f"cal_{a_}_{src}", f"cal_{b_}_{src}")
    pd.DataFrame(pr).to_csv(RES / "hyser_paired.csv", index=False)
    pd.DataFrame(vv).to_csv(RES / "hyser_variant_pairs.csv", index=False)
    # per-backbone-seed variant differences (target-level CI inside one seed; one calibrator per fold)
    bys = []
    for sd in sorted(df.seed.unique()):
        pts = _pivot_units(df[df.seed == sd], "r2")
        for src in ("mean", "own", "free"):
            for a_, b_ in BASE_PAIRS + (CONTROL_PAIRS if src == "free" else []):
                for r in paired_from_pivot(pts, f"cal_{a_}_{src}", f"cal_{b_}_{src}"):
                    bys.append({**r, "seed": sd, "src": src})
    pd.DataFrame(bys).to_csv(RES / "hyser_variant_pairs_by_seed.csv", index=False)
    info = training_info()
    cs, lev, cells = collapse_sensitivity(df, info)
    cs.to_csv(RES / "hyser_free_collapse_sensitivity.csv", index=False)
    lev.to_csv(RES / "hyser_free_collapse_levels.csv", index=False)
    cells.to_csv(RES / "hyser_free_cells.csv", index=False)
    return df, tab, st, pd.DataFrame(pr), base


def df_md(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([f"`{i}`"] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines) + "\n"


def md_table(tab, metric, order, budgets, title):
    """Markdown table of mean +- ci per method and B."""
    lines = [f"### {title}", "", "| method | " + " | ".join(f"B={b}" for b in budgets) + " |",
             "|---|" + "---|" * len(budgets)]
    idx = tab.set_index(["method", "B"])
    for m in order:
        cells = []
        for b in budgets:
            if (m, b) in idx.index:
                r = idx.loc[(m, b)]
                cells.append(f"{r[metric + '_mean']:.3f} ± {r[metric + '_ci95']:.3f}")
            else:
                cells.append("-")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def hyser_markdown(tab, st, pr):
    budgets = sorted(tab.B.unique())
    cal = sorted(m for m in tab.method.unique() if m.startswith("cal_"))
    order = ["reuse_s1_srcnorm", "reuse_s1_calnorm", "mean_table_calnorm", "mean_loso_calnorm", "rownorm_ridge_perB",
             "ui_prior_s1_calnorm", "ui_prior_mean_calnorm"] + cal
    md = md_table(tab, "r2", order, budgets, "Hyser H-A query R2 (physical units, macro over 5 forces), mean ± 95% CI over 20 targets")
    sp = st.set_index(["method", "B"])["seed_spread"].unstack("B")
    md += "\n### Backbone-seed spread (|seed0 - seed1| of the target-mean R2)\n\n" + df_md(sp.round(3)) + "\n"
    sel = pr[pr.b.str.startswith("best_deployable")]
    md += "\n### Paired difference to the best deployable baseline per B (picked post hoc on target means)\n\n"
    piv = sel.assign(cell=lambda d: d.diff_mean.map("{:+.3f}".format) + " ± " + d.diff_ci95.map("{:.3f}".format)) \
        .pivot_table(index="a", columns="B", values="cell", aggfunc="first")
    md += df_md(piv) + "\n"
    (RES / "hyser_tables.md").write_text(md)
    return md


# ------------------------------------------------------------------- DB6 ----

DB6_REF = ["reuse_latest_srcnorm", "reuse_latest_calnorm", "mean_table_calnorm", "mean_loso_calnorm", "rownorm_ridge",
           "ui_prior_latest_calnorm", "ui_prior_mean_calnorm"]
DB6_DEPLOYABLE = ["reuse_latest_calnorm", "mean_table_calnorm", "mean_loso_calnorm", "rownorm_ridge"]
DB6_B = [1, 2, 4, 8, 16]
DB6_PAIRS = [("p1", "p0"), ("p2", "p0"), ("p4", "p0"), ("p2", "p1"), ("p4", "p1"), ("p4", "p2"),
             ("p2", "p0w"), ("p4", "p0w"), ("p0w", "p0"), ("p2", "p2l"), ("p2l", "p0w"), ("p2l", "p0")]   # last 6: capacity controls


def load_db6(order="original", seeds=(0, 1), tag="db6v2"):
    keep = ["subject", "day", "tod", "gap", "B", "draw", "seed", "method", "bal_acc", "macro_f1", "vote_acc", "acc"]
    ref_dir = {"original": "eval_seed{s}", "shuffled": "extra_shuffled_order_seed{s}"}[order]
    rows = []
    for s in seeds:
        p = ROOT / "results/db6_v2_20261016" / ref_dir.format(s=s) / "rows.jsonl"
        if p.exists():
            rows += [{k: r.get(k) for k in keep} for r in read_jsonl(p) if r["method"] in DB6_REF and r["B"] in DB6_B]
        for sub in ("", "fix"):               # "fix" = post-review capacity controls (p0w / p2l, identity-free)
            q = RES / sub / tag / f"seed{s}" / f"rows_{order}.jsonl"
            if q.exists():
                rows += [{k: r.get(k) for k in keep} for r in read_jsonl(q) if r["method"].startswith("cal_")]
    df = pd.DataFrame(rows)
    if order == "shuffled" and len(df):            # the shuffled diagnostic only covers B in {4,16}, draws 0-2
        df = df[df.B.isin([4, 16]) & (df.draw < 3)]
    return df


def per_subject(df, metric):
    return df.groupby(["method", "B", "subject"])[metric].mean().reset_index().rename(columns={metric: "val"})


def db6_table(df, metric):
    ps = per_subject(df, metric)
    rows = []
    for (m, b), g in ps.groupby(["method", "B"]):
        mu, ci = tci(g["val"].values)
        rows.append({"method": m, "B": b, "n_subjects": len(g), f"{metric}_mean": mu, f"{metric}_ci95": ci})
    return pd.DataFrame(rows)


def db6_gap_table(df, metric):
    g = df.groupby(["method", "B", "gap", "subject"])[metric].mean().reset_index()
    rows = []
    for (m, b, gp), gg in g.groupby(["method", "B", "gap"]):
        mu, ci = tci(gg[metric].values)
        rows.append({"method": m, "B": b, "gap": gp, f"{metric}_mean": mu, f"{metric}_ci95": ci})
    return pd.DataFrame(rows)


def db6_seed_table(df, metric):
    g = df.groupby(["method", "B", "seed", "subject"])[metric].mean().reset_index()
    g = g.groupby(["method", "B", "seed"])[metric].mean().unstack("seed")
    g.columns = [f"seed{c}" for c in g.columns]
    g["seed_spread"] = g.max(axis=1) - g.min(axis=1)
    return g.reset_index()


def db6_paired(df, a, b, metric):
    pt = per_subject(df, metric).pivot_table(index=["B", "subject"], columns="method", values="val")
    out = []
    for B, g in pt.groupby(level=0):
        if a not in g or b not in g:
            continue
        d = (g[a] - g[b]).dropna().values
        mu, ci = tci(d)
        out.append({"a": a, "b": b, "B": B, "diff_mean": mu, "diff_ci95": ci, "n": len(d),
                    "frac_subjects_a_better": float((d > 0).mean())})
    return out


def db6_main(order="original", tag="db6v2", suffix=""):
    df = load_db6(order, tag=tag)
    out = {}
    for metric in ("bal_acc", "macro_f1"):
        tab = db6_table(df, metric)
        tab.to_csv(RES / f"db6_{order}{suffix}_table_{metric}.csv", index=False)
        out[metric] = tab
    db6_gap_table(df, "bal_acc").to_csv(RES / f"db6_{order}{suffix}_gap_table_bal_acc.csv", index=False)
    st = db6_seed_table(df, "bal_acc")
    st.to_csv(RES / f"db6_{order}{suffix}_seed_table_bal_acc.csv", index=False)
    methods = sorted(m for m in df.method.unique() if m.startswith("cal_"))
    tab = out["bal_acc"]
    bs = sorted(int(b) for b in df.B.unique())
    best = {B: max(tab[(tab.method.isin(DB6_DEPLOYABLE)) & (tab.B == B)].set_index("method")["bal_acc_mean"].items(),
                   key=lambda kv: kv[1]) for B in bs}
    pr = []
    pt = per_subject(df, "bal_acc").pivot_table(index=["B", "subject"], columns="method", values="val")
    for m in methods:
        for ref in DB6_DEPLOYABLE + ["ui_prior_latest_calnorm", "ui_prior_mean_calnorm"]:
            pr += db6_paired(df, m, ref, "bal_acc")
        for B in bs:
            bn = best[B][0]
            d = (pt.loc[B][m] - pt.loc[B][bn]).dropna().values
            mu, ci = tci(d)
            pr.append({"a": m, "b": f"best_deployable({bn})", "B": B, "diff_mean": mu, "diff_ci95": ci, "n": len(d),
                       "frac_subjects_a_better": float((d > 0).mean())})
    pd.DataFrame(pr).to_csv(RES / f"db6_{order}{suffix}_paired.csv", index=False)
    vv = []
    for src in ("mean", "own", "free"):
        for a_, b_ in DB6_PAIRS:
            if f"cal_{a_}_{src}" in methods and f"cal_{b_}_{src}" in methods:
                vv += db6_paired(df, f"cal_{a_}_{src}", f"cal_{b_}_{src}", "bal_acc")
    pd.DataFrame(vv).to_csv(RES / f"db6_{order}{suffix}_variant_pairs.csv", index=False)
    # per backbone seed (one calibrator per seed, shared by all subjects): the 2-seed average hides seed dependence
    bys = []
    for sd in sorted(df.seed.unique()):
        d1 = df[df.seed == sd]
        for src in ("mean", "own", "free"):
            for a_, b_ in DB6_PAIRS:
                if f"cal_{a_}_{src}" in methods and f"cal_{b_}_{src}" in methods:
                    for r in db6_paired(d1, f"cal_{a_}_{src}", f"cal_{b_}_{src}", "bal_acc"):
                        bys.append({**r, "seed": sd, "src": src})
    pd.DataFrame(bys).to_csv(RES / f"db6_{order}{suffix}_variant_pairs_by_seed.csv", index=False)
    return df, out, st, pd.DataFrame(pr), best


def db6_markdown(out, st, pr, order="original", suffix=""):
    budgets = [int(b) for b in sorted(out["bal_acc"].B.unique())]
    cal = sorted(m for m in out["bal_acc"].method.unique() if m.startswith("cal_"))
    order_m = ["reuse_latest_srcnorm", "reuse_latest_calnorm", "mean_table_calnorm", "mean_loso_calnorm", "rownorm_ridge",
               "ui_prior_latest_calnorm", "ui_prior_mean_calnorm"] + cal
    md = md_table(out["bal_acc"], "bal_acc", order_m, budgets,
                  f"DB6 v2 ({order} order) per-bin balanced accuracy, mean ± 95% CI over 10 subjects (all gaps)")
    md += "\n" + md_table(out["macro_f1"], "macro_f1", order_m, budgets, f"DB6 v2 ({order} order) macro-F1")
    sp = st.set_index(["method", "B"])["seed_spread"].unstack("B")
    md += "\n### Backbone-seed spread of the subject-mean balanced accuracy\n\n" + df_md(sp.round(3)) + "\n"
    sel = pr[pr.b.str.startswith("best_deployable")]
    piv = sel.assign(cell=lambda d: d.diff_mean.map("{:+.3f}".format) + " ± " + d.diff_ci95.map("{:.3f}".format)) \
        .pivot_table(index="a", columns="B", values="cell", aggfunc="first")
    md += "\n### Paired difference to the best deployable baseline per B (post hoc on subject means)\n\n" + df_md(piv)
    (RES / f"db6_{order}{suffix}_tables.md").write_text(md)
    return md


def db6_cpu_check(order="original", tag="db6v2"):
    """Agreement of the CPU-decoded chk_* rows with the stored GPU reference rows (same draws)."""
    ref_dir = {"original": "eval_seed{s}", "shuffled": "extra_shuffled_order_seed{s}"}[order]
    diffs = []
    for s in (0, 1):
        ref = {}
        p = ROOT / "results/db6_v2_20261016" / ref_dir.format(s=s) / "rows.jsonl"
        for r in read_jsonl(p):
            if r["method"] in ("reuse_latest_calnorm", "mean_table_calnorm") and r["B"] in DB6_B:
                ref[(r["target"], r["B"], r["draw"], r["method"])] = r
        q = RES / tag / f"seed{s}" / f"rows_{order}.jsonl"
        for r in read_jsonl(q):
            if r["method"].startswith("chk_"):
                g = ref.get((r["target"], r["B"], r["draw"], r["method"][4:]))
                if g is None:
                    continue
                assert g["cal_trials"] == r["cal_trials"]
                diffs.append({"seed": s, "method": r["method"][4:], "B": r["B"], "d_bal_acc": r["bal_acc"] - g["bal_acc"],
                              "d_macro_f1": r["macro_f1"] - g["macro_f1"]})
    d = pd.DataFrame(diffs)
    if d.empty:
        return {}
    out = {"n": int(len(d)), "mean_d_bal_acc": float(d.d_bal_acc.mean()), "mean_abs_d_bal_acc": float(d.d_bal_acc.abs().mean()),
           "max_abs_d_bal_acc": float(d.d_bal_acc.abs().max()), "frac_abs_gt_0.005": float((d.d_bal_acc.abs() > 0.005).mean()),
           "by_B_mean_d": {int(b): float(v) for b, v in d.groupby("B").d_bal_acc.mean().items()}}
    (RES / f"db6_cpu_vs_gpu_check_{order}.json").write_text(json.dumps(out, indent=1))
    return out


def db6_own_vs_reuse(order="original", tag="db6v2"):
    """Per backbone seed: how far the `own`-prior calibrators are from the identical-pipeline reuse row
    (``chk_reuse_latest_calnorm``, same CPU decode, same draws).  Seed 0 has best_step 0 for all variants."""
    out = {}
    for sd in (0, 1):
        q = RES / tag / f"seed{sd}" / f"rows_{order}.jsonl"
        if not q.exists():
            continue
        rows = read_jsonl(q)
        ref = {(r["target"], r["B"], r["draw"]): r["bal_acc"] for r in rows if r["method"] == "chk_reuse_latest_calnorm"}
        for v in VARIANTS_4:
            d = np.array([r["bal_acc"] - ref[(r["target"], r["B"], r["draw"])] for r in rows
                          if r["method"] == f"cal_{v}_own" and (r["target"], r["B"], r["draw"]) in ref])
            out[f"seed{sd}/cal_{v}_own"] = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None,
                                            "max_abs_diff": float(np.abs(d).max()) if len(d) else None,
                                            "frac_abs_gt_1e-4": float((np.abs(d) > 1e-4).mean()) if len(d) else None}
    (RES / f"db6_{order}_own_vs_reuse_check.json").write_text(json.dumps(out, indent=1))
    return out


VARIANTS_4 = ("p0", "p1", "p2", "p4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["hyser", "db6"])
    a = ap.parse_args()
    if a.dataset == "db6":
        for order in ("original", "shuffled"):
            df, out, st, pr, best = db6_main(order)
            print(order)
            print(out["bal_acc"].pivot_table(index="method", columns="B", values="bal_acc_mean").round(3).to_string())
            if order == "original":
                print(json.dumps(db6_own_vs_reuse(order), indent=1))
            try:
                print(db6_cpu_check(order))
            except FileNotFoundError:
                pass
            db6_markdown(out, st, pr, order)
    if a.dataset == "hyser":
        df, tab, st, pr, base = hyser_main()
        piv = tab.pivot_table(index="method", columns="B", values="r2_mean")
        print(piv.round(3).to_string())
        hyser_markdown(tab, st, pr)
