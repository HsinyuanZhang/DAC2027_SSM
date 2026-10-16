#!/usr/bin/env python
"""Aggregate Q-D rows (sEMG calibrators + reused reference rows) into compact csv/json.

Hyser: R2 per target (mean over draws and backbone seeds), t-CI over 20 targets.
DB6 : balanced accuracy / macro-F1 per subject (draws -> sessions of a day -> seeds), t-CI over 10 subjects.
"""
import argparse
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
        p = RES / "hyser" / f"seed{s}" / "rows.jsonl"
        if p.exists():
            cal += [r for r in read_jsonl(p) if r["method"].startswith("cal_")]
    keep = ["target", "B", "draw", "seed", "method", "r2", "r2_pooled", "unit_ratio"]
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in ref + cal])
    return df


def per_target(df, metric):
    return (df.groupby(["method", "B", "target"])[metric].mean().reset_index())


def table(df, metric="r2"):
    pt = per_target(df, metric)
    rows = []
    for (m, b), g in pt.groupby(["method", "B"]):
        mu, ci = tci(g[metric].values)
        rows.append({"method": m, "B": b, "n_targets": len(g), f"{metric}_mean": mu, f"{metric}_ci95": ci})
    return pd.DataFrame(rows)


def seed_table(df, metric="r2"):
    g = df.groupby(["method", "B", "seed", "target"])[metric].mean().reset_index()
    g = g.groupby(["method", "B", "seed"])[metric].mean().unstack("seed")
    g.columns = [f"seed{c}" for c in g.columns]
    g["seed_spread"] = g.max(axis=1) - g.min(axis=1)
    return g.reset_index()


def paired(df, a, b, metric="r2"):
    pt = per_target(df, metric).pivot_table(index=["B", "target"], columns="method", values=metric)
    out = []
    for B, g in pt.groupby(level=0):
        if a not in g or b not in g:
            continue
        d = (g[a] - g[b]).dropna().values
        mu, ci = tci(d)
        out.append({"a": a, "b": b, "B": B, "diff_mean": mu, "diff_ci95": ci, "n": len(d),
                    "frac_targets_a_better": float((d > 0).mean())})
    return out


def hyser_main():
    df = load_hyser()
    tab = table(df)
    tab.to_csv(RES / "hyser_table.csv", index=False)
    st = seed_table(df)
    st.to_csv(RES / "hyser_seed_table.csv", index=False)
    methods = sorted(m for m in df.method.unique() if m.startswith("cal_"))
    pr = []
    base = {B: max(tab[(tab.method.isin(HYSER_DEPLOYABLE)) & (tab.B == B)].set_index("method")["r2_mean"].items(),
                   key=lambda kv: kv[1]) for B in sorted(df.B.unique())}
    for m in methods:
        for ref in HYSER_DEPLOYABLE + ["ui_prior_s1_calnorm", "ui_prior_mean_calnorm"]:
            pr += paired(df, m, ref)
        # vs best deployable baseline per B (selected post hoc on target means: favours baselines)
        pt = per_target(df, "r2").pivot_table(index=["B", "target"], columns="method", values="r2")
        for B in sorted(df.B.unique()):
            bn = base[B][0]
            d = (pt.loc[B][m] - pt.loc[B][bn]).dropna().values
            mu, ci = tci(d)
            pr.append({"a": m, "b": f"best_deployable({bn})", "B": B, "diff_mean": mu, "diff_ci95": ci,
                       "n": len(d), "frac_targets_a_better": float((d > 0).mean())})
    # variant-vs-variant paired differences inside each prior source (pooling question)
    vv = []
    for src in ("mean", "own", "free"):
        for a_, b_ in [("p1", "p0"), ("p2", "p0"), ("p4", "p0"), ("p2", "p1"), ("p4", "p1"), ("p4", "p2")]:
            if f"cal_{a_}_{src}" in methods and f"cal_{b_}_{src}" in methods:
                vv += paired(df, f"cal_{a_}_{src}", f"cal_{b_}_{src}")
    pd.DataFrame(pr).to_csv(RES / "hyser_paired.csv", index=False)
    pd.DataFrame(vv).to_csv(RES / "hyser_variant_pairs.csv", index=False)
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


def load_db6(order="original", seeds=(0, 1), tag="db6v2"):
    keep = ["subject", "day", "tod", "gap", "B", "draw", "seed", "method", "bal_acc", "macro_f1", "vote_acc", "acc"]
    ref_dir = {"original": "eval_seed{s}", "shuffled": "extra_shuffled_order_seed{s}"}[order]
    rows = []
    for s in seeds:
        p = ROOT / "results/db6_v2_20261016" / ref_dir.format(s=s) / "rows.jsonl"
        if p.exists():
            rows += [{k: r.get(k) for k in keep} for r in read_jsonl(p) if r["method"] in DB6_REF and r["B"] in DB6_B]
        q = RES / tag / f"seed{s}" / f"rows_{order}.jsonl"
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
        for a_, b_ in [("p1", "p0"), ("p2", "p0"), ("p4", "p0"), ("p2", "p1"), ("p4", "p1"), ("p4", "p2")]:
            if f"cal_{a_}_{src}" in methods and f"cal_{b_}_{src}" in methods:
                vv += db6_paired(df, f"cal_{a_}_{src}", f"cal_{b_}_{src}", "bal_acc")
    pd.DataFrame(vv).to_csv(RES / f"db6_{order}{suffix}_variant_pairs.csv", index=False)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["hyser", "db6"])
    a = ap.parse_args()
    if a.dataset == "db6":
        for order in ("original", "shuffled"):
            df, out, st, pr, best = db6_main(order)
            print(order)
            print(out["bal_acc"].pivot_table(index="method", columns="B", values="bal_acc_mean").round(3).to_string())
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
