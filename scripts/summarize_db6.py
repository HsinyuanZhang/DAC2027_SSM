#!/usr/bin/env python
"""Aggregate DB6 D-A rows.jsonl files into reference tables (mean and 95% CI over subjects).

usage: summarize_db6.py <out_dir> <rows.jsonl> [<rows.jsonl> ...]   (one file per backbone seed)

Aggregation: row -> mean over draws (per target session) -> mean over the subject's sessions of the
same day gap -> mean over seeds -> mean over the 10 subjects; CI = t(0.975, n-1) * sd / sqrt(n).
Gap = target day - 2 (days 3, 4, 5 -> gap 1, 2, 3); 'all' averages the three gaps per subject.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["bal_acc", "macro_f1", "vote_acc", "acc"]


def t_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))


def load(paths):
    rows = []
    for si, p in enumerate(paths):
        for ln in Path(p).read_text().splitlines():
            r = json.loads(ln)
            r["seed_file"] = si
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def per_subject_table(df):
    """One value per (seed_file, method, B, subject, gap): mean over draws, then over the day's sessions."""
    g = df.groupby(["seed_file", "method", "B", "subject", "gap", "tod", "target"])[METRICS].mean().reset_index()
    g = g.groupby(["seed_file", "method", "B", "subject", "gap"])[METRICS].mean().reset_index()
    allg = g.groupby(["seed_file", "method", "B", "subject"])[METRICS].mean().reset_index()
    allg["gap"] = 0                                   # 0 encodes 'all gaps'
    return pd.concat([g, allg], ignore_index=True)


def summarize(ps):
    ps = ps.groupby(["method", "B", "subject", "gap"])[METRICS].mean().reset_index()   # average seeds
    out = []
    for (m, b, gap), grp in ps.groupby(["method", "B", "gap"]):
        row = {"method": m, "B": b, "gap": "all" if gap == 0 else gap, "n_subjects": len(grp)}
        for k in METRICS:
            row[k], row[k + "_ci95"] = t_ci(grp[k].values)
        out.append(row)
    return pd.DataFrame(out), ps


def paired(ps, pairs):
    out = []
    for a, b in pairs:
        for (B, gap), _ in ps.groupby(["B", "gap"]):
            x = ps[(ps.method == a) & (ps.B == B) & (ps.gap == gap)].set_index("subject")
            y = ps[(ps.method == b) & (ps.B == B) & (ps.gap == gap)].set_index("subject")
            idx = x.index.intersection(y.index)
            if len(idx) < 2:
                continue
            row = {"a": a, "b": b, "B": B, "gap": "all" if gap == 0 else gap, "n_subjects": len(idx)}
            for k in ("bal_acc", "macro_f1"):
                row["diff_" + k], row["diff_" + k + "_ci95"] = t_ci(x.loc[idx, k].values - y.loc[idx, k].values)
            out.append(row)
    return pd.DataFrame(out)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    paths = sys.argv[2:]
    df = load(paths)
    ps_all = per_subject_table(df)
    summ, ps = summarize(ps_all)
    summ.sort_values(["method", "gap", "B"], key=lambda c: c.astype(str)).to_csv(out / "reference_rows_main.csv", index=False)
    ps.to_csv(out / "per_subject_main.csv", index=False)
    for si, p in enumerate(paths):
        s_, _ = summarize(ps_all[ps_all.seed_file == si])
        s_.to_csv(out / f"reference_rows_seed{si}.csv", index=False)
    pairs = [("reuse_latest_calnorm", "reuse_latest_srcnorm"), ("reuse_latest_calnorm", "mean_table_calnorm"),
             ("mean_table_calnorm", "reuse_latest_calnorm"), ("mean_table_calnorm", "mean_loso_calnorm"),
             ("rownorm_ridge", "reuse_latest_calnorm"), ("rownorm_ridge_origsel", "rownorm_ridge"),
             ("ui_prior_latest_calnorm", "reuse_latest_calnorm"),
             ("ui_prior_latest_calnorm", "rownorm_ridge"), ("ui_prior_mean_calnorm", "mean_table_calnorm"),
             ("ui_prior_latest_calnorm", "ui_prior_mean_calnorm"),
             ("within_table_latestinit_long", "ui_prior_latest_calnorm"),
             ("within_table_supinit", "ui_prior_latest_calnorm")]
    paired(ps, pairs).to_csv(out / "paired_diffs_main.csv", index=False)
    # class coverage of the calibration draws and unique-draw counts
    ncount = df[df.B == 42].groupby("seed_file")["target"].nunique().to_dict()
    print("targets with B=42 rows per seed file:", ncount)
    cov = df[df.method == "reuse_latest_calnorm"].copy()
    cov["n_classes_seen"] = cov["coverage"].apply(lambda c: int(sum(v > 0 for v in c))) + 1   # + rest
    cov_tab = cov.groupby("B").agg(n_classes_seen_mean=("n_classes_seen", "mean"),
                                   n_classes_seen_min=("n_classes_seen", "min"),
                                   max_minus_min_grasp_count=("coverage", lambda s: int(max(max(c) - min(c) for c in s))),
                                   n_rows=("coverage", "size")).reset_index()
    cov_tab.to_csv(out / "calibration_coverage.csv", index=False)
    # seed disagreement and H flags
    flags = {"h1_all_true": bool(df["h1_trunk_unchanged"].all()), "h5_all_true": bool(df["h5_pass"].all()),
             "n_rows": int(len(df)), "rows_by_role": {k: int(v) for k, v in df.groupby("role").size().items()}}
    (out / "h_flags_main.json").write_text(json.dumps(flags, indent=1))
    summ.to_json(out / "reference_rows_main.json", orient="records", indent=1)
    print(summ[summ.gap == "all"][["method", "B", "bal_acc", "bal_acc_ci95", "macro_f1", "vote_acc"]].to_string())


if __name__ == "__main__":
    main()
