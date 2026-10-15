#!/usr/bin/env python
"""Markdown tables for the DB6 v2 report from the summarize_db6.py outputs.

usage: format_db6_tables.py <analysis_dir> <out.md>
"""
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
main = pd.read_csv(d / "reference_rows_main.csv")
paired = pd.read_csv(d / "paired_diffs_main.csv")
main["gap"] = main["gap"].astype(str)
paired["gap"] = paired["gap"].astype(str)
LABEL = {"reuse_latest_srcnorm": "`reuse_latest_srcnorm`（零 target 数据）", "reuse_latest_calnorm": "`reuse_latest_calnorm`",
         "mean_table_calnorm": "`mean_table_calnorm`", "mean_loso_calnorm": "`mean_loso_calnorm`",
         "rownorm_ridge": "`rownorm_ridge`（部署型 closed-form，block-free 选择）",
         "rownorm_ridge_origsel": "`rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构）",
         "ui_prior_latest_srcnorm": "`ui_prior_latest_srcnorm`（参照）", "ui_prior_latest_calnorm": "`ui_prior_latest_calnorm`（参照）",
         "ui_prior_mean_calnorm": "`ui_prior_mean_calnorm`（参照）",
         "within_table_latestinit_long": "`within_table_latestinit_long`（诊断）", "within_table_supinit": "`within_table_supinit`（诊断）"}
ORDER = list(LABEL)
BS = [1, 2, 4, 8, 16, 42]


def cell(df, m, b, gap="all", col="bal_acc"):
    r = df[(df.method == m) & (df.B == b) & (df.gap == gap)]
    if r.empty:
        return "-"
    r = r.iloc[0]
    ci = r[col + "_ci95"]
    return f"{r[col]:.3f} ± {ci:.3f}" if pd.notna(ci) else f"{r[col]:.3f}"


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


out = []
out.append("### T1 balanced accuracy, all gaps\n")
out.append(md(["method"] + [f"B={b}" + ("（诊断）" if b == 42 else "") for b in BS],
              [[LABEL[m]] + [cell(main, m, b) for b in BS] for m in ORDER if (main.method == m).any()]))
out.append("\n### T2 balanced accuracy per gap (B=4 / B=16)\n")
rows = []
for m in ORDER[:9]:
    if (main.method == m).any():
        rows.append([LABEL[m]] + [f"{main[(main.method==m)&(main.B==4)&(main.gap==g)].bal_acc.iloc[0]:.3f} / "
                                  f"{main[(main.method==m)&(main.B==16)&(main.gap==g)].bal_acc.iloc[0]:.3f}" for g in ("1", "2", "3")])
out.append(md(["method", "gap 1 (day 3)", "gap 2 (day 4)", "gap 3 (day 5)"], rows))
out.append("\n### T3 macro-F1 and vote accuracy (all gaps; B = 1 / 4 / 16)\n")
rows = []
for m in ORDER[:9]:
    if (main.method == m).any():
        rows.append([LABEL[m], " / ".join(f"{main[(main.method==m)&(main.B==b)&(main.gap=='all')].macro_f1.iloc[0]:.3f}" for b in (1, 4, 16)),
                     " / ".join(f"{main[(main.method==m)&(main.B==b)&(main.gap=='all')].vote_acc.iloc[0]:.3f}" for b in (1, 4, 16))])
out.append(md(["method", "macro-F1 B=1 / 4 / 16", "vote acc B=1 / 4 / 16"], rows))


def pair_row(a, b, gap="all", bs=BS):
    cells = []
    for B in bs:
        r = paired[(paired.a == a) & (paired.b == b) & (paired.B == B) & (paired.gap == gap)]
        cells.append("-" if r.empty else f"{r.iloc[0].diff_bal_acc:+.3f} ± {r.iloc[0].diff_bal_acc_ci95:.3f}")
    return cells


out.append("\n### T4 paired differences (a - b), balanced accuracy, all gaps, 95% CI over 10 subjects\n")
PAIRS = [("reuse_latest_calnorm", "reuse_latest_srcnorm"), ("ui_prior_latest_calnorm", "reuse_latest_calnorm"),
         ("ui_prior_latest_calnorm", "rownorm_ridge"), ("rownorm_ridge", "reuse_latest_calnorm"),
         ("rownorm_ridge_origsel", "rownorm_ridge"), ("reuse_latest_calnorm", "mean_table_calnorm"),
         ("mean_table_calnorm", "mean_loso_calnorm"), ("ui_prior_latest_calnorm", "ui_prior_mean_calnorm"),
         ("within_table_latestinit_long", "ui_prior_latest_calnorm"), ("within_table_supinit", "ui_prior_latest_calnorm")]
out.append(md(["a - b"] + [f"B={b}" for b in BS], [[f"`{a}` - `{b}`"] + pair_row(a, b) for a, b in PAIRS]))
out.append("\n### T5 own-subject latest vs population mean table, per gap (a = `mean_table_calnorm`, b = `reuse_latest_calnorm`)\n")
out.append(md(["gap"] + [f"B={b}" for b in (4, 8, 16)],
              [[g] + pair_row("mean_table_calnorm", "reuse_latest_calnorm", g, (4, 8, 16)) for g in ("1", "2", "3")]))
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print("\n".join(out))
