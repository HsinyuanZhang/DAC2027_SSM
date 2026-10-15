#!/usr/bin/env python
"""v1 (state reset at every trial) vs v2 (one contiguous causal pass) tables for DB6.

usage: compare_db6_v1_v2.py <out.md> [--v1 reference_rows_main.csv] [--v2 reference_rows_main.csv]
                            [--extra name=reference_rows_main.csv ...]

Reads 'all gaps' rows (balanced accuracy, mean +- 95% CI over 10 subjects) and writes a markdown table of
selected methods x budgets.  v1 names: rownorm_ridge_v2 -> rownorm_ridge.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["reuse_latest_srcnorm", "reuse_latest_calnorm", "mean_table_calnorm", "rownorm_ridge",
           "ui_prior_latest_calnorm", "within_table_latestinit_long"]
BUDGETS = [1, 2, 4, 8, 16, 42]


def table(csv, rename=None):
    df = pd.read_csv(csv)
    if rename:
        df["method"] = df["method"].replace(rename)
    df = df[df.gap.astype(str) == "all"]
    return {(r.method, int(r.B)): (r.bal_acc, r.bal_acc_ci95) for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v1", default=str(ROOT / "results/analysis/db6_20261015/reference_rows_main.csv"))
    ap.add_argument("--v2", default=str(ROOT / "results/analysis/db6_v2_20261016/reference_rows_main.csv"))
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cols = {"v1 (reset)": table(a.v1, {"rownorm_ridge_v2": "rownorm_ridge"})}
    for e in a.extra:
        name, path = e.split("=", 1)
        cols[name] = table(path)
    cols["v2 (contiguous)"] = table(a.v2)
    lines = []
    for b in BUDGETS:
        lines.append(f"\n**B = {b}**\n")
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for m in METHODS:
            cells = []
            for t in cols.values():
                v = t.get((m, b))
                cells.append("-" if v is None else f"{v[0]:.3f} ± {v[1]:.3f}")
            if any(c != "-" for c in cells):
                lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
