#!/usr/bin/env python
"""Aggregate the expansion rows (sub-C + sub-M days; p_prior, p0, r1s, r2, r3) with session-clustered CIs.

Why clustering: the 27 sub-C (backbone, target) pairs cover only 19 distinct target sessions (0713 and 0716 are
evaluated by three backbones; 0629, 0706, 0709, 0714 by two) and sm_0611's targets 0616 / 0617 were P1 design
targets.  Rows of the same session under different backbones are NOT independent evidence, so the unit of
inference is the target SESSION:

  1. per (backbone, target, B, comparison): mean paired difference over draws x seeds;
  2. per (subject, stratum, selection scope, B, comparison, SESSION): mean over the backbones that evaluate it;
  3. across sessions: mean, percentile bootstrap CI over sessions (10,000 resamples, seed 0) and a t interval.

``backbone_spread.csv`` reports the backbone-level spread next to it (per-backbone means across targets and
the within-session spread between backbones).  Selection scopes are never merged: the rows of the reused sm_0307
trunk are IS (selection session seen by the trunk), all new backbones are OOS (``selection_oos``); both are
separate groups in every table.  Rows without ``rule_version == expansion_v2`` (P1 / P3 / drift-curve rows,
selected on 20140307) are refused.

  python scripts/analyze_expansion.py --rows results/expansion_20261016/rows \
      --out results/analysis/expansion_20261016/summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.atomic_jsonl import read_jsonl                         # noqa: E402

RULE_VERSION = "expansion_v2"
COMPARISONS = {"p0-p_prior": ("p0", "p_prior"), "r1s-p0": ("r1s", "p0"), "r2-p0": ("r2", "p0"),
               "r3-p0": ("r3", "p0"), "r3-r2": ("r3", "r2")}
LEVELS = ["subject", "stratum", "selection_oos", "B", "comparison"]


def load_rows(rows_dir) -> pd.DataFrame:
    recs = []
    for f in sorted(Path(rows_dir).rglob("*.jsonl")):
        recs.extend(read_jsonl(f))                      # tolerates a torn last line
    if not recs:
        raise ValueError(f"no rows under {rows_dir}")
    df = pd.DataFrame(recs)
    bad = df[df.get("rule_version", pd.Series([None] * len(df))) != RULE_VERSION]
    if len(bad):
        raise ValueError(f"{len(bad)} rows lack rule_version={RULE_VERSION!r} (rows of other protocols, e.g. P3 / "
                         "drift-curve rules selected on 20140307, must not be merged)")
    return df


def paired_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (backbone, target, B, comparison): mean paired difference over (draw, seed)."""
    key = ["bid", "target", "B", "draw", "seed"]
    meta = ["subject", "stratum", "selection_oos", "gap_days", "trunk_gap_days", "tier", "apst_dev", "date"]
    wide = df.pivot_table(index=key, columns="method", values="r2", aggfunc="mean")
    # p_prior is seed-free (stored with seed 0): broadcast it to the other seed of the same draw
    if "p_prior" in wide:
        pp = wide["p_prior"].groupby(level=["bid", "target", "B", "draw"]).transform("max")
        wide["p_prior"] = pp
    md = df.drop_duplicates(["bid", "target"]).set_index(["bid", "target"])[meta]
    out = []
    for name, (a, b) in COMPARISONS.items():
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna().rename("diff").reset_index()
        g = d.groupby(["bid", "target", "B"]).agg(diff=("diff", "mean"), n=("diff", "size"),
                                                  draw_sd=("diff", "std")).reset_index()
        g["comparison"] = name
        out.append(g)
    cells = pd.concat(out, ignore_index=True)
    return cells.join(md, on=["bid", "target"])


def trunk_stratum(gap: float) -> str:
    return "days" if gap <= 7 else ("month" if 90 <= gap <= 130 else ("year" if gap >= 365 else "between"))


def boot_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def t_ci(x: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - h), float(x.mean() + h)


def session_cluster(cells: pd.DataFrame, extra_levels=()) -> pd.DataFrame:
    """Mean over backbones within a session, then mean / CI across sessions (see module docstring)."""
    levels = LEVELS + list(extra_levels)
    per_session = cells.groupby(levels + ["target"]).agg(diff=("diff", "mean"), n_backbones=("bid", "nunique"),
                                                         n_pairs=("bid", "size")).reset_index()
    rows = []
    for key, g in per_session.groupby(levels):
        x = g["diff"].to_numpy()
        lo, hi = boot_ci(x)
        tlo, thi = t_ci(x)
        rows.append({**dict(zip(levels, key)), "n_sessions": len(g), "n_backbone_target_pairs": int(g["n_pairs"].sum()),
                     "mean": float(x.mean()), "boot_lo": lo, "boot_hi": hi, "t_lo": tlo, "t_hi": thi,
                     "n_positive_sessions": int((x > 0).sum()), "multi_backbone_sessions": int((g["n_backbones"] > 1).sum())})
    return pd.DataFrame(rows)


def pooled_b4(cells: pd.DataFrame) -> pd.DataFrame:
    """B >= 4 pooled (mean over B per cell) before the session clustering; reported as B = 'B4-33'."""
    sub = cells[cells["B"] >= 4]
    m = sub.groupby(["bid", "target", "comparison"]).agg(diff=("diff", "mean")).reset_index()
    meta = cells.drop_duplicates(["bid", "target"])[["bid", "target", "subject", "stratum", "selection_oos",
                                                      "gap_days", "trunk_gap_days"]]
    m = m.merge(meta, on=["bid", "target"])
    m["B"] = -1                                       # -1 stands for "B4-33"
    return m


def backbone_spread(cells: pd.DataFrame) -> pd.DataFrame:
    """Per backbone mean across its targets; spread across backbones; within-session spread between backbones."""
    per_bb = cells.groupby(LEVELS + ["bid"]).agg(mean=("diff", "mean"), n_targets=("target", "nunique")).reset_index()
    rows = []
    for key, g in per_bb.groupby(LEVELS):
        # sessions evaluated by >= 2 backbones inside this group
        sub = cells
        for col, val in zip(LEVELS, key):
            sub = sub[sub[col] == val]
        by_session = sub.groupby(["target", "bid"])["diff"].mean().reset_index()
        spread = by_session.groupby("target")["diff"].agg(lambda v: float(v.max() - v.min()) if len(v) > 1 else np.nan)
        rows.append({**dict(zip(LEVELS, key)), "n_backbones": len(g), "backbone_mean_min": float(g["mean"].min()),
                     "backbone_mean_max": float(g["mean"].max()),
                     "backbone_mean_sd": float(g["mean"].std(ddof=1)) if len(g) > 1 else float("nan"),
                     "within_session_spread_max": float(np.nanmax(spread)) if spread.notna().any() else float("nan"),
                     "within_session_spread_mean": float(np.nanmean(spread)) if spread.notna().any() else float("nan")})
    return pd.DataFrame(rows)


def audit_flags(df: pd.DataFrame) -> dict:
    hashes = df.groupby("bid")["h1_trunk_hash"].nunique()
    return {"n_rows": int(len(df)), "n_backbones": int(df["bid"].nunique()),
            "n_backbone_target_pairs": int(df.drop_duplicates(["bid", "target"]).shape[0]),
            "n_distinct_target_sessions": int(df["target"].nunique()),
            "h2_zero_backward_all": bool((df["h2_backward_calls"] == 0).all()),
            "h4_query_disjoint_all": bool(df["h4_query_disjoint"].all()),
            "h1_trunk_unchanged_all": bool(df["h1_trunk_unchanged"].all()),
            "h1_one_hash_per_backbone": bool((hashes == 1).all()),
            "unit_check_passed_all": bool(df["unit_check_passed"].all()),
            "unit_check_failures": int((~df["unit_check_passed"]).sum()),
            "h5_within_32kb_all": bool(df["h5_within_32kb"].all()),
            "selection_scopes": sorted(df["selection"].unique().tolist()),
            "oos_backbones": sorted(df.loc[df["selection_oos"], "bid"].unique().tolist()),
            "is_backbones": sorted(df.loc[~df["selection_oos"], "bid"].unique().tolist())}


def analyze(rows_dir, out_dir) -> dict:
    df = load_rows(rows_dir)
    cells = paired_cells(df)
    pooled = pooled_b4(cells)
    cells_all = pd.concat([cells, pooled], ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "paired_cells.csv", index=False)
    sc = session_cluster(cells_all)
    sc.to_csv(out / "paired_session_cluster.csv", index=False)
    # same clustering, but stratified by the gap to the latest trunk-TRAINING session (pool[-1] is held out of
    # the new trunks, so e.g. sm_0627's "days" target is ~112 days after its trunk's last training session)
    cells_all["trunk_stratum"] = cells_all["trunk_gap_days"].map(trunk_stratum)
    session_cluster(cells_all, extra_levels=("trunk_stratum",)).to_csv(
        out / "paired_session_cluster_by_trunk_gap.csv", index=False)
    spread = backbone_spread(cells_all)
    spread.to_csv(out / "backbone_spread.csv", index=False)
    flags = audit_flags(df)
    (out / "flags.json").write_text(json.dumps(flags, indent=1))
    means = df.groupby(["subject", "stratum", "selection_oos", "B", "method"])["r2"].mean().reset_index()
    means.to_csv(out / "method_means_unclustered.csv", index=False)
    return {"flags": flags, "n_cluster_rows": len(sc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(analyze(a.rows, a.out), indent=1))


if __name__ == "__main__":
    main()
