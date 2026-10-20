#!/usr/bin/env python3
"""Tier aggregates for the experiment log (causal-first-B, 2026-10-20).

Gap is days from the latest trunk-training session to the target.
Tiers: <=30 d, 31-180 d, >180 d.

Protocol (a)/(b) rows (w_alpha_a, combo_a, w_alpha_b, combo_b) are not in
results/analysis/causal_main_20261019/per_target.csv. That CSV still has the
superseded pooled-alpha columns. This script reads
results/causal_main_20261019/rows.jsonl for those rows, checks overlapping
methods against per_target.csv, and checks 12-target means against the
reviewed README anchors.

Paired intervals are a target bootstrap (10_000 resamples, seed 0,
percentile 2.5/97.5) unless noted. n=2 intervals only restate the two
sessions. They are not a population CI and they do not include trunk
variance. M2 intervals are the pool-level t intervals already stored in
paired_pool.csv (df=2). No multiplicity correction.

Does not run experiments and does not write result files.
"""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results/causal_main_20261019/rows.jsonl"
PER_TARGET = ROOT / "results/analysis/causal_main_20261019/per_target.csv"
M2_TARGET = ROOT / "results/analysis/m2_causal_20261020/per_target.csv"
M2_PAIRED = ROOT / "results/analysis/m2_causal_20261020/paired_pool.csv"

# Reviewed README anchors: causal_main README, protocol (a)/(b), 12 drift targets.
README_DRIFT = {
    ("p_prior", 8): 0.104,
    ("p0", 8): 0.209,
    ("p0", 33): 0.215,
    ("ridge_cca", 8): 0.329,
    ("ridge_cca", 33): 0.409,
    ("w_alpha_a", 8): 0.348,
    ("w_alpha_a", 33): 0.437,
    ("s1_causal_and", 8): 0.276,
    ("combo_a", 8): 0.381,
    ("combo_a", 16): 0.406,
    ("combo_a", 33): 0.471,
    ("combo_b", 8): 0.336,
    ("combo_b", 33): 0.465,
    ("cca_s1", 33): 0.450,
    ("r2", 8): 0.346,
    ("r3", 33): 0.411,
    ("p0_ui_reg", 33): 0.450,
}

DRIFT_ORDER = [
    "20140626", "20140627", "20140929", "20141203", "20150511", "20150512",
    "20150610", "20150611", "20150612", "20150615", "20150616", "20150617",
]
# Second gap: selection/source day minus nothing. Drift selection is 20140307,
# one day after the month trunk's last training day (20140306).
DRIFT_SELECTION_LAG_DAYS = 1
SUBC = {
    "sc_0629": {
        "dates": ["20150629", "20150701", "20150706", "20150709", "20150713", "20150716"],
        # pool[-1] is 20150319; last trunk-training day is 20150313.
        "selection_minus_trunk": -6,
    },
    "sc_0708": {
        "dates": ["20151103", "20151104", "20151106", "20151109", "20151110", "20151112"],
        # pool[-1] is 20150707; last trunk-training day is 20150706.
        "selection_minus_trunk": -1,
    },
}
METHODS = [
    "p_prior", "p0", "ridge_cca", "w_alpha_a", "w_alpha_b", "s1_causal_and",
    "combo_a", "combo_b", "cca_s1", "r1_stream", "r2", "r3", "p0_ui_reg",
]
BS_FULL = [1, 2, 4, 8, 16, 33]


def load_rows() -> dict[tuple, float]:
    """Seed-mean query R^2 at latency 0. Key: stratum, backbone, date, method, B."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    gaps: dict[tuple, float] = {}
    with ROWS.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("latency") or 0) != 0:
                continue
            if row.get("r2") is None:
                continue
            key = (
                row["stratum"], row["backbone"], str(row["date"]),
                row["method"], int(row["B"]),
            )
            acc[key].append(float(row["r2"]))
            if row.get("gap_days") is not None:
                gaps[(row["stratum"], str(row["date"]))] = float(row["gap_days"])
    means = {key: float(np.mean(vals)) for key, vals in acc.items()}
    return means, gaps


def series(means, stratum, dates, method, budget, backbone) -> np.ndarray:
    out = []
    for date in dates:
        key = (stratum, backbone, date, method, budget)
        out.append(np.nan if key not in means else means[key])
    return np.asarray(out, dtype=float)


def bootstrap_diff(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n: int = 10_000):
    diff = left - right
    n_t = len(diff)
    draws = rng.integers(0, n_t, size=(n, n_t))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    pos = int(np.sum(diff > 1e-6))
    neg = int(np.sum(diff < -1e-6))
    return {
        "mean": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "pos": pos,
        "neg": neg,
        "n": n_t,
        "vmin": float(diff.min()),
        "vmax": float(diff.max()),
    }


def fmt_diff(stat: dict) -> str:
    return (
        f"{stat['mean']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}] "
        f"+{stat['pos']}/-{stat['neg']}/{stat['n']} "
        f"range {stat['vmin']:+.3f}..{stat['vmax']:+.3f}"
    )


def check_per_target(means) -> None:
    frame = pd.read_csv(PER_TARGET)
    frame = frame[frame.latency == 0]
    # per_target.csv has no protocol (a)/(b). Compare methods that both sides have.
    shared = ["p0", "ridge_cca", "s1_causal_and", "cca_s1", "p_prior", "r2", "r3"]
    worst = 0.0
    n_cmp = 0
    for row in frame.itertuples(index=False):
        if row.method not in shared:
            continue
        key = (row.stratum, row.backbone, str(int(row.date)), row.method, int(row.B))
        if key not in means:
            continue
        worst = max(worst, abs(float(row.r2) - means[key]))
        n_cmp += 1
    print(f"per_target.csv vs rows.jsonl on shared methods: n={n_cmp} max|d|={worst:.3e}")
    extra = sorted(set(frame.method) - set(shared))
    print("per_target.csv columns not in the shared check:", ", ".join(extra))
    if "w_alpha_a" not in set(frame.method):
        print("NOTE: per_target.csv has no w_alpha_a/combo_a; tier W_alpha/COMBO come from rows.jsonl.")
        print("NOTE: w_alpha_pooled / combo in that CSV are the superseded pooled-alpha rows.")


def check_readme(means) -> None:
    print("README 12-target anchors:")
    bad = False
    for (method, budget), expect in README_DRIFT.items():
        vals = series(means, "drift", DRIFT_ORDER, method, budget, "month")
        got = float(np.mean(vals))
        ok = abs(got - expect) < 0.0015
        bad = bad or not ok
        print(f"  {method:16} B={budget:<2} rows {got:.3f} readme {expect:.3f} {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise SystemExit("README anchor mismatch")


def retention(means) -> None:
    """(S1-P0)/(R2-P0) on the 12 drift targets. Not the random-draw 0.82 figure."""
    print("causal-first-B S1 retention of R2 gain (12 drift targets, AND gate):")
    for budget in (8, 33):
        p0 = series(means, "drift", DRIFT_ORDER, "p0", budget, "month")
        s1 = series(means, "drift", DRIFT_ORDER, "s1_causal_and", budget, "month")
        r2 = series(means, "drift", DRIFT_ORDER, "r2", budget, "month")
        num = float(np.mean(s1 - p0))
        den = float(np.mean(r2 - p0))
        print(f"  B={budget} S1 {s1.mean():.3f} R2 {r2.mean():.3f} P0 {p0.mean():.3f} "
              f"ratio {num / den:.3f}")


def contrast_rng(*parts) -> np.random.Generator:
    """One seed per contrast, so the interval does not depend on print order."""
    key = "|".join(str(part) for part in parts).encode()
    return np.random.default_rng(zlib.crc32(key))


def dump_block(means, title, stratum, dates, backbone, budgets) -> None:
    print(f"\n==== {title} n={len(dates)} backbone={backbone}")
    print("targets:", ", ".join(dates))
    for method in METHODS:
        cells = []
        for budget in budgets:
            vals = series(means, stratum, dates, method, budget, backbone)
            cells.append("NA" if np.any(np.isnan(vals)) else f"{vals.mean():.3f}")
        print(f"  {method:16} " + " ".join(cells))
    focus = [b for b in budgets if b in (8, 16, 33)]
    contrasts = [
        ("p0", "p_prior"),
        ("p0", "ridge_cca"),
        ("w_alpha_a", "p0"),
        ("w_alpha_a", "ridge_cca"),
        ("s1_causal_and", "p0"),
        ("combo_a", "p0"),
        ("combo_a", "ridge_cca"),
        ("combo_b", "ridge_cca"),
        ("cca_s1", "ridge_cca"),
        ("ridge_cca", "p0"),
        ("r2", "p0"),
    ]
    for budget in focus:
        print(f"  -- B={budget}")
        for left, right in contrasts:
            a = series(means, stratum, dates, left, budget, backbone)
            b = series(means, stratum, dates, right, budget, backbone)
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                print(f"    {left} - {right}: NA")
                continue
            stat = bootstrap_diff(
                a, b, contrast_rng(stratum, backbone, budget, left, right, len(dates)),
            )
            note = " n=2 session-only" if stat["n"] < 3 else ""
            print(f"    {left} - {right}: {fmt_diff(stat)}{note}")


def dump_m2() -> None:
    target = pd.read_csv(M2_TARGET)
    paired = pd.read_csv(M2_PAIRED)
    print("\n==== M2 causal-first-B, unweighted target mean (5 targets, <=30 d)")
    order = ["p_prior", "p0", "ridge_cca", "w_alpha", "s1", "combo", "cca_s1",
             "r1_lr16", "r2", "r3", "p0_ui_reg", "w_alpha_b", "combo_b"]
    for method in order:
        sub = target[target.method == method]
        cells = []
        for budget in BS_FULL:
            cells.append(f"{sub.loc[sub.B == budget, 'r2'].mean():.3f}")
        print(f"  {method:16} " + " ".join(cells))
    print("  pool t intervals from paired_pool.csv (half-width is pool_ci95):")
    want = paired[
        paired.B.isin([8, 16, 33])
        & paired.method.isin(["w_alpha", "combo", "p0", "ridge_cca", "s1", "r1_lr16"])
        & paired.baseline.isin(["p0", "p_prior", "ridge_cca"])
    ]
    for row in want.sort_values(["method", "baseline", "B"]).itertuples(index=False):
        print(
            f"    {row.method} - {row.baseline} B={row.B}: "
            f"{row.pool_mean:+.3f} +/- {row.pool_ci95:.3f} "
            f"+{int(row.n_pos_pools)}/{int(row.n_pools)}"
        )


ROUND2 = ROOT / "results/analysis/expansion_20261016/round2_summary"
# Reviewed README (round2_README.md, correction block): COMBO (a) - P0, B4-33.
ROUND2_COMBO_ANCHORS = {
    "<=30 d": (0.006, 0.001, 0.012, 7, 8),
    "31-180 d": (0.039, 0.027, 0.051, 16, 17),
    ">180 d": (0.207, 0.111, 0.315, 7, 7),
}


def _fmt_ci(row) -> str:
    return (
        f"{row['mean']:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] "
        f"({int(row.n_pos)}/{int(row.n_sessions)})"
    )


def dump_round2() -> None:
    """Print round-2 tier tables from the reviewed summary CSVs.

    Absolute R2 and differences versus P0 are checked against
    round2_README.md. Differences versus ridge-CCA are read from
    delta_vs_ridge_by_tier_lat0.csv (same aggregation, not reprinted as a
    README table).
    """
    vs_p0 = pd.read_csv(ROUND2 / "delta_vs_p0_by_tier_lat0.csv")
    vs_ridge = pd.read_csv(ROUND2 / "delta_vs_ridge_by_tier_lat0.csv")
    absolute = pd.read_csv(ROUND2 / "abs_by_tier_lat0.csv")
    membership = pd.read_csv(ROUND2 / "target_membership.csv")
    selection = pd.read_csv(ROUND2 / "selection_composition.csv")
    print("\n==== round 2 (reviewed summary CSVs, latency 0)")
    for tier, (mean, lo, hi, npos, n) in ROUND2_COMBO_ANCHORS.items():
        row = vs_p0[(vs_p0.tier == tier) & (vs_p0.method == "combo_a") & (vs_p0.B == "B4-33")]
        if len(row) != 1:
            raise SystemExit(f"missing COMBO anchor for {tier}")
        got = row.iloc[0]
        ok = (
            abs(got["mean"] - mean) < 0.0015
            and abs(got.ci_lo - lo) < 0.0015
            and abs(got.ci_hi - hi) < 0.0015
            and int(got.n_pos) == npos
            and int(got.n_sessions) == n
        )
        print(f"  COMBO(a)-P0 B4-33 {tier}: {_fmt_ci(got)} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("round-2 COMBO anchor mismatch")
    sel8 = selection[selection.B == 8]
    joined = membership.merge(sel8, on=["bid", "date"])
    for tier, group in joined.groupby("tier"):
        sessions = sorted(set(group.date.astype(str)))
        backs = sorted(set(group.bid.astype(str)))
        print(
            f"  {tier}: pairs {len(group)} sessions {len(sessions)} "
            f"alpha-bin n {int(group.n_alpha_vote_a.min())}-{int(group.n_alpha_vote_a.max())} "
            f"S1 n {int(group.n_s1_a.min())}-{int(group.n_s1_a.max())} "
            f"(b) alpha n {sorted(set(group.n_alpha_vote_b.astype(int)))}"
        )
        print(f"    backbones: {', '.join(backs)}")
        print(f"    sessions: {', '.join(sessions)}")
    methods = ["p0", "w_alpha_a", "s1_causal", "combo_a", "cca_s1", "ridge_cca"]
    for tier in ["<=30 d", "31-180 d", ">180 d"]:
        print(f"  -- {tier}")
        for method in ["p_prior", "p0", "ridge_cca", "w_alpha_a", "s1_causal", "combo_a", "cca_s1", "r2"]:
            cells = []
            for budget in ["1", "4", "8", "16", "33"]:
                row = absolute[(absolute.tier == tier) & (absolute.method == method) & (absolute.B == budget)]
                cells.append("NA" if row.empty else f"{row.iloc[0]['mean']:.3f}")
            print(f"    abs {method:12} " + " ".join(cells))
        for label, frame in (("vs P0", vs_p0), ("vs ridge", vs_ridge)):
            for method in methods:
                if label == "vs P0" and method == "p0":
                    continue
                if label == "vs ridge" and method == "ridge_cca":
                    continue
                bits = []
                for budget in ["8", "33", "B4-33"]:
                    row = frame[(frame.tier == tier) & (frame.method == method) & (frame.B == budget)]
                    bits.append("NA" if row.empty else _fmt_ci(row.iloc[0]))
                print(f"    {label:8} {method:12} " + " | ".join(bits))


def main() -> None:
    dump_round2()
    means, gaps = load_rows()
    check_per_target(means)
    check_readme(means)
    retention(means)
    print("\nDrift trunk gaps (selection gap = trunk gap - 1):")
    for date in DRIFT_ORDER:
        trunk = int(gaps[("drift", date)])
        tier = "31-180" if trunk <= 180 else ">180"
        print(f"  {date} trunk {trunk} d  selection {trunk - DRIFT_SELECTION_LAG_DAYS} d  {tier}")
    mid = [d for d in DRIFT_ORDER if gaps[("drift", d)] <= 180]
    long = [d for d in DRIFT_ORDER if gaps[("drift", d)] > 180]
    dump_block(means, "sub-M days trunk gap 1-2 d", "days",
               ["20150616", "20150617"], "days", BS_FULL)
    print("  days selection sessions are 2014 OOS, 194-497 d BEFORE the trunk (inverted); not a <=30 d source.")
    dump_block(means, "sub-M trunk gap 31-180 d", "drift", mid, "month", BS_FULL)
    dump_block(means, "sub-M trunk gap >180 d", "drift", long, "month", BS_FULL)
    for bid, spec in SUBC.items():
        dates = spec["dates"]
        print(f"\n{bid} selection gap = trunk gap {spec['selection_minus_trunk']:+d} d")
        for date in dates:
            trunk = int(gaps[("subc_month", date)])
            sel = trunk + spec["selection_minus_trunk"]
            print(f"  {date} trunk {trunk} d  selection {sel} d")
        dump_block(
            means, f"{bid} month", "subc_month", dates, bid, [1, 4, 8, 16, 33],
        )
    dump_m2()


if __name__ == "__main__":
    main()
