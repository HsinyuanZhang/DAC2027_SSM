#!/usr/bin/env python
"""Aggregate P0-Solve rows: per-stratum means, target-clustered paired CIs, projection diagnostic.

Reference rows that are protocol-identical to an existing run are read from those files
and not recomputed:

  * R3 days: P3 days-IS eval (the days headline rule)
  * R3 year: P3 year-OOS eval
  * R3 drift / month: drift-curve rows (month oosc rule, same draws)
  * P0+ui_reg drift / month: drift-curve rows (like-for-like)
  * P0+ui_reg year: P3 uiref year-OOS
  * P0+ui_reg days: P2 ``p0_ui_reg`` (NOT like-for-like; days was never re-selected from P0)

Month is not a separate GPU pass: it is the drift row for 20140626 (same trunk, same rules).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/p0solve_20261019"
OUT = ROOT / "results/analysis/p0solve_20261019"
P3 = ROOT / "results/population_calib_p3_20261016"
P3_FIX = P3 / "fix"
P2 = ROOT / "results/population_calib_p2_20261015"
DRIFT = ROOT / "results/drift_curve_20261016"
B_LIST = (1, 2, 4, 8, 16, 33)
HEADLINE_B = (1, 8, 33)
KINDS = ("full", "diag", "lr8", "lr16", "lr32")
OUR = ("p_prior", "p0", "w_alpha", "p0solve_p8", "p0solve_p16",
       "prior_solve_p8", "prior_solve_p16", "ridge_cca")
REFS = ("r3", "p0_ui_reg")
SEED_FREE = {"p_prior", "prior_solve_p8", "prior_solve_p16", "ridge_cca"}
# Order used by the headline table. p0solve_p8/p16 are both reported; neither is picked on target.
HEADLINE = ("p0", "w_alpha", "p0solve_p8", "p0solve_p16", "ridge_cca", "r3", "p0_ui_reg")
DIFF_BASES = ("p0", "w_alpha")
RNG = np.random.default_rng(0)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def chosen_kind(selection: dict, prefix: str, B: int) -> str:
    cand = {k: selection[f"{prefix}_{k}"][str(B)]["source_r2"] for k in KINDS}
    return max(cand, key=cand.get)


def load_r3(path: Path, selection: dict, stratum: str) -> pd.DataFrame:
    kind = {B: chosen_kind(selection, "r3", B) for B in B_LIST}
    rows = []
    for r in _rows(path):
        B = int(r["B"])
        if r.get("method") != f"r3_{kind[B]}":
            continue
        rows.append({"stratum": stratum, "target": r["target"], "B": B, "draw": int(r["draw"]),
                     "seed": int(r["seed"]), "method": "r3", "r2": float(r["r2"]),
                     "reused": True, "cfg": {"kind": kind[B]}})
    return pd.DataFrame(rows)


def load_ui_p2_days() -> pd.DataFrame:
    rows = []
    for r in _rows(P2 / "uireg/eval_days.jsonl"):
        if r.get("variant") != "p0_ui_reg":
            continue
        rows.append({"stratum": "days", "target": r["target"], "B": int(r["B"]), "draw": int(r["draw"]),
                     "seed": int(r["seed"]), "method": "p0_ui_reg", "r2": float(r["r2"]),
                     "reused": True, "like_for_like": False})
    return pd.DataFrame(rows)


def load_ui_year() -> pd.DataFrame:
    rows = []
    for r in _rows(P3_FIX / "uiref/eval_year_oos.jsonl"):
        rows.append({"stratum": "year", "target": r["target"], "B": int(r["B"]), "draw": int(r["draw"]),
                     "seed": int(r["seed"]), "method": "p0_ui_reg", "r2": float(r["r2"]),
                     "reused": True, "like_for_like": True})
    return pd.DataFrame(rows)


def load_drift_refs() -> pd.DataFrame:
    rows = []
    for f in sorted(DRIFT.glob("rows_*.jsonl")):
        for r in _rows(f):
            if r["method"] not in ("r3", "p0_ui_reg", "p0", "p_prior"):
                continue
            rows.append({"target": r["target"], "B": int(r["B"]), "draw": int(r["draw"]),
                         "seed": int(r["seed"]), "method": r["method"], "r2": float(r["r2"])})
    return pd.DataFrame(rows)


def load_ours() -> pd.DataFrame:
    frames = []
    for stratum in ("days", "year", "drift"):
        rows = _rows(RAW / "eval" / f"rows_{stratum}.jsonl")
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit(f"no eval rows under {RAW}/eval")
    df = pd.concat(frames, ignore_index=True)
    df["B"] = df.B.astype(int)
    df["draw"] = df.draw.astype(int)
    df["seed"] = df.seed.astype(int)
    # Month core is the drift evaluation of 20140626 (identical trunk and month rules).
    month = df[(df.stratum == "drift") & (df.date == "20140626")].copy()
    month["stratum"] = "month"
    return pd.concat([df, month], ignore_index=True)


def attach_refs(ours: pd.DataFrame) -> pd.DataFrame:
    sel_days = json.loads((P3_FIX / "select/selection_days_is.json").read_text())
    sel_year = json.loads((P3_FIX / "select/selection_year_oos.json").read_text())
    refs = [
        load_r3(P3 / "eval/eval_days_is.jsonl", sel_days, "days"),
        load_r3(P3 / "eval/eval_year_oos.jsonl", sel_year, "year"),
        load_ui_p2_days(),
        load_ui_year(),
    ]
    drift = load_drift_refs()
    # r3 and ui for drift, and the same rows relabeled as the month core target.
    for stratum, date in (("drift", None), ("month", "20140626")):
        sub = drift if date is None else drift[drift.target.str.endswith(date)]
        for method in ("r3", "p0_ui_reg"):
            block = sub[sub.method == method].copy()
            block["stratum"] = stratum
            block["reused"] = True
            block["like_for_like"] = True
            refs.append(block[["stratum", "target", "B", "draw", "seed", "method", "r2", "reused", "like_for_like"]])
    ref = pd.concat(refs, ignore_index=True)
    keep = ours[ours.method.isin(OUR)][["stratum", "target", "date", "B", "draw", "seed", "method", "r2",
                                        "n_units", "diagnostic_only"]].copy()
    keep["reused"] = False
    ref["date"] = ref.target.str[-8:]
    ref["diagnostic_only"] = False
    # n_units filled after the merge
    both = pd.concat([keep, ref], ignore_index=True, sort=False)
    return both


def broadcast_seed_free(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-free rows are stored at seed 0. Copy them onto seed 1 so pairs line up."""
    seeds = sorted(df.seed.unique())
    if len(seeds) < 2:
        return df
    free = df[df.method.isin(SEED_FREE) & (df.seed == seeds[0])].copy()
    copies = []
    have = set(zip(df.stratum, df.target, df.B, df.draw, df.seed, df.method))
    for seed in seeds[1:]:
        block = free.copy()
        block["seed"] = seed
        mask = [ (s, t, b, d, seed, m) not in have
                 for s, t, b, d, m in zip(block.stratum, block.target, block.B, block.draw, block.method)]
        copies.append(block[mask])
    if copies:
        df = pd.concat([df, *copies], ignore_index=True)
    return df


def p0_crosscheck(ours: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    a = ours[(ours.stratum == "drift") & (ours.method == "p0")][["target", "B", "draw", "seed", "r2"]]
    b = drift[drift.method == "p0"][["target", "B", "draw", "seed", "r2"]].rename(columns={"r2": "r2_stored"})
    m = a.merge(b, on=["target", "B", "draw", "seed"], how="inner")
    if m.empty:
        return m
    m["abs_diff"] = (m.r2 - m.r2_stored).abs()
    return m


def _target_means(sub: pd.DataFrame, col: str) -> np.ndarray:
    return sub.groupby("target")[col].mean().to_numpy()


def cluster_ci(values: np.ndarray, n_boot: int = 4000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    ix = RNG.integers(0, n, size=(n_boot, n))
    means = values[ix].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def draw_cluster_ci(sub: pd.DataFrame, col: str, n_boot: int = 2000):
    """Resample draws, keeping every target and both seeds of a draw together."""
    if sub.empty or sub[col].isna().all():
        return np.nan, np.nan
    keys = sub.groupby(["target", "draw"])[col].mean()
    # per target, a vector over draws; resample draws independently inside each target, then average
    mats = []
    for _t, g in sub.groupby("target"):
        mat = g.pivot_table(index="draw", columns="seed", values=col)
        mats.append(mat.to_numpy())
    if not mats:
        return np.nan, np.nan
    # different targets can have the same draw index set
    n_draw = mats[0].shape[0]
    acc = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for mat in mats:
            ix = RNG.integers(0, mat.shape[0], size=mat.shape[0])
            total += np.nanmean(mat[ix])
            count += 1
        acc.append(total / count)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def cells(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    methods = list(OUR) + list(REFS)
    for (stratum, B), sub in df.groupby(["stratum", "B"]):
        n_targets = sub.target.nunique()
        for method in methods:
            block = sub[sub.method == method]
            if block.empty:
                continue
            v = block.r2.to_numpy()
            tm = _target_means(block, "r2")
            lo, hi = cluster_ci(tm)
            dlo, dhi = draw_cluster_ci(block, "r2")
            seed = block.groupby("seed").r2.mean()
            out.append({"stratum": stratum, "B": int(B), "method": method,
                        "n_rows": int(len(block)), "n_targets": int(n_targets),
                        "n_finite": int(np.isfinite(v).sum()),
                        "r2_mean": float(np.nanmean(v)),
                        "ci_lo": lo, "ci_hi": hi,
                        "ci_kind": "target" if n_targets >= 2 else "draw_only",
                        "draw_ci_lo": dlo, "draw_ci_hi": dhi,
                        "seed0": float(seed.get(0, np.nan)), "seed1": float(seed.get(1, np.nan)),
                        "seed_spread": float(abs(seed.get(0, np.nan) - seed.get(1, np.nan)))
                        if 0 in seed.index and 1 in seed.index else np.nan})
    return pd.DataFrame(out)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (stratum, B), sub in df.groupby(["stratum", "B"]):
        wide = sub.pivot_table(index=["target", "draw", "seed"], columns="method", values="r2")
        n_targets = sub.target.nunique()
        for base in DIFF_BASES:
            if base not in wide.columns:
                continue
            # W_alpha vs P0 is required; W_alpha vs itself is not.
            methods = []
            for method in ("w_alpha",) + tuple(OUR) + REFS:
                if method != base and method != "p_prior" and method not in methods:
                    methods.append(method)
            for method in methods:
                if method not in wide.columns:
                    continue
                diff = (wide[method] - wide[base]).rename("diff").reset_index()
                tm = diff.groupby("target")["diff"].mean().to_numpy()
                lo, hi = cluster_ci(tm)
                dlo, dhi = draw_cluster_ci(diff, "diff")
                out.append({"stratum": stratum, "B": int(B), "method": method, "baseline": base,
                            "diff": float(np.nanmean(diff["diff"])),
                            "ci_lo": lo, "ci_hi": hi,
                            "ci_kind": "target" if n_targets >= 2 else "draw_only",
                            "draw_ci_lo": dlo, "draw_ci_hi": dhi,
                            "n_targets": int(n_targets), "n_pairs": int(np.isfinite(diff["diff"]).sum())})
    return pd.DataFrame(out)


def projection(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return diag
    diag = diag.copy()
    diag["B"] = diag.B.astype(int)
    # Month core from the drift diagnostic of 20140626, same as the eval rows.
    extra = diag[(diag.stratum == "drift") & (diag.date == "20140626")].copy()
    extra["stratum"] = "month"
    diag = pd.concat([diag, extra], ignore_index=True)
    out = []
    for (stratum, B, p), sub in diag.groupby(["stratum", "B", "p"]):
        g_ui = sub.gain_ui.to_numpy()
        g_d = sub.gain_diag.to_numpy()
        # Ratio of means is the stable summary. Per-episode ratios blow up when the ui gap is ~0.
        denom = float(np.nanmean(g_ui))
        numer = float(np.nanmean(g_d))
        tm_num = sub.groupby("target").gain_diag.mean().to_numpy()
        tm_den = sub.groupby("target").gain_ui.mean().to_numpy()
        # target-level ratios, only where the target's ui gap is not tiny
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = tm_num / tm_den
        ratios = ratios[np.isfinite(ratios) & (np.abs(tm_den) > 0.01)]
        lo, hi = cluster_ci(sub.groupby("target").gain_diag.mean().to_numpy())
        out.append({"stratum": stratum, "B": int(B), "p": int(p),
                    "n_rows": int(len(sub)), "n_targets": int(sub.target.nunique()),
                    "gain_ui": denom, "gain_diag": numer,
                    "retained_fraction": float(numer / denom) if abs(denom) > 1e-4 else np.nan,
                    "target_median_retained": float(np.median(ratios)) if len(ratios) else np.nan,
                    "frobenius_retained": float(sub.frobenius_retained.mean()),
                    "like_for_like": bool(sub.like_for_like.iloc[0]),
                    "gain_diag_ci_lo": lo, "gain_diag_ci_hi": hi,
                    "seed_spread_diag": float(abs(sub[sub.seed == 0].gain_diag.mean()
                                                 - sub[sub.seed == 1].gain_diag.mean()))
                    if set(sub.seed) >= {0, 1} else np.nan})
    return pd.DataFrame(out)


def selection_table() -> pd.DataFrame:
    rows = []
    for stratum in ("days", "month", "year"):
        path = RAW / "select" / f"selection_{stratum}.json"
        if not path.exists():
            continue
        sel = json.loads(path.read_text())
        for method, per_b in sel.items():
            if not isinstance(per_b, dict) or method in ("p0",):
                if method == "p0":
                    for B, info in per_b.items():
                        rows.append({"stratum": stratum, "method": "p0", "B": int(B),
                                     "source_r2": info.get("source_r2"), "cfg": ""})
                continue
            if method in ("stratum", "set", "sessions", "n_rows"):
                continue
            for B, info in per_b.items():
                if not isinstance(info, dict) or "cfg" not in info:
                    continue
                rows.append({"stratum": stratum, "method": method, "B": int(B),
                             "source_r2": info.get("source_r2"), "n_episodes": info.get("n_episodes"),
                             "cfg": json.dumps(info["cfg"], sort_keys=True)})
    return pd.DataFrame(rows)


def fmt(mean, lo, hi) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(lo):
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours = load_ours()
    drift = load_drift_refs()
    check = p0_crosscheck(ours, drift)
    if not check.empty:
        check.to_csv(OUT / "p0_vs_drift.csv", index=False)
        (OUT / "p0_vs_drift_summary.json").write_text(json.dumps({
            "n": int(len(check)), "max_abs": float(check.abs_diff.max()),
            "mean_abs": float(check.abs_diff.mean())}, indent=1))
    df = broadcast_seed_free(attach_refs(ours))
    df.to_csv(OUT / "rows_long.csv.gz", index=False)
    cell = cells(df)
    cell.to_csv(OUT / "cells.csv", index=False)
    diff = paired(df)
    diff.to_csv(OUT / "paired.csv", index=False)
    diag_rows = []
    for stratum in ("days", "year", "drift"):
        diag_rows += _rows(RAW / "diag" / f"rows_{stratum}.jsonl")
    proj = projection(pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame())
    if not proj.empty:
        proj.to_csv(OUT / "projection.csv", index=False)
    sel = selection_table()
    if not sel.empty:
        sel.to_csv(OUT / "selection.csv", index=False)
    # Compact headline: one row per stratum x B x method, B in {1, 8, 33} plus the full grid in cells.csv.
    head = cell[cell.B.isin(HEADLINE_B) & cell.method.isin(HEADLINE)].copy()
    head.to_csv(OUT / "headline.csv", index=False)
    lines = ["# headline (mean [target-cluster 95% CI]; single-target strata have no target CI)", ""]
    for stratum in ("days", "month", "year", "drift"):
        lines.append(f"## {stratum}")
        sub = head[head.stratum == stratum]
        if sub.empty:
            lines.append("(no rows)")
            continue
        for B in HEADLINE_B:
            bits = []
            for method in HEADLINE:
                row = sub[(sub.B == B) & (sub.method == method)]
                if row.empty:
                    continue
                r = row.iloc[0]
                bits.append(f"{method} {fmt(r.r2_mean, r.ci_lo, r.ci_hi)}")
            lines.append(f"B={B}: " + " | ".join(bits))
        lines.append("")
    if not proj.empty:
        lines.append("## projection retained fraction (mean diag gain / mean ui gain)")
        for _, r in proj.sort_values(["stratum", "B", "p"]).iterrows():
            if int(r.B) not in HEADLINE_B:
                continue
            lines.append(f"{r.stratum} B={int(r.B)} p={int(r.p)}: {r.retained_fraction:.3f} "
                         f"(ui gain {r.gain_ui:+.3f}, diag gain {r.gain_diag:+.3f}, "
                         f"fro {r.frobenius_retained:.3f}, like_for_like={r.like_for_like})")
    text = "\n".join(lines) + "\n"
    (OUT / "headline.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
