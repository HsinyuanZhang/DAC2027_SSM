#!/usr/bin/env python
"""Aggregate streaming-F7 rows with the stored P0 / R2 / R3 rows.

Paired differences are averaged inside each target session (draws and seeds), then a
95% t interval is taken across sessions.  A single-target stratum uses a draw-cluster
bootstrap instead (both seeds of a draw stay together).  Retained fraction is
(method - P0) / (R2 - P0) on those session means; it is left blank when |R2 - P0| < 0.01.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/stream_f7_20261019"
OUT = ROOT / "results/analysis/stream_f7_20261019"
DRIFT = ROOT / "results/drift_curve_20261016"
P3 = ROOT / "results/population_calib_p3_20261016"
P3_FIX = P3 / "fix"
EXP = ROOT / "results/expansion_round1/x/expansion_out"
B_HEAD = (1, 8, 33)
STREAM = ("s1_r2eta", "s1", "s1_r1s", "tr4", "tr8",
          "s1_joint", "s1c", "s1c_ridge", "tr4_trunc", "tr8_trunc", "tr4c", "tr8c",
          "s1c_rebase", "s1c_staged")
BASE = ("p0", "r2", "r3")
LABEL = {"p0": "P0", "r2": "R2", "r3": "R3",
         "s1_r2eta": "S1 noncausal (R2 eta)", "s1": "S1 noncausal (eta only)",
         "s1_r1s": "S1 noncausal + readout (old name R1s)",
         "tr4": "trace-M4 infinite (old)", "tr8": "trace-M8 infinite (old)",
         "s1_joint": "S1 noncausal (joint eta, rho)",
         "s1c": "S1-causal prior-anchored (superseded)",
         "s1c_ridge": "S1-causal prior-anchored + ridge (superseded)",
         "tr4_trunc": "trace-M4 truncated noncausal",
         "tr8_trunc": "trace-M8 truncated noncausal",
         "tr4c": "trace-M4 prior-anchored truncated (superseded)",
         "tr8c": "trace-M8 prior-anchored truncated (superseded)",
         "s1c_rebase": "S1-causal re-based", "s1c_staged": "S1-causal staged"}


def _jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _best_kind(sel, prefix, B):
    kinds = [k.split(f"{prefix}_", 1)[1] for k in sel if k.startswith(prefix + "_") and str(B) in sel[k]]
    return max(kinds, key=lambda k: sel[f"{prefix}_{k}"][str(B)]["source_r2"])


def load_stream() -> pd.DataFrame:
    rows = []
    names = ["rows_drift.jsonl", "rows_year.jsonl", "rows_sc_0629.jsonl", "rows_sc_0708.jsonl",
             "rows_drift_fix.jsonl", "rows_year_fix.jsonl", "rows_sc_0629_fix.jsonl", "rows_sc_0708_fix.jsonl",
             "rows_drift_rebased.jsonl", "rows_year_rebased.jsonl",
             "rows_sc_0629_rebased.jsonl", "rows_sc_0708_rebased.jsonl"]
    for name in names:
        rows += _jl(RAW / name)
    if not rows:
        raise SystemExit(f"no streaming rows in {RAW}")
    df = pd.DataFrame(rows)
    df["r2"] = df.r2.astype(float)
    if not np.isfinite(df.r2.to_numpy()).all():
        raise SystemExit("non-finite streaming r2; refusing to average or drop it")
    keys = ["stratum", "bid", "target", "B", "draw", "seed", "method"]
    dup = df.duplicated(keys)
    if dup.any():
        raise SystemExit(f"duplicate streaming rows: {int(dup.sum())}")
    return df


def load_stored() -> pd.DataFrame:
    rows = []
    # Drift curve: month causal rules, 12 targets.  Methods p0/r2/r3 are the protocol-identical rows.
    for f in sorted(DRIFT.glob("rows_*.jsonl")):
        for r in _jl(f):
            if r["method"] not in BASE:
                continue
            rows.append({"stratum": "subm_drift", "bid": "month", "target": r["target"], "date": r["date"],
                         "B": int(r["B"]), "draw": int(r["draw"]), "seed": int(r["seed"]),
                         "method": r["method"], "r2": float(r["r2"])})
    # Year core, year OOS rules.  R3 kind is the source-selected one (same rule as analyze_pc3.best_kind).
    sel_path = P3_FIX / "select/selection_year_oos.json"
    if not sel_path.exists():
        sel_path = P3 / "select/selection_year_oos.json"
        print(f"[analyze] year selection fallback {sel_path}", flush=True)
    sel = json.loads(sel_path.read_text())
    year_path = P3_FIX / "eval/eval_year_oos.jsonl"
    if not year_path.exists():
        year_path = P3 / "eval/eval_year_oos.jsonl"
        print(f"[analyze] year eval is not in the fix dir; using {year_path}", flush=True)
    wanted_r3 = {B: f"r3_{_best_kind(sel, 'r3', B)}" for B in (1, 2, 4, 8, 16, 33)}
    for r in _jl(year_path):
        m = r["method"]
        if m == "p0":
            method = "p0"
        elif m == "r2":
            method = "r2"
        elif m == wanted_r3[int(r["B"])]:
            method = "r3"
        else:
            continue
        rows.append({"stratum": "subm_year", "bid": "year", "target": r["target"], "date": r["target"][-8:],
                     "B": int(r["B"]), "draw": int(r["draw"]), "seed": int(r["seed"]),
                     "method": method, "r2": float(r["r2"])})
    for bid in ("sc_0629", "sc_0708"):
        folder = EXP / "rows" / bid
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.jsonl")):
            for r in _jl(f):
                if r.get("stratum") != "month" or r["method"] not in BASE:
                    continue
                rows.append({"stratum": "subc_month", "bid": bid, "target": r["target"], "date": r["date"],
                             "B": int(r["B"]), "draw": int(r["draw"]), "seed": int(r["seed"]),
                             "method": r["method"], "r2": float(r["r2"])})
    return pd.DataFrame(rows)


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    idx = ["stratum", "bid", "target", "B", "draw", "seed"]
    if df.duplicated(idx + ["method"]).any():
        raise SystemExit("duplicate (cell, method) rows")
    if not np.isfinite(df.r2.to_numpy()).all():
        raise SystemExit("non-finite r2")
    pv = df.pivot(index=idx, columns="method", values="r2").reset_index()
    return pv


def _t_ci(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def _draw_boot(mat: np.ndarray, rng, n=10000):
    """mat [draws, seeds] -> mean and percentile CI, resampling draws."""
    d = mat.shape[0]
    if d == 0:
        return float("nan"), float("nan"), float("nan")
    ix = rng.integers(0, d, size=(n, d))
    means = mat[ix].mean(axis=(1, 2))
    return float(mat.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cells_and_paired(pv: pd.DataFrame):
    rng = np.random.default_rng(0)
    cell_rows, pair_rows, retain_rows = [], [], []
    methods = [m for m in BASE + STREAM if m in pv.columns]
    for (stratum, B), sub in pv.groupby(["stratum", "B"]):
        targets = sorted(sub.target.unique())
        n_t = len(targets)
        # per target, per method: mean over draws x seeds
        tmean = {}
        for m in methods:
            if m not in sub.columns or sub[m].isna().all():
                continue
            means = []
            for t, g in sub.groupby("target"):
                mat = g.pivot(index="draw", columns="seed", values=m)
                if mat.isna().any().any():
                    raise SystemExit(f"missing {m} cells for {t} B={B} ({stratum})")
                means.append((t, float(mat.values.mean()), mat.values))
            tmean[m] = means
            if n_t == 1 and means:
                mean, lo, hi = _draw_boot(means[0][2], rng)
                how = "draw_bootstrap"
            else:
                mean, lo, hi = _t_ci(np.array([v for _, v, _ in means]))
                how = "session_t"
            cell_rows.append({"stratum": stratum, "B": int(B), "method": m, "n_sessions": len(means),
                              "r2": mean, "ci_lo": lo, "ci_hi": hi, "ci": how})
        if "p0" not in tmean or "r2" not in tmean:
            continue
        p0 = {t: v for t, v, _ in tmean["p0"]}
        r2m = {t: v for t, v, _ in tmean["r2"]}
        common = sorted(set(p0) & set(r2m))
        for m, pairs in tmean.items():
            if m == "p0":
                continue
            mm = {t: v for t, v, _ in pairs}
            both = [t for t in common if t in mm]
            if len(both) < 1:
                continue
            d_p0 = np.array([mm[t] - p0[t] for t in both])
            d_r2 = np.array([mm[t] - r2m[t] for t in both])
            if n_t == 1:
                # rebuild mats for the diff
                g = sub[sub.target == both[0]]
                def _mat(col, g=g):
                    return g.pivot(index="draw", columns="seed", values=col).values
                base_p, lo_p, hi_p = _draw_boot(_mat(m) - _mat("p0"), rng)
                base_r, lo_r, hi_r = _draw_boot(_mat(m) - _mat("r2"), rng)
            else:
                base_p, lo_p, hi_p = _t_ci(d_p0)
                base_r, lo_r, hi_r = _t_ci(d_r2)
            pair_rows.append({"stratum": stratum, "B": int(B), "method": m, "contrast": "vs_p0",
                              "n_sessions": len(both), "diff": base_p, "ci_lo": lo_p, "ci_hi": hi_p,
                              "n_pos": int((d_p0 > 0).sum())})
            pair_rows.append({"stratum": stratum, "B": int(B), "method": m, "contrast": "vs_r2",
                              "n_sessions": len(both), "diff": base_r, "ci_lo": lo_r, "ci_hi": hi_r,
                              "n_pos": int((d_r2 > 0).sum())})
            gain_r = np.array([r2m[t] - p0[t] for t in both])
            gain_m = d_p0
            denom = float(gain_r.mean())
            frac = float(gain_m.mean() / denom) if abs(denom) > 1e-8 else float("nan")
            if n_t == 1:
                g = sub[sub.target == both[0]]
                mp = g.pivot(index="draw", columns="seed", values=m).values
                pp = g.pivot(index="draw", columns="seed", values="p0").values
                rp = g.pivot(index="draw", columns="seed", values="r2").values
                d = mp.shape[0]
                ix = rng.integers(0, d, size=(10000, d))
                gm = (mp - pp)[ix].mean(axis=(1, 2))
                gr = (rp - pp)[ix].mean(axis=(1, 2))
            else:
                n = len(both)
                ix = rng.integers(0, n, size=(10000, n))
                gm = gain_m[ix].mean(1)
                gr = gain_r[ix].mean(1)
            ok = np.abs(gr) > 1e-8
            if ok.any():
                frs = gm[ok] / gr[ok]
                flo, fhi = float(np.percentile(frs, 2.5)), float(np.percentile(frs, 97.5))
            else:
                flo = fhi = float("nan")
            retain_rows.append({"stratum": stratum, "B": int(B), "method": m, "n_sessions": len(both),
                                "gain": float(gain_m.mean()), "r2_gain": denom,
                                "fraction": frac, "frac_lo": flo, "frac_hi": fhi,
                                "unstable_denom": bool(abs(denom) < 0.02)})
    return pd.DataFrame(cell_rows), pd.DataFrame(pair_rows), pd.DataFrame(retain_rows)


def add_month_core(pv: pd.DataFrame) -> pd.DataFrame:
    """20140626 under the drift (month-rule) rows, tagged subm_month so the headline can show it."""
    sub = pv[(pv.stratum == "subm_drift") & (pv.target.str.endswith("20140626"))].copy()
    if sub.empty:
        return pv
    sub["stratum"] = "subm_month"
    return pd.concat([pv, sub], ignore_index=True)


def _source_jbar(tag: str, n_lags: int):
    """Averaged source Jacobian, lags 0..n_lags-1.  Not the fitted exponential."""
    import torch
    if tag == "subm":
        path = P3_FIX / "jacobian_month.pt"
    else:
        path = EXP / "jacobian" / f"{tag}.pt"
    blob = torch.load(path, weights_only=False)
    tot, n = 0, 0
    for s in blob["sessions"].values():
        tot = tot + s["sum"]
        n += s["n"]
    return (tot / n)[:n_lags].double()


def fit_table() -> pd.DataFrame:
    rows = []
    import torch
    from ssm_decode.stream_f7 import kernel_tail_metrics, probe_data_term_cosine
    for tag in ("subm", "sc_0629", "sc_0708"):
        path = RAW / f"fit_{tag}.pt"
        if not path.exists():
            continue
        blob = torch.load(path, weights_only=False)
        for L, modes in blob["modes"].items():
            for m, block in modes.items():
                rho = torch.tensor(block["rho"], dtype=torch.float64)
                c = block["C"].double()
                n_lags = int(block["n_lags"])
                jbar = _source_jbar(tag, n_lags)
                tail = kernel_tail_metrics(jbar, rho, c)
                cos = probe_data_term_cosine(jbar, rho, c, seed=0)
                rows.append({"tag": tag, "L": int(L), "M": int(m), "rel_frobenius": block["rel_frobenius"],
                             "rho_abs_max": block["rho_abs_max"],
                             "rho": " ".join(f"{v:.4f}" for v in block["rho"]),
                             "head_frobenius": tail["head_frobenius"],
                             "tail_frobenius": tail["tail_frobenius"],
                             "dc_gain_head": tail["dc_gain_head"],
                             "dc_gain_true": tail["dc_gain_true"],
                             "dc_gain_infinite": tail["dc_gain_infinite"],
                             "max_abs_C": tail["max_abs_C"],
                             "cosine_truncated_vs_source": cos["cosine_truncated_trace"],
                             "cosine_infinite_vs_source": cos["cosine_infinite_trace"]})
    return pd.DataFrame(rows)


def selection_table() -> pd.DataFrame:
    rows = []
    for tag in ("month", "year", "sc_0629", "sc_0708"):
        path = RAW / f"select_{tag}.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        for B, block in blob["b"].items():
            rows.append({"tag": tag, "B": int(B), "L": block["L"], "rho": block["rho"],
                         "eta_r2": block["eta_r2"], "S_r2": block["S_r2"],
                         "eta_s1": block["s1"]["eta"], "s1_source": block["s1"]["source_r2"],
                         "eta_r1s": block["s1_r1s"]["eta"], "r1s_kind": block["s1_r1s"]["kind"],
                         "r1s_lam": block["s1_r1s"]["lam"], "r1s_source": block["s1_r1s"]["source_r2"],
                         "eta_m4": block["trace"]["4"]["eta"], "m4_source": block["trace"]["4"]["source_r2"],
                         "fit_m4": block["trace"]["4"]["fit_rel"],
                         "eta_m8": block["trace"]["8"]["eta"], "m8_source": block["trace"]["8"]["source_r2"],
                         "fit_m8": block["trace"]["8"]["fit_rel"],
                         "p0_source": block["p0_source_r2"], "n": blob["n_episodes"][B],
                         "sessions": " ".join(s[-8:] for s in block["sessions"])})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stream = load_stream()
    stored = load_stored()
    # Streaming strata names already match the join keys.
    both = pd.concat([stored, stream[["stratum", "bid", "target", "B", "draw", "seed", "method", "r2"]]],
                     ignore_index=True)
    # Drop incomplete method collisions (should not happen: stored and stream use disjoint method names).
    pv = pivot(both)
    pv = add_month_core(pv)
    cells, paired, retained = cells_and_paired(pv)
    cells.to_csv(OUT / "cells.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    retained.to_csv(OUT / "retained.csv", index=False)
    fit_table().to_csv(OUT / "fit.csv", index=False)
    selection_table().to_csv(OUT / "selection.csv", index=False)
    # headline: B in {1,8,33}
    head = cells[cells.B.isin(B_HEAD)].copy()
    head["label"] = head.method.map(LABEL)
    head.to_csv(OUT / "headline.csv", index=False)
    # wide markdown-friendly
    if not head.empty:
        wide = head.pivot_table(index=["stratum", "label"], columns="B", values="r2")
        wide.to_csv(OUT / "headline_wide.csv")
    cost = RAW / "cost.json"
    if cost.exists():
        blob = json.loads(cost.read_text())
        pd.DataFrame(blob["rows"]).to_csv(OUT / "cost.csv", index=False)
    # completeness
    exp = stream.groupby(["stratum", "bid", "method", "B"]).size().reset_index(name="n")
    exp.to_csv(OUT / "stream_counts.csv", index=False)
    print(f"wrote {OUT}", flush=True)
    if not head.empty:
        print(wide.to_string(), flush=True)


if __name__ == "__main__":
    main()
