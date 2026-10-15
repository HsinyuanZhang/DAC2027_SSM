#!/usr/bin/env python
"""P3 review-fix aggregation (like-for-like gap recovery, causal month, 2x2, bootstrap, variants).

Inputs are the original P3 rows (results/population_calib_p3_20261016/{eval,select}) and the review-fix
rows (results/population_calib_p3_20261016/fix/{eval,select,uiref}).  Whatever has been run is
aggregated; cells whose rows do not exist are reported as missing, never filled.

Reference ("gap denominator") P0+ui_reg tags, all non-deployable, rule selected on SOURCE only:
  p2     the P2 rule: selected on the six trunk-source sessions from the LOO PRIOR start (superseded)
  is0    IS sessions, P0 start (like-for-like for IS candidates)
  oos    the four OOS sessions, P0 start (like-for-like for OOS candidates; year)
  oosc   causal OOS (month: 0307 only), P0 start (like-for-like for the month causal headline)

Outputs in results/analysis/population_calib_p3_20261016/:
  cells_fix.csv, gap_recovery_fix.csv, gap_pooled_boot.csv, gap_2x2.csv, diag_cos_pooled.csv,
  diag_traj_fix.csv, variant_summary.csv, tables_p3_fix.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_pc3 import (BS, KINDS, OUTD, P1, P2, P3, best_kind, ci, jl, load_refs,  # noqa: E402
                                 seed_avg)

FIX = P3 / "fix"
STRATA = ("days", "month", "year")
# headline candidate set per stratum (month: CAUSAL; days: IS; year: OOS, already causal)
HEADLINE = {"days": "is", "month": "oosc", "year": "oos"}
LIKE_REF = {"is": "is0", "oos": "oos", "oosc": "oosc"}
VARIANTS = ("base", "contig", "cap256", "ext")
SEL_OF_VARIANT = {"base": "base", "contig": "contig", "cap256": "base", "ext": "ext"}
NBOOT = 10000


def suf(v):
    return "" if v == "base" else f"_{v}"


def load_candidates():
    frames = []
    for s in STRATA:
        for st in ("is", "oos", "oosc"):
            for v in VARIANTS:
                files = [FIX / f"eval/eval_{s}_{st}{suf(v)}.jsonl"]
                if v == "base":
                    files.append(P3 / f"eval/eval_{s}_{st}.jsonl")
                f = next((x for x in files if x.exists()), None)
                if f is None:
                    continue
                d = jl(f)
                d["set"], d["variant"] = st, v
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sel_json(s, st, v):
    sv = SEL_OF_VARIANT[v]
    for base in (FIX, P3):
        p = base / f"select/selection_{s}_{st}{suf(sv)}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def candidate_rows(ev):
    """Seed-averaged headline-method rows per (stratum, set, variant): R1, R1_on_prior, R1_stream, R2, R3."""
    out = []
    ev = ev[["stratum", "set", "variant", "method", "seed", "target", "B", "draw", "r2"]]
    sa = ev.groupby(["stratum", "set", "variant", "method", "target", "B", "draw"], as_index=False).r2.mean()
    for (s, st, v), grp in sa.groupby(["stratum", "set", "variant"]):
        sel = sel_json(s, st, v)
        for prefix, name in (("r1", "R1"), ("r1p", "R1_on_prior"), ("r1s", "R1_stream"), ("r3", "R3")):
            if sel is None or not all(f"{prefix}_{k}" in sel for k in KINDS):
                continue
            for B in BS:
                if not all(str(B) in sel[f"{prefix}_{k}"] for k in KINDS):
                    continue
                kind = best_kind(sel, prefix, B)
                src = grp[(grp.method == f"{prefix}_{kind}") & (grp.B == B)].copy()
                if len(src):
                    out.append(src.assign(method=name))
        for m, name in (("r2", "R2"), ("p0", "p0_variant")):
            src = grp[grp.method == m]
            if len(src):
                out.append(src.assign(method=name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def reference_rows():
    refs = []
    for s in STRATA:
        r = load_refs(s)
        r = r[r.method.isin(["p0", "p_prior", "p0_ui_reg", "ui_reg"])].copy()
        r["method"] = r.method.replace({"p0_ui_reg": "ref_p2"})
        refs.append(r.assign(stratum=s))
        for f in glob.glob(str(FIX / f"uiref/eval_{s}_*.jsonl")):
            d = jl(f)
            tag = Path(f).stem.split("_", 2)[2]
            d = d.assign(method=f"ref_{tag}")[["method", "seed", "target", "B", "draw", "r2"]]
            refs.append(d.assign(stratum=s))
    refs = pd.concat(refs, ignore_index=True)
    return refs.groupby(["stratum", "method", "target", "B", "draw"], as_index=False).r2.mean()


def paired(df, a, b):
    m = df[df.method == a][["target", "B", "draw", "r2"]].merge(
        df[df.method == b][["target", "B", "draw", "r2"]], on=["target", "B", "draw"], suffixes=("_a", "_b"))
    m["d"] = m.r2_a - m.r2_b
    return m


def boot_pooled(delta_by_b, gap_by_b, rng, n=NBOOT):
    """Draw-level bootstrap of sum_B mean(delta) / sum_B mean(gap), draws resampled per B (paired)."""
    keys = [b for b in delta_by_b if b in gap_by_b and len(delta_by_b[b]) == len(gap_by_b[b]) > 1]
    if not keys:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pt = sum(delta_by_b[b].mean() for b in keys) / sum(gap_by_b[b].mean() for b in keys)
    est = np.empty(n)
    ix = {b: rng.integers(0, len(delta_by_b[b]), size=(n, len(delta_by_b[b]))) for b in keys}
    num = sum(delta_by_b[b][ix[b]].mean(1) for b in keys)
    den = sum(gap_by_b[b][ix[b]].mean(1) for b in keys)
    est = num / den
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(pt), float(lo), float(hi), float((den <= 0).mean())


def gap_tables(cand, refs):
    rng = np.random.default_rng(0)
    rows, pooled = [], []
    for (s, st, v), grp in cand.groupby(["stratum", "set", "variant"]):
        r = refs[refs.stratum == s]
        p0 = r[r.method == "p0"]
        for ref in sorted(set(m for m in r.method if m.startswith("ref_"))):
            tag = ref[4:]
            refrows = r[r.method == ref]
            for m in ("R1", "R1_stream", "R2", "R3"):
                cm = grp[grp.method == m]
                if not len(cm):
                    continue
                dby, gby = {}, {}
                for B in BS:
                    a = cm[cm.B == B][["target", "draw", "r2"]]
                    p = p0[p0.B == B][["target", "draw", "r2"]]
                    f = refrows[refrows.B == B][["target", "draw", "r2"]]
                    mm = a.merge(p, on=["target", "draw"], suffixes=("_c", "_p0")).merge(
                        f.rename(columns={"r2": "r2_ref"}), on=["target", "draw"])
                    if not len(mm):
                        continue
                    d, g = (mm.r2_c - mm.r2_p0).to_numpy(), (mm.r2_ref - mm.r2_p0).to_numpy()
                    dby[B], gby[B] = d, g
                    gap = g.mean()
                    rows.append({"stratum": s, "cand_set": st, "variant": v, "ref": tag, "method": m, "B": B,
                                 "delta_vs_p0": d.mean(), "ci95": ci(d), "gap": gap, "gap_ci95": ci(g),
                                 "fraction": d.mean() / gap if abs(gap) >= 0.02 else np.nan, "n": len(d)})
                big = [b for b in dby if b >= 4]
                pt, lo, hi, bad = boot_pooled({b: dby[b] for b in big}, {b: gby[b] for b in big}, rng)
                if big and abs(np.mean([gby[b].mean() for b in big])) < 0.02:
                    pt = lo = hi = float("nan")             # no gap to recover: a ratio is meaningless
                if big:
                    pooled.append({"stratum": s, "cand_set": st, "variant": v, "ref": tag, "method": m,
                                   "pooled_delta_B4plus": np.mean([dby[b].mean() for b in big]),
                                   "pooled_gap_B4plus": np.mean([gby[b].mean() for b in big]),
                                   "pooled_fraction": pt, "boot_lo": lo, "boot_hi": hi,
                                   "boot_frac_nonpositive_gap": bad,
                                   "n_B_cells": len(big), "n_draws_per_B": len(dby[big[0]])})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def two_by_two(pooled):
    """Candidate selection (IS / OOS-or-causal) x reference selection (IS / OOS-or-causal); base variant."""
    out = []
    for s in ("month", "year", "days"):
        cs = {"IS": "is", "OOS": HEADLINE[s] if HEADLINE[s] != "is" else "oos"}
        for cname, cset in cs.items():
            for ref_name, tags in (("ref IS (P0 start, is0)", "is0"), ("ref IS (P2 rule, prior start)", "p2"),
                                   ("ref OOS (P0 start, same sessions)", LIKE_REF[cset] if cset != "is" else HEADLINE_OOS(s))):
                for m in ("R1", "R2", "R3"):
                    sub = pooled[(pooled.stratum == s) & (pooled.cand_set == cset) & (pooled.variant == "base")
                                 & (pooled.ref == tags) & (pooled.method == m)]
                    if len(sub):
                        r = sub.iloc[0]
                        out.append({"stratum": s, "candidate_selection": f"{cname} ({cset})", "reference": ref_name,
                                    "method": m, "pooled_fraction_B4plus": r.pooled_fraction, "boot_lo": r.boot_lo,
                                    "boot_hi": r.boot_hi, "pooled_gap": r.pooled_gap_B4plus,
                                    "pooled_delta": r.pooled_delta_B4plus})
    return pd.DataFrame(out)


def HEADLINE_OOS(s):
    return {"days": "oos", "month": "oosc", "year": "oos"}[s]


def diag_tables():
    """Cosine diagnostics pooled over ALL episodes (not medians of per-B medians)."""
    frames = []
    for f in sorted(list(glob.glob(str(P3 / "eval/diag_*.jsonl"))) + list(glob.glob(str(FIX / "eval/diag_*.jsonl")))):
        d = jl(f)
        d["src"] = "fix" if "/fix/" in f else "orig"
        d["set"] = Path(f).stem.split("_")[2]
        frames.append(d)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    p = d[d.kind == "p0_step1"]
    pooled = p.groupby(["stratum", "set", "src", "L"]).agg(
        cos_B_median=("cos_B", "median"), cos_B_q25=("cos_B", lambda x: x.quantile(0.25)),
        cos_B_q75=("cos_B", lambda x: x.quantile(0.75)), cos_query_median=("cos_query", "median"),
        n_episodes=("cos_B", "size")).reset_index()
    t = d[(d.kind == "selected_traj") & (d.src == "fix")]
    traj = pd.DataFrame()
    if len(t):
        traj = t.groupby(["stratum", "set", "step"]).agg(
            cos_g_B=("cos_B", "median"), cos_upd_B=("cos_upd_B", "median"),
            cos_upd_query=("cos_upd_query", "median"), n=("cos_B", "size")).reset_index()
    return pooled, traj


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, lambda x: str(x))(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else "-"
                                       for c in cols) + " |")
    return "\n".join(lines)


def main():
    ev = load_candidates()
    cand = candidate_rows(ev) if len(ev) else pd.DataFrame()
    refs = reference_rows()
    pooled = pd.DataFrame()
    if len(cand):
        cand = cand[cand.method != "p0_variant"]
        gaps, pooled = gap_tables(cand, refs)
        gaps.to_csv(OUTD / "gap_recovery_fix.csv", index=False)
        pooled.to_csv(OUTD / "gap_pooled_boot.csv", index=False)
        if (pooled.ref != "p2").any():                   # a 2x2 needs at least one P0-start reference
            two_by_two(pooled).to_csv(OUTD / "gap_2x2.csv", index=False)
        cells = cand.groupby(["stratum", "set", "variant", "method", "B"]).r2.agg(["mean", "count"]).reset_index()
        cells["ci95"] = cand.groupby(["stratum", "set", "variant", "method", "B"]).r2.apply(ci).values
        cells.to_csv(OUTD / "cells_fix.csv", index=False)
    cos, traj = diag_tables()
    if len(cos):
        cos.to_csv(OUTD / "diag_cos_pooled.csv", index=False)
    if len(traj):
        traj.to_csv(OUTD / "diag_traj_fix.csv", index=False)
    present = sorted(set(m for m in refs.method if m.startswith("ref_")))
    print("references present:", present)
    if len(pooled):
        cols = ["stratum", "cand_set", "variant", "ref", "method", "pooled_gap_B4plus", "pooled_delta_B4plus",
                "pooled_fraction", "boot_lo", "boot_hi"]
        print(pooled[pooled.variant == "base"][cols].round(3).to_string())
    if len(cos):
        print(cos[cos.L == 32].round(3).to_string())


if __name__ == "__main__":
    main()
