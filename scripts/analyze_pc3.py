#!/usr/bin/env python
"""Aggregate the P3 rows into compact csv/json (results/analysis/population_calib_p3_20261016)."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "results/population_calib_20261015"
P2 = ROOT / "results/population_calib_p2_20261015"
P3 = ROOT / "results/population_calib_p3_20261016"
OUTD = ROOT / "results/analysis/population_calib_p3_20261016"
STRATA = ("days", "month", "year")
BS = (1, 2, 4, 8, 16, 33)
HEADLINE_SET = {"days": "is", "month": "oos", "year": "oos"}      # fixed a priori (see README)
KINDS = ("full", "diag", "lr8", "lr16", "lr32")


def jl(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_refs(stratum):
    """Reference rows reused from P1/P2 (protocol-identical: same targets, draws, seeds)."""
    p0 = jl(P1 / f"runs/{stratum}_O/p0/seed*/eval_rows.jsonl")
    p0 = p0.assign(method="p0")[["method", "seed", "target", "B", "draw", "r2"]]
    pri = jl(P1 / f"reference/prior_{stratum}.jsonl")
    pri = pri[pri.variant == "p_prior"].assign(method="p_prior")[["method", "seed", "target", "B", "draw", "r2"]]
    ui = jl(P2 / f"uireg/eval_{stratum}.jsonl")
    ui = ui.assign(method=ui.variant)[["method", "seed", "target", "B", "draw", "r2"]]
    r4 = jl(P3 / f"runs/{stratum}_O/p0_nolab_na_gm/seed*/eval_rows.jsonl")
    r4 = r4.assign(method="p0_nolab_gm")[["method", "seed", "target", "B", "draw", "r2"]]
    gm = jl(P2 / f"runs/{stratum}_O/p0_na_gm/seed*/eval_rows.jsonl")
    gm = gm.assign(method="p0_gm")[["method", "seed", "target", "B", "draw", "r2"]]
    pgm = jl(P2 / f"reference/prior_{stratum}_na_gm.jsonl")
    pgm = pgm.assign(method="p_prior_gm")[["method", "seed", "target", "B", "draw", "r2"]]
    nl = jl(P1 / f"runs/{stratum}_O/p0_nolab/seed*/eval_rows.jsonl")
    nl = nl.assign(method="p0_nolab_aligned")[["method", "seed", "target", "B", "draw", "r2"]]
    return pd.concat([p0, pri, ui, r4, gm, pgm, nl], ignore_index=True)


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def cells(df):
    g = df.groupby(["stratum", "set", "method", "B"]).r2
    out = g.agg(["mean", "count"]).reset_index()
    out["ci95"] = g.apply(ci).values
    return out


def seed_avg(df):
    """Average over seeds so every method is a single value per (target, B, draw)."""
    return df.groupby(["stratum", "set", "method", "target", "B", "draw"], as_index=False).r2.mean()


def paired(df, a, b, stratum, set_a, set_b="ref"):
    da = df[(df.stratum == stratum) & (df.method == a) & (df.set == set_a)][["target", "B", "draw", "r2"]]
    db = df[(df.stratum == stratum) & (df.method == b) & (df.set == set_b)][["target", "B", "draw", "r2"]]
    m = da.merge(db, on=["target", "B", "draw"], suffixes=("_a", "_b"))
    m["d"] = m.r2_a - m.r2_b
    return m


def load_eval():
    frames = []
    for s in STRATA:
        for st in ("is", "oos"):
            f = P3 / f"eval/eval_{s}_{st}.jsonl"
            if f.exists():
                d = jl(f)
                d["set"] = st
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_kind(sel, prefix, B):
    """Readout kind with the best source-simulation score at this B (source-only choice)."""
    cand = {k: sel[f"{prefix}_{k}"][str(B)]["source_r2"] for k in KINDS if f"{prefix}_{k}" in sel}
    return max(cand, key=cand.get)


def fmt(m, c):
    return f"{m:.3f}±{c:.3f}" if np.isfinite(c) else f"{m:.3f}"


def table_r2(cl, stratum, st):
    """Markdown: method rows x B columns for one stratum; candidates use selection set ``st``."""
    cand = [("R1", "R1 readout ridge (replay, P0 table)"), ("R1_on_prior", "R1 on P-prior table (readout only)"),
            ("R1_stream", "R1-stream (no replay buffer)"), ("R2", "R2 F7-J"), ("R3", "R3 F7-J + readout ridge")]
    refs = [("p_prior", "P-prior [ref]"), ("p0", "P0"), ("ui_reg", "ui_reg [ref, backprop]"),
            ("p0_ui_reg", "P0+ui_reg [ref, backprop]")]
    lines = ["| method | " + " | ".join(f"B={B}" for B in BS) + " |", "|---|" + "---:|" * len(BS)]

    def row(label, key, sset):
        sub = cl[(cl.stratum == stratum) & (cl.method == key) & (cl.set == sset)].set_index("B")
        vals = [fmt(sub.loc[B, "mean"], sub.loc[B, "ci95"]) if B in sub.index else "-" for B in BS]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    for key, label in refs:
        row(label, key, "ref")
    for key, label in cand:
        row(label, key, st)
    return "\n".join(lines)


def table_gap(gr, stratum, st):
    sub = gr[(gr.stratum == stratum) & (gr.set == st)]
    lines = ["| method | " + " | ".join(f"B={B}" for B in BS) + " |", "|---|" + "---:|" * len(BS)]
    gap = gr[(gr.stratum == stratum)].drop_duplicates("B").set_index("B").gap_p0uireg_minus_p0
    lines.append("| gap = P0+ui_reg − P0 | " + " | ".join(f"{gap.get(B, float('nan')):+.3f}" for B in BS) + " |")
    for m in ("R1", "R1_on_prior", "R1_stream", "R2", "R3"):
        ss = sub[sub.method == m].set_index("B")
        lab = f"{m} − P-prior" if m == "R1_on_prior" else f"{m} − P0"
        vals = []
        for B in BS:
            if B in ss.index:
                r = ss.loc[B]
                frac = f" ({r.recovered_fraction * 100:.0f}%)" if np.isfinite(r.recovered_fraction) else ""
                vals.append(f"{r.delta_vs_p0:+.3f}{frac}")
            else:
                vals.append("-")
        lines.append(f"| {lab} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_tables(cl, gr, kinds_used):
    out = ["# P3 tables (auto-generated by scripts/analyze_pc3.py)", ""]
    for s in STRATA:
        for st in ("is", "oos"):
            tag = " (headline)" if HEADLINE_SET[s] == st else ""
            out += [f"## {s}, selection set = {st}{tag}: query R2 (mean±CI95)", "", table_r2(cl, s, st), "",
                    f"### {s}, selection set = {st}: delta vs P0 (paired; fraction of the P0+ui_reg−P0 gap in brackets when the gap >= 0.02)",
                    "", table_gap(gr, s, st), ""]
    ab_path = OUTD / "ablation_feedback_kernel.csv"
    if ab_path.exists():
        ag = pd.read_csv(ab_path)
        out += ["## Feedback-kernel controls: R2 delta vs P0 (mean±CI95; selected rule of the headline set)", ""]
        for s in STRATA:
            sub = ag[ag.stratum == s]
            bs = sorted(sub.B.unique())
            out += [f"### {s}", "", "| variant | " + " | ".join(f"B={b}" for b in bs) + " |", "|---|" + "---:|" * len(bs)]
            for v in ("jbar", "sign", "random", "zero", "L0", "L8", "L16"):
                row = [f"{sub[(sub.variant == v) & (sub.B == b)]['mean'].iat[0]:+.3f}±{sub[(sub.variant == v) & (sub.B == b)]['ci95'].iat[0]:.3f}"
                       if len(sub[(sub.variant == v) & (sub.B == b)]) else "-" for b in bs]
                out.append(f"| {v} | " + " | ".join(row) + " |")
            out.append("")
    # R4 and readout-kind tables
    out += ["## R4: identity-free / label-free P0 (query R2, mean±CI95; seed-averaged)", ""]
    for s in STRATA:
        out += [f"### {s}", "", "| method | " + " | ".join(f"B={B}" for B in BS) + " |", "|---|" + "---:|" * len(BS)]
        for key, label in (("p_prior_gm", "P-prior (globalmean, no identity)"), ("p0", "P0 aligned (identity + labels)"),
                           ("p0_nolab_aligned", "P0 aligned, label-free"), ("p0_gm", "P0 identity-free (globalmean), labels"),
                           ("p0_nolab_gm", "P0 identity-free + label-free [R4]")):
            sub = cl[(cl.stratum == s) & (cl.method == key) & (cl.set == "ref")].set_index("B")
            out.append(f"| {label} | " + " | ".join(fmt(sub.loc[B, "mean"], sub.loc[B, "ci95"]) if B in sub.index else "-"
                                                    for B in BS) + " |")
        out.append("")
    out += ["## Readout variants (headline selection set): query R2", ""]
    for s in STRATA:
        st = HEADLINE_SET[s]
        out += [f"### {s} (set={st})", "", "| method | " + " | ".join(f"B={B}" for B in BS) + " |", "|---|" + "---:|" * len(BS)]
        for pre, nm in (("r1", "R1 replay, P0 table"), ("r1s", "R1-stream"), ("r1p", "R1 on P-prior table"), ("r3", "R3")):
            for k in KINDS:
                sub = cl[(cl.stratum == s) & (cl.method == f"{pre}_{k}") & (cl.set == st)].set_index("B")
                out.append(f"| {nm} [{k}] | " + " | ".join(fmt(sub.loc[B, "mean"], sub.loc[B, "ci95"]) if B in sub.index else "-"
                                                          for B in BS) + " |")
        out.append("")
    (OUTD / "tables_p3.md").write_text("\n".join(out))


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    ev = load_eval()
    refs = []
    for s in STRATA:
        r = load_refs(s)
        r["stratum"], r["set"] = s, "ref"
        refs.append(r)
    refs = pd.concat(refs, ignore_index=True)
    ev = ev[["stratum", "set", "method", "seed", "target", "B", "draw", "r2"]]
    allr = pd.concat([ev, refs[ev.columns]], ignore_index=True)
    allr_sa = seed_avg(allr)                    # seed-averaged, one value per (target, B, draw)
    # --- headline method rows: per (stratum, set, B) pick the readout kind by source score
    sels = {(s, st): json.loads((P3 / f"select/selection_{s}_{st}.json").read_text())
            for s in STRATA for st in ("is", "oos") if (P3 / f"select/selection_{s}_{st}.json").exists()}
    extra = []
    for (s, st), sel in sels.items():
        for prefix, name in (("r1", "R1"), ("r1p", "R1_on_prior"), ("r1s", "R1_stream"), ("r3", "R3")):
            for B in BS:
                if not all(f"{prefix}_{k}" in sel and str(B) in sel[f"{prefix}_{k}"] for k in KINDS):
                    continue
                kind = best_kind(sel, prefix, B)
                src = allr_sa[(allr_sa.stratum == s) & (allr_sa.set == st) & (allr_sa.method == f"{prefix}_{kind}")
                              & (allr_sa.B == B)].copy()
                src["method"] = name
                src["kind"] = kind
                extra.append(src)
        src = allr_sa[(allr_sa.stratum == s) & (allr_sa.set == st) & (allr_sa.method == "r2")].copy()
        src["method"] = "R2"
        extra.append(src)
    ex = pd.concat(extra, ignore_index=True) if extra else pd.DataFrame()
    allh = pd.concat([allr_sa, ex.drop(columns=["kind"], errors="ignore")], ignore_index=True)
    allh.to_csv(OUTD / "all_rows_seedavg.csv.gz", index=False)
    cl = cells(allh.assign(stratum=allh.stratum))
    cl.to_csv(OUTD / "cells_r2.csv", index=False)
    kinds_used = ex.dropna(subset=["kind"]).groupby(["stratum", "set", "method", "B"]).kind.agg(
        lambda x: x.mode().iat[0]).reset_index() if len(ex) and "kind" in ex else pd.DataFrame()
    kinds_used.to_csv(OUTD / "chosen_readout_kind.csv", index=False)

    # --- gap recovery
    rows = []
    for s in STRATA:
        for B in BS:
            pu = paired(allh, "p0_ui_reg", "p0", s, "ref")
            pu = pu[pu.B == B]
            gap = float(pu.d.mean()) if len(pu) else float("nan")
            for st in ("is", "oos"):
                for m in ("R1", "R1_on_prior", "R1_stream", "R2", "R3"):
                    comp = "p_prior" if m == "R1_on_prior" else "p0"       # readout-only rows start from P-prior
                    pm = paired(allh, m, comp, s, st)
                    pm = pm[pm.B == B]
                    if not len(pm):
                        continue
                    d = float(pm.d.mean())
                    rows.append({"stratum": s, "set": st, "method": m, "B": B, "delta_vs_p0": d,
                                 "comparator": comp, "ci95": ci(pm.d), "n": len(pm),
                                 "frac_draws_worse_than_comparator_by_0.05": float((pm.d < -0.05).mean()),
                                 "gap_p0uireg_minus_p0": gap,
                                 "recovered_fraction": (d / gap if abs(gap) >= 0.02 else float("nan"))
                                 if comp == "p0" else float("nan")})
    gr = pd.DataFrame(rows)
    gr.to_csv(OUTD / "gap_recovery.csv", index=False)

    # --- selection summary
    srows = []
    for (s, st), sel in sels.items():
        for method, byB in sel.items():
            for B, v in byB.items():
                srows.append({"stratum": s, "set": st, "method": method, "B": int(B),
                              "cfg": json.dumps(v["cfg"]), "source_r2": v["source_r2"], "n_episodes": v["n_episodes"]})
    pd.DataFrame(srows).sort_values(["stratum", "set", "method", "B"]).to_csv(OUTD / "selection.csv", index=False)

    # --- diagnostics: gradient alignment
    dg = []
    for s in STRATA:
        for st in ("is", "oos"):
            f = P3 / f"eval/diag_{s}_{st}.jsonl"
            if f.exists():
                d = jl(f)
                d["set"] = st
                dg.append(d)
    if dg:
        dg = pd.concat(dg, ignore_index=True)
        agg = dg.groupby(["stratum", "set", "kind", "L", "step", "B"]).agg(
            cos_B=("cos_B", "median"), cos_query=("cos_query", "median"),
            cos_true_B_vs_query=("cos_true_B_vs_query", "median"), n=("cos_B", "size")).reset_index()
        agg.to_csv(OUTD / "diag_gradient_alignment.csv", index=False)

    # --- flags
    if len(ev):
        full = jl(P3 / "eval/eval_*_*.jsonl")
        flags = {"n_rows": int(len(full)),
                 "h2_all_zero_backprop": bool(full.h2_zero_backprop.all()),
                 "h4_all_disjoint": bool(full.h4_query_disjoint.all()),
                 "trunk_hash_unchanged_all": bool(full.trunk_hash_unchanged.dropna().all())}
        (OUTD / "flags.json").write_text(json.dumps(flags, indent=1))
    # --- feedback-kernel controls (R2 at the selected rule; target query only reported)
    ab = [jl(f).assign(f=Path(f).name) for f in sorted(glob.glob(str(P3 / "eval/ablate_*.jsonl")))]
    if ab:
        ab = pd.concat(ab, ignore_index=True)
        ab["delta_vs_p0"] = ab.r2 - ab.p0_r2
        agg = ab.groupby(["stratum", "set", "B", "variant"]).delta_vs_p0.agg(["mean", "count"]).reset_index()
        agg["ci95"] = ab.groupby(["stratum", "set", "B", "variant"]).delta_vs_p0.apply(ci).values
        agg["frac_worse_0.05"] = ab.groupby(["stratum", "set", "B", "variant"]).delta_vs_p0.apply(
            lambda x: float((x < -0.05).mean())).values
        agg.to_csv(OUTD / "ablation_feedback_kernel.csv", index=False)
    # pooled recovered fraction over B in {4,8,16,33}: sum(delta)/sum(gap)
    pool = []
    for s_ in STRATA:
        for st in ("is", "oos"):
            for m in ("R1", "R1_stream", "R2", "R3"):
                sub = gr[(gr.stratum == s_) & (gr.set == st) & (gr.method == m) & (gr.B >= 4)]
                if len(sub):
                    pool.append({"stratum": s_, "set": st, "method": m, "pooled_delta": sub.delta_vs_p0.mean(),
                                 "pooled_gap": sub.gap_p0uireg_minus_p0.mean(),
                                 "pooled_fraction": sub.delta_vs_p0.mean() / sub.gap_p0uireg_minus_p0.mean()})
    pd.DataFrame(pool).to_csv(OUTD / "gap_recovery_pooled_B4plus.csv", index=False)
    write_tables(cl, gr, kinds_used)
    # --- Jacobian representativeness (offline source-only quantity)
    import torch
    jr = []
    for s in STRATA:
        f = P3 / f"jacobian_{s}.pt"
        if not f.exists():
            continue
        blob = torch.load(f, weights_only=False)
        tot, sq, n = 0, 0, 0
        for v in blob["sessions"].values():
            tot, sq, n = tot + v["sum"], sq + v["sq"], n + v["n"]
        jbar = tot / n
        var = (sq / n - (jbar ** 2).sum((1, 2))).clamp_min(0)
        rel = (var.sqrt() / jbar.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)).numpy()
        for l in (0, 1, 2, 4, 8, 16, 24, 32):
            jr.append({"stratum": s, "lag": l, "jbar_frobenius": float(jbar[l].norm()),
                       "single_window_rms_frobenius": float((sq / n)[l].sqrt()), "rel_deviation": float(rel[l])})
    pd.DataFrame(jr).to_csv(OUTD / "jacobian_representativeness.csv", index=False)
    print("done", OUTD)


if __name__ == "__main__":
    main()
