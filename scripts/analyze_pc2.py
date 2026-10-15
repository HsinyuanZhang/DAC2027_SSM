#!/usr/bin/env python
"""Aggregate the P2 iteration (A'1 identity-free, A'2 ui_reg, A'3 quantisation)
into compact csv/json + markdown tables under results/analysis/population_calib_p2_20261015.

Usage: analyze_pc2.py [a1] [a2] [a3]      (default: all that have rows)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "results/population_calib_20261015"
P2 = ROOT / "results/population_calib_p2_20261015"
ANA = ROOT / "results/analysis/population_calib_p2_20261015"
STRATA = ("days", "month", "year")
BUDGETS = (1, 2, 4, 8, 16, 33)
MD: list[str] = []


def ci(x):
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def read_jsonl(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame(columns=["variant", "stratum", "target", "B", "draw", "r2", "seed", "cfg"])
    return pd.DataFrame(rows)


def md_table(df, nd=3):
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                for v in r.values) + " |\n"
    return out


def cell(df, by, value="r2"):
    g = df.groupby(list(by))[value]
    out = g.agg(["mean", "count"]).reset_index()
    out["ci95"] = g.apply(ci).values
    return out.rename(columns={"mean": value})


def fmt(m, c):
    return f"{m:.3f}±{c:.3f}" if c == c else f"{m:.3f}"


# --------------------------------------------------------------------- A'1 ----

MODES = {"aligned": "", "globalmean": "_na_gm", "zero": "_na_zero"}
BASES = ("p0", "p2", "p3_m8", "p4_m8")


def a1():
    ev2 = read_jsonl(sorted((P2 / "runs").glob("*_O/*/seed*/eval_rows.jsonl")))
    ev1 = read_jsonl(sorted(P1.glob("runs/*_O/*/seed*/eval_rows.jsonl")))
    ev1 = ev1[ev1.variant.isin(BASES)]
    ev = pd.concat([ev1, ev2], ignore_index=True)
    ev["mode"], ev["base"] = "aligned", ev.variant
    for mode, suf in MODES.items():
        if suf:
            m = ev.variant.str.endswith(suf)
            ev.loc[m, "mode"] = mode
            ev.loc[m, "base"] = ev.loc[m, "variant"].str[:-len(suf)]
    ev = ev[ev.base.isin(BASES)]
    pri = read_jsonl(sorted((P2 / "reference").glob("prior_*_na_*.jsonl")))
    pri["mode"] = np.where(pri.variant.str.endswith("_na_gm"), "globalmean", "zero")
    pri1 = read_jsonl(sorted((P1 / "reference").glob("prior_*.jsonl")))
    pri1 = pri1[pri1.variant == "p_prior"].assign(mode="aligned")
    pri = pd.concat([pri, pri1], ignore_index=True).assign(base="p_prior")
    allr = pd.concat([ev, pri], ignore_index=True)
    allr.drop(columns=[c for c in allr.columns if c.startswith("h")], errors="ignore").to_csv(
        ANA / "a1_rows.csv.gz", index=False)
    cells = cell(allr, ("stratum", "mode", "base", "B"))
    cells.to_csv(ANA / "a1_cells_r2.csv", index=False)
    # --- main table B = 1, 8, 33 -------------------------------------------------
    for mode in ("globalmean", "zero"):
        MD.append(f"### A'1 query R2, prior={mode} (identity-free) vs aligned; pool O; mean±CI\n")
        cols = [(st, b) for st in STRATA for b in (1, 8, 33)]
        head = pd.DataFrame(index=[f"{m}:{b}" for m in ("aligned", mode) for b in ("p_prior",) + BASES])
        for st, b in cols:
            vals = []
            for m in ("aligned", mode):
                for base in ("p_prior",) + BASES:
                    s = cells[(cells.stratum == st) & (cells["mode"] == m) & (cells.base == base) & (cells.B == b)]
                    vals.append(fmt(float(s.r2.iloc[0]), float(s.ci95.iloc[0])) if len(s) else "-")
            head[f"{st} B={b}"] = vals
        MD.append(md_table(head.reset_index().rename(columns={"index": "mode:variant"})))
    # --- paired diffs vs P0 within the same mode -----------------------------------
    keys = ["stratum", "target", "B", "draw", "seed"]
    rows, tab = [], {}
    for mode in ("globalmean", "zero", "aligned"):
        e = ev[ev["mode"] == mode]
        p0 = e[e.base == "p0"][keys + ["r2"]].rename(columns={"r2": "r2_p0"})
        for base in ("p2", "p3_m8", "p4_m8"):
            x = e[e.base == base][keys + ["r2"]].merge(p0, on=keys)
            x["diff"] = x.r2 - x.r2_p0
            for st in STRATA:
                xs = x[x.stratum == st]
                if not len(xs):
                    continue
                per_seed = xs.groupby("seed")["diff"].mean()
                rows.append({"mode": mode, "variant": base, "stratum": st, "B": "all",
                             "diff_vs_p0": xs["diff"].mean(), "ci95": ci(xs["diff"]), "n": len(xs),
                             "seed0": per_seed.get(0, np.nan), "seed1": per_seed.get(1, np.nan)})
                for b in BUDGETS:
                    xb = xs[xs.B == b]
                    rows.append({"mode": mode, "variant": base, "stratum": st, "B": b,
                                 "diff_vs_p0": xb["diff"].mean(), "ci95": ci(xb["diff"]),
                                 "n": len(xb), "seed0": np.nan, "seed1": np.nan})
    pd_ = pd.DataFrame(rows)
    # seed noise floor of the P0 cell mean (|seed0 - seed1| of target-averaged R2), per mode/stratum
    fl = (ev.groupby(["stratum", "mode", "base", "B", "seed"]).r2.mean().unstack("seed"))
    fl["spread"] = (fl[0] - fl[1]).abs()
    floor = fl.reset_index().groupby(["stratum", "mode"]).spread.mean().rename("seed_floor").reset_index()
    floor.to_csv(ANA / "a1_seed_floor.csv", index=False)
    pd_ = pd_.merge(floor, on=["stratum", "mode"], how="left")
    pd_["ratio_to_floor"] = pd_.diff_vs_p0.abs() / pd_.seed_floor
    pd_.to_csv(ANA / "a1_paired_vs_p0.csv", index=False)
    MD.append("### A'1 paired difference variant - P0 (same mode, target, B, draw, seed); all B pooled\n")
    MD.append(md_table(pd_[pd_.B == "all"][["mode", "variant", "stratum", "diff_vs_p0", "ci95", "seed0",
                                            "seed1", "seed_floor", "ratio_to_floor", "n"]]))
    MD.append("### A'1 paired difference by B (B in 1,8,33)\n")
    sub = pd_[pd_.B.isin([1, 8, 33])]
    pv = sub.pivot_table(index=["mode", "variant", "stratum"], columns="B", values="diff_vs_p0").reset_index()
    MD.append(md_table(pv))
    # --- gap to aligned P0 ---------------------------------------------------------
    g_rows = []
    for mode in ("globalmean", "zero"):
        for base in BASES:
            a = ev[(ev["mode"] == "aligned") & (ev.base == "p0")][keys + ["r2"]].rename(
                columns={"r2": "r2_al"})
            b_ = ev[(ev["mode"] == mode) & (ev.base == base)][keys + ["r2"]].merge(a, on=keys)
            b_["gap"] = b_.r2 - b_.r2_al
            for st in STRATA:
                for B in (1, 2, 4, 8, 16, 33):
                    s = b_[(b_.stratum == st) & (b_.B == B)]
                    if len(s):
                        g_rows.append({"mode": mode, "variant": base, "stratum": st, "B": B,
                                       "gap_to_aligned_p0": s.gap.mean(), "ci95": ci(s.gap)})
    gap = pd.DataFrame(g_rows)
    gap.to_csv(ANA / "a1_gap_to_aligned_p0.csv", index=False)
    MD.append("### A'1 gap to ALIGNED P0 (identity-free variant - aligned P0), by B\n")
    MD.append(md_table(gap.pivot_table(index=["mode", "variant", "stratum"], columns="B",
                                       values="gap_to_aligned_p0").reset_index()))
    # --- gain over the identity-free prior ------------------------------------------
    gr = []
    for mode in ("globalmean", "zero"):
        pp = pri[pri["mode"] == mode][["stratum", "target", "B", "draw", "r2"]].rename(columns={"r2": "r2_prior"})
        for base in BASES:
            x = ev[(ev["mode"] == mode) & (ev.base == base)].merge(pp, on=["stratum", "target", "B", "draw"])
            x["gain"] = x.r2 - x.r2_prior
            for st in STRATA:
                for B in (1, 8, 33):
                    s = x[(x.stratum == st) & (x.B == B)]
                    if len(s):
                        gr.append({"mode": mode, "variant": base, "stratum": st, "B": B,
                                   "gain_over_noalign_prior": s.gain.mean(), "ci95": ci(s.gain)})
    pd.DataFrame(gr).to_csv(ANA / "a1_gain_over_noalign_prior.csv", index=False)
    # --- training diagnostics ----------------------------------------------------------
    vals = []
    for p in sorted(P2.glob("runs/*_O/*/seed*/done.json")):
        d = json.loads(p.read_text())
        vals.append({"stratum": p.parts[-4].split("_")[0], "tag": p.parts[-3], "seed": int(p.parts[-2][4:]),
                     "best_val_r2": d["best_val_r2"], "val_prior_r2": d["val_prior_r2"],
                     "best_step": d["best_step"], "final_budget": d["final_budget"],
                     "converged": d["converged"]})
    vd = pd.DataFrame(vals)
    vd.to_csv(ANA / "a1_training.csv", index=False)
    MD.append("### A'1 training (source val R2 of the best step; converged = best step before the last 20% of the budget)\n")
    MD.append(md_table(vd.groupby(["stratum", "tag"]).agg(val=("best_val_r2", "mean"),
                                                         best_step=("best_step", "mean"),
                                                         converged=("converged", "mean")).reset_index()))
    # ---- constraint flags
    flags = {"rows": int(len(ev2)), "h1": float(ev2.h1_trunk_unchanged.mean()),
             "h2": float(ev2.h2_zero_backprop.mean()), "h4": float(ev2.h4_query_disjoint.mean()),
             "h5": float(ev2.h5_within_32kb.mean()), "unit_ok": float(ev2.unit_check_passed.mean())}
    (ANA / "a1_flags.json").write_text(json.dumps(flags, indent=1))
    MD.append(f"A'1 constraint flags: `{json.dumps(flags)}`\n")


# --------------------------------------------------------------------- A'2 ----

def a2():
    ur = read_jsonl(sorted((P2 / "uireg").glob("eval_*.jsonl")))
    ur.to_csv(ANA / "a2_rows.csv.gz", index=False)
    sel = {}
    for st in ("days", "month"):
        p = P2 / "uireg" / f"selection_{st}.json"
        if p.exists():
            sel[st] = json.loads(p.read_text())
    (ANA / "a2_selection.json").write_text(json.dumps(sel, indent=1))
    MD.append("### A'2 source-selected (lam, steps, lr) per B (simulated on SOURCE sessions only)\n")
    MD.append(md_table(pd.DataFrame([{"stratum": st, "B": int(b), **v} for st, d in sel.items()
                                     for b, v in d.items()])))
    ref1 = read_jsonl(sorted(P1.glob("reference/*.jsonl")))
    ref1 = ref1[ref1.variant.isin(["ui_elec", "ui_prior"])]
    ev1 = read_jsonl(sorted(P1.glob("runs/*_O/p0/seed*/eval_rows.jsonl")))
    ev1 = ev1[ev1.variant == "p0"]
    pp = read_jsonl(sorted(P1.glob("reference/prior_*.jsonl")))
    pp = pp[pp.variant == "p_prior"]
    allr = pd.concat([ur.assign(src="p2"), ev1.assign(src="p1"), ref1.assign(src="p1"), pp.assign(src="p1")],
                     ignore_index=True)
    cells = cell(allr, ("stratum", "variant", "B"))
    cells.to_csv(ANA / "a2_cells_r2.csv", index=False)
    MD.append("### A'2 query R2 (mean±CI; ui_elec/ui_prior: 5 draws, 1 seed, B in 1/8/33; others 8 draws)\n")
    order = ["p_prior", "p0", "ui_elec", "ui_prior", "ui_reg", "p0_ui_reg"]
    for st in STRATA:
        t = pd.DataFrame({"variant": order})
        for b in BUDGETS:
            vals = []
            for v in order:
                s = cells[(cells.stratum == st) & (cells.variant == v) & (cells.B == b)]
                vals.append(fmt(float(s.r2.iloc[0]), float(s.ci95.iloc[0])) if len(s) else "-")
            t[f"B={b}"] = vals
        MD.append(f"{st}:\n")
        MD.append(md_table(t))
    # paired
    keys = ["stratum", "target", "B", "draw"]
    out = []
    p0m = ev1.groupby(keys).r2.mean().rename("r2_p0").reset_index()       # seed-mean P0
    priorm = pp[keys + ["r2"]].rename(columns={"r2": "r2_prior"})
    uim = ur[ur.variant == "ui_reg"][keys + ["r2"]].rename(columns={"r2": "r2_uireg"})
    p0u = ur[ur.variant == "p0_ui_reg"].groupby(keys).r2.mean().rename("r2_p0uireg").reset_index()
    ue = ref1[ref1.variant == "ui_elec"][keys + ["r2"]].rename(columns={"r2": "r2_uielec"})
    m = p0m.merge(priorm, on=keys).merge(uim, on=keys, how="left").merge(p0u, on=keys, how="left").merge(
        ue, on=keys, how="left")
    for a, b in (("r2_uireg", "r2_prior"), ("r2_uireg", "r2_uielec"), ("r2_p0uireg", "r2_p0"),
                 ("r2_p0", "r2_uireg"), ("r2_p0uireg", "r2_uireg")):
        d = (m[a] - m[b]).rename("d")
        for st in STRATA:
            for B in BUDGETS:
                s = d[(m.stratum == st) & (m.B == B)].dropna()
                if len(s):
                    out.append({"a": a[3:], "b": b[3:], "stratum": st, "B": B, "diff": s.mean(),
                                "ci95": ci(s), "n": len(s)})
    po = pd.DataFrame(out)
    po.to_csv(ANA / "a2_paired.csv", index=False)
    MD.append("### A'2 paired differences (a - b; same target/B/draw; P0 seed-averaged)\n")
    MD.append(md_table(po.pivot_table(index=["a", "b", "stratum"], columns="B", values="diff").reset_index()))


# --------------------------------------------------------------------- A'3 ----

def a3():
    q = read_jsonl(sorted((P2 / "quant").glob("rows_*.jsonl")))
    q = q.drop_duplicates(["stratum", "variant", "seed", "kind", "session", "B", "draw", "cfg"], keep="last")
    q.to_csv(ANA / "a3_rows.csv.gz", index=False)
    order = list(dict.fromkeys(q.cfg))
    g = q.groupby(["variant", "kind", "B", "cfg"]).agg(
        r2=("r2", "mean"), r2_fp=("r2_fp", "mean"), d_r2=("d_r2", "mean"),
        d_r2_ci=("d_r2", ci), rms_delta=("rms_delta", "mean"), rel_rms=("rel_rms", "mean"),
        n=("r2", "count"), nonfinite=("finite", lambda s: int((~s).sum()))).reset_index()
    g.to_csv(ANA / "a3_cells.csv", index=False)
    gs = q.groupby(["variant", "kind", "stratum", "B", "cfg"]).agg(
        d_r2=("d_r2", "mean"), rel_rms=("rel_rms", "mean")).reset_index()
    gs.to_csv(ANA / "a3_cells_by_stratum.csv", index=False)
    for variant in q.variant.unique():
        for kind in ("val", "target"):
            sub = g[(g.variant == variant) & (g.kind == kind)]
            if not len(sub):
                continue
            t = pd.DataFrame({"cfg": order})
            for b in (1, 8, 33):
                s = sub[sub.B == b].set_index("cfg")
                t[f"dR2 B={b}"] = [f"{s.d_r2[c]:+.4f}±{s.d_r2_ci[c]:.4f}" if c in s.index else "-" for c in order]
                t[f"relRMS B={b}"] = [f"{100 * s.rel_rms[c]:.2f}%" if c in s.index else "-" for c in order]
            fp = sub[(sub.cfg == "fp")].set_index("B").r2
            MD.append(f"### A'3 {variant}, {kind} sessions (FP R2: " +
                      ", ".join(f"B={b}: {fp[b]:.3f}" for b in fp.index) +
                      "); dR2 = quantised - FP (mean±CI over rows); relRMS = RMS(pred_q - pred_FP)/std(pred_FP)\n")
            MD.append(md_table(t))


def a3_qat():
    paths = sorted((P2 / "quant").glob("rows_qat_*.jsonl"))
    if not paths:
        return
    qa = read_jsonl(paths).drop_duplicates(["stratum", "variant", "seed", "kind", "session", "B", "draw", "cfg"])
    qa["base"] = qa.variant.str.replace("_qat", "", regex=False)
    main_rows = read_jsonl(sorted((P2 / "quant").glob("rows_[dmy]*.jsonl")))
    fp = main_rows[main_rows.cfg == "fp"].drop_duplicates(
        ["stratum", "variant", "seed", "kind", "session", "B", "draw"])
    fp = fp.rename(columns={"variant": "base", "r2": "r2_fp0"})[
        ["stratum", "base", "seed", "kind", "session", "B", "draw", "r2_fp0"]]
    ptq = main_rows[main_rows.cfg == "tern"].rename(columns={"variant": "base", "r2": "r2_ptq"})[
        ["stratum", "base", "seed", "kind", "session", "B", "draw", "r2_ptq"]]
    t = qa[qa.cfg == "tern"].merge(fp, on=["stratum", "base", "seed", "kind", "session", "B", "draw"]).merge(
        ptq, on=["stratum", "base", "seed", "kind", "session", "B", "draw"])
    t["d_qat_vs_fp0"] = t.r2 - t.r2_fp0
    t["d_ptq_vs_fp0"] = t.r2_ptq - t.r2_fp0
    out = t.groupby(["base", "kind", "stratum", "B"]).agg(
        fp=("r2_fp0", "mean"), ptq_tern=("r2_ptq", "mean"), qat_tern=("r2", "mean"),
        d_ptq=("d_ptq_vs_fp0", "mean"), d_qat=("d_qat_vs_fp0", "mean"),
        d_qat_ci=("d_qat_vs_fp0", ci), rel_rms_qat=("rel_rms", "mean")).reset_index()
    out.to_csv(ANA / "a3_qat_ternary.csv", index=False)
    tot = t.groupby(["base", "kind", "B"]).agg(
        fp=("r2_fp0", "mean"), ptq_tern=("r2_ptq", "mean"), qat_tern=("r2", "mean"),
        d_ptq=("d_ptq_vs_fp0", "mean"), d_qat=("d_qat_vs_fp0", "mean"), d_qat_ci=("d_qat_vs_fp0", ci),
        rel_rms_qat=("rel_rms", "mean")).reset_index()
    tot.to_csv(ANA / "a3_qat_ternary_pooled.csv", index=False)
    MD.append("### A'3 ternary: PTQ (no retraining) vs QAT fine-tune on SOURCE (pooled over strata/seeds)\n")
    MD.append(md_table(tot))
    MD.append("by stratum:\n")
    MD.append(md_table(out[out.B.isin([1, 8, 33])]))


def main():
    ANA.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["a1", "a2", "a3"]
    for name, fn in (("a1", a1), ("a2", a2), ("a3", a3), ("a3", a3_qat)):
        if name in todo:
            try:
                fn()
            except Exception as e:                    # keep going; report which part failed
                MD.append(f"(section {name} failed: {e!r})\n")
                print("section", name, "failed:", repr(e))
    (ANA / ("tables_" + "_".join(todo) + ".md")).write_text("\n".join(MD))
    print("wrote", ANA)


if __name__ == "__main__":
    main()
