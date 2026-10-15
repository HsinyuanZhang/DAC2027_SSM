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
FIXQ = P2 / "fix" / "quant"          # A'3 rows after the review fix (per-add accumulators)
SAFE_DR2 = -0.002                    # "safe" = every (variant, kind, B, model group) cell mean dR2 >= this
GROUPS = ("norm", "ac", "cross", "dir", "quad")
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


def a2_ext():
    """Review item 6: ui_reg at B = 33 with the step grid extended to {600, 1000}."""
    fx = P2 / "fix" / "uireg"
    paths = sorted(fx.glob("eval_ext_*.jsonl"))
    if not paths:
        return
    ex = read_jsonl(paths)
    ex.to_csv(ANA / "a2_ext_rows.csv.gz", index=False)
    sel = {st: json.loads((fx / f"selection_ext_{st}.json").read_text())
           for st in ("days", "month") if (fx / f"selection_ext_{st}.json").exists()}
    (ANA / "a2_ext_selection.json").write_text(json.dumps(sel, indent=1))
    old = read_jsonl(sorted((P2 / "uireg").glob("eval_*.jsonl")))
    old = old[(old.variant == "ui_reg") & (old.B == 33)]
    keys = ["stratum", "target", "draw"]
    m = ex[keys + ["r2", "steps", "lam", "lr"]].merge(old[keys + ["r2"]], on=keys, suffixes=("_ext", "_old"))
    m["d"] = m.r2_ext - m.r2_old
    tab = m.groupby("stratum").agg(old=("r2_old", "mean"), ext=("r2_ext", "mean"), d=("d", "mean"),
                                   d_ci=("d", ci), steps=("steps", "first"), lam=("lam", "first"),
                                   lr=("lr", "first"), n=("d", "count")).reset_index()
    tab.to_csv(ANA / "a2_ext_paired.csv", index=False)
    MD.append("### A'2 ext: ui_reg at B=33 with steps grid extended to {10..1000} (reduced lam/lr grid; source-selected)\n")
    MD.append(md_table(tab))
    for st, v in sel.items():
        MD.append(f"selection_ext {st}: `{json.dumps(v)}`\n")


# --------------------------------------------------------------------- A'3 ----

def model_group(stratum):
    """month and year share the SAME calibrator weights (identical checkpoints), so they are
    one model, not two independent ones."""
    return "days" if stratum == "days" else "month_year"


def a3():
    sys.path.insert(0, str(ROOT))
    from scripts.run_pc2_quant import make_configs
    valid = {v: set(make_configs(v)) for v in ("p0", "p4_m8")}
    q = read_jsonl(sorted(FIXQ.glob("rows_[dmy]*.jsonl")))
    q = q.drop_duplicates(["stratum", "variant", "seed", "kind", "session", "B", "draw", "cfg"], keep="last")
    q = q[[c in valid[v] for v, c in zip(q.variant, q.cfg)]].copy()
    q["mgroup"] = q.stratum.map(model_group)
    q["nan_row"] = ~q["finite"].astype(bool)
    q.to_csv(ANA / "a3_rows.csv.gz", index=False)
    order = list(dict.fromkeys(q.cfg))
    agg = dict(r2=("r2", "mean"), r2_fp=("r2_fp", "mean"), d_r2=("d_r2", "mean"),
               d_r2_ci=("d_r2", ci), rms_delta=("rms_delta", "mean"), rel_rms=("rel_rms", "mean"),
               n=("r2", "size"), nonfinite=("nan_row", "sum"), sat_rows=("sat_events", lambda s: int((s > 0).sum())))
    g = q.groupby(["variant", "kind", "B", "cfg"]).agg(**agg).reset_index()
    g.to_csv(ANA / "a3_cells.csv", index=False)
    gm = q.groupby(["variant", "kind", "mgroup", "B", "cfg"]).agg(**agg).reset_index()
    gm.to_csv(ANA / "a3_cells_by_modelgroup.csv", index=False)
    gs = q.groupby(["variant", "kind", "stratum", "B", "cfg"]).agg(
        d_r2=("d_r2", "mean"), rel_rms=("rel_rms", "mean")).reset_index()
    gs.to_csv(ANA / "a3_cells_by_stratum.csv", index=False)
    # ---- minimal safe accumulator width (per group and for all data keys) -------------------
    rows = []
    for grp in ("all",) + GROUPS:
        pref = "acc{b}" if grp == "all" else "acc{b}_" + grp
        for b in sorted({int(c[3:].split("_")[0]) for c in q.cfg if c.startswith("acc") and
                         (c == f"acc{c[3:].split('_')[0]}" if grp == "all" else c.endswith("_" + grp))}):
            sub = gm[gm.cfg == pref.format(b=b)]
            if not len(sub):
                continue
            w = sub.loc[sub.d_r2.idxmin()]
            rows.append({"group": grp, "bits": b, "worst_dR2": w.d_r2, "worst_cell": f"{w.variant}/{w.kind}/{w.mgroup}/B{w.B}",
                         "max_rel_rms": sub.rel_rms.max(), "nan_rows": int(sub.nonfinite.sum()),
                         "sat_rows": int(sub.sat_rows.sum()), "n_cells": len(sub),
                         "safe": bool(w.d_r2 >= SAFE_DR2 and sub.nonfinite.sum() == 0)})
    sw = pd.DataFrame(rows)
    sw.to_csv(ANA / "a3_safe_width.csv", index=False)
    MD.append(f"### A'3 minimal safe accumulator width (per-add model, counters exact). safe = every (variant, kind, "
              f"model group, B in 1/8/33) cell-mean dR2 >= {SAFE_DR2} and no non-finite row. sat_rows = rows with >= 1 saturating add "
              f"(range headroom, independent of the width)\n")
    MD.append(md_table(sw, 4))
    for variant in q.variant.unique():
        for kind in ("val", "target"):
            sub = g[(g.variant == variant) & (g.kind == kind)]
            if not len(sub):
                continue
            t = pd.DataFrame({"cfg": order})
            for b in (1, 8, 33):
                s_ = sub[sub.B == b].set_index("cfg")
                t[f"dR2 B={b}"] = [f"{s_.d_r2[c]:+.4f}±{s_.d_r2_ci[c]:.4f}" if c in s_.index else "-" for c in order]
                t[f"relRMS B={b}"] = [f"{100 * s_.rel_rms[c]:.2f}%" if c in s_.index else "-" for c in order]
            t["NaN rows (B=1/8/33)"] = ["/".join(str(int(sub[(sub.B == b) & (sub.cfg == c)].nonfinite.sum()))
                                                for b in (1, 8, 33)) for c in order]
            fp = sub[(sub.cfg == "fp")].set_index("B").r2
            MD.append(f"### A'3 {variant}, {kind} sessions, pooled over strata (FP R2: " +
                      ", ".join(f"B={b}: {fp[b]:.3f}" for b in fp.index) +
                      "); dR2 = quantised - FP (mean±CI over rows); relRMS = RMS(pred_q - pred_FP)/std(pred_FP). "
                      "month and year share one calibrator: see a3_cells_by_modelgroup.csv\n")
            MD.append(md_table(t))
    a3_summary(q)


def _trip(sub, col="d_r2", nd=4):
    """'x/y/z' for B = 1/8/33 from a cell table (rows: one per B)."""
    sub = sub.set_index("B")
    return "/".join(f"{sub[col][b]:+.{nd}f}" if b in sub.index and sub[col][b] == sub[col][b] else "nan"
                    for b in (1, 8, 33))


def a3_summary(q=None):
    """Compact corrected-A'3 tables for the README (a3_readme_tables.md)."""
    if q is None:
        q = pd.read_csv(ANA / "a3_rows.csv.gz")
    gm = pd.read_csv(ANA / "a3_cells_by_modelgroup.csv")
    cfgs = [c for c in dict.fromkeys(q.cfg) if c != "fp"]
    out = []

    def table(names, title):
        t = []
        for c in names:
            r = {"cfg": c}
            for grp in ("days", "month_year"):
                for v in ("p0", "p4_m8"):
                    sub = gm[(gm.cfg == c) & (gm.mgroup == grp) & (gm.variant == v) & (gm.kind == "target")]
                    r[f"{grp} {v[:2]} tgt dR2 B1/8/33"] = _trip(sub) if len(sub) else "-"
            allc = gm[gm.cfg == c]
            r["worst cell dR2"] = f"{allc.d_r2.min():+.4f}" if len(allc) else "-"
            r["NaN rows"] = f"{int(allc.nonfinite.sum())}/{int(allc.n.sum())}"
            r["sat rows"] = int(allc.sat_rows.sum())
            t.append(r)
        out.append(f"#### {title}\n")
        out.append(md_table(pd.DataFrame(t)))

    table([c for c in cfgs if c.startswith("acc") and c.count("_") == 0], "per-add, all data keys (counters exact)")
    table([c for c in cfgs if c.startswith("acc") and c.endswith("_bs")], "per-add, B-adaptive exponent (final B known in advance)")
    table([c for c in cfgs if c.endswith("_2lvl")], "two-level design (wide per-trial staging, rounded once per trial)")
    for grp in GROUPS:
        table([c for c in cfgs if c.startswith("acc") and c.endswith("_" + grp)], f"per-add, group {grp} only")
    table([c for c in cfgs if not c.startswith("acc")], "non-accumulator configs (unchanged by the fix; recomputed)")
    (ANA / "a3_readme_tables.md").write_text("\n".join(out))
    MD.append("### A'3 corrected compact tables (by model group; month and year = one calibrator)\n")
    MD.extend(out)
    # old (superseded) vs new FP: numeric determinism
    return out


def a3_qat():
    paths = sorted((P2 / "quant").glob("rows_qat_*.jsonl"))
    if not paths or not len(list(FIXQ.glob("rows_[dmy]*.jsonl"))):
        return
    qa = read_jsonl(paths).drop_duplicates(["stratum", "variant", "seed", "kind", "session", "B", "draw", "cfg"])
    qa["base"] = qa.variant.str.replace("_qat", "", regex=False)
    main_rows = read_jsonl(sorted(FIXQ.glob("rows_[dmy]*.jsonl")))
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
    t["mgroup"] = t.stratum.map(model_group)
    # QAT early-stopped on the SAME source val session as the kind=val rows -> val rows are optimistic
    t["val_is_selection_set"] = t.kind == "val"
    spec = dict(fp=("r2_fp0", "mean"), ptq_tern=("r2_ptq", "mean"), qat_tern=("r2", "mean"),
                d_ptq=("d_ptq_vs_fp0", "mean"), d_qat=("d_qat_vs_fp0", "mean"),
                d_qat_ci=("d_qat_vs_fp0", ci), rel_rms_qat=("rel_rms", "mean"))
    out = t.groupby(["base", "kind", "stratum", "B"]).agg(**spec).reset_index()
    out["val_is_selection_set"] = out.kind == "val"
    out.to_csv(ANA / "a3_qat_ternary.csv", index=False)
    tot = t.groupby(["base", "kind", "mgroup", "B"]).agg(**spec).reset_index()
    tot["val_is_selection_set"] = tot.kind == "val"
    tot.to_csv(ANA / "a3_qat_ternary_pooled.csv", index=False)
    MD.append("### A'3 ternary: PTQ (no retraining) vs QAT fine-tune on SOURCE, by model group "
              "(days; month_year = month and year, ONE calibrator, not independent). "
              "kind=val rows are OPTIMISTIC for QAT: QAT early-stopped on that same val session.\n")
    MD.append(md_table(tot))
    MD.append("by stratum:\n")
    MD.append(md_table(out[out.B.isin([1, 8, 33])]))


def main():
    ANA.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["a1", "a2", "a3"]
    for name, fn in (("a1", a1), ("a2", a2), ("a2", a2_ext), ("a3", a3), ("a3", a3_qat)):
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
