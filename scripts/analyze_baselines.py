#!/usr/bin/env python
"""Aggregate the published-method baseline comparison (DAC2027, 000688 sub-M).

Reads
  results/baselines_20261016/eval/*.jsonl        (this work: Stabilizer, CCA/PCA, APST, optional NoMAD)
  results/population_calib_*                     (P1 / P2 / P3 rows, via scripts/analyze_pc3_fix helpers)
Writes results/analysis/baselines_20261016/{rows_long.csv, cells.csv, paired_vs_p0.csv, method_registry.csv,
main_table.md}.

CI: for each (method, stratum, B) the per-draw value is the mean over the stratum's targets (and over seeds
for the P-rows, which were seed-averaged per draw upstream); the interval is mean +- t(0.975, n-1) * sd / sqrt(n)
over the n = 8 draws.  It contains draw variance only (no target / trunk / source variance); days has 2 targets,
month and year 1 each.  Cells with fewer than 8 available draws report n.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_pc3_fix as A                                          # noqa: E402

BL = ROOT / "results/baselines_20261016"
OUTD = ROOT / "results/analysis/baselines_20261016"
BS = (1, 4, 8, 16, 33)
STRATA = ("days", "month", "year")

# method -> (label, family, target data used, target backprop, H4-compliant, deployable)
REG = {
    "p_prior": ("P-prior (electrode-aligned source prior + B-trial input z-score)", "ours",
                "B trials: neural only (input mean/std)", "no", True, True),
    "p0": ("P0 (fixed-weight per-unit encoder, level 1)", "ours",
           "B trials: neural + behaviour labels", "no", True, True),
    "R1_stream": ("P0 + R1-stream (readout ridge, level 2)", "ours",
                  "B trials: neural + behaviour labels", "no", True, True),
    "R3": ("P0 + R3 (F7-J local update + readout ridge, optional level 3)", "ours",
           "B trials: neural + behaviour labels (forward replay)", "no (forward-only replay)", True, True),
    "p_prior_native_eval": ("P-prior (re-evaluated, native-query subset)", "ours",
                            "B trials: neural only (input mean/std)", "no", True, True),
    "p0_native_eval": ("P0 (re-evaluated, native-query subset)", "ours",
                       "B trials: neural + behaviour labels", "no", True, True),
    "stab_a": ("Stabilizer (a): FA + Procrustes on stable channels, B-trial data", "published",
               "B trials: neural only (no labels)", "no", True, True),
    "stab_a_noalign": ("Stabilizer control: target FA, no rotation", "published-control",
                       "B trials: neural only (no labels)", "no", True, True),
    "stab_b": ("Stabilizer (b): FA + Procrustes, all target-session trials", "published",
               "ALL target usable trials: neural only (incl. query bins)", "no", False, False),
    "cca_sqrt_a": ("Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data", "published",
                   "B trials: neural + reach-direction labels", "no", True, True),
    "cca_sqrt_a_ridge": ("PCA+CCA (a), sqrt counts, ridge CCA (source-selected)", "published-variant",
                         "B trials: neural + reach-direction labels", "no", True, True),
    "cca_a": ("PCA+CCA (a), raw counts, plain CCA, B-trial data", "published-variant",
              "B trials: neural + reach-direction labels", "no", True, True),
    "cca_a_ridge": ("PCA+CCA (a), raw counts, ridge CCA (source-selected)", "published-variant",
                    "B trials: neural + reach-direction labels", "no", True, True),
    "cca_sqrt_b": ("Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials", "published",
                   "ALL target usable trials: neural + reach-direction labels (incl. query trials)", "no", False, False),
    "cca_sqrt_b_ridge": ("PCA+CCA (b), sqrt counts, ridge CCA, all target trials", "published-variant",
                         "ALL target usable trials: neural + reach-direction labels (incl. query trials)", "no", False, False),
    "cca_b": ("PCA+CCA (b), raw counts, plain CCA, all target trials", "published-variant",
              "ALL target usable trials: neural + reach-direction labels (incl. query trials)", "no", False, False),
    "cca_b_ridge": ("PCA+CCA (b), raw counts, ridge CCA, all target trials", "published-variant",
                    "ALL target usable trials: neural + reach-direction labels (incl. query trials)", "no", False, False),
    "apst_full_last": ("APST-FULL (last segment, source-only rule)", "published",
                       "B trials: neural + reach-direction labels (MOVE-T4 profile)", "no", True, True),
    "apst_act_last": ("APST ACT-only (last segment, source-only rule)", "published",
                      "B trials: neural only (no labels)", "no", True, True),
    "apst_full_devsel": ("APST-FULL (authors' dev-selected segment_09; dev = our days targets, so optimistic)", "published",
                         "B trials: neural + reach-direction labels (MOVE-T4 profile)", "no", True, True),
    "apst_act_devsel": ("APST ACT-only (authors' dev-selected segment_08; dev = our days targets, so optimistic)", "published",
                        "B trials: neural only (no labels)", "no", True, True),
    "apst_full_sel0307": ("APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only)", "published",
                          "B trials: neural + reach-direction labels (MOVE-T4 profile)", "no", True, True),
    "nomad": ("NoMAD (LFADS + alignment network; mean over targets and 3 seeds)", "published",
              "ALL target-session neural data, unlabeled", "YES (alignment net, target gradients)", False, False),
    "ui_reg": ("ui_reg: L2-to-prior table fine-tune through trunk (from prior)", "reference",
               "B trials: neural + behaviour labels", "YES (through frozen trunk)", True, False),
    "ref_p2": ("P0 + ui_reg (P2 rule)", "reference",
               "B trials: neural + behaviour labels", "YES (through frozen trunk)", True, False),
    "ref_like": ("P0 + ui_reg (like-for-like: month oosc / year oos)", "reference",
                 "B trials: neural + behaviour labels", "YES (through frozen trunk)", True, False),
}
ORDER = list(REG)


def load_ours():
    ev = A.load_candidates()
    cand = A.candidate_rows(ev)
    keep = cand.apply(lambda r: A.HEADLINE[r.stratum] == r.set and r.variant == "base", axis=1)
    cand = cand[keep & cand.method.isin(["R1_stream", "R3"])][["stratum", "method", "target", "B", "draw", "r2"]]
    refs = A.reference_rows()
    like = {"month": "ref_oosc", "year": "ref_oos"}
    parts = [cand]
    for m in ("p_prior", "p0", "ui_reg", "ref_p2"):
        parts.append(refs[refs.method == m])
    for s, tag in like.items():
        parts.append(refs[(refs.stratum == s) & (refs.method == tag)].assign(method="ref_like"))
    return pd.concat(parts, ignore_index=True)[["stratum", "method", "target", "B", "draw", "r2"]]


def load_baselines():
    rows = []
    for f in sorted(glob.glob(str(BL / "eval/eval_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            rows.append({"stratum": r["stratum"], "method": r["method"], "target": r["target"], "B": r["B"],
                         "draw": r["draw"], "r2": r["r2"], "r2_native": r.get("r2_native"), "status": r.get("status", "OK")})
    for f in sorted(glob.glob(str(BL / "eval/apst_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            rows.append({"stratum": r["stratum"], "method": r["method"], "target": r["target"], "B": r["B"],
                         "draw": r["draw"], "r2": r["r2"], "r2_native": r.get("r2_native"), "status": r.get("status", "OK")})
    for f in sorted(glob.glob(str(BL / "eval/nomad_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            rows.append({"stratum": r["stratum"], "method": "nomad", "target": r["target"], "B": r["B"],
                         "draw": r["draw"], "r2": r["r2"], "r2_native": r.get("r2_native"), "status": r.get("status", "OK")})
    for f in sorted(glob.glob(str(BL / "eval/native_p_*.jsonl"))):          # P-prior / P0 re-evaluated with r2_native
        d = pd.DataFrame([json.loads(l) for l in Path(f).read_text().splitlines()])
        d = d.groupby(["stratum", "method", "target", "B", "draw"], as_index=False)[["r2", "r2_native"]].mean()
        d["status"] = "OK"
        d["method"] = d["method"].replace({"p0": "p0_native_eval", "p_prior": "p_prior_native_eval"})
        rows.extend(d.to_dict("records"))
    return pd.DataFrame(rows)


def cell_stats(df, col="r2"):
    out = []
    for (m, s, B), g in df.groupby(["method", "stratum", "B"]):
        g = g[g[col].notna()]
        per_draw = g.groupby("draw")[col].mean()
        n = len(per_draw)
        if n == 0:
            continue
        sd = per_draw.std(ddof=1) if n > 1 else float("nan")
        ci = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
        out.append({"method": m, "stratum": s, "B": B, "mean": per_draw.mean(), "ci95": ci, "n_draws": n,
                    "n_rows": len(g)})
    return pd.DataFrame(out)


def paired_vs_p0(df):
    base = df[df.method == "p0"][["stratum", "target", "B", "draw", "r2"]].rename(columns={"r2": "r2_p0"})
    out = []
    for m in ORDER:
        if m == "p0":
            continue
        d = df[df.method == m][["stratum", "target", "B", "draw", "r2"]].merge(base, on=["stratum", "target", "B", "draw"])
        d = d[d.r2.notna()]
        d["diff"] = d.r2 - d.r2_p0
        for (s, B), g in d.groupby(["stratum", "B"]):
            pd_ = g.groupby("draw")["diff"].mean()
            n = len(pd_)
            sd = pd_.std(ddof=1) if n > 1 else float("nan")
            ci = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
            out.append({"method": m, "stratum": s, "B": B, "mean_diff_vs_p0": pd_.mean(), "ci95": ci, "n_draws": n})
    return pd.DataFrame(out)


def fmt(r):
    return "n/a" if r is None else (f"{r['mean']:.3f}±{r['ci95']:.3f}" if np.isfinite(r["ci95"]) else f"{r['mean']:.3f}")


def selection_table():
    """Source-only hyper-parameter selection (pseudo-targets = the other source sessions)."""
    rows = []
    for s in STRATA:
        f = BL / "select" / f"select_{s}.jsonl"
        if not f.exists():
            continue
        agg = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key = (r["method"], json.dumps({"transform": "raw", **r["cfg"]}, sort_keys=True))
            agg.setdefault(key, []).append(r["r2"])
        for (m, c), v in agg.items():
            rows.append({"stratum": s, "method": m, "cfg": c, "mean_pseudo_target_r2": float(np.nanmean(v)), "n_evals": len(v)})
    return pd.DataFrame(rows)


def bridge_table():
    """Same-day bridge ceiling: baseline-day session through each read-in path (sanity logs)."""
    rows = []
    for s in STRATA:
        f = BL / "logs" / f"sanity_{s}.log"
        if not f.exists():
            continue
        txt = f.read_text()
        js = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        rows.append({"stratum": s, "ref_session": js["ref"], "r2_source_table": js["r2_source_table"],
                     "r2_stabilizer_fa_bridge": js["stab_b"], "r2_pca_bridge": js["cca_b"]})
    return pd.DataFrame(rows)


def nomad_table():
    rows = []
    for s in STRATA:
        f = BL / "eval" / f"nomad_{s}.jsonl"
        if not f.exists():
            continue
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
        g = d.groupby(["stratum", "target", "seed"], as_index=False).agg(
            r2_mean_over_draws_B=("r2", "mean"), r2_native_mean=("r2_native", "mean"), epochs_align=("epochs_align", "first"))
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ctl = BL / "nomad/control_ref_day.json"
    if ctl.exists():
        c = json.loads(ctl.read_text())
        out = pd.concat([out, pd.DataFrame([{"stratum": "days", "target": c["session"] + " (control: baseline day through its own alignment net)",
                                             "seed": 0, "r2_mean_over_draws_B": c["r2_physical"], "epochs_align": c["epochs_align"]}])],
                        ignore_index=True)
    return out


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    nomad_table().to_csv(OUTD / "nomad_per_target_seed.csv", index=False)
    selection_table().to_csv(OUTD / "selection_readin.csv", index=False)
    bridge_table().to_csv(OUTD / "bridge_ceiling.csv", index=False)
    ours = load_ours()
    bl = load_baselines()
    df = pd.concat([ours.assign(status="OK"), bl], ignore_index=True)
    df = df[df.B.isin(BS)]
    df.to_csv(OUTD / "rows_long.csv.gz", index=False)
    cells = cell_stats(df)
    cells.to_csv(OUTD / "cells.csv", index=False)
    pv = paired_vs_p0(df)
    pv.to_csv(OUTD / "paired_vs_p0.csv", index=False)
    reg = pd.DataFrame([{"method": k, "label": v[0], "family": v[1], "target_data_used": v[2],
                         "target_backprop": v[3], "h4_compliant": v[4], "deployable": v[5]} for k, v in REG.items()])
    reg.to_csv(OUTD / "method_registry.csv", index=False)
    # availability of unavailable APST cells (rank < 3 etc.)
    unav = bl[bl.status.ne("OK")].groupby(["stratum", "method", "B", "status"]).size().reset_index(name="n_cells")
    unav.to_csv(OUTD / "unavailable_cells.csv", index=False)
    (OUTD / "main_table.md").write_text(render(cells, [m for m in ORDER if not m.endswith("native_eval")]))
    # native-query subset (windows fully inside a trial; APST's own query rule), methods that have r2_native
    dn = df[df.r2_native.notna()].copy()
    dn["r2"] = dn["r2_native"]
    cn = cell_stats(dn)
    cn.to_csv(OUTD / "cells_native.csv", index=False)
    nat_order = [m for m in ORDER if m in set(cn.method)]
    (OUTD / "main_table_native.md").write_text(render(cn, nat_order))
    pn = paired_vs_p0(dn.assign(method=dn.method.replace({"p0_native_eval": "p0", "p_prior_native_eval": "p_prior"})))
    pn.to_csv(OUTD / "paired_vs_p0_native.csv", index=False)
    # reproduction check of the P1 rows by the native re-evaluation
    chk = []
    for f in sorted(glob.glob(str(BL / "eval/native_p_*.jsonl"))):
        d = pd.DataFrame([json.loads(l) for l in Path(f).read_text().splitlines()])
        d = d[d.method == "p0"]
        d["abs_diff"] = (d.r2 - d.stored_p1_r2).abs()
        chk.append({"stratum": d.stratum.iloc[0], "n_rows": len(d), "max_abs_diff_vs_stored_p1": float(d.abs_diff.max()),
                    "mean_abs_diff": float(d.abs_diff.mean())})
    pd.DataFrame(chk).to_csv(OUTD / "native_p_reproduction_check.csv", index=False)
    print((OUTD / "main_table.md").read_text())


def render(cells, order):
    lines = []
    for s in STRATA:
        lines.append(f"### {s}\n")
        lines.append("| method | target data used | target backprop | " + " | ".join(f"B={b}" for b in BS) + " |")
        lines.append("|---|---|---|" + "---:|" * len(BS))
        for m in order:
            sub = cells[(cells.method == m) & (cells.stratum == s)]
            if sub.empty:
                continue
            vals = []
            for b in BS:
                r = sub[sub.B == b]
                if r.empty:
                    vals.append("n/a")
                else:
                    rr = r.iloc[0]
                    cell = fmt(rr)
                    if rr["n_draws"] < 8:
                        cell += f" (n={int(rr['n_draws'])})"
                    vals.append(cell)
            lines.append(f"| {REG[m][0]} | {REG[m][2]} | {REG[m][3]} | " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
