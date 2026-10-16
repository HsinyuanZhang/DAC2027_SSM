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
_T_NEURAL = "B trials: neural only (no labels)"
_T_LABEL = "B trials: neural + reach-direction labels"
_T_ALL_N = "ALL target usable trials: neural only (incl. query bins)"
_T_ALL_L = "ALL target usable trials: neural + reach-direction labels (incl. query trials)"
_ST = "FA + Procrustes on stable channels"
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
    # Stabilizer: closest-to-published input = electrode-pooled fixed channels (sorted spikes summed per electrode, a dataset-imposed
    # approximation of the papers' threshold-crossing channels); sorted-unit variant needs an ad-hoc unit matching
    "stab_pool_a": (f"Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): {_ST}, B-trial data", "published", _T_NEURAL, "no", True, True),
    "stab_pool_a_noalign": ("Stabilizer control (pooled): target FA, no rotation", "published-control", _T_NEURAL, "no", True, True),
    "stab_pool_b": (f"Stabilizer (b), pooled: {_ST}, all target-session trials", "published", _T_ALL_N, "no", False, False),
    "stab_a": (f"Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: {_ST}, B-trial data", "published-variant",
               _T_NEURAL, "no", True, True),
    "stab_a_noalign": ("Stabilizer control (sorted units): target FA, no rotation", "published-control", _T_NEURAL, "no", True, True),
    "stab_b": (f"Stabilizer (b), sorted units: {_ST}, all target-session trials", "published-variant", _T_ALL_N, "no", False, False),
    # Gallego-style = sqrt counts + plain (unregularised) CCA, the faithful row; ridge / raw rows are our variants
    "cca_pool_sqrt_a": ("Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input", "published", _T_LABEL, "no", True, True),
    "cca_pool_sqrt_a_ridge": ("PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected]", "published-variant",
                              _T_LABEL, "no", True, True),
    "cca_sqrt_a": ("Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input", "published", _T_LABEL, "no", True, True),
    "cca_sqrt_a_ridge": ("PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected]", "published-variant",
                         _T_LABEL, "no", True, True),
    "cca_a": ("PCA+CCA (a), sorted units: raw counts, plain CCA [our variant]", "published-variant", _T_LABEL, "no", True, True),
    "cca_a_ridge": ("PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected]", "published-variant",
                    _T_LABEL, "no", True, True),
    "cca_pool_sqrt_b": ("Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials", "published", _T_ALL_L, "no", False, False),
    "cca_pool_sqrt_b_ridge": ("PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials", "published-variant",
                              _T_ALL_L, "no", False, False),
    "cca_sqrt_b": ("Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials", "published", _T_ALL_L, "no", False, False),
    "cca_sqrt_b_ridge": ("PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials", "published-variant",
                         _T_ALL_L, "no", False, False),
    "cca_b": ("PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials", "published-variant",
              _T_ALL_L, "no", False, False),
    "cca_b_ridge": ("PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials", "published-variant",
                    _T_ALL_L, "no", False, False),
    "apst_full_last": ("APST-FULL (last segment, source-only rule; single training seed 42)", "published",
                       _T_LABEL + " (MOVE-T4 profile)", "no", True, True),
    "apst_act_last": ("APST ACT-only (last segment, source-only rule; seed 42)", "published", _T_NEURAL, "no", True, True),
    "apst_full_devsel": ("APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic)", "published",
                         _T_LABEL + " (MOVE-T4 profile)", "no", True, True),
    "apst_act_devsel": ("APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic)", "published",
                        _T_NEURAL, "no", True, True),
    "apst_full_sel0307": ("APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only)", "published",
                          _T_LABEL + " (MOVE-T4 profile)", "no", True, True),
    "nomad_pooled": ("NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds)", "published",
                     "ALL target-session neural data, unlabeled", "YES (alignment net, target gradients)", False, False),
    "nomad_units_elec": ("NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds)", "published-variant",
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
    for f in sorted(glob.glob(str(BL / "eval/nomad2_*.jsonl"))):          # reviewed pipeline (electrode-consistent columns)
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            rows.append({"stratum": r["stratum"], "method": r["method"], "target": r["target"], "B": r["B"],
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
        d["diff"] = d.r2 - d.r2_p0                        # baseline MINUS P0 (positive = baseline beats P0)
        for (s, B), g in d.groupby(["stratum", "B"]):
            pd_ = g.groupby("draw")["diff"].mean()
            n = len(pd_)
            sd = pd_.std(ddof=1) if n > 1 else float("nan")
            ci = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
            out.append({"method": m, "stratum": s, "B": B, "diff_baseline_minus_p0": pd_.mean(), "ci95": ci, "n_draws": n})
    return pd.DataFrame(out)


def per_target_cells(df, col="r2"):
    """Mean over draws (and seeds) per (method, stratum, target, B) and the paired difference baseline - P0."""
    d = df[df[col].notna()]
    cells = d.groupby(["method", "stratum", "target", "B"])[col].agg(["mean", "std", "count"]).reset_index()
    cells = cells.rename(columns={"mean": "r2_mean", "std": "r2_sd_over_rows", "count": "n_rows"})
    base = d[d.method == "p0"][["stratum", "target", "B", "draw", col]].rename(columns={col: "r2_p0"})
    m = d.merge(base, on=["stratum", "target", "B", "draw"])
    m["diff"] = m[col] - m["r2_p0"]
    dd = m.groupby(["method", "stratum", "target", "B", "draw"])["diff"].mean().reset_index()
    pt = dd.groupby(["method", "stratum", "target", "B"])["diff"].agg(["mean", "std", "count"]).reset_index()
    pt = pt.rename(columns={"mean": "diff_baseline_minus_p0", "std": "diff_sd_over_draws", "count": "n_draws"})
    pt["ci95_draws"] = [sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
                        for n, sd in zip(pt.n_draws, pt.diff_sd_over_draws)]
    return cells.merge(pt, on=["method", "stratum", "target", "B"], how="left")


def beats_counts(pt):
    """On how many of the target sessions does a method have a positive mean paired difference to P0?"""
    g = pt[pt.method != "p0"].dropna(subset=["diff_baseline_minus_p0"]).copy()
    g["pos"] = g.diff_baseline_minus_p0 > 0
    g["pos_sig"] = (g.diff_baseline_minus_p0 - g.ci95_draws) > 0
    out = g.groupby(["method", "stratum", "B"]).agg(n_targets=("target", "nunique"), n_beats_p0=("pos", "sum"),
                                                    n_beats_p0_ci_excludes_0=("pos_sig", "sum")).reset_index()
    return out


def fmt(r):
    return "n/a" if r is None else (f"{r['mean']:.3f}±{r['ci95']:.3f}" if np.isfinite(r["ci95"]) else f"{r['mean']:.3f}")


def selection_table():
    """Source-only hyper-parameter selection (pseudo-targets = the other source sessions)."""
    rows = []
    for st in STRATA:
        f = BL / "select" / f"select_{st}.jsonl"
        if not f.exists():
            continue
        agg = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            agg.setdefault((r["key"], r["stage"], json.dumps(r["cfg"], sort_keys=True)), []).append(r["r2"])
        for (k, stage, c), v in agg.items():
            rows.append({"stratum": st, "family": k, "stage": stage, "cfg": c,
                         "mean_pseudo_target_r2": float(np.nanmean(v)), "n_evals": len(v)})
    out = pd.DataFrame(rows)
    sel = []
    for st in STRATA:
        f = BL / "select" / f"selected_{st}.json"
        if f.exists():
            for k, v in json.loads(f.read_text()).items():
                sel.append({"stratum": st, "family": k, "selected_cfg": json.dumps(v["cfg"], sort_keys=True),
                            "mean_pseudo_target_r2": v["mean_pseudo_r2"]})
    return out, pd.DataFrame(sel)


def bridge_table(prefix="sanity_v3"):
    """Bridge ceiling: baseline-day session through each read-in path for the configurations used in the table."""
    rows = []
    for st in STRATA:
        f = BL / "logs" / f"{prefix}_{st}.log"
        if not f.exists():
            continue
        txt = f.read_text()
        js = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        for r in js["rows"]:
            rows.append({"stratum": st, "ref_session": js["ref"], "r2_source_table": js["r2_source_table"],
                         "method": r["method"], "cfg": json.dumps(r["cfg"], sort_keys=True),
                         "d_eff": r["info"].get("d_eff"), "r2_bridge_ceiling": r["r2_bridge_ceiling"],
                         "gap_to_source_table": js["r2_source_table"] - r["r2_bridge_ceiling"]})
    return pd.DataFrame(rows)


def nomad_table():
    rows = []
    for st in STRATA:
        for f in sorted(glob.glob(str(BL / "eval" / f"nomad2_*_{st}.jsonl"))):
            d = pd.DataFrame([json.loads(l) for l in Path(f).read_text().splitlines()])
            rows.append(d.groupby(["method", "stratum", "target", "seed"], as_index=False).agg(
                r2_mean_over_draws_B=("r2", "mean"), r2_native_mean=("r2_native", "mean"), epochs_align=("epochs_align", "first")))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ctl = []
    for f in sorted(glob.glob(str(BL / "nomad" / "controls_*.json"))):
        c = json.loads(Path(f).read_text())
        variant = Path(f).stem.split("_", 1)[1].rsplit("_", 1)[0]
        for r in c["runs"]:
            ctl.append({"method": f"nomad_{variant}", "stratum": Path(f).stem.rsplit("_", 1)[1], "target": c["session"] +
                        (" (CONTROL: baseline day, active columns shuffled)" if r["shuffled"] else " (control: baseline day, unshuffled)"),
                        "seed": int(r["file"].rsplit("_s", 1)[1].split(".")[0]), "r2_mean_over_draws_B": r["r2_all_usable"],
                        "epochs_align": r["epochs_align"]})
    return pd.concat([out, pd.DataFrame(ctl)], ignore_index=True) if ctl else out


def v1_cells():
    """Cells of the pre-review read-in rows (latent dim fixed to K=10, sorted units only); superseded, kept for the before/after table."""
    rows = []
    for f in sorted(glob.glob(str(BL / "v1_pre_review/eval/eval_*.jsonl"))):
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            rows.append({"stratum": r["stratum"], "method": r["method"] + "_v1", "target": r["target"], "B": r["B"],
                         "draw": r["draw"], "r2": r["r2"]})
    return cell_stats(pd.DataFrame(rows)) if rows else pd.DataFrame()


def nomad_legacy_table():
    """First (invalid) NoMAD comparison: sorted units in their own column order (kept for transparency)."""
    rows = []
    for st in STRATA:
        f = BL / "v1_pre_review" / "eval" / f"nomad_{st}.jsonl"
        if f.exists():
            d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
            rows.append(d.groupby(["stratum", "target", "seed"], as_index=False).agg(
                r2_mean_over_draws_B=("r2", "mean"), r2_native_mean=("r2_native", "mean")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


HEAD_BS = (1, 8, 33)
HEAD_ROWS = ["p_prior", "p0", "R1_stream", "R3", "stab_pool_a", "stab_a", "cca_pool_sqrt_a", "cca_sqrt_a",
             "cca_pool_sqrt_a_ridge", "cca_sqrt_a_ridge", "cca_a_ridge", "apst_full_last", "apst_full_devsel", "apst_full_sel0307",
             "stab_pool_b", "cca_pool_sqrt_b_ridge", "cca_b_ridge", "nomad_pooled", "nomad_units_elec", "ui_reg", "ref_p2"]


def headline(cells):
    lines = ["Headline: query R2 (physical units, our query protocol), mean ± 95% CI over 8 draws (draw variance only).", ""]
    for st in STRATA:
        lines.append(f"### {st}\n")
        lines.append("| method | H4 | " + " | ".join(f"B={b}" for b in HEAD_BS) + " |")
        lines.append("|---|---|" + "---:|" * len(HEAD_BS))
        for m in HEAD_ROWS:
            sub = cells[(cells.method == m) & (cells.stratum == st)]
            if sub.empty:
                continue
            vals = []
            for b in HEAD_BS:
                r = sub[sub.B == b]
                vals.append("n/a" if r.empty else fmt(r.iloc[0]) + (f" (n={int(r.iloc[0].n_draws)})" if r.iloc[0].n_draws < 8 else ""))
            h4 = "yes" if REG[m][4] else "NO (violates H4)"
            lines.append(f"| {REG[m][0]} | {h4} | " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines)


def render_per_target(pt, order, bs=HEAD_BS):
    lines = []
    for st in STRATA:
        for tgt in sorted(pt[pt.stratum == st].target.unique()):
            lines.append(f"### {st} / {tgt}\n")
            lines.append("| method | " + " | ".join(f"B={b}: R2 [baseline-P0]" for b in bs) + " |")
            lines.append("|---|" + "---:|" * len(bs))
            for m in order:
                sub = pt[(pt.method == m) & (pt.target == tgt)]
                if sub.empty:
                    continue
                vals = []
                for b in bs:
                    r = sub[sub.B == b]
                    if r.empty:
                        vals.append("n/a")
                    else:
                        r = r.iloc[0]
                        dv = r.diff_baseline_minus_p0
                        vals.append(f"{r.r2_mean:.3f}" + ("" if m == "p0" or not np.isfinite(dv) else f" [{dv:+.3f}]"))
                lines.append(f"| {REG[m][0]} | " + " | ".join(vals) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    nomad_table().to_csv(OUTD / "nomad_per_target_seed.csv", index=False)
    nomad_legacy_table().to_csv(OUTD / "nomad_legacy_invalid_column_order.csv", index=False)
    sel_all, sel_best = selection_table()
    sel_all.to_csv(OUTD / "selection_readin.csv", index=False)
    sel_best.to_csv(OUTD / "selected_configs.csv", index=False)
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
    v1_cells().to_csv(OUTD / "cells_v1_pre_review_superseded.csv", index=False)
    ptc = per_target_cells(df)
    ptc.to_csv(OUTD / "cells_per_target.csv", index=False)
    beats_counts(ptc).to_csv(OUTD / "beats_p0_counts_per_target.csv", index=False)
    (OUTD / "main_table_per_target.md").write_text(render_per_target(ptc, [m for m in ORDER if not m.endswith("native_eval")]))
    (OUTD / "headline_table.md").write_text(headline(cells))
    (OUTD / "headline_per_target.md").write_text(render_per_target(ptc, HEAD_ROWS))
    n_cells = int(pv.dropna(subset=["diff_baseline_minus_p0"]).shape[0])
    (OUTD / "stats_notes.json").write_text(json.dumps({
        "n_paired_cells_vs_p0_(method,stratum,B)": n_cells,
        "n_cells_with_ci": int(cells.ci95.notna().sum()), "n_methods": int(cells.method.nunique()),
        "n_target_sessions": int(df[["stratum", "target"]].drop_duplicates().shape[0]),
        "targets_per_stratum": df.groupby("stratum").target.nunique().to_dict()}, indent=1))
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
    ptn = per_target_cells(dn.assign(method=dn.method.replace({"p0_native_eval": "p0", "p_prior_native_eval": "p_prior"})))
    ptn.to_csv(OUTD / "cells_per_target_native.csv", index=False)
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
