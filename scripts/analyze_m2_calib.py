#!/usr/bin/env python
"""Aggregate the FALCON M2 calibration rows into tables (target-level CIs, paired differences).

Statistical unit = TARGET (5 targets in 3 pools; pool1 = targets 10-20 R1/R2, pool2 = 10-27 R1/R2,
pool3 = 10-28 R1).  Draws (8) and calibrator seeds are averaged inside a target first, so the CI
(Student t, df = n_targets - 1) reflects target-to-target variation only; the noise floor from
calibrator seeds / backbone seeds is reported separately.  Backbone seed 0 is the headline; the
3-seed view averages per-target values over the available backbone seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B_GRID = (1, 2, 4, 8, 16, 33)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}
MAIN = ["p_prior", "p_prior_scale", "p_prior_r1s", "p0", "p0_int8", "p0_r1s"]
ABL = ["p0_swap", "p0_nogain", "p0_nolab", "p0_r1s_nogain", "p0_r1s_nolab"]
REF = ["p_prior_nobias", "p_prior_srcnorm"]


def load_rows(root: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(root / "runs/seed*/pool*/c*/eval_rows.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["variant", "seed", "pool", "cseed", "tag", "target", "B", "draw"], keep="last")
    ui = []
    for f in glob.glob(str(root / "uireg/seed*/pool*/eval_uireg.jsonl")):
        ui += [json.loads(l) for l in open(f)]
    if ui:
        u = pd.DataFrame(ui)
        u["cseed"] = 0
        df = pd.concat([df, u[[c for c in u.columns if c in df.columns]]], ignore_index=True)
    return df


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan")
    return float(x.mean()), float(T975.get(n - 1, 1.96) * x.std(ddof=1) / np.sqrt(n))


def per_target(df, variant, seed=None):
    """[seed, pool, target, B] -> mean R2 over draws and calibrator seeds."""
    d = df[df.variant == variant]
    if seed is not None:
        d = d[d.seed == seed]
    return d.groupby(["seed", "pool", "target", "B"]).r2.mean().reset_index()


def cell_table(df, variants, seeds):
    """variant x B : mean +- CI over targets (per-target values averaged over ``seeds``)."""
    out = {}
    for v in variants:
        pt = per_target(df, v)
        pt = pt[pt.seed.isin(seeds)]
        n_seeds = pt.seed.nunique()
        pt = pt.groupby(["target", "B"]).r2.mean().reset_index()
        for B in B_GRID:
            sub = pt[pt.B == B]
            if len(sub):
                m, c = tci(sub.r2.values)
                out[(v, B)] = (m, c, len(sub), n_seeds)
    return out


def signflip_p(d):
    """Exact two-sided sign-flip test of mean(d) = 0 over the units (n <= 18)."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0 or n > 18:
        return float("nan")
    signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1)
    means = (signs * d).mean(1)
    return float((np.abs(means) >= abs(d.mean()) - 1e-12).mean())


def paired(df, a, b, seeds):
    """a - b per target on the SAME draws (and the same backbone seeds); mean +- CI over targets, #positive,
    per-pool means.  Also the (backbone seed x target) units: #positive and an exact sign-flip p-value
    (units share targets across seeds, so this is a sensitivity view, not an independent test)."""
    cols = ["seed", "pool", "target", "B", "draw"]
    da = df[(df.variant == a) & df.seed.isin(seeds)].groupby(cols).r2.mean().rename("ra")
    db = df[(df.variant == b) & df.seed.isin(seeds)].groupby(cols).r2.mean().rename("rb")
    m = pd.concat([da, db], axis=1, join="inner").reset_index()
    rows = []
    for B in B_GRID:
        mb = m[m.B == B]
        if mb.empty:
            continue
        u = mb.groupby(["seed", "pool", "target"])[["ra", "rb"]].mean()
        du = u.ra - u.rb
        t = mb.groupby(["pool", "target"])[["ra", "rb"]].mean()
        d = t.ra - t.rb
        mean, ci = tci(d.values)
        rows.append({"a": a, "b": b, "B": B, "diff": mean, "ci95": ci, "n_targets": len(d),
                     "n_pos": int((d > 0).sum()), "n_seeds": int(mb.seed.nunique()), "n_units": len(du),
                     "n_pos_units": int((du > 0).sum()), "n_draws_per_cell": float(mb.groupby(
                         ["seed", "target"]).draw.nunique().mean()),
                     "p_signflip_units": signflip_p(du.values),
                     "per_pool": d.groupby(level=0).mean().round(4).to_dict()})
    return pd.DataFrame(rows)


def draw_level_paired(df, a, b, seeds):
    """Draw-level paired SD / CI (draws x targets x seeds, NOT independent: shown only for scale)."""
    out = []
    for B in B_GRID:
        da = df[(df.variant == a) & (df.B == B) & df.seed.isin(seeds)].groupby(
            ["seed", "target", "draw"]).r2.mean()
        db = df[(df.variant == b) & (df.B == B) & df.seed.isin(seeds)].groupby(
            ["seed", "target", "draw"]).r2.mean()
        idx = da.index.intersection(db.index)
        if len(idx):
            d = da.loc[idx] - db.loc[idx]
            out.append({"a": a, "b": b, "B": B, "mean": d.mean(), "sd": d.std(ddof=1),
                        "se_naive": d.std(ddof=1) / np.sqrt(len(d)), "n": len(d)})
    return pd.DataFrame(out)


def noise_floor(df):
    """Seed-to-seed spread of per-target values: calibrator seeds (same backbone) and backbone seeds."""
    res = {}
    for v in ("p0", "p0_r1s"):
        d = df[df.variant == v].groupby(["seed", "cseed", "target", "B"]).r2.mean().reset_index()
        c = d.groupby(["seed", "target", "B"]).r2.agg(["std", "count"]).reset_index()
        c = c[c["count"] >= 2]
        res[f"{v}: calibrator-seed SD (mean over seed,target,B)"] = float(c["std"].mean()) if len(c) else None
        e = d.groupby(["cseed", "target", "B"]).r2.agg(["std", "count"]).reset_index()
        e = e[e["count"] >= 2]
        res[f"{v}: backbone-seed SD (mean over cseed,target,B)"] = float(e["std"].mean()) if len(e) else None
    d = df[df.variant == "p_prior"].groupby(["seed", "target", "B"]).r2.mean().reset_index()
    e = d.groupby(["target", "B"]).r2.agg(["std", "count"]).reset_index()
    e = e[e["count"] >= 2]
    res["p_prior: backbone-seed SD (mean over target,B)"] = float(e["std"].mean()) if len(e) else None
    return res


def round1_rows():
    rows = []
    for f in glob.glob(str(ROOT / "results/labeled_calibration_20261007/d2_*/*/seed*/eval_rows.jsonl")):
        parts = Path(f).parts
        for l in open(f):
            r = json.loads(l)
            rows.append({"pool": parts[-4], "tag": parts[-3], "seed": parts[-2], "variant": r["variant"],
                         "target": r["target"], "B": r["budget"], "r2": r["r2"]})
    return pd.DataFrame(rows)


def df_md(d, index=True):
    d = d.reset_index() if index else d
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def fmt(m, c):
    return f"{m:.3f}" if (c is None or np.isnan(c)) else f"{m:.3f}±{c:.3f}"


def md_cells(cells, variants):
    lines = ["| variant (#backbone seeds) | " + " | ".join(f"B={b}" for b in B_GRID) + " |",
             "|---|" + "---:|" * len(B_GRID)]
    for v in variants:
        if not any((v, B) in cells for B in B_GRID):
            continue
        ns = max(cells[(v, B)][3] for B in B_GRID if (v, B) in cells)
        lines.append(f"| {v} ({ns}) | " + " | ".join(fmt(*cells[(v, B)][:2]) if (v, B) in cells else "-"
                                                    for B in B_GRID) + " |")
    return "\n".join(lines)


def md_paired(p):
    if p.empty:
        return ""
    lines = ["| a − b | B | Δ R² ± CI(targets) | #targets>0 | #(seed,target)>0 | sign-flip p | per-pool Δ |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for _, r in p.iterrows():
        lines.append(f"| {r.a} − {r.b} | {r.B} | {r.n_seeds} | {r['diff']:+.3f}±{r.ci95:.3f} | {r.n_pos}/{r.n_targets} | "
                     f"{r.n_pos_units}/{r.n_units} | {r.p_signflip_units:.3f} | {r.per_pool} |")
    return "\n".join(lines)


def train_and_selection_tables(root: Path):
    tr, sel = [], []
    for f in sorted(glob.glob(str(root / "runs/seed*/pool*/c*/done.json"))):
        d = json.loads(Path(f).read_text())
        parts = Path(f).parts
        tr.append({"seed": parts[-4][4:], "pool": parts[-3], "run": parts[-2], "best_step": d["best_step"],
                   "last_step": d["last_step"], "converged": d["converged"], "doublings": d["doublings"],
                   "source_val_p0": round(d["best_val_r2"], 4), "source_val_prior": round(d["val_prior_r2"], 4),
                   "source_val_gain": round(d["best_val_r2"] - d["val_prior_r2"], 4),
                   "train_seconds": d["seconds"]})
        sf = Path(f).with_name("selection_r1.json")
        if sf.exists():
            for base, per_b in json.loads(sf.read_text()).items():
                for B, r in per_b.items():
                    sel.append({"seed": parts[-4][4:], "pool": parts[-3], "run": parts[-2], "base": base, "B": int(B),
                                "lam": r["lam"], "kind": r["kind"], "burn": r["burn"],
                                "source_r2": round(r["source_r2"], 4), "source_r2_off": round(r["source_r2_off"], 4),
                                "source_gain": round(r["source_r2"] - r["source_r2_off"], 4),
                                "n_episodes": r["n_episodes"]})
    return pd.DataFrame(tr), pd.DataFrame(sel)


def uireg_selection_table(root: Path):
    rows = []
    for f in sorted(glob.glob(str(root / "uireg/seed*/pool*/selection_uireg.json"))):
        parts = Path(f).parts
        for B, r in json.loads(Path(f).read_text()).items():
            rows.append({"seed": parts[-3][4:], "pool": parts[-2], "B": int(B), **r})
    return pd.DataFrame(rows)


def make_figure(cells_by_view, paired_by_view, round1, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cells_by_view)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 8.0), sharex=True)
    axes = np.asarray(axes).reshape(2, n)
    style = {"p_prior": ("tab:gray", "o"), "p0": ("tab:blue", "s"), "p0_r1s": ("tab:green", "^"),
             "ui_reg": ("tab:red", "D"), "p_prior_nobias": ("lightgray", "x")}
    for j, (label, cells) in enumerate(cells_by_view.items()):
        ax = axes[0, j]
        for k, (v, (col, mk)) in enumerate(style.items()):
            pts = [(B, *cells[(v, B)][:2]) for B in B_GRID if (v, B) in cells]
            if not pts:
                continue
            xs = np.array([np.log2(p[0]) for p in pts])
            ax.errorbar(xs + 0.04 * k, [p[1] for p in pts], yerr=[p[2] for p in pts], color=col, marker=mk,
                        ms=4, lw=1.2, capsize=2, label=v)
        if round1 is not None and len(round1):
            for v, col in (("f3", "tab:orange"), ("f7", "tab:purple")):
                if v in round1.index:
                    ax.plot([np.log2(b) for b in B_GRID], [round1.loc[v, b] for b in B_GRID], color=col,
                            ls="--", lw=1, label=f"round-1 {v} (stored)")
        ax.set_title(label + "\nR2, CI over targets (dominated by target-to-target level)", fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        dstyle = {"p0": ("tab:blue", "s"), "p0_r1s": ("tab:green", "^"), "ui_reg": ("tab:red", "D"),
                  "p_prior_scale": ("tab:brown", "v"), "p0_swap": ("tab:purple", "P")}
        pt = paired_by_view[label]
        for k, (v, (col, mk)) in enumerate(dstyle.items()):
            d = pt[(pt.a == v) & (pt.b == "p_prior")].sort_values("B")
            if d.empty:
                continue
            ax.errorbar(np.log2(d.B.values) + 0.04 * k, d["diff"].values, yerr=d.ci95.values, color=col,
                        marker=mk, ms=4, lw=1.2, capsize=2, label=f"{v} - p_prior")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([np.log2(b) for b in B_GRID])
        ax.set_xticklabels(B_GRID)
        ax.set_xlabel("labeled trials B")
        ax.set_title("paired difference vs p_prior (same draws), 95% CI over targets", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("query R2 (physical units)")
    axes[1, 0].set_ylabel("delta R2 vs p_prior")
    axes[0, -1].legend(fontsize=7, loc="lower right")
    axes[1, -1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "m2_r2_vs_B.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/m2_calib_20261016"))
    ap.add_argument("--out", default=str(ROOT / "results/analysis/m2_calib_20261016"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(Path(args.root))
    df.to_csv(out / "all_rows.csv.gz", index=False)
    seeds_all = sorted(df.seed.unique().tolist())
    ui_seeds = sorted(df[df.variant == "ui_reg"].seed.unique().tolist()) if (df.variant == "ui_reg").any() else []
    md = []
    flags = {"rows": int(len(df)), "seeds": seeds_all, "ui_reg_seeds": ui_seeds,
             "h1_all": bool(df.get("h1_trunk_unchanged", pd.Series([True])).fillna(True).all()),
             "h2_all_zero_backprop_candidates": bool(
                 df[df.variant.str.startswith("p")].h2_zero_backprop.all()),
             "h4_all_query_disjoint": bool(df[df.variant.str.startswith("p")].h4_query_disjoint.all()),
             "unit_check_all": bool(df[df.variant.str.startswith("p")].unit_check_passed.all()),
             "h5_bytes_fp32": int(df[df.variant == "p0"].h5_bytes.max()),
             "h5_bytes_int8": int(df[df.variant == "p0_int8"].h5_bytes.max()) if (df.variant == "p0_int8").any() else None,
             "rows_per_variant": df.groupby("variant").size().to_dict()}
    (out / "flags.json").write_text(json.dumps(flags, indent=1, default=str))
    variants = MAIN + ["ui_reg"] + ABL + REF
    cells_by_view, paired_by_view = {}, {}
    for label, seeds in (("backbone seed 0 (headline)", [0]), ("mean over available backbone seeds", seeds_all)):
        cells = cell_table(df, variants, seeds)
        cells_by_view[label] = cells
        md.append(f"### Query R² (physical units), {label}; mean ± 95% CI over targets (t, df = n_targets − 1)\n")
        md.append(md_cells(cells, variants) + "\n")
        pd.DataFrame([{"view": label, "variant": v, "B": B, "mean": m, "ci95": c, "n_targets": n,
                       "n_backbone_seeds": ns} for (v, B), (m, c, n, ns) in cells.items()]).to_csv(
            out / f"cells_{'seed0' if seeds == [0] else 'allseeds'}.csv", index=False)
        pairs = [("p_prior_scale", "p_prior"), ("p0", "p_prior_scale"), ("p0_r1s", "p_prior_scale"), ("p0", "p_prior"), ("p0_r1s", "p_prior"), ("p0_r1s", "p0"), ("p_prior_r1s", "p_prior"),
                 ("p0_int8", "p0"), ("ui_reg", "p_prior"), ("ui_reg", "p0"), ("p0_nogain", "p0"),
                 ("p0_nolab", "p0"), ("p_prior", "p_prior_nobias"), ("p0_swap", "p_prior"), ("p0", "p0_swap"),
                 ("p0_nogain", "p_prior"), ("p0_nolab", "p_prior")]
        ptab = pd.concat([paired(df, a, b, seeds) for a, b in pairs
                          if (df.variant == a).any() and (df.variant == b).any()], ignore_index=True)
        ptab.to_csv(out / f"paired_{'seed0' if seeds == [0] else 'allseeds'}.csv", index=False)
        paired_by_view[label] = ptab
        md.append(f"### Paired differences (per target, same draws), {label}\n")
        md.append(md_paired(ptab) + "\n")
    # draw-level scale
    dl = pd.concat([draw_level_paired(df, a, b, seeds_all) for a, b in (("p0", "p_prior"), ("p0_r1s", "p0"))
                    if (df.variant == a).any()], ignore_index=True)
    dl.to_csv(out / "paired_draw_level.csv", index=False)
    md.append("### Draw-level paired scale (not independent; for scale only)\n")
    md.append(df_md(dl.round(4), index=False) + "\n")
    nf = noise_floor(df)
    (out / "noise_floor.json").write_text(json.dumps(nf, indent=1))
    md.append("### Noise floor (seed-to-seed SD of per-target cell means)\n")
    md.append("\n".join(f"- {k}: {('%.4f' % v) if v is not None else 'n/a'}" for k, v in nf.items()) + "\n")
    # per-pool, per-target B=8 view
    d8 = df[(df.B == 8) & df.seed.isin([0])]
    pv = d8.groupby(["pool", "target", "variant"]).r2.mean().unstack().round(3)
    pv.to_csv(out / "per_target_B8_seed0.csv")
    md.append("### Per target, B=8, backbone seed 0\n")
    md.append(df_md(pv[[c for c in variants if c in pv.columns]]) + "\n")
    trdf, seldf = train_and_selection_tables(Path(args.root))
    trdf.to_csv(out / "train_summary.csv", index=False)
    seldf.to_csv(out / "r1_selection.csv", index=False)
    md.append("### P0 training summary (source simulation; val = last-20% trunk-validation trials of the source sessions)\n")
    md.append(df_md(trdf, index=False) + "\n")
    if len(seldf):
        md.append("### R1-stream rule selected on SOURCE per (run, base, B): fraction of cells where R1 is OFF "
                  "(lambda = none)\n")
        sd_ = seldf.assign(off=seldf.lam.isna()).groupby(["base", "B"]).agg(
            frac_off=("off", "mean"), mean_source_gain=("source_gain", "mean"), n=("off", "size"))
        md.append(df_md(sd_.round(4)) + "\n")
    ut = uireg_selection_table(Path(args.root))
    if len(ut):
        ut.to_csv(out / "uireg_selection.csv", index=False)
        md.append("### ui_reg rule selected on SOURCE (lambda, lr, steps)\n")
        md.append(df_md(ut.round(4), index=False) + "\n")
    # round 1
    r1 = round1_rows()
    if len(r1):
        r1m = r1.groupby(["variant", "B"]).r2.mean().unstack().round(3)
        r1m.to_csv(out / "round1_d2_stored.csv")
        md.append("### Round-1 (2026-10-07) stored D2 rows, mean over targets x seeds "
                  "(first-B trials, bias-free decode; NOT re-evaluated)\n")
        md.append(df_md(r1m) + "\n")
    r1m_fig = r1m if len(r1) else None
    make_figure(cells_by_view, paired_by_view, r1m_fig, out)
    (out / "tables_m2.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
