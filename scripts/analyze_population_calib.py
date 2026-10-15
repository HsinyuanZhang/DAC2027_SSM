#!/usr/bin/env python
"""Aggregate population-calibration rows into csv/json summaries (+ md tables).

CI convention (spec): mean and 95% CI (1.96 * SE) over the pooled
draws x targets x seeds rows of a cell.  Paired differences join rows on
(stratum, target, B, draw) so that every variant sees identical trial subsets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/population_calib_20261015"
ANA = ROOT / "results/analysis/population_calib_20261015"


def load_rows():
    rows = []
    for p in sorted((OUT / "runs").glob("*/*/seed*/eval_rows.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ev = pd.DataFrame(rows)
    shuf = []
    for p in sorted((OUT / "runs").glob("*/*/seed*/eval_rows_shufY.jsonl")):
        shuf += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    orc = []
    for p in sorted((OUT / "runs").glob("*/*/seed*/oracle_rows.jsonl")):
        orc += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ref = []
    for p in sorted((OUT / "reference").glob("*.jsonl")):
        part = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not p.name.startswith("prior_"):
            # p_prior rows of the combined ui run carry a process-wide backward counter
            # (H2 false positive); the clean per-row p_prior rows live in prior_*.jsonl.
            part = [r for r in part if r["variant"] != "p_prior"]
        ref += part
    return ev, pd.DataFrame(shuf), pd.DataFrame(orc), pd.DataFrame(ref)


def ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(n))


def cell_table(df, value="r2", by=("stratum", "pool", "variant", "B")):
    g = df.groupby(list(by))[value]
    out = g.agg(["mean", "count"]).reset_index()
    out["ci95"] = g.apply(ci).values
    return out.rename(columns={"mean": value})


def paired_gain(ev, ref, base="p_prior"):
    """Per-row R2 minus the P-prior R2 of the same (stratum, target, B, draw)."""
    pr = ref[ref.variant == base][["stratum", "target", "B", "draw", "r2"]].rename(
        columns={"r2": "r2_base"})
    m = ev.merge(pr, on=["stratum", "target", "B", "draw"], how="inner")
    m["gain"] = m.r2 - m.r2_base
    return m


def paired_vs(ev, a, b, by=("stratum", "pool", "B")):
    """Mean paired difference variant a - variant b (same target/B/draw/seed)."""
    keys = ["stratum", "pool", "target", "B", "draw", "seed"]
    x = ev[ev.variant == a][keys + ["r2"]].rename(columns={"r2": "r2_a"})
    y = ev[ev.variant == b][keys + ["r2"]].rename(columns={"r2": "r2_b"})
    m = x.merge(y, on=keys)
    m["diff"] = m.r2_a - m.r2_b
    g = m.groupby(list(by))["diff"]
    out = g.agg(["mean", "count"]).reset_index()
    out["ci95"] = g.apply(ci).values
    out["a"], out["b"] = a, b
    return out


def val_table():
    rows = []
    for p in sorted((OUT / "runs").glob("*/*/seed*/done.json")):
        d = json.loads(p.read_text())
        sp = p.parts
        stratum_pool, tag, seed = sp[-4], sp[-3], sp[-2]
        stratum, pool = stratum_pool.split("_")
        rows.append({"stratum": stratum, "pool": pool, "tag": tag, "seed": int(seed[4:]),
                     "best_val_r2": d["best_val_r2"], "val_prior_r2": d["val_prior_r2"],
                     "best_step": d["best_step"], "final_budget": d["final_budget"],
                     "doublings": d["doublings"], "converged": d["converged"],
                     "stopped_early": d["stopped_early"], "val_ids": ",".join(
                         x[-8:] for x in d["val_ids"]), "n_train": len(d["train_ids"])})
    return pd.DataFrame(rows)


def h_summary(ev, ref):
    allr = pd.concat([ev, ref[ref.variant == "p_prior"]], ignore_index=True)
    out = {
        "rows": int(len(allr)),
        "H1_trunk_unchanged_rate": float(allr.h1_trunk_unchanged.mean()),
        "H2_zero_backprop_rate": float(allr.h2_zero_backprop.mean()),
        "H4_query_disjoint_rate": float(allr.h4_query_disjoint.mean()),
        "H5_within_32kb_rate": float(allr.h5_within_32kb.mean()),
        "H5_max_bytes": int(allr.h5_bytes.max()),
        "unit_check_pass_rate": float(allr.unit_check_passed.mean()),
    }
    cand = ev
    out["H3_fixed_ratio_mean_by_variant"] = cand.groupby("variant").h3_fixed_ratio.mean().round(4).to_dict()
    return out


def fmt(m, c):
    return "n/a" if m != m else (f"{m:.3f}" if c != c else f"{m:.3f}±{c:.3f}")


VARIANT_ORDER = ["p_prior", "p0", "p1", "p2", "p3_m4", "p3_m8", "p3_m16",
                 "p4_m4", "p4_m8", "p4_m16", "ui_prior", "ui_elec"]


def selected_variants(sel_path):
    """Selected PMA seed counts per pool (days source-val only), from select.json."""
    if sel_path.exists():
        return json.loads(sel_path.read_text())
    return {}


def md_main(cells, refc, pool, buds=(1, 8, 33), value="r2", strata=("days", "month", "year"),
            variants=None):
    cols = [(st, b) for st in strata for b in buds]
    head = "| variant | " + " | ".join(f"{st} B={b}" for st, b in cols) + " |\n"
    head += "|---|" + "---:|" * len(cols) + "\n"
    lines = []
    sub = cells[(cells.pool == pool)]
    for v in (variants or VARIANT_ORDER):
        parts = []
        for st, b in cols:
            if v in ("p_prior", "ui_prior", "ui_elec"):
                src = refc[(refc.variant == v) & (refc.stratum == st) & (refc.B == b)] \
                    if v != "p_prior" else cells[(cells.variant == v) & (cells.stratum == st) & (cells.B == b)]
            else:
                src = sub[(sub.variant == v) & (sub.stratum == st) & (sub.B == b)]
            parts.append(fmt(float(src[value].iloc[0]), float(src.ci95.iloc[0])) if len(src) else "-")
        if all(p == "-" for p in parts):
            continue
        lines.append(f"| {v} | " + " | ".join(parts) + " |")
    return head + "\n".join(lines) + "\n"


def attach_selected(ev, sel):
    """Alias the days-selected PMA width: add rows variant 'p3'/'p4' = chosen m per pool."""
    parts = [ev]
    if not len(ev):
        return ev
    for v in ("p3", "p4"):
        for pool in ("O", "E"):
            m = sel.get(f"{v}|{pool}")
            if not m:
                continue
            for suf in ("", "_shufY", "_nolab", "_h256"):
                parts.append(ev[(ev.variant == f"{v}_m{m}{suf}") & (ev.pool == pool)
                                ].assign(variant=f"{v}{suf}"))
    return pd.concat(parts, ignore_index=True)


def md_table(df, floatfmt=3):
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v)
                                for v in r.values) + " |\n"
    return out


def build_report():
    sys.path.insert(0, str(ROOT))
    from scripts.run_pc_queue import select_m
    ev, shuf, orc, ref = load_rows()
    sel = {f"{v}|{p}": select_m(v, p) for v in ("p3", "p4") for p in ("O", "E")}
    evs = attach_selected(ev, sel)
    ANA.mkdir(parents=True, exist_ok=True)
    (ANA / "selected_m.json").write_text(json.dumps(sel, indent=1))
    cells = cell_table(pd.concat([evs, ref[ref.variant == "p_prior"].assign(pool="-")]))
    cells.to_csv(ANA / "cells_r2.csv", index=False)
    refc = cell_table(ref[ref.B.isin([1, 8, 33])], by=("stratum", "variant", "B"))
    refc.to_csv(ANA / "reference_cells.csv", index=False)
    gain = paired_gain(evs, ref)
    gcell = cell_table(gain, value="gain")
    gcell.to_csv(ANA / "cells_gain_over_prior.csv", index=False)
    md = []
    # ---- main tables -------------------------------------------------------
    for pool in ("O", "E"):
        pc = cells.copy()
        pc.loc[pc.variant == "p_prior", "pool"] = pool
        md.append(f"### Query R2, pool {pool} (mean±95% CI over draws x targets x seeds)\n")
        md.append(md_main(pc, refc, pool, variants=["p_prior", "p0", "p1", "p2", "p3", "p4", "ui_prior", "ui_elec"]))
        gg = gcell[gcell.pool == pool]
        md.append(f"### Gain over P-prior, pool {pool}\n")
        md.append(md_main(gg.rename(columns={"gain": "r2"}), refc.iloc[0:0], pool,
                          variants=["p0", "p1", "p2", "p3", "p4"]))
    # ---- paired differences between variants ---------------------------------
    pairs = [("p1", "p0"), ("p2", "p1"), ("p3", "p2"), ("p4", "p3"), ("p4", "p1"), ("p2", "p0"), ("p4", "p0")]
    diffs = pd.concat([paired_vs(evs, a, b) for a, b in pairs], ignore_index=True)
    diffs.to_csv(ANA / "paired_variant_diffs.csv", index=False)
    pooled = pd.concat([paired_vs(evs, a, b, by=("stratum", "pool")) for a, b in pairs], ignore_index=True)
    pooled.to_csv(ANA / "paired_variant_diffs_allB.csv", index=False)
    md.append("### Paired differences (variant a - b), all B pooled, same target/B/draw/seed\n")
    md.append(md_table(pooled[["a", "b", "stratum", "pool", "mean", "ci95", "count"]]))
    # ---- E - O ---------------------------------------------------------------
    keys = ["stratum", "target", "B", "draw", "seed", "variant"]
    eo = evs[evs.pool == "E"][keys + ["r2"]].merge(evs[evs.pool == "O"][keys + ["r2"]],
                                                 on=keys, suffixes=("_E", "_O"))
    eo["diff"] = eo.r2_E - eo.r2_O
    eoc = cell_table(eo, value="diff", by=("stratum", "variant"))
    eoc.to_csv(ANA / "pool_E_minus_O.csv", index=False)
    md.append("### Pool E minus pool O (paired), all B pooled\n")
    md.append(md_table(eoc[["stratum", "variant", "diff", "ci95", "count"]]))
    # ---- gap to ui references --------------------------------------------------
    gaps = []
    for uiname in ("ui_prior", "ui_elec"):
        u = ref[ref.variant == uiname][["stratum", "target", "B", "draw", "r2"]].rename(columns={"r2": "r2_ui"})
        m = evs.merge(u, on=["stratum", "target", "B", "draw"])
        m["gap"] = m.r2 - m.r2_ui
        c = cell_table(m, value="gap", by=("stratum", "pool", "variant", "B"))
        c["ref"] = uiname
        gaps.append(c)
    gaps = pd.concat(gaps)
    gaps.to_csv(ANA / "gap_to_ui.csv", index=False)
    # ---- source val -------------------------------------------------------------
    vt = val_table()
    vt.to_csv(ANA / "source_val.csv", index=False)
    vsum = vt.groupby(["stratum", "pool", "tag"]).agg(
        val=("best_val_r2", "mean"), val_prior=("val_prior_r2", "mean"),
        best_step=("best_step", "mean"), converged=("converged", "mean")).reset_index()
    md.append("### Source validation (mean of 2 seeds)\n")
    md.append(md_table(vsum))
    # ---- oracle / shuffled ---------------------------------------------------------
    if len(orc):
        oc = cell_table(orc, by=("stratum", "pool", "variant"))
        oc.to_csv(ANA / "oracle_cells.csv", index=False)
        # compare each oracle with the same generator at B=8/33 on the budgeted protocol
        orc2 = orc.copy()
        orc2["base"] = orc2.variant.str.replace("p_oracle_", "", regex=False).str.replace(
            "_half", "", regex=False)
        orc2["kind"] = np.where(orc2.variant.str.endswith("_half"), "oracle_half", "oracle_all")
        orow = orc2.groupby(["stratum", "pool", "base", "kind"]).r2.mean().unstack("kind").reset_index()
        bud = ev.groupby(["stratum", "pool", "variant", "B"]).r2.mean().unstack("B").reset_index()
        bud = bud.rename(columns={"variant": "base"})[["stratum", "pool", "base", 1, 8, 33]]
        bud.columns = [str(c) if isinstance(c, str) else f"B={c}" for c in bud.columns]
        otab = orow.merge(bud, on=["stratum", "pool", "base"], how="left")
        otab["all_minus_B33"] = otab.oracle_all - otab["B=33"]
        otab["half_minus_B33"] = otab.oracle_half - otab["B=33"]
        otab.to_csv(ANA / "oracle_vs_budgeted.csv", index=False)
        md.append("### P-oracle (diagnostic only; descriptors from all/half of the target trials)\n")
        md.append(md_table(otab))
    if len(shuf):
        shs = attach_selected(shuf, sel)
        sc = cell_table(shs, by=("stratum", "pool", "variant", "B"))
        sc.to_csv(ANA / "shuffled_label_control.csv", index=False)
        for pool in ("O", "E"):
            both = pd.concat([cells[cells.pool == pool], sc[sc.pool == pool]])
            md.append(f"### Label controls, pool {pool}: trained-with-labels, shuffled labels at eval (_shufY), "
                      "trained label-free (_nolab), wide generator (_h256)\n")
            md.append(md_main(both, refc.iloc[0:0], pool, variants=[
                "p0", "p0_shufY", "p0_nolab", "p0_h256", "p1_shufY", "p4", "p4_shufY", "p4_nolab"]))
    # ---- seed-level noise floor (R2 at each B, mean over targets+draws per seed) ----
    sd_tab = (evs.groupby(["stratum", "pool", "variant", "B", "seed"]).r2.mean()
              .unstack("seed").reset_index())
    sd_tab["seed_spread"] = (sd_tab[0] - sd_tab[1]).abs()
    sd_tab.to_csv(ANA / "seed_level_r2.csv", index=False)
    floor = sd_tab[sd_tab.variant.isin(["p0", "p1", "p2", "p3", "p4"])].groupby(
        ["stratum", "pool"]).seed_spread.mean().reset_index()
    floor.to_csv(ANA / "seed_noise_floor.csv", index=False)
    md.append("### Seed-to-seed |difference| of mean R2 (noise floor), averaged over variants and B\n")
    md.append(md_table(floor))
    # ---- full-B curves ---------------------------------------------------------------
    for pool in ("O", "E"):
        for st in ("days", "month", "year"):
            sub = cells[(cells.stratum == st) & ((cells.pool == pool) | (cells.variant == "p_prior"))]
            piv = sub.pivot_table(index="variant", columns="B", values="r2")
            civ = sub.pivot_table(index="variant", columns="B", values="ci95")
            order = [v for v in ["p_prior", "p0", "p1", "p2", "p3", "p4"] if v in piv.index]
            tab = pd.DataFrame({b: [fmt(piv.loc[v, b], civ.loc[v, b]) for v in order]
                                for b in piv.columns}, index=order).reset_index()
            md.append(f"### Full-B curve, {st}, pool {pool}\n")
            md.append(md_table(tab))
    allr = pd.concat([ev.assign(src="eval"), shuf.assign(src="shufY"), orc.assign(src="oracle"),
                      ref.assign(src="reference")], ignore_index=True)
    allr.to_csv(ANA / "all_rows.csv.gz", index=False)
    hs = h_summary(ev, ref)
    hs["n_runs_trained"] = int(len(val_table()))
    hs["n_eval_rows"] = int(len(ev))
    (ANA / "h_summary.json").write_text(json.dumps(hs, indent=1))
    (ANA / "tables.md").write_text("\n".join(md))
    print("wrote", ANA)


def main():
    build_report()


if __name__ == "__main__":
    main()
