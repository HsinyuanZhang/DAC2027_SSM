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
        ref += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
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


def main():
    ev, shuf, orc, ref = load_rows()
    ANA.mkdir(parents=True, exist_ok=True)
    ev["variant_key"] = ev["variant"]
    cells = cell_table(pd.concat([ev, ref[ref.variant == "p_prior"].assign(pool="-")]))
    cells.to_csv(ANA / "cells_r2.csv", index=False)
    gain = paired_gain(ev, ref)
    gcell = cell_table(gain, value="gain")
    gcell.to_csv(ANA / "cells_gain_over_prior.csv", index=False)
    refc = cell_table(ref[ref.B.isin([1, 8, 33])], by=("stratum", "variant", "B"))
    refc.to_csv(ANA / "reference_cells.csv", index=False)
    val_table().to_csv(ANA / "source_val.csv", index=False)
    if len(shuf):
        cell_table(shuf).to_csv(ANA / "shuffled_label_control.csv", index=False)
    if len(orc):
        cell_table(orc, by=("stratum", "pool", "variant")).to_csv(ANA / "oracle_cells.csv", index=False)
    (ANA / "h_summary.json").write_text(json.dumps(h_summary(ev, ref), indent=1))
    print("wrote", ANA)


if __name__ == "__main__":
    main()
