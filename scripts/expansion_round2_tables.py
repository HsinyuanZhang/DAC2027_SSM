#!/usr/bin/env python
"""Markdown tables for results/analysis/expansion_20261016/round2_README.md (reads round2_summary/*.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results/analysis/expansion_20261016/round2_summary"
TIERS = ["<=30 d", "31-180 d", ">180 d"]
BS = ["1", "4", "8", "16", "33", "B4-33"]
LABEL = {
    "p_prior": "P-prior", "p0": "P0", "ridge_cca": "ridge-CCA",
    "w_alpha_a": "W_alpha (a)", "w_alpha_b": "W_alpha (b)",
    "s1_causal": "S1-causal (a)", "s1_causal_cons": "S1-causal (a) cons-gate",
    "s1_causal_b": "S1-causal (b)", "s1_causal_b_cons": "S1-causal (b) cons-gate",
    "combo_a": "COMBO (a)", "combo_a_cons": "COMBO (a) cons-gate",
    "combo_b": "COMBO (b)", "combo_b_cons": "COMBO (b) cons-gate",
    "cca_s1": "CCA+S1 (a)", "cca_s1_cons": "CCA+S1 (a) cons-gate",
    "cca_s1_b": "CCA+S1 (b)", "cca_s1_b_cons": "CCA+S1 (b) cons-gate",
    "r2": "R2 (ref, replay)", "r3": "R3 (ref, replay)",
}
ORDER = list(LABEL)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    s = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def cell_abs(r):
    return f"{r['mean']:.3f}"


def cell_delta(r):
    if not np.isfinite(r["mean"]):
        return "n/a"
    if np.isfinite(r["ci_lo"]):
        return f"{r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] ({int(r['n_pos'])}/{int(r['n_sessions'])})"
    return f"{r['mean']:+.3f} (n=1)"


def pivot(df, tier_col, tier, fn, methods, bs=BS):
    d = df[(df[tier_col] == tier)] if tier_col else df
    d = d.assign(Bs=d["B"].astype(str))
    rows = []
    for m in methods:
        row = {"method": LABEL[m]}
        for B in bs:
            x = d[(d.method == m) & (d.Bs == B)]
            row["B=" + B if B != "B4-33" else "B4-33"] = fn(x.iloc[0]) if len(x) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = []
    mem = pd.read_csv(S / "target_membership.csv")
    mem = mem[mem.bid != "sm_0307"]
    out.append("### T1. 每个 gap tier 的构成（不含 IS backbone sm_0307）\n")
    rows = []
    for t in TIERS:
        m = mem[mem.tier == t]
        rows.append({"tier (trunk gap)": t, "n (backbone,target)": len(m), "n distinct target sessions": m.target.nunique(),
                     "sub-C / sub-M pairs": f"{(m.subject == 'sub-C').sum()} / {(m.subject == 'sub-M').sum()}",
                     "backbones": ", ".join(sorted(m.bid.unique())),
                     "trunk gap range (d)": f"{m.trunk_gap_days.min()}-{m.trunk_gap_days.max()}",
                     "selection gap range (d)": f"{m.gap_days.min()}-{m.gap_days.max()}"})
    out.append(md(pd.DataFrame(rows)))
    out.append("\n### T2. 每个 target 的两个 gap（第一列为 tier 依据）\n")
    mm = mem.sort_values(["tier", "bid", "date"])[["tier", "bid", "date", "stratum", "trunk_gap_days", "gap_days", "sel_tier"]]
    mm.columns = ["tier (trunk gap)", "backbone", "target", "stratum label", "gap to last trunk-training session (d)",
                  "gap to selection session pool[-1] (d)", "tier by selection gap"]
    out.append(md(mm))

    for lat in (0, 1):
        tag = "latency 0（主行）" if lat == 0 else "latency 1（sensitivity）"
        a = pd.read_csv(S / f"abs_by_tier_lat{lat}.csv")
        d = pd.read_csv(S / f"delta_vs_p0_by_tier_lat{lat}.csv")
        for t in TIERS:
            n = int(a[(a.tier == t) & (a.method == "p0") & (a.B.astype(str) == "8")].n_sessions.iloc[0])
            out.append(f"\n### T3.{lat} 绝对 R2，{tag}，tier {t}（P0 的 session 数 {n}；每个格子是 session 层面均值）\n")
            out.append(md(pivot(a, "tier", t, cell_abs, ORDER)))
            out.append(f"\n### T4.{lat} 相对 P0 的差，{tag}，tier {t}：mean [95% bootstrap CI over target sessions]（正差 session 数/总数）\n")
            out.append(md(pivot(d, "tier", t, cell_delta, [m for m in ORDER if m != "p0"])))
        sd = pd.read_csv(S / f"delta_vs_p0_by_seltier_lat{lat}.csv")
        out.append(f"\n### T5.{lat} 第二个 gap（到 selection session）分层：相对 P0 的差，{tag}，B4-33 与 B=8、B=33\n")
        for t in TIERS:
            out.append(f"\n**selection-gap tier {t}**\n")
            out.append(md(pivot(sd, "sel_tier", t, cell_delta, [m for m in ORDER if m != "p0"], bs=["8", "33", "B4-33"])))

    pb = pd.read_csv(S / "per_backbone_lat0.csv")
    cells = pd.read_csv(S / "cells.csv")
    for bid in sorted(pb.bid.unique()):
        if bid == "sm_0307":
            continue
        n = int(cells[(cells.bid == bid) & (cells.method == "p0") & (cells.B == 8) & (cells.latency == 0)].shape[0])
        out.append(f"\n### T6.{bid} backbone {bid}（{n} 个 target；latency 0）：绝对 R2 与相对 P0 的差（B4-33，CI 为 target 层面）\n")
        ab = pb[(pb.bid == bid) & (pb.base.isna())].assign(Bs=lambda x: x.B.astype(str))
        de = pb[(pb.bid == bid) & (pb.base == "p0")].assign(Bs=lambda x: x.B.astype(str))
        rows = []
        for m in ORDER:
            row = {"method": LABEL[m]}
            for B in ["1", "4", "8", "16", "33", "B4-33"]:
                x = ab[(ab.method == m) & (ab.Bs == B)]
                row["R2 B=" + B if B != "B4-33" else "R2 B4-33"] = f"{x['mean'].iloc[0]:.3f}" if len(x) else "n/a"
            x = de[(de.method == m) & (de.Bs == "B4-33")]
            if m == "p0":
                row["delta vs P0, B4-33"] = "-"
            elif len(x) and np.isfinite(x["mean"].iloc[0]):
                r = x.iloc[0]
                row["delta vs P0, B4-33"] = (f"{r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                                             if np.isfinite(r["ci_lo"]) else f"{r['mean']:+.3f}")
            else:
                row["delta vs P0, B4-33"] = "n/a"
            rows.append(row)
        out.append(md(pd.DataFrame(rows)))

    isc = pd.read_csv(S / "cells_is_backbone.csv")
    isc = isc[isc.latency == 0]
    out.append("\n### T7. IS backbone sm_0307 → 20140307（单独报告，不进任何 tier；latency 0；1 个 session，没有 CI）\n")
    p0 = isc[isc.method == "p0"].set_index("B")["r2"]
    rows = []
    for m in ORDER:
        row = {"method": LABEL[m]}
        for B in [1, 4, 8, 16, 33]:
            x = isc[(isc.method == m) & (isc.B == B)]
            v = x["r2"].iloc[0] if len(x) else np.nan
            row[f"R2 B={B}"] = f"{v:.3f}" if np.isfinite(v) else "n/a"
        rows.append(row)
    out.append(md(pd.DataFrame(rows)))

    comp = pd.read_csv(S / "selection_composition.csv")
    comp = comp[comp.bid != "sm_0307"]
    out.append("\n### T8. 每个 target 的 distinct selection sessions（protocol (a) expanding window / (b) strict；按 B）\n")
    rows = []
    for (bid, date), g in comp.groupby(["bid", "date"]):
        g = g.set_index("B")
        rows.append({"backbone": bid, "target": date,
                     "(a) n_sel B=1/4/8/16/33": "/".join(str(int(g.loc[b, "n_sel_a"])) for b in (1, 4, 8, 16, 33)),
                     "(b) n_sel B=1/4/8/16/33": "/".join(str(int(g.loc[b, "n_sel_b"])) for b in (1, 4, 8, 16, 33)),
                     "alpha (a) B=1/4/8/16/33": "/".join(f"{g.loc[b, 'alpha_a']:.2f}" for b in (1, 4, 8, 16, 33)),
                     "alpha (b) B=1/4/8/16/33": "/".join(f"{g.loc[b, 'alpha_b']:.2f}" for b in (1, 4, 8, 16, 33)),
                     "S1 / COMBO / CCA+S1 n_sel (a), B=8": f"{int(g.loc[8, 'n_s1_a'])} / {int(g.loc[8, 'n_combo_a'])} / {int(g.loc[8, 'n_cca_a'])}"})
    out.append(md(pd.DataFrame(rows).sort_values(["backbone", "target"])))

    av = pd.read_csv(S / "availability.csv")
    av = av[(av.latency == 0)]
    c = pd.read_csv(S / "cells_all.csv")
    c = c[c.latency == 0]
    rows = []
    for m in ORDER:
        x = c[c.method == m]
        row = {"method": LABEL[m], "cells (backbone,target,B)": len(x), "available": int(x.r2.notna().sum())}
        for B in (1, 4, 8, 16, 33):
            xb = x[x.B == B]
            row[f"avail B={B}"] = f"{int(xb.r2.notna().sum())}/{len(xb)}"
        rows.append(row)
    out.append("\n### T9. 方法可用性（latency 0，含 sm_0307）\n")
    out.append(md(pd.DataFrame(rows)))
    (S.parent / "round2_tables.md").write_text("\n".join(out))
    print("wrote", S.parent / "round2_tables.md")


if __name__ == "__main__":
    main()
