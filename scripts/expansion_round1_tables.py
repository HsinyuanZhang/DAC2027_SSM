#!/usr/bin/env python
"""Markdown tables for the expansion round-1 README (reads rows + analyze_expansion outputs; CPU only)."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.atomic_jsonl import read_jsonl

BASE = ROOT / "results/expansion_round1/x/expansion_out"
SUM = ROOT / "results/analysis/expansion_20261016/round1_summary"
METHODS = ["p_prior", "p0", "r1s", "r2", "r3"]
BS = [1, 4, 8, 16, 33]
NAME = {"p_prior": "P-prior", "p0": "P0", "r1s": "P0+R1-stream", "r2": "R2", "r3": "R3"}


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                      else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def load():
    recs = []
    for f in sorted((BASE / "rows").rglob("*.jsonl")):
        recs.extend(read_jsonl(f))
    return pd.DataFrame(recs)


def main():
    df = load()
    df["date"] = df["date"].astype(str)
    key = ["bid", "target", "date", "stratum", "gap_days", "trunk_gap_days", "n_units"]
    per = df.groupby(key + ["B", "method"])["r2"].mean().unstack("method").reset_index()
    per = per[per.B.isin(BS)].sort_values(["bid", "date", "B"])
    per.to_csv(SUM / "per_target_absolute_r2.csv", index=False)
    t1 = per[["bid", "date", "stratum", "gap_days", "trunk_gap_days", "n_units", "B"] + METHODS].rename(columns=NAME)
    print("## PER_TARGET\n" + md(t1))

    # per-stratum: session-level mean (average over backbones evaluating the same session), then mean over sessions
    sess = per.groupby(["stratum", "date", "B"])[METHODS].mean().reset_index()
    st = sess.groupby(["stratum", "B"])[METHODS].mean().reset_index()
    n = sess.groupby(["stratum", "B"]).date.nunique().rename("n_sessions").reset_index()
    st = st.merge(n, on=["stratum", "B"])
    st = st[["stratum", "n_sessions", "B"] + METHODS].rename(columns=NAME)
    st.to_csv(SUM / "per_stratum_absolute_r2.csv", index=False)
    print("## PER_STRATUM\n" + md(st))

    pc = pd.read_csv(SUM / "paired_session_cluster.csv")
    pc["B"] = pc["B"].astype(int)
    pc["ci95_boot"] = pc.apply(lambda r: "" if pd.isna(r.boot_lo) else f"[{r.boot_lo:+.4f}, {r.boot_hi:+.4f}]", axis=1)
    pc["ci95_t"] = pc.apply(lambda r: "" if pd.isna(r.t_lo) else f"[{r.t_lo:+.4f}, {r.t_hi:+.4f}]", axis=1)
    pc["mean"] = pc["mean"].map(lambda v: f"{v:+.4f}")
    pc["pos"] = pc.apply(lambda r: f"{int(r.n_positive_sessions)}/{int(r.n_sessions)}", axis=1)
    cols = ["stratum", "B", "comparison", "n_sessions", "n_backbone_target_pairs", "mean", "ci95_boot", "ci95_t", "pos"]
    print("## PAIRED\n" + md(pc[cols].sort_values(["stratum", "comparison", "B"])))

    tg = pd.read_csv(SUM / "paired_session_cluster_by_trunk_gap.csv")
    tg["B"] = tg["B"].astype(int)
    tg["ci95_boot"] = tg.apply(lambda r: "" if pd.isna(r.boot_lo) else f"[{r.boot_lo:+.4f}, {r.boot_hi:+.4f}]", axis=1)
    tg["mean"] = tg["mean"].map(lambda v: f"{v:+.4f}")
    tg["pos"] = tg.apply(lambda r: f"{int(r.n_positive_sessions)}/{int(r.n_sessions)}", axis=1)
    cols = ["stratum", "trunk_stratum", "B", "comparison", "n_sessions", "mean", "ci95_boot", "pos"]
    print("## TRUNKGAP\n" + md(tg[cols].sort_values(["stratum", "trunk_stratum", "comparison", "B"])))

    bs = pd.read_csv(SUM / "backbone_spread.csv")
    bs["B"] = bs["B"].astype(int)
    print("## SPREAD\n" + md(bs.sort_values(["stratum", "comparison", "B"]), "{:+.4f}"))


if __name__ == "__main__":
    main()
