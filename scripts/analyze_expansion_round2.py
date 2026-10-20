#!/usr/bin/env python
"""Aggregate the expansion round-2 causal-first-B rows (CPU only).

Input : <round2 out>/causal_rows/<bid>/<date>.jsonl  and  <out>/causal_select/composition/*.json
Output: CSVs under results/analysis/expansion_20261016/round2_summary/ and a markdown fragment
        (round2_tables.md) that the README embeds.

Rules
* latency 0 is the primary row, latency 1 is a sensitivity table.
* seeds are averaged first (p_prior / ridge_cca are seed free), then (bid, target) cells are the unit.
* the cluster is the target SESSION inside a gap tier: cells of the same session and tier from different
  backbones are averaged, then percentile bootstrap (10,000, seed 0) over sessions.
* gap tier primary = days from the latest TRUNK-TRAINING session to the target (trunk_gap_days);
  the gap from the latest selection / source session (gap_days) is reported as a second column.
* unavailable rows (r2 is null) are kept out of the means and counted.
* 20140307 (sm_0307's only target) is not reportable. It stays a legal legacy selection session.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIERS = (("<=30 d", 0, 30), ("31-180 d", 31, 180), (">180 d", 181, 10**9))
MAIN = ("p_prior", "p0", "ridge_cca", "w_alpha_a", "w_alpha_b", "s1_causal", "s1_causal_b", "combo_a", "combo_b",
        "cca_s1", "cca_s1_b")
CONS = ("s1_causal_cons", "s1_causal_b_cons", "combo_a_cons", "combo_b_cons", "cca_s1_cons", "cca_s1_b_cons")
REFS = ("r2", "r3")
ALL_METHODS = MAIN + CONS + REFS
B_ORDER = (1, 2, 4, 8, 16, 33)
N_BOOT = 10000
EXCLUDED_TARGET_DATES = frozenset({"20140307"})


def tier_of(days) -> str:
    for name, lo, hi in TIERS:
        if lo <= int(days) <= hi:
            return name
    raise ValueError(days)


def load(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((out / "causal_rows").glob("*/*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["B"] = df["B"].astype(int)
    df["latency"] = df["latency"].astype(int)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["tier"] = df["trunk_gap_days"].map(tier_of)
    df["sel_tier"] = df["gap_days"].map(tier_of)
    return df


def audit(df: pd.DataFrame) -> dict:
    bad_final = sorted(set(df["date"]) & {"20151113", "20151116", "20151117", "20151119", "20151120", "20151201",
                                          "20150623", "20150625", "20150626"})
    return {
        "n_rows": int(len(df)),
        "n_unavailable": int(df["r2"].isna().sum()),
        "max_h2_backward_calls": int(df["h2_backward_calls"].max()),
        "all_trunk_unchanged": bool(df["h1_trunk_unchanged"].all()),
        "all_query_disjoint": bool(df["h4_query_disjoint"].all()),
        "forbidden_dates_present": bad_final,
        "year_2016_present": sorted(d for d in set(df["date"]) if d.startswith("2016")),
        "rule_versions": sorted(set(df["rule_version"])),
        "protocols": sorted(set(df["protocol"])),
        "n_bids": int(df["bid"].nunique()),
        "n_targets": int(df[["bid", "date"]].drop_duplicates().shape[0]),
        "n_distinct_target_sessions": int(df["target"].nunique()),
        "selection_oos_backbone": sorted(set(map(str, df["selection_oos_backbone"]))),
    }


def cells(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (bid, target, B, latency, method): mean over seeds, NaN-aware."""
    keys = ["bid", "subject", "target", "date", "stratum", "gap_days", "trunk_gap_days", "tier", "sel_tier",
            "B", "latency", "method"]
    g = df.groupby(keys, dropna=False)["r2"]
    out = g.mean().rename("r2").reset_index()
    out["n_seed_rows"] = g.size().values
    out["n_avail"] = g.count().values
    return out


def boot(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def session_values(c: pd.DataFrame, method: str, base: str | None, by: str = "tier") -> pd.DataFrame:
    """Per (group, B, session) values of ``method`` (or ``method - base``), averaged over backbones."""
    a = c[c.method == method][["bid", "target", by, "B", "latency", "r2"]]
    if base is not None:
        b = c[c.method == base][["bid", "target", by, "B", "latency", "r2"]].rename(columns={"r2": "r2_base"})
        a = a.merge(b, on=["bid", "target", by, "B", "latency"], how="inner")
        a["v"] = a["r2"] - a["r2_base"]
    else:
        a = a.assign(v=a["r2"])
    a = a[np.isfinite(a["v"])]
    return a.groupby([by, "B", "latency", "target"])["v"].mean().reset_index()


def summarize(c: pd.DataFrame, methods, base: str | None, by: str = "tier", latency: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for m in methods:
        if m == base:
            continue
        sv = session_values(c[c.latency == latency], m, base, by)
        for (grp, B), g in sv.groupby([by, "B"]):
            mean, lo, hi = boot(g["v"].values, rng)
            out.append({by: grp, "B": B, "method": m, "base": base or "", "n_sessions": len(g),
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pos": int((g["v"] > 0).sum())})
        # B4-33 pool: average over B in {4,8,16,33} per session first
        sub = sv[sv.B.isin([4, 8, 16, 33])]
        for grp, g in sub.groupby(by):
            gg = g.groupby("target")["v"].mean()
            mean, lo, hi = boot(gg.values, rng)
            out.append({by: grp, "B": "B4-33", "method": m, "base": base or "", "n_sessions": len(gg),
                        "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pos": int((gg > 0).sum())})
    return pd.DataFrame(out)


def selection_frame(out: Path) -> pd.DataFrame:
    """Who actually votes: the alpha gap bin, and the S1 pool, not the whole allowed alpha pool."""
    from ssm_decode import expansion_causal as ec
    from ssm_decode import expansion_protocol as ep

    legacy = ROOT / "results/causal_main_20261019/select"
    bundles = ec.load_episodes(out, legacy if legacy.is_dir() else None)
    reported = ec.reported_target_dates()
    rows = []
    for bb in ep.build_backbones():
        if bb.bid == "sm_0307":
            continue
        sel_path = out / f"select/{bb.bid}.json"
        if not sel_path.exists():
            continue
        sel = json.loads(sel_path.read_text())
        for t in bb.targets:
            date = ep.date_of(t.session)
            if str(date).startswith("2016") or date in EXCLUDED_TARGET_DATES:
                continue
            for B in (1, 4, 8, 16, 33):
                if str(B) not in sel.get("b", {}):
                    continue
                eta = float(sel["b"][str(B)]["r2"]["eta"])
                rules = ec.target_rules(bundles, date, bb.trunk_gap_days(t.session), B, reported, eta)
                s1a = rules["a"]["s1_composition"]
                days_sessions = sorted({p.split(":", 1)[1] for p in s1a["pairs"] if p.startswith("days:")})
                va, vb = rules["a"]["alpha"], rules["b"]["alpha"]
                rows.append({
                    "bid": bb.bid, "date": date, "B": int(B),
                    "n_pool_a": rules["a"]["composition"]["n_sessions"],
                    "n_pool_b": rules["b"]["composition"]["n_sessions"],
                    "sessions_b": ",".join(rules["b"]["composition"]["sessions"]),
                    "pairs_b": ",".join(rules["b"]["composition"]["pairs"]),
                    "n_alpha_vote_a": int(va.get("n_sessions") or 0),
                    "n_alpha_vote_b": int(vb.get("n_sessions") or 0),
                    "alpha_bin_a": va.get("bin"), "alpha_bin_used_a": va.get("bin_used"),
                    "alpha_sessions_a": ",".join(va.get("sessions") or []),
                    "alpha_tags_a": ",".join(va.get("tags") or []),
                    "alpha_a": va.get("alpha"), "alpha_b": vb.get("alpha"),
                    "n_s1_a": s1a["n_sessions"], "n_s1_days_a": len(days_sessions),
                    "n_s1_trunk_sessions_a": s1a["n_trunk_sessions"],
                    "s1_trunks_a": ",".join(s1a["trunks"]),
                    "n_combo_a": rules["a"]["combo_composition"]["n_sessions"],
                    "n_combo_b": rules["b"]["combo_composition"]["n_sessions"],
                    "n_cca_a": rules["a"]["cca_composition"]["n_sessions"],
                })
    return pd.DataFrame(rows)


def slice_summaries(c: pd.DataFrame, latency: int) -> pd.DataFrame:
    """``>180 d`` with and without the non-converged sc_0309 trunk, plus sm_0512 alone."""
    specs = (
        ("gt180_all", c[c.tier == ">180 d"]),
        ("gt180_without_sc_0309", c[(c.tier == ">180 d") & (c.bid != "sc_0309")]),
        ("sc_0309_only", c[c.bid == "sc_0309"]),
        ("sm_0512_only", c[c.bid == "sm_0512"]),
    )
    frames = []
    methods = [m for m in ALL_METHODS if m != "p0"]
    for name, sub in specs:
        if sub.empty:
            continue
        sub = sub.copy()
        sub["slice"] = name
        frames.append(summarize(sub, ["p0"], None, "slice", latency))
        frames.append(summarize(sub, methods, "p0", "slice", latency))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt(r) -> str:
    if not np.isfinite(r["mean"]):
        return "n/a"
    if np.isfinite(r["ci_lo"]):
        return f"{r['mean']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    return f"{r['mean']:+.4f} (1 session)"


def md_table(df: pd.DataFrame, cols) -> str:
    cols = list(cols)
    s = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/expansion_round2"))
    ap.add_argument("--dest", default=str(ROOT / "results/analysis/expansion_20261016/round2_summary"))
    a = ap.parse_args()
    out, dest = Path(a.out), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    df_raw = load(out)
    # 20140307 is sm_0307's target and a legacy selection session. It is not a reportable target.
    df = df_raw[~df_raw["date"].isin(EXCLUDED_TARGET_DATES)].copy()
    au = audit(df)
    au["excluded_target_dates"] = sorted(EXCLUDED_TARGET_DATES)
    au["n_rows_including_excluded"] = int(len(df_raw))
    au["n_pairs_reportable"] = au["n_targets"]
    au["n_distinct_sessions_reportable"] = au["n_distinct_target_sessions"]
    (dest / "audit.json").write_text(json.dumps(au, indent=1))
    c = cells(df)
    c.to_csv(dest / "cells.csv", index=False)
    c.to_csv(dest / "cells_all.csv", index=False)
    stale = dest / "cells_is_backbone.csv"
    if stale.exists():
        stale.unlink()

    # availability
    av = c.groupby(["method", "latency"]).agg(cells=("r2", "size"), available=("r2", "count")).reset_index()
    av.to_csv(dest / "availability.csv", index=False)

    # tier membership: session counts and the second gap column
    members = (c[(c.latency == 0) & (c.method == "p0") & (c.B == 8)]
               [["bid", "subject", "target", "date", "stratum", "trunk_gap_days", "gap_days", "tier", "sel_tier"]]
               .sort_values(["tier", "bid", "date"]))
    members.to_csv(dest / "target_membership.csv", index=False)

    # Voting sessions are recomputed with the alpha-aware dedupe, not read from a stale composition log.
    comp = selection_frame(out)
    done = {}
    for p in sorted((out / "causal_select/composition").glob("*.json")):
        j = json.loads(p.read_text())
        if j["date"] in EXCLUDED_TARGET_DATES:
            continue
        done[(j["bid"], j["date"])] = bool(j.get("pool_complete"))
    comp["pool_complete"] = [done.get((r.bid, r.date), False) for r in comp.itertuples(index=False)]
    comp.to_csv(dest / "selection_composition.csv", index=False)
    b8 = comp[comp.B == 8]
    late = b8[b8.date.astype(str) >= "20150616"]
    notes = {
        "protocol_b_sessions": sorted(set(comp.sessions_b)),
        "protocol_b_pairs": sorted(set(comp.pairs_b)),
        "protocol_b_is_one_session": bool(comp.sessions_b.nunique() == 1),
        "s1_days_share_B8_after_20150615": {
            "n_targets": int(len(late)),
            "days_sessions_min": int(late.n_s1_days_a.min()) if len(late) else None,
            "days_sessions_max": int(late.n_s1_days_a.max()) if len(late) else None,
            "s1_sessions_min": int(late.n_s1_a.min()) if len(late) else None,
            "s1_sessions_max": int(late.n_s1_a.max()) if len(late) else None,
        },
        "alpha_vote_lt_pool_B8": int((b8.n_alpha_vote_a < b8.n_pool_a).sum()),
        "n_targets_B8": int(len(b8)),
        "le14_includes_20131220": sorted(
            b8.loc[(b8.alpha_bin_used_a == "le14") & b8.alpha_sessions_a.str.contains("20131220"), "bid"].unique()),
        "sc_0309_targets_in_gt180": int(((members.bid == "sc_0309") & (members.tier == ">180 d")).sum()),
        "sm_0512_targets_in_gt180": int(((members.bid == "sm_0512") & (members.tier == ">180 d")).sum()),
        "clustering_note": "The six sc_0309 sessions in >180 d share one trunk and one selection pool. "
                           "The session bootstrap treats them as independent, so that tier's CI is too narrow.",
    }
    (dest / "selection_notes.json").write_text(json.dumps(notes, indent=1))

    for lat in (0, 1):
        # absolute R2 per tier
        absr = summarize(c, ALL_METHODS, None, "tier", lat)
        absr.to_csv(dest / f"abs_by_tier_lat{lat}.csv", index=False)
        d0 = summarize(c, [m for m in ALL_METHODS if m != "p0"], "p0", "tier", lat)
        d0.to_csv(dest / f"delta_vs_p0_by_tier_lat{lat}.csv", index=False)
        dr = summarize(c, [m for m in ALL_METHODS if m != "ridge_cca"], "ridge_cca", "tier", lat)
        dr.to_csv(dest / f"delta_vs_ridge_by_tier_lat{lat}.csv", index=False)
        slice_summaries(c, lat).to_csv(dest / f"gt180_sensitivity_lat{lat}.csv", index=False)
        dsel = summarize(c, [m for m in ALL_METHODS if m != "p0"], "p0", "sel_tier", lat)
        dsel.to_csv(dest / f"delta_vs_p0_by_seltier_lat{lat}.csv", index=False)
        absel = summarize(c, ALL_METHODS, None, "sel_tier", lat)
        absel.to_csv(dest / f"abs_by_seltier_lat{lat}.csv", index=False)
        # per backbone (cluster = target within the backbone)
        pb = []
        rng = np.random.default_rng(0)
        for bid, cb in c[c.latency == lat].groupby("bid"):
            for m in ALL_METHODS:
                for base in (None, "p0"):
                    if base and m == "p0":
                        continue
                    sv = session_values(cb.assign(bk="all"), m, base, "bk")
                    for B, g in sv.groupby("B"):
                        mean, lo, hi = boot(g["v"].values, rng)
                        pb.append({"bid": bid, "method": m, "base": base or "", "B": B, "n_targets": len(g),
                                   "mean": mean, "ci_lo": lo, "ci_hi": hi})
                    g = sv[sv.B.isin([4, 8, 16, 33])].groupby("target")["v"].mean()
                    if len(g):
                        mean, lo, hi = boot(g.values, rng)
                        pb.append({"bid": bid, "method": m, "base": base or "", "B": "B4-33", "n_targets": len(g),
                                   "mean": mean, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(pb).to_csv(dest / f"per_backbone_lat{lat}.csv", index=False)
    print(json.dumps(au, indent=1))


if __name__ == "__main__":
    main()
