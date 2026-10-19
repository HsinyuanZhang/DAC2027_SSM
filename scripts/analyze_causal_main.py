#!/usr/bin/env python
"""Aggregate the causal-first-B main table and compare it with the random-draw rows.

Writes CSVs and a markdown fragment under results/analysis/causal_main_20261019/.
The month stratum is the drift row for 20140626 (same month trunk and the same rules);
it is not a second decode.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

OUT = ROOT / "results/analysis/causal_main_20261019"
RAW = ROOT / "results/causal_main_20261019/rows.jsonl"
B_SHOW = (1, 8, 33)
B_ALL = (1, 2, 4, 8, 16, 33)
PRIMARY = ("p_prior", "p0", "w_alpha_gap", "s1_causal", "combo", "r1_stream")
REFS = ("r2", "r3", "ridge_cca", "p0_ui_reg")
RNG = np.random.default_rng(0)


def load_rows() -> pd.DataFrame:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW}")
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    # month is the 20140626 drift decode, reported under its own stratum name.
    month = df[(df.stratum == "drift") & (df.date == "20140626")].copy()
    month["stratum"] = "month"
    return pd.concat([df, month], ignore_index=True)


def _boot_mean(values, n_boot=4000):
    x = np.asarray(values, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, "empty"
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan, "single_target"
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), "target_cluster"


def per_target(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean per (stratum, backbone, target, method, B, latency)."""
    g = (df.groupby(["stratum", "backbone", "target", "date", "method", "B", "latency"], dropna=False)
         .r2.mean().reset_index())
    return g


def cells(pt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["stratum", "backbone", "method", "B", "latency"]
    for key, block in pt.groupby(keys, dropna=False):
        mean, lo, hi, kind = _boot_mean(block.r2.to_numpy())
        rec = dict(zip(keys, key))
        rec.update({"n_targets": int(block.target.nunique()), "r2": mean, "ci_lo": lo, "ci_hi": hi,
                    "ci": kind})
        rows.append(rec)
    return pd.DataFrame(rows)


def paired(pt: pd.DataFrame, a: str, b: str, latency: int = 0) -> pd.DataFrame:
    """a - b, seed-averaged inside the target, then clustered by target."""
    sub = pt[(pt.latency == latency) & (pt.method.isin([a, b]))]
    rows = []
    for (stratum, backbone, B), block in sub.groupby(["stratum", "backbone", "B"]):
        wide = block.pivot_table(index="target", columns="method", values="r2", aggfunc="mean")
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = (wide[a] - wide[b]).dropna()
        mean, lo, hi, kind = _boot_mean(diff.to_numpy())
        rows.append({"stratum": stratum, "backbone": backbone, "B": int(B), "contrast": f"{a}-{b}",
                     "n_targets": int(len(diff)), "delta": mean, "ci_lo": lo, "ci_hi": hi, "ci": kind})
    return pd.DataFrame(rows)


def latency_delta(pt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = pt[pt.method.isin(PRIMARY + REFS)]
    for (stratum, backbone, method, B), block in sub.groupby(["stratum", "backbone", "method", "B"]):
        wide = block.pivot_table(index="target", columns="latency", values="r2", aggfunc="mean")
        if 0 not in wide.columns or 1 not in wide.columns:
            continue
        diff = (wide[1] - wide[0]).dropna()
        mean, lo, hi, kind = _boot_mean(diff.to_numpy())
        rows.append({"stratum": stratum, "backbone": backbone, "method": method, "B": int(B),
                     "n_targets": int(len(diff)), "latency1_minus_0": mean, "ci_lo": lo, "ci_hi": hi,
                     "ci": kind})
    return pd.DataFrame(rows)


def _rand_cells() -> pd.DataFrame:
    """Random-draw protocol means, labelled so they are not mixed into the causal table."""
    frames = []
    p0 = pd.read_csv(ROOT / "results/analysis/p0solve_20261019/cells.csv")
    keep = p0[p0.method.isin(["p_prior", "p0", "w_alpha_gap", "ridge_cca", "p0_ui_reg", "r3"])]
    frames.append(keep.rename(columns={"r2_mean": "r2_random"})[
        ["stratum", "B", "method", "r2_random", "n_targets"]])
    s1 = pd.read_csv(ROOT / "results/analysis/stream_f7_20261019/cells.csv")
    s1 = s1[s1.method == "s1c_rebase"].copy()
    s1["method"] = "s1_causal"
    s1["stratum"] = s1.stratum.map({"subm_drift": "drift", "subm_month": "month", "subm_year": "year",
                                    "subc_month": "subc_month"})
    frames.append(s1.rename(columns={"r2": "r2_random"})[["stratum", "B", "method", "r2_random"]])
    drift = pd.read_csv(ROOT / "results/analysis/drift_curve_20261016/cells_r2.csv")
    # drift file is per target; average to a stratum mean for r1s / r2 / p0.
    dmean = (drift[drift.method.isin(["r1s", "r2", "p0", "p_prior", "r3", "p0_ui_reg"])]
             .groupby(["B", "method"]).r2_mean.mean().reset_index())
    dmean["stratum"] = "drift"
    dmean["method"] = dmean.method.replace({"r1s": "r1_stream"})
    dmean = dmean.rename(columns={"r2_mean": "r2_random"})
    frames.append(dmean[["stratum", "B", "method", "r2_random"]])
    out = pd.concat(frames, ignore_index=True)
    out["protocol"] = "random_draw"
    return out


def protocol_delta(cell: pd.DataFrame, random_cells: pd.DataFrame) -> pd.DataFrame:
    causal = cell[(cell.latency == 0)].copy()
    # Pool sub-C backbones for the comparison against the published pooled random-draw row.
    pooled = (causal[causal.stratum == "subc_month"]
              .groupby(["stratum", "method", "B"]).r2.mean().reset_index())
    causal_m = causal.groupby(["stratum", "method", "B"]).r2.mean().reset_index()
    # Prefer the unpooled groupby; subc mean of backbone-means is what pooled is.
    merged = causal_m.merge(random_cells, on=["stratum", "method", "B"], how="left")
    merged["delta_causal_minus_random"] = merged["r2"] - merged["r2_random"]
    return merged


def cost_table() -> pd.DataFrame:
    """Compact cost from the reviewed models. Trunk MACs come from the stored stream_f7 trunk dict.

    mac_per_bin is the method's extra MAC on top of the frozen trunk forward (weight x activation
    and act x act of the calibration update only). mac_per_commit is the table update at a trial
    boundary. SRAM and ROM are separate. Replay is whether the B trials are stored and re-run.
    """
    from ssm_decode.population_calib import mac_counts, desc_dim
    from ssm_decode.p0solve import cost_model
    from ssm_decode.refine_cost import refinement_cost
    from ssm_decode.stream_f7 import (fir_bin_macs, like_for_like_memory, p0_calibrator_cost,
                                      p0_prefix_stat_bytes)

    trunk = json.loads((ROOT / "results/stream_f7_20261019/cost.json").read_text())["trunk"]
    rows = []
    k, d, h = 10, 2, 128
    b, t, lags = 8, 200, 32
    for n in (26, 256):
        nb = b * t
        p0m = mac_counts("p0", n, nb, k, desc_dim(k))
        cal = p0_calibrator_cost(k, n, 1)
        stats = p0_prefix_stat_bytes(n)
        prior = float(k * n * 4)
        norm = float(2 * n * 4)
        table = float(k * n * 4)
        fir = fir_bin_macs(lags + 1, d, k, n)
        mem = like_for_like_memory(n=n, k=k, d=d, n_lags=lags + 1, n_modes=4,
                                   replay_bins=nb, staging_bins=t, state_elems=trunk["state_elems"])
        case = {"name": f"N{n}", "N": n, "K": k, "D": d, "H": h, "T": t, "bx": 3, "by": 16,
                "trunk": trunk, "bin_ms": 20, "state_bits": 16}
        r1 = refinement_cost(case, b, "r1s_full", replay_bins=nb, staging_bins=t)
        r2 = refinement_cost(case, b, "r2", steps=10, lags=lags, replay_bins=nb, staging_bins=t)
        solve = cost_model(n, k, p=8)
        rom_cal = cal["weight_bytes"]
        rom_j = mem["jbar_rom_int8"]

        def add(method, mac_bin, mac_commit, sram, rom, replay, note):
            rows.append({"method": method, "N": n, "B_point": b, "T_point": t,
                         "mac_per_bin": mac_bin, "mac_per_commit": mac_commit,
                         "sram_bytes": sram, "rom_bytes": rom, "replay": replay, "note": note})

        add("p_prior", p0m["accumulate_macs"] / nb, p0m["closed_form_macs"] / b,
            stats + norm + table, prior, False,
            "prefix PopStats only; prior table counted as ROM (electrode prior, not learned)")
        add("p0", p0m["accumulate_macs"] / nb,
            p0m["closed_form_macs"] / b + cal["fixed_weight_macs_per_call"],
            stats + norm + table, rom_cal, False,
            "one calibrator call per commit; MLP weights are ROM")
        add("w_alpha_gap", p0m["accumulate_macs"] / nb + solve["macs_per_bin"]["total_full_G"],
            p0m["closed_form_macs"] / b + cal["fixed_weight_macs_per_call"] + k * n,
            stats + norm + table + solve["accumulator_bytes_fp32"], rom_cal, False,
            "P0 prefix plus the p0solve raw-moment accumulator as a stand-in for the CCA moments; "
            "not a separate CCA cost model")
        add("s1_causal", fir["total"] + p0m["accumulate_macs"] / nb,
            cal["fixed_weight_macs_per_call"],
            mem["s1_causal_3bit_ring_without_trunk_state"] - rom_cal - rom_j,
            rom_cal + rom_j, False,
            "FIR extra on the live forward; 3-bit ring; trunk state excluded; "
            "like_for_like_memory split into SRAM and ROM")
        add("combo", fir["total"] + p0m["accumulate_macs"] / nb + solve["macs_per_bin"]["total_full_G"],
            cal["fixed_weight_macs_per_call"] + k * n,
            mem["s1_causal_3bit_ring_without_trunk_state"] - rom_cal - rom_j + solve["accumulator_bytes_fp32"],
            rom_cal + rom_j, False,
            "S1-causal on a W_alpha base; moment accumulator added")
        # R1-stream: streaming forward includes the trunk. Report the non-trunk part per bin.
        r1_bin = (r1["macs_total"] - nb * (trunk["fixed_weight_macs"] + trunk["act_x_act_macs"] + k * n)) / nb
        add("r1_stream", max(r1_bin, 0.0),
            r1["macs"].get("readout solve [full]", 0) / b + cal["fixed_weight_macs_per_call"],
            r1["sram_total"] - r1["sram_bytes"].get("trunk_state (shared with inference)", 0),
            0.0, False,
            "refine_cost r1s_full, trunk forward removed from the per-bin figure; full readout kind")
        add("r2", (r2["macs_total"] / 10 - nb * (trunk["fixed_weight_macs"] + trunk["act_x_act_macs"])) / nb,
            r2["macs"].get("precondition + update (K*N per step)", 0) / 10,
            r2["sram_total"] - r2["sram_bytes"].get("trunk_state (shared with inference)", 0),
            r2["rom_bytes"].get("Jbar kernel (L+1) x D x K int8", 0), True,
            "refine_cost r2, S=10, one replay pass's non-trunk MAC per bin")
        add("r3", (r2["macs_total"] / 10) / nb, r2["macs_total"] / 10 / b, r2["sram_total"],
            sum(r2["rom_bytes"].values()), True,
            "same replay as r2 plus one readout pass; see refine_cost r3_full for the split")
        add("ridge_cca", solve["macs_per_bin"]["total_full_G"], solve["commit"]["total_macs_excluding_solve_flops"],
            solve["accumulator_bytes_fp32"] + table + norm, 0.0, False,
            "p0solve cost_model p=8 as the moment accumulator; CCA reference ROM is not in that model")
        add("p0_ui_reg", float("nan"), float("nan"), r2["sram_bytes"].get("raw replay buffer", 0.0),
            rom_cal, True,
            "non-deployable backprop; no MAC model in the reviewed files; replay buffer quoted from r2")
    return pd.DataFrame(rows)


def _fmt(mean, lo, hi, kind):
    if not np.isfinite(mean):
        return "—"
    if kind == "single_target" or not np.isfinite(lo):
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def markdown_tables(cell, paired_df, lat_df, proto) -> str:
    lines = ["# Causal-first-B tables (generated)", ""]
    main = cell[(cell.latency == 0) & (cell.B.isin(B_SHOW))]
    strata = ["days", "month", "year", "drift", "subc_month"]
    methods = list(PRIMARY) + list(REFS) + ["s1_causal_and", "combo_and"]
    for stratum in strata:
        lines.append(f"## {stratum}")
        lines.append("")
        lines.append("| method | B=1 | B=8 | B=33 |")
        lines.append("|---|---:|---:|---:|")
        sub = main[main.stratum == stratum]
        # Average across backbones for the stratum headline (sub-C has two).
        for method in methods:
            vals = []
            for B in B_SHOW:
                block = sub[(sub.method == method) & (sub.B == B)]
                if block.empty:
                    vals.append("—")
                    continue
                # If several backbones, average their target-means (each already a mean).
                mean, lo, hi, kind = _boot_mean(block.r2.to_numpy()) if len(block) > 1 else (
                    float(block.r2.iloc[0]), float(block.ci_lo.iloc[0]) if np.isfinite(block.ci_lo.iloc[0]) else np.nan,
                    float(block.ci_hi.iloc[0]) if np.isfinite(block.ci_hi.iloc[0]) else np.nan,
                    str(block.ci.iloc[0]))
                if len(block) > 1:
                    kind = "backbone_mean"
                    vals.append(f"{mean:.3f}")
                else:
                    vals.append(_fmt(mean, lo, hi, kind))
            if all(v == "—" for v in vals):
                continue
            lines.append(f"| {method} | " + " | ".join(vals) + " |")
        lines.append("")
    lines.append("## Paired differences, latency 0")
    lines.append("")
    if len(paired_df):
        show = paired_df[paired_df.B.isin(B_SHOW)]
        lines.append("| stratum | backbone | contrast | B | delta | CI |")
        lines.append("|---|---|---|---:|---:|---|")
        for _, r in show.sort_values(["stratum", "contrast", "B"]).iterrows():
            lines.append(f"| {r.stratum} | {r.backbone} | {r.contrast} | {int(r.B)} | {r.delta:.3f} | "
                         f"{_fmt(r.delta, r.ci_lo, r.ci_hi, r.ci)} |")
    lines.append("")
    lines.append("## Latency 1 minus latency 0")
    lines.append("")
    if len(lat_df):
        show = lat_df[lat_df.B.isin(B_SHOW) & lat_df.method.isin(PRIMARY)]
        lines.append("| stratum | method | B | delta |")
        lines.append("|---|---|---:|---:|")
        for _, r in show.sort_values(["stratum", "method", "B"]).iterrows():
            lines.append(f"| {r.stratum} | {r.method} | {int(r.B)} | "
                         f"{_fmt(r.latency1_minus_0, r.ci_lo, r.ci_hi, r.ci)} |")
    lines.append("")
    lines.append("## Protocol change (causal mean − random-draw mean)")
    lines.append("")
    if len(proto):
        show = proto[proto.B.isin(B_SHOW)]
        lines.append("| stratum | method | B | causal | random | delta |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in show.sort_values(["stratum", "method", "B"]).iterrows():
            rnd = "—" if not np.isfinite(r.r2_random) else f"{r.r2_random:.3f}"
            delta = "—" if not np.isfinite(r.delta_causal_minus_random) else f"{r.delta_causal_minus_random:+.3f}"
            cau = "—" if not np.isfinite(r.r2) else f"{r.r2:.3f}"
            lines.append(f"| {r.stratum} | {r.method} | {int(r.B)} | {cau} | {rnd} | {delta} |")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    df.to_csv(OUT / "rows_long.csv", index=False)
    pt = per_target(df)
    pt.to_csv(OUT / "per_target.csv", index=False)
    cell = cells(pt)
    cell.to_csv(OUT / "cells.csv", index=False)
    contrasts = []
    for a, b in (("p0", "p_prior"), ("w_alpha_gap", "p0"), ("s1_causal", "p0"),
                 ("s1_causal", "w_alpha_gap"), ("combo", "w_alpha_gap"), ("combo", "s1_causal"),
                 ("r1_stream", "p0"), ("s1_causal_and", "s1_causal"), ("combo_and", "combo"),
                 ("r2", "p0"), ("r3", "p0"), ("ridge_cca", "p0"), ("p0_ui_reg", "p0")):
        part = paired(pt, a, b, 0)
        if len(part):
            contrasts.append(part)
    paired_df = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    paired_df.to_csv(OUT / "paired.csv", index=False)
    lat = latency_delta(pt)
    lat.to_csv(OUT / "latency.csv", index=False)
    proto = protocol_delta(cell, _rand_cells())
    proto.to_csv(OUT / "protocol_delta.csv", index=False)
    cost = cost_table()
    cost.to_csv(OUT / "cost.csv", index=False)
    (OUT / "TABLES.md").write_text(markdown_tables(cell, paired_df, lat, proto))
    print(f"[analyze] targets={df.target.nunique()} rows={len(df)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
