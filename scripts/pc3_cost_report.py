#!/usr/bin/env python
"""Analytic cost of the P3 refinements (csv + json).  Review fix: buffers are sized from the ACTUAL
trial lengths (not the mean T), MACs are split into fixed-weight (RRAM) / SRAM-table / activation x
activation, and a fixed-length (capped) replay design is costed next to the uncapped one.

Inputs: results/analysis/.../replay_draw_stats.csv and replay_mc_summary.csv (scripts/pc3_fix_cpu.py),
selection jsons for the (S, L) of F7-J.  Nothing here is measured on hardware; P3 is evaluated in fp32
(no fixed-point simulation); Jbar and the readout basis are costed as int8 ROM only.

Sizing designs (replay_bins B-trial total; staging = first-arriving trial for R1-stream):
  mean_T           B * mean trial length (the superseded point estimate)
  draw_median/max  over the 8 real evaluation draws of the target (days: 2 targets pooled per target)
  mc_p95 / mc_max  over 2000 random B-trial draws of the same target
  worst_case       the B longest usable trials of the session
  cap{CAP}_worst   every replayed trial truncated to CAP bins -> B * CAP (staging: CAP)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.fwd_refine import trunk_macs_per_bin                      # noqa: E402
from ssm_decode.mamba3_official import build_mamba3_official              # noqa: E402
from ssm_decode.refine_cost import refinement_cost                        # noqa: E402

OUTD = ROOT / "results/analysis/population_calib_p3_20261016"
P3 = ROOT / "results/population_calib_p3_20261016"
KB = 1024.0
BS = (1, 2, 4, 8, 16, 33)
CAP = 256
METHODS = [("p0_base", {}), ("r1_full", {}), ("r1_diag", {}), ("r1_lr16", {}), ("r1s_full", {}),
           ("r1s_lr16", {}), ("r1s_lr16", {"staging": "none"}), ("r2", {"steps": 5, "lags": 16}),
           ("r2", {"steps": 10, "lags": 32}), ("r3_lr16", {"steps": 5, "lags": 16}),
           ("r3_full", {"steps": 5, "lags": 16})]
KEY = ["r1_lr16", "r1_full", "r1s_lr16", "r2_sel", "r3_lr16_sel"]


def trunk():
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # MAC counting only: CPU is enough
    return trunk_macs_per_bin(build_mamba3_official(10, 2, 128, 2, 32, 0.0).eval())


def selection_for(stratum):
    p = {"days": P3 / "select/selection_days_is.json",
         "month": P3 / "fix/select/selection_month_oosc.json",       # causal headline rule
         "year": P3 / "select/selection_year_oos.json"}[stratum]
    return json.loads(p.read_text())["r2"]


def sizing_table(mc, draws, stratum, target, B):
    """{design: (replay_bins, staging_bins)} for one target and B."""
    m = mc[(mc.target == target) & (mc.B == B)].iloc[0]
    d = draws[(draws.target == target) & (draws.B == B)]
    return {
        "mean_T": (round(m.replay_bins_mean_T), round(m.mean_trial_bins)),
        "draw_median": (int(d.replay_bins.median()), int(d.first_trial_bins.median())),
        "draw_max": (int(d.replay_bins.max()), int(d.first_trial_bins.max())),
        "mc_p95": (int(round(m.replay_bins_mc_p95)), int(round(m.first_trial_mc_p95))),
        "mc_max": (int(m.replay_bins_mc_max), int(m.first_trial_mc_max)),
        "worst_case": (int(m.replay_bins_worst_case), int(m.max_trial_bins_session)),
        f"cap{CAP}_worst": (B * CAP, CAP),
    }


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    tr = trunk()
    mc = pd.read_csv(OUTD / "replay_mc_summary.csv")
    draws = pd.read_csv(OUTD / "replay_draw_stats.csv")
    cases = []
    for (stratum, target), g in mc.groupby(["stratum", "target"]):
        cases.append({"name": f"000688 {target[-8:]} (N={int(g.N.iat[0])})", "stratum": stratum, "target": target,
                      "N": int(g.N.iat[0]), "K": 10, "D": 2, "H": 128, "T": round(float(g.mean_trial_bins.iat[0])),
                      "bx": 3, "by": 16, "trunk": tr, "bin_ms": 20})
    hd = {"name": "HD-sEMG (N=256)", "stratum": "hd", "target": None, "N": 256, "K": 16, "D": 5, "H": 128, "T": 1250,
          "bx": 8, "by": 16, "trunk": tr, "bin_ms": 20}
    rows, detail = [], []

    def add_row(case, tag, sizing, B, r, kw):
        rows.append({"case": case["name"], "stratum": case["stratum"], "method": tag, "sizing": sizing, "B": B,
                     "bins_per_trial_mean": case["T"], "replay_bins": r["replay_bins"],
                     "staging_bins": r.get("staging_bins"), "macs_total": r["macs_total"],
                     "macs_fixed_weight_rram": r.get("macs_by_class", {}).get("fixed_weight_rram"),
                     "macs_sram_table_x_act": r.get("macs_by_class", {}).get("sram_table_x_act"),
                     "macs_act_x_act": r.get("macs_by_class", {}).get("act_x_act"),
                     "fixed_weight_fraction": r.get("fixed_weight_fraction"),
                     "replay_buffer_KB": r.get("replay_buffer_bytes", 0.0) / KB,
                     "sram_total_KB": r["sram_total"] / KB, "rom_KB": sum(r["rom_bytes"].values()) / KB,
                     "trunk_state_KB": r["sram_bytes"].get("trunk_state (shared with inference)", 0) / KB,
                     "fits_128KB_replay_only": r.get("replay_buffer_bytes", 0.0) / KB <= 128,
                     "fits_128KB_total": r["sram_total"] / KB <= 128,
                     "realtime_s_equiv": r.get("realtime_seconds_equiv", 0.0)})

    for case in cases:
        sel = selection_for(case["stratum"])
        for B in BS:
            sizes = sizing_table(mc, draws, case["stratum"], case["target"], B)
            for design, (nb, stg) in sizes.items():
                for method, kw in METHODS:
                    if method == "p0_base" and design != "mean_T":
                        continue
                    if method.startswith(("r2", "r3")) and "steps" in kw and design not in ("mean_T", "worst_case"):
                        continue                                     # fixed-(S,L) rows only for two designs
                    r = refinement_cost(case, B, method, replay_bins=nb, staging_bins=stg, **kw)
                    tag = method + (f"_S{kw['steps']}_L{kw['lags']}" if "steps" in kw else "") + \
                        ("_nostaging(unevaluated)" if kw.get("staging") == "none" else "")
                    if method == "p0_base":
                        r["sram_total"], r["replay_bins"] = 0.0, nb
                    add_row(case, tag, design, B, r, kw)
                rule = sel[str(B)]["cfg"]
                for method in ("r2", "r3_lr16"):
                    r = refinement_cost(case, B, method, steps=rule["S"], lags=rule["L"], replay_bins=nb)
                    add_row(case, f"{method}_selected(S={rule['S']},L={rule['L']})", design, B, r, {})
            r = refinement_cost(case, B, "r1_lr16", replay_bins=sizes["worst_case"][0])
            detail.append({"case": case["name"], "method": "r1_lr16", "sizing": "worst_case", "B": B,
                           "macs": r["macs"], "macs_by_class": r["macs_by_class"],
                           "sram_bytes": r["sram_bytes"], "rom_bytes": r["rom_bytes"]})
    # HD-sEMG: trials have a fixed 25 s length, so B * T is exact
    for B in BS:
        for method, kw in METHODS:
            r = refinement_cost(hd, B, method, **kw)
            tag = method + (f"_S{kw['steps']}_L{kw['lags']}" if "steps" in kw else "") + \
                ("_nostaging(unevaluated)" if kw.get("staging") == "none" else "")
            add_row(hd, tag, "fixed_T", B, r, kw)
            detail.append({"case": hd["name"], "method": tag, "sizing": "fixed_T", "B": B, "macs": r["macs"],
                           "macs_by_class": r.get("macs_by_class"), "sram_bytes": r["sram_bytes"],
                           "rom_bytes": r["rom_bytes"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "cost_p3.csv", index=False)
    (OUTD / "cost_p3_detail.json").write_text(json.dumps(detail, indent=1, default=float))
    (OUTD / "cost_p3_cases.json").write_text(json.dumps(cases + [hd], indent=1))
    # --- buffer summary: raw replay buffer KB per design (3/4/8-bit counts) and largest B that fits 128 KB
    br = []
    for case in cases:
        for B in BS:
            sizes = sizing_table(mc, draws, case["stratum"], case["target"], B)
            for design, (nb, stg) in sizes.items():
                for bx in (3, 4, 8):
                    bpb = (case["N"] * bx + case["D"] * case["by"] + 1) / 8
                    br.append({"case": case["name"], "stratum": case["stratum"], "B": B, "sizing": design, "bx": bx,
                               "replay_bins": nb, "bytes_per_bin": bpb, "replay_buffer_KB": nb * bpb / KB,
                               "fits_128KB_replay_only": nb * bpb / KB <= 128,
                               "staging_bins": stg, "staging_KB": stg * bpb / KB})
    bd = pd.DataFrame(br)
    bd.to_csv(OUTD / "cost_p3_buffer_summary.csv", index=False)
    sub = df[(df.method == "r1_lr16") & (df.stratum != "hd")]
    pd.set_option("display.width", 250)
    print(sub[sub.B.isin([8, 33])][["case", "sizing", "B", "replay_bins", "replay_buffer_KB", "sram_total_KB",
                                   "fits_128KB_replay_only", "fits_128KB_total", "fixed_weight_fraction"]].to_string())


if __name__ == "__main__":
    main()
