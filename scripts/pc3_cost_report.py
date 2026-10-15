#!/usr/bin/env python
"""Analytic cost of the P3 refinements for 000688 and an HD-sEMG-sized case (csv + json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.fwd_refine import trunk_macs_per_bin                      # noqa: E402
from ssm_decode.mamba3_official import build_mamba3_official              # noqa: E402
from ssm_decode.refine_cost import refinement_cost                        # noqa: E402

OUTD = ROOT / "results/analysis/population_calib_p3_20261016"
KB = 1024.0
BS = (1, 2, 4, 8, 16, 33)
METHODS = [("p0_base", {}), ("r1_full", {}), ("r1_diag", {}), ("r1_lr16", {}), ("r1s_full", {}),
           ("r1s_lr16", {}), ("r1s_lr16", {"staging": "none"}), ("r2", {"steps": 5, "lags": 16}), ("r2", {"steps": 10, "lags": 32}),
           ("r3_lr16", {"steps": 5, "lags": 16}), ("r3_full", {"steps": 5, "lags": 16})]


def cases():
    stats = json.loads((ROOT / "results/population_calib_p3_20261016/trial_stats.json").read_text())
    t000688 = round(sum(v["len_mean"] for k, v in stats.items() if k.endswith(("20150616", "20150617"))) / 2)
    out = []
    for name, (n, k, d, t, bx, by) in {"000688 (N=26)": (26, 10, 2, t000688, 3, 16),
                                       "HD-sEMG (N=256)": (256, 16, 5, 1250, 8, 16)}.items():
        model = build_mamba3_official(k, d, 128, 2, 32, 0.0).cuda()
        out.append({"name": name, "N": n, "K": k, "D": d, "H": 128, "T": t, "bx": bx, "by": by,
                    "trunk": trunk_macs_per_bin(model.eval()), "bin_ms": 20})
    return out


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, detail = [], []
    for case in cases():
        for B in BS:
            for method, kw in METHODS:
                r = refinement_cost(case, B, method, **kw)
                tag = method + (f"_S{kw['steps']}_L{kw['lags']}" if "steps" in kw else "") + \
                    ("_nostaging(unevaluated)" if kw.get("staging") == "none" else "")
                row = {"case": case["name"], "method": tag, "B": B, "bins_per_trial": case["T"],
                       "replay_bins": r["replay_bins"], "macs_total": r["macs_total"],
                       "replay_buffer_KB": r.get("replay_buffer_bytes", 0.0) / KB,
                       "sram_total_KB": r["sram_total"] / KB,
                       "rom_KB": sum(r["rom_bytes"].values()) / KB,
                       "trunk_state_KB": r["sram_bytes"].get("trunk_state (shared with inference)", 0) / KB,
                       "fits_128KB_replay_only": r.get("replay_buffer_bytes", 0.0) / KB <= 128,
                       "fits_128KB_total": r["sram_total"] / KB <= 128,
                       "realtime_s_equiv": r.get("realtime_seconds_equiv", 0.0)}
                if method == "p0_base":
                    row["macs_total"] = r["macs"]["total"]
                rows.append(row)
                detail.append({"case": case["name"], "method": tag, "B": B,
                               "macs": r["macs"], "sram_bytes": r["sram_bytes"], "rom_bytes": r["rom_bytes"]})
        # F7-J / R3 with the (S, L) actually selected on source for the month/year headline set, per B
        sel = json.loads((ROOT / "results/population_calib_p3_20261016/select/selection_month_oos.json").read_text())
        for B in BS:
            rule = sel["r2"][str(B)]["cfg"]
            for method in ("r2", "r3_lr16"):
                r = refinement_cost(case, B, method, steps=rule["S"], lags=rule["L"])
                rows.append({"case": case["name"], "method": f"{method}_selected(S={rule['S']},L={rule['L']})", "B": B,
                             "bins_per_trial": case["T"], "replay_bins": r["replay_bins"], "macs_total": r["macs_total"],
                             "replay_buffer_KB": r["replay_buffer_bytes"] / KB, "sram_total_KB": r["sram_total"] / KB,
                             "rom_KB": sum(r["rom_bytes"].values()) / KB,
                             "trunk_state_KB": r["sram_bytes"].get("trunk_state (shared with inference)", 0) / KB,
                             "fits_128KB_replay_only": r["replay_buffer_bytes"] / KB <= 128,
                             "fits_128KB_total": r["sram_total"] / KB <= 128,
                             "realtime_s_equiv": r["realtime_seconds_equiv"]})
        # replay-buffer vs count bit-width table (raw storage only)
        for bx in (3, 4, 8):
            for B in BS:
                nb = B * case["T"]
                rows.append({"case": case["name"], "method": f"replay_buffer_only_bx{bx}", "B": B,
                             "bins_per_trial": case["T"], "replay_bins": nb,
                             "replay_buffer_KB": nb * (case["N"] * bx + case["D"] * case["by"] + 1) / 8 / KB,
                             "fits_128KB_replay_only": nb * (case["N"] * bx + case["D"] * case["by"] + 1) / 8 / KB <= 128})
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "cost_p3.csv", index=False)
    (OUTD / "cost_p3_detail.json").write_text(json.dumps(detail, indent=1, default=float))
    cs = [{k: v for k, v in c.items()} for c in cases()]
    (OUTD / "cost_p3_cases.json").write_text(json.dumps(cs, indent=1))
    print(df[df.B.isin([8, 33]) & ~df.method.str.startswith("replay")].to_string(max_colwidth=22))


if __name__ == "__main__":
    main()
