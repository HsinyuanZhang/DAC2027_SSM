#!/usr/bin/env python
"""Analytic calibration-path cost (MAC / SRAM) for the sEMG calibrators; writes json + csv."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from ssm_decode.semg_calib import ALL_VARIANTS, SemgSpec, cost_model  # noqa: E402

CASES = {
    "hyser_N256_K16": (SemgSpec("hyser", 256, 16, 5, (0, 2, 4, 6)), 1250, [1, 2, 5, 10]),
    "db6_N14_K8": (SemgSpec("db6", 14, 8, 8, (0,)), 424, [1, 2, 4, 8, 16]),
}
out, rows = {}, []
for name, (spec, bins, budgets) in CASES.items():
    for v in ALL_VARIANTS:
        for b in budgets:
            c = cost_model(spec, v, bins, b)
            out[f"{name}|{v}|B{b}"] = c
            rows.append({"case": name, "variant": v, "B": b, "bins_per_trial": bins,
                         "desc_dim": spec.desc_dim, "accumulate_macs": c["accumulate_macs"],
                         "macs_per_bin": c["accumulate_macs_per_bin"], "finalize_macs": c["finalize_macs"],
                         "calibrator_fixed_macs": c["calibrator_fixed_macs"],
                         "fixed_macs_per_channel": c["calibrator_fixed_macs_per_channel"], "total_macs": c["total_macs"],
                         "fixed_weight_fraction": round(c["fixed_weight_fraction"], 5),
                         "calibrator_weights": c["calibrator_weights"], "calibrator_params_total": c["calibrator_params_total"],
                         "sram_total_full_gram_B": c["sram_total_bytes_full_gram"],
                         "sram_total_tri_gram_B": c["sram_total_bytes_tri_gram"],
                         "sram_total_full_gram_with_trial_copy_B": c["sram_total_bytes_full_gram_with_trial_copy"],
                         "sram_total_tri_gram_with_trial_copy_B": c["sram_total_bytes_tri_gram_with_trial_copy"],
                         "per_trial_state_B": c["sram_bytes"]["per_trial_state"],
                         "per_trial_full_copy_full_gram_B": c["per_trial_full_copy_bytes"]["full_gram"],
                         "gram_full_B": c["sram_bytes"]["gram_full"], "table_B": c["sram_bytes"]["table"],
                         "gramfree_macs_per_bin_analytic": c["gram_free_estimate"]["accumulate_macs_per_bin"],
                         "gramfree_acc_sram_B_analytic": c["gram_free_estimate"]["sram_accumulators_bytes"]})
dest = ROOT / "results" / "analysis" / "semg_calib_20261016"
dest.mkdir(parents=True, exist_ok=True)
(dest / "cost_model.json").write_text(json.dumps(out, indent=1))
pd.DataFrame(rows).to_csv(dest / "cost_model.csv", index=False)
print(pd.DataFrame(rows).query("B in [1,10,16] and variant in ['p0','p2','p4']").T.to_string())
