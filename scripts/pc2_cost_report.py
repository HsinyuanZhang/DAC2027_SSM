#!/usr/bin/env python
"""A'4: analytic architecture-level cost tables (CPU only, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANA = ROOT / "results/analysis/population_calib_p2_20261015"

from ssm_decode import calib_cost as cc                                    # noqa: E402


def trial_stats():
    from ssm_decode.dandi688_data import load_dandi688_recording
    from ssm_decode.track_a_experiment import usable_trials
    out = {}
    for s in ("sub-M_ses-CO-20150616", "sub-M_ses-CO-20140626"):
        rec = load_dandi688_recording(s)
        tr = usable_trials(rec)
        out[s] = {"n_units": int(rec.neural.shape[1]), "n_trials": len(tr),
                  "mean_bins": float(np.mean([b - a for a, b in tr])),
                  "max_bins": int(max(b - a for a, b in tr))}
    return out


def main():
    ts = trial_stats()
    print(json.dumps(ts, indent=1))
    bins688 = int(round(np.mean([v["mean_bins"] for v in ts.values()])))
    cases = {
        "000688 (N=26, K=10, D=2)": dict(n=26, k=10, d=2, trial_bins=bins688),
        "HD-sEMG (N=256, K=16, D=5)": dict(n=256, k=16, d=5, trial_bins=1250),
    }
    rows = []
    for name, c in cases.items():
        for variant in ("p0", "p2", "p4"):
            for b in (1, 8, 33):
                for acc_bits, table_bits in ((32, 32), (16, 8)):
                    r = cc.full_report(c["n"], c["k"], c["d"], variant, b, c["trial_bins"],
                                       acc_bits=acc_bits, desc_bits=16 if acc_bits == 16 else 32,
                                       table_bits=table_bits)
                    r.update(case=name, acc_bits=acc_bits, table_bits=table_bits)
                    rows.append(r)
    df = pd.DataFrame(rows)
    ANA.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANA / "cost_model.csv", index=False)
    (ANA / "cost_trial_stats.json").write_text(json.dumps(ts, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 60)
    show = df[(df.B == 8) & (df.acc_bits == 32)].set_index(["case", "variant"])
    print(show.T.to_string())
    show = df[(df.B == 8) & (df.acc_bits == 16)].set_index(["case", "variant"])
    print(show[["acc_bytes", "descriptor_bytes", "table_bytes_single", "table_bytes_double"]])


if __name__ == "__main__":
    main()
