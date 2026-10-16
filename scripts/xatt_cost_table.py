#!/usr/bin/env python
"""Analytic MAC / SRAM cost of one calibration for P0, the fixed-PMA hybrid and XATT (hybrid identity v2).

Counts follow the actual module shapes (``HybridCalibrator``); a MAC is weight x activation (or the product of
two activations inside attention); exp / division are counted separately.  Gram accumulation (N x N per bin) is
shared by all variants (it feeds the descriptors) and listed once.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.hybrid_calib import HybridCalibrator    # noqa: E402
from ssm_decode.population_calib import desc_dim        # noqa: E402

OUT = ROOT / "results/analysis/hybrid_identity_v2_20261017"
K = 10
FD = desc_dim(K)
FDF = FD + 1
H, DK, DV, PH = 4, 16, 8, 16
HID, DCTX, M = 64, 32, 8


def mlp_macs(i, o, hidden=HID, layers=2):
    macs, d = 0, i
    for _ in range(layers):
        macs += d * hidden
        d = hidden
    return macs + d * o


def costs(n, n_unc):
    n_cov = n - n_unc
    rel = mlp_macs(6, 1, 16, 1)
    flat = n * mlp_macs(FDF, K)                                              # P0 / h1 style per-unit generator
    pma_pool = n * (mlp_macs(FDF, HID) + HID * M * 2 + HID * DCTX) + M * DCTX * n   # phi + scores + routing + values + pooling
    pma = n * rel + n_cov * mlp_macs(FDF, K) + n_unc * mlp_macs(FDF + DCTX + 1, K) + pma_pool
    xq = n_unc * FDF * H * DK
    xk = n_cov * FDF * H * DK
    xv = n_cov * (K + FDF) * H * DV
    pairs = n_unc * n_cov
    pair_macs = pairs * (2 * PH + PH * H)                                   # pair MLP
    qk = pairs * H * DK
    av = pairs * H * DV + pairs * H * K                                     # values + attention-averaged prior column
    xatt = (n * rel + n_cov * mlp_macs(FDF, K) + n_unc * mlp_macs(FDF + H * DV + K, K) + xq + xk + xv
            + pair_macs + qk + av)
    exps = pairs * H
    return {"N": n, "N_unc": n_unc, "N_cov": n_cov, "pairs": pairs,
            "P0_fixed_macs": flat, "PMA_hybrid_fixed_macs": pma, "XATT_fixed_macs": xatt,
            "XATT_pairwise_macs": pair_macs + qk + av, "XATT_exps": exps,
            "XATT_online_softmax_words": n_unc * H * (2 + DV + K),
            "gram_accumulate_macs_per_bin": n * n}


def main():
    rows = []
    for n in (96,):
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
            rows.append({"uncovered_frac": frac, **costs(n, int(round(frac * n)))})
    df = pd.DataFrame(rows)
    pars = {k: HybridCalibrator(k, K, FD, **kw).n_params for k, kw in
            (("flat", {}), ("hyb_pma", {"pma_fix": True}), ("hyb_xatt", {}))}
    df["P0_params"], df["PMA_params"], df["XATT_params"] = pars["flat"], pars["hyb_pma"], pars["hyb_xatt"]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "xatt_cost.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    print(pars)


if __name__ == "__main__":
    main()
