#!/usr/bin/env python
"""Quantify descriptor-level population coupling on the 12 real drift-curve targets (v1 review (a)).

For every target (natural coverage, B=8, draw 0): decorrelate ONE covered unit j (its Gram row / column
set to the uncorrelated value n*mean_j*mean_k, i.e. a perturbation of the streamed Gram only) and measure
the change of the latent-correlation descriptor features (``corr_columns``) of every UNCOVERED unit i.
Also the same with j's prior column zeroed (j treated as uncovered).  Reports mean / max over (i, j) of
max_k |delta corr_ik|.  Pure CPU.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_population_calib as rpc                               # noqa: E402
from scripts.run_drift_curve import PREFIX, STRATUM, TARGET_DATES              # noqa: E402
from ssm_decode.hybrid_calib import covered_flags, mask_prior                  # noqa: E402
from ssm_decode.population_calib import PopStats, corr_columns, describe       # noqa: E402

OUT = ROOT / "results/analysis/hybrid_identity_v2_20261017"


def main():
    dev = torch.device("cpu")
    world = rpc.World(STRATUM, "O", dev, load_sessions=False)
    cc = corr_columns(world.k)
    rows = []
    for d in TARGET_DATES:
        sd, prior = world.target(PREFIX + d)
        cov = covered_flags(world.orig_elec, sd.electrodes)
        prior = mask_prior(prior, cov)
        idx = rpc.draw_trials(PREFIX + d, 8, 0, sd.bank.n_trials)
        stats = PopStats.from_trials(sd.bank, idx)
        d0 = describe(stats, prior)[0]
        iu, ic = np.flatnonzero(~cov), np.flatnonzero(cov)
        if len(iu) == 0:
            continue
        g_dec, g_drop, own = [], [], []
        n = stats.d["n"]
        for j in ic:
            g = {k: v.clone() for k, v in stats.d.items()}
            mu = g["sx"] / n
            fix = n * mu[j] * mu
            gram = g["gram"]
            keep = gram[j, j].clone()
            gram[j, :] = fix
            gram[:, j] = fix
            gram[j, j] = keep
            d1 = describe(PopStats(g, stats.n_trials), prior)[0]
            g_dec.append((d1[iu][:, cc] - d0[iu][:, cc]).abs().max(1).values)
            p2 = prior.clone()
            p2[:, j] = 0
            d2 = describe(stats, p2)[0]
            g_drop.append((d2[iu][:, cc] - d0[iu][:, cc]).abs().max(1).values)
        a, b = torch.stack(g_dec), torch.stack(g_drop)
        rows.append({"target": d, "n_units": sd.n_units, "n_uncovered": len(iu), "n_covered": len(ic),
                     "decorrelate_mean": float(a.mean()), "decorrelate_max": float(a.max()),
                     "drop_prior_mean": float(b.mean()), "drop_prior_max": float(b.max()),
                     "corr_feature_abs_mean_uncovered": float(d0[iu][:, cc].abs().mean())})
        print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "descriptor_coupling.csv", index=False)


if __name__ == "__main__":
    main()
