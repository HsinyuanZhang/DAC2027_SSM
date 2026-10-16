#!/usr/bin/env python
"""v1 review (b): collapse monitors of the v1 PMA checkpoints (h2p, h3p; seeds 0/1) on the 12 real targets
(natural coverage, B in {2, 8, 33}, draw 0).  Pure CPU, label-free diagnostics of the pooling weights."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_population_calib as rpc                               # noqa: E402
from scripts.run_drift_curve import PREFIX, STRATUM, TARGET_DATES              # noqa: E402
from scripts.run_hybrid_identity import VARIANTS, build                        # noqa: E402
from ssm_decode.hybrid_calib import covered_flags, mask_prior                  # noqa: E402
from ssm_decode.population_calib import PopStats, desc_dim, describe           # noqa: E402

RAW = ROOT / "results/hybrid_identity_20261016"
OUT = ROOT / "results/analysis/hybrid_identity_20261016"


def main():
    dev = torch.device("cpu")
    world = rpc.World(STRATUM, "O", dev, load_sessions=False)
    models = {}
    for name in ("h2p", "h3p"):
        for seed in (0, 1):
            blob = torch.load(RAW / "runs" / name / f"seed{seed}" / "best.pt", map_location=dev, weights_only=False)
            m = build(name, world.k, desc_dim(world.k))
            missing, unexpected = m.load_state_dict(blob["calibrator"], strict=False)
            assert set(missing) <= {"corr_mask"} and not unexpected, (missing, unexpected)
            m.eval()
            models[(name, seed)] = m
    rows = []
    for d in TARGET_DATES:
        sd, prior = world.target(PREFIX + d)
        cov = covered_flags(world.orig_elec, sd.electrodes)
        prior = mask_prior(prior, cov)
        for B in (2, 8, 33):
            idx = rpc.draw_trials(PREFIX + d, B, 0, sd.bank.n_trials)
            stats = PopStats.from_trials(sd.bank, idx)
            dd, rr, _, _ = describe(stats, prior)
            for (name, seed), m in models.items():
                rows.append({"variant": name, "seed": seed, "target": d, "B": B, "n_units": sd.n_units,
                             **m.diagnostics(dd, rr, prior.float(), cov)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v1_pma_collapse_monitors.csv", index=False)
    print(df.groupby("variant").mean(numeric_only=True).to_string())
    print(df.groupby("variant").agg(["min", "max"]).drop(columns=["seed", "B", "n_units"], level=0).to_string())


if __name__ == "__main__":
    main()
