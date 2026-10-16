#!/usr/bin/env python
"""P-prior / P0 on the SAME (target, B, draw) subsets, scored on the full query set AND on APST's native query
subset (bins whose 50-bin causal window lies fully inside one trial).  Re-evaluates the P1 checkpoints
(``results/population_calib_20261015/runs/<stratum>_O/p0/seed{0,1}``); the full-query R2 must reproduce the stored
P1 rows (checked and written to the output).  Target side: no gradients.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_labeled_calibration import D1_POOLS
from scripts import run_population_calib as rpc
from scripts.run_baselines_readin import native_query_mask
from ssm_decode.population_calib import PopStats, norm_stats

BASE = ROOT / "results/baselines_20261016"
B_GRID = (1, 4, 8, 16, 33)


def main(stratum, device="cuda:0"):
    dev = torch.device(device)
    world = rpc.World(stratum, "O", dev, load_sessions=False)
    out = BASE / "eval" / f"native_p_{stratum}.jsonl"
    out.write_text("")
    stored = {}
    for seed in (0, 1):
        f = rpc.P1_OUT / "runs" / f"{stratum}_O" / "p0" / f"seed{seed}" / "eval_rows.jsonl"
        for line in f.read_text().splitlines():
            r = json.loads(line)
            stored[(seed, r["target"], r["B"], r["draw"])] = r["r2"]
    calibs = {s: rpc.load_run(rpc.P1_OUT / "runs" / f"{stratum}_O" / "p0" / f"seed{s}", world)[0] for s in (0, 1)}
    for target in D1_POOLS[stratum]["targets"]:
        sd, prior = world.target(target)
        for B in B_GRID:
            for draw in range(8):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                stats = PopStats.from_trials(sd.bank, idx)
                variants = {"p_prior": (None, 0)}
                variants.update({f"p0_s{s}": (calibs[s], s) for s in (0, 1)})
                for vname, (calib, seed) in variants.items():
                    name = vname.split("_s")[0] if vname.startswith("p0") else vname
                    with torch.no_grad():
                        if calib is None:
                            mean, std = norm_stats(stats)
                            table = prior.float()
                        else:
                            res = calib.calibrate(stats, prior)
                            mean, std, table = res.mean, res.std, res.table
                    pred = rpc.decode_full(world, sd, table, mean, std)
                    r2, _ = rpc.r2_physical(world, sd, pred, sd.query_mask(idx))
                    r2n, _ = rpc.r2_physical(world, sd, pred, native_query_mask(sd, idx))
                    ref = stored.get((seed, target, B, draw)) if name == "p0" else None
                    row = {"stratum": stratum, "method": name, "target": target, "seed": seed, "B": B, "draw": draw,
                           "r2": float(r2), "r2_native": float(r2n), "stored_p1_r2": ref, "status": "OK"}
                    with out.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
        print(stratum, target, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
