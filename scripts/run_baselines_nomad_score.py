#!/usr/bin/env python
"""Score NoMAD predictions (scripts/run_baselines_nomad_tf.py) on the shared (target, B, draw) query masks.

Usage: run_baselines_nomad_score.py <stratum> <variant>   (variant = pooled | units_elec | pooled_noinit | ...;
predictions in nomad/preds_<variant>/, rows to eval/nomad2_<variant>_<stratum>.jsonl with method ``nomad_<variant>``).
The baseline-day controls (<ref>_s*.npz = unshuffled, <ref>_shuffled_s*.npz = columns permuted) are scored on all
usable trials of the baseline day and written to nomad/controls_<variant>_<stratum>.json.

NoMAD's alignment is trained once per target on the full unlabeled recording (B-independent); the rows below
differ across (B, draw) only through the query mask.  Contiguous causal decoding: LFADS runs on causal 30-bin
sliding windows over the whole recording and the lag-3 ridge decoder reads the last bin of each window.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_labeled_calibration import D1_POOLS
from scripts.run_population_calib import draw_trials
from ssm_decode.dandi688_data import load_dandi688_recording
from ssm_decode.eval_checks import check_physical_units
from ssm_decode.population_calib import SessionData
from scripts.run_baselines_readin import native_query_mask

BASE = ROOT / "results/baselines_20261016"
B_GRID = (1, 4, 8, 16, 33)


def score_controls(stratum, variant):
    ref = D1_POOLS[stratum]["sources"].split(",")[-1]
    rec = load_dandi688_recording(ref)
    sd = SessionData(rec, (np.zeros(2), np.ones(2)), torch.device("cpu"), name=ref)
    truth = rec.behavior.astype(np.float64)
    q0 = (sd.valid & (sd.trial_id >= 0)).numpy()
    res = []
    for f in sorted((BASE / f"nomad/preds_{variant}").glob(f"{ref}*_s*.npz")):
        z = np.load(f)
        pred = z["pred"].astype(np.float64)
        q = q0 & np.isfinite(pred).all(1)
        q[:30] = False
        p, t = pred[q], truth[q]
        res.append({"file": f.name, "shuffled": "shuffled" in f.name, "r2_all_usable": float(1 - ((t - p) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()),
                    "epochs_align": int(z["epochs"])})
    if res:
        (BASE / f"nomad/controls_{variant}_{stratum}.json").write_text(json.dumps({"session": ref, "runs": res}, indent=1))
    return res


def main(stratum, variant):
    out = BASE / "eval" / f"nomad2_{variant}_{stratum}.jsonl"
    out.write_text("")
    print("controls", score_controls(stratum, variant))
    for target in D1_POOLS[stratum]["targets"]:
        rec = load_dandi688_recording(target)
        sd = SessionData(rec, (np.zeros(2), np.ones(2)), torch.device("cpu"), name=target)
        truth = rec.behavior.astype(np.float64)
        files = sorted((BASE / f"nomad/preds_{variant}").glob(f"{target}_s*.npz"))
        if not files:
            print("missing preds", target)
        for f in files:
            seed = int(f.stem.rsplit("_s", 1)[1])
            z = np.load(f)
            pred = z["pred"].astype(np.float64)
            ok = np.isfinite(pred).all(1)
            ok[:30] = False
            for B in B_GRID:
                for draw in range(8):
                    idx = draw_trials(target, B, draw, sd.bank.n_trials)
                    q = sd.query_mask(idx).numpy() & ok
                    p, t = pred[q], truth[q]
                    r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
                    unit = check_physical_units(p, t)
                    qn = native_query_mask(sd, idx).numpy() & ok
                    pn, tn = pred[qn], truth[qn]
                    r2n = 1 - ((tn - pn) ** 2).sum() / ((tn - tn.mean(0)) ** 2).sum()
                    row = {"stratum": stratum, "method": f"nomad_{variant}", "target": target, "seed": seed, "B": B, "draw": draw,
                           "r2": float(r2), "r2_native": float(r2n), "status": "OK", "n_query": int(q.sum()), "unit_ratio": unit["unit_ratio"],
                           "unit_check_passed": unit["passed"], "epochs_align": int(z["epochs"])}
                    with out.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
            print(stratum, target, "seed", seed, "scored")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
