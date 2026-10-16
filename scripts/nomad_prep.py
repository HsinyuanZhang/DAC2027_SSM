#!/usr/bin/env python
"""Dump the sessions needed by NoMAD (spint env; the TF env has no pynwb/torch).

Writes results/baselines_20261016/nomad/data/<stratum>/<session>.npz with the binned counts,
standardised velocity (source label statistics of the stratum's AP backbone) and the usable-trial bounds.
Day 0 = the most recent source session of the backbone (same baseline day as the Stabilizer / CCA rows).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_labeled_calibration import D1_POOLS
from ssm_decode.dandi688_data import load_dandi688_recording
from ssm_decode.track_a_experiment import usable_trials

OUT = ROOT / "results/baselines_20261016/nomad/data"


def main(stratum):
    spec = D1_POOLS[stratum]
    blob = torch.load(ROOT / spec["ap"], map_location="cpu", weights_only=False)
    ym, ys = [np.asarray(s, np.float64) for s in blob["source_statistics"][0][2:4]]
    ref = spec["sources"].split(",")[-1]
    d = OUT / stratum
    d.mkdir(parents=True, exist_ok=True)
    for role, name in [("ref", ref)] + [("target", t) for t in spec["targets"]]:
        rec = load_dandi688_recording(name)
        y = np.asarray(rec.behavior, np.float64)
        np.savez_compressed(d / f"{name}.npz", neural=rec.neural.astype(np.float32),
                            y_norm=((y - ym) / ys).astype(np.float32), y_mean=ym, y_std=ys,
                            eval_mask=np.asarray(rec.eval_mask), usable=np.asarray(usable_trials(rec)),
                            role=role)
        print(stratum, role, name, rec.neural.shape)
    (d / "meta.json").write_text(json.dumps({"ref": ref, "targets": spec["targets"]}))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
