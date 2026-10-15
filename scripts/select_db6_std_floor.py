#!/usr/bin/env python
"""Choose the relative std floor of the B-trial z-score on SOURCE data only (DB6 v2).

The floor applies to the per-channel std of the B-trial statistics: std_c <- max(std_c, floor * median_c std_c).
Main comparison: ONE backbone (trained with the minimal floor 1e-3, which never binds on source sessions
before the simulated stress) evaluated at several test floors, so that only the floor changes.  Each is scored
with a source simulation of the B-trial calibration:
  natural : every source session, B=8 simulated calibration (2 draws, training trials), B-trial z-score, own
            read-in table, one-pass contiguous decode, balanced accuracy on the held-out repetitions;
  stress  : S1-S4 source sessions (channel 15 is live there, 2-56 uV, and dies in later target sessions) with
            channel 15 (index 13) replaced over the whole stream by the envelope of a real dead channel (channel 15
            of S5_D1_T1, ~0.02 uV, tiled to the session length; a pure rescale would be undone by z-scoring and
            is not a dead channel), same procedure.
Rule: among floors whose natural score is within 0.005 of the natural score at the smallest floor, the best
stress score (ties -> smaller floor).  Reference candidates (backbones re-trained with a larger floor, evaluated
at their own floor) are reported but not part of the rule (different training runs differ by seed noise).
No target data are used.

usage: select_db6_std_floor.py <out.json> <main ckpt> [<reference ckpt> ...]     (CUDA_VISIBLE_DEVICES=1)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_backbone as hb  # noqa: E402

B, N_DRAWS, DEAD_IDX = 8, 2, 13
DEAD_SOURCE = (5, 1, 1)           # a session whose channel 15 is dead (< 0.2 uV std)
TEST_FLOORS = [1e-3, 0.01, 0.02, 0.05, 0.1, 0.2]


def evaluate(ckpt, dev, test_floors=None):
    """Return one result dict per test floor (default: the checkpoint's own training floor)."""
    model, blob = hb.load_backbone(ckpt, device=dev)
    fl_train = blob["cfg"].get("std_floor_train", hb.TRAIN_STD_FLOOR)
    test_floors = [fl_train] if test_floors is None else test_floors
    dead_env = hb.load_stream(DEAD_SOURCE, blob["cfg"]["tau_ms"]).x[:, DEAD_IDX]
    assert dead_env.std() < 2e-4, "dead-channel donor is not dead"          # mV
    src = hb.SourceSet([tuple(k) for k in blob["keys"]], tau_ms=blob["cfg"]["tau_ms"], floor=fl_train)
    acc = {f: {"nat": [], "stress": []} for f in test_floors}
    for s, key in enumerate(src.keys):
        S = src.sess[s]
        tr, va = src.train_idx[s], src.val_idx[s]
        for variant in ("natural", "stress"):
            if variant == "stress" and key[0] > 4:
                continue
            x = S.x.copy()
            if variant == "stress":
                x[:, DEAD_IDX] = np.resize(dead_env, S.T)
            S2 = hb.Stream(key, {"x": x, "y": S.y, "trial_start": S.tstart, "trial_end": S.tend,
                                 "trial_grasp": S.grasp, "trial_rep": S.rep})
            for d in range(N_DRAWS):
                rng = np.random.default_rng([55501, key[0], key[1] * 10 + key[2], B, d])
                cal = tr[hb.draw_calibration(B, rng, S.grasp[tr], S.rep[tr])]
                for f in test_floors:
                    xz = torch.from_numpy(hb.zscore(x, *hb.input_stats(S2, cal, f))).to(dev)
                    lg = hb.decode_stream(model.base, xz, model.tables[s].detach())[0]
                    acc[f]["nat" if variant == "natural" else "stress"].append(
                        (key[0], hb.score_stream(lg, S2, va)["bal_acc"]))
    return [{"ckpt": str(ckpt), "train_floor": fl_train, "test_floor": f, "source_val_srcnorm": blob["val_bal_acc"],
             "source_val_shuf": blob.get("val_shuf_bal_acc"),
             "natural_all": float(np.mean([v for _, v in a["nat"]])),
             "natural_s1_4": float(np.mean([v for u, v in a["nat"] if u <= 4])),
             "stress_s1_4": float(np.mean([v for _, v in a["stress"]]))} for f, a in acc.items()]


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
    main_res = evaluate(sys.argv[2], "cuda:0", TEST_FLOORS)
    ref_res = [r for c in sys.argv[3:] for r in evaluate(c, "cuda:0")]
    for r in main_res + ref_res:
        print(r, flush=True)
    base = main_res[0]["natural_all"]
    ok = [r for r in main_res if r["natural_all"] >= base - 0.005]
    best = max(ok, key=lambda r: (round(r["stress_s1_4"], 4), -r["test_floor"]))
    Path(sys.argv[1]).write_text(json.dumps({
        "chosen_floor": best["test_floor"], "ckpt": sys.argv[2],
        "rule": "same checkpoint, several test floors: best stress score among floors with natural >= natural(1e-3) - 0.005",
        "main": main_res, "reference_retrained": ref_res}, indent=1))
    print("chosen", best["test_floor"])


if __name__ == "__main__":
    main()
