#!/usr/bin/env python
"""Order-shuffle diagnostic on source validation (DB6): does the decoder exploit the 12-repetition grasp blocks?

DB6 records every grasp as one block of 12 consecutive repetitions.  A stateful decoder run contiguously over
the recorded order can learn 'same grasp as the previous trial'.  This script decodes each source session (one
causal pass, own table, source statistics) in (a) the recorded order and (b) 3 random trial orders, and scores
the held-out repetitions.  A large drop (a -> b) means the checkpoint relies on the block structure.

usage: db6_order_shuffle_diag.py <out.json> <ckpt> [<ckpt> ...]     (CUDA_VISIBLE_DEVICES=1)
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


def diag(ckpt, dev="cuda:0", n_perm=3):
    model, blob = hb.load_backbone(ckpt, device=dev)
    src = hb.SourceSet([tuple(k) for k in blob["keys"]], tau_ms=blob["cfg"]["tau_ms"],
                       floor=blob["cfg"].get("std_floor_train", hb.TRAIN_STD_FLOOR))
    rng = np.random.default_rng(0)
    orig, shuf = [], []
    for s, S in enumerate(src.sess):
        mu, sd = src.stats[s]
        tab = model.tables[s].detach()
        lg = hb.decode_stream(model.base, torch.from_numpy(src.xz[s]).to(dev), tab)[0]
        orig.append(hb.score_stream(lg, S, src.val_idx[s])["bal_acc"])
        sc = []
        for _ in range(n_perm):
            S3 = hb.reorder_stream(S, rng.permutation(S.n))
            lg = hb.decode_stream(model.base, torch.from_numpy(hb.zscore(S3.x, mu, sd)).to(dev), tab)[0]
            sc.append(hb.score_stream(lg, S3, np.where(np.isin(S3.rep, hb.SOURCE_VAL_REPS))[0])["bal_acc"])
        shuf.append(float(np.mean(sc)))
    return {"ckpt": str(ckpt), "orig_order": float(np.mean(orig)), "shuffled_order": float(np.mean(shuf)),
            "shuffle_trials_in_training": blob["cfg"].get("shuffle_trials", False)}


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
    res = [diag(c) for c in sys.argv[2:]]
    for r in res:
        print(r, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
