#!/usr/bin/env python
"""Train one DB6 D-A source backbone (days 1-2 of all subjects) on random-offset windows of the continuous
session streams (v2). Source-only; GPU1 only."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.db6_backbone import SourceSet, source_keys, train_backbone  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")  # physical GPU1 via CUDA_VISIBLE_DEVICES=1
for n, d in [("k", 8), ("width", 128), ("layers", 2), ("state-size", 32), ("batch", 16),
             ("max-steps", 15000), ("val-every", 250), ("patience", 8), ("seed", 0),
             ("win-min", 512), ("win-max", 1024), ("burn", 128)]:
    ap.add_argument("--" + n, type=int, default=d)
for n, d in [("dropout", 0.1), ("lr", 1e-3), ("weight-decay", 1e-2), ("lam-head", 0.5),
             ("lam-white", 0.1), ("gain-sigma", 0.3), ("offset-sigma", 0.1), ("channel-dropout", 0.05),
             ("tau-ms", 160.0), ("p-start", 0.05), ("std-floor", 1e-3)]:
    ap.add_argument("--" + n, type=float, default=d)
ap.add_argument("--table-init", default="supervised")
ap.add_argument("--shuffle-trials", type=int, default=1, help="1 = rebuild session streams with random trial order")
ap.add_argument("--select-on", default="shuf", choices=["shuf", "orig"])
ap.add_argument("--reshuffle-every", type=int, default=1000)
a = ap.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
kw = {k.replace("-", "_"): v for k, v in vars(a).items() if k not in ("out", "device", "tau_ms", "std_floor")}
kw["shuffle_trials"] = bool(kw["shuffle_trials"])
src = SourceSet(source_keys(), tau_ms=a.tau_ms, floor=a.std_floor)
print(train_backbone(src, a.out, device=a.device, **kw), flush=True)
