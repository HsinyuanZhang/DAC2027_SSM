#!/usr/bin/env python
"""Train one Hyser H-A source backbone (session1 of all 20 subjects). Source-only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.hyser_backbone import SourceSet, train_backbone  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")  # physical GPU1 via CUDA_VISIBLE_DEVICES=1
for n, d in [("k", 16), ("width", 128), ("layers", 2), ("state-size", 32), ("batch", 16),
             ("max-steps", 4000), ("val-every", 50), ("patience", 12), ("seed", 0)]:
    ap.add_argument("--" + n, type=int, default=d)
for n, d in [("dropout", 0.1), ("lr", 1e-3), ("weight-decay", 1e-2), ("lam-head", 0.5),
             ("lam-white", 0.1), ("gain-sigma", 0.3), ("offset-sigma", 0.1), ("channel-dropout", 0.05)]:
    ap.add_argument("--" + n, type=float, default=d)
ap.add_argument("--table-init", default="supervised")
a = ap.parse_args()
kw = {k.replace("-", "_"): v for k, v in vars(a).items() if k not in ("out", "device")}
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
src = SourceSet()
print(train_backbone(src, a.out, device=a.device, **kw), flush=True)
