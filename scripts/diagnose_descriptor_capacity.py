#!/usr/bin/env python
"""Descriptor-capacity diagnostic (review iteration-3, 4.1 step 2).

Question: is the set backbone's remaining source-val gap caused by the
descriptors carrying too little information, or by the fold structure
being unable to express the descriptor-to-table map?

Method: fit an UNCONSTRAINED MLP regressor (no softmax, no slot structure)
from the standardized descriptor block to the AP backbone's learned
per-session tables.  Decode each source session with the reconstructed
tables and report source-val R2 next to the AP reference.

- Reconstruction near the AP score -> descriptors suffice; the fold
  structure limits.
- Reconstruction equally capped -> the descriptors lack information;
  strengthen the descriptor (T4 profiles, richer activity path) first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import debug_experiment as d                       # noqa: E402
from ssm_decode.dandi688_data import load_dandi688_recording       # noqa: E402
from ssm_decode.mamba3_official import build_mamba3_official       # noqa: E402
from ssm_decode.set_attention import (normalize_descriptors,       # noqa: E402
                                      unit_descriptors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-checkpoint", required=True)
    parser.add_argument("--dandi688-sources", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda:0"
    blob = torch.load(args.ap_checkpoint, map_location="cpu", weights_only=False)
    source_ids = [s for s in args.dandi688_sources.split(",") if s]
    if list(blob["source_sessions"]) != source_ids:
        raise SystemExit("AP source order must match --dandi688-sources")
    state = blob["state_dict"]
    ap_tables = [state[f"tables.{i}"].numpy().astype(np.float64)
                 for i in range(len(source_ids))]
    ap_stats = blob["source_statistics"]
    y_stats = np.asarray(ap_stats[0][2]), np.asarray(ap_stats[0][3])
    records = [load_dandi688_recording(s) for s in source_ids]
    raw = [unit_descriptors(r, np.arange(len(r.neural)), lags=2, with_profile=True)[0]
           for r in records]
    normed, _ = normalize_descriptors(raw)

    # Unconstrained MLP: descriptor -> K table entries, shared across units
    # and sessions.  Trained by Adam on the AP tables with dropout and noise
    # on the descriptor so the map is not a memorisation.
    x = np.concatenate(normed)
    y = np.concatenate([t.T for t in ap_tables])
    mlp = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, y.shape[1])).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = rng.integers(0, len(xt), size=min(4096, len(xt)))
        xb = xt[idx]
        if rng.random() < 0.3:                      # profile-block dropout
            xb = xb.clone()
            xb[:, :6] = 0.0
        xb = xb + 0.05 * torch.randn_like(xb)
        loss = (mlp(xb) - yt[idx]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.inference_mode():
        recon = [mlp(torch.from_numpy(n.astype(np.float32)).to(device))
                 .cpu().numpy().astype(np.float64).T for n in normed]

    trunk = build_mamba3_official(blob["args"]["k"], records[0].behavior.shape[1],
                                  blob["args"]["width"], blob["args"]["layers"],
                                  blob["args"]["state_size"], 0.0).to(device)
    trunk.load_state_dict({k.removeprefix("base."): v for k, v in state.items()
                           if k.startswith("base.")})
    trunk.eval()
    per_session, scores = {}, []
    with torch.inference_mode():
        for index, record in enumerate(records):
            table = recon[index]
            x_mean, x_std = np.asarray(ap_stats[index][0]), np.asarray(ap_stats[index][1])
            xz = ((record.neural - x_mean) / x_std).astype(np.float32)
            tr, va = d._trial_split(record)
            y_arr = np.full_like(record.behavior, np.nan)
            mask = np.zeros(len(xz), dtype=bool)
            for a, b in va:
                y_arr[a:b] = (record.behavior[a:b] - y_stats[0]) / y_stats[1]
                mask[a:b] = record.eval_mask[a:b]
            window = tuple(torch.from_numpy(v) for v in (xz, y_arr, mask))
            selection = d._validation_selection([window], 128, 1024)[0]
            xb2, yb2, mb2 = d._right_aligned([window], selection, 128)
            z = xb2.to(device) @ torch.from_numpy(table.astype(np.float32)).to(device).t()
            decode = trunk(z)[:, -1].cpu()
            good = mb2[:, -1] & torch.isfinite(yb2[:, -1]).all(-1)
            r2 = d._r2(yb2[:, -1][good] * y_stats[1] + y_stats[0],
                       decode[good] * y_stats[1] + y_stats[0])
            per_session[record.session[-10:]] = float(r2)
            scores.append(float(r2))
    out = {"ap_source_val": float(blob["best_source_val_trunk_r2"]),
           "reconstructed_mean_r2": float(np.mean(scores)),
           "regressor": "MLP no-softmax", "hidden": args.hidden, "steps": args.steps,
           "final_train_mse": float(loss),
           "per_session": per_session,
           "read": "near AP score => descriptors suffice, fold structure limits; "
                   "capped equally => descriptors lack information"}
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
