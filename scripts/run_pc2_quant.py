#!/usr/bin/env python
"""A'3: calibration-path quantisation / approximate computing (evaluation only).

Loads the trained P1 calibrators (P0, P4 m=8; aligned prior; pool O; 2 seeds)
and re-evaluates them with fake-quantised weights / accumulators /
descriptors / output table / LUT softmax.  No retraining, no gradient, no
change to the trunk.  Reports ``r2`` for every configuration plus the change
against the FP configuration on the identical trial subset: ``d_r2`` and the
RMS change of the prediction (``rms_delta``; isolates the quantisation
perturbation from the noise cross-term of R2).

Rows are written for SOURCE validation sessions (kind=val, leave-one-out
prior) and for the stratum targets (kind=target).  Scale choices (accumulator
and descriptor steps) use SOURCE training sessions only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("PC_OUT", str(ROOT / "results/population_calib_p2_20261015")))
os.environ["PC_OUT"] = str(OUT)

from scripts import run_population_calib as rpc                           # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402
from ssm_decode.calib_quant import (ACC_GROUPS, acc_steps, column_steps,  # noqa: E402
                                    make_lut_softmax, quant_table,
                                    quantize_calibrator, quantized_stats, to_fixed)
from ssm_decode.dandi688_data import load_dandi688_recording             # noqa: E402
from ssm_decode.population_calib import PopStats, describe, norm_stats   # noqa: E402

BUDGETS = (1, 8, 33)
ALL_KEYS = tuple(k for g in ACC_GROUPS.values() for k in g)


def make_configs(variant: str):
    """name -> dict(w, acc, acc_keys, desc, tab, lut)."""
    c = {"fp": {}}
    for w in ("w8", "w4", "tern"):
        c[w] = {"w": w}
    for b in (16, 12, 8):
        c[f"acc{b}"] = {"acc": b}
        c[f"desc{b}"] = {"desc": b}
    for b in (16, 12, 8):                       # exponent follows the trial count B (see calibrate_cfg)
        c[f"acc{b}_bs"] = {"acc": b, "bscale": True}
    for g in ACC_GROUPS:
        for b in (12, 8):
            c[f"acc{b}_{g}"] = {"acc": b, "acc_keys": g}
    for b in (8, 4):
        c[f"tab{b}"] = {"tab": b}
    lut = variant.startswith("p3") or variant.startswith("p4")
    if lut:
        c["lut"] = {"lut": "exact"}
        c["lut_rlut"] = {"lut": "lut"}
    ext = {"lut": "exact"} if lut else {}
    c["combo_w8_acc16_desc16_tab8"] = {"w": "w8", "acc": 16, "desc": 16, "tab": 8, **ext}
    c["combo_w4_acc12_desc12_tab8"] = {"w": "w4", "acc": 12, "desc": 12, "tab": 8, **ext}
    c["combo_tern_acc16_desc16_tab4"] = {"w": "tern", "acc": 16, "desc": 16, "tab": 4, **ext}
    return c


class Scales:
    """Fixed-point steps fitted on SOURCE training episodes only."""

    def __init__(self, world, n_ep=24):
        stats, dmats, rmats = [], [], []
        for name in world.train_ids:
            sd, prior = world.sessions[name], world.priors[name]
            for e in range(n_ep // len(world.train_ids) + 1):
                for B in (1, 8, 33):
                    idx = rpc.draw_trials(f"qscale-{name}", B, e, sd.bank.n_trials)
                    st = PopStats.from_trials(sd.bank, idx)
                    d, r, _, _ = describe(st, prior)
                    dmats.append(d), rmats.append(r)
                    if B == 33:
                        stats.append(st)
        self.acc = {b: acc_steps(stats, b) for b in (16, 12, 8)}
        self.d = {b: column_steps(dmats, b) for b in (16, 12, 8)}
        self.r = {b: column_steps(rmats, b) for b in (16, 12, 8)}


def calibrate_cfg(calib_by_w, scales, sd, prior, idx, cfg):
    bank = sd.bank
    if "acc" in cfg:
        keys = ACC_GROUPS[cfg["acc_keys"]] if "acc_keys" in cfg else ALL_KEYS
        steps = scales.acc[cfg["acc"]]
        if cfg.get("bscale"):
            # block exponent set from B: sums grow ~ B, so the power-of-two step shrinks by
            # 2^ceil(log2(B/33)) relative to the B = 33 source calibration (B known to the
            # controller), keeping 2 bits (4x) of headroom for atypically long trials
            sh = 2.0 ** (math.ceil(math.log2(max(len(idx), 1) / 33.0)) + 2)
            steps = {k: v * min(sh, 1.0) for k, v in steps.items()}
        stats = quantized_stats(bank, idx, steps, cfg["acc"], keys)
    else:
        stats = PopStats.from_trials(bank, idx)
    with torch.no_grad():
        d, r, mean, std = describe(stats, prior)
        if "desc" in cfg:
            b = cfg["desc"]
            d = to_fixed(d.double(), scales.d[b].double(), b).float()
            r = to_fixed(r.double(), scales.r[b].double(), b).float()
        calib = calib_by_w[cfg.get("w")]
        calib.softmax_fn = make_lut_softmax(cfg["lut"]) if "lut" in cfg else torch.softmax
        table = calib(d, r, prior.to(torch.float32))
        calib.softmax_fn = torch.softmax
        if "tab" in cfg:
            table = quant_table(table, cfg["tab"])
    return table, mean, std


def run_point(world, sd, prior, calib_by_w, scales, idx, configs, meta, f, skip=frozenset()):
    q = sd.query_mask(idx)
    ym, ys = world.label_stats
    fp_pred = None
    for name, cfg in configs.items():           # "fp" is first (always recomputed: baseline)
        if name in skip and name != "fp":
            continue
        table, mean, std = calibrate_cfg(calib_by_w, scales, sd, prior, idx, cfg)
        pred = rpc.decode_full(world, sd, table, mean, std)
        r2, _ = rpc.r2_physical(world, sd, pred, q)
        if name == "fp":
            fp_pred, r2_fp = pred, r2
        pq = pred[q].double().cpu().numpy() * ys
        pf = fp_pred[q].double().cpu().numpy() * ys
        rms = float(np.sqrt(((pq - pf) ** 2).mean()))
        pstd = float(np.sqrt(((pf - pf.mean(0, keepdims=True)) ** 2).mean()))
        row = {**meta, "cfg": name, "r2": r2, "r2_fp": r2_fp, "d_r2": r2 - r2_fp,
               "rms_delta": rms, "pred_std_fp": pstd, "rel_rms": rms / max(pstd, 1e-12),
               "finite": bool(np.isfinite(pq).all())}
        if name not in skip:
            f.write(json.dumps(row) + "\n")
    f.flush()


def make_configs_qat():
    return {"fp": {}, "tern": {"w": "tern"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", action="store_true",
                    help="evaluate the ternary-QAT fine-tuned checkpoints (P2 runs dir)")
    ap.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    ap.add_argument("--variants", nargs="*", default=["p0", "p4_m8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    world = rpc.World(args.stratum, "O", device, load_sessions=True)
    scales = Scales(world)
    path = OUT / "quant" / (f"rows_qat_{args.stratum}.jsonl" if args.qat else f"rows_{args.stratum}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["variant"], r["seed"], r["kind"], r["session"], r["B"], r["draw"], r["cfg"]))
    mk = (lambda v: make_configs_qat()) if args.qat else make_configs
    all_cfg = {v: set(mk(v)) for v in args.variants}
    val_name = world.val_ids[-1]
    targets = {t: world.target(t) for t in D1_POOLS[args.stratum]["targets"]}
    t0 = time.perf_counter()
    with path.open("a") as f:
        for variant in args.variants:
            configs = mk(variant)
            for seed in args.seeds:
                run_dir = (OUT if args.qat else rpc.P1_OUT) / "runs" / f"{args.stratum}_O" / (
                    variant + "_qat_tern" if args.qat else variant) / f"seed{seed}"
                calib = rpc.load_run(run_dir, world)[0]
                calib_by_w = {None: calib}
                for w in ("w8", "w4", "tern"):
                    calib_by_w[w] = quantize_calibrator(calib, w)
                    calib_by_w[w].eval()
                for kind, name, sd, prior in ([("val", val_name, world.sessions[val_name],
                                                world.priors[val_name])] +
                                              [("target", t, *targets[t]) for t in targets]):
                    for B in BUDGETS:
                        for draw in range(args.draws):
                            skip = frozenset(c for c in all_cfg[variant]
                                             if (variant + ("_qat" if args.qat else ""), seed, kind, name, B, draw, c) in done)
                            if len(skip) == len(all_cfg[variant]):
                                continue
                            salt = f"qval-{name}" if kind == "val" else name
                            idx = rpc.draw_trials(salt, B, draw, sd.bank.n_trials)
                            meta = {"stratum": args.stratum, "variant": variant + ("_qat" if args.qat else ""), "seed": seed,
                                    "kind": kind, "session": name, "B": B, "draw": draw}
                            run_point(world, sd, prior, calib_by_w, scales, idx, configs, meta, f, skip)
                    print(f"[quant {args.stratum}] {variant} s{seed} {kind} {name[-8:]} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
