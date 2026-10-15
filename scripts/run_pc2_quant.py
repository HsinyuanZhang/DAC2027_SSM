#!/usr/bin/env python
"""A'3: calibration-path quantisation / approximate computing (evaluation only).

Loads the trained P1 calibrators (P0, P4 m=8; aligned prior; pool O; 2 seeds)
and re-evaluates them with fake-quantised weights / accumulators /
descriptors / output table / LUT softmax.  No retraining, no gradient, no
change to the trunk.  Reports ``r2`` for every configuration plus the change
against the FP configuration on the identical trial subset: ``d_r2`` and the
RMS change of the prediction (``rms_delta``; isolates the quantisation
perturbation from the noise cross-term of R2).

Review fix (2026-10-16): accumulators are now modelled per add (every per-bin
increment is rounded to the register step and added with saturation;
``ssm_decode.calib_quant.perbin_stats``); the old once-per-trial rounding is kept as
the labelled two-level variant ``acc*_2lvl``.  Bin counters (n, ac_n, yx_n, cdir)
are exact integer counters and are never quantised.  Accumulator statistics only
depend on (session, trials, config), so they are precomputed on CPU
(``--stage stats``, parallel) and cached; ``--stage eval`` decodes on the GPU.

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
from ssm_decode.calib_quant import (DATA_GROUPS, acc_steps, column_steps,  # noqa: E402
                                    make_lut_softmax, perbin_stats_multi, quant_table,
                                    quantize_calibrator, to_fixed, twolevel_stats)
from ssm_decode.dandi688_data import load_dandi688_recording             # noqa: E402
from ssm_decode.population_calib import PopStats, describe, norm_stats   # noqa: E402

BUDGETS = (1, 8, 33)
ACC_WIDTHS = (8, 12, 14, 16, 20, 24)         # all data keys, per-add
GROUP_WIDTHS = (8, 12, 14, 16)               # one accumulator group at a time (add 20 if a group is unsafe at 16)
BS_WIDTHS = (12, 14, 16)                     # B-adaptive exponent (needs final B in advance)
TWOLEVEL_WIDTHS = (12, 16)                   # wide per-trial staging + narrow running sum
STEP_WIDTHS = tuple(sorted(set(ACC_WIDTHS) | set(GROUP_WIDTHS) | {20}))
QOUT = Path(os.environ.get("PC_QUANT_OUT", str(OUT / "fix" / "quant")))


def acc_cfg_name(cfg: dict) -> str:
    if cfg.get("acc_mode") == "twolevel":
        return f"acc{cfg['acc']}_2lvl"
    return f"acc{cfg['acc']}" + (f"_{cfg['acc_keys']}" if "acc_keys" in cfg else "") + (
        "_bs" if cfg.get("bscale") else "")


def make_configs(variant: str):
    """name -> dict(w, acc, acc_keys, acc_mode, bscale, desc, tab, lut).

    ``accN`` = per-add fixed point (default hardware model), counters exact;
    ``accN_<group>`` = only that data group in fixed point; ``accN_bs`` = B-adaptive exponent;
    ``accN_2lvl`` = two-level design (wide per-trial staging, rounded once per trial)."""
    c = {"fp": {}}
    for w in ("w8", "w4", "tern"):
        c[w] = {"w": w}
    for b in (16, 12, 8):
        c[f"desc{b}"] = {"desc": b}
    for b in ACC_WIDTHS:
        c[f"acc{b}"] = {"acc": b}
    for b in BS_WIDTHS:
        c[f"acc{b}_bs"] = {"acc": b, "bscale": True}
    for g in DATA_GROUPS:
        for b in GROUP_WIDTHS:
            c[f"acc{b}_{g}"] = {"acc": b, "acc_keys": g}
    for b in TWOLEVEL_WIDTHS:
        c[f"acc{b}_2lvl"] = {"acc": b, "acc_mode": "twolevel"}
    for b in (8, 4):
        c[f"tab{b}"] = {"tab": b}
    lut = variant.startswith("p3") or variant.startswith("p4")
    if lut:
        c["lut"] = {"lut": "exact"}
        c["lut_rlut"] = {"lut": "lut"}
    ext = {"lut": "exact"} if lut else {}
    c["combo_w8_acc16_desc16_tab8"] = {"w": "w8", "acc": 16, "desc": 16, "tab": 8, **ext}
    c["combo_w8_acc20_desc16_tab8"] = {"w": "w8", "acc": 20, "desc": 16, "tab": 8, **ext}
    c["combo_w4_acc16_desc12_tab8"] = {"w": "w4", "acc": 16, "desc": 12, "tab": 8, **ext}
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
        self.acc = {b: acc_steps(stats, b) for b in STEP_WIDTHS}
        self.d = {b: column_steps(dmats, b) for b in (16, 12, 8)}
        self.r = {b: column_steps(rmats, b) for b in (16, 12, 8)}


def acc_steps_for(scales, cfg, n_trials):
    steps = scales.acc[cfg["acc"]]
    if cfg.get("bscale"):
        # block exponent set from B: sums grow ~ B, so the power-of-two step shrinks by
        # 2^ceil(log2(B/33)) relative to the B = 33 source calibration (B must be known to the
        # controller IN ADVANCE; an anytime stream would have to rescale), keeping 2 bits (4x)
        # of headroom for atypically long trials
        sh = 2.0 ** (math.ceil(math.log2(max(n_trials, 1) / 33.0)) + 2)
        steps = {k: v * min(sh, 1.0) for k, v in steps.items()}
    return steps


def compute_acc_stats(sd, idx, scales, cfgs: dict):
    """{cfg name: (PopStats, info)} for the accumulator configs ``cfgs`` {name: cfg}."""
    out, multi = {}, {}
    for name, cfg in cfgs.items():
        steps = acc_steps_for(scales, cfg, len(idx))
        keys = DATA_GROUPS[cfg["acc_keys"]] if "acc_keys" in cfg else None
        if cfg.get("acc_mode") == "twolevel":
            out[name] = (twolevel_stats(sd.bank, idx, steps, cfg["acc"], keys), {"sat_events": {}})
        else:
            multi[name] = (steps, cfg["acc"], keys)
    if multi:
        out.update(perbin_stats_multi(sd, idx, multi))
    return out


def calibrate_cfg(calib_by_w, scales, sd, prior, idx, cfg, acc_cache=None):
    bank = sd.bank
    sat = 0
    if "acc" in cfg:
        name = acc_cfg_name(cfg)
        if acc_cache is not None and name in acc_cache:
            stats, info = acc_cache[name]
        else:
            stats, info = compute_acc_stats(sd, idx, scales, {name: cfg})[name]
        stats = PopStats({k: v.to(sd.x.device) for k, v in stats.d.items()}, stats.n_trials)
        sat = int(sum(info["sat_events"].values()))
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
    return table, mean, std, sat


DECODE_BATCH = 8


def decode_batch(world, sd, tables, means, stds):
    """Decode several tables in one trunk pass (bit-identical to ``rpc.decode_full``)."""
    m = torch.stack([x.float() for x in means])
    s_ = torch.stack([x.float() for x in stds])
    t = torch.stack([x.float() for x in tables])
    xz = (sd.x32[None] - m[:, None, :]) / s_[:, None, :]
    with torch.inference_mode():
        return world.adapter.model.base(torch.bmm(xz, t.transpose(1, 2)))


def run_point(world, sd, prior, calib_by_w, scales, idx, configs, meta, f, skip=frozenset(),
              acc_cache=None):
    q = sd.query_mask(idx)
    ym, ys = world.label_stats
    names = [n for n in configs if n == "fp" or n not in skip]     # "fp" first: always the baseline
    fp_pred = None
    for c0 in range(0, len(names), DECODE_BATCH):
        chunk = names[c0:c0 + DECODE_BATCH]
        outs = [calibrate_cfg(calib_by_w, scales, sd, prior, idx, configs[n], acc_cache) for n in chunk]
        preds = decode_batch(world, sd, [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs])
        for name, (table, mean, std, sat), pred in zip(chunk, outs, preds):
            r2, _ = rpc.r2_physical(world, sd, pred, q)
            if name == "fp":
                fp_pred, r2_fp = pred, r2
            pq = pred[q].double().cpu().numpy() * ys
            pf = fp_pred[q].double().cpu().numpy() * ys
            rms = float(np.sqrt(((pq - pf) ** 2).mean()))
            pstd = float(np.sqrt(((pf - pf.mean(0, keepdims=True)) ** 2).mean()))
            cfg = configs[name]
            row = {**meta, "cfg": name, "r2": r2, "r2_fp": r2_fp, "d_r2": r2 - r2_fp,
                   "rms_delta": rms, "pred_std_fp": pstd, "rel_rms": rms / max(pstd, 1e-12),
                   "finite": bool(np.isfinite(pq).all() and np.isfinite(r2)), "sat_events": sat,
                   "acc_mode": cfg.get("acc_mode", "perbin") if "acc" in cfg else ""}
            if name not in skip:
                f.write(json.dumps(row) + "\n")
    f.flush()


def make_configs_qat():
    return {"fp": {}, "tern": {"w": "tern"}}


def point_list(world, stratum, draws):
    """[(kind, name, B, draw, salt)] of every evaluation point of a stratum."""
    val_name = world.val_ids[-1]
    pts = []
    for kind, name in ([("val", val_name)] + [("target", t) for t in D1_POOLS[stratum]["targets"]]):
        for B in BUDGETS:
            for draw in range(draws):
                pts.append((kind, name, B, draw, f"qval-{name}" if kind == "val" else name))
    return pts


_G = {}


def _stats_worker(pt):
    torch.set_num_threads(1)
    kind, name, B, draw, salt = pt
    world, scales, sds, accs = _G["world"], _G["scales"], _G["sds"], _G["accs"]
    sd = sds[name]
    idx = rpc.draw_trials(salt, B, draw, sd.bank.n_trials)
    res = compute_acc_stats(sd, idx, scales, accs)
    return (kind, name, B, draw), {k: ({kk: vv.clone() for kk, vv in st.d.items()}, st.n_trials,
                                        info) for k, (st, info) in res.items()}


def cmd_stats(args):
    """CPU stage: per-add / two-level accumulator statistics for every point and config."""
    import multiprocessing as mp
    torch.set_num_threads(1)
    world = rpc.World(args.stratum, "O", torch.device("cpu"), load_sessions=True)
    scales = Scales(world)
    sds = {n: world.sessions[n] for n in [world.val_ids[-1]]}
    for t in D1_POOLS[args.stratum]["targets"]:
        sds[t] = world.target(t)[0]
    accs = {acc_cfg_name(c): c for c in make_configs("p0").values() if "acc" in c}
    pts = point_list(world, args.stratum, args.draws)
    path = QOUT / f"stats_{args.stratum}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = torch.load(path, weights_only=False) if path.exists() else {}
    todo = [p for p in pts if (p[0], p[1], p[2], p[3]) not in cache or
            set(accs) - set(cache[(p[0], p[1], p[2], p[3])])]
    _G.update(world=world, scales=scales, sds=sds, accs=accs)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (key, val) in enumerate(pool.imap_unordered(_stats_worker, todo)):
            cache.setdefault(key, {}).update(val)
            if i % 12 == 0:
                print(f"[stats {args.stratum}] {i + 1}/{len(todo)} t={time.perf_counter() - t0:.0f}s", flush=True)
    torch.save(cache, path)
    print("saved", path, len(cache), flush=True)


def load_acc_cache(stratum):
    path = QOUT / f"stats_{stratum}.pt"
    if not path.exists():
        return None
    raw = torch.load(path, weights_only=False)
    from ssm_decode.population_calib import PopStats as PS
    return {k: {n: (PS(d, nt), info) for n, (d, nt, info) in v.items()} for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", action="store_true",
                    help="evaluate the ternary-QAT fine-tuned checkpoints (P2 runs dir)")
    ap.add_argument("--stage", choices=["eval", "stats"], default="eval",
                    help="stats: CPU accumulator statistics (cached); eval: GPU decoding")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    ap.add_argument("--variants", nargs="*", default=["p0", "p4_m8"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    if args.stage == "stats":
        return cmd_stats(args)
    device = torch.device("cuda:0")
    world = rpc.World(args.stratum, "O", device, load_sessions=True)
    scales = Scales(world)
    acc_cache = None if args.qat else load_acc_cache(args.stratum)
    if acc_cache is None and not args.qat:
        print("WARNING: no accumulator cache; computing per-add statistics on the fly (slow)", flush=True)
    path = (OUT / "quant" / f"rows_qat_{args.stratum}.jsonl") if args.qat else (QOUT / f"rows_{args.stratum}.jsonl")
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
                            ac = None if acc_cache is None else acc_cache.get((kind, name, B, draw))
                            run_point(world, sd, prior, calib_by_w, scales, idx, configs, meta, f, skip, ac)
                    print(f"[quant {args.stratum}] {variant} s{seed} {kind} {name[-8:]} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
