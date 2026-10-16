#!/usr/bin/env python
"""Hybrid-identity experiment on DANDI 000688 sub-M (month AP trunk, 12 drift-curve targets).

Subcommands
  train    episodic training of hybrid calibrators on the trunk's SOURCE sessions only
           (leave-one-out electrode prior + simulated uncovered units, f ~ U(0, 0.5));
           checkpoint selection on source val only.
  valsrc   source-val table (all variants + the reused P0) at several uncovered fractions.
  eval     target rows: R2 overall + per-unit-class ablations (zero / swap uncovered / swap covered)
           on the 12 drift-curve targets, B in {1,2,4,8,16,33}, 8 draws, 2 seeds, contiguous
           causal decoding of the whole session (query bins scored only).
  check    structural checks (coverage flag == zero prior column on every target etc.).

Contract: trunk frozen (hash checked); target-side code under torch.no_grad, torch.autograd.grad /
backward are counted and must be zero in ``eval``; only B labeled trials are used; never final
sessions 0623/0625/0626 and never the selection session 20140307 as target (validate_targets).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_population_calib as rpc                    # noqa: E402
from scripts.run_drift_curve import (B_LIST, PREFIX, STRATUM, TARGET_DATES,   # noqa: E402
                                     days_from_source, validate_targets)
from scripts.run_labeled_calibration import D1_POOLS                 # noqa: E402
from ssm_decode.hybrid_calib import (FRAC_MAX, HybridCalibrator, covered_flags,   # noqa: E402
                                     mask_prior, simulate_uncovered)
from ssm_decode.population_calib import PopStats, desc_dim, describe  # noqa: E402

OUT = ROOT / "results/hybrid_identity_20261016"
SEEDS = (0, 1)
VAL_FRACS = (None, 0.25, 0.5)        # None = natural leave-one-out coverage of the val session

# name -> constructor kwargs (kind, hidden, use_flag).  Widths of the h1w* controls were chosen by
# ``match_width`` against the parameter count of the pooled variant noted on the right.
VARIANTS = {
    "h0m": dict(kind="flat", use_flag=False),                    # P0 + masked training, no flag
    "h1": dict(kind="flat", use_flag=True),                      # P0 + covered flag
    "h1w113": dict(kind="flat", hidden=113, use_flag=True),      # ~ h3d  (all_ds)
    "h1w130": dict(kind="flat", hidden=130, use_flag=True),      # ~ h3p  (all_pma)
    "h1w138": dict(kind="flat", hidden=138, use_flag=True),      # ~ h2d  (hyb_ds)
    "h1w153": dict(kind="flat", hidden=153, use_flag=True),      # ~ h2p  (hyb_pma)
    "h2d": dict(kind="hyb_ds"), "h2d_loc": dict(kind="hyb_ds_loc"),
    "h2p": dict(kind="hyb_pma"), "h2p_loc": dict(kind="hyb_pma_loc"),
    "h3d": dict(kind="all_ds"), "h3d_loc": dict(kind="all_ds_loc"),
    "h3p": dict(kind="all_pma"), "h3p_loc": dict(kind="all_pma_loc"),
}
MATCHED_WIDE = {"h2d": "h1w138", "h2d_loc": "h1w138", "h2p": "h1w153", "h2p_loc": "h1w153",
                "h3d": "h1w113", "h3d_loc": "h1w113", "h3p": "h1w130", "h3p_loc": "h1w130"}
LOCAL_OF = {"h2d": "h2d_loc", "h2p": "h2p_loc", "h3d": "h3d_loc", "h3p": "h3p_loc"}


def build(name, k, fd):
    spec = dict(VARIANTS[name])
    return HybridCalibrator(spec.pop("kind"), k, fd, **spec)


def run_dir(name, seed):
    return OUT / "runs" / name / f"seed{seed}"


# --------------------------------------------------------------- episodes ----

def natural_cover(world, sd):
    """Leave-one-out coverage of a source session (matches ``World.priors`` exclusion)."""
    j = world.orig_ids.index(sd.name)
    return covered_flags(world.orig_elec, sd.electrodes, exclude=j)


def make_episode_h(world, rng, names, mask_sim=True):
    ep = rpc.make_episode(world, rng, names)
    if ep is None:
        return None
    sd, prior, idx, keep, t0, mask, w = ep
    cov = natural_cover(world, sd)[keep]
    if mask_sim:
        cov = simulate_uncovered(rng, sd.electrodes[keep], cov, rng.uniform(0.0, FRAC_MAX))
    return sd, prior, idx, keep, t0, mask, w, cov


def episode_table_h(calib, sd, prior, idx, keep, cov):
    stats = PopStats.from_trials(sd.bank, idx)
    if len(keep) < sd.n_units:
        stats = stats.select_units(keep)
        prior = prior[:, torch.as_tensor(keep, device=prior.device)]
    prior = mask_prior(prior, cov)
    d, r, mean, std = describe(stats, prior)
    return calib(d, r, prior.float(), cov), mean, std


def fit_normalizers_h(world, calib, rng, n_ep=300):
    ds, rs = [], []
    for _ in range(n_ep):
        ep = make_episode_h(world, rng, world.train_ids)
        if ep is None:
            continue
        sd, prior, idx, keep, _, _, _, cov = ep
        stats = PopStats.from_trials(sd.bank, idx)
        if len(keep) < sd.n_units:
            stats = stats.select_units(keep)
            prior = prior[:, torch.as_tensor(keep, device=prior.device)]
        d, r, _, _ = describe(stats, mask_prior(prior, cov))
        ds.append(d), rs.append(r)
    calib.set_normalizers(torch.cat(ds), torch.cat(rs))


# -------------------------------------------------------------- evaluation ----

def val_cover(world, sd, frac, tag):
    nat = natural_cover(world, sd)
    if frac is None:
        return nat
    rng = np.random.default_rng(zlib.crc32(f"valmask-{sd.name}-{tag}-{frac}".encode()))
    return simulate_uncovered(rng, sd.electrodes, nat, frac)


def r2_of_table(world, sd, table, mean, std, q):
    pred = rpc.decode_full(world, sd, table, mean, std)
    return rpc.r2_physical(world, sd, pred, q)[0]


def val_score_h(world, calib, p0=False, fracs=VAL_FRACS, bs=B_LIST, draws=2, detail=False):
    """Source-val score (mean R2 over fractions x B x draws).  ``p0``: calib is a PopulationCalibrator."""
    was_training = calib.training
    calib.eval()
    per = {}
    for name in world.val_ids:
        sd, prior0 = world.sessions[name], world.priors[name]
        for frac in fracs:
            cov = val_cover(world, sd, frac, "v")
            prior = mask_prior(prior0, cov)
            for B in bs:
                for draw in range(draws):
                    idx = rpc.draw_trials(f"val-{name}", B, draw + 1234, sd.bank.n_trials)
                    stats = PopStats.from_trials(sd.bank, idx)
                    with torch.no_grad():
                        res = calib.calibrate(stats, prior) if p0 else calib.calibrate(stats, prior, cov)
                    per.setdefault(str(frac), []).append(
                        r2_of_table(world, sd, res.table, res.mean, res.std, sd.query_mask(idx)))
    if was_training:
        calib.train()
    mean = float(np.mean([v for vs in per.values() for v in vs]))
    return (mean, {k: float(np.mean(v)) for k, v in per.items()}) if detail else mean


# ------------------------------------------------------------------ train ----

def cmd_train(args):
    device = torch.device(args.device)
    world = None
    for name in args.variants.split(","):
        for seed in map(int, args.seeds.split(",")):
            rd = run_dir(name, seed)
            if (rd / "done.json").exists():
                print("skip, done:", rd, flush=True)
                continue
            rd.mkdir(parents=True, exist_ok=True)
            if world is None:
                world = rpc.World(STRATUM, "O", device)
            train_one(world, name, seed, rd, args, device)


def train_one(world, name, seed, rd, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    calib = build(name, world.k, desc_dim(world.k)).to(device)
    fit_normalizers_h(world, calib, rng)
    params = [p for p in calib.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps, step, best, best_step, doublings, since_best = args.steps, 0, -1e9, 0, 0, 0
    log_path = rd / "train_log.jsonl"
    log_path.write_text("")
    t_start = time.perf_counter()
    stopped = False
    while True:
        while step < steps:
            step += 1
            xs, ys_w, masks = [], [], []
            for _ in range(args.batch):
                ep = make_episode_h(world, rng, world.train_ids)
                if ep is None:
                    continue
                sd, prior, idx, keep, t0, mask, w, cov = ep
                table, mean, std = episode_table_h(calib, sd, prior, idx, keep, cov)
                kk = torch.as_tensor(keep, device=device)
                xw = (sd.x32[t0:t0 + w][:, kk] - mean.float()) / std.float()
                xs.append(xw @ table.T)
                ys_w.append(sd.y_norm[t0:t0 + w].float())
                masks.append(mask)
            if not xs:
                continue
            pred = world.adapter.model.base(torch.stack(xs))
            err = (pred - torch.stack(ys_w)) ** 2
            m = torch.stack(masks).unsqueeze(-1).float()
            loss = (err * m).sum() / (m.sum() * err.shape[-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % args.val_interval == 0 or step == steps:
                score, per = val_score_h(world, calib, detail=True)
                rec = {"step": step, "loss": float(loss), "val_r2": score, "val_by_frac": per,
                       "elapsed": round(time.perf_counter() - t_start, 1)}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(name, seed, json.dumps({k: rec[k] for k in ("step", "loss", "val_r2", "elapsed")}),
                      flush=True)
                if score > best:
                    best, best_step, since_best = score, step, 0
                    torch.save({"calibrator": calib.state_dict(), "name": name, "seed": seed,
                                "spec": VARIANTS[name], "best_step": step, "best_val_r2": score},
                               rd / "best.pt")
                else:
                    since_best += 1
                if since_best >= args.patience and step >= args.min_steps:
                    stopped = True
                    break
        if stopped:
            break
        if best_step > 0.8 * steps and doublings < 4:
            steps *= 2
            doublings += 1
            print(f"{name} s{seed}: best_step {best_step} in last 20%; budget -> {steps}", flush=True)
            continue
        break
    done = {"name": name, "seed": seed, "spec": VARIANTS[name], "n_params": calib.n_params,
            "best_step": best_step, "best_val_r2": best, "final_budget": steps, "doublings": doublings,
            "stopped_early": stopped, "converged": bool(stopped or best_step <= 0.8 * steps),
            "last_step": step, "train_ids": world.train_ids, "val_ids": world.val_ids,
            "frac_max": FRAC_MAX, "seconds": round(time.perf_counter() - t_start, 1)}
    (rd / "done.json").write_text(json.dumps(done, indent=1))
    print("trained", name, seed, json.dumps({k: done[k] for k in ("best_step", "best_val_r2", "converged",
                                                                  "n_params", "seconds")}), flush=True)


def load_h(name, seed, world):
    blob = torch.load(run_dir(name, seed) / "best.pt", map_location=world.device, weights_only=False)
    calib = build(name, world.k, desc_dim(world.k)).to(world.device)
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    for p in calib.parameters():
        p.requires_grad_(False)
    return calib, blob


def load_p0(world):
    out = {}
    for seed in SEEDS:
        out[seed] = rpc.load_run(rpc.P1_OUT / "runs" / f"{STRATUM}_O" / "p0" / f"seed{seed}", world)[0]
    return out


# ------------------------------------------------------------------ valsrc ----

def cmd_valsrc(args):
    device = torch.device(args.device)
    world = rpc.World(STRATUM, "O", device)
    rows = []
    p0 = load_p0(world)
    for seed in SEEDS:
        m, per = val_score_h(world, p0[seed], p0=True, detail=True)
        rows.append({"variant": "h0", "seed": seed, "val_mean": m, **{f"val_{k}": v for k, v in per.items()}})
    for name in VARIANTS:
        for seed in SEEDS:
            if not (run_dir(name, seed) / "best.pt").exists():
                continue
            calib, blob = load_h(name, seed, world)
            m, per = val_score_h(world, calib, detail=True)
            done = json.loads((run_dir(name, seed) / "done.json").read_text())
            rows.append({"variant": name, "seed": seed, "val_mean": m, "n_params": calib.n_params,
                         "best_step": done["best_step"], "converged": done["converged"],
                         **{f"val_{k}": v for k, v in per.items()}})
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "valsrc.json").write_text(json.dumps(rows, indent=1))


# -------------------------------------------------------------------- eval ----

def count_autograd():
    cnt = {"n": 0}
    og, ob = torch.autograd.grad, torch.autograd.backward

    def g(*a, **k):
        cnt["n"] += 1
        return og(*a, **k)

    def b(*a, **k):
        cnt["n"] += 1
        return ob(*a, **k)
    torch.autograd.grad, torch.autograd.backward = g, b
    return cnt, (og, ob)


def mixed(cov_t, t_cov, t_unc):
    """Table whose covered columns come from ``t_cov`` and uncovered columns from ``t_unc``."""
    return torch.where(cov_t[None, :], t_cov, t_unc)


def cmd_eval(args):
    device = torch.device(args.device)
    cnt, orig = count_autograd()
    world = rpc.World(STRATUM, "O", device, load_sessions=False)
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(args.targets.split(","), src)
    variants = args.variants.split(",") if args.variants else list(VARIANTS)
    p0 = load_p0(world)
    models = {(n, s): load_h(n, s, world)[0] for n in variants for s in SEEDS}
    hash0 = world.adapter._hash()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval").mkdir(exist_ok=True)
    path = OUT / "eval" / f"rows_{args.tag or '-'.join(d[2:] for d in dates)}.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["variant"], r["seed"], r["target"], r["B"], r["draw"]))
    t0 = time.perf_counter()
    for d in dates:
        target = PREFIX + d
        sd, prior = world.target(target)
        cov = covered_flags(world.orig_elec, sd.electrodes)             # source metadata only
        zero_col = (prior.abs().sum(0) == 0).cpu().numpy()
        assert (zero_col == ~cov).all(), (target, zero_col, cov)        # flag == "prior column is empty"
        cov_t = torch.as_tensor(cov, device=device)
        unc_frac = float((~cov).mean())
        print(f"[hyb] {d} N={sd.n_units} uncovered={unc_frac:.2f}", flush=True)
        for B in B_LIST:
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                stats = PopStats.from_trials(sd.bank, idx)
                q = sd.query_mask(idx)
                used = np.zeros(sd.n_bins, dtype=bool)
                for i in idx:
                    a, b = sd.trials[i]
                    used[a:b] = True
                base = {"target": target, "date": d, "days": days_from_source(d), "n_units": sd.n_units,
                        "n_uncovered": int((~cov).sum()), "unc_frac": unc_frac, "B": B, "draw": draw,
                        "h4_budget": int(B), "h4_query_disjoint": bool(not (used & q.cpu().numpy()).any())}
                for seed in SEEDS:
                    todo = [v for v in ["h0"] + variants if (v, seed, target, B, draw) not in seen]
                    if not todo:
                        continue
                    n0 = cnt["n"]
                    with torch.no_grad():
                        r0 = p0[seed].calibrate(stats, prior)
                    t_p0 = r0.table
                    mean, std = r0.mean, r0.std                          # identical normalisation for all
                    rows = []
                    for v in todo:
                        if v == "h0":
                            t_h = t_p0
                        else:
                            with torch.no_grad():
                                rh = models[(v, seed)].calibrate(stats, prior, cov)
                            t_h = rh.table
                            assert torch.equal(rh.mean, mean) and torch.equal(rh.std, std)
                        row = {**base, "variant": v, "seed": seed}
                        row["r2"] = r2_of_table(world, sd, t_h, mean, std, q)
                        row["r2_zero_unc"] = r2_of_table(world, sd, t_h * cov_t[None, :], mean, std, q)
                        if v != "h0":
                            row["r2_swap_unc"] = r2_of_table(world, sd, mixed(cov_t, t_p0, t_h), mean, std, q)
                            row["r2_swap_cov"] = r2_of_table(world, sd, mixed(cov_t, t_h, t_p0), mean, std, q)
                        row["col_norm_unc"] = float(t_h[:, ~cov_t].norm(dim=0).mean()) if (~cov).any() else None
                        row["col_norm_cov"] = float(t_h[:, cov_t].norm(dim=0).mean()) if cov.any() else None
                        rows.append(row)
                    for r in rows:
                        r["autograd_calls"] = cnt["n"] - n0
                    with path.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[hyb] {d} B={B} d={draw} s{seed} t={time.perf_counter() - t0:.0f}s", flush=True)
        del sd, prior
    assert world.adapter._hash() == hash0, "trunk changed"
    torch.autograd.grad, torch.autograd.backward = orig
    print("autograd calls:", cnt["n"], flush=True)


def cmd_check(args):
    device = torch.device(args.device)
    world = rpc.World(STRATUM, "O", device, load_sessions=False)
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(TARGET_DATES, src)
    out = []
    for d in dates:
        sd, prior = world.target(PREFIX + d)
        cov = covered_flags(world.orig_elec, sd.electrodes)
        zero_col = (prior.abs().sum(0) == 0).cpu().numpy()
        out.append({"target": d, "n_units": sd.n_units, "n_uncovered": int((~cov).sum()),
                    "unc_frac": float((~cov).mean()), "flag_equals_zero_prior_col": bool((zero_col == ~cov).all())})
        print(out[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage_check.json").write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm in ("train", "valsrc", "eval", "check"):
        s = sub.add_parser(nm)
        s.add_argument("--device", default="cuda:0")
    t = sub.choices["train"]
    t.add_argument("--variants", required=True)
    t.add_argument("--seeds", default="0,1")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--min-steps", type=int, default=600)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--val-interval", type=int, default=100)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    e = sub.choices["eval"]
    e.add_argument("--targets", required=True)
    e.add_argument("--variants", default=None)
    e.add_argument("--draws", type=int, default=8)
    e.add_argument("--tag", default=None)
    args = p.parse_args()
    {"train": cmd_train, "valsrc": cmd_valsrc, "eval": cmd_eval, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
