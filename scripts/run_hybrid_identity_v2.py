#!/usr/bin/env python
"""Hybrid identity v2 on DANDI 000688 sub-M (month AP trunk, 12 drift-curve targets).

Question (v2): when the descriptor-level population coupling is CUT for uncovered units
(``cut_coupling``), can pooling (fixed PMA, pairwise cross-attention XATT) recover the information,
and does it beat capacity-matched controls, also at high uncovered fractions (stress test)?

Subcommands
  train   episodic training (source sessions only, leave-one-out prior, simulated uncovered
          f ~ U(0, 0.9)); checkpoint selection on source val only; collapse monitors in the log.
  valsrc  source-val table at several uncovered fractions (best checkpoints + reused P0).
  eval    target rows: R2 overall, zero-uncovered, swap_unc, swap_cov on the 12 drift-curve targets;
          natural coverage and stress fractions (deterministic electrode masks, shared by variants).
  oracle  DIAGNOSTIC ONLY (violates H4): fit the uncovered columns by gradient on all target trials
          (trunk frozen), covered columns fixed to the P0 table; gives the headroom.
  check   structural checks (flag == zero prior column, stress masks, parameter counts).

Contract (as v1): trunk frozen (hash checked); target-side candidate code runs under torch.no_grad,
autograd calls are counted and must be zero in ``eval``; only B labeled trials; never 0623/0625/0626
nor 20140307 as target; selection on source val only; contiguous causal decoding.
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.run_hybrid_identity import (count_autograd, load_p0, mixed, natural_cover,   # noqa: E402
                                         val_cover, val_score_h)
from scripts.run_labeled_calibration import D1_POOLS                 # noqa: E402
from ssm_decode.hybrid_calib import (HybridCalibrator, covered_flags, mask_prior, match_width,   # noqa: E402
                                     no_decay_names, pair_corr, simulate_uncovered)
from ssm_decode.population_calib import PopStats, desc_dim, describe   # noqa: E402

OUT = ROOT / "results/hybrid_identity_v2_20261017"
SEEDS = (0, 1, 2)
P0_SEEDS = (0, 1)
FRAC_MAX_V2 = 0.9                       # training range of the simulated uncovered fraction
VAL_FRACS_V2 = (None, 0.3, 0.6, 0.9)    # natural LOO coverage + simulated fractions (source val only)
STRESS_FRACS = (0.5, 0.7, 0.9)
STRESS_BS = (2, 8, 33)
STRESS_DRAWS = 4
CONDS = ("cut", "coupled")
CUT = {"cut": True, "coupled": False}

# name -> kwargs of HybridCalibrator (widths of the h1w* controls from ``match_width``):
#   h1w137 ~ hx / hx_np (27,033 params), h1w153 ~ h2pf / h2pf_loc (32,742 params)
VARIANTS = {
    "h0m": dict(kind="flat", use_flag=False),                       # P0 + masked training, no flag
    "h1": dict(kind="flat", use_flag=True),                         # P0 + masked training + flag
    "h1w137": dict(kind="flat", hidden=137, use_flag=True),         # wide control of hx
    "h1w153": dict(kind="flat", hidden=153, use_flag=True),         # wide control of h2pf
    "h2p": dict(kind="hyb_pma"),                                    # v1 PMA (collapse-prone init)
    "h2pf": dict(kind="hyb_pma", pma_fix=True),                     # fixed PMA
    "h2pf_loc": dict(kind="hyb_pma_loc", pma_fix=True),             # per-unit control of h2pf
    "hx": dict(kind="hyb_xatt"),                                    # pairwise cross-attention
    "hx_np": dict(kind="hyb_xatt_np"),                              # no-pairwise control of hx
    "hxs": dict(kind="hyb_xatt", pair_norm=True),                   # XATT with row-standardised pair input
    "hxs_np": dict(kind="hyb_xatt_np", pair_norm=True),             # its no-pairwise control
}
COND_VARIANTS = {"cut": list(VARIANTS), "coupled": ["h1", "h2pf", "h2pf_loc", "hx", "hx_np"]}
CONTROL_OF = {"hx": "hx_np", "hxs": "hxs_np", "h2pf": "h2pf_loc"}
WIDE_OF = {"hx": "h1w137", "hx_np": "h1w137", "hxs": "h1w137", "hxs_np": "h1w137", "h2pf": "h1w153", "h2pf_loc": "h1w153"}


def build(name, k, fd, cond):
    spec = dict(VARIANTS[name])
    return HybridCalibrator(spec.pop("kind"), k, fd, cut_coupling=CUT[cond], **spec)


def run_dir(cond, name, seed):
    return OUT / "runs" / cond / name / f"seed{seed}"


# --------------------------------------------------------------- episodes ----

def make_episode_v2(world, rng, names):
    ep = rpc.make_episode(world, rng, names)
    if ep is None:
        return None
    sd, prior, idx, keep, t0, mask, w = ep
    cov = natural_cover(world, sd)[keep]
    cov = simulate_uncovered(rng, sd.electrodes[keep], cov, rng.uniform(0.0, FRAC_MAX_V2))
    return sd, prior, idx, keep, t0, mask, w, cov


def _episode_stats(sd, prior, idx, keep, cov):
    stats = PopStats.from_trials(sd.bank, idx)
    if len(keep) < sd.n_units:
        stats = stats.select_units(keep)
        prior = prior[:, torch.as_tensor(keep, device=prior.device)]
    return stats, mask_prior(prior, cov)


def episode_table_v2(calib, sd, prior, idx, keep, cov):
    stats, prior = _episode_stats(sd, prior, idx, keep, cov)
    d, r, mean, std = describe(stats, prior)
    pair = pair_corr(stats) if calib.xatt else None
    return calib(d, r, prior.float(), cov, pair), mean, std


def fit_normalizers_v2(world, calib, rng, n_ep=300):
    ds, rs, cs = [], [], []
    for _ in range(n_ep):
        ep = make_episode_v2(world, rng, world.train_ids)
        if ep is None:
            continue
        sd, prior, idx, keep, _, _, _, cov = ep
        stats, pm = _episode_stats(sd, prior, idx, keep, cov)
        d, r, _, _ = describe(stats, pm)
        ds.append(d), rs.append(r), cs.append(torch.as_tensor(cov, device=d.device))
    calib.set_normalizers(torch.cat(ds), torch.cat(rs), torch.cat(cs))


def probe_diagnostics(world, calib):
    """Collapse monitors on fixed probe episodes of the source-val session (natural + 50% uncovered)."""
    if not (calib.xatt or (calib.family == "pma" and not calib.local)):
        return {}
    acc = {}
    calib.eval()
    for name in world.val_ids:
        sd, prior0 = world.sessions[name], world.priors[name]
        for frac in (None, 0.5):
            cov = val_cover(world, sd, frac, "probe")
            prior = mask_prior(prior0, cov)
            for B in (2, 8, 33):
                idx = rpc.draw_trials(f"val-{name}", B, 4321, sd.bank.n_trials)
                stats = PopStats.from_trials(sd.bank, idx)
                d, r, _, _ = describe(stats, prior)
                pair = pair_corr(stats) if calib.xatt else None
                for k, v in calib.diagnostics(d, r, prior.float(), cov, pair).items():
                    acc.setdefault(k, []).append(v)
    calib.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ------------------------------------------------------------------ train ----

def cmd_train(args):
    device = torch.device(args.device)
    world = None
    for cond in args.conds.split(","):
        for name in args.variants.split(","):
            if name not in COND_VARIANTS[cond]:
                continue
            for seed in map(int, args.seeds.split(",")):
                rd = run_dir(cond, name, seed)
                if (rd / "done.json").exists():
                    print("skip, done:", rd, flush=True)
                    continue
                rd.mkdir(parents=True, exist_ok=True)
                if world is None:
                    world = rpc.World(STRATUM, "O", device)
                train_one(world, cond, name, seed, rd, args, device)


def optimizer_groups(calib, wd):
    nd = [p for n, p in calib.named_parameters() if n in no_decay_names() and p.requires_grad]
    dec = [p for n, p in calib.named_parameters() if n not in no_decay_names() and p.requires_grad]
    groups = [{"params": dec, "weight_decay": wd}]
    if nd:
        groups.append({"params": nd, "weight_decay": 0.0})
    return groups, dec + nd


def train_one(world, cond, name, seed, rd, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    calib = build(name, world.k, desc_dim(world.k), cond).to(device)
    fit_normalizers_v2(world, calib, rng)
    groups, params = optimizer_groups(calib, 1e-4)
    opt = torch.optim.AdamW(groups, lr=args.lr)
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
                ep = make_episode_v2(world, rng, world.train_ids)
                if ep is None:
                    continue
                sd, prior, idx, keep, t0, mask, w, cov = ep
                table, mean, std = episode_table_v2(calib, sd, prior, idx, keep, cov)
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
                score, per = val_score_h(world, calib, fracs=VAL_FRACS_V2, detail=True)
                diag = probe_diagnostics(world, calib)
                rec = {"step": step, "loss": float(loss), "val_r2": score, "val_by_frac": per,
                       "diag": diag, "elapsed": round(time.perf_counter() - t_start, 1)}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(cond, name, seed, json.dumps({k: rec[k] for k in ("step", "loss", "val_r2", "diag", "elapsed")}),
                      flush=True)
                if score > best:
                    best, best_step, since_best = score, step, 0
                    torch.save({"calibrator": calib.state_dict(), "name": name, "seed": seed, "cond": cond,
                                "spec": VARIANTS[name], "best_step": step, "best_val_r2": score,
                                "diag": diag}, rd / "best.pt")
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
            print(f"{cond} {name} s{seed}: best_step {best_step} in last 20%; budget -> {steps}", flush=True)
            continue
        break
    done = {"name": name, "seed": seed, "cond": cond, "spec": VARIANTS[name], "n_params": calib.n_params,
            "best_step": best_step, "best_val_r2": best, "final_budget": steps, "doublings": doublings,
            "stopped_early": stopped, "converged": bool(stopped or best_step <= 0.8 * steps),
            "last_step": step, "train_ids": world.train_ids, "val_ids": world.val_ids,
            "frac_max": FRAC_MAX_V2, "val_fracs": [str(f) for f in VAL_FRACS_V2],
            "seconds": round(time.perf_counter() - t_start, 1)}
    (rd / "done.json").write_text(json.dumps(done, indent=1))
    print("trained", cond, name, seed, json.dumps({k: done[k] for k in ("best_step", "best_val_r2", "converged",
                                                                        "n_params", "seconds")}), flush=True)


def load_v2(cond, name, seed, world):
    blob = torch.load(run_dir(cond, name, seed) / "best.pt", map_location=world.device, weights_only=False)
    calib = build(name, world.k, desc_dim(world.k), cond).to(world.device)
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    for p in calib.parameters():
        p.requires_grad_(False)
    return calib, blob


# ------------------------------------------------------------------ valsrc ----

def cmd_valsrc(args):
    device = torch.device(args.device)
    world = rpc.World(STRATUM, "O", device)
    rows = []
    p0 = load_p0(world)
    for seed in P0_SEEDS:
        m, per = val_score_h(world, p0[seed], p0=True, fracs=VAL_FRACS_V2, detail=True)
        rows.append({"cond": "-", "variant": "h0", "seed": seed, "val_mean": m,
                     **{f"val_{k}": v for k, v in per.items()}})
    for cond in CONDS:
        for name in COND_VARIANTS[cond]:
            for seed in SEEDS:
                if not (run_dir(cond, name, seed) / "done.json").exists():
                    continue
                calib, blob = load_v2(cond, name, seed, world)
                m, per = val_score_h(world, calib, fracs=VAL_FRACS_V2, detail=True)
                done = json.loads((run_dir(cond, name, seed) / "done.json").read_text())
                rows.append({"cond": cond, "variant": name, "seed": seed, "val_mean": m,
                             "n_params": calib.n_params, "best_step": done["best_step"],
                             "converged": done["converged"], "seconds": done["seconds"],
                             **{f"val_{k}": v for k, v in per.items()},
                             **{f"diag_{k}": v for k, v in probe_diagnostics(world, calib).items()}})
                print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "valsrc.json").write_text(json.dumps(rows, indent=1))


# -------------------------------------------------------------------- eval ----

def stress_cover(target, electrodes, cov_nat, frac):
    """Deterministic simulated-uncovered mask of (target, frac): independent of variant, B, draw, seed."""
    if frac is None:
        return cov_nat
    rng = np.random.default_rng(zlib.crc32(f"stress-{target}-{frac}".encode()))
    return simulate_uncovered(rng, electrodes, cov_nat, frac)


class FastR2:
    """Physical-unit R2 on the query bins, same formula as ``rpc.r2_physical`` (checked once)."""

    def __init__(self, world, sd):
        self.ym, self.ys = world.label_stats
        self.beh = torch.from_numpy(np.asarray(sd.record.behavior, dtype=np.float64)).to(sd.device)

    def __call__(self, pred_norm, q):
        pred = pred_norm[q].double() * torch.as_tensor(self.ys, device=pred_norm.device) \
            + torch.as_tensor(self.ym, device=pred_norm.device)
        truth = self.beh[q]
        ss_res = ((truth - pred) ** 2).sum()
        ss_tot = ((truth - truth.mean(0, keepdim=True)) ** 2).sum()
        return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


def decode_r2_many(world, sd, fr2, tables, mean, std, q, chunk=4):
    xz = (sd.x32 - mean.float()) / std.float()
    out = []
    for lo in range(0, len(tables), chunk):
        tb = torch.stack([t.float() for t in tables[lo:lo + chunk]])
        inp = torch.einsum("tn,mkn->mtk", xz, tb)
        with torch.inference_mode():
            pred = world.adapter.model.base(inp)
        out.extend(fr2(p, q) for p in pred.unbind(0))
        del pred, inp, tb
    torch.cuda.empty_cache()
    return out


def eval_cells(args, targets, fracs, bs, draws):
    for d in targets:
        for f in fracs:
            for B in bs:
                for draw in range(draws):
                    yield d, f, B, draw


def cmd_eval(args):
    device = torch.device(args.device)
    cnt, orig = count_autograd()
    world = rpc.World(STRATUM, "O", device, load_sessions=False)
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(args.targets.split(","), src)
    conds = args.conds.split(",")
    fracs = [None if x == "nat" else float(x) for x in args.fracs.split(",")]
    bs = [int(x) for x in args.bs.split(",")]
    variants = args.variants.split(",") if args.variants else None
    p0 = load_p0(world)
    models = {}
    for cond in conds:
        for name in COND_VARIANTS[cond]:
            if variants and name not in variants:
                continue
            if not all((run_dir(cond, name, sd_) / "done.json").exists() for sd_ in SEEDS):
                print(f"[hyb2] skipping {cond}/{name}: not all seeds trained", flush=True)
                continue
            for seed in SEEDS:
                models[(cond, name, seed)] = load_v2(cond, name, seed, world)[0]
    hash0 = world.adapter._hash()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval").mkdir(exist_ok=True)
    path = OUT / "eval" / f"rows_{args.tag}.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["cond"], r["variant"], r["seed"], r["target"], r["frac"], r["B"], r["draw"]))
    t0 = time.perf_counter()
    masks_log = {}
    checked_r2 = False
    for d in dates:
        target = PREFIX + d
        sd, prior = world.target(target)
        cov_nat = covered_flags(world.orig_elec, sd.electrodes)
        zero_col = (prior.abs().sum(0) == 0).cpu().numpy()
        assert (zero_col == ~cov_nat).all(), (target, zero_col, cov_nat)
        fr2 = FastR2(world, sd)
        for f in fracs:
            cov = stress_cover(target, sd.electrodes, cov_nat, f)
            masks_log[f"{d}|{f}"] = {"covered": cov.astype(int).tolist(), "unc_frac": float((~cov).mean())}
            assert (cov <= cov_nat).all()                       # only ever removes coverage
            cov_t = torch.as_tensor(cov, device=device)
            prior_m = mask_prior(prior, cov)
            unc_frac = float((~cov).mean())
            print(f"[hyb2] {d} f={f} N={sd.n_units} uncovered={unc_frac:.2f}", flush=True)
            for B in bs:
                for draw in range(args.draws):
                    idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                    stats = PopStats.from_trials(sd.bank, idx)
                    q = sd.query_mask(idx)
                    used = np.zeros(sd.n_bins, dtype=bool)
                    for i in idx:
                        a, b = sd.trials[i]
                        used[a:b] = True
                    base = {"target": target, "date": d, "days": days_from_source(d), "n_units": sd.n_units,
                            "n_uncovered": int((~cov).sum()), "unc_frac": unc_frac, "unc_frac_natural":
                            float((~cov_nat).mean()), "frac": f, "B": B, "draw": draw, "h4_budget": int(B),
                            "h4_query_disjoint": bool(not (used & q.cpu().numpy()).any())}
                    todo = [(c, n, s) for (c, n, s) in models
                            if (c, n, s, target, f, B, draw) not in seen]
                    todo_p0 = [s for s in P0_SEEDS if ("-", "h0", s, target, f, B, draw) not in seen]
                    if not todo and not todo_p0:
                        continue
                    n0 = cnt["n"]
                    with torch.no_grad():
                        r0 = {s: p0[s].calibrate(stats, prior_m) for s in P0_SEEDS}
                    t_p0 = r0[0].table
                    mean, std = r0[0].mean, r0[0].std                  # identical normalisation for all
                    assert torch.equal(r0[1].mean, mean)
                    tables, tags = [t_p0, t_p0 * cov_t[None, :], r0[1].table, prior_m.float()], \
                        [("p0", 0), ("p0_zero", 0), ("p0", 1), ("prior", 0)]
                    diags = {}
                    for key in todo:
                        cond, name, seed = key
                        m = models[key]
                        with torch.no_grad():
                            rh = m.calibrate(stats, prior_m, cov)
                            if m.xatt or (m.family == "pma" and not m.local):
                                dd, rr, _, _ = describe(stats, prior_m)
                                diags[key] = m.diagnostics(dd, rr, prior_m.float(), cov,
                                                           pair_corr(stats) if m.xatt else None)
                        assert torch.equal(rh.mean, mean) and torch.equal(rh.std, std)
                        t_h = rh.table
                        tables += [t_h, t_h * cov_t[None, :], mixed(cov_t, t_p0, t_h), mixed(cov_t, t_h, t_p0)]
                        tags += [(key, "r2"), (key, "zero"), (key, "swap_unc"), (key, "swap_cov")]
                    vals = decode_r2_many(world, sd, fr2, tables, mean, std, q)
                    if not checked_r2:
                        pred = rpc.decode_full(world, sd, t_p0, mean, std)
                        ref = rpc.r2_physical(world, sd, pred, q)[0]
                        assert abs(ref - vals[0]) < 1e-5, (ref, vals[0])
                        checked_r2 = True
                    res = dict(zip(map(str, tags), vals))
                    rows = []
                    for s in todo_p0:
                        rows.append({**base, "cond": "-", "variant": "h0", "seed": s, "r2": res[str(("p0", s))],
                                     "r2_zero_unc": res[str(("p0_zero", 0))] if s == 0 else None,
                                     "r2_prior": res[str(("prior", 0))] if s == 0 else None})
                    for key in todo:
                        cond, name, seed = key
                        rows.append({**base, "cond": cond, "variant": name, "seed": seed,
                                     "r2": res[str((key, "r2"))], "r2_zero_unc": res[str((key, "zero"))],
                                     "r2_swap_unc": res[str((key, "swap_unc"))],
                                     "r2_swap_cov": res[str((key, "swap_cov"))],
                                     "r2_base": res[str(("p0", 0))], "diag": diags.get(key)})
                    for r in rows:
                        r["autograd_calls"] = cnt["n"] - n0
                    with path.open("a") as fh:
                        for r in rows:
                            fh.write(json.dumps(r) + "\n")
                    print(f"[hyb2] {d} f={f} B={B} d={draw} rows={len(rows)} t={time.perf_counter() - t0:.0f}s",
                          flush=True)
        del sd, prior
    assert world.adapter._hash() == hash0, "trunk changed"
    torch.autograd.grad, torch.autograd.backward = orig
    (OUT / "eval" / f"masks_{args.tag}.json").write_text(json.dumps(masks_log))
    print("autograd calls:", cnt["n"], flush=True)


# ------------------------------------------------------------------ oracle ----

def fit_unc_columns(world, sd, table0, mean, std, cov_t, fit_mask, rng, steps, lr, batch=16, window=256,
                    burn=64):
    """DIAGNOSTIC (H4-violating).  Gradient fit of the uncovered columns of ``table0`` on all bins in
    ``fit_mask`` (trunk frozen, covered columns fixed).  Returns the fitted table and the loss curve."""
    iu = torch.nonzero(~cov_t).squeeze(-1)
    u = table0[:, iu].clone().float().requires_grad_(True)
    fm = fit_mask.cpu().numpy()
    cs = np.concatenate([[0], np.cumsum(fm)])
    starts = np.arange(0, sd.n_bins - window)
    ok = (cs[starts + window] - cs[starts + burn]) >= 24
    starts = starts[ok]
    if len(starts) == 0:
        return table0, [float("nan")]
    opt = torch.optim.Adam([u], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xz = (sd.x32 - mean.float()) / std.float()
    losses = []
    for _ in range(steps):
        t0s = rng.choice(starts, size=batch)
        tab = table0.float().index_copy(1, iu, u)
        xs = torch.stack([xz[t:t + window] @ tab.T for t in t0s])
        ys = torch.stack([sd.y_norm[t:t + window].float() for t in t0s])
        ms = torch.stack([fit_mask[t:t + window] for t in t0s]).clone()
        ms[:, :burn] = False
        pred = world.adapter.model.base(xs)
        err = (pred - ys) ** 2
        mm = ms.unsqueeze(-1).float()
        loss = (err * mm).sum() / (mm.sum() * err.shape[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
    return table0.float().index_copy(1, iu, u.detach()), losses


def cmd_oracle(args):
    device = torch.device(args.device)
    world = rpc.World(STRATUM, "O", device, load_sessions=False)
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(args.targets.split(","), src)
    fracs = [None if x == "nat" else float(x) for x in args.fracs.split(",")]
    bs = [int(x) for x in args.bs.split(",")]
    p0 = load_p0(world)
    hash0 = world.adapter._hash()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "eval" / f"{'oracle' if args.fit_on == 'all' else 'oraclecal'}_{args.tag}.jsonl"
    path.parent.mkdir(exist_ok=True)
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["target"], r["frac"], r["B"], r["draw"]))
    t00 = time.perf_counter()
    for d in dates:
        target = PREFIX + d
        sd, prior = world.target(target)
        cov_nat = covered_flags(world.orig_elec, sd.electrodes)
        fr2 = FastR2(world, sd)
        in_trial = (sd.trial_id >= 0) & sd.valid
        par = torch.zeros_like(in_trial)
        tid = sd.trial_id.clamp_min(0)
        par[:] = (tid % 2 == 0)
        for f in fracs:
            cov = stress_cover(target, sd.electrodes, cov_nat, f)
            if (~cov).sum() == 0:
                continue
            cov_t = torch.as_tensor(cov, device=device)
            prior_m = mask_prior(prior, cov)
            for B in bs:
                for draw in range(args.draws):
                    if (target, f, B, draw) in seen:
                        continue
                    idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                    stats = PopStats.from_trials(sd.bank, idx)
                    q = sd.query_mask(idx)
                    used = np.zeros(sd.n_bins, dtype=bool)
                    for i in idx:
                        a_, b_ = sd.trials[i]
                        used[a_:b_] = True
                    with torch.no_grad():
                        r0 = p0[0].calibrate(stats, prior_m)
                    mean, std, t_p0 = r0.mean, r0.std, r0.table
                    rng = np.random.default_rng(zlib.crc32(f"oracle-{target}-{f}-{B}-{draw}".encode()))
                    if args.fit_on == "all":
                        fit_mask = in_trial
                    else:               # reference: gradient fit of the uncovered columns on the B trials only
                        fit_mask = torch.as_tensor(used, device=device) & in_trial
                    t_or, losses = fit_unc_columns(world, sd, t_p0, mean, std, cov_t, fit_mask, rng,
                                                   args.steps, args.lr)
                    zero_unc = t_p0 * cov_t[None, :]
                    r_p0, r_zero, r_or = decode_r2_many(world, sd, fr2, [t_p0, zero_unc, t_or], mean, std, q)
                    row = {"target": target, "date": d, "frac": f, "B": B, "draw": draw,
                           "unc_frac": float((~cov).mean()), "n_uncovered": int((~cov).sum()),
                           "r2_p0": r_p0, "r2_zero_unc": r_zero, "r2_oracle": r_or,
                           "loss_first": losses[0], "loss_last": float(np.mean(losses[-10:])),
                           "diagnostic_only": True, "h4_violating": args.fit_on == "all", "fit_on": args.fit_on,
                           "note": ("uncovered columns fitted on ALL target trials (in-sample on query)"
                                    if args.fit_on == "all" else
                                    "uncovered columns fitted by backprop through the trunk on the B calibration "
                                    "trials only (reference, not a candidate: target backprop)")}
                    if args.xfit and args.fit_on == "all" and B == args.xfit_b:
                        # 2-fold cross-fit check of the in-sample bias: fit on one trial-parity half,
                        # score on the query bins of the other half (P0 and in-sample oracle on the same bins)
                        xf = []
                        for fold in (0, 1):
                            fm = in_trial & (par if fold == 0 else ~par)
                            tab_x, _ = fit_unc_columns(world, sd, t_p0, mean, std, cov_t, fm, rng,
                                                       args.steps, args.lr)
                            qs = q & in_trial & (~par if fold == 0 else par)
                            a, b, c = decode_r2_many(world, sd, fr2, [t_p0, t_or, tab_x], mean, std, qs)
                            xf.append((a, b, c))
                        row["xfit_p0"] = float(np.mean([x[0] for x in xf]))
                        row["xfit_oracle_insample"] = float(np.mean([x[1] for x in xf]))
                        row["xfit_oracle_heldout"] = float(np.mean([x[2] for x in xf]))
                    with path.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(f"[orc] {d} f={f} B={B} d={draw} p0={r_p0:.3f} zero={r_zero:.3f} oracle={r_or:.3f} "
                          f"loss {losses[0]:.3f}->{row['loss_last']:.3f} t={time.perf_counter() - t00:.0f}s",
                          flush=True)
        del sd, prior
    assert world.adapter._hash() == hash0, "trunk changed"


# ------------------------------------------------------------------- check ----

def cmd_check(args):
    device = torch.device(args.device)
    world = rpc.World(STRATUM, "O", device, load_sessions=False)
    src = [s[-8:] for s in D1_POOLS[STRATUM]["sources"].split(",")]
    dates = validate_targets(TARGET_DATES, src)
    fd = desc_dim(world.k)
    out = {"n_params": {n: build(n, world.k, fd, "cut").n_params for n in VARIANTS},
           "match_width_hx": match_width(build("hx", world.k, fd, "cut").n_params, world.k, fd),
           "match_width_h2pf": match_width(build("h2pf", world.k, fd, "cut").n_params, world.k, fd),
           "targets": []}
    for d in dates:
        sd, prior = world.target(PREFIX + d)
        cov_nat = covered_flags(world.orig_elec, sd.electrodes)
        rec = {"target": d, "n_units": sd.n_units, "unc_frac_natural": float((~cov_nat).mean())}
        for f in STRESS_FRACS:
            cov = stress_cover(PREFIX + d, sd.electrodes, cov_nat, f)
            rec[f"unc_{f}"] = float((~cov).mean())
            rec[f"n_cov_{f}"] = int(cov.sum())
        out["targets"].append(rec)
        print(rec, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(out, indent=1))
    print(out["n_params"], out["match_width_hx"], out["match_width_h2pf"])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm in ("train", "valsrc", "eval", "oracle", "check"):
        s = sub.add_parser(nm)
        s.add_argument("--device", default="cuda:0")
    t = sub.choices["train"]
    t.add_argument("--variants", required=True)
    t.add_argument("--conds", default="cut")
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--min-steps", type=int, default=600)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--val-interval", type=int, default=100)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    for nm in ("eval", "oracle"):
        e = sub.choices[nm]
        e.add_argument("--targets", required=True)
        e.add_argument("--fracs", default="nat")
        e.add_argument("--bs", default=",".join(map(str, B_LIST)))
        e.add_argument("--draws", type=int, default=8)
        e.add_argument("--tag", required=True)
    e = sub.choices["eval"]
    e.add_argument("--conds", default="cut,coupled")
    e.add_argument("--variants", default=None)
    o = sub.choices["oracle"]
    o.add_argument("--steps", type=int, default=250)
    o.add_argument("--lr", type=float, default=0.02)
    o.add_argument("--xfit", action="store_true")
    o.add_argument("--fit-on", choices=("all", "calib"), default="all")
    o.add_argument("--xfit-b", type=int, default=8)
    args = p.parse_args()
    {"train": cmd_train, "valsrc": cmd_valsrc, "eval": cmd_eval, "oracle": cmd_oracle,
     "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
