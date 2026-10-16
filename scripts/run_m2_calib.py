#!/usr/bin/env python
"""FALCON M2 (fixed channels) population calibration driver (Colab round, 2026-10-16).

Subcommands
  train         episodic source training of P0 for one (backbone seed, pool, calibrator seed)
  select-r1     choose the R1-stream rule (lambda, kind, burn) per B by SOURCE simulation
  evaluate      target rows: P-prior (+ diagnostics), P0, P0 int8, P0+R1-stream, R1 on prior
  select-uireg  choose ui_reg (lambda, lr, steps) per B by SOURCE simulation
  eval-uireg    ui_reg reference rows (NON-DEPLOYABLE: gradient through the frozen trunk)

Contract: candidate rows run under torch.no_grad on the B chosen labeled trials only (the
backward counter is recorded per row, H2); the trunk is never updated (H1: hash); selection
uses SOURCE sessions only; targets are the held-in sessions of D2_POOLS (never held-out /
final splits, never results/official_evalai/).  Decoding is one contiguous causal pass over the
whole session; only query bins are scored.
"""
from __future__ import annotations

import argparse
import itertools
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

from ssm_decode import fwd_refine as fr                                   # noqa: E402
from ssm_decode import m2_calib as m2                                     # noqa: E402
from ssm_decode.population_calib import PopStats, norm_stats              # noqa: E402

OUT = Path(os.environ.get("M2_OUT", str(ROOT / "results/m2_calib_20261016")))
B_GRID = m2.B_GRID
N_DRAWS = 8
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
KINDS = ("full", "diag")
BURNS = (0, 10, 20)
UIREG_LAMS = (0.0, 1e-2, 1e-1)
UIREG_LRS = (1e-3, 3e-3, 1e-2)
UIREG_STEPS = (10, 30, 100, 300)
SEL_DRAWS = 2
SCALE_GRID = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0)


def run_dir(args, kind="runs"):
    tag = getattr(args, "tag", "p0")
    return OUT / kind / f"seed{args.seed}" / args.pool / (f"c{args.cseed}" if tag == "p0" else f"c{args.cseed}_{tag}")


def variant_suffix(args):
    tag = getattr(args, "tag", "p0")
    return "" if tag == "p0" else f"_{tag.replace('p0_', '')}"


# ------------------------------------------------------------------ train ----

def sample_B(rng, n_trials):
    b = int(math.floor(math.exp(rng.uniform(0.0, math.log(34.0)))))
    return max(1, min(b, 33, n_trials - 2))


def make_episode(world, rng, names, window=256, burn=64, p_drop_max=0.1):
    name = names[int(rng.integers(len(names)))]
    sd = world.sessions[name]
    tr = world.train_trials(name)
    B = sample_B(rng, len(tr))
    idx = np.sort(rng.choice(tr, size=B, replace=False))
    n = sd.n_units
    p = rng.uniform(0.0, p_drop_max)
    keep = np.flatnonzero(rng.random(n) >= p)
    min_keep = min(n, max(6, n // 2))
    if len(keep) < min_keep:
        keep = np.sort(rng.choice(n, size=min_keep, replace=False))
    chosen = torch.zeros(sd.bank.n_trials, dtype=torch.bool, device=sd.device)
    chosen[torch.as_tensor(idx, device=sd.device)] = True
    vs = world.val_start[name]
    for _ in range(30):
        t0 = int(rng.integers(0, vs - window))
        tid = sd.trial_id[t0:t0 + window]
        cal = (tid >= 0) & chosen[tid.clamp_min(0)]
        mask = sd.valid[t0:t0 + window] & ~cal
        mask[:burn] = False
        if int(mask.sum()) >= 24:
            return name, idx, keep, t0, mask, window
    return None


def episode_table(world, calib, name, idx, keep):
    sd = world.sessions[name]
    prior_tab, prior_bias = world.priors[name]
    stats = PopStats.from_trials(sd.bank, idx)
    if len(keep) < sd.n_units:
        stats = stats.select_units(keep)
        prior_tab = prior_tab[:, torch.as_tensor(keep, device=prior_tab.device)]
    d, r, mean, std = m2.describe_for(calib, stats, prior_tab)
    return calib(d, r, prior_tab.float()), mean, std, d, r, prior_bias


def fit_normalizers(world, calib, rng, n_ep=300):
    ds, rs = [], []
    for _ in range(n_ep):
        ep = make_episode(world, rng, world.source_ids)
        if ep is None:
            continue
        name, idx, *_ = ep
        sd = world.sessions[name]
        d, r, _, _ = m2.describe_for(calib, PopStats.from_trials(sd.bank, idx),
                                     world.priors[name][0])
        ds.append(d), rs.append(r)
    calib.set_normalizers(torch.cat(ds), torch.cat(rs))


def eval_point(world, sd, prior, calib, idx, qmask):
    """Calibrate (calib=None: prior) from trials ``idx``; score ``qmask`` bins (no autograd)."""
    prior_tab, prior_bias = prior
    stats = PopStats.from_trials(sd.bank, idx)
    with torch.no_grad():
        if calib is None:
            mean, std = norm_stats(stats)
            table = prior_tab.float()
        else:
            res = calib.calibrate(stats, prior_tab)
            mean, std, table = res.mean, res.std, res.table
    pred = m2.decode_full(world.trunk, sd, table, prior_bias, mean, std)
    return m2.r2_physical(world.label_stats, sd, pred, qmask)[0]


def val_score(world, calib, with_prior=False, draws=2, bs=B_GRID):
    scores = []
    for name in world.source_ids:
        sd, prior = world.sessions[name], world.priors[name]
        tr = world.train_trials(name)
        qmask = world.val_mask(name)
        for B in bs:
            for draw in range(draws):
                idx = tr[m2.draw_trials(f"m2val-{name}", B, draw + 1234, len(tr))]
                scores.append(eval_point(world, sd, prior, None if with_prior else calib, idx, qmask))
    return float(np.mean(scores))


def cmd_train(args):
    device = torch.device(args.device)
    rd = run_dir(args)
    if (rd / "done.json").exists():
        print("skip, done:", rd)
        return
    rd.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1000 * args.seed + args.cseed)
    rng = np.random.default_rng(1000 + 10 * args.seed + args.cseed)
    world = m2.M2World(args.seed, args.pool, device)
    calib = m2.build_calibrator(world, rank=args.rank, hidden=args.hidden, gain=not args.no_gain,
                               label_free=args.label_free)
    fit_normalizers(world, calib, rng)
    params = [p for p in calib.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    prior_val = val_score(world, None, with_prior=True)
    steps, step, best, best_step, doublings = args.steps, 0, -1e9, 0, 0
    log_path = rd / "train_log.jsonl"
    log_path.write_text("")
    started = time.perf_counter()
    since_best, stopped_early = 0, False
    while True:
        while step < steps:
            step += 1
            zs, ys, ms = [], [], []
            for _ in range(args.batch):
                ep = make_episode(world, rng, world.source_ids)
                if ep is None:
                    continue
                name, idx, keep, t0, mask, w = ep
                sd = world.sessions[name]
                table, mean, std, _, _, bias = episode_table(world, calib, name, idx, keep)
                kk = torch.as_tensor(keep, device=device)
                xw = (sd.x32[t0:t0 + w][:, kk] - mean.float()) / std.float()
                zs.append(xw @ table.T + bias.float())
                ys.append(sd.y_norm[t0:t0 + w].float())
                ms.append(mask)
            if not zs:
                continue
            pred = world.trunk.run(torch.stack(zs))
            err = (pred - torch.stack(ys)) ** 2
            m = torch.stack(ms).unsqueeze(-1).float()
            loss = (err * m).sum() / (m.sum() * err.shape[-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % args.val_interval == 0 or step == steps:
                calib.eval()
                score = val_score(world, calib)
                calib.train()
                rec = {"step": step, "loss": float(loss), "val_r2": score, "val_prior_r2": prior_val,
                       "elapsed": round(time.perf_counter() - started, 1)}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
                if score > best:
                    best, best_step, since_best = score, step, 0
                    torch.save({"calibrator": calib.state_dict(), "args": vars(args),
                                "best_step": step, "best_val_r2": score,
                                "val_prior_r2": prior_val}, rd / "best.pt")
                else:
                    since_best += 1
                if since_best >= args.patience and step >= args.min_steps:
                    stopped_early = True
                    break
        if stopped_early:
            break
        if best_step > 0.8 * steps and doublings < 4:
            steps *= 2
            doublings += 1
            print(f"best_step {best_step} in last 20%; extend budget to {steps}", flush=True)
            continue
        break
    converged = bool(stopped_early or best_step <= 0.8 * steps)
    done = {"best_step": best_step, "best_val_r2": best, "val_prior_r2": prior_val,
            "final_budget": steps, "doublings": doublings, "stopped_early": stopped_early,
            "converged": converged, "last_step": step, "args": vars(args),
            "source_ids": world.source_ids, "trunk_dir": str(world.trunk.pool_dir),
            "seconds": round(time.perf_counter() - started, 1)}
    (rd / "done.json").write_text(json.dumps(done, indent=1))
    print("trained", rd, json.dumps({k: done[k] for k in ("best_step", "best_val_r2", "converged")}))


def load_calibrator(args, world):
    rd = run_dir(args)
    blob = torch.load(rd / "best.pt", map_location=world.device, weights_only=False)
    a = blob["args"]
    calib = m2.build_calibrator(world, rank=a["rank"], hidden=a["hidden"], gain=not a["no_gain"],
                                label_free=a.get("label_free", False))
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    for p in calib.parameters():
        p.requires_grad_(False)
    return calib, blob, json.loads((rd / "done.json").read_text())


# ----------------------------------------------------------- select R1 ----

def sim_episodes(world, names, draws=SEL_DRAWS):
    for name in names:
        tr = world.train_trials(name)
        for B in B_GRID:
            for draw in range(draws):
                yield name, B, draw, tr[m2.draw_trials(f"m2sel-{name}", B, draw, len(tr))]


def cmd_select_r1(args):
    device = torch.device(args.device)
    world = m2.M2World(args.seed, args.pool, device)
    calib, _, _ = load_calibrator(args, world)
    path = run_dir(args) / "select_r1.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["base"], r["session"], r["B"], r["draw"]))
    t0 = time.perf_counter()
    trunk = world.trunk
    for name, B, draw, idx in sim_episodes(world, world.source_ids):
        sd, prior = world.sessions[name], world.priors[name]
        qmask = world.val_mask(name)
        for base in ("p0", "prior"):
            if (base, name, B, draw) in seen:
                continue
            stats = PopStats.from_trials(sd.bank, idx)
            with torch.no_grad():
                if base == "p0":
                    res = calib.calibrate(stats, prior[0])
                    mean, std, table = res.mean, res.std, res.table
                else:
                    mean, std = norm_stats(stats)
                    table = prior[0].float()
            y0, h = m2.decode_full(trunk, sd, table, prior[1], mean, std, want_h=True)
            rows = []
            r2_off = m2.r2_physical(world.label_stats, sd, y0, qmask)[0]
            rows.append({"lam": None, "kind": "full", "burn": 0, "r2": r2_off})
            for burn in BURNS:
                st = m2.stream_readout_stats(trunk, sd, prior, calib, idx, burn=burn,
                                             use_calib=(base == "p0"))
                for kind, lam in itertools.product(KINDS, LAMS):
                    v = m2.ridge_v(trunk, st, lam, kind)
                    hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                    y = hb @ v.to(hb.dtype).T
                    rows.append({"lam": lam, "kind": kind, "burn": burn,
                                 "r2": m2.r2_physical(world.label_stats, sd, y, qmask)[0]})
            with path.open("a") as f:
                for r in rows:
                    f.write(json.dumps({"base": base, "session": name, "B": B, "draw": draw, **r}) + "\n")
        print(f"[select-r1 s{args.seed} {args.pool} c{args.cseed}] {name} B={B} d={draw} "
              f"t={time.perf_counter() - t0:.0f}s", flush=True)
    write_r1_selection(run_dir(args))


def write_r1_selection(rd):
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in (rd / "select_r1.jsonl").read_text().splitlines()])
    df["lam"] = df["lam"].astype(object).where(df["lam"].notna(), None)
    df["lam_key"] = df["lam"].map(lambda x: "off" if x is None else repr(float(x)))
    g = df.groupby(["base", "B", "lam_key", "kind", "burn"]).r2.agg(["mean", "count"]).reset_index()
    sel = {}
    for (base, B), sub in g.groupby(["base", "B"]):
        best = sub.loc[sub["mean"].idxmax()]
        off = sub[sub.lam_key == "off"]["mean"].iloc[0]
        sel.setdefault(base, {})[str(B)] = {
            "lam": None if best.lam_key == "off" else float(best.lam_key), "kind": best.kind,
            "burn": int(best.burn), "source_r2": float(best["mean"]), "source_r2_off": float(off),
            "n_episodes": int(best["count"])}
    (rd / "selection_r1.json").write_text(json.dumps(sel, indent=1))
    g.to_csv(rd / "select_r1_grid.csv", index=False)
    return sel


# --------------------------------------------------------------- evaluate ----

_BWD = {"n": 0}


def _patch_backward_counter():
    orig = torch.autograd.backward

    def counting(*a, **k):
        _BWD["n"] += 1
        return orig(*a, **k)
    torch.autograd.backward = counting
    return orig


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add((r["variant"], r["target"], r["B"], r["draw"]))
    return keys


def append_rows(path, rows):
    with Path(path).open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def used_bins(sd, idx):
    used = np.zeros(sd.n_bins, dtype=bool)
    for i in idx:
        a, b = sd.trials[i]
        used[a:b] = True
    return used


def cmd_evaluate(args):
    _patch_backward_counter()
    device = torch.device(args.device)
    world = m2.M2World(args.seed, args.pool, device, load_sources=True)
    trunk = world.trunk
    calib, blob, done = load_calibrator(args, world)
    rd = run_dir(args)
    sel = json.loads((rd / "selection_r1.json").read_text()) if (rd / "selection_r1.json").exists() else None
    path = rd / "eval_rows.jsonl"
    seen = done_keys(path)
    scale_sel = json.loads(scale_path(args).read_text()) if scale_path(args).exists() else None
    hash0 = trunk.trunk_hash
    sfx = variant_suffix(args)
    tbytes = m2.table_bytes(world.k, m2.N_CHANNELS)
    src_mean, src_std = m2.srcnorm_stats(trunk, device)
    t0 = time.perf_counter()
    for target in world.targets:
        sd, prior = world.target(target)
        n_tr = sd.bank.n_trials
        for B in (args.budgets or B_GRID):
            for draw in range(args.draws):
                idx = m2.draw_trials(target, B, draw, n_tr)
                qmask = sd.query_mask(idx)
                disjoint = bool(not (used_bins(sd, idx) & qmask.cpu().numpy()).any())
                stats = PopStats.from_trials(sd.bank, idx)
                base = {"seed": args.seed, "pool": args.pool, "cseed": args.cseed, "target": target,
                        "B": B, "draw": draw, "n_trials": n_tr, "h4_budget": int(B),
                        "h4_query_disjoint": disjoint, "h4_labeled_bins": int(stats.d["n"]),
                        "tag": args.tag, "trunk_dir": trunk.pool_dir.name, "best_step": done["best_step"],
                        "val_score": done["best_val_r2"]}
                rows = []

                def emit(variant, r2, unit, table_bytes_, **extra):
                    rows.append({**base, "variant": variant, "r2": r2, "unit_ratio": unit["unit_ratio"],
                                 "unit_check_passed": unit["passed"], "h5_bytes": table_bytes_,
                                 "h5_within_32kb": table_bytes_ <= 32 * 1024,
                                 "h2_backward_calls": _BWD["n"], "h2_zero_backprop": _BWD["n"] == 0,
                                 **extra})

                def score(y):
                    return m2.r2_physical(world.label_stats, sd, y, qmask)

                _BWD["n"] = 0
                fp32 = tbytes["fp32_bytes"]
                with torch.no_grad():
                    mean_b, std_b = norm_stats(stats)
                    # ---- reference rows (calibrator-free; only written for cseed 0) ----------
                    if args.cseed == 0 and args.tag == "p0":
                        if ("p_prior", target, B, draw) not in seen:
                            y = m2.decode_full(trunk, sd, prior[0].float(), prior[1], mean_b, std_b)
                            emit("p_prior", *score(y), fp32)
                        if ("p_prior_srcnorm", target, B, draw) not in seen:
                            y = m2.decode_full(trunk, sd, prior[0].float(), prior[1], src_mean, src_std)
                            emit("p_prior_srcnorm", *score(y), fp32)
                        if ("p_prior_nobias", target, B, draw) not in seen:
                            y = m2.decode_full(trunk, sd, prior[0].float(), torch.zeros_like(prior[1]),
                                               mean_b, std_b)
                            emit("p_prior_nobias", *score(y), fp32)
                        if scale_sel is not None and ("p_prior_scale", target, B, draw) not in seen:
                            c = scale_sel[str(B)]["c"]
                            y = m2.decode_full(trunk, sd, prior[0].float() * c, prior[1], mean_b, std_b)
                            emit("p_prior_scale", *score(y), fp32, rule=scale_sel[str(B)])
                        if sel is not None and ("p_prior_r1s", target, B, draw) not in seen:
                            rule = sel["prior"][str(B)]
                            y0, h = m2.decode_full(trunk, sd, prior[0].float(), prior[1], mean_b, std_b,
                                                   want_h=True)
                            if rule["lam"] is None:
                                y = y0
                            else:
                                st = m2.stream_readout_stats(trunk, sd, prior, calib, idx,
                                                             burn=rule["burn"], use_calib=False)
                                v = m2.ridge_v(trunk, st, rule["lam"], rule["kind"])
                                hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                                y = hb @ v.to(hb.dtype).T
                            emit("p_prior_r1s", *score(y), fp32, rule=rule)
                    # ---- P0 --------------------------------------------------------------------
                    if args.tag == "p0" and ("p0_swap", target, B, draw) not in seen:
                        # CONTROL: table calibrated from the most recent SOURCE session's B trials
                        # (target-independent information), decoded with the target's own B-trial
                        # normalisation.  If p0_swap ~ p0 the gain is a static prior correction.
                        src = world.source_ids[-1]
                        sd_s, tr_s = world.sessions[src], world.train_trials(src)
                        idx_s = tr_s[m2.draw_trials(f"swap-{src}", B, draw, len(tr_s))]
                        res_s = calib.calibrate(PopStats.from_trials(sd_s.bank, idx_s), prior[0])
                        y = m2.decode_full(trunk, sd, res_s.table, prior[1], mean_b, std_b)
                        emit("p0_swap", *score(y), fp32, diagnostic_only=True, swap_session=src)
                    need = [v for v in ("p0", "p0_int8", "p0_r1s") if (v + sfx, target, B, draw) not in seen]
                    need_sens = [f"p0_r1s_lam{lam:g}" for lam in LAMS
                                 if (f"p0_r1s_lam{lam:g}{sfx}", target, B, draw) not in seen]
                    if need or (need_sens and sel is not None):
                        res = calib.calibrate(stats, prior[0])
                        y0, h = m2.decode_full(trunk, sd, res.table, prior[1], res.mean, res.std,
                                               want_h=True)
                        if "p0" in need:
                            emit("p0" + sfx, *score(y0), fp32)
                        if "p0_int8" in need:
                            tq, nb = m2.quantize_table_int8(res.table)
                            y = m2.decode_full(trunk, sd, tq, prior[1], res.mean, res.std)
                            emit("p0_int8" + sfx, *score(y), nb)
                        if sel is not None and ("p0_r1s" in need or need_sens):
                            rule = sel["p0"][str(B)]
                            hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                            stbur = {}

                            def stream(burn):
                                if burn not in stbur:
                                    stbur[burn] = m2.stream_readout_stats(trunk, sd, prior, calib, idx,
                                                                          burn=burn, use_calib=True)
                                return stbur[burn]
                            if "p0_r1s" in need:
                                if rule["lam"] is None:
                                    y = y0
                                else:
                                    v = m2.ridge_v(trunk, stream(rule["burn"]), rule["lam"], rule["kind"])
                                    y = hb @ v.to(hb.dtype).T
                                emit("p0_r1s" + sfx, *score(y), fp32, rule=rule)
                            # sensitivity (TARGET-evaluated, diagnostic only, never used for selection)
                            for lam in LAMS:
                                nm = f"p0_r1s_lam{lam:g}"
                                if nm in need_sens:
                                    v = m2.ridge_v(trunk, stream(rule["burn"]), lam, "full")
                                    y = hb @ v.to(hb.dtype).T
                                    emit(nm + sfx, *score(y), fp32, diagnostic_only=True)
                assert trunk._hash() == hash0, "trunk changed"
                for r in rows:
                    r["h1_trunk_unchanged"] = True
                    r["h1_trunk_hash"] = hash0[:16]
                if rows:
                    append_rows(path, rows)
                print(f"[eval s{args.seed} {args.pool} c{args.cseed}] {target[-8:]} B={B} d={draw} "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)


def scale_path(args):
    return OUT / "runs" / f"seed{args.seed}" / args.pool / "selection_scale.json"


def cmd_select_scale(args):
    """One-parameter deployable baseline: prior table times a scalar c(B) chosen on SOURCE sessions."""
    device = torch.device(args.device)
    world = m2.M2World(args.seed, args.pool, device)
    rows = []
    for name, B, draw, idx in sim_episodes(world, world.source_ids):
        sd, prior = world.sessions[name], world.priors[name]
        qmask = world.val_mask(name)
        mean, std = norm_stats(PopStats.from_trials(sd.bank, idx))
        for c in SCALE_GRID:
            y = m2.decode_full(world.trunk, sd, prior[0].float() * c, prior[1], mean, std)
            rows.append({"session": name, "B": B, "draw": draw, "c": c,
                         "r2": m2.r2_physical(world.label_stats, sd, y, qmask)[0]})
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["B", "c"]).r2.mean().reset_index()
    sel = {}
    for B, sub in g.groupby("B"):
        best = sub.loc[sub.r2.idxmax()]
        sel[str(B)] = {"c": float(best.c), "source_r2": float(best.r2),
                       "source_r2_c1": float(sub[sub.c == 1.0].r2.iloc[0])}
    scale_path(args).write_text(json.dumps(sel, indent=1))
    g.to_csv(scale_path(args).with_name("select_scale_grid.csv"), index=False)
    print(json.dumps(sel))


# ---------------------------------------------------------------- ui_reg ----

def uireg_prep(world, sd, idx):
    stats = PopStats.from_trials(sd.bank, idx)
    mean, std = norm_stats(stats)
    xz = ((sd.x32.double() - mean) / std).cpu().numpy()
    rec = sd.record
    y_norm = (rec.behavior - world.label_stats[0]) / world.label_stats[1]
    valid = rec.eval_mask & np.isfinite(rec.behavior).all(1)
    windows = [sd.trials[i] for i in idx]
    return mean, std, xz, y_norm, valid, windows


def uireg_scores(world, sd, prior, idx, qmask, combos, init=None, snapshots=UIREG_STEPS):
    mean, std, xz, y_norm, valid, windows = uireg_prep(world, sd, idx)
    tab0 = prior[0].cpu().numpy() if init is None else init
    for lam, lr in combos:
        traj = m2.ui_reg_trajectory_m2(world.trunk, xz, y_norm, windows, tab0, prior[1].cpu().numpy(),
                                       prior[0].cpu().numpy() if init is None else init, lam=lam, lr=lr,
                                       snapshots=snapshots, device=world.device, seed=0, valid_mask=valid)
        for step, tab in traj.items():
            y = m2.decode_full(world.trunk, sd, torch.from_numpy(tab).float().to(world.device), prior[1],
                               mean, std)
            yield lam, lr, step, m2.r2_physical(world.label_stats, sd, y, qmask)[0]


def cmd_select_uireg(args):
    device = torch.device(args.device)
    world = m2.M2World(args.seed, args.pool, device)
    rd = OUT / "uireg" / f"seed{args.seed}" / args.pool
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / "select_uireg.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["session"], r["B"], r["draw"], r["lam"], r["lr"]))
    combos_all = list(itertools.product(UIREG_LAMS, UIREG_LRS))
    t0 = time.perf_counter()
    # the most recent ``sel_sessions`` source sessions (closest to the targets); the Mamba-3 backward on
    # 24k table parameters costs ~33 ms/step, so the simulation is kept to <= 2 sessions per pool
    sel_names = world.source_ids[-args.sel_sessions:]
    for name, B, draw, idx in sim_episodes(world, sel_names, draws=args.sel_draws):
        sd, prior = world.sessions[name], world.priors[name]
        combos = [c for c in combos_all if (name, B, draw, c[0], c[1]) not in seen]
        if not combos:
            continue
        with path.open("a") as f:
            for lam, lr, step, r2 in uireg_scores(world, sd, prior, idx, world.val_mask(name), combos):
                f.write(json.dumps({"session": name, "B": B, "draw": draw, "lam": lam, "lr": lr,
                                    "steps": step, "r2": r2}) + "\n")
        print(f"[select-uireg s{args.seed} {args.pool}] {name} B={B} d={draw} "
              f"t={time.perf_counter() - t0:.0f}s", flush=True)
    write_uireg_selection(rd)


def write_uireg_selection(rd):
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in (rd / "select_uireg.jsonl").read_text().splitlines()])
    g = df.groupby(["B", "lam", "lr", "steps"]).r2.agg(["mean", "count"]).reset_index()
    sel = {}
    for B, sub in g.groupby("B"):
        best = sub.loc[sub["mean"].idxmax()]
        sel[str(B)] = {"lam": float(best.lam), "lr": float(best.lr), "steps": int(best.steps),
                       "source_r2": float(best["mean"]), "n_episodes": int(best["count"])}
    (rd / "selection_uireg.json").write_text(json.dumps(sel, indent=1))
    g.to_csv(rd / "select_uireg_grid.csv", index=False)
    return sel


def cmd_eval_uireg(args):
    _patch_backward_counter()
    device = torch.device(args.device)
    world = m2.M2World(args.seed, args.pool, device, load_sources=False)
    rd = OUT / "uireg" / f"seed{args.seed}" / args.pool
    sel = json.loads((rd / "selection_uireg.json").read_text())
    path = rd / "eval_uireg.jsonl"
    seen = done_keys(path)
    t0 = time.perf_counter()
    for target in world.targets:
        sd, prior = world.target(target)
        for B in (args.budgets or B_GRID):
            s = sel[str(B)]
            for draw in range(args.draws):
                if ("ui_reg", target, B, draw) in seen:
                    continue
                idx = m2.draw_trials(target, B, draw, sd.bank.n_trials)
                qmask = sd.query_mask(idx)
                (lam, lr, step, r2), = [x for x in uireg_scores(world, sd, prior, idx, qmask,
                                        [(s["lam"], s["lr"])], snapshots=(s["steps"],))]
                row = {"seed": args.seed, "pool": args.pool, "target": target, "B": B, "draw": draw,
                       "variant": "ui_reg", "r2": r2, "rule": s, "diagnostic_only": False,
                       "nondeployable": True,
                       "note": "gradient through the frozen trunk onto the read-in table; reference only"}
                append_rows(path, [row])
                print(f"[eval-uireg s{args.seed} {args.pool}] {target[-8:]} B={B} d={draw} r2={r2:.3f} "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------- main ----

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "select-r1", "select-scale", "evaluate", "select-uireg", "eval-uireg"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0, help="backbone seed (0,1,2)")
        s.add_argument("--pool", required=True, choices=["pool1", "pool2", "pool3"])
        s.add_argument("--cseed", type=int, default=0, help="calibrator seed")
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--tag", default="p0", help="calibrator variant tag (p0, p0_nogain, p0_nolab, ...)")
        if name == "train":
            s.add_argument("--steps", type=int, default=1000)
            s.add_argument("--min-steps", type=int, default=600)
            s.add_argument("--patience", type=int, default=8)
            s.add_argument("--val-interval", type=int, default=100)
            s.add_argument("--batch", type=int, default=8)
            s.add_argument("--lr", type=float, default=1e-3)
            s.add_argument("--rank", type=int, default=16)
            s.add_argument("--hidden", type=int, default=64)
            s.add_argument("--no-gain", action="store_true")
            s.add_argument("--label-free", action="store_true")
        if name in ("evaluate", "eval-uireg"):
            s.add_argument("--draws", type=int, default=N_DRAWS)
            s.add_argument("--budgets", type=int, nargs="*", default=None)
        if name == "select-uireg":
            s.add_argument("--sel-draws", type=int, default=1)
            s.add_argument("--sel-sessions", type=int, default=2)
    args = p.parse_args()
    {"train": cmd_train, "select-r1": cmd_select_r1, "select-scale": cmd_select_scale, "evaluate": cmd_evaluate,
     "select-uireg": cmd_select_uireg, "eval-uireg": cmd_eval_uireg}[args.cmd](args)


if __name__ == "__main__":
    main()
