#!/usr/bin/env python
"""Train / evaluate driver for population-conditioned calibration (line A).

Subcommands
  train      episodic source training of one (stratum, pool, variant, seed)
  evaluate   target rows for a trained run (resumable, one jsonl row at a time)
  reference  P-prior and ui_* gradient reference rows for a stratum
  oracle     P-oracle diagnostic rows (descriptors from all target trials)

Contract: target-side code paths run under torch.no_grad and use only the B
chosen labeled trials; backprop exists only in ``train`` (source sessions).
Never touches FINAL sessions or results/official_evalai/.
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

from scripts.run_labeled_calibration import (D1_POOLS, APBackboneAdapter,  # noqa: E402
                                             loo_prior)
from ssm_decode import association as assoc                                # noqa: E402
from ssm_decode.dandi688_data import (DEV_SESSIONS, FINAL_SESSIONS,        # noqa: E402
                                      load_dandi688_recording)
from ssm_decode.eval_checks import check_physical_units                    # noqa: E402
from ssm_decode.population_calib import (PopStats, PopulationCalibrator,   # noqa: E402
                                         SessionData, desc_dim, describe,
                                         electrode_mean_table, mac_counts,
                                         norm_stats)
from ssm_decode.track_a_experiment import (_prior_correct, _source_profiles,  # noqa: E402
                                           ui_grad_table)

OUT = ROOT / "results/population_calib_20261015"
B_GRID = (1, 2, 4, 8, 16, 33)
B_SCREEN = (1, 8, 33)
N_DRAWS = 8
INVENTORY = OUT / "session_inventory.json"


# ------------------------------------------------------------------ world ----

def sessions_for(stratum: str, pool: str):
    """Source session ids (train + val ordering by date) for pool O or E."""
    spec = D1_POOLS[stratum]
    orig = spec["sources"].split(",")
    if pool == "O":
        return orig
    inv = json.loads(INVENTORY.read_text())
    target_date = min(t.split("-")[-1] for t in spec["targets"])
    banned = set(DEV_SESSIONS) | set(FINAL_SESSIONS) | set(spec["targets"])
    ids = sorted(r["session"] for r in inv
                 if r["date"] < target_date and r["session"] not in banned)
    assert set(orig) <= set(ids), (stratum, set(orig) - set(ids))
    return ids


class World:
    def __init__(self, stratum, pool, device, load_sessions=True):
        self.stratum, self.pool, self.device = stratum, pool, device
        spec = D1_POOLS[stratum]
        self.adapter = APBackboneAdapter(ROOT / spec["ap"], device)
        for p in self.adapter.model.parameters():
            p.requires_grad_(False)
        self.k = self.adapter.k
        self.label_stats = self.adapter.label_stats
        self.orig_ids = spec["sources"].split(",")
        orig_recs = {s: load_dandi688_recording(s) for s in self.orig_ids}
        self.orig_elec = [np.asarray(orig_recs[s].unit_to_electrode) for s in self.orig_ids]
        self.orig_tables = self.adapter.tables
        self.ids = sessions_for(stratum, pool)
        n_val = 1 if pool == "O" else 2
        self.val_ids = self.ids[-n_val:]
        self.train_ids = self.ids[:-n_val]
        self.sessions, self.priors = {}, {}
        self._avg = {}
        if load_sessions:
            for s in self.ids:
                rec = orig_recs.get(s) or load_dandi688_recording(s)
                self.add_session(s, rec)

    def prior_avg(self, exclude):
        if exclude not in self._avg:
            self._avg[exclude] = electrode_mean_table(self.orig_tables, self.orig_elec,
                                                      exclude=exclude)
        return self._avg[exclude]

    def add_session(self, name, rec):
        sd = SessionData(rec, self.label_stats, self.device, name=name)
        exclude = self.orig_ids.index(name) if name in self.orig_ids else None
        avg = self.prior_avg(exclude)                          # LOO for original sources
        self.sessions[name] = sd
        self.priors[name] = torch.from_numpy(avg[:, sd.electrodes]).to(self.device)
        return sd

    def target(self, name):
        rec = load_dandi688_recording(name)
        sd = SessionData(rec, self.label_stats, self.device, name=name)
        prior = torch.from_numpy(self.prior_avg(None)[:, sd.electrodes]).to(self.device)
        return sd, prior


# ---------------------------------------------------------------- helpers ----

def draw_trials(name, B, draw, n_trials):
    seed = zlib.crc32(f"{name}-{B}-{draw}".encode())
    return np.sort(np.random.default_rng(seed).choice(n_trials, size=B, replace=False))


def decode_full(world, sd, table, mean, std):
    """One contiguous causal pass of the frozen trunk over the whole session."""
    xz = (sd.x32 - mean.float()) / std.float()
    with torch.inference_mode():
        return world.adapter.model.base((xz @ table.float().T).unsqueeze(0)).squeeze(0)


def r2_physical(world, sd, pred_norm, qmask):
    ym, ys = world.label_stats
    pred = pred_norm[qmask].double().cpu().numpy() * ys + ym
    truth = sd.record.behavior[qmask.cpu().numpy()].astype(np.float64)
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12), check_physical_units(pred, truth)


def eval_point(world, sd, prior, calib, idx, sd_stats=None):
    """Calibrate from trials ``idx`` and score the query bins (no autograd)."""
    stats = PopStats.from_trials((sd_stats or sd).bank, idx)
    t0 = time.perf_counter()
    with torch.no_grad():
        if calib is None:                                      # P-prior: no calibration
            mean, std = norm_stats(stats)
            table = prior.float()
        else:
            res = calib.calibrate(stats, prior)
            mean, std, table = res.mean, res.std, res.table
    seconds = time.perf_counter() - t0
    pred = decode_full(world, sd, table, mean, std)
    r2, unit = r2_physical(world, sd, pred, sd.query_mask(idx))
    return r2, unit, table, seconds, stats


def build_calibrator(args_like, world):
    c = PopulationCalibrator(args_like["variant"], world.k, desc_dim(world.k),
                             m=args_like.get("m", 8), output=args_like.get("output", "residual"),
                             label_free=bool(args_like.get("label_free", False)))
    return c.to(world.device)


def val_score(world, calib, rng_seed=1234, bs=B_GRID, draws=2, with_prior=False):
    scores = []
    calib.eval() if calib is not None else None
    for name in world.val_ids:
        sd, prior = world.sessions[name], world.priors[name]
        for B in bs:
            for draw in range(draws):
                idx = draw_trials(f"val-{name}", B, draw + rng_seed, sd.bank.n_trials)
                r2, _, _, _, _ = eval_point(world, sd, prior, None if with_prior else calib, idx)
                scores.append(r2)
    if calib is not None:
        calib.train()
    return float(np.mean(scores))


# ------------------------------------------------------------------ train ----

def sample_B(rng, n_trials):
    b = int(math.floor(math.exp(rng.uniform(0.0, math.log(34.0)))))
    return max(1, min(b, 33, n_trials - 2))


def make_episode(world, rng, names, window=256, burn=64, p_drop_max=0.3):
    name = names[int(rng.integers(len(names)))]
    sd, prior = world.sessions[name], world.priors[name]
    n_tr = sd.bank.n_trials
    B = sample_B(rng, n_tr)
    idx = np.sort(rng.choice(n_tr, size=B, replace=False))
    n = sd.n_units
    p = rng.uniform(0.0, p_drop_max)
    keep = np.flatnonzero(rng.random(n) >= p)
    min_keep = min(n, max(6, n // 2))
    if len(keep) < min_keep:
        keep = np.sort(rng.choice(n, size=min_keep, replace=False))
    chosen = torch.zeros(n_tr, dtype=torch.bool, device=sd.device)
    chosen[torch.as_tensor(idx, device=sd.device)] = True
    for _ in range(30):
        t0 = int(rng.integers(0, sd.n_bins - window))
        tid = sd.trial_id[t0:t0 + window]
        cal = (tid >= 0) & chosen[tid.clamp_min(0)]
        mask = sd.valid[t0:t0 + window] & ~cal
        mask[:burn] = False
        if int(mask.sum()) >= 24:
            return sd, prior, idx, keep, t0, mask, window
    return None


def episode_table(world, calib, sd, prior, idx, keep, grad=True):
    stats = PopStats.from_trials(sd.bank, idx)
    if len(keep) < sd.n_units:
        stats = stats.select_units(keep)
        prior = prior[:, torch.as_tensor(keep, device=prior.device)]
    d, r, mean, std = describe(stats, prior)
    return calib(d, r, prior.float()), mean, std, d, r


def fit_normalizers(world, calib, rng, n_ep=300):
    ds, rs = [], []
    for _ in range(n_ep):
        ep = make_episode(world, rng, world.train_ids)
        if ep is None:
            continue
        sd, prior, idx, keep, *_ = ep
        stats = PopStats.from_trials(sd.bank, idx)
        d, r, _, _ = describe(stats, prior)
        ds.append(d), rs.append(r)
    calib.set_normalizers(torch.cat(ds), torch.cat(rs))


def cmd_train(args):
    device = torch.device(args.device)
    run_dir = OUT / "runs" / f"{args.stratum}_{args.pool}" / args.tag / f"seed{args.seed}"
    if (run_dir / "done.json").exists():
        print("skip, done:", run_dir)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    world = World(args.stratum, args.pool, device)
    calib = build_calibrator(vars(args), world)
    fit_normalizers(world, calib, rng)
    params = [p for p in calib.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    ym, ys = [torch.as_tensor(s, dtype=torch.float32, device=device) for s in world.label_stats]
    prior_val = val_score(world, None, with_prior=True)
    steps, step, best, best_step, doublings = args.steps, 0, -1e9, 0, 0
    log_path = run_dir / "train_log.jsonl"
    log_path.write_text("")
    started = time.perf_counter()
    since_best = 0
    stopped_early = False
    while True:
        while step < steps:
            step += 1
            tables, xs, ys_w, masks = [], [], [], []
            for _ in range(args.batch):
                ep = make_episode(world, rng, world.train_ids)
                if ep is None:
                    continue
                sd, prior, idx, keep, t0, mask, w = ep
                table, mean, std, _, _ = episode_table(world, calib, sd, prior, idx, keep)
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
                score = val_score(world, calib)
                rec = {"step": step, "loss": float(loss), "val_r2": score,
                       "val_prior_r2": prior_val, "elapsed": round(time.perf_counter() - started, 1)}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
                if score > best:
                    best, best_step, since_best = score, step, 0
                    torch.save({"calibrator": calib.state_dict(), "args": vars(args),
                                "best_step": step, "best_val_r2": score,
                                "val_prior_r2": prior_val}, run_dir / "best.pt")
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
            "train_ids": world.train_ids, "val_ids": world.val_ids,
            "seconds": round(time.perf_counter() - started, 1)}
    (run_dir / "done.json").write_text(json.dumps(done, indent=1))
    print("trained", run_dir, json.dumps({k: done[k] for k in ("best_step", "best_val_r2", "converged")}))


# --------------------------------------------------------------- evaluate ----

_BWD = {"n": 0}


def _patch_backward_counter():
    orig = torch.autograd.backward

    def counting(*a, **k):
        _BWD["n"] += 1
        return orig(*a, **k)
    torch.autograd.backward = counting


def load_run(run_dir, world):
    blob = torch.load(Path(run_dir) / "best.pt", map_location=world.device, weights_only=False)
    calib = build_calibrator(blob["args"], world)
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    for p in calib.parameters():
        p.requires_grad_(False)
    done = json.loads((Path(run_dir) / "done.json").read_text())
    return calib, blob, done


def h_fields(world, sd, stats, calib, table, B, idx, trunk_hash_before):
    n_units = table.shape[1]
    macs = mac_counts(calib.variant if calib is not None else "p0", n_units,
                      int(stats.d["n"]), world.k, desc_dim(world.k))
    if calib is None:                                  # prior table: nothing is computed
        macs = {"accumulate_macs": 0, "fixed_weight_macs": 0, "h3_fixed_ratio": 0.0}
    nbytes = int(table.numel() * 4)
    q = sd.query_mask(idx).cpu().numpy()
    used = np.zeros(sd.n_bins, dtype=bool)
    for i in idx:
        a, b = sd.trials[i]
        used[a:b] = True
    return {"h1_trunk_unchanged": bool(world.adapter._hash() == trunk_hash_before),
            "h1_trunk_hash": trunk_hash_before[:16],
            "h2_backward_calls": _BWD["n"], "h2_zero_backprop": _BWD["n"] == 0,
            "h3_fixed_ratio": round(macs["h3_fixed_ratio"], 6),
            "h3_fixed_macs": macs["fixed_weight_macs"], "h3_accumulate_macs": macs["accumulate_macs"],
            "h4_budget": int(B), "h4_labeled_bins": int(stats.d["n"]),
            "h4_query_disjoint": bool(not (used & q).any()),
            "h5_bytes": nbytes, "h5_within_32kb": nbytes <= 32 * 1024}


def append_row(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("variant", "pool", "target", "B", "draw", "r2") if k in row}),
          flush=True)


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add((r["variant"], r["target"], r["B"], r["draw"]))
    return keys


def shuffled_label_session(world, sd):
    """Control: same neural data, behavior circularly shifted by half a session."""
    from types import SimpleNamespace
    rec = sd.record
    rec2 = SimpleNamespace(neural=rec.neural, eval_mask=rec.eval_mask,
                           trial_bounds=rec.trial_bounds, session=rec.session,
                           unit_to_electrode=rec.unit_to_electrode,
                           behavior=np.roll(np.asarray(rec.behavior), len(rec.behavior) // 2, 0))
    return SessionData(rec2, world.label_stats, world.device, name=sd.name + "-shufY",
                       usable=sd.trials)


def cmd_evaluate(args):
    _patch_backward_counter()
    device = torch.device(args.device)
    world = World(args.stratum, args.pool, device, load_sessions=False)
    run_dir = OUT / "runs" / f"{args.stratum}_{args.pool}" / args.tag / f"seed{args.seed}"
    calib, blob, done = load_run(run_dir, world)
    rows_path = run_dir / "eval_rows.jsonl"
    seen = done_keys(rows_path)
    a = blob["args"]
    variant = a["variant"] + (f"_m{a['m']}" if a["variant"] in ("p3", "p4") else "")
    if a.get("label_free"):
        variant += "_nolab"
    if args.shuffle_labels:
        variant += "_shufY"
        rows_path = run_dir / "eval_rows_shufY.jsonl"
        seen = done_keys(rows_path)
    hash_before = world.adapter._hash()
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        sd_stats = shuffled_label_session(world, sd) if args.shuffle_labels else sd
        for B in (args.budgets or B_GRID):
            for draw in range(args.draws):
                if (variant, target, B, draw) in seen:
                    continue
                idx = draw_trials(target, B, draw, sd.bank.n_trials)
                r2, unit, table, secs, stats = eval_point(world, sd, prior, calib, idx,
                                                          sd_stats=sd_stats)
                row = {"stratum": args.stratum, "pool": args.pool, "variant": variant,
                       "tag": args.tag, "seed": args.seed, "target": target, "B": B,
                       "draw": draw, "r2": r2, "unit_ratio": unit["unit_ratio"],
                       "unit_check_passed": unit["passed"], "n_units": int(table.shape[1]),
                       "calib_seconds": round(secs, 4), "output": a.get("output", "residual"),
                       "best_step": done["best_step"], "converged": done["converged"],
                       "val_score": done["best_val_r2"], "diagnostic_only": False,
                       **h_fields(world, sd, stats, calib, table, B, idx, hash_before)}
                append_row(rows_path, row)


def _ui_rows_for(world, sd, prior, target, B, draw, idx, g_fit, label_stats):
    stats = PopStats.from_trials(sd.bank, idx)
    mean, std = norm_stats(stats)
    xz = ((sd.x32.double() - mean) / std).cpu().numpy()
    rec = sd.record
    y_norm = (rec.behavior.astype(np.float64) - label_stats[0]) / label_stats[1]
    windows = [sd.trials[i] for i in idx]
    valid = rec.eval_mask & np.isfinite(rec.behavior).all(1)
    out = {}
    elec_init = prior.cpu().numpy()
    t_e, _ = ui_grad_table(world.adapter.model, xz, y_norm, windows, elec_init,
                           device=world.device, steps=300, lr=3e-3, seed=0, valid_mask=valid)
    out["ui_elec"] = t_e
    l_bins = np.concatenate([np.arange(a, b)[valid[a:b]] for a, b in windows])
    phi = assoc.behavior_features(rec.behavior, lags=2)[l_bins]
    h0 = assoc.supervised_profiles(phi, xz[l_bins], lam=1.0)["H"]
    prior_w0 = (h0 @ g_fit["g"]).T
    pc = _prior_correct(world.adapter.model.head.weight.detach().cpu().numpy().astype(np.float64),
                        prior_w0, rec, l_bins, mean.cpu().numpy(), std.cpu().numpy(), label_stats)
    t_p, _ = ui_grad_table(world.adapter.model, xz, y_norm, windows, pc["table"],
                           device=world.device, steps=300, lr=3e-3, seed=0, valid_mask=valid)
    out["ui_prior"] = t_p
    return out, mean, std


def cmd_reference(args):
    """P-prior (no calibration) and non-deployable gradient references."""
    _patch_backward_counter()
    device = torch.device(args.device)
    world = World(args.stratum, "O", device, load_sessions=False)
    rows_path = OUT / "reference" / f"{args.stratum}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(rows_path)
    hash_before = world.adapter._hash()
    # shared profile map for the original ui_prior init (sources = trunk sessions)
    recs = [load_dandi688_recording(s) for s in world.orig_ids]
    g_fit = assoc.fit_shared_profile_map(_source_profiles(recs, 2), world.orig_tables, lam=1.0)
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for B in B_GRID:
            for draw in range(args.draws):
                idx = draw_trials(target, B, draw, sd.bank.n_trials)
                base = {"stratum": args.stratum, "pool": "-", "tag": "ref", "seed": 0,
                        "target": target, "B": B, "draw": draw, "diagnostic_only": False}
                if ("p_prior", target, B, draw) not in seen:
                    r2, unit, table, secs, stats = eval_point(world, sd, prior, None, idx)
                    append_row(rows_path, {**base, "variant": "p_prior", "r2": r2,
                                           "unit_ratio": unit["unit_ratio"],
                                           "unit_check_passed": unit["passed"],
                                           "n_units": int(table.shape[1]),
                                           **h_fields(world, sd, stats, None, table, B, idx,
                                                      hash_before)})
                if B in B_SCREEN and draw < args.ui_draws and \
                        ("ui_elec", target, B, draw) not in seen:
                    tabs, mean, std = _ui_rows_for(world, sd, prior, target, B, draw, idx,
                                                   g_fit, world.label_stats)
                    q = sd.query_mask(idx)
                    for name, tab in tabs.items():
                        pred = decode_full(world, sd, torch.from_numpy(tab).float().to(device),
                                           mean, std)
                        r2, unit = r2_physical(world, sd, pred, q)
                        append_row(rows_path, {**base, "variant": name, "r2": r2,
                                               "unit_ratio": unit["unit_ratio"],
                                               "unit_check_passed": unit["passed"],
                                               "note": "gradient through trunk; non-deployable reference"})


def cmd_oracle(args):
    """P-oracle: descriptors from ALL labeled trials of the target (diagnostic)."""
    _patch_backward_counter()
    device = torch.device(args.device)
    world = World(args.stratum, args.pool, device, load_sessions=False)
    run_dir = OUT / "runs" / f"{args.stratum}_{args.pool}" / args.tag / f"seed{args.seed}"
    calib, blob, done = load_run(run_dir, world)
    a = blob["args"]
    variant = "p_oracle_" + a["variant"] + (f"_m{a['m']}" if a["variant"] in ("p3", "p4") else "")
    rows_path = run_dir / "oracle_rows.jsonl"
    seen = done_keys(rows_path)
    hash_before = world.adapter._hash()
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        n_tr = sd.bank.n_trials
        for kind in ("all", "half"):
            for draw in range(1 if kind == "all" else args.draws):
                name = variant + ("" if kind == "all" else "_half")
                if (name, target, n_tr if kind == "all" else n_tr // 2, draw) in seen:
                    continue
                if kind == "all":
                    idx = np.arange(n_tr)
                    stats = PopStats.from_trials(sd.bank, idx)
                    with torch.no_grad():
                        res = calib.calibrate(stats, prior, oracle=True)
                    assert res.diagnostic_only
                    pred = decode_full(world, sd, res.table, res.mean, res.std)
                    q = sd.valid & (sd.trial_id >= 0)        # in-sample by construction
                    r2, unit = r2_physical(world, sd, pred, q)
                else:
                    perm = np.random.default_rng(zlib.crc32(f"half-{target}-{draw}".encode())
                                                 ).permutation(n_tr)
                    idx = np.sort(perm[: n_tr // 2])
                    stats = PopStats.from_trials(sd.bank, idx)
                    with torch.no_grad():
                        res = calib.calibrate(stats, prior, oracle=True)
                    pred = decode_full(world, sd, res.table, res.mean, res.std)
                    r2, unit = r2_physical(world, sd, pred, sd.query_mask(idx))
                row = {"stratum": args.stratum, "pool": args.pool, "variant": name,
                       "tag": args.tag, "seed": args.seed, "target": target,
                       "B": n_tr if kind == "all" else len(idx), "draw": draw, "r2": r2,
                       "unit_ratio": unit["unit_ratio"], "diagnostic_only": True,
                       "oracle_kind": kind, "best_step": done["best_step"],
                       "note": "diagnostic only: descriptors use target trials beyond the budget",
                       **h_fields(world, sd, stats, calib, res.table, len(idx), idx, hash_before)}
                row["B"] = n_tr if kind == "all" else len(idx)
                append_row(rows_path, row)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "evaluate", "oracle", "reference"):
        s = sub.add_parser(name)
        s.add_argument("--stratum", required=True, choices=list(D1_POOLS))
        s.add_argument("--device", default="cuda:0")
        if name != "reference":
            s.add_argument("--pool", required=True, choices=["O", "E"])
            s.add_argument("--variant", default="p0")
            s.add_argument("--tag", default=None)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--m", type=int, default=8)
            s.add_argument("--output", default="residual", choices=["residual", "absolute"])
            s.add_argument("--label-free", action="store_true")
        if name == "train":
            s.add_argument("--steps", type=int, default=1000)
            s.add_argument("--min-steps", type=int, default=600)
            s.add_argument("--patience", type=int, default=8)
            s.add_argument("--val-interval", type=int, default=100)
            s.add_argument("--batch", type=int, default=8)
            s.add_argument("--lr", type=float, default=1e-3)
        if name in ("evaluate", "oracle", "reference"):
            s.add_argument("--draws", type=int, default=N_DRAWS)
            s.add_argument("--budgets", type=int, nargs="*", default=None)
            s.add_argument("--shuffle-labels", action="store_true")
        if name == "reference":
            s.add_argument("--ui-draws", type=int, default=5)
    args = p.parse_args()
    if getattr(args, "tag", None) is None and args.cmd != "reference":
        args.tag = args.variant + (f"_m{args.m}" if args.variant in ("p3", "p4") else "") + \
            ("" if args.output == "residual" else "_abs") + ("_nolab" if args.label_free else "")
    {"train": cmd_train, "evaluate": cmd_evaluate, "oracle": cmd_oracle,
     "reference": cmd_reference}[args.cmd](args)


if __name__ == "__main__":
    main()
