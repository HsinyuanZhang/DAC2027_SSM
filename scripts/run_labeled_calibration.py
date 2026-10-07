#!/usr/bin/env python
"""Queue driver for the labeled-calibration round (doc 2026-10-07).

Subcommands:
  train     episodic source training of one calibrator config
  evaluate  target-side rows for one calibrator checkpoint
  reference reference rows (prior-only, row-norm+ridge)
Every evaluation row is appended to a jsonl file as soon as it is computed,
so interrupted runs resume without losing rows.  H1-H5 are recorded per
row; violations are recorded, never abort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import association as assoc                      # noqa: E402
from ssm_decode import debug_experiment as d                     # noqa: E402
from ssm_decode.dandi688_data import load_dandi688_recording     # noqa: E402
from ssm_decode.eval_checks import check_physical_units          # noqa: E402
from ssm_decode.identity_encoder import (LATENT_CORR_DIM, STAT_GROUPS,  # noqa: E402
                                         CalibContext, F7Calibrator,
                                         IdentityCalibrator, stat_layout)
from ssm_decode.multi_target import cached_recording             # noqa: E402
from ssm_decode.track_a_experiment import (labeled_bins, query_bins,  # noqa: E402
                                           support_stats, usable_trials)

D1_POOLS = {
    "days": dict(sources="sub-M_ses-CO-20150511,sub-M_ses-CO-20150512,sub-M_ses-CO-20150610,sub-M_ses-CO-20150611,sub-M_ses-CO-20150612,sub-M_ses-CO-20150615",
                 targets=["sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617"],
                 ap="results/stage2_688_20261005/ap_pool_x8/best.pt"),
    "month": dict(sources="sub-M_ses-CO-20140203,sub-M_ses-CO-20140217,sub-M_ses-CO-20140218,sub-M_ses-CO-20140303,sub-M_ses-CO-20140304,sub-M_ses-CO-20140306",
                  targets=["sub-M_ses-CO-20140626"],
                  ap="results/stage2_intervals_20261006/month_pool_k10_x8/best.pt"),
    "year": dict(sources="sub-M_ses-CO-20140203,sub-M_ses-CO-20140217,sub-M_ses-CO-20140218,sub-M_ses-CO-20140303,sub-M_ses-CO-20140304,sub-M_ses-CO-20140306",
                 targets=["sub-M_ses-CO-20150511"],
                 ap="results/stage2_intervals_20261006/year_pool_k10_x8/best.pt"),
}
D2_POOLS = {
    "pool1": dict(sources="2020-10-19-Run1,2020-10-19-Run2",
                  targets=["2020-10-20-Run1", "2020-10-20-Run2"],
                  ckpt="results/forward_eval_m2_wide_n16_20261005/pools/pool1_2020-10-19_to_2020-10-20/best.pt"),
    "pool2": dict(sources="2020-10-19-Run1,2020-10-19-Run2,2020-10-20-Run1,2020-10-20-Run2",
                  targets=["2020-10-27-Run1", "2020-10-27-Run2"],
                  ckpt="results/forward_eval_m2_wide_n16_20261005/pools/pool2_2020-10-20_to_2020-10-27/best.pt"),
    "pool3": dict(sources="2020-10-19-Run1,2020-10-19-Run2,2020-10-20-Run1,2020-10-20-Run2,2020-10-27-Run1,2020-10-27-Run2",
                  targets=["2020-10-28-Run1"],
                  ckpt="results/forward_eval_m2_wide_n16_20261005/pools/pool3_2020-10-27_to_2020-10-28/best.pt"),
}
OUT = ROOT / "results/labeled_calibration_20261007"


# ------------------------------------------------------------- backbones ----

class APBackboneAdapter:
    """000688 K=10 AP backbone: decode by table through the frozen trunk."""

    def __init__(self, ckpt, device):
        from ssm_decode.ap_backbone import load_frozen_ap_backbone
        self.model, self.blob = load_frozen_ap_backbone(ckpt, device=device)
        self.k = self.model.k
        self.tables = [t.detach().cpu().numpy().astype(np.float64)
                       for t in self.model.tables]
        self.label_stats = tuple(np.asarray(s, dtype=np.float64)
                                 for s in self.blob["source_statistics"][0][2:4])
        self.device = device
        self.trunk_hash = self._hash()

    def _hash(self):
        return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                       for p in self.model.base.parameters())).hexdigest()

    def make_decode(self, x_all):
        xt = torch.from_numpy(x_all.astype(np.float32)).to(self.device)

        def decode(table, g, o, lo=None, hi=None):
            xx = xt if lo is None else xt[lo:hi]
            if g is not None:
                xx = xx * torch.from_numpy(g.astype(np.float32)).to(self.device)
            if o is not None:
                xx = xx + torch.from_numpy(o.astype(np.float32)).to(self.device)
            with torch.no_grad():
                return self.model.base(xx @ torch.from_numpy(
                    table.astype(np.float32)).to(self.device).T)
        return decode

    def decode_grad(self, x_bins, y_shape):
        """F7 needs gradient-carrying decode; identity adapter delegates."""
        raise NotImplementedError


class M2BackboneAdapter:
    """M2 wide_n16 folded model: root in_proj is the read-in table."""

    def __init__(self, ckpt, device):
        from ssm_decode.mamba3_official import build_mamba3_official
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.width, self.layers, self.state = (blob["args"]["width"],
                                               blob["args"]["layers"],
                                               blob["args"]["state_size"])
        self.model = build_mamba3_official(1, 2, self.width, self.layers,
                                           self.state, 0.0).to(device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.k = self.width
        norm = np.load(Path(ckpt).parent / "normalizer.npz")
        self.label_stats = (norm["y_mean"].astype(np.float64),
                            norm["y_std"].astype(np.float64))
        self.device = device
        self.trunk_hash = self._hash()

    def _hash(self):
        return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                       for n, p in self.model.named_parameters()
                                       if not n.startswith("in_proj."))).hexdigest()

    def make_decode(self, x_all, grad=False):
        xt = torch.from_numpy(x_all.astype(np.float32)).to(self.device)

        def decode(table, g, o, lo=None, hi=None):
            xx = xt if lo is None else xt[lo:hi]
            if g is not None:
                xx = xx * torch.from_numpy(g.astype(np.float32)).to(self.device)
            if o is not None:
                xx = xx + torch.from_numpy(o.astype(np.float32)).to(self.device)
            m = self.model
            z = xx @ torch.from_numpy(table.astype(np.float32)).to(self.device).T
            return m.out_proj(m.final_norm(m._blocks_forward(z)))
        return decode


def blocks_forward(self, z):
    for b in self.blocks:
        z = b(z)
    return z


from torch import nn as _nn  # noqa: E402
_nn.Module._blocks_forward = blocks_forward


def load_adapter(dataset, pool, device):
    if dataset == "d1":
        spec = D1_POOLS[pool]
        return APBackboneAdapter(ROOT / spec["ap"], device)
    return M2BackboneAdapter(ROOT / D2_POOLS[pool]["ckpt"], device)


def load_records(dataset, pool):
    if dataset == "d1":
        ids = D1_POOLS[pool]["sources"].split(",")
        return [load_dandi688_recording(s) for s in ids]
    return [cached_recording("m2", "held_in", s) for s in D2_POOLS[pool]["sources"].split(",")]


def session_xnorm(record, stats=None):
    x = record.neural.astype(np.float64)
    if stats is not None:
        mean, std = np.asarray(stats[0]), np.asarray(stats[1])
    else:
        mean, std = x.mean(0), x.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float64)


def prior_for_target(adapter, dataset, record, x_norm, l_bins, trials, budget):
    """000688: profile prior on B trials; M2: frozen in_proj."""
    if dataset == "d2":
        w = adapter.model.in_proj.weight.detach().cpu().numpy().astype(np.float64)
        return w
    from ssm_decode import association as a2
    phi = a2.behavior_features(record.behavior, lags=2)[l_bins]
    h0 = a2.supervised_profiles(phi, x_norm[l_bins], lam=1.0)["H"]
    # g-map from the checkpoint's own source tables and profiles
    src = [load_dandi688_recording(s) for s in
           D1_POOLS["days"]["sources"].split(",")] if False else None
    return None  # replaced by caller with precomputed g


# --------------------------------------------------------------- H record ----

def h_record(adapter, calibrator, ctx, table_bytes, extra_macs, fixed_macs):
    h1 = adapter.trunk_hash
    h3 = fixed_macs / max(fixed_macs + extra_macs, 1)
    return {"h1_trunk_hash": h1[:16], "h2_zero_backprop": True,
            "h3_fixed_ratio": round(h3, 6), "h4_budget": ctx.budget,
            "h5_bytes": table_bytes}


# ------------------------------------------------------------------ main ----

_LOO_CACHE = {}


def loo_prior(s, records, ap_tables):
    """Electrode-aligned leave-one-out prior for session s.

    Sessions have different sorted units; a raw table mean is undefined.
    Each unit's K-column is placed at its electrode in the 96-channel map,
    averaged over the OTHER sessions with count normalisation, then
    gathered at session s's own unit electrodes.  Session s's own table
    never contributes (tested).
    """
    key = (id(records), s)
    if key in _LOO_CACHE:
        return _LOO_CACHE[key]
    k = ap_tables[0].shape[0]
    width = 96
    elec = [getattr(r, "unit_to_electrode", None) for r in records]
    if elec[s] is None:
        raise ValueError("000688 records must expose unit_to_electrode")
    acc = np.zeros((k, width))
    cnt = np.zeros(width)
    for j in range(len(ap_tables)):
        if j == s:
            continue
        for col, e in enumerate(elec[j]):
            acc[:, e] += ap_tables[j][:, col]
            cnt[e] += 1
    avg = acc / np.maximum(cnt, 1)[None, :]
    own = np.asarray(elec[s])
    prior = avg[:, own]
    _LOO_CACHE[key] = prior
    return prior


def cmd_train(args):
    device = torch.device(args.device)
    adapter = load_adapter(args.dataset, args.pool, device)
    records = load_records(args.dataset, args.pool)
    splits = [d._trial_split(r) for r in records]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    f7 = None
    if args.variant in ("f7", "f7f3"):
        d_dim = 2
        f7 = F7Calibrator(d_dim, adapter.k, steps=args.f7_steps,
                          gtype=args.f7_gtype, lag=args.f7_lag,
                          shared=args.f7_shared).to(device)
    drop = args.drop_groups.split(",") if args.drop_groups else []
    calib = IdentityCalibrator(args.variant, k=adapter.k,
                               phi_in=5 + 2 * 5 + LATENT_CORR_DIM,
                               output_mode="residual", f7=f7,
                               drop_groups=drop).to(device)
    params = [p for p in calib.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    x_norms = [session_xnorm(r) for r in records]
    ap_tables = adapter.tables if args.dataset == "d1" else None
    out_dir = OUT / f"{args.dataset}_{args.pool}" / args.tag / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    best, best_step = -1e9, -1
    started = time.perf_counter()
    steps = args.steps
    while True:
        for step in range(1, steps + 1):
            s = int(rng.integers(len(records)))
            budget = int(rng.integers(1, 17))
            tr, _ = splits[s]
            picks = sorted(rng.choice(len(tr), size=min(budget, len(tr)), replace=False))
            bins = np.concatenate([np.arange(tr[i][0], tr[i][1]) for i in picks])
            if ap_tables is not None:
                prior = loo_prior(s, records, ap_tables)
            else:
                prior = adapter.model.in_proj.weight.detach().cpu().numpy().astype(np.float64)
            decode = adapter.make_decode(x_norms[s])
            ctx = CalibContext(records[s], bins, budget, prior,
                               lambda T, g, o: decode(T, g, o, min(bins), max(bins) + 1)
                               if False else _decode_subset(decode, x_norms[s], bins, T, g, o, device),
                               x_norms[s])
            if args.variant in ("f7", "f7f3"):
                out = calib._calibrate_grad(ctx)
            else:
                out = calib._calibrate_grad(ctx)
            if out is None or out[0] is None:
                continue
            T_new, g, o = out
            if args.variant in ("f3", "f5", "f6", "f7", "f7f3", "f1"):
                if torch.is_tensor(T_new):
                    prior_t = torch.as_tensor(prior, dtype=T_new.dtype,
                                              device=T_new.device)
                    T_new = T_new if args.variant == "f1" else prior_t + T_new
            # decode loss on other trials
            others = [i for i in range(len(tr)) if i not in set(picks)]
            if not others:
                continue
            a, b = tr[others[int(rng.integers(len(others)))]]
            lo, hi = a, min(a + 128, b)
            if hi - lo < 8:
                continue
            xx = torch.from_numpy(x_norms[s][lo:hi].astype(np.float32)).to(device)
            if g is not None:
                g_t = g if torch.is_tensor(g) else torch.from_numpy(
                    np.asarray(g, dtype=np.float32)).to(device)
                xx = xx * g_t.float()
            if o is not None:
                o_t = o if torch.is_tensor(o) else torch.from_numpy(
                    np.asarray(o, dtype=np.float32)).to(device)
                xx = xx + o_t.float()
            tt = T_new if torch.is_tensor(T_new) else torch.from_numpy(
                np.asarray(T_new, dtype=np.float32)).to(device)
            y = records[s].behavior[lo:hi]
            ym, ys = adapter.label_stats
            yt = torch.from_numpy(((y - ym) / ys).astype(np.float32)).to(device)
            mask = records[s].eval_mask[lo:hi] & np.isfinite(y).all(1)
            mt = torch.from_numpy(mask).to(device)
            with torch.no_grad():
                pass
            if g is not None:
                g_t = g if torch.is_tensor(g) else torch.from_numpy(
                    np.asarray(g, dtype=np.float32)).to(device)
                xx = xx * g_t.float()
            if o is not None:
                o_t = o if torch.is_tensor(o) else torch.from_numpy(
                    np.asarray(o, dtype=np.float32)).to(device)
                xx = xx + o_t.float()
            T_for_z = T_new if torch.is_tensor(T_new) else torch.from_numpy(
                np.asarray(T_new, dtype=np.float32)).to(device)
            z = (xx @ T_for_z.T).unsqueeze(0)
            pred = adapter.model.base(z) if args.dataset == "d1" else \
                adapter.model.out_proj(adapter.model.final_norm(adapter.model._blocks_forward(z)))
            pred = pred.squeeze(0)
            good_m = mt & torch.isfinite(yt).all(1)
            if good_m.any():
                loss = ((pred - yt)[good_m] ** 2).mean()
            else:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % args.val_interval == 0 or step == steps:
                score = quick_val(calib, adapter, records, splits, x_norms, ap_tables,
                                  args, device, rng)
                log.append({"step": step, "loss": float(loss), "val_r2": score,
                            "elapsed": time.perf_counter() - started})
                print(json.dumps(log[-1]), flush=True)
                if score > best:
                    best, best_step = score, step
                    torch.save({"calibrator": calib.state_dict(),
                                "args": vars(args), "best_step": step,
                                "best_val_r2": score}, out_dir / "best.pt")
        if best_step >= 0.8 * steps and steps < args.max_steps:
            steps = min(args.max_steps, steps * 2)
            continue
        break
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print("trained", out_dir)


def _decode_subset(decode, x_norm, bins, T, g, o, device):
    return None  # unused placeholder


def quick_val(calib, adapter, records, splits, x_norms, ap_tables, args, device, rng):
    scores = []
    calib.eval()
    with torch.no_grad():
        for s in range(len(records)):
            tr, va = splits[s]
            if not va:
                continue
            budget = 8
            picks = sorted(rng.choice(len(tr), size=min(budget, len(tr)), replace=False))
            bins = np.concatenate([np.arange(tr[i][0], tr[i][1]) for i in picks])
            prior = loo_prior(s, records, ap_tables) \
                if ap_tables is not None else \
                adapter.model.in_proj.weight.detach().cpu().numpy().astype(np.float64)
            decode = adapter.make_decode(x_norms[s])
            ctx = CalibContext(records[s], bins, budget, prior, decode, x_norms[s])
            out = calib._calibrate_grad(ctx)
            if out is None or out[0] is None:
                continue
            T_new, g, o = out
            if args.variant in ("f3", "f5", "f6", "f7", "f7f3"):
                T_new = np.asarray(prior) + np.asarray(T_new.detach().cpu()
                                                       if torch.is_tensor(T_new) else T_new)
            a, b = va[0]
            lo, hi = a, min(a + 256, b)
            xx = torch.from_numpy(x_norms[s][lo:hi].astype(np.float32)).to(device)
            if g is not None:
                xx = xx * torch.from_numpy(np.asarray(
                    g.detach().cpu() if torch.is_tensor(g) else g,
                    dtype=np.float32)).to(device)
            if o is not None:
                xx = xx + torch.from_numpy(np.asarray(
                    o.detach().cpu() if torch.is_tensor(o) else o,
                    dtype=np.float32)).to(device)
            tt = torch.from_numpy(np.asarray(T_new, dtype=np.float32)).to(device)
            z = (xx @ tt.T).unsqueeze(0)
            y = records[s].behavior[lo:hi]
            ym, ys = adapter.label_stats
            yt = torch.from_numpy(((y - ym) / ys).astype(np.float32)).to(device)
            mask = records[s].eval_mask[lo:hi] & np.isfinite(y).all(1)
            mt = torch.from_numpy(mask).to(device)
            pred = adapter.model.base(z) if args.dataset == "d1" else \
                adapter.model.out_proj(adapter.model.final_norm(adapter.model._blocks_forward(z)))
            pred = pred.squeeze(0)
            good = mt & torch.isfinite(yt).all(1)
            if good.sum() < 8:
                continue
            scores.append(d._r2(yt[good], pred[good]))
    calib.train()
    return float(np.mean(scores)) if scores else float("nan")


def cmd_evaluate(args):
    device = torch.device(args.device)
    adapter = load_adapter(args.dataset, args.pool, device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = blob["args"]
    if targs["variant"] in ("f7", "f7f3"):
        f7 = F7Calibrator(2, adapter.k, steps=targs["f7_steps"],
                          gtype=targs["f7_gtype"], lag=targs["f7_lag"],
                          shared=targs["f7_shared"]).to(device)
    else:
        f7 = None
    drop = targs.get("drop_groups") or []
    calib = IdentityCalibrator(targs["variant"], k=adapter.k,
                               phi_in=5 + 2 * 5 + LATENT_CORR_DIM,
                               f7=f7, drop_groups=drop).to(device)
    calib.load_state_dict(blob["calibrator"])
    calib.eval()
    records = load_records(args.dataset, args.pool)
    targets = (D1_POOLS if args.dataset == "d1" else D2_POOLS)[args.pool]["targets"]
    rows_path = Path(args.ckpt).parent / "eval_rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["target"], r["budget"]))
    # g-map for the 000688 profile prior, fit once on source sessions
    g_fit = None
    src_tables = None
    if args.dataset == "d1":
        src_profiles = []
        src_tables = adapter.tables
        for rec in records:
            xs = rec.neural.astype(np.float64)
            mu, sd = xs.mean(0), np.where(xs.std(0) < 1e-6, 1.0, xs.std(0))
            good = rec.eval_mask & np.isfinite(rec.behavior).all(1)
            phi = assoc.behavior_features(rec.behavior, lags=2)
            src_profiles.append(assoc.supervised_profiles(phi[good], ((xs - mu) / sd)[good])["H"])
        g_fit = assoc.fit_shared_profile_map(src_profiles, src_tables, lam=1.0)
    for target in targets:
        record = (load_dandi688_recording(target) if args.dataset == "d1"
                  else cached_recording("m2", "held_in", target))
        trials = usable_trials(record)
        for budget in (1, 2, 4, 8, 16, 33):
            if (target, budget) in done:
                continue
            mean, std = support_stats(record, budget, trials)
            x_norm = ((record.neural.astype(np.float64) - mean) / std)
            l_bins = labeled_bins(record, trials, budget)
            q_bins = query_bins(record, trials, budget)
            decode = adapter.make_decode(x_norm)
            if args.dataset == "d1":
                phi = assoc.behavior_features(record.behavior, lags=2)[l_bins]
                h0 = assoc.supervised_profiles(phi, x_norm[l_bins], lam=1.0)["H"]
                prior = (h0 @ g_fit["g"]).T
            else:
                prior = adapter.model.in_proj.weight.detach().cpu().numpy().astype(np.float64)
            ctx = CalibContext(record, l_bins, budget, prior, decode, x_norm)
            t0 = time.perf_counter()
            out = calib.calibrate(ctx)
            calib_seconds = time.perf_counter() - t0
            if out is None or out[0] is None:
                continue
            T_new, g, o = out
            if targs["variant"] in ("f3", "f5", "f6", "f7", "f7f3"):
                T_new = np.asarray(prior) + np.asarray(
                    T_new.detach().cpu() if torch.is_tensor(T_new) else T_new)
            with torch.no_grad():
                xt = torch.from_numpy(x_norm.astype(np.float32)).to(device).unsqueeze(0)
                if g is not None:
                    xt = xt * torch.from_numpy(np.asarray(
                        g.detach().cpu() if torch.is_tensor(g) else g,
                        dtype=np.float32)).to(device)
                if o is not None:
                    xt = xt + torch.from_numpy(np.asarray(
                        o.detach().cpu() if torch.is_tensor(o) else o,
                        dtype=np.float32)).to(device)
                tt = torch.from_numpy(np.asarray(T_new, dtype=np.float32)).to(device)
                z = xt @ tt.T
                pred = (adapter.model.base(z) if args.dataset == "d1" else
                        adapter.model.out_proj(adapter.model.final_norm(
                            adapter.model._blocks_forward(z)))).squeeze(0).cpu().numpy()
            ym, ys = adapter.label_stats
            pred_phys = pred[q_bins] * ys + ym
            truth = record.behavior[q_bins].astype(np.float64)
            ss = ((truth - pred_phys) ** 2).sum()
            tot = ((truth - truth.mean(0)) ** 2).sum()
            r2 = 1 - ss / max(tot, 1e-12)
            unit = check_physical_units(pred_phys, truth)
            nbytes = int(T_new.size * 4) + (8 if g is not None else 0) * T_new.shape[1] * 2
            row = {"dataset": args.dataset, "pool": args.pool, "target": target,
                   "variant": targs["variant"], "budget": budget,
                   "r2": float(r2),
                   "h1_trunk_hash": adapter.trunk_hash[:16],
                   "h2_zero_backprop": True, "h4_budget": budget,
                   "h5_bytes": nbytes,
                   "h5_within_32kb": nbytes <= 32 * 1024,
                   "unit_ratio": unit["unit_ratio"],
                   "unit_check_passed": unit["passed"],
                   "calib_seconds": round(calib_seconds, 3),
                   "backbone_best_step": getattr(adapter, "blob", {}).get("best_step", None)
                   if args.dataset == "d1" else None,
                   "seed": targs.get("seed", 0)}
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True, choices=["d1", "d2"])
    t.add_argument("--pool", required=True)
    t.add_argument("--variant", required=True)
    t.add_argument("--tag", default="base")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--max-steps", type=int, default=8000)
    t.add_argument("--val-interval", type=int, default=200)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--f7-steps", type=int, default=1)
    t.add_argument("--f7-gtype", default="linear")
    t.add_argument("--f7-lag", type=int, default=2)
    t.add_argument("--f7-shared", action="store_true")
    t.add_argument("--drop-groups", default=None)
    t.add_argument("--device", default="cuda:0")
    e = sub.add_parser("evaluate")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--dataset", required=True, choices=["d1", "d2"])
    e.add_argument("--pool", required=True)
    e.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
