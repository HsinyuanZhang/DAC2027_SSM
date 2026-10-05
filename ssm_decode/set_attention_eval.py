"""Track E evaluation on 000688 targets (revision 4.7.5, 4.7.6).

Loads an episodic set-attention checkpoint and evaluates budget-dependent
folded tables on one target session:

- ``e1_profile``: descriptor from the first B labeled trials (labels enter
  the descriptor only).
- ``e1_rate``: label-free descriptor (profile block zeroed).  Works at B=0.
- ``e2_ema``: slow re-folding.  The recording is split into segments; each
  segment decodes with a table folded from a causal expanding descriptor
  (strictly past bins).
- ``e4_window``: accuracy ceiling.  The descriptor is augmented with causal
  activity statistics recomputed per window, so the key changes over time.

Set-property checks ride along: permutation invariance (row shuffle), unit
dropout curves at 10, 30, 50 percent, and per-decode diagnostics.  Query
labels never enter any fit or fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import association as assoc
from .dandi688_data import load_dandi688_recording
from .set_attention import load_set_backbone, unit_descriptors
from .track_a_experiment import (decode_r2_record, labeled_bins, query_bins,
                                 support_stats, usable_trials)

BUDGETS = (0, 2, 4, 8, 16, 33)


def _desc(model_blob, record, bins, with_profile):
    """Normalized descriptor.  The label-free form zeroes the profile block
    after normalization, matching the trainer's profile-dropout convention."""
    desc, info = unit_descriptors(record, bins, lags=model_blob["args"]["lags"],
                                  with_profile=with_profile)
    stats = model_blob["desc_stats"]
    normed = (desc - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    if not with_profile:
        profile_cols = desc.shape[1] - 2
        normed[:, :profile_cols] = 0.0
    return normed, info


def _fold(model, desc, device):
    with torch.inference_mode():
        table = model.read_in.fold_table(
            torch.from_numpy(desc.astype(np.float32)).to(device))
    return table.cpu().numpy().astype(np.float64)


def _decode_segments(model, blob, record, x_z, device, *, mode, seg_frac=0.1):
    """E2/E4 decode: re-fold per recording segment with causal descriptors.

    E2 folds a label-free activity-only descriptor from strictly past bins.
    E4 augments the descriptor with a causal activity window per segment, so
    the attention key changes over time (the dynamic-key ceiling).
    """
    n = len(x_z)
    preds = np.zeros((n, record.behavior.shape[1]), dtype=np.float64)
    seg_len = max(int(n * seg_frac), 200)
    for lo in range(0, n, seg_len):
        hi = min(lo + seg_len, n)
        if mode == "e2_ema":
            past = np.arange(0, lo)
            bins = past if len(past) > 100 else np.arange(lo, hi)
            desc, _ = unit_descriptors(record, bins, lags=blob["args"]["lags"],
                                       with_profile=False)
        else:
            window_bins = np.arange(max(0, lo - 300), hi)
            desc, _ = unit_descriptors(record, window_bins, lags=blob["args"]["lags"],
                                       with_profile=False)
        stats = blob["desc_stats"]
        desc_norm = (desc - np.asarray(stats["mean"])) / np.asarray(stats["std"])
        desc_norm[:, :desc.shape[1] - 2] = 0.0
        table = _fold(model, desc_norm, device)
        preds[lo:hi] = _forward(model, x_z[lo:hi], table, device)
    return preds


def _forward(model, x_seg, table, device):
    xt = torch.from_numpy(x_seg.astype(np.float32)).to(device).unsqueeze(0)
    table_t = torch.from_numpy(table.astype(np.float32)).to(device)
    with torch.inference_mode():
        out = model.base(xt @ table_t.T).squeeze(0)
    return out.detach().cpu().numpy()


def run(args):
    global model_global, model_blob_global
    device = torch.device(args.device)
    model, blob = load_set_backbone(args.checkpoint, device=device)
    model_global, model_blob_global = model, blob
    source_ids = blob["source_sessions"]
    if args.task == "dandi688sm":
        record = load_dandi688_recording(args.target_session)
    else:
        from .multi_target import cached_recording
        record = cached_recording(args.task, "held_in", args.target_session)
    label_stats = None  # E decode path stays in raw label units via record.behavior
    trials = usable_trials(record)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else list(BUDGETS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    started = time.perf_counter()
    finite = np.isfinite(record.behavior).all(1)
    for budget in budgets:
        mean, std = support_stats(record, budget, trials)
        x_z = ((record.neural.astype(np.float64) - mean) / std)
        l_bins = labeled_bins(record, trials, budget) if budget > 0 else np.empty(0, dtype=np.int64)
        q_bins = query_bins(record, trials, budget)
        common = {"budget": budget, "labeled_bins": int(len(l_bins)),
                  "query_bins": int(len(q_bins)), "method_family": "set_attention"}
        tables = {}
        if budget > 0:
            desc, info = _desc(blob, record, l_bins, with_profile=True)
            tables["e1_profile"] = _fold(model, desc, device)
            # Prior-corrected E1 table: closed-form channel correction on the fold.
            if args.prior_correct:
                records_src = [load_dandi688_recording(s) for s in source_ids]
                from .track_a_experiment import _prior_correct, _source_latent_stats
                from . import association as assoc2
                src_profiles = []
                for rec_s in records_src:
                    good = rec_s.eval_mask & np.isfinite(rec_s.behavior).all(1)
                    phi = assoc2.behavior_features(rec_s.behavior, lags=blob["args"]["lags"])
                    xs = rec_s.neural.astype(np.float64)
                    mu, sd = xs.mean(0), np.where(xs.std(0) < 1e-6, 1.0, xs.std(0))
                    src_profiles.append(assoc2.supervised_profiles(phi[good], ((xs - mu) / sd)[good])["H"])
                phi_l = assoc2.behavior_features(record.behavior, lags=blob["args"]["lags"])[l_bins]
                h0 = assoc2.supervised_profiles(phi_l, x_z[l_bins], lam=1.0)["H"]
                # g-map from folded source tables.
                src_tables = []
                for rec_s in records_src:
                    desc_s, _ = _desc(blob, rec_s, np.arange(len(rec_s.neural)), with_profile=True)
                    src_tables.append(_fold(model, desc_s, device))
                g_fit = assoc2.fit_shared_profile_map(src_profiles, src_tables, lam=1.0)
                prior_w0 = (h0 @ g_fit["g"]).T
                y_l = record.behavior[l_bins].astype(np.float64)
                y_mean_l = y_l.mean(0)
                y_std_l = np.where(y_l.std(0) < 1e-6, 1.0, y_l.std(0))
                corrected = assoc2.prior_correction_table(
                    model.head.weight.detach().cpu().numpy().astype(np.float64),
                    prior_w0, x_z[l_bins], (y_l - y_mean_l) / y_std_l, lam=1.0)
                tables["e1_prior_correct"] = corrected["table"]
        desc_free, _ = _desc(blob, record, np.arange(len(x_z)), with_profile=False)
        tables["e1_rate"] = _fold(model, desc_free, device)

        for name, table in tables.items():
            r2 = _r2_from_preds(_forward(model, x_z, table, device), record, q_bins)
            rows.append(_row(name, r2, common, extra={"table_norm": float(np.linalg.norm(table))}))

        # E2/E4 segment decode
        for mode in ("e2_ema", "e4_window"):
            preds = _decode_segments(model, blob, record, x_z, device, mode=mode)
            rows.append(_row(mode, _r2_from_preds(preds, record, q_bins), common, extra={}))

        # Set-property checks at the budget's main table (e1_profile or e1_rate).
        main_name = "e1_profile" if budget > 0 else "e1_rate"
        main_table = tables[main_name]
        perm = rng.permutation(main_table.shape[1])
        x_perm = x_z[:, perm]
        r2_perm = _r2_from_preds(_forward(model, x_perm, main_table[:, perm], device), record, q_bins)
        rows.append(_row(f"{main_name}_permcheck", r2_perm, common,
                         extra={"check": "unit-order permutation; must match main"}))
        for frac in (0.1, 0.3, 0.5):
            drop = rng.random(x_z.shape[1]) < frac
            keep = ~drop
            if keep.sum() < 4:
                continue
            r2_drop = _r2_from_preds(_forward(model, x_z[:, keep], main_table[:, keep], device),
                                     record, q_bins)
            rows.append(_row(f"{main_name}_unitdrop{int(frac * 100)}", r2_drop, common,
                             extra={"check": "unit dropout curve", "kept_units": int(keep.sum())}))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "target_session": args.target_session, "source_sessions": source_ids,
        "budgets": budgets, "rows": rows,
        "elapsed_seconds": time.perf_counter() - started}, indent=2, allow_nan=False) + "\n")
    return out


def _r2_from_preds(preds, record, q_bins):
    pred = preds[q_bins]
    truth = record.behavior[q_bins].astype(np.float64)
    ss_res = float(((truth - pred) ** 2).sum())
    centered = truth - truth.mean(0, keepdims=True)
    ss_tot = float((centered ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _row(name, r2, common, extra):
    return {"method": name, "r2": float(r2), **common, **extra}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dandi688sm", choices=["dandi688sm", "m1", "m2"])
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    parser.add_argument("--prior-correct", action="store_true")
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
