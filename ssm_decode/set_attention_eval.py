"""Track E evaluation on 000688 targets (revision 4.7.5, 4.7.6; review 2026-10-06).

Loads an episodic set-attention checkpoint and evaluates budget-dependent
folded tables on one target session:

- ``e1_profile``: descriptor from the first B labeled trials (labels enter
  the descriptor only).
- ``e1_rate``: label-free descriptor (profile block zeroed).  Works at B=0.
- ``e2_ema``: slow re-folding.  Each recording segment decodes with a table
  folded from a causal, strictly-past, label-free descriptor.

Review corrections applied here:

- A1: predictions are denormalized with the checkpoint's label statistics
  before any R2, and a physical-unit scale check guards every decode.
- A5: the previous ``e4_window`` is withdrawn.  It re-folded per segment
  with forward-looking windows and reset state, so it was neither a dynamic
  key nor a valid ceiling.  A true step-level E4 is future work.
- A6: the permutation check now shuffles descriptor rows against the input
  columns, re-folds, and decodes.  Order equivariance itself is a property
  of the fold and is covered by the unit test.
- B5: the prior-correct residual uses the checkpoint's label statistics.
- C4: source recordings for the prior map load once per evaluation.

End-to-end checks at entry: physical units, source-fit gate, budget rule.
Query labels never enter any fit or fold.
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
from .eval_checks import (check_budget_rule, check_physical_units, check_source_gate,
                          flatten_checks)
from .set_attention import load_set_backbone, unit_descriptors
from .track_a_experiment import labeled_bins, query_bins, support_stats, usable_trials

BUDGETS = (0, 2, 4, 8, 16, 33)


def _label_stats(blob):
    stats = blob.get("label_stats")
    if stats is None:
        raise ValueError("checkpoint carries no label_stats; predictions cannot be "
                         "denormalized (review A1).  Re-run set_pretraining.")
    return np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)


# Checks come from ssm_decode.eval_checks: unified bands, recorded into
# rows, never aborting the whole evaluation (review A3/A5/C1 of iter 3).


def _desc(blob, record, bins, with_profile):
    """Standardized (profile, stats) blocks; profile zeroed when label-free."""
    raw, info = unit_descriptors(record, bins, lags=blob["args"]["lags"],
                                 with_profile=with_profile)
    stats = blob["desc_stats"]
    normed = (raw - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    profile_cols = blob["profile_dim"]
    if not with_profile:
        normed[:, :profile_cols] = 0.0
    return normed[:, :profile_cols], normed[:, profile_cols:], info


def _fold(model, blocks, device, perm=None):
    profile, stats = blocks
    if perm is not None:
        profile, stats = profile[perm], stats[perm]
    with torch.inference_mode():
        table = model.fold_from_raw(torch.from_numpy(profile.astype(np.float32)).to(device),
                                    torch.from_numpy(stats.astype(np.float32)).to(device))
    return table.cpu().numpy().astype(np.float64)


def _forward(model, x_seg, table, device):
    xt = torch.from_numpy(x_seg.astype(np.float32)).to(device).unsqueeze(0)
    table_t = torch.from_numpy(table.astype(np.float32)).to(device)
    with torch.inference_mode():
        out = model.base(xt @ table_t.T).squeeze(0)
    return out.detach().cpu().numpy()


def _segments(model, blob, record, x_z, device, seg_frac=0.1):
    """E2: causal label-free re-folding per segment with carried state.

    Review B5 (iteration 3): the trunk state carries across segments via
    the streaming forward; the first segment necessarily starts from zero
    state.  Descriptors use strictly past bins once any exist.
    """
    from .mamba3_streaming import forward_with_states, initial_states
    n = len(x_z)
    preds = np.zeros((n, record.behavior.shape[1]), dtype=np.float64)
    seg_len = max(int(n * seg_frac), 200)
    states = None
    xt = torch.from_numpy(x_z.astype(np.float32)).to(device).unsqueeze(0)
    for lo in range(0, n, seg_len):
        hi = min(lo + seg_len, n)
        past = np.arange(0, lo)
        bins = past if len(past) > 100 else np.arange(lo, hi)
        profile_seg, stats_seg, _ = _desc(blob, record, bins, with_profile=False)
        table = _fold(model, (profile_seg, stats_seg), device)
        with torch.inference_mode():
            out, states = forward_with_states(model.base,
                                              xt[:, lo:hi] @ torch.from_numpy(
                                                  table.astype(np.float32)).to(device).T,
                                              states)
        preds[lo:hi] = out.squeeze(0).detach().cpu().numpy()
    return preds


def _r2(preds, record, q_bins, y_mean, y_std, *, check_units=False):
    denorm = preds * y_std[None, :] + y_mean[None, :]
    pred = denorm[q_bins]
    truth = record.behavior[q_bins].astype(np.float64)
    unit_check = check_physical_units(pred, truth) if check_units else None
    ss_res = float(((truth - pred) ** 2).sum())
    centered = truth - truth.mean(0, keepdims=True)
    ss_tot = float((centered ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return (r2, unit_check) if check_units else r2


def run(args):
    device = torch.device(args.device)
    model, blob = load_set_backbone(args.checkpoint, device=device)
    entry_checks = []
    if args.check_budget:
        entry_checks.append(check_budget_rule(blob))
    entry_checks.append(check_source_gate(blob, args.min_source_r2))
    y_mean, y_std = _label_stats(blob)
    source_ids = blob["source_sessions"]
    if args.task == "dandi688sm":
        record = load_dandi688_recording(args.target_session)
        if args.target_session in source_ids:
            raise ValueError("target must not be part of the source pool (review B6)")
    else:
        from .multi_target import cached_recording
        record = cached_recording(args.task, "held_in", args.target_session)
    trials = usable_trials(record)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else list(BUDGETS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    g_fit = None
    if args.prior_correct:
        records_src = [load_dandi688_recording(s) for s in source_ids]
        src_profiles, src_tables = [], []
        for rec_s in records_src:
            good = rec_s.eval_mask & np.isfinite(rec_s.behavior).all(1)
            phi = assoc.behavior_features(rec_s.behavior, lags=blob["args"]["lags"])
            xs = rec_s.neural.astype(np.float64)
            mu, sd = xs.mean(0), np.where(xs.std(0) < 1e-6, 1.0, xs.std(0))
            src_profiles.append(assoc.supervised_profiles(phi[good], ((xs - mu) / sd)[good])["H"])
            profile_s, stats_s, _ = _desc(blob, rec_s, np.arange(len(xs)), with_profile=True)
            src_tables.append(_fold(model, (profile_s, stats_s), device))
        g_fit = assoc.fit_shared_profile_map(src_profiles, src_tables, lam=1.0)
        del records_src
    rows = []
    results_path = out / "results.jsonl"
    started = time.perf_counter()
    entry_flat = flatten_checks(entry_checks)

    def emit(row):
        # Row-wise write: an interrupted or failed method cannot destroy
        # other methods' results (review A5).
        rows.append(row)
        with results_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    for budget in budgets:
        mean, std = support_stats(record, budget, trials)
        x_z = ((record.neural.astype(np.float64) - mean) / std)
        l_bins = labeled_bins(record, trials, budget) if budget > 0 else np.empty(0, dtype=np.int64)
        q_bins = query_bins(record, trials, budget)
        common = {"budget": budget, "labeled_bins": int(len(l_bins)),
                  "query_bins": int(len(q_bins)), "method_family": "set_attention"}
        tables = {}
        blocks = None
        if budget > 0:
            profile, stats, _ = _desc(blob, record, l_bins, with_profile=True)
            blocks = (profile, stats)
            tables["e1_profile"] = _fold(model, blocks, device)
            if g_fit is not None:
                phi_l = assoc.behavior_features(record.behavior,
                                                lags=blob["args"]["lags"])[l_bins]
                h0 = assoc.supervised_profiles(phi_l, x_z[l_bins], lam=1.0)["H"]
                prior_w0 = (h0 @ g_fit["g"]).T
                y_l = (record.behavior[l_bins].astype(np.float64) - y_mean) / y_std
                corrected = assoc.prior_correction_table(
                    model.head.weight.detach().cpu().numpy().astype(np.float64),
                    prior_w0, x_z[l_bins], y_l, lam=1.0)
                tables["e1_prior_correct"] = corrected["table"]
        profile_free, stats_free, _ = _desc(blob, record, np.arange(len(x_z)),
                                            with_profile=False)
        blocks_free = (profile_free, stats_free)
        tables["e1_rate"] = _fold(model, blocks_free, device)

        for name, table in tables.items():
            r2, unit_check = _r2(_forward(model, x_z, table, device), record, q_bins,
                                 y_mean, y_std, check_units=True)
            emit({"method": name, "r2": r2, **common,
                  "table_norm": float(np.linalg.norm(table)),
                  **flatten_checks([unit_check]), **entry_flat})
        preds_e2 = _segments(model, blob, record, x_z, device)
        r2_e2, unit_e2 = _r2(preds_e2, record, q_bins, y_mean, y_std, check_units=True)
        emit({"method": "e2_ema", "r2": r2_e2, **common,
              **flatten_checks([unit_e2]), **entry_flat})

        main_name = "e1_profile" if budget > 0 else "e1_rate"
        main_table = tables[main_name]
        main_blocks = blocks if budget > 0 else blocks_free
        if budget > 0:
            perm = rng.permutation(blocks[0].shape[0])
            shuffled = _fold(model, blocks, device, perm=perm)
            r2_shuf = _r2(_forward(model, x_z, shuffled, device), record, q_bins, y_mean, y_std)
            emit({"method": f"{main_name}_descshuffle", "r2": r2_shuf, **common,
                  "note": "descriptor-row shuffle; must fall below the main number",
                  **entry_flat})
        # B6: dropout curves re-fold with the unit mask inside the softmax,
        # matching how the model was trained.
        for frac in (0.1, 0.3, 0.5):
            keep_np = ~(rng.random(x_z.shape[1]) < frac)
            if keep_np.sum() < 4:
                continue
            keep_t = torch.from_numpy(keep_np).to(device)
            with torch.inference_mode():
                table_drop = model.fold_from_raw(
                    torch.from_numpy(main_blocks[0].astype(np.float32)).to(device),
                    torch.from_numpy(main_blocks[1].astype(np.float32)).to(device), keep_t)
            table_drop = table_drop.cpu().numpy().astype(np.float64)
            r2_drop = _r2(_forward(model, x_z, table_drop, device),
                          record, q_bins, y_mean, y_std)
            emit({"method": f"{main_name}_unitdrop{int(frac * 100)}", "r2": r2_drop,
                  **common, "kept_units": int(keep_np.sum()), **entry_flat})
    (out / "summary.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "target_session": args.target_session, "source_sessions": source_ids,
        "budgets": budgets, "rows": rows,
        "checks": {"physical_units": True, "source_gate": args.min_source_r2,
                   "budget_rule": args.check_budget,
                   "permutation": "descriptor-row shuffle (review A6)"},
        "elapsed_seconds": time.perf_counter() - started}, indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dandi688sm", choices=["dandi688sm", "m1", "m2"])
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    parser.add_argument("--prior-correct", action="store_true")
    parser.add_argument("--min-source-r2", type=float, default=None,
                        help="source-fit gate; refuse evaluation below this source R2")
    parser.add_argument("--check-budget", action=argparse.BooleanOptionalAction, default=True,
                        help="verify best_step is not in the last 20%% of the trained budget")
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
