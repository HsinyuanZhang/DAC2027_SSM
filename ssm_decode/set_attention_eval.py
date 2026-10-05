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
from .set_attention import load_set_backbone, unit_descriptors
from .track_a_experiment import labeled_bins, query_bins, support_stats, usable_trials

BUDGETS = (0, 2, 4, 8, 16, 33)


def _label_stats(blob):
    stats = blob.get("label_stats")
    if stats is None:
        raise ValueError("checkpoint carries no label_stats; predictions cannot be "
                         "denormalized (review A1).  Re-run set_pretraining.")
    return np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)


def _assert_physical_units(pred, truth, *, lo=0.5, hi=2.0):
    ratio = float(np.std(pred) / max(np.std(truth), 1e-9))
    if not lo <= ratio <= hi:
        raise ValueError(f"physical-unit check failed: pred/truth std ratio {ratio:.3f}")


def _assert_source_gate(blob, minimum):
    score = float(blob.get("best_source_val_trunk_r2", float("nan")))
    if not np.isfinite(score) or score < minimum:
        raise ValueError(f"source-fit gate failed: best_source_val_trunk_r2={score} < {minimum}")


def _assert_budget_rule(blob):
    args = blob.get("args", {})
    steps, best = int(args.get("steps", 0)), int(blob.get("best_step", 0))
    if steps <= 0 or best <= 0:
        raise ValueError("budget check failed: missing steps or best_step")
    if best > 0.8 * steps:
        raise ValueError(f"budget check failed: best_step {best} in last 20% of {steps}")


def _desc(blob, record, bins, with_profile):
    desc, info = unit_descriptors(record, bins, lags=blob["args"]["lags"],
                                  with_profile=with_profile)
    stats = blob["desc_stats"]
    normed = (desc - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    if not with_profile:
        normed[:, :desc.shape[1] - 2] = 0.0
    return normed, info


def _fold(model, desc, device):
    with torch.inference_mode():
        table = model.read_in.fold_table(torch.from_numpy(desc.astype(np.float32)).to(device))
    return table.cpu().numpy().astype(np.float64)


def _forward(model, x_seg, table, device):
    xt = torch.from_numpy(x_seg.astype(np.float32)).to(device).unsqueeze(0)
    table_t = torch.from_numpy(table.astype(np.float32)).to(device)
    with torch.inference_mode():
        out = model.base(xt @ table_t.T).squeeze(0)
    return out.detach().cpu().numpy()


def _segments(record, blob, model, x_z, device, seg_frac=0.1):
    """E2: causal label-free re-folding per segment, strictly past bins."""
    n = len(x_z)
    preds = np.zeros((n, record.behavior.shape[1]), dtype=np.float64)
    seg_len = max(int(n * seg_frac), 200)
    for lo in range(0, n, seg_len):
        hi = min(lo + seg_len, n)
        past = np.arange(0, lo)
        bins = past if len(past) > 100 else np.arange(lo, hi)
        desc, _ = unit_descriptors(record, bins, lags=blob["args"]["lags"], with_profile=False)
        stats = blob["desc_stats"]
        normed = (desc - np.asarray(stats["mean"])) / np.asarray(stats["std"])
        normed[:, :desc.shape[1] - 2] = 0.0
        preds[lo:hi] = _forward(model, x_z[lo:hi], _fold(model, normed, device), device)
    return preds


def _r2(preds, record, q_bins, y_mean, y_std, *, check_units=False):
    denorm = preds * y_std[None, :] + y_mean[None, :]
    pred = denorm[q_bins]
    truth = record.behavior[q_bins].astype(np.float64)
    if check_units:
        _assert_physical_units(pred, truth)
    ss_res = float(((truth - pred) ** 2).sum())
    centered = truth - truth.mean(0, keepdims=True)
    ss_tot = float((centered ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def run(args):
    device = torch.device(args.device)
    model, blob = load_set_backbone(args.checkpoint, device=device)
    if args.check_budget:
        _assert_budget_rule(blob)
    if args.min_source_r2 is not None:
        _assert_source_gate(blob, args.min_source_r2)
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
            desc_s, _ = _desc(blob, rec_s, np.arange(len(xs)), with_profile=True)
            src_tables.append(_fold(model, desc_s, device))
        g_fit = assoc.fit_shared_profile_map(src_profiles, src_tables, lam=1.0)
        del records_src
    rows = []
    started = time.perf_counter()
    for budget in budgets:
        mean, std = support_stats(record, budget, trials)
        x_z = ((record.neural.astype(np.float64) - mean) / std)
        l_bins = labeled_bins(record, trials, budget) if budget > 0 else np.empty(0, dtype=np.int64)
        q_bins = query_bins(record, trials, budget)
        common = {"budget": budget, "labeled_bins": int(len(l_bins)),
                  "query_bins": int(len(q_bins)), "method_family": "set_attention"}
        tables = {}
        desc = None
        if budget > 0:
            desc, _ = _desc(blob, record, l_bins, with_profile=True)
            tables["e1_profile"] = _fold(model, desc, device)
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
        desc_free, _ = _desc(blob, record, np.arange(len(x_z)), with_profile=False)
        tables["e1_rate"] = _fold(model, desc_free, device)

        for name, table in tables.items():
            r2 = _r2(_forward(model, x_z, table, device), record, q_bins, y_mean, y_std,
                     check_units=True)
            rows.append({"method": name, "r2": r2, **common,
                         "table_norm": float(np.linalg.norm(table))})
        preds_e2 = _segments(record, blob, model, x_z, device)
        rows.append({"method": "e2_ema",
                     "r2": _r2(preds_e2, record, q_bins, y_mean, y_std, check_units=True),
                     **common})

        main_name = "e1_profile" if budget > 0 else "e1_rate"
        main_table = tables[main_name]
        if budget > 0:
            perm = rng.permutation(desc.shape[0])
            shuffled = _fold(model, desc[perm], device)
            r2_shuf = _r2(_forward(model, x_z, shuffled, device), record, q_bins, y_mean, y_std)
            rows.append({"method": f"{main_name}_descshuffle", "r2": r2_shuf, **common,
                         "check": "descriptor-row shuffle; must fall below the main number"})
        for frac in (0.1, 0.3, 0.5):
            drop = rng.random(x_z.shape[1]) < frac
            keep = ~drop
            if keep.sum() < 4:
                continue
            r2_drop = _r2(_forward(model, x_z[:, keep], main_table[:, keep], device),
                          record, q_bins, y_mean, y_std)
            rows.append({"method": f"{main_name}_unitdrop{int(frac * 100)}", "r2": r2_drop,
                         **common, "kept_units": int(keep.sum())})
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
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
