"""Track A Phase-1 evaluation: closed-form tables on a frozen AP backbone.

Protocol for one target session of one rotation:

1. Budgets are labeled trial counts B in {0, 2, 4, 8, 16, 33}.
2. Support = the first B usable trials (length >= 50 bins).  Their neural bins
   give the input z-score statistics when B > 0.  When B = 0 the full
   recording gives them.  This label-free use is recorded per run.
3. Query = every valid bin of the remaining trials.  Query labels never enter
   any fit, table, or selection rule.
4. Each method returns a table ``W`` ``[K, N]`` and an optional latent offset
   ``c``.  The frozen backbone decodes ``y = base(x_z @ W^T + c)`` in one
   contiguous pass.  R2 is physical-unit and variance-weighted, matching the
   repository convention.

Revision 2026-10-05 fixes applied here:

- Item 7: behavior features are built on the continuous recording first, then
  rows are selected.  Lags never cross a concatenation boundary.
- Item 8: the decode pass is one contiguous forward, so the recurrent state
  never resets inside a recording.
- Item 9: ``ui_grad`` trains on contiguous labeled trial windows with the
  base frozen; only the table receives gradients.
- Item 10: the checkpoint's recorded target must equal ``--target-session``.
- Items 5, 6: ``am1``, ``am3``, and ``oracle_frame`` use the whitened frame
  with explicit latent standardization, canonical null-space completion, FA
  restarts, and an explicit latent offset.

Methods:
- ``fa_identity``: FA posterior read-in, no frame.  Works at B = 0.
- ``am1``: whitened-frame A-M1 (main method).
- ``am3``: A-M1 with FA statistics from non-support bins only.
- ``am0``: supervised profiles on L plus a shared map g fit on source tables.
- ``am2``: A-M0 profiles with source-population shrinkage, then the same g.
- ``ui_grad``: gradient refinement of the FA table on L (same init as
  ``fa_identity``).  The POSSM-style baseline.
- ``mean_table``: mean of the source session tables.  Fixed-channel data only.
- ``oracle_frame``: whitened A-M1 with the frame fit on query labels.
  Diagnostic only.  It is reported separately and never used in comparisons.
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
from .ap_backbone import load_frozen_ap_backbone
from .data import DEFAULT_ROOT, sha256_file
from .multi_target import DEFAULT_CACHE, cached_recording

BUDGETS = (0, 2, 4, 8, 16, 33)
MIN_TRIAL_BINS = 50


def usable_trials(record, *, min_bins=MIN_TRIAL_BINS):
    return [(a, b) for a, b in record.trial_bounds if b - a >= min_bins]


def support_stats(record, budget, trials):
    """Input z-score statistics.  Support bins when B > 0, full recording
    otherwise.  Neural only; behavior stats stay with the source."""
    if budget > 0:
        bins = np.concatenate([np.arange(a, b) for a, b in trials[:budget]])
        x = record.neural[bins].astype(np.float64)
    else:
        x = record.neural.astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    return mean, std


def query_bins(record, trials, budget):
    rest = trials[budget:] if budget > 0 else trials
    good = record.eval_mask & np.isfinite(record.behavior).all(1)
    out = []
    for a, b in rest:
        out.append(np.arange(a, b)[good[a:b]])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def labeled_bins(record, trials, budget):
    bins = np.concatenate([np.arange(a, b) for a, b in trials[:budget]])
    good = record.eval_mask[bins] & np.isfinite(record.behavior[bins]).all(1)
    return bins[good]


def decode_r2_record(model, x_z, table, bins, record, label_stats, *, device,
                     latent_offset=None):
    """One contiguous causal pass over the whole recording (revision item 8).

    Chunking would reset the recurrent state, so the sequence goes through
    the backbone in a single forward.  ``latent_offset`` adds the frame
    offset of the whitened methods before the backbone.
    """
    xt = torch.from_numpy(x_z.astype(np.float32)).to(device).unsqueeze(0)   # [1, T, N]
    table_t = torch.from_numpy(table.astype(np.float32)).to(device)
    with torch.inference_mode():
        z = xt @ table_t.T
        if latent_offset is not None:
            z = z + torch.from_numpy(np.asarray(latent_offset, dtype=np.float32)).to(device)
        pred = model.base(z).squeeze(0).cpu().numpy()
    pred = pred[bins] * label_stats[1] + label_stats[0]
    truth = record.behavior[bins].astype(np.float64)
    ss_res = float(((truth - pred) ** 2).sum())
    centered = truth - truth.mean(0, keepdims=True)
    ss_tot = float((centered ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def ui_grad_table(model, x_z, y_norm, trial_windows, table_init, *, device,
                  steps=300, lr=3e-3, seed=0, valid_mask=None):
    """Gradient fit of the K x N table on contiguous labeled trials.

    Revision item 9: training uses whole trial segments, so the backbone sees
    continuous state, matching the evaluation decode.  The base parameters
    are frozen for the fit; only the table receives gradients.
    """
    torch.manual_seed(seed)
    saved = [p.requires_grad for p in model.base.parameters()]
    for p in model.base.parameters():
        p.requires_grad_(False)
    table = torch.nn.Parameter(torch.from_numpy(table_init.astype(np.float32)).clone().to(device))
    x = torch.from_numpy(x_z.astype(np.float32)).to(device)
    y = torch.from_numpy(y_norm.astype(np.float32)).to(device)
    mask = torch.from_numpy(valid_mask).to(device) if valid_mask is not None else None
    rng = np.random.default_rng(seed)
    windows = [(a, b) for a, b in trial_windows if b - a >= 2]
    optimizer = torch.optim.Adam([table], lr=lr)
    loss = float("nan")
    try:
        for step in range(steps):
            a, b = windows[int(rng.integers(len(windows)))]
            optimizer.zero_grad(set_to_none=True)
            decode = model.base((x[a:b] @ table.T).unsqueeze(0)).squeeze(0)
            target = y[a:b]
            good = torch.isfinite(target).all(-1)
            if mask is not None:
                good = good & mask[a:b]
            if not good.any():
                continue
            loss = (decode[good] - target[good]).square().mean()
            loss.backward()
            optimizer.step()
    finally:
        for p, state in zip(model.base.parameters(), saved):
            p.requires_grad_(state)
    return table.detach().cpu().numpy().astype(np.float64), float(loss)


def _source_latent_stats(model, records, tables):
    """Pooled mean and std of source latents over eval-valid bins.

    The whitened frame needs the source latent distribution in explicit form
    instead of the whitening penalty alone (revision 4.2.3).
    """
    rows = []
    for record, table in zip(records, tables):
        x = record.neural.astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        std[std < 1e-6] = 1.0
        x_z = (x - mean) / std
        good = record.eval_mask & np.isfinite(record.behavior).all(1)
        rows.append(x_z[good] @ table.T)
    z = np.concatenate(rows)
    return z.mean(0), np.maximum(z.std(0), 1e-6)


def _source_profiles(records, lags):
    """A-M0 basis profiles for each source session, full-session z-scores."""
    profiles = []
    for record in records:
        x = record.neural.astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        std[std < 1e-6] = 1.0
        x_z = (x - mean) / std
        good = record.eval_mask & np.isfinite(record.behavior).all(1)
        phi = assoc.behavior_features(record.behavior, lags=lags)
        profiles.append(assoc.supervised_profiles(phi[good], x_z[good], lam=1.0)["H"])
    return profiles


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        return {k2: _jsonable(v2) for k2, v2 in value.items()}
    return value


def run(args):
    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pool_sources = ([s for s in args.pool_sources.split(",") if s]
                    if getattr(args, "pool_sources", None) else None)
    if pool_sources is not None:
        # Shared-pool checkpoint: the source set must match the target's pool.
        if set(blob.get("source_sessions", [])) != set(pool_sources):
            raise ValueError(
                f"checkpoint sources {sorted(blob.get('source_sessions', []))} != pool "
                f"{sorted(pool_sources)} (revision item 10, pool form)")
    elif blob.get("target_session") != args.target_session:
        raise ValueError(
            f"checkpoint target {blob.get('target_session')!r} != --target-session "
            f"{args.target_session!r} (revision item 10)")
    model, blob = load_frozen_ap_backbone(args.checkpoint, device=device)
    k = model.k
    target_session = blob["target_session"]
    source_ids = blob["source_sessions"]
    label_stats = tuple(np.asarray(s, dtype=np.float64) for s in blob["source_statistics"][0][2:4])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    record = cached_recording(args.task, "held_in", target_session,
                              root=Path(args.data_root), cache_root=Path(args.cache_root))
    trials = usable_trials(record)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else list(BUDGETS)
    records_src = [cached_recording(args.task, "held_in", s, root=Path(args.data_root),
                                    cache_root=Path(args.cache_root)) for s in source_ids]
    source_tables = [t.detach().cpu().numpy().astype(np.float64) for t in model.tables]
    # Shared-map fit for A-M0/A-M2 and latent stats for the whitened frame.
    source_profiles = _source_profiles(records_src, args.lags)
    g_fit = assoc.fit_shared_profile_map(source_profiles, source_tables, lam=args.lam)
    prior = assoc.profile_prior(source_profiles)
    fixed_channel = len({t.shape[1] for t in source_tables}) == 1
    source_latent = _source_latent_stats(model, records_src, source_tables)
    # Revision item 7: behavior features come from the continuous recording.
    phi_full = assoc.behavior_features(record.behavior, lags=args.lags)
    finite_behavior = np.isfinite(record.behavior).all(1)
    started = time.perf_counter()
    rows = []
    shared = {"fa_seconds": None}
    for budget in budgets:
        mean, std = support_stats(record, budget, trials)
        x_z = ((record.neural.astype(np.float64) - mean) / std)
        l_bins = labeled_bins(record, trials, budget) if budget > 0 else np.empty(0, dtype=np.int64)
        q_bins = query_bins(record, trials, budget)
        if len(q_bins) == 0:
            raise ValueError(f"{target_session}: no query bins at budget {budget}")
        common = {"budget": budget, "labeled_bins": int(len(l_bins)),
                  "query_bins": int(len(q_bins)),
                  "stats_protocol": f"first-{budget}-trials" if budget > 0 else "full-recording-label-free"}
        # Label-free FA on all bins (unlabeled use of the recording).
        fa_started = time.perf_counter()
        fa = assoc.factor_analysis_em(x_z, k, iters=args.fa_iters, seed=args.seed)
        beta = assoc.posterior_readin(fa["H"], fa["psi"])
        shared["fa_seconds"] = time.perf_counter() - fa_started
        b_src = model.head.weight.detach().cpu().numpy().astype(np.float64)

        methods = {}
        methods["fa_identity"] = {"table": beta}
        if budget > 0:
            x_l = x_z[l_bins]
            y_l = (record.behavior[l_bins].astype(np.float64) - label_stats[0]) / label_stats[1]
            am1 = assoc.am1_whitened_table(x_l, x_z, y_l, b_src, source_latent, k=k,
                                           head_lam=args.head_lam, em_iters=args.fa_iters,
                                           seed=args.seed)
            methods["am1"] = {"table": am1["table"], "latent_offset": am1["latent_offset"],
                              "restart_spread": am1["restart_spread"]}
            # A-M3: FA statistics from non-support bins only.  Labels enter
            # only through the frame.  This isolates sensitivity to the
            # start-of-session distribution of the labeled support.
            support_mask = np.zeros(len(x_z), dtype=bool)
            for a, b in trials[:budget]:
                support_mask[a:b] = True
            am3_result = assoc.am1_whitened_table(
                x_l, x_z[~support_mask], y_l, b_src, source_latent, k=k,
                head_lam=args.head_lam, em_iters=args.fa_iters, seed=args.seed)
            methods["am3"] = {"table": am3_result["table"]}
            phi_l = phi_full[l_bins]
            h0 = assoc.supervised_profiles(phi_l, x_l, lam=args.lam)["H"]
            methods["am0"] = {"table": (h0 @ g_fit["g"]).T, "g_residual": g_fit["residual"]}
            h2 = assoc.shrinkage_profiles(phi_l, x_l, prior, lam=args.lam)["H"]
            methods["am2"] = {"table": (h2 @ g_fit["g"]).T}
            ui_table, ui_loss = ui_grad_table(
                model, x_z, (record.behavior.astype(np.float64) - label_stats[0]) / label_stats[1],
                trials[:budget], beta, device=device, steps=args.ui_steps,
                lr=args.ui_lr, seed=args.seed,
                valid_mask=record.eval_mask & finite_behavior)
            methods["ui_grad"] = {"table": ui_table, "final_loss": ui_loss}
        if fixed_channel:
            methods["mean_table"] = {"table": np.mean(source_tables, axis=0)}
        if args.oracle and budget > 0:
            oracle_bins = query_bins(record, trials, budget)
            x_o = x_z[oracle_bins]
            y_o = (record.behavior[oracle_bins].astype(np.float64) - label_stats[0]) / label_stats[1]
            oracle = assoc.am1_whitened_table(x_o, x_z, y_o, b_src, source_latent, k=k,
                                              head_lam=args.head_lam, em_iters=args.fa_iters,
                                              seed=args.seed)
            methods["oracle_frame"] = {"table": oracle["table"],
                                       "latent_offset": oracle["latent_offset"],
                                       "note": "frame fit on query labels; diagnostic only"}

        for name, payload in methods.items():
            r2 = decode_r2_record(model, x_z, payload["table"], q_bins, record, label_stats,
                                  device=device, latent_offset=payload.get("latent_offset"))
            row = _jsonable({"task": args.task, "target_session": target_session, "method": name,
                             "r2": r2, "fa_loglik": fa["loglik"], "fa_seconds": shared["fa_seconds"],
                             **common, **{k2: v2 for k2, v2 in payload.items() if k2 != "table"}})
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "target_sha256": sha256_file(record.path),
        "target_session": target_session, "source_sessions": source_ids,
        "budgets": budgets, "rows": rows,
        "frame_contract": "whitened am1/am3/oracle with latent offset; decode single contiguous pass",
        "code_hashes": {"track_a_experiment.py": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()},
        "elapsed_seconds": time.perf_counter() - started}, indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["m1", "m2"])
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--pool-sources", default=None,
                        help="comma list; validates the checkpoint source pool instead of the target id")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    parser.add_argument("--oracle", action="store_true")
    for name, default in [("lags", 2), ("fa-iters", 300), ("ui-steps", 300), ("seed", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in [("lam", 1.0), ("head-lam", 1.0), ("ui-lr", 3e-3)]:
        parser.add_argument("--" + name, type=float, default=default)
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
