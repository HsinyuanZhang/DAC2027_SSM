"""C8 source pretraining for the AP backbone (Track A frozen backbone).

Trains per-session K x N read-in tables, a shared lift ``W_lift`` (the base
``in_proj``), the Mamba-3 trunk, and the linear behavior head ``B_src`` on the
source sessions of one rotation.  The target session is never loaded.

Contract items from the roadmap:
1. Per-source-session input z-scores (C1.1).
2. Whitening penalty on the K-dim latent (C8.3).
3. Linear head loss on the latent (C8.2).
4. Channel gain augmentation and channel dropout (C8.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import debug_experiment as d
from .ap_backbone import APBackbone
from .cross_session_iteration import _code_hashes, _groups, _runtime_identity
from .data import DEFAULT_ROOT, sha256_file
from .multi_target import DEFAULT_CACHE, cached_recording, rotation_plans
from .session_frontend import drift_augment
from .session_pretraining import continuous_source_windows, sample_balanced_batch, source_statistics

MODULES = ("ap_pretraining.py", "ap_backbone.py", "association.py",
           "session_pretraining.py", "session_frontend.py", "multi_target.py")


def code_hashes() -> dict:
    hashes = _code_hashes()
    here = Path(__file__).parent
    for name in MODULES:
        hashes[name] = hashlib.sha256((here / name).read_bytes()).hexdigest()
    return hashes


def latent_batch(model: APBackbone, x: torch.Tensor, sessions: torch.Tensor) -> torch.Tensor:
    """Latents with each sample routed through its own session table.

    Revision item 2: sessions may have different channel counts, so tables
    cannot be stacked.  Samples are grouped by session and each group uses
    its own table.  Equal channel counts do not imply equal tables either.
    """
    z = x.new_zeros(x.shape[0], x.shape[1], model.k)
    for session in sessions.unique().tolist():
        rows = sessions == session
        z[rows] = x[rows] @ model.tables[int(session)].T
    return z


def validate_rotation(model, windows, selections, context, batch, label_stats, device):
    """Macro-average source validation R2 for the trunk and the head paths."""
    model.eval()
    trunk_scores, head_scores, latents = [], [], []
    with torch.inference_mode():
        for session, selection in enumerate(selections):
            predictions, truths, head_predictions, z_rows = [], [], [], []
            for lo in range(0, len(selection), batch):
                x, y, mask = d._right_aligned([windows[session]], selection[lo:lo + batch], context)
                z = model.latent(x, session)
                decode, head = model.base(z), model.head(z)
                good = mask[:, -1] & torch.isfinite(y[:, -1]).all(-1)
                if not torch.isfinite(decode).all() or not torch.isfinite(head).all():
                    raise FloatingPointError("source validation prediction is not finite")
                predictions.append(decode[:, -1][good])
                head_predictions.append(head[:, -1][good])
                truths.append(y[:, -1][good])
                z_rows.append(z.reshape(-1, z.shape[-1])[mask.reshape(-1)])
            if not predictions[0].numel() and not any(p.numel() for p in predictions):
                raise FloatingPointError(f"session {session} has no valid validation bins")
            truth = torch.cat(truths)
            scale = torch.as_tensor(label_stats[3], device=truth.device)
            offset = torch.as_tensor(label_stats[2], device=truth.device)
            trunk_scores.append(d._r2(truth * scale + offset, torch.cat(predictions) * scale + offset))
            head_scores.append(d._r2(truth * scale + offset, torch.cat(head_predictions) * scale + offset))
            latents.append(torch.cat(z_rows))
    z = torch.cat(latents)
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / max(len(zc) - 1, 1)
    return {"trunk_r2": float(np.mean(trunk_scores)), "head_r2": float(np.mean(head_scores)),
            "per_session_trunk_r2": [float(s) for s in trunk_scores],
            "latent_diag_cov_min": float(cov.diagonal().min()),
            "latent_diag_cov_max": float(cov.diagonal().max()),
            "latent_offdiag_absmax": float((cov - torch.diag(cov.diagonal())).abs().max())}


def _load_rotation(args):
    """Return (plan, records) for one rotation of the requested task."""
    if args.task == "dandi688sm":
        from .dandi688_data import load_dandi688_recording
        plan = {"task": args.task, "target_session": args.target_session,
                "source_sessions": [s for s in args.dandi688_sources.split(",") if s],
                "protocol": "spint_2015_dev_target"}
        records = [load_dandi688_recording(s) for s in plan["source_sessions"]]
        return plan, records
    plans = rotation_plans(args.task, root=Path(args.data_root))
    plan = next(p for p in plans if p["target_session"] == args.target_session)
    records = [cached_recording(args.task, "held_in", s, root=Path(args.data_root),
                                cache_root=Path(args.cache_root)) for s in plan["source_sessions"]]
    return plan, records


def run(args):
    if args.steps < 1 or args.val_interval < 1 or args.context < 2 or args.k < 1:
        raise ValueError("invalid training budget")
    torch.set_num_threads(1)
    d._seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = code_hashes()
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    here = Path(__file__).parent
    for name in MODULES:
        (snapshot / name).write_bytes((here / name).read_bytes())
    plan, records = _load_rotation(args)
    if getattr(args, "source_sessions", None):
        # Forward-eval pools override the rotation complement (revision 4.1).
        override = [s for s in args.source_sessions.split(",") if s]
        if set(override) == set(plan["source_sessions"]):
            plan = {**plan, "source_sessions": override,
                    "source_pool_override": "explicit --source-sessions (reordered)"}
        else:
            plan = {**plan, "source_sessions": override,
                    "source_pool_override": "explicit --source-sessions"}
        records = [cached_recording(args.task, "held_in", s, root=Path(args.data_root),
                                    cache_root=Path(args.cache_root)) for s in override]
    source_ids = plan["source_sessions"]
    source_hashes = {r.session: sha256_file(r.path) for r in records}
    splits = [d._trial_split(r) for r in records]
    if any(not train or not valid for train, valid in splits):
        raise ValueError("every source session requires train and validation trials")
    statistics = source_statistics(records, splits)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    train = continuous_source_windows(records, [s[0] for s in splits], statistics, device)
    valid = continuous_source_windows(records, [s[1] for s in splits], statistics, device)
    eligible = d._eligible(train)
    selections = [d._validation_selection([w], args.context, args.max_val_endpoints)[0] for w in valid]
    channel_counts = [r.neural.shape[1] for r in records]
    outputs = records[0].behavior.shape[1]
    model = APBackbone(channel_counts, outputs, k=args.k, width=args.width, layers=args.layers,
                       state_size=args.state_size, dropout=args.dropout).to(device)
    if args.table_init == "fa":
        fa_init = model.init_from_fa([w[0] for w in train], em_iters=args.fa_iters)
    elif args.table_init == "supervised":
        label_windows = [w[1] for w in train]
        fa_init = model.init_from_supervised([w[0] for w in train], label_windows)
    else:
        raise ValueError(f"unknown table init {args.table_init!r}")
    optimizer = torch.optim.AdamW(_groups(list(model.parameters()), args.lr, args.weight_decay))
    runtime = _runtime_identity()
    best_score, best_step = -float("inf"), 0
    log = []
    started = time.perf_counter()
    single_session = len(set(channel_counts)) > 1
    for step in range(1, args.steps + 1):
        model.train()
        if single_session:
            session = int(np.random.randint(len(train)))
            x, y, mask, sessions = sample_balanced_batch([train[session]], [eligible[session]],
                                                         args.context, args.batch_size)
            sessions = torch.full_like(sessions, session)
        else:
            x, y, mask, sessions = sample_balanced_batch(train, eligible, args.context, args.batch_size)
        x = drift_augment(x, args.gain_sigma, args.offset_sigma, args.channel_dropout)
        z = model.latent(x, int(sessions[0])) if single_session else latent_batch(model, x, sessions)
        decode, head = model.base(z)[:, args.context // 2:], model.head(z)[:, args.context // 2:]
        truth = y[:, args.context // 2:]
        good = mask[:, args.context // 2:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(decode).all():
            raise FloatingPointError("source training batch is invalid")
        optimizer.zero_grad(set_to_none=True)
        loss = (decode[good] - truth[good]).square().mean()
        loss = loss + args.lam_head * (head[good] - truth[good]).square().mean()
        loss = loss + args.lam_white * model.whitening_penalty(z[:, args.context // 2:], mask[:, args.context // 2:])
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True))
        optimizer.step()
        if step == 1 or step % args.val_interval == 0 or step == args.steps:
            metrics = validate_rotation(model, valid, selections, args.context,
                                        args.eval_batch_size, statistics[0], device)
            if not np.isfinite(metrics["trunk_r2"]):
                raise FloatingPointError("source validation score is not finite")
            event = {"step": step, "loss": float(loss.detach()), "gradient_norm": gradient,
                     "elapsed_seconds": time.perf_counter() - started, **metrics}
            log.append(event)
            with (out / "train_log.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            if metrics["trunk_r2"] > best_score:
                best_score, best_step = metrics["trunk_r2"], step
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "channel_counts": channel_counts, "outputs": outputs,
                            "source_sessions": source_ids, "target_session": args.target_session,
                            "best_step": step, "best_source_val_trunk_r2": best_score,
                            "best_source_val_head_r2": metrics["head_r2"],
                            "fa_init": fa_init, "source_statistics": statistics,
                            "code_hashes": hashes, "source_data_sha256": source_hashes,
                            "runtime_identity": runtime}, out / "best.pt")
    if best_step == 0:
        raise FloatingPointError("no validation point was reached")
    selected = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    if hashes != code_hashes() or source_hashes != {r.session: sha256_file(r.path) for r in records}:
        raise RuntimeError("source implementation or data changed during training")
    final = validate_rotation(model, valid, selections, args.context, args.eval_batch_size,
                              statistics[0], device)
    receipt = {"status": "completed", "args": vars(args), "rotation": plan,
               "source_sessions": source_ids, "target_loaded": False,
               "source_train_bounds": [s[0] for s in splits],
               "source_val_bounds": [s[1] for s in splits],
               "source_statistics_scope": "per-session training neural; shared eval-valid training labels",
               "loss": "trunk MSE + lam_head * head MSE + lam_white * whitening",
               "best_step": best_step, "best_source_val_trunk_r2": best_score,
               "final_metrics": final, "fa_init": fa_init,
               "parameters": {"base": sum(p.numel() for p in model.base.parameters()),
                              "tables": sum(t.numel() for t in model.tables),
                              "head": sum(p.numel() for p in model.head.parameters())},
               "fit_seconds": time.perf_counter() - started, "code_hashes": hashes,
               "source_data_sha256": source_hashes, "runtime_identity": runtime,
               "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
                                         if device.type == "cuda" else None,
               "train_log_sha256": hashlib.sha256((out / "train_log.jsonl").read_bytes()).hexdigest()}
    torch.save({**selected, "final_metrics": final, "ap_pretraining_receipt": receipt}, out / "best.pt")
    receipt["checkpoint_sha256"] = hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["m1", "m2", "dandi688sm"])
    parser.add_argument("--dandi688-sources", default="sub-M_ses-CO-20150511,sub-M_ses-CO-20150512,sub-M_ses-CO-20150610,sub-M_ses-CO-20150611,sub-M_ses-CO-20150612,sub-M_ses-CO-20150615")
    parser.add_argument("--source-sessions", default=None,
                        help="comma-separated held-in ids; overrides the rotation source set")
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--table-init", default="fa", choices=["fa", "supervised"])
    for name, default in [("k", 16), ("width", 128), ("layers", 2), ("state-size", 32),
                          ("context", 128), ("batch-size", 32), ("eval-batch-size", 256),
                          ("steps", 2000), ("val-interval", 100), ("max-val-endpoints", 1024),
                          ("fa-iters", 200), ("seed", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in [("lr", 1e-3), ("weight-decay", 1e-3), ("dropout", 0.2),
                          ("gain-sigma", 0.3), ("offset-sigma", 0.1), ("channel-dropout", 0.05),
                          ("lam-white", 0.1), ("lam-head", 0.5)]:
        parser.add_argument("--" + name, type=float, default=default)
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
