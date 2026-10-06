"""Episodic pretraining for the E1 set backbone on 000688 (C8.4; review 4.1).

Training stages:

0. Optional AP warm start: trunk and head load from a converged AP backbone
   (``--init-from-ap``), so the set read-in starts against a working trunk.
1. Distillation (``--distill-steps``): the read-in and activity encoder
   regress folded tables onto the AP backbone's learned per-session source
   tables.  Trunk and head stay frozen.
2. Episodic training: each step computes descriptors from a random trial
   subset, folds the table, and decodes other trials of the same session.
   Unit subsampling runs inside the softmax (renormalized over kept units),
   channel gain jitter, descriptor noise, and profile-block dropout run
   every step.
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
from .cross_session_iteration import _code_hashes, _groups, _runtime_identity
from .dandi688_data import load_dandi688_recording
from .session_frontend import drift_augment
from .set_attention import (SetAttentionBackbone, normalize_descriptors,
                            raw_unit_stats, unit_descriptors)

MODULES = ("set_pretraining.py", "set_attention.py", "association.py",
           "session_pretraining.py", "session_frontend.py", "dandi688_data.py")


def code_hashes() -> dict:
    hashes = _code_hashes()
    here = Path(__file__).parent
    for name in MODULES:
        hashes[name] = hashlib.sha256((here / name).read_bytes()).hexdigest()
    return hashes


def run(args):
    if args.steps < 1 or args.val_interval < 1 or args.k < 1:
        raise ValueError("invalid training budget")
    if args.distill_steps > 0 and not args.init_from_ap:
        raise ValueError("--distill-steps > 0 requires --init-from-ap (review B2)")
    freeze_trunk = bool(args.freeze_trunk) if args.freeze_trunk >= 0 else bool(args.init_from_ap)
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
    source_ids = [s for s in args.dandi688_sources.split(",") if s]
    records = [load_dandi688_recording(s) for s in source_ids]
    source_hashes = {r.session: hashlib.sha256(r.path.read_bytes()).hexdigest() for r in records}
    splits = [d._trial_split(r) for r in records]
    if any(not tr or not va for tr, va in splits):
        raise ValueError("every source session requires train and validation trials")
    labels = np.concatenate([r.behavior[a:b][r.eval_mask[a:b] & np.isfinite(r.behavior[a:b]).all(1)]
                             for r, (tr, _) in zip(records, splits) for a, b in tr])
    y_mean, y_std = labels.mean(0), labels.std(0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    outputs = records[0].behavior.shape[1]
    profile_cols = outputs * (args.lags + 1)

    # Raw descriptor blocks and pooled standardization.
    raw_full = [unit_descriptors(r, np.arange(len(r.neural)), lags=args.lags,
                                 with_profile=True)[0] for r in records]
    normed_ref, desc_stats = normalize_descriptors(raw_full)
    desc_mean, desc_std = desc_stats["mean"], desc_stats["std"]
    stats_dim = raw_unit_stats(records[0], np.arange(2)).shape[1]
    profile_dim = raw_full[0].shape[1] - stats_dim
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    def _input_stats(index):
        if ap_train_stats is not None:
            entry = ap_train_stats[index]
            return np.asarray(entry[0], dtype=np.float64), np.asarray(entry[1], dtype=np.float64)
        x = records[index].neural.astype(np.float64)
        return x.mean(0), np.where(x.std(0) < 1e-6, 1.0, x.std(0))

    train_windows = []
    for index, record in enumerate(records):
        x_mean, x_std = _input_stats(index)
        xz = ((record.neural - x_mean) / x_std).astype(np.float32)
        y = np.where(record.eval_mask[:, None], (record.behavior - y_mean) / y_std, np.nan)
        train_windows.append(tuple(torch.from_numpy(v).to(device) for v in (xz, y, record.eval_mask)))
    val_windows = []
    for index, (record, (_, va)) in enumerate(zip(records, splits)):
        x_mean, x_std = _input_stats(index)
        xz = ((record.neural - x_mean) / x_std).astype(np.float32)
        y = np.full_like(record.behavior, np.nan)
        mask = np.zeros(len(xz), dtype=bool)
        for a, b in va:
            y[a:b] = (record.behavior[a:b] - y_mean) / y_std
            mask[a:b] = record.eval_mask[a:b]
        val_windows.append(tuple(torch.from_numpy(v).to(device) for v in (xz, y, mask)))
    rng = np.random.default_rng(args.seed)
    model = SetAttentionBackbone(profile_dim, stats_dim, outputs, k=args.k,
                                 width=args.width, layers=args.layers,
                                 state_size=args.state_size, dropout=args.dropout,
                                 tau=args.tau, act_hidden=args.act_hidden).to(device)
    warm = None
    if args.init_from_ap:
        warm = model.load_ap_warm_start(args.init_from_ap)
        print(f"[set] AP warm start: {len(warm['warm_started'])} tensors, "
              f"AP source val {warm['ap_best_source_val']:.4f}", flush=True)
    ap_tables = None
    ap_train_stats = None
    if args.distill_steps > 0:
        blob = torch.load(args.init_from_ap, map_location="cpu", weights_only=False)
        if blob["args"].get("k") != args.k:
            raise ValueError("AP backbone K must match the set backbone K for distillation")
        if list(blob.get("source_sessions", [])) != source_ids:
            raise ValueError("AP source session order differs from --dandi688-sources "
                             "(review A4); table pairing would be silently wrong")
        state = blob["state_dict"]
        ap_tables = [state[f"tables.{i}"].detach().cpu().numpy().astype(np.float64)
                     for i in range(len(source_ids))]
        ap_train_stats = blob.get("source_statistics")

    def raw_blocks(index, bins, with_profile=True):
        raw, _ = unit_descriptors(records[index], bins, lags=args.lags,
                                  with_profile=with_profile)
        normed = (raw - desc_mean) / desc_std
        if not with_profile:
            normed[:, :profile_cols] = 0.0
        return normed[:, :profile_cols], normed[:, profile_cols:]

    def fold(index, bins, keep_units, profile_dropout, noise):
        profile, stats = raw_blocks(index, bins)
        if profile_dropout:
            profile = profile * 0.0
        if noise:
            profile = profile + noise * rng.standard_normal(profile.shape)
            stats = stats + noise * rng.standard_normal(stats.shape)
        profile_t = torch.from_numpy(profile.astype(np.float32)).to(device)
        stats_t = torch.from_numpy(stats.astype(np.float32)).to(device)
        keep_t = None
        if keep_units is not None:
            keep_t = torch.from_numpy(keep_units).to(device)
        return model.fold_from_raw(profile_t, stats_t, keep_t)

    def validate():
        model.eval()
        scores = []
        with torch.inference_mode():
            for index, window in enumerate(val_windows):
                desc_bins = val_desc_bins[index]
                table = fold(index, desc_bins, None, profile_dropout=False, noise=0.0)
                selection = d._validation_selection([window], args.context,
                                                    args.max_val_endpoints)[0]
                xb, yb, mb = d._right_aligned([window], selection, args.context)
                decode = model.decode_with_table(xb, table)
                good = mb[:, -1] & torch.isfinite(yb[:, -1]).all(-1)
                scale = torch.as_tensor(y_std, device=device)
                offset = torch.as_tensor(y_mean, device=device)
                scores.append(d._r2(yb[:, -1][good] * scale + offset,
                                    decode[:, -1][good] * scale + offset))
        model.train()
        return float(np.mean(scores))

    # Validation descriptors from TRAIN trials only (review B2).
    val_desc_bins = []
    for (tr, _) in splits:
        picks = sorted(rng.choice(len(tr), size=min(args.episode_trials, len(tr)), replace=False))
        val_desc_bins.append(np.concatenate([np.arange(tr[i][0], tr[i][1]) for i in picks]))

    # Stage 1: distill the read-in toward the AP source tables.
    distill_info = None
    if ap_tables is not None:
        frozen = [p.requires_grad for p in model.parameters()]
        for p in model.base.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr)
        started_d = time.perf_counter()
        for step in range(1, args.distill_steps + 1):
            index = int(rng.integers(len(records)))
            target_table = torch.from_numpy(
                ap_tables[index].astype(np.float32)).to(device)
            # Distill on random episode subsets with the same augmentation
            # the deployment descriptor will carry (noise, dropout), against
            # the full-session AP table.  This trains a noise-robust
            # descriptor-to-table map against a frozen good trunk.
            trials = splits[index][0]
            picks = sorted(rng.choice(len(trials),
                                      size=min(args.episode_trials, len(trials)),
                                      replace=False))
            desc_bins = np.concatenate([np.arange(trials[i][0], trials[i][1])
                                        for i in picks])
            # Review A4 (iteration 3): no unit dropout in distillation.  The
            # full AP table has no well-defined dropped-unit form, and the
            # renormalised fold regressing onto it biases values low by about
            # the keep fraction.
            table = fold(index, desc_bins, None,
                         profile_dropout=rng.random() < args.profile_dropout,
                         noise=args.desc_noise)
            loss = (table - target_table).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        distill_info = {"steps": args.distill_steps,
                        "final_mse": float(loss.detach()),
                        "seconds": time.perf_counter() - started_d}
        print(f"[set] distillation done: mse {distill_info['final_mse']:.5f}", flush=True)
        for p, state in zip(model.parameters(), frozen):
            p.requires_grad_(state)

    episodic_frozen = None
    if freeze_trunk:
        episodic_frozen = [p.requires_grad for p in model.parameters()]
        for p in model.base.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(_groups([p for p in model.parameters() if p.requires_grad],
                                          args.lr, args.weight_decay))
    runtime = _runtime_identity()
    best_score, best_step = -float("inf"), 0
    log = []
    started = time.perf_counter()

    def maybe_validate(step, loss_value, gradient):
        nonlocal best_score, best_step
        score = validate()
        if not np.isfinite(score):
            raise FloatingPointError("source validation score is not finite")
        event = {"step": step, "loss": loss_value, "gradient_norm": gradient,
                 "source_val_trunk_r2": score,
                 "elapsed_seconds": time.perf_counter() - started}
        log.append(event)
        with (out / "train_log.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if score > best_score:
            best_score, best_step = score, step
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "profile_dim": profile_dim, "stats_dim": stats_dim,
                        "outputs": outputs, "source_sessions": source_ids,
                        "best_step": step, "best_source_val_trunk_r2": best_score,
                        "desc_stats": {"mean": desc_mean.tolist(),
                                       "std": desc_std.tolist()},
                        "label_stats": {"mean": y_mean.tolist(), "std": y_std.tolist()},
                        "warm_start": warm, "distill": distill_info,
                        "code_hashes": hashes, "source_data_sha256": source_hashes,
                        "runtime_identity": runtime}, out / "best.pt")

    if args.distill_steps > 0:
        maybe_validate(0, float("nan"), 0.0)

    for step in range(1, args.steps + 1):
        model.train()
        index = int(rng.integers(len(records)))
        window = train_windows[index]
        x_all, _, _ = window
        trials = splits[index][0]
        desc_pick = sorted(rng.choice(len(trials), size=min(args.episode_trials, len(trials)),
                                      replace=False))
        keep = np.ones(x_all.shape[1], dtype=bool)
        drop = rng.random(len(keep)) < args.unit_dropout
        if drop.all():
            drop[int(rng.integers(len(keep)))] = False
        keep[drop] = False
        keep_t = torch.from_numpy(keep).to(device).float()
        desc_bins = np.concatenate([np.arange(trials[i][0], trials[i][1]) for i in desc_pick])
        table = fold(index, desc_bins, keep,
                     profile_dropout=rng.random() < args.profile_dropout,
                     noise=args.desc_noise)
        other = [i for i in range(len(trials)) if i not in set(desc_pick)]
        # Review A1 (iteration 3): sample endpoints across the whole trial,
        # not only the last bin.  Short trials fall back to their end.
        picks = []
        span = args.context - 1
        for _ in range(args.batch_size):
            a, b = trials[other[int(rng.integers(len(other)))]]
            lo, hi = a + span, b - 1
            end = int(rng.integers(lo, hi + 1)) if hi > lo else hi
            picks.append((0, int(min(end, x_all.shape[0] - 1))))
        xb, yb, mb = d._right_aligned([window], picks, args.context)
        xb = xb * keep_t
        xb = drift_augment(xb, args.gain_sigma, args.offset_sigma, 0.0)
        z = xb @ table.T
        decode = model.base(z)[:, args.context // 2:]
        head = model.head(z)[:, args.context // 2:]
        truth = yb[:, args.context // 2:]
        good = mb[:, args.context // 2:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(decode).all():
            continue
        optimizer.zero_grad(set_to_none=True)
        loss = (decode[good] - truth[good]).square().mean()
        loss = loss + args.lam_head * (head[good] - truth[good]).square().mean()
        loss = loss + args.lam_white * model.whitening_penalty(z[:, args.context // 2:],
                                                               mb[:, args.context // 2:])
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1,
                                                        error_if_nonfinite=True))
        optimizer.step()
        if step == 1 or step % args.val_interval == 0 or step == args.steps:
            maybe_validate(step, float(loss.detach()), gradient)
    if best_step < 0:
        raise FloatingPointError("no validation point was reached")
    # best_step == 0 means the distillation-only checkpoint stayed best; that
    # is a legal selection (review B1 of iteration 3).
    if episodic_frozen is not None:
        for p, state in zip(model.parameters(), episodic_frozen):
            p.requires_grad_(state)
    receipt = {"status": "completed", "args": vars(args), "source_sessions": source_ids,
               "target_loaded": False, "protocol": "episodic set-attention pretraining",
               "best_step": best_step, "best_source_val_trunk_r2": best_score,
               "profile_dim": profile_dim, "stats_dim": stats_dim,
               "episode_trials": args.episode_trials, "warm_start": warm,
               "distill": distill_info,
               "parameters": {"read_in": sum(p.numel() for p in model.read_in.parameters()),
                              "activity_encoder": sum(p.numel() for p in
                                                      model.activity_encoder.parameters()),
                              "base": sum(p.numel() for p in model.base.parameters()),
                              "head": sum(p.numel() for p in model.head.parameters())},
               "fit_seconds": time.perf_counter() - started, "code_hashes": hashes,
               "source_data_sha256": source_hashes, "runtime_identity": runtime,
               "train_log_sha256": hashlib.sha256(
                   (out / "train_log.jsonl").read_bytes()).hexdigest()}
    torch.save({**torch.load(out / "best.pt", map_location="cpu", weights_only=False),
                "set_pretraining_receipt": receipt}, out / "best.pt")
    receipt["checkpoint_sha256"] = hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dandi688-sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-from-ap", default=None,
                        help="converged AP checkpoint for trunk/head warm start and distillation")
    for name, default in [("k", 10), ("width", 128), ("layers", 2), ("state-size", 32),
                          ("context", 128), ("batch-size", 32), ("steps", 2000),
                          ("val-interval", 100), ("max-val-endpoints", 1024),
                          ("episode-trials", 8), ("lags", 2), ("seed", 0),
                          ("distill-steps", 500), ("freeze-trunk", -1),
                          ("act-hidden", 16)]:
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in [("lr", 1e-3), ("weight-decay", 1e-3), ("dropout", 0.2),
                          ("gain-sigma", 0.3), ("offset-sigma", 0.1),
                          ("lam-white", 0.1), ("lam-head", 0.5),
                          ("unit-dropout", 0.3), ("desc-noise", 0.05),
                          ("profile-dropout", 0.3), ("tau", 0.5)]:  # float group
        parser.add_argument("--" + name, type=float, default=default)
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
