"""Episodic pretraining for the E1 set-attention backbone on 000688 (C8.4).

Each training step is one episode:

1. Sample a source session and a small descriptor trial subset.
2. Compute per-unit descriptors on those trials (labels enter here only).
3. Fold the K x N table through the set read-in.
4. Decode windows from OTHER trials of the same session.

Set-property augmentation runs every step: unit subsampling (columns and
descriptor rows drop together), channel gain jitter, descriptor noise, and
profile-block dropout so the label-free descriptor stays in distribution.
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
from .session_pretraining import continuous_source_windows
from .set_attention import SetAttentionBackbone, normalize_descriptors, unit_descriptors

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
    stats = []
    full_desc = []
    for record in records:
        x = record.neural.astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        stats.append((mean.astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32)))
        desc, _ = unit_descriptors(record, np.arange(len(x)), lags=args.lags, with_profile=True)
        full_desc.append(desc)
    normed_ref, desc_stats = normalize_descriptors(full_desc)
    desc_mean = desc_stats["mean"]
    desc_std = desc_stats["std"]
    desc_dim = full_desc[0].shape[1]
    outputs = records[0].behavior.shape[1]
    profile_cols = outputs * (args.lags + 1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Full-session tensors for episode sampling; validation uses val trials only.
    train_windows = []
    for record, (x_mean, x_std) in zip(records, stats):
        x = ((record.neural - x_mean) / x_std).astype(np.float32)
        y = np.where(record.eval_mask[:, None], (record.behavior - y_mean) / y_std, np.nan)
        train_windows.append(tuple(torch.from_numpy(v).to(device) for v in (x, y, record.eval_mask)))
    val_windows = []
    for record, (x_mean, x_std), (_, va) in zip(records, stats, splits):
        x = ((record.neural - x_mean) / x_std).astype(np.float32)
        y = np.full_like(record.behavior, np.nan)
        mask = np.zeros(len(x), dtype=bool)
        for a, b in va:
            y[a:b] = (record.behavior[a:b] - y_mean) / y_std
            mask[a:b] = record.eval_mask[a:b]
        val_windows.append(tuple(torch.from_numpy(v).to(device) for v in (x, y, mask)))
    rng = np.random.default_rng(args.seed)
    model = SetAttentionBackbone(desc_dim, outputs, k=args.k, width=args.width,
                                 layers=args.layers, state_size=args.state_size,
                                 dropout=args.dropout, tau=args.tau).to(device)
    optimizer = torch.optim.AdamW(_groups(list(model.parameters()), args.lr, args.weight_decay))
    runtime = _runtime_identity()
    # Validation descriptors come from TRAIN trials only, so validation
    # labels never leak into the fold used for best-step selection (B2).
    val_desc_bins = []
    for (tr, _) in splits:
        picks = sorted(rng.choice(len(tr), size=min(args.episode_trials, len(tr)), replace=False))
        val_desc_bins.append(np.concatenate([np.arange(tr[i][0], tr[i][1]) for i in picks]))

    def episode_desc(index, trial_indices, profile_dropout, noise):
        record = records[index]
        bins = np.concatenate([np.arange(a, b) for a, b in
                               [splits[index][0][i] for i in trial_indices]])
        desc, info = unit_descriptors(record, bins, lags=args.lags, with_profile=True)
        desc = (desc - desc_mean) / desc_std
        if profile_dropout:
            desc[:, :profile_cols] = 0.0
        return desc + noise * rng.standard_normal(desc.shape), info

    def fold(desc_np, keep_units):
        """Fold with the unit mask inside the softmax (review B1): attention
        renormalizes over kept units and dropped columns are exactly zero,
        keeping the table width equal to the input width.
        """
        desc = torch.from_numpy(desc_np.astype(np.float32)).to(device)
        keep_t = None
        if keep_units is not None:
            keep_t = torch.from_numpy(keep_units).to(device)
        return model.read_in.fold_table(desc, keep_t)

    def validate():
        model.eval()
        scores = []
        with torch.inference_mode():
            for index, window in enumerate(val_windows):
                desc, _ = unit_descriptors(records[index], val_desc_bins[index],
                                           lags=args.lags, with_profile=True)
                desc = (desc - desc_mean) / desc_std
                table = fold(desc, None)
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

    best_score, best_step = -float("inf"), 0
    log = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        index = int(rng.integers(len(records)))
        window = train_windows[index]
        x_all, y_all, _ = window
        trials = splits[index][0]
        desc_pick = sorted(rng.choice(len(trials), size=min(args.episode_trials, len(trials)),
                                      replace=False))
        keep = np.ones(x_all.shape[1], dtype=bool)
        drop = rng.random(len(keep)) < args.unit_dropout
        if drop.all():
            drop[int(rng.integers(len(keep)))] = False
        keep[drop] = False
        keep_t = torch.from_numpy(keep).to(device).float()
        desc_np, _ = episode_desc(index, desc_pick,
                                  profile_dropout=rng.random() < args.profile_dropout,
                                  noise=args.desc_noise)
        table = fold(desc_np, keep)
        other = [i for i in range(len(trials)) if i not in set(desc_pick)]
        picks = []
        span = args.context - 1
        for _ in range(args.batch_size):
            a, b = trials[other[int(rng.integers(len(other)))]]
            lo, hi = a + span, b - 1
            if hi < lo:
                lo = hi = b - 1
            picks.append((0, int(min(hi, x_all.shape[0] - 1))))
        xb, yb, mb = d._right_aligned([window], picks, args.context)
        xb = xb * keep_t
        xb = drift_augment(xb, args.gain_sigma, args.offset_sigma, 0.0)
        z = xb @ table.T
        decode = model.base(z)[:, args.context // 2:]
        head = model.head(z)[:, args.context // 2:]
        truth = yb[:, args.context // 2:]
        good = mb[:, args.context // 2:] & torch.isfinite(truth).all(-1)
        if not good.any() or not torch.isfinite(decode).all():
            import os
            if os.environ.get("SET_DEBUG"):
                print(f"SKIP step={step} good={int(good.sum())} finite={bool(torch.isfinite(decode).all())} xb={tuple(xb.shape)} table={tuple(table.shape)}")
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
            score = validate()
            if not np.isfinite(score):
                raise FloatingPointError("source validation score is not finite")
            event = {"step": step, "loss": float(loss.detach()), "gradient_norm": gradient,
                     "source_val_trunk_r2": score,
                     "elapsed_seconds": time.perf_counter() - started}
            log.append(event)
            with (out / "train_log.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            if score > best_score:
                best_score, best_step = score, step
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "desc_dim": desc_dim, "outputs": outputs,
                            "source_sessions": source_ids, "best_step": step,
                            "best_source_val_trunk_r2": best_score,
                            "desc_stats": {"mean": desc_mean.tolist(),
                                           "std": desc_std.tolist()},
                            "label_stats": {"mean": y_mean.tolist(),
                                            "std": y_std.tolist()},
                            "code_hashes": hashes, "source_data_sha256": source_hashes,
                            "runtime_identity": runtime}, out / "best.pt")
    if best_step == 0:
        raise FloatingPointError("no validation point was reached")
    receipt = {"status": "completed", "args": vars(args), "source_sessions": source_ids,
               "target_loaded": False, "protocol": "episodic set-attention pretraining",
               "best_step": best_step, "best_source_val_trunk_r2": best_score,
               "desc_dim": desc_dim, "episode_trials": args.episode_trials,
               "parameters": {"read_in": sum(p.numel() for p in model.read_in.parameters()),
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
    for name, default in [("k", 16), ("width", 128), ("layers", 2), ("state-size", 32),
                          ("context", 128), ("batch-size", 32), ("steps", 2000),
                          ("val-interval", 100), ("max-val-endpoints", 1024),
                          ("episode-trials", 8), ("lags", 2), ("seed", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in [("lr", 1e-3), ("weight-decay", 1e-3), ("dropout", 0.2),
                          ("gain-sigma", 0.3), ("offset-sigma", 0.1),
                          ("lam-white", 0.1), ("lam-head", 0.5),
                          ("unit-dropout", 0.3), ("desc-noise", 0.05),
                          ("profile-dropout", 0.3), ("tau", 0.5)]:
        parser.add_argument("--" + name, type=float, default=default)
    print(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
