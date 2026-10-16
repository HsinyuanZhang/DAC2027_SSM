#!/usr/bin/env python
"""Q-D: episodic sEMG calibrators (train on SOURCE sessions only) and target-side evaluation.

Hyser H-A:
  train : python scripts/run_semg_calib.py train hyser --seed S --worker W --n-workers NW
          (all source x variant x subject-fold configs of one backbone seed, idempotent)
  eval  : python scripts/run_semg_calib.py eval hyser --seed S --targets 1-20
Outputs go to results/semg_calib_20261016/.  GPU1 only (CUDA_VISIBLE_DEVICES=1).
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_backbone as db6b  # noqa: E402
from ssm_decode import hyser_backbone as hysb  # noqa: E402
from ssm_decode.hyser_data import load_session  # noqa: E402
from ssm_decode.semg_calib import VARIANTS, SemgCalibrator, norm_stats  # noqa: E402
from ssm_decode.semg_calib_db6 import Db6CpuEnv, Db6StreamEnv, StreamStore  # noqa: E402
from ssm_decode.semg_calib_env import HyserEnv, SessStore, draw_hyser_cal, train_calibrator  # noqa: E402

OUT = ROOT / "results" / "semg_calib_20261016"
SOURCES = ("mean", "own", "free")
N_FOLDS = 5
HYSER_BUDGETS = [1, 2, 5, 10]
DB6_BUDGETS = [1, 2, 4, 8, 16]


def ckpt_paths(dataset, seed, source, variant, fold):
    d = OUT / dataset / f"seed{seed}" / f"{source}_{variant}_f{fold}"
    return d, d / "best.pt", d / "info.json"


def backbone_ckpt(dataset, seed, db6_ckpt=None):
    if dataset == "hyser":
        return ROOT / f"results/hyser_20261015/final/seed{seed}/best.pt"
    if not db6_ckpt:
        raise SystemExit("db6 needs --db6-ckpt (v2 contiguous backbone)")
    return Path(str(db6_ckpt).format(seed=seed))


def require_gpu1():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")


# ------------------------------------------------------------------- train ----

def hyser_jobs(sources, variants):
    return [(s, v, f) for s in sources for v in variants for f in range(N_FOLDS)]


def train_cmd(a):
    require_gpu1()
    env = HyserEnv(backbone_ckpt("hyser", a.seed), device="cuda:0")
    jobs = hyser_jobs(a.sources.split(","), a.variants.split(","))
    jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
    for source, variant, fold in jobs:
        d, ck, info_p = ckpt_paths("hyser", a.seed, source, variant, fold)
        if info_p.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        train_subs, val_subs, held = env.split_subjects(fold, N_FOLDS)
        val_eps = env.val_episodes(val_subs, train_subs, source)
        calib, info = train_calibrator(env, variant, source, seed=a.seed * 100 + fold, train_pool=train_subs,
                                       val_eps=val_eps, steps_cap=a.steps_cap, log=lambda s: None)
        info.update(dataset="hyser", seed=a.seed, fold=fold, train_subjects=[env.src.subjects[i] for i in train_subs],
                    val_subjects=[env.src.subjects[i] for i in val_subs],
                    heldout_subjects=[env.src.subjects[i] for i in held])
        torch.save({"state_dict": calib.state_dict(), "variant": variant, "source": source,
                    "out_scale": env.out_scale}, ck)
        info_p.write_text(json.dumps(info))
        print(f"[{time.strftime('%H:%M:%S')}] hyser seed{a.seed} {source} {variant} f{fold}: val0={info['val_step0']:.3f} "
              f"best={info['best_val']:.3f}@{info['best_step']} run={info['steps_run']} {info['seconds']:.0f}s",
              flush=True)


def db6_train_cmd(a):
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
        env = Db6StreamEnv(backbone_ckpt("db6", a.seed, a.db6_ckpt), device="cpu", std_floor=a.std_floor)
    else:
        require_gpu1()
        env = Db6StreamEnv(backbone_ckpt("db6", a.seed, a.db6_ckpt), device="cuda:0", std_floor=a.std_floor)
    jobs = [(s, v) for s in a.sources.split(",") for v in a.variants.split(",")]
    jobs = [j for i, j in enumerate(jobs) if i % a.n_workers == a.worker]
    for source, variant in jobs:
        d, ck, info_p = ckpt_paths(a.out_tag, a.seed, source, variant, "all")
        if info_p.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        pool = env.train_sessions(source)
        val_eps = env.val_episodes(source)
        calib, info = train_calibrator(env, variant, source, seed=a.seed * 100 + 7, train_pool=pool, val_eps=val_eps,
                                       steps_cap=a.steps_cap, lr=a.lr, weight_decay=a.wd, log=lambda s: None)
        info.update(dataset="db6", seed=a.seed, backbone=str(backbone_ckpt("db6", a.seed, a.db6_ckpt)),
                    train_sessions=[list(env.keys[j]) for j in pool],
                    val_sessions=[list(env.keys[j]) for j in env.val_sess])
        torch.save({"state_dict": calib.state_dict(), "variant": variant, "source": source,
                    "out_scale": env.out_scale}, ck)
        info_p.write_text(json.dumps(info))
        print(f"[{time.strftime('%H:%M:%S')}] db6 seed{a.seed} {source} {variant}: val0={info['val_step0']:.3f} "
              f"best={info['best_val']:.3f}@{info['best_step']} run={info['steps_run']} {info['seconds']:.0f}s", flush=True)


def db6_eval_cmd(a):
    if a.device == "cpu":                       # fallback when the GPU is unavailable (documented in the README)
        torch.set_num_threads(a.threads)
        env = Db6CpuEnv(backbone_ckpt("db6", a.seed, a.db6_ckpt), std_floor=a.std_floor)
        dev = torch.device("cpu")
    else:
        require_gpu1()
        dev = torch.device("cuda:0")
        env = Db6StreamEnv(backbone_ckpt("db6", a.seed, a.db6_ckpt), device="cuda:0", std_floor=a.std_floor)
    out = OUT / a.out_tag / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_{a.order}{a.rows_suffix}.jsonl"
    done = set()
    if rows_path.exists():
        for ln in rows_path.read_text().splitlines():
            r = json.loads(ln); done.add((r["target"], r["B"], r["draw"], r["method"]))
    lo, hi = (int(v) for v in a.targets.split("-"))
    sources, variants = a.sources.split(","), a.variants.split(",")
    calibs = {(s, v): load_calib(env, a.out_tag, a.seed, s, v, "all") for s in sources for v in variants}
    h1_ref = env.trunk_hash()
    table_bytes = env.k * 14 * 2
    for subj in range(lo, hi + 1):
        li = db6b.latest_source_index(env.keys, subj)
        priors = {"mean": env.prior_for("mean", li, exclude_val=False), "own": env.tables[li].clone(), "free": None}
        for key in db6b.target_keys([subj]):
            tname = f"S{key[0]}_D{key[1]}_T{key[2]}"
            S0 = db6b.load_stream(key, env.tau)
            S, inv = S0, None
            if a.order == "shuffled":                     # same permutation as scripts/run_db6_eval.py --order shuffled
                perm = np.random.default_rng([4242, key[0], key[1] * 10 + key[2]]).permutation(S0.n)
                inv = np.argsort(perm)
                S = db6b.reorder_stream(S0, perm)
            store = StreamStore(S, env.spec, dev)
            for b in a.budgets:
                for d in range(a.draws):
                    rng = np.random.default_rng([7919, key[0], key[1] * 10 + key[2], b, d])
                    cal = db6b.draw_calibration(b, rng, S0.grasp, S0.rep)
                    if inv is not None:
                        cal = sorted(int(inv[i]) for i in cal)
                    q = [i for i in range(S.n) if i not in set(cal)]
                    st = store.stats(cal)                                      # B calibration trials only
                    names, zs = [], []
                    for source in sources:
                        for v in variants:
                            name = f"cal_{v}_{source}"
                            if (tname, b, d, name) in done:
                                continue
                            res = calibs[(source, v)].calibrate(st, priors[source])
                            zs.append(((store.x - res.mean.float()) / res.std.float()) @ res.table.to(dev).T)
                            names.append((name, v, source, bool(res.grad_enabled_inside)))
                    if a.check_ref:                    # recompute two stored reference rows with this very pipeline
                        mean_c, std_c = norm_stats(st, env.spec.std_floor_ref, env.spec.std_floor)
                        xc = (store.x - mean_c.float()) / std_c.float()
                        for cname, ctab in (("chk_reuse_latest_calnorm", env.tables[li]),
                                            ("chk_mean_table_calnorm", env.tables.mean(0))):
                            if (tname, b, d, cname) not in done:
                                zs.append(xc @ ctab.to(dev).T)
                                names.append((cname, "ref", "ref", False))
                    if not zs:
                        continue
                    logits = env.decode(torch.stack(zs))
                    for i, (name, v, source, ge) in enumerate(names):
                        sc = db6b.score_stream(logits[i], S, q)
                        with rows_path.open("a") as fh:
                            fh.write(json.dumps({
                                "target": tname, "subject": key[0], "day": key[1], "tod": key[2], "gap": key[1] - 2,
                                "B": b, "draw": d, "seed": a.seed, "method": name, "variant": v, "source": source,
                                "cal_trials": cal, "coverage": db6b.coverage(cal, S.grasp), "n_query": len(q), "order": a.order, **sc,
                                "h1_trunk_unchanged": env.trunk_hash() == h1_ref, "h2_backprop_steps": 0,
                                "h2_grad_enabled_inside": ge, "h4_trials_used": len(cal),
                                "table_bytes_fp16": table_bytes, "h5_pass": table_bytes <= 32 * 1024}) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] db6 seed{a.seed} {tname} done", flush=True)


def load_calib(env, dataset, seed, source, variant, fold):
    _, ck, _ = ckpt_paths(dataset, seed, source, variant, fold)
    blob = torch.load(ck, map_location=env.dev, weights_only=False)
    mode = "free" if source == "free" else "aligned"
    m = SemgCalibrator(variant, env.spec, prior_mode=mode, out_scale=blob["out_scale"]).to(env.dev)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# -------------------------------------------------------------------- eval ----

def hyser_eval_cmd(a):
    require_gpu1()
    dev = torch.device("cuda:0")
    env = HyserEnv(backbone_ckpt("hyser", a.seed), device="cuda:0")
    model = env.model
    out = OUT / "hyser" / f"seed{a.seed}"
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for ln in rows_path.read_text().splitlines():
            r = json.loads(ln); done.add((r["target"], r["B"], r["draw"], r["method"]))
    lo, hi = (int(v) for v in a.targets.split("-"))
    sources = a.sources.split(",")
    variants = a.variants.split(",")
    calibs = {}
    for source in sources:
        for v in variants:
            for f in range(N_FOLDS):
                calibs[(source, v, f)] = load_calib(env, "hyser", a.seed, source, v, f)
    h1_ref = hysb.trunk_hash(model)
    table_bytes = env.k * 256 * 2
    tables = env.tables
    y_mean, y_std = env.y_mean, env.y_std

    def write(row):
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    for t in range(lo, hi + 1):
        fold = (t - 1) % N_FOLDS
        tgt = load_session(t, 2)
        x, y, fingers, samples = tgt["x"], tgt["y"], tgt["finger"], tgt["sample"]
        xg = torch.from_numpy(x).to(dev)
        yz = ((y - y_mean) / y_std).astype(np.float32)
        store = SessStore(xg, (lambda i, L, yz_=torch.from_numpy(yz).to(dev): yz_[i, :L].double()),
                          [x.shape[1]] * x.shape[0], env.spec, dev)
        priors = {"mean": tables[[j for j in range(tables.shape[0]) if j != t - 1]].mean(0),
                  "own": tables[t - 1].clone(), "free": None}
        for b in HYSER_BUDGETS:
            for d in range(a.draws):
                rng = np.random.default_rng([7919, t, b, d])
                cal = draw_hyser_cal(b, rng, fingers, samples)
                q = [i for i in range(15) if i not in cal]
                st = store.stats(cal)                                   # B calibration trials only
                if set(cal) - set(store.bank.rows):
                    raise RuntimeError("calibration trial missing from bank")
                for source in sources:
                    for v in variants:
                        name = f"cal_{v}_{source}"
                        if (t, b, d, name) in done:
                            continue
                        c = calibs[(source, v, fold)]
                        res = c.calibrate(st, priors[source])
                        mean, std = res.mean.float(), res.std.float()
                        xq = ((xg[q] - mean) / std)
                        pred = hysb.decode_trials(model, xq, res.table.to(dev)).cpu().numpy()
                        sc = hysb.score_prediction(pred, y[q], y_mean, y_std)
                        write({"target": t, "B": b, "draw": d, "seed": a.seed, "method": name, "variant": v,
                               "source": source, "fold": fold, "cal_trials": cal, "n_query": len(q), **sc,
                               "h1_trunk_unchanged": hysb.trunk_hash(model) == h1_ref,
                               "h2_backprop_steps": 0, "h2_grad_enabled_inside": bool(res.grad_enabled_inside),
                               "h4_trials_used": len(cal), "table_bytes_fp16": table_bytes,
                               "h5_pass": table_bytes <= 32 * 1024})
                # pipeline equivalence check against the stored reference row (reuse_s1 + B-trial stats)
                if a.check and (t, b, d, "chk_reuse_s1_calnorm") not in done:
                    mean, std = (v_.float() for v_ in norm_stats(st))
                    pred = hysb.decode_trials(model, (xg[q] - mean) / std, tables[t - 1]).cpu().numpy()
                    sc = hysb.score_prediction(pred, y[q], y_mean, y_std)
                    write({"target": t, "B": b, "draw": d, "seed": a.seed, "method": "chk_reuse_s1_calnorm",
                           "cal_trials": cal, **sc})
        print(f"[{time.strftime('%H:%M:%S')}] hyser seed{a.seed} target {t} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("dataset", choices=["hyser", "db6"])
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--sources", default="mean,own,free")
        p.add_argument("--variants", default=",".join(VARIANTS))
        p.add_argument("--worker", type=int, default=0)
        p.add_argument("--n-workers", type=int, default=1)
        p.add_argument("--steps-cap", type=int, default=4000)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-2)
        p.add_argument("--targets", default="1-20")  # hyser: subjects 1-20; db6: subjects 1-10
        p.add_argument("--draws", type=int, default=None)
        p.add_argument("--check", action="store_true")
        p.add_argument("--db6-ckpt", default=None, help="path template, {seed} is substituted")
        p.add_argument("--out-tag", default="db6v2")
        p.add_argument("--std-floor", type=float, default=None, help="db6: relative std floor of B-trial statistics")
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--rows-suffix", default="", help="db6 eval: suffix of the rows file (scratch checks)")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--check-ref", action="store_true", help="db6 eval: also recompute reuse_latest_calnorm and mean_table_calnorm")
        p.add_argument("--order", default="original", choices=["original", "shuffled"],
                       help="db6 eval: decode the target stream in recorded or trial-shuffled order")
        p.add_argument("--budgets", type=int, nargs="+", default=DB6_BUDGETS)
    a = ap.parse_args()
    if a.draws is None:
        a.draws = 8 if a.dataset == "hyser" else 5
    if a.dataset == "hyser":
        {"train": train_cmd, "eval": hyser_eval_cmd}[a.cmd](a)
    else:
        {"train": db6_train_cmd, "eval": db6_eval_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
