#!/usr/bin/env python
"""Protocol D-A target rows (v2, contiguous streaming): day-3/4/5 sessions against the day-1/2 backbone.

Every row decodes the whole target session as ONE causal stream (model state, front-end integrator and
ridge EMA carried across trials); only the bins of the non-calibration (query) trials are scored.
The trunk is frozen; target-side data are the B calibration trials only (z-score statistics, labels,
fitting segments = maximal runs of adjacent calibration trials from zero state).  Nothing is selected
on target query: the ridge (lam, ema) come from source simulation per B, the std floor from source
data, ui steps/lr are fixed in advance.  ui_prior and within_* rows are references, not candidates.
B = 42 is a diagnostic budget (half of the session) used with the within-session rows.

Rows are appended to rows.jsonl as soon as they exist (resumable).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_backbone as hb  # noqa: E402
from ssm_decode.ap_backbone import APBackbone  # noqa: E402
from ssm_decode.db6_data import N_CH, N_CLASS  # noqa: E402

BUDGETS = [1, 2, 4, 8, 16]
DIAG_B = 42
DIAG_DRAWS = 3
UI_STEPS, UI_LR = 300, 3e-3
WITHIN_STEPS = 1000


def sup_init_table(x, y, k):
    """Supervised table init (APBackbone.init_from_supervised) on flattened calibration bins x [n,N], y [n]."""
    lab = np.eye(N_CLASS, dtype=np.float32)[y]
    fake = types.SimpleNamespace(k=k, tables=[torch.nn.Parameter(torch.zeros(k, N_CH))])
    APBackbone.init_from_supervised(fake, [torch.from_numpy(x)], [torch.from_numpy(lab - lab.mean(0))])
    return fake.tables[0].detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge-cfg", required=True, help="ridge_select.json written by select_db6_ridge.py")
    ap.add_argument("--ridge-cfg-alt", default=None,
                    help="optional second ridge selection (e.g. made on original-order source streams) -> method rownorm_ridge_origsel")
    ap.add_argument("--floor", type=float, default=None, help="relative std floor for B-trial statistics (default: the checkpoint's training floor)")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--subjects", default="1-10")
    ap.add_argument("--seed-tag", type=int, default=0)
    ap.add_argument("--only", default="all", choices=["all", "light", "reuse"],
                    help="reuse = zero-backprop table rows only; light = reuse + closed-form ridge (no gradient fits); all = everything")
    ap.add_argument("--order", default="original", choices=["original", "shuffled"],
                    help="shuffled = decode the target session with its trials in a fixed random order (diagnostic: removes "
                         "the 12-repetition grasp blocks); the calibration trials are the same trials as in the original order")
    ap.add_argument("--no-stats-check", action="store_true",
                    help="skip the source-statistics consistency check (v1 checkpoints used a different std-floor formula)")
    ap.add_argument("--no-diag", action="store_true", help="skip the B=42 diagnostic budget")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
    dev = torch.device("cuda:0")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for ln in rows_path.read_text().splitlines():
            r = json.loads(ln); done.add((r["target"], r["B"], r["draw"], r["method"]))
    lo, hi = (int(v) for v in a.subjects.split("-"))
    model, blob = hb.load_backbone(a.ckpt, device="cuda:0")
    k = model.k
    keys = [tuple(x) for x in blob["keys"]]
    assert keys == hb.source_keys(), "checkpoint source keys differ from protocol keys"
    tau = blob["cfg"]["tau_ms"]
    if not a.no_stats_check:
        src = hb.SourceSet(keys, tau_ms=tau, floor=blob["cfg"].get("std_floor_train", hb.TRAIN_STD_FLOOR))
        for (m0, s0), (m1, s1) in zip(blob["stats"], src.stats):
            assert np.allclose(m0, m1, rtol=1e-4) and np.allclose(s0, s1, rtol=1e-3), "source stats differ from checkpoint"
    ridge_cfg = {int(b): v for b, v in json.loads(Path(a.ridge_cfg).read_text()).items()}
    ridge_alt = ({int(b): v for b, v in json.loads(Path(a.ridge_cfg_alt).read_text()).items()}
                 if a.ridge_cfg_alt else None)
    tables = torch.stack([t.detach() for t in model.tables])            # [S, K, N]
    mean_all = tables.mean(0)
    h1_ref = hb.trunk_hash(model)
    table_bytes_fp16 = k * N_CH * 2
    fl = a.floor if a.floor is not None else blob["cfg"].get("std_floor_train", hb.TRAIN_STD_FLOOR)

    def write(row):
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def tt(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    for subj in range(lo, hi + 1):
        li = hb.latest_source_index(keys, subj)
        tab_lat = tables[li]
        mu_lat, sd_lat = blob["stats"][li]
        own = torch.as_tensor([i for i, kk in enumerate(keys) if kk[0] != subj])
        loso = tables[own].mean(0)
        for key in hb.target_keys([subj]):
            tname = f"S{key[0]}_D{key[1]}_T{key[2]}"
            S = hb.load_stream(key, tau)
            inv = None
            if a.order == "shuffled":
                perm = np.random.default_rng([4242, key[0], key[1] * 10 + key[2]]).permutation(S.n)
                inv = np.argsort(perm)
                S0 = S
                S = hb.reorder_stream(S0, perm)
            x_src = hb.zscore(S.x, mu_lat, sd_lat)                      # source-statistics normalization, whole stream
            x_src_t = tt(x_src)
            lg_src = hb.decode_stream(model.base, x_src_t, tab_lat)[0]  # one pass, independent of B / draw
            for b in BUDGETS + ([] if a.no_diag else [DIAG_B]):
                n_draws = a.draws if b != DIAG_B else min(a.draws, DIAG_DRAWS)
                rng_all = [np.random.default_rng([7919, key[0], key[1] * 10 + key[2], b, d]) for d in range(n_draws)]
                cal_sets = [hb.draw_calibration(b, rng_all[d], S0.grasp if inv is not None else S.grasp,
                                                S0.rep if inv is not None else S.rep) for d in range(n_draws)]
                if inv is not None:
                    cal_sets = [sorted(int(inv[i]) for i in c) for c in cal_sets]
                q_sets = [[i for i in range(S.n) if i not in set(c)] for c in cal_sets]
                stats = [hb.input_stats(S, c, fl) for c in cal_sets]
                x_cal = [hb.zscore(S.x, *st) for st in stats]             # whole stream, B-trial statistics
                common = {"target": tname, "subject": key[0], "day": key[1], "tod": key[2], "gap": key[1] - 2,
                          "B": b, "seed": a.seed_tag, "k": k, "table_bytes_fp16": table_bytes_fp16,
                          "h5_pass": table_bytes_fp16 <= 32 * 1024, "latest_source": list(keys[li]),
                          "decoding": "contiguous", "order": a.order, "std_floor": fl}

                def emit(d, method, score, **extra):
                    row = {**common, "draw": d, "method": method, "cal_trials": cal_sets[d],
                           "coverage": hb.coverage(cal_sets[d], S.grasp), "n_query": len(q_sets[d]), **score, **extra}
                    row["h1_trunk_unchanged"] = hb.trunk_hash(model) == h1_ref
                    write(row)

                todo = lambda d, m: (tname, b, d, m) not in done  # noqa: E731
                # ---- zero-backprop table rows (one-pass stream decodes) ----
                emit_src = [d for d in range(n_draws) if todo(d, "reuse_latest_srcnorm")]
                for d in emit_src:
                    emit(d, "reuse_latest_srcnorm", hb.score_stream(lg_src, S, q_sets[d]), h2_backprop_steps=0, role="baseline")
                names = ["reuse_latest_calnorm", "mean_table_calnorm", "mean_loso_calnorm"]
                tabs = [tab_lat, mean_all, loso]
                jobs = [(d, n, t) for d in range(n_draws) for n, t in zip(names, tabs) if todo(d, n)]
                if jobs:
                    xs = torch.stack([tt(x_cal[d]) for d, _, _ in jobs])
                    lg = hb.decode_stream(model.base, xs, torch.stack([t for _, _, t in jobs]))
                    for j, (d, n, _) in enumerate(jobs):
                        emit(d, n, hb.score_stream(lg[j], S, q_sets[d]), h2_backprop_steps=0, role="baseline")
                    del xs
                if a.only == "reuse":
                    continue
                # ---- deployable closed-form ridge (fit on calibration segments, one EMA pass over the stream) ----
                for mname, cfgs in (("rownorm_ridge", ridge_cfg), ("rownorm_ridge_origsel", ridge_alt)):
                    if cfgs is None:
                        continue
                    rc = cfgs[b]
                    for d in range(n_draws):
                        if todo(d, mname):
                            segs, ys = hb.segment_arrays(S, x_cal[d], cal_sets[d])
                            w, bias = hb.ridge_fit(segs, ys, rc["lam"], rc["ema"])
                            lg_r = hb.ridge_logits(x_cal[d], w, bias, rc["ema"])
                            emit(d, mname, hb.score_stream(lg_r, S, q_sets[d]), h2_backprop_steps=0,
                                 role="deployable_closed_form", ridge_lam=rc["lam"], ridge_ema=rc["ema"],
                                 param_bytes_fp16=(N_CH * N_CLASS + 2 * N_CH + N_CLASS) * 2)
                if a.only == "light":
                    continue
                # ---- ui_prior (gradient through the frozen trunk; reference only) ----
                variants = [("ui_prior_latest_srcnorm", tab_lat, "src"), ("ui_prior_latest_calnorm", tab_lat, "cal"),
                            ("ui_prior_mean_calnorm", mean_all, "cal")]
                if any(todo(d, v[0]) for d in range(n_draws) for v in variants):
                    segs_all, ys_all, probs, inits = [], [], [], []
                    for vi, (_, init, norm) in enumerate(variants):
                        for d in range(n_draws):
                            sg, yl = hb.segment_arrays(S, x_src if norm == "src" else x_cal[d], cal_sets[d])
                            segs_all += [tt(v) for v in sg]; ys_all += [tt(v) for v in yl]
                            probs += [vi * n_draws + d] * len(sg); inits.append(init)
                    fit, loss = hb.fit_tables(model, segs_all, ys_all, probs, torch.stack(inits).to(dev),
                                              steps=UI_STEPS, lr=UI_LR)
                    xs = torch.stack([x_src_t if variants[vi][2] == "src" else tt(x_cal[d])
                                      for vi in range(3) for d in range(n_draws)])
                    lg = hb.decode_stream(model.base, xs, fit)
                    for vi, (name, _, _) in enumerate(variants):
                        for d in range(n_draws):
                            if todo(d, name):
                                emit(d, name, hb.score_stream(lg[vi * n_draws + d], S, q_sets[d]),
                                     h2_backprop_steps=UI_STEPS, role="reference_nondeployable",
                                     fit_loss=float(loss[vi * n_draws + d]))
                    del xs, segs_all, ys_all
                # ---- within-session diagnostics (B = DIAG_B only: half the session trains, half queries) ----
                if b == DIAG_B:
                    wnames = ("within_table_supinit", "within_table_latestinit_long")
                    if any(todo(d, n) for d in range(n_draws) for n in wnames):
                        segs_all, ys_all, probs, flat = [], [], [], []
                        for d in range(n_draws):
                            sg, yl = hb.segment_arrays(S, x_cal[d], cal_sets[d])
                            segs_all += [tt(v) for v in sg]; ys_all += [tt(v) for v in yl]; probs += [d] * len(sg)
                            flat.append((np.concatenate(sg), np.concatenate(yl)))
                        inits_sup = [sup_init_table(f, y, k) for f, y in flat]
                        xs = torch.stack([tt(x_cal[d]) for d in range(n_draws)])
                        for name, inits in zip(wnames, (inits_sup, [tab_lat] * n_draws)):
                            fit, loss = hb.fit_tables(model, segs_all, ys_all, probs, torch.stack(inits).to(dev),
                                                      steps=WITHIN_STEPS, lr=UI_LR)
                            lg = hb.decode_stream(model.base, xs, fit)
                            for d in range(n_draws):
                                if todo(d, name):
                                    emit(d, name, hb.score_stream(lg[d], S, q_sets[d]), h2_backprop_steps=WITHIN_STEPS,
                                         role="diagnostic_within_session", fit_loss=float(loss[d]))
                        del xs, segs_all, ys_all
            print(f"target {tname} done", flush=True)


if __name__ == "__main__":
    main()
