#!/usr/bin/env python
"""Protocol H-A target rows: session2 of each subject against the session1 backbone.

Rows are appended to rows.jsonl as soon as they exist (resumable).  The trunk is
frozen; target-side data are the B calibration trials only; nothing is selected
on target query (ridge lambda/ema come from source validation, ui steps/lr are
fixed in advance).  ui_prior and within_* rows are references, not candidates.
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
from ssm_decode import hyser_backbone as hb  # noqa: E402
from ssm_decode.ap_backbone import APBackbone  # noqa: E402
from ssm_decode.hyser_data import N_CH, N_FINGER, load_session  # noqa: E402

BUDGETS = [1, 2, 5, 10]
UI_STEPS, UI_LR = 300, 3e-3
WITHIN_STEPS = 1000
RIDGE_LAMS = [0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_EMAS = [0.0, 0.5, 0.8, 0.9, 0.95]


def draw_calibration(b, rng, fingers, samples):
    """Balanced draw of b trials: round-robin over a random finger order, random unused sample."""
    order = list(rng.permutation(5) + 1)
    used = {f: [] for f in order}
    chosen = []
    i = 0
    while len(chosen) < b:
        f = order[i % 5]; i += 1
        free = [s for s in (1, 2, 3) if s not in used[f]]
        s = int(rng.choice(free)); used[f].append(s)
        chosen.append(int(np.where((fingers == f) & (samples == s))[0][0]))
    return sorted(chosen)


def select_ridge(src):
    """Pick (lam, ema) by mean source-validation R2 (session1 held-out trials only)."""
    scores = {}
    for lam in RIDGE_LAMS:
        for ema in RIDGE_EMAS:
            vals = []
            for s in range(len(src.subjects)):
                tr, va = src.train_idx[s], src.val_idx[s]
                w, b = hb.ridge_fit(src.xz[s][tr], src.yz[s][tr], lam, ema)
                p = hb.ridge_predict(src.xz[s][va], w, b, ema)
                vals.append(hb.score_prediction(p, src.y[s][va], src.y_mean, src.y_std)["r2"])
            scores[f"{lam}|{ema}"] = float(np.mean(vals))
    best = max(scores, key=scores.get)
    lam, ema = (float(v) for v in best.split("|"))
    return {"lam": lam, "ema": ema, "scores": scores, "best_source_val_r2": scores[best]}


def sup_init_table(xz, yz, k):
    """Supervised table init (APBackbone.init_from_supervised) on one session's calibration data."""
    fake = types.SimpleNamespace(k=k, tables=[torch.nn.Parameter(torch.zeros(k, N_CH))])
    APBackbone.init_from_supervised(fake, [torch.from_numpy(xz.reshape(-1, N_CH))],
                                    [torch.from_numpy(yz.reshape(-1, N_FINGER))])
    return fake.tables[0].detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--targets", default="1-20")
    ap.add_argument("--seed-tag", type=int, default=0)
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
    lo, hi = (int(v) for v in a.targets.split("-"))
    model, blob = hb.load_backbone(a.ckpt, device="cuda:0")
    k = model.k
    src = hb.SourceSet()
    y_mean, y_std = blob["y_mean"], blob["y_std"]
    assert np.allclose(y_mean, src.y_mean) and np.allclose(y_std, src.y_std)
    ridge_cfg = select_ridge(src)
    (out / "ridge_select.json").write_text(json.dumps(ridge_cfg, indent=1))
    tables = torch.stack([t.detach() for t in model.tables])            # [20, K, N]
    mean_all = tables.mean(0)
    h1_ref = hb.trunk_hash(model)
    table_bytes_fp16 = k * N_CH * 2

    def write(row):
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def query_score(table, xq_z, yq):
        pred = hb.decode_trials(model, torch.from_numpy(xq_z).to(dev), table.to(dev)).cpu().numpy()
        return hb.score_prediction(pred, yq, y_mean, y_std)

    for t in range(lo, hi + 1):
        tab_s1 = tables[t - 1]
        mu_s1, sd_s1 = blob["stats"][t - 1]
        loso = (tables.sum(0) - tab_s1) / (len(tables) - 1)
        tgt = load_session(t, 2)
        x, y, fingers, samples = tgt["x"], tgt["y"], tgt["finger"], tgt["sample"]
        yz = ((y - y_mean) / y_std).astype(np.float32)
        for b in BUDGETS:
            rng_all = [np.random.default_rng([7919, t, b, d]) for d in range(a.draws)]
            cal_sets = [draw_calibration(b, rng_all[d], fingers, samples) for d in range(a.draws)]
            q_sets = [[i for i in range(15) if i not in c] for c in cal_sets]
            # per-draw calibration statistics (neural only) and z-scored tensors
            stats = [hb.input_stats(x[c]) for c in cal_sets]
            xcal_z = [hb.zscore(x[c], *st) for c, st in zip(cal_sets, stats)]
            xq_cal = [hb.zscore(x[q], *st) for q, st in zip(q_sets, stats)]
            xq_s1 = [hb.zscore(x[q], mu_s1, sd_s1) for q in q_sets]
            common = {"target": t, "B": b, "seed": a.seed_tag, "k": k,
                      "table_bytes_fp16": table_bytes_fp16, "h5_pass": table_bytes_fp16 <= 32 * 1024}

            def emit(d, method, score, **extra):
                row = {**common, "draw": d, "method": method, "cal_trials": cal_sets[d],
                       "n_query": len(q_sets[d]), **score, **extra}
                row["h1_trunk_unchanged"] = hb.trunk_hash(model) == h1_ref
                write(row)

            todo = lambda d, m: (t, b, d, m) not in done  # noqa: E731
            # ---- zero-backprop rows ----
            for d in range(a.draws):
                yq = y[q_sets[d]]
                cands = [("reuse_s1_srcnorm", tab_s1, xq_s1[d], 0, False),
                         ("reuse_s1_calnorm", tab_s1, xq_cal[d], 0, False),
                         ("mean_table_calnorm", mean_all, xq_cal[d], 0, False),
                         ("mean_loso_calnorm", loso, xq_cal[d], 0, False)]
                for name, tab, xq, bp, ref in cands:
                    if todo(d, name):
                        emit(d, name, query_score(tab, xq, yq), h2_backprop_steps=bp, role="baseline",
                             h4_uses_labels=(name.endswith("calnorm") and False) or False)
                if todo(d, "rownorm_ridge"):
                    w, bias = hb.ridge_fit(xcal_z[d], yz[cal_sets[d]], ridge_cfg["lam"], ridge_cfg["ema"])
                    p = hb.ridge_predict(xq_cal[d], w, bias, ridge_cfg["ema"])
                    emit(d, "rownorm_ridge", hb.score_prediction(p, yq, y_mean, y_std), h2_backprop_steps=0,
                         role="deployable_closed_form", ridge_lam=ridge_cfg["lam"], ridge_ema=ridge_cfg["ema"],
                         param_bytes_fp16=(N_CH * N_FINGER + 2 * N_CH + N_FINGER) * 2)
            # ---- ui_prior (gradient through the frozen trunk; reference only) ----
            variants = [("ui_prior_s1_srcnorm", tab_s1, "s1"), ("ui_prior_s1_calnorm", tab_s1, "cal"),
                        ("ui_prior_mean_calnorm", mean_all, "cal")]
            if any(todo(d, v[0]) for d in range(a.draws) for v in variants):
                xs, ys, inits = [], [], []
                for _, init, norm in variants:
                    for d in range(a.draws):
                        xs.append(hb.zscore(x[cal_sets[d]], mu_s1, sd_s1) if norm == "s1" else xcal_z[d])
                        ys.append(yz[cal_sets[d]]); inits.append(init)
                fit, loss = hb.fit_tables(model, torch.from_numpy(np.stack(xs)).to(dev),
                                          torch.from_numpy(np.stack(ys)).to(dev),
                                          torch.stack(inits).to(dev), steps=UI_STEPS, lr=UI_LR)
                for vi, (name, _, norm) in enumerate(variants):
                    for d in range(a.draws):
                        if not todo(d, name):
                            continue
                        xq = xq_s1[d] if norm == "s1" else xq_cal[d]
                        emit(d, name, query_score(fit[vi * a.draws + d], xq, y[q_sets[d]]),
                             h2_backprop_steps=UI_STEPS, role="reference_nondeployable",
                             fit_loss=float(loss[vi * a.draws + d]))
            # ---- within-session diagnostics (B=10 only: 10 train / 5 query trials) ----
            if b == 10:
                if any(todo(d, "within_table_supinit") or todo(d, "within_table_s1init_long") for d in range(a.draws)):
                    xs = [xcal_z[d] for d in range(a.draws)]
                    ys = [yz[cal_sets[d]] for d in range(a.draws)]
                    inits_sup = [sup_init_table(xcal_z[d], yz[cal_sets[d]], k) for d in range(a.draws)]
                    for name, inits in [("within_table_supinit", inits_sup),
                                        ("within_table_s1init_long", [tab_s1] * a.draws)]:
                        fit, loss = hb.fit_tables(model, torch.from_numpy(np.stack(xs)).to(dev),
                                                  torch.from_numpy(np.stack(ys)).to(dev),
                                                  torch.stack(inits).to(dev), steps=WITHIN_STEPS, lr=UI_LR)
                        for d in range(a.draws):
                            if todo(d, name):
                                emit(d, name, query_score(fit[d], xq_cal[d], y[q_sets[d]]),
                                     h2_backprop_steps=WITHIN_STEPS, role="diagnostic_within_session",
                                     fit_loss=float(loss[d]))
        print(f"target {t} done", flush=True)


if __name__ == "__main__":
    main()
