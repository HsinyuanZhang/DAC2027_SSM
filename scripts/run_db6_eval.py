#!/usr/bin/env python
"""Protocol D-A target rows: day-3/4/5 sessions of each subject against the day-1/2 backbone.

Rows are appended to rows.jsonl as soon as they exist (resumable).  The trunk is frozen;
target-side data are the B calibration trials only; nothing is selected on target query
(ridge lambda/ema come from source simulation per B, ui steps/lr are fixed in advance).
ui_prior and within_* rows are references, not candidates.  B = 42 is a diagnostic budget
(half of the session) used with the within-session rows.
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
# v1 grid used for the rows in rows.jsonl (method rownorm_ridge): LAMS 0.01..1000, EMAS 0..0.95.  Source-only
# selection hit the grid edge (lam 0.01 / ema 0.95), so --ridge-only reruns the ridge rows on the extended grid
# below (method rownorm_ridge_v2, written to rows_ridge_v2.jsonl, selection in ridge_select_v2.json).
RIDGE_LAMS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RIDGE_EMAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]


def select_ridge(src, b, n_draws=3):
    """Pick (lam, ema) for budget b: simulate b-trial calibration inside every source session
    (draws from its training trials), score balanced accuracy on its held-out repetitions."""
    scores = {f"{lam}|{ema}": [] for lam in RIDGE_LAMS for ema in RIDGE_EMAS}
    for s in range(len(src.keys)):
        S, tr, va = src.sess[s], src.train_idx[s], src.val_idx[s]
        for d in range(n_draws):
            rng = np.random.default_rng([104729, s, b, d])
            loc = hb.draw_calibration(min(b, len(tr)), rng, S.grasp[tr], S.rep[tr])
            cal = tr[loc]
            mu, sd = hb.input_stats(S.x[cal], S.lens[cal])
            xc, xv = hb.zscore(S.x[cal], mu, sd, S.lens[cal]), hb.zscore(S.x[va], mu, sd, S.lens[va])
            for lam in RIDGE_LAMS:
                for ema in RIDGE_EMAS:
                    w, bias = hb.ridge_fit(xc, S.y[cal], S.lens[cal], lam, ema)
                    lg = hb.ridge_logits(xv, w, bias, ema)
                    scores[f"{lam}|{ema}"].append(hb.score_logits(lg, S.y[va], S.lens[va], S.grasp[va])["bal_acc"])
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(mean, key=mean.get)
    lam, ema = (float(v) for v in best.split("|"))
    return {"B": b, "lam": lam, "ema": ema, "best_source_sim_bal_acc": mean[best], "scores": mean}


def sup_init_table(xz, y, lens, k):
    """Supervised table init (APBackbone.init_from_supervised) on one session's calibration data."""
    m = hb.valid_mask(lens, xz.shape[1])
    lab = np.eye(N_CLASS, dtype=np.float32)[y[m]]
    fake = types.SimpleNamespace(k=k, tables=[torch.nn.Parameter(torch.zeros(k, N_CH))])
    APBackbone.init_from_supervised(fake, [torch.from_numpy(xz[m])], [torch.from_numpy(lab - lab.mean(0))])
    return fake.tables[0].detach()


def trim(arrs, lens):
    top = int(max(int(np.max(l)) for l in lens))
    return [a[:, :top] for a in arrs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--subjects", default="1-10")
    ap.add_argument("--seed-tag", type=int, default=0)
    ap.add_argument("--ridge-only", action="store_true", help="only the v2 closed-form ridge rows")
    ap.add_argument("--no-diag", action="store_true", help="skip the B=42 diagnostic budget")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 only: launch with CUDA_VISIBLE_DEVICES=1")
    dev = torch.device("cuda:0")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / ("rows_ridge_v2.jsonl" if a.ridge_only else "rows.jsonl")
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
    src = hb.SourceSet(keys, tau_ms=tau)
    for (m0, s0), (m1, s1) in zip(blob["stats"], src.stats):
        assert np.allclose(m0, m1) and np.allclose(s0, s1)
    ridge_path = out / ("ridge_select_v2.json" if a.ridge_only else "ridge_select.json")
    if ridge_path.exists():
        ridge_cfg = {int(b): v for b, v in json.loads(ridge_path.read_text()).items()}
    else:
        ridge_cfg = {b: select_ridge(src, b) for b in BUDGETS + [DIAG_B]}
        ridge_path.write_text(json.dumps(ridge_cfg, indent=1))
    tables = torch.stack([t.detach() for t in model.tables])            # [S, K, N]
    mean_all = tables.mean(0)
    h1_ref = hb.trunk_hash(model)
    table_bytes_fp16 = k * N_CH * 2

    def write(row):
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    for subj in range(lo, hi + 1):
        li = hb.latest_source_index(keys, subj)
        tab_lat = tables[li]
        mu_lat, sd_lat = blob["stats"][li]
        own = torch.as_tensor([i for i, kk in enumerate(keys) if kk[0] != subj])
        loso = tables[own].mean(0)
        for key in hb.target_keys([subj]):
            tname = f"S{key[0]}_D{key[1]}_T{key[2]}"
            S = hb.load_sess(key, tau)
            n_tr = S.n
            for b in BUDGETS + ([] if a.no_diag else [DIAG_B]):
                n_draws = a.draws if b != DIAG_B else min(a.draws, DIAG_DRAWS)
                rng_all = [np.random.default_rng([7919, key[0], key[1] * 10 + key[2], b, d]) for d in range(n_draws)]
                cal_sets = [hb.draw_calibration(b, rng_all[d], S.grasp, S.rep) for d in range(n_draws)]
                q_sets = [[i for i in range(n_tr) if i not in set(c)] for c in cal_sets]
                stats = [hb.input_stats(S.x[c], S.lens[c]) for c in cal_sets]
                xcal_z = [hb.zscore(S.x[c], *st, S.lens[c]) for c, st in zip(cal_sets, stats)]
                xq_cal = [hb.zscore(S.x[q], *st, S.lens[q]) for q, st in zip(q_sets, stats)]
                xq_src = [hb.zscore(S.x[q], mu_lat, sd_lat, S.lens[q]) for q in q_sets]
                common = {"target": tname, "subject": key[0], "day": key[1], "tod": key[2], "gap": key[1] - 2,
                          "B": b, "seed": a.seed_tag, "k": k, "table_bytes_fp16": table_bytes_fp16,
                          "h5_pass": table_bytes_fp16 <= 32 * 1024, "latest_source": list(keys[li])}

                def query_score(table, xq_z, q):
                    top = int(S.lens[q].max())
                    lg = hb.decode_trials(model, torch.from_numpy(xq_z[:, :top]).to(dev), table.to(dev)).cpu().numpy()
                    return hb.score_logits(lg, S.y[q][:, :top], S.lens[q], S.grasp[q])

                def emit(d, method, score, **extra):
                    row = {**common, "draw": d, "method": method, "cal_trials": cal_sets[d],
                           "coverage": hb.coverage(cal_sets[d], S.grasp), "n_query": len(q_sets[d]), **score, **extra}
                    row["h1_trunk_unchanged"] = hb.trunk_hash(model) == h1_ref
                    write(row)

                todo = lambda d, m: (tname, b, d, m) not in done  # noqa: E731
                if a.ridge_only:
                    for d in range(n_draws):
                        if todo(d, "rownorm_ridge_v2"):
                            q = q_sets[d]
                            rc = ridge_cfg[b]
                            w, bias = hb.ridge_fit(xcal_z[d], S.y[cal_sets[d]], S.lens[cal_sets[d]], rc["lam"], rc["ema"])
                            lg = hb.ridge_logits(xq_cal[d], w, bias, rc["ema"])
                            emit(d, "rownorm_ridge_v2", hb.score_logits(lg, S.y[q], S.lens[q], S.grasp[q]),
                                 h2_backprop_steps=0, role="deployable_closed_form", ridge_lam=rc["lam"],
                                 ridge_ema=rc["ema"], param_bytes_fp16=(N_CH * N_CLASS + 2 * N_CH + N_CLASS) * 2)
                    continue
                # ---- zero-backprop rows ----
                for d in range(n_draws):
                    q = q_sets[d]
                    cands = [("reuse_latest_srcnorm", tab_lat, xq_src[d]),
                             ("reuse_latest_calnorm", tab_lat, xq_cal[d]),
                             ("mean_table_calnorm", mean_all, xq_cal[d]),
                             ("mean_loso_calnorm", loso, xq_cal[d])]
                    for name, tab, xq in cands:
                        if todo(d, name):
                            emit(d, name, query_score(tab, xq, q), h2_backprop_steps=0, role="baseline")
                    if todo(d, "rownorm_ridge"):
                        rc = ridge_cfg[b]
                        w, bias = hb.ridge_fit(xcal_z[d], S.y[cal_sets[d]], S.lens[cal_sets[d]], rc["lam"], rc["ema"])
                        lg = hb.ridge_logits(xq_cal[d], w, bias, rc["ema"])
                        emit(d, "rownorm_ridge", hb.score_logits(lg, S.y[q], S.lens[q], S.grasp[q]),
                             h2_backprop_steps=0, role="deployable_closed_form", ridge_lam=rc["lam"],
                             ridge_ema=rc["ema"], param_bytes_fp16=(N_CH * N_CLASS + 2 * N_CH + N_CLASS) * 2)
                # ---- ui_prior (gradient through the frozen trunk; reference only) ----
                variants = [("ui_prior_latest_srcnorm", tab_lat, "src"), ("ui_prior_latest_calnorm", tab_lat, "cal"),
                            ("ui_prior_mean_calnorm", mean_all, "cal")]
                if any(todo(d, v[0]) for d in range(n_draws) for v in variants):
                    xs, ys, inits = [], [], []
                    for _, init, norm in variants:
                        for d in range(n_draws):
                            c = cal_sets[d]
                            xs.append(hb.zscore(S.x[c], mu_lat, sd_lat, S.lens[c]) if norm == "src" else xcal_z[d])
                            ys.append(S.y[c]); inits.append(init)
                    xs, ys = trim([np.stack(xs), np.stack(ys)], [S.lens[c] for c in cal_sets])
                    fit, loss = hb.fit_tables(model, torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                                              torch.stack(inits).to(dev), steps=UI_STEPS, lr=UI_LR, chunk=128)
                    for vi, (name, _, norm) in enumerate(variants):
                        for d in range(n_draws):
                            if todo(d, name):
                                xq = xq_src[d] if norm == "src" else xq_cal[d]
                                emit(d, name, query_score(fit[vi * n_draws + d], xq, q_sets[d]),
                                     h2_backprop_steps=UI_STEPS, role="reference_nondeployable",
                                     fit_loss=float(loss[vi * n_draws + d]))
                # ---- within-session diagnostics (B = DIAG_B only: half the session trains, half queries) ----
                if b == DIAG_B:
                    names = ("within_table_supinit", "within_table_latestinit_long")
                    if any(todo(d, n) for d in range(n_draws) for n in names):
                        xs, ys = trim([np.stack(xcal_z), np.stack([S.y[c] for c in cal_sets])],
                                      [S.lens[c] for c in cal_sets])
                        inits_sup = [sup_init_table(xcal_z[d], S.y[cal_sets[d]], S.lens[cal_sets[d]], k)
                                     for d in range(n_draws)]
                        for name, inits in zip(names, (inits_sup, [tab_lat] * n_draws)):
                            fit, loss = hb.fit_tables(model, torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                                                      torch.stack(inits).to(dev), steps=WITHIN_STEPS, lr=UI_LR, chunk=128)
                            for d in range(n_draws):
                                if todo(d, name):
                                    emit(d, name, query_score(fit[d], xq_cal[d], q_sets[d]),
                                         h2_backprop_steps=WITHIN_STEPS, role="diagnostic_within_session",
                                         fit_loss=float(loss[d]))
            print(f"target {tname} done", flush=True)


if __name__ == "__main__":
    main()
