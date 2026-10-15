#!/usr/bin/env python
"""Aggregate Hyser H-A rows: mean over draws and seeds per target, then mean and 95% CI over 20 targets."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/hyser_20261015"
ROLE_ORDER = ["reuse_s1_srcnorm", "reuse_s1_calnorm", "mean_table_calnorm", "mean_loso_calnorm",
              "rownorm_ridge", "rownorm_ridge_perB", "ui_prior_s1_srcnorm", "ui_prior_s1_calnorm", "ui_prior_mean_calnorm",
              "within_table_s1init_long", "within_table_supinit",
              "reuse_s1_oracle_norm", "mean_table_oracle_norm"]


def ci95(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))


def main(dirs, tag="main"):
    rows = []
    for d in dirs:
        for fname in ("rows.jsonl", "rows_oracle_norm.jsonl"):
            if not (Path(d) / fname).exists():
                continue
            for ln in (Path(d) / fname).read_text().splitlines():
                r = json.loads(ln); r["_dir"] = str(d); rows.append(r)
    # per (method, B, target): mean over draws and seeds
    cell = {}
    for r in rows:
        cell.setdefault((r["method"], r["B"], r["target"]), []).append(r)
    table = {}
    for (m, b, t), rs in cell.items():
        table.setdefault((m, b), {})[t] = {
            "r2": np.mean([x["r2"] for x in rs]), "r2_pooled": np.mean([x["r2_pooled"] for x in rs]),
            "unit_ratio": np.mean([x["unit_ratio"] for x in rs]),
            "draw_std": float(np.std([x["r2"] for x in rs])), "n": len(rs)}
    out_rows, js = [], {}
    for m in ROLE_ORDER:
        for b in (1, 2, 5, 10):
            if (m, b) not in table:
                continue
            ts = sorted(table[(m, b)])
            v = np.array([table[(m, b)][t]["r2"] for t in ts])
            vp = np.array([table[(m, b)][t]["r2_pooled"] for t in ts])
            rec = {"method": m, "B": b, "n_targets": len(ts), "r2_mean": float(v.mean()), "r2_ci95": ci95(v),
                   "r2_pooled_mean": float(vp.mean()), "r2_pooled_ci95": ci95(vp),
                   "r2_median": float(np.median(v)), "unit_ratio_mean": float(np.mean([table[(m, b)][t]["unit_ratio"] for t in ts])),
                   "mean_draw_std": float(np.mean([table[(m, b)][t]["draw_std"] for t in ts])),
                   "rows_per_target": float(np.mean([table[(m, b)][t]["n"] for t in ts]))}
            out_rows.append(rec)
            js[f"{m}|B{b}"] = {**rec, "per_target_r2": {str(t): float(table[(m, b)][t]["r2"]) for t in ts}}
    # seen / unseen finger breakdown (finger channel c is "seen" if a calibration trial pressed finger c)
    su = {}
    for r in rows:
        if "cal_trials" not in r:
            continue
        seen = sorted({i // 3 for i in r["cal_trials"]})
        per = np.asarray(r["r2_per_channel"])
        uns = [c for c in range(5) if c not in seen]
        su.setdefault((r["method"], r["B"], r["target"]), []).append(
            (float(per[seen].mean()), float(per[uns].mean()) if uns else float("nan")))
    seen_rows = []
    for m in ROLE_ORDER:
        for b in (1, 2, 5, 10):
            keys = [k for k in su if k[0] == m and k[1] == b]
            if not keys:
                continue
            sv = np.array([np.mean([x[0] for x in su[k]]) for k in keys])
            uv = np.array([np.nanmean([x[1] for x in su[k]]) if b < 5 else np.nan for k in keys])
            seen_rows.append({"method": m, "B": b, "r2_seen_fingers": float(sv.mean()),
                              "r2_unseen_fingers": float(np.nanmean(uv)) if b < 5 else float("nan")})
    # paired differences against the reference rows
    pairs = []
    for b in (1, 2, 5, 10):
        for a_, b_ in [("ui_prior_s1_srcnorm", "reuse_s1_srcnorm"), ("ui_prior_s1_calnorm", "reuse_s1_calnorm"),
                       ("ui_prior_mean_calnorm", "mean_table_calnorm"), ("ui_prior_s1_calnorm", "mean_table_calnorm"),
                       ("reuse_s1_calnorm", "mean_table_calnorm"), ("reuse_s1_srcnorm", "reuse_s1_calnorm"),
                       ("rownorm_ridge", "reuse_s1_calnorm"), ("ui_prior_s1_calnorm", "rownorm_ridge"),
                       ("rownorm_ridge_perB", "reuse_s1_calnorm"), ("ui_prior_s1_calnorm", "rownorm_ridge_perB"),
                       ("within_table_s1init_long", "ui_prior_s1_calnorm"),
                       ("reuse_s1_oracle_norm", "reuse_s1_calnorm"), ("ui_prior_s1_calnorm", "reuse_s1_oracle_norm"),
                       ("ui_prior_s1_calnorm", "ui_prior_mean_calnorm")]:
            if (a_, b) in table and (b_, b) in table:
                ts = sorted(set(table[(a_, b)]) & set(table[(b_, b)]))
                dlt = np.array([table[(a_, b)][t]["r2"] - table[(b_, b)][t]["r2"] for t in ts])
                pairs.append({"B": b, "a": a_, "b": b_, "diff_mean": float(dlt.mean()), "diff_ci95": ci95(dlt),
                              "n_targets": len(ts), "frac_a_better": float((dlt > 0).mean())})
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with (OUT / f"reference_rows_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    with (OUT / f"paired_diffs_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    with (OUT / f"seen_unseen_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seen_rows[0].keys())); w.writeheader(); w.writerows(seen_rows)
    (OUT / f"reference_rows_{tag}.json").write_text(json.dumps(js, indent=1))
    # H flags
    h = {"h1_all_true": all(r.get("h1_trunk_unchanged", False) for r in rows),
         "h5_all_pass": all(r.get("h5_pass", False) for r in rows), "n_rows": len(rows),
         "backprop_steps_by_method": {m: sorted({r["h2_backprop_steps"] for r in rows if r["method"] == m})
                                      for m in ROLE_ORDER if any(r["method"] == m for r in rows)}}
    (OUT / f"h_flags_{tag}.json").write_text(json.dumps(h, indent=1))
    for rec in out_rows:
        print(f'{rec["method"]:28s} B={rec["B"]:2d} R2={rec["r2_mean"]:.3f}±{rec["r2_ci95"]:.3f} pooled={rec["r2_pooled_mean"]:.3f} ratio={rec["unit_ratio_mean"]:.2f}')
    for p in pairs:
        print(p)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
