"""Select the envelope time constant (and ridge lambda) on SOURCE validation only.

Per source subject (session1): fit z-score + ridge 256 -> 5 force on the two
training trials per finger, score on the held-out sample-3 trial per finger.
Only session1 is read; session2 (target) is never touched.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import hyser_data as h  # noqa: E402

TAUS = [20.0, 40.0, 80.0, 160.0, 320.0]
LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def r2_macro(y, p):
    return float(np.mean(1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def one(subject):
    folder = h.session_dir(subject, 1)
    ys, feats = [], {t: [] for t in TAUS}
    for finger, sample in h.trial_names_1dof():
        stem = folder / f"1dof_preprocess_finger{finger}_sample{sample}"
        phys, _ = h.read_record(stem)
        for t in TAUS:
            feats[t].append(h.decimate_emg(h.envelope(phys * 1e3, t)))
        ys.append(h.force_features(folder / f"1dof_force_finger{finger}_sample{sample}"))
    val = np.array([s == 3 for _, s in h.trial_names_1dof()])
    out = {}
    for t in TAUS:
        x = np.stack(feats[t]); y = np.stack(ys)
        xtr = x[~val].reshape(-1, 256).astype(np.float64); ytr = y[~val].reshape(-1, 5).astype(np.float64)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        xz = (xtr - mu) / sd
        ym = ytr.mean(0)
        a = xz.T @ xz; b = xz.T @ (ytr - ym)
        xv = ((x[val].reshape(-1, 256) - mu) / sd); yv = y[val].reshape(-1, 5)
        for lam in LAMS:
            w = np.linalg.solve(a + lam * len(xz) / 1000 * np.eye(256), b)
            out[f"{t}|{lam}"] = r2_macro(yv, xv @ w + ym)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as pool:
        res = list(pool.map(one, range(1, 21)))
    keys = res[0].keys()
    table = {k: float(np.mean([r[k] for r in res])) for k in keys}
    best = max(table, key=table.get)
    out = {"scores_mean_over_20_source_sessions": table, "best": best,
           "note": "lambda scaled as lam*N/1000; source validation only"}
    p = ROOT / "results/hyser_20261015/tau_select.json"
    p.write_text(json.dumps(out, indent=1))
    for k, v in sorted(table.items(), key=lambda kv: -kv[1])[:10]:
        print(k, round(v, 4))
    print("best", best)
