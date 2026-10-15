#!/usr/bin/env python
"""Input-level drift facts for DB6: spatial-pattern correlation and per-channel log-gain dispersion.

For a set of bins, pattern = per-channel log of the mean envelope (tau 80 ms cache).  Two views:
  overall   : all bins of the labeled trials (rest + grasps);
  per_grasp : per-grasp pattern (mean envelope over bins of that grasp), 7 x 14 flattened after
              subtracting each pattern's own overall mean (so only the spatial structure counts).
Pairs: within-session (odd vs even repetitions), same-day (T1 vs T2), across-day by day gap 1..4.
Reported: Pearson correlation of patterns (mean over subjects of the mean over pairs) and the std over
channels of the log gain ratio log(m_a / m_b) (also its mean).  Channels 7 and 15 sit at the noise
floor (<1 uV std) in every session; headline numbers use the 12 other channels, 'all14' keeps them.
S2_D2_T2 (truncated, one grasp) is excluded.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_data as d  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "analysis" / "db6_20261015" / "drift_stats.json"
EXCL = {(2, 2, 2)}
NOISE_COLS = [7, 13]          # indices inside the 14-channel array of the two noise-floor columns (7 and 15)
assert d.ACTIVE_COLUMNS[NOISE_COLS[0]] == 7 and d.ACTIVE_COLUMNS[NOISE_COLS[1]] == 15
GOOD = [i for i in range(d.N_CH) if i not in NOISE_COLS]


def patterns(blob, reps=None):
    sl = d.trial_slices(blob)
    pick = [i for i in range(len(sl)) if reps is None or int(blob["trial_rep"][i]) in reps]
    x = np.concatenate([blob["x"][sl[i]] for i in pick]).astype(np.float64)
    y = np.concatenate([blob["y"][sl[i]] for i in pick])
    overall = np.log(x.mean(0) + 1e-9)
    per = np.stack([np.log(x[y == g].mean(0) + 1e-9) for g in range(1, 8)])
    return overall, per


def pair_stats(pa, pb, cols):
    oa, ga = pa[0][cols], pa[1][:, cols]
    ob, gb = pb[0][cols], pb[1][:, cols]
    corr_o = float(np.corrcoef(oa, ob)[0, 1])
    fa, fb = ga - oa.mean() * 0 - ga.mean(), gb - gb.mean()
    corr_g = float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1])
    lr = oa - ob
    gr = (ga - gb)
    return {"corr_overall": corr_o, "corr_per_grasp": corr_g, "logratio_std": float(lr.std()),
            "logratio_mean": float(lr.mean()), "logratio_std_per_grasp": float(gr.std(1).mean())}


def main():
    keys = [k for k in d.session_keys() if k not in EXCL]
    blobs = {k: d.load_session(*k) for k in keys}
    pat = {k: patterns(b) for k, b in blobs.items()}
    odd = {k: patterns(b, reps={1, 3, 5, 7, 9, 11}) for k, b in blobs.items()}
    even = {k: patterns(b, reps={2, 4, 6, 8, 10, 12}) for k, b in blobs.items()}
    res = {}
    for view, cols in (("good12", GOOD), ("all14", list(range(d.N_CH)))):
        groups = {}
        for s in range(1, 11):
            ks = [k for k in keys if k[0] == s]
            g = {"within_session": [], "same_day": [], "gap1": [], "gap2": [], "gap3": [], "gap4": []}
            for k in ks:
                g["within_session"].append(pair_stats(odd[k], even[k], cols))
            for i, ka in enumerate(ks):
                for kb in ks[i + 1:]:
                    if ka[1] == kb[1]:
                        g["same_day"].append(pair_stats(pat[ka], pat[kb], cols))
                    else:
                        g[f"gap{abs(kb[1] - ka[1])}"].append(pair_stats(pat[ka], pat[kb], cols))
            for name, lst in g.items():
                groups.setdefault(name, []).append({m: float(np.mean([p[m] for p in lst])) for m in lst[0]})
        res[view] = {name: {m: {"mean": float(np.mean([r[m] for r in lst])),
                                "sd_over_subjects": float(np.std([r[m] for r in lst], ddof=1))}
                            for m in lst[0]} for name, lst in groups.items()}
        res[view + "_per_subject"] = groups
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for name, r in res["good12"].items():
        print(name, {m: round(v["mean"], 3) for m, v in r.items()})


if __name__ == "__main__":
    main()
