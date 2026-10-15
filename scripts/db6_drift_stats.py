#!/usr/bin/env python
"""Input-level drift facts for DB6: spatial-pattern correlation and per-channel log-gain dispersion.

For a set of bins, pattern = per-channel log of the mean envelope (tau 80 ms cache).  Two views:
  overall   : all bins of the labeled trials (rest + grasps);
  per_grasp : per-grasp pattern (mean envelope over bins of that grasp), 7 x 14 flattened after
              subtracting each pattern's own overall mean (so only the spatial structure counts).
Pairs: within-session (odd vs even repetitions), same-day (T1 vs T2), across-day by day gap 1..4.
Reported: Pearson correlation of patterns (mean over subjects of the mean over pairs) and the std over
channels of the log gain ratio log(m_a / m_b) (also its mean), for three channel sets:
  all14  : all 14 electrodes;
  good12 : the 12 always-good electrodes (drops 0-based channels 7 and 15);
  no15   : 13 electrodes (drops only channel 15, which is live early for S1-S4 and then dies).
Channel facts (``channel_facts``): 0-based channel 15 is live (5-56 uV raw std) early for subjects 1-4 and
then dies (~0.1 uV); channel 7 sits at the noise floor (mostly 0.2-0.6 uV) and exceeds 1 uV in a few files.
S2_D2_T2 (truncated, one grasp) is excluded from the pair statistics.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import db6_data as d  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "analysis" / "db6_v2_20261016" / "drift_stats.json"
EXCL = {(2, 2, 2)}
NOISE_COLS = [7, 13]          # indices inside the 14-channel array of the two noise-floor columns (7 and 15)
assert d.ACTIVE_COLUMNS[NOISE_COLS[0]] == 7 and d.ACTIVE_COLUMNS[NOISE_COLS[1]] == 15
GOOD = [i for i in range(d.N_CH) if i not in NOISE_COLS]
NO15 = [i for i in range(d.N_CH) if i != NOISE_COLS[1]]


def channel_facts(keys):
    """Raw per-file std (uV) of the 0-based channels 7 and 15 from the cache metadata, plus liveness summary."""
    rows = {}
    for k in keys:
        sd = np.array(d.load_session(*k)["meta"]["channel_std_volt"]) * 1e6
        rows[f"S{k[0]}_D{k[1]}_T{k[2]}"] = {"ch7_uV": float(sd[7]), "ch15_uV": float(sd[15])}
    ch15_dead = {}
    for s in range(1, 11):
        ks = [k for k in keys if k[0] == s]
        ch15_dead[f"S{s}"] = [f"D{k[1]}T{k[2]}" for k in ks if rows[f"S{k[0]}_D{k[1]}_T{k[2]}"]["ch15_uV"] < 1.0]
    live = [r["ch15_uV"] for r in rows.values() if r["ch15_uV"] >= 1.0]
    return {"per_file": rows, "ch15_below_1uV_sessions": ch15_dead,
            "ch15_live_range_uV": [min(live), max(live)],
            "ch7_above_1uV_files": [n for n, r in rows.items() if r["ch7_uV"] > 1.0],
            "ch7_range_uV": [min(r["ch7_uV"] for r in rows.values()), max(r["ch7_uV"] for r in rows.values())]}


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
    for view, cols in (("good12", GOOD), ("no15", NO15), ("all14", list(range(d.N_CH)))):
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
    res["channel_facts"] = channel_facts([k for k in d.session_keys()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for view in ("good12", "no15", "all14"):
        for name, r in res[view].items():
            print(view, name, {m: round(v["mean"], 3) for m, v in r.items()})
    cf = res["channel_facts"]
    print({k: v for k, v in cf.items() if k != "per_file"})


if __name__ == "__main__":
    main()
