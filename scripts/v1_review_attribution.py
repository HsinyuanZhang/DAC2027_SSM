#!/usr/bin/env python
"""v1 review (c): in-context uncovered attribution R2(v) - R2(swap_cov) from the v1 rows, B>=2, across-target t-CI."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/hybrid_identity_20261016"
OUT = ROOT / "results/analysis/hybrid_identity_20261016"
rows = []
for f in sorted(glob.glob(str(RAW / "eval/rows_*.jsonl"))):
    rows += [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
df = pd.DataFrame(rows)
df = df[(df.variant != "h0")]
df["attr"] = df.r2 - df.r2_swap_cov          # in-context effect of v's uncovered columns (vs P0's), covered cols = v
p0 = pd.DataFrame(rows)
p0 = p0[p0.variant == "h0"].set_index(["target", "B", "draw", "seed"]).r2.rename("r2_p0")
df = df.join(p0, on=["target", "B", "draw", "seed"])
df["unc"] = df.r2_swap_unc - df.r2_p0
PAIRS = [("h2p", "h2p_loc"), ("h2p", "h1w153"), ("h2d", "h2d_loc"), ("h2d", "h1w138"), ("h3d", "h3d_loc"),
         ("h3d", "h1w113"), ("h3p", "h3p_loc"), ("h3p", "h1w130"), ("h1", "h0m"), ("h0m", None)]


def ci(x):
    x = np.asarray(x)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


out = []
for a, b in PAIRS:
    for metric in ("unc", "attr"):
        sa = df[df.variant == a].set_index(["target", "B", "draw", "seed"])[metric]
        if b is not None:
            sb = df[df.variant == b].set_index(["target", "B", "draw", "seed"])[metric]
            d = (sa - sb).dropna()
        else:
            d = sa
        d = d.rename(metric).reset_index()
        d = d[d.B >= 2]
        pt = d.groupby("target")[metric].mean()
        m, lo, hi = ci(pt.values)
        out.append({"a": a, "b": b or "P0", "metric": metric, "mean": m, "lo": lo, "hi": hi, "n_pos": int((pt > 0).sum()),
                    "n_targets": len(pt)})
res = pd.DataFrame(out)
res.to_csv(OUT / "attribution_in_context.csv", index=False)
print(res.to_string())
