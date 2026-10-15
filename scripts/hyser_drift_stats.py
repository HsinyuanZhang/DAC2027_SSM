#!/usr/bin/env python
"""Cross-day input drift statistics of the Hyser 1dof envelopes (neural side only, no model)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode.hyser_data import load_session  # noqa: E402

rows = []
for s in range(1, 21):
    a, b = load_session(s, 1), load_session(s, 2)
    ma, mb = a["x"].reshape(-1, 256).mean(0), b["x"].reshape(-1, 256).mean(0)
    sa, sb = a["x"].reshape(-1, 256).std(0), b["x"].reshape(-1, 256).std(0)
    # within-day split-half reference: sample1+2 vs sample3 of session 1 (same electrode placement)
    h1 = a["x"][a["sample"] != 3].reshape(-1, 256).mean(0); h2 = a["x"][a["sample"] == 3].reshape(-1, 256).mean(0)
    lg = np.log(mb / ma)
    rows.append({"subject": s,
                 "corr_logmean_cross_day": float(np.corrcoef(np.log(ma), np.log(mb))[0, 1]),
                 "corr_logmean_within_day": float(np.corrcoef(np.log(h1), np.log(h2))[0, 1]),
                 "corr_logstd_cross_day": float(np.corrcoef(np.log(sa), np.log(sb))[0, 1]),
                 "gain_log_ratio_mean": float(lg.mean()), "gain_log_ratio_std": float(lg.std()),
                 "within_day_log_ratio_std": float(np.log(h2 / h1).std()),
                 "force_std_ratio": float(b["y"].reshape(-1, 5).std(0).mean() / a["y"].reshape(-1, 5).std(0).mean())})
summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "subject"}
out = ROOT / "results/analysis/hyser_20261015"
out.mkdir(parents=True, exist_ok=True)
(out / "drift_stats.json").write_text(json.dumps({"mean_over_subjects": summ, "per_subject": rows}, indent=1))
print(json.dumps(summ, indent=1))
