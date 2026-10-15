#!/usr/bin/env python
"""A'4: analytic architecture-level cost tables (CPU only, no GPU).

Review fix (2026-10-16): SRAM now includes the per-trial q_* scratch, the lag delay
lines, exact bin counters at their own width, the two-level staging copy, and the P4
second-pass phi buffer (alternative: recompute phi, extra MACs); the fixed-weight share
excludes activation x activation products.  Per-group accumulator widths are the minimal
safe widths of the per-add experiment (``a3_safe_width.csv``) when that file exists,
otherwise 16 bit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANA = ROOT / "results/analysis/population_calib_p2_20261015"

from ssm_decode import calib_cost as cc                                    # noqa: E402
from ssm_decode.calib_quant import counter_bits                            # noqa: E402

B_MAX = 33
STAGE_BITS = 32                  # assumed width of the wide per-trial staging copy (two-level)


def trial_stats():
    p = ANA / "cost_trial_stats.json"
    if p.exists():                                  # data are not needed again
        return json.loads(p.read_text())
    from ssm_decode.dandi688_data import load_dandi688_recording
    from ssm_decode.track_a_experiment import usable_trials
    out = {}
    for s in ("sub-M_ses-CO-20150616", "sub-M_ses-CO-20140626"):
        rec = load_dandi688_recording(s)
        tr = usable_trials(rec)
        out[s] = {"n_units": int(rec.neural.shape[1]), "n_trials": len(tr),
                  "mean_bins": float(np.mean([b - a for a, b in tr])),
                  "max_bins": int(max(b - a for a, b in tr))}
    return out


def safe_widths(default=16):
    """(uniform width for all data keys, {group: minimal safe width}) from the A'3 rerun."""
    p = ANA / "a3_safe_width.csv"
    if not p.exists():
        return default, {g: default for g in ("norm", "ac", "cross", "dir", "quad")}
    sw = pd.read_csv(p)
    safe = sw[sw.safe]
    uni = int(safe[safe.group == "all"].bits.min()) if (safe.group == "all").any() else default
    per = {g: int(safe[safe.group == g].bits.min()) if (safe.group == g).any() else uni
           for g in ("norm", "ac", "cross", "dir", "quad")}
    return uni, per


def main():
    ts = trial_stats()
    print(json.dumps(ts, indent=1))
    bins688 = int(round(np.mean([v["mean_bins"] for v in ts.values()])))
    lmax688 = max(v["max_bins"] for v in ts.values())
    uni, per_group = safe_widths()
    cases = {
        "000688 (N=26, K=10, D=2)": dict(n=26, k=10, d=2, trial_bins=bins688, lmax=lmax688,
                                         x_bits=8, y_bits=16),
        "HD-sEMG (N=256, K=16, D=5)": dict(n=256, k=16, d=5, trial_bins=1250, lmax=1250,
                                           x_bits=16, y_bits=16),
    }
    # hardware options: name -> (acc_bits, desc_bits, table_bits, stage_bits)
    options = {
        "fp32": dict(acc_bits=32, desc_bits=32, table_bits=32, cnt_bits=None, stage_bits=None),
        "perAdd_uniform": dict(acc_bits=uni, desc_bits=16, table_bits=8, derived_dir=True),
        "perAdd_perGroup": dict(acc_bits=dict(per_group), desc_bits=16, table_bits=8, derived_dir=True),
        "twoLevel_uniform": dict(acc_bits=uni, desc_bits=16, table_bits=8, stage_bits=STAGE_BITS,
                                 derived_dir=True),
    }
    # HD-sEMG features are continuous: per-add rounding needs >= 28 bit (pc2_hyser_peradd.py);
    # alternative: int8-coded inputs accumulate exactly in 21 bit (gram), same register layout
    hd_extra = {
        "perAdd_cont28": dict(acc_bits=28, desc_bits=16, table_bits=8, derived_dir=True),
        "perAdd_int8code21": dict(acc_bits=21, desc_bits=16, table_bits=8, derived_dir=True, x_bits=8),
    }
    rows = []
    for name, c in cases.items():
        cnt = counter_bits(B_MAX, c["lmax"])
        for variant in ("p0", "p2", "p4"):
            for b in (1, 8, 33):
                for opt, kw in {**options, **(hd_extra if name.startswith("HD") else {})}.items():
                    kw = dict(kw)
                    xb = kw.pop("x_bits", c["x_bits"])
                    kw.setdefault("cnt_bits", cnt)
                    kw.setdefault("stage_bits", None)
                    r = cc.full_report(c["n"], c["k"], c["d"], variant, b, c["trial_bins"],
                                       x_bits=xb, y_bits=c["y_bits"], **kw)
                    acc_bits = kw["acc_bits"]
                    r.update(case=name, option=opt, counter_bits=kw["cnt_bits"],
                             acc_bits=json.dumps(acc_bits) if isinstance(acc_bits, dict) else acc_bits,
                             table_bits=kw["table_bits"], stage_bits=kw["stage_bits"] or 0,
                             max_trial_bins=c["lmax"])
                    rows.append(r)
    df = pd.DataFrame(rows)
    ANA.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANA / "cost_model.csv", index=False)
    (ANA / "cost_trial_stats.json").write_text(json.dumps(ts, indent=1))
    print("uniform width", uni, "per group", per_group)
    write_markdown(df)


def write_markdown(df):
    kb = lambda v: f"{v / 1024:.1f}"
    md = []
    # --- MAC table (B = 8, fp32 reference arithmetic count) ---------------------------------
    md.append("### MAC / 一次校准 (B=8)\n")
    hdr = ["quantity"]
    sel = []
    for case in df.case.unique():
        for v in ("p0", "p4"):
            hdr.append(f"{case.split(' ')[0]} {v.upper()}")
            sel.append(df[(df.case == case) & (df.variant == v) & (df.B == 8) & (df.option == "fp32")].iloc[0])
    rows = [("acc MAC / valid bin (full)", lambda r: f"{r.acc_mac_per_bin:,.0f}"),
            ("acc MAC / trial", lambda r: f"{r.acc_mac_per_bin * r.trial_bins + r.acc_mac_per_trial_quad:,.0f}"),
            ("finalize MAC", lambda r: f"{r.finalize_mac:,.0f}"),
            ("calibrator forward MAC (all units)", lambda r: f"{r.calib_forward_mac:,.0f}"),
            ("  of which activation x activation", lambda r: f"{r.actact_macs:,.0f}"),
            ("  of which fixed-weight (RRAM-mappable)", lambda r: f"{r.fixed_weight_macs:,.0f}"),
            ("total MAC, one calibration (phi buffered)", lambda r: f"{r.total_mac_one_calibration:,.0f}"),
            ("total MAC, phi recomputed in pass 2", lambda r: f"{r.total_mac_one_calibration_phi_recompute:,.0f}"),
            ("fixed-weight share of all MAC (corrected)", lambda r: f"{100 * r.fixed_weight_fraction:.2f}%"),
            ("old share (forward incl. act x act)", lambda r: f"{100 * r.forward_share_incl_actact:.2f}%")]
    md.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for name, f in rows:
        md.append("| " + name + " | " + " | ".join(f(r) for r in sel) + " |\n")
    # --- SRAM tables --------------------------------------------------------------------------
    for v in ("p0", "p4"):
        md.append(f"\n### SRAM (KiB), {v.upper()}, B=8; 表 int8 双缓冲, 描述量 16 bit\n")
        cols = ["accumulators (data+counters)", "of which counters (B)", "q_* trial scratch", "lag delay lines",
                "two-level staging", "descriptors", "table x2", "pool state", "TOTAL", "TOTAL + P4 phi buffer"]
        md.append("| case | option | acc width | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 3) + "\n")
        for _, r in df[(df.variant == v) & (df.B == 8)].iterrows():
            md.append(f"| {r.case.split(' (')[0]} | {r.option} | {r.acc_bits} | {kb(r.acc_bytes)} | {r.counter_bytes:.0f} | "
                      f"{kb(r.scratch_bytes)} | {kb(r.delay_line_bytes)} | {kb(r.staging_bytes)} | {kb(r.descriptor_bytes)} | "
                      f"{kb(r.table_bytes_double)} | {kb(r.pool_state_bytes)} | {kb(r.total_sram_bytes)} | "
                      f"{kb(r.total_sram_bytes_with_phi_buffer) if v == 'p4' else '-'} |\n")
    (ANA / "cost_tables.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
