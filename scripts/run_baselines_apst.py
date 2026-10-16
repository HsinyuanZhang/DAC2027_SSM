#!/usr/bin/env python
"""APST (Zhang et al. 2026) few-shot calibration on our (target, B, draw) subsets.

APST's session calibration is closed-form / forward-only: the activity encoder E0 pools
the B calibration trials' resampled neural activity, and (APST-FULL only) a MOVE-T4
association profile is fitted from the B trials' reach-direction labels.  No target
gradient is used.  We reuse the authors' code (``/home/xinyuan/Work_host/APST/src``,
``apst.dandi_subm``) unchanged for the model, checkpoint loading, carrier estimator,
source-statistics normaliser and output EMA; only the calibration trial *selection*
(random B-subsets of our draws instead of the first M trials) and the *query set*
(our contiguous query bins) differ.

Run with PYTHONPATH containing /home/xinyuan/Work_host/APST/src (and the SSM root).
Never touches FINAL sessions or results/official_evalai/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apst.dandi_subm import protocol as P                                   # noqa: E402
from apst.dandi_subm import training as T                                   # noqa: E402
from apst.dandi_subm.carrier import apply_carrier_normalizer, estimate_move_t4  # noqa: E402
from apst.dandi_subm.subm_data import SessionData as ApstSession, _resample_trial  # noqa: E402
from scripts.run_population_calib import draw_trials                        # noqa: E402
from ssm_decode.dandi688_data import (FINAL_SESSIONS, _scan_session,        # noqa: E402
                                      dandi688_subm_root, load_dandi688_recording)
from ssm_decode.population_calib import SessionData                         # noqa: E402
from ssm_decode.eval_checks import check_physical_units                     # noqa: E402

OUT = Path(os.environ.get("BL_OUT", str(ROOT / "results/baselines_20261016")))
RUN_ROOT = Path("/mnt/data/work_host/SPINT/apst_final_week_20260916/dandi688_subm_2015_20260921/runs")
B_GRID = (1, 4, 8, 16, 33)
N_DRAWS = 8
LEAD = 16                      # EMA warm-up bins before every query run (decay (1/3)^16 ~ 2e-8)
RUN_ROOT14 = ROOT / "results/baselines_20261016/apst2014/runs"
CHECKPOINTS = {                # label -> (run dir, segment, model method)
    "apst_full_last": ("sua_dual_site_F0_s42", "segment_24.pt", "dual_site"),
    "apst_full_devsel": ("sua_dual_site_F0_s42", "segment_09.pt", "dual_site"),
    "apst_act_last": ("sua_act_only_F0_s42", "segment_24.pt", "act_only"),
    "apst_act_devsel": ("sua_act_only_F0_s42", "segment_08.pt", "act_only"),
    # 2014-roster retrain (scripts/apst_2014_train.py) for month / year
    "apst14_full_last": ("sua_dual_site_F0_s42", "segment_24.pt", "dual_site"),
    "apst14_full_sel0307": ("sua_dual_site_F0_s42", "SELECTION", "dual_site"),
    "apst14_act_last": ("sua_act_only_F0_s42", "segment_24.pt", "act_only"),
}
TARGETS = {"days": ["sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617"],
           "month": ["sub-M_ses-CO-20140626"], "year": ["sub-M_ses-CO-20150511"]}


def ckpt_path(model):
    run, seg, method = CHECKPOINTS[model]
    root = RUN_ROOT14 if model.startswith("apst14") else RUN_ROOT
    if seg == "SELECTION":                                   # source-OOS (20140307) earliest-max segment
        seg = Path(json.loads((root / run / "selection.json").read_text())["checkpoint"]).name
    return root / run / seg, f"{run}/{seg}", method


def trial_table(name, record):
    """Per trials-table row: bins, MOVE window, direction, result (APST legality inputs)."""
    from pynwb import NWBHDF5IO
    path = dandi688_subm_root() / f"{name}_behavior+ecephys.nwb"
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        df = io.read().intervals["trials"].to_dataframe()
    edges = np.asarray(record.bin_edges)
    n_bins = len(edges) - 1
    rows = {}
    for _, row in df.iterrows():
        a = max(0, int(np.searchsorted(edges, float(row.start_time))))
        b = min(n_bins, int(np.searchsorted(edges, float(row.stop_time))))
        if b <= a:
            continue
        go = row.get("go_cue_time")
        ang = row.get("target_dir")
        rows[(a, b)] = {"start_time": float(row.start_time), "stop_time": float(row.stop_time),
                        "go": float(go) if go is not None and np.isfinite(go) else None,
                        "angle": float(ang) if ang is not None and np.isfinite(ang) else float("nan"),
                        "result": str(row.get("result"))}
    return rows


class Target:
    def __init__(self, name, device):
        self.name = name
        assert name not in FINAL_SESSIONS
        self.record = load_dandi688_recording(name)
        self.scan = _scan_session(dandi688_subm_root() / f"{name}_behavior+ecephys.nwb")
        self.table = trial_table(name, self.record)
        self.sd = SessionData(self.record, (np.zeros(2), np.ones(2)), device, name=name)
        self.edges = np.asarray(self.record.bin_edges)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2.0
        self.n_units = self.record.neural.shape[1]

    def calib_trial(self, idx_i):
        """APST calibration tensors of one of our usable trials, or None if APST-illegal."""
        a, b = self.sd.trials[idx_i]
        t = self.table[(a, b)]
        if t["result"] != "R" or b - a < P.WINDOW_BINS or t["go"] is None:
            return None
        left, right = t["go"] + P.MOVE_START_AFTER_GO_SECONDS, t["go"] + P.MOVE_STOP_AFTER_GO_SECONDS
        if not (t["start_time"] <= left < right <= t["stop_time"] and self.edges[0] <= left < right <= self.edges[-1]):
            return None
        counts = np.asarray([np.searchsorted(s, right, side="left") - np.searchsorted(s, left, side="left")
                             for s in self.scan["spikes"]], dtype=np.float32)
        act = _resample_trial(self.record.neural[a:b].astype(np.float32))
        return act, counts, t["angle"]


def calibration_inputs(tg, idx, method, stats, n_pad):
    items = [tg.calib_trial(i) for i in idx]
    legal = [it for it in items if it is not None]
    info = {"B": len(idx), "B_legal": len(legal)}
    if not legal:
        return None, None, {**info, "status": "NO_LEGAL_TRIAL"}
    activity = np.zeros((len(legal), 100, n_pad), np.float32)
    activity[:, :, :tg.n_units] = np.stack([it[0] for it in legal])
    if method == "act_only":
        return activity, None, {**info, "status": "OK", "rank": None}
    counts = np.stack([it[1] for it in legal])
    angles = np.asarray([it[2] for it in legal], np.float64)
    finite = np.isfinite(angles)
    design = np.stack((np.ones(finite.sum()), np.cos(angles[finite]), np.sin(angles[finite])), axis=1)
    rank = int(np.linalg.matrix_rank(design)) if finite.sum() else 0
    if finite.sum() < 3 or rank < 3:
        return activity, None, {**info, "status": "UNAVAILABLE_RANK_LT3", "rank": rank}
    raw = estimate_move_t4(counts, angles)
    norm = apply_carrier_normalizer(raw, stats["carrier"])
    carrier = np.zeros((n_pad, 4), np.float32)
    carrier[:tg.n_units] = norm
    return activity, carrier, {**info, "status": "OK", "rank": rank}


def native_mask(tg, qend):
    """APST's own query rule: the 50-bin causal window lies fully inside one trial (>= 49 in-trial bins before)."""
    start = np.full(tg.record.neural.shape[0], -10 ** 9, dtype=np.int64)
    for a, b in tg.sd.trials:
        start[a:b] = a
    return (qend - start[qend]) >= P.WINDOW_BINS - 1


def query_endpoints(tg, idx):
    q = tg.sd.query_mask(idx).cpu().numpy()
    q[: P.WINDOW_BINS - 1] = False                      # APST windows need 50 bins of history
    return np.flatnonzero(q)


def predict(model, tg, stats, method, activity, carrier, qend, batch=256):
    """Mirror of ``training.predict_record`` (intact) restricted to the query run + EMA lead."""
    device = next(model.parameters()).device
    n_pad = T._units(model)
    model.eval()
    need = np.zeros(tg.record.neural.shape[0], bool)
    need[qend] = True
    lead = np.zeros_like(need)
    for k in range(1, LEAD + 1):
        lead[: len(lead) - k] |= need[k:]
    endpoints = np.flatnonzero(need | lead)
    endpoints = endpoints[endpoints >= 0]
    act_t = torch.from_numpy(activity).to(device)
    prof = None if carrier is None else torch.from_numpy(carrier).to(device)
    with torch.inference_mode():
        base_e0 = model.encode(act_t, None if method in {"act_only", "token_only"} else prof)
    preds = []
    rec = SimpleRecord(tg.record.neural)
    with torch.inference_mode():
        for s in range(0, len(endpoints), batch):
            ep = endpoints[s:s + batch]
            x, valid, unitmask = T._full_clock_windows(rec, ep, n_pad)
            tx, tv, um = (torch.from_numpy(x).to(device), torch.from_numpy(valid).to(device),
                          torch.from_numpy(unitmask).to(device))
            token = prof if method in {"token_only", "dual_site"} else None
            value = model.forward_identity(tx, base_e0, token, unit_mask=um, dropout_keep=None, input_valid_mask=tv)
            preds.append(value.float().cpu().numpy())
    raw = np.concatenate(preds).astype(np.float32)
    physical = raw * np.asarray(stats["velocity_std"], np.float32) + np.asarray(stats["velocity_mean"], np.float32)
    # EMA over the sorted endpoint list; every query run is preceded by >= LEAD contiguous bins
    filtered = np.empty_like(physical, dtype=np.float64)
    prev = None
    last_ep = None
    for i, ep in enumerate(endpoints):
        row = physical[i].astype(np.float64)
        if prev is None or ep != last_ep + 1:
            prev = row                                   # fresh run (state_0 = pred_0)
        else:
            prev = (1 / 3) * prev + (2 / 3) * row
        filtered[i] = prev
        last_ep = ep
    pos = np.searchsorted(endpoints, qend)
    return filtered[pos]


class SimpleRecord:
    def __init__(self, neural):
        self.neural = neural


def r2_vw(truth, pred):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def cmd_eval(args):
    device = torch.device(args.device)
    path, tag, method = ckpt_path(args.model)
    model, stats, _ = T.load_checkpoint(path, device=str(device))
    n_pad = T._units(model)
    out = OUT / "eval" / f"apst_{args.stratum}_{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["target"], r["B"], r["draw"]))
    for name in TARGETS[args.stratum]:
        tg = Target(name, device)
        truth_all = tg.record.behavior.astype(np.float64)
        for B in B_GRID:
            for draw in range(N_DRAWS):
                if (name, B, draw) in seen:
                    continue
                idx = draw_trials(name, B, draw, tg.sd.bank.n_trials)
                act, car, info = calibration_inputs(tg, idx, method, stats, n_pad)
                row = {"stratum": args.stratum, "method": args.model.replace("apst14_", "apst_"), "target": name,
                       "B": B, "draw": draw, "checkpoint": tag, "info": info}
                if act is None or (method == "dual_site" and car is None):
                    row.update(r2=None, status=info["status"])
                else:
                    t0 = time.perf_counter()
                    qend = query_endpoints(tg, idx)
                    pred = predict(model, tg, stats, method, act, car, qend, batch=args.batch)
                    truth = truth_all[qend]
                    unit = check_physical_units(pred, truth)
                    nat = native_mask(tg, qend)
                    row.update(r2=r2_vw(truth, pred), r2_native=r2_vw(truth[nat], pred[nat]),
                               n_query_native=int(nat.sum()), status="OK", n_query=int(len(qend)),
                               unit_ratio=unit["unit_ratio"], unit_check_passed=unit["passed"],
                               seconds=round(time.perf_counter() - t0, 2))
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in ("method", "target", "B", "draw", "r2", "status")}), flush=True)


def cmd_parity(args):
    """Parity of the restricted predictor with ``training.predict_record`` on a dev session prefix."""
    device = torch.device(args.device)
    run, seg, method = CHECKPOINTS[args.model]
    model, stats, _ = T.load_checkpoint(RUN_ROOT / run / seg, device=str(device))
    n_pad = T._units(model)
    name = TARGETS["days"][0]
    tg = Target(name, device)
    idx = draw_trials(name, 8, 0, tg.sd.bank.n_trials)
    act, car, info = calibration_inputs(tg, idx, method, stats, n_pad)
    print(info)
    qend_full = query_endpoints(tg, idx)
    qend = qend_full[qend_full < 6000]
    mine = predict(model, tg, stats, method, act, car, qend, batch=args.batch)
    # reference path: authors' predict_record on the 6000-bin prefix, calibration patched in
    from apst.dandi_subm.carrier import CARRIER_DIM  # noqa: F401
    import contextlib
    old_a, old_c = T._activity, T._carrier
    T._activity = lambda record, device, units: torch.from_numpy(act).to(device)
    T._carrier = lambda record, stats_, device, units: None if car is None else torch.from_numpy(car).to(device)
    n_prefix = 6000
    fake = ApstSession(name, "dev", "sua", tg.record.neural[:n_prefix], tg.record.behavior[:n_prefix],
                       tg.edges[:n_prefix + 1], qend, np.arange(10), np.arange(5), np.zeros((1, 100, tg.n_units), np.float32),
                       np.zeros((1, tg.n_units), np.float32), np.zeros(1), np.arange(tg.n_units),
                       tg.record.unit_to_electrode, tg.record.observed_mask,
                       {"array_sha256": {"query_indices": "x", "velocity": "x"}})
    try:
        ref, _ = T.predict_record(model, fake, stats, condition="intact", batch=32)
    finally:
        T._activity, T._carrier = old_a, old_c
    print("max abs diff", float(np.abs(ref - mine).max()), "n", len(qend), "scale", float(np.abs(ref).mean()))


def cmd_firstm(args):
    """Reproduce the authors' own first-M dev number with this adaptor (end-to-end check)."""
    from apst.dandi_subm.subm_data import load_cached_session
    device = torch.device(args.device)
    run, seg, method = CHECKPOINTS[args.model]
    model, stats, _ = T.load_checkpoint(RUN_ROOT / run / seg, device=str(device))
    n_pad = T._units(model)
    name = TARGETS["days"][0]
    cache = load_cached_session(RUN_ROOT.parent / "prepared_sua" / f"{name}.sua.npz")
    tg = Target(name, device)
    for B in (8, 33):
        legal = [i for i, (a, b) in enumerate(tg.sd.trials)
                 if tg.table[(a, b)]["result"] == "R" and b - a >= P.WINDOW_BINS][:B]
        act, car, info = calibration_inputs(tg, np.asarray(legal), method, stats, n_pad)
        pred = predict(model, tg, stats, method, act, car, cache.query_indices, batch=args.batch)
        truth = cache.velocity[cache.query_indices].astype(np.float64)
        print(B, info, "r2", r2_vw(truth, pred), "act parity", float(np.abs(act[:, :, :tg.n_units] - cache.activity[:B]).max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["eval", "parity", "firstm"])
    p.add_argument("--stratum", default="days")
    p.add_argument("--model", required=True, choices=list(CHECKPOINTS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    {"eval": cmd_eval, "parity": cmd_parity, "firstm": cmd_firstm}[args.cmd](args)


if __name__ == "__main__":
    main()
