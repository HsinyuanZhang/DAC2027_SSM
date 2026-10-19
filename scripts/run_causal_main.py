#!/usr/bin/env python
"""Causal-first-B main table (DAC2027, 2026-10-19).

Calibration = the first B usable trials. The whole session is decoded as one
contiguous causal pass. Tables switch only at the boundaries in
``ssm_decode.causal_protocol``. Candidates do not backprop. Source selections
that were fit on random draws are marked ``random_draw_legacy`` unless a
causal-first-B selection file is present.

  python scripts/run_causal_main.py select --tag month --what s1
  python scripts/run_causal_main.py select --tag month --what combo
  python scripts/run_causal_main.py eval --stratum drift
  python scripts/run_causal_main.py eval --stratum subc
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PC_OUT"] = str(ROOT / "results/population_calib_p3_20261016/fix")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_causal")

import torch                                                    # noqa: E402

from scripts import run_pc3_refine as pc                        # noqa: E402
from scripts import run_population_calib as rpc                 # noqa: E402
from scripts.run_drift_curve import (FINAL_DATES, PREFIX, TARGET_DATES,  # noqa: E402
                                     pick_kind, validate_targets)
from scripts.run_labeled_calibration import D1_POOLS            # noqa: E402
from scripts.run_p0solve import (CCA_CFG, UI_RULE, cca_in_w0_coordinates,  # noqa: E402
                                 score_tables)
from scripts.run_p0solve_adapt import alpha_for_target, load_gap_rules  # noqa: E402
from ssm_decode import fwd_refine as fr                         # noqa: E402
from ssm_decode import stream_f7 as sf                          # noqa: E402
from ssm_decode.causal_protocol import (COLD, INITIAL, AffineMap,  # noqa: E402
                                        assert_segment_causal, calibration_indices,
                                        commit_segments, margin_pick, query_bins_on_commit,
                                        run_rebased_many)
from ssm_decode.population_calib import PopStats, norm_stats    # noqa: E402
from ssm_decode.p0solve import shrink_tables                    # noqa: E402

OUT = ROOT / "results/causal_main_20261019"
P3_FIX = ROOT / "results/population_calib_p3_20261016/fix"
STREAM = ROOT / "results/stream_f7_20261019"
EXP_OUT = ROOT / "results/expansion_round1/x/expansion_out"
B_LIST = (1, 2, 4, 8, 16, 33)
SUBC_B = (1, 4, 8, 16, 33)
SEEDS = (0, 1)
LATENCIES = (0, 1)
SIM_MAX = pc.SIM_MAX_BINS
ETAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
RHOS = (0.0, 0.01, 0.03, 0.1, 0.3)
ANYTIME = ("p_prior", "p0", "w_alpha_gap", "s1_causal", "s1_causal_and",
           "combo", "combo_and", "r1_stream")
REFERENCES = ("r2", "r3", "ridge_cca", "p0_ui_reg")
SEED_FREE = {"p_prior", "ridge_cca"}
DRIFT_DATES = list(TARGET_DATES)

# Stratum -> frozen trunk, gap-rule file, P3 selection, S1 tag, CCA config key.
SPECS = {
    "days": {"world": "days", "gap": "days", "p3": ("days", "oos"), "s1": "days",
             "cca": "days", "ui": "days", "dates": ["20150616", "20150617"]},
    "drift": {"world": "month", "gap": "month", "p3": ("month", "oosc"), "s1": "month",
              "cca": "month", "ui": "month", "dates": DRIFT_DATES},
    "year": {"world": "year", "gap": "year", "p3": ("year", "oos"), "s1": "year",
             "cca": "year", "ui": "year", "dates": ["20150511"]},
}


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("GPU disappeared; stopping")
    return torch.device("cuda:0")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def append_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row)) + "\n")
        f.flush()


def seen_keys(path: Path, fields: tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.add(tuple(r.get(k) for k in fields))
    return out


def _date(s: str) -> dt.date:
    s = s[-8:]
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def elapsed_days(date: str, ref: str) -> int:
    return (_date(date) - _date(ref)).days


def np_map(weight, bias, mean, std, max_bin, *, staged=False, readout=None) -> AffineMap:
    if bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float64)
    return AffineMap(np.asarray(weight, np.float64), np.asarray(bias, np.float64),
                     np.asarray(mean, np.float64), np.asarray(std, np.float64),
                     max_bin_read=int(max_bin), staged_trial0=bool(staged),
                     readout=None if readout is None else np.asarray(readout, np.float64))


def to_cpu(t):
    if torch.is_tensor(t):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, np.float64)


# ---------------------------------------------------------------- selections ----

def _episodes(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rules_from_episodes(episodes: list[dict], eta_of_B) -> dict:
    """Both gates from stored per-episode grids. ``eta_of_B(B)`` is the R2 eta (tie break)."""
    by = {}
    for r in episodes:
        by.setdefault(int(r["B"]), []).append(r)
    out = {"b": {}, "protocol": episodes[0].get("protocol", "random_draw_legacy") if episodes else "empty"}
    for B, group in sorted(by.items()):
        eta_r2 = float(eta_of_B(B, group))
        out["b"][str(B)] = {
            "or": margin_pick(group, "rebase", eta_r2, mode="or"),
            "and": margin_pick(group, "rebase", eta_r2, mode="and"),
            "n_episodes": len(group),
            "sessions": sorted({g["session"][-8:] for g in group}),
        }
    return out


def load_s1_rules(tag: str) -> tuple[dict, str]:
    """Prefer a causal-first-B file. Otherwise mark the legacy random-draw grid."""
    causal = OUT / "select" / f"s1_{tag}.json"
    if causal.exists():
        return json.loads(causal.read_text()), "causal_first_B"
    legacy_ep = STREAM / f"select_{tag}_rebased_episodes.jsonl"
    if not legacy_ep.exists():
        raise FileNotFoundError(f"no S1 selection for {tag} (looked at {causal} and {legacy_ep})")
    eps = _episodes(legacy_ep)

    def eta_of(B, group, tag=tag):
        if "eta_r2" in group[0]:
            return group[0]["eta_r2"]
        # legacy rows store eta_r2
        return group[0].get("eta_r2", 1.0)

    rules = _rules_from_episodes(eps, eta_of)
    rules["source"] = str(legacy_ep)
    return rules, "random_draw_legacy"


def load_combo_rules(tag: str) -> tuple[dict | None, str]:
    path = OUT / "select" / f"combo_{tag}.json"
    if not path.exists():
        return None, "missing"
    return json.loads(path.read_text()), "causal_first_B"


def load_p3(stratum_key: str) -> dict:
    # ``month`` is the drift trunk's rule file. It is not a separate SPECS entry.
    if stratum_key in ("month", "drift"):
        stratum, sim = "month", "oosc"
    else:
        stratum, sim = SPECS[stratum_key]["p3"]
    return json.loads(pc.selection_file(stratum, sim).read_text())


def load_ui(stratum_key: str) -> dict:
    spec = UI_RULE[SPECS[stratum_key]["ui"]]
    return json.loads(Path(spec["file"]).read_text()), spec


# ---------------------------------------------------------------- prefix cache ----

def prefix_cache(sd, prior, calibs: dict, k_max: int, world, cca_stratum: str):
    """P0 table per (seed, k), shared norm, and seed-free CCA in that norm. k = 1..k_max."""
    norm, tables, cca = {}, {s: {} for s in calibs}, {}
    for k in range(1, int(k_max) + 1):
        idx = calibration_indices(sd.bank.n_trials, k)
        stats = PopStats.from_trials(sd.bank, idx)
        mean_t, std_t = norm_stats(stats)
        norm[k] = (to_cpu(mean_t), to_cpu(std_t))
        for seed, calib in calibs.items():
            res = calib.calibrate(stats, prior)
            if bool(res.grad_enabled_inside):
                raise AssertionError("P0 calibrate ran with grad enabled")
            tables[seed][k] = to_cpu(res.table)
        try:
            got = cca_in_w0_coordinates(world, sd, idx, norm[k][0], norm[k][1], cca_stratum)
        except Exception as exc:                                  # noqa: BLE001 - recorded, not hidden
            got = None
            cca[k] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if got is None:
            cca[k] = None
        else:
            w, b, info = got
            cca[k] = (np.asarray(w, np.float64), np.asarray(b, np.float64),
                      {kk: info.get(kk) for kk in ("n_common_conditions", "d_eff", "family")})
        if k == 1 or k == int(k_max) or k % 8 == 0:
            print(f"[cache] k={k}/{k_max} cca={'no' if not isinstance(cca[k], tuple) else 'yes'}", flush=True)
    return {"norm": norm, "p0": tables, "cca": cca}


def w_alpha_of(cache, seed: int, k: int, alpha: float):
    """(weight, bias, mean, std) or None when CCA is missing and alpha > 0."""
    mean, std = cache["norm"][k]
    w0 = cache["p0"][seed][k]
    if float(alpha) == 0.0:
        return w0, np.zeros(w0.shape[0]), mean, std, {"alpha": 0.0, "cca": "unused"}
    got = cache["cca"][k]
    if not isinstance(got, tuple):
        return None
    w, b = shrink_tables(w0, got[0], got[1], float(alpha))
    return w, b, mean, std, {"alpha": float(alpha), "cca": got[2]}


def cold_and_initial(prior_np, cache, spans):
    n = prior_np.shape[1]
    cold = np_map(prior_np, None, np.zeros(n), np.ones(n), -1)
    mean, std = cache["norm"][1]
    initial = np_map(prior_np, None, mean, std, spans[0][1] - 1, staged=True)
    return cold, initial


def p0_commits(cache, seed, spans, B, cold, initial):
    """Anytime P0. Commit k is P0 of the first k trials. Trial 0 itself stays on the prior."""
    maps = {COLD: cold, INITIAL: initial}
    reads = {COLD: -1, INITIAL: initial.max_bin_read}
    for k in range(1, B + 1):
        w = cache["p0"][seed][k]
        mean, std = cache["norm"][k]
        maps[k] = np_map(w, None, mean, std, spans[k - 1][1] - 1)
        reads[k] = maps[k].max_bin_read
    offline = cache["p0"][seed][B]
    if not np.allclose(maps[B].weight, offline):
        raise AssertionError("P0 final commit != offline P0 on the first B trials")
    return maps, reads


def prior_commits(prior_np, cache, spans, B, cold, initial):
    maps = {COLD: cold, INITIAL: initial}
    reads = {COLD: -1, INITIAL: initial.max_bin_read}
    for k in range(1, B + 1):
        mean, std = cache["norm"][k]
        maps[k] = np_map(prior_np, None, mean, std, spans[k - 1][1] - 1)
        reads[k] = maps[k].max_bin_read
    if not np.allclose(maps[B].weight, prior_np):
        raise AssertionError("P-prior final table is not the electrode prior")
    return maps, reads


def alpha_commits(cache, seed, spans, B, alpha, cold, initial):
    built = w_alpha_of(cache, seed, B, alpha)
    if built is None:
        return None, None, {"unavailable": True, "alpha": float(alpha)}
    maps, reads = {}, {}
    if cold is not None:
        maps[COLD] = cold
        reads[COLD] = -1
    if initial is not None:
        maps[INITIAL] = initial
        reads[INITIAL] = initial.max_bin_read
    info = None
    for k in range(1, B + 1):
        got = w_alpha_of(cache, seed, k, alpha)
        if got is None:
            return None, None, {"unavailable": True, "alpha": float(alpha), "k": k}
        w, b, mean, std, info = got
        maps[k] = np_map(w, b, mean, std, spans[k - 1][1] - 1)
        reads[k] = maps[k].max_bin_read
    # Offline on the first B is commit B by construction (same shrink of the same P0 and CCA).
    return maps, reads, info


def cca_final_only(cache, spans, B, cold, initial):
    """Reference: one CCA map of the whole block, installed at the final boundary."""
    got = cache["cca"][B]
    if not isinstance(got, tuple):
        return None, None, {"unavailable": True, "reason": got}
    w, b, info = got
    mean, std = cache["norm"][B]
    final = np_map(w, b, mean, std, spans[B - 1][1] - 1)
    maps = {COLD: cold, INITIAL: initial, B: final}
    reads = {COLD: -1, INITIAL: initial.max_bin_read, B: final.max_bin_read}
    return maps, reads, info


def _base_bias_fn(cache, seed, alpha, device):
    """CCA bias of the prefix W_alpha. Not used on the trial-0 prior forward."""

    def base_bias(k, closed):
        if not closed and int(k) == 0:
            return None
        nprefix = int(k) + 1 if closed else int(k)
        got = w_alpha_of(cache, seed, nprefix, alpha)
        if got is None:
            raise RuntimeError(f"W_alpha unavailable at prefix {nprefix}")
        return torch.as_tensor(np.asarray(got[1], np.float32), device=device)

    return base_bias


def _prepare_from(cache, seed, base_kind: str, alpha: float, device):
    """``prepare(k, closed)`` for the rebased loop. ``base_kind`` is ``p0`` or ``w_alpha``."""

    def as_t(arr):
        return torch.as_tensor(np.asarray(arr, np.float32), device=device)

    def prepare(k, closed):
        nprefix = 1 if (not closed and k == 0) else (k + 1 if closed else k)
        if base_kind == "p0":
            mean, std = cache["norm"][nprefix]
            table = None if (not closed and k == 0) else as_t(cache["p0"][seed][nprefix])
            return as_t(mean), as_t(std), table
        got = w_alpha_of(cache, seed, nprefix, alpha)
        if got is None:
            raise RuntimeError(f"W_alpha unavailable at prefix {nprefix}")
        w, _b, mean, std, _info = got
        # The loop adds a latent bias on top of the base table. Shrinkage bias is
        # part of the base, so it is folded into the table's affine bias by
        # returning the weight only here and carrying b through the map builder
        # below — see ``loop_commits``.
        table = None if (not closed and k == 0) else as_t(w)
        return as_t(mean), as_t(std), table

    return prepare


def loop_commits(base, sd, prior_t, cache, seed, spans, B, *, eta, rho, jbar, alpha, kind: str):
    """S1 or combo commits from ``rebased_feedback_loop`` (e7af2d8), one snapshot per trial.

    ``kind='p0'`` is S1 on the prefix P0. ``kind='w_alpha'`` uses the prefix W_alpha
    table as that base (its shrinkage bias is stored beside the correction).
    eta = rho = 0 returns the base itself (checked).
    """
    device = prior_t.device
    if float(eta) == 0.0 and float(rho) == 0.0:
        if kind == "p0":
            maps = {}
            reads = {}
            for k in range(1, B + 1):
                w = cache["p0"][seed][k]
                mean, std = cache["norm"][k]
                maps[k] = np_map(w, None, mean, std, spans[k - 1][1] - 1)
                reads[k] = maps[k].max_bin_read
            return maps, reads, {"eta": 0.0, "rho": 0.0, "fallback_base": True, "n_commits": 0}
        maps, reads, info = alpha_commits(cache, seed, spans, B, alpha, None, None)
        # alpha_commits included cold/initial; the caller overwrites those.
        only = {k: maps[k] for k in range(1, B + 1)}
        return only, {k: reads[k] for k in only}, {**info, "eta": 0.0, "rho": 0.0, "fallback_base": True}

    idx = calibration_indices(sd.bank.n_trials, B)
    raw = [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype), sd.valid[a:b].to(dtype=sd.x32.dtype))
           for a, b in (sd.trials[int(i)] for i in idx)]
    prepare = _prepare_from(cache, seed, kind, alpha, device)
    hist = []
    bias_fn = _base_bias_fn(cache, seed, alpha, device) if kind == "w_alpha" and float(alpha) != 0.0 else None
    ans = sf.rebased_feedback_loop(base, raw, prior_t.float(), float(eta), float(rho), prepare,
                                   mode="rebase", kind="fir", jbar=jbar, history=hist,
                                   base_bias=bias_fn)
    if len(hist) != B:
        raise AssertionError(f"expected {B} commits, got {len(hist)}")
    if not torch.allclose(ans["table"], hist[-1]["table"], atol=1e-6):
        raise AssertionError("final rebased table != last snapshot (offline loop output)")
    maps, reads = {}, {}
    for item in hist:
        k = int(item["k"]) + 1
        bias = to_cpu(item["bias"])
        if kind == "w_alpha" and float(alpha) != 0.0:
            # Base weight already contains W_alpha. Its CCA bias was not inside the
            # loop (the loop's bias is the correction). Add the shrinkage bias in
            # the same coordinates: b_alpha lives in the prefix norm of commit k.
            got = w_alpha_of(cache, seed, k, alpha)
            bias = bias + np.asarray(got[1], np.float64)
        maps[k] = np_map(to_cpu(item["table"]), bias, to_cpu(item["mean"]), to_cpu(item["std"]),
                         spans[k - 1][1] - 1)
        reads[k] = maps[k].max_bin_read
    return maps, reads, {"eta": float(eta), "rho": float(rho), "fallback_base": False, "n_commits": B}


def r1_commits(world, sd, prior, calib, cache, seed, spans, B, lam, kind, basis):
    """P3 ``stream_stats`` anytime: after trial k, P0 of trials <= k plus the ridge so far."""
    base = world.adapter.model.base
    idx = calibration_indices(sd.bank.n_trials, B)
    tot = None
    maps, reads = {}, {}
    v0 = fr.head_matrix(base).double()
    for j, tr in enumerate(idx):
        if j == 0:
            mean, std = cache["norm"][1]
            mean_t = torch.as_tensor(mean, device=sd.x32.device)
            std_t = torch.as_tensor(std, device=sd.x32.device)
            table = prior.float()
        else:
            w = cache["p0"][seed][j]
            mean, std = cache["norm"][j]
            mean_t = torch.as_tensor(mean, device=sd.x32.device)
            std_t = torch.as_tensor(std, device=sd.x32.device)
            table = torch.as_tensor(w, dtype=torch.float32, device=sd.x32.device)
        a, b = sd.trials[int(tr)]
        xz = (sd.x32[a:b] - mean_t) / std_t
        x, y, m = fr.pad_trials(xz, sd.y_norm.float()[a:b], sd.valid[a:b], [(0, b - a)])
        st = fr.readout_stats(base, x, y, m, table)
        tot = st if tot is None else fr.add_readout_stats(tot, st)
        if kind in ("full", "diag"):
            dv = fr.ridge_delta(tot, lam, kind)
        else:
            r = int(kind[2:])
            dv = fr.ridge_delta(tot, lam, "lowrank", basis=basis[:, :r])
        readout = (v0 + dv).detach().cpu().numpy()
        k = j + 1
        w_k = cache["p0"][seed][k]
        mean_k, std_k = cache["norm"][k]
        maps[k] = np_map(w_k, None, mean_k, std_k, spans[k - 1][1] - 1, readout=readout)
        reads[k] = maps[k].max_bin_read
    return maps, reads, {"kind": kind, "lam": float(lam), "streaming_passes": 1}


def final_reference_map(weight, bias, mean, std, spans, B, readout=None):
    final = np_map(weight, bias, mean, std, spans[B - 1][1] - 1, readout=readout)
    return final


# ---------------------------------------------------------------- decode ----

def _pack(amap: AffineMap, device):
    def t(arr):
        return torch.as_tensor(np.asarray(arr, np.float32), device=device)
    out = {"weight": t(amap.weight), "bias": t(amap.bias), "mean": t(amap.mean), "std": t(amap.std),
           "readout": None if amap.readout is None else t(amap.readout)}
    return out


def _build_z(x, segments, packed):
    kdim = packed[segments[0][2]]["weight"].shape[0]
    z = torch.empty(x.shape[0], kdim, device=x.device, dtype=torch.float32)
    for a, b, k in segments:
        m = packed[k]
        xz = (x[a:b] - m["mean"]) / m["std"].clamp_min(1e-8)
        z[a:b] = xz @ m["weight"].T + m["bias"]
    return z


def _forward_batch(base, z, want_h: bool):
    with torch.inference_mode():
        if not want_h:
            return base(z), None
        store = {}
        handle = base.out_proj.register_forward_pre_hook(lambda m, i: store.__setitem__("h", i[0]))
        try:
            y = base(z)
        finally:
            handle.remove()
        return y, store["h"]


def apply_readouts(y, h, segments, packed):
    """Replace trunk-head outputs on segments that carry a readout V [D, H+1]."""
    if h is None:
        return y
    out = y.clone()
    ones = torch.ones(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
    for a, b, k in segments:
        v = packed[k]["readout"]
        if v is None:
            continue
        hb = torch.cat([h[a:b], ones[a:b]], -1)
        out[a:b] = hb @ v.to(dtype=hb.dtype).T
    return out


def physical_r2(y, qmask, behavior, label_stats) -> float:
    ym, ys = label_stats
    idx = np.flatnonzero(np.asarray(qmask, dtype=bool))
    idx = idx[idx < y.shape[0]]
    if len(idx) == 0:
        return float("nan")
    pred = y[idx].detach().double().cpu().numpy() * np.asarray(ys, np.float64) + np.asarray(ym, np.float64)
    truth = np.asarray(behavior, np.float64)[idx]
    if not np.isfinite(pred).all():
        return -1.0
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def decode_many(base, x, jobs, label_stats, behavior):
    """``jobs`` is a list of dicts with segments, packed, qmask, want_h. Returns r2 per job.

    Jobs are batched. A CUDA OOM retries them one at a time.
    """
    if not jobs:
        return []
    want = [j for j in jobs if j["want_h"]]
    plain = [j for j in jobs if not j["want_h"]]
    scores = [None] * len(jobs)
    index = {id(j): i for i, j in enumerate(jobs)}

    def run(group, want_h):
        if not group:
            return
        try:
            z = torch.stack([_build_z(x, j["segments"], j["packed"]) for j in group], 0)
            y, h = _forward_batch(base, z, want_h)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            for j in group:
                z1 = _build_z(x, j["segments"], j["packed"]).unsqueeze(0)
                y1, h1 = _forward_batch(base, z1, want_h)
                yy = y1[0]
                if want_h:
                    yy = apply_readouts(yy, h1[0], j["segments"], j["packed"])
                scores[index[id(j)]] = physical_r2(yy, j["qmask"], behavior, label_stats)
            return
        for i, j in enumerate(group):
            yy = y[i]
            if want_h:
                yy = apply_readouts(yy, h[i], j["segments"], j["packed"])
            scores[index[id(j)]] = physical_r2(yy, j["qmask"], behavior, label_stats)

    # Chunk so a long session stays inside 24 GB. h is [M, T, 128].
    chunk = 4 if x.shape[0] > 150_000 else 8
    for group, want_h in ((plain, False), (want, True)):
        for i in range(0, len(group), chunk):
            run(group[i:i + chunk], want_h)
    return scores


def install_maps(body, cold, initial, B):
    """``body`` is commits 1..B. Returns the full map dict and read record."""
    maps = {COLD: cold, INITIAL: initial, **body}
    reads = {k: m.max_bin_read for k, m in maps.items()}
    missing = [k for k in range(1, B + 1) if k not in maps]
    if missing:
        raise AssertionError(f"missing commits {missing[:6]}")
    return maps, reads


def segments_for(spans, B, n_bins, latency, install):
    return commit_segments(spans, B, n_bins, latency=latency, staged_trial0=True, install=install)


# ---------------------------------------------------------------- one target ----

def _choice(rules, B, gate: str):
    if rules is None:
        return None
    block = rules["b"].get(str(int(B)))
    if block is None:
        return None
    return block.get(gate) or block.get("rebase" if gate == "or" else None)


def eval_target(world, p0, sd, prior, cache, spec_rules, jbars, basis, *,
                stratum, backbone, date, gap_info_of, seen, path, methods, Bs, seeds, latencies):
    """All methods × seeds × latencies for one loaded target. Appends rows as it goes."""
    base = world.adapter.model.base
    spans = [(int(a), int(b)) for a, b in sd.trials]
    n_bins = int(sd.n_bins)
    prior_np = to_cpu(prior)
    behavior = sd.record.behavior
    label_stats = world.label_stats
    target = sd.name
    cold, initial = cold_and_initial(prior_np, cache, spans)
    dev = sd.x32.device
    t0 = time.perf_counter()
    for B in Bs:
        if B > len(spans):
            print(f"[skip] {target} B={B} only {len(spans)} usable", flush=True)
            continue
        alpha_info = gap_info_of(B)
        raw_alpha = None if alpha_info is None else alpha_info.get("w_alpha_gap")
        alpha = None if raw_alpha is None else float(raw_alpha)
        s1_or = _choice(spec_rules["s1"], B, "or")
        s1_and = _choice(spec_rules["s1"], B, "and")
        combo_rules = spec_rules["combo"]
        # alpha 0 => combo is S1. Reuse the S1 gate instead of a second source grid.
        if alpha == 0.0 and combo_rules is None:
            combo_or, combo_and = s1_or, s1_and
            combo_proto = spec_rules["s1_proto"] + "+alpha0_equals_s1"
        else:
            combo_or = _choice(combo_rules, B, "or")
            combo_and = _choice(combo_rules, B, "and")
            combo_proto = spec_rules["combo_proto"]
        p3 = spec_rules["p3"]
        r2_cfg = p3["r2"][str(B)]["cfg"]
        k1 = pick_kind(p3, "r1s", B)
        k3 = pick_kind(p3, "r3", B)
        lam1 = p3[f"r1s_{k1}"][str(B)]["cfg"]["lam"]
        lam3 = p3[f"r3_{k3}"][str(B)]["cfg"]["lam"]
        ui_block, ui_spec = spec_rules["ui"]
        ui = ui_block[str(B)]
        jb = jbars[int(r2_cfg["L"])]
        for seed in seeds:
            need = []
            for method in methods:
                for lat in latencies:
                    if (method, stratum, backbone, target, int(B), int(seed), int(lat)) in seen:
                        continue
                    need.append((method, lat))
            if not need:
                continue
            pc._BWD["n"] = 0
            built = {}
            notes = {}

            def add_anytime(name, body, reads_body, cfg):
                maps, reads = install_maps(body, cold, initial, B)
                for lat, _m in ((lat, name) for name, lat in need if name == name):
                    pass
                built[name] = (maps, reads, "anytime", cfg)

            if any(m == "p_prior" for m, _ in need):
                maps, reads = prior_commits(prior_np, cache, spans, B, cold, initial)
                built["p_prior"] = (maps, reads, "anytime", {})
            if any(m == "p0" for m, _ in need):
                maps, reads = p0_commits(cache, seed, spans, B, cold, initial)
                built["p0"] = (maps, reads, "anytime", {})
            if any(m == "w_alpha_gap" for m, _ in need):
                maps, reads, info = alpha_commits(cache, seed, spans, B, alpha, cold, initial)
                if maps is None:
                    notes["w_alpha_gap"] = info
                else:
                    built["w_alpha_gap"] = (maps, reads, "anytime", info)
            if any(m == "ridge_cca" for m, _ in need):
                maps, reads, info = cca_final_only(cache, spans, B, cold, initial)
                if maps is None:
                    notes["ridge_cca"] = info
                else:
                    built["ridge_cca"] = (maps, reads, "final_only", info)

            def s1_body(choice, label):
                if choice is None:
                    notes[label] = {"unavailable": True, "reason": "no selection"}
                    return
                eta, rho = float(choice["eta"]), float(choice["rho"])
                body, reads, info = loop_commits(
                    base, sd, prior, cache, seed, spans, B, eta=eta, rho=rho, jbar=jb,
                    alpha=0.0, kind="p0")
                info = {**info, "gate": choice.get("mode"), "source_gain": choice.get("source_gain"),
                        "source_se": choice.get("source_se"), "fallback_base": choice.get("fallback_base",
                                                                                          choice.get("fallback_p0"))}
                built[label] = (install_maps(body, cold, initial, B)[0],
                                {**reads, COLD: -1, INITIAL: initial.max_bin_read}, "anytime", info)

            if any(m == "s1_causal" for m, _ in need):
                s1_body(s1_or, "s1_causal")
            if any(m == "s1_causal_and" for m, _ in need):
                if s1_and is not None and s1_or is not None and (
                        float(s1_and["eta"]), float(s1_and["rho"])) == (float(s1_or["eta"]), float(s1_or["rho"])):
                    if "s1_causal" in built:
                        built["s1_causal_and"] = built["s1_causal"]
                    else:
                        s1_body(s1_and, "s1_causal_and")
                else:
                    s1_body(s1_and, "s1_causal_and")
            if any(m in ("combo", "combo_and") for m, _ in need):
                for label, choice in (("combo", combo_or), ("combo_and", combo_and)):
                    if not any(m == label for m, _ in need):
                        continue
                    if choice is None or alpha is None:
                        notes[label] = {"unavailable": True, "reason": "no combo selection or alpha"}
                        continue
                    if label == "combo_and" and combo_or is not None and (
                            float(choice["eta"]), float(choice["rho"])) == (
                            float(combo_or["eta"]), float(combo_or["rho"])) and "combo" in built:
                        built["combo_and"] = built["combo"]
                        continue
                    try:
                        body, reads, info = loop_commits(
                            base, sd, prior, cache, seed, spans, B,
                            eta=float(choice["eta"]), rho=float(choice["rho"]), jbar=jb,
                            alpha=float(alpha), kind="p0" if alpha == 0.0 else "w_alpha")
                    except RuntimeError as exc:
                        notes[label] = {"unavailable": True, "reason": str(exc)}
                        continue
                    info = {**info, "alpha": alpha, "gate": choice.get("mode"),
                            "source_gain": choice.get("source_gain"),
                            "fallback_base": choice.get("fallback_base", choice.get("fallback_p0"))}
                    built[label] = (install_maps(body, cold, initial, B)[0],
                                    {**reads, COLD: -1, INITIAL: initial.max_bin_read}, "anytime", info)
            if any(m == "r1_stream" for m, _ in need):
                body, reads, info = r1_commits(world, sd, prior, p0[seed], cache, seed, spans, B,
                                              lam1, k1, basis)
                built["r1_stream"] = (install_maps(body, cold, initial, B)[0],
                                      {**reads, COLD: -1, INITIAL: initial.max_bin_read}, "anytime", info)
            bwd_after_candidates = pc._BWD["n"]
            if any(m in ("r2", "r3", "p0_ui_reg") for m, _ in need):
                idx = calibration_indices(sd.bank.n_trials, B)
                ep = pc.p0_episode(world, sd, prior, p0[seed], idx)
                if any(m in ("r2", "r3") for m, _ in need):
                    tab = ep.refine(jb, r2_cfg["eta"], r2_cfg["rho"], [r2_cfg["S"]])[r2_cfg["S"]]
                    mean_b, std_b = to_cpu(ep.mean), to_cpu(ep.std)
                    if any(m == "r2" for m, _ in need):
                        final = final_reference_map(to_cpu(tab), None, mean_b, std_b, spans, B)
                        maps = {COLD: cold, INITIAL: initial, B: final}
                        reads = {k: m.max_bin_read for k, m in maps.items()}
                        built["r2"] = (maps, reads, "final_only", dict(r2_cfg))
                    if any(m == "r3" for m, _ in need):
                        st3 = ep.stats(tab)
                        v = ep.v_from(st3, lam3, k3, basis).detach().cpu().numpy()
                        final = final_reference_map(to_cpu(tab), None, mean_b, std_b, spans, B, readout=v)
                        maps = {COLD: cold, INITIAL: initial, B: final}
                        reads = {k: m.max_bin_read for k, m in maps.items()}
                        built["r3"] = (maps, reads, "final_only", {**r2_cfg, "kind": k3, "lam": lam3})
                if any(m == "p0_ui_reg" for m, _ in need):
                    from scripts import run_pc3_uiref as uiref
                    tab = uiref.run_traj(world, sd, ep, ui["lam"], ui["lr"], (ui["steps"],))[ui["steps"]]
                    mean_b, std_b = to_cpu(ep.mean), to_cpu(ep.std)
                    final = final_reference_map(tab, None, mean_b, std_b, spans, B)
                    maps = {COLD: cold, INITIAL: initial, B: final}
                    reads = {k: m.max_bin_read for k, m in maps.items()}
                    built["p0_ui_reg"] = (maps, reads, "final_only",
                                          {"lam": ui["lam"], "lr": ui["lr"], "steps": ui["steps"],
                                           "like_for_like": bool(ui_spec["like_for_like"])})
            q = sd.query_mask(calibration_indices(sd.bank.n_trials, B)).detach().cpu().numpy()
            jobs, job_meta = [], []
            for method, lat in need:
                if method not in built:
                    continue
                maps, reads, install, _cfg = built[method]
                segs = segments_for(spans, B, n_bins, lat, install)
                assert_segment_causal(segs, spans, reads, staged_trial0=True)
                on_final, n_q = query_bins_on_commit(segs, q, B)
                packed = {k: _pack(maps[k], dev) for _, _, k in segs}
                want_h = any(packed[k]["readout"] is not None for _, _, k in segs)
                jobs.append({"segments": segs, "packed": packed, "qmask": q, "want_h": want_h})
                job_meta.append((method, lat, on_final, n_q, install))
            scores = decode_many(base, sd.x32, jobs, label_stats, behavior)
            rows = []
            for (method, lat, on_final, n_q, install), r2 in zip(job_meta, scores):
                _maps, _reads, _inst, cfg = built[method]
                proto = spec_rules["s1_proto"] if method.startswith("s1") else (
                    combo_proto if method.startswith("combo") else spec_rules["others_proto"])
                rows.append({
                    "stratum": stratum, "backbone": backbone, "target": target, "date": date,
                    "gap_days": None if alpha_info is None else alpha_info.get("elapsed"),
                    "gap_bin": None if alpha_info is None else alpha_info.get("bin"),
                    "B": int(B), "seed": int(seed), "latency": int(lat), "method": method,
                    "r2": r2, "n_query_bins": int(n_q), "query_bins_on_final": int(on_final),
                    "n_usable": len(spans), "n_cold_bins": int(spans[0][0]),
                    "install": install, "reference": method in REFERENCES,
                    "replay": method in ("r2", "r3", "p0_ui_reg"),
                    "selection_protocol": proto,
                    "h2_backward_calls_before_uireg": int(bwd_after_candidates),
                    "cfg": cfg if isinstance(cfg, dict) else {},
                })
            for method, info in notes.items():
                for lat in latencies:
                    if (method, lat) not in need:
                        continue
                    rows.append({
                        "stratum": stratum, "backbone": backbone, "target": target, "date": date,
                        "B": int(B), "seed": int(seed), "latency": int(lat), "method": method,
                        "r2": None, "unavailable": True, "cfg": info, "reference": method in REFERENCES,
                        "selection_protocol": "unavailable", "n_query_bins": int(q.sum()),
                    })
            # Seed-free maps do not depend on the P0 seed. Copy instead of a second forward.
            if seed == min(seeds):
                for row in list(rows):
                    if row["method"] not in SEED_FREE:
                        continue
                    for other in seeds:
                        if int(other) == int(seed):
                            continue
                        key = (row["method"], stratum, backbone, target, int(B), int(other), int(row["latency"]))
                        if key in seen:
                            continue
                        clone = dict(row)
                        clone["seed"] = int(other)
                        rows.append(clone)
            append_jsonl(path, rows)
            for row in rows:
                seen.add((row["method"], stratum, backbone, target, int(B), int(row["seed"]), int(row["latency"])))
            print(f"[eval] {stratum} {date} B={B} s{seed} n={len(rows)} "
                  f"t={time.perf_counter() - t0:.0f}s", flush=True)
        if not torch.cuda.is_available():
            raise SystemExit("GPU disappeared; stopping")


def gap_lookup(rules, date, B):
    return alpha_for_target(rules, int(B), elapsed_days(date, rules["ref"]), c=0.0)


def cmd_eval(args):
    require_gpu()
    pc.set_variant("base")
    pc._patch_autograd_counter()
    device = torch.device("cuda:0")
    strata = [s.strip() for s in args.stratum.split(",")]
    methods = tuple(args.methods.split(",")) if args.methods else ANYTIME + REFERENCES
    Bs = tuple(int(x) for x in args.B.split(",")) if args.B else B_LIST
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else SEEDS
    latencies = tuple(int(x) for x in args.latency.split(",")) if args.latency else LATENCIES
    path = OUT / "rows.jsonl"
    seen = seen_keys(path, ("method", "stratum", "backbone", "target", "B", "seed", "latency"))
    for stratum in strata:
        if stratum == "subc":
            _eval_subc(args, path, seen, methods, Bs, seeds, latencies, device)
            continue
        spec = SPECS[stratum]
        dates = spec["dates"] if not args.targets else args.targets.split(",")
        src = [s[-8:] for s in D1_POOLS[spec["world"]]["sources"].split(",")]
        dates = validate_targets(dates, src)
        rules_gap = load_gap_rules(spec["gap"])
        need_s1 = any(m.startswith("s1") or m.startswith("combo") for m in methods)
        if need_s1:
            s1_rules, s1_proto = load_s1_rules(spec["s1"])
            combo_rules, combo_proto = load_combo_rules(spec["s1"])
        else:
            s1_rules, s1_proto = {"b": {}}, "not_requested"
            combo_rules, combo_proto = None, "not_requested"
        p3 = load_p3(stratum)
        ui_block, ui_spec = load_ui(stratum)
        world = rpc.World(spec["world"], "O", device, load_sessions=False)
        p0 = pc.load_p0(world, spec["world"])
        basis = pc.load_basis(spec["world"])
        # L values the R2 rules actually use.
        lags = sorted({int(p3["r2"][str(B)]["cfg"]["L"]) for B in Bs if str(B) in p3["r2"]})
        jbars = {L: pc.jbar_for(spec["world"], L) for L in lags}
        hash0 = world.adapter._hash()
        bundle = {"s1": s1_rules, "s1_proto": s1_proto, "combo": combo_rules, "combo_proto": combo_proto,
                  "p3": p3, "ui": (ui_block, ui_spec),
                  "others_proto": "random_draw_legacy"}
        for date in dates:
            if not torch.cuda.is_available():
                raise SystemExit("GPU disappeared; stopping")
            target = PREFIX + date
            sd, prior = world.target(target)
            k_max = max(b for b in Bs if b <= len(sd.trials))
            print(f"[load] {stratum} {date} N={sd.n_units} trials={len(sd.trials)} bins={sd.n_bins}", flush=True)
            cache = prefix_cache(sd, prior, p0, k_max, world, spec["cca"])
            eval_target(world, p0, sd, prior, cache, bundle, jbars, basis,
                        stratum=stratum, backbone=spec["world"], date=date,
                        gap_info_of=lambda B, date=date, rules_gap=rules_gap: gap_lookup(rules_gap, date, B),
                        seen=seen, path=path, methods=methods, Bs=Bs, seeds=seeds, latencies=latencies)
            if world.adapter._hash() != hash0:
                raise SystemExit("trunk weights changed during eval")
            del sd, prior, cache
            torch.cuda.empty_cache()


def _eval_subc(args, path, seen, methods, Bs, seeds, latencies, device):
    """Month targets of sc_0629 and sc_0708. B=2 has no source rule and is skipped."""
    from ssm_decode.expansion_protocol import forbidden_in
    from scripts.run_stream_f7 import _subc_contexts
    Bs = tuple(b for b in Bs if b in SUBC_B)
    if not Bs:
        print("[subc] no B in the round-1 grid", flush=True)
        return
    bids = ["sc_0629", "sc_0708"] if not args.targets else None
    # targets filter is a date list applied inside.
    want_dates = None if not args.targets else set(args.targets.split(","))
    for bid in ("sc_0629", "sc_0708"):
        if not torch.cuda.is_available():
            raise SystemExit("GPU disappeared; stopping")
        ctxs = _subc_contexts([bid])
        bb, ctx = ctxs[bid]
        bad = forbidden_in(list(bb.pool) + [t.session for t in bb.targets])
        if bad:
            raise SystemExit(f"{bid} forbidden sessions {bad}")
        sel = json.loads((EXP_OUT / "select" / f"{bid}.json").read_text())
        s1_rules, s1_proto = load_s1_rules(bid)
        combo_rules, combo_proto = load_combo_rules(bid)
        gap_path = OUT / "select" / f"gap_rules_{bid}.json"
        gap_rules = json.loads(gap_path.read_text()) if gap_path.exists() else None
        basis = torch.load(EXP_OUT / f"basis/{bid}.pt", weights_only=False)["U"].to(device)
        lags = sorted({int(sel["b"][str(B)]["r2"]["L"]) for B in Bs})
        jbars = {L: ctx.jbar(L) for L in lags}
        # P3-shaped view so pick_kind / r2 cfg share the sub-M path.
        p3 = {"r2": {}, "r1s_full": {}, "r3_full": {}}
        for B in Bs:
            block = sel["b"][str(B)]
            p3["r2"][str(B)] = {"cfg": {"L": block["r2"]["L"], "eta": block["r2"]["eta"],
                                        "rho": block["r2"]["rho"], "S": block["r2"]["S"]}}
            for prefix, src in (("r1s", block["r1s"]), ("r3", block["r3"])):
                for kind, info in src["per_kind"].items():
                    p3.setdefault(f"{prefix}_{kind}", {})[str(B)] = {
                        "cfg": {"lam": info["lam"]}, "source_r2": info["source_r2"]}
        ui_stub = ({str(B): {"lam": None, "lr": None, "steps": None} for B in Bs},
                   {"like_for_like": False, "note": "sub-C has no like-for-like ui_reg selection"})
        bundle = {"s1": s1_rules, "s1_proto": s1_proto, "combo": combo_rules, "combo_proto": combo_proto,
                  "p3": p3, "ui": ui_stub, "others_proto": "random_draw_legacy"}
        month_targets = [t for t in bb.targets if t.stratum == "month"]
        for t in month_targets:
            date = t.session[-8:]
            if want_dates is not None and date not in want_dates:
                continue
            if date in FINAL_DATES or date == "20140307":
                raise SystemExit(f"refusing {date}")
            sd, prior = ctx.world.target(t.session)
            k_max = max(b for b in Bs if b <= len(sd.trials))
            print(f"[load] subc {bid} {date} N={sd.n_units} trials={len(sd.trials)}", flush=True)
            cache = prefix_cache(sd, prior, ctx.p0, k_max, ctx.world, "month")
            methods_here = tuple(m for m in methods if m != "p0_ui_reg")
            if "p0_ui_reg" in methods:
                # One explicit unavailable row per (B, seed, latency), not a fake trajectory.
                for B in Bs:
                    for seed in seeds:
                        for lat in latencies:
                            key = ("p0_ui_reg", "subc_month", bid, t.session, int(B), int(seed), int(lat))
                            if key in seen:
                                continue
                            append_jsonl(path, [{
                                "stratum": "subc_month", "backbone": bid, "target": t.session, "date": date,
                                "B": int(B), "seed": int(seed), "latency": int(lat), "method": "p0_ui_reg",
                                "r2": None, "unavailable": True, "reference": True,
                                "cfg": {"reason": "no sub-C like-for-like ui_reg selection"},
                                "selection_protocol": "not_run"}])
                            seen.add(key)

            def gap_of(B, date=date, gap_rules=gap_rules, t=t):
                if gap_rules is None:
                    return {"elapsed": int(t.gap_days), "bin": None, "w_alpha_gap": None}
                return gap_lookup(gap_rules, date, B)

            eval_target(ctx.world, ctx.p0, sd, prior, cache, bundle, jbars, basis,
                        stratum="subc_month", backbone=bid, date=date, gap_info_of=gap_of,
                        seen=seen, path=path, methods=methods_here, Bs=Bs, seeds=seeds,
                        latencies=latencies)
            del sd, prior, cache
            torch.cuda.empty_cache()
        del ctxs, ctx
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- select ----

def _score_np(world, sd, mean, std, tables, biases, idx):
    q = sd.query_mask(np.asarray(idx))
    q = q.clone()
    q[SIM_MAX:] = False
    if int(q.sum()) < 50:
        return None
    return score_tables(world, sd, mean, std, tables, biases, q, SIM_MAX)


def _select_tag(tag: str, what: str, limit: int | None):
    """Causal-first-B source grid for S1 (base = P0) or combo (base = W_alpha-gap)."""
    require_gpu()
    pc.set_variant("base")
    device = torch.device("cuda:0")
    if tag in ("days", "month", "year"):
        from scripts.run_p0solve import SIM_SET
        world, names = pc.sim_world(tag if tag != "days" else "days", SIM_SET[tag])
        p0 = pc.load_p0(world, "days" if tag == "days" else tag)
        # year trunk shares month's sessions but has its own P0.
        if tag == "year":
            p0 = pc.load_p0(world, "year")
        gap = load_gap_rules(tag)
        p3 = load_p3(tag)
        cca_key = tag
        jbar_stratum = tag
        Bs = B_LIST
        def eta_r2_of(B, p3=p3):
            return float(p3["r2"][str(B)]["cfg"]["eta"])
        def L_of(B, p3=p3):
            return int(p3["r2"][str(B)]["cfg"]["L"])
        sessions = [(n, world.sessions[n], world.priors[n]) for n in names]
        subc_gap = {}
    else:
        from scripts.run_stream_f7 import _subc_contexts
        ctxs = _subc_contexts([tag])
        bb, ctx = ctxs[tag]
        name = bb.val_session
        sessions = [(name, ctx.world.sessions[name], ctx.world.priors[name])]
        world = ctx.world
        p0 = ctx.p0
        gap_path = OUT / "select" / f"gap_rules_{tag}.json"
        if what == "combo" and not gap_path.exists():
            raise SystemExit(f"combo on {tag} needs {gap_path}; run --what alpha first")
        gap = json.loads(gap_path.read_text()) if gap_path.exists() else None
        sel = json.loads((EXP_OUT / "select" / f"{tag}.json").read_text())
        cca_key = "month"
        jbar_stratum = None
        ctx_holder = ctx
        Bs = SUBC_B
        def eta_r2_of(B, sel=sel):
            return float(sel["b"][str(B)]["r2"]["eta"])
        def L_of(B, sel=sel):
            return int(sel["b"][str(B)]["r2"]["L"])
        subc_gap = {bb.val_session: int(bb.trunk_gap_days(bb.val_session))}
    path = OUT / "select" / f"{what}_{tag}_episodes.jsonl"
    seen = {(r["session"], int(r["seed"]), int(r["B"])) for r in _episodes(path)} if path.exists() else set()
    configs = [(e, r) for e in ETAS for r in RHOS]
    done = 0
    for name, sd, prior in sessions:
        k_max = max(Bs)
        if k_max > sd.bank.n_trials:
            print(f"[select] {name} has only {sd.bank.n_trials} trials", flush=True)
            continue
        print(f"[select] cache {tag} {name[-8:]}", flush=True)
        cache = prefix_cache(sd, prior, p0, k_max, world, cca_key)
        lags = sorted({L_of(B) for B in Bs})
        if tag in ("days", "month", "year"):
            jbars = {L: pc.jbar_for(jbar_stratum, L) for L in lags}
        else:
            jbars = {L: ctx_holder.jbar(L) for L in lags}
        base = world.adapter.model.base
        for seed in SEEDS:
            for B in Bs:
                if (name, seed, B) in seen:
                    continue
                if limit is not None and done >= limit:
                    _write_select_summary(tag, what, path, eta_r2_of)
                    return
                idx = calibration_indices(sd.bank.n_trials, B)
                mean, std = cache["norm"][B]
                w0 = cache["p0"][seed][B]
                zeros = np.zeros(w0.shape[0])
                p0_scores = _score_np(world, sd, mean, std, [w0], [zeros], idx)
                if p0_scores is None:
                    print(f"[select] skip {name[-8:]} B={B} (query < 50 bins inside {SIM_MAX})", flush=True)
                    continue
                p0_r2 = float(p0_scores[0])
                if what == "alpha":
                    rows_a = []
                    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                        got = w_alpha_of(cache, seed, B, alpha)
                        if got is None:
                            rows_a.append({"alpha": alpha, "r2": -1.0})
                            continue
                        w, b, mean_a, std_a, _ = got
                        sc = _score_np(world, sd, mean_a, std_a, [w], [b], idx)
                        rows_a.append({"alpha": alpha, "r2": None if sc is None else float(sc[0])})
                    append_jsonl(path, [{"tag": tag, "session": name, "seed": int(seed), "B": int(B),
                                         "protocol": "causal_first_B", "p0": p0_r2, "alphas": rows_a,
                                         "gap_days": (elapsed_days(name, gap["ref"]) if gap else subc_gap.get(name))}])
                    done += 1
                    print(f"[select-alpha] {name[-8:]} B={B} s{seed} p0={p0_r2:.3f}", flush=True)
                    continue
                alpha = 0.0
                kind = "p0"
                if what == "combo":
                    info = alpha_for_target(gap, B, elapsed_days(name, gap["ref"]), c=0.0)
                    alpha = float(info["w_alpha_gap"])
                    kind = "p0" if alpha == 0.0 else "w_alpha"
                prepare = _prepare_from(cache, seed, kind, alpha, device)
                bias_fn = _base_bias_fn(cache, seed, alpha, device) if kind == "w_alpha" else None
                raw = [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype),
                        sd.valid[a:b].to(dtype=sd.x32.dtype))
                       for a, b in (sd.trials[int(i)] for i in idx)]
                t1 = time.perf_counter()
                many = run_rebased_many(base, raw, prior.float(), configs, prepare, jbar=jbars[L_of(B)],
                                        base_bias=bias_fn)
                tables, biases = [], []
                for ans in many:
                    bias = to_cpu(ans["bias"])
                    if kind == "w_alpha":
                        extra = w_alpha_of(cache, seed, B, alpha)[1]
                        bias = bias + np.asarray(extra, np.float64)
                    tables.append(to_cpu(ans["table"]))
                    biases.append(bias)
                scores = _score_np(world, sd, mean, std, tables, biases, idx)
                rebase = [{"eta": float(e), "rho": float(r), "r2": float(sc)}
                          for (e, r), sc in zip(configs, scores)]
                append_jsonl(path, [{"tag": tag, "session": name, "seed": int(seed), "B": int(B),
                                     "protocol": "causal_first_B", "p0": p0_r2, "alpha": float(alpha),
                                     "rebase": rebase, "eta_r2": eta_r2_of(B)}])
                done += 1
                print(f"[select-{what}] {tag} {name[-8:]} B={B} s{seed} p0={p0_r2:.3f} "
                      f"{time.perf_counter() - t1:.1f}s", flush=True)
        del cache
        torch.cuda.empty_cache()
    _write_select_summary(tag, what, path, eta_r2_of)


def _write_select_summary(tag, what, path, eta_r2_of):
    if what == "alpha":
        _write_alpha_rules(tag, path)
        return
    eps = [r for r in _episodes(path) if r.get("rebase")]
    if not eps:
        print(f"[select] no scored episodes in {path}", flush=True)
        return
    rules = _rules_from_episodes(eps, lambda B, group: group[0].get("eta_r2", eta_r2_of(B)))
    rules["tag"] = tag
    rules["what"] = what
    rules["protocol"] = "causal_first_B"
    dest = OUT / "select" / f"{what}_{tag}.json"
    dest.write_text(json.dumps(rules, indent=1))
    brief = {b: {"or": rules["b"][b]["or"]["eta"], "and": rules["b"][b]["and"]["eta"],
                 "or_fb": rules["b"][b]["or"]["fallback_base"],
                 "and_fb": rules["b"][b]["and"]["fallback_base"]}
             for b in rules["b"]}
    print(f"[select] wrote {dest} {json.dumps(brief)}", flush=True)


def _write_alpha_rules(tag, path):
    from ssm_decode.p0solve import ALPHAS, resolve_gap_bin, gap_bin_name
    rows = _episodes(path)
    if not rows:
        return
    # One source session -> one natural bin. Pool seeds.
    ref = rows[0]["session"]
    # gap of the source session relative to the backbone ref, stored on the row if present.
    by = {}
    for r in rows:
        by.setdefault(int(r["B"]), []).append(r)
    # Bin name: caller stored gap_days. Fall back to 'mid' only if missing — that would be a bug.
    populated = set()
    schedule = {}
    for B, group in by.items():
        bins = {}
        for r in group:
            bname = gap_bin_name(int(r["gap_days"])) if r.get("gap_days") is not None else "mid"
            bins.setdefault(bname, []).append(r)
        populated |= set(bins)
        schedule[str(B)] = {}
        for want in ("le14", "mid", "gt180"):
            used, fell = resolve_gap_bin(want, set(bins) or populated)
            block = bins.get(used, [])
            means = {}
            for alpha in ALPHAS:
                vals = []
                for ep in block:
                    hit = next(a["r2"] for a in ep["alphas"] if float(a["alpha"]) == float(alpha))
                    if hit is not None:
                        vals.append(float(hit))
                means[float(alpha)] = float(np.mean(vals)) if vals else -1.0
            best = sorted(means, key=lambda a: (means[a], -a), reverse=True)[0]
            schedule[str(B)][want] = {
                "alpha": best, "source_r2": means[best], "n_episodes": len(block),
                "bin_used": used, "fallback_from": fell, "populated_bins": sorted(populated),
            }
    # alpha_for_target needs logistic/combined even though eval only reads w_alpha_gap.
    out = {"stratum": tag, "ref": _ref_for_alpha(tag, rows), "protocol": "causal_first_B",
           "n_episodes": len(rows), "schedule": schedule,
           "logistic": {"intercept": 0.0, "slope": 0.0},
           "combined": {"intercept": 0.0, "slope": 0.0},
           "note": "Only w_alpha_gap is used. Logistic fields are placeholders so alpha_for_target can run; "
                   "they are not a fitted reliability model."}
    dest = OUT / "select" / f"gap_rules_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[select-alpha] wrote {dest}", flush=True)


def _ref_for_alpha(tag, rows):
    """Latest trunk-training date. Sub-C: gap_days was measured from that date, so reconstruct it."""
    if tag in ("days", "month", "year"):
        return load_gap_rules(tag)["ref"]
    # Stored gap_days = session_date - ref. Recover ref from the first row.
    r0 = rows[0]
    if r0.get("gap_days") is None:
        raise SystemExit("alpha rows have no gap_days; cannot place the ref date")
    session = _date(r0["session"])
    ref = session - dt.timedelta(days=int(r0["gap_days"]))
    return ref.strftime("%Y%m%d")


def cmd_select(args):
    _select_tag(args.tag, args.what, args.limit)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--tag", required=True, help="days|month|year|sc_0629|sc_0708")
    s.add_argument("--what", required=True, choices=("s1", "combo", "alpha"))
    s.add_argument("--limit", type=int, default=None, help="stop after this many new episodes")
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, help="days,drift,year,subc (comma list)")
    e.add_argument("--targets", default=None, help="comma YYYYMMDD; default is the stratum list")
    e.add_argument("--methods", default=None)
    e.add_argument("--B", default=None)
    e.add_argument("--seeds", default=None)
    e.add_argument("--latency", default=None, help="comma list, default 0,1")
    args = p.parse_args()
    if args.cmd == "select":
        cmd_select(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
