"""M2 adapter for the frozen causal-first-B protocol.

The commit schedule, prefix rule, margin gate and rebased feedback loop are the
000688 implementations (``causal_protocol``, ``stream_f7.rebased_feedback_loop``).
This module only supplies M2 data, the frozen in_proj bias, and the closed-form
maps those routines already know how to install.

Bias split.  P0 / the electrode prior are bias-free weights.  The folded root
``in_proj`` bias (mean of the source-session folds) is a frozen trunk constant.
It is added on every forward and every installed map, and it is NOT scaled by
W_alpha.  Ridge-CCA's affine bias ``b_full`` already lives in trunk-latent
coordinates (the reference-day bridge includes that session's fold bias), so the
cache stores ``b_extra = b_full - prior_bias`` and ``shrink_tables`` — whose
contract is "W0 is bias-free, so b_alpha = alpha * b_cca" — is used unchanged.
The installed bias is ``prior_bias + b_alpha``.

Condition.  Held-in NWB ``intervals/trials/tgt_loc`` is the reach target.
Absolute angle is degenerate (one quadrant); the condition is the 8-bin
direction of ``tgt_loc - finger_pos`` at the trial-start bin, via
``published_baselines.condition_label``.  There is no go cue, so CCA alignment
is ``trial`` (the whole usable trial), not ``go``.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ssm_decode import fwd_refine as fr
from ssm_decode import m2_calib as m2
from ssm_decode import stream_f7 as sf
from ssm_decode.causal_protocol import (COLD, INITIAL, AffineMap, assert_prefix_idx,
                                        assert_recorded_order, assert_segment_causal,
                                        calibration_indices, commit_segments,
                                        measured_read_bin, query_bins_on_commit,
                                        sessions_with_positive_gain)
from ssm_decode.p0solve import ALPHAS, gap_bin_name, map_cca_into_w0_coordinates, resolve_gap_bin, shrink_tables
from ssm_decode.population_calib import PopStats, norm_stats
from ssm_decode.published_baselines import build_ref_day, cca_readin, condition_label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m2_causal_20261020"
HELD_IN = Path("/mnt/data/work_host/SPINT/SPINT-main/data/000953/sub-MonkeyN-held-in-calib")
B_LIST = (1, 2, 4, 8, 16, 33)
SEEDS = (0, 1, 2)
CSEEDS = (0, 1)
POOLS = ("pool1", "pool2", "pool3")
# Month-style grid, restricted to values already used on 000688.  align is fixed
# at "trial" because M2 has no go cue.  One (phases, ridge) is chosen per pool
# on source sessions only.
CCA_PHASES = (10, 20)
CCA_RIDGES = (3.0, 10.0, 30.0)
CCA_TRANSFORM = "raw"
S1_ETAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
S1_RHOS = (0.0, 0.01, 0.03, 0.1, 0.3)
R2_LS = (8, 16, 32)
R2_ETAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
R2_RHOS = (0.0, 0.03, 0.1, 0.3)
R2_STEPS = (1, 3, 5, 10)
R1_LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
R1_BURNS = (0, 10, 20)
R1_RANK = 16
R3_KINDS = ("full", "diag", "lr16")
UI_LAMS = (0.0, 1e-2, 1e-1)
UI_LRS = (1e-3, 3e-3, 1e-2)
UI_STEPS = (10, 30, 100, 300)
JBAR_LMAX = 32
JBAR_WIDTH = 320
JBAR_WINDOWS = 32

# Latest trunk-training date and the set of dates the trunk was fit on.
# Strict protocol (b) drops a trunk whose training dates intersect a reported target.
POOL_TRUNK = {
    "pool1": {"last": "20201019", "train": ("20201019",),
              "sources": ("2020-10-19-Run1", "2020-10-19-Run2")},
    "pool2": {"last": "20201020", "train": ("20201019", "20201020"),
              "sources": ("2020-10-19-Run1", "2020-10-19-Run2", "2020-10-20-Run1", "2020-10-20-Run2")},
    "pool3": {"last": "20201027", "train": ("20201019", "20201020", "20201027"),
              "sources": ("2020-10-19-Run1", "2020-10-19-Run2", "2020-10-20-Run1",
                          "2020-10-20-Run2", "2020-10-27-Run1", "2020-10-27-Run2")},
}
# Held-in D2 targets.  Dates, matching the 000688 strict filter (a date, not a run id).
REPORTED_DATES = frozenset({"20201020", "20201027", "20201028"})
# Latest source of each trunk.  It is the CCA reference day.  Its own validation tail
# is inside the anchor fit, so it does not also vote on α / CCA config / CCA+S1.
ANCHOR_SESSION = {
    "pool1": "2020-10-19-Run2",
    "pool2": "2020-10-20-Run2",
    "pool3": "2020-10-27-Run2",
}
# Sessions a trunk has not trained on, scored as causal-first-B episodes so a later
# target can use them.  pool1's 10-20 runs vote for 10-27 and 10-28; pool2's 10-27
# runs vote for 10-28.
UNSEEN_SESSIONS = {
    "pool1": ("2020-10-20-Run1", "2020-10-20-Run2"),
    "pool2": ("2020-10-27-Run1", "2020-10-27-Run2"),
}
POOL_OF_TARGET = {
    "2020-10-20-Run1": "pool1", "2020-10-20-Run2": "pool1",
    "2020-10-27-Run1": "pool2", "2020-10-27-Run2": "pool2",
    "2020-10-28-Run1": "pool3",
}


def ymd(session: str) -> str:
    """``2020-10-19-Run1`` -> ``20201019``."""
    day = session[:10]
    if len(day) != 10 or day[4] != "-" or day[7] != "-":
        raise ValueError(f"not an M2 session id: {session}")
    return day.replace("-", "")


def ymd_date(session: str) -> dt.date:
    s = ymd(session)
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def days_between(later: str, earlier: str) -> int:
    return (ymd_date(later) - ymd_date(earlier)).days


def target_gaps(pool: str, target: str) -> dict:
    """Days from the latest trunk-training date, and from the latest source date."""
    src = POOL_TRUNK[pool]["sources"]
    latest_src = max(src, key=ymd_date)
    g_trunk = days_between(target, _date_session(POOL_TRUNK[pool]["last"]))
    g_src = days_between(target, latest_src)
    return {"gap_days_trunk": int(g_trunk), "gap_days_source": int(g_src),
            "trunk_last": POOL_TRUNK[pool]["last"], "latest_source": latest_src,
            "gap_bin": gap_bin_name(int(g_trunk)), "sources": list(src)}


def _date_session(ymd_s: str) -> str:
    """A synthetic session id so ``days_between`` can take a bare date."""
    return f"{ymd_s[:4]}-{ymd_s[4:6]}-{ymd_s[6:8]}-Run0"


def nwb_path(session: str) -> Path:
    path = HELD_IN / f"sub-MonkeyN-held-in-calib_ses-{session}_behavior+ecephys.nwb"
    text = str(path)
    if "held-out" in text or "minival" in text or "official_evalai" in text:
        raise PermissionError(path)
    if "held-in-calib" not in text:
        raise PermissionError(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_trial_conditions(session: str) -> np.ndarray:
    """Per raw-trial 8-bin direction of (target - finger position at trial start).

    Returns ``int64 [n_raw_trials]``.  Aligned to ``trial_bounds`` order.  Only the
    held-in-calib NWB is opened, and only the small behavioral arrays.
    """
    import h5py
    rec_path = m2.M2_CACHE / f"{session}.npz"
    blob = np.load(rec_path, allow_pickle=False)
    bounds = np.asarray(blob["trial_bounds"])
    neural_T = int(blob["neural"].shape[0])
    with h5py.File(nwb_path(session), "r") as f:
        loc = np.asarray(f["intervals/trials/tgt_loc"], dtype=np.float64)
        pos_x = np.asarray(f["acquisition/finger_pos/index/data"], dtype=np.float64)
        pos_y = np.asarray(f["acquisition/finger_pos/mrs/data"], dtype=np.float64)
    if pos_x.shape[0] != neural_T or pos_y.shape[0] != neural_T:
        raise AssertionError(f"{session}: finger_pos length != neural ({pos_x.shape[0]} vs {neural_T})")
    if len(loc) != len(bounds):
        raise AssertionError(f"{session}: tgt_loc {len(loc)} != trial_bounds {len(bounds)}")
    if bounds.min() < 0 or bounds.max() > neural_T:
        raise AssertionError(f"{session}: trial_bounds outside the recording")
    start = bounds[:, 0].astype(np.int64)
    origin = np.stack([pos_x[start], pos_y[start]], 1)
    delta = loc - origin
    ang = np.arctan2(delta[:, 1], delta[:, 0])
    labels = np.array([condition_label(float(a)) for a in ang], dtype=np.int64)
    if (labels < 0).any():
        raise AssertionError(f"{session}: non-finite target direction")
    return labels


def usable_conditions(sd, raw_labels: np.ndarray) -> np.ndarray:
    """Condition of each usable trial, matched to ``sd.trials`` by the start bin."""
    bounds = list(sd.record.trial_bounds)
    by_start = {}
    for i, (a, b) in enumerate(bounds):
        by_start.setdefault(int(a), i)
    out = np.empty(len(sd.trials), dtype=np.int64)
    for j, (a, b) in enumerate(sd.trials):
        i = by_start.get(int(a))
        if i is None or int(bounds[i][1]) != int(b):
            raise AssertionError(f"usable trial {(a, b)} is not a raw trial")
        out[j] = int(raw_labels[i])
    return out


class LatentTrunk:
    """Trunk after the read-in.  ``bias`` is added to z before the blocks.

    ``run_trunk`` / the rebased loop already apply ``z = x @ W.T + b_call``.
    Pass ``bias=None`` when the caller puts the frozen prior bias in ``b_call``.
    Pass the fold bias here for routines that only know a weight (F7-J replay,
    the Jacobian at the training operating point).
    """

    def __init__(self, model, bias: torch.Tensor | None = None):
        self.model = model
        self.out_proj = model.out_proj
        self.bias = None if bias is None else bias.detach().float().reshape(-1)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            z = z + self.bias.to(dtype=z.dtype, device=z.device)
        return m2.trunk_from_latent(self.model, z)


def np_map(weight, bias, mean, std, read_bin, *, staged=False, readout=None) -> AffineMap:
    if bias is None:
        bias = np.zeros(np.asarray(weight).shape[0], dtype=np.float64)
    return AffineMap(np.asarray(weight, np.float64), np.asarray(bias, np.float64),
                     np.asarray(mean, np.float64), np.asarray(std, np.float64),
                     max_bin_read=int(read_bin), staged_trial0=bool(staged),
                     readout=None if readout is None else np.asarray(readout, np.float64))


def add_bias(amap: AffineMap, bias: np.ndarray) -> AffineMap:
    return np_map(amap.weight, np.asarray(amap.bias, np.float64) + np.asarray(bias, np.float64),
                  amap.mean, amap.std, amap.max_bin_read, staged=amap.staged_trial0, readout=amap.readout)


def _as_np(t) -> np.ndarray:
    if torch.is_tensor(t):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, np.float64)


def valid_concat(neural, valid, trials, idx) -> np.ndarray:
    rows = []
    for i in np.asarray(idx, dtype=np.int64):
        a, b = trials[int(i)]
        sl = neural[a:b][valid[a:b]]
        if len(sl):
            rows.append(sl)
    if not rows:
        raise ValueError("no valid calibration bins")
    return np.concatenate(rows).astype(np.float64)


def trial_blocks(neural, trials, idx, conds):
    xs, cs = [], []
    for i in np.asarray(idx, dtype=np.int64):
        a, b = trials[int(i)]
        xs.append(np.asarray(neural[a:b], dtype=np.float64))
        cs.append(int(conds[int(i)]))
    return xs, cs


def fit_anchor(trunk: m2.M2Trunk, sd, conds: np.ndarray, *, phases: int):
    """Reference-day PCA bridge.  ``z_ref`` includes that session's fold bias."""
    j = len(trunk.source_ids) - 1
    name = trunk.source_ids[j]
    if sd.name != name:
        raise AssertionError(f"anchor session {sd.name} != latest source {name}")
    bank = torch.load(trunk.pool_dir / "session_bank_best.pt", map_location="cpu", weights_only=False)
    if list(bank["source_sessions"]) != list(trunk.source_ids):
        raise AssertionError("session bank ids != trunk source ids")
    mean = np.asarray(bank["source_statistics"][j][0], dtype=np.float64)
    std = np.asarray(bank["source_statistics"][j][1], dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    idx = np.arange(len(sd.trials))
    x_fit = valid_concat(sd.record.neural, sd.valid_np, sd.trials, idx)
    z_ref = ((x_fit - mean) / std) @ np.asarray(trunk.src_tables[j], np.float64).T
    z_ref = z_ref + np.asarray(trunk.src_bias[j], np.float64)
    active = int((x_fit.std(0) > 1e-6).sum())
    if active < 2:
        raise AssertionError(f"{name}: only {active} active channels")
    xs, cs = trial_blocks(sd.record.neural, sd.trials, idx, conds)
    ref = build_ref_day(name=name, x_fit=x_fit, z_ref=z_ref, electrodes=np.arange(x_fit.shape[1]),
                        ref_mean=mean, ref_std=std, x_ref_trials=xs, ref_conds=cs, k=active,
                        fa_min_priv=0.1, transform=CCA_TRANSFORM, t_phase=int(phases),
                        with_fa=False, with_pca=True)
    ref.m2_active = active
    ref.m2_phases = int(phases)
    return ref


def cca_prefix(ref, sd, idx, conds, mean_p, std_p, prior_bias, *, ridge: float, phases: int):
    """Ridge-CCA in the prefix norm.  Returns ``(W, b_extra, info)`` or ``None``.

    ``b_extra = b_full - prior_bias`` so ``shrink_tables`` stays bias-free.
    """
    idx = np.asarray(idx, dtype=np.int64)
    try:
        x_fit = valid_concat(sd.record.neural, sd.valid_np, sd.trials, idx)
        xs, cs = trial_blocks(sd.record.neural, sd.trials, idx, conds)
        res = cca_readin(ref, x_fit, xs, cs, k=int(ref.m2_active), ridge=float(ridge),
                         t_phase=int(phases), transform=CCA_TRANSFORM)
    except (np.linalg.LinAlgError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if res is None:
        return None, "no common condition"
    t_cca, c_cca, mean_c, std_c, info = res
    w, b_full = map_cca_into_w0_coordinates(t_cca, c_cca, mean_c, std_c, mean_p, std_p)
    # Algebraic identity on the first valid slice of the first calibration trial.
    a0, b0 = sd.trials[int(idx[0])]
    sl = slice(a0, min(a0 + 64, b0))
    raw = np.asarray(sd.record.neural[sl], dtype=np.float64)
    got = (raw - mean_p) / std_p @ w.T + b_full
    ref_z = (raw - mean_c) / std_c @ np.asarray(t_cca, np.float64).T + np.asarray(c_cca, np.float64)
    err = float(np.max(np.abs(got - ref_z)))
    if err > 1e-5:
        raise AssertionError(f"CCA/W0 coordinate error {err} on {sd.name}")
    b_extra = b_full - np.asarray(prior_bias, np.float64).reshape(-1)
    info = dict(info)
    info.update({"coordinate_max_abs": err, "transform": CCA_TRANSFORM, "family": "cca/channels/raw/ridge",
                 "phases": int(phases), "ridge": float(ridge), "d_eff": int(ref.m2_active),
                 "condition": "8-bin direction of tgt_loc minus finger_pos at trial start",
                 "align": "trial"})
    return (w, b_extra, info), None


def load_calibrators(world: m2.M2World, seed: int, pool: str):
    out = {}
    for cseed in CSEEDS:
        path = ROOT / "results/m2_calib_20261016/runs" / f"seed{seed}" / pool / f"c{cseed}" / "best.pt"
        blob = torch.load(path, map_location=world.device, weights_only=False)
        args = blob["args"]
        calib = m2.build_calibrator(world, rank=args["rank"], hidden=args["hidden"],
                                    gain=not args["no_gain"], label_free=args.get("label_free", False))
        calib.load_state_dict(blob["calibrator"])
        calib.eval()
        for p in calib.parameters():
            p.requires_grad_(False)
        out[cseed] = calib
    return out


def prefix_cache(sd, prior_table: torch.Tensor, prior_bias: np.ndarray, calibs: dict, k_max: int,
                 conds: np.ndarray, ref, *, ridge: float, phases: int):
    """P0 per calibrator and seed-free CCA for prefixes ``k = 1..k_max``.

    ``read_bin[k]`` is the measured last bin of trials ``0..k-1``.
    """
    spans = [(int(a), int(b)) for a, b in sd.trials]
    norm, tables, cca, read_bin = {}, {c: {} for c in calibs}, {}, {}
    for k in range(1, int(k_max) + 1):
        idx = calibration_indices(sd.bank.n_trials, k)
        assert_prefix_idx(idx, k)
        read_bin[k] = measured_read_bin(spans, idx)
        stats = PopStats.from_trials(sd.bank, idx)
        mean_t, std_t = norm_stats(stats)
        mean, std = _as_np(mean_t), _as_np(std_t)
        norm[k] = (mean, std)
        for cseed, calib in calibs.items():
            res = calib.calibrate(stats, prior_table)
            if bool(res.grad_enabled_inside):
                raise AssertionError("P0 calibrate ran with grad enabled")
            tables[cseed][k] = _as_np(res.table)
        got, err = cca_prefix(ref, sd, idx, conds, mean, std, prior_bias, ridge=ridge, phases=phases)
        cca[k] = got if got is not None else {"error": err}
    return {"norm": norm, "p0": tables, "cca": cca, "read_bin": read_bin, "spans": spans}


def _end(cache, k: int) -> int:
    return int(cache["read_bin"][int(k)])


def cold_initial(prior_w, prior_b, cache):
    n = prior_w.shape[1]
    cold = np_map(prior_w, prior_b, np.zeros(n), np.ones(n), -1)
    mean, std = cache["norm"][1]
    initial = np_map(prior_w, prior_b, mean, std, _end(cache, 1), staged=True)
    return cold, initial


def p0_maps(cache, cseed, B, prior_b, cold, initial):
    maps = {COLD: cold, INITIAL: initial}
    for k in range(1, B + 1):
        mean, std = cache["norm"][k]
        maps[k] = np_map(cache["p0"][cseed][k], prior_b, mean, std, _end(cache, k))
    return maps


def prior_maps(prior_w, prior_b, cache, B, cold, initial):
    maps = {COLD: cold, INITIAL: initial}
    for k in range(1, B + 1):
        mean, std = cache["norm"][k]
        maps[k] = np_map(prior_w, prior_b, mean, std, _end(cache, k))
    if not np.allclose(maps[B].weight, prior_w):
        raise AssertionError("P-prior final weight is not the mean table")
    if not np.allclose(maps[B].bias, prior_b):
        raise AssertionError("P-prior final bias dropped the folded in_proj bias")
    return maps


def w_alpha_parts(cache, cseed, k, alpha: float):
    """``(W, b_shrink, mean, std, info)`` with ``b_shrink = alpha * b_extra`` (no prior bias)."""
    mean, std = cache["norm"][k]
    w0 = cache["p0"][cseed][k]
    if float(alpha) == 0.0:
        return w0, np.zeros(w0.shape[0]), mean, std, {"alpha": 0.0, "cca": "unused"}
    got = cache["cca"][k]
    if not isinstance(got, tuple):
        return None
    w, b = shrink_tables(w0, got[0], got[1], float(alpha))
    return w, b, mean, std, {"alpha": float(alpha), "cca": got[2]}


def alpha_maps(cache, cseed, B, alpha, prior_b, cold, initial):
    maps = {COLD: cold, INITIAL: initial}
    info = None
    for k in range(1, B + 1):
        got = w_alpha_parts(cache, cseed, k, alpha)
        if got is None:
            return None, {"unavailable": True, "alpha": float(alpha), "k": k}
        w, b_shrink, mean, std, info = got
        maps[k] = np_map(w, np.asarray(prior_b) + b_shrink, mean, std, _end(cache, k))
    return maps, info


def cca_final_maps(cache, B, prior_b, cold, initial):
    got = cache["cca"][B]
    if not isinstance(got, tuple):
        return None, {"unavailable": True, "reason": got}
    w, b_extra, info = got
    mean, std = cache["norm"][B]
    final = np_map(w, np.asarray(prior_b) + b_extra, mean, std, _end(cache, B))
    return {COLD: cold, INITIAL: initial, B: final}, info


def _prepare(cache, cseed, kind, alpha, device):
    def as_t(arr):
        return torch.as_tensor(np.asarray(arr, np.float32), device=device)

    def prepare(k, closed):
        nprefix = 1 if (not closed and k == 0) else (k + 1 if closed else k)
        mean, std = cache["norm"][nprefix]
        if kind == "p0":
            table = None if (not closed and k == 0) else as_t(cache["p0"][cseed][nprefix])
            return as_t(mean), as_t(std), table
        if kind == "cca":
            got = cache["cca"][nprefix]
            if not isinstance(got, tuple):
                raise RuntimeError(f"CCA unavailable at prefix {nprefix}")
            table = None if (not closed and k == 0) else as_t(got[0])
            return as_t(mean), as_t(std), table
        got = w_alpha_parts(cache, cseed, nprefix, alpha)
        if got is None:
            raise RuntimeError(f"W_alpha unavailable at prefix {nprefix}")
        w, _b, mean, std, _info = got
        table = None if (not closed and k == 0) else as_t(w)
        return as_t(mean), as_t(std), table

    return prepare


def _loop_base_bias(cache, cseed, kind, alpha, prior_b, device):
    """Frozen prior bias on every trial, plus the closed-form base bias after trial 0.

    The second argument follows ``rebased_feedback_loop``: it is True only for
    ``mode='staged'``.  Causal-first-B uses rebase, so trial ``k>0`` is forwarded
    with the base of the strict prefix (trials ``< k``).
    """
    prior_t = torch.as_tensor(np.asarray(prior_b, np.float32), device=device)

    def base_bias(k, closed):
        extra = None
        if closed or int(k) != 0:
            nprefix = int(k) + 1 if closed else int(k)
            if kind == "cca":
                got = cache["cca"][nprefix]
                if not isinstance(got, tuple):
                    raise RuntimeError(f"CCA unavailable at prefix {nprefix}")
                extra = got[1]
            elif kind == "w_alpha" and float(alpha) != 0.0:
                got = w_alpha_parts(cache, cseed, nprefix, alpha)
                if got is None:
                    raise RuntimeError(f"W_alpha unavailable at prefix {nprefix}")
                extra = got[1]
        b = prior_t
        if extra is not None:
            b = b + torch.as_tensor(np.asarray(extra, np.float32), device=device)
        return b

    return base_bias


def rebased_maps(base, sd, prior_table, cache, cseed, B, *, eta, rho, jbar, alpha, kind, prior_b,
                 force_loop: bool = False):
    """Anytime commits of S1 / COMBO / CCA+S1.  eta = rho = 0 returns the base maps.

    ``force_loop`` still runs the rebased loop at eta = rho = 0.  The short-circuit is the
    deployed path; the loop must return the same table (the identity test).
    """
    device = prior_table.device
    if float(eta) == 0.0 and float(rho) == 0.0 and not force_loop:
        maps = {}
        for k in range(1, B + 1):
            if kind == "p0":
                w = cache["p0"][cseed][k]
                b = np.asarray(prior_b, np.float64)
                mean, std = cache["norm"][k]
            elif kind == "cca":
                got = cache["cca"][k]
                if not isinstance(got, tuple):
                    raise RuntimeError(f"CCA unavailable at prefix {k}")
                w, b_extra = got[0], got[1]
                b = np.asarray(prior_b, np.float64) + np.asarray(b_extra, np.float64)
                mean, std = cache["norm"][k]
            else:
                got = w_alpha_parts(cache, cseed, k, alpha)
                if got is None:
                    raise RuntimeError(f"W_alpha unavailable at prefix {k}")
                w, b_shrink, mean, std, _info = got
                b = np.asarray(prior_b, np.float64) + b_shrink
            maps[k] = np_map(w, b, mean, std, _end(cache, k))
        return maps, {"eta": 0.0, "rho": 0.0, "fallback_base": True, "kind": kind}

    idx = calibration_indices(sd.bank.n_trials, B)
    assert_prefix_idx(idx, B)
    raw = [(sd.x32[a:b], sd.y_norm[a:b].to(dtype=sd.x32.dtype), sd.valid[a:b].to(dtype=sd.x32.dtype))
           for a, b in (sd.trials[int(i)] for i in idx)]
    prepare = _prepare(cache, cseed, kind, alpha, device)
    bias_fn = _loop_base_bias(cache, cseed, kind, alpha, prior_b, device)
    hist = []
    ans = sf.rebased_feedback_loop(base, raw, prior_table.float(), float(eta), float(rho), prepare,
                                   mode="rebase", kind="fir", jbar=jbar, history=hist, base_bias=bias_fn)
    if len(hist) != B:
        raise AssertionError(f"expected {B} commits, got {len(hist)}")
    if not torch.allclose(ans["table"], hist[-1]["table"], atol=1e-6):
        raise AssertionError("final rebased table != last snapshot")
    maps = {}
    for item in hist:
        k = int(item["k"]) + 1
        bias = _as_np(item["bias"]) + np.asarray(prior_b, np.float64)
        if kind == "cca":
            bias = bias + np.asarray(cache["cca"][k][1], np.float64)
        elif kind == "w_alpha" and float(alpha) != 0.0:
            got = w_alpha_parts(cache, cseed, k, alpha)
            bias = bias + np.asarray(got[1], np.float64)
        maps[k] = np_map(_as_np(item["table"]), bias, _as_np(item["mean"]), _as_np(item["std"]), _end(cache, k))
    return maps, {"eta": float(eta), "rho": float(rho), "fallback_base": False, "n_commits": B, "kind": kind}


def trial_readout_stats(model, z, y, valid, burn: int):
    """Summed augmented-feature Gram of one zero-state trial.

    ``z`` is the latent ``[T, K]`` (read-in already applied, including the frozen bias).
    The first ``burn`` bins are dropped from the Gram.  Padding is not used: the trial
    is forwarded alone, so the trunk state starts at zero.
    """
    with torch.no_grad():
        store = {}
        handle = model.out_proj.register_forward_pre_hook(lambda m, inp: store.__setitem__("h", inp[0]))
        try:
            yhat = m2.trunk_from_latent(model, z.unsqueeze(0))
        finally:
            handle.remove()
        h = store["h"][0]
        yhat = yhat[0]
        msk = valid.to(dtype=z.dtype).clone()
        if burn:
            msk[:int(burn)] = 0
        hb = torch.cat([h, torch.ones_like(h[:, :1])], 1) * msk[:, None]
        r0 = (y.to(dtype=yhat.dtype) - yhat) * msk[:, None]
        G = torch.einsum("ti,tj->ij", hb, hb).double()
        R = torch.einsum("td,tj->dj", r0, hb).double()
        return {"G": G, "R": R, "n": float(msk.sum())}


def r1_maps(model, sd, cache, cseed, B, prior_b, prior_table, lam, burn, basis, v0):
    """Anytime P0 + compact readout.  ``lam is None`` leaves the trunk head in place."""
    idx = calibration_indices(sd.bank.n_trials, B)
    assert_prefix_idx(idx, B)
    tot = None
    maps = {}
    dev = sd.x32.device
    bias_t = torch.as_tensor(np.asarray(prior_b, np.float32), device=dev)
    for j, tr in enumerate(idx):
        # Trial j is forwarded from the zero state with the map of trials < j
        # (trial 0: the prior table and its own normalisation).
        if j == 0:
            mean, std = cache["norm"][1]
            table = prior_table.float()
        else:
            mean, std = cache["norm"][j]
            table = torch.as_tensor(cache["p0"][cseed][j], dtype=torch.float32, device=dev)
        a, b = sd.trials[int(tr)]
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
        xz = (sd.x32[a:b] - mean_t) / std_t.clamp_min(1e-8)
        z = xz @ table.T + bias_t
        st = trial_readout_stats(model, z, sd.y_norm[a:b], sd.valid[a:b], int(burn))
        tot = st if tot is None else fr.add_readout_stats(tot, st)
        k = j + 1
        mean_k, std_k = cache["norm"][k]
        if lam is None:
            readout = None
        else:
            dv = fr.ridge_delta(tot, float(lam), "lowrank", basis=basis[:, :R1_RANK])
            readout = (v0.double() + dv).detach().cpu().numpy()
        maps[k] = np_map(cache["p0"][cseed][k], prior_b, mean_k, std_k, _end(cache, k), readout=readout)
    return maps, {"kind": "lowrank", "rank": R1_RANK, "lam": None if lam is None else float(lam),
                  "burn": int(burn), "streaming_passes": 1}


def replay_batch(sd, idx, mean, std):
    dev = sd.x32.device
    mean_t = torch.as_tensor(np.asarray(mean, np.float32), device=dev)
    std_t = torch.as_tensor(np.asarray(std, np.float32), device=dev)
    xs, ys, ms, windows = [], [], [], []
    cursor = 0
    # pad_trials indexes into a concatenated buffer.  Build one buffer in trial order.
    chunks_x, chunks_y, chunks_m = [], [], []
    for i in np.asarray(idx, dtype=np.int64):
        a, b = sd.trials[int(i)]
        xz = (sd.x32[a:b] - mean_t) / std_t.clamp_min(1e-8)
        chunks_x.append(xz)
        chunks_y.append(sd.y_norm[a:b].float())
        chunks_m.append(sd.valid[a:b].float())
        windows.append((cursor, cursor + (b - a)))
        cursor += b - a
    xz_all = torch.cat(chunks_x, 0)
    y_all = torch.cat(chunks_y, 0)
    m_all = torch.cat(chunks_m, 0)
    return fr.pad_trials(xz_all, y_all, m_all, windows)


def r2_table(caller, sd, idx, table0, mean, std, jbar, eta, rho, steps):
    """F7-J replay of the named trials.  ``caller`` must add the frozen prior bias."""
    X, Y, M = replay_batch(sd, idx, mean, std)
    snaps = fr.fj_refine(caller, X, Y, M, table0.float(), jbar, float(eta), float(rho), (int(steps),))
    return snaps[int(steps)]


def r3_readout(caller, sd, idx, table, mean, std, lam, kind, basis, v0):
    X, Y, M = replay_batch(sd, idx, mean, std)
    st = fr.readout_stats(caller, X, Y, M, table.float())
    if kind == "lr16":
        dv = fr.ridge_delta(st, float(lam), "lowrank", basis=basis[:, :R1_RANK])
    else:
        dv = fr.ridge_delta(st, float(lam), kind)
    return (v0.double() + dv).detach().cpu().numpy()


def reads_of(maps: dict) -> dict:
    return {k: int(m.max_bin_read) for k, m in maps.items()}


def install_check(maps, spans, B, n_bins, latency, install):
    segs = commit_segments(spans, B, n_bins, latency=int(latency), staged_trial0=True, install=install)
    assert_segment_causal(segs, spans, reads_of(maps), staged_trial0=True)
    return segs


def score_static(caller_nobias: LatentTrunk, sd, label_stats, weight, bias, mean, std, qmask) -> float:
    """One map, one contiguous causal pass.  ``caller_nobias`` must not add a second bias."""
    dev = sd.x32.device
    w = torch.as_tensor(np.asarray(weight, np.float32), device=dev)
    b = torch.as_tensor(np.asarray(bias, np.float32), device=dev)
    mean_t = torch.as_tensor(np.asarray(mean, np.float32), device=dev)
    std_t = torch.as_tensor(np.asarray(std, np.float32), device=dev)
    with torch.inference_mode():
        xz = (sd.x32 - mean_t) / std_t.clamp_min(1e-8)
        y = caller_nobias((xz @ w.T + b).unsqueeze(0))[0]
    if torch.is_tensor(qmask):
        q = qmask.to(device=y.device)
    else:
        q = torch.as_tensor(np.asarray(qmask), dtype=torch.bool, device=y.device)
    return m2.r2_physical(label_stats, sd, y, q)[0]


def score_many(caller_nobias: LatentTrunk, sd, label_stats, mean, std, weights, biases, qmask, *,
               chunk: int = 4) -> list[float]:
    """Contiguous decode of several read-in maps that share one normalisation."""
    dev = sd.x32.device
    mean_t = torch.as_tensor(np.asarray(mean, np.float32), device=dev)
    std_t = torch.as_tensor(np.asarray(std, np.float32), device=dev)
    xz = (sd.x32 - mean_t) / std_t.clamp_min(1e-8)
    w = torch.as_tensor(np.stack([np.asarray(t, np.float32) for t in weights]), device=dev)
    b = torch.as_tensor(np.stack([np.asarray(t, np.float32) for t in biases]), device=dev)
    if torch.is_tensor(qmask):
        q = qmask.to(device=dev)
    else:
        q = torch.as_tensor(np.asarray(qmask), dtype=torch.bool, device=dev)
    scores = []
    with torch.inference_mode():
        for i in range(0, w.shape[0], chunk):
            ww, bb = w[i:i + chunk], b[i:i + chunk]
            z = torch.einsum("tn,mkn->mtk", xz, ww) + bb[:, None, :]
            y = caller_nobias(z)
            for row in range(y.shape[0]):
                scores.append(m2.r2_physical(label_stats, sd, y[row], q)[0])
    return scores


def decode_jobs(caller_nobias: LatentTrunk, sd, label_stats, jobs, *, chunk: int = 4) -> list[float]:
    """Piecewise maps.  ``jobs`` items: segments, maps (AffineMap), qmask."""
    dev = sd.x32.device
    scores = [None] * len(jobs)

    def pack(amap: AffineMap):
        def t(arr):
            return torch.as_tensor(np.asarray(arr, np.float32), device=dev)
        return {"weight": t(amap.weight), "bias": t(amap.bias), "mean": t(amap.mean),
                "std": t(amap.std), "readout": None if amap.readout is None else t(amap.readout)}

    def build_z(x, segments, packed):
        kdim = packed[segments[0][2]]["weight"].shape[0]
        z = torch.empty(x.shape[0], kdim, device=x.device, dtype=torch.float32)
        for a, b, k in segments:
            m = packed[k]
            xz = (x[a:b] - m["mean"]) / m["std"].clamp_min(1e-8)
            z[a:b] = xz @ m["weight"].T + m["bias"]
        return z

    grouped = []
    for i, job in enumerate(jobs):
        packed = {k: pack(job["maps"][k]) for _, _, k in job["segments"]}
        want_h = any(packed[k]["readout"] is not None for _, _, k in job["segments"])
        grouped.append((i, job, packed, want_h))

    def run(group, want_h):
        if not group:
            return
        try:
            z = torch.stack([build_z(sd.x32, j["segments"], p) for _, j, p, _ in group], 0)
            with torch.inference_mode():
                if not want_h:
                    y = caller_nobias(z)
                    h = None
                else:
                    store = {}
                    handle = caller_nobias.out_proj.register_forward_pre_hook(
                        lambda m, inp: store.__setitem__("h", inp[0]))
                    try:
                        y = caller_nobias(z)
                    finally:
                        handle.remove()
                    h = store["h"]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            for i, j, p, _wh in group:
                run([(i, j, p, want_h)], want_h)
            return
        ones = None
        for row, (i, j, p, _) in enumerate(group):
            yy = y[row]
            if want_h:
                if ones is None:
                    ones = torch.ones(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
                out = yy.clone()
                for a, b, k in j["segments"]:
                    v = p[k]["readout"]
                    if v is None:
                        continue
                    hb = torch.cat([h[row, a:b], ones[row, a:b]], -1)
                    out[a:b] = hb @ v.to(dtype=hb.dtype).T
                yy = out
            q = j["qmask"]
            if not torch.is_tensor(q):
                q = torch.as_tensor(np.asarray(q), dtype=torch.bool, device=yy.device)
            else:
                q = q.to(device=yy.device)
            scores[i] = m2.r2_physical(label_stats, sd, yy, q)[0]

    for want_h in (False, True):
        group = [g for g in grouped if g[3] is want_h]
        for i in range(0, len(group), chunk):
            run(group[i:i + chunk], want_h)
    if any(s is None for s in scores):
        raise AssertionError("a decode job was not scored")
    return scores


def episode_allowed(row: dict, target_date: str, pool: str, variant: str, *,
                    drop_anchor: bool = False) -> bool:
    """000688 ``alpha_episode_allowed``, with M2 session ids.

    ``pooled`` (primary, also accepted as ``expanding``): any episode whose session and
    whose trunk both predate the target.  A pool1-trunk episode on 10-20 may vote for a
    10-27 target.  ``in_pool`` is the narrower sensitivity (only that target's trunk).
    ``strict`` is pooled plus the reported-target filter.

    ``drop_anchor`` removes the CCA reference session of the trunk that scored the episode.
    That session's validation tail is inside ``fit_anchor``, so it cannot also vote.
    Trunk-unseen episodes are not that reference day.
    """
    if variant not in ("pooled", "expanding", "in_pool", "strict"):
        raise ValueError(variant)
    trunk = str(row["pool"])
    if variant == "in_pool" and trunk != pool:
        return False
    if drop_anchor and str(row.get("kind", "source")) != "unseen":
        if str(row["session"]) == ANCHOR_SESSION[trunk]:
            return False
    date = str(row["date"])
    last = POOL_TRUNK[trunk]["last"]
    if date >= target_date or last >= target_date:
        return False
    if variant == "strict":
        if date in REPORTED_DATES:
            return False
        if set(POOL_TRUNK[trunk]["train"]) & set(REPORTED_DATES):
            return False
    return True


def select_alpha_m2(episodes, pool: str, target: str, B: int, variant: str, *,
                    drop_anchor: bool = False) -> dict:
    """α for one M2 target.  Episodes are binned by the gap they were scored under."""
    want_gap = days_between(target, _date_session(POOL_TRUNK[pool]["last"]))
    want = gap_bin_name(int(want_gap))
    target_date = ymd(target)
    pool_eps = [r for r in episodes
                if episode_allowed(r, target_date, pool, variant, drop_anchor=drop_anchor)
                and int(r["B"]) == int(B)]
    bins = {}
    for r in pool_eps:
        bins.setdefault(gap_bin_name(int(r["gap_days"])), []).append(r)
    if not bins:
        return {"alpha": None, "bin": want, "bin_used": None, "n_episodes": 0, "sessions": [],
                "fallback_from": want, "variant": variant, "pool": pool}
    used, fell = resolve_gap_bin(want, set(bins))
    group = bins[used]
    means = {}
    for alpha in ALPHAS:
        vals = []
        for ep in group:
            hit = next(a["r2"] for a in ep["alphas"] if float(a["alpha"]) == float(alpha))
            if hit is not None and float(hit) > -0.5:
                vals.append(float(hit))
        means[float(alpha)] = float(np.mean(vals)) if vals else -1.0
    winner = sorted(means, key=lambda a: (means[a], -a), reverse=True)[0]
    sessions = sorted({str(r["session"]) for r in group})
    return {"alpha": float(winner), "source_r2": means[winner], "bin": want, "bin_used": used,
            "fallback_from": fell, "n_episodes": len(group), "sessions": sessions,
            "seeds": sorted({int(r["seed"]) for r in group if "seed" in r}),
            "trunks": sorted({str(r["pool"]) for r in group}),
            "drop_anchor": bool(drop_anchor),
            "variant": variant, "pool": pool, "gap_days_trunk": int(want_gap)}


def source_train_idx(world: m2.M2World, name: str, B: int):
    """First B usable trials.  They must lie entirely in the trunk's train region."""
    sd = world.sessions[name]
    if B > sd.bank.n_trials:
        raise ValueError(f"{name}: B={B} exceeds {sd.bank.n_trials} usable trials")
    idx = calibration_indices(sd.bank.n_trials, B)
    vs = int(world.val_start[name])
    for i in idx:
        a, b = sd.trials[int(i)]
        if int(b) > vs:
            raise AssertionError(f"{name}: first-{B} trial {(a, b)} crosses the val cut {vs}")
    return idx


def corrupt_session(sd, label_stats, start: int, seed: int = 1):
    """Replace neural and behavior at and after ``start``.  Trial spans stay put."""
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    neu = np.array(sd.record.neural, dtype=np.float64, copy=True)
    beh = np.array(sd.record.behavior, dtype=np.float64, copy=True)
    # Stay non-negative: the P0 descriptor takes log(mean), so a negative draw is NaN
    # and would not be a meaningful future-bin perturbation.
    scale = float(np.mean(np.abs(neu[:max(start, 1)]))) + 1.0
    neu[start:] = rng.random(neu[start:].shape) * scale
    beh[start:] = rng.normal(size=beh[start:].shape) * 5.0
    rec = SimpleNamespace(session=sd.record.session, neural=neu, behavior=beh,
                          eval_mask=np.array(sd.record.eval_mask, copy=True),
                          trial_bounds=sd.record.trial_bounds,
                          unit_to_electrode=np.array(sd.record.unit_to_electrode, copy=True))
    from ssm_decode.population_calib import SessionData
    return SessionData(rec, label_stats, sd.device, name=sd.name, usable=sd.trials)


def map_delta(a: AffineMap, b: AffineMap) -> float:
    worst = 0.0
    for f in ("weight", "bias", "mean", "std"):
        worst = max(worst, float(np.max(np.abs(getattr(a, f) - getattr(b, f)))))
    ra, rb = a.readout, b.readout
    if (ra is None) != (rb is None):
        return max(worst, 1.0)
    if ra is not None:
        worst = max(worst, float(np.max(np.abs(ra - rb))))
    return worst


def save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def append_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row)) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


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


def jacobian_sums(trunk: m2.M2Trunk, sessions: dict, *, lmax: int = JBAR_LMAX, width: int = JBAR_WIDTH,
                  n_windows: int = JBAR_WINDOWS, seed: int = 0) -> dict:
    """Per-source sum of lag Jacobians at that session's own folded read-in.

    Windows are drawn from eval-valid bins.  The trunk state of each window starts
    at zero (``average_jacobian_session``).  Autograd runs on source sessions only.
    """
    bank = torch.load(trunk.pool_dir / "session_bank_best.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)
    out = {}
    for j, name in enumerate(trunk.source_ids):
        sd = sessions[name]
        mean = np.asarray(bank["source_statistics"][j][0], np.float64)
        std = np.where(np.asarray(bank["source_statistics"][j][1], np.float64) < 1e-6, 1.0,
                       np.asarray(bank["source_statistics"][j][1], np.float64))
        xz = (sd.x32 - torch.as_tensor(mean, dtype=torch.float32, device=sd.device)) / torch.as_tensor(
            std, dtype=torch.float32, device=sd.device)
        valid = sd.valid_np.copy()
        ends = np.flatnonzero(valid)
        ends = ends[ends >= width - 1]
        if len(ends) < 4:
            raise AssertionError(f"{name}: not enough bins for a Jacobian window")
        take = rng.choice(ends, size=min(n_windows, len(ends)), replace=False)
        table = torch.as_tensor(trunk.src_tables[j], dtype=torch.float32, device=sd.device)
        caller = LatentTrunk(trunk.model, torch.as_tensor(trunk.src_bias[j], dtype=torch.float32, device=sd.device))
        got = fr.average_jacobian_session(caller, xz, table, take, lmax=lmax, width=width, chunk=8)
        out[name] = {"sum": got["sum"].detach().cpu(), "n": int(got["n"])}
    return {"lmax": int(lmax), "sessions": out}


def jbar_from(blob: dict, lmax: int, *, exclude: str | None = None, device=None) -> torch.Tensor:
    """Mean Jacobian ``[lmax+1, D, K]``.  ``exclude`` is the leave-one-out source session."""
    if int(blob["lmax"]) < int(lmax):
        raise AssertionError(f"Jacobian stored to lag {blob['lmax']}, need {lmax}")
    tot, n = None, 0
    for name, s in blob["sessions"].items():
        if exclude is not None and name == exclude:
            continue
        tot = s["sum"] if tot is None else tot + s["sum"]
        n += int(s["n"])
    if n <= 0:
        raise AssertionError("no Jacobian windows")
    j = (tot / n)[: int(lmax) + 1].float()
    if device is not None:
        j = j.to(device)
    return j


def session_gate_for(episodes, eta: float, rho: float, base_key: str) -> dict:
    """Conservative ≥2-session gate.

    η = ρ = 0 is the base map.  Its scored R² differs from the base by float noise, which
    must not be counted as a positive session (and must not set ``deploy``).
    """
    info = sessions_with_positive_gain(episodes, "rebase", float(eta), float(rho), base_key=base_key)
    if abs(float(eta)) < 1e-12 and abs(float(rho)) < 1e-12:
        return {**info, "n_positive_sessions": 0, "positive_sessions": [], "deploy": False,
                "session_gain": {s: 0.0 for s in info.get("session_gain", {})},
                "note": "eta=rho=0 is the base; float noise is not a positive gain"}
    means = info.get("session_gain") or {}
    positive = sorted(s for s, m in means.items() if float(m) > 1e-4)
    return {**info, "positive_sessions": positive, "n_positive_sessions": len(positive),
            "deploy": len(positive) >= 2}


def margin_rules(episodes: list[dict], base_key: str) -> dict:
    """AND (primary) and OR, plus the conservative two-session gate on the AND pick."""
    from ssm_decode.causal_protocol import margin_pick
    by = {}
    for r in episodes:
        by.setdefault(int(r["B"]), []).append(r)
    out = {"b": {}, "protocol": "causal_first_B", "base_key": base_key, "primary_gate": "and"}
    for B, group in sorted(by.items()):
        eta_r2 = float(group[0].get("eta_r2", 1.0))
        chosen = {
            "or": margin_pick(group, "rebase", eta_r2, mode="or", base_key=base_key),
            "and": margin_pick(group, "rebase", eta_r2, mode="and", base_key=base_key),
        }
        for gate, spec in chosen.items():
            spec["session_gate"] = session_gate_for(group, spec["eta"], spec["rho"], base_key)
        out["b"][str(B)] = {**chosen, "n_episodes": len(group),
                            "sessions": sorted({g["session"] for g in group})}
    return out
