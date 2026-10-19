"""Deployment-faithful causal-first-B calibration schedule.

Calibration trials are the first B usable trials of the session in recorded
order. The query is every later usable trial. A table computed from the first
k calibration trials is not installed until a commit boundary:

* latency 0 (default): the start of usable trial k (the next calibration trial,
  or the first query trial when k = B);
* latency L: L usable trials later than that.

Bins before the first usable trial use a cold map (electrode prior, mean 0,
std 1). No target bin enters that map. Trial 0 may use one-trial staging for
its own normalisation (and, where a method documents it, its own P0): the
trial is buffered before its first bin is decoded. That is the only legal
dependence on a bin at or after the decode time.

``install='final_only'`` is the reference commit (R2, R3, ridge-CCA, ui_reg):
nothing but the cold map, the initial map, and the final map is installed.
The final map still waits for the same boundary as an anytime method's commit B.

This module does not change how a method turns trials into a table. It only
decides when that table may be read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLD = -1
INITIAL = 0
MARGIN_R2 = 0.005


@dataclass(frozen=True)
class AffineMap:
    """One installed read-in map. ``max_bin_read`` is inclusive, or -1 if no target bin was read."""

    weight: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    max_bin_read: int = -1
    staged_trial0: bool = False
    readout: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "weight", np.asarray(self.weight, dtype=np.float64))
        object.__setattr__(self, "bias", np.asarray(self.bias, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "std", np.asarray(self.std, dtype=np.float64).reshape(-1))
        if self.readout is not None:
            object.__setattr__(self, "readout", np.asarray(self.readout, dtype=np.float64))


def assert_recorded_order(usable) -> list[tuple[int, int]]:
    """Usable trials as ``(start, end)`` half-open spans, recorded order, no overlap."""
    spans = [(int(a), int(b)) for a, b in usable]
    if any(b <= a for a, b in spans):
        raise ValueError(f"trial span must be non-empty and forward, got {spans[:4]}")
    starts = [a for a, _ in spans]
    if starts != sorted(starts):
        raise ValueError("usable trials are not in recorded order")
    for i in range(len(spans) - 1):
        if spans[i][1] > spans[i + 1][0]:
            raise ValueError(f"usable trials overlap: {spans[i]} vs {spans[i + 1]}")
    return spans


def calibration_indices(n_usable: int, B: int) -> np.ndarray:
    """The calibration set is exactly the first B usable trials. Not a draw."""
    if int(B) < 1:
        raise ValueError(f"B must be >= 1, got {B}")
    if int(B) > int(n_usable):
        raise ValueError(f"B={B} exceeds {n_usable} usable trials")
    return np.arange(int(B), dtype=np.int64)


def calibration_spans(usable, B: int) -> list[tuple[int, int]]:
    spans = assert_recorded_order(usable)
    idx = calibration_indices(len(spans), B)
    return [spans[int(i)] for i in idx]


def query_indices(n_usable: int, B: int) -> np.ndarray:
    calibration_indices(n_usable, B)
    return np.arange(int(B), int(n_usable), dtype=np.int64)


def _activation_bin(spans, k: int, latency: int, n_bins: int) -> int:
    """First bin at which commit ``k`` (k >= 1) may be read."""
    if k < 1:
        raise ValueError("activation is defined for commits k >= 1")
    if latency < 0:
        raise ValueError("latency must be >= 0")
    j = int(k) + int(latency)          # usable-trial index
    if j < len(spans):
        return int(spans[j][0])
    if not spans:
        return int(n_bins)
    return min(int(n_bins), int(spans[-1][1]))


def commit_segments(usable, B: int, n_bins: int, *, latency: int = 0,
                    staged_trial0: bool = True, install: str = "anytime"):
    """Half-open ``(start, end, commit_k)`` covering ``[0, n_bins)``.

    ``commit_k`` is -1 (cold), 0 (initial / trial-0 stage), or 1..B.
    """
    if install not in ("anytime", "final_only"):
        raise ValueError(install)
    spans = assert_recorded_order(usable)
    B = int(B)
    n_bins = int(n_bins)
    latency = int(latency)
    if B > len(spans):
        raise ValueError(f"B={B} exceeds {len(spans)} usable trials")
    if n_bins < 0:
        raise ValueError("n_bins")
    events: dict[int, int] = {}

    def put(bin_at: int, k: int):
        bin_at = int(bin_at)
        if bin_at < 0 or bin_at > n_bins:
            raise ValueError(f"commit boundary {bin_at} outside [0, {n_bins}]")
        prev = events.get(bin_at)
        events[bin_at] = k if prev is None else max(prev, k)

    put(0, COLD)
    if spans:
        if staged_trial0:
            put(spans[0][0], INITIAL)
        else:
            put(0, INITIAL)
    commits = (B,) if install == "final_only" else tuple(range(1, B + 1))
    for k in commits:
        put(_activation_bin(spans, k, latency, n_bins), k)
    bounds = sorted(events)
    segments = []
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else n_bins
        if end > start:
            segments.append((int(start), int(end), int(events[start])))
    covered = sum(e - a for a, e, _ in segments)
    if covered != n_bins or (segments and segments[0][0] != 0) or (n_bins and segments[-1][1] != n_bins):
        raise AssertionError(f"segments do not cover [0, {n_bins}): {segments[:6]}")
    return segments


def assert_segment_causal(segments, usable, max_bin_read: dict, *, staged_trial0: bool = True):
    """No installed map at time t depends on a bin after t, except trial-0 staging.

    ``max_bin_read[k]`` is the inclusive last target bin that built commit k.
    Cold maps must not have read any target bin.
    """
    spans = assert_recorded_order(usable)
    trial0 = spans[0] if spans else None
    for start, end, k in segments:
        if k not in max_bin_read:
            raise AssertionError(f"segment {start}:{end} uses commit {k} with no read record")
        read = int(max_bin_read[k])
        if k == COLD:
            if read >= 0:
                raise AssertionError(f"cold map read target bin {read}")
            continue
        if staged_trial0 and k == INITIAL and trial0 is not None and start >= trial0[0]:
            # The whole of trial 0 may be staged. Bins after trial 0 in this same
            # segment are past that read, so the same bound is still causal there.
            if read >= trial0[1]:
                raise AssertionError(
                    f"trial-0 staging read bin {read}, past the trial end {trial0[1]}")
            continue
        if read >= start:
            raise AssertionError(
                f"commit {k} on [{start}, {end}) read bin {read} (not strictly before the segment)")


def query_bins_on_commit(segments, query_mask, commit_k: int) -> tuple[int, int]:
    """(bins of the query whose installed commit is ``commit_k``, total query bins)."""
    mask = np.asarray(query_mask, dtype=bool)
    on = 0
    for start, end, k in segments:
        if k == int(commit_k):
            on += int(mask[start:end].sum())
    return on, int(mask.sum())


def latent_from_maps(x: np.ndarray, segments, maps: dict) -> np.ndarray:
    """Piecewise ``z_t = W (x - mean) / std + bias``. ``x`` is ``[T, N]``."""
    x = np.asarray(x, dtype=np.float64)
    weight0 = maps[segments[0][2]].weight
    z = np.zeros((x.shape[0], weight0.shape[0]), dtype=np.float64)
    for start, end, k in segments:
        amap = maps[k]
        scale = np.where(amap.std > 0, amap.std, 1.0)
        xz = (x[start:end] - amap.mean) / scale
        z[start:end] = xz @ amap.weight.T
        if amap.bias is not None:
            z[start:end] += amap.bias
    return z


def measured_read_bin(spans, idx) -> int:
    """Inclusive last bin of the trials actually named by ``idx``.

    This is not ``spans[k - 1]``: it is the max end among the indices the caller passed.
    An index outside the prefix therefore moves the returned bin.
    """
    idx = [int(i) for i in np.asarray(idx).tolist()]
    if not idx:
        return -1
    if any(i < 0 or i >= len(spans) for i in idx):
        raise AssertionError(f"trial index {idx} outside {len(spans)} usable trials")
    return int(max(spans[i][1] for i in idx) - 1)


def assert_prefix_idx(idx, k: int) -> None:
    """Commit k may name only usable trials ``0 .. k-1``."""
    got = [int(i) for i in np.asarray(idx).tolist()]
    extra = sorted(set(got) - set(range(int(k))))
    if extra or sorted(got) != list(range(int(k))):
        raise AssertionError(f"commit {k} used trials {got}, expected 0..{int(k) - 1}")


def sessions_with_positive_gain(episodes, score_key: str, eta: float, rho: float, *,
                                 base_key: str = "p0") -> dict:
    """How many distinct selection sessions have a positive mean gain at this (eta, rho).

    The AND gate's " > 2 SE" leg is not a session-level test: with one selection session
    the SE is across P0 seeds (and is 0 when the two episodes are identical, as for a
    seed-free method). This count is the conservative alternative: deploy the step only
    when at least two sessions are positive, otherwise fall back to the base.
    """
    by = {}
    for ep in episodes:
        hit = [s for s in ep[score_key]
               if abs(float(s["eta"]) - float(eta)) < 1e-9 and abs(float(s["rho"]) - float(rho)) < 1e-9]
        # eta=rho=0 is the fallback and was not always on the stored grid. Gain is then 0.
        gain = 0.0 if not hit else float(hit[0]["r2"]) - float(ep[base_key])
        by.setdefault(str(ep["session"]), []).append(gain)
    means = {s: float(np.mean(v)) for s, v in by.items()}
    positive = sorted(s for s, m in means.items() if m > 0.0)
    return {"n_sessions": len(means), "n_positive_sessions": len(positive),
            "positive_sessions": positive,
            "session_gain": means,
            "deploy": len(positive) >= 2}


def margin_pick(episodes, score_key: str, eta_r2: float, *, margin: float = MARGIN_R2,
                mode: str = "and", base_key: str = "p0") -> dict:
    """Source-only (eta, rho) pick.

    Gain is ``score - episode[base_key]``. For S1 the base is P0. For COMBO and
    CCA+S1 the base is that method's own closed-form table, not P0.

    ``mode='and'`` (primary): mean gain >= margin AND > 2 SE of the selection episodes.
    That SE is across episodes. With a single selection session those episodes are
    P0 seeds (or two identical copies for a seed-free method, so SE = 0 and the test
    reduces to gain >= margin). It is not a session-level test.
    ``mode='or'`` is the sensitivity: either leg is enough.
    Ties prefer the eta closest to the R2 rule. A non-positive gain never deploys,
    so η = 0 wins only as the fallback.
    """
    if mode not in ("or", "and"):
        raise ValueError(mode)
    if not episodes:
        raise ValueError("no source episodes")
    keys = [(float(s["eta"]), float(s["rho"])) for s in episodes[0][score_key]]
    stats = {}
    for key in keys:
        diffs = []
        for ep in episodes:
            sc = next(s["r2"] for s in ep[score_key]
                      if (float(s["eta"]), float(s["rho"])) == key)
            diffs.append(float(sc) - float(ep[base_key]))
        diffs_a = np.asarray(diffs, dtype=np.float64)
        mean = float(diffs_a.mean())
        se = float(diffs_a.std(ddof=1) / np.sqrt(len(diffs_a))) if len(diffs_a) > 1 else float("inf")
        stats[key] = {"mean_gain": mean, "se": se}
    def _ok(st):
        above = st["mean_gain"] >= float(margin)
        se_ok = st["mean_gain"] > 2.0 * st["se"]
        gate = (above and se_ok) if mode == "and" else (above or se_ok)
        return st["mean_gain"] > 0.0 and gate
    viable = [k for k, st in stats.items() if _ok(st)]
    base_vals = [float(e[base_key]) for e in episodes]
    base = {"margin": float(margin), "mode": mode, "n_episodes": len(episodes),
            "base_key": base_key,
            "p0_source_r2": float(np.mean([float(e["p0"]) for e in episodes])),
            "base_source_r2": float(np.mean(base_vals))}
    if not viable:
        return {**base, "eta": 0.0, "rho": 0.0, "fallback_base": True, "n_viable": 0,
                "source_gain": 0.0, "source_se": None,
                "best_rejected_gain": max(st["mean_gain"] for st in stats.values())}
    best_gain = max(stats[k]["mean_gain"] for k in viable)
    tied = [k for k in viable if abs(stats[k]["mean_gain"] - best_gain) <= 1e-12]
    eta, rho = min(tied, key=lambda k: (abs(k[0] - float(eta_r2)), k[0], k[1]))
    return {**base, "eta": eta, "rho": rho, "fallback_base": False, "n_viable": len(viable),
            "source_gain": stats[(eta, rho)]["mean_gain"], "source_se": stats[(eta, rho)]["se"],
            "source_r2": base["base_source_r2"] + stats[(eta, rho)]["mean_gain"]}


def run_rebased_many(base, raw_trials, prior, configs, prepare, *, mode: str = "rebase",
                     kind: str = "fir", jbar=None, eps: float = 1e-3, base_bias=None):
    """Several (eta, rho) rebased loops on one shared prefix schedule.

    The trunk forward is batched across configs. Each config matches
    ``rebased_feedback_loop`` (the data term and the shrinkage are per config).
    ``configs`` is a list of ``(eta, rho)``. Returns one dict per config, in order,
    with the same keys as the scalar loop.
    """
    import torch

    from ssm_decode import stream_f7 as sf

    if kind != "fir":
        raise ValueError("the batched loop implements the deployed FIR kernel only")
    if mode not in ("rebase", "staged"):
        raise ValueError(mode)
    if not configs:
        raise ValueError("no configs")
    if jbar is None:
        raise ValueError("jbar is required")
    device = prior.device
    jbar = jbar.to(device=device)
    m = len(configs)
    etas = prior.new_tensor([float(e) for e, _ in configs])
    rhos = prior.new_tensor([float(r) for _, r in configs])
    delta = prior.new_zeros((m,) + tuple(prior.shape))
    bias = prior.new_zeros((m, prior.shape[0]))
    coord = None
    p0_final = None
    with torch.no_grad():
        for k, (x_raw, y, mask) in enumerate(raw_trials):
            if mode == "staged":
                mean, std, base_table = prepare(k, True)
            elif k == 0:
                mean, std, _ = prepare(0, False)
                base_table = prior
            else:
                mean, std, base_table = prepare(k, False)
            if coord is None:
                d_use, b_use = delta, bias
            else:
                d_rows, b_rows = [], []
                for i in range(m):
                    d_i, b_i = sf.shift_correction(delta[i], bias[i], coord[0], coord[1], mean, std)
                    d_rows.append(d_i)
                    b_rows.append(b_i)
                d_use, b_use = torch.stack(d_rows, 0), torch.stack(b_rows, 0)
            table = base_table.unsqueeze(0) + d_use
            xz = (x_raw - mean.to(dtype=x_raw.dtype)) / std.to(dtype=x_raw.dtype).clamp_min(1e-8)
            b_forward = b_use
            if base_bias is not None:
                extra = base_bias(k, mode == "staged")
                if extra is not None:
                    b_forward = b_use + extra.to(dtype=b_use.dtype, device=b_use.device)
            z = (torch.einsum("tn,mkn->mtk", xz, table.to(dtype=xz.dtype))
                 + b_forward.to(dtype=xz.dtype)[:, None, :])
            yhat = base(z)
            y_t = y.to(dtype=yhat.dtype)
            r = (y_t.unsqueeze(0) - yhat) * mask.to(dtype=yhat.dtype).view(1, -1, 1)
            n, sum_x2 = sf.valid_sum_x2(xz, mask)
            mean_a, std_a, p0_a = prepare(k, True)
            new_delta, new_bias = [], []
            for i in range(m):
                if float(n) > 0:
                    u = sf.fir_data_term(xz, r[i], jbar)
                    step = sf.feedback_step(u, sum_x2, n, eps)
                else:
                    step = torch.zeros_like(d_use[i])
                d_c, b_c = sf.shift_correction(d_use[i], b_use[i], mean, std, mean_a, std_a)
                s_c, b_s = sf.shift_correction(step, torch.zeros_like(bias[i]), mean, std, mean_a, std_a)
                new_delta.append((1.0 - rhos[i]) * d_c + etas[i] * s_c)
                new_bias.append((1.0 - rhos[i]) * b_c + etas[i] * b_s)
            delta = torch.stack(new_delta, 0)
            bias = torch.stack(new_bias, 0)
            coord = (mean_a, std_a)
            p0_final = p0_a
    table = p0_final.unsqueeze(0) + delta
    out = []
    for i in range(m):
        out.append({"table": table[i], "bias": bias[i], "mean": coord[0], "std": coord[1],
                    "delta": delta[i], "eta": float(etas[i]), "rho": float(rhos[i])})
    return out


# Trunk dates used by the two alpha protocols. "last" is the latest session the trunk
# was trained on. A trunk may contribute episodes only when that date is strictly before
# the target. "train" lists those sessions so the strict protocol can see whether the
# trunk was fit on any reported target.
TRUNK_LAST = {
    "days": "20150615", "month": "20140306", "year": "20140306",
    "sc_0629": "20150313", "sc_0708": "20150706",
}
TRUNK_TRAIN = {
    "days": {"20150511", "20150512", "20150610", "20150611", "20150612", "20150615"},
    "month": set(), "year": set(),
    "sc_0629": set(),
    "sc_0708": {"20150629", "20150630", "20150701", "20150703", "20150706"},
}


def alpha_episode_allowed(row: dict, target_date: str, target_subject: str, variant: str,
                          reported_targets: set) -> bool:
    """Whether one source episode may vote on α for this target.

    ``expanding`` (deployment-faithful): the episode session and the trunk's last training
    session are both strictly before the target. An earlier reported target is a legal
    source: at deployment it is a past, labeled session.
    ``strict``: also drop every session that is a reported target (either subject) and
    every trunk that was trained on a reported target.
    """
    if variant not in ("expanding", "strict"):
        raise ValueError(variant)
    session = str(row["session"])
    date = session[-8:]
    tag = str(row["tag"])
    if date >= target_date or TRUNK_LAST[tag] >= target_date:
        return False
    if variant == "strict":
        if date in reported_targets:
            return False
        if set(TRUNK_TRAIN.get(tag, ())) & set(reported_targets):
            return False
    return True


def select_alpha(episodes, target_date: str, target_subject: str, gap_days: int, B: int,
                 variant: str, reported_targets: set) -> dict:
    """α for one target from the episodes that protocol ``variant`` allows.

    The target's own gap bin is fixed (days from its trunk). Episodes are binned by the
    gap they were scored under. Empty bins fall back with ``resolve_gap_bin``.
    """
    from ssm_decode.p0solve import ALPHAS, gap_bin_name, resolve_gap_bin
    want = gap_bin_name(int(gap_days))
    pool = [r for r in episodes
            if alpha_episode_allowed(r, target_date, target_subject, variant, reported_targets)]
    bins = {}
    for r in pool:
        if int(r["B"]) != int(B) or r.get("gap_days") is None:
            continue
        bins.setdefault(gap_bin_name(int(r["gap_days"])), []).append(r)
    if not bins:
        return {"alpha": None, "bin": want, "bin_used": None, "n_episodes": 0,
                "sessions": [], "tags": [], "fallback_from": want}
    used, fell = resolve_gap_bin(want, set(bins))
    group = bins[used]
    means = {}
    for alpha in ALPHAS:
        vals = [float(h["r2"]) for ep in group for h in ep["alphas"]
                if float(h["alpha"]) == float(alpha) and h.get("r2") is not None and float(h["r2"]) > -0.5]
        means[float(alpha)] = float(np.mean(vals)) if vals else -1.0
    # Smaller alpha wins an exact tie (do not turn shrinkage on for nothing).
    winner = sorted(means, key=lambda a: (means[a], -a), reverse=True)[0]
    sessions = sorted({str(r["session"])[-8:] for r in group})
    tags = sorted({str(r["tag"]) for r in group})
    return {"alpha": float(winner), "source_r2": means[winner], "bin": want, "bin_used": used,
            "fallback_from": fell, "n_episodes": len(group), "sessions": sessions, "tags": tags,
            "cross_subject": any(not s.startswith(target_subject) for s in {r["session"] for r in group})}
