"""Labeled-calibration identity encoders (F1-F8, doc 2026-10-07 second revision).

Target-side contract (H1/H2/H4): the backbone stays frozen, calibration
runs under ``torch.no_grad``-equivalent semantics, and only the B labeled
trials' neural data and labels are touched.  The calibration output is a
static read-in table (residual ``T0 + dT`` or absolute) plus per-channel
row gain and offset.  Everything except small streaming accumulations is a
fixed-weight forward: the frozen trunk, and encoder networks trained on
source and frozen at deployment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

LATENT_CORR_DIM = 8
CROSS_LAGS = (-2, -1, 0, 1, 2)
DIR_BINS = 8

STAT_GROUPS = {
    "activity": 5 + LATENT_CORR_DIM,   # rate mean/var, autocorr 1/2/5, latent corr
    "crosscov": 2 * len(CROSS_LAGS),   # behavior cross-covariance, D=2
    "direction": DIR_BINS,             # direction-binned rates
    "t4": 2,                           # cosine/sine tuning coefficients
}


def stat_layout(groups=None):
    groups = groups or list(STAT_GROUPS)
    names, spans = [], {}
    for group in groups:
        spans[group] = (len(names), len(names) + STAT_GROUPS[group])
        names.extend(f"{group}{i}" for i in range(STAT_GROUPS[group]))
    return names, spans


@dataclass
class CalibContext:
    """Everything a calibrator may see on one session."""
    record: object
    bins: np.ndarray                     # labeled bins of the B trials
    budget: int
    prior_table: np.ndarray              # [K, N]
    decode_fn: object                    # decode_fn(table, g, o, bins_slice) -> preds
    x_norm: np.ndarray                   # normalized neural [T, N]
    with_labels: bool = True


def decode_from_base(base, x, table, g=None, o=None):
    """z = (x*g + o) @ table.T then the frozen trunk after its input layer."""
    if g is not None:
        x = x * g.unsqueeze(0) if x.dim() == 3 else x * g
    if o is not None:
        x = x + (o.unsqueeze(0) if x.dim() == 3 else o)
    return base(x @ table.T)


# ---------------------------------------------------------------- stats ----

def _shift(a, k, fill=0.0):
    out = np.full_like(a, fill)
    if k == 0:
        out[:] = a
    elif k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


_SERIES_CACHE = {}


def _series_cache(ctx):
    key = (id(ctx.record), id(ctx.x_norm), id(ctx.prior_table))
    if key not in _SERIES_CACHE:
        full = ctx.x_norm
        entry = {"shift": {k: _shift(full, k) for k in (1, 2, 5, 2, -2, -1, 1, 2)}}
        entry["shift"] = {k: _shift(full, k) for k in (-2, -1, 1, 2, 5)}
        entry["z_all"] = full @ ctx.prior_table.T
        entry["y_all"] = np.asarray(ctx.record.behavior, dtype=np.float64)
        entry["y_shift"] = {lag: _shift(entry["y_all"], lag) for lag in CROSS_LAGS}
        if len(_SERIES_CACHE) > 64:
            _SERIES_CACHE.clear()
        _SERIES_CACHE[key] = entry
    return _SERIES_CACHE[key]


def stat_sums(ctx: CalibContext, groups=None):
    """Streaming accumulators: sums of outer products over labeled bins.

    Every entry is additive over bins, so chunked accumulation equals the
    batch result exactly (the required streaming test).  ``hand_statistics``
    derives normalized statistics from these sums plus the bin count.
    """
    groups = groups or list(STAT_GROUPS)
    cache = _series_cache(ctx)
    full = ctx.x_norm                                     # [T, N]
    bins = np.asarray(ctx.bins)
    x = full[bins].astype(np.float64)                     # [L, N]
    record, n = ctx.record, full.shape[1]
    sums = {}
    sums["s1"] = x.sum(0)                                 # [N] always
    if "activity" in groups:
        sums["s2"] = (x ** 2).sum(0)                      # [N]
        for k in (1, 2, 5):
            sums[f"sac{k}"] = (x * cache["shift"][k][bins]).sum(0)
        z = cache["z_all"][bins]                          # [L, K]
        sums["sz1"] = z.sum(0)                            # [K]
        sums["sz2"] = (z ** 2).sum(0)                     # [K]
        sums["sxz"] = x.T @ z[:, :LATENT_CORR_DIM]        # [N, LAT]
    if "crosscov" in groups:
        y = cache["y_all"][bins]
        for i, lag in enumerate(CROSS_LAGS):
            xs = cache["shift"][-lag][bins] if lag else x
            ys = cache["y_shift"][lag][bins] if lag else y
            sums[f"syx{i}"] = ys.T @ xs                   # [D, N]
        sums["sy1"] = y.sum(0)                            # [D]
    if "direction" in groups:
        vel = record.behavior[bins].astype(np.float64)
        ang = (np.arctan2(vel[:, 1], vel[:, 0]) + 2 * math.pi) % (2 * math.pi)
        binned = np.floor(ang / (2 * math.pi / DIR_BINS)).astype(int) % DIR_BINS
        onehot = np.eye(DIR_BINS)[binned]                 # [L, 8]
        sums["sdir"] = x.T @ onehot                       # [N, 8]
        sums["scount"] = onehot.sum(0)                    # [8]
    if "t4" in groups:
        vel = record.behavior[bins].astype(np.float64)
        ang = (np.arctan2(vel[:, 1], vel[:, 0]) + 2 * math.pi) % (2 * math.pi)
        phi = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang)], 1)  # [L, 3]
        sums["sphi"] = phi.T @ phi                        # [3, 3]
        sums["sphix"] = phi.T @ x                         # [3, N]
    sums["count"] = float(len(bins))
    return sums


def _merge_sums(parts):
    out = {}
    for part in parts:
        for key, value in part.items():
            out[key] = out.get(key, 0) + value
    return out


def hand_statistics(ctx: CalibContext, groups=None):
    """Normalized per-unit statistics derived from the streaming sums."""
    groups = groups or list(STAT_GROUPS)
    sums = stat_sums(ctx, groups)
    n = ctx.x_norm.shape[1]
    L = max(sums["count"], 1.0)
    cols = []
    if "activity" in groups:
        mean = sums["s1"] / L
        var = np.maximum(sums["s2"] / L - mean ** 2, 1e-9)
        ac = [sums[f"sac{k}"] / L / var for k in (1, 2, 5)]
        zm = sums["sz1"] / L
        zv = np.maximum(sums["sz2"] / L - zm ** 2, 1e-9)
        corr = (sums["sxz"] / L - mean[:, None] * zm[None, :LATENT_CORR_DIM]) / \
            np.sqrt(np.maximum(var[:, None] * zv[None, :LATENT_CORR_DIM], 1e-18))
        cols += [mean, np.sqrt(var), *ac, np.clip(corr, -1, 1).T]
    if "crosscov" in groups:
        ym = sums["sy1"] / L
        xmean = sums.get("s1", np.zeros(n)) / L
        for i in range(len(CROSS_LAGS)):
            block = (sums[f"syx{i}"] - ym[:, None] * xmean[None, :]) / L
            for row in block:
                cols.append(row)
    if "direction" in groups:
        dir_rate = sums["sdir"] / np.maximum(sums["scount"], 1e-6)[None, :]
        rates = sums["s1"] / L
        cols.append((dir_rate.T / (rates + 1e-6)))
    if "t4" in groups:
        tgt = np.linalg.solve(sums["sphi"] + 1e-6 * np.eye(3), sums["sphix"])  # [3, N]
        cols += [tgt[1], tgt[2]]
    blocks = [np.atleast_2d(c) for c in cols]
    stats = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, n))
    return stats.T                                        # [N, S]


def hand_statistics_streamed(ctx: CalibContext, chunks=3, groups=None):
    """Chunked accumulation of ``stat_sums`` then one normalization."""
    groups = groups or list(STAT_GROUPS)
    order = np.array_split(np.arange(len(ctx.bins)), max(chunks, 1))
    parts = []
    for part in order:
        if len(part) == 0:
            continue
        sub = CalibContext(ctx.record, np.asarray(ctx.bins)[part], ctx.budget,
                           ctx.prior_table, ctx.decode_fn, ctx.x_norm,
                           ctx.with_labels)
        parts.append(stat_sums(sub, groups))
    merged = _merge_sums(parts)
    whole = CalibContext(ctx.record, np.zeros(0, dtype=int), ctx.budget,
                         ctx.prior_table, ctx.decode_fn, ctx.x_norm,
                         ctx.with_labels)
    # temporarily bypass count guard by hand-building stats from merged sums
    n = ctx.x_norm.shape[1]
    L = max(merged["count"], 1.0)
    cols = []
    if "activity" in groups:
        mean = merged["s1"] / L
        var = np.maximum(merged["s2"] / L - mean ** 2, 1e-9)
        ac = [merged[f"sac{k}"] / L / var for k in (1, 2, 5)]
        zm = merged["sz1"] / L
        zv = np.maximum(merged["sz2"] / L - zm ** 2, 1e-9)
        corr = (merged["sxz"] / L - mean[:, None] * zm[None, :LATENT_CORR_DIM]) / \
            np.sqrt(np.maximum(var[:, None] * zv[None, :LATENT_CORR_DIM], 1e-18))
        cols += [mean, np.sqrt(var), *ac, np.clip(corr, -1, 1).T]
    if "crosscov" in groups:
        ym = merged["sy1"] / L
        xmean = merged.get("s1", np.zeros(n)) / L
        for i in range(len(CROSS_LAGS)):
            block = (merged[f"syx{i}"] - ym[:, None] * xmean[None, :]) / L
            for row in block:
                cols.append(row)
    if "direction" in groups:
        dir_rate = merged["sdir"] / np.maximum(merged["scount"], 1e-6)[None, :]
        rates = merged["s1"] / L
        cols.append((dir_rate.T / (rates + 1e-6)))
    if "t4" in groups:
        tgt = np.linalg.solve(merged["sphi"] + 1e-6 * np.eye(3), merged["sphix"])
        cols += [tgt[1], tgt[2]]
    blocks = [np.atleast_2d(c) for c in cols]
    return (np.concatenate(blocks, axis=0) if blocks else np.zeros((0, n))).T


# ------------------------------------------------------------- encoders ----

def _mlp(in_dim, out_dim, hidden=64, layers=2):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*mods)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class HandEncoder(nn.Module):
    """F2/F3/F5: statistics -> (dT or T columns, g, o)."""

    def __init__(self, stat_dim, k, hidden=64):
        super().__init__()
        self.net = _mlp(stat_dim, k + 2, hidden)

    def forward(self, stats):                            # [N, S]
        out = self.net(stats)
        return out[:, :-2].T, out[:, -2], out[:, -1]     # [K,N], [N], [N]


class PhiNet(nn.Module):
    """F6: shared per-bin feature network."""

    def __init__(self, in_dim, width=64):
        super().__init__()
        self.net = _mlp(in_dim, width, width, layers=2)

    def forward(self, feats):                            # [N, L, in]
        return self.net(feats)


class RhoNet(nn.Module):
    def __init__(self, width, k):
        super().__init__()
        self.net = _mlp(width, k + 2, 64)

    def forward(self, m):                                # [N, width]
        out = self.net(m)
        return out[:, :-2].T, out[:, -2], out[:, -1]


class FeedbackStep(nn.Module):
    """One F7 update: dT_i = eta * G(M_i, q_i, B) / (q_i + eps)."""

    def __init__(self, d_win, k, gtype="linear"):
        super().__init__()
        self.gtype = gtype
        if gtype == "linear":
            self.G = nn.Linear(d_win, k)
            nn.init.zeros_(self.G.weight)
            nn.init.zeros_(self.G.bias)
        else:
            self.G = _mlp(d_win + 2, k, 64)
        self.eta = nn.Parameter(torch.tensor(1e-2))
        self.eps = 1e-3

    def forward(self, m, q, budget):                     # m [N, d_win], q [N]
        if self.gtype == "linear":
            base = self.G(m)
        else:
            bcol = torch.full((len(m), 1), float(budget) / 16.0, device=m.device)
            base = self.G(torch.cat([m, q.unsqueeze(1), bcol], 1))
        return self.eta * base / (q.unsqueeze(1) + self.eps)


class F7Calibrator(nn.Module):
    """Unrolled error-driven updates with a learned feedback map."""

    def __init__(self, d, k, *, steps=1, gtype="linear", lag=2, shared=False):
        super().__init__()
        self.steps, self.lag, self.d, self.k = steps, lag, d, k
        mk = lambda: FeedbackStep(d * (2 * lag + 1), k, gtype)
        self.blocks = nn.ModuleList([mk() for _ in range(1 if shared else steps)])
        self.shared = shared

    def step_module(self, s):
        return self.blocks[0] if self.shared else self.blocks[s]

    def calibrate(self, ctx: CalibContext, *, output_mode="residual"):
        device = next(self.parameters()).device
        x = torch.from_numpy(ctx.x_norm[ctx.bins].astype(np.float32)).to(device)
        y = torch.from_numpy(np.asarray(ctx.record.behavior[ctx.bins],
                                        dtype=np.float32)).to(device)
        if isinstance(ctx.prior_table, torch.Tensor):
            table = ctx.prior_table.detach().clone().float().to(device)
        else:
            table = torch.from_numpy(ctx.prior_table.astype(np.float32)).to(device)
        if x.shape[0] < 2 * self.lag + 2:
            return None
        T = table.clone()
        for s in range(self.steps):
            preds = ctx.decode_fn(T)                       # [L, D] labeled bins
            r = y - preds
            win = 2 * self.lag + 1
            r_pad = torch.cat([torch.zeros(self.lag, r.shape[1], device=r.device), r,
                               torch.zeros(self.lag, r.shape[1], device=r.device)], 0)
            r_tilde = torch.stack([r_pad[i:i + len(r)] for i in range(win)], 1)  # [L, win, D]
            r_tilde = r_tilde.reshape(len(r), -1)          # [L, win*D]
            m = (r_tilde.unsqueeze(1) * x.unsqueeze(2)).mean(0)   # [N, win*D]
            q = (x ** 2).mean(0)                                   # [N]
            dT = self.step_module(s)(m, q, ctx.budget)             # [N, K]
            T = T + dT.T
        if torch.is_grad_enabled() and T.requires_grad:
            return T, None, None
        return T.detach().cpu().numpy().astype(np.float64), None, None


class InContextAttention(nn.Module):
    """F8: per-unit sequence transformer (000688 only)."""

    def __init__(self, in_dim, k, width=64, heads=4, layers=2, max_bins=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = _mlp(width, k + 2, 64)
        self.max_bins = max_bins

    def forward(self, seq):                              # [N, L, in]
        z = self.enc(self.proj(seq))
        m = z.mean(1)
        o = self.out(m)
        return o[:, :-2].T, o[:, -2], o[:, -1]


VARIANTS = ("f1", "f2", "f3", "f5", "f6", "f7", "f7f3", "f8")


class IdentityCalibrator(nn.Module):
    """Dispatcher.  ``calibrate`` is the target-side entry (no grad)."""

    def __init__(self, variant, *, k, stat_dim=None, phi_in=None,
                 output_mode="residual", f7=None, drop_groups=None, width=64,
                 f8_in=None):
        super().__init__()
        self.variant, self.k, self.output_mode = variant, k, output_mode
        self.drop_groups = drop_groups or []
        groups = [g for g in STAT_GROUPS if g not in self.drop_groups]
        self.groups = groups
        _, spans = stat_layout(groups)
        self.stat_span = spans
        eff_dim = sum(STAT_GROUPS[g] for g in groups)
        self.hand = None
        self.phi = None
        self.rho = None
        self.f7 = None
        self.f8 = None
        if variant in ("f1", "f2", "f3", "f5", "f7f3"):
            self.hand = HandEncoder(eff_dim, k, width)
        if variant == "f6":
            self.phi = PhiNet(phi_in, width)
            self.rho = RhoNet(width, k)
        if variant in ("f7", "f7f3"):
            self.f7 = f7
        if variant == "f8":
            self.f8 = InContextAttention(f8_in or phi_in, k)

    def _numpy(self, t):
        return t.detach().cpu().numpy().astype(np.float64)

    def _out(self, t):
        """Keep the graph during source training; numpy on the target side."""
        if torch.is_grad_enabled() and t.requires_grad:
            return t
        return self._numpy(t)

    @staticmethod
    def _gain_offset(g_raw, o_raw):
        """Zero-initialised heads must start at identity row parameters."""
        return 1.0 + g_raw, o_raw

    def _apply_hand(self, ctx):
        stats = hand_statistics(ctx, self.groups)
        stats = (stats - stats.mean(0)) / (stats.std(0) + 1e-6)
        device = next(self.parameters()).device
        st = torch.from_numpy(stats.astype(np.float32)).to(device)
        dT, g_raw, o_raw = self.hand(st)
        g, o = self._gain_offset(self._out(g_raw), self._out(o_raw))
        return self._out(dT), g, o

    def _apply_f6(self, ctx):
        x = ctx.x_norm                                   # [T, N]
        idx = np.asarray(ctx.bins)
        pre = 4
        pad = np.pad(x, ((pre, 0), (0, 0)), mode="constant")
        y = ctx.record.behavior.astype(np.float64)
        ypad = np.pad(y, ((2, 2), (0, 0)), mode="constant")
        z0 = x @ ctx.prior_table.T[:, :LATENT_CORR_DIM]
        # rows: [x_{t-4..t} per unit flattened per unit, y window, z0]
        per_unit = []
        for i in range(x.shape[1]):
            rows = []
            for t in idx:
                rows.append(np.concatenate([pad[t:t + pre + 1, i],
                                            ypad[t:t + 5].reshape(-1),
                                            z0[t]]))
            per_unit.append(np.stack(rows))              # [L, in]
        device = next(self.parameters()).device
        f = torch.from_numpy(np.stack(per_unit).astype(np.float32)).to(device)
        m = self.phi(f).mean(1)
        dT, g_raw, o_raw = self.rho(m)
        g, o = self._gain_offset(self._out(g_raw), self._out(o_raw))
        return self._out(dT), g, o

    def _apply_f8(self, ctx):
        x = ctx.x_norm
        y = ctx.record.behavior.astype(np.float64)
        z0 = x @ ctx.prior_table.T[:, :LATENT_CORR_DIM]
        idx = ctx.bins
        if len(idx) > self.f8.max_bins:
            idx = np.linspace(0, len(ctx.bins) - 1, self.f8.max_bins).astype(int)
            idx = ctx.bins[idx]
        seq = np.stack([np.stack([np.concatenate([x[t:t + 1].reshape(-1)[i:i + 1],
                                                  y[t],
                                                  z0[t]]) for t in idx], 0)
                        for i in range(x.shape[1])])
        device = next(self.parameters()).device
        s = torch.from_numpy(seq.astype(np.float32)).to(device)
        dT, g, o = self.f8(s)
        return dT.detach().cpu().numpy().astype(np.float64), \
            g.detach().cpu().numpy().astype(np.float64), \
            o.detach().cpu().numpy().astype(np.float64)

    def calibrate(self, ctx: CalibContext):
        """Target-side: no gradients touch anything (H2)."""
        with torch.no_grad():
            return self._calibrate_grad(ctx)

    def _calibrate_grad(self, ctx: CalibContext):
        if self.variant == "f1":
            sub = CalibContext(ctx.record, ctx.bins, ctx.budget, ctx.prior_table,
                               ctx.decode_fn, ctx.x_norm, with_labels=False)
            stats = hand_statistics(sub, self.groups)
            stats = (stats - stats.mean(0)) / (stats.std(0) + 1e-6)
            device = next(self.parameters()).device
            st = torch.from_numpy(stats.astype(np.float32)).to(device)
            T, _, _ = self.hand(st)
            return self._out(T), None, None  # f1: absolute table, no row params
        if self.variant in ("f2", "f3", "f5"):
            dT, g, o = self._apply_hand(ctx)
            if self.variant == "f2":
                return dT, g, o
            return dT, g, o                                # residual applied by caller
        if self.variant == "f6":
            return self._apply_f6(ctx)
        if self.variant in ("f7", "f7f3"):
            if self.variant == "f7f3":
                dT0, g0, o0 = self._apply_hand(ctx)
                if torch.is_tensor(dT0):
                    prior = torch.as_tensor(ctx.prior_table,
                                            dtype=dT0.dtype,
                                            device=dT0.device) + dT0
                else:
                    prior = np.asarray(ctx.prior_table) + dT0
                sub = CalibContext(ctx.record, ctx.bins, ctx.budget, prior,
                                   ctx.decode_fn, ctx.x_norm, ctx.with_labels)
                out = self.f7.calibrate(sub)
                if out is None:
                    return dT0, g0, o0
                T = out[0]
                return T, g0, o0
            out = self.f7.calibrate(ctx)
            if out is None:
                return np.zeros_like(ctx.prior_table), None, None
            return out
        if self.variant == "f8":
            return self._apply_f8(ctx)
        raise ValueError(self.variant)
