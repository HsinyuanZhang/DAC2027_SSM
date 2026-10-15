"""Population-conditioned static calibration (feasibility P1, work line A).

Target-side contract (same as the labeled-calibration round): the trunk is
frozen; calibration is (a) streaming accumulation of additive sufficient
statistics over the B labeled trials, (b) a fixed-weight forward of a
calibrator trained on source sessions, (c) a small closed-form solve (3x3
T4 coefficients) and an SRAM write of the static K x N read-in table.  No
gradient, no backprop, no iterative optimisation touches target data.

Variants
--------
p0  independent per-unit encoder on hand statistics (round-1 F2/F3 style),
    except that feature standardisation uses fixed source-fitted constants
    instead of cross-unit statistics, so no population information leaks in.
p1  p0 + reliability shrinkage  d~_i = lam_i d_i + (1 - lam_i) mu0, with
    lam_i = sigmoid(MLP(reliability features)), mu0 a learned constant.
p2  p1 + DeepSets context  mu = mean_i psi(d~_i); generator g(d~_i, mu).
p3  PMA with m learned seeds; scores s_ji = u_j . phi(d~_i) depend on unit i
    only (online-softmax compatible); per-unit fixed routing
    beta_ij = softmax_j(a_j . phi(d~_i)); mu_i = sum_j beta_ij z_j;
    generator g(d~_i, mu_i, lam_i).
p4  p3 with log(lam_i) added to the pooling scores (reliability weighting).

``calibrate(..., oracle=True)`` marks the result ``diagnostic_only``: the
descriptors then come from ALL labeled trials of the target session.  Such
a result can never be selected as a candidate (``assert_candidate``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

AC_LAGS = (1, 2, 5)
CROSS_LAGS = (-2, -1, 0, 1, 2)
LAG0 = CROSS_LAGS.index(0)
DIR_BINS = 8
D_OUT = 2
VARIANTS = ("p0", "p1", "p2", "p3", "p4")
STD_FLOOR = 1e-6          # matches track_a_experiment.support_stats

_UNIT_AXIS_KEYS = {        # key -> axes that index units
    "sx": (0,), "gram": (0, 1), "ac_xx": (1,), "ac_a": (1,), "ac_b": (1,),
    "syx": (2,), "sdir": (0,), "sphix": (1,), "q_sx2": (0,), "q_yx2": (1,),
    "q_yxy": (1,),
}
_ADDITIVE_KEYS = ("n", "sx", "gram", "ac_xx", "ac_a", "ac_b", "ac_n", "sy", "syx",
                  "yx_n", "sdir", "cdir", "sphi", "sphiphi", "sphix")
_QUAD_KEYS = ("q_sx2", "q_yx2", "q_yxy", "q_y2")


# ------------------------------------------------------------ trial bank ----

def segment_sums(x, y, ang_bin, valid):
    """Raw additive sums of ONE labeled trial segment.

    x [T, N] counts, y [T, D] standardised behavior, ang_bin [T] direction
    bin ids (int), valid [T] bool.  Lag pairs require both bins valid and
    inside the segment, so sums never straddle a trial boundary.
    """
    v = valid.to(x.dtype)
    t_len = x.shape[0]
    n_units = x.shape[1]
    y = torch.where(valid.unsqueeze(1), y, torch.zeros_like(y))
    out = {"n": v.sum(), "sx": (v.unsqueeze(1) * x).sum(0),
           "gram": x.T @ (v.unsqueeze(1) * x)}
    ac_xx, ac_a, ac_b, ac_n = [], [], [], []
    for k in AC_LAGS:
        if t_len > k:
            vv = v[k:] * v[:-k]
            xt, xl = x[k:], x[:-k]
            ac_xx.append((vv.unsqueeze(1) * xt * xl).sum(0))
            ac_a.append((vv.unsqueeze(1) * xt).sum(0))
            ac_b.append((vv.unsqueeze(1) * xl).sum(0))
            ac_n.append(vv.sum())
        else:
            z = torch.zeros(n_units, dtype=x.dtype, device=x.device)
            ac_xx.append(z), ac_a.append(z), ac_b.append(z)
            ac_n.append(torch.zeros((), dtype=x.dtype, device=x.device))
    out["ac_xx"], out["ac_a"], out["ac_b"] = (torch.stack(ac_xx), torch.stack(ac_a),
                                              torch.stack(ac_b))
    out["ac_n"] = torch.stack(ac_n)
    sy, syx, yx_n = [], [], []
    for lag in CROSS_LAGS:                       # pairs (x_t, y_{t+lag})
        if lag >= 0:
            xt, yt, vv = x[:t_len - lag], y[lag:], v[:t_len - lag] * v[lag:]
        else:
            xt, yt, vv = x[-lag:], y[:t_len + lag], v[-lag:] * v[:t_len + lag]
        sy.append((vv.unsqueeze(1) * yt).sum(0))
        syx.append((vv.unsqueeze(1) * yt).T @ xt)
        yx_n.append(vv.sum())
    out["sy"], out["syx"], out["yx_n"] = torch.stack(sy), torch.stack(syx), torch.stack(yx_n)
    onehot = torch.nn.functional.one_hot(ang_bin.long(), DIR_BINS).to(x.dtype) * v.unsqueeze(1)
    out["sdir"], out["cdir"] = x.T @ onehot, onehot.sum(0)
    ang = (ang_bin.to(x.dtype) + 0.5) * (2 * math.pi / DIR_BINS)
    phi = torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang)], 1) * v.unsqueeze(1)
    out["sphi"] = phi.sum(0)
    out["sphiphi"] = phi.T @ torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang)], 1)
    out["sphix"] = phi.T @ x
    return out


class TrialBank:
    """Per-trial raw sums of every usable trial of one session (additive)."""

    def __init__(self, rows: list[dict]):
        self.t = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        self.n_trials = len(rows)

    @classmethod
    def build(cls, x, y_norm, ang_bin, valid, trials, device=None):
        """x [T, N] float64, y_norm [T, D], ang_bin [T] int, valid [T] bool."""
        rows = [segment_sums(x[a:b], y_norm[a:b], ang_bin[a:b], valid[a:b])
                for a, b in trials]
        return cls(rows)


@dataclass
class PopStats:
    """Streaming accumulators over a set of labeled trials.

    ``add`` merges disjoint chunks exactly (chunked == batch), which is the
    on-chip accumulation contract.  Quadratic keys (q_*) carry per-trial
    energy for the cross-trial consistency features.
    """
    d: dict
    n_trials: int

    @staticmethod
    def from_trials(bank: TrialBank, idx) -> "PopStats":
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=bank.t["n"].device)
        t = bank.t
        d = {k: t[k][idx].sum(0) for k in _ADDITIVE_KEYS}
        nb = t["n"][idx].clamp_min(1.0)
        sx, syx0, sy0 = t["sx"][idx], t["syx"][idx][:, LAG0], t["sy"][idx][:, LAG0]
        d["q_sx2"] = (sx ** 2 / nb[:, None]).sum(0)
        d["q_yx2"] = (syx0 ** 2 / nb[:, None, None]).sum(0)
        d["q_yxy"] = (syx0 * sy0[:, :, None] / nb[:, None, None]).sum(0)
        d["q_y2"] = (sy0 ** 2 / nb[:, None]).sum(0)
        return PopStats(d, int(len(idx)))

    def __add__(self, other: "PopStats") -> "PopStats":
        return PopStats({k: self.d[k] + other.d[k] for k in self.d},
                        self.n_trials + other.n_trials)

    def select_units(self, keep) -> "PopStats":
        keep = torch.as_tensor(np.asarray(keep), dtype=torch.long, device=self.d["n"].device)
        out = {}
        for k, v in self.d.items():
            for ax in _UNIT_AXIS_KEYS.get(k, ()):
                v = v.index_select(ax, keep)
            out[k] = v
        return PopStats(out, self.n_trials)

    @property
    def n_units(self) -> int:
        return int(self.d["sx"].shape[0])


# ------------------------------------------------------------ descriptors ----

def desc_dim(k: int) -> int:
    # log_rate, log_std, 3 autocorr, K latent corr, 5*D cross-cov, 8 dir, 2 t4,
    # K prior column, log n, log B
    return 2 + len(AC_LAGS) + k + len(CROSS_LAGS) * D_OUT + DIR_BINS + 2 + k + 2


REL_DIM = 6


def label_columns(k: int):
    """Descriptor / reliability columns that carry behavior-label information."""
    c0 = 2 + len(AC_LAGS) + k                    # start of cross-covariance block
    cols = list(range(c0, c0 + len(CROSS_LAGS) * D_OUT + DIR_BINS + 2))
    return cols, [4, 5]                          # descriptor cols, reliability cols (kappa)


PRIOR_MODES = ("aligned", "globalmean", "zero")


def corr_columns(k: int):
    """Descriptor columns holding the correlation with the prior latents."""
    c0 = 2 + len(AC_LAGS)
    return list(range(c0, c0 + k))


def prior_columns(k: int):
    """Descriptor columns holding the (electrode-aligned) prior column itself."""
    c0 = 2 + len(AC_LAGS) + k + len(CROSS_LAGS) * D_OUT + DIR_BINS + 2
    return list(range(c0, c0 + k))


def norm_stats(stats: PopStats):
    """Per-unit input normalisation (mean, std) from the labeled bins only."""
    s = stats.d
    n = s["n"].clamp_min(1.0)
    mean = s["sx"] / n
    var = (torch.diagonal(s["gram"]) / n - mean ** 2).clamp_min(0.0)
    std = var.sqrt()
    std = torch.where(std < STD_FLOOR, torch.ones_like(std), std)
    return mean, std


def describe(stats: PopStats, prior: torch.Tensor):
    """Finalise streamed sums into per-unit descriptors.

    prior [K, N] is the (fixed, label-free) prior table.  Returns
    (d [N, Fd], r [N, REL_DIM], mean [N], std [N]); d and r are float32.
    """
    s = stats.d
    n = s["n"].clamp_min(1.0)
    n_units = s["sx"].shape[0]
    k = prior.shape[0]
    dt = s["sx"].dtype
    mean, std = norm_stats(stats)
    raw_std = (torch.diagonal(s["gram"]) / n - mean ** 2).clamp_min(1e-12).sqrt()
    # normalised Gram of x over the labeled bins
    gn = (s["gram"] - mean[:, None] * s["sx"][None, :] - s["sx"][:, None] * mean[None, :]
          + n * mean[:, None] * mean[None, :]) / (std[:, None] * std[None, :])
    p = prior.to(dt)
    cz = gn @ p.T / n                                              # [N, K]
    zvar = torch.einsum("kn,nm,km->k", p, gn, p) / n
    xvar = torch.diagonal(gn) / n
    corr = (cz / (xvar[:, None] * zvar[None, :] + 1e-12).sqrt()).clamp(-1, 1)
    feats = [torch.log(mean + 1e-2)[:, None], torch.log(raw_std + 1e-3)[:, None]]
    for j in range(len(AC_LAGS)):
        pn = s["ac_n"][j].clamp_min(1.0)
        cov = (s["ac_xx"][j] - mean * (s["ac_a"][j] + s["ac_b"][j]) + pn * mean ** 2) / pn
        feats.append((cov / std ** 2).clamp(-1, 1)[:, None])
    feats.append(corr)
    cc = []
    for li in range(len(CROSS_LAGS)):
        pn = s["yx_n"][li].clamp_min(1.0)
        block = (s["syx"][li] - mean[None, :] * s["sy"][li][:, None]) / (std[None, :] * pn)
        cc.append(block.T)                                         # [N, D]
    feats.append(torch.cat(cc, 1))
    dir_rate = (s["sdir"] - mean[:, None] * s["cdir"][None, :]) / (
        std[:, None] * (s["cdir"][None, :] + 1.0))
    feats.append(dir_rate)
    phi_x = (s["sphix"] - s["sphi"][:, None] * mean[None, :]) / std[None, :]   # [3, N]
    a = s["sphiphi"] + 1e-2 * n * torch.eye(3, dtype=dt, device=n.device)
    coef = torch.linalg.solve(a, phi_x)                            # closed-form 3x3 solve
    feats.append(coef[1:].T)
    feats.append(p.T)
    log_n = torch.log(n).expand(n_units, 1)
    log_b = torch.full((n_units, 1), math.log(max(stats.n_trials, 1)), dtype=dt, device=n.device)
    feats += [log_n, log_b]
    d = torch.cat(feats, 1).to(torch.float32)
    # reliability features: budget, rate, between-trial dispersion, behavior consistency
    if stats.n_trials >= 2:
        disp = (s["q_sx2"] - n * mean ** 2) / ((stats.n_trials - 1) * mean + 1e-3)
        fano = torch.log(disp.clamp(0.01, 100.0))
    else:
        fano = torch.zeros(n_units, dtype=dt, device=n.device)
    tot = (s["syx"][LAG0] - mean[None, :] * s["sy"][LAG0][:, None]) / std[None, :]   # [D, N]
    per_trial = (s["q_yx2"] - 2 * mean[None, :] * s["q_yxy"]
                 + mean[None, :] ** 2 * s["q_y2"][:, None]) / std[None, :] ** 2
    kappa = ((tot ** 2 / n) / (per_trial + 1e-9)).clamp(0, 1)          # [D, N]
    r = torch.cat([log_n, log_b, torch.log(mean + 1e-2)[:, None], fano[:, None],
                   kappa.T], 1).to(torch.float32)
    return d, r, mean, std


# ---------------------------------------------------------------- priors ----

def electrode_mean_table(tables: list, electrodes: list, width: int = 96, exclude=None):
    """Electrode-aligned mean of per-session tables [K, N_j] -> [K, width].

    Same construction as ``run_labeled_calibration.loo_prior`` (counts
    normalised); ``exclude`` drops one session index (leave-one-out).
    """
    k = tables[0].shape[0]
    acc = np.zeros((k, width))
    cnt = np.zeros(width)
    for j, (tab, el) in enumerate(zip(tables, electrodes)):
        if exclude is not None and j == exclude:
            continue
        for col, e in enumerate(el):
            acc[:, e] += tab[:, col]
            cnt[e] += 1
    return acc / np.maximum(cnt, 1)[None, :]


def identity_free_prior(tables: list, n_units: int, mode: str, exclude=None):
    """Prior table [K, n_units] that uses NO unit-identity (electrode) metadata.

    ``globalmean``: the mean over ALL source-unit columns of ALL source tables
    (every source unit weighted equally, ``exclude`` drops one session for
    leave-one-out); the same K-vector is assigned to every target unit.
    ``zero``: the zero table.  Neither depends on which unit is which, so the
    resulting calibrator is blind to the unit-to-electrode map.
    """
    k = tables[0].shape[0]
    if mode == "zero":
        return np.zeros((k, n_units))
    if mode != "globalmean":
        raise ValueError(mode)
    cols = [t for j, t in enumerate(tables) if exclude is None or j != exclude]
    c = np.concatenate(cols, axis=1).mean(1)
    return np.repeat(c[:, None], n_units, axis=1)


# ------------------------------------------------------------ calibrator ----

def _mlp(i, o, hidden=64, layers=2, zero_last=False):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, o))
    net = nn.Sequential(*mods)
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def online_softmax_pool(scores: torch.Tensor, values: torch.Tensor, chunk: int = 3):
    """Streaming softmax pooling over units: running max, normaliser, sum.

    scores [N, m], values [N, dv] -> [m, dv].  Equals the batch softmax
    pooling exactly (up to fp error); only O(m * dv) state is kept.
    """
    m, dv = scores.shape[1], values.shape[1]
    run_max = torch.full((m,), -float("inf"), dtype=scores.dtype, device=scores.device)
    norm = torch.zeros(m, dtype=scores.dtype, device=scores.device)
    acc = torch.zeros(m, dv, dtype=values.dtype, device=values.device)
    for lo in range(0, scores.shape[0], chunk):
        sc, va = scores[lo:lo + chunk], values[lo:lo + chunk]
        new_max = torch.maximum(run_max, sc.max(0).values)
        scale = torch.exp(run_max - new_max)
        w = torch.exp(sc - new_max)
        norm = norm * scale + w.sum(0)
        acc = acc * scale[:, None] + w.T @ va
        run_max = new_max
    return acc / norm[:, None]


class PopulationCalibrator(nn.Module):
    def __init__(self, variant: str, k: int, fd: int, fr: int = REL_DIM, *, m: int = 8,
                 hidden: int = 64, dctx: int = 32, output: str = "residual",
                 label_free: bool = False, prior_mode: str = "aligned"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        if prior_mode not in PRIOR_MODES:
            raise ValueError(prior_mode)
        self.variant, self.k, self.fd, self.m, self.output = variant, k, fd, m, output
        self.dctx = dctx
        self.hidden = hidden
        self.register_buffer("d_mu", torch.zeros(fd))
        self.register_buffer("d_sd", torch.ones(fd))
        self.register_buffer("r_mu", torch.zeros(fr))
        self.register_buffer("r_sd", torch.ones(fr))
        d_keep, r_keep = torch.ones(fd), torch.ones(fr)
        if label_free:                    # control: label-derived columns are zeroed
            dc, rc = label_columns(k)
            d_keep[dc], r_keep[rc] = 0.0, 0.0
        self.label_free = label_free
        self.prior_mode = prior_mode
        if prior_mode != "aligned":       # identity-free: the prior column carries no per-unit info
            d_keep[prior_columns(k)] = 0.0
            if prior_mode == "zero":      # zero prior latents -> correlation features are void
                d_keep[corr_columns(k)] = 0.0
        self.softmax_fn = torch.softmax   # replaceable (exp2-LUT softmax in A'3)
        self.weight_quant = None          # optional fn applied to the PMA seed matrices u, a (QAT)
        self.register_buffer("d_keep", d_keep)
        self.register_buffer("r_keep", r_keep)
        self.lam_override = None
        self.rel = self.mu0 = self.psi = self.phi = self.val = None
        gen_in = fd
        if variant != "p0":
            self.rel = _mlp(fr, 1, 16, layers=1)
            nn.init.constant_(self.rel[-1].bias, 2.0)
            self.mu0 = nn.Parameter(torch.zeros(fd))
        if variant == "p2":
            self.psi = _mlp(fd, dctx, hidden)
            gen_in = fd + dctx
        if variant in ("p3", "p4"):
            self.phi = _mlp(fd, hidden, hidden)
            self.u = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.a = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.val = nn.Linear(hidden, dctx)
            gen_in = fd + dctx + 1
        self.gen = _mlp(gen_in, k, hidden, zero_last=(output == "residual"))

    # -- normalisers are fitted once on source episodes and then frozen ------
    def set_normalizers(self, d, r):
        with torch.no_grad():
            self.d_mu.copy_(d.mean(0))
            self.d_sd.copy_(d.std(0).clamp_min(1e-3))
            self.r_mu.copy_(r.mean(0))
            self.r_sd.copy_(r.std(0).clamp_min(1e-3))

    def reliability(self, r):
        rn = (r - self.r_mu) / self.r_sd * self.r_keep
        lam = torch.sigmoid(self.rel(rn)).squeeze(-1)
        if self.lam_override is not None:
            lam = torch.full_like(lam, float(self.lam_override))
        return lam

    def forward(self, d, r, prior, return_aux=False):
        """d [N, Fd], r [N, Fr], prior [K, N] -> table [K, N]."""
        dn = (d - self.d_mu) / self.d_sd * self.d_keep
        aux = {}
        lam = None
        if self.variant != "p0":
            lam = self.reliability(r)
            dt = lam[:, None] * dn + (1.0 - lam[:, None]) * self.mu0
            aux["lam"] = lam
        else:
            dt = dn
        if self.variant in ("p0", "p1"):
            inp = dt
        elif self.variant == "p2":
            ctx = self.psi(dt).mean(0, keepdim=True).expand(dt.shape[0], -1)
            inp = torch.cat([dt, ctx], 1)
        else:
            ph = self.phi(dt)
            wq = self.weight_quant or (lambda w: w)
            scores = ph @ wq(self.u).T / math.sqrt(self.hidden)        # [N, m]
            if self.variant == "p4":
                scores = scores + torch.log(lam.clamp_min(1e-6))[:, None]
            values = self.val(ph)                                      # [N, dctx]
            zj = self.softmax_fn(scores, dim=0).T @ values             # [m, dctx]
            beta = self.softmax_fn(ph @ wq(self.a).T / math.sqrt(self.hidden), dim=1)
            mu_i = beta @ zj
            inp = torch.cat([dt, mu_i, lam[:, None]], 1)
            aux["scores"], aux["values"], aux["zj"], aux["beta"] = scores, values, zj, beta
        out = self.gen(inp).T                                          # [K, N]
        table = prior + out if self.output == "residual" else out
        return (table, aux) if return_aux else table

    # -- target-side entry ---------------------------------------------------
    def calibrate(self, stats: PopStats, prior: torch.Tensor, *, oracle: bool = False):
        """Fixed-weight forward under no_grad.  ``oracle`` flags the result
        as diagnostic-only: the caller must have built ``stats`` from ALL
        labeled trials of the target session."""
        with torch.no_grad():
            d, r, mean, std = describe(stats, prior)
            self._seen_grad_mode = torch.is_grad_enabled()
            table = self(d, r, prior.to(torch.float32))
        assert not table.requires_grad
        return CalibResult(table=table.detach(), mean=mean, std=std,
                           mode="oracle" if oracle else "candidate",
                           grad_enabled_inside=self._seen_grad_mode)


@dataclass
class CalibResult:
    table: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    mode: str = "candidate"
    grad_enabled_inside: bool = False

    @property
    def diagnostic_only(self) -> bool:
        return self.mode != "candidate"


def assert_candidate(result: CalibResult):
    """Reject diagnostic (oracle) results anywhere a candidate is required."""
    if result.diagnostic_only:
        raise ValueError("P-oracle results are diagnostic only and cannot be candidates")
    return result


def select_candidates(rows: Iterable[dict], key: str = "val_score"):
    """Best row by ``key`` among rows that are legal candidates."""
    legal = [r for r in rows if not r.get("diagnostic_only", False)
             and not str(r.get("variant", "")).startswith("p_oracle")
             and r.get("variant") != "p_prior"]
    if not legal:
        raise ValueError("no legal candidate rows")
    return max(legal, key=lambda r: r[key])


# -------------------------------------------------------------- MAC count ----

def mac_counts(variant: str, n_units: int, n_bins: int, k: int, fd: int,
               hidden: int = 64, dctx: int = 32, m: int = 8):
    """Analytic MAC counts of one calibration (accumulate vs fixed-weight)."""
    accumulate = n_bins * (n_units ** 2 + (3 * 3 + 2 * len(CROSS_LAGS) * D_OUT + 8 + 3 + 4) * n_units)
    closed = n_units * 3 * 9 + 27
    fixed = n_units * (fd * hidden + hidden * hidden + hidden * k)
    if variant != "p0":
        fixed += n_units * (REL_DIM * 16 + 16)
    if variant == "p2":
        fixed += n_units * (fd * hidden + hidden * dctx) + n_units * (dctx * hidden)
    if variant in ("p3", "p4"):
        fixed += n_units * (fd * hidden + hidden * hidden + 2 * hidden * m + hidden * dctx
                            + (dctx + 1) * hidden) + m * dctx * n_units
    total = accumulate + closed + fixed
    return {"accumulate_macs": int(accumulate), "closed_form_macs": int(closed),
            "fixed_weight_macs": int(fixed), "h3_fixed_ratio": float(fixed / max(total, 1))}


# ------------------------------------------------------------ session data ----

class SessionData:
    """One session on the device: raw counts, standardised labels, trial bank."""

    def __init__(self, record, label_stats, device, *, name=None, usable=None):
        from ssm_decode.track_a_experiment import usable_trials
        self.record = record
        self.name = name or getattr(record, "session", "session")
        self.device = device
        self.electrodes = np.asarray(getattr(record, "unit_to_electrode"))
        self.trials = usable if usable is not None else usable_trials(record)
        y = np.asarray(record.behavior, dtype=np.float64)
        ym, ys = label_stats
        self.valid_np = np.asarray(record.eval_mask) & np.isfinite(y).all(1)
        ang = (np.arctan2(y[:, 1], y[:, 0]) + 2 * math.pi) % (2 * math.pi)
        ang_bin = np.floor(ang / (2 * math.pi / DIR_BINS)).astype(np.int64) % DIR_BINS
        self.x = torch.from_numpy(np.asarray(record.neural, dtype=np.float64)).to(device)
        self.y_norm = torch.from_numpy(np.nan_to_num((y - ym) / ys)).to(device)
        self.valid = torch.from_numpy(self.valid_np).to(device)
        self.n_bins, self.n_units = self.x.shape
        self.bank = TrialBank.build(self.x, self.y_norm, torch.from_numpy(ang_bin).to(device),
                                    self.valid, self.trials)
        tid = np.full(self.n_bins, -1, dtype=np.int64)
        for i, (a, b) in enumerate(self.trials):
            tid[a:b] = i
        self.trial_id = torch.from_numpy(tid).to(device)
        self.x32 = self.x.to(torch.float32)

    def query_mask(self, idx):
        """Valid bins of the usable trials NOT in ``idx`` (target query set)."""
        chosen = torch.zeros(len(self.trials), dtype=torch.bool, device=self.device)
        chosen[torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.device)] = True
        in_trial = self.trial_id >= 0
        is_cal = torch.zeros_like(in_trial)
        is_cal[in_trial] = chosen[self.trial_id[in_trial]]
        return in_trial & ~is_cal & self.valid
