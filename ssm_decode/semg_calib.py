"""Fixed-weight forward calibrators for sEMG (feasibility Q-D, Hyser and NinaPro DB6).

Target-side contract (same as the 000688 population-calibration round): the trunk
is frozen; calibration is (a) streaming accumulation of additive sufficient
statistics over the B labeled trials, (b) a fixed-weight forward of a calibrator
trained on SOURCE sessions only, (c) an SRAM write of the static K x N read-in
table.  No gradient, no backprop and no iterative optimisation touches target
data.  Input normalisation (per-channel mean / std) also comes from the B
calibration trials only.

Units are electrode channels.  Channel identity is known on sEMG (fixed grid /
fixed 14 channels), so the "identity-aligned prior" is simply the channel-aligned
table (source mean table, or the subject's earlier-session table).  The
identity-free control withholds identity: zero prior, no prior column, no
prior-latent correlation in the descriptors.

Variants (as in ``population_calib``): p0 independent per-channel encoder, p1 = p0 +
reliability shrinkage, p2 = p1 + DeepSets context, p4 = PMA (m seeds) with
reliability-weighted scores.  This module is a self-contained copy adapted to
sEMG descriptors; it does not import ``population_calib`` (which is edited
concurrently by another work line).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

AC_LAGS = (1, 3, 10)
VARIANTS = ("p0", "p1", "p2", "p4")
# Post-review controls (identity-free capacity question, not part of the original 4 variants):
#   p0w  : P0 with the generator MLP widened so that the parameter count ~ P2 (no pooling, no reliability)
#   p0w4 : same, matched to P4
#   p2c  : P2 whose pooled context is replaced by one constant learned vector (no population information)
#   p2l  : P2 whose pooled context is replaced by the per-channel psi(d~_i) (P2 parameters, no pooling)
CONTROL_VARIANTS = ("p0w", "p0w4", "p2c", "p2l")
ALL_VARIANTS = VARIANTS + CONTROL_VARIANTS
PRIOR_MODES = ("aligned", "free")
REL_DIM = 5
STD_FLOOR_REL = 1e-3        # same rule as hyser_backbone.input_stats / db6_backbone.input_stats


@dataclass(frozen=True)
class SemgSpec:
    """Static description of the descriptor layout of one dataset."""
    name: str
    n_ch: int
    k: int
    d_label: int
    label_lags: tuple          # first entry must be 0; pairs (x_t, y_{t+lag}), lag >= 0
    ac_lags: tuple = AC_LAGS
    std_floor_ref: str = "mean"      # 'mean' (Hyser v1 rule) or 'median' (DB6 v2 rule) of the channel stds
    std_floor: float = STD_FLOOR_REL

    def __post_init__(self):
        if self.label_lags[0] != 0:
            raise ValueError("label_lags[0] must be 0")

    @property
    def groups(self) -> dict:
        sizes = [("level", 2), ("ac", len(self.ac_lags)), ("corr_prior", self.k), ("corr_glob", 1),
                 ("label", len(self.label_lags) * self.d_label), ("ymean", self.d_label),
                 ("prior", self.k), ("budget", 2)]
        out, c = {}, 0
        for name, size in sizes:
            out[name] = list(range(c, c + size))
            c += size
        return out

    @property
    def desc_dim(self) -> int:
        return sum(len(v) for v in self.groups.values())

    @property
    def max_lag(self) -> int:
        return max(max(self.ac_lags), max(self.label_lags))


# ------------------------------------------------------- additive statistics --

def accumulate_chunk(xc, yc, vc, m, spec: SemgSpec) -> dict:
    """Additive sums of the rows ``[m:]`` of a chunk, using rows ``[:m]`` only as lag context.

    xc [T, N] float64 envelope, yc [T, D] labels, vc [T] bool.  With ``m = 0`` this is the
    whole-segment sums of one trial; with m = max_lag and the previous chunk's tail as context
    it is the exact streaming update (a chunked pass equals the one-shot pass).
    """
    dt = xc.dtype
    t_len, n = xc.shape
    v = vc.to(dt)
    yc = torch.where(vc.unsqueeze(1), yc, torch.zeros_like(yc))
    xo, vo = xc[m:], v[m:]
    out = {"n": vo.sum(), "sx": (vo.unsqueeze(1) * xo).sum(0), "gram": xo.T @ (vo.unsqueeze(1) * xo)}
    ac_xx, ac_a, ac_b, ac_n = [], [], [], []
    for k in spec.ac_lags:
        lo = max(m, k)
        if t_len > lo:
            vv = v[lo:] * v[lo - k:t_len - k]
            xt, xl = xc[lo:], xc[lo - k:t_len - k]
            ac_xx.append((vv.unsqueeze(1) * xt * xl).sum(0))
            ac_a.append((vv.unsqueeze(1) * xt).sum(0))
            ac_b.append((vv.unsqueeze(1) * xl).sum(0))
            ac_n.append(vv.sum())
        else:
            z = torch.zeros(n, dtype=dt, device=xc.device)
            ac_xx.append(z), ac_a.append(z), ac_b.append(z)
            ac_n.append(torch.zeros((), dtype=dt, device=xc.device))
    out["ac_xx"], out["ac_a"], out["ac_b"] = torch.stack(ac_xx), torch.stack(ac_a), torch.stack(ac_b)
    out["ac_n"] = torch.stack(ac_n)
    sy, syx, yx_n = [], [], []
    for lag in spec.label_lags:                      # pairs (x_{j-lag}, y_j), j >= lag
        lo = max(m, lag)
        if t_len > lo:
            vv = v[lo:] * v[lo - lag:t_len - lag]
            yt, xt = yc[lo:], xc[lo - lag:t_len - lag]
            sy.append((vv.unsqueeze(1) * yt).sum(0))
            syx.append((vv.unsqueeze(1) * yt).T @ xt)
            yx_n.append(vv.sum())
        else:
            sy.append(torch.zeros(spec.d_label, dtype=dt, device=xc.device))
            syx.append(torch.zeros(spec.d_label, n, dtype=dt, device=xc.device))
            yx_n.append(torch.zeros((), dtype=dt, device=xc.device))
    out["sy"], out["syx"], out["yx_n"] = torch.stack(sy), torch.stack(syx), torch.stack(yx_n)
    return out


def segment_sums(x, y, valid, spec: SemgSpec) -> dict:
    """Raw additive sums of ONE labeled trial (x [T,N] float64, y [T,D], valid [T] bool)."""
    return accumulate_chunk(x, y, valid, 0, spec)


_ADDITIVE_KEYS = ("n", "sx", "gram", "ac_xx", "ac_a", "ac_b", "ac_n", "sy", "syx", "yx_n")


class StreamAcc:
    """Exact chunk-by-chunk accumulator (on-chip contract), minimal per-trial state.

    Keeps only the last ``max_lag`` bins as context.  Every chunk is added DIRECTLY to the session
    accumulators (one Gram, not a per-trial copy of it); the only per-trial state is the small set of
    terms the reliability features need (n, sum x, sum y x at lag 0, sum y: N + D*N + D + 1 values,
    ``trial_state_bytes``).  ``end_trial`` turns that small state into the per-trial energy terms and adds
    them to the session totals; ``result`` returns the SemgStats.

    (The first version of this class kept a full per-trial copy of every accumulator, i.e. ~2x the Gram;
    the experiments themselves used ``TrialBank`` sums, which are numerically the same quantities.)
    """

    def __init__(self, spec: SemgSpec):
        self.spec = spec
        self.tot = None
        self.n_trials = 0
        self._reset_trial()

    def _reset_trial(self):
        self.ctx = None
        self.small = None

    def update(self, x, y, valid):
        m = 0 if self.ctx is None else self.ctx[0].shape[0]
        xc = x if self.ctx is None else torch.cat([self.ctx[0], x])
        yc = y if self.ctx is None else torch.cat([self.ctx[1], y])
        vc = valid if self.ctx is None else torch.cat([self.ctx[2], valid])
        s = accumulate_chunk(xc, yc, vc, m, self.spec)
        sm = {"n": s["n"], "sx": s["sx"], "syx0": s["syx"][0], "sy0": s["sy"][0]}
        self.small = sm if self.small is None else {k: self.small[k] + sm[k] for k in sm}
        self.tot = s if self.tot is None else {**self.tot, **{k: self.tot[k] + s[k] for k in s}}
        keep = self.spec.max_lag
        self.ctx = (xc[-keep:], yc[-keep:], vc[-keep:])

    def trial_state_bytes(self, bytes_per_value: int = 4) -> int:
        """Size of the per-trial buffer (the whole of it; the Gram is NOT duplicated per trial)."""
        return 0 if self.small is None else sum(v.numel() for v in self.small.values()) * bytes_per_value

    def end_trial(self):
        add = _q_terms(self.small["n"], self.small["sx"], self.small["syx0"], self.small["sy0"])
        for k, v in add.items():
            self.tot[k] = v if k not in self.tot else self.tot[k] + v
        self.n_trials += 1
        self._reset_trial()

    def result(self) -> "SemgStats":
        return SemgStats({k: v for k, v in self.tot.items()}, self.n_trials)


def _q_terms(n, sx, syx0, sy0) -> dict:
    nb = n.clamp_min(1.0)
    return {"q_sx2": sx ** 2 / nb, "q_yx2": syx0 ** 2 / nb, "q_yxy": syx0 * sy0[:, None] / nb,
            "q_y2": sy0 ** 2 / nb}


class TrialBank:
    """Per-trial raw sums, built lazily and ONLY for requested trials (never for query trials)."""

    def __init__(self, spec: SemgSpec):
        self.spec = spec
        self.rows: dict[int, dict] = {}

    def add(self, trial_id: int, x, y, valid):
        self.rows[int(trial_id)] = segment_sums(x, y, valid, self.spec)

    def has(self, trial_id) -> bool:
        return int(trial_id) in self.rows


@dataclass
class SemgStats:
    d: dict
    n_trials: int

    @staticmethod
    def from_bank(bank: TrialBank, idx) -> "SemgStats":
        rows = [bank.rows[int(i)] for i in idx]          # KeyError if a trial was never banked
        d = {k: sum(r[k] for r in rows) for k in _ADDITIVE_KEYS}
        q = [_q_terms(r["n"], r["sx"], r["syx"][0], r["sy"][0]) for r in rows]
        for k in ("q_sx2", "q_yx2", "q_yxy", "q_y2"):
            d[k] = sum(qq[k] for qq in q)
        return SemgStats(d, len(rows))

    def __add__(self, other: "SemgStats") -> "SemgStats":
        return SemgStats({k: self.d[k] + other.d[k] for k in self.d}, self.n_trials + other.n_trials)


def norm_stats(stats: SemgStats, floor_ref: str = "mean", floor: float = STD_FLOOR_REL):
    """Per-channel input normalisation (mean, std) from the calibration bins only.

    Same rule as ``input_stats`` of the backbones: population std, floored at
    ``1e-3 * mean(std)`` (Hyser) or ``1e-3 * median(std)`` (DB6 v2: robust to dead channels).
    """
    s = stats.d
    n = s["n"].clamp_min(1.0)
    mean = s["sx"] / n
    var = (torch.diagonal(s["gram"]) / n - mean ** 2).clamp_min(0.0)
    sd = var.sqrt()
    ref = sd.median() if floor_ref == "median" else sd.mean()
    sd = torch.maximum(sd, floor * ref.clamp_min(1e-12))
    return mean, sd


# --------------------------------------------------------------- descriptors --

def describe(stats: SemgStats, prior: torch.Tensor | None, spec: SemgSpec):
    """Finalise streamed sums into per-channel descriptors d [N, Fd] and reliability r [N, 5].

    ``prior`` [K, N] is the fixed label-free prior table, or ``None`` for the identity-free
    control (prior-dependent columns are then zero).  Returns (d, r, mean, std).
    """
    s = stats.d
    dt = s["sx"].dtype
    dev = s["sx"].device
    n = s["n"].clamp_min(1.0)
    n_ch = s["sx"].shape[0]
    k = spec.k
    mean, std = norm_stats(stats, spec.std_floor_ref, spec.std_floor)
    raw_var = (torch.diagonal(s["gram"]) / n - mean ** 2).clamp_min(1e-12)
    gn = (s["gram"] - mean[:, None] * s["sx"][None, :] - s["sx"][:, None] * mean[None, :]
          + n * mean[:, None] * mean[None, :]) / (std[:, None] * std[None, :])     # normalised Gram
    xvar = torch.diagonal(gn) / n
    # level
    f_level = [torch.log(mean + 1e-4)[:, None], (0.5 * torch.log(raw_var + 1e-8))[:, None]]
    # autocorrelation
    f_ac = []
    for j in range(len(spec.ac_lags)):
        pn = s["ac_n"][j].clamp_min(1.0)
        cov = (s["ac_xx"][j] - mean * (s["ac_a"][j] + s["ac_b"][j]) + pn * mean ** 2) / pn
        f_ac.append((cov / std ** 2).clamp(-1, 1)[:, None])
    # correlation with prior latents (identity-aligned only)
    if prior is None:
        corr = torch.zeros(n_ch, k, dtype=dt, device=dev)
        p_cols = torch.zeros(n_ch, k, dtype=dt, device=dev)
    else:
        p = prior.to(dt)
        cz = gn @ p.T / n
        zvar = torch.einsum("kn,nm,km->k", p, gn, p) / n
        corr = (cz / (xvar[:, None] * zvar[None, :] + 1e-12).sqrt()).clamp(-1, 1)
        p_cols = p.T
    # correlation with the global mean activity of all channels (identity-free, population level)
    cg = gn.sum(1) / (n_ch * n)
    gvar = gn.sum() / (n_ch ** 2 * n)
    corr_glob = (cg / (xvar * gvar + 1e-12).sqrt()).clamp(-1, 1)[:, None]
    # label block: normalised cross-covariance at each label lag, D labels
    blocks = []
    for li in range(len(spec.label_lags)):
        pn = s["yx_n"][li].clamp_min(1.0)
        blocks.append(((s["syx"][li] - mean[None, :] * s["sy"][li][:, None]) / (std[None, :] * pn)).T)
    f_label = torch.cat(blocks, 1)
    ymean = (s["sy"][0] / s["yx_n"][0].clamp_min(1.0))[None, :].expand(n_ch, -1)
    log_n = torch.log(n).expand(n_ch, 1)
    log_b = torch.full((n_ch, 1), math.log(max(stats.n_trials, 1)), dtype=dt, device=dev)
    d = torch.cat(f_level + f_ac + [corr, corr_glob, f_label, ymean, p_cols, log_n, log_b], 1).to(torch.float32)
    # reliability: budget, level, between-trial dispersion (CV^2 of trial means), label consistency
    if stats.n_trials >= 2:
        b = stats.n_trials
        ssb = (s["q_sx2"] - n * mean ** 2).clamp_min(0.0)
        cv2 = (ssb / n) * b / (b - 1) / (mean ** 2 + 1e-8)
        fano = torch.log(cv2.clamp(1e-3, 1e2))
    else:
        fano = torch.zeros(n_ch, dtype=dt, device=dev)
    tot = (s["syx"][0] - mean[None, :] * s["sy"][0][:, None]) / std[None, :]               # [D, N]
    per_trial = (s["q_yx2"] - 2 * mean[None, :] * s["q_yxy"]
                 + mean[None, :] ** 2 * s["q_y2"][:, None]) / std[None, :] ** 2
    kappa = ((tot ** 2 / n) / (per_trial + 1e-9)).clamp(0, 1).mean(0)                      # [N]
    r = torch.cat([log_n, log_b, torch.log(mean + 1e-4)[:, None], fano[:, None], kappa[:, None]], 1)
    return d, r.to(torch.float32), mean, std


# -------------------------------------------------------------------- priors --

def mean_prior(tables: torch.Tensor, exclude=()) -> torch.Tensor:
    """Channel-aligned source mean table [K, N] from tables [S, K, N], dropping ``exclude`` session ids."""
    keep = [i for i in range(tables.shape[0]) if i not in set(int(e) for e in exclude)]
    if not keep:
        raise ValueError("no source tables left")
    return tables[keep].mean(0)


def build_prior(mode: str, tables: torch.Tensor, *, source: str, exclude=(), own=None) -> torch.Tensor | None:
    """Prior construction used identically in training and evaluation.

    ``source`` in {'mean', 'own'}: 'mean' = mean over source tables excluding ``exclude`` (the
    target's own sessions); 'own' = the explicit table ``own`` (the subject's earlier-session
    table).  mode 'free' returns None (no identity).  The target's own session is never an input.
    """
    if mode == "free":
        return None
    if source == "mean":
        return mean_prior(tables, exclude)
    if source == "own":
        if own is None:
            raise ValueError("own prior needs the explicit table")
        return own.clone()
    raise ValueError(source)


# ---------------------------------------------------------------- calibrator --

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


def online_softmax_pool(scores: torch.Tensor, values: torch.Tensor, chunk: int = 16):
    """Streaming softmax pooling over channels (running max / normaliser / sum); equals batch softmax."""
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


def _p0_param_count(fd: int, k: int, h: int) -> int:
    return (fd + 1) * h + (h + 1) * h + (h + 1) * k


def wide_p0_hidden(spec: SemgSpec, ref_variant: str = "p2", *, hidden: int = 64, dctx: int = 32, m: int = 8) -> int:
    """Hidden width of a plain P0 generator whose parameter count is closest to that of ``ref_variant``."""
    target = sum(p.numel() for p in SemgCalibrator(ref_variant, spec, m=m, hidden=hidden, dctx=dctx).parameters())
    return min(range(8, 2048), key=lambda h: abs(_p0_param_count(spec.desc_dim, spec.k, h) - target))


class SemgCalibrator(nn.Module):
    """p0 / p1 / p2 / p4 generator: descriptor -> K-dim table column (residual on the prior).

    Control variants (see ``CONTROL_VARIANTS``): p0w / p0w4 (wider P0), p2c (constant context), p2l
    (per-channel context).  None of the controls mixes information across channels, so the table column
    of channel i depends on (d_i, r_i, prior_i) only.
    """

    def __init__(self, variant: str, spec: SemgSpec, *, m: int = 8, hidden: int = 64, dctx: int = 32,
                 prior_mode: str = "aligned", out_scale: float = 1.0):
        super().__init__()
        if variant not in ALL_VARIANTS:
            raise ValueError(variant)
        if prior_mode not in PRIOR_MODES:
            raise ValueError(prior_mode)
        self.variant, self.spec, self.m, self.hidden, self.dctx = variant, spec, m, hidden, dctx
        self.prior_mode = prior_mode
        fd, k = spec.desc_dim, spec.k
        self.register_buffer("d_mu", torch.zeros(fd))
        self.register_buffer("d_sd", torch.ones(fd))
        self.register_buffer("r_mu", torch.zeros(REL_DIM))
        self.register_buffer("r_sd", torch.ones(REL_DIM))
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))
        keep = torch.ones(fd)
        if prior_mode == "free":                         # identity withheld: prior-dependent columns void
            g = spec.groups
            keep[g["prior"] + g["corr_prior"]] = 0.0
        self.register_buffer("d_keep", keep)
        self.rel = self.mu0 = self.psi = self.phi = self.val = self.ctx_const = None
        gen_in = fd
        gen_hidden = hidden
        if variant not in ("p0", "p0w", "p0w4"):
            self.rel = _mlp(REL_DIM, 1, 16, layers=1)
            nn.init.constant_(self.rel[-1].bias, 2.0)
            self.mu0 = nn.Parameter(torch.zeros(fd))
        if variant in ("p0w", "p0w4"):
            gen_hidden = wide_p0_hidden(spec, "p2" if variant == "p0w" else "p4", hidden=hidden, dctx=dctx, m=m)
        if variant in ("p2", "p2l"):
            self.psi = _mlp(fd, dctx, hidden)
            gen_in = fd + dctx
        if variant == "p2c":
            self.ctx_const = nn.Parameter(torch.zeros(dctx))
            gen_in = fd + dctx
        if variant == "p4":
            self.phi = _mlp(fd, hidden, hidden)
            self.u = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.a = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.val = nn.Linear(hidden, dctx)
            gen_in = fd + dctx + 1
        self.gen = _mlp(gen_in, k, gen_hidden, zero_last=True)

    def set_normalizers(self, d, r):
        with torch.no_grad():
            self.d_mu.copy_(d.mean(0))
            self.d_sd.copy_(d.std(0).clamp_min(1e-3))
            self.r_mu.copy_(r.mean(0))
            self.r_sd.copy_(r.std(0).clamp_min(1e-3))

    def reliability(self, r):
        return torch.sigmoid(self.rel((r - self.r_mu) / self.r_sd)).squeeze(-1)

    def forward(self, d, r, prior, return_aux=False):
        """d [N, Fd], r [N, Fr], prior [K, N] (zeros for the free mode) -> table [K, N]."""
        dn = (d - self.d_mu) / self.d_sd * self.d_keep
        aux = {}
        if self.variant in ("p0", "p0w", "p0w4"):
            dt, lam = dn, None
        else:
            lam = self.reliability(r)
            dt = lam[:, None] * dn + (1.0 - lam[:, None]) * self.mu0 * self.d_keep
            aux["lam"] = lam
        if self.variant in ("p0", "p1", "p0w", "p0w4"):
            inp = dt
        elif self.variant == "p2":
            ctx = self.psi(dt).mean(0, keepdim=True).expand(dt.shape[0], -1)
            inp = torch.cat([dt, ctx], 1)
        elif self.variant == "p2l":                                             # control: no pooling over channels
            inp = torch.cat([dt, self.psi(dt)], 1)
        elif self.variant == "p2c":                                             # control: constant learned context
            inp = torch.cat([dt, self.ctx_const[None, :].expand(dt.shape[0], -1)], 1)
        else:
            ph = self.phi(dt)
            scores = ph @ self.u.T / math.sqrt(self.hidden) + torch.log(lam.clamp_min(1e-6))[:, None]
            values = self.val(ph)
            zj = torch.softmax(scores, dim=0).T @ values                        # [m, dctx]
            beta = torch.softmax(ph @ self.a.T / math.sqrt(self.hidden), dim=1)
            inp = torch.cat([dt, beta @ zj, lam[:, None]], 1)
            aux["zj"] = zj
        out = self.gen(inp).T * self.out_scale                                  # [K, N]
        table = prior + out
        return (table, aux) if return_aux else table

    def calibrate(self, stats: SemgStats, prior: torch.Tensor | None) -> "CalibResult":
        """Target-side entry: fixed-weight forward under no_grad; refuses a grad context."""
        if torch.is_grad_enabled():
            with torch.no_grad():
                return self.calibrate(stats, prior)
        d, r, mean, std = describe(stats, prior, self.spec)
        pr = torch.zeros(self.spec.k, d.shape[0], device=d.device) if prior is None else prior.to(torch.float32)
        table = self(d, r, pr)
        if table.requires_grad:
            raise RuntimeError("target-side calibration must not build a graph")
        return CalibResult(table=table.detach(), mean=mean, std=std, grad_enabled_inside=torch.is_grad_enabled())


@dataclass
class CalibResult:
    table: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    grad_enabled_inside: bool = False


# ------------------------------------------------------ analytic cost (A'4) ----

def _lin_macs(i, o, h, layers=2):
    """Weight x activation MACs per channel of ``_mlp(i, o, h, layers)``."""
    return i * h + (layers - 1) * h * h + h * o


def fixed_macs_per_channel(spec: SemgSpec, variant: str, *, hidden=64, dctx=32, m=8) -> dict:
    """Fixed-weight (weight x activation) MACs per channel of the calibrator forward, by part.

    Counts every Linear layer that is executed per channel, the PMA score/route products (u, a), and the two
    PMA pooling matmuls (softmax(scores)^T @ values and beta @ zj, m*dctx each).  Mean pooling (P2) is adds only.
    """
    fd, k = spec.desc_dim, spec.k
    parts = {}
    if variant in ("p0w", "p0w4"):
        hw = wide_p0_hidden(spec, "p2" if variant == "p0w" else "p4", hidden=hidden, dctx=dctx, m=m)
        parts["gen"] = _lin_macs(fd, k, hw)
        return parts
    gen_in = fd
    if variant != "p0":
        parts["reliability"] = REL_DIM * 16 + 16
    if variant in ("p2", "p2l"):
        parts["psi"] = _lin_macs(fd, dctx, hidden)
        gen_in = fd + dctx
    if variant == "p2c":
        gen_in = fd + dctx
    if variant == "p4":
        parts["phi"] = _lin_macs(fd, hidden, hidden)
        parts["score_route"] = 2 * m * hidden
        parts["value"] = hidden * dctx
        parts["pool"] = 2 * m * dctx
        gen_in = fd + dctx + 1
    parts["gen"] = _lin_macs(gen_in, k, hidden)
    return parts


def cost_model(spec: SemgSpec, variant: str, n_bins_per_trial: int, b_trials: int, *, hidden=64,
               dctx=32, m=8, acc_bytes=4, table_bytes=2, trunk_params=None) -> dict:
    """Analytic MAC / SRAM numbers of ONE calibration of the implemented descriptor + calibrator.

    Counts tensors actually kept by this implementation.  The experiments simulated the accumulators in float64
    on GPU/CPU (no fixed-point per-add rounding / saturation was simulated, see README); bytes are reported
    for ``acc_bytes``-wide accumulators, i.e. an fp32 hardware port; the symmetric Gram can be stored as
    its upper triangle.  Buffers: session accumulators, the small per-trial state (``per_trial_state``), the lag
    delay line, descriptors, table.  The earlier ``StreamAcc`` additionally kept a full per-trial copy of
    every accumulator (``per_trial_full_copy``, ~2x the Gram); its totals are reported separately
    (``*_with_trial_copy``).
    """
    n, k, d_lab = spec.n_ch, spec.k, spec.d_label
    la, ll = len(spec.ac_lags), len(spec.label_lags)
    fd = spec.desc_dim
    bins = n_bins_per_trial * b_trials
    per_bin = {
        "gram": n * n,                              # x x^T rank-1 update (n^2 MACs / bin)
        "sx": n,                                    # running sum (adds, counted as MAC)
        "ac": la * 2 * n,                           # x_t * x_{t-l} products + sums
        "label_cross": ll * d_lab * n,              # y x^T rank-1 updates
    }
    acc_macs = bins * sum(per_bin.values())
    finalize = {
        "norm_gram": n * n * 2,                     # centre / scale of the Gram
        "corr_prior": n * n * k + n * k * k,        # gn @ P^T and zvar
        "corr_glob": n * n,
    }
    fin_macs = sum(finalize.values())
    parts = fixed_macs_per_channel(spec, variant, hidden=hidden, dctx=dctx, m=m)
    fixed_per_channel = sum(parts.values())
    fixed = n * fixed_per_channel
    n_params = sum(p.numel() for p in SemgCalibrator(variant, spec, m=m, hidden=hidden, dctx=dctx).parameters())
    nw = fixed_per_channel - parts.get("pool", 0)           # weight elements that multiply an activation
    acc_core = {"sx": n * acc_bytes, "ac": 3 * la * n * acc_bytes, "label_cross": ll * d_lab * n * acc_bytes}
    per_trial_state = (n + d_lab * n + d_lab + 1) * acc_bytes
    sram = {
        "gram_full": n * n * acc_bytes,
        "gram_upper_triangle": n * (n + 1) // 2 * acc_bytes,
        **acc_core, "per_trial_q_terms": (n + 2 * d_lab * n) * acc_bytes,
        "per_trial_state": per_trial_state,
        "stream_ctx": spec.max_lag * (n + d_lab) * acc_bytes,
        "descriptors": n * fd * acc_bytes, "table": k * n * table_bytes,
        "norm_mean_std": 2 * n * acc_bytes,
    }
    copy_core = sum(acc_core.values()) + 2 * acc_bytes          # + trial counters (n and the lag counts)
    copy_full = n * n * acc_bytes + copy_core
    copy_tri = n * (n + 1) // 2 * acc_bytes + copy_core
    # analytic-only alternative (NOT implemented / NOT evaluated): replace the N x N Gram by projections of a
    # fixed-reference-normalised stream onto the K prior latents (sxz [N,K]) and the global activity (sxm [N])
    lite_per_bin = 2 * n * k + n + n * (1 + 2 * la + ll * d_lab) + 3 * n
    lite_sram = (n * k + 2 * k + 3 * n + n + 3 * la * n + ll * d_lab * n + (n + 2 * d_lab * n)) * acc_bytes
    tot_full = int(sum(v for k_, v in sram.items() if k_ != "gram_upper_triangle"))
    tot_tri = int(sum(v for k_, v in sram.items() if k_ != "gram_full"))
    out = {"gram_free_estimate": {"accumulate_macs_per_bin": int(lite_per_bin),
                                  "accumulate_macs": int(bins * lite_per_bin),
                                  "sram_accumulators_bytes": int(lite_sram), "evaluated": False},
           "accumulate_macs": int(acc_macs), "accumulate_macs_per_bin": int(sum(per_bin.values())),
           "finalize_macs": int(fin_macs), "calibrator_fixed_macs": int(fixed),
           "calibrator_fixed_macs_per_channel": int(fixed_per_channel),
           "calibrator_fixed_macs_parts": {k_: int(v) for k_, v in parts.items()},
           "total_macs": int(acc_macs + fin_macs + fixed),
           "fixed_weight_fraction": float(fixed / max(acc_macs + fin_macs + fixed, 1)),
           "calibrator_weights": int(nw), "calibrator_params_total": int(n_params),
           "sram_bytes": {k_: int(v) for k_, v in sram.items()},
           "per_trial_full_copy_bytes": {"full_gram": int(copy_full), "tri_gram": int(copy_tri)},
           "sram_total_bytes_full_gram": tot_full, "sram_total_bytes_tri_gram": tot_tri,
           "sram_total_bytes_full_gram_with_trial_copy": int(tot_full - per_trial_state + copy_full),
           "sram_total_bytes_tri_gram_with_trial_copy": int(tot_tri - per_trial_state + copy_tri)}
    return out
