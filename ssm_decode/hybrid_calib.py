"""Hybrid-identity calibrator (aligned electrode prior + optional population pooling).

Setting: the read-in prior is electrode-aligned (a unit inherits the prior column of its electrode
averaged over the source sessions).  Units on electrodes that no source session ever used have NO
prior column (zero column).  Question: does population pooling (set pooling over the units, as in
P2/P4) help to generate the table columns of those uncovered units?

Everything here is label-free metadata + the B-trial descriptors; the target side is the usual
fixed-weight forward under ``torch.no_grad`` (no backprop, no optimisation).

Covered flag
------------
``covered_flags`` is computed from SOURCE electrode metadata only (the list of unit->electrode
arrays of the sessions that built the prior) and the target's own unit->electrode map.  It never
reads neural data, labels, or target queries.  It is exactly "the prior column of this unit is
non-trivially defined": ``covered_i  <=>  electrode(i) occurs in at least one source session``.

Uncovered simulation (episodic training on source sessions only)
-----------------------------------------------------------------
``simulate_uncovered`` masks the prior columns of whole electrodes (all units of the electrode)
chosen at random until the TOTAL uncovered unit fraction (natural leave-one-out uncovered plus
simulated) reaches a target fraction ``f``; ``f ~ U(0, 0.5)`` per episode (matches the observed
target range 0.08-0.50, a design choice from source/target metadata only).

Variants (``kind``)
-------------------
flat         one generator on [d~_i, flag_i] (P0-like; ``use_flag=False`` drops the flag)
hyb_ds       covered units: g_cov([d~_i, flag]) (aligned-prior residual path, no pooling);
             uncovered units: g_unc([d~_i, flag, ctx_i]) with ctx = mean_i psi(d~_i) (DeepSets)
hyb_pma      same with ctx_i = PMA (m seeds, log-reliability scores, per-unit routing)
all_ds       one generator on [d~_i, flag, ctx_i] for ALL units (pooling conditions covered too)
all_pma      same with PMA
*_loc        exact-parameter controls: the pooled context is replaced by the per-unit context
             psi(d~_i) (DeepSets) / val(phi(d~_i)) (PMA, pooling over a single unit).  No
             cross-unit information flows (tested).  For PMA the seed matrices u, a stay as
             (inert) parameters so the parameter count is identical.
The "wide" control is ``flat`` with a larger hidden width, chosen to match the parameter count.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ssm_decode.population_calib import (REL_DIM, CalibResult, PopStats, _mlp, describe)

KINDS = ("flat", "hyb_ds", "hyb_ds_loc", "hyb_pma", "hyb_pma_loc",
         "all_ds", "all_ds_loc", "all_pma", "all_pma_loc")
FRAC_MAX = 0.5       # uncovered-fraction range of the simulation, U(0, FRAC_MAX)


# ----------------------------------------------------------------- coverage ----

def electrode_counts(electrodes_list, width: int = 96, exclude=None) -> np.ndarray:
    """Number of source units per electrode (``exclude`` drops one source session)."""
    cnt = np.zeros(width, dtype=np.int64)
    for j, el in enumerate(electrodes_list):
        if exclude is not None and j == exclude:
            continue
        for e in np.asarray(el):
            cnt[int(e)] += 1
    return cnt


def covered_flags(source_electrodes, unit_electrodes, exclude=None, width: int = 96) -> np.ndarray:
    """bool [N]: the unit's electrode appears in at least one (non-excluded) source session.

    Uses ONLY source-session unit->electrode maps and the unit's own electrode id.
    """
    cnt = electrode_counts(source_electrodes, width, exclude)
    return cnt[np.asarray(unit_electrodes, dtype=np.int64)] > 0


def simulate_uncovered(rng: np.random.Generator, unit_electrodes, covered: np.ndarray, frac: float):
    """Extra-mask whole electrodes of covered units until the total uncovered fraction >= ``frac``.

    Returns a new bool covered vector (True = still covered).  Units already uncovered stay so.
    The number of covered units that remain is never driven below one.
    """
    el = np.asarray(unit_electrodes)
    cov = np.asarray(covered, dtype=bool).copy()
    n = len(cov)
    target_unc = int(math.ceil(frac * n - 1e-9))
    if int((~cov).sum()) >= target_unc:
        return cov
    cand = np.unique(el[cov])
    rng.shuffle(cand)
    for e in cand:
        if int((~cov).sum()) >= target_unc or int(cov.sum()) <= 1:
            break
        hit = cov & (el == e)
        if int(cov.sum()) - int(hit.sum()) < 1:
            continue
        cov &= ~hit
    return cov


def mask_prior(prior: torch.Tensor, covered) -> torch.Tensor:
    """Zero the prior columns of uncovered units (what the aligned prior does at a target)."""
    c = torch.as_tensor(np.asarray(covered), dtype=torch.bool, device=prior.device)
    return prior * c[None, :].to(prior.dtype)


# --------------------------------------------------------------- calibrator ----

class HybridCalibrator(nn.Module):
    def __init__(self, kind: str, k: int, fd: int, fr: int = REL_DIM, *, hidden: int = 64,
                 dctx: int = 32, m: int = 8, use_flag: bool = True):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.k, self.fd, self.hidden, self.dctx, self.m = kind, k, fd, hidden, dctx, m
        self.use_flag = bool(use_flag)
        self.fdf = fd + (1 if self.use_flag else 0)
        self.pooled = kind != "flat"
        self.local = kind.endswith("_loc")
        self.family = kind.split("_")[1] if self.pooled else None      # ds | pma
        self.hybrid = kind.startswith("hyb")
        self.register_buffer("d_mu", torch.zeros(fd))
        self.register_buffer("d_sd", torch.ones(fd))
        self.register_buffer("r_mu", torch.zeros(fr))
        self.register_buffer("r_sd", torch.ones(fr))
        self.rel = self.mu0 = self.psi = self.phi = self.val = None
        if self.pooled:
            self.rel = _mlp(fr, 1, 16, layers=1)
            nn.init.constant_(self.rel[-1].bias, 2.0)
            self.mu0 = nn.Parameter(torch.zeros(fd))
        if self.family == "ds":
            self.psi = _mlp(self.fdf, dctx, hidden)
            ctx_in = dctx
        elif self.family == "pma":
            self.phi = _mlp(self.fdf, hidden, hidden)
            self.u = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.a = nn.Parameter(torch.randn(m, hidden) * 0.1)
            self.val = nn.Linear(hidden, dctx)
            ctx_in = dctx + 1                                       # + lam, as in P3/P4
        else:
            ctx_in = 0
        if self.hybrid:
            self.gen_cov = _mlp(self.fdf, k, hidden, zero_last=True)
            self.gen_unc = _mlp(self.fdf + ctx_in, k, hidden, zero_last=True)
        else:
            self.gen = _mlp(self.fdf + ctx_in, k, hidden, zero_last=True)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def set_normalizers(self, d, r):
        with torch.no_grad():
            self.d_mu.copy_(d.mean(0))
            self.d_sd.copy_(d.std(0).clamp_min(1e-3))
            self.r_mu.copy_(r.mean(0))
            self.r_sd.copy_(r.std(0).clamp_min(1e-3))

    def _context(self, dtf, lam):
        if self.family == "ds":
            z = self.psi(dtf)
            return z if self.local else z.mean(0, keepdim=True).expand(dtf.shape[0], -1)
        ph = self.phi(dtf)
        scale = math.sqrt(self.hidden)
        if self.local:                  # pooling over a single unit: softmax over a size-1 axis == 1
            mu_i = self.val(ph)
        else:
            scores = ph @ self.u.T / scale + torch.log(lam.clamp_min(1e-6))[:, None]
            zj = torch.softmax(scores, dim=0).T @ self.val(ph)
            beta = torch.softmax(ph @ self.a.T / scale, dim=1)
            mu_i = beta @ zj
        return torch.cat([mu_i, lam[:, None]], 1)

    def forward(self, d, r, prior, covered):
        """d [N, Fd], r [N, Fr], prior [K, N], covered bool [N] -> table [K, N]."""
        cov = torch.as_tensor(covered, dtype=torch.bool, device=d.device)
        dn = (d - self.d_mu) / self.d_sd
        lam = None
        if self.pooled:
            rn = (r - self.r_mu) / self.r_sd
            lam = torch.sigmoid(self.rel(rn)).squeeze(-1)
            dn = lam[:, None] * dn + (1.0 - lam[:, None]) * self.mu0
        dtf = torch.cat([dn, cov[:, None].to(dn.dtype)], 1) if self.use_flag else dn
        ctx = self._context(dtf, lam) if self.pooled else None
        if self.hybrid:
            out_c = self.gen_cov(dtf)
            out_u = self.gen_unc(torch.cat([dtf, ctx], 1))
            out = torch.where(cov[:, None], out_c, out_u)
        else:
            out = self.gen(torch.cat([dtf, ctx], 1) if ctx is not None else dtf)
        return prior + out.T

    def calibrate(self, stats: PopStats, prior: torch.Tensor, covered):
        """Target-side entry: fixed-weight forward under no_grad (candidate result)."""
        cov = torch.as_tensor(np.asarray(covered), dtype=torch.bool, device=prior.device)
        if bool((prior.abs().sum(0)[~cov] != 0).any()):
            raise ValueError("uncovered units must have a zero prior column")
        with torch.no_grad():
            d, r, mean, std = describe(stats, prior)
            grad_mode = torch.is_grad_enabled()
            table = self(d, r, prior.to(torch.float32), cov)
        assert not table.requires_grad
        return CalibResult(table=table.detach(), mean=mean, std=std, mode="candidate",
                           grad_enabled_inside=grad_mode)


def match_width(target_params: int, k: int, fd: int, use_flag: bool = True, dctx: int = 32,
                m: int = 8) -> int:
    """Hidden width of a ``flat`` calibrator whose parameter count is closest to ``target_params``."""
    best, best_gap = None, None
    for h in range(16, 400):
        n = HybridCalibrator("flat", k, fd, hidden=h, dctx=dctx, m=m, use_flag=use_flag).n_params
        gap = abs(n - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best
