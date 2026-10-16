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
             cross-unit information flows INSIDE THE NETWORK (tested on d after the descriptors
             exist).  CORRECTION (v2 review, 2026-10-17): this does NOT make the controls free of
             cross-unit information.  The descriptor itself (population_calib.describe, corr
             features at the "latent correlation" columns) correlates every unit's counts with
             z = prior . x, a sum over the COVERED units, so an uncovered unit's descriptor already
             carries information about the covered population (perturbing one covered unit's Gram
             row changed an uncovered unit's corr feature by up to 0.25).  The -loc and wide
             controls therefore receive descriptor-level population coupling; only
             ``cut_coupling=True`` (v2) removes it for the uncovered units.
             For PMA the seed matrices u, a stay as (inert) parameters so the parameter count is
             identical.
The "wide" control is ``flat`` with a larger hidden width, chosen to match the parameter count.

v2 additions (hybrid identity v2)
---------------------------------
cut_coupling   zero the latent-correlation descriptor columns (``corr_columns``) of UNCOVERED units
               inside ``forward`` (their prior column is already 0), so that the only route by which
               population information reaches an uncovered unit is the model's own pooling.
pma_fix        un-collapse PMA: seed matrices u, a drawn with unit scale, learnable score / routing
               temperatures (no weight decay on u, a, temperatures: see ``no_decay_names``).
hyb_xatt       pairwise cross-attention: UNCOVERED units are queries, COVERED units are keys /
               values.  score_ij^h = q_i^h . k_j^h / sqrt(dk) + g_h([c_ij, c_ij sqrt(n/100)]) with c_ij
               the count correlation of unit i with covered unit j taken from the streamed Gram;
               values = [prior column of j, descriptor of j]; softmax over covered j (online-softmax
               compatible).  The attention output conditions the generator of the uncovered column
               and (through the attention-weighted covered prior columns, gate ``gamma``) is added to
               it.  Covered units use g_cov([d~_i, flag]) as in hyb_*.
pair_norm      (xatt kinds) row-standardise the pair input over the covered keys: the raw count
               correlations are tiny (mean |c| ~ 0.04 on the real targets), so the pair term of the
               first XATT barely varied across keys; the standardised input is O(1).  The same
               standardisation is applied to the per-key scalar of the control.
hyb_xatt_np    capacity-matched control: identical network and parameter count, but the pair input
               c_ij is replaced by a per-KEY scalar u_j (mean correlation of covered unit j with the
               other covered units), so no term depends on the (i, j) pair.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ssm_decode.population_calib import (REL_DIM, CalibResult, PopStats, _mlp, corr_columns,
                                         describe)

KINDS = ("flat", "hyb_ds", "hyb_ds_loc", "hyb_pma", "hyb_pma_loc",
         "all_ds", "all_ds_loc", "all_pma", "all_pma_loc", "hyb_xatt", "hyb_xatt_np")
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


def pair_corr(stats: PopStats):
    """Count correlation matrix [N, N] and bin count n from the streamed additive sums.

    Only ``gram``, ``sx`` and ``n`` (already accumulated for the descriptors) are used; the diagonal
    is 1 for units with non-trivial variance, rows / columns of constant units are 0.
    """
    s = stats.d
    n = s["n"].clamp_min(1.0)
    mean = s["sx"] / n
    cov = s["gram"] / n - mean[:, None] * mean[None, :]
    var = torch.diagonal(cov)
    ok = var > 1e-9
    sd = var.clamp_min(1e-12).sqrt()
    c = cov / (sd[:, None] * sd[None, :])
    c = c * (ok[:, None] & ok[None, :]).to(c.dtype)
    return c.clamp(-1, 1).to(torch.float32), n.to(torch.float32)


def no_decay_names():
    """Parameter names that must not receive weight decay (PMA seeds and temperatures)."""
    return ("u", "a", "log_ts", "log_tb")


def norm_entropy(w: torch.Tensor, dim: int) -> torch.Tensor:
    """Entropy of a probability tensor along ``dim`` divided by log(size): 1 = uniform, 0 = one-hot."""
    size = w.shape[dim]
    if size <= 1:
        return torch.ones_like(w.sum(dim))
    h = -(w * torch.log(w.clamp_min(1e-12))).sum(dim)
    return h / math.log(size)


class CrossAttention(nn.Module):
    """Pairwise cross-attention: uncovered queries over covered keys / values (see module doc)."""

    def __init__(self, fq: int, k: int, heads: int = 4, dk: int = 16, dv: int = 8,
                 pair_hidden: int = 16, pairwise: bool = True, pair_norm: bool = False):
        super().__init__()
        self.heads, self.dk, self.dv, self.pairwise = heads, dk, dv, bool(pairwise)
        self.pair_norm = bool(pair_norm)
        self.wq = nn.Linear(fq, heads * dk)
        self.wk = nn.Linear(fq, heads * dk)
        self.wv = nn.Linear(k + fq, heads * dv)
        self.pair = _mlp(2, heads, pair_hidden, layers=1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.out_dim = heads * dv + k

    def forward(self, xq, xk, prior_k, corr_uk, key_scalar, n):
        """xq [Nu, F], xk [Nc, F], prior_k [Nc, K], corr_uk [Nu, Nc], key_scalar [Nc], n scalar.

        Returns ctx [Nu, heads*dv + K], abar [Nu, K] (attention-averaged prior column), attn [Nu, Nc, H].
        """
        nu, nc = xq.shape[0], xk.shape[0]
        h = self.heads
        q = self.wq(xq).view(nu, h, self.dk)
        kk = self.wk(xk).view(nc, h, self.dk)
        v = self.wv(torch.cat([prior_k, xk], 1)).view(nc, h, self.dv)
        scale = torch.sqrt(n / 100.0)
        if self.pairwise:
            c = corr_uk
        else:                                             # per-key scalar: no (i, j) dependence
            c = key_scalar[None, :].expand(nu, nc)
        if self.pair_norm:       # row-standardise over covered keys: O(1) inputs (raw |c| is ~0.04)
            z = (c - c.mean(1, keepdim=True)) / (c.std(1, unbiased=False, keepdim=True) + 1e-2)
            pf = torch.stack([z, c * scale], -1)
        else:
            pf = torch.stack([c, c * scale], -1)          # [Nu, Nc, 2]
        s = torch.einsum("uhd,chd->uch", q, kk) / math.sqrt(self.dk) + self.pair(pf)
        attn = torch.softmax(s, dim=1)                    # over covered units
        ctx_h = torch.einsum("uch,chd->uhd", attn, v).reshape(nu, h * self.dv)
        abar = torch.einsum("uch,ck->uk", attn, prior_k) / h
        return torch.cat([ctx_h, abar], 1), abar, attn


# --------------------------------------------------------------- calibrator ----

class HybridCalibrator(nn.Module):
    def __init__(self, kind: str, k: int, fd: int, fr: int = REL_DIM, *, hidden: int = 64,
                 dctx: int = 32, m: int = 8, use_flag: bool = True, cut_coupling: bool = False,
                 pma_fix: bool = False, heads: int = 4, pair_norm: bool = False):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.k, self.fd, self.hidden, self.dctx, self.m = kind, k, fd, hidden, dctx, m
        self.use_flag = bool(use_flag)
        self.cut_coupling = bool(cut_coupling)
        self.pma_fix = bool(pma_fix)
        self.fdf = fd + (1 if self.use_flag else 0)
        self.xatt = kind.startswith("hyb_xatt")
        self.pooled = kind != "flat" and not self.xatt
        self.local = kind.endswith("_loc")
        self.family = kind.split("_")[1] if self.pooled else None      # ds | pma
        self.hybrid = kind.startswith("hyb")
        self.register_buffer("d_mu", torch.zeros(fd))
        self.register_buffer("d_sd", torch.ones(fd))
        self.register_buffer("r_mu", torch.zeros(fr))
        self.register_buffer("r_sd", torch.ones(fr))
        corr = torch.zeros(fd, dtype=torch.bool)
        corr[corr_columns(k)] = True
        self.register_buffer("corr_mask", corr)
        self.rel = self.mu0 = self.psi = self.phi = self.val = self.attn = None
        if self.pooled or self.xatt:
            self.rel = _mlp(fr, 1, 16, layers=1)
            nn.init.constant_(self.rel[-1].bias, 2.0)
            self.mu0 = nn.Parameter(torch.zeros(fd))
        if self.family == "ds":
            self.psi = _mlp(self.fdf, dctx, hidden)
            ctx_in = dctx
        elif self.family == "pma":
            self.phi = _mlp(self.fdf, hidden, hidden)
            init = 1.0 if self.pma_fix else 0.1
            self.u = nn.Parameter(torch.randn(m, hidden) * init)
            self.a = nn.Parameter(torch.randn(m, hidden) * init)
            if self.pma_fix:
                self.log_ts = nn.Parameter(torch.tensor(math.log(2.0)))
                self.log_tb = nn.Parameter(torch.tensor(math.log(2.0)))
            self.val = nn.Linear(hidden, dctx)
            ctx_in = dctx + 1                                       # + lam, as in P3/P4
        elif self.xatt:
            self.attn = CrossAttention(self.fdf, k, heads=heads, dk=16, dv=dctx // heads,
                                       pairwise=(kind == "hyb_xatt"), pair_norm=pair_norm)
            ctx_in = self.attn.out_dim
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

    def set_normalizers(self, d, r, covered=None):
        """Fit the frozen input standardisers.  With ``cut_coupling`` the corr columns are
        standardised over COVERED rows only (the uncovered rows are zeroed afterwards)."""
        with torch.no_grad():
            self.d_mu.copy_(d.mean(0))
            self.d_sd.copy_(d.std(0).clamp_min(1e-3))
            if self.cut_coupling and covered is not None:
                c = torch.as_tensor(covered, dtype=torch.bool, device=d.device)
                dc = d[c][:, self.corr_mask]
                self.d_mu[self.corr_mask] = dc.mean(0)
                self.d_sd[self.corr_mask] = dc.std(0).clamp_min(1e-3)
            self.r_mu.copy_(r.mean(0))
            self.r_sd.copy_(r.std(0).clamp_min(1e-3))

    # -- PMA ---------------------------------------------------------------
    def _pma_parts(self, dtf, lam):
        ph = self.phi(dtf)
        scale = math.sqrt(self.hidden)
        ts = torch.exp(self.log_ts) if self.pma_fix else 1.0
        tb = torch.exp(self.log_tb) if self.pma_fix else 1.0
        scores = ts * (ph @ self.u.T) / scale + torch.log(lam.clamp_min(1e-6))[:, None]
        pool_w = torch.softmax(scores, dim=0)                       # over units, per seed
        zj = pool_w.T @ self.val(ph)
        beta = torch.softmax(tb * (ph @ self.a.T) / scale, dim=1)   # per-unit routing over seeds
        return ph, pool_w, zj, beta

    def _context(self, dtf, lam):
        if self.family == "ds":
            z = self.psi(dtf)
            return z if self.local else z.mean(0, keepdim=True).expand(dtf.shape[0], -1)
        if self.local:                  # pooling over a single unit: softmax over a size-1 axis == 1
            mu_i = self.val(self.phi(dtf))
        else:
            _, _, zj, beta = self._pma_parts(dtf, lam)
            mu_i = beta @ zj
        return torch.cat([mu_i, lam[:, None]], 1)

    def _features(self, d, r, covered):
        cov = torch.as_tensor(covered, dtype=torch.bool, device=d.device)
        dn = (d - self.d_mu) / self.d_sd
        lam = None
        if self.pooled or self.xatt:
            rn = (r - self.r_mu) / self.r_sd
            lam = torch.sigmoid(self.rel(rn)).squeeze(-1)
            dn = lam[:, None] * dn + (1.0 - lam[:, None]) * self.mu0
        if self.cut_coupling:           # uncovered units: no latent-correlation (population) features
            dn = dn.masked_fill((~cov)[:, None] & self.corr_mask[None, :], 0.0)
        dtf = torch.cat([dn, cov[:, None].to(dn.dtype)], 1) if self.use_flag else dn
        return cov, dtf, lam

    def forward(self, d, r, prior, covered, pair=None):
        """d [N, Fd], r [N, Fr], prior [K, N], covered bool [N] -> table [K, N].

        ``pair`` = (corr [N, N], n) from ``pair_corr`` (only the xatt kinds read it).
        """
        cov, dtf, lam = self._features(d, r, covered)
        if self.xatt:
            return prior + self._xatt_out(dtf, prior, cov, pair)[0].T
        ctx = self._context(dtf, lam) if self.pooled else None
        if self.hybrid:
            out_c = self.gen_cov(dtf)
            out_u = self.gen_unc(torch.cat([dtf, ctx], 1))
            out = torch.where(cov[:, None], out_c, out_u)
        else:
            out = self.gen(torch.cat([dtf, ctx], 1) if ctx is not None else dtf)
        return prior + out.T

    def _xatt_out(self, dtf, prior, cov, pair):
        if pair is None:
            raise ValueError("xatt kinds need pair=(corr, n)")
        corr, n = pair
        out_c = self.gen_cov(dtf)
        iu = torch.nonzero(~cov).squeeze(-1)
        ic = torch.nonzero(cov).squeeze(-1)
        if len(iu) == 0 or len(ic) == 0:
            return out_c, None
        cc = corr[ic][:, ic]
        denom = max(len(ic) - 1, 1)
        key_scalar = (cc.sum(1) - torch.diagonal(cc)) / denom        # per-key: mean corr to other covered
        ctx, abar, attn = self.attn(dtf[iu], dtf[ic], prior.T[ic].to(dtf.dtype), corr[iu][:, ic],
                                    key_scalar, n)
        out_u = self.gen_unc(torch.cat([dtf[iu], ctx], 1)) + self.attn.gamma * abar
        return out_c.index_copy(0, iu, out_u), attn

    def calibrate(self, stats: PopStats, prior: torch.Tensor, covered):
        """Target-side entry: fixed-weight forward under no_grad (candidate result)."""
        cov = torch.as_tensor(np.asarray(covered), dtype=torch.bool, device=prior.device)
        if bool((prior.abs().sum(0)[~cov] != 0).any()):
            raise ValueError("uncovered units must have a zero prior column")
        with torch.no_grad():
            d, r, mean, std = describe(stats, prior)
            pair = pair_corr(stats) if self.xatt else None
            grad_mode = torch.is_grad_enabled()
            table = self(d, r, prior.to(torch.float32), cov, pair)
        assert not table.requires_grad
        return CalibResult(table=table.detach(), mean=mean, std=std, mode="candidate",
                           grad_enabled_inside=grad_mode)

    @torch.no_grad()
    def diagnostics(self, d, r, prior, covered, pair=None) -> dict:
        """Collapse monitors (normalised entropies are in [0, 1], 1 = uniform / collapsed).

        PMA (non-local): ``beta_entropy`` (per-unit routing over the m seeds), ``pool_entropy``
        (pooling weights over units, per seed), ``pool_maxN`` (max pooling weight x N; 1 = uniform),
        ``ctx_std`` (std of the context across units, mean over dims) and ``ctx_std_local`` (same
        for the per-unit local context, for scale).  XATT: ``attn_entropy`` (over covered keys) and
        ``attn_maxN`` (max attention weight x number of covered keys).  Empty dict otherwise.
        """
        cov, dtf, lam = self._features(d, r, covered)
        out = {}
        if self.family == "pma" and not self.local:
            ph, pool_w, zj, beta = self._pma_parts(dtf, lam)
            out["beta_entropy"] = float(norm_entropy(beta, 1).mean())
            out["pool_entropy"] = float(norm_entropy(pool_w, 0).mean())
            out["pool_maxN"] = float(pool_w.max(0).values.mean() * pool_w.shape[0])
            out["ctx_std"] = float((beta @ zj).std(0).mean())
            out["ctx_std_local"] = float(self.val(ph).std(0).mean())
        elif self.xatt:
            attn = self._xatt_out(dtf, prior, cov, pair)[1]
            if attn is not None:
                out["attn_entropy"] = float(norm_entropy(attn, 1).mean())
                out["attn_maxN"] = float(attn.max(1).values.mean() * attn.shape[1])
        return out


def match_width(target_params: int, k: int, fd: int, use_flag: bool = True, dctx: int = 32,
                m: int = 8, cut_coupling: bool = False) -> int:
    """Hidden width of a ``flat`` calibrator whose parameter count is closest to ``target_params``."""
    best, best_gap = None, None
    for h in range(16, 400):
        n = HybridCalibrator("flat", k, fd, hidden=h, dctx=dctx, m=m, use_flag=use_flag,
                                cut_coupling=cut_coupling).n_params
        gap = abs(n - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best
