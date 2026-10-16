"""Population-style static calibration for FALCON M2 (fixed channels), Colab round.

Setting.  The M2 trunk is a frozen "wide_n16" Mamba-3 decoder (width 256, 4 layers).  Its
root ``in_proj`` is the K x N read-in table (K=256, N=96) plus a K-vector bias.  Channel
identity is FIXED across sessions, so the identity-aligned prior is the source MEAN read-in
table (the folded root in_proj = W0 diag(mean gain), bias = W0 mean(offset) + b0 + mean
embedding; the per-session tables of the source sessions come from ``session_bank_best.pt``).

Target-side contract (same as P1-P3): the trunk is frozen and only run forward; calibration is
(a) additive streaming statistics over the B labeled trials, (b) a fixed-weight forward of a
calibrator trained episodically on SOURCE sessions only, (c) optionally a closed-form
readout ridge accumulated trial by trial (R1-stream).  The bias of the folded in_proj is a
frozen trunk constant (it is not calibrated).

Source simulation.  Source sessions are in-sample for the trunk, so a simulated episode takes
its B calibration trials from the first 80% of the usable trials of a source session and
scores the last 20% (the trunk's own held-out validation trials, never used for gradients).
The prior of a source session is leave-one-out (mean over the OTHER source sessions).
"""
from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from . import fwd_refine as fr
from .eval_checks import check_physical_units
from .population_calib import (CalibResult, PopStats, PopulationCalibrator, SessionData,
                               _mlp, desc_dim, describe, norm_stats)

ROOT = Path(__file__).resolve().parents[1]
M2_CACHE = ROOT / "results" / "data_cache" / "m2" / "held_in"
N_CHANNELS = 96
B_GRID = (1, 2, 4, 8, 16, 33)

BACKBONE_ROOTS = {
    0: "results/forward_eval_m2_wide_n16_20261005",
    1: "results/forward_eval_m2_wide_n16_s1_20261006",
    2: "results/forward_eval_m2_wide_n16_s2_20261006",
}
POOL_DIRS = {
    "pool1": "pools/pool1_2020-10-19_to_2020-10-20",
    "pool2": "pools/pool2_2020-10-20_to_2020-10-27",
    "pool3": "pools/pool3_2020-10-27_to_2020-10-28",
}


# ------------------------------------------------------------------- data ----

def load_m2_recording(session: str, cache_dir: Path = M2_CACHE):
    """Held-in FALCON M2 recording from the npz cache (no NWB needed).

    Only the ``held_in`` split is ever read.  Channel identity is fixed, so
    ``unit_to_electrode`` is the identity map.
    """
    path = Path(cache_dir) / f"{session}.npz"
    blob = np.load(path, allow_pickle=False)
    return SimpleNamespace(
        session=session, neural=blob["neural"].astype(np.float64),
        behavior=blob["behavior"].astype(np.float64), eval_mask=blob["eval_mask"].astype(bool),
        trial_bounds=tuple((int(a), int(b)) for a, b in blob["trial_bounds"]),
        unit_to_electrode=np.arange(blob["neural"].shape[1]))


def draw_trials(name, B, draw, n_trials):
    """Same trial-subset rule as P1 (``run_population_calib.draw_trials``)."""
    seed = zlib.crc32(f"{name}-{B}-{draw}".encode())
    return np.sort(np.random.default_rng(seed).choice(n_trials, size=B, replace=False))


def resolve_pool_dir(seed: int, pool: str, root: Path = ROOT) -> Path:
    """Backbone directory of (seed, pool).  If the base run's best step fell in the last 20% of
    its budget, the doubled-budget ``_x2`` run is the backbone (extension rule)."""
    base = Path(root) / BACKBONE_ROOTS[seed] / POOL_DIRS[pool]
    x2 = base.with_name(base.name + "_x2")
    if x2.is_dir():
        man = json.loads((base / "manifest.json").read_text())
        if man["best_step"] > 0.8 * man["args"]["steps"]:
            return x2
    return base


# ------------------------------------------------------------------ trunk ----

def trunk_from_latent(model, z, want_h=False):
    """Frozen trunk after the input layer: blocks -> final_norm -> out_proj."""
    for b in model.blocks:
        z = b(z)
    h = model.final_norm(z)
    y = model.out_proj(h)
    return (y, h) if want_h else y


def fold_bank(state_dict):
    """Per-session folded read-in tables and biases from a session-bank state dict.

    Training model: z = W0 (g_s * x + o_s) + b0 + e_s  =>  table_s = W0 diag(g_s),
    bias_s = W0 o_s + b0 + e_s.  Returns (list of [K, N] float64, list of [K] float64).
    """
    w0 = state_dict["base.in_proj.weight"].double()
    b0 = state_dict["base.in_proj.bias"].double()
    gain, off, emb = (state_dict["bank.gain"].double(), state_dict["bank.bias"].double(),
                      state_dict["bank.embedding"].double())
    tables = [(w0 * gain[j][None, :]).numpy() for j in range(gain.shape[0])]
    biases = [(w0 @ off[j] + b0 + emb[j]).numpy() for j in range(gain.shape[0])]
    return tables, biases


class M2Trunk:
    """Frozen M2 backbone + its per-source-session read-in tables (from the session bank)."""

    def __init__(self, pool_dir: Path, device):
        from .mamba3_official import build_mamba3_official
        pool_dir = Path(pool_dir)
        self.pool_dir, self.device = pool_dir, device
        blob = torch.load(pool_dir / "best.pt", map_location="cpu", weights_only=False)
        a = blob["args"]
        self.width, self.layers, self.state = a["width"], a["layers"], a["state_size"]
        self.model = build_mamba3_official(N_CHANNELS, 2, self.width, self.layers, self.state,
                                           0.0).to(device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.k = self.width
        norm = np.load(pool_dir / "normalizer.npz")
        self.x_mean, self.x_std = norm["x_mean"].astype(np.float64), norm["x_std"].astype(np.float64)
        self.label_stats = (norm["y_mean"].astype(np.float64), norm["y_std"].astype(np.float64))
        bank = torch.load(pool_dir / "session_bank_best.pt", map_location="cpu", weights_only=False)
        sd = bank["state_dict"]
        self.source_ids = list(bank["source_sessions"])
        self.src_tables, self.src_bias = fold_bank(sd)
        self.prior_table = np.mean(self.src_tables, 0)
        self.prior_bias = np.mean(self.src_bias, 0)
        self.folded_table = blob["state_dict"]["in_proj.weight"].double().numpy()
        self.folded_bias = blob["state_dict"]["in_proj.bias"].double().numpy()
        self.best_step = blob.get("best_step")
        self.trunk_hash = self._hash()

    def _hash(self):
        h = hashlib.sha256()
        for n, p in self.model.named_parameters():
            if not n.startswith("in_proj."):
                h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def loo_prior(self, exclude):
        """Mean read-in table and bias over the source sessions other than ``exclude``."""
        keep = [j for j in range(len(self.src_tables)) if j != exclude]
        return (np.mean([self.src_tables[j] for j in keep], 0),
                np.mean([self.src_bias[j] for j in keep], 0))

    # interface used by the generic routines below
    def run(self, z, want_h=False):
        return trunk_from_latent(self.model, z, want_h)

    def head(self):
        return fr.head_matrix(self.model)


# ------------------------------------------------------------------ world ----

def val_start_bin(record) -> int:
    """First bin of the trunk's own validation trials (last 20% of the raw trials)."""
    b = [z for z in record.trial_bounds if z[1] - z[0] >= 2]
    cut = max(1, int(len(b) * 0.8))
    return int(b[cut][0])


class M2World:
    def __init__(self, seed, pool, device, root: Path = ROOT, load_sources=True, trunk=None):
        from scripts.run_labeled_calibration import D2_POOLS
        self.seed, self.pool, self.device = seed, pool, device
        spec = D2_POOLS[pool]
        self.spec = spec
        self.trunk = trunk or M2Trunk(resolve_pool_dir(seed, pool, root), device)
        self.k = self.trunk.k
        self.label_stats = self.trunk.label_stats
        self.source_ids = spec["sources"].split(",")
        assert self.source_ids == self.trunk.source_ids, (self.source_ids, self.trunk.source_ids)
        self.targets = list(spec["targets"])
        self.sessions, self.priors, self.val_start = {}, {}, {}
        if load_sources:
            for j, s in enumerate(self.source_ids):
                rec = load_m2_recording(s)
                sd = SessionData(rec, self.label_stats, device, name=s)
                self.sessions[s] = sd
                tab, bias = self.trunk.loo_prior(j)
                self.priors[s] = (torch.from_numpy(tab).to(device), torch.from_numpy(bias).to(device))
                self.val_start[s] = val_start_bin(rec)
        # fixed projection of the prior latents (source-only): top-r left singular vectors
        u, _, _ = np.linalg.svd(self.trunk.prior_table, full_matrices=False)
        self.u_full = torch.from_numpy(u).float().to(device)           # [K, N]

    def target(self, name):
        rec = load_m2_recording(name)
        sd = SessionData(rec, self.label_stats, self.device, name=name)
        prior = (torch.from_numpy(self.trunk.prior_table).to(self.device),
                 torch.from_numpy(self.trunk.prior_bias).to(self.device))
        return sd, prior

    def train_trials(self, name):
        """Indices (into sd.trials) of source trials strictly before the validation region."""
        sd, vs = self.sessions[name], self.val_start[name]
        return np.array([i for i, (a, b) in enumerate(sd.trials) if b <= vs], dtype=np.int64)

    def val_mask(self, name):
        sd, vs = self.sessions[name], self.val_start[name]
        m = sd.valid.clone()
        in_val = torch.zeros_like(m)
        for a, b in sd.trials:
            if a >= vs:
                in_val[a:b] = True
        return m & in_val


# ------------------------------------------------------------ calibrator ----

class M2Calibrator(PopulationCalibrator):
    """P0 for M2: independent per-channel encoder, shared across channels.

    Descriptors are the P1 ``describe`` features computed against an r-dimensional projection
    of the K=256 prior latents (fixed source SVD basis U).  Output per channel: a K-vector
    residual plus (optionally) a scalar log-gain on the prior column,
    T_i = (1 + a_i) * prior_i + residual_i.  The last layer is zero-initialised, so the
    untrained calibrator reproduces the prior table exactly.
    """

    def __init__(self, k: int, rank: int = 16, hidden: int = 64, gain: bool = True,
                 label_free: bool = False):
        super().__init__("p0", rank, desc_dim(rank), hidden=hidden, output="residual",
                         label_free=label_free)
        self.k_full, self.rank, self.use_gain = k, rank, bool(gain)
        self.register_buffer("U", torch.zeros(k, rank))
        self.gen = _mlp(desc_dim(rank), k + int(gain), hidden, zero_last=True)

    def set_projection(self, u_full: torch.Tensor):
        with torch.no_grad():
            self.U.copy_(u_full[:, :self.rank])

    def project(self, prior: torch.Tensor):
        return self.U.T @ prior.to(self.U.dtype)                       # [r, N]

    def forward(self, d, r, prior_full, return_aux=False):
        dn = (d - self.d_mu) / self.d_sd * self.d_keep
        out = self.gen(dn)                                             # [N, K(+1)]
        table = prior_full + out[:, :self.k_full].T
        if self.use_gain:
            table = table + prior_full * out[:, self.k_full][None, :]
        return (table, {}) if return_aux else table

    def calibrate(self, stats: PopStats, prior: torch.Tensor, *, oracle: bool = False):
        with torch.no_grad():
            d, r, mean, std = describe(stats, self.project(prior))
            table = self(d, r, prior.to(torch.float32))
        assert not table.requires_grad
        return CalibResult(table=table.detach(), mean=mean, std=std,
                           mode="oracle" if oracle else "candidate")


def build_calibrator(world: M2World, rank=16, hidden=64, gain=True, label_free=False) -> M2Calibrator:
    c = M2Calibrator(world.k, rank, hidden, gain, label_free).to(world.device)
    c.set_projection(world.u_full)
    return c


def describe_for(calib: M2Calibrator, stats: PopStats, prior):
    return describe(stats, calib.project(prior))


# -------------------------------------------------------------- decoding ----

def latent(sd: SessionData, table, bias, mean, std, lo=0, hi=None):
    xz = (sd.x32[lo:hi] - mean.float()) / std.float()
    return xz @ table.float().T + bias.float()


def decode_full(trunk, sd, table, bias, mean, std, v_aug=None, want_h=False):
    """One contiguous causal pass over the whole session (state never reset)."""
    with torch.no_grad():
        z = latent(sd, table, bias, mean, std)[None]
        y, h = trunk.run(z, want_h=True)
        y, h = y[0], h[0]
        if v_aug is not None:
            hb = torch.cat([h, torch.ones_like(h[:, :1])], 1)
            y = hb @ v_aug.to(hb.dtype).T
    return (y, h) if want_h else y


def r2_physical(label_stats, sd, pred_norm, qmask):
    ym, ys = label_stats
    pred = pred_norm[qmask].double().cpu().numpy() * ys + ym
    truth = sd.record.behavior[qmask.cpu().numpy()].astype(np.float64)
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12), check_physical_units(pred, truth)


def srcnorm_stats(trunk, device):
    """Label-free source input normalisation (diagnostic prior row)."""
    std = np.where(trunk.x_std < 1e-6, 1.0, trunk.x_std)
    return (torch.from_numpy(trunk.x_mean).to(device), torch.from_numpy(std).to(device))


# ----------------------------------------------------- int8 table (size) ----

def quantize_table_int8(table: torch.Tensor):
    """Per-row symmetric int8 (post-training).  Returns (dequantised table, int8 bytes incl. scales)."""
    t = table.float()
    scale = (t.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(t / scale).clamp(-127, 127)
    return q * scale, int(q.numel() + 4 * scale.numel())


def table_bytes(k: int, n: int):
    return {"K": k, "N": n, "params": k * n, "fp32_bytes": 4 * k * n, "int8_bytes": k * n,
            "int8_plus_row_scales_bytes": k * n + 4 * k, "budget_bytes": 32 * 1024,
            "fp32_fits": 4 * k * n <= 32 * 1024, "int8_fits": k * n + 4 * k <= 32 * 1024}


# ------------------------------------------------------- R1-stream readout ----

def stream_readout_stats(trunk, sd: SessionData, prior, calib, idx, burn=0, use_calib=True):
    """Readout statistics accumulated while the B trials stream in (no replay buffer).

    Trial j (chronological) is forwarded ONCE, from the zero state, with the table and input
    normalisation obtained from trials < j (trial 0: prior table, normalisation from the trial
    itself = a one-trial buffer).  ``burn`` bins at the start of each trial are excluded from the
    statistics (the zero-state transient).  With ``use_calib=False`` the table is the prior.
    The trials are forwarded as one zero-padded batch; padding is at the END of each trial so
    it never reaches valid bins (causal trunk), i.e. identical to separate forwards.
    """
    prior_tab, prior_bias = prior
    idx = np.sort(np.asarray(idx))
    zs, ys, ms = [], [], []
    for j, tr in enumerate(idx):
        if j == 0 or not use_calib:
            mean, std = norm_stats(PopStats.from_trials(sd.bank, idx[:1] if j == 0 else idx[:j]))
            table = prior_tab.float()
        else:
            res = calib.calibrate(PopStats.from_trials(sd.bank, idx[:j]), prior_tab)
            mean, std, table = res.mean, res.std, res.table.float()
        a, b = sd.trials[tr]
        zs.append(latent(sd, table, prior_bias, mean, std, a, b))
        ys.append(sd.y_norm.float()[a:b])
        m = sd.valid[a:b].to(torch.float32).clone()
        m[:burn] = 0.0
        ms.append(m)
    tm = max(len(m) for m in ms)
    K, D = zs[0].shape[1], ys[0].shape[1]
    Z = torch.zeros(len(zs), tm, K, device=sd.device)
    Y = torch.zeros(len(zs), tm, D, device=sd.device)
    M = torch.zeros(len(zs), tm, device=sd.device)
    for i, (z, y, m) in enumerate(zip(zs, ys, ms)):
        Z[i, :len(z)], Y[i, :len(z)], M[i, :len(z)] = z, y, m
    with torch.no_grad():
        yhat, h = trunk.run(Z, want_h=True)
        hb = torch.cat([h, torch.ones_like(h[..., :1])], -1) * M.unsqueeze(-1)
        r0 = (Y - yhat) * M.unsqueeze(-1)
        G = torch.einsum("bti,btj->ij", hb, hb).double()
        R = torch.einsum("btd,btj->dj", r0, hb).double()
    return {"G": G, "R": R, "n": float(M.sum())}


def ridge_v(trunk, st, lam, kind="full"):
    """Prior-centred ridge readout V = V0 + dV (augmented [D, H+1]); lam=None means no change."""
    v0 = trunk.head().double()
    if lam is None or st["n"] < 1:
        return v0
    return v0 + fr.ridge_delta(st, lam, kind)


# ---------------------------------------------------- ui_reg (reference) ----

def ui_reg_trajectory_m2(trunk, x_z, y_norm, trial_windows, table_init, bias, center, *, lam, lr,
                         snapshots, device, seed=0, valid_mask=None, pad_multiple=64):
    """Regularised gradient fit of the read-in table only (NON-DEPLOYABLE reference).

    Same recipe as ``ui_reg.ui_reg_trajectory`` (one random labeled trial per Adam step,
    prior-centred squared-Frobenius penalty) with the frozen folded bias added to the latents.
    Returns {step: table [K, N] float64}.
    """
    snapshots = sorted(set(int(s) for s in snapshots))
    torch.manual_seed(seed)
    table = nn.Parameter(torch.from_numpy(np.asarray(table_init, np.float32)).clone().to(device))
    ctr = torch.from_numpy(np.asarray(center, np.float32)).to(device)
    b = torch.from_numpy(np.asarray(bias, np.float32)).to(device)
    x = torch.from_numpy(x_z.astype(np.float32)).to(device)
    y = torch.from_numpy(y_norm.astype(np.float32)).to(device)
    mask = torch.from_numpy(valid_mask).to(device) if valid_mask is not None else None
    rng = np.random.default_rng(seed)
    windows = [(a, c) for a, c in trial_windows if c - a >= 2]
    opt = torch.optim.Adam([table], lr=lr)
    out = {}
    for step in range(1, snapshots[-1] + 1):
        a, c = windows[int(rng.integers(len(windows)))]
        opt.zero_grad(set_to_none=True)
        # pad the window (at the END, with zeros) to a multiple of 64 bins: the Mamba-3 Triton backward
        # re-autotunes for every new sequence length, so varying trial lengths would make it unusable.
        # The trunk is causal, so the padding cannot influence the window's own outputs.
        L = c - a
        Lp = int(math.ceil(L / pad_multiple) * pad_multiple)
        z = x[a:c] @ table.T + b
        if Lp > L:
            z = torch.cat([z, z.new_zeros(Lp - L, z.shape[1])], 0)
        decode = trunk.run(z[None])[0][:L]
        target = y[a:c]
        good = torch.isfinite(target).all(-1)
        if mask is not None:
            good = good & mask[a:c]
        if good.any():
            loss = (decode[good] - target[good]).square().mean() + lam * (table - ctr).square().sum()
            loss.backward()
            opt.step()
        if step in snapshots:
            out[step] = table.detach().cpu().numpy().astype(np.float64)
    return out
