"""Deployable forward-only refinements of a calibrated read-in table (feasibility P3).

Target-side contract: the trunk is frozen and is only ever run FORWARD, on the
B labeled calibration trials (replay), under ``torch.no_grad``.  No backprop
through the trunk on target data.  Two refinement ends are implemented:

* READ-IN, local rule F7-J.  The gradient of the decode loss w.r.t. the K x N
  read-in table is  -dL/dT = (1/n) sum_t e_t x_t^T  with the latent error
  e_t = sum_l J_{t+l,l}^T r_{t+l}.  F7-J replaces the (state dependent)
  Jacobian by its average J_l over source data, computed OFFLINE ONCE with
  autograd on source sessions only.  On target a step is: forward -> residual
  -> fixed temporal convolution -> outer-product accumulation -> per-unit
  preconditioning by q_i = mean x_i^2 -> prior-shrunk update.
* READOUT, closed form.  With features h_t (input of the trunk's linear
  readout) accumulated over the replayed trials, the prior-centred ridge is
  V = V0 + R (G + lam I)^-1 with G = mean h h^T, R = mean (y - V0 h) h^T.
  A diagonal and a low-rank (fixed source PCA basis) variant are provided.

NUMERICS (review item 4, rule 8): everything here is evaluated in fp32/fp64 on a GPU.  There is
NO fixed-point simulation of the accumulators, of the Jbar kernel (costed as int8 ROM in
``refine_cost`` but evaluated in fp32) or of the readout statistics.  The per-unit preconditioner
1/(q_i+eps) is a global scalar 1/(1+eps) in practice: inputs are z-scored on the same B trials
(valid bins) that are replayed, so q_i = mean x_i^2 = 1 exactly (up to the std floor).

The gradient-alignment helper uses autograd on the target trunk; it is a
DIAGNOSTIC (never used for calibration) and is kept in clearly named
functions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------- batching ----


def pad_trials(xz: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, windows):
    """Stack trial windows into padded tensors.

    xz [T, N] normalised inputs, y [T, D] normalised labels, valid [T] bool.
    Returns X [B, Tm, N], Y [B, Tm, D], M [B, Tm] float mask.  Padding is at
    the END of each trial, so (causal trunk) it never touches valid bins.
    """
    tm = max(b - a for a, b in windows)
    bn, n, d = len(windows), xz.shape[1], y.shape[1]
    X = torch.zeros(bn, tm, n, dtype=xz.dtype, device=xz.device)
    Y = torch.zeros(bn, tm, d, dtype=y.dtype, device=xz.device)
    M = torch.zeros(bn, tm, dtype=xz.dtype, device=xz.device)
    for i, (a, b) in enumerate(windows):
        X[i, :b - a], Y[i, :b - a], M[i, :b - a] = xz[a:b], y[a:b], valid[a:b].to(xz.dtype)
    return X, Y, M


def cap_windows(windows, cap):
    """Truncate every trial window to its first ``cap`` bins (fixed-size replay design)."""
    if cap is None:
        return [tuple(w) for w in windows]
    return [(a, min(b, a + int(cap))) for a, b in windows]


def concat_trials(xz: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, windows):
    """The calibration trials as ONE contiguous stream, in the given (recorded) order.

    Returns X [1, sum T, N], Y [1, sum T, D], M [1, sum T].  Trunk state is carried across the
    boundaries between calibration trials only (no non-calibration bins are inserted), so the
    stream is a surrogate for the unknown state at the start of each trial.
    """
    X = torch.cat([xz[a:b] for a, b in windows], 0)[None]
    Y = torch.cat([y[a:b] for a, b in windows], 0)[None]
    M = torch.cat([valid[a:b].to(xz.dtype) for a, b in windows], 0)[None]
    return X, Y, M


def run_trunk(base, X, table, want_h=False):
    """Forward the frozen trunk on a table; optionally return readout features."""
    z = X @ table.to(X.dtype).T
    if not want_h:
        return base(z)
    store = {}
    handle = base.out_proj.register_forward_pre_hook(lambda m, i: store.__setitem__("h", i[0]))
    try:
        y = base(z)
    finally:
        handle.remove()
    return y, store["h"]


# ------------------------------------------------------ average Jacobian ----


def jacobian_at_last_bin(base, z_win: torch.Tensor):
    """Per-window Jacobian of the LAST output bin w.r.t. lagged latents.

    z_win [M, W, K] -> J [M, W, D, K] with J[m, W-1-l] = d y_{W-1} / d z_{W-1-l}
    (so ``J.flip(1)[:, l]`` is the lag-l block).  Uses autograd: OFFLINE and
    SOURCE-ONLY.
    """
    z = z_win.detach().clone().requires_grad_(True)
    y = base(z)[:, -1, :]                                     # [M, D]
    rows = []
    for d in range(y.shape[1]):
        g, = torch.autograd.grad(y[:, d].sum(), z, retain_graph=d + 1 < y.shape[1])
        rows.append(g)                                        # [M, W, K]
    return torch.stack(rows, 2)                               # [M, W, D, K]


def lag_jacobians(base, z_win: torch.Tensor, lmax: int):
    """[M, lmax+1, D, K] lag-resolved Jacobians (lag 0 = same bin)."""
    j = jacobian_at_last_bin(base, z_win)
    return j.flip(1)[:, :lmax + 1]


def average_jacobian_session(base, xz, table, ends, lmax=32, width=320, chunk=64):
    """Sum of lag Jacobians over windows ending at ``ends`` (bins of one source session).

    Returns dict(sum [lmax+1, D, K], sq [lmax+1] sum of squared Frobenius norms,
    n windows).  Windows start at end-width+1 with zero trunk state.
    """
    z_all = xz @ table.to(xz.dtype).T
    tot, sq, n = None, torch.zeros(lmax + 1, dtype=torch.float64, device=xz.device), 0
    ends = [int(e) for e in ends if e >= width - 1]
    for lo in range(0, len(ends), chunk):
        part = ends[lo:lo + chunk]
        win = torch.stack([z_all[e - width + 1:e + 1] for e in part])
        j = lag_jacobians(base, win, lmax).double()          # [m, L+1, D, K]
        tot = j.sum(0) if tot is None else tot + j.sum(0)
        sq += (j ** 2).sum((2, 3)).sum(0)
        n += len(part)
    return {"sum": tot, "sq": sq, "n": n}


def jacobian_representativeness(sums: dict, n: int):
    """Relative deviation of single-window Jacobians from the mean, per lag.

    rel_l = sqrt(E||J_l - Jbar_l||_F^2) / ||Jbar_l||_F, from the stored sums
    (E||J||^2 - ||Jbar||^2 = E||J-Jbar||^2).
    """
    jbar = sums["sum"] / n
    var = (sums["sq"] / n - (jbar ** 2).sum((1, 2))).clamp_min(0)
    return (var.sqrt() / jbar.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)).cpu().numpy()


# ----------------------------------------------------------- F7-J update ----


def temporal_feedback(r: torch.Tensor, jbar: torch.Tensor):
    """e_t = sum_{l=0..L} Jbar_l^T r_{t+l};  r [B, T, D], jbar [L+1, D, K] -> [B, T, K].

    A fixed (non-adaptive) temporal convolution; residuals beyond the end of a
    trial are zero (the trial is a closed calibration recording).  Needs an
    L-bin delay line of residuals on-chip.
    """
    lmax = jbar.shape[0] - 1
    t = r.shape[1]
    rp = F.pad(r, (0, 0, 0, lmax))
    e = torch.zeros(r.shape[0], t, jbar.shape[2], dtype=r.dtype, device=r.device)
    for l in range(lmax + 1):
        e = e + rp[:, l:l + t] @ jbar[l].to(r.dtype)
    return e


def fj_gradient(base, X, Y, M, table, jbar):
    """One replay forward + feedback + outer product.

    Returns (G [K, N], q [N], n_valid, r [B, T, D]) with
    G = (1/n) sum_t e_t x_t^T  (= -dL/dT for L = sum_t |r_t|^2 / (2n) when
    ``jbar`` holds the exact Jacobians) and q_i = mean_t x_i^2 over valid bins.
    """
    with torch.no_grad():
        yhat = run_trunk(base, X, table)
        r = (Y - yhat) * M.unsqueeze(-1)
        e = temporal_feedback(r, jbar)
        n = M.sum().clamp_min(1.0)
        xm = X * M.unsqueeze(-1)
        g = torch.einsum("btk,btn->kn", e, X) / n          # all bins: invalid bins still feed the trunk
        q = (xm * X).sum((0, 1)) / n
    return g, q, n, r


def fj_update(table, g, q, center, eta, lam, eps=1e-3, precondition=True):
    """T <- T + eta G diag(1/(q+eps)) - eta lam (T - center)."""
    step = g / (q + eps)[None, :] if precondition else g
    return table + eta * step - eta * lam * (table - center)


def fj_refine(base, X, Y, M, table0, jbar, eta, lam, snapshots, center=None, eps=1e-3,
              precondition=True, with_diag=None):
    """S forward-only F7-J steps from ``table0``; returns {step: table}.

    ``with_diag`` (callable(step, table, g, upd)) is invoked at each step for gradient-alignment
    diagnostics; ``g`` is the raw F7-J gradient estimate and ``upd`` = new table - table is the
    REALISED update (preconditioner and shrinkage included).
    """
    snapshots = sorted(set(int(s) for s in snapshots))
    center = table0 if center is None else center
    tab, out = table0.clone(), {}
    for step in range(1, snapshots[-1] + 1):
        g, q, _, _ = fj_gradient(base, X, Y, M, tab, jbar)
        new = fj_update(tab, g, q, center, eta, lam, eps, precondition)
        if with_diag is not None:
            with_diag(step, tab, g, new - tab)
        tab = new
        if step in snapshots:
            out[step] = tab.clone()
    return out


# --------------------------------------------- gradient alignment (DIAGNOSTIC) ----


def diagnostic_true_neg_gradient(base, X, Y, M, table):
    """-dL/dT of L = sum |r|^2/(2n) by backprop through the trunk.

    DIAGNOSTIC ONLY: offline gradient-alignment measurement on target trials;
    never used to produce a calibrated table.
    """
    tab = table.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        yhat = base(X @ tab.T)
        n = M.sum().clamp_min(1.0)
        loss = (((Y - yhat) ** 2).sum(-1) * M).sum() / (2.0 * n)
        g, = torch.autograd.grad(loss, tab)
    return -g.detach()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float("nan")


# --------------------------------------------------------- readout ridge ----


def head_matrix(base):
    """Augmented readout [D, H+1] (weights | bias) of the trunk's linear head."""
    return torch.cat([base.out_proj.weight, base.out_proj.bias[:, None]], 1).detach()


def readout_stats(base, X, Y, M, table):
    """Replay forward and accumulate readout statistics (all additive).

    Returns dict(G [H+1,H+1], R [D,H+1], n) with
    G = sum_t m_t hb_t hb_t^T, R = sum_t m_t (y_t - V0 hb_t) hb_t^T, hb = [h, 1].
    """
    with torch.no_grad():
        yhat, h = run_trunk(base, X, table, want_h=True)
        hb = torch.cat([h, torch.ones_like(h[..., :1])], -1) * M.unsqueeze(-1)
        r0 = (Y - yhat) * M.unsqueeze(-1)
        G = torch.einsum("bti,btj->ij", hb, hb).double()
        R = torch.einsum("btd,btj->dj", r0, hb).double()
    return {"G": G, "R": R, "n": float(M.sum())}


def add_readout_stats(a, b):
    return {"G": a["G"] + b["G"], "R": a["R"] + b["R"], "n": a["n"] + b["n"]}


def ridge_delta(st, lam, kind="full", basis=None):
    """Prior-centred ridge correction dV [D, H+1] (V = V0 + dV).

    kind: ``full`` (G + lam I), ``diag`` (diag(G) + lam, per feature),
    ``lowrank`` (project on the columns of ``basis`` [H+1, r], r x r solve).
    G and R are normalised by the number of valid bins.
    """
    n = max(st["n"], 1.0)
    G, R = st["G"] / n, st["R"] / n
    if kind == "full":
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return torch.linalg.solve(G + lam * eye, R.T).T
    if kind == "diag":
        return R / (torch.diagonal(G) + lam)[None, :]
    if kind == "lowrank":
        u = basis.to(G.dtype)
        gu, ru = u.T @ G @ u, R @ u
        eye = torch.eye(gu.shape[0], dtype=G.dtype, device=G.device)
        return torch.linalg.solve(gu + lam * eye, ru.T).T @ u.T
    raise ValueError(kind)


def feature_basis(second_moment: torch.Tensor, r: int):
    """Top-r eigenvectors [H+1, r] of a source feature second-moment matrix."""
    w, v = torch.linalg.eigh(second_moment.double())
    return v[:, torch.argsort(w, descending=True)[:r]]


def decode_with_readout(base, X_full, table, v_aug=None):
    """Contiguous causal decode of a whole session, optionally with a new readout."""
    with torch.inference_mode():
        y0, h = run_trunk(base, X_full.unsqueeze(0), table, want_h=True)
        if v_aug is None:
            return y0.squeeze(0)
        hb = torch.cat([h, torch.ones_like(h[..., :1])], -1)
        return (hb @ v_aug.to(hb.dtype).T).squeeze(0)


# ----------------------------------------------------------- raw counters ----


def trunk_macs_per_bin(base) -> dict:
    """MACs of one forward bin of the trunk, counted from the module tree.

    Two classes, kept apart (feasibility rule 8: "fixed-weight MAC" = weight x activation only):

    * ``fixed_weight_macs`` (= ``linear_macs``): ``nn.Linear`` layers, in*out MACs each (Mamba3
      in/out projections and the FFN); weights are fixed (RRAM-mappable), the other operand is an
      activation.
    * ``act_x_act_macs`` (= ``scan_macs``): the Mamba3 SISO selective scan.  The recurrent state
      has d_inner * d_state elements per layer; per bin each element needs one multiply-add for
      the state update (a*h + B*x) and one for the readout (C.h), i.e. 2 MACs.  Both operands are
      activations (input-dependent a, B, C and the state), so this is NOT a fixed-weight MAC.

    Gating, RoPE, layer-norm and the short convolution are not counted (``uncounted``).
    ``state_elems`` is the recurrent state kept between bins (all layers).
    """
    lin, scan, state = 0, 0, 0
    for _, mod in base.named_modules():
        if isinstance(mod, torch.nn.Linear):
            lin += mod.in_features * mod.out_features
        if hasattr(mod, "d_state") and hasattr(mod, "headdim") and hasattr(mod, "d_inner"):
            scan += 2 * int(mod.d_inner) * int(mod.d_state)
            state += int(mod.d_inner) * int(mod.d_state)
    return {"linear_macs": int(lin), "scan_macs": int(scan), "total": int(lin + scan),
            "fixed_weight_macs": int(lin), "act_x_act_macs": int(scan),
            "state_elems": int(state), "uncounted": "gating, RoPE, layer-norm, short conv"}
