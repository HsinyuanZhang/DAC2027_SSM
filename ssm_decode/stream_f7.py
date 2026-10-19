"""Replay-free streaming fixed-feedback refinement (F7-J summation reordering).

The batch F7-J data term in ``fwd_refine`` is

    e_t = sum_{l=0}^{L-1} Jbar_l^T r_{t+l},   U = sum_t e_t x_t^T

with residuals past the end of each trial equal to zero (``temporal_feedback``).
Reindexing s = t+l gives the same matrix

    U = sum_s sum_l (Jbar_l^T r_s) x_{s-l}^T

which is accumulated from the current residual and a ring of past inputs.  For one
fixed forward trajectory, one fixed kernel and the same per-trial zero padding, the
two sums are identical (not an approximation).

Streaming variants below are a different algorithm from S-step replay (R2): the table
is committed at calibration-trial boundaries and past trials are not forwarded again.
See ``CAUSALITY``.

Target-side entry points run under ``torch.no_grad`` and never call autograd.
"""
from __future__ import annotations

import math

import torch

from ssm_decode import fwd_refine as fr

EPS_Q = 1e-3

# Stated causality of each variant.  "Future" means a residual or label at a bin
# the variant is not allowed to have read yet.  A calibration trial is a closed
# labeled recording: its own bins may be accumulated before the boundary commit,
# but the per-bin contribution itself is causal, and later trials are not read.
CAUSALITY = {
    "r2_replay": (
        "Not live. Each of S steps replays the stored B trials from the zero state. "
        "The latent error e_t = sum_l Jbar_l^T r_{t+l} reads up to L-1 future residuals "
        "inside the stored trial (acausal within the trial; the trial is already recorded). "
        "No bin outside the B trials is read. Query decoding is a separate contiguous causal pass."
    ),
    "fir_reorder": (
        "Same sum as r2_replay on a fixed trajectory and the same per-trial zero padding, "
        "written as current residual times past inputs. A per-bin contribution at time t "
        "depends only on r_t and x_{<=t}. It does not read future residuals, future inputs, "
        "other trials, or query bins."
    ),
    "s1_fir": (
        "One forward per calibration trial, table held fixed inside the trial (zero trunk "
        "state at each trial start, matching pad_trials / the headline R2 boundary). "
        "The FIR accumulator is the reordered sum; the shrinkage-toward-P0 commit happens "
        "only at the trial boundary (anytime). Later trials do not change earlier commits. "
        "It does not read a future residual, a future trial, or a query bin. Because the table changes "
        "between trials, the pass is not equal to one R2 step and not equal to S-step replay."
    ),
    "trace": (
        "Same commit schedule, zero-state trial boundary and label access as s1_fir. "
        "Each trace s_{m,t} = rho_m s_{m,t-1} + x_t is reset at the trial start (x_<0 = 0), "
        "so it is a causal IIR of inputs inside the trial. It approximates the FIR when "
        "Jbar is a sum of exponentials; it does not read future residuals. The tail l >= L "
        "is the exponential extrapolation, not data from the future."
    ),
    "s1_r1s": (
        "s1_fir, and on the same live forward the readout Gram is accumulated under the "
        "table that was active during the trial (before that trial's commit). The ridge "
        "is solved once after the last trial. Statistics are not recomputed on the final "
        "table (that would be a second pass). No future trial and no query label is read."
    ),
}


def _as_batch(X: torch.Tensor, r: torch.Tensor):
    if X.dim() == 2:
        return X.unsqueeze(0), r.unsqueeze(0), True
    if X.dim() != 3:
        raise ValueError(f"X must be [T,N] or [B,T,N], got {tuple(X.shape)}")
    return X, r, False


def fir_data_term(X: torch.Tensor, r: torch.Tensor, jbar: torch.Tensor) -> torch.Tensor:
    """U [K, N] = sum_t sum_l (Jbar_l^T r_t) x_{t-l}^T.

    ``X`` is [T, N] or [B, T, N], ``r`` is [T, D] or [B, T, D] (already masked),
    ``jbar`` is [L, D, K].  Inputs before time 0 and residuals past each row's end
    are zero.  Rows do not mix (per-trial padding).
    """
    Xb, rb, _ = _as_batch(X, r)
    if rb.shape[-1] != jbar.shape[1]:
        raise ValueError(f"residual D {rb.shape[-1]} != jbar D {jbar.shape[1]}")
    t = Xb.shape[1]
    n_lags = jbar.shape[0]
    u = Xb.new_zeros(jbar.shape[2], Xb.shape[2])
    for lag in range(n_lags):
        jb = jbar[lag].to(dtype=Xb.dtype, device=Xb.device)
        if lag == 0:
            corr = torch.einsum("btd,btn->dn", rb, Xb)
        elif lag >= t:
            continue
        else:
            corr = torch.einsum("btd,btn->dn", rb[:, lag:], Xb[:, : t - lag])
        u = u + jb.transpose(0, 1) @ corr
    return u


def fir_per_time(X: torch.Tensor, r: torch.Tensor, jbar: torch.Tensor) -> torch.Tensor:
    """Per-bin contribution [T, K, N] of ``fir_data_term`` for a single trial.

    contrib[t] depends only on ``r[t]`` and ``X[:t+1]``.
    """
    if X.dim() != 2:
        raise ValueError("fir_per_time expects one trial [T, N]")
    t, n = X.shape
    n_lags, d, k = jbar.shape
    out = X.new_zeros(t, k, n)
    for lag in range(n_lags):
        if lag >= t:
            break
        v = r @ jbar[lag].to(dtype=X.dtype, device=X.device)  # [T, K]
        if lag == 0:
            out = out + v[:, :, None] * X[:, None, :]
        else:
            out[lag:] = out[lag:] + v[lag:, :, None] * X[:-lag, None, :]
    return out


def batch_data_term(X: torch.Tensor, r: torch.Tensor, jbar: torch.Tensor) -> torch.Tensor:
    """Batch F7-J data term U = sum_t e_t x_t^T with ``e = temporal_feedback(r, jbar)``."""
    Xb, rb, _ = _as_batch(X, r)
    e = fr.temporal_feedback(rb, jbar)
    return torch.einsum("btk,btn->kn", e, Xb)


def valid_sum_x2(X: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, sum_i m_t x_{t,i}^2) over a trial [T, N] or a padded batch [B, T, N]."""
    if X.dim() == 2:
        m = mask.to(dtype=X.dtype).unsqueeze(-1)
        return mask.to(dtype=X.dtype).sum(), (m * X * X).sum(0)
    m = mask.to(dtype=X.dtype).unsqueeze(-1)
    return mask.to(dtype=X.dtype).sum(), (m * X * X).sum((0, 1))


def apply_update(table, center, U, sum_x2, n, eta, rho, eps=EPS_Q):
    """T <- T + eta * (U/n) / (q+eps) - rho * (T - center), q = sum_x2 / n.

    Matches ``fj_update`` with lam = rho/eta and the default preconditioner.
    ``n`` must be the same valid-bin count that normalises U and q.
    """
    n_t = torch.as_tensor(n, dtype=U.dtype, device=U.device)
    q = sum_x2 / n_t
    g = U / n_t
    return table + float(eta) * (g / (q + eps)) - float(rho) * (table - center)


def iir_causal(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Causal filter y_t = rho * y_{t-1} + x_t, y_<0 = 0.  ``x`` is [T, N].

    Parallel doubling scan.  Exact in exact arithmetic for any finite T.
    """
    if x.dim() != 2:
        raise ValueError("iir_causal expects [T, N]")
    y = x.clone()
    step = 1
    t = x.shape[0]
    rho_f = y.new_tensor(float(rho))
    while step < t:
        y[step:] = y[step:] + (rho_f ** step) * y[:-step]
        step *= 2
    return y


def _vandermonde(rho: torch.Tensor, n_lags: int) -> torch.Tensor:
    """Phi[l, m] = rho[m]^l for l = 0..n_lags-1.  Works for negative rho."""
    cols = []
    p = torch.ones_like(rho)
    for _ in range(n_lags):
        cols.append(p)
        p = p * rho
    return torch.stack(cols, 0)


def reconstruct_exponentials(rho: torch.Tensor, jbar: torch.Tensor):
    """Least squares C for fixed rho.  Returns (rel_frobenius, C [M,D,K], residual)."""
    if rho.dim() != 1:
        raise ValueError("rho must be [M]")
    shape = jbar.shape
    j = jbar.reshape(shape[0], -1).to(dtype=rho.dtype, device=rho.device)
    phi = _vandermonde(rho, shape[0])
    gram = phi.transpose(0, 1) @ phi
    scale = gram.diagonal().mean().clamp_min(1e-18)
    eye = torch.eye(rho.shape[0], dtype=rho.dtype, device=rho.device)
    # Tiny jitter keeps the Gram invertible when two poles nearly coincide.  It is
    # far below the fit residual we report on real kernels.
    c = torch.linalg.solve(gram + (1e-14 * scale) * eye, phi.transpose(0, 1) @ j)
    resid = phi @ c - j
    rel = resid.norm() / j.norm().clamp_min(1e-18)
    return rel, c.reshape(rho.shape[0], *shape[1:]), resid


def exponential_jbar(rho: torch.Tensor, C: torch.Tensor, n_lags: int) -> torch.Tensor:
    """J[l] = sum_m rho_m^l C_m, shape [n_lags, D, K]."""
    phi = _vandermonde(rho.to(dtype=C.dtype, device=C.device), n_lags)
    flat = torch.einsum("lm,m...->l...", phi, C)
    return flat


def trace_data_term(X: torch.Tensor, r: torch.Tensor, rho: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """U [K, N] = sum_t sum_m (C_m^T r_t) s_{m,t}^T, s_m = rho_m s_m_prev + x.

    One trial (``X`` [T, N], ``r`` [T, D]).  The trace is reset (no history before t = 0).
    """
    if X.dim() != 2:
        raise ValueError("trace_data_term expects one trial; reset the trace per trial")
    k = C.shape[2]
    u = X.new_zeros(k, X.shape[1])
    rho_d = rho.to(dtype=torch.float64).detach().cpu().tolist()
    # Filter in float64 then contract in the working dtype.  The scan is exact in
    # exact arithmetic; float64 keeps long trials from drifting before the cast.
    xd = X.detach().to(dtype=torch.float64)
    rd = r.to(dtype=X.dtype)
    for m, rho_m in enumerate(rho_d):
        s = iir_causal(xd, float(rho_m)).to(dtype=X.dtype, device=X.device)
        corr = rd.transpose(0, 1) @ s                 # [D, N]
        cm = C[m].to(dtype=X.dtype, device=X.device)
        u = u + cm.transpose(0, 1) @ corr
    return u


def fit_exponential_kernel(jbar: torch.Tensor, n_modes: int, *, steps: int = 4, lr: float = 0.8,
                           restarts: int = 0, seed: int = 0, rho_max: float = 0.92,
                           grid_n: int = 36, min_sep: float = 0.07) -> dict:
    """Fit Jbar_l ≈ sum_m rho_m^l C_m on a SOURCE kernel.

    Poles lie in [-rho_max, rho_max] with rho_max < 1 and are kept at least ``min_sep``
    apart.  Clustered poles make C ill-conditioned (huge cancelling coefficients) and the
    float32 trace overflows; separation avoids that.  Greedy grid search, then a few
    coordinate sweeps.  ``steps`` is the number of sweeps.  Nothing here reads a target.
    """
    del lr, restarts, seed  # kept so older call sites stay valid; the fit is deterministic
    if n_modes < 1:
        raise ValueError("n_modes >= 1")
    if jbar.shape[0] < n_modes:
        raise ValueError(f"need at least {n_modes} lags, got {jbar.shape[0]}")
    if rho_max >= 1:
        raise ValueError("rho_max must be < 1")
    work = jbar.detach().to(dtype=torch.float64, device="cpu")
    grid = torch.linspace(-rho_max, rho_max, grid_n, dtype=torch.float64)
    grid = grid[grid.abs() > 0.04]

    def _score(rho_list):
        rel_t, c_t, _ = reconstruct_exponentials(torch.tensor(rho_list, dtype=torch.float64), work)
        return float(rel_t), c_t.detach()

    chosen: list[float] = []
    rel, C = float("inf"), None
    for _ in range(n_modes):
        best = None
        for rho in grid.tolist():
            if any(abs(rho - c) < min_sep for c in chosen):
                continue
            rel_t, c_t = _score(chosen + [float(rho)])
            if best is None or rel_t < best[0]:
                best = (rel_t, float(rho), c_t)
        if best is None:
            raise RuntimeError("pole grid cannot place another separated mode")
        chosen.append(best[1])
        rel, C = best[0], best[2]

    sweeps = max(1, min(int(steps), 6))
    for _ in range(sweeps):
        for i in range(n_modes):
            others = [chosen[j] for j in range(n_modes) if j != i]
            lo, hi = -rho_max, rho_max
            for v in others:
                if v < chosen[i]:
                    lo = max(lo, v + min_sep)
                else:
                    hi = min(hi, v - min_sep)
            if hi - lo < 1e-4:
                continue
            def _search(samples):
                best_s = None
                for rho in samples:
                    trial = list(chosen)
                    trial[i] = float(rho)
                    rel_t, c_t = _score(trial)
                    if best_s is None or rel_t < best_s[0]:
                        best_s = (rel_t, float(rho), c_t)
                return best_s

            best = _search(torch.linspace(lo, hi, 12).tolist())
            # Local refinement: the residual is sensitive to a few thousandths in rho.
            a, b = max(lo, best[1] - 0.03), min(hi, best[1] + 0.03)
            best = _search(torch.linspace(a, b, 25).tolist())
            chosen[i], rel, C = best[1], best[0], best[2]
    if not math.isfinite(rel):
        raise RuntimeError("exponential fit did not produce a finite residual")
    rho_abs = max(abs(v) for v in chosen)
    if rho_abs >= 1.0:
        raise RuntimeError(f"unstable pole {rho_abs}")
    return {"rel_frobenius": float(rel), "rho": torch.tensor(chosen, dtype=torch.float64),
            "C": C.detach().to(dtype=jbar.dtype), "n_modes": int(n_modes),
            "n_lags": int(jbar.shape[0]), "init": "separated_greedy",
            "rho_abs_max": float(rho_abs)}


def _residual(yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (y - yhat) * mask.to(dtype=y.dtype).unsqueeze(-1)


def stream_refine(base, trials, table0, eta, rho, *, kind: str, jbar=None, poles=None, coeff=None,
                  center=None, eps=EPS_Q, collect_readout: bool = False, on_commit=None):
    """One live pass over calibration trials.  Commit the shrinkage update at each trial boundary.

    ``trials`` is a list of (X [T,N], Y [T,D], M [T]) in recorded order.  Each trial is
    forwarded from the zero state (a separate ``run_trunk`` call).  Readout statistics,
    if requested, are taken under the table active during the trial, before the commit.

    Returns dict(table, n_valid, n_commits, readout or None).
    """
    if kind not in ("fir", "trace"):
        raise ValueError(kind)
    table = table0
    center_t = table0 if center is None else center
    n_valid = table.new_zeros(())
    n_commits = 0
    readout = None
    # Target-side guard: this pass must not build a graph through the trunk.
    with torch.no_grad():
        for X, Y, M in trials:
            yhat, h = fr.run_trunk(base, X.unsqueeze(0), table, want_h=True)
            yhat, h = yhat[0], h[0]
            r = _residual(yhat, Y.to(dtype=yhat.dtype), M)
            if kind == "fir":
                U = fir_data_term(X, r, jbar)
            else:
                U = trace_data_term(X, r, poles, coeff)
            n, sum_x2 = valid_sum_x2(X, M)
            if float(n) > 0:
                table = apply_update(table, center_t, U, sum_x2, n, eta, rho, eps)
                n_commits += 1
                n_valid = n_valid + n
            if on_commit is not None:
                on_commit(n_commits, table)
            if collect_readout:
                mm = M.to(dtype=h.dtype).unsqueeze(-1)
                hb = torch.cat([h, torch.ones_like(h[:, :1])], -1) * mm
                r0 = r.to(dtype=torch.float64)
                hb64 = hb.double()
                G = hb64.transpose(0, 1) @ hb64
                R = r0.transpose(0, 1) @ hb64
                block = {"G": G, "R": R, "n": float(n)}
                readout = block if readout is None else fr.add_readout_stats(readout, block)
    return {"table": table, "n_valid": float(n_valid), "n_commits": n_commits, "readout": readout}


def frozen_fir_update(base, trials, table0, jbar, eta, rho, center=None, eps=EPS_Q):
    """One F7-J step on a fixed trajectory: sum the reordered terms, then a single commit.

    Algebraically the same update as one ``fj_refine`` step when ``trials`` are the
    unpadded rows of the same ``pad_trials`` batch (trailing pad has zero mask).
    """
    center_t = table0 if center is None else center
    U = None
    sum_x2 = None
    n = table0.new_zeros(())
    with torch.no_grad():
        for X, Y, M in trials:
            yhat = fr.run_trunk(base, X.unsqueeze(0), table0)
            r = _residual(yhat[0], Y.to(dtype=yhat.dtype), M)
            term = fir_data_term(X, r, jbar)
            n_i, x2 = valid_sum_x2(X, M)
            U = term if U is None else U + term
            sum_x2 = x2 if sum_x2 is None else sum_x2 + x2
            n = n + n_i
        table = apply_update(table0, center_t, U, sum_x2, n, eta, rho, eps)
    return table


def fir_bin_macs(n_lags: int, d: int, k: int, n: int) -> dict:
    """Per-bin MACs of the reordered FIR accumulator (not the trunk).

    Per lag: J_l^T e is weight x activation (Jbar is a fixed ROM kernel);
    the outer product with the lagged input is activation x activation.
    """
    return {"weight_x_act": int(n_lags * d * k), "act_x_act": int(n_lags * k * n),
            "total": int(n_lags * d * k + n_lags * k * n)}


def trace_bin_macs(n_modes: int, d: int, k: int, n: int) -> dict:
    """Per-bin MACs of the eligibility-trace accumulator (not the trunk).

    rho * s is a fixed coefficient times an activation (weight x activation, one MAC
    per trace element, matching the design-note count).  C_m^T e is weight x activation.
    The outer product (C_m^T e) s^T is activation x activation.
    """
    w = n_modes * d * k + n_modes * n
    a = n_modes * k * n
    return {"weight_x_act": int(w), "act_x_act": int(a), "total": int(w + a)}


def stream_sram_bytes(*, n, k, n_lags, n_modes, acc_bits=32, table_bits=32, include_trace=False,
                      include_fir=True, staging_bins=0, bx=3, by=16, d=2, readout_h=None) -> dict:
    """Algorithmic SRAM of one streaming pass.  Every buffer this method keeps is listed.

    The ring / traces are the recurrent state.  The K x N accumulator is the running
    update.  ``staging_one_trial`` is the current calibration trial held while the
    frozen-within-trial forward runs (not a multi-trial replay buffer).  Trunk state
    is shared with inference and is not included here.
    """
    def b(elems, bits):
        return elems * bits / 8.0

    sram = {
        "table T": b(k * n, table_bits),
        "shrink centre T0": b(k * n, table_bits),
        "U accumulator KxN": b(k * n, acc_bits),
        "q accumulator N": b(n, acc_bits),
        "valid-bin counter": 8.0,
        "current residual D": b(d, acc_bits),
    }
    if include_fir:
        sram["FIR ring LxN"] = b(n_lags * n, acc_bits)
    if include_trace:
        sram["traces MxN"] = b(n_modes * n, acc_bits)
    if staging_bins:
        sram["one-trial staging (inputs+labels+mask)"] = staging_bins * (n * bx + d * by + 1) / 8.0
    if readout_h is not None:
        ha = readout_h + 1
        sram["readout Gram + cross (fp32)"] = b(ha * (ha + 1) // 2 + d * ha, acc_bits)
        sram["readout V Dx(H+1)"] = b(d * ha, acc_bits)
    sram["total"] = float(sum(sram.values()))
    return sram


def r2_replay_reference(case: dict, b: int, *, steps: int, lags: int, replay_bins: int,
                        staging_bins: int) -> dict:
    """R2 cost from the existing P3 model (replay buffer, delay line, S forwards)."""
    from ssm_decode.refine_cost import refinement_cost
    return refinement_cost(case, b, "r2", steps=steps, lags=lags, replay_bins=replay_bins,
                           staging_bins=staging_bins)
