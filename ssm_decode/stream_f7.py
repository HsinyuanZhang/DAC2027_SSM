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
        "NOT causal across calibration trials. One forward per trial from the zero state, "
        "and the per-bin FIR term uses only r_t and x_<=t. The normalisation and the P0 "
        "table are computed from ALL B trials before the first forward, so trial k depends "
        "on future trials. Kept only as a labeled non-causal baseline."
    ),
    "s1_causal": (
        "Trial k is forwarded with normalisation and the running table known from trials "
        "< k only. Trial 0 matches pc.stream_stats: the prior table, and normalisation from "
        "trial 0 alone (one-trial buffer). After the trial, P0 is recomputed on the prefix "
        "including it and the F7 step, rescaled by std_new/std_old, is applied on top of "
        "that P0 with shrinkage toward it. It does not read a future trial. "
        "Later trials do not change earlier commits."
    ),
    "trace": (
        "Same trial schedule as the FIR variant it is attached to. Deployed traces are "
        "truncated at L (s_t - rho^L s_{t-L}) so they equal the fitted L-tap FIR. The "
        "infinite tail is not part of the update; its norm and DC gain are reported. "
        "No future residual is read."
    ),
    "s1_readout": (
        "The old name s1_r1s collides with P3 R1-stream and was non-causal. The causal "
        "readout s1c_ridge accumulates the Gram on the s1_causal forward and solves once "
        "after trial B. It does not rerun the final table and does not read a future trial."
    ),
}
CAUSALITY["s1_r1s"] = CAUSALITY["s1_readout"]


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


def feedback_step(U, sum_x2, n, eps=EPS_Q):
    """Preconditioned mean gradient G/(q+eps) in the normalisation of the forward."""
    n_t = torch.as_tensor(n, dtype=U.dtype, device=U.device)
    q = sum_x2 / n_t
    return (U / n_t) / (q + eps)


def rescale_columns(mat: torch.Tensor, std_old: torch.Tensor, std_new: torch.Tensor) -> torch.Tensor:
    """Map a K x N table between per-unit std normalisations.

    z = (x - mean) / std, latent += z @ T.T.  Column i of T multiplies unit i.
    T_new[:, i] = T_old[:, i] * std_new[i] / std_old[i] keeps the gain on raw counts
    (x_i / std_old) * T_old = (x_i / std_new) * T_new.  A mean change is not a K x N
    update; the recomputed P0 carries it.
    """
    scale = (std_new.to(dtype=mat.dtype, device=mat.device) / std_old.to(dtype=mat.dtype, device=mat.device).clamp_min(1e-8))
    return mat * scale.unsqueeze(0)


def commit_on_p0(table_used, step, p0_new, std_used, std_new, eta, rho):
    """T = rescale(T_used) + eta * rescale(step) - rho * (rescale(T_used) - P0_new).

    ``step`` is ``feedback_step`` in the coordinates of the forward (``std_used``).
    ``p0_new`` and ``std_new`` are the prefix P0 AFTER the trial just consumed.
    If std and P0 do not change and ``table_used`` is that P0, this equals ``apply_update``.
    """
    table_r = rescale_columns(table_used, std_used, std_new)
    step_r = rescale_columns(step, std_used, std_new)
    p0 = p0_new.to(dtype=table_r.dtype, device=table_r.device)
    return table_r + float(eta) * step_r - float(rho) * (table_r - p0)


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


def truncate_iir(state: torch.Tensor, rho: float, n_lags: int) -> torch.Tensor:
    """sum_{l=0}^{n_lags-1} rho^l x_{t-l} from the infinite state s_t = sum_{l=0}^{t} rho^l x_{t-l}.

    s_t - rho^{n_lags} s_{t-n_lags} drops lags >= n_lags.  For t < n_lags the infinite
    state is already shorter than n_lags.
    """
    if n_lags >= state.shape[0]:
        return state
    out = state.clone()
    scale = float(rho) ** int(n_lags)
    out[n_lags:] = state[n_lags:] - scale * state[:-n_lags]
    return out


def trace_data_term(X: torch.Tensor, r: torch.Tensor, rho: torch.Tensor, C: torch.Tensor,
                    n_lags: int | None = None) -> torch.Tensor:
    """U [K, N] = sum_t sum_m (C_m^T r_t) s_{m,t}^T.

    One trial (``X`` [T, N], ``r`` [T, D]).  The trace is reset at t = 0.
    ``n_lags`` truncates each trace at that FIR length (the deployed method).  ``None``
    keeps the infinite tail (diagnostic only; it is not what S1 deploys).
    """
    if X.dim() != 2:
        raise ValueError("trace_data_term expects one trial; reset the trace per trial")
    k = C.shape[2]
    # Contract in float64.  A large cancelling C makes the float32 tail subtraction unstable.
    u = torch.zeros(k, X.shape[1], dtype=torch.float64, device=X.device)
    rho_d = rho.to(dtype=torch.float64).detach().cpu().tolist()
    xd = X.detach().to(dtype=torch.float64)
    rd = r.detach().to(dtype=torch.float64)
    for m, rho_m in enumerate(rho_d):
        s = iir_causal(xd, float(rho_m))
        if n_lags is not None:
            s = truncate_iir(s, float(rho_m), int(n_lags))
        corr = rd.transpose(0, 1) @ s
        cm = C[m].detach().to(dtype=torch.float64, device=X.device)
        u = u + cm.transpose(0, 1) @ corr
    return u.to(dtype=X.dtype)


def kernel_tail_metrics(jbar: torch.Tensor, rho: torch.Tensor, C: torch.Tensor, horizon: int = 400) -> dict:
    """Head (lags 0..L-1) versus the infinite exponential tail the untruncated recursion would add.

    DC gain is ||sum_l J_l||.  The infinite DC is ||sum_m C_m / (1 - rho_m)||.
    """
    rho = rho.detach().to(dtype=torch.float64).flatten()
    C = C.detach().to(dtype=torch.float64)
    jbar = jbar.detach().to(dtype=torch.float64)
    n_lags = int(jbar.shape[0])
    p = torch.ones_like(rho)
    heads = []
    for _ in range(n_lags):
        heads.append(torch.einsum("m,mdk->dk", p, C))
        p = p * rho
    head = torch.stack(heads)
    acc = torch.zeros((), dtype=torch.float64)
    for _ in range(horizon):
        block = torch.einsum("m,mdk->dk", p, C)
        acc = acc + block.pow(2).sum()
        p = p * rho
        if float(p.abs().max()) < 1e-18:
            break
    dc_inf = torch.stack([C[m] / (1.0 - rho[m]) for m in range(rho.shape[0])]).sum(0)
    return {"head_frobenius": float(head.norm()),
            "tail_frobenius": float(acc.sqrt()),
            "dc_gain_head": float(head.sum(0).norm()),
            "dc_gain_true": float(jbar.sum(0).norm()),
            "dc_gain_infinite": float(dc_inf.norm()),
            "max_abs_C": float(C.abs().max())}


def fit_exponential_kernel(jbar: torch.Tensor, n_modes: int, *, steps: int = 4, lr: float = 0.8,
                           restarts: int = 2, seed: int = 0, rho_max: float = 0.92,
                           grid_n: int = 36, min_sep: float = 0.07) -> dict:
    """Fit Jbar_l ≈ sum_m rho_m^l C_m on a SOURCE kernel.

    Poles lie in [-rho_max, rho_max] with rho_max < 1 and are kept at least ``min_sep``
    apart.  ``steps`` is the number of coordinate sweeps.  ``restarts`` extra separated
    inits are drawn from ``seed`` and the best head residual is kept (``lr`` is unused;
    this is not a gradient step).  Nothing here reads a target.
    """
    del lr
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

    def _polish(poles):
        chosen_l = [float(v) for v in poles]
        rel_l, c_l = _score(chosen_l)
        for _s in range(sweeps):
            for i in range(n_modes):
                others = [chosen_l[j] for j in range(n_modes) if j != i]
                lo, hi = -rho_max, rho_max
                for v in others:
                    if v < chosen_l[i]:
                        lo = max(lo, v + min_sep)
                    else:
                        hi = min(hi, v - min_sep)
                if hi - lo < 1e-4:
                    continue

                def _search(samples, i=i, chosen_l=chosen_l):
                    best_s = None
                    for rho in samples:
                        trial = list(chosen_l)
                        trial[i] = float(rho)
                        rel_t, c_t = _score(trial)
                        if best_s is None or rel_t < best_s[0]:
                            best_s = (rel_t, float(rho), c_t)
                    return best_s

                found = _search(torch.linspace(lo, hi, 12).tolist())
                a, b = max(lo, found[1] - 0.03), min(hi, found[1] + 0.03)
                found = _search(torch.linspace(a, b, 25).tolist())
                chosen_l[i], rel_l, c_l = found[1], found[0], found[2]
        return rel_l, chosen_l, c_l

    rel, chosen, C = _polish(chosen)
    tag = "separated_greedy"
    rng = torch.Generator().manual_seed(int(seed))
    for _r in range(int(restarts)):
        poles = []
        guard = 0
        while len(poles) < n_modes and guard < 10000:
            guard += 1
            v = float((torch.rand((), generator=rng).item() * 2.0 - 1.0) * rho_max)
            if abs(v) <= 0.04:
                continue
            if all(abs(v - p) >= min_sep for p in poles):
                poles.append(v)
        if len(poles) < n_modes:
            continue
        rel_r, poles_r, c_r = _polish(poles)
        if rel_r < rel:
            rel, chosen, C, tag = rel_r, poles_r, c_r, "restart"
    if not math.isfinite(rel):
        raise RuntimeError("exponential fit did not produce a finite residual")
    rho_abs = max(abs(v) for v in chosen)
    if rho_abs >= 1.0:
        raise RuntimeError(f"unstable pole {rho_abs}")
    tail = kernel_tail_metrics(jbar.detach().to(dtype=torch.float64), torch.tensor(chosen), C.detach().double())
    return {"rel_frobenius": float(rel), "rho": torch.tensor(chosen, dtype=torch.float64),
            "C": C.detach().to(dtype=jbar.dtype), "n_modes": int(n_modes),
            "n_lags": int(jbar.shape[0]), "init": tag, "seed": int(seed), "restarts": int(restarts),
            "rho_abs_max": float(rho_abs), **tail}


def _residual(yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (y - yhat) * mask.to(dtype=y.dtype).unsqueeze(-1)


def causal_feedback_loop(base, raw_trials, table_start, eta, rho, prepare, *, kind, jbar=None,
                         poles=None, coeff=None, n_lags=None, collect_readout=False, eps=EPS_Q):
    """One causal calibration pass.

    ``prepare(k)`` returns ``(mean, std, p0)`` for the forward of trial k.  For k = 0, ``p0``
    may be None and ``table_start`` (the prior) is used; ``mean, std`` are trial 0's own
    stats (the ``stream_stats`` one-trial buffer).  For k > 0, ``mean, std, p0`` come from
    trials < k and the running table is already in those coordinates.

    After the forward, ``prepare_after(k)`` is not used: ``prepare`` is called again as
    ``prepare(k, closed=True)`` for the prefix that includes trial k.  See the callable
    contract below.

    ``prepare(k, closed)``:
      closed=False -> stats allowed before the trial (trial 0: own norm, table is prior).
      closed=True  -> P0 and std of the prefix that includes trial k.
    Returns (mean [N], std [N], p0 [K,N] or None).
    """
    table = table_start
    readout = None
    last_mean = last_std = None
    with torch.no_grad():
        for k, (x_raw, y, mask) in enumerate(raw_trials):
            mean, std, _p0_before = prepare(k, False)
            xz = (x_raw - mean.to(dtype=x_raw.dtype)) / std.to(dtype=x_raw.dtype).clamp_min(1e-8)
            yhat, h = fr.run_trunk(base, xz.unsqueeze(0), table, want_h=True)
            yhat, h = yhat[0], h[0]
            r = _residual(yhat, y.to(dtype=yhat.dtype), mask)
            if kind == "fir":
                u = fir_data_term(xz, r, jbar)
            else:
                u = trace_data_term(xz, r, poles, coeff, n_lags=n_lags)
            n, sum_x2 = valid_sum_x2(xz, mask)
            step = feedback_step(u, sum_x2, n, eps) if float(n) > 0 else torch.zeros_like(table)
            mean_a, std_a, p0_a = prepare(k, True)
            if float(n) > 0:
                table = commit_on_p0(table, step, p0_a, std, std_a, eta, rho)
            last_mean, last_std = mean_a, std_a
            if collect_readout:
                mm = mask.to(dtype=h.dtype).unsqueeze(-1)
                hb = torch.cat([h, torch.ones_like(h[:, :1])], -1) * mm
                block = {"G": (hb.double().T @ hb.double()),
                         "R": (r.double().T @ hb.double()), "n": float(n)}
                readout = block if readout is None else fr.add_readout_stats(readout, block)
    return {"table": table, "mean": last_mean, "std": last_std, "readout": readout}


def stream_refine(base, trials, table0, eta, rho, *, kind: str, jbar=None, poles=None, coeff=None,
                  center=None, eps=EPS_Q, collect_readout: bool = False, on_commit=None,
                  n_lags: int | None = None):
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
                # Default for a streaming trace is the FIR length, not the infinite tail.
                lags = n_lags if n_lags is not None else (None if jbar is None else int(jbar.shape[0]))
                U = trace_data_term(X, r, poles, coeff, n_lags=lags)
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


def probe_data_term_cosine(jbar, rho, C, *, seed: int = 0, length: int = 48, n: int = 12) -> dict:
    """Cosine of the true-kernel FIR data term with the truncated and the infinite trace.

    Deterministic probe so the number in the README is an artifact, not a recollection.
    """
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(length, n, generator=g, dtype=torch.float64)
    r = torch.randn(length, int(jbar.shape[1]), generator=g, dtype=torch.float64)
    jb = jbar.detach().to(dtype=torch.float64)
    rho_t = torch.as_tensor(rho, dtype=torch.float64)
    c = C.detach().to(dtype=torch.float64)
    u_fir = fir_data_term(x, r, jb)

    def _cos(other):
        den = u_fir.norm() * other.norm()
        return float((u_fir.flatten() @ other.flatten()) / den) if float(den) > 0 else float("nan")

    return {"cosine_truncated_trace": _cos(trace_data_term(x, r, rho_t, c, n_lags=int(jb.shape[0]))),
            "cosine_infinite_trace": _cos(trace_data_term(x, r, rho_t, c, n_lags=None)),
            "probe_seed": int(seed), "probe_length": int(length), "probe_n": int(n)}


def p0_prefix_stat_bytes(n: int, acc_bits: int = 32) -> float:
    """Bytes of the additive PopStats prefix (the P0 running sums, not raw trials).

    Shapes follow ``population_calib`` (AC_LAGS=3, CROSS_LAGS=5, D=2, DIR_BINS=8).
    """
    ac, cross, d, dirs = 3, 5, 2, 8
    elems = (1 + n + n * n + 3 * ac * n + ac + cross * d + cross * d * n + cross
             + n * dirs + dirs + 3 + 9 + 3 * n + n + 2 * d * n + d)
    return elems * acc_bits / 8.0


def like_for_like_memory(*, n, k, d, n_lags, n_modes, replay_bins, staging_bins, bx=3, by=16,
                         state_elems=16384, state_bits=16, acc_bits=32, table_bits=32) -> dict:
    """SRAM/ROM of R2, non-causal S1 and causal S1 under the same inclusion rules.

    Trunk state is reported both excluded and included (16-bit, ``state_elems`` * 2 bytes
    = 32768 for the K=10 trunk).  The FIR ring is reported at 3-bit raw counts and at
    32-bit normalised activations.  Non-causal S1 is charged the B-trial raw buffer
    because its norm and P0 are computed from all B trials before the pass.  Causal S1
    keeps one trial plus the P0 prefix sums.
    """
    def b(elems, bits):
        return float(elems * bits / 8.0)

    trunk = b(state_elems, state_bits)
    norm16 = b(2 * n, 16)
    norm32 = b(2 * n, 32)
    rom_j = float(n_lags * d * k)          # int8 Jbar
    rom_c = float(n_modes * d * k + n_modes * 4)  # int8 C plus fp32 poles
    table = b(k * n, table_bits)
    acc = b(k * n, acc_bits) + b(n, acc_bits) + 8.0 + b(d, acc_bits)
    raw = float(replay_bins * (n * bx + d * by + 1) / 8.0)
    one = float(staging_bins * (n * bx + d * by + 1) / 8.0)
    ring3 = float(n_lags * n * bx / 8.0)
    ring32 = b(n_lags * n, 32)
    trace = b(n_modes * n, acc_bits)
    p0_stats = p0_prefix_stat_bytes(n, acc_bits)
    # R2 from the same pieces the P3 model lists, plus ROM, with and without trunk state.
    # P0 prefix sums are required before either method's first update (R2 starts from P0;
    # non-causal S1 does too).  Count them on both sides.
    r2_core = table * 2 + acc + raw + b((n_lags) * d, 16) + norm32 + rom_j + p0_stats
    s1_nc = table * 2 + acc + raw + ring32 + norm32 + rom_j + p0_stats
    s1_c = table * 2 + acc + one + ring32 + norm32 + rom_j + p0_stats
    s1_c_ring3 = s1_c - ring32 + ring3
    return {
        "trunk_state_bytes": trunk,
        "norm_mean_std_16bit": norm16,
        "norm_mean_std_32bit": norm32,
        "jbar_rom_int8": rom_j,
        "trace_rom_bytes": rom_c,
        "ring_3bit": ring3,
        "ring_32bit": ring32,
        "trace_state_32bit": trace,
        "B_trial_raw_buffer": raw,
        "one_trial_staging_3bit": one,
        "p0_prefix_stats": p0_stats,
        "r2_without_trunk_state": r2_core,
        "r2_with_trunk_state": r2_core + trunk,
        "s1_noncausal_without_trunk_state": s1_nc,
        "s1_noncausal_with_trunk_state": s1_nc + trunk,
        "s1_causal_32bit_ring_without_trunk_state": s1_c,
        "s1_causal_32bit_ring_with_trunk_state": s1_c + trunk,
        "s1_causal_3bit_ring_without_trunk_state": s1_c_ring3,
        "s1_causal_3bit_ring_with_trunk_state": s1_c_ring3 + trunk,
        "s1_causal_trace_without_trunk_state": s1_c - ring32 + trace + rom_c - rom_j,
    }


def r2_replay_reference(case: dict, b: int, *, steps: int, lags: int, replay_bins: int,
                        staging_bins: int) -> dict:
    """R2 cost from the existing P3 model (replay buffer, delay line, S forwards)."""
    from ssm_decode.refine_cost import refinement_cost
    return refinement_cost(case, b, "r2", steps=steps, lags=lags, replay_bins=replay_bins,
                           staging_bins=staging_bins)
