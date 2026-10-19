"""P0-Solve and the W_alpha shrinkage baseline.

Target side of every candidate is streaming accumulation plus one ``p x p``
linear solve. There is no backprop and no trunk Jacobian.

Coordinates
-----------
``W0`` (the P0 table, or the electrode prior) is applied as

    u = W0 @ ((r - mu) / sigma)

with ``(mu, sigma)`` from the B calibration trials (``population_calib.norm_stats``).
Corrected ridge-CCA returns an affine map ``(T, c, mean_c, std_c)`` on its own
z-scored raw counts. ``map_cca_into_w0_coordinates`` rewrites that map as
``(W_cca, b_cca)`` in the same normalized coordinates, so

    u = W_cca @ ((r - mu) / sigma) + b_cca

equals the original CCA map for every ``r`` when the CCA input transform is
linear (raw counts). A square-root count transform is not a linear map in
these coordinates and is not used for shrinkage.

Behavior basis
--------------
``phi`` reuses ridge-CCA's 8 reach-direction labels (``published_baselines.condition_label``
on the trial target direction). ``p = 8`` is that one-hot. ``p = 16`` appends
the same one-hot times a within-trial phase in ``(-1, 1)`` — the lowest-order
time expansion of CCA's phase-averaged condition trajectories. Unlabeled bins
contribute zeros.

The solve
---------
    W* = W0 + (F G - W0 H) (H^T D H + lambda G_eps)^{-1} H^T D

with ``G_eps = G + eps I``. Implemented as a linear solve, not an inverse.
``lambda -> infinity`` returns ``W0``. ``D = I`` unless a positive diagonal is
passed (source-fixed group scales; not selected in the reported rows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the experiment env always has torch
    torch = None

DIR_BINS = 8
P_GRID = (8, 16)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Source-only grids. Large lambda is the explicit "keep W0" corner.
LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4)
EPS_GRID = (1e-8, 1e-4, 1e-2, 1.0)
F_RIDGE_GRID = (1e-4, 1e-2, 1.0)
STD_FLOOR = 1e-6


@dataclass
class RawMoments:
    """Raw-coordinate accumulators. Phi does not depend on input normalization.

    ``sum_rphi`` is ``sum_t r_t phi_t^T`` ``[N, p]``. ``sum_phiphi`` is
    ``sum_t phi_t phi_t^T`` ``[p, p]``. ``n`` counts bins that entered ``sum_r``
    (the normalization support), which may exceed the bins with nonzero phi.
    """

    n: float
    sum_r: np.ndarray
    sum_r2: np.ndarray
    sum_phi: np.ndarray
    sum_rphi: np.ndarray
    sum_phiphi: np.ndarray

    @property
    def p(self) -> int:
        return int(self.sum_phi.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.sum_r.shape[0])


def phi_features(cond: np.ndarray, phase: np.ndarray, p: int) -> np.ndarray:
    """Reach-direction basis. ``cond`` is in ``0..7`` or negative (unlabeled).

    ``phase`` is the within-trial coordinate in ``(-1, 1)``; ignored for ``p=8``.
    """
    if p not in P_GRID:
        raise ValueError(f"p must be one of {P_GRID}, got {p}")
    cond = np.asarray(cond)
    n = int(cond.shape[0])
    oh = np.zeros((n, DIR_BINS), dtype=np.float64)
    good = (cond >= 0) & (cond < DIR_BINS)
    if np.any(good):
        oh[np.flatnonzero(good), cond[good].astype(np.int64)] = 1.0
    if p == DIR_BINS:
        return oh
    ph = np.asarray(phase, dtype=np.float64)
    ph = np.where(np.isfinite(ph), ph, 0.0)
    return np.concatenate([oh, oh * ph[:, None]], axis=1)


def empty_moments(n_units: int, p: int) -> RawMoments:
    z = lambda *s: np.zeros(s, dtype=np.float64)
    return RawMoments(0.0, z(n_units), z(n_units), z(p), z(n_units, p), z(p, p))


def accumulate_raw(r: np.ndarray, phi: np.ndarray, mask: np.ndarray) -> RawMoments:
    """Add the masked bins of one chunk. ``r`` ``[T, N]``, ``phi`` ``[T, p]``."""
    r = np.asarray(r, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if r.ndim != 2 or phi.ndim != 2 or r.shape[0] != phi.shape[0] or m.shape[0] != r.shape[0]:
        raise ValueError(f"shape mismatch r={r.shape} phi={phi.shape} mask={m.shape}")
    rr, pp = r[m], phi[m]
    if len(rr) == 0:
        return empty_moments(r.shape[1], phi.shape[1])
    return RawMoments(
        n=float(len(rr)),
        sum_r=rr.sum(0),
        sum_r2=np.square(rr).sum(0),
        sum_phi=pp.sum(0),
        sum_rphi=rr.T @ pp,
        sum_phiphi=pp.T @ pp,
    )


def add_moments(a: RawMoments, b: RawMoments) -> RawMoments:
    if a.p != b.p or a.n_units != b.n_units:
        raise ValueError("moment shape mismatch")
    return RawMoments(
        n=a.n + b.n,
        sum_r=a.sum_r + b.sum_r,
        sum_r2=a.sum_r2 + b.sum_r2,
        sum_phi=a.sum_phi + b.sum_phi,
        sum_rphi=a.sum_rphi + b.sum_rphi,
        sum_phiphi=a.sum_phiphi + b.sum_phiphi,
    )


def mean_std_from_raw(raw: RawMoments, std_floor: float = STD_FLOOR):
    """Population mean and std (ddof 0), with the same floor as ``norm_stats``."""
    n = max(float(raw.n), 1.0)
    mean = raw.sum_r / n
    var = np.maximum(raw.sum_r2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < std_floor, 1.0, std)
    return mean, std


def moments_to_normalized(raw: RawMoments, mean: np.ndarray, std: np.ndarray):
    """Convert raw moments to ``H = sum x phi^T``, ``G = sum phi phi^T``.

    ``x = (r - mean) / std`` (elementwise). ``G`` does not depend on ``(mean, std)``.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (raw.n_units,) or std.shape != (raw.n_units,):
        raise ValueError("mean/std shape does not match the accumulator")
    # sum (r - mean) phi^T = sum_rphi - mean[:, None] * sum_phi[None, :]
    centered = raw.sum_rphi - mean[:, None] * raw.sum_phi[None, :]
    h = centered / std[:, None]
    g = raw.sum_phiphi.copy()
    return h, g


def batch_hg(x_norm: np.ndarray, phi: np.ndarray, mask: np.ndarray):
    """Reference ``H, G`` from already-normalized inputs (for the equality test)."""
    x = np.asarray(x_norm, dtype=np.float64)[mask]
    p = np.asarray(phi, dtype=np.float64)[mask]
    if len(x) == 0:
        n, pp = x_norm.shape[1], phi.shape[1]
        return np.zeros((n, pp)), np.zeros((pp, pp))
    return x.T @ p, p.T @ p


def fit_anchor(u: np.ndarray, phi: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge map ``F`` ``[K, p]`` with ``u ≈ F phi`` (rows of ``u`` are latents).

    Solves ``(Phi^T Phi + ridge I) F^T = Phi^T U``. ``ridge`` is absolute
    (not scaled); selection of ``ridge`` is source-only.
    """
    u = np.asarray(u, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if u.shape[0] != phi.shape[0]:
        raise ValueError("u and phi must share the bin axis")
    p = phi.shape[1]
    gram = phi.T @ phi + float(ridge) * np.eye(p)
    rhs = phi.T @ u
    ft = np.linalg.solve(gram, rhs)          # [p, K]
    return ft.T


def _as_d(d, n: int) -> np.ndarray:
    if d is None:
        return np.ones(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.shape != (n,) or np.any(d <= 0) or not np.isfinite(d).all():
        raise ValueError("D must be a positive length-N diagonal")
    return d


def solve_p0(w0: np.ndarray, f: np.ndarray, h: np.ndarray, g: np.ndarray,
             lam: float, eps: float, d=None) -> np.ndarray:
    """Closed form ``W*``. ``w0`` ``[K, N]``, ``f`` ``[K, p]``, ``h`` ``[N, p]``, ``g`` ``[p, p]``."""
    w0 = np.asarray(w0, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    k, n = w0.shape
    p = g.shape[0]
    if f.shape != (k, p) or h.shape != (n, p) or g.shape != (p, p):
        raise ValueError(f"shape mismatch W0={w0.shape} F={f.shape} H={h.shape} G={g.shape}")
    dv = _as_d(d, n)
    g_eps = 0.5 * (g + g.T) + float(eps) * np.eye(p)
    # A = H^T D H + lambda G_eps,  rhs = H^T D,  both built without an N x N matrix.
    hd = h * dv[:, None]                            # D H, [N, p]
    a = h.T @ hd + float(lam) * g_eps               # [p, p]
    a = 0.5 * (a + a.T)
    rhs = hd.T                                      # [p, N]
    try:
        z = np.linalg.solve(a, rhs)                 # A Z = H^T D
    except np.linalg.LinAlgError:
        z, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    resid = f @ g - w0 @ h                         # FG - W0 H, [K, p]
    return w0 + resid @ z


def direct_ls_solve(w0, f, h, g, lam: float, eps: float, d=None) -> np.ndarray:
    """Row-wise least squares of the same proxy objective, for the equality test.

    ``min_W ||(W H - F G) G_eps^{-1/2}||_F^2 + lam ||(W - W0) D^{-1/2}||_F^2``.
    """
    w0 = np.asarray(w0, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    g = 0.5 * (np.asarray(g, dtype=np.float64) + np.asarray(g, dtype=np.float64).T)
    k, n = w0.shape
    p = g.shape[0]
    dv = _as_d(d, n)
    g_eps = g + float(eps) * np.eye(p)
    w, v = np.linalg.eigh(0.5 * (g_eps + g_eps.T))
    w = np.clip(w, 1e-18, None)
    s = (v * (1.0 / np.sqrt(w))) @ v.T             # G_eps^{-1/2}
    design_top = (h @ s).T                          # [p, N]
    scale = np.sqrt(float(lam)) / np.sqrt(dv)      # D^{-1/2} * sqrt(lam)
    design = np.vstack([design_top, np.diag(scale)])
    tgt = (g @ s)                                   # [p, p], row target is f_row @ tgt
    out = np.zeros_like(w0)
    for i in range(k):
        rhs = np.concatenate([tgt.T @ f[i], scale * w0[i]])
        sol, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        out[i] = sol
    return out


def proxy_objective(w, w0, f, h, g, lam: float, eps: float, d=None) -> float:
    """Value of the proxy objective (diagnostic / tests, not used on target)."""
    w = np.asarray(w, dtype=np.float64)
    g = 0.5 * (np.asarray(g, dtype=np.float64) + np.asarray(g, dtype=np.float64).T)
    p = g.shape[0]
    dv = _as_d(d, w.shape[1])
    g_eps = g + float(eps) * np.eye(p)
    ev, vec = np.linalg.eigh(0.5 * (g_eps + g_eps.T))
    ev = np.clip(ev, 1e-18, None)
    s = (vec * (1.0 / np.sqrt(ev))) @ vec.T
    resid = (w @ h - f @ g) @ s
    delta = (w - w0) / np.sqrt(dv)
    return float(np.sum(resid ** 2) + float(lam) * np.sum(delta ** 2))


def shrink_tables(w0: np.ndarray, w_cca: np.ndarray, b_cca: np.ndarray, alpha: float):
    """Affine interpolation. ``W0`` is bias-free, so ``b_alpha = alpha * b_cca``."""
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    w0 = np.asarray(w0, dtype=np.float64)
    w_cca = np.asarray(w_cca, dtype=np.float64)
    b_cca = np.asarray(b_cca, dtype=np.float64).reshape(-1)
    if w0.shape != w_cca.shape or b_cca.shape != (w0.shape[0],):
        raise ValueError("shrinkage shape mismatch")
    w = w0 + a * (w_cca - w0)
    b = a * b_cca
    return w, b


def map_cca_into_w0_coordinates(t_cca, c_cca, mean_c, std_c, mean_p, std_p):
    """Rewrite raw-count CCA ``u = T ((r - mean_c) / std_c) + c`` in P0's norm.

    Returns ``(W, b)`` such that ``u = W ((r - mean_p) / std_p) + b`` for every ``r``.
    """
    t_cca = np.asarray(t_cca, dtype=np.float64)
    c_cca = np.asarray(c_cca, dtype=np.float64).reshape(-1)
    mean_c = np.asarray(mean_c, dtype=np.float64)
    std_c = np.asarray(std_c, dtype=np.float64)
    mean_p = np.asarray(mean_p, dtype=np.float64)
    std_p = np.asarray(std_p, dtype=np.float64)
    if t_cca.shape[0] != c_cca.shape[0] or t_cca.shape[1] != mean_c.shape[0]:
        raise ValueError("CCA map shape mismatch")
    if mean_p.shape != mean_c.shape or std_p.shape != std_c.shape:
        raise ValueError("normalization shape mismatch")
    if np.any(std_c <= 0) or np.any(std_p <= 0):
        raise ValueError("std must be positive")
    scale = std_p / std_c
    w = t_cca * scale[None, :]
    b = c_cca + t_cca @ ((mean_p - mean_c) / std_c)
    return w, b


def apply_readin(r, w, mean, std, bias=None) -> np.ndarray:
    """``u = W ((r - mean) / std) + bias`` with ``r`` ``[T, N]``."""
    x = (np.asarray(r, dtype=np.float64) - mean) / std
    u = x @ np.asarray(w, dtype=np.float64).T
    if bias is not None:
        u = u + np.asarray(bias, dtype=np.float64).reshape(-1)
    return u


def project_update(delta_w: np.ndarray, h: np.ndarray, d=None, rank_tol: float = 1e-8) -> np.ndarray:
    """Project each row of ``delta_w`` ``[K, N]`` onto the row space of ``H^T D``.

    Diagnostic only. The projector is the right singular subspace of ``H^T D``.
    """
    delta = np.asarray(delta_w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if delta.shape[1] != h.shape[0]:
        raise ValueError("delta_w / H column mismatch")
    dv = _as_d(d, h.shape[0])
    m = (h * dv[:, None]).T                            # H^T D, [p, N]
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return np.zeros_like(delta)
    rank = int(np.sum(s > rank_tol * s[0]))
    if rank == 0:
        return np.zeros_like(delta)
    basis = vt[:rank].T                               # [N, rank]
    return (delta @ basis) @ basis.T


def target_side_solve(w0, f, raw: RawMoments, mean, std, lam: float, eps: float, d=None,
                      probe: list | None = None) -> np.ndarray:
    """Target-side commit. Forces ``torch.no_grad`` and never backprops.

    ``probe``, when passed, receives the grad-enabled flag observed inside the
    guard (always ``False``). The returned table is a numpy array.
    """
    if torch is None:
        if probe is not None:
            probe.append(False)
        h, g = moments_to_normalized(raw, mean, std)
        return solve_p0(w0, f, h, g, lam, eps, d)
    with torch.no_grad():
        if probe is not None:
            probe.append(bool(torch.is_grad_enabled()))
        h, g = moments_to_normalized(raw, mean, std)
        table = solve_p0(w0, f, h, g, lam, eps, d)
    return table


def cost_model(n: int, k: int = 10, p: int = 16) -> dict:
    """Analytic cost of the raw-coordinate accumulator and the commit solve.

    A MAC is one multiply of a weight-like or activation-like operand by
    another activation (feasibility rule 8). Adds alone are counted separately
    and are not MACs. The ``p x p`` Gram is counted both fully and as the
    symmetric triangle the hardware can actually update. Solve flops are the
    leading term of an LDL factorization plus ``N`` right-hand sides, not a
    measured cycle count.
    """
    n, k, p = int(n), int(k), int(p)
    macs_per_bin = {
        "sum_rphi": n * p,
        "sum_phiphi_full": p * p,
        "sum_phiphi_symmetric": p * (p + 1) // 2,
        "sum_r2": n,
    }
    macs_per_bin["total_full_G"] = (macs_per_bin["sum_rphi"] + macs_per_bin["sum_phiphi_full"]
                                    + macs_per_bin["sum_r2"])
    adds_per_bin = {"sum_r": n, "sum_phi": p, "count": 1}
    # FP32 bytes of one raw accumulator (the form that is legal to keep across
    # normalization changes). Double-buffered so a commit can snapshot.
    floats = n * p + p * p + 2 * n + p + 1
    commit = {
        "build_H_macs": n * p,                 # scale centered sum_rphi by 1/std
        "HtDH_macs": n * p * p,                # D = I or diagonal: still N*p*p
        "FG_macs": k * p * p,
        "W0H_macs": k * n * p,
        "solve_flops": (p ** 3) // 3 + n * p * p,
        "apply_update_macs": k * p * n,        # (K x p) @ (p x N)
    }
    commit["total_macs_excluding_solve_flops"] = (
        commit["build_H_macs"] + commit["HtDH_macs"] + commit["FG_macs"]
        + commit["W0H_macs"] + commit["apply_update_macs"])
    return {
        "N": n, "K": k, "p": p,
        "accumulator_floats": floats,
        "accumulator_bytes_fp32": floats * 4,
        "accumulator_bytes_fp32_double_buffer": floats * 8,
        "solve_matrix": f"{p}x{p}",
        "macs_per_bin": macs_per_bin,
        "adds_per_bin": adds_per_bin,
        "commit": commit,
    }
