"""Track A association-profile math: accumulators, factor analysis, frames.

All routines are closed-form or EM.  They use only:
1. outer-product accumulators,
2. small matrix solves (K x K or D x D, K <= 32),
3. one K x K SVD.

Every function takes float64 NumPy arrays and returns plain arrays or small
dicts.  No routine touches labels unless its name or argument says so.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "factor_analysis_em", "posterior_readin", "procrustes_frame",
    "behavior_features", "labeled_sufficient_stats", "supervised_profiles",
    "fit_shared_profile_map", "shrinkage_profiles", "am1_table",
    "profile_prior", "latent_covariance", "fa_readin", "am1_whitened_table",
]


def fa_readin(x: np.ndarray, k: int, *, iters: int = 300, seed: int = 0,
              restarts: int = 3) -> tuple[dict, float]:
    """FA posterior read-in with restarts; best log-likelihood wins.

    The EM warm start is deterministic (PCA plus seeded jitter).  Multiple
    restarts guard against local optima.  Returns ``(fa, readin, spread)``
    where ``spread`` is the log-likelihood range over restarts.
    """
    x = np.asarray(x, dtype=np.float64)
    restarts = max(1, int(restarts))
    results = [factor_analysis_em(x, k, iters=iters, seed=seed + 1000 * i)
               for i in range(restarts)]
    likelihoods = [r["loglik"] for r in results]
    best = results[int(np.argmax(likelihoods))]
    spread = float(max(likelihoods) - min(likelihoods)) if restarts > 1 else 0.0
    return best, posterior_readin(best["H"], best["psi"]), spread


def am1_whitened_table(x_labeled: np.ndarray, x_all: np.ndarray, y_labeled: np.ndarray,
                       b_src: np.ndarray, source_stats: tuple[np.ndarray, np.ndarray],
                       *, k: int, head_lam: float = 1.0,
                       em_iters: int = 300, seed: int = 0, restarts: int = 3,
                       null_pull: float = 0.1) -> dict:
    """A-M1 with explicit whitening on both latent sides (revision items 5, 6).

    Differences to ``am1_table``:

    1. The target FA latent is z-scored per dimension with unlabeled
       statistics, and the source latent statistics ``(m_s, s_s)`` are passed
       in.  The frame then maps whitened latents to whitened latents, where an
       orthogonal map is the correct model.
    2. The Procrustes solution completes the label-null subspace canonically:
       the SVD input gets ``null_pull`` relative identity bias, so the null
       block of ``T`` is pulled toward identity instead of an arbitrary SVD
       completion.  Same-seed and cross-seed results become stable.
    3. The deployment transform carries an explicit latent offset, so the
       frame fit on centered latents no longer drops the mean (item 6).

    Returns the table ``W`` (K x N), the latent offset ``c`` (K), the frame,
    and diagnostics.  Deployment computes ``z = W x + c``.
    """
    x_all = np.asarray(x_all, dtype=np.float64)
    x_labeled = np.asarray(x_labeled, dtype=np.float64)
    y_labeled = np.asarray(y_labeled, dtype=np.float64)
    b_src = np.asarray(b_src, dtype=np.float64)
    m_s = np.asarray(source_stats[0], dtype=np.float64)
    s_s = np.asarray(source_stats[1], dtype=np.float64)
    if x_labeled.shape[0] != y_labeled.shape[0]:
        raise ValueError("labeled rows must align with labels")
    fa, beta, spread = fa_readin(x_all, k, iters=em_iters, seed=seed, restarts=restarts)
    z_all = (x_all - fa["mean"]) @ beta.T
    m_t = z_all.mean(0)
    s_t = np.maximum(z_all.std(0), 1e-6)
    zeta_l = ((x_labeled - fa["mean"]) @ beta.T - m_t) / s_t
    y_c = y_labeled - y_labeled.mean(0)
    gram = zeta_l.T @ zeta_l + head_lam * np.eye(k)
    m_fit = np.linalg.solve(gram, zeta_l.T @ y_c).T                       # [D, K]
    b_eff = b_src * s_s[None, :]                                         # whitened-space head
    cross = b_eff.T @ m_fit
    biased = cross + null_pull * np.linalg.norm(cross, "fro") * np.eye(k)
    u, _, vt = np.linalg.svd(biased)
    t_frame = u @ vt
    # z(x) = diag(s_s) T diag(1/s_t) (beta (x - mu) - m_t) + m_s
    p_mat = s_s[:, None] * t_frame * (1.0 / s_t)[None, :]
    table = p_mat @ beta
    latent_offset = m_s - p_mat @ (beta @ fa["mean"] + m_t)
    return {"table": table, "latent_offset": latent_offset, "T": t_frame,
            "M": m_fit, "beta": beta, "fa": fa, "restart_spread": spread,
            "target_latent_mean": m_t, "target_latent_std": s_t,
            "labeled_bins": int(len(y_labeled))}


def factor_analysis_em(x: np.ndarray, k: int, *, iters: int = 300,
                       tol: float = 1e-10, seed: int = 0) -> dict:
    """Fit ``x = mean + H phi + eps`` with diagonal noise ``psi`` by EM.

    ``x`` is ``[M, N]``.  Returns ``{'mean', 'H', 'psi', 'loglik', 'iters'}``
    with ``H`` ``[N, k]`` and ``psi`` ``[N]``.  The latent scale is fixed by
    ``cov(phi) = I``; the frame of ``H`` is arbitrary.
    """
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    if k < 1 or k > n:
        raise ValueError(f"k={k} must be in [1, {n}]")
    mean = x.mean(0)
    xc = x - mean
    s = (xc.T @ xc) / max(m - 1, 1)
    rng = np.random.default_rng(seed)
    # Deterministic PCA warm start; random jitter breaks exact degeneracies.
    evals, evecs = np.linalg.eigh(s)
    order = np.argsort(evals)[::-1][:k]
    h = evecs[:, order] * np.sqrt(np.clip(evals[order], 1e-12, None))
    h = h + 0.01 * rng.standard_normal(h.shape)
    # Heywood guard: keep every noise variance a small fraction of its channel
    # variance.  Without it one channel can collapse to psi ~ 0 and dominate
    # the posterior read-in through 1/psi.
    psi_floor = np.maximum(1e-6, 1e-3 * np.diag(s))
    psi = np.clip(np.diag(s) - np.einsum("ij,ij->i", h, h), psi_floor, None)
    previous = -np.inf
    loglik = -np.inf
    used = 0
    for used in range(1, iters + 1):
        psi_inv = 1.0 / psi
        g = (h * psi_inv[:, None]).T @ h                       # k x k
        a_mat = np.linalg.inv(np.eye(k) + g)
        beta = a_mat @ (h * psi_inv[:, None]).T                # k x N
        e_phi_phi = a_mat + beta @ s @ beta.T                  # k x k
        h_new = s @ beta.T @ np.linalg.inv(e_phi_phi)
        psi_new = np.clip(np.diag(s - h_new @ beta @ s), psi_floor, None)
        if not (np.isfinite(h_new).all() and np.isfinite(psi_new).all()):
            break
        delta = max(np.abs(h_new - h).max(), np.abs(psi_new - psi).max())
        h, psi = h_new, psi_new
        # Marginal log-likelihood under the isotropic-latent FA model.
        sign, logdet = np.linalg.slogdet(h @ h.T + np.diag(psi))
        if sign > 0:
            solved = np.linalg.solve(h @ h.T + np.diag(psi), xc.T)   # [N, M]
            loglik = -0.5 * (m * logdet + float(np.einsum("ij,ji", xc, solved)))
        if delta < tol:
            break
    return {"mean": mean, "H": h, "psi": psi, "loglik": float(loglik), "iters": used}


def posterior_readin(h: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Return the posterior mean read-in ``beta = A H^T Psi^{-1}`` ``[k, N]``."""
    h = np.asarray(h, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    psi_inv = 1.0 / psi
    k = h.shape[1]
    gram = (h * psi_inv[:, None]).T @ h
    a_mat = np.linalg.inv(np.eye(k) + gram)
    return a_mat @ (h * psi_inv[:, None]).T


def procrustes_frame(b_src: np.ndarray, m_fit: np.ndarray) -> np.ndarray:
    """Orthogonal ``T`` ``[k, k]`` minimizing ``||b_src T - m_fit||_F``.

    ``b_src`` and ``m_fit`` are ``[D, k]``.  When ``D < k`` the objective is
    blind to part of ``T``; the SVD then fixes the null-space completion
    deterministically.  Callers that need a specific completion must rotate
    the inputs first.
    """
    b_src = np.asarray(b_src, dtype=np.float64)
    m_fit = np.asarray(m_fit, dtype=np.float64)
    if b_src.shape != m_fit.shape or b_src.ndim != 2:
        raise ValueError(f"shape mismatch b_src={b_src.shape} m_fit={m_fit.shape}")
    u, _, vt = np.linalg.svd(b_src.T @ m_fit)
    return u @ vt


def behavior_features(y: np.ndarray, *, lags: int = 2) -> np.ndarray:
    """Build ``phi_t = [y_t, y_{t-1}, ..., y_{t-lags}]`` ``[T, D*(lags+1)]``.

    Rows with any NaN input stay NaN.  The caller must mask them out.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must be [T, D]")
    parts = [y]
    for lag in range(1, lags + 1):
        shifted = np.full_like(y, np.nan)
        shifted[lag:] = y[:-lag]
        parts.append(shifted)
    phi = np.concatenate(parts, axis=1)
    return phi


def labeled_sufficient_stats(phi: np.ndarray, x: np.ndarray) -> dict:
    """Accumulate ``S_phiphi`` and ``S_phipx`` over finite rows of ``phi``."""
    phi = np.asarray(phi, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(phi).all(1)
    if not good.any():
        raise ValueError("no finite behavior-feature rows")
    p, xi = phi[good], x[good]
    return {"count": int(good.sum()),
            "s_phiphi": p.T @ p,
            "s_phipx": p.T @ xi,
            "mean_x": xi.mean(0)}


def supervised_profiles(phi: np.ndarray, x: np.ndarray, *, lam: float = 1.0) -> dict:
    """A-M0 step 2: ``H^T = (S_phiphi + lam I)^-1 S_phipx`` on labeled rows.

    Returns ``{'H': [N, K0], 'count', 'mean_x'}`` plus the sufficient stats.
    """
    stats = labeled_sufficient_stats(phi, x)
    k0 = stats["s_phiphi"].shape[0]
    regularized = stats["s_phiphi"] + lam * np.eye(k0)
    h_t = np.linalg.solve(regularized, stats["s_phipx"])
    return {"H": h_t.T, **stats}


def profile_prior(profiles: list[np.ndarray]) -> dict:
    """Pool unit profiles ``[N_s, K0]`` into mean and diagonal variance priors."""
    rows = np.concatenate([np.asarray(p, dtype=np.float64) for p in profiles], axis=0)
    return {"mean": rows.mean(0), "var": rows.var(0) + 1e-9, "units": int(len(rows))}


def shrinkage_profiles(phi: np.ndarray, x: np.ndarray, prior: dict, *, lam: float = 1.0) -> dict:
    """A-M2: ridge toward the source population prior ``(mean, var)``.

    ``H^T = (S_phiphi + lam*diag(1/var))^-1 (S_phipx + lam*diag(1/var)*mean^T)``.
    """
    stats = labeled_sufficient_stats(phi, x)
    prior_precision = np.diag(1.0 / np.asarray(prior["var"], dtype=np.float64))
    regularized = stats["s_phiphi"] + lam * prior_precision
    rhs = stats["s_phipx"] + lam * (prior_precision @ np.asarray(prior["mean"], dtype=np.float64))[:, None]
    h_t = np.linalg.solve(regularized, rhs)
    return {"H": h_t.T, **stats}


def fit_shared_profile_map(source_profiles: list[np.ndarray], source_tables: list[np.ndarray],
                           *, lam: float = 1.0) -> dict:
    """Fit the shared map ``g`` with ``W_s[:, i] ~= H_s[i, :] @ g``.

    ``source_profiles[s]`` is ``[N_s, K0]``; ``source_tables[s]`` is ``[K, N_s]``.
    Returns ``{'g': [K0, K], 'units', 'residual'}`` pooled over all sessions.
    """
    phi_rows = np.concatenate([np.asarray(p, dtype=np.float64) for p in source_profiles], axis=0)
    table_rows = np.concatenate([np.asarray(t, dtype=np.float64).T for t in source_tables], axis=0)
    if phi_rows.shape[0] != table_rows.shape[0]:
        raise ValueError(f"profile rows {phi_rows.shape[0]} != table rows {table_rows.shape[0]}")
    k0 = phi_rows.shape[1]
    gram = phi_rows.T @ phi_rows + lam * np.eye(k0)
    g = np.linalg.solve(gram, phi_rows.T @ table_rows)
    residual = float(np.abs(phi_rows @ g - table_rows).max())
    return {"g": g, "units": int(phi_rows.shape[0]), "residual": residual}


def latent_covariance(z: np.ndarray) -> np.ndarray:
    """Covariance of latents ``[M, K]``; used by the whitening penalty."""
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(0)
    return (zc.T @ zc) / max(len(z) - 1, 1)


def am1_table(x_labeled: np.ndarray, x_all: np.ndarray, y_labeled: np.ndarray,
              b_src: np.ndarray, *, k: int, lags: int = 2, lam: float = 1.0,
              em_iters: int = 300, seed: int = 0, head_lam: float = 1.0,
              psi: np.ndarray | None = None) -> dict:
    """A-M1: label-free unit table with a labeled frame.

    1. FA on ``x_all`` (labeled and unlabeled bins) gives ``H`` and ``psi``.
    2. ``beta = A H^T Psi^-1`` is the posterior read-in.
    3. Ridge-fit ``M`` from labels ``y`` onto latents ``beta (x - mean)``.
    4. ``T`` is the orthogonal Procrustes map from ``b_src`` to ``M``.
    5. The table is ``W = T beta`` ``[K, N]``.

    ``psi`` may be supplied externally (A-M3 uses an unlabeled estimate).
    Returns the table, the frame, and diagnostics.
    """
    x_all = np.asarray(x_all, dtype=np.float64)
    x_labeled = np.asarray(x_labeled, dtype=np.float64)
    y_labeled = np.asarray(y_labeled, dtype=np.float64)
    b_src = np.asarray(b_src, dtype=np.float64)
    if x_labeled.shape[0] != y_labeled.shape[0] or x_labeled.shape[1] != x_all.shape[1]:
        raise ValueError("labeled rows must align with labels and channel count")
    fa = factor_analysis_em(x_all, k, iters=em_iters, seed=seed)
    noise = fa["psi"] if psi is None else np.asarray(psi, dtype=np.float64)
    beta = posterior_readin(fa["H"], noise)
    z_l = (x_labeled - fa["mean"]) @ beta.T
    d = y_labeled.shape[1]
    gram = z_l.T @ z_l + head_lam * np.eye(k)
    m_fit = np.linalg.solve(gram, z_l.T @ y_labeled).T                    # [D, K]
    t_frame = procrustes_frame(b_src, m_fit)
    table = t_frame @ beta
    return {"table": table, "T": t_frame, "M": m_fit, "beta": beta,
            "fa": fa, "head_ridge": m_fit, "labeled_bins": int(len(y_labeled)), "d": d}


def fit_channel_correction(b_src: np.ndarray, prior_table: np.ndarray,
                           x_labeled: np.ndarray, y_labeled: np.ndarray,
                           *, lam: float = 1.0) -> np.ndarray:
    """Closed-form per-channel correction on a prior table (revision 4.2.1).

    The deployed table is ``W = W0 diag(g)`` with the prior ``W0`` fixed.
    Under the linear head ``B_src`` the label residual is linear in ``g``:

        y_t - B_src W0 x_t  =  sum_i g_i' x_{t,i} (B_src W0)_{:, i}

    with ``g' = g - 1``.  One ridge solve over the labeled bins gives ``g``.
    Only N parameters are fit, so a handful of trials suffices.
    """
    b_src = np.asarray(b_src, dtype=np.float64)
    prior_table = np.asarray(prior_table, dtype=np.float64)
    x_labeled = np.asarray(x_labeled, dtype=np.float64)
    y_labeled = np.asarray(y_labeled, dtype=np.float64)
    path = b_src @ prior_table                                              # [D, N]
    base = x_labeled @ path.T                                               # [L, D]
    residual = y_labeled - base
    rows, n = x_labeled.shape
    d = y_labeled.shape[1]
    design = np.empty((rows * d, n))
    for i in range(n):
        design[:, i] = np.outer(x_labeled[:, i], path[:, i]).reshape(-1)
    target = residual.reshape(-1)
    gram = design.T @ design + lam * np.eye(n)
    g_delta = np.linalg.solve(gram, design.T @ target)
    return 1.0 + g_delta


def prior_correction_table(b_src: np.ndarray, prior_table: np.ndarray,
                           x_labeled: np.ndarray, y_labeled: np.ndarray,
                           *, lam: float = 1.0) -> dict:
    """Assemble ``W = W0 diag(g)`` for a prior table and labeled bins."""
    gains = fit_channel_correction(b_src, prior_table, x_labeled, y_labeled, lam=lam)
    return {"table": prior_table * gains[None, :], "channel_gains": gains,
            "gain_deviation_max": float(np.abs(gains - 1.0).max())}
