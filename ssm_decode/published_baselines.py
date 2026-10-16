"""Published read-in-alignment baselines on a frozen AP backbone (DAC2027 evaluation).

Two methods are implemented here; both output an affine read-in
``z = T @ ((x - mean) / std) + c`` that feeds the frozen Mamba-3 trunk, so they
are scored on exactly the same contiguous causal decode as P-prior / P0.

1. ``stabilizer_readin`` -- Degenhart et al., Nat. Biomed. Eng. 4:672-685 (2020).
   Follows the authors' companion code (github.com/alandegenhart/stabilizedbci):
   factor analysis (FA, EM, multiple restarts, minimum private variance) on the
   baseline day and on the new day; stable channels are identified by iterative
   orthogonal-Procrustes row dropping (``identifyStableLoadingRows``); the new
   day's loading matrix is rotated onto the baseline loading matrix
   (``C_new <- C_new W'``); the latent is ``l = beta y + o`` with
   ``beta = C'(CC' + Psi)^-1``.  Label-free.
2. ``cca_readin`` -- Gallego et al., Nat. Neurosci. 23:260-270 (2020).  PCA to K
   latents on each day, condition-matched (reach-direction) trial averages,
   canonical correlation analysis between the two days' averaged latent
   trajectories, target latents mapped into the baseline-day latent coordinates
   through the canonical transforms.  Uses reach-direction labels of the
   calibration trials (not behaviour).

The latent dimension ``d`` of FA / PCA is a free parameter (selected on source pseudo-targets; ``k`` in the
function signatures): the bridge is then a K x d least-squares map, exact when ``d`` equals the number of
input dimensions.  The input can be sorted units or electrode-pooled channels (``ssm_decode.electrode_align``).

The baseline-day latent -> AP-latent bridge ``A`` (a K x d least-squares map fitted
on the baseline *source* session, where the AP latent ``z_ref = W_ref x`` is known)
stands in for the decoder of the original papers: the frozen trunk plays the role
of the day-1 decoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DIR_BINS = 8


# ------------------------------------------------------------------ FA ----

def fa_em(x: np.ndarray, k: int, *, iters: int = 400, tol: float = 1e-5,
          min_priv: float = 0.1, restarts: int = 5, seed: int = 0) -> dict:
    """Factor analysis by EM on ``x`` [M, N] (rows = samples).

    Mirrors ``fitFA``/``fitBaseStabilizer`` of the authors' code: multiple random
    restarts, best final log-likelihood, private variances floored at ``min_priv``
    (the authors' default 0.1 is expressed in the units of ``x``; callers z-score
    ``x`` so that it means 10 % of unit variance).  Returns mean d, loading C [N, k],
    private variance psi [N].
    """
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    d = x.mean(0)
    xc = x - d
    s = xc.T @ xc / max(m - 1, 1)
    evals, evecs = np.linalg.eigh(s)
    order = np.argsort(evals)[::-1][:k]
    c_pca = evecs[:, order] * np.sqrt(np.clip(evals[order], 1e-9, None))
    best, best_ll = None, -np.inf
    for r in range(max(1, restarts)):
        rng = np.random.default_rng(seed + 7919 * r)
        c = c_pca + (0.0 if r == 0 else 0.3) * rng.standard_normal(c_pca.shape) * np.sqrt(np.diag(s).mean())
        psi = np.maximum(np.diag(s) - np.einsum("ij,ij->i", c, c), min_priv)
        prev = -np.inf
        ll = -np.inf
        for _ in range(iters):
            pinv = 1.0 / psi
            g = (c * pinv[:, None]).T @ c
            a = np.linalg.inv(np.eye(k) + g)
            beta = a @ (c * pinv[:, None]).T                    # [k, N]
            e_zz = a + beta @ s @ beta.T
            c = s @ beta.T @ np.linalg.inv(e_zz)
            psi = np.maximum(np.diag(s - c @ beta @ s), min_priv)
            cov = c @ c.T + np.diag(psi)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                break
            ll = -0.5 * (logdet + np.trace(np.linalg.solve(cov, s)))
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        if ll > best_ll:
            best_ll, best = ll, {"d": d, "C": c.copy(), "psi": psi.copy(), "ll": float(ll)}
    return best


def fa_fit(xz: np.ndarray, k: int, **kw) -> dict:
    """``fa_em`` restricted to the columns that vary in ``xz`` (silent channels of an electrode-pooled input
    carry no information); loadings of silent columns are zero, latent dimensions beyond the number of active
    columns get zero loadings, so shapes stay ``C [N, k]``."""
    xz = np.asarray(xz, dtype=np.float64)
    act = xz.std(0) > 1e-8
    n_act = int(act.sum())
    if n_act == 0:
        raise ValueError("no active column")
    k_eff = min(k, n_act)
    fa = fa_em(xz[:, act], k_eff, **kw)
    c = np.zeros((xz.shape[1], k))
    c[np.flatnonzero(act), :k_eff] = fa["C"]
    psi = np.full(xz.shape[1], kw.get("min_priv", 0.1))
    psi[act] = fa["psi"]
    d = np.zeros(xz.shape[1])
    d[act] = fa["d"]
    return {"d": d, "C": c, "psi": psi, "ll": fa["ll"]}


def fa_beta(fa: dict) -> tuple[np.ndarray, np.ndarray]:
    """``l = beta y + o`` with ``beta = C'(CC' + Psi)^-1`` (getStabilizatonMatrices)."""
    c, psi = fa["C"], fa["psi"]
    beta = c.T @ np.linalg.inv(c @ c.T + np.diag(psi))
    return beta, -beta @ fa["d"]


# --------------------------------------------------------- Procrustes ----

def orthonormal_transform(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """``T`` minimising ||m1 - m2 T'||_F (learnOptimalOrthonormalTransformation)."""
    u, _, vt = np.linalg.svd(m1.T @ m2)
    return u @ vt


def identify_stable_rows(m1: np.ndarray, m2: np.ndarray, n_stable: int, th: float) -> np.ndarray:
    """Iterative row dropping, a port of identifyStableLoadingRows.m (boolean mask)."""
    small = (np.linalg.norm(m1, axis=1) < th) | (np.linalg.norm(m2, axis=1) < th)
    clean = np.flatnonzero(~small)
    m1c, m2c = m1[clean], m2[clean]
    cur = np.arange(len(clean))
    for i in range(1, max(len(clean) - n_stable, 0) + 1):
        w = orthonormal_transform(m1c[cur], m2c[cur])
        delta = np.linalg.norm(m1c[cur] - m2c[cur] @ w.T, axis=1)
        keep = np.argsort(delta, kind="stable")[: len(clean) - i]
        cur = np.sort(cur[keep])
    mask = np.zeros(len(m1), dtype=bool)
    mask[clean[cur]] = True
    return mask


def match_units_by_electrode(elec_a: np.ndarray, rate_a: np.ndarray,
                             elec_b: np.ndarray, rate_b: np.ndarray):
    """Pair units of two sessions that share an electrode.

    Units on the same electrode are paired in rank order of mean firing rate
    (a label-free rule; unit ids are session-specific).  Returns index arrays.
    """
    ia, ib = [], []
    for e in np.intersect1d(elec_a, elec_b):
        ua = np.flatnonzero(elec_a == e)
        ub = np.flatnonzero(elec_b == e)
        ua = ua[np.argsort(-rate_a[ua], kind="stable")]
        ub = ub[np.argsort(-rate_b[ub], kind="stable")]
        for p, q in zip(ua, ub):
            ia.append(p)
            ib.append(q)
    return np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)


# ------------------------------------------------------ baseline state ----

@dataclass
class RefDay:
    """Everything fitted on the baseline (most recent source) session."""
    name: str
    electrodes: np.ndarray
    rate: np.ndarray
    mean: np.ndarray            # AP-training input stats of this source session
    std: np.ndarray
    fa: dict | None = None
    beta: np.ndarray | None = None
    o: np.ndarray | None = None
    a_fa: np.ndarray | None = None      # FA latent -> AP latent  [K, K]
    c_fa: np.ndarray | None = None
    pca_p: np.ndarray | None = None     # [N, K]
    a_pca: np.ndarray | None = None
    c_pca: np.ndarray | None = None
    cond_avg: dict | None = None        # cond -> [T, K] mean PCA-latent trajectory
    zmean: np.ndarray | None = None     # per-session z-score stats used by FA / PCA
    zstd: np.ndarray | None = None


def _lstsq_affine(src: np.ndarray, dst: np.ndarray):
    """dst ~ src @ A.T + c  (A [K_dst, K_src])."""
    ones = np.ones((len(src), 1))
    sol, *_ = np.linalg.lstsq(np.concatenate([src, ones], 1), dst, rcond=None)
    return sol[:-1].T, sol[-1]


def pca_basis(xz: np.ndarray, k: int) -> np.ndarray:
    xc = xz - xz.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:k].T                                     # [N, <=k]
    if basis.shape[1] < k:                               # fewer directions than requested: zero-pad
        basis = np.concatenate([basis, np.zeros((basis.shape[0], k - basis.shape[1]))], 1)
    return basis                                         # [N, k]


def phase_average(x: np.ndarray, t_phase: int) -> np.ndarray:
    """Time-normalise one trial [T, N] to ``t_phase`` bins by segment averaging."""
    segs = np.array_split(np.arange(len(x)), t_phase)
    return np.stack([x[s].mean(0) for s in segs])


def condition_label(angle: float) -> int:
    if not np.isfinite(angle):
        return -1
    return int(np.round(angle / (math.pi / 4))) % DIR_BINS


def apply_transform(x, kind: str):
    """Input transform of the count data: ``raw`` or ``sqrt`` (Gallego et al. use sqrt-transformed counts)."""
    if kind == "raw":
        return x
    if kind == "sqrt":
        return np.sqrt(np.maximum(x, 0.0))
    raise ValueError(kind)


def build_ref_day(*, name, x_fit, z_ref, electrodes, ref_mean, ref_std, x_ref_trials, ref_conds,
                  k, fa_min_priv, fa_seed=0, t_phase=20, transform="raw", with_fa=True, with_pca=True):
    """Fit FA / PCA on the baseline session and the latent -> AP-latent bridges.

    x_fit [M, N] raw counts of valid bins of the baseline session, z_ref [M, K] the
    known AP latent of those bins (``W_ref (x - ref_mean)/ref_std``);
    x_ref_trials / ref_conds: per-trial raw counts [T_i, N] and reach-direction labels.
    """
    raw_fit = x_fit
    x_fit = apply_transform(x_fit, transform)
    zmean = x_fit.mean(0)
    zstd = np.where(x_fit.std(0) < 1e-6, 1.0, x_fit.std(0))
    xz = (x_fit - zmean) / zstd
    ref = RefDay(name=name, electrodes=np.asarray(electrodes), rate=x_fit.mean(0),
                 mean=ref_mean, std=ref_std, zmean=zmean, zstd=zstd)
    if with_fa:
        ref.fa = fa_fit(xz, k, min_priv=fa_min_priv, seed=fa_seed)
        ref.beta, ref.o = fa_beta(ref.fa)
        lat = xz @ ref.beta.T + ref.o
        ref.a_fa, ref.c_fa = _lstsq_affine(lat, z_ref)
    if not with_pca:
        return ref
    ref.pca_p = pca_basis(xz, k)
    lat_p = (xz - xz.mean(0)) @ ref.pca_p
    ref.a_pca, ref.c_pca = _lstsq_affine(lat_p, z_ref)
    traj: dict[int, list] = {}
    for xt, cnd in zip(x_ref_trials, ref_conds):
        if cnd < 0 or len(xt) < t_phase:
            continue
        lt = ((apply_transform(xt, transform) - zmean) / zstd) @ ref.pca_p
        traj.setdefault(cnd, []).append(phase_average(lt, t_phase))
    ref.cond_avg = {c: np.mean(v, 0) for c, v in traj.items()}
    return ref


# --------------------------------------------------------- Stabilizer ----

def stabilizer_readin(ref: RefDay, x_fit: np.ndarray, electrodes: np.ndarray, *, k: int,
                      align_frac: float, align_th: float = 0.01, fa_min_priv: float = 0.1,
                      seed: int = 0, align: bool = True, restarts: int = 5):
    """Degenhart-style stabilised read-in for the new day.

    Returns ``(T, c, mean, std, info)`` with ``z = T @ ((x-mean)/std) + c``.
    """
    mean = x_fit.mean(0)
    std = np.where(x_fit.std(0) < 1e-6, 1.0, x_fit.std(0))
    xz = (x_fit - mean) / std
    fa = fa_fit(xz, k, min_priv=fa_min_priv, seed=seed, restarts=restarts)
    ia, ib = match_units_by_electrode(ref.electrodes, ref.rate, electrodes, x_fit.mean(0))
    live = (ref.rate[ia] > 0) & (x_fit.mean(0)[ib] > 0)      # silent channels (electrode-pooled input) are no pairs
    ia, ib = ia[live], ib[live]
    info = {"n_matched": int(len(ia)), "n_units": int(xz.shape[1])}
    if align and len(ia) >= 2:
        n_stable = max(int(round(align_frac * len(ia))), min(len(ia), k))
        n_stable = min(n_stable, len(ia))
        m1 = ref.fa["C"][ia]
        m2 = fa["C"][ib]
        mask = identify_stable_rows(m1, m2, n_stable, align_th)
        if mask.sum() >= 1:
            w = orthonormal_transform(m1[mask], m2[mask])
            fa = {**fa, "C": fa["C"] @ w.T}
        info["n_stable"] = int(mask.sum())
    else:
        info["n_stable"] = 0
    beta, o = fa_beta(fa)
    t = ref.a_fa @ beta
    c = ref.a_fa @ o + ref.c_fa
    return t, c, mean, std, info


# --------------------------------------------------------------- CCA ----

def cca(lr: np.ndarray, lt: np.ndarray, ridge: float = 0.0):
    """CCA between ``lr`` and ``lt`` [S, K] (rows paired).

    Returns ``(Mr, Mt, mr, mt, corr)`` such that canonical variables are
    ``(l - m) @ M`` and are maximally correlated, unit variance.
    ``ridge`` adds ridge * mean-variance * I to both covariances (0 = plain CCA).
    """
    mr, mt = lr.mean(0), lt.mean(0)
    xr, xt = lr - mr, lt - mt
    n = len(xr)
    crr = xr.T @ xr / max(n - 1, 1)
    ctt = xt.T @ xt / max(n - 1, 1)
    crt = xr.T @ xt / max(n - 1, 1)
    kk = lr.shape[1]
    jr = ridge * np.trace(crr) / kk + 1e-9 * np.trace(crr) / kk
    jt = ridge * np.trace(ctt) / kk + 1e-9 * np.trace(ctt) / kk

    def inv_sqrt(c, j):
        w, v = np.linalg.eigh(c + j * np.eye(kk))
        w = np.clip(w, 1e-12, None)
        return (v / np.sqrt(w)) @ v.T

    ir, it = inv_sqrt(crr, jr), inv_sqrt(ctt, jt)
    u, sv, vt = np.linalg.svd(ir @ crt @ it)
    return ir @ u, it @ vt.T, mr, mt, sv


def cca_readin(ref: RefDay, x_fit: np.ndarray, tgt_trials: list, tgt_conds: list, *, k: int,
               ridge: float, t_phase: int = 20, min_conditions: int = 1, transform: str = "raw"):
    """Gallego-style CCA alignment of the new day onto the baseline-day latents.

    ``x_fit``: raw counts used to fit the target PCA; ``tgt_trials`` / ``tgt_conds``:
    per-trial raw counts and reach-direction labels used for the condition averages.
    Returns ``(T, c, mean, std, info)`` or ``None`` when no condition overlaps.
    """
    x_fit = apply_transform(x_fit, transform)
    mean = x_fit.mean(0)
    std = np.where(x_fit.std(0) < 1e-6, 1.0, x_fit.std(0))
    xz = (x_fit - mean) / std
    p_t = pca_basis(xz, k)
    traj: dict[int, list] = {}
    for xt, cnd in zip(tgt_trials, tgt_conds):
        if cnd < 0 or len(xt) < t_phase:
            continue
        traj.setdefault(cnd, []).append(phase_average(((apply_transform(xt, transform) - mean) / std) @ p_t, t_phase))
    common = sorted(set(traj) & set(ref.cond_avg))
    info = {"n_common_conditions": len(common), "n_cca_samples": len(common) * t_phase}
    if len(common) < min_conditions or not common:
        return None
    lt = np.concatenate([np.mean(traj[c], 0) for c in common])
    lr = np.concatenate([ref.cond_avg[c] for c in common])
    mr_mat, mt_mat, m_r, m_t, corr = cca(lr, lt, ridge)
    info["mean_canon_corr"] = float(corr.mean())
    # target latent -> baseline latent: (l - m_t) Mt Mr^-1 + m_r
    trans = mt_mat @ np.linalg.inv(mr_mat)                # [K, K], row-vector convention
    # z = a_pca @ [ ((xz) p_t - m_t) trans + m_r ] + c_pca
    t_total = ref.a_pca @ trans.T @ p_t.T
    c_total = ref.a_pca @ (-(m_t @ trans) + m_r) + ref.c_pca
    return t_total, c_total, mean, std, info
