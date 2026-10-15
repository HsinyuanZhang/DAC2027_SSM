"""Architecture-level cost of one calibration (P2 item A'4), analytic.

Counts are derived from the tensor shapes actually used by
``population_calib`` (segment_sums, describe, PopulationCalibrator) and are
cross-checked in the unit tests against (i) real PopStats tensors and (ii)
forward hooks on an instantiated calibrator.  A MAC is one multiply-add; plain
accumulations (adds) are reported separately.  Not measured on hardware.
"""
from __future__ import annotations

import math

from torch import nn

from ssm_decode.population_calib import (AC_LAGS, CROSS_LAGS, DIR_BINS, REL_DIM,
                                         PopulationCalibrator)

N_LAGS_AC, N_LAGS_X = len(AC_LAGS), len(CROSS_LAGS)


def desc_dim_general(k: int, d: int) -> int:
    """Descriptor width for K latents and D behaviour dims (== desc_dim at D=2)."""
    return 2 + N_LAGS_AC + k + N_LAGS_X * d + DIR_BINS + 2 + k + 2


def acc_elements(n: int, d: int) -> dict:
    """Number of scalar accumulator entries per PopStats key (see segment_sums)."""
    return {"n": 1, "sx": n, "gram": n * n, "ac_xx": N_LAGS_AC * n, "ac_a": N_LAGS_AC * n,
            "ac_b": N_LAGS_AC * n, "ac_n": N_LAGS_AC, "sy": N_LAGS_X * d,
            "syx": N_LAGS_X * d * n, "yx_n": N_LAGS_X, "sdir": n * DIR_BINS, "cdir": DIR_BINS,
            "sphi": 3, "sphiphi": 9, "sphix": 3 * n,
            "q_sx2": n, "q_yx2": d * n, "q_yxy": d * n, "q_y2": d}


def acc_per_bin(n: int, d: int, symmetric_gram: bool = False) -> dict:
    """MACs and adds needed per valid time bin by the streaming accumulators."""
    gram = n * (n + 1) // 2 if symmetric_gram else n * n
    mac = (gram                                   # x x^T
           + N_LAGS_AC * n                        # x_t * x_{t-lag}
           + N_LAGS_X * d * n                     # y_{t+lag} x_t^T
           + 3 * n                                # phi x^T (cos / sin / const)
           + 9)                                   # phi phi^T
    add = (n                                      # sx
           + N_LAGS_AC * 2 * n                    # ac_a, ac_b
           + N_LAGS_X * d                         # sy
           + n                                    # sdir (one-hot select)
           + 3 + 1 + N_LAGS_AC + N_LAGS_X + 1)    # sphi, cdir, ac_n, yx_n, n
    return {"mac": mac, "add": add}


def acc_per_trial(n: int, d: int) -> dict:
    """Per-trial updates of the consistency (q_*) accumulators."""
    return {"mac": n + 2 * d * n + d, "div": n + 2 * d * n + d}


def finalize_macs(n: int, k: int, d: int) -> dict:
    """describe(): latent correlations, 3x3 solves and elementwise features."""
    corr = n * n * k + k * n * n + k * n          # gn @ P^T, (P gn) . P, normalisation
    gram_norm = n * n                             # centre / scale the Gram matrix
    solve = n * 3 * 9 + 27                        # closed-form 3x3 per unit (+ shared factor)
    elementwise = n * desc_dim_general(k, d) * 4  # rough: logs, divides, clamps
    return {"mac": corr + gram_norm + solve, "elementwise": elementwise}


def alt_latent_accumulators_per_bin(n: int, k: int) -> int:
    """NOT IMPLEMENTED alternative: accumulate x_i z_k with z = P x on raw x
    (K*N MACs for z, N*K for the cross sums) instead of the N^2 Gram."""
    return 2 * k * n


def forward_macs(variant: str, n: int, k: int, d: int, *, hidden: int = 64, dctx: int = 32,
                 m: int = 8) -> dict:
    """Fixed-weight calibrator forward for N units, from the instantiated modules."""
    fd = desc_dim_general(k, d)
    model = PopulationCalibrator(variant, k, fd, m=m, hidden=hidden, dctx=dctx)
    per_unit = {}
    for name, mod in model.named_children():
        if isinstance(mod, (nn.Sequential, nn.Linear)):
            lin = [x for x in mod.modules() if isinstance(x, nn.Linear)]
            per_unit[name] = sum(x.in_features * x.out_features for x in lin)
    extra = 0                                       # non-Linear matmuls
    pooling = 0
    if variant == "p2":
        pooling = n * dctx                          # mean over units
    if variant in ("p3", "p4"):
        per_unit["scores"] = hidden * m             # ph @ u^T
        per_unit["route"] = hidden * m              # ph @ a^T
        per_unit["mu_i"] = m * dctx                 # beta @ z
        pooling = n * m * dctx                      # softmax(scores)^T @ values
        extra = n * m * 2                           # exps / normalisation (elementwise)
    total = n * sum(per_unit.values()) + pooling
    params = sum(p.numel() for p in model.parameters())
    return {"per_unit": per_unit, "per_unit_total": sum(per_unit.values()),
            "pooling_macs": pooling, "forward_macs": total, "softmax_elementwise": extra,
            "n_params": params}


def sram_bytes(n: int, k: int, d: int, variant: str, *, acc_bits=32, desc_bits=16, table_bits=8,
               trunk_free=True, m: int = 8, dctx: int = 32, symmetric_gram=False) -> dict:
    el = acc_elements(n, d)
    if symmetric_gram:
        el = dict(el, gram=n * (n + 1) // 2)
    acc_total = sum(el.values())
    fd = desc_dim_general(k, d)
    out = {"acc_elements": acc_total, "acc_bytes": acc_total * acc_bits / 8,
           "acc_gram_bytes": el["gram"] * acc_bits / 8,
           "acc_without_gram_bytes": (acc_total - el["gram"]) * acc_bits / 8,
           "descriptor_bytes": n * (fd + REL_DIM) * desc_bits / 8,
           "table_bytes_single": k * n * table_bits / 8,
           "table_bytes_double": 2 * k * n * table_bits / 8,
           "norm_params_bytes": 2 * n * 2}            # mean/std per unit, 16 bit
    pool_state = 0
    if variant in ("p3", "p4"):
        pool_state = m * (dctx + 2) * 4               # online-softmax: max, norm, acc per seed
    if variant == "p2":
        pool_state = dctx * 4
    out["pool_state_bytes"] = pool_state
    return out


def full_report(n: int, k: int, d: int, variant: str, b: int, trial_bins: int, *, hidden=64,
                dctx=32, m=8, acc_bits=32, desc_bits=16, table_bits=8) -> dict:
    """One calibration of a session with N units, K latents, D behaviour dims, B trials."""
    apb = acc_per_bin(n, d)
    apt = acc_per_trial(n, d)
    fin = finalize_macs(n, k, d)
    fwd = forward_macs(variant, n, k, d, hidden=hidden, dctx=dctx, m=m)
    sr = sram_bytes(n, k, d, variant, acc_bits=acc_bits, desc_bits=desc_bits,
                    table_bits=table_bits, m=m, dctx=dctx)
    table_write = k * n
    stream_trial = trial_bins * apb["mac"] + apt["mac"]
    refresh = fin["mac"] + fwd["forward_macs"]
    return {
        "n_units": n, "k": k, "d": d, "variant": variant, "B": b, "trial_bins": trial_bins,
        "acc_mac_per_bin": apb["mac"], "acc_add_per_bin": apb["add"],
        "acc_mac_per_bin_symmetric_gram": acc_per_bin(n, d, True)["mac"],
        "acc_mac_per_trial_quad": apt["mac"],
        "acc_mac_all_B_trials": b * stream_trial,
        "finalize_mac": fin["mac"], "calib_forward_mac": fwd["forward_macs"],
        "forward_mac_per_unit": fwd["per_unit_total"], "pooling_mac": fwd["pooling_macs"],
        "table_write_words": table_write,
        "total_mac_one_calibration": b * stream_trial + refresh,
        "anytime_stream_mac_per_trial": stream_trial,
        "anytime_refresh_mac": refresh,
        "anytime_total_mac_per_trial": stream_trial + refresh,
        "inference_readin_mac_per_bin": k * n,
        "inference_readin_elementwise_per_bin": 2 * n,
        "refresh_in_readin_bins": refresh / (k * n),
        "n_calibrator_params": fwd["n_params"],
        "alt_latent_acc_mac_per_bin": alt_latent_accumulators_per_bin(n, k),
        "fixed_weight_fraction": fwd["forward_macs"] / max(b * stream_trial + refresh, 1),
        **sr,
    }
