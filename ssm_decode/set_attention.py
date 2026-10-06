"""Track E: set-attention read-in for variable unit sets (revision 4.7).

E1 folds a static-key attention pool into the same K x N table the AP
backbone consumes, so the deployment shape is identical: an SRAM table plus
the shared RRAM trunk.  The table is

    W[s, i] = alpha[s, i] * v(desc_i)[s],
    alpha = softmax_i( key(desc_i) . query_s / tau ),

where ``desc_i`` is a per-unit descriptor: association-profile rows against
behavior features, plus activity statistics.  Labels enter only the
descriptor.  Target sessions need no gradient; they recompute descriptors
and re-fold the table.

E2 re-folds with slowly updated descriptors.  E0 is the linear-pooling
limit (the plain AP table).  E4 (dynamic keys) is an accuracy ceiling and
is approximated by window-wise re-folding with activity-augmented
descriptors in the evaluator.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import association as assoc
from .mamba3_official import build_mamba3_official


def profile_rows(record, bins: np.ndarray, *, lags: int = 2) -> np.ndarray:
    """Behavior-lagged association profile rows ``[N, D*(lags+1)]``.

    Rows are zero when too few finite labeled bins exist in the window.
    """
    x = record.neural[bins].astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    safe = np.where(std < 1e-6, 1.0, std)
    x_z = (x - mean) / safe
    phi_full = assoc.behavior_features(record.behavior, lags=lags)
    phi = phi_full[bins]
    finite = np.isfinite(phi).all(1) & np.isfinite(record.behavior[bins]).all(1)
    if finite.sum() >= max(10, phi.shape[1] * 3):
        return assoc.supervised_profiles(phi[finite], x_z[finite], lam=1.0)["H"]
    return np.zeros((x.shape[1], phi.shape[1]))


def raw_unit_stats(record, bins: np.ndarray) -> np.ndarray:
    """Handcrafted per-unit activity statistics over selected bins ``[N, S]``.

    Columns: log mean rate, log std, log burst fraction (bins above one
    spike), lag-one autocorrelation, and quartile-wise rates across the
    window (4).  These feed the learned activity encoder (E0-style).
    """
    x = np.asarray(record.neural[bins], dtype=np.float64)
    mean = x.mean(0)
    std = x.std(0)
    burst = (x > 1.0).mean(0)
    centered = x - mean
    denom = (centered ** 2).sum(0)
    lag1 = np.where(denom > 1e-9, (centered[:-1] * centered[1:]).sum(0) / np.maximum(denom, 1e-9), 0.0)
    n = len(bins)
    edges = [int(n * f) for f in (0, .25, .5, .75, 1)]
    quartiles = [x[edges[i]:max(edges[i + 1], edges[i] + 1)].mean(0) for i in range(4)]
    stats = np.stack([np.log(mean + 0.05), np.log(std + 0.05), np.log(burst + 0.01),
                      np.clip(lag1, -1, 1)] + [np.log(q + 0.05) for q in quartiles], axis=1)
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def unit_descriptors(record, bins: np.ndarray, *, lags: int = 2,
                     with_profile: bool = True) -> tuple[np.ndarray, dict]:
    """Raw descriptor block ``[N, K0 + S]`` (profile rows plus activity stats).

    ``with_profile=False`` zeroes the profile rows.  Standardization and the
    learned activity encoder live in the backbone.
    """
    if with_profile:
        profile = profile_rows(record, bins, lags=lags)
    else:
        profile = np.zeros((record.neural.shape[1], record.behavior.shape[1] * (lags + 1)))
    stats = raw_unit_stats(record, bins)
    desc = np.concatenate([profile, stats], axis=1)
    return desc, {"k0": int(profile.shape[1]), "stats": int(stats.shape[1]),
                  "profile_used": bool(with_profile)}


def normalize_descriptors(descriptors: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """Standardize descriptor columns against the pooled source statistics."""
    pooled = np.concatenate(descriptors, axis=0)
    mean, std = pooled.mean(0), pooled.std(0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    return [(d - mean) / std_safe for d in descriptors], {"mean": mean, "std": std_safe}


class SetAttentionReadIn(nn.Module):
    """Static-key attention pooling that folds into a K x N table."""

    def __init__(self, desc_dim: int, k: int, *, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        self.k = k
        self.queries = nn.Parameter(0.1 * torch.randn(k, desc_dim))
        self.key_proj = nn.Linear(desc_dim, desc_dim)
        self.value_proj = nn.Linear(desc_dim, k)
        nn.init.zeros_(self.value_proj.bias)

    def fold_table(self, desc: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Fold the K x N table.  ``keep`` (bool, [N]) drops units inside the
        softmax, so attention renormalizes over the surviving units and the
        dropped columns come out exactly zero (review B1).
        """
        if desc.ndim != 2 or desc.shape[1] != self.queries.shape[1]:
            raise ValueError(f"desc must be [N, {self.queries.shape[1]}], got {tuple(desc.shape)}")
        keys = self.key_proj(desc)                                  # [N, d]
        logits = (keys @ self.queries.T) / self.tau                 # [N, K]
        if keep is not None:
            logits = logits.masked_fill(~keep[:, None], float("-inf"))
        alpha = logits.softmax(dim=0)                               # over units per slot
        values = self.value_proj(desc)                              # [N, K]
        return (alpha * values).T                                   # [K, N]


class SetAttentionBackbone(nn.Module):
    """E1 backbone: folded set read-in plus the shared Mamba-3 trunk.

    The descriptor is ``[profile rows | learned activity embedding]``.  The
    activity embedding comes from a small encoder over handcrafted per-unit
    statistics (E0-style).  Everything stays session-static, so the folded
    K x N table remains the deployment form.
    """

    def __init__(self, profile_dim: int, stats_dim: int, outputs: int, *, k: int = 16,
                 width: int = 128, layers: int = 2, state_size: int = 32,
                 dropout: float = 0.2, tau: float = 0.5, act_hidden: int = 16,
                 base=None, head=None):
        super().__init__()
        self.k, self.outputs = int(k), int(outputs)
        self.profile_dim, self.stats_dim = int(profile_dim), int(stats_dim)
        self.activity_encoder = nn.Sequential(
            nn.Linear(int(stats_dim), act_hidden), nn.GELU(),
            nn.Linear(act_hidden, act_hidden))
        desc_dim = int(profile_dim) + act_hidden
        self.read_in = SetAttentionReadIn(desc_dim, int(k), tau=tau)
        if base is None:
            self.base = build_mamba3_official(int(k), int(outputs), width, layers,
                                              state_size, dropout)
        else:
            self.base = base                                        # test seam
        self.head = nn.Linear(int(k), int(outputs)) if head is None else head
        if head is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def embed(self, profile: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Assemble the descriptor with the learned activity embedding."""
        if profile.shape[1] != self.profile_dim or stats.shape[1] != self.stats_dim:
            raise ValueError("descriptor block widths do not match the backbone")
        return torch.cat([profile, self.activity_encoder(stats)], dim=1)

    def fold_from_raw(self, profile: torch.Tensor, stats: torch.Tensor,
                      keep: torch.Tensor | None = None) -> torch.Tensor:
        return self.read_in.fold_table(self.embed(profile, stats), keep)

    def decode_with_table(self, x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        if table.ndim != 2 or table.shape[0] != self.k:
            raise ValueError(f"table must be [K={self.k}, N]")
        return self.base(x @ table.T)

    def load_ap_warm_start(self, ap_checkpoint: str) -> dict:
        """Warm start trunk and head from a converged AP backbone (review 4.1).

        Tables and the read-in have no AP counterpart and stay at their
        initialisation for the distillation phase.
        """
        blob = torch.load(ap_checkpoint, map_location="cpu", weights_only=False)
        state = blob["state_dict"]
        target = {k: v for k, v in state.items() if k.startswith(("base.", "head."))}
        missing, unexpected = self.load_state_dict(target, strict=False)
        if any(k.startswith(("base.", "head.")) for k in missing):
            raise ValueError(f"AP warm start missed trunk/head keys: {missing[:5]}")
        loaded = sum(1 for k in target if k.startswith("head."))
        if loaded == 0:
            raise ValueError("AP warm start loaded no head tensors (review B3)")
        return {"warm_started": sorted(target), "ap_best_source_val":
                blob.get("best_source_val_trunk_r2")}

    def whitening_penalty(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        good = mask.reshape(-1)
        zg = z.reshape(-1, z.shape[-1])[good]
        if zg.shape[0] < 2 * self.k:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        zc = zg - zg.mean(0)
        cov = (zc.T @ zc) / (zg.shape[0] - 1)
        eye = torch.eye(self.k, device=z.device, dtype=z.dtype)
        return ((cov - eye) ** 2).sum() / self.k


def load_set_backbone(path, *, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    args = blob["args"]
    model = SetAttentionBackbone(blob["profile_dim"], blob["stats_dim"], blob["outputs"],
                                 k=args["k"], width=args["width"], layers=args["layers"],
                                 state_size=args["state_size"], dropout=0.0,
                                 tau=args.get("tau", 0.5),
                                 act_hidden=args.get("act_hidden", 16))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob
