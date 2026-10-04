"""C8 AP-aware backbone: per-session K x N read-in tables, shared lift, head.

The decoder separates session-specific from session-invariant parts:

1. ``tables[s]`` is a ``[K, N_s]`` read-in table for source session ``s``.
2. ``base.in_proj`` is the shared lift ``W_lift`` from K latent dims into the
   backbone width.  The Mamba-3 trunk and ``out_proj`` are session-invariant.
3. ``head`` is the linear behavior head ``B_src`` on the K-dim latent.  Track A
   uses it as the frame anchor for closed-form tables on new sessions.

The latent contract for Track A: source latents ``z = W_s x`` are pushed
toward identity covariance by a whitening penalty, so an orthogonal frame
``T`` is enough to align a new session's factor-analysis latent.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .association import factor_analysis_em, posterior_readin
from .mamba3_official import OfficialMamba3Config, OfficialMamba3Decoder, build_mamba3_official


class APBackbone(nn.Module):
    def __init__(self, channel_counts, outputs, *, k=16, width=128, layers=2,
                 state_size=32, dropout=0.2, base=None):
        super().__init__()
        self.channel_counts = tuple(int(c) for c in channel_counts)
        self.k, self.outputs = int(k), int(outputs)
        if base is None:
            self.base = build_mamba3_official(int(k), int(outputs), width, layers, state_size, dropout)
        else:
            # Test seam: any module mapping [B, T, K] -> [B, T, outputs] works.
            self.base = base
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(int(k), int(c)) ) for c in self.channel_counts])
        self.head = nn.Linear(int(k), int(outputs))
        for table, count in zip(self.tables, self.channel_counts):
            bound = 1.0 / np.sqrt(max(count, 1))
            nn.init.uniform_(table, -bound, bound)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def config(self) -> OfficialMamba3Config:
        return self.base.config

    def latent(self, x: torch.Tensor, session_index: int) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channel_counts[session_index]:
            raise ValueError(f"x must be [B,T,{self.channel_counts[session_index]}], got {tuple(x.shape)}")
        return x @ self.tables[session_index].T

    def forward(self, x: torch.Tensor, session_index: int):
        z = self.latent(x, session_index)
        return self.base(z), self.head(z)

    def decode_with_table(self, x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Decode bins from an externally supplied table ``[K, N]``."""
        if table.ndim != 2 or table.shape[0] != self.k:
            raise ValueError(f"table must be [K={self.k}, N], got {tuple(table.shape)}")
        return self.base(x @ table.T)

    def whitening_penalty(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``||cov(z) - I||_F^2`` over masked bins of the batch."""
        good = mask.reshape(-1)
        zg = z.reshape(-1, z.shape[-1])[good]
        if zg.shape[0] < 2 * self.k:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        zc = zg - zg.mean(0)
        cov = (zc.T @ zc) / (zg.shape[0] - 1)
        eye = torch.eye(self.k, device=z.device, dtype=z.dtype)
        return ((cov - eye) ** 2).sum() / self.k

    def init_from_fa(self, session_windows, *, em_iters=200, seed=0) -> dict:
        """Initialize each table from the posterior read-in of a local FA fit.

        ``session_windows`` holds one ``[T, N_s]`` float tensor of normalized
        source counts per session.  The frame is arbitrary; gradient training
        adapts it.  Returns per-session diagnostics.
        """
        diagnostics = {}
        with torch.no_grad():
            for index, (table, window) in enumerate(zip(self.tables, session_windows)):
                x = window.detach().cpu().numpy().astype(np.float64)
                fa = factor_analysis_em(x, self.k, iters=em_iters, seed=seed + index)
                beta = posterior_readin(fa["H"], fa["psi"])
                table.copy_(torch.from_numpy(beta.astype(np.float32)))
                diagnostics[index] = {"fa_loglik": fa["loglik"], "fa_iters": fa["iters"],
                                      "psi_min": float(fa["psi"].min()), "psi_max": float(fa["psi"].max())}
        return diagnostics

    def init_from_supervised(self, session_windows, session_labels, *, ridge=10.0) -> dict:
        """Initialize tables from label-relevant directions plus PCA fill.

        Rows 0..D-1 span the ridge ``y ~ x`` directions.  The remaining rows
        take the top covariance directions orthogonal to them.  Every row is
        scaled to unit output variance.  Source sessions are labeled, so this
        initialization is legal for pretraining; Track A keeps the label-free
        requirement only for target tables.
        """
        diagnostics = {}
        with torch.no_grad():
            for index, (table, window, labels) in enumerate(zip(self.tables, session_windows, session_labels)):
                x = window.detach().cpu().numpy().astype(np.float64)
                y = labels.detach().cpu().numpy().astype(np.float64)
                good = np.isfinite(y).all(1)
                xg, yg = x[good], y[good]
                n, d = xg.shape[1], yg.shape[1]
                cov = (xg.T @ xg) / max(len(xg) - 1, 1)
                beta = np.linalg.solve(xg.T @ xg + ridge * np.eye(n), xg.T @ yg)   # [N, D]
                # Orthonormalize label directions, then complete with covariance directions.
                u_label, _ = np.linalg.qr(beta)
                evals, evecs = np.linalg.eigh(cov)
                order = np.argsort(evals)[::-1]
                rows = [u_label[:, j] for j in range(min(d, self.k))]
                for j in order:
                    if len(rows) >= self.k:
                        break
                    direction = evecs[:, j]
                    for existing in rows:
                        direction = direction - (existing @ direction) * existing
                    if np.linalg.norm(direction) > 1e-8:
                        rows.append(direction / np.linalg.norm(direction))
                matrix = np.stack(rows[:self.k])                                  # [K, N]
                variances = np.einsum("kn,nm,km->k", matrix, cov, matrix)
                scales = 1.0 / np.sqrt(np.maximum(variances, 1e-9))
                matrix = matrix * scales[:, None]
                table.copy_(torch.from_numpy(matrix.astype(np.float32)))
                diagnostics[index] = {"label_rank": int(min(d, self.k)),
                                      "min_row_variance": float(variances.min())}
        return diagnostics


def load_frozen_ap_backbone(path, *, device="cpu") -> tuple[APBackbone, dict]:
    """Load an exported AP backbone checkpoint."""
    blob = torch.load(path, map_location=device, weights_only=False)
    args = blob["args"]
    model = APBackbone(blob["channel_counts"], blob["outputs"], k=args["k"],
                       width=args["width"], layers=args["layers"],
                       state_size=args["state_size"], dropout=0.0)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob
