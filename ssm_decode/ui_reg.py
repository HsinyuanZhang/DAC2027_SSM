"""Regularised gradient reference ``ui_reg`` (P2 item A'2).

NON-DEPLOYABLE reference: backprop through the frozen trunk onto the K x N
read-in table only, with a prior-centred L2 penalty

    loss = MSE(trunk(x_z T^T), y) + lam * ||T - T_center||_F^2 .

One Adam trajectory is run to the largest step budget and the table is
snapshotted at the requested step counts, so a step-budget sweep costs one
trajectory per (lam, lr).  The window schedule (one random labeled trial per
step, seeded) mirrors ``track_a_experiment.ui_grad_table``; with lam = 0 and
the same seed the trajectory is identical to that function.
"""
from __future__ import annotations

import numpy as np
import torch


def ui_reg_trajectory(model, x_z, y_norm, trial_windows, table_init, center, *, lam, lr,
                      snapshots, device, seed=0, valid_mask=None):
    """Return {step: table [K, N] float64} for every step in ``snapshots``."""
    snapshots = sorted(set(int(s) for s in snapshots))
    torch.manual_seed(seed)
    saved = [p.requires_grad for p in model.base.parameters()]
    for p in model.base.parameters():
        p.requires_grad_(False)
    table = torch.nn.Parameter(torch.from_numpy(np.asarray(table_init, np.float32)).clone().to(device))
    ctr = torch.from_numpy(np.asarray(center, np.float32)).to(device)
    x = torch.from_numpy(x_z.astype(np.float32)).to(device)
    y = torch.from_numpy(y_norm.astype(np.float32)).to(device)
    mask = torch.from_numpy(valid_mask).to(device) if valid_mask is not None else None
    rng = np.random.default_rng(seed)
    windows = [(a, b) for a, b in trial_windows if b - a >= 2]
    opt = torch.optim.Adam([table], lr=lr)
    out = {}
    try:
        for step in range(1, snapshots[-1] + 1):
            a, b = windows[int(rng.integers(len(windows)))]
            opt.zero_grad(set_to_none=True)
            decode = model.base((x[a:b] @ table.T).unsqueeze(0)).squeeze(0)
            target = y[a:b]
            good = torch.isfinite(target).all(-1)
            if mask is not None:
                good = good & mask[a:b]
            if good.any():
                loss = (decode[good] - target[good]).square().mean()
                loss = loss + lam * (table - ctr).square().sum()
                loss.backward()
                opt.step()
            if step in snapshots:
                out[step] = table.detach().cpu().numpy().astype(np.float64)
    finally:
        for p, state in zip(model.base.parameters(), saved):
            p.requires_grad_(state)
    return out
