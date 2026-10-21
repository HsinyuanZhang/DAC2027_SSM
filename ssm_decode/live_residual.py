"""S1 residuals from the live inference forward (H4).

L0 (``rebased_feedback_loop``) forwards each calibration trial from the zero trunk
state and zero-pads the FIR ring at the trial start. Deployment inference is one
contiguous causal pass: the trunk state is carried, and the read-in table switches
only at commit boundaries.

L1 uses that live trajectory as the residual source.

* Trial k (k > 0) is decoded under P0 of trials < k plus Delta of trials < k,
  the same table ``rebased_feedback_loop`` (mode ``rebase``) installs. Trial 0 is
  the documented one-trial stage: its own normalisation, prior table.
* Latency 0: a commit is installed at the start of the next usable trial. Bins
  between usable trials stay on the table of the trial just decoded. They are
  forwarded (the trunk state moves) but their residual is forced to 0 — they are
  not calibration labels — so they add nothing to U. Their normalised inputs are
  pushed onto the FIR ring and can be read as lags of a later labeled bin.
* The ring stores the normalised input xz actually presented to the trunk at that
  bin, in the coordinates of the map that was active then. Entries are not
  re-normalised when the map changes. This is an fp32 algorithm check, not a
  fixed-point simulation of the 3-bit count ring.
* The forward stops at the end of the last calibration trial. Later bins cannot
  change a residual or a commit.

L2 is the same live state with the ring cleared at each calibration-trial start,
so lagged inputs do not cross a trial boundary (gap bins included).

``eta = 0`` and ``rho = 0`` leave Delta at 0, so the committed table is the prefix
P0 and the scored contiguous decode is the P0 decode.
"""
from __future__ import annotations

import torch

from ssm_decode import stream_f7 as sf

# Documented for the README / hardware note. Not a second algorithm.
GAP_RING = (
    "Gap and pre-trial bins are decoded under the map installed at that time. "
    "Their residual is 0, so the FIR accumulator U does not gain a term from them. "
    "L1 appends their normalised xz to the ring; a later calibration bin can use "
    "those values as x_{t-l}. L2 clears the ring at the next trial start, so a gap "
    "never becomes a lag of the next trial. The trunk state is carried in both."
)


def calibration_pieces(spans, B: int, n_bins: int | None = None):
    """Half-open pieces of ``[0, end of calibration trial B)``.

    ``kind`` is ``cold`` (before the first usable trial), ``trial`` or ``gap``.
    ``trial_index`` is the 0-based calibration trial whose installed table covers
    the piece; cold has ``None``. The gap after trial k uses trial k's table,
    because latency 0 installs the new commit at the next usable trial.
    """
    spans = [(int(a), int(b)) for a, b in spans]
    B = int(B)
    if B < 1 or B > len(spans):
        raise ValueError(f"B={B} with {len(spans)} usable trials")
    used = spans[:B]
    for i, (a, b) in enumerate(used):
        if b <= a:
            raise ValueError(f"empty trial {i}")
        if i and a < used[i - 1][1]:
            raise ValueError("usable trials overlap or are out of order")
    if n_bins is not None and used[-1][1] > int(n_bins):
        raise ValueError("trial extends past n_bins")
    pieces = []
    if used[0][0] > 0:
        pieces.append((0, used[0][0], "cold", None))
    for k, (a, b) in enumerate(used):
        pieces.append((a, b, "trial", k))
        if k + 1 < B and used[k + 1][0] > b:
            pieces.append((b, used[k + 1][0], "gap", k))
    cursor = 0
    for start, end, _kind, _k in pieces:
        if start != cursor or end <= start:
            raise AssertionError(f"pieces do not cover the prefix: {pieces}")
        cursor = end
    if cursor != used[-1][1]:
        raise AssertionError("pieces stopped before the last calibration trial")
    return pieces


def _as_ring(ring: torch.Tensor | None, n_lags: int, n: int, like: torch.Tensor) -> torch.Tensor:
    width = max(int(n_lags) - 1, 0)
    if width == 0:
        return like.new_zeros(0, n)
    if ring is None:
        return like.new_zeros(width, n)
    if ring.shape[0] == width and ring.shape[1] == n:
        return ring
    raise ValueError(f"ring {tuple(ring.shape)} is not [{width}, {n}]")


def roll_ring(xz: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """Append ``xz`` [T, N] and keep the last ``ring.shape[0]`` rows (zeros if shorter)."""
    width, n = ring.shape
    if width == 0:
        return ring
    if xz.numel() == 0:
        return ring
    cat = torch.cat([ring, xz], 0)
    if cat.shape[0] >= width:
        return cat[-width:]
    out = xz.new_zeros(width, n)
    out[-cat.shape[0]:] = cat
    return out


def fir_increment(xz: torch.Tensor, r: torch.Tensor, jbar: torch.Tensor,
                  ring: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FIR data term of one chunk, reading ``ring`` as x before the chunk.

    ``xz`` is [T, N], ``r`` is [T, D] or [M, T, D] (already masked). Residuals on
    the ring bins are zero: this adds only the chunk's own terms, and lags that
    fall before the chunk use the ring instead of zero. A zero ring matches
    ``fir_data_term`` on the chunk alone.
    """
    if xz.dim() != 2:
        raise ValueError("xz must be [T, N]")
    single = r.dim() == 2
    if single:
        r = r.unsqueeze(0)
    if r.dim() != 3 or r.shape[1] != xz.shape[0]:
        raise ValueError(f"r {tuple(r.shape)} does not match xz {tuple(xz.shape)}")
    n_lags = int(jbar.shape[0])
    ring = _as_ring(ring, n_lags, xz.shape[1], xz)
    m = r.shape[0]
    u = xz.new_zeros(m, jbar.shape[2], xz.shape[1])
    if xz.shape[0] == 0 or float(r.abs().sum()) == 0.0:
        return (u[0] if single else u), roll_ring(xz, ring)
    xcat = torch.cat([ring, xz], 0) if ring.numel() else xz
    rcat = r if ring.numel() == 0 else torch.cat(
        [r.new_zeros(m, ring.shape[0], r.shape[-1]), r], 1)
    length = xcat.shape[0]
    for lag in range(n_lags):
        if lag >= length:
            break
        jb = jbar[lag].to(dtype=xz.dtype, device=xz.device)  # [D, K]
        if lag == 0:
            corr = torch.einsum("mtd,tn->mdn", rcat, xcat)
        else:
            corr = torch.einsum("mtd,tn->mdn", rcat[:, lag:], xcat[: length - lag])
        u = u + torch.einsum("dk,mdn->mkn", jb, corr)
    new_ring = roll_ring(xz, ring)
    return (u[0] if single else u), new_ring


def _forward_chunks(forward_fn, z: torch.Tensor, state, max_chunk: int):
    if z.shape[1] == 0:
        return z.new_zeros(z.shape[0], 0, 0), state
    outs = []
    for start in range(0, z.shape[1], int(max_chunk)):
        y, state = forward_fn(z[:, start:start + int(max_chunk)], state)
        outs.append(y)
    return torch.cat(outs, 1), state


def _mamba_forward(base):
    from ssm_decode.mamba3_streaming import forward_with_states

    def _fn(z, state):
        return forward_with_states(base, z, state, fp32_angles=False)

    return _fn


def _shift_batch(delta, bias, mean_old, std_old, mean_new, std_new):
    rows_d, rows_b = [], []
    for i in range(delta.shape[0]):
        d_i, b_i = sf.shift_correction(delta[i], bias[i], mean_old, std_old, mean_new, std_new)
        rows_d.append(d_i)
        rows_b.append(b_i)
    return torch.stack(rows_d, 0), torch.stack(rows_b, 0)


def live_rebased_many(base, x, y, valid, spans, prior, configs, prepare, jbar, B, *,
                      ring_mode: str = "carry", base_bias=None, forward_fn=None,
                      keep_yhat: bool = False, max_chunk: int = 4096, eps: float = sf.EPS_Q):
    """Batched L1/L2 rebased S1. ``configs`` is a list of ``(eta, rho)``.

    ``ring_mode='carry'`` is L1. ``ring_mode='reset'`` is L2 (ring cleared at each
    calibration-trial start; trunk state still carried). ``forward_fn`` is a test
    hook ``(z [M,T,K], state) -> (y [M,T,D], state)``; the default is the Mamba-3
    carry, which starts from the zero state when ``state`` is None.

    Returns one dict per config (same keys as ``rebased_feedback_loop``, plus
    ``eta`` / ``rho``). ``history`` on each dict is the per-trial snapshot list.
    With ``keep_yhat``, each dict also has ``yhat`` [T_prefix, D] for the bins
    that were forwarded (through the end of the last calibration trial).
    """
    if ring_mode not in ("carry", "reset"):
        raise ValueError(ring_mode)
    if not configs:
        raise ValueError("no configs")
    if jbar is None:
        raise ValueError("jbar is required")
    spans = [(int(a), int(b)) for a, b in spans]
    B = int(B)
    pieces_ok = calibration_pieces(spans, B, int(x.shape[0]))
    del pieces_ok
    device = x.device
    jbar = jbar.to(device=device, dtype=x.dtype)
    prior = prior.to(device=device, dtype=x.dtype)
    m = len(configs)
    etas = [float(e) for e, _r in configs]
    rhos = [float(r) for _e, r in configs]
    delta = prior.new_zeros((m,) + tuple(prior.shape))
    bias = prior.new_zeros((m, prior.shape[0]))
    coord = None
    n_lags = int(jbar.shape[0])
    ring = x.new_zeros(max(n_lags - 1, 0), x.shape[1])
    state = None
    yhat_out = [[] for _ in range(m)] if keep_yhat else None
    history = [[] for _ in range(m)]
    p0_final = None
    fwd = forward_fn if forward_fn is not None else _mamba_forward(base)

    def _maps(mean, std, base_table, d_use, b_use, extra):
        table = base_table.unsqueeze(0) + d_use
        b_forward = b_use if extra is None else b_use + extra.to(dtype=b_use.dtype, device=b_use.device)
        return table, b_forward

    def _decode(a, end, mean, std, table, b_forward, *, residual: bool):
        nonlocal state, ring
        if end <= a:
            return None
        xz = (x[a:end] - mean) / std.clamp_min(1e-8)
        z = torch.einsum("tn,mkn->mtk", xz, table.to(dtype=xz.dtype)) + b_forward.to(dtype=xz.dtype)[:, None, :]
        y_hat, state = _forward_chunks(fwd, z, state, max_chunk)
        if keep_yhat:
            for i in range(m):
                yhat_out[i].append(y_hat[i])
        if residual:
            return xz, y_hat
        if ring_mode == "carry":
            ring = roll_ring(xz, ring)
        return None

    with torch.no_grad():
        cursor = 0
        first = spans[0][0]
        if first > 0:
            # Cold map: electrode prior, mean 0, std 1. No label, no commit.
            cold_mean = x.new_zeros(x.shape[1])
            cold_std = x.new_ones(x.shape[1])
            cold_table = prior.unsqueeze(0).expand(m, -1, -1)
            cold_bias = bias.new_zeros(bias.shape)
            _decode(0, first, cold_mean, cold_std, cold_table, cold_bias, residual=False)
            cursor = first
        for k in range(B):
            a, b = spans[k]
            if a < cursor:
                raise AssertionError(f"trial {k} starts at {a}, before cursor {cursor}")
            if ring_mode == "reset":
                ring = x.new_zeros(ring.shape)
            if k == 0:
                mean, std, _p0 = prepare(0, False)
                base_table = prior
            else:
                mean, std, base_table = prepare(k, False)
            mean = mean.to(device=device, dtype=x.dtype)
            std = std.to(device=device, dtype=x.dtype)
            base_table = base_table.to(device=device, dtype=x.dtype)
            if coord is None:
                d_use, b_use = delta, bias
            else:
                d_use, b_use = _shift_batch(delta, bias, coord[0], coord[1], mean, std)
            extra = None
            if base_bias is not None:
                extra = base_bias(k, False)
            table, b_forward = _maps(mean, std, base_table, d_use, b_use, extra)
            got = _decode(a, b, mean, std, table, b_forward, residual=True)
            xz, y_hat = got
            y_t = y[a:b].to(dtype=y_hat.dtype)
            mask = valid[a:b].to(dtype=y_hat.dtype)
            r = (y_t.unsqueeze(0) - y_hat) * mask.view(1, -1, 1)
            n_bins, sum_x2 = sf.valid_sum_x2(xz, mask)
            u_all, ring = fir_increment(xz, r, jbar, ring)
            if u_all.dim() == 2:
                u_all = u_all.unsqueeze(0)
            mean_a, std_a, p0_a = prepare(k, True)
            mean_a = mean_a.to(device=device, dtype=x.dtype)
            std_a = std_a.to(device=device, dtype=x.dtype)
            p0_a = p0_a.to(device=device, dtype=x.dtype)
            new_delta, new_bias = [], []
            for i in range(m):
                if float(n_bins) > 0:
                    step = sf.feedback_step(u_all[i], sum_x2, n_bins, eps)
                else:
                    step = torch.zeros_like(d_use[i])
                d_c, b_c = sf.shift_correction(d_use[i], b_use[i], mean, std, mean_a, std_a)
                s_c, b_s = sf.shift_correction(step, torch.zeros_like(bias[i]), mean, std, mean_a, std_a)
                new_delta.append((1.0 - rhos[i]) * d_c + etas[i] * s_c)
                new_bias.append((1.0 - rhos[i]) * b_c + etas[i] * b_s)
            delta = torch.stack(new_delta, 0)
            bias = torch.stack(new_bias, 0)
            coord = (mean_a, std_a)
            p0_final = p0_a
            installed = p0_a.unsqueeze(0) + delta
            for i in range(m):
                history[i].append({
                    "k": int(k),
                    "table": installed[i].detach().clone(),
                    "bias": bias[i].detach().clone(),
                    "mean": mean_a.detach().clone(),
                    "std": std_a.detach().clone(),
                    "delta": delta[i].detach().clone(),
                    "base": p0_a.detach().clone(),
                })
            # Gap until the next calibration trial: same installed table as this
            # trial (the commit we just formed waits for the next trial start).
            # Residual stays 0. L1 keeps the ring; L2 will clear it at the next start.
            if k + 1 < B:
                g1 = spans[k + 1][0]
                if g1 > b:
                    _decode(b, g1, mean, std, table, b_forward, residual=False)
            cursor = spans[k + 1][0] if k + 1 < B else b

    if p0_final is None or coord is None:
        raise AssertionError("no calibration trial")
    table = p0_final.unsqueeze(0) + delta
    out = []
    for i in range(m):
        item = {"table": table[i], "bias": bias[i], "mean": coord[0], "std": coord[1],
                "delta": delta[i], "eta": etas[i], "rho": rhos[i], "history": history[i],
                "ring_mode": ring_mode, "prefix_end": int(spans[B - 1][1])}
        if keep_yhat:
            item["yhat"] = torch.cat(yhat_out[i], 0) if yhat_out[i] else x.new_zeros(0, y.shape[-1])
        out.append(item)
    return out


def live_rebased(base, x, y, valid, spans, prior, eta, rho, prepare, jbar, B, **kwargs):
    """One (eta, rho). Same return keys as one ``rebased_feedback_loop`` result, plus history."""
    return live_rebased_many(base, x, y, valid, spans, prior, [(eta, rho)], prepare, jbar, B, **kwargs)[0]
