"""Continuous stateful carry and a single-bin streaming step for Mamba-3 SISO.

State-carry contract, verified from the pinned kernel source
(.tools/mamba_official at commit e9594ce, files mamba_ssm/ops/triton/mamba3/
mamba3_siso_combined.py, mamba3_siso_fwd.py, angle_dt.py, mamba3_siso_step.py):

1. ``mamba3_siso_combined(..., Input_States=(angle, ssm, k, v),
   return_final_states=True)`` takes and returns a 4-tuple.  All four entries
   must be given together or all None.  Layout, with ``H`` heads, ``P``
   headdim (=64), ``S`` d_state (=headdim_qk), ``A`` = S//4 rope angles, and
   one row per sequence (= batch in batched mode):
   - ``angle``: (B, H, A) fp32.  RoPE angle accumulator, the running
     ``mod 2pi`` cumsum of ``tanh(Angles) * pi * DT`` at the sequence start
     (angle_dt.py builds ``Angles_Cumsum`` from it).  Carry the returned final
     angle state or rotations jump at the boundary.
   - ``ssm``: (B, H, P, S) fp32.  Attention-like state, rows indexed by
     headdim_v, columns by headdim_qk.
   - ``k``: (B, H, S) fp32-valued.  Rotated K+bias of the LAST token of the
     previous chunk, BEFORE the ``scale = gamma + shifted_gamma`` multiply
     (mamba3_siso_fwd stores Final_K_State from the pre-scale block).
   - ``v``: (B, H, P).  V (=x) of the last token of the previous chunk.
2. On resume the kernel prepends one trapezoid correction,
   ``ssm += v_out * k_out * dt[0] * (1 - sigmoid(trap[0]))``, using dt and
   trap of the FIRST token of the new chunk.  This restores the
   ``shifted_gamma`` term that the previous call masked to zero at its
   sequence end, so chunked carry reproduces the unchunked recurrence.
3. With ``return_final_states=True`` the call returns
   ``(out, final_angle, final_ssm, final_k, final_v)``.  ``final_k`` is the
   rotated K+bias of the last token (pre-scale), picked at index
   ``(seqlen-1) % chunk_size`` of the stored last chunk.  ``final_v`` is
   ``V[:, -1]``.  ``out`` is (B, T, H, P) in the V dtype.
4. Boundary rule: pass the whole previous tuple unchanged.  Chunk lengths do
   not need to align with the kernel-internal chunk_size (masks handle the
   tail), but each boundary re-factors the last/first token contribution in
   bf16, which shows up as bf16-level noise (measured in
   results/analysis/c2_streaming_20261004).
5. The public wrapper casts Angles to bf16 before the angle cumsum, so the
   Final_Angle_State it returns is bf16 and the carry loses ~2^-8 relative
   angle precision per boundary.  ``forward_with_states`` therefore drives
   the same autograd Function (``_Mamba3Function.apply``, the exact code the
   public wrapper calls) directly with fp32 Angles and the wrapper's bf16
   casts applied to Q/K/V/Trap/Z, so the carried angle state is fp32-exact.
   ``fp32_angles=False`` selects plain public-wrapper semantics instead.

The pure-PyTorch ``step`` functions re-derive the official single-bin
recurrence (mamba3_siso_step.py) in float32 with the bf16 materialization
boundaries of ssm_decode/mamba3_cpu.py.  They share the carry-state layout
with the kernel path, so chunked kernel calls and single-bin steps can be
interleaved on one state tuple.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .mamba3_official import OfficialMamba3Decoder, _import_mamba3

TwoPi = 2.0 * math.pi
CoreState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
DecoderStates = List[CoreState]

_KERNEL = None


def _siso():
    """Import the pinned combined kernel through the isolated namespace."""
    global _KERNEL
    if _KERNEL is None:
        _import_mamba3()
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import _Mamba3Function, mamba3_siso_combined
        _KERNEL = (_Mamba3Function, mamba3_siso_combined)
    return _KERNEL


def _bf(x: torch.Tensor) -> torch.Tensor:
    """Match materialized BF16 boundaries in the fused SISO kernel."""
    return x.to(torch.bfloat16).float()


def _heavy_tail(x: torch.Tensor) -> torch.Tensor:
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight.float()


def _split_sizes(core) -> List[int]:
    return [core.d_inner, core.d_inner,
            core.d_state * core.num_bc_heads, core.d_state * core.num_bc_heads,
            core.nheads, core.nheads, core.nheads, core.num_rope_angles]


def _core_prepare(core, u: torch.Tensor):
    """Mirror the official Mamba3 SISO preprocessing (modules/mamba3.py)."""
    batch, length, _ = u.shape
    heads, headdim = core.nheads, core.headdim
    z, x, b, c, raw_dt, raw_a, trap, angles = torch.split(core.in_proj(u), _split_sizes(core), dim=-1)
    z = z.reshape(batch, length, heads, headdim)
    x = x.reshape(batch, length, heads, headdim)
    b = core.B_norm(b.reshape(batch, length, core.mimo_rank, core.num_bc_heads, core.d_state))
    c = core.C_norm(c.reshape(batch, length, core.mimo_rank, core.num_bc_heads, core.d_state))
    a = (-_heavy_tail(raw_a.float())).clamp(max=-core.A_floor)
    dt = F.softplus(raw_dt + core.dt_bias)
    angles = angles.unsqueeze(-2).expand(-1, -1, heads, -1)
    return (z, x, b.squeeze(2), c.squeeze(2), (a * dt).transpose(1, 2), dt.transpose(1, 2),
            trap.transpose(1, 2), angles)


def core_initial_states(core, batch: int, device=None, dtype=torch.float32) -> CoreState:
    """Zero carry states (angle, ssm, k, v) for one Mamba3 SISO core."""
    if device is None:
        device = core.in_proj.weight.device
    return (torch.zeros(batch, core.nheads, core.num_rope_angles, device=device, dtype=dtype),
            torch.zeros(batch, core.nheads, core.headdim, core.d_state, device=device, dtype=dtype),
            torch.zeros(batch, core.nheads, core.d_state, device=device, dtype=dtype),
            torch.zeros(batch, core.nheads, core.headdim, device=device, dtype=dtype))


def initial_states(decoder: OfficialMamba3Decoder, batch: int, device=None) -> DecoderStates:
    """Zero carry states for every block of an OfficialMamba3Decoder."""
    return [core_initial_states(block.ssm, batch, device) for block in decoder.blocks]


def core_forward_with_states(core, u: torch.Tensor, states: Optional[CoreState] = None,
                             fp32_angles: bool = True):
    """Run one Mamba3 SISO core on a chunk, carrying (angle, ssm, k, v).

    Returns ``(y, new_states)`` where ``y`` is the pre-out_proj fused output
    (batch, length, nheads, headdim) matching the module's kernel call.
    ``states=None`` starts from zero state.  Gradients flow through the
    kernel's autograd Function, including into the carried states.
    """
    if u.ndim != 3 or u.shape[-1] != core.d_model:
        raise ValueError(f"u must be [batch, length, {core.d_model}]")
    z, x, b, c, adt, dt, trap, angles = _core_prepare(core, u)
    carry = states if states is not None else (None, None, None, None)
    q_bias = core.C_bias.squeeze(1)
    k_bias = core.B_bias.squeeze(1)
    if fp32_angles:
        function, _ = _siso()
        out, angle, ssm, k, v = function.apply(
            c.to(torch.bfloat16), b.to(torch.bfloat16), x.to(torch.bfloat16),
            adt, dt, trap.to(torch.bfloat16), q_bias, k_bias,
            angles.float(), core.D, z.to(torch.bfloat16),
            carry[0], carry[1], carry[2], carry[3], None, core.chunk_size, True)
    else:
        _, combined = _siso()
        out, angle, ssm, k, v = combined(
            Q=c, K=b, V=x, ADT=adt, DT=dt, Trap=trap, Q_bias=q_bias, K_bias=k_bias,
            Angles=angles, D=core.D, Z=z, chunk_size=core.chunk_size,
            Input_States=carry if carry[0] is not None else None, return_final_states=True)
    return out, (angle, ssm, k, v)


def _block_forward_with_states(block: nn.Module, x: torch.Tensor, states: Optional[CoreState],
                               fp32_angles: bool):
    y, new_states = core_forward_with_states(block.ssm, block.norm1(x), states, fp32_angles)
    x = x + block.dropout(block.ssm.out_proj(y.flatten(-2).to(x.dtype)))
    return x + block.dropout(block.ffn(block.norm2(x))), new_states


def forward_with_states(decoder: OfficialMamba3Decoder, x: torch.Tensor,
                        states: Optional[DecoderStates] = None, fp32_angles: bool = True):
    """OfficialMamba3Decoder forward with state carry across calls.

    Mirrors ``decoder.forward`` block structure exactly (pre-LN residual +
    FFN, final norm, out_proj) while feeding each core's carry states through
    the kernel.  ``states=None`` or ``initial_states(decoder, batch)`` starts
    from zero state.  Returns ``(y, new_states)``.
    """
    if x.ndim != 3 or x.shape[-1] != decoder.config.input_size:
        raise ValueError(f"x must be [batch, length, {decoder.config.input_size}]")
    if states is None:
        states = initial_states(decoder, x.shape[0], device=x.device.type)
    if len(states) != decoder.config.layers:
        raise ValueError("states must contain one tuple per block")
    h = decoder.in_proj(x)
    new_states: DecoderStates = []
    for block, block_states in zip(decoder.blocks, states):
        h, block_new = _block_forward_with_states(block, h, block_states, fp32_angles)
        new_states.append(block_new)
    return decoder.out_proj(decoder.final_norm(h)), new_states


def stream_recording(decoder: OfficialMamba3Decoder, x: torch.Tensor,
                     chunk_lens: Sequence[int], fp32_angles: bool = True) -> torch.Tensor:
    """Run one recording as consecutive carried chunks of the given lengths."""
    if sum(chunk_lens) != x.shape[1]:
        raise ValueError("chunk_lens must sum to the sequence length")
    states = initial_states(decoder, x.shape[0], device=x.device.type)
    outputs = []
    position = 0
    for length in chunk_lens:
        y, states = forward_with_states(decoder, x[:, position:position + length], states, fp32_angles)
        outputs.append(y)
        position += length
    return torch.cat(outputs, dim=1)


def _rotate(pre: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent element pairs (2i, 2i+1) by theta (..., A) radians."""
    count = theta.shape[-1]
    pairs = pre[..., :2 * count].reshape(*pre.shape[:-1], count, 2)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    first = pairs[..., 0] * cosine - pairs[..., 1] * sine
    second = pairs[..., 0] * sine + pairs[..., 1] * cosine
    rotated = torch.stack((first, second), dim=-1).flatten(-2)
    return torch.cat((rotated, pre[..., 2 * count:]), dim=-1)


def core_step(core, u: torch.Tensor, states: CoreState):
    """Single-bin pure-PyTorch step for one Mamba3 SISO core.

    ``u`` is (batch, d_model) for one time bin.  Follows the pinned
    mamba3_siso_step kernel semantics with the bf16 boundaries of
    mamba3_cpu.py: theta = mod 2pi(angle + tanh(bf16(angles)) * pi * dt),
    alpha = exp(adt), beta = alpha * dt * (1 - sigmoid(bf16(trap))),
    gamma = dt * sigmoid(bf16(trap)),
    ssm' = alpha * ssm + beta * v_prev (x) k_prev + gamma * v (x) k_rot,
    y = ssm' @ q_rot + D * v, gated by silu(bf16(z)) and out_proj'd.
    Returns ``(out, new_states)`` with out (batch, d_model).
    """
    if u.ndim != 2 or u.shape[-1] != core.d_model:
        raise ValueError(f"u must be [batch, {core.d_model}]")
    batch = u.shape[0]
    heads, headdim = core.nheads, core.headdim
    z, x, b, c, raw_dt, raw_a, trap, angles = torch.split(core.in_proj(u), _split_sizes(core), dim=-1)
    z = _bf(z).reshape(batch, heads, headdim)
    x = _bf(x).reshape(batch, heads, headdim)
    # One BC head normalized per token, broadcast to every head; Q/K biases
    # are per head (kernel adds Q_bias/K_bias indexed by pid_head).
    k_pre = _bf(_rms(b, core.B_norm.weight)).unsqueeze(1) + core.B_bias.squeeze(1).float()[None]
    q_pre = _bf(_rms(c, core.C_norm.weight)).unsqueeze(1) + core.C_bias.squeeze(1).float()[None]
    k_pre = k_pre.expand(-1, heads, -1)
    q_pre = q_pre.expand(-1, heads, -1)
    dt = F.softplus(raw_dt.float() + core.dt_bias.float())
    adt = (-_heavy_tail(raw_a.float())).clamp(max=-core.A_floor) * dt
    gate = torch.sigmoid(_bf(trap))
    gamma = dt * gate
    angle, ssm, k_prev, v_prev = states
    d_theta = torch.tanh(_bf(angles)).unsqueeze(1) * math.pi * dt[..., None]
    theta = torch.remainder(angle + d_theta, TwoPi)
    q_rot = _bf(_rotate(q_pre, theta))
    k_rot = _bf(_rotate(k_pre, theta))
    alpha = torch.exp(adt)
    beta = alpha * dt * (1 - gate)
    ssm = (alpha[..., None, None] * ssm
           + beta[..., None, None] * v_prev[..., None] * k_prev[..., None, :]
           + gamma[..., None, None] * x[..., None] * k_rot[..., None, :])
    out = torch.einsum("bhps,bhs->bhp", _bf(ssm), q_rot)
    out = out + core.D.float()[None, :, None] * x
    out = _bf(out * F.silu(z))
    y = core.out_proj(out.flatten(-2).to(u.dtype))
    return y, (theta, ssm, k_rot, x)


def decoder_step(decoder: OfficialMamba3Decoder, u: torch.Tensor, states: DecoderStates):
    """Single-bin pure-PyTorch step for the full OfficialMamba3Decoder.

    ``u`` is (batch, input_size).  Mirrors decoder.forward per bin: in_proj,
    pre-LN residual + core step + FFN per block, final norm, out_proj.
    Returns ``(y, new_states)`` with y (batch, output_size).
    """
    if u.ndim != 2 or u.shape[-1] != decoder.config.input_size:
        raise ValueError(f"u must be [batch, {decoder.config.input_size}]")
    if len(states) != decoder.config.layers:
        raise ValueError("states must contain one tuple per block")
    h = decoder.in_proj(u)
    new_states: DecoderStates = []
    for block, block_states in zip(decoder.blocks, states):
        y, block_new = core_step(block.ssm, block.norm1(h), block_states)
        h = h + block.dropout(y)
        new_states.append(block_new)
        h = h + block.dropout(block.ffn(block.norm2(h)))
    return decoder.out_proj(decoder.final_norm(h)), new_states


def streaming_reference(decoder: OfficialMamba3Decoder, x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch streaming decode of a full recording, bin by bin."""
    states = initial_states(decoder, x.shape[0], device=x.device.type)
    outputs = []
    for position in range(x.shape[1]):
        y, states = decoder_step(decoder, x[:, position], states)
        outputs.append(y)
    return torch.stack(outputs, dim=1)
