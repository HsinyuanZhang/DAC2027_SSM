"""Calibration-path quantisation / approximate computing (P2 item A'3).

Everything here is *evaluation-time* fake quantisation of the calibration
path of ``population_calib``; nothing is retrained.

  * weights: symmetric per-output-channel int-b (b = 8, 4) and ternary TWN
    (threshold 0.7 * mean|w|, per-output-channel scale = mean |w| over the kept
    entries) - the fixed-weight calibrator held in 2T2R RRAM.
  * accumulators: signed fixed point, ``bits`` total bits, a power-of-two step
    per accumulator key chosen from SOURCE episodes only, round-to-nearest
    after every trial update, saturating.
  * finalised descriptors: same fixed point, one step per feature column.
  * output table: symmetric int-b with one scale per table row (latent k).
  * softmax: exp2 via a 256-entry x 16-bit LUT plus a reciprocal.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

LOG2E = 1.4426950408889634

ACC_GROUPS = {
    "norm": ("n", "sx", "gram"),
    "ac": ("ac_xx", "ac_a", "ac_b", "ac_n"),
    "cross": ("sy", "syx", "yx_n"),
    "dir": ("sdir", "cdir", "sphi", "sphiphi", "sphix"),
    "quad": ("q_sx2", "q_yx2", "q_yxy", "q_y2"),
}


# ----------------------------------------------------------------- weights ----

def qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def quant_sym_rows(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric int-``bits`` fake quantisation, one scale per row (output channel)."""
    amax = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax(bits)
    return torch.round(w / scale).clamp(-qmax(bits), qmax(bits)) * scale


def quant_ternary_rows(w: torch.Tensor) -> torch.Tensor:
    """TWN ternary: delta = 0.7 mean|w|, alpha = mean of |w| above delta, per row."""
    aw = w.abs()
    delta = 0.7 * aw.mean(1, keepdim=True)
    mask = (aw > delta).to(w.dtype)
    alpha = (aw * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1.0)
    return alpha * torch.sign(w) * mask


def quant_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "w8":
        return quant_sym_rows(w, 8)
    if mode == "w4":
        return quant_sym_rows(w, 4)
    if mode == "tern":
        return quant_ternary_rows(w)
    raise ValueError(mode)


def quantize_calibrator(calib: nn.Module, mode: str) -> nn.Module:
    """Copy of ``calib`` whose matrix weights (nn.Linear weights and the PMA seed
    matrices u, a) are fake-quantised per output channel.  Biases, the learned
    shrinkage constant mu0 and the frozen input normalisers stay FP (digital)."""
    q = copy.deepcopy(calib)
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(quant_weight(mod.weight, mode))
        for name in ("u", "a"):
            if hasattr(q, name) and isinstance(getattr(q, name), nn.Parameter):
                getattr(q, name).copy_(quant_weight(getattr(q, name), mode))
    return q


class QLinear(nn.Linear):
    """nn.Linear whose weight is fake-quantised in the forward pass (straight-through
    estimator for the backward pass).  Used only for source-side QAT fine-tuning."""
    qat_mode = "tern"

    def forward(self, x):
        w = self.weight
        wq = w + (quant_weight(w, self.qat_mode) - w).detach()
        return nn.functional.linear(x, wq, self.bias)


def enable_qat(calib: nn.Module, mode: str) -> nn.Module:
    """Switch every nn.Linear of ``calib`` (and u, a of P3/P4) to STE fake quantisation
    in place.  The state_dict layout is unchanged (latent FP weights are stored)."""
    for mod in calib.modules():
        if type(mod) is nn.Linear:
            mod.__class__ = QLinear
            mod.qat_mode = mode

    def ste(w):
        return w + (quant_weight(w, mode) - w).detach()
    calib.weight_quant = ste
    return calib


# ------------------------------------------------------------ fixed point ----

def fixed_step(maxabs: float, bits: int) -> float:
    """Smallest power-of-two step such that ``maxabs`` fits in ``bits`` signed bits."""
    maxabs = max(float(maxabs), 1e-30)
    return 2.0 ** math.ceil(math.log2(maxabs / qmax(bits)))


def to_fixed(x: torch.Tensor, step, bits: int) -> torch.Tensor:
    """Round to the fixed-point grid ``step`` and saturate at +-qmax*step."""
    q = qmax(bits)
    return torch.round(x / step).clamp(-q, q) * step


def quant_table(table: torch.Tensor, bits: int) -> torch.Tensor:
    """Output table [K, N]: symmetric int-``bits`` with one scale per row k."""
    return quant_sym_rows(table, bits)


def acc_steps(stats_list, bits: int) -> dict:
    """Per-key power-of-two steps from SOURCE-episode PopStats (max |value|)."""
    amax: dict = {}
    for st in stats_list:
        for k, v in st.d.items():
            amax[k] = max(amax.get(k, 0.0), float(v.abs().max()))
    return {k: fixed_step(a, bits) for k, a in amax.items()}


def quantized_stats(bank, idx, steps: dict, bits: int, keys=None):
    """Stream the chosen trials into fixed-point accumulators (round after each add)."""
    from ssm_decode.population_calib import PopStats
    acc = None
    for j in idx:
        one = PopStats.from_trials(bank, [int(j)])
        acc = one if acc is None else acc + one
        for k in (keys if keys is not None else acc.d):
            acc.d[k] = to_fixed(acc.d[k], steps[k], bits)
    return acc


def column_steps(mats, bits: int) -> torch.Tensor:
    """One power-of-two step per column from SOURCE descriptor matrices [N, F]."""
    amax = torch.cat([m.abs() for m in mats], 0).amax(0)
    return torch.tensor([fixed_step(a, bits) for a in amax.tolist()], dtype=mats[0].dtype,
                        device=mats[0].device)


# --------------------------------------------------------------- softmax ----

class Exp2LUT:
    """exp(x) for x <= 0 as 2^floor(t) * LUT[frac(t)], t = x*log2(e).

    LUT: ``entries`` unsigned 16-bit codes (15 fractional bits) holding
    2^((j + 0.5)/entries) in [1, 2); mid-point entries bound the relative
    error at 2^(0.5/entries) - 1 (0.135 % for 256).  The result is rounded to
    16 fractional bits (an exp in [0, 1] stored as a 16-bit fraction).
    """

    def __init__(self, entries: int = 256, frac_bits: int = 15, out_bits: int = 16):
        self.entries, self.out_bits = entries, out_bits
        j = torch.arange(entries, dtype=torch.float64)
        codes = torch.round(2.0 ** ((j + 0.5) / entries) * 2 ** frac_bits)
        assert float(codes.max()) < 2 ** 16, "LUT code must fit 16 bits"
        self.codes = codes.to(torch.int64)
        self.table = codes / 2 ** frac_bits

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        t = x.double() * LOG2E
        i = torch.floor(t)
        f = t - i
        idx = torch.floor(f * self.entries).clamp(0, self.entries - 1).long()
        val = self.table.to(x.device)[idx] * torch.exp2(i)
        out = torch.round(val * 2 ** self.out_bits) / 2 ** self.out_bits
        return out.to(x.dtype)


def lut_reciprocal(s: torch.Tensor, entries: int = 256) -> torch.Tensor:
    """1/s from a mantissa LUT: s = m 2^e, m in [1, 2); 16-bit mid-point entries."""
    e = torch.floor(torch.log2(s.double()))
    m = s.double() / torch.exp2(e)
    idx = torch.floor((m - 1.0) * entries).clamp(0, entries - 1)
    inv_m = torch.round(65536.0 / (1.0 + (idx + 0.5) / entries)) / 65536.0
    return (inv_m * torch.exp2(-e)).to(s.dtype)


def make_lut_softmax(recip: str = "exact", entries: int = 256):
    lut = Exp2LUT(entries)

    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        e = lut.exp(x - x.amax(dim, keepdim=True))
        s = e.sum(dim, keepdim=True)
        inv = lut_reciprocal(s, entries) if recip == "lut" else 1.0 / s
        return e * inv
    return softmax
