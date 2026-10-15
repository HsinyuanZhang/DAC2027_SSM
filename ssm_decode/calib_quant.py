"""Calibration-path quantisation / approximate computing (P2 item A'3).

Everything here is *evaluation-time* fake quantisation of the calibration
path of ``population_calib``; nothing is retrained.

  * weights: symmetric per-output-channel int-b (b = 8, 4) and ternary TWN
    (threshold 0.7 * mean|w|, per-output-channel scale = mean |w| over the kept
    entries) - the fixed-weight calibrator held in 2T2R RRAM.
  * accumulators (data keys): signed fixed point, ``bits`` total bits, a power-of-two
    step per accumulator key chosen from SOURCE episodes only.  Default model
    (``perbin_stats``): every per-bin increment is rounded to the register step
    (round-to-nearest, ties-to-even) and added with saturation, i.e. rounding and
    saturation happen on EVERY add, like a real accumulator register.  The old
    "two-level design (wide per-trial staging + narrow running sum)" is kept as
    ``twolevel_stats`` (rounds only once per trial).
  * bin counters (``COUNTER_KEYS``: n, ac_n, yx_n, cdir) are exact unsigned integer
    counters of ``counter_bits`` width; they are never quantised like data.
  * redundant keys are DERIVED, not accumulated (``_set_derived``): sphi and sphiphi are
    exact functions of the 8 direction counters cdir (a constant 8x3 cos/sin table), and
    sphix[0] (the constant-1 row) equals sx.  They need no accumulator and no step.
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

# ACC_GROUPS is kept for backward compatibility (it still lists the counters); the
# hardware conclusions use DATA_GROUPS, which excludes them.
ACC_GROUPS = {
    "norm": ("n", "sx", "gram"),
    "ac": ("ac_xx", "ac_a", "ac_b", "ac_n"),
    "cross": ("sy", "syx", "yx_n"),
    "dir": ("sdir", "cdir", "sphi", "sphiphi", "sphix"),
    "quad": ("q_sx2", "q_yx2", "q_yxy", "q_y2"),
}
COUNTER_KEYS = ("n", "ac_n", "yx_n", "cdir")      # exact integer counters, never quantised
# keys that are exact functions of other registers (see _set_derived): never accumulated
DERIVED_KEYS = ("sphi", "sphiphi")
NON_DATA_KEYS = COUNTER_KEYS + DERIVED_KEYS
DATA_GROUPS = {g: tuple(k for k in ks if k not in NON_DATA_KEYS) for g, ks in ACC_GROUPS.items()}
ALL_DATA_KEYS = tuple(k for ks in DATA_GROUPS.values() for k in ks)


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
            if k == "sphix":                 # row 0 (constant-1 feature) is shared with sx
                v = v[1:]
            amax[k] = max(amax.get(k, 0.0), float(v.abs().max()))
    return {k: fixed_step(a, bits) for k, a in amax.items()}


def counter_bits(n_trials: int, max_trial_bins: int) -> int:
    """Unsigned width of the exact bin counters (n, ac_n, yx_n, cdir, sphi[0]):
    they count at most ``n_trials * max_trial_bins`` bins."""
    return int(math.ceil(math.log2(n_trials * max_trial_bins + 1)))


def quantized_stats(bank, idx, steps: dict, bits: int, keys=None):
    """LEGACY (superseded) two-level model: exact float64 per-trial sums, only the
    cross-trial running sum is rounded (once per trial).  ``keys=None`` quantises
    EVERY key including the bin counters, which is a modelling error; use
    ``perbin_stats`` (default hardware model) or ``twolevel_stats``.  Kept so the
    superseded numbers stay reproducible."""
    from ssm_decode.population_calib import PopStats
    acc = None
    for j in idx:
        one = PopStats.from_trials(bank, [int(j)])
        acc = one if acc is None else acc + one
        for k in (keys if keys is not None else acc.d):
            acc.d[k] = to_fixed(acc.d[k], steps[k], bits)
    return acc


def _set_derived(d: dict) -> None:
    """sphi / sphiphi from the exact direction counters, sphix[0] from sx (in place)."""
    from ssm_decode.population_calib import DIR_BINS
    ang = (torch.arange(DIR_BINS, dtype=torch.float64) + 0.5) * (2 * math.pi / DIR_BINS)
    base = torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang)], 1).to(d["cdir"])
    d["sphi"] = d["cdir"] @ base
    d["sphiphi"] = base.T @ (d["cdir"][:, None] * base)
    d["sphix"] = d["sphix"].clone()
    d["sphix"][0] = d["sx"]


_set_counter_aliases = _set_derived          # old name


def twolevel_stats(bank, idx, steps: dict, bits: int, keys=None, *, alias_counters: bool = True):
    """Two-level design: a WIDE per-trial staging buffer (modelled as exact float64;
    see ``calib_cost`` for its width/cost) collects each trial; at trial end it is added
    into the NARROW running sum, which is rounded to the register step and saturated
    once per trial.  Counters stay exact.  Equals the old per-trial behaviour on the
    data keys."""
    from ssm_decode.population_calib import PopStats
    keys = tuple(k for k in (keys if keys is not None else ALL_DATA_KEYS) if k not in NON_DATA_KEYS)
    exact = PopStats.from_trials(bank, idx)
    out = quantized_stats(bank, idx, steps, bits, keys)
    for k in COUNTER_KEYS:
        out.d[k] = exact.d[k]
    if alias_counters:
        _set_derived(out.d)
    return out


# ------------------------------------------------- per-add fixed-point model ----
#
# Register model (one per data key): signed ``bits`` two's-complement integer R in
# [-qmax, qmax] (the code -2^(bits-1) is not used, so the range is symmetric), real
# value R * step with a power-of-two step.  Every bin, the datapath produces an
# increment (a product or a selected input, exact in a wide datapath); it is rounded
# to the register step (round-to-nearest, ties-to-even) and added to R; the sum
# saturates at +-qmax (no wrap-around); each clipped add is counted in
# ``sat_events``.  Increments of lag pairs are attributed to the bin at which the
# second member of the pair arrives (delay lines hold the earlier member, reset at
# every trial boundary).  q_* keys use a per-trial scratch (per-add rounded sx, syx
# lag 0, sy lag 0 at the steps of those keys) and, at trial end, one rounded
# saturating add of sx^2/n, ... into the q register.

def ang_bins_of(sd):
    """Direction-bin id of every time bin of a SessionData (cached on the object)."""
    c = getattr(sd, "_pq_ang", None)
    if c is None:
        from ssm_decode.population_calib import DIR_BINS
        y = (sd.record.behavior if hasattr(sd, "record") else sd.behavior)
        import numpy as np
        y = np.asarray(y, dtype=np.float64)
        a = (np.arctan2(y[:, 1], y[:, 0]) + 2 * math.pi) % (2 * math.pi)
        c = torch.from_numpy(np.floor(a / (2 * math.pi / DIR_BINS)).astype(np.int64) % DIR_BINS)
        sd._pq_ang = c
    return c


def bin_increments(x, y, ang_bin, valid, t0: int, t1: int, want=None) -> dict:
    """Per-bin increments for bins [t0, t1) of ONE trial segment (x [T, N] ... as in
    ``segment_sums``).  ``sum over all bins`` of each key equals ``segment_sums``."""
    from ssm_decode.population_calib import AC_LAGS, CROSS_LAGS, DIR_BINS
    want = set(want) if want is not None else None
    wants = (lambda k: want is None or k in want)
    dt = x.dtype
    v = valid.to(dt)
    y = torch.where(valid.unsqueeze(1), y, torch.zeros_like(y))
    t = torch.arange(t0, t1)
    c = t1 - t0
    xv, vv_t = x[t0:t1], v[t0:t1]
    n_units, d = x.shape[1], y.shape[1]
    out = {}
    if wants("n"):
        out["n"] = vv_t
    if wants("sx"):
        out["sx"] = vv_t.unsqueeze(1) * xv
    if wants("gram"):
        out["gram"] = torch.einsum("tn,tm->tnm", xv, vv_t.unsqueeze(1) * xv)
    if any(wants(k) for k in ("ac_xx", "ac_a", "ac_b", "ac_n")):
        axx, aa, ab, an = [], [], [], []
        for k in AC_LAGS:
            ok = (t >= k).to(dt)
            tl = (t - k).clamp_min(0)
            vv = vv_t * v[tl] * ok
            axx.append(vv.unsqueeze(1) * xv * x[tl])
            aa.append(vv.unsqueeze(1) * xv)
            ab.append(vv.unsqueeze(1) * x[tl])
            an.append(vv)
        out["ac_xx"], out["ac_a"], out["ac_b"] = (torch.stack(axx, 1), torch.stack(aa, 1),
                                                  torch.stack(ab, 1))
        out["ac_n"] = torch.stack(an, 1)
    if any(wants(k) for k in ("sy", "syx", "yx_n")):
        sy, syx, yn = [], [], []
        for lag in CROSS_LAGS:                   # pairs (x_s, y_{s+lag}) -> bin max(s, s+lag)
            if lag >= 0:
                s_ix = (t - lag).clamp_min(0)
                ok = (t >= lag).to(dt)
                vv = v[s_ix] * vv_t * ok
                yt, xt = y[t0:t1], x[s_ix]
            else:
                m = -lag
                s_ix = (t - m).clamp_min(0)
                ok = (t >= m).to(dt)
                vv = vv_t * v[s_ix] * ok
                yt, xt = y[s_ix], xv
            sy.append(vv.unsqueeze(1) * yt)
            syx.append(torch.einsum("td,tn->tdn", vv.unsqueeze(1) * yt, xt))
            yn.append(vv)
        out["sy"], out["syx"], out["yx_n"] = torch.stack(sy, 1), torch.stack(syx, 1), torch.stack(yn, 1)
    if any(wants(k) for k in ("sdir", "cdir", "sphi", "sphiphi", "sphix")):
        onehot = torch.nn.functional.one_hot(ang_bin[t0:t1].long(), DIR_BINS).to(dt) * vv_t.unsqueeze(1)
        out["sdir"] = torch.einsum("tn,tb->tnb", xv, onehot)
        out["cdir"] = onehot
        ang = (ang_bin[t0:t1].to(dt) + 0.5) * (2 * math.pi / DIR_BINS)
        base = torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang)], 1)
        phi = base * vv_t.unsqueeze(1)
        out["sphi"] = phi
        out["sphiphi"] = torch.einsum("ta,tb->tab", phi, base)
        out["sphix"] = torch.einsum("ta,tn->tan", phi, xv)
    return out


def _sat_cumsum(state: torch.Tensor, ints: torch.Tensor, q: int):
    """Saturating running sum of integer increments ints [c, ...] from ``state``:
    s <- clamp(s + inc, -q, q) bin by bin.  Fast path (no clip anywhere) = cumsum."""
    cs = state + torch.cumsum(ints, 0)
    if float(cs.abs().max()) <= q:
        return cs[-1], 0
    s, ev = state, 0
    for i in range(ints.shape[0]):
        s2 = s + ints[i]
        ev += int((s2.abs() > q).sum())
        s = s2.clamp(-q, q)
    return s, ev


class _FxConfig:
    """One accumulator configuration run in lock-step on shared increments."""

    def __init__(self, steps, bits, keys, ref_shapes):
        self.steps, self.bits, self.q = steps, bits, qmax(bits)
        self.keys = tuple(k for k in (keys if keys is not None else ALL_DATA_KEYS)
                          if k not in NON_DATA_KEYS)
        self.state = {k: torch.zeros(ref_shapes[k], dtype=torch.float64) for k in self.keys
                      if not k.startswith("q_")}
        self.sat = {k: 0 for k in self.keys}
        self.use_q = any(k.startswith("q_") for k in self.keys)

    def add_chunk(self, key, inc):
        if key == "sphix":                         # row 0 is shared with sx, not accumulated
            inc = inc.clone()
            inc[:, 0] = 0.0
        if key in self.state:
            ints = torch.round(inc / self.steps[key])
            self.state[key], ev = _sat_cumsum(self.state[key], ints, self.q)
            self.sat[key] += ev


def perbin_stats(sd, idx, steps: dict, bits: int, keys=None, *, chunk: int = 256,
                 return_info: bool = False, alias_counters: bool = True):
    """Faithful per-add fixed-point accumulation of the chosen trials (see the register
    model above).  ``keys`` = data keys that live in fixed point (default: all data
    keys); every other key, and all bin counters, are exact.  Returns PopStats
    (and ``{"sat_events": {key: n}}`` if ``return_info``)."""
    res = perbin_stats_multi(sd, idx, {"_": (steps, bits, keys)}, chunk=chunk,
                             alias_counters=alias_counters)["_"]
    return (res if return_info else res[0])


def perbin_stats_multi(sd, idx, cfgs: dict, *, chunk: int = 256, alias_counters: bool = True) -> dict:
    """Run several accumulator configs {name: (steps, bits, keys)} over the same trials,
    sharing the (expensive) increment computation.  Returns {name: (PopStats, info)}."""
    from ssm_decode.population_calib import LAG0, PopStats
    exact = PopStats.from_trials(sd.bank, idx)
    shapes = {k: tuple(v.shape) for k, v in exact.d.items()}
    ang = ang_bins_of(sd)
    cache = getattr(sd, "_pq_cpu", None)
    if cache is None:
        cache = sd._pq_cpu = {"x": sd.x.detach().cpu(), "y": sd.y_norm.detach().cpu(),
                              "v": sd.valid.detach().cpu()}
    fx = {name: _FxConfig(*c[:2], c[2], shapes) for name, c in cfgs.items()}
    qstate = {name: {k: torch.zeros(shapes[k], dtype=torch.float64) for k in f.keys
                     if k.startswith("q_")} for name, f in fx.items()}
    want = set()
    for f in fx.values():
        want |= set(k for k in f.state)
        if f.use_q:
            want |= {"sx", "syx", "sy"}
    for j in idx:
        a, b = sd.trials[int(j)]
        x, y, v, ab = cache["x"][a:b], cache["y"][a:b], cache["v"][a:b], ang[a:b]
        scratch = {name: None for name in fx}
        n_tr = float(v.sum())
        for t0 in range(0, b - a, chunk):
            t1 = min(t0 + chunk, b - a)
            inc = bin_increments(x, y, ab, v, t0, t1, want)
            for name, f in fx.items():
                for k in f.state:
                    f.add_chunk(k, inc[k])
                if f.use_q:                       # per-trial scratch (per-add rounded)
                    sc = scratch[name] or {k: None for k in ("sx", "syx", "sy")}
                    for k in ("sx", "syx", "sy"):
                        arr = inc[k] if k == "sx" else inc[k][:, LAG0]
                        ints = torch.round(arr / f.steps[k])
                        s0 = sc[k] if sc[k] is not None else torch.zeros(ints.shape[1:],
                                                                         dtype=torch.float64)
                        sc[k], _ = _sat_cumsum(s0, ints, f.q)
                    scratch[name] = sc
        for name, f in fx.items():
            if not f.use_q:
                continue
            sc = scratch[name]
            nb = max(n_tr, 1.0)
            sx = sc["sx"] * f.steps["sx"]
            syx0 = sc["syx"] * f.steps["syx"]
            sy0 = sc["sy"] * f.steps["sy"]
            qinc = {"q_sx2": sx ** 2 / nb, "q_yx2": syx0 ** 2 / nb,
                    "q_yxy": syx0 * sy0[:, None] / nb, "q_y2": sy0 ** 2 / nb}
            for k in f.keys:
                if k.startswith("q_"):
                    ints = torch.round(qinc[k] / f.steps[k]).unsqueeze(0)
                    qstate[name][k], ev = _sat_cumsum(qstate[name][k], ints, f.q)
                    f.sat[k] += ev
    out = {}
    for name, f in fx.items():
        d = {k: v.clone() for k, v in exact.d.items()}
        for k in f.keys:
            val = qstate[name][k] if k.startswith("q_") else f.state[k]
            d[k] = (val * f.steps[k]).to(exact.d[k])
        if alias_counters:
            _set_counter_aliases(d)
        out[name] = (PopStats(d, exact.n_trials), {"sat_events": dict(f.sat), "bits": f.bits})
    return out


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
    16 fractional bits.  Because the mid-point entries are not exact, the output range is
    [0, 2^(0.5/entries)] = [0, 1.00135] (exp(0) -> 1.00135, not 1), so the output register
    needs one integer bit (or a saturate-to-1 step); the softmax normalisation absorbs the
    common bias.
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
