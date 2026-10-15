"""Analytic cost of the P3 forward-only refinements (R1 readout ridge, R2 F7-J, R3 both).

Everything is derived from explicit tensor shapes; every buffer is listed (no hidden scratch).
MAC = one multiply-add.  Not measured on hardware.  The trunk MAC count is read off an
instantiated trunk (``fwd_refine.trunk_macs_per_bin``).

Methods
  r1_full / r1_diag / r1_lr<r>   replay the B trials through the frozen trunk ONCE, accumulate the
                                 readout Gram (full / diagonal / projected on a fixed rank-r basis)
  r1s_full / r1s_lr<r>           "streaming" variant: no replay buffer; each calibration trial is
                                 forwarded once while it arrives (table from the previous trials)
  r2                             F7-J, S replay passes
  r3_full / r3_lr<r>             F7-J (S passes) + one readout pass

MAC classes (feasibility rule 8: a "fixed-weight MAC" is weight x activation only).  Every MAC is
assigned to one class and ``macs_by_class`` is reported:
  fixed_weight_rram    trunk nn.Linear layers (weights fixed in RRAM)
  sram_table_x_act     read-in table MVM (K x N table in SRAM, written per target) and the Jbar
                       feedback convolution (ROM weights) -- weight x activation, but NOT RRAM
  act_x_act            Mamba3 selective scan (state update and readout), e x^T outer product,
                       readout Gram / cross-statistics, solves, per-step table update, P0 finalize
Buffers are sized from the ACTUAL replayed bins (``replay_bins``) and the actual staging trial
length (``staging_bins``) when given; otherwise from the mean trial length T (``B*T``), which is
only a point estimate and under-sizes the worst case (review item 3).  Fixed-point effects are NOT
simulated anywhere in P3: all paths are evaluated in fp32 and the int8 ROM (Jbar, readout basis)
is a costing assumption only.

Reference ``p0_base`` re-derives the MAC part of the P0 calibration (accumulate + finalize +
per-unit MLP); it is an independent restatement, not an import of ``calib_cost``.
"""
from __future__ import annotations

HIDDEN = 64


def p0_macs(n, k, d, nb, b, hidden=HIDDEN):
    fd = 17 + 2 * k + 5 * d
    acc = nb * (n * n + (6 + 5 * d) * n + 9) + b * (n + 2 * d * n + d)
    fin = (2 * k + 1) * n * n + (k + 27) * n + 27
    mlp = n * (fd * hidden + hidden * hidden + hidden * k)
    return {"accumulate": acc, "finalize": fin, "calibrator_mlp": mlp, "total": acc + fin + mlp}


def _readout_stat_entries(h, d, kind):
    ha = h + 1
    if kind == "full":
        return ha * (ha + 1) // 2 + d * ha
    if kind == "diag":
        return ha + d * ha
    r = int(kind[2:])
    return r * (r + 1) // 2 + d * r


def _readout_macs(h, d, nb, kind):
    ha = h + 1
    if kind == "full":
        acc = nb * (ha * (ha + 1) // 2 + d * ha)
        solve = ha ** 3 // 6 + 2 * d * ha * ha
    elif kind == "diag":
        acc = nb * (ha + d * ha)
        solve = d * ha
    else:
        r = int(kind[2:])
        acc = nb * (ha * r + r * (r + 1) // 2 + d * r)         # project hb on the basis, then r x r Gram
        solve = r ** 3 // 6 + 2 * d * r * r + d * r * ha       # solve + back-projection of dV
    return acc, solve


_ACT_ACT = ("outer product", "precondition", "readout accumulate", "readout solve")


def _classify(case, macs, nb, method, steps):
    """Split the MAC dict into the three classes (see module docstring)."""
    n, k = case["N"], case["K"]
    tr = case["trunk"]
    lin = tr.get("fixed_weight_macs", tr["total"])
    scan = tr.get("act_x_act_macs", 0)
    passes = steps if method == "r2" else steps + 1 if method.startswith("r3_") else 1
    by = {"fixed_weight_rram": 0, "sram_table_x_act": 0, "act_x_act": 0}
    for name, v in macs.items():
        if name.startswith(("replay forward", "streaming forward")):
            reps = steps if name.endswith("x S") else 1
            by["fixed_weight_rram"] += reps * nb * lin
            by["act_x_act"] += reps * nb * scan
            by["sram_table_x_act"] += reps * nb * k * n
        elif name.startswith("feedback conv"):
            by["sram_table_x_act"] += v                    # Jbar ROM weights x residual activations
        elif name.startswith("extra P0 refreshes"):
            by["act_x_act"] += v                           # finalize (Gram algebra) + calibrator MLP;
            #   the MLP part is weight x activation but is < 0.1% of the total and kept here
        elif name.startswith(_ACT_ACT):
            by["act_x_act"] += v
        else:
            raise ValueError(f"unclassified MAC entry: {name}")
    assert sum(by.values()) == sum(macs.values()), (by, macs)
    return by


def refinement_cost(case: dict, b: int, method: str, *, steps=5, lags=16, acc_bits=32,
                    table_bits=32, replay_label_bits=None, staging="trial",
                    replay_bins=None, staging_bins=None) -> dict:
    n, k, d, h = case["N"], case["K"], case["D"], case["H"]
    t, mt = case["T"], case["trunk"]["total"]
    bx, by = case["bx"], replay_label_bits or case["by"]
    nb = b * t if replay_bins is None else int(replay_bins)      # replayed bins (actual if given)
    stg_bins = t if staging_bins is None else int(staging_bins)
    out = {"case": case["name"], "method": method, "B": b, "T": t, "replay_bins": nb,
           "staging_bins": stg_bins}
    fwd_bin = mt + k * n                            # trunk + read-in MVM per replayed bin
    macs, sram, rom = {}, {}, {}
    replay_buf = nb * (n * bx + d * by + 1) / 8     # raw counts + labels + valid bit
    state_bytes = case["trunk"]["state_elems"] * case.get("state_bits", 16) / 8
    sram["trunk_state (shared with inference)"] = state_bytes
    sram["table T"] = k * n * table_bits / 8
    sram["norm mean/std"] = 2 * n * 2

    def readout(kind, passes_bins, nbin_acc):
        a, s = _readout_macs(h, d, nbin_acc, kind)
        macs[f"readout accumulate [{kind}]"] = a
        macs[f"readout solve [{kind}]"] = s
        sram[f"readout stats [{kind}]"] = _readout_stat_entries(h, d, kind) * acc_bits / 8
        sram["readout new V (D x (H+1))"] = d * (h + 1) * acc_bits / 8
        sram["feature vector hb"] = (h + 1) * 2
        if kind.startswith("lr"):
            rom[f"basis U (H+1 x {kind[2:]}) int8"] = (h + 1) * int(kind[2:])
        if kind == "full":
            sram["solve scratch (Cholesky in place)"] = 0

    if method == "p0_base":
        out["macs"] = p0_macs(n, k, d, nb, b)
        out["macs_total"] = out["macs"]["total"]
        out["sram_bytes"], out["rom_bytes"] = {}, {}
        out["sram_total"] = 0.0
        return out
    kind = method.split("_", 1)[1] if method.startswith(("r1_", "r1s_", "r3_")) else None
    if method.startswith("r1s_"):
        kind = method[4:]
    if method.startswith("r1_"):
        macs["replay forward (trunk+read-in)"] = nb * fwd_bin
        sram["raw replay buffer"] = replay_buf
        readout(kind, nb, nb)
    elif method.startswith("r1s_"):
        macs["streaming forward (trunk+read-in), once per arriving bin"] = nb * fwd_bin
        if staging == "trial":      # evaluated: trial 0 is normalised with its own statistics
            sram["one-trial staging buffer (trial 0 normalisation)"] = stg_bins * (n * bx + d * by + 1) / 8
        # staging == "none": running (Welford) normalisation, NOT evaluated; keeps no raw samples
        # P0 refresh after every arriving trial except the last = (b-1) extra P0 finalize+MLP
        extra = (b - 1) * (p0_macs(n, k, d, 0, 0)["finalize"] + p0_macs(n, k, d, 0, 0)["calibrator_mlp"])
        macs["extra P0 refreshes (b-1)"] = extra
        readout(kind, nb, nb)
    elif method == "r2" or method.startswith("r3_"):
        s = steps
        macs["replay forward (trunk+read-in) x S"] = s * nb * fwd_bin
        macs["feedback conv (L+1)*D*K per bin x S"] = s * nb * (lags + 1) * d * k
        macs["outer product e x^T (K*N per bin) x S"] = s * nb * k * n
        macs["precondition + update (K*N per step)"] = s * 3 * k * n
        sram["raw replay buffer"] = replay_buf
        sram["residual delay line ((L+1) x D x 16 bit)"] = (lags + 1) * d * 2
        sram["G accumulator K x N"] = k * n * acc_bits / 8
        sram["T0 (shrink centre) K x N"] = k * n * table_bits / 8
        sram["q accumulator N"] = n * acc_bits / 8
        rom["Jbar kernel (L+1) x D x K int8"] = (lags + 1) * d * k
        if method.startswith("r3_"):
            macs["replay forward for readout pass"] = nb * fwd_bin
            readout(kind, nb, nb)
    else:
        raise ValueError(method)
    out["macs"] = macs
    out["macs_total"] = sum(macs.values())
    out["macs_by_class"] = _classify(case, macs, nb, method, steps)
    out["fixed_weight_fraction"] = out["macs_by_class"]["fixed_weight_rram"] / max(out["macs_total"], 1)
    out["sram_bytes"] = sram
    out["sram_total"] = sum(sram.values())
    out["rom_bytes"] = rom
    out["replay_buffer_bytes"] = sram.get("raw replay buffer", 0.0)
    out["trunk_macs_fraction"] = (nb * mt * (steps if method == "r2" else steps + 1 if method.startswith("r3_") else 1)
                                  ) / max(out["macs_total"], 1)
    out["realtime_seconds_equiv"] = nb * case.get("bin_ms", 20) / 1000.0 * (
        steps if method == "r2" else steps + 1 if method.startswith("r3_") else 1)
    return out
