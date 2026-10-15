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


def refinement_cost(case: dict, b: int, method: str, *, steps=5, lags=16, acc_bits=32,
                    table_bits=32, replay_label_bits=None, staging="trial") -> dict:
    n, k, d, h = case["N"], case["K"], case["D"], case["H"]
    t, mt = case["T"], case["trunk"]["total"]
    bx, by = case["bx"], replay_label_bits or case["by"]
    nb = b * t                                      # replayed bins
    out = {"case": case["name"], "method": method, "B": b, "T": t, "replay_bins": nb}
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
            sram["one-trial staging buffer (trial 0 normalisation)"] = t * (n * bx + d * by + 1) / 8
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
    out["sram_bytes"] = sram
    out["sram_total"] = sum(sram.values())
    out["rom_bytes"] = rom
    out["replay_buffer_bytes"] = sram.get("raw replay buffer", 0.0)
    out["trunk_macs_fraction"] = (nb * mt * (steps if method == "r2" else steps + 1 if method.startswith("r3_") else 1)
                                  ) / max(out["macs_total"], 1)
    out["realtime_seconds_equiv"] = nb * case.get("bin_ms", 20) / 1000.0 * (
        steps if method == "r2" else steps + 1 if method.startswith("r3_") else 1)
    return out
