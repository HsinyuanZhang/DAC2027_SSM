"""Source-only rule selection for the expansion queue (pure Python / numpy; no torch, no NWB).

The selection session of a backbone is ``Backbone.val_session`` (pool[-1]).  For the new backbones it is
held out of trunk training (OOS and causal); for the reused sm_0307 trunk it is in-sample (IS).  The rules are

* R1-stream: readout kind and ridge lambda per budget B (streamed readout statistics on the P0 table);
* R2 (F7-J): (L, eta, rho, S) per B;
* R3 = F7-J (the R2 rule of the same B) followed by a readout ridge: kind and lambda per B.

Selection scores are mean query R2 over the simulated B-trial calibration episodes of the selection session
(2 P0 seeds x ``sel_draws`` draws per B); ties go to the first key in sorted order (deterministic).
"""
from __future__ import annotations

import numpy as np

READOUT_KINDS = ("full", "diag", "lr8", "lr16", "lr32")
RIDGE_LAMS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0)


def cfg_key(L, eta, rho, S) -> str:
    return f"{int(L)}|{float(eta)!r}|{float(rho)!r}|{int(S)}"


def parse_cfg_key(key: str) -> dict:
    L, eta, rho, S = key.split("|")
    return {"L": int(L), "eta": float(eta), "rho": float(rho), "S": int(S)}


def _mean_by_key(rows, getter):
    acc: dict[str, list[float]] = {}
    for r in rows:
        for k, v in getter(r).items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def best_key(means: dict[str, float]) -> str:
    """argmax with the first (sorted) key winning ties."""
    return max(sorted(means), key=lambda k: means[k])


def select_r1(rows, kinds=READOUT_KINDS, lams=RIDGE_LAMS) -> dict:
    """Rows: {"B", "r2": {"p0": x, "<kind>|<lam>": y}}.  Per kind the best lambda, then the best kind."""
    means = _mean_by_key(rows, lambda r: r["r2"])
    per_kind = {}
    for kind in kinds:
        lam = max(lams, key=lambda l: means[f"{kind}|{l}"])
        per_kind[kind] = {"lam": lam, "source_r2": means[f"{kind}|{lam}"]}
    kind = max(kinds, key=lambda k: per_kind[k]["source_r2"])
    return {"kind": kind, "lam": per_kind[kind]["lam"], "source_r2": per_kind[kind]["source_r2"],
            "p0_source_r2": means["p0"], "n_episodes": len(rows), "per_kind": per_kind}


def select_r2(rows) -> dict:
    """Rows: {"r2cfg": {"L|eta|rho|S": r2}}.  Best (L, eta, rho, S) by mean source R2."""
    means = _mean_by_key(rows, lambda r: r["r2cfg"])
    key = best_key(means)
    return {**parse_cfg_key(key), "source_r2": means[key], "n_episodes": len(rows)}


def select_r3(rows, kinds=READOUT_KINDS, lams=RIDGE_LAMS) -> dict:
    """Rows: {"r2sel": x, "r3": {"<kind>|<lam>": y}}.  Same kind/lambda rule as R1, on the F7-J tables."""
    means = _mean_by_key(rows, lambda r: r["r3"])
    per_kind = {}
    for kind in kinds:
        lam = max(lams, key=lambda l: means[f"{kind}|{l}"])
        per_kind[kind] = {"lam": lam, "source_r2": means[f"{kind}|{lam}"]}
    kind = max(kinds, key=lambda k: per_kind[k]["source_r2"])
    return {"kind": kind, "lam": per_kind[kind]["lam"], "source_r2": per_kind[kind]["source_r2"],
            "r2_source_r2": float(np.mean([r["r2sel"] for r in rows])), "n_episodes": len(rows),
            "per_kind": per_kind}


def assemble_selection(bid, session, oos, scope, b_list, rows_r12, rows_r3) -> dict:
    """The final ``select/<bid>.json`` payload from the two episode logs."""
    out = {}
    for B in b_list:
        r12 = [r for r in rows_r12 if r["B"] == B]
        r3 = [r for r in rows_r3 if r["B"] == B]
        if not r12 or not r3:
            raise ValueError(f"{bid}: no selection episodes for B={B}")
        out[str(B)] = {"r1s": select_r1(r12), "r2": select_r2(r12), "r3": select_r3(r3)}
    return {"bid": bid, "rule_version": "expansion_v2", "selection_session": session, "selection_oos": bool(oos),
            "selection_scope": scope, "b": out}
