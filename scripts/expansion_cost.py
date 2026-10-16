#!/usr/bin/env python
"""Cost model of the expansion queue (RTX 3090 timings from the P1 / P3 logs, scaled to the Colab L4).

No GPU is used.  Every constant is a measured number from an existing log (source in the comment); the L4 factor
is an assumption (not measured).  Output: per-backbone and total hours (expected / sequential upper bound) and
CU (1.54 CU/h) for the whole queue and for a proposed split into rounds.

  python scripts/expansion_cost.py [--l4-factor 2.0] [--sel-draws 4] [--rounds 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import expansion_protocol as ep                      # noqa: E402

CU_PER_HOUR = 1.54
CU_CAP = 40.0
# ---- GPU seconds on the RTX 3090 (solo unless noted) -------------------------------------------------------
AP_CHAIN_UPPER = 391.0      # stage2_intervals train_log.jsonl: 2000/4000/8000/16000 steps = 99+56+87+148 s
AP_CHAIN_EXPECT = 155.0     # two attempts (2000 + 4000 steps)
P0_SEED = {"sub-M": (120.0, 172.0), "sub-C": (180.0, 260.0)}   # P1 done.json: 81-172 s per seed; sub-C N ~2x -> x1.5
JAC = 60.0                  # P3 jac_year.log: 6 sessions, lmax 32, 600 windows each: 57 s solo (incl. loading)
SEL_EP_R12 = (20.0, 28.0)   # P3 select logs: 14 s (contig, no stream) .. 28 s (3 concurrent jobs) per episode (R1-stream + 84 R2 rules)
SEL_EP_R3 = (2.0, 3.0)      # P3 select-r3_month_oosc.log: 46 s / 24 episodes
EVAL_GROUP = (1.2, 2.4)     # P3 eval_month_oosc.log: 228 s / 96 (draw, seed) groups with ALL R1/R3 kinds = 2.4 s; ours
                            # computes p_prior, p0, r1s, r2, r3 only
PROC_OVERHEAD = 30.0        # python + torch + CUDA init + triton warm-up per stage process
# ---- CPU seconds -------------------------------------------------------------------------------------------
LOAD_SMALL, LOAD_BIG, BIG_BYTES = 7.0, 20.0, 150e6   # sub-C: ~7 s per file, ~20 s for the 2015-03 files (MZ)


def load_seconds(size_bytes: float) -> float:
    return LOAD_BIG if size_bytes >= BIG_BYTES else LOAD_SMALL


def backbone_cost(bb, inv, sel_draws: int, l4_factor: float) -> dict:
    """Seconds (3090-GPU part scaled by ``l4_factor``, CPU part unscaled) for expected / upper bound."""
    pool_load = {s: load_seconds(inv[s]["size_bytes"]) for s in bb.pool}
    trunk_load = sum(pool_load[s] for s in bb.trunk_sources)
    pool_total = sum(pool_load.values())
    tgt_load = sum(load_seconds(inv[t.session]["size_bytes"]) for t in bb.targets)
    n_eval = len(bb.targets)
    n_ep = len(ep.SEEDS) * len(ep.B_LIST) * sel_draws
    out = {}
    for i, case in enumerate(("expected", "upper")):
        attempts = 2 if case == "expected" else 4
        gpu = 0.0
        cpu = 0.0
        procs = 0
        if not bb.reuse:
            gpu += AP_CHAIN_EXPECT if case == "expected" else AP_CHAIN_UPPER
            cpu += attempts * trunk_load
            gpu += 2 * P0_SEED[bb.subject][i]
            cpu += 2 * pool_total
            procs += attempts + 2
        gpu += JAC
        cpu += pool_total
        gpu += n_ep * (SEL_EP_R12[i] + SEL_EP_R3[i])
        cpu += pool_total
        gpu += n_eval * len(ep.B_LIST) * ep.N_DRAWS * len(ep.SEEDS) * EVAL_GROUP[i]
        cpu += trunk_load + tgt_load
        procs += 3
        out[case] = {"gpu_3090_s": gpu, "gpu_l4_s": gpu * l4_factor, "cpu_s": cpu,
                     "wall_s": gpu * l4_factor + cpu + procs * PROC_OVERHEAD}
    return out


def split_rounds(costs: dict, n_rounds: int, key: str) -> list[list[str]]:
    """Greedy longest-first balancing of backbones over ``n_rounds`` rounds."""
    rounds = [[] for _ in range(n_rounds)]
    load = [0.0] * n_rounds
    for bid in sorted(costs, key=lambda b: -costs[b][key]["wall_s"]):
        j = min(range(n_rounds), key=lambda r: load[r])
        rounds[j].append(bid)
        load[j] += costs[bid][key]["wall_s"]
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l4-factor", type=float, default=2.0, help="L4 time / 3090 time (assumption; range 1.5-2.5)")
    ap.add_argument("--sel-draws", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--plan", default="sc_0708,sc_0713,sc_0629;sc_0309,sm_0307,sm_0627,sm_0512,sm_0611",
                    help="rounds separated by ';', backbones by ','; use --plan balanced for greedy balancing")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    inv = ep.load_inventory()
    bbs = ep.select_tier(ep.build_backbones(), tier="primary")
    costs = {bb.bid: backbone_cost(bb, inv, a.sel_draws, a.l4_factor) for bb in bbs}
    report = {"l4_factor": a.l4_factor, "sel_draws": a.sel_draws, "backbones": {}, "rounds": {}}
    print(f"{'bid':8} {'tgts':>4} {'GPU3090 h':>10} {'exp wall h':>11} {'upper wall h':>13}")
    for bb in bbs:
        c = costs[bb.bid]
        print(f"{bb.bid:8} {len(bb.targets):4d} {c['upper']['gpu_3090_s'] / 3600:10.2f} "
              f"{c['expected']['wall_s'] / 3600:11.2f} {c['upper']['wall_s'] / 3600:13.2f}")
        report["backbones"][bb.bid] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in c.items()}
    for case in ("expected", "upper"):
        tot = sum(c[case]["wall_s"] for c in costs.values()) / 3600
        gpu = sum(c[case]["gpu_l4_s"] for c in costs.values()) / 3600
        print(f"TOTAL {case}: sequential wall {tot:.1f} h ({tot * CU_PER_HOUR:.1f} CU); L4 GPU part {gpu:.1f} h")
        report[f"total_{case}_h"] = round(tot, 2)
        report[f"total_{case}_cu"] = round(tot * CU_PER_HOUR, 1)
    rounds = (split_rounds(costs, a.rounds, "upper") if a.plan == "balanced"
              else [r.split(",") for r in a.plan.split(";")])
    assert sorted(b for r in rounds for b in r) == sorted(costs), "plan must cover every backbone exactly once"
    for i, r in enumerate(rounds, 1):
        h = {c: sum(costs[b][c]["wall_s"] for b in r) / 3600 for c in ("expected", "upper")}
        print(f"round {i}: {r}  expected {h['expected']:.1f} h ({h['expected'] * CU_PER_HOUR:.1f} CU), "
              f"upper {h['upper']:.1f} h ({h['upper'] * CU_PER_HOUR:.1f} CU)")
        report["rounds"][f"round{i}"] = {"backbones": r, **{f"{c}_h": round(v, 2) for c, v in h.items()},
                                         **{f"{c}_cu": round(v * CU_PER_HOUR, 1) for c, v in h.items()}}
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
