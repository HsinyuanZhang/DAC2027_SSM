#!/usr/bin/env python
"""Aggregate Track A rotation results and check the decision gate 3.7.

Reads every ``results.jsonl`` under the given root, builds the method-by-budget
table (mean +/- SD across target sessions), computes paired differences to the
A-M0 33-trial reference, and evaluates the gate:

    Continue Track A if A-M1 or A-M3 with <= 8 labeled trials matches A-M0
    with 33 labeled trials on most target sessions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(root: Path):
    rows = []
    for path in sorted(root.glob("*/results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gate-budget", type=int, default=8)
    parser.add_argument("--reference", default="am0")
    parser.add_argument("--reference-budget", type=int, default=33)
    parser.add_argument("--oracle", action="store_true", help="include oracle_frame rows")
    args = parser.parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"no results under {args.root}")
    excluded = set() if args.oracle else {"oracle_frame"}
    methods = sorted({r["method"] for r in rows} - excluded)
    budgets = sorted({r["budget"] for r in rows})
    sessions = sorted({r["target_session"] for r in rows})

    def get(method, budget, session):
        hits = [r["r2"] for r in rows
                if r["method"] == method and r["budget"] == budget and r["target_session"] == session]
        return hits[0] if hits else None

    print(f"{len(rows)} rows | {len(sessions)} target sessions | budgets {budgets}\n")
    header = f"{'method':<14}" + "".join(f"b={b:<8}" for b in budgets)
    print(header)
    for method in methods:
        cells = []
        for budget in budgets:
            values = [get(method, budget, s) for s in sessions]
            values = [v for v in values if v is not None]
            if values:
                cells.append(f"{np.mean(values):.3f}    ")
            else:
                cells.append("  --      ")
        print(f"{method:<14}" + "".join(cells))

    if args.reference in methods and args.reference_budget in budgets:
        print(f"\nPaired difference to {args.reference}@{args.reference_budget} "
              f"(positive = better than reference; n = sessions with both):")
        for method in methods:
            if method == args.reference:
                continue
            for budget in budgets:
                pairs = [(get(method, budget, s), get(args.reference, args.reference_budget, s))
                         for s in sessions]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if len(pairs) < 3:
                    continue
                diff = np.array([a - b for a, b in pairs])
                wins = int((diff > 0).sum())
                print(f"  {method:<12} b={budget:<3} mean {diff.mean():+.3f}  "
                      f"SD {diff.std(ddof=1):+.3f}  wins {wins}/{len(pairs)}")

        print("\nDecision gate 3.7 (A-M1 or A-M3 at <= gate budget vs A-M0 at reference budget):")
        verdicts = {}
        for method in ("am1", "am3"):
            if method not in methods:
                continue
            for budget in budgets:
                if budget > args.gate_budget:
                    continue
                pairs = [(get(method, budget, s), get(args.reference, args.reference_budget, s))
                         for s in sessions]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if len(pairs) < 3:
                    continue
                diff = np.array([a - b for a, b in pairs])
                margin = float(-diff.mean())
                wins = int((diff > 0).sum())
                verdicts[f"{method}@{budget}"] = (wins, len(pairs), margin)
                passed = wins > len(pairs) / 2
                print(f"  {method}@{budget:<3} wins {wins}/{len(pairs)} mean gap {margin:+.3f} "
                      f"{'PASS' if passed else 'FAIL'}")
        if verdicts:
            best = max(verdicts.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
            print(f"  gate outcome: best {best[0]} wins {best[1][0]}/{best[1][1]}")


if __name__ == "__main__":
    main()
