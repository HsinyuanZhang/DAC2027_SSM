#!/usr/bin/env python
"""CPU end-to-end smoke test of the expansion queue (backbone -> p0 -> select -> eval), tiny budgets.

The official Mamba-3 trunk needs CUDA/Triton, so a small GRU trunk with the same interface
(``in_proj`` / ``out_proj`` / forward [B,T,K] -> [B,T,D]) is substituted.  Real sub-C NWB data, the real
loader, protocol, driver, P0 trainer, R1-stream selection and evaluation code run unchanged.  The numbers are
meaningless; the point is that every stage runs, is resumable (a second invocation is a no-op) and writes the
expected files.

  python scripts/smoke_expansion_cpu.py [--out /tmp/exp_smoke] [--bid sc_0713]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CpuTrunk(nn.Module):
    def __init__(self, input_size, output_size, width=64, layers=1, state_size=128, dropout=0.0, ffn_mult=2):
        super().__init__()
        self.in_proj = nn.Linear(input_size, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.out_proj = nn.Linear(width, output_size)
        self.config = SimpleNamespace(width=width)

    def forward(self, x):
        h, _ = self.gru(self.in_proj(x))
        return self.out_proj(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/exp_smoke")
    ap.add_argument("--work", default=None)
    ap.add_argument("--bid", default="sc_0713")
    ap.add_argument("--target", default="20150714")
    a = ap.parse_args()
    from ssm_decode import ap_backbone, ap_pretraining
    ap_backbone.build_mamba3_official = lambda *x, **k: CpuTrunk(*x, **k)
    from scripts import run_expansion as rx
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)

    def fake_call(cmd, env=None, cwd=None):
        assert cmd[1:4] == ["-m", "ssm_decode.ap_pretraining", "--task"], cmd
        ap_pretraining.main(cmd[3:])
        return 0
    rx.subprocess.call = fake_call
    common = dict(out=str(out), work=a.work, bid=a.bid, tier="primary", exclude_apst_dev=False, device="cpu")
    ns = lambda **k: SimpleNamespace(**common, **k)                                  # noqa: E731
    for rep in range(2):                      # second pass must be a no-op (resumability)
        rx.cmd_backbone(ns(ap_steps0=20, ap_attempts=2, ap_val_interval=10))
        rx.cmd_p0(ns(p0_steps=20, p0_min_steps=10, p0_val_interval=10))
        rx.cmd_basis_and_select(ns(sel_draws=1, budgets="1,4"))
        rx.cmd_eval(ns(draws=1, targets=a.target, budgets="1,4"))
    rx.cmd_status(ns())
    rows = [json.loads(l) for f in (out / "rows" / a.bid).glob("*.jsonl") for l in f.read_text().splitlines()]
    assert rows and all(r["h2_backward_calls"] == 0 and r["h4_query_disjoint"] for r in rows), "row checks failed"
    print("SMOKE OK", len(rows), "rows;", json.dumps({m: round(sum(r["r2"] for r in rows if r["method"] == m) /
                                                          max(1, sum(r["method"] == m for r in rows)), 3)
                                                      for m in ("p_prior", "p0", "r1s")}))


if __name__ == "__main__":
    main()
