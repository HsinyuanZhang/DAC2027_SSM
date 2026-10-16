#!/usr/bin/env python
"""CPU end-to-end smoke test of the expansion queue (backbone -> p0 -> jacobian -> select -> eval), tiny budgets.

The official Mamba-3 trunk needs CUDA/Triton, so a small GRU trunk with the same interface
(``in_proj`` / ``out_proj`` / forward [B,T,K] -> [B,T,D]) is substituted.  Real sub-C NWB data, the real
loader, protocol, driver, P0 trainer, R1-stream selection and evaluation code run unchanged.  The numbers are
meaningless; the point is that every stage runs (including the R2 / R3 grid on a tiny grid and the Jacobian
stage), is resumable (a second invocation is a no-op) and writes the expected files.  It also checks the OOS
split: the trunk checkpoint must not contain pool[-1] and P0 early-stops on pool[-1].

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
    ap.add_argument("--sim-bins", type=int, default=3000, help="selection simulation length (CPU smoke only)")
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
    pc = rx.pc_module()
    pc.SIM_MAX_BINS = a.sim_bins
    common = dict(out=str(out), work=a.work, bid=a.bid, tier="primary", exclude_apst_dev=False, device="cpu")
    ns = lambda **k: SimpleNamespace(**common, **k)                                  # noqa: E731
    for rep in range(2):                      # second pass must be a no-op (resumability)
        rx.cmd_backbone(ns(ap_steps0=20, ap_attempts=2, ap_val_interval=10))
        rx.cmd_p0(ns(p0_steps=20, p0_min_steps=10, p0_val_interval=10))
        rx.cmd_jacobian(ns(jac_windows=6, jac_width=48))
        rx.cmd_basis_and_select(ns(sel_draws=1, budgets="1,4", r2_ls="8", r2_etas="10", r2_rhos="0.1", r2_steps="1,3"))
        rx.cmd_eval(ns(draws=1, targets=a.target, budgets="1,4"))
    rx.cmd_status(ns())
    bb = rx.select_backbones(ns())[0]
    blob = torch.load(out / "backbones" / bb.bid / "final" / "best.pt", map_location="cpu", weights_only=False)
    assert list(blob["source_sessions"]) == list(bb.trunk_sources) and bb.pool[-1] not in blob["source_sessions"]
    done = json.loads((out / "runs" / f"{bb.bid}_O" / "p0" / "seed0" / "done.json").read_text())
    assert done["val_ids"] == [bb.pool[-1]] and bb.pool[-1] not in done["train_ids"]
    sel = json.loads((out / "select" / f"{bb.bid}.json").read_text())
    assert sel["selection_oos"] is True and sel["selection_session"] == bb.pool[-1]
    rows = [json.loads(l) for f in (out / "rows" / a.bid).glob("*.jsonl") for l in f.read_text().splitlines()]
    assert rows and all(r["h2_backward_calls"] == 0 and r["h4_query_disjoint"] for r in rows), "row checks failed"
    need = {"h1_trunk_hash", "h1_trunk_unchanged", "h5_bytes", "h5_within_32kb", "unit_check_passed"}
    assert all(need <= set(r) for r in rows) and {r["method"] for r in rows} == {"p_prior", "p0", "r1s", "r2", "r3"}
    print("SMOKE OK", len(rows), "rows;", json.dumps({m: round(sum(r["r2"] for r in rows if r["method"] == m) /
                                                          max(1, sum(r["method"] == m for r in rows)), 3)
                                                      for m in ("p_prior", "p0", "r1s", "r2", "r3")}))


if __name__ == "__main__":
    main()
