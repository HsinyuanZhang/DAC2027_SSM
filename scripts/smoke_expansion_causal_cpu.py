#!/usr/bin/env python
"""CPU end-to-end smoke of expansion round 2: existing stages, then causal-source and causal-eval.

Official Mamba-3 is replaced by a small GRU. Real sub-C NWB, one backbone, one target, tiny grids.
The second pass must not append rows. Numbers are not meaningful.

  python scripts/smoke_expansion_causal_cpu.py [--out /tmp/exp_causal_smoke]
"""
from __future__ import annotations

import argparse
import inspect
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


def _cap_p0_extension():
    """The real trainer doubles the budget when the best step is in the last 20%. A smoke must not."""
    import scripts.run_population_calib as rpc
    src = inspect.getsource(rpc.cmd_train).replace(
        "if best_step > 0.8 * steps and doublings < 4:",
        "if False and best_step > 0.8 * steps and doublings < 4:")
    ns: dict = {}
    exec(src, rpc.__dict__, ns)                                  # noqa: S102 - smoke-only cap, not a sandbox
    rpc.cmd_train = ns["cmd_train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/exp_causal_smoke")
    ap.add_argument("--bid", default="sc_0713")
    ap.add_argument("--target", default="20150714")
    a = ap.parse_args()
    from ssm_decode import ap_backbone, ap_pretraining, expansion_causal as ec
    ap_backbone.build_mamba3_official = lambda *x, **k: CpuTrunk(*x, **k)
    from scripts import run_expansion as rx
    from scripts import run_expansion_causal as rxc
    _cap_p0_extension()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    legacy = out / "empty_legacy"
    legacy.mkdir(parents=True)

    def fake_call(cmd, env=None, cwd=None):
        assert cmd[1:4] == ["-m", "ssm_decode.ap_pretraining", "--task"], cmd
        ap_pretraining.main(cmd[3:])
        return 0

    rx.subprocess.call = fake_call
    pc = rx.pc_module()
    pc.SIM_MAX_BINS = 8000
    common = dict(out=str(out), work=None, bid=a.bid, tier="primary", exclude_apst_dev=False, device="cpu",
                  include_2016_optional=False)
    ns = lambda **k: SimpleNamespace(**common, **k)                          # noqa: E731
    for rep in range(2):
        rx.cmd_backbone(ns(ap_steps0=8, ap_attempts=1, ap_val_interval=4))
        rx.cmd_p0(ns(p0_steps=4, p0_min_steps=2, p0_val_interval=2))
        rx.cmd_jacobian(ns(jac_windows=4, jac_width=48))
        rx.cmd_basis_and_select(ns(sel_draws=1, budgets="8", r2_ls="8", r2_etas="1", r2_rhos="0", r2_steps="1"))
        rxc.cmd_causal_source(ns(artifacts=None, legacy_select=str(legacy), budgets="8", seeds="0",
                                 latency="0,1", targets=a.target, etas="0,1", rhos="0", combo_alphas="0.5",
                                 allow_incomplete_pool=True))
        before = _nrow(out, a.bid, a.target)
        rxc.cmd_causal_eval(ns(artifacts=None, legacy_select=str(legacy), budgets="8", seeds="0",
                               latency="0,1", targets=a.target, etas="0,1", rhos="0", combo_alphas="0.5",
                               allow_incomplete_pool=True))
        after = _nrow(out, a.bid, a.target)
        if rep == 0:
            assert after > 0, "causal-eval wrote no rows"
        else:
            assert after == before, "second pass appended rows"
    rows = _rows(out, a.bid, a.target)
    assert rows and all(r.get("h2_backward_calls", 0) == 0 for r in rows if not r.get("unavailable"))
    assert all(r.get("h4_query_disjoint", False) for r in rows if not r.get("unavailable"))
    assert all(r.get("protocol") == "causal_first_B" for r in rows)
    assert any(r["method"] == "p0" and r["latency"] == 0 for r in rows)
    assert any(r["method"] == "r2" and r.get("replay") for r in rows)
    comp = json.loads((out / "causal_select" / "composition" / f"{a.bid}_{a.target}.json").read_text())
    assert "n_sessions_a" in comp["by_B"]["8"]
    print("CAUSAL SMOKE OK", len(rows), "rows; sessions", comp["by_B"]["8"]["n_sessions_a"],
          {m: round(sum(r["r2"] for r in rows if r["method"] == m and r.get("r2") is not None) /
                    max(1, sum(r["method"] == m and r.get("r2") is not None for r in rows)), 3)
           for m in ("p_prior", "p0", "w_alpha_a", "s1_causal", "combo_a", "ridge_cca", "r2")})
    # touch the symbol so a refactor that drops the import is visible
    assert ec.RULE_VERSION


def _rows(out, bid, date):
    path = Path(out) / "causal_rows" / bid / f"{date}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _nrow(out, bid, date):
    path = Path(out) / "causal_rows" / bid / f"{date}.jsonl"
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text().splitlines() if l.strip())


if __name__ == "__main__":
    main()
