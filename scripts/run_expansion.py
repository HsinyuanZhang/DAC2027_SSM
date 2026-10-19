#!/usr/bin/env python
"""Driver for the 2026-10-16 evidence expansion (sub-C second subject + sub-M days rolling backbones).

One resumable stage per subcommand; every stage is idempotent (finished artefacts are skipped, partial
ones are redone or resumed row by row).  ``--out`` is the persistent root (on Colab: a Drive path),
``--work`` an optional fast local scratch directory: stage outputs are written to ``--work`` and copied to
``--out`` (atomic rename) as soon as a unit of work finishes, so a lost session loses at most one unit.

  check                 protocol audit + electrode coverage per target (CPU, inventory only)
  bundle                Colab data bundle list (files + sizes) -> <repo>/results/analysis/expansion_20261016/colab_bundle.json
  status                which stage of which backbone is done
  backbone --bid B      AP K=10 backbone on the first 5 pool sessions (pool[-1] held out), budget-extension rule
  p0 --bid B            two P0 calibrators: train on the 5 trunk sessions, early-stop on pool[-1]
  jacobian --bid B      averaged Jacobian J-bar of the trunk on the trunk SOURCE sessions only (offline, autograd)
  select --bid B        PCA readout basis + source simulation on pool[-1]: R1-stream, R2 (F7-J), R3 rules per B
  eval --bid B          rows: p_prior, p0, r1s, r2, r3; contiguous decoding, physical-unit R2, audit fields

Selection protocol (2026-10-16): every NEW backbone trains its trunk on pool[:-1]; pool[-1] (the latest pool
session, earlier than every target) is trunk-unseen and is used for P0 early stopping and for selecting the
R1-stream / R2 / R3 rules (OOS and causal).  The reused sm_0307 trunk saw all 6 pool sessions, so its selection
is in-sample (IS) and its rows must not be merged with the OOS rows or with the P3 / drift-curve R1-R3 rows.

Rules: never final sessions (sub-M 0623/0625/0626; sub-C 20151113/16/17/19/20 and 20151201; the loader refuses
them), never results/official_evalai/; the 2016 sub-C year targets stay off unless --include-2016-optional is
given (logged warning); the target-side path runs under no_grad with zero backward calls (counted per row).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import expansion_protocol as ep                       # noqa: E402
from ssm_decode import expansion_rules as er                          # noqa: E402
from ssm_decode.atomic_jsonl import append_jsonl, read_jsonl, repair_jsonl   # noqa: E402

METHODS_SEEDED = ("p0", "r1s", "r2", "r3")        # seed-dependent rows; p_prior is seed-free (stored with seed 0)
EXPANSION_CODE_FILES = (          # files the expansion queue needs on top of the M2 code bundle (all git-tracked)
    "scripts/run_expansion.py", "scripts/run_expansion_queue.sh", "scripts/run_expansion_causal.py",
    "scripts/analyze_expansion.py", "scripts/expansion_cost.py", "scripts/smoke_expansion_cpu.py",
    "scripts/smoke_expansion_causal_cpu.py", "scripts/run_population_calib.py",
    "scripts/run_causal_main.py", "scripts/run_baselines_readin.py",
    "scripts/run_pc3_refine.py", "scripts/run_labeled_calibration.py",
    "ssm_decode/expansion_protocol.py", "ssm_decode/expansion_rules.py", "ssm_decode/atomic_jsonl.py",
    "ssm_decode/expansion_causal.py", "ssm_decode/dandi688_data.py", "ssm_decode/fwd_refine.py",
    "ssm_decode/causal_protocol.py",
    "results/analysis/expansion_20261016/session_inventory.json", "results/analysis/expansion_20261016/protocol.json",
    "results/analysis/expansion_20261016/colab_bundle_subC_primary.txt",
    "tests/test_dandi688_subc.py", "tests/test_expansion_protocol.py", "tests/test_run_expansion.py",
    "tests/test_atomic_jsonl.py", "tests/test_expansion_rules.py", "tests/test_analyze_expansion.py",
    "tests/test_expansion_causal.py", "tests/test_causal_main.py",
)
JAC_LMAX = 32                                     # P3 value; the L grid is (8, 16, 32)

DEFAULT_OUT = ROOT / "results/expansion_20261016"
EXPERIMENT_FILES = ROOT / "results/analysis/expansion_20261016"


# ------------------------------------------------------------------------------------------- storage

class Store:
    """Persistent root ``out`` plus an optional local scratch ``work`` (same relative layout)."""

    def __init__(self, out, work=None):
        self.out = Path(out)
        self.work = Path(work) if work else self.out

    @property
    def split(self) -> bool:
        return self.work.resolve() != self.out.resolve()

    def w(self, rel) -> Path:
        p = self.work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def o(self, rel) -> Path:
        return self.out / rel

    def exists(self, rel) -> bool:
        return self.o(rel).exists()

    def pull(self, rel):
        if self.split and self.o(rel).exists():
            self.w(rel)
            if self.o(rel).is_dir():
                shutil.copytree(self.o(rel), self.work / rel, dirs_exist_ok=True)
            else:
                shutil.copy2(self.o(rel), self.work / rel)

    def push(self, rel):
        if not self.split:
            return
        src, dst = self.work / rel, self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        if src.is_dir():
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            if dst.exists():
                shutil.rmtree(dst)
            os.replace(tmp, dst)
        else:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def jwrite(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------ protocol

def tier_guard(args) -> None:
    """``--tier all`` adds the optional 2016 sub-C year targets, which are OFF (MZ decision 2026-10-16).
    They need the explicit extra flag ``--include-2016-optional`` and then log a warning."""
    if args.tier != "all":
        return
    if not getattr(args, "include_2016_optional", False):
        raise SystemExit("--tier all adds the optional 2016 sub-C year targets, which are switched off "
                         "(MZ decision 2026-10-16: month targets = APST dev sessions, no 2016 sessions). "
                         "Pass --include-2016-optional as well to override.")
    msg = "WARNING: --tier all --include-2016-optional: 2016 sub-C sessions (optional year tier) are included"
    warnings.warn(msg)
    print(msg, file=sys.stderr, flush=True)


def select_backbones(args) -> list[ep.Backbone]:
    tier_guard(args)
    bbs = ep.select_tier(ep.build_backbones(), tier=args.tier, exclude_apst_dev=args.exclude_apst_dev)
    if args.bid and args.bid != "all":
        want = set(args.bid.split(","))
        unknown = want - {b.bid for b in bbs}
        if unknown:
            raise SystemExit(f"unknown backbone id(s) {sorted(unknown)}; known: {[b.bid for b in bbs]}")
        bbs = [b for b in bbs if b.bid in want]
    inv = ep.load_inventory()
    for bb in bbs:
        ep.validate_backbone(bb, inv)
    return bbs


def guard_backbone(bb: ep.Backbone) -> None:
    """Defence in depth, first thing in every stage: no final session anywhere in a backbone (the protocol
    validation and the data loader refuse them too)."""
    bad = ep.forbidden_in(list(bb.pool) + [t.session for t in bb.targets])
    if bad:
        raise PermissionError(f"{bb.bid}: forbidden (final) session(s) {bad}")


def ap_path(bb: ep.Backbone, st: Store) -> Path:
    if bb.reuse:
        return ROOT / bb.reuse["ap"]
    return st.o(f"backbones/{bb.bid}/final/best.pt")


def p0_dir(bb: ep.Backbone, st: Store, seed: int) -> Path:
    if bb.reuse:
        return ROOT / bb.reuse["p0_root"] / "runs" / f"{bb.reuse['p0_stratum']}_O" / "p0" / f"seed{seed}"
    return st.o(f"runs/{bb.bid}_O/p0/seed{seed}")


def register(bbs, st: Store):
    """Make the backbones visible to the P1 World / driver code (D1_POOLS).

    ``sources`` = the trunk's source sessions (pool[:-1] for a new backbone, so they align with the trunk's
    tables); ``heldout`` = pool[-1], appended by ``sessions_for`` as the P0 validation session."""
    from scripts.run_labeled_calibration import D1_POOLS
    for bb in bbs:
        D1_POOLS[bb.bid] = ep.d1_spec(bb, str(ap_path(bb, st)))


# ---------------------------------------------------------------------------------------- stage: check

def cmd_check(args):
    inv = ep.load_inventory()
    bbs = select_backbones(args)
    rows = []
    for bb in bbs:
        audit = ep.validate_backbone(bb, inv)
        pool_elec = set()
        for s in bb.trunk_sources:           # the electrode-aligned prior is the mean of the TRUNK tables only
            pool_elec |= set(inv[s]["unit_to_electrode"])
        for t in bb.targets:
            r = inv[t.session]
            el = r["unit_to_electrode"]
            rows.append({"bid": bb.bid, "target": t.session, "stratum": t.stratum, "gap_days": t.gap_days,
                         "tier": t.tier, "apst_dev": t.apst_dev, "n_units": r["n_units"],
                         "n_usable_trials": r["n_usable_trials"], "n_bins": r["n_bins"],
                         "frac_units_electrode_in_pool": round(sum(e in pool_elec for e in el) / len(el), 3),
                         "n_distinct_electrodes": len(set(el)), "min_pool_units": audit["min_pool_units"],
                         "ledermann_limit": audit["ledermann_limit"], "k": bb.k})
    for r in rows:
        print(json.dumps(r))
    if args.write:
        jwrite(EXPERIMENT_FILES / "target_check.json", rows)
        jwrite(EXPERIMENT_FILES / "protocol.json", ep.protocol_dict(inv))


def cmd_bundle(args):
    inv = ep.load_inventory()
    prim = ep.select_tier(ep.build_backbones(), tier="primary")
    allb = ep.select_tier(ep.build_backbones(), tier="all")        # listing only; the 2016 files are not shipped
    for bb in allb:
        ep.validate_backbone(bb, inv)
    out = {}
    for name, bbs in (("primary", prim), ("all", allb)):
        files = ep.bundle_files(bbs, inv)
        out[name] = {"n_files": len(files), "total_bytes": sum(f["size_bytes"] for f in files), "files": files}
    prim_set = {f["session"] for f in out["primary"]["files"]}
    out["optional_extra"] = [f for f in out["all"]["files"] if f["session"] not in prim_set]
    art = []
    for bb in prim:
        if bb.reuse:
            for rel in (bb.reuse["ap"], f"{bb.reuse['p0_root']}/runs/{bb.reuse['p0_stratum']}_O/p0/seed0",
                        f"{bb.reuse['p0_root']}/runs/{bb.reuse['p0_stratum']}_O/p0/seed1"):
                p = ROOT / rel
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
                art.append({"path": rel, "bytes": size})
    out["reused_artefacts"] = art
    out["layout"] = ("files live under <SSM_DANDI688_ROOT>/sub-M/ and <SSM_DANDI688_ROOT>/sub-C/ "
                     "(host: /home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688)")
    subc = {k: [f for f in out[k]["files"] if f["subject"] == "sub-C"] for k in ("primary", "all")}
    out["new_bundle_subC"] = {
        "tar_name": "data_000688_subC.tar.zst", "tar_member_prefix": "dandi_000688/sub-C/",
        "primary_files": len(subc["primary"]), "primary_bytes": sum(f["size_bytes"] for f in subc["primary"]),
        "optional_2016_files_not_shipped": len(subc["all"]) - len(subc["primary"]),
        "optional_2016_bytes_not_shipped": sum(f["size_bytes"] for f in subc["all"]) - sum(f["size_bytes"] for f in subc["primary"]),
        "note": "ships ONLY allowed primary sessions; the six APST held-out sessions and the optional 2016 "
                "sessions are never listed in the member file"}
    out["subM_needed_in_existing_bundle"] = sorted(f["file"] for f in out["primary"]["files"] if f["subject"] == "sub-M")
    code = [f for f in EXPANSION_CODE_FILES if (ROOT / f).exists()]
    missing = sorted(set(EXPANSION_CODE_FILES) - set(code))
    if missing:
        raise SystemExit(f"code bundle files missing: {missing}")
    out["code_files"] = {"n_files": len(code), "total_bytes": sum((ROOT / f).stat().st_size for f in code), "files": code,
                         "note": "re-pack the SSM repo at the commit that contains these files (they are all tracked)"}
    (EXPERIMENT_FILES / "colab_bundle_code_files.txt").write_text("\n".join(code) + "\n")
    jwrite(EXPERIMENT_FILES / "colab_bundle.json", out)
    (EXPERIMENT_FILES / "colab_bundle_subC_primary.txt").write_text(
        "\n".join(f"sub-C/{f['file']}" for f in subc["primary"]) + "\n")
    print(json.dumps({k: out[k] for k in ("new_bundle_subC", "reused_artefacts", "subM_needed_in_existing_bundle")}, indent=1))


# --------------------------------------------------------------------------------------- stage: status

def stage_flags(bb: ep.Backbone, st: Store) -> dict:
    nrows = 0
    for t in bb.targets:
        f = st.o(f"rows/{bb.bid}/{ep.date_of(t.session)}.jsonl")
        if f.exists():
            repair_jsonl(f)                      # a killed writer of an older version may have left a torn line
            nrows += sum(1 for _ in f.open())
    per_target = len(ep.B_LIST) * ep.N_DRAWS * (1 + len(METHODS_SEEDED) * len(ep.SEEDS))
    return {"backbone": bool(bb.reuse) or st.exists(f"backbones/{bb.bid}/backbone.json"),
            "p0": all((p0_dir(bb, st, s) / "done.json").exists() for s in ep.SEEDS),
            "jacobian": st.exists(f"jacobian/{bb.bid}.pt"),
            "select": st.exists(f"select/{bb.bid}.json"),
            "selection_scope": "OOS" if bb.oos_selection else "IS",
            "rows": f"{nrows}/{per_target * len(bb.targets)}",
            "causal_source": (not bb.oos_selection) or st.exists(f"causal_select/{bb.bid}/done.json"),
            "causal_rows": sum(1 for t in bb.targets
                               if st.exists(f"causal_rows/{bb.bid}/{ep.date_of(t.session)}.jsonl"))}


def cmd_status(args):
    st = Store(args.out, args.work)
    for bb in select_backbones(args):
        print(bb.bid, json.dumps(stage_flags(bb, st)))


# ------------------------------------------------------------------------------------- stage: backbone

def ap_cmd(bb: ep.Backbone, dest: Path, steps: int, args) -> list[str]:
    # the trunk sees pool[:-1] only (pool[-1] is held out for P0 early stopping and rule selection)
    return [sys.executable, "-m", "ssm_decode.ap_pretraining", "--task", "dandi688sm",
            "--target-session", bb.targets[0].session, "--dandi688-sources", ",".join(bb.trunk_sources),
            "--output", str(dest), "--device", args.device, "--k", str(bb.k), "--width", "128",
            "--layers", "2", "--state-size", "32", "--steps", str(steps),
            "--val-interval", str(args.ap_val_interval)]


def cmd_backbone(args):
    st = Store(args.out, args.work)
    for bb in select_backbones(args):
        guard_backbone(bb)
        marker = f"backbones/{bb.bid}/backbone.json"
        if bb.reuse or st.exists(marker):
            print(f"[exp] backbone {bb.bid}: done/reused", flush=True)
            continue
        env = dict(os.environ, PYTHONNOUSERSITE="1", TRITON_CACHE_DIR=f"/tmp/triton_exp_{bb.bid}")
        steps, converged, chosen, t0 = args.ap_steps0, False, None, time.perf_counter()
        for attempt in range(args.ap_attempts):
            name = "ap_pool" if attempt == 0 else f"ap_pool_x{2 ** attempt}"
            rel = f"backbones/{bb.bid}/{name}"
            if not st.exists(f"{rel}/manifest.json"):
                dest = st.work / rel
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[exp] {time.strftime('%T')} backbone {bb.bid} {name} steps={steps}", flush=True)
                rc = subprocess.call(ap_cmd(bb, dest, steps, args), env=env, cwd=ROOT)
                if rc != 0:
                    shutil.rmtree(dest, ignore_errors=True)
                    raise SystemExit(f"[exp] FAILED backbone {bb.bid} {name} rc={rc}")
                st.push(rel)
            man = json.loads(st.o(f"{rel}/manifest.json").read_text())
            best, final = int(man["best_step"]), int(man["args"]["steps"])
            if final != steps:                       # artefact from different hyper-parameters: retrain larger
                steps *= 2
                continue
            chosen = (rel, man)
            if best > steps * 8 // 10 and attempt < args.ap_attempts - 1:
                print(f"[exp] {bb.bid} best_step={best} in last 20% of {steps}; extending", flush=True)
                steps *= 2
                continue
            converged = best <= steps * 8 // 10
            break
        if chosen is None:
            raise SystemExit(f"[exp] no usable attempt for {bb.bid}")
        rel, man = chosen
        fin = f"backbones/{bb.bid}/final"
        (st.work / fin).mkdir(parents=True, exist_ok=True)
        shutil.copy2(st.o(f"{rel}/best.pt"), st.work / fin / "best.pt")
        shutil.copy2(st.o(f"{rel}/manifest.json"), st.work / fin / "manifest.json")
        st.push(fin)
        jwrite(st.w(marker), {"bid": bb.bid, "attempt_dir": rel, "steps": int(man["args"]["steps"]),
                              "best_step": int(man["best_step"]), "converged": converged,
                              "source_val_trunk_r2": man["best_source_val_trunk_r2"], "k": bb.k,
                              "source_sessions": man["source_sessions"],
                              "seconds_total": round(time.perf_counter() - t0, 1),
                              "trunk_sources": list(bb.trunk_sources), "heldout": list(bb.heldout)})
        st.push(marker)
        print(f"[exp] backbone {bb.bid} converged={converged} best={man['best_step']}/{man['args']['steps']}", flush=True)


# ------------------------------------------------------------------------------------------ stage: p0

def check_backbone_matches(bb: ep.Backbone, st: Store):
    import torch
    blob = torch.load(ap_path(bb, st), map_location="cpu", weights_only=False)
    if list(blob["source_sessions"]) != list(bb.trunk_sources):
        raise SystemExit(f"{bb.bid}: checkpoint sources {blob['source_sessions']} != protocol trunk sources "
                         f"{list(bb.trunk_sources)} (pool[-1] must stay out of trunk training)")
    if bb.heldout and set(bb.heldout) & set(blob["source_sessions"]):
        raise SystemExit(f"{bb.bid}: held-out session {bb.heldout} was used for trunk training")
    if int(blob["args"]["k"]) != bb.k:
        raise SystemExit(f"{bb.bid}: checkpoint K={blob['args']['k']} != protocol K={bb.k}")


def cmd_p0(args):
    import torch
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    from scripts import run_population_calib as rpc
    for bb in bbs:
        guard_backbone(bb)
        if bb.reuse:
            print(f"[exp] p0 {bb.bid}: reusing P1 calibrators", flush=True)
            continue
        check_backbone_matches(bb, st)
        for seed in ep.SEEDS:
            rel = f"runs/{bb.bid}_O/p0/seed{seed}"
            if st.exists(f"{rel}/done.json"):
                print(f"[exp] p0 {bb.bid} seed{seed}: done", flush=True)
                continue
            rpc.OUT = st.work
            ns = SimpleNamespace(stratum=bb.bid, pool="O", variant="p0", tag="p0", seed=seed, m=8,
                                 output="residual", label_free=False, hidden=64, steps=args.p0_steps,
                                 min_steps=args.p0_min_steps, patience=8, val_interval=args.p0_val_interval,
                                 batch=8, lr=1e-3, init_from=None, qat="tern", device=args.device,
                                 prior_mode="aligned")
            shutil.rmtree(st.work / rel, ignore_errors=True)
            print(f"[exp] {time.strftime('%T')} p0 {bb.bid} seed{seed}", flush=True)
            rpc.cmd_train(ns)
            st.push(rel)


# --------------------------------------------------------------------------------- stage: select/eval

class Ctx:
    """World + P0 calibrators (+ averaged Jacobian) of one backbone (target-side no_grad)."""

    def __init__(self, bb, st, args, load_sessions):
        import torch
        from scripts import run_population_calib as rpc
        self.torch, self.rpc, self.bb, self.st = torch, rpc, bb, st
        self.device = torch.device(args.device)
        self.world = rpc.World(bb.bid, "O", self.device, load_sessions=load_sessions)
        self.p0 = {}
        for seed in ep.SEEDS:
            calib, _, done = rpc.load_run(p0_dir(bb, st, seed), self.world)
            self.p0[seed] = calib
            self.p0_done = done
        self._jblob = None
        self._jbars = {}

    def jbar(self, lag: int):
        """Averaged Jacobian kernel [lag+1, D, K] of this backbone (trunk SOURCE sessions only)."""
        if lag not in self._jbars:
            if self._jblob is None:
                self._jblob = self.torch.load(self.st.o(f"jacobian/{self.bb.bid}.pt"), weights_only=False)
            self._jbars[lag] = average_jbar(self._jblob, lag).to(self.device)
        return self._jbars[lag]


def average_jbar(blob: dict, lag: int):
    """Mean over the stored windows of all trunk-source sessions; lags 0..lag.  Pure tensor arithmetic."""
    assert blob["lmax"] >= lag, f"Jacobian stored up to lag {blob['lmax']}, need {lag}"
    tot, n = 0, 0
    for s in blob["sessions"].values():
        tot = tot + s["sum"]
        n += s["n"]
    return (tot / n)[:lag + 1].float()


def budgets(args) -> tuple[int, ...]:
    return tuple(int(x) for x in args.budgets.split(",")) if getattr(args, "budgets", None) else ep.B_LIST


def pc_module():
    from scripts import run_pc3_refine as pc
    return pc


def grid(args, pc):
    """(L grid, eta grid, rho grid, snapshot steps) of the R2 search; defaults = the P3 grid."""
    def pick(name, default, cast):
        v = getattr(args, name, None)
        return tuple(cast(x) for x in v.split(",")) if v else tuple(default)
    return (pick("r2_ls", pc.LS, int), pick("r2_etas", pc.ETAS, float), pick("r2_rhos", pc.RHOS, float),
            pick("r2_steps", pc.STEPS, int))


def append_row_checked(path: Path, rows) -> None:
    append_jsonl(path, rows)                       # atomic: temp file + os.replace


def score(ep_, table, v_aug=None) -> tuple[float, bool, float]:
    """Physical-unit query R2 of one decode plus the physical-unit check (same decode as ``Ep.r2``)."""
    import torch
    from scripts import run_population_calib as rpc
    from ssm_decode import fwd_refine as fr
    pred = fr.decode_with_readout(ep_.base, ep_.xz, table, v_aug)
    if not bool(torch.isfinite(pred).all()):
        return -1.0, False, float("nan")
    full = torch.zeros(ep_.sd.n_bins, pred.shape[1], dtype=pred.dtype, device=pred.device)
    full[:ep_.nb] = pred
    r2, unit = rpc.r2_physical(ep_.world, ep_.sd, full, ep_.q)
    return float(r2), bool(unit["passed"]), float(unit["unit_ratio"])


# ------------------------------------------------------------------------------------- stage: jacobian

def compute_jacobian(world, windows: int, width: int, lmax: int) -> dict:
    """Averaged trunk Jacobian per TRUNK SOURCE session (``world.orig_ids``; offline, autograd, source only).

    Same procedure as ``run_pc3_refine.cmd_jacobian`` (windows drawn with ``default_rng(0)`` in session order).
    The held-out pool[-1] is not a trunk source and is not used."""
    import numpy as np
    import torch
    from ssm_decode import fwd_refine as fr
    rng = np.random.default_rng(0)
    res = {"lmax": lmax, "sessions": {}, "trunk_sources": list(world.orig_ids)}
    for name in world.orig_ids:
        sd = world.sessions[name]
        j = world.orig_ids.index(name)
        table = torch.from_numpy(world.orig_tables[j]).to(world.device).float()
        valid = sd.valid.cpu().numpy() & (sd.trial_id.cpu().numpy() >= 0)
        mean = sd.x32[sd.valid].mean(0)
        std = sd.x32[sd.valid].std(0).clamp_min(1e-6)
        xz = (sd.x32 - mean) / std
        ends = np.flatnonzero(valid)
        ends = rng.choice(ends, size=min(windows, len(ends)), replace=False)
        s = fr.average_jacobian_session(world.adapter.model.base, xz, table, ends, lmax=lmax, width=width)
        res["sessions"][name] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
        print(f"[exp] jacobian {name[-8:]} n={s['n']}", flush=True)
    return res


def cmd_jacobian(args):
    import torch
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    for bb in bbs:
        guard_backbone(bb)
        rel = f"jacobian/{bb.bid}.pt"
        if st.exists(rel):
            print(f"[exp] jacobian {bb.bid}: done", flush=True)
            continue
        check_backbone_matches(bb, st)
        t0 = time.perf_counter()
        ctx = Ctx(bb, st, args, load_sessions=True)
        blob = compute_jacobian(ctx.world, args.jac_windows, args.jac_width, JAC_LMAX)
        blob["bid"] = bb.bid
        assert set(blob["sessions"]) == set(bb.trunk_sources), "Jacobian must use the trunk sources only"
        tmp = st.w(rel)
        torch.save(blob, tmp)
        st.push(rel)
        print(f"[exp] jacobian {bb.bid} done t={time.perf_counter() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------- stage: select

def source_basis(ctx, pc, st, bb):
    """Fixed PCA basis of the readout features [h, 1]; trunk SOURCE sessions only (P0 seed 0 tables)."""
    import torch
    from ssm_decode import fwd_refine as fr
    world, rpc = ctx.world, ctx.rpc
    base = world.adapter.model.base
    brel = f"basis/{bb.bid}.pt"
    if st.exists(brel):
        return torch.load(st.o(brel), weights_only=False)["U"].to(ctx.device)
    acc, n = None, 0
    for name in world.orig_ids:
        sd, prior = world.sessions[name], world.priors[name]
        idx = rpc.draw_trials(f"expbasis-{name}", 8, 0, sd.bank.n_trials)
        e = pc.p0_episode(world, sd, prior, ctx.p0[0], idx)
        with torch.inference_mode():
            _, h = fr.run_trunk(base, e.xz.unsqueeze(0), e.table0, want_h=True)
        hb = torch.cat([h[0], torch.ones_like(h[0][:, :1])], 1)[(sd.trial_id >= 0) & sd.valid].double()
        m2 = hb.T @ hb / hb.shape[0]
        acc = m2 if acc is None else acc + m2
        n += 1
    u = fr.feature_basis(acc / n, max(pc.RANKS)).float()
    torch.save({"U": u.cpu(), "sessions": list(world.orig_ids)}, st.w(brel))
    st.push(brel)
    return u.to(ctx.device)


def cmd_basis_and_select(args):
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    pc = pc_module()
    for bb in bbs:
        guard_backbone(bb)
        if st.exists(f"select/{bb.bid}.json"):
            print(f"[exp] select {bb.bid}: done", flush=True)
            continue
        if not st.exists(f"jacobian/{bb.bid}.pt"):
            raise SystemExit(f"{bb.bid}: run the jacobian stage before select")
        check_backbone_matches(bb, st)
        ctx = Ctx(bb, st, args, load_sessions=True)
        world, rpc = ctx.world, ctx.rpc
        basis = source_basis(ctx, pc, st, bb)
        name = bb.val_session
        assert name not in [t.session for t in bb.targets] and name not in ep.ALL_FORBIDDEN
        if bb.oos_selection:
            assert name not in bb.trunk_sources, "selection session must be trunk-unseen for a new backbone"
        sd, prior = world.sessions[name], world.priors[name]
        ls, etas, rhos, steps = grid(args, pc)
        # ---- phase 1: P0 / R1-stream / R2 grid on simulated B-trial episodes of the selection session
        rel1, rel3 = f"select/{bb.bid}_episodes.jsonl", f"select/{bb.bid}_r3_episodes.jsonl"
        st.pull(rel1)
        st.pull(rel3)
        path1, path3 = st.w(rel1), st.w(rel3)
        repair_jsonl(path1)
        repair_jsonl(path3)
        seen1 = {(r["seed"], r["B"], r["draw"]) for r in read_jsonl(path1)}
        t0 = time.perf_counter()
        for seed in ep.SEEDS:
            for B in budgets(args):
                for draw in range(args.sel_draws):
                    if (seed, B, draw) in seen1:
                        continue
                    idx = pc.draw_sim(name, f"expsel-s{seed}", B, draw, sd)
                    e = pc.p0_episode(world, sd, prior, ctx.p0[seed], idx, pc.SIM_MAX_BINS)
                    sts = pc.stream_stats(world, sd, prior, ctx.p0[seed], idx, None)
                    r2 = {"p0": e.r2(e.table0)}
                    for kind in er.READOUT_KINDS:
                        for lam in er.RIDGE_LAMS:
                            r2[f"{kind}|{lam}"] = e.r2(e.table0, e.v_from(sts, lam, kind, basis))
                    r2cfg = {}
                    for L in ls:
                        jb = ctx.jbar(L)
                        for eta in etas:
                            for rho in rhos:
                                for S, tab in e.refine(jb, eta, rho, steps).items():
                                    if S in steps:
                                        r2cfg[er.cfg_key(L, eta, rho, S)] = e.r2(tab)
                    append_row_checked(path1, [{"bid": bb.bid, "session": name, "seed": seed, "B": B, "draw": draw,
                                                "r2": r2, "r2cfg": r2cfg}])
                    print(f"[exp] select {bb.bid} s{seed} B={B} d={draw} p0={r2['p0']:.3f} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
            st.push(rel1)
        st.push(rel1)
        rows1 = read_jsonl(path1)
        r2_rule = {B: er.select_r2([r for r in rows1 if r["B"] == B]) for B in budgets(args)}
        # ---- phase 2: R3 = F7-J with the R2 rule of the same B, then the readout ridge
        seen3 = {(r["seed"], r["B"], r["draw"]) for r in read_jsonl(path3)}
        for seed in ep.SEEDS:
            for B in budgets(args):
                rule = r2_rule[B]
                for draw in range(args.sel_draws):
                    if (seed, B, draw) in seen3:
                        continue
                    idx = pc.draw_sim(name, f"expsel-s{seed}", B, draw, sd)
                    e = pc.p0_episode(world, sd, prior, ctx.p0[seed], idx, pc.SIM_MAX_BINS)
                    tab = e.refine(ctx.jbar(rule["L"]), rule["eta"], rule["rho"], [rule["S"]])[rule["S"]]
                    st3 = e.stats(tab)
                    r3 = {f"{kind}|{lam}": e.r2(tab, e.v_from(st3, lam, kind, basis))
                          for kind in er.READOUT_KINDS for lam in er.RIDGE_LAMS}
                    append_row_checked(path3, [{"bid": bb.bid, "session": name, "seed": seed, "B": B, "draw": draw,
                                                "r2sel": e.r2(tab), "r3": r3}])
                    print(f"[exp] select-r3 {bb.bid} s{seed} B={B} d={draw} t={time.perf_counter() - t0:.0f}s",
                          flush=True)
            st.push(rel3)
        st.push(rel3)
        out = er.assemble_selection(bb.bid, name, bb.oos_selection, bb.selection_scope, budgets(args),
                                    rows1, read_jsonl(path3))
        out["grid"] = {"L": list(ls), "eta": list(etas), "rho": list(rhos), "S": list(steps),
                       "sel_draws": args.sel_draws, "seeds": list(ep.SEEDS)}
        jwrite(st.w(f"select/{bb.bid}.json"), out)
        st.push(f"select/{bb.bid}.json")
        print(f"[exp] select {bb.bid} done ({bb.selection_scope}): "
              + json.dumps({B: (v['r1s']['kind'], v['r2']['L'], v['r2']['S'], v['r3']['kind'])
                            for B, v in out['b'].items()}), flush=True)


# ---------------------------------------------------------------------------------------- stage: eval

def cmd_eval(args):
    import numpy as np
    import torch
    from ssm_decode import fwd_refine as fr
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    pc = pc_module()
    og, ob = pc._patch_autograd_counter()
    try:
        for bb in bbs:
            guard_backbone(bb)
            check_backbone_matches(bb, st)
            sel = json.loads(st.o(f"select/{bb.bid}.json").read_text())
            assert sel["selection_session"] == bb.val_session and sel["selection_oos"] == bb.oos_selection
            basis = torch.load(st.o(f"basis/{bb.bid}.pt"), weights_only=False)["U"]
            ctx = Ctx(bb, st, args, load_sessions=False)
            world, rpc = ctx.world, ctx.rpc
            basis = basis.to(ctx.device)
            hash0 = world.adapter._hash()
            head_bytes = int(fr.head_matrix(world.adapter.model.base).numel() * 4)
            targets = [t for t in bb.targets if not args.targets or ep.date_of(t.session) in args.targets.split(",")]
            for t in targets:
                if t.session in ep.ALL_FORBIDDEN:      # BEFORE any data access (the loader refuses as well)
                    raise PermissionError(t.session)
                rel = f"rows/{bb.bid}/{ep.date_of(t.session)}.jsonl"
                st.pull(rel)
                path = st.w(rel)
                repair_jsonl(path)
                seen = {(r["method"], r["seed"], r["B"], r["draw"]) for r in read_jsonl(path)}
                sd, prior = world.target(t.session)
                t0 = time.perf_counter()
                for B in budgets(args):
                    rule = sel["b"][str(B)]
                    r1, r2r, r3r = rule["r1s"], rule["r2"], rule["r3"]
                    jb = ctx.jbar(r2r["L"])
                    for draw in range(args.draws):
                        idx = rpc.draw_trials(t.session, B, draw, sd.bank.n_trials)
                        for seed in ep.SEEDS:
                            need = [m for m in ("p_prior",) + METHODS_SEEDED
                                    if (m, seed, B, draw) not in seen and not (m == "p_prior" and seed != 0)]
                            if not need:
                                continue
                            e0 = pc.p0_episode(world, sd, prior, ctx.p0[seed], idx)
                            used = np.zeros(sd.n_bins, dtype=bool)
                            for i in idx:
                                a, b = sd.trials[i]
                                used[a:b] = True
                            base = {"bid": bb.bid, "subject": bb.subject, "target": t.session,
                                    "date": ep.date_of(t.session), "stratum": t.stratum, "gap_days": t.gap_days,
                                    "trunk_gap_days": bb.trunk_gap_days(t.session), "tier": t.tier,
                                    "apst_dev": t.apst_dev, "n_units": int(sd.n_units),
                                    "B": B, "draw": draw, "seed": seed, "h4_budget": int(B),
                                    "h4_query_disjoint": bool(not (used & e0.q.cpu().numpy()).any()),
                                    "n_query_bins": int(e0.q.sum()), "replay_bins": e0.replay_bins,
                                    "max_trial_len": max(e0.trial_lens), "rule_version": sel["rule_version"],
                                    "selection_session": sel["selection_session"], "selection_oos": sel["selection_oos"],
                                    "selection": sel["selection_scope"], "h1_trunk_hash": hash0[:16]}
                            rows = []
                            pc._BWD["n"] = 0

                            def emit(method, ep_, table, v=None, cfg=None, readout=False):
                                r2, unit_ok, ratio = score(ep_, table, v)
                                nbytes = int(table.numel() * 4)
                                rows.append({**base, "method": method, "r2": r2, "cfg": cfg or {},
                                             "unit_check_passed": unit_ok, "unit_ratio": ratio,
                                             "h5_bytes": nbytes, "h5_within_32kb": nbytes <= 32 * 1024,
                                             "h5_readout_bytes": head_bytes if readout else 0})
                            if "p_prior" in need:
                                epp = pc.prior_episode(world, sd, prior, idx)
                                emit("p_prior", epp, epp.table0)
                            if "p0" in need:
                                emit("p0", e0, e0.table0)
                            if "r1s" in need:
                                sts = pc.stream_stats(world, sd, prior, ctx.p0[seed], idx, None)
                                emit("r1s", e0, e0.table0, e0.v_from(sts, r1["lam"], r1["kind"], basis),
                                     {"kind": r1["kind"], "lam": r1["lam"]}, readout=True)
                            if "r2" in need or "r3" in need:
                                tab = e0.refine(jb, r2r["eta"], r2r["rho"], [r2r["S"]])[r2r["S"]]
                                cfg2 = {k: r2r[k] for k in ("L", "eta", "rho", "S")}
                                if "r2" in need:
                                    emit("r2", e0, tab, None, cfg2)
                                if "r3" in need:
                                    st3 = e0.stats(tab)
                                    emit("r3", e0, tab, e0.v_from(st3, r3r["lam"], r3r["kind"], basis),
                                         {**cfg2, "kind": r3r["kind"], "lam": r3r["lam"]}, readout=True)
                            bwd = pc._BWD["n"]
                            same = bool(world.adapter._hash() == hash0)
                            for r in rows:
                                r["h2_backward_calls"] = bwd
                                r["h1_trunk_unchanged"] = same
                            append_row_checked(path, rows)
                    st.push(rel)
                    print(f"[exp] eval {bb.bid} {ep.date_of(t.session)} B={B} t={time.perf_counter() - t0:.0f}s", flush=True)
                st.push(rel)
                del sd, prior
            assert world.adapter._hash() == hash0, "trunk changed"
    finally:
        torch.autograd.grad, torch.autograd.backward = og, ob


# ------------------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "bundle", "status", "backbone", "p0", "jacobian", "select", "eval",
                 "causal-source", "causal-eval"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(DEFAULT_OUT), help="persistent root (Drive on Colab)")
        s.add_argument("--work", default=None, help="fast local scratch (copied to --out per unit of work)")
        s.add_argument("--bid", default="all", help="comma list of backbone ids, or all")
        s.add_argument("--tier", default="primary", choices=["primary", "all"],
                       help="'all' adds the optional 2016 year targets and needs --include-2016-optional")
        s.add_argument("--include-2016-optional", action="store_true",
                       help="explicit extra flag required by --tier all (2016 sessions are off by default)")
        s.add_argument("--exclude-apst-dev", action="store_true", help="drop the APST dev sub-C targets")
        s.add_argument("--device", default="cuda:0")
        if name == "check":
            s.add_argument("--write", action="store_true")
        if name == "backbone":
            s.add_argument("--ap-steps0", type=int, default=2000)
            s.add_argument("--ap-attempts", type=int, default=4)
            s.add_argument("--ap-val-interval", type=int, default=100)
        if name == "p0":
            s.add_argument("--p0-steps", type=int, default=1000)
            s.add_argument("--p0-min-steps", type=int, default=600)
            s.add_argument("--p0-val-interval", type=int, default=100)
        if name == "jacobian":
            s.add_argument("--jac-windows", type=int, default=600)
            s.add_argument("--jac-width", type=int, default=320)
        if name == "select":
            s.add_argument("--sel-draws", type=int, default=4)
            s.add_argument("--budgets", default=None, help="comma list; default 1,4,8,16,33 (smoke tests only)")
            for g in ("r2-ls", "r2-etas", "r2-rhos", "r2-steps"):
                s.add_argument(f"--{g}", default=None, help="override the P3 R2 grid (smoke tests only)")
        if name == "eval":
            s.add_argument("--draws", type=int, default=ep.N_DRAWS)
            s.add_argument("--targets", default=None, help="comma list of YYYYMMDD")
            s.add_argument("--budgets", default=None, help="comma list; default 1,4,8,16,33 (smoke tests only)")
        if name in ("causal-source", "causal-eval"):
            s.add_argument("--artifacts", default=None,
                           help="where backbone/p0/jacobian/select live (default: --out)")
            s.add_argument("--legacy-select", default=str(ROOT / "results/causal_main_20261019/select"),
                           help="causal-main episode directory merged into the selection pool")
            s.add_argument("--budgets", default=None)
            s.add_argument("--seeds", default=None)
            s.add_argument("--latency", default="0,1")
            s.add_argument("--targets", default=None, help="comma list of YYYYMMDD (causal-eval)")
            s.add_argument("--etas", default=None, help="override the frozen eta grid (smoke only)")
            s.add_argument("--rhos", default=None, help="override the frozen rho grid (smoke only)")
            s.add_argument("--combo-alphas", default=None, help="alphas at which COMBO is scored (smoke only)")
            s.add_argument("--allow-incomplete-pool", action="store_true",
                           help="causal-eval: do not require every earlier held-out to be scored yet")
    args = p.parse_args()
    if args.cmd in ("causal-source", "causal-eval"):
        from scripts.run_expansion_causal import cmd_causal_eval, cmd_causal_source
        {"causal-source": cmd_causal_source, "causal-eval": cmd_causal_eval}[args.cmd](args)
        return
    {"check": cmd_check, "bundle": cmd_bundle, "status": cmd_status, "backbone": cmd_backbone,
     "p0": cmd_p0, "jacobian": cmd_jacobian, "select": cmd_basis_and_select, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
