#!/usr/bin/env python
"""Driver for the 2026-10-16 evidence expansion (sub-C second subject + sub-M days rolling backbones).

One resumable stage per subcommand; every stage is idempotent (finished artefacts are skipped, partial
ones are redone or resumed row by row).  ``--out`` is the persistent root (on Colab: a Drive path),
``--work`` an optional fast local scratch directory: stage outputs are written to ``--work`` and copied to
``--out`` (atomic rename) as soon as a unit of work finishes, so a lost session loses at most one unit.

  check                 protocol audit + electrode coverage per target (CPU, inventory only)
  bundle                Colab data bundle list (files + sizes) -> <repo>/results/analysis/expansion_20261016/colab_bundle.json
  status                which stage of which backbone is done
  backbone --bid B      AP K=10 backbone with the budget-extension rule (2000 -> 4000 -> 8000 -> 16000 steps)
  p0 --bid B            two P0 calibrators (pool O analog) per backbone
  select --bid B        PCA readout basis + source-only selection of the R1-stream (readout kind, lambda) per B
  eval --bid B          rows: p_prior, p0, r1s (R1-stream with the source-selected rule); contiguous decoding

Rules: never final sessions (sub-M 0623/0625/0626; sub-C 20151113/16/17/19/20 and 20151201), never
results/official_evalai/; selection uses source data only (the pool's last session = the P0 validation
session); the target-side path runs under no_grad with zero backward calls (counted per row).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import expansion_protocol as ep                       # noqa: E402

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

def select_backbones(args) -> list[ep.Backbone]:
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


def ap_path(bb: ep.Backbone, st: Store) -> Path:
    if bb.reuse:
        return ROOT / bb.reuse["ap"]
    return st.o(f"backbones/{bb.bid}/final/best.pt")


def p0_dir(bb: ep.Backbone, st: Store, seed: int) -> Path:
    if bb.reuse:
        return ROOT / bb.reuse["p0_root"] / "runs" / f"{bb.reuse['p0_stratum']}_O" / "p0" / f"seed{seed}"
    return st.o(f"runs/{bb.bid}_O/p0/seed{seed}")


def register(bbs, st: Store):
    """Make the backbones visible to the P1 World / driver code (D1_POOLS)."""
    from scripts.run_labeled_calibration import D1_POOLS
    for bb in bbs:
        D1_POOLS[bb.bid] = dict(sources=",".join(bb.pool), targets=[t.session for t in bb.targets],
                                ap=str(ap_path(bb, st)))


# ---------------------------------------------------------------------------------------- stage: check

def cmd_check(args):
    inv = ep.load_inventory()
    bbs = select_backbones(args)
    rows = []
    for bb in bbs:
        audit = ep.validate_backbone(bb, inv)
        pool_elec = set()
        for s in bb.pool:
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
    prim = select_backbones(SimpleNamespace(tier="primary", exclude_apst_dev=False, bid="all"))
    allb = select_backbones(SimpleNamespace(tier="all", exclude_apst_dev=False, bid="all"))
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
        "all_files": len(subc["all"]), "all_bytes": sum(f["size_bytes"] for f in subc["all"]),
        "note": "ships ONLY allowed sessions; the six APST held-out sessions are never listed"}
    out["subM_needed_in_existing_bundle"] = sorted(f["file"] for f in out["primary"]["files"] if f["subject"] == "sub-M")
    jwrite(EXPERIMENT_FILES / "colab_bundle.json", out)
    (EXPERIMENT_FILES / "colab_bundle_subC_primary.txt").write_text(
        "\n".join(f"sub-C/{f['file']}" for f in subc["primary"]) + "\n")
    (EXPERIMENT_FILES / "colab_bundle_subC_all.txt").write_text(
        "\n".join(f"sub-C/{f['file']}" for f in subc["all"]) + "\n")
    print(json.dumps({k: out[k] for k in ("new_bundle_subC", "reused_artefacts", "subM_needed_in_existing_bundle")}, indent=1))


# --------------------------------------------------------------------------------------- stage: status

def stage_flags(bb: ep.Backbone, st: Store) -> dict:
    nrows = 0
    for t in bb.targets:
        f = st.o(f"rows/{bb.bid}/{ep.date_of(t.session)}.jsonl")
        if f.exists():
            nrows += sum(1 for _ in f.open())
    per_target = len(ep.B_LIST) * ep.N_DRAWS * (1 + 2 * len(ep.SEEDS))
    return {"backbone": bool(bb.reuse) or st.exists(f"backbones/{bb.bid}/backbone.json"),
            "p0": all((p0_dir(bb, st, s) / "done.json").exists() for s in ep.SEEDS),
            "select": st.exists(f"select/{bb.bid}.json"),
            "rows": f"{nrows}/{per_target * len(bb.targets)}"}


def cmd_status(args):
    st = Store(args.out, args.work)
    for bb in select_backbones(args):
        print(bb.bid, json.dumps(stage_flags(bb, st)))


# ------------------------------------------------------------------------------------- stage: backbone

def ap_cmd(bb: ep.Backbone, dest: Path, steps: int, args) -> list[str]:
    return [sys.executable, "-m", "ssm_decode.ap_pretraining", "--task", "dandi688sm",
            "--target-session", bb.targets[0].session, "--dandi688-sources", ",".join(bb.pool),
            "--output", str(dest), "--device", args.device, "--k", str(bb.k), "--width", "128",
            "--layers", "2", "--state-size", "32", "--steps", str(steps),
            "--val-interval", str(args.ap_val_interval)]


def cmd_backbone(args):
    st = Store(args.out, args.work)
    for bb in select_backbones(args):
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
                              "seconds_total": round(time.perf_counter() - t0, 1)})
        st.push(marker)
        print(f"[exp] backbone {bb.bid} converged={converged} best={man['best_step']}/{man['args']['steps']}", flush=True)


# ------------------------------------------------------------------------------------------ stage: p0

def check_backbone_matches(bb: ep.Backbone, st: Store):
    import torch
    blob = torch.load(ap_path(bb, st), map_location="cpu", weights_only=False)
    if list(blob["source_sessions"]) != list(bb.pool):
        raise SystemExit(f"{bb.bid}: checkpoint sources {blob['source_sessions']} != protocol pool {list(bb.pool)}")
    if int(blob["args"]["k"]) != bb.k:
        raise SystemExit(f"{bb.bid}: checkpoint K={blob['args']['k']} != protocol K={bb.k}")


def cmd_p0(args):
    import torch
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    from scripts import run_population_calib as rpc
    for bb in bbs:
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
    """World + P0 calibrators + readout basis of one backbone (target-side no_grad)."""

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


def budgets(args) -> tuple[int, ...]:
    return tuple(int(x) for x in args.budgets.split(",")) if getattr(args, "budgets", None) else ep.B_LIST


def pc_module():
    from scripts import run_pc3_refine as pc
    return pc


def cmd_basis_and_select(args):
    import numpy as np
    import torch
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    pc = pc_module()
    from ssm_decode import fwd_refine as fr
    for bb in bbs:
        if st.exists(f"select/{bb.bid}.json"):
            print(f"[exp] select {bb.bid}: done", flush=True)
            continue
        check_backbone_matches(bb, st)
        ctx = Ctx(bb, st, args, load_sessions=True)
        world, rpc = ctx.world, ctx.rpc
        base = world.adapter.model.base
        # fixed source PCA basis of the readout features (pool sessions, P0 seed 0 tables)
        brel = f"basis/{bb.bid}.pt"
        if st.exists(brel):
            basis = torch.load(st.o(brel), weights_only=False)["U"].to(ctx.device)
        else:
            acc, n = None, 0
            for name in world.ids:
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
            torch.save({"U": u.cpu()}, st.w(brel))
            st.push(brel)
            basis = u.to(ctx.device)
        # source-only selection on the pool validation session (never a target of this backbone)
        name = bb.val_session
        assert name not in [t.session for t in bb.targets]
        sd, prior = world.sessions[name], world.priors[name]
        rel = f"select/{bb.bid}_episodes.jsonl"
        st.pull(rel)
        path = st.w(rel)
        seen = set()
        if path.exists():
            seen = {(r["seed"], r["B"], r["draw"]) for r in map(json.loads, path.read_text().splitlines())}
        t0 = time.perf_counter()
        for seed in ep.SEEDS:
            for B in budgets(args):
                for draw in range(args.sel_draws):
                    if (seed, B, draw) in seen:
                        continue
                    idx = pc.draw_sim(name, f"expsel-s{seed}", B, draw, sd)
                    e = pc.p0_episode(world, sd, prior, ctx.p0[seed], idx, pc.SIM_MAX_BINS)
                    sts = pc.stream_stats(world, sd, prior, ctx.p0[seed], idx, None)
                    r2 = {"p0": e.r2(e.table0)}
                    for kind in pc.READOUT_KINDS:
                        for lam in pc.RIDGE_LAMS:
                            r2[f"{kind}|{lam}"] = e.r2(e.table0, e.v_from(sts, lam, kind, basis))
                    with path.open("a") as f:
                        f.write(json.dumps({"bid": bb.bid, "session": name, "seed": seed, "B": B, "draw": draw,
                                            "r2": r2}) + "\n")
                    print(f"[exp] select {bb.bid} s{seed} B={B} d={draw} p0={r2['p0']:.3f} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
            st.push(rel)
        st.push(rel)
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        sel = {}
        for B in budgets(args):
            sub = [r for r in rows if r["B"] == B]
            per_kind = {}
            for kind in pc.READOUT_KINDS:
                best = max(pc.RIDGE_LAMS, key=lambda lam: np.mean([r["r2"][f"{kind}|{lam}"] for r in sub]))
                per_kind[kind] = {"lam": best, "source_r2": float(np.mean([r["r2"][f"{kind}|{best}"] for r in sub]))}
            kind = max(per_kind, key=lambda k: per_kind[k]["source_r2"])
            sel[str(B)] = {"kind": kind, "lam": per_kind[kind]["lam"], "source_r2": per_kind[kind]["source_r2"],
                           "p0_source_r2": float(np.mean([r["r2"]["p0"] for r in sub])),
                           "n_episodes": len(sub), "per_kind": per_kind}
        out = {"bid": bb.bid, "selection_session": name, "selection_scope": "pool validation session (source only)",
               "b": sel}
        jwrite(st.w(f"select/{bb.bid}.json"), out)
        st.push(f"select/{bb.bid}.json")
        print(f"[exp] select {bb.bid} done: " + json.dumps({B: (v['kind'], v['lam']) for B, v in sel.items()}), flush=True)


def cmd_eval(args):
    import numpy as np
    import torch
    st = Store(args.out, args.work)
    bbs = select_backbones(args)
    register(bbs, st)
    pc = pc_module()
    og, ob = pc._patch_autograd_counter()
    try:
        for bb in bbs:
            check_backbone_matches(bb, st)
            sel = json.loads(st.o(f"select/{bb.bid}.json").read_text())
            assert sel["selection_session"] == bb.val_session
            basis = torch.load(st.o(f"basis/{bb.bid}.pt"), weights_only=False)["U"]
            ctx = Ctx(bb, st, args, load_sessions=False)
            world, rpc = ctx.world, ctx.rpc
            basis = basis.to(ctx.device)
            hash0 = world.adapter._hash()
            targets = [t for t in bb.targets if not args.targets or ep.date_of(t.session) in args.targets.split(",")]
            for t in targets:
                rel = f"rows/{bb.bid}/{ep.date_of(t.session)}.jsonl"
                st.pull(rel)
                path = st.w(rel)
                seen = set()
                if path.exists():
                    seen = {(r["method"], r["seed"], r["B"], r["draw"]) for r in map(json.loads, path.read_text().splitlines())}
                sd, prior = world.target(t.session)
                if t.session in ep.ALL_FORBIDDEN:      # defence in depth; the loader already refuses
                    raise PermissionError(t.session)
                t0 = time.perf_counter()
                for B in budgets(args):
                    rule = sel["b"][str(B)]
                    for draw in range(args.draws):
                        idx = rpc.draw_trials(t.session, B, draw, sd.bank.n_trials)
                        for seed in ep.SEEDS:
                            need = [m for m in ("p_prior", "p0", "r1s")
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
                                    "tier": t.tier, "apst_dev": t.apst_dev, "n_units": int(sd.n_units),
                                    "B": B, "draw": draw, "seed": seed, "h4_budget": int(B),
                                    "h4_query_disjoint": bool(not (used & e0.q.cpu().numpy()).any()),
                                    "n_query_bins": int(e0.q.sum()), "selection": sel["selection_scope"]}
                            rows = []
                            pc._BWD["n"] = 0
                            if "p_prior" in need:
                                epp = pc.prior_episode(world, sd, prior, idx)
                                rows.append({**base, "method": "p_prior", "r2": epp.r2(epp.table0), "cfg": {}})
                            if "p0" in need:
                                rows.append({**base, "method": "p0", "r2": e0.r2(e0.table0), "cfg": {}})
                            if "r1s" in need:
                                sts = pc.stream_stats(world, sd, prior, ctx.p0[seed], idx, None)
                                rows.append({**base, "method": "r1s",
                                             "r2": e0.r2(e0.table0, e0.v_from(sts, rule["lam"], rule["kind"], basis)),
                                             "cfg": {"kind": rule["kind"], "lam": rule["lam"]}})
                            for r in rows:
                                r["h2_backward_calls"] = pc._BWD["n"]
                            with path.open("a") as f:
                                for r in rows:
                                    f.write(json.dumps(r) + "\n")
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
    for name in ("check", "bundle", "status", "backbone", "p0", "select", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(DEFAULT_OUT), help="persistent root (Drive on Colab)")
        s.add_argument("--work", default=None, help="fast local scratch (copied to --out per unit of work)")
        s.add_argument("--bid", default="all", help="comma list of backbone ids, or all")
        s.add_argument("--tier", default="primary", choices=["primary", "all"])
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
        if name == "select":
            s.add_argument("--sel-draws", type=int, default=4)
            s.add_argument("--budgets", default=None, help="comma list; default 1,4,8,16,33 (smoke tests only)")
        if name == "eval":
            s.add_argument("--draws", type=int, default=ep.N_DRAWS)
            s.add_argument("--targets", default=None, help="comma list of YYYYMMDD")
            s.add_argument("--budgets", default=None, help="comma list; default 1,4,8,16,33 (smoke tests only)")
    args = p.parse_args()
    {"check": cmd_check, "bundle": cmd_bundle, "status": cmd_status, "backbone": cmd_backbone,
     "p0": cmd_p0, "select": cmd_basis_and_select, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
