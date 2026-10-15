#!/usr/bin/env python
"""P3: deployable forward-only refinements of P0 (GPU0).

  jacobian   offline average trunk Jacobian kernels on SOURCE sessions (autograd, source only)
  select     source simulation (leave-one-out prior, P0 start, simulated B-trial calibration
             scored on the other trials of the same source session) for R1/R2/R3
  eval       target rows for every candidate with the source-selected rules, plus the
             gradient-alignment diagnostic (autograd on target trunk, NOT used for calibration)

Target-side candidate paths run under torch.no_grad (frozen-trunk forward replay of the B
labeled trials only).  Never touches FINAL sessions or results/official_evalai/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("PC_OUT", str(ROOT / "results/population_calib_p3_20261016")))
os.environ["PC_OUT"] = str(OUT)

from scripts import run_population_calib as rpc                           # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                      # noqa: E402
from ssm_decode import fwd_refine as fr                                   # noqa: E402
from ssm_decode.population_calib import PopStats, norm_stats             # noqa: E402

LMAX = 32
LS = (8, 16, 32)
STEPS = (1, 3, 5, 10)
SEEDS = (0, 1)
SEL_DRAWS = 2                      # per seed -> 4 P0 episodes per (session, B)
RIDGE_LAMS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0)
RANKS = (8, 16, 32)
READOUT_KINDS = ("full", "diag", "lr8", "lr16", "lr32")


def canon(stratum):
    """month and year share the trunk and the six source sessions."""
    return stratum


def load_world(stratum, load_sessions):
    return rpc.World(stratum, "O", torch.device("cuda:0"), load_sessions=load_sessions)


def load_p0(world, stratum):
    out = {}
    for seed in SEEDS:
        out[seed] = rpc.load_run(rpc.P1_OUT / "runs" / f"{stratum}_O" / "p0" / f"seed{seed}", world)[0]
        assert out[seed].prior_mode == world.prior_mode, (out[seed].prior_mode, world.prior_mode)
    return out


# ----------------------------------------------------------------- jacobian ----

def cmd_jacobian(args):
    world = load_world(args.stratum, True)
    base = world.adapter.model.base
    rng = np.random.default_rng(0)
    res = {"stratum": args.stratum, "lmax": LMAX, "sessions": {}}
    for name in world.ids:
        sd = world.sessions[name]
        j = world.orig_ids.index(name)
        table = torch.from_numpy(world.orig_tables[j]).to(world.device).float()
        valid = sd.valid.cpu().numpy() & (sd.trial_id.cpu().numpy() >= 0)
        mean = sd.x32[sd.valid].mean(0)
        std = sd.x32[sd.valid].std(0).clamp_min(1e-6)
        xz = (sd.x32 - mean) / std
        ends = np.flatnonzero(valid)
        ends = rng.choice(ends, size=min(args.windows, len(ends)), replace=False)
        s = fr.average_jacobian_session(base, xz, table, ends, lmax=LMAX, width=args.width)
        res["sessions"][name] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
        rel = fr.jacobian_representativeness(s, s["n"])
        print(f"[jac {args.stratum}] {name[-8:]} n={s['n']} rel_dev lag0/8/16/32 = "
              f"{rel[0]:.2f}/{rel[8]:.2f}/{rel[16]:.2f}/{rel[32]:.2f}  |J|_F lag0 "
              f"{float((s['sum'][0] / s['n']).norm()):.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(res, OUT / f"jacobian_{args.stratum}.pt")


def jbar_for(stratum, lmax, exclude=None):
    """Average Jacobian kernel [lmax+1, D, K]; ``exclude`` drops one source session (LOO)."""
    blob = torch.load(OUT / f"jacobian_{stratum}.pt", weights_only=False)
    tot, n = 0, 0
    for name, s in blob["sessions"].items():
        if name == exclude:
            continue
        tot = tot + s["sum"]
        n += s["n"]
    return (tot / n)[:lmax + 1].float().cuda()


def rep_summary(stratum):
    blob = torch.load(OUT / f"jacobian_{stratum}.pt", weights_only=False)
    tot, sq, n = 0, 0, 0
    for s in blob["sessions"].values():
        tot, sq, n = tot + s["sum"], sq + s["sq"], n + s["n"]
    return fr.jacobian_representativeness({"sum": tot, "sq": sq}, n)



# --------------------------------------------------------------- episodes ----

ETAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
RHOS = (0.0, 0.03, 0.1, 0.3)           # rho = eta * lam, the per-step shrink toward the P0 table
EPS_Q = 1e-3
_BWD = {"n": 0}


def _patch_autograd_counter():
    og, ob = torch.autograd.grad, torch.autograd.backward

    def g(*a, **k):
        _BWD["n"] += 1
        return og(*a, **k)

    def b(*a, **k):
        _BWD["n"] += 1
        return ob(*a, **k)
    torch.autograd.grad, torch.autograd.backward = g, b
    return og, ob


class Ep:
    """One simulated/real B-trial calibration episode on one session."""

    def __init__(self, world, sd, idx, table0, mean, std, max_bins=None):
        self.world, self.sd, self.idx = world, sd, np.asarray(idx)
        self.base = world.adapter.model.base
        self.table0, self.mean, self.std = table0.float(), mean.float(), std.float()
        self.xz_all = (sd.x32 - self.mean) / self.std
        self.nb = sd.n_bins if max_bins is None else min(sd.n_bins, int(max_bins))
        self.xz = self.xz_all[:self.nb]                        # decode (and score) only the first nb bins
        wins = [sd.trials[i] for i in self.idx]
        self.windows = wins
        self.X, self.Y, self.M = fr.pad_trials(self.xz_all, sd.y_norm.float(), sd.valid, wins)
        self.q = sd.query_mask(self.idx)
        self.q[self.nb:] = False
        self.replay_bins = int(sum(b - a for a, b in wins))

    def r2(self, table, v_aug=None):
        pred = fr.decode_with_readout(self.base, self.xz, table, v_aug)
        if not bool(torch.isfinite(pred).all()):
            return -1.0
        full = torch.zeros(self.sd.n_bins, pred.shape[1], dtype=pred.dtype, device=pred.device)
        full[:self.nb] = pred
        return rpc.r2_physical(self.world, self.sd, full, self.q)[0]

    def stats(self, table):
        return fr.readout_stats(self.base, self.X, self.Y, self.M, table)

    def v_from(self, st, lam, kind, basis):
        if kind in ("full", "diag"):
            d = fr.ridge_delta(st, lam, kind)
        else:
            r = int(kind[2:])
            d = fr.ridge_delta(st, lam, "lowrank", basis=basis[:, :r])
        return fr.head_matrix(self.base).double() + d

    def refine(self, jbar, eta, rho, snaps, with_diag=None):
        return fr.fj_refine(self.base, self.X, self.Y, self.M, self.table0, jbar, eta,
                            (rho / eta) if eta > 0 else 0.0, snaps, eps=EPS_Q, with_diag=with_diag)


def p0_episode(world, sd, prior, calib, idx, max_bins=None):
    res = calib.calibrate(PopStats.from_trials(sd.bank, idx), prior)
    return Ep(world, sd, idx, res.table, res.mean, res.std, max_bins)


def prior_episode(world, sd, prior, idx, max_bins=None):
    mean, std = norm_stats(PopStats.from_trials(sd.bank, idx))
    return Ep(world, sd, idx, prior.float(), mean, std, max_bins)


SIM_MAX_BINS = 50000
OOS_EXTRA = ["sub-M_ses-CO-20140203", "sub-M_ses-CO-20140217", "sub-M_ses-CO-20140218",
             "sub-M_ses-CO-20140303", "sub-M_ses-CO-20140304", "sub-M_ses-CO-20140306",
             "sub-M_ses-CO-20140307", "sub-M_ses-CO-20140627", "sub-M_ses-CO-20140929",
             "sub-M_ses-CO-20141203"]


def sim_sessions(world, stratum, sim_set):
    """Names of the source sessions used for the source simulation.

    ``is``: the six trunk-training sessions (in-sample for trunk and P0 calibrator, mostly).
    ``oos``: source sessions the trunk and the P0 calibrator never saw (pool-E extras),
    never a D1 target / dev / final session.  Days: earlier than the target.  Month/year: the
    extras dated 2014-03-07 .. 2014-12-03 (for month this includes sessions LATER than its
    target, documented in the README).
    """
    from ssm_decode.dandi688_data import load_dandi688_recording
    if sim_set == "is":
        return list(world.ids)
    banned = {t for st in D1_POOLS.values() for t in st["targets"]}
    names = [n for n in OOS_EXTRA if n not in set(world.orig_ids) | banned]
    if stratum != "days":
        names = [n for n in names if n.endswith(("0307", "0627", "0929", "1203"))]
    for n in names:
        if n not in world.sessions:
            world.add_session(n, load_dandi688_recording(n))
    return names


def draw_sim(name, tag, B, draw, sd):
    """Trial subset for a simulated calibration: trials inside the first SIM_MAX_BINS bins."""
    n_in = int(sum(1 for a, b in sd.trials if b <= SIM_MAX_BINS))
    return rpc.draw_trials(f"{tag}-{name}", B, draw, n_in)


def stream_stats(world, sd, prior, calib, idx):
    """Cheaper alternative: readout statistics accumulated while the B trials stream in.

    Trial j (chronological) is forwarded ONCE with the table/normalisation produced from trials
    < j (trial 0: prior table, normalisation from the trial itself = a one-trial buffer);
    no replay buffer of earlier trials is kept.  Returns the accumulated stats."""
    base = world.adapter.model.base
    idx = np.sort(np.asarray(idx))
    tot = None
    for j, tr in enumerate(idx):
        if j == 0:
            mean, std = norm_stats(PopStats.from_trials(sd.bank, idx[:1]))
            table = prior.float()
        else:
            res = calib.calibrate(PopStats.from_trials(sd.bank, idx[:j]), prior)
            mean, std, table = res.mean, res.std, res.table.float()
        a, b = sd.trials[tr]
        xz = (sd.x32[a:b] - mean.float()) / std.float()
        X, Y, M = fr.pad_trials(xz, sd.y_norm.float()[a:b], sd.valid[a:b], [(0, b - a)])
        st = fr.readout_stats(base, X, Y, M, table)
        tot = st if tot is None else fr.add_readout_stats(tot, st)
    return tot


def load_basis(stratum):
    return torch.load(OUT / f"basis_{stratum}.pt", weights_only=False)["U"].cuda()


def cmd_basis(args):
    """Fixed source PCA basis of the readout features [h, 1] (low-rank readout ridge)."""
    world = load_world(args.stratum, True)
    p0 = load_p0(world, args.stratum)[0]
    base = world.adapter.model.base
    acc, nsess = None, 0
    for name in world.ids:
        sd, prior = world.sessions[name], world.priors[name]
        idx = rpc.draw_trials(f"p3basis-{name}", 8, 0, sd.bank.n_trials)
        ep = p0_episode(world, sd, prior, p0, idx)
        with torch.inference_mode():
            _, h = fr.run_trunk(base, ep.xz.unsqueeze(0), ep.table0, want_h=True)
        hb = torch.cat([h[0], torch.ones_like(h[0][:, :1])], 1)[(sd.trial_id >= 0) & sd.valid].double()
        m2 = hb.T @ hb / hb.shape[0]
        acc = m2 if acc is None else acc + m2
        nsess += 1
    u = fr.feature_basis(acc / nsess, max(RANKS)).float()
    ev = torch.linalg.eigvalsh(acc / nsess).flip(0)
    cum = (ev.cumsum(0) / ev.sum())[[7, 15, 31]]
    torch.save({"U": u.cpu(), "cum_energy_8_16_32": cum.cpu()}, OUT / f"basis_{args.stratum}.pt")
    print(f"[basis {args.stratum}] cumulative energy r=8/16/32: {cum.cpu().numpy().round(4)}")


# ---------------------------------------------------------------- selection ----

def sel_path(stratum, sim_set, stage):
    return OUT / "select" / f"{stage}_{stratum}_{sim_set}.jsonl"


def seen_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add((r["session"], r["seed"], r["B"], r["draw"]))
    return keys


def sim_world(stratum, sim_set):
    world = load_world(stratum, sim_set == "is")
    names = sim_sessions(world, stratum, sim_set)
    return world, names


def jb_for_sim(stratum, L, name, sim_set):
    return jbar_for(stratum, L, exclude=name if sim_set == "is" else None)


def cmd_select(args):
    world, names = sim_world(args.stratum, args.set)
    p0 = load_p0(world, args.stratum)
    basis = load_basis(args.stratum)
    path = sel_path(args.stratum, args.set, "r12")
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = seen_keys(path)
    t0 = time.perf_counter()
    for name in names:
        sd, prior = world.sessions[name], world.priors[name]
        jb_cache = {L: jb_for_sim(args.stratum, L, name, args.set) for L in LS}
        for seed in SEEDS:
            for B in rpc.B_GRID:
                for draw in range(SEL_DRAWS):
                    if (name, seed, B, draw) in seen:
                        continue
                    idx = draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                    ep = p0_episode(world, sd, prior, p0[seed], idx, SIM_MAX_BINS)
                    epp = prior_episode(world, sd, prior, idx, SIM_MAX_BINS)
                    rows = []

                    def add(method, cfg, r2):
                        rows.append({"stratum": args.stratum, "set": args.set, "session": name, "seed": seed,
                                     "B": B, "draw": draw, "method": method, "cfg": cfg, "r2": r2})
                    add("p0", {}, ep.r2(ep.table0))
                    add("prior", {}, epp.r2(epp.table0))
                    st, stp = ep.stats(ep.table0), epp.stats(epp.table0)
                    sts = stream_stats(world, sd, prior, p0[seed], idx)
                    for kind in READOUT_KINDS:
                        for lam in RIDGE_LAMS:
                            add(f"r1_{kind}", {"lam": lam}, ep.r2(ep.table0, ep.v_from(st, lam, kind, basis)))
                            add(f"r1p_{kind}", {"lam": lam}, epp.r2(epp.table0, epp.v_from(stp, lam, kind, basis)))
                            add(f"r1s_{kind}", {"lam": lam}, ep.r2(ep.table0, ep.v_from(sts, lam, kind, basis)))
                    for L in LS:
                        for eta in ETAS:
                            for rho in RHOS:
                                snaps = ep.refine(jb_cache[L], eta, rho, STEPS)
                                for S, tab in snaps.items():
                                    if S in STEPS:
                                        add("r2", {"L": L, "eta": eta, "rho": rho, "S": S}, ep.r2(tab))
                    with path.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[select {args.stratum}/{args.set}] {name[-8:]} s{seed} B={B} d={draw} "
                          f"p0={rows[0]['r2']:.3f} t={time.perf_counter() - t0:.0f}s", flush=True)
    write_selection(args.stratum, args.set, "r12")


def _load_rows(path):
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines()])
    df["cfgk"] = df.cfg.map(lambda c: json.dumps(c, sort_keys=True))
    df["r2"] = df.r2.fillna(-1.0)
    return df


def write_selection(stratum, sim_set, stage):
    df = _load_rows(sel_path(stratum, sim_set, stage))
    p = OUT / "select" / f"selection_{stratum}_{sim_set}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    for (method, B), sub in df.groupby(["method", "B"]):
        g = sub.groupby("cfgk").r2.agg(["mean", "count"])
        best = g["mean"].idxmax()
        old.setdefault(method, {})[str(B)] = {"cfg": json.loads(best), "source_r2": float(g.loc[best, "mean"]),
                                              "n_episodes": int(g.loc[best, "count"])}
    p.write_text(json.dumps(old, indent=1))
    df.groupby(["method", "B", "cfgk"]).r2.agg(["mean", "count"]).reset_index().to_csv(
        OUT / "select" / f"grid_{stage}_{stratum}_{sim_set}.csv", index=False)
    return old


def cmd_select_r3(args):
    """Source simulation of R3 = F7-J (rule selected in R2, per B) then readout ridge."""
    world, names = sim_world(args.stratum, args.set)
    p0 = load_p0(world, args.stratum)
    basis = load_basis(args.stratum)
    sel = json.loads((OUT / "select" / f"selection_{args.stratum}_{args.set}.json").read_text())
    path = sel_path(args.stratum, args.set, "r3")
    seen = seen_keys(path)
    t0 = time.perf_counter()
    for name in names:
        sd, prior = world.sessions[name], world.priors[name]
        for seed in SEEDS:
            for B in rpc.B_GRID:
                rule = sel["r2"][str(B)]["cfg"]
                jb = jb_for_sim(args.stratum, rule["L"], name, args.set)
                for draw in range(SEL_DRAWS):
                    if (name, seed, B, draw) in seen:
                        continue
                    idx = draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                    ep = p0_episode(world, sd, prior, p0[seed], idx, SIM_MAX_BINS)
                    tab = ep.refine(jb, rule["eta"], rule["rho"], [rule["S"]])[rule["S"]]
                    st = ep.stats(tab)
                    base_row = {"stratum": args.stratum, "set": args.set, "session": name, "seed": seed,
                                "B": B, "draw": draw}
                    rows = [{**base_row, "method": "r2sel", "cfg": {}, "r2": ep.r2(tab)}]
                    for kind in READOUT_KINDS:
                        for lam in RIDGE_LAMS:
                            rows.append({**base_row, "method": f"r3_{kind}", "cfg": {"lam": lam},
                                         "r2": ep.r2(tab, ep.v_from(st, lam, kind, basis))})
                    with path.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"[select-r3 {args.stratum}/{args.set}] {name[-8:]} s{seed} B={B} d={draw} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
    write_selection(args.stratum, args.set, "r3")


# --------------------------------------------------------------------- eval ----

def eval_path(stratum, sim_set):
    return OUT / "eval" / f"eval_{stratum}_{sim_set}.jsonl"


def diag_path(stratum, sim_set):
    return OUT / "eval" / f"diag_{stratum}_{sim_set}.jsonl"


def jsonl_keyset(path, fields):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            keys.add(tuple(r[f] for f in fields))
    return keys


def query_batch(ep, n_max=30):
    """Up to ``n_max`` NON-calibration trials as one padded batch (diagnostic only)."""
    sd = ep.sd
    chosen = set(int(i) for i in ep.idx)
    rest = [i for i in range(len(sd.trials)) if i not in chosen]
    rng = np.random.default_rng(zlib_seed(sd.name, len(ep.idx)))
    pick = sorted(rng.choice(rest, size=min(n_max, len(rest)), replace=False))
    return fr.pad_trials(ep.xz_all, sd.y_norm.float(), sd.valid, [sd.trials[i] for i in pick])


def zlib_seed(name, b):
    import zlib
    return zlib.crc32(f"q-{name}-{b}".encode())


def cmd_eval(args):
    og, ob = _patch_autograd_counter()
    world = load_world(args.stratum, False)
    p0 = load_p0(world, args.stratum)
    basis = load_basis(args.stratum)
    sel = json.loads((OUT / "select" / f"selection_{args.stratum}_{args.set}.json").read_text())
    path, dpath = eval_path(args.stratum, args.set), diag_path(args.stratum, args.set)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = jsonl_keyset(path, ("method", "seed", "target", "B", "draw"))
    seen_d = jsonl_keyset(dpath, ("seed", "target", "B", "draw"))
    base = world.adapter.model.base
    hash0 = world.adapter._hash()
    t0 = time.perf_counter()
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for B in rpc.B_GRID:
            rule = sel["r2"][str(B)]["cfg"]
            jb = jbar_for(args.stratum, rule["L"])
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in SEEDS:
                    todo = lambda m: (m, seed, target, B, draw) not in seen    # noqa: E731
                    ep = p0_episode(world, sd, prior, p0[seed], idx)
                    used = np.zeros(sd.n_bins, dtype=bool)
                    for i in idx:
                        a, b = sd.trials[i]
                        used[a:b] = True
                    disjoint = bool(not (used & ep.q.cpu().numpy()).any())
                    base_row = {"stratum": args.stratum, "set": args.set, "pool": "O", "seed": seed,
                                "target": target, "B": B, "draw": draw, "h4_budget": int(B),
                                "h4_query_disjoint": disjoint, "replay_bins": ep.replay_bins}
                    rows = []

                    def emit(method, r2, cfg, passes, **extra):
                        rows.append({**base_row, "method": method, "r2": r2, "cfg": cfg,
                                     "replay_passes": passes, "h2_backward_calls": _BWD["n"],
                                     "h2_zero_backprop": _BWD["n"] == 0, **extra})
                    _BWD["n"] = 0
                    if todo("p0"):
                        emit("p0", ep.r2(ep.table0), {}, 0)
                    # R1: readout ridge on the P0 table (replay forward of the B trials, once)
                    need = [k for k in READOUT_KINDS if todo(f"r1_{k}") or todo(f"r3_{k}")]
                    if any(todo(f"r1_{k}") for k in READOUT_KINDS):
                        st = ep.stats(ep.table0)
                        for k in READOUT_KINDS:
                            if todo(f"r1_{k}"):
                                lam = sel[f"r1_{k}"][str(B)]["cfg"]["lam"]
                                emit(f"r1_{k}", ep.r2(ep.table0, ep.v_from(st, lam, k, basis)),
                                     {"lam": lam}, 1)
                    # readout-only on the P-prior table (no read-in calibration); seed-free
                    if seed == 0 and any(todo(f"r1p_{k}") for k in READOUT_KINDS):
                        epp = prior_episode(world, sd, prior, idx)
                        stp = epp.stats(epp.table0)
                        emit("p_prior", epp.r2(epp.table0), {}, 0)
                        for k in READOUT_KINDS:
                            if todo(f"r1p_{k}"):
                                lam = sel[f"r1p_{k}"][str(B)]["cfg"]["lam"]
                                emit(f"r1p_{k}", epp.r2(epp.table0, epp.v_from(stp, lam, k, basis)),
                                     {"lam": lam}, 1)
                    # R1s: streaming readout statistics (no replay buffer)
                    if any(todo(f"r1s_{k}") for k in READOUT_KINDS):
                        sts = stream_stats(world, sd, prior, p0[seed], idx)
                        for k in READOUT_KINDS:
                            if todo(f"r1s_{k}"):
                                lam = sel[f"r1s_{k}"][str(B)]["cfg"]["lam"]
                                emit(f"r1s_{k}", ep.r2(ep.table0, ep.v_from(sts, lam, k, basis)),
                                     {"lam": lam}, 0, streaming_passes=1)
                    # R2: F7-J
                    tab = None
                    if todo("r2") or any(todo(f"r3_{k}") for k in READOUT_KINDS):
                        tab = ep.refine(jb, rule["eta"], rule["rho"], [rule["S"]])[rule["S"]]
                    if todo("r2"):
                        emit("r2", ep.r2(tab), dict(rule), rule["S"])
                    # R3: F7-J then readout ridge
                    if any(todo(f"r3_{k}") for k in READOUT_KINDS):
                        st3 = ep.stats(tab)
                        for k in READOUT_KINDS:
                            if todo(f"r3_{k}"):
                                lam = sel[f"r3_{k}"][str(B)]["cfg"]["lam"]
                                emit(f"r3_{k}", ep.r2(tab, ep.v_from(st3, lam, k, basis)),
                                     {**rule, "lam": lam}, rule["S"] + 1)
                    if rows:
                        for r in rows:
                            r["trunk_hash_unchanged"] = bool(world.adapter._hash() == hash0) if r["method"] == "p0" else None
                        with path.open("a") as f:
                            for r in rows:
                                f.write(json.dumps(r) + "\n")
                    # DIAGNOSTIC (autograd on the target trunk; never feeds a calibrated table)
                    if draw < args.diag_draws and seed == 0 and (seed, target, B, draw) not in seen_d:
                        drows = diag_rows(world, args.stratum, ep, rule, jb, base_row)
                        with dpath.open("a") as f:
                            for r in drows:
                                f.write(json.dumps(r) + "\n")
                    print(f"[eval {args.stratum}/{args.set}] {target[-4:]} B={B} d={draw} s{seed} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
    assert world.adapter._hash() == hash0, "trunk changed"
    torch.autograd.grad, torch.autograd.backward = og, ob


def diag_rows(world, stratum, ep, rule, jb_rule, base_row):
    """Gradient alignment: cos(F7-J update, true gradient) on the B trials and on query trials."""
    base = ep.base
    Xq, Yq, Mq = query_batch(ep)
    out = []
    # (a) step-1 alignment from the P0 table for every feedback length L
    for L in (0,) + LS:
        jb = jbar_for(stratum, L)
        g, q, _, _ = fr.fj_gradient(base, ep.X, ep.Y, ep.M, ep.table0, jb)
        tb = fr.diagnostic_true_neg_gradient(base, ep.X, ep.Y, ep.M, ep.table0)
        tq = fr.diagnostic_true_neg_gradient(base, Xq, Yq, Mq, ep.table0)
        pre = 1.0 / (q + EPS_Q)[None, :]
        out.append({**base_row, "kind": "p0_step1", "L": L, "step": 1,
                    "cos_B": fr.cosine(g, tb), "cos_B_precond": fr.cosine(g * pre, tb * pre),
                    "cos_query": fr.cosine(g, tq), "cos_true_B_vs_query": fr.cosine(tb, tq),
                    "norm_ratio": float(g.norm() / tb.norm().clamp_min(1e-12))})
    # (b) along the selected trajectory
    rec = []

    def hook(step, tab, g):
        tb = fr.diagnostic_true_neg_gradient(base, ep.X, ep.Y, ep.M, tab)
        tq = fr.diagnostic_true_neg_gradient(base, Xq, Yq, Mq, tab)
        rec.append((step, fr.cosine(g, tb), fr.cosine(g, tq), fr.cosine(tb, tq)))
    ep.refine(jb_rule, rule["eta"], rule["rho"], [rule["S"]], with_diag=hook)
    for step, cb, cq, cbq in rec:
        out.append({**base_row, "kind": "selected_traj", "L": rule["L"], "step": step, "cos_B": cb,
                    "cos_query": cq, "cos_true_B_vs_query": cbq})
    return out


def cmd_ablate(args):
    """Controls for R2 at the source-selected (eta, rho, S) of the headline set (target query is
    used only to REPORT; nothing here selects anything):
      jbar      the selected rule (L as selected)
      sign      feedback kernel -Jbar (must hurt)
      random    kernel with random entries and the same per-lag Frobenius norm as Jbar
      zero      e_t = 0: only the shrinkage towards T0 acts (must be a no-op since T = T0)
      L0/L8/L16 truncated Jbar (feedback window ablation)
    """
    world = load_world(args.stratum, False)
    p0 = load_p0(world, args.stratum)
    sel = json.loads((OUT / "select" / f"selection_{args.stratum}_{args.set}.json").read_text())
    path = OUT / "eval" / f"ablate_{args.stratum}_{args.set}.jsonl"
    seen = jsonl_keyset(path, ("variant", "seed", "target", "B", "draw"))
    t0 = time.perf_counter()
    for target in D1_POOLS[args.stratum]["targets"]:
        sd, prior = world.target(target)
        for B in args.budgets:
            rule = sel["r2"][str(B)]["cfg"]
            jfull = jbar_for(args.stratum, LMAX)
            variants = {"jbar": jfull[:rule["L"] + 1], "sign": -jfull[:rule["L"] + 1],
                        "zero": torch.zeros_like(jfull[:rule["L"] + 1])}
            for l in (0, 8, 16):
                if l < rule["L"]:
                    variants[f"L{l}"] = jfull[:l + 1]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in SEEDS:
                    ep = p0_episode(world, sd, prior, p0[seed], idx)
                    g = torch.Generator(device="cpu").manual_seed(1000 * draw + seed + B)
                    rnd = torch.randn(jfull[:rule["L"] + 1].shape, generator=g).to(jfull.device)
                    jn = jfull[:rule["L"] + 1].flatten(1).norm(dim=1)
                    rnd = rnd * (jn / rnd.flatten(1).norm(dim=1).clamp_min(1e-12))[:, None, None]
                    variants["random"] = rnd
                    for name, jb in variants.items():
                        if (name, seed, target, B, draw) in seen:
                            continue
                        tab = ep.refine(jb, rule["eta"], rule["rho"], [rule["S"]])[rule["S"]]
                        row = {"stratum": args.stratum, "set": args.set, "variant": name, "seed": seed,
                               "target": target, "B": B, "draw": draw, "r2": ep.r2(tab), "p0_r2": ep.r2(ep.table0),
                               "cfg": dict(rule)}
                        with path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                    print(f"[ablate {args.stratum}/{args.set}] {target[-4:]} B={B} d={draw} s{seed} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jacobian")
    j.add_argument("--stratum", required=True, choices=list(D1_POOLS))
    j.add_argument("--windows", type=int, default=600)
    j.add_argument("--width", type=int, default=320)
    for name in ("basis", "select", "select-r3", "eval", "ablate"):
        s = sub.add_parser(name)
        s.add_argument("--stratum", required=True, choices=list(D1_POOLS))
        if name != "basis":
            s.add_argument("--set", required=True, choices=["is", "oos"])
        if name in ("eval", "ablate"):
            s.add_argument("--draws", type=int, default=rpc.N_DRAWS)
        if name == "eval":
            s.add_argument("--diag-draws", type=int, default=4)
        if name == "ablate":
            s.add_argument("--budgets", type=int, nargs="+", default=[2, 8, 33])
    args = p.parse_args()
    {"jacobian": cmd_jacobian, "basis": cmd_basis, "select": cmd_select,
     "select-r3": cmd_select_r3, "eval": cmd_eval, "ablate": cmd_ablate}[args.cmd](args)


if __name__ == "__main__":
    main()
