#!/usr/bin/env python
"""P0-Solve, W_alpha shrinkage, and the projection diagnostic (DAC2027, 2026-10-19).

Subcommands
  fit-anchor   ridge map F (K x p) from source-session trunk latents to phi. Source only.
  select       per-B alpha / lambda / eps on the P3 source-simulation episodes
               (trunk-unseen and earlier than the target: days oosc, month oosc, year oos).
  eval         target rows. Draws via run_population_calib.draw_trials (same RNG as P1/P3/drift).
  diag         projection diagnostic (not a candidate). Needs the ui_reg table, so this path
               backprops through the frozen trunk and is labeled diagnostic_only.
  cost         analytic accumulator / solve cost for N=26 and N=256. No GPU.

Candidate rows never backprop. Hyperparameters are read from the selection json; nothing
is chosen on a target query. Final sessions and 20140307-as-target are rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_pc3_refine as pc                                   # noqa: E402
from scripts import run_population_calib as rpc                            # noqa: E402
from scripts.run_baselines_readin import (CCA_META, RefData, decode_affine,  # noqa: E402
                                          run_method, trial_meta)
from scripts.run_drift_curve import (FINAL_DATES, PREFIX, TARGET_DATES,    # noqa: E402
                                     validate_targets)
from scripts.run_labeled_calibration import D1_POOLS                       # noqa: E402
from ssm_decode import published_baselines as pb                           # noqa: E402
from ssm_decode.p0solve import (ALPHAS, EPS_GRID, F_RIDGE_GRID, LAM_GRID, P_GRID,
                                RawMoments, accumulate_raw, add_moments, apply_readin,
                                cost_model, empty_moments, fit_anchor, map_cca_into_w0_coordinates,
                                mean_std_from_raw, moments_to_normalized, phi_features,
                                project_update, shrink_tables, target_side_solve)
from ssm_decode.population_calib import PopStats, norm_stats               # noqa: E402

OUT = Path(os.environ.get("P0SOLVE_OUT", str(ROOT / "results/p0solve_20261019")))
P3_FIX = ROOT / "results/population_calib_p3_20261016/fix"
P2 = ROOT / "results/population_calib_p2_20261015"
DRIFT_ROWS = ROOT / "results/drift_curve_20261016"
B_LIST = rpc.B_GRID
SEEDS = pc.SEEDS
K = 10
# Source-selected corrected ridge-CCA, raw sorted units (linear in the same counts as W0).
# Sqrt-count CCA is not an affine map of z-scored raw counts, so it cannot share W0's coordinates.
CCA_CFG = {
    "days": {"align": "go", "d": "N", "inp": "units", "phases": 10, "ridge": 3.0, "transform": "raw"},
    "month": {"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "raw"},
    "year": {"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "raw"},
}
# Which P3 source-simulation set supplies the episodes. days/month: causal OOS (oosc).
# year: the four OOS sessions, all earlier than 20150511 (P3's year set; already causal).
SIM_SET = {"days": "oosc", "month": "oosc", "year": "oos"}
UI_RULE = {
    "days": {"file": P2 / "uireg/selection_days.json", "like_for_like": False,
             "note": "P2 rule (selected from the electrode prior on in-sample sources, applied from P0). "
                     "Days has no like-for-like P0-start re-selection."},
    "month": {"file": P3_FIX / "uiref/selection_month_oosc.json", "like_for_like": True,
              "note": "like-for-like P0+ui_reg, causal source {20140307}"},
    # Year has no separate selection file: the trunk is shared, and the evaluated
    # like-for-like rule is month tag ``oos`` (four sessions, all before 20150511).
    "year": {"file": P3_FIX / "uiref/selection_month_oos.json", "like_for_like": True,
             "note": "like-for-like P0+ui_reg; rule_from month tag oos (20140307/0627/0929/1203)"},
}
EVAL_METHODS = ("p_prior", "p0", "w_alpha", "p0solve_p8", "p0solve_p16",
                "prior_solve_p8", "prior_solve_p16", "ridge_cca")
SEED_FREE = {"p_prior", "prior_solve_p8", "prior_solve_p16", "ridge_cca"}
_BATCH_OK = {"checked": False}


def require_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("cuda:0 is not available; stopping (do not fall back to another device)")
    return torch.device("cuda:0")


def append_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()


def seen_keys(path: Path, fields: tuple[str, ...]) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add(tuple(r[f] for f in fields))
    return keys


def cfg_dumps(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def anchor_path(stratum: str) -> Path:
    return OUT / "anchor" / f"{stratum}_F.npz"


def selection_path(stratum: str) -> Path:
    return OUT / "select" / f"selection_{stratum}.json"


def neural_np(sd) -> np.ndarray:
    cached = getattr(sd, "_p0_neural", None)
    if cached is None:
        cached = sd.x.detach().cpu().numpy().astype(np.float64, copy=False)
        sd._p0_neural = cached
    return cached


def conditions(sd):
    """Per-bin CCA reach-direction label and within-trial phase in (-1, 1)."""
    cached = getattr(sd, "_p0_cond", None)
    if cached is not None:
        return cached
    meta = trial_meta(sd.name, sd.record)
    cond = np.full(sd.n_bins, -1, dtype=np.int64)
    phase = np.zeros(sd.n_bins, dtype=np.float64)
    for a, b in sd.trials:
        ang, _go = meta[(a, b)]
        cond[a:b] = pb.condition_label(float(ang))
        length = b - a
        phase[a:b] = (np.arange(length) + 0.5) / length * 2.0 - 1.0
    sd._p0_cond = (cond, phase)
    return cond, phase


def raw_moments(sd, idx, p: int) -> RawMoments:
    x = neural_np(sd)
    valid = sd.valid_np
    cond, phase = conditions(sd)
    raw = empty_moments(sd.n_units, p)
    for i in np.asarray(idx, dtype=np.int64):
        a, b = sd.trials[int(i)]
        phi = phi_features(cond[a:b], phase[a:b], p)
        raw = add_moments(raw, accumulate_raw(x[a:b], phi, valid[a:b]))
    return raw


def norm_np(sd, idx):
    stats = PopStats.from_trials(sd.bank, idx)
    mean, std = norm_stats(stats)
    return (mean.detach().cpu().numpy().astype(np.float64),
            std.detach().cpu().numpy().astype(np.float64), stats)


def check_norm_matches_raw(raw: RawMoments, mean, std, where: str):
    m2, s2 = mean_std_from_raw(raw)
    if np.max(np.abs(m2 - mean)) > 1e-5 or np.max(np.abs(s2 - std)) > 1e-5:
        raise AssertionError(f"{where}: raw-moment norm differs from norm_stats "
                             f"mean {np.max(np.abs(m2 - mean)):.3e} std {np.max(np.abs(s2 - std)):.3e}")


def load_anchor(stratum: str) -> dict:
    blob = np.load(anchor_path(stratum))
    return {p: blob[f"F{p}"] for p in P_GRID} | {"ridge": {p: float(blob[f"ridge{p}"]) for p in P_GRID}}


def source_latent_and_phi(world, name: str, p: int):
    """Latents actually fed to the frozen trunk on one source session, and phi on the same bins."""
    sd = world.sessions[name]
    i = world.orig_ids.index(name)
    stats = world.adapter.blob["source_statistics"][i]
    mean = np.asarray(stats[0], dtype=np.float64)
    std = np.asarray(stats[1], dtype=np.float64)
    if mean.shape != (sd.n_units,) or std.shape != (sd.n_units,):
        raise AssertionError(f"{name}: source norm {mean.shape} vs N={sd.n_units}")
    std = np.where(std < 1e-6, 1.0, std)
    cond, phase = conditions(sd)
    mask = sd.valid_np & (cond >= 0)
    x = neural_np(sd)[mask]
    u = ((x - mean) / std) @ np.asarray(world.orig_tables[i], dtype=np.float64).T
    phi = phi_features(cond[mask], phase[mask], p)
    return u, phi


def cmd_fit_anchor(args):
    device = require_gpu()
    world = rpc.World(args.stratum, "O", device, load_sessions=True)
    assert world.k == K
    blocks = {p: [] for p in P_GRID}
    for name in world.orig_ids:
        for p in P_GRID:
            u, phi = source_latent_and_phi(world, name, p)
            blocks[p].append((name, u, phi))
            print(f"[anchor {args.stratum}] {name[-8:]} p={p} bins={len(u)}", flush=True)
    chosen = {}
    arrays = {}
    for p in P_GRID:
        best_ridge, best_score = None, -1e9
        for ridge in F_RIDGE_GRID:
            scores = []
            for hold in range(len(blocks[p])):
                u_tr = np.concatenate([b[1] for j, b in enumerate(blocks[p]) if j != hold])
                phi_tr = np.concatenate([b[2] for j, b in enumerate(blocks[p]) if j != hold])
                f = fit_anchor(u_tr, phi_tr, ridge)
                u_te, phi_te = blocks[p][hold][1], blocks[p][hold][2]
                pred = phi_te @ f.T
                ss_res = float(((u_te - pred) ** 2).sum())
                ss_tot = float(((u_te - u_te.mean(0)) ** 2).sum())
                scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
            score = float(np.mean(scores))
            print(f"[anchor {args.stratum}] p={p} ridge={ridge} loo_u_r2={score:.4f}", flush=True)
            if score > best_score:
                best_ridge, best_score = ridge, score
        u_all = np.concatenate([b[1] for b in blocks[p]])
        phi_all = np.concatenate([b[2] for b in blocks[p]])
        f = fit_anchor(u_all, phi_all, best_ridge)
        within = []
        for _name, u_s, phi_s in blocks[p]:
            f_s = fit_anchor(u_s, phi_s, best_ridge)
            ss_res = float(((u_s - phi_s @ f_s.T) ** 2).sum())
            ss_tot = float(((u_s - u_s.mean(0)) ** 2).sum())
            within.append(1.0 - ss_res / max(ss_tot, 1e-12))
        chosen[p] = {"ridge": best_ridge, "loo_u_r2": best_score,
                     "within_u_r2": float(np.mean(within)), "n_bins": int(len(u_all))}
        arrays[f"F{p}"] = f
        arrays[f"ridge{p}"] = np.array(best_ridge)
    path = anchor_path(args.stratum)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, sessions=np.array(world.orig_ids))
    (path.with_suffix(".json")).write_text(json.dumps({"stratum": args.stratum, "per_p": {str(k): v for k, v in chosen.items()}},
                                                      indent=1))
    print(json.dumps(chosen), flush=True)


def refdata_for(world) -> RefData:
    cached = getattr(world, "_p0_ref", None)
    if cached is not None:
        return cached
    name = world.orig_ids[-1]
    if name not in world.sessions:
        from ssm_decode.dandi688_data import load_dandi688_recording
        world.add_session(name, load_dandi688_recording(name))
    world._p0_ref = RefData(world, world.sessions[name])
    return world._p0_ref


def cca_in_w0_coordinates(world, sd, idx, mean_p, std_p, stratum: str):
    """Corrected ridge-CCA map expressed in the B-trial normalization of ``W0``.

    Returns ``(W, b, info)`` or ``None`` when no reach-direction overlaps the reference day.
    """
    res = run_method("cca", CCA_CFG[stratum], refdata_for(world), sd, np.asarray(idx), use_all=False)
    if res is None:
        return None
    t_cca, c_cca, mean_c, std_c, info = res
    if info.get("n_common_conditions", 0) < 1:
        return None
    w, b = map_cca_into_w0_coordinates(t_cca, c_cca, mean_c, std_c, mean_p, std_p)
    # Algebraic identity on the calibration bins (float64), every episode.
    x = neural_np(sd)
    a0, b0 = sd.trials[int(idx[0])]
    sl = slice(a0, min(a0 + 64, b0))
    got = apply_readin(x[sl], w, mean_p, std_p, b)
    ref = ((x[sl] - mean_c) / std_c) @ np.asarray(t_cca, dtype=np.float64).T + np.asarray(c_cca, dtype=np.float64)
    err = float(np.max(np.abs(got - ref)))
    if err > 1e-6:
        raise AssertionError(f"CCA/W0 coordinate error {err} on {sd.name}")
    info = dict(info)
    info["coordinate_max_abs"] = err
    info["transform"] = "raw"
    info["family"] = "cca/units/raw/ridge"
    return w, b, info


def query_mask(sd, idx, max_bins: int | None):
    q = sd.query_mask(idx).clone()
    if max_bins is not None:
        q[int(max_bins):] = False
    return q


def _batch_matches_single(world, xz, tables) -> bool:
    """One-time check that a batched trunk forward matches per-table forwards."""
    if _BATCH_OK["checked"]:
        return _BATCH_OK["ok"]
    base = world.adapter.model.base
    t = min(2048, xz.shape[0])
    sl = xz[:t]
    ww = tables[:2]
    with torch.inference_mode():
        single = torch.stack([base((sl @ ww[i].T).unsqueeze(0)).squeeze(0) for i in range(2)])
        batched = base(torch.einsum("tn,mkn->mtk", sl, ww))
    err = float((single - batched).abs().max())
    ok = err < 1e-4
    _BATCH_OK["checked"] = True
    _BATCH_OK["ok"] = ok
    _BATCH_OK["err"] = err
    print(f"[batch-check] max_abs={err:.3e} ok={ok}", flush=True)
    return ok


def score_tables(world, sd, mean, std, tables: list[np.ndarray], biases: list[np.ndarray],
                 qmask, max_bins: int | None) -> list[float]:
    """Contiguous causal decode of several read-in maps. One normalization for all of them."""
    dev = world.device
    nb = sd.n_bins if max_bins is None else min(sd.n_bins, int(max_bins))
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    xz = (sd.x32[:nb] - mean_t) / std_t
    w = torch.as_tensor(np.stack(tables), dtype=torch.float32, device=dev)
    b = torch.as_tensor(np.stack(biases), dtype=torch.float32, device=dev)
    base = world.adapter.model.base
    use_batch = w.shape[0] > 1 and _batch_matches_single(world, xz, w)
    chunk = 8 if nb <= pc.SIM_MAX_BINS else 4
    preds = []
    with torch.inference_mode():
        if not use_batch:
            for i in range(w.shape[0]):
                u = xz @ w[i].T + b[i]
                preds.append(base(u.unsqueeze(0)).squeeze(0))
        else:
            for i in range(0, w.shape[0], chunk):
                ww, bb = w[i:i + chunk], b[i:i + chunk]
                u = torch.einsum("tn,mkn->mtk", xz, ww) + bb[:, None, :]
                preds.append(base(u))
            # preds is a list of [m_i, T, D]; flatten below
    if use_batch:
        y = torch.cat(preds, 0)
    else:
        y = torch.stack(preds, 0)
    q_np = qmask.detach().cpu().numpy().astype(bool) if torch.is_tensor(qmask) else np.asarray(qmask, dtype=bool)
    idx = np.flatnonzero(q_np)
    idx = idx[idx < y.shape[1]]
    if len(idx) == 0:
        return [-1.0] * int(y.shape[0])
    ym, ys = world.label_stats
    truth = sd.record.behavior[idx].astype(np.float64)
    ss_tot = float(((truth - truth.mean(0, keepdims=True)) ** 2).sum())
    pred = y[:, torch.as_tensor(idx, device=y.device)].double().cpu().numpy()
    pred = pred * ys.reshape(1, 1, -1) + ym.reshape(1, 1, -1)
    finite = np.isfinite(pred).all(axis=(1, 2))
    ss_res = ((truth.reshape(1, len(idx), -1) - pred) ** 2).sum(axis=(1, 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    r2 = np.where(finite, r2, -1.0)
    return [float(v) for v in r2]


def p0_table(calib, stats, prior):
    res = calib.calibrate(stats, prior)
    if bool(res.grad_enabled_inside):
        raise AssertionError("P0 calibrate ran with grad enabled")
    return res.table.detach().cpu().numpy().astype(np.float64)


def build_candidates(w0, prior, anchor, moments, mean, std, cca, *, grid: bool):
    """Return (method, cfg, W, bias) candidates. ``grid=True`` is the source sweep."""
    cands = []
    zeros = np.zeros(w0.shape[0], dtype=np.float64)
    cands.append(("p0", {}, w0, zeros))
    cands.append(("p_prior", {}, prior, zeros))
    # cfg is only the hyperparameter. A failed CCA map still occupies alpha>0 with a null
    # table so selection scores it as -1 instead of dropping the hard episodes.
    if cca is None:
        cands.append(("w_alpha", {"alpha": 0.0}, w0, zeros))
        if grid:
            for alpha in ALPHAS:
                if alpha == 0.0:
                    continue
                cands.append(("w_alpha", {"alpha": float(alpha)}, None, None))
        cands.append(("ridge_cca", {"alpha": 1.0}, None, None))
    else:
        w_cca, b_cca, _info = cca
        if grid:
            for alpha in ALPHAS:
                w, b = shrink_tables(w0, w_cca, b_cca, alpha)
                cands.append(("w_alpha", {"alpha": float(alpha)}, w, b))
        cands.append(("ridge_cca", {"alpha": 1.0}, w_cca, b_cca))
    lam_eps = [(lam, eps) for lam in LAM_GRID for eps in EPS_GRID] if grid else None
    for p, raw in moments.items():
        h, g = moments_to_normalized(raw, mean, std)
        f = anchor[p]
        specs = lam_eps if lam_eps is not None else None
        if specs is None:
            continue
        for lam, eps in specs:
            for kind, base_w in (("p0solve", w0), ("prior_solve", prior)):
                table = target_side_solve(base_w, f, raw, mean, std, lam, eps)
                cands.append((f"{kind}_p{p}", {"lam": lam, "eps": eps, "p": p}, table, zeros))
        # stash H so a non-grid caller can reuse it; grid path doesn't need the attribute
        raw._H, raw._G = h, g          # noqa: SLF001 - local cache, not persisted
    return cands


def solve_one(base_w, anchor_f, raw, mean, std, lam, eps):
    probe = []
    table = target_side_solve(base_w, anchor_f, raw, mean, std, lam, eps, probe=probe)
    if probe != [False]:
        raise AssertionError(f"target-side solve grad probe {probe}")
    return table


def cmd_select(args):
    device = require_gpu()
    sim_set = SIM_SET[args.stratum]
    world, names = pc.sim_world(args.stratum, sim_set)
    anchor = load_anchor(args.stratum)
    p0 = pc.load_p0(world, args.stratum)
    first = min(t[-8:] for t in D1_POOLS[args.stratum]["targets"])
    banned = {t[-8:] for st in D1_POOLS.values() for t in st["targets"]} | set(FINAL_DATES) | {"20140307"}
    # 20140307 is a legal SOURCE for month/year. It is banned only as a target. Drop it from this check.
    banned.discard("20140307")
    for name in names:
        date = name[-8:]
        if date in banned or date >= first or date in FINAL_DATES:
            raise AssertionError(f"source episode session {date} is not a legal causal source for {args.stratum}")
    print(f"[select {args.stratum}/{sim_set}] sessions={[n[-8:] for n in names]}", flush=True)
    path = OUT / "select" / f"{args.stratum}_{sim_set}.jsonl"
    seen = seen_keys(path, ("session", "seed", "B", "draw"))
    t0 = time.perf_counter()
    for name in names:
        sd, prior_t = world.sessions[name], world.priors[name]
        prior = prior_t.detach().cpu().numpy().astype(np.float64)
        for seed in SEEDS:
            for B in B_LIST:
                for draw in range(pc.SEL_DRAWS):
                    if (name, seed, B, draw) in seen:
                        continue
                    idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                    mean, std, stats = norm_np(sd, idx)
                    moments = {}
                    for p in P_GRID:
                        raw = raw_moments(sd, idx, p)
                        check_norm_matches_raw(raw, mean, std, f"{name} B={B}")
                        moments[p] = raw
                    w0 = p0_table(p0[seed], stats, prior_t)
                    cca = cca_in_w0_coordinates(world, sd, idx, mean, std, args.stratum)
                    cands = build_candidates(w0, prior, anchor, moments, mean, std, cca, grid=True)
                    usable = [(m, c, w, b) for m, c, w, b in cands if w is not None]
                    dead = [(m, c) for m, c, w, b in cands if w is None]
                    q = query_mask(sd, idx, pc.SIM_MAX_BINS)
                    scores = score_tables(world, sd, mean, std,
                                          [c[2] for c in usable], [c[3] for c in usable], q, pc.SIM_MAX_BINS)
                    # alpha=0 must reproduce p0 under the same normalization
                    by_method = {}
                    rows = []
                    for (method, cfg, _w, _b), r2 in zip(usable, scores):
                        rows.append({"stratum": args.stratum, "set": sim_set, "session": name,
                                     "seed": seed, "B": int(B), "draw": int(draw), "method": method,
                                     "cfg": cfg, "r2": r2, "n_query_bins": int(q.sum())})
                        by_method.setdefault(method, []).append((cfg, r2))
                    for method, cfg in dead:
                        rows.append({"stratum": args.stratum, "set": sim_set, "session": name,
                                     "seed": seed, "B": int(B), "draw": int(draw), "method": method,
                                     "cfg": cfg, "r2": -1.0, "n_query_bins": int(q.sum())})
                    p0_r2 = next(r for c, r in by_method["p0"] if c == {})
                    a0 = next(r for c, r in by_method["w_alpha"] if c.get("alpha") == 0.0)
                    if abs(p0_r2 - a0) > 1e-4:
                        raise AssertionError(f"alpha=0 ({a0}) != p0 ({p0_r2}) on {name} B={B}")
                    append_jsonl(path, rows)
                    print(f"[select {args.stratum}] {name[-8:]} s{seed} B={B} d={draw} "
                          f"p0={p0_r2:.3f} t={time.perf_counter() - t0:.0f}s", flush=True)
    write_selection(args.stratum, sim_set)


def write_selection(stratum: str, sim_set: str):
    import pandas as pd
    path = OUT / "select" / f"{stratum}_{sim_set}.jsonl"
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    df["cfgk"] = df.cfg.map(cfg_dumps)
    df["r2"] = df.r2.fillna(-1.0)
    keep = [m for m in df.method.unique() if m not in ("p0", "p_prior")]
    sel = {"stratum": stratum, "set": sim_set, "sessions": sorted(set(df.session.str[-8:])),
           "n_rows": int(len(df))}
    grid_rows = []
    for method in keep:
        sub = df[df.method == method]
        sel[method] = {}
        for B, block in sub.groupby("B"):
            g = block.groupby("cfgk").r2.agg(["mean", "count"]).reset_index()
            g["cfg"] = g.cfgk.map(json.loads)
            g["alpha"] = g.cfg.map(lambda c: c.get("alpha", np.nan))
            g["lam"] = g.cfg.map(lambda c: c.get("lam", np.nan))
            g = g.sort_values(["mean", "alpha", "lam"], ascending=[False, True, False], na_position="last")
            best = g.iloc[0]
            sel[method][str(int(B))] = {"cfg": best.cfg, "source_r2": float(best["mean"]),
                                        "n_episodes": int(best["count"])}
            for _, row in g.iterrows():
                grid_rows.append({"method": method, "B": int(B), "cfg": row.cfgk,
                                  "mean": float(row["mean"]), "count": int(row["count"])})
    # p0 source score, per B, so the README can show the selected margin
    sel["p0"] = {}
    for B, block in df[df.method == "p0"].groupby("B"):
        sel["p0"][str(int(B))] = {"source_r2": float(block.r2.mean()), "n_episodes": int(len(block))}
    selection_path(stratum).parent.mkdir(parents=True, exist_ok=True)
    selection_path(stratum).write_text(json.dumps(sel, indent=1))
    pd.DataFrame(grid_rows).to_csv(OUT / "select" / f"grid_{stratum}_{sim_set}.csv", index=False)
    brief = {m: {b: sel[m][b]["cfg"] for b in sel[m]} for m in ("w_alpha", "p0solve_p8", "p0solve_p16") if m in sel}
    print(json.dumps(brief, indent=1), flush=True)


def load_selection(stratum: str) -> dict:
    """Drift reuses the month rules (one frozen trunk, no per-target re-selection)."""
    src = "month" if stratum == "drift" else stratum
    return json.loads(selection_path(src).read_text())


def backbone_stratum(stratum: str) -> str:
    return {"days": "days", "month": "month", "year": "year", "drift": "month"}[stratum]


def targets_for(stratum: str, explicit: str | None) -> list[str]:
    if stratum == "drift":
        dates = list(TARGET_DATES) if not explicit else explicit.split(",")
        src = [s[-8:] for s in D1_POOLS["month"]["sources"].split(",")]
        dates = validate_targets(dates, src)
        return [PREFIX + d for d in dates]
    if explicit:
        return [PREFIX + d if not d.startswith("sub-") else d for d in explicit.split(",")]
    return list(D1_POOLS[stratum]["targets"])


def h4_record(sd, idx, qmask) -> dict:
    used = np.zeros(sd.n_bins, dtype=bool)
    for i in np.asarray(idx):
        a, b = sd.trials[int(i)]
        used[a:b] = True
    q = qmask.detach().cpu().numpy() if torch.is_tensor(qmask) else np.asarray(qmask)
    return {"h4_budget": int(len(idx)), "h4_query_disjoint": bool(not (used & q).any()),
            "n_query_bins": int(np.sum(q)), "backprop": False}


def episode_maps(world, sd, prior_t, calib, idx, anchor, sel, B, stratum_cca: str):
    """Tables for one (target, B, draw, seed). CCA and the prior are seed-free but idx is not."""
    mean, std, stats = norm_np(sd, idx)
    moments = {}
    solved_h = {}
    for p in P_GRID:
        raw = raw_moments(sd, idx, p)
        check_norm_matches_raw(raw, mean, std, sd.name)
        moments[p] = raw
        solved_h[p] = moments_to_normalized(raw, mean, std)
    w0 = p0_table(calib, stats, prior_t) if calib is not None else None
    prior = prior_t.detach().cpu().numpy().astype(np.float64)
    cca = cca_in_w0_coordinates(world, sd, idx, mean, std, stratum_cca)
    zeros = np.zeros(K, dtype=np.float64)
    out = {"mean": mean, "std": std, "prior": prior, "H": solved_h, "moments": moments}
    if w0 is not None:
        out["p0"] = (w0, zeros, {})
        alpha = float(sel["w_alpha"][str(B)]["cfg"]["alpha"])
        # alpha>0 with no CCA map is unavailable. Do not silently report P0 as W_alpha.
        if cca is None and alpha > 0.0:
            out["w_alpha"] = None
        elif cca is None:
            out["w_alpha"] = (w0, zeros, {"alpha": 0.0, "cca": "unavailable"})
        else:
            w, bias = shrink_tables(w0, cca[0], cca[1], alpha)
            out["w_alpha"] = (w, bias, {"alpha": alpha})
        for p in P_GRID:
            cfg = sel[f"p0solve_p{p}"][str(B)]["cfg"]
            table = solve_one(w0, anchor[p], moments[p], mean, std, cfg["lam"], cfg["eps"])
            out[f"p0solve_p{p}"] = (table, zeros, cfg)
    cfg_names = {f"prior_solve_p{p}": p for p in P_GRID}
    for method, p in cfg_names.items():
        cfg = sel[method][str(B)]["cfg"]
        table = solve_one(prior, anchor[p], moments[p], mean, std, cfg["lam"], cfg["eps"])
        out[method] = (table, zeros, cfg)
    out["p_prior"] = (prior, zeros, {})
    if cca is None:
        out["ridge_cca"] = None
    else:
        out["ridge_cca"] = (cca[0], cca[1], {"alpha": 1.0, "n_common_conditions": cca[2].get("n_common_conditions"),
                                            "d_eff": cca[2].get("d_eff"), "family": "cca/units/raw/ridge"})
    return out


def cmd_eval(args):
    device = require_gpu()
    stratum = args.stratum
    back = backbone_stratum(stratum)
    sel = load_selection(stratum)
    if stratum == "drift" and sel.get("sessions") != ["20140307"]:
        raise AssertionError(f"drift must reuse the month oosc rules, got sessions {sel.get('sessions')}")
    anchor = load_anchor(back)
    world = rpc.World(back, "O", device, load_sessions=False)
    p0 = pc.load_p0(world, back)
    hash0 = world.adapter._hash()
    targets = targets_for(stratum, args.targets)
    for t in targets:
        if t[-8:] in FINAL_DATES or t[-8:] == "20140307":
            raise AssertionError(f"refusing target {t}")
    path = OUT / "eval" / f"rows_{stratum}.jsonl"
    seen = seen_keys(path, ("method", "seed", "target", "B", "draw"))
    checked_decode = False
    t0 = time.perf_counter()
    for target in targets:
        if not torch.cuda.is_available():
            raise SystemExit("GPU disappeared; stopping")
        sd, prior_t = world.target(target)
        print(f"[eval {stratum}] {target[-8:]} N={sd.n_units} trials={len(sd.trials)}", flush=True)
        for B in B_LIST:
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                q = query_mask(sd, idx, None)
                base_h4 = h4_record(sd, idx, q)
                # seed-free maps are rebuilt per seed only because idx is shared across seeds
                # (draw_trials does not depend on seed). Compute them once.
                shared = None
                for seed in SEEDS:
                    need = [m for m in EVAL_METHODS
                            if (m, seed, target, B, draw) not in seen and not (m in SEED_FREE and seed != 0)]
                    if not need:
                        continue
                    maps = episode_maps(world, sd, prior_t, p0[seed], idx, anchor, sel, B, back)
                    if shared is None:
                        shared = maps
                    rows_spec = []
                    for method in need:
                        spec = maps[method]
                        if spec is None:
                            rows_spec.append((method, None, None, {"cca": "unavailable"}))
                        else:
                            rows_spec.append((method, spec[0], spec[1], spec[2]))
                    live = [(m, w, b, c) for m, w, b, c in rows_spec if w is not None]
                    if live:
                        scores = score_tables(world, sd, maps["mean"], maps["std"],
                                              [t[1] for t in live], [t[2] for t in live], q, None)
                    else:
                        scores = []
                    scored = {m: r for (m, *_r), r in zip(live, scores)}
                    if not checked_decode and "ridge_cca" in scored and maps["ridge_cca"] is not None:
                        # Trunk-level check: mapped table vs the original CCA affine decode.
                        w, b, _c = maps["ridge_cca"]
                        # Recover native CCA by running it again (cached inside RefData per config, not per idx).
                        native = run_method("cca", CCA_CFG[back], refdata_for(world), sd, idx)
                        pred_native = decode_affine(world, sd, native[0], native[1], native[2], native[3], "raw", False)
                        pred_mapped = score_one_pred(world, sd, maps["mean"], maps["std"], w, b)
                        err = float((pred_native - pred_mapped).abs().max())
                        print(f"[coord-decode] max_abs_pred={err:.3e}", flush=True)
                        if err > 1e-3:
                            raise AssertionError(f"mapped CCA decode disagrees with native CCA by {err}")
                        checked_decode = True
                    rows = []
                    for method, _w, _b, cfg in rows_spec:
                        rows.append({"stratum": stratum, "backbone": back, "target": target,
                                     "date": target[-8:], "B": int(B), "draw": int(draw), "seed": int(seed),
                                     "method": method, "r2": scored.get(method, None), "cfg": cfg,
                                     "diagnostic_only": False, "n_units": int(sd.n_units),
                                     "selection": "month" if stratum == "drift" else stratum,
                                     **base_h4})
                    append_jsonl(path, rows)
                    show = " ".join(f"{m}={scored[m]:.3f}" for m in ("p0", "w_alpha", "p0solve_p8", "p0solve_p16", "ridge_cca")
                                    if m in scored and scored[m] is not None)
                    print(f"[eval {stratum}] {target[-8:]} B={B} d={draw} s{seed} {show} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
        del sd, prior_t
        torch.cuda.empty_cache()
    if world.adapter._hash() != hash0:
        raise AssertionError("trunk weights changed during eval")


def score_one_pred(world, sd, mean, std, w, b):
    dev = world.device
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    xz = (sd.x32 - mean_t) / std_t
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    bt = torch.as_tensor(b, dtype=torch.float32, device=dev)
    with torch.inference_mode():
        return world.adapter.model.base((xz @ wt.T + bt).unsqueeze(0)).squeeze(0)


def load_drift_ui() -> dict:
    """Stored drift p0_ui_reg / p0 rows keyed by (target, B, draw, seed, method)."""
    out = {}
    if not DRIFT_ROWS.exists():
        return out
    for f in DRIFT_ROWS.glob("rows_*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["method"] in ("p0_ui_reg", "p0", "r3", "p_prior"):
                out[(r["target"], int(r["B"]), int(r["draw"]), int(r["seed"]), r["method"])] = float(r["r2"])
    return out


def cmd_diag(args):
    """Projection diagnostic. ui_reg is recomputed so the table, not only its R^2, is available."""
    from scripts.run_pc3_uiref import run_traj
    device = require_gpu()
    stratum = args.stratum
    back = backbone_stratum(stratum)
    rule = UI_RULE["month" if stratum == "drift" else stratum]
    ui_sel = json.loads(Path(rule["file"]).read_text())
    world = rpc.World(back, "O", device, load_sessions=False)
    p0 = pc.load_p0(world, back)
    targets = targets_for(stratum, args.targets)
    stored = load_drift_ui() if stratum in ("drift", "month") else {}
    path = OUT / "diag" / f"rows_{stratum}.jsonl"
    seen = seen_keys(path, ("seed", "target", "B", "draw", "p"))
    t0 = time.perf_counter()
    for target in targets:
        if not torch.cuda.is_available():
            raise SystemExit("GPU disappeared; stopping")
        sd, prior_t = world.target(target)
        print(f"[diag {stratum}] {target[-8:]} like_for_like={rule['like_for_like']}", flush=True)
        for B in B_LIST:
            u = ui_sel[str(B)]
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                for seed in SEEDS:
                    if all((seed, target, B, draw, p) in seen for p in P_GRID):
                        continue
                    ep = pc.p0_episode(world, sd, prior_t, p0[seed], idx)
                    tab = run_traj(world, sd, ep, u["lam"], u["lr"], (u["steps"],))[u["steps"]]
                    w_ui = np.asarray(tab, dtype=np.float64)
                    w0 = ep.table0.detach().cpu().numpy().astype(np.float64)
                    delta = w_ui - w0
                    mean = ep.mean.detach().cpu().numpy().astype(np.float64)
                    std = ep.std.detach().cpu().numpy().astype(np.float64)
                    # Score p0 and ui with the same decoder as the diagnostic table.
                    zeros = np.zeros(K, dtype=np.float64)
                    q = ep.q
                    r_p0, r_ui = score_tables(world, sd, mean, std, [w0, w_ui], [zeros, zeros], q, None)
                    stored_ui = stored.get((target, int(B), int(draw), int(seed), "p0_ui_reg"))
                    if stored_ui is not None and abs(stored_ui - r_ui) > 1e-3:
                        print(f"[diag warn] ui r2 {r_ui:.4f} vs stored {stored_ui:.4f} "
                              f"{target[-8:]} B={B} d={draw} s{seed}", flush=True)
                    rows = []
                    for p in P_GRID:
                        if (seed, target, B, draw, p) in seen:
                            continue
                        raw = raw_moments(sd, idx, p)
                        h, _g = moments_to_normalized(raw, mean, std)
                        proj = project_update(delta, h, None)
                        w_diag = w0 + proj
                        r_diag = score_tables(world, sd, mean, std, [w_diag], [zeros], q, None)[0]
                        fro = float(np.linalg.norm(proj) / max(np.linalg.norm(delta), 1e-12))
                        rows.append({"stratum": stratum, "backbone": back, "target": target,
                                     "date": target[-8:], "B": int(B), "draw": int(draw), "seed": int(seed),
                                     "p": int(p), "method": "w_diag", "r2": r_diag, "r2_p0": r_p0,
                                     "r2_ui": r_ui, "r2_ui_stored": stored_ui,
                                     "gain_ui": r_ui - r_p0, "gain_diag": r_diag - r_p0,
                                     "frobenius_retained": fro, "diagnostic_only": True,
                                     "like_for_like": bool(rule["like_for_like"]),
                                     "cfg": {"lam": u["lam"], "lr": u["lr"], "steps": u["steps"], "p": int(p)},
                                     "note": rule["note"], "backprop": True, "n_units": int(sd.n_units)})
                    append_jsonl(path, rows)
                    print(f"[diag {stratum}] {target[-8:]} B={B} d={draw} s{seed} "
                          f"ui-p0={r_ui - r_p0:+.3f} t={time.perf_counter() - t0:.0f}s", flush=True)
        del sd, prior_t
        torch.cuda.empty_cache()


def cmd_cost(_args):
    rows = []
    for n in (26, 256):
        for p in P_GRID:
            rows.append(cost_model(n, K, p))
    path = OUT / "cost_p0solve.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    for r in rows:
        m = r["macs_per_bin"]
        c = r["commit"]
        print(f"N={r['N']} p={r['p']} accum={r['accumulator_bytes_fp32']}B "
              f"double={r['accumulator_bytes_fp32_double_buffer']}B "
              f"MAC/bin={m['total_full_G']} (H {m['sum_rphi']}, G {m['sum_phiphi_full']}, r2 {m['sum_r2']}) "
              f"commit_MAC={c['total_macs_excluding_solve_flops']} solve_flops={c['solve_flops']} "
              f"matrix={r['solve_matrix']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit-anchor")
    a.add_argument("--stratum", required=True, choices=["days", "month", "year"])
    s = sub.add_parser("select")
    s.add_argument("--stratum", required=True, choices=["days", "month", "year"])
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, choices=["days", "month", "year", "drift"])
    e.add_argument("--targets", default=None)
    e.add_argument("--draws", type=int, default=rpc.N_DRAWS)
    d = sub.add_parser("diag")
    d.add_argument("--stratum", required=True, choices=["days", "month", "year", "drift"])
    d.add_argument("--targets", default=None)
    d.add_argument("--draws", type=int, default=rpc.N_DRAWS)
    sub.add_parser("cost")
    w = sub.add_parser("write-selection")
    w.add_argument("--stratum", required=True, choices=["days", "month", "year"])
    args = p.parse_args()
    if args.cmd == "fit-anchor":
        cmd_fit_anchor(args)
    elif args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "write-selection":
        write_selection(args.stratum, SIM_SET[args.stratum])
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "diag":
        cmd_diag(args)
    elif args.cmd == "cost":
        cmd_cost(args)
    else:
        raise SystemExit(args.cmd)


if __name__ == "__main__":
    main()
