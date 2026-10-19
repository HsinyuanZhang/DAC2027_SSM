#!/usr/bin/env python
"""Follow-up selection for P0-Solve (2026-10-19 review).

  lam    wider lambda grid and eps = eps_rel * trace(G) / p, on the ORIGINAL source episodes
  gap    alpha(B, elapsed-days bin), reliability logistic, and their product, on source episodes
         dated before the stratum's earliest target (trunk sessions + causal OOS)
  eval   target rows for the adaptive alphas and, when the solve config changed, the re-selected solve

Nothing here reads a target query. Drift reuses the month gap table (no re-selection).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_p0solve as R                                          # noqa: E402
from scripts import run_pc3_refine as pc                                     # noqa: E402
from scripts import run_population_calib as rpc                              # noqa: E402
from scripts.run_labeled_calibration import D1_POOLS                         # noqa: E402
from ssm_decode.p0solve import (ALPHAS, EPS_REL_GRID, LAM_GRID,                # noqa: E402
                                combine_gap_reliability, fit_combined_logistic,
                                fit_monotone_logistic, gap_bin_name, logistic_alpha,
                                nearest_grid_alpha, resolve_gap_bin, shrink_tables)

# Lambdas the first grid already scored at negligible absolute eps. New corners plus 1e3.
NEW_LAMS = (1e3, 1e5, 1e6)
# Relative eps large enough that epsilon is not lost under trace(G).
REL_EPS = EPS_REL_GRID


def ymd(text: str) -> dt.date:
    s = str(text)[-8:]
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def elapsed_days(sim_date: str, ref_date: str) -> int:
    """Signed days from the latest trunk session to this session."""
    return (ymd(sim_date) - ymd(ref_date)).days


def ref_date(stratum: str) -> str:
    """Latest trunk-source date. Drift uses the month trunk."""
    back = "month" if stratum == "drift" else stratum
    return max(s[-8:] for s in D1_POOLS[back]["sources"].split(","))


def original_sim_names(world, stratum: str) -> list[str]:
    return list(pc.sim_sessions(world, stratum, R.SIM_SET[stratum]))


def gap_sim_names(world, stratum: str) -> list[str]:
    """Trunk sessions and causal OOS strictly before the earliest target. Not the reference day itself
    when it would be a zero-gap self-alignment (it is before the target, but CCA to itself is not a drift)."""
    first = min(t[-8:] for t in D1_POOLS[stratum]["targets"])
    ref = ref_date(stratum)
    names = []
    for name in list(world.orig_ids) + original_sim_names(world, stratum):
        if name[-8:] >= first or name[-8:] in R.FINAL_DATES:
            continue
        if name[-8:] == ref:
            continue
        if name not in names:
            names.append(name)
    for name in names:
        if name[-8:] in R.FINAL_DATES or name[-8:] == "20140307" and stratum == "days" and False:
            pass
        if name[-8:] in {t[-8:] for st in D1_POOLS.values() for t in st["targets"]} and name[-8:] >= first:
            raise AssertionError(name)
    return names


def load_alpha_scores(stratum: str) -> dict:
    """Stored W_alpha R^2 from the first source sweep, keyed by (session, seed, B, draw, alpha)."""
    path = R.OUT / "select" / f"{stratum}_{R.SIM_SET[stratum]}.jsonl"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method") != "w_alpha":
            continue
        alpha = r["cfg"].get("alpha")
        if alpha is None:
            continue
        out[(r["session"], int(r["seed"]), int(r["B"]), int(r["draw"]), float(alpha))] = float(r["r2"])
    return out


def cca_reliability(cca) -> float:
    """Top-3 mean canonical correlation from the B calibration trials. 0 if CCA failed."""
    if cca is None:
        return 0.0
    info = cca[2]
    if "top3_canon_corr" in info and np.isfinite(info["top3_canon_corr"]):
        return float(info["top3_canon_corr"])
    return float(info.get("mean_canon_corr") or 0.0)


def cmd_select(args):
    """One pass: extended (lambda, eps_rel) on the original episodes, and alpha curves for the gap set."""
    device = R.require_gpu()
    stratum = args.stratum
    world = rpc.World(stratum, "O", device, load_sessions=True)
    anchor = R.load_anchor(stratum)
    p0 = pc.load_p0(world, stratum)
    orig = original_sim_names(world, stratum)
    gap_names = gap_sim_names(world, stratum)
    # sim_sessions may have added OOS recordings already. orig_ids are loaded.
    ref = ref_date(stratum)
    print(f"[adapt {stratum}] ref={ref} orig={[n[-8:] for n in orig]} gap={[n[-8:] for n in gap_names]}",
          flush=True)
    banned_targets = {t[-8:] for st in D1_POOLS.values() for t in st["targets"]}
    first = min(t[-8:] for t in D1_POOLS[stratum]["targets"])
    for name in gap_names:
        if name[-8:] in R.FINAL_DATES or name[-8:] in banned_targets and name[-8:] >= first:
            raise AssertionError(f"gap session {name} is not a legal source for {stratum}")
    stored_alpha = load_alpha_scores(stratum)
    lam_path = R.OUT / "select" / f"{stratum}_{R.SIM_SET[stratum]}_ext.jsonl"
    gap_path = R.OUT / "select" / f"{stratum}_gap.jsonl"
    lam_seen = R.seen_keys(lam_path, ("session", "seed", "B", "draw"))
    gap_seen = R.seen_keys(gap_path, ("session", "seed", "B", "draw"))
    t0 = time.perf_counter()
    for name in gap_names:
        sd, prior_t = world.sessions[name], world.priors[name]
        prior = prior_t.detach().cpu().numpy().astype(np.float64)
        elapsed = elapsed_days(name[-8:], ref)
        for seed in R.SEEDS:
            for B in R.B_LIST:
                for draw in range(pc.SEL_DRAWS):
                    need_lam = name in orig and (name, seed, B, draw) not in lam_seen
                    need_gap = (name, seed, B, draw) not in gap_seen
                    if not need_lam and not need_gap:
                        continue
                    idx = pc.draw_sim(name, f"p3sel-s{seed}", B, draw, sd)
                    mean, std, stats = R.norm_np(sd, idx)
                    moments = {p: R.raw_moments(sd, idx, p) for p in R.P_GRID}
                    for p, raw in moments.items():
                        R.check_norm_matches_raw(raw, mean, std, name)
                    w0 = R.p0_table(p0[seed], stats, prior_t)
                    cca = R.cca_in_w0_coordinates(world, sd, idx, mean, std, stratum)
                    q = R.query_mask(sd, idx, pc.SIM_MAX_BINS)
                    zeros = np.zeros(R.K, dtype=np.float64)
                    tables, tags = [], []
                    if need_gap:
                        have = [stored_alpha.get((name, seed, int(B), int(draw), float(a))) for a in ALPHAS]
                        if cca is None:
                            # alpha 0 is P0; anything above is unavailable
                            pass
                        elif all(v is not None for v in have):
                            pass
                        else:
                            for alpha in ALPHAS:
                                w, b = shrink_tables(w0, cca[0], cca[1], alpha)
                                tables.append(w)
                                tags.append(("alpha", float(alpha), b))
                    if need_lam:
                        for p, raw in moments.items():
                            _h, g = R.moments_to_normalized(raw, mean, std)
                            for lam in LAM_GRID:
                                # Skip (lam, tiny eps) already represented by the absolute-eps sweep,
                                # except the new lambdas, which were never scored.
                                rels = REL_EPS if lam in NEW_LAMS else tuple(e for e in REL_EPS if e >= 1e-2)
                                for eps_rel in rels:
                                    eps = R.eps_of_cfg({"eps_rel": eps_rel}, g)
                                    for kind, base_w in (("p0solve", w0), ("prior_solve", prior)):
                                        table = R.solve_one(base_w, anchor[p], raw, mean, std, lam, eps)
                                        tables.append(table)
                                        tags.append(("lam", kind, p, lam, eps_rel, zeros))
                    biases = []
                    clean_tables = []
                    for tag, tab in zip(tags, tables):
                        if tag[0] == "alpha":
                            clean_tables.append(tab)
                            biases.append(tag[2])
                        else:
                            clean_tables.append(tab)
                            biases.append(tag[5])
                    scores = []
                    if clean_tables:
                        scores = R.score_tables(world, sd, mean, std, clean_tables, biases, q, pc.SIM_MAX_BINS)
                    cursor = 0
                    if need_gap:
                        r2_by = {}
                        have = {float(a): stored_alpha.get((name, seed, int(B), int(draw), float(a))) for a in ALPHAS}
                        if cca is None:
                            r2_by = {0.0: None}
                            # still score p0 so the bin is not empty of a number
                            p0_only = R.score_tables(world, sd, mean, std, [w0], [zeros], q, pc.SIM_MAX_BINS)[0]
                            r2_by = {float(a): (p0_only if a == 0.0 else -1.0) for a in ALPHAS}
                        elif all(v is not None for v in have.values()):
                            r2_by = {a: float(v) for a, v in have.items()}
                        else:
                            for alpha in ALPHAS:
                                r2_by[float(alpha)] = float(scores[cursor])
                                cursor += 1
                        R.append_jsonl(gap_path, [{
                            "stratum": stratum, "session": name, "seed": int(seed), "B": int(B),
                            "draw": int(draw), "elapsed": int(elapsed), "bin": gap_bin_name(elapsed),
                            "ref": ref, "c": cca_reliability(cca), "r2_by_alpha": r2_by,
                            "n_query_bins": int(q.sum()),
                        }])
                    if need_lam:
                        rows = []
                        for tag in tags:
                            if tag[0] != "lam":
                                continue
                            _kind_tag, kind, p, lam, eps_rel, _z = tag
                            rows.append({
                                "stratum": stratum, "set": R.SIM_SET[stratum], "session": name,
                                "seed": int(seed), "B": int(B), "draw": int(draw),
                                "method": f"{kind}_p{p}",
                                "cfg": {"lam": lam, "eps_rel": eps_rel, "p": p},
                                "r2": float(scores[cursor]), "n_query_bins": int(q.sum()),
                            })
                            cursor += 1
                        R.append_jsonl(lam_path, rows)
                    print(f"[adapt {stratum}] {name[-8:]} s{seed} B={B} d={draw} "
                          f"bin={gap_bin_name(elapsed)} el={elapsed} t={time.perf_counter() - t0:.0f}s",
                          flush=True)
    if lam_path.exists():
        merge_lam_selection(stratum)
    write_gap_rules(stratum)


def merge_lam_selection(stratum: str):
    """Re-pick lambda/eps from the original absolute-eps rows plus the relative-eps extension."""
    import pandas as pd
    paths = [R.OUT / "select" / f"{stratum}_{R.SIM_SET[stratum]}.jsonl",
             R.OUT / "select" / f"{stratum}_{R.SIM_SET[stratum]}_ext.jsonl"]
    rows = []
    for path in paths:
        if path.exists():
            rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df.method.isin([f"{k}_p{p}" for k in ("p0solve", "prior_solve") for p in (8, 16)])]
    df["cfgk"] = df.cfg.map(R.cfg_dumps)
    df["r2"] = df.r2.fillna(-1.0)
    sel = json.loads(R.selection_path(stratum).read_text())
    changed = {}
    for method, sub in df.groupby("method"):
        sel.setdefault(method, {})
        for B, block in sub.groupby("B"):
            g = block.groupby("cfgk").r2.agg(["mean", "count"]).reset_index()
            g["cfg"] = g.cfgk.map(json.loads)
            g["lam"] = g.cfg.map(lambda c: float(c.get("lam", np.nan)))
            g = g.sort_values(["mean", "lam"], ascending=[False, False])
            best = g.iloc[0]
            old = sel[method].get(str(int(B)), {}).get("cfg")
            new = best.cfg
            sel[method][str(int(B))] = {"cfg": new, "source_r2": float(best["mean"]),
                                        "n_episodes": int(best["count"]),
                                        "previous_cfg": old}
            changed.setdefault(method, {})[str(int(B))] = {"old": old, "new": new,
                                                          "source_r2": float(best["mean"]),
                                                          "n_episodes": int(best["count"])}
    sel["lam_resel"] = True
    sel["n_source_episodes_note"] = (
        "month oosc is 1 session x 2 seeds x 2 draws = 4 episodes per B; "
        "the extended grid is scored on those same episodes and pooled with the first sweep."
    )
    R.selection_path(stratum).write_text(json.dumps(sel, indent=1))
    (R.OUT / "select" / f"{stratum}_lam_changed.json").write_text(json.dumps(changed, indent=1))
    print(json.dumps(changed, indent=1)[:2000], flush=True)


def write_gap_rules(stratum: str):
    path = R.OUT / "select" / f"{stratum}_gap.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["r2_by_alpha"] = {float(k): float(v) for k, v in r["r2_by_alpha"].items()}
    # Per (B, natural bin) alpha, then fallbacks for empty bins.
    by = {}
    for r in rows:
        by.setdefault((int(r["B"]), r["bin"]), []).append(r)
    schedule = {}
    logs = []
    for B in R.B_LIST:
        populated = {b for (bb, b) in by if bb == B}
        schedule[str(B)] = {}
        for want in ("le14", "mid", "gt180"):
            used, fell = resolve_gap_bin(want, populated)
            block = by.get((B, used), [])
            means = {}
            for alpha in ALPHAS:
                vals = [ep["r2_by_alpha"][float(alpha)] for ep in block]
                means[float(alpha)] = float(np.mean(vals)) if vals else -1.0
            best = sorted(means, key=lambda a: (means[a], -a), reverse=True)[0]
            schedule[str(B)][want] = {
                "alpha": best, "source_r2": means[best], "n_episodes": len(block),
                "bin_used": used, "fallback_from": fell,
                "populated_bins": sorted(populated),
            }
            if fell:
                logs.append({"B": B, "requested": fell, "used": used, "n": len(block)})
    # Attach the bin alpha each episode would have been assigned (after fallback) for the combined fit.
    enriched = []
    for r in rows:
        spec = schedule[str(int(r["B"]))][r["bin"]]
        enriched.append({**r, "alpha_bin": float(spec["alpha"])})
    logistic = fit_monotone_logistic(enriched)
    combined = fit_combined_logistic(enriched)
    # Constant-alpha baseline on the same episodes, so the logistic has to earn its keep on source.
    const = {}
    for alpha in ALPHAS:
        const[float(alpha)] = float(np.mean([r["r2_by_alpha"][float(alpha)] for r in enriched]))
    best_const = max(const, key=const.get)
    out = {
        "stratum": stratum, "ref": ref_date(stratum),
        "n_episodes": len(rows),
        "sessions": sorted({r["session"][-8:] for r in rows}),
        "schedule": schedule,
        "fallbacks": logs,
        "logistic": logistic,
        "combined": combined,
        "constant_alpha": {"alpha": best_const, "source_r2": const[best_const], "grid": const},
    }
    dest = R.OUT / "select" / f"gap_rules_{stratum}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({"schedule": {b: {k: v["alpha"] for k, v in s.items()} for b, s in schedule.items()},
                      "logistic": logistic, "combined": combined,
                      "constant": best_const, "fallbacks": logs}, indent=1), flush=True)


def load_gap_rules(stratum: str) -> dict:
    src = "month" if stratum == "drift" else stratum
    return json.loads((R.OUT / "select" / f"gap_rules_{src}.json").read_text())


def alpha_for_target(rules: dict, B: int, elapsed: int, c: float) -> dict:
    natural = gap_bin_name(elapsed)
    spec = rules["schedule"][str(B)][natural]
    alpha_bin = float(spec["alpha"])
    logi = rules["logistic"]
    both = rules["combined"]
    return {
        "elapsed": int(elapsed),
        "bin": natural,
        "bin_used": spec["bin_used"],
        "fallback_from": spec["fallback_from"],
        "c": float(c),
        "w_alpha_gap": alpha_bin,
        "w_alpha_rel": float(logistic_alpha(c, logi["intercept"], logi["slope"])),
        "w_alpha_both": float(combine_gap_reliability(alpha_bin, c, both["intercept"], both["slope"])),
    }


def cmd_eval(args):
    device = R.require_gpu()
    stratum = args.stratum
    back = R.backbone_stratum(stratum)
    rules = load_gap_rules(stratum)
    sel = R.load_selection(stratum)
    anchor = R.load_anchor(back)
    world = rpc.World(back, "O", device, load_sessions=False)
    p0 = pc.load_p0(world, back)
    targets = R.targets_for(stratum, args.targets)
    ref = ref_date(stratum)
    # Which solve methods actually changed config.
    changed = set()
    lam_stratum = "month" if stratum == "drift" else stratum
    ch_path = R.OUT / "select" / f"{lam_stratum}_lam_changed.json"
    # Drift uses the month solve rules, so a month re-selection is re-scored here too.
    if ch_path.exists():
        changed_map = json.loads(ch_path.read_text())
        for method, per_b in changed_map.items():
            for B, info in per_b.items():
                if info.get("old") != info.get("new"):
                    changed.add((method, int(B)))
    path = R.OUT / "eval" / f"rows_{stratum}_adapt.jsonl"
    seen = R.seen_keys(path, ("method", "seed", "target", "B", "draw"))
    methods_alpha = ("w_alpha_gap", "w_alpha_rel", "w_alpha_both")
    t0 = time.perf_counter()
    for target in targets:
        if not torch.cuda.is_available():
            raise SystemExit("GPU disappeared; stopping")
        if target[-8:] in R.FINAL_DATES or target[-8:] == "20140307":
            raise AssertionError(target)
        sd, prior_t = world.target(target)
        elapsed = elapsed_days(target[-8:], ref)
        print(f"[adapt-eval {stratum}] {target[-8:]} elapsed={elapsed} bin={gap_bin_name(elapsed)}", flush=True)
        for B in R.B_LIST:
            for draw in range(args.draws):
                idx = rpc.draw_trials(target, B, draw, sd.bank.n_trials)
                q = R.query_mask(sd, idx, None)
                h4 = R.h4_record(sd, idx, q)
                for seed in R.SEEDS:
                    need_alpha = [m for m in methods_alpha if (m, seed, target, B, draw) not in seen]
                    need_solve = [m for (m, b) in changed if b == int(B) and (m, seed, target, B, draw) not in seen
                                  and not (m.startswith("prior_solve") and seed != 0)]
                    if not need_alpha and not need_solve:
                        continue
                    maps = R.episode_maps(world, sd, prior_t, p0[seed], idx, anchor, sel, B, back)
                    cca = R.cca_in_w0_coordinates(world, sd, idx, maps["mean"], maps["std"], back)
                    choice = alpha_for_target(rules, int(B), elapsed, cca_reliability(cca))
                    w0 = maps["p0"][0]
                    zeros = np.zeros(R.K, dtype=np.float64)
                    specs = []
                    if cca is None:
                        for m in need_alpha:
                            alpha = float(choice[m])
                            if alpha > 1e-8:
                                specs.append((m, None, None, {**choice, "alpha": alpha, "cca": "unavailable"}))
                            else:
                                specs.append((m, w0, zeros, {**choice, "alpha": 0.0, "cca": "unavailable"}))
                    else:
                        for m in need_alpha:
                            alpha = float(np.clip(choice[m], 0.0, 1.0))
                            w, b = shrink_tables(w0, cca[0], cca[1], alpha)
                            specs.append((m, w, b, {**choice, "alpha": alpha,
                                                    "nearest_grid": nearest_grid_alpha(alpha)}))
                    for m in need_solve:
                        spec = maps[m]
                        specs.append((m, spec[0], spec[1], spec[2]))
                    live = [s for s in specs if s[1] is not None]
                    scores = R.score_tables(world, sd, maps["mean"], maps["std"],
                                            [s[1] for s in live], [s[2] for s in live], q, None) if live else []
                    scored = {s[0]: v for s, v in zip(live, scores)}
                    rows = []
                    for method, _w, _b, cfg in specs:
                        rows.append({"stratum": stratum, "backbone": back, "target": target,
                                     "date": target[-8:], "B": int(B), "draw": int(draw), "seed": int(seed),
                                     "method": method, "r2": scored.get(method), "cfg": cfg,
                                     "diagnostic_only": False, "n_units": int(sd.n_units),
                                     "selection": "month" if stratum == "drift" else stratum,
                                     **h4})
                    R.append_jsonl(path, rows)
                    show = " ".join(f"{m}={scored[m]:.3f}" for m in methods_alpha if m in scored)
                    print(f"[adapt-eval {stratum}] {target[-8:]} B={B} d={draw} s{seed} {show} "
                          f"t={time.perf_counter() - t0:.0f}s", flush=True)
        del sd, prior_t
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--stratum", required=True, choices=["days", "month", "year"])
    e = sub.add_parser("eval")
    e.add_argument("--stratum", required=True, choices=["days", "month", "year", "drift"])
    e.add_argument("--targets", default=None)
    e.add_argument("--draws", type=int, default=rpc.N_DRAWS)
    g = sub.add_parser("write-gap")
    g.add_argument("--stratum", required=True, choices=["days", "month", "year"])
    args = p.parse_args()
    if args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "write-gap":
        write_gap_rules(args.stratum)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
