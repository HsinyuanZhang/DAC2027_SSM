"""Selection pool for expansion round 2 (frozen causal-first-B methods).

The frozen score functions stay in ``causal_protocol`` / ``run_causal_main``. This module only decides which
source episodes may vote for a target, and how a finished ``--out`` tree is packed for the Colab Drive sync.

Protocol (a), expanding-window: an episode votes iff its session and its trunk's last training day are both
strictly before the target. Protocol (b), strict: also drop any reported target session and any trunk that
was trained on a reported target. Tags ``month`` and ``year`` are the same trunk (identical weights); a
session that appears under both is kept once, under ``month``.

The pool is the legacy causal-main episodes plus every OOS held-out ``pool[-1]`` of an expansion backbone
that passes the same test. ``sm_0307`` is in-sample and never contributes. 2016 sessions and final sessions
never contribute. 20140307 is not added as a selection session by this round (it is only ``sm_0307``'s target).
"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path

from ssm_decode import expansion_protocol as ep
from ssm_decode.causal_protocol import TRUNK_LAST, TRUNK_TRAIN, alpha_episode_allowed

# Frozen reported targets of the causal-main table (dates). Round 2 adds every primary expansion target.
# Kept in sync with scripts/run_causal_main.py (_DRIFT, _SUBC); a test checks the equality.
CAUSAL_MAIN_REPORTED = frozenset({
    "20140626", "20140627", "20140929", "20141203", "20150511", "20150512",
    "20150610", "20150611", "20150612", "20150615", "20150616", "20150617",
    "20150629", "20150701", "20150706", "20150709", "20150713", "20150716",
    "20151103", "20151104", "20151106", "20151109", "20151110", "20151112",
})
# year trunk weights are the month trunk. Count the session once.
CANONICAL_TRUNK = {"year": "month"}
LEGACY_TAGS = ("days", "month", "year", "sc_0629", "sc_0708")
FAMILIES = ("alpha", "s1", "combo", "cca_s1")
# What causal-eval writes. Primary gate is AND; ``*_cons`` is the >=2-session gate.
EVAL_METHODS = (
    "p_prior", "p0", "ridge_cca",
    "w_alpha_a", "w_alpha_b",
    "s1_causal", "s1_causal_cons", "s1_causal_b", "s1_causal_b_cons",
    "combo_a", "combo_b", "combo_a_cons", "combo_b_cons",
    "cca_s1", "cca_s1_cons", "cca_s1_b", "cca_s1_b_cons",
    "r2", "r3",
)
SEED_FREE = frozenset({"p_prior", "ridge_cca"})
RULE_VERSION = "expansion_causal_v1"


def canonical_trunk(tag: str) -> str:
    return CANONICAL_TRUNK.get(str(tag), str(tag))


def reported_target_dates(backbones=None) -> set[str]:
    """Sessions that protocol (b) treats as already reported. 2016 stays out (tier off)."""
    backbones = ep.select_tier(ep.build_backbones(), tier="primary") if backbones is None else backbones
    dates = {ep.date_of(t.session) for bb in backbones for t in bb.targets}
    dates |= set(CAUSAL_MAIN_REPORTED)
    return {d for d in dates if not str(d).startswith("2016")}


def _backbone_by_bid() -> dict:
    return {bb.bid: bb for bb in ep.build_backbones()}


def expansion_trunk(tag: str) -> dict | None:
    """Trunk record for an expansion backbone id, or None if it is not an OOS selection contributor."""
    bb = _backbone_by_bid().get(str(tag))
    if bb is None or not bb.oos_selection:
        return None
    last = ep.date_of(max(bb.trunk_sources, key=ep.date_of))
    return {"last": last, "train": {ep.date_of(s) for s in bb.trunk_sources},
            "heldout": ep.date_of(bb.val_session), "subject": bb.subject,
            "canonical": bb.bid}


def session_banned(session: str) -> bool:
    """Final sessions, 2016 (tier off), and a bare date that matches one of those."""
    text = str(session)
    date = text[-8:]
    if date.startswith("2016") or "CO-2016" in text:
        return True
    if text in ep.ALL_FORBIDDEN:
        return True
    forbidden_dates = {ep.date_of(s) for s in ep.ALL_FORBIDDEN}
    return date in forbidden_dates and (text == date or text.endswith(date))


def episode_allowed(row: dict, target_date: str, variant: str, reported: set) -> bool:
    """One source episode, one target, one protocol. Frozen tags use ``alpha_episode_allowed`` as-is."""
    if variant not in ("expanding", "strict"):
        raise ValueError(variant)
    if session_banned(row.get("session", "")):
        return False
    tag = str(row["tag"])
    target_date = str(target_date)
    date = str(row["session"])[-8:]
    # Expansion backbones contribute only their held-out day, even when the tag is also in the
    # frozen trunk table (sc_0629, sc_0708). days/month/year stay on the frozen rule alone.
    meta = expansion_trunk(tag)
    if meta is not None and date != meta["heldout"]:
        return False
    if tag in TRUNK_LAST:
        return bool(alpha_episode_allowed(row, target_date, str(row.get("subject", "")), variant, reported))
    if meta is None:
        return False                                  # IS trunk (sm_0307) or an unknown tag
    if date >= target_date or meta["last"] >= target_date:
        return False
    if variant == "strict":
        if date in reported:
            return False
        if set(meta["train"]) & set(reported):
            return False
    return True


def _prefer(new: dict, old: dict) -> bool:
    """Keep the canonical tag when month and year (same trunk) both scored the session."""
    new_exact = str(new["tag"]) == canonical_trunk(new["tag"])
    old_exact = str(old["tag"]) == canonical_trunk(old["tag"])
    if new_exact != old_exact:
        return new_exact
    return False


def dedupe_same_trunk(episodes: list[dict]) -> list[dict]:
    """Drop same-trunk copies. Key is (canonical trunk, session date, seed, B)."""
    best: dict = {}
    order: list = []
    for ep_row in episodes:
        key = (canonical_trunk(ep_row["tag"]), str(ep_row["session"])[-8:],
               int(ep_row["seed"]), int(ep_row["B"]))
        prev = best.get(key)
        if prev is None:
            best[key] = ep_row
            order.append(key)
        elif _prefer(ep_row, prev):
            best[key] = ep_row
    return [best[k] for k in order]


def selection_pool(episodes: list[dict], target_date: str, variant: str, reported: set, *,
                   B: int | None = None) -> list[dict]:
    """Episodes that may vote. ``B`` limits the grid; None keeps every budget."""
    kept = [r for r in episodes if episode_allowed(r, target_date, variant, reported)
            and (B is None or int(r["B"]) == int(B))]
    return dedupe_same_trunk(kept)


def composition(episodes: list[dict]) -> dict:
    """Per-target selection log. ``n_sessions`` is distinct calendar days, after same-trunk dedupe."""
    pairs = sorted({(canonical_trunk(r["tag"]), str(r["session"])[-8:]) for r in episodes})
    sessions = sorted({p[1] for p in pairs})
    return {"n_sessions": len(sessions), "n_trunk_sessions": len(pairs),
            "sessions": sessions, "trunks": sorted({p[0] for p in pairs}),
            "pairs": [f"{t}:{s}" for t, s in pairs],
            "n_episodes": len(episodes),
            "tags": sorted({str(r["tag"]) for r in episodes}),
            "seeds": sorted({int(r["seed"]) for r in episodes})}


def eligible_contributors(target_date: str, backbones=None) -> list[str]:
    """OOS backbones whose held-out day and trunk are both strictly before ``target_date``."""
    backbones = ep.build_backbones() if backbones is None else backbones
    target_date = str(target_date)
    out = []
    for bb in backbones:
        meta = expansion_trunk(bb.bid)
        if meta is None:
            continue
        if meta["heldout"] < target_date and meta["last"] < target_date:
            out.append(bb.bid)
    return out


def contributor_ready(bid: str, alpha_episodes: list[dict]) -> bool:
    meta = expansion_trunk(bid)
    if meta is None:
        return True
    return any(str(r.get("tag")) == bid and str(r["session"]).endswith(meta["heldout"])
               for r in alpha_episodes)


def missing_contributors(target_date: str, alpha_episodes: list[dict], backbones=None) -> list[str]:
    return [bid for bid in eligible_contributors(target_date, backbones)
            if not contributor_ready(bid, alpha_episodes)]


def select_alpha_from_allowed(episodes: list[dict], trunk_gap_days: int, B: int) -> dict:
    """α from episodes that have already passed the protocol filter (no second filter).

    Binning matches ``causal_protocol.select_alpha``: the target's own gap bin, empty bins fall back,
    exact ties prefer the smaller alpha.
    """
    from ssm_decode.p0solve import ALPHAS, gap_bin_name, resolve_gap_bin
    import numpy as np
    want = gap_bin_name(int(trunk_gap_days))
    bins: dict = {}
    for r in episodes:
        if int(r["B"]) != int(B) or r.get("gap_days") is None:
            continue
        bins.setdefault(gap_bin_name(int(r["gap_days"])), []).append(r)
    comp = composition(episodes)
    if not bins:
        return {"alpha": None, "bin": want, "bin_used": None, "n_episodes": 0,
                "sessions": comp["sessions"], "tags": comp["tags"], "fallback_from": want,
                "n_sessions": comp["n_sessions"]}
    used, fell = resolve_gap_bin(want, set(bins))
    group = bins[used]
    means = {}
    for alpha in ALPHAS:
        vals = []
        for ep_row in group:
            hit = next((a["r2"] for a in ep_row["alphas"] if float(a["alpha"]) == float(alpha)), None)
            if hit is not None and float(hit) > -0.5:
                vals.append(float(hit))
        means[float(alpha)] = float(np.mean(vals)) if vals else -1.0
    winner = sorted(means, key=lambda a: (means[a], -a), reverse=True)[0]
    return {"alpha": float(winner), "source_r2": means[winner], "bin": want, "bin_used": used,
            "fallback_from": fell, "n_episodes": len(group),
            "sessions": sorted({str(r["session"])[-8:] for r in group}),
            "tags": sorted({str(r["tag"]) for r in group}),
            "n_sessions": len({str(r["session"])[-8:] for r in group})}


def gate_choices(episodes: list[dict], eta_r2: float, base_key: str) -> dict:
    """AND pick (primary) and the conservative >=2-session pick (falls back when the AND step is not)."""
    from ssm_decode.causal_protocol import margin_pick, sessions_with_positive_gain
    if not episodes:
        return {"and": None, "conservative": None, "session_gate": None}
    picked = margin_pick(episodes, "rebase", float(eta_r2), mode="and", base_key=base_key)
    gate = sessions_with_positive_gain(episodes, "rebase", float(picked["eta"]), float(picked["rho"]),
                                       base_key=base_key)
    if picked.get("fallback_base") or gate["deploy"]:
        conservative = dict(picked)
        conservative["conservative_rejected"] = False
    else:
        conservative = dict(picked)
        conservative.update({"eta": 0.0, "rho": 0.0, "fallback_base": True,
                             "conservative_rejected": True,
                             "rejected_eta": picked["eta"], "rejected_rho": picked["rho"],
                             "source_gain": 0.0})
    return {"and": picked, "conservative": conservative, "session_gate": gate}


def combo_at_alpha(episodes: list[dict], alpha: float | None) -> list[dict]:
    """COMBO rebase rows scored at this α. A missing α drops the row (it cannot vote for another α)."""
    if alpha is None:
        return []
    return [r for r in episodes if abs(float(r.get("alpha", -999.0)) - float(alpha)) < 1e-9]


def target_rules(bundles: dict, target_date: str, trunk_gap_days: int, B: int, reported: set,
                 eta_r2: float) -> dict:
    """Per-target α (a and b) and (η, ρ) for S1 / COMBO / CCA+S1, plus the pool composition."""
    out = {"B": int(B), "eta_r2": float(eta_r2)}
    for variant, key in (("expanding", "a"), ("strict", "b")):
        alpha_pool = selection_pool(bundles.get("alpha", []), target_date, variant, reported, B=B)
        alpha = select_alpha_from_allowed(alpha_pool, trunk_gap_days, B)
        s1_pool = selection_pool(bundles.get("s1", []), target_date, variant, reported, B=B)
        cca_pool = selection_pool(bundles.get("cca_s1", []), target_date, variant, reported, B=B)
        combo_pool_all = selection_pool(bundles.get("combo", []), target_date, variant, reported, B=B)
        combo_pool = combo_at_alpha(combo_pool_all, alpha["alpha"])
        out[key] = {
            "alpha": alpha,
            "composition": composition(alpha_pool),
            "s1": gate_choices(s1_pool, eta_r2, "p0"),
            "s1_composition": composition(s1_pool),
            "combo": gate_choices(combo_pool, eta_r2, "base"),
            "combo_composition": composition(combo_pool),
            "combo_dropped_alpha": len(combo_pool_all) - len(combo_pool),
            "cca_s1": gate_choices(cca_pool, eta_r2, "base"),
            "cca_composition": composition(cca_pool),
        }
    return out


def source_unit_key(row: dict, what: str):
    alpha = None if what != "combo" else round(float(row["alpha"]), 6)
    return (str(row["session"]), int(row["seed"]), int(row["B"]), alpha)


def pending_units(rows: list[dict], session: str, seeds, Bs, alphas, what: str) -> list[tuple]:
    """Units still to score. A row already on disk (including a CCA-missing skip) is done."""
    done = {source_unit_key(r, what) for r in rows}
    pending = []
    for seed in seeds:
        for B in Bs:
            if what == "combo":
                for alpha in alphas:
                    key = (session, int(seed), int(B), round(float(alpha), 6))
                    if key not in done:
                        pending.append(key)
            else:
                key = (session, int(seed), int(B), None)
                if key not in done:
                    pending.append(key)
    return pending


def eval_row_key(row: dict) -> tuple:
    return (str(row["method"]), int(row["seed"]), int(row["B"]), int(row["latency"]))


def pending_eval(rows: list[dict], methods, seeds, Bs, latencies) -> list[tuple]:
    done = {eval_row_key(r) for r in rows}
    pending = []
    for B in Bs:
        for seed in seeds:
            for method in methods:
                for lat in latencies:
                    key = (str(method), int(seed), int(B), int(lat))
                    if key not in done:
                        pending.append(key)
    return pending


def legacy_family_path(legacy_dir: Path, family: str, tag: str) -> Path:
    return Path(legacy_dir) / f"{family}_{tag}_episodes.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    from ssm_decode.atomic_jsonl import read_jsonl
    path = Path(path)
    if not path.exists():
        return []
    return read_jsonl(path)


def load_episodes(out_dir: Path, legacy_dir: Path | None) -> dict[str, list[dict]]:
    """Local causal-source rows override a legacy row with the same (family, tag, session, seed, B, alpha)."""
    out_dir = Path(out_dir)
    bundles = {fam: [] for fam in FAMILIES}
    seen: dict[str, set] = {fam: set() for fam in FAMILIES}
    select_root = out_dir / "causal_select"
    if select_root.is_dir():
        for bid_dir in sorted(p for p in select_root.iterdir() if p.is_dir()):
            for fam in FAMILIES:
                for row in load_jsonl(bid_dir / f"{fam}.jsonl"):
                    if row.get("skipped"):
                        continue
                    key = (str(row.get("tag")), str(row.get("session")), int(row["seed"]), int(row["B"]),
                           None if fam != "combo" else round(float(row.get("alpha", 0.0)), 6))
                    if key in seen[fam]:
                        continue
                    seen[fam].add(key)
                    bundles[fam].append(row)
    if legacy_dir is not None and Path(legacy_dir).is_dir():
        for fam in FAMILIES:
            for tag in LEGACY_TAGS:
                for row in load_jsonl(legacy_family_path(legacy_dir, fam, tag)):
                    if row.get("skipped"):
                        continue
                    key = (str(row.get("tag", tag)), str(row.get("session")), int(row["seed"]), int(row["B"]),
                           None if fam != "combo" else round(float(row.get("alpha", 0.0)), 6))
                    if key in seen[fam]:
                        continue
                    seen[fam].add(key)
                    if "tag" not in row:
                        row = dict(row)
                        row["tag"] = tag
                    bundles[fam].append(row)
    return bundles


def sync_tree_to_tar(src, dest) -> Path:
    """Pack ``src`` to ``dest`` atomically (write ``dest.partial``, then ``os.replace``).

    Round 1 synced a local VM ``--out`` to Drive this way: the reader never sees a half-written tar.
    ``.tgz`` / ``.tar.gz`` are gzip; anything else is an uncompressed tar.
    """
    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    name = dest.name
    mode = "w:gz" if name.endswith(".tgz") or name.endswith(".tar.gz") else "w"
    with tarfile.open(tmp, mode) as tar:
        tar.add(src, arcname=src.name)
    os.replace(tmp, dest)
    return dest


def round2_cost_plan() -> dict:
    """Wall-clock plan for one L4 (1.54 CU/h), jobs=2. CU is session wall time, not the sum of GPU jobs.

    Scaled from round 1 (3 backbones, 1 h 51 min wall, 3.1 CU) and from that round's select stage
    (~15 min per backbone for 40 source episodes). Upper bounds assume the AP budget-extension fires
    and each causal target takes ~4× the round-1 eval. Each session stays under 12 h; the sum stays
    under 40 CU. If the first causal target exceeds 15 min, split 2b as listed.
    """
    cu = 1.54
    sessions = [
        {"id": "2a", "expected_h": 4.5, "upper_h": 8.5,
         "bids": ["sc_0309", "sm_0307", "sm_0627", "sm_0512", "sm_0611",
                  "sc_0708", "sc_0713", "sc_0629"],
         "what": "existing stages for the five new backbones; causal-source for every OOS backbone "
                 "(round-1 trunks reused from expansion_out_round1)"},
        {"id": "2b", "expected_h": 2.5, "upper_h": 8.0,
         "bids": ["all primary targets, including the round-1 rescore"],
         "what": "causal-eval, latency 0 and 1"},
    ]
    fallback = [
        {"id": "2b-subc", "expected_h": 2.0, "upper_h": 6.5,
         "what": "causal-eval sub-C only, if the first target exceeds 15 min"},
        {"id": "2b-subm", "expected_h": 0.8, "upper_h": 2.5,
         "what": "causal-eval sub-M only"},
    ]
    for row in sessions + fallback:
        row["expected_cu"] = round(row["expected_h"] * cu, 2)
        row["upper_cu"] = round(row["upper_h"] * cu, 2)
    return {"cu_per_hour": cu, "cu_cap": 40.0, "session_limit_h": 12.0,
            "sessions": sessions, "fallback_if_slow": fallback,
            "expected_cu": round(sum(r["expected_h"] for r in sessions) * cu, 2),
            "upper_cu": round(sum(r["upper_h"] for r in sessions) * cu, 2)}
