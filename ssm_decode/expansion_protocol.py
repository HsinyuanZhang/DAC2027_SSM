"""Time-forward expansion protocol: sub-C (second subject) and sub-M days targets (2026-10-16).

Pure Python on top of the small session inventory
(``results/analysis/expansion_20261016/session_inventory.json``); no NWB access, no torch.  The protocol is a list
of *backbones*.  A backbone = a pool of 6 sessions (the AP trunk + P0 calibrator "pool O" analog) that
serves several time-forward targets of one or more strata.

Rules enforced by :func:`validate_backbone` (all unit-tested):

* a pool has exactly 6 distinct sessions of one subject, sorted by date;
* every pool session is strictly earlier than every target of the same backbone (no target and no
  post-target session in a pool, targets are never pool members);
* no session is forbidden: sub-M final (20150623/25/26) and the six APST held-out sub-C sessions;
* every pool session has at least ``MIN_POOL_UNITS`` M1 units so K=10 stays inside the Ledermann bound;
* every target has at least ``MIN_TARGET_TRIALS`` usable trials (B=33 calibration + 20 query trials);
* the stratum label of a target agrees with its gap (days) from the latest pool session:
  days 1-7, month 90-130, year >= 365.

Selection protocol (changed 2026-10-16 after the P3 review: in-sample selection collapses towards "no
change"): for every NEW backbone the AP trunk is trained on the first 5 of the 6 pool sessions
(``trunk_sources``) and the 6th, latest session (``heldout`` = ``pool[-1]`` = ``val_session``) is held out of
trunk training.  It is the P0 early-stopping session and the source-simulation session on which the
R1-stream / R2 (F7-J) / R3 rules are selected.  It is trunk-unseen (out-of-sample, OOS) and causal (earlier than
every target).  The reused sm_0307 backbone keeps its 6-session trunk, so its ``val_session`` is
in-sample (IS) and its rows must never be merged with the OOS rows or with the P3 / drift-curve R1-R3 rows
(selected on 20140307, which is the sm_0307 target).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ssm_decode.association import ledermann_limit
from ssm_decode.dandi688_data import FINAL_SESSIONS, FORBIDDEN_SESSIONS

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "results/analysis/expansion_20261016/session_inventory.json"
PROTOCOL_PATH = ROOT / "results/analysis/expansion_20261016/protocol.json"

POOL_SIZE = 6
K_TARGET = 10                      # parity with the sub-M P1 protocol (AP K=10)
MIN_POOL_UNITS = 15                # ledermann_limit(15) == 10: smallest N that supports K=10
MIN_TARGET_TRIALS = 33 + 20        # B=33 calibration trials plus >= 20 query trials
B_LIST = (1, 4, 8, 16, 33)
N_DRAWS = 8
SEEDS = (0, 1)
STRATUM_GAP_DAYS = {"days": (1, 7), "month": (90, 130), "year": (365, 10_000)}
PREFIX = {"sub-M": "sub-M_ses-CO-", "sub-C": "sub-C_ses-CO-"}
ALL_FORBIDDEN = frozenset(FINAL_SESSIONS) | FORBIDDEN_SESSIONS

# Disclosure only (not forbidden): sub-C sessions that other SPINT / btransform protocols use as exam faces
# (SPINT/btransform_unified_v2/dandi688_bench_v1/src/dandi688_bench_v1/plan.py: EXP1_NARROW_EXAM_SESSIONS at line
# ~682, EXP2016_EXAM_SESSIONS at line ~727; exp2_full / exp2015_full exam = the manifest val = APST dev sessions).
OTHER_PROTOCOL_EXAM_FACES = {
    **{f"sub-C_ses-CO-{d}": "exp1_narrow / exp1_poverty exam face (EXP1_NARROW_EXAM_SESSIONS 0713-0716)"
       for d in ("20150713", "20150714", "20150715", "20150716")},
    **{f"sub-C_ses-CO-{d}": "exp2_full / exp2015_full exam face (manifest val = APST development sessions)"
       for d in ("20151103", "20151104", "20151106", "20151109", "20151110", "20151112")},
}
# sub-M targets that were P1 design targets (days pool dev targets); their rows come from a different pool here.
P1_DESIGN_TARGETS = frozenset({"sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617", "sub-M_ses-CO-20140626",
                               "sub-M_ses-CO-20150511"})


@dataclass(frozen=True)
class Target:
    session: str
    stratum: str
    tier: str = "primary"          # primary | optional
    apst_dev: bool = False         # sub-C only: an APST development (not held-out) session
    gap_days: int = 0


@dataclass(frozen=True)
class Backbone:
    bid: str
    subject: str
    pool: tuple[str, ...]
    targets: tuple[Target, ...]
    k: int = K_TARGET
    reuse: dict = field(default_factory=dict)      # {} for a new backbone; else paths of existing artefacts
    note: str = ""

    @property
    def val_session(self) -> str:
        """P0 early-stopping and R1/R2/R3 selection session = last pool session (source only)."""
        return self.pool[-1]

    @property
    def oos_selection(self) -> bool:
        """True when ``val_session`` is held out of the trunk (new backbones); False for the reused sm_0307."""
        return not self.reuse

    @property
    def trunk_sources(self) -> tuple[str, ...]:
        """Sessions the AP trunk is trained on (and the Jacobian / PCA basis are computed on)."""
        return self.pool if self.reuse else self.pool[:-1]

    @property
    def heldout(self) -> tuple[str, ...]:
        return () if self.reuse else (self.pool[-1],)

    @property
    def selection_scope(self) -> str:
        return ("OOS: pool[-1] held out of trunk training (trunk-unseen, causal)" if self.oos_selection
                else "IS: pool[-1] seen by the (reused) trunk")

    def trunk_gap_days(self, session: str) -> int:
        """Days from the latest trunk-TRAINING session to ``session`` (>= the stratum gap, measured from pool[-1])."""
        return (to_date(date_of(session)) - to_date(date_of(max(self.trunk_sources, key=date_of)))).days


def to_date(date: str) -> dt.date:
    return dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))


def date_of(session: str) -> str:
    return session.rsplit("-", 1)[1]


def gap_days(pool: tuple[str, ...] | list[str], target: str) -> int:
    return (to_date(date_of(target)) - to_date(date_of(max(pool, key=date_of)))).days


def ledermann_k(n_units_min: int, k_max: int = K_TARGET) -> int:
    """Largest factor count <= k_max inside the Ledermann identifiability bound for N=n_units_min."""
    return max(1, min(k_max, ledermann_limit(int(n_units_min))))


def load_inventory(path: str | Path = INVENTORY_PATH) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    return {r["session"]: r for r in rows}


def forbidden_in(sessions) -> list[str]:
    return sorted(s for s in sessions if s in ALL_FORBIDDEN)


def rolling_pool(inventory: dict[str, dict], subject: str, first_target_date: str, *, n: int = POOL_SIZE,
                 min_units: int = MIN_POOL_UNITS) -> tuple[str, ...]:
    """The ``n`` latest eligible sessions strictly before ``first_target_date`` (time-forward pool).

    Eligible = same subject, not forbidden, at least ``min_units`` M1 units.  Used to define the sub-M days
    pools; the sub-C pools are hand-picked clusters (documented) and only validated.
    """
    rows = sorted((r for r in inventory.values() if r["subject"] == subject and r["session"] not in ALL_FORBIDDEN
                   and r["n_units"] >= min_units and r["date"] < first_target_date), key=lambda r: r["date"])
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} eligible {subject} sessions before {first_target_date}")
    return tuple(r["session"] for r in rows[-n:])


def validate_backbone(bb: Backbone, inventory: dict[str, dict], *, check_k: bool = True) -> dict:
    """Raise ValueError on any rule violation; return a small audit record."""
    pool, targets = list(bb.pool), [t.session for t in bb.targets]
    if len(pool) != POOL_SIZE or len(set(pool)) != POOL_SIZE:
        raise ValueError(f"{bb.bid}: pool must hold {POOL_SIZE} distinct sessions, got {pool}")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError(f"{bb.bid}: targets empty or duplicated: {targets}")
    prefix = PREFIX[bb.subject]
    for s in pool + targets:
        if not s.startswith(prefix):
            raise ValueError(f"{bb.bid}: {s} does not belong to {bb.subject}")
        if s not in inventory:
            raise ValueError(f"{bb.bid}: {s} is not in the session inventory (forbidden or missing)")
    bad = forbidden_in(pool + targets)
    if bad:
        raise ValueError(f"{bb.bid}: forbidden session(s) {bad}")
    if sorted(pool, key=date_of) != pool:
        raise ValueError(f"{bb.bid}: pool must be sorted by date")
    overlap = sorted(set(pool) & set(targets))
    if overlap:
        raise ValueError(f"{bb.bid}: target(s) inside the pool: {overlap}")
    latest, first_target = date_of(pool[-1]), min(date_of(t) for t in targets)
    if not latest < first_target:
        raise ValueError(f"{bb.bid}: pool session dated {latest} is not strictly before the earliest target {first_target}")
    late = [s for s in pool if date_of(s) >= first_target]
    if late:
        raise ValueError(f"{bb.bid}: post-target session(s) in pool: {late}")
    for s in pool:
        if inventory[s]["n_units"] < MIN_POOL_UNITS:
            raise ValueError(f"{bb.bid}: pool session {s} has only {inventory[s]['n_units']} units (< {MIN_POOL_UNITS})")
    for t in bb.targets:
        if inventory[t.session]["n_usable_trials"] < MIN_TARGET_TRIALS:
            raise ValueError(f"{bb.bid}: target {t.session} has {inventory[t.session]['n_usable_trials']} usable trials")
        lo, hi = STRATUM_GAP_DAYS[t.stratum]
        g = gap_days(pool, t.session)
        if not lo <= g <= hi:
            raise ValueError(f"{bb.bid}: target {t.session} gap {g} d outside {t.stratum} range {lo}-{hi}")
    min_n = min(inventory[s]["n_units"] for s in pool)
    if check_k and bb.k > ledermann_limit(min_n):
        raise ValueError(f"{bb.bid}: K={bb.k} exceeds the Ledermann limit {ledermann_limit(min_n)} for N={min_n}")
    if bb.val_session in targets:
        raise ValueError(f"{bb.bid}: selection session is a target")
    return {"bid": bb.bid, "min_pool_units": min_n, "ledermann_limit": ledermann_limit(min_n),
            "n_targets": len(targets)}


# ------------------------------------------------------------------------------------------ definitions

def _sm(d: str) -> str:
    return PREFIX["sub-M"] + d


def _sc(d: str) -> str:
    return PREFIX["sub-C"] + d


def _targets(dates, stratum, prefix_fn, pool, *, tier="primary", dev=()):
    return tuple(Target(prefix_fn(d), stratum, tier, prefix_fn(d) in dev,
                        gap_days(pool, prefix_fn(d))) for d in dates)


def build_backbones() -> list[Backbone]:
    from ssm_decode.dandi688_data import SUBC_APST_DEV_SESSIONS
    dev = set(SUBC_APST_DEV_SESSIONS)
    out: list[Backbone] = []

    # ---- sub-M days stratum with rolling backbones (task 2) ---------------------------------------
    pool = tuple(_sm(d) for d in ("20140203", "20140217", "20140218", "20140303", "20140304", "20140306"))
    out.append(Backbone(
        "sm_0307", "sub-M", pool, _targets(("20140307",), "days", _sm, pool),
        reuse={"ap": "results/stage2_intervals_20261006/month_pool_k10_x8/best.pt",
               "p0_root": "results/population_calib_20261015", "p0_stratum": "month"},
        note="existing month/year trunk + P1 P0 calibrators (pool O); 20140307 is the causal selection session of "
             "the old month rules, here it is a TARGET and is never used for selection (selection = pool val 20140306)"))
    pool = tuple(_sm(d) for d in ("20140218", "20140303", "20140304", "20140306", "20140307", "20140626"))
    out.append(Backbone("sm_0627", "sub-M", pool, _targets(("20140627",), "days", _sm, pool)))
    pool = tuple(_sm(d) for d in ("20140306", "20140307", "20140626", "20140627", "20140929", "20150511"))
    out.append(Backbone("sm_0512", "sub-M", pool, _targets(("20150512",), "days", _sm, pool),
                        note="20141203 (10 units) is skipped: below the Ledermann minimum for K=10"))
    pool = tuple(_sm(d) for d in ("20140626", "20140627", "20140929", "20150511", "20150512", "20150610"))
    out.append(Backbone("sm_0611", "sub-M", pool,
                        _targets(("20150611", "20150612", "20150615", "20150616", "20150617"), "days", _sm, pool),
                        note="20150616/17 are the old sub-M dev targets; here with a different (rolling) pool"))

    # ---- sub-C (task 1) -------------------------------------------------------------------------
    pool = tuple(_sc(d) for d in ("20150629", "20150630", "20150701", "20150703", "20150706", "20150707"))
    t = (_targets(("20150708", "20150709", "20150710", "20150713", "20150714"), "days", _sc, pool)
         + _targets(("20151103", "20151104", "20151106", "20151109", "20151110", "20151112"), "month", _sc, pool, dev=dev)
         + _targets(("20160909", "20160914", "20160919", "20160923", "20160929", "20161006"), "year", _sc, pool,
                    tier="optional"))
    out.append(Backbone("sc_0708", "sub-C", pool, t,
                        note="days targets 0708-0714; month targets are the APST dev sessions (not held out); "
                             "year targets (2016) are optional"))
    pool = tuple(_sc(d) for d in ("20150703", "20150706", "20150707", "20150708", "20150709", "20150710"))
    out.append(Backbone("sc_0713", "sub-C", pool,
                        _targets(("20150713", "20150714", "20150715", "20150716"), "days", _sc, pool)))
    pool = tuple(_sc(d) for d in ("20131220", "20150309", "20150311", "20150312", "20150313", "20150319"))
    out.append(Backbone("sc_0629", "sub-C", pool,
                        _targets(("20150629", "20150701", "20150706", "20150709", "20150713", "20150716"), "month", _sc, pool),
                        note="only 5 sessions exist in 2015-03; 20131220 completes the pool"))
    pool = tuple(_sc(d) for d in ("20131031", "20131101", "20131203", "20131204", "20131219", "20131220"))
    out.append(Backbone("sc_0309", "sub-C", pool,
                        _targets(("20150309", "20150312", "20150319"), "year", _sc, pool)
                        + _targets(("20150629", "20150706", "20150716"), "year", _sc, pool)))
    return out


def select_tier(backbones: list[Backbone], *, tier: str = "primary", exclude_apst_dev: bool = False) -> list[Backbone]:
    """Filter targets by tier ('primary' or 'all') and optionally drop the APST dev sessions."""
    out = []
    for bb in backbones:
        ts = tuple(t for t in bb.targets if (tier == "all" or t.tier == "primary")
                   and not (exclude_apst_dev and t.apst_dev))
        if ts:
            out.append(Backbone(bb.bid, bb.subject, bb.pool, ts, bb.k, bb.reuse, bb.note))
    return out


def target_clusters(backbones: list[Backbone]) -> dict[str, list[str]]:
    """target session -> backbone ids that evaluate it.  Rows of one session are NOT independent targets:
    the aggregation clusters by session (see ``scripts/analyze_expansion.py``)."""
    out: dict[str, list[str]] = {}
    for bb in backbones:
        for t in bb.targets:
            out.setdefault(t.session, []).append(bb.bid)
    return out


def duplicate_target_report(backbones: list[Backbone]) -> dict:
    cl = target_clusters(backbones)
    dup = {s: b for s, b in cl.items() if len(b) > 1}
    return {"n_pairs": sum(len(b) for b in cl.values()), "n_distinct_sessions": len(cl),
            "duplicated": dup, "p1_design_targets": sorted(set(cl) & P1_DESIGN_TARGETS),
            "other_protocol_exam_faces": {s: OTHER_PROTOCOL_EXAM_FACES[s] for s in sorted(cl)
                                          if s in OTHER_PROTOCOL_EXAM_FACES}}


def protocol_dict(inventory: dict[str, dict] | None = None) -> dict:
    inventory = inventory or load_inventory()
    bbs = build_backbones()
    audits = [validate_backbone(bb, inventory) for bb in bbs]
    sessions = sorted({s for bb in bbs for s in list(bb.pool) + [t.session for t in bb.targets]})
    return {
        "schema": "expansion_20261016_v2", "k": K_TARGET, "b_list": list(B_LIST), "n_draws": N_DRAWS,
        "seeds": list(SEEDS), "pool_size": POOL_SIZE, "min_pool_units": MIN_POOL_UNITS,
        "min_target_trials": MIN_TARGET_TRIALS, "stratum_gap_days": STRATUM_GAP_DAYS,
        "forbidden_sessions": sorted(ALL_FORBIDDEN), "audits": audits,
        "sessions": {s: {"n_units": inventory[s]["n_units"], "n_usable_trials": inventory[s]["n_usable_trials"],
                         "size_bytes": inventory[s]["size_bytes"]} for s in sessions},
        "duplicate_targets": duplicate_target_report([b for b in select_tier(bbs, tier="primary")]),
        "backbones": [{**{k: v for k, v in asdict(bb).items()}, "val_session": bb.val_session,
                       "trunk_sources": list(bb.trunk_sources), "heldout": list(bb.heldout),
                       "selection_scope": bb.selection_scope,
                       "trunk_gap_days": {t.session: bb.trunk_gap_days(t.session) for t in bb.targets}}
                      for bb in bbs],
    }


def bundle_files(backbones: list[Backbone], inventory: dict[str, dict]) -> list[dict]:
    """NWB files needed to run the given backbones (pool + targets), with sizes."""
    sessions = sorted({s for bb in backbones for s in list(bb.pool) + [t.session for t in bb.targets]})
    bad = forbidden_in(sessions)
    if bad:
        raise ValueError(f"forbidden session(s) in bundle: {bad}")
    return [{"session": s, "subject": inventory[s]["subject"], "file": inventory[s]["file"],
             "size_bytes": inventory[s]["size_bytes"]} for s in sessions]


def d1_spec(bb: Backbone, ap: str) -> dict:
    """``D1_POOLS`` entry of a backbone: ``sources`` = trunk sources (they align with the trunk's tables),
    ``heldout`` = trunk-unseen pool[-1] (appended by ``run_population_calib.sessions_for`` as the P0 validation
    session).  Entries of the P1 strata have no ``heldout`` key and are unaffected."""
    spec = dict(sources=",".join(bb.trunk_sources), targets=[t.session for t in bb.targets], ap=ap)
    if bb.heldout:
        spec["heldout"] = list(bb.heldout)
    return spec


def register_d1_pools(backbones: list[Backbone], out_root: str | Path) -> dict:
    """Add each backbone to ``run_labeled_calibration.D1_POOLS`` so the P1 World/driver code can be reused."""
    from scripts.run_labeled_calibration import D1_POOLS
    out_root = Path(out_root)
    for bb in backbones:
        ap = bb.reuse.get("ap") or str(out_root / "backbones" / bb.bid / "final" / "best.pt")
        D1_POOLS[bb.bid] = d1_spec(bb, ap)
    return D1_POOLS


if __name__ == "__main__":
    import sys
    proto = protocol_dict()
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(proto, indent=1))
    print(f"wrote {PROTOCOL_PATH} ({len(proto['backbones'])} backbones)", file=sys.stderr)
