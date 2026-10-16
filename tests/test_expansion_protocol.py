"""Pool construction, forbidden-session and tier rules of the expansion protocol (no NWB, no torch)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssm_decode import expansion_protocol as ep                      # noqa: E402

SUBM_FINAL = ("sub-M_ses-CO-20150623", "sub-M_ses-CO-20150625", "sub-M_ses-CO-20150626")
SUBC_FINAL = ("sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
              "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201")


@pytest.fixture(scope="module")
def inv():
    return ep.load_inventory()


@pytest.fixture(scope="module")
def backbones():
    return ep.build_backbones()


def fake_inv(subject="sub-M", dates=("20140101", "20140102", "20140103", "20140104", "20140105", "20140106",
                                      "20140107", "20140110", "20140401", "20150101"), units=30):
    pre = ep.PREFIX[subject]
    return {pre + d: {"session": pre + d, "subject": subject, "date": d, "n_units": units, "n_usable_trials": 200,
                      "size_bytes": 1, "file": "x", "unit_to_electrode": [0], "n_bins": 1} for d in dates}


def bb_of(pool, targets, stratum="days", subject="sub-M", k=10):
    pre = ep.PREFIX[subject]
    return ep.Backbone("t", subject, tuple(pre + p for p in pool),
                       tuple(ep.Target(pre + t, stratum) for t in targets), k=k)


def test_inventory_contains_no_forbidden_session(inv):
    assert not set(inv) & (set(SUBM_FINAL) | set(SUBC_FINAL))
    assert ep.ALL_FORBIDDEN == set(SUBM_FINAL) | set(SUBC_FINAL)


def test_protocol_validates_and_counts(inv, backbones):
    audits = [ep.validate_backbone(bb, inv) for bb in backbones]
    assert len(audits) == 8
    prim = ep.select_tier(backbones, tier="primary")
    assert sum(1 for b in prim if b.subject == "sub-M" and not b.reuse) == 3        # new sub-M backbones
    assert sum(1 for b in prim if b.subject == "sub-C") == 4                         # new sub-C backbones
    assert sum(len(b.targets) for b in prim if b.subject == "sub-C") == 27
    assert sum(len(b.targets) for b in prim if b.subject == "sub-M") == 8
    assert all(b.k == 10 for b in backbones)


def test_no_target_or_posttarget_session_in_any_pool(backbones):
    """Independent re-check of the pool rule (does not call validate_backbone)."""
    for bb in backbones:
        targets = [t.session for t in bb.targets]
        first = min(ep.date_of(t) for t in targets)
        assert len(bb.pool) == 6 and len(set(bb.pool)) == 6
        for s in bb.pool:
            assert s not in targets
            assert ep.date_of(s) < first, (bb.bid, s)
        assert [ep.date_of(s) for s in bb.pool] == sorted(ep.date_of(s) for s in bb.pool)
        assert bb.val_session == bb.pool[-1] and bb.val_session not in targets


def test_no_forbidden_session_anywhere(backbones):
    for bb in backbones:
        used = set(bb.pool) | {t.session for t in bb.targets}
        assert not used & ep.ALL_FORBIDDEN, bb.bid


def test_subm_pools_follow_the_rolling_rule(inv, backbones):
    for bb in backbones:
        if bb.subject == "sub-M" and not bb.reuse:
            first = min(ep.date_of(t.session) for t in bb.targets)
            assert bb.pool == ep.rolling_pool(inv, "sub-M", first), bb.bid
    new = {b.bid: [ep.date_of(t.session) for t in b.targets] for b in backbones}
    assert "20140307" in new["sm_0307"] and "20140627" in new["sm_0627"] and "20150512" in new["sm_0512"]
    assert {"20150612", "20150615"} <= set(new["sm_0611"])


def test_0307_is_target_only_never_selection(backbones):
    """20140307 (causal selection session of the old month rules) is only ever a target or a pool source here."""
    for bb in backbones:
        assert bb.val_session != "sub-M_ses-CO-20140307"
        if "sub-M_ses-CO-20140307" in [t.session for t in bb.targets]:
            assert "sub-M_ses-CO-20140307" not in bb.pool and bb.bid == "sm_0307"


def test_strata_gaps(backbones):
    for bb in backbones:
        for t in bb.targets:
            lo, hi = ep.STRATUM_GAP_DAYS[t.stratum]
            assert lo <= ep.gap_days(bb.pool, t.session) <= hi
            assert t.gap_days == ep.gap_days(bb.pool, t.session)


def test_tiers_and_apst_dev_flag(backbones):
    allt = {t.session: t for b in backbones for t in b.targets if b.bid == "sc_0708"}
    assert all(t.apst_dev for s, t in allt.items() if s.endswith(("1103", "1104", "1106", "1109", "1110", "1112")))
    assert all(t.tier == "optional" for s, t in allt.items() if "CO-2016" in s)
    prim = ep.select_tier(backbones, tier="primary")
    assert not any(t.tier == "optional" for b in prim for t in b.targets)
    nodev = ep.select_tier(backbones, tier="primary", exclude_apst_dev=True)
    assert not any(t.apst_dev for b in nodev for t in b.targets)
    assert sum(len(b.targets) for b in nodev) == sum(len(b.targets) for b in prim) - 6


def test_rejects_target_inside_pool():
    inv = fake_inv()
    with pytest.raises(ValueError, match="inside the pool|not strictly before"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"],
                                   ["20140106"]), inv)


def test_rejects_post_target_session_in_pool():
    inv = fake_inv()
    # pool member 20140107 is dated after target 20140106
    with pytest.raises(ValueError, match="not strictly before|post-target"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140107"],
                                   ["20140106"]), inv)


def test_rejects_wrong_pool_size_unsorted_and_duplicates():
    inv = fake_inv()
    with pytest.raises(ValueError, match="pool must hold"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105"], ["20140107"]), inv)
    with pytest.raises(ValueError, match="sorted"):
        ep.validate_backbone(bb_of(["20140102", "20140101", "20140103", "20140104", "20140105", "20140106"],
                                   ["20140107"]), inv)
    with pytest.raises(ValueError, match="duplicated|distinct"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"],
                                   ["20140107", "20140107"]), inv)


@pytest.mark.parametrize("forbidden", SUBM_FINAL + SUBC_FINAL)
def test_rejects_forbidden_in_pool_and_as_target(forbidden):
    subject = forbidden.split("_")[0]
    dates = ("20150101", "20150102", "20150103", "20150104", "20150105", "20150106", "20150107", "20150108")
    inv = fake_inv(subject, dates + (forbidden.rsplit("-", 1)[1],))
    pool = list(dates[:5]) + [forbidden.rsplit("-", 1)[1]]
    # forbidden dates are late in 2015, so build a pool ending with it and a later target
    inv.update(fake_inv(subject, ("20151231",)))
    with pytest.raises(ValueError, match="forbidden"):
        ep.validate_backbone(bb_of(sorted(pool), ["20151231"], subject=subject), inv)
    with pytest.raises(ValueError, match="forbidden"):
        ep.validate_backbone(bb_of(list(dates[:6]), [forbidden.rsplit("-", 1)[1]], subject=subject), inv)


def test_rejects_session_missing_from_inventory():
    inv = fake_inv()
    inv.pop(ep.PREFIX["sub-M"] + "20140103")
    with pytest.raises(ValueError, match="not in the session inventory"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"],
                                   ["20140107"]), inv)


def test_rejects_wrong_subject_gap_units_trials_and_k():
    inv = fake_inv()
    good = bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"], ["20140107"])
    ep.validate_backbone(good, inv)
    with pytest.raises(ValueError, match="does not belong"):
        ep.validate_backbone(ep.Backbone("t", "sub-C", good.pool, good.targets), inv)
    with pytest.raises(ValueError, match="gap"):
        ep.validate_backbone(bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"],
                                   ["20140401"]), inv)                      # 'days' label, 85-day gap
    low = copy.deepcopy(inv)
    low[ep.PREFIX["sub-M"] + "20140103"]["n_units"] = 10
    with pytest.raises(ValueError, match="units"):
        ep.validate_backbone(good, low)
    few = copy.deepcopy(inv)
    few[ep.PREFIX["sub-M"] + "20140107"]["n_usable_trials"] = 20
    with pytest.raises(ValueError, match="usable trials"):
        ep.validate_backbone(good, few)
    with pytest.raises(ValueError, match="Ledermann"):
        ep.validate_backbone(ep.Backbone("t", "sub-M", good.pool, good.targets, k=30), inv)


def test_rolling_pool_skips_low_unit_and_forbidden_sessions(inv):
    # 20141203 has 10 units: below the Ledermann minimum for K=10, must be skipped
    pool = ep.rolling_pool(inv, "sub-M", "20150512")
    assert "sub-M_ses-CO-20141203" not in pool and pool[-1] == "sub-M_ses-CO-20150511"
    # never reaches a final session even for a late target date
    assert not set(ep.rolling_pool(inv, "sub-M", "20151231")) & set(SUBM_FINAL)
    assert not set(ep.rolling_pool(inv, "sub-C", "20151231")) & set(SUBC_FINAL)


def test_ledermann_k():
    assert ep.ledermann_k(15) == 10 and ep.ledermann_k(10) == 6 and ep.ledermann_k(41) == 10
    assert ep.ledermann_k(41, k_max=40) == 32


def test_bundle_files_refuse_forbidden(inv, backbones):
    files = ep.bundle_files(ep.select_tier(backbones, tier="all"), inv)
    assert not {f["session"] for f in files} & ep.ALL_FORBIDDEN
    bad = ep.Backbone("t", "sub-C", tuple(ep.PREFIX["sub-C"] + d for d in ("20150629", "20150630", "20150701", "20150703", "20150706", "20150707")),
                      (ep.Target(SUBC_FINAL[0], "month"),))
    with pytest.raises(ValueError, match="forbidden"):
        ep.bundle_files([bad], inv)


def test_register_does_not_clobber_p1_pools(backbones, tmp_path):
    from scripts.run_labeled_calibration import D1_POOLS
    before = {k: dict(v) for k, v in D1_POOLS.items()}
    ep.register_d1_pools(backbones, tmp_path)
    for k, v in before.items():
        assert D1_POOLS[k] == v
    bb = next(b for b in backbones if b.bid == "sc_0713")
    assert D1_POOLS["sc_0713"]["sources"].split(",") == list(bb.pool[:-1])        # trunk sources only
    assert D1_POOLS["sc_0713"]["heldout"] == [bb.pool[-1]]
    reused = D1_POOLS["sm_0307"]
    assert reused["sources"].split(",") == list(next(b for b in backbones if b.bid == "sm_0307").pool)
    assert "heldout" not in reused
    for k in [b.bid for b in backbones]:
        D1_POOLS.pop(k, None)


# ---- out-of-sample selection protocol (2026-10-16) ---------------------------------------------------------

def test_new_backbones_hold_pool_last_out_of_trunk_training(backbones):
    for bb in backbones:
        if bb.reuse:
            assert bb.trunk_sources == bb.pool and not bb.heldout and not bb.oos_selection
            assert bb.selection_scope.startswith("IS")
        else:
            assert bb.trunk_sources == bb.pool[:5] and len(bb.trunk_sources) == 5
            assert bb.heldout == (bb.pool[-1],) == (bb.val_session,) and bb.oos_selection
            assert bb.val_session not in bb.trunk_sources
            assert bb.selection_scope.startswith("OOS")
        # causal: the held-out selection session is earlier than every target
        assert all(ep.date_of(bb.val_session) < ep.date_of(t.session) for t in bb.targets)
    assert [b.bid for b in backbones if b.reuse] == ["sm_0307"]


def test_trunk_gap_is_measured_from_the_latest_trunk_training_session(backbones):
    sm0627 = next(b for b in backbones if b.bid == "sm_0627")
    # pool[-1] = 20140626 is held out; the trunk's last training session is 20140307
    assert sm0627.targets[0].gap_days == 1 and sm0627.trunk_gap_days(sm0627.targets[0].session) == 112
    for bb in backbones:
        for t in bb.targets:
            assert bb.trunk_gap_days(t.session) >= t.gap_days
    reused = next(b for b in backbones if b.reuse)
    assert reused.trunk_gap_days(reused.targets[0].session) == reused.targets[0].gap_days


def test_d1_spec_alignment_for_the_p1_world():
    bb = bb_of(["20140101", "20140102", "20140103", "20140104", "20140105", "20140106"], ["20140107"])
    spec = ep.d1_spec(bb, "x.pt")
    assert spec["sources"].split(",") == list(bb.pool[:5]) and spec["heldout"] == [bb.pool[5]]
    reuse = ep.Backbone("t", "sub-M", bb.pool, bb.targets, reuse={"ap": "a"})
    assert "heldout" not in ep.d1_spec(reuse, "x.pt") and ep.d1_spec(reuse, "x.pt")["sources"].split(",") == list(bb.pool)


def test_duplicate_targets_are_reported_by_session(backbones):
    prim = ep.select_tier(backbones, tier="primary")
    subc = [b for b in prim if b.subject == "sub-C"]
    rep = ep.duplicate_target_report(subc)
    assert rep["n_pairs"] == 27 and rep["n_distinct_sessions"] == 19
    dup = {s[-4:]: len(b) for s, b in rep["duplicated"].items()}
    assert dup == {"0713": 3, "0716": 3, "0629": 2, "0706": 2, "0709": 2, "0714": 2}
    # sub-M: sm_0611 targets 0616 / 0617 were P1 design targets
    rep_m = ep.duplicate_target_report([b for b in prim if b.subject == "sub-M"])
    assert rep_m["n_pairs"] == rep_m["n_distinct_sessions"] == 8
    assert set(rep_m["p1_design_targets"]) == {"sub-M_ses-CO-20150616", "sub-M_ses-CO-20150617"}
    # disclosure: sub-C targets that are exam faces of other protocols (not forbidden)
    faces = rep["other_protocol_exam_faces"]
    assert {s[-4:] for s in faces} >= {"0713", "0714", "0716", "1103", "1112"}
    assert not set(faces) & ep.ALL_FORBIDDEN


def test_no_2016_session_in_the_primary_tier(backbones):
    prim = ep.select_tier(backbones, tier="primary")
    assert not any("CO-2016" in t.session for b in prim for t in b.targets)
    assert not any("CO-2016" in s for b in prim for s in b.pool)
    protocol = ep.protocol_dict()
    assert protocol["schema"] == "expansion_20261016_v2" and "duplicate_targets" in protocol
    assert not any("CO-2016" in s for s in protocol["duplicate_targets"]["duplicated"])
