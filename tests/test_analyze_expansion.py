"""Session-clustered aggregation of expansion rows (synthetic rows; no torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_expansion as ax                      # noqa: E402

METHODS = ("p_prior", "p0", "r1s", "r2", "r3")
GAIN = {"p_prior": -0.1, "p0": 0.0, "r1s": 0.02, "r2": 0.10, "r3": 0.12}


def make_rows(bid, targets, oos=True, stratum="days", subject="sub-C", offset=0.0, trunk_gap=3, B_list=(4, 8), n_draws=2):
    rows = []
    for ti, t in enumerate(targets):
        for B in B_list:
            for draw in range(n_draws):
                for seed in (0, 1):
                    for m in METHODS:
                        if m == "p_prior" and seed != 0:
                            continue
                        base = 0.3 + 0.05 * ti + offset
                        rows.append({"bid": bid, "subject": subject, "target": t, "date": t[-8:], "stratum": stratum,
                                     "gap_days": 3, "trunk_gap_days": trunk_gap, "tier": "primary", "apst_dev": False,
                                     "B": B, "draw": draw, "seed": seed, "method": m, "r2": base + GAIN[m],
                                     "rule_version": "expansion_v2", "selection_oos": oos,
                                     "selection": "OOS" if oos else "IS", "h1_trunk_hash": "abc",
                                     "h1_trunk_unchanged": True, "h2_backward_calls": 0, "h4_query_disjoint": True,
                                     "unit_check_passed": True, "h5_within_32kb": True})
    return rows


def write(tmp_path, rows):
    by = {}
    for r in rows:
        by.setdefault((r["bid"], r["target"]), []).append(r)
    for (bid, t), rs in by.items():
        f = tmp_path / "rows" / bid / f"{t[-8:]}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps(r) + "\n" for r in rs))
    return tmp_path / "rows"


T = lambda d: f"sub-C_ses-CO-{d}"                                 # noqa: E731


def test_clusters_by_target_session_not_by_backbone_target_pair(tmp_path):
    # session 0713 is evaluated by THREE backbones, 0714 by two, 0715 by one -> 6 pairs but 3 distinct sessions
    rows = (make_rows("b1", [T("20150713"), T("20150714")], offset=0.0)
            + make_rows("b2", [T("20150713"), T("20150714"), T("20150715")], offset=0.1)
            + make_rows("b3", [T("20150713")], offset=-0.1))
    out = ax.analyze(write(tmp_path, rows), tmp_path / "out")
    sc = pd.read_csv(tmp_path / "out/paired_session_cluster.csv")
    r = sc[(sc.comparison == "r2-p0") & (sc.B == 4)].iloc[0]
    assert r.n_sessions == 3 and r.n_backbone_target_pairs == 6 and r.multi_backbone_sessions == 2
    assert r["mean"] == pytest.approx(GAIN["r2"]) and r.n_positive_sessions == 3
    assert out["flags"]["n_backbone_target_pairs"] == 6 and out["flags"]["n_distinct_target_sessions"] == 3
    # the pooled B4-33 row exists (B == -1)
    assert ((sc.B == -1) & (sc.comparison == "r3-p0")).any()


def test_session_cluster_ci_is_wider_than_the_naive_pair_ci_when_pairs_repeat(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for k, d in enumerate(("20150713", "20150714", "20150715", "20150716")):
        for bid in ("b1", "b2", "b3"):
            rr = make_rows(bid, [T(d)], offset=0.0)
            for r in rr:
                if r["method"] == "r2":
                    r["r2"] += 0.05 * k                                  # session effect, identical across backbones
            rows += rr
    cells = ax.paired_cells(pd.DataFrame(rows))
    sc = ax.session_cluster(cells)
    one = sc[(sc.comparison == "r2-p0") & (sc.B == 4)].iloc[0]
    x = cells[(cells.comparison == "r2-p0") & (cells.B == 4)]["diff"].to_numpy()       # 12 pairs
    naive_lo, naive_hi = ax.boot_ci(x)
    assert one.n_sessions == 4 and len(x) == 12
    assert (one.boot_hi - one.boot_lo) >= (naive_hi - naive_lo) - 1e-9
    assert rng is not None


def test_backbone_spread_is_reported(tmp_path):
    rows = make_rows("b1", [T("20150713")], offset=0.0) + make_rows("b2", [T("20150713")], offset=0.0)
    for r in rows:
        if r["bid"] == "b2" and r["method"] == "r2":
            r["r2"] += 0.04
    ax.analyze(write(tmp_path, rows), tmp_path / "out")
    sp = pd.read_csv(tmp_path / "out/backbone_spread.csv")
    r = sp[(sp.comparison == "r2-p0") & (sp.B == 4)].iloc[0]
    assert r.n_backbones == 2 and r.within_session_spread_max == pytest.approx(0.04)
    assert r.backbone_mean_max - r.backbone_mean_min == pytest.approx(0.04)


def test_is_and_oos_rows_are_never_merged(tmp_path):
    rows = (make_rows("sm_0307", ["sub-M_ses-CO-20140307"], oos=False, subject="sub-M")
            + make_rows("sm_0627", ["sub-M_ses-CO-20140627"], oos=True, subject="sub-M", trunk_gap=112))
    ax.analyze(write(tmp_path, rows), tmp_path / "out")
    sc = pd.read_csv(tmp_path / "out/paired_session_cluster.csv")
    grp = sc[(sc.subject == "sub-M") & (sc.comparison == "r2-p0") & (sc.B == 4)]
    assert sorted(grp.selection_oos.tolist()) == [False, True] and (grp.n_sessions == 1).all()
    # one-session groups have no CI, never a fabricated one
    assert grp.boot_lo.isna().all() and grp.t_lo.isna().all()
    bt = pd.read_csv(tmp_path / "out/paired_session_cluster_by_trunk_gap.csv")
    assert set(bt.trunk_stratum) == {"days", "month"}               # sm_0627's "days" target is 112 d after its trunk
    flags = json.loads((tmp_path / "out/flags.json").read_text())
    assert flags["is_backbones"] == ["sm_0307"] and flags["oos_backbones"] == ["sm_0627"]


def test_rows_of_other_protocols_are_refused(tmp_path):
    rows = make_rows("b1", [T("20150713")])
    for r in rows:
        r.pop("rule_version")                                         # e.g. P3 / drift-curve rows
    with pytest.raises(ValueError, match="rule_version"):
        ax.load_rows(write(tmp_path, rows))
    rows = make_rows("b1", [T("20150713")])
    rows[0]["rule_version"] = "p3_r12"
    with pytest.raises(ValueError, match="rule_version"):
        ax.load_rows(write(tmp_path / "y", rows))


def test_torn_last_line_does_not_break_the_aggregation(tmp_path):
    rows = make_rows("b1", [T("20150713")])
    root = write(tmp_path, rows)
    f = next(root.rglob("*.jsonl"))
    f.write_text(f.read_text() + '{"bid": "b1", "target": "x", "r2"')
    df = ax.load_rows(root)
    assert len(df) == len(rows)


def test_audit_flags_detect_violations(tmp_path):
    rows = make_rows("b1", [T("20150713")])
    rows[3]["h2_backward_calls"] = 2
    rows[5]["unit_check_passed"] = False
    flags = ax.audit_flags(pd.DataFrame(rows))
    assert flags["h2_zero_backward_all"] is False and flags["unit_check_failures"] == 1
    assert flags["h4_query_disjoint_all"] is True and flags["h1_one_hash_per_backbone"] is True
