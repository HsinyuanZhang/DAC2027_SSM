"""Round-2 selection pool, resumability, Drive sync, and first-B causality. No GPU."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode import expansion_causal as ec                      # noqa: E402
from ssm_decode import expansion_protocol as ep                    # noqa: E402
from ssm_decode.causal_protocol import TRUNK_LAST, alpha_episode_allowed  # noqa: E402


def _row(tag, date, seed=0, B=8, gap=10, p0=0.2, alpha_r2=None, rebase=None, alpha=None, base=None):
    session = date if date.startswith("sub-") else f"sub-M_ses-CO-{date}"
    if tag.startswith("sc_"):
        session = f"sub-C_ses-CO-{date[-8:]}"
    out = {"tag": tag, "session": session, "seed": seed, "B": B, "gap_days": gap, "p0": p0,
           "protocol": "causal_first_B"}
    if alpha_r2 is not None:
        out["alphas"] = [{"alpha": a, "r2": r} for a, r in alpha_r2]
    if rebase is not None:
        out["rebase"] = rebase
    if alpha is not None:
        out["alpha"] = alpha
    if base is not None:
        out["base"] = base
    return out


def test_frozen_tags_match_alpha_episode_allowed():
    reported = ec.reported_target_dates()
    dates = ["20140203", "20140307", "20140627", "20150319", "20150707", "20131220"]
    targets = ["20140307", "20140627", "20150512", "20150616", "20150708", "20151103", "20150319"]
    for tag in ("days", "month", "year"):
        for date in dates:
            row = _row(tag, date)
            for target in targets:
                for variant in ("expanding", "strict"):
                    got = ec.episode_allowed(row, target, variant, reported)
                    exp = alpha_episode_allowed(row, target, "sub-M", variant, reported)
                    assert got is bool(exp), (tag, date, target, variant, got, exp)


def test_protocol_pools_drop_disallowed_sessions_and_same_trunk_duplicates():
    reported = ec.reported_target_dates()
    # month and year are one trunk. Both scored 20140307; only one copy may vote.
    month = _row("month", "20140307", gap=1, alpha_r2=[(0.0, 0.10), (0.25, 0.20)])
    year = _row("year", "20140307", gap=1, alpha_r2=[(0.0, 0.90), (0.25, 2.0)])
    year_only = _row("year", "20140627", gap=100, alpha_r2=[(0.0, 0.30), (0.25, 0.40)])
    days = _row("days", "20140307", gap=400, alpha_r2=[(0.0, 0.50), (0.25, 0.55)])
    late_held = _row("sm_0611", "20150610", gap=29, alpha_r2=[(0.0, 0.2), (0.25, 0.3)])
    early_held = _row("sc_0309", "20131220", gap=1, alpha_r2=[(0.0, 0.2), (0.25, 0.21)])
    is_row = _row("sm_0307", "20140306", gap=0, alpha_r2=[(0.0, 0.4), (0.25, 0.5)])
    forbidden = _row("month", "20150623", gap=1, alpha_r2=[(0.0, 0.1), (0.25, 0.2)])
    y2016 = _row("sc_0708", "20160909", gap=400, alpha_r2=[(0.0, 0.1), (0.25, 0.2)])
    # sc_0708's real held-out is 20150707, so a forged 20160909 row is also not that held-out.
    sc_held = _row("sc_0708", "20150707", gap=1, alpha_r2=[(0.0, 0.2), (0.25, 0.4)])
    episodes = [month, year, year_only, days, late_held, early_held, is_row, forbidden, y2016, sc_held]

    # Target 20140627: days trunk (last 20150615) is not yet before the target. sm_0611 is after it.
    # 20140307 itself is not strictly before 20140627? It is. month trunk last 20140306 is before it.
    pool_a = ec.selection_pool(episodes, "20140627", "expanding", reported, B=8)
    dates = {r["session"][-8:] for r in pool_a}
    tags = {r["tag"] for r in pool_a}
    assert "year" not in tags                          # same-trunk duplicate of month dropped
    assert "20140307" in dates and sum(r["tag"] == "month" for r in pool_a) == 1
    assert "20140627" not in dates                     # not strictly before the target
    assert "20150610" not in dates and "20150623" not in dates and "20160909" not in dates
    assert "sm_0307" not in tags
    assert "20131220" in dates                         # sc_0309 held-out, trunk before the target
    # The duplicate must not pull alpha toward the year copy's 0.95.
    picked = ec.select_alpha_from_allowed(pool_a, trunk_gap_days=1, B=8)
    assert picked["alpha"] == 0.25
    assert picked["source_r2"] < 0.5

    # Same calendar day on two trunks counts as one session for the conservative gate.
    both = [
        _row("month", "20140307", rebase=[{"eta": 1.0, "rho": 0.0, "r2": 0.3}], p0=0.2),
        _row("days", "20140307", rebase=[{"eta": 1.0, "rho": 0.0, "r2": 0.3}], p0=0.2),
    ]
    # days trunk is not before 20140627, so only month remains. Use a target after the days trunk.
    late = ec.selection_pool(both, "20150716", "expanding", reported, B=8)
    assert len(late) == 2
    comp = ec.composition(late)
    assert comp["n_sessions"] == 1 and comp["n_trunk_sessions"] == 2
    gate = ec.gate_choices(late, eta_r2=1.0, base_key="p0")
    assert gate["session_gate"]["n_sessions"] == 1 and gate["session_gate"]["deploy"] is False
    assert gate["conservative"]["fallback_base"] is True

    # (b) drops reported targets and trunks trained on them.
    pool_b = ec.selection_pool(episodes, "20150716", "strict", reported, B=8)
    b_dates = {r["session"][-8:] for r in pool_b}
    b_tags = {r["tag"] for r in pool_b}
    assert "days" not in b_tags                         # days trunk was trained on reported drift targets
    assert "20140307" not in b_dates                    # sm_0307 target, reported
    assert "20140627" not in b_dates                    # drift target
    assert "20150707" not in b_dates                    # sc_0708 trunk trained on reported sub-C targets
    assert "20150610" not in b_dates                    # drift target, and sm_0611 trunk saw reported targets
    assert "20131220" in b_dates                        # sc_0309 held-out is not a reported target
    assert "20160909" not in b_dates and "20150623" not in b_dates

    # (a) on a late target keeps the year-only session (canonical trunk month) but not a second 20140307.
    late_a = ec.selection_pool(episodes, "20150716", "expanding", reported, B=8)
    assert sum(r["session"].endswith("20140307") and r["tag"] == "year" for r in late_a) == 0
    assert any(r["session"].endswith("20140627") and r["tag"] == "year" for r in late_a)
    assert ec.composition(late_a)["n_sessions"] == len({r["session"][-8:] for r in late_a})


def test_combo_dedupe_keeps_every_alpha_and_still_collapses_the_same_trunk():
    """COMBO is one episode per alpha. The key must not keep only the first-loaded alpha."""
    reported = ec.reported_target_dates()
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows.append(_row("month", "20140307", seed=0, B=8, alpha=alpha, gap=1,
                         rebase=[{"eta": 1.0, "rho": 0.0, "r2": 0.3 + alpha}]))
        rows.append(_row("year", "20140307", seed=0, B=8, alpha=alpha, gap=1,
                         rebase=[{"eta": 1.0, "rho": 0.0, "r2": 0.9}]))
    rows.append(_row("month", "20140307", seed=0, B=8, gap=1, alpha_r2=[(0.0, 0.1), (0.5, 0.2)]))
    rows.append(_row("year", "20140307", seed=0, B=8, gap=1, alpha_r2=[(0.0, 0.9), (0.5, 0.9)]))
    pool = ec.selection_pool(rows, "20150716", "expanding", reported, B=8)
    combo = [r for r in pool if "alphas" not in r]
    alpha_rows = [r for r in pool if "alphas" in r]
    assert len(combo) == 5 and {r["tag"] for r in combo} == {"month"}
    assert {float(r["alpha"]) for r in combo} == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert len(alpha_rows) == 1 and alpha_rows[0]["tag"] == "month"
    assert len(ec.combo_at_alpha(combo, 0.5)) == 1
    # The pre-fix key (no alpha) would have left a single COMBO row, so alpha 0.5 could miss.
    assert len(ec.dedupe_same_trunk(rows)) == 6


def test_real_combo_pool_keeps_518_of_530_episodes():
    legacy = ROOT / "results/causal_main_20261019/select"
    out = ROOT / "results/expansion_round2/x/expansion_out"
    if not (legacy / "combo_month_episodes.jsonl").exists() or not (out / "causal_select").is_dir():
        pytest.skip("round-2 combo episodes are not on this machine")
    bundles = ec.load_episodes(out, legacy)

    def old_len(episodes):
        best = {}
        for r in episodes:
            key = (ec.canonical_trunk(r["tag"]), str(r["session"])[-8:], int(r["seed"]), int(r["B"]))
            prev = best.get(key)
            if prev is None or ec._prefer(r, prev):
                best[key] = r
        return len(best)

    assert len(bundles["combo"]) == 530
    assert old_len(bundles["combo"]) == 238
    assert len(ec.dedupe_same_trunk(bundles["combo"])) == 518
    for fam in ("alpha", "s1", "cca_s1"):
        assert len(ec.dedupe_same_trunk(bundles[fam])) == old_len(bundles[fam])


def test_sc_0629_trunk_train_matches_the_backbone_and_does_not_open_protocol_b():
    from ssm_decode.causal_protocol import TRUNK_LAST, TRUNK_TRAIN
    bb = next(b for b in ep.build_backbones() if b.bid == "sc_0629")
    train = {ep.date_of(s) for s in bb.trunk_sources}
    assert TRUNK_TRAIN["sc_0629"] == train
    assert TRUNK_LAST["sc_0629"] == max(train)
    assert {"20150309", "20150312"} <= train
    reported = ec.reported_target_dates()
    assert train & reported == {"20150309", "20150312"}
    held = ep.date_of(bb.val_session)
    assert held in reported                       # 20150319 is an sc_0309 target
    row = _row("sc_0629", held, gap=1)
    # Strict already drops the held-out because the date itself is reported, so filling
    # TRUNK_TRAIN does not change which episodes vote. Expanding does not read TRUNK_TRAIN.
    assert ec.episode_allowed(row, "20150716", "strict", reported) is False
    assert ec.episode_allowed(row, "20150716", "expanding", reported) is True


def test_sc_0713_heldout_is_allowed_only_when_the_trunk_is_before_the_target():
    reported = ec.reported_target_dates()
    row = _row("sc_0713", "20150710", gap=1)
    # Trunk last is 20150709. The held-out day itself is not before 20150708.
    assert ec.episode_allowed(row, "20150708", "expanding", reported) is False
    assert ec.episode_allowed(row, "20150714", "expanding", reported) is True
    # Trunk train includes 20150708 / 20150709, which are reported targets, so (b) refuses the trunk.
    assert ec.episode_allowed(row, "20150714", "strict", reported) is False


def test_reported_dates_match_the_causal_main_table_and_omit_2016():
    from scripts.run_causal_main import _DRIFT, _SUBC
    assert set(_DRIFT) | set(_SUBC["sc_0629"]) | set(_SUBC["sc_0708"]) == set(ec.CAUSAL_MAIN_REPORTED)
    reported = ec.reported_target_dates()
    assert not any(d.startswith("2016") for d in reported)
    assert "20140307" in reported                      # sm_0307's target; not a new target of anyone else
    bids = {bb.bid for bb in ep.build_backbones() for t in bb.targets if ep.date_of(t.session) == "20140307"}
    assert bids == {"sm_0307"}
    assert not any(t.tier != "primary" for bb in ep.select_tier(ep.build_backbones(), tier="primary")
                   for t in bb.targets)


def test_two_positive_sessions_deploy_and_one_does_not():
    def ep_row(tag, date, seed, r2):
        return _row(tag, date, seed=seed, rebase=[{"eta": 1.0, "rho": 0.0, "r2": r2}], p0=0.2)
    # One calendar day, two P0 seeds: AND can pass (the SE is across seeds) but the
    # conservative gate still refuses, because it is not two sessions.
    one_day = [ep_row("month", "20140307", 0, 0.25), ep_row("month", "20140307", 1, 0.24)]
    one = ec.gate_choices(one_day, 1.0, "p0")
    assert one["session_gate"]["n_sessions"] == 1 and one["session_gate"]["deploy"] is False
    assert one["conservative"]["eta"] == 0.0 and one["conservative"]["fallback_base"] is True
    two_days = one_day + [ep_row("sc_0309", "20131220", 0, 0.23), ep_row("sc_0309", "20131220", 1, 0.22)]
    good = ec.gate_choices(two_days, 1.0, "p0")
    assert good["session_gate"]["n_positive_sessions"] >= 2 and good["session_gate"]["deploy"] is True
    assert good["conservative"]["fallback_base"] is False and good["and"]["fallback_base"] is False


def test_pending_units_and_eval_rows_resume():
    session = "sub-C_ses-CO-20150710"
    done = [_row("sc_0713", "20150710", seed=0, B=8, alpha=0.5),
            {"tag": "sc_0713", "session": session, "seed": 0, "B": 1, "alpha": 0.5, "skipped": "cca_missing"}]
    pending = ec.pending_units(done, session, seeds=(0,), Bs=(1, 8), alphas=(0.5, 1.0), what="combo")
    assert (session, 0, 8, 0.5) not in pending
    assert (session, 0, 1, 0.5) not in pending          # a skip row is finished, not retried
    assert (session, 0, 8, 1.0) in pending
    rows = [{"method": "p0", "seed": 0, "B": 8, "latency": 0}]
    left = ec.pending_eval(rows, ("p0", "p_prior"), (0,), (8,), (0, 1))
    assert ("p0", 0, 8, 0) not in left and ("p0", 0, 8, 1) in left and ("p_prior", 0, 8, 0) in left


def test_real_month_and_year_episodes_collapse_to_one_session():
    legacy = ROOT / "results/causal_main_20261019/select"
    if not (legacy / "alpha_month_episodes.jsonl").exists():
        pytest.skip("legacy causal-main episodes are not on this machine")
    bundles = ec.load_episodes(ROOT / "results/does_not_exist", legacy)
    reported = ec.reported_target_dates()
    pool = ec.selection_pool(bundles["alpha"], "20150512", "expanding", reported, B=8)
    dup = [r for r in pool if r["session"].endswith("20140307") and r["tag"] in ("month", "year")]
    assert dup and {r["tag"] for r in dup} == {"month"}
    assert len(dup) == len({r["seed"] for r in dup})
    assert ec.composition(pool)["n_sessions"] == len({r["session"][-8:] for r in pool})


def test_sync_tree_to_tar_replaces_atomically(tmp_path):
    src = tmp_path / "expansion_out"
    (src / "causal_rows").mkdir(parents=True)
    (src / "causal_rows" / "a.jsonl").write_text('{"method": "p0"}\n')
    dest = tmp_path / "drive" / "expansion_out_round2.tgz"
    ec.sync_tree_to_tar(src, dest)
    assert dest.is_file() and not list(dest.parent.glob(dest.name + ".partial*"))
    import tarfile
    with tarfile.open(dest, "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith("causal_rows/a.jsonl") for n in names)
    # A second sync replaces the same path; the reader never needs the partial name.
    (src / "causal_rows" / "b.jsonl").write_text("{}\n")
    ec.sync_tree_to_tar(src, dest)
    with tarfile.open(dest, "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith("b.jsonl") for n in names)


def test_queue_runs_causal_eval_after_every_chain_and_syncs_on_dry_run(tmp_path):
    script = ROOT / "scripts/run_expansion_queue.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    drv = tmp_path / "fake_driver.py"
    drv.write_text(
        "#!" + sys.executable + "\n"
        "import os, sys\n"
        "cmd = sys.argv[1]\n"
        "bid = sys.argv[sys.argv.index('--bid') + 1] if '--bid' in sys.argv else ''\n"
        "open(os.environ['FAKE_LOG'], 'a').write(f'{cmd} {bid}\\n')\n")
    log = tmp_path / "calls.log"
    tar_path = tmp_path / "drive" / "out.tgz"
    env = {"PATH": "/usr/bin:/bin", "PY": sys.executable, "PYTHONPATH": str(ROOT),
           "EXPANSION_DRIVER": str(drv), "FAKE_LOG": str(log)}
    res = subprocess.run([str(script), "--dry-run", "--out", str(tmp_path / "o"), "--bids", "sm_0307",
                          "--sync-tar", str(tar_path), "--stages", "backbone,causal-eval"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert tar_path.is_file() and not list(tar_path.parent.glob(tar_path.name + ".partial*"))
    res = subprocess.run([str(script), "--out", str(tmp_path / "o2"), "--bids", "a,b", "--jobs", "2",
                          "--stages", "backbone,causal-eval", "--deadline-hours", "5"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = [ln for ln in log.read_text().splitlines() if ln.startswith("backbone") or ln.startswith("causal-eval")]
    assert calls.index("causal-eval a") > max(i for i, c in enumerate(calls) if c.startswith("backbone"))
    assert calls.index("causal-eval b") > max(i for i, c in enumerate(calls) if c.startswith("backbone"))
    # Default stages are unchanged: a dry status line still lists the five original stages when not overridden.
    text = script.read_text()
    assert "STAGES=backbone,p0,jacobian,select,eval\n" in text


def test_sync_loop_kill_does_not_orphan_sleep(tmp_path):
    """SIGTERM to the periodic sync subshell must reap its sleep, or an inherited flock stays held."""
    script = (ROOT / "scripts/run_expansion_queue.sh").read_text()
    assert "trap 'kill \"$sp\" 2>/dev/null; exit 0' TERM" in script
    kill_at = script.index('kill "$SYNC_PID"')
    wait_at = script.index('wait "$SYNC_PID"', kill_at)
    sync_at = script.index("\nsync_out\n", wait_at)
    assert kill_at < wait_at < sync_at
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "#!/bin/bash\nset -u\n"
        "( trap 'kill \"$sp\" 2>/dev/null; exit 0' TERM; sp=; "
        "while true; do sleep 41 & sp=$!; wait \"$sp\"; done ) &\n"
        "P=$!\nsleep 0.3\nkill \"$P\"\nwait \"$P\" 2>/dev/null || true\nsleep 0.2\n"
        "if pgrep -f '[s]leep 41' >/dev/null; then echo ORPHAN; pgrep -af sleep; exit 1; fi\n"
        "echo clean\n")
    res = subprocess.run(["bash", str(probe)], capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "clean" in res.stdout


def test_round2_cost_fits_under_the_cu_cap_and_the_session_limit():
    plan = ec.round2_cost_plan()
    assert plan["upper_cu"] < plan["cu_cap"]
    assert plan["expected_cu"] < plan["upper_cu"]
    for row in plan["sessions"] + plan["fallback_if_slow"]:
        assert row["upper_h"] < plan["session_limit_h"]
        assert row["expected_h"] < row["upper_h"]


def test_first_b_perturbation_at_b8_with_alpha_and_eta():
    """Query bins and an in-block suffix must not move commits that are not allowed to read them."""
    from scripts.run_expansion_causal import first_b_maps
    rng = np.random.default_rng(0)
    B, J = 8, 4
    x = rng.normal(size=(B * 5, 3))
    base, spans = first_b_maps(x, B=B, alpha=0.75, eta=1.0, rho=0.1)
    assert 0.75 > 0 and 1.0 > 0 and B >= 8

    def weights(maps):
        return {k: (np.array(m.weight), np.array(m.bias)) for k, m in maps.items() if k not in (-1, 0)}

    query = x.copy()
    query[spans[B - 1][1]:] = 99.0
    got, _ = first_b_maps(query, B=B, alpha=0.75, eta=1.0, rho=0.1)
    for name in ("w_alpha", "combo", "s1"):
        for k, (w, b) in weights(base[name]).items():
            gw, gb = weights(got[name])[k]
            assert np.allclose(w, gw) and np.allclose(b, gb), (name, k)

    prefix = x.copy()
    prefix[spans[J - 1][1]:] += 5.0
    got2, _ = first_b_maps(prefix, B=B, alpha=0.75, eta=1.0, rho=0.1)
    for name in ("w_alpha", "combo", "s1"):
        for k, (w, b) in weights(base[name]).items():
            if k > J:
                continue
            gw, gb = weights(got2[name])[k]
            assert np.allclose(w, gw, atol=1e-5) and np.allclose(b, gb, atol=1e-5), (name, k)
        bw, bb = weights(base[name])[B]
        gw, gb = weights(got2[name])[B]
        assert np.max(np.abs(bw - gw)) > 1e-4 or np.max(np.abs(bb - gb)) > 1e-4, name


def test_calibration_index_is_the_first_b_trials():
    from scripts.run_expansion_causal import calibration_index
    assert list(calibration_index(20, 8)) == list(range(8))
    with pytest.raises(ValueError):
        calibration_index(4, 8)


def test_margin_pick_handles_episodes_with_different_eta_grids():
    """Legacy S1 episodes have no eta = 0 point, local ones do: the pick uses the common grid (no StopIteration)."""
    from ssm_decode.causal_protocol import margin_pick

    def ep(rows, p0=0.5):
        return {"p0": p0, "rebase": [{"eta": e, "rho": r, "r2": v} for e, r, v in rows]}

    local = [ep([(0.0, 0.0, 0.5), (1.0, 0.0, 0.52), (3.0, 0.0, 0.5)]) for _ in range(4)]
    legacy = [ep([(1.0, 0.0, 0.52), (3.0, 0.0, 0.5)]) for _ in range(4)]
    mixed = margin_pick(local + legacy, "rebase", 3.0, mode="and")
    only_legacy = margin_pick(legacy, "rebase", 3.0, mode="and")
    assert mixed["eta"] == 1.0 and mixed["fallback_base"] is False
    assert (mixed["eta"], mixed["rho"]) == (only_legacy["eta"], only_legacy["rho"])
    assert margin_pick(legacy + local, "rebase", 3.0, mode="and")["eta"] == 1.0
