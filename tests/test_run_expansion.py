"""Driver plumbing of scripts/run_expansion.py: storage split (Drive vs scratch), selection, queue script (no GPU)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_expansion as rx                          # noqa: E402


def args(**k):
    base = dict(bid="all", tier="primary", exclude_apst_dev=False, include_2016_optional=False)
    base.update(k)
    return SimpleNamespace(**base)


def test_store_push_pull_roundtrip(tmp_path):
    st = rx.Store(tmp_path / "drive", tmp_path / "scratch")
    assert st.split
    f = st.w("rows/x/a.jsonl")
    f.write_text("1\n")
    d = st.w("runs/b/seed0")
    d.mkdir(parents=True)
    (d / "done.json").write_text("{}")
    assert not st.exists("rows/x/a.jsonl")
    st.push("rows/x/a.jsonl")
    st.push("runs/b/seed0")
    assert st.exists("rows/x/a.jsonl") and st.exists("runs/b/seed0/done.json")
    # a fresh scratch (new Colab VM) pulls the persistent state back
    st2 = rx.Store(tmp_path / "drive", tmp_path / "scratch2")
    st2.pull("rows/x/a.jsonl")
    assert (tmp_path / "scratch2/rows/x/a.jsonl").read_text() == "1\n"
    # overwrite is atomic and leaves no .tmp file behind
    f.write_text("1\n2\n")
    st.push("rows/x/a.jsonl")
    assert (tmp_path / "drive/rows/x/a.jsonl").read_text() == "1\n2\n"
    assert not list((tmp_path / "drive").rglob("*.tmp"))


def test_store_without_scratch_is_noop(tmp_path):
    st = rx.Store(tmp_path)
    assert not st.split
    st.w("a/b.txt").write_text("x")
    st.push("a/b.txt")
    st.pull("a/b.txt")
    assert (tmp_path / "a/b.txt").read_text() == "x"


def test_select_backbones_filters_and_validates():
    allb = rx.select_backbones(args())
    assert [b.bid for b in allb].count("sc_0708") == 1
    assert len(allb) == 8
    one = rx.select_backbones(args(bid="sm_0611,sc_0713"))
    assert {b.bid for b in one} == {"sm_0611", "sc_0713"}
    with pytest.raises(SystemExit):
        rx.select_backbones(args(bid="nope"))
    nodev = rx.select_backbones(args(bid="sc_0708", exclude_apst_dev=True))[0]
    assert not any(t.apst_dev for t in nodev.targets) and len(nodev.targets) == 5
    full = rx.select_backbones(args(bid="sc_0708", tier="all", include_2016_optional=True))[0]
    assert len(full.targets) == 17


def test_reused_backbone_points_to_existing_p1_artefacts(tmp_path):
    bb = rx.select_backbones(args(bid="sm_0307"))[0]
    st = rx.Store(tmp_path)
    assert bb.reuse and rx.ap_path(bb, st) == ROOT / "results/stage2_intervals_20261006/month_pool_k10_x8/best.pt"
    assert rx.p0_dir(bb, st, 1) == ROOT / "results/population_calib_20261015/runs/month_O/p0/seed1"
    new = rx.select_backbones(args(bid="sc_0713"))[0]
    assert rx.ap_path(new, st) == tmp_path / "backbones/sc_0713/final/best.pt"
    assert rx.p0_dir(new, st, 0) == tmp_path / "runs/sc_0713_O/p0/seed0"


def test_ap_command_trains_on_the_first_five_pool_sessions_only(tmp_path):
    bb = rx.select_backbones(args(bid="sc_0629"))[0]
    cmd = rx.ap_cmd(bb, tmp_path / "x", 2000, SimpleNamespace(device="cuda:0", ap_val_interval=100))
    srcs = cmd[cmd.index("--dandi688-sources") + 1].split(",")
    assert srcs == list(bb.pool[:5]) and bb.pool[-1] not in srcs
    assert cmd[cmd.index("--k") + 1] == "10" and cmd[cmd.index("--steps") + 1] == "2000"
    assert cmd[cmd.index("--target-session") + 1] == bb.targets[0].session


def test_queue_script_syntax_and_dry_run(tmp_path):
    script = ROOT / "scripts/run_expansion_queue.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    res = subprocess.run([str(script), "--dry-run", "--out", str(tmp_path / "o"), "--bids", "sm_0307,sm_0611"],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PY": sys.executable,
                                                              "PYTHONPATH": str(ROOT)}, timeout=100)
    assert res.returncode == 0, res.stderr[-500:]
    assert "sm_0307" in res.stdout and "sm_0611" in res.stdout and "sc_0708" not in res.stdout.split("backbones:")[1].split("|")[0]


# ---- 2016 optional tier is off ---------------------------------------------------------------------------

def test_tier_all_needs_the_explicit_flag_and_warns(capsys):
    with pytest.raises(SystemExit, match="include-2016-optional"):
        rx.select_backbones(args(tier="all"))
    with pytest.warns(UserWarning, match="2016"):
        bbs = rx.select_backbones(args(tier="all", include_2016_optional=True))
    assert "WARNING" in capsys.readouterr().err
    assert any("CO-2016" in t.session for b in bbs for t in b.targets)
    # the default selection never contains a 2016 session
    assert not any("CO-2016" in t.session for b in rx.select_backbones(args()) for t in b.targets)


def test_cli_rejects_tier_all_without_flag(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "scripts/run_expansion.py"), "check", "--tier", "all"],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
                                                              "PYTHONNOUSERSITE": "1"}, timeout=100)
    assert res.returncode != 0 and "include-2016-optional" in res.stderr


# ---- OOS split: the backbone never trains on pool[-1] -------------------------------------------------------

def test_every_new_backbone_ap_command_excludes_pool_last(tmp_path):
    for bb in rx.select_backbones(args()):
        cmd = rx.ap_cmd(bb, tmp_path / "x", 2000, SimpleNamespace(device="cpu", ap_val_interval=100))
        srcs = cmd[cmd.index("--dandi688-sources") + 1].split(",")
        if bb.reuse:
            assert srcs == list(bb.pool)                       # sm_0307: reused 6-session trunk (IS selection)
        else:
            assert bb.pool[-1] not in srcs and srcs == list(bb.pool[:5])


def _fake_ckpt(path, sources, k=10):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"source_sessions": list(sources), "args": {"k": k}}, path)


def test_checkpoint_trained_on_pool_last_is_rejected(tmp_path):
    bb = rx.select_backbones(args(bid="sc_0713"))[0]
    st = rx.Store(tmp_path)
    _fake_ckpt(rx.ap_path(bb, st), bb.pool)                     # all six sessions: leaks the selection session
    with pytest.raises(SystemExit, match="held out|trunk sources"):
        rx.check_backbone_matches(bb, st)
    _fake_ckpt(rx.ap_path(bb, st), bb.pool[:5], k=7)
    with pytest.raises(SystemExit, match="K="):
        rx.check_backbone_matches(bb, st)
    _fake_ckpt(rx.ap_path(bb, st), bb.pool[:5])
    rx.check_backbone_matches(bb, st)


def test_p0_world_trains_on_trunk_sessions_and_validates_on_pool_last(monkeypatch, tmp_path):
    """Real ``World`` bookkeeping with stubbed adapter / loader: train_ids are the trunk sources, val = pool[-1]."""
    import numpy as np
    from scripts import run_population_calib as rpc
    bb = next(b for b in rx.select_backbones(args(bid="sc_0629")))
    rx.register([bb], rx.Store(tmp_path))

    class StubAdapter:
        def __init__(self, ckpt, device):
            self.model = SimpleNamespace(parameters=lambda: [])
            self.k, self.label_stats = 10, (np.zeros(2), np.ones(2))
            self.tables = [np.zeros((10, 3)) for _ in range(5)]
    monkeypatch.setattr(rpc, "APBackboneAdapter", StubAdapter)
    monkeypatch.setattr(rpc, "load_dandi688_recording", lambda s: SimpleNamespace(unit_to_electrode=np.arange(3)))
    import torch
    w = rpc.World(bb.bid, "O", torch.device("cpu"), load_sessions=False)
    assert w.orig_ids == list(bb.pool[:5]) and w.ids == list(bb.pool)
    assert w.train_ids == list(bb.pool[:5]) and w.val_ids == [bb.pool[-1]]
    assert bb.pool[-1] not in w.train_ids and bb.pool[-1] not in w.orig_ids
    # P1 strata are unaffected (no heldout key)
    assert rpc.sessions_for("days", "O") == rpc.D1_POOLS["days"]["sources"].split(",")
    # the reused sm_0307 keeps its six-session trunk: validation session is a trunk source (IS)
    reuse = next(b for b in rx.select_backbones(args(bid="sm_0307")))
    rx.register([reuse], rx.Store(tmp_path))
    assert rpc.sessions_for(reuse.bid, "O") == list(reuse.pool)
    from scripts.run_labeled_calibration import D1_POOLS
    for k in (bb.bid, reuse.bid):
        D1_POOLS.pop(k, None)


def test_average_jbar_and_trunk_only_jacobian():
    import torch
    blob = {"lmax": 3, "sessions": {"a": {"sum": torch.ones(4, 2, 2, dtype=torch.float64) * 4, "n": 4},
                                    "b": {"sum": torch.ones(4, 2, 2, dtype=torch.float64) * 12, "n": 4}}}
    jb = rx.average_jbar(blob, 2)
    assert jb.shape == (3, 2, 2) and torch.allclose(jb, torch.full((3, 2, 2), 2.0))
    with pytest.raises(AssertionError):
        rx.average_jbar(blob, 5)


def test_compute_jacobian_uses_trunk_sources_not_the_heldout_session():
    import numpy as np
    import torch
    from torch import nn

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp, self.gru, self.out_proj = nn.Linear(10, 8), nn.GRU(8, 8, batch_first=True), nn.Linear(8, 2)

        def forward(self, x):
            return self.out_proj(self.gru(self.inp(x))[0])

    def sd(t=200, n=4):
        g = torch.Generator().manual_seed(t)
        return SimpleNamespace(valid=torch.ones(t, dtype=torch.bool), trial_id=torch.zeros(t, dtype=torch.long),
                               x32=torch.randn(t, n, generator=g))
    trunk = [f"sub-C_ses-CO-2015060{i}" for i in range(1, 6)]
    held = "sub-C_ses-CO-20150610"
    world = SimpleNamespace(device=torch.device("cpu"), orig_ids=trunk, orig_tables=[np.random.randn(10, 4) for _ in trunk],
                            sessions={**{s: sd() for s in trunk}, held: sd()},
                            adapter=SimpleNamespace(model=SimpleNamespace(base=Base())))
    blob = rx.compute_jacobian(world, windows=200, width=40, lmax=8)
    assert set(blob["sessions"]) == set(trunk) and held not in blob["sessions"] and blob["trunk_sources"] == trunk
    assert all(v["n"] == 161 and v["sum"].shape == (9, 2, 10) for v in blob["sessions"].values())


# ---- forbidden guards in the driver ------------------------------------------------------------------------

def _forbidden_bb():
    pool = tuple(f"sub-C_ses-CO-{d}" for d in ("20150629", "20150630", "20150701", "20150703", "20150706", "20150707"))
    return rx.ep.Backbone("sc_bad", "sub-C", pool, (rx.ep.Target("sub-C_ses-CO-20151113", "month"),))


@pytest.mark.parametrize("final", ["sub-C_ses-CO-20151113", "sub-M_ses-CO-20150623"])
def test_guard_backbone_refuses_final_sessions(final):
    subj = final.split("_")[0]
    pool = tuple(f"{subj}_ses-CO-{d}" for d in ("20150101", "20150102", "20150103", "20150104", "20150105", "20150106"))
    bb = rx.ep.Backbone("bad", subj, pool, (rx.ep.Target(final, "days"),))
    with pytest.raises(PermissionError):
        rx.guard_backbone(bb)
    with pytest.raises(PermissionError):
        rx.guard_backbone(rx.ep.Backbone("bad", subj, pool[:5] + (final,), (rx.ep.Target(pool[0].replace("0101", "0301"), "days"),)))


def _fake_eval_setup(monkeypatch, tmp_path, bb):
    import torch
    from torch import nn
    calls = []
    st_out = tmp_path / "out"
    (st_out / "select").mkdir(parents=True)
    (st_out / "basis").mkdir()
    rule = {"r1s": {"kind": "full", "lam": 1.0}, "r2": {"L": 8, "eta": 1.0, "rho": 0.0, "S": 1}, "r3": {"kind": "full", "lam": 1.0}}
    (st_out / f"select/{bb.bid}.json").write_text(json.dumps({
        "bid": bb.bid, "rule_version": "expansion_v2", "selection_session": bb.val_session, "selection_oos": bb.oos_selection,
        "selection_scope": bb.selection_scope, "b": {str(B): rule for B in rx.ep.B_LIST}}))
    torch.save({"U": torch.zeros(3, 3)}, st_out / f"basis/{bb.bid}.pt")

    class FakeWorld:
        adapter = SimpleNamespace(_hash=lambda: "0" * 64, model=SimpleNamespace(base=SimpleNamespace(
            out_proj=nn.Linear(4, 2))))

        def target(self, session):
            calls.append(session)
            raise RuntimeError("world.target reached")

    class FakeCtx:
        def __init__(self, *a, **k):
            self.world, self.rpc, self.device = FakeWorld(), None, "cpu"
            self.p0 = {}

    monkeypatch.setattr(rx, "select_backbones", lambda a: [bb])
    monkeypatch.setattr(rx, "check_backbone_matches", lambda *a: None)
    monkeypatch.setattr(rx, "register", lambda *a: None)
    monkeypatch.setattr(rx, "Ctx", FakeCtx)
    ns = SimpleNamespace(out=str(st_out), work=None, device="cpu", targets=None, budgets=None, draws=1)
    return ns, calls


def test_eval_checks_forbidden_before_loading_the_target(monkeypatch, tmp_path):
    """Even if the upfront guard were bypassed, the per-target check runs BEFORE ``world.target`` (review item 5)."""
    bb = _forbidden_bb()
    ns, calls = _fake_eval_setup(monkeypatch, tmp_path, bb)
    monkeypatch.setattr(rx, "guard_backbone", lambda b: None)            # bypass the upfront layer
    with pytest.raises(PermissionError):
        rx.cmd_eval(ns)
    assert calls == [], "world.target must not be called for a forbidden session"
    # control: a harmless target does reach world.target (the fake raises RuntimeError there)
    ok = rx.ep.Backbone("sc_ok", "sub-C", bb.pool, (rx.ep.Target("sub-C_ses-CO-20151103", "month"),))
    ns2, calls2 = _fake_eval_setup(monkeypatch, tmp_path / "ok", ok)
    monkeypatch.setattr(rx, "guard_backbone", lambda b: None)
    with pytest.raises(RuntimeError, match="world.target reached"):
        rx.cmd_eval(ns2)
    assert calls2 == ["sub-C_ses-CO-20151103"]


def test_every_stage_refuses_a_backbone_with_a_final_session(monkeypatch, tmp_path):
    bb = _forbidden_bb()
    monkeypatch.setattr(rx, "select_backbones", lambda a: [bb])
    monkeypatch.setattr(rx, "register", lambda *a: None)
    monkeypatch.setattr(rx, "Ctx", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Ctx built")))
    ns = SimpleNamespace(out=str(tmp_path), work=None, device="cpu", ap_steps0=10, ap_attempts=1, ap_val_interval=5,
                         p0_steps=5, p0_min_steps=2, p0_val_interval=5, jac_windows=1, jac_width=8, sel_draws=1,
                         budgets=None, draws=1, targets=None)
    for fn in (rx.cmd_backbone, rx.cmd_p0, rx.cmd_jacobian, rx.cmd_basis_and_select, rx.cmd_eval):
        with pytest.raises(PermissionError):
            fn(ns)


# ---- resume: torn rows, status -----------------------------------------------------------------------------

def test_status_repairs_a_torn_rows_file(tmp_path):
    bb = rx.select_backbones(args(bid="sc_0713"))[0]
    st = rx.Store(tmp_path)
    f = tmp_path / "rows" / bb.bid / f"{rx.ep.date_of(bb.targets[0].session)}.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"method": "p0", "seed": 0, "B": 1, "draw": 0}) + "\n" + '{"method": "p0", "se')
    flags = rx.stage_flags(bb, st)
    assert flags["rows"].startswith("1/") and flags["selection_scope"] == "OOS" and flags["jacobian"] is False
    assert f.read_text().endswith("\n")


# ---- queue exit status ---------------------------------------------------------------------------------------

FAKE_DRIVER = """#!{py}
import os, sys
cmd = sys.argv[1]
bid = sys.argv[sys.argv.index("--bid") + 1] if "--bid" in sys.argv else ""
log = os.environ["FAKE_LOG"]
open(log, "a").write(f"{{cmd}} {{bid}}\\n")
fail = os.environ.get("FAKE_FAIL", "")          # "bid:stage:rc"
if fail and fail.split(":")[0] == bid and fail.split(":")[1] == cmd:
    sys.exit(int(fail.split(":")[2]))
"""


def run_queue(tmp_path, bids, env_extra=None, extra=()):
    drv = tmp_path / "fake_driver.py"
    drv.write_text(FAKE_DRIVER.format(py=sys.executable))
    env = {"PATH": "/usr/bin:/bin", "PY": sys.executable, "PYTHONPATH": str(ROOT), "EXPANSION_DRIVER": str(drv),
           "FAKE_LOG": str(tmp_path / "calls.log"), **(env_extra or {})}
    return subprocess.run([str(ROOT / "scripts/run_expansion_queue.sh"), "--out", str(tmp_path / "o"), "--bids", bids,
                           "--jobs", "2", *extra], capture_output=True, text=True, env=env, timeout=100)


def test_queue_exit_status_zero_when_every_chain_finishes(tmp_path):
    res = run_queue(tmp_path, "a,b")
    assert res.returncode == 0, res.stdout + res.stderr
    calls = (tmp_path / "calls.log").read_text().split("\n")
    for st in ("backbone", "p0", "jacobian", "select", "eval"):
        assert f"{st} a" in calls and f"{st} b" in calls


def test_queue_propagates_a_failed_stage_as_exit_status(tmp_path):
    res = run_queue(tmp_path, "a,b", {"FAKE_FAIL": "b:select:7"})
    assert res.returncode == 7, res.stdout + res.stderr
    calls = (tmp_path / "calls.log").read_text().split("\n")
    assert "eval b" not in calls                      # the failed chain stops ...
    assert "eval a" in calls                          # ... but does not stop the other chain
    assert "FAILED rc=7" in res.stdout


def test_queue_failure_wins_over_deadline_and_first_failure_rc_is_kept(tmp_path):
    res = run_queue(tmp_path, "a,b", {"FAKE_FAIL": "a:p0:5"})
    assert res.returncode == 5


def test_queue_propagates_the_deadline_rc_3(tmp_path):
    res = run_queue(tmp_path, "a,b", extra=("--deadline-hours", "0"))
    assert res.returncode == 3, res.stdout + res.stderr
    assert "deadline reached" in res.stdout
    assert not (tmp_path / "calls.log").exists() or "backbone" not in (tmp_path / "calls.log").read_text().replace("check", "")


def test_queue_preflight_failure_is_exit_1(tmp_path):
    res = run_queue(tmp_path, "a", {"FAKE_FAIL": "a:check:9"})
    assert res.returncode == 1 and "preflight FAILED" in res.stdout


# ---- Colab bundle lists --------------------------------------------------------------------------------------

def test_subc_bundle_member_list_is_primary_only_and_never_forbidden():
    lst = (ROOT / "results/analysis/expansion_20261016/colab_bundle_subC_primary.txt").read_text().split()
    assert len(lst) == 30 and len(set(lst)) == 30
    assert all(m.startswith("sub-C/sub-C_ses-CO-") and m.endswith("_behavior+ecephys.nwb") for m in lst)
    sessions = {m.split("/")[1].removesuffix("_behavior+ecephys.nwb") for m in lst}
    assert not sessions & rx.ep.ALL_FORBIDDEN and not any("CO-2016" in s for s in sessions)
    inv = rx.ep.load_inventory()
    need = {s for b in rx.ep.select_tier(rx.ep.build_backbones(), tier="primary") if b.subject == "sub-C"
            for s in list(b.pool) + [t.session for t in b.targets]}
    assert sessions == need and all(s in inv for s in sessions)
    assert not (ROOT / "results/analysis/expansion_20261016/colab_bundle_subC_all.txt").exists()


def test_code_bundle_files_exist_and_include_the_new_modules():
    files = (ROOT / "results/analysis/expansion_20261016/colab_bundle_code_files.txt").read_text().split()
    assert all((ROOT / f).exists() for f in files) and set(files) == set(rx.EXPANSION_CODE_FILES)
    assert {"ssm_decode/expansion_rules.py", "ssm_decode/atomic_jsonl.py", "scripts/run_expansion_queue.sh"} <= set(files)


def test_score_matches_ep_r2_and_reports_the_unit_check():
    """``score`` (used for the rows) is the same decode as ``Ep.r2`` (used for selection) plus the unit check."""
    import numpy as np
    import torch
    from torch import nn
    from scripts import run_pc3_refine as pc

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj, self.out_proj = nn.Linear(4, 6), nn.Linear(6, 2)

        def forward(self, z):
            return self.out_proj(torch.tanh(self.in_proj(z)))
    torch.manual_seed(0)
    t, n = 300, 5
    beh = np.random.default_rng(0).normal(size=(t, 2)) * 2.0 + 1.0
    q = torch.zeros(t, dtype=torch.bool)
    q[100:250] = True
    stub = SimpleNamespace(base=Base(), xz=torch.randn(t, n), nb=t, q=q,
                           sd=SimpleNamespace(n_bins=t, record=SimpleNamespace(behavior=beh)),
                           world=SimpleNamespace(label_stats=(np.array([1.0, 1.0]), np.array([2.0, 2.0]))))
    for p in stub.base.parameters():
        p.requires_grad_(False)
    table = torch.randn(4, n)
    r2, ok, ratio = rx.score(stub, table)
    assert r2 == pytest.approx(pc.Ep.r2(stub, table), abs=0.0) and isinstance(ok, bool) and np.isfinite(ratio)
    v = torch.randn(2, 7)
    assert rx.score(stub, table, v)[0] == pytest.approx(pc.Ep.r2(stub, table, v), abs=0.0)
    bad = torch.full((4, n), float("nan"))
    assert rx.score(stub, bad) == (-1.0, False, pytest.approx(float("nan"), nan_ok=True))
