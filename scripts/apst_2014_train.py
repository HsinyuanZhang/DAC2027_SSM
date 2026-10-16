#!/usr/bin/env python
"""Retrain APST (authors' frozen F0-FiLM recipe, seed 42) on the 2014 sub-M source roster.

The released APST sub-M campaign hard-codes the 2015 roster in ``apst.dandi_subm.protocol``.
Our month / year strata use six 2014 source sessions (the same sessions as the frozen AP
backbones), so we override ONLY the roster constants at run time and run the authors'
unchanged pipeline (prepare -> concat/activity pretrain -> dual_site / act_only train,
24 x 3165 updates, same recipe, same seed).  The "dev" session of the authors' loop is
20140307: a sub-M CO session after the six sources and before both targets (it is not a
target and never an evaluation session).  No 2015 session, no target session and no
FINAL session is opened.

  python scripts/apst_2014_train.py prepare
  python scripts/apst_2014_train.py train --name sua_dual_site_F0_s42   (needs its encoder run first)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssm_decode.dandi688_data import dandi688_subm_root                      # noqa: E402

OUT = Path(os.environ.get("APST14_ROOT", str(ROOT / "results/baselines_20261016/apst2014")))
os.environ["APST_DATA_ROOT"] = str(dandi688_subm_root())
os.environ["APST_SUBM_ROOT"] = str(OUT)

from apst.dandi_subm import protocol                                         # noqa: E402

SOURCES = ("sub-M_ses-CO-20140203", "sub-M_ses-CO-20140217", "sub-M_ses-CO-20140218",
           "sub-M_ses-CO-20140303", "sub-M_ses-CO-20140304", "sub-M_ses-CO-20140306")
DEV = ("sub-M_ses-CO-20140307",)
protocol.TRAIN_SESSIONS = SOURCES
protocol.DEV_SESSIONS = DEV
assert not set(SOURCES + DEV) & set(protocol.FINAL_SESSIONS)

from apst.dandi_subm.contract import RECIPE, SCHEMA, sha                      # noqa: E402
from apst.dandi_subm.subm_data import load_pair, save_session                # noqa: E402

CONFIGS = [("sua_concat_pretrain_s42", "identity_only", "pretrain", None),
           ("sua_dual_site_F0_s42", "dual_site", "train", "sua_concat_pretrain_s42"),
           ("sua_activity_pretrain_s42", "act_only", "pretrain", None),
           ("sua_act_only_F0_s42", "act_only", "train", "sua_activity_pretrain_s42")]


def cmd_prepare():
    dest = OUT / "prepared_sua"
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    rows = {}
    for split, ids, purpose in (("train", SOURCES, "source"), ("dev", DEV, "development")):
        for sid in ids:
            rec = load_pair(sid, raw_root=protocol.raw_root(), purpose=purpose)["sua"]
            path = dest / f"{sid}.sua.npz"
            save_session(rec, path)
            rows[sid] = {"split": split, "file": path.name, "sha256": sha(path),
                         "channels": int(rec.neural.shape[1]), "legal_trials": len(rec.metadata["raw_trial_rows"])}
    out = {"schema": "dandi688_subm_prepared_sua_v1", "protocol": protocol.protocol_dict(), "sessions": rows,
           "final_sessions_opened_for_scoring": 0, "sua_only": True,
           "note": "2014 roster override of the released 2015 protocol; see scripts/apst_2014_train.py"}
    (dest / "prepared_receipt.json").write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    receipt = dest / "prepared_receipt.json"
    for name, method, stage, enc in CONFIGS:
        c = {"schema": SCHEMA, "output_root": str(OUT), "cache": str(dest), "prepared_receipt_sha256": sha(receipt),
             "protocol": protocol.protocol_dict(), "recipe": RECIPE, "representation": "sua", "seed": 42,
             "method": method, "stage": stage, "temporal": "F0", "target_parameter_updates": False,
             "output_ema": {"alpha": 1 / 3, "training": False, "clock": "full_recording", "reset": "session",
                            "state_0": "pred_0"},
             "encoder_family": "concat" if method in ("identity_only", "dual_site") else "activity",
             "identity_profile": method in ("identity_only", "dual_site"), "token_profile": method == "dual_site",
             "film": method == "dual_site",
             "evaluation_conditions": ["intact"] if stage == "train" else [],
             "shuffle_seed": 42, "run_dir": str(OUT / "runs" / name)}
        if enc:
            c["encoder_source"] = str(OUT / "runs" / enc)
        p = OUT / "configs" / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(c, sort_keys=True, indent=2) + "\n")
        p.with_suffix(".json.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n")
    print(json.dumps({"status": "PASS", "sessions": len(rows)}))


def cmd_train(name, device, smoke):
    from apst.dandi_subm.contract import load
    from apst.dandi_subm.training import run_training
    config = load(OUT / "configs" / f"{name}.json")
    encoder = Path(config["encoder_source"]) / "encoder.pt" if config["stage"] == "train" else None
    run_training(config, Path(config["run_dir"]), encoder_path=encoder, device=device,
                 smoke_updates=2 if smoke else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "train"])
    p.add_argument("--name")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cmd_prepare() if a.cmd == "prepare" else cmd_train(a.name, a.device, a.smoke)


if __name__ == "__main__":
    main()
