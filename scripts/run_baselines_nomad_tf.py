#!/usr/bin/env python
"""NoMAD (Karpowicz et al. 2025; github.com/snel-repo/nomad, unchanged code) on 000688 sub-M.

Runs in the TF env (/mnt/data/work_host/envs/nomad_tf; TF 2.15, the repo pins TF 2.0).  Stages
  day0     train BehaviorLFADS on the baseline day (spikes + standardised velocity, the repo's demo config)
  decoder  causal gen_states of the baseline day -> 3-lag ridge decoder (demo's ``fit_and_eval_decoder``)
  align    NoMAD alignment network on ALL unlabeled target-session spikes (frozen LFADS, KL on generator
           states + Poisson NLL), then causal gen_states of the whole target recording -> decoder ->
           velocity predictions for every bin  (results/baselines_20261016/nomad/preds/<target>.npz)
Scoring on the shared (target, B, draw) query masks is done by scripts/run_baselines_nomad_score.py
(spint env).  Target data used: the full unlabeled spike recording (query bins included) and target-side
gradient training of the alignment network: a non-deployable, H4-violating reference.

Deviations from the authors' demo (documented in the README): DATA_DIM = baseline-day unit count with the
target zero-padded; spikes are the sorted-unit 20 ms counts of our loader; per-unit z-score normalisation
(diag matrix + bias) replaces the demo's pre-computed normalization.h5; eager mode is not forced.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xinyuan/Work_host/SSM")
NOMAD = Path("/home/xinyuan/Work_host/nomad")
sys.path[:0] = [str(NOMAD / "lfads_tf2"), str(NOMAD / "nomad"), str(NOMAD / "demo")]
BASE = ROOT / "results/baselines_20261016/nomad"

import git as _git                                   # the local nomad checkout is not a git repository


class _StubRepo:                                     # lfads_tf2 only records the commit hash for provenance
    def __init__(self, path=None, **kw):
        class _O:
            hexsha = "local-copy-no-git"
        class _H:
            object = _O()
        class _I:
            @staticmethod
            def diff(x):
                return []
        self.head, self.index, self.untracked_files = _H(), _I(), []


_git.Repo = _StubRepo


def _patch_keras_compat():
    """TF 2.0 -> 2.15 serialisation shim for lfads_tf2.layers.ClippedGRUCell (architecture unchanged:
    the cell is built with reset_after=False explicitly, so the GRU equations are the same)."""
    import tensorflow as tf
    if int(tf.__version__.split(".")[1]) < 10:         # TF 2.6 env: the repo code runs unchanged
        return
    import lfads_tf2.layers as L
    from lfads_tf2.regularizers import DynamicL2

    def get_config(self):
        base = tf.keras.layers.GRUCell.get_config(self)
        base.update({"clip_value": float(self.clip_value.numpy()), "train_h0": self.train_h0})
        return base

    def from_config(cls, config):
        config = dict(config)
        reg = config.pop("recurrent_regularizer")
        config["recurrent_regularizer"] = tf.keras.regularizers.deserialize(reg, custom_objects={"DynamicL2": DynamicL2})
        return cls(**config)

    L.ClippedGRUCell.get_config = get_config
    L.ClippedGRUCell.from_config = classmethod(from_config)
SEQ, STRIDE = 30, 24


def chop(x, y, seq=SEQ, stride=STRIDE):
    starts = np.arange(0, len(x) - seq + 1, stride)
    xs = np.stack([x[s:s + seq] for s in starts]).astype(np.float32)
    ys = np.stack([y[s:s + seq] for s in starts]).astype(np.float64)
    return xs, ys, starts


def pad_units(x, d):
    if x.shape[1] > d:
        raise ValueError(f"target has {x.shape[1]} units > DATA_DIM {d}")
    out = np.zeros((x.shape[0], d), dtype=x.dtype)
    out[:, :x.shape[1]] = x
    return out


def write_h5(path, xs, ys):
    import h5py
    n = len(xs)
    idx = np.arange(n)
    valid = idx % 5 == 4                      # every 5th window validates (as in the demo data)
    with h5py.File(path, "w") as f:
        f["train_data"], f["valid_data"] = xs[~valid], xs[valid]
        f["train_behavior"], f["valid_behavior"] = ys[~valid], ys[valid]
        f["train_inds"], f["valid_inds"] = idx[~valid], idx[valid]


def write_norm(path, x):
    import h5py
    mean = x.mean(0)
    std = np.where(x.std(0) < 1e-6, 1.0, x.std(0))
    with h5py.File(path, "w") as f:
        f["matrix"] = np.diag(1.0 / std)
        f["bias"] = -mean / std


def load_npz(stratum, name):
    return np.load(BASE / "data" / stratum / f"{name}.npz", allow_pickle=True)


def lfads_yaml(d, data_dir, model_dir, max_epochs):
    return f"""MODEL:
  DATA_DIM: {d}
  SEQ_LEN: {SEQ}
  ENC_INPUT_DIM: 50
  IC_ENC_DIM: 100
  CI_ENC_DIM: 100
  CON_DIM: 100
  CO_DIM: 4
  IC_DIM: 100
  GEN_DIM: 100
  FAC_DIM: 30
  DROPOUT_RATE: 0.05
  CD_RATE: 0.3
  BEHAVIOR_DIM: 2
TRAIN:
  DATA:
    DIR: '{data_dir}/'
    PREFIX: lfads
    AUGMENT:
      JITTER_WIDTH: 2
  PATIENCE: 10
  MAX_EPOCHS: {max_epochs}
  BATCH_SIZE: 1000
  ADAM_EPSILON: 1.0e-8
  LR:
    INIT: 1.0e-3
    STOP: 1.0e-6
    DECAY: 0.95
    DECODER: 1.0e-3
  L2:
    INCREASE_EPOCH: 100
    GEN_SCALE: 1.0e-6
    CON_SCALE: 1.0e-6
  KL:
    INCREASE_EPOCH: 100
    IC_WEIGHT: 1.0e-4
    CO_WEIGHT: 1.0e-4
  DECODE:
    FROM: 'gen_states'
    SCALE: 0.1
    INCREASE_EPOCH: 100
    N_DELAY_BINS: 0
  TUNE_MODE: False
  OVERWRITE: True
  PBT_MODE: False
  USE_TB: False
  MODEL_DIR: {model_dir}/
"""


def nomad_yaml(d, model_dir, align_dir, max_epochs):
    return f"""MODEL:
  DATA_DIM: {d}
  SEQ_LEN: {SEQ}
  TYPE: FFN-KL
  HIDDEN_DIM: 50
  DAY0_MODEL_TYPE: 'behavior'
  ALIGN_DATA:
    - gen_states
  DAYK_LFADS_TRAINABLES:
    - lowd_readin
    - generator
    - rate_linear
    - decoder.rnn.cell.fac_linear
  SAMPLE_POSTERIORS: True
  INIT_READIN_FROM_DAY0: True
TRAIN:
  EARLY_STOPPING_COST: 'NLL-KL'
  ALIGN_LOSS_WEIGHT:
    - 1.0
  DATA:
    DAY0: ''
    DAYK: ''
  PATIENCE: 10
  BATCH_SIZE: 300
  MAX_EPOCHS: {max_epochs}
  MAX_GRAD_NORM: 2000.0
  LFADS_LOSS_WEIGHT: 1.0
  NLL:
    INCREASE_EPOCH: 100
    WEIGHT: 10.
  LR:
    INIT: 4.e-4
    STOP: 1.0e-9
    DECAY: 0.95
    PATIENCE: 6
  KL:
    START_EPOCH: 0
    INCREASE_EPOCH: 10
    IC_WEIGHT: 1.e-4
    CO_WEIGHT: 1.e-4
  L2:
    READIN_SCALE: 5.e-3
  LOSS_SCALE: 1.0e+4
  USE_TB: False
  MODEL_DIR: {model_dir}/
  ALIGN_DIR: {align_dir}/
  OVERWRITE: True
"""


def ref_name(stratum):
    return json.loads((BASE / "data" / stratum / "meta.json").read_text())["ref"]


def _patch_autograph_df():
    """TF >= 2.1 autograph tries to carry ``self.train_df[...] = <symbolic tensor>`` out of the
    ``if self.training`` cond; make DataFrame item assignment ignore symbolic tensors (the assignment only
    ever runs at trace time with a python condition, as in the original TF 2.0 behaviour)."""
    import pandas as pd
    import tensorflow as tf
    orig = pd.DataFrame.__setitem__

    def setitem(self, key, value):
        if tf.is_tensor(value) and not hasattr(value, "numpy"):
            return
        try:
            if tf.is_tensor(value) and not tf.executing_eagerly() and not hasattr(value, "_numpy"):
                return
        except Exception:
            pass
        return orig(self, key, value)
    pd.DataFrame.__setitem__ = setitem


def _fix_graph_step(model, d):
    """nomad.models.AlignLFADS builds ``tf.function(_step, input_signature=[4 specs])`` for a 3-argument
    ``_step(d0, dk, validation)``; TF 2.0 tolerated the extra spec, TF >= 2.1 does not.  Re-create the traced
    functions with the matching signature (validation is a static python flag, same computation)."""
    import tensorflow as tf
    spec = tf.TensorSpec(shape=[None, SEQ, d])
    train = tf.function(lambda a, b: model._step(a, b, False), input_signature=[spec, spec])
    val = tf.function(lambda a, b: model._step(a, b, True), input_signature=[spec, spec])

    def step(a, b, validation=False):
        return (val if validation else train)(a, b)
    model._graph_step = step


def _patch_align_call():
    """nomad.models.AlignLFADS.call builds ``np.zeros(shape=tf.shape(...))``, which fails when traced in TF >= 2.1;
    same tensor built with tf.zeros (it is an unused placeholder for the behaviour input)."""
    import tensorflow as tf
    from lfads_tf2.tuples import LFADSInput
    from nomad.models import AlignLFADS

    def call(self, data, training=False, use_logrates=False):
        lfads_input = LFADSInput(
            enc_input=data,
            ext_input=tf.zeros(tf.concat([tf.shape(data)[:-1], [0]], axis=0)),
            dataset_name=tf.fill((tf.shape(data)[0],), ""),
            behavior=tf.zeros(tf.shape(data)[:1], dtype=tf.float32))
        return self.lfads_dayk(lfads_input, use_logrates=use_logrates)
    AlignLFADS.call = call


def _patch_armvn_log_prob():
    """Same AR(1) controller-output prior log-probability as lfads_tf2.layers.AutoregressiveMultivariateNormal,
    written as an unrolled loop over the 30 time steps: the tf.map_fn version raises a TensorList shape error
    inside NoMAD's graph step on TF 2.6.  Equations are identical (first step: N(0, pvar); then N(alpha*prev, nvar))."""
    import tensorflow as tf
    import tensorflow_probability as tfp
    import lfads_tf2.layers as L
    tfd = tfp.distributions

    def _log_prob(self, sample):
        alphas = tf.math.exp(-1.0 / tf.math.exp(self.logtaus))
        logpvars = self.lognvars - tf.math.log(1 - alphas ** 2)
        samples = tf.unstack(sample, axis=1)
        total = tfd.MultivariateNormalDiag(tf.zeros_like(samples[0]), tf.math.exp(0.5 * logpvars)).log_prob(samples[0])
        for prev, cur in zip(samples[:-1], samples[1:]):
            total += tfd.MultivariateNormalDiag(alphas * prev, tf.math.exp(0.5 * self.lognvars)).log_prob(cur)
        return total
    L.AutoregressiveMultivariateNormal._log_prob = _log_prob


def _eager():
    """The authors' demo notebook runs with tf.config.experimental_run_functions_eagerly(True)."""
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)


def cmd_day0(a):
    _patch_keras_compat()
    _patch_autograph_df()
    from lfads_tf2.subclasses.behavior.models import BehaviorLFADS
    ref = load_npz(a.stratum, ref_name(a.stratum))
    x, y = ref["neural"].astype(np.float32), ref["y_norm"]
    d = x.shape[1]
    root = BASE / "models" / a.tag
    data_dir = root / "data_day0"
    data_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, _ = chop(x, y)
    write_h5(data_dir / "lfads_day0.h5", xs, ys)
    write_norm(data_dir / "normalization.h5", x)
    cfg = root / "lfads.yaml"
    cfg.write_text(lfads_yaml(d, data_dir, root / "day0_model", a.max_epochs))
    model = BehaviorLFADS(cfg_path=str(cfg))
    model.train()
    print("day0 trained", root)


def causal_states(model, x, d, batch_size=256):
    """Causal generator states of a (possibly aligned) LFADS model over a continuous recording: 30-bin sliding
    windows, stride 1, output of each window's last bin (same as the authors' ``get_causal_model_output``;
    the behaviour placeholder gets the model's BEHAVIOR_DIM so that the traced signature matches)."""
    from lfads_tf2.tuples import LFADSInput
    from utils import chop_and_infer
    cfg = model.cfg
    gd, seq, dd = cfg.MODEL.GEN_DIM, cfg.MODEL.SEQ_LEN, cfg.MODEL.DATA_DIM
    cfg["MODEL"]["SAMPLE_POSTERIORS"] = False            # posterior means, as in the authors' function
    data = np.zeros((batch_size, seq, dd), dtype=np.float32)
    ext = np.zeros((batch_size, seq, cfg.MODEL.EXT_INPUT_DIM), dtype=np.float32)
    beh = np.zeros((batch_size, seq, cfg.MODEL.BEHAVIOR_DIM), dtype=np.float32)
    name = np.full(shape=[1], fill_value="")

    def infer(data_in):
        data[:] = data_in
        out = model.graph_call(LFADSInput(enc_input=data, ext_input=ext, dataset_name=name, behavior=beh))
        return {"gen_states": out.gen_states[:, :seq, :].numpy()}
    xin = pad_units(x, d).astype(np.float32)
    return chop_and_infer(infer, xin, ["gen_states"], stride=1, seq_len=seq, batch_size=batch_size,
                          output_dim={"gen_states": gd})["gen_states"]


def lag_stack(g, lag=3):
    rows = [np.roll(g, i, axis=0) for i in range(lag + 1)]
    out = np.concatenate(rows, axis=1)
    out[:lag] = np.nan                                  # not enough history
    return out


def cmd_decoder(a):
    _eager()
    from lfads_tf2.subclasses.behavior.models import BehaviorLFADS
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    ref = load_npz(a.stratum, ref_name(a.stratum))
    x, y = ref["neural"].astype(np.float32), ref["y_norm"].astype(np.float64)
    d = x.shape[1]
    root = BASE / "models" / a.tag
    model = BehaviorLFADS(model_dir=str(root / "day0_model"))
    g = causal_states(model, x, d)
    lg = lag_stack(g)
    tid = np.zeros(len(x), bool)
    for s, e in ref["usable"]:
        tid[s:e] = True
    ok = tid & ref["eval_mask"] & np.isfinite(lg).all(1)
    ok[:SEQ] = False
    dec = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)})
    dec.fit(lg[ok], y[ok])
    pred = dec.predict(lg[ok])
    r2 = 1 - ((pred - y[ok]) ** 2).sum() / ((y[ok] - y[ok].mean(0)) ** 2).sum()
    import pickle
    pickle.dump({"coef": dec.best_estimator_.coef_, "intercept": dec.best_estimator_.intercept_,
                 "alpha": dec.best_params_["alpha"]}, open(root / "decoder.pkl", "wb"))
    (root / "decoder_receipt.json").write_text(json.dumps({"day0_in_sample_r2_norm_units": float(r2),
                                                           "alpha": float(dec.best_params_["alpha"])}))
    print("decoder", r2, dec.best_params_)


def cmd_align(a):
    import h5py, pickle
    import tensorflow as tf
    _patch_keras_compat()
    _patch_autograph_df()
    from lfads_tf2.subclasses.behavior.models import BehaviorLFADS  # noqa: F401
    from nomad.models import AlignLFADS
    from nomad.tuples import AlignInput
    from nomad.defaults import get_cfg_defaults
    root = BASE / "models" / a.tag
    ref = load_npz(a.stratum, ref_name(a.stratum))
    tgt = load_npz(a.stratum, a.target)
    d = ref["neural"].shape[1]
    xt = pad_units(tgt["neural"].astype(np.float32), d)
    xs0, ys0, _ = chop(ref["neural"].astype(np.float32), ref["y_norm"])
    xsk, ysk, _ = chop(xt, tgt["y_norm"])             # target behaviour is NOT used by NoMAD (unlabeled)
    if a.seed > 0:                                       # seed 0 = the first (unseeded) run of each target
        tf.random.set_seed(a.seed)
        np.random.seed(a.seed)
    sfx = f"_s{a.seed}"
    align_dir = root / "align" / (a.target + sfx)
    if align_dir.exists():
        shutil.rmtree(align_dir)
    cfg_path = root / f"nomad_{a.target}{sfx}.yaml"
    cfg_path.write_text(nomad_yaml(d, root / "day0_model", align_dir, a.max_epochs))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_path))
    _patch_align_call()
    _patch_armvn_log_prob()
    model = AlignLFADS(cfg_node=cfg)
    _fix_graph_step(model, d)
    n0, nk = len(xs0), len(xsk)
    v0, vk = np.arange(n0) % 5 == 4, np.arange(nk) % 5 == 4
    i0, ik = np.arange(n0), np.arange(nk)
    model.load_datasets(AlignInput(
        day0_train_data=xs0[~v0], day0_valid_data=xs0[v0], dayk_train_data=xsk[~vk], dayk_valid_data=xsk[vk],
        day0_train_inds=i0[~v0], day0_valid_inds=i0[v0], dayk_train_inds=ik[~vk], dayk_valid_inds=ik[vk]))
    write_norm(align_dir / "normalization_dayk.h5", xt)
    model.smart_init()
    done, epochs = False, 0
    while not done:
        res = model.train_epoch()
        done = res.get("done", False)
        epochs += 1
        if epochs % 10 == 0:
            print("epoch", epochs, {k: (float(v) if np.isscalar(v) else v) for k, v in res.items() if k != "done"}, flush=True)
    print("aligned after", epochs, "epochs")
    dec = pickle.load(open(root / "decoder.pkl", "rb"))
    g = causal_states(model.lfads_dayk, tgt["neural"].astype(np.float32), d)
    lg = lag_stack(g)
    yhat_norm = lg @ dec["coef"].T + dec["intercept"]
    ym, ys = tgt["y_mean"], tgt["y_std"]
    pred = yhat_norm * ys + ym                          # physical units, NaN for the first lag bins
    (BASE / "preds").mkdir(exist_ok=True)
    np.savez_compressed(BASE / "preds" / f"{a.target}{sfx}.npz", pred=pred, epochs=epochs)
    print("saved preds", a.target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["day0", "decoder", "align"])
    p.add_argument("--stratum", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--target")
    p.add_argument("--max-epochs", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"day0": cmd_day0, "decoder": cmd_decoder, "align": cmd_align}[a.cmd](a)


if __name__ == "__main__":
    main()
