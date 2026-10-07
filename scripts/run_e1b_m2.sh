#!/bin/bash
# E1b on the M2 forward pools: warm start + distillation from the K-sweep
# k32 AP checkpoints, selection on episode validation (deployment metric;
# source val logged for the gate record), extension rule on best_step.
# usage: run_e1b_m2.sh <gpu> <pool1|pool2|pool3>
set -u
GPU=$1
WHICH=$2
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
KS=$ROOT/results/ap_k_sweep_m2_20261005/k32
OUT=$ROOT/results/e1b_m2_20261007
mkdir -p "$OUT"

P1="2020-10-19-Run1,2020-10-19-Run2"
P2="$P1,2020-10-20-Run1,2020-10-20-Run2"
P3="$P2,2020-10-27-Run1,2020-10-27-Run2"
case $WHICH in
  pool1) SOURCES="$P1"; TARGETS="2020-10-20-Run1 2020-10-20-Run2" ;;
  pool2) SOURCES="$P2"; TARGETS="2020-10-27-Run1 2020-10-27-Run2" ;;
  pool3) SOURCES="$P3"; TARGETS="2020-10-28-Run1" ;;
  *) echo "unknown pool $WHICH"; exit 2 ;;
esac
AP_CKPT=$KS/$WHICH/best.pt
AP_VAL=$($PY -c "import json; print(round(json.load(open('$KS/$WHICH/manifest.json'))['best_source_val_trunk_r2'], 4))")
GATE=$($PY -c "print(round($AP_VAL - 0.05, 4))")
echo "[e1m2] $WHICH AP source-val=$AP_VAL gate=$GATE"

pretrain() {
  local dest=$1 steps=$2
  rm -rf "$dest"
  echo "[e1m2] $(date '+%T') GPU$GPU pretrain $WHICH steps=$steps"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_e1m2_${WHICH}_${GPU} \
    $PY -m ssm_decode.set_pretraining --task m2 --source-sessions "$SOURCES" \
    --init-from-ap "$AP_CKPT" --read-in mlp --k 32 \
    --distill-steps 8000 --episode-trials 33 --episodic-lr 0.0003 \
    --select-on episode --width 128 --layers 2 --state-size 32 \
    --output "$dest" --device cuda:0 --steps "$steps" \
    > "$OUT/${WHICH}.train.log" 2>&1 \
    || { echo "[e1m2] FAILED pretrain $WHICH"; rm -rf "$dest"; return 1; }
}

STEPS=4000
BASE=$OUT/${WHICH}_set
for attempt in 0 1 2; do
  if [ "$attempt" -eq 0 ]; then DIR=$BASE; else DIR=${BASE}_x$((2 ** attempt)); fi
  [ -f "$DIR/manifest.json" ] || pretrain "$DIR" "$STEPS" || exit 1
  BEST=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['best_step'])")
  FINAL=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['args']['steps'])")
  [ "$FINAL" -ne "$STEPS" ] && { STEPS=$((STEPS * 2)); continue; }
  if [ "$BEST" -gt $((STEPS * 8 / 10)) ] && [ "$attempt" -lt 2 ]; then
    echo "[e1m2] $WHICH best=$BEST in last 20% of $STEPS; doubling"; STEPS=$((STEPS * 2)); continue
  fi
  [ "$BEST" -gt $((STEPS * 8 / 10)) ] && echo "[e1m2] $WHICH NOT CONVERGED (best=$BEST/$STEPS)" \
                                     || echo "[e1m2] $WHICH converged (best=$BEST/$STEPS)"
  break
done
CKPT=$DIR/best.pt

for T in $TARGETS; do
  DEST=$OUT/eval/$WHICH/$T
  [ -f "$DEST/summary.json" ] && { echo "[e1m2] skip eval $T"; continue; }
  rm -rf "$DEST"
  echo "[e1m2] $(date '+%T') GPU$GPU eval $WHICH $T"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_e1m2_eval_${GPU} \
    $PY -m ssm_decode.set_attention_eval --task m2 --target-session "$T" \
    --checkpoint "$CKPT" --output "$DEST" --device cuda:0 \
    --min-source-r2 "$GATE" \
    > "$OUT/eval_${WHICH}.${T}.log" 2>&1 \
    || { echo "[e1m2] EVAL FAILED $T"; rm -rf "$DEST"; }
done
echo "[e1m2] $WHICH done $(date '+%T')"
