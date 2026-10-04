#!/bin/bash
# Evaluate Track A methods on every trained rotation, one GPU.
# usage: run_track_a_queue.sh <gpu>
set -u
GPU=$1
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
PRE=$ROOT/results/ap_pretraining_m2_20261004
OUT=$ROOT/results/track_a_m2_20261004
mkdir -p "$OUT"
for ckpt in "$PRE"/*/best.pt; do
  target=$(basename "$(dirname "$ckpt")")
  if [ -f "$OUT/$target/summary.json" ]; then echo "[tracka] skip done $target"; continue; fi
  echo "[tracka] $(date '+%T') GPU$GPU eval $target"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_ta_$GPU \
    $PY -m ssm_decode.track_a_experiment --task m2 --target-session "$target" \
    --checkpoint "$ckpt" --output "$OUT/$target" --device cuda:0 --oracle \
    > "$OUT/${target}.eval.log" 2>&1 || echo "[tracka] FAILED $target (see $OUT/${target}.eval.log)"
done
echo "[tracka] GPU$GPU all done $(date '+%T')"
