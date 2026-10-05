#!/bin/bash
# Queue AP-backbone pretraining rotations for M2 on one GPU.
# usage: run_ap_rotation_queue.sh <gpu> <target-session>...
set -u
GPU=$1; shift
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/ap_pretraining_m2_20261004
mkdir -p "$OUT"
for target in "$@"; do
  dest="$OUT/$target"
  if [ -f "$dest/manifest.json" ]; then echo "[queue] skip done $target"; continue; fi
  echo "[queue] $(date '+%T') GPU$GPU train $target"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_ap_$GPU \
    $PY -m ssm_decode.ap_pretraining --task m2 --target-session "$target" \
    --output "$dest" --device cuda:0 --k 16 --width 128 --layers 2 --state-size 32 \
    --steps 2000 --val-interval 100 > "$OUT/${target}.train.log" 2>&1 \
    || { echo "[queue] FAILED $target (see $OUT/${target}.train.log)"; rm -rf "$dest"; }
done
echo "[queue] GPU$GPU all done $(date '+%T')"
