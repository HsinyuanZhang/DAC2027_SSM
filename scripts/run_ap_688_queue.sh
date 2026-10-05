#!/bin/bash
# Pretrain AP backbones for DANDI 000688 sub-M dev targets on one GPU.
set -u
GPU=$1
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/ap_pretraining_dandi688sm_20261004
mkdir -p "$OUT"
for target in sub-M_ses-CO-20150616 sub-M_ses-CO-20150617; do
  dest="$OUT/$target"
  if [ -f "$dest/manifest.json" ]; then echo "[q688] skip done $target"; continue; fi
  echo "[q688] $(date '+%T') GPU$GPU train $target"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_ap688_$GPU \
    $PY -m ssm_decode.ap_pretraining --task dandi688sm --target-session "$target" \
    --output "$dest" --device cuda:0 --k 12 --width 128 --layers 2 --state-size 32 \
    --steps 2000 --val-interval 100 > "$OUT/${target}.train.log" 2>&1 \
    || { echo "[q688] FAILED $target"; rm -rf "$dest"; }
done
echo "[q688] GPU$GPU all done $(date '+%T')"
