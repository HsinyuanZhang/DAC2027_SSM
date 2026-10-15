#!/bin/bash
# v2 backbone selection (source validation only, contiguous decoding of held-out source repetitions, seed 0).
# Small search around the v1 winner (K=8 w128 l2 s32 d0.1 tau160); trained on trial-order-shuffled continuous streams, early stopping on shuffled-order validation.
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6v2_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_v2_20261016/select
mkdir -p $OUT
# name | args
if [ "$1" = "0" ]; then CONFIGS=(
"k8_w128_l2_s32_tau160|--k 8 --width 128 --state-size 32 --tau-ms 160"
"k8_w256_l2_s64_tau160|--k 8 --width 256 --state-size 64 --tau-ms 160"
"k8_w128_l2_s32_tau80|--k 8 --width 128 --state-size 32 --tau-ms 80"
); else CONFIGS=(
"k14_w128_l2_s32_tau160|--k 14 --width 128 --state-size 32 --tau-ms 160"
"k8_w128_l2_s32_tau320|--k 8 --width 128 --state-size 32 --tau-ms 320"
"k8_w128_l2_s32_tau160_win1024-2048|--k 8 --width 128 --state-size 32 --tau-ms 160 --win-min 1024 --win-max 2048"
); fi
for C in "${CONFIGS[@]}"; do
  NAME=${C%%|*}; ARGS=${C##*|}
  if [ ! -f $OUT/$NAME/summary.json ]; then
    $PY scripts/run_db6_train.py --out $OUT/$NAME --layers 2 --dropout 0.1 --max-steps 15000 --seed 0 $ARGS > $OUT/$NAME.log 2>&1
  fi
done
