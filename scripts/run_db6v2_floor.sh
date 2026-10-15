#!/bin/bash
# v2 std-floor candidates (source validation only): winner architecture trained with relative std floor F.
# usage: run_db6v2_floor.sh <floor>
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6v2_gpu1_floor$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_v2_20261016/floor
mkdir -p $OUT
NAME=k8_w128_l2_s32_tau160_floor$1
if [ ! -f $OUT/$NAME/summary.json ]; then
  $PY scripts/run_db6_train.py --out $OUT/$NAME --k 8 --width 128 --layers 2 --state-size 32 --dropout 0.1 --tau-ms 160 --max-steps 15000 --seed 0 --std-floor $1 > $OUT/$NAME.log 2>&1
fi
