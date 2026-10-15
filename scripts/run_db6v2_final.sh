#!/bin/bash
# v2 final backbone, second seed (seed 0 = the stage-1 selection run copied to final/seed0).
# Config chosen on contiguous, order-shuffled source validation: K=8 w128 l2 s32 d0.1 tau160 ms, windows 512-1024, std floor (train) 1e-3.
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6v2_gpu1_final$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_v2_20261016/final/seed$1
mkdir -p $OUT
if [ ! -f $OUT/summary.json ]; then
  $PY scripts/run_db6_train.py --out $OUT --k 8 --width 128 --layers 2 --state-size 32 --dropout 0.1 --tau-ms 160 --max-steps 15000 --seed $1 > results/db6_v2_20261016/final/seed$1.log 2>&1
fi
