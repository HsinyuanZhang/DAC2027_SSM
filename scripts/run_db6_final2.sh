#!/bin/bash
# Stage 3b (source validation only): tau 320 ms, two seeds, same config as the tau-160 runs. usage: run_db6_final2.sh <seed>
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_20261015/grid3
NAME=tau320_seed$1
if [ ! -f $OUT/$NAME/summary.json ]; then
  $PY scripts/run_db6_train.py --out $OUT/$NAME --k 8 --width 128 --layers 2 --state-size 32 --dropout 0.1 --max-steps 15000 --patience 15 --tau-ms 320 --seed $1 > $OUT/$NAME.log 2>&1
fi
