#!/bin/bash
# Source-validation-only grid for the DB6 backbone (GPU1). usage: run_db6_grid.sh <lane 0..2>
# stage 1: K x model size x dropout at tau 80 ms (seed 0).
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_20261015/grid
mkdir -p $OUT
LANE=$1
i=0
for K in 4 8 14; do for SZ in "64 1 32" "128 2 32" "256 2 64"; do for DR in 0.1 0.3; do
  if [ $((i % 3)) -eq $LANE ]; then
    set -- $SZ; W=$1; L=$2; S=$3
    NAME=k${K}_w${W}_l${L}_s${S}_d${DR}_tau80
    if [ ! -f $OUT/$NAME/summary.json ]; then
      $PY scripts/run_db6_train.py --out $OUT/$NAME --k $K --width $W --layers $L --state-size $S --dropout $DR --tau-ms 80 --max-steps 6000 --patience 12 --seed 0 > $OUT/$NAME.log 2>&1
    fi
  fi
  i=$((i+1))
done; done; done
