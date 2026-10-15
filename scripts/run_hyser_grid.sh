#!/bin/bash
# Source-validation-only grid for the Hyser backbone (GPU1). usage: run_hyser_grid.sh <lane 0..2>
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_hy_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/hyser_20261015/grid
mkdir -p $OUT
i=0
for K in 8 16 32; do for SZ in "64 1 32" "128 2 32" "256 2 64"; do for DR in 0.1 0.3; do
  if [ $((i % 3)) -eq $1 ]; then
    set -- $1 $SZ; W=$2; L=$3; S=$4
    NAME=k${K}_w${W}_l${L}_s${S}_d${DR}
    if [ ! -f $OUT/$NAME/summary.json ]; then
      $PY scripts/run_hyser_train.py --out $OUT/$NAME --k $K --width $W --layers $L --state-size $S --dropout $DR --max-steps 6000 --patience 16 --seed 0 > $OUT/$NAME.log 2>&1
    fi
    set -- $1
  fi
  i=$((i+1))
done; done; done
