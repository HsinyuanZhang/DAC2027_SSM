#!/bin/bash
# Stage 2 (source validation only, seed 0): longer budget (stage 1 was still improving at 6000 steps),
# then tau at the best stage-1 configs.  usage: run_db6_grid2.sh <lane 0..2>
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_20261015/grid2
mkdir -p $OUT
LANE=$1
# name | K W L S dropout tau
CONFIGS=(
"k8_w128_l2_s32_d0.1_tau80_long|8 128 2 32 0.1 80"
"k8_w256_l2_s64_d0.1_tau80_long|8 256 2 64 0.1 80"
"k14_w128_l2_s32_d0.3_tau80_long|14 128 2 32 0.3 80"
"k8_w128_l2_s32_d0.1_tau40_long|8 128 2 32 0.1 40"
"k8_w128_l2_s32_d0.1_tau160_long|8 128 2 32 0.1 160"
"k8_w128_l2_s32_d0.1_tau20_long|8 128 2 32 0.1 20"
)
i=0
for C in "${CONFIGS[@]}"; do
  if [ $((i % 3)) -eq $LANE ]; then
    NAME=${C%%|*}; set -- ${C##*|}
    if [ ! -f $OUT/$NAME/summary.json ]; then
      $PY scripts/run_db6_train.py --out $OUT/$NAME --k $1 --width $2 --layers $3 --state-size $4 --dropout $5 --tau-ms $6 --max-steps 15000 --patience 15 --seed 0 > $OUT/$NAME.log 2>&1
    fi
  fi
  i=$((i+1))
done
