#!/bin/bash
# Stage 3 (source validation only): second seed for tau 80/160 at the stage-2 config, and two Hyser-style
# perturbations (weight decay 0.1, channel dropout 0.2) at tau 160.  usage: run_db6_final.sh <lane 0..3>
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_db6_gpu1_lane$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/db6_20261015/grid3
mkdir -p $OUT
COMMON="--k 8 --width 128 --layers 2 --state-size 32 --dropout 0.1 --max-steps 15000 --patience 15"
case $1 in
 0) NAME=tau80_seed1;   ARGS="--tau-ms 80 --seed 1";;
 1) NAME=tau160_seed1;  ARGS="--tau-ms 160 --seed 1";;
 2) NAME=tau160_wd0.1_seed0; ARGS="--tau-ms 160 --seed 0 --weight-decay 0.1";;
 3) NAME=tau160_cd0.2_seed0; ARGS="--tau-ms 160 --seed 0 --channel-dropout 0.2";;
esac
if [ ! -f $OUT/$NAME/summary.json ]; then
  $PY scripts/run_db6_train.py --out $OUT/$NAME $COMMON $ARGS > $OUT/$NAME.log 2>&1
fi
