#!/bin/bash
# Stage-2 refinement around the best stage-1 config (source validation only).
export CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_hy_gpu1_g2_$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
cd /home/xinyuan/Work_host/SSM
OUT=results/hyser_20261015/grid2; mkdir -p $OUT
shift_args() { :; }
run() { NAME=$1; shift; [ -f $OUT/$NAME/summary.json ] || $PY scripts/run_hyser_train.py --out $OUT/$NAME --max-steps 10000 --patience 20 "$@" > $OUT/$NAME.log 2>&1; }
case $1 in
 0) run k16_d0.0 --k 16 --width 128 --layers 2 --state-size 32 --dropout 0.0
    run k16_d0.05 --k 16 --width 128 --layers 2 --state-size 32 --dropout 0.05 ;;
 1) run k16_d0.1_wd0.1 --k 16 --width 128 --layers 2 --state-size 32 --dropout 0.1 --weight-decay 0.1
    run k32_d0.1_w128 --k 32 --width 128 --layers 2 --state-size 32 --dropout 0.1 ;;
 2) run k16_d0.1_lr5e-4 --k 16 --width 128 --layers 2 --state-size 32 --dropout 0.1 --lr 5e-4
    run k16_d0.1_cd0.2 --k 16 --width 128 --layers 2 --state-size 32 --dropout 0.1 --channel-dropout 0.2 ;;
esac
