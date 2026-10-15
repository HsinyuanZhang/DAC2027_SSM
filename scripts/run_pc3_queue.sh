#!/bin/bash
# P3 source-simulation queue (GPU0, 3 parallel jobs). Phase 1: R1/R2 selection, phase 2: R3 selection.
cd /home/xinyuan/Work_host/SSM
source /tmp/penv.sh 2>/dev/null || true
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 PC_OUT=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016 \
  PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_pc3_gpu0
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
LOG=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016/logs
export PY LOG
job() { $PY scripts/run_pc3_refine.py $1 --stratum $2 --set $3 > $LOG/$1_$2_$3.log 2>&1; }
export -f job
printf "%s\n" "select days oos" "select month oos" "select year oos" "select days is" "select month is" "select year is" | xargs -P 3 -L 1 bash -c 'job $0 $1 $2'
printf "%s\n" "select-r3 days oos" "select-r3 month oos" "select-r3 year oos" "select-r3 days is" "select-r3 month is" "select-r3 year is" | xargs -P 3 -L 1 bash -c 'job $0 $1 $2'
