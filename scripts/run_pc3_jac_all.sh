#!/bin/bash
cd /home/xinyuan/Work_host/SSM
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 PC_OUT=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016 \
  PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_pc3_gpu0
for s in month year; do
  /home/xinyuan/miniconda3/envs/spint/bin/python scripts/run_pc3_refine.py jacobian --stratum $s > results/population_calib_p3_20261016/logs/jac_$s.log 2>&1
done
