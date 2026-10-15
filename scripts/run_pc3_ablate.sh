#!/bin/bash
cd /home/xinyuan/Work_host/SSM
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 PC_OUT=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016 \
  PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_pc3_gpu0
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
L=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016/logs
printf "%s\n" "days is" "month oos" "year oos" | xargs -P 3 -L 1 bash -c "$PY scripts/run_pc3_refine.py ablate --stratum \$0 --set \$1 > $L/ablate_\$0_\$1.log 2>&1"
