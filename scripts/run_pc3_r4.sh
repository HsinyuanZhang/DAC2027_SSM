#!/bin/bash
# R4: identity-free (globalmean prior) + label-free P0, train + evaluate, GPU0 only.
cd /home/xinyuan/Work_host/SSM
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 PC_OUT=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016 \
  PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_pc3_gpu0 OMP_NUM_THREADS=4
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
run() { # stratum seed
  common="--stratum $1 --pool O --variant p0 --seed $2 --label-free --prior-mode globalmean --tag p0_nolab_na_gm"
  $PY scripts/run_population_calib.py train $common && $PY scripts/run_population_calib.py evaluate $common
}
export -f run; export PY
printf "%s\n" "days 0" "days 1" "month 0" "month 1" "year 0" "year 1" | xargs -P 3 -L 1 bash -c 'run $0 $1 > /home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016/logs/r4_$0_s$1.log 2>&1'
