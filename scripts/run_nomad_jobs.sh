#!/bin/bash
# Usage: run_nomad_jobs.sh <jobs-file> <parallel>   (each line: "<stratum> <tag> <input> <init> <target> <seed> [extra flags]")
# Runs scripts/run_baselines_nomad_tf.py align on CPU (TF 2.6 env) for every line, logs to results/baselines_20261016/logs/nomad2_*.log
JOBS=$1; P=${2:-3}
export CUDA_VISIBLE_DEVICES="" NOMAD_THREADS=${NOMAD_THREADS:-6}
ROOT=/home/xinyuan/Work_host/SSM
run() {
  read -r st tag inp init tgt seed extra <<< "$1"
  name="nomad2_${tag}_${inp}_i${init}_${tgt}_s${seed}${extra// /}"
  cd /tmp && /mnt/data/work_host/envs/nomad_tf26/bin/python $ROOT/scripts/run_baselines_nomad_tf.py align --stratum $st --tag $tag \
     --input $inp --init-readin $init --target $tgt --seed $seed $extra > $ROOT/results/baselines_20261016/logs/$name.log 2>&1
}
export -f run; export ROOT
grep -v '^#' "$JOBS" | xargs -d '\n' -P "$P" -I{} bash -c 'run "{}"'
