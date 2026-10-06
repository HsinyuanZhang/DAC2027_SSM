#!/bin/bash
# Review step 4a: additional source seeds for the in-budget capacity
# configurations (wide_n16 seeds 1 and 2, then deep_n16 seed 1).
set -u
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python

run() {
  local name=$1 w=$2 l=$3 n=$4 seed=$5
  local root="$ROOT/results/forward_eval_m2_${name}_s${seed}_20261006"
  if [ -f "$root/summary.json" ]; then echo "[seed] skip $name s$seed"; return 0; fi
  echo "[seed] $(date '+%T') start $name seed $seed"
  PYTHONNOUSERSITE=1 $PY "$ROOT/scripts/run_forward_eval.py" \
    --task m2 --capacity "$name" --width "$w" --layers "$l" --state-size "$n" \
    --methods none,lora --seeds 0,1,2 --none-seeds 0 --lrs 0.0003,0.001 \
    --gpu 0 --root "$root" \
    --pretrain-extra=--seed --pretrain-extra="$seed" \
    > "$ROOT/results/forward_eval_m2_${name}_s${seed}_20261006.driver.log" 2>&1 \
    && echo "[seed] $name s$seed done" || echo "[seed] $name s$seed FAILED"
}

run wide_n16 256 4 16 1
run wide_n16 256 4 16 2
run deep_n16 128 6 16 1
echo "[seed] queue done $(date '+%T')"
