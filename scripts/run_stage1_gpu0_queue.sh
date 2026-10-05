#!/bin/bash
# Stage-1' GPU0 queue: recipe ablation (revision 4.4.2), arg-only items.
# 1. M2 small full recipe (reference on the forward protocol)
# 2. M2 small without drift augmentation (gain 0, offset 0, dropout 0)
set -u
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python

run() {  # run <name> <extra flags...>
  local name=$1; shift
  local root="$ROOT/results/forward_eval_m2_${name}_20261005"
  if [ -f "$root/summary.json" ]; then echo "[ablate] skip $name"; return 0; fi
  echo "[ablate] $(date '+%T') start $name"
  PYTHONNOUSERSITE=1 $PY "$ROOT/scripts/run_forward_eval.py" \
    --task m2 --capacity "$name" --width 128 --layers 2 --state-size 32 \
    --methods none,lora --seeds 0,1,2 --none-seeds 0 --lrs 0.0003,0.001 \
    --gpu 0 --root "$root" "$@" \
    > "$ROOT/results/forward_eval_m2_${name}_20261005.driver.log" 2>&1 \
    && echo "[ablate] $name done" || echo "[ablate] $name FAILED"
}

run small_ref --pretrain-extra --ffn-mult 2
run small_noaugment --pretrain-extra --ffn-mult 2 --gain-sigma 0 --offset-sigma 0 --channel-dropout 0
echo "[ablate] GPU0 queue done $(date '+%T')"
