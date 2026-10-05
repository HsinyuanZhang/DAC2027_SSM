#!/bin/bash
# C10 variant queue (revision 4.4.1): waits for the M2 wide baseline driver,
# then runs state-budget-fitting variants on GPU 1, one capacity at a time.
# Variants (INT24 recurrent state, must fit 128 KB):
#   wide_n16  256x4 N16 = 103.9 KB   capacity in width
#   deep_n16  128x6 N16 =  77.9 KB   capacity in layers
#   wide_n8   256x4 N8  =  54.9 KB   minimal state
set -u
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
GPU=1
BASE_PID=$1

# Wait for the wide baseline driver to release the GPU.
while kill -0 "$BASE_PID" 2>/dev/null; do sleep 60; done
echo "[c10] baseline driver $BASE_PID exited at $(date '+%T')"

run_variant() {  # run_variant <name> <width> <layers> <state>
  local name=$1 w=$2 l=$3 n=$4
  local root="$ROOT/results/forward_eval_m2_${name}_20261005"
  if [ -f "$root/summary.json" ]; then echo "[c10] skip $name"; return 0; fi
  echo "[c10] $(date '+%T') start $name (${w}x${l} N${n})"
  PYTHONNOUSERSITE=1 $PY "$ROOT/scripts/run_forward_eval.py" \
    --task m2 --capacity "$name" --width "$w" --layers "$l" --state-size "$n" \
    --methods none,lora --seeds 0,1,2 --none-seeds 0 --lrs 0.0003,0.001 \
    --gpu $GPU --root "$root" > "$ROOT/results/forward_eval_m2_${name}_20261005.driver.log" 2>&1 \
    && echo "[c10] $name done" || echo "[c10] $name FAILED (see driver log)"
}

run_variant wide_n16 256 4 16
run_variant deep_n16 128 6 16
run_variant wide_n8  256 4 8
echo "[c10] all variants done $(date '+%T')"
