#!/bin/bash
# AP-skeleton K sweep on the M2 forward pools (revision 4.4.3).
# Pretrains one backbone per (pool, K), then evaluates Track A methods on
# every target of the pool.  Source pools are shared across K values only
# through the data cache; each K is its own model.
# usage: run_ap_k_sweep.sh <gpu>
set -u
GPU=$1
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/ap_k_sweep_m2_20261005
mkdir -p "$OUT"

pretrain() {
  local k=$1 key=$2 sources=$3 target=$4
  local dest="$OUT/k${k}/${key}"
  if [ -f "$dest/manifest.json" ]; then echo "[ksweep] skip pretrain k=$k $key"; return 0; fi
  rm -rf "$dest"
  echo "[ksweep] $(date '+%T') GPU$GPU pretrain k=$k $key"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_ksweep_${GPU}_${k}_${key} \
    $PY -m ssm_decode.ap_pretraining --task m2 --target-session "$target" \
    --source-sessions "$sources" \
    --output "$dest" --device cuda:0 --k "$k" --width 128 --layers 2 --state-size 32 \
    --steps 2000 --val-interval 100 > "$OUT/k${k}_${key}.train.log" 2>&1 \
    || { echo "[ksweep] FAILED pretrain k=$k $key"; rm -rf "$dest"; return 1; }
}

evaluate() {
  local k=$1 key=$2 target=$3
  local ckpt="$OUT/k${k}/${key}/best.pt"
  local dest="$OUT/eval/k${k}/${key}/${target}"
  if [ -f "$dest/summary.json" ]; then echo "[ksweep] skip eval k=$k $target"; return 0; fi
  rm -rf "$dest"
  echo "[ksweep] $(date '+%T') GPU$GPU eval k=$k $target"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_ksweep_eval_${GPU} \
    $PY -m ssm_decode.track_a_experiment --task m2 --target-session "$target" \
    --checkpoint "$ckpt" --output "$dest" --device cuda:0 --oracle \
    > "$OUT/eval_k${k}_${key}_${target}.log" 2>&1 \
    || { echo "[ksweep] FAILED eval k=$k $target"; rm -rf "$dest"; return 1; }
}

# Forward pools (must match ssm_decode.forward_eval.forward_plans('m2')).
POOLS='pool1:2020-10-19-Run1,2020-10-19-Run2:2020-10-20-Run1:2020-10-20-Run1,2020-10-20-Run2
pool2:2020-10-19-Run1,2020-10-19-Run2,2020-10-20-Run1,2020-10-20-Run2:2020-10-27-Run1:2020-10-27-Run1,2020-10-27-Run2
pool3:2020-10-19-Run1,2020-10-19-Run2,2020-10-20-Run1,2020-10-20-Run2,2020-10-27-Run1,2020-10-27-Run2:2020-10-28-Run1:2020-10-28-Run1'

while IFS= read -r line; do
  [ -z "$line" ] && continue
  key=$(echo "$line" | cut -d: -f1)
  sources=$(echo "$line" | cut -d: -f2)
  nominal_target=$(echo "$line" | cut -d: -f3)
  targets=$(echo "$line" | cut -d: -f4)
  for k in 16 32 64; do
    pretrain "$k" "$key" "$sources" "$nominal_target" || continue
    IFS=',' read -ra tarr <<< "$targets"
    for t in "${tarr[@]}"; do
      evaluate "$k" "$key" "$t"
    done
  done
done <<< "$POOLS"
echo "[ksweep] GPU$GPU all done $(date '+%T')"
