#!/bin/bash
# Labeled-calibration queue driver. Usage: run_lc_queue.sh <gpu> <queue>
# queue: q2d1 | q3d2 | q4sweep | q5mix | q6hand | q7f6 | q8f8 | q9refs | q10f4 | q11seeds
set -u
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
GPU=$1
Q=$2
LC="PYTHONNOUSERSITE=1 PYTHONPATH=$ROOT/.tools/mamba_deps:$ROOT TRITON_CACHE_DIR=/tmp/triton_lc_${GPU}"

run_train() {  # run_train <dataset> <pool> <variant> <tag> <seed> <extra...>
  local ds=$1 pool=$2 variant=$3 tag=$4 seed=$5; shift 5
  local ckpt=$ROOT/results/labeled_calibration_20261007/${ds}_${pool}/${tag}/seed${seed}/best.pt
  if [ ! -f "$ckpt" ]; then
    echo "[lc] $(date '+%T') train $ds/$pool $variant $tag s$seed"
    CUDA_VISIBLE_DEVICES=$GPU env $LC $PY $ROOT/scripts/run_labeled_calibration.py train \
      --dataset "$ds" --pool "$pool" --variant "$variant" --tag "$tag" --seed "$seed" "$@" \
      > "$ROOT/results/labeled_calibration_20261007/${ds}_${pool}.${tag}.s${seed}.train.log" 2>&1 \
      || { echo "[lc] TRAIN FAILED $ds/$pool $tag s$seed"; return 1; }
  fi
  for T in $( [ "$ds" = d1 ] && echo "$POOLTARGETS" || echo "$POOLTARGETS" ); do :; done
}

train_eval() {  # train_eval <ds> <pool> <variant> <tag> <seed> <extra...>
  local ds=$1 pool=$2 variant=$3 tag=$4 seed=$5; shift 5
  local dir=$ROOT/results/labeled_calibration_20261007/${ds}_${pool}/${tag}/seed${seed}
  if [ ! -f "$dir/best.pt" ]; then
    echo "[lc] $(date '+%T') train $ds/$pool $variant/$tag s$seed"
    CUDA_VISIBLE_DEVICES=$GPU env $LC $PY $ROOT/scripts/run_labeled_calibration.py train \
      --dataset "$ds" --pool "$pool" --variant "$variant" --tag "$tag" --seed "$seed" "$@" \
      >> "$ROOT/results/labeled_calibration_20261007/logs.txt" 2>&1 \
      || { echo "[lc] TRAIN FAILED $ds/$pool $tag s$seed"; return 1; }
  fi
  if [ ! -f "$dir/eval_rows.jsonl" ] || [ "$(wc -l < "$dir/eval_rows.jsonl")" -lt 6 ]; then
    echo "[lc] $(date '+%T') eval  $ds/$pool $variant/$tag s$seed"
    CUDA_VISIBLE_DEVICES=$GPU env $LC $PY $ROOT/scripts/run_labeled_calibration.py evaluate \
      --ckpt "$dir/best.pt" --dataset "$ds" --pool "$pool" \
      >> "$ROOT/results/labeled_calibration_20261007/logs.txt" 2>&1 \
      || echo "[lc] EVAL FAILED $ds/$pool $tag s$seed"
  fi
}

mkdir -p $ROOT/results/labeled_calibration_20261007

case $Q in
  q2d1)
    for st in days month year; do
      for seed in 0 1; do
        train_eval d1 $st f7 base $seed --f7-steps 1 --f7-gtype linear --f7-lag 2 --steps 2000
      done
    done ;;
  q3d2)
    for pool in pool1 pool2 pool3; do
      for seed in 0 1; do
        train_eval d2 $pool f7 base $seed --f7-steps 1 --f7-gtype linear --f7-lag 2 --steps 2000
      done
    done ;;
  q4sweep)
    for cfg in "1 linear 0" "1 linear 2" "2 linear 2" "3 linear 2" "1 mlp 0" "1 mlp 2" "2 mlp 2" "3 mlp 2"; do
      set -- $cfg; S=$1; G=$2; L=$3
      for shared in "" "--f7-shared"; do
        tag="sw_S${S}_${G}_L${L}$([ -n "$shared" ] && echo _sh)"
        train_eval d1 days f7 $tag 0 --f7-steps $S --f7-gtype $G --f7-lag $L $shared --steps 2000
      done
    done ;;
  q5mix)
    # best config filled in by caller via BESTCFG env
    for st in month year; do
      for seed in 0 1; do
        train_eval d1 $st f7 $BESTCFG $seed $BESTARGS --steps 2000
      done
    done
    for pool in pool1 pool2 pool3; do
      for seed in 0 1; do
        train_eval d2 $pool f7 $BESTCFG $seed $BESTARGS --steps 2000
      done
    done
    for st in days month year; do
      train_eval d1 $st f7f3 mix 0 --f7-steps 1 --f7-gtype linear --f7-lag 2 --steps 2000
    done ;;
  q6hand)
    for st in days month year; do
      for seed in 0 1; do
        train_eval d1 $st f2 abs $seed --steps 2000
        train_eval d1 $st f3 res $seed --steps 2000
      done
    done
    for pool in pool1 pool2 pool3; do
      for seed in 0 1; do
        train_eval d2 $pool f2 abs $seed --steps 2000
        train_eval d2 $pool f3 res $seed --steps 2000
      done
    done ;;
  q7f6)
    for st in days month year; do
      for seed in 0 1; do
        train_eval d1 $st f6 base $seed --steps 2000
      done
    done
    for pool in pool1 pool2 pool3; do
      for seed in 0 1; do
        train_eval d2 $pool f6 base $seed --steps 2000
      done
    done ;;
  q8f8)
    for st in days month year; do
      for seed in 0 1; do
        train_eval d1 $st f8 base $seed --steps 2000
      done
    done ;;
  q9refs)
    for st in days month year; do
      train_eval d1 $st f1 ref 0 --steps 2000
    done
    for pool in pool1 pool2 pool3; do
      train_eval d2 $pool f1 ref 0 --steps 2000
    done
    for drop in activity crosscov direction t4; do
      train_eval d1 days f5 drop_$drop 0 --drop-groups $drop --steps 2000
    done ;;
  q10f4)
    for st in days month year; do
      train_eval d1 $st f3 f4att 0 --steps 2000   # placeholder: F4 rides on F3 result
    done ;;
  q11seeds)
    for pool in pool1 pool2 pool3; do
      for seed in 0 1 2; do
        train_eval d2 $pool f7 $BESTCFG $seed $BESTARGS --steps 2000
      done
    done ;;
  *) echo "unknown queue $Q"; exit 2 ;;
esac
echo "[lc] $Q done $(date '+%T')"
