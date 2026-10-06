#!/bin/bash
# Review iteration-3 step 3: ui_prior across time-gap strata on 000688 sub-M.
#   days  : 2015 train pool -> dev targets (existing ap_pool_k10 covers b=0 seeds)
#   month : 2014-02/03 pool (6 sessions) -> 2014-06-26 target (~3.7 months)
#   year  : 2014-02/03 pool              -> 2015-05-11 target (~15 months)
# AP backbone per pool with the budget-extension rule; Track A eval with
# three seeds (0,1,2) per target.
set -u
GPU=$1
STRATUM=$2
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/stage2_intervals_20261006
mkdir -p "$OUT"

EARLY="sub-M_ses-CO-20140203,sub-M_ses-CO-20140217,sub-M_ses-CO-20140218,sub-M_ses-CO-20140303,sub-M_ses-CO-20140304,sub-M_ses-CO-20140306"

case $STRATUM in
  month) SOURCES="$EARLY"; TARGET=sub-M_ses-CO-20140626 ;;
  year)  SOURCES="$EARLY"; TARGET=sub-M_ses-CO-20150511 ;;
  days)  SOURCES="sub-M_ses-CO-20150511,sub-M_ses-CO-20150512,sub-M_ses-CO-20150610,sub-M_ses-CO-20150611,sub-M_ses-CO-20150612,sub-M_ses-CO-20150615"; TARGET=sub-M_ses-CO-20150616 ;;
  *) echo "unknown stratum $STRATUM"; exit 2 ;;
esac

pretrain() {
  local dest=$1 steps=$2
  rm -rf "$dest"
  echo "[iv] $(date '+%T') GPU$GPU pretrain $STRATUM steps=$steps"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_iv_${STRATUM}_${GPU} \
    $PY -m ssm_decode.ap_pretraining --task dandi688sm --target-session "$TARGET" \
    --dandi688-sources "$SOURCES" --output "$dest" --device cuda:0 \
    --k 10 --width 128 --layers 2 --state-size 32 --steps "$steps" \
    > "$OUT/${STRATUM}.train.log" 2>&1 \
    || { echo "[iv] FAILED pretrain $STRATUM"; rm -rf "$dest"; return 1; }
}

STEPS=2000
BASE=$OUT/${STRATUM}_pool_k10
for attempt in 0 1 2 3; do
  if [ "$attempt" -eq 0 ]; then DIR=$BASE; else DIR=${BASE}_x$((2 ** attempt)); fi
  [ -f "$DIR/manifest.json" ] || pretrain "$DIR" "$STEPS" || exit 1
  BEST=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['best_step'])")
  FINAL=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['args']['steps'])")
  if [ "$FINAL" -ne "$STEPS" ]; then STEPS=$((STEPS * 2)); continue; fi
  if [ "$BEST" -gt $((STEPS * 8 / 10)) ] && [ "$attempt" -lt 3 ]; then
    echo "[iv] $STRATUM best=$BEST in last 20% of $STEPS; doubling"
    STEPS=$((STEPS * 2)); continue
  fi
  [ "$BEST" -gt $((STEPS * 8 / 10)) ] && echo "[iv] $STRATUM NOT CONVERGED (best=$BEST/$STEPS)" \
                                     || echo "[iv] $STRATUM converged (best=$BEST/$STEPS)"
  break
done
CKPT=$DIR/best.pt

for SEED in 0 1 2; do
  DEST=$OUT/eval_${STRATUM}/seed${SEED}
  [ -f "$DEST/summary.json" ] && { echo "[iv] skip eval $STRATUM s$SEED"; continue; }
  rm -rf "$DEST"
  echo "[iv] $(date '+%T') GPU$GPU eval $STRATUM seed $SEED"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_iv_eval_${GPU} \
    $PY -m ssm_decode.track_a_experiment --task dandi688sm --target-session "$TARGET" \
    --pool-sources "$SOURCES" --checkpoint "$CKPT" --output "$DEST" \
    --device cuda:0 --seed "$SEED" \
    > "$OUT/eval_${STRATUM}.s${SEED}.log" 2>&1 \
    || { echo "[iv] EVAL FAILED $STRATUM s$SEED"; rm -rf "$DEST"; }
done
echo "[iv] $STRATUM done $(date '+%T')"
