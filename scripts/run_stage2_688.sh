#!/bin/bash
# Stage 2': single 000688 sub-M pretraining per family, then dev-target evals.
# Budget rule: if best_step lands in the last 20% of the budget, retrain at
# double steps, at most twice (revision item 16).
set -u
GPU=$1
FAMILY=$2            # ap | set
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/stage2_688_20261005
SOURCES="sub-M_ses-CO-20150511,sub-M_ses-CO-20150512,sub-M_ses-CO-20150610,sub-M_ses-CO-20150611,sub-M_ses-CO-20150612,sub-M_ses-CO-20150615"
TARGETS="sub-M_ses-CO-20150616 sub-M_ses-CO-20150617"
mkdir -p "$OUT"

pretrain() {  # pretrain <dest> <steps> <extra...>
  local dest=$1; shift
  local steps=$1; shift
  rm -rf "$dest"
  echo "[s2] $(date '+%T') GPU$GPU pretrain $FAMILY steps=$steps"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
    TRITON_CACHE_DIR=/tmp/triton_s2_${FAMILY}_${GPU} \
    $PY -m ssm_decode.${FAMILY}_pretraining \
    $( [ "$FAMILY" = ap ] && echo --task dandi688sm --target-session sub-M_ses-CO-20150616 ) \
    --dandi688-sources "$SOURCES" \
    --output "$dest" --device cuda:0 --k 10 --width 128 --layers 2 --state-size 32 \
    --steps "$steps" "$@" \
    $( [ "$FAMILY" = set ] && echo --init-from-ap $ROOT/results/stage2_688_20261005/ap_pool_x8/best.pt --episode-trials 33 --distill-steps 8000 ) \
    > "$OUT/${FAMILY}.train.log" 2>&1 \
    || { echo "[s2] FAILED pretrain $FAMILY"; rm -rf "$dest"; return 1; }
}

STEPS=2000
BASE=$OUT/${FAMILY}_pool
for attempt in 0 1 2 3; do
  if [ "$attempt" -eq 0 ]; then DIR=$BASE; else DIR=${BASE}_x$((2 ** attempt)); fi
  [ -f "$DIR/manifest.json" ] || pretrain "$DIR" "$STEPS" || exit 1
  BEST=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['best_step'])")
  FINAL=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['args']['steps'])")
  if [ "$BEST" -gt $((STEPS * 8 / 10)) ] && [ "$attempt" -lt 3 ]; then
    echo "[s2] $FAMILY best_step=$BEST in last 20% of $STEPS; retraining at $((STEPS * 2)) in a fresh dir"
    STEPS=$((STEPS * 2)); continue
  fi
  echo "[s2] $FAMILY converged (best_step=$BEST, trained steps=$FINAL)"
  break
done
CKPT=$DIR/best.pt
MIN_SRC=$($PY -c "import json; print(round(json.load(open('$DIR/manifest.json'))['best_source_val_trunk_r2'] - 0.05, 4))")

for T in $TARGETS; do
  DEST=$OUT/eval_${FAMILY}/$T
  if [ ! -f "$DEST/summary.json" ]; then
    rm -rf "$DEST"
    echo "[s2] $(date '+%T') GPU$GPU eval $FAMILY $T"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
      PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT" \
      TRITON_CACHE_DIR=/tmp/triton_s2_eval_${GPU} \
      $PY -m ssm_decode.$([ "$FAMILY" = ap ] && echo track_a_experiment || echo set_attention_eval) \
      --task dandi688sm --target-session "$T" --checkpoint "$CKPT" \
      --output "$DEST" --device cuda:0 \
      --min-source-r2 "$MIN_SRC" \
      $( [ "$FAMILY" = ap ] && echo --oracle --pool-sources "$SOURCES" ) \
      $( [ "$FAMILY" = set ] && echo --prior-correct ) \
      > "$OUT/eval_${FAMILY}.${T}.log" 2>&1 \
      || { echo "[s2] EVAL FAILED $FAMILY $T"; rm -rf "$DEST"; }
  fi
done
echo "[s2] $FAMILY done $(date '+%T')"
