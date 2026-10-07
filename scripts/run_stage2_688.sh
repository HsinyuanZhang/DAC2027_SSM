#!/bin/bash
# Stage 2' single-pool pretraining + eval.  Iteration-3 review fixes:
# - the source gate compares against the AP backbone on the same pool
#   (an external reference), never against the evaluated checkpoint itself;
# - the extension loop checks the actually-trained steps, keys directories
#   by hyperparameters, and reports non-convergence honestly;
# - the AP warm-start path resolves to the newest ap_pool attempt.
set -u
GPU=$1
FAMILY=$2            # ap | set
ROOT=/home/xinyuan/Work_host/SSM
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
OUT=$ROOT/results/stage2_688_20261005
SOURCES="sub-M_ses-CO-20150511,sub-M_ses-CO-20150512,sub-M_ses-CO-20150610,sub-M_ses-CO-20150611,sub-M_ses-CO-20150612,sub-M_ses-CO-20150615"
TARGETS="sub-M_ses-CO-20150616 sub-M_ses-CO-20150617"
mkdir -p "$OUT"

AP_DIR=$(ls -dt "$OUT"/ap_pool*/ 2>/dev/null | head -1)
AP_CKPT="$AP_DIR/best.pt"
AP_VAL=$($PY -c "import json; print(round(json.load(open('$AP_DIR/manifest.json'))['best_source_val_trunk_r2'], 4))")
GATE=$($PY -c "print(round($AP_VAL - 0.05, 4))")
echo "[s2] AP reference $AP_DIR source-val=$AP_VAL -> gate $GATE"

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
    $( [ "$FAMILY" = set ] && echo --init-from-ap "$AP_CKPT" --episode-trials 33 --distill-steps 8000 --read-in mlp --episodic-lr 0.0003 ) \
    > "$OUT/${FAMILY}.train.log" 2>&1 \
    || { echo "[s2] FAILED pretrain $FAMILY"; rm -rf "$dest"; return 1; }
}

STEPS=2000
FP="k10mlp2"
BASE=$OUT/${FAMILY}_pool_${FP}
CONVERGED=0
for attempt in 0 1 2 3; do
  if [ "$attempt" -eq 0 ]; then DIR=$BASE; else DIR=${BASE}_x$((2 ** attempt)); fi
  [ -f "$DIR/manifest.json" ] || pretrain "$DIR" "$STEPS" || exit 1
  BEST=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['best_step'])")
  FINAL=$($PY -c "import json; print(json.load(open('$DIR/manifest.json'))['args']['steps'])")
  if [ "$FINAL" -ne "$STEPS" ]; then
    echo "[s2] $FAMILY dir $DIR trained $FINAL != intended $STEPS (params changed?); retraining"
    STEPS=$((STEPS * 2)); continue
  fi
  if [ "$BEST" -gt $((STEPS * 8 / 10)) ] && [ "$attempt" -lt 3 ]; then
    echo "[s2] $FAMILY best_step=$BEST in last 20% of $STEPS; retraining at $((STEPS * 2))"
    STEPS=$((STEPS * 2)); continue
  fi
  if [ "$BEST" -gt $((STEPS * 8 / 10)) ]; then
    echo "[s2] $FAMILY NOT CONVERGED after extension cap (best_step=$BEST of $STEPS)"
  else
    echo "[s2] $FAMILY converged (best_step=$BEST, trained steps=$FINAL)"
    CONVERGED=1
  fi
  break
done
CKPT=$DIR/best.pt

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
      --min-source-r2 "$GATE" \
      $( [ "$FAMILY" = ap ] && echo --oracle --pool-sources "$SOURCES" ) \
      $( [ "$FAMILY" = set ] && echo --prior-correct ) \
      > "$OUT/eval_${FAMILY}.${T}.log" 2>&1 \
      || { echo "[s2] EVAL FAILED $FAMILY $T"; rm -rf "$DEST"; }
  fi
done
echo "[s2] $FAMILY done (converged=$CONVERGED) $(date '+%T')"
