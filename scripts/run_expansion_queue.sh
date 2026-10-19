#!/bin/bash
# Resumable expansion queue: AP backbones (budget-extension rule) -> P0 calibrators (2 seeds) ->
# R1-stream source selection -> evaluation rows (P-prior, P0, P0+R1-stream; B in {1,4,8,16,33}, 8 draws).
# One chain per backbone id (backbone, p0, select, eval in order); up to --jobs chains run concurrently.
# Every stage is idempotent, so re-running the same command resumes after a crash, a deadline or a lost
# Colab session.  Paths are identical on the host and on Colab; only --out / --work / PY differ.
#
#   host   : scripts/run_expansion_queue.sh --jobs 1
#   Colab  : PY=python scripts/run_expansion_queue.sh --out /content/drive/MyDrive/DAC27/colab/expansion_out \
#                --work /content/expansion_work --jobs 4 --deadline-hours 10
#
# Options: --out DIR (persistent root, default results/expansion_20261016)  --work DIR (local scratch)
#          --bids a,b|all  --stages backbone,p0,jacobian,select,eval  --jobs N  --tier primary|all
#          --include-2016-optional (required by --tier all; logs a warning)
#          --exclude-apst-dev  --device cuda:0  --deadline-hours H  --dry-run
#          Round 2 adds stages causal-source (inside each chain) and causal-eval (a second pass, after every
#          chain's causal-source, so the cross-backbone pool is complete). Default stages are unchanged.
#          --artifacts DIR   backbone/p0/jacobian/select root for the causal stages (default: --out)
#          --legacy-select DIR   causal-main episode directory merged into the pool
#          --allow-incomplete-pool   causal-eval does not wait for every earlier held-out
#          --sync-tar PATH   pack --out to PATH atomically (temp name, then rename), as in the round-1 Drive sync
#          --sync-every-min N   repeat that pack while the queue runs (0 = once at the end only)
#
# Exit status: 0 = every chain finished; 3 = at least one chain stopped at the --deadline-hours check and none
# failed (re-run the same command to resume); otherwise the return code of the first failed stage (1 for a
# pre-flight failure or an unexplained xargs failure).  A failed chain does not stop the other chains.
# EXPANSION_DRIVER overrides the driver script (used by the exit-status tests with a fake stage).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT="$ROOT/results/expansion_20261016"; WORK=""; BIDS=all; STAGES=backbone,p0,jacobian,select,eval
JOBS=1; TIER=primary; DEV_FLAG=""; OPT_FLAG=""; DEVICE=cuda:0; DEADLINE_H=10.5; DRY=0
ARTIFACTS=""; LEGACY=""; INCOMPLETE=""; SYNC_TAR=""; SYNC_EVERY=30
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2;;
    --work) WORK=$2; shift 2;;
    --bids) BIDS=$2; shift 2;;
    --stages) STAGES=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    --tier) TIER=$2; shift 2;;
    --include-2016-optional) OPT_FLAG="--include-2016-optional"; shift;;
    --exclude-apst-dev) DEV_FLAG="--exclude-apst-dev"; shift;;
    --device) DEVICE=$2; shift 2;;
    --deadline-hours) DEADLINE_H=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    --artifacts) ARTIFACTS=$2; shift 2;;
    --legacy-select) LEGACY=$2; shift 2;;
    --allow-incomplete-pool) INCOMPLETE="--allow-incomplete-pool"; shift;;
    --sync-tar) SYNC_TAR=$2; shift 2;;
    --sync-every-min) SYNC_EVERY=$2; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
START=$(date +%s)
DEADLINE=$(awk -v s="$START" -v h="$DEADLINE_H" 'BEGIN{printf "%d", s + h * 3600}')
mkdir -p "$OUT/logs"
COMMON="--out $OUT ${WORK:+--work $WORK} --tier $TIER $OPT_FLAG $DEV_FLAG --device $DEVICE"
CAUSAL_ARGS="${ARTIFACTS:+--artifacts $ARTIFACTS} ${LEGACY:+--legacy-select $LEGACY} $INCOMPLETE"
DRIVER=${EXPANSION_DRIVER:-$ROOT/scripts/run_expansion.py}
RX="$PY $DRIVER"
sync_out() {
  [ -n "$SYNC_TAR" ] || return 0
  "$PY" -c 'import sys; from ssm_decode.expansion_causal import sync_tree_to_tar; sync_tree_to_tar(sys.argv[1], sys.argv[2])' "$OUT" "$SYNC_TAR"
}

# pre-flight: protocol and forbidden-session audit (fails loudly), then the status table
$RX check $COMMON --bid "$BIDS" > "$OUT/logs/preflight.log" 2>&1 || { echo "[queue] preflight FAILED"; tail -5 "$OUT/logs/preflight.log"; exit 1; }
if [ "$BIDS" = all ]; then
  BID_LIST=$($PY - <<PYEOF
import sys
sys.path.insert(0, "$ROOT")
from ssm_decode import expansion_protocol as ep
bbs = ep.select_tier(ep.build_backbones(), tier="$TIER", exclude_apst_dev=bool("$DEV_FLAG"))
# longest chains first so the tail of the queue is short
order = sorted(bbs, key=lambda b: -len(b.targets))
print(" ".join(b.bid for b in order))
PYEOF
)
else
  BID_LIST=${BIDS//,/ }
fi
# causal-eval reads every backbone's held-out episodes, so it runs only after every chain has finished
# causal-source. It is not part of the per-backbone chain even if it is listed in --stages.
POST=0
CHAIN_STAGES=$STAGES
case ",$STAGES," in
  *,causal-eval,*)
    POST=1
    CHAIN_STAGES=$(printf '%s' "$STAGES" | awk -F, '{out=""; for(i=1;i<=NF;i++) if($i!="causal-eval" && $i!="") out=out (out?",":"") $i; print out}')
    ;;
esac
echo "[queue] $(date '+%T') backbones: $BID_LIST | stages: $STAGES | jobs: $JOBS | out: $OUT"
[ "$DRY" = 1 ] && { $RX status $COMMON --bid "$BIDS" 2>&1 | grep -v -i warn; sync_out; exit 0; }

chain() {
  local bid=$1 log="$OUT/logs/$1.log" st rc extra
  for st in ${CHAIN_STAGES//,/ }; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "[queue] $bid: deadline reached before $st; re-run to resume" | tee -a "$log"
      echo 3 > "$OUT/logs/.rc_$bid"; return 3
    fi
    echo "[queue] $(date '+%T') $bid stage=$st" >> "$log"
    extra=""
    if [ "$st" = "causal-source" ]; then extra=$CAUSAL_ARGS; fi
    $RX "$st" $COMMON --bid "$bid" $extra >> "$log" 2>&1; rc=$?
    if [ $rc -ne 0 ]; then
      echo "[queue] $bid stage=$st FAILED rc=$rc (see $log)"
      echo "$rc" > "$OUT/logs/.rc_$bid"; return $rc
    fi
  done
  echo "[queue] $(date '+%T') $bid chain done"
  echo 0 > "$OUT/logs/.rc_$bid"
}
export -f chain
export -f sync_out
export OUT WORK STAGES CHAIN_STAGES DEADLINE COMMON RX CAUSAL_ARGS PY SYNC_TAR
SYNC_PID=""
if [ -n "$SYNC_TAR" ] && [ "$SYNC_EVERY" != 0 ]; then
  ( while true; do sleep $((SYNC_EVERY * 60)); sync_out || true; done ) &
  SYNC_PID=$!
fi
for b in $BID_LIST; do rm -f "$OUT/logs/.rc_$b"; done
# xargs only says "some invocation failed" (123); the per-chain return codes are collected from the .rc files
printf '%s\n' $BID_LIST | xargs -P "$JOBS" -I{} bash -c 'chain {}'
XRC=$?
EXIT=0; DEADLINE_HIT=0
for b in $BID_LIST; do
  if [ -f "$OUT/logs/.rc_$b" ]; then rc=$(cat "$OUT/logs/.rc_$b"); else rc=1; echo "[queue] $b: no return code recorded (chain crashed)"; fi
  if [ "$rc" = 3 ]; then DEADLINE_HIT=1
  elif [ "$rc" != 0 ] && [ "$EXIT" = 0 ]; then EXIT=$rc; fi
done
if [ "$EXIT" = 0 ] && [ "$XRC" != 0 ] && [ "$DEADLINE_HIT" = 0 ]; then EXIT=1; fi    # xargs failed without a recorded chain rc
FINAL=$EXIT
[ "$FINAL" = 0 ] && [ "$DEADLINE_HIT" = 1 ] && FINAL=3
if [ "$FINAL" = 0 ] && [ "$POST" = 1 ]; then
  for b in $BID_LIST; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "[queue] deadline reached before causal-eval $b; re-run to resume"
      FINAL=3
      break
    fi
    echo "[queue] $(date '+%T') $b stage=causal-eval"
    $RX causal-eval $COMMON --bid "$b" $CAUSAL_ARGS >> "$OUT/logs/$b.log" 2>&1; rc=$?
    if [ $rc -ne 0 ]; then
      echo "[queue] $b stage=causal-eval FAILED rc=$rc"
      [ "$FINAL" = 0 ] && FINAL=$rc
    fi
  done
fi
if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
sync_out
echo "[queue] $(date '+%T') finished (exit $FINAL); status:"
$RX status $COMMON --bid "$BIDS" 2>&1 | grep -v -i warn
exit "$FINAL"
