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
#          --bids a,b|all  --stages backbone,p0,select,eval  --jobs N  --tier primary|all
#          --exclude-apst-dev  --device cuda:0  --deadline-hours H  --dry-run
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/xinyuan/miniconda3/envs/spint/bin/python}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/.tools/mamba_deps:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT="$ROOT/results/expansion_20261016"; WORK=""; BIDS=all; STAGES=backbone,p0,select,eval
JOBS=1; TIER=primary; DEV_FLAG=""; DEVICE=cuda:0; DEADLINE_H=10.5; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2;;
    --work) WORK=$2; shift 2;;
    --bids) BIDS=$2; shift 2;;
    --stages) STAGES=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    --tier) TIER=$2; shift 2;;
    --exclude-apst-dev) DEV_FLAG="--exclude-apst-dev"; shift;;
    --device) DEVICE=$2; shift 2;;
    --deadline-hours) DEADLINE_H=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
START=$(date +%s)
DEADLINE=$(awk -v s="$START" -v h="$DEADLINE_H" 'BEGIN{printf "%d", s + h * 3600}')
mkdir -p "$OUT/logs"
COMMON="--out $OUT ${WORK:+--work $WORK} --tier $TIER $DEV_FLAG --device $DEVICE"
RX="$PY $ROOT/scripts/run_expansion.py"

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
echo "[queue] $(date '+%T') backbones: $BID_LIST | stages: $STAGES | jobs: $JOBS | out: $OUT"
[ "$DRY" = 1 ] && { $RX status $COMMON --bid "$BIDS" 2>&1 | grep -v -i warn; exit 0; }

chain() {
  local bid=$1 log="$OUT/logs/$1.log" st rc
  for st in ${STAGES//,/ }; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "[queue] $bid: deadline reached before $st; re-run to resume" | tee -a "$log"; return 3; fi
    echo "[queue] $(date '+%T') $bid stage=$st" >> "$log"
    $RX "$st" $COMMON --bid "$bid" >> "$log" 2>&1; rc=$?
    if [ $rc -ne 0 ]; then echo "[queue] $bid stage=$st FAILED rc=$rc (see $log)"; return $rc; fi
  done
  echo "[queue] $(date '+%T') $bid chain done"
}
export -f chain
export OUT WORK STAGES DEADLINE COMMON RX
printf '%s\n' $BID_LIST | xargs -P "$JOBS" -I{} bash -c 'chain {}'
echo "[queue] $(date '+%T') finished; status:"
$RX status $COMMON --bid "$BIDS" 2>&1 | grep -v -i warn
