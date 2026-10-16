#!/bin/bash
# Retrying `colab exec` (host side).  colab exec sometimes fails with "Connection was lost" or a
# TimeoutError while background jobs load the VM; background jobs are unaffected.
#
# usage: colab_exec_retry.sh SESSION FILE.py [TIMEOUT_SECONDS]
#
# Retry policy (review item 5).  An exec file that LAUNCHES background jobs must never be started twice:
#   * rc == 0  -> never re-run.  If the output still contains "Connection was lost"/"TimeoutError" the outcome
#     is unknown: the output is printed and the script exits with 3 (verify on the VM; do not blindly re-run).
#   * rc != 0  -> retried up to COLAB_EXEC_TRIES (default 6) times, unless the file contains a line starting
#     with "# NONIDEMPOTENT": then there is exactly one attempt (exit 1) and the caller must check the VM.
#   * Launch files should be idempotent anyway: create a lock with os.open(path, O_CREAT|O_EXCL) before the
#     first Popen and skip the launch when it exists (see docs/colab_runbook.md); such files need no tag.
# Environment: COLAB_EXEC_CMD (default "colab"), COLAB_EXEC_SLEEP (default 8), COLAB_EXEC_TRIES (default 6).
CMD=${COLAB_EXEC_CMD:-colab}
SLEEP=${COLAB_EXEC_SLEEP:-8}
TRIES=${COLAB_EXEC_TRIES:-6}
T=${3:-90}
if grep -q "^# NONIDEMPOTENT" "$2" 2>/dev/null; then TRIES=1; fi
for i in $(seq 1 "$TRIES"); do
  out=$(timeout $((T+20)) $CMD exec -s "$1" -f "$2" --timeout "$T" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "$out"
    if echo "$out" | grep -q "Connection was lost\|TimeoutError"; then
      echo "colab_exec_retry: rc=0 but the output reports a lost connection/timeout; NOT re-running (outcome unknown, check the VM)." >&2
      exit 3
    fi
    exit 0
  fi
  [ "$i" -lt "$TRIES" ] && sleep "$SLEEP"
done
echo "$out" | tail -5; exit 1
