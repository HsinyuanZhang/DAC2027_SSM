#!/bin/bash
# Retrying `colab exec` (host side).  colab exec sometimes fails with "Connection was lost" or a
# TimeoutError while background jobs load the VM; background jobs are unaffected, so just retry.
# usage: colab_exec_retry.sh SESSION FILE.py [TIMEOUT_SECONDS]
for i in 1 2 3 4 5 6; do
  out=$(timeout $((${3:-90}+20)) colab exec -s "$1" -f "$2" --timeout ${3:-90} 2>&1); rc=$?
  if [ $rc -eq 0 ] && ! echo "$out" | grep -q "Connection was lost\|TimeoutError"; then echo "$out"; exit 0; fi
  sleep 8
done
echo "$out" | tail -5; exit 1
