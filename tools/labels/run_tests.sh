#!/bin/sh
# every test in this directory with plain `python3 -I` (no pytest). test_w2v_torch skips itself without torch.
cd "$(dirname "$0")" || exit 1
rc=0
for t in test_spanfind test_ledger test_labeller_core test_label_runner test_calibrate_margin test_tiering test_load_manifest test_w2v_torch; do
  printf "%s: " "$t"
  out=$(python3 -I "$t.py" 2>&1); code=$?
  echo "$out" | tail -1
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
