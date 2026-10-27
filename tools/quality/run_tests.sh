#!/bin/sh
# every test in this directory with plain `python3 -I` (no pytest). test_semantic skips the real-weights test when MiniLM is not on the machine.
cd "$(dirname "$0")" || exit 1
rc=0
for t in test_dupcheck test_variety test_semantic; do
  printf "%s: " "$t"
  out=$(python3 -I "$t.py" 2>&1); code=$?
  echo "$out" | tail -1
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
