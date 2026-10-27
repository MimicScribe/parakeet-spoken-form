#!/bin/sh
# every test in this directory with plain `python3 -I` (no pytest)
cd "$(dirname "$0")" || exit 1
rc=0
for t in test_edge_rule test_boundary_cut test_boundary_cut_dip test_copycount_gate; do
  printf "%s: " "$t"
  out=$(python3 -I "$t.py" 2>&1); code=$?
  echo "$out" | tail -1
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
