#!/bin/sh
# all tool tests with plain `python3 -I` (no pytest); exit 1 if any file fails. Tests that need torch or model weights skip themselves.
cd "$(dirname "$0")" || exit 1
rc=0
run() {  # directory, test file
  printf "%s/%s: " "$1" "$2"
  out=$(cd "$1" && python3 -I "$2" 2>&1); code=$?
  echo "$out" | tail -1
  [ "$code" -eq 0 ] || rc=1
}
run numbers marker.py
run numbers/scorer test_numscore.py
run numbers/scorer test_repeatcount.py
for t in numread_golden test_numread test_fair_reader; do run numread $t.py; done
for t in test_spanfind test_ledger test_labeller_core test_label_runner test_calibrate_margin test_tiering test_load_manifest test_w2v_torch; do run labels $t.py; done
for t in test_edge_rule test_boundary_cut test_boundary_cut_dip test_copycount_gate; do run audio $t.py; done
for t in test_dupcheck test_variety test_semantic; do run quality $t.py; done
exit $rc
