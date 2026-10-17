#!/bin/bash
# R3.20 step-1000 screening: wait for 4 checkpoints -> one cached eval (skip e21_dev) -> score vs v10pt -> frozen select
S=${LOGS:-./logs}
P=$REPO/pipeline
ST=${1:-1000}; RUNS="r3_SA r3_SB r3_SC r3_SE"
for r in $RUNS; do
  until modal volume ls parakeet-spoken-form exp/$r 2>/dev/null | grep -q "step$ST.pt"; do sleep 120; done
  echo "$(date +%H:%M) $r step$ST ready"
done
L=$(for r in $RUNS; do printf "%s@$ST," $r; done); L=${L%,}
cd $P && modal run modal_app.py::eval_multi --runs "$L" --tag _c --skip e21_dev || { echo "EVAL FAILED"; exit 1; }
echo "$(date +%H:%M) eval done"
for r in $RUNS; do modal volume get --force parakeet-spoken-form "hyps/$r@${ST}_c" $S/r3d/hyps/ ; done
C=$(for r in $RUNS; do printf "%s@${ST}_c," $r; done); C=${C%,}
python3 score_retrain.py $S/r3d/hyps "$C,v10pt@1500_c" --base v10pt@1500_c --json $S/r3d/scores_$ST.json > /dev/null || { echo "SCORE FAILED"; exit 1; }
python3 - $S/r3d/scores_$ST.json "$C" <<'PY'
import json, sys
import score_retrain as R
res = json.load(open(sys.argv[1])); runs = sys.argv[2].split(",")
sel = R.select(res, runs, baseline="v10pt@1500_c")
base = R.sel_metrics({**res["v10pt@1500_c"], "wer_fleurs_delta": {}})
print("SELECT", json.dumps({"run": sel["run"], "tied": sel["tied"]}))
for r, m in list(sel["metrics"].items()) + [("v10pt@1500_c", base)]:
    print(f"{r:20s}", json.dumps(m))
PY
echo "STEP$ST DONE"
