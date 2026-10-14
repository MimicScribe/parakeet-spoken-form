#!/bin/bash
# R3.27 owner: finish SBd0 to 4000 -> soup3 (3500/3750/4000) x alpha {1,0.85,0.7,0.5} -> eval all -> frozen select (scorer v2)
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
until modal volume ls parakeet-spoken-form exp/r3_SBd0 2>/dev/null | grep -q "step4000.pt$"; do sleep 120; done
echo "$(date +%H:%M) step4000 ready"
modal run modal_app.py::soup3 --run r3_SBd0 --name sbd0soup || { echo SOUP FAILED; exit 1; }
R="r3_SBd0@2000,r3_SBd0@3000,r3_SBd0@4000"; for a in 1.0 0.85 0.7 0.5; do R="$R,sbd0soup_a$a@0"; done
modal run modal_app.py::eval_multi --runs "$R" --tag _c || { echo EVAL FAILED; exit 1; }
C=""; for r in ${R//,/ }; do modal volume get --force parakeet-spoken-form "hyps/${r}_c" $S/r3d/hyps/ >/dev/null; C="$C,${r}_c"; done; C=${C#,}
python3 score_retrain.py $S/r3d/hyps "$C,r3_SBd0@1000_c,v10pt@1500_c" --base v10pt@1500_c --json $S/r3d/scores_final_d0.json > /dev/null || { echo SCORE FAILED; exit 1; }
python3 - "$C" <<'PY'
import json, sys, score_retrain as R
S="${LOGS:-./logs}"
res=json.load(open(S+"/r3d/scores_final_d0.json")); runs=sys.argv[1].split(",")
sel=R.select(res, runs, baseline="v10pt@1500_c")
print("FINAL SELECT", json.dumps({"run": sel["run"], "tied": sel["tied"]}))
for r in runs+["r3_SBd0@1000_c","v10pt@1500_c"]:
    m=R.sel_metrics(res[r]) if r!="v10pt@1500_c" else R.sel_metrics({**res[r],"wer_fleurs_delta":{}})
    s=res[r]; print("FINAL", r.ljust(24), json.dumps(m), "| gold", s.get("gold_reading"), s.get("gold_value"), "/", s.get("gold_n"), "onset2k", s.get("e21_dev_onset_2k_flips"))
PY
echo FINAL_D0_DONE
