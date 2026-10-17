#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
modal run modal_app.py::eval_multi --runs "r3_SB@2000" --tag _c || { echo EVAL FAILED; exit 1; }
modal volume get --force parakeet-spoken-form "hyps/r3_SB@2000_c" $S/r3d/hyps/
python3 score_retrain.py $S/r3d/hyps "r3_SB@1000_c,r3_SB@2000_c,v10pt@1500_c" --base v10pt@1500_c --json $S/r3d/scores_SB2000.json > /dev/null || { echo SCORE FAILED; exit 1; }
python3 - <<'PY'
import json, score_retrain as R
S="${LOGS:-./logs}"
res=json.load(open(S+"/r3d/scores_SB2000.json"))
for r in ["v10pt@1500_c","r3_SB@1000_c","r3_SB@2000_c"]:
    m=R.sel_metrics({**res[r],"wer_fleurs_delta":{}}) if r=="v10pt@1500_c" else R.sel_metrics(res[r])
    print("SB2000",r,json.dumps(m),"garble",res[r].get("year_garble"),"onset2k",res[r].get("e21_dev_onset_2k_flips"))
PY
echo EVAL2000_DONE
