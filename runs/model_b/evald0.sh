#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
ST=$1
until modal volume ls parakeet-spoken-form exp/r3_SBd0 2>/dev/null | grep -q "step$ST.pt$"; do sleep 60; done
modal run modal_app.py::eval_multi --runs "r3_SBd0@$ST" --tag _c || { echo EVAL FAILED; exit 1; }
modal volume get --force parakeet-spoken-form "hyps/r3_SBd0@${ST}_c" $S/r3d/hyps/ >/dev/null
python3 - $ST <<'PY'
import sys, json, score_retrain as R
S="${LOGS:-./logs}/r3d/hyps"
st=sys.argv[1]
runs=["v10pt@1500_c","rt2_main_s2@3000_c","r3_SB@1000_c","r3_SB@2000_c",f"r3_SBd0@{st}_c"]
res={r:R.score(S,r,stock_run="ultra_rt_c") for r in runs}
for r in runs:
    if r!="v10pt@1500_c": res[r]["fleurs_bootstrap_vs_v10pt@1500_c"]=R.fleurs_bootstrap(S,r,"v10pt@1500_c")
for r in runs:
    m=R.sel_metrics(res[r]) if r!="v10pt@1500_c" else R.sel_metrics({**res[r],"wer_fleurs_delta":{}})
    s=res[r]; print("D0",r.ljust(22),json.dumps(m),"| gold",s.get("gold_reading"),s.get("gold_value"),"/",s.get("gold_n"),"onset2k",s.get("e21_dev_onset_2k_flips"))
json.dump(res,open(f"${LOGS:-./logs}/r3d/scores_d0_{st}.json","w"),indent=1)
PY
echo EVALD0_$ST DONE
