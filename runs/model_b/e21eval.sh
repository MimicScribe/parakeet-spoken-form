#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
modal run modal_app.py::eval_multi --runs "r3_SA@1000,r3_SB@1250" --tag _c --skip "e22_numbers,e22_other,lspc_dev,d1_dev,vox_dev,fleurs_wer,fleurs_num,noise,adv,fast,d9_dev,biz_dev" || { echo EVAL FAILED; exit 1; }
for r in r3_SA@1000_c r3_SB@1250_c; do mkdir -p $S/r3d/hyps/$r; modal volume get --force parakeet-spoken-form "hyps/$r/e21_dev.jsonl" $S/r3d/hyps/$r/e21_dev.jsonl; done
python3 - <<'PY'
import json,sys
sys.path.insert(0,".")
import yearflip, score_retrain as R
S="${LOGS:-./logs}"
for r in ["v10pt@1500_c","rt2_main_s2@3000_c","r3_SA@1000_c","r3_SB@1250_c"]:
    rows=[json.loads(l) for l in open(f"{S}/r3d/hyps/{r}/e21_dev.jsonl")]
    w=[{"call":x["id"].split("_")[0],"start":int(x["id"].split("_")[1])/16000,"duration":14.96,"id":x["id"],"hyp":x["hyp"]} for x in rows]
    f=yearflip.flips(w,"hyp"); g=sum(len(R.year_garbles(x["hyp"])) for x in rows)
    tk=sum(x["hyp"].lower().count("two thousand") for x in rows); tw=sum(len(R.re.findall(r"\btwenty (?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",x["hyp"],R.re.I)) for x in rows)
    print("E21DEV",r,"flips",f["flips"],"onset_2k",f["onset_2k_flips"],"garble",g,"'two thousand'",tk,"'twenty 10-20'",tw)
PY
