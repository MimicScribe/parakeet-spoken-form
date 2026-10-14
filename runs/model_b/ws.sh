#!/bin/bash
# R3.31 warm-start experiment: treatment (fixed mix + minimal pairs) vs control (old mix), both 500 steps from the
# SBd0 winner sbd0soup_a1.0@0, S-B config, dur 0, lr 3e-5, warmup 20.
S=${LOGS:-./logs}
cd ~/proj/parakeet-spoken-form/pipeline
until grep -q "MP_DONE\|MP FAILED" $S/mp_tts.log; do sleep 30; done
grep -q MP_DONE $S/mp_tts.log || { echo "WS FAILED (tts)"; exit 1; }
modal run modal_app.py::build_mix3b || { echo "WS FAILED (mix)"; exit 1; }
COMMON="--arm S-B --dur-weight 0 --lr 3e-5 --warmup 20 --max-steps 500 --save-every 250 --init-from sbd0soup_a1.0@0 --no-check-share"
modal run modal_app.py::retrain3 --run r3_WSmp --mix mp $COMMON > $S/r3_WSmp.log 2>&1 &
sleep 20
modal run modal_app.py::retrain3 --run r3_WSctl --mix mix3 $COMMON > $S/r3_WSctl.log 2>&1 &
wait
for r in r3_WSmp r3_WSctl; do modal volume ls parakeet-spoken-form exp/$r | grep -q "step500.pt$" || { echo "WS FAILED (train $r)"; exit 1; }; done
echo "$(date +%H:%M) WS_TRAINED"
R="r3_WSmp@250,r3_WSmp@500,r3_WSctl@250,r3_WSctl@500"
modal run modal_app.py::eval_multi --runs "$R" --tag _c > $S/ws_eval.log 2>&1 &
modal run modal_app.py::eval_mp_dev --runs "$R,sbd0soup_a1.0@0,v10pt@1500,ultra" > $S/ws_mpdev.log 2>&1 &
wait
C=""; for r in ${R//,/ } ; do modal volume get --force parakeet-spoken-form "hyps/${r}_c" $S/r3d/hyps/ >/dev/null; C="$C,${r}_c"; done; C=${C#,}
mkdir -p $S/mpdev; for r in ${R//,/ } sbd0soup_a1.0@0 v10pt@1500 ultra; do mkdir -p $S/mpdev/$r; modal volume get --force parakeet-spoken-form "hyps/$r/mp_dev.jsonl" $S/mpdev/$r/ >/dev/null; done
python3 score_mp.py $S/mpdev "r3_WSmp@250,r3_WSmp@500,r3_WSctl@250,r3_WSctl@500,sbd0soup_a1.0@0,v10pt@1500,ultra"
python3 score_retrain.py $S/r3d/hyps "$C,sbd0soup_a1.0@0_c,v10pt@1500_c" --base v10pt@1500_c --json $S/r3d/scores_ws.json > /dev/null || { echo "WS FAILED (score)"; exit 1; }
python3 - "$C" <<'PY'
import json, sys, score_retrain as R
S="${LOGS:-./logs}"
res=json.load(open(S+"/r3d/scores_ws.json"))
for r in sys.argv[1].split(",")+["sbd0soup_a1.0@0_c","v10pt@1500_c"]:
    m=R.sel_metrics(res[r]) if r!="v10pt@1500_c" else R.sel_metrics({**res[r],"wer_fleurs_delta":{}})
    s=res[r]; print("WSRES", r.ljust(24), json.dumps(m), "| gold", s.get("gold_reading"), s.get("gold_value"), "/", s.get("gold_n"), "onset2k", s.get("e21_dev_onset_2k_flips"), "wer_e22_other", s.get("wer_e22_other"))
PY
echo WS_DONE
