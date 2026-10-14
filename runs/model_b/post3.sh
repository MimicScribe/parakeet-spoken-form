#!/bin/bash
# After ws3: score every model, select the best W3 candidate (frozen SELECT_ORDER), then sealed + 44-call app bar on it.
S=${LOGS:-./logs}
cd ~/proj/parakeet-spoken-form/pipeline
until grep -q "WS3_EVALED\|WS3 FAILED" $S/ws3.log; do sleep 30; done
grep -q WS3_EVALED $S/ws3.log || { echo "POST3 FAILED (ws3)"; exit 1; }
W="r3_W3@500_c,r3_W3@1000_c,w3soup_a1.0@0_c,w3soup_a0.5@0_c"
ALL="$W,r3_W2ctl@500_c,r3_W2mp@250_c,r3_W2mp@500_c,r3_W2mpv@250_c,r3_W2mpv@500_c,r3_SBd0@4000_c,sbd0soup_a1.0@0_c,v10pt@1500_c"
python3 score_retrain.py $S/r3d/hyps "$ALL" --base v10pt@1500_c --json $S/r3d/scores_all.json > /dev/null || { echo "POST3 FAILED (score)"; exit 1; }
BEST=$(python3 - "$W" <<'PY'
import json, sys, score_retrain as R
S="${LOGS:-./logs}"
res=json.load(open(S+"/r3d/scores_all.json")); w=sys.argv[1].split(",")
print(R.select(res, w, baseline="v10pt@1500_c")["run"])
PY
)
echo "BEST3 $BEST"
B=${BEST%_c}; RUN=${B%@*}; STEP=${B#*@}
# sealed: wait for the sealed encoder cache (sealed2.sh) before decoding the candidate on it
until grep -q "SEALED_DONE\|SEALED FAILED" $S/sealed2.log; do sleep 60; done
modal run modal_app.py::eval_multi --runs "$B" --sealed --tag _sealed > $S/post3_sealed.log 2>&1 && \
  modal volume get --force parakeet-spoken-form "hyps/${B}_sealed" $S/sealed/hyps/ >/dev/null && echo "POST3 SEALED_OK"
modal run modal_app.py::export_weights --run $RUN --step $STEP || { echo "POST3 FAILED (export)"; exit 1; }
modal volume get --force parakeet-spoken-form "export/$B/decoder_joint.safetensors" $S/sent/best3.safetensors >/dev/null
python3 swap_weights.py $HOME/models/parakeet-ultra-stock-95eaa59a $S/sent/best3.safetensors $HOME/models/parakeet-ultra-best3 || { echo "POST3 FAILED (bundle)"; exit 1; }
# (app-side evaluation of the bundle ran here; that harness is private)
echo POST3_DONE
