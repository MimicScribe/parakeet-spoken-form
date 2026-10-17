#!/bin/bash
S=${LOGS:-./logs}
P=$REPO/pipeline
one() { r=$1; st=$2
  (cd $P && modal run modal_app.py::export_weights --run $r --step $st) > $S/sent/exp_$r-$st.log 2>&1 || { echo "EXPORT FAIL $r"; return; }
  modal volume get --force parakeet-spoken-form "export/$r@$st/decoder_joint.safetensors" $S/sent/$r-$st.safetensors >/dev/null 2>&1
  python3 $P/swap_weights.py $MODELS/parakeet-ultra-stock-95eaa59a $S/sent/$r-$st.safetensors $S/sent/bundle-$r-$st >> $S/sent/exp_$r-$st.log 2>&1 && echo "BUNDLE OK $r"
}
for r in r3_SA r3_SB r3_SC r3_SE; do one $r 1000 & sleep 3; done; wait
