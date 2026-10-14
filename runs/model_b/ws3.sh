#!/bin/bash
# R3.31d candidate run (owner OK required): v3 mix (pairs x2 + verbatim filter, share 39.2%), 1000 steps from
# sbd0soup_a1.0@0, L2-SP anchored to the init, lr 3e-5, dur 0; soup 500/750/1000 x alpha {1, 0.5} toward the winner.
S=${LOGS:-./logs}
cd ~/proj/parakeet-spoken-form/pipeline
modal run modal_app.py::retrain3 --run r3_W3 --mix v3 --arm S-B --dur-weight 0 --lr 3e-5 --warmup 20 --max-steps 1000 \
  --save-every 250 --init-from sbd0soup_a1.0@0 --anchor-init > $S/r3_W3.log 2>&1
modal volume ls parakeet-spoken-form exp/r3_W3 | grep -q "step1000.pt$" || { echo "WS3 FAILED (train)"; exit 1; }
modal run modal_app.py::soup3 --run r3_W3 --steps 500,750,1000 --alphas 1.0,0.5 --base sbd0soup_a1.0@0 --name w3soup || { echo "WS3 FAILED (soup)"; exit 1; }
R="r3_W3@500,r3_W3@1000,w3soup_a1.0@0,w3soup_a0.5@0"
modal run modal_app.py::eval_multi --runs "$R" --tag _c > $S/ws3_eval.log 2>&1 &
modal run modal_app.py::eval_mp_dev --name mp2 --runs "$R" > $S/ws3_mpdev.log 2>&1 &
wait
# eval_mp_dev writes hyps/<run>/mp_dev.jsonl for whichever set it decodes: fetch the mp2 copy before gold62 overwrites it
for r in ${R//,/ }; do modal volume get --force parakeet-spoken-form "hyps/${r}_c" $S/r3d/hyps/ >/dev/null
  mkdir -p $S/mpdev2/$r $S/gold62h/$r; modal volume get --force parakeet-spoken-form "hyps/$r/mp_dev.jsonl" $S/mpdev2/$r/ >/dev/null; done
modal run modal_app.py::eval_mp_dev --name gold62 --runs "$R" > $S/ws3_gold.log 2>&1
for r in ${R//,/ }; do modal volume get --force parakeet-spoken-form "hyps/$r/mp_dev.jsonl" $S/gold62h/$r/ >/dev/null; done
echo "$(date +%H:%M) WS3_EVALED"
