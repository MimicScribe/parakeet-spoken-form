#!/bin/bash
# R3.31 v2 (external review #4 fixes): 3 warm-starts from sbd0soup_a1.0@0, L2-SP anchored to the init, 500 steps,
# lr 3e-5, dur 0: ctl2 = old mix; mp2 = mix3b + minimal pairs x3 (15% "and"); mpv = mp2 + verbatim-filtered d1p/d10p.
S=${LOGS:-./logs}
cd ~/proj/parakeet-spoken-form/pipeline
until grep -q "MP2_DONE\|MP2 FAILED" $S/mp2_tts.log; do sleep 30; done
grep -q MP2_DONE $S/mp2_tts.log || { echo "WS2 FAILED (tts)"; exit 1; }
modal run modal_app.py::build_mix3c || { echo "WS2 FAILED (mix)"; exit 1; }
COMMON="--arm S-B --dur-weight 0 --lr 3e-5 --warmup 20 --max-steps 500 --save-every 250 --init-from sbd0soup_a1.0@0 --anchor-init --no-check-share"
modal run modal_app.py::retrain3 --run r3_W2ctl --mix mix3 $COMMON > $S/r3_W2ctl.log 2>&1 &
sleep 20; modal run modal_app.py::retrain3 --run r3_W2mp --mix v2 $COMMON > $S/r3_W2mp.log 2>&1 &
sleep 20; modal run modal_app.py::retrain3 --run r3_W2mpv --mix v2v $COMMON > $S/r3_W2mpv.log 2>&1 &
wait
for r in r3_W2ctl r3_W2mp r3_W2mpv; do modal volume ls parakeet-spoken-form exp/$r | grep -q "step500.pt$" || { echo "WS2 FAILED (train $r)"; exit 1; }; done
echo "$(date +%H:%M) WS2_TRAINED"
R="r3_W2ctl@250,r3_W2ctl@500,r3_W2mp@250,r3_W2mp@500,r3_W2mpv@250,r3_W2mpv@500"
modal run modal_app.py::eval_multi --runs "$R" --tag _c > $S/ws2_eval.log 2>&1 &
modal run modal_app.py::eval_mp_dev --name mp2 --runs "$R,sbd0soup_a1.0@0,v10pt@1500,ultra" > $S/ws2_mpdev.log 2>&1 &
wait
for r in ${R//,/ } ; do modal volume get --force parakeet-spoken-form "hyps/${r}_c" $S/r3d/hyps/ >/dev/null; done
mkdir -p $S/mpdev2; for r in ${R//,/ } sbd0soup_a1.0@0 v10pt@1500 ultra; do mkdir -p $S/mpdev2/$r; modal volume get --force parakeet-spoken-form "hyps/$r/mp_dev.jsonl" $S/mpdev2/$r/ >/dev/null; done
modal volume get --force parakeet-spoken-form tts/mp2/dev_ok.jsonl $S/mpdev2/ >/dev/null
echo WS2_EVALED
