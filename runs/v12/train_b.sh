#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
L=$S/v12/train_b.log
modal run modal_app.py::merge_tts --name v12b --parts v10pt,v12byear > $L 2>&1 || { echo "MERGE FAIL" >> $L; exit 1; }
echo "MERGE OK $(date +%T)" >> $L
K="--tts v12b --lspc-h 20 --save-every 500 --no-spec-augment --icsi-h 99 --batch-duration 400 --max-steps 1500 --kl-weight 10 --kl-replay-only --kl-punct 30 --kl-punct-rows replay --kl-terms blank,gnorm"
modal run modal_app.py::run --run v12b $K --seed 1 > $S/v12/v12b.log 2>&1 &
sleep 40
modal run modal_app.py::run --run v12bs2 $K --seed 2 > $S/v12/v12bs2.log 2>&1 &
wait; echo "TRAIN DONE $(date +%T)" >> $L
