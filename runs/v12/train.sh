#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
L=$S/v12/train.log
modal run modal_app.py::merge_tts --name v12 --parts v10pt,v12year > $L 2>&1 || { echo "MERGE FAIL" >> $L; exit 1; }
echo "MERGE OK $(date +%T)" >> $L
K="--tts v12 --lspc-h 20 --save-every 500 --no-spec-augment --icsi-h 99 --batch-duration 400 --max-steps 1500 --kl-weight 10 --kl-replay-only --kl-punct 30 --kl-punct-rows replay --kl-terms blank,gnorm"
modal run modal_app.py::run --run v12y $K --seed 1 > $S/v12/v12y.log 2>&1 &
sleep 40
modal run modal_app.py::run --run v12ys2 $K --seed 2 > $S/v12/v12ys2.log 2>&1 &
wait; echo "TRAIN DONE $(date +%T)" >> $L
