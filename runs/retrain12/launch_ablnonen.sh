#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
modal run --detach modal_app.py::retrain --run rt_ablnonen_s1 --arm abl_nonen --seed 1 --kl-terms "blank,gnorm" > "$S/rt/arms/rt_ablnonen_s1.log" 2>&1
