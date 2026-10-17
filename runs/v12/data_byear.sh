#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
L=$S/v12/data_byear.log
modal run modal_app.py::make_tts --name v12byear --n-rows 1800 --seed 1201 --multi-frac 0.3 --carriers carriers_year_b.txt --speed 1.15,1.5 > $L 2>&1 || { echo "MAKE FAIL" >> $L; exit 1; }
modal run modal_app.py::filter_tts --name v12byear >> $L 2>&1 || { echo "FILTER FAIL" >> $L; exit 1; }
echo "BYEAR DATA OK $(date +%T)" >> $L
