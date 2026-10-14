#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
L=$S/v12/data_year.log
modal run modal_app.py::make_tts --name v12year --n-rows 1800 --seed 1201 --multi-frac 0.3 --carriers carriers_year.txt --speed 1.15,1.5 > $L 2>&1 || { echo "MAKE FAIL" >> $L; exit 1; }
modal run modal_app.py::filter_tts --name v12year >> $L 2>&1 || { echo "FILTER FAIL" >> $L; exit 1; }
echo "YEAR DATA OK $(date +%T)" >> $L
