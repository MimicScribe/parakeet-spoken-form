#!/bin/bash
cd ~/proj/parakeet-spoken-form/pipeline
modal run modal_app.py::make_tts --name biz --n-rows 1600 --seed 13 --control-frac 0.05 --carriers carriers_biz.txt && \
modal run modal_app.py::filter_tts --name biz
echo BIZDONE
