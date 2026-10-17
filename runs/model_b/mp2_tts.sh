#!/bin/bash
cd $REPO/pipeline
modal run modal_app.py::make_tts_mp --name mp2 --n-pairs 2000 --seed 32 --and-frac 0.15 || { echo MP2 FAILED; exit 1; }
modal run modal_app.py::filter_tts --name mp2 || { echo MP2 FAILED; exit 1; }
modal run modal_app.py::mp_balance --name mp2 || { echo MP2 FAILED; exit 1; }
echo MP2_DONE
