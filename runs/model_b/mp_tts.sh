#!/bin/bash
cd ~/proj/parakeet-spoken-form/pipeline
modal run modal_app.py::make_tts_mp --name mp --n-pairs 1500 --seed 31 || { echo MP FAILED; exit 1; }
modal run modal_app.py::filter_tts --name mp || { echo MP FAILED; exit 1; }
modal run modal_app.py::mp_balance --name mp || { echo MP FAILED; exit 1; }
echo MP_DONE
