#!/bin/bash
# E2 rehearsal: 40 steps (S-A, gradient shares on), then resume to 60 (kill-and-resume check)
cd $REPO/pipeline
modal run modal_app.py::retrain3 --run r3_e2 --arm S-A --max-steps 40 --save-every 20 --gradshare
echo "=== RESUME ==="
modal run modal_app.py::retrain3 --run r3_e2 --arm S-A --max-steps 60 --save-every 20 --gradshare
echo E2DONE
