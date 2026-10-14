#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
modal run --detach modal_app.py::retrain3 --run r3_SB --arm S-B --max-steps 4000 --save-every 250 > $S/r3_SB_resume.log 2>&1
