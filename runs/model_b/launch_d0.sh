#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
modal run --detach modal_app.py::retrain3 --run r3_SBd0 --arm S-B --dur-weight 0 --max-steps 4000 --save-every 250 > $S/r3_SBd0.log 2>&1
