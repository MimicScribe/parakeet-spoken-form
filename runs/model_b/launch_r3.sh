#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
for a in S-A S-B S-C S-E; do
  r=r3_${a/-/}
  modal run --detach modal_app.py::retrain3 --run $r --arm $a --max-steps 4000 --save-every 250 > $S/$r.log 2>&1 &
  sleep 5
done
wait
