#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
while read run arm seed; do
  nohup modal run --detach modal_app.py::retrain --run "$run" --arm "$arm" --seed "$seed" > "$S/rt/arms2/$run.log" 2>&1 &
  sleep 20
done <<'LIST'
rt2_main_s1 main 1
rt2_main_s2 main 2
rt2_stock_s1 stockinit 1
rt2_stock_s2 stockinit 2
LIST
wait
