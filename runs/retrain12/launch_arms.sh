#!/bin/bash
S=${LOGS:-./logs}
cd $HOME/proj/parakeet-spoken-form/pipeline
while read run arm seed; do
  nohup modal run --detach modal_app.py::retrain --run "$run" --arm "$arm" --seed "$seed" > "$S/rt/arms/$run.log" 2>&1 &
  sleep 20
done <<'LIST'
rt_main_s1 main 1
rt_main_s2 main 2
rt_stock_s1 stockinit 1
rt_stock_s2 stockinit 2
rt_ablen_s1 abl_en 1
rt_ablnonen_s1 abl_nonen 1
LIST
wait
