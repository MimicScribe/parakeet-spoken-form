#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
RUNS="rt_main_s1 rt_main_s2 rt_stock_s1 rt_stock_s2 rt_ablen_s1 rt_ablnonen_s1"
while true; do
  left=0
  for r in $RUNS; do
    for st in 1000 3000; do
      tag="$r@$st"
      [ -f "$S/rt/ev/$tag.launched" ] && continue
      left=1
      if modal volume ls parakeet-spoken-form "/exp/$r" 2>/dev/null | grep -q "step$st.pt.done"; then
        touch "$S/rt/ev/$tag.launched"
        echo "$(date +%T) EVAL $tag"
        nohup modal run modal_app.py::eval_retrain --run "$r" --step "$st" > "$S/rt/ev/$tag.log" 2>&1 &
      fi
    done
  done
  [ $left = 0 ] && { echo "ALL EVALS LAUNCHED"; break; }
  sleep 300
done
wait
echo "ALL EVALS DONE"
