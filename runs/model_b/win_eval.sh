#!/bin/bash
S=${LOGS:-./logs}
cd ~/proj/parakeet-spoken-form/pipeline
modal run modal_app.py::export_weights --run sbd0soup_a1.0 --step 0 && \
  modal volume get --force parakeet-spoken-form "export/sbd0soup_a1.0@0/decoder_joint.safetensors" $S/sent/sbd0soup_a1.0-0.safetensors && echo EXPORT_OK &
modal run modal_app.py::eval_multi --runs "sbd0soup_a1.0@0,v10pt@1500,ultra" --sealed --tag _sealed && echo SEALED_OK
wait
