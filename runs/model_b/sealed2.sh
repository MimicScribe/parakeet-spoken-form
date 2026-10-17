#!/bin/bash
S=${LOGS:-./logs}
cd $REPO/pipeline
modal run modal_app.py::cache_beds3 --sealed || { echo SEALED FAILED; exit 1; }
modal run modal_app.py::eval_multi --runs "sbd0soup_a1.0@0,v10pt@1500,ultra" --sealed --tag _sealed || { echo SEALED FAILED; exit 1; }
for r in sbd0soup_a1.0@0_sealed ultra_rt_sealed v10pt@1500_sealed; do modal volume get --force parakeet-spoken-form hyps/$r $S/sealed/hyps/ >/dev/null; done
echo SEALED_DONE
