#!/bin/bash
cd "$(dirname "$0")"
L=de,es,fr,it,nl,pt
modal run ml_p3.py::scan --langs $L --hours 4.5 --max-files 300 > ../p3_scan2.log 2>&1 || exit 1
modal run ml_p3.py::label --langs $L --new-hours 1.5 > ../p3_label2.log 2>&1
