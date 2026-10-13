#!/bin/bash
cd "$(dirname "$0")"
modal run ml_p3.py::label --langs ${P3LANGS:-de} --limit ${P3LIMIT:-0} --new-hours ${P3NEWH:-1.5} > ${P3LOG:-/tmp/p3_label.log} 2>&1
