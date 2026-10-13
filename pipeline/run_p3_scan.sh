#!/bin/bash
cd "$(dirname "$0")"
modal run ml_p3.py::scan --langs ${P3LANGS:-de} --hours ${P3HOURS:-4.5} --max-files ${P3FILES:-10} > ${P3LOG:-/tmp/p3_scan_de.log} 2>&1
