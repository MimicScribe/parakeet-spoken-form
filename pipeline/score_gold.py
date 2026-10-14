"""R3.31 owner-gold harm check: every owner-labelled clip on the volume (c/g clips are TRAINING rows, d clips dev),
decoded by eval_mp_dev --name gold62. Per checkpoint: reading class right (two thousand [and] X vs twenty X) and value
kept, split by what the owner heard. A treatment that flips owner-verified "twenty" clips is harmful whatever else wins.

    python3 score_gold.py <hyps_root> run1,run2,... <gold62 manifest>
"""

import json
import re
import sys

import score_retrain as R

root, runs, man = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
meta = {r["id"]: r for r in map(json.loads, open(man))}
for run in runs:
    c = {}
    for line in open(f"{root}/{run}/mp_dev.jsonl"):
        r = json.loads(line)
        m = meta[r["id"]]
        heard, y = m["heard"], m["value"]
        if heard not in ("two thousand", "two thousand and", "twenty") or not 2010 <= y < 2100:
            continue
        side = "twenty" if heard == "twenty" else "two_thousand"
        tail = R._two(y - 2000)[0]
        pat = rf"twenty {tail}" if side == "twenty" else rf"two thousand (?:and )?{tail}"
        h = R._ynorm(r["hyp"])
        ok = re.search(rf" {pat} (?!(?:{R._U}|hundred|thousand|million|billion|point) )", h) is not None
        split = "dev" if r["id"].startswith("d") else "train"
        k = c.setdefault(f"{split}_{side}", [0, 0, 0])
        k[0] += 1
        k[1] += ok
        k[2] += R.year_value_kept(str(y), r["hyp"])[1]
    print("GOLD", run.ljust(24), " ".join(f"{k}: reading {v[1]}/{v[0]} value {v[2]}" for k, v in sorted(c.items())))
