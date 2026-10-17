"""R3.31 diagnostic: reading accuracy on the held-out minimal pairs (tts/<mp> dev: held-out carriers and voices).
Per run and per side ("twenty X" said / "two thousand [and] X" said): right reading, the other reading, neither.
A model that follows the audio scores high on BOTH sides; a context prior shows as one side high, the other low.
Year and side come from the dev manifest (no inference from the text); a reading must not be the
head of a longer number ("twenty twenty" inside "twenty twenty four"); "and" is optional on the "two thousand" side.

    python3 score_mp.py <hyps_root> run1,run2,... <dev_ok.jsonl>
"""

import json
import re
import sys

import readings

_U = "one|two|three|four|five|six|seven|eight|nine"
_END = rf"(?!(?:{_U}|hundred|thousand|million|billion|point) )"


def _n(t):
    return " " + " ".join(re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " ")).split()) + " "


def score(path, meta):
    out = {}
    for line in open(path):
        r = json.loads(line)
        m = meta.get(r["id"])
        if m is None:
            continue
        y, side = m["year"], m["reading"]
        tail = readings.pair(y).split(" ", 1)[1]
        pats = {"twenty": rf" twenty {tail} {_END}", "two_thousand": rf" two thousand (?:and )?{tail} {_END}"}
        h = _n(r["hyp"])
        other = "two_thousand" if side == "twenty" else "twenty"
        k = "right" if re.search(pats[side], h) else "other" if re.search(pats[other], h) else "neither"
        out.setdefault(side, {"right": 0, "other": 0, "neither": 0})[k] += 1
    return out


if __name__ == "__main__":
    root, runs, man = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    meta = {r["id"]: r for r in map(json.loads, open(man))}
    for run in runs:
        print("MPDEV", run.ljust(26), json.dumps(score(f"{root}/{run}/mp_dev.jsonl", meta)))
