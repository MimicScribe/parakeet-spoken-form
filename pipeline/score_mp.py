"""R3.31 diagnostic: reading accuracy on the held-out minimal pairs (tts/mp dev: held-out carriers and voices).
Per run and per side ("twenty X" said / "two thousand X" said): right reading, the other reading, neither.
A model that follows the audio scores high on BOTH sides; a context prior shows as one side high, the other low.

    python3 score_mp.py <hyps_root> run1,run2,...
"""

import json
import re
import sys

import readings


def _n(t):
    return " " + re.sub(r"[^a-z ]", " ", t.lower().replace("-", " ")).replace("  ", " ") + " "


def score(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        reading = r["id"].rsplit("_", 1)[1] if r["id"].endswith("_twenty") else "two_thousand"
        y = None
        for cand in range(2010, 2040):
            if _n(readings.pair(cand) if reading == "twenty" else readings.cardinal(cand)).strip() in _n(r["ref"]):
                y = cand
        if y is None:
            continue
        right = readings.pair(y) if reading == "twenty" else readings.cardinal(y)
        other = readings.cardinal(y) if reading == "twenty" else readings.pair(y)
        h = " ".join(_n(r["hyp"]).split())
        h = f" {h} "
        k = "right" if f" {right} " in h else "other" if f" {other} " in h else "neither"
        out.setdefault(reading, {"right": 0, "other": 0, "neither": 0})[k] += 1
    return out


if __name__ == "__main__":
    root, runs = sys.argv[1], sys.argv[2].split(",")
    for run in runs:
        s = score(f"{root}/{run}/mp_dev.jsonl")
        print("MPDEV", run.ljust(26), json.dumps(s))
