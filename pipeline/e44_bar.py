"""Paired comparison of two number-scoring runs over the same 44 Earnings-21 calls.

Each arm dir holds e11/ (11 calls) and e33/ (33 held-out calls), each with score.json (a per-call number-fidelity
score) and rows/<fid>_after_orphan.json (the transcript rows that were scored). The runs themselves come from the
private app-side evaluation; this script only compares two of them.

    python e44_bar.py <base_arm_dir> <cand_arm_dir>
"""

import glob
import json
import os
import sys

from score_retrain import BLEND, CHIMERA, WF

KEYS = ("hallucinatedHighSeverity", "corruptions", "droppedFromRender", "hallucinatedNumbers", "spliceDefects",
        "exactMatches")


def load(arm: str) -> dict:
    out = {}
    for part in ("e11", "e33"):
        d = json.load(open(f"{arm}/{part}/score.json"))["files"]
        for fid, v in d.items():
            row = {k: v[k] for k in KEYS}
            p = f"{arm}/{part}/rows/{fid}_after_orphan.json"
            text = " ".join(s["text"] for s in json.load(open(p))["segments"]) if os.path.exists(p) else ""
            row["blend"] = len(BLEND.findall(text))
            row["wf_chunks"] = len(WF.findall(text))
            row["chimeras"] = len(CHIMERA.findall(text))
            row["part"] = part
            out[fid] = row
    return out


def bar(base: dict, cand: dict) -> dict:
    fids = sorted(set(base) & set(cand))
    tot = lambda d, k, fs=fids: sum(d[f][k] for f in fs)  # noqa: E731
    res = {"calls": len(fids), "missing": sorted(set(base) ^ set(cand))}
    checks = {
        "high_severity_sum": tot(cand, "hallucinatedHighSeverity") <= tot(base, "hallucinatedHighSeverity"),
        "high_severity_no_call_plus2": all(cand[f]["hallucinatedHighSeverity"] - base[f]["hallucinatedHighSeverity"] < 2
                                           for f in fids),
        "corrupted": tot(cand, "corruptions") <= tot(base, "corruptions") + 2,
        "dropped": tot(cand, "droppedFromRender") <= tot(base, "droppedFromRender") + 2,
        "invented": tot(cand, "hallucinatedNumbers") <= tot(base, "hallucinatedNumbers") + 2,
        "splice": tot(cand, "spliceDefects") <= tot(base, "spliceDefects"),
        "exact": tot(cand, "exactMatches") >= tot(base, "exactMatches") - 2,
        "blend_zero": tot(cand, "blend") == 0,
        # (no English-digit key: this text is after the app's verbalizer, so digits never reach it; the model-level
        # digit count is the dev-bed key en_wf)
        "chimeras_zero": tot(cand, "chimeras") == 0,
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    for scope, fs in (("all44", fids), ("sealed33", [f for f in fids if cand[f]["part"] == "e33"])):
        res[scope] = {k: (tot(base, k, fs), tot(cand, k, fs)) for k in KEYS + ("blend", "wf_chunks", "chimeras")}
    res["per_call_rises"] = {f: {k: (base[f][k], cand[f][k]) for k in KEYS[:5] if cand[f][k] > base[f][k]}
                             for f in fids if any(cand[f][k] > base[f][k] for k in KEYS[:5])}
    res["owed"] = "seam number disagreements (rc.41 scorer) and live WER paired bootstrap are scored separately"
    return res


if __name__ == "__main__":
    b, c = load(sys.argv[1]), load(sys.argv[2])
    print(json.dumps(bar(b, c), indent=1))
