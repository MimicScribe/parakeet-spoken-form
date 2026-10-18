"""Coverage census of a built mix: what the trainer will actually read, per language x number class x source x form.

    python3 mix_report.py <dir with mix4/*.jsonl copies> [--min-spans 30] [--min-heldout 10] [--samples 3] [--json out]

For every wave-1 language and number class (the span kinds of ml_cands): spans and hours in training, split by source
(VoxPopuli, YODAS, FLEURS, synthetic) and form (full row, window with the number inside, at an edge, cut across an
edge), the number of distinct readings, and the held-out count. A cell below --min-spans (training) or --min-heldout
(held-out) is listed as a GAP; the pre-registration requires every gap to be either filled or written down as "not
claimed" before training. Also: label audits (digits / symbols left, empty labels), synthetic share per class, and a few
sample labels per cell for a spot check.
"""

import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from census_common import ext_row  # noqa: E402

WF = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°²³½¼¾⅓⅔ºª‰&]")


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def source(r):
    if r.get("src") == "tts" or r.get("origin") == "tts":
        return "synthetic"
    if r.get("origin") == "window":
        return "synthetic" if r.get("synthetic") else "real"
    s = str(r.get("src", ""))
    return {"vox": "voxpopuli", "yodas": "yodas"}.get(s, "fleurs" if "fleurs" in str(r.get("id", "")) else s or "?")


def kinds_of(r):
    """(kind, reading) per number span. Rows: their spans; windows: their parents' spans are not carried, so the kind
    comes from the window's own number tokens (counted as 'window')."""
    if r.get("spans"):
        return [(s.get("kind", "?"), s.get("reading", "")) for s in r["spans"]]
    if r.get("origin") == "window":
        return [(k, "") for k in r.get("kinds", [])]
    if r.get("cls"):
        return [(r["cls"], "")]
    return []


def main(argv):
    d = argv[0]
    opt, rest = {}, list(argv[1:])  # "--k v" (as in the usage line) and "--k=v" both work
    while rest:
        a = rest.pop(0)
        if a.startswith("--") and "=" in a:
            k, v = a.split("=", 1)
            opt[k] = v
        elif a.startswith("--"):
            opt[a] = rest.pop(0) if rest and not rest[0].startswith("--") else "1"
    min_spans, min_ho = int(opt.get("--min-spans", 30)), int(opt.get("--min-heldout", 10))
    nsamp = int(opt.get("--samples", 3))
    rep = {"langs": {}, "gaps": [], "audit": {}}
    for p3 in sorted(glob.glob(f"{d}/mix4/p3_*.jsonl")):
        lang = p3.rsplit("_", 1)[1][:2]
        train = rows(p3) + rows(f"{d}/mix4/tts_{lang}.jsonl")
        win = rows(f"{d}/mix4/win_{lang}.jsonl")
        ho = rows(f"{d}/mix4/heldout_{lang}.jsonl")
        cell = collections.defaultdict(lambda: {"spans": 0, "hours": 0.0, "by_source": collections.Counter(),
                                                "readings": set(), "samples": []})
        win_cls = collections.Counter()
        for w in win:
            for k, _ in kinds_of(w):
                win_cls[(k, "synthetic" if w.get("synthetic") else "real")] += 1
        ext_rows = [r for r in train if ext_row(r)]  # census_common.EXT_BY: decided mainly by the second transcript
        for r in train:
            for k, rd in kinds_of(r):
                c = cell[k]
                c["spans"] += 1
                c["hours"] += r["duration"] / 3600 / max(1, len(r.get("spans") or [1]))
                c["by_source"][source(r)] += 1
                if rd:
                    c["readings"].add(rd)
                if len(c["samples"]) < nsamp:
                    c["samples"].append(r["text"][:120])
        wform = collections.Counter()
        for w in win:
            wform["cut_number" if w.get("cut_number") else "edge_number" if w.get("edge_number") else "inside"] += 1
            wform["synthetic" if w.get("synthetic") else "real"] += 1
        hok = collections.Counter(k for r in ho for k, _ in kinds_of(r))
        audit = {"labels": len(train) + len(win), "symbol_or_digit": sum(bool(WF.search(r["text"])) for r in train + win),
                 "empty": sum(not r["text"].strip() for r in train + win)}
        out = {}
        for k, c in sorted(cell.items(), key=lambda t: -t[1]["spans"]):
            syn = c["by_source"].get("synthetic", 0)
            out[k] = {"spans": c["spans"], "hours": round(c["hours"], 3), "by_source": dict(c["by_source"]),
                      "synthetic_share": round(syn / c["spans"], 2) if c["spans"] else 0, "distinct_readings": len(c["readings"]),
                      "heldout_spans": hok.get(k, 0), "samples": c["samples"]}
            if c["spans"] < min_spans:
                rep["gaps"].append(f"{lang}/{k}: {c['spans']} training spans < {min_spans}")
            if hok.get(k, 0) < min_ho:
                rep["gaps"].append(f"{lang}/{k}: {hok.get(k, 0)} held-out spans < {min_ho}")
        rep["langs"][lang] = {"classes": out, "train_rows": len(train), "train_h": round(sum(r["duration"] for r in train) / 3600, 2),
                              "windows": len(win), "windows_h": round(sum(r["duration"] for r in win) / 3600, 2),
                              "window_forms": dict(wform), "heldout_rows": len(ho),
                              "window_classes": {f"{k}/{src}": v for (k, src), v in sorted(win_cls.items())},
                              "ext_rows": len(ext_rows), "ext_h": round(sum(r["duration"] for r in ext_rows) / 3600, 2)}
        rep["audit"][lang] = audit
    for lang, L in rep["langs"].items():
        print(f"\n== {lang}: {L['train_rows']} rows {L['train_h']} h | windows {L['windows']} ({L['windows_h']} h) "
              f"{L['window_forms']} | held-out {L['heldout_rows']} | audit {rep['audit'][lang]}")
        print(f"  recovered (second-recognizer) rows: {L['ext_rows']} ({L['ext_h']} h) | window classes: {L['window_classes']}")
        print(f"  {'class':12s} {'spans':>6s} {'h':>6s} {'syn%':>5s} {'readings':>8s} {'held':>5s}  sources")
        for k, c in L["classes"].items():
            print(f"  {k:12s} {c['spans']:6d} {c['hours']:6.2f} {int(100 * c['synthetic_share']):5d} "
                  f"{c['distinct_readings']:8d} {c['heldout_spans']:5d}  {c['by_source']}")
    print(f"\nGAPS ({len(rep['gaps'])}):")
    for g in rep["gaps"]:
        print("  ", g)
    if "--json" in opt:
        json.dump(rep, open(opt["--json"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bad = sum(a["symbol_or_digit"] + a["empty"] for a in rep["audit"].values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
