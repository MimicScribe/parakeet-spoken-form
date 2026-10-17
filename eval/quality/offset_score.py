"""Seam score that needs no single-sentence reference: each sentence is decoded at 5 window offsets, so a value right
at some offsets and wrong at others was broken by where a window edge fell. Usage: offset_score.py <workdir> <arm>...
Per language and slot: seam = wrong readings of sentences right at >= 1 offset; always = sentences wrong at every offset."""
import collections, json, sys
sys.argv, W, arms = sys.argv[:1], sys.argv[1], sys.argv[2:]
src = open(__file__.replace("offset_score.py", "score_values.py")).read()
ns = {}
exec("import re\ndef num(s):\n    try: return float(s)\n    except: return None\ndef slot_ok" + src.split("def slot_ok")[1].split("def correct")[0], ns)
ok = lambda s, v, rs: any(ns["slot_ok"](s, v, x) for x in rs)
SL = ["L", "PH", "P", "T", "Y", "PCT", "N", "M", "A", "B"]
ORDER = "en es fr de it nl pt pl ro uk ru sv cs hu fi sk bg hr da el lt sl".split()
res = {}
for a in arms:
    rows = [json.loads(l) for l in open(f"{W}/cases_{a}.jsonl")]
    P = {}
    for l in open(f"{W}/parse_out_{a}.jsonl"):
        if l.strip(): r = json.loads(l); P[r["key"]] = (r["values"], r.get("alt", []))
    by = collections.defaultdict(list)  # (sentence id, slot) -> [right at each offset]
    for k, r in enumerate(rows):
        lv = P.get(f"{k}:long", ([],))
        for s, v in r["values"].items(): by[(r["id"], s)].append(ok(s, v, lv))
    c = collections.defaultdict(collections.Counter)
    for (sid, s), xs in by.items():
        L = sid.split("_")[0]; t = c[L]; wrong = xs.count(False)
        t["n"] += len(xs); t["n_" + s] += len(xs)
        if wrong and wrong < len(xs): t["seam"] += wrong; t["seam_" + s] += wrong
        if wrong == len(xs): t["always"] += wrong; t["always_" + s] += wrong
    res[a] = c
print("seam-broken readings (wrong at this offset, right at another) | always-wrong readings; per language, arms " + " → ".join(arms))
tot = {a: collections.Counter() for a in arms}
for L in ORDER:
    for a in arms: tot[a].update(res[a][L])
    print(f"{L:3} seam " + " → ".join(f"{res[a][L]['seam']:3}" for a in arms) + " | always " + " → ".join(f"{res[a][L]['always']:3}" for a in arms)
          + " | seam by slot " + "  ".join(f"{s}:" + "→".join(str(res[a][L]['seam_' + s]) for a in arms) for s in SL if any(res[a][L]['seam_' + s] for a in arms)))
for a in arms:
    t = tot[a]
    print(a, f"seam={t['seam']} always={t['always']} of {t['n']} slot readings;", " ".join(f"{s}:{t['seam_' + s]}+{t['always_' + s]}/{t['n_' + s]}" for s in SL))
