"""Seam score that needs no single-sentence reference: each sentence is decoded at 5 window offsets, so a value right
at some offsets and wrong at others was broken by where a window edge fell.
Usage: offset_score.py <workdir> <arm>... [--offsets N]   (N = decodes expected per sentence and slot, default 5)
Per language and slot: seam = wrong readings of sentences right at >= 1 offset; always = sentences wrong at every offset.
Every row of cases_<arm> must have its "<k>:long" parse (a missing one is an error, never a wrong reading), and every
sentence must have exactly N decodes."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_values import load_parse, slot_ok  # noqa: E402

SL = ["L", "PH", "P", "T", "Y", "PCT", "N", "M", "A", "B"]
ORDER = "en es fr de it nl pt pl ro uk ru sv cs hu fi sk bg hr da el lt sl".split()


def ok(s, v, rs):
    return any(slot_ok(s, v, x) for x in rs)


def seam_counts(rows, P, offsets=5):
    """{lang: Counter(n, seam, always, n_<slot>, seam_<slot>, always_<slot>)}. Raises KeyError for a row without its
    long parse and ValueError for a sentence / slot without exactly `offsets` decodes."""
    by = collections.defaultdict(list)  # (sentence id, slot) -> [right at each offset]
    for k, r in enumerate(rows):
        key = f"{k}:long"
        if key not in P:
            raise KeyError(f"parse_out has no {key} (row {k}, sentence {r['id']})")
        for s, v in r["values"].items():
            by[(r["id"], s)].append(ok(s, v, P[key]))
    bad = {k: len(xs) for k, xs in by.items() if len(xs) != offsets}
    if bad:
        raise ValueError(f"{len(bad)} sentence slots without {offsets} decodes, e.g. {sorted(bad.items())[:3]}")
    c = collections.defaultdict(collections.Counter)
    for (sid, s), xs in by.items():
        t = c[sid.split("_")[0]]
        wrong = xs.count(False)
        t["n"] += len(xs); t["n_" + s] += len(xs)
        if wrong and wrong < len(xs):
            t["seam"] += wrong; t["seam_" + s] += wrong
        if wrong == len(xs):
            t["always"] += wrong; t["always_" + s] += wrong
    return c


def main(argv):
    offsets = 5
    if "--offsets" in argv:
        i = argv.index("--offsets")
        offsets = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    W, arms = argv[0], argv[1:]
    res = {}
    for a in arms:
        rows = [json.loads(l) for l in open(f"{W}/cases_{a}.jsonl") if l.strip()]
        res[a] = seam_counts(rows, load_parse(f"{W}/parse_out_{a}.jsonl"), offsets)
    print("seam-broken readings (wrong at this offset, right at another) | always-wrong readings; per language, arms " + " → ".join(arms))
    langs = sorted({L for a in arms for L in res[a]}, key=lambda L: (ORDER.index(L) if L in ORDER else len(ORDER), L))
    tot = {a: collections.Counter() for a in arms}
    for L in langs:  # every language present (ORDER first, then the rest), all of them in the totals
        for a in arms:
            tot[a].update(res[a][L])
        print(f"{L:3} seam " + " → ".join(f"{res[a][L]['seam']:3}" for a in arms) + " | always " + " → ".join(f"{res[a][L]['always']:3}" for a in arms)
              + " | seam by slot " + "  ".join(f"{s}:" + "→".join(str(res[a][L]['seam_' + s]) for a in arms) for s in SL if any(res[a][L]['seam_' + s] for a in arms)))
    for a in arms:
        t = tot[a]
        print(a, f"seam={t['seam']} always={t['always']} of {t['n']} slot readings;", " ".join(f"{s}:{t['seam_' + s]}+{t['always_' + s]}/{t['n_' + s]}" for s in SL))
    return tot


if __name__ == "__main__":
    main(sys.argv[1:])
