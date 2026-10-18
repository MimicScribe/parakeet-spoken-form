#!/usr/bin/env python3
"""Oracle recall of the public reader: if a model wrote exactly the words the speaker said, how often would the
scorer call the number right? A language below the bar (default 95%) is dropped from the benchmark's pooled rows:
there the scorer, not the model, would decide the result, and it would favour digit output.

usage: oracle_recall.py [--refs DIR] [--bed MANIFEST.json] [--bar 95] [--json out.json]
  --refs DIR       held-out label rows heldout_<lang>.jsonl (value_accuracy.py format). The spoken-form label text is
                   scored as the hypothesis against the caption (value_accuracy.span_outcomes): recall in context.
  --bed MANIFEST   a seam bed manifest {file: [{id, lang, label, values}]} (seam_harness.py format). Each sentence's
                   spoken-form label is read by the reader (common.readings_of) and its slots matched with
                   score_values.match_slots: recall of the exact seam-scoring path. The written caption is read the
                   same way (a perfect digits model); `both` = slots right in both, the scorable/ set of seam_harness.
Prints per language: spans / slots, recall %, 95% cluster-bootstrap CI, KEEP or DROP, and the most common failing
span kinds or slot types.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import value_accuracy as VA  # noqa: E402


def heldout_recall(refs_dir: Path) -> dict:
    refs = VA.load_refs(refs_dir, [])
    out = {}
    for lang, rows in refs.items():
        cl = collections.defaultdict(lambda: [0, 0])
        fails = collections.Counter()
        unscorable = 0
        for r in rows:
            for o in VA.span_outcomes(r, r["text"], lang):
                if not o["n_nums"]:
                    unscorable += 1
                    continue
                c = cl[C.cluster_of(r["id"], r.get("src", ""))]
                c[0] += o["value_right"]
                c[1] += 1
                if not o["value_right"]:
                    fails[o["kind"]] += 1
        h, n = sum(v[0] for v in cl.values()), sum(v[1] for v in cl.values())
        out[lang] = {"n": n, "unscorable": unscorable, "recall": C.pct(h, n),
                     "ci95": C.ratio_ci({k: tuple(v) for k, v in cl.items()}), "fails": dict(fails.most_common(5))}
    return out


def bed_recall(manifest: Path) -> dict:
    """per language: recall on the spoken-form labels (the words oracle), on the written captions (the digits oracle),
    and the share of slots right in both (what seam_harness.py keeps under scorable/)"""
    m = json.load(open(manifest, encoding="utf-8"))
    cl = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))
    fails = collections.defaultdict(collections.Counter)
    for f, sents in m.items():
        for s in sents:
            lang = s["lang"]
            vals = {k: str(v) for k, v in s["values"].items()}
            rl = C.SV.match_slots(vals, C.readings_of(s.get("label") or s["text"], lang))
            rc = C.SV.match_slots(vals, C.readings_of(s["text"], lang))
            for k in vals:
                c = cl[lang][f]
                c[0] += rl[k]
                c[1] += 1
                c[2] += rc[k]
                c[3] += rl[k] and rc[k]
                if not rl[k]:
                    fails[lang][C.SV.base_slot(k)] += 1
    out = {}
    for lang, d in cl.items():
        h, n = sum(v[0] for v in d.values()), sum(v[1] for v in d.values())
        out[lang] = {"n": n, "recall": C.pct(h, n), "ci95": C.ratio_ci({k: (v[0], v[1]) for k, v in d.items()}),
                     "caption_recall": C.pct(sum(v[2] for v in d.values()), n),
                     "both": C.pct(sum(v[3] for v in d.values()), n), "fails": dict(fails[lang].most_common(5))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--bed", type=Path)
    ap.add_argument("--bar", type=float, default=95.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {}
    for name, fn, src in (("heldout", heldout_recall, a.refs), ("seam_bed", bed_recall, a.bed)):
        if not src:
            continue
        r = fn(src)
        res[name] = r
        print(f"== {name}: reader recall on spoken-form labels (bar {a.bar:g}%)")
        for lang in sorted(r, key=lambda L: (C.ALL_LANGS.index(L) if L in C.ALL_LANGS else 99, L)):
            d = r[lang]
            keep = "KEEP" if (d["recall"] or 0) >= a.bar else "DROP"
            d["decision"] = keep
            extra = f" unscorable={d['unscorable']}" if "unscorable" in d else f" caption={d['caption_recall']}% both={d['both']}%"
            print(f"  {lang:3s} n={d['n']:5d} recall={d['recall']!s:>6s}% CI95={d['ci95']} {keep}{extra}  fails={d['fails']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    main()
