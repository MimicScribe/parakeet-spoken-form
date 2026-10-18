#!/usr/bin/env python3
"""False-conversion gate: on NUMBER-FREE speech, how often does a decode write a number anyway?

Counts, per arm and language, over a number-free manifest (references hold no digit and no number word, except the
look-alikes ml_balance allows: articles "ein" / "un" / "een", and homophones in their non-number sense):
  digits     hypothesis tokens carrying a digit or a number symbol (% € $ £ ° ...): always spurious here
  numwords   hypothesis number words (ml_balance.number_lexicon + its compound rule, articles excluded) that the
             reference does not have at all or has fewer times (multiset difference per row)
  total      digits + numwords; reported per 1,000 REFERENCE words (the same denominator for every arm), plus the share
             of rows with any false conversion.
With --base NAME (normally stock), each other arm also gets the paired difference vs the base per 1,000 words and a
95% bootstrap interval over manifest rows (rows resampled; both arms on the same rows; seeded).

    python3 false_conversion.py --manifest numfree_de.jsonl --lang de --arm stock=hyps_stock.jsonl \
        --arm r9=hyps_r9.jsonl --base stock [--json out.json]

Manifest rows: {"id", "text"} (+ optional "ref": used instead of "text"). Arm files: {"id", "hyp"} per line (the
hyps/<run>/<set>.jsonl files written by modal_app.transcribe). Rows an arm lacks are counted as missing and left out of
that arm's numbers (the paired difference uses the rows both arms have). Stdlib only; imports pipeline/ml_balance.
"""

import argparse
import collections
import json
import os
import random
import re
import sys

PIPE = os.environ.get("SPOKEN_FORM_PIPELINE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))
sys.path.insert(0, PIPE)
import ml_balance as B  # noqa: E402

DIGIT = re.compile(r"[0-9%€$£¥°‰½¼¾⅓⅔²³¹]")


def _toks(text: str) -> list:
    return (text or "").lower().replace("’", "'").split()


def _words(text: str) -> list:
    """Letter tokens (ml_balance.norm_tokens), hyphen parts added ("dix-sept" -> also "dix", "sept")."""
    out = []
    for t in B.norm_tokens(text or ""):
        out.append(t)
        if "-" in t:
            out += [p for p in t.split("-") if p]
    return out


def numword_tokens(text: str, lang: str, lex: set) -> collections.Counter:
    arts = B.ARTICLES.get(lang, set())
    return collections.Counter(t for t in _words(text) if t not in arts and (t in lex or B._compound_num(t, lang)))


def row_counts(ref: str, hyp: str, lang: str, lex: set) -> dict:
    digits = sum(bool(DIGIT.search(t)) for t in _toks(hyp))
    extra = numword_tokens(hyp, lang, lex) - numword_tokens(ref, lang, lex)
    return {"digits": digits, "numwords": sum(extra.values()), "ref_words": len(_toks(ref)),
            "examples": sorted(extra)[:5]}


def per_k(n: int, w: int) -> float:
    return round(1000.0 * n / max(w, 1), 3)


def score(manifest: list, arms: dict, lang: str, base: str | None = None, boot: int = 2000, seed: int = 1) -> dict:
    lex = B.number_lexicon(lang)
    refs = {r["id"]: r.get("ref") or r["text"] for r in manifest}
    out = {"lang": lang, "rows": len(refs), "ref_words": sum(len(_toks(t)) for t in refs.values()), "arms": {}}
    per_row = {}
    for name, hyps in arms.items():
        rc = {i: row_counts(refs[i], hyps[i], lang, lex) for i in refs if i in hyps}
        per_row[name] = rc
        w = sum(c["ref_words"] for c in rc.values())
        d = sum(c["digits"] for c in rc.values())
        n = sum(c["numwords"] for c in rc.values())
        ex = collections.Counter(e for c in rc.values() for e in c["examples"])
        bad = [i for i, c in rc.items() if c["digits"] or c["numwords"]]
        out["arms"][name] = {"rows": len(rc), "missing": len(refs) - len(rc), "ref_words": w, "digits": d,
                             "numwords": n, "digits_per_1k": per_k(d, w), "numwords_per_1k": per_k(n, w),
                             "total_per_1k": per_k(d + n, w), "rows_with_false": len(bad),
                             "rows_with_false_share": round(len(bad) / max(len(rc), 1), 4),
                             "top_numwords": ex.most_common(8),
                             "examples": [{"id": i, "ref": refs[i][:120], "hyp": hyps[i][:120]} for i in bad[:5]]}
    if base and base in per_row:
        rng = random.Random(seed)
        for name, rc in per_row.items():
            if name == base:
                continue
            ids = sorted(set(rc) & set(per_row[base]))
            if not ids:
                continue
            f = lambda c: c["digits"] + c["numwords"]  # noqa: E731
            da = [f(rc[i]) - f(per_row[base][i]) for i in ids]
            ws = [rc[i]["ref_words"] for i in ids]
            point = 1000.0 * sum(da) / max(sum(ws), 1)
            bs = []
            for _ in range(boot):
                k = [rng.randrange(len(ids)) for _ in ids]
                bs.append(1000.0 * sum(da[j] for j in k) / max(sum(ws[j] for j in k), 1))
            bs.sort()
            out["arms"][name]["vs_base"] = {"base": base, "rows": len(ids), "diff_per_1k": round(point, 3),
                                            "ci95": [round(bs[int(0.025 * boot)], 3), round(bs[int(0.975 * boot) - 1], 3)],
                                            "rows_worse": sum(x > 0 for x in da), "rows_better": sum(x < 0 for x in da)}
    return out


def _read(path: str, field: str = "hyp") -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r.get(field) or ""
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH (an {id, hyp} jsonl)")
    ap.add_argument("--base", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    man = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    arms = {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arms[name] = _read(path)
    res = score(man, arms, a.lang, a.base, a.boot)
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
    print(f"{a.lang}: {res['rows']} rows, {res['ref_words']} reference words")
    print(f"{'arm':24} {'rows':>5} {'dig/1k':>7} {'nw/1k':>7} {'tot/1k':>7} {'rows%':>6}  vs base (95% CI)")
    for name, r in res["arms"].items():
        vb = r.get("vs_base")
        tail = f"  {vb['diff_per_1k']:+.2f} [{vb['ci95'][0]:+.2f}, {vb['ci95'][1]:+.2f}]" if vb else ""
        print(f"{name:24} {r['rows']:5d} {r['digits_per_1k']:7.2f} {r['numwords_per_1k']:7.2f} {r['total_per_1k']:7.2f} "
              f"{100 * r['rows_with_false_share']:5.1f}%{tail}")
    return res


if __name__ == "__main__":
    main()
