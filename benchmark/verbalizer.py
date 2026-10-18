#!/usr/bin/env python3
"""Baseline "stock + a deterministic public verbalizer": rewrite every digit number of a decode as the canonical words
of the pipeline's number grammar, the way a post-hoc ITN-in-reverse step would.

usage: verbalizer.py <in_dir> <out_dir> [--langs de,es,...]
  in_dir   decodes {id, hyp, ...} as heldout_<lang>.jsonl (and e22_numbers.jsonl for English)
  out_dir  the same files with `hyp` verbalized; the original stays in `hyp_raw`

Verbalizer: pipeline/ml_cands.py finds the number spans of the text (the same span finder that labels the training
captions) and the first candidate of each span (the CANONICAL reading of pipeline/ml_grammar_table.json) replaces the
digits. Languages: de es fr it nl pt (ml_cands) and ru uk (ml_cands through ml_slav). English uses pipeline/readings.py
(cardinals, years as pairs, decimals, percent, dollars). Any other language, and any span the grammar table has no entry
for, keeps its digits; the coverage line printed per language says how many numbers were rewritten.

The point of the baseline: the verbalizer always picks one reading. A speaker who said "treize cents" gets "mille trois
cents", and a phone number read in pairs gets digit by digit. Value accuracy stays near stock; form accuracy shows the
cost of guessing.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

if str(C.PIPELINE) not in sys.path:
    sys.path.insert(0, str(C.PIPELINE))

ML_LANGS = ("de", "es", "fr", "it", "nl", "pt", "ru", "uk")


def _en_number(tok: str) -> str:
    import readings as RD  # pipeline/readings.py (read only)

    t = tok.replace(",", "")
    if re.fullmatch(r"\d+\.\d+", t):
        a, b = t.split(".")
        return f"{RD.cardinal(int(a))} point {' '.join(RD.cardinal(int(c)) for c in b)}"
    if not t.isdigit():
        return tok
    n = int(t)
    if len(t) > 1 and t.startswith("0"):
        return " ".join(RD.cardinal(int(c)) for c in t)
    if 1100 <= n <= 2099 and "," not in tok:  # a bare 4-digit number in the year range reads as a year
        return RD.year(n)
    return RD.cardinal(n)


_EN_NUM = re.compile(r"(?<![\w.,$])(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%)?(?:\s+(thousand|million|billion|trillion)\b)?(?![\w])")


def verbalize_en(text: str) -> tuple:
    """English digits -> words: a stand-alone number (not inside a code such as "Q4" or "B2B"), with "$" before and
    "%" or a scale word after it."""
    count = [0, 0]

    def rep(m):
        count[1] += 1
        sym, num, frac, pc, scale = m.groups()
        w = _en_number(num + (frac or ""))
        if w == num + (frac or ""):
            return m.group(0)
        count[0] += 1
        if scale:
            w += " " + scale
        if sym == "$":
            w += " dollars"
        if pc:
            w += " percent"
        return w

    return _EN_NUM.sub(rep, text), count


def verbalize(text: str, lang: str) -> tuple:
    """(verbalized text, [numbers rewritten, digit numbers seen])"""
    if lang == "en":
        return verbalize_en(text)
    seen = len(re.findall(r"\d+", text))
    if lang not in ML_LANGS or not seen:
        return text, [0, seen]
    import ml_cands as M  # pipeline/ml_cands.py (read only)

    try:
        sp = [s for s in M.spans(text, lang) if s.cands]
        out = M.render(text, sp, [0] * len(sp)) if sp else text
    except Exception:  # the span finder rejects some shapes: keep the digits (counted as not rewritten)
        return text, [0, seen]
    return out, [seen - len(re.findall(r"\d+", out)), seen]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--langs", default="")
    a = ap.parse_args(argv)
    langs = {x for x in a.langs.split(",") if x}
    for p in sorted(a.in_dir.glob("*.jsonl")):
        m = re.match(r"heldout_([a-z]{2})$", p.stem)
        lang = m.group(1) if m else ("en" if p.stem.startswith("e22") else None)
        if lang is None or (langs and lang not in langs):
            continue
        rows, tot = [], [0, 0]
        for r in C.read_jsonl(p):
            out, c = verbalize(r.get("hyp", ""), lang)
            tot[0] += c[0]
            tot[1] += c[1]
            rows.append({**r, "hyp": out, "hyp_raw": r.get("hyp", "")})
        C.write_jsonl(a.out_dir / p.name, rows)
        print(f"{p.name}: {tot[0]} of {tot[1]} digit numbers rewritten")


if __name__ == "__main__":
    main()
