#!/usr/bin/env python3
"""Invented-word metric: how often does a model write a word that is not a word of the language?

Method (reference-free on the hypothesis side; the lexicon is the only judge):
  1. Tokens. A hypothesis is lower-cased, punctuation and "<unk>" removed, split on whitespace. A token that carries a digit
     is not a word and is never counted (digit strings are scored by number_outcomes, not here).
  2. Lexicon per language = every token of: the reference texts, the decodes of the LEXICON ARMS (the stock model and the
     base model: their words are real words by construction), optional training-text files (`--train-dir`, any
     `*_<lang>.jsonl` with a "text" field; with the split-compound convention these hold the number morphemes), and every
     number morpheme of the language's number reader (number_outcomes), folded and unfolded. A SPLIT MORPHEME ("hundert",
     "und", "twintig", "venti") is therefore always a valid word. Whole compounds are valid only when seen somewhere above
     (or with `--accept-compounds`, when the reader segments the token wholly into number morphemes: a real word, maybe the
     wrong value).
  3. A token is INVENTED when it is in no lexicon of its language. "Sentence with an invented token" = at least one.
  4. Per arm and language: sentences with an invented token; invented tokens; converted numbers = hypothesis number chains
     written as words (number_groups, form "words": counted from the hypothesis alone) and, when a reference exists,
     reference numbers whose hypothesis form is "converted" (words AND value right, number_outcomes.form_outcomes);
     invented tokens per converted number (both denominators); invented rate by token length (invented tokens / tokens in
     the length bucket: <=3, 4-5, 6-8, 9-11, 12-15, 16+ letters).
  5. A lexicon arm is judged against a lexicon that still contains its own words, so it scores ~0 by construction. `--loo`
     adds the leave-one-out column: a token only that arm (and no reference / training / morpheme source) ever wrote.

Inputs (`--arm NAME=PATH[:FIELD]`, repeatable):
  parity decodes  {"id", "hyp"} per line; reference text and language come from `--manifest` ({"id","lang","text"}).
  app-path cases  {"file","lang","written","long","single"} per line; FIELD picks the reading (default long); the reference is
                  "written"; the row id is "file".
Usage: invented_words.py --manifest M.jsonl --arm stock=ultra.jsonl --arm B=b.jsonl --arm a1=a1.jsonl \
           --lexicon-arm stock --lexicon-arm B [--lexicon-file PATH[:FIELD]] [--train-dir DIR] [--langs de,es,fr,it,nl,pt] [--json out.json] [--loo]
Tests: python3 -m unittest eval/quality/test_invented_words.py
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N  # noqa: E402

BUCKETS = ((0, 3), (4, 5), (6, 8), (9, 11), (12, 15), (16, 10**6))
WAVE1 = ("de", "es", "fr", "it", "nl", "pt")


def tokens(text: str) -> list:
    """Word tokens of a text: lower-cased, punctuation and <unk> dropped, tokens with a digit left out."""
    t = re.sub(r"[^\w\s]|_", " ", str(text).lower().replace("<unk>", " "))
    return [w for w in t.split() if not re.search(r"\d", w)]


BUCKET_NAMES = ["<=3", "4-5", "6-8", "9-11", "12-15", "16+"]


def bucket(n: int) -> str:
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= n <= hi:
            return name
    return BUCKET_NAMES[-1]


def morphemes(lang: str) -> set:
    """Every number morpheme / connector of the language (the split-compound pieces), folded and unfolded."""
    lex = N._LEX.get(lang) or {}
    out = set(lex)
    conn = N._CONN.get(lang)
    if conn:
        out.add(conn)
    return out


class Lexicon:
    def __init__(self):
        self.src = collections.defaultdict(lambda: collections.defaultdict(set))  # lang -> token -> {sources}

    def add(self, lang: str, toks, source: str):
        d = self.src[lang]
        for t in toks:
            d[t].add(source)

    def add_morphemes(self, lang: str):
        for m in morphemes(lang):
            self.src[lang][m].add("morpheme")
            self.src[lang][N._fold(m)].add("morpheme")

    def known(self, lang: str, tok: str, exclude: str = None, accept_compounds: bool = False) -> bool:
        s = self.src[lang].get(tok) or self.src[lang].get(N._fold(tok))
        if s and (exclude is None or s - {"arm:" + exclude}):
            return True
        if accept_compounds and lang in N.WORD_LANGS:
            seg = N._segment(lang, N._fold(tok))
            return bool(seg) and any(m[1] != "C" for m in seg)
        return False


def build_lexicon(rows_by_arm: dict, lexicon_arms, train_dir: str = None, extra_rows=(), with_morphemes: bool = True) -> Lexicon:
    """rows_by_arm: arm -> [(id, lang, ref, hyp)]. References of every arm's rows and the decodes of `lexicon_arms` go in."""
    lx = Lexicon()
    langs = set()
    for arm, rows in rows_by_arm.items():
        for _i, lang, ref, hyp in rows:
            langs.add(lang)
            if ref:
                lx.add(lang, tokens(ref), "ref")
            if arm in lexicon_arms:
                lx.add(lang, tokens(hyp), "arm:" + arm)
    for _i, lang, ref, hyp in extra_rows:  # other decodes of real words (e.g. the same models read through another engine)
        langs.add(lang)
        lx.add(lang, tokens(hyp), "extra")
    for lang in langs if with_morphemes else ():
        lx.add_morphemes(lang)
    if train_dir:
        for f in sorted(glob.glob(os.path.join(train_dir, "*.jsonl"))):
            m = re.search(r"_([a-z]{2,3})\.jsonl$", f)
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                lang = r.get("lang") or (m.group(1) if m else None)
                if lang and "text" in r:
                    lx.add(lang, tokens(r["text"]), "train")
    return lx


def load_arm(path: str, manifest: dict = None, field: str = "long") -> list:
    """[(id, lang, ref, hyp)] from a parity decode file or an app-path cases file."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "hyp" in r:
            m = (manifest or {}).get(r["id"], {})
            out.append((r["id"], m.get("lang") or r.get("lang"), m.get("text") or r.get("ref"), r["hyp"]))
        elif field in r:
            out.append((r.get("file") or r["id"], r["lang"], r.get("written"), r[field]))
    return out


def score_arm(rows: list, lx: Lexicon, arm: str = None, loo: bool = False, accept_compounds: bool = False,
              langs=None) -> dict:
    """Per language and pooled ("all"): see the module docstring. Returns {lang: stats}."""
    per = collections.defaultdict(lambda: collections.Counter())
    bl = collections.defaultdict(lambda: collections.Counter())
    examples = collections.defaultdict(collections.Counter)
    for _i, lang, ref, hyp in rows:
        if langs and lang not in langs:
            continue
        c = per[lang]
        c["sentences"] += 1
        toks = tokens(hyp)
        inv = [t for t in toks if not lx.known(lang, t, accept_compounds=accept_compounds)]
        inv_loo = [t for t in toks if not lx.known(lang, t, exclude=arm, accept_compounds=accept_compounds)] if loo else []
        c["tokens"] += len(toks)
        c["sent_invented"] += bool(inv)
        c["invented"] += len(inv)
        c["sent_invented_loo"] += bool(inv_loo)
        c["invented_loo"] += len(inv_loo)
        for t in toks:
            b = bucket(len(t))
            bl[lang][b + ":n"] += 1
        for t in inv:
            bl[lang][bucket(len(t)) + ":inv"] += 1
            examples[lang][t] += 1
        words = [g for g in N.number_groups(hyp, lang) if g["form"] == "words"] if lang in N.WORD_LANGS else []
        c["words_chains"] += len(words)
        if ref and lang in N.WORD_LANGS:
            rows_f = N.form_outcomes(ref, hyp, lang)
            c["ref_numbers"] += len(rows_f)
            c["converted"] += sum(r["form"] == "converted" for r in rows_f)
            c["converted_old"] += sum(r["form_old"] == "converted" for r in rows_f)
    out = {}
    pooled, pbl = collections.Counter(), collections.Counter()
    for lang, c in per.items():
        out[lang] = _finish(c, bl[lang], examples[lang])
        pooled.update(c)
        pbl.update(bl[lang])
    out["all"] = _finish(pooled, pbl, sum(examples.values(), collections.Counter()))
    return out


def _rate(a, b):
    return round(a / b, 4) if b else None


def _finish(c, bl, ex) -> dict:
    d = dict(c)
    d["sent_invented_pct"] = round(100 * c["sent_invented"] / c["sentences"], 1) if c["sentences"] else None
    d["invented_per_words_chain"] = _rate(c["invented"], c["words_chains"])
    d["invented_per_converted"] = _rate(c["invented"], c["converted"])
    d["rate_by_length"] = {b: {"n": bl[b + ":n"], "invented": bl[b + ":inv"], "rate": _rate(bl[b + ":inv"], bl[b + ":n"])}
                           for b in BUCKET_NAMES}
    d["top_invented"] = ex.most_common(8)
    return d


def report(results: dict, langs) -> str:
    lines = []
    for lang in list(langs) + ["all"]:
        lines.append(f"== {lang} ==")
        lines.append(f"{'arm':14s} {'n':>4s} {'sent-inv':>8s} {'%':>6s} {'inv':>5s} {'words-chains':>12s} {'conv(right)':>11s} {'old conv':>8s}"
                     f" {'inv/chain':>9s} {'inv/conv':>8s}  invented rate by length " + " ".join(BUCKET_NAMES))
        for arm, res in results.items():
            d = res.get(lang)
            if not d:
                continue
            rl = " ".join(f"{(d['rate_by_length'][b]['rate'] or 0) * 100:5.1f}" for b in BUCKET_NAMES)
            lines.append(f"{arm:14s} {d['sentences']:4d} {d['sent_invented']:8d} {d['sent_invented_pct']!s:>6s} {d['invented']:5d} "
                         f"{d['words_chains']:12d} {d.get('converted', 0):11d} {d.get('converted_old', 0):8d} "
                         f"{d['invented_per_words_chain']!s:>9s} {d['invented_per_converted']!s:>8s}  {rl}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest")
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH[:FIELD]")
    ap.add_argument("--lexicon-arm", action="append", default=[])
    ap.add_argument("--lexicon-file", action="append", default=[], help="PATH[:FIELD]: decodes whose words join the lexicon, not scored")
    ap.add_argument("--train-dir")
    ap.add_argument("--langs", default=",".join(WAVE1))
    ap.add_argument("--accept-compounds", action="store_true")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--no-morphemes", action="store_true", help="do not add the number morphemes to the lexicon (the first-pass method)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    man = {}
    if a.manifest:
        for line in open(a.manifest, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                man[r["id"]] = r
    rows = {}
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        path, _, field = rest.partition(":")
        rows[name] = load_arm(path, man, field or "long")
    langs = [x for x in a.langs.split(",") if x]
    extra = [r for spec in a.lexicon_file for r in load_arm(spec.partition(":")[0], man, spec.partition(":")[2] or "long")]
    lx = build_lexicon(rows, set(a.lexicon_arm), a.train_dir, extra, not a.no_morphemes)
    results = {n: score_arm(r, lx, n, a.loo, a.accept_compounds, langs) for n, r in rows.items()}
    print(report(results, langs))
    if a.json:
        json.dump(results, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
