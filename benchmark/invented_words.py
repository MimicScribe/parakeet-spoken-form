#!/usr/bin/env python3
"""Invented words against an INDEPENDENT public lexicon: how often does a model write a token that is not a word of
the language? A wrapper around eval/quality/invented_words.py (imported, unchanged) with one difference that matters:
no model decode ever enters the lexicon. A lexicon built from stock / baseline decodes scores those models ~0 by
construction and calls any valid number word they never wrote (an inflection, a split compound) invented.

usage: invented_words.py --refs DIR --arm stock=DIR [--arm B=DIR ...] [--langs de,es,...] [--cache DIR]
                         [--no-refs] [--wordfreq N] [--json out.json]
  --arm N=DIR    decodes DIR/heldout_<lang>.jsonl {id, hyp} (value_accuracy.py inputs), or N=cases.jsonl:long for a
                 seam_harness.py cases file (the stitched reading)
  --refs DIR     held-out label rows (their spoken-form `text` is the reference; with --no-refs it stays out of the
                 lexicon too)
Lexicon per language (sources; a token is invented when it is in none):
  train     FLEURS train-split transcripts (raw and normalized), downloaded once into --cache (CC BY 4.0);
  wordfreq  optional: the N most frequent words of the `wordfreq` package (pip install wordfreq; CC BY-SA 4.0 data);
  ref       the benchmark's own spoken-form labels (human/label text, not model output; off with --no-refs);
  morpheme  every number morpheme / form of the public reader's tables (number_outcomes: en de es fr it nl pt core
            grammar; the other languages' form tables in pipeline/ml_multi.py and ml_slav.py), so a correctly split
            number ("hundert", "twintig") is always a word; fused compounds the reader segments wholly are accepted;
  function  percent / decimal / currency words (eval/quality/invented_words.ALWAYS_VALID).
Metrics (eval/quality/invented_words.py): sentences with an invented token, invented tokens per words-written
number, English number-word bleed into other languages, garbles, invented rate by token length.
"""
import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import invented_words as IW  # noqa: E402  (eval/quality, read only)

FLEURS = {"en": "en_us", "bg": "bg_bg", "cs": "cs_cz", "da": "da_dk", "de": "de_de", "el": "el_gr", "es": "es_419",
          "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "hr": "hr_hr", "hu": "hu_hu", "it": "it_it", "lt": "lt_lt",
          "lv": "lv_lv", "mt": "mt_mt", "nl": "nl_nl", "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "ru": "ru_ru",
          "sk": "sk_sk", "sl": "sl_si", "sv": "sv_se", "uk": "uk_ua"}
FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{code}/train.tsv"


def fleurs_text(lang: str, cache: Path) -> list:
    """FLEURS train transcripts of a language (raw + normalized columns), cached as a TSV"""
    code = FLEURS.get(lang)
    if not code:
        return []
    p = cache / f"fleurs_train_{code}.tsv"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FLEURS_URL.format(code=code), timeout=120) as r:
            p.write_bytes(r.read())
    out = []
    with open(p, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            out += [x for x in row[2:4] if x]
    return out


def wordfreq_words(lang: str, n: int) -> list:
    try:
        import wordfreq
    except ImportError:
        raise SystemExit("--wordfreq needs the wordfreq package (pip install wordfreq)")
    if lang not in wordfreq.available_languages(wordlist="best"):
        return []  # no list of its own (et, hr, mt): wordfreq would silently fall back to a neighbour language
    return wordfreq.top_n_list(lang, n)


def ext_morphemes(lang: str) -> set:
    """number forms of the 18 non-core languages, from the reader's own tables (empty when not available)"""
    mm = C.N._ext_reader(lang)
    if mm is None:
        return set()
    lx = mm.lexicon(lang)
    return set(getattr(lx, "forms", ())) | set(getattr(lx, "glue", ()))


def build_lexicon(langs, refs: dict, cache: Path, with_refs: bool = True, wf: int = 0) -> "IW.Lexicon":
    lx = IW.Lexicon()
    for lang in langs:
        for t in fleurs_text(lang, cache):
            lx.add(lang, IW.tokens(t), "train")
        if wf:
            lx.add(lang, [w.lower() for w in wordfreq_words(lang, wf)], "train")
        if with_refs:
            for r in refs.get(lang, []):
                lx.add(lang, IW.tokens(r["text"]), "ref")
        lx.add_morphemes(lang)
        for m in ext_morphemes(lang):
            lx.src[lang][m].add("morpheme")
            lx.src[lang][C.N._fold(m)].add("morpheme")
        lx.add_function_words(lang)
    return lx


def load_arm(spec: str, refs: dict, langs) -> list:
    """[(id, lang, ref, hyp)]"""
    path, _, field = spec.partition(":")
    p = Path(path)
    if p.is_dir():
        rows = []
        for lang in langs:
            f = p / f"heldout_{lang}.jsonl"
            if not f.exists():
                continue
            ref = {r["id"]: r["text"] for r in refs.get(lang, [])}
            rows += [(r["id"], lang, ref.get(r["id"]), r.get("hyp", "")) for r in C.read_jsonl(f)]
        return rows
    return [(r.get("file", r["id"]) + ":" + r["id"], r["lang"], None, r[field or "long"]) for r in C.read_jsonl(p)
            if r["lang"] in langs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR or NAME=cases.jsonl[:field]")
    ap.add_argument("--langs", default="de,es,fr,it,nl,pt")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "spoken_form_bench")
    ap.add_argument("--no-refs", action="store_true")
    ap.add_argument("--wordfreq", type=int, default=0, help="add the N most frequent words of the wordfreq package")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    langs = [x for x in a.langs.split(",") if x]
    refs = {}
    if a.refs:
        for p in sorted(a.refs.glob("heldout_*.jsonl")):
            refs[p.stem.split("_", 1)[1]] = C.read_jsonl(p)
    lx = build_lexicon(langs, refs, a.cache, not a.no_refs, a.wordfreq)
    results = {}
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        rows = load_arm(rest, refs, langs)
        results[name] = IW.score_arm(rows, lx, name, False, True, langs)
    print(IW.report(results, langs))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    main()
