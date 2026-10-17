"""Regional / colloquial readings (ml_cands VARIETY, 2026-10-16): hand cases + the additive guarantee.

    python3 test_ml_cands_regional.py                      # hand cases
    python3 test_ml_cands_regional.py captions.jsonl ...   # + every caption ({"text", "lang"} per line): each span's
                                                           #   spans() readings are an exact PREFIX of its
                                                           #   spans_regional() readings (same spans, same cands[0],
                                                           #   same order); spans() itself is unchanged (VARIETY off)
"""

import json
import sys
from collections import Counter

import ml_cands as M

CASES = [  # (lang, caption, reading that must be among the first span's readings)
    ("fr", "Il a 70 ans.", "septante"),
    ("fr", "Il a 71 ans.", "septante et un"),
    ("fr", "Il a 77 ans.", "septante sept"),
    ("fr", "Il a 80 ans.", "huitante"),
    ("fr", "Il a 81 ans.", "huitante et un"),
    ("fr", "Il a 90 ans.", "nonante"),
    ("fr", "Il a 91 ans.", "nonante et un"),
    ("fr", "Il a 96 ans.", "nonante six"),
    ("fr", "En 1975 déjà.", "mille neuf cent septante cinq"),
    ("fr", "En 1991 déjà.", "mille neuf cent nonante et un"),
    ("fr", "Ça coûte 2,75 €.", "deux euros septante cinq"),
    ("fr", "Rendez-vous à 14:45.", "trois heures moins le quart"),
    ("fr", "Rendez-vous à 12:30.", "midi et demi"),
    ("pt", "São 1.200.000.000 € por ano.", "mil e duzentos milhões de euros"),
    ("pt", "Às 14:45 começa.", "três menos um quarto"),
    ("pt", "Às 14:45 começa.", "um quarto para as três"),
    ("pt", "Às 12:30 começa.", "meio-dia e meia"),
    ("de", "Am 1.1.2020 war es kalt.", " Jänner"),
    ("de", "Um 14:45 beginnt es.", "Viertel vor drei"),
    ("de", "Um 14:45 beginnt es.", "dreiviertel drei"),
    ("de", "Um 9:30 beginnt es.", "halb zehn"),
    ("de", "Um 9:15 beginnt es.", "viertel zehn"),
    ("de", "Um 9:40 beginnt es.", "zwanzig vor zehn"),
    ("nl", "Om 14:45 begint het.", "kwart voor drie"),
    ("nl", "Om 9:30 begint het.", "half tien"),
    ("nl", "Om 9:15 begint het.", "kwart na negen"),
    ("nl", "Om 9:20 begint het.", "tien voor half tien"),
    ("es", "A las 14:45 empieza.", "tres menos cuarto"),
    ("es", "A las 14:45 empieza.", "un cuarto para las tres"),
    ("es", "A las 12:50 empieza.", "diez para la una"),
    ("it", "Alle 14:45 inizia.", "tre meno un quarto"),
    ("it", "Alle 12:30 inizia.", "mezzogiorno e mezza"),
    # law_abbr (on by default): the abbreviation is read as its word, never left in the label
    ("de", "In § 1 wird die Angabe ersetzt.", "=Paragraf eins"),
    ("de", "Nach Art. 12 der Verordnung.", "=Artikel zwölf"),
    ("de", "Gemäß §§ 3 und 4.", "=Paragrafen drei"),
    ("fr", "Selon l'art. 5 du traité.", "=article cinq"),
    ("es", "El art. 155 de la Constitución.", "=artículo ciento cincuenta y cinco"),
]
NEVER = [  # (lang, caption, reading that must NOT appear: two numbers joined into one, a changed value)
    ("fr", "Appelez le 06 60 10 20 30.", "septante"),  # "60" + "10" are two phone groups, not 70
    ("fr", "Il a 60 ans.", "septante"),
    ("pt", "São 1.000.000.000 de euros.", "bilião"),  # 10^9 is "mil milhões"; "bilião" is 10^12 in Portugal
]


def _all(lang, text):
    return [c for s in M.spans_regional(text, lang) for c in s.cands]


def hand() -> int:
    bad = 0
    for lang, text, want in CASES:
        sp = M.spans_regional(text, lang)
        if want.startswith("="):
            if not sp or sp[0].cands[:1] != [want[1:]]:
                print("NOT CANONICAL", lang, text, want, sp[0].cands[:2] if sp else None)
                bad += 1
            continue
        if not sp or want not in sp[0].cands and want not in _all(lang, text):
            print("MISSING", lang, text, want, sp[0].cands if sp else None)
            bad += 1
    for lang, text, no in NEVER:
        if any(no in c.split() for c in _all(lang, text)):
            print("UNSAFE", lang, text, no)
            bad += 1
    return bad


def prefix_check(paths) -> int:
    st, bad = Counter(), 0
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            lang, text = r.get("lang"), r["text"]
            if lang not in M.LANGS:
                continue
            old = M.spans(text, lang)
            new = M.spans_regional(text, lang)
            st["captions"] += 1
            if [(s.start, s.end, s.kind) for s in old] != [(s.start, s.end, s.kind) for s in new]:
                bad += 1
                print("SPANS DIFFER", lang, text)
                continue
            for a, b in zip(old, new):
                st["spans"] += 1
                if b.cands[:len(a.cands)] != a.cands:
                    bad += 1
                    print("NOT A PREFIX", lang, text, a.text)
                elif len(b.cands) > len(a.cands):
                    st[f"gained {lang} {b.kind}"] += 1
    print(json.dumps(dict(sorted(st.items())), ensure_ascii=False))
    return bad


if __name__ == "__main__":
    n = hand()
    if sys.argv[1:]:
        n += prefix_check(sys.argv[1:])
    print("FAIL" if n else "OK", n)
    sys.exit(1 if n else 0)
