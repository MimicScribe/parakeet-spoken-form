"""Build ml_multi_lex.json: every written form of each numeral morpheme (cardinal and ordinal, all cases / genders /
numbers, and the stems used inside compound numerals) for the 16 languages of ml_multi, plus the small closed-class
words a spoken number can contain (decimal separator, fraction denominators, connectors, half).

Forms are drafted by an LLM (public linguistic knowledge), then checked by `check()`: a form listed for two different
values is dropped from both unless both are the same numeral, and the nominative cardinals 0-100 must parse back to
their values with the finished reader (test_ml_multi.py). The reader only reads VALUES; a missing oblique form makes a
span unreadable (masked), it never changes a value.

    GEMINI_API_KEY=... python ml_multi_lexgen.py --langs pl,ro,sv
"""
import argparse
import json
import os
import re
import sys
import unicodedata

import ml_llm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ml_multi_lex.json")
NAMES = {"bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish", "et": "Estonian", "fi": "Finnish",
         "el": "Greek", "hu": "Hungarian", "lv": "Latvian", "lt": "Lithuanian", "mt": "Maltese", "pl": "Polish",
         "ro": "Romanian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish"}
VALUES = list(range(0, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100)) + [1000, 10**6, 10**9, 10**12]

SYSTEM = ("You are a careful descriptive linguist. You write complete morphological tables of numerals as they are "
          "spelled in standard orthography (with all diacritics), lowercase. You never invent forms; you include every "
          "form that occurs in real speech, including colloquial ones.")

PROMPT = """Language: {name}.

For each numeral value below give EVERY written form that the numeral word (or the morpheme used inside a compound numeral) takes in {name}:
- cardinal: all grammatical cases, genders, numbers (singular/plural), animacy/personal forms, definite/indefinite forms, collective forms, and the bound stems or combining forms used inside compound numerals (for example the "-on" stem of Hungarian "huszon", the Swedish "tjugo" in "tjugofem", Finnish "kymmentä" in "kaksikymmentä", Czech/Slovak "set"/"sta"/"stě"/"sto" for hundreds, Polish "dwieście"/"trzysta" for whole hundreds).
- ordinal: all forms of the ordinal adjective (every case, gender, number, definiteness) and the bound ordinal stems used in compound ordinals.
If {name} writes a whole hundred (200, 300, ...) as one word, list those words under that value; if it uses a multiplier ("two hundreds"), list the multiplier word forms under 100 and leave 200-900 as an empty list.
Values: {values}

Also give:
- "decimal": words read for the decimal separator (e.g. "comma", "point", or the "whole" word of "three whole five tenths"),
- "fraction": map from "10", "100", "1000" to all forms of the words "tenth(s)", "hundredth(s)", "thousandth(s)" used when reading decimals,
- "connector": words that can stand INSIDE one spoken number (e.g. Romanian "și", Danish "og", Bulgarian "и", Maltese "u"; empty list if none),
- "half": all forms of "half" / "and a half" words,
- "minus": words for negative numbers.

Answer as JSON: {{"cardinal": {{"<value>": [forms]}}, "ordinal": {{"<value>": [forms]}}, "decimal": [..], "fraction": {{"10": [..], "100": [..], "1000": [..]}}, "connector": [..], "half": [..], "minus": [..]}}"""

# ------------------------------------------------------------------ hand overrides
# Forms the drafts miss or misfile, written by hand. check() merges them into the raw union before settling
# ambiguous forms: an override form is authoritative (it is never dropped as ambiguous and it leaves any other
# value's list), so a base form such as Lithuanian "trys" cannot be lost to the ambiguity rule.
#   EXTRA_CARDINAL / EXTRA_ORDINAL  {lang: {value: [forms]}}
#   REPLACE_ORDINAL                 {lang: [values]} whose drafted ordinal list is discarded first
#   REMOVE_FORMS                    {lang: {kind: {value: [forms]}}} forms deleted from a list (kept out of raw too)
def _sl_adj(stem: str) -> list:
    return [stem + e for e in ("i", "a", "o", "e", "ega", "emu", "em", "im", "ih", "ima", "imi")]


def _fi_teen_ordinals() -> dict:
    """yhdestoista ... yhdeksästoista, nominative and inflected: both parts take the case ("yhdennentoista")."""
    # unit: (nominative, weak-stem base of the other cases, partitive, essive, illative, front vowel harmony)
    u = {1: ("yhdes", "yhdenne", "yhdettä", "yhdentenä", "yhdenteen", True),
         2: ("kahdes", "kahdenne", "kahdetta", "kahdentena", "kahdenteen", False),
         3: ("kolmas", "kolmanne", "kolmatta", "kolmantena", "kolmanteen", False),
         4: ("neljäs", "neljänne", "neljättä", "neljäntenä", "neljänteen", True),
         5: ("viides", "viidenne", "viidettä", "viidentenä", "viidenteen", True),
         6: ("kuudes", "kuudenne", "kuudetta", "kuudentena", "kuudenteen", False),
         7: ("seitsemäs", "seitsemänne", "seitsemättä", "seitsemäntenä", "seitsemänteen", True),
         8: ("kahdeksas", "kahdeksanne", "kahdeksatta", "kahdeksantena", "kahdeksanteen", False),
         9: ("yhdeksäs", "yhdeksänne", "yhdeksättä", "yhdeksäntenä", "yhdeksänteen", True)}
    out = {}
    for n, (nom, stem, part, ess, ill, front) in u.items():
        a = "ä" if front else "a"
        forms = [nom, stem + "n", part, ess, ill] + [stem + e for e in ("ss" + a, "st" + a, "ll" + a, "lt" + a, "lle", "ksi")]
        out[10 + n] = [f + "toista" for f in forms]
    return out


EXTRA_CARDINAL = {
    "lt": {3: ["trys"]},
    "lv": {100: ["simti"]},
    "mt": {2000: ["elfejn"], 1000: ["elef"], 3: ["tliet"], 4: ["erba'"], 5: ["ħames"], 6: ["sitt"], 7: ["seba'"],
           8: ["tmien"], 9: ["disa'"], 11: ["ħdax"], 12: ["tnax"], 13: ["tlettax"], 14: ["erbatax"], 15: ["ħmistax"],
           16: ["sittax"], 17: ["sbatax"], 18: ["tmintax"], 19: ["dsatax"]},
    "sv": {40: ["förtio"]},
    "da": {60: ["treds"]},
    "sl": {0: ["nič", "ničla", "ničlo", "ničle"], 1: ["en", "ena", "eno", "enega", "enemu", "enem", "enim", "ene"],
           2: ["dva", "dve", "dveh", "dvema", "dvem"], 3: ["tri", "trije", "treh", "trem", "tremi"],
           4: ["štiri", "štirje", "štirih", "štirim", "štirimi"], 5: ["pet", "petih", "petim", "petimi"],
           6: ["šest", "šestih", "šestim", "šestimi"], 7: ["sedem", "sedmih", "sedmim", "sedmimi"],
           60: ["šestdeset", "šestdesetih", "šestdesetim", "šestdesetimi"]},
}
EXTRA_ORDINAL = {
    "sl": {1: _sl_adj("prv"), 2: _sl_adj("drug"), 3: _sl_adj("tretj"), 4: _sl_adj("četrt"), 5: _sl_adj("pet"),
           6: _sl_adj("šest"), 7: _sl_adj("sedm"), 60: _sl_adj("šestdeset")},
    "fi": _fi_teen_ordinals(),
    "mt": {4: ["raba'"], 7: ["seba'"], 9: ["disa'"]},
}
REPLACE_ORDINAL = {"fi": list(range(11, 20))}
# the partitive "toista" ("other") is also the teen suffix: stored as ordinal 2 it made "viidestoista" read [5, 2]
REMOVE_FORMS = {"fi": {"ordinal": {2: ["toista"]}}}


def clean_form(w: str):
    """A written numeral form or None: NBSP-like spaces become one space, anything with a control character, a format
    character (soft hyphen), a template marker ("<case>"), a digit, a quote or other symbol is junk."""
    w = unicodedata.normalize("NFC", str(w))
    w = re.sub(r"[\s\u00a0\u202f\u2009]+", " ", w).strip().lower()
    if not w:
        return None
    for c in w:
        if not (c.isalpha() or unicodedata.category(c) == "Mn" or c in "'’- "):
            return None
    parts = w.split(" ")
    if any(not re.search(r"[^\W\d_]", p) for p in parts):
        return None
    return w


def gen(lang: str, samples: int = 3) -> dict:
    """Union of `samples` drafts (one at temperature 0, the rest at 0.7): one draft misses forms, the union rarely
    does. `votes` keeps how many drafts gave each (form, value) so check() can settle a form claimed twice.
    A value chunk that no draft delivered raises: a silent hole in 0-100 would make the reader lose those numbers."""
    out, votes = {}, {}
    chunks = [VALUES[k:k + 8] for k in range(0, len(VALUES), 8)]
    got_chunk = [0] * len(chunks)
    for si in range(samples):
      for ci, chunk in enumerate(chunks):
        try:
            txt = ml_llm.call(PROMPT.format(name=NAMES[lang], values=", ".join(map(str, chunk))), system=SYSTEM,
                              max_tokens=30000, temperature=0.0 if si == 0 else 0.7)
            d = json.loads(txt)
        except Exception as e:  # noqa: BLE001  (one malformed draft is tolerated while another draft covers the chunk)
            print(lang, "draft failed", repr(e)[:120], file=sys.stderr)
            continue
        got_chunk[ci] += 1
        for k in ("cardinal", "ordinal"):
            for v, ws in (d.get(k) or {}).items():
                if not str(v).isdigit() or not isinstance(ws, list):
                    continue
                cur = set(out.setdefault(k, {}).get(str(int(v)), []))
                for w in ws:
                    w = clean_form(w)
                    if w and " " not in w:
                        cur.add(w)
                        votes[(w, int(v))] = votes.get((w, int(v)), 0) + 1
                out[k][str(int(v))] = sorted(cur)
        for k in ("decimal", "connector", "half", "minus"):
            out.setdefault(k, set()).update(w for w in map(clean_form, d.get(k) or []) if w)
        out.setdefault("fraction", {})
        for den, ws in (d.get("fraction") or {}).items():
            if isinstance(ws, list):
                out["fraction"].setdefault(str(den), set()).update(w for w in map(clean_form, ws) if w)
    missing = [chunks[i] for i, n in enumerate(got_chunk) if n == 0]
    if missing:
        raise RuntimeError(f"{lang}: no draft for value chunk(s) {missing}")
    for k in ("decimal", "connector", "half", "minus"):
        out.setdefault(k, set())
        out[k] = sorted(out[k])
    out["fraction"] = {k: sorted(v) for k, v in out["fraction"].items()}
    holes = [v for v in list(range(0, 20)) + list(range(20, 100, 10)) + [100]
             if not out.get("cardinal", {}).get(str(v))]
    if holes:
        raise RuntimeError(f"{lang}: no cardinal forms drafted for {holes}")
    out["_votes"] = votes
    return out


def _lcp(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def check(lex: dict, lang: str = "") -> dict:
    """Settle a form listed under two different values (an LLM slip, e.g. "osiemnastego" under 18 and 19): it goes to
    the value whose unambiguous forms share the longest prefix with it, when that prefix beats every other value's by
    >= 2 letters; otherwise it is dropped from all. The raw union stays in "raw" so check() can be re-run.
    Junk (control / format characters, template markers, digits) is filtered out of every list, and the hand
    overrides (EXTRA_CARDINAL / EXTRA_ORDINAL / REPLACE_ORDINAL / REMOVE_FORMS) are applied before the settling, so
    a base form can never be lost to it."""
    lex.pop("_votes", None)
    raw = lex.setdefault("raw", {k: {v: list(ws) for v, ws in lex[k].items()} for k in ("cardinal", "ordinal")})
    for k in ("cardinal", "ordinal"):  # junk never survives, not even in the stored union
        raw[k] = {v: sorted({w for w in map(clean_form, ws) if w}) for v, ws in raw[k].items()}
    for k in ("decimal", "connector", "half", "minus"):
        lex[k] = sorted({w for w in map(clean_form, lex.get(k, [])) if w})
    lex["fraction"] = {d: sorted({w for w in map(clean_form, ws) if w}) for d, ws in lex.get("fraction", {}).items()}
    work = {k: {v: list(ws) for v, ws in raw[k].items()} for k in ("cardinal", "ordinal")}
    for v in REPLACE_ORDINAL.get(lang, []):
        work["ordinal"][str(v)] = []
    for k, tab in (("cardinal", REMOVE_FORMS.get(lang, {}).get("cardinal", {})),
                   ("ordinal", REMOVE_FORMS.get(lang, {}).get("ordinal", {}))):
        for v, ws in tab.items():
            work[k][str(v)] = [w for w in work[k].get(str(v), []) if w not in ws]
    forced = {}  # form -> (kind, value): an override owns its form
    for k, tab in (("cardinal", EXTRA_CARDINAL.get(lang, {})), ("ordinal", EXTRA_ORDINAL.get(lang, {}))):
        for v, ws in tab.items():
            for w in ws:
                w = clean_form(w)
                if w:
                    forced[(k, w)] = str(v)
    for (k, w), v in forced.items():
        for kk in ("cardinal", "ordinal"):  # the form leaves every other value, and the other kind unless forced there
            if kk != k and (kk, w) in forced:
                continue
            for other in work[kk]:
                if (kk, other) != (k, v) and w in work[kk][other]:
                    work[kk][other] = [x for x in work[kk][other] if x != w]
        work[k][v] = sorted(set(work[k].get(v, [])) | {w})
    owner = {}
    for kind in ("cardinal", "ordinal"):
        for v, ws in work[kind].items():
            for w in ws:
                if (kind, w) not in forced:
                    owner.setdefault(w, set()).add(v)
    amb = {w for w, vs in owner.items() if len(vs) > 1}
    clean = {}
    for kind in ("cardinal", "ordinal"):
        for v, ws in work[kind].items():
            clean.setdefault(v, set()).update(w for w in ws if w not in amb)
    settled, dropped = {}, []
    for w in sorted(amb):
        score = sorted(((max((_lcp(w, f) for f in clean.get(v, ())), default=0), v) for v in owner[w]), reverse=True)
        if score[0][0] >= 3 and score[0][0] - score[1][0] >= 2:
            settled[w] = score[0][1]
        else:
            dropped.append(w)
    for kind in ("cardinal", "ordinal"):
        lex[kind] = {v: sorted(w for w in ws if (kind, w) in forced or w not in amb or settled.get(w) == v)
                     for v, ws in work[kind].items()}
    lex["dropped_ambiguous"] = dropped
    lex["settled_ambiguous"] = settled
    return lex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(NAMES))
    ap.add_argument("--recheck", action="store_true", help="re-run check() on the stored raw union (no LLM call)")
    a = ap.parse_args()
    tab = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for lang in a.langs.split(","):
        tab[lang] = check(gen(lang), lang) if not a.recheck else check(tab[lang], lang)
        n = sum(len(v) for k in ("cardinal", "ordinal") for v in tab[lang][k].values())
        print(lang, "forms", n, "ambiguous dropped", len(tab[lang]["dropped_ambiguous"]), file=sys.stderr)
        json.dump(tab, open(OUT, "w"), ensure_ascii=False, indent=0, sort_keys=True)


if __name__ == "__main__" and "--months" not in sys.argv:
    main()


MONTH_PROMPT = """Language: {name}.
List every written form (all grammatical cases, e.g. the genitive used in dates, and any common abbreviation without the dot) of the names of the twelve months in {name}, lowercase, standard orthography.
Answer as JSON: {{"1": [forms of January], "2": [...], ..., "12": [...]}}"""
MONTHS_OUT = os.path.join(HERE, "ml_multi_months.json")
# short month abbreviations that are ordinary words once folded ("sie" = "się", "lis" = fox, "jul" = Christmas)
MONTH_COMMON_WORDS = {"pl": {"sie", "lis", "lip", "lut", "mar"}, "sv": {"jul", "may"}, "lv": {"aug"}, "fi": {"elo"}}


def months(langs: list) -> None:
    """Month-name forms (dates: "17 sierpnia" = day 17, month 8), drafted twice and unioned; a form listed under two
    months is dropped."""
    tab = json.load(open(MONTHS_OUT)) if os.path.exists(MONTHS_OUT) else {}
    for lang in langs:
        got: dict = {}
        for si in range(2):
            try:
                d = json.loads(ml_llm.call(MONTH_PROMPT.format(name=NAMES[lang]), system=SYSTEM, max_tokens=4000,
                                           temperature=0.0 if si == 0 else 0.7))
            except Exception as e:  # noqa: BLE001
                print(lang, "months draft failed", repr(e)[:100], file=sys.stderr)
                continue
            for m, ws in d.items():
                if str(m).isdigit() and isinstance(ws, list):
                    got.setdefault(str(int(m)), set()).update(str(w).strip().lower() for w in ws if str(w).strip())
        owner: dict = {}
        for m, ws in got.items():
            for w in ws:
                owner.setdefault(w, set()).add(m)
        tab[lang] = {m: sorted(w for w in ws if len(owner[w]) == 1 and len(w) >= 3 and clean_form(w) == w and " " not in w
                               and w not in MONTH_COMMON_WORDS.get(lang, ()))
                     for m, ws in got.items()}
        print(lang, "month forms", sum(len(v) for v in tab[lang].values()), file=sys.stderr)
    json.dump(tab, open(MONTHS_OUT, "w"), ensure_ascii=False, indent=0, sort_keys=True)


if __name__ == "__main__" and "--months" in sys.argv:
    months(sys.argv[sys.argv.index("--months") + 1].split(","))
