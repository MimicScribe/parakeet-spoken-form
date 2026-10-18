"""Romanian number-gender agreement for candidate readings.

The scale nouns "mii", "sute", "milioane" and "miliarde" take the feminine "două" / "douăsprezece" ("două mii", never
"doi mii"); clock hours after "ora" / "orele" are feminine too ("ora douăsprezece", "ora douăzeci și două"); so are
the feminine count nouns below ("două minute"). A reading that breaks one of these is repaired by swapping the one
token; `violations` lists them, `repair` swaps them, `grammatical` is the test.
"unu" before "ora" is left alone: "ora unu" is standard.

For SYNTHETIC / generated readings only (ml_tts_multi.check_row, grammar_rules): a real clip keeps the reading the
speaker said, and "doi mii" is common colloquial speech, so real-row candidates (ml_multi_ctc.expand_cands) are not
repaired.
"""

import re

MASC_TO_FEM = {"doi": "două", "doisprezece": "douăsprezece"}
SCALE_PL = {"mii", "sute", "milioane", "miliarde"}
FEM_COUNT_NOUNS = {"ore", "minute", "secunde", "zile", "săptămâni", "persoane", "țări"}
HOUR_WORDS = {"ora", "orele"}
TENS = {"douăzeci", "treizeci", "patruzeci", "cincizeci", "șaizeci", "șaptezeci", "optzeci", "nouăzeci"}
# words that can stand between "doi" and a scale noun in "doi virgulă cinci miliarde"
_DECIMAL_FILL = {"virgulă", "zero", "unu", "doi", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă", "zece",
                 "și", "sută", "sute", "zeci", "de", "o", "un"} | TENS | {
    "unsprezece", "doisprezece", "treisprezece", "paisprezece", "cincisprezece", "șaisprezece", "șaptesprezece",
    "optsprezece", "nouăsprezece", "două", "douăsprezece"}


def _tok(t: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", t.lower())


def _word(text: str, last: bool) -> str:
    ws = [_tok(w) for w in text.split()]
    ws = [w for w in ws if w]
    return (ws[-1] if last else ws[0]) if ws else ""


def violation_indexes(reading: str, before: str = "", after: str = "") -> list:
    """Token indexes of `reading` whose masculine form is ungrammatical here."""
    toks = reading.split()
    keys = [_tok(t) for t in toks]
    out = set()
    for i, k in enumerate(keys):
        if k not in MASC_TO_FEM:
            continue
        j = i + 1
        nxt = keys[j] if j < len(keys) else ""
        if nxt == "de" and j + 1 < len(keys):  # "doi de mii" is not Romanian either
            nxt = keys[j + 1]
        if nxt in SCALE_PL or nxt in FEM_COUNT_NOUNS:
            out.add(i)
        elif nxt == "virgulă":  # "doi virgulă cinci miliarde"
            m = j
            while m < len(keys) and keys[m] in _DECIMAL_FILL:
                m += 1
            if m < len(keys) and keys[m] in SCALE_PL:
                out.add(i)
        elif j >= len(keys) and _word(after, last=False) in FEM_COUNT_NOUNS:
            out.add(i)
    # clock hour: the hour is the first number (after "ora" / "orele", either before the span or leading it)
    hour_ctx = _word(before, last=True) in HOUR_WORDS
    start = 0
    if keys and keys[0] in HOUR_WORDS:
        hour_ctx, start = True, 1
    if hour_ctx and start < len(keys):
        h = start
        if keys[h] in TENS and h + 2 < len(keys) and keys[h + 1] == "și":
            h += 2
        if keys[h] in MASC_TO_FEM:
            out.add(h)
    return sorted(out)


def violations(reading: str, before: str = "", after: str = "") -> list:
    toks = reading.split()
    return [toks[i] for i in violation_indexes(reading, before, after)]


def grammatical(reading: str, before: str = "", after: str = "") -> bool:
    return not violation_indexes(reading, before, after)


def repair(reading: str, before: str = "", after: str = "") -> str:
    """The reading with each ungrammatical masculine token swapped for its feminine (case and punctuation kept)."""
    toks = reading.split()
    for i in violation_indexes(reading, before, after):
        m = re.fullmatch(r"(\W*)(\w+)(\W*)", toks[i])
        if m:
            fem = MASC_TO_FEM[m.group(2).lower()]
            if m.group(2)[:1].isupper():
                fem = fem[:1].upper() + fem[1:]
            toks[i] = m.group(1) + fem + m.group(3)
    return " ".join(toks)
