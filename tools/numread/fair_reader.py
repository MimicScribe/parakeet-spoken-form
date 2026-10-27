#!/usr/bin/env python3
"""Fair number reader: rewrite a transcript so the base reader (number_outcomes.extract_numbers) reads the value a
person would read, whichever way the transcript writes the number.

Why: the base reader misreads two kinds of text, so systems that write numbers differently were scored unfairly.
  1. English verbalizer-style forms: 'twenty o five' -> 20, 5; 'o nine o nine two' -> the 'o' digits vanish.
  2. Digits with ONE thousands separator: the reader returns both readings ('2,005' -> 2005 or 2.005) and a scorer
     that keeps the SMALLEST would score en '2,005' and es '125.934' as decimals. That would sink a digits-showing system.

normalize(text, lang)               the safe rewrites (unambiguous to a reader of that language)
normalize(text, lang, lenient=True) also the colloquial English readings a person accepts but that a clock or a list
                                    could also mean: 'one fifty eight' -> 158, 'forty two seventeen' -> 4217.
Report strict and lenient side by side; never only lenient.

usage: fair_reader.py --lang en "since twenty o five"        (prints the rewrite)
       python3 -I test_fair_reader.py
"""

import argparse
import os
import re
import sys

_UNITS = "zero one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_YEAR_HEAD = {w: i for i, w in enumerate(_TEENS + ["twenty"], 10)}  # 'X o Y' years 1001..2009
_OH = {"o", "oh"}

# Decimal mark per language; the other of '.' / ',' is the thousands mark. Space grouping is read by the reader itself.
_COMMA_DECIMAL = {"de", "es", "fr", "it", "pt", "nl", "pl", "ru", "uk", "ro", "cs", "sk", "sl", "hr", "bg", "da", "sv",
                  "fi", "et", "lv", "lt", "hu", "el", "mt"}


def _thousands_mark(lang):
    return "." if lang in _COMMA_DECIMAL else ","


def fix_separators(text, lang):
    """'2,005' (en) / '125.934' (es) -> '2005' / '125934': a single thousands mark followed by exactly 3 digits.
    A decimal mark ('1,5' de, '2.005' en) is left alone: the reader's smallest value is then the decimal, which is right."""
    mark = re.escape(_thousands_mark(lang))
    return re.sub(rf"(?<![\d.,])(\d{{1,3}}){mark}(\d{{3}})(?![\d]|[.,]\d)", r"\1\2", text)


def _word_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)]


def _replace_spans(text, edits):
    """edits: [(start, end, replacement)] non-overlapping."""
    out, last = [], 0
    for s, e, rep in sorted(edits):
        out += [text[last:s], rep]
        last = e
    return "".join(out + [text[last:]])


def _gap_ok(text, a_end, b_start):
    """two words are adjacent: only spaces between (a comma or period breaks a number)."""
    return re.fullmatch(r"[ \t]+", text[a_end:b_start]) is not None


def fix_english_years(text):
    """'twenty o five' / 'nineteen oh nine' -> 2005 / 1909 (the 'o' year form)."""
    w = _word_spans(text)
    edits, i = [], 0
    while i + 2 < len(w):
        a, o, u = (x[0].lower() for x in w[i:i + 3])
        if (a in _YEAR_HEAD and o in _OH and u in _UNITS[1:]
                and _gap_ok(text, w[i][2], w[i + 1][1]) and _gap_ok(text, w[i + 1][2], w[i + 2][1])):
            edits.append((w[i][1], w[i + 2][2], str(_YEAR_HEAD[a] * 100 + _UNITS.index(u))))
            i += 3
        else:
            i += 1
    return _replace_spans(text, edits)


def fix_english_oh_digits(text):
    """'o nine o nine two three' -> 'zero nine zero nine two three': inside a run of 3+ spoken digits with at least two
    real digit words, 'o'/'oh' is the digit zero (phone numbers, codes)."""
    w = _word_spans(text)
    edits, i = [], 0
    while i < len(w):
        j = i
        while j < len(w) and w[j][0].lower() in set(_UNITS) | _OH and (j == i or _gap_ok(text, w[j - 1][2], w[j][1])):
            j += 1
        run = w[i:j]
        if len(run) >= 3 and sum(1 for x in run if x[0].lower() in _UNITS) >= 2:
            edits += [(x[1], x[2], "zero") for x in run if x[0].lower() in _OH]
        i = max(j, i + 1)
    return _replace_spans(text, edits)


_CLOCK_AFTER = re.compile(r"\s*(?:a\.?m\.?|p\.?m\.?|o'?clock)\b", re.IGNORECASE)


def _tens_unit(w, k, text):
    """number (20..99) spoken at word k as 'fifty' or 'fifty eight' / 'fifty-eight'; -> (value, words used) or None."""
    x = w[k][0].lower()
    if "-" in x:
        t, _, u = x.partition("-")
        if t in _TENS and u in _UNITS[1:]:
            return _TENS[t] + _UNITS.index(u), 1
        return None
    if x not in _TENS:
        return None
    if k + 1 < len(w) and w[k + 1][0].lower() in _UNITS[1:] and _gap_ok(text, w[k][2], w[k + 1][1]):
        return _TENS[x] + _UNITS.index(w[k + 1][0].lower()), 2
    return _TENS[x], 1


def fix_english_lenient(text):
    """Colloquial English readings (lenient only):
    'one fifty eight' -> 158 (a digit word then 20..99; not before am/pm/o'clock, not after another number word);
    'forty two seventeen' -> 4217 (21..99 then 10..99, the year-pair reading of a 4-digit code)."""
    w = _word_spans(text)
    numberish = set(_UNITS) | set(_TEENS) | set(_TENS) | {"hundred", "thousand", "million", "billion", "point"}
    edits, i = [], 0
    while i + 1 < len(w):
        prev_num = i > 0 and w[i - 1][0].lower().split("-")[0] in numberish and _gap_ok(text, w[i - 1][2], w[i][1])
        a = w[i][0].lower()
        done = False
        if not prev_num and a in _UNITS[1:] and _gap_ok(text, w[i][2], w[i + 1][1]):
            tu = _tens_unit(w, i + 1, text)
            if tu:
                last = i + tu[1]
                nxt_num = last + 1 < len(w) and w[last + 1][0].lower() in numberish and _gap_ok(text, w[last][2], w[last + 1][1])
                if not _CLOCK_AFTER.match(text, w[last][2]) and not nxt_num:
                    edits.append((w[i][1], w[last][2], str(_UNITS.index(a) * 100 + tu[0])))
                    i, done = last + 1, True
        if not done and not prev_num:
            head = _tens_unit(w, i, text)
            if head and head[0] % 10 and i + head[1] < len(w):
                k = i + head[1]
                tail = None
                if w[k][0].lower() in _TEENS:
                    tail = (_TEENS.index(w[k][0].lower()) + 10, 1)
                else:
                    tail = _tens_unit(w, k, text)
                if tail and _gap_ok(text, w[k - 1][2], w[k][1]):
                    last = k + tail[1] - 1
                    nxt_num = last + 1 < len(w) and w[last + 1][0].lower() in numberish
                    if not nxt_num and not _CLOCK_AFTER.match(text, w[last][2]):
                        edits.append((w[i][1], w[last][2], str(head[0] * 100 + tail[0])))
                        i, done = last + 1, True
        if not done:
            i += 1
    return _replace_spans(text, edits)


# ---- rules 3-5: forms written with digits that the pinned reader reads differently from
# the same text in words. Language tables are small and cover the primary + secondary languages only.
_CLOCK_UNIT = re.compile(r"\s*(?:Uhr\b|h\b|heures?\b|horas?\b|ore\b|uur\b|godz\b|Std\b)")
_MIN_SEC = re.compile(r"\s*(?:Minuten|minutes?|minutos?|minuti|minuten)\s+(?:und|et|y|e|en)\s+(?:\d+|[^\W\d_]+)\s+"
                      r"(?:Sekunden|secondes?|segundos?|secondi|seconden)\b", re.IGNORECASE)


def fix_clock_dot(text):
    """'21.45 Uhr' / '14.48 Minuten und fünfzehn Sekunden' -> '21:45' / '14:48 ...': HH.MM (hour 0-23, minute 00-59) directly
    followed by a clock unit (Uhr, h, heures, horas, ore, uur) or by 'Minuten und N Sekunden' is a time. The reader reads
    '21:45' as 21 and 45, the same as the words 'einundzwanzig Uhr fünfundvierzig', instead of the number 21.45."""

    def sub(m):
        rest = text[m.end():]
        if _CLOCK_UNIT.match(rest) or _MIN_SEC.match(rest):
            return f"{m.group(1)}:{m.group(2)}"
        return m.group(0)

    return re.sub(r"(?<![\d.,:])([01]?\d|2[0-3])\.([0-5]\d)(?![\d]|[.,]\d)", sub, text)


def fix_leading_zero_codes(text):
    """'00110' / '0010000' -> '0 0 1 1 0': a bare digit string that starts with 0 (3+ digits, no separator) is a code; the
    reader drops its leading zeros ('110'); a space-grouped tail ('314 000') is not a code. The same code spoken as words reads as one digit per number."""
    return re.sub(r"(?<![\w.,:/+-])(?<!\d[ \u00a0\u202f])(0\d{2,})(?![\w]|[.,:/]\d)", lambda m: " ".join(m.group(1)), text)


_DEC_WORD = {"de": "Komma", "en": "point", "es": "coma", "fr": "virgule", "it": "virgola", "pt": "vírgula", "nl": "komma"}
_DIGIT_WORDS = {
    "de": "null eins zwei drei vier fünf sechs sieben acht neun",
    "en": "zero one two three four five six seven eight nine",
    "es": "cero uno dos tres cuatro cinco seis siete ocho nueve",
    "fr": "zéro un deux trois quatre cinq six sept huit neuf",
    "it": "zero uno due tre quattro cinque sei sette otto nove",
    "pt": "zero um dois três quatro cinco seis sete oito nove",
    "nl": "nul een twee drie vier vijf zes zeven acht negen",
}
_SCALES = {  # word -> factor, per language (singular/plural)
    "de": {"tausend": 10**3, "million": 10**6, "millionen": 10**6, "milliarde": 10**9, "milliarden": 10**9},
    "en": {"thousand": 10**3, "million": 10**6, "millions": 10**6, "billion": 10**9, "billions": 10**9},
    "es": {"mil": 10**3, "millón": 10**6, "millones": 10**6},
    "fr": {"mille": 10**3, "million": 10**6, "millions": 10**6, "milliard": 10**9, "milliards": 10**9},
    "it": {"mila": 10**3, "mille": 10**3, "milione": 10**6, "milioni": 10**6, "miliardo": 10**9, "miliardi": 10**9},
    "pt": {"mil": 10**3, "milhão": 10**6, "milhões": 10**6},
    "nl": {"duizend": 10**3, "miljoen": 10**6, "miljard": 10**9},
}
_WORDISH = r"[^\W\d_]+(?:-[^\W\d_]+)*"


def _digit_word_map(lang):
    ws = _DIGIT_WORDS.get(lang)
    return {w.lower(): str(i) for i, w in enumerate(ws.split())} if ws else {}


def fix_spoken_decimal(text, lang):
    """'16 Komma vier Prozent' -> '16,4 Prozent' ('16 point four' -> '16.4'): digits, the decimal word, then digits or single
    digit words ('4 Komma acht acht' -> '4,88'). The words-only form 'sechzehn Komma vier' is read as 16.4 by the reader."""
    dw, dmap = _DEC_WORD.get(lang), _digit_word_map(lang)
    if not dw:
        return text
    mark = "." if lang == "en" else ","
    pat = re.compile(rf"(?<![\d.,])(\d+)\s+{dw}\b((?:\s+(?:\d|{_WORDISH}))+)", re.IGNORECASE)

    def sub(m):
        frac, toks = "", re.findall(r"\S+", m.group(2))
        used = 0
        for t in toks:
            core = t.rstrip(".,;:!?").lower()
            if core.isdigit() and len(core) == 1:
                frac += core
            elif core in dmap:
                frac += dmap[core]
            else:
                break
            used += 1
            if t != t.rstrip(".,;:!?"):
                break
        if not frac:
            return m.group(0)
        rest = " ".join(toks[used:])
        tail = (" " + rest) if rest else ""
        # keep trailing punctuation of the last used token
        last = toks[used - 1]
        punct = last[len(last.rstrip(".,;:!?")):]
        return f"{m.group(1)}{mark}{frac}{punct}{tail}"

    return pat.sub(sub, text)


def _reader():
    """The base reader module: already imported, else the pinned one via numread.install()."""
    mod = sys.modules.get("number_outcomes")
    if mod is not None:
        return mod
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import numread

        return numread.install()
    except Exception:  # noqa: BLE001
        return None


def _one_word_number(words, lang):
    """value of a run of number words if the pinned reader reads them as exactly one whole number, else None"""
    N = _reader()
    if N is None:
        return None
    try:
        nums = [n for n in N.extract_numbers(" ".join(words), lang, keep_weak=True) if n.vals]
    except Exception:  # noqa: BLE001
        return None
    if len(nums) != 1 or nums[0].form != "words" or nums[0].start != 0 or nums[0].end != len(words) - 1:
        return None
    v = min(nums[0].vals)
    return int(v) if v == int(v) else None


def fix_digit_scale_words(text, lang):
    """'129 Millionen dreihundertfünfundsiebzigtausend' -> '129375000': a digit group, a scale word, then number words
    that the reader reads as one whole number between scale/1000 and scale (100..999 under 'thousand'). A person reads one
    value; the reader returns 129 and 375000 as two numbers. Without such a tail nothing changes."""
    scales = _SCALES.get(lang)
    if not scales:
        return text
    pat = re.compile(rf"(?<![\d.,])(\d+)\s+({_WORDISH})\b((?:\s+{_WORDISH}){{1,10}})")

    def sub(m):
        f = scales.get(m.group(2).lower())
        if not f:
            return m.group(0)
        toks = m.group(3).split()
        for k in range(len(toks), 0, -1):
            v = _one_word_number(toks[:k], lang)
            if v is not None and ((f > 1000 and f // 1000 <= v < f) or (f == 1000 and 100 <= v < 1000)):
                rest = " ".join(toks[k:])
                return f"{int(m.group(1)) * f + v}" + (" " + rest if rest else "")
        return m.group(0)

    return pat.sub(sub, text)


def _is_number_word(tok, lang):
    t = tok.strip(".,;:!?")
    if not t or any(ch.isdigit() for ch in t):
        return bool(t)
    try:
        return bool([n for n in _reader().extract_numbers(t, lang, keep_weak=False) if n.vals])
    except Exception:  # noqa: BLE001
        return False


def fix_spoken_digit_runs(text, lang):
    """'sieben fünf sieben zwei sechs' -> '75726' (en also 'oh'/'o' = 0): 3+ single digit words in a row, no punctuation
    between them, are ONE digit string (a phone number, a code) as a person reads it; the pinned reader otherwise counts
    each digit word as its own number. Not after the decimal word ('Komma sieben sechs sieben' stays a decimal tail,
    fix_spoken_decimal reads it). Same rule for every arm and for references."""
    import unicodedata

    def fold(w):
        return unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode().lower()

    words = _DIGIT_WORDS.get(lang)
    if not words:
        return text
    dmap = {fold(w): str(i) for i, w in enumerate(words.split())}
    if lang == "en":
        dmap.update({"oh": "0", "o": "0"})
    dec = fold(_DEC_WORD.get(lang, ""))
    toks = text.split(" ")
    out, i = [], 0
    while i < len(toks):
        j, ds = i, ""
        while j < len(toks):
            core = toks[j].rstrip(".,;:!?")
            if fold(core) not in dmap or core == "":
                break
            ds += dmap[fold(core)]
            j += 1
            if toks[j - 1] != core:  # punctuation ends the run
                break
        prev = fold(toks[i - 1].rstrip(".,;:!?")) if i > 0 else ""
        nxt = toks[j] if j < len(toks) and toks[j - 1] == toks[j - 1].rstrip(".,;:!?") else ""
        if j > i and (prev == dec or _is_number_word(nxt, lang)):
            # A decimal tail ('Komma sechs neun drei'), or a run that continues into another number word
            # ('null eins sieben … fünftausenddreihundert…', a mixed phone reading): leave the whole run as words.
            out.extend(toks[i:j])
            i = j
        elif j - i >= 3:
            last = toks[j - 1]
            out.append(ds + last[len(last.rstrip(".,;:!?")):])
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def normalize(text, lang, lenient=False):
    if not text:
        return text
    t = fix_separators(text, lang)
    t = fix_clock_dot(t)
    t = fix_spoken_decimal(t, lang)
    t = fix_digit_scale_words(t, lang)
    # Codes: ONE convention, one digit string ('00110' and 'null null eins eins null' both read as one number);
    # fix_leading_zero_codes (expand to single digits) is kept for reference but no longer applied.
    t = fix_spoken_digit_runs(t, lang)
    if lang == "en":
        t = fix_english_years(t)
        t = fix_english_oh_digits(t)
        if lenient:
            t = fix_english_lenient(t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("text")
    a = ap.parse_args()
    print(normalize(a.text, a.lang, a.lenient))


if __name__ == "__main__":
    main()
