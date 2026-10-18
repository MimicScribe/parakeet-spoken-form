"""Value-level score and the ONE slot matcher every seam scorer imports (offset_score.py and the real-bed / seam-gate
reports). Usage: score_values.py <cases_RUN.jsonl> <parse_out_RUN.jsonl> [--lenient]

A sentence's slots are matched IN ORDER against one reading's parsed numbers and each parsed number is consumed by at
most one slot: a slot searches from just past the previous slot's match, so two slots of 5 need two parsed 5s.
Strict rules (the headline):
  PH   a run of consecutive parsed numbers whose digits, joined, EQUAL the phone number;
  P    one number equal to the price, or whole and cents as two numbers next to each other in the parsed list (the cents
       as 25 or as 0.25: "78 euro en 25 cent" reads 78 and 0.25);
  T    h.mm or hmm as one number, h and m next to each other in the parsed list (h, 0, m for one-digit minutes read with a
       zero word: "eleven zero five"), or the hour alone when m = 0; a parsed
       hour whose own words end in minutes the parser dropped ("negentien uur vijf" parsed as 19) counts when those
       words read as m;
  A sum  (every slot but PH / P / T) 2-4 numbers next to each other in the parsed list, each pair split by one joining
       word ("si", "y", "i", "и" ...), each smaller than the place of the one before, adding up to the value: the
       no-joiner reading splits "cincizeci si cinci" into 50, 5 (so a range "55 si 76" is right as two slots, never
       one number 131);
  else one number equal to the value.
Readings: a slot is right when ANY reading of the same text has it right: the number grammar's greedy parse, its
no-joiner parse, and the public reader (number_outcomes.extract_numbers on the text the parse was made from, the
parse_in row of the same key; fr / it / es / pt noon and midnight words read as 12 / 0). The reader reading fixes
grammar-parser artefacts identically for every arm: grouped phone digits summed into one number, fi / da / el
decimal-word and "and" forms parsed as fractions, hu "ora" / nl "uur" minutes dropped, case-inflected counts and
ordinal years. A missing reader is an error, never a silent no-credit.
Lenient (a sensitivity row only: parser-limit forgiveness, never model errors; needs the parse's word spans):
  L    mantissa x 10^k when the scale word for 10^k follows the mantissa in the text; the mantissa is one number or a
       whole / fraction pair next to each other in the text ("three point two million" parsed 3.2, or 3 and 2);
  D    a value with zero digits next to it in the text ("1.7", "0" for 1.70): exact up to trailing zeros;
  L/N/Y >= 100: the digits of 2-4 parsed numbers next to each other in the text, joined ("19", "97" for 1997);
  T    as strict (the hour alone only when m = 0).
"""
import collections
import json
import math
import os
import re
import sys

SL = ['L', 'PH', 'P', 'T', 'Y', 'D', 'PCT', 'N', 'M', 'A', 'B']
# scale-word prefixes by power of ten (lower case); only the lenient L rule reads them
SCALE = {
    3: ("thousand", "mil", "mille", "mila", "tausend", "duizend", "tysiąc", "tysiące", "tysięcy", "тысяч", "тысяча",
        "тисяч", "тисяча", "хиляд", "tusen", "tisíc", "tisuć", "tisoč", "ezer", "tuhat", "tūkst", "χιλιάδ", "χίλια",
        "tusind", "mii", "mie"),
    6: ("million", "millón", "millones", "milhão", "milhões", "milione", "milioni", "miljoen", "milion", "milioane",
        "миллион", "мільйон", "милион", "miljon", "millió", "miljoon", "milijun", "milijon", "εκατομμύρι", "mio"),
    9: ("billion", "milliard", "miliard", "mrd", "mld", "миллиард", "мільярд", "милиард", "miljard", "milliárd",
        "miljardi", "milijard", "δισεκατομμύρι"),
}


def num(s):
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def base_slot(s):
    """'N2' -> 'N' (repeated slot types are numbered)."""
    return re.sub(r"\d+$", "", s)


class Reading:
    """One parsed reading of a text: values (strings), spans ([first, last] word index per value; None for a parse
    made before spans existed) and the words the spans index."""

    def __init__(self, vals, spans=None, words=None):
        self.vals = [str(x) for x in vals]
        self.nums = [num(x) for x in self.vals]
        self.spans = spans if spans is not None and len(spans) == len(self.vals) else None
        self.words = words or []

    def text_adjacent(self, i, j):
        """values i < j stand next to each other in the text (no word between)"""
        return self.spans is not None and self.spans[i][1] + 1 == self.spans[j][0]

    def word_after(self, i):
        if self.spans is None:
            return ""
        k = self.spans[i][1] + 1
        return self.words[k] if k < len(self.words) else ""


def as_reading(x):
    return x if isinstance(x, Reading) else Reading(x)


def _eq(a, b):
    return a is not None and abs(a - b) < 1e-6


def _digits(s):
    return re.sub(r"\D", "", str(s))


CLOCK_WORDS = {"uur": "nl"}  # hour word -> language of the minutes that follow it inside the parser's span
_N = None


def reader():
    """The public number reader (number_outcomes.py next to this file), however this file was loaded (by path, from
    another directory, as a copy). An import failure raises: no extra credit must never be silent."""
    global _N
    if _N is None:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import number_outcomes as N
        _N = N
    return _N


# noon / midnight words of a clock time ("midi quinze" = 12:15), read as the hour number for the reader reading only
NOON = {"fr": {"midi": "douze", "minuit": "zéro"}, "it": {"mezzogiorno": "dodici", "mezzanotte": "zero"},
        "es": {"mediodía": "doce", "medianoche": "cero"}, "pt": {"meio-dia": "doze", "meia-noite": "zero"}}


def reader_reading(text, lang):
    """the public reader's numbers of a text as a Reading (no spans: the lenient rules never apply to it)"""
    t = text or ""
    for w, rep in NOON.get(lang, {}).items():
        t = re.sub(rf"(?<![\w-]){re.escape(w)}(?![\w-])", rep, t, flags=re.IGNORECASE)
    vals = []
    for n in reader().extract_numbers(t, lang, keep_weak=True):
        if n.vals:
            v = min(n.vals)
            vals.append(str(int(v)) if v == int(v) else str(v))
    return Reading(vals)


# joining words that stand between the parts of ONE number in some language ("cincizeci si cinci", "treinta y nueve",
# "двадесет и пет"); the no-joiner reading splits there (same list as the grammar parse dump)
JOINERS = {"and", "und", "et", "y", "e", "i", "en", "och", "og", "si", "și", "a", "és", "ja", "и", "і", "й", "та",
           "ir", "in", "και", "ve"}


def _place(x):
    """10^k for the lowest non-zero digit place of a whole number (50 -> 10, 1000 -> 1000, 55 -> 1)"""
    p = 1
    while x and x % (p * 10) == 0:
        p *= 10
    return p


def _joined_sum(r, i, fv):
    """indices i..j (2-4 numbers) whose sum is fv, each pair split by exactly one joining word in the text and each
    part below the place of the running sum ("50 si 5" = 55; "55 si 76" never sums), else None"""
    if r.spans is None:
        return None
    n = r.nums
    if n[i] is None or n[i] != int(n[i]) or n[i] <= 0:
        return None
    tot = int(n[i])
    for j in range(i + 1, min(len(n), i + 4)):
        x = n[j]
        a, b = r.spans[j - 1][1], r.spans[j][0]
        if x is None or x != int(x) or x <= 0 or b != a + 2 or r.words[a + 1].lower() not in JOINERS:
            return None
        if int(x) >= _place(tot):
            return None
        tot += int(x)
        if abs(tot - fv) < 1e-6:
            return list(range(i, j + 1))
    return None


def _clock_tail(r, i):
    """The minutes written after the hour word inside parsed number i's own words ("negentien uur vijf" -> 5), or None.
    The parser keeps only the hour of such a span; the words after the hour word are read here with the public reader."""
    if r.spans is None:
        return None
    a, b = r.spans[i]
    ws = [w.lower() for w in r.words[a:b + 1]]
    for k, w in enumerate(ws):
        if w in CLOCK_WORDS and 0 < k < len(ws) - 1:
            ns = reader().extract_numbers(" ".join(ws[k + 1:]), CLOCK_WORDS[w], keep_weak=False)
            if len(ns) == 1 and len(ns[0].vals) == 1:
                return int(min(ns[0].vals)) if min(ns[0].vals) == int(min(ns[0].vals)) else None
    return None


def _strict(b, v, r, i):
    """indices consumed when slot base b with value v matches reading r starting AT parsed number i, else None"""
    n = r.nums
    nxt = i + 1 if i + 1 < len(n) else None
    if b == 'PH':
        want, got = _digits(v), ""
        for j in range(i, len(n)):
            got += _digits(r.vals[j])
            if got == want:
                return list(range(i, j + 1))
            if not want.startswith(got):
                return None
        return None
    if n[i] is None:
        return None
    if b == 'P':
        f = float(v)
        a, c = int(f), round((f - int(f)) * 100)
        if _eq(n[i], f):
            return [i]
        if c and nxt is not None and _eq(n[i], a) and (_eq(n[nxt], c) or _eq(n[nxt], c / 100)):
            return [i, nxt]
        return None
    if b == 'T':
        h, m = (int(x) for x in str(v).split(':'))
        if m and nxt is not None and _eq(n[i], h) and _eq(n[nxt], m):
            return [i, nxt]
        # a zero word before one-digit minutes ("eleven zero five", "atten nul fem", "osem nič pet") = h, 0, m
        if 0 < m < 10 and i + 2 < len(n) and _eq(n[i], h) and _eq(n[i + 1], 0) and _eq(n[i + 2], m) \
                and r.vals[i + 1].strip("0") == "":
            return [i, i + 1, i + 2]
        if _eq(n[i], h + m / 100) or _eq(n[i], h * 100 + m) or (m == 0 and _eq(n[i], h)):
            return [i]
        if m and _eq(n[i], h) and _clock_tail(r, i) == m:
            return [i]
        return None
    if _eq(n[i], float(v)):
        return [i]
    return _joined_sum(r, i, float(v))


def _lenient(b, v, r, i):
    """extra forms of the lenient sensitivity row (needs spans): indices consumed, or None"""
    if r.spans is None or r.nums[i] is None:
        return None
    n = r.nums
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if b == 'L':
        mants = [([i], n[i])]
        if i + 1 < len(n) and n[i + 1] is not None and r.text_adjacent(i, i + 1) and n[i] == int(n[i]):
            mants.append(([i, i + 1], float(f"{int(n[i])}.{_digits(r.vals[i + 1])}")))
        for idx, m in mants:
            nxt = r.word_after(idx[-1])
            for k, stems in SCALE.items():
                if abs(m * 10 ** k - fv) < 1e-6 * max(1.0, abs(fv)) and nxt.startswith(stems):
                    return idx
    if b == 'D':
        s, j = r.vals[i], i
        while j + 1 < len(n) and r.vals[j + 1].strip("0") == "" and r.text_adjacent(j, j + 1):
            s += r.vals[j + 1]
            j += 1
            if '.' in s and _eq(num(s), fv):
                return list(range(i, j + 1))
    if b in ('L', 'N', 'Y') and fv >= 100 and fv == int(fv):
        got = ""
        for j in range(i, min(len(n), i + 4)):
            if j > i and not r.text_adjacent(j - 1, j):
                return None
            got += _digits(r.vals[j])
            if j > i and got == str(int(fv)):
                return list(range(i, j + 1))
    return None


def match_reading(values, r, lenient=False):
    """{slot key: right} for one reading: slots in order, each parsed number consumed by one slot at most"""
    r = as_reading(r)
    out, used, cur = {}, set(), 0
    for s, v in values.items():
        b, hit = base_slot(s), None
        for i in range(cur, len(r.vals)):
            idx = _strict(b, v, r, i) or (_lenient(b, v, r, i) if lenient else None)
            if idx and not used.intersection(idx):
                hit = idx
                break
        out[s] = hit is not None
        if hit:
            used.update(hit)
            cur = max(hit) + 1
    return out


def match_slots(values, readings, lenient=False):
    """{slot key: right in ANY reading of the same text} (greedy, no-connector and reader readings)"""
    res = {s: False for s in values}
    for r in readings:
        for s, ok in match_reading(values, r, lenient).items():
            res[s] = res[s] or ok
    return res


def slot_ok(slot, v, vals, lenient=False):
    """one slot alone against one reading (a list of parsed strings or a Reading)"""
    return match_reading({slot: v}, vals, lenient)[slot]


def slot_any(slot, v, readings, lenient=False):  # a slot value found in either reading of the same text
    return any(slot_ok(slot, v, x, lenient) for x in readings)


def load_parse(path, parse_in=None):
    """parse_out rows -> {key: (greedy Reading, no-connector Reading[, reader Reading])}. The reader reading needs the
    parse input (key, lang, text): parse_in, default the parse_in_<name> file next to parse_out_<name>; without one
    the two grammar readings are returned and a warning is printed (old dumps / unit tests)."""
    src = {}
    pin = parse_in or os.path.join(os.path.dirname(path), os.path.basename(path).replace("parse_out_", "parse_in_", 1))
    if os.path.basename(pin) != os.path.basename(path) and os.path.exists(pin):
        with open(pin, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    src[r["key"]] = (r["lang"], r["text"])
    elif parse_in:
        raise FileNotFoundError(parse_in)
    else:
        print(f"score_values: no parse input next to {path}: grammar readings only (no reader reading)", file=sys.stderr)
    P = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            w = r.get('words')
            rd = (Reading(r['values'], r.get('spans'), w), Reading(r.get('alt', []), r.get('alt_spans'), w))
            if src:
                lang, text = src[r['key']]  # a parse row without its input is an error
                rd += (reader_reading(text, lang),)
            P[r['key']] = rd
    return P


def main(argv):
    lenient = "--lenient" in argv
    argv = [a for a in argv if not a.startswith("--")]
    rows = [json.loads(l) for l in open(argv[0], encoding="utf-8") if l.strip()]
    P = load_parse(argv[1])
    c = collections.Counter()
    ex = collections.defaultdict(list)
    L = collections.defaultdict(collections.Counter)  # per language, over ALL sentences
    for k, r in enumerate(rows):
        lv, sv = P[f"{k}:long"], P[f"{k}:single"]  # a missing parse is an error, not a wrong reading
        ml, ms = match_slots(r['values'], lv, lenient), match_slots(r['values'], sv, lenient)
        lo, so = all(ml.values()), all(ms.values())
        t = L[r['lang']]
        t['n'] += 1; t['long_ok'] += lo; t['single_ok'] += so; t['seam_wrong'] += so and not lo; t['seam_fixed'] += lo and not so
        for s in r['values']:
            b = base_slot(s)
            t['n_' + b] += 1; t['long_ok_' + b] += ml[s]; t['seam_wrong_' + b] += ms[s] and not ml[s]
        if r.get('same', False):
            continue  # the categories below are the differing readings only, as before
        m = bool(r['merge_gaps'])
        grp = 'en' if r['lang'] == 'en' else 'non-en'
        cls = ('form_only' if lo and so else 'LONG_WRONG_single_right' if so else 'long_right_single_wrong' if lo else 'both_wrong')
        c[f"{grp}|{cls}|{'merge' if m else 'nomerge'}"] += 1
        if cls == 'LONG_WRONG_single_right':
            ex[m].append(r)
    for k in sorted(c):
        print(k, c[k])
    print("\n== per language, all sentences: long-file value-right % | single % | seam-made wrong | seam-fixed | seam-made wrong by slot ==")
    for lang in sorted(L):
        t = L[lang]
        n = t['n']
        print(f"{lang} n={n} long={100 * t['long_ok'] / n:.1f}% single={100 * t['single_ok'] / n:.1f}% seam_wrong={t['seam_wrong']} seam_fixed={t['seam_fixed']} "
              + " ".join(f"{s}:{t['seam_wrong_' + s]}/{t['n_' + s]}" for s in SL if t['n_' + s]))
    T = collections.Counter()
    for t in L.values():
        T.update(t)
    print("ALL", f"n={T['n']} long={100 * T['long_ok'] / T['n']:.1f}% single={100 * T['single_ok'] / T['n']:.1f}% seam_wrong={T['seam_wrong']} seam_fixed={T['seam_fixed']}",
          " ".join(f"{s}:{T['seam_wrong_' + s]}/{T['n_' + s]}(long ok {100 * T['long_ok_' + s] / T['n_' + s]:.0f}%)" for s in SL if T['n_' + s]))
    print("\n== long wrong, single right, merge-touched (first 25) ==")
    for r in ex[True][:25]:
        print(r['file'], r['values'], '\n  L:', r['long'], '\n  S:', r['single'], '\n  gaps:', [(g['pick'], g['A'], g['B'], g['R']) for g in r['merge_gaps']])
    return {"langs": {k: dict(v) for k, v in L.items()}, "categories": dict(c)}


if __name__ == "__main__":
    main(sys.argv[1:])
