"""ONE reader for number GROUPS: phone numbers, clock times, fractions and digit groups, written as digit groups, as number
words, or both.  Shared by classread / symscore (eval/quality), the P7 seam matcher (score_values) and adv_b_score, so the three
agree on the same text (audit 2, D12/D14/D15: each had its own reader, and none read both styles).

Contract (known-answer tests in test_group_read.py):
  * a PERFECT digit output and a PERFECT word output of the same number both read right;
  * a changed, dropped or added FIRST digit is always caught (phone: no leading-digit leniency; the only allowed difference
    is a trunk 0 replaced by a country code, 06... = +33 6...);
  * the reader never looks at units, letters or symbols.
Pure functions; the number words come from adv_b_words (grammar tables of ml_cands) and number_outcomes (clock / half words).

    phone_options(text, lang)        -> {digit string, ...}   every digit string some parse of the text gives
    phone_matches(text, lang, exp)   -> bool                  strict: value_matches against `exp`
    clock_values(text, lang)         -> {(hour % 12, minute), ...}
    fraction_values(text, lang)      -> {Decimal, ...}   digit, vulgar and word fractions in en es fr de pt it nl
    digit_readings(text)             -> [list of literal digit strings]  (P7: leading zeros kept, thousands groups merged)
    group_matches(kind, ref, hyp, lang) -> bool
"""
import functools
import os
import re
import sys
import unicodedata
from decimal import Decimal

_QUALITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "quality")    # number_outcomes is read lazily from here
if _QUALITY not in sys.path:
    sys.path.append(_QUALITY)

REPEAT = {"double": 2, "doble": 2, "dubbel": 2, "doppel": 2, "doppio": 2, "duplo": 2, "triple": 3, "treble": 3, "tripla": 3, "triplo": 3,
          "dreifach": 3}
ZERO = {"oh", "o", "zero", "nought", "naught", "cero", "zéro", "null", "nul"}
CONNECT = {"und", "y", "e", "et", "en", "ën", "ed", "and"}
ZERO_OF = {"en": {"oh", "o", "zero", "nought", "naught"}, "es": {"cero"}, "fr": {"zero", "zéro"}, "de": {"null"}, "pt": {"zero"},
           "it": {"zero"}, "nl": {"nul", "null"}}   # the words a language says for 0 inside a phone number
PT_PT = {"quatorze": "catorze", "dezasseis": "dezesseis", "dezassete": "dezessete", "dezanove": "dezenove"}   # pt-PT teens = the pt-BR words
TRUNK_CC = ("33", "49", "31", "34", "39", "351", "44", "41", "43", "32", "55", "27", "61")  # country codes (2+ digits) that replace a trunk 0


def fold(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower().replace("ß", "ss"))
    return "".join(c for c in w if not unicodedata.combining(c))


def toks(s: str) -> list:
    """Folded tokens: words, and digit groups with their inner separators ("06.12.34", "1,200"); hyphens split."""
    s = s.replace("’", "'").replace("-", " ")
    return [fold(x) for x in re.findall(r"[^\W_]+(?:'[^\W_]+)?|\d+(?:[.,:/]\d+)*", s.lower(), re.U)]


def toks_raw(s: str) -> list:
    """Like toks but NOT folded (umlauts stay: compound_split needs "fünfundvierzig", not "funfundvierzig")."""
    s = s.replace("’", "'").replace("-", " ")
    return [x for x in re.findall(r"[^\W_]+(?:'[^\W_]+)?|\d+(?:[.,:/]\d+)*", s.lower(), re.U)]


def dig(tok: str):
    """The digit characters of a written-digit token ("1800", "06.12.34", "+33" -> "33"), None for a word."""
    return re.sub(r"\D", "", tok) if re.search(r"\d", tok) else None


# ------------------------------------------------------------------------------------------------ phone reader
_TUP = {}


def _table(lang: str) -> dict:
    """word tuple -> digit string for the readings of a phone group: digit words, pairs 10-99, hundreds 100-999, 'zero X' pairs."""
    if lang in _TUP:
        return _TUP[lang]
    import adv_b_words as W

    if lang not in W.DIGITS:                                    # no number-word tables for this language: digit groups only
        _TUP[lang] = {}
        return {}
    t = {}
    for n in range(0, 1000):
        if n >= 100 and n % 100:
            continue            # 612 is "seiscientos" + the pair "doce" (see phone_digits); only round hundreds are table entries
        try:
            w = tuple(fold(x) for x in W.card(n, lang).split())
        except Exception:  # noqa: BLE001
            continue
        t.setdefault(w, str(n))
    for d in range(10):
        t[(fold(W.DIGITS[lang][d]),)] = str(d)
    z = fold(W.DIGITS[lang][0])
    zeros = ZERO_OF.get(lang, set()) | {z}                      # "o" is zero in English only (pt "o" is the article: "passou o 41 949 ...")
    for z_ in zeros:
        t[(z_,)] = "0"
        for d in range(1, 10):
            t[(z_, fold(W.DIGITS[lang][d]))] = "0" + str(d)
    if lang == "fr":
        t[("zero", "huit", "cent")] = "0800"
        for tens, word in ((7, "septante"), (8, "huitante"), (9, "nonante")):
            for u in range(10):
                n = tens * 10 + u
                tail = () if u == 0 else ("et", "un") if u == 1 else (fold(W.DIGITS["fr"][u]),)
                t.setdefault((word,) + tail, str(n))
    for n in range(1000, 10000, 1000):                          # "zero eight hundred, two thousand" (0800 2000)
        try:
            t.setdefault(tuple(fold(x) for x in W.card(n, lang).split()), str(n))
        except Exception:  # noqa: BLE001
            pass
    for extra, d in {"de": (("zwo", "2"),), "pt": (("uma", "1"), ("duas", "2")), "es": (("una", "1"),), "fr": (("une", "1"),)}.get(lang, ()):
        t.setdefault((extra,), d)                               # spoken digits: German phone "zwo", feminine one / two
    if lang == "fr":                                            # "cents" only when final: "deux cent vingt quatre" is 224
        for d in range(2, 10):
            t.setdefault((fold(W.DIGITS["fr"][d]), "cent"), str(d * 100))
    if lang == "es":
        t.setdefault(("ciento",), "100")                        # "cien" alone, "ciento" before more
    if lang == "pt":
        t.setdefault(("cento",), "100")                         # "cem" alone, "cento" before more
    if lang == "pt":                                            # pt-PT teens (the pt-BR cardinal is catorze / dezesseis / dezessete / dezenove)
        for w_, n_ in (("quatorze", "14"), ("dezasseis", "16"), ("dezassete", "17"), ("dezanove", "19")):
            t.setdefault((w_,), n_)
    if lang == "it":                                            # "venti otto": the unelided spaced form of ventotto (28), not 20 + 8
        for tens in range(2, 10):
            tw_words = tuple(fold(x) for x in W.card(tens * 10, "it").split())
            for u in range(1, 10):
                t.setdefault(tw_words + (fold(W.DIGITS["it"][u]),), str(tens * 10 + u))
    _TUP[lang] = t
    return t


def phone_digits(words: list, lang: str, merge_single: bool = False) -> str:
    """Digit string of a phone reading (longest-match over the table; unknown words are skipped; 'plus' etc. add nothing)."""
    t = _table(lang)
    w = [fold(x) for x in words]
    out, i = [], 0
    while i < len(w):
        if dig(w[i]) is not None:                        # a written digit group
            out.append(dig(w[i]))
            i += 1
            continue
        if w[i] in REPEAT and i + 1 < len(w):            # repeat words expand first: "double five" -> five five
            j = i + 1
            best = None
            for ln in (3, 2, 1):
                if tuple(w[j:j + ln]) in t:
                    best = (ln, t[tuple(w[j:j + ln])])
                    break
            if best:
                out.append(best[1] * REPEAT[w[i]])
                i = j + best[0]
                continue
        for ln in (6, 5, 4, 3, 2, 1):
            key = tuple(w[i:i + ln])
            if len(key) == ln and key in t:
                v = t[key]
                nxt = w[i + ln] if i + ln < len(w) else ""
                if len(v) == 2 and v[0] == "0" and nxt in CONNECT:      # "null drei und dreissig": the 0 is a digit, drei und dreissig is 33
                    continue
                if len(v) == 3 and v.endswith("00") and v[0] != "0":    # a round hundred: "seiscientos" + a PAIR (doce) is 612; + a digit is 600 then that digit
                    for ln2 in (4, 3, 2, 1):
                        k2 = tuple(w[i + ln:i + ln + ln2])
                        v2 = t.get(k2) if len(k2) == ln2 else None
                        if v2 is not None and ((len(v2) == 2 and v2[0] != "0") or (merge_single and len(v2) == 1)):
                            v = v[0] + (v2 if len(v2) == 2 else "0" + v2)
                            ln += ln2
                            break
                out.append(v)
                i += ln
                break
        else:
            i += 1
    return "".join(out)


def phone_digit_set(words: list, lang: str, cap: int = 400) -> set:
    """Every digit string some parse of `words` can give (a token run is often ambiguous: "quarante quatre vingt dix sept" is 40 97 or 44 20 17;
    "seiscientos cinco" is 605 or 600 5). A hypothesis is value-right when the expected digits are among them."""
    t = _table(lang)
    w = [fold(x) for x in words]
    n = len(w)

    @functools.lru_cache(maxsize=None)
    def go(i):
        if i >= n:
            return frozenset({""})
        res = set()
        hit = False
        if dig(w[i]) is not None:                        # a written digit group
            return frozenset(dig(w[i]) + r for r in go(i + 1))
        if w[i] in REPEAT and i + 1 < n:
            for ln in (3, 2, 1):
                k = tuple(w[i + 1:i + 1 + ln])
                if len(k) == ln and k in t:
                    hit = True
                    for r in go(i + 1 + ln):
                        res.add(t[k] * REPEAT[w[i]] + r)
        for ln in (6, 5, 4, 3, 2, 1):
            k = tuple(w[i:i + ln])
            if len(k) == ln and k in t:
                hit = True
                v = t[k]
                for r in go(i + ln):
                    res.add(v + r)
                if len(v) == 3 and v.endswith("00") and v[0] != "0":       # "seiscientos" + "doce" = 612, + "cinco" = 605
                    for ln2 in (4, 3, 2, 1):
                        k2 = tuple(w[i + ln:i + ln + ln2])
                        v2 = t.get(k2) if len(k2) == ln2 else None
                        if v2 is not None and (len(v2) <= 2):
                            for r in go(i + ln + ln2):
                                res.add(v[0] + (v2 if len(v2) == 2 else "0" + v2) + r)
        if not hit:
            res |= set(go(i + 1))
        return frozenset(list(res)[:cap])

    return set(go(0))


def value_matches(g: str, exp: str) -> bool:
    """A read digit string equals the number's digits; the only allowed difference is a trunk 0 replaced by a country code (06... = +33 6...).
    No leading-digit leniency: a changed, dropped or added first digit is an error."""
    return g == exp or (exp.startswith("0") and any(g == cc + exp[1:] for cc in TRUNK_CC))


_MODE = {"de": "de", "nl": "nl", "it": "it-"}


def expand_fused(words: list, lang: str) -> list:
    """de / nl / it: a fused compound number word ("dreiunddreißig", "tweeëndertig", "ventidue") becomes its morphemes, which the table reads."""
    mode = _MODE.get(lang)
    if not mode:
        return list(words)
    import compound_split as CS

    out = []
    for x in words:
        if dig(x) is None and x.isalpha() and len(x) > 4:
            parts = CS.split_token(x.lower(), mode)
            if len(parts) > 1:
                out += [fold(p) for p in parts]
                continue
        out.append(fold(x))
    return out


def _numeric(w: list, k: int, t: dict) -> bool:
    """token k is a digit group or a word the table (or the number grammar) knows: a connector next to two of them cannot be skipped."""
    if not 0 <= k < len(w):
        return False
    if dig(w[k]) is not None:
        return True
    return any(w[k] in key and w[k] not in CONNECT for key in t) or w[k] in REPEAT


def _moves(w: list, i: int, t: dict, n: int, lang: str = "en") -> list:
    """[(digit string, next index)] for reading the token at i: a digit group, a repeat word + digits, a table entry (+ a round hundred merged
    with a following pair, optionally through its connector), or [("", i + 1)] (the word is skipped) when it reads as nothing.  A token that is
    itself a number the table cannot place (a scale word, an unlisted compound) or a connector standing BETWEEN two numbers is NOT skipped:
    skipping them read "treinta y tres" as 30 3, "two thousand eleven" as 2 11 (audit-2 review: false accepts)."""
    if dig(w[i]) is not None:
        return [(dig(w[i]), i + 1)]
    out = []
    if w[i] in REPEAT and i + 1 < n:
        for ln in (3, 2, 1):
            k = tuple(w[i + 1:i + 1 + ln])
            if len(k) == ln and k in t:
                out.append((t[k] * REPEAT[w[i]], i + 1 + ln))
    for ln in (6, 5, 4, 3, 2, 1):
        k = tuple(w[i:i + ln])
        if len(k) == ln and k in t:
            v = t[k]
            out.append((v, i + ln))
            if len(v) == 3 and v.endswith("00") and v[0] != "0":       # "seiscientos" + "doce" = 612, + "cinco" = 605 ("seiscentos e doze", "six hundred and twelve")
                for skip in (0, 1):
                    if skip and not (i + ln < n and w[i + ln] in CONNECT):
                        continue
                    for ln2 in (4, 3, 2, 1):
                        k2 = tuple(w[i + ln + skip:i + ln + skip + ln2])
                        v2 = t.get(k2) if len(k2) == ln2 else None
                        if v2 is not None and len(v2) <= 2:
                            out.append((v[0] + (v2 if len(v2) == 2 else "0" + v2), i + ln + skip + ln2))
    if out:
        return out
    if w[i] in CONNECT and _numeric(w, i - 1, t) and _numeric(w, i + 1, t):
        return []                                                       # a joiner between two numbers that no table entry consumed: no reading
    if w[i] not in CONNECT and _is_number_word(w[i], lang):
        return []                                                       # "thousand", "mil", an unlisted compound: not a carrier word
    return [("", i + 1)]


_NUMW = {}


def _is_number_word(x: str, lang: str) -> bool:
    """x parses as a number word of THIS language (so it cannot be a carrier word inside a phone number; "dos" is a pt preposition)."""
    key = (x, lang)
    if key in _NUMW:
        return _NUMW[key]
    import number_outcomes as N

    r = False
    if x.isalpha() and len(x) >= 3:
        try:
            r = N.parse_value(x, lang) is not None
        except Exception:  # noqa: BLE001
            r = False
    _NUMW[key] = r
    return r


def phone_reads(words: list, lang: str, exp: str) -> bool:
    """True when SOME parse of the words (digit groups, number words, repeat words, pairs, hundreds) is exactly `exp`.  Exact search over
    (token, digit position): no cap on the parses, so a long number cannot lose its right reading to a truncated list."""
    t = _table(lang)
    w = expand_fused(list(words), lang)
    n, ln_exp = len(w), len(exp)

    @functools.lru_cache(maxsize=None)
    def go(i, pos):
        if i >= n:
            return pos == ln_exp
        return any(exp.startswith(v, pos) and go(j, pos + len(v)) for v, j in _moves(w, i, t, n, lang))

    return go(0, 0)


def phone_style(reg: list, lang: str = "en") -> str:
    """How the hypothesis wrote the phone region: digit (written digit groups only), word (number words only), mixed, none.
    Carrier words inside the region ("area code") are neither."""
    nw = ({x for k in _table(lang) for x in k} | set(REPEAT)) - CONNECT
    d = any(dig(t) is not None for t in reg)
    w = any(dig(t) is None and fold(t) in nw for t in reg)
    return "mixed" if d and w else "digit" if d else "word" if w else "none"


def phone_in_text(text: str, lang: str, exp: str, last: bool = False) -> bool:
    """`exp` is the phone number SOMEWHERE in a sentence (with last=True: in its LAST run of phone tokens, the one a self-correction ends on): a MAXIMAL run of consecutive phone tokens (digit groups, number words, repeat words, the
    joiners inside number words) reads exactly as `exp` (a country code may replace a trunk 0).  Words that read as nothing break a run.  No
    sub-run is tried: an added or dropped first digit is an error (audit-2 review: trimming the run forgave a wrong 11-digit number)."""
    exp = re.sub(r"\D", "", exp)
    if not exp:
        return False
    t = _table(lang)
    w = expand_fused(toks_raw(text), lang)
    known = {x for k in t for x in k} | set(REPEAT)
    runs, cur = [], []
    for x in w:
        if dig(x) is not None or x in known:
            cur.append(x)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    variants = [exp] + ([cc + exp[1:] for cc in TRUNK_CC] if exp.startswith("0") else [])
    return any(phone_reads(run, lang, e) for run in (runs[-1:] if last else runs) for e in variants)    # last: only the final run of phone tokens


def _as_words(x) -> list:
    return toks_raw(x) if isinstance(x, str) else list(x)


def phone_options(x, lang: str) -> set:
    """Digit strings a text (or token list) can mean as a phone number, in ANY style (digit groups, number words, mixed)."""
    return phone_digit_set(_as_words(x), lang)


def phone_matches(x, lang: str, exp: str) -> bool:
    """The text (or token list) reads, in any style, as exactly the phone number `exp` (strict first digit; a trunk 0 may be a country code)."""
    exp = re.sub(r"\D", "", exp)
    if not exp:
        return False
    words = _as_words(x)
    variants = [exp] + ([cc + exp[1:] for cc in TRUNK_CC] if exp.startswith("0") else [])
    return any(phone_reads(words, lang, e) for e in variants)


# ------------------------------------------------------------------------------------------------ digit groups (P7 slots)
_THOUSANDS_SEP = {"en": ",", "ja": ",", "ko": ",", "zh": ",", "he": ",", "th": ",", "hi": ",", "ms": ",", "tl": ","}   # "," groups thousands; elsewhere "." does


def digit_readings(text: str, lang: str = "en") -> list:
    """Readings of the written-digit tokens of a text as LITERAL strings, for a matcher that joins digits (phone slots keep a leading zero;
    "300.000" / "300,000" / "300 000" / "1 200 250" are the thousands-grouped number).  Returns a list of readings, each a list of strings
    in text order: [as written: groups separated by white space or an apostrophe, a token such as "20,5" or "15.5" kept whole],
    [thousands groupings merged].  Only a separator that is a thousands separator IN THE LANGUAGE merges ("." in de / es / it ..., "," in en;
    a decimal "20,5" / "15.5" is never split into 20 and 5, which would credit a price of 20.05 or a time of 15:05).  The caller adds the
    readings to those a slot is searched in; a slot is right when ANY reading has it right, so this only adds right answers a reader missed."""
    sep = _THOUSANDS_SEP.get(lang, ".")
    s = text.replace("’", "'")
    plain, merged = [], []
    for g in re.findall(r"(?<![\w])\+?\d[\d.,\s\u202f\u00a0']*\d|(?<![\w])\d", s):
        pieces = [p for p in re.split(r"[\s\u202f\u00a0']+", g) if p]
        plain += pieces
        if len(pieces) > 1 and all(p.isdigit() for p in pieces) and 1 <= len(pieces[0]) <= 3 and all(len(p) == 3 for p in pieces[1:]):
            merged.append("".join(pieces))                           # "1 200 250", "300 000": one number
            continue
        for p in pieces:
            if re.fullmatch(rf"\d{{1,3}}(?:{re.escape(sep)}\d{{3}})+", p):
                merged.append(p.replace(sep, ""))                    # "300.000" (de) / "300,000" (en)
            else:
                merged.append(p)
    out = [plain]
    if merged != plain:
        out.append(merged)
    return out


# ------------------------------------------------------------------------------------------------ fractions
_VULGAR = {"½": Decimal("0.5"), "¼": Decimal("0.25"), "¾": Decimal("0.75"), "⅓": Decimal("1") / Decimal(3), "⅔": Decimal("2") / Decimal(3),
           "⅕": Decimal("0.2"), "⅖": Decimal("0.4"), "⅗": Decimal("0.6"), "⅘": Decimal("0.8"), "⅛": Decimal("0.125"),
           "⅜": Decimal("0.375"), "⅝": Decimal("0.625"), "⅞": Decimal("0.875")}


def _q(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.0001"))


def _number_values(text: str, lang: str) -> list:
    """[(Num, value)] of the words and digits of a text through the public reader (a placeholder when it is unavailable)."""
    import number_outcomes as N

    out = []
    for n in N.extract_numbers(text, lang, keep_weak=True):
        if n.vals:
            out.append((n, min(n.vals)))
    return out


_EN_ZERO_POINT = re.compile(r"\b(?:oh|o|nought|naught|zero)\s+point\b", re.I)
_EN_FRACTION_WORDS = {"half": Decimal("0.5"), "halves": Decimal("0.5"), "third": Decimal(1) / Decimal(3), "thirds": Decimal(1) / Decimal(3),
                      "quarter": Decimal("0.25"), "quarters": Decimal("0.25"), "fourth": Decimal("0.25"), "fourths": Decimal("0.25"),
                      "fifth": Decimal("0.2"), "fifths": Decimal("0.2"), "eighth": Decimal("0.125"), "eighths": Decimal("0.125"),
                      "sixth": Decimal(1) / Decimal(6), "sixths": Decimal(1) / Decimal(6), "seventh": Decimal(1) / Decimal(7),
                      "sevenths": Decimal(1) / Decimal(7), "ninth": Decimal(1) / Decimal(9), "ninths": Decimal(1) / Decimal(9),
                      "tenth": Decimal("0.1"), "tenths": Decimal("0.1")}
_DE_NL_HALF = re.compile(r"\b(\w+?)(?:einhalb|enhalf|ënhalf)\b")

# Fraction words of the other languages (folded spelling): the DENOMINATOR word -> denominator.  A fraction is [whole joiner] numerator
# denominator ("tres cuartos", "deux et trois quarts", "zwei und drei Viertel", "driekwart", "due e un terzo"); the numerator is a number word,
# an article (= 1) or digits.  A bare denominator word ("cuarto", "halb") is no fraction: it is an ordinal, a clock word or a half already read
# as part of a number by the reader ("dos y medio" = 2.5).
def _den_words(spec: str) -> dict:
    out = {}
    for item in spec.split():
        word, den = item.split("=")
        out[word] = int(den)
    return out


_WORD_DEN = {
    "es": _den_words("medio=2 media=2 medios=2 medias=2 mitad=2 tercio=3 tercios=3 cuarto=4 cuartos=4 quinto=5 quintos=5 sexto=6 sextos=6 "
                     "septimo=7 septimos=7 octavo=8 octavos=8 noveno=9 novenos=9 decimo=10 decimos=10"),
    "fr": _den_words("demi=2 demie=2 demis=2 demies=2 moitie=2 tiers=3 quart=4 quarts=4 cinquieme=5 cinquiemes=5 sixieme=6 sixiemes=6 "
                     "septieme=7 septiemes=7 huitieme=8 huitiemes=8 neuvieme=9 neuviemes=9 dixieme=10 dixiemes=10"),
    "de": _den_words("halb=2 halbe=2 halben=2 halbes=2 halfte=2 drittel=3 dritteln=3 viertel=4 vierteln=4 funftel=5 funfteln=5 sechstel=6 "
                     "sechsteln=6 siebtel=7 siebteln=7 siebentel=7 achtel=8 achteln=8 neuntel=9 neunteln=9 zehntel=10 zehnteln=10"),
    "nl": _den_words("half=2 halve=2 helft=2 derde=3 derden=3 vierde=4 vierden=4 kwart=4 kwarten=4 vijfde=5 vijfden=5 zesde=6 zesden=6 "
                     "zevende=7 zevenden=7 achtste=8 achtsten=8 negende=9 negenden=9 tiende=10 tienden=10"),
    "pt": _den_words("meio=2 meia=2 meios=2 meias=2 metade=2 terco=3 tercos=3 quarto=4 quartos=4 quinto=5 quintos=5 sexto=6 sextos=6 "
                     "setimo=7 setimos=7 oitavo=8 oitavos=8 nono=9 nonos=9 decimo=10 decimos=10"),
    "it": _den_words("mezzo=2 mezza=2 mezzi=2 mezze=2 meta=2 terzo=3 terzi=3 quarto=4 quarti=4 quinto=5 quinti=5 sesto=6 sesti=6 settimo=7 "
                     "settimi=7 ottavo=8 ottavi=8 nono=9 noni=9 decimo=10 decimi=10"),
}
_WHOLE_JOIN = {"und", "y", "e", "et", "en", "ed", "and", "con", "com"}   # "dos y tres cuartos", "zwei und drei Viertel", "drie en een half"
_FUSED_LANGS = ("de", "nl")                                              # "dreiviertel", "driekwart", "tweederde" (numerator + denominator in one word)


def _int_run(tokens: list, end: int, lang: str):
    """(value, start) of the LONGEST run of 1-3 tokens ending just before `end` that reads as ONE whole number 0-999 (a number word, an
    article = 1, digits), else None."""
    import number_outcomes as N

    for n in (3, 2, 1):
        lo = end - n
        if lo < 0:
            continue
        run = tokens[lo:end]
        if run[0] in _WHOLE_JOIN or run[-1] in _WHOLE_JOIN:
            continue
        if n == 1 and run[0].isdigit():
            v = int(run[0])
            return (v, lo) if v <= 999 else None
        if any(t.isdigit() for t in run):
            continue
        ns = N.extract_numbers(" ".join(run), lang, keep_weak=True)
        if len(ns) == 1 and ns[0].vals and ns[0].start == 0 and ns[0].end == n - 1:
            v = min(ns[0].vals)
            if v == int(v) and 0 <= v <= 999:
                return int(v), lo
    return None


def _word_fractions(text: str, lang: str) -> set:
    """The values of the word fractions of a non-English text ("tres cuartos" 0.75, "dos y tres cuartos" 2.75, "drei Viertel", "driekwart",
    "deux tiers", "um terço", "un demi").  Empty when the text holds none."""
    den = _WORD_DEN.get(lang)
    if not den:
        return set()
    tokens = toks(text)
    out = set()
    for i, tok in enumerate(tokens):
        d, lo, numer = den.get(tok), None, None
        if d is None and lang in _FUSED_LANGS:                       # numerator and denominator in one word
            for word, dd in sorted(den.items(), key=lambda kv: -len(kv[0])):
                if len(tok) > len(word) and tok.endswith(word):
                    import number_outcomes as N

                    pv = N.parse_value(tok[:-len(word)], lang)
                    if pv is not None and pv == int(pv) and 0 < pv <= 99:
                        d, lo, numer = dd, i, int(pv)
                    break
        if d is None:
            continue
        if numer is None:
            got = _int_run(tokens, i, lang)
            if got is None:
                w = _int_run(tokens, i - 1, lang) if i >= 2 and tokens[i - 1] in _WHOLE_JOIN and d in (2, 4) else None
                if w is not None:                                       # "dos y cuarto", "deux et quart": the numerator one is left out
                    out.add(_q(Decimal(w[0]) + Decimal(1) / Decimal(d)))
                continue
            numer, lo = got
        if numer == 0:
            continue
        whole = 0
        if lo >= 2 and tokens[lo - 1] in _WHOLE_JOIN:                   # "dos y tres cuartos": the whole number before the joiner
            w = _int_run(tokens, lo - 1, lang)
            if w is not None:
                whole = w[0]
        out.add(_q(Decimal(whole) + Decimal(numer) / Decimal(d)))
    return out


def fraction_values(text: str, lang: str) -> set:
    """The value(s) a text means as ONE number with a fraction or decimal part: "2 1/2" and "2½" -> {2.5} (never also 0.5 or 2), "1/2", "0.75",
    "0,75", ".5", "two and a half", "two and three quarters", "three quarters", "a third", "oh point five" (=0.5, not 5), "zero point seven five",
    "dos y medio", "drei einhalb", "dos coma cinco".  When the text holds a fraction / decimal value, ONLY those values are returned; a text
    with no fraction part returns its plain numbers, so a caller comparing two sets cannot be matched through a shared integer part
    (audit-2 review: "two" matched "2½" through the 2)."""
    t = text.strip()
    frac, whole_nums = set(), set()
    masked = t
    # mixed digit forms first, and mask them so their pieces are not read again ("2 1/2" must not also give 1/2)
    def mixed(m):
        whole = Decimal(m.group(1))
        if m.group(2) and int(m.group(3)):
            frac.add(_q(whole + Decimal(m.group(2)) / Decimal(m.group(3))))
        elif m.group(4):
            frac.add(_q(whole + _VULGAR[m.group(4)]))
        return " "
    masked = re.sub(r"(?<![\d/.,])(\d+)\s*(?:(\d+)\s*/\s*(\d+)|([½¼¾⅓⅔⅕⅖⅗⅘⅛⅜⅝⅞]))(?![\d/])", mixed, masked)
    for m in list(re.finditer(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])", masked)):
        if int(m.group(2)):
            frac.add(_q(Decimal(m.group(1)) / Decimal(m.group(2))))
    masked = re.sub(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])", " ", masked)
    for ch, v in _VULGAR.items():
        if ch in masked:
            frac.add(_q(v))
            masked = masked.replace(ch, " ")
    for m in re.finditer(r"(?<![\d.,])(\d*)[.,](\d+)(?![\d])", masked):
        try:
            frac.add(_q(Decimal((m.group(1) or "0") + "." + m.group(2))))
        except Exception:  # noqa: BLE001
            pass
    masked = re.sub(r"(?<![\d.,])(\d*)[.,](\d+)(?![\d])", " ", masked)
    f = fold(masked)
    if lang == "en":
        f = _EN_ZERO_POINT.sub("zero point", f)
        w = f.replace("-", " ").split()
        for i, tok in enumerate(w):
            tok = tok.strip(",.;")
            if tok not in _EN_FRACTION_WORDS:
                continue
            plural = tok.endswith("s") and tok != "halves"
            num, idx = Decimal(1), i
            if i > 0:
                prev = w[i - 1].strip(",.;")
                if prev in ("a", "an", "one"):
                    idx = i - 1
                elif _word_int(prev) is not None and plural:
                    num, idx = Decimal(_word_int(prev)), i - 1
                elif prev == "half":
                    idx = i - 1
            base = _EN_FRACTION_WORDS[tok]
            unit = _q(num / Decimal(3)) if tok == "thirds" else base * num
            whole = Decimal(0)
            if idx >= 2 and w[idx - 1] == "and" and _word_int(w[idx - 2]) is not None:       # "two and a half", "two and three quarters"
                whole = Decimal(_word_int(w[idx - 2]))
            frac.add(_q(whole + unit))
        masked = f
    else:
        frac |= _word_fractions(masked, lang)
    for n, v in _number_values(f if lang == "en" else masked, lang):
        try:
            d = _q(Decimal(v))
        except Exception:  # noqa: BLE001
            continue
        (frac if d != d.to_integral_value() else whole_nums).add(d)
    return frac or whole_nums


@functools.lru_cache(maxsize=None)
def _en_small():
    import number_outcomes as N

    return {w: int(min(N.extract_numbers(w, "en")[0].vals)) for w in
            "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty".split()}


def _word_int(w: str):
    w = w.strip(",.;")
    if w.isdigit():
        return int(w)
    return _en_small().get(w)


# ------------------------------------------------------------------------------------------------ clock times
_AMPM = r"(?:a\.?m\.?|p\.?m\.?)"


# relative / spoken clock forms per language: element = word alternatives "a|b" (consumed as consecutive tokens, "a b" = two
# tokens), "H" (an hour 1-24) or "M" (minutes 1-59); outcome = (hour offset, minute) as a function of (H, M)
_H, _M = "H", "M"
_CLOCK_RULES = {
    "en": [(["half past", _H], lambda h, m: (h, 30)), (["quarter past|quarter after", _H], lambda h, m: (h, 15)),
           (["quarter to|quarter of|quarter till", _H], lambda h, m: (h - 1, 45)), ([_M, "past|after", _H], lambda h, m: (h, m)),
           ([_M, "to|of|till", _H], lambda h, m: (h - 1, 60 - m)), ([_H, "oclock"], lambda h, m: (h, 0), True),
           ([_H, "oh|o|zero", _M], lambda h, m: (h, m) if m < 10 else None), ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
    "de": [(["halb", _H], lambda h, m: (h - 1, 30)), (["viertel nach", _H], lambda h, m: (h, 15)), (["viertel vor", _H], lambda h, m: (h - 1, 45)),
           ([_M, "nach", _H], lambda h, m: (h, m)), ([_M, "vor", _H], lambda h, m: (h - 1, 60 - m)),
           ([_H, "uhr", _M], lambda h, m: (h, m)), ([_H, "uhr"], lambda h, m: (h, 0), True), ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
    "nl": [(["half", _H], lambda h, m: (h - 1, 30)), (["kwart over", _H], lambda h, m: (h, 15)), (["kwart voor", _H], lambda h, m: (h - 1, 45)),
           ([_M, "over", _H], lambda h, m: (h, m)), ([_M, "voor", _H], lambda h, m: (h - 1, 60 - m)),
           ([_H, "uur", _M], lambda h, m: (h, m)), ([_H, "uur"], lambda h, m: (h, 0), True), ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
    "es": [([_H, "y media"], lambda h, m: (h, 30)), ([_H, "y cuarto"], lambda h, m: (h, 15)), ([_H, "menos cuarto"], lambda h, m: (h - 1, 45)),
           ([_H, "y", _M], lambda h, m: (h, m)), ([_H, "horas y|hora y", _M], lambda h, m: (h, m)), ([_H, "menos", _M], lambda h, m: (h - 1, 60 - m)), ([_H, "en punto"], lambda h, m: (h, 0), True),
           ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
    "fr": [([_H, "heure et demie|heures et demie|heure et demi|heures et demi|et demie|et demi"], lambda h, m: (h, 30)),
           ([_H, "heure et quart|heures et quart|et quart"], lambda h, m: (h, 15)),
           ([_H, "heure moins le quart|heures moins le quart|moins le quart|moins quart"], lambda h, m: (h - 1, 45)),
           ([_H, "heure|heures|h", _M], lambda h, m: (h, m)), ([_H, "heure moins|heures moins|moins", _M], lambda h, m: (h - 1, 60 - m)),
           ([_H, "heure|heures|h"], lambda h, m: (h, 0), True)],
    "it": [([_H, "e mezza|e mezzo"], lambda h, m: (h, 30)), ([_H, "e un quarto|e quarto"], lambda h, m: (h, 15)),
           ([_H, "meno un quarto|meno quarto"], lambda h, m: (h - 1, 45)), ([_H, "e", _M], lambda h, m: (h, m)),
           ([_H, "meno", _M], lambda h, m: (h - 1, 60 - m)), ([_H, "in punto"], lambda h, m: (h, 0), True), ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
    "pt": [([_H, "e meia|e meio"], lambda h, m: (h, 30)), ([_H, "e um quarto|e quinze"], lambda h, m: (h, 15)),
           ([_H, "menos um quarto|menos quinze"], lambda h, m: (h - 1, 45)), ([_H, "e", _M], lambda h, m: (h, m)),
           ([_H, "menos", _M], lambda h, m: (h - 1, 60 - m)), ([_H, "horas|hora|h", _M], lambda h, m: (h, m)),
           ([_H, "em ponto"], lambda h, m: (h, 0), True), ([_H, _M], lambda h, m: (h, m) if m >= 10 else None)],
}


def _tok_num(tokens: list, i: int, lang: str, lo: int, hi: int) -> list:
    """[(value, next index)] for the longest 1-3 token run at i that reads as a whole number in [lo, hi]."""
    import number_outcomes as N

    out = []
    for n in (1, 2, 3):
        run = tokens[i:i + n]
        if len(run) < n:
            break
        if n > 1 and run[-1] in CONNECT:            # a joiner at the end belongs to the NEXT word ("cinco y media")
            continue
        if n == 1 and run[0].isdigit():
            v = int(run[0])
        else:
            if any(t.isdigit() for t in run):
                continue
            ns = N.extract_numbers(" ".join(run), lang, keep_weak=True)
            # ONE number that covers every token of the run (parse_value alone ignores words around the number: "cinq heures et" -> 5)
            pv = min(ns[0].vals) if len(ns) == 1 and ns[0].vals and ns[0].start == 0 and ns[0].end == n - 1 else None
            v = int(pv) if pv is not None and pv == int(pv) else None
        if v is not None and lo <= v <= hi:
            out.append((v, i + n))
    return out[-1:]                      # the LONGEST reading only: "twenty five" is 25, never 20 followed by a stray 5 (audit-2 review: "six twenty-five" read 6:20 too)


def _is_num_tok(tok: str, lang: str) -> bool:
    import number_outcomes as N

    return tok.isdigit() or (tok.isalpha() and N.parse_value(tok, lang) is not None)


def _clock_word_forms(tokens: list, lang: str) -> set:
    full, bare = set(), {}
    for rule in _CLOCK_RULES.get(lang, ()):
        elems, fn = rule[0], rule[1]
        hour_only = len(rule) > 2
        starts_with_number = elems[0] in (_H, _M)
        for i in range(len(tokens)):
            if starts_with_number and i > 0 and _is_num_tok(tokens[i - 1], lang):
                continue                 # the match would begin inside a number ("twenty-five to six" is not "five to six")
            states = [(i, None, None)]
            for el in elems:
                nxt = []
                for pos, h, m in states:
                    if el == _H:
                        nxt += [(j, v, m) for v, j in _tok_num(tokens, pos, lang, 1, 24)]
                    elif el == _M:
                        nxt += [(j, h, v) for v, j in _tok_num(tokens, pos, lang, 1, 59)]
                    else:
                        for alt in el.split("|"):
                            ws = alt.split()
                            if tokens[pos:pos + len(ws)] == ws:
                                nxt.append((pos + len(ws), h, m))
                states = nxt
                if not states:
                    break
            for end, h, m in states:
                if hour_only and _tok_num(tokens, end, lang, 0, 59):      # "14 heures 38": the minutes follow, this is no bare hour
                    continue
                r = fn(h, m)
                if r is None or not 0 <= r[1] <= 59:
                    continue
                if hour_only:
                    bare.setdefault(i, set()).add((r[0] % 12, r[1]))
                else:
                    full.add((r[0] % 12, r[1]))
                    full.add(("start", i))
    starts = {x[1] for x in full if x[0] == "start"}
    out = {x for x in full if x[0] != "start"}
    for i, v in bare.items():
        if i not in starts:              # "cinq heures et demie" is 5:30, not also 5:00
            out |= v
    return out


def clock_values(text: str, lang: str) -> set:
    """(hour % 12, minute) a text can mean as a time of day: "5:30", "17h30", "5.30 pm", "5 pm", "10 o'clock", "five thirty",
    "half past five", "quarter to six", "ten to six", "halb sechs", "cinco y media", "dix heures moins le quart", ... (rule table above;
    the public reader's relative-clock reading is added for the languages it knows)."""
    out = set()
    t = text.strip().lower()
    for m in re.finditer(r"(?<![\d:.,])(\d{1,2})\s*(?:[:h]|\.(?=\d{2}\b))\s*(\d{2})(?!\d)", t):
        h, mi = int(m.group(1)), int(m.group(2))
        if h <= 24 and mi <= 59:
            out.add((h % 12, mi))
    for m in re.finditer(rf"(?<![\d:.,])(\d{{1,2}})\s*(?:{_AMPM}|o['’]?\s?clock|uhr|uur|heures?|horas?|ore|h)\b(?!\s*\d)", t):
        if int(m.group(1)) <= 24:
            out.add((int(m.group(1)) % 12, 0))
    m4 = re.fullmatch(r"(\d{2})(\d{2})", t.strip(" .,"))                        # "1730" alone (inside a sentence "2000" is a year or a count)
    if m4 and int(m4.group(1)) <= 24 and int(m4.group(2)) <= 59:
        out.add((int(m4.group(1)) % 12, int(m4.group(2))))
    import number_outcomes as N

    f = fold(text).replace("-", " ").replace("o'clock", "oclock").replace("o’clock", "oclock")
    tokens = N._TOKEN.findall(N._fold(f))
    out |= _clock_word_forms(tokens, lang)
    if lang != "en":
        ns = N.extract_numbers(text, lang, keep_weak=True)
        for a, b in zip(ns, ns[1:]):
            # adjacent tokens ("cinco treinta"), or the two numbers of ONE relative-clock token ("halb sechs" = 5 + 30)
            if a.vals and b.vals and (b.start == a.end + 1 or a.start == a.end == b.start == b.end):
                for h in a.vals:
                    for mi in b.vals:
                        if h == int(h) and mi == int(mi) and 0 < h <= 24 and 0 <= mi <= 59:
                            out.add((int(h) % 12, int(mi)))
    # a bare hour ("14 Uhr", "5 heures") is dropped when the same text says that hour WITH minutes ("14 Uhr dreißig")
    return {(h, m) for h, m in out if not (m == 0 and any(hh == h and mm != 0 for hh, mm in out))}


def group_matches(kind: str, ref: str, hyp: str, lang: str) -> bool:
    """The two texts mean the same phone number / clock time / fraction (exact, strict first digit for phones)."""
    if kind == "phone":
        want = re.sub(r"\D", "", ref) if re.search(r"\d", ref) and not re.search(r"[^\W\d_]{3,}", ref) else None
        exp = {want} if want else {g for g in phone_options(ref, lang) if g}
        return any(phone_matches(hyp, lang, e) for e in exp)
    if kind == "clock":
        a, b = clock_values(ref, lang), clock_values(hyp, lang)
        if a & b:
            return True
        # a bare hour ("at 10" for "ten o'clock") is the hour with no minutes: right only when the reference minutes are 0
        nv = _number_values(hyp, lang)
        if b or len(nv) != 1:                     # the text must be that one number: "5 hours 30 minutes" / "10 minutes past 5" are no bare hour
            return False
        n, v = nv[0]
        return n.form == "digits" and v == int(v) and 0 < v <= 24 and any(mi == 0 and h == int(v) % 12 for h, mi in a)
    if kind == "frac":
        a, b = fraction_values(ref, lang), fraction_values(hyp, lang)
        return bool(a & b)
    raise ValueError(kind)
