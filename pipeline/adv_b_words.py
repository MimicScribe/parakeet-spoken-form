"""Number words for the adversarial pack (adv_b): digits, pairs and cardinals in spoken form for en es fr de pt it nl.

en / es / fr come from edge_a_words; de / pt / it / nl from the grammar table of ml_cands, with de / nl / it compounds
split into spaced morphemes (the label convention: "drei und zwanzig", "twee en dertig", Italian elided forms whole,
final "tre" unaccented). Strict spoken form: no digit, no hyphen. Pure functions."""
import functools

import compound_split as CS
import edge_a_words as W
import ml_cands as M

LANGS = ("en", "es", "fr", "de", "pt", "it", "nl")
BIG = ("en", "es", "fr")
SPLIT = {"de": "de", "nl": "nl", "it": "it-"}

DIGITS = {
    "en": "zero one two three four five six seven eight nine".split(),
    "es": "cero uno dos tres cuatro cinco seis siete ocho nueve".split(),
    "fr": "zéro un deux trois quatre cinq six sept huit neuf".split(),
    "de": "null eins zwei drei vier fünf sechs sieben acht neun".split(),
    "pt": "zero um dois três quatro cinco seis sete oito nove".split(),
    "it": "zero uno due tre quattro cinque sei sette otto nove".split(),
    "nl": "nul een twee drie vier vijf zes zeven acht negen".split(),
}
# the word for a spoken zero inside a phone number besides the plain one ("oh" in English)
OH = {"en": "oh"}
DOUBLE = {"en": "double", "es": "doble", "fr": "double"}
TRIPLE = {"en": "triple", "es": "triple", "fr": "triple"}
TREBLE = {"en": "treble"}


@functools.lru_cache(maxsize=None)
def card(n: int, lang: str) -> str:
    """The cardinal of n as a count; de / nl / it compounds are spaced."""
    if lang in BIG:
        return W.card(n, lang)
    sp = M.spans(str(n), lang)
    if not (sp and sp[0].cands and sp[0].text == str(n)):
        raise ValueError((n, lang))
    out = []
    for w in sp[0].cands[0].strip().split():
        out += CS.split_token(w, SPLIT[lang]) if lang in SPLIT else [w]
    s = " ".join(out)
    if lang == "it":
        s = s.replace("tré", "tre")
    return s


def digit(d: int, lang: str, zero: str = "plain") -> str:
    """One spoken digit; zero="oh" gives the English "oh"."""
    if d == 0 and zero == "oh" and lang in OH:
        return OH[lang]
    return DIGITS[lang][d]


def pair(s: str, lang: str, zero: str = "plain") -> str:
    """Two digits ("23", "05", "40") spoken as a pair: a number, or zero + digit for a leading zero."""
    assert len(s) == 2 and s.isdigit(), s
    if s[0] == "0":
        return digit(0, lang, zero) + " " + digit(int(s[1]), lang, zero)
    return card(int(s), lang)


def digits(s: str, lang: str, zero: str = "plain") -> str:
    return " ".join(digit(int(ch), lang, zero) for ch in s)


def runs(s: str) -> list:
    """[(digit, run length)] of a digit string."""
    out = []
    for ch in s:
        if out and out[-1][0] == ch:
            out[-1][1] += 1
        else:
            out.append([ch, 1])
    return [(c, n) for c, n in out]


def digits_rep(s: str, lang: str, zero: str = "plain", triple: str = "triple") -> str:
    """Digits with the repeat words of the language: a run of 2 -> "double five", of 3 -> "triple/treble five"
    (runs of 4+ are split: a triple then the rest)."""
    out = []
    for ch, n in runs(s):
        w = digit(int(ch), lang, zero)
        while n > 0:
            if n >= 3 and triple and lang in TRIPLE:
                out.append(f"{(TREBLE if triple == 'treble' else TRIPLE)[lang]} {w}")
                n -= 3
            elif n >= 2 and lang in DOUBLE:
                out.append(f"{DOUBLE[lang]} {w}")
                n -= 2
            else:
                out.append(w)
                n -= 1
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ ordinals (regnal)
ORD = {
    "en": "zero first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
          "sixteenth seventeenth eighteenth nineteenth twentieth".split(),
    "es": "cero primero segundo tercero cuarto quinto sexto séptimo octavo noveno décimo".split(),
    "fr": "zéro premier deuxième troisième quatrième cinquième sixième septième huitième neuvième dixième".split(),
}
ORD_TENS_EN = {20: "twentieth", 30: "thirtieth"}


def ordinal_en(n: int) -> str:
    if n <= 20:
        return ORD["en"][n]
    if n in ORD_TENS_EN:
        return ORD_TENS_EN[n]
    t, u = divmod(n, 10)
    return f"{card(t * 10, 'en')} {ORD['en'][u]}"


MONTHS = {
    "en": "January February March April May June July August September October November December".split(),
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(),
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
}
