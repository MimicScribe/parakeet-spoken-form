"""Number words for the edge-case pack (edge_a): whole numbers and decimals in spoken form for en / es / fr.

English comes from readings.cardinal, Spanish and French from the grammar table of ml_cands (the reading the labeller
would give the same number in a count context). Strict spoken form: no digit, no hyphen ("twenty three", "treinta y
cinco", "quatre vingt dix"). Pure functions; tests in test_edge_a.py.
"""
import functools

import ml_cands as M
import readings as R

LANGS = ("en", "es", "fr")
POINT = {"en": "point", "es": "coma", "fr": "virgule"}
ZERO = {"en": "oh", "es": "cero", "fr": "zéro"}
DIGIT = {"en": "zero one two three four five six seven eight nine".split(),
         "es": "cero uno dos tres cuatro cinco seis siete ocho nueve".split(),
         "fr": "zéro un deux trois quatre cinq six sept huit neuf".split()}
PERCENT = {"en": "percent", "es": "por ciento", "fr": "pour cent"}


@functools.lru_cache(maxsize=None)
def card(n: int, lang: str) -> str:
    """The cardinal of n as a count (masculine, as the table gives it)."""
    if lang == "en":
        return R.cardinal(n)
    sp = M.spans(str(n), lang)
    if sp and sp[0].cands and sp[0].text == str(n):
        w = sp[0].cands[0].strip()
        if not any(ch.isdigit() for ch in w) and "zéro zéro" not in w and "uno cero" not in w:
            return w
    return _compose(n, lang)


def _compose(n: int, lang: str) -> str:
    """Whole numbers the table lacks (fr >= 30,000, es / fr millions): thousands and units by recursion."""
    if n >= 1_000_000:
        q, r = divmod(n, 1_000_000)
        head = (("un millón" if q == 1 else f"{card(q, lang)} millones") if lang == "es"
                else ("un million" if q == 1 else f"{card(q, lang)} millions"))
        return head + (" " + card(r, lang) if r else "")
    if n >= 1000:
        q, r = divmod(n, 1000)
        head = "mil" if q == 1 else f"{card(q, lang)} mil" if lang == "es" or lang == "en" else (
            "mille" if q == 1 else f"{card(q, lang)} mille")
        return head + (" " + card(r, lang) if r else "")
    raise ValueError(n)


def frac(digits: str, lang: str) -> str:
    """The digits after a decimal point, spoken. en: digit by digit ('oh' for 0); es / fr: one digit by name, two digits
    as a number (a leading zero is 'cero' / 'zéro' + the digit), three or more digit by digit (a rate: 'uno coma cero
    ocho cuatro')."""
    if lang == "en" or len(digits) >= 3:
        return " ".join((ZERO[lang] if lang == "en" else DIGIT[lang][0]) if d == "0" else DIGIT[lang][int(d)] for d in digits)
    if len(digits) == 1:
        return DIGIT[lang][int(digits)]
    if digits[0] == "0":
        return DIGIT[lang][0] + " " + DIGIT[lang][int(digits[1])]
    return card(int(digits), lang)


def dec(a: int, digits: str, lang: str) -> str:
    """'three point two', 'tres coma dos', 'trois virgule deux' (a = integer part, digits = fraction digits)."""
    whole = "zero" if (lang == "en" and a == 0) else DIGIT[lang][0] if a == 0 else card(a, lang)
    return f"{whole} {POINT[lang]} {frac(digits, lang)}"


def a_or_one(n: int, lang: str) -> str:
    """en: 'a' for 1 before a scale word ('a billion'); other languages: the cardinal."""
    return "a" if (lang == "en" and n == 1) else card(n, lang)
