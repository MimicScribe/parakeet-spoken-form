#!/usr/bin/env python3
"""Per-language NUMBER OUTCOME scorer: how does a hypothesis write each number of the reference?

Outcome per reference number (exactly one):
  WORDS_RIGHT_VALUE   the number is written as words and the parsed value equals the reference value
  DIGITS_RIGHT_VALUE  written as digits, same value
  WRONG_VALUE         something number-like stands in its place (words or digits) with another value
  DROPPED             nothing number-like stands in its place

The reference may carry digits ("12,50"), spoken words ("douze virgule cinq"), or both: one extractor reads both
kinds on both sides, so a spoken-form model is scored by VALUE, never by whether it used digits.

Word parsers (cardinals, ordinals, decimals, clock/currency pieces) exist for en de es fr it nl pt. The other 18
languages read number words (all case / gender / number forms) through the training pipeline's form tables
(ml_multi + ml_slav in pipeline/, or $SPOKEN_FORM_PIPELINE); without them they fall back to
DIGIT-ONLY scoring, where a model that writes words scores DROPPED. Every row and summary carries `mode`
("words+digits", "words+digits:ext-<table hash>" or "digits_only") so a fallback can never be read as a real drop.
A broken number word in the hypothesis ("zweitausendzwanz", "thousan") is a WRONG value, not a drop.

Known limits (by design, kept small): a number is compared by value only (units/currency symbols are not checked);
"10:00" reads as the hour only; a spoken digit-by-digit phone number is several numbers; plain-space digit grouping
("5 000") is two numbers (NBSP / narrow NBSP / apostrophe grouping is one); alignment is by value (LCS) and then in
order inside each gap, so an unrelated inserted number inside a gap can be paired as WRONG_VALUE.

Usage:  number_outcomes.py --lang fr --ref "douze virgule cinq" --hyp "12,5"      (prints the rows)
        python3 -m unittest eval/quality/test_number_outcomes.py
"""

import argparse
import collections
import json
import os
import re
import sys
import unicodedata
from decimal import Decimal

OUTCOMES = ("WORDS_RIGHT_VALUE", "DIGITS_RIGHT_VALUE", "WRONG_VALUE", "DROPPED")
_PIPE = os.environ.get("SPOKEN_FORM_PIPELINE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))
if os.path.isdir(_PIPE) and _PIPE not in sys.path:
    sys.path.append(_PIPE)  # ml_multi / ml_slav word-form tables for the 18 non-core languages (read only)

# ===== BEGIN CORE (copied verbatim into the public training scorer; edit here, then re-copy: see test_number_outcomes.py) =====
WORD_LANGS = ("en", "de", "es", "fr", "it", "nl", "pt")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("ß", "ss").replace("œ", "oe").replace("’", "'")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")
    return s.lower()


def _lex(spec: dict) -> dict:
    """spec: kind -> {value: "w1 w2 ..."}; returns word -> (kind, value, is_ordinal)."""
    out = {}
    for kind, vals in spec.items():
        for v, words in vals.items():
            for w in words.split():
                out[w] = (kind, v, False)
    return out


def _ords(lex: dict, table: dict) -> None:
    """table: kind -> {value: "stem stem ..."}; every stem is also written with the given endings (see call sites)."""
    for kind, vals in table.items():
        for v, words in vals.items():
            for w in words.split():
                lex.setdefault(w, (kind, v, True))


def _endings(words: str, ends: str) -> str:
    return " ".join(w + e for w in words.split() for e in ends.split("|"))


def _build() -> tuple[dict, dict, dict]:
    """Returns (LEX, CONN, DEC) per language. Kinds: Z zero, U 1-9, T 10-19 (and fused 21-29), D tens, H hundred multiplier,
    X lexical hundreds, S scale, C connector, P decimal marker."""
    L, C, P = {}, {}, {}
    # --- English
    en = _lex({"Z": {0: "zero"}, "U": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"},
               "T": {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"},
               "D": {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"},
               "H": {100: "hundred hundreds"}, "S": {10**3: "thousand thousands", 10**6: "million millions", 10**9: "billion billions", 10**12: "trillion"}})
    _ords(en, {"U": {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"},
               "T": {10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth"},
               "D": {20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth"},
               "H": {100: "hundredth"}, "S": {10**3: "thousandth", 10**6: "millionth", 10**9: "billionth"}})
    L["en"], C["en"], P["en"] = en, "and", "point"
    # --- German
    de = _lex({"Z": {0: "null"}, "U": {1: "ein eins eine einen einem einer eines", 2: "zwei zwo", 3: "drei", 4: "vier", 5: "funf", 6: "sechs", 7: "sieben", 8: "acht", 9: "neun"},
               "T": {10: "zehn", 11: "elf", 12: "zwolf", 13: "dreizehn", 14: "vierzehn", 15: "funfzehn", 16: "sechzehn", 17: "siebzehn", 18: "achtzehn", 19: "neunzehn"},
               "D": {20: "zwanzig", 30: "dreissig", 40: "vierzig", 50: "funfzig", 60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig"},
               "H": {100: "hundert"}, "S": {10**3: "tausend", 10**6: "million millionen", 10**9: "milliarde milliarden", 10**12: "billion billionen"}})
    e = "|e|en|er|es|em"
    _ords(de, {"U": {1: _endings("erst", e), 2: _endings("zweit", e), 3: _endings("dritt", e), 4: _endings("viert", e), 5: _endings("funft", e), 6: _endings("sechst", e), 7: _endings("siebt", e), 8: _endings("achte", "|n|r|s|m"), 9: _endings("neunt", e)},
               "T": {10: _endings("zehnt", e), 11: _endings("elft", e), 12: _endings("zwolft", e), 13: _endings("dreizehnt", e), 14: _endings("vierzehnt", e), 15: _endings("funfzehnt", e), 16: _endings("sechzehnt", e), 17: _endings("siebzehnt", e), 18: _endings("achtzehnt", e), 19: _endings("neunzehnt", e)},
               "D": {20: _endings("zwanzigst", e), 30: _endings("dreissigst", e), 40: _endings("vierzigst", e), 50: _endings("funfzigst", e), 60: _endings("sechzigst", e), 70: _endings("siebzigst", e), 80: _endings("achtzigst", e), 90: _endings("neunzigst", e)},
               "H": {100: _endings("hundertst", e)}, "S": {10**3: _endings("tausendst", e), 10**6: _endings("millionst", e)}})
    L["de"], C["de"], P["de"] = de, "und", "komma"
    # --- Dutch
    nl = _lex({"Z": {0: "nul"}, "U": {1: "een", 2: "twee", 3: "drie", 4: "vier", 5: "vijf", 6: "zes", 7: "zeven", 8: "acht", 9: "negen"},
               "T": {10: "tien", 11: "elf", 12: "twaalf", 13: "dertien", 14: "veertien", 15: "vijftien", 16: "zestien", 17: "zeventien", 18: "achttien", 19: "negentien"},
               "D": {20: "twintig", 30: "dertig", 40: "veertig", 50: "vijftig", 60: "zestig", 70: "zeventig", 80: "tachtig", 90: "negentig"},
               "H": {100: "honderd"}, "S": {10**3: "duizend", 10**6: "miljoen", 10**9: "miljard", 10**12: "biljoen"}})
    _ords(nl, {"U": {1: "eerste", 2: "tweede", 3: "derde", 4: "vierde", 5: "vijfde", 6: "zesde", 7: "zevende", 8: "achtste", 9: "negende"},
               "T": {10: "tiende", 11: "elfde", 12: "twaalfde", 13: "dertiende", 14: "veertiende", 15: "vijftiende", 16: "zestiende", 17: "zeventiende", 18: "achttiende", 19: "negentiende"},
               "D": {20: "twintigste", 30: "dertigste", 40: "veertigste", 50: "vijftigste", 60: "zestigste", 70: "zeventigste", 80: "tachtigste", 90: "negentigste"},
               "H": {100: "honderdste"}, "S": {10**3: "duizendste", 10**6: "miljoenste"}})
    L["nl"], C["nl"], P["nl"] = nl, "en", "komma"
    # --- Italian
    it = _lex({"Z": {0: "zero"}, "U": {1: "uno un una", 2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei", 7: "sette", 8: "otto", 9: "nove"},
               "T": {10: "dieci", 11: "undici", 12: "dodici", 13: "tredici", 14: "quattordici", 15: "quindici", 16: "sedici", 17: "diciassette", 18: "diciotto", 19: "diciannove"},
               "D": {20: "venti vent", 30: "trenta trent", 40: "quaranta quarant", 50: "cinquanta cinquant", 60: "sessanta sessant", 70: "settanta settant", 80: "ottanta ottant", 90: "novanta novant"},
               "H": {100: "cento cent"}, "S": {10**3: "mille mila", 10**6: "milione milioni", 10**9: "miliardo miliardi", 10**12: "bilione bilioni"}})
    o = "o|a|i|e"
    _ords(it, {"U": {1: _endings("prim unesim", o), 2: _endings("second duesim", o), 3: _endings("terz treesim", o), 4: _endings("quart quattresim", o), 5: _endings("quint cinquesim", o), 6: _endings("sest seiesim", o), 7: _endings("settim settesim", o), 8: _endings("ottav ottesim", o), 9: _endings("non novesim", o)},
               "T": {10: _endings("decim", o), 11: _endings("undicesim", o), 12: _endings("dodicesim", o), 13: _endings("tredicesim", o), 14: _endings("quattordicesim", o), 15: _endings("quindicesim", o), 16: _endings("sedicesim", o), 17: _endings("diciassettesim", o), 18: _endings("diciottesim", o), 19: _endings("diciannovesim", o)},
               "D": {20: _endings("ventesim", o), 30: _endings("trentesim", o), 40: _endings("quarantesim", o), 50: _endings("cinquantesim", o), 60: _endings("sessantesim", o), 70: _endings("settantesim", o), 80: _endings("ottantesim", o), 90: _endings("novantesim", o)},
               "H": {100: _endings("centesim", o)}, "S": {10**3: _endings("millesim", o), 10**6: _endings("milionesim", o)}})
    L["it"], C["it"], P["it"] = it, "e", "virgola"
    # --- Spanish
    es = _lex({"Z": {0: "cero"}, "U": {1: "uno un una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve"},
               "T": {10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince", 16: "dieciseis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve",
                     21: "veintiuno veintiun veintiuna", 22: "veintidos", 23: "veintitres", 24: "veinticuatro", 25: "veinticinco", 26: "veintiseis", 27: "veintisiete", 28: "veintiocho", 29: "veintinueve"},
               "D": {20: "veinte", 30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"},
               "X": {100: "cien ciento", 200: "doscientos doscientas", 300: "trescientos trescientas", 400: "cuatrocientos cuatrocientas", 500: "quinientos quinientas", 600: "seiscientos seiscientas", 700: "setecientos setecientas", 800: "ochocientos ochocientas", 900: "novecientos novecientas"},
               "S": {10**3: "mil", 10**6: "millon millones", 10**12: "billon billones"}})
    a = "o|a|os|as"
    _ords(es, {"X": {200: _endings("ducentesim", "o|a|os|as"), 300: _endings("tricentesim trecentesim", "o|a|os|as"), 400: _endings("cuadringentesim", "o|a|os|as"), 500: _endings("quingentesim", "o|a|os|as"), 600: _endings("sexcentesim", "o|a|os|as"), 700: _endings("septingentesim", "o|a|os|as"), 800: _endings("octingentesim", "o|a|os|as"), 900: _endings("noningentesim", "o|a|os|as")}})
    _ords(es, {"U": {1: _endings("primer", "|o|a|os|as") + " primer", 2: _endings("segund", a), 3: _endings("tercer", "|o|a|os|as"), 4: _endings("cuart", a), 5: _endings("quint", a), 6: _endings("sext", a), 7: _endings("septim", a), 8: _endings("octav", a), 9: _endings("noven", a)},
               "T": {10: _endings("decim", a), 11: _endings("undecim", a), 12: _endings("duodecim", a), 13: _endings("decimotercer decimotercer", "|o|a"), 14: _endings("decimocuart", a), 15: _endings("decimoquint", a), 16: _endings("decimosext", a), 17: _endings("decimoseptim", a), 18: _endings("decimoctav", a), 19: _endings("decimonoven", a)},
               "D": {20: _endings("vigesim", a), 30: _endings("trigesim", a), 40: _endings("cuadragesim", a), 50: _endings("quincuagesim", a), 60: _endings("sexagesim", a), 70: _endings("septuagesim", a), 80: _endings("octogesim", a), 90: _endings("nonagesim", a)},
               "X": {100: _endings("centesim", a)}, "S": {10**3: _endings("milesim", a)}})
    L["es"], C["es"], P["es"] = es, "y", "coma"
    # --- Portuguese
    pt = _lex({"Z": {0: "zero"}, "U": {1: "um uma", 2: "dois duas", 3: "tres", 4: "quatro", 5: "cinco", 6: "seis", 7: "sete", 8: "oito", 9: "nove"},
               "T": {10: "dez", 11: "onze", 12: "doze", 13: "treze", 14: "catorze quatorze", 15: "quinze", 16: "dezesseis dezasseis", 17: "dezessete dezassete", 18: "dezoito", 19: "dezenove dezanove"},
               "D": {20: "vinte", 30: "trinta", 40: "quarenta", 50: "cinquenta", 60: "sessenta", 70: "setenta", 80: "oitenta", 90: "noventa"},
               "X": {100: "cem cento", 200: "duzentos duzentas", 300: "trezentos trezentas", 400: "quatrocentos quatrocentas", 500: "quinhentos quinhentas", 600: "seiscentos seiscentas", 700: "setecentos setecentas", 800: "oitocentos oitocentas", 900: "novecentos novecentas"},
               "S": {10**3: "mil", 10**6: "milhao milhoes", 10**9: "bilhao bilhoes", 10**12: "trilhao trilhoes"}})
    _ords(pt, {"X": {200: _endings("ducentesim", "o|a|os|as"), 300: _endings("trecentesim", "o|a|os|as"), 400: _endings("quadringentesim", "o|a|os|as"), 500: _endings("quingentesim", "o|a|os|as"), 600: _endings("sexcentesim", "o|a|os|as"), 700: _endings("septingentesim", "o|a|os|as"), 800: _endings("octingentesim", "o|a|os|as"), 900: _endings("nongentesim", "o|a|os|as")}})
    _ords(pt, {"U": {1: _endings("primeir", a), 2: _endings("segund", a), 3: _endings("terceir", a), 4: _endings("quart", a), 5: _endings("quint", a), 6: _endings("sext", a), 7: _endings("setim", a), 8: _endings("oitav", a), 9: _endings("non", a)},
               "T": {10: _endings("decim", a), 11: _endings("undecim", a), 12: _endings("duodecim", a)},
               "D": {20: _endings("vigesim", a), 30: _endings("trigesim", a), 40: _endings("quadragesim", a), 50: _endings("quinquagesim", a), 60: _endings("sexagesim", a), 70: _endings("septuagesim setuagesim", a), 80: _endings("octogesim", a), 90: _endings("nonagesim", a)},
               "X": {100: _endings("centesim", a)}, "S": {10**3: _endings("milesim", a)}})
    L["pt"], C["pt"], P["pt"] = pt, "e", "virgula"
    # --- French (soixante-dix = 60 + 10, quatre-vingt = 4 x 20 are handled by the grammar)
    fr = _lex({"Z": {0: "zero"}, "U": {1: "un une", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf"},
               "T": {10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize"},
               "D": {20: "vingt vingts", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante", 70: "septante", 80: "octante huitante", 90: "nonante"},
               "H": {100: "cent cents"}, "S": {10**3: "mille mil", 10**6: "million millions", 10**9: "milliard milliards", 10**12: "billion billions"}})
    i = "|s|e|es"
    _ords(fr, {"U": {1: _endings("premier", "|s") + " premiere premieres", 2: _endings("deuxieme", "|s"), 3: _endings("troisieme", "|s"), 4: _endings("quatrieme", "|s"), 5: _endings("cinquieme", "|s"), 6: _endings("sixieme", "|s"), 7: _endings("septieme", "|s"), 8: _endings("huitieme", "|s"), 9: _endings("neuvieme", "|s")},
               "T": {10: _endings("dixieme", "|s"), 11: _endings("onzieme", "|s"), 12: _endings("douzieme", "|s"), 13: _endings("treizieme", "|s"), 14: _endings("quatorzieme", "|s"), 15: _endings("quinzieme", "|s"), 16: _endings("seizieme", "|s")},
               "D": {20: _endings("vingtieme", "|s"), 30: _endings("trentieme", "|s"), 40: _endings("quarantieme", "|s"), 50: _endings("cinquantieme", "|s"), 60: _endings("soixantieme", "|s"), 80: _endings("huitantieme", "|s")},
               "H": {100: _endings("centieme", "|s")}, "S": {10**3: _endings("millieme", "|s"), 10**6: _endings("millionieme", "|s")}})
    fr["unieme"] = ("U", 1, True)
    fr["unieme" + "s"] = ("U", 1, True)
    L["fr"], C["fr"], P["fr"] = fr, "et", "virgule"
    return L, C, P


_LEX, _CONN, _DEC = _build()
# spoken decimal words besides the grammar's own ("seis ponto cinco", "sei punto cinque")
_DEC_ALT = {"de": {"punkt"}, "nl": {"punt"}, "it": {"punto"}, "es": {"punto"}, "pt": {"ponto"}, "fr": {"point"}}
_SCALE_VALS = {Decimal(10) ** 3, Decimal(10) ** 6, Decimal(10) ** 9, Decimal(10) ** 12}
_WEAK = {"en": {"one", "second", "hundreds", "thousands", "millions", "billions"}, "de": {"ein", "eins", "eine", "einen", "einem", "einer", "eines"}, "nl": {"een"}, "it": {"uno", "un", "una"},
         "es": {"uno", "un", "una", "segundo", "segunda", "segundos", "segundas"}, "pt": {"um", "uma", "segundo", "segunda", "segundos", "segundas"}, "fr": {"un", "une"}}
_PCT = {"en": r"\b(?:per ?cent|percent)\b", "de": r"\bprozent\b", "nl": r"\b(?:procent|percent)\b", "fr": r"\bpour cent\b", "es": r"\bpor (cien(?:to)?)\b", "it": r"\bper cento\b", "pt": r"\bpor cento\b"}
_CURCONN = {"e", "et", "y", "con", "und", "en", "mit", "and", "com"}
_CUR = {"euro", "euros", "dollar", "dollars", "dolar", "dolares", "pfund", "pound", "pounds", "livre", "livres", "libra", "libras", "franc", "francs", "franken",
        "real", "reais", "gulden", "peso", "pesos", "dollari", "sterlina", "sterline", "franchi", "yen", "dolaren", "dollaro"}
_MINOR = {"cent", "cents", "centavo", "centavos", "céntimo", "céntimos", "cêntimo", "cêntimos", "centime", "centimes",
          "centesimo", "centesimi", "eurocent", "eurocents", "penny", "pence"}
_UNIT_FIRST = ("de", "nl")  # "dreiundzwanzig": unit + and + tens


class Num:
    """One number found in a text. vals = every value its digits/words can mean; form = "words" | "digits"."""

    __slots__ = ("vals", "form", "ordinal", "weak", "start", "end", "fodder")

    def __init__(self, vals, form, ordinal=False, weak=False, start=0, end=0, fodder=None):
        self.vals, self.form, self.ordinal, self.weak, self.start, self.end = frozenset(vals), form, ordinal, weak, start, end
        self.fodder = (not weak) if fodder is None else fodder  # may stand in a gap as a WRONG value

    def __repr__(self):
        return f"Num({sorted(self.vals)!r},{self.form},{'ord' if self.ordinal else ''}{'weak' if self.weak else ''})"


def _segment(lang: str, tok: str):
    """Split a (folded) word into number morphemes, longest first; None when it is not wholly made of number morphemes."""
    lex = _LEX[lang]
    if tok in lex:
        return [(tok,) + lex[tok]]
    if lang not in ("de", "nl", "it") or len(tok) < 4:
        return None
    memo = {}

    def go(i):
        if i == len(tok):
            return []
        if i in memo:
            return memo[i]
        res = None
        for j in range(len(tok), i, -1):
            w = tok[i:j]
            if w in lex:
                rest = go(j)
                if rest is not None:
                    res = [(w,) + lex[w]] + rest
                    break
            elif w == _CONN[lang] and j < len(tok) and i > 0:
                rest = go(j)
                if rest is not None:
                    res = [(w, "C", 0, False)] + rest
                    break
        memo[i] = res
        return res

    seg = go(0)
    if seg and all(m[1] == "C" for m in seg):
        return None
    return seg


def _new():
    return {"total": 0, "hund": 0, "low": 0, "lowk": None, "hset": False, "last": 10**15, "conn": False, "n": 0, "zero": False, "ord": False, "txt": []}


def _push(lang: str, st: dict, w: str, kind: str, v: int, is_ord: bool) -> bool:
    """Try to add one morpheme to the number being built; False means: close this number and start a new one."""
    if kind == "C":
        if st["n"]:
            st["conn"] = True
        return True
    if st["ord"] and lang in ("es", "pt") and is_ord and kind != "Z" and 0 < v < _place(_value(st)):
        # "vigésimo primero", "centésimo vigésimo primero", "milésimo ducentésimo septuagésimo séptimo": a chain of
        # ordinal words, each below the lowest place of what came before, is one number
        st["total"], st["hund"], st["low"], st["lowk"], st["hset"] = _value(st) + v, 0, 0, "DU", False
        st["n"] += 1
        st["txt"].append(w)
        return True
    if st["ord"] or st["zero"]:
        return False
    low, lk = st["low"], st["lowk"]
    if kind == "Z":
        if st["n"]:
            return False
        st["zero"] = True
    elif kind == "U":
        if lk is None:
            st["low"], st["lowk"] = v, "U"
        elif lk == "D" and lang not in _UNIT_FIRST:
            st["low"], st["lowk"] = low + v, "DU"
        elif lk in ("T", "DT") and lang == "fr" and low % 10 == 0 and low in (10, 70, 90) and v >= 7:
            st["low"], st["lowk"] = low + v, "TU"
        else:
            return False
    elif kind == "T":
        if lk is None:
            st["low"], st["lowk"] = v, "T"
        elif lk == "D" and lang == "fr" and low in (60, 80):
            st["low"], st["lowk"] = low + v, "DT"
        else:
            return False
    elif kind == "D":
        if lk is None:
            st["low"], st["lowk"] = v, "D"
        elif lk == "U" and lang in _UNIT_FIRST and st["conn"]:
            st["low"], st["lowk"] = v + low, "UD"
        elif lk == "U" and lang == "fr" and v == 20 and low == 4:
            st["low"], st["lowk"] = 80, "D"
        else:
            return False
    elif kind == "H":
        if st["hset"]:
            return False
        st["hund"], st["hset"] = (low or 1) * 100, True
        st["low"], st["lowk"] = 0, None
    elif kind == "X":
        if st["hset"] or lk is not None:
            return False
        st["hund"], st["hset"] = v, True
    elif kind == "S":
        group = st["hund"] + st["low"]
        if v > st["last"] and v >= 10**6 and (st["total"] or group):
            st["total"] = (st["total"] + group) * v  # "dos mil millones", "mil quinientos cincuenta millones"
            st["last"] = v
            st["hund"] = st["low"] = 0
            st["lowk"], st["hset"] = None, False
        elif v >= st["last"]:
            return False
        else:
            st["total"] += (group or 1) * v
            st["last"] = v
            st["hund"] = st["low"] = 0
            st["lowk"], st["hset"] = None, False
    st["n"] += 1
    st["txt"].append(w)
    st["ord"] = is_ord
    return True


def _value(st: dict) -> int:
    return st["total"] + st["hund"] + st["low"]


def _place(n: int) -> int:
    """The lowest non-zero place of n (120 -> 10, 1200 -> 100, 7 -> 1)."""
    k = 1
    while n and n % (k * 10) == 0:
        k *= 10
    return k if n else 0


def _items(lang: str, morphs: list) -> list:
    """Cardinal/ordinal items of one run of morphemes: [(value, ordinal, weak, first_morph_index, last_morph_index)]."""
    out, st, first = [], _new(), 0

    def close(last):
        nonlocal st
        if st["n"]:
            weak = len(st["txt"]) == 1 and st["txt"][0] in _WEAK.get(lang, ())
            out.append((_value(st), st["ord"], weak, first, last))
        st = _new()

    for idx, (w, kind, v, o) in enumerate(morphs):
        if kind == "C" and not st["n"]:
            continue
        if not _push(lang, st, w, kind, v, o):
            close(idx - 1)
            first = idx
            _push(lang, st, w, kind, v, o)
        elif st["n"] == 1 and kind != "C" and len(st["txt"]) == 1:
            first = idx
    close(len(morphs) - 1)
    return out


def _digit_vals(tok: str) -> set:
    """All values a digit string can mean ("1.234" is 1234 or 1.234; "12,50" is 12.5)."""
    t = re.sub(r"[  '’]", "", tok)
    seps = [c for c in t if c in ".,"]
    if not seps:
        return {Decimal(t)}
    last = t[max(t.rfind("."), t.rfind(","))]
    if len(seps) > 1 and len(set(seps)) == 1:
        return {Decimal(re.sub(r"[.,]", "", t))}  # 1.234.567
    if len(set(seps)) == 2:
        return {Decimal(t.replace("," if last == "." else ".", "").replace(last, "."))}
    head, _, tail = t.partition(last)
    if len(tail) == 3 and head not in ("0", "00"):
        return {Decimal(head + tail), Decimal(head + "." + tail)}
    return {Decimal(head + "." + tail)}


_TOKEN = re.compile(r"\d+(?:[.,  '’]\d+)*|[^\W\d_]+|%")


def _prefix_of_word(lang: str, r: str) -> bool:
    """r (3+ letters) is a proper prefix of a number word of the language ("zwanz" of "zwanzig")."""
    pre = _PREFIXES.get(lang)
    if pre is None:
        pre = _PREFIXES[lang] = {w[:k] for w in _LEX[lang] for k in range(3, len(w))}
    return r in pre


_PREFIXES: dict = {}
# "and a half" readings -> the decimal word + five, so "zweieinhalb Millionen" = 2 500 000 and "dos y media" = 2.5
# (a reference clock "2:30" still matches: see _clock_split). Only right after a number word.
_HALF = {"en": [(r"(?<=[a-z]) and a half\b", " point five")],
         "de": [(r"\banderthalb\b", "eins komma funf"), (r"\b([a-z]+?)einhalb\b", r"\1 komma funf")],
         "nl": [(r"\banderhalf\b", "een komma vijf"), (r"\b([a-z]+?)enhalf\b", r"\1 komma vijf")],
         "es": [(r"(?<=[a-z]) y medi[oa]\b", " coma cinco")], "fr": [(r"(?<=[a-z]) heures? et demie?\b", " virgule cinq heures"), (r"(?<=[a-z]) et demie?\b", " virgule cinq")],
         "it": [(r"(?<=[a-z]) e mezz[oa]\b", " virgola cinque")], "pt": [(r"(?<=[a-z]) e mei[oa]\b", " virgula cinco")]}
_GARBLE_NOT = {"it": {"quindi"}}  # real words one or two letters short of a number word ("quindi" / "quindici")


def _garble(lang: str, tok: str) -> bool:
    """A hypothesis word that is a broken number word: number morphemes followed by the cut-off start of another
    ("zweitausendzwanz", "neunzehnhundertdreiss"), or a long number word with its last 1-2 letters missing
    ("tausen", "thousan", "quinhent"). Scored as a WRONG value in place of the reference number, never as DROPPED."""
    if lang not in _LEX or tok in _LEX[lang] or len(tok) < 6 or tok in _GARBLE_NOT.get(lang, ()):
        return False
    if lang in ("de", "nl", "it"):
        for j in range(len(tok) - 2, 3, -1):
            seg = _segment(lang, tok[:j])
            if seg and len([m for m in seg if m[1] != "C"]) >= 2 and (
                    _prefix_of_word(lang, tok[j:]) or (len(tok) - j <= 7 and j >= 0.5 * len(tok))):
                return True  # "zweitausendzwanz", "neunzehnhundertsig", "duemiladunici"
    return any(w.startswith(tok) and 1 <= len(w) - len(tok) <= 2 for w in _LEX[lang] if len(w) >= 7)


_ROMAN = re.compile(r"(?<![\w-])(?=[IVXLC]*[IVX])[IVXLC]{2,7}(?![\w-])")
_ROMAN_NOT = {"XL", "CV", "LV", "CL", "CC", "LL", "LC", "IC", "CI", "LI", "VC"}


def _to_roman(n: int) -> str:
    out = ""
    for v, r in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + r, n - v
    return out


def _romans(text: str) -> str:
    """Upper-case Roman numerals ("XIX", "XXI") -> digits, only when written canonically (a symbol form, like digits)."""
    def rep(m):
        t = m.group(0)
        v = sum({"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}[c] * (-1 if k + 1 < len(t) and
                {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}[c] < {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}[t[k + 1]] else 1)
                for k, c in enumerate(t))
        return str(v) if t not in _ROMAN_NOT and 0 < v <= 199 and _to_roman(v) == t else t
    return _ROMAN.sub(rep, text)


# relative clock readings (2026-10-18, the reading-variety fill): "halb drei", "Viertel vor vier", "zehn nach zwei",
# "kwart over drie", "tien voor half vijf", "las tres menos cuarto", "cuarto para las tres", "às duas e meia", "le tre
# meno un quarto", "trois heures moins le quart", "midi". The hour is said on a 12-hour clock, so it stands for both
# halves of the day ({2, 14}); the minutes are a second number, as in a digital "14:30". Phrases that need a clock
# context (a minute count before "nach" / "over" / "menos" ...) only count after a clock preposition ("um", "om",
# "las", "às", "alle") or with the hour word ("heures").
_REL_PREP = {"de": {"um", "gegen", "ab", "bis", "ist", "seit", "von"}, "nl": {"om", "rond", "tot", "vanaf", "is", "van"},
             "es": {"las", "la"}, "pt": {"as", "a"}, "it": {"le", "alle", "dalle", "verso", "alla", "dalla", "l", "all", "dall"},
             "fr": {"a", "vers", "des", "avant", "jusqu", "est", "pour"}}  # fr: "trois heures et demie" alone is a duration
# noon / midnight words (fr "midi", it "mezzogiorno", pt "meio-dia") are a time only after a preposition or with minutes
# ("mezzogiorno e un quarto"); "le Midi" / "il Mezzogiorno" are regions and "l'après-midi" the afternoon
_NOON = {"fr": {("midi",): 12, ("minuit",): 0}, "it": {("mezzogiorno",): 12, ("mezzanotte",): 0},
         "pt": {("meio", "dia"): 12, ("meia", "noite"): 0}}
_NOON_PREP = {"fr": {"a", "vers", "avant", "jusqu", "est", "pour", "des"},
              "it": {"a", "verso", "fino", "entro", "dopo", "prima", "per", "era", "sara", "gia", "ormai", "quasi", "circa"},
              "pt": {"ao", "a", "as", "ate", "antes", "depois", "perto", "por", "desde", "apos", "era", "e"}}
# a time-span noun after "Viertel vor einem" / "kwart over drie" makes it a count ("Viertel vor einem Jahr"), not a clock
_SPAN_NOUN = {"de": {"jahr", "jahre", "jahren", "jahres", "monat", "monate", "monaten", "woche", "wochen", "tag", "tage", "tagen",
                     "stunde", "stunden", "minute", "minuten", "sekunden", "mal", "prozent", "euro", "personen", "leute", "leuten"},
              "nl": {"jaar", "jaren", "maand", "maanden", "week", "weken", "dag", "dagen", "minuut", "minuten", "seconden",
                     "keer", "procent", "euro", "personen", "mensen"}}
_DECLINED = {"de": {"einem", "einen", "einer", "eines"}}


def _noon_tail(lang: str, w: list, k: int, hr: int):
    """Minutes after a noon word: "e un quarto", "et demi", "moins le quart" -> (end, hour, minute), or None."""
    at = lambda q, *xs: q < len(w) and w[q] in xs  # noqa: E731
    qt = "quart" if lang == "fr" else "quarto"
    half = {"fr": ("demi", "demie"), "it": ("mezzo", "mezza"), "pt": ("meia", "meio")}[lang]
    conn = {"fr": "et", "it": "e", "pt": "e"}[lang]
    minus = {"fr": "moins", "it": "meno", "pt": "menos"}[lang]
    art = {"fr": (), "it": ("un",), "pt": ("um",)}[lang]
    marts = {"fr": ("le",), "it": ("un",), "pt": ("um",)}[lang]
    if at(k, conn):
        if at(k + 1, qt):
            return k + 2, hr, 15
        if art and at(k + 1, *art) and at(k + 2, qt):
            return k + 3, hr, 15
        if at(k + 1, *half):
            return k + 2, hr, 30
        if lang == "it" and at(k + 1, "tre") and at(k + 2, "quarti"):
            return k + 3, hr, 45
    if at(k, minus):
        if at(k + 1, qt):
            return k + 2, hr - 1, 45
        if at(k + 1, *marts) and at(k + 2, qt):
            return k + 3, hr - 1, 45
    return None


def _num_at(lang: str, w: list, i: int, lo: int, hi: int):
    """(value, next index) of a cardinal of 1-3 words starting at w[i] (es / pt "treinta y cinco" with the connector),
    or None; the value must lie in [lo, hi]."""
    best = None
    for j in range(i + 1, min(len(w), i + 3) + 1):
        morphs = []
        trailing = False
        for k in range(i, j):
            seg = _segment(lang, w[k])
            if seg:
                morphs += seg
            elif w[k] == _CONN[lang] and morphs and k + 1 < j:
                morphs.append((w[k], "C", 0, False))
            elif w[k] == _CONN[lang] and morphs:
                trailing = True  # "vinte e" + "cinco": the connector ends this window, a longer one holds the number
                break
            else:
                morphs = None
                break
        if trailing:
            continue
        if not morphs:
            break
        its = _items(lang, morphs)
        if len(its) == 1 and not its[0][1] and its[0][3] == 0 and its[0][4] == len(morphs) - 1 and lo <= its[0][0] <= hi:
            best = (int(its[0][0]), j)
    return best


def _rel_at(lang: str, w: list, i: int):
    """(end index, hour 0-23, minute) of a relative clock reading starting at w[i], or None."""
    def H(k):  # an hour word 1-12 (24-hour forms only where the language says them: fr "quatorze heures moins")
        r = _num_at(lang, w, k, 1, 24 if lang == "fr" else 12) if k < len(w) else None
        if r and (w[k] in _DECLINED.get(lang, ()) or (r[1] < len(w) and w[r[1]] in _SPAN_NOUN.get(lang, ()))):
            return None  # "Viertel vor einem Jahr", "halb zehn Leute"
        return r

    def M(k, hi=29):
        return _num_at(lang, w, k, 1, hi) if k < len(w) else None

    at = lambda k, *xs: k < len(w) and w[k] in xs  # noqa: E731
    prep = i > 0 and w[i - 1] in _REL_PREP[lang]
    for words, hr in _NOON.get(lang, {}).items():
        if tuple(w[i:i + len(words)]) == words:
            tail = _noon_tail(lang, w, i + len(words), hr)
            if tail:
                return tail
            return (i + len(words), hr, 0) if i > 0 and w[i - 1] in _NOON_PREP[lang] else None
    if lang in ("de", "nl"):
        half, quarter = ("halb", "viertel") if lang == "de" else ("half", "kwart")
        past, to = (("nach",), ("vor",)) if lang == "de" else (("over", "na"), ("voor",))
        if at(i, half) and (h := H(i + 1)) and (lang == "de" or h[0] != 1 or prep):
            return h[1], h[0] - 1, 30
        if at(i, quarter):
            if at(i + 1, *past) and (h := H(i + 2)):
                return h[1], h[0], 15
            if at(i + 1, *to) and (h := H(i + 2)):
                return h[1], h[0] - 1, 45
            if lang == "de" and (h := H(i + 1)):
                return h[1], h[0] - 1, 15  # "viertel drei" = 2:15
        if lang == "de" and at(i, "dreiviertel") and (h := H(i + 1)):
            return h[1], h[0] - 1, 45
        if (m := M(i)) and m[1] < len(w):
            k = m[1]
            if at(k, *(past + to)) and at(k + 1, half) and (h := H(k + 2)):
                return h[1], h[0] - 1, 30 + m[0] if w[k] in past else 30 - m[0]
            if prep and at(k, *past) and (h := H(k + 1)):
                return h[1], h[0], m[0]
            if prep and at(k, *to) and (h := H(k + 1)):
                return h[1], h[0] - 1, 60 - m[0]
        return None
    if lang in ("es", "pt"):
        minus, conn, quart, art = "menos", _CONN[lang], "cuarto" if lang == "es" else "quarto", (
            ("las", "la") if lang == "es" else ("as", "a"))
        half = "media" if lang == "es" else "meia"
        one = "un" if lang == "es" else "um"
        k0 = i + 1 if at(i, one) else i
        if at(k0, quart) and at(k0 + 1, "para") and at(k0 + 2, *art) and (h := H(k0 + 3)):
            return h[1], h[0] - 1, 45
        if (m := M(i, 30)) and at(m[1], "para") and at(m[1] + 1, *art) and (h := H(m[1] + 2)):
            return h[1], h[0] - 1, 60 - m[0]
        if prep and (h := H(i)):
            k = h[1]
            if at(k, minus):
                if at(k + 1, quart) or (at(k + 1, one) and at(k + 2, quart)):
                    return k + (2 if at(k + 1, quart) else 3), h[0] - 1, 45
                if (m := M(k + 1, 30)):
                    return m[1], h[0] - 1, 60 - m[0]
            if at(k, conn) and at(k + 1, quart):
                return k + 2, h[0], 15
            if at(k, conn) and at(k + 1, one) and at(k + 2, quart):
                return k + 3, h[0], 15
            if at(k, conn) and at(k + 1, half):
                return k + 2, h[0], 30
        return None
    if lang == "it":
        if prep and (h := H(i)):
            k = h[1]
            if at(k, "meno"):
                if at(k + 1, "quarto") or (at(k + 1, "un") and at(k + 2, "quarto")):
                    return k + (2 if at(k + 1, "quarto") else 3), h[0] - 1, 45
                if (m := M(k + 1, 30)):
                    return m[1], h[0] - 1, 60 - m[0]
            if at(k, "e") and at(k + 1, "un") and at(k + 2, "quarto"):
                return k + 3, h[0], 15
            if at(k, "e") and at(k + 1, "tre") and at(k + 2, "quarti"):
                return k + 3, h[0], 45
            if at(k, "e") and at(k + 1, "mezza", "mezzo"):
                return k + 2, h[0], 30
        return None
    if lang == "fr":
        if prep and (h := H(i)) and at(h[1], "heure", "heures"):
            k = h[1] + 1
            if at(k, "moins"):
                if at(k + 1, "quart") or (at(k + 1, "le") and at(k + 2, "quart")):
                    return k + (2 if at(k + 1, "quart") else 3), h[0] - 1, 45
                if (m := M(k + 1, 30)):
                    return m[1], h[0] - 1, 60 - m[0]
            if at(k, "et") and at(k + 1, "quart"):
                return k + 2, h[0], 15
            if at(k, "et") and at(k + 1, "demie", "demi"):
                return k + 2, h[0], 30
        return None
    return None


def _rel_clocks(lang: str, t: str) -> str:
    """Relative clock readings in the folded text -> one placeholder token each ("zzrel" + hour + minute letters),
    read back by extract_numbers as an hour (both halves of the day) and a minute number."""
    if lang not in _REL_PREP:
        return t
    toks = list(re.finditer(r"[^\W\d_]+", t))
    w = [m.group(0) for m in toks]
    out, last, i = [], 0, 0
    while i < len(w):
        r = _rel_at(lang, w, i)
        if r is None:
            i += 1
            continue
        j, h, mm = r
        out += [t[last:toks[i].start()], " zzrel" + chr(97 + h % 24) + chr(97 + mm // 10) + chr(97 + mm % 10) + " "]
        last, i = toks[j - 1].end(), j
    return "".join(out) + t[last:]


def _rel_nums(tk: str, ti: int) -> list:
    h, mm = ord(tk[5]) - 97, (ord(tk[6]) - 97) * 10 + ord(tk[7]) - 97
    h12 = h % 12
    out = [Num({Decimal(h12 or 12), Decimal(h12 + 12 if h12 else 0)}, "words", start=ti, end=ti)]
    if mm:
        out.append(Num({Decimal(mm)}, "words", start=ti, end=ti))
    return out


def _meia_six(lang: str, toks: list, ti: int, in_run: bool) -> bool:
    """pt "meia" is the digit 6 beside another number word ("nove seis meia", "meia trinta e dois"); after the article
    "uma" / "um" and before a noun it is the article's "half" ("uma meia hora", "uma meia dúzia")."""
    nxt = toks[ti + 1] if ti + 1 < len(toks) else None
    nxt_num = bool(nxt) and any(m[1] != "C" for m in _segment(lang, nxt) or ())
    prv = toks[ti - 1] if ti > 0 else None
    prv_num = in_run and bool(prv) and any(m[1] != "C" for m in _segment(lang, prv) or ())
    if prv in ("um", "uma") and nxt and not nxt_num:
        return False
    return nxt_num or prv_num


_ES_UNIT = {"metro", "metros", "kilo", "kilos", "kilogramo", "kilogramos", "gramo", "gramos", "litro", "litros", "grado",
            "grados", "euro", "euros", "dolar", "dolares", "peso", "pesos", "centimetro", "centimetros", "milimetro",
            "milimetros", "kilometro", "kilometros", "segundo", "segundos", "minuto", "minutos", "hora", "horas", "por",
            "ciento", "cien", "mil", "millon", "millones", "anos", "ano", "puntos", "punto", "toneladas", "tonelada",
            "libras", "libra", "pulgadas", "dia", "dias", "semanas", "meses", "mes", "veces", "y", "e", "o", "a", "al"}


def _con_decimal(lang: str, toks: list, ti: int) -> bool:
    """es "tres con cinco" = 3.5 when the second number ends the text or is followed by a unit / currency / another
    number; "dos con tres camas" (two ... with three beds) and "somos tres con dos niños" are counts, not 2.3."""
    k = ti + 1
    while k < len(toks) and (_segment(lang, toks[k]) or (toks[k] == _CONN[lang] and k + 1 < len(toks) and _segment(lang, toks[k + 1]))):
        k += 1
    return k >= len(toks) or toks[k] in _ES_UNIT or toks[k][0].isdigit() or toks[k] == "%"


def _pct_words(lang: str, t: str) -> str:
    """Percent words -> " % ". Spanish "por cien" is a percent only after a number ("el tres por cien", "cien por cien");
    "lo vendió por cien euros" is a price of 100 and "por cien mil euros" one of 100 000 ("por ciento" is always a percent)."""
    if lang != "es":
        return re.sub(_PCT[lang], " % ", t)

    def rep(m):
        if m.group(1) == "ciento":
            return " % "
        prev = re.search(r"([^\W_]+)\W*$", t[:m.start()])
        if prev and (prev.group(1).isdigit() or _segment(lang, prev.group(1))) and not re.match(r"\s+mil(?:lon|\b)", t[m.end():]):
            return " % "
        return m.group(0)
    return re.sub(_PCT[lang], rep, t)


def _digit_word(lang: str, tok: str) -> bool:
    seg = _segment(lang, tok)
    return bool(seg) and len(seg) == 1 and seg[0][1] in ("U", "Z")


def extract_numbers(text: str, lang: str, keep_weak: bool = True, garbles: bool = False) -> list:
    """Every number in `text`, digits and words, in order. Words: en de es fr it nl pt (grammar above) and, through
    `_ext_numbers`, the other 18 languages when the pipeline's number-form tables are importable (else digits only).
    garbles=True (hypothesis side) also returns broken number words as Num(form="garble") with no value."""
    text = _romans(text)
    if lang not in _LEX and _ext_reader(lang) is not None:
        return _ext_numbers(text, lang, keep_weak, garbles)
    t = _rel_clocks(lang, _fold(text))
    for pat, rep in _HALF.get(lang, ()):
        t = re.sub(pat, rep, t)
    if lang in _PCT:
        t = _pct_words(lang, t)
    t = re.sub(r"(\d):00\b", r"\1", t)
    # a plain-space thousands group ("800 000", "1 000 000") is one number (FLEURS references)
    for _ in range(3):
        t = re.sub(r"(?<![\d.,])(\d{1,3}) (\d{3})(?!\d)(?![.,]\d)", "\\1\u00a0\\2", t)
    toks = _TOKEN.findall(t)
    words = lang in _LEX
    runs, cur = [], []  # runs of consecutive number tokens: ("d", tok_index, vals) or ("m", tok_index, morph)
    nums = []

    def flush():
        nonlocal cur
        if cur:
            nums.extend(_run_to_nums(lang, cur))
        cur = []

    rep = 1  # English "double four" / "triple seven": the next digit word counts that many times
    for ti, tk in enumerate(toks):
        if tk[0].isdigit():
            flush()
            nums.append(Num(_digit_vals(tk), "digits", start=ti, end=ti))
            rep = 1
            continue
        if tk.startswith("zzrel") and len(tk) == 8:  # a relative clock reading (_rel_clocks)
            flush()
            nums.extend(_rel_nums(tk, ti))
            rep = 1
            continue
        seg = _segment(lang, tk) if words else None
        if lang == "pt" and tk == "meia" and (ti == 0 or toks[ti - 1] != "e") and _meia_six(lang, toks, ti, bool(cur)):
            seg = [("meia", "U", 6, False)]  # Brazilian "meia" (meia dúzia) = 6 among numbers said one by one
        if lang == "en" and seg is None:  # UK / AU digit forms (2026-10-17): "nought", "oh" inside digit runs, doubles
            if tk in ("nought", "naught"):
                seg = _segment(lang, "zero")
            elif tk == "oh" and (rep > 1 or _en_digit_ctx(lang, toks, ti, cur)):
                seg = _segment(lang, "zero")
            elif tk in ("double", "triple") and ti + 1 < len(toks) and (_en_digit(lang, toks[ti + 1]) or toks[ti + 1] == "oh"):
                rep = 2 if tk == "double" else 3
                continue
        if seg and rep > 1 and len(seg) == 1 and seg[0][1] in ("U", "Z"):
            for _ in range(rep - 1):
                cur.append((ti, seg[0]))
        rep = 1
        if seg:
            for m in seg:
                cur.append((ti, m))
        elif words and tk == _CONN[lang] and cur:
            cur.append((ti, (tk, "C", 0, False)))
        elif words and (tk == _DEC[lang] or tk in _DEC_ALT.get(lang, ())) and cur:
            cur.append((ti, (tk, "P", 0, False)))
        elif words and lang == "es" and tk == "con" and cur and ti + 1 < len(toks) and _segment(lang, toks[ti + 1]) \
                and _con_decimal(lang, toks, ti):
            cur.append((ti, (tk, "P", 0, False)))  # "tres con cinco" = 3.5 (a number on both sides)
        else:
            flush()
            if garbles and words and _garble(lang, tk):
                nums.append(Num((), "garble", start=ti, end=ti))
    flush()
    nums.sort(key=lambda n: (n.start, n.end))
    nums = _merge(lang, [n for n in nums if n.form != "garble"], toks) + [n for n in nums if n.form == "garble"]
    nums.sort(key=lambda n: (n.start, n.end))
    return [n for n in nums if keep_weak or not n.weak]


def _en_digit(lang, tok):
    seg = _segment(lang, tok)
    return bool(seg) and len(seg) == 1 and seg[0][1] in ("U", "Z")


def _en_digit_ctx(lang, toks, ti, cur):
    """'oh' is zero inside a run of single digit words ("five oh two", "oh two nine ..."), never as an interjection."""
    nxt = ti + 1 < len(toks) and (_en_digit(lang, toks[ti + 1]) or toks[ti + 1] in ("double", "triple", "oh"))
    prev = bool(cur) and cur[-1][1][1] in ("U", "Z", "P")  # "nought point oh five"
    nxt2 = ti + 2 < len(toks) and (_en_digit(lang, toks[ti + 2]) or toks[ti + 2] in ("double", "triple", "oh"))
    return (prev and nxt) or (nxt and nxt2) or (prev and cur[-1][0] == ti - 1 and len(cur) >= 2)


def _run_to_nums(lang, run):
    """Turn one run of morphemes (with decimal markers) into Num objects."""
    out, i = [], 0
    parts, curp = [], []
    for ti, m in run:
        if m[1] == "P":
            parts.append(curp)
            curp = []
            parts.append("P")
        else:
            curp.append((ti, m))
    parts.append(curp)
    pending = None  # the integer Num waiting for a decimal marker
    k = 0
    res = []
    while k < len(parts):
        p = parts[k]
        if p == "P":
            k += 1
            continue
        nxt_is_dec = k + 1 < len(parts) and parts[k + 1] == "P"
        prev_dec = k > 0 and parts[k - 1] == "P"
        morphs = [m for _, m in p]
        its = _items(lang, morphs)
        ti0 = lambda mi: p[mi][0]
        if prev_dec and res and res[-1][0] == "int":
            # fraction part: digit-by-digit if all single digits, else the first cardinal's digits
            _, intval, istart, _iend = res.pop()
            kinds = [m for m in morphs if m[1] != "C"]
            scale = 1
            if len(kinds) >= 2 and kinds[-1][1] == "S" and all(m[1] in "UZ" and m[2] < 10 for m in kinds[:-1]):
                scale, kinds = kinds[-1][2], kinds[:-1]  # "one point two million" = 1 200 000, not 1.2000000
            if kinds and all(m[1] in "UZ" and m[2] < 10 for m in kinds):
                frac = "".join(str(m[2]) for m in kinds)
                rest = []
            elif its:
                frac, rest = str(its[0][0]), its[1:]
            else:
                res.append(("int", intval, istart, _iend))
                k += 1
                continue
            vals = {Decimal(f"{intval}.{frac}") * scale}
            lastti = p[-1][0] if not rest else p[its[0][4]][0]
            out.append(Num(vals, "words", start=istart, end=lastti))
            for v, o, w, a, b in rest:
                out.append(Num({Decimal(v)}, "words", o, w, ti0(a), p[b][0]))
            k += 1
            continue
        for j, (v, o, w, a, b) in enumerate(its):
            if nxt_is_dec and j == len(its) - 1 and not o:
                res.append(("int", v, ti0(a), p[b][0]))
            else:
                out.append(Num({Decimal(v)}, "words", o, w, ti0(a), p[b][0]))
        k += 1
    for _, v, s, e in res:  # integer parts whose decimal marker had no usable fraction
        out.append(Num({Decimal(v)}, "words", start=s, end=e))
    return out


def _merge(lang, nums, toks):
    """Currency "12 euros 50" -> 12.50; English years "twenty twenty-three" -> 2023."""
    if lang == "it":
        # "venti euro e settanta cinque centesimi": the reader takes "centesimi" as the ordinal hundredth and fuses it into
        # the number (7500); the amount is the words before it, followed by the minor unit
        fixed = []
        for x in nums:
            if x.form == "words" and x.ordinal and x.end > x.start and toks[x.end] in ("centesimo", "centesimi"):
                sub = extract_numbers(" ".join(toks[x.start:x.end]), lang)
                if len(sub) == 1 and not sub[0].ordinal and sub[0].form == "words":
                    x = Num(sub[0].vals, "words", start=x.start, end=x.end - 1)
            fixed.append(x)
        nums = fixed
    out, i = [], 0
    while i < len(nums):
        a = nums[i]
        b = nums[i + 1] if i + 1 < len(nums) else None
        # "10 mil", "2,5 millones": digits right before a lone scale word are one number
        if (b and a.form == "digits" and b.form == "words" and b.start == a.end + 1 and b.start == b.end
                and min(b.vals) in _SCALE_VALS):
            out.append(Num({v * min(b.vals) for v in a.vals}, "digits", start=a.start, end=b.end))
            i += 2
            continue
        if b and a.form == b.form == "words" and not (a.ordinal or b.ordinal):
            av, bv = min(a.vals), min(b.vals)
            if av == av.to_integral() and bv == bv.to_integral() and 0 <= bv < 100:
                gap = b.start - a.end
                if gap in (2, 3) and a.end + 1 < len(toks) and toks[a.end + 1] in _CUR and (gap == 2 or toks[a.end + 2] in _CURCONN):
                    out.append(Num({av + bv / 100}, "words", start=a.start, end=b.end))
                    i += 2
                    continue
                if b.start == a.end + 1 and 10 <= bv <= 99 and (
                        (lang == "en" and 11 <= av <= 99) or (lang == "nl" and 11 <= av <= 20)):
                    # two-part years: "nineteen sixty", "negentien zestig", "twintig vijfentwintig"
                    out.append(Num({av * 100 + bv}, "words", start=a.start, end=b.end))
                    i += 2
                    continue
        if (a.form == "words" and not a.ordinal and a.end + 1 < len(toks) and toks[a.end + 1] in _MINOR
                and all(v == v.to_integral() and 0 < v < 100 for v in a.vals)):
            # "dos centavos" alone is 0.02 for "$0.02" (and still 2 for a written "2 centavos")
            out.append(Num(set(a.vals) | {v / 100 for v in a.vals}, "words", start=a.start, end=a.end))
            i += 1
            continue
        out.append(a)
        i += 1
    return out


# ---- the other 18 languages (bg cs da el et fi hr hu lt lv mt pl ro ru sk sl sv uk): number WORDS are read by the
# training pipeline's inflection-tolerant reader (ml_multi.parse_numbers over its per-language form tables of every
# case / gender / number; the ru/uk tables are built here from ml_slav's case paradigms) when those modules are
# importable; otherwise these languages are scored on digits only. `mode` then says "words+digits:ext-<hash>" (the
# hash of the tables, so two scorings compare only when the tables were the same). Safety rules of this scorer on
# top of the reader: a number that is ONE word read only through a guessed case ending ("семья" -> семь + я), and a
# lone "one" (an article in ro sv da hu el bg mt sl ...), are WEAK: they never count as a reference number and never
# stand in a gap as a wrong value, but a weak hypothesis number still matches a reference of the same value.
EXT_LANGS = ("bg", "cs", "da", "el", "et", "fi", "hr", "hu", "lt", "lv", "mt", "pl", "ro", "ru", "sk", "sl", "sv", "uk")
# forms / separators the per-language tables lack (each found by a scorer test; value-safe closed-class words)
_EXT_FIX = {
    "sl": {"cardinal": {"0": ["nic", "nicla"], "1": ["en", "ena", "eno", "enega", "enemu", "enim", "eni", "enem"],
                        "2": ["dva", "dve", "dveh", "dvema"], "3": ["trije", "tri", "treh", "trem", "tremi"],
                        "4": ["stirje", "stiri", "stirih", "stirim", "stirimi"], "5": ["pet", "petih", "petim", "petimi"],
                        "6": ["sest", "sestih", "sestim", "sestimi"], "7": ["sedem", "sedmih", "sedmim", "sedmimi"],
                        "60": ["sestdeset", "sestdesetih", "sestdesetim", "sestdesetimi"]},
           "ordinal": {"1": ["prvi", "prva", "prvo", "prvega", "prvemu", "prvem", "prvim", "prve", "prvih"],
                       "2": ["drugi", "druga", "drugo", "drugega", "drugemu", "drugem", "drugim", "druge", "drugih"],
                       "3": ["tretji", "tretja", "tretje", "tretjega", "tretjemu", "tretjem", "tretjim", "tretjih"],
                       "4": ["cetrti", "cetrta", "cetrto", "cetrtega", "cetrtemu", "cetrtem", "cetrtim", "cetrte", "cetrtih"],
                       "5": ["peti", "peta", "peto", "petega", "petemu", "petem", "pete"],
                       "6": ["sesti", "sesta", "sesto", "sestega", "sestemu", "sestem", "seste"],
                       "7": ["sedmi", "sedma", "sedmo", "sedmega", "sedmemu", "sedmem", "sedme"],
                       "60": ["sestdeseti", "sestdeseta", "sestdeseto", "sestdesetega", "sestdesetem", "sestdesetih"]}},
    # collective numerals ("погибли двое полицейских") and "с половиной" (= and a half)
    "ru": {"cardinal": {"2": ["двое", "двоих", "двоим", "двоими"], "3": ["трое", "троих", "троим", "троими"],
                        "4": ["четверо", "четверых", "четверым"], "5": ["пятеро", "пятерых"], "6": ["шестеро", "шестерых"],
                        "7": ["семеро", "семерых"]}, "connector": ["с"], "half": ["с половиной"]},
    "uk": {"cardinal": {"2": ["двоє", "двох", "двом", "двома"], "3": ["троє", "трьох"], "4": ["четверо", "чотирьох"],
                        "5": ["п'ятеро", "п'ятьох"]}, "connector": ["з"], "half": ["з половиною", "з половиной"]},
    "lt": {"cardinal": {"3": ["trys"]}},
    "lv": {"cardinal": {"100": ["simti", "simtu", "simtiem", "simtus"]}},
    "hr": {"ordinal": {"1": ["prvog", "prvoga", "prvom", "prvome", "prvoj", "prve", "prva", "prvi", "prvu", "prvo"]}},
    "bg": {"decimal": ["цяло", "цели"]},
    "sv": {"cardinal": {"40": ["fortio"]}, "ordinal": {"40": ["fortionde"]}},
    "da": {"cardinal": {"60": ["tres", "treds"]}},
    "ro": {"ordinal": {"1": ["unulea"], "8": ["optulea", "optlea", "opta"]}},
    "fi": {"ordinal": {str(10 + i): [a, b] for i, (a, b) in enumerate([
        ("yhdestoista", "yhdennentoista"), ("kahdestoista", "kahdennentoista"), ("kolmastoista", "kolmannentoista"),
        ("neljastoista", "neljannentoista"), ("viidestoista", "viidennentoista"), ("kuudestoista", "kuudennentoista"),
        ("seitsemastoista", "seitsemannentoista"), ("kahdeksastoista", "kahdeksannentoista"),
        ("yhdeksastoista", "yhdeksannentoista")], start=1)}},
    "mt": {"cardinal": {"2000": ["elfejn"]}},
    "cs": {"cardinal": {str(t + u): [a + "a" + b] for t, b in ((20, "dvacet"), (30, "tricet"), (40, "ctyricet"),
                                                               (50, "padesat"), (60, "sedesat"), (70, "sedmdesat"),
                                                               (80, "osmdesat"), (90, "devadesat"))
                        for u, a in ((1, "jedna"), (2, "dva"), (3, "tri"), (4, "ctyri"), (5, "pet"), (6, "sest"),
                                     (7, "sedm"), (8, "osm"), (9, "devet"))}},
}
# table words that break the reader: Maltese "u" (the connector "and") listed as a fraction denominator
_EXT_DROP = {"mt": {("fraction", "100"): {"u"}, ("fraction", "1000"): {"u"}}}
_EXT_NOT_NUM = {"hu": {"szazad", "szazadban", "szazadot", "szazadi", "szazadig", "szazadra", "szazadtol"},
                "cs": {"stoleti", "stoletich", "stoletim"}, "sk": {"storocie", "storocia", "storoci"},
                "sl": {"stoletje", "stoletja", "stoletju"}, "pl": {"stulecie", "stulecia"},
                "ru": {"столетие", "столетия", "семья", "семьи", "семье"}, "uk": {"столiття", "століття", "сiм'я", "сім'я"}}
# words that fold onto a number form but are another word (checked BEFORE folding: hr "što" = what, sv "förstå" =
# understand, sk "ste" = (you) are, el "εκτός" = except) and homographs that are weak like a lone "one" (hu "egyes")
_EXT_RAW_NOT_NUM = {"hr": {"što"}, "bg": set(), "sv": {"förstå"}, "sk": {"ste"}, "el": {"εκτός"}, "cs": {"ste"}}
_EXT_WEAK_WORDS = {"hu": {"egyes", "egyesek"}}
_EXT_STATE: dict = {}
_EXT_HASH: dict = {}


def _slav_table(lang: str, S) -> dict:
    """ml_multi-style form table for ru/uk from ml_slav: every case form of each numeral morpheme, cardinal and ordinal."""
    vals = list(range(0, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100))
    nom = {}
    for v in vals:
        try:
            ws = S.card_nom(lang, v)
        except Exception:  # noqa: BLE001
            continue
        if len(ws) == 1:
            nom[ws[0]] = v
    nom.update({"одна": 1, "одно": 1, "одне": 1, "две": 2, "дві": 2})
    card, ordn = {}, {}
    for w, alts in S.CARD[lang].items():
        if w in nom:
            card.setdefault(str(nom[w]), set()).update(f for case in alts for f in case)
    for v, lem in S.SCALE_NOUN[lang].items():
        _, sg, pl = S.NOUNS[lang][lem]
        card.setdefault(str(v), set()).update(f for num in (sg, pl) for case in num for f in case)
    for v in vals[1:] + [1000, 2000, 3000, 4000, 5000, 10**6]:
        try:
            for words, _tag in S.ord_readings(lang, v):
                if len(words) == 1:
                    ordn.setdefault(str(v), set()).add(words[0])
        except Exception:  # noqa: BLE001
            continue
    dec = {"ru": ["целых", "целая", "целой", "целую", "запятая"], "uk": ["цілих", "ціла", "цілої", "кома"]}[lang]
    frac = {"ru": {"10": ["десятых", "десятая", "десятой"], "100": ["сотых", "сотая", "сотой"], "1000": ["тысячных"]},
            "uk": {"10": ["десятих", "десята", "десятої"], "100": ["сотих", "сота", "сотої"], "1000": ["тисячних"]}}[lang]
    fw = {w for ws in frac.values() for w in ws}  # "десятых" is the decimal denominator here, never the ordinal 10
    ordn = {k: v - fw for k, v in ordn.items()}
    return {"cardinal": {k: sorted(v) for k, v in card.items()}, "ordinal": {k: sorted(v) for k, v in ordn.items()},
            "decimal": dec, "fraction": frac, "connector": [], "half": [], "minus": ["минус" if lang == "ru" else "мінус"]}


def _ext_reader(lang: str):
    """The ml_multi module set up for `lang`, or None (module missing or language not covered)."""
    if lang in _EXT_STATE:
        return _EXT_STATE[lang]
    mm = None
    if lang in EXT_LANGS:
        try:
            import hashlib
            import ml_multi as mm

            tab = mm._table()
            files = [mm.__file__, mm.LEX_PATH]
            if lang in ("ru", "uk") and lang not in tab:
                import ml_slav

                tab[lang] = _slav_table(lang, ml_slav)
                files.append(ml_slav.__file__)
                mm.HALF15.update({"полтора", "полторы", "півтора", "півтори"})
            if lang not in tab:
                mm = None
            else:
                for key, add in _EXT_FIX.get(lang, {}).items():
                    if isinstance(add, dict):
                        for v, ws in add.items():
                            tab[lang].setdefault(key, {}).setdefault(v, [])
                            tab[lang][key][v] = list(tab[lang][key][v]) + [w for w in ws if w not in tab[lang][key][v]]
                    else:
                        tab[lang][key] = list(tab[lang].get(key, [])) + [w for w in add if w not in tab[lang].get(key, [])]
                for (key, v), ws in _EXT_DROP.get(lang, {}).items():
                    if v in tab[lang].get(key, {}):
                        tab[lang][key][v] = [w for w in tab[lang][key][v] if w not in ws]
                mm._LEXES.pop(lang, None)
                mm._CACHE.pop(lang, None)
                h = hashlib.sha1(repr([sorted(d.get(lang, {}).items()) if isinstance(d.get(lang, {}), dict) else
                                       sorted(d.get(lang, ())) for d in (_EXT_FIX, _EXT_DROP, _EXT_NOT_NUM,
                                                                         _EXT_RAW_NOT_NUM, _EXT_WEAK_WORDS)]).encode())
                for f in files:
                    with open(f, "rb") as fh:
                        h.update(fh.read())
                _EXT_HASH[lang] = h.hexdigest()[:8]
        except Exception:  # noqa: BLE001  (no pipeline modules: digits only)
            mm = None
    _EXT_STATE[lang] = mm
    return mm


def _ext_exact(lx, tok: str) -> bool:
    """tok is wholly made of LISTED forms (and connectors), with no guessed case ending."""
    memo = {}

    def go(i):
        if i == len(tok):
            return True
        if i not in memo:
            memo[i] = any((tok[i:j] in lx.forms or (i > 0 and j < len(tok) and tok[i:j] in lx.glue)) and go(j)
                          for j in range(len(tok), i, -1))
        return memo[i]

    return go(0)


def _ext_garble(lx, w: str) -> bool:
    """A broken number word in an 18-language hypothesis ("trzydziecie", "setiņdesmit"): 6+ letters, not read, and
    its first 6 letters (or all but its last) start a listed number form of 7+ letters."""
    if len(w) < 6 or not w.isalpha():
        return False
    pre = _PREFIXES.get(("ext", lx.lang))
    if pre is None:
        pre = _PREFIXES[("ext", lx.lang)] = {f[:6] for f in lx.forms if len(f) >= 7}
    return w[:6] in pre


def _ext_numbers(text: str, lang: str, keep_weak: bool, garbles: bool = False) -> list:
    mm = _ext_reader(lang)
    lx = mm.lexicon(lang)
    t = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()  # not folded: "što" must stay "što"
    t = re.sub(r"(\d):00\b", r"\1", t)
    for _ in range(3):
        t = re.sub(r"(?<![\d.,])(\d{1,3}) (\d{3})(?!\d)(?![.,]\d)", "\\1\u00a0\\2", t)
    toks = re.split(r"[ \t\r\n]+", t.strip()) if t.strip() else []
    nums, chunk = [], []
    stop = _EXT_NOT_NUM.get(lang, set())

    def flush():
        if not chunk:
            return
        # a connector right after a decimal word ("три цяло и пет", "trei virgulă și") is not read by the reader: drop it
        keep = [k for k in range(len(chunk)) if not (k and mm.fold(chunk[k][1]) in lx.glue
                                                       and mm.fold(chunk[k - 1][1]) in lx.dec)]
        chunk[:] = [chunk[k] for k in keep]
        # a known non-number word ("století" = century, "što" = what) is hidden from the reader, so it can neither be
        # read as a number nor glue onto the number before it ("devatenáctém století" stays 19)
        words = ["_" if (mm.fold(re.sub(r"[^\w'-]", "", w)).strip("-'") in stop
                         or re.sub(r"[^\w'-]", "", w) in _EXT_RAW_NOT_NUM.get(lang, ())) else w for _, w in chunk]
        for n in mm.parse_numbers(" ".join(words), lang):
            a, b = chunk[n.start][0], chunk[n.end][0]
            span = [mm.fold(re.sub(r"[^\w'-]", "", w)).strip("-'") for w in words[n.start:n.end + 1]]
            # one word: a lone "one", a lone ordinal "second" (= "other" in sv da fi et lv hr ...), a listed homograph,
            # or a word read only through a guessed ending
            lone = n.start == n.end and ((n.value == 1 and not n.ordinal) or (n.value == 2 and n.ordinal)
                                         or span[0] in _EXT_WEAK_WORDS.get(lang, ()))
            guessed = n.start == n.end and not all(_ext_exact(lx, p) for p in span[0].split("-") if p)
            # a guessed one-word number is weak (never a reference number) but number-like: it may stand as WRONG
            nums.append(Num({Decimal(n.value)}, "words", n.ordinal, lone or guessed, a, b, fodder=not lone))
        if garbles:
            taken = {i for n in nums for i in range(n.start, n.end + 1)}
            for i, w in chunk:
                fw = mm.fold(re.sub(r"[^\w'-]", "", w)).strip("-'")
                if i not in taken and fw not in stop and _ext_garble(lx, fw):
                    nums.append(Num((), "garble", start=i, end=i))
        chunk.clear()

    for i, tk in enumerate(toks):
        if re.search(r"\d", tk):
            flush()
            for d in re.findall(r"\d+(?:[.,\u00a0\u202f'’]\d+)*", tk):
                nums.append(Num(_digit_vals(d), "digits", start=i, end=i))
        else:
            chunk.append((i, tk))
            if re.search(r"[.!?;:]$", tk):
                flush()
    flush()
    nums.sort(key=lambda n: (n.start, n.end))
    out = []
    for n in nums:  # "5 milionów", "2,5 mld": digits right before a lone scale word are one number
        if (out and n.form == "words" and n.start == n.end and out[-1].form == "digits" and n.start == out[-1].end + 1
                and n.vals
                and min(n.vals) in _SCALE_VALS):
            p = out.pop()
            out.append(Num({v * min(n.vals) for v in p.vals}, "digits", start=p.start, end=n.end))
        else:
            out.append(n)
    return [n for n in out if keep_weak or not n.weak]


def _mode(lang: str) -> str:
    if lang in WORD_LANGS:
        return "words+digits"
    return f"words+digits:ext-{_EXT_HASH[lang]}" if _ext_reader(lang) is not None else "digits_only"


def _hm(x):
    """(H, MM) when the digit number x can be a dotted time H.MM (H <= 24, MM < 60), else None."""
    if x.form in ("digits", "words"):
        for v in x.vals:
            if v.as_tuple().exponent == -2 and 0 <= v < 25 and int((v % 1) * 100) < 60:
                return Decimal(int(v)), Decimal(int(round((v % 1) * 100)))
            if x.form == "words" and v % 1 == Decimal("0.5") and 0 < v < 25:  # "las dos y media" = 2:30
                return Decimal(int(v)), Decimal(30)
    return None


def _clock_split(H: list, R: list) -> list:
    """Hypothesis side only (the reference count never depends on the hypothesis). A dotted time "23.35" is one number
    while "23:35" or a spoken time are two: split a hypothesis H.MM when the reference holds H and MM side by side."""
    out = []
    for x in H:
        hm = _hm(x)
        if hm and any(hm[0] in R[k].vals and hm[1] in R[k + 1].vals for k in range(len(R) - 1)):
            out += [Num({hm[0]}, x.form, start=x.start, end=x.start), Num({hm[1]}, x.form, start=x.end, end=x.end)]
        else:
            out.append(x)
    return out


def _clock_join(H: list, R: list) -> list:
    """The other direction: the reference writes "15.30 Uhr" (one number) and the hypothesis says or writes the hour
    and the minutes as two numbers: join them into the reference's value."""
    want = {hm: min(r.vals) for r in R for hm in [_hm(r)] if hm}
    out, k = [], 0
    while k < len(H):
        a = H[k]
        b = H[k + 1] if k + 1 < len(H) else None
        near = b is not None and b.start - a.end <= 2  # at most one word between: "15 Uhr 30", not "15 Leute und 30"
        hit = near and next((v for (h, mm), v in want.items() if h in a.vals and mm in b.vals), None)
        if hit is not None and hit is not False:
            out.append(Num({hit}, a.form, start=a.start, end=b.end))
            k += 2
        else:
            out.append(a)
            k += 1
    return out


def _pair_join(H: list, R: list) -> list:
    """Hypothesis side: "twenty twenty two" in the reference is one number (2022) by the two-part-year rule, while a
    hypothesis "20, 22" is two: join a consecutive hypothesis pair (a, b), 10 <= b <= 99, when a * 100 + b is a
    reference value and neither a nor b is one on its own."""
    rv = {v for r in R for v in r.vals}
    out, k = [], 0
    while k < len(H):
        a, b = H[k], H[k + 1] if k + 1 < len(H) else None
        if (b is not None and b.start - a.end <= 2 and a.vals and b.vals and not (a.vals & rv)
                and not (b.vals & rv)):
            hit = next((x * 100 + y for x in a.vals for y in b.vals if 10 <= y <= 99 and x == int(x) and y == int(y)
                        and x * 100 + y in rv), None)
            if hit is not None:
                out.append(Num({hit}, a.form, start=a.start, end=b.end))
                k += 2
                continue
        out.append(a)
        k += 1
    return out


def _digit_split(H: list, R: list) -> list:
    """Hypothesis side: a digit run that spells consecutive reference numbers ("C45P2" for "C four five P two", a phone
    number read in groups "zéro six quatre-vingt-dix" = 0 6 90 vs "0690" without its leading zero is not handled) is
    split into those numbers. Only integer runs of 2+ digits, only when the run's value is not itself a reference value."""
    rv = {v for r in R for v in r.vals}
    out = []
    for x in H:
        done = False
        if x.form == "digits" and len(x.vals) == 1 and not (x.vals & rv):
            v = next(iter(x.vals))
            sv = str(v) if v == int(v) else ""
            for i in range(len(R)):
                acc, j = "", i
                while j < len(R) and len(acc) < len(sv):
                    rvj = min(R[j].vals)
                    if rvj != int(rvj) or (j > i and R[j].start - R[j - 1].end > 1):  # adjacent reference numbers only
                        break
                    acc += str(int(rvj))
                    j += 1
                if j - i >= 2 and acc == sv and len(sv) >= 2:
                    out += [Num({min(R[q].vals)}, "digits", start=x.start, end=x.end) for q in range(i, j)]
                    done = True
                    break
        if not done:
            out.append(x)
    return out


def _cents_split(H: list, R: list) -> list:
    """Hypothesis side: a price "23,90" is one number while a reference said without the unit ("vinte e três e
    noventa", "zwei fünfzig") holds two: split a hypothesis A.BC (BC 10-99) when the reference has A and BC next to
    each other (_clock_split does the same for times, minutes < 60)."""
    out = []
    for x in H:
        hit = None
        for v in x.vals:
            if v.as_tuple().exponent == -2 and v >= 1:
                a, c = Decimal(int(v)), Decimal(int(round((v % 1) * 100)))
                if c >= 10 and any(a in R[k].vals and c in R[k + 1].vals and R[k + 1].start - R[k].end <= 2
                                   for k in range(len(R) - 1)):
                    hit = (a, c)
        if hit:
            out += [Num({hit[0]}, x.form, start=x.start, end=x.start), Num({hit[1]}, x.form, start=x.end, end=x.end)]
        else:
            out.append(x)
    return out


def _cents_join(H: list, R: list) -> list:
    """The other direction: the reference writes a price "2,50" and the hypothesis says it without the unit ("zwei
    fünfzig", "veinte con cincuenta" reads 20.5 already): join two adjacent hypothesis integers A, BC (BC 10-99, at
    most one word between) into A.BC when that is a reference value and neither is one on its own."""
    rv = {v for r in R for v in r.vals}
    out, k = [], 0
    while k < len(H):
        a, b = H[k], H[k + 1] if k + 1 < len(H) else None
        if b is not None and b.start - a.end <= 2 and a.vals and b.vals and not (a.vals & rv) and not (b.vals & rv):
            hit = next((x + y / 100 for x in a.vals for y in b.vals if 10 <= y <= 99 and x == int(x) and y == int(y)
                        and x + y / 100 in rv), None)
            if hit is not None:
                out.append(Num({hit}, a.form, start=a.start, end=b.end))
                k += 2
                continue
        out.append(a)
        k += 1
    return out


def number_outcomes(ref: str, hyp: str, lang: str) -> list:
    """One dict per reference number: {value, outcome, ref_form, hyp_form, mode}. See module docstring."""
    mode = _mode(lang)
    R = extract_numbers(ref, lang, keep_weak=False)
    H = extract_numbers(hyp, lang, keep_weak=True, garbles=True)
    H = _digit_split(_pair_join(_cents_join(_clock_join(_cents_split(_clock_split(H, R), R), R), R), R), R)
    n, m = len(R), len(H)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1] + 1 if R[i].vals & H[j].vals else 0)
    match = {}
    i = j = 0
    while i < n and j < m:
        if R[i].vals & H[j].vals and dp[i][j] == dp[i + 1][j + 1] + 1:
            match[i] = j
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    # gaps between anchors: pair leftover refs with leftover (non-weak) hyp numbers in order
    anchors = sorted(match.items()) + [(n, m)]
    rows, pi, pj = [], 0, 0
    for ai, aj in anchors:
        free = [x for x in range(pj, aj) if H[x].fodder]
        for k, ri in enumerate(range(pi, ai)):
            if k < len(free):
                outcome, hf = "WRONG_VALUE", H[free[k]].form
            else:
                outcome, hf = "DROPPED", None
            rows.append((ri, outcome, hf))
        if ai < n:
            rows.append((ai, "WORDS_RIGHT_VALUE" if H[aj].form == "words" else "DIGITS_RIGHT_VALUE", H[aj].form))
        pi, pj = ai + 1, aj + 1
    return [{"value": str(min(R[ri].vals)), "outcome": o, "ref_form": R[ri].form, "hyp_form": hf, "mode": mode} for ri, o, hf in sorted(rows)]
# ===== END CORE =====


# ===== FORM CLASSES (outside CORE: the public scorer is unchanged) =====
# How a hypothesis WRITES a number, independent of its value: a spoken-form model that converts only some numbers (or
# half of one: "2 mil quinientos") makes the two overlapping windows read the same audio in different forms.
# "converted" = the number is written as words AND its value is right. A words-only chain with the wrong value (a garbled
# or invented compound) is "words_wrong"; the old count (words-only chain, value ignored) stays as `form_old` / `converted_old`.
FORM_CLASSES = ("converted", "digits_left", "mixed", "words_wrong", "other")


def number_groups(text: str, lang: str) -> list:
    """Hypothesis numbers grouped as written: maximal chains of ADJACENT number tokens (no word between). Each group is
    {"form": "words" | "digits" | "mixed", "text": the clause}. "mixed" = digits and number words inside one chain ("2 mil
    quinientos", "veinte 5"); a lone "2 million" or "15h30" is plain digits. Chains never cross a clause break
    (". " "," ": ") because the tokenizer drops punctuation ("2017. Primero" is not one number); broken number words
    (garbles) carry no token position and are not grouped."""
    out = []
    for clause in re.split(r"(?<=[.,;:!?])\s+", text):
        nums = [n for n in extract_numbers(clause, lang, keep_weak=False) if not n.weak]
        cur = []
        for n in nums + [None]:
            if cur and (n is None or n.start != cur[-1].end + 1):
                forms = {x.form for x in cur}
                out.append({"form": "mixed" if len(forms) > 1 else forms.pop(), "text": clause})
                cur = []
            if n is not None:
                cur.append(n)
    return out


def form_class(text: str, lang: str) -> str:
    """How `text` (the hypothesis region standing where ONE reference number stands) writes it: "mixed" when a digit and a
    number word are joined in one chain, "converted" (words only), "digits_left" (digits only), "other" (no number)."""
    forms = {g["form"] for g in number_groups(text, lang)}
    if "mixed" in forms or forms == {"words", "digits"}:
        return "mixed"
    if forms == {"digits"}:
        return "digits_left"
    return "converted" if forms == {"words"} else "other"


def form_outcomes(ref: str, hyp: str, lang: str) -> list:
    """number_outcomes rows (semantics unchanged) plus a `form` per reference number: converted / digits_left / mixed /
    other (+ words_wrong). WORDS_RIGHT -> converted, DIGITS_RIGHT -> digits_left; for WRONG / DROPPED rows the leftover
    hypothesis chains are dealt out in order: a mixed chain makes that row `mixed` (the old scorer folded it into
    WRONG_VALUE), a digits-only chain `digits_left` (digits written, wrong value), a words-only chain `words_wrong`
    (words, wrong value: NOT converted). `form_old` keeps the previous reading, where that last case was `converted`."""
    rows = number_outcomes(ref, hyp, lang)
    left = [g["form"] for g in number_groups(hyp, lang)]
    right_digits = sum(r["outcome"] == "DIGITS_RIGHT_VALUE" for r in rows)
    right_words = sum(r["outcome"] == "WORDS_RIGHT_VALUE" for r in rows)
    for f, k in (("digits", right_digits), ("words", right_words)):  # chains already used by a right row
        for _ in range(k):
            if f in left:
                left.remove(f)
    pool = sorted(left, key=lambda f: f != "mixed")  # mixed chains are the ones to surface first
    for r in rows:
        if r["outcome"] == "WORDS_RIGHT_VALUE":
            r["form"] = "converted"
        elif r["outcome"] == "DIGITS_RIGHT_VALUE":
            r["form"] = "digits_left"
        elif pool:
            r["form"] = {"mixed": "mixed", "digits": "digits_left", "words": "converted"}[pool.pop(0)]
        else:
            r["form"] = "other"
        r["form_old"] = r["form"]  # the previous reading: a words-only chain stood for converted whatever its value
        if r["outcome"] != "WORDS_RIGHT_VALUE" and (r["form"] == "converted" or r["hyp_form"] == "garble"):
            r["form"] = "words_wrong"  # words written, value wrong / broken word: not converted
    return rows


def form_shares(counts: dict) -> dict:
    """{class: percent of n} from a Counter of FORM_CLASSES (+ n)."""
    n = sum(counts.get(k, 0) for k in FORM_CLASSES)
    out = {k: round(100 * counts.get(k, 0) / n, 1) if n else None for k in FORM_CLASSES}
    if "converted_old" in counts:  # the old reading (words-only chain of any value), for comparison
        out["converted_old"] = round(100 * counts["converted_old"] / n, 1) if n else None
    return out


def summarize(rows: list) -> dict:
    """Counts of the four outcomes (+ n, + mode) over rows from number_outcomes."""
    c = collections.Counter(r["outcome"] for r in rows)
    out = {k: c.get(k, 0) for k in OUTCOMES}
    out["n"] = len(rows)
    out["mode"] = rows[0]["mode"] if rows else None
    return out


def add_counts(total: dict, rows: list) -> None:
    """Accumulate rows into a per-language Counter dict: total[lang] handled by caller; here a flat Counter."""
    for r in rows:
        total[r["outcome"]] += 1
        total["n"] += 1


def rates(counts: dict) -> dict:
    n = counts.get("n", 0)
    return {k: round(100 * counts.get(k, 0) / n, 1) if n else None for k in OUTCOMES}


def parse_value(text: str, lang: str):
    """The value of `text` when it is exactly one number (words or digits), else None. For tests and callers."""
    ns = extract_numbers(text, lang)
    return min(ns[0].vals) if len(ns) == 1 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    a = ap.parse_args(argv)
    for r in number_outcomes(a.ref, a.hyp, a.lang):
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
