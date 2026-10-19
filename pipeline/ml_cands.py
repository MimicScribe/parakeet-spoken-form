"""Tier-1 multilingual spoken-form candidates (R3.11 P1): es, fr, de, pt, it, nl.

Given caption text with digits and a language, find the number spans and list the spoken readings a speaker may
have used. The CANONICAL reading of every value comes from per-language number grammars, pre-dumped into
`ml_grammar_table.json` by a grammar dumper that is not part of this repo. Variants (gender/apocope on 1, 2 and hundreds; "tausend" without "ein"; the hundreds reading of
1100-1999; ordinal case/gender endings; a decimal fraction read as one number; digit-by-digit codes) are derived from
the canonical words in the grammar's own spelling style. A value missing from the table yields NO candidates: the span
is then undecidable and never becomes a label.

Wave-2 span kinds (own readings, same rules): "unit" (number + unit words, signs, degrees, per mille), "date" (day,
".month", ".year" spans), "range" (left end; joiner + right end), "decade", "alnum" (5G, A4, COVID-19), "roman" (II..XXX
after a strong context word or a capitalised name), "frac" (3/4, 2 1/2, 24/7), "version" (5.1.3, IPs; "2.0" is a "dec"
with a point-word variant), "amp" ("&"). A sign directly attached to a number adds a minus / plus word to its readings.
Hyphen compounds ("13-jährig"), ISO dates and 3-part hyphen chains stay "skip" (the caption is dropped).

Two passes:
    reqs = needed(texts, lang)          # every grammar call the spans of these texts need
    (run the grammar dumper on reqs)      # build_table(...) below writes requests.json and reads answers.json
    spans(text, lang)                   # Span(start, end, text, kind, cands) with cands[0] = canonical
"""

import json
import os
import re
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "ml_grammar_table.json")
LANGS = ("es", "fr", "de", "pt", "it", "nl")
SLAV = ("ru", "uk")  # case-inflecting languages: span finding and readings live in ml_slav (the other languages keep their own path)
# Parliament-speech candidate additions (2026-10-16; old candidates: ML_CANDS_OFF=<names>):
#   range_bare  a range's two ends said with no connector for every range ("2021-2027" -> "... duemilaventisette")
#   range_or    the "or" connector ("1-2 milioni" -> "uno o due milioni", "ein oder zwei")
#   acr_word    a pronounceable letter code said as a word ("COP21" -> "COP ventuno", "SIS2" -> "SIS due")
#   dot_clock   "H.MM" (H <= 23, MM <= 59) also read as a time ("alle 10.30" -> "dieci e trenta")
#   dot_bare    "A.B" also read as two numbers ("articolo 5.2" -> "cinque due")
#   it_comma    Italian "A,B" also read with "comma" (the legal paragraph word: "articolo 75,3" -> "settantacinque comma tre")
#   int_pairs   a 4-digit non-year number also read as two pairs ("5707" -> "cinquantasette zero sette")
#   ord_card    a "N." ordinal (de / nl) also read as a cardinal that ends a sentence ("Nummer 68." -> "achtundsechzig.")
#   year_spoken a year also read as its last two digits ("1991" -> "einundneunzig", "quatre vingt onze") and as two
#               pairs ("2060" -> "zwanzig sechzig", "1904" -> "diecinueve cero cuatro"): the 2026-10-16 parliament batch
#               transcript held one of these, not a full reading, at 1,150 undecided year spans (de / fr / it / pt)
CAND_FIXES = ({"range_bare", "range_or", "acr_word", "dot_clock", "dot_bare", "it_comma", "int_pairs", "ord_card",
               "year_spoken"}
              - set(filter(None, os.environ.get("ML_CANDS_OFF", "").split(","))))
# Regional / colloquial readings (2026-10-16, variety pass; section "regional readings" at the end of this file). They
# only APPEND readings after the existing ones (cands[0] and the order of the old readings never change):
#   fr_belch   Belgian / Swiss "septante" (70s), "nonante" (90s), Swiss "huitante" (80s)
#   pt_eu      European Portuguese 10^12 "bilião" (Brazilian "trilhão"), 10^9 "mil milhões" in every reading
#   de_at      Austrian month names in dates ("Jänner", "Feber")
#   clock_rel  relative clock readings ("Viertel vor drei", "halb drei", "kwart over / na drie", "las tres menos cuarto",
#              "un cuarto para las tres", "le tre meno un quarto", "trois heures moins le quart", "midi", "meio-dia")
# OFF unless named in ML_CANDS_ON (2026-10-16 dry A/B, fr: the xlsr CTC prefers the shorter "septante" / "nonante" even
# where the audio and the transcript say "soixante-dix", so offering them to label2 cost 14 of 197 kept rows). The
# regional readings are decided from the external transcript instead: ml_variety.regional_relabel, after label2.
VARIETY_ALL = frozenset({"fr_belch", "pt_eu", "de_at", "clock_rel"})
VARIETY = VARIETY_ALL & set(filter(None, os.environ.get("ML_CANDS_ON", "").split(",")))
CAND_FIXES |= VARIETY  # offered like the parliament additions: with an external transcript only (ml_decide.spans_for)
ALL_CAND_FIXES = frozenset(CAND_FIXES)  # the start-up set; label_lang resets to it on every call
# Readings for the synthetic reading-variety fill (2026-10-18, ml_tts_variety; reading_conv names them). Offered ONLY
# when a caller names them (spans_regional(text, lang, names)); never in spans(), labelling or ml_variety's regional
# relabel, so real-speech labels are unchanged. Appended after the existing readings, like the regional ones:
#   cur_bare    a price with cents said with no unit and no joiner ("2,50 €" -> "zwei fünfzig", "twee vijftig")
#   dec_con     Spanish decimals with "con" ("3,5" -> "tres con cinco", "dos con cincuenta")
#   pct_alt     Dutch "percent" beside "procent", Spanish "por cien" beside "por ciento"
#   phone_alt   German "zwo" for a spoken digit 2, Brazilian "meia" for a spoken digit 6 in phone numbers
#   range_a     the Dutch range joiner "à" ("twee à drie")
#   clock_noon  Spanish noon said with "del mediodía" ("12:30" -> "doce y media del mediodía")
FILL_ALL = frozenset({"cur_bare", "dec_con", "pct_alt", "phone_alt", "range_a", "clock_noon"})
_TABLE: dict | None = None


def _key(lang, kind, n=0, f="", s="", h=0, m=0) -> str:
    return f"{lang}|{kind}|{n}|{f}|{s}|{h}|{m}"


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.load(open(TABLE_PATH)) if os.path.exists(TABLE_PATH) else {}
    return _TABLE


class Missing(KeyError):
    pass


class _Recorder:
    """Stands in for the table while collecting requests: records every key, answers a placeholder."""

    def __init__(self):
        self.keys = set()

    def get(self, k):
        self.keys.add(k)
        return ["_"]


_REC: _Recorder | None = None


def g(lang, kind, n=0, f="", s="", h=0, m=0) -> list[str]:
    k = _key(lang, kind, n, f, s, h, m)
    if _REC is not None:
        return _REC.get(k)
    w = table().get(k)
    if not w:
        raise Missing(k)
    return list(w)


# ---------------------------------------------------------------- span finding

MONTHS = {
    "de": "januar februar märz april mai juni juli august september oktober november dezember",
    "nl": "januari februari maart april mei juni juli augustus september oktober november december",
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre",
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre",
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre",
    "pt": "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro",
}
PCT_WORDS = r"(?:%|prozent|procent|por\s+ciento|pour\s+cent|per\s+cento|por\s+cento)"
ORD_MARK = {"de": r"\.(?=\s)", "nl": r"\.(?=\s)|ste\b|de\b|e\b", "fr": r"er\b|re\b|ère\b|ème\b|e\b",
            "es": r"\.?º|\.?ª|\.?o\b|\.?a\b", "it": r"\.?º|\.?ª", "pt": r"\.?º|\.?ª"}
GROUP = {"fr": r"(?:\u00a0|\u202f| |\.)", "de": r"(?:\.|\u00a0|\u202f)", "nl": r"(?:\.|\u00a0|\u202f)",
         "es": r"(?:\.|\u00a0|\u202f)", "it": r"(?:\.|\u00a0|\u202f)", "pt": r"(?:\.|\u00a0|\u202f)"}
_NUM = {}


def _num_re(lang):
    if lang not in _NUM:
        _NUM[lang] = re.compile(
            r"(?P<pre>(?<![A-Za-z])(?:R\$|US\$|[€$£])\s?)?(?<![\w.,/])(?P<int>\d{1,3}(?:" + GROUP[lang] + r"\d{3})+(?!\d)|\d+)"
            r"(?:(?P<dsep>[,.])(?P<frac>\d+))?(?P<ord>" + ORD_MARK[lang] + r")?(?P<deg>°)?"
            r"(?P<post>\s?(?:€|\$|£|eur\b|usd\b|gbp\b|" + PCT_WORDS + r"))?(?![\w])", re.I)
    return _NUM[lang]


CLOCK = re.compile(r"(?<![\w.,:])(?P<h>\d{1,2})(?::|h)(?P<m>\d{2})(?![\d:])")
# de / nl "15.45 Uhr"; also the first end of a clock range "18.00 tot 20.00 uur"
CLOCK_DOT = re.compile(r"(?<![\w.,:])(?P<h>\d{1,2})\.(?P<m>\d{2})(?=\s?(?:Uhr|uur)\b"
                       r"|\s?(?:–|-|bis|tot|und|en)\s?\d{1,2}\.\d{2}\s?(?:Uhr|uur)\b)")
# A written hour word / day-part marker right after a clock ("22:08 Uhr", "11:00 h", "10:30 a. m.") belongs to the span:
# the readings carry their own hour word, so the label must not keep the written one. "am" is German "on the": only
# the dotted a.m./p.m. forms are markers there.
_AMPM = r"(?P<ap>[ap]\.\s?m\.|[ap]m)"
CLOCK_SUFFIX = {
    "de": r"(?:Uhr|h|(?P<ap>[ap]\.\s?m\.))", "nl": r"(?:uur|u|h|" + _AMPM + ")",
    "es": r"(?:horas?|hrs?|hs|h|" + _AMPM + ")", "pt": r"(?:horas?|hrs?|hs|h|" + _AMPM + ")",
    "fr": r"(?:heures?|hrs?|h|" + _AMPM + ")", "it": r"(?:ore|ora|h|" + _AMPM + ")",
}
_CLOCK_SUFFIX_RE = {k: re.compile(r"\s?" + v + r"(?![\w])", re.I) for k, v in CLOCK_SUFFIX.items()}
# spoken day-part phrases after a 12-hour clock
DAYPART = {
    "es": {"a": ["de la mañana", "de la madrugada", "a eme"], "p": ["de la tarde", "de la noche", "pe eme"]},
    "pt": {"a": ["da manhã", "da madrugada"], "p": ["da tarde", "da noite"]},
    "fr": {"a": ["du matin"], "p": ["de l'après-midi", "du soir"]},
    "it": {"a": ["di mattina", "del mattino"], "p": ["del pomeriggio", "di sera"]},
    "de": {"a": ["morgens", "vormittags"], "p": ["nachmittags", "abends"]},
    "nl": {"a": ["'s ochtends", "'s morgens"], "p": ["'s middags", "'s avonds"]},
}
# Phone numbers / codes written in groups: 3+ groups of digits joined by single spaces or hyphens ("06 12 34 56 78",
# "91-234-56-78", "+34 612 345 678"). In French, a run that could be a space-grouped thousands figure ("12 345 678": every
# group after the first has 3 digits, no leading zero, no "+") is left to the figure reader.
PHONE = re.compile(r"(?<![\w.,/+\-])(?P<plus>\+\s?)?(?P<g>\d{1,4}(?:[ \-]\d{1,4}){2,})(?![\w]|[.,]\d|[ \-]\d)")


@dataclass
class Span:
    start: int
    end: int
    text: str
    kind: str
    cands: list[str] = field(default_factory=list)
    note: str = ""


def _int(s: str) -> int:
    return int(re.sub(r"[.\u00a0\u202f ]", "", s))


def _phone_groups(mt, lang: str) -> list[dict] | None:
    """One span per group of a phone-like run, or None when the run reads as a grouped figure. Each group's span takes
    the separator BEFORE it, so rendering replaces "06-12" by "zéro six douze" (the hyphen goes, one space stays)."""
    g, plus = mt["g"], mt["plus"]
    parts = re.split(r"([ \-])", g)
    groups = parts[0::2]
    digits = sum(len(x) for x in groups)
    if not 7 <= digits <= 15:
        return None
    seps = set(parts[1::2])
    # dates, year runs, score lists and round-number lists are not phone numbers (review H2): a phone run starts with
    # 0 or "+", or has at least 9 digits; never 2+ year-like groups, a year among 1-2-digit groups, or all-round groups
    years = sum(1 for x in groups if re.fullmatch(r"(?:19|20)\d\d", x))
    if not (plus or groups[0].startswith("0") or digits >= 9):
        return None
    if years >= 2 or (years and all(len(x) <= 2 or re.fullmatch(r"(?:19|20)\d\d", x) for x in groups)):
        return None
    if all(len(x) <= 2 for x in groups) and len(groups) <= 4 and not (plus or groups[0].startswith("0")):
        return None
    if all(x.endswith("0") and len(x) == len(groups[0]) for x in groups):
        return None
    if (lang == "fr" and not plus and seps == {" "} and not groups[0].startswith("0") and len(groups[0]) <= 3
            and all(len(x) == 3 for x in groups[1:])):  # only French writes thousands with spaces (GROUP)
        return None
    out, pos = [], mt.start("g")
    for i, grp in enumerate(groups):
        a = pos if i == 0 else pos - 1  # include the separator before this group
        b = pos + len(grp)
        d = {"a": a, "b": b, "kind": "phone", "digits": grp, "lead": i > 0}
        if i == 0 and plus:
            d["a"], d["plus"] = mt.start(), True
        out.append(d)
        pos = b + 1
    return out


def _find(text: str, lang: str) -> list[dict]:
    if lang in SLAV:
        import ml_slav

        return ml_slav.find(text, lang)
    out, taken = [], []
    _find_dates(text, lang, out, taken)
    for mt in PHONE.finditer(text):
        if any(x < mt.end() and mt.start() < y for x, y in taken):
            continue
        gs = _phone_groups(mt, lang)
        if gs:
            out += gs
            taken.append((mt.start(), mt.end()))
    for mt in sorted(list(CLOCK.finditer(text)) + (list(CLOCK_DOT.finditer(text)) if lang in ("de", "nl") else []),
                     key=lambda m: m.start()):
        if any(x < mt.end() and mt.start() < y for x, y in taken):
            continue
        h, m = int(mt["h"]), int(mt["m"])
        if h <= 23 and m <= 59:
            b, ap = mt.end(), ""
            sm = _CLOCK_SUFFIX_RE[lang].match(text, b)
            if sm:
                b, ap = sm.end(), (sm.groupdict().get("ap") or "")[:1].lower()
            out.append({"a": mt.start(), "b": b, "kind": "clock", "h": h, "m": m, "ap": ap})
            taken.append((mt.start(), b))
    _find_deg_ord(text, lang, out, taken)
    _find_wave2(text, lang, out, taken)
    for mt in _num_re(lang).finditer(text):
        a, b = mt.start(), mt.end()
        if any(x < b and a < y for x, y in taken):
            continue
        d = {"a": a, "b": b, "int": mt["int"], "frac": mt["frac"], "dsep": mt["dsep"]}
        ordm = mt["ord"] or ""
        pre, post = (mt["pre"] or "").strip(), (mt["post"] or "").strip().lower()
        before = text[a - 1:a] if a > 0 else ""
        after = text[b:b + 1]
        if before in ("-", "−", "+") and after not in ("-", "–") and (a == 1 or (a > 1 and SIGN_OK_BEFORE.match(text[a - 2]))):
            d["sign"], d["a"] = before, a - 1  # "-3", "+2": a sign directly attached to the number
        elif before in ("-", "–", "+", "−") or after in ("-", "–"):
            d.update(kind="skip", note="range/sign")
        elif mt["deg"]:
            d.update(kind="skip", note="degree")
        elif mt["frac"] is not None and len(mt["frac"]) == 3 and not (lang == "fr" and mt["dsep"] == ","):
            d.update(kind="skip", note="ambiguous grouping/decimal")
        elif ordm and mt["frac"] is None and not pre and not post:
            if ordm == ".":
                nxt = text[b:].lstrip().split(" ", 1)[0].lower().strip(",.;:")
                if not (nxt in MONTHS[lang].split() or (nxt[:1].islower() and nxt.isalpha())):
                    d["b"] = b - 1  # a cardinal ending a sentence / before a capitalised word
                    ordm = ""
            if ordm:
                d.update(kind="ord", fem=ordm.lstrip(".") in ("ª", "a", "re", "ère"), mark=ordm)
        if "kind" not in d:
            if pre or post in ("€", "$", "£", "eur", "usd", "gbp"):
                s = pre or post
                d.update(kind="cur", sym="R" if s.upper() == "R$" else "£" if s in ("£", "gbp") else "$" if "$" in s or s == "usd" else "€")
            elif post:
                d["kind"] = "pct"
            elif mt["frac"] is not None:
                d["kind"] = "dec"
            else:
                d["kind"] = "int"
        if d["kind"] in ("int", "dec") and re.search(r"[A-Za-zÀ-ÿ]", text[d["b"]:d["b"] + 1] + before):
            d.update(kind="skip", note="glued code")
        if d["kind"] == "int" and re.fullmatch(r"0\d+|\d{8,}", d["int"]):
            d["kind"] = "code"  # an unbroken phone number / code: same readings as "int", counted apart in the stats
        out.append(d)
    out = sorted(out, key=lambda x: x["a"])
    _attach_abbr(text, lang, out)
    return out


# ---------------------------------------------------------------- variants (grammar spelling style)

ES_APOC = {"primero": "primer", "tercero": "tercer"}


def _fem_words(ws: list[str], lang: str) -> list[str]:
    """Feminine form of an ordinal / cardinal word list (es/pt/it: -o -> -a; fr premier -> première)."""
    out = []
    for w in ws:
        if lang == "fr":
            w = {"premier": "première", "un": "une", "second": "seconde"}.get(w, w)
        elif lang in ("es", "pt", "it") and w.endswith("o") and w not in ("cento", "ciento", "zero", "cero"):
            w = w[:-1] + "a"
        out.append(w)
    return out


def _card_variants(ws: list[str], lang: str, n: int, year: bool) -> list[list[str]]:
    v = []
    last = ws[-1] if ws else ""
    if lang == "es":
        if last == "uno":
            v += [ws[:-1] + ["un"], ws[:-1] + ["una"]]
        if last == "veintiuno":
            v += [ws[:-1] + ["veintiún"], ws[:-1] + ["veintiuna"]]
        if any(w.endswith("ientos") for w in ws):  # doscientos .. novecientos, quinientos
            v.append([w[:-2] + "as" if w.endswith("ientos") else w for w in ws])
    elif lang == "pt":
        if last == "um":
            v.append(ws[:-1] + ["uma"])
        if "dois" in ws:
            v.append(["duas" if w == "dois" else w for w in ws])
        if any(w.endswith("entos") for w in ws):  # duzentos .. novecentos, quinhentos
            v.append([w[:-2] + "as" if w.endswith("entos") else w for w in ws])
        # the grammar spells Brazilian Portuguese (BR teens + short scale); European speech (VoxPopuli)
        # says/writes these. "catorze"/"quatorze" are both Brazilian.
        eu = {"dezesseis": "dezasseis", "dezessete": "dezassete", "dezenove": "dezanove", "catorze": "quatorze"}
        if any(w in eu for w in ws):
            v.append([eu.get(w, w) for w in ws])
        if any(w in ("bilhão", "bilhões") for w in ws):  # PT_PT 10^9 = "mil milhões" ("um bilhão" -> "mil milhões")
            out, i = [], 0
            while i < len(ws):
                if i + 1 < len(ws) and ws[i] == "um" and ws[i + 1] == "bilhão":
                    out += ["mil", "milhões"]
                    i += 2
                    continue
                out += ["mil", "milhões"] if ws[i] in ("bilhão", "bilhões") else [ws[i]]
                i += 1
            v.append(out)
    elif lang == "it":
        if last == "uno":
            v += [ws[:-1] + ["un"], ws[:-1] + ["una"]]
        elif last.endswith("uno") and len(last) > 3:  # ventuno -> ventun / ventuna
            v += [ws[:-1] + [last[:-1]], ws[:-1] + [last[:-1] + "a"]]
    elif lang == "fr":
        if last == "un":
            v.append(ws[:-1] + ["une"])
    elif lang == "nl":
        if ws == ["één"]:
            v.append(["een"])
    elif lang == "de":
        if ws == ["eins"]:
            v += [[x] for x in ("ein", "eine", "einen", "einem", "einer")]
        if len(ws) == 1:
            w = ws[0]
            if w.startswith("eintausend"):
                v.append([w[len("ein"):]])
            if "einhundert" in w:
                v.append([w.replace("einhundert", "hundert", 1)])
    # the hundreds reading of 1100-1999 ("fünfzehnhundert", "vijftienhonderd", "quinze cents") for non-years; years
    # read that way already in de/nl/fr, so their cardinal reading is the variant (added by the caller)
    if lang in ("de", "nl", "fr") and 1100 <= n <= 1999 and not year:
        v.append(_hundreds(lang, n))
    return v


def _year_spoken(lang: str, n: int) -> list[list[str]]:
    """year_spoken: the shortened readings of a year, appended after its full readings. Two pairs for 1000-2099 with a
    non-zero second pair ("zwanzig sechzig", "vingt cinquante sept", "diecinueve cero cuatro"); the last two digits
    alone for 1900-2099 when they are >= 10 ("einundneunzig", "novantuno"; "cinq" for 2005 is too far from a year)."""
    hi, lo = divmod(n, 100)
    out = []
    if lo:
        out.append(g(lang, "card", hi) + (g(lang, "card", 0) if lo < 10 else []) + g(lang, "card", lo))
    if 1900 <= n <= 2099 and lo >= 10:
        out.append(g(lang, "card", lo))
    return out


def _hundreds(lang: str, n: int) -> list[str]:
    hi, rem = divmod(n, 100)
    head = g(lang, "card", hi)
    tail = g(lang, "card", rem) if rem else []
    if lang == "fr":
        return head + ["cents" if not rem else "cent"] + tail
    word = {"de": "hundert", "nl": "honderd"}[lang]
    return ["".join(head) + word + "".join(tail)]


def _ord_variants(ws: list[str], lang: str, fem: bool) -> list[list[str]]:
    v = []
    if lang == "de" and len(ws) == 1:
        v += [[ws[0] + e] for e in ("n", "r", "s", "m")]
    elif lang in ("es", "pt", "it"):
        f = _fem_words(ws, lang)
        if f != ws:
            v.append(f)
        if lang == "es" and ws[-1] in ES_APOC:
            v.append(ws[:-1] + [ES_APOC[ws[-1]]])
    elif lang == "fr" and ws == ["premier"]:
        v.append(["première"])
    if fem and v:  # the written marker says feminine: put the feminine reading first among the variants
        v.sort(key=lambda x: x != _fem_words(ws, lang))
    return v


def _digits(lang: str, s: str) -> list[str]:
    out = []
    for c in s:
        out += g(lang, "card", int(c))
    return out


MAX_SEG_DIGITS = 11  # longer runs: digit by digit, pairs and triples only (segmentations grow ~1.8x per digit)


def _seg_readings(lang: str, s: str) -> list[list[str]]:
    """Every reading of a digit string as consecutive chunks of 1-3 digits, each chunk said as one number ("612" ->
    "seis doce" / "sesenta y uno dos" / "seiscientos doce" / ...). A chunk with a leading zero is said digit by digit, which
    the 1-digit chunks already cover. A group of up to 4 digits without a leading zero may also be said as one number."""
    def chunk(c):
        return g(lang, "card", int(c)) if len(c) == 1 or c[0] != "0" else None

    if len(s) > MAX_SEG_DIGITS:
        out = [_digits(lang, s)]
        for k in (2, 3):
            parts = [s[i:i + k] for i in range(0, len(s), k)]
            ws = [chunk(p) for p in parts]
            if all(w is not None for w in ws):
                out.append([x for w in ws for x in w])
        return out
    memo: dict[int, list[list[str]]] = {len(s): [[]]}

    def rec(i):
        if i in memo:
            return memo[i]
        res = []
        for k in (1, 2, 3):
            if i + k <= len(s):
                w = chunk(s[i:i + k])
                if w is not None:
                    res += [w + rest for rest in rec(i + k)]
        memo[i] = res
        return res

    out = rec(0)
    if len(s) == 4 and s[0] != "0":
        out.append(g(lang, "card", int(s)))
    return out


PLUS_WORD = {"de": "plus", "es": "más", "fr": "plus", "it": "più", "nl": "plus", "pt": "mais"}


def _phone_cands(d: dict, lang: str) -> list[str]:
    out = _uniq(_seg_readings(lang, d["digits"]))
    if d.get("plus"):
        out = [f"{PLUS_WORD[lang]} {c}" for c in out]
    if d.get("lead"):  # the span starts at the separator before the group: keep one space
        out = [" " + c for c in out]
    return out


# currency words by symbol ("R" = R$, Brazilian real): (singular, plural); cent words; the joiner between the units
# and the cents part ("doze dólares e vinte", the grammar's own style for €/$).
UNIT = {
    "de": {"€": ("Euro", "Euro"), "$": ("Dollar", "Dollar"), "£": ("Pfund", "Pfund")},
    "es": {"€": ("euro", "euros"), "$": ("dólar", "dólares"), "£": ("libra", "libras")},
    "fr": {"€": ("euro", "euros"), "$": ("dollar", "dollars"), "£": ("livre", "livres")},
    "it": {"€": ("euro", "euro"), "$": ("dollaro", "dollari"), "£": ("sterlina", "sterline")},
    "nl": {"€": ("euro", "euro"), "$": ("dollar", "dollar"), "£": ("pond", "pond")},
    "pt": {"€": ("euro", "euros"), "$": ("dólar", "dólares"), "£": ("libra", "libras"), "R": ("real", "reais")},
}
ALT_UNIT = {("es", "$"): ("peso", "pesos")}  # "$" in Latin-American Spanish speech is usually pesos
CENT = {"de": ("Cent", "Cent"), "es": ("céntimo", "céntimos"), "fr": ("centime", "centimes"),
        "it": ("centesimo", "centesimi"), "nl": ("cent", "cent"), "pt": ("cêntimo", "cêntimos")}
ALT_CENT = {("pt", "R"): ("centavo", "centavos"), ("pt", "$"): ("centavo", "centavos"), ("es", "$"): ("centavo", "centavos")}
JOIN = {"de": ["und", ""], "es": ["con", "y", ""], "fr": ["et", ""], "it": ["e", "virgola", ""], "nl": ["en", ""],
        "pt": ["e", ""]}
MILLION = {"millón", "millones", "million", "millions", "milione", "milioni", "milhão", "milhões", "miljoen", "miljoenen",
           "Million", "Millionen", "billón", "billones", "milliard", "milliards", "miliardo", "miliardi", "bilhão",
           "bilhões", "miljard", "miljarden", "Milliarde", "Milliarden"}
OF = {"es": "de", "pt": "de", "it": "di", "fr": "d'"}
# no unit-less "<units> und <cents>" in German / Dutch: "achtundvierzig und zehn" is heard as two numbers
NO_BARE_JOIN = {"de", "nl"}


FEM_UNITS = {"libra", "libras", "livre", "livres", "sterlina", "sterline"}
_MASC_ONLY = re.compile(r"^(?:un|uno|um|dois|ein|veintiún|veintiuno)$|ientos$|entos$")
_FEM_ONLY = re.compile(r"^(?:una|uma|une|duas|eine|einer|einem|einen|veintiuna)$|ientas$|entas$")


def _amount_words(lang: str, n: int, fem: bool = False) -> list[list[str]]:
    """The amount before a currency unit, agreeing with the unit's gender ("un euro", "una libra", "duzentos reais")."""
    c = g(lang, "card", n)
    vs = [c] + _card_variants(c, lang, n, year=False)
    if lang == "de":
        vs = [["ein"] if v == ["eins"] else v for v in vs if v not in (["eine"], ["einen"], ["einem"], ["einer"])]
    elif lang == "fr" and fem:
        vs = [v[:-1] + ["une"] if v and v[-1] == "un" else v for v in vs]
    bad = _MASC_ONLY if fem else _FEM_ONLY
    keep = [v for v in vs if not any(bad.search(w) for w in v)]
    if lang in ("es", "it") and not fem:  # "uno euro" -> "un euro", "veintiuno" -> "veintiún"
        keep = [v for v in keep if not v or v[-1] not in ("uno", "veintiuno")] or keep
    return keep or vs


def _with_unit(lang: str, ws: list[str], unit: str) -> list[str]:
    if ws and ws[-1] in MILLION and lang in OF:  # "un millón de euros", "un million d'euros", "un million de dollars"
        if lang == "fr":
            return ws + (["d'" + unit] if unit[0] in "aeiouéh" else ["de", unit])
        return ws + [OF[lang], unit]
    return ws + [unit]


def _cur_cands(d: dict, lang: str) -> list[str]:
    n, frac, sym = _int(d["int"]), d["frac"] or "", d["sym"]
    out = []
    if sym in ("€", "$"):
        try:
            out.append(g(lang, "cur", n, frac, sym))
        except Missing:
            pass
    if sym not in UNIT[lang]:
        return _uniq(out)
    units = [UNIT[lang][sym]] + ([ALT_UNIT[(lang, sym)]] if (lang, sym) in ALT_UNIT else [])
    # the cents word belongs to the currency: dollars, pesos and reais have centavos, never the euro's céntimos /
    # cêntimos ("veintinueve pesos con diez céntimos" was a unit x cents-word cross product)
    cents = [ALT_CENT[(lang, sym)]] if (lang, sym) in ALT_CENT else [CENT[lang]]
    cents_n = int(frac) if frac and len(frac) == 2 else 0
    if frac and len(frac) != 2:
        return _uniq(out)
    for sing, plur in units:
        unit = sing if n == 1 else plur
        for a in _amount_words(lang, n, fem=plur in FEM_UNITS) if n else [[]]:
            if not cents_n:
                if n:
                    out.append(_with_unit(lang, a, unit))
                continue
            cw = g(lang, "card", cents_n)
            for cs, cp in cents:
                cu = cs if cents_n == 1 else cp
                if not n:
                    out.append(cw + [cu])  # "0,50 €" -> "cinquenta cêntimos"
                    continue
                for j in JOIN[lang]:
                    jw = [j] if j else []
                    out.append(a + [unit] + jw + cw + [cu])
                    out.append(a + [unit] + jw + cw)
            if n and lang not in NO_BARE_JOIN:
                for j in JOIN[lang]:
                    if j and cents_n >= 10:  # "doze e cinquenta" (no unit); under 10 the zero is gone: "trinta e cinco" is 35
                        out.append(a + [j] + cw)
                # decimal reading with the unit after ("doze vírgula cinquenta reais")
                try:
                    dw = g(lang, "dec", n, frac)
                    out.append(dw + [plur])
                    out += [v + [plur] for v in _dec_frac_variant(lang, n, frac, dw)]
                except Missing:
                    pass
    return _uniq(out)


# clock: hour unit word(s), the "and" joiner, and the half / quarter words (hh:30, hh:15)
# Italian has no hour word after the hour: "sedici ore e un quarto" is a duration ("le sedici e un quarto")
HOUR_UNIT = {"de": ["Uhr"], "es": ["horas"], "fr": ["heures", "heure"], "it": [], "nl": ["uur"],
             "pt": ["horas", "hora"]}
SING_HOUR_UNIT, PLUR_HOUR_UNIT = {"hora", "heure"}, {"horas", "heures"}
# French joins hour and minutes with no word: "seize heures quinze" ("seize heures et quinze" is not French; "et quart"
# / "et demie" come from QUARTER_PAST / HALF_PAST)
CLOCK_JOIN = {"de": [""], "es": ["y", ""], "fr": [""], "it": ["e", ""], "nl": [""], "pt": ["e", ""]}
# the hour is feminine in es / fr / pt ("la una", "une heure", "vingt et une heures", "às duas", "vinte e duas"): the
# masculine last word of an hour number -> the feminine one
FEM_HOUR = {"es": {"uno": "una", "un": "una", "veintiuno": "veintiuna", "veintiún": "veintiuna"},
            "fr": {"un": "une"}, "pt": {"um": "uma", "dois": "duas"}}
MIDNIGHT = {"fr": "minuit"}  # 12:xx a.m. ("douze heures" is noon)
# 0:xx is never read with the 12 hour in any language: 0:55 is "zero e cinquantacinque", "null Uhr dreißig", "zéro
# heure trois", "cero y veintitrés", "nul uur elf", never "dodici ..." / "zwölf Uhr dreißig" / "douze trois" / "doce
# veintitrés" / "twaalf uur elf" (that is 12:xx, noon). The 12-hour reading is only offered for 13:00-23:59
NO_NOON_MIDNIGHT = True
# French says the hour word in the singular for 0 and 1 ("zéro heure trois", "une heure"; "zéro heures" is a slip)
SING_HOUR_ZERO = {"fr"}


def _fem_hour(ws: list[str], lang: str, hh: int) -> list[str]:
    """The hour words of hour hh with the hour's feminine form (FEM_HOUR); other languages / hours unchanged."""
    fh = FEM_HOUR.get(lang)
    if not fh or not ws or ws[-1] not in fh:
        return ws
    return ws[:-1] + [fh[ws[-1]]]
HALF_PAST = {"es": ["y", "media"], "fr": ["et", "demie"], "it": ["e", "mezza"], "pt": ["e", "meia"]}
QUARTER_PAST = {"es": ["y", "cuarto"], "fr": ["et", "quart"], "it": ["e", "un", "quarto"], "pt": ["e", "um", "quarto"]}


def _clock_cands(d: dict, lang: str) -> list[str]:
    base = _clock_core(d, lang)
    ap = d.get("ap")
    if not ap:
        return base
    return _uniq([c.split() + w.split() for c in base for w in [""] + DAYPART[lang][ap]])


def _clock_core(d: dict, lang: str) -> list[str]:
    h, m = d["h"], d["m"]
    canon = g(lang, "clock", h=h, m=m)
    hw0 = g(lang, "card", h)
    if canon[:len(hw0)] == hw0:  # the table's own hour words, feminine where the hour is ("um e dez" -> "uma e dez")
        canon = _fem_hour(hw0, lang, h) + canon[len(hw0):]
        if h == 0 and lang in SING_HOUR_ZERO and canon[len(hw0):len(hw0) + 1] == ["heures"]:
            canon = canon[:len(hw0)] + ["heure"] + canon[len(hw0) + 1:]  # "zéro heures trois" -> "zéro heure trois"
    out = [canon]
    hours = {h} | ({h % 12} if h > 12 else set())  # NO_NOON_MIDNIGHT: 0:xx never gets the 12 hour
    if lang in MIDNIGHT and d.get("ap") == "a" and h == 12:
        hours, out = set(), []  # 12:xx a.m.: midnight, not "douze heures"
        mw = [MIDNIGHT[lang]]
        out.append(mw if m == 0 else mw + g(lang, "card", m))
        out.append(g(lang, "card", 0) + ["heure"] + (g(lang, "card", m) if m else []))  # "zéro heure quarante cinq"
        if m == 30 and lang in HALF_PAST:
            out.append(mw + HALF_PAST[lang])
        if m == 15 and lang in QUARTER_PAST:
            out.append(mw + QUARTER_PAST[lang])
    for hh in sorted(hours):
        hw = g(lang, "card", hh)
        # "una", "uma", "duas"; a masculine hour form is never one of them (FEM_HOUR)
        hvs = [_fem_hour(v, lang, hh) for v in [hw] + _card_variants(hw, lang, hh, year=False)]
        hvs = [v for i, v in enumerate(hvs) if v not in hvs[:i]]
        for hv in hvs:
            # the hour word agrees with the hour: singular "hora" / "heure" for 1 (and 0, "zero hora", "zéro heure"),
            # plural otherwise ("quinze hora e meia", "une heures" were generator slips, judge audit 2026-10-17)
            heads = [hv] + [hv + [u] for u in HOUR_UNIT[lang]
                            if not (u in SING_HOUR_UNIT and hh not in (0, 1))
                            and not (u in PLUR_HOUR_UNIT and (hh == 1 or (hh == 0 and lang in SING_HOUR_ZERO)))]
            if m == 0:
                out += heads
                continue
            mw = g(lang, "card", m)
            mvs = [mw] + ([g(lang, "card", 0) + mw] if m < 10 else [])  # "catorce cero cinco"
            for head in heads:
                for j in CLOCK_JOIN[lang]:
                    out += [head + ([j] if j else []) + v for v in mvs]
                if m == 30 and lang in HALF_PAST:
                    out.append(head + HALF_PAST[lang])
                if m == 15 and lang in QUARTER_PAST:
                    out.append(head + QUARTER_PAST[lang])
    return _uniq(out)


# "x and a half" for a fraction of 5 ("2,5" -> "zweieinhalb", "dos y medio")
def _half_variants(lang: str, n: int) -> list[list[str]]:
    if n == 0:
        return []
    c = g(lang, "card", n)
    if lang == "de":
        head = "ein" if c == ["eins"] else "".join(c)
        return [[head + "einhalb"]] + ([["anderthalb"]] if n == 1 else [])
    if lang == "nl":
        if n == 1:
            return [["anderhalf"], ["eenenhalf"]]
        w = "".join(c)
        return [[w + ("ënhalf" if w.endswith("e") else "enhalf")]]
    tail = {"es": [["y", "medio"], ["y", "media"]], "fr": [["et", "demi"], ["et", "demie"]],
            "it": [["e", "mezzo"], ["e", "mezza"]], "pt": [["e", "meio"], ["e", "meia"]]}[lang]
    return [c + t for t in tail]


def _dec_frac_variant(lang, n_int, frac, canonical) -> list[list[str]]:
    """Fraction read as one number ("null Komma sechsundneunzig") next to the canonical digit-by-digit reading."""
    if not frac or len(frac) > 2 or frac.startswith("0") or len(frac) < 2:
        return []
    iw = g(lang, "card", n_int)
    sep = canonical[len(iw)] if len(canonical) > len(iw) else None
    if sep is None:
        return []
    return [iw + [sep] + g(lang, "card", int(frac))]


def _uniq(xs):
    seen, out = set(), []
    for x in xs:
        s = " ".join(x)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _cands_core(d: dict, lang: str) -> list[str]:
    k = d["kind"]
    if k == "skip":
        return []
    if k == "clock":
        return _clock_cands(d, lang)
    if k == "phone":
        return _phone_cands(d, lang)
    raw, frac = d["int"], d["frac"]
    n = _int(raw)
    lead0 = len(re.sub(r"\D", "", raw)) > 1 and raw.startswith("0")
    if k in ("int", "code"):
        if lead0:  # a code / phone run said in chunks ("0612" -> "zéro six douze")
            return _uniq([_digits(lang, raw)] + _seg_readings(lang, raw))
        if n > 999_999_999_999:
            return []
        out = []
        is_year = re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", raw) is not None
        if is_year:
            y = g(lang, "year", n)
            c = g(lang, "card", n)
            out += [y, c] + _card_variants(c, lang, n, year=True)
            if lang == "fr" and 1100 <= n <= 1999:
                out.append(_hundreds(lang, n))
            if lang == "nl" and 1100 <= n <= 1999 and n % 100:  # "achttien negenendertig", "negentien nul vijf"
                hi, lo = divmod(n, 100)
                out.append(g(lang, "card", hi) + (g(lang, "card", 0) if lo < 10 else []) + g(lang, "card", lo))
            if "year_spoken" in CAND_FIXES:
                out += _year_spoken(lang, n)
        else:
            c = g(lang, "card", n)
            out += [c] + _card_variants(c, lang, n, year=False)
            if len(raw) >= 5 and re.fullmatch(r"\d+", raw):
                out += [_digits(lang, raw)] + _seg_readings(lang, raw)
            elif len(raw) == 4 and "int_pairs" in CAND_FIXES and re.fullmatch(r"\d+", raw) and raw[2:] != "00":
                lo = raw[2:]  # "5707" -> "cinquantasette zero sette", "4762" -> "quarantasette sessantadue"
                out.append(g(lang, "card", int(raw[:2])) + (_digits(lang, lo) if lo[0] == "0" else g(lang, "card", int(lo))))
                out.append(_digits(lang, raw))
        return _uniq(out)
    if k == "ord":
        if n >= 1_000_000:
            return []  # the Spanish grammar has no reading for the ordinal of 1,000,000
        o = g(lang, "ord", n)
        out = [o] + _ord_variants(o, lang, d.get("fem", False))
        if d.get("fem"):
            out = [_fem_words(o, lang)] + out if lang in ("es", "pt", "it", "fr") else out
        if d.get("mark") == "." and "ord_card" in CAND_FIXES:
            # "Nummer 68. Abstimmung": a number that ends a sentence, said as a cardinal; the period stays in the label
            raw0 = re.sub(r"\D", "", raw)
            cs = [_digits(lang, raw0)] if _lead0(raw0) else _plain_int_alts(lang, raw0, cap=3)  # years: year readings
            out += [c[:-1] + [c[-1] + "."] for c in cs]
        return _uniq(out)
    if k == "dec":
        c = g(lang, "dec", n, frac)
        base = [c] + _dec_frac_variant(lang, n, frac, c)
        if d.get("dsep") == ".":  # "2.0", "3.14" read with the point word ("dos punto cero")
            iw = g(lang, "card", n)
            base += [v[:len(iw)] + [POINT[lang]] + v[len(iw) + 1:] for v in base if len(v) > len(iw) and v[:len(iw)] == iw]
        extra = []
        if frac and re.fullmatch(r"\d+", raw):
            iw = g(lang, "card", n)
            fws = [_digits(lang, frac)] + ([g(lang, "card", int(frac))] if frac[0] != "0" and len(frac) > 1 else [])
            if d.get("dsep") == "." and "dot_bare" in CAND_FIXES:  # "articolo 5.2" -> "cinque due"
                extra += [iw + f for f in fws]
            if d.get("dsep") == "," and lang == "it" and "it_comma" in CAND_FIXES:
                extra += [iw + ["comma"] + f for f in fws]
            if (d.get("dsep") == "." and "dot_clock" in CAND_FIXES and len(frac) == 2 and n <= 23 and int(frac) <= 59
                    and not d.get("sign")):
                try:  # an optional reading: a grammar gap must not cost the span its decimal readings
                    extra += [c.split() for c in _clock_core({"h": n, "m": int(frac)}, lang)]
                except Missing:
                    pass
        return _uniq(base + (_half_variants(lang, n) if frac == "5" else []) + extra)
    if k == "pct":
        f = frac or ""
        c = g(lang, "pct", n, f)
        out = [c]
        if f:
            dc = g(lang, "dec", n, f)
            tail = c[len(dc):] if c[:len(dc)] == dc else None
            if tail is not None:
                out += [v + tail for v in _dec_frac_variant(lang, n, f, dc)]
                if f == "5":
                    out += [v + tail for v in _half_variants(lang, n)]
        else:
            cc = g(lang, "card", n)
            tail = c[len(cc):] if c[:len(cc)] == cc else None
            if tail is not None:
                out += [v + tail for v in _card_variants(cc, lang, n, year=False) if v]
        return _uniq(out)
    if k == "cur":
        return _cur_cands(d, lang)
    return []


def spans(text: str, lang: str) -> list[Span]:
    out = []
    for d in _find(text, lang):
        try:
            cs = _cands(d, lang)
            if cs and lang in LANGS and CAND_FIXES & VARIETY:
                cs = _regional(d, lang, cs)
            note = d.get("note", "")
        except Missing as e:
            cs, note = [], f"missing {e}"
        except (ValueError, IndexError) as e:  # e.g. pt "0.011" read as a digit string with a point
            cs, note = [], f"error {e!r}"[:80]
        out.append(Span(d["a"], d["b"], text[d["a"]:d["b"]], d["kind"], cs, note))
    return out


def needed(texts, lang: str) -> set[str]:
    """Every table key the spans of `texts` need (components included)."""
    global _REC
    _REC = _Recorder()
    try:
        for t in texts:
            for d in _find(t, lang):
                try:
                    _cands(d, lang)
                except Exception:
                    pass
            for i in range(1, 32):  # day ordinals, roman numerals up to XXX
                _REC.get(_key(lang, "ord", i))
            for i in range(1900, 2100):  # two-digit years in dates
                _REC.get(_key(lang, "year", i))
            for i in range(0, 1000):  # digit-by-digit, phone / code chunks (1-3 digits), hundreds components, fractions
                _REC.get(_key(lang, "card", i))
        return set(_REC.keys)
    finally:
        _REC = None


_MINOR_SG = {"centavo": "centavos", "céntimo": "céntimos", "cêntimo": "cêntimos"}
_ONE_WORDS = {"un", "uno", "um", "una", "uma"}


def _fix_minor_plural(ws):
    """The dumper writes "dos centavo" for $0.02: a minor unit after any count but one is plural."""
    if not isinstance(ws, list):
        return ws
    return [_MINOR_SG[w] if w in _MINOR_SG and i > 0 and ws[i - 1] not in _ONE_WORDS else w for i, w in enumerate(ws)]


def build_table(texts_by_lang: dict, dumper: str, workdir: str) -> int:
    """Ask a grammar dumper (not in this repo) for every needed key and merge the answers into ml_grammar_table.json."""
    import subprocess

    keys = sorted({k for lang, ts in texts_by_lang.items() for k in needed(ts, lang)})
    tab = table()
    keys = [k for k in keys if k not in tab]
    reqs = []
    for k in keys:
        lang, kind, n, f, s, h, m = k.split("|")
        reqs.append({"l": lang, "k": kind, "n": int(n), "f": f, "s": s or "€", "h": int(h), "m": int(m)})
    if reqs:
        rq, an = os.path.join(workdir, "requests.json"), os.path.join(workdir, "answers.json")
        json.dump(reqs, open(rq, "w"))
        subprocess.run([dumper, rq, an], check=True)
        for k, w in zip(keys, json.load(open(an))):
            tab[k] = _fix_minor_plural(w)
    json.dump(tab, open(TABLE_PATH, "w"), ensure_ascii=False, sort_keys=True)
    return len(keys)


def render(text: str, sp: list[Span], choice: list[int]) -> str:
    out, pos = [], 0
    for s, c in zip(sp, choice):
        out.append(text[pos:s.start])
        out.append(s.cands[c])
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def all_digits_spanned(text: str, lang: str) -> bool:
    """True when the text has digits, every digit lies inside a number span that can have readings (not a skipped
    range/sign/degree/glued code/ambiguous grouping). P3 prefilter: a row with an unreadable number can never be kept."""
    ds = _find(text, lang)
    if not ds or any(d["kind"] == "skip" for d in ds) or not any(c.isdigit() for c in text):
        return False
    cover = [False] * len(text)
    for d in ds:
        for i in range(d["a"], d["b"]):
            cover[i] = True
    return all(cover[i] for i, c in enumerate(text) if c.isdigit())


# ================================================================ wave-2 classes
# Units, dates, ranges, decades, codes, roman numerals, fractions, versions, "&", per mille, signs. Each class adds its
# own span kind; the readings of a value always come from the grammar (g) or the word tables below, never a different
# value. A class that cannot be read safely leaves its digits unspanned (the caption is then dropped, never mislabelled).

POINT = {"de": "Punkt", "es": "punto", "fr": "point", "it": "punto", "nl": "punt", "pt": "ponto"}
AND_WORD = {"de": "und", "es": "y", "fr": "et", "it": "e", "nl": "en", "pt": "e"}
MINUS_WORDS = {"de": ["minus"], "es": ["menos"], "fr": ["moins"], "it": ["meno"], "nl": ["min", "minus"], "pt": ["menos"]}
PLUS_WORDS = {"de": ["plus"], "es": ["más"], "fr": ["plus"], "it": ["più"], "nl": ["plus"], "pt": ["mais"]}
SIGN_OK_BEFORE = re.compile(r"[\s(\[«\"“‘]")


def _free(taken, a, b) -> bool:
    return not any(x < b and a < y for x, y in taken)


def _add(out, taken, d) -> None:
    out.append(d)
    taken.append((d["a"], d["b"]))


def _lead0(s: str) -> bool:
    return len(s) > 1 and s.startswith("0")


def _wl(ws) -> str:
    return " ".join(ws)


# ---------------------------------------------------------------- units

_UNIT_SRC = {
    "es": """km|kilómetro|kilómetros
m|metro|metros
cm|centímetro|centímetros
mm|milímetro|milímetros
kg|kilogramo;kilo|kilogramos;kilos
g|gramo|gramos
mg|miligramo|miligramos
t|tonelada|toneladas|f
l|litro|litros
ml|mililitro|mililitros
h|hora|horas|f
min|minuto|minutos
s|segundo|segundos
km/h|kilómetro por hora;kilómetro a la hora;kilómetro hora|kilómetros por hora;kilómetros a la hora;kilómetros hora;kilómetros horarios
m²|metro cuadrado|metros cuadrados
m³|metro cúbico|metros cúbicos
km²|kilómetro cuadrado|kilómetros cuadrados
kW|kilovatio;kilowatt|kilovatios;kilowatts
kWh|kilovatio hora;kilovatio por hora;kilowatt hora|kilovatios hora;kilovatios por hora;kilowatts hora;kilowatts por hora
MW|megavatio;megawatt|megavatios;megawatts
GW|gigavatio;gigawatt|gigavatios;gigawatts
GB|gigabyte;giga|gigabytes;gigas
MB|megabyte;mega|megabytes;megas
TB|terabyte;tera|terabytes;teras
°C|grado centígrado;grado Celsius;grado|grados centígrados;grados Celsius;grados
°F|grado Fahrenheit;grado|grados Fahrenheit;grados
°|grado|grados
‰|por mil|por mil""",
    "pt": """km|quilômetro;quilómetro|quilômetros;quilómetros
m|metro|metros
cm|centímetro|centímetros
mm|milímetro|milímetros
kg|quilograma;quilo|quilogramas;quilos
g|grama|gramas
mg|miligrama|miligramas
t|tonelada|toneladas|f
l|litro|litros
ml|mililitro|mililitros
h|hora|horas|f
min|minuto|minutos
s|segundo|segundos
km/h|quilômetro por hora;quilómetro por hora;quilômetro horário;quilómetro horário|quilômetros por hora;quilómetros por hora;quilômetros horários;quilómetros horários;quilômetros hora
m²|metro quadrado|metros quadrados
m³|metro cúbico|metros cúbicos
km²|quilômetro quadrado;quilómetro quadrado|quilômetros quadrados;quilómetros quadrados
ha|hectare|hectares
kW|quilowatt;kilowatt|quilowatts;kilowatts
kWh|quilowatt hora;kilowatt hora;quilowatt por hora|quilowatts hora;quilowatts por hora;kilowatts hora;kilowatts por hora
MW|megawatt|megawatts
GW|gigawatt|gigawatts
GB|gigabyte;giga|gigabytes;gigas
MB|megabyte;mega|megabytes;megas
TB|terabyte;tera|terabytes;teras
°C|grau Celsius;grau centígrado;grau|graus Celsius;graus centígrados;graus
°F|grau Fahrenheit;grau|graus Fahrenheit;graus
°|grau|graus
‰|por mil|por mil""",
    "fr": """km|kilomètre|kilomètres
m|mètre|mètres
cm|centimètre|centimètres
mm|millimètre|millimètres
kg|kilogramme;kilo|kilogrammes;kilos
g|gramme|grammes
mg|milligramme|milligrammes
t|tonne|tonnes|f
l|litre|litres
ml|millilitre|millilitres
h|heure|heures|f
min|minute|minutes|f
s|seconde|secondes|f
km/h|kilomètre heure;kilomètre par heure;kilomètre à l'heure|kilomètres heure;kilomètres par heure;kilomètres à l'heure
m²|mètre carré|mètres carrés
m³|mètre cube|mètres cubes
km²|kilomètre carré|kilomètres carrés
ha|hectare|hectares
kW|kilowatt|kilowatts
kWh|kilowattheure;kilowatt heure|kilowattheures;kilowatts heures;kilowatt heures
MW|mégawatt|mégawatts
GW|gigawatt|gigawatts
GB|gigaoctet;giga octet;giga|gigaoctets;giga octets;gigas
MB|mégaoctet;méga octet;méga|mégaoctets;méga octets;mégas
TB|téraoctet;téra octet;téra|téraoctets;téra octets;téras
°C|degré Celsius;degré centigrade;degré|degrés Celsius;degrés centigrades;degrés
°F|degré Fahrenheit;degré|degrés Fahrenheit;degrés
°|degré|degrés
‰|pour mille|pour mille""",
    "it": """km|chilometro|chilometri
m|metro|metri
cm|centimetro|centimetri
mm|millimetro|millimetri
kg|chilogrammo;chilo|chilogrammi;chili
g|grammo|grammi
mg|milligrammo|milligrammi
t|tonnellata|tonnellate|f
l|litro|litri
ml|millilitro|millilitri
h|ora|ore|f
min|minuto|minuti
s|secondo|secondi
km/h|chilometro orario;chilometro all'ora;chilometro l'ora;chilometro per ora|chilometri orari;chilometri all'ora;chilometri l'ora;chilometri per ora
m²|metro quadrato;metro quadro|metri quadrati;metri quadri
m³|metro cubo|metri cubi
km²|chilometro quadrato;chilometro quadro|chilometri quadrati;chilometri quadri
kW|chilowatt|chilowatt
kWh|chilowattora;chilowatt ora;chilowatt all'ora|chilowattora;chilowatt ora;chilowatt all'ora
MW|megawatt|megawatt
GW|gigawatt|gigawatt
GB|gigabyte;giga|gigabyte;giga
MB|megabyte;mega|megabyte;mega
TB|terabyte;tera|terabyte;tera
°C|grado centigrado;grado Celsius;grado|gradi centigradi;gradi Celsius;gradi
°F|grado Fahrenheit;grado|gradi Fahrenheit;gradi
°|grado|gradi
‰|per mille|per mille""",
    "nl": """km|kilometer|kilometer;kilometers
m|meter|meter;meters
cm|centimeter|centimeter;centimeters
mm|millimeter|millimeter;millimeters
kg|kilogram;kilo|kilogram;kilo;kilo's
g|gram|gram;grammen
mg|milligram|milligram;milligrammen
t|ton|ton;tonnen
l|liter|liter;liters
ml|milliliter|milliliter;milliliters
h|uur|uur;uren
min|minuut|minuten
s|seconde|seconden
km/h|kilometer per uur;kilometer in het uur|kilometer per uur;kilometers per uur;kilometer in het uur
m²|vierkante meter|vierkante meter;vierkante meters
m³|kubieke meter|kubieke meter;kubieke meters
km²|vierkante kilometer|vierkante kilometer;vierkante kilometers
ha|hectare|hectare;hectares
kW|kilowatt|kilowatt
kWh|kilowattuur;kilowatt uur|kilowattuur;kilowatt uur
MW|megawatt|megawatt
GW|gigawatt|gigawatt
GB|gigabyte;giga|gigabyte;gigabytes;giga
MB|megabyte;mega|megabyte;megabytes;mega
TB|terabyte;tera|terabyte;terabytes;tera
°C|graad Celsius;graad|graden Celsius;graden
°F|graad Fahrenheit;graad|graden Fahrenheit;graden
°|graad|graden
‰|promille|promille""",
    "de": """km|Kilometer|Kilometer
m|Meter|Meter
cm|Zentimeter|Zentimeter
mm|Millimeter|Millimeter
kg|Kilogramm;Kilo|Kilogramm;Kilo
g|Gramm|Gramm
mg|Milligramm|Milligramm
t|Tonne|Tonnen|f
l|Liter|Liter
ml|Milliliter|Milliliter
h|Stunde|Stunden|f
min|Minute|Minuten|f
s|Sekunde|Sekunden|f
km/h|Kilometer pro Stunde;Kilometer in der Stunde;Kilometer die Stunde;Stundenkilometer|Kilometer pro Stunde;Kilometer in der Stunde;Kilometer die Stunde;Stundenkilometer
m²|Quadratmeter|Quadratmeter
m³|Kubikmeter|Kubikmeter
km²|Quadratkilometer|Quadratkilometer
ha|Hektar|Hektar
kW|Kilowatt|Kilowatt
kWh|Kilowattstunde|Kilowattstunden
MW|Megawatt|Megawatt
GW|Gigawatt|Gigawatt
GB|Gigabyte;Giga|Gigabyte;Gigabytes;Giga
MB|Megabyte;Mega|Megabyte;Megabytes;Mega
TB|Terabyte;Tera|Terabyte;Terabytes;Tera
°C|Grad Celsius;Grad|Grad Celsius;Grad
°F|Grad Fahrenheit;Grad|Grad Fahrenheit;Grad
°|Grad|Grad
‰|Promille|Promille""",
}


def _parse_units() -> dict:
    t = {}
    for lang, src in _UNIT_SRC.items():
        t[lang] = {}
        for line in src.splitlines():
            p = line.split("|")
            t[lang][p[0]] = (p[1].split(";"), p[2].split(";"), len(p) > 3 and p[3] == "f")
    return t


UNITS = _parse_units()
_UNIT_CI = {"km", "cm", "mm", "kg", "mg", "ml", "ha", "min", "km/h", "km²", "m²", "m³"}
_UNIT_ALIAS = {"m2": "m²", "m3": "m³", "km2": "km²", "kmh": "km/h", "L": "l"}
ALL_UNITS = sorted({a for t in UNITS.values() for a in t})  # "ha" is the verb in es / it: those tables omit it
_UNIT_ALT = "|".join(re.escape(a) for a in sorted(ALL_UNITS + list(_UNIT_ALIAS), key=len, reverse=True))
_UNIT_TAIL = re.compile(r"[   ]?(?P<u>" + _UNIT_ALT + r")(?![\w'’²³/]|\.\d)", re.I)
_UNIT_RES: dict = {}


def _unit_norm(raw: str) -> str | None:
    """Canonical abbreviation of a matched unit, or None when its case is wrong ("S", "H", "MW" written "mw")."""
    if raw.startswith("°"):
        return "°" + raw[1:].upper() if len(raw) > 1 else "°"
    if raw in _UNIT_ALIAS:
        return _UNIT_ALIAS[raw]
    if raw in ALL_UNITS:
        return raw
    low = raw.lower()
    low = _UNIT_ALIAS.get(low, low)
    return low if low in _UNIT_CI else None


def _unit_at(text: str, pos: int, lang: str):
    """(canonical unit, end) of a unit word that starts at text[pos] (optional blank first), else None."""
    m = _UNIT_TAIL.match(text, pos)
    if not m:
        return None
    u = _unit_norm(m["u"])
    return (u, m.end()) if u and u in UNITS[lang] else None


def _unit_re(lang):
    if lang not in _UNIT_RES:
        _UNIT_RES[lang] = re.compile(
            r"(?<![\w.,/])(?P<sign>[-−+])?(?P<int>\d{1,3}(?:" + GROUP[lang] + r"\d{3})+(?!\d)|\d+)"
            r"(?:(?P<dsep>[,.])(?P<frac>\d+))?(?P<tail>[   ]?(?:" + _UNIT_ALT + r"))(?![\w'’²³/]|\.\d)", re.I)
    return _UNIT_RES[lang]


def _find_units(text, lang, out, taken):
    for mt in _unit_re(lang).finditer(text):
        a, b = mt.start(), mt.end()
        um = _unit_at(text, mt.start("tail"), lang)
        if um is None or um[1] != b or not _free(taken, a, b):
            continue
        d = {"a": a, "b": b, "kind": "unit", "int": mt["int"], "frac": mt["frac"], "dsep": mt["dsep"], "unit": um[0]}
        if mt["sign"]:
            if a > 0 and not SIGN_OK_BEFORE.match(text[a - 1]):
                d.update(kind="skip", note="range/sign")
            else:
                d["sign"] = mt["sign"]
        elif a > 0 and text[a - 1] in "-–−+":
            d.update(kind="skip", note="range/sign")
        elif mt["frac"] is not None and len(mt["frac"]) == 3 and not (lang == "fr" and mt["dsep"] == ","):
            d.update(kind="skip", note="ambiguous grouping/decimal")
        if mt["sign"] and mt["frac"] is not None and len(mt["frac"]) == 3 and not (lang == "fr" and mt["dsep"] == ","):
            d.update(kind="skip", note="ambiguous grouping/decimal")
        if len(re.sub(r"\D", "", mt["int"])) > 1 and mt["int"].startswith("0"):
            d.update(kind="skip", note="leading zero before unit")
        _add(out, taken, d)


def _unit_phrases(lang: str, abbr: str, plural: bool) -> list[str]:
    sg, pl, _ = UNITS[lang][abbr]
    return list(pl if plural else sg)


def _int_amounts(lang: str, n: int, fem: bool, neutral: bool) -> list[list[str]]:
    if n > 999_999_999_999:
        return []
    if neutral:
        c = g(lang, "card", n)
        return [c] + [v for v in _card_variants(c, lang, n, year=False) if v]
    if lang == "de" and n == 1:
        return [["eine"]] if fem else [["ein"]]
    return _amount_words(lang, n, fem)


_FEM_HALF = {"media", "meia", "mezza", "demie"}


def _unit_cands(d: dict, lang: str) -> list[str]:
    abbr = d["unit"]
    sg, pl, fem = UNITS[lang][abbr]
    neutral = abbr == "‰"
    raw, frac = d["int"], d["frac"]
    n = _int(raw)
    out: list[list[str]] = []
    if frac is None:
        forms = [False] if n == 1 else [True]
        if lang == "fr" and n == 0:
            forms = [False, True]
        for plural in forms:
            for a in _int_amounts(lang, n, fem, neutral):
                for ph in _unit_phrases(lang, abbr, plural):
                    uw = ph.split()
                    if lang == "it" and fem and not plural and a == ["una"] and uw[0][0] in "aeiou":
                        out.append(["un'" + uw[0]] + uw[1:])
                        continue
                    out.append(a + uw)
        return _uniq(out)
    c = g(lang, "dec", n, frac)
    amounts = [c] + _dec_frac_variant(lang, n, frac, c)
    forms = [True]
    if lang == "fr" and (n == 0 or (n == 1)):
        forms = [False, True]
    for plural in forms:
        for a in amounts:
            for ph in _unit_phrases(lang, abbr, plural):
                out.append(a + ph.split())
    if frac == "5" and n > 0 and not neutral:
        if lang in ("de", "nl"):
            for hv in _half_variants(lang, n):
                for ph in _unit_phrases(lang, abbr, True):
                    out.append(hv + ph.split())
        else:
            cw = g(lang, "card", n)
            for a in _int_amounts(lang, n, fem, False):
                for hv in _half_variants(lang, n):
                    tail = hv[len(cw):]
                    if hv[:len(cw)] != cw or not tail or (tail[-1] in _FEM_HALF) != fem:
                        continue  # "dos horas y media" / "dos kilómetros y medio": the tail agrees with the unit
                    for ph in _unit_phrases(lang, abbr, n != 1):
                        out.append(a + ph.split() + tail)
    return _uniq(out)


# ---------------------------------------------------------------- dates (one span per part: day, ".month", ".year")

DATE_FULL = re.compile(r"(?<![\w.,/\-–+:])(?P<d>\d{1,2})(?P<s>[./-])(?P<m>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})(?!\d|[./-]\d|,\d)")
DATE_DM = re.compile(r"(?<![\w.,/\-–+:])(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?!\d|[\w]\d)")
_MONTH_ALT = {("es", 8): ["septiembre", "setiembre"]}


def _month_names(lang: str, m: int) -> list[str]:
    ws = MONTHS[lang].split()
    if lang == "es":
        ws = [w for w in ws if w != "setiembre"]
    names = _MONTH_ALT.get((lang, m - 1), [ws[m - 1]])
    return [w.capitalize() for w in names] if lang == "de" else list(names)


def _find_dates(text, lang, out, taken):
    for mt in DATE_FULL.finditer(text):
        dd, mm, ys, s = int(mt["d"]), int(mt["m"]), mt["y"], mt["s"]
        ok = 1 <= dd <= 31 and 1 <= mm <= 12
        if len(ys) == 4:
            ok = ok and 1900 <= int(ys) <= 2099
        elif s == ".":
            ok = ok and len(mt["d"]) == 2 and len(mt["m"]) == 2
        if not ok or not _free(taken, mt.start(), mt.end()):
            continue
        for part, a, b, v in (("d", mt.start("d"), mt.end("d"), dd), ("m", mt.end("d"), mt.end("m"), mm),
                              ("y", mt.end("m"), mt.end("y"), ys)):
            _add(out, taken, {"a": a, "b": b, "kind": "date", "part": part, "val": v, "lead": part != "d"})
    if lang in ("de", "nl"):  # "12.05." (no year)
        for mt in DATE_DM.finditer(text):
            dd, mm = int(mt["d"]), int(mt["m"])
            if not (1 <= dd <= 31 and 1 <= mm <= 12) or not _free(taken, mt.start(), mt.end()):
                continue
            if re.match(r"\s+[a-zäöüéë]", text[mt.end():]) is None:
                continue  # "versie 4.2." / "12.01." at a sentence end is a number, not a date
            e = mt.end()
            _add(out, taken, {"a": mt.start("d"), "b": mt.end("d"), "kind": "date", "part": "d", "val": dd, "lead": False})
            _add(out, taken, {"a": mt.end("d"), "b": e, "kind": "date", "part": "m", "val": mm, "lead": True})


def _year_alts(lang: str, n: int) -> list[list[str]]:
    c = g(lang, "card", n)
    vs = [g(lang, "year", n), c] + [v for v in _card_variants(c, lang, n, year=True)
                                    if v and not any(_FEM_ONLY.search(w) for w in v)][:2]  # a year is masculine: "dois mil"
    seen, out = set(), []
    for v in vs:
        if _wl(v) not in seen:
            seen.add(_wl(v))
            out.append(v)
    return out


def _date_year_words(lang: str, ys: str) -> list[list[str]]:
    if len(ys) == 4:
        return _year_alts(lang, int(ys))
    yy = int(ys)
    full = 2000 + yy if yy <= 40 else 1900 + yy
    out = _year_alts(lang, full) + [g(lang, "card", yy)]
    if yy < 10:
        out.append(g(lang, "card", 0) + g(lang, "card", yy))
    return out


def _day_words(lang: str, dd: int) -> list[list[str]]:
    c = g(lang, "card", dd)
    if lang == "de":
        o = g(lang, "ord", dd)
        return [o] + _ord_variants(o, lang, False)[:3]
    out = [c] + [v for v in _card_variants(c, lang, dd, year=False) if v]
    if lang == "nl":
        out.append(g(lang, "ord", dd))
    if dd == 1 and lang != "nl":
        o = g(lang, "ord", 1)
        out += [o] + ([[ES_APOC[o[-1]]] for _ in [0] if lang == "es" and o[-1] in ES_APOC])
    return out


def _date_cands(d: dict, lang: str) -> list[str]:
    part, v = d["part"], d["val"]
    if part == "d":
        return _uniq(_day_words(lang, v))
    if part == "m":
        num = g(lang, "card", v)
        names = _month_names(lang, v)
        out = []
        if lang == "de":
            o = g(lang, "ord", v)
            out = [[n] for n in names] + [o] + _ord_variants(o, lang, False)[:2]
        elif lang == "es":
            out = [["de", n] for n in names] + [[n] for n in names] + [["del"] + num, num]
        elif lang == "pt":
            out = [["de", n] for n in names] + [[n] for n in names] + [["do"] + num, num]
        elif lang == "it":
            out = [[n] for n in names] + [["di", n] for n in names] + [num]
        else:  # fr, nl
            out = [[n] for n in names] + [num]
        return [" " + c for c in _uniq(out)]
    ys = _date_year_words(lang, v)
    pref = {"es": [["de"], ["del"], []], "pt": [["de"], []], "it": [[], ["del"]]}.get(lang, [[]])
    return [" " + c for c in _uniq([p + y for p in pref for y in ys])]


# ---------------------------------------------------------------- ranges (left end; joiner + right end [+ unit / %])

RANGE = re.compile(r"(?<![\w.,/:\-–−+])(?P<l>\d{1,4})(?P<sep>[-–]|\s–\s)(?P<r>\d{1,4})(?!\d|[,.]\d|:\d|[-–]\d)")
RANGE_JOIN = {"es": ["a", "al", "hasta", "y"], "pt": ["a", "ao", "até", "e"], "it": ["a", "al", "fino a", "e"],
              "fr": ["à", "au", "et", "jusqu'à"], "de": ["bis", "zu", "und", "auf"], "nl": ["tot", "tot en met", "en", "tegen"]}
RANGE_OR = {"es": ["o", "u"], "pt": ["ou"], "it": ["o"], "fr": ["ou"], "de": ["oder"], "nl": ["of"]}
_CURRENCY_NEAR = re.compile(r"(?:[€$£]\s?|R\$\s?)$")
_CURRENCY_AFTER = re.compile(r"\s?(?:[€$£]|eur\b|usd\b|º|ª)", re.I)
_PCT_AFTER = re.compile(r"\s?%")


def _find_ranges(text, lang, out, taken):
    for mt in RANGE.finditer(text):
        l, r = mt["l"], mt["r"]
        a, b = mt.start(), mt.end()
        if _lead0(l) or (_lead0(r) and len(l) != 4) or not _free(taken, a, b) or _CURRENCY_NEAR.search(text[:a]):
            continue
        tail, e = None, b
        um = _unit_at(text, b, lang)
        pm = _PCT_AFTER.match(text, b)
        if um:
            tail, e = ("unit", um[0]), um[1]
        elif pm:
            tail, e = ("pct", None), pm.end()
        elif _CURRENCY_AFTER.match(text, b) or text[b:b + 1].isalpha() or text[b:b + 1] in "°²³/'’":
            continue
        _add(out, taken, {"a": mt.start("l"), "b": mt.end("l"), "kind": "range", "part": "l", "raw": l})
        _add(out, taken, {"a": mt.end("l"), "b": e, "kind": "range", "part": "r", "raw": r, "left": l, "tail": tail})


def _plain_int_alts(lang: str, raw: str, cap: int = 4) -> list[list[str]]:
    n = int(raw)
    if re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", raw):
        vs = _year_alts(lang, n)
    else:
        c = g(lang, "card", n)
        vs = [c] + [v for v in _card_variants(c, lang, n, year=False) if v]
    return vs[:cap]


def _range_cands(d: dict, lang: str) -> list[str]:
    raw = d["raw"]
    if d["part"] == "l":
        return _uniq(_plain_int_alts(lang, raw))
    left, tail = d["left"], d["tail"]
    ends = _plain_int_alts(lang, raw)
    if len(left) == 4 and len(raw) == 2 and 1000 <= int(left) <= 2099:  # "2010-14", "1995-96"
        full = int(left) // 100 * 100 + int(raw)
        if full <= int(left):
            full += 100
        if 0 < full - int(left) <= 99:
            ends = _year_alts(lang, full)[:2] + [g(lang, "card", int(raw))]
            if int(raw) < 10:
                ends.append(g(lang, "card", 0) + g(lang, "card", int(raw)))
    joins = list(RANGE_JOIN[lang]) + ([""] if int(left) < 100 and int(raw) < 100 or "range_bare" in CAND_FIXES else [])
    if "range_or" in CAND_FIXES:
        joins += [w for w in RANGE_OR[lang] if w not in joins]
    if tail and tail[0] == "pct":
        ends = [w.split() for w in _cands_core({"kind": "pct", "int": raw, "frac": None, "a": 0, "b": 0}, lang)]
    out = []
    for j in joins:
        for e in ends:
            if tail and tail[0] == "unit":
                for ph in _unit_phrases(lang, tail[1], int(raw) != 1):
                    out.append(([j] if j else []) + e + ph.split())
            else:
                out.append(([j] if j else []) + e)
    return [" " + c for c in _uniq(out)]


# ---------------------------------------------------------------- versions / IP addresses (2.0 is a decimal, see dec)

VER = re.compile(r"(?<![\w.,/:\-–+])(?P<g>\d{1,4}(?:\.\d{1,4}){2,})(?!\d|\.\d|,\d|\w)")


def _find_versions(text, lang, out, taken):
    for mt in VER.finditer(text):
        grp = mt["g"].split(".")
        if all(len(x) == 3 for x in grp[1:]) or not _free(taken, mt.start(), mt.end()):
            continue  # "1.000.000": a grouped figure
        pos = mt.start()
        for i, x in enumerate(grp):
            a = pos if i == 0 else pos - 1
            _add(out, taken, {"a": a, "b": pos + len(x), "kind": "version", "raw": x, "lead": i > 0})
            pos += len(x) + 1


def _group_alts(lang: str, x: str) -> list[list[str]]:
    if _lead0(x):
        return [_digits(lang, x)]
    c = g(lang, "card", int(x))
    out = [c] + [v for v in _card_variants(c, lang, int(x), year=False) if v][:2]
    if len(x) in (2, 3):
        out.append(_digits(lang, x))
    return out


def _version_cands(d: dict, lang: str) -> list[str]:
    pre = [POINT[lang]] if d["lead"] else []
    out = [pre + w for w in _group_alts(lang, d["raw"])]
    if d["lead"]:  # "48.500.000.719": said without the point word
        out += _group_alts(lang, d["raw"])
    return [(" " if d["lead"] else "") + c for c in _uniq(out)]


# ---------------------------------------------------------------- fractions

FRAC_SYM = {"½": (1, 2), "¼": (1, 4), "¾": (3, 4), "⅓": (1, 3), "⅔": (2, 3)}
FRAC_SLASH = re.compile(r"(?<![\w.,/\-–])(?:(?P<w>\d{1,3})[  ])?(?P<a>\d{1,2})/(?P<b>\d{1,3})(?![\w]|/\d|[.,]\d)")
FRAC_CHAR = re.compile(r"(?<![\w.,])(?:(?P<w>\d{1,3})[  ]?)?(?P<c>[½¼¾⅓⅔])(?!\w)")
# denominator -> (singular, plural, readings of 1/b without the numerator); alternatives are separated by ";"
_FRAC_SRC = {
    "es": {2: ("medio", "medios", "la mitad;mitad;medio;media"), 3: ("tercio", "tercios", ""), 4: ("cuarto", "cuartos", ""),
           5: ("quinto", "quintos", ""), 6: ("sexto", "sextos", ""), 7: ("séptimo", "séptimos", ""),
           8: ("octavo", "octavos", ""), 9: ("noveno", "novenos", ""), 10: ("décimo", "décimos", "")},
    "pt": {2: ("meio", "meios", "metade;a metade;meio;meia"), 3: ("terço", "terços", ""), 4: ("quarto", "quartos", ""),
           5: ("quinto", "quintos", ""), 6: ("sexto", "sextos", ""), 7: ("sétimo", "sétimos", ""),
           8: ("oitavo", "oitavos", ""), 9: ("nono", "nonos", ""), 10: ("décimo", "décimos", "")},
    "it": {2: ("mezzo", "mezzi", "metà;la metà;mezzo;mezza"), 3: ("terzo", "terzi", ""), 4: ("quarto", "quarti", ""),
           5: ("quinto", "quinti", ""), 6: ("sesto", "sesti", ""), 7: ("settimo", "settimi", ""),
           8: ("ottavo", "ottavi", ""), 9: ("nono", "noni", ""), 10: ("decimo", "decimi", "")},
    "fr": {2: ("demi", "demis", "la moitié;moitié;demi;demie"), 3: ("tiers", "tiers", ""), 4: ("quart", "quarts", ""),
           5: ("cinquième", "cinquièmes", ""), 6: ("sixième", "sixièmes", ""), 7: ("septième", "septièmes", ""),
           8: ("huitième", "huitièmes", ""), 9: ("neuvième", "neuvièmes", ""), 10: ("dixième", "dixièmes", "")},
    "de": {2: ("halb", "halb", "ein halb;halb;halbe;halber;die Hälfte;Hälfte;eine Hälfte"), 3: ("Drittel", "Drittel", ""),
           4: ("Viertel", "Viertel", ""), 5: ("Fünftel", "Fünftel", ""), 6: ("Sechstel", "Sechstel", ""),
           7: ("Siebtel", "Siebtel", ""), 8: ("Achtel", "Achtel", ""), 9: ("Neuntel", "Neuntel", ""),
           10: ("Zehntel", "Zehntel", "")},
    "nl": {2: ("half", "half", "de helft;helft;half;halve"), 3: ("derde", "derde", ""), 4: ("kwart", "kwart", "een vierde"),
           5: ("vijfde", "vijfde", ""), 6: ("zesde", "zesde", ""), 7: ("zevende", "zevende", ""),
           8: ("achtste", "achtste", ""), 9: ("negende", "negende", ""), 10: ("tiende", "tiende", "")},
}
_FRAC_EXTRA = {  # extra denominator names (same value)
    ("es", 3): ["tercera parte", "terceras partes"], ("es", 4): ["cuarta parte", "cuartas partes"],
    ("nl", 4): ["vierde", "vierde"], ("de", 4): ["Viertel", "Viertel"],
}
RATIO_SEP = {"es": ["", "por", "sobre", "entre"], "pt": ["", "por", "sobre"], "it": ["", "su", "fratto"],
             "fr": ["", "sur"], "de": ["", "durch", "zu"], "nl": ["", "op", "gedeeld door"]}


def _find_fractions(text, lang, out, taken):
    for mt in FRAC_CHAR.finditer(text):
        a, b = mt.start(), mt.end()
        if _free(taken, a, b):
            _add(out, taken, {"a": a, "b": b, "kind": "frac", "num": FRAC_SYM[mt["c"]][0], "den": FRAC_SYM[mt["c"]][1],
                              "whole": int(mt["w"]) if mt["w"] else None})
    for mt in FRAC_SLASH.finditer(text):
        a, b = mt.start(), mt.end()
        na, nb = int(mt["a"]), int(mt["b"])
        proper = 2 <= nb <= 10 and 1 <= na < nb
        w = int(mt["w"]) if mt["w"] and proper and not _lead0(mt["w"]) else None
        if mt["w"] and w is None:
            a = mt.start("a")
        if not _free(taken, a, b) or _lead0(mt["a"]) or _lead0(mt["b"]) or nb == 0 or nb > 99 or na > 99:
            continue
        _add(out, taken, {"a": a, "b": b, "kind": "frac", "num": na, "den": nb, "whole": w, "ratio": not proper})


def _frac_words(lang: str, na: int, nb: int, alone_ok: bool = True) -> list[list[str]]:
    sing, plur, alone = _FRAC_SRC[lang][nb]
    out = []
    for a in _amount_words(lang, na, fem=False)[:2]:
        out.append(a + [sing if na == 1 else plur])
    if na == 1 and alone_ok:
        out += [x.split() for x in alone.split(";") if x]
    if (lang, nb) in _FRAC_EXTRA:
        s2, p2 = _FRAC_EXTRA[(lang, nb)]
        for a in _amount_words(lang, na, fem=lang == "es")[:2]:
            out.append(a + (s2 if na == 1 else p2).split())
    if lang == "de" and nb == 4 and na == 3:
        out.append(["dreiviertel"])
    if lang == "nl" and nb == 2 and na == 1 and alone_ok:
        out.append(["een", "half"])
    return out


def _frac_cands(d: dict, lang: str) -> list[str]:
    na, nb, w = d["num"], d["den"], d.get("whole")
    if d.get("ratio"):
        a, b = g(lang, "card", na), g(lang, "card", nb)
        return _uniq([a + ([s] if s else []) + b for s in RATIO_SEP[lang]])
    fr = _frac_words(lang, na, nb)
    if w is None:
        return _uniq(fr)
    cw = g(lang, "card", w)
    # Dutch "vijf en één half" stresses the numeral ("five and ONE half"): after a whole number it is "en een half"
    out = [cw + [AND_WORD[lang]] + f for f in _frac_words(lang, na, nb, alone_ok=False)
           if not (lang == "nl" and f == ["één", "half"])]
    if na == 1 and nb == 2 and w > 0:
        out += _half_variants(lang, w)
        if lang in ("es", "pt", "it", "fr"):
            out += [cw + t for t in ([["y", "medio"], ["y", "media"]] if lang == "es" else [[AND_WORD[lang], x] for x in
                    {"pt": ["meio", "meia"], "it": ["mezzo", "mezza"], "fr": ["demi", "demie"]}[lang]])]
    return _uniq(out)


# ---------------------------------------------------------------- decades

_DEC_CTX = {"es": r"los\s+años", "it": r"anni", "fr": r"années", "nl": r"jaren", "pt": r"anos"}
_DEC_RES: dict = {}
DEC_APOS = re.compile(r"(?<![\w'’])['’](?P<n>[1-9]0)(?![\w])")
DEC_DE = re.compile(r"(?<![\w])(?P<n>(?:1[0-9]|20)[0-9]0|[1-9]0)er(?P<ern>n?)(?![\w])")


def _find_decades(text, lang, out, taken):
    if lang == "de":
        for mt in DEC_DE.finditer(text):
            a, b = mt.start(), mt.end()
            if not _free(taken, a, b):
                continue
            dash = text[b:b + 2]
            if len(dash) == 2 and dash[0] == "-" and dash[1].isalpha() and not mt["ern"]:
                b += 1
            _add(out, taken, {"a": a, "b": b, "kind": "decade", "n": int(mt["n"]), "ern": bool(mt["ern"]),
                              "dash": text[a:b].endswith("-")})
        return
    if lang not in _DEC_RES:
        _DEC_RES[lang] = re.compile(r"(?<![\w])" + _DEC_CTX[lang] + r"\s+['’]?(?P<n>[1-9]0)(?![\w])", re.I)
    for mt in _DEC_RES[lang].finditer(text):
        pre = text[:mt.start()].rstrip()
        if pre and pre[-1].isdigit():
            continue  # "5 años 90 ..." is a count followed by a number
        a, b = mt.start("n"), mt.end("n")
        if text[a - 1:a] in ("'", "’"):
            a -= 1
        if _free(taken, a, b):
            _add(out, taken, {"a": a, "b": b, "kind": "decade", "n": int(mt["n"])})
    for mt in DEC_APOS.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            _add(out, taken, {"a": mt.start(), "b": mt.end(), "kind": "decade", "n": int(mt["n"])})


def _decade_cands(d: dict, lang: str) -> list[str]:
    n = d["n"]
    if lang == "de":
        base = g(lang, "year", n) if n >= 100 else g(lang, "card", n)
        w = _wl(base) + ("ern" if d["ern"] else "er")
        if d["ern"]:
            w = w[:1].upper() + w[1:]
        return [w + (" " if d.get("dash") else "")]
    return _uniq([g(lang, "card", n)])


# ---------------------------------------------------------------- letter + number codes ("5G", "G7", "COVID-19")

_LET_SRC = {
    "es": "a:a b:be c:ce d:de e:e f:efe g:ge h:hache i:i j:jota k:ka l:ele m:eme n:ene o:o p:pe q:cu r:erre s:ese t:te "
          "u:u v:uve;ve w:uve doble;doble uve;doble ve x:equis y:i griega;ye z:zeta",
    "pt": "a:a b:bê c:cê d:dê e:é f:efe g:gê h:agá i:i j:jota k:cá l:ele m:eme n:ene o:ó p:pê q:quê r:erre s:esse t:tê "
          "u:u v:vê w:dáblio;dábliu x:xis y:ípsilon z:zê",
    "fr": "a:a b:bé c:cé d:dé e:e f:effe g:gé h:ache i:i j:ji k:ka l:elle m:emme n:enne o:o p:pé q:ku;qu r:erre s:esse "
          "t:té u:u v:vé w:double vé x:ixe y:i grec;igrec z:zède",
    "it": "a:a b:bi c:ci d:di e:e f:effe g:gi h:acca i:i j:i lunga k:cappa l:elle m:emme n:enne o:o p:pi q:cu r:erre "
          "s:esse t:ti u:u v:vu;vi w:doppia vu x:ics y:ipsilon;i greca z:zeta",
    "nl": "a:a b:bee;be c:cee;se d:dee;de e:e f:ef g:gee;ge h:ha i:i j:jee;je k:ka l:el m:em n:en o:o p:pee;pe q:kuu;ku "
          "r:er s:es t:tee;te u:uu;u v:vee;ve w:wee;we x:iks y:ei z:zet",
    "de": "a:a b:be c:ze;ce d:de e:e f:ef g:ge h:ha i:i j:jot k:ka l:el m:em n:en o:o p:pe q:ku r:er s:es t:te u:u "
          "v:fau w:we x:ix;iks y:ypsilon z:zet",
}
LETTERS = {lang: {kv.split(":")[0]: kv.split(":", 1)[1].split(";") for kv in re.split(r" (?=[a-z]:)", s)}
           for lang, s in _LET_SRC.items()}
EN_LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz", "a;bee;see;dee;e;ef;jee;aitch;i;jay;kay;el;em;en;o;pee;cue;ar;es;tee;u;"
                      "vee;double u;ex;why;zee".split(";")))


def _spell(lang: str, letters: str, k: int = 2) -> list[list[str]]:
    """Letter-name readings of `letters` (each letter: its spoken names, at most k alternatives), as word lists."""
    outs = [[]]
    for c in letters.lower():
        alts = LETTERS[lang].get(c, [c])[:k]
        outs = [o + a.split() for o in outs for a in alts]
        outs = outs[:12]
    return outs


def _code_num_alts(lang: str, s: str) -> list[list[str]]:
    if _lead0(s) or s == "0":
        return [_digits(lang, s)]
    c = g(lang, "card", int(s))
    out = [c] + [v for v in _card_variants(c, lang, int(s), year=False) if v][:2]
    if len(s) >= 2:
        out += _seg_readings(lang, s)
    return out[:7]


ALNUM_W = re.compile(r"(?<![\w/.\-])(?P<w>[A-Za-zÀ-ÿ]{4,15})(?P<h>-?)(?P<n>\d{1,3})(?!\w|[.,]\d)")
ALNUM_L = re.compile(r"(?<![\w/.\-])(?P<l>[A-Z]{1,3})(?P<n>\d{1,3})(?!\w|[.,]\d)")
ALNUM_D = re.compile(r"(?<![\w/.,\-–+])(?P<n>\d{1,2})(?P<l>[DGK])(?!\w)")


def _find_alnum(text, lang, out, taken):
    for mt in ALNUM_D.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            _add(out, taken, {"a": mt.start(), "b": mt.end(), "kind": "alnum", "form": "ND", "n": mt["n"], "l": mt["l"]})
    for mt in ALNUM_W.finditer(text):
        a = mt.start("h") if mt["h"] else mt.start("n")
        if _free(taken, a, mt.end()):
            _add(out, taken, {"a": a, "b": mt.end(), "kind": "alnum", "form": "W", "n": mt["n"], "l": mt["w"]})
    for mt in ALNUM_L.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            _add(out, taken, {"a": mt.start(), "b": mt.end(), "kind": "alnum", "form": "LN", "n": mt["n"], "l": mt["l"]})


def _alnum_cands(d: dict, lang: str) -> list[str]:
    nums = _code_num_alts(lang, d["n"])
    if d["form"] == "W":  # the word stays as written; only the number is read
        return [" " + c for c in _uniq(nums)]
    sp = _spell(lang, d["l"])
    if d["form"] == "LN":
        if "acr_word" in CAND_FIXES and len(d["l"]) >= 3 and re.search(r"[AEIOU]", d["l"]):
            sp = sp + [[d["l"]]]  # said as a word, written as in the caption ("COP ventuno")
        return _uniq([s + n for s in sp for n in nums])
    return _uniq([n + s for s in sp for n in nums])


# ---------------------------------------------------------------- roman numerals (II..XXX, strong context only)

_ROMAN_VAL = {"I": 1, "V": 5, "X": 10}


def _to_roman(n: int) -> str:
    s = ""
    for v, r in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            s += r
            n -= v
    return s


def _roman(s: str) -> int | None:
    if not re.fullmatch(r"[IVX]{1,6}", s):
        return None
    t = 0
    for i, c in enumerate(s):
        v = _ROMAN_VAL[c]
        t += -v if i + 1 < len(s) and _ROMAN_VAL[s[i + 1]] > v else v
    return t if 2 <= t <= 30 and _to_roman(t) == s else None


ROMAN_BEFORE = {"es": "siglos?|capítulo|tomo|volumen|acto", "pt": "séculos?|capítulo|tomo|volume|ato",
                "it": "secolo|capitolo|tomo|volume|atto", "fr": "chapitre|tome|acte|livre|volume",
                "de": "Jahrhundert|Kapitel|Band|Teil|Akt", "nl": "hoofdstuk|deel|boek|akte|eeuw"}
ROMAN_AFTER = {"fr": "siècles?", "it": "secolo", "de": "Jahrhundert", "nl": "eeuw", "pt": "século", "es": "siglo"}
_ROMAN_RES: dict = {}
NAME_ROMAN = re.compile(r"(?<![\w])(?P<name>[A-ZÀ-Ý][a-zà-ÿ]{2,})\s+(?P<r>[IVX]{2,6})(?P<suf>\.?)(?![\w])")


def _roman_res(lang):
    if lang not in _ROMAN_RES:
        suf = {"fr": r"(?:e|ème|er|ᵉ)?", "de": r"\.?"}.get(lang, "")
        after = ROMAN_AFTER[lang]
        _ROMAN_RES[lang] = (
            re.compile(r"(?<![\w])(?P<w>" + ROMAN_BEFORE[lang] + r")\s+(?P<r>[IVX]{1,6})(?P<suf>" + suf + r")(?![\w])",
                       re.I),
            re.compile(r"(?<![\w])(?P<r>[IVX]{1,6})(?P<suf>" + suf + r")(?=\s+(?:" + after + r")(?![\w]))", re.I),
        )
    return _ROMAN_RES[lang]


def _find_roman(text, lang, out, taken):
    cands = []  # (a, b, value, form, suffix)
    pre, post = _roman_res(lang)
    for rx in (pre, post):
        for mt in rx.finditer(text):
            if mt["r"] != mt["r"].upper():
                continue
            v = _roman(mt["r"]) or (mt["r"] in ("V", "X") and _ROMAN_VAL[mt["r"]]) or None
            if v:
                cands.append((mt.start("r"), mt.end("r") + len(mt["suf"]), v, mt["suf"], rx is post))
    for mt in NAME_ROMAN.finditer(text):
        v = _roman(mt["r"])
        if v:
            cands.append((mt.start("r"), mt.end("r") + len(mt["suf"]), v, mt["suf"], False))
    for a, b, v, suf, _century in sorted(cands):
        if suf == ".":  # only German writes the ordinal dot; keep it in the span before a lowercase word / "Jahrhundert"
            if not (lang == "de" and (_century or re.match(r"\s+[a-zäöüß,]", text[b:]) is not None)):
                b -= 1
                suf = ""
        if _free(taken, a, b):
            _add(out, taken, {"a": a, "b": b, "kind": "roman", "n": v, "suf": suf})


def _roman_cands(d: dict, lang: str) -> list[str]:
    n, suf = d["n"], d["suf"]
    c = g(lang, "card", n)
    o = g(lang, "ord", n)
    out: list[list[str]] = []
    if lang == "fr":
        out = [o] if suf else [c, o] + ([["second"], ["seconde"]] if n == 2 else [])
    elif lang == "de":
        ov = [o] + _ord_variants(o, lang, False)
        cap = [[w[:1].upper() + w[1:] for w in x] for x in ov]
        out = ov + [["der"] + cap[0], ["dem"] + cap[1], ["den"] + cap[1], ["des"] + cap[1]] + [c]
    elif lang == "nl":
        out = [o, ["de"] + o, c]
    else:
        out = [c] + [v for v in _card_variants(c, lang, n, year=False) if v][:1] + [o] + _ord_variants(o, lang, False)
    return _uniq(out)


# ---------------------------------------------------------------- "&"

AMP_ACR = re.compile(r"(?<![\w&])(?P<l>[A-Z]{1,3})(?: ?)&(?: ?)(?P<r>[A-Z]{1,3})(?![\w&])")
AMP_WORD = re.compile(r"(?<=[^\W\d_])(?P<s1> ?)&(?P<s2> ?)(?=[^\W\d_])")


def _find_amp(text, lang, out, taken):
    for mt in AMP_ACR.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            _add(out, taken, {"a": mt.start(), "b": mt.end(), "kind": "amp", "form": "acr", "l": mt["l"], "r": mt["r"]})
    for mt in AMP_WORD.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            _add(out, taken, {"a": mt.start(), "b": mt.end(), "kind": "amp", "form": "word"})


def _amp_cands(d: dict, lang: str) -> list[str]:
    if d["form"] == "word":
        return [f" {AND_WORD[lang]} ", " and "]
    l, r = d["l"], d["r"]
    out = [a + [AND_WORD[lang]] + b for a in _spell(lang, l, 2) for b in _spell(lang, r, 2)]
    out.append([EN_LETTERS[c.lower()] for c in l] + ["and"] + [EN_LETTERS[c.lower()] for c in r])  # "Q and A"
    out.append(_spell(lang, l, 1)[0] + ["and"] + _spell(lang, r, 1)[0])
    return _uniq(out)


# ---------------------------------------------------------------- dispatch

def _find_wave2(text, lang, out, taken):
    _find_ranges(text, lang, out, taken)
    _find_versions(text, lang, out, taken)
    _find_fractions(text, lang, out, taken)
    _find_units(text, lang, out, taken)
    _find_decades(text, lang, out, taken)
    _find_alnum(text, lang, out, taken)
    _find_roman(text, lang, out, taken)
    _find_amp(text, lang, out, taken)


# ---------------------------------------------------------------- abbreviations next to numbers
# "1°" as an ordinal (es / it / pt: "il 1° posto"), never "20 °C" / "20° C" / "5° grado" (a degree).
_DEG_ORD = re.compile(r"(?<![\w.,/\-–+:])(?P<n>[1-9]|10)°(?!\d)(?=\s+(?!(?:grad|grau|grados?|gradi|graus?)\b)[a-zà-ÿ])")


def _find_deg_ord(text, lang, out, taken):
    if lang not in ("es", "it", "pt"):
        return
    for mt in _DEG_ORD.finditer(text):
        if any(x < mt.end() and mt.start() < y for x, y in taken):
            continue
        out.append({"a": mt.start(), "b": mt.end(), "kind": "ord", "int": mt["n"], "frac": None, "dsep": None,
                    "fem": False, "mark": "°"})
        taken.append((mt.start(), mt.end()))


# Abbreviations that stay in a label because they sit beside a number ("Nr. 11", "ca. 3000", "5 Mio. EUR", "15,-",
# "3 x 4", "10 a. m."). Each is absorbed into the number's span and its spoken expansion becomes part of the readings.
NR_WORDS = {"de": ["Nummer"], "nl": ["nummer"], "es": ["número"], "pt": ["número"], "it": ["numero"], "fr": ["numéro"]}
CA_WORDS = {"de": ["circa", "zirka"], "nl": ["circa"], "es": ["circa"], "pt": ["circa"], "it": ["circa"], "fr": ["circa"]}
MUL_WORDS = {"de": ["mal"], "nl": ["keer", "maal"], "es": ["por"], "pt": ["por", "vezes"], "it": ["per"], "fr": ["fois"]}
# scale abbreviations: kind -> (singular, plural) spellings per language
SCALE_WORDS = {
    "mio": {"de": (["Million"], ["Millionen"]), "nl": (["miljoen"], ["miljoen", "miljoenen"]),
            "es": (["millón"], ["millones"]), "pt": (["milhão"], ["milhões"]), "it": (["milione"], ["milioni"]),
            "fr": (["million"], ["millions"])},
    "mrd": {"de": (["Milliarde"], ["Milliarden"]), "nl": (["miljard"], ["miljard", "miljarden"]),
            "es": (["mil millones", "millardo"], ["mil millones", "millardos"]),
            "pt": (["bilhão", "mil milhões"], ["bilhões", "mil milhões"]),
            "it": (["miliardo"], ["miliardi"]), "fr": (["milliard"], ["milliards"])},
}
_SCALE_KIND = {"mio": "mio", "mill": "mio", "mln": "mio", "mrd": "mrd", "mld": "mrd", "mdr": "mrd"}
_CCY_SYM = {"€": "€", "eur": "€", "$": "$", "usd": "$", "£": "£", "gbp": "£"}
_PRE_NR = re.compile(r"(?<![\w])(?:[Nn]r\.?|[Nn]ro\.?|[Nn]\.?[º°]|[Nn]úm\.?|[Nn]um\.|No\.|n\.|N\.)\s?$")
_PRE_CA = re.compile(r"(?<![\w])[Cc]a\.\s?$")
_PRE_MUL = re.compile(r"(?<=\d\s)[x×]\s$")
# Legal references written as abbreviations (the old behaviour: ML_CANDS_OFF=law_abbr). Before,
# "§ 35" kept the symbol in the label ("In § eins wird ...": 17 German rows) and "Art. 12" / "Abs. 3" dropped the row
# (abbr_left); now the abbreviation belongs to the number's span and is read as its word. Not a CAND_FIXES name: it
# changes span boundaries, so it must not depend on whether an external transcript is present.
LAW_ABBR = "law_abbr" not in set(filter(None, os.environ.get("ML_CANDS_OFF", "").split(",")))
LAW_WORDS = {
    "para": {"de": ["Paragraf"], "fr": ["paragraphe"], "es": ["párrafo"], "it": ["paragrafo"], "nl": ["paragraaf"],
             "pt": ["parágrafo"]},
    "paras": {"de": ["Paragrafen"], "fr": ["paragraphes"], "es": ["párrafos"], "it": ["paragrafi"], "nl": ["paragrafen"],
              "pt": ["parágrafos"]},
    "art": {"de": ["Artikel"], "fr": ["article"], "es": ["artículo"], "it": ["articolo"], "nl": ["artikel"],
            "pt": ["artigo"]},
    "abs": {"de": ["Absatz"]},
}
_PRE_LAW = re.compile(r"(?<![\w§])(?P<w>§§|§|[Aa]rt\.|Abs\.)\s?$")
_POST_SCALE = re.compile(r"\s?(?P<sc>Mio|Mill|Mln|Mrd|Mld|Mdr)\.?(?:\s?(?P<cu>€|\$|£|EUR|USD|GBP)(?![\w]))?(?![\w])"
                         r"|\s?(?P<mc>M)(?P<mu>€)", re.I)
_POST_DASH = re.compile(r"(?P<w>\s?(?:euro|euros|EUR|€))?,\s?[-–]{1,2}(?P<c>\s?€)?(?![\w\d-])", re.I)
_POST_AP = re.compile(r"\s?(?P<ap>[ap]\.\s?m\.|[ap]m)(?![\w])", re.I)
_ABBR_KINDS_CA = ("int", "dec", "cur", "pct", "unit")


def _attach_abbr(text, lang, out):
    for i, d in enumerate(out):
        k = d["kind"]
        if k not in _ABBR_KINDS_CA or d.get("sign"):
            continue

        def free(a, b):
            return all(o is d or o["b"] <= a or b <= o["a"] for o in out)

        ab = {}
        a, b = d["a"], d["b"]
        pre = text[:a]
        if k == "int":
            mt = _PRE_NR.search(pre) or _PRE_MUL.search(pre)
            if mt and free(mt.start(), a):
                ab["pre"] = "nr" if _PRE_NR.search(pre) else "mul"
                a = mt.start()
        if k == "int" and "pre" not in ab and LAW_ABBR:
            mt = _PRE_LAW.search(pre)
            key = {"§§": "paras", "§": "para", "Abs.": "abs"}.get(mt["w"], "art") if mt else ""
            if mt and lang in LAW_WORDS[key] and free(mt.start(), a):
                ab["pre"], a = key, mt.start()
        if "pre" not in ab and k in _ABBR_KINDS_CA:
            mt = _PRE_CA.search(pre)
            if mt and free(mt.start(), a):
                ab["pre"], a = "ca", mt.start()
        if k in ("int", "dec"):
            mt = _POST_SCALE.match(text, b)
            if mt and free(b, mt.end()):
                if mt["mc"] is not None:
                    ab["scale"], ab["cur"] = "mio", mt["mu"]
                else:
                    ab["scale"], ab["cur"] = _SCALE_KIND[mt["sc"].lower()], (mt["cu"] or "")
                b = mt.end()
            elif k == "int" and not d.get("lead"):
                mt = _POST_DASH.match(text, b)
                if mt and free(b, mt.end()):
                    ab["dash"] = ((mt["w"] or "").strip(), bool(mt["c"]))
                    b = mt.end()
                else:
                    mt = _POST_AP.match(text, b)
                    if mt and 1 <= _int(d["int"]) <= 12 and (lang != "de" or "." in mt["ap"]) and free(b, mt.end()):
                        ab["ap"] = mt["ap"][0].lower()
                        b = mt.end()
        if ab:
            d["a"], d["b"], d["ab"] = a, b, ab


def _scale_agrees(ws: list[str], lang: str) -> bool:
    """The amount before a scale word agrees with it: masculine "un millón" / "um milhão" (no "una", "duas", "uno"),
    German feminine "eine Million" ("eins", "ein", "einen" do not fit). Integers only: a decimal ends in its digit."""
    if not ws:
        return True
    if lang == "de":
        return ws[-1] not in ("eins", "ein", "einen", "einem", "einer")
    if any(_FEM_ONLY.search(w) for w in ws):
        return False
    return not (lang in ("es", "it") and ws[-1] in ("uno", "veintiuno"))


def _wrap_abbr(d: dict, lang: str, base: list[str]) -> list[str]:
    ab = d["ab"]
    pres = {"nr": NR_WORDS, "ca": CA_WORDS, "mul": MUL_WORDS, **LAW_WORDS}.get(ab.get("pre"), {}).get(lang) or [""]
    tails = [""]
    if "scale" in ab:
        one = d["kind"] == "int" and _int(d["int"]) == 1
        words = SCALE_WORDS[ab["scale"]][lang][0 if one else 1]
        cu = _CCY_SYM.get((ab["cur"] or "").lower())
        if cu:
            unit = UNIT[lang][cu][1]
            if lang == "fr":
                tails = [f"{w} d'{unit}" if unit[0] in "aeiouéh" else f"{w} de {unit}" for w in words]
            elif lang in OF:
                tails = [f"{w} {OF[lang]} {unit}" for w in words]
            else:
                tails = [f"{w} {unit}" for w in words]
        else:  # a currency word may follow in the text: with and without the linking "de" / "di"
            tails = list(words) + ([f"{w} {'de' if lang == 'fr' else OF[lang]}" for w in words] if lang in OF else [])
    out = [f"{p} {c}".strip() for p in pres for c in base] if pres != [""] else list(base)
    if ab.get("pre") in LAW_WORDS and lang != "de":  # article / artículo / paragrafo are masculine: no "article une"
        out = [o for o in out if not any(_FEM_ONLY.search(w) for w in o.split())] or out
        first = {"fr": "premier", "es": "primero", "it": "primo", "pt": "primeiro"}.get(lang)
        if first and d["kind"] == "int" and _int(d["int"]) == 1:  # legal style: "l'article premier", "artículo primero"
            out += [f"{p} {first}" for p in pres]
    if "scale" in ab:
        out = [o for o in out if d["kind"] == "dec" or _scale_agrees(o.split(), lang)]
        out = [f"{o} {t}" for o in out for t in tails]
    elif "dash" in ab:
        word, euro_sym = ab["dash"]
        extra = []
        if word or euro_sym:
            extra = [f"{o} {word}" for o in out] if word and word not in ("€",) and word.lower() != "eur" else []
            try:
                cd = dict(d, kind="cur", sym="€")
                extra += _cur_cands(cd, lang)
            except Missing:
                pass
        out = out + extra
    elif "ap" in ab:
        out = out + [f"{o} {w}" for o in out for w in DAYPART[lang][ab["ap"]]]
    return _uniq([o.split() for o in out])


def _cands(d: dict, lang: str) -> list[str]:
    if lang in SLAV:
        import ml_slav

        ml_slav.prefetch(d, lang)
        return ml_slav.cands(d, lang)
    if d.get("ab"):
        return _wrap_abbr(d, lang, _cands(dict(d, ab=None), lang))
    if d.get("sign") and d["kind"] != "skip":
        base = _cands(dict(d, sign=None), lang)
        words = (MINUS_WORDS if d["sign"] in "-−" else PLUS_WORDS)[lang]
        return _uniq([[w] + c.split() for w in words for c in base])
    k = d["kind"]
    if k == "unit":
        return _unit_cands(d, lang)
    if k == "date":
        return _date_cands(d, lang)
    if k == "range":
        return _range_cands(d, lang)
    if k == "version":
        return _version_cands(d, lang)
    if k == "frac":
        return _frac_cands(d, lang)
    if k == "decade":
        return _decade_cands(d, lang)
    if k == "alnum":
        return _alnum_cands(d, lang)
    if k == "roman":
        return _roman_cands(d, lang)
    if k == "amp":
        return _amp_cands(d, lang)
    return _cands_core(d, lang)


# ---------------------------------------------------------------- regional readings (VARIETY, 2026-10-16)
# Readings a speaker of another regional variety says for the same value. They are APPENDED after the readings above
# (spans() calls _regional last), so every existing reading keeps its place; the CTC + the external transcript then
# pick what was said. Value safety: the fr / pt rewrites run on the words of ONE grammar answer at a time (a table
# view during a second _cands pass), so digit-by-digit, pair and phone readings that join two answers ("60 10" ->
# "soixante dix") are never rewritten into one number.
_FR_TEEN = {"douze": "deux", "treize": "trois", "quatorze": "quatre", "quinze": "cinq", "seize": "six",
            "douzième": "deuxième", "treizième": "troisième", "quatorzième": "quatrième", "quinzième": "cinquième",
            "seizième": "sixième"}


def _fr_tens(head: str, ws: list[str], i: int) -> tuple[list[str], int] | None:
    """The 70s / 90s tail starting at ws[i] ("dix ...", "et onze", "onze", a teen) said with `head` ("septante" /
    "nonante"): (words, next index), or None when ws[i] is not such a tail."""
    w = ws[i] if i < len(ws) else ""
    if w == "dix":
        return [head], i + 1
    if w == "dixième":
        return [head[:-1] + "ième"], i + 1
    if w == "et" and i + 1 < len(ws) and ws[i + 1] in ("onze", "onzième"):
        return [head, "et", "un" if ws[i + 1] == "onze" else "unième"], i + 2
    if w in ("onze", "onzième"):  # 91 "quatre vingt onze" -> "nonante et un"
        return [head, "et", "un" if w == "onze" else "unième"], i + 1
    if w in _FR_TEEN:
        return [head, _FR_TEEN[w]], i + 1
    return None


def _fr_regional(ws: list[str], huit: bool, et: bool = True) -> list[str]:
    """One grammar answer in Belgian (huit=False: septante, nonante) or Swiss (huit=True: + huitante) French. et=False:
    the x1 forms without "et" ("huitante-un", as a Swiss speaker in the data says it)."""
    if not et:
        r = _fr_regional(ws, huit)
        return [w for k, w in enumerate(r) if not (w == "et" and k and r[k - 1] in ("septante", "huitante", "nonante"))]
    out, i = [], 0
    while i < len(ws):
        w = ws[i]
        if w == "soixante":
            t = _fr_tens("septante", ws, i + 1)
            if t:
                out += t[0]
                i = t[1]
                continue
        if w == "quatre" and i + 1 < len(ws) and ws[i + 1] in ("vingt", "vingts", "vingtième"):
            if ws[i + 1] == "vingt":
                t = _fr_tens("nonante", ws, i + 2)
                if t:
                    out += t[0]
                    i = t[1]
                    continue
            if huit:
                nxt = ws[i + 2] if i + 2 < len(ws) else ""
                if ws[i + 1] == "vingtième":
                    out.append("huitantième")
                    i += 2
                elif ws[i + 1] == "vingt" and nxt in ("un", "une", "unième"):
                    out += ["huitante", "et", nxt]
                    i += 3
                else:
                    out.append("huitante")
                    i += 2
                continue
        out.append(w)
        i += 1
    return out


def _pt_regional(ws: list[str]) -> list[str]:
    """One grammar answer in European Portuguese long-scale words: 10^12 "bilião" (Brazilian "trilhão"), 10^9 "mil
    milhões" merged with a following millions group ("um bilhão e duzentos milhões" -> "mil e duzentos milhões", "dez
    bilhões duzentos e quinze milhões" -> "dez mil duzentos e quinze milhões"); the joining "e" stays as written."""
    ws = [{"trilhão": "bilião", "trilhões": "biliões"}.get(w, w) for w in ws]
    out, i = [], 0
    while i < len(ws):
        w = ws[i]
        if w not in ("bilhão", "bilhões"):
            out.append(w)
            i += 1
            continue
        if w == "bilhão" and out and out[-1] == "um":
            out.pop()  # "um bilhão" -> "mil ..."
        out.append("mil")
        j = i + 1
        k = next((x for x in range(j, len(ws)) if ws[x] in ("milhão", "milhões")), None)
        group = ws[j:k] if k is not None else None
        if group is not None and group and not any(x in ("mil", "bilião", "biliões") for x in group):
            if group == ["e", "um"] or group == ["um"]:
                group = group[:-1] + ["um"]
            out += group + ["milhões"]
            i = k + 1
        else:
            out.append("milhões")
            i = j
    return out


class _TableView:
    """The grammar table seen through a per-answer rewrite (a second _cands pass)."""

    def __init__(self, base: dict, fn):
        self.base, self.fn = base, fn

    def get(self, k):
        w = self.base.get(k)
        return self.fn(list(w)) if w else w


def _with_view(d: dict, lang: str, fn) -> list[str]:
    global _TABLE
    saved = table()
    _TABLE = _TableView(saved, fn)
    try:
        return _cands(d, lang)
    except (Missing, ValueError, IndexError):
        return []
    finally:
        _TABLE = saved


_AT_MONTHS = {"Januar": "Jänner", "Februar": "Feber"}


def _regional(d: dict, lang: str, base: list[str], names=None) -> list[str]:
    on = (CAND_FIXES & VARIETY) if names is None else set(names)
    extra = []
    if lang == "fr" and "fr_belch" in on:
        for huit in (False, True):
            for et in (True, False):
                extra += _with_view(d, lang, lambda ws, h=huit, e=et: _fr_regional(ws, h, e))
    if lang == "pt" and "pt_eu" in on:
        extra += _with_view(d, lang, _pt_regional)
    if lang == "de" and "de_at" in on and d["kind"] == "date":
        extra += [" ".join(_AT_MONTHS.get(w, w) for w in c.split(" ")) for c in base]
    if d["kind"] == "clock" and "clock_rel" in on:
        try:
            rel = _clock_rel(d["h"], d["m"], lang)
        except Missing:
            rel = []
        ap = d.get("ap")
        extra += [c + (" " + w if w else "") for c in rel for w in ([""] + DAYPART[lang][ap] if ap else [""])]
    extra += _fill_readings(d, lang, base, on & FILL_ALL)
    seen = set(base)
    out = list(base)
    for c in extra:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


_ALT_DIGIT = {"de": ("zwei", "zwo"), "pt": ("seis", "meia")}  # phone_alt: one spoken digit, its colloquial word


def _fill_readings(d: dict, lang: str, base: list[str], on: set) -> list[str]:
    """The FILL_ALL readings named in `on` for one span (d = its _find dict, base = its readings)."""
    out = []
    k = d["kind"]
    if k == "cur" and "cur_bare" in on and d.get("frac") and len(d["frac"]) == 2 and int(d["frac"]) >= 10:
        n = _int(d["int"])
        if n:
            out.append(" ".join(g(lang, "card", n) + g(lang, "card", int(d["frac"]))))
    if k == "dec" and "dec_con" in on and lang == "es":
        out += [re.sub(r"\bcoma\b", "con", c) for c in base if re.search(r"\bcoma\b", c)]
    if k == "pct" and "pct_alt" in on:
        if lang == "nl":
            out += [re.sub(r"\bprocent\b", "percent", c) for c in base if re.search(r"\bprocent\b", c)]
        elif lang == "es":
            out += [re.sub(r"\bpor ciento\b", "por cien", c) for c in base if re.search(r"\bpor ciento\b", c)]
    if k == "phone" and "phone_alt" in on and lang in _ALT_DIGIT:
        a, b = _ALT_DIGIT[lang]
        alone = rf"(?<!\S)(?<!\be ){a}(?!\S)"  # the digit said alone, never inside "noventa e seis"
        out += [re.sub(alone, b, c) for c in base if re.search(alone, c)]
    if k == "range" and d.get("part") == "r" and "range_a" in on and lang == "nl":
        out += [" à" + c[len(" tot"):] for c in base if c.startswith(" tot ") and not c.startswith(" tot en met ")]
    if k == "clock" and "clock_noon" in on and lang == "es" and d.get("h") == 12 and not d.get("ap"):
        out += [c + " del mediodía" for c in base if c.split()[0] == "doce" and "horas" not in c.split()]
    return out


def spans_regional(text: str, lang: str, names=VARIETY_ALL) -> list[Span]:
    """spans() with the regional readings `names` appended to every span's readings (whatever ML_CANDS_ON says): for
    deciding them from an external transcript (ml_variety.regional_relabel). Same spans, same first readings."""
    out = spans(text, lang)
    if lang not in LANGS:
        return out
    ds = _find(text, lang)
    for sp, d in zip(out, ds):
        if sp.cands:
            try:
                sp.cands = _regional(d, lang, sp.cands, names)
            except Missing:
                pass
    return out


# relative clock readings: (the hour said is the next hour for "to" / "half" forms; 12-hour clock as spoken)
_HOUR_FEM = {"es": {1: "una"}, "pt": {1: "uma", 2: "duas"}, "it": {1: "una"}, "fr": {1: "une"}}


def _hw(lang: str, h: int) -> str:
    fem = _HOUR_FEM.get(lang, {})
    return fem[h] if h in fem else " ".join(g(lang, "card", h))


def _mw(lang: str, m: int) -> str:
    return " ".join(g(lang, "card", m))


def _clock_rel(h: int, m: int, lang: str) -> list[str]:
    h12 = h % 12 or 12
    n12 = h12 % 12 + 1
    H, N = _hw(lang, h12), _hw(lang, n12)
    out = []
    if lang == "de":
        if m == 15:
            out += [f"Viertel nach {H}", f"viertel {N}"]
        elif m == 30:
            out.append(f"halb {N}")
        elif m == 45:
            out += [f"Viertel vor {N}", f"dreiviertel {N}"]
        elif 1 <= m <= 20:
            out.append(f"{_mw(lang, m)} nach {H}")
        elif m == 25:
            out.append(f"fünf vor halb {N}")
        elif m == 35:
            out.append(f"fünf nach halb {N}")
        elif m >= 40:
            out.append(f"{_mw(lang, 60 - m)} vor {N}")
        if m == 20:
            out.append(f"zehn vor halb {N}")
        if m == 40:
            out.append(f"zehn nach halb {N}")
    elif lang == "nl":
        for over in ("over", "na"):  # Flemish "na"
            if m == 15:
                out.append(f"kwart {over} {H}")
            elif 1 <= m <= 14:
                out.append(f"{_mw(lang, m)} {over} {H}")
            elif 31 <= m <= 44:
                out.append(f"{_mw(lang, m - 30)} {over} half {N}")
            if m == 20:
                out.append(f"twintig {over} {H}")
        if m == 30:
            out.append(f"half {N}")
        elif 16 <= m <= 29:
            out.append(f"{_mw(lang, 30 - m)} voor half {N}")
        elif m == 45:
            out.append(f"kwart voor {N}")
        elif m >= 46:
            out.append(f"{_mw(lang, 60 - m)} voor {N}")
    elif lang == "fr":
        def fr_h(x24: int) -> list[str]:
            if x24 == 12:
                return ["midi"]
            if x24 in (0, 24):
                return ["minuit"]
            x = x24 % 12 or 12
            hs = [x] + ([x24] if x24 > 12 else [])
            return [f"{_hw(lang, y)} {'heure' if y == 1 else 'heures'}" for y in hs]
        if h in (0, 12):
            tail = {0: "", 15: " et quart", 30: " et demi"}.get(m, f" {_mw(lang, m)}")
            out += [w + tail for w in fr_h(h)]
        if m >= 31:
            for w in fr_h(h + 1):
                out += [f"{w} moins le quart", f"{w} moins quart"] if m == 45 else [f"{w} moins {_mw(lang, 60 - m)}"]
    elif lang == "es":
        if m >= 31:
            rest = _mw(lang, 60 - m)
            art = "la" if n12 == 1 else "las"
            if m == 45:
                out += [f"{N} menos cuarto", f"{N} menos quince", f"un cuarto para {art} {N}", f"cuarto para {art} {N}",
                        f"quince para {art} {N}"]
            else:
                out += [f"{N} menos {rest}", f"{rest} para {art} {N}"]
    elif lang == "it":
        if h in (0, 12):
            w = "mezzogiorno" if h == 12 else "mezzanotte"
            out.append(w + {0: "", 15: " e un quarto", 30: " e mezza", 45: " e tre quarti"}.get(m, f" e {_mw(lang, m)}"))
        if m == 45:
            out += [f"{N} meno un quarto", f"{N} meno quarto", f"{H} e tre quarti"]
        elif m >= 31:
            out.append(f"{N} meno {_mw(lang, 60 - m)}")
    elif lang == "pt":
        if h in (0, 12):
            w = "meio-dia" if h == 12 else "meia-noite"
            out.append(w + {0: "", 15: " e um quarto", 30: " e meia"}.get(m, f" e {_mw(lang, m)}"))
        if m >= 31:
            rest = _mw(lang, 60 - m)
            art = "a" if n12 == 1 else "as"
            if m == 45:
                out += [f"{N} menos um quarto", f"{N} menos quinze", f"um quarto para {art} {N}", f"quinze para {art} {N}"]
            else:
                out += [f"{N} menos {rest}", f"{rest} para {art} {N}"]
    return _uniq([o.split() for o in out])
