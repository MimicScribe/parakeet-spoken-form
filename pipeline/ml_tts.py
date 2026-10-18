"""Synthetic speech for adversarial spoken-number training data: de es fr it nl pt (R3.x, "tts1").

Real speech is thin for many number shapes (phone groupings, prices with cents, clock times, dates, ranges, units,
signs, fractions, ordinals, codes, roman numerals, versions, years, decades, large figures, "&"). This module plans
carrier sentences with a WRITTEN number in a slot, picks the SPOKEN reading from the ml_cands candidates of that
written form (so every common reading appears), voices the spoken sentence, and keeps a clip only when a per-language
CTC reader hears the reading it was given.

    plan(lang, n, seed)       pure Python (no GPU, no network): rows with written form, label, spans, num_chars, voice
    needed_keys(lang, n, seed) grammar-table keys the broad plan needs that ml_grammar_table.json lacks
    check_greedy(row, greedy) pure round-trip rule (CER and every reading fits the greedy text)
    tts_lang(lang, n, seed)   Modal function (L4): synthesise, verify, align, write /vol/ml/tts1/<lang>/

    python ml_tts.py plan es 40 1          # print sample rows
    python ml_tts.py report 600 1          # per language x class yield from the CURRENT grammar table
    python ml_tts.py needed es 3000 1      # write tts_needed_es.json / tts_texts_es.json for the grammar dumper
    modal run ml_tts.py --langs es,fr --n 600 --seed 1

The label is exactly the spoken words (no digits or symbols). Voices and their licences are in VOICES below.
"""

import collections
import hashlib
import math
import json
import os
import random
import re
import sys

import grammar_rules as G
import ml_cands as M
import ml_p3_core as C
import ml_tts_templates as T

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = M.LANGS
TAG = "tts1"
VOL = "/vol"
OUT = f"{VOL}/ml/{TAG}"

CER_MAX = 0.15
# planned-row fields a verified row keeps when present (the reading-variety fill: the rows it replaces, the convention
# of each number, the voice locale); rows without them are written exactly as before
PASS_THROUGH = ("supersedes", "convs", "locale", "ckey", "register")
SPEED_LO, SPEED_HI = 0.9, 1.3
WORDS_MIN, WORDS_MAX = 8, 20
FORBIDDEN = re.compile(r"[0-9$€£¥%#/+°²³½¼¾ºª‰&]")  # the build audit's list

CLASSES = ["count", "large", "dec", "pct", "permille", "unit", "temp", "frac", "range_from", "range_between",
           "price", "price_big", "clock24", "clock12", "date_num", "date_txt", "phone", "code", "roman", "version",
           "year", "decade", "ord", "amp"]
GROUP = {"count": "QTY", "large": "QTY", "dec": "QTY", "pct": "QTY", "permille": "QTY", "unit": "QTY", "temp": "TEMP",
         "frac": "FRAC", "range_from": "RANGE_FROM", "range_between": "RANGE_BETWEEN", "price": "MONEY",
         "price_big": "MONEY", "clock24": "TIME", "clock12": "TIME", "date_num": "DATE_NUM", "date_txt": "DATE_TXT",
         "phone": "PHONE", "code": "CODE", "roman": "ROMAN", "version": "VERSION", "year": "YEAR", "decade": "DECADE",
         "ord": "ORD", "amp": "AMP"}
# at least one span of these ml_cands kinds must come out of the written form (else the shape was not recognised)
EXPECT = {"count": {"int"}, "large": {"int", "dec"}, "dec": {"dec"}, "pct": {"pct"}, "permille": {"unit"}, "unit": {"unit"},
          "temp": {"unit"}, "frac": {"frac"}, "range_from": {"range"}, "range_between": {"range"}, "price": {"cur"},
          "price_big": {"cur"}, "clock24": {"clock"}, "clock12": {"clock", "int"}, "date_num": {"date"},
          "date_txt": {"int", "ord"}, "phone": {"phone"}, "code": {"alnum", "int"}, "roman": {"roman"},
          "version": {"version"}, "year": {"int", "year"}, "decade": {"decade"}, "ord": {"ord"}, "amp": {"amp"}}

# --------------------------------------------------------------------------- voices
# engine "kokoro": Kokoro-82M (Apache-2.0 weights; the model card says it was trained only on permissive or
#   non-copyrighted audio). "piper": rhasspy/piper-voices ONNX files; each licence below is copied from that voice's
#   MODEL_CARD (checked 2026-10-15). Voices whose card says "See URL", AGPL or Share-Alike, or whose data is
#   synthetic of unknown origin, are NOT used: es_ES-mls (low only), it_IT-paola/riccardo/serena, fr_FR-tom (AGPLv3),
#   fr_FR-upmc (CC BY-SA), es_AR-daniela (CC BY-SA), fr_FR-siwis (same speaker as Kokoro ff_siwis).
# spk: (lo, hi) speaker-id range of a multi-speaker Piper voice; dev voices are never trained on.
def _k(vid, lic, dev=False, w=1.0):
    return {"id": vid, "engine": "kokoro", "license": lic, "dev": dev, "w": w}


def _p(path, lic, spk=None, dev=False, w=1.0):
    return {"id": path.rsplit("/", 1)[1], "engine": "piper", "path": path, "license": lic, "spk": spk, "dev": dev, "w": w}


KOKORO = "Apache-2.0 (Kokoro-82M weights)"
CC0, CCBY4, CCBY3, APACHE = "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "Apache-2.0"
VOICES = {
    "es": [_k("ef_dora", KOKORO), _k("em_alex", KOKORO), _k("em_santa", KOKORO, dev=True),
           _p("es/es_ES/davefx/medium/es_ES-davefx-medium", CC0), _p("es/es_MX/claude/high/es_MX-claude-high", APACHE),
           _p("es/es_ES/sharvard/medium/es_ES-sharvard-medium", CCBY3, spk=(0, 1))],
    "fr": [_k("ff_siwis", KOKORO + "; voice trained on SIWIS (CC-BY-4.0)", w=1.5),
           _p("fr/fr_FR/mls/medium/fr_FR-mls-medium", CCBY4, spk=(0, 99), w=2.0),
           _p("fr/fr_FR/mls/medium/fr_FR-mls-medium", CCBY4, spk=(100, 124), dev=True),
           _p("fr/fr_FR/gilles/low/fr_FR-gilles-low", CC0, w=0.5)],
    # the only licence-clean Italian voices, plus Kokoro style blends of them with Spanish / Portuguese voices (same
    # Apache-2.0 weights; the Italian G2P is kept, only the timbre/style vector is mixed) for more speakers
    "it": [_k("if_sara", KOKORO, w=1.0), _k("im_nicola", KOKORO, w=1.0),
           _k("blend:i:if_sara*0.6+ef_dora*0.4", KOKORO, w=0.7),
           _k("blend:i:if_sara*0.6+pf_dora*0.4", KOKORO, w=0.7),
           _k("blend:i:im_nicola*0.6+em_alex*0.4", KOKORO, w=0.7),
           _k("blend:i:im_nicola*0.6+pm_alex*0.4", KOKORO, dev=True)],
    "pt": [_k("pf_dora", KOKORO), _k("pm_alex", KOKORO), _k("pm_santa", KOKORO),
           _p("pt/pt_BR/faber/medium/pt_BR-faber-medium", CC0), _p("pt/pt_BR/cadu/medium/pt_BR-cadu-medium", CC0),
           _p("pt/pt_BR/jeff/medium/pt_BR-jeff-medium", CC0, dev=True),
           _p("pt/pt_PT/tugão/medium/pt_PT-tugão-medium", CC0)],
    "de": [_p("de/de_DE/thorsten/medium/de_DE-thorsten-medium", CC0), _p("de/de_DE/thorsten/high/de_DE-thorsten-high", CC0),
           _p("de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium", CC0, spk=(0, 7)),
           _p("de/de_DE/mls/medium/de_DE-mls-medium", CCBY4, spk=(0, 199), w=3.0),
           _p("de/de_DE/mls/medium/de_DE-mls-medium", CCBY4, spk=(200, 235), dev=True)],
    "nl": [_p("nl/nl_NL/pim/medium/nl_NL-pim-medium", CC0), _p("nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium", CC0),
           _p("nl/nl_NL/alex/medium/nl_NL-alex-medium", CC0), _p("nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium", CC0),
           _p("nl/nl_BE/rdh/medium/nl_BE-rdh-medium", CC0),
           _p("nl/nl_NL/mls/medium/nl_NL-mls-medium", CCBY4, spk=(0, 43), w=3.0),
           _p("nl/nl_NL/mls/medium/nl_NL-mls-medium", CCBY4, spk=(44, 51), dev=True)],
}
DEV_FRAC = 0.1  # share of rows given to the dev voices of a language


# --------------------------------------------------------------------------- table pools (draw values the table knows)

_POOLS: dict = {}


def pools(lang: str) -> dict:
    if lang not in _POOLS:
        p = {"card": [], "year": [], "clock": [], "cur": [], "pct": [], "dec": [], "ord": []}
        for k in M.table():
            l, kind, n, f, s, h, m = k.split("|")
            if l != lang:
                continue
            if kind == "card":
                p["card"].append(int(n))
            elif kind == "year":
                p["year"].append(int(n))
            elif kind == "clock":
                p["clock"].append((int(h), int(m)))
            elif kind in ("cur", "pct", "dec"):
                p[kind].append((int(n), f))
            elif kind == "ord":
                p["ord"].append(int(n))
        for v in p.values():
            v.sort()
        _POOLS[lang] = p
    return _POOLS[lang]


def _card(rng, P, lo, hi, broad):
    xs = [n for n in P["card"] if lo <= n <= hi]
    return rng.choice(xs) if (xs and not broad) else rng.randint(lo, hi)


def _fmt_int(n: int, lang: str, rng, group_p: float = 0.3) -> str:
    s = str(n)
    if n >= 10000 and rng.random() < group_p:
        sep = " " if lang == "fr" else "."
        s = f"{n:,}".replace(",", sep)
    return s


def _dec(rng, P, lo, hi, broad):
    xs = [(n, f) for n, f in P["dec"] if lo <= n <= hi and f and len(f) <= 2 and not f.startswith("0") and not f.endswith("0")]
    if xs and not broad:
        n, f = rng.choice(xs)
    else:
        n, f = rng.randint(lo, hi), rng.choice(["5", "25", "75", "2", "8", "3", "4", "6", "75", "9", "15", "35"])
    return f"{n},{f}"


# --------------------------------------------------------------------------- slot generators: (lang, rng, P, ci, broad, gender) -> written slot text

COUNT_QUOTA: dict = {}  # count value -> rows wanted (a top-up pass aimed at thin 0-100 values / value spread); empty = usual


def g_count(lang, rng, P, ci, broad, gender):
    if COUNT_QUOTA:  # a targeted value: the gender rule below is left to reading_ok (feminine readings are rejected)
        vs = sorted(COUNT_QUOTA)
        return f"{_fmt_int(rng.choices(vs, [COUNT_QUOTA[v] for v in vs])[0], lang, rng)} {rng.choice(T.NOUNS[lang])}"
    while True:
        n = _card(rng, P, 3, 999 if rng.random() < 0.55 else 99999, broad)
        if n % 10 in (1, 2) and n % 100 not in (11, 12):
            continue  # un/uno/una, dois/duas, eins/eine agree with the noun: keep to numbers without that choice
        return f"{_fmt_int(n, lang, rng)} {rng.choice(T.NOUNS[lang])}"


def g_large(lang, rng, P, ci, broad, gender):
    sc, prep, big = T.SCALE[lang]
    noun = rng.choice(T.SCALE_NOUN[lang])
    if rng.random() < 0.3:
        x = _dec(rng, P, 2, 99, broad)
    else:
        while True:
            x = _card(rng, P, 2, 999 if rng.random() < 0.7 else 9999, broad)
            if x % 10 != 1 or x % 100 == 11:  # "ochenta y un millones": the un/uno choice is the grammar's, keep clear of it
                break
        x = str(x)
    tail = f" {prep}{noun}" if prep.endswith("'") and noun[0] in "aeiouéh" else (f" {prep} {noun}" if prep else f" {noun}")
    if prep.endswith("'") and noun[0] not in "aeiouéh":
        tail = f" de {noun}"
    return f"{x} {sc}{tail}"


def g_dec(lang, rng, P, ci, broad, gender):
    return f"{_dec(rng, P, 0, 99, broad)} {rng.choice(T.DEC_NOUNS[lang])}"


PCT_FORMS = {"es": ["{a} %", "{a}%", "{a} por ciento"], "pt": ["{a}%", "{a} %", "{a} por cento"],
             "fr": ["{a} %", "{a} pour cent", "{a}%"], "it": ["{a}%", "{a} per cento", "{a} %"],
             "de": ["{a} Prozent", "{a} %", "{a}%"], "nl": ["{a}%", "{a} procent", "{a} %"]}


def g_pct(lang, rng, P, ci, broad, gender):
    xs = [(n, f) for n, f in P["pct"] if n <= 200 and len(f) <= 2]
    n, f = rng.choice(xs) if (xs and not broad) else (rng.randint(1, 95), rng.choice(["", "", "5", "2", "75", "4"]))
    a = f"{n},{f}" if f else str(n)
    return rng.choice(PCT_FORMS[lang]).format(a=a)


def g_permille(lang, rng, P, ci, broad, gender):
    a = _dec(rng, P, 1, 20, broad) if rng.random() < 0.4 else str(_card(rng, P, 2, 30, broad))
    return f"{a} ‰"


UNIT_USE = ["km", "kg", "m²", "km/h", "ml", "l", "kWh", "GB", "min", "cm", "g", "t", "m³", "km²", "MW", "mm"]


def g_unit(lang, rng, P, ci, broad, gender):
    u = rng.choice(UNIT_USE)
    a = _dec(rng, P, 1, 99, broad) if rng.random() < 0.3 else str(_card(rng, P, 2, 999 if rng.random() < 0.7 else 9999, broad))
    sp = " " if rng.random() < 0.15 else " "
    return f"{a}{sp}{u}"


def g_temp(lang, rng, P, ci, broad, gender):
    n = _card(rng, P, 1, 40, broad)
    r = rng.random()
    sign = "-" if r < 0.6 else ("−" if r < 0.7 else ("+" if r < 0.85 else ""))
    return f"{sign}{n}{rng.choice([' ', ' ', ''])}°C"


def g_frac(lang, rng, P, ci, broad, gender):
    noun = rng.choice(T.FRAC_NOUN[lang])
    r = rng.random()
    if lang in ("de", "nl") and r < 0.4:
        return f"{rng.randint(2, 9)} 1/2 {rng.choice(T.FRAC_NOUN_PL[lang])}"
    if r < 0.45:
        # never "½" before a unit: "½ de litro" has no natural reading ("medio litro" takes no "de", "un medio de
        # litro" / "een half liter" read the half as a noun); "¼ de litro" / "¾ Liter" are fine
        sym = rng.choice(["¼", "¾"])
        return f"{sym} {noun}" if lang in ("de", "nl") else f"{sym} {_of(lang, noun)}"
    while True:
        nb = rng.choice([3, 4, 5, 8, 10])  # not /2: "1/2 hora" reads "mitad de hora"
        na = rng.randint(1, nb - 1)
        if math.gcd(na, nb) == 1:
            break
    return f"{na}/{nb} {noun}" if lang in ("de", "nl") else f"{na}/{nb} {_of(lang, noun)}"


def _of(lang, noun):
    if lang in ("es", "pt", "fr"):
        return ("d'" + noun) if lang == "fr" and noun[0] in "aeiouh" else "de " + noun
    return "di " + noun  # it


def _pair(rng, P, broad, lo=3, hi=90, nounish=False, years=False):  # nounish kept for call sites
    for _ in range(200):
        if years:
            l = rng.randint(1995, 2022)
            return l, l + rng.randint(1, 6)
        l = _card(rng, P, lo, hi, broad)
        r = l + rng.choice([1, 2, 2, 3, 4, 5, 10])
        if l % 10 in (1, 2) or r % 10 in (1, 2):
            continue  # un/uno/una, dois/duas agree with a noun or unit that is not in the slot
        if r <= 99 or not nounish:
            return l, r
    return lo, lo + 2


def g_range_from(lang, rng, P, ci, broad, gender):
    if rng.random() < 0.2:
        l, r = _pair(rng, P, broad, years=True)
        return f"{l}{rng.choice(['-', '–'])}{r}"
    l, r = _pair(rng, P, broad)
    u = rng.choice(["h", "km", "kg", "min", "%", "h"]) if rng.random() < 0.7 else ""
    sep = rng.choice(["-", "–", " – "])
    if u == "%":
        return f"{l}{sep}{r} %"
    return f"{l}{sep}{r}" + (f" {u}" if u else "")


def g_range_between(lang, rng, P, ci, broad, gender):
    l, r = _pair(rng, P, broad, nounish=True)
    return f"{l}{rng.choice(['-', '–'])}{r} {rng.choice(T.NOUNS[lang])}"


def _money_forms(lang):
    return {"es": ["{a} €", "{a} €", "$ {a}", "US$ {a}", "{a} EUR"], "pt": ["R$ {a}", "R$ {a}", "{a} €", "US$ {a}", "€ {a}"],
            "fr": ["{a} €", "{a} €", "{a} $", "{a} EUR"], "it": ["{a} €", "€ {a}", "{a} €", "$ {a}"],
            "de": ["{a} €", "{a} €", "{a} EUR", "€ {a}", "{a} $"], "nl": ["€ {a}", "{a} €", "€ {a}", "{a} EUR"]}[lang]


def g_price(lang, rng, P, ci, broad, gender):
    xs = [(n, f) for n, f in P["cur"] if n <= 5000 and len(f) <= 2]
    if xs and not broad and rng.random() < 0.75:
        n, f = rng.choice(xs)
    else:
        n = rng.choice([_card(rng, P, 2, 99, broad), rng.randint(2, 99), rng.randint(100, 900)])
        f = rng.choice(["50", "90", "99", "95", "75", "10", "25", "45", "05", "", ""])
    a = f"{n},{f}" if f else str(n)
    return rng.choice(_money_forms(lang)).format(a=a)


def g_price_big(lang, rng, P, ci, broad, gender):
    n = _card(rng, P, 1000, 2000000, broad)
    return rng.choice(_money_forms(lang)).format(a=_fmt_int(n, lang, rng, 0.8))


def _hm(rng, P, broad, hmax=23, hmin=0):
    xs = [x for x in P["clock"] if hmin <= x[0] <= hmax]
    if xs and not broad:
        return rng.choice(xs)
    return rng.randint(hmin, hmax), rng.choice([0, 0, 5, 10, 15, 15, 20, 30, 30, 40, 45, 45, 50, 55, rng.randint(1, 59)])


def g_clock24(lang, rng, P, ci, broad, gender):
    for _ in range(100):
        h, m = _hm(rng, P, broad)
        if lang in ("es", "pt", "it") and h == 1:
            continue  # "a la una" vs "a las": the clause fixes the plural
        break
    hh = f"{h:02d}" if rng.random() < 0.3 and h < 10 else str(h)
    forms = {"es": ["{h}:{m}", "{h}:{m} h", "{h}:{m} horas"], "pt": ["{h}:{m}", "{h}h{m}", "{h}:{m} h", "{h}h{m}"],
             "it": ["{h}:{m}", "{h}:{m} ore", "{h}:{m}"], "fr": ["{h}h{m}", "{h}h{m}", "{h}:{m}"],
             "de": ["{h}:{m} Uhr", "{h}.{m} Uhr", "{h}:{m}", "{h}:{m} Uhr"], "nl": ["{h}:{m} uur", "{h}.{m} uur", "{h}:{m}", "{h}:{m} u"]}[lang]
    return rng.choice(forms).format(h=hh, m=f"{m:02d}")


def g_clock12(lang, rng, P, ci, broad, gender):
    h, m = _hm(rng, P, broad, hmax=12, hmin=2)
    a = rng.choice("ap")
    if lang in ("de", "nl"):
        forms = ["{h}:{m} {a}.m.", "{h} {a}.m."] if m == 0 else ["{h}:{m} {a}.m."]
    else:
        forms = ["{h}:{m} {a}. m.", "{h}:{m} {a}m", "{h} {a}. m."] if m == 0 else ["{h}:{m} {a}. m.", "{h}:{m} {a}m"]
    return rng.choice(forms).format(h=h, m=f"{m:02d}", a=a)


def g_date_num(lang, rng, P, ci, broad, gender):
    d, mth, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1996, 2030)
    if lang in ("de", "nl") and rng.random() < 0.4:
        return f"{d:02d}.{mth:02d}."
    sep = "." if lang in ("de", "nl") else rng.choice(["/", "/", "-"])
    ys = str(y) if rng.random() < 0.85 else f"{y % 100:02d}"
    if len(ys) == 2 and sep != ".":
        ys = str(y)
    return f"{d:02d}{sep}{mth:02d}{sep}{ys}" if rng.random() < 0.6 else f"{d}{sep}{mth}{sep}{ys}"


def g_date_txt(lang, rng, P, ci, broad, gender):
    d, mth, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1996, 2030)
    names = M.MONTHS[lang].split()
    mn = [w for w in names if w != "setiembre"][mth - 1] if lang == "es" else names[mth - 1]
    if lang == "de":
        mn = mn.capitalize()
    first = rng.random() < 0.25
    if lang in ("es", "pt"):
        return f"{'1º' if first else d} de {mn} de {y}"
    if lang == "fr":
        return f"{'1er' if first else d} {mn} {y}"
    if lang == "it":
        return f"{'1º' if first else d} {mn} {y}"
    if lang == "de":
        return f"{1 if first else d}. {mn} {y}"
    return f"{d} {mn} {y}"


PHONE_PATTERNS = {
    "es": [(3, 3, 3), (2, 3, 2, 2), (3, 2, 2, 2), "+34 3 3 3"], "pt": [(2, 5, 4), (3, 3, 3), (2, 4, 4), "+351 3 3 3"],
    "fr": [(2, 2, 2, 2, 2), (2, 2, 2, 2, 2), "+33 1 2 2 2 2"], "it": [(3, 3, 4), (3, 3, 3), "+39 3 3 4", (2, 4, 4)],
    "de": [(4, 3, 2, 2), (3, 3, 3, 2), "+49 3 4 4", (4, 3, 4)], "nl": [(2, 2, 2, 2, 2), (3, 3, 2, 2), "+31 1 4 4", (2, 3, 2, 2)],
}


def g_phone(lang, rng, P, ci, broad, gender):
    for _ in range(100):
        pat = rng.choice(PHONE_PATTERNS[lang])
        if isinstance(pat, str):
            head, *sizes = pat.split()
            groups = [head] + ["".join(str(rng.randint(0, 9)) for _ in range(int(s))) for s in sizes]
            groups[1] = str(rng.randint(1, 9)) + groups[1][1:]
            sep = " "
            out = " ".join(groups)
        else:
            sep = rng.choice([" ", " ", "-"])
            lead = {"fr": "0", "de": "0", "nl": "0", "it": rng.choice(["3", "0"]), "es": rng.choice(["6", "9", "7"]),
                    "pt": rng.choice(["9", "2", "1"])}[lang]
            groups = []
            for i, s in enumerate(pat):
                g = "".join(str(rng.randint(0, 9)) for _ in range(s))
                if i == 0:
                    g = lead + g[1:]
                groups.append(g)
            out = sep.join(groups)
        digits = re.sub(r"\D", "", out)
        if 7 <= len(digits) <= 15:
            return out
    return None


def g_code(lang, rng, P, ci, broad, gender):
    r = rng.random()
    return rng.choice(T.CODE_LETTER_NUM if r < 0.45 else T.CODE_NUM_LETTER if r < 0.65 else T.CODE_WORDS)


def _roman(n):
    return M._to_roman(n)


def g_roman(lang, rng, P, ci, broad, gender):
    r = rng.random()
    n = rng.randint(2, 30)
    rn = _roman(n)
    if r < 0.4:
        if lang == "fr":
            return f"le {rn}e siècle"
        art = {"es": "el ", "pt": "o ", "it": "il ", "de": "das ", "nl": ""}[lang]
        if lang == "de":
            return f"das {rn}. Jahrhundert"
        if lang == "nl":
            return f"hoofdstuk {rn}"
        return f"{art}{T.ROMAN_AFTER[lang]} {rn}" if lang in ("es", "pt") else f"{art}{rn} {T.ROMAN_AFTER[lang]}"
    if r < 0.7:
        w = rng.choice(T.ROMAN_CTX[lang])
        art = {"es": "el ", "pt": "o ", "it": "il ", "fr": "le ", "de": "", "nl": ""}[lang]
        n2 = rng.randint(1, 12)
        return f"{art}{w} {_roman(n2)}"
    n = rng.randint(2, 16)
    suf = "." if lang == "de" else ""
    return f"{rng.choice(T.ROMAN_NAMES[lang])} {_roman(n)}{suf}"


def g_version(lang, rng, P, ci, broad, gender):
    if ci % 2 == 0:
        return ".".join(str(rng.choice([rng.randint(0, 20), rng.randint(0, 9)])) for _ in range(rng.choice([3, 3, 4])))
    return ".".join(str(x) for x in (rng.choice([192, 10, 172, 127]), rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)))


YEAR_QUOTA: dict = {}  # year value -> rows wanted (a top-up pass aimed at thin year values); empty = the usual spread


def g_year(lang, rng, P, ci, broad, gender):
    if YEAR_QUOTA:
        ys = sorted(YEAR_QUOTA)
        return str(rng.choices(ys, [YEAR_QUOTA[y] for y in ys])[0])
    xs = [y for y in P["year"] if 1900 <= y <= 2035]
    return str(rng.choice(xs) if (xs and not broad and rng.random() < 0.6) else rng.randint(1900, 2030))


def g_decade(lang, rng, P, ci, broad, gender):
    form = rng.choice(T.DECADE_FORMS[lang])
    if lang == "de" and rng.random() < 0.4:
        return "den " + str(rng.choice([1950, 1960, 1970, 1980, 1990])) + "er Jahren"
    return form.format(d=rng.choice([20, 30, 40, 50, 60, 70, 80, 90]))


def g_ord(lang, rng, P, ci, broad, gender):
    xs = [n for n in P["ord"] if 2 <= n <= 30]
    n = rng.choice(xs) if (xs and not broad) else rng.randint(2, 30)
    fem = gender == "f"
    if lang in ("es", "pt", "it"):
        return f"{n}{'ª' if fem else 'º'}"
    if lang == "fr":
        return f"{n}{'re' if (fem and n == 1) else 'er' if n == 1 else rng.choice(['e', 'e', 'ème'])}"
    if lang == "de":
        return f"{n}."
    return f"{n}{'ste' if n in (1, 8) or n >= 20 else 'de' if n in (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19) else 'e'}"


def g_amp(lang, rng, P, ci, broad, gender):  # readings with an English "and" are rejected in reading_ok
    return rng.choice(T.AMP_ACRONYMS if ci == 0 else T.AMP_NAMES)


GEN = {"count": g_count, "large": g_large, "dec": g_dec, "pct": g_pct, "permille": g_permille, "unit": g_unit,
       "temp": g_temp, "frac": g_frac, "range_from": g_range_from, "range_between": g_range_between, "price": g_price,
       "price_big": g_price_big, "clock24": g_clock24, "clock12": g_clock12, "date_num": g_date_num,
       "date_txt": g_date_txt, "phone": g_phone, "code": g_code, "roman": g_roman, "version": g_version,
       "year": g_year, "decade": g_decade, "ord": g_ord, "amp": g_amp}

# --------------------------------------------------------------------------- reading filters

FEM_RE = {"es": r"(?:ientas|\buna|una$)", "pt": r"(?:entas$|\buma$|uma$|\bduas$|duas$)", "de": r"^eine?r?$"}
HALF_BAD = re.compile(r"\b(?:e meia?|y media?|et demie?|et demi|e mezzo|und (?:ein )?halb\w*|"
                      r"en (?:(?:een|één) )?half)\b|anderhalf|anderthalb")


def _digits_int(text: str):
    d = re.sub(r"\D", "", text)
    return int(d) if d and len(d) < 12 else None


def _fem_words(lang: str, kind: str, n):
    out = set()
    if n is None:
        return out
    for k in ("ord", "card"):
        try:
            ws = M.g(lang, k, n)
        except M.Missing:
            continue
        out |= set(M._fem_words(ws, lang)) - set(ws)
    return out


def _is_fem(lang, kind, text, reading, n):
    ws = reading.split()
    fs = _fem_words(lang, kind, n)
    if any(w in fs for w in ws):
        return True
    rx = FEM_RE.get(lang)
    return bool(rx and any(re.search(rx, w) for w in ws))


CONN = re.compile(r"^\s*(?:de|del|do|di|da)\b")
SING_HOUR = re.compile(r"\b(?:hora|heure)\b")
ONE_WORDS = {"um", "uma", "un", "une", "una", "uno"}
DAYPART_RE = re.compile(r"(?:madrugada|manhã|tarde|noite|noche|mañana|matin|soir|après-midi|mattino|mattina|pomeriggio|sera|"
                        r"morgens|vormittags|nachmittags|abends|ochtends|morgens|middags|avonds|a eme|pe eme)")


DP_AFT = re.compile(r"(?:\btarde\b|après-midi|pomeriggio|nachmittags|'s middags)")
DP_EVE = re.compile(r"(?:\bnoche\b|\bnoite\b|\bsoir\b|\bsera\b|abends|'s avonds)")
DP_MOR = re.compile(r"(?:mañana|manhã|\bmatin\b|mattina|mattino|vormittags|morgens|'s ochtends)")
DP_MAD = re.compile(r"madrugada")


VARIETY_FILL = False  # the reading-variety fill (ml_tts_variety) sets it: relative / 12-hour clock readings and decimal halves allowed
REL_WORDS = {"de": re.compile(r"\b(?:halb|[Vv]iertel|dreiviertel|nach|vor)\b"),
             "nl": re.compile(r"\b(?:half|kwart|over|na|voor)\b"),
             "es": re.compile(r"\b(?:menos|cuarto|media|mediodía)\b"), "fr": re.compile(r"\b(?:moins|quart|demie?|midi|minuit)\b"),
             "it": re.compile(r"\b(?:meno|quarto|quarti|mezza|mezzo|mezzogiorno|mezzanotte)\b"),
             "pt": re.compile(r"\b(?:menos|quarto|meia|meio-dia|meia-noite)\b")}


def _clock_ok(lang, span, reading):
    """Natural spoken clock readings only (the candidate list is liberal: it also holds "tres horas" for 15:00)."""
    m = re.match(r"(\d{1,2})", span.text)
    h = int(m.group(1)) if m else None
    ap = bool(re.search(r"[ap]\.?\s?m", span.text, re.I))
    if h is None:
        return True
    ws = reading.split()
    try:
        cw = M.g(lang, "card", h)
    except M.Missing:
        cw = []
    if lang == "fr" and G.FR_ET_BAD.search(reading):
        return False  # "seize heures et quinze", "cinq et vingt deux" are not French; "et quart" / "et demie" are
    if VARIETY_FILL and not ap and (REL_WORDS[lang].search(reading) or (h > 12 and ws[:1] != cw[:1])):
        # the reading-variety fill: a 24 h time said the everyday way ("halb drei", "kwart over twee", "las tres y
        # media", "Viertel nach neun") or with the 12-hour word; a duration-like "tres horas" stays out
        return not (lang in ("es", "pt", "it") and re.search(r"\b(?:horas?|ore)\b", reading))
    if h > 12 and not ap:
        return ws[:len(cw)] == cw or ws[:len(cw)] == M._fem_hour(cw, lang, h)  # 24 h written: the 24 h number
    if lang in ("es", "pt", "it") and re.search(r"\b(?:horas?|ore)\b", reading) and h <= 12:
        return False  # "tres horas treinta" sounds like a duration
    if SING_HOUR.search(reading) and not (ws and ws[0] in ONE_WORDS):
        return False
    if lang in ("de", "nl") and h <= 12 and not ap and not re.search(r"\b(?:Uhr|uur)\b", reading):
        return False
    if ap:
        pm = bool(re.search(r"p\.?\s?m", span.text, re.I))
        aft, eve = DP_AFT.search(reading), DP_EVE.search(reading)
        mor, mad = DP_MOR.search(reading), DP_MAD.search(reading)
        if pm:
            if aft and not (h in (12, 1, 2, 3, 4, 5, 6)):
                return False
            if eve and not (6 <= h <= 11):
                return False
            if mor or mad:
                return False
        else:
            if aft or eve:
                return False
            if mad and not (h in (12, 1, 2, 3, 4, 5)):
                return False
            if mor and h in (12, 1, 2, 3, 4) and not mad:
                return False
    return True


def _roman_ctx(written, span):
    """"century" (siglo XXI / XXI siècle), "word" (capítulo IV) or "name" (Luis XIV)."""
    before = written[:span.start].rstrip()
    after = written[span.end:].lstrip()
    if re.match(r"(?:siècle|secolo|Jahrhundert|eeuw|siglo|século)\b", after) or re.search(r"(?:siglo|século)$", before):
        return "century"
    ctx_words = {w.lower() for ws in T.ROMAN_CTX.values() for w in ws}
    if before and before.split()[-1].lower() in ctx_words:
        return "word"
    return "name"


def _roman_ok(lang, written, span, reading):
    n = M._roman(re.sub(r"[^IVXivx]", "", span.text).upper()) or (5 if span.text.strip()[:1] in "Vv" else 10)
    ctx = _roman_ctx(written, span)
    try:
        ordw = M.g(lang, "ord", n)
    except M.Missing:
        ordw = []
    is_ord = reading.split() == ordw or (lang == "de" and reading.endswith(("e", "en", "er", "es", "em")) and not reading.startswith("der "))
    if any(w in _fem_words(lang, "ord", n) for w in reading.split()):
        return False
    if lang == "it" and ctx == "century":
        return is_ord  # "il settimo secolo"
    if lang in ("es", "pt", "it"):
        if is_ord and n > 10:
            return False
        if ctx == "name":
            return is_ord if n <= 10 else not is_ord
        return True
    if lang == "fr":
        suffixed = bool(re.search(r"[a-zᵉ]$", span.text))
        return True if suffixed else (not is_ord or ctx != "name")
    if lang == "de":
        if ctx == "name":
            return reading.startswith("der ")
        if ctx == "century":
            return bool(ordw) and reading == ordw[0]  # "das einundzwanzigste Jahrhundert"
        return not reading.startswith(("der ", "dem ", "den ", "des ")) and not is_ord  # Kapitel IV -> "vier"
    if lang == "nl":
        if ctx == "name":
            return reading.startswith("de ")
        return not reading.startswith("de ") and not is_ord  # "akte zes"
    return True


# feminine number forms are right only before a feminine noun (a count / amount of one) or for clock hours in pt
# ("às duas"); a year, date, percent, version, decimal or price never takes them ("mil novecentas e cinquenta",
# "duas mil e três", "cincuenta y una por ciento", "el una de mayo": judge audit 2026-10-17)
DEC_WORD_RE = re.compile(r"\b(?:komma|Komma|coma|virgule|virgola|vírgula|punto|ponto|point|Punkt|punt)\b")
CENT_WORD_RE = re.compile(r"\b(?:[Cc]ent|cents|centimes?|céntimos?|cêntimos?|centesim[oi]|centavos?)\b")
FEM_OK_CLS = {"count", "large", "range_between", "range_from", "unit", "frac", "temp"}
APOCOPE_BAD = {"es": re.compile(r"\bun\b"), "it": re.compile(r"(?:un|prima)\b")}  # "el un de", "duemilaun", "il prima"
MASC_HOUR = {"pt": re.compile(r"\b(?:um|dois) horas?\b"), "fr": re.compile(r"\bun heures?\b")}


def _gender_ok(lang, cls, gender, span, reading):
    if lang not in ("es", "pt", "it"):
        rx = MASC_HOUR.get(lang)
        return not (rx and span.kind == "clock" and rx.search(reading))
    if span.kind == "clock" or cls in ("clock12", "clock24"):
        rx = MASC_HOUR.get(lang)
        return not (rx and rx.search(reading))
    if span.kind in ("ord", "roman"):
        return True  # their own gender rule (the clause's noun) is in reading_ok
    fem = _is_fem(lang, span.kind, span.text, reading, _digits_int(span.text))
    if fem and not (gender == "f" and cls in FEM_OK_CLS and span.kind in ("int", "unit", "range", "frac", "temp")):
        return False
    ap = APOCOPE_BAD.get(lang)
    if ap and span.kind in ("date", "version", "dec") and ap.search(reading):  # "un per cento" / "un por ciento" are fine
        return False
    if lang == "it" and span.kind == "int" and cls in ("year", "date_num", "date_txt") and re.search(r"un\b", reading):
        return False
    return True


def _whole_number_reading(lang, span, reading) -> bool:
    """A count of >= 1000 read as one number (the grammar's reading or its gender / apocope variants), not in digit
    chunks; smaller counts and values the grammar table lacks are left to the other rules."""
    n = _digits_int(span.text)
    if n is None or n < 1000 or not re.fullmatch(r"\s*\d[\d.  ]*\s*", span.text):
        return True
    try:
        base = M.g(lang, "card", n)
    except M.Missing:
        return True
    ok = {" ".join(base)} | {" ".join(v) for v in M._card_variants(base, lang, n, year=False)}
    return reading.strip() in ok or reading.strip() == span.cands[0].strip()


def reading_ok(lang, grp, cls, gender, span, reading, date_pos, written=""):
    """May this candidate reading be spoken in this clause?"""
    want = T.WANT.get((lang, grp if grp != "ORD" else ("ORD_M" if gender == "m" else "ORD_F")), {})
    key = span.kind + (str(date_pos) if span.kind == "date" else "")
    rx = want.get(key) or want.get(span.kind)
    if span.kind == "range" and span.text[:1].isdigit():
        rx = None  # the left end carries no joiner word; only the right part ("-12" -> " a doce") does
    if rx and not re.search(rx, reading.strip()):
        return False
    if span.kind == "int" and cls in ("count", "range_between") and _is_fem(lang, "int", span.text, reading, _digits_int(span.text)):
        return False
    if not _gender_ok(lang, cls, gender, span, reading):
        return False
    if span.kind == "ord" and cls == "ord":
        n = _digits_int(span.text)
        fs = _fem_words(lang, "ord", n)
        has_fem = any(w in fs for w in reading.split())
        if fs and has_fem != (gender == "f"):
            return False
    if span.kind == "roman" and not _roman_ok(lang, written, span, reading):
        return False
    if span.kind == "clock" or (span.kind == "int" and cls == "clock12"):
        if not _clock_ok(lang, span, reading):
            return False
    if cls in ("large", "dec") and span.kind == "dec" and HALF_BAD.search(reading) and not VARIETY_FILL:
        return False
    if span.kind == "int" and cls in ("count", "large") and not _whole_number_reading(lang, span, reading):
        return False  # "191362 klanten" read "één negen één drie tweeënzestig": a count is said as one number
    # the known generator slips (grammar_rules: "el primera de octubre", "dois e trinta", "pesos ... céntimos",
    # "achtundvierzig und zehn", "½ de litro", "en één half", ...): the same rules that took the old rows out
    if G.span_errors(lang, {"text": span.text, "kind": span.kind, "reading": reading},
                     {"cls": cls, "caption": written}):
        return False
    if YEAR_QUOTA and cls == "year" and re.fullmatch(r"(19|20)\d\d", span.text) and reading.strip() != span.cands[0].strip():
        return False  # a year top-up trains the full reading; the clipped "'55" and pair forms add no year value
    if span.kind == "cur" and DEC_WORD_RE.search(reading) and CENT_WORD_RE.search(reading):
        return False  # "dodici euro virgola cinquanta centesimi": a decimal word AND a cents word is no reading
    if span.kind == "amp" and re.search(r"\band\b", reading):
        return False  # an English "and" in a native sentence: the voice cannot say it and the label would teach it
    return True


def _digit_style(span, reading):
    """True when the reading is digit by digit (one word per digit)."""
    nd = len(re.sub(r"\D", "", span.text))
    nw = len([w for w in reading.split() if w not in ("punto", "ponto", "point", "Punkt", "punt")])
    return nd > 0 and nw == nd


# --------------------------------------------------------------------------- planning

def _words(s):
    return len(s.split())


def build_text(lang, cls, rng, P, broad=False):
    """(written sentence, group, gender, register) or None."""
    grp, gender = GROUP[cls], None
    if cls == "ord":
        gender = rng.choice("mf")
        grp = "ORD_M" if gender == "m" else "ORD_F"
    clauses = T.CLAUSES[lang][grp]
    ci = rng.randrange(len(clauses))
    slot = GEN[cls](lang, rng, P, ci, broad, gender)
    if not slot:
        return None
    clause = clauses[ci].replace("{n}", slot)
    reg, lead = rng.choice(T.LEADS[lang])
    tail = rng.choice(T.TAILS[lang])
    if lead:
        cap = lang == "de" and lead.endswith(":")
        clause = (clause[0].upper() + clause[1:]) if cap else clause
        text = f"{lead} {clause}{tail}."
    else:
        text = f"{clause[0].upper()}{clause[1:]}{tail}."
    text = text.replace("..", ".")  # "Karl XV." + sentence period
    if not (WORDS_MIN <= _words(text) <= WORDS_MAX):
        return None
    return text, grp, gender, reg


def _natural_group(lang, span, reading) -> bool:
    """A phone group's reading has a shape speakers use (reading_conv.group_shape): digit by digit, one number,
    1+2, pairs, ... never a 2+1 / 3+1 split ("sesenta y uno dos" for "612")."""
    import reading_conv as RC

    sh = RC.group_shape(lang, re.sub(r"\D", "", span.text), reading)
    return not (sh.startswith("X") or sh == "?")


def realize(lang, cls, written, grp, gender, rng, used, names=None, pick=None):
    """Spans of the written form -> (label, spans, offsets) or (None, reason). `used` counts picks per (kind, reading).
    names (optional): ml_cands readings to add (spans_regional names: the regional / fill readings). pick (optional):
    pick(spans, ok) -> one candidate index per span chosen from ok[i] (the readings that fit the clause), or None when
    no choice gives what the caller wants (the reading-variety fill picks by convention, ml_tts_variety)."""
    try:
        sp = M.spans_regional(written, lang, names) if names else M.spans(written, lang)
    except Exception as e:  # a candidate-machinery error is a drop, not a crash
        return None, f"spans_error {type(e).__name__}"
    if not sp:
        return None, "no_spans"
    if any(not s.cands for s in sp):
        return None, "missing_grammar" if any(s.note.startswith("missing") for s in sp) else "no_cands"
    if any(c.isdigit() for c in written) and not M.all_digits_spanned(written, lang):
        return None, "digits_unspanned"
    kinds = {s.kind for s in sp}
    if not (kinds & EXPECT[cls]):
        return None, "shape_not_recognised"
    style = rng.choices(["digits", "chunks", "mix"], [0.4, 0.35, 0.25])[0]
    picks, dpos, prev_conn, oks = [], 0, None, []
    for s in sp:
        ok = [i for i, c in enumerate(s.cands) if reading_ok(lang, grp, cls, gender, s, c, dpos, written)]
        if s.kind == "phone":  # never an unnatural split (the old "chunks" fallback let 2+1 splits through for 3-digit groups)
            ok = [i for i in ok if _natural_group(lang, s, s.cands[i])]
        if s.kind == "date":
            if lang in ("es", "pt") and dpos in (1, 2):
                ok = [i for i in ok if CONN.match(s.cands[i])] or ok  # "quince de marzo de dos mil seis"
            if lang in ("es", "pt", "it") and dpos == 2 and prev_conn is not None:
                ok = [i for i in ok if bool(CONN.match(s.cands[i])) == prev_conn] or ok
            dpos += 1
        if not ok:
            return None, "no_reading_fits_clause"
        if pick is not None:
            oks.append(ok)
            continue
        if s.kind in ("phone", "version") and style != "mix":
            want_d = style == "digits"
            ok2 = [i for i in ok if _digit_style(s, s.cands[i]) == want_d]
            if not want_d:  # speakers chunk phone numbers in pairs, not in hundreds
                ok2 = [i for i in ok2 if len(s.cands[i].split()) <= 3 and len(re.sub(r"\D", "", s.text)) <= 2] or ok2
            ok = ok2 or ok
        w = []
        for i in ok:
            base = 3.0 if i == 0 else 1.5 if i <= 2 else 1.0
            w.append(base / (1.0 + 0.7 * used[(lang, s.kind, s.cands[i].strip())]))
        choice = rng.choices(ok, w)[0]
        picks.append(choice)
        if s.kind == "date" and dpos == 2:
            prev_conn = bool(CONN.match(s.cands[choice]))
    if pick is not None:
        picks = pick(sp, oks)
        if picks is None:
            return None, "no_reading_of_convention"
    offs: list = []
    label = C.render_cap(written, sp, picks, offs)
    if written.endswith(".") and not label.endswith("."):
        label += "."  # the span took the period of "a. m." with it
    if FORBIDDEN.search(label) or C.WF.search(label):
        return None, "symbol_left"
    toks = label.split()
    flags = C.number_tokens(label, offs)
    for i, t in enumerate(toks):
        if not flags[i] and C.ABBR_NEAR.match(t.strip(",;:()")) and (
                (i > 0 and flags[i - 1]) or (i + 1 < len(toks) and flags[i + 1])):
            return None, "abbr_left"
    spans = [{"text": s.text, "kind": s.kind, "reading": s.cands[p].strip(), "canonical": s.cands[0].strip(),
              "idx": p, "n_cands": len(s.cands)} for s, p in zip(sp, picks)]
    for s, p in zip(sp, picks):
        used[(lang, s.kind, s.cands[p].strip())] += 1
    return (label, spans, offs), ""


def _pick_voice(lang, rng):
    vs = VOICES[lang]
    dev = [v for v in vs if v["dev"]]
    train = [v for v in vs if not v["dev"]]
    pool = dev if (dev and rng.random() < DEV_FRAC) else train
    v = rng.choices(pool, [x["w"] for x in pool])[0]
    spk = rng.randint(*v["spk"]) if v.get("spk") else None
    return v, spk


def plan(lang: str, n: int, seed: int, report: dict | None = None, broad: bool = False, classes: tuple = (),
         tag: str = TAG) -> list[dict]:
    """~n rows, equal quota per class (a class the grammar table cannot fill is topped up by the others, <= 1.5x quota).
    Deterministic in (lang, n, seed). `report` (optional dict) receives per-class planned/drop counters. `classes`
    (optional) restricts the plan to those classes (a top-up pass for thin cells); `tag` prefixes the row ids."""
    rng = random.Random(f"{TAG}|{seed}|{lang}")
    P = pools(lang)
    CL = [c for c in CLASSES if c in classes] if classes else CLASSES
    assert CL, f"no known class in {classes}"
    quota = {c: n // len(CL) + (1 if i < n % len(CL) else 0) for i, c in enumerate(CL)}
    used = collections.Counter()
    seen: set = set()
    rows: list = []
    drops = collections.defaultdict(collections.Counter)
    got = collections.Counter()

    def fill(cls, want, cap):
        tries = 0
        while got[cls] < want and tries < cap:
            tries += 1
            bt = build_text(lang, cls, rng, P, broad)
            if bt is None:
                drops[cls]["template_reject"] += 1
                continue
            written, grp, gender, reg = bt
            res, why = realize(lang, cls, written, grp, gender, rng, used)
            if res is None:
                drops[cls][why] += 1
                continue
            label, spans, offs = res
            if label in seen:
                drops[cls]["duplicate"] += 1
                continue
            seen.add(label)
            v, spk = _pick_voice(lang, rng)
            speed = round(rng.uniform(SPEED_LO, SPEED_HI), 2)
            rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
            rows.append({"id": rid, "lang": lang, "cls": cls, "register": reg, "caption": written, "text": label,
                         "spans": spans, "num_chars": offs, "kind": sorted({s['kind'] for s in spans}),
                         "voice": v["id"], "engine": v["engine"], "voice_path": v.get("path"), "speaker": spk,
                         "voice_license": v["license"], "split": "dev" if v["dev"] else "train", "speed": speed,
                         "pitch": round(rng.uniform(0.95, 1.05), 3), "noise_scale": round(rng.uniform(0.5, 0.8), 2),
                         "noise_w": round(rng.uniform(0.6, 1.0), 2)})
            got[cls] += 1

    for cls in CL:
        fill(cls, quota[cls], 600 * max(1, quota[cls]))
    short = sum(max(0, quota[c] - got[c]) for c in CL)
    full = [c for c in CL if got[c] >= quota[c]]
    if short and full:
        extra = -(-short // len(full))
        for c in full:
            fill(c, min(got[c] + extra, int(quota[c] * 1.5) + 1), 300 * (extra + 1))
    if report is not None:
        report["quota"] = quota
        report["got"] = dict(got)
        report["drops"] = {c: dict(v) for c, v in drops.items()}
    return rows


def plan_texts(lang: str, n: int, seed: int) -> list[str]:
    """Written sentences of the BROAD plan (values from wide domains, no table check): input for the grammar dumper."""
    rng = random.Random(f"{TAG}|texts|{seed}|{lang}")
    P = pools(lang)
    out: list = []
    for i in range(n):
        cls = CLASSES[i % len(CLASSES)]
        for _ in range(50):
            bt = build_text(lang, cls, rng, P, broad=True)
            if bt:
                out.append(bt[0])
                break
    return out


def needed_keys(lang: str, n: int, seed: int) -> tuple[list[str], list[str]]:
    """(texts, grammar-table keys the broad plan needs that the current table lacks)."""
    texts = plan_texts(lang, n, seed)
    tab = M.table()
    return texts, sorted(k for k in M.needed(texts, lang) if k not in tab)


# --------------------------------------------------------------------------- verification (pure)

def check_greedy(row: dict, greedy: str) -> tuple[bool, str, dict]:
    """Round-trip rule: CER(label, greedy) <= CER_MAX and every spoken reading fits the greedy text (ml_decide.fits:
    a run of greedy words that ends at a word boundary; fuzzy only for long readings)."""
    import ml_decide as D

    c = C.cer(row["text"], greedy)
    info = {"cer_ctc": round(c, 3)}
    if c > CER_MAX:
        return False, "cer", info
    fuzzy = False
    for a, b in row["num_chars"]:
        reading = row["text"][a:b]
        if D.fits(reading, greedy):
            continue
        if D.fits(reading, greedy, fuzzy=True):
            fuzzy = True
            continue
        return False, "reading_not_heard", info
    info["fit"] = "fuzzy" if fuzzy else "exact"
    return True, "", info


# --------------------------------------------------------------------------- Modal: synthesis + verification + alignment

try:
    import modal

    _MODAL = True
except ImportError:  # local tests and `plan` need no Modal
    modal = None
    _MODAL = False

if _MODAL:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("espeak-ng", "libsndfile1", "ffmpeg")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "scipy", "numpy<2",
                          "huggingface_hub", "kokoro==0.9.4", "misaki[en]>=0.9.4", "piper-tts==1.2.0")
             .env({"HF_HOME": "/root/hf"})
             .add_local_python_source("grammar_rules", "ml_cands", "ml_decide", "ml_p3_core", "ml_tts",
                                      "ml_tts_templates")
             .add_local_file(os.path.join(HERE, "ml_grammar_table.json"), "/root/ml_grammar_table.json"))
    app = modal.App("parakeet-ml-tts", image=image)
    _fn = app.function(gpu="L4", cpu=4, memory=16384, volumes={VOL: vol}, timeout=3 * 3600, max_containers=6)
else:
    def _fn(f):
        return f


class _Synth:
    """Kokoro and Piper behind one call: text -> float32 mono at 16 kHz."""

    def __init__(self):
        self.kokoro = {}
        self.piper = {}

    def _kokoro(self, voice):
        from kokoro import KPipeline

        # a b e f h i p j z: the Kokoro language code is the first letter of the voice id ("blend:<code>:..." for blends)
        code = voice.split(":")[1] if voice.startswith("blend:") else voice[0]
        if code not in self.kokoro:
            self.kokoro[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
        return self.kokoro[code]

    def _piper(self, path):
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        if path not in self.piper:
            onnx = hf_hub_download("rhasspy/piper-voices", path + ".onnx")
            cfg = hf_hub_download("rhasspy/piper-voices", path + ".onnx.json")
            self.piper[path] = PiperVoice.load(onnx, config_path=cfg, use_cuda=False)
        return self.piper[path]

    def __call__(self, row):
        import numpy as np
        from scipy.signal import resample_poly

        if row["engine"] == "presynth":  # synthesised elsewhere (cloud voices, ml_tts_cloud.py), 16 kHz mono wav
            import soundfile as sf

            x, sr = sf.read(row["presynth"], dtype="float32")
            return x.mean(1) if x.ndim > 1 else x
        if row["engine"] == "kokoro":
            pipe = self._kokoro(row["voice"])
            voice = row["voice"]
            if voice.startswith("blend:"):  # weighted mix of Kokoro voice packs: "blend:i:if_sara*0.6+ef_dora*0.4"
                parts = [p.split("*") for p in voice.split(":", 2)[2].split("+")]
                voice = sum(float(w) * pipe.load_single_voice(n) for n, w in parts)
            ch = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                  for _, _, a in pipe(row["text"], voice=voice, speed=row["speed"], split_pattern=None)]
            x, sr = (np.concatenate(ch) if ch else np.zeros(0, dtype=np.float32)), 24000
        else:
            v = self._piper(row["voice_path"])
            raw = b"".join(v.synthesize_stream_raw(row["text"], speaker_id=row["speaker"], length_scale=1.0 / row["speed"],
                                                   noise_scale=row["noise_scale"], noise_w=row["noise_w"],
                                                   sentence_silence=0.15))
            x, sr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, v.config.sample_rate
        if len(x) < sr // 10:
            return None
        g = int(np.gcd(sr, 16000))
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        p = row["pitch"]  # small pitch/tempo wobble: a resample ratio close to 1 (voices are few)
        if abs(p - 1) > 0.005:
            num, den = int(round(p * 100)), 100
            x = resample_poly(x, den, num).astype(np.float32)
        return x


def _kept_row(row, path, x, wt, info, lang) -> dict:
    """The rows.jsonl row of a verified clip (x = the written audio, wt = its word times)."""
    return {"audio_filepath": path, "duration": round(len(x) / 16000, 3), "text": row["text"], "lang": lang, "src": "tts",
            "id": row["id"], "origin": f"{TAG}/{row['voice']}", "caption": row["caption"], "kind": row["kind"],
            "cls": row["cls"], "spans": row["spans"], "num_chars": row["num_chars"],
            "word_times": [[round(a, 3), round(b, 3)] for a, b in wt], "cer_ctc": info["cer_ctc"],
            "fit": info.get("fit"), "edge_quiet": C.edge_quiet(x), "split": row["split"], "voice": row["voice"],
            "voice_license": row["voice_license"], "speed": row["speed"], "engine": row["engine"],
            **{k: row[k] for k in PASS_THROUGH if k in row}}


@_fn
def tts_lang(lang: str, n: int = 600, seed: int = 1, hours: float = 0.0, overgen: float = 1.6, broad: bool = False,
             classes: str = "", out_tag: str = TAG, rows_file: str = "", keep_failed: bool = False) -> dict:
    """Plan ~n*overgen sentences, synthesise, keep the clips the CTC reader confirms, write /vol/ml/tts1/<lang>/.
    `hours` > 0 stops once that much audio is kept. `broad`: draw values from wide domains instead of the ones the grammar table already
    holds (use after the table was extended with `needed` keys). Writes rows.jsonl, wav/*.wav and stats.json (drops per class/voice).
    keep_failed (2026-10-18, default off = unchanged): the clips the CTC reader rejects but can still align are written too,
    to rows_ctc_failed.jsonl (with the greedy text and the reason), never to rows.jsonl: a second, independent check (a
    verbatim second transcript that matches the label word for word) may accept them later (the reading-variety fill)."""
    import soundfile as sf
    import numpy as np

    import ml_decide as D

    rep: dict = {}
    if rows_file:  # rows planned + synthesised elsewhere (engine "presynth"); verified here exactly like local voices
        rows = [json.loads(l) for l in open(f"{VOL}/{rows_file}", encoding="utf-8") if l.strip()]
    else:
        rows = plan(lang, int(n * overgen), seed, rep, broad, tuple(c for c in classes.split(",") if c), out_tag)
    random.Random(f"{TAG}|order|{seed}|{lang}").shuffle(rows)  # classes interleaved, so an `hours` cap stays balanced
    out = f"{VOL}/ml/{out_tag}/{lang}"  # a top-up pass writes its own dir (never overwrites tts1)
    os.makedirs(f"{out}/wav", exist_ok=True)
    ctc = D.LangCTC(lang)
    syn = _Synth()
    r = random.Random(f"{TAG}|audio|{seed}|{lang}")
    kept, failed, secs = [], [], 0.0
    by_cls = collections.defaultdict(collections.Counter)
    by_voice = collections.defaultdict(collections.Counter)
    for row in rows:
        vk = row["voice"] + ("/dev" if row["split"] == "dev" else "")
        by_cls[row["cls"]]["planned"] += 1
        by_voice[vk]["planned"] += 1
        if hours and secs >= hours * 3600:
            by_cls[row["cls"]]["skipped_hours_cap"] += 1
            continue
        try:
            x = syn(row)
        except Exception as e:  # one bad voice must not end the run
            x = None
            by_cls[row["cls"]]["synth_error:" + type(e).__name__] += 1
        if x is None:
            by_voice[vk]["dropped"] += 1
            continue
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (-3 / 20) * 10 ** (r.uniform(-12, 0) / 20)
        pad = lambda: np.zeros(int(16000 * r.uniform(0.2, 0.6)), dtype=np.float32)  # noqa: E731
        x = np.concatenate([pad(), x, pad()])
        lp = ctc.lp(x)
        greedy = ctc.greedy(lp)
        ok, why, info = check_greedy(row, greedy)
        if not ok:
            by_cls[row["cls"]]["drop_" + why] += 1
            by_voice[vk]["dropped"] += 1
            by_voice[vk]["drop_" + why] += 1
            if keep_failed:
                wt = ctc.word_times(lp, row["text"])
                if all(t is not None for t in wt):
                    path = f"{out}/wav/{row['id']}.wav"
                    sf.write(path, x, 16000, subtype="PCM_16")
                    failed.append(_kept_row(row, path, x, wt, info, lang) | {"ctc_fail": why, "greedy": greedy})
            continue
        wt = ctc.word_times(lp, row["text"])
        if any(t is None for t in wt):
            by_cls[row["cls"]]["drop_align"] += 1
            by_voice[vk]["dropped"] += 1
            continue
        path = f"{out}/wav/{row['id']}.wav"
        sf.write(path, x, 16000, subtype="PCM_16")
        secs += round(len(x) / 16000, 3)
        by_cls[row["cls"]]["kept"] += 1
        by_voice[vk]["kept"] += 1
        kept.append(_kept_row(row, path, x, wt, info, lang))
    with open(f"{out}/rows.jsonl", "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    if keep_failed:
        with open(f"{out}/rows_ctc_failed.jsonl", "w", encoding="utf-8") as f:
            for k in failed:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
    stats = {"lang": lang, "seed": seed, "planned": len(rows), "kept": len(kept), "hours": round(secs / 3600, 3),
             "plan_report": rep, "by_class": {c: dict(v) for c, v in by_cls.items()},
             "by_voice": {v: dict(c) for v, c in by_voice.items()}}
    if keep_failed:
        stats["ctc_failed_aligned"] = len(failed)
    json.dump(stats, open(f"{out}/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    vol.commit()
    print(lang, "kept", len(kept), "of", len(rows), f"{secs / 3600:.2f} h")
    return {"lang": lang, "kept": len(kept), "planned": len(rows), "hours": stats["hours"]}


if _MODAL:
    @app.local_entrypoint()
    def main(langs: str = "es,fr,de,it,pt,nl", n: int = 600, seed: int = 1, hours: float = 0.0, broad: bool = False,
             classes: str = "", out_tag: str = TAG, rows_file: str = "", keep_failed: bool = False):
        # rows_file: a per-language path template with {lang}, e.g. ml/tts3/{lang}/planned.jsonl
        for res in tts_lang.starmap([(lang, n, seed, hours, 1.6, broad, classes, out_tag,
                                      rows_file.format(lang=lang) if rows_file else "", keep_failed)
                                     for lang in langs.split(",")]):
            print(res)


# --------------------------------------------------------------------------- local CLI

def _cli(argv):
    cmd = argv[1] if len(argv) > 1 else "plan"
    if cmd == "plan":
        lang, n, seed = argv[2], int(argv[3]), int(argv[4])
        rep: dict = {}
        for r in plan(lang, n, seed, rep, broad="--broad" in argv):
            print(f"{r['cls']:14s} {r['voice']:12s} W: {r['caption']}\n{'':14s} {'':12s} L: {r['text']}")
        print(json.dumps(rep["got"]), file=sys.stderr)
    elif cmd == "report":
        n, seed = int(argv[2]), int(argv[3])
        print("lang " + " ".join(f"{c[:6]:>6s}" for c in CLASSES))
        for lang in LANGS:
            rep = {}
            plan(lang, n, seed, rep)
            print(f"{lang:4s} " + " ".join(f"{rep['got'].get(c, 0):6d}" for c in CLASSES), "quota", rep["quota"][CLASSES[0]])
    elif cmd == "needed":
        lang, n, seed = argv[2], int(argv[3]), int(argv[4])
        texts, keys = needed_keys(lang, n, seed)
        json.dump(keys, open(f"tts_needed_{lang}.json", "w", encoding="utf-8"))
        json.dump(texts, open(f"tts_texts_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(lang, len(texts), "texts,", len(keys), "missing grammar keys -> tts_needed_%s.json / tts_texts_%s.json" % (lang, lang))
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli(sys.argv)
