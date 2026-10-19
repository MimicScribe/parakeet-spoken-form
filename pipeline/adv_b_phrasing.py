"""Adversarial pack, family 3: unusual but common phrasings (en es fr): regional digit forms, version numbers, regnal names,
addresses, measurements and math, dates and clocks in every spoken order, ranges said fast. Labels are what is said (strict
spoken form; a letter after a house number is the letter: "twenty two B"). Each maker is (key, rng) -> Piece; the readout
splits keys (V_a / V_b) and holds whole makers out (C_b). Pure functions."""
import random

import adv_b_words as W
from edge_a_content import Piece

AM_HALF = ("half_seven",)
AM_DATE = ("date_order",)
AM_RANGE = ("fast_range",)
AM_REG = ("regional_numerals",)


def _p(label, nums, ambig=(), synth=None):
    p = Piece(label, "", list(nums), tuple(ambig))
    p.synth = synth or label
    return p


def _c(n, lang):
    return W.card(int(n), lang)


def roman(n: int) -> str:
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
                 (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out += s
            n -= v
    return out


# ================================================================================================ ENGLISH
def en_year(y: int, rng) -> str:
    if 2000 <= y <= 2009:
        return "two thousand" + (f" {_c(y - 2000, 'en')}" if y > 2000 else "")
    if y >= 2010 and rng.random() < 0.5:
        return f"two thousand {_c(y - 2000, 'en')}"
    a, b = divmod(y, 100)
    return f"{_c(a, 'en')} " + (W.pair(f"{b:02d}", "en", "oh") if b else "hundred")


def en_emerg(k, rng):
    words = {"911": "nine one one", "999": rng.choice(["nine nine nine", "triple nine", "treble nine"]), "112": "one one two",
             "404": "four oh four", "311": "three one one", "411": "four one one", "211": "two one one", "101": "one oh one",
             "505": "five oh five", "707": "seven oh seven", "808": "eight oh eight", "303": "three oh three", "606": "six oh six"}
    w = words[k]
    return _p(w, [("code", k, w)])


def en_nought(k, rng):                       # k = "0.5" / "0.25"
    frac = k.split(".")[1]
    lead = rng.choice(["nought", "zero", "oh", ""])
    f = " ".join("oh" if ch == "0" else W.DIGITS["en"][int(ch)] for ch in frac)
    lab = (lead + " " if lead else "") + "point " + f
    return _p(lab, [("dec", k, lab)])


def en_hundred_and(k, rng):
    n = int(k)
    h, r = divmod(n, 100)
    base = _c(h * 100, "en")
    head = rng.choice(["a hundred", base]) if h == 1 else base
    if h == 1 and head == "a hundred":
        pass
    tail = _c(r, "en")
    lab = f"{head} {rng.choice(['and ', ''])}{tail}"
    return _p(lab, [("int", k, lab)])


def en_version(k, rng):
    parts = k.split(".")
    head = rng.choice(["version", "version", "release", "v", "iOS", "Python", "Android"])
    def part(i, s):
        n = int(s)
        if n == 0 and i > 0:
            return rng.choice(["oh", "zero"])
        return _c(n, "en")
    body = " point ".join(part(i, s) for i, s in enumerate(parts))
    lab = f"{head} {body}"
    p = _p(lab, [("version", k, body)])
    return p


REGNAL = {"en": {"Henry": [2, 3, 4, 5, 6, 7, 8], "Louis": [9, 11, 14, 15, 16], "Elizabeth": [1, 2], "George": [3, 4, 5, 6],
                 "Charles": [1, 2, 3, 5], "Richard": [2, 3], "Edward": [1, 3, 7, 8], "William": [1, 3, 4], "James": [1, 2, 6],
                 "Pius": [9, 12], "John Paul": [2], "Napoleon": [3], "Philip": [2], "Alexander": [2, 3], "Catherine": [2],
                 "Benedict": [16], "Frederick": [2], "Peter": [1], "Ivan": [4], "Victor Emmanuel": [2]}}


def regnal_keys(lang):
    return [f"{n}:{o}" for n, os_ in REGNAL[lang].items() for o in os_]


def en_regnal(k, rng):
    name, o = k.rsplit(":", 1)
    o = int(o)
    w = W.ordinal_en(o)
    lab = f"{name} the {w}"
    return _p(lab, [("ord", roman(o), f"the {w}")])


def en_address(k, rng):
    num, letter = (k[:-1], k[-1]) if k[-1].isalpha() else (k, "")
    n = int(num)
    if n < 100:
        w = _c(n, "en")
    elif n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            w = f"{_c(h, 'en')} hundred"
        elif r < 10:
            w = rng.choice([f"{_c(h, 'en')} oh {W.DIGITS['en'][r]}", f"{_c(h, 'en')} hundred {_c(r, 'en')}"])
        else:
            w = rng.choice([f"{_c(h, 'en')} {_c(r, 'en')}", f"{_c(h, 'en')} hundred {_c(r, 'en')}", f"{_c(h, 'en')} hundred and {_c(r, 'en')}"])
    else:
        a, b = divmod(n, 100)
        w = f"{_c(a, 'en')} hundred" if b == 0 else f"{_c(a, 'en')} {W.pair(f'{b:02d}', 'en', 'oh')}"
    lab = w + (f" {letter}" if letter else "")
    return _p(lab, [("alnum" if letter else "int", k, lab)])


def en_height(k, rng):                       # k = "5-10"
    f, i = map(int, k.split("-"))
    form = rng.choice(["foot", "feet", "short", "digits"])
    if form == "foot":
        lab = f"{_c(f, 'en')} foot {_c(i, 'en')}" if i else f"{_c(f, 'en')} foot"
    elif form == "feet":
        lab = f"{_c(f, 'en')} feet {_c(i, 'en')} inches" if i != 1 else f"{_c(f, 'en')} feet one inch"
    elif form == "short":
        lab = f"{_c(f, 'en')} {_c(i, 'en')}" if i else f"{_c(f, 'en')} foot"
    else:
        lab = f"{_c(f, 'en')} {_c(i, 'en')}" if i else f"{_c(f, 'en')} foot"
    return _p(lab, [("unit", f"{f}'{i}\"", lab)])


def en_bp(k, rng):
    s, d = map(int, k.split("-"))
    hi = rng.choice([f"one {_c(s - 100, 'en')}" if s - 100 >= 10 else f"one oh {W.DIGITS['en'][s - 100]}", _c(s, "en")]) if s < 200 else _c(s, "en")
    lab = f"{hi} over {_c(d, 'en')}"
    return _p(lab, [("int", str(s), hi), ("int", str(d), _c(d, "en"))])


def en_temp(k, rng):
    if "." in k:
        a, b = k.split(".")
        lab = f"{_c(int(a), 'en')} point {W.DIGITS['en'][int(b)]}"
        return _p(lab, [("dec", k, lab)])
    n = int(k)
    neg = rng.choice(["minus", "negative"]) if n < 0 else ""
    if n < 0 and rng.random() < 0.3:
        lab = f"{_c(-n, 'en')} below zero"
    else:
        lab = (neg + " " if neg else "") + _c(abs(n), "en")
    return _p(lab, [("int", k, lab)])


FRACS = {"1/2": ["a half", "half"], "1/4": ["a quarter"], "3/4": ["three quarters"], "2/3": ["two thirds"], "1/3": ["a third"],
         "5/8": ["five eighths"], "7/8": ["seven eighths"], "1/8": ["an eighth"]}


def en_frac(k, rng):                         # k = "2+1/2" or "0+3/4"
    w, f = k.split("+")
    fw = rng.choice(FRACS[f])
    if w == "0":
        lab = fw.replace("a half", "half") if rng.random() < 0.5 else fw
    else:
        lab = f"{_c(int(w), 'en')} and {FRACS[f][0]}"
    return _p(lab, [("frac", k.replace("+", " ") if w != "0" else f, lab)])


def en_power(k, rng):                        # k = "10^6"
    b, e = k.split("^")
    base = "x" if b == "x" else _c(int(b), "en")
    if e == "2" and rng.random() < 0.6:
        lab = f"{base} squared"
    elif e == "3" and rng.random() < 0.6:
        lab = f"{base} cubed"
    elif e.startswith("-"):
        lab = f"{base} to the minus {_c(int(e[1:]), 'en')}"
    else:
        form = rng.choice(["the {o}", "the power of {c}"])
        lab = f"{base} to " + form.format(o=W.ordinal_en(int(e)), c=_c(int(e), "en"))
    return _p(lab, [("math", k, lab)])


def en_ratio(k, rng):
    a, b = k.split(":")
    lab = f"{_c(int(a), 'en')} to {_c(int(b), 'en')}"
    return _p(lab, [("ratio", k, lab)])


def en_clock_rel(k, rng):                    # k = "7:30"
    h, m = map(int, k.split(":"))
    hn = lambda x: _c((x - 1) % 12 + 1, "en")  # noqa: E731
    if m == 30:
        lab = rng.choice([f"half {hn(h)}", f"half past {hn(h)}", f"half {hn(h)}"])
    elif m == 15:
        lab = f"quarter past {hn(h)}"
    elif m == 45:
        lab = f"quarter to {hn(h + 1)}"
    elif m < 30:
        lab = rng.choice([f"{_c(m, 'en')} past {hn(h)}", f"{_c(m, 'en')} past {hn(h)}"]) if m != 25 else rng.choice([f"twenty five past {hn(h)}", f"five and twenty past {hn(h)}"])
    else:
        lab = f"{_c(60 - m, 'en')} to {hn(h + 1)}" if 60 - m != 25 else rng.choice([f"twenty five to {hn(h + 1)}", f"five and twenty to {hn(h + 1)}"])
    return _p(lab, [("clock", k, lab)], AM_HALF if m == 30 else ())


def en_clock_digital(k, rng):
    h, m = map(int, k.split(":"))
    hw = _c((h - 1) % 12 + 1, "en")
    if m == 0:
        lab = f"{hw} o'clock" if rng.random() < 0.6 else f"{hw} sharp"
    elif m < 10:
        lab = f"{hw} oh {W.DIGITS['en'][m]}"
    else:
        lab = f"{hw} {_c(m, 'en')}"
    return _p(lab, [("clock", k, lab)])


def en_clock_mil(k, rng):
    h, m = map(int, k.split(":"))
    hw = W.pair(f"{h:02d}", "en", "oh")
    if m == 0:
        lab = f"{hw} hundred hours"
    elif m < 10:
        lab = f"{hw} oh {W.DIGITS['en'][m]}" + (" hours" if rng.random() < 0.5 else "")
    else:
        lab = f"{hw} {_c(m, 'en')}" + (" hours" if rng.random() < 0.5 else "")
    return _p(lab, [("clock", k, lab)])


def en_day_ord(d):
    return W.ordinal_en(d)


def en_date_dmy(k, rng):                     # k = "12-3-2025"
    d, m, y = map(int, k.split("-"))
    yy = en_year(y, rng)
    lab = rng.choice(["the ", ""]) + f"{en_day_ord(d)} of {W.MONTHS['en'][m - 1]} {yy}"
    pre = lab[:4] if lab.startswith("the ") else ""
    return _p(lab, [("date", f"{d} {W.MONTHS['en'][m - 1]} {y}", lab[len(pre):])], AM_DATE)


def en_date_mdy(k, rng):
    d, m, y = map(int, k.split("-"))
    yy = en_year(y, rng)
    lab = f"{W.MONTHS['en'][m - 1]} {rng.choice([en_day_ord(d), _c(d, 'en')])} {yy}"
    return _p(lab, [("date", f"{W.MONTHS['en'][m - 1]} {d}, {y}", lab)], AM_DATE)


def en_date_numeric(k, rng):
    d, m, y = map(int, k.split("-"))
    f = lambda x: W.pair(f"{x:02d}", "en", "oh") if x < 10 and rng.random() < 0.5 else _c(x, "en")  # noqa: E731
    order = rng.choice(["dmy", "mdy"])
    a, b = (d, m) if order == "dmy" else (m, d)
    lab = f"{f(a)} {f(b)} {_c(y % 100, 'en') if y % 100 >= 10 else 'oh ' + W.DIGITS['en'][y % 100]}"
    return _p(lab, [("date", f"{a}/{b}/{y % 100:02d}", lab)], AM_DATE)


def en_date_iso(k, rng):
    d, m, y = map(int, k.split("-"))
    yy = en_year(y, random.Random(0)) if y >= 2010 else en_year(y, rng)
    if y >= 2010:
        a, b = divmod(y, 100)
        yy = f"{_c(a, 'en')} {W.pair(f'{b:02d}', 'en', 'oh')}"
    lab = f"{yy} {W.pair(f'{m:02d}', 'en', 'oh')} {W.pair(f'{d:02d}', 'en', 'oh')}"
    return _p(lab, [("date", f"{y}-{m:02d}-{d:02d}", lab)], AM_DATE)


def en_range_to(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"{_c(a, 'en')} to {_c(b, 'en')}"
    return _p(lab, [("range", f"{a}-{b}", lab)])


def en_range_fast(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"{_c(a, 'en')} {_c(b, 'en')}"
    return _p(lab, [("range", f"{a}-{b}", lab)], AM_RANGE)


# ================================================================================================ SPANISH
ES_REG = {"Felipe": [2, 4, 5, 6], "Alfonso": [10, 12, 13], "Carlos": [1, 3, 4, 5], "Fernando": [3, 7], "Isabel": [1, 2], "Juan Carlos": [1],
          "Pío": [12], "Luis": [14, 16], "Juan Pablo": [2], "Felipe": [2, 4, 5, 6], "Enrique": [4, 8], "Pedro": [1], "Napoleón": [3]}
ES_ORD = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno", 10: "décimo",
          12: "doce", 13: "trece", 14: "catorce", 16: "dieciséis"}


def es_regnal(k, rng):
    name, o = k.rsplit(":", 1)
    o = int(o)
    fem = name in ("Isabel",)
    w = ES_ORD[o] if o <= 10 else ES_ORD[o]
    if fem and o <= 10:
        w = w[:-1] + "a"
    if o > 10 and name != "Alfonso":
        w = ES_ORD[o]
    lab = f"{name} {w}"
    return _p(lab, [("ord", roman(o), w)])


def es_version(k, rng):
    parts = k.split(".")
    head = rng.choice(["la versión", "la versión", "la versión", "iOS", "Python", "Android"])
    body = " punto ".join(("cero" if (int(s) == 0 and i > 0) else _c(int(s), "es")) for i, s in enumerate(parts))
    return _p(f"{head} {body}", [("version", k, body)])


def es_address(k, rng):
    num, letter = (k[:-1], k[-1]) if k[-1].isalpha() else (k, "")
    n = int(num)
    w = _c(n, "es") if n != 1 else "uno"
    if 100 <= n < 1000 and n % 100 == 0:
        w = _c(n, "es")
    lab = w + (f" {letter}" if letter else "")
    return _p(lab, [("alnum" if letter else "int", k, lab)])


def es_height(k, rng):                       # "1-80" -> "uno ochenta"
    m, cm = map(int, k.split("-"))
    lab = f"{_c(m, 'es')} {_c(cm, 'es')}" if rng.random() < 0.7 else f"{_c(m, 'es')} metro {_c(cm, 'es')}"
    if lab.startswith("uno metro"):
        lab = lab.replace("uno metro", "un metro")
    return _p(lab, [("unit", f"{m},{cm:02d} m", lab)])


def es_fever(k, rng):                        # "37.5" -> "treinta y siete cinco" / "treinta y siete con cinco"
    a, b = k.split(".")
    lab = f"{_c(int(a), 'es')} {rng.choice(['', 'coma ', 'con '])}{W.DIGITS['es'][int(b)]}"
    return _p(lab, [("dec", k, lab)])


ES_FR = {"1/2": "medio", "3/4": "tres cuartos", "2/3": "dos tercios", "1/4": "un cuarto", "1/3": "un tercio"}


def es_frac(k, rng):
    w, f = k.split("+")
    if w == "0":
        lab = ES_FR[f]
    else:
        lab = f"{'uno' if int(w) == 1 else _c(int(w), 'es')} y medio" if f == "1/2" else f"{_c(int(w), 'es')} y {ES_FR[f]}"
        lab = lab.replace("uno y", "uno y")
    return _p(lab, [("frac", k.replace("+", " ") if w != "0" else f, lab)])


def es_bp(k, rng):
    s, d = map(int, k.split("-"))
    lab = f"{_c(s, 'es')} sobre {_c(d, 'es')}"
    return _p(lab, [("int", str(s), _c(s, "es")), ("int", str(d), _c(d, "es"))])


def _es_hora(h):
    return "la una" if h % 12 in (1,) else f"las {_c((h - 1) % 12 + 1, 'es')}"


def es_clock_rel(k, rng):
    h, m = map(int, k.split(":"))
    hn = _es_hora(h)
    if m == 0:
        lab = f"a {hn} en punto"
    elif m == 30:
        lab = f"a {hn} y media"
    elif m == 15:
        lab = f"a {hn} y cuarto"
    elif m == 45:
        lab = f"a {_es_hora(h + 1)} menos cuarto"
    elif m < 30:
        lab = f"a {hn} y {_c(m, 'es')}"
    else:
        lab = f"a {_es_hora(h + 1)} menos {_c(60 - m, 'es')}"
    lab = lab.replace("a las una", "a la una")
    return _p(lab, [("clock", k, lab)])


def es_clock_lat(k, rng):
    h, m = map(int, k.split(":"))
    nx = _es_hora(h + 1)
    if m == 45:
        lab = f"a cuarto para {nx}"
    elif m > 30:
        lab = f"a {_c(60 - m, 'es')} para {nx}" if rng.random() < 0.5 else f"a faltan {_c(60 - m, 'es')} para {nx}"
        lab = lab.replace("a faltan", "cuando faltan")
    else:
        lab = f"a las {_c(h, 'es')} {_c(m, 'es') if m >= 10 else 'cero ' + W.DIGITS['es'][m]}" if m else f"a las {_c(h, 'es')} horas"
    return _p(lab, [("clock", k, lab)])


def es_date_words(k, rng):
    d, m, y = map(int, k.split("-"))
    dd = "primero" if d == 1 else _c(d, "es")
    lab = f"el {dd} de {W.MONTHS['es'][m - 1]} de {_c(y, 'es')}"
    return _p(lab, [("date", f"{d} de {W.MONTHS['es'][m - 1]} de {y}", lab[3:])], AM_DATE)


def es_range_to(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"de {_c(a, 'es')} a {_c(b, 'es')}"
    return _p(lab, [("range", f"{a}-{b}", lab)])


def es_range_fast(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"{_c(a, 'es')} {_c(b, 'es')}"
    return _p(lab, [("range", f"{a}-{b}", lab)], AM_RANGE)


# ================================================================================================ FRENCH
FR_REG = {"Louis": [9, 14, 16, 18], "Henri": [2, 3, 4], "François": [1, 2], "Napoléon": [1, 3], "Charles": [5, 7, 10], "Philippe": [2, 4],
          "Jean Paul": [2], "Pie": [9, 12], "Jean": [2, 23], "Benoît": [16]}
FR_ORD = {1: "premier", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 7: "sept", 9: "neuf", 10: "dix", 12: "douze", 14: "quatorze", 16: "seize",
          18: "dix huit", 23: "vingt trois"}


def fr_regnal(k, rng):
    name, o = k.rsplit(":", 1)
    o = int(o)
    w = FR_ORD[o]
    lab = f"{name} {w}"
    return _p(lab, [("ord", roman(o), w)])


def fr_version(k, rng):
    parts = k.split(".")
    head = rng.choice(["la version", "la version", "la version", "iOS", "Python", "Android"])
    body = " point ".join(("zéro" if (int(s) == 0 and i > 0) else _c(int(s), "fr")) for i, s in enumerate(parts))
    return _p(f"{head} {body}", [("version", k, body)])


def fr_address(k, rng):
    num, letter = (k[:-1], k[-1]) if k[-1].isalpha() else (k, "")
    n = int(num)
    w = _c(n, "fr")
    lab = w + (f" {letter}" if letter else "")
    return _p(lab, [("alnum" if letter else "int", k, lab)])


def fr_height(k, rng):
    m, cm = map(int, k.split("-"))
    lab = f"un mètre {_c(cm, 'fr')}" if rng.random() < 0.7 else f"un {_c(cm, 'fr')}"
    if lab.startswith("un ") and not lab.startswith("un mètre"):
        lab = f"un mètre {_c(cm, 'fr')}"
    return _p(lab, [("unit", f"{m},{cm:02d} m", lab)])


def fr_fever(k, rng):
    a, b = k.split(".")
    lab = f"{_c(int(a), 'fr')} {rng.choice(['', 'virgule '])}{W.DIGITS['fr'][int(b)]}"
    return _p(lab, [("dec", k, lab)])


FR_FR = {"1/2": "demi", "3/4": "trois quarts", "2/3": "deux tiers", "1/4": "un quart", "1/3": "un tiers"}


def fr_frac(k, rng):
    w, f = k.split("+")
    if w == "0":
        lab = FR_FR[f]
    else:
        lab = f"{'un' if int(w) == 1 else _c(int(w), 'fr')} et demi" if f == "1/2" else f"{'un' if int(w) == 1 else _c(int(w), 'fr')} et {FR_FR[f]}"
    return _p(lab, [("frac", k.replace("+", " ") if w != "0" else f, lab)])


def fr_bp(k, rng):
    s, d = map(int, k.split("-"))
    if rng.random() < 0.5 and s % 10 == 0 and d % 10 == 0:
        lab = f"{_c(s // 10, 'fr')} {_c(d // 10, 'fr')}"            # "douze huit"
        return _p(lab, [("int", str(s // 10), _c(s // 10, "fr")), ("int", str(d // 10), _c(d // 10, "fr"))])
    lab = f"{_c(s, 'fr')} sur {_c(d, 'fr')}"
    return _p(lab, [("int", str(s), _c(s, "fr")), ("int", str(d), _c(d, "fr"))])


def _fr_h(h):
    h = h % 24
    if h == 0:
        return "minuit"
    if h == 12:
        return "midi"
    w = _c(h, "fr")
    if h == 1:
        return "une heure"
    return (w[:-2] + "une heures") if w.endswith(" un") else f"{w} heures"


def fr_clock_rel(k, rng):
    h, m = map(int, k.split(":"))
    hn = _fr_h(h % 12 or 12) if h % 12 else "midi"
    if m == 0:
        lab = f"à {hn}"
    elif m == 30:
        lab = f"à {hn} et demie" if hn not in ("midi", "minuit") else f"à {hn} et demi"
    elif m == 15:
        lab = f"à {hn} et quart"
    elif m == 45:
        lab = f"à {_fr_h((h + 1) % 12 or 12)} moins le quart"
    elif m < 30:
        lab = f"à {hn} {_c(m, 'fr')}"
    else:
        lab = f"à {_fr_h((h + 1) % 12 or 12)} moins {_c(60 - m, 'fr')}"
    return _p(lab, [("clock", k, lab)])


def fr_clock_24(k, rng):
    h, m = map(int, k.split(":"))
    lab = f"à {_c(h, 'fr')} heures" + (f" {_c(m, 'fr') if m >= 10 else 'zéro ' + W.DIGITS['fr'][m]}" if m else "")
    lab = lab.replace(" un heures", " une heures")
    return _p(lab, [("clock", k, lab)])


def fr_date_words(k, rng):
    d, m, y = map(int, k.split("-"))
    dd = "premier" if d == 1 else _c(d, "fr")
    lab = f"le {dd} {W.MONTHS['fr'][m - 1]} {_c(y, 'fr')}"
    return _p(lab, [("date", f"{'1er' if d == 1 else d} {W.MONTHS['fr'][m - 1]} {y}", lab[3:])], AM_DATE)


def fr_range_to(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"de {_c(a, 'fr')} à {_c(b, 'fr')}"
    return _p(lab, [("range", f"{a}-{b}", lab)])


def fr_range_fast(k, rng):
    a, b = map(int, k.split("-"))
    lab = f"{_c(a, 'fr')} {_c(b, 'fr')}"
    return _p(lab, [("range", f"{a}-{b}", lab)], AM_RANGE)


# ================================================================================================ value universes
def _clock_keys(minutes):
    return [f"{h}:{m:02d}" for h in range(1, 13) for m in minutes]


def date_keys(n=90, seed=7, years=(2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 1999, 2005)):
    rng = random.Random(f"advB|dates|{seed}")
    out = set()
    while len(out) < n:
        out.add(f"{rng.randint(1, 28)}-{rng.randint(1, 12)}-{rng.choice(years)}")
    return sorted(out)


def versions():
    return ["3.0", "2.4.1", "4.0.2", "17.2", "3.11", "10.15.7", "1.2.3", "5.0", "12.4.1", "2.0", "8.1", "6.0.1", "14.7", "9.3.2", "4.2", "3.9.1",
            "11.2", "7.0.4", "1.0", "16.5", "13.1", "5.1.2", "2.7.18", "15.0", "10.0.1", "18.4", "4.1.3", "6.5"]


def addresses():
    out = [str(n) for n in (7, 12, 15, 18, 22, 29, 33, 38, 44, 47, 56, 62, 71, 85, 94, 101, 108, 117, 124, 150, 205, 221, 238, 304, 322, 455, 502,
                            517, 640, 725, 833, 905, 1020, 1105, 1260, 1450, 1560, 1604, 1755, 2020, 2200, 2310, 3004, 3500)]
    out += [f"{n}{l}" for n, l in [(22, "B"), (221, "B"), (17, "A"), (12, "C"), (9, "B"), (45, "D"), (30, "A"), (8, "C"), (64, "B"), (150, "A"),
                                   (33, "E"), (76, "C"), (14, "A"), (28, "B"), (91, "D"), (5, "A"), (118, "C"), (240, "B")]]
    return out


def ranges_en():
    return [f"{a}-{b}" for a, b in [(20, 30), (5, 10), (10, 15), (15, 20), (30, 40), (40, 50), (50, 60), (2, 3), (3, 4), (5, 6), (10, 12), (25, 30),
                                    (60, 70), (100, 150), (8, 10), (12, 15), (35, 45), (70, 80), (90, 100), (6, 8), (200, 300), (4, 5), (18, 20),
                                    (45, 55), (80, 90), (11, 13)]]
