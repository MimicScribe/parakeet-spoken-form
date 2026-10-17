"""Case expansion for approach A of ml_multi (2026-10-16): every candidate reading of a caption number is re-inflected
from the language's own paradigm tables, so the spoken form is among the candidates even when the LLM did not propose
it (pl "w 2013 roku" said "w dwa tysiące trzynastym roku": the LLM offered only "trzynasty" / "trzynastego").

Polish and Romanian have curated paradigms here (cardinals in every case and gender, the scale nouns agreeing with
their multiplier, ordinals in every adjectival ending); they generate the form of a VALUE, so a form the drafted
lexicon (ml_multi_lex.json) lacks or misspells is never needed. Their forms are also added to the reader
(`extra_forms`), so a generated reading always parses back to its value. Other languages fall back to single-word
swaps among the lexicon's forms of the same value.

`expand(reading, lang)` returns the variants of one reading (the reading itself first); the caller value-checks each.
"""
import re

# ------------------------------------------------------------------ Polish
PL_SLOTS = ("nom", "nom_f", "nom_n", "nom_mp", "gen", "gen_f", "dat", "dat_f", "inst", "inst_f", "loc", "loc_f", "acc_f")
_PL_SMALL = {  # value -> slot -> form (1-4 decline by gender; 5+ have three forms)
    1: dict(nom="jeden", nom_f="jedna", nom_n="jedno", nom_mp="jeden", gen="jednego", gen_f="jednej", dat="jednemu",
            dat_f="jednej", inst="jednym", inst_f="jedną", loc="jednym", loc_f="jednej", acc_f="jedną"),
    2: dict(nom="dwa", nom_f="dwie", nom_n="dwa", nom_mp="dwaj", gen="dwóch", gen_f="dwóch", dat="dwóm", dat_f="dwóm",
            inst="dwoma", inst_f="dwiema", loc="dwóch", loc_f="dwóch", acc_f="dwie"),
    3: dict(nom="trzy", nom_f="trzy", nom_n="trzy", nom_mp="trzej", gen="trzech", gen_f="trzech", dat="trzem",
            dat_f="trzem", inst="trzema", inst_f="trzema", loc="trzech", loc_f="trzech", acc_f="trzy"),
    4: dict(nom="cztery", nom_f="cztery", nom_n="cztery", nom_mp="czterej", gen="czterech", gen_f="czterech",
            dat="czterem", dat_f="czterem", inst="czterema", inst_f="czterema", loc="czterech", loc_f="czterech",
            acc_f="cztery"),
}
_PL3 = {  # value -> (nominative/accusative, genitive = dative = locative, instrumental)
    5: ("pięć", "pięciu", "pięcioma"), 6: ("sześć", "sześciu", "sześcioma"), 7: ("siedem", "siedmiu", "siedmioma"),
    8: ("osiem", "ośmiu", "ośmioma"), 9: ("dziewięć", "dziewięciu", "dziewięcioma"),
    10: ("dziesięć", "dziesięciu", "dziesięcioma"), 11: ("jedenaście", "jedenastu", "jedenastoma"),
    12: ("dwanaście", "dwunastu", "dwunastoma"), 13: ("trzynaście", "trzynastu", "trzynastoma"),
    14: ("czternaście", "czternastu", "czternastoma"), 15: ("piętnaście", "piętnastu", "piętnastoma"),
    16: ("szesnaście", "szesnastu", "szesnastoma"), 17: ("siedemnaście", "siedemnastu", "siedemnastoma"),
    18: ("osiemnaście", "osiemnastu", "osiemnastoma"), 19: ("dziewiętnaście", "dziewiętnastu", "dziewiętnastoma"),
    20: ("dwadzieścia", "dwudziestu", "dwudziestoma"), 30: ("trzydzieści", "trzydziestu", "trzydziestoma"),
    40: ("czterdzieści", "czterdziestu", "czterdziestoma"), 50: ("pięćdziesiąt", "pięćdziesięciu", "pięćdziesięcioma"),
    60: ("sześćdziesiąt", "sześćdziesięciu", "sześćdziesięcioma"),
    70: ("siedemdziesiąt", "siedemdziesięciu", "siedemdziesięcioma"),
    80: ("osiemdziesiąt", "osiemdziesięciu", "osiemdziesięcioma"),
    90: ("dziewięćdziesiąt", "dziewięćdziesięciu", "dziewięćdziesięcioma"),
    100: ("sto", "stu", "stoma"), 200: ("dwieście", "dwustu", "dwustoma"), 300: ("trzysta", "trzystu", "trzystoma"),
    400: ("czterysta", "czterystu", "czterystoma"), 500: ("pięćset", "pięciuset", "pięciuset"),
    600: ("sześćset", "sześciuset", "sześciuset"), 700: ("siedemset", "siedmiuset", "siedmiuset"),
    800: ("osiemset", "ośmiuset", "ośmiuset"), 900: ("dziewięćset", "dziewięciuset", "dziewięciuset"),
}
# scale nouns: singular nom gen dat inst loc, plural nom gen dat inst loc
_PL_SCALE = {
    1000: ("tysiąc", "tysiąca", "tysiącowi", "tysiącem", "tysiącu", "tysiące", "tysięcy", "tysiącom", "tysiącami", "tysiącach"),
    10**6: ("milion", "miliona", "milionowi", "milionem", "milionie", "miliony", "milionów", "milionom", "milionami", "milionach"),
    10**9: ("miliard", "miliarda", "miliardowi", "miliardem", "miliardzie", "miliardy", "miliardów", "miliardom", "miliardami",
            "miliardach"),
    10**12: ("bilion", "biliona", "bilionowi", "bilionem", "bilionie", "biliony", "bilionów", "bilionom", "bilionami", "bilionach"),
}
_CASE = {"nom": "nom", "nom_f": "nom", "nom_n": "nom", "acc_f": "nom", "nom_mp": "gen", "gen": "gen", "gen_f": "gen",
         "dat": "dat", "dat_f": "dat", "inst": "inst", "inst_f": "inst", "loc": "loc", "loc_f": "loc"}
_CASE_I = {"nom": 0, "gen": 1, "dat": 2, "inst": 3, "loc": 4}


def _pl_small(v: int, slot: str, compound: bool) -> str:
    if v == 1 and compound:  # "dwudziestu jeden": 1 inside a compound never declines
        return "jeden"
    if v in _PL_SMALL:
        return _PL_SMALL[v][slot]
    n, o, i = _PL3[v]
    c = _CASE[slot]
    return n if c == "nom" else i if c == "inst" else o


def _pl_lt1000(r: int, slot: str, compound: bool) -> list:
    h, t = r // 100 * 100, r % 100
    out = []
    if h:
        out.append(_pl_small(h, slot, False))
    if t:
        if t < 20 or t % 10 == 0:
            out.append(_pl_small(t, slot, compound or bool(h)))
        else:
            out += [_pl_small(t // 10 * 10, slot, False), _pl_small(t % 10, slot, True)]
    return out


def _pl_scale(m: int, scale: int, slot: str) -> str:
    f = _PL_SCALE[scale]
    c = _CASE[slot]
    if m == 1:
        return f[_CASE_I[c]]
    if c == "nom" and slot != "nom_mp":
        return f[5] if m % 10 in (2, 3, 4) and m % 100 not in (12, 13, 14) else f[6]
    return f[5 + _CASE_I[c]] if c != "nom" else f[6]


def pl_cardinal(n: int, slot: str = "nom") -> str | None:
    if n < 0 or n >= 10**15:
        return None
    if n == 0:
        return {"gen": "zera", "inst": "zerem"}.get(_CASE[slot], "zero")
    out = []
    for scale in (10**12, 10**9, 10**6, 1000):
        m = n // scale % 1000
        if m == 1:
            out.append(_pl_scale(1, scale, slot))
        elif m:
            # scale nouns are masculine inanimate; a masculine-personal count takes the genitive ("dwóch tysięcy")
            ms = {"nom_f": "nom", "nom_n": "nom", "acc_f": "nom", "nom_mp": "gen", "gen_f": "gen", "dat_f": "dat",
                  "inst_f": "inst", "loc_f": "loc"}.get(slot, slot)
            out += _pl_lt1000(m, ms, False) + [_pl_scale(m, scale, slot)]
    if n % 1000:
        out += _pl_lt1000(n % 1000, slot, bool(out))
    return " ".join(out)


PL_ENDINGS = ("y", "a", "e", "ego", "ej", "emu", "ym", "ą", "ych", "ymi")
_PL_SOFT = {"y": "i", "a": "ia", "e": "ie", "ego": "iego", "ej": "iej", "emu": "iemu", "ym": "im", "ą": "ią", "ych": "ich",
            "ymi": "imi"}
_PL_VELAR = dict(_PL_SOFT, a="a", ą="ą")
_PL_ORD = {1: "pierwsz", 2: "drug", 3: "trzec", 4: "czwart", 5: "piąt", 6: "szóst", 7: "siódm", 8: "ósm", 9: "dziewiąt",
           10: "dziesiąt", 11: "jedenast", 12: "dwunast", 13: "trzynast", 14: "czternast", 15: "piętnast", 16: "szesnast",
           17: "siedemnast", 18: "osiemnast", 19: "dziewiętnast", 20: "dwudziest", 30: "trzydziest", 40: "czterdziest",
           50: "pięćdziesiąt", 60: "sześćdziesiąt", 70: "siedemdziesiąt", 80: "osiemdziesiąt", 90: "dziewięćdziesiąt",
           100: "setn", 200: "dwusetn", 300: "trzechsetn", 400: "czterechsetn", 500: "pięćsetn", 600: "sześćsetn",
           700: "siedemsetn", 800: "osiemsetn", 900: "dziewięćsetn", 1000: "tysięczn", 2000: "dwutysięczn"}


def _pl_ord_word(v: int, e: str) -> str:
    s = _PL_ORD[v]
    return s + (_PL_SOFT[e] if v == 3 else _PL_VELAR[e] if v == 2 else e)


def pl_ordinal(n: int, ending: str = "y") -> str | None:
    """Polish ordinal: the thousands stay cardinal ("dwa tysiące trzynastym"), the hundreds stay cardinal unless the
    number ends there ("tysiąc dziewięćset osiemdziesiątym dziewiątym", "dwusetnym"), tens and units both ordinal."""
    if n <= 0 or n >= 10**6:
        return None
    th, r = n // 1000, n % 1000
    if r == 0:
        return _pl_ord_word(n, ending) if n in (1000, 2000) else None
    out = pl_cardinal(th * 1000, "nom").split() if th else []
    h, t = r // 100 * 100, r % 100
    if t == 0:
        return " ".join(out + [_pl_ord_word(h, ending)])
    if h:
        out.append(_PL3[h][0])
    if t < 20 or t % 10 == 0:
        out.append(_pl_ord_word(t, ending))
    else:
        out += [_pl_ord_word(t // 10 * 10, ending), _pl_ord_word(t % 10, ending)]
    return " ".join(out)


# ------------------------------------------------------------------ Romanian
RO_GENDERS = ("m", "f", "m_attr", "f_attr")
_RO_U = {1: {"m": "unu", "f": "una", "m_attr": "un", "f_attr": "o"}, 2: {"m": "doi", "f": "două", "m_attr": "doi", "f_attr": "două"}}
_RO_UNITS = {3: "trei", 4: "patru", 5: "cinci", 6: "șase", 7: "șapte", 8: "opt", 9: "nouă"}
_RO_TEENS = {10: "zece", 11: "unsprezece", 12: "doisprezece", 13: "treisprezece", 14: "paisprezece", 15: "cincisprezece",
             16: "șaisprezece", 17: "șaptesprezece", 18: "optsprezece", 19: "nouăsprezece"}
_RO_TENS = {20: "douăzeci", 30: "treizeci", 40: "patruzeci", 50: "cincizeci", 60: "șaizeci", 70: "șaptezeci",
            80: "optzeci", 90: "nouăzeci"}


def _ro_unit(v: int, g: str) -> str:
    return _RO_U[v][g] if v in _RO_U else _RO_UNITS[v]


def _ro_lt100(t: int, g: str, inner: bool) -> str:
    if inner:  # "o sută unu", "douăzeci și una": the attributive "un" / "o" only stand alone
        g = {"m_attr": "m", "f_attr": "f"}.get(g, g)
    if t < 10:
        return _ro_unit(t, g)
    if t < 20:
        return "douăsprezece" if t == 12 and g.startswith("f") else _RO_TEENS[t]
    return _RO_TENS[t // 10 * 10] + (f" și {_ro_unit(t % 10, {'m_attr': 'm', 'f_attr': 'f'}.get(g, g))}" if t % 10 else "")


def _ro_lt1000(r: int, g: str, inner: bool = False) -> str:
    h, t = r // 100, r % 100
    out = []
    if h:
        out.append("o sută" if h == 1 else f"{_ro_lt100(h, 'f', False)} sute")
    if t:
        out.append(_ro_lt100(t, g, inner or bool(h)))
    return " ".join(out)


def ro_cardinal(n: int, g: str = "m") -> str | None:
    if n < 0 or n >= 10**12:
        return None
    if n == 0:
        return "zero"
    out = []
    for scale, sg, pl, mg in ((10**9, "un miliard", "miliarde", "f"), (10**6, "un milion", "milioane", "f"),
                              (1000, "o mie", "mii", "f")):
        m = n // scale % 1000
        if m == 1:
            out.append(sg)
        elif m:
            de = m % 100 == 0 or m % 100 >= 20  # "douăzeci de mii", "o sută de milioane"
            out.append(f"{_ro_lt1000(m, mg)}{' de ' if de else ' '}{pl}")
    if n % 1000:
        out.append(_ro_lt1000(n % 1000, g, bool(out)))
    return " ".join(out)


_RO_ORD_M = {2: "doilea", 3: "treilea", 4: "patrulea", 5: "cincilea", 6: "șaselea", 7: "șaptelea", 8: "optulea",
             9: "nouălea", 10: "zecelea", 11: "unsprezecelea", 12: "doisprezecelea", 13: "treisprezecelea",
             14: "paisprezecelea", 15: "cincisprezecelea", 16: "șaisprezecelea", 17: "șaptesprezecelea",
             18: "optsprezecelea", 19: "nouăsprezecelea"}
_RO_ORD_F = {2: "doua", 3: "treia", 4: "patra", 5: "cincea", 6: "șasea", 7: "șaptea", 8: "opta", 9: "noua", 10: "zecea",
             11: "unsprezecea", 12: "douăsprezecea", 13: "treisprezecea", 14: "paisprezecea", 15: "cincisprezecea",
             16: "șaisprezecea", 17: "șaptesprezecea", 18: "optsprezecea", 19: "nouăsprezecea"}


def ro_ordinal(n: int, g: str = "m") -> str | None:
    """Romanian ordinal with its article ("al doilea" / "a doua"; "primul" / "prima")."""
    f = g.startswith("f")
    if n == 1:
        return "prima" if f else "primul"
    if 2 <= n <= 19:
        return f"a {_RO_ORD_F[n]}" if f else f"al {_RO_ORD_M[n]}"
    if 20 <= n <= 99:
        tens, u = _RO_TENS[n // 10 * 10], n % 10
        if not u:
            return f"a {tens[:-1]}ea" if f else f"al {tens}lea"  # "a douăzecea" / "al douăzecilea"
        if u == 1:
            return f"a {tens} și una" if f else f"al {tens} și unulea"
        return f"a {tens} și {_RO_ORD_F[u]}" if f else f"al {tens} și {_RO_ORD_M[u]}"
    if n == 100:
        return "a o suta" if f else "al o sutălea"
    if n == 1000:
        return "a o mia" if f else "al o miilea"
    return None


# ------------------------------------------------------------------ expansion
UNIT_GROUPS = {  # inflection families of the unit / currency / percent word read with a number
    "pl": [("procent", "procenta", "procentów", "procentom", "procentami", "procentach"),
           ("złoty", "złote", "złotych", "złotego", "złotemu", "złotym", "złotymi"),
           ("grosz", "grosze", "groszy", "groszach", "groszami"),
           ("dolar", "dolara", "dolary", "dolarów", "dolarach", "dolarami"),
           ("rok", "roku", "rokiem", "rokowi"),
           ("godzina", "godziny", "godzinie", "godzinę", "godzin", "godzinach", "godzinami"),
           ("minuta", "minuty", "minut", "minutach")],
    "ro": [("leu", "lei"), ("ban", "bani"), ("dolar", "dolari"), ("eurocent", "eurocenți"), ("cent", "cenți"),
           ("oră", "ore"), ("minut", "minute")],
}


def _variants(value, ordinal: bool, lang: str) -> list:
    """Every form of an integer value in this language: [(group key, reading)]; group key = slot / ending, so the
    numbers of one reading can be swept together."""
    v = int(value)
    if lang == "pl":
        if ordinal:
            return [(e, pl_ordinal(v, e)) for e in PL_ENDINGS if pl_ordinal(v, e)]
        return [(s, pl_cardinal(v, s)) for s in PL_SLOTS if pl_cardinal(v, s)]
    if lang == "ro":
        if ordinal:
            return [(g, ro_ordinal(v, g)) for g in ("m", "f") if ro_ordinal(v, g)]
        return [(g, ro_cardinal(v, g)) for g in RO_GENDERS if ro_cardinal(v, g)]
    return []


def _lex_swaps(lang: str) -> dict:
    """Fallback (no curated paradigm): (value, is_ordinal) -> the lexicon's single-word forms."""
    import ml_multi as MM

    t = MM._table().get(lang, {})
    out = {}
    for kind, o in (("cardinal", False), ("ordinal", True)):
        for v, ws in (t.get(kind) or {}).items():
            out[(int(v), o)] = sorted({w for w in ws if " " not in w and "\xa0" not in w})
    return out


MAX_NUMS = 3  # a reading with more numbers (digit-by-digit phones, lists) is not expanded


def expand(reading: str, lang: str, cap: int = 120) -> list:
    """Variants of one reading (the reading first): each integer number re-inflected alone, all numbers of a kind
    swept together (one case / gender / ending for every cardinal, one for every ordinal), and the unit word in each
    of its forms. Non-integer numbers ("trzy przecinek pięć", "półtora") and the words around the numbers are kept."""
    import ml_multi as MM

    toks = reading.split()
    nums = [n for n in MM.parse_numbers(reading, lang) if n.value == int(n.value)]
    out = [reading]

    def add(x):
        if x not in out and len(out) < cap:
            out.append(x)

    if 0 < len(nums) <= MAX_NUMS:
        alts = []  # per number: (start, end, [(key, words)])
        for n in nums:
            a, b = n.start, n.end
            if lang == "ro" and n.ordinal and a > 0 and MM._key(toks[a - 1]) in ("al", "a"):
                a -= 1  # the article is part of a Romanian ordinal ("al doilea" / "a doua")
            if lang in ("pl", "ro"):
                vs = _variants(n.value, n.ordinal, lang)
            else:
                lx = _lex_swaps(lang)
                vs = [("w", w) for w in lx.get((int(n.value), n.ordinal), [])] if a == b else []
            if vs:
                alts.append((a, b, n.ordinal, vs))

        def render(choice):  # choice: {index in alts: words}
            res, pos = [], 0
            for k, (a, b, _o, _vs) in enumerate(alts):
                if k in choice:
                    res += toks[pos:a]
                    tail = re.search(r"[.,;:!?]+$", toks[b])
                    res.append(choice[k] + (tail.group(0) if tail else ""))
                    pos = b + 1
            return " ".join(res + toks[pos:])

        for k, (_a, _b, _o, vs) in enumerate(alts):  # one number at a time
            for _key, w in vs:
                add(render({k: w}))
        for ordinal in (False, True):  # every number of a kind in one slot / ending
            idx = [k for k, x in enumerate(alts) if x[2] == ordinal]
            if len(idx) < 2:
                continue
            keys = [key for key, _ in alts[idx[0]][3]]
            for key in keys:
                ch = {}
                for k in idx:
                    w = dict(alts[k][3]).get(key)
                    if w:
                        ch[k] = w
                add(render(ch))
    for grp in UNIT_GROUPS.get(lang, []):  # the unit word in each of its forms
        fk = {MM._key(f) for f in grp}
        pos = [i for i, t in enumerate(toks) if MM._key(t) in fk]
        if pos:
            for base in list(out):
                bt = base.split()
                bpos = [i for i, t in enumerate(bt) if MM._key(t) in fk]
                for f in grp:
                    nt = list(bt)
                    for i in bpos:
                        tail = re.search(r"[.,;:!?]+$", nt[i])
                        nt[i] = f + (tail.group(0) if tail else "")
                    add(" ".join(nt))
    return out


def extra_forms(lang: str) -> dict:
    """{"cardinal": {value: [forms]}, "ordinal": {...}}: single-word forms of the curated paradigms, merged into the
    reader's lexicon so every generated reading parses back."""
    if lang not in ("pl", "ro"):
        return {}
    card, ordn = {}, {}
    vals = list(range(1, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100))
    for v in vals:
        for _k, w in _variants(v, False, lang):
            if len(w.split()) == 1:
                card.setdefault(str(v), set()).add(w)
        for _k, w in _variants(v, True, lang):
            ws = w.split()
            if lang == "ro" and ws[0] in ("al", "a"):
                ws = ws[1:]
            if len(ws) == 1:
                ordn.setdefault(str(v), set()).add(ws[0])
    if lang == "ro":  # parts of compound ordinals ("al douăzeci și unulea", "al o miilea")
        ordn.setdefault("1", set()).add("unulea")
        ordn.setdefault("1000", set()).update({"miilea", "mia"})
    if lang == "pl":
        card.setdefault("0", set()).update({"zero", "zera", "zerem"})
        for e in PL_ENDINGS:
            ordn.setdefault("1000", set()).add(_pl_ord_word(1000, e))
            ordn.setdefault("2000", set()).add(_pl_ord_word(2000, e))
        for scale, fs in _PL_SCALE.items():
            card.setdefault(str(scale), set()).update(fs)
    return {"cardinal": {k: sorted(v) for k, v in card.items()}, "ordinal": {k: sorted(v) for k, v in ordn.items()}}
