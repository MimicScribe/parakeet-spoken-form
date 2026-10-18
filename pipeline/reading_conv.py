"""Reading conventions of spoken numbers (de es fr it nl pt): which way a number was SAID (2026-10-18).

The reading-variety census and the variety fill (ml_tts_variety) name each spoken number by its convention:
phones by group shape ("pairs", "1+2_splits", "digits", "hundreds+pairs", an UNNATURAL 2+1 split...), years
("thousands", "hundreds", "pairs", "two_digit"), clock times ("digital_hourword", "half_next" = "halb drei",
"quarter_to", ...), prices with cents ("unit+cents", "bare_no_unit" = "zwei fünfzig", "bare_joiner", ...), decimals
("comma_1digit", "comma_as_number", "half_word", "con"), percent words, big counts ("hundreds" = "fünfzehnhundert")
and range joiners ("bis", "von..bis", "between..und", ...).

Number words come from the grammar table (ml_cands.g), normalised (lower case, accents stripped, spaces / hyphens
removed), so split compounds ("sieben und neunzig", "venti uno") match joined ones.
"""
import re
import unicodedata

import compound_split as CS
import ml_cands as M


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z' ]+", " ", s).replace("'", " ' ").strip()


_IT_ELIDE = re.compile(r"(vent)i(?=[uo])|(trent|quarant|cinquant|sessant|settant|ottant|novant)a(?=[uo])|(cent)o(?=ottant)")


def joined(s: str) -> str:
    """Letters only; the Italian split spelling elided again ("venti uno" -> "ventuno"), so both spellings match."""
    return _IT_ELIDE.sub(lambda m: m.group(1) or m.group(2) or m.group(3), re.sub(r"[^a-z]", "", norm(s)))


def toks(s: str) -> list[str]:
    return norm(s).split()


_CARD: dict = {}
ALT = {"de": {"zwo": 2, "ein": 1, "eine": 1}, "es": {"un": 1, "una": 1, "uno": 1}, "pt": {"uma": 1, "duas": 2, "um": 1},
       "it": {"un": 1, "una": 1}, "fr": {"une": 1}, "nl": {"een": 1}}


def cards(lang):
    """joined word string -> value for 0..999 (+ a few alternative forms)."""
    if lang not in _CARD:
        d = {}
        for n in range(1000):
            w = " ".join(M.g(lang, "card", n))
            d.setdefault(joined(w), n)
            if lang in CS.LANGS:  # the training-text split spelling ("venti uno" for "ventuno") reads the same
                d.setdefault(joined(CS.split_text(w, lang)[0]), n)
        for w, n in ALT.get(lang, {}).items():
            d.setdefault(w, n)
        _CARD[lang] = d
    return _CARD[lang]


def cw(lang, n) -> str:
    return joined(" ".join(M.g(lang, "card", n)))


def segment(lang, s: str, digits: str | None = None, maxseg=200):
    """All segmentations of joined string s into 0..999 card words; with digits, only those whose concatenated
    values spell the digit string (each chunk's digits; a chunk 'null' / 'zero' = '0')."""
    d = cards(lang)
    maxlen = max(len(k) for k in d)
    memo = {}

    def rec(i):
        if i == len(s):
            return [[]]
        if i in memo:
            return memo[i]
        out = []
        for j in range(i + 1, min(len(s), i + maxlen) + 1):
            w = s[i:j]
            if w in d:
                for rest in rec(j):
                    out.append([d[w]] + rest)
                    if len(out) > maxseg:
                        break
        memo[i] = out
        return out

    segs = rec(0)
    if digits is not None:
        segs = [sg for sg in segs if "".join(str(v) for v in sg) == digits]
    return segs


# ------------------------------------------------------------------ phone
def group_shape(lang, digits, reading):
    """Shape of one written digit group's reading: 'd' digit by digit, 'n' one number, '1+2', '2+2', 'z' + shape
    (leading zeros said one by one, then the shape of the rest), 'X' a split no speaker uses (2+1, 3+1, ...)."""
    r = norm(reading)
    for p in PLUS[lang]:
        if r.startswith(p + " ") or r == p:
            r = r[len(p):]
    s = re.sub(r"[^a-z]", "", r)
    segs = segment(lang, s, digits)
    if not segs:
        return "?"
    sg = min(segs, key=lambda x: (len(x), x))
    sizes = [len(str(v)) for v in sg]
    lead = 0
    while lead < len(sg) - 1 and sg[lead] == 0 and digits[lead] == "0":
        lead += 1
    rest = sizes[lead:]
    pre = "z" if lead else ""
    if all(z == 1 for z in sizes):
        return "d"
    if len(rest) == 1:
        return pre + "n" + str(rest[0])
    if rest == [1, 2]:
        return pre + "1+2"
    if all(z == 2 for z in rest):
        return pre + "pairs"
    if rest in ([2, 1, 1], [1, 1, 2]) and len(digits) == 4:
        return pre + "2/d"  # "cuarenta y ocho cero cero", "nove nove setenta"
    if rest in ([3, 3], [2, 3], [3, 2]) or (len(rest) >= 2 and all(z in (2, 3) for z in rest) and rest[0] != 1):
        return pre + "+".join(map(str, rest))
    return "X:" + "+".join(map(str, sizes))


def phone_groups(lang, pieces):
    """pieces: [(written group, reading)] -> (convention, flags) at number level."""
    shapes = []
    for w, rd in pieces:
        dg = re.sub(r"\D", "", w)
        if not dg:
            continue
        shapes.append((len(dg), group_shape(lang, dg, rd)))
    flags = {"shapes": " ".join(f"{n}:{sh}" for n, sh in shapes)}
    if not shapes or any(sh == "?" for _, sh in shapes):
        return "other", flags
    if any(sh.startswith("X") for _, sh in shapes):
        return "UNNATURAL_split", flags
    if all(sh == "d" for _, sh in shapes):
        return "digits", flags
    kinds = set()
    for n, sh in shapes:
        core = sh.lstrip("z")
        if sh == "d" or n == 1:
            kinds.add("d")
        elif core in ("n2", "pairs"):
            kinds.add("pairs")
        elif core.startswith("n") and core != "n2":
            kinds.add("hundreds" if core == "n3" else "thousands")
        elif core in ("1+2", "2/d"):
            kinds.add("1+2")
        else:
            kinds.add("2/3mix")
    if kinds <= {"pairs", "d"} and "pairs" in kinds:
        # mostly pairs: French / Dutch / Spanish 2-digit grouping
        return "pairs", flags
    if "hundreds" in kinds and kinds <= {"hundreds", "pairs", "d"}:
        return "hundreds+pairs" if "pairs" in kinds else "hundreds", flags
    if "1+2" in kinds and kinds <= {"1+2", "pairs", "d"}:
        return "1+2_splits", flags
    return "mixed_natural", flags


def phone(lang, written, reading):
    digits = re.sub(r"\D", "", written)
    plus = PLUS[lang]
    r = norm(reading)
    has_plus = False
    for p in plus:
        if r.startswith(p + " ") or r == p:
            r = r[len(p):]
            has_plus = True
    s = re.sub(r"[^a-z]", "", r)
    segs = segment(lang, s, digits)
    if not segs:
        return "other", {"plus": has_plus}
    sg = min(segs, key=len) if False else segs[0]
    # prefer the segmentation with the most chunks of the dominant kind is ambiguous; take the one with fewest chunks
    sg = min(segs, key=lambda x: (len(x), x))
    sizes = [len(str(v)) for v in sg]
    nd = sum(sizes)
    flags = {"plus": has_plus, "zwo": "zwo" in toks(reading), "n_digits": nd}
    if all(z == 1 for z in sizes):
        return "digits", flags
    if any(z == 3 for z in sizes):
        return "with_hundreds", flags
    pair_digits = sum(z for z in sizes if z == 2)
    if pair_digits / nd >= 0.6:
        return "pairs", flags
    return "mixed_1_2", flags


PLUS = {"de": ["plus"], "es": ["mas"], "fr": ["plus"], "it": ["piu"], "nl": ["plus"], "pt": ["mais"]}


# ------------------------------------------------------------------ year
HUND = {"de": ["hundert", "hundertund"], "nl": ["honderd", "honderden"], "fr": ["cent", "cents"], "es": [], "it": [],
        "pt": []}


def year_forms(lang, v):
    hi, lo = divmod(v, 100)
    f = {}
    t = cw(lang, v) if v < 1000 else joined(" ".join(M.g(lang, "card", v)))
    f.setdefault(t, "thousands")
    if lang == "de" and t.startswith("eintausend"):
        f.setdefault(t[3:], "thousands")
    if lang == "nl" and t.startswith("duizend"):
        f.setdefault("een" + t, "thousands")
    if lang == "de" and v >= 2001 and lo:
        f.setdefault(cw(lang, 2000) + "und" + cw(lang, lo), "thousands_und")
    if lang == "nl" and v >= 2001 and lo:
        f.setdefault(cw(lang, 2000) + "en" + cw(lang, lo), "thousands_und")
    for h in HUND[lang]:
        f.setdefault(cw(lang, hi) + h + (cw(lang, lo) if lo else ""), "hundreds")
    if lo >= 10:
        f.setdefault(cw(lang, hi) + cw(lang, lo), "pairs")
        f.setdefault(cw(lang, lo), "two_digit")
    elif lo > 0:
        f.setdefault(cw(lang, hi) + cw(lang, 0) + cw(lang, lo), "pairs_zero")
    else:
        f.setdefault(cw(lang, hi) + cw(lang, 0) + cw(lang, 0), "pairs_zero")
    try:
        y = joined(" ".join(M.g(lang, "year", v)))
        f.setdefault(y, "table_year")
    except Exception:
        pass
    return f


def year(lang, v, reading):
    j = joined(reading)
    f = year_forms(lang, v)
    if j in f:
        return f[j], {}
    # prefixes like "im Jahr" are outside the span; a trailing period or a preposition fused: try stripping
    return "other", {}


# ------------------------------------------------------------------ int_big (1000-9999 non-years mostly)
def int_big(lang, v, reading):
    j = joined(reading)
    if 1000 <= v <= 9999:
        hi, lo = divmod(v, 100)
        if v >= 1100 and hi % 10 != 0:
            for h in HUND[lang] or []:
                if j == cw(lang, hi) + h + (cw(lang, lo) if lo else ""):
                    return "hundreds", {}
        if lang == "de" and v < 2000 and j.startswith("tausend"):
            return "thousands_no_ein", {}
        if lang == "de" and v < 2000 and j.startswith("eintausend"):
            return "thousands_ein", {}
        if lang == "nl" and v < 2000 and j.startswith("eenduizend"):
            return "thousands_een", {}
    segs = segment(lang, j, re.sub(r"\D", "", str(v)))
    if segs and min(len(s) for s in segs) >= 2:
        return "chunked", {}
    return "full", {}


# ------------------------------------------------------------------ decimal
COMMA_W = {"komma", "coma", "virgule", "virgola", "virgula", "comma"}
POINT_W = {"punkt", "punto", "point", "punt", "ponto"}
HALF_RE = re.compile(r"(einhalb|anderthalb|enhalf|anderhalf|\b(y|e|et) (medio|media|demi|demie|mezzo|mezza|meio|meia)\b)")


def dec(lang, written, reading):
    t = toks(reading)
    jr = " ".join(t)
    if HALF_RE.search(jr):
        return "half_word", {}
    sep = next((w for w in t if w in COMMA_W | POINT_W), None)
    m = re.search(r"[.,](\d+)\s*$", written.strip().rstrip("."))
    frac = m.group(1) if m else ""
    if sep is None:
        if "con" in t:
            return "con", {}
        if re.fullmatch(r"\s*\d{1,2}\.\d{2}\s*", written) and (" e " in f" {jr} " or len(t) <= 4):
            return "no_sep(clock/bare)", {}
        return "no_sep(bare)", {}
    base = ("it_legal_comma" if sep == "comma" else "comma") if sep in COMMA_W else "point"
    after = t[t.index(sep) + 1:]
    style = "_1digit" if len(frac) == 1 else ""
    if len(frac) >= 2:
        segs = segment(lang, "".join(after), frac)
        if segs:
            sg = min(segs, key=len)
            style = "_digitwise" if all(v < 10 for v in sg) and len(sg) == len(frac) else "_as_number" if len(sg) == 1 else "_chunked"
        else:
            style = "_?"
    return base + style, {}


# ------------------------------------------------------------------ percent
PCT = {"de": [("prozent", "Prozent"), ("vomhundert", "vom Hundert"), ("prozentpunkt", "Prozentpunkt")],
       "es": [("porciento", "por ciento"), ("porcien", "por cien")],
       "fr": [("pourcent", "pour cent"), ("pourmille", "pour mille")],
       "it": [("percento", "per cento"), ("percent", "per cent")],
       "nl": [("procent", "procent"), ("percent", "percent"), ("procentpunt", "procentpunt")],
       "pt": [("porcento", "por cento"), ("porcem", "por cem")]}


def pct(lang, written, reading, after):
    j = joined(reading + " " + " ".join(toks(after)[:3]))
    word = "no_word"
    for k, name in sorted(PCT[lang], key=lambda x: -len(x[0])):
        if k in j:
            word = name
            break
    t = toks(reading)
    d = ""
    if any(w in COMMA_W for w in t):
        d = "+comma"
    elif any(w in POINT_W for w in t):
        d = "+point"
    elif HALF_RE.search(" ".join(t)):
        d = "+half"
    return word + d, {}


# ------------------------------------------------------------------ currency
UNIT_W = {"euro", "euros", "dollar", "dollars", "dolar", "dolares", "dollaro", "dollari", "pfund", "libra", "libras",
          "livre", "livres", "sterlina", "sterline", "pond", "real", "reais", "peso", "pesos", "franken", "francs", "franc",
          "franchi", "franco", "yen", "yuan", "kronen", "kronor", "zloty", "rubel", "rublos", "bitcoin", "bitcoins"}
CENT_W = {"cent", "cents", "centime", "centimes", "centimo", "centimos", "centesimo", "centesimi", "centavo",
          "centavos", "rappen", "eurocent"}
SCALE_W = {"million", "millionen", "milliarde", "milliarden", "millon", "millones", "millions", "milliard", "milliards",
           "milione", "milioni", "miliardo", "miliardi", "miljoen", "miljoenen", "miljard", "miljarden", "milhao", "milhoes",
           "bilhao", "bilhoes", "billon", "billones", "mld", "mrd", "mio"}
JOIN_W = {"und", "con", "y", "et", "e", "en"}


def cur(lang, written, reading, after, is_cur_kind):
    t = toks(reading)
    a = toks(after)
    if not is_cur_kind:  # int span followed by a currency word ("treinta euros"): look at the next words
        t = t + a[:4]
    m = re.search(r"(\d)[.,](\d{1,2})(?!\d)", written)
    has_frac = bool(m) and m.group(2).strip("0") != "" and not re.search(r"\d[.,]\d{3}", written[m.start():]) if m else False
    if any(w in SCALE_W for w in t):
        digits = re.sub(r"\D", "", re.sub(r"[.,]0+(?!\d)", "", written))
        round_ = any(w in SCALE_W for w in toks(written)) or len(digits.rstrip("0")) <= 3
        return ("scale_round" if round_ else "full_figure_with_million"), {}
    if not has_frac:
        if any(w in UNIT_W for w in t):
            return "whole_unit", {}
        if any(w in CENT_W for w in t):
            return "cents_only", {}
        return "whole_no_unit", {}
    unit_i = next((i for i, w in enumerate(t) if w in UNIT_W), None)
    cent = any(w in CENT_W for w in t)
    decw = any(w in COMMA_W | POINT_W for w in t)
    if decw:
        return "decimal_word" + ("+unit" if unit_i is not None else ""), {}
    if unit_i is None:
        if cent:
            return "cents_only", {}
        if any(w in JOIN_W for w in t):
            return "bare_joiner", {}
        return "bare_no_unit", {}
    rest = t[unit_i + 1:]
    j = bool(rest) and rest[0] in JOIN_W
    if cent:
        return "unit" + ("+join" if j else "") + "+cents+centword", {}
    if rest:
        return "unit" + ("+join" if j else "") + "+cents", {}
    return "unit_only(cents dropped?)", {}


# ------------------------------------------------------------------ clock
REL = {
    "de": [(r"\b(vor|nach) halb\b", "near_half"), (r"\bhalb\b(?!.*\b(nach|vor)\b)", "half_next"),
           (r"\bviertel nach\b|^viertel\b", "quarter_past"), (r"\bdreiviertel\b|\bviertel vor\b", "quarter_to"),
           (r"\bnach\b", "min_past"), (r"\bvor\b", "min_to")],
    "nl": [(r"\b(voor|over|na) half\b", "near_half"), (r"\bhalf\b", "half_next"), (r"\bkwart (over|na)\b", "quarter_past"),
           (r"\bkwart voor\b", "quarter_to"), (r"\b(over|na)\b", "min_past"), (r"\bvoor\b", "min_to")],
    "es": [(r"\bmediodia\b|\bmedianoche\b", "noon_midnight"), (r"\by media\b", "half_past"), (r"\by cuarto\b", "quarter_past"),
           (r"\bmenos cuarto\b|\bcuarto para\b|\bquince para\b", "quarter_to"), (r"\bmenos\b|\bpara la", "min_to")],
    "fr": [(r"\bmidi\b|\bminuit\b", "noon_midnight"), (r"\bet demie?\b", "half_past"), (r"\bet quart\b", "quarter_past"),
           (r"\bmoins (le )?quart\b", "quarter_to"), (r"\bmoins\b", "min_to")],
    "it": [(r"\bmezzogiorno\b|\bmezzanotte\b", "noon_midnight"), (r"\be mezz[ao]\b", "half_past"),
           (r"\be (un )?quarto\b", "quarter_past"), (r"\bmeno (un )?quarto\b|\btre quarti\b", "quarter_to"),
           (r"\bmeno\b", "min_to")],
    "pt": [(r"\bmeio dia\b|\bmeia noite\b", "noon_midnight"), (r"\be meia\b", "half_past"), (r"\be (um )?quarto\b", "quarter_past"),
           (r"\bmenos (um )?quarto\b|\bquarto para\b|\bquinze para\b", "quarter_to"), (r"\bmenos\b|\bpara as?\b", "min_to")],
}
HOURW = {"de": {"uhr"}, "nl": {"uur", "u"}, "fr": {"heure", "heures", "h"}, "es": {"horas", "hora"}, "it": {"ore", "ora"},
         "pt": {"horas", "hora", "h"}}
DAYP = {"de": ["morgens", "vormittags", "nachmittags", "abends", "nachts", "fruh", "mittags"],
        "nl": ["s ochtends", "s morgens", "s middags", "s avonds", "s nachts"],
        "fr": ["du matin", "de l ' apres midi", "de l apres midi", "du soir", "de la nuit"],
        "es": ["de la manana", "de la madrugada", "de la tarde", "de la noche", "a eme", "pe eme", "del mediodia"],
        "it": ["di mattina", "del mattino", "del pomeriggio", "di sera", "di notte"],
        "pt": ["da manha", "da madrugada", "da tarde", "da noite"]}
CJOIN = {"es": "y", "it": "e", "pt": "e"}


def clock(lang, written, reading, after):
    t = toks(reading)
    jr = " ".join(t)
    a = " ".join(toks(after)[:4])
    flags = {}
    if any(f" {d} " in f" {jr} {a} " for d in DAYP[lang]):
        flags["daypart"] = True
    for pat, name in REL[lang]:
        if re.search(pat, jr):
            return name, flags
    m = re.search(r"(\d{1,2})\s*[:h.]\s*(\d{2})", written)
    h = mm = None
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"\d{1,2}", written)
        if m2:
            h, mm = int(m2.group(0)), 0
    hw = any(w in HOURW[lang] for w in t) or (a.split()[:1] and a.split()[0] in HOURW[lang])
    # hour number said: parse the leading number words
    first = []
    for w in t:
        if w in HOURW[lang] or w == CJOIN.get(lang):
            break
        first.append(w)
    hv = None
    segs = segment(lang, "".join(first)) if first else []
    if segs:
        hv = min(segs, key=len)[0]
    if h is not None and hv is not None and h > 12 and hv == h - 12:
        flags["12h_of_24h"] = True
    if mm == 0 and len([w for w in t if w not in HOURW[lang]]) <= len(first):
        return "oclock" + ("_hourword" if hw else ""), flags
    if lang in CJOIN and CJOIN[lang] in t:
        return "digital_joiner" + ("_hourword" if hw else ""), flags
    if hw:
        return "digital_hourword", flags
    return "digital_bare", flags


# ------------------------------------------------------------------ ordinal
def ordinal(lang, reading):
    t = toks(reading)
    j = "".join(t)
    if segment(lang, j):
        return "cardinal", {}
    if lang in ("es", "it", "pt"):
        last = t[-1] if t else ""
        if last in ("primer", "tercer"):
            return "ord_apocope", {}
        return ("ord_f" if last.endswith("a") or last.endswith("as") else "ord_m"), {}
    if lang == "fr":
        return ("premier/premiere" if j.startswith("premi") else "ord_ieme"), {}
    if lang == "de":
        mm = re.search(r"(te|ten|ter|tes|tem)$", j)
        return ("ord_-" + mm.group(1)) if mm else "ord_other", {}
    if lang == "nl":
        return "ord", {}
    return "ord", {}


# ------------------------------------------------------------------ date (merged pieces)
MONTH = {l: set(joined(w) for w in M.MONTHS[l].split()) | ({"janner", "feber"} if l == "de" else set()) for l in M.LANGS}
FIRST = {"es": ("primero", "uno", "un"), "pt": ("primeiro", "um"), "it": ("primo", "uno"), "fr": ("premier", "un"),
         "de": ("ersten", "erste", "erster", "eins"), "nl": ("eerste", "een")}


def date(lang, written, reading, after):
    t = toks(reading)
    a = toks(after)[:3]
    names = [w for w in t + a if w in MONTH[lang]]
    month = "month_name" if names else "month_number?"
    if not names:
        # month said as a number ("dritter zehnter", "tres del diez")
        if re.search(r"\d{1,2}\.\d{1,2}\.", written) or re.search(r"\d{1,2}[/-]\d{1,2}", written):
            month = "month_number"
        else:
            month = "day_only"
    head = []
    for w in t:
        if w in MONTH[lang] or w in ("de", "del", "do", "da", "di"):
            if head:
                break
            continue
        head.append(w)
    day_tok = " ".join(head)
    j = joined(day_tok)
    if head:
        # month said as a number after the day ("dritten zehnten"): keep only the first ordinal word
        for k in range(len(head), 0, -1):
            jj = joined(" ".join(head[:k]))
            if jj in cards(lang) or re.search(r"(te|ten|ter|tes|tem|de|ste|ero|era|eiro|eira|imo|ima|ier|iere)$", jj):
                j, day_tok = jj, " ".join(head[:k])
                break
    if lang == "de":
        day = "day_ord" if re.search(r"(te|ten|ter|tes|tem)$", j) else "day_card"
    elif lang == "nl":
        day = "day_ord" if re.search(r"(de|ste)$", j) and j not in cards(lang) else "day_card"
    else:
        day = "day_card" if j in cards(lang) else ("day_ord" if j else "?")
    first = ""
    if re.match(r"\D*0?1\b", written) or j in FIRST.get(lang, ()):
        if j in FIRST.get(lang, ()):
            first = f"[1st={day_tok}]"
    return f"{day}+{month}{first}", {}


# ------------------------------------------------------------------ range (merged pieces)
RJOIN = {"de": ["bis", "oder", "und"], "es": ["a", "al", "hasta", "y", "o", "u"], "fr": ["a", "au", "et", "ou"],
         "it": ["a", "al", "fino", "e", "o"], "nl": ["tot", "en", "of", "a"], "pt": ["a", "ao", "ate", "e", "ou"]}
BETW = {"de": "zwischen", "es": "entre", "fr": "entre", "it": "tra|fra", "nl": "tussen", "pt": "entre"}


def rng_(lang, reading_right, before):
    t = toks(reading_right)
    b = " ".join(toks(before)[-4:])
    j = t[0] if t and t[0] in RJOIN[lang] else ("" if not t else None)
    betw = bool(re.search(rf"\b(?:{BETW[lang]})\b", b))
    if j is None:
        return ("between_bare" if betw else "bare"), {}
    if not j:
        return "bare", {}
    if betw:
        return f"between..{j}", {}
    if lang == "es" and re.search(r"\b(de|del|desde)\s*$", b):
        return f"de..{j}", {}
    if lang == "fr" and re.search(r"\b(de|du|d)\s*$", b):
        return f"de..{j}", {}
    if lang == "de" and re.search(r"\b(von|vom)\s*$", b):
        return f"von..{j}", {}
    if lang == "nl" and re.search(r"\bvan\s*$", b):
        return f"van..{j}", {}
    if lang in ("it", "pt") and re.search(r"\b(da|dal|dalla|de|do|dos)\s*$", b):
        return f"da..{j}", {}
    return j, {}


# ------------------------------------------------------------------ units of a row (census + fill)
CLASSES = ("phone", "year", "dec", "cur", "clock", "pct", "ord", "date", "int_big", "range")


def _locate(text, sp, nc):
    rd = (sp.get("reading") or "").strip()
    if nc:
        a, b = nc
        if joined(text[a:b]) == joined(rd):
            return a, b
    if rd:
        i = text.find(rd)
        if i >= 0:
            return i, i + len(rd)
    return (nc[0], nc[1]) if nc else (None, None)


def units(r: dict, lang: str) -> list[dict]:
    """The spoken numbers of a row: its spans with phone groups, date pieces and the two ends of a range merged into
    one unit each. A unit: cls (the data gate's class), val, kind, written, reading, a / b (label offsets), before /
    after (60 label characters around it), tok (label token indices), right (a range's right part), pieces."""
    import census_common as CC

    text = r.get("text", "")
    sps = r.get("spans") or []
    ncs = r.get("num_chars") or []
    items = []
    for i, sp in enumerate(sps):
        nc = ncs[i] if i < len(ncs) else None
        try:
            c, v = CC.gate_class_value(r, sp, lang)
        except Exception:  # noqa: BLE001  (a span the gate cannot class is the span's own kind)
            c, v = sp.get("kind"), None
        a, b = _locate(text, sp, nc)
        items.append(dict(cls=c, val=v, kind=sp.get("kind"), written=sp.get("text", ""), reading=sp.get("reading") or "",
                          a=a, b=b))
    out, i = [], 0
    while i < len(items):
        it = dict(items[i])
        j = i + 1
        merge_cls = it["cls"] if it["cls"] in ("phone", "date") else ("range" if it["kind"] == "range" else None)
        while merge_cls and j < len(items):
            nx = items[j]
            same = (nx["cls"] == merge_cls) if merge_cls != "range" else (nx["kind"] == "range" and
                                                                           nx["written"].lstrip()[:1] in "-–")
            if not same or it["b"] is None or nx["a"] is None:
                break
            gap = text[it["b"]:nx["a"]]
            if merge_cls == "phone" and re.search(r"[A-Za-zÀ-ÿ]", gap):
                break
            if merge_cls == "date" and len(gap.strip()) > 4:
                break
            if merge_cls == "range" and len(gap.strip()) > 2:
                break
            it["right"] = nx["reading"] if merge_cls == "range" else it.get("right")
            it.setdefault("pieces", [(items[i]["written"], items[i]["reading"])]).append((nx["written"], nx["reading"]))
            it["written"] = it["written"] + nx["written"]
            it["reading"] = it["reading"] + " " + nx["reading"]
            it["b"] = nx["b"]
            j += 1
        if merge_cls == "range":
            it["cls"] = "range" if "right" in it else "range_left_only"
        a, b = it["a"], it["b"]
        it["before"] = text[max(0, (a or 0) - 60):a or 0]
        it["after"] = text[b:b + 60] if b is not None else ""
        it["tok"] = ([k for k, m in enumerate(re.finditer(r"\S+", text)) if m.start() < b and m.end() > a]
                     if a is not None else [])
        out.append(it)
        i = j
    return out


def classify(lang: str, u: dict) -> tuple:
    """(convention, flags) of one unit (units()); (None, {}) for a class without conventions here."""
    c = u["cls"]
    w, rd = u["written"], u["reading"]
    try:
        if c == "phone":
            return phone_groups(lang, u.get("pieces") or [(w, rd)])
        if c == "year" and isinstance(u["val"], int):
            conv, f = year(lang, u["val"], rd)
            return conv, dict(f, era="1800s-1900s" if u["val"] < 2000 else "2000s")
        if c == "int_big" and isinstance(u["val"], int):
            return int_big(lang, u["val"], rd)
        if c == "dec":
            return dec(lang, w, rd)
        if c == "pct":
            return pct(lang, w, rd, u["after"])
        if c == "cur":
            return cur(lang, w, rd, u["after"], u["kind"] == "cur")
        if c == "clock":
            return clock(lang, w, rd, u["after"])
        if c == "ord":
            return ordinal(lang, rd)
        if c == "date":
            return date(lang, w, rd, u["after"])
        if c == "range":
            return rng_(lang, u.get("right") or "", u["before"])
    except Exception as e:  # noqa: BLE001  (a grammar gap is counted, not a crash)
        return f"err:{type(e).__name__}", {}
    return None, {}


def cell(lang: str, u: dict) -> str | None:
    """The fill-plan cell of a unit: its class, years split by era ("year:1800s-1900s" / "year:2000s")."""
    if u["cls"] == "year" and isinstance(u.get("val"), int):
        return "year:" + ("1800s-1900s" if u["val"] < 2000 else "2000s")
    return u["cls"] if u["cls"] in CLASSES else None
