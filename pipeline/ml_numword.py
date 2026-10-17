"""Number-WORD captions (2026-10-16): real rows whose caption already spells its numbers as words (de es fr it nl pt).

The digit pipeline (ml_p3.scan2_lang) keeps only captions with digits and relabels each digit run with a decided
reading. A caption that already says "halb drei", "le dix-huit juin", "cinco euros" or "mil novecientos ochenta" is in
spoken form as it stands: it needs no reading decision, only a row-level check that the audio says the same words.

This module is text-only (no audio, no Modal) and has no data of its own besides small context-word lists:
    nums(text, lang)        every number found in the caption: [Num(a, b, value, ordinal, weak, cls)], char offsets
    analyse(text, lang)     -> {"ok", "why", "spans", "classes"}: the row-level verdict of the caption
    template(text, spans)   the caption with every number replaced by "#" (the sentence-template cap key)
    select(cands, n, ...)   variety-first pick: years by value (forward years first), rare classes first, per-video
                            and per-template caps
    make_row(cand, lang, audio_filepath, duration)   a label2-shaped row (spans, num_chars; reading = caption words)

Number words come from the grammar table (ml_cands.table(): every "card" / "year" / "ord" form) plus the scale words;
compounds (de / nl / it "neunzehnhundertachtzig", "tweeduizendtwintig", "millenovecentottanta") are split into table
pieces. A caption qualifies only with a real number: a value >= 2, an ordinal, or a value 1 with a unit / currency /
time word. Articles ("ein", "une", "uno", "um", "een") are not numbers; a few words that are numbers only sometimes
("sei" it, "neuf" fr, "acht" de / nl, "cuarto" / "quarto") are resolved from their neighbours, and a caption where one
stays unresolved is dropped (an unmarked real number would sit outside the number spans of the row).
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal

import ml_cands as M

LANGS = ("de", "es", "fr", "it", "nl", "pt")
DIGIT = re.compile(r"\d")
SYMBOL = re.compile(r"[$€£¥¢₹₩₽%#/+°²³¹½¼¾⅓⅔&‰@=]")  # mix_build.WF_LABEL and a few more
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")  # letters, hyphen compounds kept ("dix-huit"), apostrophes split
BREAK = re.compile(r"[,.;:!?()\"«»“”…–—/\[\]]")  # punctuation between two words ends a number

SCALES = {
    "de": {"hundert": 100, "tausend": 1000, "million": 10**6, "millionen": 10**6, "milliarde": 10**9, "milliarden": 10**9},
    "nl": {"honderd": 100, "duizend": 1000, "miljoen": 10**6, "miljoenen": 10**6, "miljard": 10**9, "miljarden": 10**9},
    "it": {"cento": 100, "mille": 1000, "mila": 1000, "milione": 10**6, "milioni": 10**6, "miliardo": 10**9,
           "miliardi": 10**9},
    "es": {"cien": 100, "ciento": 100, "mil": 1000, "millón": 10**6, "millones": 10**6, "millon": 10**6,
           "billón": 10**12, "billones": 10**12},
    "fr": {"cent": 100, "cents": 100, "mille": 1000, "mil": 1000, "million": 10**6, "millions": 10**6, "milliard": 10**9,
           "milliards": 10**9},
    "pt": {"cem": 100, "cento": 100, "mil": 1000, "milhão": 10**6, "milhões": 10**6, "bilhão": 10**9, "bilhões": 10**9,
           "milhao": 10**6, "milhoes": 10**6},
}
CONN = {"de": {"und"}, "nl": {"en"}, "es": {"y"}, "pt": {"e"}, "it": {"e"}, "fr": {"et"}}
# value-1 words that are articles far more often than numbers: a number only before a scale word (or a unit /
# currency / time word, then a value-1 span that alone never qualifies a caption)
ARTICLE = {"de": {"ein", "eine", "einen", "einem", "einer", "eines"}, "nl": {"een"}, "es": {"un", "una", "uno"},
           "pt": {"um", "uma"}, "it": {"un", "una", "uno"}, "fr": {"un", "une"}}
# ordinal-looking words that mostly mean something else ("secondo me", "según", "a second"): never numbers here
NOT_NUM = {"es": {"segundo", "segunda", "segundos", "segundas"}, "pt": {"segundo", "segunda", "segundos", "segundas"},
           "it": {"secondo", "seconda", "secondi", "seconde", "prima"},  # it "prima" = "before" far more often
           "fr": {"second", "seconde", "seconds", "secondes"},
           "de": set(), "nl": set()}
DEC = {"de": {"komma"}, "nl": {"komma"}, "es": {"coma"}, "pt": {"vírgula", "virgula"}, "it": {"virgola"},
       "fr": {"virgule"}}
PCT = {"de": [("prozent",)], "nl": [("procent",)], "es": [("por", "ciento"), ("por", "cien")], "fr": [("pour", "cent")],
       "it": [("per", "cento")], "pt": [("por", "cento")]}
CUR = {"euro", "euros", "eur", "cent", "cents", "centime", "centimes", "céntimo", "céntimos", "centesimo", "centesimi",
       "centavo", "centavos", "cêntimo", "cêntimos", "dollar", "dollars", "dólar", "dólares", "dollaro", "dollari",
       "pfund", "pound", "pounds", "livre", "livres", "libra", "libras", "sterlina", "sterline", "franken", "franc",
       "francs", "franchi", "franco", "francos", "peso", "pesos", "real", "reais", "yen", "rappen", "gulden", "pond"}
# currency words that are also common nouns ("un livre" = a book, "um peso" = a weight, "ein Pfund Mehl"): after an
# article they are not money
CUR_NOUN = {"livre", "livres", "peso", "pesos", "real", "reais", "franco", "francos", "pond", "pfund", "pound", "libra",
            "libras", "franc"}
CUR_LINK = {"de", "di", "d", "van", "of"}  # "cinco millones de euros", "dieci milioni di euro"
UNITS = {
    "de": "jahr jahre jahren jährige monat monate monaten woche wochen tag tage tagen stunde stunden minute minuten "
          "sekunde sekunden meter metern kilometer kilometern zentimeter millimeter kilo kilogramm gramm tonne tonnen "
          "liter grad punkte euro",
    "nl": "jaar jaren maand maanden week weken dag dagen uur uren minuut minuten seconde seconden meter kilometer "
          "centimeter millimeter kilo kilogram gram liter graden ton punten",
    "es": "año años mes meses semana semanas día días hora horas minuto minutos metro metros kilómetro kilómetros "
          "centímetro centímetros milímetros kilo kilos kilogramo kilogramos gramo gramos litro litros grado grados "
          "tonelada toneladas puntos",
    "pt": "ano anos mês meses semana semanas dia dias hora horas minuto minutos metro metros quilômetro quilômetros "
          "quilómetro quilómetros centímetro centímetros quilo quilos grama gramas litro litros grau graus tonelada "
          "toneladas pontos",
    "it": "anno anni mese mesi settimana settimane giorno giorni ora ore minuto minuti metro metri chilometro chilometri "
          "centimetro centimetri chilo chili chilogrammi grammi litro litri grado gradi tonnellata tonnellate punti",
    "fr": "an ans année années mois semaine semaines jour jours heure heures minute minutes mètre mètres kilomètre "
          "kilomètres centimètre centimètres kilo kilos kilogramme kilogrammes gramme grammes litre litres degré "
          "degrés tonne tonnes points",
}
UNITS = {k: set(v.split()) for k, v in UNITS.items()}
MONTHS = {
    "de": "januar jänner februar märz april mai juni juli august september oktober november dezember",
    "nl": "januari februari maart april mei juni juli augustus september oktober november december",
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre",
    "pt": "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro",
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre",
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre",
}
MONTHS = {k: set(v.split()) for k, v in MONTHS.items()}
YEAR_CUE = {"de": {"jahr", "jahre", "jahres", "anno", "seit", "im"}, "nl": {"jaar", "sinds", "in"},
            "es": {"año", "en", "desde"}, "pt": {"ano", "em", "desde"}, "it": {"anno", "nel", "dal"},
            "fr": {"année", "an", "en", "depuis"}}
# clock: a word right before (b1) / after (a1) the hour number, or two words before (b2, the second-last word given)
CLOCK_AFTER = {"de": {"uhr"}, "nl": set(), "fr": set(), "es": set(), "pt": set(), "it": set()}
CLOCK_BEFORE = {"de": {"halb"}, "nl": {"half"}, "es": set(), "pt": set(), "it": set(), "fr": set()}
CLOCK_PREP = {  # "a las tres", "às duas", "alle otto", "om drie uur", "à huit heures"
    "es": ({"las", "la"}, {"a", "de", "desde", "hasta", "son", "eran", "es", "era", "hacia", "sobre", "las", "entre"}),
    "pt": ({"às", "as", "à"}, None),
    "it": ({"alle", "dalle", "le", "all", "dall", "l"}, None),  # "le": "sono le tre", "verso le otto" (tails below)
}
CLOCK_TAIL = {"es": [("y", "media"), ("y", "cuarto"), ("menos", "cuarto"), ("en", "punto")],
              "pt": [("e", "meia"), ("horas",), ("hora",), ("em", "ponto")],
              "it": [("e", "mezza"), ("e", "mezzo"), ("e", "un", "quarto"), ("meno", "un", "quarto"), ("in", "punto")],
              "fr": [("heures",), ("heure",)], "nl": [("uur",)], "de": [("uhr",)]}
DAYPART = {"es": {("de", "la", "mañana"), ("de", "la", "tarde"), ("de", "la", "noche")},
           "pt": {("da", "manhã"), ("da", "tarde"), ("da", "noite")},
           "it": {("di", "mattina"), ("del", "mattino"), ("di", "sera"), ("del", "pomeriggio"), ("di", "notte")},
           "fr": {("du", "matin"), ("du", "soir"), ("de", "l", "après-midi")}, "de": set(), "nl": set()}
CLOCK_NEXT = {"en", "in", "circa", "precise", "pm", "am", "y", "e"}
CLOCK_WORDS = {"uhr", "heures", "heure", "uur", "horas", "hora", "ore"}
FR_CLOCK_PREP = {"à", "vers", "jusqu", "dès", "avant", "après", "est", "de"}
NL_CLOCK_PREP = {"om", "tot", "vanaf", "rond", "sinds"}


def fold(w: str) -> str:
    return w.lower().replace("’", "'")


# ---------------------------------------------------------------- lexicon (grammar table + scale words)
_LEX: dict = {}


@dataclass
class Lex:
    phrases: dict = field(default_factory=dict)  # tuple(words) -> (value, ordinal, year_form)
    pieces: dict = field(default_factory=dict)  # single-word card forms < 1000 + scale words (compound splitting)
    maxlen: int = 1


def _ord_variants(lang: str, w: str) -> list:
    if lang in ("es", "pt", "it") and w[-1:] in "oa":
        # it plurals ("terzi" = third parties, "ottave" = octaves, "quinte") are nouns more often: singular only
        out = [w[:-1] + e for e in ("o", "a", "os", "as")] if lang != "it" else [w[:-1] + e for e in "oa"]
        if lang == "es" and w in ("primero", "tercero"):
            out.append(w[:-2])  # "primer", "tercer"
        return out
    if lang == "fr":
        return [w, w + "s"] + (["première", "premières", "premiers"] if w == "premier" else [])
    if lang == "de" and w.endswith("e"):
        return [w[:-1] + e for e in ("e", "en", "er", "es", "em", "ens")]
    if lang == "nl":
        return [w]
    return [w]


def _card_variants(lang: str, w: str) -> list:
    out = [w]
    if lang in ("es", "pt"):
        if w.endswith("ientos") or w.endswith("entos"):
            out.append(w[:-3] + "tas")
        if lang == "es" and w.endswith("uno"):
            out += [w[:-1] + "a", w[:-3] + "ún"]
        if lang == "pt" and w == "dois":
            out.append("duas")
    if lang == "nl" and "ë" in w:
        out.append(w.replace("ë", "e"))
    return out


def lexicon(lang: str) -> Lex:
    if lang in _LEX:
        return _LEX[lang]
    lx = Lex()
    for k, forms in M.table().items():
        l, kind, n, f, s, h, m = k.split("|")
        if l != lang or (f, s, h, m) != ("", "", "0", "0") or kind not in ("card", "year", "ord") or not forms:
            continue
        v = int(n)
        words = [fold(w) for w in forms]
        if kind == "ord":
            for i, last in enumerate(_ord_variants(lang, words[-1])):
                lx.phrases.setdefault(tuple(words[:-1] + [last]), (v, True, False))
            continue
        variants = [words] + [words[:-1] + [x] for x in _card_variants(lang, words[-1])[1:]]
        if len(words) == 1:
            variants += [[x] for x in _card_variants(lang, words[0])[1:]]
        for ws in variants:
            key = tuple(ws)
            old = lx.phrases.get(key)
            if old is None or (kind == "year" and not old[1]):
                lx.phrases[key] = (v, False, kind == "year")
    for w, v in SCALES[lang].items():
        lx.phrases.setdefault((w,), (v, False, False))
    for key, (v, o, _) in lx.phrases.items():
        if len(key) == 1 and not o and (v < 1000 or w_is_scale(lang, key[0])):
            lx.pieces[key[0]] = v
    if lang == "de":
        lx.pieces.update({"ein": 1, "eins": 1})
    if lang == "nl":
        lx.pieces.update({"een": 1, "één": 1})
    if lang == "it":
        lx.pieces.update({"un": 1})
    lx.maxlen = max(len(k) for k in lx.phrases)
    _LEX[lang] = lx
    return lx


def w_is_scale(lang: str, w: str) -> bool:
    return w in SCALES[lang]


def split_compound(lang: str, w: str):
    """The piece values of a one-word compound ("neunzehnhundertachtzig" -> [19, 100, 80]) or None (fewest pieces)."""
    if lang not in ("de", "nl", "it") or len(w) < 6:
        return None
    pc = lexicon(lang).pieces
    best: list = [None] * (len(w) + 1)
    best[0] = []
    for i in range(len(w)):
        if best[i] is None:
            continue
        for j in range(i + 2, len(w) + 1):
            p = w[i:j]
            if p in pc and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + [pc[p]]
            elif lang in ("de", "nl") and p in ("und", "en") and best[i] and (best[j] is None or len(best[j]) > len(best[i])):
                best[j] = best[i] + ["+"]  # "dreiund|zwanzig": the joiner of unit-first compounds
    r = best[len(w)]
    if not r or len(r) < 2:
        return None
    # unit-first joiner: "drei und zwanzig" -> 23
    out = []
    for v in r:
        if out and out[-1] == "+" and isinstance(v, int):
            out.pop()
            u = out.pop() if out and isinstance(out[-1], int) and out[-1] < 10 else None
            if u is None or v % 10 or v >= 100:
                return None
            out.append(v + u)
        else:
            out.append(v)
    if "+" in out:
        return None
    return out


def _ord_compound(lang: str, w: str):
    """Ordinal of a compound / derived word: de "zweiundzwanzigsten", nl "achtste", fr "dix-huitième", it
    "ventitreesimo". -> value or None."""
    if lang == "de":
        m = re.fullmatch(r"(.+?)(ens|en|er|es|em|e)", w)
        if not m:
            return None
        st = m.group(1)
        irr = {"erst": 1, "dritt": 3, "siebt": 7, "siebent": 7, "acht": 8}
        if st in irr:
            return irr[st]
        for cut in ("st", "t"):
            if st.endswith(cut):
                v = _card_word(lang, st[:-len(cut)])
                if v is not None and (cut == "st") == (v >= 20):
                    return v
        return None
    if lang == "nl":
        irr = {"eerste": 1, "derde": 3, "achtste": 8}
        if w in irr:
            return irr[w]
        for cut in ("ste", "de"):
            if w.endswith(cut):
                v = _card_word(lang, w[:-len(cut)])
                if v is not None and (cut == "ste") == (v >= 20 or v in (1, 8)):
                    return v
        return None
    if lang == "fr":
        m = re.fullmatch(r"(.+)ièmes?", w)
        if not m:
            return None
        st = m.group(1)
        parts = st.split("-")
        last = parts[-1]
        for c in (last, last + "e", {"cinqu": "cinq", "neuv": "neuf"}.get(last, last)):
            vv = _value_of_words(lang, parts[:-1] + [c])
            if vv is not None:
                return vv
        return None
    if lang == "it":
        m = re.fullmatch(r"(.+)esim[oaie]", w)
        if not m:
            return None
        st = m.group(1)
        for c in (st, st + "e", st + "i", st + "o", st + "a"):
            v = _card_word(lang, c)
            if v is not None and v > 10:
                return v
        return None
    return None


def _card_word(lang: str, w: str):
    lx = lexicon(lang)
    p = lx.phrases.get((w,))
    if p and not p[1]:
        return p[0]
    pcs = split_compound(lang, w)
    if pcs:
        r = compose(pcs)
        if len(r) == 1:
            return r[0]
    return None


def _value_of_words(lang: str, ws: list):
    vals = []
    for w in ws:
        v = _card_word(lang, w)
        if v is None:
            return None
        vals.append(v)
    r = compose(vals)
    return r[0] if len(r) == 1 else None


def _place(n: int) -> int:
    p = 1
    while n and n % (p * 10) == 0:
        p *= 10
    return p


def compose(vals: list) -> list:
    """Piece values -> the numbers they make, split where the sequence cannot be one number ("zwei drei" -> [2, 3])."""
    out, groups = [], []  # groups: per number, the piece indices consumed
    total = cur = 0
    hi = None
    started = False

    def emit():
        nonlocal total, cur, hi, started
        if started:
            out.append(total + cur)
        total = cur = 0
        hi = None
        started = False

    for v in vals:
        if v == 100:
            if not started or (cur == 0 and total):
                cur = 100
            elif 0 < cur < 100:
                cur *= 100
            else:
                emit()
                cur = 100
        elif v >= 1000 and _place(v) == v and v in (10**3, 10**6, 10**9, 10**12):
            if hi is not None and v >= hi:
                emit()
            g = cur or (0 if total else 1)
            if not g and total:
                emit()
                g = 1
            total += g * v
            cur = 0
            hi = v
        else:
            if not started:
                cur = v
            elif cur == 0:
                if total and v < (hi or 10**15):
                    cur = v
                else:
                    emit()
                    cur = v
            elif _place(cur) > v:
                cur += v
            else:
                emit()
                cur = v
        started = True
    emit()
    return out


# ---------------------------------------------------------------- numbers in a caption
@dataclass
class Num:
    a: int  # char offsets of the whole span in the caption (number words + their unit / currency / clock words)
    b: int
    value: Decimal
    ordinal: bool = False
    weak: bool = False  # a value-1 article word: never qualifies a caption alone
    cls: str = "int"
    year_form: bool = False
    toks: tuple = ()  # token indices of the number words
    bare_scale: bool = False  # one scale word with no multiplier ("mille", "tausend")


def tokens(text: str) -> list:
    return [(fold(m.group()), m.start(), m.end()) for m in WORD.finditer(text)]


def _broken(text: str, toks: list, i: int) -> bool:
    """Punctuation (or a digit) between token i and token i + 1."""
    return bool(BREAK.search(text[toks[i][2]:toks[i + 1][1]]) or DIGIT.search(text[toks[i][2]:toks[i + 1][1]]))


def _match(lang: str, toks: list, i: int, text: str):
    """Longest number phrase at token i -> (n_tokens, value, ordinal, year_form) or None."""
    lx = lexicon(lang)
    for n in range(min(lx.maxlen, len(toks) - i), 0, -1):
        if any(_broken(text, toks, i + k) for k in range(n - 1)):
            continue
        key = tuple(t[0] for t in toks[i:i + n])
        if key in lx.phrases:
            v, o, y = lx.phrases[key]
            return n, v, o, y
    w = toks[i][0]
    pcs = split_compound(lang, w)
    if pcs:
        r = compose(pcs)
        if len(r) == 1:
            return 1, r[0], False, False
    v = _ord_compound(lang, w)
    if v is not None:
        return 1, v, True, False
    if "-" in w:  # "quatre-vingt-dix", "vingt-et-un": the table's spaced phrase
        key = tuple(p for p in w.split("-") if p)
        if key in lx.phrases:
            v, o, y = lx.phrases[key]
            return 1, v, o, y
    if "-" in w and lang in ("fr", "es", "pt", "it"):  # "vingt-deux" not in the table as one word
        vv = _value_of_words(lang, w.split("-"))
        if vv is not None:
            return 1, vv, False, False
    return None


def _skip_pct(lang: str, toks: list, i: int) -> int:
    for ph in PCT[lang]:
        if tuple(t[0] for t in toks[i:i + len(ph)]) == ph:
            return len(ph)
    return 0


def raw_numbers(text: str, lang: str) -> list:
    """[(i0, i1, value, ordinal, year_form)] token ranges of numbers (scale composition across words, joiners)."""
    toks = tokens(text)
    out = []
    i = 0
    while i < len(toks):
        if _skip_pct(lang, toks, i):
            i += _skip_pct(lang, toks, i)
            continue
        if toks[i][0] in NOT_NUM[lang]:
            i += 1
            continue
        m = _match(lang, toks, i, text)
        if not m:
            i += 1
            continue
        n, v, o, y = m
        if o:  # an ordinal ends its number
            out.append((i, i + n, v, True, False))
            i += n
            continue
        seq = [(i, i + n, v, y)]
        j = i + n
        while j < len(toks) and not _broken(text, toks, j - 1):
            if _skip_pct(lang, toks, j):
                break
            jj = j
            if toks[j][0] in CONN[lang] and j + 1 < len(toks) and not _broken(text, toks, j):
                jj = j + 1
            m2 = _match(lang, toks, jj, text) if toks[jj][0] not in NOT_NUM[lang] else None
            if not m2 or m2[2]:
                break
            n2, v2 = m2[0], m2[1]
            vals = [s[2] for s in seq] + [v2]
            if len(compose(vals)) != 1:
                break
            seq.append((jj, jj + n2, v2, m2[3]))
            j = jj + n2
        val = compose([s[2] for s in seq])[0]
        out.append((i, seq[-1][1], val, False, len(seq) == 1 and seq[0][3]))
        i = seq[-1][1]
    return out


def _words(toks, i, n):
    return tuple(t[0] for t in toks[i:i + n])


def nums(text: str, lang: str) -> list:
    """Every number of the caption, with its class and its span (char offsets). Raises nothing; may return ambiguous
    look-alikes as Num(cls="ambiguous")."""
    toks = tokens(text)
    raw = raw_numbers(text, lang)
    out: list = []
    k = 0
    while k < len(raw):
        i0, i1, v, o, y = raw[k]
        val = Decimal(v)
        a, b = toks[i0][1], toks[i1 - 1][2]
        nxt = toks[i1][0] if i1 < len(toks) and not _broken(text, toks, i1 - 1) else None
        prv = toks[i0 - 1][0] if i0 > 0 and not _broken(text, toks, i0 - 1) else None
        prv2 = toks[i0 - 2][0] if i0 > 1 and prv is not None and not _broken(text, toks, i0 - 2) else None
        first = toks[i0][0]
        single = i1 - i0 == 1
        weak = single and first in ARTICLE[lang] and v == 1
        cls = "ord" if o else "int"
        # decimal: "zwei Komma fünf"
        if (not o and nxt in DEC[lang] and k + 1 < len(raw) and raw[k + 1][0] == i1 + 1 and not raw[k + 1][3]
                and not _broken(text, toks, i1)):
            j0, j1, v2 = raw[k + 1][0], raw[k + 1][1], raw[k + 1][2]
            frac, kk = str(v2), k + 2
            # digit by digit after the separator: "zwei Komma null fünf" = 2.05, "drei Komma zwei fünf" = 3.25
            while (len(frac) == 1 and kk < len(raw) and raw[kk][0] == j1 and raw[kk][1] - raw[kk][0] == 1
                   and not raw[kk][3] and 0 <= raw[kk][2] <= 9 and not _broken(text, toks, j1 - 1)):
                j1, frac, kk = raw[kk][1], frac + str(raw[kk][2]), kk + 1
            out.append(Num(a, toks[j1 - 1][2], Decimal(f"{v}.{frac}"), cls="dec", toks=tuple(range(i0, j1))))
            k = kk
            continue
        # percent: "zwanzig Prozent", "veinte por ciento"
        n_pct = _skip_pct(lang, toks, i1) if nxt else 0
        if n_pct and not o:
            out.append(Num(a, toks[i1 + n_pct - 1][2], val, cls="pct", toks=tuple(range(i0, i1))))
            k += 1
            continue
        # currency: "fünf Euro", "cinco millones de euros"
        cur_end = None
        if nxt in CUR:
            cur_end = i1
        elif nxt in CUR_LINK and i1 + 1 < len(toks) and toks[i1 + 1][0] in CUR and not _broken(text, toks, i1):
            cur_end = i1 + 1
        if cur_end is not None and weak and toks[cur_end][0] in CUR_NOUN:
            cur_end = None
        if cur_end is not None and not o:
            out.append(Num(a, toks[cur_end][2], val, weak=False, cls="cur", toks=tuple(range(i0, i1)),
                           bare_scale=single and first in SCALES[lang] and v >= 1000))
            k += 1
            continue
        # clock
        clock = _clock(lang, toks, text, i0, i1, v, o, prv, prv2, nxt)
        if clock is not None:
            ca, cb = clock
            k += 1
            # "drei Uhr dreißig", "huit heures trente", "às duas horas e dez": the minutes join the clock span
            if (k < len(raw) and cb < len(toks) and not _broken(text, toks, cb - 1) and not raw[k][3] and raw[k][2] < 60
                    and ((raw[k][0] == cb and toks[cb - 1][0] in CLOCK_WORDS)
                         or (raw[k][0] == cb + 1 and toks[cb][0] in CONN[lang] and not _broken(text, toks, cb)))):
                cb = raw[k][1]
                k += 1
            out.append(Num(toks[ca][1], toks[cb - 1][2], val, cls="clock", toks=tuple(range(i0, i1))))
            continue
        # date: "am dritten Mai", "el dos de mayo", "le dix-huit juin", "drie mei"
        mon = nxt in MONTHS[lang] or (nxt in ("de", "di", "du", "van") and i1 + 1 < len(toks)
                                      and toks[i1 + 1][0] in MONTHS[lang] and not _broken(text, toks, i1))
        if mon and 1 <= v <= 31:
            out.append(Num(a, b, val, ordinal=o, cls="date", toks=tuple(range(i0, i1))))
            k += 1
            continue
        if (not o and (y or 1100 <= v <= 2099) and v >= 1100 and (v % 100 or prv in YEAR_CUE[lang])
                and nxt not in UNITS[lang]):
            cls = "year"
        elif not o and v >= 1000:
            cls = "big"
        elif not o and nxt in UNITS[lang]:
            cls = "unit"
        amb = _ambiguous(lang, first, single, o, prv, nxt, toks, i1, text)
        if amb == "not":
            k += 1
            continue
        if amb == "?":
            cls = "ambiguous"
        if weak:
            k += 1  # an article, not a number ("un jour", "ein Jahr" are left out too: mostly idioms)
            continue
        out.append(Num(a, b, val, ordinal=o, weak=weak or (v == 1 and not o and cls == "int"), cls=cls,
                       year_form=y, toks=tuple(range(i0, i1)), bare_scale=single and first in SCALES[lang] and v >= 1000))
        k += 1
    return out


def _clock(lang, toks, text, i0, i1, v, o, prv, prv2, nxt):
    """(token start, token end) of a clock span around number tokens [i0, i1), or None."""
    if o or v > 24:
        return None
    n = len(toks)

    def tail(j):
        for ph in CLOCK_TAIL.get(lang, []):
            if _words(toks, j, len(ph)) == ph and not any(_broken(text, toks, j - 1 + q) for q in range(len(ph))):
                return j + len(ph)
        for ph in DAYPART.get(lang, set()):
            if _words(toks, j, len(ph)) == ph and not any(_broken(text, toks, j - 1 + q) for q in range(len(ph))):
                return j + len(ph)
        return None

    if lang == "de":
        if nxt == "uhr":
            return i0, i1 + 1
        if prv == "halb" and v <= 12:
            return i0 - 1, i1
        if prv in ("nach", "vor") and prv2 == "viertel" and v <= 12:
            return i0 - 2, i1
        return None
    if lang == "nl":
        if prv == "half" and v <= 12:
            return i0 - 1, i1
        if prv in ("over", "voor") and prv2 == "kwart" and v <= 12:
            return i0 - 2, i1
        if nxt == "uur" and prv in NL_CLOCK_PREP:
            return i0, i1 + 1
        return None
    if lang == "fr":
        if nxt in ("heures", "heure") and (prv in FR_CLOCK_PREP or (i1 + 1 < n and toks[i1 + 1][0] in ("et", "moins")
                                                                   and i1 + 2 < n and toks[i1 + 2][0] in ("demie", "quart", "le"))):
            e = i1 + 1
            if e < n and toks[e][0] == "et" and e + 1 < n and toks[e + 1][0] in ("demie", "quart"):
                e += 2
            return i0, e
        return None
    arts, preps = CLOCK_PREP[lang]
    t = tail(i1) if nxt else None
    if prv not in arts or (preps is not None and prv2 not in preps and not t):
        return None
    # "a las tres hermanas", "alle tre ragazze", "as duas casas": an article + number, not a time. A time ends its
    # phrase, or goes on with a clock tail ("y media", "e mezza", "horas"), minutes ("e dieci") or a day part.
    end = nxt is None
    minutes = nxt in CONN[lang] and i1 + 1 < n and _match(lang, toks, i1 + 1, text) is not None
    if t or end or minutes or nxt in CLOCK_NEXT:
        return i0, (t or i1)
    return None


def _ambiguous(lang, w, single, o, prv, nxt, toks, i1, text):
    """"num" (a number here), "not" (another word), "?" (cannot tell) for the few words that are both."""
    if not single:
        return "num"
    plural_ok = UNITS[lang] | CUR | set(SCALES[lang])
    if lang == "it" and w == "sei":
        if prv in {"tu", "che", "non", "ci", "lo", "la", "ne", "mi", "ti", "se", "ma", "chi", "dove", "come", "cosa",
                   "quanto", "quando", "perché", "sì", "già", "mai", "ora", "e", "o"}:
            return "not"
        if nxt in plural_ok or (nxt or "").endswith(("i", "e")) and prv in {"i", "le", "dei", "delle", "alle", "dalle",
                                                                            "tra", "fra", "per", "sulle", "degli", "gli"}:
            return "num"
        if prv in {"le", "alle", "dalle", "delle", "dei", "tra", "fra", "sulle"}:
            return "num"
        return "?"
    if lang == "fr" and w == "neuf":
        if prv in {"tout", "toute", "flambant", "du", "rien", "de", "quoi", "état"} or nxt in {"et", "ou"}:
            return "not" if prv in {"tout", "toute", "flambant", "du", "rien", "quoi", "état"} else "?"
        if nxt in plural_ok or (nxt or "").endswith(("s", "x")) or prv in {"les", "des", "à", "chapitre", "numéro"}:
            return "num"
        return "?"
    if lang == "de" and w == "acht":
        if prv in {"außer", "ausser", "in", "gib", "gebt", "geben"} or nxt in {"geben", "gegeben", "nehmen", "genommen",
                                                                               "lassen", "gelassen", "gab"}:
            return "not"
        return "num"
    if lang == "de" and o and w in ("achte", "achten", "achtet"):
        return "num" if prv in {"am", "zum", "im", "dem", "den", "der", "des", "die", "das", "vom", "beim", "ihr"} else "?"
    if lang == "nl" and w == "acht":
        if nxt in {"nemen", "genomen", "neemt", "neem", "slaan", "sloeg", "geslagen", "slaat"}:
            return "not"
        return "num"
    if lang == "pt" and w in ("terça", "quarta", "quinta", "sexta", "terças", "quartas", "quintas", "sextas"):
        return "?"  # weekdays ("de segunda a sexta", "na quinta-feira") as often as ordinals
    if lang in ("es", "pt", "it") and w in ("cuarto", "cuarta", "quarto", "quarta", "cuartos", "quartos", "quarti"):
        return "?"  # "room", "quarter", "fourth": the clock readings ("y cuarto") never reach here
    return "num"


def analyse(text: str, lang: str) -> dict:
    """{"ok": bool, "why": reason, "spans": [Num], "classes": [cls]} for one caption."""
    if DIGIT.search(text):
        return {"ok": False, "why": "digits", "spans": [], "classes": []}
    if SYMBOL.search(text):
        return {"ok": False, "why": "symbol", "spans": [], "classes": []}
    try:
        ns = nums(text, lang)
    except Exception as e:  # noqa: BLE001  (one odd caption must not stop a scan)
        return {"ok": False, "why": f"error {type(e).__name__}", "spans": [], "classes": []}
    if any(n.cls == "ambiguous" for n in ns):
        return {"ok": False, "why": "ambiguous", "spans": ns, "classes": []}
    for n in ns:
        if BREAK.search(text[n.a:n.b]):
            return {"ok": False, "why": "punct_in_span", "spans": ns, "classes": []}
    real = [n for n in ns if qualifies(n)]
    if not real:
        return {"ok": False, "why": "no_real_number" if ns else "no_number", "spans": ns, "classes": []}
    cls = sorted({("ord1" if n.cls == "ord" and n.value == 1 else n.cls) for n in real})
    return {"ok": True, "why": "", "spans": ns, "classes": cls}


def qualifies(n: Num) -> bool:
    if n.ordinal:
        return True
    if n.bare_scale:  # "grazie mille", "mil gracias", "des milliards de dollars": an idiom or a vague amount
        return False
    if n.value >= 2:
        return True
    return n.cls in ("cur", "pct", "clock", "unit", "dec", "date")


# ---------------------------------------------------------------- selection
PRIORITY = ("clock", "cur", "pct", "dec", "date", "year", "ord", "unit", "big", "int", "ord1")


def template(text: str, spans: list) -> str:
    t, last = [], 0
    for n in sorted(spans, key=lambda x: x.a):
        t.append(text[last:n.a])
        t.append("#")
        last = n.b
    t.append(text[last:])
    return re.sub(r"\s+", " ", re.sub(r"[^\w#\s]", " ", "".join(t).lower())).strip()


def primary(classes: list) -> str:
    for c in PRIORITY:
        if c in classes:
            return c
    return "int"


def select(cands: list, n: int, per_video: int = 8, per_template: int = 3, per_year: int = 3,
           shares: dict | None = None, seed: int = 1) -> list:
    """Variety-first pick of up to n candidates. cand = {"id", "text", "video", "classes", "years", "values",
    "template", ...}. Years: at most per_year rows per year value, forward years (>= 2025) first, then values the
    pick lacks; other classes: round robin in PRIORITY order with a share cap each (shares: class -> fraction of n),
    rows with an unseen value first. Caps: per_video rows per video, per_template rows per sentence template, one row
    per caption text."""
    import random

    shares = shares or {"clock": 0.15, "cur": 0.12, "pct": 0.1, "dec": 0.1, "date": 0.12, "year": 0.2, "ord": 0.12,
                        "unit": 0.1, "big": 0.07, "int": 0.12, "ord1": 0.03}
    rng = random.Random(seed)
    pool = list(cands)
    rng.shuffle(pool)
    by = {}
    for c in pool:
        by.setdefault(primary(c["classes"]), []).append(c)
    vid, tpl, texts, ycount, seen_vals = {}, {}, set(), {}, {}
    took, n_cls = [], {}

    def ok(c):
        return (vid.get(c["video"], 0) < per_video and tpl.get(c["template"], 0) < per_template
                and c["text"].strip().lower() not in texts and all(ycount.get(y, 0) < per_year for y in c.get("years", [])))

    def take(c, cl):
        took.append(dict(c, why=cl))
        vid[c["video"]] = vid.get(c["video"], 0) + 1
        tpl[c["template"]] = tpl.get(c["template"], 0) + 1
        texts.add(c["text"].strip().lower())
        n_cls[cl] = n_cls.get(cl, 0) + 1
        for y in c.get("years", []):
            ycount[y] = ycount.get(y, 0) + 1
        for v in c.get("values", []):
            seen_vals.setdefault(cl, set()).add(v)

    # years first: forward years, then every year value once, twice, ... up to per_year
    ys = by.get("year", [])
    cap_y = int(shares.get("year", 0) * n)
    for rnd in range(per_year):
        for fwd in (True, False):
            for c in sorted(ys, key=lambda c: min(c.get("years") or [0])):
                if n_cls.get("year", 0) >= cap_y or len(took) >= n:
                    break
                yv = [y for y in c.get("years", []) if (y >= 2025) == fwd]
                if not yv or c.get("_taken") or not ok(c) or any(ycount.get(y, 0) > rnd for y in c["years"]):
                    continue
                c["_taken"] = True
                take(c, "year")
    # the other classes: round robin, unseen values first
    order = [c for c in PRIORITY if c != "year"]
    caps = {c: int(shares.get(c, 0) * n) for c in order}
    queues = {c: sorted((x for x in by.get(c, []) if not x.get("_taken")), key=lambda x: rng.random()) for c in order}
    progress = True
    while len(took) < n and progress:
        progress = False
        for cl in order:
            if n_cls.get(cl, 0) >= caps[cl] or len(took) >= n:
                continue
            q = queues[cl]
            pick = None
            for idx, c in enumerate(q):  # an unseen value first
                if ok(c) and not set(c.get("values", [])) <= seen_vals.get(cl, set()):
                    pick = idx
                    break
            if pick is None:
                pick = next((idx for idx, c in enumerate(q) if ok(c)), None)
            if pick is None:
                continue
            c = q.pop(pick)
            c["_taken"] = True
            take(c, cl)
            progress = True
    # fill: anything left, round robin over the classes (a class short of rows leaves its room to the others); a
    # value-1 ordinal ("erste", "primero", "premier") never fills: thousands of them, little to learn
    while len(took) < n:
        moved = False
        for cl in [c for c in order if c != "ord1"]:
            q = queues[cl]
            idx = next((i for i, c in enumerate(q) if ok(c)), None)
            if idx is None or len(took) >= n:
                continue
            c = q.pop(idx)
            c["_taken"] = True
            take(c, cl)
            moved = True
        if not moved:
            break
    for c in pool:
        c.pop("_taken", None)
    return [{k: v for k, v in c.items() if k != "_taken"} for c in took]


# ---------------------------------------------------------------- rows
KIND = {"year": "int", "big": "int", "int": "int", "ord": "ord", "ord1": "ord", "clock": "clock", "cur": "cur",
        "pct": "pct", "dec": "dec", "date": "date", "unit": "unit"}


def make_row(text: str, lang: str, rid: str, audio_filepath: str, duration: float, src: str = "yodas",
             by: str = "caption", extra: dict | None = None) -> dict | None:
    """A label2-shaped row whose label IS the caption: one span per number (weak value-1 articles excluded), reading =
    the caption's own words. None when the caption does not qualify."""
    r = analyse(text, lang)
    if not r["ok"]:
        return None
    sp = [n for n in r["spans"] if not (n.weak and n.cls == "int")]
    spans = [{"text": text[n.a:n.b], "kind": KIND.get(n.cls, "int"), "cls": n.cls, "reading": text[n.a:n.b],
              "canonical": text[n.a:n.b], "value": str(n.value), "by": by} for n in sp]
    row = {"audio_filepath": audio_filepath, "duration": duration, "text": text, "lang": lang, "src": src, "id": rid,
           "origin": "numword", "caption": text, "spans": spans, "num_chars": [[n.a, n.b] for n in sp],
           "kinds": sorted({s["kind"] for s in spans}), "classes": r["classes"]}
    if extra:
        row.update(extra)
    return row
