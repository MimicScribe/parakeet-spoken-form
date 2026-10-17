"""Spoken-form number data for the 16 languages outside wave 1 and ru/uk (2026-10-16):
bg hr cs da et fi el hu lv lt mt pl ro sk sl sv.

Most of these languages inflect numbers by case, gender and number ("w 2025 roku" is said "w dwa tysiące dwudziestym
piątym roku"; "do 5 milionów euro" "do pięciu milionów euro"). A nominative number grammar would teach wrong
endings, so the readings come from elsewhere:

  A (real speech)  ml_multi_ctc.py: an LLM proposes the inflected readings of each caption number (each must parse
                   back to the caption value), a public CTC reader of the language picks the one spoken, and an
                   external transcript (which writes these languages' numbers as DIGITS, 2026-10-16 pilot) confirms
                   the value. `label_row` here also takes a transcript's own number WORDS where it has them (rare for
                   these languages); a span it cannot confirm is masked and the clip is cut around it.
  B (synthetic)    ml_tts_multi.py: shapes real speech lacks, written by an LLM from real number-bearing sentences;
                   `label_row` aligns the written sentence to the LLM's spoken form so every digit stretch must be
                   read by words that parse back to its value; voiced, then checked against an external transcript.

The value check is an inflection-tolerant number-word reader (`parse_numbers`) built on per-language word-form tables
(ml_multi_lex.json: every case / gender / number form of each numeral morpheme, cardinal and ordinal; month names in
ml_multi_months.json). It reads values only: an unknown form makes a span unreadable (masked), it never changes the
value of a readable one.

Pure functions only (no Modal here): the Modal side is ml_multi_vox.py.
"""

import json
import os
import re
import unicodedata
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_PATH = os.path.join(HERE, "ml_multi_lex.json")
L16 = ("bg", "hr", "cs", "da", "et", "fi", "el", "hu", "lv", "lt", "mt", "pl", "ro", "sk", "sl", "sv")
# languages that glue number morphemes into one written word ("tjugofem", "huszonöt", "kaksikymmentäviisi")
COMPOUND = {"sv", "da", "hu", "fi", "et", "el", "mt"}
UNIT_FIRST = {"da", "sl", "mt", "cs", "sk"}  # "femogtyve", "petindvajset", "ħamsa u għoxrin", "pětadvacet" = 5 + 20
# a spoken hundreds MULTIPLIER across words ("două sute", "dvě stě", "två hundra"); pl / bg / el have lexical
# hundreds only, so "trzy sto" there is two numbers. A multiplier inside one written word is always allowed.
MULT100 = {"sv", "da", "fi", "et", "hu", "ro", "cs", "sk", "sl", "hr", "lv", "lt", "mt"}
MULT10 = {"fi", "et"}  # "kaksi kymmentä" may be written apart; elsewhere x10 only inside one word ("divdesmit")

# ------------------------------------------------------------------ selection words (text-only class strata)
PCT = {"bg": r"процент\w*", "hr": r"posto|postot\w*", "cs": r"procent\w*", "da": r"procent", "et": r"protsen\w*",
       "fi": r"prosent\w*", "el": r"τοις\s+εκατό|ποσοστ\w*", "hu": r"százalék\w*", "lv": r"procent\w*",
       "lt": r"procent\w*", "mt": r"fil-mija|perċent\w*|percent\w*", "pl": r"procent\w*", "ro": r"la\s+sută|procent\w*",
       "sk": r"percent\w*", "sl": r"odstot\w*", "sv": r"procent\w*"}
CUR = (r"€|\$|£|eur\w*|евро|ευρώ|euró\w*|eiro|dolar\w*|dollar\w*|долар\w*|δολάρι\w*|dollár\w*|zł\b|złot\w*|lei\b|leu\b|"
       r"kr\b|kronor|kronur|kroner|krón\w*|forint\w*|ft\b|лв\b|лев\w*|kun\w*\b|kn\b|kč\b|korun\w*|frank\w*|liras?\b|"
       r"lir\w*|sterlin\w*|funt\w*|pound\w*|jen\w*")
SCALE = (r"mln\.?|mld\.?|mlrd\.?|mrd\.?|bln\.?|tys\.?|mil\.|mio\.?|mia\.?|milj\.?|милиона?|милиарда?|млн\.?|млрд\.?|"
         r"εκατ\w*|δισ\w*|milion\w*|miliard\w*|milijun\w*|milijard\w*|milijon\w*|miljon\w*|miljard\w*|millió\w*|"
         r"milliárd\w*|miljoon\w*|miljoni\w*|miljardi\w*|million\w*|milliard\w*|miljun\w*|biljun\w*|tusen|tusind|"
         r"tuhat\w*|tuhan\w*|tuhande\w*|ezer\w*|хиляд\w*|χιλιάδ\w*|tisuć\w*|tisoč\w*|tisíc\w*|tūkst\w*|tūkstan\w*")
TIME = {"bg": r"часа?|ч\.", "hr": r"sati|sat|h", "cs": r"hod\.?|hodin\w*|h", "da": r"kl\.?|klokken", "et": r"kell|kl\.?",
        "fi": r"klo|kello", "el": r"ώρα|η\s+ώρα", "hu": r"óra\w*|órakor", "lv": r"plkst\.?|pulksten\w*", "lt": r"val\.?|valand\w*",
        "mt": r"fil-|ħin", "pl": r"godz\.?|godzin\w*", "ro": r"ora|orele", "sk": r"hod\.?|hodin\w*|h", "sl": r"ur[aeio]?|h",
        "sv": r"kl\.?|klockan"}
MINUTE = {"bg": r"минут\w*", "hr": r"minut\w*", "cs": r"minut\w*", "da": r"minut\w*", "et": r"minut\w*",
          "fi": r"minuut\w*", "el": r"λεπτ\w*", "hu": r"perc\w*", "lv": r"minūt\w*", "lt": r"minut\w*", "mt": r"minut\w*",
          "pl": r"minut\w*", "ro": r"minut\w*", "sk": r"minút\w*", "sl": r"minut\w*", "sv": r"minut\w*"}
YEAR_RE = re.compile(r"(?<![\d.,])(1[89]\d\d|20\d\d)(?![\d%]|[.,]\d)")


_FOLD_EXTRA = str.maketrans({"ł": "l", "đ": "d", "ħ": "h", "ı": "i", "ø": "o", "æ": "ae", "ß": "ss"})


def fold(s: str) -> str:
    """Lowercase, NFKC, strip combining marks (ą -> a, ő -> o, ά -> α), typographic apostrophes -> '."""
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace("ʼ", "'").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if unicodedata.category(c) != "Mn")
    return s.replace("ς", "σ").translate(_FOLD_EXTRA)


def classes(text: str, lang: str) -> dict:
    """Variety strata of one caption (selection only, never labelling): year values (to 2099) and flags for clock /
    currency / large figure / decimal / percent / ordinal / range / phone."""
    t = text.lower()
    tw = TIME.get(lang, "h")
    years = [int(m.group(1)) for m in YEAR_RE.finditer(text)]
    clock = bool(re.search(r"(?:^|\s)(?:" + tw + r")\s+([01]?\d|2[0-3])(?:[.:]([0-5]\d))?(?![\d.,]\d)", t)
                 or re.search(r"(?<![\d.,])([01]?\d|2[0-3])[.:]([0-5]\d)(?![\d.,]\d)\s*(?:" + tw + r")\b", t)
                 or re.search(r"(?<![\d.,:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])", t))
    cur = bool(re.search(r"\d[\d.,  ]*\s?(?:(?:" + SCALE + r")\s?)?(?:de\s|of\s)?(?:" + CUR + r")", t)
               or re.search(r"(?:€|\$|£)\s?\d", t))
    big = bool(re.search(r"(?<![\d.,])\d{1,3}(?:[.,  ]\d{3})+(?![\d,.]\d)", text)
               or re.search(r"\d\s?(?:" + SCALE + r")(?!\w)", t) or re.search(r"(?<![\d.,])\d{5,}(?![\d.,])", text))
    dec = bool(re.search(r"(?<![\d.,])\d+[.,]\d{1,2}(?![\d.,])", text))
    pct = "%" in text or bool(re.search(r"\d\s?(?:" + PCT.get(lang, "procent") + r")", t))
    rng = bool(re.search(r"\d\s?[-–]\s?\d", text))
    ordn = bool(re.search(r"(?<![\d.,])\d{1,3}\.\s+[a-zà-žα-ωа-я]", text)) or bool(
        re.search(r"\d+(?:-?(?:lea|a|ος|η|ο|ти|ри|ви|ми|ий|та|то|ят|е|:e|:a))\b", t))
    phone = bool(re.search(r"(?<!\d)(?:\+\d{2,3}\s?)?\d{2,4}(?:[\s-]\d{2,4}){2,}(?!\d)", text)) and not big
    return {"years": years, "clock": clock, "cur": cur, "big": big, "dec": dec and not clock, "pct": pct, "range": rng,
            "ord": ordn, "phone": phone}


def kinds(text: str, lang: str) -> list[str]:
    c = classes(text, lang)
    out = ["year"] if c["years"] else []
    out += [k for k in ("clock", "cur", "big", "dec", "pct", "range", "ord", "phone") if c[k]]
    if not out and re.search(r"\d", text):
        out = ["int"]
    return out


# ------------------------------------------------------------------ number-word reader (values only)
HALF15 = {"poltora", "poltorej", "poltorak", "masfel", "halvanden", "halvannan", "puolitoista", "poolteist", "pusantro",
          "pusantros", "pusantrų", "pusantru", "pusotra", "pusotru", "pusotrog", "jedenapol", "ενάμιση", "ενάμισι",
          "ενάμισυ", "ενάμισης", "ενάμιση", "ενάμιση", "pustrecias"}
SCALE_WORDS = (100, 1000, 10**6, 10**9, 10**12)
_TAB = None
_CACHE: dict = {}
# closed-class words the drafts miss (value-safe: separators / connectors only)
EXTRA_DEC = {"cs": ["celá", "celé", "celých", "celou"], "sk": ["celá", "celé", "celých"], "sl": ["cela", "cele", "celih", "vejica"],
             "hr": ["cijela", "cijele", "cijelih", "zarez"], "bg": ["цяло", "цели", "запетая"], "lt": ["kablelis"],
             "lv": ["komats"], "et": ["koma"], "el": ["κόμμα"], "mt": ["punt"]}
EXTRA_GLUE = {"ro": ["de"], "bg": ["и"], "mt": ["u"], "da": ["og"], "sl": ["in"], "hr": ["i"]}
# a half word counts only in the language's own construction "<number> <and> [a] <half>" ("dwa i pół", "tre och en
# halv", "kolme ja puoli"); a bare "viis poolt" / "kolme puolella" / "tre halv" is not 3.5 (parliament text: "for",
# "on the side of", "half past"). HALF_GLUE adds the "and" words that are not between-number connectors elsewhere.
HALF_GLUE = {"fi": ["ja"], "et": ["ja"], "cs": ["a"], "sv": ["och"], "lt": ["ir", "su"], "lv": ["un", "ar"],
             "hu": ["és"], "da": ["og"], "mt": ["u"], "bg": ["и"], "hr": ["i"], "sk": ["a"], "sl": ["in"],
             "pl": ["i"], "ro": ["și"], "el": ["και"]}
HALF_ARTICLE = {"sv": ["en", "ett"], "da": ["en", "et"], "et": [], "hu": ["egy"]}
# words that are never read as numbers although they fold or stem onto number forms. Exact lowercase tokens and
# prefixes (a prefix covers the case endings of a weekday); "cased" ones count only when written with a capital
# after the first word (a Greek weekday is the feminine ordinal, a Maltese weekday the cardinal).
STOP_WORDS = {
    "hr": ["što"], "cs": ["ste"], "sk": ["ste"], "hu": ["egyetem", "hatóság"], "da": ["treenighed"],
    "fi": ["toista"],  # "other" (partitive of "toinen"); the teen suffix is read inside "viidestoista" etc.
}
STOP_STEMS = {
    "pl": ["czwartek", "czwartk", "piątek", "piątk"],
    "cs": ["pátek", "pátk", "čtvrtek", "čtvrtk"],
    "sk": ["piatok", "piatk", "štvrtok", "štvrtk"],
    "hr": ["petak", "petk", "četvrtak", "četvrtk"],
    "sl": ["četrtek", "četrtk", "petek", "petk"],
    "bg": ["четвъртък", "четвъртъц", "петък", "петъц", "стотинк"],  # "стотинки" (cents) folds onto "сто"
    "et": ["esmaspäev", "teisipäev", "kolmapäev", "neljapäev"],
    "hu": ["hétfő", "hétvég", "egyetem", "hatóság"],
    "lt": ["pirmadien", "antradien", "trečiadien", "ketvirtadien", "penktadien", "šeštadien"],
    "lv": ["pirmdien", "otrdien", "trešdien", "ceturtdien", "piektdien", "sestdien"],
    "da": ["treenighed"],
}
STOP_CAPITAL = {
    "el": ["τρίτη", "τετάρτη", "πέμπτη", "δευτέρα", "τρίτης", "τετάρτης", "πέμπτης", "δευτέρας"],
    "mt": ["tnejn", "tlieta", "erbgħa"],
}
# forms valid only straight after another number word ("pět set"); alone they are a different word ("set")
AFTER_NUMBER_ONLY = {"cs": ["set"], "sk": ["set"]}
# case / number / gender endings a number word may carry although the table does not list that exact form; the
# listed forms' own tails (Lex.endings) are added to these
COMMON_ENDINGS = {
    "hu": "ben ban ba be ból ből ról ről tól től hoz hez höz nak nek val vel ig ra re on en ön n t at et ot öt ért kor "
          "ként ik dik odik edik adik ödik s i ek ak ok ök k",
    "fi": "ssa ssä sta stä lla llä lta ltä lle ksi na nä a ä ta tä n en in iin ihin seen ten tta ttä t",
    "et": "s st sse l lt le ks ni na ta t d de te ga sid id i",
    "da": "e ene s er ende de te nde",
    "sv": "e a de te nde s",
    "el": "ς ν ο η ου ων υ α ες ους ης ι ε ω",
    "ro": "a le lea lui ul ii i e ă lea",
    "mt": "a i ijiet in",
    "bg": "а и то та ата те ите ия",
    "lv": "s a u i as ām ās ā os iem ais ajam ajā ajiem ajām ajos ajai o am",
    "lt": "s as is i ai ais ams ų ių ui iui u ą ę į e ė ės ėje ie iems iesiems",
}
for _l in ("pl", "cs", "sk", "sl", "hr"):
    COMMON_ENDINGS[_l] = ("ego emu ym im ych ich ymi imi ej ą ę y i a e o u ma mi em ami ach ów om ie ia ią iej iu ého ému "
                          "ým ou ách ami ům ům ah ih ega mu im ime")
MT_ARTICLES = {"l", "il", "it", "id", "ir", "is", "in", "ix", "iz", "ic", "ik", "iħ"}
NOT_NUMBER = {"ro": [("la", "suta")], "el": [("τοισ", "εκατο")], "mt": [("fil-mija",), ("fil", "mija")]}


def _cases_module():
    """The curated paradigm module, or None when it is absent; an ImportError raised INSIDE it propagates."""
    try:
        import ml_multi_cases
    except ModuleNotFoundError as e:
        if e.name != "ml_multi_cases":
            raise
        return None
    return ml_multi_cases


def _table() -> dict:
    global _TAB
    if _TAB is None:
        if os.path.exists(LEX_PATH):
            with open(LEX_PATH) as f:
                _TAB = json.load(f)
        else:
            _TAB = {}
    return _TAB


class Lex:
    """Folded word forms of one language: form -> {(value, is_ordinal)}, plus prefix stems (forms of 6+ letters
    give a stem of max(5, len-3) letters that also matches an unlisted ending of <= 4 letters, used only when one
    value owns the longest matching stem)."""

    def __init__(self, lang: str):
        t = dict(_table().get(lang, {}))
        cases = _cases_module()
        if cases is not None:  # the curated paradigm forms of ml_multi_cases (pl, ro) are read too, so every generated reading parses
            extra = cases.extra_forms(lang)
            owner = {w: v for vs in extra.values() for v, ws in vs.items() for w in ws}
            for kind in ("cardinal", "ordinal"):
                # a curated form owns its value: a drafted entry listing it under another value is dropped there
                # (pl drafts had "trzydziestoma" under 13)
                merged = {k: [w for w in v if owner.get(w, k) == k] for k, v in (t.get(kind) or {}).items()}
                for v, ws in (extra.get(kind) or {}).items():
                    merged[v] = sorted(set(merged.get(v, [])) | set(ws))
                t[kind] = merged
        self.lang = lang
        self.forms: dict = {}
        multi = []
        for kind, is_ord in (("cardinal", False), ("ordinal", True)):
            for v, ws in (t.get(kind) or {}).items():
                for w in ws:
                    parts = [p for p in re.split(r"[\s]+", fold(w)) if p]
                    if len(parts) == 1:
                        for p in parts[0].split("-"):  # "kétezer-huszonöt": each half is a form of its own value
                            p = p.strip("'")  # the reader strips edge apostrophes from tokens ("raba'")
                            if p and "-" not in parts[0]:
                                self.forms.setdefault(p, set()).add((int(v), is_ord))
                    else:
                        multi.append((parts, int(v), is_ord))
        # multi-word entries ("două sute", "o mie", "al doilea"): a part is added only when its value follows from
        # the other parts (multiplier / scale: "sute" = 200 / "două"); article-like parts are ignored
        single = {f: {x for x, _ in vs} for f, vs in self.forms.items()}
        for parts, v, is_ord in multi:
            unknown = [p for p in parts if p not in single]
            if len(parts) == 2 and len(unknown) == 1 and parts[1] == unknown[0] and len(single[parts[0]]) == 1:
                k = next(iter(single[parts[0]]))
                if k and v % k == 0 and v // k in (10, 100, 1000, 10**6, 10**9, 10**12):
                    self.forms.setdefault(parts[1], set()).add((v // k, is_ord))
        self.forms = {f: vs for f, vs in self.forms.items() if len({x for x, _ in vs}) == 1}
        self.stems: dict = {}
        for f, vs in self.forms.items():
            if len(f) >= 6:
                self.stems.setdefault(f[:max(5, len(f) - 3)], set()).update(v for v, _ in vs)
        last = lambda w: fold(w).split()[-1] if fold(w).split() else ""  # noqa: E731
        self.glue = {fold(w) for w in t.get("connector", []) + EXTRA_GLUE.get(lang, []) if " " not in fold(w)}
        self.dec = ({fold(w) for w in t.get("decimal", []) + EXTRA_DEC.get(lang, []) if " " not in fold(w)}
                    - self.glue - set(self.forms))
        # a glue or decimal word is never a fraction denominator (Maltese "u" is "and", not the 1000 of "u elf sulf")
        self.frac = {fold(w): int(den) for den, ws in (t.get("fraction") or {}).items() for w in ws
                     if " " not in fold(w) and fold(w) not in self.forms and fold(w) not in self.glue
                     and fold(w) not in self.dec}
        # the unlisted endings a number word may carry: what the listed forms of one value add to a shared stem
        # ("dwudziest" + "ego" / "ym"); an unlisted tail outside this set does not make a longer word a number
        self.endings = set()
        by_val: dict = {}
        for f, vs in self.forms.items():
            for v, _ in vs:
                by_val.setdefault(v, []).append(f)
        for fs in by_val.values():
            for a in fs:
                for b in fs:
                    n = _lcp(a, b)
                    if a != b and n >= 2 and 0 < len(b) - n <= 5:
                        self.endings.add(b[n:])
        self.endings |= {fold(e) for e in COMMON_ENDINGS.get(lang, "").split()}
        self.half = set()
        for w in t.get("half", []):
            ps = fold(w).split()
            if len(ps) == 1 or (len(ps) == 2 and ps[0] in self.glue):  # "pół", "și jumătate"; never "pół roku"
                self.half.add(ps[-1])
        self.half -= HALF15 | self.glue
        self.minus = {fold(w) for w in t.get("minus", [])}
        self.max_form = max((len(f) for f in self.forms), default=0)
        self.pct = re.compile(fold(PCT.get(lang, "procent\\w*")))

    def stem_value(self, s: str):
        best, val = 0, None
        for k in range(len(s), 4, -1):
            vs = self.stems.get(s[:k])
            if vs and len(s) - k <= 4 and s[k:] in self.endings:
                if len(vs) == 1 and k > best:
                    return next(iter(vs)), False
                return None
        return val

    def morphs(self, tok: str):
        """Morpheme values of one folded word ([(value, is_ord), ...]) or None when it is not wholly a number word.
        Compounds are segmented into listed forms (fewest pieces); the last piece may carry an unlisted ending of
        <= 4 letters (case suffixes: "huszonötben", "dwudziestym")."""
        if tok in self.forms:
            v, o = min(self.forms[tok])
            return [(v, o)]
        if not tok.isalpha() or len(tok) < 3:
            return None
        key = tok
        if key in _CACHE.get(self.lang, {}):
            return _CACHE[self.lang][key]
        memo: dict = {}

        def go(i):
            """(cost, pieces) for tok[i:]: listed forms (cost 1 each), a connector inside a compound, and at most
            one unlisted case ending of <= 5 letters after the last form (cost 0.6 per ending letter, so listed forms win; the form before it must have 4+
            letters at the start of the word, 2+ inside a compound)."""
            if i == len(tok):
                return (0, [])
            if i in memo:
                return memo[i]
            best = None
            for j in range(min(len(tok), i + self.max_form), i + 1, -1):
                w = tok[i:j]
                cands = []
                if w in self.forms:
                    rest = go(j)
                    if rest is not None:
                        cands.append((1 + rest[0], [min(self.forms[w])] + rest[1]))
                    if (j < len(tok) and len(tok) - j <= 5 and tok[j:] in self.endings
                            and (len(w) >= 4 or (i > 0 and len(w) >= 2))):
                        cands.append((1 + 0.6 * (len(tok) - j), [min(self.forms[w])]))
                elif w in self.glue and i > 0 and j < len(tok):
                    rest = go(j)
                    if rest is not None:
                        cands.append((0.5 + rest[0], [("glue", False)] + rest[1]))
                for c in cands:
                    if best is None or c[0] < best[0]:
                        best = c
            if best is None:
                sv = self.stem_value(tok[i:])
                if sv is not None and (i > 0 or len(tok) >= 6):
                    best = (2, [(sv[0], False)])
            memo[i] = best
            return best

        res = go(0)
        res = res[1] if res is not None else None
        if res is not None and (not res or res[0][0] == "glue" or res[-1][0] == "glue"):
            res = None
        _CACHE.setdefault(self.lang, {})[key] = res
        return res


_LEXES: dict = {}


def lexicon(lang: str) -> Lex:
    if lang not in _LEXES:
        _LEXES[lang] = Lex(lang)
    return _LEXES[lang]


class Num:
    __slots__ = ("value", "start", "end", "ordinal", "words")

    def __init__(self, value, start, end, ordinal=False, words=()):
        self.value, self.start, self.end, self.ordinal, self.words = value, start, end, ordinal, tuple(words)

    def __repr__(self):
        return f"Num({self.value}{' ord' if self.ordinal else ''} {' '.join(self.words)!r})"


def _lcp(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _low(n: int) -> int:
    if n == 0:
        return 10**18
    k = 1
    while n % (k * 10) == 0:
        k *= 10
    return k


def _combine(lang: str, pieces: list) -> list:
    """pieces: [(value | "glue", is_ord, tok_index)] of one run -> [(int value, start_tok, end_tok, is_ord)].
    Additive place-value grammar: multiplier 10 / 100 after a smaller piece ("dwie stě", "kaksikymmentä",
    "nittonhundra"), scales >= 1000 in falling order, unit-first tens after a connector ("femogtyve",
    "petindvajset", "ħamsa u għoxrin"). A piece that cannot continue the number starts a new one."""
    out = []
    st = {}

    def reset(v=None, ti=None, is_ord=False):
        st.update(total=0, cur=0, lim=10**18, piece=0, unit=None, glue=False, start=ti, end=ti, ord=is_ord)
        if v is not None:
            add(v)

    def add(v):
        if v >= 1000:
            st["total"], st["lim"], st["cur"], st["piece"], st["unit"] = (st["total"] + (st["cur"] or 1) * v, v, 0, 0, None)
        else:
            st["cur"] += v
            st["piece"] += v
            st["unit"] = v if v < 10 else None

    def close():
        if st.get("start") is not None:
            out.append((st["total"] + st["cur"], st["start"], st["end"], st["ord"]))

    reset()
    for v, is_ord, ti in pieces:
        if v == "glue":
            st["glue"] = st["start"] is not None
            continue
        if st["start"] is None:
            reset(v, ti, is_ord)
            continue
        ok = True
        if v == 0 or st["total"] + st["cur"] == 0:  # "zero" stands alone (digit-by-digit readings)
            ok = False
        elif v >= 1000:
            if v < st["lim"]:
                add(v)
            else:
                ok = False
        elif (v in (10, 100) and 0 < st["piece"] < v and not st["glue"]
              and (ti == st["end"] or (lang in (MULT10 if v == 10 else MULT100) and st["piece"] <= 19))):
            st["cur"] += st["piece"] * v - st["piece"]
            st["piece"], st["unit"] = 0, None
        elif st["glue"] and lang in UNIT_FIRST and st["unit"] is not None and 20 <= v <= 90 and v % 10 == 0:
            st["cur"] += v
            st["piece"] += v
            st["unit"] = None
        elif v < _low(st["cur"]) and st["cur"] + v < min(st["lim"], 10**4):
            add(v)
        else:
            ok = False
        if ok:
            st["end"], st["ord"], st["glue"] = ti, is_ord, False
        else:
            close()
            reset(v, ti, is_ord)
    close()
    return out


def _token_kind(lx: "Lex", w: str):
    """("num", morphs) | ("glue",) | ("dec",) | ("half",) | ("den", 10**k) | ("digits", str) | None for a folded word."""
    if re.fullmatch(r"\d+", w):
        return ("digits", w)
    if w in HALF15:
        return ("num15",)
    if w in lx.frac:
        return ("den", lx.frac[w])
    if lx.pct.fullmatch(w):
        return None
    parts = [p for p in w.split("-") if p]
    if lx.lang == "mt" and len(parts) > 1 and parts[0] in MT_ARTICLES:  # "l-ewwel", "it-tieni", "ir-raba'"
        parts = parts[1:]
    ms = []
    for p in parts:
        m = lx.morphs(p)
        if m is None:
            ms = None
            break
        ms += m
    if ms:
        return ("num", ms)
    if w in lx.glue:
        return ("glue",)
    if w in lx.dec:
        return ("dec",)
    if w in lx.half:
        return ("half",)
    if w in lx.frac:
        return ("den", lx.frac[w])
    if w in HALF15:
        return ("num15",)
    return None


MONTHS_PATH = os.path.join(HERE, "ml_multi_months.json")
_MONTHS: dict = {}


def month_forms(lang: str) -> dict:
    if not _MONTHS and os.path.exists(MONTHS_PATH):
        with open(MONTHS_PATH) as f:
            for L, ms in json.load(f).items():
                _MONTHS[L] = {fold(w): int(m) for m, ws in ms.items() for w in ws}
    return _MONTHS.get(lang, {})


def _gate_halves(lx: "Lex", lang: str, ft: list, kinds_: list) -> list:
    """Keep a half word only after "<number> <and> [article]"; an article between "and" and the half word ("och en
    halv") becomes a connector so the number run stays whole."""
    glue = lx.glue | {fold(w) for w in HALF_GLUE.get(lang, ())}
    arts = {fold(w) for w in HALF_ARTICLE.get(lang, ())}
    out = list(kinds_)
    for i, k in enumerate(kinds_):
        if not (k and k[0] == "half"):
            continue
        a = i - 1
        if a >= 0 and ft[a] in arts and a >= 1 and ft[a - 1] in glue:
            a -= 1
        ok = a >= 1 and ft[a] in glue and kinds_[a - 1] is not None and kinds_[a - 1][0] == "num"
        if not ok:
            out[i] = None
        else:
            out[a] = ("glue",)
            if a < i - 1:
                out[i - 1] = ("glue",)
    return out


# "a few / several / many" before a scale word: "nekaj tisoč" is not 1000
_VAGUE_RAW = ("nekaj nekoliko par kilka kilkanascie kilkadziesiat pare wiele nekolik niekolko mnoho cateva cativa "
              "nagra flera manga nogle flere mange muutama useita monta nehany tobb moni mitu keli kelios keletas dazi "
              "dazas vairaki vairakas няколко μερικα μερικες μερικοι ftit")


def _count_like(lx, w: str) -> bool:
    """An unknown word that starts like a 2-9 cardinal form of the language and is about as long ("dva" for "dve",
    "trys" for "trys"): plausibly a count the tables lack, not a function word."""
    units = getattr(lx, "_units", None)
    if units is None:
        units = lx._units = {f for f, vs in lx.forms.items() if any(2 <= v <= 9 and not o for v, o in vs)}
    return any(w[:2] == f[:2] and abs(len(w) - len(f)) <= 2 for f in units)


def parse_numbers(text: str, lang: str, months: bool = False) -> list:
    """Every number in `text` (words or digits) as Num(Decimal value, start token, end token). A decimal word joins
    an integer and what is read after it ("trzy przecinek zero pięć" = 3.05; "három egész öt tized" = 3.5 with a
    fraction denominator; "trzy przecinek pięć miliona" = 3 500 000); a half word adds 0.5; "półtora" = 1.5."""
    lx = lexicon(lang)
    toks = text.split()
    ft = [re.sub(r"[^\w'-]", "", fold(t)).strip("-'") for t in toks]
    kinds_ = [_token_kind(lx, w) for w in ft]
    if months:  # a numeric date "17.8.1951" is read "siedemnastego sierpnia ...": the month word is its middle value
        mf = month_forms(lang)
        kinds_ = [("digits", str(mf[w])) if w in mf else k for w, k in zip(ft, kinds_)]
    raw_low = [re.sub(r"[^\w'-]", "", unicodedata.normalize("NFC", t)).strip("-'") for t in toks]
    for i, w in enumerate(raw_low):
        low = w.lower()
        if (low in STOP_WORDS.get(lang, ()) or any(low.startswith(st) and len(low) - len(st) <= 4
                                                    for st in STOP_STEMS.get(lang, ()))):
            kinds_[i] = None
        seg = w.split("-")[-1]
        if seg[:1].isupper() and seg.lower() in STOP_CAPITAL.get(lang, ()) and (i > 0 or "-" in w):
            kinds_[i] = None
        if ft[i] in AFTER_NUMBER_ONLY.get(lang, ()) and not (i > 0 and kinds_[i - 1] and kinds_[i - 1][0] == "num"):
            kinds_[i] = None
    kinds_ = _gate_halves(lx, lang, ft, kinds_)
    for ph in NOT_NUMBER.get(lang, ()):  # "la sută" / "τοις εκατό" / "fil-mija" are the percent word, not 100
        n = len(ph)
        for i in range(len(ft) - n + 1):
            if ft[i:i + n] == list(ph):
                for k in range(i, i + n):
                    kinds_[k] = None
    nums: list = []
    i = 0
    while i < len(ft):
        k = kinds_[i]
        if k and k[0] == "digits":
            nums.append(Num(Decimal(k[1]), i, i, False, toks[i:i + 1]))
            i += 1
            continue
        if k and k[0] == "num15":
            sc = kinds_[i + 1][1][0][0] if i + 1 < len(ft) and kinds_[i + 1] and kinds_[i + 1][0] == "num" else 0
            if sc >= 1000:  # "półtora miliona"
                nums.append(Num(Decimal("1.5") * sc, i, i + 1, False, toks[i:i + 2]))
                i += 2
                continue
            nums.append(Num(Decimal("1.5"), i, i, False, toks[i:i + 1]))
            i += 1
            continue
        if not k or k[0] != "num":
            i += 1
            continue
        # one run: number words, connectors between number words, one decimal part, a half word
        j, left = i, []
        while j < len(ft):
            kj = kinds_[j]
            if kj and kj[0] == "num":
                left += [(v, o, j) for v, o in kj[1]]
                j += 1
                if re.search(r"[,;:.!?]$", toks[j - 1]):  # "noll sjuttio, hundrasjuttionio": a comma ends the number
                    break
            elif kj and kj[0] == "glue" and j + 1 < len(ft) and kinds_[j + 1] and kinds_[j + 1][0] in ("num", "half"):
                left.append(("glue", False, j))
                j += 1
            else:
                break
        got = [list(x) for x in _combine(lang, left)]
        # decimal part
        if got and j + 1 < len(ft) and kinds_[j] and kinds_[j][0] == "dec" and kinds_[j + 1] and kinds_[j + 1][0] == "num":
            m, right = j + 1, []
            while m < len(ft) and kinds_[m] and (kinds_[m][0] == "num" or (
                    kinds_[m][0] == "glue" and right and m + 1 < len(ft) and kinds_[m + 1] and kinds_[m + 1][0] == "num")):
                right += [(v, o, m) for v, o in kinds_[m][1]] if kinds_[m][0] == "num" else [("glue", False, m)]
                m += 1
            scale = 1
            while right and right[-1][0] == "glue":
                right.pop()
            if right and right[-1][0] >= 1000:  # "trzy przecinek pięć miliona"
                scale = right[-1][0]
                while right and right[-1][0] != "glue" and right[-1][0] >= 1000:
                    right.pop()
            fr = _combine(lang, right)
            frac = "".join(str(v) for v, *_ in fr)
            end = m - 1
            if m < len(ft) and kinds_[m] and kinds_[m][0] == "den":
                frac, end = frac.zfill(len(str(kinds_[m][1])) - 1), m
                m += 1
            if frac:
                v0, a0, _, _ = got[-1]
                got[-1] = [Decimal(f"{v0}.{frac}") * scale, a0, end, False]
                j = m
        h = j
        while h < min(len(ft), j + 2) and kinds_[h] and kinds_[h][0] == "glue":  # "<and> [article] <half>"
            h += 1
        if got and h < len(ft) and kinds_[h] and kinds_[h][0] == "half":
            got[-1][0] = Decimal(got[-1][0]) + Decimal("0.5")
            got[-1][2], j = h, h + 1
            sc = kinds_[j][1][0][0] if j < len(ft) and kinds_[j] and kinds_[j][0] == "num" else 0
            if sc >= 1000 and got[-1][0] < 1000:  # "dwa i pół miliona"
                got[-1][0] *= sc
                got[-1][2], j = j, j + 1
        # a hundred / thousand / million word straight after an alphabetic word that is not a number word may be the
        # tail of a count the tables lack ("dva tisoč" with an unknown "dva"): reading the scale word alone would
        # change the value, so that number is left unread (the span is masked)
        # change the value, so that number is left unread (the span is masked). Only when the word looks like a count
        # the tables lack (it starts like a 2-9 cardinal form): "w roku tysiąc", "več kot tisoč" read normally.
        refuse_first = (k[1][0][0] in SCALE_WORDS and i > 0 and kinds_[i - 1] is None and len(ft[i - 1]) > 1
                        and ft[i - 1].replace("-", "").isalpha() and not re.search(r"[,;:.!?]$", toks[i - 1])
                        and (_count_like(lx, ft[i - 1]) or ft[i - 1] in _VAGUE))
        for v, a, b, o in got:
            if refuse_first and a == i:
                continue
            nums.append(Num(Decimal(v), a, b, o, toks[a:b + 1]))
        i = max(j, i + 1)
    return nums


# ------------------------------------------------------------------ caption side: written values of a span
SCALE_ABBR = {"tys": 10**3, "tys.": 10**3, "tis.": 10**3, "tūkst.": 10**3, "tuh.": 10**3, "mln": 10**6, "mln.": 10**6,
              "mil.": 10**6, "mio.": 10**6, "mio": 10**6, "milj.": 10**6, "млн": 10**6, "млн.": 10**6, "εκατ.": 10**6,
              "mld": 10**9, "mld.": 10**9, "mlrd": 10**9, "mlrd.": 10**9, "mrd": 10**9, "mrd.": 10**9, "mia.": 10**9,
              "mia": 10**9, "млрд": 10**9, "млрд.": 10**9, "bn": 10**9, "δισ.": 10**9}


def written_values(toks: list, lang: str = "") -> list:
    """Alternative value sequences a caption span can mean (each a list of Decimals). Handles thousands grouping
    ("1 500", "1.500", "1,500" when 3 digits follow), decimal comma/point, scale abbreviations ("5 mln"), clock /
    score pairs ("15:30", "15.30" -> [15, 30] and [15] for ":00")."""
    s = " ".join(toks)
    s = s.replace(" ", " ").replace(" ", " ")
    # join space-grouped thousands: "1 500 000" -> "1500000"
    joined = re.sub(r"(?<![\d.,])(\d{1,3})((?: \d{3})+)(?!\d)", lambda m: m.group(1) + m.group(2).replace(" ", ""), s)
    if not re.search(r"\d", joined):
        return []
    alts = [[]]
    for m in re.finditer(r"\d+(?:[.,:']\d+)*", joined):
        g = m.group(0)
        after = joined[m.end():].lstrip()
        nxt = after.split(" ")[0].lower() if after else ""
        if nxt == "de" and len(after.split(" ")) > 1:  # Romanian "20 de milioane": the scale word follows "de"
            nxt = after.split(" ")[1].lower()
        scale = SCALE_ABBR.get(nxt) or SCALE_ABBR.get(nxt.rstrip(",;:"))
        if not scale and lang and nxt:  # "750 miliardów", "60 miljoonan": a scale WORD after the digits
            ms = lexicon(lang).morphs(_key(nxt))
            if ms and len(ms) == 1 and ms[0][0] >= 1000:
                scale = ms[0][0]
        opts = []
        parts = re.split(r"[.,:']", g)
        seps = re.findall(r"[.,:']", g)
        if not seps:
            opts.append([Decimal(g)])
            if len(g) == 4 and g[0] in "12" and g[2] != "0":  # a year read as two pairs ("Horyzont dwadzieścia dwadzieścia")
                opts.append([Decimal(g[:2]), Decimal(g[2:])])
        else:
            if all(len(p) == 3 for p in parts[1:]) and len(set(seps)) == 1 and seps[0] in ".,'":  # thousands grouping
                opts.append([Decimal("".join(parts))])
            if len(parts) == 2 and seps[0] in ".,":
                opts.append([Decimal(parts[0] + "." + parts[1])])
            if len(parts) == 2 and len(parts[1]) == 2 and seps[0] in ".:":  # clock / score / money pair
                opts.append([Decimal(parts[0]), Decimal(parts[1])])
                if parts[1] == "00":
                    opts.append([Decimal(parts[0])])
            if len(parts) == 2 and seps[0] == ",":  # money "5,50" read as 5 ... 50
                opts.append([Decimal(parts[0]), Decimal(parts[1])])
            if (len(parts) >= 3 and not all(len(p) == 3 for p in parts[1:])) or (len(parts) == 2 and seps[0] == "."):
                opts.append([Decimal(p) for p in parts])  # dates / versions: each group
        if scale:
            opts = [o[:-1] + [o[-1] * scale] for o in opts] + opts
        alts = [a + o for a in alts for o in opts][:64]
    return alts


def spoken_values(nums: list) -> list:
    """Alternative value sequences of parsed spoken numbers: as read, plus "X <currency> Y" money pairs joined
    (X.YY) and a "hour minute" pair kept apart (both are compared against written_values)."""
    vals = [n.value for n in nums]
    alts = [vals]
    if len(vals) == 2 and vals[1] < 100 and vals[1] == int(vals[1]):
        alts.append([Decimal(f"{int(vals[0])}.{int(vals[1]):02d}")] if vals[0] == int(vals[0]) else vals)
    split = []  # "osiem przecinek sześć przecinek cztery" for a version 8.6.4: each group
    for v in vals:
        t = str(v)
        if "." in t and not t.split(".")[1].startswith("0"):
            split += [Decimal(t.split(".")[0]), Decimal(t.split(".")[1])]
        else:
            split.append(v)
    if split != vals:
        alts.append(split)
    return alts


def digits_parse(spoken: str, lang: str, target: str) -> bool:
    """True when the spoken number words can be cut into consecutive numbers whose digits spell `target` (phone
    numbers / codes read in any grouping: "patruzeci șapte sute cincizeci și șase" = "40" "756")."""
    lx = lexicon(lang)
    pieces = []
    for ti, t in enumerate(spoken.split()):
        k = _token_kind(lx, re.sub(r"[^\w'-]", "", fold(t)).strip("-'"))
        if k and k[0] == "num":
            pieces += [(v, o, ti) for v, o in k[1]]
        elif k and k[0] == "digits":
            return False
    memo = {}

    def go(i, j):
        if i == len(pieces):
            return j == len(target)
        if (i, j) in memo:
            return memo[(i, j)]
        ok = False
        for k in range(i + 1, min(len(pieces), i + 6) + 1):
            c = _combine(lang, pieces[i:k])
            if len(c) == 1:
                d = str(c[0][0])
                if target.startswith(d, j) and go(k, j + len(d)):
                    ok = True
                    break
        memo[(i, j)] = ok
        return ok

    return bool(pieces) and go(0, 0)


def values_match(cap_toks: list, spoken: str, lang: str) -> tuple:
    """(match?, parsed Nums) for one caption span vs spoken words."""
    written = " ".join(cap_toks)
    is_date = bool(re.search(r"(?<!\d)\d{1,2}[./]\d{1,2}[./](?:\d{2}|\d{4})(?!\d)", written))
    nums = parse_numbers(spoken, lang, months=is_date)
    w = written_values(cap_toks, lang)
    for a in spoken_values(nums):
        if any(x == a for x in w):
            return True, nums
    # phone numbers / codes: the digit string, whatever the grouping ("512 345 678" = "pięćset dwanaście ...",
    # "07260" = "zero siedem dwa sześć zero"); only for spans written as digit groups or with a leading zero
    groups = re.findall(r"\d+", written)
    money_or_dec = bool(re.search(r"\d[.,]\d{1,2}(?!\d)", written)) or bool(re.search(CUR, written.lower()))
    if nums and not money_or_dec and all(n.value == int(n.value) and n.value >= 0 for n in nums) and (
            len(groups) >= 3 or re.match(r"\D*0\d", written) or re.search(r"\d[ -]\d{2,4}[ -]\d", written)
            or (len(nums) >= 2 and all(n.value < 10 for n in nums))):  # read digit by digit / in groups
        if "".join(str(int(n.value)) for n in nums) == "".join(groups) or digits_parse(spoken, lang, "".join(groups)):
            return True, nums
    return False, nums


# ------------------------------------------------------------------ approach A: label a real clip
PUNCT_TAIL = re.compile(r"[.,;:!?)\]»\"”']+$")
PUNCT_HEAD = re.compile(r"^[(\[«\"„“']+")


def _key(t: str) -> str:
    return re.sub(r"[^\w]", "", fold(t))


def label_row(caption: str, ext_words: list, lang: str, max_extra: int = 1) -> dict:
    """Approach A. caption: the pseudo-label with digits; ext_words: the external transcript's words
    [{"w", "s", "e"}] (ms). Every caption stretch that holds a digit is aligned to the external transcript; when
    the external words there parse back to the caption's value(s), they replace the stretch (their own case /
    gender endings), else the stretch is MASKED (its time range is returned so the clip can be cut around it).

    Returns {"text": label with masked stretches as "\\u2047", "spans": [...accepted...], "masked": [...],
    "pieces": [(t0, t1, text)] (seconds; None when nothing is masked)}."""
    import difflib

    ctoks = caption.split()
    ckeys = [f"\x00{i}" if re.search(r"\d", t) else _key(t) for i, t in enumerate(ctoks)]
    ewords = [w for w in ext_words if _key(w["w"])]
    ekeys = [_key(w["w"]) for w in ewords]
    sm = difflib.SequenceMatcher(None, ckeys, ekeys, autojunk=False)
    blocks = []
    ops = sm.get_opcodes()
    k = 0
    while k < len(ops):
        tag, i1, i2, j1, j2 = ops[k]
        if tag != "equal" and any(re.search(r"\d", t) for t in ctoks[i1:i2]):
            # "90,05 lei" said "nouăzeci de lei și cinci bani": the currency word sits between the two parts
            while (k + 2 < len(ops) and ops[k + 1][0] == "equal" and ops[k + 1][2] - ops[k + 1][1] == 1
                   and ops[k + 2][0] in ("insert", "replace")
                   and not any(re.search(r"\d", t) for t in ctoks[ops[k + 2][1]:ops[k + 2][2]])
                   and parse_numbers(" ".join(w["w"] for w in ewords[ops[k + 2][3]:ops[k + 2][4]]), lang)
                   and ops[k + 2][2] - ops[k + 2][1] <= 1):
                i2, j2 = ops[k + 2][2], ops[k + 2][4]
                k += 2
            blocks.append((i1, i2, j1, j2))
        k += 1
    out_toks: list = []
    pos = 0
    spans, masked = [], []
    mask_ranges = []  # (caption token index, time t0, t1)
    for i1, i2, j1, j2 in blocks:
        out_toks += ctoks[pos:i1]
        cap = ctoks[i1:i2]
        ext = [w["w"] for w in ewords[j1:j2]]
        # the replacement must be WORDS: a transcript that writes the number as digits cannot label it
        ok, nums = values_match(cap, " ".join(ext), lang) if ext and not any(re.search(r"\d", x) for x in ext) else (False, [])
        num_tok = {k for n in nums for k in range(n.start, n.end + 1)}
        extra_ext = sum(1 for k in range(len(ext)) if k not in num_tok)
        cap_nondigit = sum(1 for t in cap if not re.search(r"\d", t)) + sum(
            1 for t in cap if re.search(r"\d", t) and re.search(r"[^\d.,:\s'\-–]", PUNCT_TAIL.sub("", PUNCT_HEAD.sub("", t))))
        # words around the numbers: a symbol or abbreviation is read as 1-2 words ("zł" -> "złote ... groszy",
        # "°C" -> "stopni Celsjusza"), a separator as a connector ("i", "kropka")
        seps = sum(len(re.findall(r"\d[.,:]\d", t)) for t in cap)
        sign_ok = bool(re.search(r"(?:^|\s)[-−–]\d", " " + " ".join(cap))) == any(
            _key(x) in lexicon(lang).minus for x in ext)
        # a clock time ("15:30", "10.05") is accepted only as "hour minute" with no relational word: "tio över fem"
        # (5:10) would otherwise pass as [10, 5]; hour words of the language ("klockan", "godzinie") are allowed
        clock_like = bool(re.search(r"(?<![\d.,])([01]?\d|2[0-3])[:.][0-5]\d(?![\d.,]\d)", " ".join(cap)))
        if clock_like and len(nums) >= 2:
            tw = re.compile(r"(?:" + fold(TIME.get(lang, "h")) + "|" + fold(MINUTE.get(lang, "minut")) + r")\w*")
            glue = lexicon(lang).glue
            rel = [x for k_, x in enumerate(ext) if k_ not in num_tok and not tw.fullmatch(_key(x)) and _key(x) not in glue]
            if rel:
                ok = False
        if ok and sign_ok and extra_ext <= 2 * cap_nondigit + 2 * seps + max_extra:
            rep = PUNCT_TAIL.sub("", " ".join(w.lower() for w in ext)).strip()
            head = PUNCT_HEAD.match(cap[0])
            tail = PUNCT_TAIL.search(cap[-1])
            sent_start = (i1 == 0 or re.search(r"[.!?]$", ctoks[i1 - 1]) is not None)
            if sent_start and rep:
                rep = rep[0].upper() + rep[1:]
            t = tail.group(0) if tail else ""
            if t.startswith(".") and i2 < len(ctoks) and ctoks[i2][:1].islower():
                t = t[1:]  # "5. maja": the ordinal dot is read, not punctuation
            out_toks.append((head.group(0) if head else "") + rep + t)
            spans.append({"cap": " ".join(cap), "words": rep, "values": [str(n.value) for n in nums],
                          "t": [ewords[j1]["s"] / 1000, ewords[j2 - 1]["e"] / 1000]})
        else:
            if ext:
                t0, t1 = ewords[j1]["s"] / 1000, ewords[j2 - 1]["e"] / 1000
            else:  # the external transcript has nothing there: cut between its neighbours
                t0 = ewords[j1 - 1]["e"] / 1000 if j1 > 0 else 0.0
                t1 = ewords[j1]["s"] / 1000 if j1 < len(ewords) else 1e9
            masked.append({"cap": " ".join(cap), "ext": " ".join(ext), "values_ext": [str(n.value) for n in nums],
                           "written": [[str(x) for x in a] for a in written_values(cap)[:3]], "t": [t0, t1]})
            mask_ranges.append((len(out_toks), t0, t1))
            out_toks.append("⁇")
        pos = i2
    out_toks += ctoks[pos:]
    text = " ".join(out_toks)
    pieces = None
    if masked:
        pieces = []
        start_tok, t_start = 0, 0.0
        for k, (ti, t0, t1) in enumerate(mask_ranges):
            # the mask's neighbours in the external transcript bound the cut (small guard either side)
            pieces.append((t_start, max(t_start, t0 - 0.05), " ".join(out_toks[start_tok:ti])))
            start_tok, t_start = ti + 1, t1 + 0.05
        pieces.append((t_start, None, " ".join(out_toks[start_tok:])))
    return {"text": text, "spans": spans, "masked": masked, "pieces": pieces, "blocks": len(blocks)}


def word_agreement(caption_label: str, ext_words: list) -> float:
    """Word-level agreement (1 - WER) of a finished label against the external transcript, punctuation and case
    folded; the number words are part of it."""
    import difflib

    a = [_key(t) for t in caption_label.split() if _key(t)]
    b = [_key(w["w"]) for w in ext_words if _key(w["w"])]
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return max(0.0, 1 - errs / len(a))


_VAGUE = {fold(w) for w in _VAGUE_RAW.split()}
