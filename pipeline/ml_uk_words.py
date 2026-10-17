"""Ukrainian number words -> values (2026-10-17): find the spoken numbers in a Ukrainian text that already writes them
as words (a human transcript in spoken form) and give each one a value and a span kind.

The other direction of ml_slav: ml_slav turns a written number into every case / gender form of its reading; this
module reads those forms back. The word forms come from ml_slav's closed-class tables (cardinal paradigms, scale nouns,
ordinal adjective endings, counted nouns), so the two stay in step. A span is the longest run of number words (one
whitespace between words, no punctuation) that parses as one number, places descending: hundreds, tens / teens, units,
a scale noun ("тисяча", "мільйон", ...) closing each group; an ordinal word may end the run ("дві тисячі двадцятому").

Precision over recall (a missed number is fine, a wrong span is not):
  * a bare "один / одна / одне" form is a number only before a counted noun (a unit, a currency, a percent, a time
    unit); "один з них", "не один", "один одного", "одного разу" are pronoun / idiom uses and are not tagged;
  * a bare "другий" form (also "the other") only before a noun that ordinals count (half, class, place, ...);
    plural "другі / других" alone never ("одні ... другі");
  * a bare plural scale ("тисячі / тисяч / мільйони") is "thousands of", not a value; bare "сорока" is also a noun;
  * words joined by a hyphen ("по-перше", "п'яти-шести") are one token and never match.

Span kinds and written text (what scripts reading the rows expect, as the digit-caption sources write them):
  int     ("7", "125", "1 800", "5 000 000"; counts >= 1000 grouped with spaces so a 4-digit count is never a year),
  int     year: an ordinal 1000-2099 ("у тисяча дев'ятсот сорок п'ятому році") -> "1945" (a bare 4-digit token),
  ord     ("5-й", "21-го"), decade ("90-ті", a plural tens ordinal before "роки / років ..."),
  pct     ("15%"; the span takes the noun), cur ("20 грн", "$5"), unit ("3 км"), dec ("2,5": "дві цілих п'ять десятих").
Each span: {"text", "kind", "value", "reading" (the words as written), "canonical", "a", "b" (char offsets), "by"}.

    python3 test_ml_uk_words.py
"""

import re

import ml_slav as S

LANG = "uk"
APOS = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "՚": "'", "‘": "'", "́": None})
WORD = "а-яіїєґА-ЯІЇЄҐ'’ʼ`́"
TOKEN = re.compile(rf"[{WORD}]+(?:-[{WORD}]+)*")


def fold(w: str) -> str:
    return w.lower().translate(APOS)


# ---------------------------------------------------------------- lexicon (from ml_slav's tables)

class E:
    """One reading of a word form: value, place (Z zero, U 1-9, T 10-19, X tens, H hundreds, S scale), ordinal or not,
    for ordinals the (gender/number, case), for scale nouns whether the form is singular / plural."""
    __slots__ = ("val", "place", "ord", "gn", "case", "sg", "pl", "fused")

    def __init__(self, val, place, ord_=False, gn=None, case=None, sg=False, pl=False, fused=False):
        self.val, self.place, self.ord, self.gn, self.case, self.sg, self.pl, self.fused = (
            val, place, ord_, gn, case, sg, pl, fused)

    def __repr__(self):
        return f"E({self.val},{self.place},{'ord' if self.ord else 'card'},{self.gn},{self.case})"


def _place(n: int) -> str:
    return "Z" if n == 0 else "U" if n < 10 else "T" if n < 20 else "X" if n < 100 else "H"


BASES = list(range(0, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100))


def _build() -> dict:
    lex: dict[str, list] = {}

    def add(form, e):
        lst = lex.setdefault(fold(form), [])
        if not any(repr(x) == repr(e) and x.sg == e.sg and x.pl == e.pl for x in lst):
            lst.append(e)

    val = {}
    for n in BASES:
        ws = S.card_nom(LANG, n)
        assert len(ws) == 1, (n, ws)
        val[ws[0]] = n
    val.update({"одна": 1, "одне": 1, "дві": 2})
    for lemma, cases in S.CARD[LANG].items():
        for case, alts in zip(S.CASES, cases):
            for f in alts:
                add(f, E(val[lemma], _place(val[lemma]), case=case))
    for lemma, (gnd, sg, pl) in ((k, S.NOUNS[LANG][k]) for k in S.SCALE_NOUN[LANG].values()):
        v = next(k for k, x in S.SCALE_NOUN[LANG].items() if x == lemma)
        sg_f = {f for alts in sg for f in alts}
        pl_f = {f for alts in pl for f in alts}
        for f in sg_f | pl_f:
            add(f, E(v, "S", sg=f in sg_f, pl=f in pl_f))
    for n in BASES[1:]:
        for ws in S.ord_nom(LANG, n):
            assert len(ws) == 1, (n, ws)
            for (gn, case), fs in S.adj_forms(ws[0], LANG).items():
                for f in fs:
                    add(f, E(n, _place(n), ord_=True, gn=gn, case=case))
    # fused scale ordinals ("двохтисячний" 2000, "п'ятисотмільйонний" is not needed); bare "тисячний / мільйонний" is
    # an adjective far more often ("тисячний натовп") and is left out
    for q in list(range(2, 20)) + list(range(20, 100, 10)) + list(range(100, 1000, 100)):
        for ws in S.ord_nom(LANG, q * 1000):
            for (gn, case), fs in S.adj_forms(ws[0], LANG).items():
                for f in fs:
                    add(f, E(q * 1000, "S", ord_=True, gn=gn, case=case, fused=True))
    return lex


LEX = _build()


def _forms(lemma: str) -> set:
    _, sg, pl = S.NOUNS[LANG][lemma]
    return {fold(f) for part in (sg, pl) for alts in part for f in alts}


PCT = _forms("відсоток") | _forms("процент")
CUR = {"гривня": "грн", "долар": "$", "євро": "€", "цент": "цент.", "євроцент": "євроцент.", "копійка": "коп.",
       "рубль": "руб."}
UNIT = {"кілометр": "км", "метр": "м", "сантиметр": "см", "міліметр": "мм", "мілілітр": "мл", "кілограм": "кг",
        "грам": "г", "тонна": "т", "літр": "л", "градус": "°"}
CUR_F = {f: abbr for lem, abbr in CUR.items() for f in _forms(lem)}
UNIT_F = {f: abbr for lem, abbr in UNIT.items() for f in _forms(lem)}
NOUN_LEMMA = {f: lem for lem in ("відсоток", "процент", *CUR, *UNIT) for f in _forms(lem)}


def agrees(v: int, form: str) -> bool:
    """The counted noun has a form the number takes in some case ("п'ять літрів", not "сімсот літр", the old unit
    "літра")."""
    lem = NOUN_LEMMA[form]
    cls = S.count_class(v)
    return any(form in {fold(f) for f in S.noun_form(LANG, lem, cls, c)} for c in S.CASES)
YEAR_F = _forms("рік") | {"рр", "р"}
YEAR_PL = {"роки", "років", "рокам", "роками", "роках"}
TIME_F = _forms("година") | _forms("хвилина") | _forms("секунда")
# a bare one-form is a count before these (not "рік": "одного року" is "one year, once"; not "раз": "одного разу")
ONE_NOUNS = PCT | set(CUR_F) | set(UNIT_F) | TIME_F | {
    f for lem in ("тисяча", "мільйон", "мільярд") for f in _forms(lem)}
# a bare "другий" form is the ordinal before these (else it is often "the other")
_ORD_NOUN_SRC = """половина половини половині половину половиною
світова світової світовій світову світовою
клас класу класі класом
курс курсу курсі курсом
місце місця місці місцем
день дня дню днем дні
тиждень тижня тижні тижнем
місяць місяця місяці місяцем
тур туру турі
етап етапу етапі
раунд раунду раунді
частина частини частині частину частиною
сезон сезону сезоні
поверх поверху поверсі
квартал кварталу кварталі
семестр семестру семестрі
століття тисячоліття
сторінка сторінки сторінці сторінку
номер номера номеру
рядок рядка рядку
пункт пункту пункті
розділ розділу розділі
кадр кадру
категорія категорії категорію
група групи групі групу
хвиля хвилі хвилю
рівень рівня рівні
спроба спроби спробі спробу"""
ORD_NOUNS = {w for line in _ORD_NOUN_SRC.splitlines() for w in line.split()} | YEAR_F | ONE_NOUNS
_ONE = {f for f, es in LEX.items() if any(e.val == 1 and not e.ord and e.place == "U" for e in es)}
_DRUG = {f for f, es in LEX.items() if any(e.val == 2 and e.ord for e in es)}
# bare forms that are also other words: "сьому" (the pronoun "сей"), "п'ята / п'яту / п'яти" (the heel: "ахіллесова
# п'ята", "душа в п'яти"); a number only before a counted noun
NEED_NOUN = {"сьому", "п'ята", "п'яту", "п'яти", "п'ятою", "п'яті"}
CILI = {"ціла", "цілих", "цілої", "цілій", "цілу", "цілою", "цілим", "цілими", "ціле"}
FRAC = {10: "десят", 100: "сот", 1000: "тисячн"}


# ---------------------------------------------------------------- parsing

_DEN_END = ("а", "ої", "ій", "у", "ою", "і", "их", "им", "ими")


def _is_den(w) -> bool:
    """A fraction denominator form: "десята / десятих / сотих / тисячних ..." (feminine or plural)."""
    return bool(w) and any(w.startswith(st) and w[len(st):] in _DEN_END for st in FRAC.values())


def _num_word(w) -> bool:
    return any(not e.ord for e in LEX.get(w, ()))


def _cili_before(toks, i) -> bool:
    """The number at token i is the fraction part of "X цілих Y десятих" (a decimal span already took it)."""
    return i > 0 and toks[i - 1][0] in CILI

IMPLICIT_ONE = {"тисяча", "тисячу", "тисячею", "мільйон", "мільйона", "мільйону", "мільйоном", "мільйоні", "мільярд",
                "мільярда", "мільярду", "мільярдом", "мільярді"}
# a form that is both a cardinal and an ordinal: the cardinal is kept ("до дев'яноста"), except "сьома" ("сьома
# година", "о сьомій": the instrumental cardinal "з сьома" is rare)
PREFER_ORD = {"сьома"}


def _ok_in_group(group_places: list, p: str) -> bool:
    if p == "H":
        return not group_places
    if p in ("X", "T"):
        return not any(q in ("X", "T", "U") for q in group_places)
    if p == "U":
        return not any(q in ("U", "T") for q in group_places)
    return False


def _parse(toks: list, i: int) -> tuple | None:
    """Longest parse of a number starting at token i: (end, value, entries)."""
    best = None

    def dfs(j, total, group, places, last_scale, used):
        nonlocal best
        if used and (best is None or j > best[0]):
            best = (j, total + group, list(used))
        if j >= len(toks) or (j > i and not toks[j - 1][3]):  # tokens must be joined by plain whitespace
            return
        if used and used[-1].ord:
            return  # an ordinal ends the number
        es = LEX.get(toks[j][0], ())
        if toks[j][0] in PREFER_ORD:
            es = sorted(es, key=lambda e: not e.ord)
        for e in es:
            if e.place == "Z":
                if not used:
                    dfs(j + 1, 0, 0, ["Z"], None, used + [e])
                continue
            if used and used[-1].place == "Z":
                continue
            if e.place == "S":
                if e.fused:
                    if not used:
                        dfs(j + 1, e.val, 0, [], e.val, used + [e])
                    continue
                if last_scale is not None and e.val >= last_scale:
                    continue
                mult = group
                if not places:  # implicit "one": "тисяча дев'ятсот", "у тисячі дев'ятсот сорок п'ятому"
                    if total or not e.sg:
                        continue
                    mult = 1
                dfs(j + 1, total + mult * e.val, 0, [], e.val, used + [e])
                continue
            if not _ok_in_group(places, e.place):
                continue
            dfs(j + 1, total, group + e.val, places + [e.place], last_scale, used + [e])

    dfs(i, 0, 0, [], None, [])
    return best


def _tokens(text: str) -> list:
    """[(folded, start, end, joined_to_next_by_whitespace_only)]"""
    ms = list(TOKEN.finditer(text))
    out = []
    for k, m in enumerate(ms):
        nxt = ms[k + 1].start() if k + 1 < len(ms) else None
        gap = text[m.end():nxt] if nxt is not None else ""
        out.append((fold(m.group()), m.start(), m.end(), nxt is not None and gap.isspace() and gap != ""))
    return out


_SUF = {"m": "й го му й м му", "f": "а ї й у ю й", "n": "е го му е м му", "pl": "і х м і ми х"}


def _ord_suffix(e: E) -> str:
    """The written ending of an ordinal ("5-й", "21-го", "7-у") from its gender / number and case."""
    return _SUF[e.gn].split()[S.CASES.index(e.case)]


def _int_text(v: int) -> str:
    return str(v) if v < 1000 else f"{v:,}".replace(",", " ")


def find(text: str) -> list[dict]:
    toks = _tokens(text)
    out = []
    i = 0
    while i < len(toks):
        p = _parse(toks, i)
        if p is None:
            i += 1
            continue
        j, v, used = p
        nxt = toks[j][0] if j < len(toks) and toks[j - 1][3] else None
        prev = toks[i - 1][0] if i > 0 and toks[i - 1][3] else None
        w = toks[i][0]
        single = j - i == 1
        e0 = used[0]
        bad = False
        if single:
            if w in _ONE and not e0.ord:
                bad = nxt not in ONE_NOUNS
            elif e0.ord and w in _DRUG:
                bad = e0.gn == "pl" or nxt not in ORD_NOUNS
            elif e0.place == "S" and not e0.ord:
                bad = not e0.sg or e0.pl  # "тисячі", "мільйонів": "thousands / millions of"
            elif w == "сорока" or w in NEED_NOUN:
                bad = nxt not in ORD_NOUNS
            elif w == "сім":
                bad = prev in ("у", "в", "на", "при")  # "на сім світі": the pronoun "сей"
        # a number word not in the tables right before ("двадесять сьоме", archaic): the value would be part of it
        if prev is not None and prev not in LEX and re.search(r"(?:дцят|десят|сот)[а-яіїєґ']*$", prev):
            bad = True
        if w in _ONE and prev in ("не", "ні", "до"):
            bad = True  # "не одну тисячу" (many thousands), "до одного" (every last one)
        # a fraction without "цілих": "п'ять десятих", "три тисячних" (0.5, 0.003). The count is a numerator, the
        # "десятих / сотих / тисячних" word a denominator, not the ordinal 10 / 100 / 1000: neither is spanned
        # ("X цілих Y десятих" is read as a decimal below)
        if _is_den(nxt) and not (nxt in CILI) and not _cili_before(toks, i):
            bad = True
        if _is_den(w) and single and (prev in CILI or (prev is not None and _num_word(prev))):
            bad = True
        if e0.place == "S" and not e0.ord and not e0.fused and not used[-1].ord and w not in IMPLICIT_ONE:
            bad = True  # "тисячі три" is "about three thousand"; "тисячі" alone starts only a year ("тисячі ... п'ятому")
        if j + 1 < len(toks) and toks[j][0] in ("з", "із", "зі") and toks[j + 1][0].startswith("половин") and toks[j - 1][3]:
            bad = True  # "двох з половиною годин" is 2.5, not 2
        if bad:
            i = j
            continue
        a, b = toks[i][1], toks[j - 1][2]
        last = used[-1]
        sp = None
        # decimal: "<int> цілих <int> десятих / сотих / тисячних"
        if not last.ord and nxt in CILI and j + 1 < len(toks) and toks[j][3]:
            q = _parse(toks, j + 1)
            if q is not None and q[0] < len(toks) and toks[q[0] - 1][3] and not q[2][-1].ord:
                den = toks[q[0]][0]
                for d, stem in FRAC.items():
                    if den.startswith(stem) and q[1] < d:
                        digs = len(str(d)) - 1
                        sp = {"kind": "dec", "text": f"{v},{q[1]:0{digs}d}", "value": v + q[1] / d,
                              "a": a, "b": toks[q[0]][2]}
                        j = q[0] + 1
                        break
        if sp is None and last.ord:
            if 1000 <= v <= 2099 and nxt in YEAR_F:  # "у тисяча дев'ятсот сорок п'ятому році": a year
                sp = {"kind": "int", "text": str(v), "value": v, "year": True}
            elif last.gn == "pl" and v % 10 == 0 and 20 <= v <= 90 and nxt in YEAR_PL:
                sp = {"kind": "decade", "text": f"{v}-ті", "value": v}
            else:
                sp = {"kind": "ord", "text": f"{v}-{_ord_suffix(last)}", "value": v}
        if sp is None and nxt is not None:
            if nxt in PCT and agrees(v, nxt):
                sp = {"kind": "pct", "text": f"{v}%", "value": v, "b": toks[j][2]}
            elif nxt in CUR_F and agrees(v, nxt):
                ab = CUR_F[nxt]
                sp = {"kind": "cur", "text": f"{ab}{_int_text(v)}" if ab in "$€" else f"{_int_text(v)} {ab}",
                      "value": v, "b": toks[j][2]}
            elif nxt in UNIT_F and agrees(v, nxt):
                sp = {"kind": "unit", "text": f"{_int_text(v)} {UNIT_F[nxt]}", "value": v, "b": toks[j][2]}
            if sp is not None:
                j += 1
        if sp is None:
            sp = {"kind": "int", "text": _int_text(v), "value": v}
        sp.setdefault("a", a)
        sp.setdefault("b", b)
        sp["reading"] = text[sp["a"]:sp["b"]]
        sp["canonical"] = sp["reading"]
        sp["by"] = "human"
        out.append(sp)
        i = j
    return _drop_joined(text, out)


def _drop_joined(text: str, spans: list) -> list:
    """"сто й десять", "триста й п'ятдесят", "двадцять й п'ять тисяч": one number said with a conjunction (110, 350,
    25 000); its parts would get the wrong values, so both spans go. The right part's leading group (before its
    scale word) must fit into the round left part: 20 + 5 (of 5 000), 300 + 50."""
    bad = set()
    for k in range(len(spans) - 1):
        s1, s2 = spans[k], spans[k + 1]
        if re.fullmatch(r"\s+(?:й|і|та)\s+", text[s1["b"]:s2["a"]]) and s1["kind"] == "int" and s2["kind"] in (
                "int", "unit", "cur", "pct"):
            v1, v2 = s1["value"], s2["value"]
            if not (isinstance(v1, int) and isinstance(v2, int) and 0 < v2):
                continue
            lead = v2
            while lead >= 1000 and lead % 1000 == 0:
                lead //= 1000
            if v1 < 1000 and 0 < lead < v1 and v1 % (10 ** len(str(lead))) == 0:
                bad |= {k, k + 1}
    return [s for k, s in enumerate(spans) if k not in bad]


# ---------------------------------------------------------------- text clean-up for a row label

_LAT2CYR = str.maketrans("iIaAeEoOcCpPxXyKMHTB", "іІаАеЕоОсСрРхХуКМНТВ")
_CYR = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
_LAT = re.compile(r"[A-Za-z]")


def clean(text: str) -> str | None:
    """The transcript as a label: Latin look-alike letters inside Cyrillic words ("вiн") -> Cyrillic, apostrophes ->
    ASCII, stress marks dropped, whitespace collapsed. None when a row cannot be a label (Latin words, digits, symbols,
    a broken transcript)."""
    if "\t" in text or "\n" in text or re.search(r"\d", text):
        return None
    t = text.translate(APOS)
    t = re.sub(r"\S+", lambda m: m.group().translate(_LAT2CYR) if _CYR.search(m.group()) or m.group() in ("i", "I")
               else m.group(), t)
    t = re.sub(r"\s+", " ", t).strip()
    if _LAT.search(t) or not _CYR.search(t) or re.search(r"[$€£¥¢₹₩₽%#/+°²³¹½¼¾⅓⅔&‰@*=<>\[\]{}|\\^~]", t):
        return None
    return t
