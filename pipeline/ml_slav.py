"""Russian and Ukrainian spoken-form candidates (2026-10-16). ml_cands dispatches here for lang in SLAV.

Russian and Ukrainian numbers inflect by case, gender and number: "в 1939 году" is said "в тысяча девятьсот тридцать
девятом году", "с 5 до 10 процентов" "с пяти до десяти процентов", "в 90-х" "в девяностых". The number grammar
(ml_grammar_table.json, ml_cands.g) gives the NOMINATIVE readings only (cardinals, and ordinals below 1000); this module
inflects them. Every case / gender / number form of a span's reading is offered as a candidate and the CTC reader plus
the external transcript choose (ml_decide); a light context rule (the preposition before the number, a case-marked noun
after it) only ORDERS the candidates, it never removes one. A written ending ("1917-м", "15-ти", "90-х") keeps only the
readings whose last word ends with it.

The inflection tables below are closed-class word forms (numerals, ordinal adjective endings, the counted nouns a
number span can carry), written out by hand and cross-checked against a morphological dictionary in
test_ml_slav.py (skipped when the dictionary is not installed). Emitted words never contain "ё" (the target model's
vocabulary cannot spell it) and use the ASCII apostrophe in Ukrainian, as the number grammar does.

Span kinds (same names as ml_cands): int (years included), ord (a hyphen ending), decade, dec, pct, unit, cur, range,
clock, phone (space-grouped digit runs that are not a thousands figure), code (leading zero / 5+ digits), skip (dates,
versions, glued compounds such as "48-річний", letter-glued digits): a skipped span drops its caption.
"""

import re

import ml_cands as M

SLAV = ("ru", "uk")
CASES = ("nom", "gen", "dat", "acc", "ins", "loc")
OBLIQUE = ("gen", "dat", "ins", "loc")
L = "а-яёіїєґА-ЯЁІЇЄҐ'’"  # Cyrillic word letters (Ukrainian apostrophe included)


def _t(s: str) -> tuple:
    """'a b/c d ...' -> six cases, each a tuple of alternatives (first = standard)."""
    xs = s.split()
    assert len(xs) == 6, s
    return tuple(tuple(x.split("/")) for x in xs)


# ---------------------------------------------------------------- cardinal word paradigms (nom gen dat acc ins loc)

_CARD_SRC = {
    "ru": """ноль ноля/нуля нолю/нулю ноль нолем/нулем ноле/нуле
один одного одному один/одного одним одном
одна одной одной одну одной/одною одной
одно одного одному одно одним одном
два двух двум два/двух двумя двух
две двух двум две/двух двумя двух
три трех трем три/трех тремя трех
четыре четырех четырем четыре/четырех четырьмя четырех
пять пяти пяти пять пятью пяти
шесть шести шести шесть шестью шести
семь семи семи семь семью семи
восемь восьми восьми восемь восемью/восьмью восьми
девять девяти девяти девять девятью девяти
десять десяти десяти десять десятью десяти
одиннадцать одиннадцати одиннадцати одиннадцать одиннадцатью одиннадцати
двенадцать двенадцати двенадцати двенадцать двенадцатью двенадцати
тринадцать тринадцати тринадцати тринадцать тринадцатью тринадцати
четырнадцать четырнадцати четырнадцати четырнадцать четырнадцатью четырнадцати
пятнадцать пятнадцати пятнадцати пятнадцать пятнадцатью пятнадцати
шестнадцать шестнадцати шестнадцати шестнадцать шестнадцатью шестнадцати
семнадцать семнадцати семнадцати семнадцать семнадцатью семнадцати
восемнадцать восемнадцати восемнадцати восемнадцать восемнадцатью восемнадцати
девятнадцать девятнадцати девятнадцати девятнадцать девятнадцатью девятнадцати
двадцать двадцати двадцати двадцать двадцатью двадцати
тридцать тридцати тридцати тридцать тридцатью тридцати
сорок сорока сорока сорок сорока сорока
пятьдесят пятидесяти пятидесяти пятьдесят пятьюдесятью пятидесяти
шестьдесят шестидесяти шестидесяти шестьдесят шестьюдесятью шестидесяти
семьдесят семидесяти семидесяти семьдесят семьюдесятью семидесяти
восемьдесят восьмидесяти восьмидесяти восемьдесят восемьюдесятью восьмидесяти
девяносто девяноста девяноста девяносто девяноста девяноста
сто ста ста сто ста ста
двести двухсот/двухста двумстам двести двумястами двухстах
триста трехсот/трехста тремстам триста тремястами трехстах
четыреста четырехсот/четырехста четыремстам четыреста четырьмястами четырехстах
пятьсот пятисот/пятиста пятистам пятьсот пятьюстами пятистах
шестьсот шестисот/шестиста шестистам шестьсот шестьюстами шестистах
семьсот семисот/семиста семистам семьсот семьюстами семистах
восемьсот восьмисот/восьмиста восьмистам восемьсот восемьюстами/восьмьюстами восьмистах
девятьсот девятисот/девятиста девятистам девятьсот девятьюстами девятистах""",
    "uk": """нуль нуля нулю нуль нулем нулі/нулю
один одного одному один/одного одним одному/однім
одна однієї/одної одній одну однією/одною одній
одне одного одному одне одним одному/однім
два двох двом два/двох двома двох
дві двох двом дві/двох двома двох
три трьох трьом три/трьох трьома трьох
чотири чотирьох чотирьом чотири/чотирьох чотирма/чотирьома чотирьох
п'ять п'яти/п'ятьох п'яти/п'ятьом п'ять п'ятьма/п'ятьома п'яти/п'ятьох
шість шести/шістьох шести/шістьом шість шістьма/шістьома шести/шістьох
сім семи/сімох семи/сімом сім сьома/сімома семи/сімох
вісім восьми/вісьмох восьми/вісьмом вісім вісьма/вісьмома восьми/вісьмох
дев'ять дев'яти/дев'ятьох дев'яти/дев'ятьом дев'ять дев'ятьма/дев'ятьома дев'яти/дев'ятьох
сорок сорока сорока сорок сорока сорока
дев'яносто дев'яноста дев'яноста дев'яносто дев'яноста дев'яноста
сто ста ста сто ста ста
двісті двохсот/двохста двомстам двісті двомастами двохстах
триста трьохсот/трьохста трьомстам триста трьомастами трьохстах
чотириста чотирьохсот/чотирьохста чотирьомстам чотириста чотирмастами чотирьохстах
п'ятсот п'ятисот/п'ятиста п'ятистам п'ятсот п'ятьмастами/п'ятьомастами п'ятистах
шістсот шестисот/шестиста шестистам шістсот шістьмастами/шістьомастами шестистах
сімсот семисот/семиста семистам сімсот сьомастами/сімомастами семистах
вісімсот восьмисот/восьмиста восьмистам вісімсот вісьмастами/вісьмомастами восьмистах
дев'ятсот дев'ятисот/дев'ятиста дев'ятистам дев'ятсот дев'ятьмастами/дев'ятьомастами дев'ятистах""",
}
# Ukrainian 10..19, 20, 30, 50..80: one regular pattern ("десять" -> "десяти/десятьох", "десятьма/десятьома")
_UK_REG = ("десять одинадцять дванадцять тринадцять чотирнадцять п'ятнадцять шістнадцять сімнадцять вісімнадцять "
           "дев'ятнадцять двадцять тридцять п'ятдесят шістдесят сімдесят вісімдесят").split()


def _parse_card() -> dict:
    out = {}
    for lang, src in _CARD_SRC.items():
        t = {}
        for line in src.splitlines():
            t[line.split(" ", 1)[0]] = _t(line)
        out[lang] = t
    for w in _UK_REG:
        s = w[:-1] if w.endswith("ь") else w  # drop the soft sign ("десять" -> "десяти", "п'ятдесят" -> "п'ятдесяти")
        out["uk"][w] = ((w,), (s + "и", s + "ьох"), (s + "и", s + "ьом"), (w,), (s + "ьма", s + "ьома"), (s + "и", s + "ьох"))
    return out


CARD = _parse_card()

# ---------------------------------------------------------------- counted nouns: (gender, singular cases, plural cases)

_NOUN_SRC = {
    "ru": """процент m процент процента проценту процент процентом проценте|проценты процентов процентам проценты процентами процентах
год m год года году год годом году/годе|годы годов/лет годам годы годами годах
рубль m рубль рубля рублю рубль рублем рубле|рубли рублей рублям рубли рублями рублях
доллар m доллар доллара доллару доллар долларом долларе|доллары долларов долларам доллары долларами долларах
евро m евро евро евро евро евро евро|евро евро евро евро евро евро
гривна f гривна гривны гривне гривну гривной гривне|гривны гривен гривнам гривны гривнами гривнах
копейка f копейка копейки копейке копейку копейкой копейке|копейки копеек копейкам копейки копейками копейках
цент m цент цента центу цент центом центе|центы центов центам центы центами центах
евроцент m евроцент евроцента евроценту евроцент евроцентом евроценте|евроценты евроцентов евроцентам евроценты евроцентами евроцентах
километр m километр километра километру километр километром километре|километры километров километрам километры километрами километрах
метр m метр метра метру метр метром метре|метры метров метрам метры метрами метрах
сантиметр m сантиметр сантиметра сантиметру сантиметр сантиметром сантиметре|сантиметры сантиметров сантиметрам сантиметры сантиметрами сантиметрах
миллиметр m миллиметр миллиметра миллиметру миллиметр миллиметром миллиметре|миллиметры миллиметров миллиметрам миллиметры миллиметрами миллиметрах
миллилитр m миллилитр миллилитра миллилитру миллилитр миллилитром миллилитре|миллилитры миллилитров миллилитрам миллилитры миллилитрами миллилитрах
килограмм m килограмм килограмма килограмму килограмм килограммом килограмме|килограммы килограмм/килограммов килограммам килограммы килограммами килограммах
грамм m грамм грамма грамму грамм граммом грамме|граммы грамм/граммов граммам граммы граммами граммах
тонна f тонна тонны тонне тонну тонной тонне|тонны тонн тоннам тонны тоннами тоннах
литр m литр литра литру литр литром литре|литры литров литрам литры литрами литрах
градус m градус градуса градусу градус градусом градусе|градусы градусов градусам градусы градусами градусах
час m час часа часу час часом часе|часы часов часам часы часами часах
минута f минута минуты минуте минуту минутой минуте|минуты минут минутам минуты минутами минутах
секунда f секунда секунды секунде секунду секундой секунде|секунды секунд секундам секунды секундами секундах
миллисекунда f миллисекунда миллисекунды миллисекунде миллисекунду миллисекундой миллисекунде|миллисекунды миллисекунд миллисекундам миллисекунды миллисекундами миллисекундах
тысяча f тысяча тысячи тысяче тысячу тысячей/тысячью тысяче|тысячи тысяч тысячам тысячи тысячами тысячах
миллион m миллион миллиона миллиону миллион миллионом миллионе|миллионы миллионов миллионам миллионы миллионами миллионах
миллиард m миллиард миллиарда миллиарду миллиард миллиардом миллиарде|миллиарды миллиардов миллиардам миллиарды миллиардами миллиардах
триллион m триллион триллиона триллиону триллион триллионом триллионе|триллионы триллионов триллионам триллионы триллионами триллионах""",
    "uk": """відсоток m відсоток відсотка відсотку/відсоткові відсоток відсотком відсотку/відсоткові|відсотки відсотків відсоткам відсотки відсотками відсотках
процент m процент процента проценту/процентові процент процентом проценті/проценту|проценти процентів процентам проценти процентами процентах
рік m рік року року/рокові рік роком році/року|роки років рокам роки роками роках
гривня f гривня гривні гривні гривню гривнею гривні|гривні гривень гривням гривні гривнями гривнях
копійка f копійка копійки копійці копійку копійкою копійці|копійки копійок копійкам копійки копійками копійках
цент m цент цента центу/центові цент центом центі/центу|центи центів центам центи центами центах
євроцент m євроцент євроцента євроценту/євроцентові євроцент євроцентом євроценті/євроценту|євроценти євроцентів євроцентам євроценти євроцентами євроцентах
долар m долар долара долару/доларові долар доларом доларі/долару|долари доларів доларам долари доларами доларах
євро m євро євро євро євро євро євро|євро євро євро євро євро євро
рубль m рубль рубля рублю/рублеві рубль рублем рублі/рублю|рублі рублів рублям рублі рублями рублях
кілометр m кілометр кілометра кілометру/кілометрові кілометр кілометром кілометрі/кілометру|кілометри кілометрів кілометрам кілометри кілометрами кілометрах
метр m метр метра метру/метрові метр метром метрі/метру|метри метрів метрам метри метрами метрах
сантиметр m сантиметр сантиметра сантиметру/сантиметрові сантиметр сантиметром сантиметрі/сантиметру|сантиметри сантиметрів сантиметрам сантиметри сантиметрами сантиметрах
міліметр m міліметр міліметра міліметру/міліметрові міліметр міліметром міліметрі/міліметру|міліметри міліметрів міліметрам міліметри міліметрами міліметрах
мілілітр m мілілітр мілілітра мілілітру/мілілітрові мілілітр мілілітром мілілітрі/мілілітру|мілілітри мілілітрів мілілітрам мілілітри мілілітрами мілілітрах
кілограм m кілограм кілограма кілограму/кілограмові кілограм кілограмом кілограмі/кілограму|кілограми кілограмів кілограмам кілограми кілограмами кілограмах
грам m грам грама граму/грамові грам грамом грамі/граму|грами грамів грамам грами грамами грамах
тонна f тонна тонни тонні тонну тонною тонні|тонни тонн тоннам тонни тоннами тоннах
літр m літр літра літру/літрові літр літром літрі/літру|літри літрів літрам літри літрами літрах
градус m градус градуса градусу/градусові градус градусом градусі/градусу|градуси градусів градусам градуси градусами градусах
година f година години годині годину годиною годині|години годин годинам години годинами годинах
хвилина f хвилина хвилини хвилині хвилину хвилиною хвилині|хвилини хвилин хвилинам хвилини хвилинами хвилинах
секунда f секунда секунди секунді секунду секундою секунді|секунди секунд секундам секунди секундами секундах
мілісекунда f мілісекунда мілісекунди мілісекунді мілісекунду мілісекундою мілісекунді|мілісекунди мілісекунд мілісекундам мілісекунди мілісекундами мілісекундах
тисяча f тисяча тисячі тисячі тисячу тисячею тисячі|тисячі тисяч тисячам тисячі тисячами тисячах
мільйон m мільйон мільйона мільйону/мільйонові мільйон мільйоном мільйоні/мільйону|мільйони мільйонів мільйонам мільйони мільйонами мільйонах
мільярд m мільярд мільярда мільярду/мільярдові мільярд мільярдом мільярді/мільярду|мільярди мільярдів мільярдам мільярди мільярдами мільярдах
трильйон m трильйон трильйона трильйону/трильйонові трильйон трильйоном трильйоні/трильйону|трильйони трильйонів трильйонам трильйони трильйонами трильйонах""",
}


def _parse_nouns() -> dict:
    out = {}
    for lang, src in _NOUN_SRC.items():
        t = {}
        for line in src.splitlines():
            head, pl = line.split("|")
            w, gnd, sg = head.split(" ", 2)
            t[w] = (gnd, _t(sg), _t(pl))
        out[lang] = t
    return out


NOUNS = _parse_nouns()
SCALE_NOUN = {"ru": {1000: "тысяча", 10**6: "миллион", 10**9: "миллиард", 10**12: "триллион"},
              "uk": {1000: "тисяча", 10**6: "мільйон", 10**9: "мільярд", 10**12: "трильйон"}}
_SCALE_FORMS = {lang: {f: lem for lem in SCALE_NOUN[lang].values() for num in NOUNS[lang][lem][1:] for alts in num
                      for f in alts} for lang in SLAV}


def count_class(n: int) -> int:
    """0 one (21, 101), 1 few (2-4, 22-24), 2 many (5-20, 25-30, 100): the Slavic count-noun rule."""
    n100, n10 = n % 100, n % 10
    if n10 == 1 and n100 != 11:
        return 0
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return 1
    return 2


def noun_form(lang: str, lemma: str, cls: int, case: str, frac: bool = False) -> list[str]:
    """The counted noun after a number of count class `cls` in `case` ("пять процентов", "двух процентов", "два
    процента", uk "два відсотки"); a fractional number takes the genitive singular in every case."""
    _, sg, pl = NOUNS[lang][lemma]
    i = CASES.index(case)
    if frac:
        return list(sg[1])
    if case in ("nom", "acc"):
        if cls == 0:
            return list(sg[i])
        if cls == 1:
            return list(sg[1]) if lang == "ru" else list(pl[0])
        return list(pl[1])
    return list(sg[i] if cls == 0 else pl[i])


# ---------------------------------------------------------------- ordinal adjective endings

# (m, f, n, pl) x (nom gen dat acc ins loc); "+" = the stem; alternatives "/"
_ADJ = {
    "ru_hard": {"m": "+ый +ого +ому +ый/+ого +ым +ом", "f": "+ая +ой +ой +ую +ой/+ою +ой",
                "n": "+ое +ого +ому +ое +ым +ом", "pl": "+ые +ых +ым +ые/+ых +ыми +ых"},
    "ru_soft": {"m": "+ий +ьего +ьему +ий/+ьего +ьим +ьем", "f": "+ья +ьей +ьей +ью +ьей/+ьею +ьей",
                "n": "+ье +ьего +ьему +ье +ьим +ьем", "pl": "+ьи +ьих +ьим +ьи/+ьих +ьими +ьих"},
    "uk_hard": {"m": "+ий +ого +ому +ий/+ого +им +ому/+ім", "f": "+а +ої +ій +у +ою +ій",
                "n": "+е +ого +ому +е +им +ому/+ім", "pl": "+і +их +им +і/+их +ими +их"},
    "uk_soft": {"m": "+ій +ього +ьому +ій/+ього +ім +ьому/+ім", "f": "+я +ьої +ій +ю +ьою +ій",
                "n": "+є +ього +ьому +є +ім +ьому/+ім", "pl": "+і +іх +ім +і/+іх +іми +іх"},
}
GNUM = ("m", "f", "n", "pl")


def adj_forms(word: str, lang: str) -> dict:
    """{(gnum, case): [forms]} of an ordinal adjective given in the masculine nominative."""
    if lang == "ru":
        if word.endswith("ий"):  # третий
            stem, pat = word[:-2], _ADJ["ru_soft"]
        elif word.endswith(("ый", "ой")):
            stem, pat = word[:-2], _ADJ["ru_hard"]
        else:
            raise ValueError(f"not an ordinal: {word}")
    else:
        if word.endswith("ій"):  # третій
            stem, pat = word[:-2], _ADJ["uk_soft"]
        elif word.endswith("ий"):
            stem, pat = word[:-2], _ADJ["uk_hard"]
        else:
            raise ValueError(f"not an ordinal: {word}")
    out = {}
    for gn in GNUM:
        for c, alts in zip(CASES, pat[gn].split()):
            fs = [a.replace("+", stem) for a in alts.split("/")]
            if lang == "ru" and word.endswith("ой") and gn == "m" and c in ("nom", "acc"):
                fs = [f.replace(stem + "ый", stem + "ой") for f in fs]  # второй, сороковой
            out[(gn, c)] = fs
    return out


# ---------------------------------------------------------------- readings

def _fold(w: str) -> str:
    return w.replace("ё", "е").replace("’", "'").replace("ʼ", "'")


def card_nom(lang: str, n: int) -> list[str]:
    return [_fold(w) for w in M.g(lang, "card", n)]


_GENDERED = {"ru": {"один": {"m": "один", "f": "одна", "n": "одно"}, "одна": {"m": "один", "f": "одна", "n": "одно"},
                    "два": {"m": "два", "f": "две", "n": "два"}, "две": {"m": "два", "f": "две", "n": "два"}},
             "uk": {"один": {"m": "один", "f": "одна", "n": "одне"}, "одна": {"m": "один", "f": "одна", "n": "одне"},
                    "два": {"m": "два", "f": "дві", "n": "два"}, "дві": {"m": "два", "f": "дві", "n": "два"}}}


def _word_case(lang: str, w: str, case: str, scale_cls: int | None) -> tuple:
    """Alternatives of one word of a nominative cardinal reading in `case`. A scale noun ("тысячи") keeps its count
    form in the nominative / accusative and takes the singular (after "one") or plural case form otherwise."""
    if w in _SCALE_FORMS[lang]:
        lem = _SCALE_FORMS[lang][w]
        _, sg, pl = NOUNS[lang][lem]
        if case == "nom":
            return (w,)
        if case == "acc":
            return sg[3] if scale_cls == 0 else (w,)
        return sg[CASES.index(case)] if scale_cls == 0 else pl[CASES.index(case)]
    if w not in CARD[lang]:
        raise ValueError(f"no paradigm for {w!r}")
    return CARD[lang][w][CASES.index(case)]


def _scale_cls(lang: str, w: str) -> int:
    lem = _SCALE_FORMS[lang][w]
    _, sg, pl = NOUNS[lang][lem]
    if w in sg[0]:
        return 0
    if (lang == "ru" and w in sg[1]) or (lang == "uk" and w in pl[0]):
        return 1
    return 2


def card_case(lang: str, ws: list[str], case: str, gender: str = "m", partial: bool = False) -> list[list[str]]:
    """A nominative cardinal reading inflected into `case`, every part ("тысячи девятисот тридцати девяти"); the final
    1 / 2 agree with `gender`. partial: only the last below-100 words inflect ("тысяча девятьсот тридцати девяти", a
    colloquial reduction). Ukrainian doublets (п'яти / п'ятьох) give two readings, each one consistent."""
    ws = list(ws)
    last_scale = max([i for i, w in enumerate(ws) if w in _SCALE_FORMS[lang]], default=-1)
    if ws and ws[-1] in _GENDERED[lang] and last_scale != len(ws) - 1:
        ws[-1] = _GENDERED[lang][ws[-1]][gender]
    tail0 = len(ws)
    if partial:
        tail0 = len(ws)
        while tail0 > 0 and ws[tail0 - 1] not in _SCALE_FORMS[lang] and _below100(lang, ws[tail0 - 1]):
            tail0 -= 1
    per = []
    for i, w in enumerate(ws):
        if partial and i < tail0:
            per.append((w,))
            continue
        cls = _scale_cls(lang, w) if w in _SCALE_FORMS[lang] else None
        per.append(_word_case(lang, w, case, cls))
    nalt = max(len(p) for p in per) if per else 1
    return [[p[min(k, len(p) - 1)] for p in per] for k in range(nalt)]


_HUNDREDS = {"ru": {"сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"},
             "uk": {"сто", "двісті", "триста", "чотириста", "п'ятсот", "шістсот", "сімсот", "вісімсот", "дев'ятсот"}}


def _below100(lang: str, w: str) -> bool:
    return w in CARD[lang] and w not in _HUNDREDS[lang]


def card_readings(lang: str, n: int, cases=CASES, genders=("m",), partial: bool = True) -> list[tuple]:
    """(words, tag) for the cardinal n in each case / final gender; "одна тысяча" also without "одна"."""
    base = card_nom(lang, n)
    bases = [base]
    if len(base) > 1 and base[0] in ("одна", "один") and base[1] in _SCALE_FORMS[lang]:
        bases.append(base[1:])
    out = []
    for b in bases:
        for c in cases:
            for gnd in genders:
                for ws in card_case(lang, b, c, gnd):
                    out.append((ws, ("card", c, gnd)))
                if partial and c in OBLIQUE and len(b) > 1:
                    for ws in card_case(lang, b, c, gnd, partial=True):
                        out.append((ws, ("card", c, gnd)))
                    ls = max([i for i, w in enumerate(b) if w in _SCALE_FORMS[lang]], default=-1)
                    if 0 <= ls < len(b) - 1:  # "тысяча двухсот": the group after the last scale noun only
                        for ws in card_case(lang, b[ls + 1:], c, gnd):
                            out.append((b[:ls + 1] + ws, ("card", c, gnd)))
    return out


_ORD_SCALE = {"ru": {1000: "тысячный", 10**6: "миллионный", 10**9: "миллиардный"},
              "uk": {1000: "тисячний", 10**6: "мільйонний", 10**9: "мільярдний"}}


def _fused_stems(lang: str, q: int) -> list[str]:
    """The combining stem of a multiplier in a fused ordinal: 2 -> "двух" ("двухтысячный"), 25 -> "двадцатипяти",
    100 -> "сто"; Ukrainian 2 -> "двох" / "дво"."""
    if q == 1:
        return [""]
    if q == 100:
        return ["сто"]
    if q == 90:
        return ["девяносто" if lang == "ru" else "дев'яносто"]
    ws = card_nom(lang, q)
    gens = []
    for w in ws:
        if w in _GENDERED[lang]:
            w = _GENDERED[lang][w]["m"]
        gens.append(CARD[lang][w][1])
    outs = ["".join(p[0] for p in gens)]
    if lang == "uk" and q in (2, 3, 4):
        outs.append({2: "дво", 3: "три", 4: "чотири"}[q])
    if lang == "ru" and q == 2:
        outs.append("дву")
    return outs


def ord_nom(lang: str, n: int) -> list[list[str]]:
    """Masculine nominative ordinal readings of n (the grammar's below 1000 and with a remainder; fused when a scale
    ends the number: "двухтысячный")."""
    if n < 1:
        return []  # the grammar has no ordinal of zero
    if n < 1000:
        return [[_fold(w) for w in M.g(lang, "ord", n)]]
    for sc in (10**9, 10**6, 1000):
        if n >= sc:
            q, r = divmod(n, sc)
            if r:
                if n >= 10**6:
                    return []  # the grammar has no ordinal above a million
                return [[_fold(w) for w in M.g(lang, "ord", n)]]
            if q >= 1000:
                return []
            return [[s + _ORD_SCALE[lang][sc]] for s in _fused_stems(lang, q)]
    return []


def ord_readings(lang: str, n: int, elide: bool = False, inanimate: bool = False) -> list[tuple]:
    """(words, tag) for every gender / number / case of the ordinal n; elide: also the last two digits alone, the way
    a year is often said ("2020 года" -> "двадцатого года", "в 90-х" for 1990s)."""
    out = []
    bases = ord_nom(lang, n)
    if elide and 1000 <= n <= 2100 and n % 100:
        bases = bases + ord_nom(lang, n % 100)
    if lang == "uk":  # Ukrainian speakers often say the thousand in the genitive: "у тисячі дев'ятсот сорок п'ятому"
        bases = bases + [["тисячі"] + b[1:] for b in bases if len(b) > 1 and b[0] == "тисяча"]
    for b in bases:
        forms = adj_forms(b[-1], lang)
        for (gn, c), fs in forms.items():
            for f in (fs[:1] if inanimate and c == "acc" else fs):
                out.append((b[:-1] + [f], ("ord", c, gn)))
    return out


# ---------------------------------------------------------------- span finding

_SUF = r"(?:-(?P<suf>[" + L + r"]{1,3})(?![" + L + r"\w]))"
_GLUED = re.compile(r"-[" + L + r"]{4,}")
UNITS = {
    "ru": {"км": ("километр", []), "м": ("метр", []), "см": ("сантиметр", []), "мм": ("миллиметр", []),
           "кг": ("килограмм", []), "гр": ("грамм", []), "т": ("тонна", []), "л": ("литр", []),
           "мл": ("миллилитр", []), "ч": ("час", []), "мин": ("минута", []), "сек": ("секунда", []),
           "мс": ("миллисекунда", []),
           "°": ("градус", []), "°C": ("градус", ["Цельсия"]), "°С": ("градус", ["Цельсия"]),
           "км/ч": ("километр", ["в", "час"])},
    "uk": {"км": ("кілометр", []), "м": ("метр", []), "см": ("сантиметр", []), "мм": ("міліметр", []),
           "кг": ("кілограм", []), "гр": ("грам", []), "т": ("тонна", []), "л": ("літр", []),
           "мл": ("мілілітр", []), "год": ("година", []), "хв": ("хвилина", []), "сек": ("секунда", []),
           "мс": ("мілісекунда", []),
           "°": ("градус", []), "°C": ("градус", ["Цельсія"]), "°С": ("градус", ["Цельсія"]),
           "км/год": ("кілометр", ["на", "годину"])},
}
ALT_TAILS = {("uk", "км/год"): [["за", "годину"]], ("ru", "км/ч"): [["час"]]}
CUR = {"ru": {"$": "доллар", "€": "евро", "₽": "рубль", "руб.": "рубль", "руб": "рубль", "₴": "гривна", "грн": "гривна",
              "грн.": "гривна"},
       "uk": {"$": "долар", "€": "євро", "₴": "гривня", "грн": "гривня", "грн.": "гривня", "₽": "рубль", "руб.": "рубль",
              "руб": "рубль"}}
# the minor unit of each currency ("пять долларов девяносто девять центов", "дві гривні п'ятдесят копійок")
MINOR = {"ru": {"доллар": ["цент"], "евро": ["цент", "евроцент"], "рубль": ["копейка"], "гривна": ["копейка"]},
         "uk": {"долар": ["цент"], "євро": ["цент", "євроцент"], "гривня": ["копійка"], "рубль": ["копійка"]}}
SCALE_ABBR = {"ru": {"тыс.": 1000, "тыс": 1000, "млн": 10**6, "млн.": 10**6, "млрд": 10**9, "млрд.": 10**9,
                     "трлн": 10**12, "трлн.": 10**12},
              "uk": {"тис.": 1000, "тис": 1000, "млн": 10**6, "млн.": 10**6, "млрд": 10**9, "млрд.": 10**9,
                     "трлн": 10**12, "трлн.": 10**12}}
YEAR_ABBR = {"ru": {"г.": "sg", "гг.": "pl"}, "uk": {"р.": "sg", "рр.": "pl"}}
_YEAR_NOUN = {"ru": "год", "uk": "рік"}
PCT_NOUNS = {"ru": ["процент"], "uk": ["відсоток", "процент"]}


def _alt(xs) -> str:
    return "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))


_POST = {}


def _post_re(lang: str):
    if lang not in _POST:
        alts = ["%"] + list(UNITS[lang]) + list(CUR[lang]) + list(SCALE_ABBR[lang]) + list(YEAR_ABBR[lang])
        _POST[lang] = re.compile(r"[   ]?(?P<p>" + _alt(alts) + r")(?![" + L + r"\w])")
    return _POST[lang]


NUM = re.compile(r"(?<![\w.,/:])(?P<int>\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+)(?:(?P<dsep>[,.])(?P<frac>\d+))?")
CLOCK = re.compile(r"(?<![\w.,:])(?P<h>\d{1,2}):(?P<m>\d{2})(?![\d:]|[.,]\d)")
RANGE = re.compile(r"(?<![\w.,/:\-–−+])(?P<l>\d{1,4})(?P<sep>[-–—]|\s[-–—]\s)(?P<r>\d{1,4})(?!\d|[,.]\d|:\d|[-–]\d)")
DATE_DMY = re.compile(r"(?<![\w.])(\d{1,2})([./])(\d{1,2})\2(\d{4}|\d{2})(?!\d|[.,]\d)")  # 17.12.2000, 3/9/20
DATE_ISO = re.compile(r"(?<![\w.])(\d{4})-(\d{2})-(\d{2})(?!\d)")
VERSION = re.compile(r"(?<![\w.,])\d+(?:\.\d+){2,}")  # versions, IPs, dot-grouped numbers
CHAIN = re.compile(r"\d+(?:[-–]\d+){2,}")  # 3-part hyphen chains
TIMES = re.compile(r"(?<![\w.])(\d{1,4})\s?([xх×])\s?(\d{1,4})(?!\d)")  # "3x4"
_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
GLUED_UNITS = {"s": "сек", "sec": "сек", "ms": "мс", "kg": "кг", "km": "км", "cm": "см", "mm": "мм"}  # Latin, glued
_GLUED_WORD = re.compile(r"[A-Za-zкК]{1,3}(?![" + L + r"\w])")


def _valid_dm(day: int, mon: int) -> bool:
    return 1 <= mon <= 12 and 1 <= day <= _DAYS[mon - 1]


def _free(taken, a, b) -> bool:
    return not any(x < b and a < y for x, y in taken)


def _ctx(text: str, a: int, b: int) -> dict:
    pm = re.search(r"([" + L + r"A-Za-z]+)[\s,]*$", text[:a])
    nm = re.match(r"[\s,]*([" + L + r"A-Za-z]+)", text[b:])
    return {"prev": pm.group(1).lower() if pm and text[pm.end(1):a].strip(" ") == "" else None,
            "next": nm.group(1).lower() if nm else None}


def find(text: str, lang: str) -> list[dict]:
    out, taken = [], []

    def add(d):
        d.update(_ctx(text, d["a"], d["b"]))
        out.append(d)
        taken.append((d["a"], d["b"]))

    def letter_after(b):
        return bool(re.match(r"[" + L + r"A-Za-z]", text[b:b + 1]))

    for mt in DATE_DMY.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            dd, sep, mm, yy = int(mt[1]), mt[2], int(mt[3]), mt[4]
            reads = [(dd, mm)] if _valid_dm(dd, mm) else []
            if sep == "/" and _valid_dm(mm, dd) and (mm, dd) not in reads:
                reads.append((mm, dd))  # the month-first order
            d = {"a": mt.start(), "b": mt.end(), "kind": "skip", "note": "date/version/chain"}
            if letter_after(mt.end()):
                d["note"] = "glued letters"
            elif reads:
                d.update(kind="date", dm=reads, year=yy, parts=[mt[1], mt[3], yy] if sep == "." else None)
            elif sep == ".":
                d.update(kind="version", parts=[mt[1], mt[3], yy])
            add(d)
    for mt in DATE_ISO.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            yy, mm, dd = mt[1], int(mt[2]), int(mt[3])
            d = {"a": mt.start(), "b": mt.end(), "kind": "skip", "note": "date/version/chain"}
            if letter_after(mt.end()):
                d["note"] = "glued letters"
            elif _valid_dm(dd, mm):
                d.update(kind="date", dm=[(dd, mm)], year=yy, parts=None, chain=[mt[1], mt[2], mt[3]])
            else:
                d.update(kind="chain", parts=[mt[1], mt[2], mt[3]])
            add(d)
    for mt in VERSION.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            d = {"a": mt.start(), "b": mt.end(), "kind": "version", "parts": mt.group(0).split(".")}
            if letter_after(mt.end()):
                d.update(kind="skip", note="glued letters")
            add(d)
    for mt in CHAIN.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            ps = re.split(r"[-–]", mt.group(0))
            d = {"a": mt.start(), "b": mt.end(), "kind": "chain", "parts": ps}
            if letter_after(mt.end()):
                d.update(kind="skip", note="glued letters")
            elif len(ps) == 3 and len(ps[2]) in (2, 4) and len(ps[0]) <= 2 and len(ps[1]) <= 2 and _valid_dm(
                    int(ps[0]), int(ps[1])):
                d.update(dm=[(int(ps[0]), int(ps[1]))], year=ps[2])  # 17-12-2000 is a date as well
            add(d)
    for mt in TIMES.finditer(text):
        if _free(taken, mt.start(), mt.end()):
            d = {"a": mt.start(), "b": mt.end(), "kind": "times", "l": mt[1], "r": mt[3], "x": mt[2]}
            if letter_after(mt.end()):
                d.update(kind="skip", note="glued letters")
            elif mt[1] == "0" and mt[2] == "x":
                d.update(kind="skip", note="hex literal")
            add(d)
    for mt in CLOCK.finditer(text):
        h, m = int(mt["h"]), int(mt["m"])
        if h <= 23 and m <= 59 and _free(taken, mt.start(), mt.end()):
            add({"a": mt.start(), "b": mt.end(), "kind": "clock", "h": h, "m": m})
    for mt in RANGE.finditer(text):
        a, b = mt.start(), mt.end()
        if not _free(taken, a, b):
            continue
        d = {"a": a, "b": b, "kind": "range", "l": mt["l"], "r": mt["r"]}
        if re.match(r"-[" + L + r"]", text[b:]):
            d.update(kind="skip", note="range with an ending")
        else:
            pm = _post_re(lang).match(text, b)
            if pm and pm["p"] in ("%",) + tuple(UNITS[lang]):
                d.update(b=pm.end(), post=pm["p"])
        if a > 0 and text[a - 1] in "-−+":
            d.update(kind="skip", note="range/sign")
        add(d)
    for mt in NUM.finditer(text):
        a, b = mt.start(), mt.end()
        if not _free(taken, a, b):
            continue
        raw, frac, dsep = mt["int"], mt["frac"], mt["dsep"]
        d = {"a": a, "b": b, "int": raw, "frac": frac, "dsep": dsep, "kind": "int"}
        if re.search(r"[   ]", raw):
            d["grouped"] = True
        before = text[a - 1] if a else ""
        pre = re.search(r"(?P<p>[$€₽₴]|№)[  ]?$", text[:a])
        if pre and _free(taken, pre.start(), a):
            d["a"] = pre.start()
            d["pre"] = pre["p"]
        elif before and before in "-−+" and (a == 1 or re.match(r"[\s(«\"“]", text[a - 2])):
            d["sign"], d["a"] = before, a - 1
        elif before and before in "-–−" and a >= 2 and re.match(r"[" + L + r"A-Za-z\d]", text[a - 2]):
            d["codepart"] = True  # "Т-62", "БЗ-58-2", "ГОСТ 51511-2000": the number is read, the letters / dash stay
            d["afterdigit"] = text[a - 2].isdigit()
        elif before and before in "-–+−":
            d.update(kind="skip", note="range/sign")
        sm = re.match(_SUF, text[b:])
        if d["kind"] != "skip" and sm and frac is None and not d.get("grouped"):
            d.update(b=b + sm.end(), kind="ord", suf=_fold(sm["suf"].lower()))
            n = M._int(raw)
            if n % 10 == 0 and re.fullmatch(r"(?:е|х|ые|ых|ми|ыми|ті|х|их|і|ими)", d["suf"]):
                d["kind"] = "decade"
        elif d["kind"] != "skip" and _GLUED.match(text[b:]):
            cm = re.match(r"-([" + L + r"]{4,})(?![" + L + r"\w-])", text[b:])
            if cm and frac is None and not d.get("grouped") and not d.get("pre") and not d.get("sign") and (
                    not d.get("codepart")) and 0 < M._int(raw) < 1000 and not raw.startswith("0"):
                d.update(kind="compound", adj=_fold(cm[1].lower()), b=b + cm.end())
            else:
                d.update(kind="skip", note="glued compound")
        elif d["kind"] != "skip":
            pm = _post_re(lang).match(text, b)
            if pm:
                p = pm["p"]
                d["b"] = pm.end()
                if p == "%":
                    d["kind"] = "pct"
                elif p in UNITS[lang]:
                    d.update(kind="unit", unit=p)
                elif p in CUR[lang]:
                    d.update(kind="cur", cur=p)
                elif p in SCALE_ABBR[lang]:
                    d["scale"] = SCALE_ABBR[lang][p]
                elif p in YEAR_ABBR[lang]:
                    if frac is None and 1000 <= M._int(raw) <= 2100:
                        d["yabbr"] = YEAR_ABBR[lang][p]
                    else:
                        d["b"] = b  # "г." after a non-year: left in the caption (the row is then dropped)
            if d.get("pre") in ("$", "€", "₽", "₴"):
                if d["kind"] == "int":
                    d.update(kind="cur", cur=d["pre"])
                else:
                    d.update(kind="skip", note="two units")
        if d["kind"] == "int":
            if frac is not None:
                d["kind"] = "dec"
            elif re.fullmatch(r"0\d+|\d{8,}", raw):
                d["kind"] = "code"  # a code / phone run (wave-1 rule); 5-7 digits stay "int" with chunk readings too
            elif d.get("grouped"):
                d["kind"] = "int"
        if d["kind"] in ("int", "dec") and re.match(r"[" + L + r"A-Za-z]", text[d["b"]:d["b"] + 1]):
            gm = _GLUED_WORD.match(text, d["b"])
            if gm and not d.get("pre") and not d.get("scale"):
                w = gm.group(0)
                if w in ("k", "K", "к", "К"):  # "300к", "5K": thousands, or the letter itself
                    d.update(scale=1000, klet=w, b=gm.end())
                elif w in GLUED_UNITS and not d.get("grouped"):  # "4,4s"
                    d.update(kind="unit", unit=GLUED_UNITS[w], b=gm.end())
        if d["kind"] not in ("skip",) and re.match(r"[" + L + r"A-Za-z]", text[d["b"]:d["b"] + 1]):
            d.update(kind="skip", note="glued letters")
        if d["kind"] not in ("skip",) and frac is not None and d["kind"] in ("ord", "decade"):
            d.update(kind="skip", note="decimal with an ending")
        add(d)
    return sorted(out, key=lambda x: x["a"])


# ---------------------------------------------------------------- context ranking (orders, never filters)

PREP = {
    "ru": {"в": ("loc", "acc"), "во": ("loc", "acc"), "на": ("loc", "acc"), "о": ("loc",), "об": ("loc",),
           "при": ("loc",), "с": ("gen", "ins"), "со": ("gen", "ins"), "от": ("gen",), "до": ("gen",), "из": ("gen",),
           "без": ("gen",), "у": ("gen",), "около": ("gen",), "после": ("gen",), "для": ("gen",), "более": ("gen",),
           "менее": ("gen",), "свыше": ("gen",), "порядка": ("gen",), "больше": ("gen",), "меньше": ("gen",),
           "кроме": ("gen",), "среди": ("gen",), "к": ("dat",), "ко": ("dat",), "по": ("dat", "acc"),
           "между": ("ins",), "над": ("ins",), "под": ("ins", "acc"), "перед": ("ins",), "за": ("acc", "ins"),
           "через": ("acc",), "про": ("acc",)},
    "uk": {"в": ("loc", "acc"), "у": ("loc", "acc"), "на": ("loc", "acc"), "о": ("loc", "acc"), "об": ("loc", "acc"),
           "при": ("loc",), "з": ("gen", "ins"), "із": ("gen", "ins"), "зі": ("gen", "ins"), "від": ("gen",),
           "до": ("gen",), "без": ("gen",), "біля": ("gen",), "після": ("gen",), "для": ("gen",), "близько": ("gen",),
           "понад": ("gen",), "більше": ("gen",), "менше": ("gen",), "крім": ("gen",), "серед": ("gen",),
           "протягом": ("gen",), "к": ("dat",), "по": ("loc", "acc", "dat"), "між": ("ins",), "над": ("ins",),
           "під": ("ins", "acc"), "перед": ("ins",), "за": ("acc", "ins"), "через": ("acc",), "про": ("acc",)},
}
MONTHS_GEN = {"ru": "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(),
              "uk": "січня лютого березня квітня травня червня липня серпня вересня жовтня листопада грудня".split()}
# a following noun that asks for an ordinal: word -> (gender/number, cases)
ORD_NEXT = {
    "ru": {"год": ("m", ("nom", "acc")), "года": ("m", ("gen",)), "году": ("m", ("loc", "dat")), "годом": ("m", ("ins",)),
           "годы": ("pl", ("nom", "acc")), "годов": ("pl", ("gen",)), "годах": ("pl", ("loc",)),
           "годам": ("pl", ("dat",)), "годами": ("pl", ("ins",)), "числа": ("n", ("gen",)), "число": ("n", ("nom", "acc")),
           "век": ("m", ("nom", "acc")), "века": ("m", ("gen",)), "веке": ("m", ("loc",)),
           "место": ("n", ("nom", "acc")), "места": ("n", ("gen",)), "месте": ("n", ("loc",)),
           "класс": ("m", ("nom", "acc")), "класса": ("m", ("gen",)), "классе": ("m", ("loc",))},
    "uk": {"рік": ("m", ("nom", "acc")), "року": ("m", ("gen", "dat", "loc")), "році": ("m", ("loc",)),
           "роком": ("m", ("ins",)), "роки": ("pl", ("nom", "acc")), "років": ("pl", ("gen",)), "роках": ("pl", ("loc",)),
           "числа": ("n", ("gen",)), "число": ("n", ("nom", "acc")), "століття": ("n", ("nom", "gen", "acc")),
           "місце": ("n", ("nom", "acc")), "місця": ("n", ("gen",)), "місці": ("n", ("loc",)),
           "клас": ("m", ("nom", "acc")), "класу": ("m", ("gen",)), "класі": ("m", ("loc",))},
}
for _l in SLAV:
    for _m in MONTHS_GEN[_l]:
        ORD_NEXT[_l][_m] = ("n", ("nom", "gen"))


def _rank(cands: list[tuple], d: dict, lang: str, prefer_card: bool = True) -> list[tuple]:
    prev, nxt = d.get("prev"), d.get("next")
    pc = PREP[lang].get(prev or "", ())
    on = ORD_NEXT[lang].get(nxt or "")

    def score(item):
        _, tag = item
        typ, case, gn = tag
        s = 0
        if on:
            s += 0 if (typ == "ord" and gn == on[0] and case in on[1]) else 4
        elif prefer_card:
            s += 0 if typ != "ord" else 2
        if pc:
            s += pc.index(case) if case in pc else len(pc)
        else:
            s += 0 if case == "nom" else 1
        return s

    return [x for _, x in sorted(enumerate(cands), key=lambda t: (score(t[1]), t[0]))]


def _strs(items: list[tuple]) -> list[str]:
    seen, out = set(), []
    for ws, _ in items:
        s = " ".join(w for w in ws if w)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------- per-kind candidates

def _digits(lang: str, s: str) -> list[str]:
    out = []
    for c in s:
        out += card_nom(lang, int(c))
    return out


def _chunked(lang: str, s: str) -> list[list[str]]:
    """A digit string said digit by digit, or in chunks of 2-3 digits (a chunk with a leading zero: "ноль" + rest)."""
    res = [_digits(lang, s)]
    memo = {len(s): [[]]}

    def rec(i):
        if i in memo:
            return memo[i]
        r = []
        for k in (2, 3):
            if i + k <= len(s):
                c = s[i:i + k]
                w = (card_nom(lang, 0) + card_nom(lang, int(c[1:]))) if c[0] == "0" and c[1] != "0" else (
                    _digits(lang, c) if c[0] == "0" else card_nom(lang, int(c)))
                r += [w + rest for rest in rec(i + k)]
        memo[i] = r[:64]
        return memo[i]

    return res + rec(0)[:24]


def _int_items(lang: str, n: int, d: dict, ords: bool = True) -> list[tuple]:
    items = card_readings(lang, n, genders=("m", "f", "n"))
    if ords and n < 10**6:
        items += ord_readings(lang, n, elide=n <= 2100)
    if n == 1:  # counting aloud: "раз, два, три"
        items.append((["раз"], ("other", "nom", "m")))
    return items


def _year_abbr(lang: str, n: int, num: str) -> list[tuple]:
    noun = _YEAR_NOUN[lang]
    _, sg, pl = NOUNS[lang][noun]
    gn = "m" if num == "sg" else "pl"
    out = []
    for ws, tag in ord_readings(lang, n, elide=True, inanimate=True):
        if tag[2] != gn:
            continue
        forms = (sg if num == "sg" else pl)[CASES.index(tag[1])]
        for f in forms[:1] if tag[1] == "acc" else forms:
            out.append((ws + [f], tag))
    return out


def _dec_items(lang: str, n: int, frac: str, dsep: str, cases=CASES) -> list[tuple]:
    """Decimal readings: "две целых пять десятых" (every case), "два с половиной", "ноль двадцать пять", the grammar's
    "ноль запятая два пять", "три точка пять" for a point."""
    out = []
    k = len(frac)
    v = int(frac)
    cel = {"ru": ("целая", "целых", "целые"), "uk": ("ціла", "цілих", "цілі")}[lang]
    den = {"ru": {1: "десятый", 2: "сотый", 3: "тысячный"}, "uk": {1: "десятий", 2: "сотий", 3: "тисячний"}}[lang]
    and_w = {"ru": "и", "uk": "і"}[lang]
    if 1 <= k <= 3 and not frac.startswith("0" * k):
        cel_f = adj_forms({"ru": "целый", "uk": "цілий"}[lang], lang)
        den_f = adj_forms(den[k], lang)
        iw, vw = card_nom(lang, n), card_nom(lang, v)
        for c in cases:
            ci = count_class(n)
            cw = cel_f[("f", c)] if ci == 0 else (["целых" if lang == "ru" else "цілих"] if c in ("nom", "acc")
                                                    else cel_f[("pl", c)])
            if c == "nom" and ci == 1:
                cw = cw + [cel[2]]
            vi = count_class(v)
            dw = den_f[("f", c)] if vi == 0 else (den_f[("pl", "gen")] if c in ("nom", "acc") else den_f[("pl", c)])
            for a in card_case(lang, iw, c, "f"):
                for b in card_case(lang, vw, c, "f"):
                    for cwf in cw[:2]:
                        for dwf in dw[:1]:
                            out.append((a + [cwf] + b + [dwf], ("dec", c, "f")))
                            if c in ("nom", "acc"):
                                out.append((a + [cwf, and_w] + b + [dwf], ("dec", c, "f")))
                    if c in ("nom", "acc"):
                        out.append((a + [and_w] + b + [dw[0]], ("dec", c, "f")))
    if frac == "5" and n >= 1:
        half = {"ru": ["с", "половиной"], "uk": ["з", "половиною"]}[lang]
        for c in cases:
            for gnd in ("m", "f"):
                for a in card_case(lang, card_nom(lang, n), c, gnd):
                    out.append((a + half, ("half", c, gnd)))
    if frac == "5" and n == 1:
        p = {"ru": (["полтора"], ["полторы"], ["полутора"]), "uk": (["півтора"], ["півтори"], ["півтора"])}[lang]
        out += [(p[0], ("poltora", "nom", "m")), (p[1], ("poltora", "nom", "f"))]
        out += [(p[2], ("poltora", c, "m")) for c in OBLIQUE]
    iw = card_nom(lang, n)
    fws = [_digits(lang, frac)]
    if not frac.startswith("0") and len(frac) > 1:
        fws.append(card_nom(lang, v))
    if frac.startswith("0") and len(frac) == 2 and frac[1] != "0":
        fws.append(card_nom(lang, 0) + card_nom(lang, int(frac[1])))
    sep = {"ru": ["запятая"] + (["точка"] if dsep == "." else []), "uk": ["кома"] + (["крапка"] if dsep == "." else [])}
    and_w = {"ru": "и", "uk": "і"}[lang]
    cel_pl = {"ru": "целых", "uk": "цілих"}[lang]
    cel_sg = {"ru": "целая", "uk": "ціла"}[lang]
    for f in fws:
        out.append((iw + f, ("other", "nom", "m")))
        out.append((iw + [and_w] + f, ("other", "nom", "m")))
        for a in card_case(lang, iw, "nom", "f"):
            out.append((a + [cel_sg if count_class(n) == 0 else cel_pl] + f, ("other", "nom", "f")))
            out.append((a + f, ("other", "nom", "f")))
        for s in sep[lang]:
            out.append((iw + [s] + f, ("other", "nom", "m")))
    try:
        out.append(([_fold(w) for w in M.g(lang, "dec", n, frac)], ("other", "nom", "m")))
    except M.Missing:
        pass
    return out


def _quantity(lang: str, n: int, lemma: str, tail: list[str] = (), cases=CASES) -> list[tuple]:
    """A number with a counted noun after it, agreeing in case and count ("пять процентов", "двух процентов", "одна
    минута", uk "два відсотки")."""
    gnd = NOUNS[lang][lemma][0]
    cls = count_class(n)
    out = []
    for ws, tag in card_readings(lang, n, cases=cases, genders=(gnd,)):
        for nf in noun_form(lang, lemma, cls, tag[1]):
            out.append((ws + [nf] + list(tail), tag))
    return out


def _cents_items(lang: str, n: int, frac: str, dsep: str, major: str) -> list[tuple]:
    """An amount with cents, every way it is said: "пять долларов девяносто девять центов" (both nouns agreeing in the
    same case), "пять долларов девяносто девять", "пять девяносто девять", "ноль евро девяносто девять центов" /
    "девяносто девять центов" for 0,99, and the decimal readings ("пять целых девяносто девять сотых доллара",
    "полтора евро"). "1.000 ₽" is a dot-grouped thousand."""
    if dsep == "." and len(frac) == 3:
        return _quantity(lang, n * 1000 + int(frac), major)
    dec = list(_dec_quantity(lang, n, frac, dsep, major))
    if len(frac) > 2:
        return dec
    cents = int(frac.ljust(2, "0"))
    big = _quantity(lang, n, major)
    if cents == 0:
        return big + dec
    out = []  # the everyday forms first, the decimal readings after them
    for minor in MINOR[lang].get(major, []):
        small = _quantity(lang, cents, minor)
        by_case = {}
        for ws, tag in small:
            by_case.setdefault(tag[1], []).append(ws)
        for ws, tag in big:
            for ms in by_case.get(tag[1], [])[:2]:
                out.append((ws + ms, tag))
        if n == 0:
            out += small
    cw = card_nom(lang, cents)
    if MINOR[lang].get(major) and NOUNS[lang][MINOR[lang][major][0]][0] == "f":
        cw = _fem(lang, cw)  # the unsaid noun is "копейки": "два рубля две"
    for ws, tag in big:
        if tag[1] == "nom":
            out.append((ws + cw, tag))  # "пять долларов девяносто девять"
    out.append((card_nom(lang, n) + cw, ("card", "nom", "m")))  # "пять девяносто девять"
    return out + dec


def _dec_quantity(lang: str, n: int, frac: str, dsep: str, lemma: str, tail=()) -> list[tuple]:
    """A decimal with its counted noun: the genitive singular after a fraction ("две целых пять десятых процента");
    after "N с половиной" the noun agrees with N ("пять с половиной миллионов", "двух с половиной миллионов"); after
    "полтора" the genitive singular, plural in the oblique cases ("около полутора миллионов")."""
    gsg = noun_form(lang, lemma, 0, "gen", frac=True)[0]
    fem = _fold(lemma).endswith(("а", "я"))  # тысяча / минута / гривна; миллион, процент, рубль, евро are not
    out = []
    for ws, tag in _dec_items(lang, n, frac, dsep):
        typ, c, g = tag
        if typ == "half" and n == 1:
            continue  # 1.5 with a noun is "полтора / полторы" ("один с половиной миллион" is not said)
        if typ in ("half", "poltora") and (g == "f") != fem:
            continue  # the count agrees with the noun: "полтора миллиона", "полторы тысячи", never "полторы миллиона"
        if typ == "half":
            nfs = noun_form(lang, lemma, count_class(n), c)[:1]
        elif typ == "poltora":
            nfs = [gsg] if c in ("nom", "acc") else noun_form(lang, lemma, 2, c)[:1] + [gsg]
        else:
            nfs = [gsg]
        out += [(ws + [nf] + list(tail), tag) for nf in nfs]
    return out


def _with_ending(items: list[tuple], suf: str) -> list[tuple]:
    return [x for x in items if x[0] and _fold(x[0][-1]).endswith(suf)]


def _range_items(lang: str, d: dict) -> list[tuple]:
    lv, rv = int(d["l"]), int(d["r"])
    joins = {"ru": ["", "до", "или"], "uk": ["", "до", "або", "чи"]}[lang]
    frm = {"ru": ["от", "с"], "uk": ["від", "з"]}[lang]
    post = d.get("post")
    lemma, tail = None, []
    if post == "%":
        lemma = PCT_NOUNS[lang][0]
    elif post:
        lemma, tail = UNITS[lang][post]
    out = []
    for c in CASES:
        ls = card_case(lang, card_nom(lang, lv), c, "m")
        rs = card_case(lang, card_nom(lang, rv), c, "m")
        for a, b in zip(ls, rs):
            if lemma:
                for nf in noun_form(lang, lemma, count_class(rv), c):
                    for j in joins:
                        if j != "до" or c == "gen":
                            out.append((a + ([j] if j else []) + b + [nf] + tail, ("card", c, "m")))
            else:
                for j in joins:
                    if j != "до" or c == "gen":
                        out.append((a + ([j] if j else []) + b, ("card", c, "m")))
    lg = card_case(lang, card_nom(lang, lv), "gen", "m")
    rg = card_case(lang, card_nom(lang, rv), "gen", "m")
    for a, b in zip(lg, rg):
        nfs = [noun_form(lang, lemma, 2, "nom")[0]] if lemma else []
        nfs2 = [noun_form(lang, lemma, count_class(rv), "gen")[0]] if lemma else []
        for f in frm:
            out.append(([f] + a + ["до"] + b + (nfs2 or []) + (tail if lemma else []), ("card", "gen", "m")))
        del nfs
    if not lemma:  # a score: "три ноль", "три — ноль", "три к нулю"
        lw, rw = card_nom(lang, lv), card_nom(lang, rv)
        out.append((lw + ["—"] + rw, ("card", "nom", "m")))
        if lang == "ru":
            for b in card_case(lang, rw, "dat", "m"):
                out.append((lw + ["к"] + b, ("card", "nom", "m")))
    if not lemma and lv <= 2100 and rv <= 2100:
        lo = {}
        for ws, tag in ord_readings(lang, lv):
            lo.setdefault(tag, []).append(ws)
        for ws, tag in ord_readings(lang, rv, elide=1000 <= rv <= 2100):
            for a in lo.get(tag, [])[:1]:
                for j in joins[:2]:
                    out.append((a + ([j] if j else []) + ws, tag))
    return out


# ---------------------------------------------------------------- dates, versions, chains, products, compounds, "-ка"

def _part_words(lang: str, s: str) -> list[str]:
    """One digit group said as a number ("20" -> "двадцать"; "09" -> "ноль девять"; "000" digit by digit)."""
    if len(s) > 1 and s[0] == "0":
        return (card_nom(lang, 0) + card_nom(lang, int(s[1:]))) if s[1] != "0" and len(s) == 2 else _digits(lang, s)
    try:
        return card_nom(lang, int(s))
    except M.Missing:
        return _digits(lang, s)


def _parts_items(lang: str, parts: list[str], joins: list[str]) -> list[tuple]:
    """Digit groups said one after another, as numbers or digit by digit: "три точка девять точка двадцать"."""
    out = []
    for ws in ([_part_words(lang, p) for p in parts], [_digits(lang, p) for p in parts]):
        for j in joins:
            seq = []
            for i, w in enumerate(ws):
                seq += ([j] if j and i else []) + w
            out.append((seq, ("other", "nom", "m")))
    return out


def _day_forms(lang: str, day: int) -> list[tuple]:
    """The day of a date: the neuter ordinal in the cases a date takes ("семнадцатое", "семнадцатого", "семнадцатому", "семнадцатым")."""
    out = []
    for ws, tag in ord_readings(lang, day):
        if tag[2] == "n" and tag[1] in ("nom", "gen", "dat", "ins"):
            out.append((ws, tag))
    return out


def _date_items(lang: str, d: dict) -> list[tuple]:
    """"17.12.2000": "семнадцатого декабря двухтысячного года" (the day in any case, the month in the genitive, the year
    an ordinal in the genitive, with or without the year noun unless the text has it next), and the all-ordinal
    colloquial "семнадцатое двенадцатое". Two digits: the ordinal of those two digits ("двадцатого")."""
    out = []
    yr = d.get("year")
    nxt = d.get("next") or ""
    noun_next = nxt in ("год", "года", "году", "г", "гг", "рік", "року", "році", "р", "рр")
    ynoun = NOUNS[lang][_YEAR_NOUN[lang]][1][1][0]
    ywords = []
    if yr is not None:
        for ws, tag in ord_readings(lang, int(yr), elide=len(yr) == 4):
            if tag[1] == "gen" and tag[2] == "m" and not (ws[-1].startswith("дву") and not ws[-1].startswith("двух")):
                ywords.append(ws)
    for dd, mm in d["dm"]:
        month = MONTHS_GEN[lang][mm - 1]
        mon_ord = [ws for ws, tag in ord_readings(lang, mm) if tag[1:] == ("nom", "n")]
        for dw, tag in _day_forms(lang, dd):
            if yr is None:
                out.append((dw + [month], tag))
                continue
            for yw in ywords:
                out.append((dw + [month] + yw, tag))
                if not noun_next:
                    out.append((dw + [month] + yw + [ynoun], tag))
        if yr is not None and d.get("parts") is not None:
            for dw, tag in _day_forms(lang, dd):
                if tag[1] == "nom":
                    for mw in mon_ord:
                        for yw in ywords:
                            out.append((dw + mw + yw, tag))
    return out


def _compound_stems(lang: str, n: int) -> list[str]:
    """The numeral as the first part of a compound adjective: 24 -> "двадцатичетырех" ("двадцатичетырехсильный"),
    12 -> "двенадцати", 100 -> "сто", 1 -> "одно", 2 -> "двух" (Ukrainian "дво" / "двох")."""
    if not 0 < n < 1000:
        return []
    stems = [""]
    for w in card_nom(lang, n):
        if w in ("один", "одна"):
            alts = ["одно"]
        elif w not in CARD[lang]:
            return []
        else:
            alts = list(CARD[lang][w][1][:1])
            if alts == ["ста"]:
                alts = ["сто"]
            elif alts and alts[0].endswith("носта"):
                alts = [alts[0][:-1] + "о"]
            if lang == "uk" and w in ("два", "три", "чотири"):
                alts += [{"два": "дво", "три": "три", "чотири": "чотири"}[w]]
        stems = [a + b for a in stems for b in alts]
    seen = []
    for x in stems:
        if x not in seen:
            seen.append(x)
    return seen


# colloquial -ка nouns of a number: nominative singular and genitive plural
KA = {"ru": {2: ("двойка", "двоек"), 3: ("тройка", "троек"), 4: ("четверка", "четверок"), 5: ("пятерка", "пятерок"),
              6: ("шестерка", "шестерок"), 7: ("семерка", "семерок"), 8: ("восьмерка", "восьмерок"),
              9: ("девятка", "девяток"), 10: ("десятка", "десяток"), 20: ("двадцатка", "двадцаток"),
              30: ("тридцатка", "тридцаток"), 40: ("сороковка", "сороковок"), 100: ("сотка", "соток"),
              200: ("двухсотка", "двухсоток"), 300: ("трехсотка", "трехсоток"), 400: ("четырехсотка", "четырехсоток"),
              500: ("пятисотка", "пятисоток"), 600: ("шестисотка", "шестисоток"), 700: ("семисотка", "семисоток"),
              800: ("восьмисотка", "восьмисоток"), 900: ("девятисотка", "девятисоток"), 1000: ("тысячка", None)},
      "uk": {2: ("двійка", "двійок"), 3: ("трійка", "трійок"), 4: ("четвірка", "четвірок"),
             5: ("п'ятірка", "п'ятірок"), 6: ("шістка", "шісток"), 7: ("сімка", "сімок"), 8: ("вісімка", "вісімок"),
             9: ("дев'ятка", "дев'яток"), 10: ("десятка", "десяток"), 20: ("двадцятка", "двадцяток"),
             30: ("тридцятка", "тридцяток"), 100: ("сотка", "соток"), 200: ("двохсотка", "двохсоток"),
             300: ("трьохсотка", "трьохсоток"), 400: ("чотирьохсотка", "чотирьохсоток"),
             500: ("п'ятисотка", "п'ятисоток"), 600: ("шестисотка", "шестисоток"), 700: ("семисотка", "семисоток"),
             800: ("восьмисотка", "восьмисоток"), 900: ("дев'ятисотка", "дев'ятисоток"), 1000: ("тисячка", None)}}


def _ka_items(lang: str, n: int) -> list[tuple]:
    """"600-ка", "30-ку", "20-ки": the colloquial noun of a round number ("шестисотка", "тридцатку", "двадцатки"),
    every case, singular and plural."""
    if n not in KA[lang]:
        return []
    lem, genpl = KA[lang][n]
    st = lem[:-2]
    sg = {"ru": ("ка", "ки", "ке", "ку", "кой", "ке"), "uk": ("ка", "ки", "ці", "ку", "кою", "ці")}[lang]
    pl = ("ки", genpl, "кам", "ки", "ками", "ках")
    out = []
    for c, e in zip(CASES, sg):
        out.append(([st + e], ("ka", c, "f")))
    for c, e in zip(CASES, pl):
        if e is not None:
            out.append(([e if e == genpl else st + e], ("ka", c, "pl")))
    return out


def _times_items(lang: str, d: dict) -> list[tuple]:
    """"3x4" / "1920x1080": "три на четыре", "три икс четыре", "четыре по девятьсот восемьдесят" ("по" with the
    accusative only after a count that is not one: "по пять", never "по одну")."""
    lv, rv = int(d["l"]), int(d["r"])

    def bases(n):
        b = card_nom(lang, n)
        return [b] + ([b[1:]] if len(b) > 1 and b[0] in ("одна", "один") and b[1] in _SCALE_FORMS[lang] else [])

    out = []
    for lw in bases(lv):
        for rw in bases(rv):
            for lg in ("m", "f", "n"):
                for a in card_case(lang, lw, "nom", lg)[:1]:
                    for rg in ("m", "f", "n"):
                        for b in card_case(lang, rw, "acc", rg)[:1]:
                            out.append((a + ["на"] + b, ("card", "nom", lg)))
                            if count_class(rv) != 0:
                                out.append((a + ["по"] + b, ("card", "nom", lg)))
                        for b in card_case(lang, rw, "dat", rg)[:1]:
                            out.append((a + ["по"] + b, ("card", "nom", lg)))
            out.append((lw + ["икс"] + rw, ("card", "nom", "m")))
    return out


def _fem(lang: str, ws: list[str]) -> list[str]:
    """A nominative cardinal agreeing with a feminine noun ("две минуты", "двадцать одна минута", "дві години")."""
    return card_case(lang, ws, "nom", "f")[0] if ws else ws


def _clock_items(lang: str, h: int, m: int) -> list[tuple]:
    hw = card_nom(lang, h)
    hours = [(hw, h)]
    if h > 12:
        hours.append((card_nom(lang, h - 12), h - 12))
    mins = []
    if m:
        mins.append(card_nom(lang, m))
        if m < 10:
            mins.append(card_nom(lang, 0) + card_nom(lang, m))
    else:
        mins += [[], card_nom(lang, 0) + card_nom(lang, 0)]
    hour_n, min_n = {"ru": ("час", "минута"), "uk": ("година", "хвилина")}[lang]
    hour_fem = lang == "uk"  # година is feminine, час masculine; минута / хвилина are feminine
    out = []
    for hv, hn in hours:
        hnoun = noun_form(lang, hour_n, count_class(hn), "nom")
        hv_n = _fem(lang, hv) if hour_fem else hv  # the number before the hour noun agrees with it
        for mv in mins:
            alts = [mv] + ([_fem(lang, mv)] if mv and _fem(lang, mv) != mv else [])  # bare "девять ноль две" too
            for a in alts:
                out.append((hv + a, ("card", "nom", "m")))
            if mv:  # a counted minute noun takes the feminine count ("две минуты", never "два минуты")
                out.append((hv_n + hnoun + _fem(lang, mv) + noun_form(lang, min_n, count_class(m), "nom"),
                            ("card", "nom", "f")))
            else:
                out.append((hv_n + hnoun, ("card", "nom", "f" if hour_fem else "m")))
    if lang == "uk":  # "об одинадцятій двадцять", "о п'ятій", "одинадцята нуль нуль"
        for hh in {h, h - 12 if h > 12 else h}:
            if hh < 1:
                continue
            f = adj_forms(ord_nom(lang, hh)[0][-1], lang)
            pre = ord_nom(lang, hh)[0][:-1]
            for c in ("nom", "loc", "gen"):
                for mv in mins:
                    out.append((pre + f[("f", c)][:1] + mv, ("ord", c, "f")))
    if lang == "ru":  # "до одиннадцати", "к одиннадцати двадцати", "после одиннадцати ноль ноль", "в час пятнадцать"
        for c in ("gen", "dat"):
            for a in card_case(lang, hw, c, "m"):
                out.append((a, ("card", c, "m")))
                for mv in mins:
                    if mv:
                        out.append((a + mv, ("card", c, "m")))
                        if m >= 10:
                            out += [(a + b, ("card", c, "m")) for b in card_case(lang, mv, c, "m")]
        if h in (1, 13):
            out += [(["час"] + mv, ("card", "nom", "m")) for mv in mins]
    return out


def _grouped_items(lang: str, raw: str) -> list[tuple]:
    groups = re.split(r"[   ]", raw)
    seqs = [[], []]  # the groups said as numbers; the second with feminine 1 / 2 before a dropped "тысячи"
    for k, gstr in enumerate(groups):
        if gstr.startswith("0") and len(gstr) > 1:
            ws = (card_nom(lang, 0) + card_nom(lang, int(gstr[1:]))) if gstr[1] != "0" else _digits(lang, gstr)
        else:
            ws = card_nom(lang, int(gstr))
        seqs[0] += ws
        fem = list(ws)
        if k == len(groups) - 2 and fem and fem[-1] in _GENDERED[lang]:
            fem[-1] = _GENDERED[lang][fem[-1]]["f"]
        seqs[1] += fem
    return [(x, ("other", "nom", "m")) for x in seqs]


_ZERO = {"ноль": "нуль", "ноля": "нуля", "нолю": "нулю", "нолем": "нулем", "ноле": "нуле"}


def cands(d: dict, lang: str) -> list[str]:
    k = d["kind"]
    if k == "skip":
        return []
    sign = d.get("sign")
    items = _cands_items(d, lang)
    out = _strs(items)
    if lang == "ru":  # "ноль" is also said "нуль"
        out += [v for v in dict.fromkeys(" ".join(_ZERO.get(w, w) for w in c.split()) for c in out) if v not in out]
    if sign:
        w = {"ru": ("минус", "плюс"), "uk": ("мінус", "плюс")}[lang][0 if sign in "-−" else 1]
        out = [f"{w} {c}" for c in out]
    if d.get("pre") == "№":
        out = [f"номер {c}" for c in out]
    return out


def _cands_items(d: dict, lang: str) -> list[tuple]:
    k = d["kind"]
    if k == "clock":
        return _rank(_clock_items(lang, d["h"], d["m"]), d, lang)
    if k == "range":
        return _rank(_range_items(lang, d), d, lang)
    if k == "date":
        items = _date_items(lang, d)
        if d.get("parts"):
            items += _parts_items(lang, d["parts"], [{"ru": "точка", "uk": "крапка"}[lang], ""])
        if d.get("chain"):
            items += _parts_items(lang, d["chain"], ["", "тире"])
        return _rank(items, d, lang, prefer_card=False)
    if k in ("version", "chain"):
        joins = [{"ru": "точка", "uk": "крапка"}[lang], ""] if k == "version" else ["", "тире"]
        items = _parts_items(lang, d["parts"], joins)
        if k == "version" and all(len(p) == 3 for p in d["parts"][1:]) and len(d["parts"][0]) <= 3 and (
                not d["parts"][0].startswith("0")):
            n = int("".join(d["parts"]))  # "1.000.000": dot-grouped
            items += card_readings(lang, n, genders=("m", "f", "n"), partial=False)
        if d.get("dm"):
            items += _date_items(lang, d)
        return _rank(items, d, lang, prefer_card=False)
    if k == "times":
        return _times_items(lang, d)
    raw, frac, dsep = d["int"], d.get("frac"), d.get("dsep")
    n = M._int(raw)
    if n > 999_999_999_999:
        return []
    if k == "compound":
        return [([st + d["adj"]], ("other", "nom", "m")) for st in _compound_stems(lang, n)]
    if d.get("codepart") and k in ("int", "dec"):
        if frac is not None:
            return _dec_items(lang, n, frac, dsep)
        items = card_readings(lang, n, genders=("m", "f", "n")) + [(w, ("other", "nom", "m")) for w in _chunked(lang, raw)]
        if d.get("afterdigit") and 1900 <= n <= 2100:  # "51511-2000": the year of a standard
            items += [x for x in ord_readings(lang, n, elide=True) if x[1][1:] == ("gen", "m")]
        return _rank(items, d, lang, prefer_card=False)
    if k == "code":
        return [(w, ("other", "nom", "m")) for w in _chunked(lang, raw)] + (
            [] if raw.startswith("0") else card_readings(lang, n, partial=False))
    if k in ("ord", "decade"):
        try:
            ords = ord_readings(lang, n, elide=True)
        except M.Missing:
            if not _ka_items(lang, n):
                raise
            ords = []  # the noun readings do not need the ordinal
        items = card_readings(lang, n, genders=("m", "f", "n")) + ords + _ka_items(lang, n)
        return _rank(_with_ending(items, d["suf"]), d, lang, prefer_card=False)
    if k == "pct":
        lemmas = PCT_NOUNS[lang]
        items = []
        for lem in lemmas:
            items += _dec_quantity(lang, n, frac, dsep, lem) if frac is not None else _quantity(lang, n, lem)
        return _rank(items, d, lang)
    if k == "unit":
        lem, tail = UNITS[lang][d["unit"]]
        items = (_dec_quantity(lang, n, frac, dsep, lem, tail) if frac is not None else _quantity(lang, n, lem, tail))
        if tail and d["unit"] in ("°C", "°С"):
            items += (_dec_quantity(lang, n, frac, dsep, lem) if frac is not None else _quantity(lang, n, lem))
        for t2 in ALT_TAILS.get((lang, d["unit"]), []):
            items += (_dec_quantity(lang, n, frac, dsep, lem, t2) if frac is not None else _quantity(lang, n, lem, t2))
        return _rank(items, d, lang)
    if k == "cur":
        lem = CUR[lang][d["cur"]]
        if frac is not None:
            return _rank(_cents_items(lang, n, frac, dsep, lem), d, lang)
        return _rank(_quantity(lang, n, lem), d, lang)
    if d.get("scale"):
        lem = SCALE_NOUN[lang][d["scale"]]
        if frac is not None:
            items = _dec_quantity(lang, n, frac, dsep, lem)
        else:
            items = _quantity(lang, n, lem)
        if d.get("klet") and frac is None:  # "5К" said as the letter: "пять ка", "пять кей"
            names = ["ка", "кэ"] + (["кей"] if d["klet"] in "kK" else [])
            items += [(ws + [nm], tag) for ws, tag in card_readings(lang, n, cases=("nom",)) for nm in names]
        return _rank(items, d, lang)
    if d.get("yabbr"):
        return _rank(_year_abbr(lang, n, d["yabbr"]), d, lang, prefer_card=False)
    if k == "dec":
        items = _dec_items(lang, n, frac, dsep)
        if dsep == "." and len(frac) == 3:  # "1.000": a dot-grouped thousands figure
            items += card_readings(lang, n * 1000 + int(frac), cases=("nom",))
        if dsep == "." and len(frac) == 2 and n <= 23 and int(frac) <= 59:  # "8.27" said as a time
            items += _clock_items(lang, n, int(frac))
        if dsep == "." and len(frac) == 2 and _valid_dm(n, int(frac)):  # "5.10" said as a date: "пятое октября"
            items += _date_items(lang, {"dm": [(n, int(frac))], "year": None, "next": d.get("next")})
        return _rank(items, d, lang)
    if d.get("grouped"):
        items = card_readings(lang, n, genders=("m", "f")) + _grouped_items(lang, raw)
        if 1000 < n < 10**6 and n % 1000 == 500:
            items += _dec_quantity(lang, n // 1000, "5", ",", SCALE_NOUN[lang][1000])
        return _rank(items, d, lang)
    # plain integer: every case / gender of the cardinal, and (up to 2100: years, dates, "4 корректирующая волна")
    # every form of the ordinal; 5-7 digits may also be a code said in chunks ("122746")
    items = _int_items(lang, n, d)
    if 1000 < n < 10**6 and n % 1000 == 500:  # "6500" -> "шесть с половиной тысяч"
        items += _dec_quantity(lang, n // 1000, "5", ",", SCALE_NOUN[lang][1000])
    if len(raw) >= 5:
        items += [(w, ("other", "nom", "m")) for w in _chunked(lang, raw)]
    return _rank(items, d, lang)


def prefetch(d: dict, lang: str) -> None:
    """Touch every grammar key a span needs before any word is inflected (ml_cands.needed records keys through the
    placeholder table, whose "_" words the paradigms cannot inflect)."""
    if d["kind"] in ("skip",):
        return
    ns = set()
    if d["kind"] == "clock":
        ns |= {d["h"], d["m"], 0, d["h"] - 12 if d["h"] > 12 else d["h"]}
        for x in (d["h"], d["h"] - 12):
            if x >= 1:
                M.g(lang, "ord", x)
    elif d["kind"] == "range":
        lv, rv = int(d["l"]), int(d["r"])
        ns |= {lv, rv}
        for x in (lv, rv, rv % 100):
            if 0 < x <= 2100:
                _touch_ord(lang, x)
    elif d["kind"] in ("date", "version", "chain", "times"):
        if d["kind"] == "times":
            ns |= {int(d["l"]), int(d["r"])}
        else:
            ns |= set(range(0, 100)) | {0}
            for p in d.get("parts") or d.get("chain") or []:
                ns |= {int(p), int(p[1:]) if len(p) > 1 else 0} | {int(c) for c in p}
                if len(p) <= 12:
                    ns.add(int(p))
            if d["kind"] == "version" and d["parts"] and all(len(p) == 3 for p in d["parts"][1:]):
                ns.add(int("".join(d["parts"])))
            if d.get("year"):
                y = int(d["year"])
                ns |= {y, y % 100}
                for x in (y, y % 100):
                    _touch_ord(lang, x)
            for dd, mm in d.get("dm") or []:
                for x in (dd, mm):
                    _touch_ord(lang, x)
                    ns.add(x)
    else:
        n = M._int(d["int"])
        if n > 999_999_999_999:
            return
        ns.add(n)
        frac = d.get("frac")
        if frac is not None:
            ns |= {int(frac), 0, n * 1000 + int(frac) if len(frac) == 3 else 0}
            ns |= {int(c) for c in frac}
            if len(frac) == 2:
                ns.add(int(frac[1]))
            try:
                M.g(lang, "dec", n, frac)
            except M.Missing:
                pass
        for gstr in re.split(r"[   ]", d["int"]):
            ns.add(int(gstr))
            ns.add(int(gstr[1:]) if len(gstr) > 1 else 0)
        if d["kind"] == "code" or (len(d["int"]) >= 5 and d["int"].isdigit()):
            ns |= set(range(0, 100)) | {int(d["int"][i:i + 3]) for i in range(len(d["int"]) - 2)}
        if n < 10**6:
            _touch_ord(lang, n)
        if n <= 2100 and n % 100:
            _touch_ord(lang, n % 100)
        for sc in (1000, 10**6, 10**9):
            if n >= sc and n % sc == 0:
                ns.add(n // sc)
        if 1000 < n < 10**6:
            ns.add(n // 1000)
    for x in ns:
        if 0 <= x <= 999_999_999_999:
            try:
                M.g(lang, "card", x)
            except M.Missing:
                if d["kind"] not in ("date", "version", "chain"):  # their long groups fall back to digit by digit
                    raise


def _touch_ord(lang: str, n: int) -> None:
    if n < 1:
        return
    if n < 1000 or (n < 10**6 and n % 1000):
        try:
            M.g(lang, "ord", n)
        except M.Missing:
            pass  # reading the span raises it again where the ordinal is needed
