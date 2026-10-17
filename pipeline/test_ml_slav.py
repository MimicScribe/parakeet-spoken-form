"""Russian / Ukrainian spoken-form candidates and decisions (ml_slav, 2026-10-16). Local, no GPU:
python3 test_ml_slav.py

Red on the old code: ml_cands had no ru / uk span tables (KeyError on "ru"), and ml_decide / ml_p3_core keys dropped
every Cyrillic letter (each reading's key was empty). The last block cross-checks the hand-written inflection tables
against a morphological dictionary when pymorphy3 (+ its ru / uk dictionaries) is installed; it is skipped otherwise."""
import ml_cands as M
import ml_decide as D
import ml_p3_core as C


def cands(cap, lang, text=None):
    sp = [s for s in M.spans(cap, lang) if text is None or s.text == text]
    assert sp, (cap, [s.text for s in M.spans(cap, lang)])
    return sp[0].cands


def has(cap, lang, text, want):
    cs = cands(cap, lang, text)
    assert want in cs, (cap, want, len(cs), cs[:12])
    assert len(cs) <= 120, (cap, len(cs))  # bounded
    return cs


# ---------------------------------------------------------------- the maintainers' examples (case-inflected readings)
assert has("в 1939 году", "ru", "1939", "тысяча девятьсот тридцать девятом")[0] == "тысяча девятьсот тридцать девятом"
cs = has("с 5 до 10 процентов", "ru", "5", "пяти")
assert cs[0] == "пяти"
assert has("с 5 до 10 процентов", "ru", "10", "десяти")[0] == "десяти"
has("2020 года", "ru", "2020", "двадцатого")
has("2020 года", "ru", "2020", "две тысячи двадцатого")
assert cands("в 90-х", "ru", "90-х") == ["девяностых"]
# compound numerals inflect every part
has("из 1939 стран", "ru", "1939", "тысячи девятисот тридцати девяти")
# ordinals above 999 that end in a scale are one fused word
assert has("в 2000 году", "ru", "2000", "двухтысячном")[0] == "двухтысячном"
has("у 2000 році", "uk", "2000", "двохтисячному")
# Ukrainian
assert has("у 2003 році", "uk", "2003", "дві тисячі третьому")[0] == "дві тисячі третьому"
assert has("З 2009 року ми", "uk", "2009", "дві тисячі дев'ятого")[0] == "дві тисячі дев'ятого"
has("понад 5 відсотків", "uk", "5", "п'яти")
has("понад 5 відсотків", "uk", "5", "п'ятьох")
has("Це 100%.", "uk", "100%", "сто відсотків")
has("Це 100%.", "uk", "100%", "сто процентів")
assert cands("двох кінців 13-ї боблі", "uk", "13-ї") == ["тринадцятої"]
has("У 1980-ті роки", "uk", "1980-ті", "вісімдесяті")
has("50-60 центнерів", "uk", "50-60", "п'ятдесят шістдесят")
has("50-60 центнерів", "uk", "50-60", "від п'ятдесяти до шістдесяти")
print("maintainer examples ok")

# ---------------------------------------------------------------- kinds
has("Это 100%", "ru", "100%", "сто процентов")
has("на 2,5% выше", "ru", "2,5%", "две целых пять десятых процента")
has("на 2,5% выше", "ru", "2,5%", "два с половиной процента")
has("рост на 21%", "ru", "21%", "двадцать один процент")
has("около 3%", "ru", "3%", "трех процентов")
has("не четверти, 0,25.", "ru", "0,25", "ноль целых двадцать пять сотых")
has("не четверти, 0,25.", "ru", "0,25", "ноль двадцать пять")
has("около 1,5 млн человек", "ru", "1,5 млн", "полутора миллионов")
has("до 12 тыс. рублей", "ru", "12 тыс.", "двенадцати тысяч")
has("Обшивка 19 мм.", "ru", "19 мм", "девятнадцать миллиметров")
has("скорость 83 км/год", "uk", "83 км/год", "вісімдесят три кілометри на годину")
has("Все платят 30 $.", "ru", "30 $", "тридцать долларов")
has("27-28 градусов.", "ru", "27-28", "двадцать семь двадцать восемь")
has("27-28 градусов.", "ru", "27-28", "от двадцати семи до двадцати восьми")
has("Об 11:20 поліція", "uk", "11:20", "одинадцятій двадцять")
has("в 11:20 приехал", "ru", "11:20", "одиннадцать двадцать")
has("в 1939 г. началась", "ru", "1939 г.", "тысяча девятьсот тридцать девятом году")
has("реакторы № 1 и № 2", "ru", "№ 2", "номер два")
has("при -5 градусах", "ru", "-5", "минус пяти")
has("Типа по 10-ти бально", "ru", "10-ти", "десяти")
has("В 1917-м на", "ru", "1917-м", "тысяча девятьсот семнадцатом")
has("4 корректирующая волна", "ru", "4", "четвертая")  # an ordinal written without its ending
has("К 17 сентября 1939 года", "ru", "17", "семнадцатому")
has("Мой номер 8 904 436 096.", "ru", "8 904 436 096", "восемь девятьсот четыре четыреста тридцать шесть ноль девяносто шесть")
has("21 минута", "ru", "21", "двадцать одна")
has("2 хвилини", "uk", "2", "дві")
has("до 500 миллионов", "ru", "500", "пятиста")  # colloquial genitive, heard in FLEURS speech
has("собрать 1,2 миллиона", "ru", "1,2", "один и два")
has("около 11:00 по", "ru", "11:00", "одиннадцати ноль ноль")
has("В 01:15 в субботу", "ru", "01:15", "час пятнадцать")
has("насчитала 6500 протестующих", "ru", "6500", "шесть с половиной тысяч")
has("из 1200 компаний", "ru", "1200", "тысяча двухсот")
has("1, 2, 3", "ru", "1", "раз")
has("Сегодня 7528 лето.", "ru", "7528", "семь тысяч пятьсот двадцать восьмое")
has("То есть 22 500 это", "ru", "22 500", "двадцать две пятьсот")
print("kinds ok")

# ---------------------------------------------------------------- what must stay unread (the caption is dropped)
for cap, lang in [("0123sfast", "ru"), ("4x980T", "ru"), ("0x1F", "ru"), ("5000-метровый бег", "ru"),
                  ("2,5-литровый", "ru"), ("3.9.20a", "ru")]:
    assert not M.all_digits_spanned(cap, lang), cap
for cap, lang in [("48-річний міст", "uk"), ("5-летний план", "ru"), ("40к", "ru"), ("16.04.2022 года", "ru"),
                  ("версия 1.2.3", "ru")]:  # skipped before the slav extension
    assert M.all_digits_spanned(cap, lang), cap
for cap, lang in [("в 1939 году", "ru"), ("Це 100%.", "uk"), ("У 1980-ті роки", "uk"), ("с 5 до 10 процентов", "ru")]:
    assert M.all_digits_spanned(cap, lang), cap
# a hyphen ending no reading has: no candidates (never a wrong ending)
assert cands("15-ка на полянке", "ru", "15-ка") == []
print("unreadable spans ok")

# ---------------------------------------------------------------- spelling: no "ё", ASCII apostrophe
for cap, lang in [("в 1914 году", "ru"), ("на 4%", "ru"), ("з 3 до 4", "uk"), ("у 1995 році", "uk"), ("49 разів", "uk")]:
    for s in M.spans(cap, lang):
        assert all("ё" not in c and "’" not in c and "ʼ" not in c for c in s.cands), (cap, s.cands)
assert "п'ятому" in cands("у 1995 році", "uk", "1995")[0]
print("spelling ok")

# ---------------------------------------------------------------- the grammar keys a ru / uk caption needs are recorded
keys = M.needed(["в 1939 году", "с 2,5% до 1 000 000", "у 2003 році 15-го"], "ru") | M.needed(["у 2003 році"], "uk")
assert "ru|card|1939|||0|0" in keys and "ru|ord|1939|||0|0" in keys and "uk|ord|2003|||0|0" in keys
assert not (keys - set(M.table())), sorted(keys - set(M.table()))[:5]
print("needed ok")

# ---------------------------------------------------------------- decisions: Cyrillic keys, ext context
assert D.key("тысяча девятьсот") == ""  # wave-1 mode: Latin letters only (unchanged)
D.set_script("ru")
try:
    assert D.key("Тысяча девятьсот") == "тысячадевятьсот"
    assert D.key("четвёртый") == D.key("четвертый")
finally:
    D.set_script("de")
# row CER counts Latin and Cyrillic letters in every caller (windows / seam halves / TTS checks never set a language)
assert C.letters("один два") == "одиндва" and C.letters("Dev ять") == "devять"
assert C.letters("Дев'ять років") == "девятьроків" and C.letters("їжак ёж") == "іжакеж"
assert C.cer("девять", "девять") == 0.0


class FakeCTC:
    def __init__(self, lang, spoken, greedy, per_letter=3.0):
        C.set_script(lang)
        self.lang, self.spoken, self.g, self.k = lang, C.letters(spoken), greedy, per_letter

    def greedy(self, lp):
        return self.g

    def nll(self, lp, text):
        C.set_script(self.lang)
        return self.k * D._lev(C.letters(text), self.spoken)


def label(cap, lang, spoken, greedy, ext):
    r = D.decide(FakeCTC(lang, spoken, greedy), cap, None, ext=ext)
    if not all(s["decided"] for s in r["spans"]):
        return None
    return M.render(cap, M.spans(cap, lang), [s["pick"] for s in r["spans"]])


sp = "в тысяча девятьсот тридцать девятом году"
assert label("в 1939 году", "ru", sp, sp, sp) == sp
sp = "с пяти до десяти процентов"
assert label("с 5 до 10 процентов", "ru", sp, sp, sp) == sp
# the transcript holds a different ending than the CTC best: refused (never a changed word)
assert label("в 1939 году", "ru", "в тысяча девятьсот тридцать девятом году", "в тысяча девятьсот тридцать девятом году",
             "в тысяча девятьсот тридцать девятый год") is None
sp = "у дві тисячі третьому році"
assert label("у 2003 році", "uk", sp, sp, sp) == sp
# ext_pick: the weak CTC slightly prefers another form, the transcript holds one reading in place: it decides
assert label("вот 18 стих", "ru", "вот восемнадцат стих", "вот восемна стих", "вот восемнадцатый стих") == \
    "вот восемнадцатый стих"
# ... but not when the CTC is strongly against it (the audio says something else)
_s = "вот восемнадцать стих"
assert D.decide(FakeCTC("ru", _s, "вот восемна стих", per_letter=12.0), "вот 18 стих", None,
                ext="вот восемнадцатый стих")["spans"][0]["decided"] is False
row = {"audio_filepath": "x.wav", "duration": 3.0, "text": "в 1939 году", "lang": "ru", "src": "yodas", "id": "a",
       "origin": "new"}
sp = "в тысяча девятьсот тридцать девятом году"
res = C.make_row(row, D.decide(FakeCTC("ru", sp, sp), row["text"], None, ext=sp), ext=sp)
assert res.get("row", {}).get("text") == sp, res
assert C.make_row(dict(row, text="в 1939 г. и 5 тыс."), {"spans": [], "greedy": ""})["reason"] in ("span_mismatch", "no_spans")
# ru / uk: a CTC-confident ending the transcript does not have is refused (uk dry run: "восьмі" vs "восьми")
sp = "псалом восьмі тільки"
r = D.decide(FakeCTC("uk", sp, sp, per_letter=8.0), "Псалом 8. Тільки", None, ext="псим восьми тільки")  # the real row: the veto finds no neighbour
assert not r["spans"][0]["decided"] and r["spans"][0]["by"] == "ext_missing", r["spans"][0]
has("у 1945 році", "uk", "1945", "тисячі дев'ятсот сорок п'ятому")
print("decisions ok")

# ---------------------------------------------------------------- inflection tables vs a morphological dictionary
try:
    import pymorphy3
except ImportError:
    pymorphy3 = None
if pymorphy3 is None:
    print("dictionary cross-check skipped (pip install pymorphy3 pymorphy3-dicts-ru pymorphy3-dicts-uk)")
else:
    import ml_slav as S

    TAG = dict(zip(S.CASES, ("nomn", "gent", "datv", "accs", "ablt", "loct")))
    NOUN_ALLOW = {"лет", "килограммов", "граммов", "году"}  # "пять лет"; colloquial genitive plurals; "в году"
    ALLOW = {"восьмьюстами", "чотирьома"}  # standard variants the dictionary does not list
    # colloquial hundreds genitives, said by FLEURS ru readers ("до 500 миллионов" -> "до пятиста миллионов")
    ALLOW |= {"двохста", "трьохста", "чотирьохста", "п'ятиста", "шестиста", "семиста", "восьмиста", "дев'ятиста"}
    ALLOW |= {"двухста", "трехста", "четырехста", "пятиста", "шестиста", "семиста", "восьмиста", "девятиста"}
    bad = []
    for lang in ("ru", "uk"):
        an = pymorphy3.MorphAnalyzer(lang=lang)

        def dict_forms(word, case, pos=("NUMR",), extra=()):
            out = set()
            for p in an.parse(word):
                if p.tag.POS not in pos or p.word != word:
                    continue
                for f in p.lexeme:
                    if TAG[case] in f.tag and all(e in f.tag for e in extra):
                        out.add(f.word.replace("ё", "е").replace("ʼ", "'").replace("’", "'"))
            return out

        for w, par in S.CARD[lang].items():
            for c, alts in zip(S.CASES, par):
                df = dict_forms(w, c)
                if df and c != "acc" and (alts[0] not in df or not set(alts) <= df | ALLOW):
                    bad.append((lang, w, c, alts, sorted(df)))
        for n in (1, 2, 3, 4, 5, 7, 9, 20, 40, 90):
            w = S.ord_nom(lang, n)[0][-1]
            ours = S.adj_forms(w, lang)
            for (gn, c), alts in ours.items():
                ex = {"m": ("masc", "sing"), "f": ("femn", "sing"), "n": ("neut", "sing"), "pl": ("plur",)}[gn]
                df = dict_forms(w, c, pos=("ADJF",), extra=ex)
                if df and alts[0] not in df:
                    bad.append((lang, w, gn, c, alts, sorted(df)))
        for lem, (gnd, sg, pl) in S.NOUNS[lang].items():
            if lem in ("евро", "євро"):
                continue
            for num, par in (("sing", sg), ("plur", pl)):
                for c, alts in zip(S.CASES, par):
                    df = dict_forms(lem, c, pos=("NOUN",), extra=(num,)) if num == "plur" else {
                        x for x in dict_forms(lem, c, pos=("NOUN",)) if x not in dict_forms(lem, c, pos=("NOUN",), extra=("plur",))}
                    if df and not set(alts) <= df | NOUN_ALLOW:
                        bad.append((lang, lem, num, c, alts, sorted(df)))
    assert not bad, bad[:10]
    print("dictionary cross-check ok")
print("all ok")

# clock: a counted minute (and the Ukrainian hour, година) takes the feminine count
import ml_slav as _S
_ck = {l: {" ".join(w) for w, _ in _S._clock_items(l, h, m)} for l, h, m in (("ru", 9, 2), ("uk", 2, 22))}
assert "девять часов две минуты" in _ck["ru"] and "девять часов два минуты" not in _ck["ru"], _ck["ru"]
assert "дві години двадцять дві хвилини" in _ck["uk"] and "два години двадцять два хвилини" not in _ck["uk"], _ck["uk"]
print("clock agreement ok")

# prices with cents are read (they were dropped): both nouns agree, the minor unit's own gender
_pc = {(l, t): M.spans(t, l)[0].cands for l, t in (("ru", "Цена 5,99 € сегодня."), ("ru", "Всего 2,02 ₽."),
                                                    ("uk", "Ціна 12,75 грн."), ("ru", "Всего 0,99 $."))}
assert _pc[("ru", "Цена 5,99 € сегодня.")][0] == "пять евро девяносто девять центов", _pc[("ru", "Цена 5,99 € сегодня.")][:3]
assert "две копейки" in " | ".join(_pc[("ru", "Всего 2,02 ₽.")]) and "два копейки" not in " | ".join(_pc[("ru", "Всего 2,02 ₽.")])
assert "дванадцять гривень сімдесят п'ять копійок" in _pc[("uk", "Ціна 12,75 грн.")]
assert "девяносто девять центов" in _pc[("ru", "Всего 0,99 $.")]
print("cents ok")


# ---------------------------------------------------------------- skipped span types now read (never a changed value)
def nohave(cap, lang, text, bad):
    cs = cands(cap, lang, text)
    assert bad not in cs, (cap, bad)


# 1. codes with letters + dash + number: the number is read; scores
has("Т-62 был", "ru", "62", "шестьдесят два")
has("БЗ-58-2 тут", "ru", "58", "пятьдесят восемь")
has("БЗ-58-2 тут", "ru", "2", "два")
has("по ГОСТу 51511-2000 это", "ru", "51511", "пятьдесят одна тысяча пятьсот одиннадцать")
has("по ГОСТу 51511-2000 это", "ru", "2000", "две тысячи")
has("по ГОСТу 51511-2000 это", "ru", "2000", "двухтысячного")
nohave("Т-62 был", "ru", "62", "минус шестьдесят два")
nohave("Т-62 был", "ru", "62", "шестьдесят второй")  # a code is not an ordinal
has("Счёт уже 3-0.", "ru", "3-0", "три ноль")
has("Счёт уже 3-0.", "ru", "3-0", "три нуль")
has("Счёт уже 3-0.", "ru", "3-0", "три — ноль")
has("Счёт уже 3-0.", "ru", "3-0", "три к нулю")
has("2-2 и 4-0 было.", "ru", "2-2", "два два")
has("2-2 и 4-0 было.", "ru", "4-0", "четыре ноль")
has("Ну, собственно, 1-0.", "ru", "1-0", "один ноль")
nohave("Ну, собственно, 1-0.", "ru", "1-0", "один к нулям")
has("рахунок 3-0", "uk", "3-0", "три нуль")
has("рахунок 3-0", "uk", "3-0", "три — нуль")
has("Т-62 був", "uk", "62", "шістдесят два")

# 2. glued letters: thousands, the letter, seconds, products
has("300к и 5К", "ru", "300к", "триста тысяч")
has("300к и 5К", "ru", "300к", "триста ка")
has("300к и 5К", "ru", "5К", "пять тысяч")
has("300к и 5К", "ru", "5К", "пять кэ")
nohave("300к и 5К", "ru", "5К", "пять тысяча")
has("шесть 5K экран", "ru", "5K", "пять кей")
has("за 1,5к рублей", "ru", "1,5к", "полторы тысячи")
has("Всего 4,4s было", "ru", "4,4s", "четыре целых четыре десятых секунды")
has("за 10ms", "ru", "10ms", "десять миллисекунд")
has("Всього 4,4s було", "uk", "4,4s", "чотири цілих чотири десятих секунди")
has("3x4 метра", "ru", "3x4", "три на четыре")
has("3x4 метра", "ru", "3x4", "три икс четыре")
has("1920x1080", "ru", "1920x1080", "тысяча девятьсот двадцать на тысячу восемьдесят")
has("4x980 штук", "ru", "4x980", "четыре по девятьсот восемьдесят")
nohave("3x4 метра", "ru", "3x4", "три на четыре четыре")
assert [s.kind for s in M.spans("4x980T", "ru")] == ["skip"]  # a trailing letter: still unread
assert [s.kind for s in M.spans("0x1F", "ru")] == ["skip"]  # hex

# 3. dates and versions
has("17.12.2000 года", "ru", "17.12.2000", "семнадцатое декабря двухтысячного")
nohave("17.12.2000 года", "ru", "17.12.2000", "семнадцатое декабря двухтысячного года")  # "года" is in the caption
has("от 17.12.2000.", "ru", "17.12.2000", "семнадцатого декабря двухтысячного года")
has("от 17.12.2000.", "ru", "17.12.2000", "семнадцатого декабря двухтысячного")
has("2017-03-05", "ru", "2017-03-05", "пятое марта две тысячи семнадцатого года")
has("3.9.20 версия", "ru", "3.9.20", "три точка девять точка двадцать")
has("3.9.20 версия", "ru", "3.9.20", "три девять двадцать")
has("3.9.20 версия", "ru", "3.9.20", "третье сентября двадцатого года")
has("10.0.190", "ru", "10.0.190", "десять точка ноль точка сто девяносто")
has("10.0.19044", "ru", "10.0.19044", "десять точка ноль точка один девять ноль четыре четыре")  # a group the table lacks
nohave("10.0.190", "ru", "10.0.190", "десятое")  # not a date
has("1.000.000 рублей", "ru", "1.000.000", "один точка ноль ноль ноль точка ноль ноль ноль")
has("12.05.2017 року", "uk", "12.05.2017", "дванадцяте травня дві тисячі сімнадцятого")
has("до 12.05.2010.", "uk", "12.05.2010", "дванадцятого травня дві тисячі десятого року")
has("5.10 числа", "ru", "5.10", "пятое октября")
has("123-45-67", "ru", "123-45-67", "сто двадцать три сорок пять шестьдесят семь")
has("17-12-2000", "ru", "17-12-2000", "семнадцатое декабря двухтысячного года")

# 4. glued compounds: the numeral fuses with the adjective
has("24-сильной", "ru", "24-сильной", "двадцатичетырехсильной")
assert cands("24-сильной", "ru") == ["двадцатичетырехсильной"]
assert cands("12-фушечный", "ru") == ["двенадцатифушечный"]
assert cands("5-летний план", "ru") == ["пятилетний"]
assert cands("100-летний", "ru") == ["столетний"]
assert cands("40-градусный", "ru") == ["сорокаградусный"]
assert cands("90-летие", "ru") == ["девяностолетие"]
assert cands("2-этажный", "ru") == ["двухэтажный"]
assert cands("1-комнатная", "ru") == ["однокомнатная"]
assert cands("200-метровый", "ru") == ["двухсотметровый"]
assert cands("125-миллиметровый", "ru") == ["стодвадцатипятимиллиметровый"]
assert "двадцятичотирисильної" in cands("24-сильної", "uk")
assert "тристоронній" in cands("3-сторонній", "uk")
assert cands("2,5-литровый", "ru") == []  # a fraction in a compound stays unread
assert cands("5000-метровый", "ru") == []

# 5. a lone zero, and the colloquial -ка nouns
for cap, lang in (("Нажимаем клавишу 0.", "ru"), ("fc равно 0.", "ru")):
    cs = cands(cap, lang, "0")
    assert "ноль" in cs and "нуль" in cs and "нуля" in cs and "нолю" in cs and "нулю" in cs, cs
assert "нуль" in cands("клавіша 0", "uk", "0") and "нуля" in cands("клавіша 0", "uk", "0")
has("новая 600-ка", "ru", "600-ка", "шестисотка")
assert cands("новая 600-ка", "ru", "600-ка") == ["шестисотка"]
assert cands("взял 30-ку", "ru", "30-ку") == ["тридцатку"]
assert cands("с 20-ки", "ru", "20-ки") == ["двадцатки"]
assert cands("на 10-ке", "ru", "10-ке") == ["десятке"]
assert cands("десять 10-ок", "ru", "10-ок") == ["десяток"]
assert cands("на 100-ку", "ru", "100-ку") == ["сотку"]
assert cands("500-кою", "uk", "500-кою") == ["п'ятисоткою"]
assert cands("20-ці", "uk", "20-ці") == ["двадцятці"]
assert cands("взяв 30-ку", "uk", "30-ку") == ["тридцятку"]
assert cands("15-ка на полянке", "ru", "15-ка") == []
print("slav skipped types ok")
