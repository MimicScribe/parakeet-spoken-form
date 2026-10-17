"""Ukrainian number words -> values (ml_uk_words, 2026-10-17). Local, no GPU: python3 test_ml_uk_words.py

Red on the old code: there was no Ukrainian words -> value reader (ml_multi.parse_numbers has no uk; ml_slav only
generates readings), so none of these spans could be found."""
import ml_uk_words as U


def tags(text):
    return [(s["text"], s["kind"], s["reading"]) for s in U.find(text)]


def one(text, written, kind, reading=None):
    got = tags(text)
    assert len(got) == 1, (text, got)
    assert got[0][:2] == (written, kind), (text, got)
    if reading is not None:
        assert got[0][2] == reading, (text, got)


def none(text):
    assert tags(text) == [], (text, tags(text))


# ---------------------------------------------------------------- cardinals, every case, scale groups
one("Мені двадцять один рік.", "21", "int", "двадцять один")
one("Розкопали — двадцять два трупи.", "22", "int")
one("Це тривало близько двадцяти чотирьох секунд.", "24", "int", "двадцяти чотирьох")
one("сто двадцять п'ять тисяч триста", "125 300", "int")
one("сто дев'яносто вісім тисяч сімсот двадцять", "198 720", "int")
one("Зібралося близько десяти тисяч озброєних чоловіків.", "10 000", "int")
one("На острові живуть приблизно чотири мільйони людей.", "4 000 000", "int")
one("В умовах двох", "2", "int")
one("Мав близько чотирьохсот червоноармійців", "400", "int", "чотирьохсот")
one("Народу було зо дві тисячі.", "2 000", "int")  # a count >= 1000 is grouped: never read as a year
one("тисяча дев'ятсот", "1 900", "int")
one("Нуль", "0", "int")
assert [s["value"] for s in U.find("Спершу давав десять червінців, потім двадцять, далі вже й сто.")] == [10, 20, 100]
assert [s["value"] for s in U.find("п'ять двадцять")] == [5, 20]  # places must descend: two numbers
assert [s["value"] for s in U.find("двадцять, тридцять")] == [20, 30]  # punctuation splits
# Typographic apostrophes and stress marks read like the ASCII apostrophe
one("П’ятнадцять", "15", "int")
one("Пʼять", "5", "int")

# ---------------------------------------------------------------- ordinals, years, decades
one("У тисяча дев'ятсот сорок п'ятому році війна скінчилася.", "1945", "int", "тисяча дев'ятсот сорок п'ятому")
one("У тисячі дев'ятсот сорок п'ятому році", "1945", "int")  # the genitive thousand of Ukrainian years
# review 2026-10-17 (red on the old code): an ordinal 1000-2099 is a year only before the year noun
one("У тисячі дев'ятсот сорок п'ятому", "1945-му", "ord")
one("Це був тисяча п'ятсот перший день.", "1501-й", "ord")
one("Сьогодні прийшов двохтисячний гість.", "2000-й", "ord")
# fractions without "цілих": no span (the old code gave an int numerator and an ord 10 / 100 with an invented value)
none("Показник становить п'ять десятих відсотка.")
none("Різниця лише три тисячних.")
none("двадцять п'ять сотих")
one("Дві цілих п'ять десятих.", "2,5", "dec")
one("десята година", "10-а", "ord")  # a lone ordinal stays an ordinal
# a number said with a conjunction across a scale word: both parts refused (the old code gave 20 and 5 000)
none("Там було двадцять й п'ять тисяч людей.")
none("сто й десять")
one("Прем'єра запланована на осінь дві тисячі двадцятого року.", "2020", "int")
one("У дві тисячі п'ятому році", "2005", "int")
assert U.find("дві тисячі двадцятого року")[0].get("year")
one("Сходження було здійснено двадцять дев'ятого травня.", "29-го", "ord")
one("Це буде вже двадцять п'ята експедиція.", "25-а", "ord")
one("сто першим", "101-м", "ord")
one("О сьомій годині", "7-й", "ord")
one("сьому годину", "7-у", "ord")
one("В умовах двадцятих років не можна було", "20-ті", "decade")
one("друга половина", "2-а", "ord")
one("На другий день після припадку", "2-й", "ord")

# ---------------------------------------------------------------- percent, currency, units (the noun joins the span)
one("сто відсотків", "100%", "pct", "сто відсотків")
one("два відсотки", "2%", "pct")
one("Коли дасте п'ять процентів", "5%", "pct")
one("Припустімо — нових п'ятсот доларів.", "$500", "cur")
one("Усього за сто гривень", "100 грн", "cur")
one("двадцять одна копійка", "21 коп.", "cur")
one("більш ніж на два кілометри", "2 км", "unit")
one("до трьох літрів", "3 л", "unit")
one("Ще біля чотирьох мільйонів тонн!", "4 000 000 т", "unit")
one("сімсот літр золота", "700", "int")  # "літр" after 700 is the old unit "літра", not litres: no unit

# ---------------------------------------------------------------- decimals
one("три цілих п'ять десятих", "3,5", "dec", "три цілих п'ять десятих")
one("нуль цілих двадцять п'ять сотих", "0,25", "dec")

# ---------------------------------------------------------------- pronoun / determiner / idiom uses (precision)
none("Один з них сказав, що це багато.")
none("Коли вийшов за двері, один з червоноармійців махнув рукою:")
none("одні пішли, другі лишилися")
none("Вони допомагали один одному.")
none("Одного разу я пішов.")
none("Не один прийшов.")
none("А по дорозі не одну тисячу повстанців зустрінемо.")
none("Усі до одного рубля!")
none("Тисячі людей вийшли.")
none("мільйони глядачів")
none("Ахіллесова п'ята — слабке місце")
none("Душа в п'яти втекла.")
none("У сьому твоя безпека.")
none("Коли, може, краще було утопитися, ніж на сім світі жити!")
none("Я поїхав до друга.")
none("По-перше, по-друге")
none("двадцять-тридцять")  # hyphen-joined: one token, never a span
none("Богданові пішло лише, мабути, двадесять сьоме літо.")  # an archaic number word before: no partial value
none("Хіба сто й десять")  # one number said with a conjunction (110): its parts would get wrong values
none("ми ще мали тисячі три місцевих селян")  # "about three thousand", not 1003
none("близько двох з половиною годин.")
none("Кругла сорока сіла на тин.")
none("тисячний натовп")
# a bare one-form is a count before a counted noun
one("Одну хвилину я собі дозволю.", "1", "int")
one("мільйон доларів", "$1 000 000", "cur")
one("по сім за цибулю", "7", "int")
one("п'ята частина полку", "5-а", "ord")

# ---------------------------------------------------------------- label clean-up
assert U.clean("Не всi, правда.") == "Не всі, правда."  # Latin look-alike letters inside Cyrillic words
assert U.clean("i пішов") == "і пішов"
assert U.clean("П’ять") == "П'ять"
assert U.clean("-") is None
assert U.clean("У 2020 році") is None
assert U.clean("Мова.\t\t2\t0\ttwenties") is None
assert U.clean("Це Facebook") is None
assert U.clean("сто %") is None

# ---------------------------------------------------------------- span offsets
t = "Мені двадцять один рік, а йому п'ять відсотків."
for s in U.find(t):
    assert t[s["a"]:s["b"]] == s["reading"], s
print("ok")
