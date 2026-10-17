"""Russian synthetic sentences for spoken-form number coverage (2026-10-17), the Russian twin of ml_tts_uk.

Real Russian speech in the mix has years (few values above the bar) but almost no clock times, prices, phone numbers
or dates said aloud with a word-form transcript. Each class has carrier sentences whose case is fixed by the frame
("в 2025 году" locative, "с 1998 года" genitive, "стоит 240,50 ₽" nominative, "подписали 12.03.2021" genitive date),
and the spoken reading is built with the ml_slav inflection tables (case, gender, the 1 / 2-4 / 5+ noun rule).

A row is kept only when ml_cands.spans(caption, "ru") finds the expected spans and the intended reading is one of each
span's candidates; the label is rendered exactly like a real row (ml_p3_core.render_cap).

    plan(n, seed, report=None, classes=(), tag="tts9ru")  pure Python: rows for ml_tts_cloud (no voice fields yet)
    python3 ml_tts_ru.py plan 60 1                          print sample rows
    python3 ml_tts_ru.py needed 2000 1 out.json             grammar-table keys the plan needs that the table lacks
"""

import collections
import hashlib
import json
import random
import re
import sys

import ml_cands as M
import ml_p3_core as C
import ml_slav as S

LANG = "ru"
TAG = "tts9ru"
# class -> relative quota (year / count / big share the "int" cap in the mix)
CLASSES = {"year": 80, "count": 40, "big": 60, "price": 80, "clock": 80, "phone": 80, "date": 80, "pct": 70,
           "dec": 70, "ord": 70, "range": 70}
KINDS = {"year": ["int"], "count": ["int"], "big": ["int"], "price": ["cur"], "clock": ["clock"],
         "clock2": ["clock", "clock"], "phone": ["chain"], "phone2": ["int", "chain"], "date": ["date"], "pct": ["pct"],
         "dec": ["dec"], "ord": ["ord"], "range": ["range"]}
WORDS_MIN, WORDS_MAX = 6, 22
DEV_FRAC = 0.1
FORBIDDEN = re.compile(r"[0-9$€£¥%#/+°²³½¼¾ºª‰&₴₽()]")
LEADS_REPORT = ["По нашим данным, ", "Как сообщили, ", "Напомню, что ", "Насколько я знаю, ", "Говорят, что "]
LEADS_NEUTRAL = ["Кстати, ", "Итак, ", "Честно говоря, "]
PERSONAL = re.compile(r"(?<![" + S.L + r"])(?:я|мы|мой|моя|меня|наш\w*|давай|запиши)(?![" + S.L + r"])", re.I)
IMPERATIVE = re.compile(r"^(?:Звоните|Оставьте|Откройте|Давай)")


# ---------------------------------------------------------------- readings (ml_slav tables)

def card(n: int, case: str = "nom", g: str = "m", drop_one: bool = True) -> str:
    ws = S.card_nom(LANG, n)
    if drop_one and len(ws) > 1 and ws[0] in ("одна", "один") and ws[1] in ("тысяча",):
        ws = ws[1:]  # "тысяча пятьсот", the everyday form
    return " ".join(S.card_case(LANG, ws, case, g)[0])


def ordw(n: int, gn: str, case: str) -> str:
    """Ordinal n in gender/number gn ("m", "f", "n", "pl") and case, the standard (first) form."""
    base = S.ord_nom(LANG, n)[0]
    f = S.adj_forms(base[-1], LANG)[(gn, case)][0]
    return " ".join(base[:-1] + [f])


def noun(lemma: str, n: int, case: str) -> str:
    return S.noun_form(LANG, lemma, S.count_class(n), case)[0]


def gender(lemma: str) -> str:
    return S.NOUNS[LANG][lemma][0]


# ---------------------------------------------------------------- value schedules (spread, not random clumps)

def _years(rng, k):
    """k year values: every 1990-2030 once per round, then 2031-2060 (2035 / 2040 / 2050 twice per round)."""
    base = list(range(1990, 2031))
    fwd = list(range(2031, 2061)) + [2035, 2040, 2050]
    out = []
    while len(out) < k:
        a, b = base[:], fwd[:]
        rng.shuffle(a)
        rng.shuffle(b)
        while a or b:
            if a:
                out.append(a.pop())
            if a:
                out.append(a.pop())
            if b:
                out.append(b.pop())
    return out[:k]


def _small(rng):
    r = rng.random()
    if r < 0.5:
        return rng.randint(1, 20)
    if r < 0.85:
        return rng.randint(21, 99)
    return rng.choice([100, 30, 40, 50, 60, 70, 80, 90])


# ---------------------------------------------------------------- generators: (rng, i, ctx) -> (caption, [reading per span], kinds) | None

Y_FRAMES = [  # (template, case, era)
    ("Этот закон приняли в {y} году после долгих обсуждений.", "loc", "past"),
    ("Компания открыла свой первый офис в {y} году.", "loc", "past"),
    ("В {y} году мы планируем закончить строительство моста.", "loc", "future"),
    ("Новую линию метро обещают запустить в {y} году.", "loc", "future"),
    ("Он работает на этом заводе с {y} года.", "gen", "past"),
    ("Программа будет действовать до {y} года включительно.", "gen", "future"),
    ("После {y} года цены на жильё заметно изменились.", "gen", "past"),
    ("Эти данные собраны по состоянию на конец {y} года.", "gen", "past"),
    ("Стратегию развития города расписали до {y} года.", "gen", "future"),
    ("{y} год стал очень тяжёлым для всей отрасли.", "nom", "past"),
    ("Мы часто вспоминаем события {y} года.", "gen", "past"),
    ("Цель сократить выбросы поставили на {y} год.", "acc", "future"),
]


def g_year(rng, i, ctx):
    y = ctx["years"][i % len(ctx["years"])]
    era = "past" if y <= 2025 else "future"
    tpl, case, _ = rng.choice([f for f in Y_FRAMES if f[2] in (era, "any")])
    c = "nom" if case == "acc" else case  # inanimate accusative = nominative ("на две тысячи тридцатый год")
    return tpl.format(y=y), [ordw(y, "m", c)], KINDS["year"]


COUNT_FRAMES = [  # (template with {n} {w}, case, nouns)
    ("На дорогу до работы я трачу {n} {w}.", "acc", ["минута", "час"]),
    ("Совещание длилось около {n} {w}, и все устали.", "gen", ["минута", "час"]),
    ("Мы доехали до моря за {n} {w}.", "acc", ["час", "минута"]),
    ("До ближайшей остановки всего {n} {w}.", "nom", ["километр", "метр"]),
    ("Для этого рецепта нужно {n} {w} муки.", "nom", ["килограмм", "грамм"]),
    ("За неделю мы привезли {n} {w} воды.", "acc", ["литр", "тонна"]),
    ("Пробежка занимает у меня примерно {n} {w}.", "acc", ["минута"]),
]


def g_count(rng, i, ctx):
    tpl, case, nouns = rng.choice(COUNT_FRAMES)
    lem = rng.choice(nouns)
    n = _small(rng)
    if lem == "час":  # a plausible number of hours for these frames
        n = rng.randint(1, 12)
    return tpl.format(n=n, w=noun(lem, n, case)), [card(n, case, gender(lem))], KINDS["count"]


BIG_FRAMES = [  # (template, case, largest sensible value)
    ("Бюджет проекта составляет {n} рублей.", "nom", 10**11),
    ("Государство выделило на ремонт дорог {n} рублей.", "acc", 10**11),
    ("На счёте компании осталось около {n} долларов.", "gen", 10**11),
    ("В нашей области живёт примерно {n} человек.", "nom", 5 * 10**6),
    ("В прошлом году город посетили {n} туристов.", "nom", 2 * 10**7),
    ("Ущерб от наводнения оценивают в {n} рублей.", "acc", 10**11),
]


def _big_value(rng):
    r = rng.random()
    if r < 0.4:  # thousands
        return rng.choice([rng.randint(2, 99) * 1000, rng.randint(1, 9) * 100000 + rng.choice([0, 50000]),
                           rng.randint(11, 99) * 10000, rng.randint(1, 9) * 1000 + 500])
    if r < 0.8:  # millions
        return rng.choice([rng.randint(1, 9), rng.randint(10, 99), rng.randint(1, 9) * 100]) * 10**6 + rng.choice(
            [0, 0, 0, 500000, 200000])
    return rng.choice([rng.randint(1, 9), rng.randint(10, 50)]) * 10**9


def _group(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def g_big(rng, i, ctx):
    tpl, case, top = rng.choice(BIG_FRAMES)
    n = _big_value(rng)
    while n > top:
        n = _big_value(rng)
    if rng.random() < 0.3 and n >= 10**6 and n % 10**6 == 0 and n < 10**9:
        w = f"{n // 10**6} млн"
    elif rng.random() < 0.3 and n >= 10**9 and n % 10**9 == 0:
        w = f"{n // 10**9} млрд"
    else:
        w = _group(n)
    return tpl.format(n=w), [card(n, case)], KINDS["big"]


PRICE_FRAMES = [  # (template, case)
    ("Билет на концерт стоит {p}, и это недорого.", "nom"),
    ("Цена одной порции без скидки составляет {p}.", "nom"),
    ("За коммунальные услуги мы заплатили {p} в этом месяце.", "acc"),
    ("Стоимость доставки составляет {p} по всей стране.", "nom"),
    ("Этот товар в магазине у дома стоит {p}.", "nom"),
    ("Итого к оплате {p}, можно картой или наличными.", "nom"),
]
CUR_W = [("₽", "рубль", 5), ("руб.", "рубль", 1), ("$", "доллар", 2), ("€", "евро", 2)]


def _amount(lemma, n, case):
    return f"{card(n, case, gender(lemma))} {noun(lemma, n, case)}"


def g_price(rng, i, ctx):
    tpl, case = rng.choice(PRICE_FRAMES)
    sym, major, _ = rng.choices(CUR_W, [w for *_, w in CUR_W])[0]
    r = rng.random()
    n = rng.randint(1, 20) if r < 0.35 else rng.randint(21, 999) if r < 0.85 else rng.randint(1000, 9999)
    cents = rng.choice([0, 0, rng.randint(1, 99), rng.randint(1, 99), 50, 99, rng.randint(1, 9)])
    if case == "acc":  # the accusative frame takes whole amounts only
        cents = 0
    a = _group(n) if n >= 1000 else str(n)
    a = f"{a},{cents:02d}" if cents else a
    w = f"${a}" if sym == "$" and rng.random() < 0.5 else f"{a} {sym}"
    read = _amount(major, n, case)
    if cents:
        read += " " + _amount(S.MINOR[LANG][major][0], cents, case)
    return tpl.format(p=w), [read], KINDS["price"]


def _hm(rng):
    h = rng.choice(list(range(7, 23)) + [9, 10, 14, 15, 18, 19])
    m = rng.choice([0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 50, 5, 25, 35, 55, rng.randint(3, 9), rng.randint(13, 59)])
    if m % 10 in (1, 2) and m not in (11, 12):
        m += 2  # "сорок одна": the minute would agree with an unsaid feminine noun
    return h, m


def _clock_nom(rng, h, m):
    """"четырнадцать тридцать", "девять ноль пять", "девять ноль ноль" or "девять часов"."""
    if m == 0:
        return f"{card(h)} {noun('час', h, 'nom')}" if rng.random() < 0.5 else f"{card(h)} ноль ноль"
    if m < 10:
        return f"{card(h)} ноль {card(m)}"
    return f"{card(h)} {card(m, 'nom', 'f')}"


def g_clock(rng, i, ctx):
    r = rng.random()
    if r < 0.6:
        h, m = _hm(rng)
        tpl = rng.choice(["Совещание начнётся в {t}, не опаздывайте.", "Поезд отправляется в {t} с первого пути.",
                          "Давай встретимся в {t} у входа.", "Трансляция начнётся в {t} по московскому времени.",
                          "Мой будильник звонит в {t} каждый день.", "Самолёт приземлился в {t} по местному времени."])
        return tpl.format(t=f"{h}:{m:02d}"), [_clock_nom(rng, h, m)], KINDS["clock"]
    if r < 0.8:
        h, m = _hm(rng)
        tpl = rng.choice(["Сейчас на часах {t}, пора выходить.", "На табло было {t}, когда мы пришли."])
        return tpl.format(t=f"{h}:{m:02d}"), [_clock_nom(rng, h, m)], KINDS["clock"]
    h1, h2 = rng.randint(7, 10), rng.randint(16, 22)
    if h2 == 21:
        h2 = 20  # "до двадцати одного": keep the plain genitive forms
    tpl = rng.choice(["Магазин работает с {a} до {b} без выходных.", "Приём граждан ведётся с {a} до {b}."])
    return tpl.format(a=f"{h1}:00", b=f"{h2}:00"), [card(h1, "gen"), card(h2, "gen")], KINDS["clock2"]


PHONE_FRAMES = ["Звоните по номеру {p} в рабочее время.", "Мой новый номер {p}, запиши, пожалуйста.",
                "Справки можно получить по телефону {p}.", "Оставьте сообщение на номер {p} до вечера.",
                "Горячая линия работает по номеру {p} круглосуточно."]
MOBILE = ["900", "903", "905", "909", "910", "912", "915", "916", "917", "919", "925", "926", "929", "950", "960",
          "977", "985", "999"]
CITY = ["495", "499", "812", "343", "383", "846", "861", "863"]


def _grp(s: str) -> str:
    """A written group said as a number ("45" -> "сорок пять"); a leading zero is said digit by digit ("05")."""
    if s[0] == "0":
        return " ".join(card(int(c)) if c != "0" else "ноль" for c in s)
    return card(int(s))


def g_phone(rng, i, ctx):
    g3 = f"{rng.randint(100, 999)}"
    g2a, g2b = f"{rng.randint(10, 99)}", f"{rng.randint(10, 99)}"
    if rng.random() < 0.7:  # "8-916-123-45-67", one hyphen chain
        code = rng.choice(MOBILE + ["800"])
        parts = ["8", code, g3, g2a, g2b]
        digits = rng.random() < 0.25
        read = " ".join(card(int(c)) if c != "0" else "ноль" for c in "".join(parts)) if digits else " ".join(
            _grp(p) for p in parts)
        return rng.choice(PHONE_FRAMES).format(p="-".join(parts)), [read], KINDS["phone"]
    code = rng.choice(CITY)  # "(495) 123-45-67": the code is its own number
    rest = " ".join(_grp(p) for p in (g3, g2a, g2b))
    return rng.choice(PHONE_FRAMES).format(p=f"({code}) {g3}-{g2a}-{g2b}"), [card(int(code)), rest], KINDS["phone2"]


DATE_FRAMES = ["Договор подписали {d}, и он уже действует.", "Заявление нужно подать до {d} включительно.",
               "Новые правила действуют с {d}.", "Заседание прошло {d} в большом зале.",
               "Окончательное решение приняли {d} после голосования.", "Регистрация продлится до {d}."]
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def g_date(rng, i, ctx):
    mth = rng.randint(1, 12)
    d = rng.randint(1, MONTH_DAYS[mth - 1])
    y = rng.choice(list(range(1995, 2031)) + list(range(2024, 2031)) + [2035, 2040])
    w = f"{d:02d}.{mth:02d}.{y}" if rng.random() < 0.6 else f"{d}.{mth:02d}.{y}"
    read = f"{ordw(d, 'n', 'gen')} {S.MONTHS_GEN[LANG][mth - 1]} {ordw(y, 'm', 'gen')} года"
    return rng.choice(DATE_FRAMES).format(d=w), [read], KINDS["date"]


PCT_FRAMES = [("Инфляция за год составила {x}.", "nom"), ("Цены на продукты выросли на {x} за месяц.", "acc"),
              ("Эту идею поддержали около {x} опрошенных.", "gen"), ("Явка на выборах составила {x}.", "nom"),
              ("Кредит выдают под {x} годовых.", "acc"), ("Скидка на все товары до воскресенья составляет {x}.", "nom")]


def g_pct(rng, i, ctx):
    tpl, case = rng.choice(PCT_FRAMES)
    n = rng.choice([rng.randint(1, 10), rng.randint(1, 10), rng.randint(11, 99), rng.randint(11, 99), 100,
                    rng.choice([21, 22, 23, 31, 32, 41, 42, 51, 52, 3, 4, 2, 1])])
    c = "nom" if case == "acc" else case
    w = f"{n} %" if rng.random() < 0.5 else f"{n}%"
    return tpl.format(x=w), [f"{card(n, c)} {noun('процент', n, c)}"], KINDS["pct"]


DEC_FRAMES = ["Средний балл этого класса составляет {x}.", "Коэффициент за последний квартал равен {x}.",
              "Показатель за месяц составил {x}, это чуть выше ожиданий.", "Рейтинг фильма на сайте составляет {x}.",
              "Индекс на этой неделе составил {x}."]


def _dec_reading(rng, n, frac):
    v = int(frac)
    if rng.random() < 0.3:  # "три запятая пять"
        fw = card(v) if not frac.startswith("0") else " ".join(card(int(c)) if c != "0" else "ноль" for c in frac)
        return f"{card(n)} запятая {fw}"
    cel = "целая" if S.count_class(n) == 0 else "целых"
    den = {1: "десят", 2: "сот"}[len(frac)]
    dw = den + ("ая" if S.count_class(v) == 0 else "ых")
    return f"{card(n, 'nom', 'f')} {cel} {card(v, 'nom', 'f')} {dw}"


def g_dec(rng, i, ctx):
    n = rng.choice([0, rng.randint(1, 9), rng.randint(1, 9), rng.randint(10, 99)])
    frac = rng.choice([str(rng.randint(1, 9)), str(rng.randint(1, 9)), str(rng.randint(11, 99))])
    if frac.endswith("0"):
        frac = frac[:-1] or "5"
    return rng.choice(DEC_FRAMES).format(x=f"{n},{frac}"), [_dec_reading(rng, n, frac)], KINDS["dec"]


ORD_FRAMES = [  # (template with {o}, gender/number, case)
    ("Наша команда заняла {o} место на чемпионате.", "n", "nom"),
    ("Мы живём на {o} этаже нового дома.", "m", "loc"),
    ("Мой сын учится в {o} классе.", "m", "loc"),
    ("Это уже {o} случай за этот год.", "m", "nom"),
    ("Оплату нужно внести до {o} числа каждого месяца.", "n", "gen"),
    ("Откройте учебник на {o} странице.", "f", "loc"),
    ("Он финишировал {o} среди всех участников.", "m", "ins"),
]


def _suffix(gn: str, case: str) -> str:
    """The written ending of an ordinal ("5-й", "5-е", "5-м", "5-го")."""
    if case == "gen" and gn in ("m", "n"):
        return "го"
    if case in ("loc", "ins") and gn in ("m", "n"):
        return "м"
    if (gn, case) == ("n", "nom"):
        return "е"
    return "й"  # m nom, f loc


def g_ord(rng, i, ctx):
    tpl, gn, case = rng.choice(ORD_FRAMES)
    hi = 31 if "числа" in tpl else 11 if "классе" in tpl else 30
    n = rng.randint(1, hi) if rng.random() < 0.85 else rng.choice([40, 50, 100, 25, 33])
    if "классе" in tpl and n > 11:
        n = rng.randint(1, 11)
    return tpl.format(o=f"{n}-{_suffix(gn, case)}"), [ordw(n, gn, case)], KINDS["ord"]


RANGE_UNITS = {"мин": "минута", "ч": "час", "км": "километр", "%": "процент", "кг": "килограмм"}
RANGE_FRAMES = {"мин": ["Дорога займёт {r}, в зависимости от пробок.", "Обычно я жду автобус {r}."],
                "ч": ["Ремонт займёт {r}, не больше.", "Подготовка к экзамену отнимает {r} каждый день."],
                "км": ["Расстояние до лагеря составляет {r}.", "Каждый день он пробегает {r}."],
                "%": ["Ожидаемый рост экономики составит {r} в год.", "Годовая инфляция, по прогнозу, составит {r}."],
                "кг": ["Каждая посылка весит {r}.", "Такой груз обычно весит {r}."]}


def g_range(rng, i, ctx):
    u = rng.choice(list(RANGE_FRAMES))
    lo = rng.choice([rng.randint(1, 10), rng.randint(1, 10), rng.randint(10, 40)])
    hi = lo + rng.choice([1, 2, 3, 5, 5, 10, 10, 20])
    if hi % 10 == 1 and hi % 100 != 11:
        hi += 1  # "до двадцати одного": the gender of a 1 would need the noun's
    if lo % 10 == 1 and lo % 100 != 11:
        lo += 1
    if u == "ч":  # hours of a working day
        lo, hi = rng.randint(1, 5), 0
        hi = lo + rng.choice([1, 2, 3])
    if lo >= hi:
        hi = lo + 2
    read = f"от {card(lo, 'gen')} до {card(hi, 'gen')} {noun(RANGE_UNITS[u], hi, 'gen')}"
    w = f"{lo}–{hi}" + (u if u == "%" else f" {u}")
    return rng.choice(RANGE_FRAMES[u]).format(r=w), [read], KINDS["range"]


GEN = {"year": g_year, "count": g_count, "big": g_big, "price": g_price, "clock": g_clock, "phone": g_phone,
       "date": g_date, "pct": g_pct, "dec": g_dec, "ord": g_ord, "range": g_range}


# ---------------------------------------------------------------- realise: the intended reading must be a candidate

def _strip_parens(label: str, offs: list) -> tuple:
    """Drop "(" and ")" from the label (a phone code is written "(097)", not said), shifting the reading offsets."""
    keep = [i for i, ch in enumerate(label) if ch not in "()"]
    pos = {old: new for new, old in enumerate(keep)}
    out = "".join(label[i] for i in keep)
    new_offs = []
    for a, b in offs:
        na = pos[a]
        nb = pos[b - 1] + 1
        new_offs.append([na, nb])
    return out, new_offs


def realize(caption: str, readings: list, kinds: list):
    try:
        sp = M.spans(caption, LANG)
    except Exception as e:  # noqa: BLE001  (a candidate-machinery error is a drop)
        return None, f"spans_error {type(e).__name__}"
    if [s.kind for s in sp] != kinds:
        return None, "kinds " + ",".join(s.kind for s in sp)
    if not M.all_digits_spanned(caption, LANG):
        return None, "digits_unspanned"
    picks = []
    for s, want in zip(sp, readings):
        cs = [c.strip() for c in s.cands]
        if not cs:
            return None, ("missing_grammar:" + s.note.split(" ", 1)[-1].strip("'\"")) if s.note.startswith("missing") else "no_cands"
        if want not in cs:
            return None, "reading_not_a_candidate"
        picks.append(cs.index(want))
    offs: list = []
    label = C.render_cap(caption, sp, picks, offs)
    label, offs = _strip_parens(label, offs)
    label = re.sub(r"\s+([,.])", r"\1", re.sub(r"\s{2,}", " ", label)).strip()
    if caption.endswith(".") and not label.endswith("."):
        label += "."
    if FORBIDDEN.search(label) or C.WF.search(label):
        return None, "symbol_left"
    if any(not re.search(r"\w", t) for t in label.split()):
        return None, "letterless_token"  # "—": the CTC aligner cannot place a token without letters
    for (a, b), want in zip(offs, readings):
        if label[a:b].lower() != want.lower():
            return None, "offset_mismatch"
    spans = [{"text": s.text, "kind": s.kind, "reading": s.cands[p].strip(), "canonical": s.cands[0].strip(),
              "idx": p, "n_cands": len(s.cands)} for s, p in zip(sp, picks)]
    return (label, spans, offs), ""


def _lead(rng, text: str) -> str:
    if rng.random() < 0.35 or IMPERATIVE.match(text):
        return text
    ld = rng.choice(LEADS_NEUTRAL if PERSONAL.search(text) else LEADS_REPORT + LEADS_NEUTRAL)
    return ld + text[0].lower() + text[1:]


def plan(n: int, seed: int, report: dict | None = None, classes: tuple = (), tag: str = TAG) -> list[dict]:
    """~n rows split over CLASSES by weight (or `classes`, equal shares, for a pilot). Deterministic in (n, seed)."""
    rng = random.Random(f"{tag}|{seed}|{LANG}")
    if classes:
        quota = {c: n // len(classes) + (1 if i < n % len(classes) else 0) for i, c in enumerate(classes)}
    else:
        tot = sum(CLASSES.values())
        quota = {c: max(1, round(n * w / tot)) for c, w in CLASSES.items()}
    ctx = {"years": _years(random.Random(f"{tag}|years|{seed}"), max(quota.get("year", 0), 1))}
    rows, seen = [], set()
    drops = collections.defaultdict(collections.Counter)
    got = collections.Counter()
    missing: set = set()
    for cls, q in quota.items():
        tries, i = 0, 0
        while got[cls] < q and tries < 200 * max(q, 1):
            tries += 1
            try:
                g = GEN[cls](rng, i, ctx)
            except M.Missing as e:  # a value the grammar table lacks (add it with the dumper, then plan again)
                drops[cls]["missing_grammar"] += 1
                missing.add(str(e).strip("'\""))
                continue
            if g is None:
                drops[cls]["generator_skip"] += 1
                continue
            caption, readings, kinds = g
            caption = _lead(rng, caption)
            res, why = realize(caption, readings, kinds)
            if res is None:
                if why.startswith("missing_grammar:"):
                    missing.add(why.split(":", 1)[1])
                    why = "missing_grammar"
                drops[cls][why] += 1
                continue
            label, spans, offs = res
            if not (WORDS_MIN <= len(label.split()) <= WORDS_MAX):
                drops[cls]["length"] += 1
                continue
            if label in seen:
                drops[cls]["duplicate"] += 1
                continue
            i += 1  # the next value of a schedule (years) only once this one is used
            seen.add(label)
            rid = f"{tag}_{LANG}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
            split = "dev" if rng.random() < DEV_FRAC else "train"
            rows.append({"id": rid, "lang": LANG, "cls": cls, "register": "general", "caption": caption, "text": label,
                         "spans": spans, "num_chars": offs, "kind": sorted({s["kind"] for s in spans}), "split": split})
            got[cls] += 1
    if report is not None:
        report.update(quota=quota, got=dict(got), drops={c: dict(v) for c, v in drops.items()}, missing_keys=sorted(missing))
    return rows


def needed(n: int, seed: int) -> list[str]:
    """Grammar-table keys the plan's captions need that the table lacks (generators run without the candidate check)."""
    rng = random.Random(f"needed|{seed}")
    tot = sum(CLASSES.values())
    ctx = {"years": _years(rng, 200)}
    texts = []
    for cls, w in CLASSES.items():
        for i in range(max(1, round(n * w / tot)) * 3):
            try:
                g = GEN[cls](rng, i, ctx)
            except M.Missing:
                g = None
            if g:
                texts.append(g[0])
    tab = M.table()
    return sorted(k for k in M.needed(texts, LANG) if k not in tab)


def missing_for_readings(n: int, seed: int) -> set:
    """Keys the READING builders (ml_slav.card_nom / ord_nom) miss: run every generator and record M.Missing."""
    rng = random.Random(f"needed|{seed}")
    out = set()
    ctx = {"years": _years(rng, 200)}
    for cls, w in CLASSES.items():
        for i in range(n):
            try:
                GEN[cls](rng, i, ctx)
            except M.Missing as e:
                out.add(str(e).strip("'\""))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        rep: dict = {}
        cl = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ()
        for r in plan(int(sys.argv[2]), int(sys.argv[3]), rep, cl):
            print(f"{r['cls']:6s} {r['split']:5s} W: {r['caption']}\n{'':12s} L: {r['text']}")
        print(json.dumps(rep, ensure_ascii=False), file=sys.stderr)
    elif cmd == "needed":
        ks = sorted(set(needed(int(sys.argv[2]), int(sys.argv[3]))) | missing_for_readings(int(sys.argv[2]), int(sys.argv[3])))
        json.dump(ks, open(sys.argv[4], "w"))
        print(len(ks), "missing keys ->", sys.argv[4])
    else:
        print(__doc__)
