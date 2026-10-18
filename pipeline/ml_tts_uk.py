"""Ukrainian synthetic sentences for spoken-form number coverage (2026-10-17).

Real Ukrainian speech has almost no years, prices, clock times, phone numbers or dates said aloud with a word-form
transcript, so this small set gives every number group SOME coverage. ml_tts.plan has no Ukrainian templates; here each
class has carrier sentences whose case is fixed by the frame ("у 2025 році" locative, "з 1998 року" genitive, "о 14:30"
locative feminine hour, "коштує 240,50 грн" nominative), and the spoken reading is built with the ml_slav inflection
tables (case, gender and the 1 / 2-4 / 5+ noun rule) and the grammar table.

A row is kept only when ml_cands.spans(caption, "uk") finds the expected spans and the intended reading is one of each
span's candidates: the label is then rendered exactly like a real row (ml_p3_core.render_cap), so its spans match
what the labelling pipeline would say about the caption.

    plan(n, seed, report=None, classes=(), tag="tts6uk")  pure Python: rows for ml_tts_cloud (no voice fields yet)
    python3 ml_tts_uk.py plan 60 1                          print sample rows
    python3 ml_tts_uk.py needed 2000 1 out.json             grammar-table keys the plan needs that the table lacks
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

LANG = "uk"
TAG = "tts6uk"
# class -> relative quota. year / count / big all carry span kind "int" and share one synthetic cap in the mix, so
# small counts (a real-speech source covers 0-100) get the smallest share
CLASSES = {"year": 80, "count": 25, "big": 40, "price": 70, "clock": 70, "phone": 70, "date": 70, "pct": 70,
           "dec": 70, "ord": 70, "range": 70}
# the span kinds ml_slav gives each class (a row whose spans differ is dropped)
KINDS = {"year": ["int"], "count": ["int"], "big": ["int"], "price": ["cur"], "clock": ["clock"],
         "clock2": ["clock", "clock"], "phone": ["code", "chain"], "date": ["date"], "pct": ["pct"], "dec": ["dec"],
         "ord": ["ord"], "range": ["range"]}
WORDS_MIN, WORDS_MAX = 6, 22
DEV_FRAC = 0.1
FORBIDDEN = re.compile(r"[0-9$€£¥%#/+°²³½¼¾ºª‰&₴()]")
# sentence openers: reported-speech leads only before third-person statements, none before an imperative
LEADS_REPORT = ["За нашими даними, ", "Як повідомили, ", "Нагадаю, що ", "Наскільки я знаю, ", "Кажуть, що "]
LEADS_NEUTRAL = ["До речі, ", "Отже, ", "Чесно кажучи, "]
PERSONAL = re.compile(r"(?<![" + S.L + r"])(?:я|ми|мій|моя|мене|наш\w*|давай|запиши)(?![" + S.L + r"])", re.I)
IMPERATIVE = re.compile(r"^(?:Телефонуйте|Залиште|Відкрийте|Давай)")


# ---------------------------------------------------------------- readings (ml_slav tables)

def card(n: int, case: str = "nom", g: str = "m", drop_one: bool = True) -> str:
    ws = S.card_nom(LANG, n)
    if drop_one and len(ws) > 1 and ws[0] in ("одна", "один") and ws[1] in ("тисяча",):
        ws = ws[1:]  # "тисяча п'ятсот", the everyday form
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
        # about 2 of 3 from 1990-2030, 1 of 3 forward
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

Y_FRAMES = [  # (template, case, era)  era: past / future / any
    ("Цей закон ухвалили у {y} році після довгих дискусій.", "loc", "past"),
    ("Компанія відкрила свій перший офіс у {y} році.", "loc", "past"),
    ("У {y} році ми плануємо завершити будівництво мосту.", "loc", "future"),
    ("Нову лінію метро обіцяють запустити у {y} році.", "loc", "future"),
    ("Він працює на цьому заводі з {y} року.", "gen", "past"),
    ("Програма діятиме до {y} року включно.", "gen", "future"),
    ("Після {y} року ціни на житло суттєво змінилися.", "gen", "past"),
    ("Ці дані зібрані станом на кінець {y} року.", "gen", "past"),
    ("Стратегію розвитку міста розписали до {y} року.", "gen", "future"),
    ("{y} рік став дуже важким для всієї галузі.", "nom", "past"),
    ("Ми часто згадуємо події {y} року.", "gen", "past"),
    ("Мету скоротити викиди поставили на {y} рік.", "acc", "future"),
]


def g_year(rng, i, ctx):
    y = ctx["years"][i % len(ctx["years"])]
    era = "past" if y <= 2025 else "future"
    fr = [f for f in Y_FRAMES if f[2] in (era, "any")]
    tpl, case, _ = rng.choice(fr)
    acc_case = "nom" if case == "acc" else case  # inanimate accusative = nominative ("на дві тисячі тридцятий рік")
    return tpl.format(y=y), [ordw(y, "m", acc_case)], KINDS["year"]


YEARS3_FRAMES = [  # (template, case of each year) -- tense-neutral, years mid / late
    ("У документі згадуються {ys} роки, і це важливо.", "nom"),
    ("Таблиця охоплює {ys} роки по всіх регіонах.", "nom"),
    ("Експерти порівняли показники {ys} років між собою.", "gen"),
    ("У звіті зібрані дані за {ys} роки.", "nom"),
    ("Комісія окремо розглянула підсумки {ys} років.", "gen"),
    ("Особливу увагу в доповіді приділено {ys} рокам.", "dat"),
]
YEARS_POOL: list = []  # year values g_years3 draws (pop order); empty = ctx["years"]


def g_years3(rng, i, ctx):
    """2-3 years in one sentence (cheaper per year span), each an ordinal in the frame's case, at normal rates."""
    k = rng.choice([2, 3, 3])
    ys, j = [], 0
    while len(ys) < k and j < 50:
        if "years3" not in ctx:  # a plan without a year quota has a one-value year list
            ctx["years3"] = _years(random.Random(f"years3|{i}"), 300)
        y = YEARS_POOL.pop() if YEARS_POOL else ctx["years3"][(3 * i + j) % len(ctx["years3"])]
        j += 1
        if y not in ys:
            ys.append(y)
    tpl, case = rng.choice(YEARS3_FRAMES)
    written = ", ".join(map(str, ys[:-1])) + " та " + str(ys[-1])
    return tpl.format(ys=written), [ordw(y, "m", case) for y in ys], ["int"] * k


COUNT_FRAMES = [  # (template with {n} {w}, case, nouns)
    ("На дорогу до роботи я витрачаю {n} {w}.", "acc", ["хвилина", "година"]),
    ("Нарада тривала близько {n} {w}, і всі втомилися.", "gen", ["хвилина", "година"]),
    ("Ми доїхали до моря за {n} {w}.", "acc", ["година", "хвилина"]),
    ("До найближчої зупинки лише {n} {w}.", "nom", ["кілометр", "метр"]),
    ("Для цього рецепта потрібно {n} {w} борошна.", "nom", ["кілограм", "грам"]),
    ("За тиждень ми привезли {n} {w} води.", "acc", ["літр", "тонна"]),
    ("Пробіжка займає в мене приблизно {n} {w}.", "acc", ["хвилина"]),
]


def g_count(rng, i, ctx):
    tpl, case, nouns = rng.choice(COUNT_FRAMES)
    lem = rng.choice(nouns)
    n = _small(rng)
    g = gender(lem)
    if case == "acc":
        r = card(n, "acc", g)
    else:
        r = card(n, case, g)
    return tpl.format(n=n, w=noun(lem, n, case)), [r], KINDS["count"]


BIG_FRAMES = [  # (template, case, largest sensible value)
    ("Бюджет проєкту становить {n} гривень.", "nom", 10**11),
    ("Держава виділила на ремонт доріг {n} гривень.", "acc", 10**11),
    ("На рахунку компанії залишилося близько {n} доларів.", "gen", 10**11),
    ("У нашій області живе приблизно {n} людей.", "nom", 5 * 10**6),
    ("Минулого року місто відвідали {n} туристів.", "nom", 2 * 10**7),
    ("Збитки від повені оцінюють у {n} гривень.", "acc", 10**11),
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
    # accusative: "мільйон" / "дві тисячі" keep the nominative form, a "тисяча" after one becomes "тисячу"
    if rng.random() < 0.3 and n >= 10**6 and n % 10**6 == 0 and n < 10**9:
        w = f"{n // 10**6} млн"
    elif rng.random() < 0.3 and n >= 10**9 and n % 10**9 == 0:
        w = f"{n // 10**9} млрд"
    else:
        w = _group(n)
    return tpl.format(n=w), [card(n, case)], KINDS["big"]


PRICE_FRAMES = [  # (template, case); only the "paid" frame (WHOLE_ONLY) takes whole amounts only
    ("Квиток на концерт коштує {p}, і це недорого.", "acc"),  # the verb takes the accusative: "коштує одну гривню"
    ("Ціна однієї порції без знижки становить {p}.", "nom"),
    ("За комунальні послуги ми заплатили {p} цього місяця.", "acc"),
    ("Вартість доставки становить {p} по всій країні.", "nom"),
    ("Цей товар у магазині біля дому коштує {p}.", "acc"),
    ("Разом до сплати {p}, оплата карткою або готівкою.", "nom"),
]
WHOLE_ONLY = {t for t, c in PRICE_FRAMES if "заплатили" in t}
CUR_W = [("грн", "гривня", 6), ("$", "долар", 2), ("€", "євро", 2)]


def _amount(lemma, n, case):
    return f"{card(n, case, gender(lemma))} {noun(lemma, n, case)}"


def g_price(rng, i, ctx):
    tpl, case = rng.choice(PRICE_FRAMES)
    sym, major, _ = rng.choices(CUR_W, [w for *_, w in CUR_W])[0]
    r = rng.random()
    n = rng.randint(1, 20) if r < 0.35 else rng.randint(21, 999) if r < 0.85 else rng.randint(1000, 9999)
    cents = rng.choice([0, 0, rng.randint(1, 99), rng.randint(1, 99), 50, 99, rng.randint(1, 9)])
    if tpl in WHOLE_ONLY:  # the "paid" frame takes whole amounts only ("заплатили одну гривню")
        cents = 0
    a = _group(n) if n >= 1000 else str(n)
    a = f"{a},{cents:02d}" if cents else a
    if sym == "грн":
        w = f"{a} грн"
    elif sym == "$":
        w = f"${a}" if rng.random() < 0.5 else f"{a} $"
    else:
        w = f"{a} €"
    read = _amount(major, n, case)
    if cents:
        minor = S.MINOR[LANG][major][0]
        read += " " + _amount(minor, cents, case)
    return tpl.format(p=w), [read], KINDS["price"]


def _hm(rng):
    h = rng.choice(list(range(7, 23)) + [9, 10, 14, 15, 18, 19])
    m = rng.choice([0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 50, 5, 25, 35, 55, rng.randint(3, 9), rng.randint(13, 59)])
    if m % 10 in (1, 2) and m not in (11, 12):
        m += 2  # "двадцять одна хвилина": the minute count would agree with an unsaid feminine noun
    return h, m


def _mins(rng, m):
    if m == 0:
        return ""
    if m < 10:
        return " " + card(0) + " " + card(m)  # "нуль п'ять"
    return " " + card(m)


def g_clock(rng, i, ctx):
    r = rng.random()
    if r < 0.45:
        h, m = _hm(rng)
        pre = "об" if h == 11 else "о"
        tpl = rng.choice(["Нарада почнеться {pre} {t}, не запізнюйтеся.", "Потяг відправляється {pre} {t} з першої колії.",
                          "Давай зустрінемося {pre} {t} біля входу.", "Трансляція стартує {pre} {t} за київським часом."])
        return (tpl.format(pre=pre, t=f"{h}:{m:02d}"), [ordw(h, "f", "loc") + _mins(rng, m)], KINDS["clock"])
    if r < 0.6:
        h = rng.randint(5, 11)
        day, hh = rng.choice([("ранку", h), ("вечора", rng.randint(6, 11))])
        pre = "об" if hh == 11 else "о"
        tpl = rng.choice(["Я прокидаюся {pre} {t} {d} щодня.", "Вони повернулися додому {pre} {t} {d}."])
        return tpl.format(pre=pre, t=f"{hh}:00", d=day), [ordw(hh, "f", "loc")], KINDS["clock"]
    if r < 0.8:
        h, m = _hm(rng)
        tpl = rng.choice(["Зараз на годиннику {t}, пора виходити.", "На табло було {t}, коли ми прийшли."])
        return tpl.format(t=f"{h}:{m:02d}"), [ordw(h, "f", "nom") + _mins(rng, m)], KINDS["clock"]
    h1, h2 = rng.randint(7, 10), rng.randint(16, 22)
    tpl = rng.choice(["Магазин працює з {a} до {b} без вихідних.", "Прийом громадян триває з {a} до {b}."])
    return (tpl.format(a=f"{h1}:00", b=f"{h2}:00"), [ordw(h1, "f", "gen"), ordw(h2, "f", "gen")], KINDS["clock2"])


PHONE_FRAMES = ["Телефонуйте за номером {p} у робочий час.", "Мій новий номер {p}, запиши, будь ласка.",
                "Довідки можна отримати за телефоном {p}.", "Залиште повідомлення на номер {p} до вечора.",
                "Гаряча лінія працює за номером {p} цілодобово."]
CODES = ["050", "063", "066", "067", "068", "073", "093", "095", "096", "097", "098", "099", "044", "032", "048", "056"]


def _chunk(s: str) -> str:
    if s[0] == "0":
        return " ".join(card(int(c)) for c in s) if len(s) != 2 or s[1] == "0" else f"{card(0)} {card(int(s[1]))}"
    return card(int(s))


def g_phone(rng, i, ctx):
    code = rng.choice(CODES)
    g3 = f"{rng.randint(100, 999)}"
    g2a, g2b = f"{rng.randint(0, 99):02d}", f"{rng.randint(0, 99):02d}"
    digits = rng.random() < 0.3
    if digits:
        r_code = " ".join(card(int(c)) for c in code)
        r_rest = " ".join(card(int(c)) for c in g3 + g2a + g2b)
    else:
        r_code = f"{card(0)} {card(int(code[1:]))}"
        r_rest = " ".join(_chunk(x) for x in (g3, g2a, g2b))
    tpl = rng.choice(PHONE_FRAMES)
    return tpl.format(p=f"({code}) {g3}-{g2a}-{g2b}"), [r_code, r_rest], KINDS["phone"]


DATE_FRAMES = ["Договір підписали {d}, і він уже діє.", "Заява має надійти до {d} включно.",
               "Нові правила діють з {d}.", "Засідання відбулося {d} у великій залі.",
               "Остаточне рішення ухвалили {d} після голосування.", "Реєстрація триває до {d}."]
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def g_date(rng, i, ctx):
    mth = rng.randint(1, 12)
    d = rng.randint(1, MONTH_DAYS[mth - 1])
    y = rng.choice(list(range(1995, 2031)) + list(range(2024, 2031)) + [2035, 2040])
    w = f"{d:02d}.{mth:02d}.{y}" if rng.random() < 0.6 else f"{d}.{mth:02d}.{y}"
    read = f"{ordw(d, 'n', 'gen')} {S.MONTHS_GEN[LANG][mth - 1]} {ordw(y, 'm', 'gen')} року"
    return rng.choice(DATE_FRAMES).format(d=w), [read], KINDS["date"]


PCT_FRAMES = [("Інфляція за рік становила {x}.", "nom"), ("Ціни на продукти зросли на {x} за місяць.", "acc"),
              ("Цю ідею підтримали близько {x} опитаних.", "gen"), ("Явка на виборах склала {x}.", "nom"),
              ("Кредит видають під {x} річних.", "acc"), ("Знижка на всі товари до неділі становить {x}.", "nom")]


def g_pct(rng, i, ctx):
    tpl, case = rng.choice(PCT_FRAMES)
    n = rng.choice([rng.randint(1, 10), rng.randint(1, 10), rng.randint(11, 99), rng.randint(11, 99), 100,
                    rng.choice([21, 22, 23, 31, 32, 41, 42, 51, 52, 3, 4, 2, 1])])
    c = "nom" if case == "acc" else case
    w = f"{n} %" if rng.random() < 0.5 else f"{n}%"
    return tpl.format(x=w), [f"{card(n, c)} {noun('відсоток', n, c)}"], KINDS["pct"]


DEC_FRAMES = ["Середній бал цього класу становить {x}.", "Коефіцієнт за останній квартал дорівнює {x}.",
              "Показник за місяць склав {x}, що трохи вище очікувань.", "Рейтинг фільму на сайті становить {x}.",
              "Індекс цього тижня зупинився на позначці {x}."]


def _dec_reading(rng, n, frac):
    v = int(frac)
    if rng.random() < 0.4:  # "три кома п'ять"
        fw = card(v) if len(frac) == 1 or not frac.startswith("0") else " ".join(card(int(c)) for c in frac)
        return f"{card(n)} кома {fw}"
    cel = "ціла" if S.count_class(n) == 0 else "цілих"
    den = {1: "десят", 2: "сот"}[len(frac)]
    dw = den + ("а" if S.count_class(v) == 0 else "их")
    return f"{card(n, 'nom', 'f')} {cel} {card(v, 'nom', 'f')} {dw}"


def g_dec(rng, i, ctx):
    n = rng.choice([0, rng.randint(1, 9), rng.randint(1, 9), rng.randint(10, 99)])
    frac = rng.choice([str(rng.randint(1, 9)), str(rng.randint(1, 9)), str(rng.randint(11, 99))])
    if frac.endswith("0"):
        frac = frac[:-1] or "5"
    tpl = rng.choice(DEC_FRAMES)
    if "позначці" in tpl:  # "на позначці" + locative: keep to the nominative frames
        tpl = DEC_FRAMES[0]
    return tpl.format(x=f"{n},{frac}"), [_dec_reading(rng, n, frac)], KINDS["dec"]


ORD_FRAMES = [  # (template with {o}, gender/number, case)
    ("Наша команда посіла {o} місце на чемпіонаті.", "n", "nom"),
    ("Ми живемо на {o} поверсі нового будинку.", "m", "loc"),
    ("Мій син навчається у {o} класі.", "m", "loc"),
    ("Це вже {o} випадок за цей рік.", "m", "nom"),
    ("Оплату треба внести до {o} числа кожного місяця.", "n", "gen"),
    ("Відкрийте підручник на {o} сторінці.", "f", "loc"),
    ("Він фінішував {o} серед усіх учасників.", "m", "ins"),
]


def _suffix(word: str, gn: str, case: str) -> str:
    """The written ending of an ordinal ("3-тє", "5-те", "2-ге", "12-му", "7-й", "15-го", "4-ій")."""
    if (gn, case) == ("m", "nom"):
        return "й"
    if case in ("loc", "dat") and gn in ("m", "n"):
        return "му"
    if case == "gen" and gn in ("m", "n"):
        return "го"
    if case == "ins" and gn in ("m", "n"):
        return "м"
    if (gn, case) == ("f", "loc"):
        return "й"
    return word[-2:]


def g_ord(rng, i, ctx):
    tpl, gn, case = rng.choice(ORD_FRAMES)
    hi = 31 if "числа" in tpl else 11 if "класі" in tpl else 30
    n = rng.randint(1, hi) if rng.random() < 0.85 else rng.choice([40, 50, 100, 25, 33])
    if "класі" in tpl and n > 11:
        n = rng.randint(1, 11)
    read = ordw(n, gn, case)
    return tpl.format(o=f"{n}-{_suffix(read.split()[-1], gn, case)}"), [read], KINDS["ord"]


RANGE_UNITS = {"хв": "хвилина", "год": "година", "км": "кілометр", "%": "відсоток", "кг": "кілограм"}
RANGE_FRAMES = {"хв": ["Дорога займе {r}, залежно від заторів.", "Зазвичай я чекаю автобус {r}."],
                "год": ["Ремонт займе {r}, не більше.", "Підготовка до іспиту забирає {r} щодня."],
                "км": ["Відстань до табору становить {r}.", "Щодня він пробігає {r}."],
                "%": ["Очікуване зростання економіки становить {r} на рік.", "Річна інфляція, за прогнозом, становитиме {r}."],
                "кг": ["Кожна посилка важить {r}.", "Такий вантаж зазвичай важить {r}."],
                "": ["Чекати на відповідь доведеться {r} днів.", "У групі зазвичай {r} учнів."]}


def g_range(rng, i, ctx):
    u = rng.choice(list(RANGE_FRAMES))
    lo = rng.choice([rng.randint(1, 10), rng.randint(1, 10), rng.randint(10, 40)])
    hi = lo + rng.choice([1, 2, 3, 5, 5, 10, 10, 20])
    if hi % 10 == 1 and hi % 100 != 11:
        hi += 1  # "до двадцяти одного": the gender of a 1 after a range would need the noun's
    if lo % 10 == 1 and lo % 100 != 11:
        lo += 1
    if lo >= hi:
        hi = lo + 2
    read = f"від {card(lo, 'gen')} до {card(hi, 'gen')}"
    if u:
        read += " " + noun(RANGE_UNITS[u], hi, "gen")
    w = f"{lo}–{hi}" + (f" {u}" if u else "")
    return rng.choice(RANGE_FRAMES[u]).format(r=w), [read], KINDS["range"]


# classes whose span kind plan relabels after the candidate check (written-shape kind -> census kind)
RELABEL_KIND = {"phone_sp": "phone"}
GEN = {"phone_sp": lambda rng, i, ctx: g_phone(rng, i, ctx), "year": g_year, "count": g_count, "big": g_big, "price": g_price, "clock": g_clock, "phone": g_phone, "years3": g_years3,
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


# ---------------------------------------------------------------- dev split by carrier template (like gen.is_dev_template)

_SLOT = re.compile(r"[($]?\d(?:[\d\s.,:–()-]*[\d)])?(?:\s?(?:%|грн|\$|€|млн|млрд|хв|год|км|кг)(?![" + S.L + r"]))?"
                   r"(?:-[" + S.L + r"]{1,3})?")


def template_key(caption: str) -> str:
    """The carrier sentence of a caption: the opening lead removed, each written number (with its unit / currency /
    ordinal ending and the word right after it, which agrees with the value) replaced by "#"."""
    t = caption
    for ld in LEADS_REPORT + LEADS_NEUTRAL:
        if t.startswith(ld):
            t = t[len(ld)].upper() + t[len(ld) + 1:]
            break
    t = re.sub(r"#(?:\s+#)+", "#", _SLOT.sub("#", t))
    t = re.sub(r"#\s+[" + S.L + r"]+", "#", t)  # the counted noun / unit word agrees with the value: part of the slot
    return re.sub(r"\bоб #", "о #", t).strip()


def dev_templates(rows: list, dev_frac: float = DEV_FRAC) -> set:
    """Held-out carrier templates: a hash split (zlib.crc32, as gen.is_dev_template), adjusted so each class with at
    least two templates has one in train and one in dev."""
    import zlib

    by_cls: dict = {}
    for r in rows:
        by_cls.setdefault(r["cls"], set()).add(template_key(r["caption"]))
    crc = lambda t: zlib.crc32(t.encode())  # noqa: E731
    dev = {t for ts in by_cls.values() for t in ts if (crc(t) % 1000) / 1000 < dev_frac}
    for c, ts in sorted(by_cls.items()):
        ts = sorted(ts, key=crc)
        if len(ts) < 2:
            dev -= set(ts)
            continue
        if all(t in dev for t in ts):
            dev.discard(ts[0])
        if not any(t in dev for t in ts):
            dev.add(ts[0])
    return dev


def resplit(rows: list, dev_voices: set) -> list:
    """Re-split rows already synthesised: dev = a held-out template; train = a train template read by a train voice;
    a train-template row read by a held-out voice is set "unused" (it would put that voice in training)."""
    dev = dev_templates(rows)
    out = []
    for r in rows:
        t = template_key(r["caption"])
        sp = "dev" if t in dev else ("unused" if r.get("voice") in dev_voices else "train")
        out.append(dict(r, template=t, split=sp))
    return out


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
            if cls in RELABEL_KIND:
                spans = [dict(sp, kind=RELABEL_KIND[cls]) for sp in spans]
            if not (WORDS_MIN <= len(label.split()) <= WORDS_MAX):
                drops[cls]["length"] += 1
                continue
            if label in seen:
                drops[cls]["duplicate"] += 1
                continue
            i += 1  # the next value of a schedule (years) only once this one is used
            seen.add(label)
            rid = f"{tag}_{LANG}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
            rng.random()  # (was the per-row split draw; kept so the plan's values and ids stay as they were)
            rows.append({"id": rid, "lang": LANG, "cls": cls, "register": "general", "caption": caption, "text": label,
                         "spans": spans, "num_chars": offs, "kind": sorted({s["kind"] for s in spans}), "split": "train"})
            got[cls] += 1
    dev = dev_templates(rows)
    for r in rows:
        r["template"] = template_key(r["caption"])
        r["split"] = "dev" if r["template"] in dev else "train"
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
    elif cmd == "resplit":  # resplit <rows.jsonl in> <rows.jsonl out>: dev voices = the voices of the old dev rows
        rs = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
        dv = {r["voice"] for r in rs if r.get("split") == "dev"}
        new = resplit(rs, dv)
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"dev_voices": sorted(dv), "split": collections.Counter(r["split"] for r in new),
                          "templates_dev": sorted({r["template"] for r in new if r["split"] == "dev"})}, ensure_ascii=False))
    elif cmd == "needed":
        ks = sorted(set(needed(int(sys.argv[2]), int(sys.argv[3]))) | missing_for_readings(int(sys.argv[2]), int(sys.argv[3])))
        json.dump(ks, open(sys.argv[4], "w"))
        print(len(ks), "missing keys ->", sys.argv[4])
    else:
        print(__doc__)
