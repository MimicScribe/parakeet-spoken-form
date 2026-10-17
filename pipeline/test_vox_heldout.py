"""Decision-set parliament pull (no Modal): python3 test_vox_heldout.py"""
import collections

import vox_heldout as V
from ml_p3 import _gv_classes

c = lambda t, l="es": V.classes(t, l, _gv_classes)  # noqa: E731
assert c("el 15 de marzo de 2019")["date"] and c("el 15 de marzo de 2019")["year"]
assert c("a las 10 horas")["clock"] and not c("durante 10 horas")["clock"]
assert c("un 25 % más")["pct"] and c("son 30 euros")["cur"] and c("en los años 90")["decade"]
assert c("unos 40 km")["unit"] and c("el 3º punto")["ord"] and not c("unos 40 kilómetros")["unit"]
assert not c("de 10 a 12")["ord"]
assert c("le 3 mars", "fr")["date"] and c("à 15h30", "fr")["clock"] and c("le 2e rapport", "fr")["ord"]
assert c("on 22nd November", "en")["date"] and c("May 30th", "en")["date"] and c("the 1990s", "en")["decade"]
assert not c("in 2019 we", "en")["phone"]
assert V.day_of("20190115-0900-PLENARY-3_es_12") == "20190115" and V.day_of("es000_1_abc") is None

# select: never a used day, only the chosen debate years, caps per session
rows = []
for d in range(1, 29):
    for s in range(12):
        day = f"2019{1 + d % 12:02d}{d:02d}"
        rows.append({"utt_id": f"{day}-0900-PLENARY-{s}_es_1", "day": day,
                     "text": f"El {d} de marzo de 20{10 + s} se aprobaron enmiendas sobre el plan {"abcdefghijklmnopqrstuvwxyz"[d % 26]}{"abcdefghijkl"[s]} anual"})
for d in range(1, 10):
    day = f"2018{d:02d}{d:02d}"
    rows.append({"utt_id": f"{day}-0900-PLENARY-1_es_1", "day": day, "text": "un 25 % del total en 2017 y algo más"})
used = {"20190202", "20190303"}
sel, st = V.select(rows, used, lambda t: c(t), per_class=20, n_year=40, n_debate_years=1)
assert sel and not ({r["day"] for r in sel} & used)
assert st["debate_years"] == ["2019"] and all(r["day"].startswith("2019") for r in sel)
assert max(collections.Counter(r["utt_id"].rsplit("_", 1)[0] for r in sel).values()) <= 8
assert st["have"]["date"] >= 20 and st["have"]["year"] >= 40
lines = V.held_lines(["20190101-0900-PLENARY-1_es_1", "20190101-0900-PLENARY-2_es_1", "20190105-0900-PLENARY-1_es_1"])
assert lines.splitlines() == ["# group vox:20190101", "20190101-0900-PLENARY-1_es_1", "20190101-0900-PLENARY-2_es_1",
                              "# group vox:20190105", "20190105-0900-PLENARY-1_es_1"], lines
print("ok")

# long numbers (de / it / nl / pt): months, ordinals, decades, int_big, phone
ib = lambda t, l: c(t, l)["int_big"]  # noqa: E731
assert c("am 15. März 2019", "de")["date"] and c("il 3 gennaio", "it")["date"] and c("op 12 maart", "nl")["date"]
assert c("em 3 de março", "pt")["date"] and c("em 1.º de maio", "pt")["ord"] and c("in den 90er Jahren", "de")["decade"]
assert c("nei anni 90", "it")["decade"] and c("in de jaren 90", "nl")["decade"] and c("nos anos 80", "pt")["decade"]
assert c("der 3. Tagesordnungspunkt", "de")["ord"] and not c("Es waren 3. Dann kam", "de")["ord"]
assert c("la 2ª fase", "it")["ord"] and c("de 2e lezing", "nl")["ord"]
# int_big positives: grouped, plain >= 4 digits not a year, digits + scale word
assert ib("rund 3.500 Euro", "de") and ib("12 Millionen Euro", "de") and ib("2,5 Milliarden", "de")
assert ib("3 500 000 persone", "it") and ib("12 milioni", "it") and ib("5 miliardi", "it") and ib("30 mila posti", "it")
assert ib("12 miljoen", "nl") and ib("3 miljard", "nl") and ib("1.250.000 mensen", "nl") and ib("2500 mensen", "nl")
assert ib("12 milhões", "pt") and ib("2 mil milhões", "pt") and ib("15 mil euros", "pt") and ib("cerca de 4500", "pt")
assert ib("12 millones", "es") and ib("3 millions", "fr") and ib("2 million", "en")
# int_big negatives: years (bare, in a date, in a list), small numbers, decimals, dates, phone, year-like words
assert not ib("im Jahr 2019", "de") and not ib("am 15. März 2019", "de") and not ib("von 1990 bis 2030", "de")
assert not ib("in 1850 en 2099", "nl") and not ib("em 2024", "pt") and not ib("nel 2020", "it")
assert not ib("tra 12 e 15 persone", "it") and not ib("3,500 Euro", "de") and not ib("12.05.2019", "de")
assert not ib("rund 500 Euro", "de") and not ib("Millionen Menschen", "de") and not ib("het milieu van 12 mensen", "nl")
assert not ib("0151 123 4567", "de")  # a phone number is a phone, not an integer
assert ib("im Jahr 2019 waren es 3.500", "de")  # a year beside a real big number still counts
# phone: a real number, never a year list / thousands group / date
assert c("rufen Sie 0800 123 4567 an", "de")["phone"] and c("ligue para 21 123 45 67", "pt")["phone"]
assert not c("2019 2020 2021", "de")["phone"] and not c("1.250.000 mensen", "nl")["phone"]
assert not c("am 12.05.2019", "de")["phone"] and not c("3 500 000", "it")["phone"] and not c("in 12 34 56", "de")["phone"]
# selection by chosen classes only
rows2 = []
for d in range(1, 40):
    day = f"2019{1 + d % 12:02d}{d % 28 + 1:02d}"
    for s in range(6):
        t = (f"Wir haben {10 + s} Millionen Euro im Fonds Nummer {'abcdefgh'[s]}{d}" if s % 2 else
             f"Im Jahr 20{10 + s} wurde der Plan {'abcdefgh'[s]}{d} beschlossen und 15% mehr")
        rows2.append({"utt_id": f"{day}-0900-PLENARY-{s}_de_{d}", "day": day, "text": t})
sel2, st2 = V.select(rows2, set(), lambda t: c(t, "de"), per_class=6, n_debate_years=1, only=("phone", "int_big"))
assert sel2 and all(c(r["text"], "de")["int_big"] for r in sel2) and st2["have"]["int_big"] >= 6 and "year" not in st2["want"]
assert not any(c(r["text"], "de")["pct"] and not c(r["text"], "de")["int_big"] for r in sel2)
# int_big_digit: digits only, a scale word alone does not count
d = lambda t, l: c(t, l)["int_big_digit"]  # noqa: E731
assert d("rund 3.500 Euro", "de") and d("cerca de 4500", "pt") and d("1.250.000 mensen", "nl")
assert not d("12 Millionen Euro", "de") and not d("15 mil euros", "pt") and not d("im Jahr 2019", "de")
rows3 = [dict(r, text=r["text"].replace("Millionen", "Millionen und 3.500")) if i % 2 else r for i, r in enumerate(rows2)]
sel3, st3 = V.select(rows3, set(), lambda t: c(t, "de"), per_class=6, n_debate_years=1, only=("int_big_digit",))
assert sel3 and all(c(r["text"], "de")["int_big_digit"] for r in sel3)
print("ok long")
