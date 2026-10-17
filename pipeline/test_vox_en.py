"""Local tests for the English parliament selection (vox_en.py) and the English additions to ml_p3._gv_classes.
Needs the `modal` package (ml_p3 is a Modal app module): run from pipeline/ with a Python that has it,
    python test_vox_en.py
"""
import re

import ml_p3 as P
import vox_en as V

# ---- _gv_classes: every non-English language keeps exactly the pre-English behaviour (frozen copy below) ----
_TW = {"de": r"uhr", "fr": r"h|heures?", "it": r"ore|alle", "es": r"horas?|a las", "pt": r"horas?|às", "nl": r"uur"}
_CUR = r"€|\$|£|euros?|eur\b|dollars?|dólares|dolares|pfund|sterline|livres?|libras?|franken|francs?|kronen"
_SCALE = (r"milli(?:on|ard)(?:en|s|i|e)?|milion(?:i|e)|miliard(?:i|o)|millones|millón|milhões|milhão|mil milhões|"
          r"miljoen|miljard|tausend|mille|mila")


def _old(text, lang):
    t = text.lower()
    years = [int(m.group(1)) for m in re.finditer(r"(?<![\d.,])(1[89]\d\d|20\d\d)(?![\d%]|[.,]\d)", text)]
    clock = bool(re.search(r"(?<![\d.,])([01]?\d|2[0-3])[.:h]([0-5]\d)(?![\d.,])\s*(?:" + _TW.get(lang, "h") + r")\b", t)
                 or re.search(r"(?:" + _TW.get(lang, "h") + r")\s+([01]?\d|2[0-3])[.:]([0-5]\d)(?![\d.,])", t)
                 or re.search(r"(?<![\d.,])([01]?\d|2[0-3])\s*(?:" + _TW.get(lang, "h") + r")\b", t)
                 or re.search(r"(?<![\d.,])([01]?\d|2[0-3])h(?:[0-5]\d)?\b", t))
    cur = bool(re.search(r"\d[\d.,]*\s?(?:(?:" + _SCALE + r")\s?)?(?:(?:de|di|d'|of|van|von)\s?)?(?:" + _CUR + r")", t)
               or re.search(r"(?:€|\$|£)\s?\d", t))
    big = bool(re.search(r"(?<![\d.,])\d{1,3}(?:[.,  ]\d{3})+(?![\d,.]\d)", text) or re.search(r"\d\s?(?:" + _SCALE + r")\b", t)
               or re.search(r"(?<![\d.,])\d{5,}(?![\d.,])", text))
    dec = bool(re.search(r"(?<![\d.,])\d+[.,]\d{1,2}(?![\d.,])", text))
    pct = "%" in text or bool(re.search(r"\d\s?(?:prozent|pour cent|per cento|por ciento|por cento|procent)", t))
    rng = bool(re.search(r"\d\s?[-–]\s?\d", text))
    return {"years": years, "clock": clock, "cur": cur, "big": big, "dec": dec and not clock, "pct": pct, "range": rng}


SAMPLES = ["Die Sitzung beginnt um 15.00 Uhr, 5 Milliarden Euro für 2030.", "La séance reprend à 9h30, 3,5 milliards d'euros.",
           "alle 10.30 si vota; 3,5 miliardi di euro nel 2050", "a las 12.00 horas, 20 millones de dólares, 12%",
           "às 15 horas, 2 milhões de euros em 2035", "om 14.00 uur, 2,5 miljard euro, 3-4 procent", "at 12.30 we vote on 5 billion pounds",
           "from 1.25 to 1.50 per cent", "3 per cent of 10 billion", "it costs 2.50 dollars", "1.234.567 tonnes in 1999"]
for lang in ("de", "fr", "it", "es", "pt", "nl"):
    for s in SAMPLES:
        assert P._gv_classes(s, lang) == _old(s, lang), (lang, s)

# ---- English additions ----
c = P._gv_classes("The vote will take place at 12.30, on 5 billion pounds, about 3 per cent.", "en")
assert c["clock"] and c["cur"] and c["big"] and c["pct"], c
assert P._gv_classes("We meet again at 3 p.m. tomorrow.", "en")["clock"]
assert P._gv_classes("by 9 o'clock", "en")["clock"]
assert not P._gv_classes("from 1.25 per cent to 1.50 per cent", "en")["clock"]  # a decimal after "from" is no clock
assert not P._gv_classes("at 2.50 euros", "en")["clock"]
assert P._gv_classes("We have 500 euros per refugee and $1.30", "en")["cur"]
assert P._gv_classes("by 2050 we need 40 %", "en")["years"] == [2050]
assert V.classes_en("The deadline is 25th of March and May 30th.", P._gv_classes)["date"]
assert not V.classes_en("In 25 member states.", P._gv_classes)["date"]

# ---- select(): held days, caps, strata, labeller gate ----
assert V.template("The vote is at 12.00.") == V.template("The vote is at 11.30.")
rows = []
for i in range(400):
    y = 1990 + i % 41
    day = f"2015{(i % 12) + 1:02d}{(i % 28) + 1:02d}"
    w = "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))  # a distinct word: one template per row
    rows.append({"utt_id": f"{day}-0900-PLENARY_en_{i}", "day": day,
                 "text": f"In {y} {w} spent {i} million euros on project number {i * 7} in the regions of the union."})
rows.append({"utt_id": "20150101-0900-PLENARY_en_999", "day": "20150101",
             "text": "In 2030 we spent 9 million euros on project number 1 in the regions of the union and more."})
held = {"20150101"}
sel, st = V.select(rows, lambda t: V.classes_en(t, P._gv_classes), held, per_year=3, per_forward=0, per_target=3,
                   per_old=0, quotas={"cur": 5}, per_template=2, per_session=3, total_cap=1000, min_s=1, max_s=60,
                   ok=lambda t: "project number 7 " not in t)
assert all(r["day"] not in held for r in sel)
assert st["max_per_session"] <= 3 and st["max_per_template"] <= 2, st
assert all("project number 7 " not in r["text"] for r in sel)
yv = {}
for r in sel:
    for y in P._gv_classes(r["text"], "en")["years"]:
        yv[y] = yv.get(y, 0) + 1
assert all(yv.get(y, 0) >= 3 for y in range(1990, 2031)), yv  # every value reaches its stratum
assert st["clips"] == len(sel) and st["held_days"] == 1

same = [{"utt_id": f"2016030{i}-0900-PLENARY_en_{i}", "day": f"2016030{i}", "text": f"The vote will take place at 1{i}.30 tomorrow and so on."}
        for i in range(1, 6)]
sel2, st2 = V.select(same, lambda t: V.classes_en(t, P._gv_classes), set(), per_year=0, per_forward=0, per_target=0,
                     per_old=0, quotas={"clock": 10}, per_template=2, min_s=1, max_s=60, ok=lambda t: True)
assert len(sel2) == 2 and st2["classes"]["clock"] == 2, (len(sel2), st2)

# ---- manifests for label_gpu ----
m = V.manifests([{"id": "20150202-0900-PLENARY_en_3", "duration": 12.0, "text": "In 2030 we will.", "audio_filepath": "/vol/x.wav"}] * 3, per=2)
assert [len(p) for p in m] == [2, 1]
r = m[0][0]
assert r["kind"] == "digit" and r["original_audio_id"] == "20150202-0900-PLENARY_en" and r["utt_id"] == "20150202-0900-PLENARY_en_3"
assert V.labelable("In 2030 we spent 5 million euros.") and not V.labelable("No digits here.")

print("test_vox_en OK")
