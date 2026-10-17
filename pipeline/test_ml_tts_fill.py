"""Targeted synthetic top-up (no Modal, no API): python3 test_ml_tts_fill.py"""
import collections
import random
import re

import gen
import ml_tts as T
import ml_tts_fill as F
import readings

# ---- gen.targeted_rows: exact values, one slot, normal readings, no digits in labels
q = {"year": {2043: 5, 2060: 3}, "count": {82: 4, 350: 2}}
rows = list(gen.targeted_rows(q, 3))
assert len(rows) == 14
got = collections.Counter((r["target"], r["value"]) for r in rows)
assert got == collections.Counter({("year", 2043): 5, ("year", 2060): 3, ("count", 82): 4, ("count", 350): 2}), got
for r in rows:
    assert not gen.readings_written_form(r["text"]), r["text"]
    assert len(gen.SLOT.findall(r["template"])) == 1 and "{any}" not in r["template"]
    assert r["kinds"][0] in gen.TARGET_SLOTS[r["target"]]
    if r["target"] == "year":
        assert readings.year(r["value"]) in r["text"].lower() or "two thousand" in r["text"].lower(), r["text"]
    else:
        assert readings.cardinal(r["value"]) in r["text"].lower() or readings.cardinal(r["value"], True) in r["text"].lower()
assert list(gen.targeted_rows(q, 3)) == rows  # deterministic
assert {r["split"] for r in gen.targeted_rows({"year": {2045: 60}}, 1)} == {"train", "dev"}

# ---- ml_tts.COUNT_QUOTA: the count class voices exactly the quota values (French 61-99 gaps)
T.COUNT_QUOTA = {61: 2, 71: 2, 82: 2, 97: 2}
try:
    rep = {}
    rs = T.plan("fr", 8, 5, rep, False, ("count",), "t")
finally:
    T.COUNT_QUOTA = {}
vals = {int(re.sub(r"\D", "", s["text"])) for r in rs for s in r["spans"]}
assert vals <= {61, 71, 82, 97} and len(rs) == 8, (vals, len(rs))
assert all(not re.search(r"\d", r["text"]) for r in rs)
# the quota is off again: the usual spread comes back
rs2 = T.plan("fr", 20, 5, {}, False, ("count",), "t")
assert {int(re.sub(r"\D", "", s["text"])) for r in rs2 for s in r["spans"]} - {61, 71, 82, 97}

# ---- wave_rows: year + count quotas planned together, quotas reset afterwards
rep = {}
wr = F.wave_rows("es", {"year": {"2028": 3, "2036": 2}, "count": {"75": 2}}, "t", 9, rep)
assert collections.Counter(r["cls"] for r in wr) == {"year": 5, "count": 2}, collections.Counter(r["cls"] for r in wr)
assert F.value_counts(wr, "es")["year"] == {2028: 3, 2036: 2}
assert not T.YEAR_QUOTA and not T.COUNT_QUOTA

# ---- assign_engines: never years, at most 1/3 of each class, only voices of the row's split
rows = [{"cls": "year", "split": "train", "engine": "presynth"} for _ in range(30)]
rows += [{"cls": "count", "split": "train" if i % 5 else "dev", "engine": "presynth"} for i in range(30)]
F.assign_engines(rows, 0.9, random.Random(1), lambda s: ["kv_train"] if s == "train" else [])
assert all(r["engine"] == "presynth" for r in rows if r["cls"] == "year")
k = [r for r in rows if r["engine"] == "kokoro"]
assert len(k) == 10, len(k)  # share capped at 1/3 of 30
assert all(r["split"] == "train" and r["voice"] == "kv_train" for r in k)
rows = [{"cls": "count", "split": "train", "engine": "presynth"} for _ in range(30)]
F.assign_engines(rows, 0.2, random.Random(1), lambda s: ["v"])
assert sum(r["engine"] == "kokoro" for r in rows) == 6
print("ok")

# ---- English slot rows (phone top-up): carriers.txt + carriers_phone.txt, every row has a phone slot, no digits
er = F.en_rows({"slots": {"phone": 40}}, "t", 3)
assert len(er) == 40 and all(r["cls"] == "phone" and "phone" in r["kinds"] for r in er)
assert len({r["id"] for r in er}) == 40
assert not any(re.search(r"\d", r["text"]) for r in er)
assert {r["template"] for r in er} & set(gen.load_carriers(gen.os.path.join(gen.HERE, "carriers_phone.txt")))
print("ok slots")

# ---- UK / AU forms: rows carry their locales and the class of the form
uk = F.en_rows({"ukau": {"phone": 5, "money": 5}}, "t", 3)
assert len(uk) == 10 and all(r["locales"] == ["en-GB", "en-AU"] for r in uk)
assert {r["cls"] for r in uk} == {"phone", "money"}
print("ok ukau")
