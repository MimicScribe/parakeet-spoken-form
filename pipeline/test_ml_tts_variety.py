"""Reading-variety fill (2026-10-18; no Modal, no API): python3 test_ml_tts_variety.py

reading_conv conventions, the fill-only readings (offered only when named), the natural phone-group fix and the
decimal + cents ban in ml_tts, the convention pick + replacement planning of ml_tts_variety, and mix_build's
order_synthetic (B2: superseded ids leave, replacing rows first, the old rows' order unchanged)."""
import collections
import random

import ml_cands as M
import ml_tts as T
import ml_tts_variety as V
import mix_build as MB
import reading_conv as RC

# ---- reading_conv: conventions of hand cases (and the Italian split spelling reads the same as the joined one)
assert RC.clock("de", "14:30", "halb drei", "")[0] == "half_next"
assert RC.clock("de", "14:25", "fünf vor halb drei", "")[0] == "near_half"  # before 2026-10-18: half_next
assert RC.clock("nl", "16:40", "tien over half vijf", "")[0] == "near_half"
assert RC.clock("es", "14:45", "tres menos cuarto", "")[0] == "quarter_to"
assert RC.year("nl", 1990, "negentien negentig")[0] == "pairs"
assert RC.year("it", 1998, "mille nove cento novanta otto")[0] == RC.year("it", 1998, "millenovecentonovantotto")[0] == "thousands"
assert RC.phone_groups("es", [("612", "seis doce"), (" 34", " treinta y cuatro")])[0] == "1+2_splits"
assert RC.phone_groups("es", [("612", "sesenta y uno dos")])[0] == "UNNATURAL_split"
assert RC.cur("de", "2,50 €", "zwei fünfzig", "", True)[0] == "bare_no_unit"
assert RC.dec("es", "3,5", "tres con cinco")[0] == "con"
assert RC.pct("nl", "12%", "twaalf percent", "")[0] == "percent"

# ---- ml_cands: the fill readings only when named; spans() and the regional default are unchanged
txt = {"de": "Der Kaffee kostet 2,50 € und die Nummer ist 0171 23 45 62.", "es": "Mide 3,5 metros, un 20 % más.",
       "nl": "Er kwamen 2-3 mensen, 12% meer.", "pt": "Liga para 11 96 34 56 78 hoje."}
for lang, t in txt.items():
    base = [s.cands for s in M.spans(t, lang)]
    reg = [s.cands for s in M.spans_regional(t, lang)]
    fill = [s.cands for s in M.spans_regional(t, lang, M.VARIETY_ALL | M.FILL_ALL)]
    assert not any(set(f) & {"zwei fünfzig", "tres con cinco", " à drie", "twaalf percent", " nove meia"} for f in base + reg)
    assert all(f[:len(r)] == r for f, r in zip(fill, reg))  # only appended
allc = lambda t, l: {c for s in M.spans_regional(t, l, M.FILL_ALL) for c in s.cands}  # noqa: E731
assert "zwei fünfzig" in allc(txt["de"], "de") and " zwo drei" in allc(txt["de"], "de")
assert "tres con cinco" in allc(txt["es"], "es") and "veinte por cien" in allc(txt["es"], "es")
assert " à drie" in allc(txt["nl"], "nl") and "twaalf percent" in allc(txt["nl"], "nl")
assert " nove meia" in allc(txt["pt"], "pt") and not any("noventa e meia" in c for c in allc(txt["pt"], "pt"))

# ---- ml_tts (R2): a 3-digit phone group is never read 2+1 ("sesenta y uno dos"); before 2026-10-18 the "chunks"
# style fell back to every non-digit reading for groups of 3+ digits
rng, used, n, bad = random.Random(3), collections.Counter(), 0, 0
for k in range(120):
    w = f"Llama al {rng.choice('69')}{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} hoy."
    res, _ = T.realize("es", "phone", w, "PHONE", None, rng, used)
    if res:
        n += 1
        bad += RC.phone_groups("es", [(s["text"], s["reading"]) for s in res[1]])[0] == "UNNATURAL_split"
assert n > 50 and bad == 0, (n, bad)

# ---- ml_tts (R5): a decimal word and a cents word in one price reading is no reading
sp = [s for s in M.spans("Costa 12,50 € oggi.", "it") if s.kind == "cur"][0]
assert not T.reading_ok("it", "MONEY", "price", None, sp, "dodici euro virgola cinquanta centesimi", 0, "Costa 12,50 € oggi.")
assert T.reading_ok("it", "MONEY", "price", None, sp, "dodici euro e cinquanta centesimi", 0, "Costa 12,50 € oggi.")

# ---- ml_tts_variety: replacement planning on a small synthetic mix
def row(rid, lang, caption, cls):
    rs, why = T.realize(lang, cls, caption, T.GROUP[cls], None, random.Random(rid), collections.Counter())
    assert rs, (caption, why)
    label, spans, offs = rs
    return {"id": rid, "lang": lang, "src": "tts", "caption": caption, "text": label, "spans": spans, "num_chars": offs}


mix = [row(f"old_de_{i}", "de", f"Der Zug fährt um {h}:{m:02d} Uhr pünktlich ab.", "clock24")
       for i, (h, m) in enumerate([(14, 30), (9, 15), (16, 45), (18, 30), (7, 30), (20, 15), (11, 45), (13, 30)])]
mix += [row(f"oldy_nl_{i}", "nl", f"Het bedrijf werd in {y} opgericht.", "year") for i, y in enumerate([1990, 1975, 1984, 1966])]
cells = [{"lang": "de", "cls": "clock", "mode": "replace", "add": 0,
          "target": {"digital_hourword": .5, "half_next": .25, "quarter_past": .125, "quarter_to": .125}},
         {"lang": "nl", "cls": "year:1800s-1900s", "mode": "replace", "add": 0, "target": {"pairs": .5, "hundreds": .5}}]
for lang in ("de", "nl"):
    rep = {}
    rows = V.plan_lang(lang, cells, [r for r in mix if r["lang"] == lang], "ttsRVt", 1, rep)
    assert rows, rep
    ids = {r["id"] for r in mix}
    sup = [i for r in rows for i in r["supersedes"]]
    assert sup and len(sup) == len(set(sup)) and set(sup) <= ids
    for r in rows:  # the census reads every number with its planned convention, and nothing else is in the row
        us = RC.units(dict(r, src="tts"), lang)
        assert sorted(RC.classify(lang, u)[0] for u in us) == sorted(r["convs"]), r
        assert not any(ch.isdigit() for ch in r["text"])
    if lang == "nl":  # years keep the values of the rows they replace
        old_vals = sorted(int(s["text"]) for x in mix if x["id"] in sup for s in x["spans"])
        new_vals = sorted(int(s["text"]) for r in rows for s in r["spans"])
        assert old_vals == new_vals and all(c == "pairs" for r in rows for c in r["convs"]), (old_vals, new_vals)
    else:
        assert {c for r in rows for c in r["convs"]} <= {"half_next", "quarter_past", "quarter_to"}

# ---- mix_build.order_synthetic (B2): without a replacing set = the old shuffle; with one: superseded rows out,
# replacing rows first, the other rows in the order the old build had
old = [{"id": f"r{i}", "_set": "tts1"} for i in range(40)]
a = list(old)
random.Random(7).shuffle(a)
got, st = MB.order_synthetic(list(old), random.Random(7), 1, "de")
assert got == a and st["replacing_rows"] == 0
repl = [{"id": "n1", "_set": "ttsRV1b", "supersedes": ["r3", "r5"]}, {"id": "n2", "_set": "ttsRV1b", "supersedes": ["r9"]}]
got, st = MB.order_synthetic(list(old) + repl, random.Random(7), 1, "de")
assert {x["id"] for x in got[:2]} == {"n1", "n2"} and [x for x in got[2:]] == [x for x in a if x["id"] not in ("r3", "r5", "r9")]
assert st == {"replacing_rows": 2, "superseded_ids": 3, "superseded_dropped": 3}
assert "ttsRV1b" in MB.TTS_TARGETED and ("ttsRV1b", "ttsRV1b_judge") in MB.TTS_SETS and MB.TTS_REPLACING == {"ttsRV1b"}
assert "supersedes" in T.PASS_THROUGH and "convs" in T.PASS_THROUGH
# ---- census_common: a decimal span read as two numbers ("8.30" -> "ocho treinta") is counted apart (dec_pair), a
# decimal reading (decimal word, es "con", a half) stays "dec"
import census_common as CC  # noqa: E402

for lang, cap, rd, want in [("es", "a las 8.30", "ocho treinta", "dec_pair"), ("es", "mide 3,5", "tres coma cinco", "dec"),
                            ("es", "mide 3,5", "tres con cinco", "dec"), ("es", "mide 2,5", "dos y medio", "dec"),
                            ("de", "Artikel 5.2", "fünf zwei", "dec_pair"), ("de", "es sind 2,5", "zweieinhalb", "dec"),
                            ("nl", "het is 2,5", "twee en half", "dec"), ("fr", "il fait 3,2", "trois virgule deux", "dec")]:
    num = cap.split()[-1]
    sp = {"text": num, "kind": "dec", "reading": rd}
    r = {"text": cap.replace(num, rd), "caption": cap, "spans": [sp]}
    assert CC.gate_class(r, sp, lang) == want, (lang, cap, rd, CC.gate_class(r, sp, lang))
print("ml_tts_variety ok")
