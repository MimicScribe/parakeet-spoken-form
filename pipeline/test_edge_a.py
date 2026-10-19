"""Edge-case pack, tier A: spoken-form labels, frozen splits, the planner draws the A halves only (no GPU, no network).
python3 test_edge_a.py   (new module; every assert is a control, red-on-base: control)"""
import hashlib
import json
import os
import random
import re

import edge_a as E
import edge_a_content as K
import edge_a_frames as F

BAD = re.compile(r"[0-9{}$%€£#@+_/\-]|[−×=]")
BAD_TEXT = re.compile(r"[0-9{}$%€£#@+_/]|[−×=]")     # a sentence may hold an ordinary hyphen (fr "moment-là"); a number phrase never

# 1. every phrase of every construction: spoken form only, readings are substrings, letters are letters
for lang in E.LANGS:
    cons = E.inventory(lang)
    ids = [c.id for c in cons]
    assert len(set(ids)) == len(ids)
    for c in cons:
        assert c.cat in F.FRAMES[lang] or c.value_cores, (c.id, c.cat)
        assert c.lang == lang and c.family in ("fin", "col", "dis")
        assert len(set(c.values)) == len(c.values), c.id
        for k in c.values:
            p = c.make(k, random.Random(k))
            assert not BAD.search(p.label), (c.id, k, p.label)
            for kind, written, reading in p.nums:
                assert reading in p.label, (c.id, k, reading, p.label)
            if c.value_cores:
                assert k in c.value_cores and all("{X}" in x for x in c.value_cores[k]), (c.id, k)
        cores = F.FRAMES[lang].get(c.cat, [])
        assert all("{X}" in x and not BAD.search(x.replace("{X}", "")) for x in cores), c.id
    for fam in ("fin", "col", "dis"):
        assert F.LEADS[lang][fam] and F.TAILS[lang][fam]
# letters as letters, letter + digit codes keep the number word (policy a)
en = {c.id: c for c in E.inventory("en")}
assert en["en.fin.q_code"].make("3:2026", random.Random(1)).label == "Q three twenty twenty six"
assert en["en.fin.sec_form"].make("10K", random.Random(1)).label == "ten K"
assert en["en.fin.eps_dollar"].make("1.05", random.Random(2)).label.count("oh five") == 1
assert en["en.fin.scale_ellipsis"].make("b1.2", random.Random(1)).label in ("a billion two", "one billion two")
assert en["en.fin.bips"].make("25", random.Random(1)).label == "twenty five bips"

# 2. frozen splits: halves are disjoint and exhaustive, held-out constructions exist, the file hash is stable
for lang in E.LANGS:
    s = E.freeze(lang)
    assert s == E.freeze(lang)
    for cid, c in s["constructions"].items():
        assert not set(c["V_a"]) & set(c["V_b"]), cid
        assert sorted(c["V_a"] + c["V_b"]) == sorted(next(x for x in E.inventory(lang) if x.id == cid).values), cid
        if c["split_values"]:
            assert abs(len(c["V_a"]) - len(c["V_b"])) <= 1 or c["n_values"] % 2, cid
            assert c["V_b"], cid
        else:
            assert not c["V_b"]
    held = {cid for cid, c in s["constructions"].items() if c["side"] == "C_b"}
    assert held == E.C_B[lang]
    for fam in ("fin", "col", "dis"):
        n = s["n_constructions"][fam]
        assert 0 < s["n_held_out"][fam] <= max(1, round(0.3 * n)), (lang, fam)
    assert E.side_of(s, sorted(held)[0], "x") == "Cb"
    cid = next(i for i, c in s["constructions"].items() if c["side"] == "C_a" and c["V_b"])
    assert E.side_of(s, cid, s["constructions"][cid]["V_a"][0]) == "A" and E.side_of(s, cid, s["constructions"][cid]["V_b"][0]) == "B"
    assert E.hash_side(lang, cid, "zz") in ("A", "B") and E.hash_side(lang, cid, "zz") == E.hash_side(lang, cid, "zz")
    # an on-disk split (if present) must be the frozen one
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_a_split", f"ec1split_{lang}.json")
    if os.path.exists(path):
        disk = json.load(open(path, encoding="utf-8"))
        man = json.load(open(os.path.join(os.path.dirname(path), "manifest.json")))
        assert disk == s and hashlib.sha256(E.canonical(s)).hexdigest() == man["sha256"][lang], lang

# 3. the planner: A halves only, no digit, offsets point at the phrases and the readings, tags follow the value
for lang in E.LANGS:
    s = E.freeze(lang)
    rows, rep = E.plan(lang, s, {"fin": 120, "col": 80, "dis": 30})
    E.add_gate_classes(rows, lang)
    assert len(rows) == 230 and len({r["text"] for r in rows}) == 230
    for r in rows:
        assert not BAD_TEXT.search(r["text"]), r["text"]
        for sp in r["spans"]:
            assert "-" not in sp["reading"], r["text"]
        assert r["half"] == "A" and r["split"] == "train"
        cs = s["constructions"][r["construction"]]
        assert cs["side"] == "C_a"
        for cid, k in zip(r["constructions"], r["values"]):
            assert E.side_of(s, cid, k) == "A", (cid, k)
        assert len(r["num_chars"]) == len(r["spans"]) and r["edge_spans"]
        for (a, b), sp in zip(r["num_chars"], r["spans"]):
            assert r["text"][a:b] == sp["reading"], (r["text"], r["text"][a:b], sp["reading"])
            assert any(ea <= a and b <= eb for ea, eb in r["edge_spans"])
        assert r["gclasses"] and r["ckey"]
    # nothing from a held-out construction or V_b
    assert not {r["construction"] for r in rows} & E.C_B[lang]

# 4. open-decision tags travel with the rows
rows, _ = E.plan("en", E.freeze("en"), {"fin": 400, "col": 200, "dis": 60}, seed=3)
tags = {t for r in rows for t in r["ambig"]}
assert {"slang_value", "elided_hundreds", "disfluency_verbatim", "bips_bil_mil_k", "tickers_indices", "paired_years", "scores", "nine_eleven"} <= tags, tags
for r in rows:
    if r["construction"] == "en.fin.q_code":
        assert ("paired_years" in r["ambig"]) == any(":" in k for k in r["values"])
    if r["construction"] == "en.dis.self_fix":
        assert "disfluency_verbatim" in r["ambig"]

# 5. French elision across the phrase / template boundary
fr = {c.id: c for c in E.inventory("fr")}
r = E.assemble("fr", "col", "le projet a coûté {X} euros", {"X": fr["fr.col.dizaine"].make("ct", random.Random(1))}, random.Random(1), None, None)
assert r["text"] == "Le projet a coûté une centaine d'euros." and r["text"][slice(*r["edge_spans"][0])] == "une centaine d'", r
r = E.assemble("fr", "fin", "le groupe a annoncé un chiffre d'affaires de {X}", {"X": fr["fr.fin.echelle_ellipse"].make("b1.2", random.Random(1))},
               random.Random(1), None, None)
assert "d'un milliard deux d'euros" in r["text"], r["text"]
# no-comma leads and a sentence that starts with the phrase
es = {c.id: c for c in E.inventory("es")}
r = E.assemble("es", "dis", "costó {X} euros", {"X": es["es.dis.eh_pair"].make("20-30", random.Random(1))}, random.Random(1), "Creo que", None)
assert r["text"].startswith("Creo que costó veinte "), r["text"]
r = E.assemble("en", "col", "{X} users signed up overnight", {"X": en["en.col.couple"].make("ch", random.Random(1))}, random.Random(1), None, None)
assert r["text"] == "A couple hundred users signed up overnight." and r["edge_spans"][0] == [0, 16], r
print("test_edge_a ok")
