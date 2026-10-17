"""Local unit tests for ml_tts (no GPU, no network, no Modal): python3 test_ml_tts.py"""
import collections
import re

import ml_cands as M
import ml_p3_core as C
import ml_tts as X
import ml_tts_templates as T

N = 240  # 10 per class
FORBID = re.compile(r"[0-9$€£¥%#/+°²³½¼¾ºª‰&]")
PLANS = {lang: X.plan(lang, N, 7) for lang in X.LANGS}


def test_labels_clean():
    for lang, rows in PLANS.items():
        assert rows, lang
        for r in rows:
            assert not FORBID.search(r["text"]), (lang, r["text"])
            assert not C.WF.search(r["text"]), (lang, r["text"])
            assert WORDS_OK(r["caption"]), (lang, r["caption"])
            assert r["text"][-1] in ".!?", r["text"]


def WORDS_OK(text):
    return X.WORDS_MIN <= len(text.split()) <= X.WORDS_MAX


def test_every_class_every_language():
    for lang, rows in PLANS.items():
        have = collections.Counter(r["cls"] for r in rows)
        for c in X.CLASSES:
            assert have[c] >= 8, (lang, c, have[c])  # quota is 10; a class may lose a duplicate or two, never more


def test_class_balance():
    for lang, rows in PLANS.items():
        have = collections.Counter(r["cls"] for r in rows)
        assert max(have.values()) <= 15 and min(have.values()) >= 8, (lang, dict(have))


def test_reading_variants_distributed():
    # across classes the picks are not all the canonical reading, and the same written form gets different readings
    for lang in X.LANGS:
        rows = X.plan(lang, 600, 3)
        multi = [sp for r in rows for sp in r["spans"] if sp["n_cands"] > 1]
        assert sum(1 for sp in multi if sp["idx"] > 0) > 0.3 * len(multi), lang  # not just the canonical reading
        by_text = collections.defaultdict(set)
        for r in rows:
            for sp in r["spans"]:
                by_text[(sp["kind"], sp["text"])].add(sp["reading"])
        assert any(len(v) > 1 for v in by_text.values()), lang
    es = X.plan("es", 1200, 5)
    pt = X.plan("pt", 1200, 5)
    # prices read with and without the cent word / the "y": "doce reais e noventa" and "doze e noventa" style
    assert len({sp["reading"] for r in pt if r["cls"] == "price" for sp in r["spans"]}) > 8
    # clock times: 24 h and 12 h with a day part both occur
    clocks = [sp["reading"] for r in es if r["cls"] in ("clock24", "clock12") for sp in r["spans"]]
    assert any(re.search(r"de la (mañana|tarde|noche|madrugada)", c) for c in clocks), clocks[:5]
    assert any(" y media" in c or " y treinta" in c for c in clocks) or any(re.search(r"\b(?:dieci|veinti)", c) for c in clocks)


def test_num_chars_offsets():
    for lang, rows in PLANS.items():
        for r in rows:
            assert len(r["num_chars"]) == len(r["spans"])
            for (a, b), sp in zip(r["num_chars"], r["spans"]):
                assert 0 <= a < b <= len(r["text"])
                assert r["text"][a:b].strip() == sp["reading"], (lang, r["text"], r["text"][a:b], sp)
            flags = C.number_tokens(r["text"], r["num_chars"])
            assert any(flags), r["text"]


def test_deterministic_by_seed():
    for lang in ("es", "de"):
        a, b = X.plan(lang, 120, 11), X.plan(lang, 120, 11)
        assert [r["text"] for r in a] == [r["text"] for r in b]
        assert [(r["voice"], r["speed"], r["speaker"]) for r in a] == [(r["voice"], r["speed"], r["speaker"]) for r in b]
        c = X.plan(lang, 120, 12)
        assert [r["text"] for r in a] != [r["text"] for r in c]


def test_unique_labels_and_ids():
    for lang, rows in PLANS.items():
        assert len({r["text"] for r in rows}) == len(rows)
        assert len({r["id"] for r in rows}) == len(rows)


def test_written_form_kept_and_spanned():
    for lang, rows in PLANS.items():
        for r in rows:
            sp = M.spans(r["caption"], lang)
            assert [s.text for s in sp] == [s["text"] for s in r["spans"]]
            assert all(s.cands for s in sp)


def test_voices_and_licences():
    ok_lic = {X.CC0, X.CCBY4, X.CCBY3, X.APACHE, X.KOKORO} | {v["license"] for v in X.VOICES["fr"] if v["engine"] == "kokoro"}
    for lang, vs in X.VOICES.items():
        for v in vs:
            assert v["license"] in ok_lic, (lang, v)
            assert "SA" not in v["license"] and "GPL" not in v["license"]
        assert any(not v["dev"] for v in vs)
    for lang, rows in PLANS.items():
        for r in rows:
            assert X.SPEED_LO <= r["speed"] <= X.SPEED_HI
            assert r["split"] == ("dev" if next(v for v in X.VOICES[lang] if v["id"] == r["voice"] and v["dev"] == (r["split"] == "dev"))["dev"] else "train")
            if r["engine"] == "piper" and r["speaker"] is not None:
                v = next(v for v in X.VOICES[lang] if v["id"] == r["voice"] and v["dev"] == (r["split"] == "dev"))
                assert v["spk"][0] <= r["speaker"] <= v["spk"][1]
    # dev voices never leak into train rows and vice versa
    for lang in ("es", "fr", "de", "nl", "pt"):
        rows = X.plan(lang, 600, 9)
        assert any(r["split"] == "dev" for r in rows), lang
    # Italian: two Kokoro voices + style blends; one blend is the held-out (dev) voice
    assert any(r["split"] == "dev" for r in PLANS["it"]) and any(r["voice"].startswith("blend:") for r in PLANS["it"])


def test_gender_and_joiner_constraints():
    # masculine clauses never carry a feminine number word ("doscientas personas"); ranges use the clause's joiner
    for lang, rows in PLANS.items():
        for r in rows:
            for sp in r["spans"]:
                if r["cls"] == "range_from" and not sp["text"][:1].isdigit():
                    assert re.search(T.WANT[(lang, "RANGE_FROM")]["range"], sp["reading"]), (lang, sp)
                if r["cls"] == "range_between" and not sp["text"][:1].isdigit():
                    assert re.search(T.WANT[(lang, "RANGE_BETWEEN")]["range"], sp["reading"]), (lang, sp)
    for r in PLANS["es"]:
        if r["cls"] == "ord":
            fem = "ª" in r["caption"]
            word = r["spans"][0]["reading"]
            assert word.endswith("a") == fem, (r["caption"], word)


def test_check_greedy_rule():
    row = next(r for r in PLANS["es"] if r["cls"] == "count")
    ok, why, info = X.check_greedy(row, row["text"].lower())
    assert ok and info["fit"] == "exact", (why, info)
    # a different number heard: CER small but the reading is not in the greedy text
    a, b = row["num_chars"][0]
    wrong = (row["text"][:a] + "mil novecientos" + row["text"][b:]).lower()
    assert not X.check_greedy(row, wrong)[0]
    # garbage: CER gate
    ok, why, _ = X.check_greedy(row, "xx yy zz")
    assert not ok and why == "cer"


def test_no_forbidden_chars_in_templates():
    for lang, groups in T.CLAUSES.items():
        for g, cl in groups.items():
            for c in cl:
                assert c.count("{n}") == 1, (lang, g, c)
                assert not FORBID.search(c.replace("{n}", "")), (lang, g, c)
    for lang in X.LANGS:
        for _, lead in T.LEADS[lang]:
            assert not FORBID.search(lead)
        for tail in T.TAILS[lang]:
            assert not FORBID.search(tail)


def test_needed_keys_lists_only_missing():
    texts, keys = X.needed_keys("es", 48, 1)
    tab = M.table()
    assert texts and all(k not in tab for k in keys)


def test_gender_outside_counts():
    """Judge audit 2026-10-17: feminine / apocopated number forms in years, dates, percents and masculine clock hours."""
    class Sp:
        def __init__(self, text, kind):
            self.text, self.kind = text, kind
    cases = [("pt", "year", "m", Sp("1956", "int"), "mil novecentas e cinquenta e seis", False),
             ("pt", "year", "m", Sp("1956", "int"), "mil novecentos e cinquenta e seis", True),
             ("pt", "year", "m", Sp("2003", "int"), "duas mil e três", False),
             ("es", "pct", "m", Sp("51 %", "pct"), "cincuenta y una por ciento", False),
             ("es", "pct", "m", Sp("51 %", "pct"), "cincuenta y uno por ciento", True),
             ("es", "date_txt", "m", Sp("1", "date"), "un", False),
             ("pt", "count", "f", Sp("2", "int"), "duas", True),
             ("pt", "clock24", "m", Sp("02:10", "clock"), "duas horas e dez", True),
             ("pt", "clock24", "m", Sp("02:10", "clock"), "dois horas e dez", False),
             ("fr", "clock24", "m", Sp("1:34", "clock"), "un heure trente quatre", False),
             ("it", "pct", "m", Sp("71 %", "pct"), "settantuna per cento", False),
             ("it", "year", "m", Sp("2001", "int"), "duemilaun", False),
             ("it", "year", "m", Sp("2001", "int"), "duemilauno", True),
             ("es", "ord", "f", Sp("3.ª", "ord"), "tercera", True)]
    for lang, cls, g, sp, rd, want in cases:
        assert X._gender_ok(lang, cls, g, sp, rd) == want, (lang, rd)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(len(tests), "tests passed")
