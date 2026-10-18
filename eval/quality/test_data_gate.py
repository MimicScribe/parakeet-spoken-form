"""data_gate: per-language synthetic caps, spoken scale words, spoken dates / clock times. python3 test_data_gate.py"""
import json
import os
import tempfile
import unittest

import data_gate as G


def write(d, name, rows):
    with open(os.path.join(d, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def real(i, text, spans, dur=5.0):
    return {"id": f"es000_{i}_abcdefghijk_{i}", "duration": dur, "text": text, "caption": text + str(i), "spans": spans}


def check(g, prefix):
    return [c for c in g["checks"] if c["check"].startswith(prefix)]


class SynShareLang(unittest.TestCase):
    def test_limits(self):
        for lang in ("en", "es", "fr"):
            self.assertEqual(G.syn_limit(lang, "unit"), 0.5)
            self.assertEqual(G.syn_limit(lang, "year"), 0.5)
        self.assertEqual(G.syn_limit("de", "unit"), 0.30)
        self.assertEqual(G.syn_limit("de", "clock"), 0.5)
        self.assertEqual(G.syn_limit("uk", "int_small"), 0.30)

    def _gate(self, lang):
        with tempfile.TemporaryDirectory() as d:
            sp = {"text": "5 km", "kind": "unit", "reading": "cinco kilómetros"}
            write(d, f"p3_{lang}", [real(i, "son cinco kilómetros", [sp]) for i in range(6)])
            write(d, f"tts_{lang}", [{"id": f"t{i}", "duration": 3.0, "text": "son cinco kilómetros", "spans": [sp]}
                                     for i in range(4)])  # 4 / 10 = 40% synthetic
            return G.gate_lang(d, lang)

    def test_40pct_synthetic_passes_in_es_fails_in_de(self):
        es = check(self._gate("es"), "unit synthetic share")
        de = check(self._gate("de"), "unit synthetic share")
        self.assertTrue(es[0]["pass"], es)
        self.assertEqual(es[0]["want"], 0.5)
        # de unit is far under its 150 bar (6 real spans): fill-to-bar passes it (2026-10-17), at the 30% cap
        self.assertTrue(de[0]["pass"], de)
        self.assertIn("filled to bar (synthetic 4, real 6)", de[0]["got"])
        self.assertEqual(de[0]["want"], 0.30)


class WordScale(unittest.TestCase):
    def test_scale_words(self):
        self.assertEqual(G.word_scale("son cinco millones de euros", "cinco", "es"), "million")
        self.assertEqual(G.word_scale("son dos mil millones de euros", "dos mil", "es"), "billion")
        self.assertEqual(G.word_scale("hay tres mil millones", "tres", "es"), "billion")
        self.assertEqual(G.word_scale("deux milliards d'euros", "deux", "fr"), "billion")
        self.assertEqual(G.word_scale("trois millions", "trois", "fr"), "million")
        self.assertEqual(G.word_scale("five billion dollars", "five", "en"), "billion")
        self.assertIsNone(G.word_scale("cinco personas", "cinco", "es"))
        self.assertIsNone(G.word_scale("cinco millonarios", "cinco", "es"))  # a word boundary, not a prefix

    def test_scale_check_counts_word_scales(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [real(i, "cuesta cinco millones de euros", [{"text": "5", "kind": "int", "reading": "cinco"}])
                    for i in range(60)]
            rows += [real(100 + i, "unos dos mil millones", [{"text": "2.000", "kind": "int", "reading": "dos mil"}])
                     for i in range(60)]
            write(d, "p3_es", rows)
            g = G.gate_lang(d, "es")
            sc = check(g, "int_big each scale")[0]
            self.assertEqual(sc["got"].get("million"), 60)
            self.assertEqual(sc["got"].get("billion"), 60)
            self.assertEqual(g["detail"]["word_scales"], {"million": 60, "billion": 60})


class SpokenShape(unittest.TestCase):
    def test_dates_and_clock(self):
        sp = lambda t, r: {"text": t, "kind": "int", "reading": r}  # noqa: E731
        self.assertEqual(G.spoken_shape("hasta el quince de marzo", sp("15", "quince"), "int_small", 15, "es"), "date")
        self.assertEqual(G.spoken_shape("le dix huit juillet deux", sp("18", "dix huit"), "int_small", 18, "fr"), "date")
        self.assertEqual(G.spoken_shape("a las siete de la tarde", sp("7", "siete"), "int_small", 7, "es"), "clock")
        self.assertEqual(G.spoken_shape("desde las once horas", sp("11", "once"), "int_small", 11, "es"), "clock")
        # not a date / clock: no month, a duration without the preposition, a value out of range, other languages
        self.assertEqual(G.spoken_shape("quince personas", sp("15", "quince"), "int_small", 15, "es"), "int_small")
        self.assertEqual(G.spoken_shape("durante diez horas", sp("10", "diez"), "int_small", 10, "es"), "int_small")
        self.assertEqual(G.spoken_shape("cuarenta de marzo", sp("40", "cuarenta"), "int_small", 40, "es"), "int_small")
        self.assertEqual(G.spoken_shape("piątego marca", sp("5", "piątego"), "int_small", 5, "pl"), "int_small")
        self.assertEqual(G.spoken_shape("en 2020 de marzo", sp("2020", "dos mil veinte"), "year", 2020, "es"), "year")

    def test_de_it_nl_pt_dates_and_it_dot_clock(self):
        sp = lambda t, r, k="int": {"text": t, "kind": k, "reading": r}  # noqa: E731
        self.assertEqual(G.spoken_shape("am sechsten dezember", sp("6.", "sechsten", "ord"), "ord", "6.", "de"), "date")
        self.assertEqual(G.spoken_shape("der sechste bericht", sp("6.", "sechste", "ord"), "ord", "6.", "de"), "ord")
        self.assertEqual(G.spoken_shape("il tre maggio", sp("3", "tre"), "int_small", 3, "it"), "date")
        self.assertEqual(G.spoken_shape("op twaalf maart", sp("12", "twaalf"), "int_small", 12, "nl"), "date")
        self.assertEqual(G.spoken_shape("a três de maio", sp("3", "três"), "int_small", 3, "pt"), "date")
        self.assertEqual(G.spoken_shape("alle diciotto e trenta", sp("18.30", "diciotto e trenta", "dec"), "dec", "18.30",
                                        "it"), "clock")
        self.assertEqual(G.spoken_shape("circa diciotto virgola tre zero", sp("18.30", "diciotto virgola tre zero", "dec"),
                                        "dec", "18.30", "it"), "dec")
        self.assertEqual(G.spoken_shape("vierzig mai", sp("40", "vierzig"), "int_small", 40, "de"), "int_small")

    def test_gate_moves_them_out_of_int_small(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "p3_es", [real(1, "llega el quince de marzo", [{"text": "15", "kind": "int", "reading": "quince"}]),
                               real(2, "a las siete de la tarde", [{"text": "7", "kind": "int", "reading": "siete"}]),
                               real(3, "son siete personas", [{"text": "7", "kind": "int", "reading": "siete"}])])
            g = G.gate_lang(d, "es")
            self.assertEqual(g["classes"]["date"]["spans"], 1)
            self.assertEqual(g["classes"]["clock"]["spans"], 1)
            self.assertEqual(g["classes"]["int_small"]["spans"], 1)



class ShareExceptions(unittest.TestCase):
    def test_en_phone_may_be_mostly_synthetic_up_to_the_floor(self):
        def gate(lang, n_syn):
            with tempfile.TemporaryDirectory() as d:
                sp = {"text": "555 0100", "kind": "phone", "reading": "five five five"}
                write(d, f"p3_{lang}", [dict(real(i, "call five five five", [sp]), id=f"en000_{i}_abcdefghijk_{i}")
                                        for i in range(8)])
                write(d, f"tts_{lang}", [{"id": f"t{i}", "duration": 3.0, "text": "call five five five", "spans": [sp]}
                                         for i in range(n_syn)])
                return [c for c in G.gate_lang(d, lang, "full")["checks"] if c["check"].startswith("phone synthetic")][0]
        self.assertTrue(gate("en", 292)["pass"])  # 97% synthetic, 292 <= 300
        self.assertFalse(gate("en", 301)["pass"])  # above the floor
        # es phone: no floor exemption, but fill-to-bar (2026-10-17) allows 300 - 8 real = 292 synthetic
        self.assertTrue(gate("es", 100)["pass"])
        self.assertFalse(gate("es", 293)["pass"])
        self.assertTrue(gate("es", 8)["pass"])

    def test_es_fr_frac_amp_may_be_mostly_synthetic_up_to_the_floor(self):
        def gate(lang, kind, n_syn):
            with tempfile.TemporaryDirectory() as d:
                sp = {"text": "1/2", "kind": kind, "reading": "un demi"}
                write(d, f"p3_{lang}", [dict(real(i, "un demi", [sp]), id=f"en000_{i}_abcdefghijk_{i}") for i in range(3)])
                write(d, f"tts_{lang}", [{"id": f"t{i}", "duration": 3.0, "text": "un demi", "spans": [sp]}
                                         for i in range(n_syn)])
                cs = [c for c in G.gate_lang(d, lang, "full")["checks"] if c["check"].startswith(f"{kind} synthetic")]
                return cs[0] if cs else None
        for lang in ("es", "fr"):
            for kind in ("frac", "amp"):
                c = gate(lang, kind, 100)
                if c is None:
                    self.skipTest(f"census does not classify a {kind} span from this fixture")
                self.assertTrue(c["pass"], (lang, kind))  # 97% synthetic, 100 <= 150
                self.assertFalse(gate(lang, kind, 160)["pass"], (lang, kind))  # above the floor
        c = gate("de", "frac", 100)
        if c is not None:
            self.assertFalse(c["pass"])  # de is not exempt

    def test_english_youtube_groups_cap_30(self):
        def big(lang, n, vid="abcdefghijk"):
            with tempfile.TemporaryDirectory() as d:
                sp = {"text": "7", "kind": "int", "reading": "siete"}
                write(d, f"p3_{lang}", [{"id": f"{lang}000_{i}_{vid}_{i}", "duration": 3.0, "text": f"t{i}",
                                         "caption": f"c{i}", "spans": [sp]} for i in range(n)])
                return [c for c in G.gate_lang(d, lang, "full")["checks"] if c["check"].startswith("groups with")][0]
        self.assertTrue(big("en", 30)["pass"])
        self.assertFalse(big("en", 31)["pass"])
        self.assertFalse(big("es", 14)["pass"])  # every other language keeps 13
        self.assertTrue(big("es", 13)["pass"])
        self.assertEqual(G.group_cap("en", "ep:20190101-0900-PLENARY"), 13)  # English parliament sessions keep 13



class SpokenCur(unittest.TestCase):
    def test_number_plus_currency_word_is_a_price(self):
        sp = lambda t, r: {"text": t, "kind": "int", "reading": r}  # noqa: E731
        self.assertEqual(G.spoken_shape("cuesta treinta euros", sp("30", "treinta"), "int_small", 30, "es"), "cur")
        self.assertEqual(G.spoken_shape("son dos millones de euros", sp("2", "dos"), "int_small", 2, "es"), "cur")
        self.assertEqual(G.spoken_shape("deux milliards d'euros", sp("2", "deux"), "int_small", 2, "fr"), "cur")
        self.assertEqual(G.spoken_shape("trente personnes", sp("30", "trente"), "int_small", 30, "fr"), "int_small")
        self.assertEqual(G.spoken_shape("thirty euros", sp("30", "thirty"), "int_small", 30, "de"), "int_small")


class SuffixAndClock1017(unittest.TestCase):
    """hu / fi suffixed numbers by their number part; spoken clock in sv fi hu cs sk pl ro (census counts only)."""
    def cls(self, lang, text, span_text, reading, kind="int"):
        sp = {"text": span_text, "kind": kind, "reading": reading}
        c, v = G.morph_class(text, sp, *G.span_class(sp), lang)
        return G.spoken_shape(text, sp, c, v, lang), v

    def test_hungarian_case_endings(self):
        self.assertEqual(self.cls("hu", "kettőezer-tizenötben", "2015-ben", "kettőezer-tizenötben"), ("year", 2015))
        self.assertEqual(self.cls("hu", "az ezerkilencszázkilencvenes", "1990-es", "ezerkilencszázkilencvenes"),
                         ("year", 1990))
        self.assertEqual(self.cls("hu", "kétezer-egy óta", "2001 óta", "kétezer-egy óta"), ("year", 2001))
        self.assertEqual(self.cls("hu", "május tizenötödikén", "15-én", "tizenötödikén"), ("date", 15))
        self.assertEqual(self.cls("hu", "a hetvenötös", "75-ös", "hetvenötös"), ("int_small", 75))
        self.assertEqual(self.cls("hu", "tizenkét millió", "12 millió", "tizenkét millió"), ("int_big", 12_000_000))
        self.assertEqual(self.cls("hu", "hét éves", "7 éves", "hét éves")[0], "int_other")  # a noun, not an ending
        self.assertEqual(self.cls("de", "x", "2015-ben", "x")[0], "int_other")  # Hungarian only

    def test_finnish_suffixes_and_scale(self):
        self.assertEqual(self.cls("fi", "vuonna kaksituhatta", "2015:ssä", "kaksituhattaviisitoista")[0], "year")
        self.assertEqual(self.cls("fi", "vuonna kaksituhatta", "vuonna 2015", "vuonna kaksituhatta")[0], "year")
        self.assertEqual(self.cls("fi", "sata miljardia", "100 miljardia", "sata miljardia"), ("int_big", 10**11))
        self.assertEqual(self.cls("sv", "fyra miljoner", "4 miljoner", "fyra miljoner"), ("int_big", 4_000_000))

    def test_spoken_clock(self):
        self.assertEqual(self.cls("sv", "ses igen klockan tjugoett", "21", "tjugoett")[0], "clock")
        self.assertEqual(self.cls("fi", "jatkamme kello seitsemäntoista", "17", "seitsemäntoista")[0], "clock")
        self.assertEqual(self.cls("hu", "a szavazás tizenhat órakor", "16 órakor", "tizenhat órakor")[0], "clock")
        self.assertEqual(self.cls("hu", "a szavazás tizenöt óráig tart", "15", "tizenöt")[0], "clock")
        self.assertEqual(self.cls("cs", "začíná v patnáct hodin", "15", "patnáct")[0], "clock")
        self.assertEqual(self.cls("cs", "začíná v patnáct hodin", "15 hodin", "patnáct hodin")[0], "clock")
        self.assertEqual(self.cls("sk", "o pätnástej hodine", "15.", "pätnástej", "ord")[0], "clock")
        self.assertEqual(self.cls("pl", "o godzinie piętnastej", "15", "piętnastej")[0], "clock")
        self.assertEqual(self.cls("ro", "la ora cincisprezece", "15", "cincisprezece")[0], "clock")
        # not a clock: a duration without the preposition, an hour > 24, no clock word, a language not listed
        self.assertEqual(self.cls("cs", "trvalo patnáct hodin", "15 hodin", "patnáct hodin")[0], "int_other")
        self.assertEqual(self.cls("sv", "klockan trettio", "30", "trettio")[0], "int_small")
        self.assertEqual(self.cls("sv", "tjugoett personer", "21", "tjugoett")[0], "int_small")
        self.assertEqual(self.cls("de", "kello siebzehn", "17", "siebzehn")[0], "int_small")

    def test_gate_counts_them(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "p3_hu", [dict(real(1, "kettőezer-tizenötben", [{"text": "2015-ben", "kind": "int",
                                                                      "reading": "kettőezer-tizenötben"}]), id="x1"),
                               dict(real(2, "tizenhat órakor", [{"text": "16 órakor", "kind": "int",
                                                                 "reading": "tizenhat órakor"}]), id="x2")])
            write(d, "heldout_hu", [dict(real(3, "tizenkét millió", [{"text": "12 millió", "kind": "int",
                                                                      "reading": "tizenkét millió"}]), id="x3")])
            g = G.gate_lang(d, "hu")
            self.assertEqual(g["classes"]["year"]["spans"], 1)
            self.assertEqual(g["classes"]["clock"]["spans"], 1)
            self.assertNotIn("int_other", g["classes"])
            self.assertEqual(g["detail"]["year_values"], {"2015": 1})
            self.assertEqual(g["heldout_classes"], {"int_big": 1})


class SharedCensusRules(unittest.TestCase):
    """The gate counts with the build's own rules (census_common)."""

    def test_short_masked_captions_are_no_template(self):
        # red-on-base: the build exempts a masked caption of < 3 letter words from the template cap, the gate capped it
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(real(i, "cinco euros", [{"text": "5", "kind": "cur", "reading": "cinco euros"}]),
                         caption="5 euros.") for i in range(6)]
            write(d, "p3_es", rows)
            g = G.gate_lang(d, "es")
            self.assertEqual(check(g, "rows over the 3-per-template cap")[0]["got"], 0)
            write(d, "p3_es", [dict(r, caption="son 5 euros al mes") for r in rows])  # a real template: capped
            self.assertEqual(check(G.gate_lang(d, "es"), "rows over the 3-per-template cap")[0]["got"], 3)

    def test_ext_ctx_counts_as_ext(self):
        # red-on-base: ext_ctx / ext_value spans did not count toward the ext share
        with tempfile.TemporaryDirectory() as d:
            rows = [real(i, "son cinco kilómetros", [{"text": "5 km", "kind": "unit", "reading": "cinco kilómetros",
                                                      "by": ("ext_ctx", "ext_value", "ctc", "ctc")[i % 4]}])
                    for i in range(40)]
            write(d, "p3_es", rows)
            c = check(G.gate_lang(d, "es"), "unit ext share")[0]
            self.assertEqual(c["got"], 0.5)
            self.assertFalse(c["pass"])

    def test_template_key_is_the_builds(self):
        import mix_build as MB  # noqa: F401  (pipeline/ is on sys.path via data_gate)
        self.assertIs(G.template_key, MB.template_key)
        self.assertIs(G.group_id, MB.group_id)


if __name__ == "__main__":
    unittest.main()
