"""Generator grammar fixes (grammar_rules) and the build's exclusion of synthetic rows that spoke the old slips.
Every test is red on the commit before them. Needs the modal + numpy packages for the build4 part:
uv run --no-project --with modal==1.6.1 --with numpy python test_grammar_fix.py"""
import json
import random
import unittest
from collections import Counter

import census_common as CC
import grammar_rules as G
import mix_build as MB
import ml_cands as M
import ml_tts as X
from test_census_common import _fixture, _tts_fixture, rj


def cands(text, lang):
    sp = M.spans(text, lang)
    assert sp, (text, lang)
    return sp[0].cands


def errs(lang, text, kind, reading, cls="", caption="", spoken=""):
    return G.span_errors(lang, {"text": text, "kind": kind, "reading": reading},
                         {"cls": cls, "caption": caption, "text": spoken})


class Rules(unittest.TestCase):
    """Each rule fires on the slip the review found and not on the right form."""

    def test_rules(self):
        # red-on-base: control (grammar_rules is new; the generator tests below are red on base)
        cases = [  # (lang, written piece, kind, cls, caption, spoken sentence, bad reading, good reading, code)
            ("es", "1º", "ord", "date_txt", "el 1º de octubre de 2020", "", "primera", "primero", "date_ord_gender"),
            ("es", "1º", "ord", "date_txt", "el 1º de octubre de 2020", "", "primer", "primero", "date_ord_apocope"),
            ("pt", "1º", "ord", "date_txt", "o 1º de março", "", "primeira", "primeiro", "date_ord_gender"),
            ("it", "1º", "ord", "date_txt", "il 1º febbraio", "", "prima", "primo", "date_ord_gender"),
            ("fr", "1er", "ord", "date_txt", "le 1er octobre", "", "première", "premier", "date_ord_gender"),
            ("pt", "2:30 pm", "clock", "clock12", "", "", "dois e trinta", "duas e trinta", "pt_clock_hour_masc"),
            ("pt", "21h15", "clock", "clock24", "", "", "vinte e um e quinze", "vinte e uma e quinze",
             "pt_clock_hour_masc"),
            ("fr", "16h15", "clock", "clock24", "", "", "seize heures et quinze", "seize heures quinze", "fr_clock_et"),
            ("fr", "12:45 a. m.", "clock", "clock12", "", "", "douze heures quarante cinq", "minuit quarante cinq",
             "fr_12am_douze"),
            ("fr", "21h05", "clock", "clock24", "", "", "vingt et un heures cinq", "vingt et une heures cinq",
             "fr_clock_hour_un"),
            ("fr", "18h45", "clock", "clock24", "", "", "dix huit heure quarante cinq", "dix huit heures quarante cinq",
             "hour_noun_number"),
            ("es", "$ 29,10", "cur", "price", "", "", "veintinueve pesos con diez céntimos",
             "veintinueve pesos con diez centavos", "cents_word"),
            ("pt", "R$ 91,99", "cur", "price", "", "", "noventa e um reais e noventa e nove cêntimos",
             "noventa e um reais e noventa e nove centavos", "cents_word"),
            ("de", "48,10 €", "cur", "price", "", "", "achtundvierzig und zehn", "achtundvierzig Euro und zehn",
             "price_two_numbers"),
            ("nl", "€ 83,10", "cur", "price", "", "", "drieëntachtig en tien", "drieëntachtig euro en tien",
             "price_two_numbers"),
            ("es", "½", "frac", "frac", "La receta lleva ½ de litro.", "", "medio", None, "frac_half_unit"),
            ("nl", "6 1/2", "frac", "frac", "", "", "zes en één half", "zes en een half", "nl_een_half"),
            ("it", "26º", "ord", "ord", "", "", "ventiseesimo", "ventiseiesimo", "it_spelling"),
            ("it", "2003", "int", "year", "", "", "duemilatre", "duemilatré", "it_spelling"),
            ("it", "16:15", "clock", "clock24", "", "", "sedici ore e un quarto", "sedici e un quarto", "it_clock_ore"),
            ("pt", "51000", "int", "count", "", "", "cinco dez zero zero", "cinquenta e um mil", "count_digit_chunks"),
            ("uk", "1 грн.", "cur", "price", "", "Цей товар коштує одна гривня.", "одна гривня", "одну гривню",
             "slav_cost_acc"),
            ("ro", "49,50 lei", "cur", "price", "", "", "patruzeci și nouă lei și cincizeci de bani",
             "patruzeci și nouă de lei și cincizeci de bani", "ro_de_linker"),
            ("ro", "1990", "int", "year", "", "", "mie nouă sute nouăzeci", "o mie nouă sute nouăzeci", "ro_o_mie"),
            ("ro", "592,99 €", "cur", "price", "", "", "cinci sute nouăzeci și două de euro",
             "cinci sute nouăzeci și doi de euro", "ro_noun_gender"),
            ("ro", "2002", "int", "year", "", "", "doi mii doi", "două mii doi", "ro_gender"),
            ("pl", "18.55", "clock", "clock", "", "", "osiemnastej pięćdziesiątej piątej", "osiemnastej pięćdziesiąt pięć",
             "pl_clock_minute"),
            ("ro", "3,50 €", "cur", "price", "", "", "trei euro și cincizeci de bani", "trei euro și cincizeci de cenți",
             "cur_word_mismatch"),
            ("pl", "74,11 €", "cur", "price", "", "", "siedemdziesiąt cztery złote jedenaście groszy",
             "siedemdziesiąt cztery euro jedenaście centów", "cur_word_mismatch"),
            ("pl", "2 680", "int", "big", "", "Zabrakło nam dwa tysiące sześćset osiemdziesiąt euro.",
             "dwa tysiące sześćset osiemdziesiąt", "dwóch tysięcy sześciuset osiemdziesięciu", "pl_gen_case"),
        ]
        for lang, st, kind, cls, cap, spoken, bad, good, code in cases:
            sb = spoken.replace("одна гривня", bad) if spoken else ""
            sg = spoken.replace(bad, good) if spoken and good else ""
            self.assertIn(code, errs(lang, st, kind, bad, cls, cap, sb), (lang, bad))
            if good is not None:  # "½ de litro": no reading is right, the written form is never drawn now
                self.assertNotIn(code, errs(lang, st, kind, good, cls, cap, sg), (lang, good))
        # right forms the rules must leave alone
        self.assertEqual(errs("fr", "21h05", "clock", "vingt et une heures cinq", "clock24"), [])
        self.assertEqual(errs("fr", "1:15", "clock", "une heure et quart", "clock12"), [])
        self.assertEqual(errs("es", "¼", "frac", "un cuarto", "frac", "La receta lleva ¼ de litro."), [])
        self.assertEqual(errs("ro", "19,50 lei", "cur", "nouăsprezece lei și cincizeci de bani", "price"), [])
        self.assertEqual(errs("ro", "105 lei", "cur", "o sută cinci lei", "price"), [])
        self.assertEqual(errs("ro", "3.510.000", "int", "trei milioane cinci sute zece mii de euro", "big"), [])
        self.assertEqual(errs("pl", "2003", "int", "dwa tysiące trzeciego", "year", spoken="od dwa tysiące trzeciego"), [])


class Generators(unittest.TestCase):
    """The fixed candidate machinery and planners never produce the slips (red on base: each did)."""

    def test_clock_hours(self):
        self.assertEqual(cands("às 1h05 hoje", "pt")[0], "uma e cinco")  # table + code: "um e cinco" before
        self.assertFalse([c for c in cands("às 2:30 pm hoje", "pt") if c.split()[0] == "dois"])
        self.assertFalse([c for c in cands("às 21:15 hoje", "pt") if c.startswith("vinte e um ")])
        self.assertEqual(cands("à 21h05 demain", "fr")[0], "vingt et une heures cinq")
        self.assertFalse([c for c in cands("à 16h15 demain", "fr") if G.FR_ET_BAD.search(c)])
        am = cands("Rendez-vous à 12:45 a. m. demain.", "fr")
        self.assertTrue(am and not [c for c in am if c.startswith("douze")], am)
        self.assertIn("minuit quarante cinq", am)
        self.assertFalse([c for c in cands("a las 21:40 h hoy", "es") if c.startswith("veintiuno")])
        self.assertFalse([c for c in cands("alle 16:15 oggi", "it") if " ore" in c])
        self.assertFalse([c for c in cands("alle 0:55 oggi", "it") if c.startswith("dodici")])

    def test_prices(self):
        for text, lang, bad in [("Cuesta $ 29,10 hoy.", "es", "céntimo"), ("Custa R$ 91,99 hoje.", "pt", "cêntimo"),
                                ("Custa US$ 1,55 hoje.", "pt", "cêntimo")]:
            self.assertFalse([c for c in cands(text, lang) if bad in c], text)
        self.assertTrue([c for c in cands("Custa 12,50 € hoje.", "pt") if "cêntimos" in c])  # the euro keeps them
        for text, lang in [("Es kostet 48,10 € heute.", "de"), ("Het kost € 83,10 vandaag.", "nl")]:
            self.assertEqual([c for c in cands(text, lang) if not G.UNIT_WORDS[lang].search(c)], [], text)

    def test_fractions_and_spelling(self):
        self.assertNotIn("vijf en één half", cands("Het is 5 1/2 kilo.", "nl"))
        self.assertIn("vijf en een half", cands("Het is 5 1/2 kilo.", "nl"))
        self.assertTrue(X.HALF_BAD.search("zes en één half"))
        self.assertEqual(cands("il 26º posto", "it")[0], "ventiseiesimo")
        self.assertEqual(cands("Nel 2003 siamo andati.", "it")[0], "duemilatré")
        self.assertEqual(cands("Costa $ 1003 oggi.", "it")[0], "milletré dollari")
        rng = random.Random(0)
        for lang in X.LANGS:
            for _ in range(200):
                self.assertNotIn("½", X.g_frac(lang, rng, {}, 0, True, None))

    def test_date_ordinal_reading_ok(self):
        sp = M.spans("El contrato se firmó el 1º de octubre de 2020.", "es")[0]
        ok = [c for c in sp.cands if X.reading_ok("es", "QTY", "date_txt", None, sp, c, 0, "el 1º de octubre de 2020")]
        self.assertEqual(ok, ["primero"])

    def test_plans_have_no_slips(self):
        for lang in X.LANGS:
            rows = X.plan(lang, 400, 11, classes=("date_txt", "clock12", "clock24", "price", "frac", "count"))
            self.assertTrue(rows, lang)
            bad = [(r["text"], G.row_errors(lang, r)) for r in rows if G.row_errors(lang, r)]
            self.assertEqual(bad, [], lang)

    def test_llm_rows_checked(self):
        # red-on-base: check_row only checked values, so "patruzeci și nouă lei" / "pięćdziesiątej piątej" were voiced
        import ml_tts_multi as B

        def row(w, s, frag, cls):
            return {"text_written": w, "text": s, "check": "values", "written": frag, "cls": cls}

        bad = [("ro", row("Biletul costă 49,50 lei pentru adulți.", "Biletul costă patruzeci și nouă lei și cincizeci de "
                          "bani pentru adulți.", "49,50 lei", "price"), "ro_de_linker"),
               ("pl", row("Spotkanie o 18.55 w sali.", "Spotkanie o osiemnastej pięćdziesiątej piątej w sali.", "18.55",
                          "clock"), "pl_clock_minute")]
        for lang, r, code in bad:
            ok, why = B.check_row(dict(r), lang)
            self.assertFalse(ok)
            self.assertIn(f"grammar:{code}", why)
        good = [("ro", row("Biletul costă 49,50 lei pentru adulți.", "Biletul costă patruzeci și nouă de lei și cincizeci "
                           "de bani pentru adulți.", "49,50 lei", "price")),
                ("pl", row("Spotkanie o 18.55 w sali.", "Spotkanie o osiemnastej pięćdziesiąt pięć w sali.", "18.55",
                           "clock"))]
        for lang, r in good:
            self.assertEqual(B.check_row(dict(r), lang), (True, []))

    def test_slav_price_after_cost_verb(self):
        import ml_tts_ru as R
        import ml_tts_uk as U

        for mod, verb in ((U, "коштує"), (R, "стоит")):
            rng = random.Random(5)
            seen = 0
            for i in range(2000):
                try:
                    w, rd, _ = mod.g_price(rng, i, None)
                except M.Missing:
                    continue
                if verb in w and ("одна" in rd[0].split() or "одну" in rd[0].split()):
                    self.assertNotIn("одна", rd[0].split(), (w, rd))
                    seen += 1
            self.assertGreater(seen, 0, verb)


class Build(unittest.TestCase):
    def test_grammar_rejects_leave_the_mix(self):
        # red-on-base: build4 never read grammar_reject_ids.txt, so a row speaking an old slip trained
        vol = _fixture(n_win=0)
        tts, _ = _tts_fixture(vol)
        open(f"{vol}/ml/p3v2/de/{MB.GRAMMAR_REJECTS}", "w").write("# list\nnot_a_row\n")
        open(f"{vol}/ml/p3v2/de/grammar_reject_ids_2.txt", "w").write(f"{tts[0]['id']}\n")  # a later scan's list
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        self.assertNotIn(tts[0]["id"], {x["id"] for x in rj(f"{vol}/mix4/tts_de.jsonl")})
        self.assertNotIn("de_x0", {w["id"] for w in rj(f"{vol}/mix4/win_de.jsonl")})  # its window leaves too
        self.assertEqual(st["grammar_rejected"]["de"], 1)

    def test_scan_never_rewrites_a_list(self):
        import tempfile

        import grammar_scan as GS

        with tempfile.TemporaryDirectory() as d:
            res = {"ids": {"pl": {"a": ["x"], "b": ["y"]}}}
            p1 = GS.write_lists(res, d)["pl"]
            self.assertTrue(p1.endswith("/grammar_reject_ids.txt"))
            res["ids"]["pl"]["c"] = ["z"]
            p2 = GS.write_lists(res, d)["pl"]
            self.assertTrue(p2.endswith("/grammar_reject_ids_2.txt"))
            self.assertEqual([l.strip() for l in open(p2) if not l.startswith("#")], ["c"])
            self.assertEqual(GS.write_lists(res, d), {})  # nothing new: nothing written

    def test_ft5_sets_wired(self):
        # red-on-base: ttsFT5 / 5c / 5d were not read by build4
        tags = {t for t, _ in MB.TTS_SETS}
        for t in ("ttsFT5", "ttsFT5c", "ttsFT5d", "ttsFT6c", "ttsFT6r", "ttsFT6rd"):
            self.assertIn(t, tags)
            self.assertIn(t, MB.TTS_TARGETED)
            self.assertIn((t, f"{t}_judge"), MB.TTS_SETS)

    def test_rare_class_floor(self):
        # red-on-base: a class real speech never has got no synthetic rows at all (share 1.0 > 0.30)
        real = [{"id": f"r{i}", "text": "es waren fünfzig Prozent", "spans": [{"kind": "pct", "text": "50 %"}]}
                for i in range(40)]
        tts = [{"id": f"t{i}", "text": "AT und T", "src": "tts", "_targeted": False, "duration": 6.5,
                "spans": [{"kind": "amp", "text": "AT&T"}]} for i in range(120)]
        took = MB.select_synthetic(tts, real, "de", {"pct": 1.0})
        self.assertEqual(len(took), int(MB.TTS_FLOOR_H * 3600 / 6.5))  # 83 rows = 0.15 h
        chk = MB.post_build_check(real, took, "de")
        self.assertEqual(chk["synthetic_over"], {})
        self.assertEqual(set(chk["waived_synthetic_floor"]), {"amp"})
        self.assertEqual(chk["waived_synthetic_floor"]["amp"]["real"], 0)
        self.assertFalse([f for f in chk["fail"] if "amp" in f])
        self.assertEqual(MB.trim_synthetic(took, real, "de")[1], 0)  # the floor holds after the later caps too
        # a class WITH real speech keeps the share rule
        self.assertIn("pct", MB.post_build_check(
            [{"id": "r", "text": "x", "spans": [{"kind": "pct", "text": "45 %"}]} for _ in range(400)],
            [{"id": f"y{i}", "text": "x", "duration": 1.0, "spans": [{"kind": "pct", "text": "46 %"}]} for i in range(200)],
            "de")["synthetic_over"])


class HeldoutPersona(unittest.TestCase):
    def test_persona_across_locales(self):
        # red-on-base: the filter compared exact names, so es-ES-...-Erinome trained while es-US-...-Erinome was held out
        self.assertTrue(MB.is_heldout_voice("es", "es-ES-Chirp3-HD-Erinome"))
        self.assertTrue(MB.is_heldout_voice("es", MB.HELDOUT_VOICES["es"]))
        self.assertFalse(MB.is_heldout_voice("es", "es-ES-Chirp3-HD-Puck"))
        self.assertFalse(MB.is_heldout_voice("bg", "bg-BG-Chirp3-HD-Erinome"))
        self.assertTrue(MB.is_heldout_voice("fr", "fr-FR-Chirp3-HD-Sadaltager"))

    def test_every_build_language_has_a_heldout_voice_or_is_exempt(self):
        # red-on-base: fr had no held-out voice and nothing checked it
        MB.check_heldout_voices(("de", "es", "fr", "it", "nl", "pt", "ru", "uk", "pl", "ro", "sv", "cs", "hu", "fi", "sk",
                                 "bg", "hr", "en"))
        with self.assertRaises(AssertionError):
            MB.check_heldout_voices(("de", "et"))

    def test_no_silent_fallback_to_the_heldout_voice(self):
        # red-on-base: an emptied train pool fell back to every voice ("or train_v")
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("ml_tts_cloud.py", "ml_tts_multi.py"):
            src = open(os.path.join(here, f)).read()
            self.assertIn("is_heldout_voice(lang", src)
            self.assertNotIn("HELDOUT_VOICES.get(lang)] or train_v", src)
            self.assertNotIn("HELDOUT_VOICES.get(lang)] or train_v", src)
            self.assertIn("no train voice left once the held-out voice is out", src)


class FillToBar(unittest.TestCase):
    """Rule (2026-10-17): synthetic may fill a thin class up to its bar; rich classes keep the share caps."""

    def test_thin_class_fills_to_bar(self):
        # red-on-base: nl range (real 60 of a 150 bar) stopped at 30% synthetic (25 spans)
        real = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "range", "text": "20-25"}]} for i in range(60)]
        tts = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True, "duration": 5.0,
                "spans": [{"kind": "range", "text": f"{30 + i}-{40 + i}"}]} for i in range(200)]
        took = MB.select_synthetic(tts, real, "nl", {"range": 9.0})
        self.assertEqual(len(took), 90 + CC.FILL_OVERSHOOT)  # 60 real + 90 synthetic = the 150 bar (+ slack)
        chk = MB.post_build_check(real, took, "nl")
        self.assertEqual(chk["synthetic_over"], {})
        self.assertEqual(chk["filled_to_bar"]["range"], {"synthetic": 90 + CC.FILL_OVERSHOOT, "real": 60, "bar": 150})
        self.assertEqual(MB.trim_synthetic(took, real, "nl")[1], 0)
        self.assertEqual(len(MB.select_synthetic(tts, real, "bg", {"range": 9.0})), 25)  # bg keeps the caps

    def test_rich_class_keeps_the_cap(self):
        # red-on-base (control on the share arithmetic): a class whose real spans meet the bar stays at 30%
        real = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "pct", "text": "45 %"}]} for i in range(400)]
        tts = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True,
                "spans": [{"kind": "pct", "text": f"{i % 90} %"}]} for i in range(400)]
        self.assertEqual(len(MB.select_synthetic(tts, real, "de", {"pct": 9.0})), 171)  # 171 / 571 = 0.2995

    def test_thin_values_fill_to_their_floor(self):
        # red-on-base: sv years (real 380 spans, all 1980: the class bar met) took 30% synthetic whatever the values
        real = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "int", "text": "1980"}]} for i in range(380)]
        tts = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True,
                "spans": [{"kind": "int", "text": str(1990 + i % 41)}]} for i in range(41 * 12)]
        took = MB.select_synthetic(tts, real, "sv", {"int": 9.0})
        vals = Counter(int(x["spans"][0]["text"]) for x in took)
        self.assertTrue(all(vals[v] >= 8 for v in range(1990, 2031)), vals)  # the moderate floor of 8 per value
        self.assertGreater(len(took), 162)  # more than the 30% share alone (162)

    def test_real_rows_never_displaced(self):
        # red-on-base: a thin class stopped at the 0.15 h floor (270 rows) instead of filling to its bar; real rows stay
        vol = _fixture(n_win=0)
        from test_census_common import wj
        tts = [{"id": f"tts1_de_{k}", "text": "fünfzig Prozent da", "duration": 2.0, "src": "tts", "origin": "tts",
                "audio_filepath": "t.wav", "split": "train", "num_chars": [[0, 15]],
                "spans": [{"kind": "pct", "text": f"{k % 99} %", "by": "tts"}]} for k in range(400)]
        wj(vol, "ml/tts1/de/rows.jsonl", tts)
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        before = {x["id"] for x in rj(f"{vol}/ml/p3v2/de/rows.jsonl")}
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        p3 = {x["id"] for x in rj(f"{vol}/mix4/p3_de.jsonl")}
        self.assertEqual(len(rj(f"{vol}/mix4/tts_de.jsonl")), 292 + CC.FILL_OVERSHOOT)  # 8 real + 292 synthetic = the 300 pct bar (+ slack)
        self.assertEqual(len(p3), 8)  # every real row kept, with the synthetic rows beside them
        self.assertTrue(p3 <= before)


    def test_gate_shows_filled_to_bar(self):
        # red-on-base: the data gate failed a thin class's synthetic share however short of its bar the class was
        import importlib.util
        import os
        import tempfile

        from test_census_common import wj
        spec = importlib.util.spec_from_file_location("data_gate", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "eval", "quality", "data_gate.py"))
        DG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DG)
        d = tempfile.mkdtemp()
        wj(d, "p3_nl.jsonl", [{"id": f"r{i}", "text": "x", "duration": 3.0, "spans": [{"kind": "range", "text": "20-25"}]}
                              for i in range(60)])
        wj(d, "tts_nl.jsonl", [{"id": f"t{i}", "text": "x", "duration": 3.0, "spans": [{"kind": "range", "text": "1-9"}]}
                               for i in range(90)])
        res = DG.gate_lang(d, "nl")
        c = [x for x in res["checks"] if x["check"].startswith("range synthetic share")]
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["pass"], c)
        self.assertIn("filled to bar (synthetic 90, real 60)", str(c[0]["got"]))
        wj(d, "tts_nl.jsonl", [{"id": f"t{i}", "text": "x", "duration": 3.0, "spans": [{"kind": "range", "text": "1-9"}]}
                               for i in range(91 + CC.FILL_OVERSHOOT)])
        c = [x for x in DG.gate_lang(d, "nl")["checks"] if x["check"].startswith("range synthetic share")]
        self.assertFalse(c[0]["pass"])  # one over the bar + its slack


class LongGenerators(unittest.TestCase):
    # (test_long_sets_degraded_more: behavioural now, test_fill_review.LongSets)
    def test_counts_read_as_one_number(self):
        # red-on-base: "191362 klanten" could be read "één negen één drie tweeënzestig" (digit chunks)
        sp = M.spans("Het bureau noemt 191362 klanten.", "nl")[0]
        ok = [c for c in sp.cands if X.reading_ok("nl", "QTY", "count", None, sp, c, 0, "Het bureau noemt 191362 klanten.")]
        self.assertEqual(ok, ["honderdeenennegentigduizenddriehonderdtweeënzestig"])

    def test_multi_big_fragment(self):
        # red-on-base: ml_tts_multi had no long-number class
        import ml_tts_multi as B

        rng = random.Random(3)
        vals = [int(B.fragment("big", "pl", rng)["written"].replace(" ", "")) for _ in range(400)]
        self.assertFalse([v for v in vals if 1100 <= v <= 2099])
        self.assertGreater(max(vals), 10 ** 9)
        self.assertGreater(len(set(vals)), 350)
        self.assertIn("never first", B.HINTS["big"])

    def test_years_two_or_three_per_sentence(self):
        # red-on-base: no multi-year class (one year per voiced sentence)
        import ml_tts_multi as B
        import ml_tts_ru as R
        import ml_tts_uk as U

        rng = random.Random(4)
        fr = [B.fragment("years", "pl", rng) for _ in range(50)]
        self.assertTrue(all(2 <= len(f["years"]) <= 3 and f["check"] == "years" for f in fr))
        r = {"text_written": "W latach 2015 i 2031 budżet rósł.", "text": "W latach dwa tysiące piętnastym i dwa tysiące "
             "trzydziestym pierwszym budżet rósł.", "check": "years", "years": [2015, 2032], "cls": "years", "written": ""}
        self.assertIn("fragment_missing", B.check_row(dict(r), "pl")[1])
        for M_, l in ((R, "ru"), (U, "uk")):
            rows = M_.plan(6, 5, {}, ("years3",), "x")
            self.assertEqual(len(rows), 6)
            for row in rows:
                cl = [CC.gate_class(row, sp, l) for sp in row["spans"]]
                self.assertTrue(2 <= len(cl) <= 3 and set(cl) == {"year"}, (row["text"], cl))

    def test_ru_phone_is_a_phone_span(self):
        # red-on-base: ru phones were "chain" spans (no census phone class)
        import ml_tts_ru as R

        rows = R.plan(6, 3, {}, ("phone_sp",), "x")
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertEqual([CC.gate_class(r, sp, "ru") for sp in r["spans"]], ["phone"])


class HeldoutVoice(unittest.TestCase):
    def test_heldout_voice_never_trains(self):
        # red-on-base: every voice could train; no language had a voice kept out of training
        vol = _fixture(n_win=0)
        tts, _ = _tts_fixture(vol)
        hv = MB.HELDOUT_VOICES["de"]
        rows = [dict(tts[0], voice="de-DE-Chirp3-HD-Puck"), dict(tts[0], id="tts1_de_hv", voice=hv),
                dict(tts[0], id="tts1_de_hvdev", voice=hv, split="dev")]
        from test_census_common import wj
        wj(vol, "ml/tts1/de/rows.jsonl", rows)
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        train = rj(f"{vol}/mix4/tts_de.jsonl")
        self.assertFalse([x for x in train if x.get("voice") == hv])
        self.assertIn("tts1_de_hvdev", {x["id"] for x in rj(f"{vol}/mix4/tts_dev_de.jsonl")})
        self.assertEqual(st["heldout_voice_train_dropped"]["de"], 1)
        for t in ("ttsHOw", "ttsHOr", "ttsHOm", "ttsYRm", "ttsYRr", "ttsYRs", "ttsYRn", "ttsCNm", "ttsLTw", "ttsLTr",
                  "ttsHOx"):  # the held-out, year, count and top-off sets are read
            self.assertIn((t, f"{t}_judge"), MB.TTS_SETS)
        self.assertNotIn("bg", MB.HELDOUT_VOICES)
        for l in ("nl", "it", "pt", "de", "es", "fr", "ru", "uk", "pl", "ro", "sv", "cs", "hu", "fi", "sk", "hr"):
            self.assertIn(l, MB.HELDOUT_VOICES)


class Census(unittest.TestCase):
    def test_sentence_final_cardinal_is_not_an_ordinal(self):
        # red-on-base: "Punkt 10." read "zehn" and "2008." read "zweitausendacht" counted as ordinals
        self.assertEqual(CC.gate_class_value({"text": "um Punkt zehn."}, {"kind": "ord", "text": "10.",
                                                                          "reading": "zehn."}, "de"), ("int_small", 10))
        self.assertEqual(CC.gate_class_value({"text": "Das war zweitausendacht."},
                                             {"kind": "ord", "text": "2008.", "reading": "zweitausendacht."}, "de"),
                         ("year", 2008))
        self.assertEqual(CC.gate_class({"text": "Het is precies tien."}, {"kind": "ord", "text": "10.",
                                                                           "reading": "tien"}, "nl"), "int_small")
        # real ordinals stay ordinals; a span with no reading is unchanged
        self.assertEqual(CC.gate_class({"text": "Er wurde Zehnter."}, {"kind": "ord", "text": "10.",
                                                                        "reading": "Zehnter"}, "de"), "ord")
        self.assertEqual(CC.gate_class({"text": "Hij werd tiende."}, {"kind": "ord", "text": "10.",
                                                                       "reading": "tiende"}, "nl"), "ord")
        self.assertEqual(CC.year_of({"text": "x"}, {"kind": "ord", "text": "2015."}, "de"), None)


if __name__ == "__main__":
    unittest.main()
