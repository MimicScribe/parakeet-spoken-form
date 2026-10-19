"""symscore.py (the symmetric value / unit / decimal scorer) and its wiring into classread.py and score_heldout.py.
    python3 -m unittest test_symscore        (stdlib; imports pipeline/ml_cands; the classread table test needs numpy)
Each case runs the real path: caption spans (ml_cands) -> classread.score_rows -> symscore, on a hand-built held-out row."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import classread as CR
import ml_cands as M
import score_heldout as SH
import symscore as SS


def case(lang, caption, label, seg, hyp, sym=True):
    """The records score_rows returns for one hand-built row. seg: the label words of each caption span, in order."""
    segs = [seg] if isinstance(seg, str) else list(seg)
    sps = [s for s in M.spans_regional(caption, lang, SH.READINGS) if s.cands]
    assert len(sps) == len(segs), (len(sps), [s.text for s in sps])
    nc, spans, pos = [], [], 0
    for s, g in zip(sps, segs):
        a = label.index(g, pos)
        nc.append([a, a + len(g)])
        pos = a + len(g)
        spans.append({"text": s.text, "kind": s.kind})
    caps = {"r1": {"id": "r1", "caption": caption, "text": label, "spans": spans, "num_chars": nc}}
    return list(CR.score_rows([{"id": "r1", "hyp": hyp, "ref": label}], lang, caps, sym=sym).values())


def one(*a, **k):
    (rec,) = case(*a, **k)
    return rec


PT = ("pt", "Existem cerca de 400.000 casos.", "existem cerca de quatrocentos mil casos", "quatrocentos mil")
FR = ("fr", "Environ 28% des coûts.", "environ vingt-huit pour cent des coûts", "vingt-huit pour cent")
DE_Y = ("de", "Im Jahr 2018 war es.", "im Jahr zweitausendachtzehn war es", "zweitausendachtzehn")
ES_C = ("es", "Cuesta 2,80 €.", "cuesta dos euros con ochenta", "dos euros con ochenta")
DE_D = ("de", "Es sind 3,5 Prozent.", "es sind drei Komma fünf Prozent", "drei Komma fünf Prozent")


class DigitsAndWordsAreGradedAlike(unittest.TestCase):
    def test_digits_with_a_scale_word_are_the_value(self):
        r = one(*PT, "existem cerca de 400 mil casos")
        self.assertEqual(r["out"], "WRONG_VALUE")  # the positional scorer: the digit groups are "400" and not "400000"
        self.assertTrue(r["value_ok"])
        self.assertEqual(r["sform"], "digits")
        self.assertTrue(one(*PT, "existem cerca de quatrocentos mil casos")["value_ok"])
        self.assertTrue(one(*PT, "existem cerca de 400.000 casos")["value_ok"])
        self.assertFalse(one(*PT, "existem cerca de 400 casos")["value_ok"])

    def test_roman_numeral_and_letter_digit_code(self):
        r = one("it", "Il XIII secolo fu.", "il tredicesimo secolo fu", "tredicesimo", "il XIII secolo fu")
        self.assertEqual((r["out"], r["value_ok"]), ("WRONG_VALUE", True))
        g = one("es", "El G20 no pasó.", "el G veinte no pasó", "G veinte", "el G veinte no pasó")
        self.assertEqual((g["out"], g["value_ok"], g["cls"]), ("WRONG_VALUE", True, "alnum"))

    def test_words_with_the_symbol_glued_on(self):
        r = one(*FR, "environ vingt-huit% des coûts")
        self.assertEqual((r["out"], r["value_ok"], r["sform"]), ("WRONG_VALUE", True, "words"))
        self.assertTrue(r["unit_spoken"] and r["unit_present"])

    def test_a_bare_year_tail_is_not_the_year(self):
        r = one(*DE_Y, "im Jahr tausend achtzehn war es")
        self.assertEqual(r["out"], "WORDS_RIGHT_VALUE")  # the positional scorer credits it (1018)
        self.assertFalse(r["value_ok"])
        for hyp in ("im Jahr zweitausendachtzehn war es", "im Jahr 2018 war es"):
            self.assertTrue(one(*DE_Y, hyp)["value_ok"], hyp)

    def test_a_missing_unit_is_missing_in_either_form(self):
        spoken = lambda h: (lambda r: (r["unit_spoken"], r["unit_present"], r["sform"]))(one(*FR, h))  # noqa: E731
        self.assertEqual(spoken("environ 28 des coûts"), (True, False, "digits"))
        self.assertEqual(spoken("environ vingt-huit des coûts"), (True, False, "words"))
        self.assertEqual(spoken("environ 28% des coûts"), (True, True, "digits"))
        self.assertEqual(spoken("environ vingt-huit pour cent des coûts"), (True, True, "words"))

    def test_a_decimal_is_not_the_integer_without_its_point(self):
        for hyp in ("cuesta 280", "cuesta 28"):
            self.assertFalse(one(*ES_C, hyp)["value_ok"], hyp)
        for hyp in ("cuesta 2,80 €", "cuesta 2 80", "cuesta dos euros con ochenta"):
            self.assertTrue(one(*ES_C, hyp)["value_ok"], hyp)
        self.assertFalse(one(*DE_D, "es sind 35 Prozent")["value_ok"])
        for hyp in ("es sind 3,5 Prozent", "es sind 3.5 Prozent", "es sind drei Komma fünf Prozent"):
            r = one(*DE_D, hyp)
            self.assertTrue(r["value_ok"] and r["dec_present"], hyp)

    def test_a_glued_phone_or_pair_is_still_the_value(self):
        segs = ["nove nove sete", "oito dois", "quatro quatro", "dois seis"]
        for hyp in ("ligue 997824426 hoje", "ligue 997 82 44 26 hoje"):
            recs = case("pt", "Ligue 997 82 44 26 hoje.", "ligue nove nove sete oito dois quatro quatro dois seis hoje", segs, hyp)
            self.assertTrue(all(r["value_ok"] for r in recs), hyp)

    def test_unreadable_language_fails_loudly(self):
        with self.assertRaises(SS.UnscorableLanguage):
            SS.check_lang("xx")
        SS.check_lang("de")


class ClassreadWiring(unittest.TestCase):
    def test_legacy_records_are_the_old_records(self):
        a = one(*PT, "existem cerca de 400 mil casos", sym=False)
        b = one(*PT, "existem cerca de 400 mil casos", sym=True)
        self.assertNotIn("value_ok", a)
        self.assertEqual({k: v for k, v in b.items() if k not in CR.SYM_FIELDS}, a)

    def test_counts_and_selectors(self):
        recs = {("1", "x", 0): one(*FR, "environ 28 des coûts"), ("2", "x", 0): one(*FR, "environ vingt-huit des coûts"),
                ("3", "x", 0): one(*FR, "environ vingt-huit pour cent des coûts")}
        c = CR.sym_counts(recs)["ALL"]
        self.assertEqual((c["n"], c["V"], c["W"]), (3, 3, 2))
        self.assertEqual((c["Uw"], c["Ud"]), ([1, 2], [1, 1]))  # [missing, spoken] among words / digits spans
        rr = list(recs.values())
        self.assertEqual(CR._sel(rr, "Uw"), ([False, True, False], [False, True, True]))
        self.assertEqual(CR._sel(rr, "Ud"), ([True, False, False], [True, False, False]))
        self.assertEqual(CR._sel(rr, "V"), ([True] * 3, [True] * 3))
        self.assertEqual(CR._sel(rr, "W")[0], [False, True, True])
        self.assertEqual(CR.sym_counts({k: {kk: vv for kk, vv in v.items() if kk != "value_ok"} for k, v in recs.items()}), {})

    def test_table_prints_the_symmetric_columns_and_legacy_prints_the_old_ones(self):
        recs = {("1", "x", 0): one(*FR, "environ 28 des coûts")}
        arms = {"a__m": recs, "b__m": dict(recs)}
        keys = list(recs)
        for legacy, header, absent in ((False, "U_w", "unitdrop"), (True, "unitdrop", "U_w")):
            mets = CR.METRICS if legacy else CR.METRICS + CR.SYM_METRICS
            res = {"fr/ALL": {"n": 1, "sym": not legacy, "arms": CR.aggregate(arms, keys, ["a__m"], 20, metrics=mets)}}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                CR.print_table(res, ["a__m"], 1, legacy=legacy)
            self.assertIn(header, buf.getvalue())
            self.assertNotIn(absent, buf.getvalue())


class ScoreHeldoutWiring(unittest.TestCase):
    def run_main(self, *extra):
        row = {"id": "r1", "caption": PT[1], "text": PT[2], "spans": [{"text": "400.000", "kind": "int"}],
               "num_chars": [[PT[2].index(PT[3]), PT[2].index(PT[3]) + len(PT[3])]]}
        hyps = [{"id": "r1", "hyp": "existem cerca de 400 mil casos", "ref": PT[2]}]
        old = (SH.fetch, SH._captions)
        SH.fetch = lambda run, lang, work, set_="heldout": hyps
        SH._captions = lambda lang, work, set_="heldout", mix_dir="mix4": {"r1": row}
        try:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "o.json")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    SH.main(["x", "--langs", "pt", "--work", d, "--json", out, *extra])
                return buf.getvalue(), json.load(open(out))
        finally:
            SH.fetch, SH._captions = old

    def test_default_adds_the_symmetric_block(self):
        text, js = self.run_main()
        self.assertIn("SYM", text)
        self.assertEqual(js["x"]["pt"]["sym"]["ALL"]["V"], 1)  # the old line says wrong 1
        self.assertEqual(js["x"]["pt"]["spans"]["all"]["WRONG_VALUE"], 1)

    def test_legacy_is_the_old_output_only(self):
        text, js = self.run_main("--legacy")
        self.assertNotIn("SYM", text)
        self.assertNotIn("sym", js["x"]["pt"])


if __name__ == "__main__":
    unittest.main()
