"""Known-answer tests of the adversarial scorer (adv_b_score): hyp = label is exact for every construction of every language; hyp = caption
(digits) and a one-word corruption are not; phone value is read from any style; control rows score the same way."""
import json
import re
import unittest

import adv_b as A
import adv_b_score as S

ROWS = {}


def rows(lang):
    if lang not in ROWS:
        split = A.freeze(lang)
        ROWS[lang] = A.test_plan(lang, split, per_cell=2)
    return ROWS[lang]


class KnownAnswers(unittest.TestCase):
    def test_label_is_exact_and_phone_value_is_read(self):
        for lang in A.LANGS:
            t, c = rows(lang)
            for r in t + c:
                with self.subTest(r["id"]):
                    x = S.score_row(r, r["text"])
                    self.assertTrue(x["edge_exact"], (r["construction"], r["text"], x))
                    self.assertFalse(x["digits"])
                    if r["family"] == "phone":
                        self.assertTrue(x["value_ok"], (r["construction"], x))

    def test_digits_written_are_an_error_and_flagged(self):
        t, c = rows("en")
        for r in t[:40]:
            x = S.score_row(r, r["caption"])
            self.assertFalse(x["edge_exact"], r["caption"])
            self.assertTrue(x["digits"])

    def test_one_wrong_number_word_is_caught(self):
        for lang in ("en", "es", "fr"):
            t, _ = rows(lang)
            n = 0
            for r in t:
                if r["family"] != "phone":
                    continue
                a, b = r["edge_spans"][0]
                words = r["text"][a:b].split()
                i = max(range(len(words)), key=lambda k: len(words[k]))
                bad = r["text"][:a] + " ".join(words[:i] + ["banana"] + words[i + 1:]) + r["text"][b:]
                x = S.score_row(r, bad)
                self.assertFalse(x["edge_exact"])
                n += 1
                if n > 15:
                    break

    def test_style_folds(self):
        t, _ = rows("en")
        r = next(x for x in t if x["construction"] == "en.phone.us_double" or "double" in x["text"])
        a, b = r["edge_spans"][0]
        phrase = r["text"][a:b]
        alt = phrase.replace("double ", "").replace("treble ", "").replace("triple ", "")      # the repeat word dropped is NOT the same words
        self.assertFalse(S.score_row(r, r["text"][:a] + alt + r["text"][b:])["edge_exact"] and phrase != alt)
        # oh / zero fold
        self.assertEqual(S.canon(["oh", "five"]), S.canon(["zero", "five"]))
        self.assertEqual(S.canon(["double", "five"]), S.canon(["five", "five"]))
        self.assertEqual(S.phone_digits("eight oh eight nine three six twenty three twenty three".split(), "en"), "8089362323")
        self.assertEqual(S.phone_digits("plus forty four double seven one three".split(), "en"), "447713")
        self.assertEqual(S.phone_digits("seiscientos doce treinta y cuatro cincuenta y seis setenta y ocho".split(), "es"), "612345678")
        self.assertEqual(S.phone_digits("zéro six douze trente quatre cinquante six soixante dix huit".split(), "fr"), "0612345678")

    def test_report_pairs_control(self):
        t, c = rows("en")
        hyps = {r["id"]: r["text"] for r in t + c}
        rep = S.report(t, c, hyps)
        self.assertEqual(rep["adversarial"]["all"]["edge_exact"], 100.0)
        self.assertEqual(rep["control"]["all"]["edge_exact"], 100.0)
        self.assertGreater(rep["paired"]["n"], 20)


if __name__ == "__main__":
    unittest.main()
