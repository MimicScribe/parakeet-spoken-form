"""number_outcomes / score_heldout fixes of 2026-10-18 (no Modal, no API): python3 test_scorer_fixes.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N
import score_heldout as S


def vals(text, lang):
    return [sorted(str(v) for v in x.vals) for x in N.extract_numbers(text, lang, keep_weak=True)]


def hours(text, lang):
    return vals(text, lang)


class PorCien(unittest.TestCase):
    def test_price_not_percent(self):
        self.assertEqual(vals("lo vendió por cien euros", "es"), [["100"]])
        self.assertEqual(vals("por cien mil euros", "es"), [["100000"]])

    def test_percent_after_a_number(self):
        self.assertEqual(vals("cien por cien seguro", "es"), [["100"]])
        self.assertEqual(vals("el tres por cien", "es"), [["3"]])
        self.assertEqual(vals("el tres por ciento", "es"), [["3"]])
        self.assertEqual([o["outcome"] for o in N.number_outcomes("El 3 %", "el tres por cien", "es")], ["WORDS_RIGHT_VALUE"])


class Clocks(unittest.TestCase):
    def test_split_tens_in_minutes(self):
        self.assertEqual(vals("às onze menos vinte e cinco", "pt"), [["10", "22"], ["35"]])
        self.assertEqual(vals("a las once menos veinte y cinco", "es"), [["10", "22"], ["35"]])

    def test_noon_words_keep_their_minutes(self):
        self.assertEqual(vals("mezzogiorno e un quarto", "it"), [["0", "12"], ["15"]])
        self.assertEqual(vals("a mezzogiorno e mezza", "it"), [["0", "12"], ["30"]])
        self.assertEqual(vals("à midi et demi", "fr"), [["0", "12"], ["30"]])
        self.assertEqual(vals("vers midi moins le quart", "fr"), [["11", "23"], ["45"]])
        self.assertEqual(vals("ao meio-dia e meia", "pt"), [["0", "12"], ["30"]])

    def test_italian_all_una(self):
        self.assertEqual(vals("all'una e mezza", "it"), [["1", "13"], ["30"]])

    def test_french_duration_is_not_a_clock(self):
        self.assertEqual(vals("il dort trois heures et demie", "fr"), [["3.5"]])
        self.assertEqual(vals("à trois heures et demie", "fr"), [["15", "3"], ["30"]])
        o = N.number_outcomes("Il dort 3,5 heures", "Il dort trois heures et demie", "fr")
        self.assertEqual([x["outcome"] for x in o], ["WORDS_RIGHT_VALUE"])

    def test_german_viertel_vor_einem_jahr(self):
        self.assertEqual(vals("Viertel vor einem Jahr", "de"), [["1"]])
        self.assertEqual(vals("um Viertel vor vier", "de"), [["15", "3"], ["45"]])

    def test_regions_are_no_numbers(self):
        self.assertEqual(vals("il Mezzogiorno è povero", "it"), [])
        self.assertEqual(vals("le Midi de la France", "fr"), [])
        self.assertEqual(vals("l'après-midi", "fr"), [])
        self.assertEqual(vals("a mezzogiorno", "it"), [["0", "12"]])
        self.assertEqual(vals("à midi", "fr"), [["0", "12"]])

    def test_es_con_counts(self):
        self.assertEqual(vals("dos con tres camas", "es"), [["2"], ["3"]])
        self.assertEqual(vals("somos tres con dos niños", "es"), [["3"], ["2"]])
        self.assertEqual(vals("mide dos con tres metros", "es"), [["2.3"]])
        self.assertEqual(vals("tres con cinco", "es"), [["3.5"]])

    def test_pt_meia(self):
        self.assertEqual(vals("meia trinta e dois", "pt"), [["6"], ["32"]])
        self.assertEqual(vals("uma meia hora", "pt"), [["1"]])
        self.assertEqual(vals("nove seis meia", "pt"), [["9"], ["6"], ["6"]])


class BarePrice(unittest.TestCase):
    cap, lab = "Das kostet 2,50 € heute.", "Das kostet zwei Euro fünfzig heute."

    def score(self, hyp):
        sp = S.M.spans_regional(self.cap, "de", S.READINGS)[0]
        a = self.lab.index("zwei")
        b = self.lab.index("fünfzig") + 7
        caps = {"r1": {"caption": self.cap, "text": self.lab, "spans": [{"text": sp.text, "kind": "cur"}], "num_chars": [[a, b]]}}
        return S.score_spans([{"id": "r1", "hyp": hyp, "ref": self.lab}], "de", caps)[0]["all"]

    def test_unit_drop_is_its_own_outcome(self):
        self.assertEqual(self.score("Das kostet zwei fünfzig heute."), {"BARE_PRICE": 1})

    def test_unit_kept_is_right(self):
        self.assertEqual(self.score("Das kostet zwei Euro fünfzig heute."), {"WORDS_RIGHT_VALUE": 1})
        self.assertEqual(self.score("Das kostet zwei fünfzig Euro heute."), {"WORDS_RIGHT_VALUE": 1})


class ItalianElision(unittest.TestCase):
    def test_cento_before_a_vowel(self):
        self.assertEqual(S._elide("centoottantotto"), "centottantotto")
        self.assertEqual(S._elide("centouno"), S._elide("centuno"))
        self.assertEqual(S._elide("centootto"), S._elide("centotto"))
        self.assertEqual(S._elide("ventiuno"), "ventuno")


if __name__ == "__main__":
    unittest.main()
