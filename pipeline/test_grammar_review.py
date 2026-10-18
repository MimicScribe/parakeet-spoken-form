"""Grammar rule and candidate fixes from the 2026-10-17 review: right readings the rules rejected, slips they missed,
and 12-hour readings offered for 0:xx clock times. Every test is red on the commit before them unless marked.
python3 test_grammar_review.py"""
import unittest

import grammar_rules as G
import ml_cands as M
import ro_gender as R


def errs(lang, written, reading, kind="clock", cls=""):
    return G.span_errors(lang, {"text": written, "kind": kind, "reading": reading}, {"cls": cls})


class FalsePositives(unittest.TestCase):
    def test_right_readings_pass(self):
        # red-on-base: each of these right readings was rejected
        for lang, written, reading in (
                ("it", "16:30", "alle ore sedici e trenta"),
                ("it", "9:15", "dalle ore nove e un quarto"),
                ("es", "2:45", "un cuarto para las tres"),
                ("es", "2:59", "un minuto para las tres"),
                ("pt", "2:45", "um quarto para as três"),
                ("pt", "2:58", "dois minutos para as três"),
                ("fr", "3:15", "trois heures et un quart")):
            self.assertEqual(errs(lang, written, reading), [], (lang, reading))

    def test_ro_decimal_scan_stops_at_the_next_number(self):
        # red-on-base: "și" let the decimal scan run into the next number ("... și două mii") -> ['doi']
        self.assertEqual(R.violations("doi virgulă cinci și două mii"), [])

    def test_years_class_name(self):
        # red-on-base: ml_tts_multi's "years" rows were not exempt from the Polish genitive rule as "year" rows are
        row = {"cls": "years", "text": "od dwa tysiące piętnaście"}
        sp = {"text": "2015", "kind": "int", "reading": "dwa tysiące piętnaście"}
        self.assertEqual(G.span_errors("pl", sp, dict(row, text="Od dwa tysiące piętnaście")), [])
        self.assertIn("pl_gen_case", G.span_errors("pl", sp, {"cls": "count", "text": "Od dwa tysiące piętnaście"}))


class Misses(unittest.TestCase):
    def test_slips_caught(self):
        # red-on-base: each slip passed
        self.assertIn("hour_noun_number", errs("fr", "21:00", "vingt et une heure"))
        self.assertIn("hour_noun_number", errs("pt", "21:00", "vinte e uma hora"))
        self.assertIn("hour_noun_number", errs("fr", "0:03", "zéro heures trois"))
        self.assertEqual(R.violations("douăzeci și doi", "", " de minute"), ["doi"])

    def test_rules_still_fire(self):
        # red-on-base: control (the slips the rules were written for still fail; the right forms still pass)
        self.assertIn("it_clock_ore", errs("it", "16:15", "sedici ore e un quarto"))
        self.assertIn("es_clock_hour_masc", errs("es", "1:05", "uno y cinco"))
        self.assertIn("es_clock_hour_masc", errs("es", "21:00", "veintiún horas"))
        self.assertIn("pt_clock_hour_masc", errs("pt", "2:05", "dois e cinco"))
        self.assertIn("fr_clock_et", errs("fr", "16:15", "seize heures et quinze"))
        for lang, w, rd in (("fr", "21:00", "vingt et une heures"), ("fr", "0:03", "zéro heure trois"),
                            ("fr", "1:00", "une heure"), ("es", "0:23", "cero horas y veintitrés"),
                            ("pt", "0:20", "zero hora e vinte"), ("fr", "3:15", "trois heures et quart")):
            self.assertEqual(errs(lang, w, rd), [], rd)
        self.assertEqual(R.violations("doi virgulă douăzeci și cinci miliarde"), ["doi"])
        self.assertEqual(R.violations("douăzeci și doi", "", " minute"), ["doi"])
        self.assertEqual(R.violations("douăzeci și doi", "", " de euro"), [])


class MidnightCandidates(unittest.TestCase):
    def test_no_twelve_hour_reading_for_zero(self):
        # red-on-base: 0:xx was offered the 12 hour ("zwölf Uhr dreißig" is 12:30) in every language but it
        for text, lang, bad in (("um 00:30 Uhr", "de", "zwölf"), ("à 0h03", "fr", "douze"),
                                ("a las 00:23", "es", "doce"), ("om 0.11 uur", "nl", "twaalf")):
            c = M.spans(text, lang)[0].cands
            self.assertTrue(c, text)
            self.assertFalse([x for x in c if x.split()[0] == bad], (text, c))
        self.assertNotIn("zéro heures trois", M.spans("à 0h03", "fr")[0].cands)
        self.assertIn("zéro heure trois", M.spans("à 0h03", "fr")[0].cands)

    def test_afternoon_keeps_the_twelve_hour_reading(self):
        # red-on-base: control (13:00-23:59 keep the 12-hour reading; 12:xx stays twelve)
        self.assertIn("eins Uhr dreißig", M.spans("um 13:30 Uhr", "de")[0].cands)
        self.assertIn("douze heures trente", M.spans("à 12h30", "fr")[0].cands)


if __name__ == "__main__":
    unittest.main()
