"""Scorer / census fixes of 2026-10-18 (no Modal, no API): python3 test_reading_fixes.py
reading_conv (pt "meia", Italian elision), census_common (dotted clocks, fraction / scale words), ml_tts (French "cent")."""
import unittest

import ml_cands as M
import ml_tts as T
import reading_conv as RC
import census_common as CC


def gc(lang, cap, rd):
    num = cap.split()[-1]
    sp = {"text": num, "kind": "dec", "reading": rd}
    return CC.gate_class({"text": cap.replace(num, rd), "caption": cap, "spans": [sp]}, sp, lang)


class ReadingFixes(unittest.TestCase):
    def test_pt_meia_is_a_phone_digit(self):
        self.assertEqual(RC.phone_groups("pt", [("966", "nove seis meia")])[0], "digits")
        self.assertEqual(RC.phone_groups("pt", [("96", "nove meia")])[0], "digits")

    def test_italian_elision(self):
        self.assertEqual(RC.joined("cento otto"), RC.joined("centotto"))
        self.assertEqual(RC.joined("cento uno"), RC.joined("centuno"))
        self.assertEqual(RC.joined("centouno"), RC.joined("centuno"))
        self.assertEqual(RC.joined("venti uno"), RC.joined("ventuno"))
        self.assertEqual(RC.joined("cento ottanta"), RC.joined("centottanta"))

    def test_dotted_clock_and_decimal_fractions(self):
        for lang, cap, rd, want in [
                ("es", "a las 8.30", "ocho y media", "clock"), ("es", "a las 8.15", "ocho y cuarto", "clock"),
                ("es", "a las 8.30", "ocho treinta", "dec_pair"), ("es", "mide 2,5", "dos y medio", "dec"),
                ("es", "mide 3,25", "tres y cuarto", "dec"), ("it", "misura 2.50", "due e mezzo", "dec"),
                ("it", "alle 2.30", "due e mezza", "clock"), ("it", "alle 18.30", "diciotto e trenta", "clock"),
                ("it", "misura 2,5", "due virgola cinque", "dec"),
                ("es", "son 3,5", "tres millones quinientos mil", "dec"),
                ("fr", "il y a 3,5", "trois millions cinq cent mille", "dec"), ("de", "Artikel 5.2", "fünf zwei", "dec_pair")]:
            self.assertEqual(gc(lang, cap, rd), want, (lang, cap, rd))

    def test_french_cent_is_a_number_not_the_minor_unit(self):
        cap = "Ça coûte 120,50 € ici."
        sp = [s for s in M.spans(cap, "fr") if s.kind == "cur"][0]
        self.assertTrue(T.reading_ok("fr", "MONEY", "price", None, sp, "cent vingt euros virgule cinquante", 0, cap))
        self.assertTrue(T.reading_ok("fr", "MONEY", "price", None, sp, "deux cents euros virgule cinq", 0, cap))
        self.assertFalse(T.reading_ok("fr", "MONEY", "price", None, sp, "douze euros virgule cinquante centimes", 0, cap))
        spn = [s for s in M.spans("Dat kost 12,50 € hier.", "nl") if s.kind == "cur"][0]
        self.assertFalse(T.reading_ok("nl", "MONEY", "price", None, spn, "twaalf euro komma vijftig cent", 0, "Dat kost 12,50 € hier."))


if __name__ == "__main__":
    unittest.main()
