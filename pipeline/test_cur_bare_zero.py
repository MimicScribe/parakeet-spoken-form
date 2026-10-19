"""A price with cents under 10 never gets the joined, unit-less reading (2026-10-18): "siete con cinco" / "trinta e cinco"
leave the zero out and say 7,5 / 35. python3 test_cur_bare_zero.py"""
import unittest

import ml_cands as M


def cands(lang, text):
    return [c for s in M.spans_regional(text, lang, M.FILL_ALL) if s.kind == "cur" for c in s.cands]


class CurBareZero(unittest.TestCase):
    def test_single_digit_cents_have_no_unitless_joined_reading(self):
        for lang, text, bad in [("es", "Cuesta 7,05 €.", ["siete con cinco", "siete y cinco"]),
                                ("es", "Cuesta 7,05 $.", ["siete con cinco", "siete y cinco"]),
                                ("pt", "Custa 30,05 €.", ["trinta e cinco"]),
                                ("pt", "Custa R$ 30,05.", ["trinta e cinco"]),
                                ("fr", "Ça coûte 7,05 €.", ["sept et cinq"]),
                                ("it", "Costa 7,05 €.", ["sette e cinque", "sette virgola cinque"])]:
            got = cands(lang, text)
            self.assertTrue(got, (lang, text))
            for b in bad:
                self.assertNotIn(b, got, (lang, text))

    def test_unit_forms_and_two_digit_cents_stay(self):
        # red-on-base: control (the fix must not move these)
        self.assertIn("siete euros con cinco", cands("es", "Cuesta 7,05 €."))
        self.assertIn("trinta euros e cinco", cands("pt", "Custa 30,05 €."))
        self.assertIn("siete con cincuenta", cands("es", "Cuesta 7,50 €."))
        self.assertIn("trinta e cinquenta", cands("pt", "Custa 30,50 €."))
        self.assertIn("siete con diez", cands("es", "Cuesta 7,10 €."))


if __name__ == "__main__":
    unittest.main()
