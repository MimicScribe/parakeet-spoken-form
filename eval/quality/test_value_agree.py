"""value_agree: label vs second transcript, number-aware. python3 test_value_agree.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import value_agree as V


class Agree(unittest.TestCase):
    def test_digits_for_words(self):
        self.assertTrue(V.agree("We spreken om half zeven af", "we spreken om half 7 af", "nl")[0])
        self.assertTrue(V.agree("Begint om vijf na elf in de zaal", "begint om 5 na 11 in de zaal", "nl")[0])
        self.assertTrue(V.agree("is negenentwintig euro negenennegentig inclusief", "is 29,99 euro inclusief", "nl")[0])

    def test_split_tens_and_coma(self):
        self.assertTrue(V.agree("het tre sette ottantatré trentanove", "il tre sette ottanta tre trentanove".replace("il ", "het "), "it")[0])
        self.assertTrue(V.agree("cuarenta y cuatro con cuatro litros", "cuarenta y cuatro coma cuatro litros", "es")[0])

    def test_phone_digit_string(self):
        ref = "é o zero nove seis cinco seis cinco seis oito três dois"
        self.assertTrue(V.agree(ref, "é o 0965 656832", "pt")[0])
        self.assertFalse(V.agree(ref, "é o 0965 656833", "pt")[0])

    def test_rejects(self):
        self.assertEqual(V.agree("bei vierundneunzig zehn inklusive", "bei vierundneunzig euro zehn inklusive", "de"), (False, "words"))
        self.assertEqual(V.agree("zahlte fünf neunzig in die", "zahlte fünfundneunzig in die", "de"), (False, "numbers"))
        self.assertFalse(V.agree("um einem Uhr ab", "um ein uhr ab", "de")[0])  # a declined "ein" is a word
        self.assertFalse(V.agree("no seis três trinta e cinco", "no seis três três cinco", "pt")[0])

    def test_rescue_on_digit_text(self):
        # the old rescue compared letters with the digits stripped: "0965 656832" never equalled the label
        self.assertTrue(V.rescued("é o zero nove seis cinco seis cinco seis oito três dois", "é o 0965 656832", "pt"))
        self.assertTrue(V.rescued("Das ist gut.", "das ist gut", "de"))
        self.assertFalse(V.rescued("Das ist gut.", "das ist nicht gut", "de"))
        self.assertFalse(V.rescued("Das ist gut.", "", "de"))

    def test_digitize(self):
        self.assertEqual(V.digitize("zahlte neunundfünfzig euro fünfundzwanzig für", "de"), "zahlte 59 euro 25 für")
        self.assertEqual(V.digitize("il tre sette ottanta tre", "it"), "il 3 7 83")


if __name__ == "__main__":
    unittest.main()
