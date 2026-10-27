"""Tests for fair_reader.py (number normaliser). Run: python3 -I test_fair_reader.py"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_reader as F  # noqa: E402



class Normalize(unittest.TestCase):
    def n(self, text, lang="en", lenient=False):
        return F.normalize(text, lang, lenient)

    def test_english_o_years(self):
        self.assertEqual(self.n("since twenty o five which"), "since 2005 which")
        self.assertEqual(self.n("in Nineteen Oh Nine, we"), "in 1909, we")
        self.assertEqual(self.n("twenty, o five"), "twenty, o five")  # a comma breaks the number

    def test_oh_is_zero_only_inside_spoken_digit_runs(self):
        self.assertEqual(self.n("call o nine o nine two"), "call 09092")  # spoken digit run = one string
        self.assertEqual(self.n("oh I see two things"), "oh I see two things")
        self.assertEqual(self.n("oh two"), "oh two")  # one real digit word is not a digit string

    def test_thousands_mark_per_language(self):
        self.assertEqual(self.n("the 2,005 units"), "the 2005 units")
        self.assertEqual(self.n("2.005 percent"), "2.005 percent")  # English decimal stays
        self.assertEqual(self.n("1.500 Euro", "de"), "1500 Euro")
        self.assertEqual(self.n("1,5 Prozent", "de"), "1,5 Prozent")
        self.assertEqual(self.n("125.934 envíos", "es"), "125934 envíos")
        self.assertEqual(self.n("1,234,567"), "1,234,567")  # repeated marks: the reader already reads these
        self.assertEqual(self.n("v1.500.2", "de"), "v1.500.2")  # dotted chain is not one thousands mark

    def test_english_rules_never_touch_other_languages(self):
        self.assertEqual(self.n("twenty o five", "de"), "twenty o five")

    def test_lenient_only_when_asked(self):
        self.assertEqual(self.n("one fifty eight units"), "one fifty eight units")
        self.assertEqual(self.n("one fifty eight units", lenient=True), "158 units")
        self.assertEqual(self.n("ISO forty two seventeen codes", lenient=True), "ISO 4217 codes")
        self.assertEqual(self.n("five twenty-three", lenient=True), "523")

    def test_lenient_keeps_clocks_and_plain_counts(self):
        self.assertEqual(self.n("at four fifty pm", lenient=True), "at four fifty pm")
        self.assertEqual(self.n("between one and fifty", lenient=True), "between one and fifty")
        self.assertEqual(self.n("two thousand twenty five", lenient=True), "two thousand twenty five")
        self.assertEqual(self.n("seventeen twenty five", lenient=True), "seventeen twenty five")


class FormRules(unittest.TestCase):
    """clock dot, leading-zero codes, digit + decimal word, digit + scale word + number words."""

    def n(self, text, lang="en"):
        return F.normalize(text, lang)

    def test_clock_dot(self):
        self.assertEqual(self.n("um 21.45 Uhr", "de"), "um 21:45 Uhr")
        self.assertEqual(self.n("ist 14.48 Minuten und fünfzehn Sekunden.", "de"), "ist 14:48 Minuten und fünfzehn Sekunden.")
        self.assertEqual(self.n("à 9.05 heures", "fr"), "à 9:05 heures")
        self.assertEqual(self.n("a las 16.55 horas", "es"), "a las 16:55 horas")
        self.assertEqual(self.n("um 21:45 Uhr", "de"), "um 21:45 Uhr")

    def test_clock_dot_leaves_decimals_and_amounts(self):
        self.assertEqual(self.n("3.50 Euro", "de"), "3.50 Euro")  # no clock unit: not a time
        self.assertEqual(self.n("14.48 Minuten lang", "de"), "14.48 Minuten lang")  # no seconds: a duration
        self.assertEqual(self.n("25.45 Uhr", "de"), "25.45 Uhr")  # not an hour
        self.assertEqual(self.n("12.345 Uhr", "de"), "12345 Uhr")  # thousands mark, not a clock

    def test_leading_zero_codes(self):
        # One convention since the spoken-digit-run rule: a code stays ONE digit string (normalize no longer expands).
        self.assertEqual(self.n("Code 00110 ok", "de"), "Code 00110 ok")
        self.assertEqual(F.fix_leading_zero_codes("0010000"), "0 0 1 0 0 0 0")

    def test_leading_zero_codes_leave_decimals_dates_and_plain_numbers(self):
        for t in ("314 000 euros", "5 065, neuf", "0,05 Prozent", "0.5 percent", "05", "2020-01-05", "10:05", "1005", "12/030"):
            self.assertEqual(self.n(t, "de"), t)

    def test_spoken_decimal(self):
        self.assertEqual(self.n("16 Komma vier Prozent", "de"), "16,4 Prozent")
        self.assertEqual(self.n("4 Komma acht acht Milliarden", "de"), "4,88 Milliarden")
        self.assertEqual(self.n("16 point four percent"), "16.4 percent")
        self.assertEqual(self.n("3 virgule 5.", "fr"), "3,5.")
        self.assertEqual(self.n("10 mal 0 Komma zwei plus", "de"), "10 mal 0,2 plus")

    def test_spoken_decimal_needs_digits_after(self):
        self.assertEqual(self.n("16 Komma und dann", "de"), "16 Komma und dann")
        self.assertEqual(self.n("sechzehn Komma vier", "de"), "sechzehn Komma vier")

    def test_digit_scale_words(self):
        self.assertEqual(self.n("129 Millionen dreihundertfünfundsiebzigtausend Euro", "de"), "129375000 Euro")
        self.assertEqual(self.n("3 millions trois cent mille de", "fr"), "3300000 de")
        self.assertEqual(self.n("3 million five hundred thousand dollars"), "3500000 dollars")

    def test_digit_scale_words_leave_other_text(self):
        self.assertEqual(self.n("222 millions en deux mille trente cinq", "fr"), "222 millions en deux mille trente cinq")
        self.assertEqual(self.n("10 million five", "en"), "10 million five")
        self.assertEqual(self.n("65 Millionen Euro", "de"), "65 Millionen Euro")

    def test_a_text_without_these_forms_is_unchanged(self):
        for lang, t in (("de", "um einundzwanzig Uhr fünfundvierzig sind es sechzehn Komma vier Prozent"),
                        ("en", "call zero zero one one zero at twenty o five"),
                        ("es", "ciento veinte millones trescientos mil euros")):
            self.assertEqual(F.normalize(t, lang), F.normalize(F.normalize(t, lang), lang))
        self.assertEqual(self.n("um einundzwanzig Uhr fünfundvierzig sind es sechzehn Komma vier Prozent", "de"),
                         "um einundzwanzig Uhr fünfundvierzig sind es sechzehn Komma vier Prozent")


class WithPinnedReader(unittest.TestCase):
    """The rewrite makes the pinned reader return the value a person reads (smallest value, as the scorers keep)."""

    def read(self, text, lang, lenient=False):
        import numread

        N = numread.install()
        nums = [n for n in N.extract_numbers(F.normalize(text, lang, lenient), lang, keep_weak=True) if n.vals]
        return [str(int(v)) if v == int(v) else str(v) for v in (min(n.vals) for n in nums)]

    def test_values(self):
        self.assertEqual(self.read("since twenty o five which", "en"), ["2005"])
        self.assertEqual(self.read("the 2,005 units", "en"), ["2005"])
        self.assertEqual(self.read("1.500 Euro", "de"), ["1500"])
        self.assertEqual(self.read("125.934 envíos", "es"), ["125934"])
        self.assertEqual(self.read("1,5 Prozent", "de"), ["1.5"])
        self.assertEqual(self.read("one fifty eight units", "en", lenient=True), ["158"])

    def test_new_rules_read_like_the_words(self):
        self.assertEqual(self.read("um 21.45 Uhr", "de"), self.read("um einundzwanzig Uhr fünfundvierzig", "de"))
        self.assertEqual(self.read("14.48 Minuten und fünfzehn Sekunden", "de"),
                         self.read("vierzehn Uhr achtundvierzig Minuten und fünfzehn Sekunden", "de"))
        self.assertEqual(self.read("16 Komma vier Prozent", "de"), ["16.4"])
        self.assertEqual(self.read("129 Millionen dreihundertfünfundsiebzigtausend", "de"), ["129375000"])
        self.assertEqual(self.read("00110", "en"), self.read("zero zero one one zero", "en"))



class SpokenDigitRuns(unittest.TestCase):
    def test_phone_words_become_one_string(self):
        self.assertEqual(F.fix_spoken_digit_runs("ruf null eins sieben zwei an", "de"), "ruf 0172 an")
        self.assertEqual(F.fix_spoken_digit_runs("call nine nine nine two seven oh two.", "en"), "call 9992702.")

    def test_decimal_tail_is_left_alone(self):
        self.assertEqual(F.fix_spoken_digit_runs("neunundfünfzig Komma sieben sechs sieben", "de"),
                         "neunundfünfzig Komma sieben sechs sieben")

    def test_whole_decimal_tail_and_mixed_phone_are_left_alone(self):
        self.assertEqual(F.fix_spoken_digit_runs("siebenundsechzig Komma sechs neun drei sechs.", "de"),
                         "siebenundsechzig Komma sechs neun drei sechs.")
        t = "null eins fünf eins zwei neun neun zweitausendsiebenhunderteinundsechzig an"
        self.assertEqual(F.fix_spoken_digit_runs(t, "de"), t)

    def test_two_digit_words_and_punctuation_break_runs(self):
        self.assertEqual(F.fix_spoken_digit_runs("zwei drei Leute", "de"), "zwei drei Leute")
        self.assertEqual(F.fix_spoken_digit_runs("one, two, three", "en"), "one, two, three")

if __name__ == "__main__":
    unittest.main()
