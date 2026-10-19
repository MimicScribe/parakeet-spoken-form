import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wer_sentinel as W  # noqa: E402


class Normalise(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(W.norm_tokens("Hello, World! It's «fine»."), ["hello", "world", "its", "fine"])
        self.assertEqual(W.norm_tokens("dix-sept l’homme"), ["dix", "sept", "lhomme"])

    def test_join_spaced_number_words(self):
        self.assertEqual(W.words("zwei und zwanzig Hunde", "de"), ["zweiundzwanzig", "hunde"])
        self.assertEqual(W.words("vier en twintig", "nl"), ["vierentwintig"])
        self.assertEqual(W.words("venti tre gatti", "it"), ["ventitré", "gatti"])
        self.assertEqual(W.words("zwei und zwanzig", "fr"), ["zwei", "und", "zwanzig"])  # only de / nl / it join

    def test_plain_words_untouched(self):
        self.assertEqual(W.words("ein Hund und eine Katze", "de"), ["ein", "hund", "und", "eine", "katze"])


class Scoring(unittest.TestCase):
    def rows(self, pairs):
        return [{"id": f"{l}_{i}", "ref": r, "hyp": h} for i, (l, r, h) in enumerate(pairs)]

    def test_wer(self):
        sc = W.score_rows(self.rows([("en", "a b c d", "a b x d"), ("en", "a b", "a b"), ("de", "a b", "a")]))
        self.assertAlmostEqual(W.wer(sc, ("en",)), 100 / 6)
        self.assertAlmostEqual(W.wer(sc, ("de",)), 50.0)
        self.assertEqual(W.pooled(sc, ("en", "de")), (2, 8))

    def test_number_spacing_is_not_an_error(self):
        sc = W.score_rows(self.rows([("de", "zweiundzwanzig Hunde", "zwei und zwanzig Hunde")]))
        self.assertEqual(sc["de_0"][1], 0)

    def test_report_verdict(self):
        base = self.rows([("en", "a b c d e f g h", "a b c d e f g h")] * 20 + [("ru", "а б", "а б")] * 20)
        bad = self.rows([("en", "a b c d e f g h", "a b c d e f g x")] * 20 + [("ru", "а б", "а б")] * 20)
        arms = {"ctl": W.score_rows(base), "bad": W.score_rows(bad), "same": W.score_rows(base)}
        _, summ = W.report(arms, "ctl")
        self.assertEqual(summ["verdict"]["same"], "PASS")
        self.assertTrue(summ["verdict"]["bad"].startswith("FAIL"))
        self.assertIn("en", summ["verdict"]["bad"])


if __name__ == "__main__":
    unittest.main()
