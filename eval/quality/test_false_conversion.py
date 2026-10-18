"""python3 -m unittest test_false_conversion  (stdlib; imports pipeline/ml_balance)"""
import unittest

import false_conversion as F


class FalseConversion(unittest.TestCase):
    def test_counts(self):
        man = [{"id": "a", "text": "Das ist ein guter Tag für uns alle"},
               {"id": "b", "text": "Wir brauchen neuen Schwung"},
               {"id": "c", "text": "Außer Acht lassen wir das nicht"}]
        stock = {"a": "Das ist ein guter Tag für uns alle", "b": "Wir brauchen neuen Schwung",
                 "c": "Außer Acht lassen wir das nicht"}
        arm = {"a": "Das ist 1 guter Tag für uns alle", "b": "Wir brauchen neun Schwung",
               "c": "Außer Acht lassen wir das nicht"}
        r = F.score(man, {"stock": stock, "arm": arm}, "de", base="stock", boot=200)
        s, a = r["arms"]["stock"], r["arms"]["arm"]
        self.assertEqual((s["digits"], s["numwords"]), (0, 0))  # the article and the homophone in the reference count 0
        self.assertEqual((a["digits"], a["numwords"]), (1, 1))  # "1" for "ein", "neun" for "neuen"
        self.assertEqual(a["rows_with_false"], 2)
        self.assertAlmostEqual(a["total_per_1k"], 1000 * 2 / r["ref_words"], places=2)
        self.assertGreater(a["vs_base"]["diff_per_1k"], 0)
        self.assertEqual((a["vs_base"]["rows_worse"], a["vs_base"]["rows_better"]), (2, 0))

    def test_reference_number_word_is_not_false(self):
        # a homophone token the reference has ("sei" = you are) is matched, an extra one is not
        man = [{"id": "x", "text": "tu sei sicuro"}]
        r = F.score(man, {"arm": {"x": "tu sei sei sicuro"}}, "it")
        self.assertEqual(r["arms"]["arm"]["numwords"], 1)
        r = F.score(man, {"arm": {"x": "tu sei sicuro"}}, "it")
        self.assertEqual(r["arms"]["arm"]["numwords"], 0)

    def test_compound_and_symbols(self):
        man = [{"id": "x", "text": "es war ein schöner Abend"}]
        r = F.score(man, {"arm": {"x": "es war zweieinhalb schöne Abend 5 %"}}, "de")
        self.assertEqual(r["arms"]["arm"]["numwords"], 1)
        self.assertEqual(r["arms"]["arm"]["digits"], 2)


if __name__ == "__main__":
    unittest.main()
