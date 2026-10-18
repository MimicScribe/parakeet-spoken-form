"""Run: python3 test_ro_gender.py (red on base: ro_gender does not exist there and expand_cands lets "doi mii" through)."""
import unittest

import ro_gender as G


class RoGender(unittest.TestCase):
    def test_scale_nouns_take_feminine(self):
        for bad, good in [("doi mii șaptesprezece", "două mii șaptesprezece"),
                          ("doi mii douăzeci și unu", "două mii douăzeci și unu"),
                          ("douăzeci și doi mii", "douăzeci și două mii"),
                          ("doi sute cincizeci", "două sute cincizeci"),
                          ("doi milioane de euro", "două milioane de euro"),
                          ("doi miliarde", "două miliarde"),
                          ("doisprezece mii", "douăsprezece mii")]:
            self.assertFalse(G.grammatical(bad), bad)
            self.assertEqual(G.repair(bad), good)
            self.assertTrue(G.grammatical(good), good)

    def test_decimal_before_scale(self):
        self.assertEqual(G.repair("doi virgulă cinci miliarde de euro"), "două virgulă cinci miliarde de euro")
        self.assertTrue(G.grammatical("doi virgulă cinci la sută"))
        self.assertTrue(G.grammatical("doi virgulă cinci"))

    def test_clock_hours(self):
        self.assertEqual(G.repair("doisprezece", before="Este ora"), "douăsprezece")
        self.assertEqual(G.repair("doisprezece și treizeci", before="la ora"), "douăsprezece și treizeci")
        self.assertEqual(G.repair("douăzeci și doi treizeci", before="orele"), "douăzeci și două treizeci")
        self.assertEqual(G.repair("doi treizeci", before="la ora"), "două treizeci")
        self.assertTrue(G.grammatical("doisprezece", before="în anul"))
        self.assertTrue(G.grammatical("unu", before="ora"))

    def test_feminine_nouns_and_masculine_ok(self):
        self.assertFalse(G.grammatical("doi", after="minute."))
        self.assertFalse(G.grammatical("doisprezece minute"))
        self.assertTrue(G.grammatical("doi ani"))
        self.assertTrue(G.grammatical("doi", after="ani"))
        self.assertTrue(G.grammatical("doi la sută"))
        self.assertTrue(G.grammatical("două mii douăzeci și una"))

    def test_expand_cands_never_returns_masculine_scale(self):
        import ml_multi_ctc as C

        out = C.expand_cands("2021", ["doi mii douăzeci și unu"], "ro")
        self.assertTrue(out)
        self.assertTrue(all(G.grammatical(x) for x in out), out)
        out = C.expand_cands("12", ["doisprezece"], "ro", before="la ora ")
        self.assertTrue(out and all(G.grammatical(x, "la ora") for x in out), out)


if __name__ == "__main__":
    unittest.main()
