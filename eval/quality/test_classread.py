"""classread.py: the pure helpers (unit / decimal-word flags, label window, cluster ids, paired aggregation)."""
import unittest

import classread as C


class Sp:
    cands = ["zehn kilometer"]


class ClassreadTests(unittest.TestCase):
    def test_cluster(self):
        self.assertEqual(C.cluster("fleursval_123_a", "de"), "de:fleurs:123")
        self.assertEqual(C.cluster("20170517-0900-PLENARY-16-nl_x", "nl"), "nl:vox:20170517-0900-PLENARY")
        self.assertTrue(C.cluster("zz999_20200101_abcdefghijk_3", "es").startswith("es:yt:abcdefghijk"))

    def test_symbol(self):
        self.assertTrue(C._has_symbol(["30%"]))
        self.assertTrue(C._has_symbol(["30", "%"]))
        self.assertTrue(C._has_symbol(["1", "kg,"], "1kg"))
        self.assertFalse(C._has_symbol(["30", "de"], "30 km"))
        self.assertFalse(C._has_symbol(["30"]))

    def test_pct_flags(self):
        self.assertEqual(C.unit_flags("pct", Sp, ["dreißig", "Prozent"], ["30"], ["30"], "de"), (True, True))
        self.assertEqual(C.unit_flags("pct", Sp, ["dreißig", "Prozent"], ["30", "%"], ["30", "%"], "de"), (True, False))
        self.assertEqual(C.unit_flags("pct", Sp, ["dreißig", "Prozent"], ["dreißig", "Prozent"], ["dreissig", "prozent"], "de"),
                         (True, False))
        self.assertEqual(C.unit_flags("int_small", Sp, ["zehn"], ["10"], ["10"], "de"), (False, False))

    def test_dec_flags(self):
        self.assertEqual(C.dec_flags("dec", ["drei", "Komma", "fünf"], ["drei", "fünf"], "de"), (True, True))
        self.assertEqual(C.dec_flags("dec", ["drei", "Komma", "fünf"], ["3,5"], "de"), (True, False))
        self.assertEqual(C.dec_flags("dec", ["drei", "Komma", "fünf"], ["drei", "Komma", "fünf"], "de"), (True, False))
        self.assertEqual(C.dec_flags("pct", ["drei", "Komma", "fünf"], [], "de"), (False, False))

    def test_label_window(self):
        lab = "a b zehn c d"
        self.assertEqual(C._label_window(lab, (4, 8)), ["b", "zehn", "c"])

    def test_paired_identical_arms_have_zero_difference(self):
        rec = lambda cl, r: dict(cl=cl, right=r, words=True, digits=False, us=True, ud=False, ds=False, dd=False)  # noqa: E731
        keys = [("de", str(i), "x", 0) for i in range(30)]
        a = {k: rec(f"de:vox:{i % 6}", i % 3 > 0) for i, k in enumerate(keys)}
        out = C.aggregate({"a": a, "b": dict(a)}, keys, ["a"], 200)
        d = out["b"]["vr"]["d_a"]
        self.assertEqual(d[0], 0.0)
        self.assertEqual(d[1:], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
