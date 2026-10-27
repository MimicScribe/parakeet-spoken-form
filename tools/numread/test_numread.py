#!/usr/bin/env python3
"""Tests for numread: pins, install, golden file, red-on-base mutations, Want / classify / repeat edge cases. python3 -I test_numread.py"""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numread as R  # noqa: E402
import numread_golden as G  # noqa: E402


class Pins(unittest.TestCase):
    def test_verify_ok(self):
        got = R.verify_pins()
        self.assertIn("eval/quality/number_outcomes.py", got)

    def test_changed_pin_refuses(self):
        pins = copy.deepcopy(R.load_pins())
        pins["base"]["eval/quality/number_outcomes.py"] = "0" * 64
        with self.assertRaises(R.PinError):
            R.verify_pins(pins)

    def test_changed_normaliser_pin_refuses(self):
        pins = copy.deepcopy(R.load_pins())
        pins["repo"]["fair_reader.py"] = "1" * 64
        with self.assertRaises(R.PinError):
            R.verify_pins(pins)

    def test_missing_base_refuses(self):
        with self.assertRaises(R.PinError):
            R.verify_pins(base="/nonexistent/base")

    def test_install_idempotent_and_resolves_pinned(self):
        n1 = R.install()
        n2 = R.install()
        self.assertIs(n1, n2)
        import number_outcomes
        self.assertEqual(os.path.realpath(number_outcomes.__file__), os.path.realpath(n1.__file__))
        self.assertIn("reader", number_outcomes.__file__)

    def test_stamp(self):
        s = R.stamp()
        self.assertEqual(len(s["numread"]), 12)
        self.assertIn("tree@", R.stamp_line())


class Golden(unittest.TestCase):
    def test_golden_all_pass(self):
        pairs, bad = G.run()
        self.assertGreaterEqual(len(pairs), 150)
        self.assertEqual([p for p, _ in bad], [])

    def test_golden_has_the_required_negatives(self):
        keys = {(p.get("key"), p.get("hyp")) for p in G.load() if p["t"] == "acc"}
        for k in [("twenty five", "in 2005"), ("twenty", "2"), ("seven", "17"), ("three hundred", "3"), ("five", "50"), ("five", "500"), ("fifty", "500")]:
            self.assertIn(k, keys)
        reads = {p["text"] for p in G.load() if p["t"] == "reads"}
        for t in ("in twenty five", "sixty si", "twenty", "seven", "three hundred", "half a percent growth", "drei einhalb Prozent", "40 mil millones", "twee en half", "double oh seven", "triple seven"):
            self.assertIn(t, reads)


class RedOnBase(unittest.TestCase):
    """the golden file must go red when the rules it protects are broken"""

    def run_golden_with(self, name, fn):
        orig = getattr(R, name)
        setattr(R, name, fn)
        try:
            return len(G.run()[1])
        finally:
            setattr(R, name, orig)

    def test_split_without_adjacency_goes_red(self):
        def loose(want, rr, maxparts=8):
            targets = set(want.alts) | set(want.joined)
            vals = [r.val for r in rr]  # any numbers anywhere in the bucket (the old scorer)
            for i in range(len(vals)):
                acc = ""
                for j in range(i, min(i + maxparts, len(vals))):
                    acc += vals[j]
                    if j > i and acc in targets:
                        return "grouped", vals[i:j + 1]
            return None
        self.assertGreater(self.run_golden_with("_split_hit", loose), 0)

    def test_trailing_zero_tolerance_goes_red(self):
        orig = R.Want
        pairs, bad0 = G.run()
        orig_want_from = R.want_from

        def lenient(reading, text=None, lang="en"):
            w = orig_want_from(reading, text, lang)
            extra = {a.rstrip("0") for a in w.alts if a.rstrip("0")} | {a + "0" for a in w.alts}
            return R.Want(set(w.alts) | extra, w.parts, w.joined, w.conflict, w.primary)
        R.want_from = lenient
        try:
            self.assertGreater(len(G.run()[1]), 0)
        finally:
            R.want_from = orig_want_from
        self.assertEqual(len(G.run()[1]), 0)

    def test_no_decimal_scale_fix_goes_red(self):
        self.assertGreater(self.run_golden_with("fix_decimal_scale", lambda t, lang: t), 0)


class Units(unittest.TestCase):
    def test_want_conflict_is_flagged_and_text_ignored(self):
        w = R.want_from("three five", "3", "en")
        self.assertTrue(w.conflict)
        self.assertEqual(R.classify(w, "35", "en")[0], "ok")
        self.assertNotEqual(R.classify(w, "3", "en")[0], "ok")

    def test_want_text_adds_decimal_when_digits_agree(self):
        w = R.want_from("five ninety nine", "5.99", "en")
        self.assertFalse(w.conflict)
        self.assertIn("5.99", w.alts)

    def test_dup_refcount(self):
        w = R.want_from("ten", None, "en")
        self.assertEqual(R.classify(w, "ten, ten", "en", refcount=1)[0], "dup")
        self.assertEqual(R.classify(w, "ten, ten", "en", refcount=2)[0], "ok")

    def test_split_requires_same_clause(self):
        w = R.want_from("twenty five", None, "en")
        self.assertEqual(R.classify(w, "2. 5", "en")[0], "trunc")  # sentence break between the parts: not a split
        self.assertEqual(R.classify(w, "2 5", "en")[0], "split")

    def test_unscorable(self):
        self.assertEqual(R.classify(R.want_from("hello", None, "en"), "five", "en")[0], "unscorable")

    def test_normalize_idempotent(self):
        for t in ("twenty o five", "point four", "two point seventeen billion", "2,005", "double oh seven", "one hundred, and five"):
            once = R.normalize(t, "en")
            self.assertEqual(R.normalize(once, "en"), once, t)

    def test_repeat_count_modes(self):
        self.assertEqual(R.repeat_count("five five five", "5"), 3)
        self.assertEqual(R.repeat_count("fifty five", "5"), 0)
        self.assertEqual(R.run_lengths("nine five five five", "en"), [("5", 3)])

    def test_dstr(self):
        self.assertEqual(R.dstr("2.170"), "2.17")
        self.assertEqual(R.dstr("1E+3"), "1000")
        self.assertEqual(R.dstr("0.50"), "0.5")


if __name__ == "__main__":
    unittest.main(verbosity=1)
