"""Tests of the adversarial pack (adv_b): labels, readings, splits, planner, test planner. No network, no model."""
import collections
import json
import random
import re
import unittest

import adv_b as A
import adv_b_content as K
import adv_b_phone as PH
import adv_b_words as W

LANGS = A.LANGS


class Labels(unittest.TestCase):
    def test_every_piece_is_strict_spoken_form(self):
        for lang in LANGS:
            for c in K.BUILD[lang]():
                for i, k in enumerate(c.values):
                    for mk in (c.make, c.ctrl):
                        if mk is None:
                            continue
                        p = mk(k, random.Random(i))
                        self.assertFalse(re.search(r"[0-9\-]", p.label), (c.id, k, p.label))
                        cur = 0
                        for kind, w, r in p.nums:
                            j = p.label.find(r, cur)
                            self.assertGreaterEqual(j, 0, (c.id, k, p.label, r))
                            cur = j + len(r)

    def test_phone_styles_are_one_number_many_readings(self):
        us = PH.universe("us", 40)
        rng = random.Random(1)
        seen = set()
        for st in ("digits_zero", "digits_oh", "mixed", "triplet", "double", "area_code", "country_us"):
            g, plus, pre = PH.us_style(st, us[5], rng)
            seen.add((pre + " " + " ".join(g)).strip())
        self.assertGreaterEqual(len(seen), 6)

    def test_known_readings(self):
        self.assertEqual(PH.us_style("mixed", "8089362323", random.Random(0))[0], ["eight oh eight", "nine three six", "twenty three", "twenty three"])
        self.assertEqual(PH.us_style("digits_zero", "8089362323", random.Random(0))[0][0], "eight zero eight")
        self.assertEqual(W.digits_rep("8055", "en", "oh"), "eight oh double five")
        self.assertEqual(W.digits_rep("555", "en", "oh", "treble"), "treble five")
        self.assertEqual(PH.es_style("hundreds_pairs", "612345678", random.Random(0), "es_mob")[0][:2], ["seiscientos doce", "treinta y cuatro"])
        self.assertEqual(PH.fr_style("pairs", "0612345678", random.Random(0), "fr")[0][:3], ["zéro six", "douze", "trente quatre"])
        self.assertEqual(W.card(23, "de"), "drei und zwanzig")
        self.assertEqual(W.card(23, "it"), "venti tre")


class Italian(unittest.TestCase):
    def test_no_accented_tre_and_no_apocope(self):
        """Plan section 0: the final "tre" is unaccented; elided forms stay whole; -un / -una apocope forms are not generated."""
        for n in range(0, 1000):
            s = W.card(n, "it")
            self.assertNotIn("tré", s, n)
            for w in s.split():
                self.assertFalse(re.fullmatch(r"(vent|trent|quarant|cinquant|sessant|settant|ottant|novant)(un|una)", w), (n, w))
        self.assertEqual(W.card(21, "it"), "ventuno")
        self.assertEqual(W.card(23, "it"), "venti tre")
        for c in K.BUILD["it"]():
            for k in c.values[:40]:
                self.assertNotIn("é", c.make(k, random.Random(1)).label)


class Splits(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        for lang in ("en", "es", "fr", "de"):
            a, b = A.freeze(lang), A.freeze(lang)
            self.assertEqual(A.canonical(a), A.canonical(b))
            for cid, c in a["constructions"].items():
                self.assertFalse(set(c["V_a"]) & set(c["V_b"]), cid)

    def test_frozen_files_match_the_inventory(self):
        import os
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_b_split")
        man = json.load(open(os.path.join(d, "manifest.json")))
        import hashlib
        for lang in LANGS:
            s = json.load(open(os.path.join(d, f"advBsplit_{lang}.json"), encoding="utf-8"))
            self.assertEqual(hashlib.sha256(A.canonical(s)).hexdigest(), man["sha256"][lang])
            self.assertEqual(A.canonical(s), A.canonical(A.freeze(lang)), f"{lang}: inventory drifted from the frozen split")

    def test_phone_group_shares_one_cut(self):
        s = A.freeze("en")["constructions"]
        a, b = s["en.phone.us_mixed"], s["en.phone.us_area_code"]
        self.assertEqual(set(a["V_b"]) & set(b["V_a"]), set())

    def test_held_out_constructions_never_planned(self):
        for lang in ("en", "es", "fr"):
            split = A.freeze(lang)
            rows, rep = A.plan(lang, split, {"phone": 60, "stumble": 40, "phrasing": 60}, min_each=2)
            self.assertTrue(rows)
            held = A.C_B[lang]
            vb = {cid: set(c["V_b"]) for cid, c in split["constructions"].items()}
            for r in rows:
                self.assertNotIn(r["construction"], held)
                for k in r["values"]:
                    self.assertNotIn(k, vb[r["construction"]])
                self.assertFalse(re.search(r"[0-9]", r["text"]))
                self.assertEqual(r["half"], "A")
                self.assertIn(r["family"], A.FAMILIES)
                self.assertEqual(r["edge"], [r["family"]])
                self.assertTrue(r["edge_spans"])

    def test_test_plan_uses_only_b_halves_and_has_matched_controls(self):
        split = A.freeze("en")
        rows, ctl = A.test_plan("en", split, per_cell=2)
        self.assertTrue(rows and ctl)
        va = {cid: set(c["V_a"]) for cid, c in split["constructions"].items()}
        for r in rows:
            side = split["constructions"][r["construction"]]["side"]
            if side == "C_a":
                self.assertEqual(r["quad"], "C_a_V_b")
                for k in r["values"]:
                    self.assertNotIn(k, va[r["construction"]])
        ids = {r["id"] for r in rows}
        for c in ctl:
            self.assertIn(c["control_of"], ids)
            self.assertEqual(c["bed"], "control")


if __name__ == "__main__":
    unittest.main()
