"""Known-answer tests for the held-out scorers (symscore.py via classread.score_rows, and score_heldout's four outcomes).

    python3 -m unittest test_known_answers          (stdlib; imports pipeline/ml_cands)

fixtures/known_answer_rows.json holds SYNTHETIC rows (invented carrier sentences, 30 per language for de es fr it nl pt;
make_known_answer_fixture.py rebuilds it with num2words, a number-to-words library independent of the scorer's lexicon).
Each row has a caption (digits), a label (spoken form) and hypotheses built from the label by one deliberate corruption.
Hypotheses with a KNOWN answer:
  (a) hyp = label                        value right, written in words, no unit / decimal word missing
  (b) hyp = caption                      value right, written in digits, unit symbol present
  (c) one corruption per hypothesis      swap (another value), trunc (scale word cut), drop_unit, drop_dec,
                                         respace (same words, other spaces / hyphens: still right), fuse and misplit
                                         (invented / garbled words), en_words (English number words)
Tests marked SCORER BUG are expectedFailure: they state the right answer, fail on the current scorer, and turn into an
"unexpected success" failure the day the scorer is fixed (then drop the marker).
"""
import collections
import contextlib
import io
import json
import os
import random
import tempfile
import unittest

import classread as CR
import ml_cands as M
import score_heldout as SH

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "known_answer_rows.json")
ROWS = json.load(open(FIX, encoding="utf-8"))
LANGS = ["de", "es", "fr", "it", "nl", "pt"]
_MEMO = {}


def spans(row):
    return [s for s in M.spans_regional(row["caption"], row["lang"], SH.READINGS) if s.cands]


def score_many(row, hyps):
    """{id: record} of the row scored against several hypotheses (ids 'v0', 'v1' ...), through the real path
    classread.score_rows -> symscore."""
    (sp,) = spans(row)
    a = row["label"].index(row["seg"])
    caps = {}
    hy = []
    for k, h in enumerate(hyps):
        caps[f"v{k}"] = {"id": f"v{k}", "caption": row["caption"], "text": row["label"],
                         "spans": [{"text": sp.text, "kind": sp.kind}], "num_chars": [[a, a + len(row["seg"])]]}
        hy.append({"id": f"v{k}", "hyp": h, "ref": row["label"]})
    out = CR.score_rows(hy, row["lang"], caps)
    # "cl" is the recording cluster derived from the (here synthetic) hypothesis id: not part of the score
    return [{k: v for k, v in out[(f"v{k}", sp.text, 0)].items() if k != "cl"} for k in range(len(hyps))]


def rec(row, name):
    """The score record of one named hypothesis of a row: 'label', 'caption' or a key of row['hyps']."""
    key = (row["id"], name)
    if key not in _MEMO:
        hyp = row["label"] if name == "label" else row["caption"] if name == "caption" else row["hyps"][name]
        _MEMO[key] = score_many(row, [hyp])[0]
    return _MEMO[key]


def with_hyp(name):
    return [r for r in ROWS if name in ("label", "caption") or name in r["hyps"]]


class Fixture(unittest.TestCase):
    def test_thirty_rows_per_language(self):
        self.assertEqual(collections.Counter(r["lang"] for r in ROWS), {l: 30 for l in LANGS})

    def test_every_caption_has_exactly_one_scored_span(self):
        for r in ROWS:
            self.assertEqual(len(spans(r)), 1, (r["id"], r["caption"]))

    def test_label_and_caption_differ_only_where_the_number_stands(self):
        for r in ROWS:
            self.assertIn(r["seg"], r["label"])
            self.assertTrue(set(r["hyps"]) <= {"swap", "trunc", "drop_unit", "drop_dec", "respace", "fuse", "misplit",
                                               "en_words"}, r["id"])


class A_HypIsTheLabel(unittest.TestCase):
    def test_every_span_is_value_right_in_words_with_nothing_missing(self):
        for r in ROWS:
            x = rec(r, "label")
            with self.subTest(r["id"]):
                self.assertEqual(x["out"], "WORDS_RIGHT_VALUE")
                self.assertTrue(x["value_ok"])
                self.assertEqual(x["sform"], "words")
                self.assertTrue(x["clean"])
                self.assertFalse(x["unit_spoken"] and not x["unit_present"])
                self.assertFalse(x["dec_spoken"] and not x["dec_present"])
                self.assertEqual(bool(x["unit_spoken"]), bool(r["unit"]))
                if r["dec"]:
                    self.assertTrue(x["dec_spoken"])

    def test_aggregate_per_language_V_100_W_100_U_0(self):
        for lang in LANGS:
            recs = {(r["id"], r["caption"], 0): rec(r, "label") for r in ROWS if r["lang"] == lang}
            c = CR.sym_counts(recs)["ALL"]
            with self.subTest(lang):
                self.assertEqual((c["n"], c["V"], c["W"]), (30, 30, 30))
                self.assertEqual(c["Uw"][0], 0)
                self.assertEqual(c["Ud"], [0, 0])
                self.assertGreater(c["Uw"][1], 0)  # unit words are spoken in the fixture, so U has a denominator


class B_HypIsTheCaption(unittest.TestCase):
    def test_every_span_is_value_right_in_digits(self):
        for r in ROWS:
            x = rec(r, "caption")
            with self.subTest(r["id"]):
                self.assertEqual(x["out"], "DIGITS_RIGHT_VALUE")
                self.assertTrue(x["value_ok"])
                self.assertEqual(x["sform"], "digits")
                self.assertTrue(x["clean"])
                self.assertFalse(x["unit_spoken"] and not x["unit_present"])
                self.assertFalse(x["dec_spoken"] and not x["dec_present"])

    def test_aggregate_per_language_V_100_W_0_U_0(self):
        for lang in LANGS:
            recs = {(r["id"], r["caption"], 0): rec(r, "caption") for r in ROWS if r["lang"] == lang}
            c = CR.sym_counts(recs)["ALL"]
            with self.subTest(lang):
                self.assertEqual((c["n"], c["V"], c["W"]), (30, 30, 0))
                self.assertEqual(c["Uw"], [0, 0])
                self.assertEqual(c["Ud"][0], 0)
                self.assertGreater(c["Ud"][1], 0)


class C_Corruptions(unittest.TestCase):
    def each(self, name):
        rows = with_hyp(name)
        self.assertGreaterEqual(len(rows), 12, name)  # the corruption exists in every language that can have it
        return [(r, rec(r, name)) for r in rows]

    def test_a_swapped_morpheme_is_a_wrong_value_in_words(self):
        for r, x in self.each("swap"):
            with self.subTest(r["id"]):
                self.assertFalse(x["value_ok"])
                self.assertEqual((x["out"], x["sform"]), ("WRONG_VALUE", "words"))
                self.assertFalse(x["clean"])
                self.assertFalse(x["unit_spoken"] and not x["unit_present"])  # the unit word is still there

    def test_a_truncated_scale_word_is_a_wrong_value(self):
        langs = {r["lang"] for r, _ in self.each("trunc")}
        self.assertEqual(langs, set(LANGS))
        for r, x in self.each("trunc"):
            with self.subTest(r["id"]):
                self.assertFalse(x["value_ok"])
                self.assertEqual(x["out"], "WRONG_VALUE")

    def test_a_dropped_unit_word_is_a_unit_miss_not_a_value_error(self):
        for r, x in self.each("drop_unit"):
            if r["id"] == "ka_fr_21":  # SCORER BUG, see test_french_hundred_is_not_the_euro_cent
                continue
            with self.subTest(r["id"]):
                self.assertTrue(x["value_ok"])
                self.assertEqual(x["sform"], "words")
                self.assertTrue(x["unit_spoken"] and not x["unit_present"])
                self.assertFalse(x["clean"])

    def test_a_dropped_decimal_word_is_a_decimal_miss(self):
        for r, x in self.each("drop_dec"):
            with self.subTest(r["id"]):
                self.assertTrue(x["dec_spoken"] and not x["dec_present"])
                self.assertFalse(x["clean"])

    def test_respaced_words_are_still_right(self):
        for r, x in self.each("respace"):
            with self.subTest(r["id"]):
                self.assertTrue(x["value_ok"])
                self.assertEqual((x["out"], x["sform"]), ("WORDS_RIGHT_VALUE", "words"))

    def test_fused_and_garbled_words_do_not_carry_the_value(self):
        for name in ("fuse", "misplit"):
            for r, x in self.each(name):
                with self.subTest(f"{name} {r['id']}"):
                    self.assertFalse(x["value_ok"])
                    self.assertFalse(x["clean"])

    def test_english_number_words_are_a_wrong_or_dropped_value(self):
        for r, x in self.each("en_words"):
            with self.subTest(r["id"]):
                self.assertFalse(x["value_ok"])
                self.assertIn(x["out"], ("DROPPED", "WRONG_VALUE"))

    def test_aggregate_rates_of_each_corruption(self):
        """V / W / U_w over the 180 rows, one corruption at a time: the numbers a readout would print."""
        want = {"swap": (0, 100), "trunc": (0, 100), "en_words": (0, None), "fuse": (0, 0), "misplit": (0, None),
                "respace": (100, 100)}
        for name, (v, w) in want.items():
            recs = {(r["id"], r["caption"], 0): rec(r, name) for r in with_hyp(name)}
            c = CR.sym_counts(recs)["ALL"]
            with self.subTest(name):
                self.assertEqual(100 * c["V"] / c["n"], v)
                if w is not None:
                    self.assertEqual(100 * c["W"] / c["n"], w)
        recs = {(r["id"], r["caption"], 0): rec(r, "drop_unit") for r in with_hyp("drop_unit") if r["id"] != "ka_fr_21"}
        c = CR.sym_counts(recs)["ALL"]
        self.assertEqual((c["Uw"][0], c["Uw"][1]), (c["n"], c["n"]))  # every dropped unit is a miss: U_w = 100%
        self.assertEqual(c["V"], c["n"])


class D_ScorerBugs(unittest.TestCase):
    """Right answers the current scorer does not give. expectedFailure: see the module docstring."""

    @unittest.expectedFailure
    def test_french_hundred_is_not_the_euro_cent(self):
        # SCORER BUG (symscore.unit_sub): "cent" (hundred) is in the label's words and counts as the euro-cent unit, so
        # "deux cent cinquante" for "deux cent cinquante euros" has its unit "present" and the miss is not counted.
        r = next(r for r in ROWS if r["id"] == "ka_fr_21")
        x = rec(r, "drop_unit")
        self.assertTrue(x["unit_spoken"] and not x["unit_present"])

    @unittest.expectedFailure
    def test_legacy_four_outcome_does_not_credit_fused_or_garbled_number_words(self):
        # SCORER BUG (score_heldout._whole joins the region's words with no space): "dosmilcinco", "quatrecentmille",
        # "sie ben", "zweihunde rtfünfzig" are WORDS_RIGHT_VALUE in the four-outcome table; symscore calls them wrong.
        bad = [(r["id"], n) for n in ("fuse", "misplit") for r in with_hyp(n) if rec(r, n)["out"] == "WORDS_RIGHT_VALUE"]
        self.assertEqual(bad, [])

    @unittest.expectedFailure
    def test_english_leak_is_never_written_in_words(self):
        # SCORER BUG (symscore.read): the language's own reader takes a local word out of an English phrase ("ninety-SIX"
        # in fr, "hundred" / "thousand" pieces), so a leak can count as a number written in words (W) with a wrong value.
        bad = [r["id"] for r in with_hyp("en_words") if rec(r, "en_words")["sform"] == "words"]
        self.assertEqual(bad, [])

    @unittest.expectedFailure
    def test_every_caption_span_is_scored(self):
        # SCORER BUG (score_rows / score_spans skip a span whose caption has no candidate reading, without counting it):
        # these invented captions hold a number and are never scored, in any rate.
        gaps = [("de", "Es sind 3.000.000 hier."), ("it", "Sono 0,75 qui."), ("nl", "Het zijn 0,75 hier."),
                ("pt", "São 12,4 aqui."), ("pt", "São 300.000 aqui."), ("fr", "Il y a 6,75 ici."), ("it", "Sono 15,2 qui.")]
        unscored = [(lang, cap) for lang, cap in gaps
                    if not [s for s in M.spans_regional(cap, lang, SH.READINGS) if s.cands]]
        self.assertEqual(unscored, [])


class E_Determinism(unittest.TestCase):
    def all_records(self, rows):
        hy = []
        recs = {}
        for r in rows:
            for name in ["label", "caption"] + sorted(r["hyps"]):
                hyp = r["label"] if name == "label" else r["caption"] if name == "caption" else r["hyps"][name]
                recs[(r["id"], name)] = score_many(r, [hyp])[0]
        return recs

    def test_the_same_input_scored_twice_is_identical(self):
        rows = [r for r in ROWS if r["id"].endswith(("_03", "_11", "_17", "_21", "_23", "_28"))]
        a, b = self.all_records(rows), self.all_records(rows)
        self.assertEqual(a, b)
        dump = lambda d: json.dumps({"|".join(k): v for k, v in d.items()}, sort_keys=True, default=str)  # noqa: E731
        self.assertEqual(dump(a), dump(b))

    def test_a_batch_scores_the_same_as_one_hypothesis_at_a_time(self):
        r = next(r for r in ROWS if r["id"] == "ka_de_21")
        names = ["label", "caption"] + sorted(r["hyps"])
        hyps = [r["label"] if n == "label" else r["caption"] if n == "caption" else r["hyps"][n] for n in names]
        batch = score_many(r, hyps)
        self.assertEqual(batch, [score_many(r, [h])[0] for h in hyps])
        order = list(range(len(hyps)))
        random.Random(4).shuffle(order)
        shuffled = score_many(r, [hyps[k] for k in order])
        self.assertEqual([shuffled[order.index(k)] for k in range(len(hyps))], batch)

    def test_score_heldout_main_writes_the_same_json_twice(self):
        row = next(r for r in ROWS if r["id"] == "ka_pt_11")
        (sp,) = spans(row)
        a = row["label"].index(row["seg"])
        cap = {"r1": {"id": "r1", "caption": row["caption"], "text": row["label"],
                      "spans": [{"text": sp.text, "kind": sp.kind}], "num_chars": [[a, a + len(row["seg"])]]}}
        hyps = [{"id": "r1", "hyp": row["hyps"]["swap"], "ref": row["label"]}]
        old = (SH.fetch, SH._captions)
        SH.fetch = lambda run, lang, work, set_="heldout": hyps
        SH._captions = lambda lang, work, set_="heldout", mix_dir="mix4": cap
        outs = []
        try:
            for _ in range(2):
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "o.json")
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        SH.main(["x", "--langs", "pt", "--work", d, "--json", path])
                    with open(path, encoding="utf-8") as f:
                        outs.append((buf.getvalue(), f.read()))
        finally:
            SH.fetch, SH._captions = old
        self.assertEqual(outs[0], outs[1])
        js = json.loads(outs[0][1])
        self.assertEqual(js["x"]["pt"]["spans"]["all"]["WRONG_VALUE"], 1)
        self.assertEqual(js["x"]["pt"]["sym"]["ALL"]["V"], 0)


if __name__ == "__main__":
    unittest.main()
