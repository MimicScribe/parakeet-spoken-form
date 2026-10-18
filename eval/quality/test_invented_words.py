import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import invented_words as I  # noqa: E402


def lex(refs, arms=(), train=()):
    rows = {"ref": [(str(i), l, r, "") for i, (l, r) in enumerate(refs)]}
    for name, hyps in arms:
        rows[name] = [(str(i), l, None, h) for i, (l, h) in enumerate(hyps)]
    d = tempfile.mkdtemp()
    if train:
        for lang, text in train:
            with open(f"{d}/p3_{lang}.jsonl", "a") as f:
                f.write(json.dumps({"text": text}) + "\n")
    return I.build_lexicon(rows, {n for n, _ in arms}, d if train else None)


class Tokens(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(I.tokens("Es kostet 5,99 Euro <unk> — zwei-hundert."), ["es", "kostet", "euro", "zwei", "hundert"])

    def test_bucket(self):
        self.assertEqual([I.bucket(n) for n in (1, 3, 4, 8, 9, 15, 16, 30)], ["<=3", "<=3", "4-5", "6-8", "9-11", "12-15", "16+", "16+"])


class Invented(unittest.TestCase):
    def test_garbled_compound_is_invented_real_one_is_not(self):
        lx = lex([("de", "es kostet zweihundertsiebzig euro")], arms=[("B", [("de", "es kostet zweihundertsiebzig euro")])])
        rows = [("1", "de", "es kostet zweihundertsiebzig euro", "es kostet zweiziebzig euro"),
                ("2", "de", "es kostet zweihundertsiebzig euro", "es kostet zweihundertsiebzig euro")]
        r = I.score_arm(rows, lx, accept_compounds=False)["de"]
        self.assertEqual((r["sentences"], r["sent_invented"], r["invented"]), (2, 1, 1))
        self.assertEqual(r["top_invented"], [("zweiziebzig", 1)])

    def test_split_morphemes_are_valid_words(self):  # red-on-base: control (a split morpheme is never a new word)
        lx = lex([("de", "es kostet euro")])
        for lang, hyp in (("de", "zwei hundert siebzig und vier"), ("nl", "vier en twintig"), ("it", "cento venti cinque")):
            lx2 = lex([(lang, "x")])
            self.assertEqual(I.score_arm([("1", lang, None, hyp)], lx2)[lang]["invented"], 0, hyp)
        self.assertEqual(I.score_arm([("1", "de", None, "zwei hundert siebzig")], lx)["de"]["invented"], 0)
        # the first-pass method (no morphemes) calls "hundert" and "siebzig" invented when nothing else lists them
        bare = I.build_lexicon({"r": [("1", "de", "x", "")]}, set(), None, with_morphemes=False)
        self.assertEqual(I.score_arm([("1", "de", None, "hundert siebzig")], bare, accept_compounds=False)["de"]["invented"], 2)

    def test_training_text_is_lexicon(self):
        lx = lex([("de", "x")], train=[("de", "wir haben zweihundertsiebzig mitarbeiter")])
        self.assertEqual(I.score_arm([("1", "de", None, "zweihundertsiebzig mitarbeiter")], lx)["de"]["invented"], 0)

    def test_digits_are_not_words(self):
        lx = lex([("de", "x")])
        self.assertEqual(I.score_arm([("1", "de", None, "2undzwanzig 270 euro blorf")], lx)["de"]["invented"], 1)  # only "blorf"

    def test_accept_compounds(self):
        lx = lex([("de", "x")])
        r = lambda ac: I.score_arm([("1", "de", None, "zweihundertsiebzig zweiziebzig")], lx, accept_compounds=ac)["de"]["invented"]  # noqa: E731
        self.assertEqual((r(False), r(True)), (2, 1))

    def test_per_converted_number_and_length(self):
        lx = lex([("de", "es kostet 270 euro")], arms=[("B", [("de", "es kostet zweihundertsiebzig euro")])])
        rows = [("1", "de", "es kostet 270 euro", "es kostet zwanzigtausend euro"),
                ("2", "de", "es kostet 270 euro", "es kostet zwei hundert siebzig euro")]
        r = I.score_arm(rows, lx, accept_compounds=False)["de"]
        self.assertEqual(r["words_chains"], 2)
        self.assertEqual(r["converted"], 1)  # the wrong-value chain is words but not converted
        self.assertEqual(r["converted_old"], 2)
        self.assertEqual(r["invented_per_converted"], 1.0)
        self.assertEqual(r["rate_by_length"]["12-15"], {"n": 1, "invented": 1, "rate": 1.0})

    def test_loo_judges_a_lexicon_arm_without_its_own_words(self):
        lx = lex([("de", "x")], arms=[("B", [("de", "viernevenig")])])
        self.assertEqual(I.score_arm([("1", "de", None, "viernevenig")], lx, "B", loo=True)["de"]["invented"], 0)
        self.assertEqual(I.score_arm([("1", "de", None, "viernevenig")], lx, "B", loo=True)["de"]["invented_loo"], 1)

    def test_loaders(self):
        d = tempfile.mkdtemp()
        open(f"{d}/m.jsonl", "w").write(json.dumps({"id": "a", "lang": "de", "text": "ref"}) + "\n")
        open(f"{d}/p.jsonl", "w").write(json.dumps({"id": "a", "hyp": "hyp"}) + "\n")
        open(f"{d}/c.jsonl", "w").write(json.dumps({"file": "f", "lang": "it", "written": "w", "long": "l", "single": "s"}) + "\n")
        man = {"a": {"id": "a", "lang": "de", "text": "ref"}}
        self.assertEqual(I.load_arm(f"{d}/p.jsonl", man), [("a", "de", "ref", "hyp")])
        self.assertEqual(I.load_arm(f"{d}/c.jsonl", None, "single"), [("f", "it", "w", "s")])
        self.assertEqual(I.load_arm(f"{d}/c.jsonl"), [("f", "it", "w", "l")])


class FunctionWords(unittest.TestCase):
    CASES = {"de": "zwei prozent komma euro cent", "nl": "twee procent komma euro cent", "it": "due per cento virgola euro centesimi",
             "es": "dos por ciento coma euros céntimos", "fr": "deux pour cent virgule euros centimes",
             "pt": "dois por cento vírgula euros cêntimos centavos"}

    def test_percent_decimal_currency_words_are_valid(self):  # red on base: these were invented
        for lang, hyp in self.CASES.items():
            lx = lex([(lang, "x")])
            r = I.score_arm([("1", lang, None, hyp)], lx)[lang]
            self.assertEqual(r["invented"], 0, (lang, r["top_invented"]))

    def test_function_words_not_in_other_languages(self):
        lx = lex([("de", "x"), ("es", "x")])
        self.assertEqual(I.score_arm([("1", "de", None, "centimes virgule")], lx)["de"]["invented"], 2)

    def test_compounds_accepted_by_default_and_flag_turns_it_off(self):  # red on base: default was off
        lx = lex([("de", "x")])
        rows = [("1", "de", None, "zweihundertsiebzig zweiziebzig")]
        self.assertEqual(I.score_arm(rows, lx)["de"]["invented"], 1)
        self.assertEqual(I.score_arm(rows, lx, accept_compounds=False)["de"]["invented"], 2)


class EnglishBleed(unittest.TestCase):
    def score(self, lang, hyp, refs=(), arm_hyp=None):
        lx = lex([(lang, "x")] + [(lang, r) for r in refs], arms=[("B", [(lang, arm_hyp)])] if arm_hyp else ())
        return I.score_arm([("1", lang, None, hyp)], lx)[lang]

    def test_bleed_garble_and_total_are_separate(self):  # red on base: no such fields
        rows = [("1", "de", None, "wir haben twenty euro"), ("2", "de", None, "wir haben zweiziebzig euro"),
                ("3", "de", None, "wir haben twenty zweiziebzig euro"), ("4", "de", None, "wir haben zwanzig euro")]
        r = I.score_arm(rows, lex([("de", "wir haben")]))["de"]
        self.assertEqual((r["sent_bleed"], r["sent_garble"], r["sent_bad"], r["sent_invented"]), (2, 2, 3, 3))
        self.assertEqual((r["bleed"], r["garble"]), (2, 2))

    def test_native_words_are_not_bleed(self):
        self.assertEqual(self.score("de", "elf uhr")["sent_bleed"], 0)
        self.assertEqual(self.score("it", "nove sette zero")["sent_bleed"], 0)
        self.assertEqual(self.score("nl", "ten huize")["sent_bleed"], 0)
        self.assertEqual(self.score("fr", "six point zero")["sent_bleed"], 0)
        self.assertEqual(self.score("pt", "zero")["sent_bleed"], 0)

    def test_model_decode_does_not_make_english_native(self):
        r = self.score("es", "twenty", arm_hyp="twenty")
        self.assertEqual((r["sent_bleed"], r["invented"]), (1, 0))  # known to the arm, still bleed
        self.assertEqual(r["garble"], 0)

    def test_reference_text_can_own_a_word(self):
        self.assertEqual(self.score("de", "first", refs=["the first"])["sent_bleed"], 0)

    def test_english_sentence_has_no_bleed(self):
        lx = lex([("en", "x")])
        self.assertEqual(I.score_arm([("1", "en", None, "twenty one")], lx)["en"]["sent_bleed"], 0)

    def test_bad_per_converted(self):
        lx = lex([("de", "es kostet 270 euro")])
        rows = [("1", "de", "es kostet 270 euro", "es kostet zwei hundert siebzig twenty euro")]
        r = I.score_arm(rows, lx)["de"]
        self.assertEqual((r["converted"], r["bleed"], r["bad_per_converted"]), (1, 1, 1.0))


if __name__ == "__main__":
    unittest.main()
