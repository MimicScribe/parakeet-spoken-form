"""score_heldout: a missing language fails no-damage; reference fetches fail closed; --mix-dir. python3 test_score_heldout.py"""
import os
import unittest
from pathlib import Path

import score_heldout as S
from test_volume_io import FakeVolume


def sc(wr, dr, wrong, drop):
    return {"spans": {"all": {"WORDS_RIGHT_VALUE": wr, "DIGITS_RIGHT_VALUE": dr, "WRONG_VALUE": wrong, "DROPPED": drop}},
            "forms": {"converted": wr, "digits_left": dr, "n": wr + dr}}


class NoDamage(unittest.TestCase):
    def test_missing_language_fails(self):
        # red-on-base: a language the run (or the baseline) had no scores for was skipped, so the run "passed"
        base = {"pl": sc(80, 10, 5, 5), "ro": sc(80, 10, 5, 5)}
        run = {"pl": sc(85, 5, 5, 5)}
        nd = S.no_damage(run, base)
        self.assertTrue(nd["pl"]["pass"])
        self.assertFalse(nd["ro"]["pass"])
        self.assertEqual(nd["ro"]["missing"], "run")
        nd2 = S.no_damage(run, {"pl": base["pl"]}, langs=["pl", "uk"])
        self.assertEqual(nd2["uk"], {"pass": False, "missing": "both"})

    def test_pick_skips_a_run_missing_a_language(self):
        out = {"B": {"pl": sc(80, 10, 5, 5), "ro": sc(80, 10, 5, 5)},
               "full": {"pl": sc(85, 5, 5, 5), "ro": sc(85, 5, 5, 5)},
               "partial": {"pl": sc(95, 0, 3, 2)}}
        pk = S.pick_most_converted(out, "B")
        self.assertEqual(pk["pick"], "full")
        self.assertEqual(pk["failed"]["partial"], ["ro"])


class SmallNAndCI(unittest.TestCase):
    def test_report_only_under_min_spans(self):
        # n is reported per language; under MIN_SPANS spans the language is report-only
        nd = S.no_damage({"pl": sc(60, 10, 20, 10), "de": sc(600, 100, 200, 100)},
                         {"pl": sc(80, 10, 5, 5), "de": sc(800, 100, 50, 50)})
        self.assertEqual((nd["pl"]["spans"], nd["pl"]["report_only"]), (100, False))
        nd = S.no_damage({"pl": sc(30, 5, 10, 5)}, {"pl": sc(40, 5, 3, 2)})
        self.assertTrue(nd["pl"]["report_only"])
        self.assertFalse(nd["pl"]["pass"])  # the point verdict is still computed
        out = {"B": {"pl": sc(40, 5, 3, 2)}, "x": {"pl": sc(30, 5, 10, 5)}}
        self.assertIsNone(S.pick_most_converted(out, "B")["pick"])  # legacy: fails
        self.assertEqual(S.pick_most_converted(out, "B", honour_report_only=True)["failed"]["x"], [])

    @staticmethod
    def ps(n, right_words, right_digits, rows=40):
        """n spans over `rows` rows: the first right_words WORDS_RIGHT, then right_digits DIGITS_RIGHT, rest WRONG."""
        out = []
        for i in range(n):
            o = ("WORDS_RIGHT_VALUE" if i < right_words else "DIGITS_RIGHT_VALUE" if i < right_words + right_digits
                 else "WRONG_VALUE")
            out.append((f"r{i % rows}", f"t{i}", "int", o))
        return out

    def test_paired_bootstrap_harm_needs_the_whole_ci_beyond_2pp(self):
        base = self.ps(400, 0, 360)  # 90% value-right
        clear = S.paired_bootstrap(self.ps(400, 0, 280), base)  # 70%: -20 pp
        self.assertTrue(clear["harm"])
        self.assertEqual(clear["digits_right"]["diff_pp"], -20.0)
        noisy = S.paired_bootstrap(self.ps(400, 0, 350), base)  # -2.5 pp point, CI reaches above -2
        self.assertLess(noisy["value_right"]["diff_pp"], -2)
        self.assertFalse(noisy["harm"])  # the point rule would fail this language; the CI does not
        conv = S.paired_bootstrap(self.ps(400, 360, 0), base)  # digits traded for words: value-right unchanged
        self.assertFalse(conv["harm"])
        self.assertEqual(conv["words_right"]["diff_pp"], 90.0)
        self.assertTrue(S.paired_bootstrap(self.ps(60, 0, 20), self.ps(60, 0, 55))["report_only"])
        self.assertEqual(S.paired_bootstrap([], base)["n_paired"], 0)

    def test_pick_rule_ci(self):
        sp = lambda w, d, x, dr, extra: {**sc(w, d, x, dr), **extra}  # noqa: E731
        out = {"B": {"pl": sc(360, 0, 20, 20)},
               "x": sp(350, 0, 25, 25, {"_no_damage_ci": {}})}
        # point rule fails (-2.5 pp); ci rule with a non-harm verdict passes
        out["x"] = {"pl": sc(350, 0, 25, 25), "_no_damage_ci": {"pl": {"harm": False}}}
        self.assertEqual(S.pick_most_converted(out, "B")["failed"]["x"], ["pl"])
        self.assertEqual(S.pick_most_converted(out, "B", rule="ci")["failed"]["x"], [])


class MinConversion(unittest.TestCase):
    def test_long_class(self):
        self.assertEqual(S.long_class("phone", "555 1234"), "phone")
        self.assertEqual(S.long_class("cur", "12,50 €"), "price")
        self.assertEqual(S.long_class("date", "8. 8. 1997"), "date")
        self.assertEqual(S.long_class("int", "330.000"), "int_big")
        self.assertEqual(S.long_class("int", "1997"), "")  # a bare year
        self.assertEqual(S.long_class("int", "25"), "")
        self.assertEqual(S.long_class("pct", "10 %"), "")

    def test_min_ranking_differs_from_mean(self):
        def lang(conv, n=100):
            return {**sc(80, 10, 5, 5), "forms_by_class": {"phone": {"converted": conv, "n": n}}}
        full = {l: lang(90) for l in S.FULL_TIER}
        spread = {l: lang(100) for l in S.FULL_TIER}
        spread["pt"] = lang(40)  # higher mean (94 vs 90), one weak language
        out = {"B": {l: sc(80, 10, 5, 5) for l in S.FULL_TIER}, "even": full, "weak": spread}
        for l in S.FULL_TIER:
            for r in ("even", "weak"):
                cv = out[r][l]["forms_by_class"]["phone"]["converted"]
                out[r][l]["forms"] = {"converted": cv, "digits_left": 100 - cv, "n": 100}
        pk = S.pick_most_converted(out, "B")
        self.assertEqual(pk["pick"], "weak")  # mean ranking (existing behaviour)
        self.assertEqual(pk["pick_min"], "even")  # min over the full tier
        self.assertEqual(pk["ranked_min_long"][-1][:3], ("weak", 40.0, "pt"))
        self.assertEqual(S.FULL_TIER, ("en", "de", "es", "fr", "it", "nl", "pt"))


class References(unittest.TestCase):
    def test_mix_dir_and_dec18(self):
        # red-on-base: the references were hard-coded to mix4/
        self.assertEqual(S.caption_src("de", "heldout", "mix4final"), "mix4final/heldout_de.jsonl")
        self.assertEqual(S.caption_src("de", "ttsdev", "mix4final"), "mix4final/tts_dev_de.jsonl")
        self.assertEqual(S.caption_src("da", "dec18", "mix4final"), "mix4/dec18_da.jsonl")  # dec18 lives in mix4 only

    def test_fetch_failures_are_errors(self):
        # red-on-base: a failed fetch gave an empty reference set / an empty validation list and scoring went on
        v = FakeVolume()
        try:
            v.write("mix4/heldout_de.jsonl", '{"id": "a", "caption": "x", "text": "x"}\n')
            v.write("ml/p3v2/de/val_vox.jsonl", '{"id": "v1"}\n')
            v.write("ml/p3v2/es/rows.jsonl", "")  # es: a language without a validation pass
            old = dict(os.environ)
            os.environ.update(v.env(fail_get="mix4/heldout_de.jsonl,ml/p3v2/de/val_vox.jsonl"))
            try:
                with self.assertRaises(Exception):
                    S._captions("de", v.tmp / "w", "heldout", "mix4")
                with self.assertRaises(Exception):
                    S._val_vox_ids("de", v.tmp / "w")
                os.environ["FAKE_FAIL_GET"] = ""
                self.assertEqual(set(S._captions("de", v.tmp / "w", "heldout", "mix4")), {"a"})
                self.assertEqual(S._val_vox_ids("de", v.tmp / "w"), {"v1"})
                self.assertEqual(S._val_vox_ids("es", v.tmp / "w"), set())  # no validation pass: empty, not an error
                with self.assertRaises(Exception):
                    S._captions("es", v.tmp / "w", "heldout", "mix4")  # no references at all: an error
            finally:
                os.environ.clear()
                os.environ.update(old)
        finally:
            v.close()


if __name__ == "__main__":
    unittest.main()
