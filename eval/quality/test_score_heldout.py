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
