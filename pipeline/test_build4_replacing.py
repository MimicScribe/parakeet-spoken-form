"""build4 review items (2026-10-18): a refused replacing synthetic row (TTS_REPLACING, the committed set name ttsRV1b)
brings its superseded rows back and the selection reruns (fatal after REPLACE_ROUNDS); the Italian split form is
recorded in the stats and "it+" / "it-" force it."""
import unittest

import compound_split as CS
import mix_build as MB
from test_census_common import _fixture, rj, wj


def _syn(vol, tag, k, **kw):
    return dict({"id": f"{tag}_de_{k}", "text": "fünfzig Prozent da", "duration": 2.0, "src": "tts", "origin": "tts",
                 "audio_filepath": "t.wav", "split": "train", "num_chars": [[0, 15]],
                 "spans": [{"kind": "pct", "text": "50 %", "by": "tts"}]}, **kw)


class Replacing(unittest.TestCase):
    def setUp(self):
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0  # no degraded copies (no audio in the fixture)
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        self.vol = _fixture(n_win=0)
        wj(self.vol, "ml/tts1/de/rows.jsonl", [_syn(self.vol, "tts1", 0)])
        wj(self.vol, "ml/ttsRV1b/de/rows.jsonl", [_syn(self.vol, "ttsRV1b", 0, supersedes=["tts1_de_0"])])
        self.sel = MB.select_synthetic
        self.addCleanup(setattr, MB, "select_synthetic", self.sel)

    def refuse(self, rounds):
        n = {"calls": 0}

        def sel(tts, *a, **kw):  # the caps refuse every replacing row in the first `rounds` selections
            n["calls"] += 1
            got = self.sel(tts, *a, **kw)
            return [x for x in got if not (x.get("_set") in MB.TTS_REPLACING and n["calls"] <= rounds)]
        MB.select_synthetic = sel
        return n

    def test_admitted_replaces(self):
        st = MB.build4(self.vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        ids = {x["id"] for x in rj(f"{self.vol}/mix4/tts_de.jsonl")}
        self.assertIn("ttsRV1b_de_0", ids)
        self.assertNotIn("tts1_de_0", ids)  # superseded
        self.assertEqual(st["synthetic_replaced"]["de"]["refused_restored"], 0)

    def test_refused_restores_superseded(self):
        # red-on-base: the refusal only warned; tts1_de_0 stayed out and the class lost a row
        n = self.refuse(1)
        st = MB.build4(self.vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        ids = {x["id"] for x in rj(f"{self.vol}/mix4/tts_de.jsonl")}
        self.assertIn("tts1_de_0", ids)
        self.assertNotIn("ttsRV1b_de_0", ids)
        self.assertEqual(n["calls"], 2)
        self.assertEqual(st["synthetic_replaced"]["de"]["refused_restored"], 1)

    def test_italian_form_recorded(self):
        for sl, want in ((("de", "it"), CS.IT_SPLIT_ELIDED), (("it+",), True), (("de", "it-"), False)):
            st = MB.build4(self.vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, split_langs=sl)
            self.assertIs(st["italian_split_elided"], want, sl)
        st = MB.build4(self.vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, split_langs=("de",))
        self.assertNotIn("italian_split_elided", st)
        self.assertTrue(CS.IT_SPLIT_ELIDED)  # default: always split


if __name__ == "__main__":
    unittest.main()
