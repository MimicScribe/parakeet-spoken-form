"""Seam pairs / windows from the synthetic sets' train rows (ml_p3.syn_seam_crops; build4 seam_long / windows).
No GPU, no Modal: python3 test_syn_seam.py"""
import os
import unittest

import mix_build as MB
import ml_p3_core as C
from test_census_common import _fixture, rj, wj

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = open(os.path.join(HERE, "ml_p3.py")).read()  # ml_p3 needs modal: read its source


def _syn_rows(lang="de"):
    hv = MB.HELDOUT_VOICES[lang]
    base = {"text": "fünfzig Prozent da", "duration": 2.0, "src": "tts", "origin": "tts1/x", "audio_filepath": "t.wav",
            "split": "train", "num_chars": [[0, 15]], "spans": [{"kind": "pct", "text": "50 %", "by": "tts"}],
            "voice": f"{lang}-DE-Chirp3-HD-Puck"}
    return [dict(base, id="ttsYRm_de_ok"), dict(base, id="ttsYRm_de_hv", voice=hv),
            dict(base, id="ttsYRm_de_hvdev", voice=hv, split="dev"), dict(base, id="ttsYRm_de_dev", split="dev"),
            dict(base, id="ttsYRm_de_rej"), dict(base, id="ttsYRm_de_gram")]


def _pair(pid, parent):
    return [{"id": f"de_seam_long_syn_{pid}_{s[0]}", "pair": pid, "side": s, "pos": "gap1", "pos_class": "gap",
             "kind": "pct", "text": "fünfzig Prozent" if s == "left" else "Prozent da",
             "num_tokens": [True, True] if s == "left" else [True, False],
             "duration": 3.0, "parents": [parent], "lang": "de", "synthetic": True, "audio_filepath": "s.wav"}
            for s in ("left", "right")]


class TrainRows(unittest.TestCase):
    def test_heldout_voice_dev_and_rejects_never_enter(self):
        # red-on-base: control (new filter for the synthetic seam / window pass)
        rows = _syn_rows()
        kept, drop = C.synthetic_train_rows({"ttsYRm": rows}, {"ttsYRm_de_rej", "ttsYRm_de_gram"},
                                            MB.HELDOUT_VOICES["de"])
        self.assertEqual([r["id"] for r in kept], ["ttsYRm_de_ok"])
        self.assertEqual(drop, {"dev_split": 2, "heldout_voice": 1, "rejected": 2})
        self.assertTrue(all(r["origin"] == "tts" and r["src"] == "tts" for r in kept))

    def test_pass_uses_the_filter_and_never_overwrites(self):
        # red-on-base: control (source checks of the new windows2_lang branch: modal is not importable here)
        src = P3[P3.index("def windows2_lang("):P3.index("def windows2(")]
        for s in ("C.synthetic_train_rows(sets, rej_syn, heldout_voice)", 'grammar_reject_ids_*.txt',
                  'judged_{marker}.json', "assert not os.path.exists(dest_)", "include_tts = False",
                  'r.get("voice") != heldout_voice'):
            self.assertIn(s, src)
        ep = P3[P3.index("def syn_seam_crops("):]
        self.assertIn("heldout_voices.get(l", ep)
        self.assertIn("seam_min_words=min_words", ep)

    def test_sets_and_consts(self):
        # red-on-base: control
        sets, hv = C.mix_build_consts()
        self.assertEqual(sets, tuple(MB.TTS_SETS))
        self.assertEqual(hv, MB.HELDOUT_VOICES)
        tm = C.syn_set_markers("ttsLNw,ttsLNr,ttsLNm,ttsYR*,ttsHO*", sets)
        self.assertIn("ttsLNw:ttsLNw_judge", tm)
        self.assertIn("ttsYRm:ttsYRm_judge", tm)
        self.assertFalse([t for t in tm if t.startswith("ttsHO") or t.startswith("tts1:")])


class Build4(unittest.TestCase):
    def setUp(self):
        a, b = MB.TTS_AUG_FRAC, MB.TTS_AUG_FRAC_LONG  # no degraded copies (no audio in the fixture)
        MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC_LONG = 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", a)
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC_LONG", b)

    def _vol(self):
        vol = _fixture(n_win=0)
        rows = _syn_rows()
        wj(vol, "ml/ttsYRm/de/rows.jsonl", rows)
        open(f"{vol}/ml/p3v2/de/reject_ids.txt", "w").write("ttsYRm_de_rej\n")
        open(f"{vol}/ml/p3v2/de/grammar_reject_ids.txt", "w").write("ttsYRm_de_gram\n")
        pairs = []
        for r in rows:
            pairs += _pair(r["id"].split("_")[-1], r["id"])
        wj(vol, "ml/p3v2/de/seam_long_syn.jsonl", pairs)
        win = lambda k, par: {"id": f"de_win_syn_w{k}", "text": "fünfzig Prozent da", "num_tokens": [True, True, False],  # noqa: E731
                              "duration": 3.0, "parents": [par], "audio_filepath": "w.wav", "synthetic": True,
                              "cut_number": True}
        wj(vol, "ml/p3v2/de/windows_syn.jsonl", [win(k, r["id"]) for k, r in enumerate(rows)])
        return vol

    def test_synthetic_long_pairs_and_windows_are_picked_up(self):
        # red-on-base: build4 read only seam_long.jsonl and windows.jsonl (synthetic seam pairs never trained)
        vol = self._vol()
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, seam_long_h=6.0)
        took = {x["id"] for x in rj(f"{vol}/mix4/tts_de.jsonl")}
        self.assertEqual(took, {"ttsYRm_de_ok"})
        sl = rj(f"{vol}/mix4/seam_long_de.jsonl")
        self.assertEqual(sorted({x["pair"] for x in sl}), ["seam_long_syn:ok"])  # whole pair, namespaced
        self.assertEqual(len(sl), 2)
        self.assertEqual(st["seam_long_synthetic"]["de"]["dropped"], {"parent_not_chosen": 3, "unsafe_parent": 2})
        self.assertEqual(st["langs"]["de"]["seam_long_rows"], 2)
        win = [w["id"] for w in rj(f"{vol}/mix4/win_de.jsonl")]
        self.assertEqual(win, ["de_win_syn_w0"])
        self.assertEqual(st["syn_set_windows"]["de"]["taken"], 1)
        # counts as synthetic: the gate's real window hours leave it out, the share code sees synthetic rows
        import census_common as CC
        self.assertTrue(all(CC.synthetic(x) for x in sl))
        self.assertEqual(MB.number_share_of("mix4/seam_long_de.jsonl", sl), 1.0)

    def test_heldout_voice_and_dev_parents_never_train(self):
        # red-on-base: build4 never read synthetic long seam pairs (no part to check)
        vol = self._vol()
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, seam_long_h=6.0)
        bad = {"ttsYRm_de_hv", "ttsYRm_de_hvdev", "ttsYRm_de_dev", "ttsYRm_de_rej", "ttsYRm_de_gram"}
        for part in ("seam_long_de", "win_de", "tts_de"):
            for x in rj(f"{vol}/mix4/{part}.jsonl"):
                self.assertFalse(set(x.get("parents", [x["id"]])) & bad, (part, x["id"]))
        self.assertTrue(rj(f"{vol}/mix4/seam_long_de.jsonl"))

    def test_real_topup_file_joins_seam_long(self):
        # red-on-base: a seam_long top-up run (seam_long_b) was never read
        vol = self._vol()
        real = [x["id"] for x in rj(f"{vol}/ml/p3v2/de/rows.jsonl")]
        main = [dict(h, synthetic=False, id=h["id"] + "m") for h in _pair("l0", real[0])]
        top = [dict(h, synthetic=False, id=h["id"] + "t", pos="gap0", text=h["text"] + " x") for h in _pair("s0", real[1])]
        top += [dict(h, pair="s1") for h in main]  # a copy of the main file's pair: dropped
        top += [dict(h, synthetic=False, pair="s2", text=h["text"] + " y") for h in _pair("s2", real[7])]  # ext parent
        wj(vol, "ml/p3v2/de/seam_long.jsonl", main)
        wj(vol, "ml/p3v2/de/seam_long_b.jsonl", top)
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, seam_long_h=6.0)
        pairs = sorted({x["pair"] for x in rj(f"{vol}/mix4/seam_long_de.jsonl")})
        self.assertEqual(pairs, ["l0", "seam_long_b:s0", "seam_long_syn:ok"])

    def test_keep_syn_pairs_cap_out(self):
        # red-on-base: control
        rows = _pair("a", "x1") + _pair("b", "x2") + _pair("c", "x3")[:1]
        kept, st = MB.keep_syn_pairs(rows, bad=set(), cap_out={"x2"}, took_ids={"x1", "x2", "x3"})
        self.assertEqual({x["pair"] for x in kept}, {"a"})
        self.assertEqual(st, {"cap_out": 1, "incomplete_or_not_synthetic": 1})


if __name__ == "__main__":
    unittest.main()
