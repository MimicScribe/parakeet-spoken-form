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


class EdgeFix(unittest.TestCase):
    def test_fixed_before_selection(self):
        # red-on-base: the fix was applied at write time, after the caps and fills had counted the unfixed spans
        import edge_fix as EF

        vol = _fixture(n_win=0)
        rows = rj(f"{vol}/ml/p3v2/de/rows.jsonl")
        r = rows[2] = dict(rows[2], text="Es waren fünfzig Prozent heute da.", num_chars=[[9, 24]])
        wj(vol, "ml/p3v2/de/rows.jsonl", rows)
        rec = EF.plan_row(r, "es waren fünfzig", "fünfzig prozent heute", 5.0)
        self.assertEqual(rec["sides"]["end"], {"act": "trim", "k": 1})
        wj(vol, "ml/ef/de/fixes_0.jsonl", [rec])
        seen = []
        sel = MB.select_synthetic
        self.addCleanup(setattr, MB, "select_synthetic", sel)
        MB.select_synthetic = lambda tts, real, *a, **kw: seen.append([x["text"] for x in real]) or sel(tts, real, *a, **kw)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, edge_fix_dir="ml/ef")
        p3 = {x["id"]: x for x in rj(f"{vol}/mix4/p3_de.jsonl")}
        self.assertEqual(p3[r["id"]]["text"], "Es waren fünfzig Prozent heute")
        self.assertEqual(p3[r["id"]]["num_chars"], [[9, 24]])
        self.assertIn("Es waren fünfzig Prozent heute", seen[0])  # the selection saw the fixed label
        self.assertEqual((st["edge_fix"]["de"]["fixed"], st["edge_fix"]["de"]["trim"]), (1, 1))


class Inputs(unittest.TestCase):
    """extra_passes (judged, read with the language's passes), relabel_rows (source rows replaced by id), relabel_parts_dir
    (written windows replaced by id) and relabel_tts (letter+digit policy on the chosen synthetic rows)."""

    def setUp(self):
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        self.vol = _fixture(n_win=2)

    def test_inputs(self):
        v = self.vol
        rows = rj(f"{v}/ml/p3v2/de/rows.jsonl")
        new = dict(rows[5], id="de100_00000009_V9VVVVVVVVV_10_00_5_00", caption="Es waren 9 % iota Leute da.")
        wj(v, "ml/l1conv/de/rows_l1conv.jsonl", [new])
        with self.assertRaises(AssertionError):  # an extra pass is judged like every pass
            MB.build4(v, ("de",), rows_h=99, win_h=1, extra_passes=("ml/l1conv/{lang}/rows_l1conv.jsonl",))
        wj(v, "ml/l1conv/de/judged_rows_l1conv.json", [])
        rl = dict(rows[6], text="Es waren fünfzig Prozent da!")
        wj(v, "ml/alnumA/de/rows_alnumA.jsonl", [rl])
        win = rj(f"{v}/ml/p3v2/de/windows.jsonl")
        wj(v, "m8a/de/changed_win_de.jsonl", [dict(win[0], text="FÜNFZIG Prozent da")])
        alnum = _syn(v, "tts1", 0, text="Werte von ge zwanzig heute", num_chars=[[10, 20]],
                     spans=[{"kind": "alnum", "text": "G20", "reading": "ge zwanzig"}])
        wj(v, "ml/tts1/de/rows.jsonl", [alnum])
        for f in ("judged_rows.json", "judged_tts_judge.json"):
            wj(v, f"ml/p3v2/de/{f}", [])
        st = MB.build4(v, ("de",), rows_h=99, win_h=1, extra_passes=("ml/l1conv/{lang}/rows_l1conv.jsonl",),
                       relabel_rows="ml/alnumA/{lang}/rows_alnumA.jsonl", relabel_parts_dir="m8a", relabel_tts=True)
        p3 = {x["id"]: x for x in rj(f"{v}/mix4/p3_de.jsonl")}
        self.assertIn(new["id"], p3)
        self.assertEqual(st["extra_passes"]["de"], {"rows_l1conv.jsonl": 1})
        self.assertEqual(st["relabel_rows"]["de"], {"file": 1, "replaced": 1})
        self.assertTrue(p3[rl["id"]]["text"].endswith("!"))
        w = {x["id"]: x for x in rj(f"{v}/mix4/win_de.jsonl")}
        self.assertEqual(w[win[0]["id"]]["text"], "FÜNFZIG Prozent da")
        tts = rj(f"{v}/mix4/tts_de.jsonl")
        self.assertTrue(tts and tts[0]["text"] == "Werte von G zwanzig heute", tts[:1])


if __name__ == "__main__":
    unittest.main()
