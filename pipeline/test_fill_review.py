"""Fill-to-bar review fixes (2026-10-17): order-independent value fill, the value-fill share cap, the rare-class floor
in the build and the gate alike, synthetic rows trimmed (never real ones) after selection, the held-out voice set,
loud missing inputs and a fatal final check. Every test is red on the commit before them unless marked.
Needs numpy (build4): python3 test_fill_review.py"""
import importlib.util
import os
import random
import tempfile
import unittest
from collections import Counter

import census_common as CC
import mix_build as MB
from test_census_common import rj, wj


def data_gate():
    spec = importlib.util.spec_from_file_location("data_gate", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "eval", "quality", "data_gate.py"))
    DG = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DG)
    return DG


def syn(i, kind, text, dur=2.0, tag="tts", **kw):
    return dict({"id": f"{tag}_{i}", "text": "eine Zahl hier", "src": "tts", "_targeted": True, "duration": dur,
                 "spans": [{"kind": kind, "text": text}]}, **kw)


def real(i, kind, text, by="ctc", dur=2.0, lang="de"):
    return {"id": f"r{kind}{i}", "text": "eine Zahl hier", "duration": dur, "lang": lang, "num_chars": [[0, 4]],
            "spans": [{"kind": kind, "text": text, "by": by}]}


def volume(rows, lang="de", sets=None):
    """A build4 volume: real rows (judged, none held out / rejected), synthetic sets {tag: rows}."""
    vol = tempfile.mkdtemp()
    for p in ("mix3/d2.jsonl", "mix3/d2p.jsonl", "mix3/fleurs_train_multi.jsonl"):
        wj(vol, p, [])
    base = f"ml/p3v2/{lang}"
    wj(vol, f"{base}/rows.jsonl", rows)
    wj(vol, f"{base}/judged_rows.json", [])
    wj(vol, f"{base}/windows.jsonl", [])
    open(f"{vol}/ml/p3v2/heldout_{lang}.txt", "w").write("")
    open(f"{vol}/{base}/reject_ids.txt", "w").write("")
    for tag, rs in (sets or {}).items():
        wj(vol, f"ml/{tag}/{lang}/rows.jsonl", [dict(x, audio_filepath="t.wav", origin="tts", num_chars=[[0, 4]],
                                                     split=x.get("split", "train")) for x in rs])
    return vol


class NoAug(unittest.TestCase):
    def setUp(self):
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0  # no degraded copies (no audio in the fixtures)
        augl, MB.TTS_AUG_FRAC_LONG = MB.TTS_AUG_FRAC_LONG, 0.0
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC_LONG", augl)


class ValueFillOrder(unittest.TestCase):
    def test_thin_years_fill_in_any_order(self):
        # red-on-base: sv years (380 real, all 1980) filled 41/41 values in the input order but 0/41 shuffled: the
        # first non-needed spans used the share room and locked the thin values out
        realr = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "int", "text": "1980"}]} for i in range(380)]
        tts = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True,
                "spans": [{"kind": "int", "text": str(1990 + i % 41)}]} for i in range(41 * 12)]
        tts += [{"id": f"u{i}", "text": "x", "src": "tts", "_targeted": True,
                 "spans": [{"kind": "int", "text": "1985"}]} for i in range(300)]  # a value with no floor
        for seed in range(4):
            order = list(tts)
            random.Random(seed).shuffle(order)
            took = MB.select_synthetic(order, realr, "sv", {"int": 9.0})
            vals = Counter(int(x["spans"][0]["text"]) for x in took)
            self.assertEqual(sum(vals[v] >= 8 for v in range(1990, 2031)), 41, (seed, vals))

    def test_int_big_distinct_in_any_order(self):
        # red-on-base: the int_big distinct bar (200) had the same lock-out (one repeated value taken first)
        realr = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "int", "text": "5000"}]} for i in range(400)]
        tts = [{"id": f"d{i}", "text": "x", "src": "tts", "_targeted": True,
                "spans": [{"kind": "int", "text": str(10000 + 7 * i)}]} for i in range(250)]
        tts += [{"id": f"s{i}", "text": "x", "src": "tts", "_targeted": True,
                 "spans": [{"kind": "int", "text": "7000"}]} for i in range(300)]
        random.Random(1).shuffle(tts)
        took = MB.select_synthetic(tts, realr, "de", {"int": 9.0})
        distinct = {x["spans"][0]["text"] for x in took} | {"5000"}
        self.assertGreaterEqual(len(distinct), 200)

    def test_trim_keeps_value_fill_rows(self):
        # red-on-base: trim_synthetic dropped the last chosen rows whatever they held, value-fill rows included
        realr = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "int", "text": "500"}]} for i in range(3000)]
        need = [{"id": f"n{i}", "text": "x", "src": "tts", "spans": [{"kind": "int", "text": str(i % 101)}]}
                for i in range(1010)]
        other = [{"id": f"o{i}", "text": "x", "src": "tts", "spans": [{"kind": "int", "text": "777"}]} for i in range(1285)]
        took = other + need  # the value-fill rows last: the old trim's first victims
        self.assertEqual(MB.trim_synthetic(took, realr, "de")[1], 0)
        kept, n = MB.trim_synthetic(took, realr[:2600], "de")  # room 1114 + 1010 value fill < 2295 chosen
        self.assertEqual(n, 171)
        vals = Counter(int(x["spans"][0]["text"]) for x in kept)
        self.assertTrue(all(vals[v] >= 10 for v in range(101)), vals)
        self.assertEqual(vals[777], 1114)


class ValueFillCap(unittest.TestCase):
    def test_value_fill_capped_at_half_when_the_bar_is_met(self):
        # red-on-base: value floors took a class whose real spans met its bar far over the cap (de years: 48.7%);
        # interim rule: the value fill there stays <= VALUE_FILL_MAX_SHARE of the class
        self.assertEqual(CC.VALUE_FILL_MAX_SHARE, 0.5)
        sv = Counter({v: 40 for v in range(1990, 2031)})  # 1640 synthetic spans of thin values
        self.assertEqual(CC.syn_allowed("pt", "year", 1500, Counter({1985: 1500}), sv), 1500)  # 50%, not 1640
        self.assertFalse(CC.syn_ok("pt", "year", 1640, 3140, real_vals=Counter({1985: 1500}), syn_vals=sv))
        # a class under its bar: the bar room plus the full value fill, no cap
        self.assertEqual(CC.syn_allowed("pt", "year", 100, Counter({1985: 100}), sv), 1400 + 1640)

    def test_thin_values_fill_on_top_of_a_class_at_its_bar(self):
        # red-on-base: ru years (55 real, bar 825) stopped at exactly 770 synthetic from spread 1900-1989 years: at the
        # bar a row was only taken if EVERY synthetic span filled a thin value, so one non-thin span froze the class
        realr = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "int", "text": "1980"}]} for i in range(55)]
        spread = [{"id": f"s{i}", "text": "x", "src": "tts", "_targeted": True,
                   "spans": [{"kind": "int", "text": str(1900 + i % 90)}]} for i in range(900)]
        thin = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True,
                 "spans": [{"kind": "int", "text": str(1990 + i % 41)}]} for i in range(41 * 25)]
        for tts in (spread + thin, thin + spread):
            took = MB.select_synthetic(tts, realr, "ru", {"int": 9.0})
            vals = Counter(int(x["spans"][0]["text"]) for x in took)
            # every thin value to its floor (820 spans) on top of the class allowance (bar - real = 770), which the
            # spread years and the thin values' spans over their floors share
            self.assertTrue(all(vals[v] >= 20 for v in range(1990, 2031)), vals)
            self.assertEqual(len(took), 770 + 41 * 20)


class RareFloor(unittest.TestCase):
    def test_floor_only_for_classes_without_a_bar(self):
        # red-on-base: the rare-class floor (0.15 h) applied to classes with a bar (de unit, bar 150) and to bg
        realr = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "unit", "text": "5 km"}]} for i in range(5)]
        tts = [syn(i, "unit", f"{i} km", dur=1.0) for i in range(600)]
        self.assertEqual(len(MB.select_synthetic(tts, realr, "de", {"unit": 0.01})), 145)  # fill to the 150 bar
        amp = [syn(i, "amp", "AT&T", dur=6.5) for i in range(120)]
        self.assertEqual(len(MB.select_synthetic(amp, [], "bg", {})), 0)  # bg keeps the share caps exactly
        self.assertEqual(len(MB.select_synthetic(amp, [], "de", {})), 83)  # no bar: the floor (0.15 h)
        self.assertFalse(CC.rare_floor_ok("de", "unit", 10, 5, 0.01))
        self.assertTrue(CC.rare_floor_ok("de", "amp", 10, 5, 0.01))

    def test_gate_applies_the_same_waiver(self):
        # red-on-base: the build kept a no-bar class at the floor, the gate failed its synthetic share
        DG = data_gate()
        d = tempfile.mkdtemp()
        wj(d, "p3_de.jsonl", [{"id": f"r{i}", "text": "x", "duration": 3.0, "spans": [{"kind": "pct", "text": "5 %"}]}
                              for i in range(40)])
        wj(d, "tts_de.jsonl", [{"id": f"t{i}", "text": "x", "duration": 6.5, "spans": [{"kind": "amp", "text": "&"}]}
                               for i in range(83)])
        c = [x for x in DG.gate_lang(d, "de")["checks"] if x["check"].startswith("amp synthetic")]
        self.assertTrue(c[0]["pass"], c)
        self.assertIn("waived floor", str(c[0]["got"]))
        wj(d, "tts_de.jsonl", [{"id": f"t{i}", "text": "x", "duration": 6.5, "spans": [{"kind": "amp", "text": "&"}]}
                               for i in range(84)])  # 0.152 h: over the floor
        c = [x for x in DG.gate_lang(d, "de")["checks"] if x["check"].startswith("amp synthetic")]
        self.assertFalse(c[0]["pass"], c)


class BuildGateAgree(NoAug):
    def test_gate_agrees_with_the_build(self):
        # red-on-base: the gate failed the rare-class floor the build allowed (amp), so the two disagreed
        rows = [real(i, "pct", "5 %") for i in range(40)]
        sets = {"tts5y": [syn(i, "amp", "AT&T", dur=6.5, tag="a") for i in range(120)]
                + [syn(i, "range", f"{i}-{i + 5}", tag="g") for i in range(200)]
                + [syn(i, "pct", f"{i % 90} %", tag="p") for i in range(100)]}
        vol = volume(rows, sets=sets)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        chk = st["post_build_check"]["de"]
        g = data_gate().gate_lang(f"{vol}/mix4", "de")
        gate_syn = {c["check"].split()[0]: c["pass"] for c in g["checks"] if " synthetic share" in c["check"]}
        self.assertEqual(set(gate_syn), {"amp", "range", "pct"})
        for k, ok in gate_syn.items():
            self.assertEqual(ok, k not in chk["synthetic_over"], (k, chk))
        self.assertTrue(all(gate_syn.values()), gate_syn)
        self.assertIn("amp", chk["waived_synthetic_floor"])


class SyntheticNeverDisplacesReal(NoAug):
    def test_post_selection_steps_trim_synthetic_rows(self):
        # red-on-base: 5 synthetic unit spans took the class (15 real, 5 of them ext) over the 20-span ext floor and
        # the second ext cap removed a REAL ext row; the year trim then ran with synthetic 2020s in the share and
        # removed real 2020 rows too
        rows = [real(i, "int", str(1950 + i % 100)) for i in range(1000)]
        rows += [real(1000 + i, "int", "2020") for i in range(18)]  # 2020: 28 of 1018 (2.75%)
        rows += [real(i, "unit", f"{i} km", by="ext" if i < 5 else "ctc") for i in range(15)]
        sets = {"tts5y": [syn(i, "int", "2020", tag="y") for i in range(10)]
                + [syn(i, "unit", f"{i} m", tag="u") for i in range(5)]}
        vol = volume(rows, sets=sets)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        p3 = {x["id"] for x in rj(f"{vol}/mix4/p3_de.jsonl")}
        self.assertEqual(p3, {x["id"] for x in rows})  # every real row kept
        tts = Counter(x["id"].split("_")[0] for x in rj(f"{vol}/mix4/tts_de.jsonl"))
        self.assertEqual(tts, {"u": 4, "y": 2})
        self.assertEqual(st["synthetic_trimmed_not_real"]["de"], {"ext_floor": {"unit": 1}, "top_year": 8})
        self.assertEqual(st["post_build_check"]["de"]["fail"], [])


class HeldoutSet(NoAug):
    def test_tts_dev_is_the_heldout_voice_after_rejects(self):
        # red-on-base: tts_dev held every dev row of every voice, judge-rejected rows included
        hv = MB.HELDOUT_VOICES["de"]
        rows = [real(i, "pct", "5 %") for i in range(40)]
        sets = {"tts1": [syn(0, "pct", "6 %", tag="hv", voice=hv, split="dev"),
                         syn(1, "pct", "6 %", tag="hv", voice=hv, split="dev"),
                         syn(2, "pct", "6 %", tag="ov", voice="de-DE-Other-Voice", split="dev")]}
        vol = volume(rows, sets=sets)
        open(f"{vol}/ml/p3v2/de/reject_ids.txt", "w").write("hv_1\n")
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        self.assertEqual([x["id"] for x in rj(f"{vol}/mix4/tts_dev_de.jsonl")], ["hv_0"])
        self.assertEqual([x["id"] for x in rj(f"{vol}/mix4/tts_dev_other_de.jsonl")], ["ov_2"])
        self.assertEqual(st["tts_dev"]["de"], {"heldout_voice": 1, "other_voices": 1, "rejected": 1})

    def test_heldout_voice_never_reaches_windows_seams_or_dedupe(self):
        # red-on-base: control (the parent rules already keep a held-out-voice row's windows / seam pairs out, and
        # dedupe protects its dev text); seam_crops itself (ml_p3) is not covered here
        hv = MB.HELDOUT_VOICES["de"]
        rows = [real(i, "pct", "5 %") for i in range(40)]
        tr = syn(0, "pct", "6 %", tag="ok", voice="de-DE-Other-Voice")
        hvt = syn(1, "pct", "7 %", tag="hvt", voice=hv)
        dev = dict(syn(2, "pct", "8 %", tag="hvd", voice=hv, split="dev"), text="acht Prozent sagte sie heute")
        dup = dict(syn(3, "pct", "8 %", tag="dup", voice="de-DE-Other-Voice"), text="acht Prozent sagte sie heute")
        vol = volume(rows, sets={"tts1": [tr, hvt, dev, dup]})
        w = lambda k, par: {"id": f"de_win_syn_{k}", "text": "sechs Prozent", "num_tokens": [True, True],  # noqa: E731
                            "duration": 3.0, "parents": par, "audio_filepath": "w.wav", "synthetic": True}
        wj(vol, "ml/p3v2/de/windows_syn.jsonl", [w(0, [tr["id"]]), w(1, [hvt["id"]]), w(2, [dev["id"]])])
        half = lambda p, side, par: {"id": f"{p}{side}", "pair": p, "side": side, "text": "sieben Prozent",  # noqa: E731
                                     "num_tokens": [True, True], "duration": 2.0, "parents": [par],
                                     "audio_filepath": "s.wav", "synthetic": True}
        wj(vol, "ml/p3v2/de/seam_long_syn.jsonl", [half("p0", "L", tr["id"]), half("p0", "R", tr["id"]),
                                                    half("p1", "L", hvt["id"]), half("p1", "R", hvt["id"])])
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, dedupe=True, seam_long_h=1.0)
        bad = {hvt["id"], dev["id"]}
        self.assertEqual({x["id"] for x in rj(f"{vol}/mix4/tts_de.jsonl")}, {tr["id"]})  # dup left with dedupe
        self.assertFalse([x for x in rj(f"{vol}/mix4/win_de.jsonl") if set(x["parents"]) & bad])
        self.assertTrue([x for x in rj(f"{vol}/mix4/win_de.jsonl") if x["parents"] == [tr["id"]]])
        sl = rj(f"{vol}/mix4/seam_long_de.jsonl")
        self.assertEqual({x["pair"] for x in sl}, {"seam_long_syn:p0"})


class LoudInputs(NoAug):
    def test_missing_sets_and_scan_lists_recorded(self):
        # red-on-base: a set with no rows for the language and a missing grammar scan list were silent
        rows = [real(i, "pct", "5 %") for i in range(40)]
        vol = volume(rows, sets={"tts1": [syn(0, "pct", "6 %", tag="a")]})
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        self.assertEqual(st["tts_sets"]["de"]["read"], ["tts1"])
        self.assertIn("ttsLNw", st["tts_sets"]["de"]["missing"])
        self.assertEqual(st["grammar_reject_lists"]["de"], [])
        self.assertEqual(st["grammar_reject_lists_missing"], ["de"])
        open(f"{vol}/ml/p3v2/de/grammar_reject_ids.txt", "w").write("# none\n")
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        self.assertEqual(st["grammar_reject_lists"]["de"], ["grammar_reject_ids.txt"])
        self.assertNotIn("grammar_reject_lists_missing", st)

    def test_final_check_fails_the_build(self):
        # red-on-base: a final post-build failure only printed a WARNING; the template / group caps stay warnings
        rows = [real(i, "pct", "5 %") for i in range(40)]
        vol = volume(rows)
        orig = MB.post_build_check

        def fake(extra):
            def f(*a, **k):
                r = orig(*a, **k)
                return dict(r, fail=r["fail"] + [extra])
            return f
        MB.post_build_check = fake("pct synthetic share 0.9 > 0.3")
        self.addCleanup(setattr, MB, "post_build_check", orig)
        with self.assertRaises(AssertionError):
            MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        for waived in ("rows over the template cap 3", "groups over the cap 2"):
            MB.post_build_check = fake(waived)
            MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)


class LongSets(unittest.TestCase):
    def test_long_sets_degraded_more(self):
        # red-on-base: control (replaces the source-grep test: the build degrades a long set's rows at
        # TTS_AUG_FRAC_LONG and the others at TTS_AUG_FRAC; behaviour since the long-number top-up)
        class Stub:
            def __init__(self, *a):
                pass

            def __call__(self, inp, out):
                return "stub"
        deg, MB.Degrader = MB.Degrader, Stub
        self.addCleanup(setattr, MB, "Degrader", deg)
        rows = [real(i, "pct", "5 %") for i in range(40)]
        vol = volume(rows, sets={"ttsLNw": [syn(i, "unit", f"{i} km", tag="L") for i in range(150)],
                                 "tts2": [syn(i, "range", f"{i}-{i + 3}", tag="T") for i in range(150)]})
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        took = rj(f"{vol}/mix4/tts_de.jsonl")
        frac = {t: sum(1 for x in took if x["id"].startswith(t + "_") and x.get("aug") == "stub")
                / sum(1 for x in took if x["id"].startswith(t + "_")) for t in ("L", "T")}
        self.assertGreater(frac["L"], frac["T"] + 0.05, frac)
        self.assertAlmostEqual(frac["L"], MB.TTS_AUG_FRAC_LONG, delta=0.08)
        self.assertAlmostEqual(frac["T"], MB.TTS_AUG_FRAC, delta=0.08)
        self.assertFalse([x for x in took if "_set" in x or "_targeted" in x])
        self.assertEqual(MB.TTS_AUG_FRAC_LONG, 0.75)
        for t in MB.TTS_LONG_SETS:
            self.assertIn((t, f"{t}_judge"), MB.TTS_SETS)
            self.assertIn(t, MB.TTS_TARGETED)


if __name__ == "__main__":
    unittest.main()
