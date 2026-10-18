"""The shared census rules (census_common) and the build steps that use them. Needs the modal + numpy packages for the
ml_p3 / build4 parts: uv run --no-project --with modal==1.6.1 --with numpy python test_census_common.py"""
import json
import os
import random
import tempfile
import unittest
from collections import Counter

import census_common as CC
import mix_build as MB


def wj(vol, p, rs):
    os.makedirs(os.path.dirname(f"{vol}/{p}"), exist_ok=True)
    with open(f"{vol}/{p}", "w") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs))


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class ExtDefinition(unittest.TestCase):
    def test_every_transcript_decision_counts(self):
        # red-on-base: ext_ctx / ext_value / ext_regional escaped every cap (only "ext" and "ext_pick" were counted)
        for by in ("ext", "ext_pick", "ext_ctx", "ext_value", "ext_regional"):
            self.assertTrue(CC.ext_span({"by": by}), by)
            self.assertTrue(CC.ext_row({"spans": [{"by": "ctc"}, {"by": by}]}), by)
        self.assertFalse(CC.ext_span({"by": "ctc"}))
        # the 16-language path: the CTC decides, the transcript's value check is a filter on top (not counted)
        self.assertFalse(CC.ext_span({"by": "ctc", "ext_value": True}))
        self.assertIs(MB.EXT_BY, CC.EXT_BY)

    def test_build_ext_cap_counts_ext_ctx(self):
        # red-on-base: 12 ext_ctx dec spans of 30 (40%) passed the per-class cap untouched
        rs = [{"id": f"x{i}", "spans": [{"kind": "dec", "text": "1,5", "by": "ext_ctx" if i < 12 else "ctc"}]}
              for i in range(30)]
        kept, st = MB.cap_ext_per_class(rs, 0.24, random.Random(0))
        self.assertLessEqual(sum(CC.ext_span(sp) for x in kept for sp in x["spans"]), 0.24 * len(kept))
        self.assertEqual(st["classes"]["dec"][0], 0.4)


class Classes(unittest.TestCase):
    def test_year_early_is_its_own_class(self):
        # red-on-base: a 1100-1799 year was int_big in the build, year_early in the gate (es / pt ext share failed)
        self.assertEqual(MB.ext_class({"kind": "int", "text": "1500"}), "year_early")
        self.assertEqual(MB.ext_class({"kind": "int", "text": "1999"}), "year")

    def test_gate_classes_with_row(self):
        r = {"text": "cuesta tres euros", "spans": []}
        sp = {"kind": "int", "text": "3", "reading": "tres"}
        self.assertEqual(MB.ext_class(sp), "int_small")
        self.assertEqual(MB.ext_class(sp, r, "es"), "cur")  # the gate counts a number + currency word as a price

    def test_ext_cap_counts_synthetic_spans(self):
        # red-on-base (no `extra`): fr amp, 10 real spans (8 ext) under the 20-span floor, + 15 synthetic spans the
        # gate counts: 8 / 25 = 0.32 failed the gate while the build's cap never looked at the class
        rs = [{"id": f"a{i}", "spans": [{"kind": "amp", "text": "&", "by": "ext" if i < 8 else "ctc"}]} for i in range(10)]
        kept, st = MB.cap_ext_per_class(rs, 0.24, random.Random(0), extra={"amp": 15})
        n_ext = sum(CC.ext_span(sp) for x in kept for sp in x["spans"])
        self.assertLessEqual(n_ext, 0.24 * (len(kept) + 15))
        self.assertEqual(st["still_over"], {})

    def test_still_over_reported(self):
        # every span is ext: the cap cannot get under the share without emptying the class; it says so
        rs = [{"id": f"a{i}", "spans": [{"kind": "dec", "text": "1,5", "by": "ext"}]} for i in range(30)]
        kept, st = MB.cap_ext_per_class(rs, 0.24, random.Random(0), min_spans=1)
        self.assertTrue(st["still_over"] or not kept)


class HeldUnit(unittest.TestCase):
    def test_day_covers_committee_sessions(self):
        # red-on-base: build4's day unit matched PLENARY only; the carve held out committee days whole
        self.assertEqual(CC.held_unit("20180618-1500-COMMITTEE-ITRE_sk_302_p0"), "day:20180618")
        self.assertEqual(CC.held_unit("20150609-0900-PLENARY-6-de_x"), "day:20150609")
        self.assertEqual(CC.held_unit("de102_00000032_qyM1XkMLEco_4_2_1_5"), "vid:qyM1XkMLEco")
        self.assertIsNone(CC.held_unit("fleurs_780_9112906313545406763_w"))


class PostBuildCheck(unittest.TestCase):
    def test_reports_the_gate_failures(self):
        # red-on-base: no post-build check
        rows = [{"id": f"20190101-0900-PLENARY-1-es_{i}", "caption": "Son 5 euros al año.", "text": "son cinco",
                 "spans": [{"kind": "dec", "text": "1,5", "by": "ext"}]} for i in range(25)]
        chk = MB.post_build_check(rows, [], "es")
        self.assertTrue(any("ext share" in f for f in chk["fail"]))
        self.assertEqual(chk["rows_over_template_cap"], 22)
        self.assertEqual(chk["groups_over_cap"], 1)


def _en_year_words(y):
    ones = "one two three four five six seven eight nine".split()
    teens = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    if 1990 <= y <= 1999:
        return "nineteen ninety" + ("" if y == 1990 else " " + ones[y - 1991])
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if y == 2000 else " and " + ones[y - 2001])
    if 2010 <= y <= 2019:
        return "twenty " + teens[y - 2010]
    if y == 2020:
        return "twenty twenty"
    raise ValueError(y)


class EnglishTopYear(unittest.TestCase):
    def test_capped_copies_get_the_second_top_year_trim(self):
        # red-on-base: en_balance had no top-year trim after its caps and share trim (English top year 3.1% vs 3%)
        part = MB.EN_TOPUP
        rows = [{"text": f"back in {_en_year_words(1990 + i % 30)} it was", "video": f"v{i}"} for i in range(3000)]
        rows += [{"text": f"back in {_en_year_words(2020)} it was", "video": f"w{i}"} for i in range(600)]
        flat = [(part, i) for i in range(len(rows))]
        other = Counter({y: 100 for y in range(1800, 1900)})  # the other English parts' year spans (e.g. synthetic)
        cut, st = MB.en_top_year_trim({part: rows}, set(), flat, other, MB.TOP_YEAR_SHARE - MB.TOP_YEAR_MARGIN, (part,))
        left = Counter(y for i, x in enumerate(rows) if (part, i) not in cut for y in MB._en_values(x["text"])) + other
        self.assertLessEqual(left[2020] / sum(left.values()), MB.TOP_YEAR_SHARE - MB.TOP_YEAR_MARGIN + 1e-9)
        self.assertEqual(st[2020]["before"], 600)
        self.assertEqual(list(st), [2020])
        self.assertTrue(all(left[y] == 100 for y in range(1990, 2020)))  # no other year loses a crop


class LabelRerun(unittest.TestCase):
    def test_refuses_a_finished_pass(self):
        import ml_p3 as P  # red-on-base: label2 reopened a finished pass's outputs for writing

        self.assertEqual((P.pass_suffix("rows"), P.pass_suffix("rows_x"), P.pass_suffix("x")), ("", "_x", "_x"))
        with tempfile.TemporaryDirectory() as d:
            P.refuse_rerun(d, "_x")  # nothing there yet
            open(f"{d}/stats_x.json", "w").write("{}")
            with self.assertRaises(FileExistsError):
                P.refuse_rerun(d, "_x")
            P.refuse_rerun(d, "_x", force=True)
            P.refuse_rerun(d, "_y")

    def test_pass_scoped_audio_paths(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_p3.py")).read()
        lab = src[src.index("def label2_lang("):src.index("def label2(")]
        # red-on-base: the D2 copy and the crops were written to paths shared by every pass of the language
        self.assertNotIn('f"{out}/wav/{row[\'origin\']}_{row[\'id\']}.wav"', lab)
        self.assertNotIn('f"{out}/crops/{nr[\'id\']}_c{k}.wav"', lab)
        self.assertIn('f"{wdir}/{row[\'origin\']}_{row[\'id\']}.wav"', lab)
        self.assertIn('crop_dir = f"{out}/crops/{name}"', lab)


def _fixture(langs=("de",), n_win=10):
    vol = tempfile.mkdtemp()
    wj(vol, "mix3/d2.jsonl", [])
    wj(vol, "mix3/d2p.jsonl", [])
    wj(vol, "mix3/fleurs_train_multi.jsonl", [])
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda".split()
    for lang in langs:
        base = f"ml/p3v2/{lang}"
        rs = []
        for i in range(8):
            # rows 0-3 share one caption template (non-rare pct, no thin value): the template cap takes one out
            cap = "Es waren 50 % der Leute da." if i < 4 else f"Es waren {i} % {words[i]} Leute da."
            rs.append({"id": f"{lang}100_0000000{i}_V{i}VVVVVVVVV_10_00_5_00", "text": "Es waren fünfzig Prozent da.",
                       "caption": cap, "duration": 5.0, "lang": lang, "src": "yodas", "num_chars": [[9, 26]],
                       "spans": [{"kind": "pct", "text": "50 %", "by": "ext_ctx" if i == 7 else "ctc"}]})
        wj(vol, f"{base}/rows.jsonl", rs)
        wj(vol, f"{base}/judged_rows.json", [])
        open(f"{vol}/ml/p3v2/heldout_{lang}.txt", "w").write("")
        open(f"{vol}/{base}/reject_ids.txt", "w").write("")
        wins = [{"id": f"{lang}_w{k}", "text": "fünfzig Prozent da", "num_tokens": [True, True, False], "duration": 3.0,
                 "parents": [rs[k % 8]["id"]], "audio_filepath": "w.wav"} for k in range(n_win)]
        wj(vol, f"{base}/windows.jsonl", wins)
    return vol


class Build4(unittest.TestCase):
    def test_capped_and_ext_ctx_parents_leave_windows(self):
        # red-on-base: a parent the template cap removed (and an ext_ctx parent) still fed windows
        vol = _fixture(n_win=8)
        st = MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True, balance=True)
        p3 = {x["id"] for x in rj(f"{vol}/mix4/p3_de.jsonl")}
        win = rj(f"{vol}/mix4/win_de.jsonl")
        self.assertEqual(st["balance_caps"]["de"]["template_out"], 1)
        self.assertTrue(win)
        for w in win:
            self.assertTrue(set(w["parents"]) <= p3, (w["id"], w["parents"]))
        self.assertFalse(any(w["parents"][0].startswith("de100_00000007") for w in win))  # the ext_ctx parent
        self.assertGreaterEqual(st["cap_out_dropped"]["de"]["windows_dropped"], 1)
        self.assertIn("de", st["post_build_check"])

    def test_one_generator_per_language(self):
        # red-on-base: one Random for every language; de's draws depended on whether es was built before it
        v1, v2 = _fixture(("de",), n_win=40), _fixture(("de", "es"), n_win=40)
        MB.build4(v1, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        MB.build4(v2, ("es", "de"), rows_h=99, win_h=1, allow_no_judge=True)
        a = [w["id"] for w in rj(f"{v1}/mix4/win_de.jsonl")]
        b = [w["id"] for w in rj(f"{v2}/mix4/win_de.jsonl")]
        self.assertEqual(a, b)


# ---------------------------------------------------------------- review round 2 (Gemini Argon on 45bc56f)
def _tts_fixture(vol, lang="de"):
    """Synthetic rows (tts1) for a fixture volume: t0 chosen, t1 judge-rejected; windows: a synthetic window over each,
    and a real window over a real parent that was rejected after windows2 built it."""
    base = f"ml/p3v2/{lang}"
    tts = [{"id": f"tts1_{lang}_{k}", "text": "fünfzig Prozent da", "duration": 2.0, "src": "tts", "origin": "tts",
            "audio_filepath": "t.wav",
            "split": "train", "num_chars": [[0, 15]], "spans": [{"kind": "pct", "text": "50 %", "by": "tts"}]}
           for k in range(2)]
    wj(vol, f"ml/tts1/{lang}/rows.jsonl", tts)
    rows = rj(f"{vol}/{base}/rows.jsonl")
    open(f"{vol}/{base}/reject_ids.txt", "w").write(f"{tts[1]['id']}\n{rows[5]['id']}\n")
    win = lambda k, par, syn: {"id": f"{lang}_x{k}", "text": "fünfzig Prozent da", "num_tokens": [True, True, False],  # noqa: E731
                               "duration": 3.0, "parents": par, "audio_filepath": "w.wav", "synthetic": syn}
    wj(vol, f"{base}/windows.jsonl", [win(0, [tts[0]["id"]], True), win(1, [tts[1]["id"]], True),
                                      win(2, [rows[5]["id"]], False), win(3, [rows[4]["id"]], False)])
    return tts, rows


class Round2(unittest.TestCase):
    def test_synthetic_window_with_a_removed_parent_leaves(self):
        # red-on-base: tts_ids was built after the reject filter, so a rejected synthetic parent vanished from the
        # check (set() <= took_ids) and its window stayed; a real parent rejected after windows2 crashed the build
        vol = _fixture(n_win=0)
        tts, rows = _tts_fixture(vol)
        aug, MB.TTS_AUG_FRAC = MB.TTS_AUG_FRAC, 0.0  # no degraded copies (no audio in the fixture)
        self.addCleanup(setattr, MB, "TTS_AUG_FRAC", aug)
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        ids = [w["id"] for w in rj(f"{vol}/mix4/win_de.jsonl")]
        self.assertNotIn("de_x1", ids)  # its only parent was judge-rejected
        self.assertNotIn("de_x2", ids)  # a real parent rejected after the windows were built: dropped, no crash
        self.assertIn("de_x3", ids)

    def test_one_year_definition(self):
        # red-on-base: _year_of missed hu / fi suffixed years (the gate counts them) and took "2015." ordinals (the
        # gate does not); post_build_check keyed years on the raw caption token
        hu = {"text": "a kétezer-tizenötben", "spans": []}
        self.assertEqual(MB._year_of({"kind": "int", "text": "2015-ben"}, hu, "hu"), 2015)
        self.assertEqual(MB._year_of({"kind": "ord", "text": "2015."}, {}, "de"), None)
        self.assertEqual(MB._year_of({"kind": "int", "text": "1999"}), 1999)
        rows = [{"id": f"h{i}", "text": "x", "spans": [{"kind": "int", "text": "2015-ben"}]} for i in range(40)]
        rows += [{"id": f"o{i}", "text": "x", "spans": [{"kind": "int", "text": str(1950 + i % 70)}]} for i in range(700)]
        kept, st = MB._cap_top_years(rows, 0.03, random.Random(0), lang="hu")
        self.assertIn(2015, st)
        self.assertLessEqual(MB.post_build_check(kept, [], "hu")["top_year"], 0.03 + 1e-9)

    def test_regressions_fail_loudly(self):
        # red-on-base: no before / after comparison; a later step could undo a fill floor or the top-year trim
        rows = [{"id": f"y{i}", "text": "x", "spans": [{"kind": "int", "text": str(1990 + i % 41)}]} for i in range(41 * 45)]
        before = MB.gate_counts(rows, [], "de")
        after = [x for x in rows if x["spans"][0]["text"] != "1995"][:]  # 1995: 45 -> 0
        reg = MB.post_build_check(after, [], "de", before)["regressions"]
        self.assertTrue(any(r.startswith("year1995") for r in reg), reg)
        self.assertEqual(MB.post_build_check(rows, [], "de", before)["regressions"], [])

    def test_ext_cap_keeps_floor_rows_last(self):
        rs = [{"id": f"x{i}", "spans": [{"kind": "dec", "text": "1,5", "by": "ext" if i < 12 else "ctc"}]} for i in range(30)]
        kept, _ = MB.cap_ext_per_class(rs, 0.24, random.Random(0), keep_last=lambda x: x["id"] in ("x0", "x1"))
        self.assertTrue({"x0", "x1"} <= {x["id"] for x in kept})

    def test_synthetic_selected_by_gate_class_spans(self):
        # red-on-base: synthetic rows were capped by hours per raw kind, the gate counts spans per gate class
        # (a class whose real spans meet its bar: the share rule decides; fill-to-bar covers thin classes)
        real = [{"id": f"r{i}", "text": "x", "spans": [{"kind": "pct", "text": "45 %"}]} for i in range(400)]
        tts = [{"id": f"t{i}", "text": "x", "src": "tts", "_targeted": True,
                "spans": [{"kind": "pct", "text": "46 %"}]} for i in range(200)]
        took = MB.select_synthetic(tts, real, "de", {"pct": 1.0})
        self.assertEqual(len(took), 171)  # 171 / 571 = 0.2995 <= 0.30; a 172nd would be 0.3007
        chk = MB.post_build_check(real, took + tts[171:175], "de")
        self.assertIn("pct", chk["synthetic_over"])
        took2, n = MB.trim_synthetic(took, real[:350], "de")  # real rows a later cap removed: shares re-checked
        self.assertEqual((len(took2), n), (150, 21))  # 150 / 500 = 0.30

    def test_windows2_held_unit(self):
        # red-on-base: windows2 held out PLENARY days only (committee sessions of a held-out day trained)
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_p3.py")).read()
        win = src[src.index("def windows2_lang("):src.index("def windows2(")]
        self.assertNotIn("-PLENARY", win)
        self.assertIn("held_unit(r[\"id\"]) in held_u", win)

    def test_assemble_refuses_and_scopes(self):
        import ml_multi_scale as MS  # red-on-base: assemble overwrote an existing (maybe judged) pass, flat wav/

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(MS.assemble_paths(d, "rows_x")[2], f"{d}/wav/rows_x")
            open(f"{d}/rows_x.jsonl", "w").write("")
            with self.assertRaises(FileExistsError):
                MS.assemble_paths(d, "rows_x")
            MS.assemble_paths(d, "rows_x", force=True)

    def test_held_list_comment_and_group_lines(self):
        # red-on-base: "#" lines were read as ids, and a "# group yodas:<video>" line (vox_heldout / the carve) held
        # nothing out: rows of that video trained
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/h.txt", "w").write("# list\n# group yodas:V3VVVVVVVVV\n# group vox:20190101\nde_x\n")
            ids, units = CC.read_held_list(f"{d}/h.txt")
            self.assertEqual(ids, {"de_x"})
            self.assertEqual(units, {"vid:V3VVVVVVVVV", "day:20190101"})
        vol = _fixture(n_win=0)
        open(f"{vol}/ml/p3v2/heldout_de.txt", "w").write("# carve\n# group yodas:V3VVVVVVVVV\n")
        MB.build4(vol, ("de",), rows_h=99, win_h=1, allow_no_judge=True)
        p3 = {x["id"] for x in rj(f"{vol}/mix4/p3_de.jsonl")}
        ho = {x["id"] for x in rj(f"{vol}/mix4/heldout_de.jsonl")}
        self.assertNotIn("de100_00000003_V3VVVVVVVVV_10_00_5_00", p3)
        self.assertIn("de100_00000003_V3VVVVVVVVV_10_00_5_00", ho)


if __name__ == "__main__":
    unittest.main()
