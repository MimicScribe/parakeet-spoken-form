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


if __name__ == "__main__":
    unittest.main()
