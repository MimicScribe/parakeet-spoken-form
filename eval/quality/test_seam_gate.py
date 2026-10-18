"""Seam gate tests (no Modal): python3 -m unittest test_seam_gate (from eval/quality)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seam_gate as G  # noqa: E402


def pair(pid, reading, lpart, rpart, ltext, rtext, kind="int", parent="p1", lang="de"):
    base = {"pair": pid, "reading": reading, "kind": kind, "lang": lang, "pos": "in0_hi"}
    return [dict(base, id=f"{pid}_l", side="left", text=ltext, reading_part=lpart, parents=[parent]),
            dict(base, id=f"{pid}_r", side="right", text=rtext, reading_part=rpart, parents=[parent])]


P = (pair("a", "dreihundert fünfundzwanzig", "dreihundert", "fünfundzwanzig", "Es kostet dreihundert",
          "fünfundzwanzig Euro heute", kind="cur")
     + pair("b", "zweitausendzwanzig", "", "zweitausendzwanzig", "Es war im Jahr", "zweitausendzwanzig sehr gut",
            parent="p2")
     + pair("c", "zwölftausend fünfhundert", "zwölftausend", "fünfhundert", "Es kamen zwölftausend",
            "fünfhundert Leute dazu", parent="p3"))
SINGLE = {"p1": "Es kostet dreihundert fünfundzwanzig Euro heute", "p2": "Es war im Jahr zweitausendzwanzig sehr gut",
          "p3": "Es kamen zwölftausend fünfhundert Leute dazu"}


class SeamGate(unittest.TestCase):
    def test_right_merge(self):
        hyp = {"a_l": "es kostet dreihundert", "a_r": "fünfundzwanzig euro heute",
               "c_l": "es kamen zwölftausend", "c_r": "fünfhundert leute dazu",
               "b_l": "es war im jahr", "b_r": "zweitausendzwanzig sehr gut"}
        r = G.score_pairs(P, hyp, SINGLE, "de")
        self.assertEqual(r["all_long"]["pairs"], 2)  # the year is reported apart
        self.assertEqual(r["all_long"].get("seam_right"), 2)
        self.assertEqual(r["year"]["pairs"], 1)
        self.assertEqual(r["cur"]["seam_right"], 1)
        self.assertEqual(r["big"]["seam_right"], 1)

    def test_merge_made_wrong(self):
        # the left half writes the whole compound it only partly heard, the right half repeats its tail: 325 + 25
        hyp = {"a_l": "es kostet dreihundertfünfundzwanzig", "a_r": "fünfundzwanzig euro heute",
               "c_l": "es kamen zwölftausend", "c_r": "leute dazu", "b_l": "es war im jahr", "b_r": "sehr gut"}
        r = G.score_pairs(P, hyp, SINGLE, "de")
        a = r["all_long"]
        self.assertEqual(a["seam_wrong"], 2)  # 325 25 (duplication) and 12000 (the right half lost "fünfhundert")
        self.assertEqual(a["merge_made_wrong"], 2)
        self.assertEqual(r["year"].get("seam_dropped"), 1)  # the dropped year: reported, not in the long pool
        self.assertTrue(r["examples"] and r["examples"][0]["merged"])

    def test_not_merge_made_when_whole_decode_is_wrong(self):
        hyp = {"a_l": "es kostet dreihundert", "a_r": "euro heute", "c_l": "", "c_r": "", "b_l": "", "b_r": ""}
        single = dict(SINGLE, p1="Es kostet dreihundert Euro heute")
        r = G.score_pairs(P, hyp, single, "de")
        self.assertEqual(r["cur"]["seam_wrong"], 1)
        self.assertEqual(r["cur"].get("merge_made_wrong", 0), 0)
        self.assertEqual(r["cur"]["single_wrong"], 1)

    def test_digits_count_by_value(self):  # a digit writer is judged by value too: "3" + "25" is a wrong merge
        hyp = {"a_l": "es kostet 300", "a_r": "25 euro heute", "c_l": "es kamen 12", "c_r": "500 leute dazu",
               "b_l": "es war im jahr", "b_r": "2020 sehr gut"}
        r = G.score_pairs(P, hyp, SINGLE, "de")
        self.assertEqual(r["cur"].get("seam_wrong"), 1)  # 300 | 25 is not 325
        self.assertEqual(r["year"].get("seam_right"), 1)

    def test_gate(self):
        def m(w, sw=0, en=0):
            return {"de": {"all_long": {"merge_made_wrong": w, "single_wrong": sw}},
                    "en": {"all_long": {"merge_made_wrong": en}}}
        g = G.gate({"B": m(4, 3), "x": m(2, 3), "y": m(3, 3), "z": m(1, 4), "e": m(0, 0, 1)}, "B", en_rule="zero")
        self.assertTrue(g["runs"]["x"]["pass"])  # 2 <= 0.5 x 4
        self.assertIn("y", g["vetoed"])  # 3 > 2
        self.assertIn("z", g["vetoed"])  # single-decode wrong 4 > 3
        self.assertIn("e", g["vetoed"])  # English merge-made wrong must be 0
        g0 = G.gate({"B": m(0), "x": m(1)}, "B", en_rule="zero")
        self.assertIn("x", g0["vetoed"])  # a baseline at 0 allows 0
        gb = G.gate({"B": m(4, 3, 2), "x": m(2, 3, 2)}, "B", en_rule="base")
        self.assertTrue(gb["runs"]["x"]["pass"])  # English at B's level passes under the "base" rule
        self.assertIn("x", G.gate({"B": m(4, 3, 2), "x": m(2, 3, 2)}, "B", en_rule="zero")["vetoed"])  # fails "zero"
        g1 = G.gate({"B": m(2), "x": {"de": {"all_long": {"merge_made_wrong": 0}}}}, "B", en_rule="zero")
        self.assertIn("English seam pairs not measured", g1["vetoed"]["x"])


class DigitGrouping(unittest.TestCase):  # a digit writer's grouping is judged by its digits
    def test_same_value(self):
        from decimal import Decimal as D
        self.assertTrue(G.same_value([D("2.255")], [D("2255")]))
        self.assertTrue(G._contiguous([D(32), D(36)], [D(809), D(290), D(3236)]))
        self.assertFalse(G.same_value([D(3), D(25)], [D(325)]))  # a split value is wrong unless it is a phone
        self.assertTrue(G.same_value([D(809), D(2903236)], [D(809), D(290), D(3236)], grouped=True))
        self.assertFalse(G.same_value([D(12)], [D(12500)]))


def full_details(per_lang):
    """A details dict covering every gated language: per_lang overrides, the rest is 40 clean slots."""
    ok = [(True, False)] * 5
    d = {l: {(f"{l}{k}", "L"): ok for k in range(40)} for l in G.WAVE1 + ("en",)}
    d.update(per_lang)
    return d


class OffsetScore(unittest.TestCase):  # streaming-path offset score: edge-conditioned, paired bootstrap vs B
    def test_slot_scores(self):
        self.assertEqual(G.slot_scores([(True, False)] * 5), {"total": 0, "seam": 0, "seam_edge": 0, "always": 0})
        self.assertEqual(G.slot_scores([(False, True), (False, False), (True, False), (True, True), (True, False)]),
                         {"total": 2, "seam": 2, "seam_edge": 1, "always": 0})  # one of two wrong offsets was spliced
        self.assertEqual(G.slot_scores([(False, True)] * 5)["always"], 5)  # wrong everywhere: not a seam break
        self.assertEqual(G.slot_scores([(False, True)] * 5)["total"], 5)  # ... but 5 wrong readings in the total

    def test_seam_always_trade_is_vetoed(self):
        # red-on-base: the veto read seam_edge, so a run turning 1-of-5 breaks into 5-of-5 failures (seam 20 -> 0,
        # always 0 -> 50) passed; on total wrong (20 -> 50) it is vetoed
        one = [(False, True)] + [(True, False)] * 4
        base = full_details({"es": {(f"s{k}", "L"): one for k in range(20)}})
        run = full_details({"es": {(f"s{k}", "L"): ([(False, True)] * 5 if k < 10 else [(True, False)] * 5)
                                   for k in range(20)}})
        g = G.offset_gate({"B": base, "R": run}, "B", {"b": "B", "r": "R"})
        self.assertIn("r", g["vetoed"])
        self.assertEqual(g["report"]["r"]["es"]["diff"], 30)
        self.assertNotIn("r", G.offset_gate({"B": base, "R": run}, "B", {"b": "B", "r": "R"}, key="seam_edge")["vetoed"])

    def test_bootstrap_resamples_files(self):
        # red-on-base: slots were resampled one by one; 12 worse slots that all sit in ONE of 12 files are one draw of
        # a file cluster, so the file-level 90% CI reaches 0 where the slot-level one did not
        ok, bad = [(True, False)] * 5, [(False, True)] + [(True, False)] * 4
        base = {"es": {(f"f{f}s{k}", "L"): ok for f in range(12) for k in range(12)}}
        run = {"es": {(f"f{f}s{k}", "L"): (bad if f == 0 else ok) for f in range(12) for k in range(12)}}
        flat = G.boot_diff(run, base, "total", ["es"])
        self.assertGreater(flat["ci90"][0], 0)  # each slot its own group
        groups = {f"f{f}s{k}": f"file{f}" for f in range(12) for k in range(12)}
        run["_groups"] = base["_groups"] = groups
        clus = G.boot_diff(run, base, "total", ["es"])
        self.assertEqual((clus["n_groups"], clus["group_level"]), (12, "file"))
        self.assertEqual(clus["ci90"][0], 0)
        self.assertEqual((clus["worse"], clus["better"]), (12, 0))
        self.assertLess(clus["sign_p"], 0.001)
        few = {k: f"file{f}" for k, f in ((f"f{f}s{k}", f % 2) for f in range(12) for k in range(12))}
        run["_groups"] = base["_groups"] = few  # 2 files: too few to resample, sentences instead
        self.assertEqual(G.boot_diff(run, base, "total", ["es"])["group_level"], "sentence")

    def test_offset_detail_errors_and_groups(self):
        # red-on-base: a missing parse was read as a wrong decode; the file grouping did not exist
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "es_v0_00", "lang": "es", "file": f"adv_es_v0_o{o}", "values": {"N": 5}, "merge_gaps": []}
                    for o in ("0.0", "1.9", "3.8", "5.7", "7.6")]
            open(f"{d}/cases_a.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
            open(f"{d}/parse_out_a.jsonl", "w").write("".join(json.dumps({"key": f"{k}:long", "values": ["5"]}) + "\n"
                                                              for k in range(5)))
            det = G.offset_detail(d, "a")
            self.assertEqual(det["_groups"], {"es_v0_00": "adv_es_v0"})
            self.assertEqual(det["es"][("es_v0_00", "N")], [(True, False)] * 5)
            open(f"{d}/parse_out_a.jsonl", "w").write(json.dumps({"key": "0:long", "values": ["5"]}) + "\n")
            with self.assertRaises(KeyError):
                G.offset_detail(d, "a")


class OffsetGateVsB(unittest.TestCase):
    full = staticmethod(lambda per_lang: full_details(per_lang))

    def test_gate(self):
        ok, bad = [(True, False)] * 5, [(False, True)] + [(True, False)] * 4
        base = self.full({"es": {(f"s{k}", "L"): ok for k in range(40)}, "it": {(f"s{k}", "L"): bad for k in range(10)}})
        worse = self.full({"es": {(f"s{k}", "L"): (bad if k < 14 else ok) for k in range(40)},  # es 0 -> 14
                           "it": {(f"s{k}", "L"): (bad if k < 4 else ok) for k in range(10)}})   # it 10 -> 4
        same = self.full({"es": {**base["es"], ("s0", "L"): bad}, "it": base["it"]})      # es 0 -> 1: noise
        g = G.offset_gate({"B": base, "R": worse, "S": same}, "B", {"b": "B", "r": "R", "s": "S"})
        self.assertIn("r", g["vetoed"])
        self.assertTrue(any("es" in x for x in g["vetoed"]["r"]))
        self.assertNotIn("s", g["vetoed"])  # +1 is inside the bootstrap CI
        self.assertTrue(g["report"]["r"]["it"]["target_met"])  # 4 <= 0.5 x 10
        self.assertEqual(g["invalid"], {})

    def test_missing_scope_or_slot_fails(self):
        # red-on-base: a scope with 0 slots, a missing language, or a different slot set was skipped and the run passed
        ok = [(True, False)] * 5
        base = self.full({})
        arms = {"b": "B", "r": "R"}
        no_en = {l: v for l, v in base.items() if l != "en"}
        g = G.offset_gate({"B": base, "R": no_en}, "B", arms)
        self.assertIn("r", g["vetoed"])
        self.assertIn("r", g["invalid"])
        self.assertTrue(any("scope en" in x and "arm has no slots" in x for x in g["invalid"]["r"]))
        empty_fr = dict(base, fr={})
        self.assertTrue(any("scope fr" in x for x in G.offset_gate({"B": base, "R": empty_fr}, "B", arms)["invalid"]["r"]))
        no_nl = {l: v for l, v in base.items() if l != "nl"}  # a pooled-wave-1 language: named
        self.assertTrue(any("wave-1 language nl" in x for x in G.offset_gate({"B": base, "R": no_nl}, "B", arms)["invalid"]["r"]))
        extra = dict(base, de={**base["de"], ("new", "L"): ok})  # one extra slot in the arm
        sx = G.offset_gate({"B": base, "R": extra}, "B", arms)
        self.assertTrue(any("slot sets differ" in x for x in sx["invalid"]["r"]))
        short = dict(base, de={k: v for k, v in list(base["de"].items())[:-1]})  # one slot missing
        self.assertIn("r", G.offset_gate({"B": base, "R": short}, "B", arms)["invalid"])
        empty_base = dict(base, es={})
        self.assertIn("r", G.offset_gate({"B": empty_base, "R": base}, "B", arms)["invalid"])
        self.assertEqual(G.offset_gate({"B": base, "R": self.full({})}, "B", arms)["invalid"], {})

    def test_offset_detail_slot_set_differs_between_offsets(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "es_v0_00", "lang": "es", "file": f"adv_es_v0_o{o}", "values": {"N": 5}, "merge_gaps": []}
                    for o in ("0.0", "1.9", "3.8", "5.7", "7.6")]
            rows[4]["values"] = {"N": 5, "N2": 6}
            open(f"{d}/cases_a.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
            open(f"{d}/parse_out_a.jsonl", "w").write("".join(json.dumps({"key": f"{k}:long", "values": ["5", "6"]}) + "\n"
                                                              for k in range(5)))
            with self.assertRaises(ValueError):  # N2 has 1 decode (and the sets differ): never accepted
                G.offset_detail(d, "a")

    def test_lost_word_rate(self):
        rows = [{"lang": "de", "single": "wir haben dreihundert Leute gesehen", "long": "wir haben Leute gesehen"},
                {"lang": "de", "single": "alles gut", "long": "alles gut"},
                {"lang": "es", "single": "uno dos tres cuatro", "long": "uno dos tres cuatro"}]
        r = G.lost_word_rate(rows)
        self.assertEqual(r["de"], [1, 7])
        self.assertEqual(r["all"], [1, 11])

    def test_two_work_dirs(self):
        # red-on-base: the baseline's files had to sit in the same dir as the arm's (same arm name collided)
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        def bed(d, langs, wrong):
            rows, parse = [], []
            for l in langs:
                for k in range(3):
                    for o in range(5):
                        rows.append({"id": f"{l}_v{k}_00", "lang": l, "file": f"adv_{l}_v{k}_o{o}", "values": {"N": 5},
                                     "merge_gaps": [], "single": "a b", "long": "a b"})
                        parse.append({"key": f"{len(parse)}:long", "values": ["7" if wrong else "5"]})
            open(f"{d}/cases_x.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
            open(f"{d}/parse_out_x.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in parse))
        langs = list(G.WAVE1) + ["en"]
        with tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as r, tempfile.TemporaryDirectory() as o:
            bed(b, langs, False)
            bed(r, langs, True)
            out = f"{o}/g.json"
            with redirect_stdout(io.StringIO()):
                rc = G.main(["R", "--base", "B", "--langs", "", "--no-fetch", "--offset-work", r, "--baseline-work", b,
                             "--offset-arms", "B=x,R=x", "--json", out])
            self.assertEqual(rc, 0)
            self.assertIn("R", json.load(open(out))["vetoed"])  # R (all wrong) vs B (all right), same arm name "x"
            with self.assertRaises(SystemExit):  # --json is required with --offset-work
                G.main(["R", "--base", "B", "--no-fetch", "--offset-work", r, "--offset-arms", "B=x,R=x"])

    def test_main_exit_code_for_invalid(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "es_v0_00", "lang": "es", "file": f"adv_es_v0_o{o}", "values": {"N": 5}, "merge_gaps": []}
                    for o in range(5)]
            open(f"{d}/cases_x.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
            open(f"{d}/parse_out_x.jsonl", "w").write("".join(json.dumps({"key": f"{k}:long", "values": ["5"]}) + "\n" for k in range(5)))
            with redirect_stdout(io.StringIO()):
                rc = G.main(["R", "--base", "B", "--langs", "", "--no-fetch", "--offset-work", d, "--offset-arms",
                             "B=x,R=x", "--json", f"{d}/g.json"])
            self.assertEqual(rc, 2)  # only es present: wave-1 languages and en missing -> INVALID, non-zero

    def test_proxy_gate_is_report_only(self):
        # red-on-base: the proxy's union veto was merged into the vetoed set
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            pj = [h for h in pair("a", "dreihundert fünfundzwanzig", "dreihundert", "fünfundzwanzig",
                                  "Es kostet dreihundert", "fünfundzwanzig Euro heute", kind="cur")]
            os.makedirs(f"{d}/ml/p3v2")
            open(f"{d}/ml/p3v2/heldout_seam_de.jsonl", "w").write("".join(json.dumps(h) + "\n" for h in pj))
            for run, lh in (("B", "es kostet dreihundert fünfundzwanzig"), ("R", "es kostet dreihundert")):
                os.makedirs(f"{d}/hyps/{run}")
                open(f"{d}/hyps/{run}/hseam_de.jsonl", "w").write(
                    json.dumps({"id": "a_l", "hyp": lh}) + "\n" + json.dumps({"id": "a_r", "hyp": "euro heute"}) + "\n")
                open(f"{d}/hyps/{run}/hseam_parents_de.jsonl", "w").write(
                    json.dumps({"id": "p1", "hyp": "Es kostet dreihundert fünfundzwanzig Euro heute"}) + "\n")
            with redirect_stdout(io.StringIO()):
                G.main(["R", "--base", "B", "--langs", "de", "--no-fetch", "--work", d, "--json", f"{d}/g.json"])
            res = json.load(open(f"{d}/g.json"))
            self.assertEqual(res["vetoed"], {})
            self.assertIn("R", res["proxy"]["vetoed"])  # reported, not merged

    def test_gap_has_number(self):
        self.assertTrue(G._gap_has_number([{"A": " vingt", "B": " vingt-deux", "R": " vingt"}], "fr"))
        self.assertFalse(G._gap_has_number([{"A": " le", "B": " la", "R": " la"}], "fr"))


class NoDamage(unittest.TestCase):  # score_heldout.no_damage: next / moderate tiers vs the baseline, 2 pp
    def test_rule(self):
        try:
            import score_heldout as H
        except ImportError as e:  # needs the pipeline modules (ml_cands / ml_decide)
            self.skipTest(str(e))
        sp = lambda w, d, x, dr: {"spans": {"all": {"WORDS_RIGHT_VALUE": w, "DIGITS_RIGHT_VALUE": d,  # noqa: E731
                                                    "WRONG_VALUE": x, "DROPPED": dr}}}
        base = {"pl": sp(0, 80, 10, 10), "ro": sp(0, 80, 10, 10)}
        run = {"pl": sp(60, 19, 11, 10), "ro": sp(50, 20, 20, 10)}  # pl: 79 vs 80 right, 21 vs 20 wrong: pass
        nd = H.no_damage(run, base)
        self.assertTrue(nd["pl"]["pass"])
        self.assertFalse(nd["ro"]["pass"])  # 70 vs 80 value-right


class PickMostConverted(unittest.TestCase):  # score_heldout.pick_most_converted + span_form
    def setUp(self):
        try:
            import score_heldout as H
        except ImportError as e:
            self.skipTest(str(e))
        self.H = H

    def test_span_form(self):
        self.assertEqual(self.H.span_form("WORDS_RIGHT_VALUE", "dos mil", "es"), "converted")
        self.assertEqual(self.H.span_form("DIGITS_RIGHT_VALUE", "2500", "es"), "digits_left")
        self.assertEqual(self.H.span_form("WRONG_VALUE", "2 mil quinientos", "es"), "mixed")
        self.assertEqual(self.H.span_form("DROPPED", "y", "es"), "other")
        self.assertEqual(self.H.span_form("WRONG_VALUE", "zweitausendzwanz", "de"), "words_wrong")  # garbled words are not converted
        self.assertEqual(self.H.span_form("WRONG_VALUE", "zwanzig tausend", "de"), "words_wrong")

    def test_pick_ignores_garbled_conversion(self):
        sp = lambda w, d, x, dr, fm: {"spans": {"all": {"WORDS_RIGHT_VALUE": w, "DIGITS_RIGHT_VALUE": d,  # noqa: E731
                                                        "WRONG_VALUE": x, "DROPPED": dr}}, "forms": fm}
        out = {"B": {"de": sp(0, 90, 5, 5, {"digits_left": 100, "n": 100})},
               "garbler": {"de": sp(80, 10, 5, 5, {"converted": 40, "words_wrong": 55, "digits_left": 5, "n": 100, "converted_old": 95})},
               "clean": {"de": sp(80, 10, 5, 5, {"converted": 60, "digits_left": 40, "n": 100, "converted_old": 60})}}
        pk = self.H.pick_most_converted(out, "B")
        self.assertEqual(pk["pick"], "clean")  # on the old count the garbler (95%) would have won
        self.assertEqual(pk["per_language"]["de"]["garbler"]["converted_old"], 95.0)

    def test_pick(self):
        sp = lambda w, d, x, dr, fm: {"spans": {"all": {"WORDS_RIGHT_VALUE": w, "DIGITS_RIGHT_VALUE": d,  # noqa: E731
                                                        "WRONG_VALUE": x, "DROPPED": dr}}, "forms": fm}
        f = lambda c, dl, m: {"converted": c, "digits_left": dl, "mixed": m, "n": c + dl + m}  # noqa: E731
        out = {"B": {"es": sp(0, 90, 5, 5, f(0, 100, 0))},
               "half": {"es": sp(40, 50, 5, 5, f(40, 50, 10))},   # passes, 40% converted
               "full": {"es": sp(80, 10, 5, 5, f(80, 20, 0))},    # passes, 80% converted
               "bad": {"es": sp(60, 10, 25, 5, f(95, 0, 5))}}     # most converted, but wrong + dropped 30 vs 10
        pk = self.H.pick_most_converted(out, "B")
        self.assertEqual(pk["pick"], "full")
        self.assertEqual(pk["failed"]["bad"], ["es"])
        self.assertEqual(pk["ranked"][0][:2], ("full", 80.0))


if __name__ == "__main__":
    unittest.main()
