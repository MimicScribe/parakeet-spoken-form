"""offset_score.py / score_values.py on tiny synthetic dumps: python3 test_seam_scorers.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def case(sid, values, lang="en"):
    return {"id": sid, "lang": lang, "values": values, "merge_gaps": [], "long": "", "single": "", "file": "f", "same": False}


def run_offset(rows, got, *extra, keys=None):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "cases_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        keys = keys if keys is not None else range(len(got))
        (d / "parse_out_a.jsonl").write_text("".join(json.dumps({"key": f"{k}:long", "values": [got[k]]}) + "\n" for k in keys))
        return subprocess.run([sys.executable, str(HERE / "offset_score.py"), str(d), "a", *extra], capture_output=True, text=True)


class OffsetScore(unittest.TestCase):
    def test_counts_seam_and_always(self):  # the counting itself is unchanged (red on base only for the new --offsets flag)
        # s1: slot N right at offsets 0-1, wrong at 2 (seam break); s2 wrong at every offset
        rows = [case("en_s1", {"N": "5"})] * 3 + [case("en_s2", {"N": "9"})] * 3
        r = run_offset(rows, ["5", "5", "7", "1", "2", "3"], "--offsets", "3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("a total=4 seam=1 always=3 of 6 slot readings", r.stdout)

    def test_missing_parse_is_an_error(self):
        # red-on-base: a row without its "<k>:long" parse counted as a wrong reading
        rows = [case("en_s1", {"N": "5"})] * 5
        r = run_offset(rows, ["5"] * 5, keys=[0, 1, 2, 3])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("4:long", r.stderr)

    def test_offsets_per_sentence_asserted(self):
        # red-on-base: a sentence decoded at 4 offsets instead of 5 was scored as if complete
        rows = [case("en_s1", {"N": "5"})] * 5 + [case("en_s2", {"N": "9"})] * 4
        r = run_offset(rows, ["5"] * 9)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("decodes", r.stderr)

    def test_totals_cover_every_language_present(self):
        # red-on-base: a language outside the hard-coded ORDER ("et") was left out of the totals
        rows = [case("en_s1", {"N": "5"})] * 5 + [case("et_s1", {"N": "9"}, "et")] * 5
        r = run_offset(rows, ["5"] * 5 + ["1"] * 5)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("a total=5 seam=0 always=5 of 10 slot readings", r.stdout)
        self.assertIn("et  total", r.stdout)


def write_arm(d, name, rows, got):
    (d / f"cases_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / f"parse_out_{name}.jsonl").write_text("".join(json.dumps({"key": f"{k}:long", "values": [g]}) + "\n" for k, g in enumerate(got)))


def offs(sid, values, lang="en", fname="f", **kw):
    return [dict(case(sid, values, lang), file=f"{fname}_o{o}", **kw) for o in ("0.0", "1.9", "3.8", "5.7", "7.6")]


class Paired(unittest.TestCase):
    def test_total_wrong_headline_and_transitions(self):
        # red-on-base: seam and always were reported apart, so a run turning 1-of-5 breaks into 5-of-5 failures read
        # as a seam IMPROVEMENT; total wrong (seam + always), paired by slot, shows it 2 -> 5 worse
        import offset_score as O
        rows = offs("en_s1", {"N": "5"}) + offs("en_s2", {"N": "9"})
        base = ["5", "5", "5", "5", "1"] + ["9", "9", "9", "9", "1"]   # 1 wrong of 5 in each slot: seam 2
        run = ["1"] * 5 + ["9"] * 5                                      # s1 wrong everywhere, s2 right
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_arm(d, "b", rows, base); write_arm(d, "r", rows, run)
            tb = O.slot_table(*O.load_arm(str(d), "b")); tr = O.slot_table(*O.load_arm(str(d), "r"))
            p = O.paired(tb, tr)
            self.assertEqual((p["seam_base"], p["seam_run"]), (2, 0))
            self.assertEqual((p["total_base"], p["total_run"], p["diff"]), (2, 5, 3))
            self.assertEqual(p["transitions"], {"4->0": 1, "4->5": 1})
            self.assertEqual((p["worse"], p["better"]), (1, 1))
            self.assertEqual(p["seam_both_base"], 1)  # only s2 is right at least once in both arms
            out = subprocess.run([sys.executable, str(HERE / "offset_score.py"), str(d), "b", "r"], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("TOTAL WRONG 2 -> 5", out.stdout)

    def test_bootstrap_and_sign(self):
        import offset_score as O
        self.assertAlmostEqual(O.sign_p(10, 0), 2 / 1024)
        self.assertEqual(O.sign_p(0, 0), 1.0)
        lo, hi = O.boot_ci({f"f{k}": 1 for k in range(20)})
        self.assertEqual((lo, hi), (20, 20))
        lo, hi = O.boot_ci({"f0": 12, **{f"f{k}": 0 for k in range(1, 12)}})  # one file holds every move
        self.assertEqual(lo, 0)

    def test_boundary_guard(self):
        # red-on-base: no guard. edge_num [start, end] flags the first / last slot of that sentence reading
        import offset_score as O
        rows = offs("en_s1", {"A": "1", "B": "2"}, edge_num=[False, False])
        rows[2]["edge_num"] = [False, True]  # B's number word sat at the sentence end cut at offset 3.8
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_arm(d, "b", rows, ["1"] * 5)
            t = O.slot_table(*O.load_arm(str(d), "b"))
            self.assertEqual(O.guarded_keys([t]), {("en_s1", "A")})

    def test_near_interior(self):
        import offset_score as O
        rows = offs("en_s1", {"N": "5"}, num_t=[10.0], file_gaps=[30.0])
        rows[0]["file_gaps"] = [10.5]  # a merge point 0.5 s from the number at offset 0.0 only
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_arm(d, "b", rows, ["1", "5", "5", "5", "5"])
            t = O.slot_table(*O.load_arm(str(d), "b"))
            ni = O.near_interior([t])[0]
            self.assertEqual(ni["near"], [1, 1])
            self.assertEqual(ni["interior"], [0, 4])
            self.assertEqual(ni["within_slot_near"], [1, 1])


class SlotOk(unittest.TestCase):
    def setUp(self):
        import score_values  # red-on-base: not importable (the script ran on import)
        self.ok = score_values.slot_ok

    def test_price(self):
        # red-on-base: slot P parsed its string value with int() and crashed on "12.50"
        self.assertTrue(self.ok("P", "12.50", ["12.5"]))
        self.assertTrue(self.ok("P", "12.50", ["12", "50"]))
        self.assertFalse(self.ok("P", "12.50", ["12", "40"]))
        self.assertTrue(self.ok("P", "7", ["7"]))
        self.assertFalse(self.ok("P", "7", ["70"]))

    def test_time(self):
        self.assertTrue(self.ok("T", "9:05", ["9", "5"]))
        self.assertTrue(self.ok("T", "9:05", ["9.05"]))
        self.assertTrue(self.ok("T", "9:05", ["905"]))
        self.assertTrue(self.ok("T", "9:00", ["9"]))
        self.assertFalse(self.ok("T", "9:05", ["9", "15"]))

    def test_phone(self):
        self.assertTrue(self.ok("PH", "5551234", ["555", "1234"]))
        self.assertTrue(self.ok("PH", "5551234", ["555.1234"]))
        self.assertFalse(self.ok("PH", "5551234", ["555", "1243"]))

    def test_plain(self):
        self.assertTrue(self.ok("N", "2.5", ["2.5", "x"]))
        self.assertFalse(self.ok("N", "25", ["2.5"]))

    def test_positional(self):
        # red-on-base: PH was a substring of every parsed digit joined, T / P parts anywhere in the list, and one
        # parsed number could serve two slots
        import score_values as S
        self.assertFalse(self.ok("PH", "5551234", ["555", "12345"]))   # digits must be EQUAL, not contained
        self.assertFalse(self.ok("PH", "5551234", ["15551234"]))
        self.assertTrue(self.ok("PH", "5551234", ["7", "555", "1234"]))  # a run of consecutive parsed numbers
        self.assertFalse(self.ok("PH", "5551234", ["555", "7", "1234"]))
        self.assertFalse(self.ok("T", "9:05", ["9", "3", "5"]))          # hour and minute next to each other
        self.assertFalse(self.ok("P", "12.50", ["12", "7", "50"]))       # whole and cents next to each other
        self.assertEqual(S.match_reading({"N": "5", "N2": "5"}, ["5"]), {"N": True, "N2": False})
        self.assertEqual(S.match_reading({"N": "5", "N2": "5"}, ["5", "7", "5"]), {"N": True, "N2": True})
        self.assertEqual(S.match_reading({"P": "12.50", "N": "50"}, ["12", "50"]), {"P": True, "N": False})
        self.assertEqual(S.match_reading({"A": "3", "B": "2"}, ["2", "3"]), {"A": True, "B": False})  # in order

    def test_lenient(self):
        # red-on-base: lenient took any n x 10^k for L, any prefix / suffix for D, the hour alone for any T and
        # concatenated values that stood apart in the text
        import score_values as S
        R = S.Reading
        mil = R(["3.2"], [[0, 2]], ["three", "point", "two", "million"])
        self.assertTrue(S.slot_ok("L", 3200000, mil, lenient=True))
        self.assertFalse(S.slot_ok("L", 3200000, mil))  # strict: never
        self.assertFalse(S.slot_ok("L", 3200000, R(["3.2"], [[0, 2]], ["three", "point", "two", "messages"]), lenient=True))
        pair = R(["3", "2"], [[0, 0], [1, 1]], ["three", "two", "million"])
        self.assertTrue(S.slot_ok("L", 3200000, pair, lenient=True))  # mantissa pair next to each other + scale word
        self.assertFalse(S.slot_ok("D", 1.7, R(["1.71"], [[0, 2]], ["a", "b", "c"]), lenient=True))
        self.assertFalse(S.slot_ok("T", "9:30", R(["9"], [[0, 0]], ["nine"]), lenient=True))
        self.assertTrue(S.slot_ok("T", "9:00", R(["9"], [[0, 0]], ["nine"]), lenient=True))
        self.assertTrue(S.slot_ok("Y", 1997, R(["19", "97"], [[0, 0], [1, 1]], ["nineteen", "ninety", "seven"]), lenient=True))
        self.assertFalse(S.slot_ok("Y", 1997, R(["19", "97"], [[0, 0], [2, 2]], ["nineteen", "and", "ninety"]), lenient=True))
        self.assertFalse(S.slot_ok("Y", 1997, ["19", "97"], lenient=True))  # no spans: never concatenated

    def test_minor_unit_and_clock_minutes_the_parser_reads_differently(self):
        # red-on-base: a price read "78" + "0.25" (nl "achtenzeventig euro en vijfentwintig cent") and a time whose
        # parsed span kept only the hour ("negentien uur vijf" parsed 19) scored wrong
        import score_values as S
        R = S.Reading
        self.assertTrue(S.slot_ok("P", "78.25", R(["78", "0.25"], [[3, 4], [6, 7]], ["a"] * 8)))
        self.assertTrue(S.slot_ok("P", "78.25", ["78", "25"]))  # unchanged
        self.assertFalse(S.slot_ok("P", "78.25", ["78", "0.26"]))
        self.assertFalse(S.slot_ok("P", "78.25", ["79", "0.25"]))
        w = "de vergadering begint om negentien uur vijf en duurt acht minuten".split()
        hour = R(["19", "8"], [[4, 6], [9, 9]], w)
        self.assertTrue(S.slot_ok("T", "19:05", hour))
        self.assertFalse(S.slot_ok("T", "19:06", hour))
        self.assertFalse(S.slot_ok("T", "19:05", R(["19", "8"], [[4, 5], [9, 9]], w)))  # no minutes in the span
        self.assertFalse(S.slot_ok("T", "19:05", ["19"]))  # no words: no extra credit
        w2 = "begint om negentien uur vijf en veertig".split()
        self.assertTrue(S.slot_ok("T", "19:45", R(["19"], [[2, 6]], w2)))


def parse_files(d, rows):
    """rows: (lang, text, grammar-parse values, words) -> parse_in_x / parse_out_x in d (spans: one word per value, unused)"""
    with open(d / "parse_in_x.jsonl", "w") as fi, open(d / "parse_out_x.jsonl", "w") as fo:
        for k, (lang, text, vals, words) in enumerate(rows):
            fi.write(json.dumps({"key": f"{k}:long", "lang": lang, "text": text}, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"key": f"{k}:long", "values": vals, "words": words, "alt": []}, ensure_ascii=False) + "\n")


class ParserArtefacts(unittest.TestCase):
    """Grammar parses (as dumped on a real bed) of readings that say the right value: the scorer must not call
    them wrong, and must still call wrong readings wrong. The same readings apply to every arm."""

    def test_reader_reading_fixes_parser_artefacts(self):
        # red-on-base: load_parse returned the two grammar readings only; ranges joined by a number-internal "and" (ro 131),
        # grouped phone digits summed (el 214, sv 967), fi / da clocks read as fractions (0.15), hu / nl minutes after
        # the hour word dropped, fr "midi quinze" read as 15 were all scored wrong
        import score_values as S
        rows = [
            ("ro", "Au venit între cincizeci și cinci și șaptezeci și șase de oameni.", ["131"], None, {"A": 55, "B": 76}),
            ("el", "Ο αριθμός μου είναι μηδέν έξι, είκοσι επτά, σαράντα εννέα, δύο, τριάντα πέντε.", ["0", "119"], None,
             {"PH": "062749235"}),
            ("sv", "Mitt nummer är noll nio sextiosju sjuhundrasjuttioett åttioett trettionio.", ["0", "967"], None,
             {"PH": "09677718139"}),
            ("da", "Mødet starter klokken otte komma et fem og varer treogtyve minutter", ["8", "0.15", "23"], None,
             {"T": "8:15", "M": 23}),
            ("fi", "Kokous alkaa kello neljätoista pilkku yksi viisi ja kestää kolmekymmentäkuusi minuuttia.", ["0.15", "36"],
             None, {"T": "14:15", "M": 36}),
            ("hu", "A megbeszélés kezdete húsz óra öt perc, hossza hatvankettő perc.", ["20", "62"], None, {"T": "20:05", "M": 62}),
            ("fr", "La réunion commence à midi quinze et dure trente minutes.", ["15", "30"], None, {"T": "12:15", "M": 30}),
            ("fi", "Liikevaihto kasvoi kaksitoista pilkku viisi prosenttimerkki kaksitoistatuhatta euroon.", ["0.5", "12000"],
             None, {"PCT": 12.5, "L": 12000}),
        ]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            parse_files(d, [(l, t, v, w or t.lower().split()) for l, t, v, w, _ in rows])
            P = S.load_parse(str(d / "parse_out_x.jsonl"))
            for k, (lang, text, _, _, want) in enumerate(rows):
                self.assertEqual(len(P[f"{k}:long"]), 3)
                got = S.match_slots(want, P[f"{k}:long"])
                self.assertTrue(all(got.values()), (lang, text, got))

    def test_reader_reading_keeps_wrong_readings_wrong(self):
        # red-on-base: control (these were wrong before and must stay wrong)
        import score_values as S
        rows = [
            ("hu", "Összesen hetvenkétezerszáztizenkilenc ember jött el.", ["72119"], {"A": 72, "B": 119}),
            ("el", "Ο αριθμός μου είναι μηδέν έξι, είκοσι οκτώ, σαράντα εννέα, δύο, τριάντα πέντε.", ["0", "120"],
             {"PH": "062749235"}),
            ("da", "Mødet starter klokken otte komma et seks og varer treogtyve minutter", ["8", "0.16", "23"],
             {"T": "8:15", "M": 23}),
            ("fr", "La réunion commence à midi vingt et dure trente minutes.", ["20", "30"], {"T": "12:15", "M": 30}),
            ("ro", "Au venit între cincizeci și cinci și șaptezeci și șapte de oameni.", ["132"], {"A": 55, "B": 76}),
        ]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            parse_files(d, [(l, t, v, t.lower().split()) for l, t, v, _ in rows])
            P = S.load_parse(str(d / "parse_out_x.jsonl"))
            for k, (lang, text, _, want) in enumerate(rows):
                self.assertFalse(all(S.match_slots(want, P[f"{k}:long"]).values()), (lang, text))

    def test_range_parts_split_at_the_joiner(self):
        # red-on-base: the no-joiner reading splits "cincizeci si cinci" into 50, 5 and nothing re-added them
        import score_values as S
        R = S.Reading
        w = "au venit între cincizeci și cinci și șaptezeci și șase de oameni".split()
        alt = R(["50", "5", "70", "6"], [[3, 3], [5, 5], [7, 7], [9, 9]], w)
        self.assertEqual(S.match_reading({"A": 55, "B": 76}, alt), {"A": True, "B": True})
        self.assertEqual(S.match_reading({"A": 131}, alt), {"A": False})  # the two ends never sum
        bg = "дойдоха между двайсет и девет и шейсет и три души".split()
        self.assertEqual(S.match_reading({"A": 29, "B": 63}, R(["20", "9", "60", "3"], [[2, 2], [4, 4], [6, 6], [8, 8]], bg)),
                         {"A": True, "B": True})
        far = "cincizeci de oameni și cinci".split()  # not one joiner between the parts
        self.assertFalse(S.slot_ok("A", 55, R(["50", "5"], [[0, 0], [4, 4]], far)))
        self.assertFalse(S.slot_ok("A", 55, R(["5", "50"], [[0, 0], [2, 2]], ["cinci", "și", "cincizeci"])))  # place order
        self.assertFalse(S.slot_ok("A", 55, ["50", "5"]))  # no spans: never summed

    def test_clock_zero_minutes_word(self):
        # red-on-base: "eleven zero five" (11, 0, 5) for 11:05 was wrong; only "eleven oh five" (11, 5) scored
        import score_values as S
        self.assertTrue(S.slot_ok("T", "11:05", ["11", "0", "5"]))
        self.assertEqual(S.match_reading({"T": "18:05", "M": 51}, ["18", "0", "5", "51"]), {"T": True, "M": True})
        self.assertFalse(S.slot_ok("T", "11:15", ["11", "0", "15"]))  # two-digit minutes never take the zero
        self.assertFalse(S.slot_ok("T", "11:05", ["11", "1", "5"]))

    def test_reader_found_however_loaded(self):
        # red-on-base: the reader import inside _clock_tail failed silently (no credit) when score_values was loaded
        # from another directory or as a copy without the reader next to it
        prog = ("import importlib.util, sys\n"
                f"spec = importlib.util.spec_from_file_location('sv', {str(HERE / 'score_values.py')!r})\n"
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "w = 'de vergadering begint om negentien uur vijf en duurt acht minuten'.split()\n"
                "print(m._clock_tail(m.Reading(['19', '8'], [[4, 6], [9, 9]], w), 0))\n")
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True, cwd=d)
            self.assertEqual(r.stdout.strip(), "5", r.stderr)
            # a copy WITHOUT the reader next to it must fail loudly, never score silently
            (Path(d) / "score_values.py").write_text((HERE / "score_values.py").read_text())
            prog2 = prog.replace(str(HERE / "score_values.py"), str(Path(d) / "score_values.py"))
            r = subprocess.run([sys.executable, "-I", "-c", prog2], capture_output=True, text=True, cwd=d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("number_outcomes", r.stderr)


if __name__ == "__main__":
    unittest.main()
