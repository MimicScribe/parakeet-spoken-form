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
        self.assertIn("a seam=1 always=3 of 6 slot readings", r.stdout)

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
        self.assertIn("a seam=0 always=5 of 10 slot readings", r.stdout)
        self.assertIn("et  seam", r.stdout)


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


if __name__ == "__main__":
    unittest.main()
