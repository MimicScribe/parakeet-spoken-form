"""Smoke test for offset_score.py / score_values.py on a tiny synthetic dump: python3 test_seam_scorers.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def case(i, sid, values):
    return {"id": sid, "lang": "en", "values": values, "merge_gaps": [], "long": "", "single": "", "file": "f", "same": False}


class SeamScorers(unittest.TestCase):
    def test_offset_score_counts_seam_and_always(self):
        # sentence s1: slot N is 5 at offsets 0-1 (right) and 7 at offset 2 (seam break); s2 is wrong at every offset
        rows = [case(0, "en_s1", {"N": "5"}), case(1, "en_s1", {"N": "5"}), case(2, "en_s1", {"N": "5"}),
                case(3, "en_s2", {"N": "9"}), case(4, "en_s2", {"N": "9"})]
        got = ["5", "5", "7", "1", "2"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cases_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / "parse_out_a.jsonl").write_text("".join(json.dumps({"key": f"{k}:long", "values": [g]}) + "\n" for k, g in enumerate(got)))
            out = subprocess.run([sys.executable, str(HERE / "offset_score.py"), str(d), "a"], capture_output=True, text=True, check=True).stdout
        self.assertIn("a seam=1 always=2 of 5 slot readings", out)


if __name__ == "__main__":
    unittest.main()
