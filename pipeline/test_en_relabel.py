"""English label fixes (en_relabel.py): python3 test_en_relabel.py"""
import unittest

import en_relabel as R


class EnRelabel(unittest.TestCase):
    def test_titles(self):
        self.assertEqual(R.fix_text("Mr. Alsop and Dr. Smith met Mrs. Jones.")[0], "Mister Alsop and Doctor Smith met Missus Jones.")
        self.assertEqual(R.fix_text("It is on Elm Dr. near the park")[0], "It is on Elm Dr. near the park")  # drive: unchanged
        self.assertEqual(R.fix_text("Visit St. Patrick's Day")[0], "Visit St. Patrick's Day")  # ambiguous: left alone

    def test_per_cent_and_fraction(self):
        self.assertEqual(R.fix_text("Up three per cent. Per cent matters.")[0], "Up three percent. Percent matters.")
        self.assertEqual(R.fix_text("one-third of it, two-thirds of that, a one-piece suit")[0],
                         "one third of it, two thirds of that, a one-piece suit")

    def test_symbols(self):
        self.assertEqual(R.fix_text("is −one point four and −eight")[0], "is minus one point four and minus eight")
        for t in ("eight × eight blocks", "f−one is continuous", "India === the", "of the reform == although"):
            self.assertIsNone(R.fix_text(t)[0], t)

    def test_capitalised_number_at_a_join_only_in_joined_parts(self):
        t = "we paid for it Thirty two people came. Seven left"
        self.assertEqual(R.fix_text(t, joined=True)[0], "we paid for it thirty two people came. Seven left")
        self.assertEqual(R.fix_text(t, joined=False)[0], t)
        self.assertEqual(R.fix_text("Thirty people came", joined=True)[0], "Thirty people came")  # sentence start stays

    def test_timed_row_with_changed_token_count_is_dropped(self):
        rows = [{"audio_filepath": "a.wav", "text": "up three per cent", "word_times": [[0, 1]] * 4},
                {"audio_filepath": "b.wav", "text": "Mr. Smith", "word_times": [[0, 1]] * 2}]
        out, drops, c = R.relabel_rows(rows, "win_en.jsonl", True)
        self.assertEqual([r["audio_filepath"] for r in out], ["b.wav"])
        self.assertEqual(out[0]["text"], "Mister Smith")
        self.assertEqual([d["audio_filepath"] for d in drops], ["a.wav"])


if __name__ == "__main__":
    unittest.main()
