"""python3 -m unittest test_ml_tight  (tightening of rows against an external transcript)."""
import unittest

import ml_multi_tight as T


def row(text, num=None, spans=None, times=True, dur=None, step=0.5):
    toks = text.split()
    wt = [[round(i * step + 0.1, 3), round(i * step + 0.4, 3)] for i in range(len(toks))] if times else None
    return {"id": "c_p0", "origin": "piece", "parent": "c", "lang": "pl", "duration": dur or round(len(toks) * step + 0.2, 3),
            "text": text, "spans": spans or [], "num_chars": num or [], "word_times": wt, "t0": 0.0}


def ext(text, step=0.5):
    return {"id": "c_p0", "words": [{"w": w, "s": int((i * step + 0.1) * 1000), "e": int((i * step + 0.4) * 1000)}
                                    for i, w in enumerate(text.split())]}


def with_number(text, reading):
    a = text.index(reading)
    return row(text, num=[[a, a + len(reading)]], spans=[{"text": "5", "reading": reading}])


class Tighten(unittest.TestCase):
    def test_unchanged(self):
        r = row("to jest dobra sprawa dla nas wszystkich")
        out, info = T.tighten_row(r, ext("to jest dobra sprawa dla nas wszystkich"))
        self.assertEqual(out, [r])
        self.assertEqual(info["status"], "unchanged")

    def test_edge_disagreement_trims(self):
        r = row("xxxxx to jest dobra sprawa dla nas wszystkich")
        out, info = T.tighten_row(r, ext("to jest dobra sprawa dla nas wszystkich"))
        self.assertEqual([o["text"] for o in out], ["to jest dobra sprawa dla nas wszystkich"])
        self.assertEqual(info["invented"], 1)
        self.assertEqual(out[0]["origin"], "tight")
        self.assertEqual(out[0]["parent"], "c_p0")
        self.assertAlmostEqual(out[0]["word_times"][0][0], out[0]["word_times"][0][0])
        self.assertGreaterEqual(out[0]["crop_s"][0], 0.0)
        self.assertLessEqual(out[0]["crop_s"][0], r["word_times"][1][0])

    def test_middle_disagreement_splits(self):
        r = row("to jest dobra sprawa qqqqqq dla nas wszystkich zawsze dobrze")
        out, _ = T.tighten_row(r, ext("to jest dobra sprawa dla nas wszystkich zawsze dobrze"))
        self.assertEqual([o["text"] for o in out], ["to jest dobra sprawa", "dla nas wszystkich zawsze dobrze"])
        # never cuts into a neighbour word
        self.assertLessEqual(out[0]["crop_s"][1], r["word_times"][4][0])
        self.assertGreaterEqual(out[0]["crop_s"][1], r["word_times"][3][1])
        self.assertGreaterEqual(out[1]["crop_s"][0], r["word_times"][4][1])

    def test_lost_word_splits(self):
        r = row("to jest dobra sprawa dla nas wszystkich zawsze dobrze")
        out, info = T.tighten_row(r, ext("to jest dobra sprawa bardzo dla nas wszystkich zawsze dobrze"))
        self.assertEqual(info["lost"], 1)
        self.assertEqual([o["text"] for o in out], ["to jest dobra sprawa", "dla nas wszystkich zawsze dobrze"])

    def test_inflection_and_compound_kept(self):
        r = row("komisarz powiedział słusznie że rolnictwo wielkopolskie jest ważne")
        e = ext("komisarza powiedział słusznie że rolnictwo wielko polskie jest ważne")
        out, info = T.tighten_row(r, e)
        self.assertEqual(out, [r])
        self.assertEqual(info["invented"], 0)

    def test_number_never_cut(self):
        text = "w roku dwa tysiące siódmy qqqqq mamy wiele spraw do omówienia"
        r = with_number(text, "dwa tysiące siódmy")
        out, _ = T.tighten_row(r, ext("w roku 2007 mamy wiele spraw do omówienia"))
        for o in out:
            for a, b in o["num_chars"]:
                self.assertEqual(o["text"][a:b], "dwa tysiące siódmy")
            self.assertEqual(len(o["num_chars"]), len(o["spans"]))
        # a word boundary separates the number from the disagreeing word: the number stays, whole, in its run
        self.assertEqual([o["text"] for o in out], ["w roku dwa tysiące siódmy", "mamy wiele spraw do omówienia"])
        self.assertFalse(out[0]["plain"])
        self.assertLessEqual(out[0]["crop_s"][1], r["word_times"][5][0])
        self.assertGreaterEqual(out[0]["crop_s"][1], r["word_times"][4][1])

    def test_number_beside_lost_word_kept_with_boundary(self):
        text = "mamy wiele spraw do omówienia w roku dwa tysiące siódmy"
        r = with_number(text, "dwa tysiące siódmy")
        # the transcript has an extra word before "w roku", clearly apart from the number
        e = ext("mamy wiele spraw do omówienia bardzo w roku 2007")
        out, info = T.tighten_row(r, e)
        self.assertEqual(info["numbers_kept"], 1)
        self.assertTrue(any(o["spans"] for o in out))

    def test_number_glued_to_lost_word_given_up(self):
        text = "mamy wiele spraw do omówienia dwa tysiące siódmy rok mamy dużo"
        r = with_number(text, "dwa tysiące siódmy")
        e = ext("mamy wiele spraw do omówienia 2007 rok mamy dużo")
        e["words"].insert(5, {"w": "sztuk", "s": 2600, "e": 3100})  # overlaps the number's first word
        out, info = T.tighten_row(r, e)
        self.assertEqual(info["number_reasons"]["glued_to_lost_word"], 1)
        self.assertTrue(all("dwa" not in o["text"] for o in out))

    def test_digit_forms_match_adjacent_reading(self):
        for label, spans, e in (
            ("było to na konferencji COP dwadzieścia jeden w Paryżu", "dwadzieścia jeden",
             "było to na konferencji COP21 w Paryżu"),
            ("redukcja emisji CO dwa jest ważna w Paryżu dziś", "dwa", "redukcja emisji CO₂ jest ważna w Paryżu dziś"),
            ("posiedzenie wznowione o godz. dwunastej zero zero w Paryżu", "dwunastej zero zero",
             "posiedzenie wznowione o godz. 12.00 w Paryżu"),
        ):
            r = with_number(label, spans)
            out, info = T.tighten_row(r, ext(e))
            self.assertEqual(out, [r], label)
            self.assertEqual(info["invented"] + info["lost"], 0, label)

    def test_abbreviation_and_case_punctuation(self):
        r = row("Posiedzenie wznowione o godzinie, dobrze, dla nas wszystkich")
        e = ext("posiedzenie wznowione o godz. dobrze dla nas wszystkich")
        e["words"][3]["w"] = "godz."
        out, info = T.tighten_row(r, e)
        self.assertEqual(out, [r])

    def test_prefix_difference_is_disagreement(self):
        r = row("po eksportowaniu cukru mamy wiele spraw dobrych")
        out, info = T.tighten_row(r, ext("po wyeksportowaniu cukru mamy wiele spraw dobrych"))
        self.assertEqual(info["invented"], 1)

    def test_number_kept_whole_inside_run(self):
        text = "w roku dwa tysiące siódmy mamy wiele spraw do omówienia qqqqq"
        r = with_number(text, "dwa tysiące siódmy")
        out, _ = T.tighten_row(r, ext("w roku 2007 mamy wiele spraw do omówienia"))
        self.assertEqual(len(out), 1)
        o = out[0]
        self.assertFalse(o["plain"])
        a, b = o["num_chars"][0]
        self.assertEqual(o["text"][a:b], "dwa tysiące siódmy")
        self.assertEqual(len(o["spans"]), 1)

    def test_missing_word_times_not_cut(self):
        r = row("to jest dobra sprawa qqqqqq dla nas wszystkich zawsze dobrze", times=False)
        out, info = T.tighten_row(r, ext("to jest dobra sprawa dla nas wszystkich zawsze dobrze"))
        self.assertEqual(info["status"], "emptied")
        self.assertEqual(out, [])
        rows, stats, _, _ = T.tighten([r], {"c_p0": ext("to jest dobra sprawa dla nas wszystkich zawsze dobrze")})
        self.assertEqual(len(rows), 1)  # kept as it is, flagged, never dropped
        self.assertTrue(rows[0]["tight_failed"])

    def test_plain_flag_and_min_words(self):
        r = row("to jest dobra sprawa qqqqqq ab cd")
        out, _ = T.tighten_row(r, ext("to jest dobra sprawa ab cd"))
        self.assertEqual([o["text"] for o in out], ["to jest dobra sprawa"])
        self.assertTrue(out[0]["plain"])

    def test_no_ext_unchanged_and_stats(self):
        r = row("to jest dobra sprawa dla nas wszystkich")
        rows, stats, _, _ = T.tighten([r], {})
        self.assertEqual(rows, [r])
        self.assertEqual(stats["pl"]["rows_no_ext"], 1)

    def test_start_end_keys(self):
        e = {"id": "c_p0", "words": [{"w": "x", "start": 100, "end": 400}]}
        self.assertEqual(T.ext_times(e["words"][0]), (0.1, 0.4))


if __name__ == "__main__":
    unittest.main()
