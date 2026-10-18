"""Unit tests for the public benchmark (stdlib only): python3 -m unittest test_benchmark  (from benchmark/)"""
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import seam_harness as SH  # noqa: E402
import value_accuracy as VA  # noqa: E402
import verbalizer as VB  # noqa: E402


def fake_windows(sents, off, dur, win=SH.WIN_S, stride=SH.STRIDE_S, field="label"):
    """a perfect decoder: each sentence's words spread evenly over its interval; a window sees the words inside it"""
    words = []
    for s in sents:
        ws = s[field].split()
        step = (s["end"] - s["start"]) / len(ws)
        for i, w in enumerate(ws):
            t0 = off + s["start"] + i * step
            words.append([w, t0, t0 + step * 0.8])
    out = []
    for st in SH.window_starts(dur, win, stride):
        out.append({"start": st, "words": [[w, a - st, b - st] for w, a, b in words if st <= a and b <= st + win]})
    return out


def fake_decoded(manifest, n_off=5, field="label"):
    long = {}
    for f, sents in manifest.items():
        for off in SH.offsets(n_off):
            dur = sents[-1]["end"] + off + 0.3
            long[f"{f}_o{off:g}"] = {"dur": dur, "windows": fake_windows(sents, off, dur, field=field)}
    single = {s["id"]: s[field] for ss in manifest.values() for s in ss}
    return {"long": long, "single": single, "geometry": {"win": SH.WIN_S, "stride": SH.STRIDE_S, "offsets": SH.offsets(n_off)}}


BED = {"t_de_f00": [
    {"id": "de_0", "lang": "de", "start": 0.3, "end": 6.0, "text": "Im Jahr 2010 kamen 37 Leute.",
     "label": "Im Jahr zweitausendzehn kamen siebenunddreißig Leute.", "values": {"Y": 2010, "N": 37}},
    {"id": "de_1", "lang": "de", "start": 6.4, "end": 13.9, "text": "Der Preis war 12,50 Euro um 10:30 Uhr.",
     "label": "Der Preis war zwölf Euro fünfzig um zehn Uhr dreißig.", "values": {"P": 12.5, "T": "10:30"}},
    {"id": "de_2", "lang": "de", "start": 14.3, "end": 24.0, "text": "Wählen Sie 0170 123 45 67, das sind 55 %.",
     "label": "Wählen Sie null eins sieben null eins zwei drei vier fünf sechs sieben, das sind fünfundfünfzig Prozent.",
     "values": {"PH": "01701234567", "PCT": 55}},
]}


class Geometry(unittest.TestCase):
    def test_offsets_and_windows(self):
        self.assertEqual(SH.offsets(), [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual(SH.window_starts(30.0), [0.0, 10.0, 20.0])
        self.assertEqual(SH.window_starts(15.0), [0.0])
        self.assertEqual(SH.window_starts(15.1), [0.0, 10.0])

    def test_modal_copy_agrees(self):
        src = (Path(__file__).resolve().parent / "modal_bench.py").read_text(encoding="utf-8")
        body = re.search(r"def window_starts\(.*?\n    return starts\n", src, re.S).group(0)
        ns = {}
        exec(body, ns)  # the container's copy, without importing modal
        for d in (3.0, 15.0, 15.01, 29.99, 30.0, 81.2):
            self.assertEqual(ns["window_starts"](d, 15.0, 10.0), SH.window_starts(d))


class Merge(unittest.TestCase):
    def test_agreeing_windows_no_duplicates(self):
        w = [{"start": 0.0, "words": [["a", 1, 2], ["b", 11, 12], ["c", 13.5, 14]]},
             {"start": 10.0, "words": [["b", 1, 2], ["c", 3.5, 4], ["d", 6, 7]]}]
        words, gaps = SH.merge_windows(w, 20.0)
        self.assertEqual([x[0] for x in words], ["a", "b", "c", "d"])
        self.assertEqual(gaps, [])

    def test_disagreement_is_a_gap_and_midpoint_picks(self):
        w = [{"start": 0.0, "words": [["vier", 11, 11.5], ["zehn", 13.0, 13.4]]},
             {"start": 10.0, "words": [["4", 1, 1.5], ["10", 3.0, 3.4]]}]
        words, gaps = SH.merge_windows(w, 20.0)
        self.assertEqual([x[0] for x in words], ["vier", "10"])  # midpoint 12.5: A before, B after
        self.assertEqual(len(gaps), 1)
        self.assertEqual((gaps[0]["A"], gaps[0]["B"], gaps[0]["R"]), ("vier zehn", "4 10", "vier 10"))
        self.assertEqual(gaps[0]["t"], 12.5)


class Harness(unittest.TestCase):
    def build(self, field):
        dec = fake_decoded(BED, field=field)
        return SH.build_cases(BED, dec)

    def test_perfect_decoder_scores_zero_wrong(self):
        import offset_score as OS

        cases = self.build("label")
        self.assertEqual(len(cases), 15)
        self.assertTrue(all(r["long"] == r["single"] for r in cases))
        with tempfile.TemporaryDirectory() as d:
            SH.write_arm(Path(d), "perfect", cases, BED)
            rows, P = OS.load_arm(d, "perfect")
            T = OS.slot_table(rows, P)
            self.assertEqual(sum(OS.wrong(t) for t in T.values()), 0, {k: t["reads"] for k, t in T.items() if OS.wrong(t)})

    def test_digit_decoder_also_right(self):
        import offset_score as OS

        cases = self.build("text")
        with tempfile.TemporaryDirectory() as d:
            SH.write_arm(Path(d), "digits", cases, BED)
            rows, P = OS.load_arm(d, "digits")
            T = OS.slot_table(rows, P)
            self.assertEqual(sum(OS.wrong(t) for t in T.values()), 0, {k: t["reads"] for k, t in T.items() if OS.wrong(t)})

    def test_oracle_filter_drops_unreadable_slot(self):
        bed = {"f": [{"id": "de_9", "lang": "de", "start": 0.3, "end": 5, "text": "Zeile 34.0202",
                      "label": "Zeile vierunddreißig null zwei null zwei", "values": {"D": 34.0202}}]}
        self.assertEqual(SH.oracle_slots(bed), {"de_9": set()})

    def test_parse_rows_format(self):
        po, pi = SH.parse_rows([{"lang": "ro", "long": "cincizeci și cinci", "single": "55"}])
        self.assertEqual([r["key"] for r in po], ["0:long", "0:single"])
        self.assertEqual(po[0]["alt"], ["50", "5"])
        self.assertEqual(pi[1]["text"], "55")


ROW = {"id": "20120119-0900-PLENARY-3-de_x", "src": "vox", "lang": "de",
       "caption": "in der Zeit von 2000 bis 2010 um 12,5 % und Tel. 0170 1234",
       "text": "in der Zeit von zweitausend bis zwanzig zehn um zwölf Komma fünf Prozent und Tel. null eins sieben null eins zwei drei vier",
       "spans": [{"text": "2000", "kind": "int", "reading": "zweitausend", "canonical": "zweitausend"},
                 {"text": "2010", "kind": "int", "reading": "zwanzig zehn", "canonical": "zweitausendzehn"},
                 {"text": "12,5 %", "kind": "pct", "reading": "zwölf Komma fünf Prozent", "canonical": "zwölf Komma fünf Prozent"},
                 {"text": "0170 1234", "kind": "phone", "reading": "null eins sieben null eins zwei drei vier", "canonical": ""}]}


class Value(unittest.TestCase):
    def outcomes(self, hyp):
        return [(o["value_right"], o["form_match"], o["form_match_join"]) for o in VA.span_outcomes(ROW, hyp, "de")]

    def test_label_is_right_and_form_matches(self):
        self.assertEqual(self.outcomes(ROW["text"]), [(True, True, True)] * 4)

    def test_digits_right_value_no_form(self):
        self.assertEqual(self.outcomes(ROW["caption"]), [(True, False, False)] * 4)

    def test_wrong_and_fused(self):
        o = self.outcomes("in der Zeit von zweitausend bis zweitausendelf um zwölf Komma fünf Prozent und Tel. 0170 1235")
        self.assertEqual([x[0] for x in o], [True, False, True, False])
        o = self.outcomes("von zweitausend bis zwanzigzehn um 12,5 % und Tel. 0170 1234")
        self.assertEqual(o[1], (True, False, True))  # split label, fused hypothesis: join-insensitive match only

    def test_noncanonical_flag(self):
        self.assertEqual([o["noncanonical"] for o in VA.span_outcomes(ROW, ROW["text"], "de")], [False, True, False, False])

    def test_end_to_end_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            C.write_jsonl(d / "refs" / "heldout_de.jsonl", [ROW])
            C.write_jsonl(d / "a" / "heldout_de.jsonl", [{"id": ROW["id"], "hyp": ROW["caption"]}])
            with redirect_stdout(io.StringIO()):
                res = VA.main(["--refs", str(d / "refs"), "--arm", f"stock={d / 'a'}", "--arm", "oracle",
                               "--json", str(d / "o.json")])
            self.assertEqual(res["langs"]["de"]["all"]["stock"]["value_right"], 100.0)
            self.assertEqual(json.load(open(d / "o.json"))["oracle_recall"]["de"], 100.0)


class Misc(unittest.TestCase):
    def test_clusters(self):
        self.assertEqual(C.cluster_of("20180911-0900-PLENARY-witholdRO-de_20180911-18:23:06_3"), "vox:20180911")
        self.assertEqual(C.cluster_of("pt100_00000491_5YMf471zNdI_4266_76_14_94"), "yt:5YMf471zNdI")
        self.assertEqual(C.cluster_of("fleursval_1960_15225685887826839193"), "fleurs:1960")
        self.assertEqual(C.cluster_of("4483338_281.wav"), "call:4483338")
        self.assertEqual(C.cluster_of("cv22_91c7e5", "cv22"), "cv22:cv22_91c7e5")

    def test_bootstrap(self):
        ci = C.ratio_ci({i: (1, 1) for i in range(50)})
        self.assertEqual(ci, [100.0, 100.0])
        lo, hi = C.paired_diff_ci({i: (1 if i % 2 else 0, 0, 1) for i in range(200)})
        self.assertTrue(lo < 50 < hi)

    def test_english_verbalizer(self):
        out, n = VB.verbalize_en("in Q4 we made $3.7 billion, up 12% since 2019 and 1,250 units")
        self.assertEqual(out, "in Q4 we made three point seven billion dollars, up twelve percent since twenty nineteen "
                              "and one thousand two hundred fifty units")
        self.assertEqual(n, [4, 4])


if __name__ == "__main__":
    unittest.main()
