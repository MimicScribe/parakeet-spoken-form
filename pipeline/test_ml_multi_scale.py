"""python3 -m unittest test_ml_multi_scale  (pure parts of ml_multi_scale: selection, pieces, assembly)."""
import unittest

import ml_multi_scale as S


def _times(toks, step=0.4, gap=0.1):
    return [(round(i * step, 3), round(i * step + step - gap, 3)) for i in range(len(toks))]


def _words(text, t0=0.0, step=0.4):
    return [{"w": w, "s": int((t0 + i * step) * 1000), "e": int((t0 + i * step + 0.3) * 1000)} for i, w in enumerate(text.split())]


class Selection(unittest.TestCase):
    def _rows(self):
        rows = []
        for i in range(400):
            y = 1990 + i % 41
            rows.append({"utt_id": f"2018010{i % 9}-0900-PLENARY-{i % 50}_pl_{i}", "day": f"2018010{i % 9}",
                         "text": f"W {y} roku przyjęliśmy rezolucję numer {i} w sprawie budżetu na kolejne lata, panie przewodniczący."})
        rows.append({"utt_id": "20180101-0900-PLENARY-77_pl_1", "day": "20180101",
                     "text": "Posiedzenie zostanie wznowione o godz. 15.30, a potem zagłosujemy nad całym tekstem."})
        return rows

    def test_template_and_session_caps(self):
        sel, st = S.select(self._rows(), "pl", ["2018"], per_template=3, per_session=8)
        self.assertLessEqual(st["max_per_template"], 3)
        self.assertLessEqual(st["max_per_session"], 8)
        self.assertTrue(any(r["why"] == "clock" for r in sel))

    def test_only_chosen_years(self):
        sel, _ = S.select(self._rows(), "pl", ["2017"])
        self.assertEqual(sel, [])

    def test_ro_price_with_de(self):
        self.assertTrue(S.classes("Am alocat 20 de milioane de euro pentru acest program.", "ro")["cur"])
        self.assertTrue(S.classes("Biletul costă 15 lei.", "ro")["cur"])

    def test_span_kinds(self):
        self.assertEqual(S.span_kind("2013 r.", "W 2013 r. Komisja", 2, "pl"), ("int", "year"))
        self.assertEqual(S.span_kind("5 mln zł", "wydamy 5 mln zł", 7, "pl")[0], "cur")
        self.assertEqual(S.span_kind("3,5%", "wzrost o 3,5%", 9, "pl")[0], "pct")
        self.assertEqual(S.span_kind("15.30", "o godz. 15.30", 8, "pl")[0], "clock")


class Pieces(unittest.TestCase):
    def setUp(self):
        self.text = "Na początku sesji powiem coś. W 2013 roku wydaliśmy 5 mln zł na drogi, a potem 7 projektów upadło."
        a = self.text.index("2013")
        b = self.text.index("5 mln zł")
        c = self.text.index("7 projektów")
        self.spans = [{"written": "2013", "a": a, "b": a + 4, "cands": ["dwa tysiące trzynastym"]},
                      {"written": "5 mln zł", "a": b, "b": b + 8, "cands": ["pięć milionów złotych"]},
                      {"written": "7", "a": c, "b": c + 1, "cands": []}]

    def test_render_provenance(self):
        toks = S.render_tokens(self.text, self.spans, [0, 0, None])
        self.assertIn(("trzynastym", 0), toks)
        self.assertIn(("złotych", 1), toks)
        self.assertIn(("7", 2), toks)  # no reading: the written digits stay, marked with their span

    def test_pieces_avoid_dead_span(self):
        toks = S.render_tokens(self.text, self.spans, [0, 0, None])
        times = _times(toks)
        ps = S.plan_pieces(toks, times, {0, 1}, {2}, dur=times[-1][1] + 0.2)
        self.assertTrue(ps)
        dead_i = [i for i, (_t, j) in enumerate(toks) if j == 2][0]
        for p in ps:
            self.assertFalse(p["i0"] <= dead_i < p["i1"])
            self.assertLess(p["t1"] - p["t0"], 15)
        covered = {j for p in ps for j in p["spans"]}
        self.assertEqual(covered, {0, 1})

    def test_piece_caption_rebased(self):
        toks = S.render_tokens(self.text, self.spans, [0, 0, None])
        ps = S.plan_pieces(toks, _times(toks), {0, 1}, {2}, dur=50)
        cap, sp = S.piece_caption(self.text, self.spans, toks, ps[0]["i0"], ps[0]["i1"])
        for s in sp:
            self.assertEqual(cap[s["a"]:s["b"]], s["written"])

    def test_num_chars_no_punctuation(self):
        toks = [("Mamy", None), ("pięć,", 0), ("sześć", 0), ("lat.", None)]
        nc = S._num_chars(toks, {0})
        text = " ".join(t for t, _ in toks)
        self.assertEqual([text[a:b] for a, b in nc], ["pięć", "sześć"])


class Fit(unittest.TestCase):
    def test_polish_fit_is_strict(self):
        S._FIT_LANG[0] = "pl"
        self.assertFalse(S.fits("dwadzieścia do dwadzieścia pięć procent", "mamy dwadziescia dwadzieścia pięć procenty emisji"))
        self.assertTrue(S.fits("dwa tysiące trzynastym", "w dwa tysiące trzynastym roku"))

    def test_romanian_fit_tolerates_noisy_greedy(self):
        S._FIT_LANG[0] = "ro"
        self.assertTrue(S.fits("doi mii douăzeci și patru", "acum nu î domi dozeci și patru d"))
        S._FIT_LANG[0] = ""


class GreedyVeto(unittest.TestCase):
    def test_rival_closer_to_greedy(self):
        c = ["czternastego trzydzieści", "czternastej trzydzieści"]
        self.assertTrue(S.greedy_veto(c, 0, "my się a o czternastej trzydzieści wznowimy nasze obrady"))
        self.assertFalse(S.greedy_veto(c, 1, "my się a o czternastej trzydzieści wznowimy nasze obrady"))


class Assemble(unittest.TestCase):
    def _case(self, ext_text_by_piece, cer=0.1):
        text = "Komisja mówi, że w 2013 roku wydaliśmy 5 mln zł na drogi w regionie."
        a, b = text.index("2013"), text.index("5 mln zł")
        spans = [{"written": "2013", "a": a, "b": a + 4, "cands": ["dwa tysiące trzynastym"]},
                 {"written": "5 mln zł", "a": b, "b": b + 8, "cands": ["pięć milionów złotych"]}]
        toks = S.render_tokens(text, spans, [0, 0])
        times = _times(toks)
        pieces = S.plan_pieces(toks, times, {0, 1}, set(), dur=times[-1][1] + 0.2)
        sc = {"tokens": toks, "good": [0, 1], "dead": [], "pick": [0, 0], "times": times, "pieces": pieces,
              "dur": times[-1][1] + 0.2, "cer_ctc": cer,
              "dec": {"0": {"margin": 9.0, "top": [], "fit": True}, "1": {"margin": 7.0, "top": [], "fit": True}}}
        ext = {}
        for k, pc in enumerate(pieces):
            cap, _ = S.piece_caption(text, spans, toks, pc["i0"], pc["i1"])
            et = ext_text_by_piece(cap)
            ext[f"x_p{k}"] = {"status": "completed", "words": _words(et)}
        return S.assemble_row({"id": "x", "text": text, "spans": spans}, sc, ext, "pl")

    def test_whole_clip_when_all_pass(self):
        res = self._case(lambda cap: cap.replace(",", "").replace(".", "").lower())
        self.assertTrue(res["whole"], res)
        row = res["rows"][0]
        self.assertEqual(row["origin"], "new")
        self.assertIn("dwa tysiące trzynastym roku", row["text"])
        self.assertNotRegex(row["text"], r"\d")
        for a, b in row["num_chars"]:
            self.assertNotRegex(row["text"][a:b], r"[,.;:!?]")
        self.assertEqual({sp["by"] for sp in row["spans"]}, {"ctc"})

    def test_value_mismatch_drops_span(self):
        res = self._case(lambda cap: cap.replace("2013", "2014").replace(",", "").lower())
        self.assertFalse(res["whole"])
        self.assertNotIn(0, res["span_ok"])
        self.assertTrue(all("2013" not in r["caption"] or "trzynastym" not in r["text"] for r in res["rows"]))

    def test_clipped_edge_still_checks_whole_clip(self):
        # the cut lost the piece's first word: no piece row, but the values and inner words still check the clip
        res = self._case(lambda cap: " ".join(cap.replace(",", "").replace(".", "").lower().split()[1:]))
        self.assertTrue(res["whole"], res)

    def test_core_agreement_ignores_edges(self):
        words = [{"w": w} for w in "b c d e f g h".split()]
        self.assertEqual(S.core_agreement("a b c d e f g h", words), 1.0)
        self.assertLess(S.core_agreement("a b c x e f g h", words), 1.0)

    def test_single_reading_with_separator_word_needs_fit(self):
        self.assertFalse(S.single_safe("pięć przecinek trzy", "pl"))  # "ASP 5.3" was said "pięć kreska trzy"
        self.assertTrue(S.single_safe("două mii douăzeci și cinci", "ro"))

    def test_bad_ctc_fit_keeps_no_whole_clip(self):
        res = self._case(lambda cap: cap.lower(), cer=0.4)
        self.assertFalse(res["whole"])


if __name__ == "__main__":
    unittest.main()
