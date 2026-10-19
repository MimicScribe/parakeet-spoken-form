"""Local tests for realx1_core / realx1_select / realx1_geom (no GPU): python3 test_realx1_core.py"""
import random

import realx1_core as R
import realx1_geom as G
import realx1_select as S

# probes: a word deleted / replaced by a near vocabulary word, per word of the reading
T = "en dos mil diecinueve pagamos tres coma cinco millones de euros"
a = T.index("tres")
b = T.index(" de euros")
us = R.units(T, a, b, "es")
assert [u["word"] for u in us] == ["tres", "coma", "cinco", "millones"], us
kinds = [k for k, _ in R.variants(T, a, b, "es")]
assert kinds.count("wdel") == 4 and kinds.count("wsub") >= 8, kinds
assert all("tres coma cinco millones" not in t for _, t in R.variants(T, a, b, "es"))
# compound languages: a morpheme is deleted / replaced inside the token
T2 = "es waren zweihundertdreiundzwanzig euro"
a2 = T2.index("zwei")
b2 = a2 + len("zweihundertdreiundzwanzig")
vs = dict((t, k) for k, t in R.variants(T2, a2, b2, "de"))
assert "es waren hundertdreiundzwanzig euro" in vs and vs["es waren hundertdreiundzwanzig euro"] == "wdel"
assert "es waren dreihundertdreiundzwanzig euro" in vs

# a model that prefers the label by a wide margin passes every probe; one that cannot tell "tres" from "diez" fails wsub
base = 100.0


def sure(text):
    return base if text == T else base + 30.0


res = R.span_probe(sure, T, a, b, "es", ["tres coma cinco millones", "tres y medio millones"], R.D.key("tres coma cinco millones"))
assert R.span_ok(res, True) == (True, ""), res
unsure = lambda text: base if text == T else base + (1.0 if "diez" in text else 30.0)  # noqa: E731
res = R.span_probe(unsure, T, a, b, "es", [], "x")
ok, why = R.span_ok(res, True)
assert not ok and why == "wsub", (ok, why, res)
assert R.span_ok(res, False)[0] is True  # a CTC-decided span keeps the labelling bar (no candidates: inf)
# a word the audio does not need: deleting it costs nothing -> wdel
free = lambda text: base if text == T else (base - 5.0 if "coma" not in text else base + 30.0)  # noqa: E731
ok, why = R.span_ok(R.span_probe(free, T, a, b, "es", [], "x"), True)
assert not ok and why in ("wdel", "del"), (ok, why)

# the edge check of a window cut through a number
assert G.edges_ok_cut("dos mil quinientos", "dos mil quinientos", None, None, [False, False])
assert not G.edges_ok_cut("dos mil quinientos", "dos mil quinientas", None, None, [False, False])
assert G.edges_ok_cut("mil quinientos", "os mil quinientos", "dos", None, [True, False])  # tail of the dropped "dos"
assert not G.edges_ok_cut("mil quinientos", "os mil quinientos", "dos", None, [False, False])

# cuts: inside a reading only where the letter gap is wide enough
words = [("a", 0.0, 0.3, None, 0, 0), ("dos", 0.4, 0.7, (0, 0), 0, 1), ("mil", 0.72, 1.0, (0, 0), 0, 2), ("tres", 1.2, 1.5, (0, 0), 0, 3)]
cuts = G.cut_list(words, 2.0, lambda t0, t1: None)
inside = [c for c in cuts if c[2] is not None]
assert len(inside) == 1 and inside[0][1] == 3, cuts  # only the 0.2 s gap, not the 0.02 s one

# selection: caps are counted against the mix
assert S.eligible({"spans": [{}], "num_chars": [[0, 1]], "word_times": [[0, 1]], "text": "x", "duration": 3.0, "cer_ctc": 0.1}) is None
assert S.eligible({"spans": [{}], "num_chars": [[0, 1]], "word_times": [[0, 1]], "text": "x1", "duration": 3.0, "cer_ctc": 0.1}) == "digit_in_label"
print("ok")
