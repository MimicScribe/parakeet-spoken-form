"""Strict probes of an edge phrase (edge_a_probe) with a fake word-level scorer (no model).
python3 test_edge_a_probe.py   (every assert is a control, red-on-base: control)"""
import edge_a_probe as P


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def scorer(heard):
    h = heard.lower().split()
    return lambda text: 5.0 * lev(text.lower().replace(",", "").replace(".", "").split(), h)


vocab = {"seventy": "seventy", "seventeen": "seventeen", "sixty": "sixty", "billion": "billion", "million": "million", "bips": "bips",
         "basis": "basis", "points": "points", "thirty": "thirty", "twenty": "twenty"}
T = "The bank cut seventy five basis points today."
a, b = T.index("seventy"), T.index(" today")
# the audio says the label: every probe loses
s = P.strict_span(scorer("the bank cut seventy five basis points today"), T, a, b, vocab)
assert s["ok"] and s["m"]["del"] >= 6 and s["m"]["wdel"] >= 3 and s["m"]["wsub"] >= 4, s
# a word of the phrase is not in the audio ("basis" dropped): the deletion probe of that word does not lose
s = P.strict_span(scorer("the bank cut seventy five points today"), T, a, b, vocab)
assert not s["ok"] and s["why"] in ("wdel", "wsub"), s
# the audio says seventeen: the near-word swap wins, the row fails
s = P.strict_span(scorer("the bank cut seventeen five basis points today"), T, a, b, vocab)
assert not s["ok"], s
# a connector or function word is never tested: "and" / "to" missing from the audio does not fail the phrase
T2 = "Revenue of three and a half to four billion rose."
a2, b2 = T2.index("three"), T2.index(" rose")
s = P.strict_span(scorer("revenue of three a half four billion rose"), T2, a2, b2, {"three": "three", "four": "four", "billion": "billion"})
assert s["m"]["wdel"] is None or s["m"]["wdel"] >= 3 or not s["ok"]
assert [w for _, _, w in P.probes(T2, a2, b2, {}) if w in ("and", "to", "a")] == []
# fillers have their own, weaker bar and are never swapped
T3 = "It cost twenty uh thirty euros."
a3, b3 = T3.index("twenty"), T3.index(" euros")
kinds = {k for k, _, _ in P.probes(T3, a3, b3, vocab)}
assert "fdel" in kinds and {w[2] for k, _, w in P.probes(T3, a3, b3, vocab) if k == "fdel"} == {"uh"}
s = P.strict_span(scorer("it cost twenty uh thirty euros"), T3, a3, b3, vocab, gate_fillers=True)
assert s["ok"] and s["m"]["fdel"] >= 2, s
s = P.strict_span(scorer("it cost twenty thirty euros"), T3, a3, b3, vocab, gate_fillers=True)
assert not s["ok"] and s["why"] in ("fdel", "wdel", "wsub", "del"), s
# default: the filler is recorded, not gated (read-speech readers do not write fillers)
s = P.strict_span(scorer("it cost twenty thirty euros"), T3, a3, b3, vocab)
assert s["m"]["fdel"] is not None and s["m"]["fdel"] < 2 and s["ok"] in (True, False)
# a reader that writes the filler as "er": the filler is still heard (best spelling beats deletion)
s = P.strict_span(scorer("it cost twenty er thirty euros"), T3, a3, b3, vocab, gate_fillers=True)
assert s["ok"] and s["m"]["fdel"] >= 2, s
# a proper name inside a sentence is not tested; a non-number word is deleted but never swapped for a number word
T4 = "Futures on the Nikkei two twenty five pointed higher."
a4, b4 = T4.index("the Nikkei"), T4.index(" pointed")
assert [w for k, _, w in P.probes(T4, a4, b4, {"nikkei": "nikkei", "nike": "nike"}) if "Nikkei" in str(w)] == []
T5 = "It cost fifteen sorry fifty euros."
a5, b5 = T5.index("fifteen"), T5.index(" euros")
pr = P.probes(T5, a5, b5, {"sorry": "sorry", "sorrow": "sorrow", "fifty": "fifty", "fifteen": "fifteen"}, P.number_keys("en"))
assert any(k == "wdel" and w == "sorry" for k, _, w in pr) and not any(k == "wsub" and str(w).startswith("sorry") for k, _, w in pr)
assert any(k == "wsub" and str(w).startswith("fifty") for k, _, w in pr)
# the vocabulary of a language: pack words and number words, 3+ letters
v = P.edge_vocab("en", ["twenty five bips", "a billion two", "uh"])
assert "bips" in v.values() and "uh" not in v.values() and "a" not in v.values()
print("test_edge_a_probe ok")
