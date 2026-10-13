"""Local unit tests for ml_p3_core (no GPU): python3 test_ml_p3.py"""
import numpy as np

import ml_cands as M
import ml_p3_core as C


def fake(cap, lang, picks_text, decided=True, greedy=None):
    sp = M.spans(cap, lang)
    ds = []
    for s, want in zip(sp, picks_text):
        i = s.cands.index(want) if want in s.cands else 0
        ds.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": i, "margin": 9.0, "fit": True, "decided": decided})
    return {"spans": ds, "greedy": greedy if greedy is not None else M.render(cap, sp, [d["pick"] for d in ds]).lower()}


row = {"audio_filepath": "x.wav", "duration": 3.0, "text": "Es waren 21 Leute da.", "lang": "de", "src": "vox", "id": "a", "origin": "d2"}
r = C.make_row(row, fake(row["text"], "de", ["einundzwanzig"]))
assert r["row"]["text"] == "Es waren einundzwanzig Leute da.", r
assert r["row"]["spans"][0]["reading"] == "einundzwanzig"
# undecided -> dropped
assert C.make_row(row, fake(row["text"], "de", ["einundzwanzig"], decided=False)) == {"reason": "undecided"}
# a second digit outside any span would remain -> digit_left / no candidates
row2 = dict(row, text="Es waren 3-4 Leute da.")
d = {"spans": [{"text": s.text, "kind": s.kind, "decided": False, "why": "range/sign"} for s in M.spans(row2["text"], "de")], "greedy": "x"}
assert C.make_row(row2, d)["reason"] == "no_candidates"
# CER guard
rr = C.make_row(row, fake(row["text"], "de", ["einundzwanzig"], greedy="voellig anderer satz hier"))
assert rr["reason"] == "cer", rr
# gender variant kept
row3 = dict(row, text="Ich habe 1 Frage.")
f = fake(row3["text"], "de", ["eine"])
assert C.make_row(row3, f)["row"]["text"] == "Ich habe eine Frage.", C.make_row(row3, f)
# percent / decimal / year
for cap, lang in [("Rund 3,5 % mehr.", "de"), ("En 1984 hubo 12 casos.", "es")]:
    ro = dict(row, text=cap, lang=lang)
    out = C.make_row(ro, fake(cap, lang, [s.cands[0] for s in M.spans(cap, lang)]))
    assert "row" in out and not any(c.isdigit() for c in out["row"]["text"]), (cap, out)
    print(cap, "->", out["row"]["text"])
# quiet edges
t = np.arange(16000 * 3) / 16000
x = (np.random.RandomState(0).randn(len(t)) * (0.2 + 0.1 * np.sin(2 * np.pi * 3 * t))).astype("float32")
assert not C.edge_quiet(x)
x[:3200] = 0
x[-3200:] = 0
assert C.edge_quiet(x)
a = C.Acc("de")
a.add(row, fake(row["text"], "de", ["einundzwanzig"]), C.make_row(row, fake(row["text"], "de", ["einundzwanzig"])))
s = a.stats([C.make_row(row, fake(row["text"], "de", ["einundzwanzig"]))["row"]])
assert s["rows_out"]["d2"] == 1 and s["decide_rate_d2"] == 1.0
r = C.make_row(dict(row, text="2009 wurde es 1 Euro."), fake("2009 wurde es 1 Euro.", "de", ["zweitausendneun", "eins"]))
assert r["row"]["text"].startswith("Zweitausendneun wurde"), r
r = C.make_row(row, fake(row["text"], "de", ["einundzwanzig"]))
assert r["row"]["text"] == "Es waren einundzwanzig Leute da."
print("ok")
