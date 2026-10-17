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

# v2: reading offsets, number-token flags, window-edge crop planning
offs = []
cap = "Custa R$ 1,50 hoje e 2009 foi bom."
sp = M.spans(cap, "pt")
lab = C.render_cap(cap, sp, [0] * len(sp), offs)
fl = C.number_tokens(lab, offs)
toks = lab.split()
assert [t for t, f in zip(toks, fl) if f] == [w for o in offs for w in lab[o[0]:o[1]].split()], (lab, offs, fl)
assert not fl[0] and not fl[-1], fl
import random as _r
SR = 16000
toks_t = [(0.3 * i, 0.3 * i + 0.15) for i in range(len(toks))]  # 150 ms words, 150 ms gaps
dur = 0.3 * len(toks) + 0.3
xx = np.zeros(int(dur * SR), dtype="float32")
rs = np.random.RandomState(1)
for a, b in toks_t:
    xx[int(a * SR):int(b * SR)] = rs.randn(int(b * SR) - int(a * SR)) * 0.2
xx += rs.randn(len(xx)).astype("float32") * 0.002
cps = C.plan_crops(len(toks), fl, toks_t, xx, _r.Random(0))
assert cps and all(c["b"] - c["a"] >= C.MIN_CROP for c in cps), cps
assert {c["type"].split("_")[1] for c in cps} <= {"start", "end"} and len(cps) <= 2, cps
for c in cps:
    if c["type"] == "in_start":
        assert fl[c["i0"]] and fl[c["i0"] - 1]
    if c["type"] == "in_end":
        assert fl[c["i1"] - 1] and fl[c["i1"]]
    cut = c["a"] if c["type"].endswith("start") else c["b"]
    assert any(b <= cut <= a2 for (_, b), (a2, _) in zip(toks_t, toks_t[1:])), (cut, "cut must sit in a gap")
# gaps under MIN_GAP -> no cut
assert C.plan_crops(len(toks), fl, [(0.3 * i, 0.3 * i + 0.25) for i in range(len(toks))], xx, _r.Random(0)) == []
# no quiet frame in the gap (continuous speech) -> no cut
loud = (rs.randn(len(xx)) * 0.2).astype("float32")
assert C.plan_crops(len(toks), fl, toks_t, loud, _r.Random(0)) == []
# edge words must match exactly; a leftover tail word or a clipped edge word rejects the crop
assert C.edges_match("e vinte e dois", "e vinte e dois")
assert not C.edges_match("e vinte e dois", "ois e vinte e dois")
assert not C.edges_match("vinte e dois", "inte e dois")
# phone groups: the separator is replaced, one space kept
cap = "Ligue 11-9-4704-8697 agora."
sp = M.spans(cap, "pt")
lab = C.render_cap(cap, sp, [0] * len(sp))
assert "-" not in lab and "  " not in lab, lab
print("v2 ok", lab)

# streaming-geometry windows
def _row(text, t0, words_dur=0.3, gap=0.15, lang="es"):
    toks = text.split()
    wt = [[round(t0 + i * (words_dur + gap), 3), round(t0 + i * (words_dur + gap) + words_dur, 3)] for i in range(len(toks))]
    dur = wt[-1][1] + 0.2
    return {"text": text, "duration": round(dur, 3), "word_times": wt, "audio_filepath": "x.wav", "lang": lang}
r1 = _row("el precio subió a veinte y tres euros este año en la región", 0.2)
r1["num_chars"] = [[r1["text"].index("veinte"), r1["text"].index("veinte") + len("veinte y tres euros")]]
r2 = _row("y hubo doscientos cuarenta y cinco casos nuevos en total", 0.2)
r2["num_chars"] = [[r2["text"].index("doscientos"), r2["text"].index("doscientos") + len("doscientos cuarenta y cinco")]]
rows_s = [r1, r2, r1, r2, r1]
ws = []
for seed in range(30):
    ws += C.build_windows(rows_s, _r.Random(seed))
assert ws and all(w["b"] - w["a"] <= C.WIN + 1e-6 for w in ws)
assert any(w["cut_number"] for w in ws) and any(w["edge_number"] for w in ws), "numbers must land at and across edges"
segs, words, total = C.stream_words(rows_s)
for w in ws:  # labelled = words with >= KEEP_FRAC of their sound inside; no word between DROP_FRAC and KEEP_FRAC
    toks = w["text"].split()
    near = [x for x in words if C.inside_frac(x, w["a"], w["b"]) > 0]
    assert not any(C.DROP_FRAC < C.inside_frac(x, w["a"], w["b"]) < C.KEEP_FRAC for x in near), w
    inside = [x for x in near if C.inside_frac(x, w["a"], w["b"]) >= C.KEEP_FRAC]
    assert toks == [x[0] for x in inside], (w, inside[:3])
    assert w["num_tokens"] == [x[3] for x in inside]
    assert w["kept_cut_words"] == sum(C.inside_frac(x, w["a"], w["b"]) < 1 - 1e-9 for x in inside)
assert any(w["kept_cut_words"] for w in ws) and any(w["cut_words"] for w in ws), "both edge rules must be exercised"
print("windows ok", len(ws), sum(w["cut_number"] for w in ws), "split numbers")

# stream grouping
rs_ = [{"id": "20160913-0900-PLENARY-21-es_20160913-23:07:5%d_2" % i, "src": "vox"} for i in range(5)] + \
      [{"id": "pt100_00000486_dJjRBXP8rfo_%d_26_2_14" % (100 * i), "src": "yodas"} for i in range(3)] + \
      [{"id": "fleurs_%d" % i, "src": "fleurs"} for i in range(4)]
gs = C.group_streams(rs_, _r.Random(0), max_rows=4)
keyed = [g for g in gs if len({C.stream_key(r)[0] for r in g}) == 1 and C.stream_key(g[0])[0] != "misc"]
assert len(keyed) == 2 and sum(len(g) for g in gs) >= 11, [len(g) for g in gs]  # leftovers are stitched together
vox = [g for g in gs if C.stream_key(g[0])[0].startswith("vox")][0]
assert [r["id"] for r in vox] == sorted(r["id"] for r in vox)
print("streams ok", [len(g) for g in gs])

# review M2: an audible fragment of the cut word may appear at the edge of the re-decode
assert C.edges_match("we opened the store", "astly we opened the store", cut_left="Lastly,")
assert not C.edges_match("we opened the store", "astly we opened the store")
assert C.edges_match("veinte y tres euros", "veinte y tres euros es", cut_right="este")
assert not C.edges_match("veinte y tres euros", "veinte y tres euros es", cut_right="nada")
print("edge fragments ok")

# second-recognizer tie-break (ml_decide.ext_agrees): whole words only, any rival heard refuses
import ml_decide as _D
assert not _D.ext_agrees(["twintig", "twintigste"], 0, "het kost twintigduizend euro")  # caption-value error stays out
assert _D.ext_agrees(["twintig"], 0, "het kost twintig euro")
assert _D.ext_agrees(["fünfte", "fünften"], 1, "erst nach der fünften Klasse")
assert not _D.ext_agrees(["trente trois et quatre vingt dix", "trente trois dollars quatre vingt dix"], 0,
                         "trente-trois dollars quatre-vingt-dix")
assert _D.ext_agrees(["vingt et un"], 0, "il a vingt-et-un ans")
assert not _D.ext_agrees(["seis nueve", "sesenta y nueve"], 0, "once sesenta y nueve setenta")
print("ext tie-break ok")
# the scale guard also runs on the second transcript ("8 puestos" spoken "ocho mil puestos")
import ml_cands as _M
_sp = [s for s in _M.spans("hay 8 puestos", "es") if s.cands][0]
assert not _D.context_fits("hay 8 puestos", _sp, "ocho", "hay ocho mil puestos", "es")
assert _D.context_fits("hay 8 puestos", _sp, "ocho", "hay ocho puestos", "es")
print("ext scale guard ok")
