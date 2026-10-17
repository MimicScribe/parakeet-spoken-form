"""Yield fixes (2026-10-16): row CER / caption-gap against the better of two transcripts, and long rows split at
pauses instead of dropped. Local, no GPU: python3 test_ml_p3_yield.py"""
import random

import numpy as np

import ml_cands as M
import ml_p3_core as C


def fake(cap, lang, picks_text, greedy):
    sp = M.spans(cap, lang)
    ds = []
    for s, want in zip(sp, picks_text):
        i = s.cands.index(want) if want in s.cands else 0
        ds.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": i, "margin": 9.0, "fit": True, "decided": True})
    return {"spans": ds, "greedy": greedy}


row = {"audio_filepath": "x.wav", "duration": 2.0, "text": "Leia Provérbios 2.", "lang": "pt", "src": "yodas", "id": "a",
       "origin": "new"}
noisy = "leia problemas doi"  # a real xlsr greedy of a pt YouTube row: CER 0.39 > 0.12
r = fake(row["text"], "pt", ["dois"], noisy)

# 1. old behaviour (no external transcript): dropped on CER
assert C.make_row(row, r)["reason"] == "cer"
# 2. external transcript that matches the label: kept, both CERs recorded
k = C.make_row(row, r, ext="leia provérbios dois")
assert "row" in k, k
assert k["row"]["text"] == "Leia Provérbios dois.", k
assert k["row"]["cer_ctc"] > 0.12 and k["row"]["cer_ext"] == 0.0, k["row"]
# 3. external transcript that also disagrees: still dropped
assert C.make_row(row, r, ext="eu vi o filme ontem à noite")["reason"] == "cer"
# 4. an empty external transcript changes nothing
assert C.make_row(row, r, ext="")["reason"] == "cer"
# 5. caption gap: the greedy misses two words the caption has, the external transcript has them -> kept
row5 = dict(row, src="vox", text="Wir haben am Ende 21 Leute dort gezählt.", lang="de")
r5 = fake(row5["text"], "de", ["einundzwanzig"], "wir haben einundzwanzig leute dort gezählt")
assert C.make_row(row5, r5)["reason"] == "caption_gap"
assert "row" in C.make_row(row5, r5, ext="wir haben am ende einundzwanzig leute dort gezählt")
# 6. ... but a gap both transcripts show stays a drop
assert C.make_row(row5, r5, ext="wir haben einundzwanzig leute dort gezählt")["reason"] == "caption_gap"
print("cer / caption_gap vs external transcript ok")


# ---------------------------------------------------------------- long rows split at pauses
def speechy(dur, quiet):
    """16 kHz noise 'speech' with quiet stretches [(t0, t1)] s."""
    x = np.random.RandomState(0).randn(int(dur * 16000)).astype(np.float32) * 0.3
    for a, b in quiet:
        x[int(a * 16000):int(b * 16000)] *= 0.01
    return x


# 30 s row, 30 tokens of ~0.8 s each, 0.2 s gaps; tokens 12-14 are one number reading ("neunzehnhundert neununddreißig")
times, t = [], 0.1
for i in range(30):
    times.append((t, t + 0.8))
    t += 1.0
gaps = [(times[i][1], times[i + 1][0]) for i in range(29)]
x = speechy(30.5, gaps)
flags = [12 <= i <= 14 for i in range(30)]
ps = C.plan_splits(30, flags, times, x, max_s=20.0)
assert ps, ps
assert all(p["b"] - p["a"] <= 20.0 for p in ps), ps
assert ps[0]["i0"] == 0 and ps[-1]["i1"] == 30 and all(a["i1"] == b["i0"] for a, b in zip(ps, ps[1:])), ps
for p in ps:  # never inside the number reading
    assert not (flags[p["i0"] - 1] and flags[p["i0"]]) if p["i0"] else True, p
# every cut sits in a pause between two words
for p in ps[1:]:
    k = p["i0"]
    assert times[k - 1][1] <= p["a"] <= times[k][0], (p, times[k - 1], times[k])
# a number run spanning the only possible cut region forces the cut elsewhere
flags2 = [8 <= i <= 22 for i in range(30)]
ps2 = C.plan_splits(30, flags2, times, x, max_s=20.0)
assert ps2 is None or all(not (flags2[p["i0"] - 1] and flags2[p["i0"]]) for p in ps2[1:]), ps2
# no quiet frame anywhere -> no split (None), the row stays dropped
assert C.plan_splits(30, flags, times, speechy(30.5, []), max_s=20.0) is None
# a token without a time cannot carry a cut next to it
times3 = list(times)
for i in range(1, 29):
    times3[i] = None
assert C.plan_splits(30, flags, times3, x, max_s=20.0) is None
# a row already short enough: one piece, untouched
assert C.plan_splits(10, [False] * 10, times[:10], x[:int(10.5 * 16000)], max_s=20.0) == [
    {"a": 0.0, "b": 10.5, "i0": 0, "i1": 10}]
print("plan_splits ok")

# piece rows: text, number offsets and spans follow the piece
cap = "Im Jahr 1939 begann der Krieg, und 1945 war er vorbei, sagte er dann."
sp = M.spans(cap, "de")
picks = [s.cands.index("neunzehnhundertneununddreißig") for s in sp[:1]] + [
    sp[1].cands.index("neunzehnhundertfünfundvierzig")]
offs = []
label = C.render_cap(cap, sp, picks, offs)
nr = {"audio_filepath": "p.wav", "duration": 26.0, "text": label, "lang": "de", "src": "vox", "id": "p", "origin": "new",
      "caption": cap, "spans": [{"text": s.text, "reading": s.cands[c]} for s, c in zip(sp, picks)], "num_chars": offs}
toks = label.split()
cut = toks.index("und")  # piece 1 = "Im Jahr ... Krieg,", piece 2 = "und ... dann."
pieces = C.split_rows(nr, [{"a": 0.0, "b": 12.0, "i0": 0, "i1": cut}, {"a": 12.0, "b": 26.0, "i0": cut, "i1": len(toks)}])
assert [p["text"] for p in pieces] == ["Im Jahr neunzehnhundertneununddreißig begann der Krieg,",
                                       "und neunzehnhundertfünfundvierzig war er vorbei, sagte er dann."], pieces
for p in pieces:
    (a, b), = p["num_chars"]
    assert p["text"][a:b] == p["spans"][0]["reading"], p
    assert p["parent"] == "p" and p["duration"] == round(p["piece"][1] - p["piece"][0], 3)
assert [p["id"] for p in pieces] == ["p_p0", "p_p1"]
# each piece carries its own caption, so a check that re-finds the spans in the caption (the positional judge) works
assert [p["caption"] for p in pieces] == ["Im Jahr 1939 begann der Krieg,", "und 1945 war er vorbei, sagte er dann."], pieces
for p in pieces:
    psp = M.spans(p["caption"], "de")
    assert [s.text for s in psp] == [s["text"] for s in p["spans"]], p
    picks = [s.cands.index(i["reading"]) for s, i in zip(psp, p["spans"])]
    assert C.render_cap(p["caption"], psp, picks) == p["text"], p
print("split_rows ok")

# ---------------------------------------------------------------- partial rows: cut around undecided spans (2026-10-16)
# the 30 s row above, an undecided span at tokens 5-6: pieces leave tokens 5-6 out, keep every other stretch
bad = {5, 6}
pp = C.plan_partial(30, flags, bad, times, x, max_s=20.0)
assert pp and all(not (bad & set(range(p["i0"], p["i1"]))) for p in pp), pp
assert pp[0] == {"a": 0.0, "b": round(pp[0]["b"], 3), "i0": 0, "i1": 5}, pp  # the row's start needs no cut
assert all(p["b"] - p["a"] <= 20.0 for p in pp) and pp[-1]["i1"] == 30, pp
for p in pp:  # cuts in pauses, never inside the number reading
    if p["i0"]:
        assert times[p["i0"] - 1][1] <= p["a"] <= times[p["i0"]][0], p
    assert not (p["i0"] and flags[p["i0"] - 1] and flags[p["i0"]]), p
# a stretch too short (2 words) is left out, the rest of the row is not
pp2 = C.plan_partial(30, flags, {2, 27}, times, x, max_s=20.0)
assert all(p["i1"] - p["i0"] >= 3 for p in pp2) and not any(p["i0"] == 0 for p in pp2) and pp2, pp2
# no quiet frame: nothing (the row stays a drop)
assert C.plan_partial(30, flags, bad, times, speechy(30.5, []), max_s=20.0) == []
# a short row with an undecided span in the middle
assert C.plan_partial(10, [False] * 10, {4}, times[:10], x[:int(10.5 * 16000)], max_s=20.0) == [
    {"a": 0.0, "b": round(C.quiet_cut(x, times[3][1], times[4][0]), 3), "i0": 0, "i1": 4},
    {"a": round(C.quiet_cut(x, times[4][1], times[5][0]), 3), "b": 10.5, "i0": 5, "i1": 10}]
# the undecided span's tokens in the label
cap = "Im Jahr 1939 begann der Krieg, und 1945 war er vorbei."
sp = M.spans(cap, "de")
offs = []
label = C.render_cap(cap, sp, [0, 0], offs)
assert C.undecided_tokens(label, offs, [{}, {"undecided": True}]) == {label.split().index("neunzehnhundertfünfundvierzig")}
print("plan_partial ok")
