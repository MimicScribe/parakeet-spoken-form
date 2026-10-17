"""Local tests for en_win (no GPU, no Modal): python3 test_en_win.py"""
import random

import en_win as E
import ml_balance as B
import ml_p3_core as C


def sp(reading, kind="year", decided=True):
    return {"kind": kind, "text": "x", "cands": [reading], "pick": 0, "decided": decided}


# ---- prep: readings found in the crop text, in order, whole words only, undecided skipped -----------------------
t = "In nineteen eighty two, women earn fifty four percent of degrees, and fifty percent of these."
offs = E.reading_offsets(t, [sp("nineteen eighty two"), sp("fifty four percent", "percent"),
                             sp("fifty percent", "percent"), sp("seven", decided=False)])
assert [t[a:b] for a, b, _ in offs] == ["nineteen eighty two", "fifty four percent", "fifty percent"], offs
# a reading of the parent that the crop does not hold is skipped; whole words only ("one" not inside "someone")
assert E.reading_offsets("someone said ten", [sp("one", "int"), sp("ten", "int")])[0][:2] == (13, 16)
# the same reading twice: two distinct ranges
o2 = E.reading_offsets("five years or five more", [sp("five", "spelled"), sp("five", "spelled")])
assert [(a, b) for a, b, _ in o2] == [(0, 4), (14, 18)], o2
# labels with a digit / symbol never become rows
crop = {"audio_filepath": "/vol/d1/wav3/x/en000_1_abcdefghijk_12_3_4_5_1.wav", "duration": 3.0, "text": "it cost 5 dollars",
        "src": "en000_1_abcdefghijk_12_3_4_5", "video": "abcdefghijk"}
assert E.label_row(crop, [], "d1") is None
r = E.label_row(dict(crop, text="it cost  five dollars"), [sp("five dollars", "money")], "d1")
assert r["text"] == "it cost five dollars" and r["num_chars"] == [[8, 20]] and r["src"] == "yodas"
assert r["id"] == "en000_1_abcdefghijk_12_3_4_5_1" and C.stream_key(r)[0] == "yt:abcdefghijk"
v = E.label_row({"audio_filepath": "/vol/v/20091216-0900-PLENARY-8_en_101_1.wav", "duration": 2.0, "text": "two",
                 "src": "20091216-0900-PLENARY-8_en_101", "video": "20091216-0900-PLENARY-8_en"}, [], "vox")
assert v["src"] == "vox" and C.stream_key(v)[0] == "vox:20091216-0900-PLENARY-8"
# held out: YouTube dev video, VoxPopuli held-out day
assert E.held_out(r, {"abcdefghijk"}, set()) and not E.held_out(r, set(), {"20091216"})
assert E.held_out(v, set(), {"20091216"}) and not E.held_out(v, set(), {"20091217"})
# one crop per parent: the one with more readings
a = dict(r, id="p_0", parent="P", num_chars=[[0, 1]])
b = dict(r, id="p_1", parent="P", num_chars=[[0, 1], [2, 3]])
assert [x["id"] for x in E.one_per_parent([a, b, dict(r, id="q_0", parent="Q")])] == ["p_1", "q_0"]

# ---- number groups: labeller readings + unmarked number-word runs ------------------------------------------------
g = E.number_groups("we had two hundred and five cars and one dog", [])
assert g == [frozenset({2, 3, 4, 5})], g  # "and" joins only between number words; "one" alone is no group
g = E.number_groups("in nineteen eighty two, twenty more", [[3, 22]])
assert frozenset({1, 2, 3}) in g and len(g) == 1, g  # "twenty more": single word; comma ends a run
g = E.number_groups("call oh two oh one now", [])
assert g == [frozenset({1, 2, 3, 4})], g
assert E.number_groups("twenty-two and five-year plans", []) == [frozenset({0, 1})] or \
    E.number_groups("twenty-two and five-year plans", []) == [], "hyphenated forms parse without error"


# ---- windows never split a reading; pairs split only their own reading, exactly once ---------------------------
def mk(text, num_chars, rid, start=0.2, step=0.45):
    toks = text.split(" ")
    times = [[round(start + i * step, 3), round(start + i * step + 0.3, 3)] for i in range(len(toks))]
    return {"text": text, "word_times": times, "num_chars": num_chars, "spans": [{"kind": "year"}] * len(num_chars),
            "duration": times[-1][1] + 0.3, "id": rid, "src": None}


s1 = "so back in nineteen eighty two the plant made fifty four percent of the cars we had"
s2 = "and later two hundred and five more came in a year or so after that time"
stream = [mk(s1, [[11, 30], [46, 64]], "a"), mk(s2, [], "b"), mk(s1, [[11, 30], [46, 64]], "c")]
n_split = n_ok = 0
for seed in range(60):
    for w in C.build_windows(stream, random.Random(seed), win=4.0, stride=1.7, jitter=0.6, min_words=1):
        lab = {tuple(x) for x in w["src"]}
        bad = E.split_groups(stream, lab, w["rows"])
        assert E.window_ok(stream, w) == (not bad)
        for i, grp in bad:
            n = sum((i, k) in lab for k in grp)
            assert 0 < n < len(grp)
        n_split += bool(bad)
        n_ok += not bad
        if not bad:  # every reading in the label is whole
            for i in w["rows"]:
                for grp in E.number_groups(stream[i]["text"], stream[i]["num_chars"]):
                    n = sum((i, k) in lab for k in grp)
                    assert n in (0, len(grp)), (w["text"], grp)
assert n_split and n_ok, (n_split, n_ok)  # both cases occur on this geometry

kept = rej = 0
for p in B.seam_crops(stream, random.Random(1), 0, win=2.5):
    ok = E.pair_ok(stream, p)
    kept += ok
    rej += not ok
    if ok:
        joined = " ".join(x for x in (p["left"]["reading_part"], p["right"]["reading_part"]) if x)
        assert joined == p["reading"], (p["pos"], joined, p["reading"])
        if p["pos"].startswith("gap"):  # a cut between two reading words: each half holds a part
            assert p["left"]["reading_part"] and p["right"]["reading_part"], p["pos"]
assert kept and rej, (kept, rej)  # some far edges split the other reading / the unmarked run: rejected

# a hand-made pair that splits two readings is rejected; one that splits only its own reading passes
L = {"src": [[0, k] for k in range(0, 5)], "rows": [0]}  # "so back in nineteen eighty"
R = {"src": [[0, k] for k in range(5, 10)], "rows": [0]}  # "two the plant made fifty" (cuts "fifty four percent")
assert not E.pair_ok(stream, {"left": L, "right": R, "reading": "nineteen eighty two"})
R2 = {"src": [[0, k] for k in range(5, 9)], "rows": [0]}  # "two the plant made"
assert E.pair_ok(stream, {"left": L, "right": R2, "reading": "nineteen eighty two"})
assert not E.pair_ok(stream, {"left": L, "right": R2, "reading": "eighty two"})  # the reading text must match

# ---- CTC spelling ------------------------------------------------------------------------------------------------
assert E.greedy_norm("FIFTY PER CENT AT TEN P M O ONE") == "fifty percent at ten pm oh one"
voc = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")}
voc["|"] = 99
ids = E.en_ids("percent", voc, 99)
assert 99 in ids and len(ids) == len("PERCENT") + 1  # LibriSpeech spells it as two words
assert E.en_ids("oh", voc, 99) == [voc["O"]]

# ---- shards keep a video together; selection caps ---------------------------------------------------------------
ra = dict(r, id="en000_1_abcdefghijk_12_1")
rb = dict(r, id="en000_1_abcdefghijk_99_0")
assert E.shard_of(ra, 7) == E.shard_of(rb, 7)
ws = [{"duration": 15.0, "edge_number": i % 3 == 0, "id": i} for i in range(30)]
sel = E.select_windows(ws, 15.0 * 10 / 3600)
assert len(sel) == 10 and all(w["edge_number"] for w in sel)  # 10 edge-number windows exist: they come first
halves = []
for i, (pc, kind) in enumerate([("gap", "year")] * 3 + [("in_word", "percent")] * 3 + [("before", "year")] * 3):
    for side in ("left", "right"):
        halves.append({"pair": f"p{i}", "side": side, "pos_class": pc, "kind": kind, "duration": 5.0})
out = E.select_pairs(halves[:-1], 4)  # the last pair is incomplete: never taken
assert len(out) == 8 and {h["pos_class"] for h in out} == {"gap", "in_word"}
assert [h["side"] for h in out[:2]] == ["left", "right"]
out = E.select_pairs(halves, 8)
assert len(out) == 16 and sum(h["pos_class"] == "before" for h in out) == 4 and len({h["pair"] for h in out}) == 8
print("test_en_win: ok")

# held-out days cover every session type of the day (committee sessions were missed: 2026-10-17)
import en_win as _E  # noqa: E402
assert _E.held_out({"parent": "20090330-1630-COMMITTEE-ECON_en_9"}, set(), {"20090330"})
assert _E.held_out({"parent": "20090115-0900-PLENARY-3_en_12"}, set(), {"20090115"})
assert not _E.held_out({"parent": "20090115-0900-PLENARY-3_en_12"}, set(), {"20100101"})
print("test_en_win: held-out day ok")
