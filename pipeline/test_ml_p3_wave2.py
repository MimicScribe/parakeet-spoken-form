"""Local tests for the windows / seam-crop helpers beyond wave 1 (ml_p3_core, ml_balance; no GPU, no Modal):
python3 test_ml_p3_wave2.py"""
import random

import ml_balance as B
import ml_p3_core as C

# held-out days: every session type and the carve's own group lines
held = ["# carve_heldout.py first list for cs", "# group vox:20150921", "20150518-0900-PLENARY-18_cs_13_t0",
        "20150921-1500-COMMITTEE-ECON_cs_7", "cv22_00d6a2435fa255ad", ""]
assert C.held_days(held) == {"20150921", "20150518"}, C.held_days(held)
assert C.vox_day("20150921-0900-COMMITTEE-AGRI_cs_124_t0") == "20150921"
assert C.vox_day("20150518-0900-PLENARY-3_cs_1") == "20150518"
assert C.vox_day("pl000_00000109_ZiL9-KuoSwA_319_52_4_04") is None
assert C.vox_day("cv22_00d6a2435fa255ad") is None

# word times: every token timed, one entry per token
assert C.has_times({"text": "a b", "word_times": [[0, 0.1], [0.2, 0.3]]})
assert not C.has_times({"text": "a b", "word_times": [[0, 0.1], None]})
assert not C.has_times({"text": "a b c", "word_times": [[0, 0.1], [0.2, 0.3]]})
assert not C.has_times({"text": "a b"})

# solo rows never join a stitched stream
rows = [{"id": "cv22_a", "src": "cv22"}, {"id": "uk000_1_abcdefghijk_1", "src": "yodas"}, {"id": "cv22_b", "src": "cv22"}]
stitch, solo = C.split_solo(rows, ("cv22",))
assert [r["id"] for r in stitch] == ["uk000_1_abcdefghijk_1"]
assert [[r["id"] for r in s] for s in solo] == [["cv22_a"], ["cv22_b"]]
assert C.split_solo(rows, ()) == (rows, [])

# a seam pair cut inside ONE clip: the two halves partition the reading (left + right = the full reading)
text = "Ми заплатили двісті сорок п'ять гривень за квиток"
toks = text.split()
times, t = [], 0.3
for w in toks:
    d = 0.08 * len(w)
    times.append([round(t, 3), round(t + d, 3)])
    t += d + 0.25
a = text.index("двісті")
b = text.index("гривень") - 1
row = {"id": "cv22_x", "src": "cv22", "text": text, "duration": round(t + 0.3, 3), "word_times": times,
       "num_chars": [[a, b]], "spans": [{"kind": "int", "reading": "двісті сорок п'ять"}]}
pairs = B.seam_crops([row], random.Random(1), positions=0)
assert pairs, "no seam pair inside a single clip"
reading = "двісті сорок п'ять".split()
for p in pairs:
    lw, rw = p["left"]["text"].split(), p["right"]["text"].split()
    assert lw + rw == toks, (p["pos"], lw, rw)  # nothing lost or doubled across the seam
    lp, rp = p["left"]["reading_part"].split(), p["right"]["reading_part"].split()
    assert lp + rp == reading, (p["pos"], lp, rp)
    assert p["left"]["b"] == p["right"]["a"] and p["left"]["a"] >= 0 and p["right"]["b"] <= row["duration"]
assert {B.pos_class(p["pos"], p["n_words"]) for p in pairs} >= {"before", "gap", "in_word", "after"}

# compound / inflected number words (num_keep): no cut inside a number word, the reading still partitions
pairs = B.seam_crops([row], random.Random(1), positions=0, num_keep=C.NUM_KEEP_FRAC)
assert pairs and all(B.pos_class(p["pos"], p["n_words"]) != "in_word" for p in pairs)
for p in pairs:
    assert p["left"]["text"].split() + p["right"]["text"].split() == toks

print("ok")
