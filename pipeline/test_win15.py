"""Local tests for win15 (no GPU, no Modal): python3 test_win15.py"""
import random

import win15 as W


def row(rid, words, nums, dur=None, src="vox", step=0.5, **kw):
    """A labelled row: one word every `step` s (0.25 s long), number readings at the given token index lists."""
    text = " ".join(words)
    starts, pos = [], 0
    for t in words:
        starts.append(pos)
        pos += len(t) + 1
    num_chars = [[starts[g[0]], starts[g[-1]] + len(words[g[-1]])] for g in nums]
    wt = [[round(0.2 + k * step, 3), round(0.2 + k * step + 0.25, 3)] for k in range(len(words))]
    return dict({"id": rid, "text": text, "num_chars": num_chars, "word_times": wt,
                 "duration": dur or round(0.2 + len(words) * step + 0.2, 3), "src": src,
                 "spans": [{"kind": "int", "text": "1", "reading": " ".join(words[k] for k in g)} for g in nums]}, **kw)


always_quiet = lambda t0, t1: (t0 + t1) / 2  # noqa: E731

# ---- layout: rows end to end with the given gaps; reading keys per token -----------------------------------------
a = row("20100101-0900-PLENARY-1_de_1", ["wir", "haben", "zwei", "hundert", "Leute"], [[2, 3]])
b = row("20100101-0900-PLENARY-1_de_2", ["und", "drei", "Kinder"], [[1]])
starts, words, total = W.layout([a, b], [0.4])
assert starts == [0.0, a["duration"] + 0.4] and abs(total - (a["duration"] + 0.4 + b["duration"])) < 1e-9
assert [w[3] for w in words] == [None, None, (0, 0), (0, 0), None, None, (1, 0), None]

# ---- cuts: never inside a number reading, never in a gap shorter than MIN_GAP, stream start / end included -------
cuts = W.cut_points(words, total, always_quiet)
js = [j for _, j in cuts]
assert 3 not in js, "a cut between 'zwei' and 'hundert' splits one reading"
assert js[0] == 0 and js[-1] == len(words) and 5 in js
tight = [("x", 0.0, 0.3, None, 0, 0), ("y", 0.35, 0.6, None, 0, 1)]
assert [j for _, j in W.cut_points(tight, 0.8, always_quiet)] == [0, 2]  # 50 ms between letters: no cut
assert [j for _, j in W.cut_points(words, total, lambda t0, t1: None)] == [0, len(words)]  # nothing quiet

# ---- windows: >= 2 number rows, number words in the first and last third, whole words, exact labels -------------
rng = random.Random(1)
fill = ["und", "dann", "noch", "mehr", "Text", "hier"]
rows = []
for i in range(8):  # 8 rows of ~6 s, one number near each end of every row
    ws = ["sieben"] + [fill[(i + k) % 6] for k in range(9)] + ["acht"]
    rows.append(row(f"20100101-0900-PLENARY-1_de_{i + 3}", ws, [[0], [10]], step=0.55))
gaps = W.stream_gaps(len(rows), rng)
assert all(W.GAP[0] <= g <= W.GAP[1] for g in gaps) and all(abs(g * 16000 - round(g * 16000)) < 1e-6 for g in gaps)
wins = W.plan_windows(rows, gaps, always_quiet, rng)
assert wins, "no window planned"
_, words, total = W.layout(rows, gaps)
for w in wins:
    assert W.MIN_LEN <= w["b"] - w["a"] <= W.WIN + 1e-9, (w["a"], w["b"])
    assert w["n_num_rows"] >= 2 and len(w["rows"]) >= 2
    toks = w["text"].split()
    assert len(toks) == len(w["num_tokens"]) == len(w["src"]) == len(w["word_times"])
    for tok, (i, k), (t0, t1) in zip(toks, w["src"], w["word_times"]):
        assert rows[i]["text"].split()[k] == tok  # the parent's word, verbatim
        assert 0 <= t0 < t1 <= w["b"] - w["a"] + 1e-9  # whole word inside the window
    span = w["b"] - w["a"]
    nt = [t for t, f in zip(w["word_times"], w["num_tokens"]) if f]
    assert nt[0][0] <= span / 3 and span - nt[-1][1] <= span / 3
    # no heard word left out: every stream word whose sound lies inside the window is labelled
    inside = [x for x in words if x[1] >= w["a"] and x[2] <= w["b"]]
    assert [x[0] for x in inside] == toks
assert len({(w["j0"], w["j1"]) for w in wins}) == len(wins)
assert any(w["cut_left"] for w in wins) or any(w["cut_right"] for w in wins), "edges never cut a sentence"
# one number-bearing row only: no window
lone = [row("20100101-0900-PLENARY-2_de_1", ["eins"] + fill * 4, [[0]], step=0.55),
        row("20100101-0900-PLENARY-2_de_2", fill * 4, [], step=0.55)]
assert not W.plan_windows(lone, W.stream_gaps(2, rng), always_quiet, random.Random(2))

# a row whose start does not re-decode never has its first word inside a window (its end likewise)
bad_start = [dict(r, _start_ok=(i % 2 == 0)) for i, r in enumerate(rows)]
for w2 in W.plan_windows(bad_start, gaps, always_quiet, random.Random(5)):
    for i in w2["rows"]:
        ks = [k for j, k in w2["src"] if j == i]
        assert i % 2 == 0 or 0 not in ks
assert W.end_close("Jahr haben wir", "jahr habn wir", "start") and not W.end_close("Jahr haben wir", "haben wir", "start")

# ---- streams: one session / voice each, never "misc" strangers, >= 2 rows ----------------------------------------
g = W.group([dict(r, id=f"20100101-0900-PLENARY-1_de_{i}") for i, r in enumerate(rows)]
            + [row("fleurs_1_x", ["a"], [], src="fleurs"), row("fleurs_2_y", ["b"], [], src="fleurs")], random.Random(3))
assert len(g) == 1 and all(r["id"].startswith("20100101") for r in g[0])

# ---- sources: unsplit original with the built label's letters, no ext / no times / loud edges ------------------
orig = {"r1": {"id": "r1", "text": "zweihundertsiebzig Leute", "word_times": [[0, 1], [1, 2]], "edge_quiet": True,
               "spans": [{"by": "ctc"}]},
        "r2": {"id": "r2", "text": "zehn", "word_times": [[0, 1]], "edge_quiet": True, "spans": [{"by": "ext"}]},
        "r3": {"id": "r3", "text": "elf", "word_times": [None], "edge_quiet": True, "spans": []},
        "r4": {"id": "r4", "text": "zwölf", "word_times": [[0, 1]], "edge_quiet": False, "spans": []},
        "r5": {"id": "r5", "text": "dreizehn", "word_times": [[0, 1]], "edge_quiet": True, "spans": []}}
built = [{"id": "r1", "text": "zwei hundert siebzig Leute"}, {"id": "r2", "text": "zehn"}, {"id": "r3", "text": "elf"},
         {"id": "r4", "text": "zwölf"}, {"id": "r5", "text": "vierzehn"}, {"id": "r6", "text": "x"}]
got, st = W.pick_sources(built, orig, real=True)
assert [x["id"] for x in got] == ["r1", "r4"] and got[0]["text"] == "zweihundertsiebzig Leute"
assert st == {"in": 6, "ext": 1, "no_times": 1, "edge_not_quiet": 1, "label_changed": 1, "no_original": 1, "kept": 2,
              "survival": 0.5}

import numpy as np  # noqa: E402

# ---- quiet cut never reads outside the audio (a word time before 0 / past the end) -----------------------------
z = np.zeros(16000, dtype=np.float32)  # noqa: F821
assert W.safe_quiet(z, -0.5, -0.1) is None and W.safe_quiet(z, 1.2, 1.5) is None

# red-on-base (2026-10-18 review): a compound split that changes letters (it "ventuno" -> "venti uno") is the same
# label; the old letters-only check dropped every Italian row with an elided number as label_changed
it_orig = {"i1": {"id": "i1", "text": "ventuno persone e trentotto euro", "word_times": [[0, 1]] * 5, "spans": []},
           "i2": {"id": "i2", "text": "centottanta metri", "word_times": [[0, 1]] * 2, "spans": []}}
it_built = [{"id": "i1", "text": "venti uno persone e trenta otto euro"}, {"id": "i2", "text": "centottanta metri"}]
got, st = W.pick_sources(it_built, it_orig, real=False, lang="it")
assert [x["id"] for x in got] == ["i1", "i2"] and st.get("label_changed", 0) == 0 and st["survival"] == 1.0, st
assert got[0]["text"] == "ventuno persone e trentotto euro"  # the window keeps the on-disk form; the build splits
assert W.letters_eq("venti uno", "ventuno", "it") and not W.letters_eq("venti due", "ventuno", "it")
assert not W.letters_eq("venti uno", "ventuno")  # without a language: letters only

# ---- trim: a row keeps EDGE_PAD around its words, its word times follow --------------------------------------
import numpy as np  # noqa: E402

tr, y = W.trim_row({"word_times": [[1.0, 1.4], [1.6, 2.0]], "duration": 3.0}, np.zeros(48000, dtype=np.float32))
assert abs(tr["word_times"][0][0] - 0.15) < 1e-3 and abs(tr["duration"] - (2.2 - 0.85)) < 1e-3 and len(y) == 21601

# ---- checker: a label on its audio passes; words moved across source rows fail; the swap control is caught ------
w = dict(wins[0])
w["src"] = [[rows[i]["id"], k] for i, k in w["src"]]
w["seg"] = [[rows[i]["id"], a_, b_] for i, a_, b_ in w["seg"]]
assert W.check_alignment(w, [tuple(t) for t in w["word_times"]]) == []
assert W.check_alignment(w, [None] + [tuple(t) for t in w["word_times"][1:]])[0] == "unaligned:0"
shifted = [(t0 + 1.0, t1 + 1.0) for t0, t1 in w["word_times"]]
assert any(r.startswith("moved") for r in W.check_alignment(w, shifted))
cw = W.swap_control(w)
assert cw and sorted(cw["text"].split()) == sorted(w["text"].split()) and cw["text"] != w["text"]
# a forced alignment of the swapped label puts its words in audio order: the expected times no longer match
assert W.check_alignment(cw, [tuple(t) for t in sorted(cw["word_times"])])

assert not W.boundary_fail(["moved:3:+0.5"]) and W.boundary_fail(["moved:3:+0.5", "outside_source:3"])
ps = W.pieces(w)
assert [p[0] for p in ps] == [p for p, _, _ in w["seg"]] and " ".join(p[3] for p in ps) == w["text"]
assert W.piece_ok("zwei Leute", "zwei leute") and not W.piece_ok("zwei Leute", "leute") and not W.piece_ok("", "x")
assert W.edges_close("eben OPC.", "eben opz") and W.edges_close("Favorites an Blut", "faverits an blut")
assert not W.edges_close("Die Gesamtuntersetzung der Achse", "gesamtuntersetzung der achse")  # first word missing
assert not W.edges_close("kamen nach Hause", "kamen nach hause die")  # an extra word at the end
sc = W.shift_control(w)
assert sc and sc["text"] == w["text"] and sum(a != b for a, b in zip(sc["src"], w["src"])) == 1
moved = [p for p in W.pieces(sc)]
assert moved[0][3] != W.pieces(w)[0][3] and " ".join(p[3] for p in moved).split() != []

# ---- build hook: parents must be trained with the same label; synthetic <= 30% of hours; value spread kept ------
def yrow(rid, year, word):  # a parent row holding one year reading
    return {"id": rid, "text": f"im Jahr {word}", "num_chars": [[8, 8 + len(word)]],
            "spans": [{"text": str(year), "kind": "int", "reading": word}]}


mk = lambda i, syn, ps: {"id": f"w{i}", "duration": 15.0, "synthetic": syn, "parents": ps,  # noqa: E731
                         "parent_sha": {p: W.sha(f"t{p}") for p in ps}, "greedy": "g", "src": []}
pool = [mk(i, False, ["a", "b"]) for i in range(10)] + [mk(10 + i, True, ["s1", "s2"]) for i in range(10)] \
    + [mk(30, False, ["a", "zz"]), mk(31, False, ["a", "c"]), mk(32, False, ["a", "bad"])]
pt = {p: {"id": p, "text": f"t{p}"} for p in ("a", "b", "s1", "s2", "bad")} | {"c": {"id": "c", "text": "changed"}}
out, st = W.select_for_build(pool, pt, {"bad"}, hours=100 * 15 / 3600, rng=random.Random(4), lang="de")
assert st["parent_not_trained"] == 1 and st["parent_label_changed"] == 1 and st["unsafe_parent"] == 1
assert sum(w["synthetic"] for w in out) == 10 and sum(not w["synthetic"] for w in out) == 10
assert all("greedy" not in w for w in out)
out, st = W.select_for_build(pool, pt, {"bad"}, hours=10 * 15 / 3600, rng=random.Random(4), lang="de")
assert sum(w["synthetic"] for w in out) == 3 and sum(not w["synthetic"] for w in out) == 7, st
# value spread: 40 real windows of one year 2010 among varied years -> the top year stays <= 3% past the floor;
# forward-year synthetic windows stop at 1.25 x the parents' forward share
par = {f"p{y}": yrow(f"p{y}", y, f"jahr{y}") for y in range(1950, 2031)} | {"p2010x": yrow("p2010x", 2010, "zehn")}
par |= {f"f{y}": yrow(f"f{y}", y, f"jahr{y}") for y in range(2031, 2041)}
yw = lambda i, pid, syn=False: {"id": f"y{i}", "duration": 15.0, "synthetic": syn, "parents": [pid],  # noqa: E731
                                "parent_sha": {pid: W.sha(par[pid]["text"])}, "src": [[pid, 2]]}
wins = [yw(i, f"p{1950 + i % 81}") for i in range(300)] + [yw(400 + i, "p2010x") for i in range(40)] \
    + [yw(500 + i, f"f{2031 + i % 10}", True) for i in range(120)]
out, st = W.select_for_build(wins, par, set(), hours=1000, rng=random.Random(6), lang="de")
assert st["top_year_share"] <= max(W.TOP_YEAR_SHARE, W.VALUE_FLOOR / st["year_spans"]) + 1e-9, st
assert st["fwd_share"] <= st["fwd_cap"] + W.VALUE_FLOOR / st["year_spans"], st
assert st["value_spread_skipped"] > 0 and st["fwd_cap"] == round(1.25 * 10 / 92, 4)

# ---- edge check: word split does not matter, a missing / extra edge word does --------------------------------
assert W.edges_ok("Neunzehnhundert Leute kamen. Acht", "neunzehn hundert leute kamen acht")
assert not W.edges_ok("Neunzehnhundert Leute kamen acht", "hundert leute kamen acht")
assert not W.edges_ok("Leute kamen acht", "leute kamen acht und")
print("test_win15 OK")
