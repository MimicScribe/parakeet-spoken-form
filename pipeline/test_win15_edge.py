"""win15 edge windows (plan_edge_windows): a long reading at the window's right edge ("tail") or left edge ("head"),
never cut, labels exact. Pure Python. python3 test_win15_edge.py"""
import random

import win15 as W


def _row(i, words, readings, t0=0.2, step=0.45):
    """A row whose words are `words` (strings), readings = [(first word, last word)] number readings."""
    text = " ".join(words)
    offs, pos = [], 0
    for w in words:
        a = text.index(w, pos)
        offs.append((a, a + len(w)))
        pos = a + len(w)
    nc = [[offs[a][0], offs[b][1]] for a, b in readings]
    wt = [[round(t0 + k * step, 3), round(t0 + k * step + 0.35, 3)] for k in range(len(words))]
    return {"id": f"r{i}", "text": text, "num_chars": nc, "word_times": wt, "duration": wt[-1][1] + 0.2,
            "spans": [{"kind": "phone"} for _ in nc], "src": "tts", "voice": "v"}


def _stream():
    rows = []
    for i in range(6):
        ws = [f"w{i}x{k}" for k in range(9)] + ["n1", "n2", "n3", "n4"] + [f"t{i}"]
        rows.append(_row(i, ws, [(9, 12)]))
    return rows


def test_edges():
    rows = _stream()
    rng = random.Random(3)
    gaps = W.stream_gaps(len(rows), rng)
    quiet = lambda t0, t1: (t0 + t1) / 2  # noqa: E731
    ws = W.plan_edge_windows(rows, gaps, quiet, rng)
    assert ws, "no edge window"
    starts, words, total = W.layout(rows, gaps)
    modes = set()
    for w in ws:
        modes.add(w["edge"])
        assert W.MIN_LEN - 1e-6 <= w["b"] - w["a"] <= W.WIN + 1e-6
        inside = words[w["j0"]:w["j1"]]
        keys = {x[3] for x in inside if x[3] is not None}
        for k in keys:  # every reading in the window is whole (no cut inside a reading)
            assert all(w["j0"] <= n < w["j1"] for n, x in enumerate(words) if x[3] == k)
        nums = [x for x in inside if x[3] is not None]
        assert w["edge_pos_s"][0] >= 0 and w["edge_pos_s"][1] >= 0
        if w["edge"] == "middle":
            pass
        elif w["edge"] == "tail":  # the last long reading ends just before the right edge
            assert 0 <= w["b"] - nums[-1][2] <= W.EDGE_TAIL[1] + 0.6 + 1e-6, w
            assert w["edge_gap_s"] == round(w["b"] - nums[-1][2], 3)
        else:
            assert 0 <= nums[0][1] - w["a"] <= W.EDGE_HEAD[1] + 0.6 + 1e-6, w
        assert w["edge_words"] >= W.LONG_MIN_WORDS
        assert w["text"] == " ".join(x[0] for x in inside)
    assert modes == {"tail", "head", "middle"}, modes


def test_short_readings_ignored():
    rows = [_row(i, [f"a{i}{k}" for k in range(20)] + ["n1"], [(20, 20)]) for i in range(4)]
    rng = random.Random(1)
    assert W.plan_edge_windows(rows, W.stream_gaps(4, rng), lambda a, b: (a + b) / 2, rng) == []


def _edge_stream():
    """Six rows; each holds an edge phrase "n1 n2 n3 n4 basis points" whose number reading (num_chars) is only n1..n4, and a
    two-word edge phrase "a billion two" that is its own reading's superset (edge_spans only)."""
    rows = []
    for i in range(6):
        ws = [f"w{i}x{k}" for k in range(9)] + ["n1", "n2", "n3", "n4", "basis", "points"] + [f"t{i}"]
        r = _row(i, ws, [(9, 12)])
        text = r["text"]
        a = text.index("n1")
        r["edge_spans"] = [[a, text.index("points") + len("points")]]
        rows.append(r)
    return rows


def _cuts_inside_edge(rows, seeds, guard):
    """Windows over many seeds; the number of windows that hold only part of an edge phrase."""
    bad = total = 0
    old = W.guard_ids
    W.guard_ids = guard
    try:
        for seed in range(seeds):
            rng = random.Random(seed)
            gaps = W.stream_gaps(len(rows), rng)
            starts, words, _ = W.layout(rows, gaps)
            for w in W.plan_edge_windows(rows, gaps, lambda a, b: (a + b) / 2, rng):
                total += 1
                for i in range(len(rows)):
                    toks = [n for n, x in enumerate(words) if x[4] == i and x[0] in ("n1", "n2", "n3", "n4", "basis", "points")]
                    inside = [n for n in toks if w["j0"] <= n < w["j1"]]
                    bad += 0 < len(inside) < len(toks)
    finally:
        W.guard_ids = old
    return bad, total


def test_edge_phrase_never_cut():
    rows = _edge_stream()
    # red on the base: with only the number reading protected ("basis points" may be cut off) the cut happens
    bad_base, total = _cuts_inside_edge(rows, 60, W.reading_ids)
    assert total > 20 and bad_base > 0, (bad_base, total)
    bad, total = _cuts_inside_edge(rows, 60, W.guard_ids)
    assert total > 20 and bad == 0, (bad, total)
    # a row without edge_spans is unchanged
    r = _row(0, ["a", "b", "n1", "n2", "n3", "n4"], [(2, 5)])
    assert W.guard_ids(r) == W.reading_ids(r)
    # tokens of one edge phrase share one key; a reading outside every edge span keeps its index
    r = dict(_edge_stream()[0], num_chars=[[0, 2]] + _edge_stream()[0]["num_chars"])
    ids = W.guard_ids(r)
    ph = [ids[n] for n, t in enumerate(r["text"].split()) if t in ("n1", "n2", "n3", "n4", "basis", "points")]
    assert len(set(ph)) == 1 and ph[0][0] == "edge"




if __name__ == "__main__":
    test_edges()
    test_edge_phrase_never_cut()
    test_short_readings_ignored()
    print("ok")
