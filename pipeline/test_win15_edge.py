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


if __name__ == "__main__":
    test_edges()
    test_short_readings_ignored()
    print("ok")
