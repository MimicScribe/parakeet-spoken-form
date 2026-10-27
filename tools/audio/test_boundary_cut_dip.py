import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import boundary_cut as BC
from boundary_cut import Word
import edge_rule as E
from test_edge_rule import synth, fake_vad


def words_continuous(n, dur=0.3, t=0.2, key_at=None):
    ws = []
    for i in range(n):
        ws.append(Word(round(t, 3), round(t + dur, 3), "n1" if key_at is not None and i in key_at else None))
        t += dur + 0.01                   # 10 ms between words: no eligible gap anywhere
    return ws, round(t + 0.2, 3)


def track_with_pause(at, width, depth):
    segs = [(at, 0.5), (width, depth), (6.0 - at - width, 0.5)]
    x = synth(segs)
    return E.Track(x, fake_vad(x))


def test_no_gap_clip_is_cut_at_a_real_dip_and_the_straddled_word_leaves():
    ws, total = words_continuous(18)
    tr = track_with_pause(3.0, 0.30, 0.002)              # a real pause around 3.0-3.3 s, but the (biased) word times show no gap
    e = BC.place_edge(ws, 6.0, 2.0, "start", search=2.0, track=tr)
    assert e.ok and e.kind == "dip" and 2.9 <= e.t <= 3.4 and e.dip_db >= 12
    assert all(ws[i].t0 >= e.t - 1e-6 for i in range(e.j, len(ws)))      # kept words start after the cut
    assert e.clipped == tuple(i for i, w in enumerate(ws) if w.t0 < e.t < w.t1)


def test_no_gap_clip_without_a_real_dip_is_excluded():
    ws, total = words_continuous(18)
    x = synth([(3.0, 0.5), (0.16, 0.2), (2.84, 0.5)])      # 8 dB shallow dip only
    tr = E.Track(x, fake_vad(x))
    e = BC.place_edge(ws, 6.0, 2.0, "start", search=2.0, track=tr)
    assert not e.ok and e.reason == "no_dip" and 0 < e.dip_db < 12


def test_dip_inside_a_number_word_excludes_the_clip():
    ws, total = words_continuous(18, key_at=set(range(8, 12)))
    tr = track_with_pause(3.0, 0.30, 0.002)
    mid = [i for i, w in enumerate(ws) if w.t0 < 3.15 < w.t1]
    assert mid and ws[mid[0]].key is not None
    e = BC.place_edge(ws, 6.0, 2.0, "start", search=2.0, track=tr)
    assert not e.ok and e.reason == "number_mid_word"


def test_gap_clip_uses_the_valley_position_inside_the_gap():
    ws = [Word(0.3, 0.9), Word(0.95, 1.5), Word(2.3, 2.9), Word(3.0, 3.6)]           # a 0.8 s gap 1.5-2.3
    x = synth([(1.7, 0.5), (0.3, 0.003), (0.1, 0.02), (0.4, 0.001), (2.5, 0.5)])      # quietest 80 ms frame at about 2.0-2.1 s
    tr = E.Track(x, fake_vad(x))
    e = BC.place_edge(ws, 4.5, 1.6, "start", search=2.0, track=tr)
    assert e.ok and e.kind in ("valley", "forced") and e.j == 2 and 1.56 <= e.t <= 2.24 and e.t > 1.95


def test_without_a_track_nothing_changes_and_no_gap_still_excludes():
    ws, total = words_continuous(18)
    e = BC.place_edge(ws, 6.0, 2.0, "start", search=2.0)
    assert not e.ok and e.reason == "no_gap_in_range"


def test_cut_window_end_to_end_with_dip_edges():
    ws, total = words_continuous(30)
    x = synth([(2.0, 0.5), (0.3, 0.002), (3.0, 0.5), (0.3, 0.002), (3.5, 0.5)])
    tr = E.Track(x, fake_vad(x))
    c = BC.cut_window(ws, round(len(x) / 16000, 3), 1.6, 5.4, search=1.0, min_len=1.0, min_words=2, track=tr, at_bound=False)
    assert c.ok and c.start.kind == "dip" and c.end.kind == "dip" and c.b - c.a > 2.5 and c.j1 - c.j0 >= 5


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
