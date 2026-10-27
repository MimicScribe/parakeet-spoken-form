"""Unit tests of boundary_cut.py (the fixtures are the spec of the rule).
Run: python3 -I test_boundary_cut.py            (no pytest on this machine: the __main__ block runs every test_*)
     python3 -m pytest -p no:cacheprovider test_boundary_cut.py -q     (where pytest exists)"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boundary_cut as BC  # noqa: E402
from boundary_cut import Word, Energy  # noqa: E402


def stream(spec, t=0.5):
    """spec = [(text, gap_before_s, key, dur_s)] -> (words, texts, total)"""
    ws, tx = [], []
    for text, gap, key, dur in spec:
        t += gap
        ws.append(Word(round(t, 4), round(t + dur, 4), key))
        tx.append(text)
        t += dur
    return ws, tx, round(t + 0.5, 4)


NUMBER = [("five", 0.0, "n1", 0.30), ("hundred", 0.20, "n1", 0.35), ("sixty", 0.20, "n1", 0.35)]


def sentence(n_pre=6, n_post=6, num=NUMBER, gap=0.3, dur=0.3):
    spec = [(f"p{i}", gap, None, dur) for i in range(n_pre)] + [(t, g if i == 0 else g, k, d) for i, (t, g, k, d) in enumerate(num)]
    spec[n_pre] = (num[0][0], gap, num[0][2], num[0][3])
    spec += [(f"q{i}", gap, None, dur) for i in range(n_post)]
    return stream(spec)


def check_label_invariants(words, cut, pad=BC.MIN_PAD):
    """No labelled word is cut, no unlabelled word is audible (within pad of the clip)."""
    for i, w in enumerate(words):
        if cut.j0 <= i < cut.j1:
            assert w.t0 >= cut.a + pad - 1e-6 and w.t1 <= cut.b - pad + 1e-6 or (cut.start.kind == "bound" and w.t0 >= cut.a - 1e-6) or (cut.end.kind == "bound" and w.t1 <= cut.b + 1e-6), (i, w, cut)
        else:
            assert w.t1 <= cut.a - pad + 1e-6 or w.t0 >= cut.b + pad - 1e-6, ("audible fragment", i, w, cut)


# ------------------------------------------------------------------------------------------------------- positions
def test_edge_in_gap_goes_to_gap_middle_without_audio():
    ws, tx, total = sentence()
    target = ws[3].t0 - 0.1  # inside the gap before word 3
    e = BC.place_edge(ws, total, target, "start")
    assert e.ok and e.j == 3 and e.shift == 0.0
    assert abs(e.t - (ws[2].t1 + ws[3].t0) / 2) < 1e-3
    assert e.t - ws[2].t1 >= BC.MIN_PAD - 1e-6 and ws[3].t0 - e.t >= BC.MIN_PAD - 1e-6
    assert e.kind == "valley" and e.silence > BC.FORCED_SILENCE  # 0.3 s gap, middle cut


def test_keep_valid_keeps_an_already_good_edge():
    ws, tx, total = sentence()
    target = ws[3].t0 - 0.09
    e = BC.place_edge(ws, total, target, "start", keep_valid=True)
    assert e.ok and abs(e.t - target) < 1e-4


def test_quietest_frame_in_the_gap_when_energy_exists():
    ws, tx, total = sentence()
    lo, hi = ws[2].t1 + BC.MIN_PAD, ws[3].t0 - BC.MIN_PAD
    vals = [1.0] * int(total / 0.01 + 5)
    k = int((lo + 0.02) / 0.01)  # one very quiet frame inside the interval
    vals[k] = 0.0
    e = BC.place_edge(ws, total, ws[3].t0 - 0.1, "start", energy=Energy(vals, 0.01))
    assert abs(e.t - (k + 0.5) * 0.01) < 1e-6 and lo <= e.t <= hi
    # all equal: nearest the middle
    flat = BC.quietest(lo, hi, Energy([1.0] * len(vals), 0.01))
    assert abs(flat - (lo + hi) / 2) <= 0.005 + 1e-9


# ------------------------------------------------------------------------------------------------------- clipped words
def test_clipped_plain_word_leaves_audio_and_label():
    ws, tx, total = sentence()
    w = ws[3]
    c = BC.cut_window(ws, total, w.t0 + 0.05, total, min_len=1.0)  # start edge inside word 3 (not a number)
    assert w.key is None
    assert c.ok and c.j0 == 4 and c.removed_left >= 1
    assert c.a >= ws[3].t1 + BC.MIN_PAD - 1e-6  # the word is not audible any more
    check_label_invariants(ws, c)


def test_end_edge_clipped_plain_word_moves_back():
    ws, tx, total = sentence()
    k = 11  # a plain 'q' word (the number is words 6..8)
    assert ws[k].key is None
    c = BC.cut_window(ws, total, 0.0, ws[k].t0 + 0.1)  # end edge inside plain word k: the nominal window holds words 0..k
    assert c.ok and c.j1 == k and c.removed_right == 1
    assert c.b <= ws[k].t0 - BC.MIN_PAD + 1e-6
    check_label_invariants(ws, c)


def test_number_cut_mid_word_excludes_the_clip():
    ws, tx, total = sentence()
    n0 = next(i for i, w in enumerate(ws) if w.key == "n1")
    mid = (ws[n0 + 2].t0 + ws[n0 + 2].t1) / 2  # inside 'sixty'
    c = BC.cut_window(ws, total, 0.0, mid)
    assert not c.ok and c.reason == "end:number_mid_word"
    c2 = BC.cut_window(ws, total, (ws[n0].t0 + ws[n0].t1) / 2, total)  # inside 'five'
    assert not c2.ok and c2.reason == "start:number_mid_word"


def test_pad_short_next_to_number_word_also_excludes():
    ws, tx, total = sentence()
    n0 = next(i for i, w in enumerate(ws) if w.key == "n1")
    c = BC.cut_window(ws, total, 0.0, ws[n0].t0 - 0.02)  # 20 ms before 'five': too close to cut, clips the number
    assert not c.ok and c.reason == "end:number_mid_word"


# ------------------------------------------------------------------------------------------------------- number straddle
def test_number_split_at_a_gap_keeps_each_sides_words():
    ws, tx, total = sentence()
    n0 = next(i for i, w in enumerate(ws) if w.key == "n1")
    p = BC.seam_pair(ws, total, n0 + 2, win=3.0)  # cut in the gap 'hundred' | 'sixty'
    assert p is not None
    L, R = p["left"], p["right"]
    left_words = [tx[i] for i in BC.window_label(ws, L)]
    right_words = [tx[i] for i in BC.window_label(ws, R)]
    assert left_words[-2:] == ["five", "hundred"] and right_words[0] == "sixty"
    assert p["c"].straddle == "n1" and L.straddles == ["n1"] and R.straddles == ["n1"]
    assert L.j1 == R.j0 == n0 + 2  # halves partition the stream
    assert L.b == R.a  # middle cut: one shared time
    ph = BC.seam_pair(ws, total, n0 + 2, win=3.0, hug=(True, True))
    assert ph['left'].end.kind == ph['right'].start.kind == 'forced' and ph['left'].b < ph['right'].a
    assert ph['left'].j1 == ph['right'].j0 == n0 + 2  # words still partition
    check_label_invariants(ws, ph['left']); check_label_invariants(ws, ph['right'])
    check_label_invariants(ws, L)
    check_label_invariants(ws, R)


def test_window_edge_in_a_gap_inside_a_number_is_kept_and_flagged():
    ws, tx, total = sentence()
    n0 = next(i for i, w in enumerate(ws) if w.key == "n1")
    tgt = (ws[n0].t1 + ws[n0 + 1].t0) / 2  # gap 'five' | 'hundred'
    c = BC.cut_window(ws, total, tgt, total)
    assert c.ok and c.j0 == n0 + 1 and c.straddles == ["n1"]


def test_number_whole_inside_has_no_straddle():
    ws, tx, total = sentence()
    c = BC.cut_window(ws, total, 0.0, total)
    assert c.ok and c.straddles == [] and c.start.kind == "bound" and c.end.kind == "bound"


# ------------------------------------------------------------------------------------------------------- continuous speech
def test_continuous_speech_has_no_gap_so_the_clip_is_excluded():
    spec = [(f"w{i}", 0.04, None, 0.30) for i in range(30)]
    ws, tx, total = stream(spec)
    c = BC.cut_window(ws, total, 3.0, 12.0)
    assert not c.ok and c.reason == "start:no_gap_in_range"
    assert BC.gaps(ws, total)[0].j == 0 and all(g.j in (0, len(ws)) for g in BC.gaps(ws, total))


def test_gap_beyond_the_search_range_is_not_used():
    spec = [(f"w{i}", 0.04, None, 0.30) for i in range(10)] + [("x", 0.5, None, 0.3)] + [(f"v{i}", 0.04, None, 0.30) for i in range(10)]
    ws, tx, total = stream(spec)
    tgt = ws[2].t0 + 0.1  # mid-word, far from the single gap before 'x'
    far = BC.place_edge(ws, total, tgt, "start", search=1.0)
    assert not far.ok and far.reason == "no_gap_in_range"
    near = BC.place_edge(ws, total, tgt, "start", search=ws[10].t0 - tgt + 0.01)
    assert near.ok and near.j == 10


def test_min_gap_boundaries():
    for width, eligible in ((0.11, False), (0.119, False), (0.121, True), (0.30, True)):
        ws, tx, total = stream([("a", 0, None, 0.3), ("b", width, None, 0.3)])
        got = any(g.j == 1 for g in BC.gaps(ws, total))
        assert got == eligible, (width, got)


# ------------------------------------------------------------------------------------------------------- stitched joins
def stitched():
    """two parents joined by a 0.5 s synthetic pause; intra-parent gaps 0.12 (tight) and 0.2 (valley)."""
    spec = []
    for i in range(6):
        spec.append((f"a{i}", 0.12 if i % 2 else 0.2, None, 0.25))
    spec.append(("b0", 0.5, None, 0.25))
    for i in range(1, 7):
        spec.append((f"b{i}", 0.12 if i % 2 else 0.2, None, 0.25))
    ws, tx, total = stream(spec)
    ws[6] = Word(ws[6].t0, ws[6].t1, None, True)  # the join before b0
    return ws, tx, total


def test_intra_stays_inside_the_parent_and_default_takes_the_nearest_join():
    ws, tx, total = stitched()
    tgt = ws[5].t0 + 0.05  # mid-word of a5; nearest gap after it is the join (j=6)
    plain = BC.place_edge(ws, total, tgt, "start", search=3.0)
    assert plain.ok and plain.j == 6 and plain.join
    inside = BC.place_edge(ws, total, tgt, "start", search=3.0, intra=True)
    assert inside.ok and not inside.join and inside.j > 6 and not inside.intra_unmet


def test_intra_unmet_is_flagged_when_only_the_join_is_in_range():
    ws, tx, total = stitched()
    tgt = ws[5].t0 + 0.05
    e = BC.place_edge(ws, total, tgt, "start", search=ws[6].t0 - tgt + 0.01, intra=True)
    assert e.ok and e.join and e.intra_unmet


def test_hug_makes_a_forced_edge_flush_with_the_kept_word():
    ws, tx, total = sentence()  # 0.3 s gaps: the middle cut is a valley, the hugging cut is forced
    tgt = ws[3].t0 - 0.2
    mid = BC.place_edge(ws, total, tgt, "start")
    hug = BC.place_edge(ws, total, tgt, "start", hug=True)
    assert mid.kind == "valley" and hug.kind == "forced" and mid.j == hug.j == 3
    assert abs(hug.silence - BC.MIN_PAD) < 1e-3 and abs(ws[3].t0 - hug.t - BC.MIN_PAD) < 1e-3
    end = BC.place_edge(ws, total, ws[4].t1 + 0.1, "end", hug=True)
    assert end.kind == "forced" and abs(end.t - ws[4].t1 - BC.MIN_PAD) < 1e-3
    # the hugged edge still drops the clipped word from audio and label
    c = BC.cut_window(ws, total, ws[3].t0 + 0.05, total, hug=(True, False))
    assert c.ok and c.j0 == 4 and c.start.kind == "forced"
    check_label_invariants(ws, c)


def test_narrow_gap_is_forced_even_with_a_middle_cut():
    ws, tx, total = stream([("a", 0, None, 0.3), ("b", 0.13, None, 0.3), ("c", 0.13, None, 0.3), ("d", 0.5, None, 0.3)])
    e = BC.place_edge(ws, total, ws[1].t0 + 0.1, "start", search=3.0)
    assert e.ok and e.j == 2 and e.kind == "forced" and e.silence <= BC.FORCED_SILENCE + 1e-9


def test_forced_share_counts():
    ws, tx, total = stitched()
    edges = [BC.place_edge(ws, total, ws[i].t0 + 0.05, "start", search=3.0, hug=True, intra=True) for i in (1, 3, 8, 10)]
    fs = BC.forced_share(edges)
    assert fs["forced"] == 4 and fs["forced_share"] == 1.0


# ------------------------------------------------------------------------------------------------------- audit
def test_edge_status_classes():
    ws, tx, total = sentence()
    n0 = next(i for i, w in enumerate(ws) if w.key == "n1")
    assert BC.edge_status(ws, total, 0.0)["status"] == "bound"
    assert BC.edge_status(ws, total, (ws[2].t1 + ws[3].t0) / 2)["status"] == "gap"
    s = BC.edge_status(ws, total, ws[n0].t1 - 0.1)
    assert s["status"] == "in_word" and s["number"]
    assert BC.edge_status(ws, total, ws[3].t0 - 0.02)["status"] == "pad_short"


# ------------------------------------------------------------------------------------------------------- properties
def test_random_streams_never_leave_an_audible_fragment_or_a_cut_word():
    rng = random.Random(7)
    n_ok = n_ex = 0
    for trial in range(400):
        spec, key = [], 0
        for i in range(rng.randint(12, 40)):
            gap = rng.choice([0.03, 0.05, 0.08, 0.11, 0.13, 0.2, 0.4, 0.8])
            spec.append((f"w{i}", gap, f"n{key}" if rng.random() < 0.25 else None, rng.uniform(0.15, 0.5)))
            if spec[-1][2] is None or rng.random() < 0.5:
                key += 1
        ws, tx, total = stream(spec)
        a = rng.uniform(0, total * 0.4)
        b = a + rng.uniform(2.0, total * 0.6)
        c = BC.cut_window(ws, total, a, min(b, total), search=rng.choice([0.5, 1.0, 2.0]), hug=(rng.random() < 0.5, rng.random() < 0.5))
        if not c.ok:
            n_ex += 1
            continue
        n_ok += 1
        check_label_invariants(ws, c)
        assert c.a >= a - 0.5 * 1.0 - 1e-6  # a gap that contains the nominal edge may start up to its own width earlier
    assert n_ok > 50 and n_ex > 20


def test_deterministic():
    ws, tx, total = sentence()
    r1 = BC.cut_window(ws, total, 1.3, 7.7)
    r2 = BC.cut_window(ws, total, 1.3, 7.7)
    assert r1 == r2


if __name__ == "__main__":
    names = sorted(n for n in globals() if n.startswith("test_"))
    bad = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n)
        except Exception as ex:  # noqa: BLE001
            bad += 1
            import traceback
            print("FAIL", n, repr(ex))
            traceback.print_exc()
    print(f"{len(names) - bad}/{len(names)} passed")
    sys.exit(1 if bad else 0)
