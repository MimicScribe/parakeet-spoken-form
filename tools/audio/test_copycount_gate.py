import sys, os, json, tempfile, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import copycount_gate as G


def test_selftest():
    with contextlib.redirect_stdout(io.StringIO()):
        G._selftest()


def test_load_margins_reads_a_calibration_file_and_falls_back_to_the_floor():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.json")
    json.dump({"contrasts": {"minus": {"margin": 9.5}, "plus": {"margin": 12.25}}}, open(p, "w"))
    assert G.load_margins(p) == {"minus": 9.5, "plus": 12.25, "provisional": False}
    for bad in (None, os.path.join(d, "missing.json")):
        assert G.load_margins(bad) == {"minus": G.FLOOR, "plus": G.FLOOR, "provisional": True}
    open(p, "w").write("{not json")
    assert G.load_margins(p)["minus"] == G.FLOOR


def test_copy_durations_and_indices():
    words = ["pin", "double", "five", "nine"]
    assert G.copy_word_indices([(1, 3, 2, 2)], ["mult"], words) == [2]
    assert G.copy_word_indices([(0, 1, 1, 0), (1, 2, 1, 1)], ["whole", "whole"]) == [0, 1]
    wt = [(0.0, 0.3), (0.3, 0.38), None]
    assert G.copy_durations_ok(wt, [0]) and not G.copy_durations_ok(wt, [0, 1]) and not G.copy_durations_ok(wt, [2])


def test_onset_between_needs_a_dip_between_two_plain_copies():
    sr = 16000
    rng = np.random.default_rng(0)
    loud = lambda s: 0.5 * rng.uniform(-1, 1, int(s * sr)).astype(np.float32)
    quiet = lambda s: 0.01 * rng.uniform(-1, 1, int(s * sr)).astype(np.float32)
    split = np.concatenate([loud(0.3), quiet(0.1), loud(0.3)])             # five (pause) five
    run_on = np.concatenate([loud(0.3), loud(0.1), loud(0.3)])             # collapsed into one stretch
    wt = [(0.0, 0.3), (0.4, 0.7)]
    assert G.onset_between(split, sr, wt, 0, 1) is True
    assert G.onset_between(run_on, sr, wt, 0, 1) is False


def test_calibrate_margin_and_cli():
    m, sep = G.calibrate_margin([12.0] * 20 + [2.0])
    assert sep and m >= G.FLOOR
    m2, sep2 = G.calibrate_margin([])
    assert (m2, sep2) == (G.FLOOR, False)
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.json")
    json.dump([5, 6, 7, 8, 9, 10, 4, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10], open(p, "w"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert G.main(["calibrate", p]) == 0
    assert json.loads(buf.getvalue())["separable"] is True
    with contextlib.redirect_stdout(io.StringIO()):
        assert G.main(["bogus"]) == 2


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
