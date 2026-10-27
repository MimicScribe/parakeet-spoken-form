import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import edge_rule as E


def synth(segments, sr=16000):
    """segments: [(seconds, amplitude)] -> float32 audio (noise-like so |x| mean is the amplitude)"""
    rng = np.random.default_rng(0)
    return np.concatenate([a * rng.uniform(-1, 1, int(s * sr)).astype(np.float32) for s, a in segments])


def fake_vad(x, quiet_below=0.02):
    """chunk probability 1 when the chunk is loud, 0.02 when quiet (stands in for Silero in unit tests)"""
    n = (len(x) + E.VAD_CHUNK - 1) // E.VAD_CHUNK
    return np.array([0.02 if np.abs(x[i * E.VAD_CHUNK:(i + 1) * E.VAD_CHUNK]).mean() < quiet_below else 0.95 for i in range(n)], dtype=np.float32)


def test_snap_picks_lowest_energy_in_widest_valley():
    x = synth([(2, 0.5), (0.5, 0.004), (1.0, 0.5), (1.0, 0.0005), (2, 0.5)])
    tr = E.Track(x, fake_vad(x))
    p = E.snap(tr, 0.0, 6.0)
    assert p["used_valley"] and 3.5 <= p["t"] <= 4.5          # the longer / quieter valley wins


def test_fallback_to_lowest_energy_without_vad():
    x = synth([(2, 0.5), (0.3, 0.01), (2, 0.5)])
    tr = E.Track(x, None)
    p = E.snap(tr, 0.5, 4.0)
    assert not p["used_valley"] and 1.9 <= p["t"] <= 2.4


def test_dip_rule_excludes_continuous_speech():
    x = synth([(3, 0.5), (0.16, 0.2), (3, 0.5)])               # a 0.16 s shallow dip (8 dB) inside continuous speech
    tr = E.Track(x, fake_vad(x))
    p, d, ok = E.snap_with_dip(tr, 2.0, 4.0, min_dip=12.0)
    assert d is not None and d < 12 and not ok
    y = synth([(3, 0.5), (0.3, 0.002), (3, 0.5)])               # a real pause
    tr2 = E.Track(y, fake_vad(y))
    p2, d2, ok2 = E.snap_with_dip(tr2, 2.0, 4.0, min_dip=12.0)
    assert ok2 and d2 > 30 and 2.9 <= p2["t"] <= 3.4


def test_relaxed_threshold_when_no_probability_below_015():
    x = synth([(1, 0.5), (0.5, 0.02), (1, 0.5)])
    v = np.array([0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    v = np.concatenate([v, np.ones(max(0, (len(x) + E.VAD_CHUNK - 1) // E.VAD_CHUNK - len(v)), dtype=np.float32)])
    tr = E.Track(x, v)
    p = E.snap(tr, 0.0, 2.4)
    assert p["used_valley"]                                      # min 0.2 in (0.15, 0.35): threshold = min(0.40, 0.28) -> valley found


def test_frame_and_chunk_sizes_are_parameters():
    x = synth([(2, 0.5), (0.5, 0.004), (1.0, 0.5), (1.0, 0.0005), (2, 0.5)])
    frame, chunk = 640, 2048                                    # a 40 ms frame, 128 ms chunks
    vad = np.array([0.02 if np.abs(x[i * chunk:(i + 1) * chunk]).mean() < 0.02 else 0.95 for i in range((len(x) + chunk - 1) // chunk)], dtype=np.float32)
    tr = E.Track(x, vad, frame=frame, vad_chunk=chunk)
    assert len(tr.energy) == len(x) // frame
    p = E.snap(tr, 0.0, 6.0)
    assert p["used_valley"] and 3.5 <= p["t"] <= 4.5 and abs(p["t_center"] - p["t"] - 0.5 * frame / E.SR) < 1e-9


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
