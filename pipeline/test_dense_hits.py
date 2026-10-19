"""Tests for dense_hits (class finder, scoring, pick caps) and dense_win.overlap_pairs. No GPU, no Modal.
python3 test_dense_hits.py"""
import random

import dense_hits as H
import dense_win as DW
import ml_p3_core as C

# class finder: the shapes the thin classes are made of
c = H.classes_of("Las ventas subieron 3,5% hasta 45.000.000 de euros en 2027.", "es")
assert c == {"pct": 1, "big": 1, "fwd": 1}, c
assert "cents" in H.classes_of("Cuesta 12,50 € y son pocos.", "es")
assert "clock" in H.classes_of("Nos vemos a las 10:30 en casa.", "es")
assert H.classes_of("Dos mil veinte fue un año largo.", "es") == {}  # no digit
assert H.classes_of("Lo vi en 1999-2003.", "es") == {}  # an unreadable range: the labeller drops the row, so it never hits
assert H.score_of({"big": 1}, "es") > H.score_of({"pct": 1}, "es") > 0
assert H.words_est(1234567, "es") >= 6 and H.words_est(7, "es") == 1


# pick: per-video cap, held-out video, used id, forward-year cap, quota, shard limit, hours
def hit(i, v, text, hits, score, shard="s0", dur=10.0):
    return {"utt_id": f"u{i}", "original_audio_id": v, "text": text, "hits": hits, "score": score, "shard": shard, "duration": dur}


hs = [hit(i, "vidA", "x", ["big"], 9 - i * 0.1) for i in range(8)] + [hit(20, "vidB", "x", ["big"], 5), hit(21, "held", "x", ["big"], 9),
                                                                     hit(22, "vidC", "x", ["big"], 5)]
out = H.pick(hs, 100, per_video=3, skip_videos={"held"}, skip_ids={"u22"})
assert sum(h["original_audio_id"] == "vidA" for h in out) == 3 and not any(h["original_audio_id"] == "held" for h in out)
assert not any(h["utt_id"] == "u22" for h in out)
out = H.pick(hs, 100, per_video=9, video_counts={"vidA": 28})  # 30 rows per video including the parts' own
assert sum(h["original_audio_id"] == "vidA" for h in out) == 2
yrs = [hit(i, f"v{i}", "en 2040 y 2041", ["fwd"], 4) for i in range(5)]
assert len(H.pick(yrs, 100, year_cap=2)) == 2
mix = [hit(i, f"v{i}", "x", ["pct"], 1) for i in range(6)] + [hit(10 + i, f"w{i}", "x", ["big"], 8, shard="s1") for i in range(3)]
assert sum("pct" in h["hits"] for h in H.pick(mix, 100, quota={"pct": 2})) == 2
assert {h["shard"] for h in H.pick(mix, 100, max_shards=1)} == {"s1"}
assert len(H.pick(mix, 100, max_hours=10 / 3600 * 2)) == 2


# overlap pairs: a long reading that no window holds whole, the two labels together hold all of it
def stream(n_words=12, step=0.5):
    toks = [f"dos{i}mil" for i in range(n_words)]
    wt = [[round(0.3 + i * step, 3), round(0.3 + i * step + 0.35, 3)] for i in range(n_words)]
    text = " ".join(["ya"] + toks + ["fin"])
    wt = [[0.0, 0.2]] + wt + [[round(0.3 + n_words * step + 0.1, 3), round(0.3 + n_words * step + 0.4, 3)]]
    a = len("ya ")
    return [{"text": text, "word_times": wt, "duration": wt[-1][1] + 0.3, "num_chars": [[a, a + len(" ".join(toks))]], "spans": [{"kind": "int"}]}]


import ml_balance as B

rows = stream()
_, words, total = C.stream_words(rows)
rd = B.readings(words, rows)[0]
pairs = []
for seed in range(8):
    pairs += list(DW.overlap_pairs(words, total, rd, random.Random(seed), 0.9))
assert pairs, "no overlap pair from a 6 s reading"
for p in pairs:
    assert 3.0 <= p["overlap_s"] <= 6.1 and p["reading_s"] >= DW.OVL_MIN_READING_S
    lp, rp = p["left"]["reading_part"].split(), p["right"]["reading_part"].split()
    allw = p["reading"].split()
    assert 0 < len(lp) < len(allw) and 0 < len(rp) < len(allw) and set(lp) | set(rp) == set(allw)
short = stream(n_words=5)  # 2.5 s: fits whole in any overlap
_, w2, t2 = C.stream_words(short)
assert not list(DW.overlap_pairs(w2, t2, B.readings(w2, short)[0], random.Random(0), 0.9))
print("ok")
