"""English number-class caption scan (no Modal): python3 test_en_hits_num.py"""
import en_hits_num as H

c = H.classes_of
assert "fwd_year" in c("We expect this by 2038 at the latest") and "fwd_year_common" in c("by 2030 we")
assert "fwd_year" not in c("back in 2019 we") and "fwd_year" not in c("in 2075 maybe")
assert "phone" in c("call 555-123-4567 today") and "phone" not in c("page 12")
assert "clock" in c("at 3:45 PM sharp") and "clock" in c("we start at 7pm")
assert "code" in c("the F-35 jet") and "money" in c("it cost $5.50 each") and "dec" in c("it cost $5.50 each")
assert "scaled" in c("worth $20 million") and "scaled" in c("about 3 billion people")
assert "pct" in c("up 15% in a year") and "ord" in c("the 21st century") and "big" in c("about 1,500 people")
assert "date" in c("on March 5th we") and "date" in c("the 12th of June")
assert "accent" in c("it cost £5.50 in London") and "accent" not in c("it cost $5.50 in Boston")
assert c("no numbers here") == {} and c("see http://x.com/2 now 2038") == {}  # unsupported rows never hit
assert H.score_of(c("call 555-123-4567 today")) >= 8.0 > H.score_of(c("up 15% in a year")) >= 0.5
rows = [{"utt_id": f"u{i}", "duration": 5.0, "text": t, "original_audio_id": v, "lang": "en", "original_audio_offset": 0}
        for i, (t, v) in enumerate([("at 3:45 PM", "a"), ("by 2038 and 2041", "a"), ("call 555-123-4567", "b"),
                                    ("up 15%", "c"), ("nothing", "d")])]
hs = H.scan_rows(rows, "s")
assert [h["utt_id"] for h in hs] == ["u0", "u1", "u2"], [h["utt_id"] for h in hs]  # 15% alone scores under the floor
sel = H.pick(hs, 5, per_video=1)
assert [h["utt_id"] for h in sel] == ["u2", "u0"], [h["utt_id"] for h in sel]  # best row of each video first
assert [h["utt_id"] for h in H.pick(hs, 5, skip_videos={"b"})] == ["u0", "u1"]
assert [h["utt_id"] for h in H.pick(hs, 5, video_counts={"a": 30})] == ["u2"]
print("ok")
