"""English letter+digit codes keep the number word, never its homophone (2026-10-19): "B two B", not "B to B".

The labeller's candidate readings, the synthetic-sentence templates and the English relabel pass must all agree.
python3 test_alnum_homophone.py   (no GPU; red on the base: labeller offered "B to B", readings wrote "B to B")"""
import random
import re

import en_relabel as R
import labeller as L
import readings

# labeller: a code's candidates never hold the homophone
for cap, want in (("Our B2B and D2C sales.", ["B two B", "D two C"]), ("The B2C arm.", ["B two C"]), ("A Tele2 offer.", ["Tele two"]),
                  ("Up in Q4.", ["Q four"]), ("On 5G and 4K.", ["five G", "four K"])):
    got = [s for s in L.spans(cap) if s.kind in ("code", "unit")]
    assert [s.cands[0] for s in got] == want, (cap, [s.cands for s in got])
    for s in got:
        assert not any(re.search(r"\b(?:to|too|for|ate|won)\b", c) for c in s.cands if c.split()[0] in ("B", "D", "Tele")), s.cands

# synthetic templates (readings.py): the spoken-form sentence for a code never writes the homophone between letters
HOM = re.compile(r"\b[A-Z] (?:to|too) [A-Z]\b")
for kind in (readings.k_identifier, readings.k_biz_code):
    seen = {kind(random.Random(i)) for i in range(400)}
    assert not any(HOM.search(t) for t in seen), [t for t in seen if HOM.search(t)]
assert {"B two B", "B two C", "D two C"} <= {readings.k_identifier(random.Random(i)) for i in range(2000)} | \
    {readings.k_biz_code(random.Random(i)) for i in range(2000)}

# en_relabel R8: the closed list of codes only
for t, want in (("We spent the morning on B to B.", "We spent the morning on B two B."),
                ("Are you still running D to C?", "Are you still running D two C?"),
                ("B to B, B to C and P to P", "B two B, B two C and P two P"),
                ("it is b to c demand", "it is b two c demand"),
                ("run our B to B business", "run our B two B business")):
    got, rules = R.fix_text(t)
    assert got == want and rules == ["R8"], (t, got, rules)
for t in ("from A to B, in our example", "a homeomorphism from X to Y", "E to P in Library of Congress", "I to C bus", "c sub n to c",
          "B to Beta", "Okay B to B's", "back to B two B", "B two B", "list A to H what"):
    assert R.fix_text(t) == (t, []), (t, R.fix_text(t))
# a timed row keeps its token count (to -> two is one-for-one), so it is relabelled, not dropped
rows = [{"audio_filepath": "a.wav", "text": "on B to B support", "word_times": [[0, 1]] * 5}]
out, drops, c = R.relabel_rows(rows, "tts/biz/train_ok_np.jsonl", False)
assert out[0]["text"] == "on B two B support" and out[0]["text_old"] == "on B to B support" and not drops and c == {"R8": 1}
print("ok")
