"""edge_fix: clipped / missing edge words are trimmed or masked (label + audio cut), readings stay whole, rows stay."""
import edge_fix as E

TEXT = "und dann kamen zwei hundert Leute in den Saal heute"
TIMES = [[0.1, 0.2], [0.3, 0.5], [0.6, 0.9], [1.2, 1.4], [1.5, 1.9], [2.0, 2.3], [2.4, 2.5], [2.6, 2.7], [2.8, 3.1],
         [3.5, 3.9]]
ROW = {"id": "r1", "text": TEXT, "num_chars": [[15, 27]], "word_times": TIMES, "duration": 4.2,
       "spans": [{"text": "200", "reading": "zwei hundert"}]}
assert TEXT[15:27] == "zwei hundert"

# both edges heard: nothing changes
rec = E.plan_row(ROW, "und dann kamen zwei", "in den saal heute", 4.2)
assert rec["status"] == "ok" and rec["crop"] is None

# first word missing from the audio: the label drops it, the audio is unchanged
rec = E.plan_row(ROW, "dann kamen zwei hun", "in den saal heute", 4.2)
assert rec["sides"]["start"] == {"act": "trim", "k": 1} and rec["crop"] is None and rec["status"] == "fixed"
x = E.cut_label(ROW, rec["k_start"], rec["k_end"], rec["crop"])
assert x["text"] == "dann kamen zwei hundert Leute in den Saal heute"
assert x["text"][x["num_chars"][0][0]:x["num_chars"][0][1]] == "zwei hundert" and x["spans"] == ROW["spans"]
assert len(x["word_times"]) == len(x["text"].split()) and x["word_times"][0] == [0.3, 0.5]

# last word clipped (a fragment the reader cannot place): masked = out of the label AND cut out of the audio
rec = E.plan_row(ROW, "und dann kamen zwei", "in den saal heu", 4.2)
assert rec["sides"]["end"]["act"] == "mask" and rec["crop"][0] == 0.0 and 3.1 < rec["crop"][1] < 3.5
x = E.cut_label(ROW, rec["k_start"], rec["k_end"], rec["crop"])
assert x["text"].endswith("in den Saal") and x["duration"] == rec["crop"][1]

# an edge word inside a reading takes the whole reading with it (never half a number)
R2 = dict(ROW, text="zwei hundert Leute in den Saal heute dort", num_chars=[[0, 12]],
          word_times=[[0.1, 0.3], [0.4, 0.8], [1.0, 1.3], [1.4, 1.5], [1.6, 1.7], [1.8, 2.1], [2.2, 2.5], [2.7, 3.0]])
rec = E.plan_row(R2, "ndert leute in den", "saal heute dort", 3.4)
assert rec["sides"]["start"] == {"act": "mask", "k": 2}, rec
x = E.cut_label(R2, rec["k_start"], rec["k_end"], [0.9, 3.4])
assert x["text"] == "Leute in den Saal heute dort" and x["num_chars"] == [] and x["spans"] == []
assert x["word_times"][0] == [0.1, 0.4]  # shifted by the crop start
assert E.widen(1, "start", E.reading_of(R2["text"], R2["num_chars"])) == 2

# too short after the cut: kept as is (never dropped)
R3 = dict(ROW, text="drei Leute heute", num_chars=[[0, 4]], word_times=[[0.1, 0.3], [0.4, 0.8], [0.9, 1.2]],
          spans=[{"text": "3"}])
assert E.plan_row(R3, "eute heute", "leute heute", 1.4)["status"] == "kept_unfixable"

# apply: sha-gated, counts, the cropped row points at its new audio, nothing dropped
fx = {"r1": dict(E.plan_row(ROW, "und dann kamen zwei", "in den saal heu", 4.2), audio="/vol/x/r1.wav",
                 verified={"end": True}),
      "r9": dict(E.plan_row(dict(ROW, id="r9"), "dann kamen zwei hun", "in den saal heute", 4.2), sha="000000000000")}
rows, st = E.apply_fixes([ROW, dict(ROW, id="r9"), dict(ROW, id="r7")], fx)
assert len(rows) == 3 and st["fixed"] == 1 and st["mask"] == 1 and st["mask_verified"] == 1
assert st["label_changed"] == 1 and st["unchecked"] == 1 and rows[0]["audio_filepath"] == "/vol/x/r1.wav"
assert rows[1] == dict(ROW, id="r9") and rows[0]["edge_fix"]["from_sha"] == E.sha(TEXT)
# a dropped edge token without letter frames: the cut sits up to LOOK s before the kept token; a kept token without
# a time cannot be cut (kept as is)
T4 = [None] + TIMES[1:]
rec = E.plan_row(dict(ROW, word_times=T4), "en dan kamen zwei", "in den saal heute", 4.2)
assert rec["status"] == "fixed" and 0.0 <= rec["crop"][0] < 0.3, rec
rec = E.plan_row(dict(ROW, word_times=[TIMES[0], None] + TIMES[2:]), "en dan kamen zwei", "in den saal heute", 4.2)
assert rec["status"] == "kept_unfixable"
print("test_edge_fix: ok")
