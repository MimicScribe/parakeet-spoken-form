"""Unit tests for ml_fleurs_fix matching (no network): python3 test_ml_fleurs_fix.py"""
import ml_fleurs_fix as F

tsv = "\n".join("\t".join(map(str, r)) for r in [
    (10, "a1.wav", "Sentence ten 5.", "x", "c", 80000, "MALE"),       # 5.00 s
    (10, "a2.wav", "Sentence ten 5.", "x", "c", 96000, "FEMALE"),     # 6.00 s
    (10, "a3.wav", "Sentence ten 5.", "x", "c", 96800, "FEMALE"),     # 6.05 s
    (11, "b1.wav", "Eleven.", "x", "c", 64000, "MALE"),               # 4.00 s
    (11, "b2.wav", "Eleven.", "x", "c", 64000, "FEMALE"),             # 4.00 s (identical duration)
    (12, "c1.wav", "Twelve 12.", "x", "c", 48000, "MALE"),
    (12, "c2.wav", "Twelve 12.", "x", "c", 56000, "FEMALE"),
    (13, "d1.wav", "No digits.", "x", "c", 40000, "MALE"),
]) + "\nbad line\n"
recs = F.parse_tsv(tsv)
assert len(recs) == 8 and recs[0].dur == 5.0 and recs[1].gender == "FEMALE" and recs[0].stem == "a1"


def row(sid, dur, text, src="fleurs"):
    return {"audio_filepath": f"/vol/d2/de/wav/fleurs_{sid}.wav", "duration": dur, "text": text, "lang": "de",
            "src": src, "id": f"fleurs_{sid}"}


# exact + tolerance; 6.0 s (a2) vs 6.05 s (a3) are 0.05 apart -> a row at 6.0 sees both = ambiguous
m = F.match_rows([row(10, 5.0, "t"), row(10, 6.0, "t")], recs)
assert [(r["duration"], c.file_name) for r, c in m["matched"]] == [(5.0, "a1.wav")], m
assert len(m["ambiguous"]) == 1
m = F.match_rows([row(10, 6.049, "t")], recs)  # only a3 within 0.05 of 6.049? a2 is 0.049 away too -> both
assert len(m["ambiguous"]) == 1
m = F.match_rows([row(10, 6.1, "t")], recs)  # 6.1-6.05 = 0.05 -> a3 only (a2 is 0.1 away)
assert [c.file_name for _, c in m["matched"]] == ["a3.wav"], m
# no candidate / unknown id / non-fleurs id
m = F.match_rows([row(10, 9.0, "t"), row(99, 4.0, "t"), dict(row(10, 5.0, "t"), id="vox_1")], recs)
assert len(m["unmatched"]) == 3 and not m["matched"]
# two rows wanting the same single recording: neither is placed one-to-one
m = F.match_rows([row(12, 3.0, "t"), row(12, 3.01, "t")], recs)
assert len(m["ambiguous"]) == 2 and not m["matched"]
# identical-duration recordings: ambiguous
m = F.match_rows([row(11, 4.0, "t")], recs)
assert len(m["ambiguous"]) == 1

# sibling labels need unanimity (whitespace-normalised)
d = [row(10, 5.0, "Sentence ten 5."), row(10, 6.0, "Sentence  ten 5."), row(12, 3.0, "Twelve 12."), row(12, 3.5, "Zwölf 12.")]
assert F.sibling_labels(d) == {10: "Sentence ten 5."}
assert F.sibling_labels([row(1, 1, "Es gab 6.500, ok."), row(1, 2, "es gab 6.500 ok")]) == {1: "Es gab 6.500, ok."}
assert F.sibling_labels([row(1, 1, "Es gab 6.500."), row(1, 2, "Es gab 6500.")]) == {}

# plan: matched row keeps own label (origin d2), siblings added once each, non-fleurs ignored, no duplicates
digit = [row(10, 5.0, "Sentence ten 5."), row(10, 6.1, "Sentence ten 5."), row(11, 4.0, "Eleven 11."),
         row(12, 3.0, "Twelve 12."), row(12, 3.5, "Zwölf 12."), row(10, 5.0, "x", src="vox")]
digit[-1]["id"] = "vox_7"
p = F.plan(digit, [], recs, "de", "/vol")
fns = [r["file_name"] for r in p["digit"]]
assert len(fns) == len(set(fns)) == len(p["jobs"]), fns
by = {r["file_name"]: r for r in p["digit"]}
assert by["a1.wav"]["origin"] == "d2" and by["a3.wav"]["origin"] == "d2"
assert by["a2.wav"]["origin"] == "d2fix" and by["a2.wav"]["label_src"] == "sibling" and by["a2.wav"]["text"] == "Sentence ten 5."
assert by["a1.wav"]["audio_filepath"] == "/vol/d2fix/de/wav/fleurs_10_a1.wav" and by["a1.wav"]["speaker"] == "a1"
assert "b1.wav" in by and "b2.wav" in by  # id 11: one D2 label -> both recordings via sibling pass
assert by["c1.wav"]["origin"] == "d2" and by["c2.wav"]["text"] == "Zwölf 12."  # exact matches keep own label
assert "d1.wav" not in by  # sentence never a D2 digit sentence
s = p["stats"]
assert s["digit"]["matched"] == 4 and s["digit"]["ambiguous"] == 1 and s["sentences_label_disagree"] == 1, s
assert s["sibling_added"] == 3 and s["digit_rows_out"] == 7, s
print("ok")
