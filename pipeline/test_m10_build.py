"""Main-run mix build plumbing (2026-10-18; no GPU, no volume). Needs modal + numpy:
    uv run --no-project --with modal==1.6.1 --with numpy python test_m10_build.py
English windows from several win15 dirs, the English drop list of the balance copies, the number shares of the
English parts the main run lists with extra_parts."""
import json
import os
import re
import tempfile

import mix_build as MB
import modal_app as MA

# ---- English windows read EVERY win15 dir of "a,b" (a comma list used to name one directory that does not exist)
v = tempfile.mkdtemp()
for d, n in (("ml/win15b", 2), ("ml/win15sym2", 3)):
    os.makedirs(f"{v}/{d}/en"); os.makedirs(f"{v}/mix4x", exist_ok=True)
    with open(f"{v}/{d}/en/windows.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"{os.path.basename(d)}{i}", "duration": 15.0, "text": "x", "stream": "yt:none", "parents": []}) + "\n")
os.makedirs(f"{v}/d1/en4_ok")
open(f"{v}/d1/en4_ok/heldout_videos_enh.txt", "w").write("# none\n")
open(f"{v}/d1/en4_ok/heldout_days_vox_en.txt", "w").write("# none\n")
MA.VOL = v
st = MA.en_win15_part("mix4x", "ml/win15b,ml/win15sym2", write=True)
assert st["written"] and st["windows"] == 5, st
assert len(open(f"{v}/mix4x/en_win15.jsonl").read().splitlines()) == 5
st = MA.en_win15_part("mix4x", "ml/win15b", write=True)  # one dir: as before
assert st["windows"] == 2, st
print("test_m10_build: en_win15_part multi-dir ok")

# ---- en_balance drops the listed rows before any cap, and says how many it found
v2 = tempfile.mkdtemp()
os.makedirs(f"{v2}/mix3b"); os.makedirs(f"{v2}/out")
part = "mix3b/d1_ycap.jsonl"
rows = [{"audio_filepath": f"/vol/a{i}.wav", "duration": 10.0, "text": "seventy five", "video": f"v{i}",
         "caption": f"call {i} now"} for i in range(6)]
open(f"{v2}/{part}", "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
mix = [(part, 0, "span")]
st = MB.en_balance(v2, mix, non_en=("mix4/p3_",), write=True, out_dir=f"{v2}/out", drop_audio=frozenset({"/vol/a1.wav", "/vol/a4.wav"}))
kept = [json.loads(l)["audio_filepath"] for l in open(f"{v2}/out/{MB.en_bcap_path(part)}")]
assert "/vol/a1.wav" not in kept and "/vol/a4.wav" not in kept and len(kept) == 4, kept
assert st["dropped_listed"] == 2 and st["parts"][part]["rows"][0] == 4, st
st0 = MB.en_balance(v2, mix, non_en=("mix4/p3_",))
assert st0["dropped_listed"] == 0 and st0["parts"][part]["rows"][0] == 6  # default: nothing dropped
print("test_m10_build: en_balance drop list ok")

# ---- the English parts of the main run have a number share (an unlisted path counts as number-free)
for p in ("d1/en4_ok/ttsSYM1_en_train_v2.jsonl", "d1/en4_ok/ttsSYM1en2_en_train.jsonl", "d1/en4_ok/yodas_en5_train_ycap.jsonl"):
    assert 0.9 < MB.number_share_of(p) <= 1.0, p
print("test_m10_build: number shares ok")

# ---- extra parts of other languages (rehearsal anchors) stay out of the English number share
mix = [("replay/lspc_train.jsonl", 10.0, "anchor"), ("ml/anchor/ancRH2/ru/plain.jsonl", 0, "nonen")]
assert any(m[0].startswith(MA._NON_EN_PARTS) for m in mix[1:]) and not mix[0][0].startswith(MA._NON_EN_PARTS)
print("test_m10_build: anchors outside the English share ok")

# ---- build4 leaves out a pass named in exclude_passes (an unjudged pass in ml/p3v2 stops the build otherwise)
import inspect
assert "exclude_passes" in inspect.signature(MB.build4).parameters
src = inspect.getsource(MB.build4)
assert "not in exclude_passes" in src
assert 'exclude_passes: str = ""' in open(MA.__file__).read()
print("test_m10_build: exclude_passes wired ok")

# ---- seam_extra: a seam file outside ml/p3v2 is read with the main seam files (pairs namespaced by file)
v3 = tempfile.mkdtemp()
os.makedirs(f"{v3}/ml/p3v2/xx"); os.makedirs(f"{v3}/ml/realx1wb/xx")
mk = lambda pid, t: {"pair": pid, "side": "a", "text": t, "parents": ["p1"], "num_tokens": [0]}  # noqa: E731
open(f"{v3}/ml/p3v2/xx/seam.jsonl", "w").write(json.dumps(mk("s0", "alpha")) + "\n")
open(f"{v3}/ml/realx1wb/xx/seam.jsonl", "w").write(json.dumps(mk("s0", "beta")) + "\n")
base = f"{v3}/ml/p3v2/xx"
got = MB._seam_rows(base, MB.SEAM_FILES + ("../../realx1wb/xx/seam",))
assert sorted(x["text"] for x in got) == ["alpha", "beta"] and len({x["pair"] for x in got}) == 2, got
assert "seam_extra" in inspect.signature(MB.build4).parameters and 'seam_extra: str = ""' in open(MA.__file__).read()
print("test_m10_build: seam_extra ok")

# ---- seam_long_extra, and the tier-1 sets in the synthetic-set tables
assert "seam_long_extra" in inspect.signature(MB.build4).parameters and 'seam_long_extra: str = ""' in open(MA.__file__).read()
tags = [t_ for t_, _ in MB.TTS_SETS]
for t_ in ("ttsDW1", "ttsPC1", "tvar1v"):
    assert t_ in tags, t_
assert {"ttsDW1", "ttsPC1"} <= MB.TTS_TARGETED and {"ttsDW1", "ttsPC1"} <= MB.TTS_LONG_SETS
assert 0.9 < MB.number_share_of("d1/en4_ok/ttsDW1_en_train.jsonl") <= 1.0
print("test_m10_build: seam_long_extra, tier-1 sets ok")

# ---- V_b control half: a chosen value-spread row with a V_b value is a violation; V_a values and other sets are not
rows_ = [{"id": "a", "_set": "ttsVS1", "spans": [{"kind": "int", "text": "14.500"}]},      # int_big:14500 (V_a)
         {"id": "b", "_set": "ttsVS1", "spans": [{"kind": "int", "text": "19.500"}]},      # int_big:19500 (V_b)
         {"id": "c", "_set": "ttsRF1", "spans": [{"kind": "int", "text": "105"}]},         # int_small:105 (V_b)
         {"id": "d", "_set": "ttsSYM1", "spans": [{"kind": "int", "text": "105"}]},        # another set: not this check
         {"id": "e", "_set": "ttsVS1", "spans": [{"kind": "int", "text": "2025"}]}]        # year:2025 (V_b)
got = MB.vb_violations(rows_, {"int_big:19500", "int_small:105", "year:2025"})
assert [g[0] for g in got] == ["b", "c", "e"], got
assert [t_ for t_, _ in MB.TTS_SETS if t_ in ("ttsVS1", "ttsRF1")] == ["ttsVS1", "ttsRF1"] and {"ttsVS1", "ttsRF1"} <= MB.TTS_TARGETED
assert not ({"ttsVS1", "ttsRF1"} & MB.TTS_LONG_SETS) and 'vsplit_dir: str = ""' in open(MA.__file__).read()
print("test_m10_build: V_b control ok")

# ---- English relabels (owner-approved fixes): applied by en_balance to its capped copies, other parts get en_rl copies the listing uses
v4 = tempfile.mkdtemp()
os.makedirs(f"{v4}/mix3b"); os.makedirs(f"{v4}/d1/en4_ok"); os.makedirs(f"{v4}/mix4x")
mk_ = lambda i, t: {"audio_filepath": f"/vol/r{i}.wav", "duration": 10.0, "text": t, "video": f"v{i}"}  # noqa: E731
open(f"{v4}/mix3b/d1_ycap.jsonl", "w").write("\n".join(json.dumps(mk_(i, "Mr. Smith paid seventy five per cent")) for i in range(4)) + "\n")
open(f"{v4}/d1/en4_ok/win_en.jsonl", "w").write("\n".join(json.dumps(mk_(10 + i, "Dr. Jones")) for i in range(3)) + "\n")
rl = f"{v4}/rl_rows.jsonl"; rd = f"{v4}/rl_drops.jsonl"
open(rl, "w").write(json.dumps({"part": "mix3b/d1_ycap.jsonl", "audio_filepath": "/vol/r1.wav", "text": "Mister Smith paid seventy five percent"}) + "\n"
                    + json.dumps({"part": "d1/en4_ok/win_en.jsonl", "audio_filepath": "/vol/r11.wav", "text": "Doctor Jones"}) + "\n")
open(rd, "w").write(json.dumps({"part": "mix3b/d1_ycap.jsonl", "audio_filepath": "/vol/r2.wav"}) + "\n")
rel = MB.load_en_relabel(rl, rd)
assert rel[("mix3b/d1_ycap.jsonl", "/vol/r2.wav")] is None and len(rel) == 3
st = MB.en_balance(v4, [("mix3b/d1_ycap.jsonl", 0, "span")], non_en=("mix4/p3_",), write=True, out_dir=f"{v4}/out", bcap_dir="mix4x", relabel=rel)
got = [json.loads(l) for l in open(f"{v4}/out/{MB.en_bcap_path('mix3b/d1_ycap.jsonl', 'mix4x')}")]
assert st["relabeled"] == 1 and st["relabel_dropped"] == 1 and len(got) == 3, st
assert [g["text"] for g in got if g["audio_filepath"] == "/vol/r1.wav"] == ["Mister Smith paid seventy five percent"]
wst = MB.write_en_relabel_parts(v4, "mix4x", rel, skip=MB.EN_REAL_PARTS)
assert wst == {"d1/en4_ok/win_en.jsonl": [3, 1, 0]}, wst
cp = [json.loads(l)["text"] for l in open(f"{v4}/{MB.en_rl_path('d1/en4_ok/win_en.jsonl', 'mix4x')}")]
assert cp == ["Dr. Jones", "Doctor Jones", "Dr. Jones"], cp
MA.VOL = v4
lst = MA.retrain3_mix_v4(("de",), win_repeat=2, mix_dir="mix4x")
assert any(p == MB.en_rl_path("d1/en4_ok/win_en.jsonl", "mix4x") for p, _, _ in lst) and not any(p == "d1/en4_ok/win_en.jsonl" for p, _, _ in lst)
assert MB.number_share_of(MB.en_rl_path("d1/en4_ok/win_en.jsonl", "mix4x")) == MB.NUMBER_PART["d1/en4_ok/win_en.jsonl"]
print("test_m10_build: English relabels ok")

# ---- a sentence period left on the end of a number span is trimmed at the build (the build used to stop on it)
x_ = {"text": "ab zwanzig. Dann", "num_chars": [[3, 11]]}
assert MB.trim_num_punct(x_) == 1 and x_["text"][slice(*x_["num_chars"][0])] == "zwanzig", x_
y_ = {"text": "drei, vier", "num_chars": [[0, 5], [6, 10]]}
assert MB.trim_num_punct(y_) == 1 and y_["num_chars"] == [[0, 4], [6, 10]], y_
z_ = {"text": "ein.hundert", "num_chars": [[0, 11]]}
assert MB.trim_num_punct(z_) == 0 and "." in z_["text"][0:11]  # inside: left for the assert
print("test_m10_build: trailing punctuation of number spans ok")
