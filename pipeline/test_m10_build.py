"""Main-run mix build plumbing (2026-10-18; no GPU, no volume). Needs modal + numpy:
    uv run --no-project --with modal==1.6.1 --with numpy python test_m10_build.py
English windows from several win15 dirs, the English drop list of the balance copies, the number shares of the
English parts the main run lists with extra_parts."""
import json
import os
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
