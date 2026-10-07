"""Earnings-22 test set (CC BY-SA 4.0) from the Open ASR Leaderboard mirror.

Writes /vol/e22/wav/*.wav, /vol/e22/all.jsonl, and /vol/e22/numbers.jsonl (utterances whose
reference contains a digit, a currency/percent sign, or an English number word).
"""

import io
import json
import os

import librosa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from common import NUMBER_WORDS, WRITTEN_FORM

REPO = "hf-audio/open-asr-leaderboard"


def prepare(vol: str) -> None:
    out = f"{vol}/e22"
    os.makedirs(f"{out}/wav", exist_ok=True)
    rows = []
    for i in range(5):
        path = hf_hub_download(REPO, f"earnings22/test-0000{i}-of-00005.parquet", repo_type="dataset")
        t = pq.read_table(path)
        if i == 0:
            print("columns", t.schema)
        for r in t.to_pylist():
            uid = str(r["id"]).replace("/", "_")
            wav = f"{out}/wav/{uid}.wav"
            if not os.path.exists(wav):
                audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sf.write(wav, audio, 16000, subtype="PCM_16")
                dur = len(audio) / 16000
            else:
                dur = sf.info(wav).duration
            rows.append({"audio_filepath": wav, "duration": dur, "text": r["text"], "id": uid})
    with open(f"{out}/all.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    num = [r for r in rows if WRITTEN_FORM.search(r["text"]) or NUMBER_WORDS.search(r["text"])]
    with open(f"{out}/numbers.jsonl", "w") as f:
        for r in num:
            f.write(json.dumps(r) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"e22: {len(rows)} utterances, {hours:.2f} h; number-bearing {len(num)}")
    for r in num[:15]:
        print("  ", r["text"][:120])
