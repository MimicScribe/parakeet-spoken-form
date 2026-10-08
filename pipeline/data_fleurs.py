"""Non-English replay: FLEURS train split (CC BY 4.0), rows whose label has no written-form character.

Keeps the other 24 Parakeet languages anchored while the English spoken-form data trains. Labels
are `raw_transcription` (cased, punctuated). The test split is never used (it is an evaluation set).
Writes /vol/replay/fleurs/<lang>/*.wav and /vol/replay/fleurs_train_multi.jsonl.
"""

import io
import json
import os
import random

import librosa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from common import WRITTEN_FORM

LANGS = ("bg_bg hr_hr cs_cz da_dk nl_nl et_ee fi_fi fr_fr de_de el_gr hu_hu it_it lv_lv lt_lt mt_mt "
         "pl_pl pt_br ro_ro sk_sk sl_si es_419 sv_se ru_ru uk_ua").split()


def prepare(vol: str, hours_per_lang: float = 0.4, seed: int = 1) -> None:
    out = f"{vol}/replay/fleurs"
    rows = []
    for lang in LANGS:
        path = hf_hub_download("google/fleurs", f"{lang}/train/0000.parquet", repo_type="dataset",
                               revision="refs/convert/parquet")
        t = pq.read_table(path, columns=["id", "audio", "raw_transcription"]).to_pylist()
        random.Random(f"{seed}:{lang}").shuffle(t)
        os.makedirs(f"{out}/{lang}", exist_ok=True)
        acc, kept, dropped = 0.0, 0, 0
        for r in t:
            if acc >= hours_per_lang * 3600:
                break
            if WRITTEN_FORM.search(r["raw_transcription"]):
                dropped += 1
                continue
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            wav = f"{out}/{lang}/{r['id']}.wav"
            sf.write(wav, audio, 16000, subtype="PCM_16")
            dur = len(audio) / 16000
            acc += dur
            kept += 1
            rows.append({"audio_filepath": wav, "duration": dur, "text": r["raw_transcription"],
                         "id": f"{lang}_{r['id']}", "lang": lang})
        print(f"{lang}: {kept} rows, {acc / 3600:.2f} h, skipped {dropped} with written-form characters")
    with open(f"{vol}/replay/fleurs_train_multi.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"fleurs replay: {len(rows)} rows, {sum(r['duration'] for r in rows) / 3600:.1f} h")
