"""Voice generated rows with Kokoro-82M and write 16 kHz WAVs plus NeMo manifest lines."""

import hashlib
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Grade C+ or better (Kokoro VOICES.md). Two held out for dev.
TRAIN_VOICES = ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "am_fenrir",
                "am_michael", "bf_emma"]
DEV_VOICES = ["af_sarah", "am_puck"]

_pipes = {}


def _pipe(voice: str):
    from kokoro import KPipeline

    code = voice[0]  # 'a' American, 'b' British
    if code not in _pipes:
        _pipes[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
    return _pipes[code]


def synth(text: str, voice: str, speed: float) -> np.ndarray:
    chunks = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
              for _, _, a in _pipe(voice)(text, voice=voice, speed=speed, split_pattern=None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def voice_rows(rows: list[dict], out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for row in rows:
        seed = int(hashlib.md5(row["id"].encode()).hexdigest()[:8], 16)
        r = random.Random(seed)
        voice = r.choice(DEV_VOICES if row["split"] == "dev" else TRAIN_VOICES)
        speed = round(r.uniform(0.85, 1.3), 2)
        audio24 = synth(row["tts_text"], voice, speed)
        if len(audio24) < 2400:
            print("empty audio", row["id"])
            continue
        x = resample_poly(audio24, 2, 3).astype(np.float32)
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (-3 / 20) * 10 ** (r.uniform(-12, 0) / 20)
        pad = lambda: np.zeros(int(16000 * r.uniform(0.2, 0.6)), dtype=np.float32)  # noqa: E731
        x = np.concatenate([pad(), x, pad()])
        path = f"{out_dir}/{row['id']}.wav"
        sf.write(path, x, 16000, subtype="PCM_16")
        out.append({"audio_filepath": path, "duration": round(len(x) / 16000, 3), "text": row["text"],
                    "id": row["id"], "split": row["split"], "voice": voice, "speed": speed,
                    "kinds": row["kinds"], "template": row["template"]})
    return out
