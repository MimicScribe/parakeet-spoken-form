"""Second-recognizer verbatim transcripts for Russian / Ukrainian rows (2026-10-16), the `--ext` input of
ml_p3.label2. The recognizer must write numbers as words (ml_decide compares readings word for word), so it is an
open model trained on spoken-form text, independent of the CTC reader (ml_decide.MODELS):
    ru  ai-sage/GigaAM-v3, revision "rnnt" (MIT; Conformer RNN-T, lowercase words, no punctuation)
    uk  nvidia/stt_uk_citrinet_1024_gamma_0_25 (CC-BY-4.0; Citrinet CTC, lowercase words). Not
        Yehor/w2v-bert-uk-v2.1: on FLEURS uk it drops or garbles most number words ("у 2003 році" -> "тртому році").

    modal run ml_slav_ext.py --lang ru --manifests ml/p3v2/ru/cands_yodasr1.jsonl --ids dry/ids_dry.txt --out ext_dry.jsonl

Rows: {"id", "text", "status": "completed", "model"} in /vol/ml/p3v2/<lang>/<out>. `ids` (optional, relative to
/vol/ml/p3v2/<lang>/) limits the rows; manifests are volume paths of jsonl rows with "id" and "audio_filepath".
"""

import json
import os

import modal

VOL = "/vol"
OUT2 = f"{VOL}/ml/p3v2"
vol = modal.Volume.from_name("parakeet-spoken-form")
EXT_MODELS = {"ru": ("ai-sage/GigaAM-v3", "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8"),  # the "rnnt" branch, pinned (remote code)
              "uk": ("nvidia/stt_uk_citrinet_1024_gamma_0_25", None)}
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("torch==2.8.0", "torchaudio==2.8.0", "transformers==4.56.2", "hydra-core", "omegaconf",
                      "sentencepiece", "soundfile", "librosa", "numpy<2")
         # the GigaAM modeling file imports pyannote for its long-form VAD only; rows here are <= 24 s per call, so an
         # empty package satisfies the import check
         .run_commands("mkdir -p /usr/local/lib/python3.11/site-packages/pyannote",
                       "touch /usr/local/lib/python3.11/site-packages/pyannote/__init__.py"))
nemo_image = (modal.Image.debian_slim(python_version="3.11")
              .apt_install("ffmpeg", "libsndfile1", "build-essential")
              .pip_install("torch==2.4.1", "torchaudio==2.4.1", "Cython", "packaging")
              .pip_install("nemo_toolkit[asr]==2.1.0", "soundfile", "librosa", "huggingface_hub", "numpy<2", "pyarrow==16.1.0", "datasets==2.21.0"))
app = modal.App("parakeet-ml-slav-ext")


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600)
def ext_ru(manifests: list, ids: str = "", out: str = "ext.jsonl") -> dict:
    return _ext("ru", manifests, ids, out)


@app.function(image=nemo_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600)
def ext_uk(manifests: list, ids: str = "", out: str = "ext.jsonl") -> dict:
    return _ext("uk", manifests, ids, out)


def _ext(lang: str, manifests: list, ids: str = "", out: str = "ext.jsonl") -> dict:
    import librosa
    import soundfile as sf
    import torch

    rows = []
    for m in manifests:
        rows += [json.loads(l) for l in open(f"{VOL}/{m}") if l.strip()]
    if ids:
        want = {l.strip() for l in open(f"{OUT2}/{lang}/{ids}") if l.strip()}
        rows = [r for r in rows if r["id"] in want]
    name, rev = EXT_MODELS[lang]
    if lang == "ru":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name, revision=rev, trust_remote_code=True).to("cuda").eval()

        def run(path, x):
            if len(x) <= 24 * 16000:
                return model.transcribe(path)
            parts = []
            for k in range(0, len(x), 20 * 16000):  # longer rows: 20 s pieces (rare; a word at a cut may be lost)
                p = f"/tmp/piece_{k}.wav"
                sf.write(p, x[k:k + 20 * 16000], 16000)
                parts.append(model.transcribe(p))
            return " ".join(parts)
    else:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(name).to("cuda").eval()

        def run(path, x):
            if path != f"/tmp/in.wav":
                sf.write("/tmp/in.wav", x, 16000)
            with torch.inference_mode():
                h = model.transcribe(["/tmp/in.wav"], batch_size=1, verbose=False)
            h = h[0] if isinstance(h, tuple) else h
            h = h[0]
            return h if isinstance(h, str) else h.text

    os.makedirs(f"{OUT2}/{lang}", exist_ok=True)
    n_ok = 0
    with open(f"{OUT2}/{lang}/{out}", "w") as f:
        for r in rows:
            try:
                x, sr = sf.read(r["audio_filepath"], dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                path = r["audio_filepath"]
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                    path = f"/tmp/{r['id']}.wav"
                    sf.write(path, x, 16000)
                t = run(path, x).strip().lower()
                f.write(json.dumps({"id": r["id"], "text": t, "status": "completed", "model": name}, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"id": r["id"], "status": "error", "error": repr(e)[:200]}) + "\n")
    vol.commit()
    return {"lang": lang, "rows": len(rows), "ok": n_ok, "out": f"{lang}/{out}"}


@app.local_entrypoint()
def main(lang: str, manifests: str, ids: str = "", out: str = "ext.jsonl"):
    fn = ext_ru if lang == "ru" else ext_uk
    print(json.dumps(fn.remote(manifests.split(","), ids, out)))
