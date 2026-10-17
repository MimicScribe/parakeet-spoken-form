"""Number-free look-alike rows from YODAS-Granary (2026-10-16): captions WITHOUT digits that hold a word the model
could mistake for a number (ml_variety look_homo: "sei", "sem", "sans", "acht", "раз"), number words used as plain
words (look_idiom: "uno de ellos", "zu zweit", "à deux") and plain controls (no number word at all). They teach the
model never to write a number that was not said. Labels are the captions, checked later against an external
transcript (ml_variety.plain_check); nothing here labels.

    modal run ml_variety_modal.py --langs de,es,fr,it,nl,pt --tag ylook1 --per-class 300
    -> /vol/ml/p3v2/<lang>/cands_<tag>.jsonl (+ wav/<tag>_<id>.wav), "kinds": ["look_homo"] / ["look_idiom"] / ["control"]
"""
import json
import os
import random
import re

import modal

VOL = "/vol"
OUT2 = f"{VOL}/ml/p3v2"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa")
         .add_local_python_source("ml_variety"))
app = modal.App("parakeet-ml-variety")


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=4 * 3600, max_containers=12)
def look_scan_lang(lang: str, tag: str, per_class: int = 300, max_files: int = 60, part: str = "asr_only",
                   max_dur: float = 15.0, per_video: int = 3) -> dict:
    import io
    from collections import Counter

    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    import ml_variety as MV

    out = f"{OUT2}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    numw = re.compile(r"\b(?:" + MV._NUMW[lang] + r")\b", re.I)
    pats = MV.patterns(lang)
    fl = sorted(f for f in HfApi().list_repo_files("espnet/yodas-granary", repo_type="dataset")
                if f.startswith("data/") and f"/{part}/" in f and f.split("/")[1][:2] == lang)
    random.Random(f"look-{lang}").shuffle(fl)
    got, vids, st = Counter(), Counter(), Counter()
    cf = open(f"{out}/cands_{tag}.jsonl", "w")
    for path in fl[:max_files]:
        if all(got[k] >= per_class for k in ("look_homo", "look_idiom", "control")):
            break
        local = hf_hub_download("espnet/yodas-granary", path, repo_type="dataset", local_dir="/tmp/src")
        st["files"] += 1
        pf = pq.ParquetFile(local)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=["utt_id", "audio", "text", "duration"]).to_pylist():
                st["rows_seen"] += 1
                cap = (r.get("text") or "").strip()
                if not cap or re.search(r"\d", cap) or len(cap.split()) < 5:
                    continue
                c = MV.classify(cap, lang)
                k = ("look_homo" if "look_homo" in c else "look_idiom" if "look_idiom" in c
                     else "control" if not numw.search(cap) else None)
                if k is None or got[k] >= per_class:
                    continue
                if k == "control" and random.Random(r["utt_id"]).random() > 0.05:
                    continue  # controls: a thin random sample of plain rows
                vid = MV.group_of({"id": r["utt_id"]})
                if vids[vid] >= per_video:
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                d = len(x) / 16000
                if d < 1.5 or d > max_dur:
                    continue
                wav = f"{out}/wav/{tag}_{r['utt_id']}.wav"
                sf.write(wav, x, 16000, subtype="PCM_16")
                row = {"audio_filepath": wav, "duration": round(d, 3), "text": cap, "lang": lang, "src": "yodas",
                       "id": r["utt_id"], "origin": "new", "kinds": [k], "variety": sorted(c)}
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
                got[k] += 1
                vids[vid] += 1
        os.remove(local)
        cf.flush()
        vol.commit()
    cf.close()
    vol.commit()
    return {"lang": lang, "tag": tag, **st, "kept": dict(got),
            "hours": None}


@app.local_entrypoint()
def main(langs: str = "de,es,fr,it,nl,pt", tag: str = "ylook1", per_class: int = 300, max_files: int = 60,
         part: str = "asr_only"):
    for r in look_scan_lang.starmap([(l, tag, per_class, max_files, part) for l in langs.split(",")]):
        print(json.dumps(r, ensure_ascii=False))


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=3600, max_containers=12)
def pack_lang(lang: str, cands_rel: str, out_rel: str) -> dict:
    """Pack the wavs of a cands file (path relative to /vol) as 16 kHz FLAC into one tar (for the second transcript)."""
    import io
    import tarfile

    import soundfile as sf

    rows = [json.loads(l) for l in open(f"{VOL}/{cands_rel}") if l.strip()]
    n = 0
    with tarfile.open(f"{VOL}/{out_rel}", "w") as tar:
        for r in rows:
            try:
                x, sr = sf.read(r["audio_filepath"], dtype="float32")
            except Exception:  # noqa: BLE001
                continue
            b = io.BytesIO()
            sf.write(b, x, sr, format="FLAC")
            data = b.getvalue()  # (libsndfile seeks back to finish the header: tell() is not the size)
            ti = tarfile.TarInfo(f"{r['id']}.flac")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
            n += 1
    vol.commit()
    return {"lang": lang, "rows": len(rows), "packed": n, "tar": out_rel}


@app.local_entrypoint()
def pack(langs: str, tag: str):
    """modal run ml_variety_modal.py::pack --langs de,fr --tag ylook1 -> /vol/ml/p3v2/<lang>/pack_<tag>.tar"""
    jobs = [(l, f"ml/p3v2/{l}/cands_{tag}.jsonl", f"ml/p3v2/{l}/pack_{tag}.tar") for l in langs.split(",")]
    for r in pack_lang.starmap(jobs):
        print(json.dumps(r))
