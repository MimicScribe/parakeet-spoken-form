"""Modal side of the number-word caption pass (ml_numword): audio for a selection only, then a second-recognizer
transcript of every fetched clip (Qwen3-ASR-1.7B, value vote + fit check done locally).

    modal run ml_numword_modal.py::fetch --sel <local sel.jsonl> --tag nw1
        reads only the parquet ROW GROUPS that hold selected rows (columns utt_id + audio), writes
        /vol/ml/p3v2/<lang>/wav/<tag>_<id>.wav (16 kHz mono) and /vol/ml/numword/<tag>/fetched_<lang>.jsonl
    modal run ml_numword_modal.py::qwen --tag nw1
        /vol/ml/numword/<tag>/fetched_*.jsonl -> /vol/ml/numword/<tag>/qwen.jsonl ({id, lang, hyp}), resumable

sel.jsonl rows: {"id", "lang", "path" (repo file), "rg" (row group), ...} as ml_numword.select returns them.
"""

import json
import os

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
REPO = "espnet/yodas-granary"
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa")  # = ml_p3.scan_image
             .add_local_python_source("ml_cands", "ml_p3_core"))
qwen_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
              .pip_install("qwen-asr", "soundfile", "librosa"))
app = modal.App("parakeet-numword")
LN = {"de": "German", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
      "en": "English"}


@app.function(image=cpu_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=1, memory=4096,
              timeout=3600, max_containers=8, retries=1)  # 8: the HF resolver rate limit is shared
def fetch_shard(path: str, items: list, tag: str) -> list:
    """Audio of the selected rows of one shard: only their row groups are read."""
    import io

    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem

    import ml_p3_core as C

    pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/{path}", block_size=1 << 20), pre_buffer=True)
    want = {}
    for it in items:
        want.setdefault(it["rg"], {})[it["id"]] = it
    out = []
    for rg, ids in sorted(want.items()):
        for r in pf.read_row_group(rg, columns=["utt_id", "audio"]).to_pylist():
            it = ids.get(r["utt_id"])
            if it is None:
                continue
            try:
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            except Exception as e:  # noqa: BLE001
                out.append({"id": it["id"], "lang": it["lang"], "error": type(e).__name__})
                continue
            lang = it["lang"]
            os.makedirs(f"{VOL}/ml/p3v2/{lang}/wav", exist_ok=True)
            wav = f"{VOL}/ml/p3v2/{lang}/wav/{tag}_{it['id']}.wav"
            sf.write(wav, x, 16000, subtype="PCM_16")
            out.append({"id": it["id"], "lang": lang, "wav": wav, "duration": round(len(x) / 16000, 3),
                        "edge_quiet": C.edge_quiet(x)})
    vol.commit()
    return out


@app.local_entrypoint()
def fetch(sel: str, tag: str = "nw1", out_dir: str = "fetched"):
    items = [json.loads(l) for l in open(sel) if l.strip()]
    by = {}
    for it in items:
        by.setdefault(it["path"], []).append(it)
    res = []
    for r in fetch_shard.starmap([(p, its, tag) for p, its in sorted(by.items())], return_exceptions=True):
        if isinstance(r, Exception):
            print("shard failed:", repr(r)[:200])
            continue
        res += r
    langs = sorted({r["lang"] for r in res})
    os.makedirs(out_dir, exist_ok=True)
    for lang in langs:
        p = f"{out_dir}/{tag}_{lang}.jsonl"
        with open(p, "w") as f:
            for r in res:
                if r["lang"] == lang:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({lang: sum(1 for r in res if r["lang"] == lang and "wav" in r) for lang in langs},
          "errors", sum(1 for r in res if "error" in r))


@app.function(image=qwen_image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600)
def qwen_run(items: list, out: str, bs: int = 32) -> int:
    import time

    import librosa
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    vol.reload()
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = sorted([r for r in items if r["id"] not in done], key=lambda r: r.get("duration", 0))
    print("todo", len(todo), "done", len(done), flush=True)
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                      max_inference_batch_size=bs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, "a")
    t0 = time.time()
    for b in range(0, len(todo), bs):
        chunk, auds, ok = todo[b:b + bs], [], []
        for r in chunk:
            try:
                x, sr = sf.read(r["wav"], dtype="float32")
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                auds.append((x, 16000))
                ok.append(r)
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"id": r["id"], "lang": r["lang"], "hyp": None, "err": type(e).__name__}) + "\n")
        try:
            hs = [h.text for h in m.transcribe(audio=auds, language=[LN[r["lang"]] for r in ok])] if ok else []
        except Exception:  # noqa: BLE001  one bad batch: one by one
            hs = []
            for a, r in zip(auds, ok):
                try:
                    hs.append(m.transcribe(audio=a, language=LN[r["lang"]])[0].text)
                except Exception:  # noqa: BLE001
                    hs.append(None)
        for r, h in zip(ok, hs):
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "hyp": h}, ensure_ascii=False) + "\n")
        if (b // bs) % 10 == 0:
            f.flush()
            vol.commit()
            print(f"{b + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    f.close()
    vol.commit()
    return len(todo)


@app.local_entrypoint()
def qwen(fetched: str, tag: str = "nw1"):
    """fetched: comma list of local fetched_*.jsonl files (fetch output)."""
    items = [json.loads(l) for p in fetched.split(",") for l in open(p) if l.strip()]
    items = [r for r in items if "wav" in r]
    n = qwen_run.remote(items, f"{VOL}/ml/numword/{tag}/qwen.jsonl")
    print("transcribed", n)


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def pack_run(wavs: list, dest: str) -> int:
    """One tar of <id>.flac files (the second-opinion transcript step uploads local audio)."""
    import tarfile
    import tempfile

    import soundfile as sf

    vol.reload()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    n = 0
    with tempfile.TemporaryDirectory() as td, tarfile.open(dest, "w") as tf:
        for rid, wav in wavs:
            if not os.path.exists(wav):
                continue
            x, sr = sf.read(wav, dtype="int16")
            p = f"{td}/{rid}.flac"
            sf.write(p, x, sr, format="FLAC")
            tf.add(p, arcname=f"{rid}.flac")
            os.remove(p)
            n += 1
    vol.commit()
    return n


@app.local_entrypoint()
def pack(fetched: str, tag: str = "nw1", only: str = ""):
    """fetched: one local fetched_<lang>.jsonl; only: optional local file of ids (one per line) to pack."""
    rows = [json.loads(l) for l in open(fetched) if l.strip()]
    rows = [r for r in rows if "wav" in r]
    if only:
        keep = {l.strip() for l in open(only) if l.strip()}
        rows = [r for r in rows if r["id"] in keep]
    lang = rows[0]["lang"]
    name = os.path.basename(only)[:-4] if only else "all"
    dest = f"{VOL}/ml/numword/{tag}/pack_{lang}_{name}.tar"
    print("packed", pack_run.remote([(r["id"], r["wav"]) for r in rows], dest), "->", dest[len(VOL) + 1:])
