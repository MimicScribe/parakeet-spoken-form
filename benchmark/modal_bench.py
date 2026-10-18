"""GPU side of the public benchmark (Modal): plain NeMo greedy decodes of the held-out rows and of the stitched seam bed.

Run from the repo root:
  modal run benchmark/modal_bench.py::heldout --arm stock --refs DIR --out DIR [--langs de,fr] [--limit 40]
  modal run benchmark/modal_bench.py::seam --arm stock --manifest bed.json --bed-dir ml/realbed/out --out dec.json \
      [--langs de,fr] [--max-files 1]
Arms: "stock" = Parakeet Ultra (ultra.nemo from the pinned Hugging Face revision); "ckpt:<path on the volume>" = Ultra
with a fine-tuned decoder + joint state dict (strict=False over Ultra; encoder tensors in the file are loaded too).

Decoding: m.transcribe() with the checkpoint's own greedy TDT decoding, batch 16 (held-out) / 16 windows (seam), fp32,
no language pinning, no post-processing. Seam windows are decoded with word timestamps (timestamps=True); single
sentences and held-out rows without, so their text matches a plain transcribe call. Audio is read at 16 kHz mono from
the volume; nothing is written to the volume (results come back to the local machine).
"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

app = modal.App("parakeet-spoken-form-bench")
vol = modal.Volume.from_name("parakeet-spoken-form")
VOL = "/vol"
ULTRA_REPO, ULTRA_FILE = "ncannings/parakeet-ultra-nemo", "ultra.nemo"
ULTRA_REV = "68d18c58f3d5012f885f4e01039684cd398aff19"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("ffmpeg", "sox", "libsndfile1", "git", "wget", "build-essential")
    .uv_pip_install("torch==2.8.0", "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("Cython", "packaging")
    .uv_pip_install("nemo_toolkit[asr,cu12]==2.7.3", "huggingface_hub", "soundfile", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

SR = 16000


def window_starts(dur: float, win: float, stride: float) -> list:
    """the same as seam_harness.window_starts (kept here so the container needs no repo code; test_benchmark.py
    checks they agree)"""
    starts = [0.0]
    while starts[-1] + win < dur:
        starts.append(round(starts[-1] + stride, 3))
    return starts


def _model(arm: str):
    import nemo.collections.asr as nemo_asr
    import torch
    from huggingface_hub import hf_hub_download

    path = f"{VOL}/ultra.nemo"
    if not os.path.exists(path):
        path = hf_hub_download(ULTRA_REPO, ULTRA_FILE, revision=ULTRA_REV)
    m = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu")
    if arm != "stock":
        assert arm.startswith("ckpt:"), arm
        sd = torch.load(f"{VOL}/{arm[5:]}", map_location="cpu")
        sd = sd.get("state_dict", sd)
        res = m.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, f"{len(res.unexpected_keys)} unexpected keys, e.g. {res.unexpected_keys[:3]}"
        print(f"{arm}: loaded {len(sd)} tensors")
    return m.cuda().eval()


def _load(path: str):
    import numpy as np
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio

        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return np.ascontiguousarray(a, dtype="float32")


def _texts(hyps) -> list:
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def decode_files(arm: str, paths: list, batch: int = 16) -> list:
    """plain transcribe of whole files (held-out rows, single sentences) -> texts in order"""
    m = _model(arm)
    out = []
    for i in range(0, len(paths), batch):
        out += _texts(m.transcribe([_load(p) for p in paths[i:i + batch]], batch_size=batch, verbose=False,
                                   return_hypotheses=True))
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def decode_windows(arm: str, files: list, offs: list, win: float, stride: float, batch: int = 16) -> dict:
    """files: [(key, path)]. For every file and offset: leading silence of `off` seconds, windows of `win` s every
    `stride` s, each decoded with word timestamps. -> {"<key>_o<off>": {"dur", "windows": [{"start", "words"}]}}"""
    import numpy as np

    m = _model(arm)
    frame = float(m.cfg.preprocessor.window_stride) * int(getattr(m.encoder, "subsampling_factor", 8))
    jobs = []
    out = {}
    for key, path in files:
        a = _load(path)
        for off in offs:
            x = np.concatenate([np.zeros(int(round(off * SR)), dtype="float32"), a])
            dur = len(x) / SR
            k = f"{key}_o{off:g}"
            out[k] = {"dur": dur, "windows": []}
            for s in window_starts(dur, win, stride):
                jobs.append((k, s, x[int(s * SR):int((s + win) * SR)]))
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        hyps = m.transcribe([c[2] for c in chunk], batch_size=batch, verbose=False, return_hypotheses=True,
                            timestamps=True)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        for (k, s, _), h in zip(chunk, hyps):
            words = []
            for w in (h.timestamp or {}).get("word", []):
                t0 = w.get("start", w.get("start_offset", 0) * frame)
                t1 = w.get("end", w.get("end_offset", 0) * frame)
                words.append([w["word"], round(float(t0), 3), round(float(t1), 3)])
            out[k]["windows"].append({"start": s, "words": words, "text": h.text})
    return out


@app.local_entrypoint()
def heldout(arm: str = "stock", refs: str = "", out: str = "", langs: str = "", limit: int = 0):
    """decode the held-out label rows (refs/heldout_<lang>.jsonl, audio_filepath on the volume) ->
    out/heldout_<lang>.jsonl rows {id, ref, hyp} (value_accuracy.py input)"""
    import common as C

    for p in sorted(Path(refs).glob("heldout_*.jsonl")):
        lang = p.stem.split("_", 1)[1]
        if langs and lang not in langs.split(","):
            continue
        rows = [r for r in C.read_jsonl(p) if r.get("spans")]
        rows = rows[:limit] if limit else rows
        texts = decode_files.remote(arm, [r["audio_filepath"] for r in rows])
        C.write_jsonl(Path(out) / p.name, [{"id": r["id"], "ref": r["text"], "hyp": t} for r, t in zip(rows, texts)])
        print(f"{lang}: {len(rows)} rows -> {out}/{p.name}")


@app.local_entrypoint()
def seam(arm: str = "stock", manifest: str = "", bed_dir: str = "ml/realbed/out", out: str = "", langs: str = "",
         max_files: int = 0, n_offsets: int = 5, win: float = 15.0, stride: float = 10.0):
    """decode a stitched seam bed: base/<file>.wav at n_offsets offsets in windows, and sent/<id>.wav alone ->
    out (JSON for seam_harness.py build)"""
    import seam_harness as SH

    m = json.load(open(manifest, encoding="utf-8"))
    per_lang = {}
    files = []
    for f, sents in m.items():
        L = sents[0]["lang"]
        if langs and L not in langs.split(","):
            continue
        if max_files and per_lang.get(L, 0) >= max_files:
            continue
        per_lang[L] = per_lang.get(L, 0) + 1
        files.append(f)
    offs = SH.offsets(n_offsets, stride)
    sent_ids = [s["id"] for f in files for s in m[f]]
    long_call = decode_windows.spawn(arm, [(f, f"{VOL}/{bed_dir}/base/{f}.wav") for f in files], offs, win, stride)
    single = decode_files.remote(arm, [f"{VOL}/{bed_dir}/sent/{i}.wav" for i in sent_ids])
    res = {"arm": arm, "geometry": {"win": win, "stride": stride, "offsets": offs},
           "long": long_call.get(), "single": dict(zip(sent_ids, single))}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False)
    print(f"{arm}: {len(files)} files x {len(offs)} offsets, {len(sent_ids)} sentences -> {out}")
