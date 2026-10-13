"""Retrain-4 step P3 (R3.11): spoken-form DATA for the Tier-1 languages (es fr de pt it nl). No training.

Stage A (CPU)  scan_lang   : new real rows. VoxPopuli TRAIN (CC0) files beyond the 4 the D2 harvest used (pt is not in
                             VoxPopuli: YODAS-Granary asr_only shards beyond the harvest's 40); keep rows whose raw_text
                             has digits, every digit run inside a candidate-able number span, 1-20 s. Writes
                             /vol/ml/p3/<lang>/{cands.jsonl,wav/}.
Stage B (L4)   label_lang  : the per-language xlsr-53 CTC reading check (ml_decide.decide) over (a) the D2 digit rows
                             and (b) the new rows. A row is kept ONLY when every number span is decided and no digit
                             remains; label = caption with each span replaced by its decided reading. A row whose label
                             differs from the CTC greedy text by CER > MAX_CER is dropped (raw_text edges that do not
                             match the audio). Output /vol/ml/p3/<lang>/{rows.jsonl,stats.json,wav/}.

Everything is written under /vol/ml/p3/ only. Crops: both sources are capped at 20 s (longer rows are never taken); each
kept row is flagged `edge_quiet` (a quiet 20 ms frame in each outer 160 ms, the mix_build.trim_edges rule) for the mix
builder's trim step. No word-level cutting is done: the CTC reader gives no per-word times worth trusting here.

    modal run ml_p3.py::scan --langs de,es --hours 4.5
    modal run ml_p3.py::label --langs de
"""

import json
import os
import re
import unicodedata

import modal

VOL = "/vol"
OUT = f"{VOL}/ml/p3"
MAX_CER = 0.25
vol = modal.Volume.from_name("parakeet-spoken-form")
LANGS = ["de", "es", "fr", "it", "nl", "pt"]
VOX = {"de", "es", "fr", "it", "nl"}
VOX_SKIP = {"nl": 0, "it": 0}  # default 4: harvest.sources() used the first 4 sorted VoxPopuli files; nl/it have only
# ~4-5 files, so they restart at 0 and rely on the seen-id exclusion (every D2 decision is in audit.jsonl)
YODAS_SKIP = 40  # and the first 40 shuffled YODAS shards

scan_image = (modal.Image.debian_slim(python_version="3.11")
              .apt_install("ffmpeg", "libsndfile1")
              .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa")
              .add_local_python_source("ml_cands"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-ml-p3")


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=4 * 3600, max_containers=6)
def scan_lang(lang: str, hours: float = 4.5, max_files: int = 10) -> dict:
    import io

    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    import ml_cands as M

    out = f"{OUT}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    seen = set()
    for line in open(f"{VOL}/d2/{lang}/audit.jsonl"):
        seen.add(json.loads(line)["id"])
    api = HfApi()
    if lang in VOX:
        files = sorted(f for f in api.list_repo_files("facebook/voxpopuli", repo_type="dataset")
                       if f.startswith(f"{lang}/train-"))[VOX_SKIP.get(lang, 4):]
        repo, kind, cols = "facebook/voxpopuli", "vox", ["audio_id", "audio", "raw_text"]
    else:
        import random

        fl = sorted(f for f in api.list_repo_files("espnet/yodas-granary", repo_type="dataset")
                    if f.startswith("data/") and "/asr_only/" in f and f.split("/")[1][:2] == lang)
        random.Random(lang).shuffle(fl)
        files = fl[YODAS_SKIP:]
        repo, kind, cols = "espnet/yodas-granary", "yodas", ["utt_id", "audio", "text", "duration"]
    st = {"lang": lang, "kind": kind, "files": 0, "rows_seen": 0, "digit_rows": 0, "kept": 0, "hours": 0.0,
          "skipped_unspannable": 0, "skipped_dur": 0, "skipped_seen": 0}
    rows = []
    for path in files[:max_files]:
        if st["hours"] >= hours:
            break
        local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/src")
        st["files"] += 1
        pf = pq.ParquetFile(local)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                st["rows_seen"] += 1
                cap = (r.get("raw_text") or r.get("text") or "").strip()
                rid = r.get("audio_id") or r.get("utt_id")
                if not re.search(r"\d", cap):
                    continue
                st["digit_rows"] += 1
                if rid in seen:
                    st["skipped_seen"] += 1
                    continue
                if not M.all_digits_spanned(cap, lang):
                    st["skipped_unspannable"] += 1
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                d = len(x) / 16000
                if d < 1.0 or d > 20.0:
                    st["skipped_dur"] += 1
                    continue
                seen.add(rid)  # ids repeat inside VoxPopuli/YODAS shards: one audio file per id
                wav = f"{out}/wav/new_{rid}.wav"
                sf.write(wav, x, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": wav, "duration": round(d, 3), "text": cap, "lang": lang, "src": kind,
                             "id": rid, "origin": "new"})
                st["kept"] += 1
                st["hours"] += d / 3600
                if st["hours"] >= hours:
                    break
            if st["hours"] >= hours:
                break
        os.remove(local)
    with open(f"{out}/cands.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["hours"] = round(st["hours"], 2)
    json.dump(st, open(f"{out}/scan_stats.json", "w"))
    vol.commit()
    return st


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=6)
def label_lang(lang: str, limit: int = 0, new_hours: float = 1.5) -> dict:
    import shutil

    import librosa
    import soundfile as sf

    import ml_decide as D
    import ml_p3_core as C

    ctc = D.LangCTC(lang)
    out = f"{OUT}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    d2 = [dict(json.loads(l), origin="d2") for l in open(f"{VOL}/d2/{lang}/digit.jsonl")]
    new = [json.loads(l) for l in open(f"{out}/cands.jsonl")] if os.path.exists(f"{out}/cands.jsonl") else []
    import random

    random.Random(lang).shuffle(new)  # the quota takes rows from all scanned files, not just the first
    if limit:
        d2, new = d2[:limit], new[:limit]
    acc = C.Acc(lang)
    kept_new_h = 0.0
    kept = []
    for row in d2 + new:
        if row["origin"] == "d2":
            # D2 names wavs by id and FLEURS ids repeat across speakers: the later write won, so a row whose duration
            # differs from its file points at someone else's audio. Keep only rows that match their file.
            if abs(sf.info(row["audio_filepath"]).duration - row["duration"]) > 0.05:
                acc.add_skip(row, "d2_audio_mismatch")
                continue
        if row["origin"] == "new" and kept_new_h >= new_hours * 3600:
            acc.add_skip(row, "new_quota_reached")
            continue
        try:
            x, sr = sf.read(row["audio_filepath"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            r = D.decide(ctc, row["text"], ctc.lp(x))
        except Exception as e:  # noqa: BLE001
            acc.add_skip(row, f"error {type(e).__name__}")
            continue
        res = C.make_row(row, r, x)
        acc.add(row, r, res)
        if res.get("row"):
            nr = res["row"]
            dst = f"{out}/wav/{row['origin']}_{row['id']}.wav"
            if row["audio_filepath"] != dst:  # new rows already sit in wav/ (scan wrote them); D2 rows are copied
                shutil.copyfile(row["audio_filepath"], dst)
            nr["audio_filepath"] = dst
            kept.append(nr)
            if row["origin"] == "new":
                kept_new_h += nr["duration"]
    keep = {os.path.basename(nr["audio_filepath"]) for nr in kept}
    for fn in os.listdir(f"{out}/wav"):  # scan wavs of rejected / over-quota new rows
        if fn not in keep:
            os.remove(f"{out}/wav/{fn}")
    with open(f"{out}/rows.jsonl", "w") as f:
        for nr in kept:
            f.write(json.dumps(nr, ensure_ascii=False) + "\n")
    st = acc.stats(kept)
    json.dump(st, open(f"{out}/stats.json", "w"), ensure_ascii=False, indent=1)
    vol.commit()
    return st


@app.local_entrypoint()
def scan(langs: str = ",".join(LANGS), hours: float = 4.5, max_files: int = 10):
    ls = langs.split(",")
    for r in scan_lang.starmap([(l, hours, max_files) for l in ls]):
        print(json.dumps(r))


@app.local_entrypoint()
def label(langs: str = ",".join(LANGS), limit: int = 0, new_hours: float = 1.5):
    ls = langs.split(",")
    for r in label_lang.starmap([(l, limit, new_hours) for l in ls]):
        print(json.dumps({k: v for k, v in r.items() if k not in ("examples", "wrong_looking")}, ensure_ascii=False))
        for e in r.get("examples", [])[:10]:
            print("   ", e)
