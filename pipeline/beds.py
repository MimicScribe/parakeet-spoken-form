"""Evaluation beds for the real-speech retrain (never trained on).

- vox: VoxPopuli TEST split (CC0), 15 languages, utterances whose STOCK decode carries a digit = real non-English
  number speech. The English-number-word leak gate. 1/3 dev, 2/3 SEALED (hash of the id).
- fleurs: FLEURS TEST, 24 languages: 100 utterances per language (WER gate) + every utterance whose transcript has a
  digit (leak gate, dev).
- noise: ~200 clips of noise only (OpenSLR 28 isotropic real noises; training augmentation uses only its
  point-source noises and simulated RIRs, so the bed is disjoint): no numbers may be emitted.
Writes /vol/beds/<bed>/{dev,sealed}.jsonl (+ wavs).
"""

import io
import json
import os
import random
import re
import subprocess
import zlib

from harvest import VOX

FLEURS = ("bg_bg hr_hr cs_cz da_dk nl_nl et_ee fi_fi fr_fr de_de el_gr hu_hu it_it lv_lv lt_lt mt_mt "
          "pl_pl pt_br ro_ro sk_sk sl_si es_419 sv_se ru_ru uk_ua").split()


def _split(i: str) -> str:
    return "dev" if zlib.crc32(i.encode()) % 3 == 0 else "sealed"


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def vox(vol: str, model, cap: int = 90) -> dict:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    import ft_eval

    out = f"{vol}/beds/vox"
    os.makedirs(f"{out}/wav", exist_ok=True)
    files = [f for f in HfApi().list_repo_files("facebook/voxpopuli", repo_type="dataset") if "/test-" in f]
    rows = {"dev": [], "sealed": []}
    stats = {}
    for lang in sorted(VOX):
        f = next(x for x in files if x.startswith(f"{lang}/test-"))
        t = pq.read_table(hf_hub_download("facebook/voxpopuli", f, repo_type="dataset"),
                          columns=["audio_id", "audio", "raw_text"]).to_pylist()
        random.Random(lang).shuffle(t)
        paths, meta = [], []
        for r in t[:1500]:
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                import librosa

                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            if not 1.0 <= len(x) / 16000 <= 25:
                continue
            p = f"{out}/wav/{lang}_{r['audio_id']}.wav"
            sf.write(p, x, 16000, subtype="PCM_16")
            paths.append(p)
            meta.append((r, len(x) / 16000))
        hyps = ft_eval.decode(model, paths, batch_size=32)
        n = 0
        for p, (r, d), h in zip(paths, meta, hyps):
            if re.search(r"\d", h["text"]) and n < cap:
                n += 1
                rows[_split(r["audio_id"])].append({"audio_filepath": p, "duration": d, "text": r["raw_text"],
                                                    "stock": h["text"], "lang": lang, "id": f"{lang}_{r['audio_id']}"})
            else:
                os.remove(p)
        stats[lang] = n
    for k, v in rows.items():
        _write(f"{out}/{k}.jsonl", v)
    return stats


def fleurs(vol: str, per_lang: int = 100) -> dict:
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    out = f"{vol}/beds/fleurs"
    os.makedirs(f"{out}/wav", exist_ok=True)
    wer_rows, num_rows, stats = [], [], {}
    for lang in FLEURS:
        path = hf_hub_download("google/fleurs", f"{lang}/test/0000.parquet", repo_type="dataset",
                               revision="refs/convert/parquet")
        t = pq.read_table(path, columns=["id", "audio", "raw_transcription"]).to_pylist()
        random.Random(lang).shuffle(t)
        nw = nn = 0
        for r in t:
            has_d = bool(re.search(r"\d", r["raw_transcription"]))
            if not has_d and nw >= per_lang:
                continue
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            p = f"{out}/wav/{lang}_{r['id']}.wav"
            sf.write(p, x, 16000, subtype="PCM_16")
            row = {"audio_filepath": p, "duration": len(x) / 16000, "text": r["raw_transcription"], "lang": lang,
                   "id": f"{lang}_{r['id']}"}
            if has_d:
                num_rows.append(row)
                nn += 1
            else:
                wer_rows.append(row)
                nw += 1
        stats[lang] = (nw, nn)
    _write(f"{out}/wer.jsonl", wer_rows)
    _write(f"{out}/num.jsonl", num_rows)
    return stats


def slr28(vol: str) -> str:
    root = f"{vol}/slr28"
    if not os.path.isdir(f"{root}/RIRS_NOISES"):
        os.makedirs(root, exist_ok=True)
        subprocess.run(f"cd {root} && wget -q --tries=5 https://www.openslr.org/resources/28/rirs_noises.zip "
                       f"&& unzip -q rirs_noises.zip && rm rirs_noises.zip", shell=True, check=True)
    return f"{root}/RIRS_NOISES"


def noise(vol: str, n: int = 200) -> int:
    import glob

    import librosa
    import numpy as np
    import soundfile as sf

    base = slr28(vol)
    files = sorted(glob.glob(f"{base}/real_rirs_isotropic_noises/*noise*.wav"))
    r = random.Random(7)
    out = f"{vol}/beds/noise"
    os.makedirs(f"{out}/wav", exist_ok=True)
    rows = []
    for i in range(n):
        f = r.choice(files)
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x[:, r.randrange(x.shape[1])]
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        d = r.uniform(4, 12)
        s = r.randint(0, max(0, len(x) - int(d * 16000)))
        seg = x[s:s + int(d * 16000)]
        gain = 10 ** (r.uniform(-45, -15) / 20) / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
        seg = np.clip(seg * gain, -1, 1)
        p = f"{out}/wav/noise_{i}.wav"
        sf.write(p, seg, 16000, subtype="PCM_16")
        rows.append({"audio_filepath": p, "duration": len(seg) / 16000, "text": "", "id": f"noise_{i}"})
    _write(f"{out}/all.jsonl", rows)
    return len(rows)


E21_LOCAL = {"4320211", "4341191", "4346818", "4359971", "4365024", "4366522", "4366893", "4367535", "4383161",
             "4384964", "4387332"}  # the 11-file dev/gate set; the other 33 calls are the SEALED real-number bed


def e21_ref(nlp: str) -> str:
    out = []
    for line in nlp.splitlines()[1:]:
        p = line.split("|")
        if len(p) < 6:
            continue
        tok = p[0]
        if p[5] == "UC":
            tok = tok[:1].upper() + tok[1:]
        out.append(tok + (p[4] or ""))
    return " ".join(out)


def e21x(vol: str, chunk_s: float = 20.0) -> dict:
    """SEALED: the 33 Earnings-21 calls outside the gate set, cut into fixed 20 s chunks (both arms see the same
    cuts; scoring is numbers by VALUE per call, so chunk edges cost every arm alike). Refs to /vol/beds/e21x/refs/."""
    import urllib.request

    import soundfile as sf

    out = f"{vol}/beds/e21x"
    os.makedirs(f"{out}/wav", exist_ok=True)
    os.makedirs(f"{out}/refs", exist_ok=True)
    api = "https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings21"
    media = json.load(urllib.request.urlopen(f"{api}/media"))
    refs = json.load(urllib.request.urlopen(f"{api}/transcripts/nlp_references"))
    rows, ids = [], []
    ref_url = {x["name"].split(".")[0]: x["download_url"] for x in refs}
    for m in media:
        fid = m["name"].split(".")[0]
        nlp = urllib.request.urlopen(ref_url[fid]).read().decode()
        open(f"{out}/refs/{fid}.txt", "w").write(e21_ref(nlp))
        if fid in E21_LOCAL:
            continue
        mp3 = f"/tmp/{fid}.mp3"
        urllib.request.urlretrieve(m["download_url"], mp3)
        wav = f"/tmp/{fid}.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", wav],
                       check=True)
        x, _ = sf.read(wav, dtype="float32")
        n = int(chunk_s * 16000)
        for k in range(0, len(x), n):
            seg = x[k:k + n]
            if len(seg) < 16000:
                continue
            p = f"{out}/wav/{fid}_{k // n:04d}.wav"
            sf.write(p, seg, 16000, subtype="PCM_16")
            rows.append({"audio_filepath": p, "duration": len(seg) / 16000, "text": "", "id": f"{fid}_{k // n:04d}",
                         "file": fid})
        ids.append(fid)
        os.remove(mp3)
        os.remove(wav)
    _write(f"{out}/sealed.jsonl", rows)
    return {"files": len(ids), "chunks": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 1)}


# ---- retrain-3 bed rebuild --------------------------------------------------

EN_STOP = set("the and of to is that we in it this for are have be not on with you our i was they what so but".split())


def vox3(vol: str) -> dict:
    """vox dev/sealed minus English floor speech (stock decode >= 25% English function words) and duplicate ids."""
    st = {}
    for split in ("dev", "sealed"):
        rows = [json.loads(line) for line in open(f"{vol}/beds/vox/{split}.jsonl")]
        seen, keep, en = set(), [], 0
        for r in rows:
            w = re.findall(r"[a-z']+", r["stock"].lower())
            if w and sum(x in EN_STOP for x in w) / len(w) >= 0.25:
                en += 1
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            keep.append(r)
        _write(f"{vol}/beds/vox/{split}3.jsonl", keep)
        st[split] = {"in": len(rows), "english_dropped": en, "kept": len(keep)}
    return st


def fleurs3(vol: str, screen_per_lang: int = 100) -> dict:
    """FLEURS TEST, the FULL split per language (M2), one wav per ROW (the v1 bed keyed wavs by sentence id, so the
    speakers of one sentence overwrote each other). wer_full = every number-free row; wer = the first
    `screen_per_lang` of them (screening); num = every digit row."""
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    out = f"{vol}/beds/fleurs3"
    os.makedirs(f"{out}/wav", exist_ok=True)
    full, screen, num, stats = [], [], [], {}
    for lang in FLEURS:
        path = hf_hub_download("google/fleurs", f"{lang}/test/0000.parquet", repo_type="dataset",
                               revision="refs/convert/parquet")
        t = pq.read_table(path, columns=["id", "audio", "raw_transcription"]).to_pylist()
        random.Random(lang).shuffle(t)
        nw = nn = 0
        for k, r in enumerate(t):
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            p = f"{out}/wav/{lang}_{r['id']}_{k}.wav"
            sf.write(p, x, 16000, subtype="PCM_16")
            row = {"audio_filepath": p, "duration": len(x) / 16000, "text": r["raw_transcription"], "lang": lang,
                   "id": f"{lang}_{r['id']}_{k}"}
            if re.search(r"\d", r["raw_transcription"]):
                num.append(row)
                nn += 1
            else:
                full.append(row)
                if nw < screen_per_lang:
                    screen.append(row)
                nw += 1
        stats[lang] = (nw, nn)
    _write(f"{out}/wer_full.jsonl", full)
    _write(f"{out}/wer.jsonl", screen)
    _write(f"{out}/num.jsonl", num)
    return stats


def e21_dev_cut(vol: str, win_s: float = 14.96, stride_s: float = 9.44, phases=(0.0, 3.15, 6.3)) -> list:
    """The 11 E21_LOCAL calls as streaming-style windows (187-frame window, 118-frame stride) at
    three phase offsets, so every number sits at several left-edge positions (M1). Returns every window row."""
    import soundfile as sf

    out = f"{vol}/beds/e21_dev"
    os.makedirs(f"{out}/wav", exist_ok=True)
    rows = []
    for fid in sorted(E21_LOCAL):
        x, sr = sf.read(f"{vol}/e21local/audio/{fid}.wav", dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        assert sr == 16000, sr
        n, s = int(win_s * 16000), int(stride_s * 16000)
        for ph in phases:
            for a in range(int(ph * 16000), len(x) - 16000, s):
                seg = x[a:a + n]
                p = f"{out}/wav/{fid}_{a}.wav"
                sf.write(p, seg, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": p, "duration": round(len(seg) / 16000, 3), "text": "", "call": fid,
                             "start": round(a / 16000, 2), "phase": ph, "id": f"{fid}_{a}"})
    return rows
