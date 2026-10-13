"""D2/D2p: non-English real speech, pseudo-labelled by STOCK Ultra (the contract: stock writes digits there).

Sources per language: VoxPopuli train (CC0; 15 of our languages) and YODAS-Granary asr_only shards. Rows are decoded
with stock; a row is kept when
  - digit row (D2): stock's text has a digit, every 2+ digit run of the caption appears in stock's text, CER(caption,
    stock) <= 0.20, and text LID of stock's text = the language;
  - plain row (D2p): no digit in stock's text or the caption, CER <= 0.15, LID ok.
Label = stock's text. Quotas stop a language early. Writes /vol/d2/<lang>/{digit,plain}.jsonl + wavs, and an audit file
with every decision (LID contamination audit, stop rule > 1%).
"""

import io
import json
import os
import re

import numpy as np

VOX = {"cs", "de", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"}
ALL = "bg cs da de el es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split()


def _cer(a, b):
    a = re.sub(r"[\W_]", "", a.lower())
    b = re.sub(r"[\W_]", "", b.lower())
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def _runs(t):
    return [re.sub(r"[,.\s]", "", m) for m in re.findall(r"\d[\d,.\s]*\d|\d", t)]


def sources(lang: str, yodas_shards: int = 40, vox_files: int = 4):
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    if lang in VOX:
        fl = sorted(f for f in api.list_repo_files("facebook/voxpopuli", repo_type="dataset")
                    if f.startswith(f"{lang}/train-"))
        out += [("vox", f) for f in fl[:vox_files]]
    fl = sorted(f for f in api.list_repo_files("espnet/yodas-granary", repo_type="dataset")
                if f.startswith("data/") and "/asr_only/" in f and f.split("/")[1][:2] == lang)
    import random

    random.Random(lang).shuffle(fl)
    out += [("yodas", f) for f in fl[:yodas_shards]]
    return out


def rows_of(kind: str, path: str):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    repo = "facebook/voxpopuli" if kind == "vox" else "espnet/yodas-granary"
    local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/src")
    pf = pq.ParquetFile(local)
    cols = ["audio_id", "audio", "raw_text"] if kind == "vox" else ["utt_id", "audio", "text", "duration"]
    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg, columns=cols).to_pylist():
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                import librosa

                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            d = len(x) / 16000
            if d < 1.0 or d > 20.0:
                continue
            yield {"id": (r.get("audio_id") or r.get("utt_id")), "caption": r.get("raw_text") or r.get("text") or "",
                   "x": x, "dur": d, "src": kind}
    os.remove(local)


def harvest(vol: str, model, lang: str, digit_h: float = 1.2, plain_h: float = 1.2, max_decode_h: float = 40.0,
            exclude_ids: set | None = None):
    import langid
    import soundfile as sf

    import ft_eval

    langid.set_languages(ALL + ["en"])
    out = f"{vol}/d2/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    got = {"digit": [], "plain": []}
    hours = {"digit": 0.0, "plain": 0.0}
    audit = open(f"{out}/audit.jsonl", "w")
    decoded = 0.0
    quota = {"digit": digit_h, "plain": plain_h}
    stats = {"rows": 0, "lid_reject": 0, "cer_reject": 0, "digit_mismatch": 0}
    for kind, path in sources(lang):
        if hours["digit"] >= digit_h and hours["plain"] >= plain_h or decoded >= max_decode_h * 3600:
            break
        batch = []
        for r in rows_of(kind, path):
            if exclude_ids and r["id"] in exclude_ids:
                continue
            batch.append(r)
            if len(batch) < 64:
                continue
            decoded += _process(batch, model, lang, out, got, hours, audit, stats, langid, sf, ft_eval, quota)
            batch = []
            if hours["digit"] >= digit_h and hours["plain"] >= plain_h or decoded >= max_decode_h * 3600:
                break
        if batch:
            decoded += _process(batch, model, lang, out, got, hours, audit, stats, langid, sf, ft_eval, quota)
    audit.close()
    for k in got:
        with open(f"{out}/{k}.jsonl", "w") as f:
            for r in got[k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"lang": lang, "decoded_h": round(decoded / 3600, 2), "digit_h": round(hours["digit"], 2),
            "plain_h": round(hours["plain"], 2), "digit_rows": len(got["digit"]), **stats}


def _process(batch, model, lang, out, got, hours, audit, stats, langid, sf, ft_eval, quota):
    paths = []
    for r in batch:
        p = f"/tmp/b_{len(paths)}.wav"
        sf.write(p, r["x"], 16000, subtype="PCM_16")
        paths.append(p)
    hyps = [h["text"] for h in ft_eval.decode(model, paths, batch_size=32)]
    tot = 0.0
    for r, h in zip(batch, hyps):
        tot += r["dur"]
        stats["rows"] += 1
        cap = r["caption"]
        has_d = bool(re.search(r"\d", h))
        lid = langid.classify(h)[0] if h.strip() else "?"
        cer = _cer(cap, h)
        dec = {"id": r["id"], "src": r["src"], "cap": cap, "stock": h, "lid": lid, "cer": round(cer, 3)}
        kind = None
        if lid != lang:
            stats["lid_reject"] += 1
        elif has_d:
            cap_runs = [x for x in _runs(cap) if len(x) >= 2]
            if cer > 0.20:
                stats["cer_reject"] += 1
            elif any(x not in _runs(h) for x in cap_runs):
                stats["digit_mismatch"] += 1
            else:
                kind = "digit"
        elif not re.search(r"\d", cap) and cer <= 0.15:
            kind = "plain"
        else:
            stats["cer_reject"] += 1
        dec["kept"] = kind
        audit.write(json.dumps(dec, ensure_ascii=False) + "\n")
        if kind and hours[kind] < quota[kind]:
            wav = f"{out}/wav/{r['id']}.wav"
            sf.write(wav, r["x"], 16000, subtype="PCM_16")
            got[kind].append({"audio_filepath": wav, "duration": round(r["dur"], 3), "text": h, "lang": lang,
                              "src": r["src"], "id": r["id"]})
            hours[kind] += r["dur"] / 3600
    return tot
