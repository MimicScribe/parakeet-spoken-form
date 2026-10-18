"""Open-licence voices for synthetic sets (2026-10-18, the symbol-class set ttsSYM1; engines chosen in the open-TTS
pilot). NEW PATHS ONLY: an existing output path is an error, nothing is overwritten.

refs:      clone-reference speakers. VoxPopuli (CC0) transcribed TRAIN split for de / nl / it / es / fr: a speaker is
           excluded if any of their segments is in the VoxPopuli dev / test split, on a dev / test plenary day, or in
           the project's held-out list (ml/p3v2/heldout_<lang>.txt), or listed in --exclude. Multilingual LibriSpeech
           Portuguese (CC BY 4.0, LibriVox audio) TRAIN split for pt (MLS dev / test speakers excluded). Up to
           --per-gender speakers per gender, one 7-13 s clip each, best crude SNR -> ml/<tag>/refs/<lang>/<spk>.wav +
           meta.json.
synth:     planned rows (ml_tts_symbol.py) -> 16 kHz mono clips at each row's "presynth" path. engine "qwen3":
           Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0) voice clone, batches of 6 rows of one reference, max_new_tokens =
           2 x characters; engine "chatterbox": Chatterbox Multilingual V3 (MIT). The row's speaking rate ("speed",
           0.9-1.25) is applied after synthesis with a pitch-keeping tempo change (sox tempo -s); neither engine has a
           rate control. Verification is the usual per-clip CTC check afterwards:
               modal run ml_tts.py --langs <lang> --out-tag <tag>c --rows-file "ml/<tag>v/{lang}/planned.jsonl"

    modal run ml_tts_open.py::run_refs --tag ttsSYM1r --langs de,es,fr,pt --per-gender 6 --exclude de:197643
    modal run ml_tts_open.py::run_synth --planned ml/ttsSYM1v/es/planned.jsonl --engine qwen3 --parts 4
"""
import os

import modal

vol = modal.Volume.from_name("parakeet-spoken-form", version=2)
VOL = "/vol"
app = modal.App("parakeet-tts-open")

refs_img = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
            .pip_install("pandas==2.2.3", "pyarrow==17.0.0", "huggingface_hub", "hf_transfer", "soundfile", "numpy<2",
                         "librosa==0.11.0")
            .env({"HF_HOME": "/root/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))
CB_SHA = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
cb_img = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1", "git", "sox")
          .pip_install(f"chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@{CB_SHA}", "soundfile")
          .env({"HF_HOME": "/root/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"}))
qw_img = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1", "sox")
          .pip_install("torch==2.6.0", "torchaudio==2.6.0", "qwen-tts==0.1.1", "soundfile")
          .env({"HF_HOME": "/root/hf"}))

VOX_LANGS = ["de", "nl", "it", "es", "fr"]
QW_LANG = {"de": "German", "it": "Italian", "es": "Spanish", "fr": "French", "pt": "Portuguese"}
QW_BATCH = 6


def _snr_db(x, sr):
    import numpy as np
    n = int(0.025 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    return float(np.percentile(e, 95) - np.percentile(e, 10))


def _f0(x, sr):
    import librosa
    import numpy as np
    f, v, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr)
    f = f[~np.isnan(f)]
    return float(np.median(f)) if len(f) else 0.0


@app.function(image=refs_img, volumes={VOL: vol}, cpu=4, memory=16384, timeout=3600)
def refs(tag: str, lang: str, per_gender: int = 6, shards: int = 2, exclude: str = "", min_snr: float = 30.0):
    import collections
    import csv
    import gzip
    import io
    import json
    import random
    import urllib.request

    import pandas as pd
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    vol.reload()
    out = f"{VOL}/ml/{tag}/refs/{lang}"
    assert not os.path.exists(out), out
    rng = random.Random(1018)
    excl_spk = {x for x in exclude.split(",") if x}
    if lang in VOX_LANGS:
        p = f"/tmp/asr_{lang}.tsv.gz"
        urllib.request.urlretrieve(f"https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_{lang}.tsv.gz", p)
        with gzip.open(p, "rt") as f:
            asr = list(csv.DictReader(f, delimiter="|"))
        held = {l.strip() for l in open(f"{VOL}/ml/p3v2/heldout_{lang}.txt") if l.strip() and not l.startswith("#")}
        bad_days = {r["session_id"][:8] for r in asr if r["split"] in ("dev", "test")}
        bad_spk = {r["speaker_id"] for r in asr if r["split"] in ("dev", "test") or (r["session_id"] + "-" + r["id_"]) in held}
        held_days = {h[:8] for h in held if h[:8].isdigit()}
        files = sorted(f for f in HfApi().list_repo_files("facebook/voxpopuli", repo_type="dataset")
                       if f.startswith(f"{lang}/train-"))[:shards]
        rows = []
        for fn in files:
            df = pd.read_parquet(hf_hub_download("facebook/voxpopuli", fn, repo_type="dataset"))
            for r in df.itertuples():
                if (r.speaker_id in bad_spk or r.speaker_id in excl_spk or r.audio_id[:8] in bad_days
                        or r.audio_id[:8] in held_days or not r.is_gold_transcript or r.gender not in ("male", "female")):
                    continue
                x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
                d = len(x) / sr
                if 7 <= d <= 13:
                    rows.append({"gender": r.gender, "spk": r.speaker_id, "id": r.audio_id, "x": x, "sr": sr, "dur": d,
                                 "text": r.raw_text, "snr": _snr_db(x, sr), "accent": r.accent})
        src, lic = "VoxPopuli transcribed train split (facebook/voxpopuli)", "CC0-1.0"
        excl = {"bad_spk": len(bad_spk), "bad_days": len(bad_days), "held_ids": len(held), "shards": files,
                "excluded_listed": sorted(excl_spk)}
    else:  # pt: MLS Portuguese
        files = sorted(f for f in HfApi().list_repo_files("facebook/multilingual_librispeech", repo_type="dataset")
                       if f.startswith("portuguese/"))
        bad_spk = set()
        for sp in ("dev", "test"):
            f = [x for x in files if x.startswith(f"portuguese/{sp}-")][0]
            bad_spk |= set(pd.read_parquet(hf_hub_download("facebook/multilingual_librispeech", f, repo_type="dataset"),
                                           columns=["speaker_id"])["speaker_id"].astype(str))
        trs = [x for x in files if x.startswith("portuguese/train-")][:shards]
        rows, per = [], collections.Counter()
        for tr in trs:
            df = pd.read_parquet(hf_hub_download("facebook/multilingual_librispeech", tr, repo_type="dataset"))
            for r in df.sample(frac=1, random_state=1018).itertuples():
                spk = str(r.speaker_id)
                if spk in bad_spk or spk in excl_spk or per[spk] >= 6:
                    continue
                x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
                d = len(x) / sr
                if not 7 <= d <= 13:
                    continue
                per[spk] += 1
                rows.append({"gender": None, "spk": spk, "id": r.id, "x": x, "sr": sr, "dur": d, "text": r.transcript,
                             "snr": _snr_db(x, sr), "accent": None})
        f0s = {}
        for spk in {r["spk"] for r in rows}:
            best = max((r for r in rows if r["spk"] == spk), key=lambda r: r["snr"])
            f0s[spk] = _f0(best["x"], best["sr"])
        for r in rows:
            r["f0"] = f0s[r["spk"]]
            r["gender"] = "female" if f0s[r["spk"]] >= 165 else ("male" if 0 < f0s[r["spk"]] < 145 else None)
        rows = [r for r in rows if r["gender"]]
        src, lic = "Multilingual LibriSpeech Portuguese train split (facebook/multilingual_librispeech; LibriVox audio)", "CC-BY-4.0"
        excl = {"bad_spk": len(bad_spk), "shards": trs, "excluded_listed": sorted(excl_spk)}
    best = {}
    for r in rows:  # best clip per speaker by the crude SNR
        if r["spk"] not in best or r["snr"] > best[r["spk"]]["snr"]:
            best[r["spk"]] = r
    picked = []
    for g in ("female", "male"):
        pool = sorted((r for r in best.values() if r["gender"] == g and r["snr"] >= min_snr), key=lambda r: -r["snr"])
        pool = pool[:max(2 * per_gender, 12)]
        rng.shuffle(pool)
        picked += pool[:per_gender]
    os.makedirs(out)
    meta = []
    for r in picked:
        p = f"{out}/{r['spk']}.wav"
        sf.write(p, r["x"], r["sr"], subtype="PCM_16")
        meta.append({"lang": lang, "spk": r["spk"], "gender": r["gender"], "clip_id": r["id"], "dur": round(r["dur"], 2),
                     "snr_db": round(r["snr"], 1), "text": r["text"], "accent": r.get("accent"), "f0": r.get("f0"),
                     "wav": p, "source": src, "license": lic})
    with open(f"{out}/meta.json", "w") as f:
        json.dump({"refs": meta, "exclusions": excl, "n_candidate_speakers": len(best)}, f, ensure_ascii=False, indent=1)
    vol.commit()
    return {"lang": lang, "picked": [(m["spk"], m["gender"], m["snr_db"]) for m in meta], "cands": len(best)}


def _rows(path):
    import json
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _to16k(src, dst, speed):
    """dst (16 kHz mono PCM16) = src at speaking rate `speed` (pitch kept)."""
    import subprocess
    cmd = ["sox", src, "-r", "16000", "-c", "1", "-b", "16", dst]
    if abs(speed - 1.0) > 1e-3:
        cmd += ["tempo", "-s", f"{speed:.3f}"]
    subprocess.run(cmd, check=True, capture_output=True)


def _log_path(planned, engine, part):
    """A new log per attempt (a pre-empted container restarts its input; clips already written are skipped)."""
    base = f"{os.path.dirname(f'{VOL}/{planned}')}/synth_log_{engine}_{part}"
    k = 0
    while os.path.exists(f"{base}{'' if k == 0 else f'_r{k}'}.jsonl"):
        k += 1
    return f"{base}{'' if k == 0 else f'_r{k}'}.jsonl"


def _done(path, rec, log):
    import json
    log.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.function(image=qw_img, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, memory=32768, max_containers=8)
def synth_qwen(planned: str, part: int, parts: int):
    import collections
    import json
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    vol.reload()
    t0 = time.time()
    rows = [r for k, r in enumerate(_rows(f"{VOL}/{planned}")) if k % parts == part and r["engine_src"] == "qwen3"]
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["ref_wav"]].append(r)
    logp = _log_path(planned, "qwen3", part)
    log = open(logp, "w")
    n_ok, gen_s, audio_s, chars = 0, 0.0, 0.0, 0
    for refwav, js in by.items():
        prompt = m.create_voice_clone_prompt(ref_audio=refwav, ref_text=js[0]["ref_text"], x_vector_only_mode=False)
        for k in range(0, len(js), QW_BATCH):
            b = [r for r in js[k:k + QW_BATCH] if not os.path.exists(r["presynth"])]
            if not b:
                continue
            t1 = time.time()
            torch.manual_seed(k + len(refwav))
            cap = max(160, int(2.0 * max(len(r["text"]) for r in b)))
            try:
                wavs, sr = m.generate_voice_clone(text=[r["text"] for r in b], language=[QW_LANG[r["lang"]] for r in b],
                                                  voice_clone_prompt=prompt, max_new_tokens=cap)
            except Exception as e:  # noqa: BLE001  (one bad batch must not end the part)
                for r in b:
                    _done(r["presynth"], {"id": r["id"], "status": "error", "err": repr(e)[:200]}, log)
                continue
            torch.cuda.synchronize()
            gen_s += time.time() - t1
            for r, w in zip(b, wavs):
                w = np.asarray(w, dtype=np.float32)
                dur = len(w) / sr
                tmp = f"/tmp/{r['id']}.wav"
                sf.write(tmp, w, sr, subtype="PCM_16")
                os.makedirs(os.path.dirname(r["presynth"]), exist_ok=True)
                _to16k(tmp, r["presynth"], r["speed"])
                os.remove(tmp)
                cap_hit = dur >= 0.95 * cap / 12
                audio_s += dur
                chars += len(r["text"])
                n_ok += 1
                _done(r["presynth"], {"id": r["id"], "status": "ok", "audio_s": round(dur, 2), "cap_hit": cap_hit,
                                      "chars": len(r["text"])}, log)
        log.flush()
        vol.commit()
    log.close()
    vol.commit()
    return {"engine": "qwen3", "part": part, "n": n_ok, "of": len(rows), "gen_s": round(gen_s), "audio_s": round(audio_s),
            "chars": chars, "wall_s": round(time.time() - t0)}


@app.function(image=cb_img, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, memory=32768, max_containers=8)
def synth_cb(planned: str, part: int, parts: int):
    import json
    import time

    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    vol.reload()
    t0 = time.time()
    rows = [r for k, r in enumerate(_rows(f"{VOL}/{planned}")) if k % parts == part and r["engine_src"] == "chatterbox"]
    rows.sort(key=lambda r: r["ref_wav"])
    m = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    logp = _log_path(planned, "chatterbox", part)
    log = open(logp, "w")
    cur, n_ok, gen_s, audio_s, chars = None, 0, 0.0, 0.0, 0
    for i, r in enumerate(rows):
        if os.path.exists(r["presynth"]):
            continue
        t1 = time.time()
        try:
            torch.manual_seed(int(r["id"][-8:], 16) % 2 ** 31)
            if cur != r["ref_wav"]:
                m.prepare_conditionals(r["ref_wav"], exaggeration=0.5)
                cur = r["ref_wav"]
            w = m.generate(r["text"], language_id=r["lang"]).squeeze(0).numpy()
        except Exception as e:  # noqa: BLE001
            _done(r["presynth"], {"id": r["id"], "status": "error", "err": repr(e)[:200]}, log)
            continue
        torch.cuda.synchronize()
        gen_s += time.time() - t1
        tmp = f"/tmp/{r['id']}.wav"
        sf.write(tmp, w, m.sr, subtype="PCM_16")
        os.makedirs(os.path.dirname(r["presynth"]), exist_ok=True)
        _to16k(tmp, r["presynth"], r["speed"])
        os.remove(tmp)
        audio_s += len(w) / m.sr
        chars += len(r["text"])
        n_ok += 1
        _done(r["presynth"], {"id": r["id"], "status": "ok", "audio_s": round(len(w) / m.sr, 2), "chars": len(r["text"])}, log)
        if i % 50 == 0:
            log.flush()
            vol.commit()
    log.close()
    vol.commit()
    return {"engine": "chatterbox", "part": part, "n": n_ok, "of": len(rows), "gen_s": round(gen_s),
            "audio_s": round(audio_s), "chars": chars, "wall_s": round(time.time() - t0)}


try:  # English: Kokoro synthesis + the English CTC reader (en_win / label_gpu, as win15 English windows)
    import ml_tts as _T

    en_img = _T.image.add_local_python_source("en_win", "label_gpu", "labeller", "readings", "census_common",
                                              "ml_slav", "ml_balance", "compound_split")
except Exception:  # noqa: BLE001  (local tests without the pipeline image)
    en_img = None


@app.function(image=en_img, volumes={VOL: vol}, gpu="L4", cpu=4, memory=16384, timeout=4 * 3600, max_containers=4)
def en_verify(planned: str, out_tag: str, part: int, parts: int):
    """English rows (ml_tts_symbol plan-en, engine kokoro): synthesise (readings' tts form), level + pad like
    ml_tts.tts_lang, write the clip to its presynth path, and check it with the English CTC reader: CER <= ml_tts.CER_MAX
    and every reading fits the greedy text (ml_decide.fits; fuzzy for long readings, the greedy kept for the number
    check in ml_tts_symbol.finalize). -> ml/<out_tag>/en/rows_<part>.jsonl (kept) + stats_<part>.json."""
    import json
    import random as _r

    import numpy as np
    import soundfile as sf

    import en_win as E
    import ml_decide as D
    import ml_p3_core as C
    import ml_tts as T
    from label_gpu import W2V

    class EnCTC(D.LangCTC):
        def __init__(self):
            super().__init__("en", "cuda", model=W2V)

        def ids(self, text):
            return E.en_ids(text, self.vocab, self.delim)

        def greedy(self, lp):
            return E.greedy_norm(super().greedy(lp))

    vol.reload()
    out = f"{VOL}/ml/{out_tag}/en"
    os.makedirs(out, exist_ok=True)
    rp = f"{out}/rows_{part}.jsonl"
    assert not os.path.exists(rp), rp
    rows = [r for k, r in enumerate(_rows(f"{VOL}/{planned}")) if k % parts == part]
    ctc, syn = EnCTC(), T._Synth()
    rng = _r.Random(f"en_verify|{part}")
    st, kept = {}, []
    bump = lambda k: st.__setitem__(k, st.get(k, 0) + 1)  # noqa: E731
    for r in rows:
        try:
            x = syn(dict(r, text=r["synth_text"]))
        except Exception:  # noqa: BLE001
            x = None
        if x is None:
            bump("synth_error")
            continue
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (-3 / 20) * 10 ** (rng.uniform(-12, 0) / 20)
        pad = lambda: np.zeros(int(16000 * rng.uniform(0.2, 0.6)), dtype=np.float32)  # noqa: E731
        x = np.concatenate([pad(), x, pad()]).astype(np.float32)
        lp = ctc.lp(x)
        g = ctc.greedy(lp)
        lab = E.greedy_norm(r["text"])
        cer = C.cer(lab, g)
        if cer > T.CER_MAX:
            bump("drop_cer")
            continue
        fit = "exact"
        for a, b in r["num_chars"]:
            rd = E.greedy_norm(r["text"][a:b])
            if D.fits(rd, g):
                continue
            if D.fits(rd, g, fuzzy=True):
                fit = "fuzzy"
                continue
            fit = None
            break
        if fit is None:
            bump("drop_reading_not_heard")
            continue
        wt = ctc.word_times(lp, r["text"])
        if any(t is None for t in wt):
            bump("drop_align")
            continue
        os.makedirs(os.path.dirname(r["presynth"]), exist_ok=True)
        sf.write(r["presynth"], x, 16000, subtype="PCM_16")
        bump("kept")
        kept.append({"audio_filepath": r["presynth"], "duration": round(len(x) / 16000, 3), "text": r["text"],
                     "lang": "en", "src": "tts", "id": r["id"], "origin": f"tts/{r['voice']}", "kinds": r["kinds"],
                     "template": r["template"], "cls": r["cls"], "ckey": r["ckey"], "spans": r["spans"],
                     "num_chars": r["num_chars"], "gclasses": r["gclasses"],
                     "word_times": [[round(a, 3), round(b, 3)] for a, b in wt], "cer_ctc": round(cer, 3), "fit": fit,
                     "greedy": g, "split": r["split"], "voice": r["voice"], "voice_license": r["voice_license"],
                     "speed": r["speed"], "engine": "kokoro", "engine_src": "kokoro"})
    with open(rp, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    json.dump(dict(st, planned=len(rows)), open(f"{out}/stats_{part}.json", "w"), indent=1)
    vol.commit()
    return dict(st, planned=len(rows), part=part)


@app.local_entrypoint()
def run_en(planned: str, out_tag: str = "ttsSYM1c", parts: int = 3):
    for res in en_verify.starmap([(planned, out_tag, p, parts) for p in range(parts)], return_exceptions=True):
        print(res)


@app.local_entrypoint()
def run_refs(tag: str, langs: str = "de,es,fr,pt", per_gender: int = 6, exclude: str = "", shards: int = 2):
    ex = {}
    for x in exclude.split(","):
        if ":" in x:
            l, s = x.split(":", 1)
            ex.setdefault(l, []).append(s)
    for res in refs.starmap([(tag, l, per_gender, shards, ",".join(ex.get(l, []))) for l in langs.split(",")]):
        print(res)


@app.local_entrypoint()
def run_synth(planned: str, engine: str, parts: int = 4):
    fn = synth_qwen if engine == "qwen3" else synth_cb
    tot = {}
    for res in fn.starmap([(planned, p, parts) for p in range(parts)], return_exceptions=True):
        if isinstance(res, BaseException):
            print("PART FAILED", repr(res)[:300])
            continue
        print(res)
        for k in ("n", "of", "gen_s", "audio_s", "chars", "wall_s"):
            tot[k] = tot.get(k, 0) + res[k]
    print("TOTAL", tot, "L4 container-h", round(tot.get("wall_s", 0) / 3600, 3))
