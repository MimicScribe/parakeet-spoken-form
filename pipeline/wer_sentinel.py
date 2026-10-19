"""General-speech word-error sentinel: number-free FLEURS dev utterances in 13 languages.

A spoken-form fine-tune must not damage ordinary words. This bed holds ~200 number-free read utterances per language
from the public FLEURS DEV split (never the test split that public numbers come from). Scoring is eval/quality/wer_sentinel.py.

    modal run wer_sentinel.py::build                                  # CPU, once -> eval/wer_sentinel_v1/ (refuses to overwrite)
    modal run wer_sentinel.py::decode --runs "ultra,w3soup_a0.5@0"    # L4 -> hyps/<run>/wer_sentinel_v1.jsonl (keeps existing)

Selection per language: one recording per sentence id, caption without a digit, symbol or number word (ml_balance lexicon
+ the multilingual lexicon), sentence not present (normalised text) in any training manifest of the spoken-form runs,
shuffled with a fixed seed, first N kept. Volume writes go to NEW paths only.
"""

import hashlib
import json
import os
import random
import re
import sys
import unicodedata

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_app import VOL, _load_runs, _stock_model, nemo_image, vol  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "wer_sentinel_v1"
LANGS = ("en", "de", "es", "fr", "it", "nl", "pt", "ru", "uk", "pl", "ro", "cs", "hu")
FLEURS_CODE = {"en": "en_us", "de": "de_de", "es": "es_419", "fr": "fr_fr", "it": "it_it", "nl": "nl_nl", "pt": "pt_br",
               "ru": "ru_ru", "uk": "uk_ua", "pl": "pl_pl", "ro": "ro_ro", "cs": "cs_cz", "hu": "hu_hu"}
EN_EXTRA = set("one first second oh half halves dozen hundreds thousands millions billions twice thrice".split())
# training sources whose text a sentinel utterance must not appear in (volume paths; globs are expanded in the job)
TRAIN_GLOBS = ("exp/spike_*/train_manifest.jsonl", "mix4final9b/*.jsonl", "mix4final8/*.jsonl", "mix3/*.jsonl",
               "mix3b/*.jsonl", "ml/anchor/*/*/*.jsonl", "d2/*/digit.jsonl", "d2/*/plain.jsonl",
               "replay/fleurs_train_multi.jsonl", "replay/lspc_train.jsonl")

app = modal.App("wer-sentinel")
build_image = (modal.Image.debian_slim(python_version="3.11")
               .apt_install("ffmpeg", "libsndfile1")
               .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa", "numpy")
               .add_local_python_source("ml_balance", "census_common", "ml_p3_core", "ml_cands", "ml_slav", "harvest", "common", "modal_app")
               .add_local_file(os.path.join(HERE, "ml_grammar_table.json"), "/root/ml_grammar_table.json")
               .add_local_file(os.path.join(HERE, "ml_multi_lex.json"), "/root/ml_multi_lex.json"))
decode_image = nemo_image.add_local_python_source("modal_app")


def text_key(text: str) -> str:
    """Case-, punctuation- and accent-insensitive key of a sentence (letters and digits only)."""
    s = unicodedata.normalize("NFKD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c)[0] in "LN")
    return hashlib.sha1(s.encode("utf8")).hexdigest()[:16] if s else ""


def _fold(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if unicodedata.category(c) != "Mn")


def multi_lex_words(lang: str) -> set:
    p = "/root/ml_multi_lex.json" if os.path.exists("/root/ml_multi_lex.json") else os.path.join(HERE, "ml_multi_lex.json")
    lex = json.load(open(p)).get(lang) or {}
    out = set()
    for k in ("cardinal", "ordinal"):
        for forms in lex.get(k, {}).values():
            for f in forms:
                out |= {_fold(t) for t in re.split(r"[\s\-]+", f) if len(t) >= 3}
    return out


def number_free_caption(text: str, lang: str, lexicon: set, multi: set) -> bool:
    """No digit or symbol, no number word of the language (look-alike articles allowed by ml_balance.number_free)."""
    import ml_balance as B

    if re.search(r"\d", text) or not B.number_free(text, lang, lexicon):
        return False
    toks = {_fold(t) for t in re.findall(r"[^\W\d_]+", text)}
    if toks & multi:
        return False
    if lang == "en":
        from common import NUMBER_WORDS

        if NUMBER_WORDS.search(text) or toks & EN_EXTRA:
            return False
    return True


@app.function(image=build_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=3600)
def train_index() -> dict:
    """Hashes of the normalised text of every row in the training manifests -> eval/wer_sentinel_v1/train_text_keys.json."""
    import glob

    vol.reload()
    out = f"{VOL}/eval/{TAG}"
    assert not os.path.exists(f"{out}/train_text_keys.json"), "exists: new tags only"
    keys, files = set(), []
    for g in TRAIN_GLOBS:
        for p in sorted(glob.glob(f"{VOL}/{g}")):
            n = 0
            for line in open(p, encoding="utf8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                for f in ("text", "caption", "ref"):
                    if isinstance(r.get(f), str):
                        k = text_key(r[f])
                        if k:
                            keys.add(k)
                n += 1
            files.append((p[len(VOL) + 1:], n))
    os.makedirs(out, exist_ok=True)
    json.dump(sorted(keys), open(f"{out}/train_text_keys.json", "w"))
    vol.commit()
    return {"files": len(files), "keys": len(keys), "rows": sum(n for _, n in files)}


@app.function(image=build_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=2 * 3600, max_containers=13)
def build_lang(lang: str, n: int = 200, seed: int = 1) -> dict:
    """FLEURS dev of `lang`: number-free, untrained, one recording per sentence -> n rows + wavs under eval/wer_sentinel_v1."""
    import io
    import tarfile

    import librosa
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import ml_balance as B

    vol.reload()
    out = f"{VOL}/eval/{TAG}"
    assert not os.path.exists(f"{out}/{lang}.jsonl"), f"{out}/{lang}.jsonl exists: new tags only"
    trained = set(json.load(open(f"{out}/train_text_keys.json")))
    lexicon, multi = B.number_lexicon(lang), multi_lex_words(lang)
    code = FLEURS_CODE[lang]
    tsv = open(hf_hub_download("google/fleurs", f"data/{code}/dev.tsv", repo_type="dataset", local_dir="/tmp/fl"),
               encoding="utf8").read()
    recs = [c for c in (l.rstrip("\r").split("\t") for l in tsv.split("\n")) if len(c) >= 7 and c[0].isdigit()]
    st = {"lang": lang, "recordings": len(recs)}
    free = [c for c in recs if number_free_caption(c[2].strip(), lang, lexicon, multi)]
    st["number_free"] = len(free)
    fresh = [c for c in free if text_key(c[2]) not in trained]
    st["in_training_text"] = len(free) - len(fresh)
    rng = random.Random(f"{TAG}:{seed}:{lang}")
    rng.shuffle(fresh)
    seen, pick = set(), []
    for c in fresh:  # one recording per sentence id, then (if short) second recordings
        if c[0] not in seen:
            seen.add(c[0])
            pick.append(c)
    st["sentences"] = len(pick)
    pick = pick[:n]
    want = {c[1]: c for c in pick}
    tar = hf_hub_download("google/fleurs", f"data/{code}/audio/dev.tar.gz", repo_type="dataset", local_dir="/tmp/fl")
    os.makedirs(f"{out}/wav/{lang}", exist_ok=True)
    rows = []
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not m.isfile() or name not in want:
                continue
            c = want[name]
            x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            stem = os.path.splitext(name)[0]
            wav = f"{out}/wav/{lang}/{stem}.wav"
            sf.write(wav, x, 16000, subtype="PCM_16")
            rows.append({"audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": c[2].strip(), "lang": lang,
                         "id": f"{lang}_{c[0]}_{stem}", "gender": c[6]})
    rows.sort(key=lambda r: r["id"])
    with open(f"{out}/{lang}.jsonl", "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    st.update(written=len(rows), words=sum(len(r["text"].split()) for r in rows),
              h=round(sum(r["duration"] for r in rows) / 3600, 3))
    return st


@app.local_entrypoint()
def build(n: int = 200, langs: str = ",".join(LANGS)):
    print(json.dumps(train_index.remote()))
    res = list(build_lang.starmap([(l, n) for l in langs.split(",")]))
    for r in res:
        print(json.dumps(r))
    merge_manifests.remote(langs)


@app.function(image=build_image, volumes={VOL: vol}, cpu=1, memory=4096, timeout=1800)
def merge_manifests(langs: str = ",".join(LANGS)):
    """eval/wer_sentinel_v1/all.jsonl = the per-language manifests in order (new path only)."""
    vol.reload()
    out = f"{VOL}/eval/{TAG}/all.jsonl"
    assert not os.path.exists(out), "exists"
    n = 0
    with open(out, "w", encoding="utf8") as f:
        for l in langs.split(","):
            for line in open(f"{VOL}/eval/{TAG}/{l}.jsonl", encoding="utf8"):
                f.write(line)
                n += 1
    vol.commit()
    print(n, "rows")


@app.function(image=decode_image, volumes={VOL: vol}, gpu="L4", timeout=2 * 3600)
def decode(runs: str, manifest: str = f"eval/{TAG}/all.jsonl", name: str = TAG):
    """Decode the sentinel with each checkpoint in `runs` ("ultra", "<run>@<step>", "<path>.pt"; unfreeze_top
    checkpoints re-encode, as decoding always runs the whole model). -> hyps/<run>/<name>.jsonl, kept if it exists."""
    import ft_eval

    vol.reload()
    m, stock = _stock_model()
    for run in runs.split(","):
        if os.path.exists(f"{VOL}/hyps/{run}/{name}.jsonl"):
            print(f"KEEP hyps/{run}/{name}.jsonl (exists; new paths only)")
            continue
        n_enc = _load_runs(m, stock, run)
        ft_eval.transcribe_loaded(VOL, m, run, manifest, name)
        print(f"{run}: encoder tensors {n_enc}")
        vol.commit()
