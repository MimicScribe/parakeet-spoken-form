"""Real-speech candidates for the edge-case pack (edge_a): captions that hold an inventory phrase, as words or as one of the
digit shapes whose spoken form is fixed ("the 90s", "25 basis points", "24/7", "Q3"). A candidate becomes a row only after
the audio agrees (edge_a_ctc strict probes on the edge phrase, a second transcript); this module is text only.

  index        {normalised phrase: (construction, value key)} of the inventory (all variants of every value)
  convert      digit shapes -> spoken words in a caption (None when a digit or symbol is left)
  candidate    a caption -> {text, edge_spans, num_chars, spans, construction, values, side} or None
  Modal        scan YODAS-style caption shards for candidates, fetch the audio of the picked ones (new volume paths)

    modal run edge_a_real.py::scan --lang en --files files.json --n 150 --out hits_en.jsonl
    modal run edge_a_real.py::fetch --hits sel_en.jsonl --lang en --tag r1
"""
import json
import os
import random
import re
import unicodedata

import edge_a as E
import edge_a_words as W

REPO = "espnet/yodas-granary"
COLS = ["utt_id", "duration", "lang", "text", "original_audio_id", "original_audio_offset"]
DUR = (3.0, 24.0)

DECADE = {
    "en": {"20": "twenties", "30": "thirties", "40": "forties", "50": "fifties", "60": "sixties", "70": "seventies", "80": "eighties",
           "90": "nineties"},
    "es": {"20": "veinte", "30": "treinta", "40": "cuarenta", "50": "cincuenta", "60": "sesenta", "70": "setenta", "80": "ochenta",
           "90": "noventa"},
    "fr": {"10": "dix", "20": "vingt", "30": "trente", "40": "quarante", "50": "cinquante", "60": "soixante", "70": "soixante dix",
           "80": "quatre vingt", "90": "quatre vingt dix"},
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", s)).strip()


_INDEX: dict = {}


def index(lang: str) -> tuple:
    """({normalised phrase: (construction id, value key)}, compiled alternation longest first)."""
    if lang not in _INDEX:
        d: dict = {}
        for c in E.inventory(lang):
            for k in c.values:
                for seed in range(6):
                    p = c.make(k, random.Random(f"{k}|{seed}"))
                    d.setdefault(norm(p.label), (c.id, k))
        words = sorted(d, key=lambda x: -len(x))
        rx = re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(w).replace(r"\ ", r"[\s,]+") for w in words) + r")(?![\w'])", re.I)
        _INDEX[lang] = (d, rx)
    return _INDEX[lang]


# ------------------------------------------------------------------------------------------------ digit shapes
def _c(n, lang):
    return W.card(int(n), lang)


def convert(lang: str, text: str) -> str | None:
    """Digit shapes with a fixed spoken form -> words. None when any digit or symbol is left afterwards."""
    t = text
    sub = t
    if lang == "en":
        rules = [(r"\b(\d{1,3}) (?:basis points|bps|bp)\b", lambda m: f"{_c(m[1], 'en')} basis points"),
                 (r"\bQ([1-4])\b", lambda m: f"Q {_c(m[1], 'en')}"),
                 (r"\bthe (?:19)?([2-9]0)s\b", lambda m: f"the {DECADE['en'][m[1]]}"),
                 (r"\b(\d{2})-something\b", lambda m: f"{_c(m[1], 'en')} something"),
                 (r"\b(\d{1,3}) grand\b", lambda m: ("a grand" if m[1] == "1" else f"{_c(m[1], 'en')} grand")),
                 (r"\b(\d{1,3})[kK]\b(?!\w)", lambda m: f"{_c(m[1], 'en')} K"),
                 (r"\b24/7\b", lambda m: "twenty four seven"), (r"\b50[-/]50\b", lambda m: "fifty fifty"),
                 (r"\b(\d{1,3}) bucks\b", lambda m: f"{_c(m[1], 'en')} bucks")]
    elif lang == "es":
        rules = [(r"\b(\d{1,3}) puntos (básicos|base)\b", lambda m: f"{_c(m[1], 'es')} puntos {m[2]}"),
                 (r"\blos (años )?([2-9]0)\b", lambda m: f"los {m[1] or ''}{DECADE['es'][m[2]]}"),
                 (r"\b(\d{2,4}) y (pico|tantos|pocos)\b", lambda m: f"{_c(m[1], 'es')} y {m[2]}"),
                 (r"\b24/7\b", lambda m: "veinticuatro siete")]
    else:
        rules = [(r"\b(\d{1,3}) points de base\b", lambda m: f"{_c(m[1], 'fr')} points de base"),
                 (r"\bles années ([1-9]0)\b", lambda m: f"les années {DECADE['fr'][m[1]]}"),
                 (r"\b(\d{1,2}) (jours|heures) sur (\d{1,2})\b", lambda m: f"{_c(m[1], 'fr')} {m[2]} sur {_c(m[3], 'fr')}"),
                 (r"\b(\d{2,4}) et quelques\b", lambda m: f"{_c(m[1], 'fr')} et quelques")]
    for pat, fn in rules:
        sub = re.sub(pat, fn, sub)
    if re.search(r"[0-9%$€£#@+=<>/\\_]", sub):
        return None
    return sub


SPORT = {"en": re.compile(r"\b(?:score|scored|won|lost|beat|leading|led|trailing|halftime|half time|match|game|final|set|goal|goals|points|lead)\b", re.I),
         "es": re.compile(r"\b(?:partido|marcador|ganaba|ganaban|gol|goles|descanso|resultado|ganó|perdió|empate|victoria|derrota)\b", re.I),
         "fr": re.compile(r"\b(?:match|score|menait|menaient|mène|mi temps|mi-temps|but|buts|jeu|partie|gagné|perdu|victoire)\b", re.I)}


def candidate(lang: str, caption: str, split: dict | None = None) -> dict | None:
    """The first index phrase found in the (converted) caption as a planned-row skeleton, else None. `side`: A (may train),
    B (V_b value), Cb (held-out construction)."""
    d, rx = index(lang)
    text = convert(lang, caption.strip())
    if text is None or len(text.split()) < 4:
        return None
    ms = [m for m in rx.finditer(text)]
    if not ms:
        return None
    edge, nums, spans, cons, vals, sides = [], [], [], [], [], []
    last = -1
    for m in ms:
        if m.start() < last:
            continue
        cid, key = d[norm(m.group(0))]
        edge.append([m.start(), m.end()])
        nums.append([m.start(), m.end()])
        spans.append({"text": m.group(0), "kind": "edge", "reading": m.group(0), "canonical": m.group(0), "idx": 0, "n_cands": 1, "slot": "X"})
        cons.append(cid)
        vals.append(key)
        last = m.end()
        if split is not None:
            sides.append(E.side_of(split, cid, key))
    if any(c.endswith((".score", ".marcador")) for c in cons) and not SPORT[lang].search(text):
        return None            # "three, two, one, go" is a countdown, not a score
    side = "A" if all(s == "A" for s in sides) else next((s for s in sides if s != "A"), "?")
    if split is not None and side == "?":
        side = E.hash_side(lang, cons[0], vals[0])
    return {"text": text, "caption": caption, "edge_spans": edge, "num_chars": nums, "spans": spans, "constructions": cons, "values": vals,
            "construction": cons[0], "family": cons[0].split(".")[1], "edge": [cons[0].split(".")[1]], "side": side}


# ------------------------------------------------------------------------------------------------ Modal (scan + fetch)
try:
    import modal

    app = modal.App("parakeet-edge-a-real")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "numpy", "librosa")
             .apt_install("libsndfile1", "ffmpeg")
             .add_local_python_source("edge_a_real", "edge_a", "edge_a_content", "edge_a_frames", "edge_a_words", "ml_cands", "readings")
             .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_grammar_table.json"), "/root/ml_grammar_table.json")
             .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_a_split"), "/root/edge_a_split"))
    hf = modal.Secret.from_name("hf")
    vol = modal.Volume.from_name("parakeet-spoken-form")

    @app.function(image=image, secrets=[hf], cpu=1, memory=4096, timeout=3600, retries=2, max_containers=12)
    def scan_shard(path: str, lang: str) -> list:
        import time

        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        split = json.load(open(f"/root/edge_a_split/ec1split_{lang}.json", encoding="utf-8"))
        name = path.split("/")[1] + "_" + path.rsplit("/", 1)[-1][:-8]
        for attempt in range(8):
            try:
                local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/tmp/shard")
                try:
                    rows = pq.ParquetFile(local).read(columns=COLS).to_pylist()
                finally:
                    os.remove(local)
                break
            except Exception as e:  # noqa: BLE001
                if "429" not in repr(e) or attempt == 7:
                    raise
                time.sleep(min(300, 15 * 2 ** attempt))
        out = []
        for r in rows:
            if not DUR[0] <= (r.get("duration") or 0) <= DUR[1]:
                continue
            c = candidate(lang, r["text"] or "", split)
            if c:
                out.append(dict({k: r[k] for k in COLS}, shard=path, cand=c))
        return out + [{"_shard": name, "_rows": len(rows)}]

    @app.local_entrypoint()
    def scan(lang: str, files: str, n: int = 100, offset: int = 0, seed: int = 71, out: str = "hits.jsonl", part: str = "asr_only"):
        import collections
        import random as _r

        paths = sorted(p for p, _ in json.load(open(files)) if re.match(rf"data/{lang}\d{{3}}/{part}/.*\.parquet$", p))
        _r.Random(seed).shuffle(paths)
        paths = paths[offset:offset + n]
        tot, nrows, ok = collections.Counter(), 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for res in scan_shard.map(paths, kwargs={"lang": lang}, return_exceptions=True, order_outputs=False):
                if isinstance(res, Exception):
                    print("FAIL", repr(res)[:200])
                    continue
                ok += 1
                for h in res:
                    if "_rows" in h:
                        nrows += h["_rows"]
                        continue
                    tot[h["cand"]["construction"]] += 1
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(json.dumps({"lang": lang, "shards": ok, "rows": nrows, "hits": sum(tot.values()), "by_construction": dict(tot)}))

    @app.function(image=image, secrets=[hf], volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600, retries=3, max_containers=6)
    def fetch_shard(path: str, rows: list, lang: str, tag: str) -> list:
        import io

        import librosa
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download

        want = {r["utt_id"]: r for r in rows}
        out_dir = f"/vol/ml/edgeA/real/{tag}/{lang}/wav"
        os.makedirs(out_dir, exist_ok=True)
        local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/tmp/shard")
        pf = pq.ParquetFile(local)
        kept = []
        for rg in range(pf.metadata.num_row_groups):
            for x in pf.read_row_group(rg, columns=["utt_id", "audio"]).to_pylist():
                r = want.get(x["utt_id"])
                if r is None:
                    continue
                try:
                    a, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32")
                    a = a.mean(1) if a.ndim > 1 else a
                    if sr != 16000:
                        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
                    wav = f"{out_dir}/{x['utt_id']}.wav"
                    sf.write(wav, a.astype(np.float32), 16000, subtype="PCM_16")
                except Exception:  # noqa: BLE001
                    continue
                kept.append(dict(r, audio_filepath=wav, duration=round(len(a) / 16000, 3)))
        os.remove(local)
        vol.commit()
        return kept

    @app.local_entrypoint()
    def fetch(hits: str, lang: str, tag: str, out: str = ""):
        import collections

        rows = [json.loads(l) for l in open(hits) if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["shard"]].append(r)
        kept = []
        for res in fetch_shard.starmap([(p, rs, lang, tag) for p, rs in by.items()], return_exceptions=True):
            if isinstance(res, Exception):
                print("FAIL", repr(res)[:200])
                continue
            kept += res
        out = out or hits.replace(".jsonl", "_fetched.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"rows": len(kept), "shards": len(by), "hours": round(sum(r["duration"] for r in kept) / 3600, 2)}))
except ImportError:
    pass
