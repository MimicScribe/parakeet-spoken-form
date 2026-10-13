"""Stage 0 of the real-speech number retrain: caption-only scan of espnet/yodas-granary.

Reads ONLY the label columns of each parquet shard (column projection; the audio column is never fetched), flags rows
whose caption carries a number (digits, English number words, number symbols) and writes them, with per-shard stats, to
/vol/s0/scan/<lang>/<shard>.jsonl. A 2% sample of number-free rows is kept too (filler-loss audit, plain replay).
"""

import json
import os
import random
import re

from common import NUMBER_WORDS

REPO = "espnet/yodas-granary"
DIGIT = re.compile(r"\d")
SYMBOL = re.compile(r"[$%€£@/]")
ROMAN = re.compile(r"\b(?:II|III|IV|VI|VII|VIII|IX|XI|XII|XIV|XV|XX|LVII)\b")
COLS = ["utt_id", "duration", "lang", "text", "original_audio_id", "original_audio_offset"]


def scan_shard(vol: str, path: str, lang: str, keep_plain: float = 0.02) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_dir = f"{vol}/s0/scan/{lang}"
    os.makedirs(out_dir, exist_ok=True)
    name = path.split("/")[1] + "_" + os.path.basename(path)[:-8]
    out = f"{out_dir}/{name}.jsonl"
    if os.path.exists(out + ".stats"):
        return json.load(open(out + ".stats"))
    fs = HfFileSystem()
    tb = pq.ParquetFile(fs.open(f"datasets/{REPO}/{path}", block_size=1 << 20)).read(columns=COLS).to_pylist()
    rnd = random.Random(name)
    st = {"shard": path, "lang": lang, "rows": len(tb), "h": 0.0, "digit_rows": 0, "digit_h": 0.0,
          "word_rows": 0, "word_h": 0.0, "symbol_rows": 0, "roman_rows": 0, "videos": 0}
    vids = set()
    with open(out, "w") as f:
        for r in tb:
            d, t = r["duration"], r["text"] or ""
            st["h"] += d / 3600
            vids.add(r["original_audio_id"])
            has_d, has_w = bool(DIGIT.search(t)), bool(NUMBER_WORDS.search(t)) if lang == "en" else False
            st["digit_rows"] += has_d
            st["digit_h"] += d / 3600 if has_d else 0
            st["word_rows"] += has_w and not has_d
            st["word_h"] += d / 3600 if has_w and not has_d else 0
            st["symbol_rows"] += bool(SYMBOL.search(t))
            st["roman_rows"] += bool(ROMAN.search(t))
            plain = not (has_d or has_w)
            if plain and rnd.random() >= keep_plain:
                continue
            r["kind"] = "digit" if has_d else ("word" if has_w else "plain")
            r["shard"] = path
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["videos"] = len(vids)
    json.dump(st, open(out + ".stats", "w"))
    return st


def pick_shards(langs: list[str], en_shards: int, big_shards: int, seed: int = 1) -> list[tuple[str, str]]:
    """(path, lang) for asr_only shards: `en_shards` English spread over every en dir, up to `big_shards` per other
    language (all of them for the small ones)."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.startswith("data/") and "/asr_only/" in f and f.endswith(".parquet")]
    rnd = random.Random(seed)
    out = []
    for lang in langs:
        mine = sorted(f for f in files if f.split("/")[1][:2] == lang)
        n = en_shards if lang == "en" else big_shards
        if len(mine) > n:
            mine = sorted(rnd.sample(mine, n))
        out += [(f, lang) for f in mine]
    return out
