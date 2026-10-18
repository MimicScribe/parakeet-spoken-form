"""English number-class caption scan (2026-10-18): a TEXT-ONLY pass over YODAS-Granary English captions that ranks
each caption by how thin the classes in it are in the English real-speech parts, then the same audio / label path as
en_hits.py (fetch -> s3_label -> s3_veto / s3_recrop / s3_decode_crops -> filter3). Label rules, CTC verification and
the held-out exclusions are not touched; only WHICH rows are fetched is new.

Classes: forward years 2026-2060, phone numbers, clock times, codes (letters + digits), money, decimals, percent,
big numbers with a magnitude word, dates, ordinals; a UK / Australian / Indian money or unit word adds a small bonus
(a proxy for non-US channels). Rows the labeller would drop (unsupported symbols, leftover written forms) never hit.

    modal run en_hits_num.py::scan --n 60 --seed 51 --out hits.jsonl --exclude-shards shards.txt
    python3 en_hits_num.py pick hits.jsonl sel.jsonl 1500 --skip-videos videos.txt --video-counts counts.json
    modal run en_hits.py::fetch --hits sel.jsonl --tag <tag> --per-manifest 800   # unchanged
"""
import json
import re
import sys

import labeller as L

VOL = "/vol"
REPO = "espnet/yodas-granary"
COLS = ["utt_id", "duration", "lang", "text", "original_audio_id", "original_audio_offset"]
MIN_SCORE = 1.0
MAX_SCORE = 8.0
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
         r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE = re.compile(r"\b" + MONTH + r"\.? \d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}(?:st|nd|rd|th)? (?:of )?" + MONTH + r"\b")
ACCENT = re.compile(r"£|\bpence\b|\bquid\b|\blakhs?\b|\bcrores?\b|\brupees?\b|\bINR\b|\bAUD\b|\bNZD\b|\bnaira\b|\brand\b",
                    re.I)
# weights = how thin the class is in the English real parts (2026-10-17 census: phone 74, clock 322, code 272, forward
# years 459 with most values under 10 spans, scaled money / numbers 973, decimals 1,103, money 1,302, ordinals 1,349)
W = {"phone": 8.0, "clock": 4.0, "code": 3.0, "scaled": 2.0, "dec": 1.0, "money": 1.0, "date": 1.5, "pct": 0.5,
     "ord": 0.5, "big": 0.5, "range": 0.3, "unit": 0.3}
COMMON_FWD = (2030, 2035, 2040, 2050)  # already >= 40 spans each


def classes_of(text: str) -> dict:
    """{class: count} of the thin-class shapes in a caption, via the labeller's own spans; {} when the labeller would
    drop the row (unsupported / bad token / leftover written form) or the caption has no digit."""
    if not re.search(r"\d", text or ""):
        return {}
    sp = L.spans(text)
    if L.row_ok(text, sp):
        return {}
    out: dict = {}

    def add(k, n=1):
        out[k] = out.get(k, 0) + n

    for s in sp:
        k = s.kind
        if k == "year":
            v = int(re.sub(r"\D", "", s.text)[:4] or 0)
            if 2026 <= v <= 2060:
                add("fwd_year" if v not in COMMON_FWD else "fwd_year_common")
        elif k in ("phone", "clock", "code", "scaled", "ordinal", "range", "unit"):
            add({"ordinal": "ord"}.get(k, k))
        elif k == "money":
            add("money")
            if "." in s.text or "million" in s.text.lower():
                add("scaled" if re.search(r"million|billion|trillion|thousand", s.text, re.I) else "dec")
        elif k == "percent":
            add("pct")
        elif k == "decimal":
            add("dec")
        elif k == "int" and re.search(r"\d,\d{3}", s.text):
            add("big")
    if DATE.search(text):
        add("date")
    if out and ACCENT.search(text):
        add("accent")
    return out


def score_of(cls: dict) -> float:
    s = 0.0
    for k, n in cls.items():
        w = {"fwd_year": 3.0, "fwd_year_common": 1.0, "accent": 1.5}.get(k, W.get(k, 0.0))
        s += w * (1.0 + 0.25 * (min(n, 3) - 1))
    return round(min(s, MAX_SCORE), 2)


def scan_rows(rows: list, shard: str) -> list:
    out = []
    for r in rows:
        if not 1.0 <= (r.get("duration") or 0) <= 30.0:
            continue
        cls = classes_of(r["text"])
        sc = score_of(cls)
        if sc >= MIN_SCORE:
            out.append(dict(r, hits=sorted(cls), score=sc, shard=shard))
    return out


def pick(hits: list, n: int, per_video: int = 4, skip_videos: set | None = None, video_counts: dict | None = None,
         skip_ids: set | None = None, max_video_total: int = 30, quota: dict | None = None) -> list:
    """Highest score first, at most per_video rows per video, never past max_video_total rows of a video counting the rows
    the English parts already hold (video_counts), skipping held-out videos / used utterances. quota: {class: max rows}
    keeps one cheap class (percent, money) from filling the selection."""
    import collections

    skip_videos, video_counts, skip_ids, quota = skip_videos or set(), video_counts or {}, skip_ids or set(), quota or {}
    n_v, got, out = collections.Counter(), collections.Counter(), []
    for h in sorted(hits, key=lambda h: (-h["score"], h["utt_id"])):
        v = h["original_audio_id"]
        if v in skip_videos or h["utt_id"] in skip_ids:
            continue
        if n_v[v] >= per_video or n_v[v] + video_counts.get(v, 0) >= max_video_total:
            continue
        if any(got[k] >= quota[k] for k in h["hits"] if k in quota):
            continue
        n_v[v] += 1
        for k in h["hits"]:
            got[k] += 1
        out.append(h)
        if len(out) >= n:
            break
    return out


try:
    import modal

    app = modal.App("parakeet-en5-scan")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer")
             .add_local_python_source("en_hits_num", "labeller", "readings"))
    hf = modal.Secret.from_name("hf")

    @app.function(image=image, secrets=[hf], cpu=1, memory=4096, timeout=3600, retries=2, max_containers=14)
    def scan_shard(path: str) -> list:
        """Text columns of one English shard -> scored hit rows (backoff on HTTP 429, as en_hits.scan_shard)."""
        import random as _r
        import time

        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        for attempt in range(8):
            try:
                fs = HfFileSystem()
                rows = pq.ParquetFile(fs.open(f"datasets/{REPO}/{path}", block_size=1 << 20)).read(columns=COLS).to_pylist()
                return scan_rows(rows, path)
            except Exception as e:  # noqa: BLE001
                if "429" not in repr(e) or attempt == 7:
                    raise
                time.sleep(min(300, 15 * 2 ** attempt) * (0.5 + _r.random()))

    @app.local_entrypoint()
    def scan(files: str, n: int = 60, seed: int = 51, out: str = "hits.jsonl", exclude_shards: str = ""):
        """files: JSON list of [path, size] (the repo's file listing); picks n random English asr_only shards not named in
        exclude_shards (one "en006_00000278" per line) and writes the scored hits."""
        import collections
        import os
        import random

        skip = {l.strip() for l in open(exclude_shards) if l.strip()} if exclude_shards else set()
        name = lambda p: p.split("/")[1] + "_" + os.path.basename(p)[:-8]  # noqa: E731
        paths = sorted(p for p, _ in json.load(open(files)) if re.match(r"data/en\d{3}/asr_only/.*\.parquet$", p)
                       and name(p) not in skip)
        random.Random(seed).shuffle(paths)
        paths = paths[:n]
        tot, n_ok = collections.Counter(), 0
        with open(out, "w", encoding="utf-8") as f:
            for res in scan_shard.map(paths, return_exceptions=True):
                if isinstance(res, Exception):
                    print("FAIL", repr(res)[:200])
                    continue
                n_ok += 1
                for h in res:
                    tot.update(h["hits"])
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(json.dumps({"shards": n_ok, "of": len(paths), "hits": dict(tot)}))
except ImportError:
    pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:  # pick <hits.jsonl> <out.jsonl> <n> [--skip-videos f] [--video-counts f.json] [--skip-ids f]
        import collections

        a = sys.argv[2:]
        opt = {a[i]: a[i + 1] for i in range(3, len(a) - 1, 2)}
        hs = [json.loads(l) for l in open(a[0]) if l.strip()]
        read = lambda p: {l.strip() for l in open(p) if l.strip()} if p else set()  # noqa: E731
        sel = pick(hs, int(a[2]), int(opt.get("--per-video", 4)), read(opt.get("--skip-videos")),
                   json.load(open(opt["--video-counts"])) if "--video-counts" in opt else {}, read(opt.get("--skip-ids")),
                   quota=json.loads(opt.get("--quota", "{}")))
        with open(a[1], "w", encoding="utf-8") as f:
            for h in sel:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(len(sel), dict(collections.Counter(k for h in sel for k in h["hits"])),
              "shards", len({h["shard"] for h in sel}), "videos", len({h["original_audio_id"] for h in sel}))
