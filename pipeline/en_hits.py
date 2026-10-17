"""English rare-shape hunt in YODAS-Granary captions (2026-10-17): a TEXT-ONLY scan (caption columns, no audio) of
English shards for phone numbers, clock times and UK / Australian spoken forms ("double four", "nought", "half past",
"quarter to"), then audio for the HIT rows only, written as label_gpu manifests (the D1 path labels them).

    modal run en_hits.py::scan --n 100 --seed 3 --out hits_pilot.jsonl      # shards never scanned before; free-ish
    modal run en_hits.py::fetch --hits hits_sel.jsonl --tag enh1            # audio of the chosen hits -> manifests
    then modal_app.s3_label --lang en_<tag> ..., s3_veto / s3_recrop / s3_decode_crops, filter3 (as the parliament pull)
"""
import json
import re
import sys

VOL = "/vol"
REPO = "espnet/yodas-granary"
COLS = ["utt_id", "duration", "lang", "text", "original_audio_id", "original_audio_offset"]
_N = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
PATTERNS = {
    # 555-1234, (212) 555-1234, 0161 496 0000, 1-800-555-0199, +44 20 7946 0958; not years, prices or decimals
    "phone": re.compile(r"(?<![\d.,$£€/:])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)\s?\d{3}[ .-]\d{4}|\d{3}[ .-]\d{3}[ .-]\d{4}|"
                        r"\b\d{3}[ .-]\d{4}\b|\b0\d{2,4}[ .-]\d{3,4}[ .-]\d{3,4}|\b1[ -]?8[0-9]{2}[ .-]\d{3}[ .-]\d{4})"
                        r"(?![\d.,%])"),
    "clock": re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b\d{1,2}(?:[.:][0-5]\d)?\s?(?:a\.?m\.?|p\.?m\.?)(?![a-z])|"
                        r"\b(?:\d{1,2}|" + _N + r") o'clock\b", re.I),
    # "five to six years" is a range: "N to M" counts only before a time word; "N past M" and "half past" always
    "clock_words": re.compile(r"\bhalf past (?:" + _N + r"|\d{1,2}|the hour)\b|\bquarter (?:to|past) (?:" + _N + r"|\d{1,2})\b|"
                              r"\b(?:five|ten|twenty|twenty-five|twenty five) past (?:" + _N + r"|\d{1,2})\b|"
                              r"\b(?:five|ten|twenty|twenty-five|twenty five) to (?:" + _N + r"|\d{1,2})"
                              r"(?= in the| this| tonight| tomorrow| [ap]\.?m|[.,!?]|$)", re.I),
    # the labeller's own phone shape (10-11 digit North American numbers): only these get a phone span, so only these are hits
    "phone10": re.compile(r"(?:\+?1[-.\s])?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "double": re.compile(r"\bdouble (?:\d|oh|zero|nought|one|two|three|four|five|six|seven|eight|nine)\b", re.I),
    "nought": re.compile(r"\bnought\b", re.I),
}


# 2026-10-17: the shapes English real speech had none of (Roman numerals, "&", versions, fractions); es / fr: fractions
# and "&". Roman hits use the labeller's own rule (a capitalised word + II..LXXXIX, not "XL", "The IV").
_FRAC = r"(?<![\d/.])(?:\d+\s)?\d{1,2}/(?:2|3|4|5|8|10)(?![\d/])|[½¼¾⅓⅔⅛]"
RARE = {
    "en": {"amp": re.compile(r"\b[A-Z]{1,4}&[A-Z]{1,4}\b|(?<=\w) & (?=\w)"), "frac": re.compile(_FRAC),
           "version": re.compile(r"(?<![\w.])[vV]?\d{1,3}\.\d{1,3}\.\d{1,3}(?![\w.])|\b(?:version|iOS|Android|Windows|"
                                 r"Python|Java|PHP|HTML|Bluetooth|USB|Wi-?Fi|Web|macOS|Ubuntu|Chrome|Firefox|Node|"
                                 r"Angular|Unity|Unreal Engine|Photoshop|Excel|Office) v?\d{1,3}(?:\.\d{1,3})+\b", re.I)},
    "es": {"amp": re.compile(r"\b[A-Z]{1,4}&[A-Z]{1,4}\b|(?<=\w) & (?=\w)"), "frac": re.compile(_FRAC)},
    "fr": {"amp": re.compile(r"\b[A-Z]{1,4}&[A-Z]{1,4}\b|(?<=\w) & (?=\w)"), "frac": re.compile(_FRAC)},
}


# 2026-10-17: es / fr phone numbers, clock times and prices with cents (the decision set and the seam bed were thin in them).
# A phone is 8-12 digits in separated groups (a date "12-03-2020" and a plain decimal are not), or an unseparated
# 9-10 digit run that starts like a number line; a price carries a decimal part and a currency sign / word.
_PH_GROUPS = (r"(?<![\d.,$€£/:-])(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]){1,4}\d{2,4}(?![\d.,%:/-])")
_ES_PHONE = re.compile(_PH_GROUPS + r"|(?<![\d.,])[6-9]\d{8}(?![\d.,])")
_FR_PHONE = re.compile(r"(?<![\d.,$€£/:-])(?:(?:\+33|0033)[ .]?[1-9]|0[1-9])(?:[ .]?\d{2}){4}(?![\d.,%:/-])|" + _PH_GROUPS)
_CUR_ES = re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})*,\d{2}\s?(?:€|euros?|dólares|pesos|\$)|[$€]\s?\d+[.,]\d{2}(?![\d])")
_CUR_FR = re.compile(r"(?<![\d.,])\d{1,3}(?:[ .]\d{3})*,\d{2}\s?(?:€|euros?|dollars?|\$)|\d+\s?€\s?\d{2}\b|[$€]\s?\d+[.,]\d{2}(?![\d])")
_CLOCK_ES = re.compile(r"(?<![\d.,/])(?:[01]?\d|2[0-3])[:.][0-5]\d\s?(?:h\b|horas|a\.?m\.?|p\.?m\.?)|"
                       r"(?<![\d.,/])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:.,])|\ba las? \d{1,2}(?:\s?(?:y|h)\s?\d{2})?\b", re.I)
_CLOCK_FR = re.compile(r"(?<![\d.,/])(?:[01]?\d|2[0-3])\s?h\s?[0-5]\d\b|(?<![\d.,/])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:.,])|"
                       r"\b(?:à|a) \d{1,2}\s?(?:h|heures?)\b", re.I)


class _Phone:
    """A phone regex plus a filter: 8-12 digits, and not a date ("12-03-2020", "12 03 20")."""

    def __init__(self, rx):
        self.rx = rx

    def search(self, text):
        for m in self.rx.finditer(text or ""):
            d = re.sub(r"\D", "", m.group(0))
            if 8 <= len(d) <= 12 and not re.fullmatch(r"\s*\d{1,2}[ .-]\d{1,2}[ .-]\d{2,4}\s*", m.group(0)):
                return m
        return None


_ES_PHONE, _FR_PHONE = _Phone(_ES_PHONE), _Phone(_FR_PHONE)
RARE["es"].update(phone=_ES_PHONE, cur=_CUR_ES, clock=_CLOCK_ES)
RARE["fr"].update(phone=_FR_PHONE, cur=_CUR_FR, clock=_CLOCK_FR)


def _roman_hit(text: str) -> bool:
    import labeller as L

    for mt in L.ROMAN.finditer(text or ""):
        r = mt.group("r")
        if r not in L.ROMAN_NOT and mt.group(0).split()[0] not in L.ROMAN_STOP and L.roman_value(r):
            return True
    return False


def hits_of(text: str, lang: str = "en", kinds: tuple = ()) -> list:
    """Shapes found in one caption. kinds: only these (default: every English shape, or every rare shape of es / fr)."""
    pats = dict(PATTERNS, **RARE["en"]) if lang == "en" else dict(RARE[lang])
    out = [k for k, p in pats.items() if (not kinds or k in kinds) and p.search(text or "")]
    if lang == "en" and (not kinds or "roman" in kinds) and _roman_hit(text):
        out.append("roman")
    return out


def scan_rows(rows: list, shard: str, lang: str = "en", kinds: tuple = ()) -> list:
    out = []
    for r in rows:
        h = hits_of(r["text"], lang, kinds)
        if h and 1.0 <= (r.get("duration") or 0) <= 30.0:
            out.append(dict(r, hits=h, shard=shard))
    return out


def pick(hits: list, per_video: int = 3, quota: dict | None = None) -> list:
    """Spread: at most per_video rows per video, rarest shape first, up to quota[shape] rows each."""
    import collections

    quota = quota or {}
    n_v, got, out, seen = collections.Counter(), collections.Counter(), [], set()
    avail = collections.Counter(k for h in hits for k in h["hits"])
    for k in sorted(avail, key=lambda k: avail[k]):
        for h in hits:
            if k not in h["hits"] or h["utt_id"] in seen or n_v[h["original_audio_id"]] >= per_video:
                continue
            if got[k] >= quota.get(k, 10**9):
                break
            seen.add(h["utt_id"])
            n_v[h["original_audio_id"]] += 1
            for kk in h["hits"]:
                got[kk] += 1
            out.append(h)
    return out


try:
    import modal

    app = modal.App("parakeet-en-hits")
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer")
             .add_local_python_source("en_hits", "labeller", "readings"))
    hf = modal.Secret.from_name("hf")

    @app.function(image=image, secrets=[hf], cpu=1, memory=4096, timeout=3600, retries=2, max_containers=14)
    def scan_shard(path: str, lang: str = "en", kinds: tuple = ()) -> list:
        """14 containers at most and a backoff on HTTP 429 (the hub rate-limits 64 parallel readers)."""
        import random as _r
        import time

        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        for attempt in range(8):
            try:
                fs = HfFileSystem()
                rows = pq.ParquetFile(fs.open(f"datasets/{REPO}/{path}", block_size=1 << 20)).read(columns=COLS).to_pylist()
                return scan_rows(rows, path, lang, tuple(kinds))
            except Exception as e:  # noqa: BLE001
                if "429" not in repr(e) or attempt == 7:
                    raise
                time.sleep(min(300, 15 * 2 ** attempt) * (0.5 + _r.random()))

    @app.function(image=image, secrets=[hf], volumes={VOL: vol}, timeout=3600)
    def list_new(n: int, seed: int, lang: str = "en", fresh: bool = False, exclude: tuple = ()) -> list:
        import glob
        import os
        import random

        from huggingface_hub import HfApi

        files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                       if f.startswith(f"data/{lang}") and "/asr_only/" in f and f.endswith(".parquet"))
        # fresh: skip shards scanned / labelled before (the phone / clock hunt); a rare-shape scan may reread them
        done = {os.path.basename(p)[:-6] for p in glob.glob(f"{VOL}/s0/scan/{lang}/*.jsonl")} if fresh else set()
        name = lambda f: f.split("/")[1] + "_" + os.path.basename(f)[:-8]  # noqa: E731
        skip = done | set(exclude)  # exclude: shard names ("fr102_00000171") an earlier pass already took rows from
        new = [f for f in files if name(f) not in skip]
        random.Random(seed).shuffle(new)
        return new[:n]

    @app.local_entrypoint()
    def scan(n: int = 100, seed: int = 3, out: str = "hits.jsonl", lang: str = "en", kinds: str = "", fresh: bool = True,
         exclude_file: str = ""):
        import collections

        skip = tuple(l.strip() for l in open(exclude_file) if l.strip()) if exclude_file else ()
        paths = list_new.remote(n, seed, lang, fresh, skip)
        ks = tuple(k for k in kinds.split(",") if k)
        tot, n_ok = collections.Counter(), 0
        with open(out, "w", encoding="utf-8") as f:
            for res in scan_shard.map(paths, kwargs={"lang": lang, "kinds": ks}, return_exceptions=True):
                if isinstance(res, Exception):
                    continue
                n_ok += 1
                for h in res:
                    tot.update(h["hits"])
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(json.dumps({"shards": n_ok, "of": len(paths), "hits": dict(tot)}))

    @app.function(image=image, secrets=[hf], volumes={VOL: vol}, cpu=2, memory=8192, timeout=3600, retries=3,
                  max_containers=6)
    def fetch_shard(path: str, rows: list, tag: str, lang: str = "en") -> list:
        """Audio of the given rows of one shard -> /vol/yodas/en_<tag>/<shard>/<utt_id>.wav; returns manifest rows."""
        import os

        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        want = {r["utt_id"]: r for r in rows}
        name = path.split("/")[1] + "_" + os.path.basename(path)[:-8]
        out_dir = f"{VOL}/yodas/en_{tag}/{name}" if lang == "en" else f"{VOL}/ml/p3v2/{lang}/wav"
        os.makedirs(out_dir, exist_ok=True)
        local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/tmp/shard")
        pf = pq.ParquetFile(local)
        kept = []
        for rg in range(pf.metadata.num_row_groups):
            for x in pf.read_row_group(rg, columns=["utt_id", "audio"]).to_pylist():
                r = want.get(x["utt_id"])
                if r is None:
                    continue
                if lang != "en":  # a label2 candidate row (+ the same clip for the second transcript)
                    wav = f"{out_dir}/{tag}_{x['utt_id']}.wav"
                    open(wav, "wb").write(x["audio"]["bytes"])
                    os.makedirs(f"{VOL}/ml/p3v2/{lang}/flac_{tag}", exist_ok=True)
                    open(f"{VOL}/ml/p3v2/{lang}/flac_{tag}/{x['utt_id']}.wav", "wb").write(x["audio"]["bytes"])
                    kept.append({"audio_filepath": wav, "duration": r["duration"], "text": r["text"], "lang": lang,
                                 "src": "yodas", "id": x["utt_id"], "origin": "new", "hits": r["hits"]})
                    continue
                wav = f"{out_dir}/{x['utt_id']}.wav"
                open(wav, "wb").write(x["audio"]["bytes"])
                kept.append({k: r[k] for k in COLS} | {"kind": "digit" if re.search(r"\d", r["text"]) else "word",
                                                       "shard": path, "audio_filepath": wav, "hits": r["hits"]})
        os.remove(local)
        vol.commit()
        return kept

    @app.local_entrypoint()
    def fetch(hits: str, tag: str, per_manifest: int = 400, lang: str = "en"):
        import collections
        import io

        rows = [json.loads(l) for l in open(hits) if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["shard"]].append(r)
        kept = []
        for res in fetch_shard.starmap([(p, rs, tag, lang) for p, rs in by.items()], return_exceptions=True):
            if isinstance(res, Exception):
                print("FAIL", repr(res)[:200])
                continue
            kept += res
        if lang != "en":
            with vol.batch_upload(force=True) as b:
                b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode()),
                           f"ml/p3v2/{lang}/cands_{tag}.jsonl")
            print(json.dumps({"rows": len(kept), "shards": len(by), "hours": round(sum(r["duration"] for r in kept) / 3600, 2)}))
            return
        with vol.batch_upload(force=True) as b:
            for i in range(0, len(kept), per_manifest):
                part = kept[i:i + per_manifest]
                b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part).encode()),
                           f"yodas/en_{tag}/manifests/{tag}_{i // per_manifest:03d}.jsonl")
        print(json.dumps({"rows": len(kept), "shards": len(by), "hours": round(sum(r["duration"] for r in kept) / 3600, 2),
                          "manifests": -(-len(kept) // per_manifest)}))
except ImportError:
    pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:  # pick <hits.jsonl> <out.jsonl> [per_video] [shape:n,...]
        hs = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
        q = dict((k, int(v)) for k, v in (x.split(":") for x in sys.argv[5].split(","))) if len(sys.argv) > 5 else None
        sel = pick(hs, int(sys.argv[4]) if len(sys.argv) > 4 else 3, q)
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            for h in sel:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        import collections

        print(len(sel), dict(collections.Counter(k for h in sel for k in h["hits"])))
