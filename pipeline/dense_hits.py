"""Number-density caption scan for es / fr / de / it (2026-10-18): a TEXT-ONLY pass over YODAS-Granary captions that ranks each
caption by how thin the classes in it are in the real-speech parts, then the same audio / label path as the earlier
caption-hit pulls (en_hits.fetch with --lang -> label2 -> judge -> edge fix). Label rules, CTC verification and the
held-out exclusions are not touched; only WHICH rows are fetched is new.

Classes (the captions' own digit shapes, found with ml_cands._find, so a row the labeller cannot read never hits):
decimals, percent, prices with cents, big numbers (>= 10^6 written in full), long numbers (>= 6 spoken words), dates,
forward years 2026-2060, phone numbers, clock times. A business / finance / news word adds a small bonus, and a video
whose captions are dense in such words gets a video bonus (the channel proxy).

    modal run dense_hits.py::scan --lang es --files files.json --n 60 --seed 71 --out hits_es.jsonl
    python3 dense_hits.py pick hits.jsonl sel.jsonl N --skip-videos v.txt --skip-ids i.txt --video-counts c.json
    modal run en_hits.py::fetch --hits sel.jsonl --tag dense1 --lang es      # unchanged
"""
import json
import os
import re
import sys

REPO = "espnet/yodas-granary"
COLS = ["utt_id", "duration", "lang", "text", "original_audio_id", "original_audio_offset"]
MIN_SCORE = 2.0
MAX_SCORE = 14.0
DUR = (3.0, 24.0)
DUR_AST = (3.0, 32.0)  # the ast config's rows run to 40 s (label2 splits rows over 20 s at pauses)

# class weights per language: how thin the class is in the real parts (2026-10-18 census of the candidate mix)
W = {
    "es": {"big": 8, "long6": 3, "cents": 4, "dec": 2, "clock": 1.5, "date": 0.5, "fwd": 2, "phone": 0.3, "pct": 0.3},
    "fr": {"big": 8, "long6": 2, "cents": 3.5, "dec": 2, "clock": 3, "date": 0.5, "fwd": 2, "phone": 1, "pct": 0.3},
    "de": {"big": 8, "long6": 3, "cents": 4, "dec": 0.8, "clock": 1.5, "date": 0.5, "fwd": 2, "phone": 6, "pct": 0.2},
    "it": {"big": 8, "long6": 6, "cents": 6, "dec": 2, "clock": 3, "date": 0.6, "fwd": 2, "phone": 2, "pct": 0.4},
    "pt": {"big": 8, "long6": 4, "cents": 4, "dec": 2, "clock": 2, "date": 0.6, "fwd": 2, "phone": 1.5, "pct": 0.4},
}
COMMON_FWD = (2030, 2035, 2040, 2050)  # already >= 40 spans each

# business / finance / economy / news vocabulary (a ranking bonus only; the label never depends on it)
DOMAIN = {
    "es": r"empresa|ingresos|beneficios?|acciones|bolsa|econom[ií]a|inflaci[oó]n|trimestre|mercado|inversi[oó]n|inversores?|"
          r"d[oó]lar|euros?|millones|miles de millones|precio|d[eé]ficit|deuda|PIB|banco|crecimiento|ventas|gobierno|"
          r"impuestos?|tasa|presupuesto|salario|desempleo|exportaciones|petr[oó]leo|bonos|cotiza",
    "fr": r"entreprise|chiffre d'affaires|b[eé]n[eé]fices?|actions?|bourse|[eé]conomie|inflation|trimestre|march[eé]|"
          r"investissement|investisseurs?|dollars?|euros?|millions|milliards|prix|d[eé]ficit|dette|PIB|banque|croissance|"
          r"ventes|gouvernement|imp[oô]ts?|taux|budget|salaire|ch[oô]mage|exportations|p[eé]trole|obligations|cote",
    "de": r"Unternehmen|Umsatz|Gewinn|Aktien?|B[oö]rse|Wirtschaft|Inflation|Quartal|Markt|Investitionen?|Investoren?|Dollar|"
          r"Euro|Millionen|Milliarden|Preis|Defizit|Schulden|BIP|Bank|Wachstum|Verk[aä]ufe|Regierung|Steuern?|Zinsen?|"
          r"Haushalt|Gehalt|Arbeitslosigkeit|Exporte|[OÖ]l|Anleihen|Kurs",
    "it": r"azienda|impresa|ricavi|utili?|azioni|borsa|economia|inflazione|trimestre|mercato|investimenti|investitori|dollari?|"
          r"euro|milioni|miliardi|prezzo|deficit|debito|PIL|banca|crescita|vendite|governo|tasse|tasso|bilancio|stipendio|"
          r"disoccupazione|esportazioni|petrolio|titoli|quotazione",
    "pt": r"empresa|receitas?|lucros?|a[cç][oõ]es|bolsa|economia|infla[cç][aã]o|trimestre|mercado|investimentos?|investidores?|"
          r"d[oó]lar|d[oó]lares|euros?|milh[oõ]es|bilh[oõ]es|pre[cç]os?|d[eé]fice|d[ií]vida|PIB|banco|crescimento|vendas|governo|"
          r"impostos?|taxa|or[cç]amento|sal[aá]rio|desemprego|exporta[cç][oõ]es|petr[oó]leo|t[ií]tulos|cota[cç][aã]o|reais|real",
}
_DOM = {l: re.compile(r"\b(?:" + p + r")\b", re.I) for l, p in DOMAIN.items()}


def _grp(n: int, lang: str) -> int:
    h, r = divmod(n, 100)
    t = 0
    if h:
        t += 1 if h == 1 else 2
    if r:
        if lang == "es":
            t += 1 if r <= 30 or r % 10 == 0 else 3
        elif lang == "fr":
            t += 1 if r <= 16 or r % 10 == 0 else 2 if r < 70 else 3
        else:  # de / it, split-compound form
            t += 1 if r <= 20 or r % 10 == 0 else 3 if lang == "de" else 2
    return t


def words_est(v: int, lang: str) -> int:
    """Rough count of spoken words (split-compound form for de / it) of a whole number; ranking only."""
    if v == 0:
        return 1
    t = 0
    for sc in (10**9, 10**6, 1000):
        q, v = divmod(v, sc)
        if q:
            t += (1 if q == 1 and sc == 1000 else _grp(q, lang)) + 1
    return t + _grp(v, lang)


def classes_of(text: str, lang: str) -> dict:
    """{class: count} of the thin-class shapes in a caption; {} when the labeller could not read every digit."""
    import ml_cands as M

    if not re.search(r"\d", text or ""):
        return {}
    try:
        if not M.all_digits_spanned(text, lang):
            return {}
        ds = M._find(text, lang)
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}

    def add(k, n=1):
        out[k] = out.get(k, 0) + n

    mon = set(M.MONTHS[lang].split())
    for d in ds:
        k = d["kind"]
        if k == "dec":
            add("dec")
        elif k == "pct":
            add("pct")
        elif k == "cur":
            if d.get("frac") and len(d["frac"]) == 2:
                add("cents")
            elif d.get("int") and re.fullmatch(r"\d{1,3}(?:[.   ]\d{3})+", d["int"]):
                add("big")
        elif k == "phone":
            add("phone")
        elif k == "clock":
            add("clock")
        elif k == "date":
            add("date")
        elif k in ("int", "code") and d.get("int"):
            digits = re.sub(r"\D", "", d["int"])
            if not digits:
                continue
            v = int(digits)
            if re.fullmatch(r"\d{4}", d["int"]) and 2026 <= v <= 2060:
                add("fwd" if v not in COMMON_FWD else "fwd_common")
            elif v >= 10**6 and k == "int":
                add("big")
            elif k == "int" and v >= 1000 and words_est(v, lang) >= 6:
                add("long6")
            after = text[d["b"]:d["b"] + 24].lower().split()
            if after and (after[0].strip(".,") in mon or (len(after) > 1 and after[0] in ("de", "del") and after[1].strip(".,") in mon)):
                add("date")
    return out


def score_of(cls: dict, lang: str) -> float:
    w = W[lang]
    s = 0.0
    for k, n in cls.items():
        s += (w.get("fwd", 2) * 0.4 if k == "fwd_common" else w.get(k, 0.0)) * (1.0 + 0.25 * (min(n, 3) - 1))
    return round(min(s, MAX_SCORE), 2)


def domain_hits(text: str, lang: str) -> int:
    return len(_DOM[lang].findall(text or ""))


def scan_rows(rows: list, shard: str, lang: str, min_score: float = MIN_SCORE) -> list:
    dur = DUR_AST if "/ast/" in shard else DUR
    """Scored hit rows of one shard. The video bonus: rows of a video whose captions are dense in business / news words."""
    import collections

    vid_n, vid_d = collections.Counter(), collections.Counter()
    for r in rows:
        vid_n[r["original_audio_id"]] += 1
        vid_d[r["original_audio_id"]] += 1 if domain_hits(r["text"], lang) else 0
    out = []
    for r in rows:
        if not dur[0] <= (r.get("duration") or 0) <= dur[1]:
            continue
        cls = classes_of(r["text"], lang)
        if not cls:
            continue
        sc = score_of(cls, lang)
        v = r["original_audio_id"]
        dens = vid_d[v] / max(vid_n[v], 1)
        dh = domain_hits(r["text"], lang)
        sc += min(dh, 3) * 0.5 + (1.5 if vid_n[v] >= 8 and dens >= 0.12 else 0.0)
        sc = round(min(sc, MAX_SCORE + 4.5), 2)
        if sc >= min_score:
            out.append(dict(r, hits=sorted(cls), score=sc, shard=shard, domain=dh, vdens=round(dens, 3)))
    return out


def pick(hits: list, n: int, per_video: int = 6, skip_videos: set | None = None, video_counts: dict | None = None,
         skip_ids: set | None = None, max_video_total: int = 30, quota: dict | None = None, year_cap: int = 12,
         max_shards: int = 0, max_hours: float = 0.0) -> list:
    """Highest score first, at most per_video rows per video here and max_video_total with the rows the parts already
    hold (video_counts), skipping held-out videos / used utterances; quota: {class: max rows}; year_cap: rows per forward
    year value (the per-value year cap)."""
    import collections

    skip_videos, video_counts, skip_ids, quota = skip_videos or set(), video_counts or {}, skip_ids or set(), quota or {}
    n_v, got, out, yr = collections.Counter(), collections.Counter(), [], collections.Counter()
    if max_shards:  # fetching costs per shard (~600 MB for the ast config): keep the shards with the best rows
        by = collections.defaultdict(list)
        for h in hits:
            if h["original_audio_id"] not in skip_videos and h["utt_id"] not in skip_ids:
                by[h["shard"]].append(h["score"])
        val = {s_: sum(sorted(v, reverse=True)[:25]) for s_, v in by.items()}
        keep = set(sorted(val, key=lambda s_: -val[s_])[:max_shards])
        hits = [h for h in hits if h["shard"] in keep]
    secs = 0.0
    for h in sorted(hits, key=lambda h: (-h["score"], h["utt_id"])):
        v = h["original_audio_id"]
        if v in skip_videos or h["utt_id"] in skip_ids:
            continue
        if n_v[v] >= per_video or n_v[v] + video_counts.get(v, 0) >= max_video_total:
            continue
        if any(got[k] >= quota[k] for k in h["hits"] if k in quota):
            continue
        ys = re.findall(r"(?<![\d.,])(20[2-6]\d)(?![\d.,]\d)", h["text"]) if "fwd" in h["hits"] else []
        if any(yr[y] >= year_cap for y in ys):
            continue
        n_v[v] += 1
        got.update(h["hits"])
        yr.update(ys)
        out.append(h)
        secs += h["duration"]
        if len(out) >= n or (max_hours and secs >= max_hours * 3600):
            break
    return out


try:
    import modal

    app = modal.App("parakeet-dense-scan")
    image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("pyarrow", "huggingface_hub", "hf_transfer")
             .add_local_python_source("dense_hits", "ml_cands")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
    hf = modal.Secret.from_name("hf")

    @app.function(image=image, secrets=[hf], cpu=1, memory=4096, timeout=3600, retries=2, max_containers=12)
    def scan_shard(path: str, lang: str, min_score: float = MIN_SCORE) -> list:
        """Text columns of one shard -> scored hit rows (backoff on HTTP 429)."""
        import random as _r
        import time

        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        name = path.split("/")[1] + "_" + path.rsplit("/", 1)[-1][:-8]
        for attempt in range(8):
            try:
                if "/ast/" in path:  # ~600 MB shards: read the text columns only (range requests)
                    from huggingface_hub import HfFileSystem

                    rows = pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/{path}", block_size=1 << 20)).read(columns=COLS).to_pylist()
                else:
                    local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/tmp/shard")  # one request per shard
                    try:
                        rows = pq.ParquetFile(local).read(columns=COLS).to_pylist()
                    finally:
                        os.remove(local)
                return scan_rows(rows, path, lang, min_score) + [{"_shard_rows": len(rows), "shard": name, "_digit": sum(1 for r in rows if re.search(r"\d", r['text'] or ''))}]
            except Exception as e:  # noqa: BLE001
                if "429" not in repr(e) or attempt == 7:
                    raise
                time.sleep(min(300, 15 * 2 ** attempt) * (0.5 + _r.random()))

    @app.local_entrypoint()
    def scan(lang: str, files: str, n: int = 60, seed: int = 71, out: str = "hits.jsonl", exclude_shards: str = "", min_score: float = MIN_SCORE, offset: int = 0, part: str = "asr_only"):
        """files: JSON list of [path, size] (the repo's listing); n random asr_only shards of `lang` not in exclude_shards."""
        import collections
        import os
        import random

        skip = {l.strip() for l in open(exclude_shards) if l.strip()} if exclude_shards else set()
        name = lambda p: p.split("/")[1] + "_" + os.path.basename(p)[:-8]  # noqa: E731
        paths = sorted(p for p, _ in json.load(open(files)) if re.match(rf"data/{lang}\d{{3}}/{part}/.*\.parquet$", p)
                       and name(p) not in skip)
        random.Random(seed).shuffle(paths)
        paths = paths[offset:offset + n]
        tot, n_ok, n_rows, n_dig = collections.Counter(), 0, 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for res in scan_shard.map(paths, kwargs={"lang": lang, "min_score": min_score}, return_exceptions=True, order_outputs=False):
                if isinstance(res, Exception):
                    print("FAIL", repr(res)[:200])
                    continue
                n_ok += 1
                for h in res:
                    if "_shard_rows" in h:
                        n_rows += h["_shard_rows"]
                        n_dig += h["_digit"]
                        continue
                    tot.update(h["hits"])
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
                f.flush()
        print(json.dumps({"lang": lang, "shards": n_ok, "of": len(paths), "rows": n_rows, "digit_rows": n_dig, "hits": dict(tot)}))
except ImportError:
    pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:  # pick <hits.jsonl> <out.jsonl> <n> [--skip-videos f] [--video-counts f.json] [--skip-ids f] ...
        import collections

        a = sys.argv[2:]
        opt = {a[i]: a[i + 1] for i in range(3, len(a) - 1, 2)}
        hs = [json.loads(l) for l in open(a[0]) if l.strip()]
        read = lambda p: {l.strip() for l in open(p) if l.strip()} if p else set()  # noqa: E731
        sel = pick(hs, int(a[2]), int(opt.get("--per-video", 6)), read(opt.get("--skip-videos")),
                   json.load(open(opt["--video-counts"])) if "--video-counts" in opt else {}, read(opt.get("--skip-ids")),
                   quota=json.loads(opt.get("--quota", "{}")), year_cap=int(opt.get("--year-cap", 12)),
                   max_shards=int(opt.get("--shards", 0)), max_hours=float(opt.get("--hours", 0)))
        with open(a[1], "w", encoding="utf-8") as f:
            for h in sel:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(len(sel), dict(collections.Counter(k for h in sel for k in h["hits"])),
              "shards", len({h["shard"] for h in sel}), "videos", len({h["original_audio_id"] for h in sel}),
              "hours", round(sum(h["duration"] for h in sel) / 3600, 2))
