"""Parliament speech for the 16 languages outside wave 1 (2026-10-16): NVIDIA Granary VoxPopuli pseudo-labels
(CC-BY-4.0, digits kept) over the VoxPopuli unlabelled plenary audio (CC0), cut by the published segment timings.

    modal run ml_multi_vox.py::index --langs pl,ro,sv            # text-only index -> /vol/ml/multi/granvox/<lang>/
    modal run ml_multi_vox.py::pilot --langs pl --year 2020      # ~130 variety-stratified clips of one debate year

Held out: every plenary DAY of the VoxPopuli transcribed dev / test splits (where the language has them), so a later
VoxPopuli-test bed never shares a day with training clips. Class strata come from ml_multi.classes (text only).
"""
import json
import os
import re

import modal

VOL = "/vol"
OUT = f"{VOL}/ml/multi"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("huggingface_hub", "hf_transfer", "soundfile", "numpy")
         .add_local_python_source("ml_multi"))
app = modal.App("parakeet-ml-multi-vox")
ASR_SPLITS = "https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_{lang}.tsv.gz"
SEGS = "https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz"


def held_days(lang: str, tries: int = 4, wait: float = 2.0) -> set:
    """Plenary days of the VoxPopuli transcribed dev/test splits (empty when the language has no transcribed set).
    Only HTTP 403 / 404 means "no transcribed set"; any other failure is retried with backoff and then raised, so a
    network error can never pass as "nothing held out" (evaluation days would leak into training)."""
    import csv
    import gzip
    import time
    import urllib.error
    import urllib.request

    p = f"/tmp/asr_{lang}.tsv.gz"
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(ASR_SPLITS.format(lang=lang), p)
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return set()
            err = e
        except (urllib.error.URLError, OSError) as e:
            err = e
        if attempt + 1 < tries:
            time.sleep(wait * 2 ** attempt)
    else:
        raise RuntimeError(f"held-out day list for {lang} unavailable after {tries} tries: {err!r}") from err
    with gzip.open(p, "rt") as f:
        return {r["session_id"][:8] for r in csv.DictReader(f, delimiter="|") if r["split"] in ("dev", "test")}


@app.function(image=image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=2 * 3600, max_containers=4)
def index_lang(lang: str) -> dict:
    from collections import Counter

    from huggingface_hub import hf_hub_download

    import ml_multi as MM

    local = hf_hub_download("nvidia/Granary", f"{lang}/voxpopuli/{lang}_asr.jsonl", repo_type="dataset",
                            local_dir="/tmp/granary")
    held = held_days(lang)
    out = f"{OUT}/granvox/{lang}"
    os.makedirs(out, exist_ok=True)
    st = {"lang": lang, "rows": 0, "digit_rows": 0, "held_out": 0, "kinds": Counter(), "days": set(), "years": Counter(),
          "debate_years": Counter()}
    with open(local) as f, open(f"{out}/index.jsonl", "w") as w:
        for line in f:
            st["rows"] += 1
            r = json.loads(line)
            cap = (r.get("text") or "").strip()
            if not re.search(r"\d", cap):
                continue
            st["digit_rows"] += 1
            day = (r.get("utt_id") or "")[:8]
            if day in held:
                st["held_out"] += 1
                continue
            ks = MM.kinds(cap, lang)
            st["days"].add(day)
            st["kinds"].update(ks)
            st["debate_years"][day[:4]] += 1
            st["years"].update(MM.classes(cap, lang)["years"])
            w.write(json.dumps({"utt_id": r["utt_id"], "audio_filepath": r["audio_filepath"], "text": cap, "kinds": ks,
                                "day": day}, ensure_ascii=False) + "\n")
    os.remove(local)
    yv = st.pop("years")
    st["year_spans"] = sum(yv.values())
    st["year_values_1990_2030_ge40"] = sum(yv[y] >= 40 for y in range(1990, 2031))
    st["forward_2031_2060_ge10"] = sum(yv[y] >= 10 for y in range(2031, 2061))
    st["forward_spans_2025_2060"] = sum(yv[y] for y in range(2025, 2061))
    st["kinds"], st["days"], st["held_days"] = dict(st["kinds"]), len(st["days"]), len(held)
    st["debate_years"] = dict(sorted(st["debate_years"].items()))
    json.dump(st, open(f"{out}/index_stats.json", "w"))
    vol.commit()
    return st


def select(rows: list, lang: str, seed: int = 1, quotas: dict | None = None, per_value: int = 2, per_event: int = 3) -> list:
    """Variety-first pilot selection: year VALUES spread (forward years 2025-2060 first), then class quotas, at most
    `per_event` clips per plenary session and one per masked-digit template."""
    import collections
    import random

    import ml_multi as MM

    quotas = quotas or {"clock": 12, "cur": 15, "big": 12, "dec": 12, "pct": 12, "range": 8, "ord": 8, "phone": 5, "int": 10}
    rng = random.Random(f"{seed}-{lang}")
    rows = [r for r in rows if 40 <= len(r["text"]) <= 600]
    rng.shuffle(rows)
    tmpl = lambda r: re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", r["text"].lower()))[:60]  # noqa: E731
    seen_t, ev_n, sel = set(), collections.Counter(), []

    def take(r, why):
        ev = r["utt_id"].rsplit("_", 1)[0]
        if tmpl(r) in seen_t or ev_n[ev] >= per_event or any(s["utt_id"] == r["utt_id"] for s in sel):
            return False
        seen_t.add(tmpl(r)); ev_n[ev] += 1
        sel.append(dict(r, why=why))
        return True

    by_val = collections.defaultdict(list)
    for r in rows:
        for y in set(MM.classes(r["text"], lang)["years"]):
            by_val[y].append(r)
    fwd = [y for y in by_val if 2025 <= y <= 2060]
    rest = [y for y in by_val if y not in fwd]
    got = collections.Counter()
    for y in fwd + sorted(rest, key=lambda y: -len(by_val[y]))[:20]:
        for r in by_val[y]:
            if got[y] >= per_value:
                break
            if take(r, f"year:{y}"):
                got[y] += 1
    for k, q in quotas.items():
        n = 0
        for r in rows:
            if n >= q:
                break
            if k in r["kinds"] and take(r, k):
                n += 1
    return sel


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=4 * 3600, max_containers=6)
def pilot_lang(lang: str, year: str, tag: str = "pilot") -> dict:
    """Select ~130 clips of one debate year and cut them from both session tars (wav for training, FLAC for the
    external transcript) under /vol/ml/multi/<lang>/<tag>/."""
    import csv
    import gzip
    import subprocess
    import tarfile
    import urllib.request

    import numpy as np
    import soundfile as sf

    rows = [json.loads(l) for l in open(f"{OUT}/granvox/{lang}/index.jsonl") if l.startswith('{"utt_id": "' + year)]
    sel = {r["utt_id"]: r for r in select(rows, lang)}
    events = {u.rsplit("_", 1)[0] for u in sel}  # "<session>_<lang>"
    urllib.request.urlretrieve(SEGS, "/tmp/segs.tsv.gz")
    segs = {}
    with gzip.open("/tmp/segs.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["event_id"] in events:
                segs[f'{r["event_id"]}_{r["segment_no"]}'] = (float(r["start"]), float(r["end"]))
    out = f"{OUT}/{lang}/{tag}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    os.makedirs(f"{out}/flac", exist_ok=True)
    st = {"lang": lang, "year": year, "selected": len(sel), "events": len(events), "cut": 0, "no_seg": 0}
    res = []
    for tname in (f"{lang}_{year}", f"{lang}_{year}_2"):
        try:
            resp = urllib.request.urlopen(f"https://dl.fbaipublicfiles.com/voxpopuli/audios/{tname}.tar")
        except Exception:  # noqa: BLE001
            continue
        with tarfile.open(fileobj=resp, mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                ev = os.path.basename(m.name).rsplit(".", 1)[0]
                if ev not in events:
                    continue
                p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f",
                                    "s16le", "pipe:1"], input=tar.extractfile(m).read(), capture_output=True)
                x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768
                for u in [u for u in sel if u.rsplit("_", 1)[0] == ev]:
                    if u not in segs:
                        st["no_seg"] += 1
                        continue
                    s0, e0 = segs[u]
                    clip = x[int(s0 * 16000): int(e0 * 16000)]
                    if len(clip) < 16000:
                        continue
                    sf.write(f"{out}/wav/{u}.wav", clip, 16000, subtype="PCM_16")
                    sf.write(f"{out}/flac/{u}.flac", clip, 16000, format="FLAC")
                    res.append({"id": u, "audio_filepath": f"{out}/wav/{u}.wav", "duration": round(len(clip) / 16000, 3),
                                "text": sel[u]["text"], "lang": lang, "src": "vox", "kinds": sel[u]["kinds"],
                                "why": sel[u]["why"], "day": sel[u]["day"]})
                    st["cut"] += 1
        vol.commit()
    with open(f"{out}/rows.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    return st


@app.local_entrypoint()
def index(langs: str = "pl,ro,sv,hu,fi"):
    for r in index_lang.map(langs.split(",")):
        print(json.dumps(r, ensure_ascii=False))


@app.local_entrypoint()
def pilot(langs: str = "pl,ro,sv,hu,fi", year: str = "2020", tag: str = "pilot"):
    for r in pilot_lang.starmap([(l, year, tag) for l in langs.split(",")]):
        print(json.dumps(r, ensure_ascii=False))
