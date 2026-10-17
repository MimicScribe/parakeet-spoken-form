"""Cloud-voice synthesis for the TTS sets (2026-10-16): plan sentences with ml_tts.plan, voice the spoken label with
Google Cloud Text-to-Speech Chirp 3 HD voices at several speaking rates, upload the clips, and write a planned-rows file
that ml_tts.tts_lang verifies exactly like the local voices (engine "presynth", CTC check, alignment).

Uses the caller's gcloud login (`gcloud auth print-access-token`, project from `gcloud config`); nothing secret is stored.

    python ml_tts_cloud.py --langs de,es --n 60 --tag tts3            # pilot: plan, synthesise, upload
    modal run ml_tts.py --langs de,es --out-tag tts3 --rows-file ml/tts3/{lang}/planned.jsonl   # verify

Ukrainian plans with ml_tts_uk (case-fixed carrier sentences, readings from the ml_slav tables) at normal speaking
rates only; --uk-classes gives those classes equal shares (a pilot), otherwise ml_tts_uk.CLASSES weights apply:

    python ml_tts_cloud.py --langs uk --n 700 --tag tts6uk --rm-local
    python ml_tts_cloud.py --langs ru --n 800 --tag tts9ru --rm-local   # Russian: ml_tts_ru, same rules

A per-class top-up at normal speaking rates, count values weighted toward the thin ones:

    python ml_tts_cloud.py --langs nl --tag <tag> --normal-rates --class-n '{"phone": 90, "count": 270}' \
        --count-quota quota.json

Synthetic speech may cover any class, years and plain counts included, at <= 30% of the class in the mix (mix_build
TTS_SHARE). The default class list covers the shapes real speech lacks; pass --classes to add others.
"""
import argparse
import base64
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import ml_tts as T

API = "https://texttospeech.googleapis.com/v1"
LOCALES = {"en": ["en-GB", "en-AU", "en-IN", "en-US"], "de": ["de-DE"], "es": ["es-ES", "es-US"], "fr": ["fr-FR", "fr-CA"], "it": ["it-IT"], "nl": ["nl-NL", "nl-BE"],
           "pt": ["pt-BR"], "uk": ["uk-UA"], "ru": ["ru-RU"]}
DEFAULT_CLASSES = ("price,price_big,clock24,clock12,date_num,date_txt,phone,code,roman,version,temp,frac,permille,"
                   "range_from,range_between,unit,ord,decade,amp,dec,pct")
SPEEDS = [(0.9, 1), (1.0, 2), (1.15, 2), (1.35, 1)]
SPEEDS_YEAR = [(0.95, 1), (1.0, 2), (1.1, 1)]  # years at normal rates: fast synthetic years hurt real ones
DEV_VOICES = 3  # voices per language kept for dev rows only
# English: accent variety (UK and Australian weighted first), shapes real English lacks; never years / counts
EN_SLOTS = ("money,clock,phone,date,roman,fraction,version,negative,measure,decimal,code,alnum,identifier,ip,port,score,"
            "quarter,basis_points,multiplier,thousands_k,slash,range,ordinal,percent,big")
EN_LOCALE_W = {"en-GB": 4, "en-AU": 4, "en-IN": 2, "en-US": 1}


class Auth:
    def __init__(self):
        self.lock, self.tok, self.t = threading.Lock(), None, 0.0
        self.project = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True,
                                      text=True).stdout.strip()

    def headers(self):
        with self.lock:
            if not self.tok or time.time() - self.t > 1800:
                self.tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                          text=True, check=True).stdout.strip()
                self.t = time.time()
            return {"Authorization": f"Bearer {self.tok}", "x-goog-user-project": self.project,
                    "Content-Type": "application/json"}


def _req(auth, url, body=None):
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=auth.headers(),
                               method="POST" if body is not None else "GET")
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.loads(f.read())


def voices(auth, lang):
    out = []
    for loc in LOCALES[lang]:
        d = _req(auth, f"{API}/voices?languageCode={loc}")
        out += sorted((v["name"], loc) for v in d.get("voices", []) if "Chirp3-HD" in v["name"]
                      and loc in v.get("languageCodes", [loc]))
    return out


# a written day said as an ordinal before the month ("zesde maart"): unusual in everyday Dutch speech
ORD_DAY = {"nl": re.compile(r"(?:de|ste)$")}


def natural_date(r: dict, lang: str) -> bool:
    """A date row read with a month NAME and, where ORD_DAY covers the language, no ordinal day ("zeventien maart",
    not "zeventiende drie" / "zesde maart"); the row's date spans are its pieces (day, month, year). Rows without
    date spans always pass."""
    import ml_cands as M

    ws = [w for sp in r.get("spans", []) if sp.get("kind") == "date" for w in (sp.get("reading") or "").lower().split()]
    if not ws:
        return True
    if not set(M.MONTHS.get(lang, "").split()) & set(ws):
        return False
    return not (lang in ORD_DAY and any(ORD_DAY[lang].search(w) for w in ws))


def plan_per_class(lang: str, class_n: dict, seed: int, rep: dict, tag: str, natural_dates: bool = False) -> list[dict]:
    """ml_tts.plan once per class until that class has its row count (a few seeds if the filter drops rows); rows with
    a label already planned are dropped; natural_dates keeps only natural_date rows."""
    rows, seen = [], set()
    for cls, n in class_n.items():
        got = 0
        for k in range(4):
            if got >= int(n):
                break
            r_ = {}
            for r in T.plan(lang, 3 * (int(n) - got) if natural_dates else int(n) - got, seed * 10 + k, r_, False,
                            (cls,), tag):
                if got >= int(n) or r["text"] in seen or (natural_dates and not natural_date(r, lang)):
                    continue
                seen.add(r["text"])
                rows.append(r)
                got += 1
        rep[cls] = got
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="de,es,fr,it,nl,pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="tts3")
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--work", default="/tmp/tts_cloud")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--year-quota", help="json {lang: {year: rows}}: a year-only pass that fills those values")
    ap.add_argument("--uk-classes", default="", help="Ukrainian / Russian: only these ml_tts_uk / ml_tts_ru classes, equal shares (pilot)")
    ap.add_argument("--rm-local", action="store_true", help="delete the local wavs once uploaded")
    ap.add_argument("--class-n", default="", help='json {class: rows}: plan each class to its own row count '
                    '(instead of --n split equally over --classes)')
    ap.add_argument("--count-quota", help="json {lang: {value: rows}}: weights the count values (ml_tts.COUNT_QUOTA)")
    ap.add_argument("--normal-rates", action="store_true", help="normal speaking rates only (SPEEDS_YEAR)")
    ap.add_argument("--natural-dates", action="store_true", help="with --class-n: dates read with a month name only")
    a = ap.parse_args()
    auth = Auth()
    total_chars = 0
    yq = json.load(open(a.year_quota)) if a.year_quota else {}
    for lang in a.langs.split(","):
        if yq:
            T.YEAR_QUOTA = {int(y): k for y, k in yq.get(lang, {}).items()}
            a.n, a.classes = sum(T.YEAR_QUOTA.values()), "year"
        vs = voices(auth, lang)
        rng = random.Random(f"{a.tag}|{a.seed}|{lang}")
        rng.shuffle(vs)
        dev_v, train_v = vs[:DEV_VOICES], vs[DEV_VOICES:]
        rep = {}
        if lang == "en":
            import gen
            rows = [dict(r, id=f"{a.tag}_en_{r['id']}", cls=",".join(r["kinds"]), synth_text=r["tts_text"])
                    for r in gen.rows(a.n, a.seed, dev_frac=0.2, control_frac=0.05, multi_frac=0.15, slots=EN_SLOTS)]
            by_loc = {}
            for v in vs:
                by_loc.setdefault(v[1], []).append(v)
            locs = [l for l in EN_LOCALE_W if l in by_loc]
            dev_v = [by_loc[l][0] for l in locs]  # one held-out voice per accent
            train_v = [v for l in locs for v in by_loc[l][1:]]
        elif lang in ("uk", "ru"):  # case-fixed carrier sentences (ml_tts_uk / ml_tts_ru)
            import importlib

            slav = importlib.import_module(f"ml_tts_{lang}")
            rows = slav.plan(a.n, a.seed, rep, tuple(c for c in a.uk_classes.split(",") if c), a.tag)
        elif a.class_n:
            T.COUNT_QUOTA = {int(v): k for v, k in (json.load(open(a.count_quota)).get(lang, {}) if a.count_quota
                                                     else {}).items()}
            try:
                rows = plan_per_class(lang, json.loads(a.class_n), a.seed, rep, a.tag, a.natural_dates)
            finally:
                T.COUNT_QUOTA = {}
        else:
            rows = T.plan(lang, a.n, a.seed, rep, False, tuple(c for c in a.classes.split(",") if c), a.tag)
        w = os.path.join(a.work, lang)
        os.makedirs(w, exist_ok=True)
        speeds, wts = zip(*(SPEEDS_YEAR if (yq or lang in ("uk", "ru") or a.normal_rates) else SPEEDS))  # uk / ru: normal rates only
        for r in rows:
            pool = dev_v if r["split"] == "dev" else train_v
            if lang == "en":  # accent weights, then a voice of that accent
                loc = rng.choices(sorted({l for _, l in pool}), [EN_LOCALE_W[l] for l in sorted({l for _, l in pool})])[0]
                pool = [v for v in pool if v[1] == loc]
            name, loc = rng.choice(pool)
            r.update(engine="presynth", voice=name, voice_license="Google Cloud Text-to-Speech output", voice_path=None,
                     speaker=None, speed=rng.choices(speeds, wts)[0], locale=loc,
                     presynth=f"/vol/ml/{a.tag}/{lang}/presynth/{r['id']}.wav")

        def synth(r):
            p = os.path.join(w, r["id"] + ".wav")
            if os.path.exists(p):
                return True
            body = {"input": {"text": r.get("synth_text") or r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
            for attempt in range(4):
                try:
                    d = _req(auth, f"{API}/text:synthesize", body)
                    open(p, "wb").write(base64.b64decode(d["audioContent"]))
                    return True
                except Exception:  # noqa: BLE001  (rate limits / transient errors: back off)
                    time.sleep(2 ** attempt)
            return False

        with ThreadPoolExecutor(a.threads) as ex:
            ok = list(ex.map(synth, rows))
        rows = [r for r, k in zip(rows, ok) if k]
        chars = sum(len(r["text"]) for r in rows)
        total_chars += chars
        with open(os.path.join(w, "planned.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", w + "/",
                        f"ml/{a.tag}/{lang}/presynth/"], cwd=here, check=True, capture_output=True)
        subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                        f"ml/{a.tag}/{lang}/planned.jsonl"], cwd=here, check=True, capture_output=True)
        if a.rm_local:
            import shutil

            shutil.rmtree(w, ignore_errors=True)
        print(json.dumps({"lang": lang, "rows": len(rows), "failed": ok.count(False), "chars": chars,
                          "voices_train": len(train_v), "voices_dev": len(dev_v), "plan": rep}, ensure_ascii=False)[:600])
    print("total chars", total_chars, "≈ $", round(total_chars / 1e6 * 30, 2), "(Chirp 3 HD list price)")


if __name__ == "__main__":
    main()
