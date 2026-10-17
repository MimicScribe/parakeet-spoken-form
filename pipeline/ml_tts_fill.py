"""Targeted synthetic top-up against census gaps (2026-10-17): given exact year / count values per language, plan rows,
voice them with cloud voices (Chirp 3 HD, ml_tts_cloud) and, for up to `--kokoro-share` of the NON-year rows of a
class, with local Kokoro voices (voice variety at no API cost). Years are never voiced by Kokoro (lesson v12: years
learned from one clean voice hurt real speech); every rate is a normal speaking rate.

    python ml_tts_fill.py plan.json --tag <tag> --pilot 20 --dry        # rows + characters + cost, nothing synthesised
    python ml_tts_fill.py plan.json --tag <tag> --pilot 20               # pilot: plan, synthesise, upload
    modal run ml_tts.py --langs es,fr --out-tag <tag> --rows-file ml/<tag>/{lang}/planned.jsonl   # verify es / fr (CTC)
    modal run ml_tts_fill.py::kokoro_en --tag <tag>                      # English Kokoro rows -> presynth wavs

plan.json: {"es": {"year": {"2028": 9}, "count": {"61": 4}}, "en": {...}}. Planned rows go to ml/<tag>/<lang>/planned.jsonl
(engine "presynth" for cloud clips, "kokoro" for local ones; English Kokoro rows also get a presynth path that
kokoro_en fills). English has no CTC reader in ml_decide: English clips are checked against a second transcript
(the same check as the earlier English cloud set) before they enter a manifest (finalize_en).
"""
import argparse
import collections
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/vol"
SPEEDS = [(0.95, 1), (1.0, 2), (1.1, 1)]  # normal speaking rates only (ml_tts_cloud.SPEEDS_YEAR)
# English Kokoro voices (the ones the earlier English sets used); two are held out for dev rows
EN_KOKORO_TRAIN = ("af_heart", "af_bella", "af_sarah", "af_sky", "am_adam", "am_michael", "am_onyx", "bf_emma",
                   "bf_isabella", "bf_lily", "bm_george", "bm_daniel")
EN_KOKORO_DEV = ("af_nicole", "bm_lewis")
KOKORO_LICENSE = "Apache-2.0 (Kokoro-82M weights)"
MAX_KOKORO_SHARE = 1 / 3  # 2026-10-17: cloud voices stay the main source


def assign_engines(rows: list[dict], share: float, rng: random.Random, kokoro_voices) -> None:
    """Per class (row["cls"]): up to `share` of the rows get engine "kokoro" (never year rows, never above 1/3).
    kokoro_voices(split) -> list of voice ids for that split (empty: that split stays on cloud voices)."""
    share = min(share, MAX_KOKORO_SHARE)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    for cls, rs in by.items():
        if cls == "year":
            continue
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        n = int(len(rs) * share)
        for i in idx:
            if n <= 0:
                break
            vs = kokoro_voices(rs[i]["split"])
            if not vs:
                continue
            rs[i].update(engine="kokoro", voice=rng.choice(vs), voice_license=KOKORO_LICENSE, voice_path=None,
                         speaker=None)
            n -= 1


def en_rows(quota: dict, tag: str, seed: int) -> list[dict]:
    import gen

    out = []
    q = {k: v for k, v in quota.items() if k in ("year", "count")}
    for r in gen.targeted_rows(q, seed):
        out.append(dict(r, id=f"{tag}_en_{r['id']}", cls=r["target"], synth_text=r["tts_text"], lang="en"))
    # "ukau": {form: n} -> UK / Australian spoken forms (ml_ukau), voiced by en-GB / en-AU voices only
    if quota.get("ukau"):
        import ml_ukau

        for r in ml_ukau.rows(quota["ukau"], seed):
            out.append(dict(r, id=f"{tag}_en_{r['id']}", synth_text=r["tts_text"], lang="en", locales=["en-GB", "en-AU"]))
    # "slots": {"phone": n} -> n generic gen.rows rows of that slot kind (carriers.txt + carriers_<kind>.txt if present)
    for kind, n in (quota.get("slots") or {}).items():
        files = ["carriers.txt"] + ([f"carriers_{kind}.txt"] if os.path.exists(os.path.join(HERE, f"carriers_{kind}.txt")) else [])
        per = -(-int(n) // len(files))
        for fi, cf in enumerate(files):
            for r in gen.rows(per, seed + 31 * fi, dev_frac=0.2, control_frac=0.0, slots=kind, carriers=cf):
                out.append(dict(r, id=f"{tag}_en_{kind}{fi}_{r['id']}", cls=kind, synth_text=r["tts_text"], lang="en"))
    return out


def wave_rows(lang: str, quota: dict, tag: str, seed: int, rep: dict) -> list[dict]:
    """ml_tts.plan rows with the value quotas set (year: ml_tts.YEAR_QUOTA, count: ml_tts.COUNT_QUOTA)."""
    import ml_tts as T

    out = []
    for kind, cls in (("year", "year"), ("count", "count")):
        q = {int(v): int(n) for v, n in (quota.get(kind) or {}).items() if int(n) > 0}
        if not q:
            continue
        # the quota only weights the draw: plan more and keep exactly q[v] rows per value (a few seeds if short)
        have, seen = collections.Counter(), set()
        try:
            for k in range(6):
                left = {v: n - have[v] for v, n in q.items() if n > have[v]}
                if not left:
                    break
                T.YEAR_QUOTA, T.COUNT_QUOTA = (left, {}) if kind == "year" else ({}, left)
                for r in T.plan(lang, 3 * sum(left.values()) + 5, seed * 100 + k, {}, False, (cls,), tag):
                    v = _value(r)
                    if v in left and have[v] < q[v] and r["text"] not in seen:
                        seen.add(r["text"])
                        have[v] += 1
                        out.append(r)
        finally:
            T.YEAR_QUOTA, T.COUNT_QUOTA = {}, {}
        rep[kind] = {"wanted": sum(q.values()), "planned": sum(have.values()),
                     "short": {v: n - have[v] for v, n in q.items() if n > have[v]}}
    return out


def _value(r: dict):
    for s in r.get("spans", []):
        t = s["text"].split()[0].replace(".", "").replace("\u202f", "")
        if t.isdigit():
            return int(t)
    return None


def value_counts(rows: list[dict], lang: str) -> dict:
    """{cls: {value: rows}} of planned rows (what the plan actually voices)."""
    c = collections.defaultdict(collections.Counter)
    for r in rows:
        if lang == "en":
            c[r["cls"]][r.get("value", "-")] += 1
        elif _value(r) is not None:
            c[r["cls"]][_value(r)] += 1
    return {k: dict(sorted(v.items())) for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--kokoro-share", type=float, default=MAX_KOKORO_SHARE)
    ap.add_argument("--pilot", type=int, default=0, help="only the first N rows per language (pilot)")
    ap.add_argument("--langs", default="")
    ap.add_argument("--work", default="/tmp/tts_fill")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--keep-local", action="store_true", help="keep local wavs (English: needed for the second transcript)")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    langs = [l for l in (a.langs.split(",") if a.langs else plan)]
    total_chars = 0
    auth = None
    for lang in langs:
        rng = random.Random(f"{a.tag}|{a.seed}|{lang}")
        rep = {}
        rows = en_rows(plan[lang], a.tag, a.seed) if lang == "en" else wave_rows(lang, plan[lang], a.tag, a.seed, rep)
        rng.shuffle(rows)
        if a.pilot:
            rows = rows[:a.pilot]
        import ml_tts as T

        if lang == "en":
            kv = lambda split: list(EN_KOKORO_DEV if split == "dev" else EN_KOKORO_TRAIN)  # noqa: E731
        else:
            kv = lambda split, l=lang: [v["id"] for v in T.VOICES[l] if v["engine"] == "kokoro"  # noqa: E731
                                        and v["dev"] == (split == "dev")]
        for r in rows:
            r["engine"] = "presynth"
        assign_engines(rows, a.kokoro_share, rng, kv)
        chirp = [r for r in rows if r["engine"] == "presynth"]
        chars = sum(len(r.get("synth_text") or r["text"]) for r in chirp)
        total_chars += chars
        eng = collections.Counter((r["cls"], r["engine"]) for r in rows)
        print(json.dumps({"lang": lang, "rows": len(rows), "engines": {f"{c}/{e}": n for (c, e), n in sorted(eng.items())},
                          "cloud_chars": chars, "usd_cloud": round(chars / 1e6 * 30, 2), "plan_got": rep,
                          "values": value_counts(rows, lang)}, ensure_ascii=False)[:1500])
        if a.dry:
            continue
        import ml_tts_cloud as CL

        auth = auth or CL.Auth()
        vs = CL.voices(auth, lang)
        rng.shuffle(vs)
        if lang == "en":  # one held-out voice per accent (ml_tts_cloud), accent weights for the rest
            by_loc = collections.defaultdict(list)
            for v in vs:
                by_loc[v[1]].append(v)
            locs = [l for l in CL.EN_LOCALE_W if l in by_loc]
            dev_v, train_v = [by_loc[l][0] for l in locs], [v for l in locs for v in by_loc[l][1:]]
        else:
            dev_v, train_v = vs[:CL.DEV_VOICES], vs[CL.DEV_VOICES:]
        speeds, wts = zip(*SPEEDS)
        for r in rows:
            r["speed"] = rng.choices(speeds, wts)[0]
            r["presynth"] = f"{VOL}/ml/{a.tag}/{lang}/presynth/{r['id']}.wav"
            r.setdefault("pitch", 1.0)
            if r["engine"] != "presynth":
                r["locale"] = None
                continue
            pool = dev_v if r["split"] == "dev" else train_v
            if lang == "en":
                if r.get("locales"):
                    pool = [v for v in pool if v[1] in r["locales"]]
                ls = sorted({l for _, l in pool})
                loc = rng.choices(ls, [CL.EN_LOCALE_W[l] for l in ls])[0]
                pool = [v for v in pool if v[1] == loc]
            name, loc = rng.choice(pool)
            r.update(voice=name, voice_license="Google Cloud Text-to-Speech output", voice_path=None, speaker=None,
                     locale=loc)
        w = os.path.join(a.work, a.tag, lang)
        os.makedirs(w, exist_ok=True)
        import base64
        import time
        from concurrent.futures import ThreadPoolExecutor

        def synth(r):
            p = os.path.join(w, r["id"] + ".wav")
            if os.path.exists(p):
                return True
            body = {"input": {"text": r.get("synth_text") or r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
            for attempt in range(4):
                try:
                    d = CL._req(auth, f"{CL.API}/text:synthesize", body)
                    open(p, "wb").write(base64.b64decode(d["audioContent"]))
                    return True
                except Exception:  # noqa: BLE001  (rate limits / transient errors: back off)
                    time.sleep(2 ** attempt)
            return False

        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(synth, chirp))
        bad = {r["id"] for r, k in zip(chirp, ok) if not k}
        rows = [r for r in rows if r["id"] not in bad]
        with open(os.path.join(w, "planned.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        import subprocess

        wavs = [p for p in os.listdir(w) if p.endswith(".wav")]
        if wavs:
            subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", w + "/",
                            f"ml/{a.tag}/{lang}/presynth/"], cwd=HERE, check=True, capture_output=True)
        subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                        f"ml/{a.tag}/{lang}/planned.jsonl"], cwd=HERE, check=True, capture_output=True)
        if not (a.keep_local or lang == "en"):
            for p in wavs:
                os.remove(os.path.join(w, p))
        print(json.dumps({"lang": lang, "uploaded_rows": len(rows), "cloud_failed": len(bad)}))
    print("total cloud chars", total_chars, "≈ $", round(total_chars / 1e6 * 30, 2), "(Chirp 3 HD list price)")


# --------------------------------------------------------------------------- Modal: English Kokoro clips + manifest

try:
    import modal

    import ml_tts as _T

    if _T._MODAL:
        vol = _T.vol
        app = modal.App("parakeet-tts-fill", image=_T.image.add_local_python_source("mix_build", "common", "ml_tts_fill"))

        @app.function(gpu="L4", cpu=4, memory=16384, volumes={VOL: vol}, timeout=2 * 3600)
        def kokoro_en_fn(tag: str) -> dict:
            """Voice the planned English Kokoro rows to their presynth paths (16 kHz mono)."""
            import soundfile as sf

            rows = [json.loads(l) for l in open(f"{VOL}/ml/{tag}/en/planned.jsonl") if l.strip()]
            syn = _T._Synth()
            os.makedirs(f"{VOL}/ml/{tag}/en/presynth", exist_ok=True)
            n = bad = 0
            for r in rows:
                if r["engine"] != "kokoro":
                    continue
                x = syn(dict(r, text=r.get("synth_text") or r["text"]))
                if x is None:
                    bad += 1
                    continue
                sf.write(r["presynth"], x, 16000, subtype="PCM_16")
                n += 1
            vol.commit()
            return {"kokoro_written": n, "failed": bad}

        @app.function(cpu=4, memory=8192, volumes={VOL: vol}, timeout=3600)
        def finalize_en_fn(tag: str, keep_ids: list, aug_frac: float = 0.6, seed: int = 47) -> dict:
            """English manifest d1/en4_ok/<tag>_en_{train,dev}.jsonl from the rows the second transcript confirmed;
            aug_frac of the train rows point at a degraded copy (mix_build.Degrader, as the wave-1 synthetic rows)."""
            import soundfile as sf

            import mix_build as MB

            keep = set(keep_ids)
            rows = [json.loads(l) for l in open(f"{VOL}/ml/{tag}/en/planned.jsonl") if l.strip()]
            deg = MB.Degrader(f"{VOL}/slr28/RIRS_NOISES", seed)
            rng = random.Random(seed)
            os.makedirs(f"{VOL}/ml/{tag}/en/aug", exist_ok=True)
            out = {"train": [], "dev": []}
            for r in rows:
                if r["id"] not in keep or not os.path.exists(r["presynth"]):
                    continue
                info = sf.info(r["presynth"])
                x = {"audio_filepath": r["presynth"], "duration": round(info.frames / info.samplerate, 3), "text": r["text"],
                     "id": r["id"], "lang": "en", "origin": "tts", "src": "chirp3hd" if r["engine"] == "presynth" else "kokoro",
                     "engine": r["engine"], "split": r["split"], "kinds": r["kinds"], "template": r["template"],
                     "cls": r["cls"], "value": r.get("value"), "voice": r["voice"], "locale": r.get("locale"),
                     "speed": r["speed"]}
                if r["split"] == "train" and rng.random() < aug_frac:
                    o = f"{VOL}/ml/{tag}/en/aug/{r['id']}.wav"
                    x["aug"] = deg(r["presynth"], o)
                    x["audio_filepath"] = o
                out[r["split"]].append(x)
            for k, xs in out.items():
                with open(f"{VOL}/d1/en4_ok/{tag}_en_{k}.jsonl", "w", encoding="utf-8") as f:
                    for x in xs:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
            vol.commit()
            return {k: len(v) for k, v in out.items()}

        @app.local_entrypoint()
        def kokoro_en(tag: str):
            print(kokoro_en_fn.remote(tag))

        @app.local_entrypoint()
        def finalize_en(tag: str, keep_file: str):
            """keep_file: local json list of row ids the second transcript confirmed."""
            print(finalize_en_fn.remote(tag, json.load(open(keep_file))))
except ImportError:
    pass


if __name__ == "__main__":
    main()
