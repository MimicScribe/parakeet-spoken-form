"""Targeted synthetic top-up with varied, natural carrier sentences and 2-3 number slots per sentence (cheaper per span).

Reads a plan, writes sentences from `carriers_topup.json` (per language: clauses with {a} {b} {c} slots, leads, tails),
realises every written sentence through the same candidate machinery as ml_tts (ml_cands readings, reading_ok,
grammar rules), voices it with cloud voices at normal speaking rates (the language's held-out voice never voices a
train row; the 3 dev voices keep the usual dev rows) and uploads clips + planned rows to NEW volume paths only (a path
that exists is an error, nothing is overwritten). The rows are then verified exactly like every cloud set:

    python ml_tts_topup.py plan.json --tag ttsNL1 --dry                 # rows, characters, cost, samples; no network
    python ml_tts_topup.py plan.json --tag ttsNL1 --pilot 24            # synthesise + upload the first 24 rows
    modal run ml_tts.py --langs nl --out-tag <tag>v --rows-file "ml/<tag>v/{lang}/planned.jsonl"   # CTC verify

plan.json: {"lang": "nl", "seed": 21, "classes": {
   "year":      {"values": {"1993": 12, "2044": 6}},        # exact year values (full reading only), 2-3 per sentence
   "count":     {"values": {"58": 7, "73": 7}},              # exact plain counts, two per sentence
   "count_big": {"n": 70, "avoid": [1234, ...]},             # n NEW distinct counts 1000-999999 (never 1100-2099)
   "dec": 140, "price": 110, "price_big": 40, "clock24": 48, "phone": 60, "decade": 50}}   # a number = spans wanted
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess

import ml_tts as T

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/vol"
SPEEDS = [(0.95, 1), (1.0, 2), (1.1, 1)]  # normal speaking rates only (ml_tts_cloud.SPEEDS_YEAR)
# class -> (carrier group(s) by slots per sentence, ml_tts class, ml_tts group for the reading rules)
CARRIERS = {"year": ("year", {2: "year2", 3: "year3"}), "count": ("count", {2: "count2"}),
            "count_big": ("count", {2: "big2"}), "dec": ("dec", {2: "dec2"}),
            "price": ("price", {2: "price2"}), "price_big": ("price_big", {2: "pricebig2"}),
            "clock24": ("clock24", {2: "time2"}), "phone": ("phone", {1: "phone1", 2: "phone2"}),
            "decade": ("decade", {2: "decade2"})}
SLOTS = "abc"


def load_carriers(lang: str) -> dict:
    with open(os.path.join(HERE, "carriers_topup.json"), encoding="utf-8") as f:
        return json.load(f)[lang]


def _slot(lang, cls, rng, P):
    """One written slot of class cls (the ml_tts slot generators)."""
    if cls == "dec":
        lo = 100 if rng.random() < 0.2 else 0
        return T._dec(rng, P, lo, 999 if lo else 99, False)
    return T.GEN[cls](lang, rng, P, 0, False, None)


def _distinct_big(rng, n, avoid, P):
    """n distinct counts 1000-999999 the grammar table can read (never year-shaped 1100-2099, never in `avoid`)."""
    pool = [v for v in P["card"] if 1000 <= v <= 999999 and not 1100 <= v <= 2099 and v not in set(avoid)
            and v % 10 not in (1, 2) and v % 100 != 0]  # un / twee endings agree with the noun; round numbers add nothing
    rng.shuffle(pool)
    return pool[:n]


def _value_sentences(values: dict, rng, kinds):
    """Group a {value: rows} multiset into sentences of 2-3 distinct values, the neediest first (a lone last value is
    paired with a random other value of the plan)."""
    need = {int(v): int(n) for v, n in values.items() if int(n) > 0}
    every = sorted(need)
    groups = []
    while need:
        k = min(rng.choice(kinds), max(2, len(need)))
        vs = rng.sample(sorted(need, key=lambda v: -need[v])[:max(k, 8)], min(k, len(need)))
        while len(vs) < k:
            v = rng.choice(every)
            if v not in vs:
                vs.append(v)
        groups.append(vs)
        for v in vs:
            if v in need:
                need[v] -= 1
                if need[v] <= 0:
                    del need[v]
    return groups


def sentence(lang, cls_key, slots, rng, C, tag):
    """(written, ml_tts class) for the slot texts; None when it is too short / long."""
    cls, by_n = CARRIERS[cls_key]
    group = by_n[len(slots)]
    clause = rng.choice(C["clauses"][group])
    for s, t in zip(SLOTS, slots):
        clause = clause.replace("{" + s + "}", t)
    reg, lead = rng.choice(C["leads"])
    tail = rng.choice(C["tails"])
    text = f"{lead} {clause}{tail}." if lead else f"{clause[0].upper()}{clause[1:]}{tail}."
    if not (T.WORDS_MIN <= len(text.split()) <= T.WORDS_MAX):
        return None
    return text, cls, reg


def plan_rows(plan: dict, tag: str, rep: dict) -> list:
    lang, seed = plan["lang"], int(plan.get("seed", 21))
    C, P = load_carriers(lang), T.pools(lang)
    rng = random.Random(f"{tag}|{seed}|{lang}")
    used, seen, rows = collections.Counter(), set(), []
    for cls_key, spec in plan["classes"].items():
        cls, by_n = CARRIERS[cls_key]
        sents = []  # one list of slot texts per sentence
        if cls_key in ("year", "count"):
            for g in _value_sentences(spec["values"], rng, [2, 3] if cls_key == "year" else [2]):
                sents.append([str(v) for v in g])
        elif cls_key == "count_big":
            vs = _distinct_big(rng, int(spec["n"]), spec.get("avoid", []), P)
            sents = [[str(vs[i]), str(vs[i + 1])] for i in range(0, len(vs) - 1, 2)]
        else:
            n_sl = max(by_n)
            single = isinstance(spec, dict)  # {"n": spans, "single": true}: one number per sentence (phones are heard better)
            want = int(spec["n"] if single else spec)
            for _ in range(-(-want // n_sl)):
                k = 1 if (single or (cls_key == "phone" and rng.random() >= 0.4)) else n_sl
                sents.append([_slot(lang, cls, rng, P) for _ in range(k)])
        got = 0
        q_year = {2000: 1} if cls_key == "year" else {}
        for sl in sents:
            ok = None
            for _ in range(8):  # a few carriers / voices of the realisation before giving the sentence up
                bt = sentence(lang, cls_key, sl, rng, C, tag)
                if bt is None:
                    rep.setdefault(cls_key, collections.Counter())["length"] += 1
                    continue
                written, rcls, reg = bt
                T.YEAR_QUOTA = q_year  # a non-empty quota makes reading_ok keep the full year reading only
                try:
                    res, why = T.realize(lang, rcls, written, T.GROUP[rcls], None, rng, used)
                finally:
                    T.YEAR_QUOTA = {}
                if res is None:
                    rep.setdefault(cls_key, collections.Counter())[why] += 1
                    continue
                ok = (written, rcls, reg, res)
                break
            if ok is None:
                continue
            written, rcls, reg, (label, spans, offs) = ok
            if label in seen:
                rep.setdefault(cls_key, collections.Counter())["duplicate"] += 1
                continue
            seen.add(label)
            rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
            rows.append({"id": rid, "lang": lang, "cls": rcls, "ckey": cls_key, "register": reg, "caption": written,
                         "text": label, "spans": spans, "num_chars": offs, "kind": sorted({s["kind"] for s in spans}),
                         "split": "dev" if rng.random() < T.DEV_FRAC else "train"})
            got += 1
        rep.setdefault(cls_key, collections.Counter())["rows"] = got
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--tag", required=True, help="the set tag; clips + planned rows go to ml/<tag>v/<lang>/ (new paths)")
    ap.add_argument("--pilot", type=int, default=0, help="only the first N rows (pilot)")
    ap.add_argument("--skip", type=int, default=0, help="leave the first N rows out (they were a pilot chunk)")
    ap.add_argument("--vtag", default="", help="scratch tag of the clips + planned rows (default <tag>v); one per chunk")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--work", default="/tmp/tts_topup")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    lang = plan["lang"]
    rep: dict = {}
    rows = plan_rows(plan, a.tag, rep)
    rows = rows[a.skip:]
    if a.pilot:
        rows = rows[:a.pilot]
    chars = sum(len(r["text"]) for r in rows)
    by = collections.Counter(r["ckey"] for r in rows)
    spans = collections.Counter(r["ckey"] for r in rows for _ in r["spans"])
    print(json.dumps({"rows": len(rows), "by_class": dict(by), "spans": dict(spans), "chars": chars,
                      "usd_list": round(chars / 1e6 * 30, 2), "plan_drops": {k: dict(v) for k, v in rep.items()}}))
    for r in rows[:6]:
        print("  ", r["ckey"], "|", r["caption"], "|", r["text"])
    if a.dry:
        return
    import base64
    import time
    from concurrent.futures import ThreadPoolExecutor

    import ml_tts_cloud as CL
    from mix_build import is_heldout_voice

    auth = CL.Auth()
    vs = CL.voices(auth, lang)
    rng = random.Random(f"{a.tag}|{plan.get('seed', 21)}|{lang}")
    rng.shuffle(vs)
    dev_v = vs[:CL.DEV_VOICES]
    train_v = [v for v in vs[CL.DEV_VOICES:] if not is_heldout_voice(lang, v[0])]
    assert train_v, "no train voice left once the held-out voice is out"
    speeds, wts = zip(*SPEEDS)
    vtag = a.vtag or a.tag + "v"
    for r in rows:
        name, loc = rng.choice(dev_v if r["split"] == "dev" else train_v)
        r.update(engine="presynth", voice=name, voice_license="Google Cloud Text-to-Speech output", voice_path=None,
                 speaker=None, speed=rng.choices(speeds, wts)[0], locale=loc, pitch=1.0,
                 presynth=f"{VOL}/ml/{vtag}/{lang}/presynth/{r['id']}.wav")
    assert not any(is_heldout_voice(lang, r["voice"]) and r["split"] != "dev" for r in rows)
    w = os.path.join(a.work, vtag, lang)
    os.makedirs(w, exist_ok=True)

    def synth(r):
        p = os.path.join(w, r["id"] + ".wav")
        if os.path.exists(p):
            return True
        body = {"input": {"text": r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
        for attempt in range(4):
            try:
                d = CL._req(auth, f"{CL.API}/text:synthesize", body)
                open(p, "wb").write(base64.b64decode(d["audioContent"]))
                return True
            except Exception:  # noqa: BLE001  (rate limits / transient errors: back off)
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(a.threads) as ex:
        ok = list(ex.map(synth, rows))
    rows = [r for r, k in zip(rows, ok) if k]
    with open(os.path.join(w, "planned.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # new paths only: no --force, an existing path is an error
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", w + "/", f"ml/{vtag}/{lang}/presynth/"],
                   cwd=HERE, check=True, capture_output=True)
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                    f"ml/{vtag}/{lang}/planned.jsonl"], cwd=HERE, check=True, capture_output=True)
    print(json.dumps({"uploaded_rows": len(rows), "cloud_failed": ok.count(False), "chars": chars,
                      "usd_list": round(chars / 1e6 * 30, 2)}))


if __name__ == "__main__":
    main()
