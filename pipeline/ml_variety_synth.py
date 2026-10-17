"""Variety synthetic sets (2026-10-16): look-alike minimal pairs / idioms / controls (ml_variety_bank.LOOK, every
language) and business shapes real speech lacks (ml_variety_bank.BIZ, wave 1: Q3 / H1 / T3 / S1, "2,5 Mrd. €" /
"3,2 M€" / "2,5 mld €", "4 x 3 Meter"). Voiced with Google Cloud Chirp 3 HD voices (ml_tts_cloud) at the four speaking
rates 0.9 / 1.0 / 1.15 / 1.35, one voice per rendition; held-out sentences use held-out voices only.

Labels: wave-1 rows go through ml_cands + ml_tts.realize like every synthetic row (the voice says the label); the other
languages carry the bank's hand-written label. Never a year or a plain count (ml_tts_cloud rule).

    python ml_variety_synth.py --tag tvar1 --langs fr,de --dry          # plan only, print rows + character count
    python ml_variety_synth.py --tag tvar1                               # plan, synthesise, upload
    modal run ml_tts.py --langs de,es,fr,it,nl,pt --out-tag tvar1 --rows-file ml/tvar1/{lang}/planned.jsonl  # verify
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import ml_variety_bank as B

WAVE1 = ("de", "es", "fr", "it", "nl", "pt")
LOCALES = {"en": ["en-GB", "en-AU", "en-IN", "en-US"], "de": ["de-DE"], "es": ["es-ES", "es-US"], "fr": ["fr-FR", "fr-CA"],
           "it": ["it-IT"], "nl": ["nl-NL", "nl-BE"], "pt": ["pt-BR"], "ro": ["ro-RO"], "pl": ["pl-PL"], "ru": ["ru-RU"],
           "uk": ["uk-UA"]}
RATES = (0.9, 1.0, 1.15, 1.35)
DEV_VOICES = 3
DEV_FRAC = 0.2  # share of sentences held out (voiced by the held-out voices only)
COMMA = {"de": ",", "es": ",", "fr": ",", "it": ",", "nl": ",", "pt": ","}
# the ml_tts class whose expected span kinds a sentence's spans meet (no class-specific reading filters)
_CLS_BY_KIND = (("cur", "price"), ("clock", "clock24"), ("pct", "pct"), ("unit", "unit"), ("alnum", "code"),
                ("ord", "date_txt"), ("dec", "large"), ("int", "large"))


def _slots(rng, lang):
    d = f"{rng.randint(2, 9)}{COMMA.get(lang, ',')}{rng.randint(1, 9)}"
    return {"Q": rng.randint(1, 4), "H": rng.randint(1, 2), "D": d, "M": rng.randint(2, 12), "A": rng.randint(2, 9),
            "B": rng.randint(10, 90)}


def _realize(lang, written, rng, used):
    """(label, spans, num_chars, kinds) for a wave-1 written sentence, or (None, reason). Every span takes its
    CANONICAL reading (ml_cands cands[0]): a minimal pair or a business shape needs no reading variety, and some
    variants are unsafe to voice ("2,40 €" -> "zwei und vierzig" sounds like 42)."""
    import ml_cands as M
    import ml_p3_core as C
    import ml_tts as T

    if not re.search(r"\d", written):
        return (written, [], [], []), ""
    sp = M.spans(written, lang)
    if not sp:
        return None, "no_spans"
    if any(not s.cands for s in sp):
        return None, "missing_grammar" if any(s.note.startswith("missing") for s in sp) else "no_cands"
    if not M.all_digits_spanned(written, lang):
        return None, "digits_unspanned"
    offs = []
    label = C.render_cap(written, sp, [0] * len(sp), offs)
    if written.endswith(".") and not label.endswith("."):
        label += "."
    if T.FORBIDDEN.search(label) or C.WF.search(label):
        return None, "symbol_left"
    toks, flags = label.split(), C.number_tokens(label, offs)
    for i, t in enumerate(toks):
        if not flags[i] and C.ABBR_NEAR.match(t.strip(",;:()")) and (
                (i > 0 and flags[i - 1]) or (i + 1 < len(toks) and flags[i + 1])):
            return None, "abbr_left"
    spans = [{"text": s.text, "kind": s.kind, "reading": s.cands[0].strip(), "canonical": s.cands[0].strip(), "idx": 0,
              "n_cands": len(s.cands)} for s in sp]
    return (label, spans, offs, sorted({s.kind for s in sp})), ""


def all_texts(lang: str) -> list[str]:
    """Every written sentence the plan can draw (grammar-table requests)."""
    out = [w for _, w, _ in B.LOOK.get(lang, [])]
    for tpls in B.BIZ.get(lang, {}).values():
        for t in tpls:
            for q in range(1, 5):
                for m in range(2, 13):
                    for a in range(2, 10):
                        out.append(t.format(Q=q, H=1 + q % 2, D=f"{a}{COMMA[lang]}{m % 10}", M=m, A=a, B=10 * a))
    return sorted(set(out))


def plan(lang: str, tag: str, seed: int, biz_per_class: int) -> tuple[list, dict]:
    rng = random.Random(f"{tag}|{seed}|{lang}")
    used = collections.Counter()
    rep = collections.Counter()
    sents = []  # (cls, written, label, spans, offs, kinds, renditions)
    for kind, written, spoken in B.LOOK.get(lang, []):
        if lang in WAVE1:
            res, why = _realize(lang, written, rng, used)
            if res is None:
                rep[f"look_drop_{why}"] += 1
                continue
            label, spans, offs, kinds = res
        else:
            label, spans, offs, kinds = spoken, [], [], []
        sents.append((f"look_{kind}", written, label, spans, offs, kinds, len(RATES)))
    for cls, tpls in B.BIZ.get(lang, {}).items():
        seen, tries = set(), 0
        while sum(1 for s in sents if s[0] == f"biz_{cls}") < biz_per_class and tries < 50 * biz_per_class:
            tries += 1
            written = rng.choice(tpls).format(**_slots(rng, lang))
            if written in seen:
                continue
            seen.add(written)
            res, why = _realize(lang, written, rng, used)
            if res is None:
                rep[f"biz_drop_{why}"] += 1
                continue
            label, spans, offs, kinds = res
            sents.append((f"biz_{cls}", written, label, spans, offs, kinds, 1))
    rows = []
    for cls, written, label, spans, offs, kinds, nrend in sents:
        h = hashlib.md5(f"{seed}|{lang}|{written}".encode()).hexdigest()[:10]
        # dev by template for the slot-filled business sentences ("Im Q1 ..." and "Im Q3 ..." share a split)
        unit = re.sub(r"\d", "#", written) if cls.startswith("biz_") else written
        dev = int(hashlib.md5(f"{seed}|{lang}|{unit}".encode()).hexdigest()[:4], 16) / 65535 < DEV_FRAC
        rates = list(RATES) if nrend == len(RATES) else [rng.choice(RATES)]
        for k, rate in enumerate(rates):
            rows.append({"id": f"{tag}_{lang}_{h}_{k}", "lang": lang, "cls": cls, "register": "variety", "caption": written,
                         "text": label, "spans": spans, "num_chars": offs, "kind": kinds, "split": "dev" if dev else "train",
                         "speed": rate, "engine": "presynth", "voice_license": "Google Cloud Text-to-Speech output",
                         "voice_path": None, "speaker": None})
        rep[cls] += 1
    return rows, dict(rep)


def _assign_voices(rows, vs, rng):
    rng.shuffle(vs)
    dev_v, train_v = vs[:DEV_VOICES], vs[DEV_VOICES:]
    by_sent = collections.defaultdict(list)
    for r in rows:
        by_sent[r["id"].rsplit("_", 1)[0]].append(r)
    for rs in by_sent.values():
        pool = dev_v if rs[0]["split"] == "dev" else train_v
        picks = rng.sample(pool, min(len(rs), len(pool)))  # a different voice for every rendition
        for r, (name, loc) in zip(rs, picks * 4):
            r.update(voice=name, locale=loc)


def main():
    import ml_tts_cloud as TC

    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="de,es,fr,it,nl,pt,en,ro,pl,ru,uk")
    ap.add_argument("--tag", default="tvar1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--biz-per-class", type=int, default=40)
    ap.add_argument("--work", default="/tmp/variety_synth")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    auth = None if a.dry else TC.Auth()
    total = 0
    for lang in a.langs.split(","):
        rows, rep = plan(lang, a.tag, a.seed, a.biz_per_class)
        chars = sum(len(r["text"]) for r in rows)
        total += chars
        if a.dry:
            print(json.dumps({"lang": lang, "rows": len(rows), "chars": chars, "plan": rep}, ensure_ascii=False))
            for r in rows[::7][:8]:
                print("   ", r["cls"], "|", r["caption"], "->", r["text"])
            continue
        vs = []
        for loc in LOCALES[lang]:
            d = TC._req(auth, f"{TC.API}/voices?languageCode={loc}")
            vs += sorted((v["name"], loc) for v in d.get("voices", []) if "Chirp3-HD" in v["name"]
                         and loc in v.get("languageCodes", [loc]))
        _assign_voices(rows, vs, random.Random(f"{a.tag}|voices|{lang}"))
        w = os.path.join(a.work, a.tag, lang)
        os.makedirs(w, exist_ok=True)

        def synth(r):
            p = os.path.join(w, r["id"] + ".wav")
            if os.path.exists(p):
                return True
            body = {"input": {"text": r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
            for attempt in range(4):
                try:
                    d = TC._req(auth, f"{TC.API}/text:synthesize", body)
                    open(p, "wb").write(base64.b64decode(d["audioContent"]))
                    return True
                except Exception:  # noqa: BLE001  (rate limits / transient errors: back off)
                    time.sleep(2 ** attempt)
            return False

        with ThreadPoolExecutor(a.threads) as ex:
            ok = list(ex.map(synth, rows))
        rows = [r for r, k in zip(rows, ok) if k]
        for r in rows:
            r["presynth"] = f"/vol/ml/{a.tag}/{lang}/presynth/{r['id']}.wav"
        with open(os.path.join(w, "planned.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", w + "/", f"ml/{a.tag}/{lang}/presynth/"],
                       cwd=here, check=True, capture_output=True)
        subprocess.run(["modal", "volume", "put", "--force", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                        f"ml/{a.tag}/{lang}/planned.jsonl"], cwd=here, check=True, capture_output=True)
        print(json.dumps({"lang": lang, "rows": len(rows), "failed": ok.count(False), "chars": chars, "voices": len(vs),
                          "plan": rep}, ensure_ascii=False))
    print("total chars", total, "≈ $", round(total / 1e6 * 30, 2), "(Chirp 3 HD list price)")


if __name__ == "__main__":
    main()
