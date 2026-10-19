"""CTC verification of adversarial-pack clips (adv_b rows voiced elsewhere; same checks as edge_a_ctc, inventory = adv_b): per clip the round-trip rule of the other synthetic
sets (CER <= ml_tts.CER_MAX, every number reading heard, word times) PLUS the strict probes of edge_a_probe on every edge
phrase. Writes NEW volume paths only:

    ml/<out>/<lang>/rows_<part>.jsonl         every checked clip, with the verdicts (kept = basic_ok and strict_ok)
    ml/<out>/<lang>/wav/<id>.wav              the level-normalised, padded clip of a kept row (or of every row: --write-all)
    ml/<out>/<lang>/stats_<part>.json

    modal run adv_b_ctc.py --planned ml/advB/plan/pilot/es/planned.jsonl --lang es --out advBc/pilot --parts 2 [--write-all]
"""
import json
import os
import random

import modal

import ml_tts as T

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = T.image.add_local_python_source("en_win", "label_gpu", "labeller", "readings", "census_common", "ml_slav", "ml_balance",
                                        "compound_split", "realx1_core", "edge_a_probe", "edge_a", "edge_a_content",
                                        "edge_a_words", "edge_a_frames", "win15", "adv_b", "adv_b_content", "adv_b_words", "adv_b_phone", "adv_b_phrasing", "adv_b_stumble", "adv_b_frames", "adv_b_ctc")
app = modal.App("parakeet-adv-b-ctc")


# how the readers write a spoken letter name ("Q" heard as "cu" / "queue"): both sides are mapped to the letter before the
# reading-heard check, so a letter code ("Q three", "T trois") is judged on its number word and its letter
LETTER_SPELL = {"en": {"cu": "q", "que": "q", "queue": "q", "cue": "q", "kew": "q", "kay": "k", "ess": "s", "pee": "p", "eff": "f",
                       "tea": "t", "tee": "t", "ex": "x", "why": "y", "wy": "y", "c": "q", "ce": "q"},
                "es": {"cu": "q", "ku": "k", "ka": "k", "te": "t", "efe": "f"},
                "fr": {"te": "t", "cu": "q", "ka": "k", "ef": "f"}}


def letters_norm(s: str, lang: str) -> str:
    import unicodedata

    m = LETTER_SPELL.get(lang, {})
    fold = lambda w: "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))  # noqa: E731
    return " ".join(m.get(fold(w), w) for w in s.replace("-", " ").split())   # a reader writes "trente-quatre": same words


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=4, memory=16384, timeout=4 * 3600, max_containers=4)
def check(planned: str, lang: str, out: str, part: int, parts: int, write_all: bool = False, real: bool = False) -> dict:
    import numpy as np
    import soundfile as sf

    import edge_a as E
    import edge_a_probe as P
    import win15
    import en_win as EW
    import ml_decide as D
    import ml_p3_core as C

    vol.reload()
    odir = f"{VOL}/ml/{out}/{lang}"
    os.makedirs(f"{odir}/wav", exist_ok=True)
    rp = f"{odir}/rows_{part}.jsonl"
    assert not os.path.exists(rp), rp
    rows = [r for k, r in enumerate(_rows(f"{VOL}/{planned}")) if k % parts == part]
    if lang == "en":
        from label_gpu import W2V

        class EnCTC(D.LangCTC):
            def __init__(self):
                super().__init__("en", "cuda", model=W2V)

            def ids(self, text):
                return EW.en_ids(text, self.vocab, self.delim)

            def greedy(self, lp):
                return EW.greedy_norm(super().greedy(lp))
        ctc = EnCTC()
        norm = lambda s: letters_norm(EW.greedy_norm(s), "en")  # noqa: E731
    else:
        ctc = D.LangCTC(lang, model=D.STRONG.get(lang))
        norm = lambda s: letters_norm(s, lang)  # noqa: E731
    import adv_b as AB
    import adv_b_words as AW

    vocab = P.edge_vocab(lang, [c.make(k, random.Random(k)).label for c in AB.inventory(lang) for k in c.values[:60]])
    numkeys = {D.key(w) for n in list(range(0, 101)) + [200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000, 100000, 1000000]
               for w in AW.card(n, lang).split()} | {D.key(w) for w in AW.DIGITS[lang]}
    numkeys |= {D.key(w) for w in "point oh double triple treble plus doble coma virgule komma punto virgola ponto zeven half quarter".split()}
    numkeys.discard("")
    if lang != "en":
        import realx1_core as R

        numkeys |= set(R.lexicon(lang))
    rng = random.Random(f"edgeAc|{part}|{lang}")
    st: dict = {}
    bump = lambda k: st.__setitem__(k, st.get(k, 0) + 1)  # noqa: E731
    out_rows = []
    for r in rows:
        try:
            x, _sr = sf.read(r.get("presynth") or r["audio_filepath"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
        except Exception:  # noqa: BLE001
            x = None
        if x is None or len(x) < 1600:
            bump("no_audio")
            continue
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (-3 / 20) * 10 ** (rng.uniform(-12, 0) / 20)
        pad = lambda: np.zeros(int(16000 * rng.uniform(0.2, 0.6)), dtype=np.float32)  # noqa: E731
        x = np.concatenate([pad(), x, pad()]).astype(np.float32)
        lp = ctc.lp(x)
        g = ctc.greedy(lp)
        T_ = r["text"]
        cer = C.cer(EW.greedy_norm(T_) if lang == "en" else T_, g)
        gn = norm(g)
        fit, why = "exact", ""
        if cer > (0.10 if real else T.CER_MAX):       # real speech: a tighter round trip (a clipped edge word is a wrong label)
            why = "cer"
        elif real and not (win15.end_close(T_, g, "start") and win15.end_close(T_, g, "end")):
            why = "edge_word"
        else:
            for a, b in r["num_chars"]:
                rd = norm(T_[a:b])
                if D.fits(rd, gn):
                    continue
                if D.fits(rd, gn, fuzzy=True):
                    fit = "fuzzy"
                    continue
                fit, why = None, "reading_not_heard"
                break
        wt = ctc.word_times(lp, T_)
        if not why and any(t is None for t in wt):
            why = "align"
        base = ctc.nll(lp, T_)
        strict = []
        if base != float("inf"):
            for a, b in r["edge_spans"]:
                s = P.strict_span(lambda t: ctc.nll(lp, t), T_, a, b, vocab, base, numkeys)
                s.update(a=a, b=b, exact_fit=D.fits(norm(T_[a:b]), gn))
                strict.append(s)
        basic_ok = not why
        strict_ok = bool(strict) and all(s["ok"] for s in strict)
        keep = basic_ok and strict_ok
        bump("kept" if keep else ("basic_" + why if why else "strict_fail"))
        clip = f"{odir}/wav/{r['id']}.wav"
        wrote = keep or write_all or basic_ok     # a clip that passed the round trip is kept even if a probe failed (second look)
        if wrote:
            sf.write(clip, x, 16000, subtype="PCM_16")
        rec = dict(r, audio_filepath=clip if wrote else None, duration=round(len(x) / 16000, 3), greedy=g,
                   cer_ctc=round(cer, 3), fit=fit, basic_ok=basic_ok, basic_why=why, strict=strict, strict_ok=strict_ok, ctc_ok=keep,
                   word_times=[[round(t[0], 3), round(t[1], 3)] if t else None for t in wt], edge_quiet=C.edge_quiet(x), src=r.get("src") or "tts")
        out_rows.append(rec)
    with open(rp, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    json.dump(dict(st, planned=len(rows)), open(f"{odir}/stats_{part}.json", "w"), indent=1)
    vol.commit()
    return dict(st, planned=len(rows), part=part, lang=lang)


@app.local_entrypoint()
def run(planned: str, lang: str, out: str, parts: int = 2, write_all: bool = False, real: bool = False):
    for res in check.starmap([(planned, lang, out, p, parts, write_all, real) for p in range(parts)], return_exceptions=True):
        print(res)
