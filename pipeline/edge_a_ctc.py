"""CTC verification of edge-pack clips (edge_a rows voiced elsewhere): per clip the round-trip rule of the other synthetic
sets (CER <= ml_tts.CER_MAX, every number reading heard, word times) PLUS the strict probes of edge_a_probe on every edge
phrase. Writes NEW volume paths only:

    ml/<out>/<lang>/rows_<part>.jsonl         every checked clip, with the verdicts (kept = basic_ok and strict_ok)
    ml/<out>/<lang>/wav/<id>.wav              the level-normalised, padded clip of a kept row (or of every row: --write-all)
    ml/<out>/<lang>/stats_<part>.json

    modal run edge_a_ctc.py --planned ml/edgeA/pilot/es/planned.jsonl --lang es --out edgeAc/pilot --parts 2 [--write-all]
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
                                        "edge_a_words", "edge_a_frames")
app = modal.App("parakeet-edge-a-ctc")


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=4, memory=16384, timeout=4 * 3600, max_containers=4)
def check(planned: str, lang: str, out: str, part: int, parts: int, write_all: bool = False) -> dict:
    import numpy as np
    import soundfile as sf

    import edge_a as E
    import edge_a_probe as P
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
        norm = EW.greedy_norm
    else:
        ctc = D.LangCTC(lang, model=D.STRONG.get(lang))
        norm = lambda s: s  # noqa: E731
    vocab = P.edge_vocab(lang, [c.make(k, random.Random(k)).label for c in E.inventory(lang) for k in c.values])
    rng = random.Random(f"edgeAc|{part}|{lang}")
    st: dict = {}
    bump = lambda k: st.__setitem__(k, st.get(k, 0) + 1)  # noqa: E731
    out_rows = []
    for r in rows:
        try:
            x, _sr = sf.read(r["presynth"], dtype="float32")
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
        cer = C.cer(norm(T_) if lang == "en" else T_, g)
        fit, why = "exact", ""
        if cer > T.CER_MAX:
            why = "cer"
        else:
            for a, b in r["num_chars"]:
                rd = norm(T_[a:b])
                if D.fits(rd, g):
                    continue
                if D.fits(rd, g, fuzzy=True):
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
                s = P.strict_span(lambda t: ctc.nll(lp, t), T_, a, b, vocab, base)
                s.update(a=a, b=b, exact_fit=D.fits(norm(T_[a:b]), g))
                strict.append(s)
        basic_ok = not why
        strict_ok = bool(strict) and all(s["ok"] for s in strict)
        keep = basic_ok and strict_ok
        bump("kept" if keep else ("basic_" + why if why else "strict_fail"))
        clip = f"{odir}/wav/{r['id']}.wav"
        if keep or write_all:
            sf.write(clip, x, 16000, subtype="PCM_16")
        rec = dict(r, audio_filepath=clip if (keep or write_all) else None, duration=round(len(x) / 16000, 3), greedy=g,
                   cer_ctc=round(cer, 3), fit=fit, basic_ok=basic_ok, basic_why=why, strict=strict, strict_ok=strict_ok, ctc_ok=keep,
                   word_times=[[round(t[0], 3), round(t[1], 3)] if t else None for t in wt], edge_quiet=C.edge_quiet(x), src="tts")
        out_rows.append(rec)
    with open(rp, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    json.dump(dict(st, planned=len(rows)), open(f"{odir}/stats_{part}.json", "w"), indent=1)
    vol.commit()
    return dict(st, planned=len(rows), part=part, lang=lang)


@app.local_entrypoint()
def run(planned: str, lang: str, out: str, parts: int = 2, write_all: bool = False):
    for res in check.starmap([(planned, lang, out, p, parts, write_all) for p in range(parts)], return_exceptions=True):
        print(res)
