"""Modal entry points for the Tier-1 multilingual pilot (R3.11). Separate app: retrain-3 code is not touched.

    modal run ml_modal.py::p2 --gold <dir with items.json + gold/wav>   # literal-reading gold + real D2 decide rate
"""

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("torch==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
         .add_local_python_source("ml_cands", "ml_decide")
         .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-ml", image=image)


@app.function(gpu="L4", volumes={VOL: vol}, timeout=3600, max_containers=6)
def p2_lang(lang: str, gold: list, n_real: int = 300) -> dict:
    import io
    import json

    import librosa
    import soundfile as sf

    import ml_decide as D

    ctc = D.LangCTC(lang)

    def load(b=None, path=None):
        x, sr = sf.read(io.BytesIO(b) if b is not None else path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x

    g_rows = []
    for it in gold:
        r = D.decide(ctc, it["caption"], ctc.lp(load(b=it["wav_bytes"])))
        s0 = r["spans"][0]
        pick = s0["cands"][s0["pick"]] if "pick" in s0 else None
        g_rows.append({"id": it["id"], "gold": it["gold"], "pick": pick, "decided": s0["decided"],
                       "margin": s0.get("margin"), "fit": s0.get("fit"), "greedy": r["greedy"][:160],
                       "right": pick is not None and D.key(pick) == D.key(it["gold"])})
    real = []
    rows = [json.loads(line) for line in open(f"{VOL}/d2/{lang}/digit.jsonl")][:n_real]
    for row in rows:
        try:
            r = D.decide(ctc, row["text"], ctc.lp(load(path=row["audio_filepath"])))
        except Exception as e:
            real.append({"id": row.get("id"), "err": repr(e)[:120]})
            continue
        real.append({"id": row.get("id"), "src": row.get("src"), "text": row["text"][:200], "greedy": r["greedy"][:200],
                     "spans": [{k: v for k, v in s.items() if k != "cands"} | {"pick_text": s["cands"][s["pick"]]}
                               if "pick" in s else s for s in r["spans"]]})
    return {"lang": lang, "gold": g_rows, "real": real}


@app.local_entrypoint()
def p2(gold: str, out: str = "p2_out.json", n_real: int = 300):
    import json
    import os

    items = json.load(open(os.path.join(gold, "items.json")))
    by = {}
    for it in items:
        it = dict(it, wav_bytes=open(os.path.join(gold, "..", it["wav"]) if not os.path.isabs(it["wav"]) else it["wav"], "rb").read())
        by.setdefault(it["lang"], []).append(it)
    langs = sorted(by)
    res = list(p2_lang.starmap([(l, by[l], n_real) for l in langs]))
    json.dump(res, open(out, "w"), ensure_ascii=False)
    for r in res:
        g = r["gold"]
        dec = [x for x in g if x["decided"]]
        wrong = [x for x in dec if not x["right"]]
        import ml_decide as D

        w_end = [x for x in wrong if D.ending_only(x["gold"], x["pick"])]
        sp = [s for x in r["real"] for s in x.get("spans", []) if s.get("why") != "no candidates" and "pick" in s]
        rd = sum(s["decided"] for s in sp)
        var = sum(1 for s in sp if s["decided"] and s["pick"] != 0)
        print(f"{r['lang']}: gold decided {len(dec)}/{len(g)} ({len(dec) / max(len(g), 1):.0%}) wrong {len(wrong)} "
              f"(ending-only {len(w_end)}, different words {len(wrong) - len(w_end)}) "
              f"| real spans decided {rd}/{len(sp)} ({rd / max(len(sp), 1):.0%}), decided non-canonical {var}")
