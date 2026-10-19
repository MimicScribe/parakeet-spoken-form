"""CTC confirmation of second-transcript-decided number readings (realx1; realx1_core has the tests).

Reads a candidate list the caller uploaded (one JSON row per line: id, audio_filepath, text, caption, spans, num_chars)
and scores each row with the language's CTC reader: per span the minimum margins of the four probes (candidate
readings, whole-span deletion, per-word deletion, per-word near-word substitution). Writes one result row per input
row under a NEW path; nothing existing is touched. Controls (--controls): known-good readings must pass, a reading
with one number word swapped for its nearest vocabulary word must fail.

    modal volume put parakeet-spoken-form <local cands.jsonl> ml/realx1/_in/<lang>/<name>.jsonl
    modal run realx1_confirm.py::run --lang es --name cands --tag a [--controls]
"""

import json
import os
import random
import re
import time

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1")
         .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
         .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "ml_slav", "census_common", "compound_split",
                                  "realx1_core")
         .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-realx1-confirm")
CHUNK = 500


def _read(p):
    import librosa
    import soundfile as sf

    x, sr = sf.read(p, dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)


def corrupt(T: str, a: int, b: int, lang: str, rng) -> tuple:
    """(text with one word of T[a:b] replaced by a near vocabulary word, new span end, None), or None."""
    import realx1_core as R

    cands = [text for kind, text in R.variants(T, a, b, lang) if kind == "wsub"]
    if not cands:
        return None
    text = rng.choice(cands[:3])
    return text, b + (len(text) - len(T)), None


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=12)
def score_chunk(lang: str, in_path: str, start: int, end: int, out_path: str, controls: bool = False) -> dict:
    import ml_cands as M
    import ml_decide as D
    import realx1_core as R
    from census_common import ext_span

    vol.reload()
    rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()][start:end]
    ctc = D.LangCTC(lang, model=D.STRONG.get(lang))
    rng = random.Random(f"realx1-{lang}-{start}")
    st = {"rows": 0, "pass": 0, "unreadable": 0}
    t_start = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {"id": r["id"], "control": r.get("control")}
            try:
                x = _read(r["audio_filepath"])
            except Exception as e:  # noqa: BLE001
                out.update(ok=False, why="unreadable")
                st["unreadable"] += 1
                f.write(json.dumps(out) + "\n")
                continue
            lp = ctc.lp(x)
            greedy = ctc.greedy(lp)
            nll = lambda t: ctc.nll(lp, t)  # noqa: E731
            T = r["text"]
            base = nll(T)
            spans, nc = r.get("spans") or [], r.get("num_chars") or []
            out.update(greedy=greedy, base_nll=round(base, 2) if base != float("inf") else None, spans=[])
            if base == float("inf") or len(spans) != len(nc):
                out.update(ok=False, why="no_ids" if base == float("inf") else "span_char_mismatch")
                f.write(json.dumps(out) + "\n")
                continue
            try:
                ms = M.spans(r["caption"], lang)
                cand_lists = [s.cands for s in ms] if len(ms) == len(spans) and all(
                    s.text == sp.get("text") for s, sp in zip(ms, spans)) else None
            except Exception:  # noqa: BLE001
                cand_lists = None
            ok_all, why = True, ""
            for j, (sp, (a, b)) in enumerate(zip(spans, nc)):
                reading = sp.get("reading") or T[a:b]
                strict = ext_span(sp) or bool(r.get("control"))
                t_use, b_use, base_use = T, b, base
                if r.get("control") == "neg":
                    c = corrupt(T, a, b, lang, rng)
                    if c is None:
                        out["spans"].append({"j": j, "skip": "no_alt"})
                        continue
                    t_use, b_use, _ = c
                    base_use = nll(t_use)
                res = R.span_probe(lambda t: ctc.nll(lp, t), t_use, a, b_use, lang, (cand_lists or [[]] * len(spans))[j]
                                   if r.get("control") != "neg" else [], D.key(reading), base_use)
                ok, w = R.span_ok(res, strict)
                fit = D.fits(reading, greedy, fuzzy=False)
                out["spans"].append({"j": j, "kind": sp.get("kind"), "by": sp.get("by"), "ok": ok, "why": w,
                                     "m": {k: (round(res[k], 1) if res[k] != float("inf") else None)
                                           for k in ("cand", "del", "wdel", "wsub")},
                                     "n": [res["n_cand"], res["n_wdel"], res["n_wsub"]], "worst": res["worst"],
                                     "exact_fit": fit, "cands": cand_lists is not None})
                if not ok:
                    ok_all, why = False, why or w
            out.update(ok=ok_all, why=why)
            st["rows"] += 1
            st["pass"] += ok_all
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    st["sec"] = round(time.time() - t_start, 1)
    vol.commit()
    return st


@app.local_entrypoint()
def run(lang: str, name: str = "cands", tag: str = "a", controls: bool = False, limit: int = 0, n_rows: int = 0):
    """n_rows: the number of rows in the uploaded list (counted by the caller; the list lives on the volume)."""
    in_path = f"{VOL}/ml/realx1/_in/{lang}/{name}.jsonl"
    n = n_rows
    jobs = [(lang, in_path, s, min(s + CHUNK, n), f"{VOL}/ml/realx1/_conf/{lang}/{tag}/part_{s:06d}.jsonl", controls)
            for s in range(0, n, CHUNK)]
    if limit:
        jobs = jobs[:limit]
    for st in score_chunk.starmap(jobs):
        print(json.dumps(st))
