"""Clipped / missing edge words of real training rows (2026-10-18; ~25% of real YouTube rows, tight caption crops).

A row whose audio starts after its label's first word began (or ends before its last word ended) teaches the model to
write a word it cannot hear. The check is win15's per-row edge re-decode: the language's CTC reader on the row's first
and last EDGE_S s, compared with the label's first / last EDGE_LETTERS letters (win15.end_close). A failing side gets:
  * "trim": the re-decode matches the label without its first (last) 1-2 words -> those words are not in the audio:
    the label drops them, the audio is unchanged;
  * "mask": anything else (a clipped word, a reader slip, a fragment) -> the edge word leaves the loss: it leaves the
    label AND its audio is cut off at the quietest point between it and the next word (ml_p3_core.quiet_cut, else the
    midpoint), so neither the word nor its fragment is trained on. The cut edge is re-decoded once ("verified"); an
    unverified mask is still applied (unsure = mask, never drop the row);
  * an edge word inside a number reading takes the whole reading with it (a partial reading would change the value);
    a row that would keep < MIN_WORDS words or < MIN_S s, or whose word times do not match its label, is left as is
    ("kept_unfixable").
Rows are never dropped. The build applies the records at write time (mix_build.build4 edge_fix_dir) to rows whose label
still hashes to the checked one; every selection / cap runs on the unfixed labels.

    modal run edge_fix.py::main --langs de,es,fr,it,nl,pt --mix-dir mix4final8 --tag 1
"""
import hashlib
import json
import os
import sys

import ml_p3_core as C

VOL = "/vol"
EDGE_S = 3.0  # win15.EDGE_S
EDGE_LETTERS = 6  # win15.EDGE_LETTERS
EDGE_TOL = 0.34  # win15.EDGE_TOL
MAX_TRIM = 2  # words a "trim" may drop per side
MIN_WORDS = 3
MIN_S = 1.0
LOOK = 0.3  # s searched for a cut next to a dropped token without letter frames
CUT_PAD = 0.02  # s kept clear of the letter frames on both sides of a cut


def sha(text: str) -> str:  # win15.sha
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def end_close(label: str, greedy: str, side: str, n: int = EDGE_LETTERS, tol: float = EDGE_TOL) -> bool:
    """win15.end_close: the re-decode's first / last n letters are close to the label's."""
    lab, g = C.letters(label), C.letters(greedy)
    if not lab or not g:
        return False
    return C.cer(lab[:n], g[:n]) <= tol if side == "start" else C.cer(lab[-n:], g[-n:]) <= tol


def reading_of(text: str, num_chars) -> list:
    """Per whitespace token: the index of the number reading (num_chars range) it lies in, or None."""
    out, pos = [], 0
    for tok in text.split():
        a = text.index(tok, pos)
        b = pos = a + len(tok)
        out.append(next((i for i, (x, y) in enumerate(num_chars or []) if x < b and a < y), None))
    return out


def widen(k: int, side: str, rid: list) -> int:
    """k edge tokens dropped on `side`, widened so no number reading is cut in two."""
    n = len(rid)
    if side == "start":
        while 0 < k < n and rid[k - 1] is not None and rid[k] == rid[k - 1]:
            k += 1
    else:
        while 0 < k < n and rid[n - k] is not None and rid[n - k - 1] == rid[n - k]:
            k += 1
    return k


def drop_words(text: str, k: int, side: str) -> str:
    w = text.split()
    return " ".join(w[k:] if side == "start" else w[:len(w) - k])


def side_plan(text: str, greedy: str, side: str, rid: list) -> dict:
    """What one side needs: {"act": "ok" | "trim" | "mask", "k": tokens dropped}."""
    if end_close(text, greedy, side):
        return {"act": "ok", "k": 0}
    for k0 in range(1, MAX_TRIM + 1):
        k = widen(k0, side, rid)
        if k >= len(rid):
            break
        if end_close(drop_words(text, k, side), greedy, side):
            return {"act": "trim", "k": k}
    return {"act": "mask", "k": widen(1, side, rid)}


def cut_time(times: list, k: int, side: str, x=None) -> float | None:
    """The audio cut that leaves the k dropped edge tokens out: between the last dropped and the first kept token (a
    dropped token without letter frames: up to LOOK s before / after the kept one). None: the kept token has no time."""
    n = len(times)
    kept, drop = (times[k], times[k - 1]) if side == "start" else (times[n - k - 1], times[n - k])
    if kept is None:
        return None
    if side == "start":
        t0, t1 = (drop[1] if drop else max(0.0, kept[0] - LOOK)) + CUT_PAD, kept[0] - CUT_PAD
    else:
        t0, t1 = kept[1] + CUT_PAD, (drop[0] if drop else kept[1] + LOOK) - CUT_PAD
    if t1 <= t0:  # touching letter frames: their midpoint
        return round((t0 + t1) / 2, 3)
    q = C.quiet_cut(x, t0, t1) if x is not None else None
    return round(q if q is not None else (t0 + t1) / 2, 3)


def cut_label(row: dict, k_start: int, k_end: int, crop: list | None) -> dict:
    """The row with k_start / k_end edge tokens out of its label (num_chars, spans, word_times follow; a dropped
    reading's span leaves) and, with crop = [a, b] s, its audio window (word times shifted by a). Pure."""
    text = row["text"]
    toks = text.split()
    n = len(toks)
    keep = list(range(k_start, n - k_end))
    starts, pos = [], 0
    for tok in toks:
        a = text.index(tok, pos)
        starts.append(a)
        pos = a + len(tok)
    c0 = starts[keep[0]]
    c1 = starts[keep[-1]] + len(toks[keep[-1]])
    new_text = " ".join(toks[i] for i in keep)
    # char offsets: map each kept token's start (whitespace collapses to one space)
    remap, p = {}, 0
    for i in keep:
        remap[starts[i]] = p
        p += len(toks[i]) + 1

    def mp(c):  # an offset inside a kept token
        base = max(s for s in remap if s <= c)
        return remap[base] + (c - base)

    out = dict(row, text=new_text)
    if "num_chars" in row:
        nc, sp, kept_idx = [], [], []
        spans = row.get("spans") or []
        for i, (x, y) in enumerate(row["num_chars"]):
            if x >= c0 and y <= c1:
                nc.append([mp(x), mp(y - 1) + 1])
                kept_idx.append(i)
        out["num_chars"] = nc
        if len(spans) == len(row["num_chars"]):
            out["spans"] = [spans[i] for i in kept_idx]
    if row.get("word_times") and len(row["word_times"]) == n:
        sh = crop[0] if crop else 0.0
        out["word_times"] = [None if row["word_times"][i] is None else
                             [round(row["word_times"][i][0] - sh, 3), round(row["word_times"][i][1] - sh, 3)] for i in keep]
    if crop:
        out["duration"] = round(crop[1] - crop[0], 3)
    out["edge_fix"] = {"k_start": k_start, "k_end": k_end, "crop": crop, "from_sha": sha(text)}
    return out


def plan_row(row: dict, g_start: str, g_end: str, dur: float, x=None) -> dict:
    """The fix record of one row from its edge re-decodes: {"sides": {start, end}, "k_start", "k_end", "crop"
    ([a, b] s or None), "status"}; status "ok" | "fixed" | "kept_unfixable" (rid / times / length rules)."""
    text = row["text"]
    rid = reading_of(text, row.get("num_chars"))
    sides = {"start": side_plan(text, g_start, "start", rid), "end": side_plan(text, g_end, "end", rid)}
    rec = {"id": row["id"], "sha": sha(text), "sides": sides, "k_start": 0, "k_end": 0, "crop": None, "status": "ok"}
    if all(s["act"] == "ok" for s in sides.values()):
        return rec
    ks, ke = sides["start"]["k"], sides["end"]["k"]
    n = len(rid)
    times = row.get("word_times") or []
    need_crop = any(s["act"] == "mask" for s in sides.values())
    if n - ks - ke < MIN_WORDS or (need_crop and len(times) != n):
        rec["status"] = "kept_unfixable"
        return rec
    a = cut_time(times, ks, "start", x) if sides["start"]["act"] == "mask" else 0.0
    b = cut_time(times, ke, "end", x) if sides["end"]["act"] == "mask" else round(dur, 3)
    if need_crop and (a is None or b is None or b - a < MIN_S):
        rec["status"] = "kept_unfixable"
        return rec
    rec.update(k_start=ks, k_end=ke, crop=[a, b] if need_crop else None, status="fixed")
    return rec


def apply_fixes(rows: list, fixes: dict, audio_of=None) -> tuple:
    """(rows, counts): each row with a "fixed" record whose sha matches its label, cut (cut_label); a cropped row's
    audio_filepath = audio_of(record) (the record's "audio"). Rows without a record / with a changed label are kept
    unchanged and counted. Never drops a row."""
    st = {"rows": len(rows), "checked": 0, "unchecked": 0, "label_changed": 0, "ok": 0, "fixed": 0, "trim": 0,
          "mask": 0, "mask_verified": 0, "kept_unfixable": 0, "spans_dropped": 0, "words_dropped": 0}
    out = []
    for r in rows:
        f = fixes.get(r["id"])
        if f is None:
            st["unchecked"] += 1
            out.append(r)
            continue
        if f["sha"] != sha(r["text"]):
            st["label_changed"] += 1
            out.append(r)
            continue
        st["checked"] += 1
        st[f["status"]] += 1
        if f["status"] != "fixed":
            out.append(r)
            continue
        for s in ("start", "end"):
            act = f["sides"][s]["act"]
            if act != "ok":
                st[act] += 1
                st["mask_verified"] += act == "mask" and bool(f.get("verified", {}).get(s))
        x = cut_label(r, f["k_start"], f["k_end"], f["crop"])
        if f["crop"]:
            assert f.get("audio"), f"{r['id']}: a cropped fix without its audio"
            x["audio_filepath"] = audio_of(f) if audio_of else f["audio"]
        st["spans_dropped"] += len(r.get("num_chars") or []) - len(x.get("num_chars") or [])
        st["words_dropped"] += f["k_start"] + f["k_end"]
        out.append(x)
    return out, st


def load_fixes(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return {f["id"]: f for f in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


# ------------------------------------------------------------------------------------------------ Modal (GPU reader)
try:
    import modal
except ImportError:  # local tests
    modal = None

if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "census_common", "labeller", "readings",
                                      "compound_split", "win15", "en_win", "label_gpu", "ml_slav", "ml_balance"))
    app = modal.App("parakeet-edgefix")

    def _rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

    def _sources(lang: str, mix_dir: str) -> list:
        """The real rows of <mix_dir>/p3_<lang> as labelled on disk BEFORE the build's compound split (win15._sources:
        the ml/p3v2 pass whose letters match the built label)."""
        import glob

        import win15 as W

        built = _rows(f"{VOL}/{mix_dir}/p3_{lang}.jsonl")
        orig = {}
        for p in sorted(glob.glob(f"{VOL}/ml/p3v2/{lang}/rows*.jsonl"),
                        key=lambda p: (not os.path.basename(p).endswith("_regional.jsonl"), p)):
            for x in _rows(p):
                orig.setdefault(x["id"], []).append(x)
        out = []
        for b in built:
            if b.get("synthetic") or b.get("src") == "tts":
                continue
            m = [x for x in orig.get(b["id"], []) if x["text"] == b["text"]] or \
                [x for x in orig.get(b["id"], []) if W.letters_eq(x["text"], b["text"], lang)]
            out.append(dict(m[0], _built_sha=sha(b["text"])) if m else dict(b, _no_source=True))
        return out

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=8)
    def check_lang(lang: str, mix_dir: str, tag: str, shard: int = 0, n_shards: int = 1, limit: int = 0) -> dict:
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor

        import librosa
        import ml_decide as D
        import numpy as np
        import soundfile as sf

        out = f"{VOL}/ml/edgefix{tag}/{lang}"
        fpath = f"{out}/fixes_{shard}.jsonl"
        assert not os.path.exists(fpath), f"{fpath} exists: use a new tag"
        os.makedirs(f"{out}/wav", exist_ok=True)
        rows = _sources(lang, mix_dir)[shard::n_shards]
        rows = rows[:limit] if limit else rows  # a smoke run
        ctc = D.LangCTC(lang)
        st = Counter()
        n_edge = int(EDGE_S * 16000)

        def read(p):
            try:
                x, sr = sf.read(p, dtype="float32")
            except Exception:  # noqa: BLE001
                return None
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        def greedy_batch(chunks):
            torch = ctc.torch
            res = []
            for i in range(0, len(chunks), 32):
                part = chunks[i:i + 32]
                same = len({len(c) for c in part}) == 1
                if not same:
                    res += [ctc.greedy(ctc.lp(c)) for c in part]
                    continue
                with torch.no_grad():
                    iv = ctc.proc(part, sampling_rate=16000, return_tensors="pt", padding=False).input_values.to(ctc.device)
                    lg = ctc.model(iv).logits.argmax(-1)
                res += [ctc.proc.decode(t.tolist()).lower() for t in lg]
            return res

        f = open(fpath, "w", encoding="utf-8")
        pool = ThreadPoolExecutor(16)
        B = 256
        for i0 in range(0, len(rows), B):
            batch = rows[i0:i0 + B]
            xs = list(pool.map(lambda r: read(r["audio_filepath"]), batch))
            live = [(r, x) for r, x in zip(batch, xs) if x is not None and not r.get("_no_source")]
            st["unreadable_or_no_source"] += len(batch) - len(live)
            gs = greedy_batch([x[:n_edge] for _, x in live])
            ge = greedy_batch([x[-n_edge:] for _, x in live])
            recs = []
            for (r, x), a, b in zip(live, gs, ge):
                rec = plan_row(r, a, b, len(x) / 16000, x)
                rec.update(src=r.get("src"), built_sha=r["_built_sha"], g_start=a, g_end=b)
                st[f"{r.get('src')}_rows"] += 1
                st[f"{r.get('src')}_{rec['status']}"] += 1
                for s in ("start", "end"):
                    st[f"{r.get('src')}_{s}_{rec['sides'][s]['act']}"] += 1
                if rec["crop"]:
                    a_, b_ = rec["crop"]
                    y = x[int(a_ * 16000):int(b_ * 16000)]
                    wav = f"{out}/wav/{r['id']}.wav"
                    sf.write(wav, y, 16000, subtype="PCM_16")
                    rec["audio"] = wav
                    rec["_y"] = y
                recs.append((r, rec))
            # one re-decode of each masked (cut) edge with the new label
            ver = [(r, rec, s) for r, rec in recs if rec["crop"] for s in ("start", "end") if rec["sides"][s]["act"] == "mask"]
            vg = greedy_batch([rec["_y"][:n_edge] if s == "start" else rec["_y"][-n_edge:] for _, rec, s in ver])
            for (r, rec, s), g in zip(ver, vg):
                new = drop_words(drop_words(r["text"], rec["k_start"], "start"), rec["k_end"], "end")
                ok = end_close(new, g, s)
                rec.setdefault("verified", {})[s] = ok
                st[f"{r.get('src')}_mask_{s}_verified"] += ok
            for _, rec in recs:
                rec.pop("_y", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i0 // B) % 10 == 0:
                vol.commit()
        f.close()
        res = {"lang": lang, "shard": shard, "rows": len(rows), "counts": dict(st)}
        json.dump(res, open(f"{out}/stats_{shard}.json", "w"), indent=1)
        vol.commit()
        return res

    @app.local_entrypoint()
    def main(langs: str = "de,es,fr,it,nl,pt", mix_dir: str = "mix4final8", tag: str = "1", shards: int = 2, limit: int = 0):
        jobs = [(l, mix_dir, tag, s, shards, limit) for l in langs.split(",") for s in range(shards)]
        res = list(check_lang.starmap(jobs))
        print(json.dumps(res, indent=1))
        sys.stdout.flush()
