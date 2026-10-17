"""Ukrainian spoken-form rows from Common Voice 22 (speech-uk/cv22 on the Hugging Face hub: the CC0 `validated`
subset, 89,248 clips, 115 h), 2026-10-17.

The Ukrainian sources that write numbers as digits cannot decide the case ending a number was said with. Common Voice
transcripts are the sentence the speaker read, and its numbers are written as words, so the ending is right by
construction: the label is the transcript (look-alike Latin letters, apostrophes and stress marks normalised,
ml_uk_words.clean) and the spans come from the Ukrainian words -> value reader (ml_uk_words.find).

Rows:
  * every clip whose transcript has at least one number span, plus number-free controls (about 15% of the number
    rows); a transcript with a digit, a symbol or a Latin word is skipped;
  * at most 3 recordings of one sentence (template cap; CV records each sentence by several speakers);
  * a held-out split of 10% by SENTENCE hash, so no sentence is in both: written as the validation-only pass
    val_cv22.jsonl (mix_build.build4 and ml_p3 windows treat val_*.jsonl ids as held out, never trained). The mirror
    has no speaker / client id column (audio, duration, transcription only), so the split cannot also be made by
    speaker: a Common Voice speaker may read sentences on both sides;
  * the check: a public wav2vec2 Ukrainian CTC reader (ml_decide.MODELS["uk"]) decodes each clip; a row whose label
    differs from the greedy text by more than 25% of its letters (ml_p3_core.cer, MAX_CER) is dropped;
  * a span-local check: each span's reading must be heard in that greedy text (ml_decide.fits, fuzzy for long
    readings, as the synthetic-speech check); a span that is not is MASKED (dropped from spans / num_chars, the row and
    its label kept; "spans_masked" lists it). The judge markers (judged_rows_cv22.json, judged_val_cv22.json) record
    that the reference is a human transcript plus these checks.

Outputs on the volume: ml/p3v2/uk/wav/cv22/<id>.wav (16 kHz mono PCM16), ml/p3v2/uk/rows_cv22.jsonl,
ml/p3v2/uk/val_cv22.jsonl, ml/p3v2/uk/ctc_cv22.jsonl (per-row check), ml/p3v2/uk/judged_*_cv22.json.

    python3 cv22_uk.py select <out_dir>            # local, transcript columns only -> selection.jsonl, select_stats.json
    python3 cv22_uk.py recheck <rows.jsonl> <ctc_cv22.jsonl> <out.jsonl>   # re-tag + span check from stored greedy text
    modal run cv22_uk.py --selection <out_dir>/selection.jsonl     # audio (CPU, <= 8 containers) + CTC check (1 GPU)
"""
import collections
import hashlib
import io
import json
import os
import random
import re
import sys

try:
    import modal
except ImportError:  # select runs without Modal
    modal = None

REPO = "datasets/speech-uk/cv22"
SRC = "cv22"
LANG = "uk"
PER_TEXT = 3
DEV_SHARE = 0.10
CONTROL_SHARE = 0.15
MIN_S, MAX_S = 1.0, 20.0
VOL = "/vol"
OUT = f"{VOL}/ml/p3v2/{LANG}"


def text_key(text: str) -> str:
    """One sentence, whatever its punctuation and case."""
    return " ".join(re.sub(r"[^\w' ]+", " ", text.lower()).split())


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def row_id(file: str, idx: int) -> str:
    return f"{SRC}_{_h(f'{file}:{idx}')[:16]}"


def is_dev(key: str) -> bool:
    return int(_h("dev:" + key)[:8], 16) / 0xFFFFFFFF < DEV_SHARE


def select(index: list, seed: int = 17) -> tuple[list, dict]:
    """index: [{"f", "i", "path", "text", "dur"}] -> (selected rows with label, spans, split), stats."""
    import ml_uk_words as U

    st = collections.Counter()
    num, free = collections.defaultdict(list), collections.defaultdict(list)
    for r in index:
        t = U.clean(r["text"])
        if t is None:
            st["skip_unusable_text"] += 1
            continue
        if not MIN_S <= r["dur"] <= MAX_S:
            st["skip_duration"] += 1
            continue
        k = text_key(t)
        sp = U.find(t)
        (num if sp else free)[k].append(dict(r, label=t, spans=sp))
    out = []

    def take(groups):
        for k, rs in groups.items():
            rs = sorted(rs, key=lambda r: _h(f"{r['f']}:{r['i']}"))  # stable choice of the <= 3 recordings
            st["over_template_cap"] += max(0, len(rs) - PER_TEXT)
            for r in rs[:PER_TEXT]:
                yield dict(r, split="dev" if is_dev(k) else "train", key=k)

    nr = list(take(num))
    out += nr
    rng = random.Random(seed)
    keys = sorted(free)
    rng.shuffle(keys)
    want = round(CONTROL_SHARE * len(nr))
    ctl = []
    for k in keys:  # one recording per control sentence: more sentences, same hours
        if len(ctl) >= want:
            break
        r = sorted(free[k], key=lambda r: _h(f"{r['f']}:{r['i']}"))[0]
        ctl.append(dict(r, split="dev" if is_dev(k) else "train", key=k))
    out += ctl
    st.update(number_rows=len(nr), control_rows=len(ctl), number_sentences=len(num), free_sentences=len(free),
              dev_rows=sum(r["split"] == "dev" for r in out))
    st["number_h"] = round(sum(r["dur"] for r in nr) / 3600, 2)
    st["control_h"] = round(sum(r["dur"] for r in ctl) / 3600, 2)
    return out, dict(st)


def row_of(sel: dict, duration: float) -> dict:
    """The pipeline row (rows_*.jsonl schema; the caption is the label itself: the transcript is already spoken form)."""
    i = row_id(sel["f"], sel["i"])
    spans = [{k: s[k] for k in ("text", "kind", "reading", "canonical", "by", "value") if k in s} for s in sel["spans"]]
    return {"audio_filepath": f"{OUT}/wav/{SRC}/{i}.wav", "duration": round(duration, 3), "text": sel["label"],
            "caption": sel["label"], "spans": spans, "num_chars": [[s["a"], s["b"]] for s in sel["spans"]], "id": i,
            "src": SRC, "origin": "new", "lang": LANG, "split": sel["split"],
            "kinds": sorted({s["kind"] for s in sel["spans"]}) or ["control"], "cv_file": sel["f"], "cv_row": sel["i"],
            "cv_clip": sel["path"]}


def span_dicts(spans: list) -> tuple[list, list]:
    """ml_uk_words spans -> (row spans, num_chars)."""
    return ([{k: s[k] for k in ("text", "kind", "reading", "canonical", "by", "value") if k in s} for s in spans],
            [[s["a"], s["b"]] for s in spans])


def span_check(row: dict, greedy: str) -> dict:
    """Re-tag the row's label with ml_uk_words.find, then keep only the spans whose reading the CTC greedy text holds
    (ml_decide.fits: exact, or fuzzy for long readings); the others are masked (listed in "spans_masked")."""
    import ml_decide as D
    import ml_uk_words as U

    D.set_script(LANG)
    keep, masked = [], []
    for s in U.find(row["text"]):
        (keep if D.fits(s["reading"], greedy) or D.fits(s["reading"], greedy, fuzzy=True) else masked).append(s)
    spans, nc = span_dicts(keep)
    out = dict(row, spans=spans, num_chars=nc, spans_masked=[s["reading"] for s in masked])
    out["kinds"] = sorted({s["kind"] for s in spans}) or (["masked"] if masked else ["control"])
    return out


def hf_index() -> list:
    """Transcript, duration and clip name of every row (no audio is read)."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted((f for f in fs.ls(REPO, detail=False) if f.endswith(".parquet")),
                   key=lambda f: int(os.path.basename(f).split(".")[0]))

    def one(f):
        t = pq.ParquetFile(fs.open(f, "rb")).read(columns=["audio.path", "transcription", "duration"]).to_pylist()
        return [{"f": os.path.basename(f), "i": i, "path": r["audio"]["path"], "text": r["transcription"],
                 "dur": r["duration"]} for i, r in enumerate(t)]

    with ThreadPoolExecutor(8) as ex:
        return [r for rs in ex.map(one, files) for r in rs]


# ------------------------------------------------------------------ Modal steps
if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    cpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .apt_install("libsndfile1")
                 .pip_install("pyarrow==17.0.0", "huggingface_hub", "soundfile", "soxr", "numpy<2")
                 .add_local_python_source("cv22_uk"))
    gpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .apt_install("ffmpeg", "libsndfile1")
                 .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa",
                              "numpy<2")
                 .add_local_python_source("cv22_uk", "ml_cands", "ml_decide", "ml_p3_core", "ml_slav", "ml_uk_words")
                 .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
    app = modal.App("parakeet-cv22-uk")

    @app.function(image=cpu_image, volumes={VOL: vol}, cpu=2, memory=6144, timeout=3600, max_containers=8,
                  secrets=[modal.Secret.from_name("hf")])
    def ingest_file(job: tuple) -> list:
        """One parquet file: decode the selected clips, 16 kHz mono PCM16 wav on the volume, the rows back."""
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        import soxr
        from huggingface_hub import HfFileSystem

        file, sels = job
        fs = HfFileSystem()  # the hub client reads its token from the attached secret's environment
        t = pq.ParquetFile(fs.open(f"{REPO}/{file}", "rb")).read(columns=["audio"])
        col = t.column("audio")
        os.makedirs(f"{OUT}/wav/{SRC}", exist_ok=True)
        rows = []
        for s in sels:
            a = col[s["i"]].as_py()
            assert a["path"] == s["path"], (file, s["i"], a["path"], s["path"])  # the index still matches the file
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != 16000:
                x = soxr.resample(x, sr, 16000, quality="HQ")
            x = np.clip(x, -1.0, 1.0)
            r = row_of(s, len(x) / 16000)
            sf.write(r["audio_filepath"], x, 16000, subtype="PCM_16")
            rows.append(r)
        vol.commit()
        return rows

    @app.function(image=gpu_image, volumes={VOL: vol}, gpu="L4", memory=16384, timeout=2 * 3600, max_containers=1)
    def ctc_check(rows: list) -> dict:
        """Greedy text of the Ukrainian CTC reader per row; drop rows whose label is > MAX_CER from it; write the
        passes and their judge markers."""
        import soundfile as sf

        import ml_decide as D
        import ml_p3_core as C

        vol.reload()
        ctc = D.LangCTC(LANG, device="cuda")
        diag, keep = [], {"train": [], "dev": []}
        st = collections.Counter()
        for r in rows:
            x, _ = sf.read(r["audio_filepath"], dtype="float32")
            g = ctc.greedy(ctc.lp(x))
            c = C.cer(r["text"], g)
            ok = c <= C.MAX_CER
            r.update(span_check(r, g))
            r["cer_ctc"] = round(c, 3)
            diag.append({"id": r["id"], "split": r["split"], "cer_ctc": round(c, 3), "greedy": g, "text": r["text"],
                         "number": bool(r["spans"]), "pass": ok})
            st[f"{r['split']}_{'number' if r['spans'] else 'control'}_{'pass' if ok else 'fail'}"] += 1
            if ok:
                keep[r["split"]].append(r)
        for split, name in (("train", "rows_cv22"), ("dev", "val_cv22")):
            with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
                for r in keep[split]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ks = collections.Counter(s["kind"] for r in keep[split] for s in r["spans"])
            marker = {"rows": len(keep[split]), "rejected": sum(1 for d in diag if d["split"] == split and not d["pass"]),
                      "reasons": {"cer_ctc": sum(1 for d in diag if d["split"] == split and not d["pass"])},
                      "lang": LANG, "pass": name, "max_cer": C.MAX_CER, "ctc_reader": D.MODELS[LANG],
                      "judge": "human transcript (Common Voice, spoken form as read) + CTC agreement check "
                               "(greedy CER <= max_cer, failing rows dropped)",
                      "hours": round(sum(r["duration"] for r in keep[split]) / 3600, 3), "span_kinds": dict(ks)}
            json.dump(marker, open(f"{OUT}/judged_{name}.json", "w"), indent=1, ensure_ascii=False)
        with open(f"{OUT}/ctc_cv22.jsonl", "w", encoding="utf-8") as f:
            for d in diag:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        vol.commit()
        return dict(st)

    @app.local_entrypoint()
    def main(selection: str):
        sels = [json.loads(l) for l in open(selection) if l.strip()]
        by_file = collections.defaultdict(list)
        for s in sels:
            by_file[s["f"]].append(s)
        jobs = sorted(by_file.items(), key=lambda kv: int(kv[0].split(".")[0]))
        rows = [r for part in ingest_file.map(jobs) for r in part]
        print(f"ingested {len(rows)} rows from {len(jobs)} files, {sum(r['duration'] for r in rows) / 3600:.2f} h")
        print(json.dumps(ctc_check.remote(rows)))


def _main(argv):
    if argv[0] == "recheck":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        greedy = {json.loads(l)["id"]: json.loads(l)["greedy"] for l in open(argv[2], encoding="utf-8") if l.strip()}
        st = collections.Counter()
        with open(argv[3], "w", encoding="utf-8") as f:
            for l in open(argv[1], encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                n = span_check(r, greedy[r["id"]])
                st["rows"] += 1
                st["spans_kept"] += len(n["spans"])
                st["spans_masked"] += len(n["spans_masked"])
                import ml_uk_words as U

                st["retag_changed"] += [(s["text"], s["kind"]) for s in r["spans"]] != [
                    (s["text"], s["kind"]) for s in U.find(r["text"])]
                st["rows_all_masked"] += n["kinds"] == ["masked"]
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(json.dumps(dict(st)))
        return
    if argv[0] == "select":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        out = argv[1]
        os.makedirs(out, exist_ok=True)
        if "--index" in argv:
            index = [json.loads(l) for l in open(argv[argv.index("--index") + 1]) if l.strip()]
        else:
            index = hf_index()
        sel, st = select(index)
        with open(f"{out}/selection.jsonl", "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump(st, open(f"{out}/select_stats.json", "w"), indent=1)
        print(json.dumps(st))


if __name__ == "__main__":
    _main(sys.argv[1:])
