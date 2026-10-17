"""Recover D2 FLEURS rows lost to an audio mix-up.

D2 wrote FLEURS train audio as <vol>/d2/<lang>/wav/fleurs_<id>.wav, but a FLEURS sentence id is read by 2-3 speakers,
so a later write replaced an earlier one and a manifest row can point at another speaker's recording (its duration no
longer matches the wav). This re-fetches the official google/fleurs train files (train.tsv + audio/train.tar.gz),
matches every D2 FLEURS row to its exact recording, and writes speaker-unique wavs plus corrected manifests:

    <vol>/d2fix/<lang>/wav/<id>_<file_name stem>.wav
    <vol>/d2fix/<lang>/{digit,plain}.jsonl   # FLEURS rows only (other D2 sources were never affected)
    <vol>/d2fix/<lang>/stats.json

Matching a D2 row to a recording: same sentence id, then duration within 0.05 s of train.tsv's num_samples/16000
(FLEURS has no speaker id: file_name identifies the recording, gender is carried along). A row whose duration fits
exactly one unclaimed recording is "matched" and keeps its own label (origin "d2"). Rows that fit several recordings
("ambiguous") or none ("unmatched") are not placed one-to-one; their recordings are still emitted by the sibling pass.
Sibling pass (origin "d2fix"): every other train recording of a sentence that D2 kept as a digit row is emitted too,
labelled with the sentence's D2 label when all D2 labels for that sentence agree (label_src "sibling"); the audio
adds speakers. The FLEURS dev/test splits are never read.

    modal run ml_fleurs_fix.py::plan --dir <local copy of d2/> --langs de,es,fr,it,nl,pt   # free, no audio
    modal run ml_fleurs_fix.py --langs de,es,fr,it,nl,pt                                   # CPU, ~2 GB/language
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

FLEURS_CODE = {"bg": "bg_bg", "hr": "hr_hr", "cs": "cs_cz", "da": "da_dk", "nl": "nl_nl", "et": "et_ee", "fi": "fi_fi",
               "fr": "fr_fr", "de": "de_de", "el": "el_gr", "hu": "hu_hu", "it": "it_it", "lv": "lv_lv", "lt": "lt_lt",
               "mt": "mt_mt", "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "es": "es_419",
               "sv": "sv_se", "ru": "ru_ru", "uk": "uk_ua"}
TOL = 0.05
SR = 16000


@dataclass(frozen=True)
class Rec:
    sid: int
    file_name: str
    raw: str
    dur: float
    gender: str

    @property
    def stem(self) -> str:
        return os.path.splitext(self.file_name)[0]


def parse_tsv(text: str) -> list[Rec]:
    """train.tsv columns: id, file_name, raw_transcription, transcription, characters, num_samples, gender."""
    out = []
    for line in text.split("\n"):
        c = line.rstrip("\r").split("\t")
        if len(c) < 7 or not c[0].strip().isdigit():
            continue
        out.append(Rec(int(c[0]), c[1], c[2], int(c[5]) / SR, c[6]))
    return out


def sid_of(row: dict) -> int | None:
    m = re.fullmatch(r"fleurs_(\d+)", str(row.get("id", "")))
    return int(m.group(1)) if m else None


def _norm(t: str) -> str:
    """Compare labels ignoring case, spacing and punctuation, except punctuation inside a number (6.500 != 6500)."""
    t = re.sub(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", "", t.lower())
    return " ".join(t.split())


def match_rows(rows: list[dict], recs: list[Rec], tol: float = TOL) -> dict:
    """Place D2 FLEURS rows on recordings. Returns {"matched": [(row, rec)], "ambiguous": [row], "unmatched": [row]}.

    A recording is claimed at most once. A row with exactly one candidate in tolerance that no other row also wants
    is matched; a row sharing its only candidate with another row, or with several candidates, is ambiguous."""
    by_sid = defaultdict(list)
    for r in recs:
        by_sid[r.sid].append(r)
    cands = []  # per row: list of rec indices-by-file_name within tolerance
    for row in rows:
        sid = sid_of(row)
        cands.append([r for r in by_sid.get(sid, []) if abs(r.dur - row["duration"]) <= tol] if sid is not None else [])
    want = defaultdict(int)
    for c in cands:
        if len(c) == 1:
            want[c[0].file_name] += 1
    out = {"matched": [], "ambiguous": [], "unmatched": []}
    for row, c in zip(rows, cands):
        if not c:
            out["unmatched"].append(row)
        elif len(c) == 1 and want[c[0].file_name] == 1:
            out["matched"].append((row, c[0]))
        else:
            out["ambiguous"].append(row)
    return out


def sibling_labels(digit_rows: list[dict]) -> dict:
    """sentence id -> label, only for ids whose D2 digit rows all carry the same text (whitespace-normalised)."""
    texts = defaultdict(set)
    first = {}
    for r in digit_rows:
        s = sid_of(r)
        if s is not None:
            texts[s].add(_norm(r["text"]))
            first.setdefault(s, r["text"])
    return {s: first[s] for s, t in texts.items() if len(t) == 1}


def plan(digit_rows: list[dict], plain_rows: list[dict], recs: list[Rec], lang: str, vol: str = "/vol") -> dict:
    """Pure planning: returns {"jobs": [...], "digit": [...], "plain": [...], "stats": {...}}. Each job is
    (file_name, out wav path); each manifest row is final except for the measured duration, which is checked later."""
    fd = [r for r in digit_rows if r.get("src") == "fleurs"]
    fp = [r for r in plain_rows if r.get("src") == "fleurs"]
    out_dir = f"{vol}/d2fix/{lang}/wav"
    seen: set[str] = set()
    jobs, rows = [], {"digit": [], "plain": []}
    stats = {"lang": lang, "train_recordings": len(recs)}

    def emit(kind, rec, text, origin, label_src, extra=None):
        if rec.file_name in seen:
            return False
        seen.add(rec.file_name)
        wav = f"{out_dir}/fleurs_{rec.sid}_{rec.stem}.wav"
        jobs.append((rec.file_name, wav))
        rows[kind].append({"audio_filepath": wav, "duration": round(rec.dur, 3), "text": text, "lang": lang,
                           "src": "fleurs", "id": f"fleurs_{rec.sid}", "speaker": rec.stem,
                           "file_name": rec.file_name, "gender": rec.gender, "origin": origin,
                           "label_src": label_src, **(extra or {})})
        return True

    for kind, src in (("digit", fd), ("plain", fp)):
        m = match_rows(src, recs)
        stats[kind] = {"d2_fleurs_rows": len(src), "matched": len(m["matched"]), "ambiguous": len(m["ambiguous"]),
                       "unmatched": len(m["unmatched"])}
        for row, rec in m["matched"]:
            emit(kind, rec, row["text"], "d2", "own")
    labels = sibling_labels(fd)
    disagree = {sid_of(r) for r in fd} - set(labels)
    n_sib = 0
    for rec in recs:
        if rec.sid in labels and emit("digit", rec, labels[rec.sid], "d2fix", "sibling"):
            n_sib += 1
    stats["sibling_added"] = n_sib
    stats["sentences_digit"] = len({sid_of(r) for r in fd})
    stats["sentences_label_disagree"] = len(disagree)
    stats["digit_rows_out"] = len(rows["digit"])
    stats["plain_rows_out"] = len(rows["plain"])
    stats["hours_out"] = round(sum(r["duration"] for k in rows for r in rows[k]) / 3600, 3)
    return {"jobs": jobs, "digit": rows["digit"], "plain": rows["plain"], "stats": stats}


# ---------------------------------------------------------------------------------------------- Modal

try:
    import modal
except ImportError:  # unit tests run without modal
    modal = None

if modal is not None:
    VOL = "/vol"
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11").apt_install("libsndfile1")
             .pip_install("huggingface_hub", "soundfile", "numpy<2", "librosa"))
    app = modal.App("parakeet-fleurs-fix", image=image)

    @app.function(volumes={VOL: vol}, cpu=2, memory=8192, timeout=3600, max_containers=8,
                  secrets=[modal.Secret.from_name("hf")])
    def fleurs_fix_lang(lang: str) -> dict:
        import io
        import tarfile

        import soundfile as sf
        from huggingface_hub import hf_hub_download

        code = FLEURS_CODE[lang]
        base = f"{VOL}/d2/{lang}"
        digit = [json.loads(x) for x in open(f"{base}/digit.jsonl") if x.strip()]
        plain = [json.loads(x) for x in open(f"{base}/plain.jsonl") if x.strip()]
        tsv = hf_hub_download("google/fleurs", f"data/{code}/train.tsv", repo_type="dataset", local_dir="/tmp/fl")
        recs = parse_tsv(open(tsv, encoding="utf8").read())
        p = plan(digit, plain, recs, lang, VOL)
        want = dict(p["jobs"])
        by_fn = {r["file_name"]: r for k in ("digit", "plain") for r in p[k]}
        os.makedirs(f"{VOL}/d2fix/{lang}/wav", exist_ok=True)
        tar = hf_hub_download("google/fleurs", f"data/{code}/audio/train.tar.gz", repo_type="dataset", local_dir="/tmp/fl")
        written, bad_dur, missing = set(), [], []
        with tarfile.open(tar, "r|gz") as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if not m.isfile() or name not in want or name in written:
                    continue
                data = tf.extractfile(m).read()
                info = sf.info(io.BytesIO(data))
                row = by_fn[name]
                if abs(info.duration - row["duration"]) > TOL:
                    bad_dur.append(name)
                    continue
                if info.samplerate == SR and info.channels == 1:
                    open(want[name], "wb").write(data)
                else:
                    import librosa

                    x, sr = sf.read(io.BytesIO(data), dtype="float32")
                    x = x.mean(1) if x.ndim > 1 else x
                    x = librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x
                    sf.write(want[name], x, SR, subtype="PCM_16")
                row["duration"] = round(info.duration, 3)
                written.add(name)
        missing = [n for n in want if n not in written and n not in bad_dur]
        drop = set(missing) | set(bad_dur)
        for kind in ("digit", "plain"):
            with open(f"{VOL}/d2fix/{lang}/{kind}.jsonl", "w") as f:
                for r in p[kind]:
                    if r["file_name"] not in drop:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = p["stats"] | {"wavs_written": len(written), "audio_duration_mismatch": len(bad_dur),
                           "not_in_tar": len(missing)}
        json.dump(st, open(f"{VOL}/d2fix/{lang}/stats.json", "w"), indent=1)
        os.remove(tar)
        vol.commit()
        return st

    def _langs(s: str) -> list[str]:
        ls = list(FLEURS_CODE) if s == "all" else s.split(",")
        bad = [x for x in ls if x not in FLEURS_CODE]
        if bad:
            raise SystemExit(f"unknown languages {bad}; known: {sorted(FLEURS_CODE)}")
        return ls

    @app.local_entrypoint()
    def main(langs: str = "de,es,fr,it,nl,pt"):
        res = list(fleurs_fix_lang.map(_langs(langs)))
        for st in res:
            print(json.dumps(st))

    @app.local_entrypoint()
    def plan_local(dir: str, langs: str = "de,es,fr,it,nl,pt"):
        """Free dry run: <dir>/<lang>/{digit,plain}.jsonl are local copies (modal volume get); train.tsv from the Hub."""
        from huggingface_hub import hf_hub_download

        for lang in _langs(langs):
            tsv = hf_hub_download("google/fleurs", f"data/{FLEURS_CODE[lang]}/train.tsv", repo_type="dataset")
            rd = lambda k: [json.loads(x) for x in open(f"{dir}/{lang}/{k}.jsonl") if x.strip()]  # noqa: E731
            print(json.dumps(plan(rd("digit"), rd("plain"), parse_tsv(open(tsv, encoding="utf8").read()), lang)["stats"]))
