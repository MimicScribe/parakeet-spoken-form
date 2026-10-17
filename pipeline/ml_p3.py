"""Retrain-4 step P3 (R3.11): spoken-form DATA for the Tier-1 languages (es fr de pt it nl). No training.

Stage A (CPU)  scan_lang   : new real rows. VoxPopuli TRAIN (CC0) files beyond the 4 the D2 harvest used (pt is not in
                             VoxPopuli: YODAS-Granary asr_only shards beyond the harvest's 40); keep rows whose raw_text
                             has digits, every digit run inside a candidate-able number span, 1-20 s. Writes
                             /vol/ml/p3/<lang>/{cands.jsonl,wav/}.
Stage B (L4)   label_lang  : the per-language xlsr-53 CTC reading check (ml_decide.decide) over (a) the D2 digit rows
                             and (b) the new rows. A row is kept ONLY when every number span is decided and no digit
                             remains; label = caption with each span replaced by its decided reading. A row whose label
                             differs from the CTC greedy text by CER > MAX_CER is dropped (raw_text edges that do not
                             match the audio). Output /vol/ml/p3/<lang>/{rows.jsonl,stats.json,wav/}.

Everything is written under /vol/ml/p3/ only. Crops: both sources are capped at 20 s (longer rows are never taken); each
kept row is flagged `edge_quiet` (a quiet 20 ms frame in each outer 160 ms, the mix_build.trim_edges rule) for the mix
builder's trim step. No word-level cutting is done: the CTC reader gives no per-word times worth trusting here.

    modal run ml_p3.py::scan --langs de,es --hours 4.5
    modal run ml_p3.py::label --langs de
"""

import json
import os
import re
import unicodedata

import modal

VOL = "/vol"
OUT = f"{VOL}/ml/p3"
MAX_CER = 0.25
vol = modal.Volume.from_name("parakeet-spoken-form")
LANGS = ["de", "es", "fr", "it", "nl", "pt"]
VOX = {"de", "es", "fr", "it", "nl"}
VOX_SKIP = {"nl": 0, "it": 0}  # default 4: harvest.sources() used the first 4 sorted VoxPopuli files; nl/it have only
# ~4-5 files, so they restart at 0 and rely on the seen-id exclusion (every D2 decision is in audit.jsonl)
YODAS_SKIP = 40  # and the first 40 shuffled YODAS shards

scan_image = (modal.Image.debian_slim(python_version="3.11")
              .apt_install("ffmpeg", "libsndfile1")
              .pip_install("pyarrow", "huggingface_hub", "hf_transfer", "soundfile", "librosa")
              .add_local_python_source("ml_cands"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-ml-p3")


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=4 * 3600, max_containers=6)
def scan_lang(lang: str, hours: float = 4.5, max_files: int = 10) -> dict:
    import io

    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    import ml_cands as M

    out = f"{OUT}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    seen = set()
    for line in open(f"{VOL}/d2/{lang}/audit.jsonl"):
        seen.add(json.loads(line)["id"])
    api = HfApi()
    if lang in VOX:
        files = sorted(f for f in api.list_repo_files("facebook/voxpopuli", repo_type="dataset")
                       if f.startswith(f"{lang}/train-"))[VOX_SKIP.get(lang, 4):]
        repo, kind, cols = "facebook/voxpopuli", "vox", ["audio_id", "audio", "raw_text"]
    else:
        import random

        fl = sorted(f for f in api.list_repo_files("espnet/yodas-granary", repo_type="dataset")
                    if f.startswith("data/") and "/asr_only/" in f and f.split("/")[1][:2] == lang)
        random.Random(lang).shuffle(fl)
        files = fl[YODAS_SKIP:]
        repo, kind, cols = "espnet/yodas-granary", "yodas", ["utt_id", "audio", "text", "duration"]
    st = {"lang": lang, "kind": kind, "files": 0, "rows_seen": 0, "digit_rows": 0, "kept": 0, "hours": 0.0,
          "skipped_unspannable": 0, "skipped_dur": 0, "skipped_seen": 0}
    rows = []
    for path in files[:max_files]:
        if st["hours"] >= hours:
            break
        local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/src")
        st["files"] += 1
        pf = pq.ParquetFile(local)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                st["rows_seen"] += 1
                cap = (r.get("raw_text") or r.get("text") or "").strip()
                rid = r.get("audio_id") or r.get("utt_id")
                if not re.search(r"\d", cap):
                    continue
                st["digit_rows"] += 1
                if rid in seen:
                    st["skipped_seen"] += 1
                    continue
                if not M.all_digits_spanned(cap, lang):
                    st["skipped_unspannable"] += 1
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                d = len(x) / 16000
                if d < 1.0 or d > 20.0:
                    st["skipped_dur"] += 1
                    continue
                seen.add(rid)  # ids repeat inside VoxPopuli/YODAS shards: one audio file per id
                wav = f"{out}/wav/new_{rid}.wav"
                sf.write(wav, x, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": wav, "duration": round(d, 3), "text": cap, "lang": lang, "src": kind,
                             "id": rid, "origin": "new"})
                st["kept"] += 1
                st["hours"] += d / 3600
                if st["hours"] >= hours:
                    break
            if st["hours"] >= hours:
                break
        os.remove(local)
    with open(f"{out}/cands.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["hours"] = round(st["hours"], 2)
    json.dump(st, open(f"{out}/scan_stats.json", "w"))
    vol.commit()
    return st


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=6)
def label_lang(lang: str, limit: int = 0, new_hours: float = 1.5) -> dict:
    import shutil

    import librosa
    import soundfile as sf

    import ml_decide as D
    import ml_p3_core as C

    ctc = D.LangCTC(lang)
    out = f"{OUT}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    d2 = [dict(json.loads(l), origin="d2") for l in open(f"{VOL}/d2/{lang}/digit.jsonl")]
    new = [json.loads(l) for l in open(f"{out}/cands.jsonl")] if os.path.exists(f"{out}/cands.jsonl") else []
    import random

    random.Random(lang).shuffle(new)  # the quota takes rows from all scanned files, not just the first
    if limit:
        d2, new = d2[:limit], new[:limit]
    acc = C.Acc(lang)
    kept_new_h = 0.0
    kept = []
    for row in d2 + new:
        if row["origin"] == "d2":
            # D2 names wavs by id and FLEURS ids repeat across speakers: the later write won, so a row whose duration
            # differs from its file points at someone else's audio. Keep only rows that match their file.
            if abs(sf.info(row["audio_filepath"]).duration - row["duration"]) > 0.05:
                acc.add_skip(row, "d2_audio_mismatch")
                continue
        if row["origin"] == "new" and kept_new_h >= new_hours * 3600:
            acc.add_skip(row, "new_quota_reached")
            continue
        try:
            x, sr = sf.read(row["audio_filepath"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            r = D.decide(ctc, row["text"], ctc.lp(x))
        except Exception as e:  # noqa: BLE001
            acc.add_skip(row, f"error {type(e).__name__}")
            continue
        res = C.make_row(row, r, x)
        acc.add(row, r, res)
        if res.get("row"):
            nr = res["row"]
            dst = f"{out}/wav/{row['origin']}_{row['id']}.wav"
            if row["audio_filepath"] != dst:  # new rows already sit in wav/ (scan wrote them); D2 rows are copied
                shutil.copyfile(row["audio_filepath"], dst)
            nr["audio_filepath"] = dst
            kept.append(nr)
            if row["origin"] == "new":
                kept_new_h += nr["duration"]
    keep = {os.path.basename(nr["audio_filepath"]) for nr in kept}
    for fn in os.listdir(f"{out}/wav"):  # scan wavs of rejected / over-quota new rows
        if fn not in keep:
            os.remove(f"{out}/wav/{fn}")
    with open(f"{out}/rows.jsonl", "w") as f:
        for nr in kept:
            f.write(json.dumps(nr, ensure_ascii=False) + "\n")
    st = acc.stats(kept)
    json.dump(st, open(f"{out}/stats.json", "w"), ensure_ascii=False, indent=1)
    vol.commit()
    return st


# ---------------------------------------------------------------- v2 (2026-10-15): price / phone / clock / decimal
# coverage and window-edge crops. Writes /vol/ml/p3v2/<lang>/ only; v1 (/vol/ml/p3) is left as it was.
#   scan2  (CPU) : VoxPopuli AND YODAS-Granary for every language. VoxPopuli (parliament) has almost no prices or phone
#                  numbers; YouTube speech does. Rows whose spans include a RARE kind are kept up to `rare_hours`, plain
#                  integer/year rows up to `int_hours`. Every scanned wav is kept (re-labelling never needs a re-scan).
#   (local)      : the grammar table must cover the new captions before labelling:
#                  modal volume get ... ml/p3v2/<lang>/cands_*.jsonl, then ml_cands.build_table(...) with the dumper.
#   label2 (L4)  : as label_lang, plus diag.jsonl (every span's decision), CTC word times of each kept label, and up to
#                  two window-edge crops per kept row (ml_p3_core.plan_crops), each re-decoded and kept only when its
#                  label still matches the audio (CER <= MAX_CER) and its first and last words are heard.
OUT2 = f"{VOL}/ml/p3v2"
RARE = {"cur", "dec", "pct", "clock", "phone", "code", "ord", "unit", "date", "range", "decade", "alnum", "roman", "frac",
        "version", "amp"}
YODAS_SKIP2 = {"pt": 40}  # pt v1 scanned shuffled shards 40..340; the other languages never used YODAS here
gpu_image2 = (modal.Image.debian_slim(python_version="3.11")  # gpu_image + torchaudio (forced alignment); add_local_*
              .apt_install("ffmpeg", "libsndfile1")           # must stay the last steps, so this is a separate image
              .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa",
                           "numpy<2")
              .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "ml_slav", "ml_balance")  # ml_slav: ru / uk
              .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))


def _src_files(api, lang, src):
    import random

    if src in ("vox", "voxval"):
        if lang not in VOX:
            return [], None, None
        # voxval: VoxPopuli's own dev + test splits, held out
        pre = (f"{lang}/validation-", f"{lang}/test-") if src == "voxval" else (f"{lang}/train-",)
        files = sorted(f for f in api.list_repo_files("facebook/voxpopuli", repo_type="dataset") if f.startswith(pre))
        return (files if src == "voxval" else files[VOX_SKIP.get(lang, 4):]), "facebook/voxpopuli", ["audio_id", "audio", "raw_text"]
    # yodasast (2026-10-16): the "ast" config of yodas-granary = different, ~45x larger rows (same `text` column)
    part = "/ast/" if src == "yodasast" else "/asr_only/"
    fl = sorted(f for f in api.list_repo_files("espnet/yodas-granary", repo_type="dataset")
                if f.startswith("data/") and part in f and f.split("/")[1][:2] == lang)
    random.Random(lang).shuffle(fl)
    return fl[YODAS_SKIP2.get(lang, 0):], "espnet/yodas-granary", ["utt_id", "audio", "text", "duration"]


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=2 * 3600, max_containers=6)
def fleurs_val_lang(lang: str) -> dict:
    """Validation-only candidates from the FLEURS TEST split ("as much validation data as you can";
    never trained: training reads FLEURS train only). Rows whose caption has digits, every digit spannable ->
    <lang>/cands_fleursval.jsonl + wav/fleursval_*.wav. Label with `label2 --srcs fleursval --name val_fleurs`."""
    import io
    import tarfile

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import ml_cands as M

    out = f"{OUT2}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    code = {"de": "de_de", "es": "es_419", "fr": "fr_fr", "it": "it_it", "nl": "nl_nl", "pt": "pt_br"}[lang]  # = ml_fleurs_fix
    tsv = open(hf_hub_download("google/fleurs", f"data/{code}/test.tsv", repo_type="dataset", local_dir="/tmp/fl"),
               encoding="utf8").read()
    from types import SimpleNamespace

    # test.tsv columns: id, file_name, raw_transcription, transcription, characters, num_samples, gender
    recs = [SimpleNamespace(sid=c[0], file_name=c[1], raw=c[2]) for c in (l.rstrip("\r").split("\t") for l in tsv.split("\n"))
            if len(c) >= 7 and c[0].isdigit()]
    want = {}
    for r in recs:
        cap = r.raw.strip()
        if re.search(r"\d", cap) and M.all_digits_spanned(cap, lang):
            want[r.file_name] = (r, cap, sorted({d["kind"] for d in M._find(cap, lang)} - {"skip"}))
    tar = hf_hub_download("google/fleurs", f"data/{code}/audio/test.tar.gz", repo_type="dataset", local_dir="/tmp/fl")
    rows = []
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not m.isfile() or name not in want:
                continue
            r, cap, kinds = want[name]
            x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            wav = f"{out}/wav/fleursval_{os.path.splitext(name)[0]}.wav"
            sf.write(wav, x, sr, subtype="PCM_16")
            rows.append({"audio_filepath": wav, "duration": round(len(x) / sr, 3), "text": cap, "lang": lang,
                         "src": "fleurs", "id": f"fleursval_{r.sid}_{os.path.splitext(name)[0]}", "origin": "new",
                         "kinds": kinds})
    with open(f"{out}/cands_fleursval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    return {"lang": lang, "test_recordings": len(recs), "with_digits": len(want), "written": len(rows),
            "h": round(sum(r["duration"] for r in rows) / 3600, 2)}


FLEURS_CODES = {"bg": "bg_bg", "hr": "hr_hr", "cs": "cs_cz", "da": "da_dk", "nl": "nl_nl", "et": "et_ee", "fi": "fi_fi",
                "fr": "fr_fr", "de": "de_de", "el": "el_gr", "hu": "hu_hu", "it": "it_it", "lv": "lv_lv", "lt": "lt_lt",
                "mt": "mt_mt", "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si", "es": "es_419",
                "sv": "sv_se", "ru": "ru_ru", "uk": "uk_ua"}  # = ml_fleurs_fix.FLEURS_CODE


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=2 * 3600, max_containers=12)
def fleurs_digits_lang(lang: str, split: str, dest: str, prefix: str, spanned: bool = False) -> dict:
    """FLEURS rows of `split` whose raw caption has digits -> manifest `dest` (volume path) + wav/<prefix>_*.wav, each
    row with its OWN recording (no D2 speaker mix-up). Run-2 (2026-10-16): split dev = decision set (wave-1 cands with
    spanned=True for labelling; the 18 others raw, digits as references); split train = the 18 languages' digit anchor."""
    import io
    import tarfile
    from types import SimpleNamespace

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import ml_cands as M

    code = FLEURS_CODES[lang]
    wdir = f"{VOL}/fleursdig/{split}/{lang}"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(os.path.dirname(f"{VOL}/{dest}"), exist_ok=True)
    tsv = open(hf_hub_download("google/fleurs", f"data/{code}/{split}.tsv", repo_type="dataset", local_dir="/tmp/fl"),
               encoding="utf8").read()
    recs = [SimpleNamespace(sid=c[0], file_name=c[1], raw=c[2]) for c in (l.rstrip("\r").split("\t") for l in tsv.split("\n"))
            if len(c) >= 7 and c[0].isdigit()]
    want = {}
    for r in recs:
        cap = r.raw.strip()
        if not re.search(r"\d", cap) or (spanned and not M.all_digits_spanned(cap, lang)):
            continue
        kinds = sorted({d["kind"] for d in M._find(cap, lang)} - {"skip"}) if spanned else []
        want[r.file_name] = (r, cap, kinds)
    tar = hf_hub_download("google/fleurs", f"data/{code}/audio/{split}.tar.gz", repo_type="dataset", local_dir="/tmp/fl")
    rows = []
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not m.isfile() or name not in want:
                continue
            r, cap, kinds = want[name]
            x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            stem = os.path.splitext(name)[0]
            wav = f"{wdir}/{prefix}_{stem}.wav"
            sf.write(wav, x, sr, subtype="PCM_16")
            rows.append({"audio_filepath": wav, "duration": round(len(x) / sr, 3), "text": cap, "lang": lang, "src": "fleurs",
                         "id": f"{prefix}_{r.sid}_{stem}", "origin": "new", "kinds": kinds})
    with open(f"{VOL}/{dest}", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    return {"lang": lang, "split": split, "recordings": len(recs), "with_digits": len(want), "written": len(rows),
            "h": round(sum(r["duration"] for r in rows) / 3600, 2), "dest": dest}


@app.local_entrypoint()
def fleurs_run2():
    """Run-2 data: wave-1 FLEURS-dev candidates (decision set, labelled as val_fleursdev), FLEURS-dev digit rows of the
    other 18 (decision set, raw), FLEURS-train digit rows of the other 18 (digit anchor, tagged nonen)."""
    other = [l for l in FLEURS_CODES if l not in LANGS]
    jobs = [(l, "dev", f"ml/p3v2/{l}/cands_fleursdev.jsonl", "fleursdev", True) for l in LANGS]
    jobs += [(l, "dev", f"mix4/dec18_{l}.jsonl", "fleursdev", False) for l in other]
    jobs += [(l, "train", f"mix4/d2anchor_{l}.jsonl", "fleurstr", False) for l in other]
    for r in fleurs_digits_lang.starmap(jobs):
        print(json.dumps(r))


@app.local_entrypoint()
def fleurs_val(langs: str = ",".join(LANGS)):
    for r in fleurs_val_lang.map(langs.split(",")):
        print(json.dumps(r))


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=6 * 3600, max_containers=40)
def scan2_lang(lang: str, src: str, rare_hours: float = 3.0, int_hours: float = 1.5, max_files: int = 400,
               start_file: int = 0, tag: str = "", only_kinds: str = "", max_dur: float = 20.0) -> dict:
    """only_kinds (comma list): the rare kinds this pass collects (default: all of RARE); a row with none of them is
    skipped by quota when int_hours is 0. max_dur: the longest row kept (s); above 20, label2 splits long rows at
    pauses (2026-10-16 probe: 14-33% of de/es/fr/it digit rows are longer than 20 s)."""
    import io
    from collections import Counter

    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    import ml_cands as M

    out = f"{OUT2}/{lang}"
    os.makedirs(f"{out}/wav", exist_ok=True)
    seen = {json.loads(l)["id"] for l in open(f"{VOL}/d2/{lang}/audit.jsonl")}
    import glob as _g

    for pp in sorted(_g.glob(f"{out}/cands_*.jsonl")):  # never rescan a row an earlier pass kept (every pass)
        if pp != f"{out}/cands_{src}{tag}.jsonl":
            seen |= {json.loads(l)["id"] for l in open(pp)}
    # "year" (2026-10-16) is a scan target too: years sit inside the int kind, so without it the int quota decided
    # which years were kept (one German shard skipped 93k rows); a year-bearing caption counts as kind "year"
    target = set(only_kinds.split(",")) & (RARE | {"year"}) if only_kinds else RARE
    assert target, f"only_kinds {only_kinds!r} names no rare kind"
    files, repo, cols = _src_files(HfApi(), lang, src)
    files = files[start_file:]  # a second, class-targeted pass continues where the first stopped
    st = {"lang": lang, "src": src, "files": 0, "rows_seen": 0, "digit_rows": 0, "kept_rare": 0, "kept_int": 0,
          "hours_rare": 0.0, "hours_int": 0.0, "skipped_unspannable": 0, "skipped_dur": 0, "skipped_seen": 0,
          "skipped_quota": 0, "rare_kinds": Counter()}
    kind_h = Counter()  # review H2: one quota per rare kind (rare_hours / len(RARE)), so dec/ord do not fill the pool
    per_kind = rare_hours / len(target)
    rows = []
    cf = open(f"{out}/cands_{src}{tag}.jsonl", "w")  # written as rows are found: a crash keeps what was scanned

    def full():
        return all(kind_h[k] >= per_kind for k in target) and st["hours_int"] >= int_hours

    for path in files[:max_files]:
        if full():
            break
        local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/src")
        st["files"] += 1
        pf = pq.ParquetFile(local)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                st["rows_seen"] += 1
                cap = (r.get("raw_text") or r.get("text") or "").strip()
                rid = r.get("audio_id") or r.get("utt_id")
                if not re.search(r"\d", cap):
                    continue
                st["digit_rows"] += 1
                if rid in seen:
                    st["skipped_seen"] += 1
                    continue
                try:  # one odd caption must not kill a multi-hour scan
                    if lang not in M.LANGS:  # raw harvest (ru/uk, 2026-10-16): no span tables yet, labelled later
                        spannable, kinds = True, []
                    else:
                        spannable = M.all_digits_spanned(cap, lang)
                        kinds = sorted({d["kind"] for d in M._find(cap, lang)}) if spannable else []
                except Exception:  # noqa: BLE001
                    st["skipped_error"] = st.get("skipped_error", 0) + 1
                    continue
                if not spannable:
                    st["skipped_unspannable"] += 1
                    continue
                if "year" in target and re.search(r"(?<![\d.,])(1[89]\d\d|20\d\d)(?![\d%]|[.,]\d)", cap):
                    kinds = sorted(set(kinds) | {"year"})
                rk = target & set(kinds)
                rare = bool(rk)
                if (rare and all(kind_h[k] >= per_kind for k in rk)) or (not rare and st["hours_int"] >= int_hours):
                    st["skipped_quota"] += 1
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                d = len(x) / 16000
                if d < 1.0 or d > max_dur:
                    st["skipped_dur"] += 1
                    continue
                seen.add(rid)
                wav = f"{out}/wav/{src}{tag}_{rid}.wav"
                sf.write(wav, x, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": wav, "duration": round(d, 3), "text": cap, "lang": lang, "src": "yodas" if src == "yodasast" else src,
                             "id": rid, "origin": "new", "kinds": kinds})
                cf.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                k = "rare" if rare else "int"
                st[f"kept_{k}"] += 1
                st[f"hours_{k}"] += d / 3600
                for kk in rk:
                    st["rare_kinds"][kk] += 1
                    kind_h[kk] += d / 3600
                if full():
                    break
            if full():
                break
        os.remove(local)
        cf.flush()
        vol.commit()
    cf.close()
    st["hours_rare"], st["hours_int"] = round(st["hours_rare"], 2), round(st["hours_int"], 2)
    st["rare_kinds"] = dict(st["rare_kinds"])
    st["kind_hours"] = {k: round(v, 3) for k, v in kind_h.items()}
    json.dump(st, open(f"{out}/scan_stats_{src}{tag}.json", "w"))
    vol.commit()
    return st


def _diag(row, r, res) -> dict:
    return {"id": row["id"], "origin": row["origin"], "src": row["src"],
            "reason": res.get("reason") or ("kept_salvage" if res.get("salvaged") else "kept"), "trim": row.get("trim"),
            "greedy": r["greedy"],  # review M8: full text + every candidate's score, so later filters need no GPU pass
            "spans": [{k: d.get(k) for k in ("text", "kind", "decided", "margin", "fit", "why", "by")}
                      | ({"pick": d["cands"][d["pick"]], "n": len(d["cands"]),
                          "cands": d["cands"], "scores": d.get("scores")} if "pick" in d else {}) for d in r["spans"]]}


@app.function(image=gpu_image2, volumes={VOL: vol}, gpu="L4", timeout=6 * 3600, max_containers=12)
def label2_lang(lang: str, limit: int = 0, int_new_hours: float = 1.5, crops: bool = True, dry_per_kind: int = 0,
                srcs: str = "vox,yodas", with_d2: bool = True, name: str = "rows", ext: str = "",
                cer_ext: bool = True, split_long: bool = True, max_row: float = 20.0, only_ids: str = "",
                trim_ext: bool = True, piece_ext: bool = True, salvage: bool = True, decide_off: str = "",
                cands_off: str = "", partial: bool = True, ctc_model: str = "") -> dict:
    """ext: a file in <lang>/ of {"id", "text"} verbatim transcripts by a second recognizer. When given,
    ONLY those ids are labelled, and a span the CTC greedy text misspells is decided when ext agrees (ml_decide).
    cer_ext (2026-10-16): the row CER / caption-gap guards take the better of the CTC greedy and the ext transcript
    (--no-cer-ext = the old greedy-only guard). split_long: a kept row longer than max_row s (scan2 --max-dur > 20) is cut
    at pauses into pieces (ml_p3_core.plan_splits); each piece must match its own decode (CER, edge words) and hold a
    number. only_ids: a file in <lang>/ of ids to label (one per line); like dry_per_kind it writes to <lang>/dry/.
    Parliament fixes (2026-10-16), each needing ext rows with word times ("words": [{"w", "s", "e"}] ms) except the
    last: trim_ext cuts a row's audio to its caption's stretch of the external transcript (ml_p3_core.ext_trim) before
    decoding; piece_ext lets a piece of a long row pass on the transcript words inside it (ml_p3_core.piece_ok);
    salvage keeps the verified pieces of a long row that hold none of its undecided spans; partial (2026-10-16, with
    salvage) cuts any row right around its undecided spans and around a long row's failing edge word, keeping the
    verified pieces (origin "partial"; --no-partial = the old salvage). decide_off / cands_off:
    comma lists of ml_decide.FIXES / ml_cands.CAND_FIXES names to switch off (the old decisions / candidates).
    ctc_model: another CTC reader than ml_decide.MODELS[lang] ("strong" = ml_decide.STRONG[lang]); give the pass its own
    --name so the outputs of the default reader stay."""
    import random
    import shutil

    import librosa
    import soundfile as sf

    import ml_decide as D
    import ml_p3_core as C

    import ml_cands as M

    # a warm container serves the next call too: start from the full switch sets, then turn this call's off
    D.FIXES.clear(); D.FIXES.update(D.ALL_FIXES)
    M.CAND_FIXES.clear(); M.CAND_FIXES.update(M.ALL_CAND_FIXES)
    D.FIXES -= set(filter(None, decide_off.split(",")))
    M.CAND_FIXES -= set(filter(None, cands_off.split(",")))
    ctc = D.LangCTC(lang, model=(D.STRONG[lang] if ctc_model == "strong" else ctc_model) or None)
    dry = bool(dry_per_kind or only_ids)
    out = f"{OUT2}/{lang}" + ("/dry" if dry else "")  # a dry run never touches the real outputs
    os.makedirs(f"{out}/wav", exist_ok=True)
    # audio this pass cuts (trims, pieces) lives under its own name: passes of one source run side by side and reuse
    # row ids, and a shared path let one pass overwrite the audio another pass's labels point at
    wdir = f"{out}/wav/{name}"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(f"{out}/crops", exist_ok=True)
    d2 = [dict(json.loads(l), origin="d2") for l in open(f"{VOL}/d2/{lang}/digit.jsonl")] if with_d2 else []
    fix = f"{VOL}/d2fix/{lang}/digit.jsonl"
    if with_d2 and os.path.exists(fix):
        # FLEURS rows re-fetched per recording (ml_fleurs_fix: sentence ids repeat across speakers, so the D2 wav of
        # a FLEURS row was often another speaker's): they replace D2's FLEURS rows, the other D2 rows stay
        fx = [json.loads(l) for l in open(fix)]
        for r in fx:
            r.setdefault("origin", "d2fix")
            r["id"] = f"{r['id']}_{r.get('speaker') or r.get('file_name', '')}"
        d2 = [r for r in d2 if r.get("src") != "fleurs" and not str(r["id"]).startswith("fleurs_")] + fx
    new = []
    for src in srcs.split(","):  # a later pass (e.g. "yodas2") labels only its own source into rows_<pass>.jsonl
        p = f"{OUT2}/{lang}/cands_{src}.jsonl"
        if os.path.exists(p):
            new += [json.loads(l) for l in open(p)]
    random.Random(lang).shuffle(new)
    ext_t, ext_w = None, {}
    if ext:
        ext_rows = [r for r in map(json.loads, open(f"{OUT2}/{lang}/{ext}")) if r.get("status", "completed") == "completed"]
        ext_t = {r["id"]: r.get("text") or "" for r in ext_rows}
        ext_w = {r["id"]: r["words"] for r in ext_rows if r.get("words")}
        del ext_rows
        new = [r for r in new if r["id"] in ext_t]
        d2 = [r for r in d2 if r["id"] in ext_t]  # review H4: an ext pass labels ONLY its ext ids (never D2 again)
    # dedupe (leakage check 2026-10-15): one VoxPopuli utterance can sit under two agenda ids
    # ("...PLENARY-18-de_20180704-18:01:49_9" / "...PLENARY-de_20180704-18:01:49_9"); YODAS repeats short phrases
    # ("1 2 3") across videos. Key on the time suffix / on the text of rows of <= 4 words.
    seen_k, dd = set(), []
    for r in d2 + new:
        m = re.search(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$", str(r["id"]))
        k = (m.group(1) if m else None) or (r["text"].lower() if len(r["text"].split()) <= 4 else ("id", r["id"]))
        if k in seen_k:
            continue
        seen_k.add(k)
        dd.append(r)
    d2, new = [r for r in dd if r["origin"] in ("d2", "d2fix")], [r for r in dd if r["origin"] not in ("d2", "d2fix")]
    if limit:
        d2, new = d2[:limit], new[:limit]
    if dry_per_kind:  # dry run: up to N rows per span kind (by the CURRENT ml_cands), so every class is exercised
        import ml_cands as M

        by, pick = {}, []
        for r in d2 + new:
            for k in sorted({d["kind"] for d in M._find(r["text"], lang)} - {"skip"}):
                if len(by.setdefault(k, [])) < dry_per_kind:
                    by[k].append(r["id"])
        ids = {i for v in by.values() for i in v}
        d2, new = [r for r in d2 if r["id"] in ids], [r for r in new if r["id"] in ids]
        int_new_hours = 1e9
    if only_ids:  # dry run on a fixed id sample (yield audit)
        ids = _id_set(f"{OUT2}/{lang}/{only_ids}")
        d2, new = [r for r in d2 if r["id"] in ids], [r for r in new if r["id"] in ids]
        int_new_hours = 1e9
    rng = random.Random(f"crops-{lang}")
    acc = C.Acc(lang)
    kept, crop_rows, kept_int_h = [], [], 0.0
    cst = {"planned": 0, "kept": 0, "cer": 0, "edge_word": 0, "no_times": 0}
    pst = {"trimmed": 0, "trim_kept": 0, "salvage_tried": 0, "salvage_rows": 0, "salvage_pieces": 0, "piece_by_ext": 0}
    sfx = "" if name == "rows" else name[len("rows"):] if name.startswith("rows") else "_" + name
    diag = open(f"{out}/diag{sfx}.jsonl", "w")
    rows_f, crops_f = open(f"{out}/{name}.jsonl", "w"), open(f"{out}/crops{sfx}.jsonl", "w")
    n_done = 0
    for row in d2 + new:
        n_done += 1
        if n_done % 500 == 0:  # review H1: a crash late in a multi-hour pass keeps what was written
            for fh in (diag, rows_f, crops_f):
                fh.flush()
            vol.commit()
        if row["origin"] in ("d2", "d2fix") and abs(sf.info(row["audio_filepath"]).duration - row["duration"]) > 0.05:
            acc.add_skip(row, "d2_audio_mismatch")  # D2 wav overwritten by a same-id FLEURS row (see label_lang)
            continue
        rare = bool(RARE & set(row.get("kinds") or []))
        if row["origin"] == "new" and not rare and kept_int_h >= int_new_hours * 3600:
            acc.add_skip(row, "new_quota_reached")
            continue
        try:
            x, sr = sf.read(row["audio_filepath"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            row0 = row
            ext_row = ext_t.get(row["id"]) if ext_t is not None else None
            words = ext_w.get(row["id"])
            tr = C.ext_trim(row["text"], words, x) if trim_ext and words else None
            if tr:  # the caption covers only part of the audio: cut the audio to it
                x = x[int(tr["a"] * 16000):int(tr["b"] * 16000)]
                words = C.shift_words(words, tr["a"], tr["b"])
                ext_row = " ".join(w["w"] for w in words)
                twav = f"{wdir}/{row['id']}_t.wav"
                sf.write(twav, x, 16000, subtype="PCM_16")
                row = dict(row, audio_filepath=twav, duration=round(len(x) / 16000, 3), trim=[tr["a"], tr["b"]])
                pst["trimmed"] += 1
            lp = ctc.lp(x)
            r = D.decide(ctc, row["text"], lp, ext=ext_row)
        except Exception as e:  # noqa: BLE001
            acc.add_skip(row, f"error {type(e).__name__}")
            continue
        res = C.make_row(row, r, x, ext=ext_row if cer_ext else None)
        pw = words if piece_ext else None
        pieces = None
        if res.get("row") and split_long and row["duration"] > max_row:
            pieces, res = _split_pieces(ctc, C, lp, x, res, row, wdir, max_row, pw, pst, partial=partial)
        elif (res.get("reason") == "undecided" and salvage and (partial or (split_long and row["duration"] > max_row))
              and all("pick" in d for d in r["spans"])):
            # salvage: the row's undecided spans stay out, its other pieces are kept when each one checks out. With
            # partial (2026-10-16) the cuts are made right around the undecided spans (C.plan_partial), for a row of
            # any length; the pieces are origin "partial"
            pst["salvage_tried"] += 1
            r2 = dict(r, spans=[d if d["decided"] else dict(d, decided=True, salvage=True) for d in r["spans"]])
            res2 = C.make_row(row, r2, x, ext=ext_row if cer_ext else None)
            if res2.get("row"):
                pieces2, res2 = _split_pieces(ctc, C, lp, x, res2, row, wdir, max_row, pw, pst, skip_undecided=True,
                                              partial=partial)
                if pieces2:
                    pieces, res = pieces2, dict(res2, salvaged=True)
                    pst["salvage_rows"] += 1
                    pst["salvage_pieces"] += len(pieces2)
        if tr and res.get("row"):
            pst["trim_kept"] += 1
        acc.add(row0, r, res)
        diag.write(json.dumps(_diag(row, r, res), ensure_ascii=False) + "\n")
        if not res.get("row"):
            continue
        if pieces is not None:  # a long row: its verified pieces replace it (no crops)
            for pr in pieces:
                kept.append(pr)
                rows_f.write(json.dumps(pr, ensure_ascii=False) + "\n")
                if row["origin"] == "new" and not rare:
                    kept_int_h += pr["duration"]
            continue
        nr = res["row"]
        dst = f"{out}/wav/{row['origin']}_{row['id']}.wav" if row["origin"] == "d2" else row["audio_filepath"]
        if row["audio_filepath"] != dst and not dry:
            shutil.copyfile(row["audio_filepath"], dst)
        if dry:
            dst = row["audio_filepath"]
        nr["audio_filepath"] = dst
        toks = nr["text"].split()
        try:
            times = ctc.word_times(lp, nr["text"])
        except Exception:  # noqa: BLE001  (review H1: one alignment failure must not lose the language)
            times = [None] * len(toks)
        nr["word_times"] = [None if t is None else [round(t[0], 3), round(t[1], 3)] for t in times]
        kept.append(nr)
        rows_f.write(json.dumps(nr, ensure_ascii=False) + "\n")
        if row["origin"] == "new" and not rare:
            kept_int_h += nr["duration"]
        if not crops:
            continue
        if all(t is None for t in times):
            cst["no_times"] += 1
            continue
        flags = C.number_tokens(nr["text"], nr["num_chars"])
        for k, cp in enumerate(C.plan_crops(len(toks), flags, times, x, rng)):
            cst["planned"] += 1
            xc = x[int(cp["a"] * 16000):int(cp["b"] * 16000)]
            label = " ".join(toks[cp["i0"]:cp["i1"]])
            g = ctc.greedy(ctc.lp(xc))
            if C.cer(label, g) > C.CROP_MAX_CER:
                cst["cer"] += 1
                continue
            if not C.edges_match(label, g):
                cst["edge_word"] += 1  # an edge word clipped, or a neighbour's tail left in the audio (review H3)
                continue
            wav = f"{out}/crops/{nr['id']}_c{k}.wav"
            sf.write(wav, xc, 16000, subtype="PCM_16")
            sub = flags[cp["i0"]:cp["i1"]]
            crop_rows.append({"audio_filepath": wav, "duration": round(len(xc) / 16000, 3), "text": label,
                              "lang": lang, "src": nr["src"], "id": f"{nr['id']}_c{k}", "origin": "crop",
                              "parent": nr["id"], "crop_type": cp["type"], "num_tokens": sub})
            crops_f.write(json.dumps(crop_rows[-1], ensure_ascii=False) + "\n")
            cst["kept"] += 1
    for fh in (diag, rows_f, crops_f):
        fh.close()
    st = acc.stats(kept)
    st["crops"] = cst
    st["parl"] = pst
    st["crop_types"] = dict(__import__("collections").Counter(c["crop_type"] for c in crop_rows))
    st["crop_hours"] = round(sum(c["duration"] for c in crop_rows) / 3600, 3)
    json.dump(st, open(f"{out}/stats{sfx}.json", "w"), ensure_ascii=False, indent=1)
    vol.commit()
    return st


def _split_pieces(ctc, C, lp, x, res, row, wdir, max_row, words=None, pst=None, skip_undecided=False, partial=False):
    """(pieces, res) for a kept row longer than max_row: cut at pauses (C.plan_splits); a piece is kept when it holds a
    number, its own CTC decode matches its label (row CER bound of the source) and its first / last words are heard
    (C.edges_match) -- or, with transcript words (piece_ext), the transcript words inside it do (C.piece_ok).
    skip_undecided (salvage): a piece holding a span marked undecided is left out. No piece kept -> res becomes a drop
    ("long_no_split" / "long_pieces_failed").
    partial (2026-10-16): with skip_undecided the cuts are made right around the undecided spans (C.plan_partial) and
    the row may be of any length; a piece at the row's start / end that fails its check is tried once more without its
    edge word (a caption that runs past the audio's edge), cut at the next usable pause. Pieces of a partial pass are
    origin "partial"."""
    import soundfile as sf

    nr = res["row"]
    toks = nr["text"].split()
    try:
        times = ctc.word_times(lp, nr["text"])
    except Exception:  # noqa: BLE001
        times = [None] * len(toks)
    flags = C.number_tokens(nr["text"], nr["num_chars"])
    if partial and skip_undecided:
        plan = C.plan_partial(len(toks), flags, C.undecided_tokens(nr["text"], nr["num_chars"], nr["spans"]), times, x,
                              max_s=max_row)
        if not plan:
            return None, {"reason": "undecided"}
    else:
        plan = C.plan_splits(len(toks), flags, times, x, max_s=max_row)
        if not plan:
            return None, {"reason": "long_no_split"}
    dur = len(x) / 16000

    def check(pr):
        xc = x[int(pr["piece"][0] * 16000):int(pr["piece"][1] * 16000)]
        lpc = ctc.lp(xc)
        g = ctc.greedy(lpc)
        ok, _, ce = C.piece_ok(pr["text"], g, C.CER_BY_SRC.get(row.get("src"), C.MAX_CER), words, pr["piece"][0],
                               pr["piece"][1], dur)
        return ok, ce, xc, lpc, g

    pieces, n_pc = [], 0
    for p0 in plan:
        pr = C.split_rows(nr, [p0])[0]
        if not pr["num_chars"]:
            continue
        if skip_undecided and any(sp.get("undecided") for sp in pr["spans"]):
            continue
        ok, ce, xc, lpc, g = check(pr)
        if not ok and partial and (p0["i0"] == 0 or p0["i1"] == len(toks)):
            # the row's edge word may be one the audio lacks: leave it out and cut at the next pause
            edge = ({p0["i0"]} if p0["i0"] == 0 else set()) | ({p0["i1"] - 1} if p0["i1"] == len(toks) else set())
            bad = set(range(0, p0["i0"])) | set(range(p0["i1"], len(toks))) | edge
            for p1 in C.plan_partial(len(toks), flags, bad, times, x, max_s=max_row):
                pr1 = C.split_rows(nr, [p1])[0]
                if not pr1["num_chars"]:
                    continue
                ok1, ce1, xc1, lpc1, g1 = check(pr1)
                if ok1:
                    p0, pr, ok, ce, xc, lpc, g = p1, pr1, ok1, ce1, xc1, lpc1, g1
                    pr["edge_retry"] = True
                    break
        if not ok:
            continue
        if ce is not None and pst is not None:
            pst["piece_by_ext"] += 1
        try:
            tc = ctc.word_times(lpc, pr["text"])
        except Exception:  # noqa: BLE001
            tc = [None] * len(pr["text"].split())
        pr["id"] = f"{nr['id']}_p{n_pc}"
        n_pc += 1
        if partial and (skip_undecided or pr.get("edge_retry")):
            pr["origin"] = "partial"
        wav = f"{wdir}/{pr['id']}.wav"
        sf.write(wav, xc, 16000, subtype="PCM_16")
        pieces.append(pr | {"audio_filepath": wav, "cer_ctc": round(C.cer(pr["text"], g), 3), "edge_quiet": C.edge_quiet(xc),
                            "word_times": [None if t is None else [round(t[0], 3), round(t[1], 3)] for t in tc]})
    if not pieces:
        return None, {"reason": "undecided" if skip_undecided and partial else "long_pieces_failed"}
    return pieces, {"row": dict(nr, duration=round(sum(p["duration"] for p in pieces), 3))}


WIN_MAX_CER = 0.15
# Languages beyond wave 1 / ru (2026-10-17). Their CTC reader comes from ml_multi_ctc.READERS (passed in by the
# entrypoints), their synthetic rows from the tags below, and rows without word times (YODAS look-ups, synthetic rows,
# read clips) are force-aligned with the reader before stitching. A held-out parliament DAY excludes every session of
# that day, committees included.
WAVE2 = ("pl", "ro", "uk", "sv", "cs", "hu", "fi", "sk", "bg", "hr")
WAVE2_TTS = {"pl": ("ttsm1",), "ro": ("ttsm1",), "sv": ("ttsm2",), "cs": ("ttsm2",), "hu": ("ttsm2",), "fi": ("ttsm2",),
             "sk": ("ttsm2",), "uk": ("tts6uk",)}
# uk read-speech clips (Common Voice): a different speaker every few seconds, and one sentence read by up to three
# speakers, so they are never stitched into streams. They give seam pairs cut inside the one clip, no windows.
SOLO_SRCS = {"uk": ("cv22",)}


def _num_keep(lang: str):
    """ml_p3_core.num_keep_frac, plus every WAVE2 language: their number words are compounds (sv / hu / fi) or long
    case-inflected forms (the Slavic languages, ro), so an edge that cuts one keeps it only when ~whole (>= 0.9)."""
    import ml_p3_core as C

    return C.num_keep_frac(lang) or (C.NUM_KEEP_FRAC if lang in WAVE2 else None)


def _reader_model(lang: str) -> str:
    """'' when ml_decide.MODELS has the language, else its ml_multi_ctc reader (resolved locally: that module builds
    its own Modal app)."""
    import ml_decide as D

    if lang in D.MODELS:
        return ""
    import ml_multi_ctc as MC

    return MC.READERS[lang]


def _lang_ctc(D, lang: str, model: str):
    """ml_decide.LangCTC; a reader with an UPPERCASE alphabet (sv / hu / fi) looks letters up in that case, as
    ml_multi_ctc.score_lang does, so forced alignment does not come back empty."""

    class CaseCTC(D.LangCTC):
        def ids(self, text):
            letters = [k for k in self.vocab if len(k) == 1 and k.isalpha()]
            if letters and sum(k.isupper() for k in letters) > len(letters) / 2:
                out = []
                for w in re.sub(r"[^\w' ]+", " ", text.upper()).split():
                    cs = [self.vocab[c] for c in w if c in self.vocab]
                    if cs:
                        if out and self.delim is not None:
                            out.append(self.delim)
                        out += cs
                return out
            return super().ids(text)

    return CaseCTC(lang, model=model or None)


def _vox_suffix(rid: str):
    """VoxPopuli time suffix: one utterance can sit under two agenda ids (leakage check 2026-10-15)."""
    m = re.search(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$", str(rid))
    return m.group(1) if m else None


def _id_set(path: str) -> set:
    return {l.strip() for l in open(path) if l.strip()} if os.path.exists(path) else set()


def _window_kinds(stream: list, w: dict) -> list:
    """Number kinds whose words this window labels (the census must see window classes): parent row i,
    word k -> the span whose num_chars range holds that word's first character."""
    out = set()
    for i, k in w["src"]:
        r = stream[i]
        starts, pos = [], 0
        for tok in r["text"].split(" "):
            starts.append(pos)
            pos += len(tok) + 1
        c = starts[k] if k < len(starts) else -1
        for sp, (a, b) in zip(r.get("spans") or [], r.get("num_chars") or []):
            if a <= c < b:
                out.add(sp.get("kind", "?"))
    return sorted(out)


@app.function(image=gpu_image2, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, max_containers=6)
def windows2_lang(lang: str, per_stream: int = 3, max_rows: int = 8, seed: int = 1, allow_no_judge: bool = False,
                  include_tts: bool = True, seam: bool = False, seam_positions: int = 0, seam_limit: int = 0,
                  seam_tag: str = "", ctc_model: str = "", win_max_h: float = 0.0, heldout: bool = False,
                  heldout_tts: str = "", seam_min_words: int = 0) -> dict:
    """Streaming-geometry windows (ml_p3_core.build_windows) over stitched streams of this language's labelled rows. Held-out
    ids (/vol/ml/p3v2/heldout_<lang>.txt) and judge-rejected ids (<lang>/reject_ids.txt) never enter a stream. Each window
    is re-decoded: kept when CER(label, CTC greedy) <= WIN_MAX_CER and its first and last words match. per_stream = random
    phases per stream. Writes <lang>/windows.jsonl + win/*.wav.
    ctc_model: the reader for a language ml_decide.MODELS lacks (_reader_model). WAVE2 languages: see WAVE2 above.
    win_max_h > 0: streams in random order, stop once that many hours of windows are kept (build4 takes win_h of them).
    heldout=True (seam gate, 2026-10-17): the same windows / seam pairs built from the HELD-OUT rows only (held-out ids,
    their videos / days, val_* passes; minus judge rejects and second-recognizer rows; no synthetic rows) ->
    /vol/ml/p3v2/heldout_win_<lang>.jsonl (windows) or heldout_seam_<lang>.jsonl (seam=True). Eval only.
    seam_min_words > 0 (seam=True, 2026-10-17): only cuts INSIDE a reading of >= that many words (gap / in-word
    positions; no before / after) -- the long-number seam part (seam_long)."""
    import random
    from collections import Counter

    import librosa
    import soundfile as sf

    import ml_decide as D
    import ml_p3_core as C

    out = f"{OUT2}/{lang}"
    os.makedirs(f"{out}/win", exist_ok=True)
    # review H3: a missing held-out or judge file must stop the job, not silently train on held-out / rejected rows
    assert os.path.exists(f"{OUT2}/heldout_{lang}.txt"), f"{lang}: held-out list missing"
    assert allow_no_judge or os.path.exists(f"{out}/reject_ids.txt"), f"{lang}: reject_ids.txt missing (run the judge)"
    held = _id_set(f"{OUT2}/heldout_{lang}.txt")
    import glob as _gv

    held |= {json.loads(l)["id"] for p in _gv.glob(f"{out}/val_*.jsonl") for l in open(p)}  # validation-only rows
    held_sfx = {_vox_suffix(i) for i in held} - {None}
    import glob as _g

    rows_all = [json.loads(l) for p in sorted(_g.glob(f"{out}/rows*.jsonl")) for l in open(p)]  # every labelling pass
    vid = lambda i: (re.match(r"[a-z]{2}\d+_\d+_(.{11})_", str(i)) or [None, None])[1]  # noqa: E731
    held_vid = {vid(i) for i in held} - {None}  # the held-out unit is the YODAS video
    day = lambda i: (re.match(r"(\d{8})-\d{4}-PLENARY", str(i)) or [None, None])[1]  # noqa: E731
    held_day = {day(i) for i in held} - {None}  # review H2: and the VoxPopuli plenary day
    skip = held | _id_set(f"{out}/reject_ids.txt") | {r["id"] for r in rows_all if _vox_suffix(r["id"]) in held_sfx
                                                       or vid(r["id"]) in held_vid or day(r["id"]) in held_day}
    # rows decided by a second recognizer (span by in EXT_BY: "ext" and "ext_pick", as in mix_build.build4) never enter
    # windows or pairs
    from ml_balance import EXT_BY

    ext_ids = {r["id"] for r in rows_all if any(sp.get("by") in EXT_BY for sp in r.get("spans") or [])}
    skip |= ext_ids
    wave2 = lang in WAVE2
    if wave2:
        hdays = C.held_days(held)
        skip |= {r["id"] for r in rows_all if C.vox_day(r["id"]) in hdays}
    if heldout:  # eval windows: the held-out side of the same split (val_* rows included), never training rows
        val_rows = [json.loads(l) for p in sorted(_g.glob(f"{out}/val_*.jsonl")) for l in open(p)]
        ext_ids |= {r["id"] for r in val_rows if any(sp.get("by") in EXT_BY for sp in r.get("spans") or [])}
        rows_all = rows_all + val_rows
        ho_ids = (skip | {r["id"] for r in val_rows}) - _id_set(f"{out}/reject_ids.txt") - ext_ids
        include_tts = False
        # heldout_tts ("tag,tag"): also the synthetic sets' held-out VOICES (split "dev"), for languages whose real
        # held-out rows are too few for a seam check (moderate tier); their pairs carry synthetic=True
        for tag in [t for t in heldout_tts.split(",") if t]:
            tts_p = f"{VOL}/ml/{tag}/{lang}/rows.jsonl"
            if os.path.exists(tts_p):
                dev = [dict(r, origin="tts") for r in map(json.loads, open(tts_p)) if r.get("split") == "dev"]
                rows_all = rows_all + dev
                ho_ids |= {r["id"] for r in dev} - _id_set(f"{out}/reject_ids.txt")
        skip = {r["id"] for r in rows_all} - ho_ids  # from here on "skip" = every row that is NOT held out
    # synthetic adversarial rows (ml_tts; tts2 = thin-class top-up; WAVE2_TTS) join the streams
    for tag in ("tts1", "tts2") + (WAVE2_TTS.get(lang, ()) if wave2 else ()):
        tts_p = f"{VOL}/ml/{tag}/{lang}/rows.jsonl"
        if include_tts and os.path.exists(tts_p):
            rows_all = rows_all + [dict(r, origin="tts") for r in map(json.loads, open(tts_p))
                                   if r.get("split", "train") == "train"]  # dev voices stay held out
    rng = random.Random(f"win-{lang}-{seed}")
    ctc = _lang_ctc(D, lang, ctc_model) if wave2 else D.LangCTC(lang)
    cache = {}
    n_aligned = 0
    if wave2:  # rows without word times: force-align the label against the row's own audio
        rows_all = list(rows_all)
        for j, r in enumerate(rows_all):
            if r["id"] in skip or C.has_times(r) or (not seam and r.get("src") in SOLO_SRCS.get(lang, ())):
                continue
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            x = x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)
            wt = ctc.word_times(ctc.lp(x), r["text"])
            rows_all[j] = dict(r, word_times=[None if t is None else [round(t[0], 3), round(t[1], 3)] for t in wt],
                               times_aligned=True)
            n_aligned += C.has_times(rows_all[j])
        rows = [r for r in rows_all if r["id"] not in skip and C.has_times(r)]
    else:
        rows = [r for r in rows_all if r["id"] not in skip and r.get("word_times") and all(t is not None for t in r["word_times"])]
    assert not {r["id"] for r in rows} & skip, "held-out / rejected row in the stream rows"
    solo = []
    if wave2 and SOLO_SRCS.get(lang):
        rows, solo = C.split_solo(rows, SOLO_SRCS[lang])

    def read(p):
        if p not in cache:
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            cache[p] = x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)
        return cache[p]

    if seam:  # 2026-10-16 balance: mirrored cut-number crops (ml_balance); windows.jsonl is not touched
        return _seam_pass(lang, out, rows, rng, ctc, read, cache, max_rows, seam_positions, seam_limit,
                          "_heldout" if heldout else seam_tag, len(skip), solo=solo, n_aligned=n_aligned,
                          dest=f"{OUT2}/heldout_seam_{lang}.jsonl" if heldout else "", heldout=heldout, min_words=seam_min_words)
    st = Counter()
    win_dest, win_dir = (f"{OUT2}/heldout_win_{lang}.jsonl", f"{out}/heldout_win") if heldout else (f"{out}/windows.jsonl", f"{out}/win")
    os.makedirs(win_dir, exist_ok=True)
    streams = C.group_streams(rows, rng, max_rows=max_rows)
    if win_max_h:
        rng.shuffle(streams)  # a capped run samples streams across sources instead of taking the first ones
    with open(win_dest, "w") as f:
        for si, stream in enumerate(streams):
            if win_max_h and st["kept_s"] >= win_max_h * 3600:
                break
            cache.clear()
            # review M1: lay the stream out with the REAL audio lengths (manifest durations may differ by up to 50 ms)
            stream = [dict(r, duration=len(read(r["audio_filepath"])) / 16000) for r in stream]
            _, swords, stotal = C.stream_words(stream)  # census: number words an edge splits (compound rule)
            seen = set()
            for ph in range(per_stream):
                for k, w in enumerate(C.build_windows(stream, rng, num_keep=_num_keep(lang))):
                    key = (round(w["a"], 1), w["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    st["planned"] += 1
                    x = C.window_audio(stream, w["a"], w["b"], read)
                    g = ctc.greedy(ctc.lp(x))
                    if C.cer(w["text"], g) > WIN_MAX_CER:
                        st["cer"] += 1
                        continue
                    if not C.edges_match(w["text"], g, w.get("cut_left"), w.get("cut_right")):
                        st["edge_word"] += 1
                        continue
                    wid = f"{'hw' if heldout else 'w'}{si}_{ph}_{k}"
                    wav = f"{win_dir}/{wid}.wav"
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    f.write(json.dumps({"audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": w["text"],
                                        "lang": lang, "id": f"{lang}_{wid}", "origin": "window",
                                        "num_tokens": w["num_tokens"], "cut_words": w["cut_words"],
                                        "kept_cut_words": w["kept_cut_words"], "edge_fracs": w["edge_fracs"],
                                        "cut_number": w["cut_number"], "edge_number": w["edge_number"],
                                        "parents": [stream[i]["id"] for i in w["rows"]],
                                        "src": [[stream[i]["id"], k] for i, k in w["src"]],
                                        "kinds": _window_kinds(stream, w),
                                        "synthetic": any(stream[i].get("origin") == "tts" for i in w["rows"])},
                                       ensure_ascii=False) + "\n")
                    st["kept"] += 1
                    st["kept_cut_number"] += w["cut_number"]
                    st["kept_cut_word_labelled"] += w["kept_cut_words"]
                    st["kept_edge_number"] += w["edge_number"]
                    st["kept_compound_split"] += C.compound_splits(swords, w["a"], w["b"], total=stotal) > 0
                    st["kept_s"] += len(x) / 16000
            if si % 50 == 0:
                f.flush()
                vol.commit()
    st = dict(st)
    st["kept_h"] = round(st.pop("kept_s", 0.0) / 3600, 2)
    st["rows_used"], st["rows_skipped_heldout_or_rejected"] = len(rows), len(skip)
    if wave2:
        st["rows_aligned"], st["solo_rows_not_windowed"] = n_aligned, len(solo)
    st["heldout"] = heldout
    json.dump(st, open(f"{out}/{'heldout_' if heldout else ''}windows_stats.json", "w"), indent=1)
    vol.commit()
    return {"lang": lang, **st}


@app.local_entrypoint()
def windows2(langs: str = ",".join(LANGS), per_stream: int = 3, allow_no_judge: bool = False, max_h: float = 0.0,
             heldout: bool = False):
    """max_h > 0: stop each language at that many hours of kept windows; heldout: eval windows from held-out rows."""
    args = [(l, per_stream, 8, 1, allow_no_judge, True, False, 0, 0, "", _reader_model(l), max_h, heldout)
            for l in langs.split(",")]
    for r in windows2_lang.starmap(args):
        print(json.dumps(r))


def _seen_pair_keys(out: str, tag: str) -> set:
    """(left parents, right parents, position) of every pair in this language's other seam files (seam.jsonl, seam_b,
    seam_long ...), so a new long-number run adds cuts instead of repeating them. Rows of a pair share its id."""
    import glob

    keys = set()
    for p in glob.glob(f"{out}/seam*.jsonl"):
        if os.path.basename(p) == f"seam{tag}.jsonl":
            continue  # this run's own output
        halves = {}
        for l in open(p):
            if l.strip():
                x = json.loads(l)
                halves.setdefault(x["pair"], {})[x["side"]] = x
        for h in halves.values():
            if "left" in h and "right" in h:
                keys.add((tuple(h["left"]["parents"]), tuple(h["right"]["parents"]), h["left"]["pos"]))
    return keys


def _long_pairs(stream, rng, positions, num_keep, min_words):
    """ml_balance.seam_crops for the readings of >= min_words words only (lazy: the other readings cost planning time
    and are never cut; the long-number part planned every reading of a stream before this filter)."""
    import ml_balance as B
    import ml_p3_core as C

    _, words, total = C.stream_words(stream)
    for rd in B.readings(words, stream):
        if len(rd["idx"]) < min_words:
            continue
        cands = [(pp, c) for pp, c in B.cut_positions(words, rd) if not (num_keep and pp.startswith("in"))]
        if positions and len(cands) > positions:
            cands = rng.sample(cands, positions)
        reading = " ".join(words[j][0] for j in rd["idx"])
        for pos, c in cands:
            pr = B.seam_pair(words, total, c, C.WIN, num_keep)
            if pr is None:
                continue
            if not B.pair_partitions(words, pr, rd):
                B.SEAM_SKIPS["partition"] += 1
                continue
            yield {"reading": reading, "kind": rd["kind"], "pos": pos, "n_words": len(rd["idx"]),
                   "left": B._half_row(words, pr["left"], "left", rd, pos),
                   "right": B._half_row(words, pr["right"], "right", rd, pos)}


def _seam_pass(lang, out, rows, rng, ctc, read, cache, max_rows, positions, limit, tag, n_skip, solo=(), n_aligned=0,
               dest: str = "", heldout: bool = False, min_words: int = 0):
    """Mirrored cut-number crops (ml_balance.seam_crops) over the same streams windows2 uses (held-out / rejected / ext
    rows already out; rows of passes without a judge marker are skipped too). Each half is re-decoded: CER <= WIN_MAX_CER
    and ml_balance.seam_edges_ok (the windows2 edge check with the cut edge read as a cut: a piece of the dropped word
    may show, a kept cut word may come out shortened, the whole dropped word may not); a pair is kept only when both
    halves pass. Writes <lang>/seam<tag>.jsonl (two rows
    per pair, same pair id) + seam<tag>/*.wav + seam_stats<tag>.json.
    min_words > 0: only pairs cut inside a reading of >= min_words words (long numbers); the others are not decoded."""
    import time
    from collections import Counter

    import soundfile as sf

    import ml_balance as B
    import ml_p3_core as C

    import glob

    judged = set()  # "from judged rows": only passes the judge step marked (a pass still being labelled is skipped)
    for p in glob.glob(f"{out}/rows*.jsonl") + (glob.glob(f"{out}/val_*.jsonl") if heldout else []):
        if os.path.exists(f"{out}/judged_{os.path.basename(p)[:-6]}.json"):
            judged |= {json.loads(l)["id"] for l in open(p) if l.strip()}
    if heldout:  # held-out synthetic voices (heldout_tts): their sets are judged as a whole, rejects already out
        judged |= {r["id"] for r in rows if r.get("origin") == "tts"}
    n_unjudged = sum(r["id"] not in judged for r in rows) + sum(s[0]["id"] not in judged for s in solo)
    rows = [r for r in rows if r["id"] in judged]
    solo = [s for s in solo if s[0]["id"] in judged]  # one-row streams (SOLO_SRCS): a seam pair inside the one clip
    os.makedirs(f"{out}/seam{tag}", exist_ok=True)
    seen_pairs = _seen_pair_keys(out, tag) if min_words else None
    st = Counter()
    st["rows_unjudged_skipped"] = n_unjudged
    n_pairs = 0
    t_start = time.time()
    with open(dest or f"{out}/seam{tag}.jsonl", "w") as f:
        streams = C.group_streams(rows, rng, max_rows=max_rows) + list(solo)
        rng.shuffle(streams)  # with a pair limit, sample streams across sources instead of taking the first ones
        for si, stream in enumerate(streams):
            if limit and n_pairs >= limit:
                break
            cache.clear()
            if min_words and not any(len(rd["idx"]) >= min_words for rd in B.readings(C.stream_words(stream)[1], stream)):
                st["streams_without_long"] += 1  # no long reading: nothing to cut (skip the audio reads)
                continue
            stream = [dict(r, duration=len(read(r["audio_filepath"])) / 16000) for r in stream]
            # the long part uses the 0.9 number-word edge rule in EVERY language (no in-word cuts; an edge number word
            # is labelled only when ~whole), so a cut inside a long number never leaves a clipped number word labelled
            nk = C.NUM_KEEP_FRAC if min_words else _num_keep(lang)
            pairs = _long_pairs(stream, rng, positions, nk, min_words) if min_words else B.seam_crops(stream, rng, positions, num_keep=nk)
            for k, p in enumerate(pairs):
                if limit and n_pairs >= limit:
                    break
                if min_words and (p["n_words"] < min_words or p["pos"] in ("before", "after")):
                    st["not_long_inside"] += 1
                    continue
                if seen_pairs is not None:  # a cut an earlier seam file already holds (same parents and position)
                    key = (tuple(stream[i]["id"] for i in p["left"]["rows"]), tuple(stream[i]["id"] for i in p["right"]["rows"]), p["pos"])
                    if key in seen_pairs:
                        st["dup_of_earlier_file"] += 1
                        continue
                st["planned_pairs"] += 1
                st[f"planned_{B.pos_class(p['pos'], p['n_words'])}"] += 1
                halves, ok = [], True
                for h in (p["left"], p["right"]):
                    x = C.window_audio(stream, h["a"], h["b"], read)
                    g = ctc.greedy(ctc.lp(x))
                    if C.cer(h["text"], g) > WIN_MAX_CER:
                        st[f"{h['side']}_cer"] += 1
                        ok = False
                    elif not B.seam_edges_ok(h, g):
                        st[f"{h['side']}_edge_word"] += 1
                        ok = False
                    halves.append((h, x, g))
                if not ok:
                    continue
                pid = f"s{si}_{k}"
                for h, x, g in halves:
                    hid = f"{pid}_{h['side'][0]}"
                    wav = f"{out}/seam{tag}/{hid}.wav"
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    f.write(json.dumps({"audio_filepath": wav, "duration": round(len(x) / 16000, 3), "text": h["text"],
                                        "lang": lang, "id": f"{lang}_seam{tag}_{hid}", "origin": "seam", "pair": pid,
                                        "side": h["side"], "pos": p["pos"], "pos_class": B.pos_class(p["pos"], p["n_words"]),
                                        "kind": p["kind"], "reading": p["reading"], "reading_part": h["reading_part"],
                                        "num_tokens": h["num_tokens"], "cut_words": h["cut_words"],
                                        "kept_cut_words": h["kept_cut_words"], "edge_fracs": h["edge_fracs"],
                                        "cut_number": True, "edge_number": True, "greedy": g,
                                        "parents": [stream[i]["id"] for i in h["rows"]],
                                        "src": [[stream[i]["id"], kk] for i, kk in h["src"]],
                                        "synthetic": any(stream[i].get("origin") == "tts" for i in h["rows"])},
                                       ensure_ascii=False) + "\n")
                    st["kept_s"] += len(x) / 16000
                st["kept_pairs"] += 1
                st[f"kept_{B.pos_class(p['pos'], p['n_words'])}"] += 1
                st[f"kept_kind_{p['kind']}"] += 1
                if min_words:
                    st[f"kept_words_{min(p['n_words'], 8)}{'+' if p['n_words'] >= 8 else ''}"] += 1
                n_pairs += 1
            if si % 50 == 0:
                f.flush()
                vol.commit()
    st = dict(st)
    st["kept_h"] = round(st.pop("kept_s", 0.0) / 3600, 3)
    st["pass_wall_s"] = round(time.time() - t_start, 1)  # decode loop only (model load and row reads come on top)
    st["skipped_partition"] = B.SEAM_SKIPS["partition"]  # pairs whose halves did not hold the reading (skipped)
    st["rows_used"], st["rows_skipped_heldout_or_rejected"] = len(rows) + len(solo), n_skip
    if lang in WAVE2:
        st["rows_aligned"], st["solo_rows"] = n_aligned, len(solo)
    json.dump(st, open(f"{out}/seam_stats{tag}.json", "w"), indent=1)
    vol.commit()
    return {"lang": lang, "seam": True, **st}


@app.local_entrypoint()
def seam_crops(langs: str = ",".join(LANGS), positions: int = 0, limit: int = 0, tag: str = "", allow_no_judge: bool = False,
               seed: int = 1, heldout: bool = False, heldout_tts: str = "", min_words: int = 0):
    """Mirrored cut-number crops per language (real rows only; windows.jsonl untouched): <lang>/seam<tag>.jsonl.
    A top-up run (tag "_b", seed 2: other streams and cut positions) adds pairs to an existing seam.jsonl; build4 reads
    both (mix_build.SEAM_FILES) and drops a half the main file already has."""
    args = [(l, 3, 8, seed, allow_no_judge, False, True, positions, limit, tag, _reader_model(l), 0.0, heldout, heldout_tts,
             min_words) for l in langs.split(",")]
    for r in windows2_lang.starmap(args):
        print(json.dumps(r))


@app.function(image=scan_image, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf")], cpu=2, memory=8192,
              timeout=3 * 3600, max_containers=12)
def granvox_index_lang(lang: str) -> dict:
    """Text-only index of NVIDIA Granary's VoxPopuli pseudo-labels (CC-BY-4.0; audio = VoxPopuli unlabelled, CC0) for
    one language, 2026-10-16: rows whose transcript has a digit -> /vol/ml/granvox/<lang>/index.jsonl (utt_id,
    audio_filepath, text, kinds, plenary day). No audio is fetched here; held-out by plenary DAY happens later."""
    from collections import Counter

    from huggingface_hub import hf_hub_download

    import ml_cands as M

    local = hf_hub_download("nvidia/Granary", f"{lang}/voxpopuli/{lang}_asr.jsonl", repo_type="dataset",
                            local_dir="/tmp/granary")
    out = f"{VOL}/ml/granvox/{lang}"
    os.makedirs(out, exist_ok=True)
    st = {"lang": lang, "rows": 0, "digit_rows": 0, "kept": 0, "unspannable": 0, "error": 0, "kinds": Counter(),
          "days": set()}
    with open(local) as f, open(f"{out}/index.jsonl", "w") as w:
        for line in f:
            st["rows"] += 1
            r = json.loads(line)
            cap = (r.get("text") or "").strip()
            if not re.search(r"\d", cap):
                continue
            st["digit_rows"] += 1
            try:
                if lang in M.LANGS:
                    if not M.all_digits_spanned(cap, lang):
                        st["unspannable"] += 1
                        continue
                    kinds = sorted({d["kind"] for d in M._find(cap, lang)})
                else:
                    kinds = []
            except Exception:  # noqa: BLE001
                st["error"] += 1
                continue
            day = (r.get("utt_id") or "")[:8]
            st["days"].add(day)
            st["kept"] += 1
            st["kinds"].update(kinds)
            w.write(json.dumps({"utt_id": r["utt_id"], "audio_filepath": r["audio_filepath"], "text": cap,
                                "kinds": kinds, "day": day}, ensure_ascii=False) + "\n")
    os.remove(local)
    st["kinds"], st["days"] = dict(st["kinds"]), len(st["days"])
    json.dump(st, open(f"{out}/index_stats.json", "w"))
    vol.commit()
    return st


@app.function(image=scan_image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=6 * 3600, max_containers=24)
def granvox_cut_lang(lang: str, year: str, ids: list, tag: str) -> dict:
    """Cut the selected Granary-VoxPopuli clips of one language-year out of the VoxPopuli unlabelled session audio
    (both tars, `<lang>_<year>.tar` and `<lang>_<year>_2.tar`), using the published segment timings. Writes 16 kHz wavs
    (training) + FLAC copies (second transcript) and returns candidate rows (src "vox", the plenary day = held-out unit)."""
    import csv, gzip, subprocess, tarfile, urllib.error, urllib.request

    import numpy as np
    import soundfile as sf

    want = set(ids)
    events = {u.rsplit("_", 1)[0] for u in want}
    seg_gz = "/tmp/unlabelled_v2.tsv.gz"
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz", seg_gz)
    segs = {}
    with gzip.open(seg_gz, "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["event_id"] in events:
                segs[f'{r["event_id"]}_{r["segment_no"]}'] = (float(r["start"]), float(r["end"]))
    out = f"{OUT2}/{lang}/wav"
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{OUT2}/{lang}/flac_{tag}", exist_ok=True)
    idx = {}
    for l in open(f"{VOL}/ml/granvox/{lang}/index.jsonl"):
        if l.startswith('{"utt_id": "' + year):
            r = json.loads(l)
            if r["utt_id"] in want:
                idx[r["utt_id"]] = r
    part = f"{OUT2}/{lang}/granvox_parts/{tag}_{year}.jsonl"
    if os.path.exists(part):  # finished on an earlier attempt
        rows = [json.loads(l) for l in open(part)]
        return {"stats": {"lang": lang, "year": year, "cut": len(rows), "resumed": True}, "rows": rows}
    rows, st = [], {"lang": lang, "year": year, "wanted": len(want), "seg_found": sum(u in segs for u in want), "cut": 0,
                    "too_short": 0, "stream_retries": 0}
    done_ev = set()
    for tname in (f"{lang}_{year}", f"{lang}_{year}_2"):
      for attempt in range(5):  # a dropped stream restarts the tar; events already cut are skipped
        try:
            resp = urllib.request.urlopen(f"https://dl.fbaipublicfiles.com/voxpopuli/audios/{tname}.tar", timeout=300)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # not every year has a second tar
                break
            st["stream_retries"] += 1
            continue
        except OSError:
            st["stream_retries"] += 1
            continue
        try:
          with tarfile.open(fileobj=resp, mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                ev = os.path.basename(m.name).rsplit(".", 1)[0]
                if ev not in events or ev in done_ev:
                    continue
                p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le",
                                    "pipe:1"], input=tar.extractfile(m).read(), capture_output=True)
                x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768
                for u in [u for u in want if u.rsplit("_", 1)[0] == ev and u in segs and u in idx]:
                    s0, e0 = segs[u]
                    clip = x[int(s0 * 16000): int(e0 * 16000)]
                    if len(clip) < 16000:
                        st["too_short"] += 1
                        continue
                    w = f"{out}/{tag}_{u}.wav"
                    sf.write(w, clip, 16000, subtype="PCM_16")
                    sf.write(f"{OUT2}/{lang}/flac_{tag}/{u}.flac", clip, 16000, format="FLAC")
                    rows.append({"audio_filepath": w, "duration": round(len(clip) / 16000, 3), "text": idx[u]["text"],
                                 "lang": lang, "src": "vox", "id": u, "origin": "new", "kinds": idx[u]["kinds"]})
                    st["cut"] += 1
                done_ev.add(ev)  # only once its clips are cut, so a dropped stream retries this event
          break
        except (tarfile.ReadError, OSError, EOFError):
            st["stream_retries"] += 1
            vol.commit()
            continue
      else:  # five failed attempts: no part file, so the next run retries this year instead of keeping a thin one
        vol.commit()
        raise RuntimeError(f"{tname}.tar: stream failed 5 times ({st['cut']} clips cut so far)")
      vol.commit()
    os.makedirs(os.path.dirname(part), exist_ok=True)
    with open(part, "w") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    vol.commit()
    return {"stats": st, "rows": rows}


@app.local_entrypoint()
def granvox_batch(langs: str = ",".join(LANGS), n_year: int = 2000, n_other: int = 1000, years_per_lang: int = 3,
                  tag: str = "granvox1", seed: int = 1, dry: bool = False):
    """Select clips from the Granary-VoxPopuli text index (year-bearing clips first, then decimals / percentages /
    large figures / ordinals), never from a held-out or validation plenary day, from each language's richest years;
    cut them (granvox_cut_lang) and write <lang>/cands_<tag>.jsonl for label2."""
    import collections
    import io
    import random

    year_re = re.compile(r"(?<![\d.,])(1[89]\d\d|20[0-4]\d)(?![\d.,%])")
    jobs, plan = [], {}
    for lang in langs.split(","):
        held_days = set()
        for name in (f"ml/p3v2/heldout_{lang}.txt",):
            for l in b"".join(vol.read_file(name)).decode().splitlines():
                m = re.match(r"(?:# group vox:)?(\d{8})", l.strip())
                if m:
                    held_days.add(m.group(1))
        try:
            for l in b"".join(vol.read_file(f"ml/p3v2/{lang}/cands_voxval.jsonl")).decode().splitlines():
                if l.strip():
                    held_days.add(json.loads(l)["id"][:8])
        except FileNotFoundError:
            pass
        rows = [json.loads(l) for l in b"".join(vol.read_file(f"ml/granvox/{lang}/index.jsonl")).decode().splitlines() if l]
        rows = [r for r in rows if r["day"] not in held_days]
        by_year = collections.Counter(r["day"][:4] for r in rows if year_re.search(r["text"]))
        years = [y for y, _ in by_year.most_common(years_per_lang)]
        rng = random.Random(f"{seed}-{lang}")
        pool = [r for r in rows if r["day"][:4] in years]
        rng.shuffle(pool)
        yr = [r for r in pool if year_re.search(r["text"])][:n_year]
        taken = {r["utt_id"] for r in yr}
        other_k = {"dec", "pct", "ord", "cur", "unit", "range", "date"}
        ot = [r for r in pool if r["utt_id"] not in taken and (set(r["kinds"]) & other_k or re.search(r"\d[\d.]{4,}", r["text"]))][:n_other]
        sel = yr + ot
        plan[lang] = {"years": years, "held_days": len(held_days), "year_clips": len(yr), "other_clips": len(ot)}
        for y in years:
            jobs.append((lang, y, [r["utt_id"] for r in sel if r["day"][:4] == y], tag))
    print(json.dumps(plan))
    if dry:
        return
    out = collections.defaultdict(list)
    for res in granvox_cut_lang.starmap(jobs):
        print(json.dumps(res["stats"]))
        out[res["stats"]["lang"]] += res["rows"]
    for lang, rs in out.items():
        with vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs).encode()),
                       f"ml/p3v2/{lang}/cands_{tag}.jsonl")
        print(lang, "rows", len(rs), "hours", round(sum(r["duration"] for r in rs) / 3600, 2))


GV_TIME_WORD = {"de": r"uhr", "fr": r"h|heures?", "it": r"ore|alle", "es": r"horas?|a las", "pt": r"horas?|às",
                "nl": r"uur"}
GV_CUR = r"€|\$|£|euros?|eur\b|dollars?|dólares|dolares|pfund|sterline|livres?|libras?|franken|francs?|kronen"
GV_SCALE = (r"milli(?:on|ard)(?:en|s|i|e)?|milion(?:i|e)|miliard(?:i|o)|millones|millón|milhões|milhão|mil milhões|"
            r"miljoen|miljard|tausend|mille|mila")
# English (2026-10-16, English parliament top-up): extra time / currency / scale / percent words, used only when
# lang == "en", so every other language keeps exactly the patterns above.
GV_TIME_WORD["en"] = r"[ap]\.?\s?m\b|o'clock\b"
GV_CLOCK_EXTRA = {"en": r"\b(?:at|until|till|by|from|around|before|after)\s+([01]?\d|2[0-3])[.:]([0-5]\d)(?![\d%]|[.,]\d)"
                        r"(?!\s*(?:per|percent|%|euros?|dollars?|pounds?|million|billion|bn|times|years?|tonnes?))"}
GV_CUR_EXTRA = {"en": r"pounds?|sterling|cents?|usd\b|gbp\b"}
GV_SCALE_EXTRA = {"en": r"billions?|trillions?|bn\b|thousand"}
GV_PCT_EXTRA = {"en": r"per\s?cent"}


def _gv_classes(text: str, lang: str) -> dict:
    """Variety strata of one caption: year values (incl. forward years to 2099), and flags for clock / currency /
    large figure / decimal / percent / ordinal / range / unit. Text-only (selection, not labelling)."""
    t = text.lower()
    gv_cur = GV_CUR + ("|" + GV_CUR_EXTRA[lang] if lang in GV_CUR_EXTRA else "")
    gv_scale = GV_SCALE + ("|" + GV_SCALE_EXTRA[lang] if lang in GV_SCALE_EXTRA else "")
    years = [int(m.group(1)) for m in re.finditer(r"(?<![\d.,])(1[89]\d\d|20\d\d)(?![\d%]|[.,]\d)", text)]
    clock = bool(re.search(r"(?<![\d.,])([01]?\d|2[0-3])[.:h]([0-5]\d)(?![\d.,])\s*(?:" + GV_TIME_WORD.get(lang, "h") + r")\b", t)
                 or re.search(r"(?:" + GV_TIME_WORD.get(lang, "h") + r")\s+([01]?\d|2[0-3])[.:]([0-5]\d)(?![\d.,])", t)
                 or re.search(r"(?<![\d.,])([01]?\d|2[0-3])\s*(?:" + GV_TIME_WORD.get(lang, "h") + r")\b", t)
                 or re.search(r"(?<![\d.,])([01]?\d|2[0-3])h(?:[0-5]\d)?\b", t)
                 or (lang in GV_CLOCK_EXTRA and re.search(GV_CLOCK_EXTRA[lang], t)))
    cur = bool(re.search(r"\d[\d.,]*\s?(?:(?:" + gv_scale + r")\s?)?(?:(?:de|di|d'|of|van|von)\s?)?(?:" + gv_cur + r")", t)
               or re.search(r"(?:€|\$|£)\s?\d", t))
    big = bool(re.search(r"(?<![\d.,])\d{1,3}(?:[.,  ]\d{3})+(?![\d,.]\d)", text) or re.search(r"\d\s?(?:" + gv_scale + r")\b", t)
               or re.search(r"(?<![\d.,])\d{5,}(?![\d.,])", text))
    dec = bool(re.search(r"(?<![\d.,])\d+[.,]\d{1,2}(?![\d.,])", text))
    pct = "%" in text or bool(re.search(r"\d\s?(?:prozent|pour cent|per cento|por ciento|por cento|procent)", t)) or bool(
        lang in GV_PCT_EXTRA and re.search(r"\d\s?(?:" + GV_PCT_EXTRA[lang] + r")", t))
    rng = bool(re.search(r"\d\s?[-–]\s?\d", text))
    return {"years": years, "clock": clock, "cur": cur, "big": big, "dec": dec and not clock, "pct": pct, "range": rng}


@app.function(image=scan_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=2 * 3600, max_containers=6)
def granvox_select_lang(lang: str, seed: int = 1, per_year: int = 60, per_forward: int = 20, per_target: int = 60,
                        per_old: int = 6, quotas: str = "clock:400,cur:400,big:500,dec:500,pct:350,range:200",
                        per_template: int = 3, per_session: int = 8, total_cap: int = 5000) -> dict:
    """Variety-first selection from the Granary-VoxPopuli index (no duplicates, spread evenly):
    year VALUES stratified (1990-2030 per_year each; forward 2031-2060 per_forward, the 2030/2035/2040/2050 targets
    per_target; 1900-1989 per_old), then per-class quotas, a cap per masked-digit sentence template and per session,
    never a held-out or validation day; all debate years."""
    import collections
    import random

    held = set()
    for l in open(f"{OUT2}/heldout_{lang}.txt"):
        m = re.match(r"(?:# group vox:)?(\d{8})", l.strip())
        if m:
            held.add(m.group(1))
    vv = f"{OUT2}/{lang}/cands_voxval.jsonl"
    if os.path.exists(vv):
        held |= {json.loads(l)["id"][:8] for l in open(vv) if l.strip()}
    rng = random.Random(f"{seed}-{lang}")
    rows = [json.loads(l) for l in open(f"{VOL}/ml/granvox/{lang}/index.jsonl")]
    rows = [r for r in rows if r["day"] not in held and 1.0 <= len(r["text"]) / 15 <= 40]
    rng.shuffle(rows)
    tmpl = lambda r: re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", r["text"].lower()))[:60]  # noqa: E731
    t_n, s_n, taken, sel = collections.Counter(), collections.Counter(), set(), []
    cls = {r["utt_id"]: _gv_classes(r["text"], lang) for r in rows}

    def take(r, why):
        u = r["utt_id"]
        if u in taken or t_n[tmpl(r)] >= per_template or s_n[u.rsplit("_", 1)[0]] >= per_session or len(sel) >= total_cap:
            return False
        taken.add(u); t_n[tmpl(r)] += 1; s_n[u.rsplit("_", 1)[0]] += 1
        sel.append(dict(r, why=why))
        return True

    want = {}
    for y in range(1900, 2061):
        want[y] = per_old if y < 1990 else per_year if y <= 2030 else per_forward
    for y in (2030, 2035, 2040, 2050):
        want[y] = max(want.get(y, 0), per_target)
    got = collections.Counter()
    by_val = collections.defaultdict(list)
    for r in rows:
        for y in set(cls[r["utt_id"]]["years"]):
            by_val[y].append(r)
    for y in sorted(want, key=lambda y: len(by_val[y])):  # rarest values first, so common ones never crowd them out
        for r in by_val[y]:
            if got[y] >= want[y]:
                break
            if take(r, f"year:{y}"):
                for yy in set(cls[r["utt_id"]]["years"]):
                    got[yy] += 1
    q = {k: int(v) for k, v in (x.split(":") for x in quotas.split(","))}
    have = collections.Counter(k for r in sel for k in q if cls[r["utt_id"]][k])
    for k in sorted(q, key=lambda k: sum(cls[r["utt_id"]][k] for r in rows)):
        for r in rows:
            if have[k] >= q[k]:
                break
            if cls[r["utt_id"]][k] and take(r, k):
                for kk in q:
                    have[kk] += bool(cls[r["utt_id"]][kk])
    yv = collections.Counter(y for r in sel for y in cls[r["utt_id"]]["years"])
    tot = sum(yv.values()) or 1
    st = {"lang": lang, "clips": len(sel), "hours_est": round(sum(min(30, len(r["text"]) / 15) for r in sel) / 3600, 1),
          "year_spans": sum(yv.values()), "year_values": len(yv),
          "y1990_2030_ge40": sum(yv[y] >= 40 for y in range(1990, 2031)),
          "y1990_2030_ge10": sum(yv[y] >= 10 for y in range(1990, 2031)),
          "forward_2031_2060_values": sum(1 for y in yv if 2031 <= y <= 2060),
          "forward_spans": sum(c for y, c in yv.items() if 2031 <= y <= 2099),
          "targets": {y: yv[y] for y in (2030, 2035, 2040, 2050)},
          "old_values": sum(1 for y in yv if 1900 <= y < 1990), "top_value_share": round(max(yv.values()) / tot, 3),
          "top5_share": round(sum(c for _, c in yv.most_common(5)) / tot, 3), "classes": dict(have),
          "sessions": len(s_n), "max_per_session": max(s_n.values()), "max_per_template": max(t_n.values()),
          "debate_years": dict(sorted(collections.Counter(r["day"][:4] for r in sel).items())),
          "index_avail": {k: sum(cls[r["utt_id"]][k] for r in rows) for k in q}}
    return {"stats": st, "ids_by_year": {y: [r["utt_id"] for r in sel if r["day"][:4] == y]
                                         for y in sorted({r["day"][:4] for r in sel})}}


@app.local_entrypoint()
def granvox_batch2(langs: str = ",".join(LANGS), tag: str = "granvox2", dry: bool = True, total_cap: int = 5000):
    """Variety-first parliament batch: granvox_select_lang per language (prints the variety table), then (dry=False)
    cuts every language-year in parallel and writes <lang>/cands_<tag>.jsonl."""
    import collections
    import io

    sel = list(granvox_select_lang.starmap([(l, 1, 60, 20, 60, 6, "clock:700,cur:700,big:500,dec:500,pct:350,range:200",
                                              3, 8, total_cap) for l in langs.split(",")]))
    for s_ in sel:
        print(json.dumps(s_["stats"], ensure_ascii=False))
    if dry:
        return
    jobs = [(s_["stats"]["lang"], y, ids, tag) for s_ in sel for y, ids in s_["ids_by_year"].items() if ids]
    out, failed = collections.defaultdict(list), []
    for job, res in zip(jobs, granvox_cut_lang.starmap(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append(f"{job[0]}_{job[1]}: {type(res).__name__}")
            continue
        print(json.dumps(res["stats"]))
        out[res["stats"]["lang"]] += res["rows"]
    print("FAILED jobs (rerun to resume):", failed)
    for lang, rs in out.items():
        with vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs).encode()),
                       f"ml/p3v2/{lang}/cands_{tag}.jsonl")
        print(lang, "rows", len(rs), "hours", round(sum(r["duration"] for r in rs) / 3600, 2))
    assert not failed, f"{len(failed)} cut jobs failed: {failed}"


fill_image = scan_image.add_local_python_source("vox_fill")


@app.function(image=fill_image, volumes={VOL: vol}, cpu=2, memory=16384, timeout=3600, max_containers=6)
def granvox_fill_select_lang(lang: str, quotas: dict, years_per_lang: int = 4, seed: int = 1) -> dict:
    """vox_fill.select next to the index: never a held-out / validation day, never a clip any earlier pass cut
    (every <lang>/cands_*.jsonl id)."""
    import glob

    import vox_fill

    held = set()
    for l in open(f"{OUT2}/heldout_{lang}.txt"):
        m = re.match(r"(?:# group vox:)?(\d{8})", l.strip())
        if m:
            held.add(m.group(1))
    vv = f"{OUT2}/{lang}/cands_voxval.jsonl"
    if os.path.exists(vv):
        held |= {json.loads(l)["id"][:8] for l in open(vv) if l.strip()}
    exclude = set()
    for p in glob.glob(f"{OUT2}/{lang}/cands_*.jsonl"):
        exclude |= {json.loads(l)["id"] for l in open(p) if l.strip()}
    rows = [json.loads(l) for l in open(f"{VOL}/ml/granvox/{lang}/index.jsonl")]
    sel, st = vox_fill.select(rows, lang, quotas, held, exclude, seed=seed, years_per_lang=years_per_lang)
    st["examples"] = {k: [r["text"][:160] for r in sel if r["why"] == k][:3] for k in quotas}
    return {"stats": st, "sel": [{"utt_id": r["utt_id"], "day": r["day"], "why": r["why"]} for r in sel]}


@app.local_entrypoint()
def granvox_fill(langs: str = "de,it,nl,pt", tag: str = "gvfill", quotas: str = "scale_m:150,scale_b:150,date:350",
                 it_clock: int = 400, years_per_lang: int = 4, pilot_n: int = 40, dry: bool = True):
    """Targeted parliament fill (vox_fill): select per language, then (dry=False) cut every language-year in
    parallel. The pilot subset (pilot_n per language, even over classes) goes to cands_<tag>p.jsonl, the rest to
    cands_<tag>.jsonl, so the two passes never share an id."""
    import collections
    import io

    import vox_fill

    q = {k: int(v) for k, v in (x.split(":") for x in quotas.split(","))}
    args = [(l, dict(q, clock=it_clock if l == "it" else 0), years_per_lang) for l in langs.split(",")]
    sel = list(granvox_fill_select_lang.starmap(args))
    for s_ in sel:
        print(json.dumps(s_["stats"], ensure_ascii=False))
    if dry:
        return
    pilot = {}
    for s_ in sel:
        pilot[s_["stats"]["lang"]] = {r["utt_id"] for r in vox_fill.pilot(s_["sel"], pilot_n)}
    jobs = []
    for s_ in sel:
        by_y = collections.defaultdict(list)
        for r in s_["sel"]:
            by_y[r["utt_id"][:4]].append(r["utt_id"])
        jobs += [(s_["stats"]["lang"], y, ids, tag) for y, ids in sorted(by_y.items())]
    out, failed = collections.defaultdict(list), []
    for job, res in zip(jobs, granvox_cut_lang.starmap(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append(f"{job[0]}_{job[1]}: {type(res).__name__}")
            continue
        print(json.dumps(res["stats"]))
        out[res["stats"]["lang"]] += res["rows"]
    print("FAILED jobs (rerun to resume):", failed)
    for lang, rs in out.items():
        for t, part in ((tag + "p", [r for r in rs if r["id"] in pilot[lang]]),
                        (tag, [r for r in rs if r["id"] not in pilot[lang]])):
            with vol.batch_upload(force=True) as b:
                b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part).encode()),
                           f"ml/p3v2/{lang}/cands_{t}.jsonl")
            print(lang, t, "rows", len(part), "hours", round(sum(r["duration"] for r in part) / 3600, 2))
    assert not failed, f"{len(failed)} cut jobs failed: {failed}"


@app.local_entrypoint()
def granvox_index(langs: str = ",".join(LANGS)):
    for r in granvox_index_lang.map(langs.split(",")):
        print(json.dumps(r))


@app.local_entrypoint()
def granvox_cut_ids(lang: str, ids_file: str, tag: str):
    """Cut a selection made elsewhere (e.g. vox_en.py for English): `ids_file` is a local JSON {debate year: [utt_id]};
    every language-year is cut in parallel (granvox_cut_lang) and the rows go to <lang>/cands_<tag>.jsonl."""
    import io

    ids_by_year = json.load(open(ids_file))
    rows = []
    for res in granvox_cut_lang.starmap([(lang, y, ids, tag) for y, ids in sorted(ids_by_year.items()) if ids]):
        print(json.dumps(res["stats"]))
        rows += res["rows"]
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()),
                   f"ml/p3v2/{lang}/cands_{tag}.jsonl")
    print(lang, "rows", len(rows), "hours", round(sum(r["duration"] for r in rows) / 3600, 2))


@app.local_entrypoint()
def scan2(langs: str = ",".join(LANGS), srcs: str = "vox,yodas", rare_hours: float = 3.0, int_hours: float = 1.5,
          max_files: int = 400, start_file: int = 0, tag: str = "", only_kinds: str = "", max_dur: float = 20.0):
    jobs = [(l, s, rare_hours, int_hours, max_files, start_file, tag, only_kinds, max_dur) for l in langs.split(",") for s in srcs.split(",")
            if not (s == "vox" and l not in VOX)]
    for r in scan2_lang.starmap(jobs):
        print(json.dumps(r, ensure_ascii=False))


@app.local_entrypoint()
def label2(langs: str = ",".join(LANGS), limit: int = 0, int_new_hours: float = 1.5, crops: bool = True,
           dry_per_kind: int = 0, srcs: str = "vox,yodas", with_d2: bool = True, name: str = "rows", ext: str = "",
           cer_ext: bool = True, split_long: bool = True, max_row: float = 20.0, only_ids: str = "",
           trim_ext: bool = True, piece_ext: bool = True, salvage: bool = True, decide_off: str = "",
           cands_off: str = "", partial: bool = True, ctc_model: str = ""):
    import ml_cands as M

    M.CAND_FIXES -= set(filter(None, cands_off.split(",")))  # the coverage check below sees the same candidates
    if ctc_model == "strong":
        import ml_decide as D
        bad = [l for l in langs.split(",") if l not in D.STRONG]
        assert not bad, f"--ctc-model strong has no reader for {bad}"
        assert name not in ("rows", "rows_" + srcs), "a strong-reader pass needs its own --name"

    for l in langs.split(","):  # review M9: a missing grammar key silently drops its span; refuse to start instead
        texts = []
        for src in srcs.split(","):
            try:
                data = b"".join(vol.read_file(f"ml/p3v2/{l}/cands_{src}.jsonl"))
            except FileNotFoundError:
                continue
            texts += [json.loads(x)["text"] for x in data.decode().splitlines() if x.strip()]
        for d2p in ((f"d2/{l}/digit.jsonl", f"d2fix/{l}/digit.jsonl") if with_d2 else ()):
            try:
                texts += [json.loads(x)["text"] for x in b"".join(vol.read_file(d2p)).decode().splitlines() if x.strip()]
            except FileNotFoundError:
                pass
        missing = M.needed(texts, l) - set(M.table())
        assert not missing, f"{l}: {len(missing)} grammar keys missing (run build_table), e.g. {sorted(missing)[:5]}"
        print(f"{l}: grammar table covers {len(texts)} captions")
    for r in label2_lang.starmap([(l, limit, int_new_hours, crops, dry_per_kind, srcs, with_d2, name, ext, cer_ext,
                                   split_long, max_row, only_ids, trim_ext, piece_ext, salvage, decide_off, cands_off,
                                   partial, ctc_model)
                                  for l in langs.split(",")]):
        print(json.dumps({k: v for k, v in r.items() if k not in ("examples", "row_samples", "noncanonical_different_words")},
                         ensure_ascii=False))
        for e in r.get("examples", [])[:10]:
            print("   ", e)


@app.local_entrypoint()
def scan(langs: str = ",".join(LANGS), hours: float = 4.5, max_files: int = 10):
    ls = langs.split(",")
    for r in scan_lang.starmap([(l, hours, max_files) for l in ls]):
        print(json.dumps(r))


@app.local_entrypoint()
def label(langs: str = ",".join(LANGS), limit: int = 0, new_hours: float = 1.5):
    ls = langs.split(",")
    for r in label_lang.starmap([(l, limit, new_hours) for l in ls]):
        print(json.dumps({k: v for k, v in r.items() if k not in ("examples", "wrong_looking")}, ensure_ascii=False))
        for e in r.get("examples", [])[:10]:
            print("   ", e)
