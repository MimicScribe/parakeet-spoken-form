"""Build the real-speech retrain mix (V2 SPEC data table) from the labelled parts on the volume.

Components (each its own manifest under /vol/mix/, with its KL tag):
  d1        English real number crops (labeller output), minus a 3% video-held-out dev split      span
  d1p       English real number-free YODAS rows (caption labels, CER <= 0.15 vs stock, <= 20 s)    anchor
  d2        non-English digit rows (stock labels) + FLEURS-train digit rows                          nonen
  d2p       non-English number-free rows (stock labels)                                              nonen
  d10n      WSOLA 1.15-1.3x copies of d1 (<= 5.5 words/s after)                                      span
  d10p      WSOLA copies of number-free rows (d1p/LSPC/ICSI), same hours as d10n                     anchor
AUG: ~25% of d1/d2 rows are replaced IN PLACE by a degraded copy (Opus 12-24 kbps, 8 kHz phone band, SLR28
point-source noise at 5-20 dB SNR, SLR28 simulated RIR). Dedupe: rows whose text shares an 8-gram with any eval text
(/vol/mix/eval_texts.txt) are dropped.
"""

import glob
import json
import os
import random
import re
import subprocess
import zlib

import numpy as np

from common import NUMBER_WORDS, WRITTEN_FORM


def _rows(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += [json.loads(line) for line in open(f) if line.strip()]  # a blank line (a writer mid-run) is no row
    return out


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _h(rows):
    return sum(r["duration"] for r in rows) / 3600


def ngrams(text, n=8):
    w = re.sub(r"[^\w' ]", " ", text.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _cer(a, b):
    a = re.sub(r"[\W_]", "", a.lower())
    b = re.sub(r"[\W_]", "", b.lower())
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


# ---- audio transforms -------------------------------------------------------------------------------------------

def _ff(inp, out, *args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", inp, *args, out], check=True)


def speed(inp, out, factor):
    _ff(inp, out, "-filter:a", f"atempo={factor:.3f}", "-ac", "1", "-ar", "16000")


class Degrader:
    def __init__(self, slr28: str, seed: int):
        self.r = random.Random(seed)
        self.noises = sorted(glob.glob(f"{slr28}/pointsource_noises/*.wav"))
        self.rirs = sorted(glob.glob(f"{slr28}/simulated_rirs/*/*/*.wav"))

    def __call__(self, inp, out):
        import soundfile as sf
        from scipy.signal import fftconvolve

        kind = self.r.choice(["opus", "phone", "noise", "rir"])
        if kind == "opus":
            tmp = out + ".opus"
            _ff(inp, tmp, "-c:a", "libopus", "-b:a", f"{self.r.choice([12, 16, 20, 24])}k", "-ar", "16000")
            _ff(tmp, out, "-ar", "16000", "-ac", "1")
            os.remove(tmp)
            return kind
        if kind == "phone":
            _ff(inp, out, "-af", "aresample=8000,highpass=f=300,lowpass=f=3400,aresample=16000", "-ac", "1")
            return kind
        x, _ = sf.read(inp, dtype="float32")
        if kind == "noise":
            n, sr = sf.read(self.r.choice(self.noises), dtype="float32")
            if n.ndim > 1:
                n = n[:, 0]
            if sr != 16000:
                import librosa

                n = librosa.resample(n, orig_sr=sr, target_sr=16000)
            n = np.tile(n, int(np.ceil(len(x) / max(len(n), 1))) + 1)[:len(x)]
            snr = self.r.uniform(5, 20)
            px, pn = np.mean(x ** 2) + 1e-9, np.mean(n ** 2) + 1e-9
            y = x + n * np.sqrt(px / (pn * 10 ** (snr / 10)))
        else:
            h, sr = sf.read(self.r.choice(self.rirs), dtype="float32")
            if h.ndim > 1:
                h = h[:, 0]
            y = fftconvolve(x, h)[:len(x)]
            y *= np.sqrt((np.mean(x ** 2) + 1e-9) / (np.mean(y ** 2) + 1e-9))
        sf.write(out, np.clip(y, -1, 1), 16000, subtype="PCM_16")
        return kind


def build(vol: str, seed: int = 1, aug_frac: float = 0.25, d10_h: float = 4.0, d1p_h: float = 10.0,
          d2p_h: float = 12.0, d1_h: float = 16.0, d2_h: float = 11.0) -> dict:
    r = random.Random(seed)
    out = f"{vol}/mix"
    os.makedirs(f"{out}/wav", exist_ok=True)
    stats = {}
    eval_ng = set()
    texts = []
    for f in glob.glob(f"{vol}/beds/e21x/refs/*.txt"):
        texts += re.split(r"(?<=[.?!])\s+", open(f).read())
    for pat in ("e22/all.jsonl", "beds/fleurs/*.jsonl", "beds/vox/*.jsonl", "beds/adv/all.jsonl"):
        for x in _rows(f"{vol}/{pat}"):
            texts.append(x["text"].split("\t")[0])
    for t in texts:
        eval_ng |= ngrams(t)
    stats["eval_texts"] = len(texts)
    stats["eval_ngrams"] = len(eval_ng)
    assert len(texts) > 5000, f"eval texts missing ({len(texts)}): build the beds first"

    def dedupe(rows, name):
        # a row overlapping eval text drops its whole video (an earnings call on YouTube is the same call)
        hit = {id(x) for x in rows if ngrams(x["text"]) & eval_ng}
        bad_v = {x.get("video") for x in rows if id(x) in hit and x.get("video")}
        keep = [x for x in rows if not (id(x) in hit or (x.get("video") and x["video"] in bad_v))]
        stats[f"{name}_dedupe_dropped"] = len(rows) - len(keep)
        stats[f"{name}_dedupe_videos"] = len(bad_v)
        return keep

    # d1 + dev split by video. Percent-pass crops (*_pct) replace the original crops of the same rows;
    # HuBERT-vetoed readings drop the crops that contain them.
    crops_all, pct_src = [], set()
    for f in sorted(glob.glob(f"{vol}/d1/crops/*.jsonl")):
        rows_f = [json.loads(line) for line in open(f)]
        if f.endswith("_pct.jsonl"):
            pct_src |= {x["src"] for x in rows_f}
        crops_all.append((f.endswith("_pct.jsonl"), rows_f))
    d1 = [x for is_pct, rows_f in crops_all for x in rows_f if is_pct or x["src"] not in pct_src]
    vetoed = {}
    for f in glob.glob(f"{vol}/d1/veto/*.json"):
        for u, rd in json.load(open(f))["vetoed"].items():
            vetoed.setdefault(u, set()).update(rd)
    has = lambda text, w: re.search(r"(?<![\w'])" + re.escape(w) + r"(?![\w'])", text, re.I) is not None  # noqa
    before = len(d1)
    d1 = [x for x in d1 if not any(has(x["text"], w) for w in vetoed.get(x["src"], ()))]
    stats["d1_vetoed_crops"] = before - len(d1)
    # crops carrying a decided MULTI-reading span are the informative ones: they fill D1 first
    amb = {}
    for rec in _rows(f"{vol}/d1/rows/*.jsonl"):
        for sp_ in rec.get("spans") or []:
            if sp_.get("decided") and sp_.get("n", 1) > 1:
                amb.setdefault(rec["utt_id"], set()).add(sp_["cands"][sp_["pick"]])
    for x in d1:
        x["amb"] = any(has(x["text"], w) for w in amb.get(x["src"], ()))
    d1 = dedupe(d1, "d1")
    dev_v = {x["video"] for x in d1 if zlib.crc32(x["video"].encode()) % 100 < 3}
    d1_dev = [x for x in d1 if x["video"] in dev_v]
    d1 = [x for x in d1 if x["video"] not in dev_v and x["duration"] <= 20 and not WRITTEN_FORM.search(x["text"])
          and "+" not in x["text"]]
    d1 = trim_edges(d1, f"{out}/wav", stats)
    r.shuffle(d1)
    d1 = sorted(d1, key=lambda x: not x["amb"])  # stable: informative crops first, random within
    d1 = _take(d1, d1_h)
    stats["d1_amb_h"] = round(sum(x["duration"] for x in d1 if x["amb"]) / 3600, 2)
    _write(f"{out}/d1_dev.jsonl", d1_dev)

    # d1p: number-free YODAS rows with a clean caption
    d1p = []
    for rec in _rows(f"{vol}/d1/rows/*.jsonl"):
        if rec["kind"] != "plain" or rec["dur"] > 20 or rec["dur"] < 1:
            continue
        cap = rec["caption"]
        if WRITTEN_FORM.search(cap) or re.search(r"[#=*_<>\[\]{}|~^@&]", cap) or _cer(cap, rec["stock"]) > 0.15:
            continue
        d1p.append(rec)
    man = {x["utt_id"]: x for x in _rows(f"{vol}/yodas/en/manifests/*.jsonl")}
    d1p = [{"audio_filepath": man[x["utt_id"]]["audio_filepath"], "duration": x["dur"], "text": x["caption"],
            "video": man[x["utt_id"]]["original_audio_id"]} for x in d1p if x["utt_id"] in man]
    d1p = [x for x in dedupe(d1p, "d1p") if x["video"] not in dev_v]
    r.shuffle(d1p)
    d1p = _take(d1p, d1p_h)

    d2, d2p = [], []
    for f in sorted(glob.glob(f"{vol}/d2/*/digit.jsonl")):
        d2 += [json.loads(line) for line in open(f)]
    for f in sorted(glob.glob(f"{vol}/d2/*/plain.jsonl")):
        d2p += [json.loads(line) for line in open(f)]
    # "⁇" = stock's unknown-piece placeholder (Greek final sigma, Maltese letters): never a target
    stats["d2_unk_dropped"] = sum("\u2047" in x["text"] for x in d2 + d2p)
    d2 = [x for x in d2 if "\u2047" not in x["text"]]
    d2p = [x for x in d2p if "\u2047" not in x["text"]]
    d2, d2p = dedupe(d2, "d2"), dedupe(d2p, "d2p")
    d2 = _take_per_lang_src(d2, d2_h, r)
    d2p = _take_per_lang_src(d2p, d2p_h, r)
    stats["d2_src_h"] = {k: round(sum(x["duration"] for x in d2 if x["src"] == k) / 3600, 2) for k in ("vox", "yodas", "fleurs")}

    # AUG in place
    deg = Degrader(f"{vol}/slr28/RIRS_NOISES", seed)
    aug_counts = {}
    for name, rows in (("d1", d1), ("d2", d2)):
        for i, x in enumerate(rows):
            if r.random() >= aug_frac:
                continue
            o = f"{out}/wav/aug_{name}_{i}.wav"
            try:
                k = deg(x["audio_filepath"], o)
            except Exception:
                continue
            aug_counts[k] = aug_counts.get(k, 0) + 1
            rows[i] = {**x, "audio_filepath": o, "aug": k}
    stats["aug"] = aug_counts

    # D10 speed copies
    def speed_copies(src, hours, tag):
        cand = list(src)
        r.shuffle(cand)
        outr, acc = [], 0.0
        for i, x in enumerate(cand):
            if acc >= hours * 3600:
                break
            f = r.uniform(1.15, 1.3)
            wps = x.get("wps") or len(x["text"].split()) / max(x["duration"], 0.1)
            if wps * f > 5.5:
                f = 5.5 / wps
            if f < 1.1:
                continue
            o = f"{out}/wav/spd_{tag}_{i}.wav"
            try:
                speed(x["audio_filepath"], o, f)
            except Exception:
                continue
            d = x["duration"] / f
            outr.append({**x, "audio_filepath": o, "duration": round(d, 3), "speed": round(f, 3)})
            acc += d
        return outr

    d10n = speed_copies(d1, d10_h, "n")
    plain_src = d1p + _rows(f"{vol}/replay/lspc_train.jsonl")[:3000] + _rows(f"{vol}/replay/icsi_train.jsonl")
    d10p = speed_copies([x for x in plain_src if x["duration"] <= 20], _h(d10n), "p")
    stats["d1p_short_of_target_h"] = round(max(0.0, d1p_h - _h(d1p)), 2)
    for name, rows in (("d1", d1), ("d1p", d1p), ("d2", d2), ("d2p", d2p), ("d10n", d10n), ("d10p", d10p)):
        _write(f"{out}/{name}.jsonl", rows)
        stats[f"{name}_h"] = round(_h(rows), 2)
        stats[f"{name}_rows"] = len(rows)
    stats["d2_h_by_lang"] = {}
    for x in d2:
        stats["d2_h_by_lang"][x["lang"]] = stats["d2_h_by_lang"].get(x["lang"], 0) + x["duration"] / 3600
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
    return stats


def _take(rows, hours):
    out, acc = [], 0.0
    for x in rows:
        if acc >= hours * 3600:
            break
        out.append(x)
        acc += x["duration"]
    return out


def _take_per_lang(rows, hours):
    langs = sorted({x["lang"] for x in rows})
    per = hours / max(len(langs), 1)
    out = []
    for lang in langs:
        out += _take([x for x in rows if x["lang"] == lang], per)
    return out


def _take_per_lang_src(rows, hours, r):
    """Equal hours per language; within a language VoxPopuli/YODAS first and FLEURS (read speech, the same domain as
    the FLEURS gates) at most 40%."""
    langs = sorted({x["lang"] for x in rows})
    per = hours / max(len(langs), 1)
    out = []
    for lang in langs:
        mine = [x for x in rows if x["lang"] == lang]
        r.shuffle(mine)
        other = [x for x in mine if x["src"] != "fleurs"]
        fl = [x for x in mine if x["src"] == "fleurs"]
        got = _take(other, per)
        room = max(0.0, per - _h(got))
        cap_fl = 0.4 * per if other else per  # a language with no other source keeps FLEURS (reported)
        got += _take(fl, min(room, cap_fl))
        out += got
    return out


def _rms_frames(x, hop=320):
    n = len(x) // hop
    return np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12) if n else np.zeros(0)


def _trim_one(args):
    x_path, out_path = args
    import soundfile as sf

    x, sr = sf.read(x_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    fr = _rms_frames(x)
    if len(fr) < 20:
        return None
    speech = np.median(fr[fr > np.percentile(fr, 30)])
    w = min(8, len(fr) // 4)  # 160 ms windows at each edge
    a = int(np.argmin(fr[:w]))
    b = len(fr) - w + int(np.argmin(fr[-w:]))
    if fr[a] > 0.3 * speech or fr[b] > 0.3 * speech:
        return "loud"
    y = x[a * 320:(b + 1) * 320]
    sf.write(out_path, y, 16000, subtype="PCM_16")
    return len(y) / 16000


def trim_edges(rows, wav_dir, stats):
    """Move each crop edge to the quietest 20 ms frame in its outer 160 ms and drop crops whose edges stay inside speech
    (the wav2vec2 alignment midpoints often landed in the previous word)."""
    from multiprocessing import Pool

    jobs = [(x["audio_filepath"], f"{wav_dir}/trim_{i}.wav") for i, x in enumerate(rows)]
    with Pool(8) as p:
        res = p.map(_trim_one, jobs, chunksize=64)
    out, loud = [], 0
    for x, (_, o), d in zip(rows, jobs, res):
        if d == "loud":
            loud += 1
        elif d:
            out.append({**x, "audio_filepath": o, "duration": round(d, 3)})
    stats["trim_dropped_loud_edge"] = loud
    stats["trim_kept"] = len(out)
    return out


NUMBER_PART = {"tts/v10pt/train_ok.jsonl": 0.9, "tts/d9sup/train_ok.jsonl": 1.0, "replay/lspc_train_numwords.jsonl": 1.0,
               "mix/d1.jsonl": 1.0, "mix/d1_up.jsonl": 1.0, "mix/d10n.jsonl": 1.0, "mix/d2.jsonl": 1.0,
               "tts/biz/train_ok.jsonl": 1.0, "mix3/d1.jsonl": 1.0, "mix3/d10n.jsonl": 1.0, "mix3/d2.jsonl": 1.0,
               "tts/biz/train_ok_np.jsonl": 1.0, "mix3b/d1.jsonl": 1.0, "mix3b/d10n.jsonl": 1.0, "tts/mp/train_ok.jsonl": 1.0,
               "tts/mp2/train_ok_x3.jsonl": 1.0, "tts/mp2/train_ok_x2.jsonl": 1.0}


# per-language parts any build4 language writes (wave 1 and beyond): spoken-form rows, windows, synthetic rows and seam
# pairs are number-bearing; d2anchor_<lang> holds an untrained language's digit rows
_LANG_PART = re.compile(r"^mix4\w*/(p3|win|tts|seam|seam_long|d2anchor|d2_phonefmt|d2phone)_([a-z]{2})\.jsonl$")


def number_share_of(path: str, rows: list | None = None) -> float:
    """Number-bearing share of a mix part's hours. Seam-pair parts are measured on their rows (a half cut before or
    after a reading holds no number word); per-language mix4 parts of ANY language count 1.0; else NUMBER_PART."""
    path = re.sub(r"^mix4\w+/", "mix4/", path)  # a dry build's dir (build4 mix_dir) counts like mix4
    m = _LANG_PART.match(path)
    if rows is not None and rows and ((m and m.group(1) in ("seam", "seam_long")) or os.path.basename(path).startswith("seam_")) \
            and all("num_tokens" in x for x in rows):
        return round(_h([x for x in rows if any(x["num_tokens"])]) / max(_h(rows), 1e-9), 3)
    if path in NUMBER_PART:
        return NUMBER_PART[path]
    return 1.0 if m else 0.0


def mix_share(vol: str, mix: list) -> dict:
    """Hours each part contributes (as write_mix takes them), the number-bearing share (V2: 35-40%) and the hours per
    language of the batches (lang_h / lang_share; a row's own "lang" field, English when it has none)."""
    tot = num = 0.0
    parts, lang_h = {}, {}
    for path, hours, _tag in mix:
        rows = _rows(f"{vol}/{path}")
        h = _h(rows)
        take = min(h, hours) if hours > 0 else h
        parts[path] = parts.get(path, 0.0) + take  # a part listed twice (oversampling) counts twice
        tot += take
        num += take * number_share_of(path, rows)
        f = take / h if h else 0.0
        for x in rows:
            lc = (x.get("lang") or "en")[:2]
            lang_h[lc] = lang_h.get(lc, 0.0) + x["duration"] / 3600 * f
    return {"parts": {k: round(v, 2) for k, v in parts.items()}, "total_h": round(tot, 2), "number_h": round(num, 2),
            "share": round(num / max(tot, 1e-9), 4),
            "lang_h": {k: round(v, 2) for k, v in sorted(lang_h.items(), key=lambda kv: -kv[1])},
            "lang_share": {k: round(v / max(tot, 1e-9), 4) for k, v in sorted(lang_h.items(), key=lambda kv: -kv[1])}}


# ---- retrain-3 mix (R3.12.2, R3.16 #2, R3.17 F1/F4) ------------------------------------------------------------------

KEEP_ALL_KINDS = {"code", "money", "percent", "scaled", "ordinal", "decade", "clock", "decimal", "ratio", "slash",
                  "phone", "unit", "range"}
TWO_K_LOW = re.compile(r"\btwo thousand (?:and )?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                       r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?: \w+)?)\b(?! (?:hundred|thousand|million|billion))", re.I)
TWO_K_0X = re.compile(r"\btwo thousand (?:and )?(?:one|two|three|four|five|six|seven|eight|nine)\b(?! (?:hundred|thousand|million|billion|\w+teen))", re.I)
TWENTY_YR = re.compile(r"\btwenty (?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                       r"twenty(?: \w+)?)\b", re.I)


def build3(vol: str, seed: int = 1, aug_frac: float = 0.25, d1_h: float = 20.0, d10_h: float = 6.0,
           d1p_h: float = 10.0, d2_h: float = 16.0, d2p_h: float = 14.0) -> dict:
    """Retrain-3 mix -> /vol/mix3/. D1 = filtered crops3 (d1/crops3_ok: D-1b/D-4/veto/label assert done, fillers
    restored) with the stratified quota; D1p = plain3 (fillers restored when d1/plain3_ok.jsonl exists); speed copies
    from CLEAN audio, weighted to years; AUG after; eval dedupe by text 8-grams AND by wav path."""
    r = random.Random(seed)
    out = f"{vol}/mix3"
    os.makedirs(f"{out}/wav", exist_ok=True)
    stats = {}
    eval_ng, eval_paths, texts = set(), set(), []
    for f in glob.glob(f"{vol}/beds/e21x/refs/*.txt"):
        texts += re.split(r"(?<=[.?!])\s+", open(f).read())
    for pat in ("e22/all.jsonl", "beds/fleurs/*.jsonl", "beds/vox/*.jsonl", "beds/adv/all.jsonl", "beds/e21_dev/*.jsonl",
                "beds/fleurs3/*.jsonl", "beds/vox/*3.jsonl"):
        for x in _rows(f"{vol}/{pat}"):
            texts.append(x["text"].split("\t")[0])
            eval_paths.add(os.path.basename(x.get("audio_filepath", "")))
    eval_paths.discard("")
    for t in texts:
        eval_ng |= ngrams(t)
    stats["eval_texts"], stats["eval_paths"] = len(texts), len(eval_paths)
    assert len(texts) > 5000, f"eval texts missing ({len(texts)}): build the beds first"

    def dedupe(rows, name):
        hit = {id(x) for x in rows if ngrams(x["text"]) & eval_ng}
        bad_v = {x.get("video") for x in rows if id(x) in hit and x.get("video")}
        keep = [x for x in rows if not (id(x) in hit or (x.get("video") and x["video"] in bad_v))]
        by_path = [x for x in keep if os.path.basename(x["audio_filepath"]) not in eval_paths]
        stats[f"{name}_dedupe_dropped"] = len(rows) - len(keep)
        stats[f"{name}_dedupe_videos"] = len(bad_v)
        stats[f"{name}_dedupe_path_dropped"] = len(keep) - len(by_path)
        return by_path

    def assert_spoken(rows, name):
        bad = [x for x in rows if WRITTEN_FORM.search(x["text"]) or re.search(r"[+#°¥⁇²³¹½¼¾]", x["text"])]
        stats[f"{name}_written_form_dropped"] = len(bad)
        return [x for x in rows if x not in bad] if bad else rows

    # D1 + 3% video dev split
    d1 = _rows(f"{vol}/d1/crops3_ok/*.jsonl")
    stats["d1_in_rows"], stats["d1_in_h"] = len(d1), round(_h(d1), 2)
    d1 = assert_spoken(dedupe(d1, "d1"), "d1")
    dev_v = {x["video"] for x in d1 if zlib.crc32(x["video"].encode()) % 100 < 3}
    _write(f"{out}/d1_dev.jsonl", [x for x in d1 if x["video"] in dev_v])
    d1 = [x for x in d1 if x["video"] not in dev_v and 0.5 <= x["duration"] <= 20]
    # stratified quota: every informative crop and every onset sub-crop; plain int/year/spelled crops fill the rest
    keep = [x for x in d1 if x["onset"] or set(x["kinds"]) & KEEP_ALL_KINDS]
    pool = [x for x in d1 if not (x["onset"] or set(x["kinds"]) & KEEP_ALL_KINDS)]
    r.shuffle(pool)
    n_2k = sum(bool(TWO_K_LOW.search(x["text"])) for x in keep + pool)
    n_20 = sum(bool(TWENTY_YR.search(x["text"])) for x in keep + pool)
    stats["year_ratio_in"] = {"two_thousand_1_29": n_2k, "twenty_10_29": n_20}
    if n_2k > n_20 / 2.5:  # R3.16 #2: cap the 2000-2009 crops to restore the ratio
        excess = int(n_2k - n_20 / 2.5)
        drop = set()
        for x in pool:
            if len(drop) >= excess:
                break
            if TWO_K_0X.search(x["text"]):
                drop.add(id(x))
        pool = [x for x in pool if id(x) not in drop]
        stats["year_cap_dropped"] = len(drop)
    room = max(0.0, d1_h - _h(keep))
    d1 = keep + _take(pool, room)
    stats["d1_keep_all_h"], stats["d1_pool_taken_h"] = round(_h(keep), 2), round(_h(d1) - _h(keep), 2)
    stats["year_ratio_out"] = {"two_thousand_1_29": sum(bool(TWO_K_LOW.search(x["text"])) for x in d1),
                               "twenty_10_29": sum(bool(TWENTY_YR.search(x["text"])) for x in d1)}
    stats["d1_onset_share"] = round(sum(x["duration"] for x in d1 if x["onset"]) / max(1e-9, _h(d1) * 3600), 3)
    r.shuffle(d1)

    # D1p: plain3 (stock CER <= 0.15), fillers restored when available
    src = f"{vol}/d1/plain3_ok.jsonl" if os.path.exists(f"{vol}/d1/plain3_ok.jsonl") else f"{vol}/d1/plain3.jsonl"
    stats["d1p_source"] = src
    d1p = [x for x in _rows(src) if 1 <= x["duration"] <= 20 and not re.search(r"[#=*_<>\[\]{}|~^@&]", x["text"])]
    d1p = [x for x in assert_spoken(dedupe(d1p, "d1p"), "d1p") if x["video"] not in dev_v]
    r.shuffle(d1p)
    d1p = _take(d1p, d1p_h)

    d2, d2p = [], []
    for f in sorted(glob.glob(f"{vol}/d2/*/digit.jsonl")):
        d2 += [json.loads(line) for line in open(f)]
    for f in sorted(glob.glob(f"{vol}/d2/*/plain.jsonl")):
        d2p += [json.loads(line) for line in open(f)]
    stats["d2_unk_dropped"] = sum("⁇" in x["text"] for x in d2 + d2p)
    d2 = [x for x in d2 if "⁇" not in x["text"]]
    d2p = [x for x in d2p if "⁇" not in x["text"]]
    d2, d2p = dedupe(d2, "d2"), dedupe(d2p, "d2p")
    d2 = _take_per_lang_src(d2, d2_h, r)
    d2p = _take_per_lang_src(d2p, d2p_h, r)
    stats["d2_src_h"] = {k: round(sum(x["duration"] for x in d2 if x["src"] == k) / 3600, 2) for k in ("vox", "yodas", "fleurs")}

    # D10 speed copies from CLEAN audio (before AUG), half of D10n from year crops
    def speed_copies(cand, hours, tag):
        outr, acc = [], 0.0
        for i, x in enumerate(cand):
            if acc >= hours * 3600:
                break
            f = r.uniform(1.15, 1.3)
            wps = x.get("wps") or len(x["text"].split()) / max(x["duration"], 0.1)
            if wps * f > 5.5:
                f = 5.5 / wps
            if f < 1.1:
                continue
            o = f"{out}/wav/spd_{tag}_{i}.wav"
            try:
                speed(x["audio_filepath"], o, f)
            except Exception:
                continue
            d = x["duration"] / f
            outr.append({**x, "audio_filepath": o, "duration": round(d, 3), "speed": round(f, 3)})
            acc += d
        return outr

    yrs = [x for x in d1 if "year" in x["kinds"] or "decade" in x["kinds"]]
    rest = [x for x in d1 if not ("year" in x["kinds"] or "decade" in x["kinds"])]
    r.shuffle(yrs)
    r.shuffle(rest)
    d10n = speed_copies(yrs, d10_h / 2, "ny")
    d10n += speed_copies(rest, d10_h - _h(d10n), "n")
    stats["d10n_year_h"] = round(sum(x["duration"] for x in d10n if "year" in x["kinds"] or "decade" in x["kinds"]) / 3600, 2)
    plain_src = d1p + _rows(f"{vol}/replay/lspc_train.jsonl")[:3000] + _rows(f"{vol}/replay/icsi_train.jsonl")
    plain_src = [x for x in plain_src if x["duration"] <= 20 and not NUMBER_WORDS.search(x["text"])]  # review #4
    r.shuffle(plain_src)
    d10p = speed_copies(plain_src, _h(d10n), "p")

    # AUG in place (after the speed copies)
    deg = Degrader(f"{vol}/slr28/RIRS_NOISES", seed)
    aug_counts = {}
    for name, rows in (("d1", d1), ("d2", d2)):
        for i, x in enumerate(rows):
            if r.random() >= aug_frac:
                continue
            o = f"{out}/wav/aug_{name}_{i}.wav"
            try:
                k = deg(x["audio_filepath"], o)
            except Exception:
                continue
            aug_counts[k] = aug_counts.get(k, 0) + 1
            rows[i] = {**x, "audio_filepath": o, "aug": k}
    stats["aug"] = aug_counts
    stats["d1p_short_of_target_h"] = round(max(0.0, d1p_h - _h(d1p)), 2)
    for name, rows in (("d1", d1), ("d1p", d1p), ("d2", d2), ("d2p", d2p), ("d10n", d10n), ("d10p", d10p)):
        _write(f"{out}/{name}.jsonl", rows)
        stats[f"{name}_h"] = round(_h(rows), 2)
        stats[f"{name}_rows"] = len(rows)
    stats["d2_h_by_lang"] = {}
    for x in d2:
        stats["d2_h_by_lang"][x["lang"]] = stats["d2_h_by_lang"].get(x["lang"], 0) + x["duration"] / 3600
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
    return stats


def _label_noise(x: dict, dec: dict | None) -> str:
    """Retrain-3 reviews: caption labels that dropped a spoken "percent"/"dollars" (stock AND v10pt hear it), number
    words with a prime apostrophe ("five' to three'"), "World War I" for a spoken "one"."""
    t = x["text"]
    if re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)'(?!s)", t, re.I):
        return "prime"
    if re.search(r"\bWorld War I\b", t):
        return "roman"
    if dec:
        for word, sym in (("percent", "%"), ("dollars?", "$")):
            n_lab = len(re.findall(rf"\b{word}\b", t, re.I))
            n_v, n_s = (len(re.findall(rf"\b{word}\b|\{sym}", dec.get(k, ""), re.I)) for k in ("v10", "stock"))
            if n_v > n_lab and n_s > n_lab:
                return "dropped_" + word.rstrip("s?")
    return ""


def patch3(vol: str, dec: dict, seed: int = 3) -> dict:
    """Post-build fixes to /vol/mix3 (retrain-3 reviews), kept as code: (1) the year-ratio cap with case-insensitive
    regexes on D1 and D10n ("two thousand 1-29" <= "twenty 10-29" / 2.5, dropping plain 2000-2009 rows only); (2) the
    label-noise drops (_label_noise) on D1 and D10n (D10n rows inherit their source crop's decode via src+text).
    `dec`: {crop wav: {"stock","v10"}} from d1/dec3. Originals are kept as <part>.pre_patch.jsonl."""
    r = random.Random(seed)
    st = {}
    crop_dec = {}
    for x in _rows(f"{vol}/d1/crops3_ok/*.jsonl"):
        if x["audio_filepath"] in dec:
            crop_dec[(x["src"], x["text"])] = dec[x["audio_filepath"]]
    for part in ("d1", "d10n"):
        rows = _rows(f"{vol}/mix3/{part}.jsonl")
        n0 = len(rows)
        noise = {}
        keep = []
        for x in rows:
            why = _label_noise(x, crop_dec.get((x.get("src"), x["text"])))
            if why:
                noise[why] = noise.get(why, 0) + 1
            else:
                keep.append(x)
        n2k = sum(bool(TWO_K_LOW.search(x["text"])) for x in keep)
        n20 = sum(bool(TWENTY_YR.search(x["text"])) for x in keep)
        excess = max(0, int(n2k - n20 / 2.5 + 0.999))
        cand = [i for i, x in enumerate(keep) if TWO_K_0X.search(x["text"])
                and not set(x.get("kinds", [])) & KEEP_ALL_KINDS]  # onset year crops count too (R3.16 #2 caps 2000-2009)
        r.shuffle(cand)
        drop = set(cand[:excess])
        keep = [x for i, x in enumerate(keep) if i not in drop]
        if not os.path.exists(f"{vol}/mix3/{part}.pre_patch.jsonl"):
            os.replace(f"{vol}/mix3/{part}.jsonl", f"{vol}/mix3/{part}.pre_patch.jsonl")
        _write(f"{vol}/mix3/{part}.jsonl", keep)
        st[part] = {"rows_in": n0, "label_noise": noise, "year_in": [n2k, n20], "year_cap_dropped": len(drop),
                    "year_cap_short": excess - len(drop),
                    "year_out": [sum(bool(TWO_K_LOW.search(x["text"])) for x in keep),
                                 sum(bool(TWENTY_YR.search(x["text"])) for x in keep)],
                    "rows_out": len(keep), "h_out": round(_h(keep), 2)}
    return st


def trim_d1(vol: str, hours: float, seed: int = 7) -> dict:
    """The F4 share trim (R3.17: cap order = D1 plain int/year first): drop `hours` of plain int/year crops (never
    onset or keep-all kinds) from mix3/d1.jsonl. Applied 2026-10-13 with hours=4.2 (share 41.5% -> 39.5%)."""
    src = f"{vol}/mix3/d1.pre_trim.jsonl"
    if not os.path.exists(src):  # idempotent: always trim from the untrimmed file
        os.replace(f"{vol}/mix3/d1.jsonl", src)
    rows = _rows(src)
    pool = [i for i, x in enumerate(rows) if not (x["onset"] or set(x["kinds"]) & KEEP_ALL_KINDS)]
    random.Random(seed).shuffle(pool)
    drop, acc = set(), 0.0
    for i in pool:
        if acc >= hours * 3600:
            break
        drop.add(i)
        acc += rows[i]["duration"]
    keep = [x for i, x in enumerate(rows) if i not in drop]
    _write(f"{vol}/mix3/d1.jsonl", keep)
    return {"rows": [len(rows), len(keep)], "dropped_h": round(acc / 3600, 2)}


def restore_unk_nonen(vol: str, min_h: float = 0.4, tol: float = 0.1) -> dict:
    """Retrain-3 dropping every stock label with "\u2047" left Greek D2 at 0.05 h (final sigma is not in
    the vocab, so stock writes \u2047). Stock emits \u2047 there itself, so a "\u2047" label on a NON-ENGLISH row
    reproduces stock (no new behaviour). For languages under `min_h` in mix3/d2 (digit) and mix3/d2p (plain), add back
    their audited \u2047 rows.

    2026-10-14 (retrain-4 plan v3): the first version restored only `digit.jsonl` into mix3/d2. Greek PLAIN rows stayed
    filtered (165 of 272 d2/el/plain rows carry \u2047, every sentence with a word-final sigma), so retrain-3 learned
    Greek only from sigma-free sentences and candidate B stopped emitting the <unk> that `UnknownPieceRepair` turns into
    \u03c2 (app FLEURS el +1.73 pp vs v10pt). Both kinds are restored now; per kind, idempotent from its `.pre_unk` file."""
    import soundfile as sf

    eval_ng, eval_paths = set(), set()  # the build3 eval dedupe (re-review: restored rows skipped it)
    for f in glob.glob(f"{vol}/beds/e21x/refs/*.txt"):
        for t in re.split(r"(?<=[.?!])\s+", open(f).read()):
            eval_ng |= ngrams(t)
    for pat in ("e22/all.jsonl", "beds/fleurs/*.jsonl", "beds/vox/*.jsonl", "beds/adv/all.jsonl", "beds/e21_dev/*.jsonl",
                "beds/fleurs3/*.jsonl", "beds/vox/*3.jsonl"):
        for x in _rows(f"{vol}/{pat}"):
            eval_ng |= ngrams(x["text"].split("\t")[0])
            eval_paths.add(os.path.basename(x.get("audio_filepath", "")))
    out = {}
    for kind, mixname in (("digit", "d2"), ("plain", "d2p")):
        base = f"{vol}/mix3/{mixname}.pre_unk.jsonl"
        cur = f"{vol}/mix3/{mixname}.jsonl"
        rows = _rows(base if os.path.exists(base) else cur)  # idempotent: rebuild from the pre-restore file
        have = {}
        for x in rows:
            have[x["lang"]] = have.get(x["lang"], 0) + x["duration"] / 3600
        allsrc = [x for f in sorted(glob.glob(f"{vol}/d2/*/{kind}.jsonl")) for x in _rows(f)]
        # A language where the filter removed a large SHARE of rows lost a kind of sentence, not just hours (Greek: every
        # sentence with a word-final sigma). Those languages get all their audited rows back, not just up to `min_h`.
        tot, unk = {}, {}
        for x in allsrc:
            tot[x["lang"]] = tot.get(x["lang"], 0) + 1
            unk[x["lang"]] = unk.get(x["lang"], 0) + ("\u2047" in x["text"])
        biased = {k for k in tot if unk[k] / tot[k] > 0.2}
        src = [x for x in allsrc if "\u2047" in x["text"]]
        src = [x for x in src if not ngrams(x["text"]) & eval_ng and os.path.basename(x["audio_filepath"]) not in eval_paths]
        random.Random(11).shuffle(src)
        seen = {x["audio_filepath"] for x in rows}
        added = {}
        for x in src:
            lang = x["lang"]
            if (lang not in biased and have.get(lang, 0) >= min_h) or x["audio_filepath"] in seen:
                continue
            try:
                i = sf.info(x["audio_filepath"])
            except Exception:
                continue
            if i.samplerate != 16000 or abs(i.frames / i.samplerate - x["duration"]) > tol:
                continue
            seen.add(x["audio_filepath"])
            rows.append(x)
            have[lang] = have.get(lang, 0) + x["duration"] / 3600
            added[lang] = added.get(lang, 0) + 1
        if added or os.path.exists(base):
            if not os.path.exists(base):
                os.replace(cur, base)
            _write(cur, rows)
        out[mixname] = {"added_rows": added, "biased_langs": sorted(biased), "h_by_lang": {k: round(v, 3) for k, v in sorted(have.items())}}
    return out


def build3b(vol: str, per_video: int = 12, seed: int = 31) -> dict:
    """R3.31 fixed mix -> /vol/mix3b/ (the warm-start experiment's treatment; the control keeps mix3 unchanged):
    - the 2000-2009 cap (build3, R3.16 #2) is undone: the crops it dropped come back from d1/crops3_ok (same filters
      as build3: not a dev video, 0.5-20 s, no written form, not in eval text or eval wavs, not already in d1);
    - year rows per video are capped at `per_video` in d1 and in d10n (one call held 44 + 53);
    - rows repeating another row's text are dropped;
    - tts/biz without the k_twenty_pause rows (maintainer gold e2 refutes its premise) -> tts/biz/train_ok_np.jsonl."""
    r = random.Random(seed)
    out = f"{vol}/mix3b"
    os.makedirs(out, exist_ok=True)
    stats = {}
    eval_ng, eval_paths = set(), set()
    texts = []
    for f in glob.glob(f"{vol}/beds/e21x/refs/*.txt"):
        texts += re.split(r"(?<=[.?!])\s+", open(f).read())
    for pat in ("e22/all.jsonl", "beds/fleurs/*.jsonl", "beds/vox/*.jsonl", "beds/adv/all.jsonl", "beds/e21_dev/*.jsonl",
                "beds/fleurs3/*.jsonl", "beds/vox/*3.jsonl"):
        for x in _rows(f"{vol}/{pat}"):
            texts.append(x["text"].split("\t")[0])
            eval_paths.add(os.path.basename(x.get("audio_filepath", "")))
    for t in texts:
        eval_ng |= ngrams(t)
    assert len(texts) > 5000, len(texts)

    d1 = _rows(f"{vol}/mix3/d1.jsonl")
    dev_v = {x["video"] for x in _rows(f"{vol}/mix3/d1_dev.jsonl")}
    have = {(x["src"], x["text"]) for x in _rows(f"{vol}/mix3/d1.pre_trim.jsonl")} | {(x["src"], x["text"]) for x in d1}
    back = [x for x in _rows(f"{vol}/d1/crops3_ok/*.jsonl")
            if TWO_K_0X.search(x["text"]) and not (x["onset"] or set(x["kinds"]) & KEEP_ALL_KINDS)
            and x["video"] not in dev_v and 0.5 <= x["duration"] <= 20 and (x["src"], x["text"]) not in have
            and not WRITTEN_FORM.search(x["text"]) and not re.search(r"[+#°¥⁇²³¹½¼¾]", x["text"])
            and not (ngrams(x["text"]) & eval_ng) and os.path.basename(x["audio_filepath"]) not in eval_paths]
    stats["cap_restored"] = len(back)
    d1 = d1 + back

    def year_row(x):
        return bool(TWO_K_LOW.search(x["text"]) or TWENTY_YR.search(x["text"]))

    def cap(rows, name):
        r.shuffle(rows)
        n, keep, seen, dup = {}, [], set(), 0
        for x in rows:
            if x["text"] in seen:
                dup += 1
                continue
            seen.add(x["text"])
            if year_row(x) and x.get("video"):
                n[x["video"]] = n.get(x["video"], 0) + 1
                if n[x["video"]] > per_video:
                    continue
            keep.append(x)
        stats[f"{name}_rows"] = [len(rows), len(keep)]
        stats[f"{name}_dup_text_dropped"] = dup
        stats[f"{name}_h"] = [round(_h(rows), 2), round(_h(keep), 2)]
        stats[f"{name}_year_ratio"] = {"two_thousand_1_29": sum(bool(TWO_K_LOW.search(x["text"])) for x in keep),
                                       "twenty_10_29": sum(bool(TWENTY_YR.search(x["text"])) for x in keep)}
        return keep

    _write(f"{out}/d1.jsonl", cap(d1, "d1"))
    _write(f"{out}/d10n.jsonl", cap(_rows(f"{vol}/mix3/d10n.jsonl"), "d10n"))
    biz = _rows(f"{vol}/tts/biz/train_ok.jsonl")
    nb = [x for x in biz if "twenty_pause" not in x["kinds"]]
    stats["biz_twenty_pause_dropped"] = len(biz) - len(nb)
    _write(f"{vol}/tts/biz/train_ok_np.jsonl", nb)
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
    return stats


def _caption_deletions(caption: str, stock: str) -> int:
    """Words stock hears that the caption leaves out (inserts + the extra length of replacements; fillers ignored)."""
    import difflib

    fill = {"uh", "um", "hmm", "mm", "uhm", "er", "ah", "oh"}
    a, b = ([x for x in re.sub(r"[^a-z' ]", " ", t.lower()).split() if x not in fill] for t in (caption, stock))
    n = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "insert":
            n += j2 - j1
        elif op == "replace" and j2 - j1 > i2 - i1:
            n += (j2 - j1) - (i2 - i1)
    return n


def build3c(vol: str, mp: str = "mp2", mp_repeat: int = 3) -> dict:
    """R3.31 v2:
    - tts/<mp>/train_ok_x<mp_repeat>.jsonl: the minimal pairs repeated so the warm-start sees each pair ~mp_repeat x 0.43;
    - mix3b/d1p_v.jsonl, mix3b/d10p_v.jsonl: plain rows whose caption leaves out a word stock hears are dropped (11.7%
      of d1p: "But that's ..." captioned "That's ..."; the e22_other deletion pattern). Rows without a stock field
      (LibriSpeech / ICSI speed copies in d10p) are kept."""
    stats = {}
    rows = _rows(f"{vol}/tts/{mp}/train_ok.jsonl")
    _write(f"{vol}/tts/{mp}/train_ok_x{mp_repeat}.jsonl", rows * mp_repeat)
    stats["mp_rows"], stats["mp_h_repeated"] = len(rows) * mp_repeat, round(_h(rows) * mp_repeat, 2)
    for name in ("d1p", "d10p"):
        src = _rows(f"{vol}/mix3/{name}.jsonl")
        keep = [x for x in src if "stock" not in x or _caption_deletions(x["text"], x["stock"]) == 0]
        _write(f"{vol}/mix3b/{name}_v.jsonl", keep)
        stats[f"{name}_v"] = {"rows": [len(src), len(keep)], "h": [round(_h(src), 2), round(_h(keep), 2)]}
    json.dump(stats, open(f"{vol}/mix3b/stats_v.json", "w"), indent=1)
    return stats


# ---- wave-1 mix (2026-10-15): spoken-form numbers in de/es/fr/it/nl/pt -----------------------------------------------

WAVE1 = ("de", "es", "fr", "it", "nl", "pt")
EXT_BY = ("ext", "ext_pick")  # the second recognizer decided where the CTC did not agree (ext_pick: 10-16)
EXT_SHARE = 0.25  # max share of a language's real row hours decided by a second recognizer
# per class (cap_ext_per_class, fill): a point under the gate's 0.25 -- the gate's classes (scale words, spoken shapes)
# differ slightly from ext_class (dry build 10-17: es dec 0.254 at a 0.25 build cap)
EXT_SHARE_CLASS = 0.24
COMMON_KINDS = {"int", "pct", "dec", "ord"}  # number kinds real speech has plenty of; the rest are "rare"
TTS_SHARE = 0.30  # synthetic speech: at most 30% of a class's (or the windows') hours
TTS_SHARE_KIND = {"clock": 0.5, "cur": 0.5, "phone": 0.5, "date": 0.5}  # shapes real speech rarely has: up to 50%
# decision 2026-10-17: English, then Spanish and French, filled to the full bar even with more synthetic speech:
# in these languages every class may be up to 50% synthetic (data_gate.SYN_SHARE_LANG mirrors it); the rest keep the caps
TTS_SHARE_LANG = {"en": 0.5, "es": 0.5, "fr": 0.5}


def tts_share(lang: str, kind: str) -> float:
    """Max synthetic share of a class's hours in a language."""
    return max(TTS_SHARE_KIND.get(kind, TTS_SHARE), TTS_SHARE_LANG.get(lang, 0.0))


TTS_FLOOR_H = 0.15  # h per class allowed when real speech has (almost) none of that class (0.05 -> 0.15, owner: fill thin cells)
TTS_SETS = (("tts1", "tts_judge"), ("tts2", "tts2_judge"), ("tts5y", "tts5y_judge"),
            ("tts6uk", "tts6uk_judge"),  # (volume dir, judge pass name); tts6uk: Ukrainian coverage (ml_tts_uk)
            ("tts7", "tts7_judge"), ("tts8", "tts8_judge"),  # tts7/tts8 (2026-10-17): es/fr thin year + count values (ml_tts_fill)
            ("tts9nl", "tts9nl_judge"),  # tts9nl (2026-10-17): Dutch thin 0-100 counts, dates, phones, prices
            ("tts9ru", "tts9ru_judge"), ("tts9ru2", "tts9ru2_judge"),  # tts9ru/2 (2026-10-17): Russian coverage (ml_tts_ru)
            ("ttsm2", "ttsm2_judge"),  # ttsm2: sv / cs / hu / fi / sk clock times + prices (ml_tts_multi)
            ("ttsm1", "ttsm1_judge"),  # ttsm1: the pl / ro synthetic set
            ("ttsL0", "ttsL0_judge"), ("ttsL1", "ttsL1_judge"),
            ("ttsL2p", "ttsL2p_judge"), ("ttsL2", "ttsL2_judge"))  # ttsL2p/2: ro / bg / fi phones read digit by
            # digit and hu dates, the reading built in code from the drawn value (not by the LLM)  # ttsL0/1: LONG numbers (big, phone, price, date,
            # year 1990-2060, big ranges) for pl ro sv cs hu fi sk bg hr; normal rates, dev by carrier template
# sets planned against the census's per-value gaps (tts5y: thin year values, normal speaking rates): the share cap
# applies, the "real speech has enough of this class" cut-off does not (synthetic years and counts
# are fine within the cap)
# tts6uk (2026-10-17): Ukrainian has almost no real years / prices / phones / dates; its years and counts share the
# "int" cap with the real Common Voice counts, so the enough-real-speech cut-off must not zero them
# tts7 / tts8 (2026-10-17, ml_tts_fill): exact thin year values and count values planned against the es / fr census
# tts9nl (2026-10-17, ml_tts_cloud --class-n): Dutch counts planned against the thin 0-100 values share the "int" cap
# tts9ru / tts9ru2 (2026-10-17, ml_tts_ru): Russian has almost no real clock times / prices / phones / dates and few
# year values above the bar; like tts6uk its years and counts share the "int" cap, so the cut-off must not zero them
# ttsm2 (2026-10-17, ml_tts_multi): clock times and prices for the moderate-tier languages, each class planned at or
# below its real span count so the per-class share cap holds
TTS_TARGETED = {"tts5y", "tts6uk", "tts7", "tts8", "tts9nl", "tts9ru", "tts9ru2", "ttsm2", "ttsm1", "ttsL0", "ttsL1", "ttsL2p", "ttsL2"}
TTS_AUG_FRAC = 0.6  # share of chosen synthetic rows given a degraded copy instead of the clean audio
TTS_ENOUGH_H = 0.25  # h of real speech in a class above which no synthetic rows of that class are used
WF_LABEL = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°²³¹½¼¾⅓⅔&‰]")  # ft_train.WF_CHARS (+ & ‰): never in a spoken-form label


def _reading_ranges(row: dict) -> list[tuple[int, int, str]]:
    """[(i0, i1, reading)] whitespace-token ranges of each number reading in a labelled row (from num_chars)."""
    out, toks, pos, spans = [], row["text"].split(), 0, []
    for tok in toks:
        a = row["text"].index(tok, pos)
        spans.append((a, a + len(tok)))
        pos = a + len(tok)
    for a, b in row.get("num_chars") or []:
        idx = [k for k, (x, y) in enumerate(spans) if x < b and a < y]
        if idx:
            out.append((idx[0], idx[-1] + 1, " ".join(toks[idx[0]:idx[-1] + 1])))
    return out


def consistency_pairs(rows: list[dict], wins: list[dict], lang: str, rng, cap: int = 2000) -> list[dict]:
    """Arm-c pair list (ft_train.load_pairs format): the SAME reading of the same parent row inside two training examples
    (a row and a window, or two windows at different phases). Readings must sit wholly inside each window (src records
    every window token's parent word); readings that occur twice in one text are skipped (ambiguous). Edge-vs-centre
    pairs first."""
    by_id = {r["id"]: r for r in rows}
    occ = {}  # (parent, i0) -> [(example, j0, j1, edge)]
    for r in rows:
        rd = _reading_ranges(r)
        texts = [t for *_, t in rd]
        for i0, i1, t in rd:
            if texts.count(t) == 1:
                occ.setdefault((r["id"], i0), []).append((r, i0, i1, False, t))
    for w in wins:
        src = [tuple(x) for x in w.get("src", [])]
        toks = w["text"].split()
        for (pid, i0), lst in list(occ.items()):
            p = by_id.get(pid)
            if p is None:
                continue
            i1, t = lst[0][2], lst[0][4]
            want = [(pid, k) for k in range(i0, i1)]
            for j in range(len(src) - len(want) + 1):
                if src[j:j + len(want)] == want and " ".join(toks[j:j + len(want)]) == t:
                    edge = j < 3 or len(src) - (j + len(want)) < 3
                    lst.append((w, j, j + len(want), edge, t))
                    break
    pairs = []
    for (pid, _), lst in occ.items():
        if len(lst) < 2:
            continue
        ranked = sorted(((a, b) for x, a in enumerate(lst) for b in lst[x + 1:]),
                        key=lambda ab: -(ab[0][3] != ab[1][3]))  # edge vs centre first
        for a, b in ranked[:3]:
            ent = lambda e: {"audio_filepath": e["audio_filepath"], "duration": e["duration"], "text": e["text"]}  # noqa: E731
            pairs.append({"a": ent(a[0]), "b": ent(b[0]), "a_span": [a[1], a[2]], "b_span": [b[1], b[2]],
                          "reading": a[4], "parent": pid, "lang": lang})
    rng.shuffle(pairs)
    return pairs[:cap]




TOP_YEAR_SHARE = 0.03  # max share of one year value among a language's year spans in the mix (2026-10-17)
_YEAR = re.compile(r"^(1[89]\d\d|20\d\d)\.?$")


def _year_of(sp):
    m = _YEAR.match(str(sp.get("text", "")).strip())
    return int(m.group(1)) if m and sp.get("kind") in ("int", "year", "ord") else None


def _cap_top_years(rows: list, share: float, rng) -> tuple:
    """Leave out rows whose every number span is an over-represented year value until no value exceeds `share` of the
    year spans (rows with any other number stay: they carry other classes)."""
    from collections import Counter
    cnt = Counter(y for x in rows for sp in x.get("spans", []) if (y := _year_of(sp)))
    total = sum(cnt.values())
    if not total:
        return rows, {}
    # one fixed per-value limit T, solved so the capped values sit at `share` of the year spans left: no cascade onto
    # values that were under the share to begin with
    over = [y for y, n in cnt.items() if n > share * total]
    t = share * total
    for _ in range(5):
        t = share * (total - sum(max(0, cnt[y] - t) for y in over))
    drop, out = set(), {}
    for y in over:
        n = cnt[y]
        only = [(x["id"], len(x["spans"])) for x in rows if x["id"] not in drop and x.get("spans")
                and all(_year_of(sp) == y for sp in x["spans"])]
        rng.shuffle(only)
        k = 0
        while n > t and k < len(only):
            i, per = only[k]
            drop.add(i); n -= per; k += 1
        out[y] = {"before": cnt[y], "after": n, "rows_out": k}
    return [x for x in rows if x["id"] not in drop], out

# balance caps (the data gate's variety rules, applied in the build): <= TEMPLATE_CAP real number rows per digit-masked
# caption template, <= GROUP_CAP per video / session (English YouTube: 30). Rows over a cap leave the MIX (they stay on
# disk); within a template / group the rows with a rare number class are kept first. Mirrors data_gate.group_id /
# group_cap / the template key.
TEMPLATE_CAP = 3
GROUP_CAP = 13
GROUP_CAP_LANG = {"en": {"yt": 30}}


def group_id(r: dict) -> str:
    """Video (YODAS), parliament session, FLEURS sentence or Common Voice sentence: the unit of speaker / topic variety."""
    i = str(r.get("id", ""))
    m = re.match(r"^[a-z]{2}\d{3}_\d+_(.{11})_", i)
    if m:
        return "yt:" + m.group(1)
    if "PLENARY" in i or re.match(r"^\d{8}-", i):
        return "ep:" + i.split("_")[0]
    m = re.match(r"^fleurs\w*?_(\d+)_", i)
    if m:
        return "fl:" + m.group(1)
    if i.startswith("cv22_"):
        t = str(r.get("caption") or r.get("text") or "")
        return "cv:" + " ".join(re.findall(r"[^\W_]+", t.lower().replace("\u2019", "'")))
    return "id:" + i


def group_cap(lang: str, g: str) -> int:
    return GROUP_CAP_LANG.get(lang, {}).get(g.split(":", 1)[0], GROUP_CAP)


TEMPLATE_MIN_WORDS = 3  # a masked caption with fewer letter words ("#.", "# euro.") is a short answer, not a template


def template_key(r: dict) -> str:
    """data_gate's digit-masked caption template; "" (not capped) when it has < TEMPLATE_MIN_WORDS letter words."""
    t = re.sub(r"\d+", "#", str(r.get("caption", "")).lower()).strip()
    return t if len(re.findall(r"[^\W\d_]+", t)) >= TEMPLATE_MIN_WORDS else ""


def _synthetic(r: dict) -> bool:
    return r.get("src") == "tts" or str(r.get("origin", "")).startswith("tts") or bool(r.get("synthetic"))


def _rare(r: dict) -> bool:
    return any(sp.get("kind") not in COMMON_KINDS for sp in r.get("spans") or [])


def _span_values(r: dict) -> list:
    """("year", v) / ("int", v) keys of a row's integer spans (caption digits), for the thin-value protection."""
    out = []
    for sp in r.get("spans") or []:
        c = ext_class(sp)
        d = re.sub(r"\D", "", str(sp.get("text", "")))
        if c == "year" and d:
            out.append(("year", int(d)))
        elif c == "int_small" and d and int(d) <= 100:
            out.append(("int", int(d)))
    return out


def _thin_floor(v) -> int:
    kind, x = v
    if kind == "year":
        return 60 if 1990 <= x <= 2060 else 15  # data_gate: every 1990-2030 >= 40, every 2031-2060 >= 10
    return 15  # every 0-100 >= 10


FILL_FLOOR = {"year_main": 42, "year_fwd": 11, "int": 11}  # data_gate's 40 / 10 / 10 floors + a small margin


def _fill_floor(v) -> int:
    kind, x = v
    if kind == "year":
        return FILL_FLOOR["year_main"] if 1990 <= x <= 2030 else FILL_FLOOR["year_fwd"] if 2031 <= x <= 2060 else 0
    return FILL_FLOOR["int"]


def _ext_ok(rows: list):
    """A running per-class ext-share check for fill_thin: adding a row keeps every class's ext spans <= EXT_SHARE."""
    from collections import Counter
    n, e = Counter(), Counter()
    for x in rows:
        for sp in x.get("spans") or []:
            n[ext_class(sp)] += 1
            e[ext_class(sp)] += sp.get("by") in EXT_BY

    def ok(x) -> bool:
        dn, de = Counter(), Counter()
        for sp in x.get("spans") or []:
            dn[ext_class(sp)] += 1
            de[ext_class(sp)] += sp.get("by") in EXT_BY
        if any(de[c] and e[c] + de[c] > EXT_SHARE_CLASS * (n[c] + dn[c]) for c in dn):
            return False
        n.update(dn); e.update(de)
        return True
    return ok


def fill_thin(rows: list, pool: list, rng, top_share: float = TOP_YEAR_SHARE, ok=None) -> tuple:
    """Add pool rows (judged rows a cap took out) that hold a year (1990-2060) or 0-100 count still under its gate floor,
    until it is met; a row is added only if no year value then exceeds `top_share` of the year spans."""
    from collections import Counter
    cnt = Counter(v for x in rows for v in _span_values(x))
    years = sum(c for (k, _), c in cnt.items() if k == "year")
    pool = list(pool)
    rng.shuffle(pool)
    added, out = 0, list(rows)
    for x in pool:
        vs = _span_values(x)
        if not any(cnt[v] < _fill_floor(v) for v in vs):
            continue
        new = Counter(vs)
        ny = years + sum(c for (k, _), c in new.items() if k == "year")
        if any(cnt[v] + c > top_share * ny for v, c in new.items() if v[0] == "year"):
            continue  # one of its years would pass the 3% top-year share
        if ok is not None and not ok(x):
            continue
        out.append(x)
        cnt.update(new)
        years = ny
        added += 1
    short = sorted(f"{k}{x}" for (k, x), c in cnt.items() if c < _fill_floor((k, x)) - 2 and k == "year")
    return out, {"added": added, "still_short_years": short[:20]}


def balance_caps(rows: list, lang: str, rng, template_cap: int = TEMPLATE_CAP, use_groups: bool = True) -> tuple:
    """(kept rows in their input order, stats). Only real rows with number spans are counted and capped (synthetic and
    number-free rows pass), as the data gate counts them. Template cap first (an empty caption is no template), then
    the group cap; inside each, rare-class rows are kept before the rest (random order otherwise), and a rare-class row
    over a cap stays (counted in rare_over_cap_kept): the caps never cost a thin cell."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    # thin VALUES are protected like rare classes (dry build 10-17: the session cap cost es its "every 1990-2030 >= 40"):
    # a row holding a year value or a 0-100 count that the language has few of is never capped out
    from collections import Counter
    vc = Counter(v for x in rows for v in _span_values(x))
    thin = lambda r: any(vc[v] <= _thin_floor(v) for v in _span_values(r))  # noqa: E731
    order.sort(key=lambda i: not _rare(rows[i]))
    counted = lambda r: r.get("spans") and not _synthetic(r)  # noqa: E731
    out_t, out_g, nt, ng = set(), set(), {}, {}
    rare_kept = 0
    for i in order:
        r = rows[i]
        if not counted(r):
            continue
        rare = _rare(r) or thin(r)  # a rare-class / thin-value row counts toward the caps but is never removed
        t = template_key(r)
        if t and template_cap:
            nt[t] = nt.get(t, 0) + 1
            if nt[t] > template_cap:
                if not rare:
                    out_t.add(i)
                    continue
                rare_kept += 1
        if use_groups:
            g = group_id(r)
            if not g.startswith("id:"):
                ng[g] = ng.get(g, 0) + 1
                if ng[g] > group_cap(lang, g):
                    if rare:
                        rare_kept += 1
                    else:
                        out_g.add(i)
    out = out_t | out_g
    kept = [r for i, r in enumerate(rows) if i not in out]
    st = {"rows_in": len(rows), "template_out": len(out_t), "group_out": len(out_g), "rare_over_cap_kept": rare_kept,
          "h_out": round(_h([rows[i] for i in out]), 2)}
    return kept, st


# English real span parts of the v4 mix (year-capped manifests): balance caps + the English share trim apply to these
EN_REAL_PARTS = ("mix3b/d1_ycap.jsonl", "mix3b/d10n_ycap.jsonl", "d1/en4_ok/yodas_en4_cap30_ycap.jsonl",
                 "d1/en4_ok/vox_en_crops_ok_ycap.jsonl", "d1/en4_ok/yodas_enh_train_ycap.jsonl")
EN_TOPUP = "d1/en4_ok/yodas_en4_cap30_ycap.jsonl"  # the YouTube top-up: the only part the share trim takes rows from
EN_RARE_KINDS = KEEP_ALL_KINDS - {"percent"}  # an English crop with one of these kinds is never trimmed (phone, clock, ...)
_EN_NUM = re.compile(r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                     r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                     r"hundred|thousand|million|billion|and a half|point)(?:[ -](?:zero|oh|one|two|three|four|five|six|"
                     r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                     r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|point))*\b", re.I)


def en_bcap_path(path: str) -> str:
    return path[:-6] + "_bcap.jsonl"


def _en_group(r: dict) -> str:
    v = str(r.get("video") or "")
    if re.match(r"^\d{8}-", v):
        return "ep:" + v
    return "yt:" + v if v else "id:" + str(r.get("src") or r.get("audio_filepath"))


def _en_template(r: dict) -> str:
    """English crops carry the spoken label only: the template is the label with every number-word run masked (a speed
    copy in d10n shares its source crop's template, as data_gate counts it)."""
    t = " ".join(re.sub(r"[^a-z# ]+", " ", _EN_NUM.sub("#", str(r.get("text", "")).lower())).split())
    return t if len([w for w in t.split() if w != "#"]) >= TEMPLATE_MIN_WORDS else ""


_EN_DIG = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
# shapes the crop kinds do not mark but the data gate counts as their own (rare) classes: a number read digit by digit
# ("four five six", "nine two eight two": code / phone), and a date (month name next to a number or ordinal)
_EN_DIGIT_RUN = re.compile(rf"\b{_EN_DIG}(?:[ ,-]+{_EN_DIG}){{2,}}\b", re.I)
_EN_DATE = re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
                      r"(?:\W+\w+){0,2}?\W+(?:the\s+)?(?:\w+(?:teen|ty|th|st|nd|rd)|one|two|three|four|five|six|seven|eight|"
                      r"nine|ten|eleven|twelve)\b|\b(?:\w+(?:th|st|nd|rd)|first|second|third)\s+of\s+(?:january|february|march|"
                      r"april|may|june|july|august|september|october|november|december)\b", re.I)


def _en_rare(r: dict) -> bool:
    if set(r.get("kinds") or []) & EN_RARE_KINDS:
        return True
    t = str(r.get("text", ""))
    return bool(_EN_DIGIT_RUN.search(t) or _EN_DATE.search(t))


# YouTube crops the share trim may take, in trim order: the top-up first, then the speed copies, B's own crops last
EN_TRIM_PARTS = ("d1/en4_ok/yodas_en4_cap30_ycap.jsonl", "mix3b/d10n_ycap.jsonl", "mix3b/d1_ycap.jsonl")


# value floors of the English share trim (data_gate's English bar: every 1990-2030 year >= 40 spans, each 0-100 >= 10,
# year synthetic share <= 50%): a crop leaves only while every value it holds stays above its floor. Real-span counts
# only (synthetic spans come on top), so the floors sit a little above the bar.
TOP_YEAR_TRIM = 0.9  # only years with >= 90% of the top year's count may lose crops in the share trim
EN_YEAR_REAL_FLOOR = 5500  # real English year spans kept: the English synthetic sets hold 5,186 year spans (gate 10-17)


def _en_floor(v) -> int:
    if 1990 <= v <= 2030:
        return 45
    if 1800 <= v <= 2099 or 0 <= v <= 100:
        return 12
    return 3 if v < 1000 else 1


def _en_values(text: str) -> list:
    import score_retrain as SR  # the four-outcome scorer's number reader (English words -> values)

    out = []
    for n in SR.extract_numbers(text, "en", keep_weak=False):
        v = min(n.vals)
        if v == int(v):
            out.append(int(v))
    return out


def _en_value_trim(rows: dict, out: set, flat: list, tparts: list, done, year_real_floor: int, chunk: int = 400):
    """Value-aware share trim: common-class crops of the YouTube parts leave, the most over-represented values first
    (and, inside a value, the videos with the most crops first; the top-up before the older mix3b crops), while every
    value they hold stays above _en_floor and the real year spans stay >= year_real_floor. Stops when done(cut)."""
    from collections import Counter

    vals = {}
    cnt = Counter()
    for p, i in flat:
        if (p, i) in out:
            continue
        v = vals[(p, i)] = _en_values(rows[p][i].get("text", ""))
        cnt.update(v)
    years = sum(c for v, c in cnt.items() if 1800 <= v <= 2099)
    per_video = Counter(_en_group(rows[p][i]) for p, i in flat if (p, i) not in out)
    rank = {p: k for k, p in enumerate(tparts)}
    # crops holding a year stay whole: the English year spread was tuned by the per-value year cap (3%) and trimming
    # any year crop moves the capped values' share up (gate check 10-17); kinds "year" / "decade" or a parsed 1800-2099
    no_year = lambda pi: not ({"year", "decade"} & set(rows[pi[0]][pi[1]].get("kinds") or [])) and not any(  # noqa: E731
        1800 <= v <= 2099 for v in vals[pi])
    cand = [(p, i) for p, i in flat if p in rank and (p, i) not in out and not _en_rare(rows[p][i]) and vals.get((p, i))
            and no_year((p, i))]
    cut, st = set(), {"rounds": 0, "skipped_floor": 0}
    while cand and not done(cut):
        st["rounds"] += 1
        margin = lambda pi: min(cnt[v] - _en_floor(v) for v in vals[pi])  # noqa: E731
        cand.sort(key=lambda pi: (-margin(pi), -per_video[_en_group(rows[pi[0]][pi[1]])], rank[pi[0]]))
        took, rest = 0, []
        for pi in cand:
            if took >= chunk:
                rest.append(pi)
                continue
            vs = vals[pi]
            ny = sum(1800 <= v <= 2099 for v in vs)
            need = Counter(vs)
            # a crop holding a year leaves only when that year is (near) the most frequent one: trimming any other year
            # raises the top year's share of the year spans (data_gate: top year <= 3%)
            top = max((c for v, c in cnt.items() if 1800 <= v <= 2099), default=0)
            abundant = all(cnt[v] >= TOP_YEAR_TRIM * top for v in vs if 1800 <= v <= 2099)
            if abundant and all(cnt[v] - k >= _en_floor(v) for v, k in need.items()) and years - ny >= year_real_floor:
                cut.add(pi); took += 1
                cnt.subtract(need); years -= ny
                per_video[_en_group(rows[pi[0]][pi[1]])] -= 1
            else:
                st["skipped_floor"] += 1
        if not took:
            break
        cand = rest
    st["years_real_left"] = years
    st["stopped"] = "target" if done(cut) else "floors"
    return cut, st


def en_balance(vol: str, mix: list, seed: int = 43, band: tuple = (0.35, 0.40), target: float = 0.39,
               non_en: tuple = (), write: bool = False, trim_parts: tuple = EN_TRIM_PARTS, fixed_n: int | None = None,
               out_dir: str = "", value_aware: bool = True, year_real_floor: int = EN_YEAR_REAL_FLOOR) -> dict:
    """English balance for the v4 mix (2026-10-17), on the real span parts (EN_REAL_PARTS) as `mix` lists them:
    1. the data gate's caps across those parts jointly: <= TEMPLATE_CAP crops per number-masked label, <= GROUP_CAP_LANG
       rows per YouTube video (30) and GROUP_CAP per parliament session (13); rare-class crops kept first;
    2. if the English number share of `mix` (parts not starting with `non_en`) is still above band[1]: the YouTube
       parts (trim_parts: the top-up first, then the mix3b crops and their speed copies) keep at most N COMMON-class
       crops per video, N the largest value that brings the share to <= target; a crop with a rare class
       (EN_RARE_KINDS: phone, clock, money, unit, code, ...) is never trimmed and does not count toward N; parliament
       crops, caption-scan hits, synthetic sets, windows and seam pairs stay whole.
    The number-free side (LibriSpeech hours etc.) is whatever `mix` lists. write=True writes <part>_bcap.jsonl for
    each real part (rows over a cap leave the mix; the source manifests are untouched). Returns the stats and the
    English share before / after caps / after the trim (share_after above band[1] = the trim alone cannot reach it)."""
    rng = random.Random(seed)
    dry = lambda p: re.sub(r"^mix4\w+/", "mix4/", p)  # noqa: E731  (a dry build's dir is non-English like mix4)
    en_mix = [m for m in mix if not dry(m[0]).startswith(non_en)] if non_en else list(mix)
    orig = {en_bcap_path(p): p for p in EN_REAL_PARTS}  # a mix listing the _bcap copies is judged on their sources
    en_mix = [(orig.get(m[0], m[0]),) + tuple(m[1:]) for m in en_mix]
    real = [p for p in EN_REAL_PARTS if any(m[0] == p for m in en_mix)]
    rows = {p: _rows(f"{vol}/{p}") for p in real}
    cache = {}
    flat = [(p, i) for p in real for i in range(len(rows[p]))]
    rng.shuffle(flat)
    flat.sort(key=lambda pi: not _en_rare(rows[pi[0]][pi[1]]))
    nt, ng, out, why = {}, {}, set(), {"template": 0, "group": 0, "rare_over_cap_kept": 0}
    for p, i in flat:
        r = rows[p][i]
        rare = _en_rare(r)  # a rare-class crop counts toward the caps but is never removed by them
        t = _en_template(r)
        if t:
            nt[t] = nt.get(t, 0) + 1
            if nt[t] > TEMPLATE_CAP:
                if rare:
                    why["rare_over_cap_kept"] += 1
                else:
                    out.add((p, i)); why["template"] += 1
                    continue
        g = _en_group(r)
        if not g.startswith("id:"):
            ng[g] = ng.get(g, 0) + 1
            if ng[g] > group_cap("en", g):
                if rare:
                    why["rare_over_cap_kept"] += 1
                else:
                    out.add((p, i)); why["group"] += 1

    def share(cut: set) -> dict:
        num = tot = 0.0
        for path, hours, _t in en_mix:
            if path in rows:
                rs = [x for i, x in enumerate(rows[path]) if (path, i) not in cut]
            else:
                rs = cache.setdefault(path, _rows(f"{vol}/{path}"))
            h = _h(rs)
            take = min(h, hours) if hours > 0 else h
            tot += take
            num += take * number_share_of(path, rs)
        return {"share": round(num / max(tot, 1e-9), 4), "number_h": round(num, 2), "total_h": round(tot, 2)}

    before = share(set())
    capped = share(out)
    trim, n_per_video = set(), None
    tparts = [p for p in trim_parts if p in rows]
    if (fixed_n is not None or capped["share"] > band[1]) and tparts and value_aware and fixed_n is None:
        trim, vst = _en_value_trim(rows, out, flat, tparts, lambda c: share(out | c)["share"] <= target,
                                   year_real_floor=year_real_floor)
    elif (fixed_n is not None or capped["share"] > band[1]) and tparts:
        rank = {p: k for k, p in enumerate(tparts)}  # the top-up's crops are trimmed before the older mix3b crops
        common = sorted(((p, i) for p, i in flat if p in rank and (p, i) not in out and not _en_rare(rows[p][i])),
                        key=lambda pi: -rank[pi[0]])  # kept first = the older parts' crops

        def cut_at(n: int) -> set:
            seen, c = {}, set()
            for p, i in common:
                g = _en_group(rows[p][i])
                seen[g] = seen.get(g, 0) + 1
                if seen[g] > n:
                    c.add((p, i))
            return c

        lo, hi = (fixed_n, fixed_n) if fixed_n is not None else (0, max(GROUP_CAP_LANG["en"]["yt"], 1))
        while lo < hi:  # the largest N with share <= target (the share falls as N falls)
            mid = (lo + hi + 1) // 2
            if share(out | cut_at(mid))["share"] <= target:
                lo = mid
            else:
                hi = mid - 1
        n_per_video = lo
        trim = cut_at(lo)
    after = share(out | trim)
    gone = out | trim
    st = {"share_before": before, "share_after_caps": capped, "share_after": after, "band": list(band),
          "in_band": band[0] <= after["share"] <= band[1], "caps_out": why,
          "common_per_video": n_per_video, "value_trim": vst if value_aware and fixed_n is None and trim else None,
          "trim_rows": len(trim), "trim_h": round(_h([rows[p][i] for p, i in trim]), 2),
          "parts": {p: {"rows": [len(rows[p]), len(rows[p]) - sum(1 for q, _ in gone if q == p)],
                        "h": [round(_h(rows[p]), 2), round(_h([x for i, x in enumerate(rows[p]) if (p, i) not in gone]), 2)]}
                    for p in real}}
    assert not any(_en_rare(rows[p][i]) for p, i in trim), "the share trim must never take a rare-class crop"
    if write:
        for p in real:
            _write(f"{out_dir or vol}/{en_bcap_path(p)}", [x for i, x in enumerate(rows[p]) if (p, i) not in gone])
    return st


def _audio_ok(x) -> bool:
    """The row's audio still has the row's duration (side-by-side passes once shared wav/<id>.wav and overwrote it)."""
    import soundfile as sf
    try:
        inf = sf.info(x["audio_filepath"])
    except Exception:  # noqa: BLE001
        return False
    return abs(inf.frames / inf.samplerate - float(x.get("duration", 0))) <= 0.05


def _one_per_id(paths: list) -> tuple:
    """Rows of every labelling pass, one per id. Passes that re-cut one source (recovery rounds, the variety pass) can
    share ids: a copy whose audio no longer matches its duration is dropped, then a regional-reading pass
    (rows_*_regional) is preferred where its label differs, else the first pass in name order."""
    by_id, order, n_dup, bad = {}, [], 0, 0
    for p in sorted(paths, key=lambda p: (not os.path.basename(p).endswith("_regional.jsonl"), p)):
        for x in _rows(p):
            i = x["id"]
            if i not in by_id:
                by_id[i] = [x]; order.append(i)
            else:
                by_id[i].append(x); n_dup += 1
    out = []
    for i in order:
        cands = by_id[i]
        if len(cands) > 1:
            ok = [x for x in cands if _audio_ok(x)]
            bad += len(cands) - len(ok)
            if not ok:
                continue
            cands = ok
        out.append(cands[0])
    return out, {"dup_copies": n_dup, "dup_audio_mismatch": bad}

def ext_class(sp: dict) -> str:
    """data_gate's span class (span_class): an int is a year (4-digit 1800-2099 caption token), int_small (< 1000) or
    int_big; every other kind is its own class."""
    k, t = sp.get("kind", "?"), str(sp.get("text", "")).strip()
    if k != "int":
        return k
    if re.fullmatch(r"\d{4}", t) and 1800 <= int(t) <= 2099:
        return "year"
    d = re.sub(r"[\s.,'\u00a0\u202f\u2019]", "", t)
    if d.isdigit():
        return "int_small" if int(d) < 1000 else "int_big"
    return "int_other"


def cap_ext_per_class(rows: list, share: float, rng, min_spans: int = 20) -> tuple:
    """Second-recognizer spans (by in EXT_BY) <= `share` of each class's spans, as data_gate checks it (per class, not
    per language): rows holding an ext span of a class over the cap leave the mix (random order) until every class
    with >= min_spans spans is at or under it. Returns (rows, {class: [ext share before, after], ...})."""
    from collections import Counter

    def tally(rs):
        n, e = Counter(), Counter()
        for x in rs:
            for sp in x.get("spans") or []:
                c = ext_class(sp)
                n[c] += 1
                e[c] += sp.get("by") in EXT_BY
        return n, e

    n0, e0 = tally(rows)
    out = set()
    for _ in range(6):
        n, e = tally([x for i, x in enumerate(rows) if i not in out])
        over = [c for c in n if n[c] >= min_spans and e[c] > share * n[c]]
        if not over:
            break
        for c in over:
            idx = [i for i, x in enumerate(rows) if i not in out
                   and any(ext_class(sp) == c and sp.get("by") in EXT_BY for sp in x.get("spans") or [])]
            rng.shuffle(idx)
            ec, nc = e[c], n[c]
            for i in idx:
                if ec <= share * nc:
                    break
                k = [sp for sp in rows[i].get("spans") or [] if ext_class(sp) == c]
                ec -= sum(sp.get("by") in EXT_BY for sp in k)
                nc -= len(k)
                out.add(i)
    kept = [x for i, x in enumerate(rows) if i not in out]
    n1, e1 = tally(kept)
    st = {c: [round(e0[c] / n0[c], 3), round(e1[c] / max(n1[c], 1), 3)] for c in n0 if e0[c] and n0[c] >= min_spans}
    return kept, {"rows_out": len(out), "classes": st}


SEAM_FILES = ("seam", "seam_b")  # seam_crops outputs build4 reads: tag "" and the top-up run (tag "_b", seed 2)


def _seam_rows(base: str, files=SEAM_FILES) -> list:
    """Seam halves of every listed seam_crops output: pair ids namespaced by file (each run numbers its pairs from s0),
    a half whose (side, text, parents) another file already gave is dropped with its pair."""
    out, seen = [], set()
    for name in files:
        rows = _rows(f"{base}/{name}.jsonl")
        by_pair, mine = {}, set()
        for x in rows:
            by_pair.setdefault(x["pair"], []).append(x)
        for pid, hs in by_pair.items():
            keys = [(h.get("side"), h["text"], tuple(h.get("parents", []))) for h in hs]
            if any(k in seen for k in keys):  # an earlier file has it (within one file pairs are distinct cuts)
                continue
            mine.update(keys)
            out += [dict(h, pair=f"{name}:{pid}") for h in hs]
        seen |= mine
    return out


def build_d2_phonefmt(vol: str, langs=None, mix_dir: str = "mix4", accept=None) -> dict:
    """mix4/d2_phonefmt_<lang>.jsonl: a digit-anchor part (mix4/d2anchor_<lang>) with its phone / code spans written
    spoken-style (phonefmt: space-separated groups, no comma / dot inside, leading zero kept; separators only). Written
    only for a language where at least one span changed, so the listing (modal_app.retrain3_mix_v4 phonefmt=True) swaps
    just those. accept(lang, before, after) -> bool: only spans a reader round-trip check passed are rewritten. Idempotent; the source part is never touched. Returns {lang: [rows, rows_changed, spans_changed]}."""
    import phonefmt as PF

    out = {}
    for f in sorted(glob.glob(f"{vol}/{mix_dir}/d2anchor_*.jsonl")):
        lang = os.path.basename(f)[len("d2anchor_"):-len(".jsonl")]
        if langs and lang not in langs:
            continue
        rows = _rows(f)
        new, rc, sc = [], 0, 0
        for x in rows:
            t, n = PF.rewrite_text(x["text"], accept=(lambda b, a, _l=lang: accept(_l, b, a)) if accept else None)
            new.append(dict(x, text=t) if n else x)
            rc, sc = rc + (n > 0), sc + n
        dst = f"{vol}/{mix_dir}/d2_phonefmt_{lang}.jsonl"
        if sc:
            _write(dst, new)
        elif os.path.exists(dst):
            os.remove(dst)
        out[lang] = [len(rows), rc, sc]
    return out


def build4(vol: str, langs=WAVE1, rows_h: float = 3.0, win_h: float = 1.0, seed: int = 41, allow_no_judge: bool = False,
           dedupe: bool = False, seam_h: float = 0.0, plain_share: float = 0.0,
           plain_glob: str = "ml/plain/{lang}/plain*.jsonl", protected_globs: tuple = (), balance: bool = False,
           seam_files: tuple = SEAM_FILES, mix_dir: str = "mix4", seam_long_h: float = 0.0) -> dict:
    """mix4/ manifests for the wave-1 run (all CPU, idempotent):
    - mix4/d2.jsonl: mix3/d2 (Greek restore applied) WITHOUT wave-1 languages: their digit rows were relabelled by P3
      (origin d2) and would otherwise teach digits through the stock token KL;
    - mix4/p3_<lang>.jsonl: P3 v2 rows (label2) minus held-out and judge-rejected ids, <= rows_h hours;
    - mix4/win_<lang>.jsonl: streaming-geometry windows (windows2), <= win_h hours, never trimmed (their edges are cuts);
    - mix4/heldout_<lang>.jsonl: the held-out rows (eval only, never in a training part).
    Audits (raise): spoken-form labels carry no digit / symbol (WF_LABEL); every p3/window row has num_chars / num_tokens;
    no held-out id or held-out parent inside a training part.
    Balance options (2026-10-16, ml_balance; the defaults keep the behaviour above):
    - dedupe: one row per utterance across passes / sources (ml_balance.dedupe); a duplicate of a held-out, validation,
      synthetic-dev or protected_globs row ("{lang}" filled in) leaves TRAINING (rows, synthetic rows, windows, d2p /
      FLEURS anchor rows of the language, plain and seam rows); protected rows are never touched;
    - seam_h > 0: mix4/seam_<lang>.jsonl, whole mirrored cut-number pairs (ml_p3.seam_crops output <lang>/seam*.jsonl),
      round-robin over cut position and number class, <= seam_h hours (reads <lang>/<seam_files>.jsonl: the main
      seam_crops output plus top-up runs; pair ids are namespaced per file, a half seen in two files is kept once);
    - balance (2026-10-17): balance_caps on each language's real rows (<= 3 per caption template, <= 13 per video /
      session); rows over a cap leave the mix, not the disk.
    - mix_dir: the output dir under the volume ("mix4"; a DRY build writes e.g. "mix4dry" and leaves mix4 alone);
    - seam_long_h > 0: mix4/seam_long_<lang>.jsonl, long-number seam pairs (<lang>/seam_long.jsonl: same row format and
      the same parent filters as seam pairs, whole pairs, <= seam_long_h hours; its weight is the listing's
      seam_long_repeat in modal_app.retrain3_mix_v4).
    Always (2026-10-17): a window / seam pair with a parent row decided by a second recognizer (EXT_BY) or a parent that
    is held out NOW (the held-out carve can move after windows were built) leaves the mix; a language in `langs` with
    no labelled rows keeps its mix4/d2 digit rows (it gets no spoken-form rows to replace them).
    - plain_share > 0: mix4/plain_<lang>.jsonl, number-free real rows (plain_glob, stock labels) up to number-free /
      (number-free + number) = plain_share per language, look-alike controls first (ml_balance.pick_plain)."""
    import ml_balance as B

    rng = random.Random(seed)
    stats = {"langs": {}}
    os.makedirs(f"{vol}/{mix_dir}", exist_ok=True)
    # a language's D2 digit rows leave only when it has labelled spoken-form rows (rows*.jsonl) to replace them
    no_rows = tuple(l for l in langs if not glob.glob(f"{vol}/ml/p3v2/{l}/rows*.jsonl"))
    stats["langs_without_rows"] = list(no_rows)
    langs = tuple(l for l in langs if l not in no_rows)
    d2 = _rows(f"{vol}/mix3/d2.jsonl")
    keep = [x for x in d2 if x.get("lang") not in langs]
    _write(f"{vol}/{mix_dir}/d2.jsonl", keep)
    stats["d2"] = {"rows": [len(d2), len(keep)], "h": [round(_h(d2), 2), round(_h(keep), 2)]}
    # review M5: wave-1 rows that SAY a number (no digits, but number words) inside the stock-anchored parts get the
    # full token KL to stock, which writes digits there: drop them from those parts (they are anchors, not targets)
    import ml_cands as MC

    # review 2026-10-15 H1: German "eine/einen/..." are articles and "null" an adjective far more often than numbers
    not_num = {"eine", "einen", "einem", "einer", "eines", "null"}
    lex = {l: {w.lower() for k, v in MC.table().items() if k.startswith(f"{l}|card|") for w in v if len(w) >= 4} - not_num
           for l in langs}
    anchor_h = {}  # wave-1 number-free anchor hours per language (mix4/d2p + mix4/fleurs_train_multi)
    protect = {}  # language -> protected rows (dedupe)
    held_units = set()  # held-out days / videos of every language (dedupe: anchor rows)
    for src in ("mix3/d2p.jsonl", "mix3/fleurs_train_multi.jsonl"):
        rows_ = _rows(f"{vol}/{src}")
        # rehearsal crash 2026-10-15: fleurs_train_multi carries FLEURS codes ("de_de"), so the language test never
        # matched and wave-1 number sentences stayed in this stock-anchored part (the same text was a spoken-form target)
        lc = lambda x: (x.get("lang") or "")[:2]  # noqa: E731
        kept = [x for x in rows_ if lc(x) not in langs
                or not ({w.lower().strip(".,;:!?\"'()") for w in x["text"].split()} & lex[lc(x)])]
        for x in kept:
            if lc(x) in langs:
                anchor_h[lc(x)] = anchor_h.get(lc(x), 0.0) + x["duration"] / 3600
        _write(f"{vol}/{mix_dir}/{os.path.basename(src)}", kept)
        stats[os.path.basename(src)] = {"rows": [len(rows_), len(kept)], "h": [round(_h(rows_), 2), round(_h(kept), 2)]}
    for lang in langs:
        base = f"{vol}/ml/p3v2/{lang}"
        hp, rp = f"{vol}/ml/p3v2/heldout_{lang}.txt", f"{base}/reject_ids.txt"
        assert os.path.exists(hp), f"{lang}: held-out list missing"  # review H3: never silently skip these
        assert allow_no_judge or os.path.exists(rp), f"{lang}: reject_ids.txt missing (run the judge first)"
        held = {l.strip() for l in open(hp) if l.strip()}
        rej = {l.strip() for l in open(rp) if l.strip()} if os.path.exists(rp) else set()
        # validation-only passes (val_*.jsonl, e.g. VoxPopuli dev/test): judged like every pass, never trained; their
        # ids (and below their days / videos) join the held-out set
        val = _rows(f"{base}/val_*.jsonl")
        for pth in glob.glob(f"{base}/val_*.jsonl"):
            name = os.path.basename(pth)[:-6]
            assert allow_no_judge or os.path.exists(f"{base}/judged_{name}.json"), f"{lang}: {name} was never judged"
        held |= {x["id"] for x in val}
        # every pass must have been judged (marker written by the judge step)
        for pth in glob.glob(f"{base}/rows*.jsonl"):
            name = os.path.basename(pth)[:-6]
            assert allow_no_judge or os.path.exists(f"{base}/judged_{name}.json"), f"{lang}: {name} was never judged"
        rows, dup = _one_per_id(sorted(glob.glob(f"{base}/rows*.jsonl")))
        stats.setdefault("dup_ids", {})[lang] = dup
        sfx = lambda i: (re.search(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$", str(i)) or [None, None])[1]  # noqa: E731
        held_sfx = {sfx(i) for i in held} - {None}
        held |= {x["id"] for x in rows if sfx(x["id"]) in held_sfx}
        # S5: the held-out unit is the YODAS video, not the utterance: rows from a held-out video stay out too
        vid = lambda i: (re.match(r"[a-z]{2}\d+_\d+_(.{11})_", str(i)) or [None, None])[1]  # noqa: E731
        held_vid = {vid(i) for i in held} - {None}
        held |= {x["id"] for x in rows if vid(x["id"]) in held_vid}
        # review H2: and the VoxPopuli plenary day (leakage check 2026-10-15: the safe unit)
        day = lambda i: (re.match(r"(\d{8})-\d{4}-PLENARY", str(i)) or [None, None])[1]  # noqa: E731
        held_day = {day(i) for i in held} - {None}
        held |= {x["id"] for x in rows if day(x["id"]) in held_day}
        for x in rows:
            assert "num_chars" in x, f"{lang}: P3 row without num_chars: {x['id']}"
            assert not WF_LABEL.search(x["text"]), f"{lang}: symbol in label: {x['text'][:80]}"
            for a, b in x["num_chars"]:  # no punctuation inside a spoken number (seam rule: nothing for windows to disagree on)
                assert not re.search(r"[,.;:!?]", x["text"][a:b]), f"{lang}: punctuation inside a number: {x['text'][a:b]!r}"
        ho = [x for x in rows + val if x["id"] in held and x["id"] not in rej]  # review M6: a rejected row is no gold
        tr = [x for x in rows if x["id"] not in held and x["id"] not in rej]
        held_dup = set()  # training ids removed for duplicating a protected row (their windows / pairs go too)
        if dedupe:
            part = {x["id"]: os.path.basename(p)[:-6] for p in glob.glob(f"{base}/rows*.jsonl") for x in _rows(p)}
            # this build's synthetic dev rows (held-out voices), read from the sets, not last build's mix4 copy
            tts_dev = [x for tag, _ in TTS_SETS for x in _rows(f"{vol}/ml/{tag}/{lang}/rows.jsonl") if x.get("split") == "dev"]
            prot = [x for x in rows + val if x["id"] in held] + tts_dev
            for pat in protected_globs:
                prot += _rows(f"{vol}/{pat.format(lang=lang)}")
            protect[lang] = prot
            held_units |= {B.held_unit(i) for i in held} - {None}
            tr, drep = B.dedupe(tr, prot, part_of=lambda x: part.get(x["id"], "?"))
            held_dup = {i for i, w in drep["removed"].items() if w.startswith("heldout")}
            stats.setdefault("dedupe", {})[lang] = {"rows": drep["counts"]}
        rng.shuffle(tr)
        # rare classes first (census 2026-10-15: rare cells were thin while plain integers filled the cap)
        tr = sorted(tr, key=lambda x: not any(sp["kind"] not in COMMON_KINDS for sp in x.get("spans", [])))
        # rows whose reading a second recognizer decided (span by == "ext") have one acoustic check
        # for the value: at most EXT_SHARE of the language's real row hours, never in windows or held-out
        ext = [x for x in tr if any(sp.get("by") in EXT_BY for sp in x.get("spans", []))]
        ext_ids = {x["id"] for x in ext}
        tr = [x for x in tr if x["id"] not in ext_ids]
        tr = _take(tr, rows_h)
        # review M7: the ext allowance sits on top of rows_h (rare classes first), so a full cap never starves it
        ext = sorted(ext, key=lambda x: not any(sp["kind"] not in COMMON_KINDS for sp in x.get("spans", [])))
        tr += _take(ext, EXT_SHARE * _h(tr) / (1 - EXT_SHARE))
        # 2026-10-17: the gate checks the ext share PER CLASS (de dec 0.36, int_big 0.28-0.30 under the language cap)
        tr, xst = cap_ext_per_class(tr, EXT_SHARE_CLASS, random.Random(f"{seed}-extcls-{lang}"))
        stats.setdefault("ext_per_class", {})[lang] = xst
        # 2026-10-17: no single year value above TOP_YEAR_SHARE of the language's year spans (2020 was 4-6%);
        # only rows whose every number is that year leave this mix (they stay on disk)
        n_tr = {x["id"] for x in tr}
        tr, ycap = _cap_top_years(tr, TOP_YEAR_SHARE, random.Random(f"{seed}-years-{lang}"))
        year_out = n_tr - {x["id"] for x in tr}  # their windows / seam pairs leave too (review 04:50: windows bypassed the cap)
        stats.setdefault("year_cap", {})[lang] = ycap
        if balance:
            before_caps = tr
            tr, bst = balance_caps(tr, lang, random.Random(f"{seed}-caps-{lang}"))
            # free fill: judged rows the caps took out come back when they hold a value the
            # language is still short of (data_gate's year / count floors), never past the 3% top-year rule
            kept_ids = {x["id"] for x in tr}
            # the pool: rows the caps took out, plus second-recognizer rows over the ext allowance (only while every
            # class stays at <= EXT_SHARE ext spans) -- never held-out / rejected / year-capped rows
            ext_left = [x for x in ext if x["id"] not in kept_ids and x["id"] not in year_out]
            pool = [x for x in before_caps if x["id"] not in kept_ids] + ext_left
            tr, fst = fill_thin(tr, pool, random.Random(f"{seed}-fill-{lang}"), ok=_ext_ok(tr))
            bst["fill"] = fst
            stats.setdefault("balance_caps", {})[lang] = bst
        ho = [x for x in ho if not any(sp.get("by") in EXT_BY for sp in x.get("spans", []))]
        # windows / seam pairs: no parent decided by a second recognizer (windows2 excluded only "ext" before 10-17),
        # no parent held out now
        ext_all = {x["id"] for x in rows if any(sp.get("by") in EXT_BY for sp in x.get("spans") or [])}
        unsafe = ext_all | held
        win = _rows(f"{base}/windows.jsonl") if os.path.exists(f"{base}/windows.jsonl") else []
        n_w0 = len(win)
        win = [w for w in win if not set(w.get("parents", [])) & unsafe]
        stats.setdefault("win_unsafe_parent_dropped", {})[lang] = n_w0 - len(win)
        win = [w for w in win if not set(w.get("parents", [])) & year_out]  # with or without dedupe
        if dedupe:
            n0 = len(win)
            win = [w for w in win if not set(w.get("parents", [])) & held_dup]
            win, wrep = B.dedupe(win, protect[lang], within=False, part_of=lambda x: "win")
            stats["dedupe"][lang]["windows"] = dict(wrep["counts"], parent_heldout_dup=n0 - wrep["counts"]["train_in"])
        for w in win:
            assert "num_tokens" in w and not WF_LABEL.search(w["text"]), f"{lang}: bad window {w['id']}"
            assert not set(w.get("parents", [])) & held, f"{lang}: window {w['id']} built from a held-out row"
            assert not set(w.get("parents", [])) & rej, f"{lang}: window {w['id']} built from a judge-rejected row"
        # synthetic rows (ml_tts): per class at most TTS_SHARE of that class's hours (real + synthetic); a class with
        # no real speech gets TTS_FLOOR_H so it is not left out entirely
        tts = []
        for tag, marker in TTS_SETS:  # tts1 = the first synthetic set, tts2 = the thin-class top-up (2026-10-15)
            if os.path.exists(f"{vol}/ml/{tag}/{lang}/rows.jsonl"):
                # review H5: synthetic rows are judged too (the judge step writes judged_<marker>.json)
                assert allow_no_judge or os.path.exists(f"{base}/judged_{marker}.json"), f"{lang}: {tag} rows never judged"
                tts += [dict(x, _targeted=tag in TTS_TARGETED) for x in _rows(f"{vol}/ml/{tag}/{lang}/rows.jsonl")]
        for x in tts:
            assert "num_chars" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad synthetic row {x.get('id')}"
        real_h = {}
        for x in tr:
            for sp in x.get("spans", []):
                real_h[sp["kind"]] = real_h.get(sp["kind"], 0) + x["duration"] / 3600
        _write(f"{vol}/{mix_dir}/tts_dev_{lang}.jsonl", [x for x in tts if x.get("split") == "dev"])  # held-out voices
        tts = [x for x in tts if x.get("split", "train") == "train" and x["id"] not in rej]  # judged like real rows
        if dedupe:  # a synthetic text equal to a held-out / eval text leaves training too (text keys only)
            tts, trep = B.dedupe(tts, protect[lang], within=False, part_of=lambda x: "tts")
            stats["dedupe"][lang]["tts"] = trep["counts"]
        rng.shuffle(tts)
        took, tts_h = [], {}
        for x in tts:
            # keyed by the row's span kinds (the ml_cands names real spans carry), not its template class: review
            # 2026-10-15 found "count"/"year"/"price" never matched a real kind, so every class got the floor cap
            ks = sorted({sp["kind"] for sp in x.get("spans", [])}) or ["?"]
            h = x["duration"] / 3600
            # lesson v12 (fast-year TTS broke real "twenty twenty two"): synthetic speech only FILLS classes real speech
            # lacks; a class with >= TTS_ENOUGH_H of real speech gets none. A row is taken only if every kind fits.
            def cap(k, targeted):
                rh = real_h.get(k, 0)
                sh = tts_share(lang, k)
                if rh >= TTS_ENOUGH_H and not targeted and k not in TTS_SHARE_KIND:
                    return 0.0
                return max(TTS_FLOOR_H, rh * sh / (1 - sh))
            if all(tts_h.get(k, 0) + h <= cap(k, x.get("_targeted")) for k in ks):
                took.append(x)
                for k in ks:
                    tts_h[k] = tts_h.get(k, 0) + h
        # a synthetic window may only stitch synthetic rows the class caps above admitted (v12 lesson)
        took_ids = {x["id"] for x in took}
        tts_ids = {x["id"] for x in tts}
        rng.shuffle(win)
        real_win = [w for w in win if not w.get("synthetic")]
        syn_win = [w for w in win if w.get("synthetic") and {p for p in w.get("parents", []) if p in tts_ids} <= took_ids]
        n_syn_dropped = sum(1 for w in win if w.get("synthetic")) - len(syn_win)
        # seam adversarial cases first: a number cut by an
        # edge, then a number at an edge, then the rest (the shuffle above keeps the order random inside each group)
        seam_first = lambda ws: sorted(ws, key=lambda w: (not w.get("cut_number"), not w.get("edge_number")))  # noqa: E731
        win = _take(seam_first(real_win), win_h * (1 - TTS_SHARE)) + _take(seam_first(syn_win), win_h * TTS_SHARE)
        # research 2026-10-15 + the TTS-prior lesson: synthetic speech is too clean; 60% of the chosen synthetic rows are
        # degraded (room reverb, noise 5-20 dB SNR, phone band, Opus) with the same Degrader as the real rows
        deg = Degrader(f"{vol}/slr28/RIRS_NOISES", seed + 7)
        os.makedirs(f"{vol}/{mix_dir}/tts_aug/{lang}", exist_ok=True)
        for i, x in enumerate(took):
            if rng.random() < TTS_AUG_FRAC:
                o = f"{vol}/{mix_dir}/tts_aug/{lang}/{i}_{os.path.basename(x['audio_filepath'])}"
                x["aug"] = deg(x["audio_filepath"], o)
                x["audio_filepath"] = o
        for x in took:
            x.pop("_targeted", None)
        _write(f"{vol}/{mix_dir}/tts_{lang}.jsonl", took)
        seam = []
        for k, on in (("seam", seam_h > 0), ("plain", plain_share > 0)):  # a part this build does not make must not
            if not on and os.path.exists(f"{vol}/{mix_dir}/{k}_{lang}.jsonl"):  # survive from an earlier build (the v4 mix
                os.remove(f"{vol}/{mix_dir}/{k}_{lang}.jsonl")                  # lists whichever of them exist)
        if seam_h > 0:
            seam = _seam_rows(base, seam_files)  # ml_p3.seam_crops tags "" and "_b" (dry runs write other tags)
            bad = held | rej | held_dup | year_out | ext_all
            seam = [x for x in seam if not set(x.get("parents", [])) & bad and not x.get("synthetic")]
            for x in seam:
                assert "num_tokens" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad seam crop {x['id']}"
            if dedupe:
                seam, srep = B.dedupe(seam, protect[lang], within=False, part_of=lambda x: "seam")
                stats["dedupe"][lang]["seam"] = srep["counts"]
            seam = B.take_seam(seam, seam_h, rng)
            _write(f"{vol}/{mix_dir}/seam_{lang}.jsonl", seam)
            NUMBER_PART[f"mix4/seam_{lang}.jsonl"] = round(_h([x for x in seam if any(x["num_tokens"])]) / max(_h(seam), 1e-9), 3)
            pc = {}
            for x in seam:
                pc[f"{x.get('pos_class')}/{x.get('side')}"] = pc.get(f"{x.get('pos_class')}/{x.get('side')}", 0) + 1
            seam_st = {"seam_rows": len(seam), "seam_h": round(_h(seam), 2), "seam_by_pos_side": dict(sorted(pc.items()))}
        seam_long = []
        if os.path.exists(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl") and not seam_long_h:
            os.remove(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl")  # a part this build does not make must not survive
        if seam_long_h > 0 and os.path.exists(f"{base}/seam_long.jsonl"):
            bad = held | rej | held_dup | year_out | ext_all
            seam_long = [x for x in _rows(f"{base}/seam_long.jsonl") if not set(x.get("parents", [])) & bad]
            # the long-number pairs come from the same streams as seam.jsonl: a pair whose (parents, cut position)
            # the main seam part already has leaves this part (whole pairs)
            have = {(tuple(x.get("parents", [])), x.get("pos")) for x in seam}
            dup = {x["pair"] for x in seam_long if (tuple(x.get("parents", [])), x.get("pos")) in have}
            seam_long = [x for x in seam_long if x["pair"] not in dup]
            stats.setdefault("seam_long_dup_pairs", {})[lang] = len(dup)
            for x in seam_long:
                assert "num_tokens" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad long seam crop {x['id']}"
            if dedupe:
                seam_long, lrep = B.dedupe(seam_long, protect[lang], within=False, part_of=lambda x: "seam_long")
                stats["dedupe"][lang]["seam_long"] = lrep["counts"]
            seam_long = B.take_seam(seam_long, seam_long_h, rng)
            _write(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl", seam_long)
        if plain_share > 0:
            pool = _rows(f"{vol}/{plain_glob.format(lang=lang)}")
            units = {B.held_unit(i) for i in held} - {None}
            pool = [x for x in pool if x.get("lang", lang)[:2] == lang and x["id"] not in held
                    and B.held_unit(x["id"]) not in units and not WF_LABEL.search(x["text"])]
            if dedupe:
                pool, prep = B.dedupe(pool, protect[lang], part_of=lambda x: "plain")
                stats["dedupe"][lang]["plain"] = prep["counts"]
            number_h = _h(tr) + _h(win) + _h(took) + _h(seam) + _h(seam_long)
            need = B.plain_target_h(number_h, plain_share, anchor_h.get(lang, 0.0))
            plain, pst = B.pick_plain(pool, lang, need, rng)
            _write(f"{vol}/{mix_dir}/plain_{lang}.jsonl", plain)
            free_h = anchor_h.get(lang, 0.0) + _h(plain)
            plain_st = {"plain_rows": len(plain), "plain_h": round(_h(plain), 2), "plain_pool_rows": len(pool),
                        "plain_pool_h": round(_h(pool), 2), "number_h": round(number_h, 2),
                        "anchor_free_h": round(anchor_h.get(lang, 0.0), 2), "plain_need_h": round(need, 2),
                        "number_free_share": round(free_h / max(free_h + number_h, 1e-9), 3), "plain_pick": pst}
        _write(f"{vol}/{mix_dir}/pairs_{lang}.jsonl", consistency_pairs(tr + took, win, lang, rng))
        _write(f"{vol}/{mix_dir}/p3_{lang}.jsonl", tr)
        _write(f"{vol}/{mix_dir}/win_{lang}.jsonl", win)
        _write(f"{vol}/{mix_dir}/heldout_{lang}.jsonl", ho)
        stats["langs"][lang] = {"rows": len(tr), "rows_h": round(_h(tr), 2), "windows": len(win), "win_h": round(_h(win), 2),
                                "tts_rows": len(took), "tts_h": round(_h(took), 2), "ext_rows": len(set(ext_ids) & {x["id"] for x in tr}),
                                "ext_h": round(_h([x for x in tr if x["id"] in ext_ids]), 2),
                                "syn_windows_outside_caps_dropped": n_syn_dropped,
                                "tts_h_by_class": {k: round(v, 3) for k, v in sorted(tts_h.items())},
                                "heldout": len(ho), "rejected": len(rej & {x["id"] for x in rows})}
        if seam_h > 0:
            stats["langs"][lang].update(seam_st)
        if seam_long:
            stats["langs"][lang].update({"seam_long_rows": len(seam_long), "seam_long_h": round(_h(seam_long), 2)})
        if plain_share > 0:
            stats["langs"][lang].update(plain_st)
    if dedupe:  # the stock-anchored parts' wave-1 rows: held-out day / video and duplicates of protected rows leave too
        for part in ("d2p.jsonl", "fleurs_train_multi.jsonl"):
            pth = f"{vol}/{mix_dir}/{part}"
            rows_ = _rows(pth)
            keep_, n_unit = [], 0
            by_lang = {}
            for x in rows_:
                lc = (x.get("lang") or "")[:2]
                if lc in protect and B.held_unit(x.get("id", "")) in held_units:
                    n_unit += 1
                    continue
                by_lang.setdefault(lc, []).append(x)
            for lc, xs in by_lang.items():
                if lc in protect:
                    xs, arep = B.dedupe(xs, protect[lc], part_of=lambda x: part)
                    stats["dedupe"].setdefault(lc, {})[part] = arep["counts"]
                keep_ += xs
            stats["dedupe"].setdefault("anchor_heldout_unit_dropped", {})[part] = n_unit
            _write(pth, keep_)
    # rehearsal crash 2026-10-15: a text that is a wave-1 spoken-form target (nonen_span) must not also sit in a
    # stock-anchored part (ft_train asserts it); drop such rows from the anchor parts instead of crashing at launch
    targets = {x["text"] for lang in langs for k in ("p3", "win", "tts") + (("seam",) if seam_h > 0 else ())
               + (("seam_long",) if seam_long_h > 0 else ())
               for x in _rows(f"{vol}/{mix_dir}/{k}_{lang}.jsonl")}
    for part in ("d2p.jsonl", "fleurs_train_multi.jsonl", "d2.jsonl") + tuple(f"plain_{l}.jsonl" for l in langs if plain_share > 0):
        pth = f"{vol}/{mix_dir}/{part}"
        if os.path.exists(pth):
            rows_ = _rows(pth)
            kept = [x for x in rows_ if x["text"] not in targets]
            if len(kept) != len(rows_):
                _write(pth, kept)
            stats.setdefault("target_text_collisions_dropped", {})[part] = len(rows_) - len(kept)
    json.dump(stats, open(f"{vol}/{mix_dir}/stats.json", "w"), indent=1)
    return stats


for _l in WAVE1:  # wave-1 spoken-form rows and windows are number-bearing rows (the 35-40% share counts them)
    NUMBER_PART[f"mix4/tts_{_l}.jsonl"] = 1.0
    NUMBER_PART[f"mix4/p3_{_l}.jsonl"] = 1.0
    NUMBER_PART[f"mix4/win_{_l}.jsonl"] = 1.0
NUMBER_PART["mix4/d2.jsonl"] = 1.0  # (mix4/d2p and mix4/fleurs_train_multi stay 0: plain anchor rows)
# English parts added 2026-10-17 (modal_app EN4_PARTS / EN_WIN_PARTS and the year-capped mix3b manifests): every row
# carries a number except the cloud set's control sentences and a few seam crops (shares measured on the manifests)
NUMBER_PART.update({"mix3b/d1_ycap.jsonl": 1.0, "mix3b/d10n_ycap.jsonl": 1.0,
                    "d1/en4_ok/yodas_en4_cap30_ycap.jsonl": 1.0, "d1/en4_ok/vox_en_crops_ok_ycap.jsonl": 1.0,
                    "d1/en4_ok/chirp_en_train.jsonl": 0.981, "d1/en4_ok/tts7_en_train.jsonl": 1.0,
                    "d1/en4_ok/tts9_en_train.jsonl": 1.0, "d1/en4_ok/tts10_en_train.jsonl": 1.0,
                    "d1/en4_ok/tts11_en_train.jsonl": 1.0, "d1/en4_ok/yodas_enh_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/tts12_en_train.jsonl": 1.0, "d1/en4_ok/yodas_rare_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/vox_rare_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/win_en.jsonl": 1.0, "d1/en4_ok/seam_en.jsonl": 0.977})
# the balance-capped English copies (en_balance) carry their source part's number share
NUMBER_PART.update({en_bcap_path(p): NUMBER_PART[p] for p in EN_REAL_PARTS})
