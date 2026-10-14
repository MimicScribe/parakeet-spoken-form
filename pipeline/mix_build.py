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
        out += [json.loads(line) for line in open(f)]
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

    # d1 + dev split by video. Percent-pass crops (*_pct) replace the original crops of the same rows (Opus B2);
    # HuBERT-vetoed readings drop the crops that contain them (Opus H1).
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
    # crops carrying a decided MULTI-reading span are the informative ones (Opus B1): they fill D1 first
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
    # "⁇" = stock's unknown-piece placeholder (Greek final sigma, Maltese letters): never a target (Opus B3)
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
    the FLEURS gates) at most 40% (Opus H2)."""
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
    (the wav2vec2 alignment midpoints often landed in the previous word: Opus H3)."""
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
               "tts/biz/train_ok_np.jsonl": 1.0, "mix3b/d1.jsonl": 1.0, "mix3b/d10n.jsonl": 1.0, "tts/mp/train_ok.jsonl": 1.0}


def mix_share(vol: str, mix: list) -> dict:
    """Hours each part contributes (as write_mix takes them) and the number-bearing share (V2: 35-40%)."""
    tot = num = 0.0
    parts = {}
    for path, hours, _tag in mix:
        h = _h(_rows(f"{vol}/{path}"))
        take = min(h, hours) if hours > 0 else h
        parts[path] = round(take, 2)
        tot += take
        num += take * NUMBER_PART.get(path, 0.0)
    return {"parts": parts, "total_h": round(tot, 2), "number_h": round(num, 2), "share": round(num / max(tot, 1e-9), 4)}


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
    """Retrain-3 external review: dropping every stock label with "\u2047" left Greek D2 at 0.05 h (final sigma is not in
    the vocab, so stock writes \u2047). Stock emits \u2047 there itself, so a "\u2047" label on a NON-ENGLISH row
    reproduces stock (no new behaviour). For languages under `min_h` in mix3/d2, add back their audited \u2047 rows."""
    import soundfile as sf

    base = f"{vol}/mix3/d2.pre_unk.jsonl"
    d2 = _rows(base if os.path.exists(base) else f"{vol}/mix3/d2.jsonl")  # idempotent: rebuild from the pre-restore file
    have = {}
    for x in d2:
        have[x["lang"]] = have.get(x["lang"], 0) + x["duration"] / 3600
    src = [x for f in sorted(glob.glob(f"{vol}/d2/*/digit.jsonl")) for x in _rows(f) if "\u2047" in x["text"]]
    eval_ng, eval_paths = set(), set()  # the build3 eval dedupe (re-review: restored rows skipped it)
    for f in glob.glob(f"{vol}/beds/e21x/refs/*.txt"):
        for t in re.split(r"(?<=[.?!])\s+", open(f).read()):
            eval_ng |= ngrams(t)
    for pat in ("e22/all.jsonl", "beds/fleurs/*.jsonl", "beds/vox/*.jsonl", "beds/adv/all.jsonl", "beds/e21_dev/*.jsonl",
                "beds/fleurs3/*.jsonl", "beds/vox/*3.jsonl"):
        for x in _rows(f"{vol}/{pat}"):
            eval_ng |= ngrams(x["text"].split("\t")[0])
            eval_paths.add(os.path.basename(x.get("audio_filepath", "")))
    src = [x for x in src if not ngrams(x["text"]) & eval_ng and os.path.basename(x["audio_filepath"]) not in eval_paths]
    random.Random(11).shuffle(src)
    seen = {x["audio_filepath"] for x in d2}
    added = {}
    for x in src:
        lang = x["lang"]
        if have.get(lang, 0) >= min_h or x["audio_filepath"] in seen:
            continue
        try:
            i = sf.info(x["audio_filepath"])
        except Exception:
            continue
        if i.samplerate != 16000 or abs(i.frames / i.samplerate - x["duration"]) > tol:
            continue
        seen.add(x["audio_filepath"])
        d2.append(x)
        have[lang] = have.get(lang, 0) + x["duration"] / 3600
        added[lang] = added.get(lang, 0) + 1
    if added or os.path.exists(base):
        if not os.path.exists(base):
            os.replace(f"{vol}/mix3/d2.jsonl", base)
        _write(f"{vol}/mix3/d2.jsonl", d2)
    return {"added_rows": added, "h_by_lang": {k: round(v, 3) for k, v in sorted(have.items())}}


def build3b(vol: str, per_video: int = 12, seed: int = 31) -> dict:
    """R3.31 fixed mix -> /vol/mix3b/ (the warm-start experiment's treatment; the control keeps mix3 unchanged):
    - the 2000-2009 cap (build3, R3.16 #2) is undone: the crops it dropped come back from d1/crops3_ok (same filters
      as build3: not a dev video, 0.5-20 s, no written form, not in eval text or eval wavs, not already in d1);
    - year rows per video are capped at `per_video` in d1 and in d10n (one call held 44 + 53);
    - rows repeating another row's text are dropped;
    - tts/biz without the k_twenty_pause rows (owner gold e2 refutes its premise) -> tts/biz/train_ok_np.jsonl."""
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
    """R3.31 v2 (external review #4):
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
