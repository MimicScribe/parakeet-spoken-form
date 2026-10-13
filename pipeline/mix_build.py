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

from common import WRITTEN_FORM


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
        stats["d2_h_by_lang"][x["lang"]] = round(stats["d2_h_by_lang"].get(x["lang"], 0) + x["duration"] / 3600, 2)
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
               "mix/d1.jsonl": 1.0, "mix/d1_up.jsonl": 1.0, "mix/d10n.jsonl": 1.0, "mix/d2.jsonl": 1.0}


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
