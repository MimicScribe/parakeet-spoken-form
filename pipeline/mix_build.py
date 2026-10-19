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
from collections import Counter

import numpy as np

from census_common import (EXT_BY, GROUP_CAP, GROUP_CAP_LANG, TEMPLATE_CAP, TEMPLATE_MIN_WORDS, ext_row,  # noqa: F401
                           ext_span, gate_class, gate_class_value, group_cap, group_id, held_unit, span_class,
                           SYN_SHARE_KIND, SYN_SHARE_LANG, class_bar, filled_to_bar, syn_limit, syn_ok, template_key,
                           year_of, read_held_list)
from census_common import RARE_FLOOR_H, RARE_REAL_MAX, distinct_bar, rare_floor_ok, value_needs  # noqa: F401
from census_common import synthetic as _synthetic
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
_LANG_PART = re.compile(r"^mix4\w*/(p3|win|win15|tts|seam|seam_long|d2anchor|d2_phonefmt|d2phone)_([a-z]{2})\.jsonl$")


def number_share_of(path: str, rows: list | None = None) -> float:
    """Number-bearing share of a mix part's hours. Seam-pair parts are measured on their rows (a half cut before or
    after a reading holds no number word); per-language mix4 parts of ANY language count 1.0; else NUMBER_PART."""
    src = _bcap_source(path)
    if src:  # an English capped copy carries its source part's number share
        return NUMBER_PART.get(src, 1.0)
    m_rl = re.match(r"^mix4\w*/en_rl/(.+)$", path)
    if m_rl:  # a relabelled English copy carries its source part's number share (an unlisted source: number-free)
        return NUMBER_PART.get(m_rl.group(1).replace("__", "/"), 0.0)
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
    with open(f"{vol}/mix3b/stats_v.json", "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# ---- wave-1 mix (2026-10-15): spoken-form numbers in de/es/fr/it/nl/pt -----------------------------------------------

WAVE1 = ("de", "es", "fr", "it", "nl", "pt")
# EXT_BY (census_common): the span kinds decided mainly by the second transcript, shared with the data gate
EXT_SHARE = 0.25  # max share of a language's real row hours decided by a second recognizer
# per class (cap_ext_per_class, fill): a point under the gate's 0.25; the classes are the gate's own (ext_class with the
# row and language = census_common.gate_class), the point is margin for the rows later steps remove
EXT_SHARE_CLASS = 0.24
COMMON_KINDS = {"int", "pct", "dec", "ord"}  # number kinds real speech has plenty of; the rest are "rare"
TTS_SHARE = 0.30  # synthetic windows: at most 30% of the windows' hours
# the synthetic share per class is the data gate's rule (census_common.SYN_SHARE_KIND / SYN_SHARE_LANG / syn_limit,
# spans per gate class); these names stay for callers
TTS_SHARE_KIND = SYN_SHARE_KIND  # shapes real speech rarely has (clock / price / phone / date): up to 50%
TTS_SHARE_LANG = SYN_SHARE_LANG  # en / es / fr: every class up to 50% (decision 2026-10-17)


def tts_share(lang: str, kind: str) -> float:
    """Max synthetic share of a class's spans in a language (census_common.syn_limit)."""
    return syn_limit(lang, kind)


TTS_SETS = (("tts1", "tts_judge"), ("tts2", "tts2_judge"), ("tts5y", "tts5y_judge"),
            ("tts6uk", "tts6uk_judge"),  # (volume dir, judge pass name); tts6uk: Ukrainian coverage (ml_tts_uk)
            ("tts7", "tts7_judge"), ("tts8", "tts8_judge"),  # tts7/tts8 (2026-10-17): es/fr thin year + count values (ml_tts_fill)
            ("tts9nl", "tts9nl_judge"),  # tts9nl (2026-10-17): Dutch thin 0-100 counts, dates, phones, prices
            ("tts9ru", "tts9ru_judge"), ("tts9ru2", "tts9ru2_judge"),  # tts9ru/2 (2026-10-17): Russian coverage (ml_tts_ru)
            ("ttsm2", "ttsm2_judge"),  # ttsm2: sv / cs / hu / fi / sk clock times + prices (ml_tts_multi)
            ("ttsm1", "ttsm1_judge"),  # ttsm1: the pl / ro synthetic set
            # ttsL0/1: LONG numbers (big, phone, price, date, year 1990-2060, big ranges) for pl ro sv cs hu fi sk bg hr;
            # normal rates, dev by carrier template
            ("ttsL0", "ttsL0_judge"), ("ttsL1", "ttsL1_judge"),
            # ttsL2p/2: ro / bg / fi phones read digit by digit and hu dates, the reading built in code from the drawn
            # value (not by the LLM)
            ("ttsL2p", "ttsL2p_judge"), ("ttsL2", "ttsL2_judge"),
            ("ttsES1", "ttsES1_judge"),  # ttsES1: es exact thin values (years 1993 / 1994 / 1998 / 2052, count 79)
            ("ttsFT1", "ttsFT1_judge"), ("ttsFT2", "ttsFT2_judge"),  # ttsFT1/2: exact-value fills for fr / de / pt / it
            ("ttsFT3", "ttsFT3_judge"), ("ttsFT4", "ttsFT4_judge"),  # ttsFT3/4: it / nl counts 0-100
            # ttsFT5: es / pt / fr / it thin years + fr / pt counts; ttsFT5c / 5d: fr decimals, it 24 h clock times
            ("ttsFT5", "ttsFT5_judge"), ("ttsFT5c", "ttsFT5c_judge"), ("ttsFT5d", "ttsFT5d_judge"),
            # ttsFT6c / ttsFT6r / ttsFT6rd: grammar-checked replacements for excluded rows (grammar_scan): it 24 h
            # clock times, ro prices, ro dates (judge rejects left out of the sets)
            ("ttsFT6c", "ttsFT6c_judge"), ("ttsFT6r", "ttsFT6r_judge"), ("ttsFT6rd", "ttsFT6rd_judge"),
            # ttsLNw / ttsLNr / ttsLNm (2026-10-17): long numbers (int_big, prices with cents, phones) placed mid / late
            # in the sentence, for wave-1 (ml_tts), uk / ru (ml_tts_uk / ru) and the LLM-written languages (ml_tts_multi)
            ("ttsLNw", "ttsLNw_judge"), ("ttsLNr", "ttsLNr_judge"), ("ttsLNm", "ttsLNm_judge"),
            # ttsYRm / ttsYRr (2026-10-17): years for the next / moderate tiers, 2-3 per sentence, normal rates
            ("ttsYRm", "ttsYRm_judge"), ("ttsYRr", "ttsYRr_judge"),
            # ttsHOw / ttsHOr / ttsHOm: the held-out voice's decision-set rows (split dev only, HELDOUT_VOICES)
            ("ttsHOw", "ttsHOw_judge"), ("ttsHOr", "ttsHOr_judge"), ("ttsHOm", "ttsHOm_judge"),
            # second batch (2026-10-17): years ru / pl / sv / fi (ttsYRs, ttsYRn), pl / ro small counts 2-3 per sentence
            # (ttsCNm), the thinnest long cells de phone / ru phone + price (ttsLTw, ttsLTr), es / fr held-out rows (ttsHOx)
            ("ttsYRs", "ttsYRs_judge"), ("ttsYRn", "ttsYRn_judge"), ("ttsCNm", "ttsCNm_judge"),
            ("ttsLTw", "ttsLTw_judge"), ("ttsLTr", "ttsLTr_judge"), ("ttsHOx", "ttsHOx_judge"),
            # Dutch top-up (2026-10-18, ml_tts_topup): years / decimals / clock / decades / counts (ttsNL1) and the long
            # classes prices / phone / big counts (ttsNL2), planned against the nl census gaps
            ("ttsNL1", "ttsNL1_judge"), ("ttsNL2", "ttsNL2_judge"),
            # near-miss top-up (2026-10-18, ml_tts_topup): it decimals / decades / counts, pt big counts + years, es years,
            # de phone, nl long classes + headroom over the bar
            ("ttsW1", "ttsW1_judge"),
            # reading-variety fill (2026-10-18, ml_tts_variety): the same numbers read the other ways people say them
            # (relative clock times, natural phone groupings, paired years, prices without the unit, ...); its rows
            # REPLACE the older synthetic rows they name (TTS_REPLACING)
            ("ttsRV1b", "ttsRV1b_judge"), ("ttsRV1r2", "ttsRV1r2_judge"),
            # symbol-class set (2026-10-18, ml_tts_symbol): percent with the unit word, decimals with the decimal word,
            # prices, units, big numbers with magnitude words, phones, letter+digit codes; 2-4 numbers per sentence,
            # open-licence cloned voices (ml_tts_open); sized to the synthetic room of each class in the current mix
            ("ttsSYM1", "ttsSYM1_judge"),
            # ttsSYM1r: the same generator's single-class rows for the cells at their synthetic cap (nl / it cur, phone,
            # dec, int_big; de phone; pt int_big; de / it Roman numerals); each REPLACES older single-class synthetic
            # rows of its class (TTS_REPLACING, never a ttsRV1 row), so the class sizes stay as they are
            ("ttsSYM1r", "ttsSYM1r_judge"),
            # tier-1 packs (2026-10-18, ml_tts_pack): decimal words (de es fr it nl pt), percent with the unit word
            # (es fr pt), and the verified look-alike / business-reading set (tvar1v)
            ("ttsDW1", "ttsDW1_judge"), ("ttsPC1", "ttsPC1_judge"), ("tvar1v", "tvar1v_judge"),
            # value-spread (ttsVS1, V_a values only) and rare-form (ttsRF1: feminine hundreds, apocope, pt-PT teens) packs
            ("ttsVS1", "ttsVS1_judge"), ("ttsRF1", "ttsRF1_judge"))
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
TTS_TARGETED = {"tts5y", "tts6uk", "tts7", "tts8", "tts9nl", "tts9ru", "tts9ru2", "ttsm2", "ttsm1", "ttsL0", "ttsL1", "ttsL2p", "ttsL2", "ttsES1",
                "ttsFT1", "ttsFT2", "ttsFT3", "ttsFT4", "ttsFT5", "ttsFT5c", "ttsFT5d",
                "ttsFT6c", "ttsFT6r", "ttsFT6rd", "ttsLNw", "ttsLNr", "ttsLNm", "ttsYRm", "ttsYRr",
                "ttsYRs", "ttsYRn", "ttsCNm", "ttsLTw", "ttsLTr", "ttsNL1", "ttsNL2", "ttsW1", "ttsRV1b", "ttsRV1r2", "ttsSYM1",
                "ttsSYM1r", "ttsDW1", "ttsPC1", "ttsVS1", "ttsRF1"}
# sets whose rows replace older synthetic rows (2026-10-18, B2): a row's "supersedes" lists the ids it replaces (rows of
# the same class and value spread, read one convention; ml_tts_variety picked them). order_synthetic drops those ids
# and puts the set's rows first, so they take the room the dropped rows leave (a class at its fill-to-bar or share cap
# would otherwise admit none of them)
TTS_REPLACING = {"ttsRV1b", "ttsRV1r2", "ttsSYM1r"}  # ttsRV1r2: re-judged ttsRV1b rows, century-dropped years out (ttsRV1r: mix4final9b)
REPLACE_ROUNDS = 3  # build4 reruns of the synthetic selection when a replacing row is refused (then fatal)
TTS_AUG_FRAC = 0.6  # share of chosen synthetic rows given a degraded copy instead of the clean audio
# the long-number top-up sets (2026-10-17): mostly synthetic in their classes (fill-to-bar), so more of them are degraded
# ttsRV1 (2026-10-18): its cells are synthetic-heavy too (clock / phone / price rows mostly synthetic), and it replaces
# rows of the long sets, so it keeps their degraded share
# ttsSYM1 (2026-10-18): long numbers (phones, big amounts, decimals with units) in mostly-synthetic cells, like the long sets
TTS_LONG_SETS = {"ttsLNw", "ttsLNr", "ttsLNm", "ttsLTw", "ttsLTr", "ttsNL2", "ttsW1", "ttsRV1b", "ttsRV1r2", "ttsSYM1",
                 "ttsSYM1r", "ttsDW1", "ttsPC1"}
TTS_AUG_FRAC_LONG = 0.75


def aug_frac(row: dict) -> float:
    """The degraded share for a chosen synthetic row, by its set (row["_set"])."""
    return TTS_AUG_FRAC_LONG if row.get("_set") in TTS_LONG_SETS else TTS_AUG_FRAC
TTS_ENOUGH_H = 0.25  # h of real speech in a class above which no synthetic rows of that class are used
# a gate class with no bar that real speech (almost) never has (de / it / nl "&", fractions, roman numerals, versions;
# the moderate tier's dates / phones) may hold up to TTS_FLOOR_H hours of synthetic rows whatever its synthetic share
# (census_common.rare_floor_ok: never a class with a bar, never bg); the data gate passes it as a "waived floor" and
# post_build_check reports it as "waived_synthetic_floor"
TTS_FLOOR_H = RARE_FLOOR_H
# one cloud voice per language kept entirely out of training (2026-10-17): its rows are the language's held-out
# synthetic decision set (split "dev": tts_dev, seam_crops --heldout-tts); a train-split row in that voice never trains
HELDOUT_VOICES = {
    "cs": "cs-CZ-Chirp3-HD-Achird", "de": "de-DE-Chirp3-HD-Algieba", "es": "es-US-Chirp3-HD-Erinome",
    "fr": "fr-CA-Chirp3-HD-Sadaltager",
    "fi": "fi-FI-Chirp3-HD-Laomedeia", "hr": "hr-HR-Chirp3-HD-Alnilam",
    "hu": "hu-HU-Chirp3-HD-Erinome", "it": "it-IT-Chirp3-HD-Alnilam", "nl": "nl-NL-Chirp3-HD-Iapetus",
    "pl": "pl-PL-Chirp3-HD-Orus", "pt": "pt-BR-Chirp3-HD-Iapetus", "ro": "ro-RO-Chirp3-HD-Kore",
    "ru": "ru-RU-Chirp3-HD-Aoede", "sk": "sk-SK-Chirp3-HD-Achird", "sv": "sv-SE-Chirp3-HD-Rasalgethi",
    "uk": "uk-UA-Chirp3-HD-Achird"}
# languages a build may take without a held-out voice: English has its own held-out route, bg stays at stock
HELDOUT_EXEMPT = {"en", "bg"}


def heldout_persona(voice: str) -> str:
    """A cloud voice's persona ("es-US-Chirp3-HD-Erinome" -> "Erinome"): personas repeat across a language's locales
    (es-ES / es-US, fr-FR / fr-CA, nl-NL / nl-BE), so the held-out voice is the persona in every locale."""
    return (voice or "").rsplit("-", 1)[-1]


def is_heldout_voice(lang: str, voice: str) -> bool:
    hv = HELDOUT_VOICES.get(lang)
    return bool(hv and voice and heldout_persona(voice) == heldout_persona(hv))


def check_heldout_voices(langs) -> None:
    """Every build language has a held-out voice or is exempt (HELDOUT_EXEMPT)."""
    missing = sorted(l for l in langs if l not in HELDOUT_VOICES and l not in HELDOUT_EXEMPT)
    assert not missing, f"no held-out voice for {missing} (add one to HELDOUT_VOICES or exempt it)"
# languages with generator slip rules (grammar_rules): a build with synthetic rows and no grammar scan list says so
GRAMMAR_RULE_LANGS = {"de", "es", "fr", "it", "nl", "pt", "pl", "ro", "ru", "uk"}
# post_build_check failures that do not fail the build (accepted waivers; the gate still reports them)
POST_CHECK_WAIVED = ("rows over the template cap", "groups over the cap")
GRAMMAR_REJECTS = "grammar_reject_ids.txt"  # per language in ml/p3v2/<lang>/: synthetic rows whose audio speaks a
# grammar slip of an old generator (grammar_scan.py); they never train


def syn_room(lang: str, c: str, syn: int, real: int, syn_h: float, real_vals=None, syn_vals=None) -> bool:
    """May class c hold `syn` synthetic spans (in rows of `syn_h` hours) beside `real` real spans: the gate's rule
    (census_common.syn_ok: the share caps, or fill-to-bar: up to the class bar and the thin values' floors; the
    {value: spans} counts real_vals / syn_vals), or the rare-class floor (census_common.rare_floor_ok: a class with no
    bar, not bg, real <= RARE_REAL_MAX and syn_h <= TTS_FLOOR_H; the gate applies the same waiver)."""
    return (syn_ok(lang, c, syn, real + syn, real_vals=real_vals, syn_vals=syn_vals)
            or rare_floor_ok(lang, c, syn, real, syn_h))


def class_values(rows: list, lang: str) -> tuple:
    """({class: spans}, {class: Counter(value -> spans)}) in the gate's classes."""
    n, vals = Counter(), {}
    for x in rows:
        for sp in x.get("spans") or []:
            c, v = gate_class_value(x, sp, lang)
            n[c] += 1
            vals.setdefault(c, Counter())[v] += 1
    return n, vals


def _syn_hours(rows: list, lang: str) -> Counter:
    """Hours of synthetic rows per gate class they hold (a row with spans of two classes counts in both)."""
    h = Counter()
    for x in rows:
        for c in {gate_class(x, sp, lang) for sp in x.get("spans") or []}:
            h[c] += x.get("duration", 0.0) / 3600
    return h


def read_id_list(path: str) -> set:
    """Ids of a one-id-per-line list ("#" lines are comments); an absent file is an empty list."""
    if not os.path.exists(path):
        return set()
    return {t for t in (l.strip() for l in open(path, encoding="utf-8")) if t and not t.startswith("#")}
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
# the second top-year trim (after the balance caps) aims this far under TOP_YEAR_SHARE: the per-class ext cap that runs
# after it removes a few more real rows (each removal raises the other years' shares), and the gate reads the share on
# the written rows, so the trim leaves 0.4 pp of room instead of landing exactly on 3%
TOP_YEAR_MARGIN = 0.004
def _year_of(sp, r=None, lang: str = ""):
    """The year value of a span as the data gate counts years (census_common.year_of: a 4-digit 1800-2099 int caption
    token, hu / fi suffixed years by their number part; an ordinal "2015." is not a year), else None."""
    return year_of(r or {}, sp, lang)


def _cap_top_years(rows: list, share: float, rng, mixed_ok=None, lang: str = "") -> tuple:
    """Leave out rows whose every number span is an over-represented year value until no value exceeds `share` of the
    year spans (rows with any other number stay: they carry other classes). Years as the gate counts them (_year_of)."""
    from collections import Counter
    ys = {x["id"]: [_year_of(sp, x, lang) for sp in x.get("spans") or []] for x in rows}
    cnt = Counter(y for x in rows for y in ys[x["id"]] if y)
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
        only = [(x["id"], len(ys[x["id"]])) for x in rows if x["id"] not in drop and ys[x["id"]]
                and all(v == y for v in ys[x["id"]])]
        rng.shuffle(only)
        if mixed_ok is not None:  # then rows that hold the year next to other numbers, when mixed_ok(row) allows it
            mixed = [(x["id"], ys[x["id"]].count(y)) for x in rows if x["id"] not in drop
                     and y in ys[x["id"]] and not all(v == y for v in ys[x["id"]]) and mixed_ok(x)]
            rng.shuffle(mixed)
            only += mixed
        k = 0
        while n > t and k < len(only):
            i, per = only[k]
            drop.add(i); n -= per; k += 1
        out[y] = {"before": cnt[y], "after": n, "rows_out": k}
    return [x for x in rows if x["id"] not in drop], out

# balance caps (the data gate's variety rules, applied in the build): <= TEMPLATE_CAP real number rows per digit-masked
# caption template, <= GROUP_CAP per video / session (English YouTube: 30). Rows over a cap leave the MIX (they stay on
# disk); within a template / group the rows with a rare number class are kept first. group_id / group_cap /
# template_key / TEMPLATE_CAP come from census_common, which the data gate imports too.


def _rare(r: dict) -> bool:
    return any(sp.get("kind") not in COMMON_KINDS for sp in r.get("spans") or [])


def _span_values(r: dict, lang: str = "") -> list:
    """("year", v) / ("int", v) keys of a row's integer spans (caption digits), for the thin-value protection; with
    `lang` in the data gate's classes (a day number said with its month is a date, not a count)."""
    out = []
    for sp in r.get("spans") or []:
        c, v = gate_class_value(r, sp, lang)  # the gate's (class, value): the same years the top-year rule counts
        if c == "year" and isinstance(v, int):
            out.append(("year", v))
        elif c == "int_small" and isinstance(v, int) and v <= 100:
            out.append(("int", v))
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


def _ext_ok(rows: list, lang: str = ""):
    """A running per-class ext-share check for fill_thin: adding a row keeps every class's ext spans <= EXT_SHARE_CLASS
    (classes as the data gate counts them when `lang` is given). Returns ok(row) -> bool; True also records the row."""
    from collections import Counter
    n, e = Counter(), Counter()
    for x in rows:
        for sp in x.get("spans") or []:
            c = ext_class(sp, x, lang)
            n[c] += 1
            e[c] += ext_span(sp)

    def ok(x) -> bool:
        dn, de = Counter(), Counter()
        for sp in x.get("spans") or []:
            c = ext_class(sp, x, lang)
            dn[c] += 1
            de[c] += ext_span(sp)
        if any(de[c] and e[c] + de[c] > EXT_SHARE_CLASS * (n[c] + dn[c]) for c in dn):
            return False
        n.update(dn); e.update(de)
        return True
    return ok


def _caps_ok(rows: list, lang: str, template_cap: int = TEMPLATE_CAP):
    """A running template / session cap check for fill_thin, counted as balance_caps and the data gate count (real rows
    with number spans): adding a row keeps its template at <= template_cap rows and its video / session at <= its group
    cap. Rows already over a cap (rare-class rows balance_caps kept) stay; nothing more joins them.
    Returns (fits(row) -> bool, add(row))."""
    from collections import Counter
    nt, ng = Counter(), Counter()
    counted = lambda r: r.get("spans") and not _synthetic(r)  # noqa: E731
    for x in rows:
        if counted(x):
            nt[template_key(x)] += 1
            ng[group_id(x)] += 1

    def fits(x) -> bool:
        if not counted(x):
            return True
        t, g = template_key(x), group_id(x)
        return not ((t and nt[t] + 1 > template_cap) or (not g.startswith("id:") and ng[g] + 1 > group_cap(lang, g)))

    def add(x):
        if counted(x):
            nt[template_key(x)] += 1
            ng[group_id(x)] += 1
    return fits, add


def fill_thin(rows: list, pool: list, rng, top_share: float = TOP_YEAR_SHARE, ok=None, lang: str = "",
              caps: bool = True) -> tuple:
    """Add pool rows (judged rows a cap took out) that hold a year (1990-2060) or 0-100 count still under its gate floor,
    until it is met; a row is added only if no year value then exceeds `top_share` of the year spans, ok(row) allows it
    (the ext-share check) and, with `caps`, its template and video / session stay within the balance caps (_caps_ok)."""
    from collections import Counter
    cnt = Counter(v for x in rows for v in _span_values(x, lang))
    cap_fits, cap_add = _caps_ok(rows, lang) if caps else (None, None)
    years = sum(c for (k, _), c in cnt.items() if k == "year")
    pool = list(pool)
    rng.shuffle(pool)
    added, out = 0, list(rows)
    n_cap = 0
    for x in pool:
        vs = _span_values(x, lang)
        if not any(cnt[v] < _fill_floor(v) for v in vs):
            continue
        new = Counter(vs)
        ny = years + sum(c for (k, _), c in new.items() if k == "year")
        if any(cnt[v] + c > top_share * ny for v, c in new.items() if v[0] == "year"):
            continue  # one of its years would pass the 3% top-year share
        if cap_fits is not None and not cap_fits(x):
            n_cap += 1
            continue  # it would put its template / session over the cap the gate checks
        if ok is not None and not ok(x):
            continue
        if cap_add is not None:
            cap_add(x)
        out.append(x)
        cnt.update(new)
        years = ny
        added += 1
    short = sorted(f"{k}{x}" for (k, x), c in cnt.items() if c < _fill_floor((k, x)) - 2 and k == "year")
    return out, {"added": added, "skipped_over_cap": n_cap, "still_short_years": short[:20]}



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
    vc = Counter(v for x in rows for v in _span_values(x, lang))
    thin = lambda r: any(vc[v] <= _thin_floor(v) for v in _span_values(r, lang))  # noqa: E731
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


def load_en_relabel(rows_path: str, drops_path: str = "") -> dict:
    """{(part, audio_filepath): new text, or None for a drop} from en_relabel.py's output files (relabelled rows keep
    their old keys and carry the new "text" and the "part" they came from; the drops file lists rows with no spoken form)."""
    rel = {}
    for path, drop in ((rows_path, False), (drops_path, True)):
        if path and os.path.exists(path):
            for r in _rows(path):
                rel[(r["part"], r["audio_filepath"])] = None if drop else r["text"]
    return rel


def apply_en_relabel(part: str, rows: list, rel: dict) -> tuple:
    """(rows with the part's relabels applied and its drops removed, n relabelled, n dropped). Only "text" changes."""
    out, n_rl, n_dr = [], 0, 0
    for x in rows:
        k = (part, x.get("audio_filepath"))
        if k not in rel:
            out.append(x)
        elif rel[k] is None:
            n_dr += 1
        else:
            n_rl += rel[k] != x.get("text")
            out.append(dict(x, text=rel[k]))
    return out, n_rl, n_dr


def en_rl_path(part: str, mix_dir: str) -> str:
    """Where a relabelled copy of an English part lives: <mix_dir>/en_rl/<part with / as __> (the listing uses it when it exists)."""
    return f"{mix_dir}/en_rl/{part.replace('/', '__')}"


def write_en_relabel_parts(vol: str, mix_dir: str, rel: dict, skip: tuple = ()) -> dict:
    """Relabelled copies of every part in `rel` except `skip` (the EN_REAL_PARTS: en_balance applies the relabels to its
    capped copies). Returns {part: [rows in, relabelled, dropped]}."""
    st = {}
    for part in sorted({k[0] for k in rel} - set(skip)):
        if not os.path.exists(f"{vol}/{part}"):
            st[part] = "source missing"
            continue
        rows, n_rl, n_dr = apply_en_relabel(part, _rows(f"{vol}/{part}"), rel)
        _write(f"{vol}/{en_rl_path(part, mix_dir)}", rows)
        st[part] = [len(rows) + n_dr, n_rl, n_dr]
    return st


def en_bcap_path(path: str, mix_dir: str = "mix4") -> str:
    """Where en_balance writes a part's capped copy: next to the source for the mix4 build, under <mix_dir>/en_bcap/
    for a build into another dir (a candidate mix never overwrites the copies the mix4 build uses)."""
    if mix_dir and mix_dir != "mix4":
        return f"{mix_dir}/en_bcap/{os.path.basename(path)[:-6]}_bcap.jsonl"
    return path[:-6] + "_bcap.jsonl"


def _bcap_source(path: str) -> str | None:
    """The EN_REAL_PARTS source of a _bcap copy path (either location), else None."""
    for p in EN_REAL_PARTS:
        if path == p[:-6] + "_bcap.jsonl" or (path.endswith("/en_bcap/" + os.path.basename(p)[:-6] + "_bcap.jsonl")):
            return p
    return None


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


def _en_years(vals) -> list:
    return [v for v in vals if 1800 <= v <= 2099]


def en_top_year_trim(rows: dict, gone: set, flat: list, other_years, share: float, tparts=EN_TRIM_PARTS) -> tuple:
    """The second top-year trim of the non-English build (build4 after the balance caps), for the English capped copies:
    the caps and the share trim shrink the year spans, so a top year creeps back over 3%. Year values are counted over
    the kept real crops (rows minus `gone`) plus `other_years` (a Counter: the other English parts of the listing, e.g.
    the synthetic sets). While a year is over `share`, crops of the YouTube trim parts that hold it leave: crops whose
    every number is that year first, then crops holding it beside other numbers; never a rare-class crop, never one
    holding a value at its _en_floor. Returns (cut {(part, i)}, {year: {"before", "after", "rows_out"}})."""
    from collections import Counter

    rank = {p: k for k, p in enumerate(tparts)}
    vals = {pi: _en_values(rows[pi[0]][pi[1]].get("text", "")) for pi in flat if pi not in gone}
    cnt = Counter(other_years)
    allv = Counter(other_years)
    for v in vals.values():
        cnt.update(_en_years(v))
        allv.update(v)
    total = sum(cnt.values())
    cut, st = set(), {}
    if not total:
        return cut, st
    over = sorted((y for y, n in cnt.items() if n > share * total), key=lambda y: -cnt[y])
    # one fixed per-value limit t, solved so the capped values sit at `share` of the year spans left (as _cap_top_years):
    # no cascade onto values that were under the share to begin with
    t = share * total
    for _ in range(5):
        t = share * (total - sum(max(0, cnt[y] - t) for y in over))
    for y in over:
        n0 = cnt[y]
        cand = [pi for pi, v in vals.items() if pi not in cut and pi[0] in rank and y in v and not _en_rare(rows[pi[0]][pi[1]])]
        cand.sort(key=lambda pi: (not all(x == y for x in _en_years(vals[pi])), rank[pi[0]], pi))
        k = 0
        for pi in cand:
            if cnt[y] <= t:
                break
            need = Counter(vals[pi])
            if any(allv[v] - c < _en_floor(v) for v, c in need.items()):
                continue
            cut.add(pi)
            k += 1
            ny = _en_years(vals[pi])
            cnt.subtract(Counter(ny)); allv.subtract(need)
        st[y] = {"before": n0, "after": cnt[y], "rows_out": k}
    return cut, st


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
               out_dir: str = "", value_aware: bool = True, year_real_floor: int = EN_YEAR_REAL_FLOOR,
               bcap_dir: str = "mix4", drop_audio: frozenset = frozenset(), relabel: dict | None = None) -> dict:
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
    en_mix = [((_bcap_source(m[0]) or m[0]),) + tuple(m[1:]) for m in en_mix]  # _bcap copies: judged on their sources
    real = [p for p in EN_REAL_PARTS if any(m[0] == p for m in en_mix)]
    rows = {p: _rows(f"{vol}/{p}") for p in real}
    n_listed = 0
    if drop_audio:
        n_listed = sum(1 for p in real for x in rows[p] if x.get("audio_filepath") in drop_audio)
        rows = {p: [x for x in rows[p] if x.get("audio_filepath") not in drop_audio] for p in real}
    n_rl = n_rd = 0
    if relabel:
        for p in real:
            rows[p], a_, b_ = apply_en_relabel(p, rows[p], relabel)
            n_rl += a_
            n_rd += b_
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
    trim, n_per_video, vst = set(), None, None
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
    # the same second top-year trim build4 gives the other languages (TOP_YEAR_SHARE - TOP_YEAR_MARGIN), counted over
    # the kept real crops and every other English part of the listing that holds numbers
    from collections import Counter as _C
    other_years = _C()
    for path, _hrs, _t in en_mix:
        if path not in rows and number_share_of(path) > 0:
            for x in cache.setdefault(path, _rows(f"{vol}/{path}")):
                other_years.update(_en_years(_en_values(x.get("text", ""))))
    ytrim, yst = en_top_year_trim(rows, out | trim, flat, other_years, TOP_YEAR_SHARE - TOP_YEAR_MARGIN,
                                  tuple(p for p in trim_parts if p in rows))
    trim |= ytrim
    after = share(out | trim)
    gone = out | trim
    st = {"share_before": before, "share_after_caps": capped, "share_after": after, "band": list(band),
          "in_band": band[0] <= after["share"] <= band[1], "caps_out": why,
          "common_per_video": n_per_video, "value_trim": vst, "dropped_listed": n_listed,
          "relabeled": n_rl, "relabel_dropped": n_rd,
          "top_year_trim": {str(y): v for y, v in yst.items()}, "top_year_trim_rows": len(ytrim),
          "trim_rows": len(trim), "trim_h": round(_h([rows[p][i] for p, i in trim]), 2),
          "parts": {p: {"rows": [len(rows[p]), len(rows[p]) - sum(1 for q, _ in gone if q == p)],
                        "h": [round(_h(rows[p]), 2), round(_h([x for i, x in enumerate(rows[p]) if (p, i) not in gone]), 2)]}
                    for p in real}}
    assert not any(_en_rare(rows[p][i]) for p, i in trim), "the share trim must never take a rare-class crop"
    if write:
        for p in real:
            _write(f"{out_dir or vol}/{en_bcap_path(p, bcap_dir)}", [x for i, x in enumerate(rows[p]) if (p, i) not in gone])
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

def ext_class(sp: dict, r: dict | None = None, lang: str = "") -> str:
    """The data gate's class of a span: with the row and its language exactly the gate's (census_common.gate_class:
    spoken dates / clock times / prices, suffix and scale rules), else the caption-token class (census_common.span_class:
    year 1800-2099, year_early 1100-1799, int_small < 1000, int_big, int_other; every other kind is its own class)."""
    if r is not None and lang:
        return gate_class(r, sp, lang)
    return span_class(sp)[0]


def cap_ext_per_class(rows: list, share: float, rng, min_spans: int = 20, lang: str = "", extra=None,
                      keep_last=None, real_floor: bool = False) -> tuple:
    """Second-recognizer spans (census_common.EXT_BY) <= `share` of each class's spans, as data_gate checks it (per
    class, not per language, in the gate's classes when `lang` is given): rows holding an ext span of a class over the cap
    leave the mix (random order) until every class with >= min_spans spans is at or under it. extra = {class: spans}
    of rows the gate counts beside these (the chosen synthetic rows: never ext, but they make a class reach min_spans
    and they add to its denominator). keep_last(row) -> True: such rows leave only after every other candidate (rows
    holding a thin value, so a later cap does not undo the fill). real_floor: only a class whose REAL spans (without
    `extra`) reach min_spans loses rows (a class over the floor only with the synthetic spans is left to
    trim_synthetic_ext: synthetic rows never displace real ones). Returns (rows, {"rows_out", "classes": {class: [share before, after]},
    "still_over": {class: share}}); still_over lists a class this cap could not bring under `share`."""
    from collections import Counter

    extra = Counter(extra or {})

    def tally(rs):
        n, e = Counter(extra), Counter()
        for x in rs:
            for sp in x.get("spans") or []:
                c = ext_class(sp, x, lang)
                n[c] += 1
                e[c] += ext_span(sp)
        return n, e

    n0, e0 = tally(rows)
    out = set()
    for _ in range(6):
        n, e = tally([x for i, x in enumerate(rows) if i not in out])
        over = [c for c in n if n[c] >= min_spans and e[c] > share * n[c]
                and (not real_floor or n[c] - extra[c] >= min_spans)]
        if not over:
            break
        for c in over:
            idx = [i for i, x in enumerate(rows) if i not in out
                   and any(ext_class(sp, x, lang) == c and ext_span(sp) for sp in x.get("spans") or [])]
            rng.shuffle(idx)
            if keep_last is not None:
                idx.sort(key=lambda i: bool(keep_last(rows[i])))
            ec, nc = e[c], n[c]
            for i in idx:
                if ec <= share * nc:
                    break
                k = [sp for sp in rows[i].get("spans") or [] if ext_class(sp, rows[i], lang) == c]
                ec -= sum(ext_span(sp) for sp in k)
                nc -= len(k)
                out.add(i)
    kept = [x for i, x in enumerate(rows) if i not in out]
    n1, e1 = tally(kept)
    st = {c: [round(e0[c] / n0[c], 3), round(e1[c] / max(n1[c], 1), 3)] for c in n0 if e0[c] and n0[c] >= min_spans}
    still = {c: round(e1[c] / n1[c], 3) for c in n1 if n1[c] >= min_spans and e1[c] > share * n1[c]}
    return kept, {"rows_out": len(out), "classes": st, "still_over": still}


GATE_EXT_SHARE = 0.25  # data_gate: ext share per class (classes with >= 20 spans)
GATE_TOP_YEAR = 0.03  # data_gate: top year value <= 3% of the year spans


def gate_counts(rows: list, syn: list, lang: str) -> dict:
    """The gate's per-class counts over real rows + synthetic rows (data_gate.gate_lang's p3_ + tts_ parts): spans,
    synthetic spans and ext spans per class, year values, the thin-value counts (_span_values), template / group rows."""
    n, e, sy, years, vals, nt, ng = Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    for x in list(rows) + [dict(y, synthetic=True) for y in syn]:
        sps = x.get("spans") or []
        is_syn = _synthetic(x)
        if sps and not is_syn:
            t = template_key(x)
            if t:
                nt[t] += 1
            ng[group_id(x)] += 1
        vals.update(_span_values(x, lang))
        for sp in sps:
            c, v = gate_class_value(x, sp, lang)
            n[c] += 1
            e[c] += ext_span(sp)
            sy[c] += is_syn
            if c == "year" and isinstance(v, int):
                years[v] += 1
    return {"spans": n, "ext": e, "syn": sy, "years": years, "vals": vals, "templates": nt, "groups": ng}


def top_year_share(years) -> float:
    return max(years.values(), default=0) / max(1, sum(years.values()))


def post_build_check(rows: list, syn: list, lang: str, before: dict | None = None) -> dict:
    """The data gate's checks that the build's caps decide, recomputed on a built language (real rows `rows` + chosen
    synthetic rows `syn`) with the gate's own classes, values and keys (census_common): ext share per class, synthetic
    share per class (syn_ok), rows over the template cap, videos / sessions over the group cap, top year value. A
    report: eval/quality/data_gate.py stays the gate. With `before` (gate_counts at an earlier step): "regressions"
    lists what a later step broke (a thin value that was at its fill floor and fell under it; a top year that was within
    the gate's 3% and went over). A class with no real speech held at the rare-class floor (syn_room) is reported under
    "waived_synthetic_floor", not as a failure. Returns {..., "fail": [...], "regressions": [...]}."""
    g = gate_counts(rows, syn, lang)
    cls_n, cls_e, cls_s = g["spans"], g["ext"], g["syn"]
    ext = {c: round(cls_e[c] / cls_n[c], 3) for c in cls_n if cls_n[c] >= 20 and cls_e[c]}
    _, rv = class_values(rows, lang)
    _, sv = class_values(syn, lang)
    syn_bad = {c: round(cls_s[c] / cls_n[c], 3) for c in cls_n
               if not syn_ok(lang, c, cls_s[c], cls_n[c], real_vals=rv.get(c), syn_vals=sv.get(c))}
    # fill-to-bar: a thin class over its share cap, synthetic up to its bar (the gate shows it as "filled to bar")
    filled = {c: {"synthetic": cls_s[c], "real": cls_n[c] - cls_s[c], "bar": class_bar(lang, c)} for c in cls_n
              if filled_to_bar(lang, c, cls_s[c], cls_n[c], rv.get(c), sv.get(c))}
    # classes with no bar and no real speech held at the rare-class floor: the gate's share check fails there
    syn_h = _syn_hours(syn, lang)
    waived = {c: {"synthetic": cls_s[c], "real": cls_n[c] - cls_s[c], "synthetic_h": round(syn_h[c], 3)}
              for c in syn_bad if syn_room(lang, c, cls_s[c], cls_n[c] - cls_s[c], syn_h[c])}
    syn_bad = {c: v for c, v in syn_bad.items() if c not in waived}
    tmpl = sum(c - TEMPLATE_CAP for c in g["templates"].values() if c > TEMPLATE_CAP)
    grp = sum(1 for k, c in g["groups"].items() if not k.startswith("id:") and c > group_cap(lang, k))
    top = top_year_share(g["years"])
    fail = [f"{c} ext share {v}" for c, v in ext.items() if v > GATE_EXT_SHARE]
    fail += [f"{c} synthetic share {v} > {syn_limit(lang, c)}" for c, v in syn_bad.items()]
    fail += [f"rows over the template cap {tmpl}"] if tmpl else []
    fail += [f"groups over the cap {grp}"] if grp else []
    fail += [f"top year {top:.3f}"] if top > GATE_TOP_YEAR else []
    reg = []
    if before is not None:
        reg += [f"{k}{v} {before['vals'][(k, v)]} -> {g['vals'][(k, v)]} (fill floor {_fill_floor((k, v))})"
                for (k, v) in sorted(before["vals"]) if before["vals"][(k, v)] >= _fill_floor((k, v)) > g["vals"][(k, v)]]
        t0 = top_year_share(before["years"])
        if t0 <= GATE_TOP_YEAR < top:
            reg.append(f"top year {t0:.3f} -> {top:.3f}")
    return {"ext_share": ext, "synthetic_over": syn_bad, "waived_synthetic_floor": waived, "filled_to_bar": filled,
            "rows_over_template_cap": tmpl, "groups_over_cap": grp, "top_year": round(top, 4), "fail": fail, "regressions": reg}


def _needs_any(lang: str, add_v: dict, real_v: dict, syn_v: dict, n_dist: Counter) -> bool:
    """Does a row (its {class: Counter(value)}) hold a span that fills a thin value or a new int_big distinct value
    (census_common.value_needs) given the synthetic spans chosen so far?"""
    return any(value_needs(lang, c, v, real_v.get(c), syn_v.get(c), n_distinct=n_dist[c])
               for c, cnt in add_v.items() for v in cnt)


def order_synthetic(tts: list, rng, seed, lang: str) -> tuple:
    """The synthetic pool in selection order: the rows of the other sets shuffled with the build's rng exactly as a
    build without the TTS_REPLACING sets shuffles them, minus the ids that a replacing row (one that is still in the
    pool) supersedes; the replacing rows first, in their own seeded order. Returns (rows, stats)."""
    repl = [x for x in tts if x.get("_set") in TTS_REPLACING]
    rest = [x for x in tts if x.get("_set") not in TTS_REPLACING]
    rng.shuffle(rest)
    gone = {i for x in repl for i in x.get("supersedes") or []}
    n0 = len(rest)
    rest = [x for x in rest if x["id"] not in gone]
    random.Random(f"{seed}-replace-{lang}").shuffle(repl)
    return repl + rest, {"replacing_rows": len(repl), "superseded_ids": len(gone), "superseded_dropped": n0 - len(rest)}


def select_synthetic(tts: list, real: list, lang: str, real_h: dict, enough_h: float = None) -> list:
    """Synthetic rows taken while every class they hold stays within the data gate's synthetic rule
    (census_common.syn_ok: spans per gate class over real + synthetic spans, syn_limit, the exempt floors, fill-to-bar),
    counted in the gate's classes; a class with no bar and no real speech up to the rare-class floor (syn_room).
    Two passes over `tts` in its order: first the rows that fill a thin value (a year / count under its floor, a new
    int_big value under the distinct bar: census_common.value_needs), then the rest up to the share / bar fill, so the
    value fill does not depend on the input order (one non-needed row taken first would lock the needed ones out).
    The real-speech cut-off stays: a non-targeted row of a raw kind with >= enough_h hours of real speech (and not a
    TTS_SHARE_KIND shape) is not used (lesson v12: synthetic speech only FILLS what real speech lacks). Rows carry
    "_targeted". Returns the chosen rows, the value-fill rows first."""
    enough_h = TTS_ENOUGH_H if enough_h is None else enough_h
    real_n, real_v = class_values(real, lang)
    syn_n, syn_h, syn_v, took = Counter(), Counter(), {}, []
    n_dist = Counter({c: sum(1 for n in cnt.values() if n) for c, cnt in real_v.items()})
    pool = []
    for x in tts:
        ks = {sp["kind"] for sp in x.get("spans", [])} or {"?"}
        if not x.get("_targeted") and any(real_h.get(k, 0) >= enough_h and k not in TTS_SHARE_KIND for k in ks):
            continue
        pool.append((x, *class_values([x], lang)))

    def fits(add, add_v, dh) -> bool:
        return all(syn_room(lang, c, syn_n[c] + k, real_n[c], syn_h[c] + dh, real_v.get(c),
                            syn_v.get(c, Counter()) + add_v[c]) for c, k in add.items())

    taken = [False] * len(pool)
    for needed_pass in (True, False):
        for i, (x, add, add_v) in enumerate(pool):
            if taken[i] or (needed_pass and not _needs_any(lang, add_v, real_v, syn_v, n_dist)):
                continue
            dh = x.get("duration", 0.0) / 3600
            if fits(add, add_v, dh):
                taken[i] = True
                took.append(x)
                syn_n.update(add)
                for c in add:
                    syn_h[c] += dh
                    have = syn_v.setdefault(c, Counter())
                    rv = real_v.get(c, {})
                    n_dist[c] += sum(1 for v in add_v[c] if not have.get(v) and not rv.get(v))
                    have.update(add_v[c])
    return took


def trim_synthetic(took: list, real: list, lang: str) -> tuple:
    """Drop chosen synthetic rows holding a class whose synthetic share no longer passes syn_room (real rows a later
    cap removed), until every class passes: of the rows holding a failing class, first the ones that fill no thin
    value (census_common.value_needs with the row removed), last chosen first; a value-fill row only when no other is
    left. Returns (rows, n dropped)."""
    real_n, real_v = class_values(real, lang)
    cv = [class_values([x], lang) for x in took]
    syn_n = Counter()
    syn_v = {}
    for a, v in cv:
        syn_n.update(a)
        for c, cnt in v.items():
            syn_v.setdefault(c, Counter()).update(cnt)
    syn_h = _syn_hours(took, lang)
    keep = [True] * len(took)

    def fills(i) -> bool:  # would removing row i undo a value fill?
        for c, cnt in cv[i][1].items():
            rv, sv = real_v.get(c, {}), syn_v.get(c, Counter())
            nd = len({v for v, n in rv.items() if n} | {v for v, n in sv.items() if n}) if distinct_bar(lang, c) else 0
            for v, k in cnt.items():
                left = Counter({v: sv[v] - k}) if sv[v] - k > 0 else Counter()
                nd1 = nd - (1 if sv[v] - k <= 0 and not rv.get(v) else 0)
                # the value's other synthetic spans stay; only the count of v matters for its floor / distinct check
                if value_needs(lang, c, v, rv, left, n_distinct=nd1):
                    return True
        return False

    order = list(range(len(took) - 1, -1, -1))
    while True:
        bad = {c for c in syn_n if not syn_room(lang, c, syn_n[c], real_n[c], syn_h[c], real_v.get(c), syn_v.get(c))}
        if not bad:
            break
        cand = [i for i in order if keep[i] and bad & set(cv[i][0])]
        if not cand:
            break
        i = next((j for j in cand if not fills(j)), cand[0])
        keep[i] = False
        syn_n.subtract(cv[i][0])
        syn_n = +syn_n
        for c in cv[i][0]:
            syn_h[c] -= took[i].get("duration", 0.0) / 3600
            syn_v[c].subtract(cv[i][1][c])
            syn_v[c] = +syn_v[c]
    return [x for x, k in zip(took, keep) if k], keep.count(False)


def trim_synthetic_ext(took: list, real: list, lang: str, share: float = EXT_SHARE_CLASS,
                       min_spans: int = 20) -> tuple:
    """Synthetic rows never displace real ones (2026-10-17): a class whose REAL spans are under the gate's ext floor
    (min_spans) but whose real + synthetic spans reach it, with its real ext spans over `share` of them, loses
    synthetic rows (last chosen first) until it is under the floor again; the per-class ext cap then never removes a
    real row for it (cap_ext_per_class real_floor). Returns (rows, {class: synthetic rows dropped})."""
    rn, re_ = Counter(), Counter()
    for x in real:
        for sp in x.get("spans") or []:
            c = ext_class(sp, x, lang)
            rn[c] += 1
            re_[c] += ext_span(sp)
    cl = [Counter(ext_class(sp, x, lang) for sp in x.get("spans") or []) for x in took]
    sn = sum(cl, Counter())
    keep = [True] * len(took)
    out = Counter()
    for c in sorted(sn):
        if not (rn[c] < min_spans <= rn[c] + sn[c] and re_[c] > share * (rn[c] + sn[c])):
            continue
        for i in range(len(took) - 1, -1, -1):
            if rn[c] + sn[c] < min_spans:
                break
            if keep[i] and cl[i][c]:
                keep[i] = False
                sn.subtract(cl[i])
                out[c] += 1
    return [x for x, k in zip(took, keep) if k], dict(out)


def trim_synthetic_years(took: list, real: list, lang: str, share: float = TOP_YEAR_SHARE) -> tuple:
    """Synthetic rows never displace real ones (2026-10-17): while the top year value (real + synthetic spans, as the
    gate counts it) is over `share` and synthetic rows hold it, a synthetic row holding it leaves (rows holding only
    that year first, last chosen first). A top year real speech alone puts over the share is left to _cap_top_years.
    Returns (rows, synthetic rows dropped)."""
    ys = [[y for sp in x.get("spans") or [] if (y := _year_of(sp, x, lang))] for x in took]
    cnt = Counter(y for x in real for sp in x.get("spans") or [] if (y := _year_of(sp, x, lang)))
    for y in ys:
        cnt.update(y)
    keep = [True] * len(took)
    n = 0
    while cnt:
        top, c = cnt.most_common(1)[0]
        if c <= share * sum(cnt.values()):
            break
        cand = [i for i in range(len(took) - 1, -1, -1) if keep[i] and top in ys[i]]
        if not cand:
            break
        i = next((j for j in cand if set(ys[j]) == {top}), cand[0])
        keep[i] = False
        cnt.subtract(ys[i])
        cnt = +cnt
        n += 1
    return [x for x, k in zip(took, keep) if k], n


SEAM_FILES = ("seam", "seam_b")  # seam_crops outputs build4 reads: tag "" and the top-up run (tag "_b", seed 2)


def _seam_rows(base: str, files=SEAM_FILES, bare_first: bool = False) -> list:
    """Seam halves of every listed seam_crops output: pair ids namespaced by file (each run numbers its pairs from s0;
    bare_first: the first file's ids stay as they are), a half whose (side, text, parents) another file already gave is
    dropped with its pair."""
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
            out += [dict(h, pair=pid if bare_first and name == files[0] else f"{name}:{pid}") for h in hs]
        seen |= mine
    return out


# long-number seam pairs build4 reads into mix4/seam_long_<lang> (seam_long_h): the first seam_crops --min-words 3 run,
# real top-up runs under new tags (other seeds / the full stream set; ml_p3._seen_pair_keys skips cuts an earlier file
# has), and the synthetic sets' train rows (ml_p3.syn_seam_crops, SYN_SEAM_LONG_FILES: their own parent rule,
# keep_syn_pairs). Synthetic-set windows (windows_syn) join the synthetic windows (TTS_SHARE of win_h). Every synthetic
# pair / window row carries synthetic=True.
# real top-ups: _b the full stream set (seed 1), _c / _d / _e seeds 2 / 3 / 4 (re-stitched streams, new cuts only)
SEAM_LONG_FILES = ("seam_long", "seam_long_b", "seam_long_c", "seam_long_d", "seam_long_e")
# syn2: the second-batch sets (ttsYRs / YRn / LTw / LTr); syn3 / syn4: every set, seeds 2 / 3 (re-stitched streams)
SYN_SEAM_LONG_FILES = ("seam_long_syn", "seam_long_syn2", "seam_long_syn3", "seam_long_syn4")
WIN_FILES = ("windows", "windows_syn", "windows_syn2")


def vb_violations(rows: list, vb) -> list:
    """Rows of the value-spread fill sets (ttsVS1 / ttsRF1, row["_set"]) that carry a span whose "class:value" is in the
    held-back V_b half (ml/ttsVS1/vsplit_<lang>.json): the control half must get no new synthetic fill. Class: year =
    a four-digit 1900-2099, int_big = 1000+, else int_small; only "int" spans (digits of the written text)."""
    out = []
    for x in rows:
        if x.get("_set") not in ("ttsVS1", "ttsRF1"):
            continue
        for sp in x.get("spans") or []:
            t = re.sub(r"\D", "", str(sp.get("text", "")))
            if sp.get("kind") != "int" or not t:
                continue
            n = int(t)
            c = "year" if len(t) == 4 and 1900 <= n <= 2099 else "int_big" if n >= 1000 else "int_small"
            if f"{c}:{n}" in vb:
                out.append((x["id"], f"{c}:{n}"))
    return out


def trim_num_punct(x: dict) -> int:
    """Cut trailing punctuation off the end of every number span of a row's text ("zwanzig." -> "zwanzig"): the spoken number
    does not contain the sentence's period or comma. Returns the number of spans changed (text and spans otherwise untouched;
    punctuation INSIDE a span still stops the build)."""
    out, n = [], 0
    for a, b in x.get("num_chars") or []:
        e = b
        while e > a and x["text"][e - 1] in ",.;:!?":
            e -= 1
        n += e != b
        out.append([a, e])
    if n:
        x["num_chars"] = out
    return n


def keep_syn_pairs(rows: list, bad: set, cap_out: set, took_ids: set) -> tuple[list, dict]:
    """Whole synthetic seam pairs whose every parent is a synthetic row this build trains on (took_ids: the chosen
    synthetic rows; the rule synthetic windows follow), none held out / rejected / ext (bad) or capped out (cap_out).
    A pair missing a half leaves too. Returns (rows, {reason: pairs dropped})."""
    by = {}
    for x in rows:
        by.setdefault(x["pair"], []).append(x)
    out, st = [], Counter()
    for hs in by.values():
        ps = [set(h.get("parents") or []) for h in hs]
        if len(hs) != 2 or not all(ps) or not all(h.get("synthetic") for h in hs):
            st["incomplete_or_not_synthetic"] += 1
        elif any(p & bad for p in ps):
            st["unsafe_parent"] += 1
        elif any(p & cap_out for p in ps):
            st["cap_out"] += 1
        elif not all(p <= took_ids for p in ps):
            st["parent_not_chosen"] += 1
        else:
            out += hs
    return out, dict(st)


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
           seam_files: tuple = SEAM_FILES, mix_dir: str = "mix4", seam_long_h: float = 0.0,
           split_langs: tuple = (), win15_share: float = 0.0, win15_dir: str = "ml/win15b", edge_fix_dir: str = "",
           extra_passes: tuple = (), relabel_rows: str = "", relabel_parts_dir: str = "", relabel_tts: bool = False,
           exclude_passes: tuple = (), seam_extra: tuple = (), seam_long_extra: tuple = (), vsplit_dir: str = "", syn_cap_raise: str = "") -> dict:
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
      Its files (2026-10-17): SEAM_LONG_FILES (real; the first file's pair ids unchanged, top-ups namespaced) plus
      SYN_SEAM_LONG_FILES (synthetic-set pairs: whole pairs whose parents are all chosen synthetic rows of this build,
      keep_syn_pairs); seam_long_h caps each LANGUAGE's part (not a total). Synthetic-set windows (windows_syn) join
      the synthetic windows (WIN_FILES).
    Always (2026-10-17): a window / seam pair with a parent row decided by a second recognizer (EXT_BY), a parent a cap
    took out of training (cap_out: rows_h, the ext allowance and per-class ext cap, the top-year trims, the balance caps,
    dedupe; stats cap_out_dropped) or a parent that is held out NOW (the held-out carve can move after windows were
    built) leaves the mix; a language in `langs` with
    no labelled rows keeps its mix4/d2 digit rows (it gets no spoken-form rows to replace them).
    - plain_share > 0: mix4/plain_<lang>.jsonl, number-free real rows (plain_glob, stock labels) up to number-free /
      (number-free + number) = plain_share per language, look-alike controls first (ml_balance.pick_plain).
    - split_langs (2026-10-18): for these languages (de / nl / it) every written part of the language (p3, tts, win,
      seam, seam_long, plain, pairs, heldout, tts_dev, tts_dev_other) spells compound number words as separate
      morphemes (compound_split: "zweihundertsiebzig" -> "zwei hundert siebzig"). Applied at WRITE time only: every
      selection, cap, dedupe and pair choice above runs on the unsplit text, so the rows are the ones the same build
      without the flag writes; the anchor parts (d2, d2p, fleurs_train_multi) keep their stock labels. Italian:
      "it+" / "it-" force the elided forms split / whole ("it-" reproduces mix4final8), "it" follows
      compound_split.IT_SPLIT_ELIDED (ON); stats["italian_split_elided"] records the form written.
    - win15_share > 0 (2026-10-18): mix4/win15_<lang>.jsonl, 15 s multi-number windows (win15.py, <win15_dir>/<lang>/
      (win15_dir may list several dirs, "ml/win15b,ml/win15sym"; one share over all of them)
      windows.jsonl) whose every parent is a row this build trains on with the same label, none held out / rejected /
      ext; <= win15_share x the language's number hours (rows + synthetic + windows + seam parts), synthetic windows
      <= TTS_SHARE of that target, the year value spread kept (win15.select_for_build).
    - edge_fix_dir (2026-10-18): the real rows' clipped / missing edge words (edge_fix.py records,
      <edge_fix_dir>/<lang>/fixes_*.jsonl): the p3 part is written with those rows' labels trimmed or their edge word
      masked (label + audio cut; edge_fix.apply_fixes, sha-gated, never a row dropped). Applied to the source rows
      before any selection, so the caps, fills and share checks count the fixed labels (a reading masked at an edge
      is a span the fill can replace); win15 parents are checked against the unfixed labels they were cut from;
      consistency pairs skip fixed rows.
    - seam_extra (2026-10-18): more seam_crops outputs for the short seam part, names relative to ml/p3v2/<lang>/ with
      {lang} filled ("../../realx1wb/{lang}/seam" = ml/realx1wb/<lang>/seam.jsonl); read like seam_files.
    - vsplit_dir (2026-10-18): "ml/ttsVS1" holds vsplit_<lang>.json; the chosen ttsVS1 / ttsRF1 rows must carry no V_b value
      (vb_violations; the build stops), so the V_b half stays the unfilled control of the value-spread readout.
    - seam_long_extra (2026-10-18): the same for the long-number seam part (realx1 / dense-scan pair files).
    - exclude_passes (2026-10-18): names ("rows_nlreal1a") of ml/p3v2/<lang>/rows*.jsonl passes the build leaves out (a pass
      still being labelled has no judged marker and would stop the build; its judged copy can come in with extra_passes).
    - extra_passes (2026-10-18): more labelling passes outside ml/p3v2 ("ml/l1conv/{lang}/rows_l1conv.jsonl"), read
      with the language's passes (_one_per_id) and judged like them (judged_<name>.json beside the file).
    - relabel_rows / relabel_parts_dir / relabel_tts (2026-10-18, letter+digit policy, alnum_relabel.py): relabel_rows
      ("ml/alnumA/{lang}/rows_alnumA.jsonl") replaces source rows by id before any selection; relabel_parts_dir
      ("mix4final8_alnumA") holds <lang>/changed_<part>_<lang>.jsonl rows that replace win / seam / seam_long rows by id
      at write time (the same source rows, re-labelled); relabel_tts applies the policy to the chosen synthetic rows at
      write time (alnum_relabel.relabel_row; synthetic rows still spoke the letter names)."""
    import compound_split as CS
    import ml_balance as B

    import census_common as CC

    CC.SYN_CAP_RAISE.clear()
    if syn_cap_raise:
        kinds, _, rest = syn_cap_raise.partition("=")
        share, _, lgs = rest.partition("@")
        CC.SYN_CAP_RAISE.update({(l_, k_): float(share) for l_ in lgs.split(",") for k_ in kinds.split(",") if l_ and k_})
    CS.check_split_langs(split_langs)  # before any work: the restored-vowel Italian form misaligns num_chars
    stats = {"langs": {}, "syn_cap_raise": {f"{l_}.{k_}": v_ for (l_, k_), v_ in sorted(CC.SYN_CAP_RAISE.items())}}
    os.makedirs(f"{vol}/{mix_dir}", exist_ok=True)
    unsplit_targets = set()  # the unsplit texts of split target rows (the anchor-collision check below)

    # Italian form (review 2026-10-18): "it+" = elided forms split too ("venti uno"), "it-" = elided forms kept whole
    # ("ventuno"; the mix4final8 form), plain "it" = compound_split.IT_SPLIT_ELIDED (default ON: always split)
    it_mode = {"it+": True, "it-": False}.get(next((l for l in split_langs if l in ("it+", "it-")), ""), None)
    split_set = {("it" if l in ("it+", "it-") else l) for l in split_langs}
    if "it" in split_set:
        stats["italian_split_elided"] = CS.IT_SPLIT_ELIDED if it_mode is None else it_mode

    def S(rows_, lang_, part, pairs=False):
        """rows_ as written: re-labelled (relabel_parts_dir / relabel_tts, counted), then split (a copy) when lang_ is
        a split language, else unchanged (Italian: it_mode above)."""
        rows_ = relabel_written(rows_, lang_, part)
        if lang_ not in split_set:
            return rows_
        out_, st_ = CS.split_rows(rows_, lang_, pairs=pairs, it_elided=it_mode if lang_ == "it" else None)
        stats.setdefault("split_compounds", {}).setdefault(lang_, {})[part] = st_
        if part in ("p3", "win", "win15", "tts", "seam", "seam_long"):
            unsplit_targets.update(x["text"] for x in rows_)
        return out_
    def relabel_written(rows_, lang_, part):
        n = 0
        if relabel_parts_dir and part in ("win", "seam", "seam_long"):
            ch = {x["id"]: x for x in _rows(f"{vol}/{relabel_parts_dir}/{lang_}/changed_{part}_{lang_}.jsonl")}
            if ch:
                n = sum(x["id"] in ch for x in rows_)
                rows_ = [ch.get(x["id"], x) for x in rows_]
        elif relabel_tts and part == "tts":
            import alnum_relabel as AR

            out_ = []
            for x in rows_:
                y, tm = AR.relabel_row({k: v for k, v in x.items() if k != "synthetic"}, lang_)
                if tm is not None:
                    n += 1
                    x = dict(y, synthetic=x["synthetic"]) if "synthetic" in x else y
                out_.append(x)
            rows_ = out_
        if n:
            stats.setdefault("relabel_written", {}).setdefault(lang_, {})[part] = n
        return rows_

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
    check_heldout_voices(langs)
    for lang in langs:
        # one generator per language: a language's draws never depend on which languages were built before it
        rng = random.Random(f"{seed}-{lang}")
        base = f"{vol}/ml/p3v2/{lang}"
        hp, rp = f"{vol}/ml/p3v2/heldout_{lang}.txt", f"{base}/reject_ids.txt"
        assert os.path.exists(hp), f"{lang}: held-out list missing"  # review H3: never silently skip these
        assert allow_no_judge or os.path.exists(rp), f"{lang}: reject_ids.txt missing (run the judge first)"
        held, held_u = read_held_list(hp)  # "#" comment lines are no ids; "# group" lines add their unit
        rej = {l.strip() for l in open(rp) if l.strip()} if os.path.exists(rp) else set()
        gram_rej = set()  # synthetic rows only (their audio speaks the slip); a later scan adds grammar_reject_ids_<n>
        g_lists = sorted(glob.glob(f"{base}/{GRAMMAR_REJECTS}") + glob.glob(f"{base}/{GRAMMAR_REJECTS[:-4]}_*.txt"))
        for gp in g_lists:
            gram_rej |= read_id_list(gp)
        stats.setdefault("grammar_reject_lists", {})[lang] = [os.path.basename(p_) for p_ in g_lists]
        # validation-only passes (val_*.jsonl, e.g. VoxPopuli dev/test): judged like every pass, never trained; their
        # ids (and below their days / videos) join the held-out set
        val = _rows(f"{base}/val_*.jsonl")
        for pth in sorted(glob.glob(f"{base}/val_*.jsonl")):
            name = os.path.basename(pth)[:-6]
            assert allow_no_judge or os.path.exists(f"{base}/judged_{name}.json"), f"{lang}: {name} was never judged"
        held |= {x["id"] for x in val}
        # every pass must have been judged (marker written by the judge step)
        passes = [p_ for p_ in sorted(glob.glob(f"{base}/rows*.jsonl")) if os.path.basename(p_)[:-6] not in exclude_passes]
        extra = [f"{vol}/{p_.format(lang=lang)}" for p_ in extra_passes if os.path.exists(f"{vol}/{p_.format(lang=lang)}")]
        for pth in passes + extra:
            name = os.path.basename(pth)[:-6]
            assert allow_no_judge or os.path.exists(f"{os.path.dirname(pth)}/judged_{name}.json"), f"{lang}: {name} was never judged"
        rows, dup = _one_per_id(passes + extra)
        stats.setdefault("extra_passes", {})[lang] = {os.path.basename(p_): len(_rows(p_)) for p_ in extra}
        if relabel_rows:  # letter+digit relabel: the re-labelled copy of a source row replaces it (same id)
            rl = {x["id"]: x for x in _rows(f"{vol}/{relabel_rows.format(lang=lang)}")}
            stats.setdefault("relabel_rows", {})[lang] = {"file": len(rl), "replaced": sum(x["id"] in rl for x in rows)}
            rows = [rl.get(x["id"], x) for x in rows]
        unfixed = {}  # id -> the row before its edge fix (win15 parents were cut from it)
        if edge_fix_dir:  # before any selection: the caps, fills and share checks count the fixed labels
            import edge_fix as EF

            fx = {}
            for p_ in sorted(glob.glob(f"{vol}/{edge_fix_dir}/{lang}/fixes_*.jsonl")):
                fx.update(EF.load_fixes(p_))
            by_id0 = {x["id"]: x for x in rows}
            rows, stats.setdefault("edge_fix", {})[lang] = EF.apply_fixes(rows, fx)
            unfixed = {x["id"]: by_id0[x["id"]] for x in rows if "edge_fix" in x}
        stats.setdefault("dup_ids", {})[lang] = dup
        sfx = lambda i: (re.search(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$", str(i)) or [None, None])[1]  # noqa: E731
        held_sfx = {sfx(i) for i in held} - {None}
        held |= {x["id"] for x in rows if sfx(x["id"]) in held_sfx}
        # the held-out unit is the YODAS video or the parliament day (census_common.held_unit, as the carve and the
        # dedupe count it), not the utterance: every row of a held-out video / day stays out too
        held_u |= {held_unit(i) for i in held} - {None}
        held |= {x["id"] for x in rows if held_unit(x["id"]) in held_u}
        n_trim = sum(trim_num_punct(x) for x in rows)  # a sentence period / comma a pass left on the end of a number span
        stats.setdefault("num_punct_trimmed", {})[lang] = n_trim
        for x in rows:
            assert "num_chars" in x, f"{lang}: P3 row without num_chars: {x['id']}"
            assert not WF_LABEL.search(x["text"]), f"{lang}: symbol in label: {x['text'][:80]}"
            for a, b in x["num_chars"]:  # no punctuation inside a spoken number (seam rule: nothing for windows to disagree on)
                assert not re.search(r"[,.;:!?]", x["text"][a:b]), f"{lang}: punctuation inside a number: {x['text'][a:b]!r}"
        ho = [x for x in rows + val if x["id"] in held and x["id"] not in rej]  # review M6: a rejected row is no gold
        tr = [x for x in rows if x["id"] not in held and x["id"] not in rej]
        held_dup = set()  # training ids removed for duplicating a protected row (their windows / pairs go too)
        if dedupe:
            part = {x["id"]: os.path.basename(p)[:-6] for p in sorted(glob.glob(f"{base}/rows*.jsonl")) for x in _rows(p)}
            # this build's synthetic dev rows (held-out voices), read from the sets, not last build's mix4 copy
            tts_dev = [x for tag, _ in TTS_SETS for x in _rows(f"{vol}/ml/{tag}/{lang}/rows.jsonl") if x.get("split") == "dev"]
            prot = [x for x in rows + val if x["id"] in held] + tts_dev
            for pat in protected_globs:
                prot += _rows(f"{vol}/{pat.format(lang=lang)}")
            protect[lang] = prot
            held_units |= held_u
            tr, drep = B.dedupe(tr, prot, part_of=lambda x: part.get(x["id"], "?"))
            held_dup = {i for i, w in drep["removed"].items() if w.startswith("heldout")}
            stats.setdefault("dedupe", {})[lang] = {"rows": drep["counts"]}
        cand_ids = {x["id"] for x in tr}  # every id that could train; the ones the caps below remove go into cap_out
        rng.shuffle(tr)
        # rare classes first (census 2026-10-15: rare cells were thin while plain integers filled the cap)
        tr = sorted(tr, key=lambda x: not any(sp["kind"] not in COMMON_KINDS for sp in x.get("spans", [])))
        # rows a span of which a second recognizer decided (census_common.EXT_BY) have one acoustic check
        # for the value: at most EXT_SHARE of the language's real row hours, never in windows or held-out
        ext = [x for x in tr if ext_row(x)]
        ext_ids = {x["id"] for x in ext}
        tr = [x for x in tr if x["id"] not in ext_ids]
        tr = _take(tr, rows_h)
        # review M7: the ext allowance sits on top of rows_h (rare classes first), so a full cap never starves it
        ext = sorted(ext, key=lambda x: not any(sp["kind"] not in COMMON_KINDS for sp in x.get("spans", [])))
        tr += _take(ext, EXT_SHARE * _h(tr) / (1 - EXT_SHARE))
        # 2026-10-17: the gate checks the ext share PER CLASS (de dec 0.36, int_big 0.28-0.30 under the language cap)
        tr, xst = cap_ext_per_class(tr, EXT_SHARE_CLASS, random.Random(f"{seed}-extcls-{lang}"), lang=lang)
        stats.setdefault("ext_per_class", {})[lang] = xst
        # 2026-10-17: no single year value above TOP_YEAR_SHARE of the language's year spans (2020 was 4-6%);
        # only rows whose every number is that year leave this mix (they stay on disk)
        n_tr = {x["id"] for x in tr}
        tr, ycap = _cap_top_years(tr, TOP_YEAR_SHARE, random.Random(f"{seed}-years-{lang}"), lang=lang)
        year_out = n_tr - {x["id"] for x in tr}  # (their windows / seam pairs leave with every other capped parent: cap_out)
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
            tr, fst = fill_thin(tr, pool, random.Random(f"{seed}-fill-{lang}"), ok=_ext_ok(tr, lang), lang=lang)
            bst["fill"] = fst
            stats.setdefault("balance_caps", {})[lang] = bst
            # the caps shrink the year spans, so the top values creep over 3% (es 2014 at 3.3%, final dry build):
            # the same top-year trim once more, a little under the gate's 3%
            n_tr = {x["id"] for x in tr}
            # the caps only drop rows whose EVERY number is the year; here a row holding it next to other numbers may
            # go too when it has no rare class and no thin value (final build: 2020 stayed at 3.1-3.4% via mixed rows)
            from collections import Counter as _C
            _vc = _C(v for x in tr for v in _span_values(x, lang))
            plain = lambda x: not _rare(x) and not any(_vc[v] <= _thin_floor(v) for v in _span_values(x, lang))  # noqa: E731
            tr, ycap2 = _cap_top_years(tr, TOP_YEAR_SHARE - TOP_YEAR_MARGIN, random.Random(f"{seed}-years2-{lang}"),
                                       mixed_ok=plain, lang=lang)
            year_out |= n_tr - {x["id"] for x in tr}
            stats.setdefault("year_cap_after_caps", {})[lang] = ycap2
        ho = [x for x in ho if not ext_row(x)]
        # windows / seam pairs: no parent decided by a second recognizer (census_common.EXT_BY), no parent held out now
        ext_all = {x["id"] for x in rows if ext_row(x)}
        unsafe = ext_all | held | rej  # rej: a reject that landed after windows2 built the windows
        win = [w for name in WIN_FILES for w in _rows(f"{base}/{name}.jsonl")]  # + synthetic-set windows (windows_syn)
        n_w0 = len(win)
        win = [w for w in win if not set(w.get("parents", [])) & unsafe]
        stats.setdefault("win_unsafe_parent_dropped", {})[lang] = n_w0 - len(win)
        if dedupe:
            n0 = len(win)
            win = [w for w in win if not set(w.get("parents", [])) & held_dup]
            win, wrep = B.dedupe(win, protect[lang], within=False, part_of=lambda x: "win")
            stats["dedupe"][lang]["windows"] = dict(wrep["counts"], parent_heldout_dup=n0 - wrep["counts"]["train_in"])
        for w in win:
            assert "num_tokens" in w and not WF_LABEL.search(w["text"]), f"{lang}: bad window {w['id']}"
            assert not set(w.get("parents", [])) & held, f"{lang}: window {w['id']} built from a held-out row"
            assert not set(w.get("parents", [])) & rej, f"{lang}: window {w['id']} built from a judge-rejected row"
        # synthetic rows (ml_tts): per gate class at most syn_limit of that class's spans (real + synthetic), the data
        # gate's own rule (select_synthetic); exempt classes up to their floor
        tts = []
        sets_read, sets_missing = [], []
        for tag, marker in TTS_SETS:  # tts1 = the first synthetic set, tts2 = the thin-class top-up (2026-10-15)
            (sets_read if os.path.exists(f"{vol}/ml/{tag}/{lang}/rows.jsonl") else sets_missing).append(tag)
            if os.path.exists(f"{vol}/ml/{tag}/{lang}/rows.jsonl"):
                # synthetic rows are judged too (the judge step writes judged_<marker>.json)
                assert allow_no_judge or os.path.exists(f"{base}/judged_{marker}.json"), f"{lang}: {tag} rows never judged"
                tts += [dict(x, _targeted=tag in TTS_TARGETED, _set=tag) for x in _rows(f"{vol}/ml/{tag}/{lang}/rows.jsonl")]
        for x in tts:
            assert "num_chars" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad synthetic row {x.get('id')}"
        # never silent: which synthetic sets this language read and which it has no rows.jsonl for
        stats.setdefault("tts_sets", {})[lang] = {"read": sets_read, "missing": sets_missing}
        print(f"NOTE {lang}: synthetic sets read {sets_read or 'none'}; no rows for {sets_missing}")
        if tts and lang in GRAMMAR_RULE_LANGS and not g_lists:
            # the generators' slip rules cover this language but no grammar scan list exists: rows that speak an
            # old slip would train (run grammar_scan first)
            print(f"WARNING {lang}: {len(tts)} synthetic rows but no {GRAMMAR_REJECTS[:-4]}*.txt (grammar_scan never ran)")
            stats.setdefault("grammar_reject_lists_missing", []).append(lang)
        all_tts_ids = {x["id"] for x in tts}  # every synthetic id, before the reject / dedupe / split filters
        real_h = {}
        for x in tr:
            for sp in x.get("spans", []):
                real_h[sp["kind"]] = real_h.get(sp["kind"], 0) + x["duration"] / 3600
        stats.setdefault("grammar_rejected", {})[lang] = len(all_tts_ids & gram_rej)
        tts = [x for x in tts if x["id"] not in gram_rej]  # a grammar slip in the audio: neither train nor dev
        ho_train = lambda x: is_heldout_voice(lang, x.get("voice")) and x.get("split", "train") != "dev"  # noqa: E731
        stats.setdefault("heldout_voice_train_dropped", {})[lang] = sum(1 for x in tts if ho_train(x))
        tts = [x for x in tts if not ho_train(x)]
        # the held-out synthetic decision set (probe_heldout, score_heldout --set ttsdev): the held-out voice's dev rows
        # only, judge rejects out; a training voice's dev rows go to tts_dev_other (never trained, never that set)
        dev = [x for x in tts if x.get("split") == "dev" and x["id"] not in rej]
        dev_ho = [x for x in dev if is_heldout_voice(lang, x.get("voice"))]
        _write(f"{vol}/{mix_dir}/tts_dev_{lang}.jsonl", S(dev_ho, lang, "tts_dev"))
        _write(f"{vol}/{mix_dir}/tts_dev_other_{lang}.jsonl",
               S([x for x in dev if not is_heldout_voice(lang, x.get("voice"))], lang, "tts_dev_other"))
        stats.setdefault("tts_dev", {})[lang] = {"heldout_voice": len(dev_ho), "other_voices": len(dev) - len(dev_ho),
                                                 "rejected": sum(1 for x in tts if x.get("split") == "dev" and x["id"] in rej)}
        tts = [x for x in tts if x.get("split", "train") == "train" and x["id"] not in rej]  # judged like real rows
        if dedupe:  # a synthetic text equal to a held-out / eval text leaves training too (text keys only)
            tts, trep = B.dedupe(tts, protect[lang], within=False, part_of=lambda x: "tts")
            stats["dedupe"][lang]["tts"] = trep["counts"]
        # B2 (review 2026-10-18): a replacing row the caps / trims refuse must not leave its class short. Its
        # superseded rows come back into the pool and the whole selection reruns from the same state (rng, real rows,
        # year-cap set), without the refused replacing rows; after REPLACE_ROUNDS a refusal is fatal
        tts0, tr0, year_out0, rng0 = tts, list(tr), set(year_out), rng.getstate()
        refused = set()
        for _round in range(REPLACE_ROUNDS + 1):
            tts, tr, year_out = [x for x in tts0 if x["id"] not in refused], list(tr0), set(year_out0)
            rng.setstate(rng0)
            tts, rst = order_synthetic(tts, rng, seed, lang)  # = rng.shuffle(tts) when no TTS_REPLACING set has rows
            stats.setdefault("synthetic_replaced", {})[lang] = rst
            before = gate_counts(tr, [], lang)  # the first-stage caps' result: a later step must not undo it
            took = select_synthetic(tts, tr, lang, real_h)
            before_syn = gate_counts(tr, took, lang)
            # synthetic rows never displace real ones (2026-10-17): a class the synthetic spans alone push over the gate's
            # 20-span ext floor, or a year value they push over the top-year share, loses synthetic rows, not real ones
            took, sx = trim_synthetic_ext(took, tr, lang)
            took, sy = trim_synthetic_years(took, tr, lang)
            stats.setdefault("synthetic_trimmed_not_real", {})[lang] = {"ext_floor": sx, "top_year": sy}
            # the per-class ext cap once more, now that the synthetic rows are chosen: the gate counts them in each class
            # (they make a thin class reach its 20-span floor and add to its denominator). Rows holding a value at its fill
            # floor leave last, so this does not undo fill_thin
            syn_cls = Counter(ext_class(sp, x, lang) for x in took for sp in x.get("spans") or [])
            _vc2 = gate_counts(tr, [], lang)["vals"]
            at_floor = lambda x: any(_vc2[v] <= _fill_floor(v) for v in _span_values(x, lang))  # noqa: E731
            tr, xst2 = cap_ext_per_class(tr, EXT_SHARE_CLASS, random.Random(f"{seed}-extcls2-{lang}"), lang=lang,
                                         extra=syn_cls, keep_last=at_floor, real_floor=True)
            stats.setdefault("ext_per_class_with_synthetic", {})[lang] = xst2
            if xst2["still_over"]:
                print(f"WARNING {lang}: ext share still over {EXT_SHARE_CLASS} after the cap: {xst2['still_over']}")
            if xst2["rows_out"]:
                # the removed real rows raise the other years' shares and the synthetic shares: the top-year trim once
                # more if a year went over (synthetic rows of that year first; real rows only for a year real speech
                # alone puts over), then the synthetic rows a class can no longer hold leave
                took, sy2 = trim_synthetic_years(took, tr, lang)
                stats["synthetic_trimmed_not_real"][lang]["top_year_after_ext2"] = sy2
                if top_year_share(gate_counts(tr, took, lang)["years"]) > GATE_TOP_YEAR:
                    n_tr = {x["id"] for x in tr}
                    tr, ycap3 = _cap_top_years(tr, TOP_YEAR_SHARE - TOP_YEAR_MARGIN, random.Random(f"{seed}-years3-{lang}"),
                                               mixed_ok=lambda x: not _rare(x) and not at_floor(x), lang=lang)
                    year_out |= n_tr - {x["id"] for x in tr}
                    stats.setdefault("year_cap_after_ext2", {})[lang] = ycap3
                took, n_syn_trim = trim_synthetic(took, tr, lang)
                stats.setdefault("synthetic_trimmed_after_ext2", {})[lang] = n_syn_trim
            miss = {x["id"] for x in tts if x.get("_set") in TTS_REPLACING} - {x["id"] for x in took}
            if not miss:
                break
            refused |= miss
            print(f"{lang}: {len(miss)} replacing synthetic rows refused (round {_round}); their superseded rows "
                  f"return, selection reruns: e.g. {sorted(miss)[:5]}")
        assert not miss, f"{lang}: replacing synthetic rows still refused after {REPLACE_ROUNDS} reruns: {sorted(miss)[:5]}"
        stats["synthetic_replaced"][lang].update(refused_restored=len(refused), rounds=_round)
        stats["synthetic_replaced"][lang]["not_admitted"] = 0
        reg = post_build_check(tr, took, lang, before_syn)["regressions"]
        # fail loudly: a later step undid what the caps / fill had reached (the gate would fail where it passed)
        assert not reg, f"{lang}: the build's later steps undid the earlier caps: {reg}"
        # every parent the caps took out of training (rows_h / ext allowance / per-class ext / top-year / balance caps /
        # dedupe), not only the year-capped ones: its windows / seam pairs leave the mix with it
        cap_out = cand_ids - {x["id"] for x in tr}
        n_w1, n_wy = len(win), sum(1 for w in win if set(w.get("parents", [])) & year_out)
        win_h_before = _h(win)
        win = [w for w in win if not set(w.get("parents", [])) & cap_out]  # with or without dedupe
        # windows: dropped by the cap_out rule, of them dropped by the year cap alone (the rule before 10-17)
        cap_drop = {"parents_out": len(cap_out), "parents_out_not_ext": len(cap_out - ext_all),
                    "windows_dropped": n_w1 - len(win), "windows_year_rule": n_wy,
                    "windows_h": [round(win_h_before, 2), round(_h(win), 2)]}
        # a synthetic window may only stitch rows this build trains on: every parent a chosen synthetic row or a kept
        # real row (v12 lesson; a parent the reject / dedupe / class caps removed is not "no synthetic parent")
        took_ids = {x["id"] for x in took}
        ok_parents = took_ids | {x["id"] for x in tr}
        rng.shuffle(win)
        real_win = [w for w in win if not w.get("synthetic")]
        syn_win = [w for w in win if w.get("synthetic") and w.get("parents")
                   and set(w["parents"]) <= ok_parents and set(w["parents"]) & all_tts_ids]
        n_syn_dropped = sum(1 for w in win if w.get("synthetic")) - len(syn_win)
        # seam adversarial cases first: a number cut by an
        # edge, then a number at an edge, then the rest (the shuffle above keeps the order random inside each group)
        seam_first = lambda ws: sorted(ws, key=lambda w: (not w.get("cut_number"), not w.get("edge_number")))  # noqa: E731
        syn_pool_h = _h(syn_win)
        n_wsyn_set = sum(1 for w in syn_win if "_win_syn" in w["id"])  # windows_syn (synthetic-set windows) in the pool
        win = _take(seam_first(real_win), win_h * (1 - TTS_SHARE)) + _take(seam_first(syn_win), win_h * TTS_SHARE)
        stats.setdefault("syn_set_windows", {})[lang] = {
            "pool": n_wsyn_set, "pool_h": round(_h([w for w in syn_win if "_win_syn" in w["id"]]), 3),
            "taken": sum(1 for w in win if "_win_syn" in w["id"]),
            "taken_h": round(_h([w for w in win if "_win_syn" in w["id"]]), 3),
            "syn_pool_h": round(syn_pool_h, 3), "syn_cap_h": round(win_h * TTS_SHARE, 3)}
        # research 2026-10-15 + the TTS-prior lesson: synthetic speech is too clean; 60% of the chosen synthetic rows are
        # degraded (room reverb, noise 5-20 dB SNR, phone band, Opus) with the same Degrader as the real rows
        deg = Degrader(f"{vol}/slr28/RIRS_NOISES", seed + 7)
        os.makedirs(f"{vol}/{mix_dir}/tts_aug/{lang}", exist_ok=True)
        for i, x in enumerate(took):
            if rng.random() < aug_frac(x):
                o = f"{vol}/{mix_dir}/tts_aug/{lang}/{i}_{os.path.basename(x['audio_filepath'])}"
                x["aug"] = deg(x["audio_filepath"], o)
                x["audio_filepath"] = o
        if vsplit_dir and os.path.exists(f"{vol}/{vsplit_dir}/vsplit_{lang}.json"):
            vb_ = set(json.load(open(f"{vol}/{vsplit_dir}/vsplit_{lang}.json"))["V_b"])
            bad_ = vb_violations(took, vb_)
            stats.setdefault("vb_violations", {})[lang] = len(bad_)
            assert not bad_, f"{lang}: {len(bad_)} chosen ttsVS1 / ttsRF1 rows carry a held-back V_b value, e.g. {bad_[:3]}"
        for x in took:
            x.pop("_targeted", None)
            x.pop("_set", None)
        _write(f"{vol}/{mix_dir}/tts_{lang}.jsonl", S(took, lang, "tts"))
        seam = []
        for k, on in (("seam", seam_h > 0), ("plain", plain_share > 0)):  # a part this build does not make must not
            if not on and os.path.exists(f"{vol}/{mix_dir}/{k}_{lang}.jsonl"):  # survive from an earlier build (the v4 mix
                os.remove(f"{vol}/{mix_dir}/{k}_{lang}.jsonl")                  # lists whichever of them exist)
        if seam_h > 0:
            seam = _seam_rows(base, tuple(seam_files) + tuple(p_.format(lang=lang) for p_ in seam_extra))  # ml_p3.seam_crops tags "" and "_b" (dry runs write other tags)
            bad = held | rej | held_dup | ext_all
            seam = [x for x in seam if not set(x.get("parents", [])) & bad and not x.get("synthetic")]
            n_s0, n_sy = len(seam), sum(1 for x in seam if set(x.get("parents", [])) & year_out)
            seam = [x for x in seam if not set(x.get("parents", [])) & cap_out]
            cap_drop.update(seam_dropped=n_s0 - len(seam), seam_year_rule=n_sy)
            for x in seam:
                assert "num_tokens" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad seam crop {x['id']}"
            if dedupe:
                seam, srep = B.dedupe(seam, protect[lang], within=False, part_of=lambda x: "seam")
                stats["dedupe"][lang]["seam"] = srep["counts"]
            seam = B.take_seam(seam, seam_h, rng)
            _write(f"{vol}/{mix_dir}/seam_{lang}.jsonl", S(seam, lang, "seam"))
            NUMBER_PART[f"mix4/seam_{lang}.jsonl"] = round(_h([x for x in seam if any(x["num_tokens"])]) / max(_h(seam), 1e-9), 3)
            pc = {}
            for x in seam:
                pc[f"{x.get('pos_class')}/{x.get('side')}"] = pc.get(f"{x.get('pos_class')}/{x.get('side')}", 0) + 1
            seam_st = {"seam_rows": len(seam), "seam_h": round(_h(seam), 2), "seam_by_pos_side": dict(sorted(pc.items()))}
        seam_long = []
        if os.path.exists(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl") and not seam_long_h:
            os.remove(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl")  # a part this build does not make must not survive
        if seam_long_h > 0 and any(os.path.exists(f"{base}/{n}.jsonl") for n in SEAM_LONG_FILES + SYN_SEAM_LONG_FILES):
            bad = held | rej | held_dup | ext_all
            seam_long = [x for x in _seam_rows(base, SEAM_LONG_FILES + tuple(p_.format(lang=lang) for p_ in seam_long_extra), bare_first=True)
                         if not set(x.get("parents", [])) & bad and not x.get("synthetic")]
            n_l0, n_ly = len(seam_long), sum(1 for x in seam_long if set(x.get("parents", [])) & year_out)
            seam_long = [x for x in seam_long if not set(x.get("parents", [])) & cap_out]
            cap_drop.update(seam_long_dropped=n_l0 - len(seam_long), seam_long_year_rule=n_ly)
            # the long-number pairs come from the same streams as seam.jsonl: a pair whose (parents, cut position)
            # the main seam part already has leaves this part (whole pairs)
            have = {(tuple(x.get("parents", [])), x.get("pos")) for x in seam}
            dup = {x["pair"] for x in seam_long if (tuple(x.get("parents", [])), x.get("pos")) in have}
            seam_long = [x for x in seam_long if x["pair"] not in dup]
            stats.setdefault("seam_long_dup_pairs", {})[lang] = len(dup)
            # synthetic-set pairs: every parent a synthetic row this build trains on (none held out / rejected / grammar
            # rejected / capped out), whole pairs
            syn_long = _seam_rows(base, SYN_SEAM_LONG_FILES)
            n_sl0 = len({x["pair"] for x in syn_long})
            syn_long, sl_drop = keep_syn_pairs(syn_long, bad | gram_rej, cap_out, took_ids)
            seam_long += syn_long
            for x in seam_long:
                assert "num_tokens" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad long seam crop {x['id']}"
            if dedupe:
                seam_long, lrep = B.dedupe(seam_long, protect[lang], within=False, part_of=lambda x: "seam_long")
                stats["dedupe"][lang]["seam_long"] = lrep["counts"]
            n_syn_ok = len({x["pair"] for x in seam_long if x.get("synthetic")})
            seam_long = B.take_seam(seam_long, seam_long_h, rng)
            sl_syn = [x for x in seam_long if x.get("synthetic")]
            stats.setdefault("seam_long_synthetic", {})[lang] = {
                "pairs_in": n_sl0, "dropped": sl_drop, "pairs_eligible": n_syn_ok,
                "pairs_taken": len({x["pair"] for x in sl_syn}), "taken_h": round(_h(sl_syn), 3),
                "real_taken_h": round(_h(seam_long) - _h(sl_syn), 3)}
            if n_sl0 and not sl_syn:
                print(f"WARNING {lang}: synthetic long seam pairs on disk but none kept: {stats['seam_long_synthetic'][lang]}")
            _write(f"{vol}/{mix_dir}/seam_long_{lang}.jsonl", S(seam_long, lang, "seam_long"))
        win15 = []
        if os.path.exists(f"{vol}/{mix_dir}/win15_{lang}.jsonl") and not win15_share:
            os.remove(f"{vol}/{mix_dir}/win15_{lang}.jsonl")  # a part this build does not make must not survive
        if win15_share > 0:
            import win15 as W15

            w15 = [x for d_ in win15_dir.split(",") if d_ for x in _rows(f"{vol}/{d_}/{lang}/windows.jsonl")]  # dirs "a,b"
            if dedupe:
                w15, _ = B.dedupe(w15, protect[lang], within=False, part_of=lambda x: "win15")
            w15_h = win15_share * (_h(tr) + _h(took) + _h(win) + _h(seam) + _h(seam_long))
            win15, w15st = W15.select_for_build(w15, {x["id"]: unfixed.get(x["id"], x) for x in tr + took},
                                                held | rej | held_dup | ext_all | gram_rej, w15_h,
                                                random.Random(f"{seed}-win15-{lang}"), lang=lang)
            for x in win15:
                assert "num_tokens" in x and not WF_LABEL.search(x["text"]), f"{lang}: bad win15 window {x['id']}"
            stats.setdefault("win15", {})[lang] = dict(w15st, target_h=round(w15_h, 3))
            _write(f"{vol}/{mix_dir}/win15_{lang}.jsonl", S(win15, lang, "win15"))
        if plain_share > 0:
            pool = _rows(f"{vol}/{plain_glob.format(lang=lang)}")
            units = {B.held_unit(i) for i in held} - {None}
            pool = [x for x in pool if x.get("lang", lang)[:2] == lang and x["id"] not in held
                    and B.held_unit(x["id"]) not in units and not WF_LABEL.search(x["text"])]
            if dedupe:
                pool, prep = B.dedupe(pool, protect[lang], part_of=lambda x: "plain")
                stats["dedupe"][lang]["plain"] = prep["counts"]
            number_h = _h(tr) + _h(win) + _h(took) + _h(seam) + _h(seam_long) + _h(win15)
            need = B.plain_target_h(number_h, plain_share, anchor_h.get(lang, 0.0))
            plain, pst = B.pick_plain(pool, lang, need, rng)
            _write(f"{vol}/{mix_dir}/plain_{lang}.jsonl", S(plain, lang, "plain"))
            free_h = anchor_h.get(lang, 0.0) + _h(plain)
            plain_st = {"plain_rows": len(plain), "plain_h": round(_h(plain), 2), "plain_pool_rows": len(pool),
                        "plain_pool_h": round(_h(pool), 2), "number_h": round(number_h, 2),
                        "anchor_free_h": round(anchor_h.get(lang, 0.0), 2), "plain_need_h": round(need, 2),
                        "number_free_share": round(free_h / max(free_h + number_h, 1e-9), 3), "plain_pick": pst}
        tr_w = tr  # (edge fixes applied to the source rows above)
        fixed_ids = {x["id"] for x in tr_w if "edge_fix" in x}
        _write(f"{vol}/{mix_dir}/pairs_{lang}.jsonl",
               S(consistency_pairs([x for x in tr if x["id"] not in fixed_ids] + took, win, lang, rng), lang, "pairs",
                 pairs=True))
        _write(f"{vol}/{mix_dir}/p3_{lang}.jsonl", S(tr_w, lang, "p3"))
        _write(f"{vol}/{mix_dir}/win_{lang}.jsonl", S(win, lang, "win"))
        _write(f"{vol}/{mix_dir}/heldout_{lang}.jsonl", S(ho, lang, "heldout"))
        stats.setdefault("cap_out_dropped", {})[lang] = cap_drop
        # the data gate's checks that the build's caps decide, on what this build wrote (report; data_gate is the gate)
        chk = post_build_check(tr, took, lang, before)
        stats.setdefault("post_build_check", {})[lang] = chk
        # the build fails where the gate would, except the accepted waivers: rows over the template cap / videos over
        # the group cap (rare-class and thin-value rows balance_caps keeps; no caps at all without `balance`)
        fatal = [f for f in chk["fail"] if not f.startswith(POST_CHECK_WAIVED)]
        if chk["fail"]:
            print(f"WARNING {lang}: post-build check fails {chk['fail']}")
        assert not fatal, f"{lang}: the built language fails the gate's checks: {fatal}"
        stats["langs"][lang] = {"rows": len(tr_w), "rows_h": round(_h(tr_w), 2), "windows": len(win), "win_h": round(_h(win), 2),
                                "tts_rows": len(took), "tts_h": round(_h(took), 2), "ext_rows": len(set(ext_ids) & {x["id"] for x in tr}),
                                "ext_h": round(_h([x for x in tr if x["id"] in ext_ids]), 2),
                                "syn_windows_outside_caps_dropped": n_syn_dropped,
                                "tts_spans_by_class": dict(sorted(Counter(gate_class(x, sp, lang) for x in took
                                                                          for sp in x.get("spans") or []).items())),
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
               + (("seam_long",) if seam_long_h > 0 else ()) + (("win15",) if win15_share > 0 else ())
               for x in _rows(f"{vol}/{mix_dir}/{k}_{lang}.jsonl")} | unsplit_targets  # + the same rows unsplit
    for part in ("d2p.jsonl", "fleurs_train_multi.jsonl", "d2.jsonl") + tuple(f"plain_{l}.jsonl" for l in langs if plain_share > 0):
        pth = f"{vol}/{mix_dir}/{part}"
        if os.path.exists(pth):
            rows_ = _rows(pth)
            kept = [x for x in rows_ if x["text"] not in targets]
            if len(kept) != len(rows_):
                _write(pth, kept)
            stats.setdefault("target_text_collisions_dropped", {})[part] = len(rows_) - len(kept)
    with open(f"{vol}/{mix_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    return stats


for _l in WAVE1:  # wave-1 spoken-form rows and windows are number-bearing rows (the 35-40% share counts them)
    NUMBER_PART[f"mix4/tts_{_l}.jsonl"] = 1.0
    NUMBER_PART[f"mix4/p3_{_l}.jsonl"] = 1.0
    NUMBER_PART[f"mix4/win_{_l}.jsonl"] = 1.0
    NUMBER_PART[f"mix4/win15_{_l}.jsonl"] = 1.0
NUMBER_PART["mix4/en_win15.jsonl"] = 1.0  # English 15 s windows (modal_app.build_mix4 en_win15)
NUMBER_PART["mix4/d2.jsonl"] = 1.0  # (mix4/d2p and mix4/fleurs_train_multi stay 0: plain anchor rows)
# English parts added 2026-10-17 (modal_app EN4_PARTS / EN_WIN_PARTS and the year-capped mix3b manifests): every row
# carries a number except the cloud set's control sentences and a few seam crops (shares measured on the manifests)
NUMBER_PART.update({"mix3b/d1_ycap.jsonl": 1.0, "mix3b/d10n_ycap.jsonl": 1.0,
                    "d1/en4_ok/yodas_en4_cap30_ycap.jsonl": 1.0, "d1/en4_ok/vox_en_crops_ok_ycap.jsonl": 1.0,
                    "d1/en4_ok/chirp_en_train.jsonl": 0.981, "d1/en4_ok/tts7_en_train.jsonl": 1.0,
                    "d1/en4_ok/tts9_en_train.jsonl": 1.0, "d1/en4_ok/tts10_en_train.jsonl": 1.0,
                    "d1/en4_ok/tts11_en_train.jsonl": 1.0, "d1/en4_ok/yodas_enh_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/tts12_en_train.jsonl": 1.0, "d1/en4_ok/yodas_rare_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/ttsSYM1_en_train.jsonl": 1.0,
                    # 2026-10-18 (listed with extra_parts): the slot-fixed symbol set, its cloned-voice companion and the
                    # caption-scan top-up; shares measured on the manifests (number words in the label)
                    "d1/en4_ok/ttsSYM1_en_train_v2.jsonl": 0.992, "d1/en4_ok/ttsSYM1en2_en_train.jsonl": 0.992,
                    "d1/en4_ok/yodas_en5_train_ycap.jsonl": 0.975,
                    "d1/en4_ok/ttsDW1_en_train.jsonl": 1.0, "d1/en4_ok/ttsLK1_en_train.jsonl": 0.971,
                    "d1/en4_ok/vox_rare_train_ycap.jsonl": 1.0,
                    "d1/en4_ok/win_en.jsonl": 1.0, "d1/en4_ok/seam_en.jsonl": 0.977})
# the balance-capped English copies (en_balance) carry their source part's number share
NUMBER_PART.update({en_bcap_path(p): NUMBER_PART[p] for p in EN_REAL_PARTS})
