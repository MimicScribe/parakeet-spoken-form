"""Re-crop the labelled English rows at QUIET word gaps (2026-10-13 retrain-3 data fix).

The first crops cut at the midpoint between two aligned words; mix_build.trim_edges then dropped every crop whose
edge had no quiet 20 ms frame in its outer 160 ms: 5,376 of 10,672 crops, and nearly all fast speech (native fast
D1 was 0.12 h). Here a crop may only start or end at a gap whose quietest frame is below 0.3x the row's speech level,
so edges are quiet by construction and fast rows keep every gap they do have.

Also relaxes the decide rule for single-reading CODE spans ("F one", "Q one", "RM eight"): the exact-fit check
failed on the letter part (wav2vec2 spells a spoken letter as its name, "EF"), 596 of 769 code spans undecided.
With one candidate there is no reading to choose; the stock-value precheck and the row CER guard still apply.
Multi-reading code spans still need the CTC margin and an exact fit of their number words.
"""

import json
import os
import random

import numpy as np

import labeller as L
from common import NUMBER_WORDS
from label_gpu import W2V, Scorer, fits, norm_ctc

HOP = 320  # 20 ms at 16 kHz


def _rms(x):
    n = len(x) // HOP
    return np.sqrt(np.mean(x[:n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12) if n else np.zeros(0)


def gap_cuts(times, tok_span, x, quiet=0.3):
    """cut[k] = index of a quiet 20 ms frame before token k (k = 0..n), or None when the gap has no quiet frame or
    lies inside a number span."""
    fr = _rms(x)
    if len(fr) < 20:
        return [None] * (len(times) + 1)
    speech = np.median(fr[fr > np.percentile(fr, 30)])
    n, dur = len(times), len(x) / 16000
    cuts = []
    for k in range(n + 1):
        if 0 < k < n and tok_span[k] and tok_span[k] == tok_span[k - 1]:
            cuts.append(None)
            continue
        lo = times[k - 1][1] - 0.08 if k > 0 else max(0.0, times[0][0] - 0.4)
        hi = times[k][0] + 0.08 if k < n else min(dur, times[n - 1][1] + 0.4)
        a, b = max(0, int(lo / 0.02)), min(len(fr), int(np.ceil(hi / 0.02)))
        if b <= a:
            cuts.append(None)
            continue
        j = a + int(np.argmin(fr[a:b]))
        cuts.append(j if fr[j] <= quiet * speech else None)  # quiet FRAME index: a crop keeps the whole frame
    return cuts


def quiet_crops(cuts, bad, rnd, max_s=15.0, min_s=1.0, sub_p=0.2):
    """Token ranges [i, j) between quiet cuts, <= max_s, no bad token inside; greedy longest. Plus preview-shaped
    2-8 s sub-crops (~20%) between quiet cuts inside a kept crop."""
    n = len(bad)
    ok = [k for k in range(n + 1) if cuts[k] is not None]
    out = []
    p = 0
    while p < len(ok) - 1:
        i = ok[p]
        best = None
        for q in range(p + 1, len(ok)):
            j = ok[q]
            if any(bad[i:j]) or (cuts[j] - cuts[i]) * 0.02 > max_s:
                break
            best = q
        if best is None:
            p += 1
            continue
        j = ok[best]
        if (cuts[j] - cuts[i]) * 0.02 >= min_s:
            out.append((i, j))
        p = best
    extra = []
    for i, j in out:
        inner = [k for k in ok if i <= k <= j]
        if rnd.random() >= sub_p or len(inner) < 3:
            continue
        a = rnd.choice(inner[:-1])
        want = rnd.uniform(2, 8)
        ends = [k for k in inner if k > a and (cuts[k] - cuts[a]) * 0.02 <= want]
        if ends and (cuts[ends[-1]] - cuts[a]) * 0.02 >= min_s and (a, ends[-1]) != (i, j):
            extra.append((a, ends[-1]))
    return out + extra


class W2VScorer(Scorer):
    """wav2vec2 only (align + greedy): the re-crop needs no TDT model."""

    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(W2V, cache_dir="/tmp/w2v")
        self.w2v = Wav2Vec2ForCTC.from_pretrained(W2V, cache_dir="/tmp/w2v").cuda().eval()
        self.vocab = self.proc.tokenizer.get_vocab()

    def w2v_lp(self, x):
        with self.torch.no_grad():
            iv = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            return self.w2v(iv).logits.float().log_softmax(-1)[0]


def code_decided(s: dict) -> bool:
    if s["kind"] != "code" or s.get("decided") or not s.get("pre"):
        return bool(s.get("decided"))
    return len(s["cands"]) == 1 or bool(s.get("ok", True) and s.get("num_fit"))


def recrop_shard(vol: str, sc, tag: str, keep_audio: bool = True) -> dict:
    import soundfile as sf

    man = {}
    for line in open(f"{vol}/yodas/en/manifests/{tag}.jsonl"):
        r = json.loads(line)
        man[r["utt_id"]] = r
    recs = {}
    for suf in ("", "_pct"):  # percent-pass labels replace the first pass (Opus B2)
        p = f"{vol}/d1/rows/{tag}{suf}.jsonl"
        if os.path.exists(p):
            for line in open(p):
                rec = json.loads(line)
                if rec["kind"] != "plain":
                    recs[rec["utt_id"]] = rec
    wav_dir = f"{vol}/d1/wav2/{tag}"
    os.makedirs(wav_dir, exist_ok=True)
    rnd = random.Random(tag)
    out, st = [], {"rows": 0, "code_redecided": 0, "no_label": 0, "align": 0, "no_cut": 0}
    for uid, rec in recs.items():
        if rec.get("drop") == "cer" or "label" not in rec or uid not in man:
            st["no_label"] += 1
            continue
        r = man[uid]
        sp = L.spans(rec["caption"])
        if len(sp) != len(rec["spans"]):
            st["no_label"] += 1
            continue
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        lp = sc.w2v_lp(x)
        w2v = norm_ctc(sc.greedy_w2v(lp)).split()
        decided = []
        for s in rec["spans"]:
            if s["kind"] == "code" and not s.get("decided") and len(s["cands"]) > 1:
                nums = [w for w in norm_ctc(s["cands"][s["pick"]]).split() if NUMBER_WORDS.search(w.lower()) or w in ("ONE", "O")]
                s["num_fit"] = bool(nums) and fits(nums, w2v)
            d = code_decided(s)
            st["code_redecided"] += d and not s.get("decided")
            decided.append(d)
        if not any(decided):
            continue
        label = rec["label"]
        toks = label.split()
        starts, pos = [], 0
        for t in toks:
            k = label.find(t, pos)
            starts.append(k)
            pos = k + len(t)
        tok_span, bad = [0] * len(toks), [False] * len(toks)
        shift = 0
        for j, (s, so) in enumerate(zip(sp, rec["spans"]), 1):
            c = so["pick"]
            a = s.start + shift
            b = a + len(s.cands[c])
            shift += len(s.cands[c]) - (s.end - s.start)
            for ti, s0 in enumerate(starts):
                if a <= s0 < b or (s0 < a < s0 + len(toks[ti])):
                    tok_span[ti] = j
                    if not decided[j - 1]:
                        bad[ti] = True
        times = sc.align(lp, label)
        if times is None:
            st["align"] += 1
            continue
        bad = [b or (tm[2] is not None and tm[2] < 0.05) or tm[0] is None for b, tm in zip(bad, times)]
        times = [(tm[0] if tm[0] is not None else 0.0, tm[1] if tm[1] is not None else 0.0) for tm in times]
        cuts = gap_cuts(times, tok_span, x)
        st["rows"] += 1
        got = 0
        for ci, (a, b) in enumerate(quiet_crops(cuts, bad, rnd)):
            if not any(tok_span[a:b]):
                continue
            txt = " ".join(toks[a:b])
            txt = txt[0].upper() + txt[1:]
            seg = x[cuts[a] * HOP:(cuts[b] + 1) * HOP]  # both edge frames are the quiet ones (trim_edges sees them)
            wav = f"{wav_dir}/{uid}_{ci}.wav"
            if keep_audio:
                sf.write(wav, seg, 16000, subtype="PCM_16")
            d = len(seg) / 16000
            out.append({"audio_filepath": wav, "duration": round(d, 3), "text": txt, "src": uid,
                        "wps": round(len(txt.split()) / max(d, 0.1), 2),
                        "kinds": sorted({sp[j - 1].kind for j in set(tok_span[a:b]) if j}),
                        "video": r["original_audio_id"], "quiet": True})
            got += 1
        st["no_cut"] += got == 0
    os.makedirs(f"{vol}/d1/crops2", exist_ok=True)
    with open(f"{vol}/d1/crops2/{tag}.jsonl", "w") as fo:
        for c in out:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    st.update(shard=tag, crops=len(out), crop_h=round(sum(c["duration"] for c in out) / 3600, 3))
    return st
