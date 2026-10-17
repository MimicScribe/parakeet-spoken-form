"""Re-crop the labelled English rows at QUIET word gaps (2026-10-13 retrain-3 data fix).

The first crops cut at the midpoint between two aligned words; mix_build.trim_edges then dropped every crop whose
edge had no quiet 20 ms frame in its outer 160 ms: 5,376 of 10,672 crops, and nearly all fast speech (native fast
D1 was 0.12 h). Here a crop may only start or end at a gap whose quietest frame is below 0.3x the row's speech level,
so edges are quiet by construction and fast rows keep every gap they do have.

v2 (retrain-3): crops come from the relabel (d1/rows3); decisions are the labeller's (the v1 code-span relaxation
made corrupt labels and is gone); number-onset sub-crops added (D-5).
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


def onset_crops(cuts, bad, tok_span, out, rnd, p=0.3, max_s=8.0, min_s=2.0):
    """D-5: for kept crops, extra 2-8 s sub-crops that START at the quiet cut right before a number span (left-edge /
    u = 0 exposure: 8.9% of retrain-2 D1 crops started with a number word)."""
    extra = []
    for i, j in out:
        for k in range(i, j):
            if not tok_span[k] or (k > 0 and tok_span[k - 1] == tok_span[k]) or cuts[k] is None or k == i:
                continue
            if rnd.random() >= p:
                continue
            want = rnd.uniform(min_s, max_s)
            ends = [e for e in range(k + 1, j + 1) if cuts[e] is not None and (cuts[e] - cuts[k]) * 0.02 <= want
                    and not any(bad[k:e])]
            if ends and (cuts[ends[-1]] - cuts[k]) * 0.02 >= min_s:
                extra.append((k, ends[-1]))
    return extra


def recrop_shard(vol: str, sc, tag: str, keep_audio: bool = True, rows_dir: str = "d1/rows3",
                 out_dir: str = "d1/crops3", wav_root: str = "d1/wav3", man_dir: str = "yodas/en/manifests") -> dict:
    """Quiet-gap crops from the retrain-3 relabel (rows3): labels, spans and decisions come from the rows; nothing is
    re-decided here."""
    import soundfile as sf

    man = {}
    for line in open(f"{vol}/{man_dir}/{tag}.jsonl"):
        r = json.loads(line)
        man[r["utt_id"]] = r
    recs = [json.loads(line) for line in open(f"{vol}/{rows_dir}/{tag}.jsonl")]
    wav_dir = f"{vol}/{wav_root}/{tag}"
    os.makedirs(wav_dir, exist_ok=True)
    rnd = random.Random(tag)
    out, st = [], {"rows": 0, "skipped": 0, "align": 0, "no_cut": 0, "onset": 0}
    for rec in recs:
        uid = rec["utt_id"]
        if rec["kind"] == "plain" or rec.get("drop") or "label" not in rec or uid not in man:
            st["skipped"] += 1
            continue
        r = man[uid]
        sp = L.spans(rec["caption"])
        if len(sp) != len(rec["spans"]):
            st["skipped"] += 1
            continue
        choice = [so["pick"] for so in rec["spans"]]
        label, ranges = L.render_ranges(rec["caption"], sp, choice)
        if label != rec["label"]:
            st["skipped"] += 1
            continue
        decided = [bool(so["decided"]) for so in rec["spans"]]
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        lp = sc.w2v_lp(x)
        toks = label.split()
        starts, pos = [], 0
        for t in toks:
            k = label.find(t, pos)
            starts.append(k)
            pos = k + len(t)
        tok_span, bad = [0] * len(toks), [False] * len(toks)
        for j, (ra, rb) in enumerate(ranges, 1):
            for ti, s0 in enumerate(starts):
                if ra <= s0 < rb or (s0 < ra < s0 + len(toks[ti])):
                    tok_span[ti] = j
                    if not decided[j - 1]:
                        bad[ti] = True
        bad = [b or "\x00" in t for b, t in zip(bad, toks)]
        times = sc.align(lp, label)
        if times is None:
            st["align"] += 1
            continue
        bad = [b or (tm[2] is not None and tm[2] < 0.05) or tm[0] is None for b, tm in zip(bad, times)]
        times = [(tm[0] if tm[0] is not None else 0.0, tm[1] if tm[1] is not None else 0.0) for tm in times]
        cuts = gap_cuts(times, tok_span, x)
        st["rows"] += 1
        base = quiet_crops(cuts, bad, rnd)
        ons = onset_crops(cuts, bad, tok_span, [c for c in base], rnd)
        got = 0
        for ci, (a, b) in enumerate(base + ons):
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
                        "video": r["original_audio_id"], "onset": bool(tok_span[a]) and ci >= len(base)})
            got += 1
            st["onset"] += ci >= len(base)
        st["no_cut"] += got == 0
    os.makedirs(f"{vol}/{out_dir}", exist_ok=True)
    with open(f"{vol}/{out_dir}/{tag}.jsonl", "w") as fo:
        for c in out:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    st.update(shard=tag, crops=len(out), crop_h=round(sum(c["duration"] for c in out) / 3600, 3))
    return st
