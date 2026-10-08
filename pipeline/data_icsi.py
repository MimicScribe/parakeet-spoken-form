"""Conversational replay: window-shaped crops of ICSI meetings (CC BY 4.0), human word labels.

The app decodes 15 s windows plus short 2–8 s previews of live meeting audio, so the crops have the
same shapes: they start and end wherever a window would, including on silence or the tail of the
previous speaker. Edges are snapped into gaps between words so no label word is cut in half.

Labels follow the project's spoken-form rule: fillers kept as transcribed ("uh", "um", "mm hmm"),
hyphens dropped, "OK" written "okay", acronyms as joined capitals, quotes and comments dropped.
Crops with more than 2% overlapped speech (cross-talk interleaves the labels), cut-off word
fragments, symbols, digits or impossible label rates are skipped; crops never reuse audio.

Runs locally (the corpus is on disk):
    python data_icsi.py <icsi_dir> <meetings.json> <out_dir> [hours]
writes <out_dir>/wav/*.wav and <out_dir>/icsi_train.jsonl with paths under /vol/replay/icsi/;
upload with `modal volume put parakeet-spoken-form <out_dir> replay/icsi`, then move the manifest
to replay/icsi_train.jsonl.
"""

import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np
import soundfile as sf

from common import WRITTEN_FORM

SR = 16000
ATTACH = {"CM", ".", "APOSS"}  # joined to the previous word
SKIP_CROP = {"SYM", "CD", "TRUNCW"}  # TRUNCW: cut-off fragments ("th", "w"), labels unreliable
DROP = {"LQUOTE", "RQUOTE", "QUOTE"}


def load_words(icsi: str, meeting: str):
    """All speakers' tokens as (start, end, speaker, class, text), times filled from neighbours."""
    out = []
    for path in sorted(glob(f"{icsi}/annotations/ICSI/Words/{meeting}.*.words.xml")):
        spk = path.split(".")[-3]
        toks = []
        for el in ET.parse(path).getroot():
            if not el.tag.endswith("w"):
                continue
            s, e = el.get("starttime") or "", el.get("endtime") or ""
            toks.append([float(s) if s else None, float(e) if e else None, spk, el.get("c"), el.text or "",
                         el.get("t")])
        for i, t in enumerate(toks):  # "we" + "'re" share one span: borrow the missing ends
            if t[0] is None:
                t[0] = next((toks[j][1] for j in range(i - 1, -1, -1) if toks[j][1] is not None), None)
            if t[1] is None:
                t[1] = next((toks[j][0] for j in range(i + 1, len(toks)) if toks[j][0] is not None), t[0])
        out += [t for t in toks if t[0] is not None and t[1] is not None and t[1] >= t[0]]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def render(toks) -> str | None:
    parts = []
    for s, e, spk, c, text, t in toks:
        if c in SKIP_CROP:
            return None
        if c in DROP or not text:
            continue
        if c == "HYPH":  # owner rule: no hyphens ("mm-hmm" -> "mm hmm", "M-one" -> "M one")
            continue
        if text == "OK":
            text = "okay"
        if c in ATTACH or text.startswith("'"):
            if parts:
                parts[-1] += text
            continue
        parts.append(text)
    s = " ".join(parts)
    s = re.sub(r"\s+([,.?!])", r"\1", s).strip()
    s = re.sub(r"^[,.?!\s]+", "", s)
    if not s or WRITTEN_FORM.search(s):
        return None
    return s[0].upper() + s[1:]


def phrases_of(toks, gap: float):
    """Each speaker's tokens grouped into phrases (split at pauses > `gap`), sorted by start time.
    A phrase carries its own punctuation tokens."""
    phrases, last = [], {}
    for t in toks:
        spk = t[2]
        if spk in last and t[0] - last[spk][-1][1] <= gap:
            last[spk].append(t)
        else:
            last[spk] = [t]
            phrases.append(last[spk])
    return sorted(phrases, key=lambda p: p[0][0])


def n_words(p):
    return sum(t[3] == "W" for t in p)


def by_turns(toks):
    """Cross-talk label, both speakers: phrases (pause > 1.5 s splits a phrase) in start order; a short
    backchannel (≤ 2 words) that starts inside another speaker's phrase goes right after that phrase
    instead of splicing it mid-clause (Gemini review 2026-10-08)."""
    out = []  # (phrase tokens incl. attached backchannels, the phrase's own start, its own end, its own words)
    for p in phrases_of(toks, 1.5):
        host = next((q for q in reversed(out) if q[0][0][2] != p[0][2] and q[1] <= p[0][0] <= q[2]), None)
        if host is not None and n_words(p) <= 2 < host[3]:
            host[0].extend(p)  # the host keeps its own span: an attached backchannel does not shrink it (Gemini)
        else:
            out.append((list(p), p[0][0], p[-1][1], n_words(p)))
    return [t for q in out for t in q[0]]


def dominant(toks):
    """Cross-talk label, dominant speaker: the speaker with the most words keeps everything; another
    speaker's phrase is dropped whole (words and punctuation) if any of its words overlaps the
    dominant speaker's words in time, and kept otherwise."""
    from collections import Counter
    n = Counter(t[2] for t in toks if t[3] == "W")
    if not n:
        return toks
    dom = n.most_common(1)[0][0]
    spans = [(t[0], t[1]) for t in toks if t[2] == dom and t[3] == "W"]
    keep = [p for p in phrases_of(toks, 0.5)
            if p[0][2] == dom or not any(t[3] == "W" and s < t[1] and t[0] < e for t in p for s, e in spans)]
    return [t for p in keep for t in p]


def overlap_fraction(toks, a: float, b: float) -> float:
    """Share of speech time inside [a, b] where two or more speakers talk at once."""
    grid = np.zeros(int((b - a) * 100) + 1, dtype=np.int16)
    per = {}
    for s, e, spk, c, *_ in toks:
        if c == "W":
            per.setdefault(spk, np.zeros_like(grid, dtype=bool))[
                max(0, int((s - a) * 100)):max(0, int((e - a) * 100))] = True
    for m in per.values():
        grid += m
    speech = (grid > 0).sum()
    return float((grid > 1).sum()) / speech if speech else 0.0


def snap(toks, x: float, lo: float, hi: float) -> float | None:
    """Move x off any word: the nearest point within [lo, hi] that no word covers."""
    lo, hi = max(lo, x - 1.0), min(hi, x + 1.0)
    covered = [(s, e) for s, e, _, c, *_ in toks if c == "W" and e > lo and s < hi and e - s < 5]
    for d in np.arange(0, max(x - lo, hi - x) + 0.01, 0.02):
        for y in (x - d, x + d):
            if lo <= y <= hi and not any(s < y < e for s, e in covered):
                return float(y)
    return None


def crops(meeting: str, words, dur: float, r: random.Random, n: int, overlap=(0.0, 0.02)):
    used = []
    for _ in range(n):
        length = 15.0 if r.random() < 0.7 else r.uniform(2.0, 8.0)
        a = snap(words, r.uniform(0, dur - length), 0, dur)
        if a is None:
            continue
        b = snap(words, a + length, a + length - 1.0, min(a + length, dur))
        if b is None or b - a < 1.5 or any(a < y and x < b for x, y in used):
            continue
        inside = [t for t in words if t[0] >= a and t[1] <= b]
        if not any(t[3] == "W" for t in inside):
            continue
        ov = overlap_fraction(inside, a, b)
        if not overlap[0] <= ov <= overlap[1]:
            continue
        text = render(by_turns(inside) if ov > 0.02 else inside)
        # Digit-reading sections carry collapsed word times (400+ characters on 2–3 s of audio);
        # conversational speech is ~13 characters a second, two people talking at once up to ~26.
        if text and len(text) / (b - a) <= (30 if ov > 0.02 else 22):
            used.append((a, b))
            yield a, b, (text, render(dominant(inside)) if ov > 0.02 else text)


def main():
    icsi, meetings, out = [a for a in sys.argv[1:] if not a.startswith("--")][:3]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    hours = float(pos[3]) if len(pos) > 3 else 10.0
    # "--overlap": cross-talk crops only (2–40% overlapped speech), labels ordered by speaker phrase.
    ov = (0.02, 0.40) if "--overlap" in sys.argv else (0.0, 0.02)
    sub = "icsi_ovl" if "--overlap" in sys.argv else "icsi"
    meetings = json.load(open(meetings))
    import os
    os.makedirs(f"{out}/wav", exist_ok=True)
    r = random.Random(7)
    per_meeting = hours * 3600 / len(meetings)
    rows, skipped = [], 0
    for m in meetings:
        words = load_words(icsi, m)
        audio, sr = sf.read(f"{icsi}/audio/{m}.wav", dtype="int16")
        assert sr == SR and audio.ndim == 1, (m, sr, audio.shape)
        dur, acc, k = len(audio) / SR, 0.0, 0
        for a, b, (text, text_dom) in crops(m, words, dur, r, n=int(per_meeting), overlap=ov):
            if acc >= per_meeting:
                break
            cid = f"{m}_{int(a * 100):07d}_{int((b - a) * 100):04d}"
            sf.write(f"{out}/wav/{cid}.wav", audio[int(a * SR):int(b * SR)], SR, subtype="PCM_16")
            rows.append({"audio_filepath": f"/vol/replay/{sub}/wav/{cid}.wav", "duration": round(b - a, 3),
                         "text": text, "text_dominant": text_dom, "id": cid})
            acc += b - a
            k += 1
        print(f"{m}: {k} crops, {acc / 3600:.2f} h", flush=True)
    with open(f"{out}/{sub}_train.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    if sub == "icsi_ovl":  # the same crops, labelled with the dominant speaker only
        with open(f"{out}/{sub}_dom_train.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps({**x, "text": x["text_dominant"]}) + "\n")
    print(f"icsi replay: {len(rows)} crops, {sum(x['duration'] for x in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
