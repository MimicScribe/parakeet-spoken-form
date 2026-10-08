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


def crops(meeting: str, words, dur: float, r: random.Random, n: int):
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
        if not any(t[3] == "W" for t in inside) or overlap_fraction(inside, a, b) > 0.02:
            continue
        text = render(inside)
        # Digit-reading sections carry collapsed word times (400+ characters on 2–3 s of audio);
        # conversational speech is ~13 characters a second.
        if text and len(text) / (b - a) <= 22:
            used.append((a, b))
            yield a, b, text


def main():
    icsi, meetings, out = sys.argv[1:4]
    hours = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
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
        for a, b, text in crops(m, words, dur, r, n=int(per_meeting)):
            if acc >= per_meeting:
                break
            cid = f"{m}_{int(a * 100):07d}_{int((b - a) * 100):04d}"
            sf.write(f"{out}/wav/{cid}.wav", audio[int(a * SR):int(b * SR)], SR, subtype="PCM_16")
            rows.append({"audio_filepath": f"/vol/replay/icsi/wav/{cid}.wav", "duration": round(b - a, 3),
                         "text": text, "id": cid})
            acc += b - a
            k += 1
        print(f"{m}: {k} crops, {acc / 3600:.2f} h", flush=True)
    with open(f"{out}/icsi_train.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"icsi replay: {len(rows)} crops, {sum(x['duration'] for x in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
