#!/usr/bin/env python3
"""Build a stitched seam bed from held-out label rows: real sentences joined into ~1-minute files with short pauses,
so the sliding windows of seam_harness.py cut through numbers at different places.

usage: stitch_bed.py plan --refs DIR --out bed/ [--langs de,es,...] [--per-lang 6] [--target-s 60] [--seed 11]
       stitch_bed.py render --manifest bed/manifest.json --out bed/ [--audio-map /vol=/path/to/audio]
plan    picks rows (seeded shuffle; rows of 2-20 s with at least one scorable span), groups them per language into
        files of about --target-s seconds, and writes bed/manifest.json: {file: [{id, lang, start, end, text (the
        caption), label (the spoken-form text), values {slot: value}, kinds, src_id, audio}]}. Slots come from the
        spans read in caption context (row_values).
render  writes bed/base/<file>.wav (the sentences with their pauses) and bed/sent/<id>.wav (each sentence alone),
        16 kHz mono, from the rows' audio (numpy + soundfile; --audio-map rewrites path prefixes).
Pauses: 0.3 s before the first sentence, then 0.30-0.45 s (seeded) between sentences. The audio is never published:
the manifest names each row's source (`src_id`), and the audio comes from the source datasets.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import value_accuracy as VA  # noqa: E402

SR = 16000
LEAD_S, GAP_S = 0.3, (0.30, 0.45)


def _vstr(v) -> str:
    return str(int(v)) if v == int(v) else format(v.normalize(), "f")


def row_values(row: dict, lang: str) -> dict:
    """{slot: value} for a label row: each span tied to its caption numbers IN CONTEXT (value_accuracy.tie_spans, so
    "1,7 Milliarden" is 1.7e9 and "1.000" is 1000); an ambiguous digit string takes the value the spoken-form label
    also reads, else the largest. Digit runs -> PH, clocks -> T, one number of a currency / decimal / percent span ->
    P / D / PCT, anything else -> N per number. Repeated slot types are numbered N, N2, N3."""
    R, ties = VA.tie_spans(row, lang)
    heard = {x for x in C.SV.reader_reading(row.get("text") or "", lang).vals}
    vals, seen = {}, {}

    def put(base, v):
        seen[base] = seen.get(base, 0) + 1
        vals[base if seen[base] == 1 else f"{base}{seen[base]}"] = v

    for sp, tie in zip(row.get("spans") or [], ties):
        if tie[0] == "slot":
            put(tie[1], tie[2])
        elif tie[0] == "nums":
            vs = []
            for j in tie[1]:
                cands = sorted(R[j].vals)
                pick = next((c for c in cands if _vstr(c) in heard), cands[-1])
                vs.append(_vstr(pick))
            base = {"cur": "P", "dec": "D", "pct": "PCT"}.get(sp.get("kind"), "N") if len(vs) == 1 else "N"
            for v in vs:
                put(base, v)
    return vals


def plan(refs: dict, per_lang: int = 6, target_s: float = 60.0, seed: int = 11) -> dict:
    """refs: {lang: rows}. Deterministic for a seed."""
    rng = random.Random(seed)
    out = {}
    for lang in sorted(refs):
        rows = [r for r in refs[lang] if 2.0 <= float(r.get("duration", 0)) <= 20.0 and row_values(r, lang)]
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
        k = 0
        for fi in range(per_lang):
            t, sents = LEAD_S, []
            while k < len(rows) and t < target_s:
                r = rows[k]
                k += 1
                d = float(r["duration"])
                sents.append({"id": f"{lang}_s{len(out):02d}{len(sents):02d}", "lang": lang, "start": round(t, 3),
                              "end": round(t + d, 3), "text": r["caption"], "label": r["text"],
                              "values": row_values(r, lang), "kinds": [s.get("kind") for s in r["spans"]],
                              "src_id": r["id"], "src": r.get("src"), "audio": r.get("audio_filepath")})
                t += d + rng.uniform(*GAP_S)
            if not sents:
                break
            out[f"bed_{lang}_{fi:02d}"] = sents
    return out


def render(manifest: dict, out: Path, audio_map: dict) -> None:
    import numpy as np
    import soundfile as sf

    def load(p):
        for a, b in audio_map.items():
            if p.startswith(a):
                p = b + p[len(a):]
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == SR, f"{p}: {sr} Hz (resample to 16 kHz first)"
        return x.mean(axis=1)

    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "sent").mkdir(parents=True, exist_ok=True)
    for f, sents in manifest.items():
        total = int(round((sents[-1]["end"] + LEAD_S) * SR))
        buf = np.zeros(total, dtype="float32")
        for s in sents:
            x = load(s["audio"])
            sf.write(out / "sent" / f"{s['id']}.wav", x, SR)
            a = int(round(s["start"] * SR))
            buf[a:a + len(x)] = x[:max(0, total - a)]
        sf.write(out / "base" / f"{f}.wav", buf, SR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--refs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--langs", default="")
    p.add_argument("--per-lang", type=int, default=6)
    p.add_argument("--target-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=11)
    r = sub.add_parser("render")
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--audio-map", action="append", default=[], help="PREFIX=REPLACEMENT")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        langs = {x for x in a.langs.split(",") if x}
        refs = {}
        for f in sorted(a.refs.glob("heldout_*.jsonl")):
            lang = f.stem.split("_", 1)[1]
            if not langs or lang in langs:
                refs[lang] = [x for x in C.read_jsonl(f) if x.get("spans")]
        m = plan(refs, a.per_lang, a.target_s, a.seed)
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=1)
        n = sum(len(v) for v in m.values())
        print(f"{len(m)} files, {n} sentences, {sum(len(s['values']) for v in m.values() for s in v)} slots -> {a.out}/manifest.json")
    else:
        m = json.load(open(a.manifest, encoding="utf-8"))
        render(m, a.out, dict(x.split("=", 1) for x in a.audio_map))


if __name__ == "__main__":
    main()
