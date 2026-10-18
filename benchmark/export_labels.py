#!/usr/bin/env python3
"""Write the publishable label files of the benchmark: span labels and source references, never audio.

usage: export_labels.py --refs DIR --out DIR [--bed MANIFEST.json]
  --refs DIR   held-out label rows heldout_<lang>.jsonl (the build's own format)
  --out DIR    heldout_<lang>.jsonl with only the public fields (below) and, with --bed, seam_bed.json (the stitch plan:
               file -> sentences with times, caption, label, slot values and source ids; no audio paths)
Public fields per row: id, lang, dataset, source_id, crop (start / end seconds inside the source clip, when the row is
a piece of it), duration, caption (the dataset's own transcript), text (the spoken-form label), spans [{text, kind,
reading, canonical}]. Internal fields (audio paths, alignment scores, word times, labelling-pass bookkeeping) are
dropped. Licences follow the source: VoxPopuli CC0, FLEURS CC BY 4.0, YODAS-Granary CC BY 3.0, Common Voice CC0
(see README.md).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DATASET = {"vox": "facebook/voxpopuli", "voxval": "facebook/voxpopuli", "fleurs": "google/fleurs",
           "yodas": "espnet/yodas-granary", "cv22": "common-voice"}
SPAN_FIELDS = ("text", "kind", "reading", "canonical")


def public_row(r: dict) -> dict:
    crop = None
    if r.get("t0") is not None and r.get("t1") is not None:
        crop = [r["t0"], r["t1"]]
    elif r.get("piece"):
        crop = list(r["piece"])
    return {"id": r["id"], "lang": r["lang"], "dataset": DATASET.get(r.get("src"), r.get("src")),
            "source_id": r.get("parent") or r["id"], "crop": crop, "duration": r.get("duration"),
            "caption": r["caption"], "text": r["text"],
            "spans": [{k: sp.get(k) for k in SPAN_FIELDS} for sp in r.get("spans") or []]}


def public_bed(m: dict) -> dict:
    keep = ("id", "lang", "start", "end", "text", "label", "values", "kinds", "src_id")
    return {f: [{k: s[k] for k in keep if k in s} for s in sents] for f, sents in m.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bed", type=Path)
    a = ap.parse_args(argv)
    for p in sorted(a.refs.glob("heldout_*.jsonl")):
        rows = [public_row(r) for r in C.read_jsonl(p) if r.get("spans")]
        if rows:
            C.write_jsonl(a.out / p.name, rows)
            print(f"{p.name}: {len(rows)} rows, {sum(len(r['spans']) for r in rows)} spans")
    if a.bed:
        with open(a.out / "seam_bed.json", "w", encoding="utf-8") as f:
            json.dump(public_bed(json.load(open(a.bed, encoding="utf-8"))), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
