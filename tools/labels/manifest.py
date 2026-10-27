#!/usr/bin/env python3
"""Ledger rows -> labeller manifest rows (stdlib). Spans are recomputed with spanfind on the exact label that is scored, so char offsets always match.
manifest row: {id (= row_id), audio_filepath, label, spans:[{ord, char_start, char_end, cands, kind, w_start, w_end, weak}], alt_labels, coded_cell, meta}
The label scored is the current label when it has no digits, else the planned alternative (`text_v2`) when that has none; a row with neither is skipped.
  python3 manifest.py --db ledger.db --out manifest.jsonl [--part P ...] [--limit N] [--alt-origin ORIGIN ...]
Env: LABELS_DB (default for --db)."""
import argparse, json, os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spanfind as SF

DIGIT = re.compile(r"\d")


def scoring_label(row):
    """the all-words label to score: the current label when digit-free, else the planned text_v2 (None when neither is digit-free)"""
    if not DIGIT.search(row["label"]):
        return row["label"]
    if row["text_v2"] and not DIGIT.search(row["text_v2"]):
        return row["text_v2"]
    return None


def manifest_row(L, row_id, label=None, alt_origins=(), max_cands=64, meta=None):
    """alt_origins: candidate-label origins (ledger `candidates.origin`) to score as row-level alternatives against the chosen label"""
    row = L.db.execute("SELECT * FROM rows WHERE row_id=?", (row_id,)).fetchone()
    lab = label if label is not None else scoring_label(row)
    if lab is None:
        return None
    spans = []
    coded = False
    for k, s in enumerate(SF.find_spans(lab)):
        if s.kind == "unparsed":
            continue
        c = SF.candidates(s, max_cands=max_cands)
        spans.append(dict(ord=k, char_start=s.char_start, char_end=s.char_end, cands=c, kind=s.kind, w_start=s.w_start, w_end=s.w_end, weak=s.weak))
        coded = coded or bool(s.prefix or s.suffix)
    alts = []
    if alt_origins:
        alts = [r["label"] for r in L.db.execute("SELECT label FROM candidates WHERE row_id=? AND origin IN (%s)" % ",".join("?" * len(alt_origins)), (row_id,) + tuple(alt_origins))]
    return dict(id=row_id, audio_filepath=row["audio_path"], label=lab, spans=spans, alt_labels=alts, coded_cell=coded,
                meta=dict(meta or {}, part=row["part"], source_class=row["source_class"], duration=row["duration"], key=row["key"]))


def write_manifest(path, rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=os.environ.get("LABELS_DB"), required="LABELS_DB" not in os.environ)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", action="append", help="only these parts")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--alt-origin", action="append", default=[], help="candidate-label origin to score as a row-level alternative")
    ap.add_argument("--max-cands", type=int, default=64)
    a = ap.parse_args(argv)
    import ledger as LG
    L = LG.Ledger(a.db)
    if a.part:
        ids = [r["row_id"] for r in L.db.execute("SELECT row_id FROM rows WHERE part IN (%s) ORDER BY part, idx" % ",".join("?" * len(a.part)), a.part)]
    else:
        ids = [r["row_id"] for r in L.db.execute("SELECT row_id FROM rows ORDER BY part, idx")]
    out = []
    for rid in ids:
        m = manifest_row(L, rid, alt_origins=tuple(a.alt_origin), max_cands=a.max_cands)
        if m is not None and any(len(s["cands"]) > 1 for s in m["spans"]):
            out.append(m)
            if a.limit and len(out) >= a.limit:
                break
    print("manifest rows:", write_manifest(a.out, out))
    L.close()


if __name__ == "__main__":
    main()
