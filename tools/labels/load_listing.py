#!/usr/bin/env python3
"""Load training listings (jsonl) into a label ledger (ledger.py): rows + provenance, optionally the number spans of each label (stdlib).

Each listing line is a json object with at least `audio_filepath` and the label field (`text` by default); optional `text_v2` (a planned alternative
label, recorded as a candidate), `duration`, `lang`, `video`, `kinds`, `synthetic`, `src`. The part name is the file stem. Nothing about parts is built in:
which part is human-transcribed, caption-derived, a stock-ASR decode or synthetic comes from a rules file.

rules file (json): [{"part_regex": "regex searched in the part name", "source_class": "human_transcript", "dataset": "any text", "real_synth": "real", "chirp": 0}, ...]
First match wins. No match: a row with `"synthetic": true` is `synthetic_script`/`synthetic`, anything else `caption`/`real`.
The tools use these source classes: human_transcript, caption, stock_decode, synthetic_script.

  python3 load_listing.py --db ledger.db --listing 'listings/*.jsonl' [--rules rules.json] [--tag listing] [--spans] [--local-audio-dir DIR] [--force]
Env: LABELS_DB (default for --db)."""
import argparse, glob, json, os, re, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ledger as LG

DIGIT = re.compile(r"\d")


def load_rules(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        rules = json.load(f)
    return [dict(r, _rx=re.compile(r["part_regex"])) for r in rules]


def classify(part, row, rules):
    """-> (source_class, dataset, real_synth, chirp)"""
    for r in rules:
        if r["_rx"].search(part):
            return r.get("source_class", "caption"), r.get("dataset"), r.get("real_synth", "real"), int(r.get("chirp", 0))
    if row.get("synthetic") is True:
        return "synthetic_script", "synthetic", "synthetic", 0
    return "caption", None, "real", 0


def local_audio_index(root):
    """basename -> path for wav / flac files under root"""
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".wav", ".flac")):
                idx.setdefault(fn, os.path.join(dp, fn))
    return idx


def load(db_path, listing_globs, rules=(), tag="listing", label_field="text", with_spans=False, local_audio_dir=None, force=False, limit=None, log=print):
    """build the ledger at db_path from the listing files. Returns the Ledger (open)."""
    if os.path.exists(db_path):
        if not force:
            import sqlite3
            c = sqlite3.connect(db_path)
            try:
                n = c.execute("SELECT COUNT(*) FROM gold WHERE answered_by='human'").fetchone()[0]
            except sqlite3.Error:
                n = 0
            c.close()
            if n:
                raise SystemExit("%s holds %d human answers: refusing to rebuild (use --force after exporting them)" % (db_path, n))
        os.remove(db_path)
        for ext in ("-wal", "-shm"):
            if os.path.exists(db_path + ext):
                os.remove(db_path + ext)
    L = LG.Ledger(db_path)
    aud = local_audio_index(local_audio_dir) if local_audio_dir else {}
    SF = None
    if with_spans:
        import spanfind as SF
    files = sorted({f for g in listing_globs for f in glob.glob(os.path.expanduser(g))})
    for f in files:
        part = os.path.splitext(os.path.basename(f))[0]
        n = 0
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n")
                if not line:
                    continue
                r = json.loads(line)
                path = r["audio_filepath"]
                label = r.get(label_field) or ""
                video = r.get("video")
                cls, ds, rs, chirp = classify(part, r, rules)
                key = LG.stable_key(path, label, video)
                var = LG.variant_of(path)
                local = aud.get(os.path.basename(path))
                kinds = r.get("kinds")
                row_id = "%s#%d" % (part, i)
                L.add_row(row_id=row_id, part=part, idx=i, key=key, variant=var, parent_key=(key if var != "clean" else None), audio_path=path, path_sha1=LG.sha1(path),
                          audio_sha1=LG.file_sha1(local) if local else None, audio_local=local, source_class=cls, dataset=ds, real_synth=rs, tts_narrated=None, chirp=chirp,
                          video=video, lang=r.get("lang"), duration=r.get("duration"), label=label, text_v2=r.get("text_v2"), digits_in_label=int(bool(DIGIT.search(label))),
                          kinds=json.dumps(kinds) if kinds else None, raw_line=line)
                if r.get("text_v2") and r["text_v2"] != label:
                    L.add_candidate(row_id, r["text_v2"], "text_v2")
                if SF is not None and label and not DIGIT.search(label):
                    for k, s in enumerate(SF.find_spans(label)):
                        if s.kind == "unparsed":
                            continue
                        L.add_span(row_id, k, key, s.w_start, s.w_end, s.text, s.value, s.kind, s.reading, SF.candidates(s),
                                   dict(weak=s.weak, prefix=s.prefix, suffix=s.suffix, nvalues=len(s.values)))
                n += 1
                if limit and n >= limit:
                    break
        L.add_part(part, f, tag, n, "")
        L.commit()
        if log:
            log("%s\t%d" % (part, n))
    L.commit()
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=os.environ.get("LABELS_DB"), required="LABELS_DB" not in os.environ)
    ap.add_argument("--listing", action="append", required=True, help="glob of listing jsonl files (repeatable)")
    ap.add_argument("--rules", help="json rules file (part name -> source class)")
    ap.add_argument("--tag", default="listing")
    ap.add_argument("--label-field", default="text")
    ap.add_argument("--spans", action="store_true", help="also store the number spans (spanfind) of every digit-free label")
    ap.add_argument("--local-audio-dir", help="hash audio found under this directory")
    ap.add_argument("--limit", type=int, help="rows per part (smoke runs)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    L = load(a.db, a.listing, load_rules(a.rules), a.tag, a.label_field, a.spans, a.local_audio_dir, a.force, a.limit)
    print(json.dumps(L.stats()["by_class"], indent=1))
    print("loaded in %.1fs" % (time.time() - t0))
    L.close()


if __name__ == "__main__":
    main()
