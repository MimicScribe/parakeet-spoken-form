#!/usr/bin/env python3
"""Label ledger: one SQLite file as the source of truth for training labels (stdlib only).

One row per listed training row, an append-only EVIDENCE table and an append-only DECISIONS table.
A mix part is `base listing line + latest decision`: `export` rewrites each part's jsonl from the stored raw line (byte-identical when no decision applies).

Tables
  parts      one row per listing part (file name, origin file, listing tag, row count)
  rows       row_id 'part#idx' (PK), key (stable parent key), variant clean|aug|spd, audio path (+ sha1 of the path, + sha1 of the audio when a local copy exists),
             source_class (free text; the tools use human_transcript / caption / stock_decode / synthetic_script), real_synth, flags (tts_narrated, chirp),
             label (current), text_v2 (a planned alternative label), raw (zlib of the original jsonl line)
  spans      one row per number span found in the CURRENT label: row_id, ord, span_key, word range, text, value, kind, reading, candidates (JSON)
  candidates candidate LABELS (whole-row alternatives) with their origin
  evidence   APPEND-ONLY: row_id, span_ord, source, value, form, score, margin, ts, tool, sha, payload
  decisions  APPEND-ONLY: row_id, span_ord, action keep|relabel|re-cut|drop|unconfirmed, reason, decided_by rule|referee|human, new_label, ts
  gold       APPEND-ONLY known answers (human verdicts) for calibrating referees, judges and the margin
  queues / queue_items   listening queues a review app reads; record_verdict() writes the answers back as gold + decisions
Views: latest_decision, row_state, review_page (rows whose latest decision is `unconfirmed`).
Part names, source classes and queue names are inputs, never hard-coded. See load_listing.py for filling the ledger.
API: Ledger(path).add_evidence(...), .add_decision(...), .export_mix(outdir), .stats() ; CLI: python3 ledger.py --db FILE --help
"""
import argparse, hashlib, json, os, re, sqlite3, sys, time, zlib

ACTIONS = ("keep", "relabel", "re-cut", "drop", "unconfirmed")
DECIDERS = ("rule", "referee", "human")
SOURCE_CLASSES = ("synthetic_script", "human_transcript", "caption", "stock_decode")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS parts(
  part TEXT PRIMARY KEY, src_file TEXT, listing TEXT, n_rows INTEGER, notes TEXT);
CREATE TABLE IF NOT EXISTS rows(
  row_id TEXT PRIMARY KEY, part TEXT NOT NULL, idx INTEGER NOT NULL,
  key TEXT NOT NULL, variant TEXT NOT NULL, parent_key TEXT,
  audio_path TEXT NOT NULL, path_sha1 TEXT NOT NULL, audio_sha1 TEXT, audio_local TEXT,
  source_class TEXT NOT NULL, dataset TEXT, real_synth TEXT NOT NULL, tts_narrated INTEGER, chirp INTEGER DEFAULT 0,
  video TEXT, lang TEXT, duration REAL, label TEXT NOT NULL, text_v2 TEXT, digits_in_label INTEGER DEFAULT 0,
  kinds TEXT, raw BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS rows_part ON rows(part);
CREATE INDEX IF NOT EXISTS rows_key ON rows(key);
CREATE INDEX IF NOT EXISTS rows_audio ON rows(audio_path);
CREATE INDEX IF NOT EXISTS rows_class ON rows(source_class);
CREATE TABLE IF NOT EXISTS spans(
  row_id TEXT NOT NULL, ord INTEGER NOT NULL, span_key TEXT NOT NULL,
  w_start INTEGER, w_end INTEGER, text TEXT, value TEXT, kind TEXT, reading TEXT, cands TEXT, flags TEXT,
  PRIMARY KEY(row_id, ord));
CREATE INDEX IF NOT EXISTS spans_key ON spans(span_key);
CREATE TABLE IF NOT EXISTS candidates(
  id INTEGER PRIMARY KEY AUTOINCREMENT, row_id TEXT NOT NULL, label TEXT NOT NULL, origin TEXT NOT NULL, ts TEXT NOT NULL,
  UNIQUE(row_id, label, origin));
CREATE TABLE IF NOT EXISTS evidence(
  id INTEGER PRIMARY KEY AUTOINCREMENT, row_id TEXT NOT NULL, span_ord INTEGER, source TEXT NOT NULL,
  value TEXT, form TEXT, score REAL, margin REAL, ts TEXT NOT NULL, tool TEXT NOT NULL, sha TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ev_row ON evidence(row_id);
CREATE INDEX IF NOT EXISTS ev_src ON evidence(source);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, row_id TEXT NOT NULL, span_ord INTEGER,
  action TEXT NOT NULL CHECK(action IN ('keep','relabel','re-cut','drop','unconfirmed')),
  reason TEXT NOT NULL, decided_by TEXT NOT NULL CHECK(decided_by IN ('rule','referee','human')),
  new_label TEXT, evidence_ids TEXT, ts TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS dec_row ON decisions(row_id);
CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT,'evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT,'evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions BEGIN SELECT RAISE(ABORT,'decisions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions BEGIN SELECT RAISE(ABORT,'decisions are append-only'); END;
CREATE TABLE IF NOT EXISTS gold(
  id INTEGER PRIMARY KEY AUTOINCREMENT, bed TEXT NOT NULL, item TEXT NOT NULL, row_id TEXT, question TEXT, answer TEXT NOT NULL,
  answered_by TEXT NOT NULL DEFAULT 'human', ts TEXT NOT NULL, source_file TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS gold_bed ON gold(bed);
CREATE TRIGGER IF NOT EXISTS gold_no_update BEFORE UPDATE ON gold BEGIN SELECT RAISE(ABORT,'gold is append-only'); END;
CREATE TRIGGER IF NOT EXISTS gold_no_delete BEFORE DELETE ON gold BEGIN SELECT RAISE(ABORT,'gold is append-only'); END;
CREATE TABLE IF NOT EXISTS queues(
  name TEXT PRIMARY KEY, title TEXT, question TEXT, button_labels TEXT, scope TEXT DEFAULT 'label', test_bed INTEGER DEFAULT 0, created_ts TEXT);
CREATE TABLE IF NOT EXISTS queue_items(
  id INTEGER PRIMARY KEY AUTOINCREMENT, queue TEXT NOT NULL, pos INTEGER NOT NULL, item_id TEXT NOT NULL, row_id TEXT, audio_path TEXT, audio_span TEXT,
  label TEXT, span TEXT, candidates TEXT, evidence TEXT, context TEXT, UNIQUE(queue, item_id));
CREATE INDEX IF NOT EXISTS qi_queue ON queue_items(queue, pos);
CREATE TABLE IF NOT EXISTS listing_runs(
  run TEXT PRIMARY KEY, source_log TEXT, seed INTEGER, drop_nonen_span INTEGER, n_entries INTEGER, log_hours REAL, ts TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS listing(
  run TEXT NOT NULL, entry INTEGER NOT NULL, part TEXT, vol_path TEXT NOT NULL, cap_hours REAL, tag TEXT, weight INTEGER, repeat_no INTEGER, log_rows INTEGER, log_hours REAL,
  PRIMARY KEY(run, entry));
CREATE INDEX IF NOT EXISTS listing_part ON listing(part);
CREATE VIEW IF NOT EXISTS latest_decision AS
  SELECT d.* FROM decisions d JOIN (SELECT row_id, span_ord IS NULL AS rowlevel, MAX(id) AS mid FROM decisions GROUP BY row_id, rowlevel) m ON d.id = m.mid WHERE d.span_ord IS NULL;
CREATE VIEW IF NOT EXISTS row_state AS
  SELECT r.row_id, r.part, r.key, r.source_class, r.real_synth, r.label,
         COALESCE(l.action,'keep') AS action, l.reason, l.decided_by, l.new_label,
         CASE WHEN l.action='relabel' AND l.new_label IS NOT NULL THEN l.new_label ELSE r.label END AS final_label
  FROM rows r LEFT JOIN latest_decision l ON l.row_id = r.row_id;
CREATE VIEW IF NOT EXISTS review_page AS
  SELECT s.row_id, s.part, s.source_class, s.label, s.reason, s.decided_by FROM row_state s WHERE s.action='unconfirmed';
"""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def file_sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------- stable keys
_NORM = re.compile(r"[^a-z0-9]+")


def norm_text(t):
    return _NORM.sub(" ", (t or "").lower()).strip()


def stem_of(path):
    b = os.path.basename(path)
    return b.rsplit(".", 1)[0] if "." in b else b


def variant_of(path, aug_suffix="__aug", speed_prefix="spd_"):
    """clean | aug | spd (speed copy). aug = stem ends with aug_suffix or the file lives under /aug/. The suffix / prefix are the file naming of your augmentation step."""
    s = stem_of(path)
    if s.startswith(speed_prefix):
        return "spd"
    if s.endswith(aug_suffix) or "/aug/" in path:
        return "aug"
    return "clean"


def stable_key(path, text="", video=None, aug_suffix="__aug", speed_prefix="spd_"):
    """audio stem minus the augmentation suffix; speed copies (no parent id in the name) key on video + normalised text."""
    s = stem_of(path)
    if s.startswith(speed_prefix):
        return "vt:%s|%s" % (video or "", sha1(norm_text(text))[:16]) if video else "tx:" + sha1(norm_text(text))[:16]
    if s.endswith(aug_suffix):
        s = s[: -len(aug_suffix)]
    return "a:" + s


def span_key(key, ordinal):
    return "%s#s%d" % (key, ordinal)


# ---------------------------------------------------------------------------- non-English parts (option A: stock labels, untouched)
def part_name_of(vol_path):
    """volume path -> part name: 'ml/anchor/stockDEF/p3/de/digit.jsonl' -> 'ml__anchor__stockDEF__p3__de__digit' (the local mirror's file name, minus .jsonl)"""
    return (vol_path[:-6] if vol_path.endswith(".jsonl") else vol_path).replace("/", "__")


def english_part_for(vol_path, known):
    """volume path of an English listing entry -> the part name build_ledger.py gave it (file stem for the ml/lv3*, ml/en_v3* manifests, else the path with '/' -> '_'); None when not a known part"""
    stem = os.path.basename(vol_path)[:-6]
    flat = (vol_path[:-6]).replace("/", "_")
    for c in (stem, flat):
        if c in known:
            return c
    return None


def lang_of(r):
    """primary language subtag of a row's `lang` ('pt_br' -> 'pt', 'es_419' -> 'es', 'cs_cz' -> 'cs'); None when absent"""
    l = r.get("lang")
    return l.split("_")[0].lower() if isinstance(l, str) and l else None


def classify_nonen(vol_path, r):
    """non-English row -> (source_class, dataset, real_synth, tts_narrated, chirp). Provenance only, NO relabel decision (option A keeps every stock label as is).
    - TTS audio (src tts / synthetic / not is_real / origin tts*): synthetic_script (the label is the narrated script), tts_narrated 1
    - stock-label anchors: stockDEF parts, plain_{de,es,fr}, rows marked ancdef/g1 stockmatch|stock_harvest|stock_text, and ancRH2-style rows whose `text` differs from their `caption`
      (the label is the stock model's decode of the audio): stock_decode
    - everything else keeps its source's own transcript: fleurs / cv22 -> human_transcript, yodas / vox / other -> caption"""
    src = r.get("src")
    src_s = src if isinstance(src, str) else None
    origin = str(r.get("origin") or "")
    synth = r.get("synthetic") is True or src_s == "tts" or r.get("is_real") is False or origin.startswith("tts")
    ds = src_s or origin or "stitched"
    chirp = int("chirp" in (vol_path + origin).lower())
    if synth:
        return "synthetic_script", "tts_" + (origin or "unknown"), "synthetic", 1, chirp
    stock = ("/stockDEF/" in vol_path or "mix5gated1022h/plain_" in vol_path or r.get("ancdef") in ("stockmatch", "mix5_plain_stock_text")
             or r.get("g1") in ("stockmatch", "d2_stock_harvest") or ("caption" in r and r.get("caption") != r.get("text") and "anchor" in r))
    if stock:
        return "stock_decode", ds, "real", 0, chirp
    if src_s in ("fleurs", "cv22"):
        return "human_transcript", ds, "real", 0, chirp
    return "caption", ds, "real", 0, chirp


# ---------------------------------------------------------------------------- the ledger
class Ledger:
    def __init__(self, path, check_same_thread=True):
        self.path = path
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=check_same_thread)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """additive-only: columns added to shipped tables after the English build (never drop / rename)"""
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(parts)")}
        for col, typ in (("vol_path", "TEXT"), ("lang", "TEXT"), ("langs", "TEXT"), ("hours", "REAL"), ("src_sha256", "TEXT"), ("n_bytes", "INTEGER")):
            if col not in have:
                self.db.execute("ALTER TABLE parts ADD COLUMN %s %s" % (col, typ))
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()

    def commit(self):
        self.db.commit()

    # -- writes
    def add_part(self, part, src_file, listing, n_rows, notes=""):
        self.db.execute("INSERT INTO parts(part,src_file,listing,n_rows,notes) VALUES(?,?,?,?,?) ON CONFLICT(part) DO UPDATE SET "
                        "src_file=excluded.src_file, listing=excluded.listing, n_rows=excluded.n_rows, notes=excluded.notes", (part, src_file, listing, n_rows, notes))

    def add_row(self, **r):
        raw = r.pop("raw_line")
        cols = ["row_id", "part", "idx", "key", "variant", "parent_key", "audio_path", "path_sha1", "audio_sha1", "audio_local", "source_class",
                "dataset", "real_synth", "tts_narrated", "chirp", "video", "lang", "duration", "label", "text_v2", "digits_in_label", "kinds"]
        vals = [r.get(c) for c in cols] + [zlib.compress(raw.encode("utf-8"), 6)]
        self.db.execute("INSERT INTO rows(%s,raw) VALUES(%s)" % (",".join(cols), ",".join("?" * (len(cols) + 1))), vals)

    def add_span(self, row_id, ordinal, key, w_start, w_end, text, value, kind, reading, cands, flags=None):
        self.db.execute("INSERT OR REPLACE INTO spans VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (row_id, ordinal, span_key(key, ordinal), w_start, w_end, text, value, kind, reading,
                         json.dumps(cands, ensure_ascii=False), json.dumps(flags or {}, ensure_ascii=False)))

    def add_candidate(self, row_id, label, origin):
        self.db.execute("INSERT OR IGNORE INTO candidates(row_id,label,origin,ts) VALUES(?,?,?,?)", (row_id, label, origin, now()))

    def add_evidence(self, row_id, source, tool, value=None, form=None, score=None, margin=None, span_ord=None, sha=None, payload=None, ts=None):
        cur = self.db.execute("INSERT INTO evidence(row_id,span_ord,source,value,form,score,margin,ts,tool,sha,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (row_id, span_ord, source, None if value is None else str(value), form, score, margin, ts or now(), tool, sha,
                               None if payload is None else json.dumps(payload, ensure_ascii=False)))
        return cur.lastrowid

    def add_decision(self, row_id, action, reason, decided_by, new_label=None, span_ord=None, evidence_ids=None, ts=None):
        if action not in ACTIONS:
            raise ValueError("action %r not in %r" % (action, ACTIONS))
        if decided_by not in DECIDERS:
            raise ValueError("decided_by %r not in %r" % (decided_by, DECIDERS))
        if action == "relabel" and not new_label:
            raise ValueError("relabel needs new_label")
        cur = self.db.execute("INSERT INTO decisions(row_id,span_ord,action,reason,decided_by,new_label,evidence_ids,ts) VALUES(?,?,?,?,?,?,?,?)",
                              (row_id, span_ord, action, reason, decided_by, new_label, json.dumps(evidence_ids) if evidence_ids else None, ts or now()))
        return cur.lastrowid

    def add_gold(self, bed, item, answer, row_id=None, question=None, answered_by="human", source_file=None, payload=None, ts=None):
        """known answers (human verdicts) for calibration of referees / judges / the margin. bed = a named set of items: a training-row set (row_id given) or a held-out TEST set (row_id NULL)."""
        cur = self.db.execute("INSERT INTO gold(bed,item,row_id,question,answer,answered_by,ts,source_file,payload) VALUES(?,?,?,?,?,?,?,?,?)",
                              (bed, item, row_id, question, answer, answered_by, ts or now(), source_file, None if payload is None else json.dumps(payload, ensure_ascii=False)))
        return cur.lastrowid

    # -- listening queues (the listening app reads these and writes verdicts as gold + decisions)
    def add_queue(self, name, title, question, button_labels=None, scope="label", test_bed=False):
        self.db.execute("INSERT OR REPLACE INTO queues VALUES(?,?,?,?,?,?,?)", (name, title, question, json.dumps(button_labels or {}), scope, int(bool(test_bed)), now()))

    def add_queue_item(self, queue, item_id, label, audio_path=None, row_id=None, candidates=None, evidence=None, span=None, audio_span=None, context=None):
        pos = self.db.execute("SELECT COALESCE(MAX(pos),-1)+1 FROM queue_items WHERE queue=?", (queue,)).fetchone()[0]
        self.db.execute("INSERT OR REPLACE INTO queue_items(queue,pos,item_id,row_id,audio_path,audio_span,label,span,candidates,evidence,context) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (queue, pos, item_id, row_id, audio_path, json.dumps(audio_span) if audio_span else None, label, json.dumps(span) if span else None,
                         json.dumps(candidates or [], ensure_ascii=False), json.dumps(evidence or [], ensure_ascii=False), json.dumps(context or {}, ensure_ascii=False)))

    def latest_verdicts(self, queue):
        """item_id -> latest human gold row of this queue (answer UNDO counts as pending)"""
        out = {}
        for g in self.db.execute("SELECT * FROM gold WHERE bed=? AND answered_by='human' ORDER BY id", (queue,)):
            out[g["item"]] = dict(g)
        return out

    def queue_summary(self):
        res = []
        for q in self.db.execute("SELECT * FROM queues ORDER BY name"):
            lv = self.latest_verdicts(q["name"])
            total = self.db.execute("SELECT COUNT(*) FROM queue_items WHERE queue=?", (q["name"],)).fetchone()[0]
            done = sum(1 for v in lv.values() if v["answer"] != "UNDO")
            res.append(dict(name=q["name"], title=q["title"], total=total, done=done, test_bed=bool(q["test_bed"])))
        return res

    def record_verdict(self, queue, item_id, verdict, heard=None, note=None, allow_digits=False, tool="listen"):
        """CORRECT / WRONG (+ heard) / UNSURE / UNDO -> a gold row (calibration) and, when the item belongs to a training row, a human DECISION.
        WRONG with a typed reading relabels (span scope: only the span; label scope: the whole label) unless the row's label changed since the queue was built
        or the typed text has digits (refused unless allow_digits: spoken-form labels are words only)."""
        verdict = verdict.upper()
        if verdict not in ("CORRECT", "WRONG", "UNSURE", "UNDO"):
            raise ValueError("verdict %r" % verdict)
        q = self.db.execute("SELECT * FROM queues WHERE name=?", (queue,)).fetchone()
        it = self.db.execute("SELECT * FROM queue_items WHERE queue=? AND item_id=?", (queue, item_id)).fetchone()
        if not q or not it:
            raise KeyError("no such queue item")
        heard = (heard or "").strip()
        payload = dict(label=it["label"], span=json.loads(it["span"]) if it["span"] else None, candidates=json.loads(it["candidates"] or "[]"), evidence=json.loads(it["evidence"] or "[]"),
                       context=json.loads(it["context"] or "{}"), heard=heard or None, note=note, scope=q["scope"], TEST_BED=bool(q["test_bed"]))
        gid = self.add_gold(queue, item_id, verdict, row_id=it["row_id"], question=q["question"], source_file=tool, payload=payload)
        res = dict(gold_id=gid, decision_id=None, action=None)
        if verdict == "UNDO" or not it["row_id"] or q["test_bed"] or q["scope"] == "gold_only":
            return res                       # gold only: test beds, undo, and queues whose question does not map to keep / relabel (listening sets, calibration)
        row = self.db.execute("SELECT label FROM rows WHERE row_id=?", (it["row_id"],)).fetchone()
        if verdict == "CORRECT":
            act, why, new = "keep", "human listened (%s): label is correct" % queue, None
        elif verdict == "UNSURE":
            act, why, new = "unconfirmed", "human listened (%s): unsure" % queue, None
        else:
            act, why, new = "unconfirmed", "human listened (%s): label is wrong, heard reading not typed" % queue, None
            if heard:
                if re.search(r"\d", heard) and not allow_digits:
                    raise ValueError("spoken-form labels have no digits: type the reading in words")
                heard = re.sub(r"\s*,\s*", " ", heard)
                if row is None or row["label"] != it["label"]:
                    why = "human listened (%s): label wrong (heard: %s) but the row label changed since the queue was built" % (queue, heard)
                else:
                    sp = json.loads(it["span"]) if it["span"] else None
                    new = it["label"][:sp[0]] + heard + it["label"][sp[1]:] if (q["scope"] == "span" and sp) else heard
                    act, why = "relabel", "human listened (%s): heard '%s'" % (queue, heard)
        eid = self.add_evidence(it["row_id"], "human_listen", tool, value=verdict, form=queue, payload=dict(item=item_id, gold_id=gid, heard=heard or None))
        res["decision_id"] = self.add_decision(it["row_id"], act, why, "human", new_label=new, evidence_ids=[eid])
        res["action"] = act
        return res

    # -- reads
    def raw_line(self, row_id):
        r = self.db.execute("SELECT raw FROM rows WHERE row_id=?", (row_id,)).fetchone()
        return zlib.decompress(r["raw"]).decode("utf-8") if r else None

    def parts(self):
        return [r["part"] for r in self.db.execute("SELECT part FROM parts ORDER BY part")]

    def stats(self):
        q = lambda s, *a: [dict(r) for r in self.db.execute(s, a)]
        return dict(
            rows=q("SELECT COUNT(*) n FROM rows")[0]["n"],
            by_part=q("SELECT part, COUNT(*) n, SUM(digits_in_label>0) digit_rows FROM rows GROUP BY part ORDER BY part"),
            by_class=q("SELECT source_class, real_synth, COUNT(*) n FROM rows GROUP BY source_class, real_synth ORDER BY n DESC"),
            by_variant=q("SELECT variant, COUNT(*) n FROM rows GROUP BY variant"),
            flags=q("SELECT SUM(chirp) chirp, SUM(tts_narrated=1) tts_narrated, SUM(tts_narrated=0) tts_not_narrated, SUM(tts_narrated IS NULL) tts_unknown, SUM(audio_sha1 IS NOT NULL) audio_hashed FROM rows")[0],
            spans=q("SELECT COUNT(*) n, COUNT(DISTINCT row_id) rows_with FROM spans")[0],
            evidence=q("SELECT source, COUNT(*) n, COUNT(DISTINCT row_id) rows FROM evidence GROUP BY source ORDER BY n DESC"),
            decisions=q("SELECT action, decided_by, COUNT(*) n FROM decisions GROUP BY action, decided_by"),
            gold=q("SELECT bed, answer, COUNT(*) n FROM gold GROUP BY bed, answer"),
            candidates=q("SELECT origin, COUNT(*) n FROM candidates GROUP BY origin"))

    # -- mix export: base listing + decisions
    def export_mix(self, outdir, only_parts=None, apply_decisions=True, label_field="text"):
        """write one jsonl per part (original file name). drop -> row omitted; relabel -> `label_field` replaced by new_label
        (original kept in `label_before_ledger`); everything else byte-identical to the stored line. Returns counts per part."""
        os.makedirs(outdir, exist_ok=True)
        counts = {}
        for p in self.db.execute("SELECT * FROM parts ORDER BY part").fetchall():
            if only_parts and p["part"] not in only_parts:
                continue
            n_in = n_out = n_rel = n_drop = 0
            dec = {}
            if apply_decisions:
                for d in self.db.execute("SELECT l.row_id, l.action, l.new_label FROM latest_decision l JOIN rows r ON r.row_id=l.row_id WHERE r.part=?", (p["part"],)):
                    dec[d["row_id"]] = (d["action"], d["new_label"])
            with open(os.path.join(outdir, p["part"] + ".jsonl"), "w", encoding="utf-8") as f:
                for r in self.db.execute("SELECT row_id, raw FROM rows WHERE part=? ORDER BY idx", (p["part"],)):
                    n_in += 1
                    line = zlib.decompress(r["raw"]).decode("utf-8")
                    act, new = dec.get(r["row_id"], ("keep", None))
                    if act == "drop":
                        n_drop += 1
                        continue
                    if act == "relabel" and new:
                        o = json.loads(line)
                        o["label_before_ledger"] = o.get(label_field)
                        o[label_field] = new
                        line = json.dumps(o, ensure_ascii=False)
                        n_rel += 1
                    f.write(line + "\n")
                    n_out += 1
            counts[p["part"]] = dict(rows_in=n_in, rows_out=n_out, relabelled=n_rel, dropped=n_drop)
        return counts


# ---------------------------------------------------------------------------- CLI
def _print_table(rows):
    if not rows:
        print("(none)")
        return
    cols = list(rows[0].keys())
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r[c] is None else str(r[c]) for c in cols))


def main(argv=None):
    ap = argparse.ArgumentParser(description="label ledger query / export CLI")
    ap.add_argument("--db", default=os.environ.get("LABELS_DB"), required="LABELS_DB" not in os.environ, help="ledger file (or env LABELS_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="row counts per part / source class / evidence source")
    q = sub.add_parser("query", help="run a read-only SELECT")
    q.add_argument("sql")
    s = sub.add_parser("show", help="one row with its spans, evidence and decisions")
    s.add_argument("row_id")
    e = sub.add_parser("export-mix", help="write the mix parts (base listing + decisions)")
    e.add_argument("outdir")
    e.add_argument("--part", action="append")
    e.add_argument("--no-decisions", action="store_true")
    d = sub.add_parser("decide", help="append a decision")
    d.add_argument("row_id")
    d.add_argument("action", choices=ACTIONS)
    d.add_argument("--reason", required=True)
    d.add_argument("--by", required=True, choices=DECIDERS)
    d.add_argument("--new-label")
    v = sub.add_parser("evidence", help="append an evidence record")
    v.add_argument("row_id")
    v.add_argument("source")
    v.add_argument("--tool", required=True)
    v.add_argument("--value")
    v.add_argument("--form")
    v.add_argument("--score", type=float)
    v.add_argument("--margin", type=float)
    v.add_argument("--span-ord", type=int)
    a = ap.parse_args(argv)
    L = Ledger(a.db)
    if a.cmd == "stats":
        print(json.dumps(L.stats(), indent=1))
    elif a.cmd == "query":
        if not re.match(r"\s*(select|with)\b", a.sql, re.I):
            sys.exit("read-only: SELECT / WITH only")
        _print_table([dict(r) for r in L.db.execute(a.sql)])
    elif a.cmd == "show":
        r = L.db.execute("SELECT * FROM rows WHERE row_id=?", (a.row_id,)).fetchone()
        if not r:
            sys.exit("no such row")
        o = {k: r[k] for k in r.keys() if k != "raw"}
        o["spans"] = [dict(x) for x in L.db.execute("SELECT * FROM spans WHERE row_id=? ORDER BY ord", (a.row_id,))]
        o["evidence"] = [dict(x) for x in L.db.execute("SELECT * FROM evidence WHERE row_id=? ORDER BY id", (a.row_id,))]
        o["decisions"] = [dict(x) for x in L.db.execute("SELECT * FROM decisions WHERE row_id=? ORDER BY id", (a.row_id,))]
        o["candidates"] = [dict(x) for x in L.db.execute("SELECT label,origin FROM candidates WHERE row_id=?", (a.row_id,))]
        print(json.dumps(o, indent=1, ensure_ascii=False))
    elif a.cmd == "export-mix":
        print(json.dumps(L.export_mix(a.outdir, set(a.part) if a.part else None, not a.no_decisions), indent=1))
    elif a.cmd == "decide":
        print(L.add_decision(a.row_id, a.action, a.reason, a.by, a.new_label)); L.commit()
    elif a.cmd == "evidence":
        print(L.add_evidence(a.row_id, a.source, a.tool, a.value, a.form, a.score, a.margin, a.span_ord)); L.commit()
    L.close()


if __name__ == "__main__":
    main()
