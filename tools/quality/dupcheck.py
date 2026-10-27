#!/usr/bin/env python3
"""Near-duplicate and leakage check of a new sentence file against existing training rows and held-out (eval) texts.
SAME FIELD everywhere: SPOKEN text, number-masked word shingles (tools/numbers/neardup.py, threshold 0.6). Per new row the checked fields are
`said` (what the voice says), optionally `clean` (without fillers) and `label_A` / any other --fields; the row's score is the max over fields.

  python3 dupcheck.py --rows new.jsonl --train 'train/*.jsonl' --eval 'eval/*.jsonl' --out dupcheck.txt [--eval-text-field text] [--gate]

Reports: max-similarity distribution vs TRAIN and vs EVAL (histogram, quantiles, hits >= 0.5 with the matched old text), within-pack sequential similarity,
exact duplicate sentences, and LEAKAGE of held-out content into the new rows:
  * digit strings of >= --min-digits (5) digits that appear in the new row's fields and in any eval text (written, or spoken digit by digit)
  * spoken number runs of >= --min-run-words (4) number words that occur verbatim in an eval text
With --gate the exit code is 1 when any leakage or any exact duplicate is found.
Units are folded to one token ('153.7 GHz' = 'gigahertz') with --unit-words FILE (one word per line); without it only the symbols % and degrees are folded.
Files are JSONL; an old row's text is its `said`, else `text`."""
import argparse, collections, glob, json, os, re, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "numbers"))
import neardup as N

DEFAULT_UNITS = {"degrees", "celsius", "fahrenheit", "percent"}
THRESHOLDS = (0.3, 0.5, 0.6)
HIST_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]


def unitnorm(t, units=DEFAULT_UNITS):
    """written units ('9.84 °F') and spoken units ('degrees fahrenheit') become one token, so a written-form eval row matches its spoken twin"""
    return " ".join("unit" if w.lower().strip(".,;:!?°") in units else w for w in t.replace("°", " ").replace("%", " percent ").split())


class Idx(N.Index):
    """neardup.Index that also remembers where each document came from and which document is closest"""

    def __init__(self):
        super().__init__()
        self.meta = []

    def add2(self, sh, src, txt):
        self.add(sh)
        self.meta.append((src, txt))

    def best(self, sh):
        c = collections.Counter(k for s in sh for k in self.inv.get(s, ()))
        if not c:
            return 0.0, None
        k, n = max(c.items(), key=lambda kv: kv[1] / len(sh | self.docs[kv[0]]))
        return n / len(sh | self.docs[k]), k


def read_jsonl(globs):
    """-> [(source file, row dict)]"""
    out = []
    for g in globs:
        for f in sorted(glob.glob(os.path.expanduser(g))):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.append((os.path.basename(f), json.loads(line)))
    return out


def build(items, shingler):
    """items: [(source, text)] -> Idx"""
    ix = Idx()
    for src, txt in items:
        if txt and txt.strip():
            ix.add2(shingler(txt), src, txt)
    return ix


def hist(vals):
    h = collections.Counter()
    for v in vals:
        for lo, hi in zip(HIST_BINS, HIST_BINS[1:]):
            if lo <= v < hi:
                h["%.1f-%.1f" % (lo, min(hi, 1.0))] += 1
    return dict(h)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0


def digit_strings(text, min_digits):
    """digit strings of >= min_digits digits: written (separators dropped inside a group) or spoken digit by digit"""
    out = set()
    for m in re.findall(r"\d[\d ,.\-]*\d|\d", text):
        d = re.sub(r"\D", "", m)
        if len(d) >= min_digits:
            out.add(d)
    out |= {d for d in (spoken_digit_strings(text)) if len(d) >= min_digits}
    return out


_DW = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def spoken_digit_strings(text):
    """maximal runs of spoken digit words ('double' / 'triple' expand the next word) -> digit strings"""
    toks = re.findall(r"[a-z']+|\d+|[^\sa-z\d]", text.lower())
    out, cur, i = [], "", 0
    while i < len(toks):
        w = toks[i]
        if w in ("double", "triple") and i + 1 < len(toks) and toks[i + 1] in _DW:
            cur += _DW[toks[i + 1]] * (2 if w == "double" else 3)
            i += 2
            continue
        if w in _DW:
            cur += _DW[w]
        else:
            if cur:
                out.append(cur)
            cur = ""
        i += 1
    if cur:
        out.append(cur)
    return out


def number_runs(text, min_words):
    """maximal runs of >= min_words number words (neardup.NW plus 'and' inside a run), lower-case, space-joined"""
    toks = re.findall(r"[a-z']+|\d+|[^\sa-z\d]", text.lower())
    runs, cur = [], []
    for w in toks + ["."]:
        if w in N.NW or (w == "and" and cur):
            cur.append(w)
        else:
            while cur and cur[-1] == "and":
                cur.pop()
            if len(cur) >= min_words:
                runs.append(" ".join(cur))
            cur = []
    return runs


def eval_leakage(rows, fields, eval_texts, min_digits=5, min_run_words=4):
    """-> dict(digit_hits=[(row index, digit string)], run_hits=[(row index, run)]) of held-out content reproduced in the new rows"""
    ev_join = " " + re.sub(r" +", " ", " \n ".join(re.sub(r"[^a-z0-9 ]+", " ", t.lower()) for t in eval_texts)) + " "
    ev_digits = set()
    for t in eval_texts:
        ev_digits |= digit_strings(t, min_digits)
    dig, run = [], []
    for i, r in enumerate(rows):
        for f in fields:
            t = str(r.get(f) or "")
            for d in digit_strings(t, min_digits):
                if d in ev_digits:
                    dig.append((i, d))
            for rr in number_runs(t, min_run_words):
                if (" " + re.sub(r"[^a-z0-9 ]+", " ", rr) + " ") in ev_join:
                    run.append((i, rr))
    return dict(digit_hits=sorted(set(dig)), run_hits=sorted(set(run)))


def run(rows, train_items, eval_items, fields=("said",), out=None, thr=0.6, units=DEFAULT_UNITS, min_digits=5, min_run_words=4, id_field="n"):
    t0 = time.time()
    SH = lambda t: N.shingles(unitnorm(t, units))
    R = []
    train, ev = build(train_items, SH), build(eval_items, SH)
    R.append("TRAIN index: %d rows" % len(train.docs))
    R.append("EVAL index: %d texts" % len(ev.docs))
    res = []
    for r in rows:
        best = {"train": (0.0, None, None), "eval": (0.0, None, None)}
        for f in fields:
            if not r.get(f):
                continue
            sh = SH(str(r[f]))
            for name, ix in (("train", train), ("eval", ev)):
                v, k = ix.best(sh)
                if v > best[name][0]:
                    best[name] = (v, k, f)
        res.append((r, best))
    rid = lambda i, r: r.get(id_field, i)
    for name, ix in (("train", train), ("eval", ev)):
        v = [b[name][0] for _, b in res]
        if not v:
            continue
        R.append("\n[max similarity vs %s] n=%d mean=%.3f p50=%.2f p90=%.2f p99=%.2f max=%.2f" % (name.upper(), len(v), sum(v) / len(v), pct(v, .5), pct(v, .9), pct(v, .99), max(v)))
        R.append("  hist " + json.dumps(hist(v)))
        for t in THRESHOLDS:
            R.append("  rows >= %s: %d (%.1f%%)" % (t, sum(x >= t for x in v), 100 * sum(x >= t for x in v) / len(v)))
        hits = sorted([(b[name][0], i, r, b[name]) for i, (r, b) in enumerate(res) if b[name][0] >= 0.5], key=lambda x: -x[0])
        R.append("  HITS >= 0.5: %d" % len(hits))
        for sim, i, r, (v_, k, f) in hits[:25]:
            R.append("   %.2f [%s field=%s] NEW: %s\n          OLD (%s): %s" % (sim, rid(i, r), f, str(r.get(f))[:100], ix.meta[k][0], ix.meta[k][1][:100]))
    pk, w, wh = Idx(), [], []
    for i, r in enumerate(rows):
        txt = str(r.get(fields[0]) or "")
        sh = SH(txt)
        v, k = pk.best(sh)
        w.append(v)
        if v >= 0.5:
            wh.append((v, rid(i, r), txt[:90], pk.meta[k][1][:90]))
        pk.add2(sh, rid(i, r), txt)
    if w:
        R.append("\n[within-pack, sequential] mean=%.3f p90=%.2f max=%.2f rows>=0.5: %d rows>=%.1f: %d" % (sum(w) / len(w), pct(w, .9), max(w), len(wh), thr, sum(x >= thr for x in w)))
        R.append("  hist " + json.dumps(hist(w)))
        for x in sorted(wh, reverse=True)[:12]:
            R.append("   %.2f %s: %s  ~  %s" % x)
    ex = collections.Counter(re.sub(r"\W+", " ", str(r.get(fields[0]) or "").lower()).strip() for r, _ in res)
    n_exact = sum(v - 1 for v in ex.values() if v > 1)
    R.append("  exact duplicate sentences in pack: %d" % n_exact)
    lk = eval_leakage(rows, fields, [t for _, t in eval_items], min_digits, min_run_words)
    R.append("\n[leakage] eval digit strings (>= %d digits) reproduced: %d %s" % (min_digits, len(lk["digit_hits"]), lk["digit_hits"][:6]))
    R.append("  spoken number runs (>= %d words) occurring verbatim in an eval text: %d %s" % (min_run_words, len(lk["run_hits"]), lk["run_hits"][:6]))
    leak = bool(lk["digit_hits"] or lk["run_hits"])
    R.append("  GATE leakage 0 and exact duplicates 0 -> " + ("PASS" if not (leak or n_exact) else "FAIL"))
    R.append("\n(elapsed %ds)" % round(time.time() - t0))
    text = "\n".join(R) + "\n"
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return dict(report=text, results=res, leakage=lk, exact_duplicates=n_exact, ok=not (leak or n_exact))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--train", action="append", default=[], help="glob of existing training jsonl (repeatable)")
    ap.add_argument("--eval", action="append", default=[], help="glob of held-out jsonl (repeatable)")
    ap.add_argument("--fields", default="said", help="comma list of row fields to check (default said)")
    ap.add_argument("--eval-text-field", help="text field of eval rows (default: said, else text)")
    ap.add_argument("--unit-words", help="file with one unit word per line to fold into one token")
    ap.add_argument("--min-digits", type=int, default=5)
    ap.add_argument("--min-run-words", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--gate", action="store_true", help="exit 1 on leakage or exact duplicates")
    a = ap.parse_args(argv)
    rows = [r for _, r in read_jsonl([a.rows])]
    pick = lambda r, fld=None: (r.get(fld) if fld else None) or N.spoken(r)
    tr = [(s, pick(r)) for s, r in read_jsonl(a.train)]
    ev = [(s, pick(r, a.eval_text_field)) for s, r in read_jsonl(a.eval)]
    units = DEFAULT_UNITS
    if a.unit_words:
        with open(a.unit_words, encoding="utf-8") as f:
            units = DEFAULT_UNITS | {l.strip().lower() for l in f if l.strip()}
    res = run(rows, tr, ev, tuple(a.fields.split(",")), a.out, units=units, min_digits=a.min_digits, min_run_words=a.min_run_words)
    print(res["report"])
    return 1 if (a.gate and not res["ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
