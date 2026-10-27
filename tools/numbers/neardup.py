"""Drop new sentences that near-duplicate any existing pack row (old synthetic packs + all generated pack sentences) or each other.
Similarity = Jaccard of number-masked word shingles; >= 0.6 is a near-dup (Opus review metric).
  python3 neardup.py --tag v3 --root DATA --index 'old/*.jsonl' --index 'DATA/sentences/v2.jsonl'   -> sentences/<tag>_dedup.jsonl
  add --out PATH to write the kept rows elsewhere (sentences/ stays untouched) and --flagged PATH to save the dropped rows.
Field: SPOKEN text everywhere. New rows use "said"; index rows use "said" when present, else "text" (spoken = what the voice says).
Masking: every number-ish token becomes "#" (consecutive "#" collapse): cardinals, ordinals, decades, fractions, months, weekdays,
  clock/currency/unit words (am/pm, o'clock, noon, midnight, dollars, cents, bucks, percent, grand, K) and single capital letters in codes ("B twelve").
Shingles: word 4-grams; sentences of <= 7 masked words use word 3-grams instead. EVERY row also gets one whole-frame shingle, so rows with an
  identical masked frame always match (short rows are never skipped). Index files are JSONL."""
import argparse, collections, glob, json, os, re

NW = set("zero oh o one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point double triple".split())
ORD = set("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
          "eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth millionth".split())
DECADES = set("twenties thirties forties fifties sixties seventies eighties nineties tens noughties hundreds thousands millions billions dozens".split())
FRACT = set("half halves quarter quarters thirds fourths fifths sixths sevenths eighths ninths tenths".split())
MONTHS = set("january february march april may june july august september october november december".split())
DAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
UNITS = set("am pm oclock noon midnight dollar dollars cent cents buck bucks percent grand k euro euros pound pounds".split())
MASK = NW | ORD | DECADES | FRACT | MONTHS | DAYS | UNITS
NUMLIKE = NW | ORD | {"a", "p"}


def spoken(r): return r.get("said") or r.get("text") or ""


def frame(t):
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+", t.replace("’", "'"))
    w = []
    for i, x in enumerate(toks):
        lo = x.lower(); nxt = toks[i + 1].lower() if i + 1 < len(toks) else ""
        if lo == "o'clock": lo = "oclock"
        code = len(x) == 1 and x.isupper() and (x not in "IA" or nxt in NW or nxt.isdigit())  # "B twelve", not the pronoun I / article A
        ampm = lo in ("a", "p") and nxt == "m"  # "a m" / "p m"
        m = lo in MASK or x.isdigit() or code or ampm or (lo == "m" and i and toks[i - 1].lower() in ("a", "p"))
        tok = "#" if m else lo
        if not (tok == "#" and w and w[-1] == "#"): w.append(tok)
    return w


def shingles(t):
    w = frame(t); n = 3 if len(w) <= 7 else 4
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} | {"\x00F " + " ".join(w)}


class Index:
    def __init__(self): self.inv, self.docs = collections.defaultdict(list), []

    def add(self, sh):
        k = len(self.docs); self.docs.append(sh)
        for s in sh: self.inv[s].append(k)

    def max_sim(self, sh):
        c = collections.Counter(k for s in sh for k in self.inv.get(s, ()))
        return max((n / len(sh | self.docs[k]) for k, n in c.items()), default=0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--root", default="."); ap.add_argument("--index", action="append", default=[], help="glob of existing JSONL packs to compare against")
    ap.add_argument("--out", help="kept-rows path (default sentences/<tag>_dedup.jsonl under --root)"); ap.add_argument("--flagged", help="optional path for the dropped rows")
    a = ap.parse_args(); idx = Index()
    for g in a.index:
        for f in glob.glob(os.path.expanduser(g)):
            for l in open(f):
                if l.strip(): idx.add(shingles(spoken(json.loads(l))))
    os.chdir(a.root)
    keep, drop, flagged = [], collections.Counter(), []
    for r in map(json.loads, open(f"sentences/{a.tag}.jsonl")):
        sh = shingles(spoken(r))
        if idx.max_sim(sh) >= a.thr: drop[r["cell"]] += 1; flagged.append(r); continue
        idx.add(sh); keep.append(r)
    dump = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out = a.out or f"sentences/{a.tag}_dedup.jsonl"
    open(out, "w").write(dump(keep))
    if a.flagged: open(a.flagged, "w").write(dump(flagged))
    print(f"kept {len(keep)}  near-dups dropped {sum(drop.values())} {dict(drop)}  (index {len(idx.docs)} rows)")


if __name__ == "__main__": main()
