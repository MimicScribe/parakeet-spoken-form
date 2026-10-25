"""Drop new sentences that near-duplicate any existing pack row (old synthetic packs + all ttsBIG1 sentences) or each other.
Similarity = Jaccard of number-masked word 4-grams (numbers -> #); >= 0.6 is a near-dup (Opus review metric).
  python3 neardup.py --tag v3 --root DATA --index 'old/*.jsonl' --index 'DATA/sentences/v2.jsonl'   -> sentences/<tag>_dedup.jsonl
Index files are JSONL with a "said" or "text" field."""
import argparse, collections, glob, json, os, re

NW = set("zero oh o one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point double triple".split())


def shingles(t):
    w = []
    for x in re.findall(r"[a-z']+|\d+", t.lower()):
        tok = "#" if (x in NW or x.isdigit()) else x
        if not (tok == "#" and w and w[-1] == "#"): w.append(tok)
    return {" ".join(w[i:i + 4]) for i in range(max(1, len(w) - 3))}


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
    a = ap.parse_args(); idx = Index()
    for g in a.index:
        for f in glob.glob(os.path.expanduser(g)):
            for l in open(f):
                r = json.loads(l); idx.add(shingles(r.get("said") or r.get("text") or ""))
    os.chdir(a.root)
    keep, drop = [], collections.Counter()
    for r in map(json.loads, open(f"sentences/{a.tag}.jsonl")):
        sh = shingles(r["said"])
        if len(sh) >= 2 and idx.max_sim(sh) >= a.thr: drop[r["cell"]] += 1; continue
        idx.add(sh); keep.append(r)
    open(f"sentences/{a.tag}_dedup.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print(f"kept {len(keep)}  near-dups dropped {sum(drop.values())} {dict(drop)}  (index {len(idx.docs)} rows)")


if __name__ == "__main__": main()
