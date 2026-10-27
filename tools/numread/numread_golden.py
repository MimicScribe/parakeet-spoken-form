#!/usr/bin/env python3
"""Golden pairs for numread (10-26 scoring review fix 2): run BEFORE reading any table; must pass 100 %.
Pair types in numread_golden.jsonl: reads (text -> literal values), acc (spoken key vs hypothesis -> classify outcome, negatives
included: 'in twenty five' != 2005, 'sixty si' != 66, 5 != 50 != 500), rep (exact repeat count).
usage: python3 -I scripts/spoken_form_eval/numread_golden.py [-v]     exit 1 on any failure; prints the reader stamp."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numread as R  # noqa: E402

GOLDEN = os.path.join(HERE, "numread_golden.jsonl")


def load(path=GOLDEN):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(p):
    t = p["t"]
    if t == "reads":
        got = [r.val for r in R.reads(p["text"], p["lang"])]
        return got == p["want"], got
    if t == "acc":
        w = R.want_from(p["key"], p.get("written"), p["lang"])
        got = R.classify(w, p["hyp"], p["lang"], refcount=p.get("refcount", 1))[0]
        return got == p["want"], got
    if t == "rep":
        got = R.repeat_count(p["text"], p["unit"], p["lang"])
        return got == p["want"], got
    raise ValueError(t)


def run(path=GOLDEN, verbose=False):
    pairs = load(path)
    bad = []
    for p in pairs:
        ok, got = check(p)
        if not ok:
            bad.append((p, got))
        if verbose:
            print("PASS" if ok else "FAIL", p["t"], json.dumps({k: v for k, v in p.items() if k not in ("t", "why")}, ensure_ascii=False), "->", got)
    return pairs, bad


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pairs, bad = run(verbose="-v" in argv)
    print(R.stamp_line())
    for p, got in bad:
        print("FAIL", p["t"], json.dumps(p, ensure_ascii=False), "got", got)
    n = {k: sum(1 for p in pairs if p["t"] == k) for k in ("reads", "acc", "rep")}
    neg = sum(1 for p in pairs if p["t"] == "acc" and p["want"] in ("trunc", "wrong", "lost"))
    print(f"golden: {len(pairs) - len(bad)}/{len(pairs)} pass  (reads {n['reads']}, acc {n['acc']} of which {neg} negative, rep {n['rep']})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
