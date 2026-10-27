#!/usr/bin/env python3
"""Semantic paraphrase check: MiniLM cosine on number-masked frames of new rows vs existing training texts, held-out texts and the new rows themselves.
A paraphrase written with different wording AND different numbers has lexical Jaccard ~0 (dupcheck.py cannot see it) but a clearly raised cosine here:
the 8 pairs of example_pairs.jsonl score cosine 0.44-0.78 at lexical 0.0-0.05. So this is a WEAK second detector: flag >= 0.80, hard-review >= 0.90.
Calibrate on your own pairs before trusting a threshold.

  python3 semantic.py index --texts 'train/*.jsonl' --name train --out cache/        embed the number-masked frames of a text collection (resumable)
  python3 semantic.py check --rows new.jsonl --index cache/ --names train,eval --out semantic.json [--field said]
  python3 semantic.py calibrate --pairs example_pairs.jsonl [--rows new.jsonl --out calibration.json]   pairs: {"a": ..., "b": ...} per line

`check` writes {<name>_p50, _p99, _max, _ge_0.80, _ge_0.90, within_*, max_any, rows, hits_top}; variety.py's semantic gate reads within_p99 / <name>_p99 / max_any
from this file (--sem). Env: MINILM_DIR (snapshot of sentence-transformers/all-MiniLM-L6-v2) or HF_HOME. Needs numpy only; no network."""
import argparse, glob, json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I does not put the script directory on the path
sys.path.insert(0, os.path.join(HERE, "..", "numbers"))
import neardup as N

FLAG, HARD = 0.80, 0.90


def frame_text(t):
    """the number-masked frame of a sentence as text"""
    return " ".join(N.frame(t))


def usable_frames(texts, min_words=3):
    """unique masked frames with at least min_words non-mask words, in first-seen order"""
    seen, out = set(), []
    for t in texts:
        f = frame_text(t)
        if len(f.replace("#", "").split()) >= min_words and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def read_texts(globs, field=None):
    out = []
    for g in globs:
        for f in sorted(glob.glob(os.path.expanduser(g))):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        t = (r.get(field) if field else None) or N.spoken(r)
                        if t:
                            out.append(t)
    return out


def build_index(name, texts, out_dir, embedder, step=4096, log=print):
    """embed the frames in steps and save out_dir/<name>_frames.json + <name>_emb.npy; an interrupted run resumes from <name>_emb_part.npy"""
    os.makedirs(out_dir, exist_ok=True)
    fp, part, out = (os.path.join(out_dir, name + s) for s in ("_frames.json", "_emb_part.npy", "_emb.npy"))
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            frames = json.load(f)
    else:
        frames = usable_frames(texts)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(frames, f)
    done = np.load(part) if os.path.exists(part) else None
    t0 = time.time()
    for i in range(0 if done is None else len(done), len(frames), step):
        chunk = embedder.encode(frames[i:i + step]).astype(np.float16)
        done = chunk if done is None else np.concatenate([done, chunk])
        np.save(part, done)
        if log:
            log("%s %d / %d %ds" % (name, len(done), len(frames), round(time.time() - t0)))
    if done is None:
        done = np.zeros((0, 1), np.float16)
    np.save(out, done)
    return frames, done


def best_matches(X, T, block=20000):
    """for every row of X (n, d) the best cosine against T (m, d) and its index; both L2-normalised"""
    best = np.full(len(X), -1.0)
    arg = np.zeros(len(X), int)
    for i in range(0, len(T), block):
        S = X @ T[i:i + block].T
        m, k = S.max(1), S.argmax(1) + i
        upd = m > best
        best[upd], arg[upd] = m[upd], k[upd]
    return best, arg


def check(rows, embedder, indexes, field="said", flag=FLAG, hard=HARD, n_hits=8):
    """indexes: {name: (frames list, embeddings (m, d))} -> result dict"""
    X = embedder.encode([frame_text(str(r[field])) for r in rows]).astype(np.float32)
    res, hits = {}, []
    for name, (fr, T) in indexes.items():
        if len(T) == 0:
            continue
        best, arg = best_matches(X, np.asarray(T, dtype=np.float32))
        res["%s_p50" % name], res["%s_p99" % name], res["%s_max" % name] = float(np.median(best)), float(np.quantile(best, .99)), float(best.max())
        res["%s_ge_%.2f" % (name, flag)], res["%s_ge_%.2f" % (name, hard)] = int((best >= flag).sum()), int((best >= hard).sum())
        for j in np.argsort(-best)[:n_hits]:
            hits.append((name, float(best[j]), rows[j].get("n", int(j)), str(rows[j][field])[:90], fr[arg[j]][:90]))
    if len(rows) > 1:
        S = X @ X.T
        np.fill_diagonal(S, -1)
        w = S.max(1)
        res.update(within_p50=float(np.median(w)), within_p99=float(np.quantile(w, .99)), within_max=float(w.max()))
        res["within_ge_%.2f" % flag] = int((w >= flag).sum())
    res["max_any"] = max([v for k, v in res.items() if k.endswith("_max")] or [0.0])
    res["rows"] = len(rows)
    res["hits_top"] = hits
    return res


def calibrate(pairs, embedder, flag=FLAG, hard=HARD, rows=None, field="said"):
    """pairs: [(a, b)] paraphrases (different wording AND numbers). Reports the cosine of the pairs against the lexical Jaccard of neardup, and the pack's own nearest-neighbour spread"""
    ft = frame_text
    A, B = embedder.encode([ft(a) for a, _ in pairs]), embedder.encode([ft(b) for _, b in pairs])
    cos = (A * B).sum(1)
    lex = []
    for a, b in pairs:
        x, y = N.shingles(a), N.shingles(b)
        lex.append(len(x & y) / len(x | y))
    out = dict(pairs=len(pairs), paraphrase_cos=[round(float(c), 2) for c in cos], paraphrase_lexical=[round(x, 2) for x in lex], paraphrase_cos_median=float(np.median(cos)),
               paraphrase_cos_min=float(cos.min()), lexical_caught_ge_0_6=int(sum(x >= .6 for x in lex)), cos_caught_ge_hard=int((cos >= hard).sum()), cos_caught_ge_flag=int((cos >= flag).sum()))
    if rows and len(rows) > 1:
        X = embedder.encode([ft(str(r[field])) for r in rows])
        S = X @ X.T
        np.fill_diagonal(S, -1)
        top1 = S.max(1)
        out["pack_within_top1_cos"] = dict(mean=float(top1.mean()), p50=float(np.median(top1)), p90=float(np.quantile(top1, .9)), p99=float(np.quantile(top1, .99)), max=float(top1.max()))
    return out


def load_index(index_dir, names):
    out = {}
    for name in names:
        with open(os.path.join(index_dir, name + "_frames.json"), encoding="utf-8") as f:
            fr = json.load(f)
        out[name] = (fr, np.load(os.path.join(index_dir, name + "_emb.npy")))
    return out


def main(argv=None, embedder=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--texts", action="append", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--field")
    c = sub.add_parser("check")
    c.add_argument("--rows", required=True)
    c.add_argument("--index", required=True)
    c.add_argument("--names", default="train,eval")
    c.add_argument("--field", default="said")
    c.add_argument("--out", required=True)
    k = sub.add_parser("calibrate")
    k.add_argument("--pairs", required=True)
    k.add_argument("--rows")
    k.add_argument("--field", default="said")
    k.add_argument("--out")
    a = ap.parse_args(argv)
    if embedder is None:
        from minilm_np import Embedder
        embedder = Embedder()
    if a.cmd == "index":
        fr, emb = build_index(a.name, read_texts(a.texts, a.field), a.out, embedder)
        print("DONE %s %s" % (a.name, emb.shape))
        return 0
    if a.cmd == "check":
        with open(a.rows, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        res = check(rows, embedder, load_index(a.index, a.names.split(",")), a.field)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "hits_top"}))
        for h in res["hits_top"]:
            print(h)
        return 0
    with open(a.pairs, encoding="utf-8") as f:
        pairs = [(r["a"], r["b"]) for r in map(json.loads, (l for l in f if l.strip()))]
    rows = None
    if a.rows:
        with open(a.rows, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    out = calibrate(pairs, embedder, rows=rows, field=a.field)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
