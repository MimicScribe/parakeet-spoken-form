#!/usr/bin/env python3
"""Variety gates for a synthetic sentence pack: text, categorical balance, slot values and (optionally) the audio plan. Every line prints PASS / FAIL with the measured number.

  python3 variety.py --rows sentences.jsonl [--audio audio_plan.jsonl] [--sem semantic.json] [--config gates.json] [--out report.txt]      exit 1 when a gate fails

Row schema (jsonl): `said` (the spoken sentence, required); optional `carrier` (text outside the number slots, `{0}` placeholders; defaults to the masked `said`);
`slots` (list of dicts with at least `value`; `cell` and `kind` when you use closed-space cells); any categorical fields you want balanced (domain, register, position, ...).
Audio plan rows (same order as the rows): `tempo`, `voice`, `gender`, `engine`, ... as you use them.

Text gates (defaults, all overridable in the config json under "text"):
  distinct3_min 0.80          distinct word 3-grams of the carriers (numbers masked)
  patterns_min 0.95, top50_max 0.05    number-masked sentence frames per row, and the share of the 50 most common frames
  opening3_max 0.03, ending2_max 0.04  most common opening 3-gram / ending 2-gram share
  questions_min 0.04          share of sentences with a question mark
  len_p10_max 8, len_p90_min 18, len_min 3    carrier length spread in words
Config sections: "semantic" {within_p99_max 0.90, index_p99_max 0.88, hard 0.90} (used with --sem); "caps" {field: max share of any one value}; "shares" [{"field", "value", "min", "max"}]; "slots" {"closed_cells": [...], "max_repeat": 3, "max_scene_share": 0.01};
"audio" {"tempo": [lo, hi], "tempo_exact_min": share of rows at exactly 1.0, "gender_band": [lo, hi], "max_rows_per_voice": n, "shares": [{"field", "value", "min", "max"}]}.
Sampling noise: with fewer than 1,000 rows, gates marked [n] are relaxed by 2 * sqrt(p(1-p)/n) (p = 0.05); the full build is gated at the strict value."""
import argparse, collections, json, math, os, re, statistics, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "numbers"))
import neardup as N

DEFAULTS = {
    "text": dict(distinct3_min=0.80, patterns_min=0.95, top50_max=0.05, opening3_max=0.03, ending2_max=0.04, questions_min=0.04, len_p10_max=8, len_p90_min=18, len_min=3),
    "caps": {}, "shares": [],
    "slots": dict(closed_cells=[], max_repeat=3, max_scene_share=0.01),
    "semantic": dict(within_p99_max=0.90, index_p99_max=0.88, hard=0.90),
    "audio": dict(tempo=None, tempo_exact_min=None, gender_band=None, max_rows_per_voice=None, shares=[]),
}


def noise(n, p=0.05):
    return 2.0 * math.sqrt(p * (1 - p) / max(n, 1))


def merged(cfg):
    out = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def carrier_words(r):
    c = r.get("carrier")
    if c is None:
        return [w for w in N.frame(r["said"]) if w != "#"]
    return words(re.sub(r"\{\d\}", " ", c))


def text_gates(rows, cfg, strict, mg):
    n = len(rows)
    t = cfg["text"]
    out = []
    w = [carrier_words(r) for r in rows]

    def distinct(k):
        g = [tuple(x[i:i + k]) for x in w for i in range(len(x) - k + 1)]
        return len(set(g)) / max(len(g), 1)
    d1, d2, d3 = distinct(1), distinct(2), distinct(3)
    out.append(("T1 carrier distinct-3 >= %.2f" % t["distinct3_min"], d3 >= t["distinct3_min"], "d1=%.2f d2=%.2f d3=%.3f" % (d1, d2, d3)))
    fr = collections.Counter(" ".join(N.frame(r["said"])) for r in rows)
    top50 = sum(c for _, c in fr.most_common(50)) / n
    out.append(("T2 patterns/row >= %.2f and top-50 frames <= %.2f" % (t["patterns_min"], t["top50_max"]), len(fr) / n >= t["patterns_min"] and top50 <= max(t["top50_max"], 1.2 * 50 / n),
                "patterns/row=%.3f top50=%.3f" % (len(fr) / n, top50)))
    op = collections.Counter(" ".join(x[:3]) for x in w if len(x) >= 3)
    en = collections.Counter(" ".join(x[-2:]) for x in w if len(x) >= 2)
    o1, e1 = (op.most_common(1) or [("", 0)])[0], (en.most_common(1) or [("", 0)])[0]
    out.append(("T3 top opening 3-gram <= %.0f%% and top ending 2-gram <= %.0f%% [n]" % (100 * t["opening3_max"], 100 * t["ending2_max"]),
                o1[1] / n <= t["opening3_max"] + mg and e1[1] / n <= t["ending2_max"] + mg, "opening '%s' %.3f; ending '%s' %.3f" % (o1[0], o1[1] / n, e1[0], e1[1] / n)))
    q = sum(1 for r in rows if "?" in r["said"]) / n
    out.append(("T8 question sentences >= %.0f%%" % (100 * t["questions_min"]), q >= t["questions_min"], "%.3f" % q))
    nw = sorted(len(x) for x in w)
    p10, p90 = nw[int(.1 * n)], nw[min(n - 1, int(.9 * n))]
    out.append(("T9 carrier length spread: p10 <= %d and p90 >= %d words, no row < %d" % (t["len_p10_max"], t["len_p90_min"], t["len_min"]),
                p10 <= t["len_p10_max"] and p90 >= t["len_p90_min"] and nw[0] >= t["len_min"], "min=%d p10=%d p50=%s p90=%d max=%d" % (nw[0], p10, statistics.median(nw), p90, nw[-1])))
    return out


def share_gates(rows, cfg, mg, label="rows"):
    n = len(rows)
    out = []
    for fld, cap in sorted(cfg["caps"].items()):
        c = collections.Counter(r.get(fld) for r in rows if r.get(fld) is not None)
        if c:
            top = c.most_common(1)[0]
            out.append(("C %s: max value share <= %.0f%% [n]" % (fld, 100 * cap), top[1] / n <= cap + mg, "%d values, top '%s' %.3f" % (len(c), top[0], top[1] / n)))
    for s in cfg["shares"]:
        got = sum(1 for r in rows if r.get(s["field"]) == s["value"]) / n
        lo, hi = s.get("min"), s.get("max")
        ok = (lo is None or got >= lo - mg) and (hi is None or got <= hi + mg)
        out.append(("S %s=%s share in [%s, %s] [n]" % (s["field"], s["value"], lo, hi), ok, "%.3f" % got))
    return out


def slot_gates(rows, cfg, strict, mg):
    n = len(rows)
    sc = cfg["slots"]
    out = []
    S = [(r, s) for r in rows for s in (r.get("slots") or [])]
    if not S:
        return out
    npr = collections.Counter(len(r.get("slots") or []) for r in rows)
    out.append(("N numbers per row: 1 slot >= 55%, >=2 slots >= 10%", npr[1] / n >= .55 and sum(v for k, v in npr.items() if k >= 2) / n >= .10, dict(sorted(npr.items()))))
    vals = [str(s["value"]) for _, s in S]
    vc = collections.Counter(vals)
    closed = set(sc["closed_cells"])
    closed_vals = {str(s["value"]) for _, s in S if s.get("cell") in closed}
    open_dups = sum(c - 1 for v, c in vc.items() if c > 1 and v not in closed_vals)
    worst = max(vc.values())
    out.append(("V1 slot values: open-space values unique, closed-space cells repeat <= %d times" % sc["max_repeat"], open_dups <= (0 if strict else 3) and worst <= sc["max_repeat"],
                "%d/%d distinct, open-space repeats=%d, max multiplicity=%d" % (len(set(vals)), len(vals), open_dups, worst)))
    sid = collections.Counter(s.get("scene_id") for _, s in S if s.get("scene_id"))
    ns = sum(sid.values())
    if ns >= 100:
        lim = sc["max_scene_share"] if n >= 1000 else 3.5 * sc["max_scene_share"]
        out.append(("T13 topic repetition: no scene holds > %.1f%% of scene-bearing slots" % (100 * lim), sid.most_common(1)[0][1] / ns <= lim,
                    "%d scenes over %d slots, top %.4f, median rows/scene %s" % (len(sid), ns, sid.most_common(1)[0][1] / ns, statistics.median(sid.values()))))
    return out


def audio_gates(audio, cfg, mg):
    A = cfg["audio"]
    n = len(audio)
    out = []
    if A.get("tempo"):
        lo, hi = A["tempo"]
        tp = [x["tempo"] for x in audio]
        exact = sum(t == 1.0 for t in tp) / n
        ok = min(tp) >= lo and max(tp) <= hi and (A.get("tempo_exact_min") is None or exact >= A["tempo_exact_min"])
        out.append(("A1 tempo in [%s, %s], >= %s exactly 1.00" % (lo, hi, A.get("tempo_exact_min")), ok, "min=%s max=%s exactly1.0=%.2f" % (min(tp), max(tp), exact)))
    if A.get("gender_band"):
        lo, hi = A["gender_band"]
        gd = collections.Counter(x["gender"] for x in audio)
        out.append(("A2 gender share in [%s, %s] each" % (lo, hi), len(gd) == 2 and all(lo <= v / n <= hi for v in gd.values()), dict(gd)))
    if A.get("max_rows_per_voice"):
        vc = collections.Counter(x["voice"] for x in audio)
        out.append(("A3 no voice voices more than %d rows" % A["max_rows_per_voice"], max(vc.values()) <= A["max_rows_per_voice"], "%d voices, max %d rows" % (len(vc), max(vc.values()))))
    for s in A.get("shares", []):
        got = sum(1 for x in audio if x.get(s["field"]) == s["value"]) / n
        lo, hi = s.get("min"), s.get("max")
        out.append(("A %s=%s share in [%s, %s] [n]" % (s["field"], s["value"], lo, hi), (lo is None or got >= lo - mg) and (hi is None or got <= hi + mg), "%.3f" % got))
    return out


def semantic_gate(sem, cfg):
    """sem = the json written by semantic.py check: within-pack and vs-index p99 cosine caps, and no row at or above the hard bar"""
    c = cfg["semantic"]
    p99s = {k: v for k, v in sem.items() if k.endswith("_p99")}
    ok = all(v <= (c["within_p99_max"] if k.startswith("within") else c["index_p99_max"]) for k, v in p99s.items()) and sem.get("max_any", 0) < c["hard"]
    return [("T11 semantic (MiniLM): within-pack p99 cosine <= %.2f; vs indexes p99 <= %.2f; none >= %.2f" % (c["within_p99_max"], c["index_p99_max"], c["hard"]), ok,
             json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sem.items() if k != "hits_top"}))]


def evaluate(rows, audio=None, config=None, sem=None):
    """-> list of (name, ok, message)"""
    cfg = merged(config)
    n = len(rows)
    strict = n >= 1000
    mg = 0 if strict else noise(n)
    gates = text_gates(rows, cfg, strict, mg) + share_gates(rows, cfg, mg) + slot_gates(rows, cfg, strict, mg)
    if audio:
        gates += audio_gates(audio, cfg, mg)
    if sem:
        gates += semantic_gate(sem, cfg)
    return gates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--sem", help="json written by semantic.py check (adds the semantic gate)")
    ap.add_argument("--config", help="json with threshold overrides and categorical gates")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    audio = [json.loads(l) for l in open(a.audio, encoding="utf-8") if l.strip()] if a.audio else None
    cfg = json.load(open(a.config, encoding="utf-8")) if a.config else None
    sem = json.load(open(a.sem, encoding="utf-8")) if a.sem else None
    gates = evaluate(rows, audio, cfg, sem)
    lines = ["%s  %s: %s" % ("PASS" if ok else "FAIL", name, msg) for name, ok, msg in gates]
    fails = [name for name, ok, _ in gates if not ok]
    lines.append("\n%d/%d gates pass; FAIL: %s" % (len(gates) - len(fails), len(gates), fails))
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
