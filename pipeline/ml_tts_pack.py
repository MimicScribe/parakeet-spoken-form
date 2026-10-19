"""Tier-1 data packs (2026-10-18): English look-alikes (ttsLK1), decimal words (ttsDW1, six languages + English) and
percent with the word (ttsPC1, es / fr / pt). The planning, voicing and verification are ml_tts_symbol.py and
ml_tts_open.py; this module adds what the packs need on top of them:

  voices-en   English rows -> Kokoro voices (share) + cloned voices (ml_tts_symbol.assign_voices_en2)
  select      CTC-verified rows -> the pack: finalize (exact fit, or fuzzy + every number agrees), convention trim at the
              real-speech rates, <= --per-carrier rows per carrier sentence, --n rows, report (value spread, conventions)
  markers     the judge marker a set needs beside its rows (judged_<set>_judge.json; per-clip CTC is the check)

    python ml_tts_pack.py voices-en --planned plan/planned_en.jsonl --refs refs/meta_en.json --vtag ttsLK1v --out v/en.jsonl
    python ml_tts_pack.py select --rows ctc/es.jsonl --lang es --n 400 --per-carrier 12 --out pack/es.jsonl --report pack/es.json
"""
import argparse
import collections
import json
import os
import random
import re
import sys

import ml_tts_symbol as S


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_voices(rows: list, kokoro_share: float, seed: str) -> tuple:
    """(kokoro rows, cloned rows): a seeded shuffle, the first kokoro_share of the rows go to Kokoro."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    k = int(round(len(rows) * kokoro_share))
    return rows[:k], rows[k:]


def cap_per_carrier(rows: list, n: int, per_carrier: int, rng, key="template") -> tuple:
    """Up to n rows, no carrier sentence in more than per_carrier of them. Rows are taken round robin over the carriers
    (a carrier with fewer rows gives all of them first), so the pack spreads over as many carriers as it can."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get(key) or r["text"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    rng.shuffle(keys)
    out, rnd = [], 0
    while len(out) < n and rnd < per_carrier:
        for k in keys:
            if len(out) >= n:
                break
            if rnd < len(by[k]):
                out.append(by[k][rnd])
        rnd += 1
    return out, {"carriers_in": len(by), "carriers_used": len({r.get(key) or r["text"] for r in out}),
                 "rows_in": len(rows), "rows_out": len(out)}


def value_report(rows: list, kinds=("dec", "pct")) -> dict:
    """Top written-value share per slot kind (spans' captions); the pack's rule is every value under 3% (ttsSYM1)."""
    out = {}
    for kd in kinds:
        c = collections.Counter(sp.get("text") or sp.get("reading") for r in rows for sp in r.get("spans") or []
                                if sp.get("kind") == kd)
        tot = sum(c.values())
        if tot:
            top = c.most_common(1)[0]
            out[kd] = {"spans": tot, "distinct": len(c), "top": top[0], "top_share": round(top[1] / tot, 4)}
    return out


def convention_report(rows: list, lang: str) -> dict:
    t = collections.Counter()
    for r in rows:
        for sp in r.get("spans") or []:
            if sp.get("kind") == "pct" and lang in S.MINORITY:
                t["pct_minority" if S.MINORITY[lang].search(sp["reading"]) else "pct_major"] += 1
            st = S.dec2_style(lang, sp.get("text", ""), sp.get("reading", "")) if lang in S.DEC2_DIGITWISE else None
            if st:
                t["dec2_" + st] += 1
    out = dict(t)
    if t["dec2_digitwise"] + t["dec2_number"]:
        out["dec2_digitwise_share"] = round(t["dec2_digitwise"] / (t["dec2_digitwise"] + t["dec2_number"]), 3)
        out["dec2_digitwise_real"] = S.DEC2_DIGITWISE.get(lang)
    if t["pct_minority"] + t["pct_major"]:
        out["pct_minority_share"] = round(t["pct_minority"] / (t["pct_minority"] + t["pct_major"]), 3)
    return out


def select(rows: list, lang: str, n: int, per_carrier: int, seed: int = 18) -> tuple:
    rng = random.Random(f"pack|{seed}|{lang}")
    fin, st = S.finalize(rows, lang)
    if lang != "en":
        fin, tst = S.convention_trim(fin, lang, rng)
        st["trim"] = tst
    fin.sort(key=lambda r: r["id"])
    keep, cst = cap_per_carrier(fin, n, per_carrier, rng)
    keep.sort(key=lambda r: r["id"])
    rep = {"finalize": st, "carriers": cst, "rows": len(keep), "hours": round(sum(r["duration"] for r in keep) / 3600, 3),
           "values": value_report(keep), "conventions": convention_report(keep, lang),
           "voices": len({r["voice"] for r in keep}), "classes": dict(collections.Counter(g for r in keep for g in r.get("gclasses") or []))}
    return keep, rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voices-en")
    v.add_argument("--planned", required=True)
    v.add_argument("--refs", required=True)
    v.add_argument("--vtag", required=True)
    v.add_argument("--kokoro-share", type=float, default=0.35)
    v.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rows", required=True, help="CTC-verified rows (comma-separated files)")
    s.add_argument("--lang", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--per-carrier", type=int, default=12)
    s.add_argument("--seed", type=int, default=18)
    s.add_argument("--out", required=True)
    s.add_argument("--report", default="")
    m = sub.add_parser("marker")
    m.add_argument("--set", required=True)
    m.add_argument("--lang", required=True)
    m.add_argument("--rows", type=int, required=True)
    m.add_argument("--chunks", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "voices-en":
        rows = _rows(a.planned)
        ko, cl = split_voices(rows, a.kokoro_share, f"{a.vtag}|split")
        S.assign_voices(ko, "en", a.vtag)
        refs = json.load(open(a.refs))["refs"]
        by = S.assign_voices_en2(cl, refs, a.vtag)
        _write(a.out, ko + cl)
        print(len(ko), "kokoro,", len(cl), "cloned", by)
    elif a.cmd == "select":
        rows = [x for p in a.rows.split(",") for x in _rows(p)]
        keep, rep = select(rows, a.lang, a.n, a.per_carrier, a.seed)
        _write(a.out, keep)
        if a.report:
            json.dump(rep, open(a.report, "w"), ensure_ascii=False, indent=1)
        print(json.dumps(rep, ensure_ascii=False))
    elif a.cmd == "marker":
        json.dump({"lang": a.lang, "set": a.set, "rows": a.rows,
                   "judge": "per-clip CTC check only (ml_tts.tts_lang check_greedy: CER <= 0.15, every reading heard); a fuzzy fit "
                            "kept only when the greedy text holds every number value and every unit / decimal / percent word "
                            "(ml_tts_symbol.finalize); the reading's value read back through the scorer's reader at planning; "
                            "no second transcript",
                   "chunks": a.chunks.split(","), "date": "2026-10-18"}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
