"""Cents (prices with cents), phone numbers: planning for the ttsCYP1 pack from the symbol-class machinery
(ml_tts_symbol.plan_lang, carriers_sym.json) against the current mix's synthetic room, and the replacing variant
(ttsCYP1r: each new row supersedes older single-class synthetic rows of its class) for cells at their synthetic cap.

    python ml_tts_cyp.py plan --cov cov.json --mix <dir with tts_<lang>.jsonl> --out <dir> [--langs es,fr,pt,de,it,nl]
"""
import argparse
import json
import os
import random

import ml_tts_spread as X
import ml_tts_symbol as S

MAIN_ROWS = {"es": 450, "fr": 450, "pt": 180, "de": 80}
REPL_ROWS = {"it": 200, "nl": 200, "de": 120}
FOCUS = ("cur", "phone")
ZERO = {c: 0 for c in ("int_small", "year", "date", "ord")}


_MISSING: list = []
_orig_realize = S.realize_mixed


def _recording(lang, written, ranges, rng, used):
    res, why = _orig_realize(lang, written, ranges, rng, used)
    if res is None and why == "missing_grammar" and len(_MISSING) < 4000:
        _MISSING.append(written)
    return res, why


S.realize_mixed = _recording


rep: dict = {}


def with_table(lang, fn, a):
    """fn() with the grammar overlay: sentences that lacked a table value are asked from the dumper (a copy of the
    shared table), then planned again."""
    for _ in range(2):
        del _MISSING[:]
        out = fn()
        if not _MISSING or not a.dumper:
            return out
        n = X.ensure_table(lang, list(_MISSING), a.dumper, a.out, os.path.join(a.out, f"gt_{lang}.json"))
        print(lang, "grammar keys added", n)
    del _MISSING[:]
    return fn()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("--cov", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="es,fr,pt,de,it,nl")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--dumper", default="")
    a = ap.parse_args()
    cov = json.load(open(a.cov))
    os.makedirs(a.out, exist_ok=True)
    for lang in a.langs.split(","):
        room = {c: {"room": int(cov[lang]["room"][c] * 0.9)} for c in FOCUS}
        if lang in MAIN_ROWS and any(room[c]["room"] > 20 for c in FOCUS):
            def run():
                global rep
                rep = {}
                return S.plan_lang(lang, room, "ttsCYP1", a.seed, 1.0, rep, MAIN_ROWS[lang], "carriers_sym.json", None, FOCUS,
                                   False, ZERO, 0)
            rows = with_table(lang, run, a)
            rep["chars"] = sum(len(r["text"]) for r in rows)
            S._write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            print(lang, "main", json.dumps({k: rep[k] for k in ("rows", "chars", "target", "got", "drops")}, ensure_ascii=False))
        if lang in REPL_ROWS:
            pool = S.pure_rows(S._rows(os.path.join(a.mix, f"tts_{lang}.jsonl")), lang)
            room = {c: {"room": sum(k for _, k in pool[c])} for c in FOCUS}
            def run_r():
                global rep
                rep = {}
                return S.plan_lang(lang, room, "ttsCYP1r", a.seed, 1.0, rep, REPL_ROWS[lang], "carriers_sym3.json", None, FOCUS,
                                   False, ZERO, 0)
            rows = with_table(lang, run_r, a)
            rows = [r for r in rows if set(r["gclasses"]) <= set(FOCUS)]
            rep["supersede"] = S.assign_supersedes(rows, {c: pool[c] for c in FOCUS}, random.Random(a.seed))
            rep["chars"] = sum(len(r["text"]) for r in rows)
            S._write(os.path.join(a.out, f"plannedr_{lang}.jsonl"), rows)
            print(lang, "repl", json.dumps({"rows": len(rows), "chars": rep["chars"], "got": rep.get("got"), "supersede": rep["supersede"], "drops": rep.get("drops")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
