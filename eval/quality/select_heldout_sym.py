#!/usr/bin/env python3
"""Pick Granary-VoxPopuli index clips from HELD-OUT plenary days that carry pct / dec / cur / unit spans, to grow the
symbol-class held-out manifest (build_heldout_sym.py) toward --target spans per class and language.

usage: select_heldout_sym.py <lang> --work DIR --have-report report.json --out sel_<lang>.json
                             [--target 100] [--over 2.0] [--max-hours 1.25] [--years 3] [--per-session 8] [--per-template 3]
Reads (all local, under DIR): p3v2/heldout_<lang>.txt (the held-out list: only its DAYS are used), granvox/<lang>/index.jsonl
(the Granary text index), p3v2/<lang>/{cands_,rows,val_}*.jsonl (clips already cut or labelled: never picked again).
A clip is a candidate when its day is held out, it lasts 3-25 s (len(text)/15), and ml_cands finds a symbol span in it.
need[class] = (target - spans the manifest already has) * over (labelling and the judge lose about half). The years with the
most candidate clips are chosen first (a language-year is one tar stream to cut), then clips are taken class by class,
scarcest first, with a cap per session and per digit-masked template. Writes {"<year>": [utt_id, ...]} (the
`ml_p3.py::granvox_cut_ids --ids-file` format) and <out>.full.json with the stats."""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import volume_io as V  # noqa: E402

sys.path.insert(0, str(V.PIPE))
import census_common as CC  # noqa: E402
import ml_cands as M  # noqa: E402
import number_outcomes as N  # noqa: E402
import vox_heldout as VH  # noqa: E402

SYM = ("unit", "dec", "cur", "pct")  # scarcest first
REG = {"unit": VH.UNIT, "pct": None, "cur": None, "dec": None}


CUR = re.compile(r"\d[\d.,]*\s?(?:(?:mill\w+|milliard\w+|miljo\w+|miljard\w+|milh\w+|mil)\s?)?(?:(?:de|di|d'|van|von)\s?)?(?:"
                 + "|".join(sorted(map(re.escape, N._CUR | N._MINOR), key=len, reverse=True)) + r")\b|[€$£]\s?\d|\d\s?[€$£]")


def sym_counts(text: str, lang: str) -> collections.Counter:
    """Symbol spans a clip's index text carries: ml_cands kinds pct / dec / unit, plus currency (a number followed by a
    currency word is labelled kind int and counted cur by census_common.spoken_shape, so M.spans has no cur for it)."""
    c = collections.Counter(s.kind for s in M.spans(text, lang) if s.kind in SYM)
    n_cur = len(CUR.findall(text.lower()))
    if n_cur:
        c["cur"] = max(c["cur"], n_cur)
    if re.search(VH.UNIT, text.lower()):
        c["unit"] = max(c["unit"], 1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--work", required=True)
    ap.add_argument("--have-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--over", type=float, default=2.0)
    ap.add_argument("--max-hours", type=float, default=1.25)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--per-session", type=int, default=8)
    ap.add_argument("--per-template", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    w, lang = Path(a.work), a.lang
    _, units = CC.read_held_list(w / "p3v2" / f"heldout_{lang}.txt")
    days = {u[4:] for u in units if u.startswith("day:")}
    have = json.load(open(a.have_report))[lang]["symbol_spans"]
    need = {c: max(0, a.target - have.get(c, 0)) * a.over for c in SYM}
    used = set()
    for p in glob.glob(str(w / "p3v2" / lang / "*.jsonl")):
        if re.search(r"/(cands_|rows|val_)", p):
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    used.add(json.loads(l)["id"])
    cand = []
    for l in open(w / "granvox" / lang / "index.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["day"] not in days or r["utt_id"] in used or not 3.0 <= len(r["text"]) / 15 <= 25.0:
            continue
        c = sym_counts(r["text"], lang)
        if c:
            cand.append((r, c))
    by_year = collections.Counter(r["day"][:4] for r, _ in cand)
    # the years that cover the needs best: clips carrying a still-needed class, per year
    score = {y: sum(1 for r, c in cand if r["day"][:4] == y and any(need[k] > 0 and c[k] for k in SYM)) for y in by_year}
    years = [y for y, _ in sorted(score.items(), key=lambda t: -t[1])[:a.years]]
    rng = random.Random(f"{a.seed}-{lang}")
    pool = [(r, c) for r, c in cand if r["day"][:4] in years]
    rng.shuffle(pool)
    tmpl = lambda r: re.sub(r"\d+", "#", re.sub(r"[^\w\s]", "", r["text"].lower()))[:60]  # noqa: E731
    t_n, s_n, got, sel, hours = collections.Counter(), collections.Counter(), collections.Counter(), [], 0.0
    taken = set()
    for k in SYM:
        for r, c in sorted(pool, key=lambda t: -t[1][k]):
            if got[k] >= need[k] or hours >= a.max_hours * 3600:
                break
            u = r["utt_id"]
            if not c[k] or u in taken or t_n[tmpl(r)] >= a.per_template or s_n[u.rsplit("_", 1)[0]] >= a.per_session:
                continue
            taken.add(u)
            t_n[tmpl(r)] += 1
            s_n[u.rsplit("_", 1)[0]] += 1
            got.update(c)
            hours += len(r["text"]) / 15
            sel.append(r)
    out = collections.defaultdict(list)
    for r in sel:
        out[r["day"][:4]].append(r["utt_id"])
    st = {"lang": lang, "held_days": len(days), "candidates": len(cand), "years_all": dict(by_year), "years_chosen": years,
          "need_spans": need, "est_spans_in_selection": {k: got[k] for k in SYM}, "clips": len(sel),
          "est_hours": round(hours / 3600, 2), "days_used": len({r["day"] for r in sel})}
    print(json.dumps(st, ensure_ascii=False))
    json.dump(dict(out), open(a.out, "w"))
    json.dump(dict(st, ids_by_year=dict(out)), open(a.out + ".full.json", "w"))


if __name__ == "__main__":
    main()
