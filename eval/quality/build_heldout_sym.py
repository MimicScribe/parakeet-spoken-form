#!/usr/bin/env python3
"""Held-out manifest for the symbol classes (pct, dec, dec_pair, cur, unit): every labelled, judged, non-rejected real-speech
row of a HELD-OUT day / video that carries one of those spans, nothing from a training session.

usage: build_heldout_sym.py [--langs de,es,fr,it,nl,pt] [--work DIR] [--base-mix mix4final9b] [--train-mixes mix4final9b,mix4final8]
                            [--out DIR] [--put eval/heldout_sym_v1]

Pools (all of them already labelled and judged, nothing is decoded or labelled here):
  base   <base-mix>/heldout_<lang>.jsonl rows with a symbol span (what the decision sets already hold);
  extra  rows*.jsonl / val_*.jsonl rows (one per id, regional pass first, as carve_heldout / mix_build) whose id, day or video
         is in ml/p3v2/heldout_<lang>.txt (census_common.read_held_list) or whose pass is validation-only, that are not judge
         rejects and are not in base.
No-overlap checks against the training parts of every --train-mixes build (p3, win, pairs, plain, seam, seam_long, win15, d2,
fleurs_train): an extra row is dropped when its id, audio basename, held-out unit (day / video) or normalised label text
appears in a training part; base rows are only reported. The report counts what was dropped and why.
Output rows keep the schema of heldout_<lang>.jsonl and add `sym_origin` ("base" | "extra:<pass>") and `sym_classes`
(census class -> spans). <out>/<lang>.jsonl and <out>/all.jsonl (rows get `lang`); --put uploads to a NEW volume path
(refuses an existing one)."""
import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volume_io as V  # noqa: E402

sys.path.insert(0, str(V.PIPE))
import census_common as CC  # noqa: E402

LANGS = "de es fr it nl pt".split()
SYMBOL = ("pct", "dec", "dec_pair", "cur", "unit")
CLASSES = ["pct", "dec", "dec_pair", "cur", "unit", "int_big", "int_small", "year", "phone", "clock", "alnum"]
TRAIN_PARTS = ("p3", "win", "pairs", "plain", "seam", "seam_long", "win15")
TRAIN_FILES = ("d2", "d2p", "fleurs_train_multi")


def fetch(work: Path, remote: str) -> Path:
    """Local copy of a volume file (ml/ prefix dropped: work/p3v2/<lang>/rows.jsonl, work/<mix>/p3_<lang>.jsonl)."""
    p = work / (remote[3:] if remote.startswith("ml/") else remote)
    if not (p.exists() and p.stat().st_size):
        p.parent.mkdir(parents=True, exist_ok=True)
        V.vget(remote, p)
    return p


def rd(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def norm(t: str) -> str:
    return re.sub(r"[^\w]+", " ", str(t).lower()).strip()


def base_name(audio: str) -> str:
    return re.sub(r"_p\d+$", "", os.path.basename(str(audio)).rsplit(".", 1)[0])


def ids_of(r: dict) -> set:
    """Every identifier a row carries: its id, audio basename, parent(s); pair rows carry two audio files."""
    out = {str(r.get("id", "")), base_name(r.get("audio_filepath", "")), str(r.get("parent", ""))}
    out |= {str(x) for x in r.get("parents", [])}
    for k in ("a", "b"):
        if isinstance(r.get(k), dict):
            out.add(base_name(r[k].get("audio_filepath", "")))
    return {x for x in out if x}


def unit_of(i: str):
    u = CC.held_unit(i)
    if u:
        return u
    m = re.match(r"^(\d{8})-\d{4}-[A-Z]", i)
    return ("day:" + m.group(1)) if m else None


def training_index(work: Path, mixes: list, lang: str) -> dict:
    ids, units, texts = set(), set(), set()
    for mix in mixes:
        files = [f"{mix}/{p}_{lang}.jsonl" for p in TRAIN_PARTS] + [f"{mix}/{f}.jsonl" for f in TRAIN_FILES]
        for f in files:
            for r in rd(fetch(work, f)):
                if str(r.get("lang", lang))[:2] != lang:  # the same plenary day in another language is another audio track
                    continue
                for i in ids_of(r):
                    ids.add(i)
                    u = unit_of(i)
                    if u:
                        units.add(u)
                for t in (r.get("text"), r.get("caption")) + tuple(x.get("text") for x in (r.get("a"), r.get("b")) if isinstance(x, dict)):
                    if t and len(norm(t).split()) >= 4:
                        texts.add(norm(t))
    return {"ids": ids, "units": units, "texts": texts}


def row_classes(r: dict, lang: str) -> collections.Counter:
    return collections.Counter(CC.gate_class(r, sp, lang) for sp in r.get("spans") or [])


def build(lang: str, work: Path, base_mix: str, train_mixes: list) -> tuple:
    w = work / "p3v2" / lang
    fetch(work, f"ml/p3v2/heldout_{lang}.txt")
    listing = V.vlist(f"ml/p3v2/{lang}")
    names = sorted(n for n in listing if re.fullmatch(r"(rows[a-z0-9_]*|val_[a-z0-9_]*)\.jsonl", n))
    for n in names:
        fetch(work, f"ml/p3v2/{lang}/{n}")
    rej_p = fetch(work, f"ml/p3v2/{lang}/reject_ids.txt")
    rej = {l.strip() for l in open(rej_p) if l.strip()} if rej_p.exists() else set()
    held, units = CC.read_held_list(work / "p3v2" / f"heldout_{lang}.txt")
    vals = [n for n in names if n.startswith("val_")]
    val_rows = [(n, r) for n in vals for r in rd(w / n)]
    held |= {r["id"] for _, r in val_rows}
    passes = sorted((w / n for n in names if n.startswith("rows")), key=lambda p: (not p.name.endswith("_regional.jsonl"), str(p)))
    row_src = {}
    for p in reversed(passes):  # carve_heldout / mix_build._one_per_id: a regional pass first, else the first in name order
        for r in rd(p):
            row_src[r["id"]] = (p.stem, r)
    for n, r in val_rows:
        row_src.setdefault(r["id"], (n[:-6], r))
    in_held = lambda i: i in held or unit_of(i) in units  # noqa: E731
    base = rd(work / base_mix / f"heldout_{lang}.jsonl")
    base_ids = {r["id"] for r in base}
    tr = training_index(work, train_mixes, lang)
    rep = collections.Counter()
    out = []
    for r in base:
        c = row_classes(r, lang)
        if not any(c[k] for k in SYMBOL):
            continue
        rep["base_rows_with_symbol"] += 1
        if (ids_of(r) & tr["ids"]) or unit_of(r["id"]) in tr["units"] or norm(r.get("text", "")) in tr["texts"]:
            rep["base_overlap_reported"] += 1
        dup = norm(r.get("text", "")) in tr["texts"]  # a short generic phrase also trained on, other video: flagged, kept
        out.append(dict(r, sym_origin="base", sym_classes={k: v for k, v in c.items() if v}, **({"sym_text_dup_train": True} if dup else {})))
    units_hit = set()
    for i, (name, r) in sorted(row_src.items()):
        if i in base_ids or not in_held(i):
            continue
        if i in rej:
            rep["extra_rejected"] += 1
            continue
        if not (r.get("spans") and r.get("num_chars") and r.get("caption") and r.get("audio_filepath")):
            rep["extra_incomplete"] += 1
            continue
        c = row_classes(r, lang)
        if not any(c[k] for k in SYMBOL):
            continue
        why = "id" if ids_of(r) & tr["ids"] else "unit" if unit_of(i) in tr["units"] else \
            "text" if norm(r.get("text", "")) in tr["texts"] or norm(r.get("caption", "")) in tr["texts"] else ""
        if why:
            rep["extra_dropped_" + why] += 1
            if why == "unit":
                units_hit.add(unit_of(i))
            continue
        rep["extra_kept"] += 1
        out.append(dict(r, sym_origin="extra:" + name, sym_classes={k: v for k, v in c.items() if v}))
    rep["overlap_units"] = len(units_hit)
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--work", default="/tmp/spoken_form_heldout_sym")
    ap.add_argument("--base-mix", default="mix4final9b")
    ap.add_argument("--train-mixes", default="mix4final9b,mix4final8")
    ap.add_argument("--out", default="")
    ap.add_argument("--put", default="", help="NEW volume path, e.g. eval/heldout_sym_v1 (refused when it exists)")
    a = ap.parse_args()
    work = Path(a.work)
    out_dir = Path(a.out or work / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows, table = [], {}
    for lang in a.langs.split(","):
        rows, rep = build(lang, work, a.base_mix, a.train_mixes.split(","))
        with open(out_dir / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        allrows += [dict(r, lang=lang) for r in rows]
        tot = collections.Counter()
        for r in rows:
            tot.update(r["sym_classes"])
        spans = collections.Counter()
        for r in rows:
            spans.update(row_classes(r, lang))
        table[lang] = {"rows": len(rows), "report": dict(rep), "symbol_spans": {k: spans[k] for k in SYMBOL},
                       "other_spans": {k: spans[k] for k in CLASSES if k not in SYMBOL}}
        print(lang, json.dumps(table[lang], ensure_ascii=False))
    with open(out_dir / "all.jsonl", "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(table, open(out_dir / "report.json", "w"), indent=1)
    if a.put:
        have = V.vlist(a.put.rsplit("/", 1)[0])
        if a.put.rsplit("/", 1)[1] in have:
            sys.exit(f"{a.put} already exists on the volume: pick a new path")
        for p in sorted(out_dir.glob("*.jsonl")):
            V.put(p, f"{a.put}/{p.name}")
        V.put(out_dir / "report.json", f"{a.put}/report.json")
        print("uploaded to", a.put)


if __name__ == "__main__":
    main()
