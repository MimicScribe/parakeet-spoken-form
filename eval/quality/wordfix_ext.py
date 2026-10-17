#!/usr/bin/env python3
"""Word-repair pass helpers (pipeline ml_wordfix.py): judge rejects whose only fault is a non-number word.

usage:
  wordfix_ext.py cands <lang> <pass> [--upload]      rows of <pass> rejected for a word (lost_word / extra_word / wer)
                                                     with every number span agreeing -> <work>/<lang>/<pass>/wordfix_ext.jsonl
                                                     ({"id", "text", "words"}: the row's own transcript); --upload puts it
                                                     at ml/p3v2/<lang>/wordfix_<pass>.jsonl
  wordfix_ext.py seed <lang> <src pass> <new pass>   before run_judge on the new pass: copies the src pass's transcript
                                                     cache and adds each repaired row's transcript under its new id, so
                                                     no audio is transcribed again
  wordfix_ext.py report <lang> <src pass> <new pass> [--gate gate_all.json]
                                                     classes the repaired + judge-kept rows carry (thin cells first)
Nothing here is paid: transcripts come from the existing cache.
"""
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("JUDGE_WORK", "/tmp/spoken_form_judge"))
PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"
WORD_REASONS = {"lost_word", "extra_word", "wer", "negation"}  # negation: a lost / invented "não" / "nicht" (2026-10-17)
sys.path.insert(0, str(HERE))


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def last_per_id(p):
    out = {}
    for j in jl(p):
        out[j["id"]] = j
    return out


SFX = re.compile(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$")
VID = re.compile(r"[a-z]{2}\d+_\d+_(.{11})_")
DAY = re.compile(r"(\d{8})-\d{4}-PLENARY")


def heldout_test(lang):
    """-> id -> bool: is the row held out, by build4's own units (exact id, VoxPopuli segment suffix, YODAS video,
    plenary day), over ml/p3v2/heldout_<lang>.txt plus every val_* pass id. A repaired copy gets a new id (<id>_w), so
    a held-out row must never be repaired: build4 could not tell it from a training row (review 2026-10-17)."""
    d = WORK / "_heldout"
    d.mkdir(parents=True, exist_ok=True)
    hp = d / f"heldout_{lang}.txt"
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"ml/p3v2/heldout_{lang}.txt", str(hp)], cwd=PIPE,
                       capture_output=True, text=True)
    assert r.returncode == 0 and hp.exists(), f"{lang}: held-out list missing"
    held = {l.strip() for l in open(hp) if l.strip()}
    for p in (WORK / lang).glob("val_*/val_*.jsonl"):
        held |= {x["id"] for x in jl(p)}
    units = [(rx, {m.group(1) for m in map(rx.search if rx is SFX else rx.match, held) if m}) for rx in (SFX, VID, DAY)]

    def test(i):
        if i in held:
            return True
        for rx, us in units:
            m = rx.search(i) if rx is SFX else rx.match(i)
            if m and m.group(1) in us:
                return True
        return False

    return test


def cands(lang, name, upload=False):
    w = WORK / lang / name
    jr = jl(w / "jr" / "judge_rows.jsonl")
    ext = last_per_id(w / "ext.jsonl")
    out, n = [], collections.Counter()
    is_held = heldout_test(lang)
    for r in jr:
        if r.get("reason") not in WORD_REASONS:
            continue
        if is_held(r["id"]):
            n["heldout"] += 1
            continue
        n[r["reason"]] += 1
        if any(s.get("v") != "agree" for s in r.get("spans") or []):
            n["span_not_agree"] += 1
            continue
        a = ext.get(r["id"])
        if not a or a.get("status") != "completed":
            continue
        out.append({"id": r["id"], "text": a.get("text"), "words": [{"w": x["w"]} for x in a.get("words") or []]})
    p = w / "wordfix_ext.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{lang}/{name}: {len(out)} word-reject rows with agreeing numbers -> {p}  {dict(n)}")
    if upload:
        subprocess.run(["modal", "volume", "put", "--force", VOLUME, str(p), f"ml/p3v2/{lang}/wordfix_{name}.jsonl"],
                       cwd=PIPE, check=True, capture_output=True)
        print(f"uploaded ml/p3v2/{lang}/wordfix_{name}.jsonl")


def seed(lang, src, new):
    dst = WORK / lang / new
    dst.mkdir(parents=True, exist_ok=True)
    rows_p = dst / f"{new}.jsonl"
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"ml/p3v2/{lang}/{new}.jsonl", str(rows_p)],
                       cwd=PIPE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = jl(rows_p)
    ext = last_per_id(WORK / lang / src / "ext.jsonl")
    lines = list(ext.values())
    miss = 0
    for x in rows:
        a = ext.get(x["wordfix"]["of"])
        if a is None:
            miss += 1
            continue
        lines.append(dict(a, id=x["id"]))
    with open(dst / "ext.jsonl", "w", encoding="utf-8") as f:
        for a in lines:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"{lang}/{new}: {len(rows)} rows, transcript cache seeded ({miss} without a source transcript)")


BIG = re.compile(r"milli|miljo|miljar|bilh|billon|milhõ|milhã|milion|miliard", re.I)


def thin_classes(row, lang, gate):
    """Census cells a row's spans fill (thin ones named after their bar: "year<40", "big_scale", ...)."""
    import data_gate as G

    spans, text = row.get("spans") or [], row.get("text") or ""
    after = []
    for a, b in row.get("num_chars") or []:
        nxt = text[b:].split()
        after.append(nxt[0] if nxt else "")

    g = (gate or {}).get(lang, {}).get("detail", {})
    years = {int(k): v for k, v in g.get("year_values", {}).items()}
    smalls = {int(k): v for k, v in g.get("int_small_values", {}).items()}
    out = []
    for n, sp in enumerate(spans):
        cls, v = G.span_class(sp)
        rd = (sp.get("reading") or "").lower() + " " + (after[n] if n < len(after) else "")
        if cls == "year":
            want = 40 if 1990 <= v <= 2030 or v in (2035, 2040, 2050) else 10 if 2031 <= v <= 2060 else 0
            out.append(f"year<{want}" if years.get(v, 0) < want else "year_ok")
        elif (cls == "int_big" and v >= 10**6) or BIG.search(rd):
            out.append("big_scale")
        elif cls == "int_small" and v <= 100:
            out.append("int0_100<10" if smalls.get(v, 0) < 10 else "int_small")
        elif cls == "cur":
            out.append("cur_cents" if re.search(r"\d[.,]\d\d\b", sp.get("text", "")) or "cent" in rd else "cur")
        elif cls in ("phone", "date", "clock", "dec", "pct"):
            out.append(cls)
        else:
            out.append(cls if cls not in ("int_small",) else "int_small")
    return out


THIN = ("year<40", "year<10", "big_scale", "phone", "cur_cents", "date", "clock", "dec", "pct", "int0_100<10")


def report(lang, src, new, gate_p=None):
    gate = json.load(open(gate_p)) if gate_p and os.path.exists(gate_p) else None
    rows = jl(WORK / lang / new / f"{new}.jsonl")
    jr = {r["id"]: r for r in jl(WORK / lang / new / "jr" / "judge_rows.jsonl")}
    kept = [x for x in rows if jr.get(x["id"], {}).get("reason", "x") is None]
    cls, rows_thin, h = collections.Counter(), 0, 0.0
    for x in kept:
        cs = thin_classes(x, lang, gate)
        for c in set(cs):
            cls[c] += 1
        if any(c in THIN for c in cs):
            rows_thin += 1
        h += x["duration"]
    print(json.dumps({"lang": lang, "repaired": len(rows), "judge_kept": len(kept), "hours": round(h / 3600, 3),
                      "rows_with_thin_cell": rows_thin, "classes": dict(cls.most_common())}, ensure_ascii=False))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[:1] == ["cands"]:
        cands(a[1], a[2], "--upload" in sys.argv)
    elif a[:1] == ["seed"]:
        seed(a[1], a[2], a[3])
    elif a[:1] == ["report"]:
        g = sys.argv[sys.argv.index("--gate") + 1] if "--gate" in sys.argv else None
        report(a[1], a[2], a[3], g)
    else:
        print(__doc__)
