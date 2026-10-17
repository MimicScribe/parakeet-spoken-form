#!/usr/bin/env python3
"""Carve a rare-class held-out set from labelled real-speech rows (the first held-out lists
had 0 spans in every rare class).

usage: carve_heldout.py <lang>[,<lang>...] [--target 10] [--max-share 0.25] [--work DIR] [--dry]
                        [--kinds unit,dec --max-share 0.10 --avoid phone]
--kinds (2026-10-17, thin decision sets for common classes): carve only these kinds (int / pct / dec / ord allowed),
richest groups first, counted in data_gate's classes (a dec read as a price is cur); the cap is then --max-share of
the class's spans still in TRAINING, checked for every listed kind on each group; --avoid skips any group holding a
span of those labeller kinds.
Per language: reads every labelling pass (ml/p3v2/<lang>/rows*.jsonl) and the current heldout_<lang>.txt, then for each
rare kind (not int/pct/dec/ord) adds whole groups (a YODAS video, a VoxPopuli plenary day; leakage check 2026-10-15)
until that kind has >= --target held-out spans or would exceed --max-share of its real spans. Groups already held
out count first. Appends "# group <src>:<key>" + ids to heldout_<lang>.txt and uploads it (unless --dry).
Run BEFORE windows2 / build_mix4: both drop held-out ids and assert that no window or pair uses one."""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"
COMMON = {"int", "pct", "dec", "ord"}


def vget(remote, local):
    return subprocess.run(["modal", "volume", "get", "--force", VOLUME, remote, str(local)], cwd=PIPE,
                          capture_output=True).returncode == 0


def group(r):
    i = str(r["id"])
    if r.get("src") == "vox":
        m = re.match(r"(\d{8})-", i)
        return f"vox:{m.group(1)}" if m else None
    if r.get("src") == "yodas":
        m = re.match(r"[a-z]{2}\d+_\d+_(.{11})_", i)
        return f"yodas:{m.group(1)}" if m else None
    return None  # FLEURS / D2 rows: sentence-disjoint splits already exist, not carved here


def kind_counts(rs, cls=None):
    """Spans per kind (the labeller kind, or cls(row, span): the census class)."""
    return collections.Counter(cls(r, s) if cls else s["kind"] for r in rs for s in r.get("spans", []))


def gate_class(lang):
    """data_gate's class of a span (dec read as a price is cur, etc.): --kinds counts what the decision-set check counts."""
    import data_gate as G

    def f(r, sp):
        t = r.get("text", "")
        return G.spoken_shape(t, sp, *G.morph_class(t, sp, *G.span_class(sp), lang), lang)
    return f


def plan(lang, rows, held, target, max_share, only=None, avoid=frozenset(), cls=None):
    """Groups to add to the held-out list -> (add {group: ids}, held-out spans per kind after, real spans per kind).
    only None: every rare kind, cap = max_share of ALL its real spans (2026-10-15 behaviour). only given: those
    kinds, richest groups first, and a group is taken only if every listed kind stays within max_share of the spans it
    had in training before this run. cls: span -> class (kind_counts); avoid is checked on the labeller kinds."""
    by_group = collections.defaultdict(list)
    for r in rows:
        g = group(r)
        if g:
            by_group[g].append(r)
    # a group with any held-out id is held out whole (mix_build.build4 S5: the video / plenary day is the unit)
    held_g = {g for g, rs in by_group.items() if any(r["id"] in held for r in rs)}
    total = kind_counts(rows, cls)
    have = kind_counts([r for r in rows if r["id"] in held or group(r) in held_g], cls)
    add = {}
    if only is None:
        order, caps = [k for k in sorted(total, key=lambda k: total[k]) if k not in COMMON], None
    else:
        order = list(only)
        caps = {k: have[k] + int((total[k] - have[k]) * max_share) for k in order}
    for k in order:
        cap = int(total[k] * max_share) if caps is None else caps[k]
        nk = lambda g: kind_counts(by_group[g], cls)[k]  # noqa: E731
        cands = [g for g, rs in by_group.items() if g not in add and g not in held_g and nk(g)
                 and not (avoid and set(kind_counts(rs)) & avoid)]
        # deterministic order (sha of lang+group); rare mode: smallest groups first so little else leaves training;
        # --kinds mode: richest in the kind first (fewest groups leave training per held-out span)
        h = lambda g: hashlib.sha256(f"{lang}{g}".encode()).hexdigest()  # noqa: E731
        cands.sort(key=(lambda g: (len(by_group[g]), h(g))) if caps is None else (lambda g: (-nk(g), len(by_group[g]), h(g))))
        for g in cands:
            if have[k] >= target:
                break
            gk = kind_counts(by_group[g], cls)
            if have[k] + gk[k] > cap:  # cap 0 (< 4 real spans): the class stays in training, "not claimed"
                continue
            if caps is not None and any(have[kk] + gk[kk] > caps[kk] for kk in caps):
                continue
            add[g] = [r["id"] for r in by_group[g]]
            for kk, v in gk.items():
                have[kk] += v
    return add, have, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("langs")
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--max-share", type=float, default=0.25)
    ap.add_argument("--work", default="/tmp/spoken_form_carve")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--avoid", default="")
    a = ap.parse_args()
    for lang in a.langs.split(","):
        w = Path(a.work) / lang
        w.mkdir(parents=True, exist_ok=True)
        # review H3: every labelling pass on the volume, and a failed fetch stops the carve
        ls = subprocess.run(["modal", "volume", "ls", VOLUME, f"ml/p3v2/{lang}"], cwd=PIPE, capture_output=True, text=True)
        names = sorted({m.group(1) for m in re.finditer(r"(?:^|/|\s)(rows[a-z0-9_]*)\.jsonl", ls.stdout)})  # not pre_rows_*
        # validation-only passes (val_*.jsonl): build4 holds their ids and days / videos out, so they count as held out
        vals = sorted({m.group(1) for m in re.finditer(r"(?:^|/|\s)(val_[a-z0-9_]*)\.jsonl", ls.stdout)})
        assert names, f"{lang}: no rows*.jsonl on the volume"
        for n in names + vals:
            assert vget(f"ml/p3v2/{lang}/{n}.jsonl", w / f"{n}.jsonl"), f"{lang}: fetch of {n}.jsonl failed"
        if not vget(f"ml/p3v2/heldout_{lang}.txt", w / "heldout.txt"):  # a language's first held-out list starts empty
            (w / "heldout.txt").write_text(f"# carve_heldout.py first list for {lang}\n")
        held = {l.strip() for l in open(w / "heldout.txt") if l.strip() and not l.startswith("#")}
        rej = set()
        for stale in [*w.glob("rows*.jsonl"), *w.glob("val_*.jsonl")]:
            if stale.stem not in names + vals:
                stale.unlink()
        if vget(f"ml/p3v2/{lang}/reject_ids.txt", w / "reject.txt"):
            rej = {l.strip() for l in open(w / "reject.txt") if l.strip()}
        val = [json.loads(l) for n in vals for l in open(w / f"{n}.jsonl") if l.strip()]
        held |= {r["id"] for r in val}
        # one row per id across passes, as mix_build._one_per_id: a regional pass first, else the first in name order
        paths = sorted(glob.glob(str(w / "rows*.jsonl")), key=lambda p: (not p.endswith("_regional.jsonl"), p))
        rows = list({r["id"]: r for p in reversed(paths) for l in open(p) if l.strip() for r in [json.loads(l)]}.values())
        seen = {x["id"] for x in rows}
        rows += [r for r in val if r["id"] not in seen]
        rows = [r for r in rows if r["id"] not in rej and not any(sp.get("by") == "ext" for sp in r.get("spans", []))]
        add, have, total = plan(lang, rows, held, a.target, a.max_share, only=a.kinds.split(",") if a.kinds else None,
                                avoid=set(filter(None, a.avoid.split(","))), cls=gate_class(lang) if a.kinds else None)
        print(f"{lang}: +{len(add)} groups, +{sum(map(len, add.values()))} rows held out")
        shown = a.kinds.split(",") if a.kinds else [k for k in sorted(total) if k not in COMMON]
        print("   held-out spans per kind (now / real total):", {k: f"{have[k]}/{total[k]}" for k in shown})
        if add and not a.dry:
            with open(w / "heldout.txt", "a") as f:
                what = f"kinds {a.kinds}, max share of training {a.max_share}" if a.kinds else \
                    f"rare classes, max share {a.max_share}"
                f.write(f"\n# carve_heldout.py {datetime.date.today()} ({what}, target {a.target})\n")
                for g, ids in sorted(add.items()):
                    f.write(f"# group {g}\n" + "\n".join(ids) + "\n")
            subprocess.run(["modal", "volume", "put", "--force", VOLUME, str(w / "heldout.txt"),
                            f"ml/p3v2/heldout_{lang}.txt"], cwd=PIPE, check=True, capture_output=True)


if __name__ == "__main__":
    main()
