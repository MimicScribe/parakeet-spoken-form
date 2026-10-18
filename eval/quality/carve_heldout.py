#!/usr/bin/env python3
"""Carve a rare-class held-out set from labelled real-speech rows (the first held-out lists
had 0 spans in every rare class).

usage: carve_heldout.py <lang>[,<lang>...] [--target 20] [--max-share 0.25] [--work DIR] [--dry]
                        [--kinds unit,dec --max-share 0.10 --avoid phone]
--kinds (2026-10-17, thin decision sets for common classes): carve only these kinds (int / pct / dec / ord allowed),
richest groups first, counted in data_gate's classes (a dec read as a price is cur); the cap is then --max-share of
the class's spans still in TRAINING, checked for every listed kind on each group; --avoid skips any group holding a
span of those labeller kinds.
Per language: reads every labelling pass (ml/p3v2/<lang>/rows*.jsonl) and the current heldout_<lang>.txt, then for each
rare kind (not int/pct/dec/ord) adds whole groups (a YODAS video, a parliament day: census_common.held_unit, the unit
mix_build.build4 holds out)
until that kind has >= --target held-out spans or would exceed --max-share of its real spans. Groups already held
out count first. Appends "# group <src>:<key>" + ids to heldout_<lang>.txt and uploads it (unless --dry).
Rows any span of which the second transcript decided (census_common.EXT_BY) never enter the carve: build4 keeps them
out of the held-out set too. The held-out and reject lists are read fail-closed (volume_io.read_shared): a list that
exists on the volume but cannot be fetched stops the carve, and the held-out list is only uploaded after it was read
(or the volume was seen to have none).
Run BEFORE windows2 / build_mix4: both drop held-out ids and assert that no window or pair uses one."""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import volume_io as V

PIPE = V.PIPE
VOLUME = V.VOLUME
sys.path.insert(0, str(PIPE))
from census_common import ext_row, held_unit, read_held_list  # noqa: E402  (the build's definitions)

COMMON = {"int", "pct", "dec", "ord"}
vget = V.vget


def group(r):
    """The held-out unit of a row (census_common.held_unit, as build4 holds it out), named vox:<day> / yodas:<video>;
    None for FLEURS / D2 rows (sentence-disjoint splits already exist, not carved here)."""
    u = held_unit(r["id"])
    if u is None:
        return None
    k, v = u.split(":", 1)
    return ("vox:" if k == "day" else "yodas:") + v


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


def plan(lang, rows, held, target, max_share, only=None, avoid=frozenset(), cls=None, held_units=frozenset()):
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
    # and a unit the list holds out with no id of it here ("# group vox:<day>" lines, e.g. vox_heldout days)
    held_g |= {g for g in by_group if g.replace("vox:", "day:", 1).replace("yodas:", "vid:", 1) in held_units}
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
        # every labelling pass on the volume, and a failed fetch stops the carve
        listing = V.vlist(f"ml/p3v2/{lang}")  # a failed listing stops the carve
        names = sorted(n[:-6] for n in listing if re.fullmatch(r"rows[a-z0-9_]*\.jsonl", n))  # not pre_rows_*
        # validation-only passes (val_*.jsonl): build4 holds their ids and days / videos out, so they count as held out
        vals = sorted(n[:-6] for n in listing if re.fullmatch(r"val_[a-z0-9_]*\.jsonl", n))
        assert names, f"{lang}: no rows*.jsonl on the volume"
        for n in names + vals:
            assert vget(f"ml/p3v2/{lang}/{n}.jsonl", w / f"{n}.jsonl"), f"{lang}: fetch of {n}.jsonl failed"
        # fail closed: a held-out list the volume has but this run cannot read is an error, never "a first list"
        first = not V.read_shared(f"ml/p3v2/heldout_{lang}.txt", w / "heldout.txt")
        if first:  # the volume has no list for this language yet: it starts empty
            (w / "heldout.txt").write_text(f"# carve_heldout.py first list for {lang}\n")
        held, held_units = read_held_list(w / "heldout.txt")  # "# group" lines hold their unit out too
        rej = set()
        for stale in [*w.glob("rows*.jsonl"), *w.glob("val_*.jsonl")]:
            if stale.stem not in names + vals:
                stale.unlink()
        if V.read_shared(f"ml/p3v2/{lang}/reject_ids.txt", w / "reject.txt"):
            rej = {l.strip() for l in open(w / "reject.txt") if l.strip()}
        else:
            print(f"{lang}: no reject_ids.txt on the volume (no judge run yet): no rows rejected")
        val = [json.loads(l) for n in vals for l in open(w / f"{n}.jsonl") if l.strip()]
        held |= {r["id"] for r in val}
        # one row per id across passes, as mix_build._one_per_id: a regional pass first, else the first in name order
        paths = sorted(glob.glob(str(w / "rows*.jsonl")), key=lambda p: (not p.endswith("_regional.jsonl"), p))
        rows = list({r["id"]: r for p in reversed(paths) for l in open(p) if l.strip() for r in [json.loads(l)]}.values())
        seen = {x["id"] for x in rows}
        rows += [r for r in val if r["id"] not in seen]
        rows = [r for r in rows if r["id"] not in rej and not ext_row(r)]
        add, have, total = plan(lang, rows, held, a.target, a.max_share, only=a.kinds.split(",") if a.kinds else None,
                                avoid=set(filter(None, a.avoid.split(","))), cls=gate_class(lang) if a.kinds else None,
                                held_units=held_units)
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
            if first and f"heldout_{lang}.txt" in V.vlist("ml/p3v2"):  # another run wrote one since: never overwrite it
                raise V.VolumeReadError(f"heldout_{lang}.txt appeared on the volume during the carve: rerun")
            V.put(w / "heldout.txt", f"ml/p3v2/heldout_{lang}.txt")


if __name__ == "__main__":
    main()
