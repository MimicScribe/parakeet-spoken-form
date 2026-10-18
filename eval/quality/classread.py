#!/usr/bin/env python3
"""Per-language x class held-out readout for any checkpoint (one command).

usage: classread.py <run>@<step>[__<mix>] [<run>@<step> ...] [--mix mix4final8] [--work DIR] [--langs de,es,fr,it,nl,pt]
                    [--refs ultra,spike_enc0_all@500] [--json out.json] [--boot 1000] [--min-n 5]

Reads the held-out hyps probe_heldout / eval_ckpt.sh wrote (hyps/<run>@<step>__<mix>/heldout_<lang>.jsonl; fetched with the
modal CLI into --work when not there yet) and scores every caption span with the SAME positional rules as
score_heldout.score_spans (words-right / digits-right / wrong / dropped / bare price; the +-1 word slack; the same READINGS).
Per language x census class (census_common.gate_class: pct dec dec_pair cur unit int_big int_small year phone clock alnum,
everything else "other", and ALL) and per arm it reports

  n          spans (the same spans for every arm: the intersection)
  wshare     share of spans WRITTEN in words (converted + words_wrong)
  VR         value-right overall (words or digits with the right value)
  VRw / VRd  value-right among the spans the arm wrote in words / in digits
  unitdrop   pct / cur / unit: spans whose label SPEAKS the unit (percent, euro, kilometre ...) where the hyp region holds
             neither the unit word nor a unit symbol or abbreviation (%, EUR, km ...)
  decdrop    dec: spans whose label speaks the decimal word (komma, coma, virgule ...) where the hyp region holds neither
             the decimal word nor a digit decimal separator

Every number is also given as a paired difference to each --refs arm (stock = ultra, and spike_enc0_all@500 by default) with
a recording-cluster bootstrap CI95 (clusters = FLEURS sentence / YODAS video / parliament session, resampled within
language). The reference arms are always printed first. Nothing here touches the volume except reading hyps / captions."""
import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N  # noqa: E402
import score_heldout as SH  # noqa: E402

PIPE = str(Path(__file__).resolve().parents[2] / "pipeline")
sys.path.insert(0, PIPE)
import census_common as CC  # noqa: E402

M, D = SH.M, SH.D
LANGS = "de es fr it nl pt".split()
CLASSES = ["pct", "dec", "dec_pair", "cur", "unit", "int_big", "int_small", "year", "phone", "clock", "alnum"]
RIGHT = ("WORDS_RIGHT_VALUE", "DIGITS_RIGHT_VALUE")
DEFAULT_REFS = ("ultra", "spike_enc0_all@500")
SYMBOLS = set("%€$£°‰")
METRICS = ("wshare", "vr", "vrw", "vrd", "unitdrop", "decdrop")


def cluster(rid: str, lang: str) -> str:
    """Recording of a row id: FLEURS sentence, YODAS video, parliament session (the unit a bootstrap must resample)."""
    if rid.startswith("fleurs"):
        return f"{lang}:fleurs:{rid.split('_')[1]}"
    m = re.match(r"^[a-z]{2}\d{3}_\d{8}_(.{11})_", rid)
    if m:
        return f"{lang}:yt:{m.group(1)}"
    m = re.match(r"^(\d{8}-\d{4}-[A-Z]+)", rid)
    if m:
        return f"{lang}:vox:{m.group(1)}"
    return f"{lang}:other:{rid}"


def klass(row: dict, sp: dict, lang: str) -> str:
    c = CC.gate_class(row, sp, lang)
    return c if c in CLASSES else "other"


def _has_symbol(words: list, written: str = "") -> bool:
    """A unit symbol or abbreviation stands in the hyp region: a symbol char, a digit token with letters / symbols glued
    on ("30%", "5km"), or the span's own written unit ("kg" of "53 kg") as a token after the first digit token."""
    unit = re.sub(r"[\d\s.,'’\-:/]+", "", written).lower()
    seen_digit = False
    for w in (x.strip(",;:!?") for x in words):
        if re.search(r"\d", w):
            seen_digit = True
            if re.sub(r"[\d\s.,'’\-:/]+", "", w):
                return True
        elif any(ch in SYMBOLS for ch in w):
            return True
        elif seen_digit and unit and w.lower().rstrip(".") == unit:
            return True
    return False


def unit_flags(cls: str, sp, lab_words: list, hyp_words: list, hyp_norm: list, lang: str) -> tuple:
    """(unit spoken in the label, unit missing from the hyp region) for pct / cur / unit spans, else (False, False)."""
    if cls not in ("pct", "cur", "unit"):
        return False, False
    lab_text = N._fold(" ".join(lab_words))
    if cls == "pct":
        rx = N._PCT.get(lang)
        spoken = bool(rx and re.search(rx, lab_text))
        present = bool(rx and re.search(rx, N._fold(" ".join(hyp_words)))) or "%" in " ".join(hyp_words)
        return spoken, spoken and not present
    lw = {SH._nw(w) for w in lab_words}
    uw = SH._unit_words(sp.cands, lang) & lw
    if cls == "cur":  # "5 euros": the labeller kind is int, the unit word is outside the span's readings
        uw |= {w for w in lw if w in N._CUR or w in N._MINOR}
    if not uw:
        return False, False
    present = any(h == u or (len(h) >= 4 and (h.startswith(u) or u.startswith(h))) for h in hyp_norm for u in uw)
    return True, not (present or _has_symbol(hyp_words, sp.text))


def dec_flags(cls: str, lab_words: list, hyp_words: list, lang: str) -> tuple:
    """(decimal word spoken in the label, missing from the hyp region) for dec spans."""
    rx = CC.DEC_SAID.get(lang)
    if cls != "dec" or not rx:
        return False, False
    spoken = bool(rx.search(" ".join(lab_words).lower()))
    h = " ".join(hyp_words).lower()
    present = bool(rx.search(h)) or bool(re.search(r"\d[.,]\d", h))
    return spoken, spoken and not present


def _label_window(lab: str, place: tuple) -> list:
    """The label words overlapping the span's characters plus one word on each side."""
    ws, pos = [], 0
    for w in lab.split(" "):
        ws.append((pos, pos + len(w), w))
        pos += len(w) + 1
    inside = [k for k, (a, b, _) in enumerate(ws) if a < place[1] and place[0] < b]
    if not inside:
        return []
    return [w for _, _, w in ws[max(0, inside[0] - 1):inside[-1] + 2] if w]


def _caps(lang: str, work: Path, mix: str) -> dict:
    """id -> reference row of <mix>/heldout_<lang>.jsonl. The local copy under work is reused when it parses (a build's
    held-out file is frozen; score_heldout._captions re-fetches on every call and a concurrent reader can see it half
    written); otherwise it is read from the volume."""
    p = work / mix / f"heldout_{lang}.jsonl"
    if p.exists() and p.stat().st_size:
        try:
            return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
        except ValueError:
            pass
    return SH._captions(lang, work, "heldout", mix)


def score_rows(rows: list, lang: str, caps: dict) -> dict:
    """{(id, span text, occurrence): record} for one decoded run; mirrors SH.score_spans (same placement, regions,
    outcomes) and adds the class, the form and the unit / decimal-word flags."""
    out, seen = {}, collections.Counter()
    for r in rows:
        m = caps.get(r["id"]) or {}
        cap = m.get("caption")
        if not cap:
            continue
        hyp_raw = r["hyp"].split()
        hw_all = [SH._nw(w) for w in hyp_raw]
        row_spans, nc = m.get("spans") or [], m.get("num_chars") or []
        lab = m.get("text", r.get("ref", ""))
        allc = tuple(tuple(x) for x in nc)
        k = 0
        nobare = M.spans_regional(cap, lang, SH.READINGS - {"cur_bare"})
        for sp, sp0 in zip(M.spans_regional(cap, lang, SH.READINGS), nobare):
            if not sp.cands:
                continue
            place, rsp = None, None
            for q in range(k, len(row_spans)):
                if row_spans[q].get("text") == sp.text and q < len(nc):
                    place, rsp, k = tuple(nc[q]), row_spans[q], q + 1
                    break
            i, j = SH._region(lab, place, hyp_raw, 1, allc) if place else (0, len(hyp_raw))
            i0, j0 = SH._region(lab, place, hyp_raw, 0, allc) if place else (0, len(hyp_raw))
            dig = re.sub(r"\D", "", sp.text)
            reg = " ".join(hyp_raw[i:j])
            if SH._heard(sp.cands, hw_all, i0, j0, i, j):
                o = "WORDS_RIGHT_VALUE"
                if sp.kind == "cur" and not SH._heard(sp0.cands, hw_all, i0, j0, i, j) and not (
                        SH._unit_words(sp0.cands, lang) & set(hw_all[i:j])):
                    o = "BARE_PRICE"
            elif dig and SH._digit_groups_match(dig, reg):
                o = "DIGITS_RIGHT_VALUE"
            elif re.search(r"\d", reg) or N.extract_numbers(reg, lang, keep_weak=False, garbles=True):
                o = "WRONG_VALUE"
            else:
                o = "DROPPED"
            form, _ = SH._span_forms(o, reg, lang)
            cls = klass(m, rsp or {"text": sp.text, "kind": sp.kind}, lang)
            lab_words = _label_window(lab, place) if place else []
            # the unit / decimal word may stand just outside the span's own words ("14" + "euros"), so both sides use the
            # +-1 word window (the label's neighbours; the hyp's slack region)
            hw0, n0 = hyp_raw[i:j], hw_all[i:j]
            us, ud = unit_flags(cls, sp, lab_words, hw0, n0, lang) if place else (False, False)
            ds, dd = dec_flags(cls, lab_words, hw0, lang) if place else (False, False)
            occ = seen[(r["id"], sp.text)]
            seen[(r["id"], sp.text)] += 1
            out[(r["id"], sp.text, occ)] = dict(lang=lang, cl=cluster(r["id"], lang), cls=cls, out=o, form=form,
                                                words=form in ("converted", "words_wrong"), digits=form == "digits_left",
                                                right=o in RIGHT, us=us, ud=ud, ds=ds, dd=dd, placed=place is not None)
    return out


def run_name(arg: str, mix: str) -> str:
    return arg if "__" in arg else f"{arg}__{mix}"


def load_arm(run: str, langs: list, work: Path, caps_by_lang: dict) -> dict:
    """{(lang, id, text, occ): record} of one arm over the langs. A language with no hyps is skipped (printed)."""
    res = {}
    for lang in langs:
        p = work / run / f"heldout_{lang}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else SH.fetch(run, lang, work)
        if not rows:
            print(f"# {run} {lang}: no hyps", file=sys.stderr)
            continue
        for k, v in score_rows(rows, lang, caps_by_lang[lang]).items():
            res[(lang,) + k] = v
    return res


# ------------------------------------------------------------------------------------------------ aggregation
def _sel(recs: list, metric: str) -> tuple:
    """(numerator flags, denominator flags) per record for one metric."""
    if metric == "wshare":
        return [r["words"] for r in recs], [True] * len(recs)
    if metric == "vr":
        return [r["right"] for r in recs], [True] * len(recs)
    if metric == "vrw":
        return [r["right"] and r["words"] for r in recs], [r["words"] for r in recs]
    if metric == "vrd":
        return [r["right"] and r["digits"] for r in recs], [r["digits"] for r in recs]
    if metric == "unitdrop":
        return [r["ud"] for r in recs], [r["us"] for r in recs]
    return [r["dd"] for r in recs], [r["ds"] for r in recs]  # decdrop


def aggregate(arms: dict, keys: list, refs: list, boot: int, seed: int = 0) -> dict:
    """For the span keys of one subset: per arm and metric (value %, n_den, CI95) and, per ref, the paired difference with a
    cluster bootstrap CI95 (clusters resampled within language). arms: {name: {key: rec}}."""
    import numpy as np
    first = arms[next(iter(arms))]
    cl = [first[k]["cl"] for k in keys]
    names = sorted(set(cl))
    ix = {c: i for i, c in enumerate(names)}
    ci = np.array([ix[c] for c in cl])
    cl_lang = np.array([c.split(":")[0] for c in names])
    groups = [np.where(cl_lang == l)[0] for l in sorted(set(cl_lang))]
    rng = np.random.default_rng(seed)
    pick_w = np.zeros((boot, len(names)))  # multiplicity of each cluster in each draw
    for b in range(boot):
        for g in groups:
            np.add.at(pick_w[b], g[rng.integers(0, len(g), len(g))], 1)
    sums = {}
    for a, recs in arms.items():
        rr = [recs[k] for k in keys]
        sums[a] = {}
        for m in METRICS:
            num, den = _sel(rr, m)
            sums[a][m] = (np.bincount(ci, weights=np.array(num, float), minlength=len(names)),
                          np.bincount(ci, weights=np.array(den, float), minlength=len(names)))
    out = {}
    for a in arms:
        out[a] = {}
        for m in METRICS:
            nn, dd = sums[a][m]
            tot = dd.sum()
            ent = {"v": float(100 * nn.sum() / tot) if tot else None, "den": int(tot)}
            with np.errstate(divide="ignore", invalid="ignore"):
                bv = 100 * (pick_w @ nn) / (pick_w @ dd)
            if np.isfinite(bv).sum() > boot * 0.5:
                ent["ci"] = [float(np.percentile(bv[np.isfinite(bv)], 2.5)), float(np.percentile(bv[np.isfinite(bv)], 97.5))]
            for r in refs:
                if r == a or r not in sums:
                    continue
                n2, d2 = sums[r][m]
                with np.errstate(divide="ignore", invalid="ignore"):
                    bd = bv - 100 * (pick_w @ n2) / (pick_w @ d2)
                ok, t2 = np.isfinite(bd), d2.sum()
                if tot and t2 and ok.sum() > boot * 0.5:
                    ent["d_" + r] = [float(100 * nn.sum() / tot - 100 * n2.sum() / t2),
                                     float(np.percentile(bd[ok], 2.5)), float(np.percentile(bd[ok], 97.5))]
            out[a][m] = ent
    return out


def fmt(e) -> str:
    return "   -  " if not e or e.get("v") is None else f"{e['v']:5.1f}"


def fmt_diff(e, ref) -> str:
    x = (e or {}).get("d_" + ref)
    if not x:
        return "        -        "
    return f"{x[0]:+5.1f}{'*' if x[1] > 0 or x[2] < 0 else ' '}[{x[1]:+5.1f},{x[2]:+5.1f}]"


def short(a: str) -> str:
    return a.split("__")[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="<run>@<step>  ('w3soup_a0.5@0' = B, 'ultra' = stock)")
    ap.add_argument("--mix", default="mix4final8", help="the build the probe decoded; its heldout_<lang>.jsonl are the references")
    ap.add_argument("--work", default=os.environ.get("PROBE_WORK", "/tmp/spoken_form_probe"))
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--refs", default=",".join(DEFAULT_REFS), help="reference arms every number is paired against")
    ap.add_argument("--json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=5, help="hide (lang, class) blocks with fewer spans than this")
    ap.add_argument("--classes", default="", help="comma list of classes to print (default all, plus other and ALL)")
    a = ap.parse_args(argv)
    langs, work = a.langs.split(","), Path(a.work)
    refs = [run_name(r, a.mix) for r in a.refs.split(",") if r]
    want = [run_name(x, a.mix) for x in a.arms]
    order = refs + [w for w in want if w not in refs]
    caps_by_lang = {l: _caps(l, work, a.mix) for l in langs}
    arms = {r: load_arm(r, langs, work, caps_by_lang) for r in order}
    arms = {r: v for r, v in arms.items() if v}
    refs = [r for r in refs if r in arms]
    keys_all = sorted(set.intersection(*[set(v) for v in arms.values()]))
    lost = {short(r): len(v) - len(keys_all) for r, v in arms.items()}
    print(f"# mix {a.mix}  arms {len(arms)}  common spans {len(keys_all)}  (spans not shared with every arm: {lost})")
    result = {}
    for lang in langs + ["ALL"]:
        for c in (a.classes.split(",") if a.classes else CLASSES + ["other", "ALL"]):
            keys = [k for k in keys_all if (lang == "ALL" or k[0] == lang)
                    and (c == "ALL" or arms[next(iter(arms))][k]["cls"] == c)]
            if keys:
                result[f"{lang}/{c}"] = {"n": len(keys), "arms": aggregate(arms, keys, refs, a.boot)}
    hdr = f"{'arm':26s} {'n':>5s} {'wshare':>6s} {'VR':>6s} {'VRw':>6s} {'VRd':>6s} {'unitdrop':>9s} {'decdrop':>8s}"
    print("\nValues in %. Under each non-reference arm: difference to each reference, * = cluster CI95 excludes 0, [lo,hi] = CI95.")
    for key, blk in result.items():
        if blk["n"] < a.min_n:
            continue
        print(f"\n== {key}  (n={blk['n']} spans)")
        print(hdr)
        for arm, mets in blk["arms"].items():
            u, dd = mets["unitdrop"], mets["decdrop"]
            print(f"{short(arm):26s} {blk['n']:5d} {fmt(mets['wshare']):>6s} {fmt(mets['vr']):>6s} {fmt(mets['vrw']):>6s} "
                  f"{fmt(mets['vrd']):>6s} {(fmt(u) + '/' + str(u['den'])) if u['den'] else '-':>9s} "
                  f"{(fmt(dd) + '/' + str(dd['den'])) if dd['den'] else '-':>8s}")
            if arm in refs:
                continue
            for r in refs:
                print(f"{'  d vs ' + short(r):26s} VR {fmt_diff(mets['vr'], r)} VRw {fmt_diff(mets['vrw'], r)} "
                      f"VRd {fmt_diff(mets['vrd'], r)} wshare {fmt_diff(mets['wshare'], r)} unitdrop "
                      f"{fmt_diff(mets['unitdrop'], r)} decdrop {fmt_diff(mets['decdrop'], r)}")
    if a.json:
        json.dump(result, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return result


if __name__ == "__main__":
    main()
