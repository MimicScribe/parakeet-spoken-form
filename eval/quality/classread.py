#!/usr/bin/env python3
"""Per-language x class held-out readout for any checkpoint (one command).

usage: classread.py <run>@<step>[__<mix>] [<run>@<step> ...] [--mix mix4final8] [--work DIR] [--langs de,es,fr,it,nl,pt]
                    [--refs ultra,spike_enc0_all@500] [--json out.json] [--boot 1000] [--min-n 5] [--legacy]

Reads the held-out hyps probe_heldout / eval_ckpt.sh wrote (hyps/<run>@<step>__<mix>/heldout_<lang>.jsonl; fetched with the
modal CLI into --work when not there yet) and scores every caption span with the SAME positional rules as
score_heldout.score_spans (words-right / digits-right / wrong / dropped / bare price; the +-1 word slack; the same READINGS).
SYMMETRIC readout (default, symscore.py: digits and words graded by one procedure; --legacy prints the old table below
instead). Per language x census class (census_common.gate_class: pct dec dec_pair cur unit int_big int_small year phone
clock alnum, everything else "other", and ALL) and per arm it reports

  n          spans (the same spans for every arm: the intersection)
  W          % of spans written in words (what the reader finds standing where the number stands)
  V          % of spans whose VALUE is right in either form ("40 mil", "XIII", "22" for 22:00, a glued phone number count;
             a bare year tail does not); never looks at units, decimal words or symbols
  U_w / U_d  % of the spans that SPEAK a unit or decimal word (percent, euro, km, komma ...) and are written in words / in
             digits where the region holds neither the word, a symbol, an abbreviation nor a digit separator, over those
             spans (shown as rate/denominator)

The OLD table (--legacy; asymmetric, kept for comparison; the default JSON holds both) reports

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
import symscore as SS  # noqa: E402

PIPE = str(Path(__file__).resolve().parents[2] / "pipeline")
sys.path.insert(0, PIPE)
import census_common as CC  # noqa: E402

M, D = SH.M, SH.D
LANGS = "de es fr it nl pt".split()
CLASSES = ["pct", "dec", "dec_pair", "cur", "unit", "int_big", "int_small", "year", "phone", "clock", "alnum"]
RIGHT = ("WORDS_RIGHT_VALUE", "DIGITS_RIGHT_VALUE")
DEFAULT_REFS = ("ultra", "spike_enc0_all@500")
SYMBOLS = set("%€$£°‰")
METRICS = ("wshare", "vr", "vrw", "vrd", "unitdrop", "decdrop")  # the old table
SYM_METRICS = ("W", "V", "Uw", "Ud")  # the symmetric table (symscore)


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
        # a word that is also a number word ("cent" = 100 in fr) inside the amount is no unit: the unit must not match a
        # number morpheme of the span's own reading
        uw |= {w for w in lw if (w in N._CUR or w in N._MINOR) and N.parse_value(w, lang) is None}
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


def _chains(sps: list, cap: str) -> dict:
    """span index -> digits of its caption chain, for spans that touch (gap only space / hyphen / slash / dot): a phone number
    or a range the caption splits into several spans ("22 14 19 44 49", "43-98"); the chain's digits are one value."""
    chain_of, cur = {}, []

    def close():
        if len(cur) > 1:
            for c in cur:
                chain_of[c] = "".join(re.sub(r"\D", "", sps[x].text) for x in cur)

    for si, sp in enumerate(sps):
        if cur and re.fullmatch(r"[ \-\u2013\u2212/.]{0,2}", cap[sps[cur[-1]].end:sp.start]) and sp.start >= sps[cur[-1]].end:
            cur.append(si)
        else:
            close()
            cur = [si]
    close()
    return chain_of


SYM_FIELDS = ("value_ok", "sform", "sym_via", "unit_spoken", "unit_present", "dec_spoken", "dec_present", "clean")


def _sym_missing(r: dict, form: str) -> tuple:
    """(speaks a unit or decimal word, and it is missing) for a span written in `form` (words / digits); (False, False)
    for any other form. The same test for both forms: the word, a symbol, an abbreviation or a digit separator."""
    if r.get("sform") != form:
        return False, False
    spoken = r.get("unit_spoken", False) or r.get("dec_spoken", False)
    missing = (r.get("unit_spoken", False) and not r.get("unit_present", False)) or (
        r.get("dec_spoken", False) and not r.get("dec_present", False))
    return bool(spoken), bool(missing)


def score_rows(rows: list, lang: str, caps: dict, sym: bool = True) -> dict:
    """{(id, span text, occurrence): record} for one decoded run; mirrors SH.score_spans (same placement, regions,
    outcomes) and adds the class, the form and the unit / decimal-word flags (the old table). sym: also the symmetric
    sub-scores of symscore (value_ok, sform, unit_* , dec_*, clean) for every span; a language symscore cannot read gets
    none (SYM_FIELDS absent) and the symmetric table skips it. A caption span with NO candidate reading (the reading
    generator has none for e.g. "3.000.000" or it "0,75") used to be skipped without a trace; with sym it is kept, flagged
    nocand=True and scored by symscore only (value from the caption digits, hyp read by the number reader); the old
    positional fields are meaningless for it (out="NOCAND") and every old metric leaves it out. Without sym it is skipped as
    before."""
    out, seen, recs = {}, collections.Counter(), []
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
        sps = M.spans_regional(cap, lang, SH.READINGS)
        chain_of = _chains(sps, cap)
        for si, (sp, sp0) in enumerate(zip(sps, nobare)):
            nocand = not sp.cands  # no candidate reading for the caption value ("3.000.000", it "0,75"): see below
            if nocand and not sym:
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
            if nocand:  # no word reading to compare with: the positional outcome means nothing; symscore still reads the value
                o = "NOCAND"
            occ = seen[(r["id"], sp.text)]
            seen[(r["id"], sp.text)] += 1
            rec = dict(lang=lang, cl=cluster(r["id"], lang), cls=cls, out=o, form=form,
                       words=form in ("converted", "words_wrong"), digits=form == "digits_left",
                       right=o in RIGHT, us=us, ud=ud, ds=ds, dd=dd, placed=place is not None, nocand=nocand)
            out[(r["id"], sp.text, occ)] = rec
            if sym:  # what symscore needs, dropped again after scoring
                recs.append(dict(rec, key=(lang, r["id"], sp.text, occ), id=r["id"], si=si, kind=sp.kind, text=sp.text,
                                 cap=cap, a=sp.start, b=sp.end, labseg=lab[place[0]:place[1]] if place else "", lab=lab,
                                 place=place, lab_words=lab_words, cands=list(sp.cands), cands0=list(sp0.cands),
                                 reg_words=hyp_raw[i:j], hyp=r["hyp"], ireg=(i, j), chain_tgt=chain_of.get(si),
                                 form_old=form, right_old=rec["right"], out_old=o))
    if sym and recs:
        try:
            by = collections.defaultdict(list)
            for x in recs:
                by[x["id"]].append(x)
            for row_recs in by.values():
                scored = SS.sym_score_row(row_recs)
                for x in row_recs:
                    sc = scored[x["key"]]
                    rec = out[(x["id"], x["text"], x["key"][3])]
                    rec.update(value_ok=sc["value_ok"], sform=sc["form"], sym_via=sc["via"],
                               unit_spoken=sc["unit_spoken"], unit_present=sc["unit_present"],
                               dec_spoken=sc["dec_spoken"], dec_present=sc["dec_present"], clean=sc["clean"])
        except SS.UnscorableLanguage:
            print(f"# {lang}: symscore has no number reader for this language; symmetric table skips it", file=sys.stderr)
    return out


def sym_counts(recs: dict) -> dict:
    """{class or "ALL": {n, W, V, Uw, Ud}} counts of one run's score_rows output (symmetric sub-scores): W / V spans, and for
    U_w / U_d [misses, spans that speak a unit or decimal word] written in words / digits. Rates are counts[k] / counts["n"]
    or miss / den. {} when no record carries symmetric scores (an unreadable language)."""
    out = {}
    for r in recs.values():
        if "value_ok" not in r:
            continue
        for key in (r["cls"], "ALL"):
            c = out.setdefault(key, {"n": 0, "W": 0, "V": 0, "Uw": [0, 0], "Ud": [0, 0], "nc": 0})
            c["n"] += 1
            c["nc"] += bool(r.get("nocand"))
            c["W"] += r["sform"] == "words"
            c["V"] += bool(r["value_ok"])
            for name, form in (("Uw", "words"), ("Ud", "digits")):
                spoken, missing = _sym_missing(r, form)
                c[name][0] += missing
                c[name][1] += spoken
    return out


def fmt_sym(c: dict) -> str:
    """One line for a sym_counts entry: n, W %, V %, U_w and U_d as missing% (misses/spans)."""
    pct = lambda a, b: f"{100 * a / b:5.1f}" if b else "   - "  # noqa: E731
    rate = lambda x: f"{pct(x[0], x[1])}% ({x[0]}/{x[1]})"  # noqa: E731
    nc = f" (nocand {c['nc']})" if c.get("nc") else ""
    return f"n {c['n']:5d}{nc}  W {pct(c['W'], c['n'])}  V {pct(c['V'], c['n'])}  U_w {rate(c['Uw'])}  U_d {rate(c['Ud'])}"


def run_name(arg: str, mix: str) -> str:
    return arg if "__" in arg else f"{arg}__{mix}"


def load_arm(run: str, langs: list, work: Path, caps_by_lang: dict, sym: bool = True) -> dict:
    """{(lang, id, text, occ): record} of one arm over the langs. A language with no hyps is skipped (printed)."""
    res = {}
    for lang in langs:
        p = work / run / f"heldout_{lang}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else SH.fetch(run, lang, work)
        if not rows:
            print(f"# {run} {lang}: no hyps", file=sys.stderr)
            continue
        for k, v in score_rows(rows, lang, caps_by_lang[lang], sym=sym).items():
            res[(lang,) + k] = v
    return res


# ------------------------------------------------------------------------------------------------ aggregation
def _sel(recs: list, metric: str) -> tuple:
    """(numerator flags, denominator flags) per record for one metric. The old (positional) metrics leave out spans with
    no candidate reading (nocand): their outcome is not defined there."""
    num, den = _sel_raw(recs, metric)
    if metric in METRICS:
        keep = [not r.get("nocand") for r in recs]
        num, den = [a and k for a, k in zip(num, keep)], [a and k for a, k in zip(den, keep)]
    return num, den


def _sel_raw(recs: list, metric: str) -> tuple:
    if metric == "W":
        return [r.get("sform") == "words" for r in recs], [True] * len(recs)
    if metric == "V":
        return [bool(r.get("value_ok")) for r in recs], [True] * len(recs)
    if metric in ("Uw", "Ud"):
        f = _sym_missing
        form = "words" if metric == "Uw" else "digits"
        return [f(r, form)[1] for r in recs], [f(r, form)[0] for r in recs]
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


def aggregate(arms: dict, keys: list, refs: list, boot: int, seed: int = 0, metrics: tuple = METRICS) -> dict:
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
        for m in metrics:
            num, den = _sel(rr, m)
            sums[a][m] = (np.bincount(ci, weights=np.array(num, float), minlength=len(names)),
                          np.bincount(ci, weights=np.array(den, float), minlength=len(names)))
    out = {}
    for a in arms:
        out[a] = {}
        for m in metrics:
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
    ap.add_argument("--legacy", action="store_true",
                    help="print the OLD asymmetric table (wshare / VR / VRw / VRd / unitdrop / decdrop) and compute only its "
                         "metrics, instead of the symmetric W / V / U_w / U_d table (symscore)")
    a = ap.parse_args(argv)
    langs, work = a.langs.split(","), Path(a.work)
    refs = [run_name(r, a.mix) for r in a.refs.split(",") if r]
    want = [run_name(x, a.mix) for x in a.arms]
    order = refs + [w for w in want if w not in refs]
    caps_by_lang = {l: _caps(l, work, a.mix) for l in langs}
    arms = {r: load_arm(r, langs, work, caps_by_lang, sym=not a.legacy) for r in order}
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
                has_sym = not a.legacy and all("value_ok" in v[k] for v in arms.values() for k in keys)
                result[f"{lang}/{c}"] = {"n": len(keys), "sym": has_sym,
                                         "nocand": sum(1 for k in keys if arms[next(iter(arms))][k].get("nocand")),
                                         "arms": aggregate(arms, keys, refs, a.boot,
                                                           metrics=METRICS + SYM_METRICS if has_sym else METRICS)}
    print("\nValues in %. Under each non-reference arm: difference to each reference, * = cluster CI95 excludes 0, [lo,hi] = CI95.")
    print_table(result, refs, a.min_n, legacy=a.legacy)
    if a.json:
        json.dump(result, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return result


def print_table(result: dict, refs: list, min_n: int, legacy: bool = False) -> None:
    """The per language x class table: symmetric (W V U_w U_d) for blocks that have it, else (or with legacy) the old one."""
    old_hdr = f"{'arm':26s} {'n':>5s} {'wshare':>6s} {'VR':>6s} {'VRw':>6s} {'VRd':>6s} {'unitdrop':>9s} {'decdrop':>8s}"
    sym_hdr = f"{'arm':26s} {'n':>5s} {'W':>6s} {'V':>6s} {'U_w':>9s} {'U_d':>9s}"
    rate = lambda e: (fmt(e) + "/" + str(e["den"])) if e["den"] else "-"  # noqa: E731
    for key, blk in result.items():
        if blk["n"] < min_n:
            continue
        sym = blk.get("sym") and not legacy
        nc = f", {blk['nocand']} without a candidate reading: V / W / U only, not in the old columns" if blk.get("nocand") else ""
        print(f"\n== {key}  (n={blk['n']} spans{nc})")
        print(sym_hdr if sym else old_hdr)
        for arm, mets in blk["arms"].items():
            if sym:
                print(f"{short(arm):26s} {blk['n']:5d} {fmt(mets['W']):>6s} {fmt(mets['V']):>6s} "
                      f"{rate(mets['Uw']):>9s} {rate(mets['Ud']):>9s}")
            else:
                u, dd = mets["unitdrop"], mets["decdrop"]
                print(f"{short(arm):26s} {blk['n']:5d} {fmt(mets['wshare']):>6s} {fmt(mets['vr']):>6s} {fmt(mets['vrw']):>6s} "
                      f"{fmt(mets['vrd']):>6s} {rate(u):>9s} {rate(dd):>8s}")
            if arm in refs:
                continue
            for r in refs:
                if sym:
                    print(f"{'  d vs ' + short(r):26s} V {fmt_diff(mets['V'], r)} W {fmt_diff(mets['W'], r)} "
                          f"U_w {fmt_diff(mets['Uw'], r)} U_d {fmt_diff(mets['Ud'], r)}")
                else:
                    print(f"{'  d vs ' + short(r):26s} VR {fmt_diff(mets['vr'], r)} VRw {fmt_diff(mets['vrw'], r)} "
                          f"VRd {fmt_diff(mets['vrd'], r)} wshare {fmt_diff(mets['wshare'], r)} unitdrop "
                          f"{fmt_diff(mets['unitdrop'], r)} decdrop {fmt_diff(mets['decdrop'], r)}")


if __name__ == "__main__":
    main()
