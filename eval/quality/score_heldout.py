#!/usr/bin/env python3
"""Score the held-out probe (modal_app.py::probe_heldout) with the four-outcome number scorer.

usage: score_heldout.py <run>[,<run>...] [--langs de,es,fr,it,nl,pt] [--work DIR] [--json out.json] [--mix-dir mix4] [--legacy]
--mix-dir: the build the probe decoded (probe_heldout --mix-dir; a candidate build such as mix4final): its
heldout_<lang> / tts_dev_<lang> rows are the references. The dec18 sets live in mix4 only (DEC18_DIR).
A caption / validation list that exists on the volume but cannot be fetched is an error (volume_io.read_shared); a
language the run or the baseline has no scores for FAILS the no-damage rule.
Fetches hyps/<run>/heldout_<lang>.jsonl from the Modal volume and prints per language: numbers, words-right (the
spoken-form target, "exact"), digits-right, wrong value, dropped. Primary mode scores each caption span against
ALL its candidate readings (the value parser split valid variants such as "negentien zestig"
or pairwise phone numbers into several numbers), POSITIONALLY: only the hypothesis words standing where the span's label
words stand count (a reading or digit run elsewhere in the sentence used to count as right). Wrong /
dropped spans are listed for a listen; nothing here kills an arm."""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N  # noqa: E402
import volume_io as V  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import ml_cands as M  # noqa: E402
import ml_decide as D  # noqa: E402
import census_common as CC  # noqa: E402

FULL_TIER = tuple(CC.TIER_LANGS["full"].split())  # en de es fr it nl pt: the tier every language of which must convert
LONG_CLASSES = ("phone", "int_big", "price", "date")
MIN_SPANS = 100  # a language with fewer scored spans is report-only in the no-damage rule (the CI is too wide to gate)

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"


DEC18_DIR = "mix4"  # the dec18 sets (FLEURS-dev digit rows of untrained languages) are built into mix4 only


def caption_src(lang: str, set_: str = "heldout", mix_dir: str = "mix4") -> str:
    """The volume path of a set's reference rows: <mix_dir>/heldout_<lang>.jsonl, <mix_dir>/tts_dev_<lang>.jsonl, or
    mix4/dec18_<lang>.jsonl (DEC18_DIR, whatever the build)."""
    d = DEC18_DIR if set_ == "dec18" else mix_dir
    return f"{d}/{ {'ttsdev': 'tts_dev', 'dec18': 'dec18'}.get(set_, 'heldout') }_{lang}.jsonl"


def _captions(lang: str, work: Path, set_: str = "heldout", mix_dir: str = "mix4") -> dict:
    """id -> evaluation row of caption_src. Always re-fetched (the held-out set changes with every rebuild); a missing
    or unreadable file is an error, never an empty reference set."""
    src = caption_src(lang, set_, mix_dir)
    p = (work / mix_dir / f"{set_}_{lang}.jsonl").resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not V.read_shared(src, p):
        raise V.VolumeReadError(f"{src} is not on the volume: no references to score {lang} against")
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def _val_vox_ids(lang: str, work: Path) -> set:
    """Ids of the VoxPopuli dev/test validation pass; empty when the language has none (not on the volume), an error
    when it exists but cannot be fetched."""
    p = (work / f"val_vox_{lang}.jsonl").resolve()
    if not V.read_shared(f"ml/p3v2/{lang}/val_vox.jsonl", p):
        return set()
    return {json.loads(l)["id"] for l in open(p, encoding="utf-8") if l.strip()}


def fetch(run: str, lang: str, work: Path, set_: str = "heldout") -> list:
    p = (work / run / f"{set_}_{lang}.jsonl").resolve()  # modal runs in the pipeline dir: never a relative path
    p.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"hyps/{run}/{set_}_{lang}.jsonl", str(p)],
                       cwd=PIPE, capture_output=True, text=True)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if r.returncode == 0 else []


# Italian elision (2026-10-18): the split label form "venti uno" / "cento ottanta" and a joined "ventiuno" are the
# same number words as "ventuno" / "centottanta"; the audio cannot tell them apart
_IT_ELIDE = re.compile(r"(vent)i(?=[uo])|(trent|quarant|cinquant|sessant|settant|ottant|novant)a(?=[uo])|(cent)o(?=[uo])")


def _elide(k: str) -> str:
    return _IT_ELIDE.sub(lambda m: m.group(1) or m.group(2) or m.group(3), k)


def _whole(reading: str, hyp_words: list) -> bool:
    k = _elide(D.key(reading))
    return any(_elide("".join(hyp_words[i:j])) == k for i in range(len(hyp_words))
               for j in range(i + 1, min(len(hyp_words), i + 14) + 1))


# the readings a span may be said with: the candidates plus the regional and the reading-variety fill readings
# (relative clock times, "zwei fünfzig", "tres con cinco", "zwo" / "meia", nl "à" / "percent", es "por cien"), the
# readings the training rows now use; each is a value-preserving reading, so it only adds right answers
READINGS = M.VARIETY_ALL | M.FILL_ALL


def _region(label: str, chars: tuple, hyp_raw: list, slack: int = 1, all_chars: tuple = ()) -> tuple:
    """Hypothesis words [i, j) standing where the label's number words (char span `chars`) stand: the words between
    the last label word before the span and the first label word after it that align with the hypothesis
    (difflib on normalised words; the words of EVERY number span of the row are left out of the anchors, so "seis
    seis" cannot anchor on itself), widened by `slack` words."""
    import difflib

    lw = label.split(" ")
    starts, pos = [], 0
    for w in lw:
        starts.append(pos)
        pos += len(w) + 1
    a, b = chars
    inside = [k for k, st in enumerate(starts) if st < b and a < st + len(lw[k])]
    if not inside:
        return 0, len(hyp_raw)
    lo, hi = inside[0], inside[-1]
    numw = {k for ca, cb in (all_chars or [chars]) for k, st in enumerate(starts) if st < cb and ca < st + len(lw[k])}
    ln = [_nw(w) if k not in numw else "\x00" for k, w in enumerate(lw)]
    hn = [_nw(w) for w in hyp_raw]
    prev, nxt = -1, len(hyp_raw)
    for blk in difflib.SequenceMatcher(None, ln, hn, autojunk=False).get_matching_blocks():
        for t in range(blk.size):
            li, hj = blk.a + t, blk.b + t
            if li < lo and hn[hj]:
                prev = max(prev, hj)
            elif li > hi and hn[hj] and nxt == len(hyp_raw):
                nxt = hj
    return max(0, prev + 1 - slack), min(len(hyp_raw), nxt + slack)


def _nw(w: str) -> str:
    return D.DIALECT.get(D.norm_word(w), D.norm_word(w))


def _heard(cands: list, hw_all: list, i0: int, j0: int, i: int, j: int) -> bool:
    return (any(_whole(x, hw_all[i0:j0]) for x in cands)
            or any(len(D.key(x)) >= 5 and _whole(x, hw_all[i:j]) for x in cands))


def _unit_words(cands: list, lang: str) -> set:
    """The currency-unit words of a price's readings ("euro", "cent"): reading words that are no number word and no
    joiner. A price heard through its unit-less reading (cur_bare) with none of these in the region lost the unit."""
    return {w for x in cands for w in map(_nw, x.split()) if len(w) >= 3 and w not in N._CURCONN
            and N.parse_value(w, lang) is None}


def _digit_groups_match(dig: str, text: str) -> bool:
    """True when `dig` equals one digit group of `text` or several consecutive ones joined ("06 12 34" = "061234")."""
    gs = re.findall(r"\d+", text)
    for a in range(len(gs)):
        acc = ""
        for g in gs[a:]:
            acc += g
            if acc == dig:
                return True
            if not dig.startswith(acc):
                break
    return False


def span_form(outcome: str, region: str, lang: str) -> str:
    """How the hypothesis region of one caption span is written: converted / digits_left / mixed / words_wrong / other
    (number_outcomes FORM_CLASSES). converted = words AND the value is right; a digit-right span is digits_left unless a
    number word joins its digits (mixed); a wrong span written in words (or with a broken number word) is words_wrong;
    other wrong / dropped spans take the form of what stands there."""
    return _span_forms(outcome, region, lang)[0]


def _span_forms(outcome: str, region: str, lang: str) -> tuple:
    """(form, form_old): form_old is the reading before words with a wrong value stopped counting as converted."""
    if outcome in ("WORDS_RIGHT_VALUE", "BARE_PRICE"):  # a bare price is words with the value right (its unit is counted apart)
        return "converted", "converted"
    f = N.form_class(region, lang)
    old = "digits_left" if f == "other" and outcome == "DIGITS_RIGHT_VALUE" else f
    if outcome == "WRONG_VALUE" and (old == "converted" or (old == "other" and not re.search(r"\d", region))):
        return "words_wrong", old
    return old, old


def long_class(kind: str, text: str) -> str:
    """The long class of a caption span, else "": phone, price (cur), date, int_big (a plain integer >= 1000 that is not
    a bare year 1800-2099)."""
    if kind in ("phone", "date"):
        return kind
    if kind == "cur":
        return "price"
    digits = re.sub(r"[\s.,'\u00a0\u202f]", "", text)
    if kind in ("int", "code") and digits.isdigit() and int(digits) >= 1000 and not (
            len(digits) == 4 and 1800 <= int(digits) <= 2099):
        return "int_big"
    return ""


def score_spans(rows: list, lang: str, caps: dict, per_span: list = None, forms: dict = None,
                forms_by_class: dict = None) -> tuple[dict, list]:
    """Primary mode POSITIONAL: each caption span is scored on the
    hypothesis words that stand where its label words stand (`_region`, +-1 word), never on the whole hypothesis.
    WORDS_RIGHT_VALUE: the region holds ANY candidate reading of the span as whole words (a valid variant is not a
    wrong value); DIGITS_RIGHT_VALUE: a region token carries exactly the caption's digits; WRONG_VALUE: something
    number-like (digits, number words, a broken number word) stands there instead; DROPPED: nothing does; BARE_PRICE
    (own outcome, not value-right): the price was heard only through its unit-less reading ("zwei fünfzig" for "2,50 €")
    and none of the currency-unit words stands in the region. WRONG and
    DROPPED spans are listed for a listen. A span that cannot be placed (no label position) is scored on the whole
    hypothesis and counted as `unplaced`."""
    c, check = collections.defaultdict(collections.Counter), []
    for r in rows:
        m = caps.get(r["id"]) or {}
        cap = m.get("caption")
        if not cap:
            continue
        hyp_raw = r["hyp"].split()
        hw_all = [_nw(w) for w in hyp_raw]
        row_spans, nc = m.get("spans") or [], m.get("num_chars") or []
        k = 0
        nobare = M.spans_regional(cap, lang, READINGS - {"cur_bare"})  # the same spans without the unit-less price reading
        for sp, sp0 in zip(M.spans_regional(cap, lang, READINGS), nobare):
            if not sp.cands:  # no word reading to compare with: not scored here, but counted (the symmetric block reads these)
                c["nocand"]["all"] += 1
                c["nocand"][sp.kind] += 1
                continue
            place = None
            for q in range(k, len(row_spans)):
                if row_spans[q].get("text") == sp.text and q < len(nc):
                    place, k = tuple(nc[q]), q + 1
                    break
            lab = m.get("text", r.get("ref", ""))
            allc = tuple(tuple(x) for x in nc)
            i, j = _region(lab, place, hyp_raw, 1, allc) if place else (0, len(hyp_raw))
            i0, j0 = _region(lab, place, hyp_raw, 0, allc) if place else (0, len(hyp_raw))
            if not place:
                c["unplaced"]["n"] += 1
            dig = re.sub(r"\D", "", sp.text)
            reg = " ".join(hyp_raw[i:j])
            # words: exactly in place; the +-1 word slack only for readings of 5+ letters (a short reading such as
            # "één" / "un" must not be matched by a neighbouring article). Digits: the span's digits as whole digit
            # groups of the region, one or several in a row (phone groups, "15h30", "2,80 €"); "5" inside "15" or
            # "2015" is a wrong value, not a right one.
            if _heard(sp.cands, hw_all, i0, j0, i, j):
                o = "WORDS_RIGHT_VALUE"
                if sp.kind == "cur" and not _heard(sp0.cands, hw_all, i0, j0, i, j) and not (
                        _unit_words(sp0.cands, lang) & set(hw_all[i:j])):
                    o = "BARE_PRICE"  # "zwei fünfzig" for "2,50 €": the amount heard, the currency unit dropped
            elif dig and _digit_groups_match(dig, reg):
                o = "DIGITS_RIGHT_VALUE"
            elif re.search(r"\d", reg) or N.extract_numbers(reg, lang, keep_weak=False, garbles=True):
                o = "WRONG_VALUE"
            else:
                o = "DROPPED"
            c[sp.kind][o] += 1
            c["all"][o] += 1
            if forms is not None:  # form shares (converted / digits_left / mixed / other) over the same spans
                fc, fold = _span_forms(o, reg, lang)
                forms[fc] = forms.get(fc, 0) + 1
                forms["n"] = forms.get("n", 0) + 1
                forms["converted_old"] = forms.get("converted_old", 0) + (fold == "converted")
                lc = long_class(sp.kind, sp.text)
                if forms_by_class is not None and lc:  # converted share on the long classes, per class
                    fb = forms_by_class.setdefault(lc, {})
                    fb[fc] = fb.get(fc, 0) + 1
                    fb["n"] = fb.get("n", 0) + 1
                    fb["converted_old"] = fb.get("converted_old", 0) + (fold == "converted")
            if per_span is not None:  # paired comparisons (bootstrap / McNemar between two runs on the same spans)
                per_span.append((r["id"], sp.text, sp.kind, o))
            if o in ("WRONG_VALUE", "DROPPED", "BARE_PRICE"):
                check.append({"id": r["id"], "kind": sp.kind, "outcome": o, "caption_span": sp.text,
                              "hyp_region": reg, "ref": r["ref"], "hyp": r["hyp"]})
    return {k: dict(v) for k, v in c.items()}, check


def _norm_words(text: str) -> list:
    return [D.DIALECT.get(D.norm_word(w), D.norm_word(w)) for w in text.split()]


def drops_inserts(rows: list, lang: str, meta: dict) -> dict:
    """Non-English lost/invented words (previously unmeasured): the judged label is the reference. Number words
    (label num_chars and any other number word, on both sides), the hypothesis words standing at a caption number's
    place, and tokens with digits are left out, so digits vs words is not counted.
    Returns content words (>= 4 letters) deleted / inserted per 100 reference content words, with examples."""
    import difflib

    ref_n = dele = ins = 0
    ex = []
    for r in rows:
        m = meta.get(r["id"])
        if not m:
            continue
        lab = m["text"]
        nums = [False] * len(lab.split())
        pos = 0
        for i, tok in enumerate(lab.split(" ")):
            if any(a <= pos < b for a, b in m.get("num_chars") or []):
                nums[i] = True
            pos += len(tok) + 1
        # number words are left out on BOTH sides (2026-10-16 fix: only the hypothesis side was filtered, so every
        # number word of the label outside a caption span - "zwei, drei Beispiele", "eine" - counted as lost)
        ref = [w for w, raw, f in zip(_norm_words(lab), lab.split(" "), nums)
               if not f and w and not re.search(r"\d", raw) and N.parse_value(raw, lang) is None]
        # the hypothesis words standing where a caption number stands ("drei Komma fünf Prozent" vs "3,5 %", a garbled
        # "zweitausendfalf") belong to the number scorer, not to lost / invented words
        hr = r["hyp"].split()
        allc = tuple(tuple(x) for x in m.get("num_chars") or [])
        spanw = {w for a, b in allc for w in _norm_words(lab[a:b])}  # "komma", "prozent", "euro", "uhr" of the readings
        inspan = {k for c in allc for k in range(*_region(lab, c, hr, 0, allc))}
        hyp = [w for k, (w, raw) in enumerate(zip(_norm_words(r["hyp"]), hr))
               if w and not (k in inspan and (w in spanw or N.extract_numbers(raw, lang, garbles=True)))
               and not re.search(r"\d", raw) and N.parse_value(raw, lang) is None]
        ref_n += sum(len(w) >= 4 for w in ref)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if op in ("delete", "replace") and i2 - i1 > j2 - j1:
                d = [w for w in ref[i1 + (j2 - j1):i2] if len(w) >= 4]
                dele += len(d)
                if d and len(ex) < 8:
                    ex.append(("lost", " ".join(d), r["hyp"][:100]))
            if op in ("insert", "replace") and j2 - j1 > i2 - i1:
                n = [w for w in hyp[j1 + (i2 - i1):j2] if len(w) >= 4]
                ins += len(n)
                if n and len(ex) < 8:
                    ex.append(("invented", " ".join(n), r["hyp"][:100]))
    per = lambda x: round(100 * x / ref_n, 2) if ref_n else None  # noqa: E731
    return {"ref_content_words": ref_n, "lost_per100": per(dele), "invented_per100": per(ins), "examples": ex}


def score(rows: list, lang: str) -> tuple[dict, list]:
    """Secondary mode: the four-outcome scorer on the whole label (ref) vs hypothesis."""
    c, wrong = collections.Counter(), []
    for r in rows:
        for o in N.number_outcomes(r["ref"], r["hyp"], lang):
            c[o["outcome"]] += 1
            c["n"] += 1
            if o["outcome"] == "WRONG_VALUE":
                wrong.append({"id": r["id"], "value": o["value"], "ref": r["ref"], "hyp": r["hyp"]})
    return dict(c), wrong


NO_DAMAGE_PP = 2.0  # next / moderate tiers ship only when no worse than the baseline by > 2 pp


def no_damage(run: dict, base: dict, pp: float = NO_DAMAGE_PP, langs=None) -> dict:
    """Per-language no-damage rule (run 2, next / moderate tiers) on the positional span outcomes of the same held-out
    rows: value-right (words OR digits with the right value) >= baseline - pp, and wrong + dropped <= baseline + pp
    (percentage points of the spans). WER non-inferiority is the paired FLEURS bootstrap of score_retrain.py
    (fleurs_bootstrap_vs_<stock>), read next to this. {lang: {pass, value_right [run, base], wrong_dropped [...]}}.
    langs: the languages that must pass (default: every language either side scored); a language without span scores on
    either side FAILS ({"pass": False, "missing": "run" / "base" / "both"}), it is never skipped."""
    out = {}
    if langs is None:
        langs = sorted({l for l in list(run) + list(base) if not str(l).startswith("_")})
    for lang in langs:
        r, b = run.get(lang) or {}, base.get(lang) or {}
        if "spans" not in r or "spans" not in b:
            miss = "both" if "spans" not in r and "spans" not in b else "run" if "spans" not in r else "base"
            out[lang] = {"pass": False, "missing": miss}
            continue

        def pct(sc):
            al = sc["spans"].get("all", {})
            n = sum(al.values()) or 1
            return (100 * (al.get("WORDS_RIGHT_VALUE", 0) + al.get("DIGITS_RIGHT_VALUE", 0)) / n,
                    100 * (al.get("WRONG_VALUE", 0) + al.get("DROPPED", 0)) / n, n)

        vr, wd, n = pct(r)
        bvr, bwd, _ = pct(b)
        out[lang] = {"pass": vr >= bvr - pp and wd <= bwd + pp, "spans": n, "report_only": n < MIN_SPANS,
                     "value_right": [round(vr, 1), round(bvr, 1)], "wrong_dropped": [round(wd, 1), round(bwd, 1)]}
    return out


BOOT_N, BOOT_ALPHA = 2000, 0.05


def _span_keys(ps: list) -> dict:
    """{(row id, span text, occurrence): (kind, outcome)} from a per_span list; the occurrence index keeps repeated
    texts of one row apart."""
    seen, out = collections.Counter(), {}
    for rid, text, kind, o in ps:
        k = (rid, text, seen[(rid, text)])
        seen[(rid, text)] += 1
        out[k] = (kind, o)
    return out


def paired_bootstrap(run_ps: list, base_ps: list, pp: float = NO_DAMAGE_PP, seed: int = 0) -> dict:
    """Paired bootstrap (rows resampled whole) of run - baseline over the spans both scored, in percentage points, for
    words-right, digits-right and value-right (their sum). `harm` = the 95% CI of value-right lies entirely below
    -pp: the only case that fails (a point estimate under -pp inside a wide CI does not). Words-right and digits-right
    are reported; converting trades digits for words, so only their sum gates."""
    import random
    r, b = _span_keys(run_ps), _span_keys(base_ps)
    keys = sorted(set(r) & set(b))
    if not keys:
        return {"n_paired": 0, "harm": False}
    by = collections.defaultdict(lambda: [0, 0, 0, 0])  # row id -> spans, dwords, ddigits, dvalue
    for k in keys:
        x = by[k[0]]
        x[0] += 1
        rw, bw = r[k][1] == "WORDS_RIGHT_VALUE", b[k][1] == "WORDS_RIGHT_VALUE"
        rd, bd = r[k][1] == "DIGITS_RIGHT_VALUE", b[k][1] == "DIGITS_RIGHT_VALUE"
        x[1] += rw - bw
        x[2] += rd - bd
        x[3] += (rw or rd) - (bw or bd)
    rows = list(by.values())
    n = len(keys)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT_N):
        t = [0, 0, 0, 0]
        for _ in rows:
            for i, v in enumerate(rng.choice(rows)):
                t[i] += v
        draws.append(t)
    out = {"n_paired": n, "n_rows": len(rows)}
    for i, name in ((1, "words_right"), (2, "digits_right"), (3, "value_right")):
        ds = sorted(100 * d[i] / d[0] for d in draws if d[0])
        lo, hi = ds[int(BOOT_ALPHA / 2 * len(ds))], ds[int((1 - BOOT_ALPHA / 2) * len(ds)) - 1]
        out[name] = {"diff_pp": round(100 * sum(x[i] for x in rows) / n, 2), "ci95": [round(lo, 2), round(hi, 2)]}
    out["harm"] = out["value_right"]["ci95"][1] < -pp
    out["report_only"] = n < MIN_SPANS
    return out


def min_long_converted(res: dict) -> tuple | None:
    """(min converted% over the full-tier languages scored, the worst language, how many were counted) on the long
    classes (LONG_CLASSES) pooled per language; None when no full-tier language has long-class spans."""
    vals = {}
    for l in FULL_TIER:
        fb = (res.get(l) or {}).get("forms_by_class") or {}
        n = sum(fb.get(c, {}).get("n", 0) for c in LONG_CLASSES)
        if n:
            vals[l] = round(100 * sum(fb.get(c, {}).get("converted", 0) for c in LONG_CLASSES) / n, 1)
    if not vals:
        return None
    worst = min(vals, key=lambda l: (vals[l], l))
    return vals[worst], worst, len(vals)


def pick_most_converted(out: dict, base: str, honour_report_only: bool = False, rule: str = "point") -> dict:
    """Pick rule : among runs that PASS the per-language no-damage gate (`no_damage`
    vs `base`) in EVERY language they were scored on, take the one whose numbers are most fully converted (highest
    mean converted share; ties: lower mixed + digits-left). Half-converted candidates (words for some numbers, digits
    left for others) are what a downstream digit-to-words step turns into two different readings of one audio, so converted
    share beats raw words-right at equal damage. Returns {"pick", "ranked": [(run, converted%, digits_left%, mixed%)],
    "per_language": {lang: {run: {converted, digits_left, mixed, other, pass}}}, "failed": {run: [langs]},
    "ranked_min_long": [(run, min converted% over the full-tier languages on the long classes pooled, worst language,
    languages counted)], "pick_min": best of that ranking}. The min ranking is an addition printed beside the mean one
    (one weak language cannot hide in a mean). honour_report_only: a language under MIN_SPANS spans cannot fail the
    no-damage rule. rule "ci": the paired-bootstrap harm verdict (out[run]["_no_damage_ci"]) replaces the point rule."""
    langs = sorted({l for r in out.values() for l in r if not l.startswith("_")})
    per, failed, ranked, ranked_min = {}, {}, [], []
    for run, res in out.items():
        if run == base:
            continue
        nd = res.get("_no_damage") or no_damage(res, out[base], langs=langs)
        ci = res.get("_no_damage_ci") or {}

        def ok(l):
            v = nd.get(l) or {}
            if "missing" in v:
                return False
            if honour_report_only and v.get("report_only"):
                return True
            return not (ci.get(l) or {}).get("harm") if rule == "ci" and l in ci else bool(v.get("pass"))

        bad = [l for l in langs if not ok(l)]  # not scored = not passed
        failed[run] = bad
        sh = {l: N.form_shares(res[l].get("forms") or {}) for l in langs if l in res and res[l].get("forms")}
        for l, s in sh.items():
            per.setdefault(l, {})[run] = {**s, "pass": l not in bad}
        if sh and not bad:
            mean = lambda k: round(sum(s[k] or 0 for s in sh.values()) / len(sh), 1)  # noqa: E731
            ranked.append((run, mean("converted"), mean("digits_left"), mean("mixed")))
            lm = min_long_converted(res)
            if lm:
                ranked_min.append((run, *lm))
    ranked.sort(key=lambda t: (-t[1], t[2] + t[3]))
    ranked_min.sort(key=lambda t: (-t[1], t[0]))
    return {"pick": ranked[0][0] if ranked else None, "ranked": ranked, "per_language": per, "failed": failed,
            "ranked_min_long": ranked_min, "pick_min": ranked_min[0][0] if ranked_min else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--langs", default="de,es,fr,it,nl,pt")
    ap.add_argument("--work", default=os.environ.get("PROBE_WORK", "/tmp/spoken_form_probe"))
    ap.add_argument("--json")
    ap.add_argument("--subset", default="all", choices=["all", "dev", "final", "decision"],
                    help="dev = FLEURS-test validation rows only (recipe decisions, Amendment 14); final = everything "
                         "else (VoxPopuli dev/test + carved + original held-out), read once on the selected model")
    ap.add_argument("--set", default="heldout", choices=["heldout", "ttsdev", "dec18"],
                    help="heldout = real held-out + VoxPopuli dev/test rows; ttsdev = held-out synthetic voices")
    ap.add_argument("--mix-dir", default="mix4", help="the build the probe decoded (probe_heldout --mix-dir)")
    ap.add_argument("--legacy", action="store_true",
                    help="only the old positional outputs (words-right / digits-right / wrong / dropped); default adds the "
                         "symmetric V / U_w / U_d / W block per language x class (symscore via classread) and a 'sym' key in --json")
    ap.add_argument("--no-damage-vs", default="",
                    help="baseline run (B or ultra): per-language no-damage rule for every other run (next / moderate tiers)")
    ap.add_argument("--no-damage-rule", default="point", choices=["point", "ci"],
                    help="point = the 2 pp point rule (default); ci = fail only when the paired-bootstrap CI95 of "
                         "value-right lies entirely beyond -2 pp")
    ap.add_argument("--small-n-report-only", action="store_true",
                    help=f"a language with under {MIN_SPANS} scored spans cannot fail the no-damage rule (still printed)")
    a = ap.parse_args(argv)
    if a.no_damage_vs and a.set == "dec18":
        ap.error("--no-damage-vs needs span scores: not available for --set dec18")
    out = {}
    if a.no_damage_vs and a.no_damage_vs not in a.runs.split(","):
        a.runs = a.no_damage_vs + "," + a.runs
    print(f"{'run':24s} {'lang':4s} {'spans':>5s} {'words%':>7s} {'digit%':>7s} {'wrong':>5s} {'drop':>5s}   words-right per kind")
    for run in a.runs.split(","):
        out[run] = {}
        for lang in a.langs.split(","):
            rows = fetch(run, lang, Path(a.work), a.set)
            if a.subset == "decision":  # run 2: FLEURS-dev rows + carved/original held-out (never in any bed); NOT
                # FLEURS-test (fleursval_) and NOT the VoxPopuli dev/test validation pass (val_vox overlaps vox_dev)
                vv = _val_vox_ids(lang, Path(a.work))
                rows = [r for r in rows if not str(r["id"]).startswith("fleursval_") and r["id"] not in vv]
            elif a.subset != "all":
                rows = [r for r in rows if str(r["id"]).startswith("fleursval_") == (a.subset == "dev")]
            if a.set == "dec18":  # untrained languages: digits kept? (digits-only four-outcome scoring vs the raw caption)
                c, wrong = score(rows, lang)
                n = c.get("n", 0) or 1
                out[run][lang] = {"four_outcomes": c, "wrong": wrong}
                print(f"{run:24s} {lang:4s} {n:5d} digits-right {100 * c.get('DIGITS_RIGHT_VALUE', 0) / n:5.1f}% "
                      f"wrong {c.get('WRONG_VALUE', 0):4d} dropped {c.get('DROPPED', 0):4d}")
                continue
            if not rows:
                print(f"{run:24s} {lang:4s}  (no hyps)")
                continue
            caps = _captions(lang, Path(a.work), a.set, a.mix_dir)
            fm, fbc, ps = {}, {}, []
            sc, check = score_spans(rows, lang, caps, per_span=ps, forms=fm, forms_by_class=fbc)
            c, wrong = score(rows, lang)
            di = drops_inserts(rows, lang, caps)
            out[run][lang] = {"spans": sc, "check": check, "four_outcomes": c, "wrong": wrong, "drops_inserts": di,
                              "forms": fm, "forms_by_class": fbc, "per_span": ps}
            sym = {}
            if not a.legacy and a.set == "heldout":
                import classread as CR  # the symmetric scorer's records (classread imports this module: lazy)

                sym = out[run][lang]["sym"] = CR.sym_counts(CR.score_rows(rows, lang, caps, sym=True))
            al = sc.get("all", {})
            n = sum(al.values()) or 1
            print(f"{run:24s} {lang:4s} {n:5d} {100 * al.get('WORDS_RIGHT_VALUE', 0) / n:7.1f} "
                  f"{100 * al.get('DIGITS_RIGHT_VALUE', 0) / n:7.1f} {al.get('WRONG_VALUE', 0):5d} {al.get('DROPPED', 0):5d} bare-price {al.get('BARE_PRICE', 0):3d}  "
                  + " ".join(f"{k}:{v.get('WORDS_RIGHT_VALUE', 0)}/{sum(v.values())}" for k, v in sorted(sc.items()) if k not in ("all", "unplaced", "nocand"))
                  + (f" unscored(no reading) {sc['nocand']['all']}" if sc.get("nocand") else "")
                  + f"   | lost {di['lost_per100']} invented {di['invented_per100']} per 100 words (n={di['ref_content_words']})")
            sh = N.form_shares(fm)
            print(f"{'':24s} {'':4s} FORM converted {sh['converted']}% digits-left {sh['digits_left']}% mixed {sh['mixed']}% words-wrong {sh['words_wrong']}% other {sh['other']}% (old converted {sh.get('converted_old')}%)")
            if sym:
                print(f"{'':24s} {'':4s} SYM  {CR.fmt_sym(sym['ALL'])}   (V = value right in either form, W = written in words, "
                      f"U_w / U_d = unit or decimal word missing among spans written in words / digits)")
                for cl in sorted(k for k in sym if k != "ALL"):
                    print(f"{'':24s} {'':4s}   {cl:9s} {CR.fmt_sym(sym[cl])}")
    if a.no_damage_vs:
        for run in out:
            if run == a.no_damage_vs:
                continue
            nd = out[run]["_no_damage"] = no_damage(out[run], out[a.no_damage_vs], langs=a.langs.split(","))
            ci = out[run]["_no_damage_ci"] = {
                l: paired_bootstrap(out[run][l]["per_span"], out[a.no_damage_vs][l]["per_span"])
                for l in nd if "missing" not in nd[l]}
            for lang, v in nd.items():
                if "missing" in v:
                    print(f"NO-DAMAGE {run} {lang}: FAIL (no scores: {v['missing']})")
                    continue
                ro = f" REPORT-ONLY (n < {MIN_SPANS})" if v["report_only"] else ""
                print(f"NO-DAMAGE {run} {lang}: {'PASS' if v['pass'] else 'FAIL (stays on stock digits)'}{ro} "
                      f"value-right {v['value_right'][0]}% vs {v['value_right'][1]}%, wrong+dropped "
                      f"{v['wrong_dropped'][0]}% vs {v['wrong_dropped'][1]}% (n={v['spans']})")
                c = ci[lang]
                if c.get("n_paired"):
                    print(f"{'':12s} paired CI95 vs {a.no_damage_vs} (n={c['n_paired']} spans, {c['n_rows']} rows): "
                          + ", ".join(f"{k.replace('_', '-')} {c[k]['diff_pp']:+.1f} pp {c[k]['ci95']}"
                                      for k in ("words_right", "digits_right", "value_right"))
                          + f" -> {'HARM beyond ' + str(NO_DAMAGE_PP) + ' pp' if c['harm'] else 'no proven harm'}"
                          + (" (report-only)" if c["report_only"] else ""))
    if a.no_damage_vs:
        pk = out["_pick_most_converted"] = pick_most_converted({k: v for k, v in out.items() if not k.startswith("_")}, a.no_damage_vs,
                                                      honour_report_only=a.small_n_report_only, rule=a.no_damage_rule)
        print(f"PICK (most converted among no-damage passers): {pk['pick']}")
        for run, cv, dl, mx in pk["ranked"]:
            print(f"  {run:40s} converted {cv}% digits-left {dl}% mixed {mx}%")
        print(f"PICK by MIN converted over full-tier languages ({','.join(FULL_TIER)}), long classes "
              f"({','.join(LONG_CLASSES)}) pooled: {pk['pick_min']}")
        for run, mn, worst, k in pk["ranked_min_long"]:
            print(f"  {run:40s} min converted {mn}% ({worst}; {k} full-tier languages counted)")
        for run, bad in pk["failed"].items():
            if bad:
                print(f"  {run:40s} fails no-damage in {','.join(bad)}")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
