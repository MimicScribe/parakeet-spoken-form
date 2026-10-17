#!/usr/bin/env python3
"""Score the held-out probe (modal_app.py::probe_heldout) with the four-outcome number scorer.

usage: score_heldout.py <run>[,<run>...] [--langs de,es,fr,it,nl,pt] [--work DIR] [--json out.json]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
import ml_cands as M  # noqa: E402
import ml_decide as D  # noqa: E402

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"


def _captions(lang: str, work: Path, set_: str = "heldout") -> dict:
    """id -> evaluation row (mix4/heldout_<lang>.jsonl, mix4/tts_dev_<lang>.jsonl or mix4/dec18_<lang>.jsonl)."""
    src = f"mix4/{ {'ttsdev': 'tts_dev', 'dec18': 'dec18'}.get(set_, 'heldout') }_{lang}.jsonl"
    p = (work / f"{set_}_{lang}.jsonl").resolve()
    if True:  # review M3: always re-fetch (the held-out set changes with every rebuild)
        subprocess.run(["modal", "volume", "get", "--force", VOLUME, src, str(p)], cwd=PIPE, capture_output=True)
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


def _val_vox_ids(lang: str, work: Path) -> set:
    p = (work / f"val_vox_{lang}.jsonl").resolve()
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"ml/p3v2/{lang}/val_vox.jsonl", str(p)], cwd=PIPE,
                   capture_output=True)
    return {json.loads(l)["id"] for l in open(p, encoding="utf-8")} if p.exists() else set()


def fetch(run: str, lang: str, work: Path, set_: str = "heldout") -> list:
    p = (work / run / f"{set_}_{lang}.jsonl").resolve()  # modal runs in the pipeline dir: never a relative path
    p.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"hyps/{run}/{set_}_{lang}.jsonl", str(p)],
                       cwd=PIPE, capture_output=True, text=True)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if r.returncode == 0 else []


def _whole(reading: str, hyp_words: list) -> bool:
    k = D.key(reading)
    return any("".join(hyp_words[i:j]) == k for i in range(len(hyp_words)) for j in range(i + 1, min(len(hyp_words), i + 14) + 1))


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
    """How the hypothesis region of one caption span is written: converted / digits_left / mixed / other (number_outcomes
    FORM_CLASSES). A right-words span is converted; a digit-right span is digits_left unless a number word joins its
    digits (mixed); wrong / dropped spans take the form of what stands there."""
    if outcome == "WORDS_RIGHT_VALUE":
        return "converted"
    f = N.form_class(region, lang)
    return "digits_left" if f == "other" and outcome == "DIGITS_RIGHT_VALUE" else f


def score_spans(rows: list, lang: str, caps: dict, per_span: list = None, forms: dict = None) -> tuple[dict, list]:
    """Primary mode POSITIONAL: each caption span is scored on the
    hypothesis words that stand where its label words stand (`_region`, +-1 word), never on the whole hypothesis.
    WORDS_RIGHT_VALUE: the region holds ANY candidate reading of the span as whole words (a valid variant is not a
    wrong value); DIGITS_RIGHT_VALUE: a region token carries exactly the caption's digits; WRONG_VALUE: something
    number-like (digits, number words, a broken number word) stands there instead; DROPPED: nothing does. WRONG and
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
        for sp in M.spans(cap, lang):
            if not sp.cands:
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
            if (any(_whole(x, hw_all[i0:j0]) for x in sp.cands)
                    or any(len(D.key(x)) >= 5 and _whole(x, hw_all[i:j]) for x in sp.cands)):
                o = "WORDS_RIGHT_VALUE"
            elif dig and _digit_groups_match(dig, reg):
                o = "DIGITS_RIGHT_VALUE"
            elif re.search(r"\d", reg) or N.extract_numbers(reg, lang, keep_weak=False, garbles=True):
                o = "WRONG_VALUE"
            else:
                o = "DROPPED"
            c[sp.kind][o] += 1
            c["all"][o] += 1
            if forms is not None:  # form shares (converted / digits_left / mixed / other) over the same spans
                fc = span_form(o, reg, lang)
                forms[fc] = forms.get(fc, 0) + 1
                forms["n"] = forms.get("n", 0) + 1
            if per_span is not None:  # paired comparisons (bootstrap / McNemar between two runs on the same spans)
                per_span.append((r["id"], sp.text, sp.kind, o))
            if o in ("WRONG_VALUE", "DROPPED"):
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


def no_damage(run: dict, base: dict, pp: float = NO_DAMAGE_PP) -> dict:
    """Per-language no-damage rule (run 2, next / moderate tiers) on the positional span outcomes of the same held-out
    rows: value-right (words OR digits with the right value) >= baseline - pp, and wrong + dropped <= baseline + pp
    (percentage points of the spans). WER non-inferiority is the paired FLEURS bootstrap of score_retrain.py
    (fleurs_bootstrap_vs_<stock>), read next to this. {lang: {pass, value_right [run, base], wrong_dropped [...]}}."""
    out = {}
    for lang, r in run.items():
        b = base.get(lang)
        if not b or "spans" not in r or "spans" not in b:
            continue

        def pct(sc):
            al = sc["spans"].get("all", {})
            n = sum(al.values()) or 1
            return (100 * (al.get("WORDS_RIGHT_VALUE", 0) + al.get("DIGITS_RIGHT_VALUE", 0)) / n,
                    100 * (al.get("WRONG_VALUE", 0) + al.get("DROPPED", 0)) / n, n)

        vr, wd, n = pct(r)
        bvr, bwd, _ = pct(b)
        out[lang] = {"pass": vr >= bvr - pp and wd <= bwd + pp, "spans": n,
                     "value_right": [round(vr, 1), round(bvr, 1)], "wrong_dropped": [round(wd, 1), round(bwd, 1)]}
    return out


def pick_most_converted(out: dict, base: str) -> dict:
    """Pick rule : among runs that PASS the per-language no-damage gate (`no_damage`
    vs `base`) in EVERY language they were scored on, take the one whose numbers are most fully converted (highest
    mean converted share; ties: lower mixed + digits-left). Half-converted candidates (words for some numbers, digits
    left for others) are what a downstream digit-to-words step turns into two different readings of one audio, so converted
    share beats raw words-right at equal damage. Returns {"pick", "ranked": [(run, converted%, digits_left%, mixed%)],
    "per_language": {lang: {run: {converted, digits_left, mixed, other, pass}}}, "failed": {run: [langs]}}."""
    langs = sorted({l for r in out.values() for l in r if not l.startswith("_")})
    per, failed, ranked = {}, {}, []
    for run, res in out.items():
        if run == base:
            continue
        nd = res.get("_no_damage") or no_damage(res, out[base])
        bad = [l for l in langs if l in nd and not nd[l]["pass"]]
        failed[run] = bad
        sh = {l: N.form_shares(res[l].get("forms") or {}) for l in langs if l in res and res[l].get("forms")}
        for l, s in sh.items():
            per.setdefault(l, {})[run] = {**s, "pass": l not in bad}
        if sh and not bad:
            mean = lambda k: round(sum(s[k] or 0 for s in sh.values()) / len(sh), 1)  # noqa: E731
            ranked.append((run, mean("converted"), mean("digits_left"), mean("mixed")))
    ranked.sort(key=lambda t: (-t[1], t[2] + t[3]))
    return {"pick": ranked[0][0] if ranked else None, "ranked": ranked, "per_language": per, "failed": failed}


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
    ap.add_argument("--no-damage-vs", default="",
                    help="baseline run (B or ultra): per-language no-damage rule for every other run (next / moderate tiers)")
    a = ap.parse_args(argv)
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
            caps = _captions(lang, Path(a.work), a.set)
            fm = {}
            sc, check = score_spans(rows, lang, caps, forms=fm)
            c, wrong = score(rows, lang)
            di = drops_inserts(rows, lang, caps)
            out[run][lang] = {"spans": sc, "check": check, "four_outcomes": c, "wrong": wrong, "drops_inserts": di,
                              "forms": fm}
            al = sc.get("all", {})
            n = sum(al.values()) or 1
            print(f"{run:24s} {lang:4s} {n:5d} {100 * al.get('WORDS_RIGHT_VALUE', 0) / n:7.1f} "
                  f"{100 * al.get('DIGITS_RIGHT_VALUE', 0) / n:7.1f} {al.get('WRONG_VALUE', 0):5d} {al.get('DROPPED', 0):5d}   "
                  + " ".join(f"{k}:{v.get('WORDS_RIGHT_VALUE', 0)}/{sum(v.values())}" for k, v in sorted(sc.items()) if k not in ("all", "unplaced"))
                  + f"   | lost {di['lost_per100']} invented {di['invented_per100']} per 100 words (n={di['ref_content_words']})")
            sh = N.form_shares(fm)
            print(f"{'':24s} {'':4s} FORM converted {sh['converted']}% digits-left {sh['digits_left']}% mixed {sh['mixed']}% other {sh['other']}%")
    if a.no_damage_vs:
        for run in out:
            if run == a.no_damage_vs:
                continue
            nd = out[run]["_no_damage"] = no_damage(out[run], out[a.no_damage_vs])
            for lang, v in nd.items():
                print(f"NO-DAMAGE {run} {lang}: {'PASS' if v['pass'] else 'FAIL (stays on stock digits)'} "
                      f"value-right {v['value_right'][0]}% vs {v['value_right'][1]}%, wrong+dropped "
                      f"{v['wrong_dropped'][0]}% vs {v['wrong_dropped'][1]}% (n={v['spans']})")
    if a.no_damage_vs:
        pk = out["_pick_most_converted"] = pick_most_converted({k: v for k, v in out.items() if not k.startswith("_")}, a.no_damage_vs)
        print(f"PICK (most converted among no-damage passers): {pk['pick']}")
        for run, cv, dl, mx in pk["ranked"]:
            print(f"  {run:40s} converted {cv}% digits-left {dl}% mixed {mx}%")
        for run, bad in pk["failed"].items():
            if bad:
                print(f"  {run:40s} fails no-damage in {','.join(bad)}")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
