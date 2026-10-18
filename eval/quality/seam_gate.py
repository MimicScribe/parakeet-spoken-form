#!/usr/bin/env python3
"""Seam gate: does a checkpoint mangle LONG numbers where a streaming window edge cuts them?

usage: seam_gate.py <run>[,<run>...] --base <B run> [--langs de,es,...] [--work DIR] [--json gate.json] [--no-fetch]
       [--offset-work DIR --offset-arms B=<arm>,<run>=<arm> [--baseline-work DIR]]   (--json required with --offset-work)

Inputs (fetched from the Modal volume unless --no-fetch; the work dir then holds the same paths):
  pairs   ml/p3v2/heldout_seam_<lang>.jsonl (en: d1/en4_ok/heldout_seam_en.jsonl): mirrored seam pairs built from
          HELD-OUT rows only (ml_p3 seam_crops --heldout / en_win.py prep --heldout); the left half ends at the cut, the
          right half starts there, and their labels partition the reading;
  hyps    hyps/<run>/hseam_<lang>.jsonl and hyps/<run>/hseam_parents_<lang>.jsonl (the pairs' parent rows decoded
          whole), both written by modal_app.py::probe_seam.
Per pair: the hypothesis words of each half that stand where its reading part stands (after the left half's last
anchored non-reading word / before the right half's first one) are joined -- the merge a streaming decoder makes at the
seam -- and their number values compared IN ORDER with the reading's values (number_outcomes.extract_numbers, digits and
words alike): RIGHT, DROPPED (nothing number-like) or WRONG. MERGE-MADE = wrong / dropped at the seam while the parent
row decoded whole holds the reading's values (contiguous). Long numbers only: phone, price (cur), date, clock and big
integers (>= 1000, not a 1800-2099 year); years are reported apart.
The Python proxy gate below is REPORT ONLY (it over-counts); the veto is the streaming-path offset score (offset_gate).
Proxy gate (checkpoint selection, run 2): wave-1 pooled merge-made wrong <= 0.5 x the baseline's (0 when the baseline
has 0), English merge-made wrong = 0, single-decode wrong + dropped on the same readings <= the baseline's (wave 1
pooled and English). Next / moderate tiers: per-language seam check (merge-made wrong <= the baseline's) for the
deploy gate, not a selection veto."""
import argparse
import collections
import difflib
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N  # noqa: E402

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"
WAVE1 = ("de", "es", "fr", "it", "nl", "pt")
NEXT_MODERATE = ("ru", "uk", "pl", "ro", "sv", "cs", "hu", "fi", "sk", "bg", "hr")
LONG_KINDS = {"phone", "cur", "date", "clock"}
MERGE_WRONG_FACTOR = 0.5


def pairs_path(lang: str) -> str:
    return "d1/en4_ok/heldout_seam_en.jsonl" if lang == "en" else f"ml/p3v2/heldout_seam_{lang}.jsonl"


def _get(src: str, dst: Path, fetch: bool) -> list:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fetch:
        subprocess.run(["modal", "volume", "get", "--force", VOLUME, src, str(dst.resolve())], cwd=PIPE,
                       capture_output=True)
    return [json.loads(l) for l in open(dst, encoding="utf-8") if l.strip()] if dst.exists() else []


def _w(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if c.isalnum())


def values(text: str, lang: str, hyp: bool = False) -> list:
    out = []
    for n in N.extract_numbers(text, lang, keep_weak=hyp, garbles=hyp):
        if n.vals:
            out.append(min(n.vals))
    return out


def _region(label: str, hyp: str, n_reading: int, side: str) -> str:
    """Hypothesis words standing where the half's reading part stands: a left half's reading part is its last
    n_reading label words, a right half's its first. Anchors = the other label words matched in order (difflib)."""
    lw = label.split()
    hw = hyp.split()
    keep = range(0, len(lw) - n_reading) if side == "left" else range(n_reading, len(lw))
    la = [_w(lw[k]) if k in keep else "\x00" for k in range(len(lw))]
    hn = [_w(x) for x in hw]
    matched = []
    for b in difflib.SequenceMatcher(None, la, hn, autojunk=False).get_matching_blocks():
        matched += [(b.a + t, b.b + t) for t in range(b.size) if la[b.a + t] != "\x00" and la[b.a + t]]
    if side == "left":
        j = max((h for _, h in matched), default=-1)
        return " ".join(hw[j + 1:])
    j = min((h for _, h in matched), default=len(hw))
    return " ".join(hw[:j])


def _digits(vals: list) -> str:
    """The values as one digit string ("2.255" and 2255 alike; 809 290 3236 -> "8092903236")."""
    return "".join("".join(c for c in str(v) if c.isdigit()) for v in vals)


def same_value(got: list, want: list, grouped: bool = False) -> bool:
    """In-order value match. "2.255" for 2255 (same count of numbers, same digits) is right; a different split counts
    by its digits only for digit-group shapes (grouped: phone numbers, "809-290-3236"): "3 25" for 325 is wrong."""
    if got == want:
        return True
    return bool(want) and _digits(got) == _digits(want) and (grouped or len(got) == len(want))


def _contiguous(sub: list, seq: list) -> bool:
    """The reading's values inside the whole-row decode: a run of values, or its digits inside a run's digits (a phone
    piece "treinta y dos treinta y seis" inside "809-290-3236")."""
    if not sub:
        return False
    if any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1)):
        return True
    d = _digits(sub)
    return any(d in _digits(seq[i:j]) for i in range(len(seq)) for j in range(i + 1, min(len(seq), i + 6) + 1))


KIND_ALIAS = {"money": "cur"}  # the English labeller's name for a price


def long_class(kind: str, vals: list) -> str | None:
    kind = KIND_ALIAS.get(kind, kind)
    if kind in LONG_KINDS:
        return kind
    if not vals:
        return None
    v = max(vals)
    if len(vals) == 1 and v == int(v) and 1800 <= v <= 2099:
        return "year"
    return "big" if v >= 1000 else None


def score_pairs(pairs: list, seam_hyp: dict, single_hyp: dict, lang: str) -> dict:
    """Counts per long class (+ "all_long"): pairs, right / wrong / dropped at the seam, merge-made wrong / dropped
    (single decode right), single-decode wrong / dropped (parent decoded whole), and up to 8 merge-made examples."""
    by = collections.defaultdict(list)
    for h in pairs:
        by[h["pair"]].append(h)
    c = collections.defaultdict(collections.Counter)
    ex = []
    for pid, hs in sorted(by.items()):
        side = {h["side"]: h for h in hs}
        if set(side) != {"left", "right"} or any(side[s]["id"] not in seam_hyp for s in side):
            continue
        L, R = side["left"], side["right"]
        rv = values(L.get("reading", ""), lang)
        cls = long_class(L.get("kind", ""), rv)
        if not cls or not rv:
            continue
        nl = len((L.get("reading_part") or "").split())
        nr = len((R.get("reading_part") or "").split())
        merged = (_region(L["text"], seam_hyp[L["id"]], nl, "left") + " "
                  + _region(R["text"], seam_hyp[R["id"]], nr, "right")).strip()
        mv = values(merged, lang, hyp=True)
        out = "right" if same_value(mv, rv, grouped=cls == "phone") else ("dropped" if not mv else "wrong")
        # the reading's row: the parent of the left half's last labelled word (reading part at its cut edge), else of
        # the right half's first one (src = [[row id, word index], ...])
        if L.get("reading_part"):
            parent = (L.get("src") or [[None]])[-1][0] or (L.get("parents") or [None])[-1]
        else:
            parent = (R.get("src") or [[None]])[0][0] or (R.get("parents") or [None])[0]
        sh = single_hyp.get(parent)
        single = None if sh is None else ("right" if _contiguous(rv, values(sh, lang, hyp=True)) else
                                          ("dropped" if not values(sh, lang, hyp=True) else "wrong"))
        for k in (cls, "all_long") if cls != "year" else (cls,):
            x = c[k]
            x["pairs"] += 1
            x[f"seam_{out}"] += 1
            if single is not None:
                x["single_known"] += 1
                x[f"single_{single}"] += 1
                if single == "right" and out != "right":
                    x[f"merge_made_{out}"] += 1
                    if len(ex) < 8 and k == "all_long":
                        ex.append({"pair": pid, "class": cls, "pos": L.get("pos"), "reading": L.get("reading"),
                                   "left_hyp": seam_hyp[L["id"]][-80:], "right_hyp": seam_hyp[R["id"]][:80],
                                   "merged": merged, "single": sh[:120]})
    res = {k: dict(v) for k, v in c.items()}
    res["examples"] = ex
    return res


def _tot(per_lang: dict, langs, key: str) -> int:
    return sum(per_lang.get(l, {}).get("all_long", {}).get(key, 0) for l in langs)


def gate(metrics: dict, base: str, en_rule: str = "base") -> dict:
    """metrics[run][lang] = score_pairs output. Returns {"runs": {run: {pass, reasons, ...}}, "vetoed": {run: reasons}}.
    en_rule "base" (default, calibrated on B): English merge-made wrong <= the baseline's; "zero": must be 0. The
    Python proxy over-counts (B: 14 / 59 English long pairs here vs 0 / 150 on the streaming path), so "zero" vetoes every
    candidate; the streaming-path offset score (--offset-work) is the primary seam veto."""
    b = metrics[base]
    out, vetoed = {}, {}
    w1 = [l for l in WAVE1 if l in b]
    for run, m in metrics.items():
        if run == base:
            continue
        reasons = []
        mw, bw = _tot(m, w1, "merge_made_wrong"), _tot(b, w1, "merge_made_wrong")
        if mw > MERGE_WRONG_FACTOR * bw or (bw == 0 and mw > 0):
            reasons.append(f"wave-1 merge-made wrong {mw} > {MERGE_WRONG_FACTOR} x B {bw}")
        if not w1:
            reasons.append("no wave-1 seam pairs measured")
        if "en" in m:
            e = m["en"].get("all_long", {}).get("merge_made_wrong", 0)
            eb = b.get("en", {}).get("all_long", {}).get("merge_made_wrong", 0)
            if en_rule == "zero" and e:
                reasons.append(f"English merge-made wrong {e} (must be 0; B {eb})")
            elif en_rule == "base" and e > eb:
                reasons.append(f"English merge-made wrong {e} > B {eb}")
        else:
            reasons.append("English seam pairs not measured")
        for name, langs in (("wave-1", w1), ("English", ["en"] if "en" in m else [])):
            if not langs:
                continue
            sr = _tot(m, langs, "single_wrong") + _tot(m, langs, "single_dropped")
            sb = _tot(b, langs, "single_wrong") + _tot(b, langs, "single_dropped")
            if sr > sb:
                reasons.append(f"{name} single-decode wrong + dropped {sr} > B {sb}")
        per_lang = {l: {"merge_made_wrong": m[l].get("all_long", {}).get("merge_made_wrong", 0),
                        "base": b.get(l, {}).get("all_long", {}).get("merge_made_wrong", 0)}
                    for l in m if l in NEXT_MODERATE or l in WAVE1}
        for l, v in per_lang.items():
            v["pass"] = v["merge_made_wrong"] <= v["base"]
        out[run] = {"pass": not reasons, "reasons": reasons, "wave1_merge_made_wrong": [mw, bw],
                    "per_lang_seam_check": per_lang}
        if reasons:
            vetoed[run] = reasons
    return {"base": base, "runs": out, "vetoed": vetoed}


NON_REGRESSION = ("es", "fr", "de", "en")  # streaming-path seam breaks at B are ~0-4 here: the gate is "no worse than B"
REDUCE = ("it", "nl", "pt") + NEXT_MODERATE  # B breaks more here: reported against a target of <= 0.5 x B
BOOT_N, BOOT_ALPHA = 2000, 0.10
MIN_GROUPS = 10  # fewer files than this in a scope: the bootstrap resamples sentences instead


import score_values as SV  # noqa: E402  (the one slot matcher, shared with offset_score.py)


def _gap_has_number(gaps: list, lang: str) -> bool:
    """A merge gap whose two window readings (A / B) or result (R) hold a number: a merge splice can touch a value."""
    return any(values(" ".join(str(g.get(k, "")) for k in ("A", "B", "R")), lang, hyp=True) for g in gaps or [])


def offset_detail(work: str, arm: str, offsets: int = 5) -> dict:
    """{lang: {(sentence id, slot): [(right, merge-touched), one per window offset]}, "_groups": {sentence id: file}}
    from seam_number_test's cases_<arm>.jsonl + parse_out_<arm>.jsonl (the streaming path: 5 window offsets per sentence).
    A row without its long parse is an error (never a wrong reading); every slot must have `offsets` decodes."""
    rows = [json.loads(l) for l in open(f"{work}/cases_{arm}.jsonl", encoding="utf-8") if l.strip()]
    P = SV.load_parse(f"{work}/parse_out_{arm}.jsonl")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    groups = {}
    for k, r in enumerate(rows):
        if f"{k}:long" not in P:
            raise KeyError(f"{arm}: parse_out has no {k}:long (sentence {r['id']})")
        right = SV.match_slots(r["values"], P[f"{k}:long"])
        touched = _gap_has_number(r.get("merge_gaps"), r["lang"])
        groups[r["id"]] = r["file"].rsplit("_o", 1)[0]
        for slot in r["values"]:
            out[r["lang"]][(r["id"], slot)].append((right[slot], touched))
    bad = [(l, k, len(v)) for l, d in out.items() for k, v in d.items() if len(v) != offsets]
    if bad:
        raise ValueError(f"{arm}: {len(bad)} sentence slots without {offsets} decodes, e.g. {bad[:3]}")
    # one sentence = one slot set at every offset (a row of another offset that names other slots is a broken bundle)
    sets = collections.defaultdict(list)
    for r in rows:
        sets[(r["lang"], r["id"])].append(frozenset(r["values"]))
    split = [k for k, v in sets.items() if len(set(v)) > 1]
    if split:
        raise ValueError(f"{arm}: {len(split)} sentences whose slot set differs between window offsets, e.g. {split[:3]}")
    out["_groups"] = groups
    return out


def slot_scores(xs: list) -> dict:
    """One sentence slot over its offsets: total = wrong readings (seam + always: the two trade off, so total is the
    veto); seam = wrong readings while right at another offset; seam_edge = those wrong readings at an offset whose
    merge spliced a number (edge-conditioned: a merge gap at the number); always = readings of a slot wrong at every
    offset."""
    wrong = [not r for r, _ in xs]
    if not any(wrong) or all(wrong):
        return {"total": sum(wrong), "seam": 0, "seam_edge": 0, "always": len(xs) if all(wrong) else 0}
    return {"total": sum(wrong), "seam": sum(wrong), "seam_edge": sum(w and t for w, (_, t) in zip(wrong, xs)),
            "always": 0}


def boot_diff(run: dict, base: dict, key: str, langs, seed: int = 0) -> dict:
    """Paired bootstrap of the total run - B of `key` over the sentence slots both arms decoded, resampling whole
    files (the "_groups" map; a sentence without one is its own group: slots of one file share a voice and a window
    grid, so they are not independent), or sentences when the scope has fewer than MIN_GROUPS files. Returns its 90%
    CI and a sign test over the slots that moved."""
    import math
    import random
    groups = {**base.get("_groups", {}), **run.get("_groups", {})}
    pairs = []
    for l in langs:
        if l not in run or l not in base:
            continue
        for s in base[l]:
            if s in run[l]:
                pairs.append((s[0], slot_scores(run[l][s])[key], slot_scores(base[l][s])[key]))
    if not pairs:
        return {"n_slots": 0}
    by = collections.Counter()
    for sid, a, b in pairs:
        by[groups.get(sid, sid)] += a - b
    level = "file"
    if len(by) < MIN_GROUPS:  # too few files to resample (the synthetic bank has 2 per language): sentences
        level, by = "sentence", collections.Counter()
        for sid, a, b in pairs:
            by[sid] += a - b
    pairs = [(a, b) for _, a, b in pairs]
    d = list(by.values())
    rng = random.Random(seed)
    tots = sorted(sum(rng.choice(d) for _ in d) for _ in range(BOOT_N))
    lo, hi = tots[int(BOOT_ALPHA / 2 * BOOT_N)], tots[int((1 - BOOT_ALPHA / 2) * BOOT_N) - 1]
    worse, better = sum(a > b for a, b in pairs), sum(a < b for a, b in pairs)
    n, k = worse + better, min(worse, better)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n_slots": len(pairs), "n_groups": len(d), "group_level": level, "run": sum(a for a, _ in pairs), "base": sum(b for _, b in pairs),
            "diff": sum(a - b for a, b in pairs), "ci90": [lo, hi], "worse": worse, "better": better, "sign_p": p}


SCOPES = (("wave-1", WAVE1), ("es", ("es",)), ("fr", ("fr",)), ("de", ("de",)), ("en", ("en",)))


def slot_set(d: dict, langs) -> set:
    return {(l, k) for l in langs for k in d.get(l, {})}


def scope_problems(x: dict, b: dict) -> list:
    """Reasons the new arm's slots cannot be compared with B's: for every gated scope (wave-1 pooled, es, fr, de, en)
    each language must have slots (n > 0) and the same slot set in both arms. A missing scope, language or slot is a
    problem to name, never a skipped comparison."""
    out = []
    for name, langs in SCOPES:
        for l in langs:
            sx, sb = slot_set(x, [l]), slot_set(b, [l])
            where = f"scope {name}" + (f" language {l}" if len(langs) > 1 else "")
            if not sb:
                out.append(f"{where}: baseline has no slots")
            if not sx:
                out.append(f"{where}: arm has no slots")
            if sx and sb and sx != sb:
                out.append(f"{where}: slot sets differ (arm-only {len(sx - sb)}, baseline-only {len(sb - sx)})")
    return out


def lost_word_rate(rows: list) -> dict:
    """Seam lost-word rate of one arm from its cases rows: words of the sentence decoded alone ("single") that the same
    sentence inside the stitched file ("long") lacks, per 100 words of "single" (difflib on normalised words, net
    deletions). {lang: [lost, words]} plus "all". Report only, no veto."""
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if "long" not in r or "single" not in r:
            continue
        a, b = [w for w in (_w(x) for x in r["single"].split()) if w], [w for w in (_w(x) for x in r["long"].split()) if w]
        lost = sum(max(0, (i2 - i1) - (j2 - j1)) for op, i1, i2, j1, j2 in
                   difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op in ("delete", "replace"))
        for k in (r["lang"], "all"):
            c[k][0] += lost
            c[k][1] += len(a)
    return dict(c)


def load_cases(work: str, arm: str) -> list:
    return [json.loads(l) for l in open(f"{work}/cases_{arm}.jsonl", encoding="utf-8") if l.strip()]


def offset_gate(details: dict, base_arm: str, arm_of: dict, key: str = "total") -> dict:
    """Streaming-path seam veto (calibrated on B): a run is vetoed when its TOTAL wrong readings (seam + always; seam alone
    rewards turning a 1-of-5 break into 5-of-5) are SIGNIFICANTLY above B's (paired file-cluster bootstrap 90% CI
    lower bound > 0) pooled over wave 1, or in any of es / fr / de / en (non-regression). it / nl / pt and the next /
    moderate tiers are reported against a reduction target (<= 0.5 x B), not vetoed.
    A run whose slots cannot be compared with B's (a scope, language or slot missing or different: scope_problems) is
    INVALID: vetoed with named reasons and listed under "invalid", never passed on what remained.
    Returns {"vetoed": {run: reasons}, "report": {run: {...}}, "invalid": {run: reasons}}."""
    b = details[base_arm]
    vet, rep, inv = {}, {}, {}
    for run, arm in arm_of.items():
        if arm == base_arm:
            continue
        if arm not in details:
            inv[run] = vet[run] = [f"INVALID: no streaming-path details for arm {arm}"]
            continue
        x, why, r = details[arm], [], {}
        bad = scope_problems(x, b)
        if bad:
            inv[run] = vet[run] = ["INVALID: " + p for p in bad]
            continue
        r["wave1"] = boot_diff(x, b, key, WAVE1)
        if r["wave1"].get("n_slots") and r["wave1"]["ci90"][0] > 0:
            why.append(f"streaming-path {key} wave-1 {r['wave1']['run']} > B {r['wave1']['base']} (CI90 {r['wave1']['ci90']})")
        for l in NON_REGRESSION:
            r[l] = boot_diff(x, b, key, [l])
            if r[l].get("n_slots") and r[l]["ci90"][0] > 0:
                why.append(f"streaming-path {key} {l} {r[l]['run']} > B {r[l]['base']} (CI90 {r[l]['ci90']})")
        for l in REDUCE:
            q = boot_diff(x, b, key, [l])
            if q.get("n_slots"):
                q["target_met"] = q["run"] <= 0.5 * q["base"]
                r[l] = q
        rep[run] = r
        if why:
            vet[run] = why
    return {"vetoed": vet, "report": rep, "invalid": inv}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--base", required=True)
    ap.add_argument("--langs", default=",".join(WAVE1 + NEXT_MODERATE + ("en",)))
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "spoken_form_seam"))
    ap.add_argument("--json", help="result file; REQUIRED with --offset-work (one file per bed, so beds never overwrite)")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--en-rule", default="base", choices=["zero", "base"],
                    help="English merge-made wrong: <= B (default; the Python proxy over-counts: B 14/59 vs 0/150 on "
                         "the streaming path) or 0")
    ap.add_argument("--offset-key", default="total", choices=["total", "seam_edge", "seam"])
    ap.add_argument("--offset-work", default="", help="seam_number_test workdir: also veto on offset_score.py vs B")
    ap.add_argument("--baseline-work", default="",
                    help="seam_number_test workdir of the baseline's files when they are not in --offset-work")
    ap.add_argument("--offset-arms", default="", help="run=arm,... (B included): the seam_number_test arm of each run")
    a = ap.parse_args(argv)
    if a.offset_work and not a.json:
        ap.error("--json is required with --offset-work (a shared default path would overwrite the other bed)")
    work, fetch = Path(a.work), not a.no_fetch
    runs = [a.base] + [r for r in a.runs.split(",") if r != a.base]
    metrics = {}
    for lang in a.langs.split(","):
        pairs = _get(pairs_path(lang), work / pairs_path(lang), fetch)
        if not pairs:
            print(f"{lang}: no held-out seam pairs")
            continue
        for run in runs:
            sh = {h["id"]: h["hyp"] for h in _get(f"hyps/{run}/hseam_{lang}.jsonl", work / "hyps" / run / f"hseam_{lang}.jsonl", fetch)}
            single = {h["id"]: h["hyp"] for h in _get(f"hyps/{run}/hseam_parents_{lang}.jsonl",
                                                       work / "hyps" / run / f"hseam_parents_{lang}.jsonl", fetch)}
            if not sh:
                print(f"{run} {lang}: no seam hyps (run modal_app.py::probe_seam)")
                continue
            r = metrics.setdefault(run, {})[lang] = score_pairs(pairs, sh, single, lang)
            al = r.get("all_long", {})
            print(f"{run:28s} {lang:3s} long pairs {al.get('pairs', 0):5d}  seam wrong {al.get('seam_wrong', 0):4d} "
                  f"dropped {al.get('seam_dropped', 0):4d}  merge-made wrong {al.get('merge_made_wrong', 0):4d} "
                  f"dropped {al.get('merge_made_dropped', 0):4d}  single wrong+dropped "
                  f"{al.get('single_wrong', 0) + al.get('single_dropped', 0):4d}/{al.get('single_known', 0)}")
    res = {"metrics": metrics, "vetoed": {}, "invalid": {}}
    if a.base in metrics:  # the Python proxy over-counts: printed, never a veto
        px = gate(metrics, a.base, a.en_rule)
        res["proxy"] = px
        for run, g in px["runs"].items():
            print(f"PROXY (report only) {run}: {'PASS' if g['pass'] else 'WOULD VETO'} " + "; ".join(g["reasons"]))
    if a.offset_work:  # the streaming-path offset score: total wrong readings vs B, paired bootstrap
        arm_of = dict(x.split("=", 1) for x in a.offset_arms.split(",") if x)
        assert a.base in arm_of, "--offset-arms must name the baseline's arm (B=<arm>)"
        det = {}
        for run, arm in arm_of.items():  # keyed by run: the same arm name may exist in two work dirs
            wd = a.baseline_work if run == a.base and a.baseline_work else a.offset_work
            det[run] = offset_detail(wd, arm)
            res.setdefault("lost_word_rate", {})[run] = lost_word_rate(load_cases(wd, arm))
        og = offset_gate(det, a.base, {r: r for r in arm_of}, a.offset_key)
        res["offset"] = og
        res["vetoed"] = og["vetoed"]
        res["invalid"] = og["invalid"]
        for run, rates in res["lost_word_rate"].items():
            al = rates.get("all", [0, 0])
            print(f"LOST-WORDS {run}: {100 * al[0] / al[1] if al[1] else 0:.2f}% of single-decode words lost in the "
                  f"stitched decode ({al[0]}/{al[1]}; report only)")
        for run, r in og["report"].items():
            print(f"OFFSET {run}: {'VETO ' + '; '.join(og['vetoed'][run]) if run in og['vetoed'] else 'PASS'}")
            for k, v in r.items():
                if v.get("n_slots"):
                    print(f"   {k:6s} {a.offset_key} {v['run']:4d} vs B {v['base']:4d}  diff {v['diff']:+4d} CI90 {v['ci90']}"
                          f" worse/better {v['worse']}/{v['better']} sign p={v['sign_p']:.2g}"
                          + (f"  target<=0.5xB {'met' if v['target_met'] else 'NOT met'}" if "target_met" in v else ""))
        for run, why in og["invalid"].items():
            print(f"OFFSET {run}: INVALID (cannot be compared with B): " + "; ".join(why), file=sys.stderr)
    else:
        print("no streaming-path offset gate was run (--offset-work): no veto issued", file=sys.stderr)
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 2 if res["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
