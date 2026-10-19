"""Which unused labelled real rows may join the build as an extra pass (realx1), and how a confirmed row is written.

Pure (no audio, no Modal). The caller supplies the pool rows (one per id), the rows the mix already trains on, and the
held-out / validation / evaluation rows of the language.

  eligible(row)            the cheap, row-level conditions (word times, length, no letter+digit codes, label fits the reader)
  candidates(...)          unused rows that pass eligible() and duplicate neither a held-out / evaluation row nor a mix row
  priority(row, ...)       thin cells first: classes under twice their bar, long readings, values under twice their fill floor
  select(...)              greedy by priority under the build's own caps counted against the mix: <= 13 rows per session,
                           <= 3 per caption template, no year value above 3% of the year spans
  promote_row(row, scores) a row whose second-transcript-decided spans all passed realx1_core.span_ok: those spans are
                           written as CTC-decided ("by": "ctc"; "by_orig" and the margins kept), the row flagged "realx1"

Spans decided by the CTC itself keep the labelling bar; spans decided mainly by the second transcript must pass the four
probes of realx1_core. A row with any failing span is not promoted.
"""

import collections
import json
import random
import re

import census_common as CC
import ml_balance as B

BAD_KINDS = {"alnum", "amp", "code", "roman"}  # letter+digit codes are re-labelled by the build (alnum policy): left out
TOP_YEAR = 0.03
FILL_FLOOR = {"year_main": 42, "year_fwd": 11, "int": 11}  # mix_build.FILL_FLOOR


def eligible(r: dict) -> str | None:
    """None, or why the row cannot be an extra-pass row."""
    sp = r.get("spans") or []
    if not sp or len(sp) != len(r.get("num_chars") or []):
        return "span_mismatch"
    if any(s.get("kind") in BAD_KINDS for s in sp):
        return "letter_digit_code"
    wt = r.get("word_times")
    if not wt or any(t is None for t in wt) or len(wt) != len(r["text"].split()):
        return "no_word_times"
    if not 1.5 <= r.get("duration", 0) <= 20:
        return "duration"
    if (r.get("cer_ctc") if r.get("cer_ctc") is not None else 1.0) > 0.25:
        return "cer_ctc"
    if re.search(r"\d", r["text"]):
        return "digit_in_label"
    return None


def candidates(unused: list, mix: list, protected: list) -> tuple:
    """(rows, counts): unused rows that pass eligible() and are no duplicate (text / audio) of a protected row
    (held-out, validation, evaluation), of a mix row, or of each other."""
    st, keep = collections.Counter(), []
    for r in unused:
        why = eligible(r)
        if why:
            st["drop_" + why] += 1
        else:
            keep.append(r)
    k1, _ = B.dedupe(keep, protected, near=True, within=False, part_of=lambda x: "x")
    st["drop_dup_protected"] = len(keep) - len(k1)
    k2, _ = B.dedupe(k1, mix, near=True, within=False, part_of=lambda x: "x")
    st["drop_dup_mix"] = len(k1) - len(k2)
    k3, _ = B.dedupe(k2, [], near=True, within=True, part_of=lambda x: "x")
    st["drop_dup_within"] = len(k2) - len(k3)
    return k3, dict(st)


def span_values(r: dict, lang: str) -> list:
    """("year", v) / ("int", v <= 100) keys of the thin-value protection (mix_build._span_values)."""
    out = []
    for sp in r.get("spans") or []:
        c, v = CC.gate_class_value(r, sp, lang)
        if c == "year" and isinstance(v, int):
            out.append(("year", v))
        elif c == "int_small" and isinstance(v, int) and v <= 100:
            out.append(("int", v))
    return out


def fill_floor(v: tuple) -> int:
    kind, x = v
    if kind == "year":
        return FILL_FLOOR["year_main"] if 1990 <= x <= 2030 else FILL_FLOOR["year_fwd"] if 2031 <= x <= 2060 else 0
    return FILL_FLOOR["int"]


def class_weights(lang: str, mix: list) -> dict:
    """Per class: 1 - count / (2 x bar), clipped to [0.03, 1]; classes with no bar 0.03."""
    n = collections.Counter(CC.gate_class(r, sp, lang) for r in mix for sp in r["spans"] if "text" in sp)
    out = {}
    for c, cnt in n.items():
        bar = CC.class_bar(lang, c) or 0
        out[c] = 0.03 if bar == 0 else max(0.03, min(1.0, 1 - cnt / (2 * max(bar, 150))))
    return out


def priority(r: dict, lang: str, w: dict, vc: collections.Counter) -> float:
    s = 0.0
    for sp in r["spans"]:
        if "text" not in sp:
            continue
        s += w.get(CC.gate_class(r, sp, lang), 0.1)
        nw = len((sp.get("reading") or "").split())
        s += 0.6 if nw >= 6 else 0.2 if nw >= 3 else 0
    s += 0.3 * sum(1 for v in span_values(r, lang) if vc[v] < 2 * fill_floor(v))
    return s / (max(r["duration"], 1.0) ** 0.5)


def select(cands: list, mix: list, lang: str, hours: float, seed: int = 7) -> tuple:
    """(rows, skipped by reason): greedy by priority under the session / template / top-year caps counted against the mix,
    until `hours` of audio."""
    rng = random.Random(f"{seed}-{lang}")
    w = class_weights(lang, mix)
    vc = collections.Counter(v for r in mix for v in span_values(r, lang))
    ng = collections.Counter(CC.group_id(r) for r in mix if r.get("spans"))
    nt = collections.Counter(CC.template_key(r) for r in mix if r.get("spans"))
    yc = collections.Counter(y for r in mix for sp in r["spans"] for y in [CC.year_of(r, sp, lang)] if y)
    ytot = sum(yc.values())
    out, acc, skip = [], 0.0, collections.Counter()
    for r in sorted(cands, key=lambda r: (-priority(r, lang, w, vc), rng.random())):
        if acc >= hours * 3600:
            break
        g, t = CC.group_id(r), CC.template_key(r)
        if not g.startswith("id:") and ng[g] + 1 > CC.group_cap(lang, g):
            skip["session_cap"] += 1
            continue
        if t and nt[t] + 1 > CC.TEMPLATE_CAP:
            skip["template_cap"] += 1
            continue
        ys = [y for y in (CC.year_of(r, sp, lang) for sp in r["spans"]) if y]
        if ys and any(yc[y] + 1 > max(5, TOP_YEAR * (ytot + len(ys))) for y in ys):
            skip["year_cap"] += 1
            continue
        ng[g] += 1
        nt[t] += 1
        yc.update(ys)
        ytot += len(ys)
        vc.update(span_values(r, lang))
        out.append(r)
        acc += r["duration"]
    return out, dict(skip)


def promote_row(r: dict, span_scores: list) -> dict:
    """The extra-pass row: a copy of r whose second-transcript-decided spans are CTC-decided ("by": "ctc", the old value
    in "by_orig", the four margins in "ctc_confirm"); span_scores[j] = {"m": {"cand", "del", "wdel", "wsub"}}."""
    r2 = json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))
    was_ext = CC.ext_row(r2)
    for sp, sc in zip(r2["spans"], span_scores):
        if CC.ext_span(sp):
            sp["by_orig"], sp["by"] = sp["by"], "ctc"
            sp["ctc_confirm"] = {k: sc["m"][k] for k in ("cand", "del", "wdel", "wsub")}
    r2["realx1"] = {"status": "ctc_confirmed" if was_ext else "non_ext", "pass": r.get("_pass")}
    return r2
