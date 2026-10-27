#!/usr/bin/env python3
"""Margin calibration on human-form rows + the keep-current-label fallback (stdlib).
Known answer: in human-transcript rows (a person transcribed the speech) the CURRENT reading is how a human wrote it, so a span is CORRECT when the labeller's chosen reading equals it.
  python3 calibrate_margin.py curve   <scored.jsonl>... --gold <manifest.jsonl>   accuracy / coverage vs margin, per alternative type (oh/zero, a-vs-one, and, other)
  python3 calibrate_margin.py decide  <scored.jsonl>... --threshold T [--db ledger.db]   apply the fallback rule (writes DECISIONS when --db is given)
Fallback rule: status scored AND min margin >= T AND no tie -> accept the chosen label (relabel if it differs, keep otherwise);
anything else (undecided, cer_high, align_fail, error, below T) -> action `unconfirmed`, the CURRENT label stays. Nothing is rejected or dropped."""
import argparse, collections, json, math, os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import labeller_core as C

BARS = dict(acc=0.90, wilson_lb=0.85, coverage=0.70)


def wilson(k, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def alt_type(cur, chosen):
    """what kind of disagreement is it (for the per-type report)"""
    if cur == chosen:
        return "same"
    a, b = cur.lower().split(), chosen.lower().split()
    if C.canon_key(cur) == C.canon_key(chosen) and re.search(r"\b(oh|zero|double)\b", cur + " " + chosen):
        return "oh_zero_double"
    if re.sub(r"\b(a|one)\b", "N", cur.lower()) == re.sub(r"\b(a|one)\b", "N", chosen.lower()):
        return "a_vs_one"
    if [w for w in a if w != "and"] == [w for w in b if w != "and"]:
        return "and"
    if sorted(a) == sorted(b):
        return "order"
    if abs(len(a) - len(b)) <= 1 and set(a) ^ set(b) and len(set(a) ^ set(b)) <= 2 and any(w in ("five", "two", "three", "four", "six", "seven", "eight", "nine", "one") for w in a):
        return "repeat_or_digit"
    return "other"


def load_jsonl(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def span_table(scored, gold_ids):
    """rows -> list of dicts for every span with ncand > 1 of a gold row whose status is not error / align_fail-with-no-spans"""
    t = []
    for r in scored:
        if r["id"] not in gold_ids:
            continue
        for s in r.get("spans") or []:
            if s.get("ncand", 1) < 2 or s.get("cur_loss") is None:
                continue
            t.append(dict(id=r["id"], margin=s["margin"], margin_x=s["margin_x"], tie=s["tie"], correct=s["chosen"] == s["current"], type=alt_type(s["current"], s["top2"][0][0] if s["top2"][0][0] != s["current"] else (s["top2"][1][0] if len(s["top2"]) > 1 else s["current"])),
                          kind=s.get("kind"), status=r["status"], chosen=s["chosen"], current=s["current"]))
    return t


def curve(table, thresholds=(0, 0.25, 0.5, 1, 2, 3, 5, 8, 12)):
    n = len(table); out = []
    for t in thresholds:
        sel = [x for x in table if x["margin"] >= t and not (t > 0 and x["tie"])]
        k = sum(x["correct"] for x in sel)
        lo, hi = wilson(k, len(sel))
        out.append(dict(threshold=t, spans=len(sel), coverage=round(len(sel) / n, 3) if n else 0, correct=k, accuracy=round(k / len(sel), 4) if sel else None, wilson_lb=round(lo, 4)))
    return out


def pick_threshold(cv, bars=BARS):
    """smallest threshold meeting accuracy, Wilson lower bound AND coverage; None when no threshold does"""
    for c in cv:
        if c["accuracy"] is not None and c["accuracy"] >= bars["acc"] and c["wilson_lb"] >= bars["wilson_lb"] and c["coverage"] >= bars["coverage"]:
            return c["threshold"]
    return None


def by_type(table, threshold):
    res = {}
    for ty in sorted({x["type"] for x in table} | {"same"}):
        rows = [x for x in table if (x["type"] == ty)]
        sel = [x for x in rows if x["margin"] >= threshold and not x["tie"]]
        k = sum(x["correct"] for x in sel)
        res[ty] = dict(spans=len(rows), above_threshold=len(sel), correct_above=k, accuracy_above=round(k / len(sel), 3) if sel else None)
    return res


def decide(rec, threshold):
    """-> (action, reason, new_label|None) for one scored record. The fallback rule; never a drop."""
    if rec["status"] in ("error", "align_fail", "cer_high"):
        return "unconfirmed", "status=%s: CTC cannot judge, current label kept" % rec["status"], None
    if rec.get("min_margin", 99.0) < threshold or rec.get("any_tie"):
        return "unconfirmed", "margin %.2f < %.2f or tie: current label kept" % (rec.get("min_margin", 99.0), threshold), None
    if rec.get("changed"):
        return "relabel", "audio-scored form differs from the current label (margin %.2f >= %.2f)" % (rec.get("min_margin", 99.0), threshold), rec["chosen_label"]
    return "keep", "audio-scored form equals the current label (margin %.2f >= %.2f)" % (rec.get("min_margin", 99.0), threshold), None


def apply_to_ledger(L, scored, threshold, ids_to_row_id=None, tool="ctc_labeller"):
    """write one DECISION per scored row (decided_by=rule) plus one evidence record per span (`tool` names the labeller run); returns counts"""
    cnt = collections.Counter()
    for r in scored:
        rid = (ids_to_row_id or {}).get(r["id"], r["id"])
        for s in r.get("spans") or []:
            if s.get("ncand", 1) > 1:
                L.add_evidence(rid, "ctc_cand_scored", tool, value=s["chosen"], form="chosen" if s["chosen"] != s["current"] else "current", margin=s["margin"],
                               score=s["best_loss"], span_ord=s.get("ord"), payload=dict(top2=s["top2"], margin_x=s["margin_x"], tie=s["tie"], same_key_alt=s["same_key_alt"], cur_rank=s["cur_rank"], status=r["status"]))
        act, why, new = decide(r, threshold)
        L.add_decision(rid, act, why, "rule", new_label=new)
        cnt[act] += 1
    L.commit()
    return dict(cnt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve"); c.add_argument("scored", nargs="+"); c.add_argument("--gold", required=True, help="manifest jsonl; rows with gold_form=true are the calibration set")
    d = sub.add_parser("decide"); d.add_argument("scored", nargs="+"); d.add_argument("--threshold", type=float, required=True); d.add_argument("--db"); d.add_argument("--tool", default="ctc_labeller", help="evidence tool name")
    a = ap.parse_args(argv)
    scored = load_jsonl(a.scored)
    if a.cmd == "curve":
        gold = {r["id"] for r in load_jsonl([a.gold]) if r.get("gold_form")}
        tab = span_table(scored, gold)
        cv = curve(tab)
        thr = pick_threshold(cv)
        print(json.dumps(dict(gold_rows=len(gold), spans=len(tab), curve=cv, threshold=thr, by_type=by_type(tab, thr if thr is not None else 99), bars=BARS), indent=1))
    else:
        cnt = collections.Counter(decide(r, a.threshold)[0] for r in scored)
        print(dict(cnt))
        if a.db:
            import ledger as LG
            L = LG.Ledger(a.db); print(apply_to_ledger(L, scored, a.threshold, tool=a.tool)); L.close()


if __name__ == "__main__":
    main()
