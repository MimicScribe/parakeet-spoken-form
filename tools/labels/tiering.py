#!/usr/bin/env python3
"""Tiering rule + expected share and cost of paid referee passes (stdlib; reads a ledger db).
Rule (free evidence first; the paid referee only when free evidence is not enough):
  Tier 0  human_transcript rows: the label IS a human reference; no paid pass (CTC form check only).
  Tier 1  accept on FREE evidence: status scored, min margin >= T (calibrated, calibrate_margin.py), no tie, free CTC greedy value equals the label value,
          and any stored votes from other free decoders agree. No paid pass.
  Tier 2  paid referee (a hosted ASR, priced per audio minute) when ANY of: status undecided / cer_high / align_fail / error; margin < T or tie; free greedy value differs;
          class is high risk (digit runs >= 7 digits = phone / long digits, codes with letters, adjacent repeated digits, self-corrections / repeats);
          caption rows whose caption IS a machine transcript need the referee AND CTC to agree (a machine caption never decides alone).
  Tier 3  still disputed after Tier 2 -> code referee, text LLM referee (calibrated), a second paid system on a small set, a human listens.
Cost: --usd-per-min is the referee's price per audio minute (planning default 0.004; set it from your provider's price list).
usage: python3 tiering.py --db ledger.db [--out tiering_report.json] [--usd-per-min 0.004]"""
import argparse, collections, json, os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ledger as LG

USD_PER_MIN = 0.004  # planning default; override with --usd-per-min
HIGH_RISK_KINDS = ("digits_long", "code", "repeat")


def span_risk(kind, value, prefix, suffix, reading):
    r = []
    if kind == "digits" and len(re.sub(r"\D", "", value or "")) >= 7:
        r.append("digits_long")
    if prefix or re.search(r"[A-Z]", " ".join(suffix or [])):
        r.append("code")
    if (kind in ("digits", "pairs") and re.search(r"\b(\w+) \1\b", reading or "")) or re.search(r"\b(double|triple)\b", reading or ""):
        r.append("repeat")
    return r


def tier_of(rec, thr, free_value_equal, votes_agree, high_risk, source_class):
    """post-run tier for one scored record (see module doc). returns 0..3"""
    if source_class == "human_transcript":
        return 0
    bad = rec["status"] != "scored" or rec.get("min_margin", 99) < thr or rec.get("any_tie") or not free_value_equal or not votes_agree
    if bad or high_risk:
        return 2
    return 1


def estimate(db_path, usd_per_min=USD_PER_MIN):
    L = LG.Ledger(db_path)
    q = lambda s, *a: [dict(r) for r in L.db.execute(s, a)]
    rows = {}
    for r in q("""SELECT r.row_id, r.part, r.source_class, r.real_synth, r.duration d FROM rows r
                  WHERE EXISTS (SELECT 1 FROM spans s WHERE s.row_id=r.row_id)"""):
        rows[r["row_id"]] = dict(r, risk=set(), multi=False)
    for s in q("SELECT row_id, kind, value, reading, cands, flags FROM spans"):
        fl = json.loads(s["flags"] or "{}"); c = json.loads(s["cands"] or "[]")
        row = rows.get(s["row_id"])
        if not row:
            continue
        if len(c) > 1 and not fl.get("weak"):
            row["multi"] = True
        for k in span_risk(s["kind"], s["value"], fl.get("prefix"), fl.get("suffix"), s["reading"]):
            row["risk"].add(k)
    pop = [r for r in rows.values() if r["multi"]]
    out = dict(rows_with_number_spans=len(rows), rows_needing_form_choice=len(pop))
    def agg(sel):
        h = sum((r["d"] or 0) for r in sel) / 3600
        return dict(rows=len(sel), hours=round(h, 2), usd_if_all=round(h * 60 * usd_per_min, 2))
    by = collections.defaultdict(list)
    for r in pop:
        by[(r["source_class"], r["real_synth"])].append(r)
    out["tier_all_referee_every_row"] = {"%s/%s" % k: agg(v) for k, v in sorted(by.items())}
    out["tier_all_referee_every_row"]["TOTAL"] = agg(pop)
    # tier 0: human_transcript needs no paid pass; synthetic rows: the script is the truth, the referee only on high-risk classes (garble grows with run length)
    real = [r for r in pop if r["source_class"] in ("caption", "stock_decode")]
    syn = [r for r in pop if r["source_class"] == "synthetic_script"]
    out["real_caption_or_stock_rows"] = agg(real)
    out["synthetic_rows"] = agg(syn)
    rrisk = [r for r in real if r["risk"]]; rrest = [r for r in real if not r["risk"]]
    srisk = [r for r in syn if r["risk"]]
    out["high_risk_rows"] = dict(real=dict(agg(rrisk), by_risk={k: sum(1 for r in rrisk if k in r["risk"]) for k in HIGH_RISK_KINDS}),
                                 synthetic=dict(agg(srisk), by_risk={k: sum(1 for r in srisk if k in r["risk"]) for k in HIGH_RISK_KINDS}))
    hr = lambda sel: sum((r["d"] or 0) for r in sel) / 3600
    scen = {}
    for name, p in (("low", 0.10), ("planning", 0.20), ("high", 0.35)):
        mins = (hr(rrisk) + p * hr(rrest) + hr(srisk)) * 60
        rows_n = len(rrisk) + int(p * len(rrest)) + len(srisk)
        scen[name] = dict(unresolved_share_of_real_rest=p, referee_rows=rows_n, referee_min=round(mins), usd=round(mins * usd_per_min, 2),
                          real_only_usd=round((hr(rrisk) + p * hr(rrest)) * 60 * usd_per_min, 2))
    out["scenarios_tiered"] = scen
    out["usd_per_min"] = usd_per_min
    out["note"] = ("unresolved share is an ASSUMPTION; run the labeller on a random sample and replace it with the measured share of undecided + cer_high + free-greedy mismatch rows. "
                   "Human-transcript rows are excluded. Quote the cost and get approval before any paid run.")
    L.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="tiering_report.json")
    ap.add_argument("--usd-per-min", type=float, default=USD_PER_MIN)
    a = ap.parse_args(argv)
    rep = estimate(a.db, a.usd_per_min)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
