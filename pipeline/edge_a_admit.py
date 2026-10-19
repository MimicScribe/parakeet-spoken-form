"""How much of the edge pack the synthetic-share caps would admit (design section 5), per language: for every gate class
the pack's synthetic spans go on top of the synthetic spans a built mix already holds, and mix_build.syn_room (the
build's own rule: share caps, fill-to-bar, rare-class floor) says how many of them fit. Rows are admitted whole, so the
row share is the share of rows whose every class still has room (classes are filled in the order the rows come).

    python edge_a_admit.py --mix <dir with p3_<lang>.jsonl / tts_<lang>.jsonl> --rows final_es.jsonl --lang es [--out admit_es.json]
"""
import argparse
import collections
import json

import mix_build as MB


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def estimate(mix: str, rows: list, lang: str, extra_syn: list | None = None) -> dict:
    real, syn = _rows(f"{mix}/p3_{lang}.jsonl"), _rows(f"{mix}/tts_{lang}.jsonl") + (extra_syn or [])
    rn, rv = MB.class_values(real, lang)
    sn, sv = MB.class_values(syn, lang)
    sh = MB._syn_hours(syn, lang)
    # per row: the classes of its spans (the build counts a span in its class), its hours
    add_n, add_h = collections.Counter(), collections.Counter()
    row_cls = []
    for r in rows:
        cl = [MB.gate_class_value(r, sp, lang) for sp in r.get("spans") or []]
        row_cls.append(cl)
        for c, _ in cl:
            add_n[c] += 1
        for c in {c for c, _ in cl}:
            add_h[c] += r.get("duration", 0.0) / 3600
    # greedy over the rows in file order: a row is taken when every class it holds still has room after it
    take, got_n, got_h = [], collections.Counter(), collections.Counter()
    for r, cl in zip(rows, row_cls):
        cs = collections.Counter(c for c, _ in cl)
        ok = True
        for c, k in cs.items():
            if not MB.syn_room(lang, c, sn[c] + got_n[c] + k, rn[c], sh[c] + got_h[c] + r.get("duration", 0.0) / 3600,
                               rv.get(c), sv.get(c)):
                ok = False
                break
        if ok:
            take.append(True)
            for c, k in cs.items():
                got_n[c] += k
                got_h[c] += r.get("duration", 0.0) / 3600
        else:
            take.append(False)
    by_class = {}
    for c in sorted(add_n):
        by_class[c] = {"pack_spans": add_n[c], "admitted_spans": got_n[c], "real_spans": rn[c], "mix_syn_spans": sn[c],
                       "share": round(got_n[c] / add_n[c], 3) if add_n[c] else None}
    fam = collections.defaultdict(lambda: [0, 0])
    for r, t in zip(rows, take):
        f = (r.get("family") or r.get("edge", ["?"])[0])
        fam[f][0] += 1
        fam[f][1] += t
    return {"lang": lang, "rows": len(rows), "rows_admitted": sum(take), "row_share": round(sum(take) / max(len(rows), 1), 3),
            "span_share": round(sum(got_n.values()) / max(sum(add_n.values()), 1), 3),
            "hours": round(sum(r.get("duration", 0.0) for r in rows) / 3600, 3),
            "hours_admitted": round(sum(r.get("duration", 0.0) for r, t in zip(rows, take) if t) / 3600, 3),
            "by_class": by_class, "by_family_rows": {k: {"rows": v[0], "admitted": v[1], "share": round(v[1] / v[0], 3)} for k, v in fam.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = estimate(a.mix, _rows(a.rows), a.lang)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "by_class"}))
    for c, v in res["by_class"].items():
        print(c, v)


if __name__ == "__main__":
    main()
