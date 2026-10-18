"""Seam score that needs no single-sentence reference: each sentence is decoded at N window offsets (default 5), so a
value right at some offsets and wrong at others was broken by where a window edge fell.
Usage: offset_score.py <workdir> <arm>... [--offsets N] [--lenient] [--guard] [--json out.json]
  an arm is a name (cases_<arm>.jsonl + parse_out_<arm>.jsonl in <workdir>) or <dir>::<name> for another workdir;
  --guard drops slots whose number words sit near a sentence cut at any offset in any arm (the cases' edge_num
  flags); --lenient is the parser-limit sensitivity row (score_values.py).
Per slot (sentence x value) over its N decodes: total wrong = wrong decodes (the HEADLINE: seam + always);
seam = wrong decodes of a slot right at >= 1 offset; always = decodes of a slot wrong at every offset.
Seam and always trade off (a model that turns a 1-of-5 break into 5-of-5 'improves' seam), so arms are compared on
total wrong, paired by slot against the FIRST arm: transition table of right-counts, seam on slots both arms get right
at least once, a paired bootstrap 90% CI of the total-wrong difference (resampling files, and sentences), and a sign
test over the slots that moved. Every row must have its "<k>:long" parse (a missing one is an error, never a wrong
reading) and every slot exactly N decodes."""
import collections
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_values import SL, base_slot, load_parse, match_slots  # noqa: E402

ORDER = "en es fr de it nl pt pl ro uk ru sv cs hu fi sk bg hr da el lt sl".split()
NEAR_S = 1.5
BOOT_N, BOOT_ALPHA = 2000, 0.10


def file_base(f):
    return f.rsplit("_o", 1)[0]


def slot_table(rows, P, offsets=5, lenient=False):
    """{(sentence id, slot): {"lang", "file", "reads": [{off, right, edge, num_t, gaps}], sorted by offset}}.
    Raises KeyError for a row without its long parse and ValueError for a slot without exactly `offsets` decodes."""
    T = {}
    for k, r in enumerate(rows):
        key = f"{k}:long"
        if key not in P:
            raise KeyError(f"parse_out has no {key} (row {k}, sentence {r['id']})")
        right = match_slots(r["values"], P[key], lenient)
        keys = list(r["values"])
        en = r.get("edge_num")
        off = float(r["file"].rsplit("_o", 1)[1]) if "_o" in r["file"] else k
        for j, s in enumerate(keys):
            t = T.setdefault((r["id"], s), {"lang": r["lang"], "file": file_base(r["file"]), "reads": []})
            edge = None if en is None else bool((j == 0 and en[0]) or (j == len(keys) - 1 and en[1]))
            t["reads"].append({"off": off, "right": right[s], "edge": edge, "num_t": r.get("num_t"),
                               "gaps": r.get("file_gaps")})
    bad = {k: len(t["reads"]) for k, t in T.items() if len(t["reads"]) != offsets}
    if bad:
        raise ValueError(f"{len(bad)} sentence slots without {offsets} decodes, e.g. {sorted(bad.items())[:3]}")
    for t in T.values():
        t["reads"].sort(key=lambda x: x["off"])
    return T


def wrong(t):
    return sum(not x["right"] for x in t["reads"])


def seam_counts(rows, P, offsets=5, lenient=False, table=None):
    """{lang: Counter(n, seam, always, total, n_<slot>, seam_<slot>, always_<slot>)}"""
    T = table if table is not None else slot_table(rows, P, offsets, lenient)
    c = collections.defaultdict(collections.Counter)
    for (sid, s), t in T.items():
        x, b, n, w = c[t["lang"]], base_slot(s), len(t["reads"]), wrong(t)
        x["n"] += n; x["n_" + b] += n; x["total"] += w; x["total_" + b] += w
        if 0 < w < n:
            x["seam"] += w; x["seam_" + b] += w
        if w == n:
            x["always"] += w; x["always_" + b] += w
    return c


def guarded_keys(tables):
    """slots never flagged near a sentence cut, at any offset, in any arm"""
    keys = set.intersection(*(set(t) for t in tables))
    return {k for k in keys if not any(x["edge"] for t in tables for x in t[k]["reads"])}


def sign_p(a, b):
    """two-sided exact sign test p for a vs b moves"""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(diffs_by_group, seed=0):
    """paired cluster bootstrap: resample groups (each = summed run - base over its slots); 90% CI of the total"""
    g = list(diffs_by_group.values())
    if not g:
        return [0, 0]
    rng = random.Random(seed)
    tots = sorted(sum(rng.choice(g) for _ in g) for _ in range(BOOT_N))
    return [tots[int(BOOT_ALPHA / 2 * BOOT_N)], tots[int((1 - BOOT_ALPHA / 2) * BOOT_N) - 1]]


def paired(tb, tc, keys=None, langs=None):
    """candidate tc vs base tb over their common slots (or `keys`), optionally only `langs`."""
    ks = set(tb) & set(tc) if keys is None else set(keys) & set(tb) & set(tc)
    if langs is not None:
        ks = {k for k in ks if tb[k]["lang"] in langs}
    out = collections.Counter()
    trans = collections.Counter()
    by_file, by_sent = collections.Counter(), collections.Counter()
    for k in sorted(ks):
        b, c = tb[k], tc[k]
        n = len(b["reads"])
        wb, wc = wrong(b), wrong(c)
        trans[(n - wb, n - wc)] += 1
        out["slots"] += 1; out["readings"] += n
        out["total_base"] += wb; out["total_run"] += wc
        out["seam_base"] += wb if 0 < wb < n else 0; out["seam_run"] += wc if 0 < wc < n else 0
        out["always_base"] += wb if wb == n else 0; out["always_run"] += wc if wc == n else 0
        if wb < n and wc < n:  # both arms right at least once
            out["both_right_slots"] += 1; out["seam_both_base"] += wb; out["seam_both_run"] += wc
        out["worse"] += wc > wb; out["better"] += wc < wb
        by_file[b["file"]] += wc - wb
        by_sent[k[0]] += wc - wb
    r = dict(out)
    r["diff"] = out["total_run"] - out["total_base"]
    r["ci90_file"] = boot_ci(by_file)
    r["ci90_sentence"] = boot_ci(by_sent)
    r["n_files"], r["n_sentences"] = len(by_file), len(by_sent)
    r["sign_p"] = sign_p(out["worse"], out["better"])
    r["transitions"] = {f"{a}->{b}": v for (a, b), v in sorted(trans.items())}
    return r


def near_interior(tables, keys=None, near_s=NEAR_S):
    """Wrong rate of decodes whose number words sit within near_s of a merge point at that offset vs interior ones.
    One partition for every arm: number-word times and merge points (the decoder's merge-gap log, written where two windows
    disagreed) are the union over arms. Returns {arm index: {near: [wrong, n], interior: [wrong, n],
    within_slot_near / within_slot_interior: same, on slots with both kinds}} or None without the cases' num_t / file_gaps fields."""
    ks = set.intersection(*(set(t) for t in tables)) if keys is None else set(keys)
    if not ks or any(x["num_t"] is None for t in tables for x in t[next(iter(ks))]["reads"]):
        return None
    res = {i: collections.defaultdict(lambda: [0, 0]) for i in range(len(tables))}
    for k in ks:
        n = len(tables[0][k]["reads"])
        near = []
        for o in range(n):
            nt = {x for t in tables for x in (t[k]["reads"][o]["num_t"] or [])}
            gp = {x for t in tables for x in (t[k]["reads"][o]["gaps"] or [])}
            near.append(any(abs(a - g) <= near_s for a in nt for g in gp))
        mixed = any(near) and not all(near)
        for i, t in enumerate(tables):
            for o, x in enumerate(t[k]["reads"]):
                for name in ("near" if near[o] else "interior",) + (
                        ("within_slot_" + ("near" if near[o] else "interior"),) if mixed else ()):
                    res[i][name][0] += not x["right"]
                    res[i][name][1] += 1
    return {i: dict(v) for i, v in res.items()}


def load_arm(W, a):
    d, name = a.split("::", 1) if "::" in a else (W, a)
    rows = [json.loads(l) for l in open(f"{d}/cases_{name}.jsonl", encoding="utf-8") if l.strip()]
    return rows, load_parse(f"{d}/parse_out_{name}.jsonl")


def _pct(a, n):
    return f"{100 * a / n:.1f}%" if n else "-"


def print_paired(name, p):
    print(f"{name}: slots={p['slots']} readings={p['readings']} | TOTAL WRONG {p['total_base']} -> {p['total_run']} "
          f"(diff {p['diff']:+d}; 90% CI files {p['ci90_file']} sentences {p['ci90_sentence']}) | seam "
          f"{p['seam_base']} -> {p['seam_run']} | always {p['always_base']} -> {p['always_run']} | seam on slots both "
          f"right >=1x ({p.get('both_right_slots', 0)}): {p.get('seam_both_base', 0)} -> {p.get('seam_both_run', 0)} | "
          f"slots worse/better {p['worse']}/{p['better']} sign p={p['sign_p']:.3g}")


def print_transitions(p, n=5):
    t = {tuple(int(x) for x in k.split("->")): v for k, v in p["transitions"].items()}
    print("   right-count base (rows) x run (cols): " + " ".join(f"{c:>5}" for c in range(n + 1)))
    for a in range(n + 1):
        print(f"   {a:>38} " + " ".join(f"{t.get((a, c), 0):>5}" for c in range(n + 1)))


def main(argv):
    offsets, js = 5, None
    if "--offsets" in argv:
        i = argv.index("--offsets"); offsets = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    if "--json" in argv:
        i = argv.index("--json"); js = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    lenient, guard = "--lenient" in argv, "--guard" in argv
    argv = [a for a in argv if not a.startswith("--")]
    W, arms = argv[0], argv[1:]
    tabs = {a: slot_table(*load_arm(W, a), offsets, lenient) for a in arms}
    keys = guarded_keys(list(tabs.values())) if guard else None
    res = {}
    for a in arms:
        t = tabs[a] if keys is None else {k: v for k, v in tabs[a].items() if k in keys}
        res[a] = seam_counts(None, None, table=t)
    if guard:
        print(f"boundary guard: {len(keys)} of {len(set.intersection(*(set(t) for t in tabs.values())))} slots kept")
    print("per language: total wrong | seam | always   (arms " + " -> ".join(arms) + ")")
    langs = sorted({L for a in arms for L in res[a]}, key=lambda L: (ORDER.index(L) if L in ORDER else len(ORDER), L))
    tot = {a: collections.Counter() for a in arms}
    for L in langs:  # every language present (ORDER first, then the rest), all of them in the totals
        for a in arms:
            tot[a].update(res[a][L])
        print(f"{L:3} total " + " -> ".join(f"{res[a][L]['total']:3}" for a in arms) + " | seam " + " -> ".join(f"{res[a][L]['seam']:3}" for a in arms)
              + " | always " + " -> ".join(f"{res[a][L]['always']:3}" for a in arms)
              + " | total by slot " + "  ".join(f"{s}:" + "->".join(str(res[a][L]['total_' + s]) for a in arms) for s in SL if any(res[a][L]['total_' + s] for a in arms)))
    for a in arms:
        t = tot[a]
        print(a, f"total={t['total']} seam={t['seam']} always={t['always']} of {t['n']} slot readings;", " ".join(f"{s}:{t['seam_' + s]}+{t['always_' + s]}/{t['n_' + s]}" for s in SL if t['n_' + s]))
    out = {"totals": {a: dict(tot[a]) for a in arms}, "paired": {}}
    for a in arms[1:]:
        p = paired(tabs[arms[0]], tabs[a], keys)
        print_paired(f"\nPAIRED {arms[0]} -> {a}", p)
        print_transitions(p, offsets)
        for L in langs:
            q = paired(tabs[arms[0]], tabs[a], keys, {L})
            if q["slots"]:
                print(f"   {L:3} total {q['total_base']:3} -> {q['total_run']:3} diff {q['diff']:+4d} CI90(sent) {q['ci90_sentence']} "
                      f"worse/better {q['worse']}/{q['better']} p={q['sign_p']:.2g}")
        ni = near_interior([tabs[arms[0]], tabs[a]], keys)
        if ni:
            for i, nm in enumerate((arms[0], a)):
                v = ni[i]
                g = lambda k: v.get(k, [0, 0])  # noqa: E731
                print(f"   near-merge vs interior {nm}: near {g('near')[0]}/{g('near')[1]} ({_pct(*g('near'))}) interior "
                      f"{g('interior')[0]}/{g('interior')[1]} ({_pct(*g('interior'))}); within slots "
                      f"{_pct(*g('within_slot_near'))} vs {_pct(*g('within_slot_interior'))}")
            p["near_interior"] = {nm: ni[i] for i, nm in enumerate((arms[0], a))}
        out["paired"][a] = p
    if js:
        json.dump(out, open(js, "w"), indent=1)
    return tot


if __name__ == "__main__":
    main(sys.argv[1:])
