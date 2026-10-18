"""Reading-variety census of a built mix (2026-10-18): per language x cell x reading convention (reading_conv), how
many real and synthetic spoken numbers use each way of saying the number, and how far the synthetic shares are from
the fill plan's targets (variety_plan_rv1.json).

    python3 variety_census.py <mix dir> [--langs de,nl] [--json out.json]
    python3 variety_census.py <mix dir> --add <ml/ttsRV1/de/rows.jsonl> ... --sim-dir <dir>   # the mix after a fill

<mix dir> holds the mix parts (p3_<lang>.jsonl, tts_<lang>.jsonl, as `modal volume get` writes them). --add simulates
the mix with a replacing set: its rows join tts_<lang> and the rows they supersede leave (what mix_build's
order_synthetic does when every replacing row is chosen); --sim-dir writes that simulated mix (tts_<lang>.jsonl, the
other parts linked) for eval/quality/data_gate.py.
"""
import argparse
import collections
import json
import os

import reading_conv as RC

HERE = os.path.dirname(os.path.abspath(__file__))
CENT_CONVS = {"unit+cents", "unit+join+cents", "unit+join+cents+centword", "unit+cents+centword", "decimal_word+unit",
              "decimal_word", "bare_joiner", "bare_no_unit", "cents_only"}


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def mix_rows(d: str, lang: str, add: list) -> tuple:
    """(real rows, synthetic rows) of a language; `add` = replacing rows (their superseded ids leave)."""
    import census_common as CC

    p3 = _rows(os.path.join(d, f"p3_{lang}.jsonl"))
    tts = _rows(os.path.join(d, f"tts_{lang}.jsonl"))
    if add:
        gone = {i for r in add for i in r.get("supersedes") or []}
        tts = [r for r in tts if r["id"] not in gone] + [dict(r, src="tts") for r in add]  # planned rows: synthetic
    rows = p3 + tts
    return [r for r in rows if not CC.synthetic(r)], [r for r in rows if CC.synthetic(r)]


def count(rows: list, lang: str) -> dict:
    """cell -> Counter(convention)."""
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        for u in RC.units(r, lang):
            c = RC.cell(lang, u)
            if c is None:
                continue
            conv, _ = RC.classify(lang, u)
            if conv is None or (c == "cur" and conv not in CENT_CONVS):
                continue
            out[c][conv] += 1
    return out


def distance(have: collections.Counter, target: dict) -> float:
    """Half the L1 distance between the synthetic convention shares and the plan's target shares (0 = on target)."""
    n, t = sum(have.values()), sum(target.values())
    if not n or not t:
        return 0.0
    keys = set(have) | set(target)
    return round(0.5 * sum(abs(have.get(k, 0) / n - target.get(k, 0) / t) for k in keys), 3)


def census(d: str, langs: list, add: dict, plan: list) -> dict:
    out = {}
    tgt = {(c["lang"], c["cls"]): c["target"] for c in plan}
    for lang in langs:
        real, syn = mix_rows(d, lang, add.get(lang, []))
        cr, cs = count(real, lang), count(syn, lang)
        out[lang] = {c: {"real": dict(cr.get(c, {})), "syn": dict(cs.get(c, {})), "syn_n": sum(cs.get(c, {}).values()),
                         "real_n": sum(cr.get(c, {}).values()),
                         "target": tgt.get((lang, c)),
                         "dist": distance(cs.get(c, collections.Counter()), tgt[(lang, c)]) if (lang, c) in tgt else None}
                     for c in sorted(set(cr) | set(cs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix")
    ap.add_argument("--langs", default="de,es,fr,it,nl,pt")
    ap.add_argument("--add", nargs="*", default=[], help="rows.jsonl of a replacing set (any languages)")
    ap.add_argument("--plan", default=os.path.join(HERE, "variety_plan_rv1.json"))
    ap.add_argument("--json")
    ap.add_argument("--sim-dir")
    a = ap.parse_args()
    add = collections.defaultdict(list)
    for p in a.add:
        for r in _rows(p):
            add[r["lang"]].append(r)
    langs = a.langs.split(",")
    plan = json.load(open(a.plan))["cells"] if a.plan and os.path.exists(a.plan) else []
    res = census(a.mix, langs, add, plan)
    for lang in langs:
        for c, v in res[lang].items():
            if v["target"] is None and not v["syn_n"]:
                continue
            print(f"{lang} {c:17s} real {v['real_n']:5d} syn {v['syn_n']:5d} dist {v['dist']}  syn {v['syn']}")
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
    if a.sim_dir:
        os.makedirs(a.sim_dir, exist_ok=True)
        for f in os.listdir(a.mix):
            src, dst = os.path.join(os.path.abspath(a.mix), f), os.path.join(a.sim_dir, f)
            lang = f[4:6] if f.startswith("tts_") else None
            if lang in add:
                gone = {i for r in add[lang] for i in r.get("supersedes") or []}
                with open(dst, "w", encoding="utf-8") as g:
                    for r in _rows(src):
                        if r["id"] not in gone:
                            g.write(json.dumps(r, ensure_ascii=False) + "\n")
                    for r in add[lang]:
                        g.write(json.dumps(r, ensure_ascii=False) + "\n")
            elif not os.path.exists(dst):
                os.symlink(src, dst)


if __name__ == "__main__":
    main()
