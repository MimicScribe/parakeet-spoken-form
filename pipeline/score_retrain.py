"""Score retrain arms from hyps pulled off the volume (modal volume get parakeet-spoken-form /hyps/<run> <dir>).

    python score_retrain.py <hyps_root> <run>[,<run>...] [--json out.json]

Per run: WER on e22_other / lspc_dev / d1_dev / fleurs_wer (per language); English-number-word leak on vox_dev and
fleurs_num (per language); digits emitted in English sets; noise bed numbers; adversarial value accuracy (English: the
spoken value; other languages: the digits) by category and language; fast-year accuracy; fillers per minute.
Leak and digit counts are absolute; the gates compare them against the stock and v10pt rows of the same table.
"""

import json
import re
import sys
from collections import defaultdict

import jiwer

import labeller as L

EN_NUM = re.compile(r"\b(two|three|four|five|seven|eight|nine|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
                    r"thousand|billion|percent)\b", re.I)
FILL = re.compile(r"\b(uh|um|uhm|er|erm|mm|hmm|mhm)\b", re.I)


def load(root, run, name):
    try:
        return [json.loads(x) for x in open(f"{root}/{run}/{name}.jsonl")]
    except FileNotFoundError:
        return []


def norm(t):
    t = re.sub(r"[^\w' ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def wer(rows):
    rows = [r for r in rows if norm(r["ref"])]
    if not rows:
        return None
    return round(100 * jiwer.wer([norm(r["ref"]) for r in rows], [norm(r["hyp"]) or "x" for r in rows]), 2)


def values_spoken(text):
    out = set()
    for s in L.spans(text):
        if s.kind == "spelled":
            v = L.spelled_value(s.gold)
            if v:
                out.add(str(v[0]))
    for d in re.findall(r"\d[\d,.]*", text):
        out.add(d.replace(",", "").rstrip("."))
    return out


def lang_of(r):
    i = r.get("id", "")
    return i.split("_")[0] if "_" in i else "?"


def score(root, run):
    s = {}
    for name in ("e22_other", "lspc_dev", "d1_dev", "e22_numbers"):
        s[f"wer_{name}"] = wer(load(root, run, name))
    fl = load(root, run, "fleurs_wer")
    by = defaultdict(list)
    for r in fl:
        by[lang_of(r)].append(r)
    s["wer_fleurs"] = {k: wer(v) for k, v in sorted(by.items())}
    for name in ("vox_dev", "fleurs_num"):
        by = defaultdict(lambda: [0, 0])
        for r in load(root, run, name):
            k = lang_of(r)
            by[k][0] += 1
            by[k][1] += bool(EN_NUM.search(r["hyp"]))
        s[f"leak_{name}"] = {k: v[1] for k, v in sorted(by.items())}
        s[f"leak_{name}_total"] = sum(v[1] for v in by.values())
    for name in ("e22_other", "lspc_dev", "d1_dev", "e22_numbers"):
        s[f"digits_{name}"] = sum(bool(re.search(r"\d", r["hyp"])) for r in load(root, run, name))
    nz = load(root, run, "noise")
    s["noise_numbers"] = sum(bool(EN_NUM.search(r["hyp"]) or re.search(r"\d", r["hyp"])) for r in nz)
    s["noise_words"] = sum(len(r["hyp"].split()) for r in nz)
    # adversarial: refs carry "expected" values after a tab: "text\tv1|v2|lang|category"
    adv = defaultdict(lambda: [0, 0])
    adv_leak = 0
    for r in load(root, run, "adv"):
        text, exp, lang, cat = r["ref"].split("\t")
        want = [x for x in exp.split("|") if x]
        got = values_spoken(r["hyp"]) if lang == "en" else {d.replace(",", "").replace(" ", "") for d in
                                                              re.findall(r"\d[\d,. ]*\d|\d", r["hyp"])}
        if lang != "en":
            got |= {g.replace(".", "") for g in got}
            adv_leak += bool(EN_NUM.search(r["hyp"]))
        ok = all(w.replace(",", "").replace(".", "") in {g.replace(".", "") for g in got} or w in got for w in want)
        key = ("en" if lang == "en" else "xx") + ":" + cat
        adv[key][0] += 1
        adv[key][1] += ok
    s["adv"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(adv.items())}
    s["adv_en_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("en:"))
    s["adv_xx_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("xx:"))
    s["adv_xx_leak"] = adv_leak
    # fast: refs "caption\tgold" (gold = the spoken reading, or the value for E22)
    fast = defaultdict(lambda: [0, 0])
    for r in load(root, run, "fast"):
        cap, gold, tag = r["ref"].split("\t")
        ok = norm(gold) in norm(r["hyp"]) if not gold.isdigit() else gold in values_spoken(r["hyp"])
        fast[tag][0] += 1
        fast[tag][1] += ok
    s["fast"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(fast.items())}
    eo = load(root, run, "e22_other")
    mins = sum(len(r["ref"].split()) for r in eo) / 150 or 1  # ~150 wpm proxy for minutes
    s["fillers_per_min_e22_other"] = round(sum(len(FILL.findall(r["hyp"])) for r in eo) / mins, 3)
    return s


if __name__ == "__main__":
    root, runs = sys.argv[1], sys.argv[2].split(",")
    res = {run: score(root, run) for run in runs}
    print(json.dumps(res, indent=1))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
