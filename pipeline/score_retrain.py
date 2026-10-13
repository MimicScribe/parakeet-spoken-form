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
    err = tot = 0
    for r in rows:
        a, b = norm(r["ref"]).split(), norm(r["hyp"]).split()
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, y in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
            prev = cur
        err += prev[-1]
        tot += len(a)
    return round(100 * err / max(tot, 1), 2)


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


def has_words(hyp, words):
    return re.search(r"(?<![\w'])" + re.escape(norm(words)) + r"(?![\w'])", norm(hyp)) is not None


def _num_norm(v):
    v = v.strip().replace("\u00a0", "").replace(" ", "")
    m = re.match(r"^(.*?)[.,](\d{1,2})$", v)
    if m and len(v) - len(m.group(2)) - 1 > 0 and not re.match(r"^\d{1,3}([.,]\d{3})+$", v):
        return re.sub(r"[.,]", "", m.group(1)) + "." + m.group(2)
    return re.sub(r"[.,]", "", v)


def adv_ok(text, exp, lang, hyp):
    if lang == "en":
        sp = [x for x in L.spans(text) if x.kind != "spelled"]
        if sp:
            return all(any(has_words(hyp, c) for c in x.cands) for x in sp)
        # spoken-form script rows: every run of number words must be heard exactly as said
        from common import number_span
        words = text.split()
        flags = number_span(words)
        runs, cur = [], []
        for w, f in zip(words, flags):
            if f:
                cur.append(w)
            elif cur:
                runs.append(" ".join(cur))
                cur = []
        if cur:
            runs.append(" ".join(cur))
        if not runs:  # homophone rows ("We won one deal today."): the sentence itself must come out
            return norm(hyp) == norm(text)
        return all(has_words(hyp, r) for r in runs)
    want = [w for w in exp.split("|") if w]
    got = {_num_norm(t) for t in re.findall(r"\d+(?:[.,\u00a0 ]\d+)*", hyp)}
    return all(_num_norm(w) in got for w in want)


def lang_of(r):
    i = r.get("id", "")
    return i.split("_")[0] if "_" in i else "?"


def digit_runs(t):
    return sorted(re.sub(r"[.,\u00a0 ]", "", d) for d in re.findall(r"\d+(?:[.,\u00a0 ]\d+)*", t))


def score(root, run, want_split="dev", exclude=frozenset(), stock_run="ultra_rt"):
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
    # adversarial (dev or sealed split; D.2 exclusions): English = every caption number heard as one of its spoken
    # readings (labeller candidates, word-boundary match); other languages = every expected value present as digits.
    adv = defaultdict(lambda: [0, 0])
    adv_leak = 0
    # non-English: the contract is STOCK's digits (labels were stock's), so a row is right when its digit runs equal
    # stock's on the same clip and no English number word appears (the script's canonical values like "1200000000"
    # for "1,2 mil millones" are not a writing anyone produces)
    stock_d = {r["id"]: digit_runs(r["hyp"]) for r in load(root, stock_run, "adv")}
    for r in load(root, run, "adv"):
        text, exp, lang, cat, split = r["ref"].split("\t")
        if split != want_split or r["id"] in exclude:
            continue
        if lang == "en":
            ok = adv_ok(text, exp, lang, r["hyp"])
        else:
            ok = digit_runs(r["hyp"]) == stock_d.get(r["id"]) and not EN_NUM.search(r["hyp"])
        if lang != "en":
            adv_leak += bool(EN_NUM.search(r["hyp"]))
        key = ("en" if lang == "en" else "xx") + ":" + cat
        adv[key][0] += 1
        adv[key][1] += ok
    s["adv"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(adv.items())}
    s["adv_en_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("en:"))
    s["adv_en_n"] = sum(v[0] for k, v in adv.items() if k.startswith("en:"))
    s["adv_xx_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("xx:"))
    s["adv_xx_n"] = sum(v[0] for k, v in adv.items() if k.startswith("xx:"))
    s["adv_xx_leak"] = adv_leak
    # fast: refs "caption\tgold\ttag"; D1 (macOS say, literal readings) = reading match with word boundaries;
    # E22 = value match. Arm selection uses D1 only (D.9 / Opus H4); E22 is reported.
    fast = defaultdict(lambda: [0, 0])
    for r in load(root, run, "fast"):
        cap, gold, tag = r["ref"].split("\t")
        ok = has_words(r["hyp"], gold) if not gold.isdigit() else gold in values_spoken(r["hyp"])
        fast[tag][0] += 1
        fast[tag][1] += ok
    s["fast"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(fast.items())}
    s["fast_d1_ok"] = sum(v[1] for k, v in fast.items() if k.startswith("d1_"))
    s["fast_d1_n"] = sum(v[0] for k, v in fast.items() if k.startswith("d1_"))
    eo = load(root, run, "e22_other")
    mins = sum(len(r["ref"].split()) for r in eo) / 150 or 1  # ~150 wpm proxy for minutes
    s["fillers_per_min_e22_other"] = round(sum(len(FILL.findall(r["hyp"])) for r in eo) / mins, 3)
    return s


def value_ok(exp, hyp):
    """Every expected VALUE present, written as digits or spoken (reading-agnostic)."""
    want = [w for w in exp.split("|") if w]
    got = {_num_norm(t) for t in re.findall(r"\d+(?:[.,\u00a0 ]\d+)*", hyp)} | {_num_norm(v) for v in values_spoken(hyp)}
    return bool(want) and all(_num_norm(w) in got for w in want)


def exclusions(root, stock_run, v10_run):
    """D.2: adversarial rows where BOTH stock and v10pt miss the expected VALUE are excluded (the TTS or the script is
    suspect). Value, not reading: stock writes digits, so judging it by spoken reading would exclude every English
    row v10pt gets wrong (circular)."""
    def stock_wrong():
        # English rows only (non-English is scored against stock itself). stock writes digits ("4:40", "555-0142", "-5"): every expected value's digits must occur in the digit
        # stream of the hypothesis
        out = set()
        for r in load(root, stock_run, "adv"):
            text, exp, lang, cat, split = r["ref"].split("\t")
            if lang != "en":
                continue
            stream = re.sub(r"\D", "", r["hyp"])
            want = [re.sub(r"\D", "", w) for w in exp.split("|") if re.sub(r"\D", "", w)]
            if want and not all(w in stream for w in want):
                out.add(r["id"])
        return out

    def v10_wrong():
        out = set()
        for r in load(root, v10_run, "adv"):
            text, exp, lang, cat, split = r["ref"].split("\t")
            if exp and not (adv_ok(text, exp, lang, r["hyp"]) or value_ok(exp, r["hyp"])):
                out.add(r["id"])
        return out
    return stock_wrong() & v10_wrong()


def e21x_values(root, run, refs_dir):
    """SEALED Earnings-21 (33 calls): numbers by VALUE per call. Reference values = digit tokens of the transcript;
    hypothesis values = digits + spelled numbers. Returns matched / ref / hyp counts (recall, invented)."""
    import glob
    import os
    from collections import Counter

    by = defaultdict(list)
    for r in load(root, run, "sealed/e21x"):
        by[r["id"].rsplit("_", 1)[0]].append((r["id"], r["hyp"]))
    tot = Counter()
    for f in glob.glob(f"{refs_dir}/*.txt"):
        fid = os.path.basename(f)[:-4]
        if fid not in by:
            continue
        ref = Counter(re.sub(r"[,]", "", d).rstrip(".") for d in re.findall(r"\d[\d,.]*", open(f).read()))
        hyp = Counter()
        for _, h in sorted(by[fid]):
            for v in values_spoken(h):
                hyp[v] += 1
        m = sum((ref & hyp).values())
        tot["matched"] += m
        tot["ref"] += sum(ref.values())
        tot["hyp"] += sum(hyp.values())
    return dict(tot)


SELECT_ORDER = [("leak", -1), ("fast_d1_ok", +1), ("adv_en_ok", +1), ("wer_dev", -1)]


def sel_metrics(s):
    return {"leak": s["leak_vox_dev_total"] + s["leak_fleurs_num_total"] + s["adv_xx_leak"],
            "fast_d1_ok": s["fast_d1_ok"], "adv_en_ok": s["adv_en_ok"],
            "wer_dev": (s["wer_lspc_dev"] or 0) + (s["wer_d1_dev"] or 0)}


def select(res, main_runs, stock_runs):
    """Pre-registered (V2 'Selection'): MAIN ships unless STOCK-INIT beats it, on the first metric in SELECT_ORDER
    where the arm means differ by more than the larger within-arm seed spread. Within the chosen arm, seeds are
    ranked lexicographically by SELECT_ORDER."""
    m = {r: sel_metrics(res[r]) for r in main_runs + stock_runs}
    arm = "main"
    for k, sign in SELECT_ORDER:
        a = [m[r][k] for r in main_runs]
        b = [m[r][k] for r in stock_runs]
        spread = max(max(a) - min(a), max(b) - min(b))
        diff = (sum(b) / len(b) - sum(a) / len(a)) * sign
        if abs(diff) > spread:
            arm = "stockinit" if diff > 0 else "main"
            break
    runs = main_runs if arm == "main" else stock_runs
    best = sorted(runs, key=lambda r: tuple(-sign * m[r][k] for k, sign in SELECT_ORDER))[0]
    return {"arm": arm, "run": best, "metrics": m}


if __name__ == "__main__":
    root, runs = sys.argv[1], sys.argv[2].split(",")
    excl = frozenset()
    if "--exclude-from" in sys.argv:
        st, v10 = sys.argv[sys.argv.index("--exclude-from") + 1].split(",")
        excl = frozenset(exclusions(root, st, v10))
        print(f"D.2 exclusions: {len(excl)}")
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    res = {run: score(root, run, split, excl) for run in runs}
    print(json.dumps(res, indent=1))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
