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
# retrain-3 (R3.12.5 / R3.15): language-aware leak words (homographs per language removed), full written-form set,
# the "twenty thousand <1-99>" blend (not followed by a scale), chimeras (digits glued to letters / number word + ".5")
LEAK_EXTRA = ["zero", "six", "ten", "point", "million", "first", "second", "third"]
LEAK_HOMOGRAPH = {"fr": {"six", "million", "point", "second"}, "nl": {"ten", "zero"}, "it": {"zero"}, "pt": {"zero"},
                  "es": {"million"}, "de": {"million"}, "sv": {"million"}, "da": {"million"}, "ro": {"zero"},
                  "mt": {"ten"}, "pl": {"ten"}, "cs": {"ten"}, "sk": {"ten"}}
WF = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°\u2047²³¹½¼¾]")
_U = "one|two|three|four|five|six|seven|eight|nine"
_T = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_D = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
BLEND = re.compile(rf"\btwenty thousand (?:(?:{_D})(?:[ -](?:{_U}))?|{_T}|{_U})\b(?![ -](?:hundred|thousand|million|"
                   rf"billion|trillion|point|and|{_U}|{_T}|{_D})\b)", re.I)
# R3.23 (external review, verified): a year read as "two"/"twenty" + a non-word that starts a number ("two televen",
# "Two tinete", "two tenty") — the model began "thousand" and ran out of frames. v10pt 1, r3 arms 14-23 on the EN beds.
_NUMW = set(f"zero {_U} {_T} {_D} hundred thousand million billion".replace("|", " ").split())
YEAR_GARBLE = re.compile(r"\b(?:two|twenty)\s+(t[a-z]*(?:een|even|ete|enty|elve|en|ty))\b", re.I)


def year_garbles(text):
    return [m.group(0) for m in YEAR_GARBLE.finditer(text) if m.group(1).lower() not in _NUMW]


CHIMERA = re.compile(rf"[a-z]{{2,}}\d|\d[a-z]{{3,}}\b|\b(?i:zero|{_U}|{_T}|{_D})[.,]\d")  # "Q1" (a code) is WF, not here


def leak_re(lang):
    words = EN_NUM.pattern[len(r"\b("):-len(r")\b")].split("|") + LEAK_EXTRA  # review: [5:-5] cut "two"/"percent"
    words = [w for w in words if w not in LEAK_HOMOGRAPH.get(lang, set())]
    return re.compile(r"\b(" + "|".join(sorted(set(words))) + r")\b", re.I)


def load(root, run, name):
    try:
        return [json.loads(x) for x in open(f"{root}/{run}/{name}.jsonl")]
    except FileNotFoundError:
        return []


def norm(t):
    t = re.sub(r"[^\w' ]", " ", t.lower())
    t = FILL.sub(" ", t)  # filler-free (owner 2026-10-13: fillers are never gated)
    return re.sub(r"\s+", " ", t).strip()


def _ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def fleurs_bootstrap(root, run, base, name="fleurs_wer", reps=2000, seed=0, bar=0.3):
    """R3.17 M2: per-language paired bootstrap of the WER difference run - base over utterances (same ids). A language
    FAILS only when the mean delta > `bar` points AND the 95% CI excludes 0."""
    import random

    a = {r["id"]: r for r in load(root, run, name)}
    b = {r["id"]: r for r in load(root, base, name)}
    by = defaultdict(list)
    for i, rb in b.items():
        ra = a.get(i)
        if ra is None or not norm(rb["ref"]):
            continue
        ref = norm(rb["ref"]).split()
        by[lang_of(rb)].append((_ed(ref, norm(ra["hyp"]).split()), _ed(ref, norm(rb["hyp"]).split()), len(ref)))
    out = {}
    rnd = random.Random(seed)
    for lang, xs in sorted(by.items()):
        n = len(xs)
        d = lambda s_: 100 * (sum(x[0] for x in s_) - sum(x[1] for x in s_)) / max(1, sum(x[2] for x in s_))  # noqa
        ds = sorted(d([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(reps))
        lo, hi = ds[int(0.025 * reps)], ds[int(0.975 * reps) - 1]
        mean = d(xs)
        out[lang] = {"n": n, "delta": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                     "fail": mean > bar and lo > 0}
    return out


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
    # "49, 99" == "49,99" (one optional space after a separator inside a number: review #5)
    return sorted(re.sub(r"[.,\u00a0 ]", "", d) for d in re.findall(r"\d+(?:(?:[.,]\s?|[\u00a0 ])\d+)*", t))


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
        by = defaultdict(lambda: [0, 0, 0])
        for r in load(root, run, name):
            k = lang_of(r)
            by[k][0] += 1
            by[k][1] += bool(leak_re(k).search(r["hyp"]))
            by[k][2] += digit_runs(r["hyp"]) == digit_runs(r["ref"].split("\t")[0])
        s[f"leak_{name}"] = {k: v[1] for k, v in sorted(by.items())}
        s[f"leak_{name}_total"] = sum(v[1] for v in by.values())
        s[f"digitok_{name}"] = {k: f"{v[2]}/{v[0]}" for k, v in sorted(by.items())}
        s[f"digitok_{name}_total"] = sum(v[2] for v in by.values())
    blend = chim = 0
    # e21_dev is a STOP check only (never selection, R3.12.5): reported on its own keys
    e21 = load(root, run, "e21_dev")
    s["e21_dev_blend"] = sum(len(BLEND.findall(r["hyp"])) for r in e21)
    s["e21_dev_blend_hits"] = [m.group(0) + " @ " + r["id"] for r in e21 for m in BLEND.finditer(r["hyp"])][:40]
    s["e21_dev_wf_windows"] = sum(bool(WF.search(r["hyp"])) for r in e21)
    s["e21_dev_chimeras"] = sum(len(CHIMERA.findall(r["hyp"])) for r in e21)
    # the blend precursor (R3.18): one spoken year read "two thousand X" and "twenty X" by overlapping windows
    import yearflip

    wins = [{"call": r["id"].split("_")[0], "start": int(r["id"].split("_")[1]) / 16000, "duration": 14.96,
             "id": r["id"], "hyp": r["hyp"]} for r in e21]
    yf = yearflip.flips(wins, "hyp") if wins else {"flips": 0, "year_occurrences": 0}
    s["e21_dev_year_flips"], s["e21_dev_year_occ"] = yf["flips"], yf["year_occurrences"]
    s["e21_dev_onset_2k_flips"] = yf.get("onset_2k_flips", 0)  # R3.23 blend precursor: v10pt 1, rt2 7
    g = [x for name in ("d1_dev", "e22_numbers", "d9_dev", "fast", "biz_dev", "adv", "lspc_dev", "e22_other")
         for r in load(root, run, name) for x in year_garbles(r["hyp"])]
    s["year_garble"], s["year_garble_hits"] = len(g), g[:20]
    biz = load(root, run, "biz_dev")
    if biz:
        s["wer_biz_dev"] = wer(biz)
        s["digits_biz_dev"] = sum(bool(WF.search(r["hyp"])) for r in biz)
    for name in ("e22_other", "lspc_dev", "d1_dev", "e22_numbers", "fast", "adv"):
        rows = load(root, run, name)
        if name in ("e22_other", "lspc_dev", "d1_dev", "e22_numbers"):
            s[f"digits_{name}"] = sum(bool(WF.search(r["hyp"])) for r in rows)
            s[f"wf_tokens_{name}"] = sum(len(WF.findall(r["hyp"])) for r in rows)
        # adv: English rows of the scored split only (Opus review: sealed adv rows were feeding selection keys 1 and 3)
        en_rows = [r for r in rows if name != "adv" or (r["ref"].split("\t")[2] == "en"
                                                        and r["ref"].split("\t")[4] == want_split)]
        blend += sum(len(BLEND.findall(r["hyp"])) for r in en_rows)
        chim += sum(len(CHIMERA.findall(r["hyp"])) for r in en_rows)
    s["blend_twenty_thousand"] = blend
    s["chimeras"] = chim
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
    stock_rows = load(root, stock_run, "adv")
    stock_d = {r["id"]: digit_runs(r["hyp"]) for r in stock_rows}
    # pre-registered from the STOCK decode only: stock wrote English number words -> the TTS clip sounds English
    stock_en = {r["id"] for r in stock_rows if r["ref"].split("\t")[2] != "en" and EN_NUM.search(r["hyp"])}
    adv3 = defaultdict(lambda: [0, 0])
    for r in load(root, run, "adv"):
        text, exp, lang, cat, split = r["ref"].split("\t")
        if split != want_split or r["id"] in exclude:
            continue
        if lang == "en":
            ok = adv_ok(text, exp, lang, r["hyp"])
        else:
            ok = digit_runs(r["hyp"]) == stock_d.get(r["id"]) and not leak_re(lang).search(r["hyp"])
        if lang != "en":
            adv_leak += bool(leak_re(lang).search(r["hyp"]))
            cs = cat.startswith("codeswitch")
            for view, inc in (("all", True), ("nocs", not cs), ("clean", not cs and r["id"] not in stock_en)):
                if inc:
                    adv3[view][0] += 1
                    adv3[view][1] += ok
        key = ("en" if lang == "en" else "xx") + ":" + cat
        adv[key][0] += 1
        adv[key][1] += ok
    s["adv"] = {k: f"{v[1]}/{v[0]}" for k, v in sorted(adv.items())}
    s["adv_en_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("en:"))
    s["adv_en_n"] = sum(v[0] for k, v in adv.items() if k.startswith("en:"))
    s["adv_xx_ok"] = sum(v[1] for k, v in adv.items() if k.startswith("xx:"))
    s["adv_xx_n"] = sum(v[0] for k, v in adv.items() if k.startswith("xx:"))
    s["adv_xx_leak"] = adv_leak
    s["adv_xx_views"] = {k: f"{v[1]}/{v[0]}" for k, v in adv3.items()}
    s["adv_xx_clean_ok"] = adv3["clean"][1]
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


# RETRAIN-3 frozen selection order (plan R3.12.5; written before any retrain-3 number was read). Lexicographic; a
# difference within the stated noise band passes to the next key.
SELECT_ORDER = [("blend", -1, 0), ("en_wf", -1, 2), ("chimeras", -1, 0), ("adv_en_ok", +1, 3), ("fast_d1_ok", +1, 5),
                ("nonen_leak", -1, 1), ("fleurs_worst", -1, 0), ("wer_dev", -1, 0.1)]


def sel_metrics(s):
    fl = [v for v in (s.get("wer_fleurs_delta") or {}).values() if v is not None]
    boot = next((v for k, v in s.items() if k.startswith("fleurs_bootstrap_vs_")), None)
    return {"blend": s["blend_twenty_thousand"], "en_wf": s["digits_e22_numbers"] + s["digits_d1_dev"],
            "chimeras": s["chimeras"], "adv_en_ok": s["adv_en_ok"], "fast_d1_ok": s["fast_d1_ok"],
            "nonen_leak": s["leak_vox_dev_total"] + s["leak_fleurs_num_total"],
            # R3.17 M2 (final fresh-eyes review): languages failing the paired bootstrap, not the raw worst delta
            "fleurs_worst": sum(x["fail"] for x in boot.values()) if boot else (max(fl) if fl else 0.0),
            "wer_dev": (s["wer_lspc_dev"] or 0) + (s["wer_d1_dev"] or 0)}


def select(res, runs, baseline=None):
    """Pick one run: walk SELECT_ORDER; at each key keep the runs within the noise band of the best; stop when one is
    left (ties after the last key: the first listed). `baseline` (e.g. v10pt) fills wer_fleurs_delta."""
    if baseline is not None:
        for r in runs:
            res[r]["wer_fleurs_delta"] = {k: (v - res[baseline]["wer_fleurs"].get(k, v))
                                          for k, v in res[r]["wer_fleurs"].items() if v is not None}
    m = {r: sel_metrics(res[r]) for r in runs}
    alive = list(runs)
    for k, sign, band in SELECT_ORDER:
        best = max(sign * m[r][k] for r in alive)
        alive = [r for r in alive if best - sign * m[r][k] <= band]
        if len(alive) == 1:
            break
    return {"run": alive[0], "tied": alive, "metrics": m}


if __name__ == "__main__":
    root, runs = sys.argv[1], sys.argv[2].split(",")
    excl = frozenset()
    if "--exclude-from" in sys.argv:
        st, v10 = sys.argv[sys.argv.index("--exclude-from") + 1].split(",")
        excl = frozenset(exclusions(root, st, v10))
        print(f"D.2 exclusions: {len(excl)}")
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    # cached-encoder runs (eval_multi, "<run>_c") are compared only with cached baselines: stock = ultra_rt_c
    stock = sys.argv[sys.argv.index("--stock") + 1] if "--stock" in sys.argv else (
        "ultra_rt_c" if all(r.endswith("_c") for r in runs) else "ultra_rt")
    res = {run: score(root, run, split, excl, stock_run=stock) for run in runs}
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
        for run in runs:
            if run != base:
                res[run]["fleurs_bootstrap_vs_" + base] = fleurs_bootstrap(root, run, base)
    print(json.dumps(res, indent=1))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
