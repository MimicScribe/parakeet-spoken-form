"""Census of a synthetic number-sentence pack: counts and value spread per number type (AGENTS.md rule #1).
Pre-synthesis:  python3 -I census_v2.py [--tag v2]
Post-synthesis: python3 -I census_v2.py --kept <final rows jsonl>   (adds yield by cell, word_times edge position, aug, speed)
Value-spread stats use ORDINARY numbers only (repeat==False); repeat-cell values are reported in the distinct-readings table.
Writes census/<tag>[_kept].json, prints a summary and FLAG lines where a bar fails (parsed must equal n for every parser).

Sentence mode also checks the label-v3 invariant on every number (digit-by-digit runs >= 5 digits -> digits only; ordinary kinds -> no digits except
AM/PM; word+number codes -> no digits) and crowding (distinct-2/3 of number-masked templates, share of the top-50 templates; flag if > 5%).

MIX MODE (text only, no audio): python3 -I census.py --mix SPEC.json
  SPEC.json = list of {"path": "rows.jsonl", "weight": 1.0, "synthetic": true|false}. Rows are JSONL with a label ("label_v3", "label" or "text", in that
  order) and "duration" (seconds); an optional "lang" (rows without one count as English; only lang == "en" is counted). Hours are duration * weight / 3600,
  so the shares are PER-HOUR shares of the whole weighted mix. Reports English hours, synthetic share of English hours, digit-string share (a run of >= 5
  digits, written as digits or spoken digit by digit), number-free share (no digit, no number word; "one" counts as a number word, so this is conservative),
  and the repeat-free share among digit-string hours (no two equal adjacent digits). Flags: synthetic digit-string share outside 4-5% of English hours,
  number-free share < 0.40, synthetic share > 0.30, repeat-free digit share outside 35-65%."""
import argparse, collections, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
import neardup

U = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())} | {"oh": 0}
T = {w: 10 + i for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
D = {w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SC = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
MONTHS = "january february march april may june july august september october november december".split()
ORD = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())} | {"thirtieth": 30}
NOW = 2026
ORDINARY_KINDS = ["a year", "a time of day", "an amount of money", "a count of things", "a percentage", "a measurement with a decimal", "a date",
                  "a span of years", "a span of years (season or fiscal year)", "a range of quantities", "a page range", "game score"]


def small(ws):
    if len(ws) == 1: return U.get(ws[0], T.get(ws[0], D.get(ws[0])))
    if len(ws) == 2 and ws[0] in D and ws[1] in U and U[ws[1]]: return D[ws[0]] + U[ws[1]]
    if len(ws) == 2 and ws[0] == "oh" and ws[1] in U: return U[ws[1]]
    return None


def cardinal(ws):  # standard cardinal; adjacent same-rank words ("twenty twenty") are not cardinals -> None
    total, cur, seen, last = 0, 0, False, None
    for w in ws:
        if w == "and": continue
        rank = "u" if w in U and w != "oh" else "t" if w in T else "d" if w in D else w
        if rank in ("u", "t") and last in ("u", "t"): return None
        if rank == "d" and last in ("u", "t", "d"): return None
        if rank == "u": cur += U[w]
        elif rank == "t": cur += T[w]
        elif rank == "d": cur += D[w]
        elif w == "hundred": cur = (cur or 1) * 100
        elif w in SC: total += (cur or 1) * SC[w]; cur = 0
        else: return None
        seen, last = True, rank
    return total + cur if seen else None


def year(ws):
    ws = [w for w in ws if w != "and"]
    if not ws: return None
    if ws[:2] == ["two", "thousand"]:
        if len(ws) == 2: return 2000
        t = cardinal(ws[2:])  # an unparseable tail is NOT 2000: return None so the unparsed counter fires
        return None if t is None else 2000 + t
    for k in (1, 2):
        a, b = small(ws[:k]), small(ws[k:]) if ws[k:] else None
        if a and 10 <= a <= 99 and (b is not None or ws[k:] == ["hundred"]): return a * 100 + (b or 0)
    return None


RANGE_KINDS = ("a range of quantities", "a page range", "range (between X and Y)", "rough count (X or Y)")


def range_ends(reading):  # (low, high) of "X to Y" / "X through Y" / "X or Y" / "between X and Y"; None when no split parses both sides
    ws = [w for w in re.sub(r"[,.]", " ", reading.lower()).split() if w not in ("between", "um", "uh", "er")]
    for i, w in enumerate(ws):
        if w in ("to", "through", "or", "and") and 0 < i < len(ws) - 1:
            a, b = cardinal(ws[:i]), cardinal(ws[i + 1:])
            if a is not None and b is not None: return a, b
    return None


def span_start(reading):
    low = reading.lower()
    parts = re.split(r"\b(?:to|through)\b", low)
    if len(parts) > 1: return year(parts[0].split())
    ws = low.split()  # season form "nineteen seventy nine eighty": longest year prefix whose remainder is a 1-2 word number < 100
    for cut in range(len(ws) - 1, 0, -1):
        if small(ws[cut:]) is not None and year(ws[:cut]): return year(ws[:cut])
    return None


def money(r):
    ws = r.lower().replace("bucks", "").replace("dollars", "").replace("dollar", "").split()
    if "cents" in ws or "cent" in ws:
        i = ws.index("and") if "and" in ws else 0
        d = cardinal(ws[:i]) if i else 0; c = cardinal([w for w in ws[i:] if w not in ("and", "cents", "cent")])
        return None if d is None or c is None else d + c / 100
    if ws[:2] == ["a", "buck"]: c = cardinal(ws[2:]); return None if c is None else 1 + c / 100
    if "point" in ws:
        i = ws.index("point"); sc = SC.get(ws[-1], 1)
        a = cardinal(ws[:i]); frac = "".join(str(U[w]) for w in ws[i + 1:] if w in U)
        return (a + float("0." + frac)) * sc if a is not None and frac else None
    v = cardinal(ws)
    if v is None and "dollars" in r.lower():  # "twelve dollars fifty"
        i = r.lower().split().index("dollars"); d, c = cardinal(r.lower().split()[:i]), cardinal(r.lower().split()[i + 1:])
        if d is not None and c is not None and c < 100: return d + c / 100
    if v is None:  # bare price "thirty two twenty eight", "four oh five"
        for k in (1, 2):
            d, c = small(ws[:k]), small(ws[k:])
            if d is not None and c is not None: return d + c / 100
    return v


def decimal(r):
    ws = r.lower().split()
    if "point" not in ws: return None, None
    i = ws.index("point"); a = cardinal(ws[:i]) if ws[:i] else 0; dec = [w for w in ws[i + 1:] if w in U]
    return (None, None) if a is None or not dec else (a + float("0." + "".join(str(U[w]) for w in dec)), len(dec))


SUFFIX = (("a m", "am"), ("p m", "pm"), ("in the morning", "morning"), ("in the evening", "evening"), ("in the afternoon", "afternoon"),
          ("at night", "night"), ("o'clock", "oclock"), ("noon", "noon"), ("midnight", "midnight"), ("hundred hours", "24h"), ("hundred", "24h"))


def clock(r):
    s = f" {r.lower()} "; suf = "none"
    for k, tag in SUFFIX:  # strip every suffix ("five o'clock in the morning"); keep the first tag found
        if f" {k} " in s: suf = tag if suf == "none" else suf; s = s.replace(f" {k} ", " ", 1)
    ws = s.split()
    if ws in (["noon"], ["twelve", "noon"]) or suf == "noon" and not ws: return 12, 0, "noon"
    if suf == "midnight" and not ws: return 0, 0, "midnight"
    if "past" in ws or (len(ws) >= 3 and ws[-2] == "to"):
        i = ws.index("past") if "past" in ws else len(ws) - 2; mm = {"quarter": 15, "half": 30}.get(ws[0]) or small(ws[:i]); h = small(ws[i + 1:])
        if mm is None or h is None: return None, None, suf
        return (h, mm, "past") if ws[i] == "past" else ((h - 1) % 24, 60 - mm, "to")
    if ws[:1] == ["oh"] and len(ws) >= 2: h = U.get(ws[1]); m = small(ws[2:]) if ws[2:] else 0; return (h, m, "24h") if h is not None and m is not None else (None, None, suf)
    if ws[:2] in (["quarter", "past"], ["half", "past"]): return small(ws[2:]), 15 if ws[0] == "quarter" else 30, "past"
    if ws[:2] == ["quarter", "to"]: h = small(ws[2:]); return (None if h is None else (h - 1) % 24, 45, "to")
    if not ws: return None, None, suf
    h = small(ws[:1]); m = small(ws[1:]) if ws[1:] else 0
    if (m is None or h is None) and len(ws) >= 3 and (small(ws[:2]) or 99) <= 23: h, m = small(ws[:2]), small(ws[2:])
    if suf == "24h" and h is not None and m is None: m = 0
    if h is None or m is None or h > 24 or m > 59: return None, None, suf
    return h, m, ("24h" if h > 12 and suf == "none" else suf)


def digits_profile(ds):
    runs = [len(m.group()) for m in re.finditer(r"(\d)\1+", ds)]
    return len(ds), "none" if not runs else ("triple+" if max(runs) >= 3 else ("multi_pair" if len(runs) > 1 else "pair"))


def mag(v):
    for lim, tag in ((10, "1-9"), (100, "10-99"), (1e3, "100-999"), (1e4, "1k-9k"), (1e5, "10k-99k"), (1e6, "100k-999k")):
        if v < lim: return tag
    return "1M+"


DIGIT_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())} | {"oh": "0", "o": "0"}
CODE_KINDS = ("alphanumeric code", "letter-digit code", "short code", "model number")  # single-token letter-digit codes may carry digits
TENS = r"\b(?:ten|eleven|twelve|thir\w*teen|\w*teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|thousand|million|plus)\b"
WORDCODE_KINDS = ("product or code name", "name with a number in it")  # word+number codes ("Chapter eleven"): words only


def digit_runs(reading):  # maximal runs of spoken digits (zero/oh/one..nine, double/triple X, "hundred" = 00) -> digit strings; any other word breaks a run
    runs, cur, mult = [], "", 1
    for w in re.sub(r"[,.-]", " ", reading.lower()).split() + ["|"]:
        if w in DIGIT_WORDS: cur += DIGIT_WORDS[w] * mult; mult = 1
        elif w in ("double", "triple") and mult == 1: mult = 2 if w == "double" else 3
        elif w == "hundred" and cur: cur += "00"
        else:
            if cur: runs.append(cur)
            cur, mult = "", 1
    return runs


def spoken_digits(reading):  # the digits of a PURE digit-by-digit reading ("seven seven double five"), else None
    if re.sub(r"[,.-]", " ", reading.lower()).split() and all(w in DIGIT_WORDS or w in ("double", "triple") for w in re.sub(r"[,.-]", " ", reading.lower()).split()):
        r = digit_runs(reading); return r[0] if len(r) == 1 else None
    return None


def label_violations(r):  # label-v3 invariant on one sentence row -> list of messages
    bad = []
    for n in r["numbers"]:
        lab, kind, g = n.get("label") or "", n["kind"], str(n.get("groups") or "-"); nodig = re.sub(r"\b[AP]M\b", "", lab)
        if g not in ("-", "") and "L" not in g and not kind.startswith(WORDCODE_KINDS):  # digit-by-digit reading: runs >= 5 digits -> digits, shorter runs stay words
            if not re.search(TENS, n["reading"].lower()):  # pairs like "twenty seven forty nine" are not digit-by-digit: only the dash check applies
                segs = re.split(r"\bext(?:ension)?\b", n["reading"].lower())  # an extension is its own number; fillers inside one number ("442, yes, 5884") do not split it
                want = sum(t for t in (sum(len(x) for x in digit_runs(sg)) for sg in segs) if t >= 5); have = len(re.findall(r"\d", lab))
                if want != have: bad.append(f"digit runs: expected {want} digit chars, label has {have}: {n['reading']!r} -> {lab!r}")
            if re.search(r"\d-\d", lab): bad.append(f"digit label has dashes: {lab!r}")
        elif kind.startswith(WORDCODE_KINDS):  # single-token codes (MP3, 10K, I-9) keep digits; "word number" codes must not
            if re.search(r"\w\s+\d|\d\s+\w", lab): bad.append(f"word+number code has digits: {n['reading']!r} -> {lab!r}")
        elif kind.startswith(CODE_KINDS): pass
        elif re.search(r"\d", nodig): bad.append(f"ordinary ({kind}) label has digits: {n['reading']!r} -> {lab!r}")
        if lab and lab.lower() not in r.get("label_v3", lab).lower(): bad.append(f"label not in label_v3: {lab!r}")
    return bad


def words(t): return [m.group().lower() for m in re.finditer(r"[A-Za-z']+", t)]


def locate(said, reading):  # (first word index, last word index, n words) of reading inside said
    S, Rw = words(said), words(reading)
    for i in range(len(S) - len(Rw) + 1):
        if S[i:i + len(Rw)] == Rw: return i, i + len(Rw) - 1, len(S)
    return None


def q(xs):
    if not xs: return {}
    xs = sorted(xs); n = len(xs)
    return dict(n=n, min=xs[0], p10=xs[n // 10], p50=xs[n // 2], p90=xs[9 * n // 10], max=xs[-1], distinct=len(set(xs)))


def share(c, keys, n): return sum(c.get(k, 0) for k in keys) / max(n, 1)


NUMWORDS = (neardup.NW | neardup.ORD | neardup.DECADES) - {"oh", "o", "point", "double", "triple", "tens", "hundreds", "thousands", "millions", "billions", "dozens"}


def digit_strings(t):  # all digit strings >= 5 long in a label: written digits or spoken digit-by-digit runs
    return [m.group() for m in re.finditer(r"\d{5,}", t)] + [d for d in digit_runs(t.lower()) if len(d) >= 5]


def mix(path):
    spec = json.load(open(os.path.expanduser(path))); flags = []
    tot = collections.Counter(); files = []
    for sp in spec:
        c = collections.Counter(); w = float(sp.get("weight", 1.0)); syn = bool(sp.get("synthetic"))
        for l in open(os.path.expanduser(sp["path"])):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("lang", "en") != "en" or not r.get("duration"): c["skipped_rows"] += 1; continue
            t = r.get("label_v3") or r.get("label") or r.get("text") or ""; h = r["duration"] * w / 3600
            ds = digit_strings(t); free = not re.search(r"\d", t) and not any(x in NUMWORDS for x in re.findall(r"[a-z']+", t.lower()))
            c["en"] += h; c["syn"] += h * syn; c["rows"] += 1
            if ds: c["dig"] += h; c["dig_syn"] += h * syn; c["dig_free"] += h * (not any(re.search(r"(\d)\1", d) for d in ds))
            if free: c["free"] += h
        files.append(dict(path=sp["path"], weight=w, synthetic=syn, **{k: round(v, 3) for k, v in c.items()})); tot.update(c)
    en = max(tot["en"], 1e-9)
    res = dict(english_hours=round(tot["en"], 2), synthetic_share=round(tot["syn"] / en, 4), digit_string_share=round(tot["dig"] / en, 4),
               synthetic_digit_string_share=round(tot["dig_syn"] / en, 4), real_digit_string_share=round((tot["dig"] - tot["dig_syn"]) / en, 4),
               number_free_share=round(tot["free"] / en, 4), repeat_free_digit_share=round(tot["dig_free"] / max(tot["dig"], 1e-9), 4), files=files)
    if not 0.04 <= res["synthetic_digit_string_share"] <= 0.05: flags.append(f"mix: synthetic digit-string share {res['synthetic_digit_string_share']:.2%} of English hours (want 4-5%)")
    if res["number_free_share"] < 0.40: flags.append(f"mix: number-free share {res['number_free_share']:.1%} < 40% of English hours")
    if res["synthetic_share"] > 0.30: flags.append(f"mix: synthetic share {res['synthetic_share']:.1%} of English hours > 30%")
    if not 0.35 <= res["repeat_free_digit_share"] <= 0.65: flags.append(f"mix: repeat-free digit share {res['repeat_free_digit_share']:.0%} (want 35-65%)")
    res["flags"] = flags
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1))
    for f in files: print("  ", json.dumps(f))
    print(f"\nFLAGS ({len(flags)}):"); [print("  FLAG", f) for f in flags]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v2"); ap.add_argument("--kept"); ap.add_argument("--briefs", action="store_true", help="census briefs/<tag>.jsonl (no sentences yet)")
    ap.add_argument("--root", default=".", help="data directory (sentences/, briefs/, plan/, voices.json; census/ is written here)")
    ap.add_argument("--prefix", default="pack", help="audio row id prefix: <prefix>_<tag>_<sha1[:10]>")
    ap.add_argument("--addon", action="store_true", help="add-on pack: skip value-spread flags for number types it does not contain")
    ap.add_argument("--outdir", default="census", help="where census/<tag>.json is written (relative to --root)")
    ap.add_argument("--mix", help="JSON mix spec (see docstring): per-hour shares over a weighted mix of label files; text only")
    a = ap.parse_args()
    if a.mix: return mix(a.mix)
    os.chdir(a.root)
    if a.briefs:
        allrows = [dict(b, said=" ... ".join(n["reading"] for n in b["numbers"]) or "none") for b in map(json.loads, open(f"briefs/{a.tag}.jsonl"))]
    else:
        allrows = [json.loads(l) for l in open(f"sentences/{a.tag}.jsonl")]
    rid = lambda b: f"{a.prefix}_{a.tag}_" + hashlib.sha1((b["cell"] + b["said"] + b["id"]).encode()).hexdigest()[:10]
    kept = {}
    if a.kept:
        for l in open(a.kept):
            if l.strip(): x = json.loads(l); kept[x["id"]] = x
    R = [b for b in allrows if not a.kept or rid(b) in kept]
    out, flags = {}, []

    def F(ok, msg):  # add-on packs (--addon) only flag checks whose number type is present
        if not ok and not (a.addon and re.search(r" 0%| only 0 |never|miss|< 5% five|only \[\]", msg)): flags.append(msg)

    by, byo = collections.defaultdict(list), collections.defaultdict(list)
    for r in R:
        for n in r["numbers"]:
            by[n["kind"]].append((r, n))
            if not n["repeat"] and not re.search(r"\b(um|uh|er)\b", n["reading"]): byo[n["kind"]].append((r, n))  # hesitation readings: coverage only
    out["rows"] = len(R); out["cells"] = dict(collections.Counter(r["cell"] for r in R))
    out["numbers_per_row"] = dict(collections.Counter(len(r["numbers"]) for r in R))
    out["kinds"] = {k: len(v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    cxk = collections.defaultdict(collections.Counter)
    for k, v in by.items():
        for r, _ in v: cxk[k][r["cell"]] += 1
    out["kind_by_cell"] = {k: dict(c) for k, c in cxk.items()}

    if a.kept:
        missing = set(kept) - {rid(b) for b in allrows}
        F(not missing, f"kept: {len(missing)} kept ids not in sentences/{a.tag}.jsonl")
        planned = collections.Counter(b["cell"] for b in allrows)
        out["yield_by_cell"] = {c: round(out["cells"].get(c, 0) / n, 3) for c, n in planned.items()}
        for c, y in out["yield_by_cell"].items(): F(y >= 0.6, f"yield: cell {c} kept {y:.0%}")

    # --- distinct readings per kind (catches tiny value lists in repeat/code cells)
    dist = {}
    for k, v in by.items():
        c = collections.Counter(n["reading"].lower() for _, n in v); top, tn = c.most_common(1)[0]
        dist[k] = dict(n=len(v), distinct=len(c), top=top, top_share=round(tn / len(v), 3))
        if len(v) >= 300 and k not in ("small count", "short number said aloud"): F(len(c) >= 50 and tn / len(v) <= 0.05, f"distinct: {k} has {len(c)} readings, top '{top}' {tn / len(v):.0%}")
    out["distinct_readings"] = dist

    # --- years (ordinary)
    ys, yforms, un = [], collections.Counter(), 0
    for _, n in byo["a year"]:
        ws = n["reading"].lower().split(); y = year(ws)
        if y is None: un += 1; continue
        ys.append(y); yforms["two_thousand" if ws[:2] == ["two", "thousand"] else "oh" if "oh" in ws else "hundred" if "hundred" in ws else "pair"] += 1
    F(un == 0, f"years: {un} unparsed")
    sp = [span_start(n["reading"]) for k in ("a span of years", "a span of years (season or fiscal year)") for _, n in byo[k]]
    F(None not in sp, f"year spans: {sp.count(None)} unparsed"); sp = [y for y in sp if y]
    n_y = max(len(ys), 1)
    cur, fut = sum(2010 <= y <= 2035 for y in ys) / n_y, sum(y > NOW for y in ys) / n_y
    out["years"] = dict(q(ys), forms=dict(yforms), centuries=dict(sorted(collections.Counter(y // 100 * 100 for y in ys).items())),
                        decades=len({y // 10 for y in ys}), share_2010_2035=round(cur, 3), share_2000_2009=round(sum(2000 <= y <= 2009 for y in ys) / n_y, 3),
                        future_share=round(fut, 3), span_start=q(sp), span_2000s=round(sum(y >= 2000 for y in sp) / max(len(sp), 1), 3))
    F(cur >= 0.30, f"years: only {cur:.0%} in 2010-2035 (meeting-era years)")
    F(fut >= 0.10, f"years: future share {fut:.0%} < 10%")
    F(out["years"]["span_2000s"] >= 0.5, f"year spans: only {out['years']['span_2000s']:.0%} start in 2000+")

    # --- clock (ordinary)
    hs, ms, sfx, un = [], [], collections.Counter(), 0
    for _, n in byo["a time of day"]:
        h, m, s = clock(n["reading"])
        if h is None: un += 1; continue
        hs.append(h); ms.append(m); sfx[s] += 1
    F(un == 0, f"times: {un} unparsed")
    offgrid = sum(m % 5 != 0 for m in ms) / max(len(ms), 1)
    out["times"] = dict(n=len(hs), hours=dict(sorted(collections.Counter(hs).items())), distinct_minutes=len(set(ms)), offgrid_share=round(offgrid, 3),
                        distinct_hm=len(set(zip(hs, ms))), suffixes=dict(sfx))
    F(offgrid >= 0.30, f"times: only {offgrid:.0%} ordinary minutes off the 5-minute grid ({len(set(ms))} distinct)")
    F(len(sfx) >= 6, f"times: only {len(sfx)} suffix styles {sorted(sfx)} (want o'clock/noon/evening/24h/quarter past...)")

    # --- money
    mv, un, styles = [], 0, collections.Counter()
    for _, n in byo["an amount of money"]:
        v = money(n["reading"]); r = n["reading"].lower()
        if v is None: un += 1; continue
        mv.append(v); styles["cents" if "cent" in r else "hundred_bucks" if "hundred bucks" in r else "point_scale" if "point" in r else "dollars" if "dollar" in r else "bare"] += 1
    F(un == 0, f"money: {un} unparsed")
    mm = collections.Counter(mag(v) for v in mv); nm = max(len(mv), 1)
    odd = sum(bool(re.match(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety) hundred", n["reading"].lower())) for _, n in byo["an amount of money"])
    out["money"] = dict(n=len(mv), magnitude=dict(mm), styles=dict(styles), with_cents=styles["cents"], unnatural_tens_hundred=odd)
    for b in ("1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k-999k", "1M+"): F(mm[b] / nm >= 0.05, f"money: {b} only {mm[b] / nm:.0%}")
    F(styles["cents"] / nm >= 0.15, f"money: cents {styles['cents'] / nm:.0%} < 15%")
    F(odd == 0, f"money: {odd} unnatural 'tens hundred' readings")

    # --- decimals, percent, counts
    dv, dp, un = [], collections.Counter(), 0
    for _, n in byo["a measurement with a decimal"]:
        v, p = decimal(n["reading"])
        if v is None: un += 1; continue
        dv.append(v); dp[p] += 1
    F(un == 0, f"decimals: {un} unparsed")
    lt1 = sum(v < 1 for v in dv) / max(len(dv), 1)
    oh = sum(bool(re.search(r"point oh\b|\boh\b", n["reading"].lower())) for _, n in byo["a measurement with a decimal"])
    out["decimals"] = dict(q(dv), places=dict(sorted(dp.items())), lt1_share=round(lt1, 3), says_oh=oh, magnitude=dict(collections.Counter(mag(v) for v in dv)))
    F(lt1 >= 0.10, f"decimals: only {lt1:.0%} below 1")
    F(len(dp) >= 3, f"decimals: places only {sorted(dp)}")
    F(oh > 0, "decimals: never 'point oh'")
    pv, un = [], 0
    for _, n in byo["a percentage"]:
        r = n["reading"].lower().replace("percent", "").split()
        v = decimal(" ".join(r))[0] if "point" in r else cardinal(r)
        if v is None: un += 1
        else: pv.append(v)
    F(un == 0, f"percent: {un} unparsed")
    pdec = sum(v != int(v) for v in pv) / max(len(pv), 1)
    out["percent"] = dict(q(pv), decimal_share=round(pdec, 3), over_100=sum(v > 100 for v in pv))
    F(pdec >= 0.10, f"percent: decimal share {pdec:.0%}")
    cv = [cardinal(n["reading"].lower().split()) for _, n in byo["a count of things"]]
    F(None not in cv, f"counts: {cv.count(None)} unparsed"); cv = [v for v in cv if v is not None]
    cm = collections.Counter(mag(v) for v in cv)
    out["counts"] = dict(q(cv), magnitude=dict(cm))
    for b in ("1-9", "10-99", "100-999", "1k-9k", "10k-99k"): F(cm[b] / max(len(cv), 1) >= 0.05, f"counts: {b} only {cm[b] / max(len(cv), 1):.0%}")

    # --- dates
    mo, dy, un = collections.Counter(), collections.Counter(), 0
    for _, n in byo["a date"]:
        ws = n["reading"].lower().split(); m = next((w for w in ws if w in MONTHS), None)
        d = None
        for i, w in enumerate(ws):
            if w in ORD: d = ORD[w] + (D.get(ws[i - 1], 0) if i else 0)
        if d is None: un += 1; continue
        mo[m or "no_month"] += 1; dy[d] += 1
    out["dates"] = dict(n=len(byo["a date"]), months=dict(mo), distinct_days=len(dy), distinct_readings=dist.get("a date", {}).get("distinct"),
                        with_year=sum(bool(year(n["reading"].lower().split()[-2:])) for _, n in byo["a date"]))
    F(un == 0, f"dates: {un} unparsed")
    F(len(set(mo) - {"no_month"}) == 12, f"dates: only {len(set(mo) - {'no_month'})} months")
    F(len(dy) >= 28, f"dates: only {len(dy)} distinct days")

    # --- ranges and scores
    rr = [range_ends(n["reading"]) for k in RANGE_KINDS for _, n in byo[k]]
    out["ranges"] = dict(q([e[1] for e in rr if e]), unparsed=rr.count(None), reversed=sum(bool(e) and e[0] > e[1] for e in rr))
    F(rr.count(None) == 0, f"ranges: {rr.count(None)} unparsed")

    # --- digit strings, by cell
    dig = {}
    for k, v in by.items():
        vals = [(r["cell"], n["value"], n["reading"].lower()) for r, n in v if n["value"] and re.fullmatch(r"\d+", n["value"])]
        if len(vals) < 0.8 * len(v): continue
        prof = [digits_profile(x) for _, x, _ in vals]
        dig[k] = dict(n=len(vals), distinct=len({x for _, x, _ in vals}), cells=dict(collections.Counter(c for c, _, _ in vals)),
                      length=dict(sorted(collections.Counter(p[0] for p in prof).items())), repeat=dict(collections.Counter(p[1] for p in prof)),
                      adjacent_same_word=sum(bool(re.search(r"\b(\w+) \1\b", r)) for _, _, r in vals),
                      cross_group_repeat=sum(bool(re.search(r"\b(\w+), \1\b", r)) for _, _, r in vals),
                      says_double=sum("double" in r for _, _, r in vals), says_triple=sum("triple" in r for _, _, r in vals),
                      says_oh=sum(bool(re.search(r"\boh\b", r)) for _, _, r in vals), says_zero=sum("zero" in r for _, _, r in vals))
        if max(len(x) for _, x, _ in vals) >= 6: F(dig[k]["distinct"] >= 0.95 * len(vals), f"{k}: duplicate values")
    out["digit_strings"] = dig
    F(set(dig.get("phone number", {}).get("length", {})) >= {7, 10, 11}, "phones: lengths miss 7/10/11")

    # --- v3 checks (Opus review 10-24): repeat balance, natural grouping, dash-free labels, 5-digit coverage
    ds_all = [str(n["value"]) for r in R for n in r["numbers"] if re.fullmatch(r"\d{5,}", str(n.get("value") or "")) and n.get("label") != n.get("reading")]
    if ds_all:
        free = sum(not re.search(r"(\d)\1", d) for d in ds_all) / len(ds_all)
        out["digit_repeat_free_share"] = round(free, 3)
        F(0.35 <= free <= 0.65, f"digits: repeat-free share {free:.0%} (want 35-65%; 93% repeats risks insertions)")
        F(sum(len(d) == 5 for d in ds_all) >= 0.05 * len(ds_all), "digits: < 5% five-digit strings")
    ph = [n for r in R for n in r["numbers"] if n["kind"] == "phone number" and n.get("groups")]
    if ph:
        nat = sum(n["groups"] in ("3-3-4", "1-3-3-4", "3-4") for n in ph) / len(ph)
        out["phone_natural_grouping"] = round(nat, 3)
        F(nat >= 0.8, f"phones: natural grouping {nat:.0%} < 80%")
    dashed = sum(1 for r in R for n in r["numbers"] if re.fullmatch(r"[\d-]+", n.get("label") or "") and "-" in n["label"])
    out["dashed_digit_labels"] = dashed
    F(dashed == 0, f"labels: {dashed} digit labels with dashes (real labels have none)")
    if not a.briefs:
        viol = [(r, m) for r in R for m in label_violations(r)]
        out["label_v3_violations"] = dict(n=len(viol), examples=[m for _, m in viol[:8]], by_type=dict(collections.Counter(m.split(":")[0].split(" (")[0] for _, m in viol)))
        F(not viol, f"label v3: {len(viol)} violations {out['label_v3_violations']['by_type']} e.g. {[m for _, m in viol[:2]]}")
        fr = [neardup.frame(r["said"]) for r in R]
        for k in (2, 3):
            g = [tuple(w[i:i + k]) for w in fr for i in range(len(w) - k + 1)]
            out[f"distinct_{k}"] = round(len(set(g)) / max(len(g), 1), 4)
        tp = collections.Counter(" ".join(w) for w in fr); top50 = sum(c for _, c in tp.most_common(50)) / max(len(R), 1)
        out["top50_template_share"] = round(top50, 4); out["top_templates"] = tp.most_common(5)
        F(top50 <= 0.05, f"crowding: top-50 number-masked templates are {top50:.1%} of rows (want <= 5%)")
        nm = collections.Counter(m for r in R for m in re.findall(r"\b(?!I\b)[A-Z][a-z]+\b", r["said"][1:]))
        stop = {"The", "We", "Please", "That", "This", "It", "Our", "Can", "Could", "Did", "Do", "Is", "Are", "What", "Let", "So", "And", "But", "If", "When", "Your", "You", "Mr", "Mrs", "Dr", "Ms"}
        names = [(k, v) for k, v in nm.most_common(60) if k not in stop]
        caps = sum(v for k, v in nm.items() if k not in stop)
        if names and caps:
            out["top_name"] = names[0]
            F(names[0][1] / caps <= 0.03, f"variety: top name '{names[0][0]}' is {names[0][1] / caps:.1%} of capitalized words")
        op = collections.Counter(" ".join(words(r["said"])[:3]) for r in R if r["numbers"] and not re.match(r"\d|zero|one|two|three|four|five|six|seven|eight|nine", r["said"].lower()))
        if op:
            o, k = op.most_common(1)[0]; out["top_opening"] = (o, k)
            F(k <= max(5, 0.01 * len(R)), f"variety: opening '{o}' used {k}x")

    # --- position of each number in its sentence (start / middle / end) per kind
    pos = {}
    for k, v in by.items():
        c = collections.Counter()
        for r, n in v:
            L = locate(r["said"], n["reading"])
            c["unlocated" if not L else "start" if L[0] == 0 else "end" if L[1] >= L[2] - 1 else "middle"] += 1
        pos[k] = dict(c)
        if k in ORDINARY_KINDS and len(v) >= 300 and not a.briefs:  # briefs have no sentence yet: positions are only real on sentences
            for p in ("start", "end"): F(c[p] / len(v) >= 0.10, f"position: {k} at sentence {p} only {c[p] / len(v):.0%}")
    out["position"] = pos

    # --- duplicates, length
    said = collections.Counter(r["said"].lower() for r in R)
    out["dup_sentences"] = sum(n - 1 for n in said.values() if n > 1)
    ln = collections.defaultdict(list)
    for r in R: ln[r["cell"]].append(len(words(r["said"])))
    out["length_by_cell"] = {c: q(v) for c, v in ln.items()}
    F(a.briefs or out["dup_sentences"] <= 0.01 * len(R), f"duplicates: {out['dup_sentences']} repeated sentences")

    # --- voices / engine / speed / gender per kind (plan) and audio facts (kept)
    planp = f"plan/{a.tag}.jsonl"
    if os.path.exists(planp):
        P = {x["id"]: x for x in map(json.loads, open(planp))}
        V = json.load(open("voices.json")); gm = {x["ref_spk"]: x.get("gender") for x in V["refs"]}
        vk = {}
        for k, v in by.items():
            rows = [P[rid(r)] for r, _ in v if rid(r) in P]
            g = collections.Counter("kokoro" if x["engine_src"] == "kokoro" else gm.get(x.get("ref_spk")) or "unknown" for x in rows)
            vk[k] = dict(engines=dict(collections.Counter(x["engine_src"] for x in rows)), voices=len({x["voice"] for x in rows}), gender=dict(g),
                         speed=q([x["speed"] for x in rows]))
        out["voices_by_kind"] = vk
        g = collections.Counter(gm.get(x.get("ref_spk")) for x in P.values() if x["engine_src"] != "kokoro"); mf = g["male"] + g["female"]
        F(0.4 <= g["female"] / max(mf, 1) <= 0.6, f"gender: cloned voices {g['female'] / max(mf, 1):.0%} female ({g[None]} rows unknown)")
    if a.kept:
        edge, aug = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        for k, v in by.items():
            for r, n in v:
                x = kept[rid(r)]; L = locate(r["said"], n["reading"]); wt = x.get("word_times")
                aug[k]["clean" if not x.get("aug") else "+".join(sorted(str(t) for t in x["aug"])) if isinstance(x.get("aug"), list) else str(x["aug"])] += 1
                if not L or not wt or len(wt) != L[2]: edge[k]["no_times"] += 1; continue
                t0, t1, dur = wt[L[0]][0], wt[L[1]][1], x.get("duration") or wt[-1][1]
                edge[k]["start<1s" if t0 < 1.0 else "end<1s" if dur - t1 < 1.0 else "interior"] += 1
        out["time_edge_by_kind"] = {k: dict(c) for k, c in edge.items()}
        out["aug_by_kind"] = {k: dict(c.most_common(12)) for k, c in aug.items()}

    out["flags"] = flags
    os.makedirs(a.outdir, exist_ok=True)
    p = f"{a.outdir}/{a.tag}{'_kept' if a.kept else ''}.json"
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("rows", "cells", "numbers_per_row", "dup_sentences")}))
    for k in ("years", "times", "money", "decimals", "percent", "counts", "dates", "ranges"): print(f"[{k}]", json.dumps(out[k], default=str)[:700])
    print("[distinct, kinds >=300 rows]", {k: (v["distinct"], v["top_share"]) for k, v in dist.items() if v["n"] >= 300})
    print("[position]", {k: v for k, v in pos.items() if k in ORDINARY_KINDS})
    print(f"\nFLAGS ({len(flags)}):"); [print("  FLAG", f) for f in flags]
    print("->", p)


if __name__ == "__main__": main()
