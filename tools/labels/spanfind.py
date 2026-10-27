#!/usr/bin/env python3
"""Words -> number spans and candidate readings for English labels that carry no digit caption ($0, stdlib; self-contained, does NOT import numeq / numkeys).
find_spans(label) -> [Span]   maximal runs of number words (+ adjacent unit / letter tokens), with every digit-string reading of the run
candidates(span)  -> list[str] candidate spoken readings of the same VALUE (no commas, no digits, no hyphens); the CURRENT reading is always first.
apply_candidate(label, span, cand) -> new label (commas inside the span stripped, trailing punctuation kept)
No commas inside number readings, so the current reading is comma-free as well (`reading`)."""
import itertools, re, sys
sys.dont_write_bytecode = True

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENSW = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
UNIT = {w: i for i, w in enumerate(ONES) if 0 < i < 10}
TEEN = {w: i for i, w in enumerate(ONES) if 10 <= i < 20}
TENS = {w: i * 10 for i, w in enumerate(TENSW) if i >= 2}
SCALE = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12}
ORD_UNIT = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
ORD_TEEN = {"tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19}
ORD_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}
ORD_BIG = {"hundredth": 100, "thousandth": 1000, "millionth": 10 ** 6}
ORD_ALL = {**ORD_UNIT, **ORD_TEEN, **ORD_TENS, **ORD_BIG}
DIGW = {w: str(i) for i, w in enumerate(ONES) if i < 10}
DIGW["oh"] = "0"
UNITS_AFTER = {"percent", "dollars", "dollar", "cents", "cent", "euros", "euro", "pounds", "pound", "degrees", "degree", "o'clock", "oclock"}
NUMVOC = set(ONES) | set(TENS) | {"hundred", "oh", "double", "triple"} | set(SCALE)
DECADES = {w[:-1] + "ies": v for w, v in TENS.items()}   # twenties .. nineties
PUNCT = ".,;:!?\"()[]“”"
LETTER = re.compile(r"^[B-HJ-Z]$")      # a single capital letter that is not the article / pronoun
CAPS = re.compile(r"^[A-Z]{2,4}$")
AMPM = re.compile(r"^(?:[AaPp]\.?[Mm]\.?|AM|PM)$")


def below100(n):
    return ONES[n] if n < 20 else TENSW[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


def below1000(n, and_=False):
    h, r = divmod(n, 100)
    out = []
    if h:
        out.append(ONES[h] + " hundred")
    if r:
        if h and and_:
            out.append("and")
        out.append(below100(r))
    return " ".join(out)


def cardinal(n, and_=False):
    if n == 0:
        return "zero"
    parts = []
    for div, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        q, n = divmod(n, div)
        if q:
            parts.append(below1000(q, and_) + " " + name)
    if n:
        if parts and and_ and n < 100:
            parts.append("and")
        parts.append(below1000(n, and_))
    return " ".join(parts)


def ordinal(n):
    w = cardinal(n).split(" ")
    last = w[-1]
    inv = {v: k for k, v in ORD_UNIT.items()}
    if last in ("one", "two", "three", "five", "eight", "nine", "twelve"):
        w[-1] = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}[last]
    elif last.endswith("y"):
        w[-1] = last[:-1] + "ieth"
    else:
        w[-1] = last + "th"
    return " ".join(w)


# ---------------------------------------------------------------------------- tokens
class Tok:
    __slots__ = ("raw", "core", "low", "start", "end", "lead", "trail")

    def __init__(self, raw, start):
        self.raw = raw; self.start = start; self.end = start + len(raw)
        a = 0; b = len(raw)
        while a < b and raw[a] in PUNCT:
            a += 1
        while b > a and raw[b - 1] in PUNCT:
            b -= 1
        self.lead = raw[:a]; self.core = raw[a:b]; self.trail = raw[b:]
        self.low = self.core.lower()


def tokenize(text):
    return [Tok(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]


# ---------------------------------------------------------------------------- group parse / readings
def parse_group(ws):
    """cardinal group value or None (twenty one, one hundred and five, two thousand ...)"""
    total, cur, last_scale = 0, 0, None
    for w in ws:
        if w == "and":
            if not (cur >= 100 or total > 0):
                return None
        elif w in UNIT or w in TEEN or w in TENS:
            v = UNIT.get(w) or TEEN.get(w) or TENS[w]
            m = cur % 100
            if not (m == 0 or (w in UNIT and m >= 20 and m % 10 == 0)):
                return None
            cur += v
        elif w == "hundred":
            if not (0 < cur < 10 or (10 <= cur < 100 and total == 0 and last_scale is None)):
                return None
            cur *= 100
        elif w in SCALE:
            s = SCALE[w]
            if cur == 0 or (last_scale is not None and s >= last_scale):
                return None
            total += cur * s; cur = 0; last_scale = s
        else:
            return None
    if ws and ws[-1] == "and":
        return None
    return total + cur


def readings(ws, cap=60):
    """-> list of (digit string, ngroups): every partition of the word run into cardinal groups / digit words / double-triple groups"""
    n = len(ws); memo = {}

    def go(i):
        if i == n:
            return {("", 0)}
        if i in memo:
            return memo[i]
        out = set()
        w = ws[i]
        if w in ("oh", "zero"):
            for r, g in go(i + 1):
                out.add(("0" + r, g + 1))
        if w in ("double", "triple") and i + 1 < n and ws[i + 1] in DIGW:
            k = 2 if w == "double" else 3
            for r, g in go(i + 2):
                out.add((DIGW[ws[i + 1]] * k + r, g + 1))
        if w not in ("oh", "zero", "double", "triple"):
            for j in range(i + 1, min(n, i + 14) + 1):
                v = parse_group(ws[i:j])
                if v is not None:
                    for r, g in go(j):
                        out.add((str(v) + r, g + 1))
                    if len(out) > cap:
                        break
        memo[i] = set(sorted(out, key=lambda t: (t[1], len(t[0]), t[0]))[:cap])
        return memo[i]

    return sorted(go(0), key=lambda t: (t[1], len(t[0]), t[0]))


def segment(ws):
    """min-group segmentation of the run -> [(i, j, value string)] (token ranges), or None"""
    best = None
    n = len(ws)
    memo = {}

    def go(i):
        if i == n:
            return (0, [])
        if i in memo:
            return memo[i]
        res = None
        w = ws[i]
        opts = []
        if w in ("oh", "zero"):
            opts.append((i + 1, "0"))
        if w in ("double", "triple") and i + 1 < n and ws[i + 1] in DIGW:
            opts.append((i + 2, DIGW[ws[i + 1]] * (2 if w == "double" else 3)))
        if w not in ("oh", "zero", "double", "triple"):
            for j in range(i + 1, min(n, i + 14) + 1):
                v = parse_group(ws[i:j])
                if v is not None:
                    opts.append((j, str(v)))
        for j, val in opts:
            sub = go(j)
            if sub is None:
                continue
            cand = (sub[0] + 1, [(i, j, val)] + sub[1])
            if res is None or cand[0] < res[0]:
                res = cand
        memo[i] = res
        return res

    r = go(0)
    return r[1] if r else None


# ---------------------------------------------------------------------------- spans
class Span:
    def __init__(self):
        self.w_start = self.w_end = 0
        self.char_start = self.char_end = 0
        self.text = ""
        self.reading = ""           # comma-free lowercase-preserved tokens of the whole span
        self.kind = ""
        self.value = ""             # primary digit-string / decimal / ordinal value
        self.values = []            # every digit-string reading
        self.prefix = []            # fixed leading tokens (letters / caps codes)
        self.core = []              # number-word tokens (lowercase)
        self.suffix = []            # fixed trailing tokens (units, letters, am/pm)
        self.weak = False           # a lone one..nine / 'a hundred' style token that may not be a number at all
        self.notes = []

    def as_dict(self):
        return {k: getattr(self, k) for k in ("w_start", "w_end", "char_start", "char_end", "text", "reading", "kind", "value", "values", "prefix", "core", "suffix", "weak", "notes")}


def _is_num(t, toks, i):
    w = t.low
    if t.core == "" or not t.core.replace("'", "").replace("-", "").isalpha():
        return False
    if w in NUMVOC and w != "oh" and w != "double" and w != "triple":
        return True
    if w == "oh" or w == "double" or w == "triple":
        nxt = toks[i + 1].low if i + 1 < len(toks) else ""
        prv = toks[i - 1].low if i > 0 else ""
        if w == "oh":
            return (nxt in DIGW or nxt in ("point",)) and (prv in NUMVOC and prv not in ("hundred",) or True) and (prv in DIGW or prv in NUMVOC or nxt in DIGW)
        return nxt in DIGW
    return False


def find_spans(text):
    toks = tokenize(text)
    # split hyphenated number words ('twenty-one') by rewriting low
    for t in toks:
        if "-" in t.core and all(p in NUMVOC for p in t.low.split("-")):
            pass
    spans = []
    i = 0; n = len(toks)
    while i < n:
        t = toks[i]
        w = t.low
        start = None
        if w in DECADES:
            sp = Span(); sp.kind = "decade"; sp.core = [w]; sp.value = str(DECADES[w]); sp.values = [sp.value]
            _finish(sp, toks, i, i, [], []); spans.append(sp); i += 1; continue
        if w in ORD_ALL or (w in TENS and i + 1 < n and toks[i + 1].low in ORD_UNIT):
            j = i
            if w in TENS:
                j = i + 1
            sp = Span(); sp.kind = "ordinal"; sp.core = [toks[k].low for k in range(i, j + 1)]
            v = ORD_ALL.get(toks[j].low, 0) + (TENS[w] if w in TENS else 0)
            if toks[j].low in ORD_UNIT or toks[j].low in ORD_TEEN or toks[j].low in ORD_TENS or toks[j].low in ORD_BIG:
                sp.value = str(v); sp.values = [str(v)]
                _finish(sp, toks, i, j, [], []); sp.weak = (v <= 10 and j == i and toks[i].low in ("first", "second", "third"))
                spans.append(sp); i = j + 1; continue
        if _is_num(t, toks, i) or (w == "a" and i + 1 < n and toks[i + 1].low in ("hundred", "thousand", "million", "billion") and (i == 0 or not _is_num(toks[i - 1], toks, i - 1))):
            j = i
            run = []
            while j < n:
                tj = toks[j]
                lw = tj.low
                if "-" in tj.core and all(p in NUMVOC for p in lw.split("-")):
                    run.extend(lw.split("-")); j += 1; continue
                if _is_num(tj, toks, j) or (j == i and lw == "a"):
                    run.append(lw); j += 1; continue
                if lw == "and" and run and run[-1] in ("hundred", "thousand", "million", "billion") and j + 1 < n and _is_num(toks[j + 1], toks, j + 1):
                    run.append("and"); j += 1; continue
                if lw == "point" and run and j + 1 < n and (toks[j + 1].low in DIGW):
                    run.append("point"); j += 1; continue
                break
                # unreachable
            last = j - 1
            if run and run[0] == "a":
                run_n = ["one"] + run[1:]
            else:
                run_n = run
            # trailing separators inside the run are not part of the number
            sp = Span(); sp.core = list(run)
            pre = []; suf = []
            # prefix: adjacent caps / single capital letters directly before
            k = i - 1
            while k >= 0 and (LETTER.match(toks[k].core) or CAPS.match(toks[k].core)) and not toks[k].trail.strip(PUNCT.replace(",", "")) and i - k <= 3:
                pre.insert(0, toks[k].core); k -= 1
            first = i - len(pre)
            # suffix: units / am pm / letters directly after
            m = last + 1
            while m < n:
                tm = toks[m]
                if tm.low in UNITS_AFTER or AMPM.match(tm.core) and tm.core.lower() not in ("a",):
                    suf.append(tm.core); m += 1
                elif (tm.core in ("a", "p", "A", "P") and m + 1 < n and toks[m + 1].core in ("m", "M") and not suf):
                    suf.extend([tm.core, toks[m + 1].core]); m += 2
                elif (LETTER.match(tm.core) or CAPS.match(tm.core)) and len(suf) < 3 and not toks[m - 1].trail.strip(PUNCT.replace(",", "")):
                    suf.append(tm.core); m += 1
                else:
                    break
            lastw = last + len(suf)
            _classify(sp, run_n)
            sp.prefix = pre; sp.suffix = suf
            sp.weak = (len(run) == 1 and run[0] in UNIT and not pre and not suf) or (len(run) == 1 and run[0] in ("zero", "hundred", "thousand")) or run == ["a", "hundred"]
            _finish(sp, toks, first, lastw, pre, suf)
            spans.append(sp); i = lastw + 1; continue
        i += 1
    return spans


def _classify(sp, run):
    if "point" in run:
        k = run.index("point")
        ip = run[:k]; fp = run[k + 1:]
        rs = readings(ip) if ip else [("0", 1)]
        fd = "".join(DIGW.get(w, "?") for w in fp)
        if "?" in fd or not fd or not rs:
            sp.kind = "unparsed"; sp.values = []; sp.value = ""
            return
        mg = min(g for _, g in rs)
        sp.values = list(dict.fromkeys(r + "." + fd for r, g in rs if g == mg))[:6]
        sp.value = sp.values[0]; sp.kind = "decimal"
        return
    rd = readings(run)
    if rd:
        mg = min(g for _, g in rd)
        rd = [(r, g) for r, g in rd if g == mg]
    sp.values = list(dict.fromkeys(r for r, _ in rd[:12]))
    sp.value = sp.values[0] if sp.values else ""
    digit_only = all(w in DIGW or w in ("double", "triple") for w in run)
    if not rd:
        sp.kind = "unparsed"
    elif digit_only and len(sp.value) >= 2:
        sp.kind = "digits"
    elif any(w in SCALE or w == "hundred" for w in run):
        sp.kind = "int"
    elif len(run) == 1:
        sp.kind = "int"
    else:
        sp.kind = "pairs"     # twenty fifteen / seven fifty three / ten thirty: group-concatenated readings


def text_of(toks, a, b):
    return " ".join(t.raw for t in toks[a:b]) + (" " if b > a else "") + toks[b].lead + toks[b].core if True else ""


def _finish(sp, toks, a, b, pre, suf):
    sp.w_start, sp.w_end = a, b
    sp.char_start = toks[a].start + len(toks[a].lead)
    sp.char_end = toks[b].end - len(toks[b].trail)
    sp.text = text_of(toks, a, b)
    words = []
    for t in toks[a:b + 1]:
        c = t.core
        words.extend(c.split("-") if "-" in c and all(p.lower() in NUMVOC for p in c.split("-")) else [c])
    sp.reading = " ".join(words)


# ---------------------------------------------------------------------------- candidates
def digit_words(d, z, dbl=False):
    w = [(z if c == "0" else ONES[int(c)]) for c in d]
    if not dbl:
        return " ".join(w)
    out, i = [], 0
    while i < len(d):
        if i + 1 < len(d) and d[i] == d[i + 1]:
            out.append("double " + w[i]); i += 2
        else:
            out.append(w[i]); i += 1
    return " ".join(out)


def pair_words(g):
    """a 1-3 digit group read as one number ('oh five', 'fifty three', 'one oh five', 'one hundred five')"""
    if g[0] == "0" and len(g) > 1:
        return ["oh " + pair_words(g[1:])[0], "zero " + pair_words(g[1:])[0]] if g[1:] != "0" * len(g[1:]) else [digit_words(g, "oh"), digit_words(g, "zero")]
    return [cardinal(int(g))]


def groupings(d):
    out = set()
    L = len(d)
    if L < 3 or L > 8:
        return out
    for size in (2, 3):
        for left in (True, False):
            parts = []
            i = 0
            if left:
                if L % size:
                    continue
                while i < L:
                    parts.append(d[i:i + size]); i += size
            else:
                r = L % size
                if r:
                    parts.append(d[:r]); i = r
                while i < L:
                    parts.append(d[i:i + size]); i += size
            opts = [pair_words(p) for p in parts]
            if len(parts) < 2:
                continue
            for combo in itertools.product(*opts):
                out.add(" ".join(combo))
                if len(out) > 40:
                    return out
    # hundreds / thousands pairs: 1200 -> twelve hundred
    if L == 4 and d[2:] == "00" and d[0] != "0":
        out.add(cardinal(int(d[:2])) + " hundred")
    return out


def year_words(d):
    n = int(d)
    out = []
    hi, lo = divmod(n, 100)
    if 1100 <= n <= 1999 or 2010 <= n <= 2099:
        if lo == 0:
            out.append(below100(hi) + " hundred")
        elif lo < 10:
            out += [below100(hi) + " oh " + ONES[lo], below100(hi) + " zero " + ONES[lo]]
        else:
            out.append(below100(hi) + " " + below100(lo))
    if 2000 <= n <= 2099:
        out.append("two thousand" if lo == 0 else "two thousand " + below100(lo))
        if lo:
            out.append("two thousand and " + below100(lo))
        if 2001 <= n <= 2009:
            out.append("twenty oh " + ONES[lo])
    return out


def num_variants(d, with_pairs=True, big=True):
    """every plausible spoken reading of the digit string d (an integer / code number, no decimals)"""
    out = []
    if d == "":
        return out
    lead0 = len(d) > 1 and d[0] == "0"
    if not lead0 and len(d) <= (13 if big else 4):
        n = int(d)
        out += [cardinal(n)]
        c2 = cardinal(n, True)
        if c2 != out[0]:
            out.append(c2)
        for x in list(out):
            if (n >= 100 and x.startswith("one hundred")) or (n >= 1000 and x.startswith("one thousand")) or (n >= 10 ** 6 and x.startswith("one million")):
                out.append("a" + x[3:])
        if len(d) == 4:
            out += year_words(d)
        if with_pairs and 3 <= len(d) <= 8:
            out += sorted(groupings(d))
    elif lead0 and with_pairs and len(d) <= 8:
        out += sorted(groupings(d))
    if len(d) >= 2 or True:
        for z in ("oh", "zero"):
            out.append(digit_words(d, z))
            dw = digit_words(d, z, True)
            if dw != out[-1]:
                out.append(dw)
    return list(dict.fromkeys(out))


def repeat_variants(core_words):
    """+-1 repeat-count variants of a digit-word run ('five five' <-> 'five' <-> 'five five five'); one run changed at a time (at most 6)."""
    toks = list(core_words)
    runs = []
    i = 0
    while i < len(toks):
        j = i
        while j + 1 < len(toks) and toks[j + 1] == toks[i] and toks[i] in DIGW:
            j += 1
        if toks[i] in DIGW:
            runs.append((i, j))
        i = j + 1
    out = []
    for a, b in runs[:6]:
        k = b - a + 1
        out.append(toks[:a] + [toks[a]] * (k + 1) + toks[b + 1:])
        if k >= 2:
            out.append(toks[:a] + [toks[a]] * (k - 1) + toks[b + 1:])
    return [" ".join(o) for o in out if o != toks]


def candidates(sp, max_cands=64, repeats=True):
    """candidate spoken readings of the whole span (prefix + core variants + suffix), current reading first. Commas never appear."""
    cur = sp.reading
    pre = " ".join(sp.prefix); suf = " ".join(sp.suffix)
    core_vars = []
    if sp.kind == "decade":
        return [cur]
    if sp.kind == "ordinal":
        core_vars = [" ".join(sp.core)]
        n = int(sp.value)
        core_vars.append(ordinal(n))
    elif sp.kind == "decimal":
        for v in sp.values[:4]:
            ip, fp = v.split(".")
            ipw = ["zero", ""] if ip == "0" else (num_variants(ip, False)[:3] if ip else [""])
            for a in ipw:
                for z in ("oh", "zero"):
                    for dbl in (False,):
                        core_vars.append((a + " " if a else "") + "point " + digit_words(fp, z, dbl))
    else:
        for v in sp.values[:8]:
            scaled = sp.kind == "int" and any(w in SCALE or w == "hundred" for w in sp.core)
            for c in num_variants(v, with_pairs=not scaled, big=(sp.kind in ("int", "decimal") or len(v) <= 6)):
                core_vars.append(c)
    # scale words / and / a as in the current core are kept by the generated cardinals; prefix / suffix are fixed
    cands = [cur]
    core_cur = " ".join(sp.core)
    for c in core_vars:
        full = " ".join(x for x in (pre, c, suf) if x)
        full = re.sub(r"\s+", " ", full).strip()
        if full not in cands:
            cands.append(full)
    if sp.kind in ("digits", "pairs", "int") and len(sp.core) >= 2 and "point" not in sp.core:
        seg = segment([("one" if w == "a" else w) for w in sp.core])
        if seg and len(seg) >= 2:
            toks_ = [("one" if w == "a" else w) for w in sp.core]
            for gi, (a, b, val) in enumerate(seg):
                for alt in num_variants(val, with_pairs=False, big=(len(val) <= 6))[:6]:
                    new = toks_[:a] + alt.split() + toks_[b:]
                    full = re.sub(r"\s+", " ", " ".join(x for x in (pre, " ".join(new), suf) if x)).strip()
                    if full not in cands:
                        cands.append(full)
    if repeats and sp.kind == "digits":
        for c in repeat_variants(sp.core):
            full = re.sub(r"\s+", " ", " ".join(x for x in (pre, c, suf) if x)).strip()
            if full not in cands:
                cands.append(full)
    cands = _dedupe(cands, cur)
    # keep the top max_cands: current + digit-wise + standard cardinal first, then the rest (stable order)
    return cands[:max_cands]


def _ctc_key(c):
    out = []
    for t in c.split():
        u = re.sub(r"[^A-Z']", "", t.upper())
        if u == "OH":
            u = "O"
        if u:
            out.append(u)
    return " ".join(out)


def _dedupe(cands, cur):
    """drop candidates the CTC cannot tell apart from an earlier one (case, punctuation); non-current candidates take the capitalisation of the first token of the current reading"""
    cap = bool(cur[:1].isupper())
    seen, out = set(), []
    for k, c in enumerate(cands):
        if k and cap and c[:1].islower():
            c = c[:1].upper() + c[1:]
        key = _ctc_key(c)
        if key in seen:
            continue
        seen.add(key); out.append(c)
    return out


def apply_candidate(label, sp, cand):
    """replace the span's character range by cand; every comma that sat INSIDE the span disappears (cand has none); the punctuation after the span stays."""
    return label[:sp.char_start] + cand + label[sp.char_end:]


def strip_inner_commas(label, spans):
    """No commas inside number readings. Apply to every span right-to-left; returns the new label"""
    out = label
    for sp in sorted(spans, key=lambda s: -s.char_start):
        seg = out[sp.char_start:sp.char_end]
        if "," in seg:
            out = out[:sp.char_start] + re.sub(r"\s*,\s*", " ", seg) + out[sp.char_end:]
    return out


if __name__ == "__main__":
    for t in ["The last reading on the meter was forty four thousand two hundred. That seemed high.", "Call two three nine, two one eight before ten thirty AM.",
              "In twenty fifteen the P G thirteen cost eight hundred fifty million dollars, five point two percent, the thirty first.", "ring seven four six nine nine two two six eight eight straight away"]:
        print(t)
        for s in find_spans(t):
            print("   ", s.kind, s.weak, repr(s.text), s.value, s.values[:3], len(candidates(s)), candidates(s)[:4])


def generated(sp):
    """True when the generator reproduces the current reading with its own pieces: (a) the whole-span generator yields it, or (b) every group of its
    min-group segmentation is a known reading of that group's value (compositional: 'eight hundred' + digit words). Insertion of the current reading does not count."""
    norm = lambda x: re.sub(r"\s+", " ", x.lower()).strip()
    old = sp.reading
    sp.reading = "\0"
    try:
        c = candidates(sp, max_cands=10 ** 6, repeats=False)
    finally:
        sp.reading = old
    if norm(old) in {norm(x) for x in c[1:]}:
        return True
    if sp.kind in ("decade",):
        return True
    if sp.kind == "ordinal":
        return sp.value != "" and norm(" ".join(sp.core)) == norm(ordinal(int(sp.value)))
    seg = segment([("one" if w == "a" else w) for w in sp.core]) if "point" not in sp.core else None
    if not seg:
        return False
    toks_ = [("one" if w == "a" else w) for w in sp.core]
    for a, b, val in seg:
        g = " ".join(toks_[a:b])
        allowed = {norm(x) for x in num_variants(val, with_pairs=False)} | {norm(x) for x in num_variants(val, with_pairs=True)}
        if g not in allowed:
            return False
    return True
