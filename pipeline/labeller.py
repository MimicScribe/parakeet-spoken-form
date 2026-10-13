"""Real-speech English number relabel: caption number spans -> every plausible spoken reading (v10pt orthography).

The caption (Granary: Whisper text, Qwen punctuation) writes numbers as digits. The spoken-form model must be trained
on the words actually said, so each number span gets a candidate set and two independent acoustic scorers pick one
(see label_gpu.py). Orthography = readings.py (the v10pt contract): lowercase number words, tens and ones spaced, no
hyphens, "PM"/"AM", "o'clock", letters of codes kept as capitals.

A caption number is any token with a digit, plus spelled number words when the caption already wrote them (those rows
are the labeller's gold: the caption reading is hidden and must be recovered).

>>> [s.text for s in spans("We grew 15-20% in 2019, up from $5.5 million.")]
['15-20%', '2019', '$5.5 million']
>>> 'twenty nineteen' in spans("in 2019 we")[0].cands and 'two thousand nineteen' in spans("in 2019 we")[0].cands
True
>>> spans("in 2019 we")[0].cands[:2]
['twenty nineteen', 'two thousand nineteen']
>>> sorted(spans("It was $1.50.")[0].cands)[:3]
['a dollar and fifty cents', 'a dollar fifty', 'a dollar fifty cents']
>>> 'fifteen hundred' in spans("about 1,500 people")[0].cands
True
>>> spans("at 3:30 PM")[0].cands[:2]
['three thirty PM', 'half past three PM']
>>> spans("the 21st century")[0].cands
['twenty first']
>>> render("in 2019 we", spans("in 2019 we"), [1])
'in two thousand nineteen we'
>>> render("2019 was big", spans("2019 was big"), [0])
'Twenty nineteen was big'
>>> unsupported("Call http://x.com/2 now") , unsupported("World War II") , unsupported("in 2019 we")
(True, True, False)
>>> spans("COVID-19 hit")[0].cands
['COVID nineteen', 'COVID one nine']
>>> [(s.gold, s.cands[:2]) for s in spans("about nineteen ninety eight and twenty five percent")]
[('nineteen ninety eight', ['nineteen ninety eight', 'one thousand nine hundred ninety eight']), ('twenty five percent', ['twenty five percent'])]
>>> spelled_value("two thousand and five"), spelled_value("fifteen hundred"), spelled_value("twenty oh five")
((2005, 'int'), (1500, 'int'), (2005, 'int'))
>>> spelled_value("three point one four"), spelled_value("a hundred and twenty")
(('3.14', 'decimal'), (120, 'int'))
"""

import itertools
import re
from dataclasses import dataclass, field

from readings import ONES, cardinal, digits, ordinal, pair

SCALE_W = r"(?:thousand|million|billion|trillion)"
CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence")}
UNIT_WORDS = {"km": ["kilometers", "K"], "kg": ["kilograms", "kilos", "K G"], "mph": ["miles per hour", "M P H"],
              "gb": ["gigabytes", "gigs", "G B"], "mb": ["megabytes", "megs", "M B"], "tb": ["terabytes", "T B"],
              "ghz": ["gigahertz"], "mhz": ["megahertz"], "hz": ["hertz"], "cm": ["centimeters"], "mm": ["millimeters"],
              "ms": ["milliseconds"], "kw": ["kilowatts"], "kwh": ["kilowatt hours"], "lbs": ["pounds"], "lb": ["pounds"],
              "ft": ["feet"], "x": ["x", "times"], "k": ["K", "thousand", "grand"], "m": ["million", "M"],
              "bn": ["billion"], "b": ["billion", "B"], "g": ["G"], "p": ["P"], "d": ["D"]}
NUMBER_WORD_RE = re.compile(
    r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|trillion|oh|and|a|point)(?:[\s-]+|\b))+(?:percent\b)?", re.I)
NUMWORD = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                     r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                     r"hundred|thousand|million|billion|trillion)\b", re.I)

# One caption number token. Order matters: the widest shapes first.
TOKEN = re.compile(
    r"(?P<cur>[$€£])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:(?P<range>\s?(?:-|–|to)\s?)(?P<cur2>[$€£])?(?P<num2>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?))?"
    r"(?P<suf>%|st\b|nd\b|rd\b|th\b|'s\b|s\b|(?:km|kg|mph|gb|mb|tb|ghz|mhz|hz|cm|mm|ms|kwh|kw|lbs|lb|ft|bn|x|k|m|b|g|p|d)\b)?"
    r"(?:\s(?P<scale>" + SCALE_W + r"))?"
    r"(?:\s?(?P<pct2>percent\b))?", re.I)
CLOCK = re.compile(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})(?:\s?(?P<ap>[ap]\.?m\.?)(?=\W|$))?|\b(?P<h2>\d{1,2})\s?(?P<ap2>[ap]\.m\.|[ap]m\b)",
                   re.I)
SLASH = re.compile(r"\b(?P<a>\d+)/(?P<b>\d+)\b")
RATIO = re.compile(r"\b(?P<a>\d+):(?P<b>\d+)\b(?!\s?[ap]\.?m)")
DECADE_APOS = re.compile(r"(?<!\w)['’](?P<d>\d0)s\b")
ALNUM = re.compile(r"\b(?P<pre>[A-Za-z]+)-?(?P<num>\d+)(?P<post>[A-Za-z]*)\b(?![.,]\d)")
PHONE = re.compile(r"(?:\+?1[-.\s])?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
BAD = re.compile(r"https?://|www\.|@\w|\.com\b|\b(?:II|III|IV|VI|VII|VIII|IX|XI|XII|XIV|XV|XX)\b|[#^*_=<>\[\]{}|\\~]")


@dataclass
class Span:
    start: int
    end: int
    text: str
    cands: list[str]
    kind: str = "num"
    gold: str = ""  # the caption's own reading, for spelled-out spans
    meta: dict = field(default_factory=dict)


def _uniq(xs):
    out = []
    for x in xs:
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in out:
            out.append(x)
    return out


def int_readings(n: int, raw: str = "", allow_year: bool = True) -> list[str]:
    """Every common way to say the integer n (most likely first)."""
    out = []
    if allow_year and 1100 <= n <= 2099 and not raw.startswith("0") and "," not in raw:
        if 2000 <= n <= 2009:
            out += [cardinal(n), cardinal(n, use_and=True), f"twenty oh {ONES[n - 2000]}" if n > 2000 else "twenty hundred"]
        elif n % 100 == 0:
            out += [pair(n), cardinal(n)]
        else:
            out += [pair(n), cardinal(n), cardinal(n, use_and=True)]
            if n % 100 < 10:
                out += [f"{cardinal(n // 100)} {ONES[n % 100]}"]
    out += [cardinal(n), cardinal(n, use_and=True)]
    if 1000 < n < 10000 and n % 100 == 0 and n % 1000:
        out.append(pair(n))  # "fifteen hundred"
    if 1000 < n < 10000 and n % 100 and not allow_year:
        out.append(pair(n))
    words = cardinal(n)
    for s in ("one hundred", "one thousand", "one million", "one billion"):
        if words.startswith(s):
            out += ["a" + w[3:] for w in (cardinal(n), cardinal(n, use_and=True))]
    if 100 <= n <= 999 and n % 100 >= 10:
        out.append(f"{ONES[n // 100]} {cardinal(n % 100)}")  # "three twenty"
    if len(raw) >= 3 and (raw.startswith("0") or n >= 100):
        out += [digits(raw), digits(raw, oh=True)]
    if raw.startswith("0") and len(raw) == 2:
        out = [f"oh {ONES[n]}", f"zero {ONES[n]}"] + out
    return _uniq(out)


def decimal_readings(raw: str) -> list[str]:
    whole, frac = raw.split(".")
    i = int(whole) if whole else 0
    tails = _uniq([f"point {digits(frac)}", f"point {digits(frac, oh=True)}"])
    out = []
    for t in tails:
        if i == 0:
            out += [f"zero {t}", t, f"oh {t}"]
        else:
            out += [f"{w} {t}" for w in int_readings(i, whole, allow_year=False)[:3]]
    if frac in ("5", "50") and i:
        out += [f"{w} and a half" for w in int_readings(i, whole, allow_year=False)[:2]]
    if frac in ("5", "50") and not i:
        out += ["a half", "half"]
    if frac == "25" and not i:
        out += ["a quarter"]
    if len(frac) == 2 and frac[0] != "0" and i:
        out.append(f"{cardinal(i)} point {cardinal(int(frac))}")
    return _uniq(out)


def number_readings(raw: str, allow_year=True) -> list[str]:
    raw = raw.replace(",", "")
    if "." in raw:
        return decimal_readings(raw)
    return int_readings(int(raw), raw, allow_year)


def money_readings(cur: str, amt: str, scale: str | None) -> list[str]:
    one, many, sub1, subs = CURRENCY[cur]
    amt = amt.replace(",", "")
    if scale:
        return _uniq([f"{a} {scale} {many}" for a in number_readings(amt, False)[:6]]
                     + [f"{a} {scale}" for a in number_readings(amt, False)[:3]])
    if "." in amt:
        whole, frac = amt.split(".")
        d, c = int(whole or 0), int((frac + "0")[:2])
        if len(frac) != 2:
            return _uniq([f"{a} {many}" for a in decimal_readings(amt)])
        unit = one if d == 1 else many
        dws = ["a", "one"] if d == 1 else int_readings(d, whole, allow_year=False)[:2]
        out = []
        for dw in dws:
            if c == 0:
                out += [f"{dw} {unit}"]
                continue
            cw = cardinal(c)
            out += [f"{dw} {unit} {cw}", f"{dw} {unit} and {cw} {sub1 if c == 1 else subs}",
                    f"{dw} {unit} {cw} {sub1 if c == 1 else subs}"]
        if c:
            out.append(f"{cardinal(d)} {cardinal(c)}")
        if d == 0:
            out = [f"{cardinal(c)} {sub1 if c == 1 else subs}"] + out
        return _uniq(out)
    d = int(amt)
    unit = one if d == 1 else many
    out = [f"{w} {unit}" for w in int_readings(d, amt, allow_year=False)[:5]]
    if d == 1:
        out.insert(0, f"a {one}")
    if cur == "$" and d > 1:
        out.append(f"{cardinal(d)} bucks")
    return _uniq(out)


def clock_readings(h: int, m: int, ap: str | None) -> list[str]:
    suf = f" {ap.replace('.', '').upper()}" if ap else ""
    hw = cardinal(h)
    if m == 0:
        out = [f"{hw}{suf}", f"{hw} o'clock", f"{hw} o'clock{suf}" if suf else "", f"{hw} hundred"]
    elif m < 10:
        out = [f"{hw} oh {ONES[m]}{suf}", f"{hw} zero {ONES[m]}{suf}"]
    else:
        out = [f"{hw} {cardinal(m)}{suf}"]
        if m == 30:
            out.append(f"half past {hw}{suf}")
        if m == 15:
            out.append(f"a quarter past {hw}{suf}")
            out.append(f"quarter past {hw}{suf}")
        if m == 45 and h < 12:
            out.append(f"a quarter to {cardinal(h + 1)}{suf}")
            out.append(f"quarter to {cardinal(h + 1)}{suf}")
    return _uniq(out)


_SMALL = {w: i for i, w in enumerate(ONES)}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_BIG = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}


def _groups(words: list[str]) -> list[int] | None:
    """Words of a number without scales -> its two-digit groups ("nineteen ninety eight" -> [19, 98])."""
    out, cur, after_oh = [], None, False
    for w in words:
        if after_oh and w in _SMALL and _SMALL[w] < 10:
            cur, after_oh = _SMALL[w], False
            continue
        after_oh = False
        if w in _TENS:
            if cur is not None:
                out.append(cur)
            cur = _TENS[w]
        elif w in _SMALL:
            if cur is not None and cur % 10 == 0 and cur >= 20 and _SMALL[w] < 10:
                cur += _SMALL[w]
            else:
                if cur is not None:
                    out.append(cur)
                cur = _SMALL[w]
        elif w == "oh" and cur is not None:
            out.append(cur)
            cur, after_oh = 0, True
        else:
            return None
    if cur is not None:
        out.append(cur)
    return out if len(out) <= 2 else None


def spelled_value(reading: str):
    """(value, kind) of a spelled number, or None. kind: int | decimal | percent."""
    w = [x for x in re.sub(r"[-,]", " ", reading.lower()).split()]
    kind = "int"
    if w and w[-1] == "percent":
        kind, w = "percent", w[:-1]
    if "point" in w:
        i = w.index("point")
        whole = spelled_value(" ".join(w[:i])) if w[:i] else (0, "int")
        frac = [_SMALL.get(x, 0 if x == "oh" else None) for x in w[i + 1:]]
        if whole is None or not frac or any(f is None or f > 9 for f in frac):
            return None
        v = f"{whole[0]}.{''.join(str(f) for f in frac)}"
        return (v, "percent" if kind == "percent" else "decimal")
    if not w:
        return None
    if w[0] == "a":
        w = ["one"] + w[1:]
    w = [x for x in w if x != "and"]
    total, cur = 0, 0
    seg: list[str] = []

    def flush_seg():
        nonlocal cur, seg
        if seg:
            g = _groups(seg)
            if g is None:
                return False
            v = 0
            for x in g:
                v = v * 100 + x if len(g) > 1 else x
            cur += v
            seg = []
        return True

    for x in w:
        if x == "hundred":
            if not flush_seg():
                return None
            cur = (cur or 1) * 100
        elif x in _BIG:
            if not flush_seg():
                return None
            total += (cur or 1) * _BIG[x]
            cur = 0
        else:
            seg.append(x)
    if not flush_seg():
        return None
    v = total + cur
    return (v, kind)


def spelled_cands(gold: str) -> list[str]:
    """Candidates regenerated from the VALUE of a caption's spelled number (the caption reading is not given)."""
    sv = spelled_value(gold)
    if sv is None:
        return [gold]
    v, kind = sv
    if kind == "decimal":
        return decimal_readings(v)
    if kind == "percent":
        base = decimal_readings(v) if isinstance(v, str) else int_readings(v, str(v), allow_year=False)
        return _uniq([f"{b} percent" for b in base])
    return int_readings(v, str(v), allow_year=1100 <= v <= 2099)


def _token_cands(mt) -> tuple[list[str], str]:
    g = mt.groupdict()
    num, suf, scale, cur = g["num"], (g["suf"] or "").lower(), g["scale"], g["cur"]
    pct = suf == "%" or bool(g["pct2"])
    if g["num2"]:
        a = number_readings(num, allow_year=True)[:4]
        b = number_readings(g["num2"], allow_year=True)[:4]
        unit = " percent" if pct else (f" {scale}" if scale else "")
        out = []
        for x, y in itertools.product(a[:3], b[:3]):
            out += [f"{x} to {y}{unit}", f"{x} {y}{unit}"]
        if cur:
            many = CURRENCY[cur][1]
            out = [o + f" {many}" for o in out] + out
        out += [f"between {a[0]} and {b[0]}{unit}", f"{a[0]} through {b[0]}{unit}", f"{a[0]} or {b[0]}{unit}"]
        return _uniq(out), "range"
    if cur:
        return money_readings(cur, num, scale), "money"
    if pct:
        return _uniq([f"{r} percent" for r in number_readings(num, allow_year=False)[:6]]), "percent"
    if "." in num and suf in ("st", "nd", "rd", "th", "s", "'s"):
        return [], "bad"
    if suf in ("st", "nd", "rd", "th"):
        n = int(num.replace(",", ""))
        return [ordinal(n)], "ordinal"
    if suf in ("s", "'s"):
        n = int(num.replace(",", ""))
        if n % 10 == 0 and 1100 <= n <= 2090:
            base = pair(n) if n % 100 else cardinal(n // 100) + " hundreds"
            plural = re.sub(r"y$", "ies", base) if not base.endswith("hundreds") else base
            alts = [plural]
            if n % 100 == 0 and n >= 2000:
                alts = [cardinal(n // 1000) + " thousands", "twenty hundreds"]
            return _uniq(alts), "decade"
        if n % 10 == 0 and n < 100:
            return [re.sub(r"y$", "ies", cardinal(n))], "decade"
        return [], "bad"
    if suf and suf in UNIT_WORDS:
        base = number_readings(num, allow_year=False)[:4]
        out = [f"{b} {u}" for b in base for u in UNIT_WORDS[suf]]
        return _uniq(out), "unit"
    if scale:
        return _uniq([f"{r} {scale}" for r in number_readings(num, allow_year=False)[:6]]), "scaled"
    raw = num.replace(",", "")
    if "." in raw:
        # Version-like "2.0" / "3.11": also "two point oh".
        return decimal_readings(raw), "decimal"
    n = int(raw)
    kind = "year" if 1900 <= n <= 2099 and "," not in num else "int"
    return number_readings(num, allow_year=kind == "year" or (1100 <= n <= 2099 and "," not in num)), kind


def unsupported(text: str) -> bool:
    return bool(BAD.search(text))


def spans(text: str) -> list[Span]:
    """Number spans of a caption, left to right, each with its candidate readings. Overlaps resolved by order of the
    passes below (the more specific shape wins)."""
    taken = [False] * len(text)
    out: list[Span] = []

    def add(a, b, cands, kind, gold=""):
        if a >= b or any(taken[a:b]) or not cands:
            return
        for i in range(a, b):
            taken[i] = True
        out.append(Span(a, b, text[a:b], cands, kind, gold))

    for mt in PHONE.finditer(text):
        ds = re.sub(r"\D", "", mt.group(0))
        groups = [ds[:-10], ds[-10:-7], ds[-7:-4], ds[-4:]] if len(ds) > 10 else [ds[:3], ds[3:6], ds[6:]]
        groups = [x for x in groups if x]
        out_c = _uniq([" ".join(digits(x, oh) for x in groups) for oh in (False, True)]
                      + [", ".join(digits(x, oh) for x in groups) for oh in (False, True)])
        add(mt.start(), mt.end(), out_c, "phone")
    for mt in CLOCK.finditer(text):
        if mt.group("h"):
            h, m, ap = int(mt.group("h")), int(mt.group("m")), mt.group("ap")
            if h > 23 or m > 59:
                continue
            add(mt.start(), mt.end(), clock_readings(h, m, ap), "clock")
        else:
            h = int(mt.group("h2"))
            if 1 <= h <= 12:
                add(mt.start(), mt.end(), clock_readings(h, 0, mt.group("ap2"))[:1] +
                    [f"{cardinal(h)} o'clock {mt.group('ap2').replace('.', '').upper()}"], "clock")
    for mt in RATIO.finditer(text):
        a, b = int(mt.group("a")), int(mt.group("b"))
        add(mt.start(), mt.end(), _uniq([f"{cardinal(a)} to {cardinal(b)}", f"{cardinal(a)} on {cardinal(b)}",
                                         f"{cardinal(a)} {cardinal(b)}"]), "ratio")
    for mt in SLASH.finditer(text):
        a, b = int(mt.group("a")), int(mt.group("b"))
        c = [f"{cardinal(a)} {cardinal(b)}", f"{cardinal(a)} over {cardinal(b)}", f"{cardinal(a)} slash {cardinal(b)}"]
        if a < b <= 10 and b in (2, 3, 4, 5, 8, 10):
            nm = {2: ("half", "halves"), 3: ("third", "thirds"), 4: ("quarter", "quarters"), 5: ("fifth", "fifths"),
                  8: ("eighth", "eighths"), 10: ("tenth", "tenths")}[b]
            c = ([f"one {nm[0]}", f"a {nm[0]}"] if a == 1 else [f"{cardinal(a)} {nm[1]}"]) + c
        add(mt.start(), mt.end(), _uniq(c), "slash")
    for mt in DECADE_APOS.finditer(text):
        add(mt.start(), mt.end(), [re.sub(r"y$", "ies", cardinal(int(mt.group("d"))))], "decade")
    for mt in ALNUM.finditer(text):
        pre, num, post = mt.group("pre"), mt.group("num"), mt.group("post")
        if pre.lower() in ("am", "pm") or (not post and pre.lower() in UNIT_WORDS):
            continue
        hy = mt.start() + len(pre)
        if text[hy] == "-" and not pre.isupper() and len(num) == 4 and (not post or post == "s"):
            tm = TOKEN.match(text, hy + 1)  # "mid-2015", "Pre-2020", "mid-1960s": an ordinary word + a year
            if tm and tm.end() == mt.end():
                cands, kind = _token_cands(tm)
                if kind != "bad":
                    add(hy + 1, tm.end(), cands, kind)
                    continue
        if (pre[0].islower() and len(pre) > 2) or any(ch.islower() for ch in post):
            # "mid-2015", "mid-1960s", "query2vec", "2Learn"-style words: no defined code reading (retrain-3 review)
            add(mt.start(), mt.end(), ["\x00"], "bad")
            continue
        n = int(num)
        nums = _uniq(([cardinal(n)] if n < 100 or n % 100 == 0 else [pair(n) if n >= 1000 else cardinal(n)])
                     + [digits(num), digits(num, oh=True)] + int_readings(n, num, allow_year=False)[:2])
        if num == "2" and post:
            nums = _uniq(nums + ["to"])  # "B2B", "D2C": said "B to B" (retrain-3 review)
        pre_w = pre.upper() if pre.isupper() or len(pre) <= 2 else pre
        post_w = (" " + post.upper()) if post else ""
        add(mt.start(), mt.end(), _uniq([f"{pre_w} {x}{post_w}" for x in nums]), "code")
    for mt in TOKEN.finditer(text):
        a = mt.start()
        if a > 0 and (text[a - 1].isalnum() or text[a - 1] in "/.:" or
                      (text[a - 1] == "-" and a > 1 and text[a - 2] != " ")):
            continue
        nxt = text[mt.end()] if mt.end() < len(text) else " "
        if nxt.isalpha() or nxt in "°º":
            add(mt.start(), mt.end(), ["\x00"], "bad")  # "3DS", "5V", "4ac", "84°C": glued, no defined reading
            continue
        cands, kind = _token_cands(mt)
        if a > 0 and text[a - 1] == "-" and (a == 1 or text[a - 2] == " ") and kind in ("int", "decimal", "percent"):
            cands = [f"{p} {c}" for p in ("minus", "negative") for c in cands[:3]]  # "-5 degrees"
            add(a - 1, mt.end(), cands, "negative")
            continue
        if kind == "bad":
            add(mt.start(), mt.end(), ["\x00"], "bad")
            continue
        add(mt.start(), mt.end(), cands, kind)
    for mt in NUMBER_WORD_RE.finditer(text):
        seg = mt.group(0).rstrip(" -")
        a, b = mt.start(), mt.start() + len(seg)
        words = seg.split()
        while words and words[0].lower() in ("and", "a", "oh", "point"):
            a += len(words[0]) + 1
            words = words[1:]
        while words and words[-1].lower() in ("and", "a", "oh", "point"):
            words = words[:-1]
        seg = " ".join(words)
        b = a + len(seg) if seg else a
        if not seg or not NUMWORD.search(seg) or seg.lower() in ("one",):
            continue
        reading = re.sub(r"-", " ", seg).lower()
        pieces = [reading]
        if spelled_value(reading) is None and " and " in reading:
            pieces = reading.split(" and ")
        pos = a
        for pc in pieces:
            k = text.lower().find(pc.split()[0], pos)
            ln = len(pc)
            cands = spelled_cands(pc)
            if pc not in cands:
                cands = cands + [pc]  # coverage miss: kept so the row can still be labelled; counted in reports
            add(k, k + ln, cands, "spelled", gold=pc)
            pos = k + ln
    out.sort(key=lambda s: s.start)
    return out


def render(text: str, sp: list[Span], choice: list[int]) -> str:
    """The caption with each span replaced by its chosen candidate (sentence-initial capital kept). A hyphen joining
    the number to a following word ("20-foot", "twenty-second") becomes a space: no hyphens in number words."""
    return render_ranges(text, sp, choice)[0]


def render_ranges(text: str, sp: list[Span], choice: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """render() plus each span's [start, end) in the RENDERED label. Callers map label tokens to spans with these
    ranges; recomputing them from the candidate lengths drifts by 2 after a stripped "a " (retrain-3 review: every
    span after "a hundred" marked the wrong words)."""
    parts, pos, ranges = [], 0, []
    for s, c in zip(sp, choice):
        gap = text[pos:s.start]
        if gap.endswith("-") and len(gap) >= 2 and gap[-2].isalpha():
            gap = gap[:-1] + " "  # "mid-2015" -> "mid twenty fifteen"
        parts.append(gap)
        w = s.cands[c]
        prev = "".join(parts).rstrip().split(" ")[-1].lower() if "".join(parts).strip() else ""
        if prev in ("a", "an") and w.startswith("a "):
            w = w[2:]  # the caption already has the "a" of "a hundred" (Opus review: "a a hundred")
        before = "".join(parts).rstrip()
        if not before or before[-1] in ".?!":
            w = w[0].upper() + w[1:]
        start = len("".join(parts))
        parts.append(w)
        ranges.append((start, start + len(w)))
        pos = s.end
        if pos < len(text) - 1 and text[pos] == "-" and text[pos + 1].isalpha():
            parts.append(" ")
            pos += 1
    parts.append(text[pos:])
    return "".join(parts), ranges


WRITTEN = re.compile(r"[0-9$%€£/@&#]")


def row_ok(text: str, sp: list[Span]) -> str:
    """'' when the row can be labelled, else the reason it is dropped."""
    if unsupported(text):
        return "unsupported"
    if any(s.kind == "bad" for s in sp):
        return "bad_token"
    probe = render(text, sp, [0] * len(sp))
    if WRITTEN.search(probe):
        return "leftover_written"
    return ""
