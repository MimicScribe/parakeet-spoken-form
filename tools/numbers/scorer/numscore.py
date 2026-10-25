"""Form-blind number scorer v2.2 (2026-10-24; v2 of 2026-10-23 is frozen in numscore_v2_frozen.py).
v2.2 (blind spots found by independent review; each fix can be switched off via FIX[...] to measure it alone):
  FIX["slot"]    slots of a " | " reference never share token ids, so "8 | 9" can no longer match the hyp "89";
  FIX["hyphen"]  hyphen chains ("2-3", "1932-33") join across the hyphen only if >= 3 segments or >= 7 digits
                 (phones, ISBNs, IDs); short 2-segment chains are ranges/scores and match part by part;
  FIX["mant"]    the mantissa rule ("7.5" matches "seven point five million") applies to slot references only;
  FIX["filler"]  fillers (um/uh/er...) are no longer deleted: like a comma they end the current word run, so
                 "twenty, uh, two" is 20 2 (not 22) while "five five um five one two" still joins as digits.
  New metrics (verdict() itself is unchanged): insertions(), nonnum_wer(), form_report() / form_summary().
v2.1 changes: (1) segmentation cap 64 -> 4096, longest chunks enumerated first, alternative-product cap 256 -> 4096,
  cap hits counted in CAP_HITS; (2) ok_point needs a spoken/written point (point/dot/digit.digit) or money word/$ in the
  hypothesis ("two five" for 2.5, "three fourteen" for 3.14 are no longer ok_point); (3) ok_point point-word must be
  between numbers (plural "points" no longer counts), so "two points, three points, four points" != 3.4.
  (4) a bare price ("five fifty" for $5.50) stays ok_point when the REFERENCE carries money context; "one.6" counts as a point.
Does a hypothesis carry every number VALUE of the reference, in order?

Both sides become "atoms" (one number each). Reference atoms must all be matched, in order, by hypothesis atoms;
extra hypothesis atoms are allowed (insertions are scored elsewhere). A match is:
  - one ref atom == one hyp atom by value (1.50 == 1.5, 23,000 == twenty three thousand), or
  - one ref atom == the digit concatenation of several consecutive UNSCALED hyp atoms
    ("sixty two eight forty one" = 62841, "twenty twenty" = 2020, "nineteen sixty six" = 1966,
     "one nine two one six eight two hundred forty six" = 192.168.246), or
  - several ref atoms from ONE ref token (0-521-64696-0, 10:30, 2019-2020) == a hyp group by concatenation.
Word runs are parsed into every valid segmentation (so "two thousand twenty two thousand twenty one" = 2020 2021),
except splits that change meaning: tens|unit ("twenty two" is never 20 2 unless the unit starts a scaled group),
hundred|rest ("two hundred forty six" is never 200 46).
NOT accepted: cardinal readings of digit strings ("nine trillion ..." for an ISBN), dropped or extra digits inside a
run, decimal points moved ("twenty five" for 2.5), sound-alikes in strict mode.

Verdict ladder (first that passes): ok, ok_point (decimal point not spoken, parts aligned: "five fifty" = 5.50),
ok_glue (right after splitting glued tokens: "nineam", "ISBNO", "TenS"), ok_repeat (hyp drops an exact adjacent
repeat of a ref number), sa (right after to/for/won/ate -> two/four/one/eight), bad.
"""
import re, unicodedata
from decimal import Decimal, InvalidOperation

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
UNITS.update({"oh": 0, "o": 0, "nought": 0, "naught": 0})
TEENS = {w: 10 + i for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"thousand": 10**3, "k": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
ORD_S = {"hundredth": 100, "thousandth": 1000, "millionth": 10**6}
ORD_U = {"first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
ORD_T = {"tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
         "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
         "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}
DECADES = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90}
MISSPELL = {"fourty": "forty", "ninty": "ninety", "nineth": "ninth", "twelth": "twelfth", "forteen": "fourteen",
            "fourtieth": "fortieth", "points": "point", "hundered": "hundred", "thousands": None, "zeroes": None}
FILLERS = {"um", "uh", "uhm", "umm", "uhh", "er", "erm", "ah", "eh", "hmm", "mm", "mhm", "huh"}
FRAC_D = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourths": 4, "fifths": 5,
          "sixths": 6, "sevenths": 7, "eighths": 8, "ninths": 9, "tenths": 10}
SA = {"to": "two", "too": "two", "for": "four", "won": "one", "ate": "eight"}
NUMWORD = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES) | {"hundred"} | set(ORD_U) | set(ORD_T) | set(ORD_S) | set(DECADES)
NUMWORD.discard("k")


class Atom:
    __slots__ = ("s", "scaled", "tok", "nojoin", "mant", "form", "kind")

    def __init__(self, s, scaled=False, tok=None, nojoin=False, mant=None):
        self.s, self.scaled, self.tok, self.nojoin, self.mant = s, scaled, tok, nojoin, mant
        self.form, self.kind = None, None  # form: "d" digits / "w" words (hyp side); kind: "long" / "code" (ref side)

    def __repr__(self): return self.s + ("^" if self.scaled else "")

    def joinable(self): return not self.scaled and "." not in self.s and not self.s.startswith("f")


FIX = {"slot": True, "hyphen": True, "mant": True, "filler": True}
_CTX = {"slot_mode": True}  # set by verdict(): is the reference a " | " slot list?


def dval(s):
    try: return Decimal(s)
    except (InvalidOperation, ValueError): return None


def canon(v):
    if isinstance(v, Decimal):
        v = v.normalize()
        if v == v.to_integral(): return str(int(v))
        return format(v, "f")
    return str(v)


# ---------------------------------------------------------------- text normalisation
def _pre(t, glue=False):
    t = unicodedata.normalize("NFKC", t or "").replace("’", "'").replace("‘", "'")
    t = t.replace("½", " 1/2 ").replace("¼", " 1/4 ").replace("¾", " 3/4 ")
    if glue:  # split CamelCase / CAPS glue: "TenS" -> "Ten S", "ISBNO" -> "ISBN O", "nineam" -> "nine am"
        t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
        t = re.sub(r"\b([A-Z]{2,})(O)\b", r"\1 \2", t)
        def g(m):
            w = m.group(0); lw = w.lower()
            if lw in NUMWORD or lw in FILLERS: return w
            for n in sorted(NUMWORD, key=len, reverse=True):
                if lw.startswith(n) and 0 < len(lw) - len(n) <= 3 and lw[len(n):] not in ("s", "es", "th", "ths", "ty", "teen"): return w[:len(n)] + " " + w[len(n):]
                if lw.endswith(n) and len(n) >= 3 and 0 < len(lw) - len(n) <= 4: return w[:-len(n)] + " " + w[-len(n):]
            return w
        t = re.sub(r"[A-Za-z]+", g, t)
    t = t.replace("??", " - ").replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"(?<=\d)\s*%", " ", t).replace("$", " ").replace("£", " ").replace("€", " ")
    t = re.sub(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)", " ", t)  # 3D -> 3 D, M24KB -> M 24 KB, 9am -> 9 am
    t = re.sub(r"\b(\d+) (st|nd|rd|th)s?\b", r"\1", t)  # 22nd, 4th
    t = re.sub(r"\b(\d+) s\b", r"\1", t)  # 1980s, 80s
    t = re.sub(r"'(\d\d)\b", r"\1", t)  # '90s
    t = re.sub(r"\b(point|dot) (\d+)\b", r"0.\2", t, flags=re.I)  # "point 1"
    t = re.sub(r"\b(\d+(?:\.\d+)?) (m|bn|b)\b(?=\s|$)", lambda m: m.group(1) + {"m": " million", "bn": " billion", "b": " billion"}[m.group(2)], t)
    # Roman numerals (uppercase, 2+ letters or after a capitalised word): World War II, Henry VIII, Part IX
    ROM = {"II": 2, "III": 3, "IV": 4, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20}
    t = re.sub(r"\b(" + "|".join(sorted(ROM, key=len, reverse=True)) + r")(?=\b|[.,;:])", lambda m: " " + str(ROM[m.group(1)]) + " ", t)
    low = t.lower()
    low = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", low)  # twenty-four = twenty four
    low = re.sub(r"\bper cent\b", " ", low)
    low = re.sub(r"\bo'clock\b", " ", low)
    low = re.sub(r"\b(noon|midday|midnight)\b", " twelve ", low)
    # clock phrases -> "H MM"
    hw = "|".join(list(UNITS)[1:10] + list(TEENS)[:3])
    low = re.sub(rf"\bhalf past ({hw})\b", r"\1 thirty", low)
    low = re.sub(rf"\b(a )?quarter past ({hw})\b", r"\2 fifteen", low)
    def qto(m):
        h = {**{k: v for k, v in UNITS.items() if v}, **TEENS}[m.group(2)]
        h = 12 if h == 1 else h - 1
        return f"{h} 45"
    low = re.sub(rf"\b(a )?quarter to ({hw})\b", qto, low)
    # shared scale in ranges: "ten to twenty thousand" = 10,000 to 20,000
    nw = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(set(UNITS) | set(TEENS) | set(TENS), key=len, reverse=True)) + ")"
    numw = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES) | {"hundred", "and", "point"}
    def shs(m):
        before = low[:m.start()].split()[-1:] ; after = low[m.end():].split()[:1]
        if (before and (before[0] in numw or before[0].isdigit())) or (after and (after[0] in numw or after[0].isdigit())): return m.group(0)
        return f"{m.group(1)} {m.group(4)} {m.group(2)} {m.group(3)} {m.group(4)}"
    low = re.sub(rf"\b({nw}) (to|or|-) ({nw}(?: {nw})?) (thousand|million|billion|trillion)\b", shs, low)
    return low


def _words(low):
    """tokens: ('d', digit-token) | ('w', word) | ('f', filler) | ('|', break). With FIX["filler"] a filler acts like a comma."""
    out = []
    for m in re.finditer(r"\d+(?:[.,:/]\d+)*|[a-z]+(?:'[a-z]+)?|[-]|[^\sa-z\d]", low):
        x = m.group(0)
        if x[0].isdigit(): out.append(("d", x))
        elif x == "-": out.append(("-", x))
        elif x[0].isalpha():
            x = MISSPELL.get(x, x)
            if x is None: out.append(("|", "")); continue
            if x in FILLERS:
                if FIX["filler"]: out.append(("f", x))
                continue
            out.append(("w", x))
        elif x in ".,;:!?": out.append(("p", x))
        else: out.append(("|", x))
    return out


# ---------------------------------------------------------------- digit tokens
def _digit_atoms(x, tok, nxt_scale=None):
    """'1,500,000' -> [1500000^]; '7.3' -> [7.3]; '0-521' handled by caller; '192.168.0.1' -> 4 atoms; '10:30' -> 2; '1/2' -> frac"""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", x):
        v = Decimal(x.replace(",", ""))
        if nxt_scale: v *= nxt_scale
        return [[Atom(canon(v), True, tok)]]
    if re.fullmatch(r"\d+\.\d+", x):
        v = Decimal(x)
        if nxt_scale: return [[Atom(canon(v * nxt_scale), True, tok, mant=canon(v))]]
        a, b = x.split(".")
        return [[Atom(canon(v), False, tok)], "POINT", [Atom(a, False, tok, nojoin=True), Atom(b, False, tok)]]
    if re.fullmatch(r"\d+/\d+", x):
        n, d = x.split("/")
        return [[Atom(f"f{int(n)}/{int(d)}", False, tok)], [Atom(n, False, tok), Atom(d, False, tok)]]
    if ":" in x and re.fullmatch(r"\d+:\d\d", x):
        h, mm = x.split(":")
        alts = [[Atom(h, False, tok), Atom(mm, False, tok)]]
        if mm == "00": alts.append([Atom(h, False, tok)])
        return alts
    parts = re.split(r"[.,:/]", x)
    if nxt_scale and len(parts) == 1: return [[Atom(canon(Decimal(x) * nxt_scale), True, tok, mant=x)]]
    return [[Atom(p, False, tok) for p in parts]]


# ---------------------------------------------------------------- word-run grammar
def _chunk_ends(ws, i):
    """all (j, Atom) such that ws[i:j] is one valid number chunk"""
    res = []
    n = len(ws)
    w0 = ws[i]
    # double/triple X
    if w0 in ("double", "triple") and i + 1 < n and ws[i + 1] in UNITS:
        res.append((i + 2, Atom(str(UNITS[ws[i + 1]]) * (2 if w0 == "double" else 3))))
        return res
    if w0 in DECADES: return [(i + 1, Atom(str(DECADES[w0])))]
    if w0 in ORD_U: return [(i + 1, Atom(str(ORD_U[w0]), True))]
    if w0 in ORD_T: return [(i + 1, Atom(str(ORD_T[w0]), True))]
    if w0 == "second": return [(i + 1, Atom("2", True))]
    # cardinal state machine
    total, cur, last, last_scale, j, hund = 0, 0, None, 10**18, i, False
    zero_lead = False
    while j < n:
        w = ws[j]
        if w == "<f>":
            if last is None or last == "tens": break
            j += 1; continue
        if w in ("a", "an") and last is None and j + 1 < n and (ws[j + 1] == "hundred" or ws[j + 1] in SCALES):
            cur, last = 1, "unit"
        elif w in UNITS:
            if UNITS[w] == 0:
                if last is not None: break
                cur, last, zero_lead = 0, "zero", True
                j += 1; res.append((j, Atom("0"))); break
            if last in (None, "hundred", "scale", "and") or (last == "tens"):
                if last == "tens" or last in ("hundred", "scale", "and") or last is None: cur += UNITS[w]; last = "unit"
            else: break
        elif w in TEENS:
            if last in (None, "hundred", "scale", "and"): cur += TEENS[w]; last = "teen"
            else: break
        elif w in TENS:
            if last in (None, "hundred", "scale", "and"): cur += TENS[w]; last = "tens"
            else: break
        elif w == "hundred":
            if last in ("unit", "teen", "tens") and not hund and 0 < cur < 100: cur *= 100; last = "hundred"; hund = True
            else: break
        elif w in SCALES:
            s = SCALES[w]
            if last in ("unit", "teen", "tens", "hundred") and cur > 0 and s < last_scale:
                total += cur * s; cur = 0; last = "scale"; last_scale = s; hund = False
            else: break
        elif w == "and":
            if last in ("hundred", "scale") and j + 1 < n and (ws[j + 1] in UNITS or ws[j + 1] in TEENS or ws[j + 1] in TENS or ws[j + 1] == "a"):
                if ws[j + 1] == "a":  # "one and a half"
                    if j + 2 < n and ws[j + 2] == "half": pass
                    else: break
                last = "and"
            else: break
        elif w in ORD_S and last in ("unit", "teen", "tens", "hundred") and cur > 0:
            j += 1; res.append((j, Atom(canon(Decimal(total + cur * ORD_S[w])), True))); break
        elif (w in ORD_U or w in ORD_T) and last in ("hundred", "scale", "and"):
            cur += ORD_U.get(w) or ORD_T[w]; j += 1
            res.append((j, Atom(canon(Decimal(total + cur)), True))); break
        elif w in ORD_U and last == "tens":
            cur += ORD_U[w]; last = "ord"; j += 1
            res.append((j, Atom(canon(Decimal(total + cur)), total > 0 and last_scale < 10**18)))
            break
        elif w == "second" and last == "tens":
            cur += 2; last = "ord"; j += 1
            res.append((j, Atom(canon(Decimal(total + cur)), last_scale < 10**18))); break
        else: break
        j += 1
        if last in ("unit", "teen", "tens", "hundred", "scale"):
            v = total + cur
            mant = canon(Decimal(v) / last_scale) if last_scale < 10**18 else None
            res.append((j, Atom(str(v), last_scale < 10**18, mant=mant)))
    # "X and a half" / "one point five (million)" extensions on the longest chunk
    ext = []
    for (e, a) in list(res):
        if e < n and ws[e] in SCALES and dval(a.s) is not None and 1000 <= dval(a.s) < SCALES[ws[e]]:
            ext.append((e + 1, Atom(canon(dval(a.s) * SCALES[ws[e]]), True, mant=a.s)))
    for (e, a) in res:
        if e + 2 < n + 1 and ws[e:e + 3] == ["and", "a", "half"]:
            ext.append((e + 3, Atom(canon(Decimal(a.s) + Decimal("0.5")), a.scaled)))
        if e < n and ws[e] in ("point", "dot"):
            k, frac = e + 1, ""
            while k < n and ws[k] in UNITS: frac += str(UNITS[ws[k]]); k += 1
            if not frac and k < n and (ws[k] in TEENS or ws[k] in TENS):  # point twenty five
                v = TEENS.get(ws[k]) or TENS[ws[k]]; k += 1
                if v >= 20 and k < n and ws[k] in UNITS and UNITS[ws[k]]: v += UNITS[ws[k]]; k += 1
                frac = str(v)
            if frac and not a.scaled:
                d = Decimal(a.s + "." + frac)
                ext.append((k, Atom(canon(d) if not frac.endswith("0") else a.s + "." + frac)))
                if k < n and ws[k] in SCALES: ext.append((k + 1, Atom(canon(d * SCALES[ws[k]]), True, mant=canon(d))))
    if w0 in ("point", "dot") and i + 1 < n and ws[i + 1] in UNITS:
        k, frac = i + 1, ""
        while k < n and ws[k] in UNITS: frac += str(UNITS[ws[k]]); k += 1
        ext.append((k, Atom(canon(Decimal("0." + frac)))))
    # fractions: "one half", "a quarter", "three quarters", bare "half"
    if w0 in ("a", "one", "an") and i + 1 < n and ws[i + 1] in FRAC_D: ext.append((i + 2, Atom(f"f1/{FRAC_D[ws[i + 1]]}")))
    if w0 in UNITS and UNITS[w0] > 1 and i + 1 < n and ws[i + 1] in FRAC_D and ws[i + 1].endswith("s"):
        ext.append((i + 2, Atom(f"f{UNITS[w0]}/{FRAC_D[ws[i + 1]]}")))
    if w0 == "half": ext.append((i + 1, Atom("f1/2")))
    return res + ext


BAR = Atom("#", True)


def _split_ok(ws, j):
    """may a chunk end before ws[j]? (j = index of next word)"""
    if j <= 0 or j >= len(ws): return True
    p, q = ws[j - 1], ws[j]
    nxt = ws[j + 1] if j + 1 < len(ws) else ""
    if p in TENS and q in UNITS and UNITS[q] > 0 and nxt not in SCALES: return False
    if p == "hundred" and (q in UNITS and UNITS[q] > 0 or q in TEENS or q in TENS): return False
    if q in SCALES or q == "hundred": return False  # "one | thousand" never ends before its scale word
    return True


CAP_HITS = {"segs": 0, "product": 0}


def _segs(ws, cap=4096):
    """all segmentations of a word run into atoms (list of lists); words that start no chunk are dropped"""
    memo = {}

    def go(i):
        if i >= len(ws): return [[]]
        if i in memo: return memo[i]
        out = []
        if ws[i] == "<f>": return go(i + 1)  # filler between chunks: no atom, no barrier (digit strings continue)
        ends = _chunk_ends(ws, i)
        if not ends:  # a word inside the run that starts no number: barrier (no joins across it)
            out = [[BAR] + r for r in go(i + 1)]
        else:
            for (e, a) in sorted(ends, key=lambda t: -t[0]):  # longest chunk first
                if not _split_ok(ws, e) and e < len(ws) and any(x[0] > e for x in ends): continue
                for rest in go(e):
                    out.append([a] + rest)
                    if len(out) >= cap: CAP_HITS["segs"] += 1; break
                if len(out) >= cap: break
        memo[i] = out
        return out
    return go(0) or [[]]


_CODE_SUFFIX = {"am", "pm", "st", "nd", "rd", "th", "s", "k", "m", "b", "bn", "x"}


def _code_digits(text):
    """digit groups that sit inside a letter-digit code in the ORIGINAL text (M24KB, ESP8266, A-10, 3D, 4A);
    plain suffixes (9am, 22nd, 1980s, 5k) are not codes. Approximation: matched by digit string."""
    out = set()
    for tk in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", unicodedata.normalize("NFKC", text or "")):
        al, dg = re.findall(r"[A-Za-z]+", tk), re.findall(r"\d+", tk)
        if al and dg and not all(a.lower() in _CODE_SUFFIX and a == a.lower() or a in ("S", "K", "M", "B") for a in al):
            out.update(dg)
    return out


def atom_lattice(text, glue=False, tok0=0):
    """list of alternatives; each alternative = (atoms, point_mode) where point_mode marks the decimal-split variant"""
    low = _pre(text, glue)
    toks = _words(low)
    seq = []  # list of 'slots': each slot is list of alternative atom lists (with tag)
    i, tokid = 0, tok0
    run = []
    codes = _code_digits(text)

    def flush():
        nonlocal run
        while run and run[-1] == "<f>": run.pop()
        if run in (["oh"], ["o"]) and not near_digit: run = []  # interjection, not zero
        if run:
            segs = _segs(run)
            for sg in segs:
                for a in sg:
                    if a is not BAR: a.form = "w"
            seq.append([("w", s) for s in segs])
            run = []
    def nd(q):  # digit token at q or across a '-'
        return 0 <= q < len(toks) and (toks[q][0] == "d" or (toks[q][0] == "-" and 0 <= q + (1 if q > i else -1) < len(toks) and toks[q + (1 if q > i else -1)][0] == "d"))
    near_digit = False
    while i < len(toks):
        k, x = toks[i]
        if k == "w" and x in ("oh", "o"): near_digit = nd(i - 1) or nd(i + 1)
        if k == "w":
            if x in NUMWORD or x == "k" or x in ("a", "an", "and", "point", "dot", "double", "triple", "half", "second") or x in FRAC_D:
                run.append(x)
            elif x in ("dash", "hyphen", "slash"):
                flush()  # spoken separators: like punctuation, no barrier
            else:
                flush()
                if seq and seq[-1] != [("w", [BAR])]: seq.append([("w", [BAR])])
        elif k == "d":
            flush()
            prev_d = i > 0 and toks[i - 1][0] == "d"
            if not prev_d: tokid += 1
            if prev_d:  # "0041 44212 4404": one id only if the spaced group is long (phone/ID), else a new number
                g, q = 0, i - 1
                while q >= 0 and toks[q][0] == "d": q -= 1
                q2 = i
                while q2 < len(toks) and toks[q2][0] == "d": q2 += 1
                g = sum(len(re.sub(r"\D", "", toks[z][1])) for z in range(q + 1, q2))
                if g < 7: tokid += 1
            # join digit chain "0-521-64696-0" / "2019-2020" into one token id
            chain = [x]
            while i + 2 < len(toks) and toks[i + 1][0] == "-" and toks[i + 2][0] == "d":
                chain.append(toks[i + 2][1]); i += 2
            nxt_scale = None
            if len(chain) == 1 and i + 1 < len(toks) and toks[i + 1][0] == "w" and toks[i + 1][1] in SCALES:
                nxt_scale = SCALES[toks[i + 1][1]]; i += 1
            if len(chain) == 1:
                alts = _digit_atoms(x, tokid, nxt_scale)
                slot, pm = [], False
                kind = "code" if x in codes else ("long" if re.fullmatch(r"\d{5,}", x) else None)
                for a in alts:
                    if a == "POINT": pm = True; continue
                    for at in a: at.form, at.kind = "d", kind
                    slot.append(("p" if pm else "w", a))
                seq.append(slot)
            else:
                ndig = sum(len(re.sub(r"\D", "", c)) for c in chain)
                joined = (not FIX["hyphen"]) or len(chain) >= 3 or ndig >= 7
                atoms = []
                for n, c in enumerate(chain):
                    if n and not joined: tokid += 1  # short range/score: each part its own token, never concatenated
                    atoms += _digit_atoms(c, tokid)[0]
                for at in atoms: at.form, at.kind = "d", ("long" if joined and ndig >= 5 else None)
                seq.append([("w", atoms)])
        else:
            if k == "f":
                if run: run.append("<f>")  # in-run barrier: blocks tens+unit, nothing else
            elif k in ("p", "|", "-"):
                flush()
        i += 1
    flush()
    # expand product (capped)
    alts = [([], False)]
    for slot in seq:
        new = []
        for (a, pm) in alts:
            for (tag, s) in slot:
                new.append((a + list(s), pm or tag == "p"))
                if len(new) > 4096: CAP_HITS["product"] += 1; break
        alts = new[:4096]
    return alts


# ---------------------------------------------------------------- matching
def _eq1(r, h):
    if r.s == h.s: return True
    if h.mant is not None and (_CTX["slot_mode"] or not FIX["mant"]) and not r.scaled and dval(r.s) is not None and dval(r.s) == dval(h.mant): return True  # slot "7.616" vs "7.616 billion"
    a, b = dval(r.s), dval(h.s)
    if a is not None and b is not None and a == b and not (r.s.startswith("0") and len(r.s) > 1 and "." not in r.s): return True
    return False


def match(R, H, allow_repeat=False):
    """can ref atoms R be matched, in order, by hyp atoms H (extra hyp atoms allowed)?"""
    from functools import lru_cache
    nR, nH = len(R), len(H)

    def small(a): return a.joinable() and len(a.s) == 1

    @lru_cache(None)
    def f(i, j, st="0"):
        # st: "0" free, "M" previous hyp atom ended a match, "S" previous hyp atom was a skipped single digit
        if i == nR: return not (st == "M" and j < nH and small(H[j]))
        if j == nH: return False
        a = H[j]
        if a is BAR or a.s == "#": return f(i, j + 1, "0")
        if not (small(a) and st == "M"):  # skip a hyp atom (never a stray digit glued to a matched number)
            if f(i, j + 1, "S" if small(a) else "0"): return True
        if allow_repeat and i > 0 and R[i].s == R[i - 1].s and f(i + 1, j, st): return True
        if st == "S": return False
        # ref group i..i2 (same token, joinable, no nojoin boundary) vs hyp group j..j2 (joinable)
        rs = ""
        for i2 in range(i, min(nR, i + 8)):
            if i2 > i and (R[i2].tok is None or R[i2].tok != R[i].tok or R[i2 - 1].nojoin or not R[i2].joinable() or not R[i2 - 1].joinable()): break
            rs += R[i2].s
            if i2 == i and _eq1(R[i], H[j]) and f(i + 1, j + 1, "M"): return True
            hs = ""
            for j2 in range(j, min(nH, j + 16)):
                if not H[j2].joinable(): break
                hs += H[j2].s
                if len(hs) > len(rs): break
                if hs == rs and (j2 > j or i2 > i) and f(i2 + 1, j2 + 1, "M"): return True
                if not rs.startswith(hs): break
        return False
    return f(0, 0)


_PTW = r"(?:\d|%s)" % "|".join(sorted(NUMWORD, key=len, reverse=True))
POINT_CTX = re.compile(r"\b%s\s*[,]?\s*(?:point|dot)\s+%s|\d\.\d|\b%s\.\d" % (_PTW, _PTW, _PTW), re.I)
MONEY_CTX = re.compile(r"\$|\b(?:dollars?|bucks?|cents?|euros?|pounds?)\b|[€£]", re.I)


def point_context(hyp_text, ref_money=False):
    return bool(ref_money or POINT_CTX.search(hyp_text) or MONEY_CTX.search(hyp_text))


def _ok(ref_alts, hyp_text, glue=False, allow_point=False, allow_repeat=False, ref_money=False):
    H = atom_lattice(hyp_text, glue)
    if allow_point and not point_context(hyp_text, ref_money): allow_point = False
    for (R, pm) in ref_alts:
        if pm and not allow_point: continue
        for (Hs, _) in H:
            if match(tuple(R), tuple(Hs), allow_repeat): return True
    return False


def verdict(ref_text, hyp_text, edge=False, slot_mode=None):
    """ref_text: digits-form reference; ' | ' separates slots (each matched on its own, slot order is not speech
    order); or a listener transcript (all numbers, in order). Clip verdict = worst slot verdict. slot_mode: True when ref_text is a slot list (even a single slot, which has no
    " | "); default = " | " in ref_text. It gates the mantissa rule (v2.2)."""
    import itertools
    hyp_text = re.sub(r"(?<=\S)\u00a1", "", hyp_text)  # repeat marker "5¡5" / "two¡ two" is display-stripped, never a number
    parts = [x for x in ref_text.split(" | ") if x.strip()] if " | " in ref_text else [ref_text]
    _CTX["slot_mode"] = (" | " in ref_text) if slot_mode is None else slot_mode
    lat = [[([x for x in r if x.s != "#"], pm) for r, pm in atom_lattice(p, tok0=(k * 100000 if FIX["slot"] else 0))] for k, p in enumerate(parts)]
    lat = [l for l in lat if any(r for r, _ in l)]
    if not lat:  # reference carries no number: OK when the hypothesis carries none either
        return "ok" if not [a for r, _ in atom_lattice(hyp_text)[:1] for a in r if a.s != "#"] else "bad"
    R = []
    perms = list(itertools.permutations(range(len(lat)))) if len(lat) <= 5 else [tuple(range(len(lat)))]
    for pr in perms:
        for combo in itertools.product(*[lat[k] for k in pr]):
            R.append(([a for r, _ in combo for a in r], any(pm for _, pm in combo)))
            if len(R) > 600: break
    return verdict1(R, hyp_text, edge, ref_money=bool(MONEY_CTX.search(ref_text)))


RANK = ["noref", "ok", "ok_point", "ok_edge", "ok_glue", "ok_repeat", "sa", "bad"]


def _edge_ok(R, hyp_text):
    """reference cut at the clip edge: every ref digit chain (>=5 digits) appears in a hyp digit run that is at most
    2 digits longer, the extra digits at one end; all other ref atoms must match normally"""
    hs = " ".join("".join(a.s if a.joinable() else " " for a in Hs) for Hs, _ in atom_lattice(hyp_text)[:8])
    runs = set(hs.split())
    for (r, _) in R:
        toks = {}
        for a in r:
            if a.tok is not None and a.joinable(): toks.setdefault(a.tok, []).append(a.s)
        chains = ["".join(v) for v in toks.values() if len("".join(v)) >= 5]
        if chains and all(any(0 < len(x) - len(c) <= 2 and (x.startswith(c) or x.endswith(c)) for x in runs) or c in runs for c in chains):
            rest = tuple(a for a in r if not (a.tok in toks and len("".join(toks[a.tok])) >= 5))
            if not rest or any(match(rest, tuple(Hs)) for Hs, _ in atom_lattice(hyp_text)): return True
    return False


def verdict1(R, hyp_text, edge=False, ref_money=False):
    if _ok(R, hyp_text): return "ok"
    if _ok(R, hyp_text, allow_point=True, ref_money=ref_money): return "ok_point"
    if edge and _edge_ok(R, hyp_text): return "ok_edge"
    if _ok(R, hyp_text, glue=True, allow_point=True): return "ok_glue"
    if _ok(R, hyp_text, glue=True, allow_point=True, allow_repeat=True): return "ok_repeat"
    sa = re.sub(r"\b(%s)\b" % "|".join(SA), lambda m: SA[m.group(1).lower()], hyp_text, flags=re.I)
    if sa != hyp_text and _ok(R, sa, glue=True, allow_point=True, allow_repeat=True): return "sa"
    return "bad"


OK_SET = {"ok", "ok_point", "ok_edge", "ok_glue", "ok_repeat"}
HEAD = {"ok", "ok_point", "ok_edge"}  # counted as right in the headline


# ================================================================ v2.2 metrics (verdict() is unaffected)
def _clean_hyp(hyp_text): return re.sub(r"(?<=\S)\u00a1", "", hyp_text or "")


def _ref_alts(ref_text, slot_mode=None):
    """reference atom-list alternatives, exactly as verdict() builds them (slot permutations x point variants)"""
    import itertools
    parts = [x for x in ref_text.split(" | ") if x.strip()] if " | " in ref_text else [ref_text]
    _CTX["slot_mode"] = (" | " in ref_text) if slot_mode is None else slot_mode
    lat = [[([x for x in r if x.s != "#"], pm) for r, pm in atom_lattice(p, tok0=(k * 100000 if FIX["slot"] else 0))] for k, p in enumerate(parts)]
    lat = [l for l in lat if any(r for r, _ in l)]
    if not lat: return []
    R = []
    perms = list(itertools.permutations(range(len(lat)))) if len(lat) <= 5 else [tuple(range(len(lat)))]
    for pr in perms:
        for combo in itertools.product(*[lat[k] for k in pr]):
            R.append([a for r, _ in combo for a in r])
            if len(R) > 600: return R
    return R


def _align(R, H):
    """best in-order alignment of ref atoms R to hyp atoms H: maximise matched ref atoms, then consumed hyp atoms.
    Ref atoms may stay unmatched, hyp atoms may be skipped (= insertions). Same match rules as match().
    Returns (ref_matched, hyp_consumed, [(i, i2, j, j2), ...]) with inclusive atom index ranges."""
    from functools import lru_cache
    nR, nH = len(R), len(H)

    @lru_cache(None)
    def f(i, j):
        if i == nR or j == nH: return (0, 0, ())
        best = f(i + 1, j) if i < nR else (0, 0, ())
        a = H[j]
        s2 = f(i, j + 1)
        if (s2[0], s2[1]) > (best[0], best[1]): best = s2
        if a is BAR or a.s == "#": return best
        rs = ""
        for i2 in range(i, min(nR, i + 8)):
            if i2 > i and (R[i2].tok is None or R[i2].tok != R[i].tok or R[i2 - 1].nojoin or not R[i2].joinable() or not R[i2 - 1].joinable()): break
            rs += R[i2].s
            cands = []
            if i2 == i and _eq1(R[i], H[j]): cands.append(j)
            hs = ""
            for j2 in range(j, min(nH, j + 16)):
                if not H[j2].joinable(): break
                hs += H[j2].s
                if len(hs) > len(rs): break
                if hs == rs and (j2 > j or i2 > i): cands.append(j2)
                if not rs.startswith(hs): break
            for j2 in cands:
                r = f(i2 + 1, j2 + 1)
                c = (r[0] + i2 - i + 1, r[1] + j2 - j + 1, ((i, i2, j, j2),) + r[2])
                if (c[0], c[1]) > (best[0], best[1]): best = c
        return best
    return f(0, 0)


def _best_alignment(ref_text, hyp_text, budget=1500, slot_mode=None):
    """(unmatched_hyp_atoms, ref_atoms_matched, n_ref_atoms, R, Hs, matches) of the best (ref alt, hyp alt) pair"""
    Rs = _ref_alts(ref_text, slot_mode)
    Hl, seen = [], set()
    for Hs, _ in atom_lattice(_clean_hyp(hyp_text)):
        Hs = [a for a in Hs if a.s != "#"]
        key = tuple(a.s for a in Hs)
        if key in seen: continue
        seen.add(key); Hl.append(Hs)
        if len(Hl) >= 64: break
    if not Rs: Rs = [[]]
    best, n = None, 0
    for R in Rs:
        for Hs in Hl:
            n += 1
            if n > budget: break
            rm, hc, mt = _align(tuple(R), tuple(a for a in Hs))
            cand = (-rm, len([a for a in Hs if a is not BAR]) - hc, R, Hs, mt, rm, len(R))
            if best is None or cand[:2] < best[:2]: best = cand
            if best[0] == -len(R) and best[1] == 0: break
        if n > budget or (best and best[0] == -len(best[2]) and best[1] == 0): break
    return best


def insertions(ref_text, hyp_text, slot_mode=None):
    """number of hypothesis number atoms left unmatched after the best in-order match against the reference:
    extra numbers, stutters ("twenty, twenty five" for 25 -> 1). A hyp atom counts once whether it is a word-number
    ("twenty five") or a digit token. A hypothesis with no number against a reference with numbers -> 0."""
    b = _best_alignment(ref_text, hyp_text, slot_mode=slot_mode)
    return b[1]


def form_report(ref_text, hyp_text, slot_mode=None):
    """for each reference number group matched by the best alignment: (ref_kind, hyp_form).
    ref_kind: 'long' (digit string >= 5 digits, no thousands commas/decimal point; or a joined hyphen chain >= 5
    digits), 'code' (letter-digit code in the reference: M24KB, ESP8266, A-10), else 'ordinary'.
    hyp_form: 'digits' (all hyp atoms of the match came from digit tokens), 'words', or 'mixed'."""
    b = _best_alignment(ref_text, hyp_text, slot_mode=slot_mode)
    _, _, R, Hs, mt, _, _ = b
    out = []
    for (i, i2, j, j2) in mt:
        ks = {a.kind for a in R[i:i2 + 1]}
        kind = "code" if "code" in ks else "long" if "long" in ks else "ordinary"
        fs = {Hs[z].form for z in range(j, j2 + 1)}
        form = "digits" if fs == {"d"} else "words" if fs == {"w"} else "mixed"
        out.append((kind, form))
    return out


def form_summary(reports):
    """reports: iterable of form_report() lists -> counts {kind: {form: n}} and the three headline rates:
    ordinary_words_pct, long_digits_pct, code_digits_pct (None when no such numbers)"""
    c = {k: {"digits": 0, "words": 0, "mixed": 0} for k in ("ordinary", "long", "code")}
    for rep in reports:
        for k, f in rep: c[k][f] += 1
    def pct(k, f):
        n = sum(c[k].values())
        return None if not n else 100.0 * c[k][f] / n
    return {"counts": c, "ordinary_words_pct": pct("ordinary", "words"), "long_digits_pct": pct("long", "digits"), "code_digits_pct": pct("code", "digits")}


_NN_DROP = {"percent", "dollar", "dollars", "and", "a", "an", "point", "dot", "double", "triple", "oclock", "clock", "o"}


def nonnum_words(text):
    """lowercased words of text with number tokens (digits, number words, ordinals, decades) and fillers removed.
    'and'/'a'/'point'/'double' are removed only when adjacent to a number token; percent/dollar(s) always."""
    t = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
    t = re.sub(r"o'clock", " ", t)
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    toks = re.findall(r"\d[\d.,:/-]*[a-z]{0,2}|[a-z]+(?:'[a-z]+)?", t)
    def isnum(w):
        if w[0].isdigit(): return True
        w = MISSPELL.get(w, w) or ""
        return w in NUMWORD or w in FILLERS or w in FRAC_D and False
    num = [isnum(w) for w in toks]
    out = []
    for k, w in enumerate(toks):
        if num[k]: continue
        if w in ("percent", "dollar", "dollars"): continue
        if w in ("and", "a", "an", "point", "dot", "double", "triple"):
            nb = (k > 0 and num[k - 1]) or (k + 1 < len(toks) and num[k + 1])
            if nb: continue
        out.append(w)
    return out


def nonnum_wer(ref_text, hyp_text):
    """(edits, ref_words): word Levenshtein between ref and hyp after removing number tokens and fillers from both,
    lowercasing and stripping punctuation. Shows prose drops/garbles around numbers. ref_text should be PROSE
    (a caption/transcript), not a ' | ' slot list."""
    a, b = nonnum_words(ref_text), nonnum_words(hyp_text)
    prev = list(range(len(b) + 1))
    for x in a:
        cur = [prev[0] + 1]
        for j, y in enumerate(b): cur.append(min(prev[j] + (x != y), prev[j + 1] + 1, cur[j] + 1))
        prev = cur
    return prev[-1], len(a)
