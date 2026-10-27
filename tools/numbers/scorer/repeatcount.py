#!/usr/bin/env python3
"""repeatcount: EXACT repeat-count scorer (pure python, no model).

Does the hypothesis carry the reference digit string with every repeated digit or period intact? 'two two' vs 'two' is a failure;
'double two' = 'two two' = '22' = 'twenty two' (form-blind).

Truth is a canonical STRING (digits, plus letters in alnum mode); digit strings are exact: value-exact == repeat-count-exact because the run-length coding of a
string is lossless. The repeat taxonomy therefore classifies FAILURES (what the system did to the repeat): exact | repeat_drop | repeat_add | repeat_mixed |
period_drop | period_add | other_value | slot_dropped (+ extra_runs when the hypothesis span holds more number runs than the reference).

Reader rules: zero = oh = o = nought = naught; double / triple / quadruple / quintuple X = X X (X X X ...);
any grouping of the same digit sequence is one string ('twenty two thirty three' = 2233 = 'double two double three'); separators , . - / space and punctuation are ignored
INSIDE a number run (only a non-number WORD ends a run); to / too = 2, for / fore = 4, won = 1, ate = 8 only inside a run (a number token directly before them AND another
number token directly before or after); NO trailing-zero tolerance ('twenty' != '2'); a tens word followed by a unit is ONE group
('twenty two' = 22, never 202). Negatives are unit-tested (test_repeatcount.py): 'two two' vs 'two' FAIL, 'in twenty five' != 2005, 'sixty si' != 66.

Entry points: score_value(ref, span_text), score_sequence([ref...], span_text) (self-corrections), score_chunk(ref_text, slot_range, truth, hyp_text)
(locates the slot in the hypothesis by alignment first), score_other(accept_forms, hyp_text) (times, ranges, powers, lakh / crore, ...).
The reader step can be replaced with set_reader(fn(text, mode) -> list[set[str]]); a sibling tools/numread install, when importable, is used for a diagnostic cross-check only.
  python3 -I test_repeatcount.py     # golden pairs incl. negatives
"""
import difflib, itertools, os, re, sys
from collections import namedtuple

ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALE = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
ZERO_W = {"oh", "nought", "naught"}
MULT = {"double": 2, "triple": 3, "treble": 3, "quadruple": 4, "quintuple": 5, "sextuple": 6}
HOMO = {"to": "2", "too": "2", "two": "2", "for": "4", "fore": "4", "won": "1", "ate": "8"}
NUMW = set(ONES) | set(TENS) | set(SCALE) | ZERO_W | {"hundred"}
LETTER_NAMES = {"ay": "A", "bee": "B", "cee": "C", "see": "C", "sea": "C", "dee": "D", "eff": "F", "gee": "G", "aitch": "H", "jay": "J", "kay": "K", "ell": "L", "em": "M", "en": "N",
                "pee": "P", "cue": "Q", "are": "R", "ess": "S", "tee": "T", "vee": "V", "ex": "X", "zed": "Z", "zee": "Z"}
NATO = {w: w[0].upper() for w in ("alfa alpha bravo charlie delta echo foxtrot golf hotel india juliet juliett kilo lima mike november oscar papa quebec romeo sierra tango "
                                  "uniform victor whiskey xray yankee zulu").split()}
NATO["x-ray"] = "X"
Tok = namedtuple("Tok", "kind text cands s e", defaults=(-1, -1))  # kind: 'd' digit token | 'w' number word | 'L' letter | 'mult' | 'x' other word ; cands: set of strings for d / w groups

_READER = [None]  # numread hook


def set_reader(fn):
    """fn(text, mode) -> list[set[str]] (one candidate set per number run). None = built-in reader."""
    _READER[0] = fn


def reader_id():
    return "numread.digit_candidates" if _READER[0] else "repeatcount.builtin"



# ------------------------------------------------------------------------------------------------------------------------------ word groups
def parse_group(ws):
    """words of ONE number group -> int or None. ('twenty two' = 22, 'one hundred and five' = 105, 'two thousand twenty' = 2020)"""
    total, cur, last_scale = 0, 0, None
    for w in ws:
        if w == "and":
            if not (cur >= 100 or total > 0):
                return None
        elif w in ONES or w in TENS:
            v = ONES.get(w, TENS.get(w))
            m = cur % 100
            if w == "zero":
                return None  # zero / oh are their own group (run_readings)
            if not (m == 0 or (w in ONES and ONES[w] < 10 and m >= 20 and m % 10 == 0)):
                return None
            cur += v
        elif w == "hundred":
            if not (0 < cur < 100):  # 'two hundred', colloquial 'fifteen hundred' (1500)
                return None
            cur *= 100
        elif w in SCALE:
            s = SCALE[w]
            if cur == 0 or (last_scale is not None and s >= last_scale):
                return None
            total += cur * s
            cur = 0
            last_scale = s
        else:
            return None
    if ws and ws[-1] == "and":
        return None
    return total + cur


SCALE_MIN = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}


def run_readings(ws, cap=64):
    """all digit-string readings of a run of number words: partitions into groups; zero / oh are their own group; a tens word is never split from a following unit;
    a group that ends in hundred / thousand / million is never followed by a group SMALLER than that scale (it would have been absorbed: 'two hundred two' = 202, never 200|2 = 2002)."""
    n = len(ws)
    memo = {}

    def go(i, pm):
        if i == n:
            return {""}
        if (i, pm) in memo:
            return memo[(i, pm)]
        out = set()
        if ws[i] in ZERO_W or ws[i] == "zero":
            if pm <= 1:
                for r in go(i + 1, 1):
                    out.add("0" + r)
        else:
            for j in range(i + 1, min(n, i + 14) + 1):
                g = parse_group(ws[i:j])
                if g is None or g < pm:
                    continue
                if j < n and ws[j] in ONES and 1 <= ONES[ws[j]] <= 9 and j > i and ws[j - 1] in TENS and g >= 20 and g % 10 == 0:
                    continue  # 'twenty' | 'two' is never a reading of 'twenty two'
                for r in go(j, SCALE_MIN.get(ws[j - 1], 1)):
                    out.add(str(g) + r)
                if len(out) > cap:
                    break
        memo[(i, pm)] = set(sorted(out, key=lambda s: (len(s), s))[:cap])
        return memo[(i, pm)]

    return go(0, 1)


# ------------------------------------------------------------------------------------------------------------------------------ tokenizer
_SPLIT = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\sa-z\d]", re.I)


def _norm_text(text):
    """lower-cased text with the SAME length as the input (token offsets index the original string)"""
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)  # twenty-two, double-oh-seven
    t = re.sub(r"(?<=\d)[-/ ](?=\d)", " ", t)     # 877-664-2233 -> digit tokens
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", " ", t)    # 2,233 -> '2 233' (a separator inside the run)
    t = re.sub(r"(?<=\d)\.(?=\d)", " ", t)        # 29.29 -> 29 29 (the point is a separator inside a run)
    assert len(t) == len(text), "length-preserving normalisation"
    return t


AMB_NAMES = {"a": "A", "i": "I", "are": "R", "see": "C", "sea": "C", "ex": "X", "em": "M", "en": "N", "bee": "B"}  # letter names that are also ordinary words
HOMW = {"to": "two", "too": "two", "for": "four", "fore": "four", "won": "one", "ate": "eight"}
_ALNUM_COMPACT = r"[a-z]{1,4}\d[a-z0-9]*|\d+[a-z]{1,4}\b"
_SEP_CODE = re.compile(r"(?P<sep>[a-z]{2,4}[- ]\d+[a-z0-9]*)|(?P<cmp>" + _ALNUM_COMPACT + r")|(?P<w>[a-z]+(?:'[a-z]+)?)|(?P<d>\d+)|(?P<p>[^\sa-z\d])")
_PLAIN = re.compile(r"(?P<cmp>" + _ALNUM_COMPACT + r")|(?P<w>[a-z]+(?:'[a-z]+)?)|(?P<d>\d+)|(?P<p>[^\sa-z\d])")
_PLAIN_DIG = re.compile(r"(?P<w>[a-z]+(?:'[a-z]+)?)|(?P<d>\d+)|(?P<p>[^\sa-z\d])")
FILLERS = {"um", "uh", "er", "erm", "hmm", "ah", "umm", "uhh", "dot", "colon"}  # hesitations / IP separators spoken as words are transparent inside a run


def raw_tokens(text, mode="digits"):
    out = []
    low = _norm_text(text)
    pos = 0
    while pos < len(low):
        if low[pos].isspace():
            pos += 1
            continue
        rx = _SEP_CODE if mode == "alnum" else _PLAIN_DIG
        m = rx.match(low, pos)
        if mode == "alnum" and m.lastgroup == "sep":
            k = len(re.match(r"[a-z]+", m.group(0)).group(0))
            if not text[pos:pos + k].isupper():  # 'tee 5' stays words; only ALL-CAPS 'KK 5009' / 'KK-5009' is a stock letter code
                m = _PLAIN.match(low, pos)
        w, kind = m.group(0), m.lastgroup
        s0, e0 = m.span()
        pos = e0
        if kind in ("sep", "cmp") and mode == "alnum":
            c = re.sub(r"[- ]", "", w)
            out.append(Tok("alnum", c, {c.upper()}, s0, e0))
        elif kind == "d":
            out.append(Tok("d", w, {w}, s0, e0))
        elif kind == "p":
            out.append(Tok("p", w, set(), s0, e0))
        else:
            if w in MULT:
                out.append(Tok("mult", w, set(), s0, e0))
            elif w in FILLERS:
                out.append(Tok("p", w, set(), s0, e0))
            elif w == "oh":
                out.append(Tok("o", w, set(), s0, e0))  # zero only next to a number token (interjection otherwise)
            elif w in NUMW or w == "and":
                out.append(Tok("w", w, set(), s0, e0))
            elif w in HOMW and w != "two":
                out.append(Tok("h", w, set(), s0, e0))
            elif mode == "alnum" and w in AMB_NAMES:
                out.append(Tok("la", w, {AMB_NAMES[w]}, s0, e0))  # a / i / are / see ...: a letter only next to a code token (resolve)
            elif mode == "alnum" and w in NATO:
                out.append(Tok("L", w, {NATO[w]}, s0, e0))
            elif mode == "alnum" and w in LETTER_NAMES:
                out.append(Tok("L", w, {LETTER_NAMES[w]}, s0, e0))
            elif mode == "alnum" and len(w) == 1:
                out.append(Tok("L", w, {w.upper()}, s0, e0))
            elif w == "o":
                out.append(Tok("o", w, set(), s0, e0))
            else:
                out.append(Tok("x", w, set(), s0, e0))
    return out


def _is_num(t):
    return t.kind in ("d", "w", "L", "alnum") or (t.kind == "mult")


def resolve(toks, hom=False):
    """second pass: oh / homophones / ambiguous letter names / 'and'. -> list of Tok (offsets kept) with kind in d w L alnum mult x p."""
    ks = [t.kind for t in toks]
    nonp = [i for i, t in enumerate(toks) if t.kind != "p"]
    pos = {i: k for k, i in enumerate(nonp)}

    def nb(i, d):  # neighbour non-punctuation token kind
        k = pos[i] + d
        return ks[nonp[k]] if 0 <= k < len(nonp) else None

    def numlike(k):
        return k in ("d", "w", "L", "alnum", "mult", "o")

    def imm(i, d):  # IMMEDIATE neighbour kind: sentence punctuation stops it, spoken separators / fillers ('dot', 'uh') are looked through
        k = i + d
        while 0 <= k < len(toks) and toks[k].kind == "p" and toks[k].text.isalpha():
            k += d
        return ks[k] if 0 <= k < len(toks) else None

    res = list(toks)
    for _ in range(3):
        for i in nonp:
            t = res[i]
            if t.kind == "o":
                if numlike(imm(i, -1)) or numlike(imm(i, 1)):
                    res[i] = t._replace(kind="w", text="oh")
                else:
                    res[i] = t._replace(kind="x", text="o")
            elif t.kind == "la":
                strict = t.text in ("a", "i")
                p_, n_ = nb(i, -1), nb(i, 1)
                if strict:
                    ok = p_ in ("mult", "L", "alnum") or n_ in ("L", "alnum")
                else:
                    ok = p_ in ("d", "w", "L", "alnum", "mult", "la") or n_ in ("d", "w", "L", "alnum", "mult", "la")
                res[i] = t._replace(kind="L") if ok else t._replace(kind="x", cands=set())
            elif t.kind == "h":
                # homophones (to too for fore won ate) are number WORDS ONLY when the caller scores an aligned slot span (hom=True) and a number token touches them; never in free text
                if hom and (numlike(nb(i, -1)) or numlike(nb(i, 1))):
                    res[i] = t._replace(kind="w", text=HOMW[t.text])
                else:
                    res[i] = t._replace(kind="x")
            elif t.kind == "w" and t.text == "and":
                pv, nx = nb(i, -1), nb(i, 1)
                prev_tok = res[nonp[pos[i] - 1]] if pos[i] >= 1 else None
                if not (pv == "w" and prev_tok.text in ("hundred", "thousand", "million", "billion") and nx == "w"):
                    res[i] = t._replace(kind="x")
            ks = [t.kind for t in res]
    return res


def tokenize(text, mode="digits", hom=False):
    return resolve(raw_tokens(text, mode), hom)


def runs(toks):
    """maximal runs of number tokens; punctuation inside a run is transparent, any other word ends it. -> list of (start, end_exclusive, [tokens without punct])"""
    out, cur, s = [], [], None
    for i, t in enumerate(toks):
        if t.kind in ("d", "w", "L", "alnum", "mult"):
            if not cur:
                s = i
            cur.append(t)
        elif t.kind == "p":
            continue
        else:
            if cur:
                out.append((s, i, cur))
                cur = []
    if cur:
        out.append((s, len(toks), cur))
    return out


def run_candidates(rt, cap=200):
    """candidate canonical strings of one run (list of Tok). 'double X' expands to X X; adjacent word tokens are partitioned into groups."""
    seq = []  # items: ('s', {str}) a set of alternatives per segment
    i, n = 0, len(rt)
    while i < n:
        t = rt[i]
        if t.kind == "mult":
            k = MULT[t.text]
            nx = rt[i + 1] if i + 1 < n else None
            if nx is not None and nx.kind in ("d", "L", "alnum") or (nx is not None and nx.kind == "w" and (nx.text in ONES and ONES[nx.text] < 10 or nx.text in ZERO_W)):
                if nx.kind == "w":
                    sub = "0" if nx.text in ZERO_W or nx.text == "zero" else str(ONES[nx.text])
                    seq.append({sub * k})
                else:
                    seq.append({"".join(sorted(nx.cands)[0] if nx.kind != "d" else nx.text for _ in range(k))})
                i += 2
                continue
            i += 1  # 'double' not followed by a digit / letter: no reading for the word itself
            continue
        if t.kind == "d":
            seq.append({t.text})
            i += 1
        elif t.kind in ("L", "alnum"):
            seq.append({sorted(t.cands)[0]})
            i += 1
        else:  # a run of number words
            j = i
            while j < n and rt[j].kind == "w":
                j += 1
            seq.append(run_readings([x.text for x in rt[i:j]]) or {""})
            i = j
    outs = {""}
    for alts in seq:
        outs = {a + b for a in outs for b in alts}
        if len(outs) > cap:
            outs = set(sorted(outs, key=lambda s: (len(s), s))[:cap])
    return outs


def digit_candidates(text, mode="digits", hom=False):
    """text -> list (one entry per number run, in order) of candidate canonical strings. hom=True resolves homophones (to / for / ate ...) next to a number token: ONLY for an aligned slot span."""
    if _READER[0]:
        return _READER[0](text, mode)
    return [run_candidates(r) for _, _, r in runs(tokenize(text, mode, hom)) if run_candidates(r) - {""}]


def joined_candidates(text, mode="digits", cap=400, hom=False):
    """all runs concatenated (separators and ordinary words between runs ignored): the span as ONE string. -> set of strings"""
    outs = {""}
    for c in digit_candidates(text, mode, hom):
        outs = {a + b for a in outs for b in c}
        if len(outs) > cap:
            outs = set(sorted(outs, key=lambda s: (len(s), s))[:cap])
    return outs


# ------------------------------------------------------------------------------------------------------------------------------ run-length coding + taxonomy
def rle(s):
    return [(k, len(list(g))) for k, g in itertools.groupby(s)]


def rle_str(s):
    return " ".join(f"{k}x{n}" if n > 1 else k for k, n in rle(s))


def primitive_period(s):
    for p in range(1, len(s) // 2 + 1):
        if len(s) % p == 0 and s[:p] * (len(s) // p) == s:
            return s[:p]
    return None


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _deletable_subseq(long, short):
    """True when `short` is a subsequence of `long` and every deleted char of `long` lies in a run of >= 2 equal chars of `long` (a copy)"""
    n, m = len(long), len(short)
    if m >= n:
        return False
    runlen = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and long[j] == long[i]:
            j += 1
        for k in range(i, j):
            runlen[k] = j - i
        i = j
    NEG = -1
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n):
        for j in range(m + 1):
            if not dp[i][j]:
                continue
            if runlen[i] >= 2:
                dp[i + 1][j] = True
            if j < m and long[i] == short[j]:
                dp[i + 1][j + 1] = True
    return dp[n][m]


def classify(ref, hyp):
    """failure class of hyp string against ref string (both canonical). 'exact' when equal."""
    if hyp == ref:
        return "exact"
    if not hyp:
        return "slot_dropped"
    a, b = rle(ref), rle(hyp)
    if [k for k, _ in a] == [k for k, _ in b]:
        cr, ch = [n for _, n in a], [n for _, n in b]
        if all(x <= y for x, y in zip(ch, cr)):
            return "repeat_drop"
        if all(x >= y for x, y in zip(ch, cr)):
            return "repeat_add"
        return "repeat_mixed"
    if _deletable_subseq(ref, hyp):
        return "repeat_drop"   # e.g. 007 -> 7, 2233 -> 23: only copies inside runs of >= 2 were removed
    if _deletable_subseq(hyp, ref):
        return "repeat_add"
    for p_ in range(2, len(ref) // 2 + 1):  # a whole period removed from / added to a square inside a longer string (abab -> ab, 3232 -> 32, 123123 -> 123)
        for i_ in range(0, len(ref) - 2 * p_ + 1):
            if ref[i_:i_ + p_] == ref[i_ + p_:i_ + 2 * p_] and len(set(ref[i_:i_ + p_])) > 1 and ref[:i_] + ref[i_ + p_:] == hyp:
                return "period_drop"
    for p_ in range(2, len(hyp) // 2 + 1):
        for i_ in range(0, len(hyp) - 2 * p_ + 1):
            if hyp[i_:i_ + p_] == hyp[i_ + p_:i_ + 2 * p_] and len(set(hyp[i_:i_ + p_])) > 1 and hyp[:i_] + hyp[i_ + p_:] == ref:
                return "period_add"
    for p_ in range(2, len(ref) // 2 + 1):  # a period ADDED next to an existing square also shows as a duplicated block
        for i_ in range(0, len(ref) - p_ + 1):
            if len(set(ref[i_:i_ + p_])) > 1 and ref[:i_ + p_] + ref[i_:] == hyp:
                return "period_add" if ref[i_:i_ + p_] == ref[max(0, i_ - p_):i_] or ref[i_:i_ + p_] == ref[i_ + p_:i_ + 2 * p_] else "other_value"
    p = primitive_period(ref)
    if p and len(p) >= 1 and len(ref) > len(p):
        k = len(ref) // len(p)
        if hyp == p * (len(hyp) // len(p)) and len(hyp) % len(p) == 0 and len(hyp) != len(ref):
            return "period_drop" if len(hyp) < len(ref) else "period_add"
    q = primitive_period(hyp)
    if q and hyp != ref and len(hyp) > len(q) and ref == q * (len(ref) // len(q)) and len(ref) % len(q) == 0:
        return "period_add" if len(hyp) > len(ref) else "period_drop"
    return "other_value"


def best_candidate(ref, cands):
    """the candidate closest to ref (edit distance, then equal length). exact wins."""
    if ref in cands:
        return ref
    return min(cands, key=lambda c: (lev(ref, c), abs(len(c) - len(ref)), c)) if cands else ""


# --- counted repeat markers (see tools/numbers/marker.py): English TRAINING labels mark copies 1..n-1 of identical neighbouring number words with a bare piece glued
# after the word: '\u00a1' = copy 1, '\u00bf' = copy 2, '\u00a1\u00a1' = copy 3 and later ('two\u00a1 two\u00bf two\u00a1\u00a1 two'). TEST references stay UNMARKED. Hypotheses are stripped
# of markers BEFORE reading; the raw marker output is reported separately. Spanish word-INITIAL '\u00a1' / '\u00bf' (an inverted mark after a space or at the start, as in '\u00bfQu\u00e9?') are NOT markers: only a mark attached
# directly after a non-space character counts. '\u00a1\u00a1' is ONE mark (copy 3+).
MARKS = ["\u00a1\u00a1", "\u00a1", "\u00bf"]  # longest first
MARKER_CHARS = "".join(sorted(set("".join(MARKS))))  # kept for callers that test membership
_MARK_RE = [None]


def set_marker_chars(marks):
    """marks: list of strings (longest first) or a string of single characters"""
    global MARKS, MARKER_CHARS
    MARKS = sorted(list(marks), key=lambda m: -len(m))
    MARKER_CHARS = "".join(sorted(set("".join(MARKS))))
    _MARK_RE[0] = None


def _marker_re():
    if _MARK_RE[0] is None:
        _MARK_RE[0] = re.compile(r"(?<=\S)(" + "|".join(re.escape(m) for m in MARKS) + ")")
    return _MARK_RE[0]


def strip_markers(text):
    """-> (clean text, list of (position in raw text, mark)). A mark glued after a character is removed; between two letters ('five\u00a1five') it becomes a space so the words do not fuse.
    A mark preceded by whitespace or at the start (Spanish '\u00bfQu\u00e9?') is left alone."""
    found = [(m.start(), m.group(1)) for m in _marker_re().finditer(text)]
    if not found:
        return text, []

    def sub(m):
        a, b = m.start(), m.end()
        return " " if (a > 0 and text[a - 1].isalpha() and b < len(text) and text[b].isalpha()) else ""
    return _marker_re().sub(sub, text), found


def expected_marker_seq(n_copies):
    """marks the trained label carries for a run of n identical neighbouring number words: copy 1 '\u00a1', copy 2 '\u00bf', copy 3+ '\u00a1\u00a1' (n-1 marks)"""
    return [("\u00a1", "\u00bf")[k] if k < 2 else "\u00a1\u00a1" for k in range(max(0, n_copies - 1))]


def marker_report(text, expected=None, copies=None):
    """raw marker output of a hypothesis: count, marks, and whether it equals the expected count (int) / the expected sequence for a run of `copies` words"""
    clean, found = strip_markers(text)
    marks = [m for _, m in found]
    rep = {"n_markers": len(marks), "marks": marks, "chars": "".join(marks), "as_expected": (None if expected is None else len(marks) == expected)}
    if copies is not None:
        rep["expected_seq"] = expected_marker_seq(copies)
        rep["seq_ok"] = marks == rep["expected_seq"]
    return rep


def score_value(ref, span_text, mode="digits", expected_markers=None):
    """ref canonical string vs the hypothesis span text (marker symbols stripped first; raw markers in `markers`). -> dict(exact, cls, hyp, ref_rle, hyp_rle, n_runs, markers)"""
    mk = marker_report(span_text, expected_markers)
    span_text, _ = strip_markers(span_text)
    res = _score_value(ref, span_text, mode)
    res["markers"] = mk
    return res


def _score_value(ref, span_text, mode="digits"):
    runs_c = digit_candidates(span_text, mode, hom=True)
    allc = joined_candidates(span_text, mode, hom=True)
    if not allc - {""}:
        return {"exact": False, "cls": "slot_dropped", "hyp": "", "ref_rle": rle_str(ref), "hyp_rle": "", "n_runs": 0}
    h = best_candidate(ref, allc)
    cls = classify(ref, h)
    res = {"exact": cls == "exact", "cls": cls, "hyp": h, "ref_rle": rle_str(ref), "hyp_rle": rle_str(h), "n_runs": len(runs_c)}
    if len(runs_c) > 1 and res["exact"]:  # a single value split by a word ('five and five', 'two of two'): the digits are right but the run is broken -> not exact
        res.update(exact=False, cls="split_runs")
    res["numread"] = _numread_check(ref, span_text)
    return res


def _numread_check(ref, span_text):
    """cross-check with the pinned reader (tools/numread) when it is importable: does one of its digit chains equal the reference? Diagnostic only (never decides)."""
    try:
        nr = _numread()
        if nr is None:
            return None
        chains = nr.digit_chains(span_text, "en")
        return {"chains": chains, "agrees": ("".join(chains) == ref), "stamp": nr.stamp_line()[:60]}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


_NR = [False]


def _numread():
    """the sibling tools/numread reader when it is importable (diagnostic cross-check only), else None"""
    if _NR[0] is False:
        _NR[0] = None
        try:
            d = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "numread"))
            if os.path.exists(os.path.join(d, "numread.py")):
                sys.path.insert(0, d)
                import numread as nr  # noqa: E402

                nr.install()
                _NR[0] = nr
        except Exception:  # noqa: BLE001
            _NR[0] = None
    return _NR[0]


def score_sequence(refs, span_text, mode="digits"):
    span_text, _ = strip_markers(span_text)
    return _score_sequence(refs, span_text, mode)


def _score_sequence(refs, span_text, mode="digits"):
    """self-correction: the retracted AND the corrected value, each exact, in order (accept-set 6). Connector words are scored elsewhere (words lost)."""
    runs_c = digit_candidates(span_text, mode, hom=True)
    got = []
    for c in runs_c:
        got.append(best_candidate(refs[len(got)] if len(got) < len(refs) else refs[-1], c))
    res = {"exact": False, "cls": None, "hyp_runs": got, "n_runs": len(runs_c), "parts": []}
    if not runs_c:
        res["cls"] = "slot_dropped"
        return res
    if len(runs_c) == 1 and len(refs) == 2:
        c = runs_c[0]
        if refs[1] in c:
            res["cls"] = "retraction_dropped"  # only the corrected value survived (merge / model collapsed 'a sorry b' to b)
            return res
        if refs[0] in c:
            res["cls"] = "correction_dropped"
            return res
    ok, classes = True, []
    for k, r in enumerate(refs):
        if k < len(got):
            cl = classify(r, got[k])
        else:
            cl = "slot_dropped"
        classes.append(cl)
        ok &= cl == "exact"
    res["parts"] = classes
    res["exact"] = ok and len(runs_c) == len(refs)
    res["cls"] = "exact" if res["exact"] else ("extra_runs" if ok else next(c for c in classes if c != "exact"))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ span location
def _skel(toks, want_map=False):
    """collapse each run of number tokens (punctuation transparent) into one 'N' symbol. -> (symbols, spans[symbol_index] = (tok_start, tok_end) for N symbols, wordpos)"""
    sym, spans, tmap = [], {}, {}
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t.kind in ("d", "w", "L", "alnum", "mult"):
            s = i
            j = i
            last = i
            while j < n and toks[j].kind in ("d", "w", "L", "alnum", "mult", "p"):
                if toks[j].kind != "p":
                    last = j
                j += 1
            spans[len(sym)] = (s, last + 1)
            if want_map:
                for k in range(s, last + 1):
                    tmap[k] = len(sym)
            sym.append("N")
            i = last + 1
        elif t.kind == "p":
            i += 1
        else:
            if want_map:
                tmap[i] = len(sym)
            sym.append(t.text)
            i += 1
    return (sym, spans, tmap) if want_map else (sym, spans)


def locate_span(ref_text, slot_char_range, hyp_text, mode="digits"):
    """hyp text of the region aligned to the reference slot. ref_text = the chunk's reference words, slot_char_range = (a, b) chars of the slot inside ref_text.
    Number runs are collapsed to 'N' symbols; the ref N-runs overlapping the slot are aligned with the hyp N-runs by difflib over the other words.
    -> (span_text, info)"""
    a, b = slot_char_range
    rt, ht = tokenize(ref_text, mode), tokenize(hyp_text, mode)
    # reference N-runs overlapping the slot, from CHARACTER OFFSETS of the single full tokenisation (a neighbouring 'a' / 'see' / 'oh' can merge into the slot's run: the run is still the slot's run)
    rs, rspans = _skel(rt)
    hs, hspans, htmap = _skel(ht, True)
    want = [k for k, (ts, te) in sorted(rspans.items()) if rt[ts].s < b and rt[te - 1].e > a]
    # slot symbols that are words (connectors, stutters) are part of the span (the range want[0]..want[-1] is contiguous)
    if not want:
        return "", {"aligned": False}
    sm = difflib.SequenceMatcher(None, rs, hs, autojunk=False)
    hidx = []
    equal_h = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            equal_h.update(range(j1, j2))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        lo, hi = max(i1, want[0]), min(i2, want[-1] + 1)
        if tag == "equal":
            if lo < hi:
                hidx += list(range(j1 + (lo - i1), j1 + (hi - i1)))
        elif lo < hi or (i1 <= want[0] < i2) or (i1 < want[-1] + 1 <= i2):
            hidx += list(range(j1, j2))
        elif i1 == i2 and want[0] <= i1 <= want[-1] + 1:
            hidx += list(range(j1, j2))  # an insertion inside the slot
    hn = [h for h in hidx if h in hspans]
    if not hn:
        return "", {"aligned": True, "n_hyp_runs": 0}
    t0, t1 = hspans[min(hn)][0], hspans[max(hn)][1]
    # homophone tokens (to too for fore won ate) touching the span and NOT aligned to any reference word: kept apart as rescue candidates (score_chunk tries them only when the plain span is not exact)
    ext_l = ext_r = ""
    k = t0 - 1
    chain = []
    while k >= 0 and (ht[k].kind == "p" or (ht[k].text in HOMO and ht[k].kind == "x" and htmap.get(k) not in equal_h)):
        if ht[k].kind != "p":
            chain.insert(0, ht[k].text)
        k -= 1
    ext_l = " ".join(chain)  # the WHOLE adjacent chain, all or nothing ('four four for for' must not be rescued by its first 'for')
    k = t1
    chain = []
    while k < len(ht) and (ht[k].kind == "p" or (ht[k].text in HOMO and ht[k].kind == "x" and htmap.get(k) not in equal_h)):
        if ht[k].kind != "p":
            chain.append(ht[k].text)
        k += 1
    ext_r = " ".join(chain)
    # words between the first and last N run belong to the span (self-correction connectors)
    return " ".join(t.text for t in ht[t0:t1]), {"aligned": True, "n_hyp_runs": len(hn), "tok_range": (t0, t1), "ext_l": ext_l, "ext_r": ext_r}


def span_text_from_tokens(ht, t0, t1):
    return " ".join(t.text for t in ht[t0:t1])


def score_chunk(ref_chunk_text, slot_char_range, slot_truth, hyp_text, mode="digits", expected_markers=None):
    """one slot of a chunk: locate, then score. slot_truth = str (single value) or list[str] (self-correction sequence)."""
    mk = marker_report(hyp_text, expected_markers)  # raw marker output of the whole hypothesis chunk, reported separately
    hyp_text, _ = strip_markers(hyp_text)
    span, info = locate_span(ref_chunk_text, slot_char_range, hyp_text, mode)
    if isinstance(slot_truth, (list, tuple)):
        r = score_sequence(list(slot_truth), span, mode)
    else:
        r = score_value(slot_truth, span, mode)
        if not r["exact"] and (info.get("ext_l") or info.get("ext_r")):  # rescue only: a misheard last / first digit word ('four four for') may complete the exact value, never create a false one
            for cand in ((info["ext_l"] + " " + span).strip(), (span + " " + info["ext_r"]).strip(), (info["ext_l"] + " " + span + " " + info["ext_r"]).strip()):
                r2 = score_value(slot_truth, cand, mode)
                if r2["exact"]:
                    r, span = r2, cand
                    r["rescued_homophone"] = True
                    break
    r["span"] = span
    r["markers"] = mk
    r["aligned"] = info.get("aligned", False)
    return r


# ------------------------------------------------------------------------------------------------------------------------------ other number forms: canonical token form
_SCI = [(re.compile(r"\b(?:times|x|×)\s*(?:ten|10)\s*(?:to the|\^)?\s*(?:power of\s*)?(?:the\s*)?(minus|negative|-)\s*(\w+)\b"), r"e-\2"),
        (re.compile(r"\b(?:times|x|×)\s*(?:ten|10)\s*(?:to the|\^)?\s*(?:power of\s*)?(\w+)\b"), r"e\1")]
_OTHER_SUBS = [(r"\bhundred\b", "hundred"), (r"\bstroke\b|\bslash\b|/", " stroke "), (r"\bdecimal\b|\bdot\b|\bpoint\b", " point "), (r"\bnegative\b", "minus"),
               (r"\b(nine) (one) (one)\b", "911"), (r"\bsevenths?\b", "7th")]


ORDW = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
SUPER = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b", "01234567" + "89-")
_NUM = r"(\d+(?:\.\d+)?)"


def _sci(s):
    """scientific notation / powers -> 'MeK' | '10^K' | 'B^K' (digit forms; run before and after the word conversion)"""
    s = re.sub(_NUM + r" ?(?:times|x|\*|by) ?10 ?(?:\^|\*\*|to the power of|to the|power) ?(-|minus|negative)? ?(\d+)(?:th|st|nd|rd)?\b", lambda m: f"{m.group(1)}e{'-' if m.group(2) else ''}{m.group(3)}", s)
    s = re.sub(_NUM + r"e ?(-?\d+)\b", lambda m: f"{m.group(1)}e{m.group(2)}", s)
    s = re.sub(r"(?<![\d.])10 ?(?:\^|\*\*|to the power of|to the) ?(-|minus|negative)? ?(\d+)(?:th|st|nd|rd)?\b", lambda m: f"10^{'-' if m.group(1) else ''}{m.group(2)}", s)
    s = re.sub(r"(?<![\d.^])(\d+) ?(?:\^|\*\*|to the power of) ?(\d+)\b", lambda m: f"{m.group(1)}^{m.group(2)}", s)
    return s


def segment_run(ws):
    """number words -> canonical TOKENS that keep the spoken structure: consecutive single-digit words (and oh / zero) merge into ONE digit string ('nine one one' -> '911', 'one nine two' -> '192'),
    every other valid group is its own token ('nine eleven' -> '9 11', 'twenty four seven' -> '24 7', 'three sixty five' -> '3 65')."""
    out, digs, i, n = [], "", 0, len(ws)
    while i < n:
        if ws[i] in ZERO_W or ws[i] == "zero":
            digs += "0"
            i += 1
            continue
        best = None
        for j in range(n, i, -1):
            g = parse_group(ws[i:j])
            if g is not None:
                best = (j, g)
                break
        if best is None:
            if digs:
                out.append(digs)
                digs = ""
            out.append(ws[i])
            i += 1
            continue
        j, g = best
        if j == i + 1 and ws[i] in ONES and ONES[ws[i]] < 10:
            digs += str(g)
        else:
            if digs:
                out.append(digs)
                digs = ""
            out.append(str(g))
        i = j
    if digs:
        out.append(digs)
    return out


def canon_other(text, scale=False):
    """text -> canonical token string for other-form matching. Number-word runs -> digit strings (a run that is not one number is read digit by digit), ordinals -> digits, 'point' / 'dot' / 'decimal' -> '.',
    'stroke' / 'slash' -> '/', scientific notation -> 'MeK', powers -> 'B^K' / '10^K', lakh / crore (also 'N crore M lakh', Indian grouping '2,50,000') -> the integer value,
    niner / fife / zed / zee -> nine / five / z, sentence periods and punctuation dropped, 'x' between digits -> 'by' (N by M)."""
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"[\u2070-\u207b\u00b9\u00b2\u00b3]+", lambda m: "^" + m.group(0).translate(SUPER), t)
    t = re.sub(r"\bniner\b", "nine", t)
    t = re.sub(r"\bfife\b", "five", t)
    t = re.sub(r"\b(zed|zee)\b", "z", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"(?<=[a-z]{2})-(?=[a-z0-9])|(?<=\d)-(?=[a-z]{2})", " ", t)  # twenty-four seven, nine-to-five, Catch-22, 7-Eleven (keeps e-5 / 10^-5)
    t = t.replace("\u00d7", " x ").replace("\u2044", "/").replace("\u2212", "-").replace("\u20b9", " ")
    t = _sci(t)
    t = t.replace("*", " star ").replace("+", " plus ").replace(":", " : ")  # keypad / country-code symbols and colons are TOKENS (dropping them must not match)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " by ", t)
    t = re.sub(r"[^a-z0-9.^/:\- ']+", " ", t.replace("/", " / "))
    t = re.sub(r"(?<=\d)-(?=\d)", " ", t)  # 9-5 -> 9 5, but keep e-5 / ^-5 below
    t = re.sub(r"(?<![\d.e^])-", " - ", t)
    toks = t.split()
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        if w in ORDW and toks[max(0, i - 2):i] == ["to", "the"]:  # 'ten to the sixth' only; an ordinary 'first' stays a word
            out.append(str(ORDW[w]))
            i += 1
        elif w in NUMW and w != "and":
            j = i
            while j < len(toks) and (toks[j] in NUMW or (toks[j] == "and" and j + 1 < len(toks) and toks[j + 1] in NUMW and j > i and toks[j - 1] in ("hundred", "thousand"))):
                j += 1
            out.extend(segment_run(toks[i:j]))
            i = j
        else:
            out.append(w)
            i += 1
    s = " ".join(out)
    s = re.sub(r"(?<=\d) \. (?=\d)", ".", s)
    s = re.sub(r"\b(point|dot|decimal)\b", ".", s)
    s = re.sub(r"(?<=\d) \. (?=\d)", ".", s)
    s = re.sub(r"\b(stroke|slash)\b", "/", s)
    s = re.sub(r"\b(minus|negative)\b", "-", s)
    s = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"(?<=\d) - (?=\d)", " ", s)
    s = re.sub(r"to the - (\d)", r"to the minus \1", s)
    s = _sci(s.replace("minus", "-") if False else s)
    s = re.sub(_NUM + r" ?(?:times|x|by) ?10 ?(?:to the)? ?- ?(\d+)", r"\1e-\2", s)
    s = re.sub(_NUM + r" ?(?:times|x) ?10 ?(?:to the)? ?(\d+)", r"\1e\2", s)
    s = re.sub(r"(?<![\d.])10 (?:to the)? ?- ?(\d+)\b", r"10^-\1", s)
    s = re.sub(r"(?<![\d.^])10 to the (\d+)\b", r"10^\1", s)
    s = re.sub(r"(\d+) to the power of (\d+)", r"\1^\2", s)
    s = re.sub(_NUM + r" ?crore ?" + _NUM + r" ?lakh", lambda m: str(int(round(float(m.group(1)) * 10 ** 7 + float(m.group(2)) * 10 ** 5))), s)
    s = re.sub(_NUM + r" ?(lakhs?|crores?)\b", lambda m: str(int(round(float(m.group(1)) * (10 ** 5 if m.group(2).startswith("lakh") else 10 ** 7)))), s)
    if scale:  # value reading only: '4.5 million' = 'forty five lakh' (flagged by the caller)
        s = re.sub(_NUM + r" ?(million|billion|thousand)\b", lambda m: str(int(round(float(m.group(1)) * {"million": 10 ** 6, "billion": 10 ** 9, "thousand": 10 ** 3}[m.group(2)]))), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUMLIKE = re.compile(r"^[\d.^e/-]*\d[\d.^e-]*$")


def _find_bounded(h, a):
    """token list a as an EXACT run inside token list h whose flanks are not number-like when the run's own edge is number-like (no '7 6 5 4 3' inside '7 6 5 4 3 2')"""
    n = len(a)
    for i in range(0, len(h) - n + 1):
        if h[i:i + n] == a:
            if _NUMLIKE.match(a[0]) and i > 0 and _NUMLIKE.match(h[i - 1]):
                continue
            if _NUMLIKE.match(a[-1]) and i + n < len(h) and _NUMLIKE.match(h[i + n]):
                continue
            return True
    return False


def score_other(accept, hyp_chunk_text):
    """accept = list of surface forms (canonical forms are derived with canon_other). A hit = the canonical token run of one accept form occurs EXACTLY (token equality, number-like flanks refused)
    in the canon of the hypothesis chunk. Counted repeat markers attached to a LETTER ('one\u00b9') are stripped first; markers on digits (powers '7\u2077', '10\u207b\u2075') are kept.
    Fallback (flag value_read=True): scale words as values ('4.5 million' = 'forty five lakh')."""
    hyp = re.sub(r"(?<=[A-Za-z])(" + "|".join(re.escape(m) for m in MARKS) + ")", "", hyp_chunk_text)  # a mark glued to a WORD; powers (7\u2077) are not marks
    h = canon_other(hyp).split()
    for a in accept:
        at = canon_other(a).split()
        if at and _find_bounded(h, at):
            return {"exact": True, "cls": "exact", "matched": a, "value_read": False}
    hv = canon_other(hyp, scale=True).split()
    for a in accept:
        at = canon_other(a, scale=True).split()
        if at and _find_bounded(hv, at):
            return {"exact": True, "cls": "exact", "matched": a, "value_read": True}
    return {"exact": False, "cls": "other_value", "matched": None, "value_read": False}


if __name__ == "__main__":
    for t in sys.argv[1:]:
        print(t, "->", digit_candidates(t), "| joined", sorted(joined_candidates(t))[:5])
