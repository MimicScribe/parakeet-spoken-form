#!/usr/bin/env python3
"""numread: one number reader + normaliser for scoring spoken-form transcripts.

Goal: score a hypothesis against a reference by MEANING (digits vs words irrelevant; no dropped, mangled or changed
numbers), never by surface form.

What it is
  base reader   number_outcomes.py + score_values.py + group_read.py (vendored under reader/) and the pipeline/ml_* form tables
                of this repo, pinned by sha256 in numread_pins.json and VERIFIED at import (a changed file refuses to load).
                It reads half / scale words / decimals in 7 languages with word parsers and the others through the form tables.
  normaliser    normalize(text, lang): English colloquial forms ('double oh seven', 'a buck fifty', 'quarter past three') ->
                fair_reader.normalize (strict: thousands separators, spoken decimals, digit+scale words, digit runs,
                'twenty o five') -> decimal+scale ('two point seventeen billion') -> comma breaks between number words.
                fair_reader.py is pinned too.
  reads         reads(text, lang) -> [Read]: literal numbers per clause (sentence punctuation is a hard break), one value each.
  accept-set    want_from(reading, text, lang): from the SPOKEN WORDS (never from digit strings alone): single value, or the parts
                of a multi-number reading plus their concatenation / decimal join. 5 != 50 != 500 (no trailing-zero tolerance).
  classify      classify(want, hyp, lang, refcount) -> ok | dup | split | embedded | trunc | wrong | lost | unscorable.
                split = ADJACENT reads (same clause, no word between) whose digits concatenate to the wanted value
                ('twenty o five', 'one fifty eight').
  repeats       repeat_count(text, unit, lang): exact run length of `unit` inside adjacent digit reads ('five five five' = 3).
  stamp()       sha256 of numread.py, fair_reader.py, every pinned base file: print it in every result header.

Every scorer calls install() first so that `import number_outcomes` / `import score_values` anywhere resolve to the pinned files.
Environment: NUMREAD_BASE (reader tree, default ./reader), SPOKEN_FORM_PIPELINE (the ml_* tables, default <repo>/pipeline).
Tests: python3 -I numread_golden.py  (>= 150 pairs, must pass 100 %) and python3 -I test_numread.py.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import namedtuple
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
PINS_FILE = os.path.join(HERE, "numread_pins.json")
DEFAULT_BASE = os.path.join(HERE, "reader")
DEFAULT_PIPELINE = os.path.normpath(os.path.join(HERE, "..", "..", "pipeline"))


class PinError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_dir():
    return os.path.expanduser(os.environ.get("NUMREAD_BASE", DEFAULT_BASE))


def pipeline_dir():
    return os.path.expanduser(os.environ.get("SPOKEN_FORM_PIPELINE", DEFAULT_PIPELINE))


def load_pins():
    with open(PINS_FILE) as f:
        return json.load(f)


def _check(group, root, pins, got, key_prefix=""):
    for rel, want in pins.get(group, {}).items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise PinError(f"numread: pinned file missing: {p}")
        got[key_prefix + rel] = sha256_file(p)
        if got[key_prefix + rel] != want:
            raise PinError(f"numread: pinned file CHANGED: {rel} sha256 {got[key_prefix + rel][:12]} != pinned {want[:12]}")


def verify_pins(pins=None, base=None, pipeline=None):
    """-> {relpath: sha256}; raises PinError on any missing / changed pinned file.
    groups: base (relative to the reader tree), pipeline (the ml_* form tables, relative to the pipeline dir), repo (normaliser files next to numread.py)."""
    pins, base, pipeline = pins or load_pins(), base or base_dir(), pipeline or pipeline_dir()
    got = {}
    _check("base", base, pins, got)
    _check("pipeline", pipeline, pins, got, "pipeline:")
    _check("repo", HERE, pins, got, "repo:")
    return got


_N = _SV = _F = None
_installed = None


def _load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # before exec: the module may import itself by name
    spec.loader.exec_module(mod)
    return mod


def install():
    """Verify the pins and make `number_outcomes`, `score_values` (and `fair_reader`) resolve to the pinned files for the
    whole process. Idempotent. Returns the base reader module."""
    global _N, _SV, _F, _installed
    if _N is not None:
        return _N
    _installed = verify_pins()
    base = base_dir()
    pipe = pipeline_dir()
    os.environ["SPOKEN_FORM_PIPELINE"] = pipe  # number_outcomes reads the ml_* tables from here
    for p in (os.path.join(base, "pipeline"), os.path.join(base, "eval", "quality"), pipe):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, pipe)  # eval/quality first on the path, the reader's pipeline second, the form tables last
    sys.path.insert(0, os.path.join(base, "pipeline"))
    sys.path.insert(0, os.path.join(base, "eval", "quality"))
    for stale in ("number_outcomes", "score_values", "fair_reader"):
        sys.modules.pop(stale, None)
    _N = _load_as("number_outcomes", os.path.join(base, "eval", "quality", "number_outcomes.py"))
    _SV = _load_as("score_values", os.path.join(base, "eval", "quality", "score_values.py"))
    _F = _load_as("fair_reader", os.path.join(HERE, "fair_reader.py"))
    return _N


def N():
    return install()


def stamp():
    """Everything a result header must print: numread / normaliser / base-file shas."""
    install()
    base_sha = hashlib.sha256(json.dumps(_installed, sort_keys=True).encode()).hexdigest()
    return {"numread": sha256_file(os.path.abspath(__file__))[:12], "fair_reader": sha256_file(os.path.join(HERE, "fair_reader.py"))[:12],
            "base_tree": base_sha[:12], "base_dir": base_dir(), "number_outcomes": _installed["eval/quality/number_outcomes.py"][:12]}


def stamp_line():
    s = stamp()
    return f"reader numread@{s['numread']} fair_reader@{s['fair_reader']} base number_outcomes@{s['number_outcomes']} tree@{s['base_tree']}"


# ------------------------------------------------------------------------------------------------ canonical values
def dstr(v):
    """Decimal / number -> canonical digit string: '5', '0.5', '2170000000' (no exponent, no trailing zeros)."""
    d = Decimal(v)
    if d == d.to_integral_value():
        return str(int(d))
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


# ------------------------------------------------------------------------------------------------ normaliser
_DIG = "zero one two three four five six seven eight nine".split()
_UNITS12 = "one two three four five six seven eight nine ten eleven twelve".split()
_HOUR = {w: i + 1 for i, w in enumerate(_UNITS12)}
_BACK = {v: k for k, v in _HOUR.items()}
_MIN = {"five": 5, "ten": 10, "twenty": 20, "twenty five": 25}
_MW = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 35: "thirty five", 40: "forty", 45: "forty five", 50: "fifty", 55: "fifty five"}
_NUMW = set(_DIG) | set("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million".split())


def colloquial_en(t):
    """English colloquial number forms the base reader does not understand ('double oh seven', 'a buck fifty', 'quarter past three',
    'three nil', 'point four'). Applied to hypotheses and to key readings alike."""
    t = re.sub(r"[—–]", " ", t)
    t = re.sub(r"(?i)\b(double|triple)\s+(" + "|".join(_DIG + ["oh"]) + r")\b", lambda m: (m.group(2) + " ") * (2 if m.group(1).lower() == "double" else 3), t)
    w = t.split(" ")
    for _ in range(3):
        for i, x in enumerate(w):
            if x.lower() == "oh":
                prev = w[i - 1].lower().strip(",.") if i else ""
                nxt = w[i + 1].lower().strip(",.") if i + 1 < len(w) else ""
                nn = w[i + 2].lower().strip(",.") if i + 2 < len(w) else ""
                if nxt == "hundred" or (nxt in _DIG and nn == "hundred"):
                    continue
                if prev in _DIG or prev in ("oh", "zero") or (prev not in _NUMW and (nxt in _DIG or nxt in ("oh", "zero", "point"))):
                    w[i] = "zero"
    t = " ".join(w)
    t = re.sub(r"(?i)\boh\s+point\b", "zero point", t)
    t = re.sub(r"(?i)\bpoint\s+oh\b", "point zero", t)
    t = re.sub(r"(?i)\ba buck\b", "one dollar", t)
    t = re.sub(r"(?i)\bbucks?\b", "dollars", t)
    t = re.sub(r"(?i)\bgrand\b", "thousand", t)
    t = re.sub(r"\b(" + "|".join(sorted(_NUMW, key=len, reverse=True)) + r")\s+K\b", r"\1 thousand", t)
    t = re.sub(r"\bmil\b", "million", t, flags=re.I)
    t = re.sub(r"(?i)\ba couple (?:of )?(hundred|thousand|million)", r"two \1", t)
    t = re.sub(r"(?i)\bquarter past (" + "|".join(_UNITS12) + r")\b", r"\1 fifteen", t)
    t = re.sub(r"(?i)\bhalf past (" + "|".join(_UNITS12) + r")\b", r"\1 thirty", t)
    t = re.sub(r"(?i)\bhalf (" + "|".join(_UNITS12) + r")\b", r"\1 thirty", t)
    t = re.sub(r"(?i)\bquarter to (" + "|".join(_UNITS12) + r")\b", lambda m: f"{_BACK[(_HOUR[m.group(1).lower()] - 2) % 12 + 1]} forty five", t)
    t = re.sub(r"(?i)\b(five|ten|twenty five|twenty) past (" + "|".join(_UNITS12) + r")\b", lambda m: f"{m.group(2)} {m.group(1)}", t)
    t = re.sub(r"(?i)\b(five|ten|twenty five|twenty) to (" + "|".join(_UNITS12) + r")\b(?! (?:percent|thousand|hundred|million|dollars))",
               lambda m: f"{_BACK[(_HOUR[m.group(2).lower()] - 2) % 12 + 1]} {_MW[60 - _MIN[m.group(1).lower()]]}", t)
    t = re.sub(r"(?i)\ban hour and a half\b", "one and a half hours", t)
    t = re.sub(r"(?i)\ba day and a half\b", "one and a half days", t)
    t = re.sub(r"(?i)\bhalf an hour\b", "zero point five hours", t)
    for den, n in (("thirds", "three"), ("fifths", "five"), ("sixths", "six"), ("eighths", "eight")):
        t = re.sub(r"(?i)\b(" + "|".join(_DIG[1:]) + ") " + den + r"\b", r"\1 over " + n, t)
    t = re.sub(r"(?i)\band three quarters\b", "point seven five", t)
    t = re.sub(r"(?i)\band a quarter\b", "point two five", t)
    t = re.sub(r"(?i)\bthree quarters\b", "zero point seven five", t)
    t = re.sub(r"(?i)\b(" + "|".join(_DIG[1:]) + r") nil\b", r"\1 zero", t)
    t = re.sub(r"(?i)\bnil\b", "zero", t)
    t = re.sub(r"(?i)\b(" + "|".join(_DIG[1:]) + r") all\b", r"\1 \1", t)
    w = t.split(" ")
    out = []
    for i, x in enumerate(w):  # a number that starts with "point": zero point ...
        if x.lower().strip(",.") == "point" and i + 1 < len(w) and re.match(r"(?i)(" + "|".join(_DIG) + r")\b", w[i + 1]) and not (out and out[-1].lower().strip(",.") in _NUMW):
            out.append("zero")
        out.append(x)
    return re.sub(r" {2,}", " ", " ".join(out))


_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
_SCALE_EXP = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12, "tausend": 3, "millionen": 6, "million_": 6, "milliarde": 9, "milliarden": 9, "mil": 3, "millón": 6, "millones": 6,
              "mille": 3, "millions": 6, "milliard": 9, "milliards": 9, "millions_": 6}
_POINT = {"en": "point", "de": "komma", "es": "coma", "fr": "virgule", "it": "virgola", "pt": "vírgula", "nl": "komma"}


def _is_num_word(tok, lang, weak_ok=False):
    t = tok.strip(".,;:!?")
    if not t or any(ch.isdigit() for ch in t) or not re.fullmatch(_WORD, t):
        return False
    try:
        return any(n.vals for n in N().extract_numbers(t, lang, keep_weak=weak_ok))
    except Exception:  # noqa: BLE001
        return False


def _single_int(words, lang):
    """the whole number a run of number words reads as, when it is exactly one integer; else None"""
    try:
        nums = [n for n in N().extract_numbers(" ".join(words), lang, keep_weak=True) if n.vals]
    except Exception:  # noqa: BLE001
        return None
    if len(nums) != 1 or nums[0].start != 0 or nums[0].end != len(words) - 1:
        return None
    v = min(nums[0].vals)
    return int(v) if v == int(v) else None


def fix_decimal_scale(text, lang):
    """'two point seventeen billion' -> '2.17 billion' (the base reader returned 2.17000000000): words INT <point> FRAC <scale>
    where FRAC is digit words ('one two' -> 12) or one number word ('seventeen' -> 17). The rewritten digits+scale form is read
    exactly by the base reader. A text without such a phrase is returned unchanged."""
    pt = _POINT.get(lang)
    if not pt:
        return text
    toks = text.split(" ")
    out, i = [], 0
    scales = {w for w in _SCALE_EXP if not w.endswith("_")}
    while i < len(toks):
        if toks[i].lower().strip(",.;:!?") == pt:
            j = i + 1
            while j < len(toks) and _is_num_word(toks[j], lang, weak_ok=True) and toks[j].lower().strip(",.;:!?") not in scales:
                j += 1
            if j > i + 1 and j < len(toks) and toks[j].lower().strip(",.;:!?") in scales:
                k = len(out)
                while k > 0 and _is_num_word(out[k - 1], lang, weak_ok=True) and out[k - 1][-1] not in ",.;:!?":
                    k -= 1
                left = [x for x in out[k:]]
                lv = _single_int([x.strip(",.;:!?") for x in left], lang) if left else 0
                fw = [x.strip(",.;:!?").lower() for x in toks[i + 1:j]]
                dw = {w: str(n) for n, w in enumerate(_DIG)} if lang == "en" else {}
                if all(w in dw for w in fw):
                    frac = "".join(dw[w] for w in fw)
                else:
                    fv = _single_int(fw, lang)
                    frac = None if fv is None else str(fv)
                if lv is not None and frac is not None:
                    mark = "." if lang == "en" else ","
                    del out[k:]
                    out.append(f"{lv}{mark}{frac}")
                    i = j
                    continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


def comma_breaks_words(text, lang):
    """A comma + space between two number WORDS is a hard number break (';'): the reader ignores punctuation inside word runs
    ('trois cent quarante, trois' would read 343) but splits '340, 3' (digits). Same text, same reading."""
    out, last = [], 0
    for m in re.finditer(rf"({_WORD}),(?=\s+({_WORD}))", text):
        if _is_num_word(m.group(1), lang) and _is_num_word(m.group(2), lang):
            out += [text[last:m.end(1)], ";"]
            last = m.end(1) + 1
    return "".join(out + [text[last:]])


def normalize(text, lang):
    """The one normaliser (strict; idempotent on its own output). Apply to hypotheses AND to key readings."""
    if not text:
        return text or ""
    N()  # install
    t = text
    if lang == "en":
        t = colloquial_en(t)
        # tens + unit is ONE number ('thirty three'): hyphenate so the digit-run rule never takes its unit word into a run ('thirty three oh three' = 30 + 3303, not 30303)
        t = re.sub(r"(?i)\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[ ]+(one|two|three|four|five|six|seven|eight|nine)\b", r"\1-\2", t)
    t = _F.normalize(t, lang)
    t = fix_decimal_scale(t, lang)
    return comma_breaks_words(t, lang)


# ------------------------------------------------------------------------------------------------ reads
Read = namedtuple("Read", "clause start end val alts form weak toks")  # val: canonical digit string; alts: all canonical values; start/end: token idx in the clause; toks: the clause token list (shared)


def _clauses(t):
    """clause split: sentence punctuation + whitespace (and the ';' number breaks) are hard breaks"""
    return [c for c in re.split(r"(?<=[.!?;])\s+|\s*;\s*", t) if c.strip()]


def _blob_clean(text):
    """The base reader raises on separator soup ('1.,2', '3..4,'): drop doubled / trailing marks inside digit runs."""
    t = re.sub(r"(?<=\d)[.,]{2,}(?=\d)", ".", text)
    t = re.sub(r"(?<=\d)[.,]+(?=\s|$)", "", t)

    def split_bad(m):
        tok = m.group(0)
        try:
            N().extract_numbers(tok, "en", keep_weak=True)
            return tok
        except Exception:  # noqa: BLE001
            return re.sub(r"[.,]", " ", tok)

    return re.sub(r"\d[\d.,]*\d", split_bad, t)


def _prep(clause, lang):
    t = clause
    for w, rep in _SV.NOON.get(lang, {}).items():
        t = re.sub(rf"(?<![\w-]){re.escape(w)}(?![\w-])", rep, t, flags=re.IGNORECASE)
    return t


def _reader_tokens(text, lang):
    """The token list the base reader indexes Num.start / Num.end into (same pre-processing as the base reader's own scorer)."""
    n = N()
    try:
        t = n._romans(text)
        t = n._rel_clocks(lang, n._fold(t))
        for pat, rep in n._HALF.get(lang, ()):
            t = re.sub(pat, rep, t)
        if lang == "en":
            t = n._EN_A_CUR.sub("one ", t)
            t = re.sub(r"(?<![\d.,])(\d+) (thousand|million|billion|trillion)s?\b",
                       lambda m: str(int(m.group(1)) * 10 ** {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}[m.group(2)]), t)
        if lang in n._PCT:
            t = n._pct_words(lang, t)
        t = re.sub(r"(\d):00\b", r"\1", t)
        for _ in range(3):
            t = re.sub(r"(?<![\d.,])(\d{1,3}) (\d{3})(?!\d)(?![.,]\d)", "\\1\u00a0\\2", t)
        return n._TOKEN.findall(t)
    except Exception:  # noqa: BLE001
        return []


def _extract(clause, lang):
    t = _prep(clause, lang)
    try:
        return N().extract_numbers(t, lang, keep_weak=True)
    except Exception:  # noqa: BLE001
        try:
            return N().extract_numbers(_blob_clean(t), lang, keep_weak=True)
        except Exception:  # noqa: BLE001
            out = []
            for w in t.split():
                try:
                    out += N().extract_numbers(w, lang, keep_weak=True)
                except Exception:  # noqa: BLE001
                    pass
            return out


def reads(text, lang, normalized=False):
    """Literal numbers of `text`, in order: [Read]. Normalises first unless `normalized`."""
    N()
    t = text or ""
    if not normalized:
        t = normalize(t, lang)
    out = []
    for ci, clause in enumerate(_clauses(t)):
        toks = None
        for n in sorted(_extract(clause, lang), key=lambda n: (n.start, n.end)):
            if not n.vals:
                continue
            if toks is None:
                toks = _reader_tokens(_prep(clause, lang), lang)
            vals = sorted({dstr(v) for v in n.vals}, key=lambda s: (len(s), s))
            out.append(Read(ci, n.start, n.end, dstr(min(n.vals)), tuple(vals), n.form, bool(n.weak), toks))
    return out


# ------------------------------------------------------------------------------------------------ accept-set
class Want:
    """What a key accepts. alts: single-read values (set of canonical strings); parts: ordered component values of a
    multi-number spoken reading (or None); joined: the components concatenated / decimal-joined (digit strings that an
    adjacent group of hypothesis reads may spell). conflict: the written form disagrees with the spoken words."""

    __slots__ = ("alts", "parts", "joined", "conflict", "primary")

    def __init__(self, alts, parts, joined, conflict, primary):
        self.alts, self.parts, self.joined, self.conflict, self.primary = frozenset(alts), parts, frozenset(joined), conflict, primary

    def __repr__(self):
        return f"Want(alts={sorted(self.alts)}, parts={self.parts}, joined={sorted(self.joined)}, conflict={self.conflict})"

    def empty(self):
        return not self.alts and not self.parts and not self.joined


def _digits_only(s):
    return re.sub(r"\D", "", s)


def want_from(reading, text=None, lang="en"):
    """Accept-set from the SPOKEN words `reading` (e.g. 'one hundred and sixty percent') and, optionally, the written form `text`
    ('160%'). Spoken reading decides; the written form adds its own reads only when its digits agree with the spoken
    components (same digits once separators are dropped), else conflict=True (a REFERENCE-suspect key) and it is ignored."""
    rr = reads(reading, lang)
    alts, joined, parts, conflict = set(), set(), None, False
    if len(rr) == 1:
        alts |= set(rr[0].alts)
    elif len(rr) > 1:
        parts = tuple(r.val for r in rr)
        j = "".join(parts)
        joined.add(j)
        if j.lstrip("0"):
            joined.add(j.lstrip("0"))
        alts |= joined
    if text:
        tr = reads(text, lang)
        spoken_digits = "".join(r.val for r in rr) if rr else ""
        t_digits = _digits_only(text)
        agree = (not rr) or (_digits_only(spoken_digits) == t_digits) or (len(rr) == 1 and (set(tr[0].alts if tr else ()) & set(rr[0].alts)))
        if len(rr) > 1 and _digits_only(spoken_digits).lstrip("0") == t_digits.lstrip("0"):
            agree = True
        if len(rr) > 1 and len(tr) == 1 and "".join(r.val for r in rr).rstrip("0") == _digits_only(tr[0].val).rstrip("0") and False:
            agree = True
        if agree:
            if len(tr) == 1:
                alts |= set(tr[0].alts)
            elif len(tr) > 1 and parts is None:
                parts = tuple(r.val for r in tr)
        else:
            conflict = True
    primary = min(alts, key=lambda s: (len(s), s)) if alts else None
    return Want(alts, parts, joined, conflict, primary)


# ------------------------------------------------------------------------------------------------ classify
def _adjacent_chains(rr):
    """maximal chains of reads that are text-adjacent: same clause, next.start == prev.end + 1"""
    chains, cur = [], []
    for r in rr:
        if cur and r.clause == cur[-1].clause and r.start == cur[-1].end + 1:
            cur.append(r)
        else:
            if cur:
                chains.append(cur)
            cur = [r]
    if cur:
        chains.append(cur)
    return chains


SCALE_VALS = ("1000", "1000000", "1000000000", "1000000000000")
# words that may stand BETWEEN the parts of a clock / price / pair reading ('23 Uhr 34', '5 dollars 99', '3 and 30'); never between digits of a concatenation
CONNECT = {"uhr", "h", "heures", "heure", "horas", "hora", "ore", "ora", "uur", "godz", "and", "und", "et", "y", "e", "en", "oclock", "hours", "minutes", "minuten", "minutos", "minuti",
           "dollars", "dollar", "euros", "euro", "dolares", "pounds", "cents", "cent", "centavos", "centimes", "centimos"}


def _parts_hit(want, rr):
    """the ordered components of a multi-number spoken reading, as separate reads: adjacent, or with ONE CONNECT word between."""
    parts = want.parts
    if not parts or len(parts) < 2:
        return None
    for i in range(len(rr)):
        if rr[i].val != parts[0] and parts[0] not in rr[i].alts:
            continue
        k, ok = i, True
        for p in parts[1:]:
            if k + 1 >= len(rr):
                ok = False
                break
            a, b = rr[k], rr[k + 1]
            gap = b.start - a.end - 1
            same = a.clause == b.clause
            if not (same and (gap == 0 or (gap == 1 and a.toks and 0 <= a.end + 1 < len(a.toks) and a.toks[a.end + 1].lower() in CONNECT))) or (b.val != p and p not in b.alts):
                ok = False
                break
            k += 1
        if ok:
            return list(parts)
    return None


def _split_hit(want, rr, maxparts=8):
    """-> (kind, parts) when ADJACENT reads spell the wanted value as a grouped / digit-by-digit / year-style reading, else None."""
    targets = set(want.alts) | set(want.joined)
    for chain in _adjacent_chains(rr):
        vals = [r.val for r in chain]
        for i in range(len(vals)):
            acc = ""
            for j in range(i, min(i + maxparts, len(vals))):
                acc += vals[j]
                if j > i and (acc in targets or any(w.startswith("0") and acc.lstrip("0") == w.lstrip("0") for w in targets)):
                    return "grouped", vals[i:j + 1]
                # digit-by-digit then a scale word ('two five zero million' = 250000000)
                if j > i and j + 1 < len(vals) and vals[j + 1] in SCALE_VALS and (acc + vals[j + 1][1:]) in targets:
                    return "grouped_scale", vals[i:j + 2]
    ph = _parts_hit(want, rr)
    if ph:
        return "parts", ph
    return None


def _embedded(want, hyp, lang):
    """The wanted digits sit INSIDE a longer run of spoken single digit words (a phone number said digit by digit with no grouping:
    'cero tres cuatro seis dos cuatro dos uno ...' for key 03462). Same meaning, boundary unknowable: reported apart as `embedded`,
    never counted right and never silently wrong (READER-suspect for the attribution step)."""
    dmap = dict(_F._digit_word_map(lang))
    if lang == "en":
        dmap.update({"oh": "0", "o": "0"})
    if not dmap:
        return None
    import unicodedata
    fold = lambda w: unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode().lower()  # noqa: E731
    dmap = {fold(k): v for k, v in dmap.items()}
    targets = {w.lstrip("0") or w for w in (set(want.alts) | set(want.joined)) if w.isdigit() and len(w.lstrip("0")) >= 3}
    if not targets:
        return None
    for clause in _clauses(hyp or ""):
        run = ""
        for tok in re.findall(r"[^\W_]+|[.,;:!?]", clause):
            d = dmap.get(fold(tok))
            if d is not None:
                run += d
                continue
            if run and any(t in run and len(run) > len(t) for t in targets):
                return run
            run = ""
        if run and any(t in run and len(run) > len(t) for t in targets):
            return run
    return None


def classify(want, hyp, lang, refcount=1, adjacent=True):
    """-> (outcome, detail) with outcome in ok | dup | split | embedded | trunc | wrong | lost | unscorable.
    ok = exactly one read equals a wanted value (dup when more reads than `refcount` equal it: a repeated number);
    split = adjacent reads spell it (grouped / digit-by-digit / parts); embedded = it sits inside a longer spoken digit-by-digit run (not right,
    not wrong: boundary unknowable); trunc = a read is a strict prefix / suffix of it; lost = no number in the hypothesis region; wrong = numbers but none right."""
    if want.empty():
        return "unscorable", None
    rr = reads(hyp, lang)
    hits = [r for r in rr if set(r.alts) & want.alts]
    if hits:
        return ("dup" if len(hits) > refcount else "ok"), [h.val for h in hits]
    sp = _split_hit(want, rr) if adjacent else None
    if sp:
        return "split", sp
    emb = _embedded(want, hyp, lang)
    if emb:
        return "embedded", emb
    for W in sorted(want.alts, key=len):
        for r in rr:
            vs = r.val
            if vs.isdigit() and W.isdigit() and 1 <= len(vs) < len(W) and (W.startswith(vs) or W.endswith(vs)):
                return "trunc", vs
    if not rr:
        return "lost", None
    return "wrong", [r.val for r in rr]


def accepts(key, hyp, lang="en", written=None):
    """Is `hyp` a correct rendering of the spoken / written number(s) in `key`?  -> classify outcome ('ok' / 'split' = yes)."""
    return classify(want_from(key, written, lang), hyp, lang)[0]


def same_value(a, b, lang="en"):
    return accepts(a, b, lang) in ("ok", "split", "dup")


# ------------------------------------------------------------------------------------------------ repeats
def digit_chains(text, lang):
    """-> list of digit strings: the concatenated integer reads of each adjacent chain ('five five five' -> '555', '555' -> '555')"""
    out = []
    for chain in _adjacent_chains(reads(text, lang)):
        cur = ""
        for r in chain:  # a spoken multi-digit number ('fifty five') is a value, not repeated digits: it ends the chain
            if r.val.isdigit() and (len(r.val) == 1 or r.form == "digits"):
                cur += r.val
            else:
                if cur:
                    out.append(cur)
                cur = ""
        if cur:
            out.append(cur)
    return out


def repeat_count(text, unit, lang="en"):
    """Exact repeat count of `unit` (a digit string) as a contiguous block inside any adjacent chain: the largest k such that
    unit * k occurs. 'five five five' / '5 5 5' / '555' with unit '5' = 3; 'five five' = 2; '55' with unit '55' = 1.
    Counted over reads, so a grouped '55' and two spoken 'five five' are the same two repeats (form irrelevant)."""
    best = 0
    for d in digit_chains(text, lang):
        k = 1
        while unit * k in d:
            k += 1
        best = max(best, k - 1)
    return best


def run_lengths(text, lang="en"):
    """-> [(digit, run)] maximal runs of the same digit inside adjacent digit chains with run >= 2 (repeat detector)."""
    out = []
    for d in digit_chains(text, lang):
        for m in re.finditer(r"(\d)\1+", d):
            out.append((m.group(1), len(m.group(0))))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="numread: print the normalised text and reads of a string")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--stamp", action="store_true")
    ap.add_argument("text", nargs="?", default="")
    a = ap.parse_args()
    if a.stamp:
        print(json.dumps(stamp(), indent=1))
    if a.text:
        print(normalize(a.text, a.lang))
        print([(r.val, r.alts, r.form) for r in reads(a.text, a.lang)])
