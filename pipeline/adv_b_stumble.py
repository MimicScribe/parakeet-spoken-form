"""Adversarial pack, family 2: stumbles and disfluency inside numbers (en es fr).

Restarts mid-number ("thirty thirty five"), repeated words ("about about thirty five"), fillers inside numbers ("two thousand uh
twenty three"), self-corrections ("fifteen sorry fifty"), trailing off and resuming ("two hundred hang on two hundred sixty four"),
false starts on the unit ("five per five percent"), a digit run corrected ("four one five no wait four one six").
Labels are VERBATIM (every word said, both numbers of a correction kept: ambig `disfluency_verbatim`); each number reading
inside the stumble is its own span. English fillers are "uh" / "um" (the judges hear them); es / fr use only fillers that are
real words the judges can verify (es "este", "o sea", "pues"; fr "ben", "bon", "alors"): an "eh" / "euh" is dropped by every
transcriber (tier A finding). `synth` carries "..." and commas for the pause. Pure functions."""
import random

import adv_b_words as W
from edge_a_content import Piece

AMB = ("disfluency_verbatim",)

FIX_MARK = {
    "en": ["sorry", "I mean", "no wait", "or rather", "make that", "scratch that", "correction", "actually", "no no"],
    "es": ["perdón", "digo", "o sea", "mejor dicho", "no espera", "quiero decir", "bueno"],
    "fr": ["pardon", "je veux dire", "enfin", "plutôt", "non attends", "ou plutôt", "c'est à dire"],
}
FILLERS = {"en": ["uh", "um", "uh"], "es": ["este", "o sea", "pues", "este"], "fr": ["ben", "bon", "alors", "ben"]}
DETS = {"en": ["about", "around", "like", "over", "nearly", "just", "up to"],
        "es": ["unos", "como", "casi", "hasta", "por", "de", "más de"],
        "fr": ["environ", "à", "de", "pour", "presque", "plus de", "jusqu'à"]}
INTERRUPT = {"en": ["hang on", "wait", "let me think", "sorry where was I", "let me check", "hold on"],
             "es": ["espera", "a ver", "déjame pensar", "perdón dónde estaba", "un momento"],
             "fr": ["attends", "voyons", "laisse moi réfléchir", "pardon où j'en étais", "un instant"]}
PCT = {"en": "percent", "es": "por ciento", "fr": "pour cent"}


def _c(n, lang):
    return W.card(int(n), lang)


def prefixes(n: int, lang: str) -> list:
    """[(v, words)]: leading parts v < n whose cardinal is a token prefix of n's cardinal (30 of 35, 200 of 235, 2000 of 2023)."""
    tn = _c(n, lang).split()
    out = []
    for v in sorted({n - n % 10, n - n % 100, n - n % 1000}):
        if 0 < v < n:
            tv = _c(v, lang).split()
            if tn[:len(tv)] == tv:
                out.append((v, " ".join(tv)))
    return out


def splits(n: int, lang: str) -> list:
    """[(v, r)]: n = v + r where card(v) + card(r) is exactly card(n) (a filler can sit at the join): 2000 + 23, 200 + 35, 20 + 3."""
    tn = _c(n, lang).split()
    out = []
    for v in sorted({n - n % 10, n - n % 100, n - n % 1000}):
        r = n - v
        if 0 < v < n and r > 0 and _c(v, lang).split() + _c(r, lang).split() == tn:
            out.append((v, r))
    return out


def _num_keys(lang: str, which: str) -> list:
    """Number keys of a stumble construction (strings). es / fr: no value ending in 1 (un / uno apocope and agreement)."""
    ok = (lambda n: n % 10 != 1) if lang in ("es", "fr") else (lambda n: True)
    if which == "prefix":      # >= 2 tokens and a valid prefix
        pool = [n for n in list(range(22, 100)) + list(range(110, 1000, 5)) + [1200, 1500, 2500, 3500, 4200, 5600, 7500] if ok(n)]
        return [str(n) for n in pool if len(_c(n, lang).split()) >= 2 and prefixes(n, lang)]
    if which == "split":
        pool = [n for n in list(range(22, 100)) + list(range(105, 1000, 5)) + list(range(1005, 5000, 55)) + [2020, 2021, 2022, 2023, 2024, 2025, 2026] if ok(n)]
        return [str(n) for n in pool if splits(n, lang) and not (n % 100 == 0)]
    if which == "any":
        return [str(n) for n in list(range(12, 100)) + list(range(100, 1000, 25)) if ok(n)]
    raise ValueError(which)


def near_pairs(lang: str) -> list:
    """(a, b) near-miss pairs a self-correction swaps: teen / ty confusions, swapped digits, neighbouring hundreds."""
    out = []
    for a, b in [(15, 50), (13, 30), (16, 60), (14, 40), (17, 70), (18, 80), (19, 90), (22, 12), (30, 13), (60, 16), (25, 52), (45, 54),
                 (33, 43), (27, 72), (38, 83), (36, 63), (18, 19), (24, 34), (56, 65), (14, 15), (80, 18), (90, 19), (70, 17),
                 (115, 150), (213, 230), (340, 314), (260, 216), (418, 480), (350, 315), (175, 157), (230, 223), (480, 418),
                 (550, 515), (612, 662), (700, 770), (320, 330), (240, 250), (125, 135), (410, 420), (360, 350)]:
        if lang in ("es", "fr") and (a % 10 == 1 or b % 10 == 1):
            continue
        out.append((a, b))
    return out


def _span(kind, n, lang):
    return (kind, str(n), _c(n, lang))


# ------------------------------------------------------------------------------------------------ makers (key, rng) -> Piece
def restart_prefix(lang):
    def make(k, rng):
        n = int(k)
        v, pw = rng.choice(prefixes(n, lang))
        full = _c(n, lang)
        lab = f"{pw} {full}"
        return _mk(lab, f"{pw}... {full}", [("int", str(v), pw), ("int", k, full)], rng)
    return make


def restart_scale(lang):
    def make(k, rng):                          # k = "500": "five hundred five hundred thousand"
        n = int(k)
        base = _c(n, lang)
        big = _c(n * 1000, lang)
        lab = f"{base} {big}"
        return _mk(lab, f"{base}... {big}", [("int", k, base), ("int", str(n * 1000), big)], rng)
    return make


def repeat_det(lang):
    def make(k, rng):
        d = rng.choice(DETS[lang])
        num = _c(int(k), lang)
        lab = f"{d} {d} {num}"
        return _mk(lab, f"{d}... {d} {num}", [("int", k, num)], rng, edge_all=True)
    return make


def filler_mid(lang):
    def make(k, rng):
        n = int(k)
        v, r = rng.choice(splits(n, lang))
        f = rng.choice(FILLERS[lang])
        a, b = _c(v, lang), _c(r, lang)
        lab = f"{a} {f} {b}"
        return _mk(lab, f"{a}... {f}... {b}", [("int", str(v), a), ("int", str(r), b)], rng)
    return make


def self_fix(lang):
    def make(k, rng):
        a, b = map(int, k.split("-"))
        m = rng.choice(FIX_MARK[lang])
        wa, wb = _c(a, lang), _c(b, lang)
        lab = f"{wa} {m} {wb}"
        return _mk(lab, f"{wa}, {m}, {wb}", [("int", str(a), wa), ("int", str(b), wb)], rng)
    return make


def trail_resume(lang):
    def make(k, rng):
        n = int(k)
        v, pw = rng.choice(prefixes(n, lang))
        full = _c(n, lang)
        it = rng.choice(INTERRUPT[lang])
        lab = f"{pw} {it} {full}"
        return _mk(lab, f"{pw}... {it}... {full}", [("int", str(v), pw), ("int", k, full)], rng)
    return make


# pilot gate (10-19): "kilo" -> "kilos" in every transcriber (the judges cannot confirm it) and the unit sat before a second noun in the
# count frames; only the fragments a transcriber keeps are used, in frames that end on the unit (cat "ufs")
UNIT_PAIRS = {
    "en": [("per", "percent", "pct"), ("dollar", "dollars", "cur"), ("per", "percent", "pct")],
    "es": [("por", "por ciento", "pct")],
    "fr": [("pour", "pour cent", "pct")],
}


def unit_false_start(lang):
    def make(k, rng):
        n = int(k)
        w = _c(n, lang)
        frag, unit, kind = rng.choice(UNIT_PAIRS[lang])
        if frag == "dollar" and n == 1:
            frag, unit = "dollar", "dollar"
        lab = f"{w} {frag} {w} {unit}"
        wr = {"pct": "%", "unit": " km", "cur": " dollars"}[kind]
        return _mk(lab, f"{w} {frag}... {w} {unit}", [("int", k, w), (kind, f"{n}{wr}", f"{w} {unit}")], rng)
    return make


def digit_fix(lang):
    def make(k, rng):
        a, b = k.split("-")
        m = rng.choice(FIX_MARK[lang])
        wa, wb = W.digits(a, lang, "oh"), W.digits(b, lang, "oh")
        lab = f"{wa} {m} {wb}"
        return _mk(lab, f"{wa}, {m}, {wb}", [("code", a, wa), ("code", b, wb)], rng)
    return make


def digit_pairs() -> list:
    """3-4 digit code pairs (a, b) that differ in one digit, as strings 'a-b'."""
    rng = random.Random("advB|digitfix")
    out, seen = [], set()
    while len(out) < 80:
        n = rng.choice([3, 3, 4])
        a = [rng.randint(0, 9) for _ in range(n)]
        if a[0] == 0:
            continue
        b = list(a)
        i = rng.randrange(n)
        b[i] = (b[i] + rng.randint(1, 8)) % 10
        k = "".join(map(str, a)) + "-" + "".join(map(str, b))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _mk(label, synth, nums, rng, edge_all=False):
    """Piece with a `synth` text (pauses) ; spans in order (every reading is a substring of the label, in order)."""
    p = Piece(label, "", [(kind, written, reading) for kind, written, reading in nums], AMB)
    p.synth = synth
    p.whole_edge = True       # the edge phrase is the whole piece
    return p


# ------------------------------------------------------------------------------------------------ controls (fluent)
def c_prefix(lang):
    return lambda k, rng: _plain(k, lang)


def _plain(k, lang):
    w = _c(int(k), lang)
    p = Piece(w, "", [("int", k, w)], ())
    p.synth = w
    return p


def c_selffix(lang):
    return lambda k, rng: _plain(k.split("-")[1], lang)


def c_digits(lang):
    def mk(k, rng):
        w = W.digits(k.split("-")[1], lang, "oh")
        p = Piece(w, "", [("code", k.split("-")[1], w)], ())
        p.synth = w
        return p
    return mk


def c_unit(lang):
    def mk(k, rng):
        w = _c(int(k), lang)
        lab = f"{w} {PCT[lang]}"
        p = Piece(lab, "", [("pct", f"{k}%", lab)], ())
        p.synth = lab
        return p
    return mk


def c_scale(lang):
    def mk(k, rng):
        n = int(k) * 1000
        w = _c(n, lang)
        p = Piece(w, "", [("int", str(n), w)], ())
        p.synth = w
        return p
    return mk
