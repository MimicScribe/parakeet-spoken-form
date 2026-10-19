"""Adversarial pack, family 1: phone numbers read in mixed styles (en es fr; de pt it nl smaller).

One number, many readings: all single digits; pairs ("twenty three twenty three"); mixed groups ("eight oh eight, nine
three six, twenty three twenty three"); double / treble / triple; "oh" vs "zero"; area-code grouping; country-code forms;
"one eight hundred". The label is exactly what is said (no digit, no hyphen, no punctuation inside a number); the
synthesiser gets the same words with commas between the groups (`synth`) so the groups are voiced as groups.

A value key is the digit string. Numbers of one kind (US, UK, ...) share ONE value universe and ONE split, so a number
held out (V_b) is held out of every reading style. Pure functions; tests in test_adv_b.py."""
import random

import adv_b_words as W

# ------------------------------------------------------------------------------------------------ value universes
US_AREA = ["212", "310", "415", "617", "808", "206", "305", "404", "512", "702", "646", "713", "202", "303", "503", "602",
           "312", "214", "415", "919", "720", "801", "907", "313", "757", "954", "480", "615", "704", "816"]


def _no00(s: str, aligned_from: int, size: int = 2) -> bool:
    """No aligned pair of `s` (from `aligned_from`, `size` digits) is "00"."""
    return all(s[i:i + size] != "00" for i in range(aligned_from, len(s) - size + 1, size))


def _rep(rng: random.Random, digs: list, kind: int) -> list:
    """Force a repeated digit (kind 2: a pair, 3: a triple) somewhere inside digs."""
    i = rng.randrange(0, len(digs) - kind + 1)
    for j in range(kind):
        digs[i + j] = digs[i]
    return digs


def universe(kind: str, n: int = 300, seed: int = 1019) -> list:
    """Digit strings of a number kind: us (10), uk_mob (11, 07...), uk_lon (11, 020...), es_mob (9), mx (10), fr (10),
    de (11), pt (11, Brazilian mobile), it (10), nl (10), tf_us (11, 1 8xx), tf_mx (12, 01 800), tf_fr (10, 0 800),
    tf_es? none. Deterministic. A third of the numbers hold a doubled digit, a sixth a tripled one."""
    rng = random.Random(f"advB|phone|{kind}|{seed}")
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < n * 400:
        guard += 1
        rep = rng.choice([0, 0, 2, 2, 3, 0])
        d = lambda lo=0: rng.randint(lo, 9)  # noqa: E731
        if kind == "us":
            tail = [d() for _ in range(7)]
            tail[0] = rng.randint(2, 9)
            if rep:
                tail[1:] = _rep(rng, tail[1:], rep)
            s = rng.choice(US_AREA) + "".join(map(str, tail))
            ok = _no00(s, 6)
        elif kind == "tf_us":
            tail = [d() for _ in range(7)]
            tail[0] = rng.randint(2, 9)
            if rep:
                tail[1:] = _rep(rng, tail[1:], rep)
            s = "1" + rng.choice(["800", "888", "877", "866", "855"]) + "".join(map(str, tail))
            ok = _no00(s, 7)
        elif kind == "uk_mob":
            tail = [d() for _ in range(9)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = "07" + "".join(map(str, [rng.choice([7, 8, 9, 5, 4, 3])] + tail[1:]))
            ok = _no00(s, 5)
        elif kind == "uk_lon":
            tail = [d() for _ in range(8)]
            tail[0] = rng.choice([3, 7, 8])
            if rep:
                tail = _rep(rng, tail, rep) if rep == 2 else tail
            s = "020" + "".join(map(str, tail))
            ok = _no00(s, 3)
        elif kind == "es_mob":
            tail = [d() for _ in range(8)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = rng.choice("67") + "".join(map(str, tail))
            ok = _no00(s, 3)
        elif kind == "mx":
            tail = [d() for _ in range(8)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = rng.choice(["55", "33", "81"]) + "".join(map(str, tail))
            ok = _no00(s, 2)
        elif kind == "tf_mx":
            tail = [d() for _ in range(7)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = "01800" + "".join(map(str, tail))
            ok = _no00(s, 5, 1) and "00" not in "".join(map(str, tail))
        elif kind == "fr":
            tail = [d() for _ in range(8)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = "0" + rng.choice("1234567") + "".join(map(str, tail)) if rng.random() < 0.7 else "0" + rng.choice("67") + "".join(map(str, tail))
            ok = _no00(s, 0)
        elif kind == "fr_be":     # at least two pairs in 70-79 / 90-99 so the Belgian / Swiss words occur
            pr = [rng.choice([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 80, 81, 82, 83, 84, 85]) if rng.random() < 0.6 else rng.randint(11, 69) for _ in range(4)]
            s = rng.choice(["02", "04", "03", "09"]) + "".join(f"{p:02d}" for p in pr)
            ok = sum(1 for p in pr if 70 <= p < 80 or p >= 90) >= 2 and _no00(s, 0)
        elif kind == "tf_fr":
            tail = [d() for _ in range(6)]
            s = "0800" + "".join(map(str, tail))
            ok = _no00(s, 4)
        elif kind == "de":
            tail = [d() for _ in range(8)]
            if rep:
                tail = _rep(rng, tail, rep)
            s = rng.choice(["0151", "0152", "0160", "0170", "0171", "0175", "0176"]) + "".join(map(str, tail))
            ok = _no00(s, 4)
        elif kind == "pt":     # Brazilian mobile: DD 9 XXXX XXXX; pair words 14, 16, 17, 19 differ by dialect: left out of the pairs
            tail = [d() for _ in range(8)]
            s = str(rng.choice([11, 21, 31, 41, 51, 61, 71, 81, 85, 91])) + "9" + "".join(map(str, tail))
            pr = [int(s[i:i + 2]) for i in (3, 5, 7, 9)]
            ok = _no00(s, 3) and not any(p in (14, 16, 17, 19) for p in pr + [int(s[:2])])
        elif kind == "it":
            tail = [d() for _ in range(7)]
            s = "3" + rng.choice("2349") + str(rng.randint(0, 9)) + "".join(map(str, tail))
            ok = _no00(s, 3)
        elif kind == "nl":
            tail = [d() for _ in range(8)]
            s = "06" + "".join(map(str, tail))
            ok = _no00(s, 2)
        else:
            raise ValueError(kind)
        if ok and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ captions
def _spaced(s: str, sizes: list) -> str:
    out, i = [], 0
    for z in sizes:
        out.append(s[i:i + z])
        i += z
    if i < len(s):
        out.append(s[i:])
    return " ".join(x for x in out if x)


def caption(kind: str, s: str, plus: str = "") -> str:
    """Digit caption of a number (groups separated by single spaces, as phonefmt writes them). `plus`: the country code of a
    "plus ..." reading (the caption then drops the national trunk zero)."""
    sizes = {"us": [3, 3, 4], "tf_us": [1, 3, 3, 4], "uk_mob": [5, 6], "uk_lon": [3, 4, 4], "es_mob": [3, 3, 3], "mx": [2, 4, 4],
             "tf_mx": [2, 3, 3, 4], "fr": [2, 2, 2, 2, 2], "fr_be": [2, 2, 2, 2, 2], "tf_fr": [4, 2, 2, 2], "de": [4, 8],
             "pt": [2, 5, 4], "it": [3, 7], "nl": [2, 8]}[kind]
    if not plus:
        return _spaced(s, sizes)
    national = {"uk_mob": [4, 6], "uk_lon": [2, 4, 4], "fr": [1, 2, 2, 2, 2], "de": [3, 8], "nl": [1, 8]}
    if kind in national:
        return "+" + plus + " " + _spaced(s[1:], national[kind])
    return "+" + plus + " " + _spaced(s, sizes)


# ------------------------------------------------------------------------------------------------ styles
def groups_join(groups: list) -> tuple:
    """(label words, synth words with commas between groups)."""
    groups = [g for g in groups if g]
    return " ".join(groups), ", ".join(groups)


def pairs_l(s: str, lang: str, zero: str = "plain") -> list:
    """Each aligned pair of `s` spoken as a pair, one group per pair (the synthesiser voices the pairs as pairs)."""
    assert len(s) % 2 == 0, s
    return [W.pair(s[i:i + 2], lang, zero) for i in range(0, len(s), 2)]


def pairs(s: str, lang: str, zero: str = "plain") -> str:
    return " ".join(pairs_l(s, lang, zero))


def _us_mixed(s, rng, zero="oh"):
    return [W.digits(s[:3], "en", zero), W.digits(s[3:6], "en", zero), *pairs_l(s[6:], "en", zero)]


def _us_triplet(s, rng):
    b = s[3:6]
    mid = W.digit(int(b[0]), "en", "oh") + " " + (W.pair(b[1:], "en", "oh") if b[1] != "0" else W.digits(b[1:], "en", "oh"))
    return [W.digits(s[:3], "en", "oh"), mid, *pairs_l(s[6:], "en", "oh")]


def us_style(style: str, s: str, rng: random.Random) -> tuple:
    """(groups, plus, prefix words) for a US number."""
    if style == "digits_zero":
        return [W.digits(s[:3], "en"), W.digits(s[3:6], "en"), W.digits(s[6:], "en")], "", ""
    if style == "digits_oh":
        return [W.digits(s[:3], "en", "oh"), W.digits(s[3:6], "en", "oh"), W.digits(s[6:], "en", "oh")], "", ""
    if style == "mixed":
        return _us_mixed(s, rng), "", ""
    if style == "triplet":
        return _us_triplet(s, rng), "", ""
    if style == "double":
        tri = rng.choice(["triple", "triple"])
        return [W.digits_rep(s[:3], "en", "oh", tri), W.digits_rep(s[3:6], "en", "oh", tri), W.digits_rep(s[6:], "en", "oh", tri)], "", ""
    if style == "area_code":
        return _us_mixed(s, rng), "", "area code"
    if style == "country_us":
        g = _us_mixed(s, rng) if rng.random() < 0.6 else [W.digits(s[:3], "en", "oh"), W.digits(s[3:], "en", "oh")]
        return ["plus one"] + g, "1", ""
    raise ValueError(style)


def tf_us_style(style: str, s: str, rng: random.Random) -> list:
    """Toll-free 1 8xx NXX XXXX. hundred: "one eight hundred" (800) / "one eight triple ..." none; ohoh: every digit by digit,
    "one eight oh oh" / "one eight eight eight"."""
    a = s[1:4]
    if style == "tollfree_hundred":
        head = "one eight hundred" if a == "800" else "one " + " ".join(W.digit(int(c), "en", "oh") for c in a)
    else:
        head = "one " + " ".join(W.digit(int(c), "en", "oh") for c in a)
    tail = pairs_l(s[7:], "en", "oh") if (style == "tollfree_hundred" and rng.random() < 0.5) else [W.digits(s[7:], "en", "oh")]
    return [head, W.digits(s[4:7], "en", "oh"), *tail]


def uk_style(style: str, s: str, rng: random.Random) -> tuple:
    if style == "digits":
        return [W.digits(s[:5], "en", "oh"), W.digits(s[5:], "en", "oh")], "", ""
    if style == "double":
        return [W.digits_rep(s[:5], "en", "oh", "treble"), W.digits_rep(s[5:], "en", "oh", "treble")], "", ""
    if style == "mobile_pairs":
        return [W.digits(s[:5], "en", "oh"), *pairs_l(s[5:], "en", "oh")], "", ""
    if style == "london_pairs":
        return [W.digits(s[:3], "en", "oh"), *pairs_l(s[3:7], "en", "oh"), *pairs_l(s[7:], "en", "oh")], "", ""
    if style == "country_uk":
        head = "plus four four" if rng.random() < 0.6 else "plus forty four"
        rest = s[1:]
        return [head, W.digits_rep(rest[:4], "en", "oh", "treble"), W.digits_rep(rest[4:], "en", "oh", "treble")], "44", ""
    raise ValueError(style)


def es_style(style: str, s: str, rng: random.Random, kind: str) -> tuple:
    L = "es"
    if kind == "tf_mx":
        return ["cero uno", "ochocientos", W.digits(s[5:8], L), W.digits(s[8:], L)], "", ""
    if kind == "mx":
        return [W.card(int(s[:2]), L), *pairs_l(s[2:6], L), *pairs_l(s[6:], L)], "", ""
    h = W.card(int(s[:3]), L)
    if style == "digits":
        return [W.digits(s[:3], L), W.digits(s[3:6], L), W.digits(s[6:], L)], "", ""
    if style == "hundreds_pairs":
        return [h, *pairs_l(s[3:], L)], "", ""
    if style == "mixed":
        return [W.digits(s[:3], L), *pairs_l(s[3:], L)], "", ""
    if style == "country":
        return ["más treinta y cuatro", h, *pairs_l(s[3:], L)] if rng.random() < 0.6 else \
            ["más treinta y cuatro", W.digits(s[:3], L), W.digits(s[3:6], L), W.digits(s[6:], L)], "34", ""
    if style == "doble":
        return [W.digits_rep(s[:3], L), W.digits_rep(s[3:6], L), W.digits_rep(s[6:], L)], "", ""
    raise ValueError(style)


def _be(n: int) -> str:
    """Belgian / Swiss pair words: septante, nonante, huitante (octante is not used)."""
    tens = {7: "septante", 9: "nonante", 8: "huitante"}
    t, u = divmod(n, 10)
    if t in tens:
        if u == 0:
            return tens[t]
        return f"{tens[t]} et {W.DIGITS['fr'][1]}" if u == 1 else f"{tens[t]} {W.DIGITS['fr'][u]}"
    return W.card(n, "fr")


def fr_style(style: str, s: str, rng: random.Random, kind: str) -> tuple:
    L = "fr"
    P = lambda x: W.pair(x, L)  # noqa: E731
    if kind == "tf_fr":
        return ["zéro huit cent", P(s[4:6]), P(s[6:8]), P(s[8:])], "", ""
    if kind == "fr_be":
        sw = "ch" if int(s[-1]) % 2 else "be"        # a fixed choice per number (Belgian: septante / nonante; Swiss also huitante)
        f = (lambda n: _be(n) if (n >= 90 or 70 <= n < 80 or (sw == "ch" and 80 <= n < 90)) else W.card(n, L))
        prs = [int(s[i:i + 2]) for i in range(2, 10, 2)]
        return [W.digits(s[:2], L), " ".join(f(p) if p >= 10 else P(f"{p:02d}") for p in prs)], "", ""
    if style == "digits":
        return [W.digits(s[i:i + 2], L) for i in range(0, 10, 2)], "", ""
    if style == "pairs":
        return [W.digits(s[:2], L)] + [P(s[i:i + 2]) for i in range(2, 10, 2)], "", ""
    if style == "mixed":
        return [W.digits(s[:2], L), P(s[2:4]), P(s[4:6]), W.digits(s[6:], L)], "", ""
    if style == "country":
        return ["plus trente trois", W.digit(int(s[1]), L)] + [P(s[i:i + 2]) for i in range(2, 10, 2)], "33", ""
    if style == "double":
        return [W.digits_rep(s[i:i + 2], L) for i in range(0, 10, 2)], "", ""
    raise ValueError(style)


def small_style(style: str, s: str, rng: random.Random, lang: str, kind: str) -> tuple:
    P = lambda x: W.pair(x, lang)  # noqa: E731
    D = lambda x: W.digits(x, lang)  # noqa: E731
    cc = {"de": "plus neun und vierzig", "pt": "mais cinquenta e cinco", "it": "più trenta nove", "nl": "plus drie en dertig"}[lang]
    cut = {"de": 4, "pt": 2, "it": 3, "nl": 2}[lang]
    if lang == "pt":
        head, mid, rest = s[:2], s[2:3], s[3:]
        if style == "digits":
            return [D(head), D(mid) + " " + D(rest[:4]), D(rest[4:])], "", ""
        if style == "pairs":
            return [P(head), D(mid), *pairs_l(rest, lang)], "", ""
        if style == "mixed":
            return [P(head), D(mid), D(rest[:4]), *pairs_l(rest[4:], lang)], "", ""
        if style == "country":
            return [cc, P(head), D(mid), *pairs_l(rest, lang)], "55", ""
    if lang == "it":
        if style == "digits":
            return [D(s[:3]), D(s[3:6]), D(s[6:])], "", ""
        if style == "pairs":
            return [D(s[:3]), *pairs_l(s[3:9], lang), D(s[9:])], "", ""
        if style == "mixed":
            return [D(s[:3]), D(s[3:6]), *pairs_l(s[6:], lang)] if len(s[6:]) % 2 == 0 else [D(s[:3]), D(s[3:7]), *pairs_l(s[7:], lang)], "", ""
        if style == "country":
            return [cc, D(s[:3]), *pairs_l(s[3:9], lang), D(s[9:])], "39", ""
    head, rest = s[:cut], s[cut:]
    if style == "digits":
        return [D(head), D(rest[:4]), D(rest[4:])], "", ""
    if style == "pairs":
        return [D(head), *pairs_l(rest, lang)], "", ""
    if style == "mixed":
        return [D(head), D(rest[:4]), *pairs_l(rest[4:], lang)], "", ""
    if style == "country":
        return [cc, D(head[1:]), *pairs_l(rest, lang)], {"de": "49", "nl": "31"}[lang], ""
    raise ValueError(style)
