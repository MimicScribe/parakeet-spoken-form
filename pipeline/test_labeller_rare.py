"""English labeller: Roman numerals, "&", versions, fractions (2026-10-17). Every candidate parses back to the
caption's value / letters. python3 test_labeller_rare.py"""
import re
from fractions import Fraction

import labeller as L
from readings import ONES

W = {w: i for i, w in enumerate(ONES[:20])} | {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                                               "seventy": 70, "eighty": 80, "ninety": 90}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
       "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "fortieth": 40, "fiftieth": 50}
DEN = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourth": 4, "fourths": 4,
       "fifth": 5, "fifths": 5, "eighth": 8, "eighths": 8, "tenth": 10, "tenths": 10}


def num(words):
    """Cardinal / ordinal words (up to 99) -> int."""
    v = 0
    for w in words:
        v += W[w] if w in W else ORD[w]
    return v


def frac_value(reading):
    ws = reading.split()
    whole = 0
    if "and" in ws:
        i = ws.index("and")
        whole, ws = num(ws[:i]), ws[i + 1:]
    if ws == ["half"]:
        return whole + Fraction(1, 2)
    a = 1 if ws[0] in ("a", "one") else num(ws[:-1])
    return whole + Fraction(a, DEN[ws[-1]])


def one(t, kind):
    sp = [s for s in L.spans(t) if s.kind == kind]
    assert len(sp) == 1, (t, [(s.text, s.kind) for s in L.spans(t)])
    assert L.row_ok(t, L.spans(t)) == "", (t, L.row_ok(t, L.spans(t)))
    return sp[0]


# Roman numerals: value parses back; regnal names read "the <ordinal>" first, titles / wars the cardinal first
for t, raw, v, first in [("World War II ended", "II", 2, "two"), ("Henry VIII had wives", "VIII", 8, "the eighth"),
                         ("Chapter IV begins", "IV", 4, "four"), ("Super Bowl LVII was", "LVII", 57, "fifty seven"),
                         ("Rocky III is", "III", 3, "three"), ("Pope John Paul II came", "II", 2, "the second"),
                         ("Louis XIV built it", "XIV", 14, "the fourteenth")]:
    s = one(t, "roman")
    assert s.text == raw and s.cands[0] == first, (t, s.cands)
    for c in s.cands:
        assert num([w for w in c.split() if w != "the"]) == v, (t, c)
for t in ("I said V for X", "The XL size", "The IV bag", "Malcolm X said", "an IV drip", "Version IIII", "Part VX"):
    assert not [s for s in L.spans(t) if s.kind == "roman"], t

# "&": the letters (or the words) either side are kept, the symbol becomes "and" / "n"
for t, l, r in [("AT&T merged", "AT", "T"), ("R&D spend", "R", "D"), ("B&Q opened", "B", "Q")]:
    s = one(t, "amp")
    assert s.cands == [f"{l} and {r}"] and re.sub(r" and ", "&", s.cands[0]) == f"{l}&{r}"
assert one("Johnson & Johnson sued", "amp").cands == ["and"]
assert L.render("rock & roll", L.spans("rock & roll"), [0]) == "rock and roll"

# versions: every component value comes back, in order
for t, comps in [("v1.2.3 is out", [1, 2, 3]), ("version 5.1.3", [5, 1, 3]), ("build 10.0.19041 or 2.4.10.1", None)]:
    if comps is None:
        continue
    s = one(t, "version")
    for c in s.cands:
        parts = re.split(r" (?:point|dot) ", re.sub(r"^(?:version|V) ", "", c))
        vals = [num(p.split()) if all(w in W for w in p.split()) and len(p.split()) <= 2 and not (
            len(p.split()) == 2 and p.split()[0] in ("zero", "oh")) else int("".join(str(W.get(w, 0)) for w in p.split()))
                for p in parts]
        assert vals == comps, (t, c, vals)
assert not [s for s in L.spans("on 12.05.19 we") if s.kind == "version"]  # a date

# fractions: value parses back (symbols, slashes, mixed numbers)
for t, v in [("about 3/4 of them", Fraction(3, 4)), ("2 1/2 cups", Fraction(5, 2)), ("½ cup", Fraction(1, 2)),
             ("1½ cups", Fraction(3, 2)), ("two 2/3 majority", Fraction(2, 3)), ("add ¼ teaspoon", Fraction(1, 4))]:
    s = one(t, "frac")
    fr = [c for c in s.cands if any(w in DEN for w in c.split())]
    assert fr and all(frac_value(c) == v for c in fr), (t, s.cands)
assert "three fourths" in one("about 3/4 of them", "frac").cands  # both common readings: the acoustic scorer picks
assert [s.kind for s in L.spans("in 24/7 mode")] == ["slash"]  # no named denominator: not a fraction
print("ok")
