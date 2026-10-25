"""Repeat marker for training labels: "¡" between identical neighbours inside a number, so a transducer has a token to
emit between the two halves of a repeat ("5¡5", "two¡ two", "1¡1¡1"). Ordinary words are never marked ("very very").
strip_marker() removes every "¡" that directly follows a non-space character, so Spanish "¡Hola!" (after a space or at
the start) survives.
  python3 marker.py     (runs the self-tests)"""
import re

M = "¡"
NUMWORDS = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
               "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion".split())


def add_marker(label):
    # digits: every adjacent identical pair inside a digit run
    out = re.sub(r"\d{2,}", lambda m: re.sub(r"(\d)(?=\1)", r"\1" + M, m.group()), label)
    # number words: identical adjacent number words ("two two", "twenty twenty", "five, five" is NOT adjacent -> untouched)
    toks = re.split(r"(\s+)", out)
    for i in range(2, len(toks), 2):
        a, b = toks[i - 2], toks[i]
        if a.lower() == b.lower() and a.lower() in NUMWORDS:
            toks[i - 2] = a + M
    return "".join(toks)


def strip_marker(text):
    return re.sub(r"(?<=\S)" + M, "", text)


def _test():
    cases = [("call 5551234 now", "call 5¡5¡51234 now"), ("room two two oh one", "room two¡ two oh one"), ("1112", "1¡1¡12"),
             ("twenty twenty to twenty twenty one", "twenty¡ twenty to twenty¡ twenty one"), ("very very good", "very very good"),
             ("we need five, five is plenty", "we need five, five is plenty"), ("ISBN 9780814659311", "ISBN 978081465931¡1"),
             ("B12 and 3D", "B12 and 3D"), ("Two two", "Two¡ two"), ("no repeats 123", "no repeats 123")]
    bad = 0
    for src, want in cases:
        got = add_marker(src)
        ok = got == want and strip_marker(got) == src
        bad += not ok
        print("ok " if ok else "BAD", repr(src), "->", repr(got))
    for s in ("¡Hola! dijo", "Ella dijo ¡vamos!"):
        assert strip_marker(s) == s, s
    print("fail", bad)


if __name__ == "__main__": _test()
