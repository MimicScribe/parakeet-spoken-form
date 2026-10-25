"""Repeat marker for training labels: "¡" between identical neighbours inside a number, so a transducer has a token to
emit between the two halves of a repeat ("5¡5", "two¡ two", "1¡1¡1"). Ordinary words are never marked ("very very").
strip_marker() removes every "¡" that directly follows a non-space character, so Spanish "¡Hola!" (after a space or at
the start) survives.
Not marked: a pair spoken as one event ("double five", "triple oh"; pass the spoken text as `spoken`), "one one" outside a
longer number (pronoun / stutter: "one one aspect"), scale-word stutters ("billion billion"), a pair whose second word starts
a new clause ("ten Ten fifteen"), year ranges ("1999-2000"), repeats across a spoken dash ("3-3").
  python3 marker.py     (runs the self-tests)"""
import re

M = "¡"
DIGITW = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "nine": "9"}
NUMWORDS = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
               "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion".split())
SCALE = {"hundred", "thousand", "million", "billion", "trillion"}
_core = lambda t: re.sub(r"^[\"'(]+|[\"').,;:!?]+$", "", t).lower()  # noqa: E731


def spoken_digits(spoken):
    """Digits the speaker said one by one, in order, as (digit string, flags): flags[i] is True when digit i is the 2nd / 3rd
    of a "double X" / "triple X" (one spoken event: no marker before it)."""
    ds, fl, rep = [], [], 0
    for w in (_core(t) for t in spoken.split()):
        if w in ("double", "triple"):
            rep = 2 if w == "double" else 3
            continue
        if w in DIGITW:
            n = rep or 1
            ds += [DIGITW[w]] * n
            fl += [False] + [True] * (n - 1)
        rep = 0
    return "".join(ds), fl


def _mark_run(run, said):
    """Marks inside one digit run ("5551234" or "979-3-2231"); never across a hyphen. `said` = spoken_digits(...) or None."""
    flags = None
    if said is not None:
        bare = run.replace("-", "")
        k = said[0].find(bare)
        if k < 0:
            return run  # the spoken source does not show this run digit by digit: leave it unmarked
        flags = said[1][k:k + len(bare)]
    out, i = [], 0  # i = index among the run's digits (hyphens skipped)
    for j, ch in enumerate(run):
        out.append(ch)
        if ch.isdigit():
            if j + 1 < len(run) and run[j + 1] == ch and not (flags and flags[i + 1]):
                out.append(M)
            i += 1
    return "".join(out)


def add_marker(label, spoken=None):
    """`spoken`: the row's spoken-form source (the words as said) when the label writes numbers as digits; used to skip
    pairs spoken as "double X" / "triple X"."""
    said = spoken_digits(spoken) if spoken and re.search(r"\b(double|triple)\b", spoken, re.I) else None

    # digits: only digit-by-digit strings (>= 5 digits in all, the label-v3 rule, incl. spoken-dash groups);
    # never years, year ranges, amounts or codes ("1999", "1999-2000", "100", "B11")
    def run(m):
        s = m.group()
        if sum(c.isdigit() for c in s) < 5 or re.fullmatch(r"\d{4}-\d{2,4}", s):
            return s
        return _mark_run(s, said)
    out = re.sub(r"(?<![\w.\-])\d+(?:-\d+)*(?![\w\-]|\.\d)", run, label)
    # number words: identical adjacent number words, punctuation ignored for the comparison ("two two.", "twenty twenty,")
    toks = re.split(r"(\s+)", out)
    words = toks[::2]
    for k in range(1, len(words)):
        a, b = words[k - 1], words[k]
        w = _core(a)
        if w != _core(b) or w not in NUMWORDS or re.search(r"[.,;:!?]$", a):  # "five, five" (pause) stays unmarked
            continue
        if w in SCALE or (b[:1].isupper() and not a[:1].isupper()):  # "billion billion", "ten Ten fifteen"
            continue
        if w == "one":  # "one one" only inside a longer spoken number ("room one one two")
            before = _core(words[k - 2]) if k >= 2 and not re.search(r"[.,;:!?]$", words[k - 2]) else ""
            after = _core(words[k + 1]) if k + 1 < len(words) and not re.search(r"[.,;:!?]$", b) else ""
            if before not in NUMWORDS and after not in NUMWORDS:
                continue
        toks[2 * (k - 1)] = a + M
    return "".join(toks)


def strip_marker(text):
    return re.sub(r"(?<=\S)" + M, "", text)


def _test():
    cases = [("call 5551234 now", "call 5¡5¡51234 now"), ("room two two oh one", "room two¡ two oh one"), ("11123", "1¡1¡123"),
             ("twenty twenty to twenty twenty one", "twenty¡ twenty to twenty¡ twenty one"), ("very very good", "very very good"),
             ("we need five, five is plenty", "we need five, five is plenty"), ("ISBN 9780814659311", "ISBN 978081465931¡1"),
             ("B12 and 3D", "B12 and 3D"), ("Two two", "Two¡ two"), ("no repeats 123", "no repeats 123"),
             ("in 1999 we paid 100", "in 1999 we paid 100"), ("aisle B11", "aisle B11"), ("it was two two.", "it was two¡ two."),
             ("version 802.11", "version 802.11"), ("call 5551234.", "call 5¡5¡51234."),
             # stutters, scale words, clause starts, hyphen runs
             ("there is one one aspect", "there is one one aspect"), ("room one one two", "room one¡ one two"),
             ("five billion billion dollars", "five billion billion dollars"), ("it was ten Ten fifteen", "it was ten Ten fifteen"),
             ("ISBN 979-3-2231-8063-7", "ISBN 979-3-2¡231-8063-7"), ("from 1999-2000", "from 1999-2000"),
             ("call 555-1234", "call 5¡5¡5-1234"), ("call 55-12", "call 55-12"), ("3-3-3", "3-3-3"),
             ("one one. Then", "one one. Then")]
    bad = 0
    for src, want in cases:
        got = add_marker(src)
        ok = got == want and strip_marker(got) == src
        bad += not ok
        print("ok " if ok else "BAD", repr(src), "->", repr(got))
    # pairs spoken as "double X" / "triple X" stay unmarked; plain repeats in the same run are still marked
    spoken_cases = [("27769681 again.", "Two double seven six nine six eight one again.", "27769681 again."),
                    ("It's 008390572008.", "It's double zero eight three nine zero five seven two double zero eight.",
                     "It's 008390572008."),
                    ("call 55512334", "call five double five one two three three four", "call 5¡55123¡34"),
                    ("call 5551234", "call triple five one two three four", "call 5551234"),
                    ("ref 7722111", "ref double seven two two triple one", "ref 772¡2111"),
                    ("ref 99123 or 66", "ref nine nine one two three or double six", "ref 9¡9123 or 66"),
                    ("ref 12345", "ref twelve three double four", "ref 12345")]
    for lab, sp, want in spoken_cases:
        got = add_marker(lab, sp)
        ok = got == want and strip_marker(got) == lab
        bad += not ok
        print("ok " if ok else "BAD", repr(lab), "|", repr(sp), "->", repr(got))
    assert add_marker("call 5551234", "call five five five one two three four") == "call 5¡5¡51234"  # no double: spoken unused
    for s in ("¡Hola! dijo", "Ella dijo ¡vamos!"):
        assert strip_marker(s) == s, s
    print("fail", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(_test())
