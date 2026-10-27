"""Repeat marker for training labels: "¡" between identical neighbours inside a number, so a transducer has a token to
emit between the two halves of a repeat ("5¡5", "two¡ two", "1¡1¡1"). Ordinary words are never marked ("very very").
strip_marker() removes every mark that directly follows a letter or digit, so Spanish "¡Hola!" (after a space or at
the start) survives.
Not marked: a pair spoken as one event ("double five", "triple oh"; pass the spoken text as `spoken`), "one one" outside a
longer number (pronoun / stutter: "one one aspect"), scale-word stutters ("billion billion"), a pair whose second word starts
a new clause ("ten Ten fifteen"), year ranges ("1999-2000"), repeats across a spoken dash ("3-3").
Counted mode (add_marker(..., counted=True)): the mark names the copy's position in its run, so the decoder history differs at
every copy: copy 1 "¡", copy 2 "¿", copy 3 and later "¡¡" ("two¡ two¿ two¡¡ two"); the last copy is never marked. Extra rules in
counted mode: no marks on a word that follows double / triple / quadruple (it is that phrase's operand), on tens / teens pairs
("twenty twenty", "fifty fifty": years, clock times, scores), on a copy that multiplies a scale word ("two two hundred": a
different place value), or on "one" / "oh" with no other number word beside the run.
  python3 marker.py     (runs the self-tests)"""
import re

M = "¡"
COUNTED = ("¡", "¿", "¡¡")  # copy 1, copy 2, copy 3 and later (counted mode); M stays the single-marker mode's symbol
DIGITW = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "nine": "9"}
NUMWORDS = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
               "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion".split())
SCALE = {"hundred", "thousand", "million", "billion", "trillion"}
MULTIPLIER = {"double", "triple", "quadruple", "quintuple", "treble"}
TENS_TEENS = set("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty "
                 "seventy eighty ninety".split())
AMBIGUOUS = {"one", "oh"}  # a pronoun / interjection unless another number word sits beside the run
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


def pos_mark(pos):
    """The counted-mode symbol of the copy at 1-based position `pos` in its run (3 and later share the last symbol)."""
    return COUNTED[min(max(pos, 1), len(COUNTED)) - 1]


def _mark_run(run, said, counted=False):
    """Marks inside one digit run ("5551234" or "979-3-2231"); never across a hyphen. `said` = spoken_digits(...) or None.
    counted: the k-th copy of a chain of identical digits gets pos_mark(k) instead of M."""
    flags = None
    if said is not None:
        bare = run.replace("-", "")
        k = said[0].find(bare)
        if k < 0:
            return run  # the spoken source does not show this run digit by digit: leave it unmarked
        flags = said[1][k:k + len(bare)]
    out, i, pos = [], 0, 0  # i = index among the run's digits (hyphens skipped); pos = copies already chained
    for j, ch in enumerate(run):
        out.append(ch)
        if ch.isdigit():
            if j + 1 < len(run) and run[j + 1] == ch and not (flags and flags[i + 1]):
                pos += 1
                out.append(pos_mark(pos) if counted else M)
            else:
                pos = 0
            i += 1
        else:
            pos = 0
    return "".join(out)


def _link_ok(words, k, counted):
    """May words[k-1] and words[k] be a repeat pair? -> (ok, reason). reason names the rule that said no (census)."""
    a, b = words[k - 1], words[k]
    w = _core(a)
    if w != _core(b) or w not in NUMWORDS:
        return False, "not_pair"
    if re.search(r"[.,;:!?]$", a):  # "five, five" (pause) stays unmarked
        return False, "punct"
    if w in SCALE:  # "billion billion"
        return False, "scale_word"
    if b[:1].isupper() and not a[:1].isupper():  # "ten Ten fifteen"
        return False, "clause_start"
    if counted:
        if k >= 2 and _core(words[k - 2]) in MULTIPLIER and not re.search(r"[.,;:!?]$", words[k - 2]):
            return False, "multiplier_operand"  # "double five five": the first five belongs to "double five"
        if w in TENS_TEENS:
            return False, "tens_pair"  # "twenty twenty", "fifty fifty"
        if k + 1 < len(words) and not re.search(r"[.,;:!?]$", b) and _core(words[k + 1]) in SCALE:
            return False, "place_value"  # "two two hundred": the second two multiplies the scale word
    if w in (AMBIGUOUS if counted else {"one"}):  # "one one" / "oh oh" only inside a longer spoken number ("room one one two")
        before = _core(words[k - 2]) if k >= 2 and not re.search(r"[.,;:!?]$", words[k - 2]) else ""
        after = _core(words[k + 1]) if k + 1 < len(words) and not re.search(r"[.,;:!?]$", b) else ""
        if before not in NUMWORDS and after not in NUMWORDS:  # a third copy beside the pair counts as a number word
            return False, "ambiguous_word"
    return True, ""


def word_runs(words, counted=False):
    """Chains of identical neighbouring number words. -> (runs, skipped): runs = [(first_index, n_copies)] with the pairs
    that pass _link_ok; skipped = [(index_of_first_copy, reason)] for equal neighbours some rule refused (reasons: punct,
    scale_word, clause_start, multiplier_operand, tens_pair, place_value, ambiguous_word)."""
    runs, skipped, start, n = [], [], None, 1
    for k in range(1, len(words)):
        ok, why = _link_ok(words, k, counted)
        if ok:
            if start is None:
                start, n = k - 1, 1
            n += 1
        else:
            if start is not None:
                runs.append((start, n))
                start = None
            if why and why != "not_pair":
                skipped.append((k - 1, why))
    if start is not None:
        runs.append((start, n))
    return runs, skipped


def add_marker(label, spoken=None, counted=False, strict=None):
    """`spoken`: the row's spoken-form source (the words as said) when the label writes numbers as digits; used to skip
    pairs spoken as "double X" / "triple X". counted=True: position symbols (pos_mark) and the counted-mode rules.
    strict: the counted-mode rules without the position symbols (None = same as counted); strict=True, counted=False writes the
    single "¡" on exactly the copies counted mode marks (same rows, one symbol: the control arm of a single-vs-counted test)."""
    strict = counted if strict is None else strict
    said = spoken_digits(spoken) if spoken and re.search(r"\b(double|triple)\b", spoken, re.I) else None

    # digits: only digit-by-digit strings (>= 5 digits in all, the label-v3 rule, incl. spoken-dash groups);
    # never years, year ranges, amounts or codes ("1999", "1999-2000", "100", "B11")
    def run(m):
        s = m.group()
        if sum(c.isdigit() for c in s) < 5 or re.fullmatch(r"\d{4}-\d{2,4}", s):
            return s
        return _mark_run(s, said, counted)
    out = re.sub(r"(?<![\w.\-])\d+(?:-\d+)*(?![\w\-]|\.\d)", run, label)
    # number words: identical adjacent number words, punctuation ignored for the comparison ("two two.", "twenty twenty,")
    toks = re.split(r"(\s+)", out)
    words = toks[::2]
    runs, _ = word_runs(words, strict)
    for first, n in runs:
        for c in range(n - 1):  # copies 1 .. n-1; the last copy is never marked
            toks[2 * (first + c)] += pos_mark(c + 1) if counted else M
    return "".join(toks)


def add_counted(label, spoken=None):
    return add_marker(label, spoken, counted=True)


_STRIP = re.compile(r"(?<=\w)(?:\u00a1\u00a1|[\u00a1\u00bf])")


def strip_marker(text):
    """Removes every marker of either mode: any "¡", "¿" or "¡¡" that directly follows a letter or digit.
    Spanish "¡Hola!" / "¿Qué?" (after a space, at the start, after a quote or bracket) survive."""
    return _STRIP.sub("", text)


def analyze(label, spoken=None):
    """Census record of one counted-mode pass: {"words": [(first_word_index, n_copies, word)], "skipped": [(index, reason)],
    "digit_marks": marks written inside digit strings (digit-form labels), "marked": the marked label}."""
    marked = add_marker(label, spoken, counted=True)
    words = re.split(r"(\s+)", label)[::2]
    runs, skipped = word_runs(words, True)
    return {"words": [(f, n, _core(words[f])) for f, n in runs], "skipped": skipped, "marked": marked,
            "digit_marks": len(re.findall(r"(?<=\d)(?:\u00a1\u00a1|\u00a1|\u00bf)", marked))}


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


def _test_counted():
    cases = [
        # positions: copy 1 "¡", copy 2 "¿", copy 3 and later "¡¡"; the last copy is never marked
        ("room two two oh one", "room two¡ two oh one"),
        ("call five five five one two three four", "call five¡ five¿ five one two three four"),
        ("pin zero zero zero zero", "pin zero¡ zero¿ zero¡¡ zero"),
        ("nine nine nine nine nine", "nine¡ nine¿ nine¡¡ nine¡¡ nine"),
        ("five five six six", "five¡ five six¡ six"),
        ("Two two", "Two¡ two"), ("it was two two.", "it was two¡ two."), ("oh oh seven", "oh¡ oh seven"),
        ("one one one", "one¡ one¿ one"), ("room one one two", "room one¡ one two"),
        ("double five five five", "double five five¡ five"), ("five five five hundred", "five¡ five five hundred"),
        ("code four four B", "code four¡ four B"),
        # digit-form labels (before the all-spoken relabel): the same positions
        ("call 5551234 now", "call 5¡5¿51234 now"), ("11123", "1¡1¿123"), ("ISBN 979-3-2231-8063-7", "ISBN 979-3-2¡231-8063-7"),
        # no mark
        ("double five one two", "double five one two"), ("triple five five", "triple five five"),
        ("twenty twenty five", "twenty twenty five"), ("it is fifty fifty", "it is fifty fifty"), ("eleven eleven", "eleven eleven"),
        ("very very good", "very very good"), ("there is one one aspect", "there is one one aspect"), ("oh oh no", "oh oh no"),
        ("we need five, five is plenty", "we need five, five is plenty"), ("it was ten Ten fifteen", "it was ten Ten fifteen"),
        ("five billion billion dollars", "five billion billion dollars"), ("two two hundred", "two two hundred"),
        ("two two thousand five", "two two thousand five"), ("thirty two thirty two", "thirty two thirty two"),
        ("one one. Then", "one one. Then"), ("B11 and 3D", "B11 and 3D"), ("in 1999 we paid 100", "in 1999 we paid 100"),
        ("from 1999-2000", "from 1999-2000"), ("3-3-3", "3-3-3"),
    ]
    bad = 0
    for src, want in cases:
        got = add_counted(src)
        ok = got == want and strip_marker(got) == src
        bad += not ok
        print("ok " if ok else "BAD", "counted", repr(src), "->", repr(got))
    # double / triple spoken source with a digit label: the paired digits stay unmarked, the plain repeats are counted
    got = add_marker("call 55512334", "call five double five one two three three four", counted=True)
    ok = got == "call 5¡55123¡34" and strip_marker(got) == "call 55512334"
    bad += not ok
    print("ok " if ok else "BAD", "counted spoken-source", repr(got))
    # strict single mode: counted rules, one symbol (same marked copies as counted mode)
    for src in ("pin zero zero zero zero", "twenty twenty five and two two hundred", "call five five five one", "oh oh no and nine nine"):
        c, st = add_counted(src), add_marker(src, strict=True)
        ok = strip_marker(st) == src and set(st) - set(src) <= {"\u00a1"}
        # same copies marked: the positions of the marks in the word list agree
        wc = [w.rstrip("\u00a1\u00bf") != w for w in c.split()]
        ws = [w.rstrip("\u00a1") != w for w in st.split()]
        ok = ok and wc == ws
        bad += not ok
        print("ok " if ok else "BAD", "strict single", repr(src), "->", repr(st))
    # single mode is untouched by the counted rules
    for src, want in (("twenty twenty five", "twenty¡ twenty five"), ("two two hundred", "two¡ two hundred"), ("oh oh no", "oh¡ oh no")):
        got = add_marker(src)
        ok = got == want
        bad += not ok
        print("ok " if ok else "BAD", "single mode", repr(src), "->", repr(got))
    # strip: both modes, Spanish punctuation survives
    for s in ("¡Hola! dijo", "Ella dijo ¡vamos! y ¿qué?", "\"¿Qué dijo?\"", "(¡Ay!)"):
        ok = strip_marker(s) == s
        bad += not ok
        print("ok " if ok else "BAD", "spanish kept", repr(s))
    ok = strip_marker("two¡ two¿ two¡¡ two") == "two two two two" and strip_marker("5¡5¿51234") == "5551234"
    bad += not ok
    print("ok " if ok else "BAD", "strip all symbols")
    # idempotent: a marked label is not marked again
    once = add_counted("pin zero zero zero")
    ok = add_counted(once) == once
    bad += not ok
    print("ok " if ok else "BAD", "idempotent", repr(once))
    # position symbols are single code points / one piece each
    ok = pos_mark(1) == "¡" and pos_mark(2) == "¿" and pos_mark(3) == "¡¡" and pos_mark(9) == "¡¡"
    bad += not ok
    print("ok " if ok else "BAD", "pos_mark")
    # census record names the refusing rule
    a = analyze("twenty twenty five and two two hundred and double five five and oh oh no and five, five")
    why = sorted(r for _, r in a["skipped"])
    ok = a["words"] == [] and why == ["ambiguous_word", "multiplier_operand", "place_value", "punct", "tens_pair"], (a["words"], why)
    bad += not ok
    print("ok " if ok else "BAD", "analyze reasons", why)
    print("counted fail", bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if (_test() + _test_counted()) else 0)
