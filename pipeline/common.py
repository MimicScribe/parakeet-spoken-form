"""Shared constants."""

import re

# The written-form characters the fine-tuned model must never emit. They are the
# only non-letter, non-punctuation pieces in Parakeet's 8,192-piece vocabulary.
WRITTEN_FORM = re.compile(r"[0-9$%€£/]")

# English number words, for picking number-bearing utterances out of real speech. "one", "oh",
# "first" and "second" are left out: they are mostly not numbers.
NUMBER_WORDS = re.compile(
    r"\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand|million|billion|trillion|percent|"
    r"third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twelfth|twentieth)\b",
    re.I,
)

# Words that belong to a spoken number but are not in NUMBER_WORDS. "one" and "oh" are left
# out of NUMBER_WORDS because they are mostly not numbers; inside a number they are ("one
# point one", "point oh five"). "point" separates decimals, "o" is a spoken zero, "and"/"a"
# link ("a hundred and five"), "half"/"quarter" the fractions people say ("one and a half
# million"). Used by ft_train's span-selective duration anchor (kl_terms durspan).
SPAN_CORE = set("one oh o point half halves quarter quarters".split())
SPAN_LINK = set(("and", "a"))
NUM_UNITS = set("percent dollar dollars euro euros cent cents basis bps year years month months day days".split())


def number_span(words: list[str]) -> list[bool]:
    """True where a word is part of a spoken number: the span a duration anchor must leave
    free (ft_train kl_terms durspan).

    A word is in a span when NUMBER_WORDS matches it (search, so "twenty-first" counts), when
    "point" sits next to a number-ish word ("one point one" is a decimal NUMBER_WORDS cannot
    see), or when a core or link word touches a span word ("a hundred and five", "one and a
    half million"). Words, not tokenizer pieces: "seventy" may split into several pieces.

    >>> number_span("we made two point seven five million dollars".split())
    [False, False, True, True, True, True, True, False]
    >>> number_span("twenty one.".split())
    [True, True]
    >>> number_span("one point one of the best".split())
    [True, True, True, False, False, False]
    >>> number_span("one and a half million units".split())
    [True, True, True, True, True, False]
    >>> number_span("one and a half.".split())
    [True, True, True, True]
    >>> number_span("one percent growth".split())
    [True, True, False]
    >>> number_span("a good point of view".split())
    [False, False, False, False, False]
    >>> number_span("the tenant won one of the prizes".split())
    [False, False, False, False, False, False, False]
    >>> number_span("we made five million and a dog".split())
    [False, False, True, True, False, False, False]
    """
    raw_w = [x.lower() for x in words]
    clean_w = [re.sub(r"^[^\w]+|[^\w]+$", "", x) for x in raw_w]

    m = [bool(NUMBER_WORDS.search(cw)) if cw else False for cw in clean_w]

    for j, cw in enumerate(clean_w):
        if cw == "point" and ((j > 0 and (clean_w[j - 1] in SPAN_CORE or m[j - 1])) or
                              (j + 1 < len(clean_w) and (clean_w[j + 1] in SPAN_CORE or m[j + 1]))):
            m[j] = True
        elif cw in ("one", "half", "quarter", "halves", "quarters") and j + 1 < len(clean_w) and clean_w[j + 1] in NUM_UNITS:
            m[j] = True
        elif cw == "one" and j + 3 < len(clean_w) and clean_w[j + 1] == "and" and clean_w[j + 2] == "a" and clean_w[j + 3] in ("half", "quarter", "halves", "quarters"):
            m[j] = m[j + 1] = m[j + 2] = m[j + 3] = True
        elif cw == "half" and j >= 3 and clean_w[j - 3] in ("one", "two", "three", "four", "five") and clean_w[j - 2] == "and" and clean_w[j - 1] == "a":
            m[j - 3] = m[j - 2] = m[j - 1] = m[j] = True

    changed = True
    while changed:
        changed = False
        for j, cw in enumerate(clean_w):
            if not m[j]:
                if cw in SPAN_CORE and ((j > 0 and m[j - 1]) or (j + 1 < len(clean_w) and m[j + 1])):
                    m[j] = changed = True
                elif cw in SPAN_LINK:
                    left_ok = (j > 0 and (m[j - 1] or (clean_w[j - 1] in SPAN_LINK and j > 1 and m[j - 2])))
                    right_ok = (j + 1 < len(clean_w) and (m[j + 1] or (clean_w[j + 1] in SPAN_LINK and j + 2 < len(clean_w) and m[j + 2])))
                    if left_ok and right_ok:
                        m[j] = changed = True
    return m
