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
