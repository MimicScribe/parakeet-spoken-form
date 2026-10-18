"""Label vs second transcript, number-aware (2026-10-18). The second transcript writes digits where the label has number
words ("5 na 11", "half 7", "0965 656832", "29,99 euro"), splits compounds ("ottanta tre"), says "coma" where the label
says "con", and joins what the label splits ("zwanzigvierundvierzig"). A letter-for-letter or digit-for-word comparison
rejects those rows although the audio says what the label says. This module compares the two texts in two parts:

  skeleton   the words that are no numbers (digit tokens, number words and the joiners / decimal words between two
             numbers are left out): both texts must have the SAME skeleton, so a unit word the audio added or dropped
             ("vierundneunzig euro zehn" for "vierundneunzig zehn") or any other word difference still rejects the row;
  numbers    number_outcomes (the held-out scorer's reader) of the label as the reference against the transcript:
             every number of the label must come out right (words or digits). Small plain integers of the transcript are
             first written out as words (relative clocks "half 7", "5 over 11" need the words); larger ones, prices and
             phone numbers stay digits (number_outcomes splits / joins them against the label's numbers).

usage: agree(label, transcript, lang) -> (ok, why)"""
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import number_outcomes as N  # noqa: E402

sys.path.insert(0, N._PIPE)
import ml_cands as M  # noqa: E402

_TOK = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.U)
# joiners and decimal words that sit BETWEEN two numbers inside one spoken value ("cuarenta y cuatro", "tres con cinco",
# "novantacinque e settantacinque", "dos coma tres"): not part of the skeleton on either side
_BETWEEN = {"e", "et", "y", "und", "en", "and", "com", "con", "coma", "comma", "komma", "virgule", "virgola", "virgula",
            "vírgula", "punto", "punt", "punkt", "ponto", "point", "mit"}


# declined forms of "one" are words, not numbers, here: "um einem Uhr" (a generator slip) vs "um ein Uhr" must differ
_DECLINED = {"de": {"einem", "einen", "einer", "eines"}}


def expand_small_ints(text: str, lang: str, top: int = 60) -> str:
    """Plain integers 0..top written as words of the language ("half 7" -> "half zeven"): relative clocks, split tens
    and the like are read from words. Leading zeros, separators and larger numbers are left as digits."""
    if lang not in N._LEX:
        return text

    def rep(m):
        tok = m.group(0)
        if tok.isdigit() and (tok == "0" or not tok.startswith("0")) and int(tok) <= top:
            try:
                return " ".join(M.g(lang, "card", int(tok)))
            except Exception:  # noqa: BLE001  (a number the grammar table cannot say stays a digit)
                return tok
        return tok
    return re.sub(r"(?<![\d.,])\d+(?!\d)(?![.,]\d)", rep, text)


def digitize(text: str, lang: str) -> str:
    """The transcript's runs of number words written as digits ("neunundfünfzig euro fünfundzwanzig" -> "59 euro 25",
    "ottanta tre" -> "83"), so a transcript that writes words is read by the same digit rules as one that writes digits.
    Only runs of consecutive number words (a joiner may sit between two of them); other text is unchanged."""
    if lang not in N._LEX:
        return text
    toks = list(re.finditer(r"[^\W\d_]+", text, re.U))
    words = [N._fold(m.group(0)) for m in toks]
    isn = [bool(N._segment(lang, w)) for w in words]
    out, last, i = [], 0, 0
    while i < len(toks):
        if not isn[i]:
            i += 1
            continue
        j = i + 1
        while j < len(toks) and (isn[j] or (words[j] == N._CONN[lang] and j + 1 < len(toks) and isn[j + 1]
                                            and text[toks[j - 1].end():toks[j].start()].strip() == "")):
            j += 1
        run = " ".join(words[i:j])
        nums = [x for x in N.extract_numbers(run, lang, keep_weak=True) if x.form == "words"]
        rep, k = [], i
        for x in nums:
            if x.end - x.start + 1 > j - i or x.start < k - i:
                rep = None
                break
            v = min(x.vals)
            rep.append((i + x.start, i + x.end, str(int(v)) if v == int(v) else format(v, "f")))
            k = i + x.end + 1
        if rep:
            pos = i
            for a, b, d in rep:
                out.append(text[last:toks[a].start()])
                out.append(d)
                last = toks[b].end()
        i = j
    return "".join(out) + text[last:]


def skeleton(text: str, lang: str) -> str:
    """The letters of the non-number words, in order (see the module doc)."""
    toks = _TOK.findall(N._fold(text))
    isnum = [t[0].isdigit() or (bool(N._segment(lang, t)) and t not in _DECLINED.get(lang, ())) if lang in N._LEX
             else t[0].isdigit() for t in toks]
    keep = []
    for i, t in enumerate(toks):
        if isnum[i]:
            continue
        if t in _BETWEEN and 0 < i < len(toks) - 1 and isnum[i - 1] and isnum[i + 1]:
            continue
        keep.append(t)
    return "".join(keep)


def digit_run(label: str, transcript: str, lang: str) -> bool:
    """Every number of the transcript is a digit token and the digits of the label's whole numbers, joined in order,
    are the digits of the transcript's tokens joined in order ("0965 656832" for a phone number said in pairs or
    hundreds): digit tokens carry no grouping, so only the digit string can be compared (leading zeros included)."""
    t = [m.group(0) for m in re.finditer(r"\d+", transcript)]
    words = [x for x in N.extract_numbers(transcript, lang, keep_weak=True) if x.form != "digits"]
    ref = N.extract_numbers(label, lang, keep_weak=False)
    if not t or words or not ref or any(len(x.vals) != 1 or min(x.vals) != int(min(x.vals)) for x in ref):
        return False
    return "".join(str(int(min(x.vals))) for x in ref) == "".join(t)


def agree(label: str, transcript: str, lang: str) -> tuple:
    """(True, "") when the transcript says the label; else (False, why) with why = "words" or "numbers"."""
    if skeleton(label, lang) != skeleton(transcript, lang):
        return False, "words"
    outs = N.number_outcomes(label, expand_small_ints(transcript, lang), lang)
    if any("RIGHT" not in o["outcome"] for o in outs) and not digit_run(label, transcript, lang):
        return False, "numbers"
    return True, ""


def letters(text: str) -> str:
    """Letters only (digits, spaces, punctuation and case dropped): the verbatim-match rescue's old comparison."""
    return re.sub(r"[^\w]|\d|_", "", unicodedata.normalize("NFKC", text).lower())


def rescued(label: str, transcript: str, lang: str) -> bool:
    """A row whose CTC check failed is kept when the verbatim second transcript says the label: the same letters (the
    old rule), or, where the transcript writes digits for number words, `agree`."""
    return letters(transcript or "") == letters(label) or bool(transcript) and agree(label, transcript, lang)[0]
