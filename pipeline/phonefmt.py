"""Digit layout of phone numbers and codes in digit-labelled rows (separators only; digits are never touched).

A digit label teaches how the model writes a spoken phone number. Written with commas or dots ("08,72, 960",
"0,498,250") a later reader takes the span for a decimal or a thousands figure. The layout used here: groups separated by
single spaces, no comma / dot / dash / bracket inside, a leading zero kept inside its group. Groups follow the speaker
(`groups`: digit counts taken from word timings or a second transcript) when given, else the caption's own grouping.
An unbroken run with no hint is left as written (no grouping is invented).

Pure functions, no I/O:  phone_span(text) -> bool,  reformat(text, groups=None) -> str,  rewrite_text(text) -> (str, n).
"""
import re

_SPAN = re.compile(r"(?<![\w(])\+?\d[\d ().,/–\- ]*\d(?![\w])")
_THOUSANDS = re.compile(r"\d{1,3}([ ., ]\d{3})+")
_DATE = re.compile(r"\d{1,2}[./ ]+\d{1,2}[./ ]+\d{2,4}")
_RANGE = re.compile(r"\d{1,4}\.?\s?[–\-/]\s?\d{1,4}\.?")
_DASH = re.compile(r"\s*[\u2013\-/]\s*")
_PLAIN_NUMBER = re.compile(r"[1-9]\d*([ .,\u00a0]\d{3})*")
_SEP = re.compile(r"[ ().,/–\- ]+")


def phone_span(s: str) -> bool:
    """A phone / code-shaped span: >= 7 digits, not a thousands figure, date or range. A leading 0 or + marks a phone
    even when its groups are three digits wide ("0,498,250")."""
    d = re.sub(r"\D", "", s)
    if len(d) < 7:
        return False
    if _DATE.fullmatch(s) or _RANGE.fullmatch(s):
        return False
    if "\u2013" in s or re.search(r"(?<!\d)\d{1,2}[./]\d{1,2}[./]", s):
        return False  # an en dash joins a range; "d.m." starts a date (and date ranges)
    parts = [x.strip(" .") for x in _DASH.split(s)]
    if len(parts) >= 2 and not s.startswith("+") and all(_PLAIN_NUMBER.fullmatch(x) for x in parts):
        return False  # a range of plain numbers ("1418 - 1450", "11 000-22 500"), not a phone
    if _THOUSANDS.fullmatch(s) and not (s.startswith("0") or s.startswith("+")):
        return False
    return True


def _groups(s: str) -> list:
    return [g for g in _SEP.split(s.lstrip("+")) if g]


def reformat(s: str, groups=None) -> str:
    """The span with spoken-style groups. `groups`: digit counts per group (must sum to the digit count), else the
    caption's own separators decide. The digit sequence is always preserved."""
    plus = "+" if s.startswith("+") else ""
    digits = re.sub(r"\D", "", s)
    if groups and sum(groups) == len(digits) and all(g > 0 for g in groups):
        out, i = [], 0
        for g in groups:
            out.append(digits[i:i + g])
            i += g
        return plus + " ".join(out)
    gs = _groups(s)
    if len(gs) <= 1:
        return s
    return plus + " ".join(gs)


def rewrite_text(text: str, groups=None, accept=None) -> tuple:
    """Rewrite every phone span of a caption; returns (new text, spans changed). Non-phone spans are untouched.
    accept(before, after) -> bool (optional): a reader-verified guard; a span it refuses stays as written."""
    n = 0

    def sub(m):
        nonlocal n
        s = m.group(0)
        if not phone_span(s):
            return s
        r = reformat(s, groups)
        if r != s and (accept is None or accept(s, r)):
            assert re.sub(r"\D", "", r) == re.sub(r"\D", "", s)
            n += 1
            return r
        return s

    return _SPAN.sub(sub, text), n
