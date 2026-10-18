"""Shared helpers for the public benchmark: file I/O, the public number reader, cluster keys and bootstrap CIs.

Everything here is stdlib Python. The number reader and the seam slot matcher are imported read-only from
eval/quality/ (number_outcomes.py, score_values.py); nothing in eval/quality/ is copied or changed.
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
QUALITY = REPO / "eval" / "quality"
PIPELINE = REPO / "pipeline"
for _p in (str(QUALITY),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import number_outcomes as N  # noqa: E402  (the public reader)
import score_values as SV  # noqa: E402  (the one seam slot matcher)

# The languages Parakeet v3 / Ultra transcribes (24 European languages + English)
ALL_LANGS = ("en bg cs da de el es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk").split()


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- clusters
def cluster_of(row_id: str, src: str = "") -> str:
    """The resampling unit of a row: rows from one recording (or one read sentence) are not independent.
    VoxPopuli: the parliament day; YODAS: the video; FLEURS: the sentence id (one sentence read by several speakers);
    Earnings: the call; anything else: the row itself."""
    rid = str(row_id)
    m = re.match(r"(\d{8})-\d{4}-[A-Z]", rid)
    if m:
        return "vox:" + m.group(1)
    m = re.match(r"[a-z]{2}\d+_\d+_(.{11})_", rid)
    if m:
        return "yt:" + m.group(1)
    m = re.match(r"fleurs[a-z]*_(\d+)_", rid)
    if m:
        return "fleurs:" + m.group(1)
    m = re.match(r"(\d{5,8})_\d+\.wav$", rid)
    if m:
        return "call:" + m.group(1)
    return (src + ":" if src else "row:") + rid


# ---------------------------------------------------------------- bootstrap
BOOT_N = 2000


def _quantiles(xs, alpha):
    xs = sorted(xs)
    n = len(xs)
    return [xs[int(alpha / 2 * n)], xs[min(n - 1, int((1 - alpha / 2) * n))]]


def ratio_ci(clusters: dict, alpha: float = 0.05, n_boot: int = BOOT_N, seed: int = 0):
    """clusters: key -> (hits, n). Cluster bootstrap CI of 100 * sum(hits) / sum(n)."""
    g = [v for v in clusters.values() if v[1]]
    if not g:
        return [None, None]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        h = n = 0
        for _ in g:
            a, b = rng.choice(g)
            h += a
            n += b
        stats.append(100.0 * h / n if n else 0.0)
    return [round(x, 2) for x in _quantiles(stats, alpha)]


def paired_diff_ci(clusters: dict, alpha: float = 0.05, n_boot: int = BOOT_N, seed: int = 0):
    """clusters: key -> (hits_run, hits_base, n) over the SAME items. Cluster bootstrap CI of the percentage-point
    difference run - base."""
    g = [v for v in clusters.values() if v[2]]
    if not g:
        return [None, None]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        a = b = n = 0
        for _ in g:
            x, y, m = rng.choice(g)
            a += x
            b += y
            n += m
        stats.append(100.0 * (a - b) / n if n else 0.0)
    return [round(x, 2) for x in _quantiles(stats, alpha)]


def pct(a, n):
    return round(100.0 * a / n, 2) if n else None


# ---------------------------------------------------------------- text
def fold_tokens(text: str) -> list:
    """Lower-cased, accent-folded word tokens with punctuation removed (hyphens split words)."""
    t = N._fold(str(text or "")).replace("-", " ")
    t = re.sub(r"[^\w\s%]|_", " ", t)
    return t.split()


def number_values(text: str, lang: str, keep_weak: bool = False) -> list:
    """The reader's values of a text as plain strings ("2010", "12.5")."""
    out = []
    for n in N.extract_numbers(text, lang, keep_weak=keep_weak):
        if n.vals:
            v = min(n.vals)
            out.append(str(int(v)) if v == int(v) else str(v.normalize()))
    return out


# ---------------------------------------------------------------- reader readings (the reader-only parse)
def _reader_vals(text: str, lang: str) -> list:
    """the reader's values like score_values.reader_reading (noon / midnight words too), except that a digit string
    with two meanings takes the larger: "1.000" / "1,500" are thousands (the greedy reading keeps the smaller one)"""
    t = text or ""
    for w, rep in SV.NOON.get(lang, {}).items():
        t = re.sub(rf"(?<![\w-]){re.escape(w)}(?![\w-])", rep, t, flags=re.IGNORECASE)
    out = []
    for n in N.extract_numbers(t, lang, keep_weak=True):
        if n.vals:
            v = max(n.vals) if n.form == "digits" else min(n.vals)
            out.append(str(int(v)) if v == int(v) else str(v))
    return out


def no_joiner_reading(text: str, lang: str):
    """The reader's numbers with every joining word ("and", "und", "y", "si", "и", ...) treated as a break, as a
    score_values.Reading WITH word spans, so the slot matcher's joined-sum rule ("cincizeci si cinci" = 50 + 5) and
    the lenient rules can see where each number stands. Words = whitespace tokens, lower-cased, punctuation stripped.
    A digit group written with a leading zero ("0170") keeps its digits as written: the reader's values drop the
    zero, which would make every phone number that starts with 0 wrong for a digits model under the PH rule; a digit
    string with two meanings ("1.000") takes the larger (_reader_vals)."""
    words = [re.sub(r"^[^\w%]+|[^\w%]+$", "", w).lower() for w in str(text or "").split()]
    words = [w for w in words if w]
    vals, spans = [], []
    seg = []

    def flush():
        if not seg:
            return
        a, b = seg[0], seg[-1]
        got = _reader_vals(" ".join(words[a:b + 1]), lang)
        for k, v in enumerate(got):
            s = a if k == 0 else min(b, a + k)
            e = b if k == len(got) - 1 else min(b, a + k)
            vals.append(v)
            spans.append([s, max(s, e)])
        seg.clear()

    for i, w in enumerate(words):
        if w in SV.JOINERS:
            flush()
        elif re.fullmatch(r"0\d+", w):  # a digit group with a leading zero ("0170"): its digits as written
            flush()
            vals.append(w)
            spans.append([i, i])
        else:
            seg.append(i)
    flush()
    return SV.Reading(vals, spans, words)


def readings_of(text: str, lang: str) -> list:
    """Every reader reading of a text: greedy (number_outcomes.extract_numbers through score_values.reader_reading)
    and no-joiner. A slot counts as right when ANY reading has it right (score_values.match_slots)."""
    return [SV.reader_reading(text, lang), no_joiner_reading(text, lang)]


# ---------------------------------------------------------------- span -> seam-scorer slots
_CLOCK = re.compile(r"(\d{1,2})\s*[:.hH]\s*(\d{2})\b")
DIGIT_RUN_KINDS = {"phone", "code", "chain", "version", "alnum"}


def span_slots(span: dict, lang: str) -> list:
    """A labelled caption number as slots of score_values (base, value): phone / code / chain / version / alnum
    -> PH (its digit string); a clock -> T "h:mm"; a currency amount -> P; a decimal -> D; a percentage -> PCT;
    anything else (and any span the reader splits into several numbers) -> one N per number. [] when the reader
    finds no number in the caption text (the span is then unscorable)."""
    t, kind = span.get("text") or "", span.get("kind") or ""
    vals = number_values(t, lang)
    if kind in DIGIT_RUN_KINDS:
        d = re.sub(r"\D", "", t)
        return [("PH", d)] if d else []
    if kind == "clock":
        m = _CLOCK.search(t)
        if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
            return [("T", f"{int(m.group(1))}:{m.group(2)}")]
    if len(vals) == 1:
        base = {"cur": "P", "dec": "D", "pct": "PCT"}.get(kind, "N")
        return [(base, vals[0])]
    return [("N", v) for v in vals]
