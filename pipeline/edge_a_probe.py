"""Strict acoustic probes for an edge phrase (edge_a rows): pure, the scorer is injected (nll(text) -> nats).

The rules are the number-by-number confirmation of realx1_core (whole-phrase deletion >= 6 nats, any one word deleted
>= 3, any one word replaced by a near word of the pack's vocabulary >= 4) with the fixes the strict re-check of the
synthetic sets found necessary: words the reader rarely hears (connectors, function words) are never tested, a swap TO
such a word is no evidence, and a filler ("uh", "euh", "eh") is held to a weaker bar of its own because it is short and
hesitant by nature. An edge phrase has no digit caption, so there are no rival-reading probes: the near-word swaps
cover a misheard numeral ("seventy" for "seventeen").
"""
import re

import ml_decide as D
import realx1_core as R

DEL_MIN, WDEL_MIN, WSUB_MIN, FILLER_MIN = 6.0, 3.0, 4.0, 2.0
CONN = {"en", "et", "e", "und", "y", "i", "ed", "met"}
FUNC = {"e", "et", "y", "de", "à", "a", "au", "en", "of", "and", "und", "ao", "del", "d", "o", "zu", "met", "u", "ou", "tot", "do",
        "der", "the", "to", "or", "at", "in", "on", "for", "with", "un", "una", "une", "los", "las", "les", "des", "du", "la", "le", "el"}
FILLER = {"uh", "um", "er", "eh", "euh", "heu", "hum", "hmm", "mmm", "este"}
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.U)


def tidy(t: str) -> str:
    return re.sub(r"  +", " ", t).strip()


def words(T: str, a: int, b: int) -> list:
    return [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(T, a, b)]


def skip(w: str) -> bool:
    k = D.key(w)
    return len(k) < R.WORD_MIN_LETTERS or w.lower() in FUNC or w.lower() in CONN


# how a reader spells a spoken filler: the filler is "heard" when ANY of these beats its deletion
FILLER_SPELL = ("uh", "um", "er", "a", "ah", "eh", "euh", "hm", "mm", "eu", "e", "o", "hum", "heu", "este")


def probes(T: str, a: int, b: int, vocab: dict) -> list:
    """[(kind, text, word)] kind in del / wdel / wsub / fdel (filler deleted; the filler probes are scored apart, see strict_span)."""
    out = [("del", tidy(T[:a] + " " + T[b:]), None)]
    for s, e, w in words(T, a, b):
        lw = w.lower()
        if lw in FILLER:
            out.append(("fdel", tidy(T[:s] + T[e:]), (s, e, w)))
            continue
        if skip(w):
            continue
        out.append(("wdel", tidy(T[:s] + T[e:]), w))
        for alt in R.near_words(w, "xx", vocab):
            if alt.lower() in FUNC or alt.lower() in CONN or alt.lower() in FILLER:
                continue
            out.append(("wsub", T[:s] + alt + T[e:], f"{w}>{alt}"))
    return out


def strict_span(nll, T: str, a: int, b: int, vocab: dict, base: float | None = None) -> dict:
    """Minimum margin per probe kind over the edge phrase T[a:b] and the verdict."""
    base = nll(T) if base is None else base
    bars = {"del": DEL_MIN, "wdel": WDEL_MIN, "wsub": WSUB_MIN, "fdel": FILLER_MIN}
    m = {k: None for k in bars}
    worst = {}
    low = []
    for kind, text, w in probes(T, a, b, vocab):
        if kind == "fdel":  # deletion vs the filler under its best spelling (a reader writes "uh" as a / er / um ...)
            fs, fe, w = w
            d = nll(text) - min([base] + [nll(T[:fs] + v + T[fe:]) for v in FILLER_SPELL])
        else:
            d = nll(text) - base
        if m[kind] is None or d < m[kind]:
            m[kind] = d
            worst[kind] = [w, round(d, 1)]
        if d < bars[kind] and len(low) < 8:
            low.append([kind, w, round(d, 1)])
    why = next((k for k in ("del", "wdel", "wsub", "fdel") if m[k] is not None and m[k] < bars[k]), "")
    return {"ok": not why, "why": why, "m": {k: (None if v is None else round(v, 1)) for k, v in m.items()},
            "worst": worst, "low": low}


def edge_vocab(lang: str, texts: list) -> dict:
    """{key: word}: the number grammar's words (es / fr, ml_cands) plus every word of 3+ letters of the pack's own phrases."""
    out: dict = {}
    if lang != "en":
        try:
            out.update(R.lexicon(lang))
        except Exception:  # noqa: BLE001
            pass
    for t in texts:
        for w in WORD.findall(t):
            k = D.key(w)
            if len(k) >= R.WORD_MIN_LETTERS and w.lower() not in FILLER:
                out.setdefault(k, w.lower())
    return out
