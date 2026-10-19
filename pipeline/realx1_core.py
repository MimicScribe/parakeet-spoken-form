"""Acoustic confirmation of a number reading, number by number (pure: no audio, no Modal, no torch).

Why: a row whose number span was decided mainly by the second transcript (census_common.EXT_BY) has one acoustic
check for its value. The fuzzy fit of ml_decide passes real misreads ("bilhoes" for "milhoes", "por cien" for
"porcion"), so a promotion needs a stricter, per-word test with the same CTC reader the labelling used:

  cand   every other reading of the span (ml_cands, all additions on) must lose to the label's by >= CAND_MIN nats
  del    the row without the span's words must lose by >= SPAN_DEL_MIN nats (the number is in the audio at all)
  wdel   deleting any one word of 3+ letters of the reading must cost >= WDEL_MIN nats (every word is heard:
         a scale word "mil", a unit "euros", a decimal word "coma")
  wsub   replacing any one word of the reading by a near word of the language's number vocabulary (edit distance
         <= 3 on the folded letters, gender / ending variants excepted) must lose by >= WSUB_MIN nats

A span passes when all four hold. Non-ext spans of the same row keep the labelling bar (cand >= MARGIN).
The scorer is injected: nll(text) -> nats (ml_decide.LangCTC.nll over one row's log-probs), so the logic is
testable without a model.
"""

import re

import ml_decide as D

MARGIN = 3.0  # ml_decide.MARGIN: the labelling bar (spans that were decided by the CTC itself)
CAND_MIN = 6.0
SPAN_DEL_MIN = 6.0
WDEL_MIN = 3.0
WSUB_MIN = 4.0
WORD_MIN_LETTERS = 3  # shorter words ("y", "e", "de", "et") are dropped by the reader; they are not tested
MAX_ALTS = 10
MAX_DIST = 3

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.U)
_LEX: dict = {}
_MORPH: dict = {}
COMPOUND = ("de", "nl", "it")


def lexicon(lang: str) -> dict:
    """{folded key: word} of every word of the language's number grammar table (cardinals, ordinals, years, decimals,
    currency, percent, clock readings) -- the vocabulary a misread number word comes from."""
    if lang not in _LEX:
        import ml_cands as M

        out = {}
        for k, v in M.table().items():
            if not k.startswith(lang + "|"):
                continue
            for reading in v:
                for w in str(reading).split():
                    kk = D.key(w)
                    if kk:
                        out.setdefault(kk, w.lower())
        _LEX[lang] = out
    return _LEX[lang]


def morphemes(lang: str) -> dict:
    """Compound languages: {key: morpheme} of the closed morpheme set (compound_split)."""
    if lang not in _MORPH:
        import compound_split as CS

        out = {}
        for w in lexicon(lang).values():
            for m in CS.split_token(w, lang):
                kk = D.key(m)
                if kk:
                    out.setdefault(kk, m.lower())
        _MORPH[lang] = out
    return _MORPH[lang]


def near_words(w: str, lang: str, vocab: dict | None = None) -> list:
    """Words of the vocabulary within MAX_DIST edits of w (folded), nearest first, ending-only variants and the same
    sound dropped; at most MAX_ALTS."""
    vocab = vocab if vocab is not None else lexicon(lang)
    kw = D.key(w)
    out = []
    for kk, u in vocab.items():
        if kk == kw or abs(len(kk) - len(kw)) > MAX_DIST:
            continue
        d = D._lev(kk, kw)
        if d > MAX_DIST or D.ending_only(w.lower(), u):
            continue
        out.append((d, abs(len(kk) - len(kw)), u))
    out.sort()
    return [u for *_, u in out[:MAX_ALTS]]


def units(T: str, a: int, b: int, lang: str) -> list:
    """The testable words of T[a:b]: [{"ta","tb": the token's chars, "word", "i": morpheme index or None, "parts"}].
    A compound token of de / nl / it is tested morpheme by morpheme (parts); any other word as itself."""
    import compound_split as CS

    out = []
    for m in _WORD.finditer(T, a, b):
        tok = m.group(0)
        if lang in COMPOUND:
            parts = CS.split_token(tok, lang)
            if len(parts) > 1 and CS.join_morphemes(parts, lang).lower() == tok.lower():
                for i, p in enumerate(parts):
                    out.append({"ta": m.start(), "tb": m.end(), "word": p, "i": i, "parts": parts})
                continue
        out.append({"ta": m.start(), "tb": m.end(), "word": tok, "i": None, "parts": [tok]})
    return out


def _tidy(t: str) -> str:
    return re.sub(r"  +", " ", t).strip()


def variants(T: str, a: int, b: int, lang: str) -> list:
    """The per-word probes of the reading T[a:b]: [(kind, text)] with kind "wdel" (one word deleted) or "wsub" (one
    word replaced by a near word of the vocabulary)."""
    import compound_split as CS

    out = []
    for u in units(T, a, b, lang):
        w, k = u["word"], D.key(u["word"])
        if len(k) < WORD_MIN_LETTERS:
            continue
        parts = list(u["parts"])
        if u["i"] is None:
            out.append(("wdel", _tidy(T[:u["ta"]] + T[u["tb"]:])))
            vocab = lexicon(lang)
            for alt in near_words(w, lang, vocab):
                out.append(("wsub", T[:u["ta"]] + alt + T[u["tb"]:]))
            continue
        # a compound token: delete / replace one morpheme, join the rest back
        i = u["i"]
        rest = parts[:i] + parts[i + 1:]
        if rest:
            out.append(("wdel", T[:u["ta"]] + CS.join_morphemes(rest, lang) + T[u["tb"]:]))
        for alt in near_words(w, lang, morphemes(lang)):
            new = parts[:i] + [alt] + parts[i + 1:]
            out.append(("wsub", T[:u["ta"]] + CS.join_morphemes(new, lang) + T[u["tb"]:]))
    return out


def span_probe(nll, T: str, a: int, b: int, lang: str, cands: list, chosen_key: str, base: float | None = None) -> dict:
    """Run the four tests on the span T[a:b] (nll(text) -> nats, base = nll(T)). Returns the minimum margin of each test
    ("cand", "del", "wdel", "wsub"; inf = no probe), the probe counts ("n_*") and the weakest probe ("worst")."""
    base = nll(T) if base is None else base
    res = {k: float("inf") for k in ("cand", "del", "wdel", "wsub")}
    res.update(n_cand=0, n_del=1, n_wdel=0, n_wsub=0, worst=None)
    probes = [("cand", T[:a] + c.strip() + T[b:]) for c in cands if D.key(c) != chosen_key and c.strip()]
    probes.append(("del", _tidy(T[:a] + " " + T[b:])))
    probes += variants(T, a, b, lang)
    for kind, text in probes:
        res["n_" + kind] += 1 if kind != "del" else 0
        m = nll(text) - base
        if m < res[kind]:
            res[kind] = m
            if res["worst"] is None or m < res["worst"][2]:
                res["worst"] = (kind, text[max(0, a - 20):b + 25], round(m, 1))
    return res


def span_ok(res: dict, ext_span: bool) -> tuple:
    """(ok, why). An ext-decided span must pass every test at its bar; any other span keeps the labelling bar."""
    if not ext_span:
        return (res["cand"] >= MARGIN, "cand" if res["cand"] < MARGIN else "")
    for k, bar in (("cand", CAND_MIN), ("del", SPAN_DEL_MIN), ("wdel", WDEL_MIN), ("wsub", WSUB_MIN)):
        if res[k] < bar:
            return False, k
    return True, ""
