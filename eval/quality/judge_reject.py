"""Second-judge filter for labelled rows. usage: judge_reject.py <lang> <rows.jsonl|.json> <ext.jsonl> <out_dir> [--wer 0.25]

Everything is compared BY POSITION. The label and the second-transcript text are aligned (letters, spaces and accents ignored,
Levenshtein); each number reading's word range in the label is located from the caption's span offsets; the judge letters
aligned to that range (plus any inserted letters glued into the same judge word) must equal the reading exactly, and the
judge word next to the range (either side) must not be a number / scale / rival-reading word that the label does not have.
  agree     the judge wrote this reading here
  disagree  anything else (a rival, a missing or extra scale word, a glued extra word, nothing at that position)
Row-level rejects (counted separately, first match wins):
  no_judge        no completed judge transcript (missing row, error, client_error)
  span_mismatch   the stored spans do not match M.spans(caption) of the current grammar (or the reading is not a
                  candidate, or the label cannot be reproduced): cannot locate the reading, so unverifiable
  span_disagree   any span disagrees
  wer             WER between label and judge on the NON-number words > --wer (lost / invented words guard)
Dutch (the second transcript writes digits whatever the settings): the caption is aligned with the judge instead; the judge tokens at
the span position must carry the caption's value (digits, or digit x scale word), or, where the judge wrote words, the
reading. Spoken numbers the judge wrote as digits elsewhere are ignored.
Writes <out_dir>/reject_ids.txt, judge_stats.json and judge_rows.jsonl (per-row verdicts for inspection)."""

import collections
import json
import os
import re
import sys

PIPE = os.environ.get("SPOKEN_FORM_PIPELINE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))
sys.path.insert(0, PIPE)
import ml_cands as M  # noqa: E402
import ml_decide as D  # noqa: E402
import ml_p3_core as C  # noqa: E402

WER_MAX = 0.25
TOKEN = re.compile(r"[^\s'’\-–—]+")
JOINERS = {"und", "y", "e", "et", "en", "con", "de", "di", "d", "of", "and"}
SCALES = {
    "de": {"hundert", "tausend", "million", "millionen", "milliarde", "milliarden"},
    "es": {"cien", "ciento", "cientos", "mil", "millon", "millones", "billon", "billones"},
    "fr": {"cent", "cents", "mille", "milles", "million", "millions", "milliard", "milliards"},
    "it": {"cento", "cent", "mille", "mila", "milione", "milioni", "miliardo", "miliardi"},
    "nl": {"honderd", "duizend", "miljoen", "miljard", "miljoenen", "miljarden"},
    "pt": {"cem", "cento", "centos", "mil", "milhao", "milhoes", "bilhao", "bilhoes"},
}
NL_SCALE_VALUE = {"honderd": 100, "duizend": 1000, "miljoen": 10**6, "miljoenen": 10**6, "miljard": 10**9, "miljarden": 10**9}
# words that can follow / precede a number in the judge text and that a caption may have dropped: percent, decimal point,
# clock-hour units (the unit words of currencies and measures sit inside the span's own rival candidates)
QUALIFIERS = {
    "de": {"prozent", "komma", "uhr"}, "es": {"por", "ciento", "coma", "horas", "hora"}, "fr": {"pour", "cent", "virgule", "heures", "heure"},
    "it": {"per", "cento", "virgola", "ore", "ora"}, "nl": {"procent", "komma", "uur"}, "pt": {"por", "cento", "virgula", "horas", "hora"},
}
COMPOUND_STEMS = {"de": ("tausend", "hundert", "million", "milliard"), "nl": ("duizend", "honderd", "miljoen", "miljard"),
                  "it": ("mila", "cento", "milion", "miliard")}

_NUMWORDS: dict = {}


def numwords(lang: str) -> set:
    """Number words of `lang` (letters only, accents folded) from every "card" entry of the grammar table + scale words;
    joiners ("und", "y", "et" ...) are not number words."""
    if lang not in _NUMWORDS:
        with open(os.path.join(PIPE, "ml_grammar_table.json")) as f:
            tab = json.load(f)
        ws = set()
        for k, v in tab.items():
            kl, kind = k.split("|")[:2]
            if kl == lang and kind == "card":
                for c in v:
                    ws.update(D.key(w) for w in c.split())
        _NUMWORDS[lang] = (ws | SCALES.get(lang, set())) - JOINERS - {""}
    return _NUMWORDS[lang]


def is_numberish(w: str, lang: str) -> bool:
    if w in numwords(lang):
        return True
    return len(w) >= 6 and any(w.startswith(s) or w.endswith(s) for s in COMPOUND_STEMS.get(lang, ()))


def nw(w: str) -> str:
    return D.DIALECT.get(D.norm_word(w), D.norm_word(w))


def tokenize(text: str):
    """[(norm letters, start, end)] for whitespace / apostrophe / hyphen separated pieces that have letters."""
    out = []
    for m in TOKEN.finditer(text or ""):
        k = nw(m.group())
        if k:
            out.append((k, m.start(), m.end()))
    return out


def align(a, b, prefer="diag"):
    """Levenshtein alignment: list over a of the matching index in b (or None) for substitutions and matches alike
    (substitution = aligned), plus the set of b indices that are aligned."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    mp, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if prefer == "ins" and d[i][j] == d[i][j - 1] + 1:
            j -= 1
        elif d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            mp[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mp


def wer(ref: list, hyp: list) -> float:
    return edits(ref, hyp) / len(ref) if ref else 0.0


def edits(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- label side

def resolve_label(row, lang):
    """-> (label, [token ranges], [spans]) or (None, reason). The stored spans must be the ones the current grammar finds in
    the caption (texts equal) and every stored reading must be one of that span's candidates."""
    cap = row["caption"]
    sp = M.spans(cap, lang)
    st = row.get("spans") or []
    if len(sp) != len(st) or any(s.text != i["text"] for s, i in zip(sp, st)):
        return None, "spans differ from M.spans(caption)"
    picks = []
    for s, i in zip(sp, st):
        ks = [D.key(c) for c in s.cands]
        k = D.key(i["reading"])
        if k not in ks:
            return None, f"reading {i['reading']!r} not a candidate of {s.text!r}"
        picks.append(ks.index(k))
    offs = []
    lab = C.render_cap(cap, sp, picks, offs)
    if lab.lower() != (row.get("text") or "").lower():
        # the stored label was post-processed: keep it, find each reading in it left to right
        lab, offs, pos = row.get("text") or "", [], 0
        for s, i in zip(sp, st):
            r = i["reading"].strip().lower()
            a = lab.lower().find(r, pos)
            if a < 0:
                return None, "label does not contain the stored reading"
            offs.append([a, a + len(r)])
            pos = a + len(r)
    return lab, offs, sp


def token_ranges(toks, offs):
    out = []
    for a, b in offs:
        idx = [i for i, (_, s, e) in enumerate(toks) if s < b and a < e]
        out.append((idx[0], idx[-1] + 1) if idx else None)
    return out


# ---------------------------------------------------------------- judge side

def judge_tokens(a):
    words = a.get("words")
    text = " ".join(w["w"] for w in words) if words else (a.get("text") or "")
    return tokenize(text)


def neighbours(toks, i, step, n=2):
    """The first n tokens from index i going `step` that are not joiners ("e", "y", "und" ...)."""
    out = []
    while 0 <= i < len(toks) and len(out) < n:
        if toks[i][0] not in JOINERS:
            out.append(toks[i][0])
        i += step
    return out


def letter_variants(lang: str, span_text: str, reading: str) -> set:
    """Keys of `reading` with any letter-name word of a letter that the span text writes (G20 -> "ge zwanzig") collapsed
    to the bare letter, as the second transcript writes acronym letters ("g zwanzig", "co deux" for "cé o deux"). The
    number words and the letters themselves stay compared exactly (audit 2026-10-17: ~half of alnum span_disagree)."""
    letters = {c for c in span_text.lower() if "a" <= c <= "z"}
    names = []  # (word keys of a letter name, the letter)
    for c in letters:
        for nm in M.LETTERS.get(lang, {}).get(c, []):
            names.append(([nw(w) for w in nm.split()], c))
    ws = [nw(w) for w in reading.split()]
    ws = [w for w in ws if w]
    paths = [(0, "")]
    done = set()
    while paths and len(done) < 64:
        i, acc = paths.pop()
        if i == len(ws):
            done.add(acc)
            continue
        paths.append((i + 1, acc + ws[i]))
        for nk, c in names:
            if ws[i:i + len(nk)] == nk and nk != [c]:
                paths.append((i + len(nk), acc + c))
    return done - {""}


def is_synthetic(row) -> bool:
    """Synthetic speech rows (the audio was generated from the label text): ids of the ml_tts sets."""
    return bool(re.match(r"tts\w*_[a-z]{2}_", str(row.get("id", ""))))


ROMAN = re.compile(r"^[ivxlcdm]+$")


def verdict_for_range(lang, ltoks, jtoks, mp_chars, lchars, jchars, lw, jw, lo, hi, reading_key, rivals, alts=frozenset(),
                      synthetic=False, span_text=""):
    """Position check for label tokens [lo, hi). mp_chars: optimal letter alignments (label char -> judge char or None);
    letters inserted next to a repeated letter ("cinco" + "porciento") can be placed in several equally good ways, so the
    reading agrees when ANY optimal alignment puts exactly its letters on the range."""
    first = None
    for mp_char in mp_chars:
        r = _verdict(lang, ltoks, jtoks, mp_char, lchars, jchars, lw, jw, lo, hi, reading_key, rivals, alts, synthetic,
                     span_text)
        if r[0] == "agree":
            return r
        first = first or r
    return first


def _verdict(lang, ltoks, jtoks, mp_char, lchars, jchars, lw, jw, lo, hi, reading_key, rivals, alts=frozenset(),
             synthetic=False, span_text=""):
    pos = [p for p in range(len(lchars)) if lo <= lw[p] < hi]
    expected = "".join(ltoks[i][0] for i in range(lo, hi))
    if expected != reading_key:
        return "disagree", f"label range {expected!r} != reading {reading_key!r}", None
    js = [mp_char[p] for p in pos if mp_char[p] is not None]
    if not js:
        return "disagree", "judge has nothing at this position", None
    used = {x for x in mp_char if x is not None}
    a, b = min(js), max(js)
    # judge letters inserted next to the range but glued into the same judge word belong to it ("einhundert" vs "hundert")
    while a - 1 >= 0 and jw[a - 1] == jw[a] and a - 1 not in used:
        a -= 1
    while b + 1 < len(jchars) and jw[b + 1] == jw[b] and b + 1 not in used:
        b += 1
    got = "".join(jchars[a:b + 1])
    span = (jw[a], jw[b] + 1)
    if got != reading_key and got not in alts and not (synthetic and _synthetic_same(lang, jtoks, span, ltoks, lo, hi,
                                                                                         span_text)):
        return "disagree", f"judge wrote {got!r}", span
    ban = numwords(lang) | rivals | QUALIFIERS.get(lang, set())
    for step, side in ((1, "after"), (-1, "before")):
        nl = neighbours(ltoks, hi if step == 1 else lo - 1, step)
        nj = (neighbours(jtoks, span[1] if step == 1 else span[0] - 1, step, 1) or [None])[0]
        # a judge word the label has within its next two words is a dropped / swapped word, a matter for the WER guard
        if nj and nj not in nl and (nj in ban or is_numberish(nj, lang)):
            return "disagree", f"judge has {nj!r} {side} the reading, label has {nl[:1]}", span
    return "agree", "", span


def _synthetic_same(lang, jtoks, span, ltoks, lo, hi, span_text) -> bool:
    """Synthetic rows only: the speech was generated from the label, so the second transcript's own habits inside the
    span are not label errors: joiners it adds or drops ("seis e dezenove" for "seis dezenove", "un millón y"), the
    hour / percent words it adds to clock times ("treize heures trente"), compound spacing, German "ein" before
    hundert / tausend (the voice read the label's form), and a Roman numeral written as the caption
    writes it ("xii" for XII). Every number word must still be the same, in order."""
    q = QUALIFIERS.get(lang, set())
    # hour / percent words only where the judge ADDED them: a label that says "quinze hora e meia" (a generator slip
    # for "horas") is compared word for word (spot check 2026-10-17)
    skip_l = JOINERS
    skip_j = JOINERS | (set() if any(ltoks[i][0] in q for i in range(lo, hi)) else q)
    ein = (lambda w: re.sub(r"^ein(?=hundert|tausend)", "", w)) if lang == "de" else (lambda w: w)  # noqa: E731
    jw_ = "".join(ein(jtoks[i][0]) for i in range(*span) if jtoks[i][0] not in skip_j)
    lw_ = "".join(ein(ltoks[i][0]) for i in range(lo, hi) if ltoks[i][0] not in skip_l)
    if jw_ and jw_ == lw_:
        return True
    rom = re.sub(r"[^a-z]", "", span_text.lower())
    return bool(rom) and ROMAN.match(rom) is not None and [jtoks[i][0] for i in range(*span)] == [rom]


def judge_row_romance(row, a, lang):
    res = resolve_label(row, lang)
    if res[0] is None:
        return {"status": "span_mismatch", "why": res[1]}
    lab, offs, sp = res
    ltoks = tokenize(lab)
    jtoks = judge_tokens(a)
    lchars = [c for k, _, _ in ltoks for c in k]
    lw = [i for i, (k, _, _) in enumerate(ltoks) for _ in k]
    jchars = [c for k, _, _ in jtoks for c in k]
    jw = [i for i, (k, _, _) in enumerate(jtoks) for _ in k]
    mp = [align(lchars, jchars, "diag"), align(lchars, jchars, "ins")]
    rngs = token_ranges(ltoks, offs)
    spans_out, covered_l, covered_j = [], set(), set()
    for s, info, rg in zip(sp, row["spans"], rngs):
        if rg is None:
            spans_out.append({"text": s.text, "v": "disagree", "why": "no label token for the span"})
            continue
        pk = D.key(info["reading"])
        rivals = {nw(w) for c in s.cands if D.key(c) != pk for w in c.split()} - {""}
        alts = letter_variants(lang, s.text, info["reading"]) - {pk}
        v, why, jspan = verdict_for_range(lang, ltoks, jtoks, mp, lchars, jchars, lw, jw, rg[0], rg[1], pk, rivals, alts,
                                          is_synthetic(row), s.text)
        spans_out.append({"text": s.text, "reading": info["reading"], "kind": info.get("kind"), "v": v, "why": why})
        covered_l.update(range(*rg))
        if jspan:
            covered_j.update(range(*jspan))
    ref = [k for i, (k, _, _) in enumerate(ltoks) if i not in covered_l]
    # a span's place stays marked while stutters are collapsed: "weniger als [zwanzig prozent] weniger als" is no repeat
    hyp_m = [k if i not in covered_j else SPAN_MARK for i, (k, _, _) in enumerate(jtoks)]
    hyp_m = [k for n, k in enumerate(hyp_m) if k != SPAN_MARK or n == 0 or hyp_m[n - 1] != SPAN_MARK]
    hyp = respace(ref, judge_side_repeats(ref, hyp_m))
    return {"status": "ok", "spans": spans_out, "wer": wer(ref, hyp), "wer_edits": edits(ref, hyp), "label": lab,
            "lost_extra": lost_or_extra(ref, hyp), "negation": negation_mismatch(lang, ref, hyp)}


def collapse_repeats(ws: list, n_max: int = 3) -> list:
    """An immediately repeated run of 1-3 words said once ("structurellement structurellement", "para para", "poi bisogna
    poi bisogna"): a stutter the edited caption dropped is not a lost word (audit 2026-10-17). Applied to both sides."""
    out = []
    for w in ws:
        out.append(w)
        for n in range(1, n_max + 1):
            if len(out) >= 2 * n and out[-n:] == out[-2 * n:-n] and SPAN_MARK not in out[-n:]:
                del out[-n:]
                break
    return out


SPAN_MARK = "\x00"


def judge_side_repeats(ref: list, hyp: list) -> list:
    """The judge's words with its stutters collapsed, when that brings them closer to the label. Only the judge side:
    a repeat the label has and the judge did not hear stays an extra (invented) word (spot check 2026-10-17). `hyp` may
    hold SPAN_MARK where number spans were taken out; the marks are dropped from the result."""
    plain = [k for k in hyp if k != SPAN_MARK]
    hc = [k for k in collapse_repeats(hyp) if k != SPAN_MARK]
    return hc if hc != plain and edits(ref, hc) < edits(ref, plain) else plain


def respace(ref: list, hyp: list) -> list:
    """hyp with a run of words re-split the way ref splits it when both sides have the same letters ("nochmal" /
    "noch mal", "corapporteurs" / "co rapporteurs"): compound spacing is not a word error (yield audit 2026-10-16)."""
    import difflib

    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        out += ref[i1:i2] if op == "replace" and "".join(ref[i1:i2]) == "".join(hyp[j1:j2]) else hyp[j1:j2]
    return out


# ---------------------------------------------------------------- Dutch: digits

# a date / clock / version figure ("9.12.2003", "0:30", "6.1.12") is several numbers; a thousands figure ("1.200",
# "40.529,59") is one
PARTS = re.compile(r"\d+(?:[.:/]\d+)+")
THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")


def dtok(text, split_in=None):
    """[(key, start, end)] where a token with digits has key '#<digits>' and a word its letters; punctuation-only dropped.
    Date / clock / version figures are split into their parts on both sides (audit 2026-10-17: "9.12.2003" was one
    token aligned to the judge's "2003", so the day and month spans were judged against the year)."""
    out = []
    for m in TOKEN.finditer(text or ""):
        t = m.group()
        dg = re.sub(r"\D", "", t)
        core = t.strip(".,;:!?()\"'")
        # caption side (split_in = char ranges of its date / clock / version spans): only those figures split, so a
        # ratio "3:2" (said "drie tot twee") stays one token; judge side: any such figure
        ok = split_in is None or any(a < m.end() and m.start() < b for a, b in split_in)
        if dg and ok and PARTS.fullmatch(core) and not THOUSANDS.fullmatch(core):
            off = m.start() + t.index(core)
            for q in re.finditer(r"\d+", core):
                out.append(("#" + q.group(), off + q.start(), off + q.end()))
        elif dg:
            out.append(("#" + dg, m.start(), m.end()))
        elif nw(t):
            out.append((nw(t), m.start(), m.end()))
    return out


def judge_row_nl(row, a):
    lang = "nl"
    cap = row["caption"]
    sp = M.spans(cap, lang)
    st = row.get("spans") or []
    if len(sp) != len(st) or any(s.text != i["text"] for s, i in zip(sp, st)):
        return {"status": "span_mismatch", "why": "spans differ from M.spans(caption)"}
    words = a.get("words")
    jtext = " ".join(w["w"] for w in words) if words else (a.get("text") or "")
    split = [(s.start, s.end) for s, i in zip(sp, st) if i.get("kind") in ("date", "clock", "version")]
    # the judge's figures split only when the caption has such a span: "voor dertien uur" vs the judge's "13.15" must
    # stay one value (13:15 != 13; spot check 2026-10-17)
    ctoks, jtoks = dtok(cap, split), dtok(jtext, None if split else [])
    mp = align([k for k, _, _ in ctoks], [k for k, _, _ in jtoks])
    aligned = {x for x in mp if x is not None}
    spans_out, covered_c, covered_j = [], set(), set()
    run = "".join(re.sub(r"\D", "", s.text) for s, i in zip(sp, st) if i.get("kind") in ("phone", "code", "chain"))
    # runs of adjacent judge digit tokens ("085 0043063" -> "0850043063"); a phone / code must equal one run exactly
    jruns, cur = set(), ""
    for k, _, _ in jtoks + [("", 0, 0)]:
        if k.startswith("#"):
            cur += k[1:]
        elif cur:
            jruns.add(cur)
            cur = ""
    for s, info in zip(sp, st):
        idx = [i for i, (_, x, y) in enumerate(ctoks) if x < s.end and s.start < y]
        covered_c.update(idx)
        want = re.sub(r"\D", "", s.text)
        js = [mp[i] for i in idx if mp[i] is not None]
        if info.get("kind") in ("phone", "code", "chain") and len(run) >= 4 and run in jruns:
            # the digits of all the row's phone / code spans, in order, in the judge's digit stream ("082760592")
            spans_out.append({"text": s.text, "reading": info["reading"], "kind": info.get("kind"), "v": "agree", "why": ""})
            covered_j.update(i for i, (k, _, _) in enumerate(jtoks) if k.startswith("#") and k[1:] in run)
            continue
        if not js:
            spans_out.append({"text": s.text, "reading": info["reading"], "kind": info.get("kind"), "v": "disagree",
                              "why": "judge has nothing at this position"})
            continue
        lo, hi = min(js), max(js)
        # judge tokens inserted at the edges that are digits or number words belong to the number
        while lo - 1 >= 0 and lo - 1 not in aligned and (jtoks[lo - 1][0].startswith("#") or is_numberish(jtoks[lo - 1][0], lang)):
            lo -= 1
        while hi + 1 < len(jtoks) and hi + 1 not in aligned and (jtoks[hi + 1][0].startswith("#") or is_numberish(jtoks[hi + 1][0], lang)):
            hi += 1
        # judge words the reading itself says ("euro", "cent", "uur") belong to the span ("21,25 euro" for "eenentwintig
        # euro en vijfentwintig cent")
        rw = {nw(w) for w in info["reading"].split()} - {""}
        while lo - 1 >= 0 and lo - 1 not in aligned and jtoks[lo - 1][0] in rw:
            lo -= 1
        while hi + 1 < len(jtoks) and hi + 1 not in aligned and jtoks[hi + 1][0] in rw:
            hi += 1
        reg = jtoks[lo:hi + 1]
        covered_j.update(range(lo, hi + 1))
        digs = [k[1:] for k, _, _ in reg if k.startswith("#")]
        wds = [k for k, _, _ in reg if not k.startswith("#")]
        v, why = "disagree", ""
        if digs:
            val = "".join(digs)
            scale = 1
            for w in wds:
                scale *= NL_SCALE_VALUE.get(w, 1)
            if val == want or (len(digs) == 1 and scale > 1 and want.isdigit() and int(digs[0]) * scale == int(want)):
                v = "agree"
            else:
                why = f"judge wrote {' '.join(k for k, _, _ in reg)!r}, caption value {want}"
        else:
            if "".join(wds) == D.key(info["reading"]):
                v = "agree"
            else:
                why = f"judge wrote {' '.join(wds)!r}, reading {info['reading']!r}"
        spans_out.append({"text": s.text, "reading": info["reading"], "kind": info.get("kind"), "v": v, "why": why})
    ref = [k for i, (k, _, _) in enumerate(ctoks) if i not in covered_c]
    # the judge writes spoken numbers (and "procent") its own way anywhere in the sentence: not word errors
    skip = lambda k: k.startswith("#") or is_numberish(k, lang) or k in QUALIFIERS["nl"]  # noqa: E731
    ref = [k for k in ref if not skip(k)]
    hyp = judge_side_repeats(ref, [SPAN_MARK if i in covered_j or skip(k) else k for i, (k, _, _) in enumerate(jtoks)])
    return {"status": "ok", "spans": spans_out, "wer": wer(ref, hyp), "wer_edits": edits(ref, hyp),
            "lost_extra": lost_or_extra(ref, hyp), "negation": negation_mismatch(lang, ref, hyp)}


# ---------------------------------------------------------------- row / driver

FILLERS = {"ähm", "äh", "ehm", "euh", "heu", "hum", "mmm", "uhm", "ahm", "eeh", "bueno", "pues", "também", "allora",
           "dunque", "alors", "quindi", "also", "nou", "zeg", "ecco", "então", "okay", "vale", "voilà", "genau",
           # discourse particles the edited captions drop ("so" / "I mean" class; audit 2026-10-17)
           "donc", "ouais", "diciamo", "appunto", "cioè", "cioe"}


# short meaning-bearing words the length rule (>= 4 letters) would let through: negations. Counted whatever their
# length; keys as nw() writes them (accents folded). Italian "né" is left out: it folds to the pronoun "ne".
NEGATIONS = {
    "de": {"nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines", "nie", "niemals", "nichts", "niemand",
           "weder", "nirgends", "nix"},
    "es": {"no", "ni", "nunca", "nada", "nadie", "ningun", "ninguno", "ninguna", "tampoco", "jamas"},
    "fr": {"ne", "n", "pas", "non", "jamais", "rien", "aucun", "aucune", "ni", "guere"},
    "it": {"non", "mai", "nessuno", "nessuna", "nessun", "niente", "nulla", "neanche", "nemmeno", "neppure"},
    "nl": {"niet", "geen", "nooit", "niets", "niks", "niemand", "nergens", "noch"},
    "pt": {"nao", "nem", "nunca", "nada", "ninguem", "nenhum", "nenhuma", "jamais"},
}
NEG_SAME = {"n": "ne", "niks": "niets", "nix": "nichts"}  # French elided "n'" is "ne"; colloquial "niks" / "nix"


def negation_mismatch(lang: str, ref: list, hyp: list) -> str | None:
    """A negation the judge heard and the label lacks, or the reverse (quality bar: a lost or invented "não" /
    "nicht" / "ne" flips the meaning; the 4-letter rule never saw them). Counts, not positions: the label must say each
    negation word as often as the judge heard it."""
    ng = NEGATIONS.get(lang, set())

    def c(ws):
        # French "on n'a" / "on a" sound the same (liaison): an elided "n'" right after "on" is not counted
        return collections.Counter(NEG_SAME.get(w, w) for k, w in enumerate(ws) if w in ng
                                   and not (lang == "fr" and w == "n" and k > 0 and ws[k - 1] == "on"))
    a, b = c(ref), c(hyp)
    if a == b:
        return None
    lost = b - a
    return ("lost " + " ".join(sorted(lost.elements()))) if lost else ("extra " + " ".join(sorted((a - b).elements())))


def lost_or_extra(ref: list, hyp: list) -> str | None:
    """A whole content word (>= 4 letters, not a filler) the judge heard but the label lacks (the label
    would teach a deletion) or the label has but the judge did not hear (an insertion). Substitutions (spelling) don't
    count."""
    import difflib

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op == "insert" and any(len(w) >= 4 and w not in FILLERS for w in hyp[j1:j2]):
            return "lost_word"
        if op == "delete" and any(len(w) >= 4 and w not in FILLERS for w in ref[i1:i2]):
            return "extra_word"
    return None


_TTS = None


def synthetic_bad_reading(row, lang) -> str | None:
    """Synthetic rows only: a reading the synthetic planner's own rule now refuses (feminine or apocopated number forms
    outside a feminine count, masculine clock hours: "Boeing setecentas e trinta e sete", "vinte e duas de julho",
    "un heure quarante"; judge audit 2026-10-17). The voice says the label, so the transcript cannot catch these."""
    global _TTS
    if not is_synthetic(row) or lang not in ("es", "pt", "it", "fr"):
        return None
    if _TTS is None:
        import ml_tts as _T  # noqa: PLC0415  (pipeline module; heavy, so only for synthetic rows)
        _TTS = _T
    cls = row.get("cls")
    g = "f" if cls in _TTS.FEM_OK_CLS else "m"  # the clause noun is not stored: a count / amount may be feminine
    for sp in row.get("spans") or []:
        s = type("Sp", (), {"text": sp.get("text", ""), "kind": sp.get("kind")})()
        if not _TTS._gender_ok(lang, cls, g, s, (sp.get("reading") or "").strip()):
            return f"{sp.get('text')!r} read {sp.get('reading')!r}"
    return None


def judge_row(row, a, lang, wer_max=WER_MAX):
    """-> dict with 'reason' (None = keep) and details."""
    bad = synthetic_bad_reading(row, lang)
    if bad:
        return {"reason": "bad_reading", "why": bad}
    if not a or a.get("status") != "completed" or not (a.get("text") or a.get("words")):
        return {"reason": "no_judge", "judge_status": (a or {}).get("status", "missing")}
    r = judge_row_nl(row, a) if lang == "nl" else judge_row_romance(row, a, lang)
    if r["status"] == "span_mismatch":
        return {"reason": "span_mismatch", "why": r["why"]}
    if any(s["v"] != "agree" for s in r["spans"]):
        r["reason"] = "span_disagree"
    elif r.get("negation"):
        r["reason"] = "negation"
    elif r["wer"] > wer_max and r["wer_edits"] >= 2:
        r["reason"] = "wer"
    elif r.get("lost_extra"):
        r["reason"] = r["lost_extra"]
    elif r["wer"] > wer_max:
        r["reason"] = None
        r["wer_borderline"] = True  # one edit (a filler or a short word) in a very short row: reported, not rejected
    else:
        r["reason"] = None
    return r


def load_rows(p):
    with open(p) as f:
        t = f.read()
    if t.lstrip().startswith("["):
        return json.loads(t)
    return [json.loads(l) for l in t.splitlines() if l.strip()]


def load_ext(p):
    """Last line per id wins, so a retry appended later replaces an earlier failure."""
    out = {}
    with open(p) as f:
        for l in f:
            if l.strip():
                j = json.loads(l)
                out[j["id"]] = j
    return out


def main(lang, rows_p, ext_p, out, wer_max=WER_MAX):
    rows, ext = load_rows(rows_p), load_ext(ext_p)
    os.makedirs(out, exist_ok=True)
    rej, reasons = [], collections.Counter()
    spans, kinds, wers, detail = collections.Counter(), collections.defaultdict(collections.Counter), [], []
    for r in rows:
        res = judge_row(r, ext.get(r["id"]), lang, wer_max)
        reasons[res["reason"] or "kept"] += 1
        for s in res.get("spans", []):
            spans[s["v"]] += 1
            kinds[s.get("kind")][s["v"]] += 1
        if "wer" in res:
            wers.append(res["wer"])
        if res["reason"]:
            rej.append(r["id"])
        detail.append({"id": r["id"], "caption": r["caption"], "label": r["text"], "judge": (ext.get(r["id"]) or {}).get("text"), **res})
    open(f"{out}/reject_ids.txt", "w").write("\n".join(rej) + ("\n" if rej else ""))
    with open(f"{out}/judge_rows.jsonl", "w") as f:
        for d in detail:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    ws = sorted(wers)
    q = lambda p: round(ws[min(len(ws) - 1, int(p * len(ws)))], 3) if ws else None  # noqa: E731
    res = {"lang": lang, "rows": len(rows), "rejected": len(rej), "reject_rate": round(len(rej) / max(1, len(rows)), 4),
           "row_reasons": dict(reasons), "spans": dict(spans), "by_kind": {str(k): dict(v) for k, v in kinds.items()},
           "non_number_wer": {"threshold": wer_max, "p50": q(0.5), "p75": q(0.75), "p90": q(0.9), "p95": q(0.95), "max": q(1.0),
                              "over_threshold": sum(w > wer_max for w in wers),
                              "borderline_one_edit_kept": sum(1 for d in detail if d.get("wer_borderline"))}}
    json.dump(res, open(f"{out}/judge_stats.json", "w"), indent=1)
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    w = float(sys.argv[sys.argv.index("--wer") + 1]) if "--wer" in sys.argv else WER_MAX
    if "--wer" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--wer") + 1])
    main(*args[:4], wer_max=w)
