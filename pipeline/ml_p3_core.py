"""Pure (no Modal, no torch) parts of retrain-4 P3: row building from a reading decision, quiet-edge flag, stats."""

import re
import unicodedata
from collections import Counter, defaultdict

import ml_cands as M

MAX_CER = 0.25


# Latin and Cyrillic letters both count, in every caller (windows, seam halves and the TTS check never set a language);
# accents fold first (NFD), so ё / ї / й reduce to е / і / и
_NON_LETTER = re.compile(r"[^a-zа-яієґ]")


def set_script(lang: str) -> None:  # kept for callers; the letter set no longer depends on the language
    return None


def letters(s: str) -> str:
    s = s.lower().replace("ß", "ss")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return _NON_LETTER.sub("", s)


def cer(a: str, b: str) -> float:
    a, b = letters(a), letters(b)
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def edge_quiet(x) -> bool:
    """mix_build.trim_edges rule: a quiet (< 0.3 x speech level) 20 ms frame in each outer 160 ms."""
    import numpy as np

    n = len(x) // 320
    if n < 20:
        return False
    fr = np.sqrt(np.mean(np.asarray(x[:n * 320]).reshape(n, 320) ** 2, axis=1) + 1e-12)
    speech = np.median(fr[fr > np.percentile(fr, 30)])
    q = 0.3 * speech
    return bool(fr[:8].min() <= q and fr[-8:].min() <= q)


ABBR = re.compile(r"(?:^|\s)(?:[^\W\d_]{1,3}|(?:[^\W\d_]\.){2,})\.\s*$")  # "Nr.", "Sr.", "ca.", "z.B." before the span
WF = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°⁇²³¹½¼¾⅓⅔ºª‰&]|,-|\s[x×]\s")  # ft_train.WF_CHARS + ordinal marks, "15,-", "3 x 4"
# abbreviations written next to a reading ("fünf Mio. EUR", "Nr. elf", "ca. dreitausend", "n. undici"); a plain
# sentence-final word ("mehr.") is not one
ABBR_NEAR = re.compile(r"^(?:(?:mio|mrd|mill|mld|mln|nr|no|núm|num|n|ca|art|vol|cap|pág|pag|pp|p|s|st|bzw|tsd|"
                       r"km|kg|std|min|sek|ha|eur|usd)\.|n\.?º|EUR|USD|GBP|BRL|CHF)$", re.I)
# ru / uk abbreviations a reading may leave next to it ("пять тыс.", "в тысяча девятьсот тридцать девятом г.")
ABBR_NEAR_SLAV = re.compile(r"^(?:тыс|тис|млн|млрд|трлн|руб|грн|г|гг|р|рр|ст|стр|кв|коп)\.?$|^№$", re.I)
CER_BY_SRC = {"yodas": 0.12, "vox": 0.15}  # review H1: machine captions (YODAS) need the tighter bound


def _opens_sentence(before: str) -> bool:
    b = before.rstrip()
    return not b or (b[-1] in ".!?" and not ABBR.search(b))


def render_cap(cap, sp, picks, offsets=None):
    """M.render, capitalising a reading that opens a sentence ("2009 wurde" -> "Zweitausendneun wurde"). `offsets`, when
    a list, receives the [start, end) character range of each reading in the label (leading space excluded)."""
    out, pos = [], 0
    for s, c in zip(sp, picks):
        out.append(cap[pos:s.start])
        w = s.cands[c]
        if not w.startswith(" ") and _opens_sentence("".join(out)):  # review M7: not after "Nr." or a previous reading
            w = w[:1].upper() + w[1:]
        if offsets is not None:
            a = len("".join(out)) + (len(w) - len(w.lstrip()))
            offsets.append([a, a + len(w.strip())])
        out.append(w)
        pos = s.end
    out.append(cap[pos:])
    return "".join(out)


def number_tokens(label: str, offsets) -> list[bool]:
    """Per whitespace token of the label: True when it lies inside a number reading."""
    flags, pos = [], 0
    for tok in label.split():
        a = label.index(tok, pos)
        b = a + len(tok)
        pos = b
        flags.append(any(x < b and a < y for x, y in offsets))
    return flags


MIN_GAP = 0.10  # s between two words' CTC letter frames: letter spikes are peaky, a word's sound runs 40-100 ms past them
MIN_CROP = 1.0  # s
CROP_MAX_CER = 0.10
QUIET = 0.3  # a cut frame must be below 0.3 x the row's speech level (the edge_quiet rule)


def quiet_cut(x, t0: float, t1: float) -> float | None:
    """The centre of the quietest 10 ms frame inside [t0, t1] s, or None when no frame there is quiet."""
    import numpy as np

    hop = 160
    n = len(x) // hop
    if n < 20:
        return None
    fr = np.sqrt(np.mean(np.asarray(x[:n * hop]).reshape(n, hop) ** 2, axis=1) + 1e-12)
    speech = np.median(fr[fr > np.percentile(fr, 30)])
    a, b = int(t0 * 100), min(int(t1 * 100), n)
    if b <= a:
        return None
    i = a + int(np.argmin(fr[a:b]))
    return (i + 0.5) / 100 if fr[i] <= QUIET * speech else None


def plan_crops(n_tok: int, flags: list[bool], times: list, x, rng, max_crops: int = 2) -> list[dict]:
    """Window-edge crops of a labelled row: cut at word boundaries inside a number reading ("in_start"/"in_end": the
    window starts or ends between two number words) or at a number's edge ("num_start"/"num_end"). times[i] = (t0, t1)
    of token i or None; x = the row's 16 kHz audio. A cut needs >= MIN_GAP between the two words' CTC letters and is
    placed at the quietest 10 ms frame of that gap, which must be quiet (review H3). Each crop keeps whole words. At
    most one start-type and one end-type crop per row, number-inside cuts first."""
    dur = len(x) / 16000
    starts, ends = {"in": [], "num": []}, {"in": [], "num": []}
    for k in range(1, n_tok):
        if times[k - 1] is None or times[k] is None or times[k][0] - times[k - 1][1] < MIN_GAP:
            continue
        cut = quiet_cut(x, times[k - 1][1], times[k][0])
        if cut is None:
            continue
        if flags[k - 1] and flags[k]:
            starts["in"].append((k, cut))
            ends["in"].append((k, cut))
        elif flags[k] and not flags[k - 1]:
            starts["num"].append((k, cut))
        elif flags[k - 1] and not flags[k]:
            ends["num"].append((k, cut))
    out = []
    for side, pool in (("start", starts), ("end", ends)):
        for typ in ("in", "num"):
            cands = [(k, c) for k, c in pool[typ]
                     if (dur - c if side == "start" else c) >= MIN_CROP and (n_tok - k if side == "start" else k) >= 2]
            if cands:
                k, c = cands[rng.randrange(len(cands))]
                out.append({"type": f"{typ}_{side}", "a": c if side == "start" else 0.0, "b": dur if side == "start" else c,
                            "i0": k if side == "start" else 0, "i1": n_tok if side == "start" else k})
                break
    return out[:max_crops]


def edges_match(label: str, greedy: str, cut_left: str | None = None, cut_right: str | None = None) -> bool:
    """The re-decoded crop starts and ends with the label's first and last words (letters only, accents folded): no
    clipped edge word, no leftover tail of a neighbouring word (review H3). For a window whose edge cut a word
    (cut_left / cut_right), a greedy edge token that is a piece of that word (its tail on the left, its head on the
    right) is allowed: the label leaves the fragment out on purpose (review M2)."""
    lw = [letters(w) for w in label.split() if letters(w)]
    gw = [letters(w) for w in greedy.split() if letters(w)]
    cl, cr = letters(cut_left or ""), letters(cut_right or "")
    if gw and lw and cl and gw[0] != lw[0] and len(gw[0]) >= 2 and cl.endswith(gw[0]) and gw[0] != cl:
        gw = gw[1:]
    if gw and lw and cr and gw[-1] != lw[-1] and len(gw[-1]) >= 2 and cr.startswith(gw[-1]) and gw[-1] != cr:
        gw = gw[:-1]
    return bool(lw and gw) and lw[0] == gw[0] and lw[-1] == gw[-1]


def word_gap(label: str, greedy: str) -> int:
    """Longest run of consecutive words that one side has and the other lacks entirely (difflib insert / delete on
    letters-only words). Spelling slips are replace ops and do not count."""
    import difflib

    a = [letters(w) for w in label.split() if letters(w)]
    b = [letters(w) for w in greedy.split() if letters(w)]
    best = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "insert":
            best = max(best, j2 - j1)
        elif op == "delete":
            best = max(best, i2 - i1)
    return best


def make_row(row: dict, r: dict, x=None, ext: str | None = None) -> dict:
    """r = ml_decide.decide(...). Returns {"row": new row} or {"reason": why dropped}.
    ext (optional): an external verbatim transcript of the same audio. The row CER and caption-gap guards then take the
    better of the two transcripts (2026-10-16 yield audit: the xlsr greedy garbles short YouTube rows, and 73% of the
    rows dropped on CER vs the greedy passed the external positional judge). Number decisions are not affected."""
    cap = row["text"]
    set_script(row["lang"])
    sp = M.spans(cap, row["lang"])
    ds = r["spans"]
    if not sp or len(sp) != len(ds):
        return {"reason": "no_spans" if not sp else "span_mismatch"}
    # the readings the decision chose from (decide may have listed fewer: ml_decide.spans_for)
    from dataclasses import replace

    sp = [replace(s, cands=d["cands"]) if "cands" in d else s for s, d in zip(sp, ds)]
    bad = [d for d in ds if not d["decided"]]
    if bad:
        return {"reason": "no_candidates" if any(d.get("why") for d in bad) else "undecided"}
    picks = [d["pick"] for d in ds]
    offs = []
    label = render_cap(cap, sp, picks, offs)
    if re.search(r"\d", label):
        return {"reason": "digit_left"}
    if WF.search(label):  # a leftover symbol ("A$ cinquenta") would be both a target and a wf_pen target (review L4)
        return {"reason": "symbol_left"}
    toks = label.split()
    flags = number_tokens(label, offs)
    for i, t in enumerate(toks):  # review B3: an abbreviation written next to a reading ("fünf Mio. EUR", "Nr. elf")
        near = ABBR_NEAR.match(t.strip(",;:()")) or (row["lang"] in M.SLAV and ABBR_NEAR_SLAV.match(t.strip(",;:()")))
        if not flags[i] and near and (
                (i > 0 and flags[i - 1]) or (i + 1 < len(toks) and flags[i + 1])):
            return {"reason": "abbr_left"}
    c = cer(label, r["greedy"])
    ce = cer(label, ext) if ext else None
    if (c if ce is None else min(c, ce)) > CER_BY_SRC.get(row.get("src"), MAX_CER):
        return {"reason": "cer", "cer": c} | ({"cer_ext": ce} if ce is not None else {})
    gap = word_gap(label, r["greedy"])
    if ext:
        gap = min(gap, word_gap(label, ext))
    if gap >= 2:  # review H1: the caption left out (or added) 2+ consecutive words the audio has (or lacks)
        return {"reason": "caption_gap", "gap": gap}
    spans = [{"text": d["text"], "kind": d["kind"], "reading": d["cands"][d["pick"]], "canonical": d["cands"][0],
              "margin": d["margin"], "n_cands": len(d["cands"]), "by": d.get("by", "ctc")}
             | ({"undecided": True} if d.get("salvage") else {}) for d in ds]
    nr = {"audio_filepath": row["audio_filepath"], "duration": row["duration"], "text": label, "lang": row["lang"],
          "src": row["src"], "id": row["id"], "origin": row["origin"], "caption": cap, "cer_ctc": round(c, 3),
          "spans": spans, "num_chars": offs}
    if ce is not None:
        nr["cer_ext"] = round(ce, 3)
    if x is not None:
        nr["edge_quiet"] = edge_quiet(x)
    return {"row": nr}


# ---------------------------------------------------------------- external-transcript row edges (2026-10-16)
# Parliament pseudo-labels (Granary over VoxPopuli segments) often cover only part of the segment's speech: on the
# Italian probe 70 of 87 "cer" drops and 33 of 47 "caption_gap" drops were a caption that matches the external
# transcript word for word in one stretch, with the segment's audio running on before and / or after it. ext_trim cuts
# the audio to the caption's stretch (at a quiet frame between two transcript words) so the label holds every word
# of its audio; the row guards (CER, caption gap, edges) then judge the cut row as usual.
TRIM_SLACK = 0.03  # s a cut may lie outside the gap between the two transcript words (their times are approximate)
STRADDLE = 0.04  # s: a transcript word whose sound runs this far past a cut on both sides is cut through
MIN_TRIM = 2.0  # s: the shortest trimmed row


def ext_words_in(words: list, a: float, b: float, dur: float | None = None) -> tuple[str, bool]:
    """(text of the transcript words whose midpoint lies in [a, b] s, whether a word is cut through at a or b). words =
    [{"w", "s", "e"}] with times in ms. A cut at the audio's own start / end (0 or dur) cuts nothing."""
    out, straddle = [], False
    for w in words:
        s, e = w["s"] / 1000, w["e"] / 1000
        if a <= (s + e) / 2 <= b:
            out.append(w["w"])
        for c in (a, b):
            if c <= 0 or (dur is not None and c >= dur):
                continue
            if s + STRADDLE < c < e - STRADDLE:
                straddle = True
    return " ".join(out), straddle


def ext_trim(caption: str, words: list, x) -> dict | None:
    """Where the caption sits in the row's audio by the transcript's word times: {"a", "b", "lead", "trail"} (cut
    seconds, transcript words cut off before / after) or None (nothing to cut, or no safe cut). A side is cut only when
    the caption's first (last) token is a word that opens (closes) a run of >= 2 words matching the transcript, the
    transcript has words beyond it, and a quiet frame lies between the two transcript words there."""
    import difflib

    dur = len(x) / 16000
    toks = caption.split()
    A = [(i, letters(t)) for i, t in enumerate(toks) if letters(t) and not re.search(r"\d", t)]
    B = [letters(w["w"]) for w in words]
    if len(A) < 4 or not B:
        return None
    blocks = [m for m in difflib.SequenceMatcher(None, [a for _, a in A], B, autojunk=False).get_matching_blocks() if m.size]
    if not blocks:
        return None
    first, last = blocks[0], blocks[-1]
    a, b, lead, trail = 0.0, dur, 0, 0
    if A[0][0] == 0 and first.a == 0 and first.size >= 2 and first.b > 0:
        t0, t1 = words[first.b - 1]["e"] / 1000, words[first.b]["s"] / 1000
        c = quiet_cut(x, min(t0, t1) - TRIM_SLACK, max(t0, t1) + TRIM_SLACK)
        if c is not None:
            a, lead = c, first.b
    j1 = last.b + last.size - 1
    if A[-1][0] == len(toks) - 1 and last.a + last.size == len(A) and last.size >= 2 and j1 < len(B) - 1:
        t0, t1 = words[j1]["e"] / 1000, words[j1 + 1]["s"] / 1000
        c = quiet_cut(x, min(t0, t1) - TRIM_SLACK, max(t0, t1) + TRIM_SLACK)
        if c is not None:
            b, trail = c, len(B) - 1 - j1
    if not (lead or trail) or b - a < MIN_TRIM:
        return None
    if ext_words_in(words, a, b, dur)[1]:
        return None
    return {"a": round(a, 3), "b": round(b, 3), "lead": lead, "trail": trail}


def shift_words(words: list, a: float, b: float) -> list:
    """Transcript words inside [a, b] s with times relative to a (ms)."""
    return [dict(w, s=w["s"] - a * 1000, e=w["e"] - a * 1000) for w in words if a <= (w["s"] + w["e"]) / 2000 <= b]


def piece_ok(text: str, greedy: str, max_cer: float, words: list | None, a: float, b: float, dur: float) -> tuple:
    """(ok, cer_ctc, cer_ext) for a piece [a, b] of a row: its label matches its own CTC decode (CER <= max_cer and
    the same first / last word, edges_match) or, with transcript words (piece_ext), the transcript words inside the
    piece when no transcript word is cut through at the piece's cuts."""
    c = cer(text, greedy)
    if c <= max_cer and edges_match(text, greedy):
        return True, c, None
    if not words:
        return False, c, None
    et, straddle = ext_words_in(words, a, b, dur)
    ce = cer(text, et) if et else 1.0
    return (not straddle and ce <= max_cer and edges_match(text, et)), c, ce


MAX_ROW = 20.0  # s: the longest row the mix takes; a longer row is split at pauses (plan_splits) instead of dropped


def plan_splits(n_tok: int, flags: list[bool], times: list, x, max_s: float = MAX_ROW, min_s: float = MIN_CROP):
    """Cut a long labelled row into pieces of <= max_s at pauses. A cut lies between two words whose CTC letter frames
    are >= MIN_GAP apart, at the quietest 10 ms frame of that gap, which must be quiet (quiet_cut, the crop rule), and
    never between two words of a number reading. Each piece ends at the latest cut that keeps it <= max_s (and
    >= min_s). Returns [{"a", "b", "i0", "i1"}] (seconds, token range) covering every token, or None when some stretch
    has no usable cut (the row stays dropped). A row already <= max_s is one piece."""
    dur = len(x) / 16000
    if dur <= max_s:
        return [{"a": 0.0, "b": round(dur, 3), "i0": 0, "i1": n_tok}]
    cuts = []
    for k in range(1, n_tok):
        if times[k - 1] is None or times[k] is None or times[k][0] - times[k - 1][1] < MIN_GAP:
            continue
        if flags[k - 1] and flags[k]:
            continue
        c = quiet_cut(x, times[k - 1][1], times[k][0])
        if c is not None:
            cuts.append((k, c))
    out, a, i0 = [], 0.0, 0
    while dur - a > max_s:
        ok = [(k, c) for k, c in cuts if k > i0 and min_s <= c - a <= max_s]
        if not ok:
            return None
        k, c = ok[-1]
        out.append({"a": round(a, 3), "b": round(c, 3), "i0": i0, "i1": k})
        a, i0 = c, k
    out.append({"a": round(a, 3), "b": round(dur, 3), "i0": i0, "i1": n_tok})
    return out


PARTIAL_MIN_S = 1.5  # s: the shortest partial piece (ml_multi_scale.MIN_PIECE_S)
PARTIAL_MIN_WORDS = 3


def undecided_tokens(label: str, offsets, spans) -> set:
    """Token indices of the label that belong to a span make_row marked undecided (salvage)."""
    toks, pos = [], 0
    for t in label.split():
        a = label.index(t, pos)
        toks.append((a, a + len(t)))
        pos = a + len(t)
    bad = set()
    for (x, y), sp in zip(offsets or [], spans or []):
        if sp.get("undecided"):
            bad |= {i for i, (a, b) in enumerate(toks) if a < y and b > x}
    return bad


def plan_partial(n_tok: int, flags: list[bool], bad: set, times: list, x, max_s: float = MAX_ROW,
                 min_s: float = PARTIAL_MIN_S, min_words: int = PARTIAL_MIN_WORDS) -> list[dict]:
    """Partial rows (2026-10-16): pieces of a labelled row that leave out its undecided spans (`bad` token indices).
    Each maximal stretch of tokens without a bad token is cut free at the first usable cut at or after its start and
    the last one at or before its end (a cut as in plan_splits: >= MIN_GAP between the two words' CTC letters, a quiet
    frame there, never inside a number reading; the row's own start / end need no cut), then split at pauses into
    pieces of <= max_s. A stretch, or a part of it, with no usable cut is left out; the rest of the row is not.
    Returns [{"a", "b", "i0", "i1"}] (seconds, token range) of the pieces with >= min_s and >= min_words; the caller
    keeps those that hold a number and pass their own checks."""
    dur = len(x) / 16000
    cut_at = {}
    for k in range(1, n_tok):
        if times[k - 1] is None or times[k] is None or times[k][0] - times[k - 1][1] < MIN_GAP:
            continue
        if flags[k - 1] and flags[k]:
            continue
        c = quiet_cut(x, times[k - 1][1], times[k][0])
        if c is not None:
            cut_at[k] = c
    cut_at[0], cut_at[n_tok] = 0.0, dur
    stretches, cur = [], None
    for i in range(n_tok):
        if i in bad:
            if cur is not None:
                stretches.append(cur)
            cur = None
        else:
            cur = [i, i + 1] if cur is None else [cur[0], i + 1]
    if cur is not None:
        stretches.append(cur)
    out = []
    for r0, r1 in stretches:
        ks = sorted(k for k in cut_at if r0 <= k <= r1)
        if len(ks) < 2:
            continue
        i0, i1 = ks[0], ks[-1]
        a, end = cut_at[i0], cut_at[i1]
        inner = [k for k in ks if i0 < k < i1]
        while i0 < i1:
            if end - a <= max_s:
                k = i1
            else:
                ok = [k for k in inner if k > i0 and min_s <= cut_at[k] - a <= max_s]
                if not ok:  # no pause short enough: move on to the next cut and try again from there
                    nxt = [k for k in inner if k > i0]
                    if not nxt:
                        break
                    i0, a = nxt[0], cut_at[nxt[0]]
                    continue
                k = ok[-1]
            b = cut_at[k]
            if b - a >= min_s and k - i0 >= min_words:
                out.append({"a": round(a, 3), "b": round(b, 3), "i0": i0, "i1": k})
            i0, a = k, b
    return out


def split_rows(nr: dict, pieces: list[dict]) -> list[dict]:
    """Rows for the pieces of a labelled row (make_row output): the label's exact characters from the first to the last
    token of each piece, its number offsets and spans shifted into the piece. The caller cuts the audio, re-checks each
    piece against its own decode and adds word times."""
    label = nr["text"]
    starts, pos = [], 0
    for tok in label.split():
        a = label.index(tok, pos)
        starts.append((a, a + len(tok)))
        pos = a + len(tok)
    # label position -> caption position (outside readings the label is the caption): each piece gets its own caption
    # substring, so a later check that re-finds the spans in the caption (the positional judge) sees the piece's spans
    cap_sp = M.spans(nr["caption"], nr["lang"]) if nr.get("caption") else []
    shifts = [(y, (y - x) - (s.end - s.start)) for (x, y), s in zip(nr.get("num_chars") or [], cap_sp)]

    def cap_pos(p):
        return p - sum(d for y, d in shifts if y <= p)

    out = []
    for j, p in enumerate(pieces):
        lo, hi = starts[p["i0"]][0], starts[p["i1"] - 1][1]
        offs, sps = [], []
        for (x, y), s in zip(nr.get("num_chars") or [], nr.get("spans") or []):
            if lo <= x and y <= hi:
                offs.append([x - lo, y - lo])
                sps.append(s)
        extra = {"caption": nr["caption"][cap_pos(lo):cap_pos(hi)]} if len(cap_sp) == len(nr.get("num_chars") or []) and cap_sp else {}
        out.append({k: v for k, v in nr.items() if k not in ("text", "num_chars", "spans", "duration", "id", "word_times")}
                   | {"text": label[lo:hi], "num_chars": offs, "spans": sps, "id": f"{nr['id']}_p{j}", "parent": nr["id"],
                      "piece": [p["a"], p["b"]], "duration": round(p["b"] - p["a"], 3)} | extra)
    return out


class Acc:
    def __init__(self, lang):
        self.lang = lang
        self.rows_in = Counter()
        self.hours_in = Counter()
        self.rows_out = Counter()
        self.hours_out = Counter()
        self.reasons = Counter()
        self.span_tot = Counter()  # kind -> spans seen
        self.span_cand = Counter()  # kind -> spans with candidates
        self.span_dec = Counter()  # kind -> decided
        self.noncanon = Counter()
        self.noncanon_diffwords = []
        self.edge = Counter()
        self.examples = []

    def add_skip(self, row, why):
        self.rows_in[row["origin"]] += 1
        self.hours_in[row["origin"]] += row["duration"] / 3600
        self.reasons[f"{row['origin']}:{why}"] += 1

    def add(self, row, r, res):
        o = row["origin"]
        self.rows_in[o] += 1
        self.hours_in[o] += row["duration"] / 3600
        for d in r["spans"]:
            self.span_tot[(o, d["kind"])] += 1
            if "pick" in d:
                self.span_cand[(o, d["kind"])] += 1
                if d["decided"]:
                    self.span_dec[(o, d["kind"])] += 1
                    if d["pick"] != 0:
                        self.noncanon[(o, d["kind"])] += 1
        if "row" in res:
            nr = res["row"]
            self.rows_out[o] += 1
            self.hours_out[o] += nr["duration"] / 3600
            self.edge[(o, nr.get("edge_quiet"))] += 1
            for s in nr["spans"]:
                if s["reading"] != s["canonical"]:
                    from ml_decide import ending_only

                    if not ending_only(s["reading"], s["canonical"]) and len(self.noncanon_diffwords) < 40:
                        self.noncanon_diffwords.append({"id": nr["id"], "caption_span": s["text"], "kind": s["kind"],
                                                        "reading": s["reading"], "canonical": s["canonical"]})
            self.examples.append(nr)
        else:
            self.reasons[f"{o}:{res['reason']}"] += 1

    def stats(self, kept):
        import random

        def agg(c):
            return {f"{o}/{k}": v for (o, k), v in sorted(c.items())}

        def tot(c, o):
            return sum(v for (oo, _), v in c.items() if oo == o)

        out = {"lang": self.lang, "max_cer": MAX_CER,
               "rows_in": dict(self.rows_in), "rows_out": dict(self.rows_out),
               "hours_in": {k: round(v, 3) for k, v in self.hours_in.items()},
               "hours_out": {k: round(v, 3) for k, v in self.hours_out.items()},
               "drop_reasons": dict(sorted(self.reasons.items())),
               "span_total": agg(self.span_tot), "span_with_candidates": agg(self.span_cand),
               "span_decided": agg(self.span_dec), "decided_non_canonical": agg(self.noncanon),
               "edge_quiet": {f"{o}/{q}": v for (o, q), v in self.edge.items()}}
        for o in ("d2", "new"):
            c = tot(self.span_cand, o)
            out[f"decide_rate_{o}"] = round(tot(self.span_dec, o) / c, 3) if c else None
        c, dd = sum(self.span_cand.values()), sum(self.span_dec.values())
        out["decide_rate_all"] = round(dd / c, 3) if c else None
        out["hours_out_total"] = round(sum(self.hours_out.values()), 3)
        out["kinds_kept"] = dict(Counter(s["kind"] for nr in kept for s in nr["spans"]))
        out["noncanonical_different_words"] = self.noncanon_diffwords
        rnd = random.Random(7)
        by_kind = defaultdict(list)
        for nr in self.examples:
            for s in nr["spans"]:
                by_kind[s["kind"]].append((s["text"], s["reading"], nr["id"]))
        ex = []
        kinds = sorted(by_kind)
        while len(ex) < 10 and kinds:
            for k in list(kinds):
                if not by_kind[k]:
                    kinds.remove(k)
                    continue
                ex.append(f"[{k}] {self.examples[0]['lang']}  " + " -> ".join(by_kind[k].pop(rnd.randrange(len(by_kind[k])))[:2]))
                if len(ex) >= 10:
                    break
        out["examples"] = ex
        # sample whole rows for eyeballing
        out["row_samples"] = [{"caption": nr["caption"], "label": nr["text"]} for nr in rnd.sample(self.examples, min(15, len(self.examples)))]
        return out


# ---------------------------------------------------------------- streaming-geometry training windows (2026-10-15)
# A streaming decoder that uses 14.96 s windows (187 encoder frames of 80 ms) every 9.44 s (118 frames; 5.52 s overlap) and may move a
# window start up to 2.0 s to a pause. Windows
# cut from stitched streams of labelled rows put every number in the centre, at an edge and across an edge. Label = the
# words the window can be heard to hold: a word wholly inside, or one the edge cuts with >= KEEP_FRAC of its sound inside;
# a word with <= DROP_FRAC inside is dropped (the overlapping window holds it whole). A window with a cut word in between
# is not built. Edge check 2026-10-15 (B and stock on the old drop-every-cut-word windows): at cut edges the models wrote
# the whole cut word on 16% of real-speech edges, so dropping it taught deleting audible words.
WIN, STRIDE, JITTER, PAUSE = 187 * 0.08, 118 * 0.08, 2.0, 0.3
LEAD, TAIL = 0.05, 0.08  # s of sound before a word's first and after its last CTC letter frame
KEEP_FRAC, DROP_FRAC = 2 / 3, 1 / 3
# compound number words (one token = a whole number: "dreihundertfünfundzwanzig", "vijfentwintig"): a piece of the word
# is itself a number word ("zwanzig", "twintig"), so the 2/3 rule labels a window that hears only part of the value.
# In these languages a NUMBER word cut by an edge is labelled whole only with >= NUM_KEEP_FRAC of its sound inside and
# left out only with <= 1 - NUM_KEEP_FRAC; anything in between makes the window (or the seam cut) unusable.
NUM_KEEP_FRAC = 0.9
# 2026-10-17: EVERY language (not only de / nl): es "veintiuno", it "ventuno", fr "quatre-vingt", pt / Romance
# multi-syllable number words too -- the es pilot labelled 1,231 of 1,673 in-word seam pairs with a word only partly
# heard; the current es / fr / it / pt seam parts had 46-54% of their halves cut that way, windows 7-10%
COMPOUND_NUM_LANGS = None  # None = all languages


def num_keep_frac(lang: str) -> float | None:
    """The number-word edge rule of a language (None: the plain KEEP_FRAC / DROP_FRAC rule)."""
    return NUM_KEEP_FRAC if COMPOUND_NUM_LANGS is None or lang in COMPOUND_NUM_LANGS else None


def num_edge_ambiguous(w: tuple, fr: float, num_keep: float | None) -> bool:
    """A cut number word whose share inside is neither ~whole nor ~none under the compound rule."""
    return bool(num_keep) and w[3] and (1 - num_keep) < fr < num_keep


def compound_splits(words: list, a: float, b: float, num_keep: float = NUM_KEEP_FRAC, total: float | None = None) -> int:
    """Number words of [a, b] cut with a share inside strictly between 1 - num_keep and num_keep (census helper)."""
    n = 0
    for w in words:
        if w[1] is None or not w[3]:
            continue
        fr = inside_frac(w, a, b, 0.0, total)
        if 0 < fr < 1 and (1 - num_keep) < fr < num_keep:
            n += 1
    return n


def inside_frac(w: tuple, a: float, b: float, t0: float | None = None, t1: float | None = None) -> float:
    """Share of word w's sound span [t0 - LEAD, t1 + TAIL] that lies inside the window [a, b]. t0 / t1: the stream's
    bounds; the span is clipped to them (a stream's first word has no audio before 0 to lose: unclipped, LEAD put 89%
    of a short first word "inside" a window starting at 0, and the 0.9 compound rule refused a whole word)."""
    lo, hi = w[1] - LEAD, w[2] + TAIL
    if t0 is not None:
        lo = max(lo, t0)
    if t1 is not None:
        hi = min(hi, t1)
    return max(0.0, min(b, hi) - max(a, lo)) / (hi - lo) if hi > lo else 0.0


def stream_words(rows: list[dict], pause: float = PAUSE) -> tuple[list[dict], list[tuple], float]:
    """Lay rows end to end (pause s apart). Returns (segments [{row, start}], words [(tok, t0, t1, is_num, row_i, k)],
    total s). A token without a time gets t0 = t1 = None."""
    segs, words, t = [], [], 0.0
    for i, r in enumerate(rows):
        segs.append({"row": i, "start": t})
        flags = number_tokens(r["text"], r.get("num_chars") or [])
        for k, (tok, tm, fl) in enumerate(zip(r["text"].split(), r["word_times"], flags)):
            words.append((tok, None if tm is None else tm[0] + t, None if tm is None else tm[1] + t, fl, i, k))
        t += r["duration"] + pause
    return segs, words, t - pause


def build_windows(rows: list[dict], rng, win: float = WIN, stride: float = STRIDE, jitter: float = JITTER,
                  min_words: int = 3, num_keep: float | None = None) -> list[dict]:
    """Streaming-geometry windows over one stream of labelled rows (each with text, duration, word_times, num_chars). Window k
    starts at phase + k * stride + U(0, jitter); the last one may be shorter (a stream's final chunk). A window is kept when
    every token near it has a time, it holds >= min_words whole words and >= 1 number word.
    num_keep (num_keep_frac(lang)): the compound number-word edge rule; a number word cut with a share inside between
    1 - num_keep and num_keep skips the window."""
    segs, words, total = stream_words(rows)
    out = []
    s = rng.uniform(0, stride) - stride  # the first window may start before the stream (it then starts at 0)
    while s < total - 1.0:
        a = round(max(0.0, s + rng.uniform(0, jitter)), 3)
        b = round(min(total, a + win), 3)
        near = [w for w in words if w[1] is None or (w[1] < b and w[2] > a) or (w[2] is not None and a <= w[2] <= b)]
        if any(w[1] is None for w in near):
            s += stride
            continue
        whole = [w for w in near if max(0.0, w[1] - LEAD) >= a and min(total, w[2] + TAIL) <= b]
        part = [w for w in near if w not in whole]
        fr = {id(w): inside_frac(w, a, b, 0.0, total) for w in part}
        if any(DROP_FRAC < fr[id(w)] < KEEP_FRAC or num_edge_ambiguous(w, fr[id(w)], num_keep) for w in part):
            s += stride  # half a word in the window: neither label is what the audio holds
            continue
        kept_cut = [w for w in part if fr[id(w)] >= KEEP_FRAC]
        cut = [w for w in part if fr[id(w)] <= DROP_FRAC]
        inside = sorted(whole + kept_cut, key=lambda w: w[1])
        cut_left = [w[0] for w in cut if w[1] < a + 1.0]
        cut_right = [w[0] for w in cut if w[2] > b - 1.0]
        if len(inside) >= min_words and any(w[3] for w in inside):
            out.append({"a": round(a, 3), "b": round(b, 3), "text": " ".join(w[0] for w in inside),
                        "num_tokens": [w[3] for w in inside], "cut_words": len(cut), "kept_cut_words": len(kept_cut),
                        "edge_fracs": [[w[0], round(fr[id(w)], 2), "kept" if w in kept_cut else "dropped"] for w in part],
                        "cut_left": cut_left[-1] if cut_left else None, "cut_right": cut_right[0] if cut_right else None,
                        "cut_number": any(w[3] for w in part),
                        "edge_number": any(w[3] and (w[1] - a < 2.0 or b - w[2] < 2.0) for w in inside),
                        "rows": sorted({w[4] for w in inside} | {w[4] for w in cut}),
                        "src": [[w[4], w[5]] for w in inside]})  # (stream row, word index) of every labelled token
        s += stride
    return out


def window_audio(rows: list[dict], a: float, b: float, read, pause: float = PAUSE):
    """The window's samples from the stitched stream (read(path) -> float32 16 kHz array)."""
    import numpy as np

    parts, t = [], 0.0
    rs = np.random.RandomState(int(a * 1000) % (2 ** 31))
    for r in rows:
        x = read(r["audio_filepath"])
        parts.append(x)
        # review M3: the pause between stitched rows carries the row's own noise floor, not digital silence
        n = len(x) // 160
        fr = np.sqrt(np.mean(x[:n * 160].reshape(n, 160) ** 2, axis=1) + 1e-12) if n else np.array([1e-4])
        level = float(np.percentile(fr, 10))
        parts.append((rs.randn(int(pause * 16000)) * level).astype(np.float32))
        t += r["duration"] + pause
    s = np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype=np.float32)
    return s[int(a * 16000):int(b * 16000)]


def stream_key(row: dict) -> tuple:
    """(source, order) for stitching: VoxPopuli ids "<session>_<time>_<n>", YODAS ids carrying an 11-char video id; any
    other row (D2 / FLEURS) is its own source and gets stitched with random neighbours of the same language."""
    rid = str(row.get("id", ""))
    if row.get("src") == "vox" and "_" in rid:
        sess, _, rest = rid.partition("_")
        return ("vox:" + sess, rest)
    if row.get("src") == "tts":  # one synthetic voice per stream (a window never switches voice mid-way)
        return ("tts:" + str(row.get("voice", "")).split("/")[0], rid)
    if row.get("src") == "yodas":
        m = re.search(r"_([A-Za-z0-9_-]{11})_(\d+)", rid)
        if m:
            return ("yt:" + m.group(1), m.group(2).zfill(9) + rid)
    return ("misc", rid)


def group_streams(rows: list[dict], rng, max_rows: int = 8, min_rows: int = 2) -> list[list[dict]]:
    """Consecutive rows of one session / video, in time order, in streams of <= max_rows; leftover single rows and
    unkeyed rows are stitched in random groups of max_rows (same language)."""
    by = {}
    for r in rows:
        k, o = stream_key(r)
        by.setdefault(k, []).append((o, r))
    out, loose = [], []
    for k, v in sorted(by.items()):
        v = [r for _, r in sorted(v, key=lambda t: t[0])]
        if k == "misc":
            loose += v
            continue
        for i in range(0, len(v), max_rows):
            chunk = v[i:i + max_rows]
            (out.append(chunk) if len(chunk) >= min_rows else loose.extend(chunk))
    rng.shuffle(loose)
    out += [loose[i:i + max_rows] for i in range(0, len(loose), max_rows) if len(loose[i:i + max_rows]) >= min_rows]
    return out


# ---------------------------------------------------------------- windows / seam crops beyond wave 1 (2026-10-17)
_VOX_DAY = re.compile(r"^(?:# group vox:)?(\d{8})(?:-\d{4}-|\s*$)")


def held_days(held_ids) -> set:
    """Parliament days of a held-out list: any session type ("20150921-1500-COMMITTEE-..." as well as PLENARY ids) and
    the list's own "# group vox:<day>" lines (the carve unit is the day)."""
    out = set()
    for i in held_ids:
        m = _VOX_DAY.match(str(i).strip())
        if m:
            out.add(m.group(1))
    return out


def vox_day(rid: str) -> str | None:
    """Parliament day of a row id of any session type, or None."""
    m = re.match(r"^(\d{8})-\d{4}-", str(rid))
    return m.group(1) if m else None


def has_times(row: dict) -> bool:
    wt = row.get("word_times")
    return bool(wt) and len(wt) == len(row["text"].split()) and all(t is not None for t in wt)


def split_solo(rows: list[dict], solo_srcs) -> tuple[list[dict], list[list[dict]]]:
    """(rows to stitch, one-row streams). Rows of a solo source (short read clips, each its own speaker) are never
    stitched with other rows: a seam pair is cut inside the one clip."""
    solo = set(solo_srcs or ())
    return [r for r in rows if r.get("src") not in solo], [[r] for r in rows if r.get("src") in solo]


def same_persona(voice: str | None, heldout_voice: str | None) -> bool:
    """mix_build.is_heldout_voice's rule (mix_build.heldout_persona): the held-out voice is its PERSONA in every locale
    ("nl-BE-...-Iapetus" is held out with "nl-NL-...-Iapetus")."""
    return bool(heldout_voice and voice and voice.rsplit("-", 1)[-1] == heldout_voice.rsplit("-", 1)[-1])


def synthetic_train_rows(sets: dict, reject: set, heldout_voice: str = "") -> tuple[list[dict], dict]:
    """Rows of synthetic sets ({tag: rows}) that may build TRAINING windows / seam pairs: split "train" only (a row
    without a split is train, as build4 reads it), never the language's held-out voice persona in any locale or split
    (same_persona), never a judge or grammar reject. Returns (rows tagged origin "tts", {reason: rows dropped})."""
    out, drop = [], {"dev_split": 0, "heldout_voice": 0, "rejected": 0}
    for rows in sets.values():
        for r in rows:
            if r.get("split", "train") != "train":
                drop["dev_split"] += 1
            elif same_persona(r.get("voice"), heldout_voice):
                drop["heldout_voice"] += 1
            elif r["id"] in reject:
                drop["rejected"] += 1
            else:
                out.append(dict(r, origin="tts", src="tts"))
    return out, drop


def syn_set_markers(patterns: str, tts_sets) -> list[str]:
    """["tag:judge_marker"] of the build4 synthetic sets (tts_sets: mix_build.TTS_SETS) matching the comma-separated
    fnmatch patterns; the held-out voice's decision sets (ttsHO*) never match."""
    import fnmatch

    pats = [p for p in patterns.split(",") if p]
    return [f"{t}:{m}" for t, m in tts_sets if not t.startswith("ttsHO") and any(fnmatch.fnmatch(t, p) for p in pats)]


def mix_build_consts(path: str | None = None) -> tuple[tuple, dict]:
    """(mix_build.TTS_SETS, mix_build.HELDOUT_VOICES) read from mix_build.py's source with ast.literal_eval, for a local
    entrypoint whose Python has no numpy (mix_build imports it)."""
    import ast
    import os

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_build.py")
    found = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("TTS_SETS", "HELDOUT_VOICES"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return tuple(found["TTS_SETS"]), dict(found["HELDOUT_VOICES"])
