"""boundary_cut: ONE shared edge-placement rule for clips cut out of longer audio with word times. Pure stdlib (edge_rule needs numpy only when a `track` is used), no audio read.

Every clip edge is cut at a WORD GAP; a word the nominal edge would clip leaves BOTH the audio and the label (audio ends after the last complete word /
starts before the first complete word). Never drop a fragment from the label while it stays audible; never label a partly heard word whole.
A number split at a gap BETWEEN its words keeps each side's words ('five hundred' | 'sixty'); a number cut MID-WORD excludes the clip.
With no word gap in range (continuous speech) the clip is excluded, unless a `track` (edge_rule.Track: frame energy + VAD chunk probabilities) is given: then the
quietest valley in the search range is used when it is a real dip (>= min_dip dB below the local speech level, edge_rule.snap_with_dip), else the clip is
excluded as `no_dip`; words that dip cut straddles leave the label (a straddled NUMBER word excludes the clip). With a `track`, the position inside a word gap
is chosen by the same valley rule (widest VAD valley, lowest-energy frame) instead of the gap middle / energy minimum.

Inputs are word times in seconds on the clip's / stream's own clock: Word(t0, t1, key=None, join=False)
  key   = reading id for words of a number reading (None for other words); two adjacent words with the same key are one number;
  join  = True when a stitched join (row boundary: synthetic pause) lies between the previous word and this one.
A cut position is a time x; words[:j] lie before it, words[j:] after it. The window length is the caller's: cut_window(words, total, a_nom, b_nom, search=...)
takes the nominal window [a_nom, b_nom] and moves each edge inward by at most `search` seconds (stride = how the caller steps a_nom / b_nom).

Rule (every constant below is a parameter or a module default; fixtures = test_boundary_cut.py):
  eligible gap  : raw width (t0 of next - t1 of previous) >= max(MIN_GAP, 2 * MIN_PAD); cut interval [t1_prev + MIN_PAD, t0_next - MIN_PAD]
                  (stream start / end: no pad on the stream side).
  position      : the quietest energy frame whose centre is inside the cut interval (ties: nearest the interval middle) when an Energy track is
                  given, else the middle of the interval. keep_valid=True keeps an existing nominal position that already lies in an interval
                  (re-cuts of rows whose edge was already chosen on the audio). hug=True puts the cut at the pad boundary of the KEPT word instead
                  (silence = MIN_PAD): a flush ('forced') edge at will.
  search        : start edge moves FORWARD (inward), end edge BACKWARD (inward) at most `search` s from the nominal edge; the nearest eligible gap
                  wins (least audio discarded). A gap that contains the nominal edge costs 0.
  mid-word      : the nominal edge sits within MIN_PAD of a word (inside it, or too close to cut) = that word is CLIPPED. Clipped non-number
                  words are dropped from audio + label by moving the edge to the next gap; a clipped NUMBER word excludes the clip.
  kind          : 'forced' = the cut leaves <= FORCED_SILENCE (0.08 s) of silence between the cut and the kept word (the clip starts / ends flush
                  with speech, as a decoder window edge does), 'valley' = more silence, 'dip' = no gap, cut at a real dip, 'bound' = the
                  stream's own start / end (nothing to cut).
  intra         : prefer gaps INSIDE a parent (not a stitched join) so forced edges fall mid-sentence; a join is taken only when
                  no inside-parent gap is in range (edge.intra_unmet). Stitched streams pause at every join, so without this a nearest-gap search
                  lands on joins and the forced class vanishes.
  exclusion     : reasons 'number_mid_word', 'no_gap_in_range', 'no_dip', 'rejected' (caller ok() said no), 'too_short', 'too_few_words'.
"""
from __future__ import annotations

import dataclasses, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_rule as ER
from typing import Callable, Optional, Sequence

MIN_GAP = 0.10  # s raw gap between two words' letter frames
MIN_PAD = 0.06  # s between a cut and the nearest word-time edge (0.08 is also reasonable; sweep it on your alignments)
FORCED_SILENCE = 0.08  # s: a cut with <= this much silence on the kept side is 'forced'
EPS = 1e-6
INF = float("inf")
ACTIVE_TRACK = None   # dry runs / cutters that cannot pass `track` through their own signatures set this per clip; place_edge falls back to it
SPEC = {"min_gap": MIN_GAP, "min_pad": MIN_PAD, "forced_silence": FORCED_SILENCE, "search_s": 2.0}


@dataclasses.dataclass(frozen=True)
class Word:
    t0: float
    t1: float
    key: object = None
    join: bool = False


@dataclasses.dataclass(frozen=True)
class Energy:
    """Per-frame energy (lower = quieter); frame i covers [t0 + i * frame_s, t0 + (i + 1) * frame_s)."""
    values: Sequence[float]
    frame_s: float = 0.01
    t0: float = 0.0


@dataclasses.dataclass(frozen=True)
class Gap:
    j: int  # the gap before word j (0 = stream start, n = stream end)
    lo: float  # cut interval (pad included)
    hi: float
    width: float  # raw gap width
    join: bool
    key: object  # reading id when both neighbours belong to one number reading (a straddle cut), else None
    t_prev: float  # end of the word before the gap (-inf at the stream start)
    t_next: float  # start of the word after the gap (+inf at the stream end)


@dataclasses.dataclass
class Edge:
    ok: bool
    t: float = 0.0
    j: int = 0  # words[:j] before the cut
    kind: str = ""  # bound / forced / valley
    shift: float = 0.0  # s the edge moved from the nominal position (inward positive)
    width: float = 0.0  # raw width of the gap the cut sits in
    straddle: object = None  # reading key split by this cut
    reason: str = ""
    join: bool = False  # the gap is a stitched join (synthetic pause)
    intra_unmet: bool = False  # intra preferred but only a join was in range
    silence: float = 0.0  # s between the cut and the kept word
    clipped: tuple = ()  # indices of words the nominal edge clipped
    dip_db: float = 0.0  # dB below the local speech level at the cut (track given)


@dataclasses.dataclass
class Cut:
    ok: bool
    reason: str = ""
    a: float = 0.0
    b: float = 0.0
    j0: int = 0  # label = words[j0:j1]
    j1: int = 0
    start: Optional[Edge] = None
    end: Optional[Edge] = None
    removed_left: int = 0  # words of the nominal window that left (clipped word + words passed over)
    removed_right: int = 0

    @property
    def straddles(self) -> list:
        return [e.straddle for e in (self.start, self.end) if e is not None and e.straddle is not None]


# ---------------------------------------------------------------------------------------------------------------- gaps / positions
def gaps(words: Sequence[Word], total: float, min_gap: float = MIN_GAP, min_pad: float = MIN_PAD) -> list:
    """Eligible gaps in time order (see module docstring)."""
    n, out = len(words), []
    if n == 0:
        return [Gap(0, 0.0, total, total, False, None, -INF, INF)]
    if words[0].t0 - min_pad >= -EPS:
        out.append(Gap(0, 0.0, max(0.0, words[0].t0 - min_pad), words[0].t0, False, None, -INF, words[0].t0))
    need = max(min_gap, 2 * min_pad) - EPS
    for j in range(1, n):
        p, w = words[j - 1], words[j]
        width = w.t0 - p.t1
        if width >= need:
            same = p.key is not None and p.key == w.key
            out.append(Gap(j, p.t1 + min_pad, w.t0 - min_pad, width, bool(w.join), p.key if same else None, p.t1, w.t0))
    if total - words[-1].t1 - min_pad >= -EPS:
        out.append(Gap(n, min(total, words[-1].t1 + min_pad), total, total - words[-1].t1, False, None, words[-1].t1, INF))
    return out


def quietest(lo: float, hi: float, energy: Optional[Energy] = None) -> float:
    """Cut position in [lo, hi]: the quietest frame centre (ties: nearest the middle), else the middle."""
    mid = (lo + hi) / 2
    if energy is None or hi - lo < EPS:
        return mid
    best, bt = None, mid
    i = max(0, int((lo - energy.t0) / energy.frame_s) - 1)
    while i < len(energy.values):
        c = energy.t0 + (i + 0.5) * energy.frame_s
        if c > hi + EPS:
            break
        if c >= lo - EPS:
            key = (energy.values[i], abs(c - mid))
            if best is None or key < best:
                best, bt = key, c
        i += 1
    return bt if best is not None else mid


def clipped_words(words: Sequence[Word], x: float, min_pad: float = MIN_PAD) -> tuple:
    """Words a cut at x would clip: x inside the word, or closer than min_pad to it."""
    return tuple(i for i, w in enumerate(words) if w.t0 - min_pad + EPS < x < w.t1 + min_pad - EPS)


def _position(g: Gap, side: str, energy: Optional[Energy], hug: bool, track=None) -> float:
    if hug and side == "start" and g.t_next < INF:
        return g.hi
    if hug and side == "end" and g.t_prev > -INF:
        return g.lo
    if track is not None and g.hi - g.lo >= 0:
        # valley selection restricted to the gap interval; a frame is a candidate when its centre lies inside [lo, hi]
        p = ER.snap(track, g.lo - 0.5 * track.frame / track.sr, g.hi - 0.5 * track.frame / track.sr)
        if p is not None and g.lo - EPS <= p["t_center"] <= g.hi + EPS:
            return p["t_center"]
    return quietest(g.lo, g.hi, energy)


def _edge_from_gap(g: Gap, x: float, shift: float, side: str, unmet: bool = False) -> Edge:
    """side: which word the edge keeps ('start' keeps the word after the cut, 'end' the word before, 'mid' = both, a seam cut)."""
    after, before = g.t_next - x, x - g.t_prev
    sil = after if side == "start" else before if side == "end" else min(after, before)
    kind = "forced" if sil <= FORCED_SILENCE + EPS else "valley"
    return Edge(True, round(x, 4), g.j, kind, round(shift, 4), round(g.width, 4), g.key, "", g.join, unmet, round(sil, 4) if sil < INF else INF)


def edge_at_gap(words: Sequence[Word], total: float, j: int, energy: Optional[Energy] = None, min_gap: float = MIN_GAP,
                min_pad: float = MIN_PAD, side: str = "mid", hug: bool = False, track=None) -> Edge:
    """The cut in the gap before word j (explicit gap, e.g. a seam position inside a reading). side 'start' / 'end' = which clip keeps the
    word next to the cut (hug puts the cut flush with it); 'mid' = the shared middle / quietest position."""
    for g in gaps(words, total, min_gap, min_pad):
        if g.j == j:
            return _edge_from_gap(g, _position(g, side, energy, hug, track), 0.0, side)
    return Edge(False, reason="no_gap_in_range")


# ---------------------------------------------------------------------------------------------------------------- one edge
def place_edge(words: Sequence[Word], total: float, target: float, side: str, *, search: float = 2.0, energy: Optional[Energy] = None,
               min_gap: float = MIN_GAP, min_pad: float = MIN_PAD, hug: bool = False, intra: bool = False, keep_valid: bool = False,
               at_bound: bool = True, direction: Optional[str] = None, ok: Optional[Callable[[Edge], bool]] = None, track=None,
               min_dip: float = ER.DIP_DB) -> Edge:
    """Place ONE edge near `target`. side: 'start' (moves forward) / 'end' (moves backward); direction='any' lets a cut move either way.
    at_bound: a target at the stream start / end stays (nothing to cut)."""
    n = len(words)
    if track is None:
        track = ACTIVE_TRACK
    direction = direction or ("fwd" if side == "start" else "back")
    if at_bound and ((side == "start" and target <= EPS) or (side == "end" and target >= total - EPS)):
        return Edge(True, target, 0 if side == "start" else n, "bound", 0.0, 0.0, silence=(words[0].t0 if side == "start" else total - words[-1].t1) if n else 0.0)
    allg = gaps(words, total, min_gap, min_pad)
    contain = next((g for g in allg if g.lo - EPS <= target <= g.hi + EPS), None)
    if contain is not None and keep_valid:
        return _edge_from_gap(contain, target, 0.0, side)
    clipped = clipped_words(words, target, min_pad) if contain is None else ()
    if any(words[i].key is not None for i in clipped):
        return Edge(False, reason="number_mid_word", clipped=clipped)
    cands = []
    for g in allg:
        x = _position(g, side, energy, hug, track)
        if direction == "fwd":
            shift = max(0.0, x - target) if g.hi + EPS >= target else None
        elif direction == "back":
            shift = max(0.0, target - x) if g.lo - EPS <= target else None
        else:
            shift = abs(x - target)
        if shift is None or shift > search + EPS:
            continue
        cands.append((shift, g, x))
    cands.sort(key=lambda c: (c[0], c[1].j))
    unmet = False
    if intra and cands:
        inside = [c for c in cands if not c[1].join]
        if inside:
            cands = inside
        else:
            unmet = True
    for shift, g, x in cands:
        e = _edge_from_gap(g, x, shift, side, unmet)
        e.clipped = clipped
        if ok is None or ok(e):
            return e
    if not cands and track is not None:
        return _dip_edge(words, total, target, side, direction, search, track, min_dip, clipped)
    return Edge(False, reason="rejected" if cands else "no_gap_in_range", clipped=clipped)


def _dip_edge(words, total, target, side, direction, search, track, min_dip, clipped):
    """No word gap in range (continuous speech): the valley pick over the search range if it is a real dip, else exclude."""
    if direction == "fwd":
        lo, hi = target, target + search
    elif direction == "back":
        lo, hi = target - search, target
    else:
        lo, hi = target - search, target + search
    p, d, okk = ER.snap_with_dip(track, max(0.0, lo), min(total, hi), min_dip)
    if p is None or not okk:
        return Edge(False, reason="no_dip", clipped=clipped, dip_db=0.0 if d is None else round(float(d), 2))
    x = p["t_center"]
    strad = [i for i, w in enumerate(words) if w.t0 < x < w.t1]
    if any(words[i].key is not None for i in strad):
        return Edge(False, reason="number_mid_word", clipped=tuple(strad), dip_db=round(float(d), 2))
    if side == "start":
        j = next((i for i, w in enumerate(words) if w.t0 >= x - EPS), len(words))
    else:
        j = sum(1 for w in words if w.t1 <= x + EPS)
    return Edge(True, round(x, 4), j, "dip", round(abs(x - target), 4), 0.0, None, "", False, False, 0.0, tuple(strad), round(float(d), 2))


# ---------------------------------------------------------------------------------------------------------------- a window
def cut_window(words: Sequence[Word], total: float, a_nom: float, b_nom: float, *, search: float = 2.0, search_end: Optional[float] = None,
               min_len: float = 1.0, min_words: int = 1, energy: Optional[Energy] = None, hug: tuple = (False, False), intra: tuple = (False, False),
               keep_valid: bool = False, min_gap: float = MIN_GAP, min_pad: float = MIN_PAD, at_bound: bool = True,
               ok: Optional[Callable[["Cut"], bool]] = None, track=None, min_dip: float = ER.DIP_DB) -> Cut:
    """Both edges of a nominal window [a_nom, b_nom] at word gaps (inward moves only); label = words[j0:j1], every one complete."""
    s = place_edge(words, total, a_nom, "start", search=search, energy=energy, min_gap=min_gap, min_pad=min_pad, hug=hug[0], intra=intra[0],
                   keep_valid=keep_valid, at_bound=at_bound, track=track, min_dip=min_dip)
    if not s.ok:
        return Cut(False, "start:" + s.reason, start=s)
    e = place_edge(words, total, b_nom, "end", search=search if search_end is None else search_end, energy=energy, min_gap=min_gap, min_pad=min_pad,
                   hug=hug[1], intra=intra[1], keep_valid=keep_valid, at_bound=at_bound, track=track, min_dip=min_dip)
    if not e.ok:
        return Cut(False, "end:" + e.reason, start=s, end=e)
    nominal_j0 = sum(1 for w in words if w.t1 <= a_nom + EPS)  # words wholly before the nominal start
    nominal_j1 = sum(1 for w in words if w.t0 < b_nom - EPS)  # words that begin before the nominal end
    c = Cut(True, "", s.t, e.t, s.j, e.j, s, e, max(0, s.j - nominal_j0), max(0, nominal_j1 - e.j))
    if c.b - c.a < min_len - EPS:
        c.ok, c.reason = False, "too_short"
    elif c.j1 - c.j0 < min_words:
        c.ok, c.reason = False, "too_few_words"
    elif ok is not None and not ok(c):
        c.ok, c.reason = False, "rejected"
    return c


def window_label(words: Sequence[Word], cut: Cut) -> list:
    """Indices of the labelled words (all complete by construction)."""
    return list(range(cut.j0, cut.j1))


def seam_pair(words: Sequence[Word], total: float, j: int, win: float, *, search: float = 2.0, energy: Optional[Energy] = None,
              min_half: float = 1.0, min_gap: float = MIN_GAP, min_pad: float = MIN_PAD, hug: tuple = (False, False),
              hug_far: tuple = (False, False), intra_far: bool = False) -> Optional[dict]:
    """Mirrored pair at the explicit cut in the gap before word j: left [c - win, c], right [c, c + win], far edges moved inward to gaps.
    hug = (left clip's end, right clip's start) flush with its kept word (the two clips then do not share one cut time; the words still partition).
    Returns {'c': Edge (middle), 'left': Cut, 'right': Cut} or None (cut gap not eligible, a half too short / empty / no far gap)."""
    c = edge_at_gap(words, total, j, energy, min_gap, min_pad)
    if not c.ok:
        return None
    ce = edge_at_gap(words, total, j, energy, min_gap, min_pad, "end", hug[0])  # the left clip's end keeps the word before
    cs = edge_at_gap(words, total, j, energy, min_gap, min_pad, "start", hug[1])  # the right clip's start keeps the word after
    ls = place_edge(words, total, max(0.0, c.t - win), "start", search=search, energy=energy, min_gap=min_gap, min_pad=min_pad, hug=hug_far[0], intra=intra_far)
    re_ = place_edge(words, total, min(total, c.t + win), "end", search=search, energy=energy, min_gap=min_gap, min_pad=min_pad, hug=hug_far[1], intra=intra_far)
    if not ls.ok or not re_.ok or ls.j >= c.j or re_.j <= c.j:
        return None
    if ce.t - ls.t < min_half - EPS or re_.t - cs.t < min_half - EPS:
        return None
    return {"c": c, "left": Cut(True, "", ls.t, ce.t, ls.j, c.j, ls, ce), "right": Cut(True, "", cs.t, re_.t, c.j, re_.j, cs, re_)}


# ---------------------------------------------------------------------------------------------------------------- audit of existing edges
def edge_status(words: Sequence[Word], total: float, x: float, min_gap: float = MIN_GAP, min_pad: float = MIN_PAD) -> dict:
    """Classify an existing edge at x: 'bound' / 'gap' / 'pad_short' (between words, closer than min_pad to one) / 'in_word' (+ whether a
    number word is clipped). Used to audit stored clips."""
    if x <= EPS or x >= total - EPS:
        return {"status": "bound", "clipped": ()}
    cl = clipped_words(words, x, min_pad)
    if not cl:
        return {"status": "gap", "clipped": ()}
    inside = [i for i in cl if words[i].t0 < x < words[i].t1]
    return {"status": "in_word" if inside else "pad_short", "clipped": cl, "number": any(words[i].key is not None for i in cl)}


def forced_share(edges: Sequence[Edge]) -> dict:
    """Counts of edge kinds and the forced share of the non-bound edges."""
    kinds = {k: sum(1 for e in edges if e.ok and e.kind == k) for k in ("bound", "valley", "forced", "dip")}
    cut = kinds["valley"] + kinds["forced"] + kinds["dip"]
    return dict(kinds, forced_share=round(kinds["forced"] / cut, 4) if cut else None)
