"""15 s multi-number windows (long-context top-up, 2026-10-18).

A spoken-form fine-tune trained on single-sentence clips writes digits again when OTHER number-bearing sentences sit in
the encoder's context (the long-context flip). A streaming decoder feeds ~15 s windows. These windows stitch verified
training clips of one speaker / voice into ~15 s of audio that hold >= 2 number-bearing sentences, with number words in
the first and the last third, and edges that cut sentences at varied offsets.

Rules:
  * sources: the rows a finished build trains on (<mix_dir>/p3_<lang> real, tts_<lang> synthetic), read back from
    their originals as labelled (ml/p3v2/<lang>/rows*.jsonl, ml/<set>/<lang>/rows.jsonl; the same label as the built
    one up to the compound-number form, letters_eq) with their clean audio. Rows decided by a second recognizer
    (census_common.ext_row) and rows without word times never enter. English: the en_win prep rows (real), held-out
    videos / days re-checked, word times by forced alignment with the English reader (as en_win). Every row is trimmed to EDGE_PAD s around its first and last word
    before stitching, so a sound at a row's edge that its label does not hold never lands inside a window;
  * one stream = consecutive rows of one session / video (real) or one voice (synthetic); rows are joined by
    0.15-0.6 s gaps that carry the row's own noise floor;
  * edges are cut only in a quiet gap between two words (>= MIN_GAP between their letter frames and a quiet 10 ms
    frame there, the crop rule), never inside a number reading, so every labelled word is whole and every heard word is
    labelled; the left edge sits near a random offset, the right edge is the last such gap <= WIN s later;
  * a window is kept when it holds >= MIN_NUM_ROWS rows with a number reading, a number word in its first and in its
    last third, and passes the windows2 re-decode check (CER <= WIN_MAX_CER; the re-decode starts / ends with the
    label's first / last word, edges_ok);
  * labels are the parents' words verbatim (no re-fusing, no splitting here): the label FORM comes from the source
    rows and the build's compound setting (mix_build.build4 split_langs splits at write time, like every other
    part). `num_tokens` keeps each word's number flag (the R3 nonen_span / suffix-anchor losses read it), `src` /
    `parents` / `parent_sha` tie every word to its source row.

Outputs (new paths only; an existing output is never overwritten): /vol/ml/win15<tag>/<lang>/windows.jsonl, wav/,
stats.json; check_<n>.json (checker). The build reads them with build4 win15_share (mix4/win15_<lang>.jsonl).

    modal run win15.py::main --langs de,es --tag a --share 0.12    # build
    modal run win15.py::check --langs de --tag a --n 120             # alignment checker
    modal run win15.py::edge --tag sym --set-tag ttsSYM1 --hours de:0.6   # edge windows of one synthetic set
    python win15.py census <dir with windows_<lang>.jsonl + pool_<lang>.jsonl> [--langs ...]
"""

import hashlib
import json
import os
import random
import re
import sys

import ml_p3_core as C

VOL = "/vol"
WIN = C.WIN  # 14.96 s: the streaming decoder's window (ml_p3_core)
MIN_LEN = 12.0  # s: shortest window (the right edge is the last quiet gap <= WIN after the left one)
GAP = (0.15, 0.6)  # s between two stitched rows
MIN_NUM_ROWS = 2
END_FRAC = 1 / 3  # a number word must start in the first and end in the last END_FRAC of the window
WIN_MAX_CER = 0.15  # ml_p3.WIN_MAX_CER
MAX_ROWS = 12  # rows per stream
PHASES = 2  # passes over the streams while a language is short of its hours (new gaps / offsets each time)
SYN_SHARE = 0.30  # mix_build.TTS_SHARE: synthetic windows <= 30% of a language's window hours
EDGE_S = 3.0  # s of a row's start / end re-decoded for its edge check (ends_ok)
EDGE_PAD = (0.15, 0.2)  # s kept before a row's first and after its last word (letter frames; LEAD / TAIL are 0.05 / 0.08)


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ geometry (pure)
def reading_ids(row: dict) -> list:
    """Per whitespace token: the index of the number reading (num_chars range) it belongs to, or None."""
    out, pos, text = [], 0, row["text"]
    for tok in text.split():
        a = text.index(tok, pos)
        b = a + len(tok)
        pos = b
        out.append(next((j for j, (x, y) in enumerate(row.get("num_chars") or []) if x < b and a < y), None))
    return out


def guard_ids(row: dict) -> list:
    """reading_ids plus the row's `edge_spans` (the whole edge phrase: "a billion two", "twenty five basis points",
    "fifty uh sixty", "septante cinq"): every token of one edge phrase carries one key, so no head / middle / tail window and
    no cut puts an edge inside a multi-word phrase. A token in a number reading but outside every edge span keeps its
    reading index; rows without edge_spans are unchanged."""
    ids = reading_ids(row)
    text, pos = row["text"], 0
    for tok_i, tok in enumerate(text.split()):
        a = text.index(tok, pos)
        b = a + len(tok)
        pos = b
        for j, (x, y) in enumerate(row.get("edge_spans") or []):
            if x < b and a < y:
                ids[tok_i] = ("edge", j)
                break
    return ids


def layout(rows: list, gaps: list) -> tuple:
    """Rows end to end, row i+1 starting gaps[i] s after row i ends. Returns (starts, words, total); words =
    [(tok, t0, t1, reading key or None, row i, token k)] in stream seconds. Every row must carry word times."""
    starts, words, t = [], [], 0.0
    for i, r in enumerate(rows):
        starts.append(t)
        rid = r.get("_reading_ids") or guard_ids(r)
        for k, (tok, tm) in enumerate(zip(r["text"].split(), r["word_times"])):
            words.append((tok, tm[0] + t, tm[1] + t, None if rid[k] is None else (i, rid[k]), i, k))
        t += r["duration"] + (gaps[i] if i < len(rows) - 1 else 0.0)
    return starts, words, t


def cut_points(words: list, total: float, quiet) -> list:
    """[(time, j)]: a cut at `time` leaves words[:j] before it and words[j:] after it. A cut between two words needs
    >= MIN_GAP between their letter frames, a quiet frame there (quiet(t0, t1) -> time or None) and the two words not
    in one number reading; the stream's start and end are cuts too."""
    out = [(0.0, 0)]
    for j in range(1, len(words)):
        p, n = words[j - 1], words[j]
        if n[1] - p[2] < C.MIN_GAP or (p[3] is not None and p[3] == n[3]):
            continue
        c = quiet(p[2], n[1])
        if c is not None:
            out.append((round(c, 3), j))
    out.append((round(total, 3), len(words)))
    return out


def window_at(words: list, cuts: list, k: int, win: float = WIN, min_len: float = MIN_LEN):
    """The window from cut k to the last cut <= win s later (None when that is < min_len s or holds no word)."""
    a, j0 = cuts[k]
    best = None
    for c, j in cuts[k + 1:]:
        if c - a > win:
            break
        best = (c, j)
    if best is None or best[0] - a < min_len or best[1] <= j0:
        return None
    return {"a": a, "b": best[0], "j0": j0, "j1": best[1]}


def ends_ok(rows: list, words: list, w: dict, edge_ok=None) -> bool:
    """Every row whose first word is in the window re-decodes with that start, every row whose last word is in it
    with that end: edge_ok(i, "start" | "end") -> bool (None: row["_start_ok"] / row["_end_ok"], missing = fine). A
    source clip whose edge word is clipped or missing from its audio (tight caption segments) would put a label word
    on silence inside the window."""
    ok = edge_ok or (lambda i, side: rows[i].get(f"_{side}_ok", True))
    ws = words[w["j0"]:w["j1"]]
    for i in sorted({x[4] for x in ws}):
        ks = [x[5] for x in ws if x[4] == i]
        n = len(rows[i]["text"].split())
        if (0 in ks and not ok(i, "start")) or (n - 1 in ks and not ok(i, "end")):
            return False
    return True


def multi_ok(words: list, w: dict, min_rows: int = MIN_NUM_ROWS, end_frac: float = END_FRAC) -> tuple:
    """(ok, n number rows): >= min_rows rows with a number word inside the window, a number word starting in the
    first end_frac and one ending in the last end_frac of it."""
    ws = words[w["j0"]:w["j1"]]
    num = [x for x in ws if x[3] is not None]
    n_rows = len({x[4] for x in num})
    span = w["b"] - w["a"]
    first = any(x[1] - w["a"] <= end_frac * span for x in num)
    last = any(w["b"] - x[2] <= end_frac * span for x in num)
    return n_rows >= min_rows and first and last, n_rows


def plan_windows(rows: list, gaps: list, quiet, rng, win: float = WIN, min_len: float = MIN_LEN,
                 min_rows: int = MIN_NUM_ROWS, end_frac: float = END_FRAC, edge_ok=None) -> list:
    """Windows over one stream: the first left edge near U(0, 6) s, each next one near the previous right edge + U(-4,
    2) s (some overlap, never the same window twice); a left edge whose window fails multi_ok moves on by U(1, 3) s."""
    starts, words, total = layout(rows, gaps)
    cuts = cut_points(words, total, quiet)
    out, seen = [], set()
    target = rng.uniform(0.0, 6.0)
    while target < total - min_len:
        k = min(range(len(cuts)), key=lambda i: abs(cuts[i][0] - target))
        w = window_at(words, cuts, k, win, min_len)
        ok, n_rows = multi_ok(words, w, min_rows, end_frac) if w else (False, 0)
        ok = ok and ends_ok(rows, words, w, edge_ok)
        if not ok or (w["j0"], w["j1"]) in seen:
            target = max(target, cuts[k][0]) + rng.uniform(1.0, 3.0)
            continue
        seen.add((w["j0"], w["j1"]))
        ws = words[w["j0"]:w["j1"]]
        rws = sorted({x[4] for x in ws})
        # an edge inside a row (not at a row boundary): the window cuts that sentence
        cut_l = w["j0"] > 0 and words[w["j0"] - 1][4] == words[w["j0"]][4]
        cut_r = w["j1"] < len(words) and words[w["j1"]][4] == words[w["j1"] - 1][4]
        out.append(dict(w, text=" ".join(x[0] for x in ws), num_tokens=[x[3] is not None for x in ws],
                        word_times=[[round(x[1] - w["a"], 3), round(x[2] - w["a"], 3)] for x in ws],
                        src=[[x[4], x[5]] for x in ws], rows=rws, n_num_rows=n_rows, cut_left=cut_l, cut_right=cut_r,
                        seg=[[i, round(max(starts[i], w["a"]) - w["a"], 3),
                              round(min(starts[i] + rows[i]["duration"], w["b"]) - w["a"], 3)] for i in rws]))
        target = w["b"] + rng.uniform(-4.0, 2.0)
    return out


# edge windows (2026-10-18): a LONG number reading
# (>= LONG_MIN_WORDS words: phones, big amounts, decimals with units) placed at the window's start, middle or end --
# its end EDGE_TAIL s before the right edge ("tail"), its start EDGE_HEAD s after the left edge ("head": the previous
# window would end right before it) or its middle near the window's middle ("middle").
# Edges are the usual quiet cuts between words, never inside a reading, so the reading is whole and labels stay exact.
LONG_MIN_WORDS = 3
EDGE_TAIL = (0.1, 2.0)
EDGE_HEAD = (0.0, 1.5)
EDGE_MODES = ("head", "middle", "tail")  # start, middle and end, about evenly
MIN_PIECE_S = 0.25  # s: shortest stretch of a source row an edge window may hold


def long_targets(words: list, min_words: int = LONG_MIN_WORDS) -> list:
    """[(reading key, start, end)] of the readings with >= min_words words, in stream order."""
    by = {}
    for x in words:
        if x[3] is not None:
            by.setdefault(x[3], []).append(x)
    return [(k, xs[0][1], xs[-1][2]) for k, xs in by.items() if len(xs) >= min_words]


def edge_bucket(w: dict) -> str:
    """Where a window's long reading sits: "start" (begins <= 2 s after the left edge), "end" (ends <= 3 s before the
    right edge), "both" (a short window holding it from edge to edge), else "middle"."""
    a, b = w["edge_pos_s"]
    return "both" if a <= 2.0 and b <= 3.0 else "start" if a <= 2.0 else "end" if b <= 3.0 else "middle"


def plan_edge_windows(rows: list, gaps: list, quiet, rng, win: float = WIN, min_len: float = MIN_LEN,
                      min_rows: int = MIN_NUM_ROWS, edge_ok=None) -> list:
    """One window per long reading of the stream (when the geometry allows): "tail" -- the right edge is the cut
    after the reading closest to its end + U(EDGE_TAIL), the left edge a random cut min_len-win s earlier; "head" --
    the left edge is the cut before the reading closest to its start - U(EDGE_HEAD), the right edge a random cut
    min_len-win s later. Kept when >= min_rows rows have a number word inside, the target reading is whole and every
    cut row edge re-decodes (ends_ok). Window dicts as plan_windows, plus edge / edge_gap_s / edge_words."""
    starts, words, total = layout(rows, gaps)
    cuts = cut_points(words, total, quiet)
    out, seen = [], set()
    for key, t0, t1 in long_targets(words):
        mode = rng.choice(EDGE_MODES)
        if mode == "tail":
            want = t1 + rng.uniform(*EDGE_TAIL)
            near = [k for k, (c, _) in enumerate(cuts) if t1 - 1e-6 <= c <= t1 + EDGE_TAIL[1] + 0.6]
            if not near:
                continue
            kb = min(near, key=lambda k: abs(cuts[k][0] - want))
            far = [k for k, (c, _) in enumerate(cuts) if min_len <= cuts[kb][0] - c <= win]
            if not far:
                continue
            ka = rng.choice(far)
        elif mode == "middle":  # the reading's middle near the window's middle (+- 1 s)
            mid = (t0 + t1) / 2
            span = rng.uniform(min_len, win)
            want_a = mid - span / 2 + rng.uniform(-1.0, 1.0)
            ka = min(range(len(cuts)), key=lambda k: abs(cuts[k][0] - want_a))
            far = [k for k, (c, _) in enumerate(cuts) if min_len <= c - cuts[ka][0] <= win and c >= t1]
            if not far or cuts[ka][0] > t0:
                continue
            kb = min(far, key=lambda k: abs(cuts[k][0] - (cuts[ka][0] + span)))
        else:
            want = t0 - rng.uniform(*EDGE_HEAD)
            near = [k for k, (c, _) in enumerate(cuts) if t0 - EDGE_HEAD[1] - 0.6 <= c <= t0 + 1e-6]
            if not near:
                continue
            ka = min(near, key=lambda k: abs(cuts[k][0] - want))
            far = [k for k, (c, _) in enumerate(cuts) if min_len <= c - cuts[ka][0] <= win]
            if not far:
                continue
            kb = rng.choice(far)
        (a, j0), (b, j1) = cuts[ka], cuts[kb]
        if (j0, j1) in seen or j1 <= j0:
            continue
        w = {"a": a, "b": b, "j0": j0, "j1": j1}
        ws = words[j0:j1]
        if not all(x[3] != key or j0 <= n < j1 for n, x in enumerate(words)):
            continue  # the target reading must be whole inside
        n_rows = len({x[4] for x in ws if x[3] is not None})
        if n_rows < min_rows or not ends_ok(rows, words, w, edge_ok):
            continue
        rws = sorted({x[4] for x in ws})
        if any(min(starts[i] + rows[i]["duration"], b) - max(starts[i], a) < MIN_PIECE_S for i in rws):
            continue  # a source row with only a sliver inside the window (one short word at a cut)
        seen.add((j0, j1))
        rws = sorted({x[4] for x in ws})
        cut_l = j0 > 0 and words[j0 - 1][4] == words[j0][4]
        cut_r = j1 < len(words) and words[j1][4] == words[j1 - 1][4]
        out.append(dict(w, text=" ".join(x[0] for x in ws), num_tokens=[x[3] is not None for x in ws],
                        word_times=[[round(x[1] - a, 3), round(x[2] - a, 3)] for x in ws],
                        src=[[x[4], x[5]] for x in ws], rows=rws, n_num_rows=n_rows, cut_left=cut_l, cut_right=cut_r,
                        seg=[[i, round(max(starts[i], a) - a, 3), round(min(starts[i] + rows[i]["duration"], b) - a, 3)]
                             for i in rws],
                        edge=mode, edge_gap_s=round((b - t1) if mode == "tail" else (t0 - a), 3),
                        edge_pos_s=[round(t0 - a, 3), round(b - t1, 3)], edge_words=sum(1 for x in words if x[3] == key)))
    return out


def safe_quiet(x, t0: float, t1: float):
    """ml_p3_core.quiet_cut on [t0, t1] clipped to the audio (None when nothing is left of it)."""
    t0, t1 = max(0.0, t0), min(len(x) / 16000, t1)
    if t1 - t0 < 0.01:
        return None
    return C.quiet_cut(x, t0, t1)


def stitch(rows: list, gaps: list, seed: int):
    """The stream's samples: rows joined by gaps that carry the previous row's noise floor (windows2 rule)."""
    import numpy as np

    rs = np.random.RandomState(seed % (2 ** 31))
    parts = []
    for i, r in enumerate(rows):
        x = r["_x"]
        parts.append(x)
        if i < len(rows) - 1:
            n = len(x) // 160
            fr = np.sqrt(np.mean(x[:n * 160].reshape(n, 160) ** 2, axis=1) + 1e-12) if n else np.array([1e-4])
            parts.append((rs.randn(int(round(gaps[i] * 16000))) * float(np.percentile(fr, 10))).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def stream_gaps(n: int, rng) -> list:
    """n - 1 gaps, rounded to whole samples so layout() and stitch() agree."""
    return [round(rng.uniform(*GAP) * 16000) / 16000 for _ in range(max(n - 1, 0))]


def group(rows: list, rng, max_rows: int = MAX_ROWS) -> list:
    """Streams of consecutive rows of one session / video / voice (ml_p3_core.stream_key), <= max_rows each; rows
    without a source key ("misc") and leftover single rows are never stitched with strangers."""
    by = {}
    for r in rows:
        k, o = C.stream_key(r)
        if k != "misc":
            by.setdefault(k, []).append((o, r))
    out = []
    for k, v in sorted(by.items()):
        v = [r for _, r in sorted(v, key=lambda t: t[0])]
        for i in range(0, len(v), max_rows):
            if len(v[i:i + max_rows]) >= MIN_NUM_ROWS:
                out.append(v[i:i + max_rows])
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------------------ sources (pure)
def letters_eq(a: str, b: str, lang: str = "") -> bool:
    """Same label up to the compound-number form: equal letters as written, or after compound_split on both sides in
    any of its forms (fused / split / elided forms written out: it "ventuno" = "venti uno"). The build may write either
    form (its compound setting), so this never decides the form; it only refuses a different label."""
    if C.letters(a) == C.letters(b):
        return True
    if not lang:
        return False
    import compound_split as CS

    for kw in ({}, {"it_elided": True}):
        try:
            if C.letters(CS.split_text(a, lang, **kw)[0]) == C.letters(CS.split_text(b, lang, **kw)[0]):
                return True
        except TypeError:  # a compound_split without that option
            continue
    return False


def pick_sources(built: list, originals: dict, real: bool, lang: str = "") -> tuple:
    """Originals (as labelled on disk) of the rows a build trains on. built: the build's rows (written split or fused,
    per the build's split setting); originals: id -> row as labelled. A row is kept when its original exists with the
    same label up to the compound form (letters_eq), has a word time for every token and is not decided by a second
    recognizer (edges: trim_row). The window keeps the original's words; the build applies its own split setting.
    Returns (rows, {reason: n}); edge_not_quiet counts kept real rows without quiet edges (they are trimmed)."""
    from census_common import ext_row

    out, st = [], {"in": len(built)}
    for b in built:
        o = originals.get(b["id"])
        why = None
        if o is None:
            why = "no_original"
        elif not letters_eq(o["text"], b["text"], lang):
            why = "label_changed"
        elif ext_row(o):
            why = "ext"
        elif not C.has_times(o):
            why = "no_times"
        if why:
            st[why] = st.get(why, 0) + 1
            continue
        out.append(o)
        if real and not o.get("edge_quiet"):
            st["edge_not_quiet"] = st.get("edge_not_quiet", 0) + 1
    st["kept"] = len(out)
    # survival: kept / rows that could be used (not ext, with word times); the label check should drop almost none
    elig = st["in"] - st.get("ext", 0) - st.get("no_times", 0)
    st["survival"] = round(len(out) / max(elig, 1), 4)
    return out, st


def trim_row(r: dict, x, pad: tuple = EDGE_PAD) -> tuple:
    """(row, samples) cut to [first word - pad[0], last word + pad[1]] (clipped to the audio); word times follow."""
    n = len(x)
    a = max(0, int((min(t[0] for t in r["word_times"]) - pad[0]) * 16000))
    b = min(n, int((max(t[1] for t in r["word_times"]) + pad[1]) * 16000) + 1)
    s = a / 16000
    wt = [[round(t0 - s, 3), round(t1 - s, 3)] for t0, t1 in r["word_times"]]
    y = x[a:b]
    return dict(r, word_times=wt, duration=len(y) / 16000, trim=[round(s, 3), round(b / 16000, 3)]), y


def edges_ok(label: str, greedy: str) -> bool:
    """ml_p3_core.edges_match without the word split: the re-decode starts with the label's first word's letters and
    ends with its last word's letters (the reader writes "neunzehn hundert" for "neunzehnhundert"; an edge here is
    always a quiet gap, so no cut-word allowance is needed)."""
    lw = [C.letters(w) for w in label.split() if C.letters(w)]
    g = C.letters(greedy)
    return bool(lw and g) and g.startswith(lw[0]) and g.endswith(lw[-1])


def window_row(w: dict, stream: list, lang: str, wid: str, wav: str, dur: float, greedy: str, cer: float) -> dict:
    """The manifest row of a kept window (window-relative word times; parents' label hashes for the build check)."""
    par = [stream[i] for i in w["rows"]]
    syn = any(r.get("src") == "tts" for r in par)
    row = {"audio_filepath": wav, "duration": round(dur, 3), "text": w["text"], "lang": lang, "id": f"{lang}_{wid}",
           "origin": "win15", "num_tokens": w["num_tokens"], "word_times": w["word_times"],
           "parents": [r["id"] for r in par], "parent_sha": {r["id"]: sha(r["text"]) for r in par},
           "src": [[stream[i]["id"], k] for i, k in w["src"]],
           "seg": [[stream[i]["id"], a, b] for i, a, b in w["seg"]],
           "n_num_rows": w["n_num_rows"], "cut_left": w["cut_left"], "cut_right": w["cut_right"],
           "kinds": sorted({sp.get("kind", "?") for r in par for sp in r.get("spans") or []}),
           "stream": C.stream_key(par[0])[0], "greedy": greedy, "cer": round(cer, 3), "synthetic": syn}
    if syn:
        row["voice"] = par[0].get("voice")
    for k in ("edge", "edge_gap_s", "edge_pos_s", "edge_words"):  # edge windows (plan_edge_windows)
        if k in w:
            row[k] = w[k]
    return row


# ------------------------------------------------------------------------------------------------ census (pure)
def window_spans(w: dict, parents: dict, lang: str) -> list:
    """[(class, value, parent id, span index)] of every number reading the window holds whole (gate_class_value)."""
    from census_common import gate_class_value

    have = {(p, k) for p, k in w["src"]}
    out = []
    for pid in w["parents"]:
        r = parents.get(pid)
        if r is None:
            continue
        rid = reading_ids(r)
        for j, sp in enumerate(r.get("spans") or []):
            toks = [k for k, x in enumerate(rid) if x == j]
            if toks and all((pid, k) in have for k in toks):
                c, v = gate_class_value(r, sp, lang)
                out.append((c, v, pid, j))
    return out


def census(windows: list, pool: list, lang: str) -> dict:
    """Per class: spans in the windows vs the source pool (share, distinct values, top-1 / top-5 value share); years:
    top-year share, forward years (2031-2060), distinct; reuse: how many windows hold each source reading."""
    from collections import Counter

    from census_common import gate_class_value

    parents = {r["id"]: r for r in pool}

    def summ(spans):
        by = {}
        for c, v, *_ in spans:
            by.setdefault(c, []).append(v)
        n = sum(len(v) for v in by.values())
        out = {}
        for c, vs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            vc = Counter(str(v) for v in vs)
            top = vc.most_common(5)
            out[c] = {"spans": len(vs), "share": round(len(vs) / max(n, 1), 4), "distinct": len(vc),
                      "top1": round(top[0][1] / len(vs), 4), "top5": round(sum(x[1] for x in top) / len(vs), 4)}
            if c == "year":
                ys = [v for v in vs if isinstance(v, int)]
                out[c].update(fwd=sum(1 for y in ys if 2031 <= y <= 2060), fwd_distinct=len({y for y in ys if 2031 <= y <= 2060}),
                              top_year=top[0][0])
        return n, out

    ws = [s for w in windows for s in window_spans(w, parents, lang)]
    ps = [gate_class_value(r, sp, lang) + (r["id"], j) for r in pool for j, sp in enumerate(r.get("spans") or [])]
    nw, cw = summ(ws)
    npool, cp = summ(ps)
    reuse = Counter((p, j) for _, _, p, j in ws)
    skew = {c: round(cw[c]["share"] - cp.get(c, {}).get("share", 0.0), 4) for c in cw}
    return {"windows": len(windows), "hours": round(sum(w["duration"] for w in windows) / 3600, 3),
            "synthetic_h": round(sum(w["duration"] for w in windows if w.get("synthetic")) / 3600, 3),
            "spans": nw, "pool_spans": npool, "classes": cw, "pool_classes": cp, "share_minus_pool": skew,
            "max_abs_skew_pp": round(100 * max((abs(v) for v in skew.values()), default=0.0), 2),
            "reuse": {"readings": len(reuse), "max": max(reuse.values(), default=0),
                      "mean": round(sum(reuse.values()) / max(len(reuse), 1), 3)},
            "n_num_rows": dict(sorted(Counter(min(w["n_num_rows"], 5) for w in windows).items())),
            "cut_edges": {"left_in_sentence": sum(bool(w["cut_left"]) for w in windows),
                          "right_in_sentence": sum(bool(w["cut_right"]) for w in windows)},
            "duration": {"min": min((w["duration"] for w in windows), default=0),
                         "mean": round(sum(w["duration"] for w in windows) / max(len(windows), 1), 2),
                         "max": max((w["duration"] for w in windows), default=0)}}


# ------------------------------------------------------------------------------------------------ checker (pure)
def check_alignment(w: dict, aligned: list, tol: float = 0.4, seg_slack: float = 0.3) -> list:
    """Reasons a window's label does not sit on its audio, from a fresh forced alignment of the label against the
    window (aligned[k] = (t0, t1) or None per token): a token that does not align, one whose centre is > tol s from
    where its source row puts it, or one outside its source row's stretch of the window. [] = pass."""
    seg = {p: (a, b) for p, a, b in w["seg"]}
    why = []
    for k, ((pid, _), exp, got) in enumerate(zip(w["src"], w["word_times"], aligned)):
        if got is None:
            why.append(f"unaligned:{k}")
            continue
        c_exp, c_got = (exp[0] + exp[1]) / 2, (got[0] + got[1]) / 2
        if abs(c_exp - c_got) > tol:
            why.append(f"moved:{k}:{c_got - c_exp:+.2f}")
        a, b = seg[pid]
        if not (a - seg_slack <= c_got <= b + seg_slack):
            why.append(f"outside_source:{k}")
    return why


def boundary_fail(why: list) -> bool:
    """A check_alignment result that puts a word outside its source row (or nowhere): the label crosses a source
    boundary. A word only "moved" inside its own row is a timing disagreement, not a wrong label."""
    return any(r.startswith(("unaligned", "outside_source")) for r in why)


def pieces(w: dict) -> list:
    """[(parent id, a, b, label words)]: each source row's stretch of the window and the words the label gives it."""
    out = []
    for pid, a, b in w["seg"]:
        toks = [t for t, (p, _) in zip(w["text"].split(), w["src"]) if p == pid]
        out.append((pid, a, b, " ".join(toks)))
    return out


EDGE_LETTERS = 6  # letters compared at each end of a piece
EDGE_TOL = 0.34  # letter error rate allowed there (<= 2 of 6): a reader slip passes, a missing or extra word does not


def end_close(label: str, greedy: str, side: str, n: int = EDGE_LETTERS, tol: float = EDGE_TOL) -> bool:
    """The re-decode's first ("start") or last ("end") n letters are close to the label's (letter error rate <= tol)."""
    lab, g = C.letters(label), C.letters(greedy)
    if not lab or not g:
        return False
    return C.cer(lab[:n], g[:n]) <= tol if side == "start" else C.cer(lab[-n:], g[-n:]) <= tol


def edges_close(label: str, greedy: str, n: int = EDGE_LETTERS, tol: float = EDGE_TOL) -> bool:
    """Both ends close (end_close): edges_ok with room for the reader's spelling slips, still failing a word missing
    or added at an end."""
    return end_close(label, greedy, "start", n, tol) and end_close(label, greedy, "end", n, tol)


def piece_ok(label: str, greedy: str, max_cer: float = C.MAX_CER) -> bool:
    """A piece (or a whole source row) re-decodes to its label: CER <= max_cer and close ends (edges_close)."""
    return bool(label) and C.cer(label, greedy) <= max_cer and edges_close(label, greedy)


def shift_control(w: dict) -> dict | None:
    """Negative control for the piece check: the first source row's last word given to the second row (a label word
    on the wrong side of a source boundary). None when the window holds < 2 rows or row 1 has one word."""
    order = []
    for pid, _ in w["src"]:
        if pid not in order:
            order.append(pid)
    if len(order) < 2:
        return None
    k = max(i for i, (p, _) in enumerate(w["src"]) if p == order[0])
    if sum(p == order[0] for p, _ in w["src"]) < 2:
        return None
    src = [list(x) for x in w["src"]]
    src[k] = [order[1], -1]
    return dict(w, src=src)


def swap_control(w: dict) -> dict | None:
    """Negative control: the window's label with its first two source rows' words swapped (a label crossing source
    boundaries wrongly); word times / src follow the words. None when the window holds < 2 rows."""
    order = []
    for pid, _ in w["src"]:
        if pid not in order:
            order.append(pid)
    if len(order) < 2:
        return None
    toks = w["text"].split()
    idx = list(range(len(toks)))
    g0 = [i for i in idx if w["src"][i][0] == order[0]]
    g1 = [i for i in idx if w["src"][i][0] == order[1]]
    new = g1 + g0 + [i for i in idx if i not in g0 and i not in g1]
    return dict(w, text=" ".join(toks[i] for i in new), src=[w["src"][i] for i in new],
                word_times=[w["word_times"][i] for i in new])


# ------------------------------------------------------------------------------------------------ build hook (pure)
TOP_YEAR_SHARE = 0.03  # mix_build.TOP_YEAR_SHARE: no year value above 3% of the windows' year spans
FWD_SLACK = 1.25  # forward years (2031-2060) <= 1.25 x their share among the build's own year spans
VALUE_FLOOR = 5  # a value may always reach this many spans (the 3% rule is meaningless on few spans)


def _fwd(v) -> bool:
    return isinstance(v, int) and 2031 <= v <= 2060


def select_for_build(windows: list, parents: dict, bad: set, hours: float, rng, syn_share: float = SYN_SHARE,
                     lang: str = "") -> tuple:
    """mix_build.build4's win15 part: windows whose every parent is a row this build trains on with the label the
    window was cut from (parents: id -> the kept real + chosen synthetic rows as labelled on disk; parent_sha must
    match), none in `bad`; then real windows <= hours x (1 - syn_share), synthetic <= hours x syn_share (the build4
    window rule), real first, each window only while it keeps the value spread: no year value above TOP_YEAR_SHARE
    of the taken year spans (VALUE_FLOOR spans always allowed) and forward years <= FWD_SLACK x their share among
    the parents' year spans (synthetic sets are forward-heavy; 30% synthetic hours would double the forward share).
    Returns (rows, stats)."""
    from collections import Counter

    st = {"in": len(windows), "parent_not_trained": 0, "parent_label_changed": 0, "unsafe_parent": 0,
          "value_spread_skipped": 0}
    ok = []
    for w in windows:
        ps = w.get("parents") or []
        if not ps or any(p not in parents for p in ps):
            st["parent_not_trained"] += 1
        elif any(sha(parents[p]["text"]) != w.get("parent_sha", {}).get(p) for p in ps):
            st["parent_label_changed"] += 1
        elif set(ps) & bad:
            st["unsafe_parent"] += 1
        else:
            ok.append(w)
    rng.shuffle(ok)
    years = [v for r in parents.values() for c, v in _row_classes(r, lang) if c == "year"]
    fwd_cap = FWD_SLACK * sum(map(_fwd, years)) / max(len(years), 1)
    yc, n_fwd, out, h = Counter(), 0, [], {"real": 0.0, "syn": 0.0}
    for kind in ("real", "syn"):
        cap = hours * ((1 - syn_share) if kind == "real" else syn_share)
        for w in [x for x in ok if bool(x.get("synthetic")) == (kind == "syn")]:
            if h[kind] + w["duration"] / 3600 > cap + 1e-9:
                break
            ys = [v for c, v, *_ in window_spans(w, parents, lang) if c == "year"]
            n_y = sum(yc.values()) + len(ys)
            yc2 = yc + Counter(ys)
            if ys and (any(yc2[y] > max(VALUE_FLOOR, TOP_YEAR_SHARE * n_y) for y in set(ys))
                       or (n_fwd + sum(map(_fwd, ys)) > max(VALUE_FLOOR, fwd_cap * n_y) and any(map(_fwd, ys)))):
                st["value_spread_skipped"] += 1
                continue
            yc, n_fwd = yc2, n_fwd + sum(map(_fwd, ys))
            out.append(w)
            h[kind] += w["duration"] / 3600
    n_y = sum(yc.values())
    st.update(eligible=len(ok), taken=len(out), real_h=round(h["real"], 3), syn_h=round(h["syn"], 3),
              year_spans=n_y, top_year_share=round(max(yc.values(), default=0) / max(n_y, 1), 4),
              fwd_share=round(n_fwd / max(n_y, 1), 4), fwd_cap=round(fwd_cap, 4))
    return [{k: v for k, v in w.items() if k not in ("greedy",)} for w in out], st


def _row_classes(r: dict, lang: str) -> list:
    from census_common import gate_class_value

    return [gate_class_value(r, sp, lang) for sp in r.get("spans") or []]


# ------------------------------------------------------------------------------------------------ local census
def census_cli(argv: list) -> None:
    d = argv[0]
    opt = lambda k, dv: argv[argv.index(k) + 1] if k in argv else dv  # noqa: E731
    langs = opt("--langs", "de,es,fr,it,nl,pt,en").split(",")
    rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []  # noqa: E731
    out = {}
    for lang in langs:
        w, p = rd(f"{d}/windows_{lang}.jsonl"), rd(f"{d}/pool_{lang}.jsonl")
        if w:
            out[lang] = census(w, p, lang)
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ Modal
try:
    import modal
except ImportError:  # local tests / census need no Modal
    modal = None

if modal is not None:
    vol = modal.Volume.from_name("parakeet-spoken-form")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa", "numpy<2")
             .add_local_python_source("ml_cands", "ml_decide", "ml_p3_core", "census_common", "label_gpu", "labeller",
                                      "readings", "en_win", "ml_slav", "ml_balance", "compound_split")
             .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json")
             .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_build.py"), "/root/mix_build_src.py"))
    app = modal.App("parakeet-win15")

    def _rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

    def _tts_sets():
        return C.mix_build_consts("/root/mix_build_src.py")[0]

    def _ctc(lang: str):
        """The windows2 reader of the language; English: the en_win reader and spelling."""
        import ml_decide as D

        if lang != "en":
            return D.LangCTC(lang)
        import en_win as E
        from label_gpu import W2V

        class EnCTC(D.LangCTC):
            def __init__(self):
                super().__init__("en", "cuda", model=W2V)

            def ids(self, text):
                return E.en_ids(text, self.vocab, self.delim)

            def greedy(self, lp):
                return E.greedy_norm(super().greedy(lp))

        return EnCTC()

    def _sources(lang: str, mix_dir: str) -> tuple:
        """(real rows, synthetic rows, stats) for a wave-1 language: the build's rows as labelled on disk, clean audio."""
        import glob

        st = {}
        built_real = _rows(f"{VOL}/{mix_dir}/p3_{lang}.jsonl")
        built_syn = _rows(f"{VOL}/{mix_dir}/tts_{lang}.jsonl")
        orig = {}
        for p in sorted(glob.glob(f"{VOL}/ml/p3v2/{lang}/rows*.jsonl"),
                        key=lambda p: (not os.path.basename(p).endswith("_regional.jsonl"), p)):
            for x in _rows(p):  # mix_build._one_per_id order: a regional pass first, else the first pass by name
                if x["id"] not in orig:
                    orig[x["id"]] = x
                elif not letters_eq(orig[x["id"]]["text"], x["text"], lang):
                    orig[x["id"]].setdefault("_alt", []).append(x)
        # an id two passes label differently: the copy whose letters match the build's label
        built_txt = {b["id"]: b["text"] for b in built_real}
        for i, o in orig.items():
            if "_alt" in o and i in built_txt and not letters_eq(o["text"], built_txt[i], lang):
                m = [x for x in o["_alt"] if letters_eq(x["text"], built_txt[i], lang)]
                if m:
                    orig[i] = m[0]
        real, st["real"] = pick_sources(built_real, orig, real=True, lang=lang)
        sorig = {}
        for tag, _ in _tts_sets():
            for x in _rows(f"{VOL}/ml/{tag}/{lang}/rows.jsonl"):
                sorig.setdefault(x["id"], dict(x, _set=tag))
        syn, st["synthetic"] = pick_sources(built_syn, sorig, real=False, lang=lang)
        syn = [dict(x, src="tts") for x in syn]
        return real, syn, st

    def _en_sources(rows: list) -> tuple:
        """English prep rows minus held-out videos / days of the CURRENT lists."""
        import en_win as E

        vids = {l.strip() for l in open(f"{VOL}/d1/en4_ok/heldout_videos_enh.txt") if l.strip() and not l.startswith("#")}
        days = set()
        for l in open(f"{VOL}/d1/en4_ok/heldout_days_vox_en.txt"):
            m = re.match(r"(?:# group vox:)?(\d{8})", l.strip())
            if m:
                days.add(m.group(1))
        out = [r for r in rows if not E.held_out(r, vids, days)]
        return out, {"in": len(rows), "held_out": len(rows) - len(out), "kept": len(out)}

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=7)
    def build_lang(lang: str, tag: str, mix_dir: str, hours: float, seed: int = 1, en_rows: list | None = None) -> dict:
        from collections import Counter

        import librosa
        import soundfile as sf

        out = f"{VOL}/ml/win15{tag}/{lang}"
        assert not os.path.exists(f"{out}/windows.jsonl"), f"{out}/windows.jsonl exists: use a new tag"
        os.makedirs(f"{out}/wav", exist_ok=True)
        rng = random.Random(f"win15-{lang}-{seed}")
        st = Counter()
        ctc = _ctc(lang)
        if lang == "en":
            import en_win as E

            real, src_st = _en_sources(en_rows or [])
            syn = []
            src_st = {"real": src_st}
        else:
            real, syn, src_st = _sources(lang, mix_dir)

        def read(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        def prepare(stream):
            """Real audio lengths; English rows get word times by forced alignment (rows that do not align or whose
            re-decode is off leave the stream, as en_win.align); every token's reading id."""
            got = []
            for r in stream:
                try:
                    x = read(r["audio_filepath"])
                except Exception:  # noqa: BLE001
                    st["row_unreadable"] += 1
                    continue
                r = dict(r, duration=len(x) / 16000)
                if lang == "en":
                    lp = ctc.lp(x)
                    tm = ctc.word_times(lp, r["text"])
                    if any(t is None for t in tm) or C.cer(r["text"], ctc.greedy(lp)) > C.MAX_CER:
                        st["en_row_unaligned_or_cer"] += 1
                        continue
                    r["word_times"] = [[round(a, 3), round(b, 3)] for a, b in tm]
                    # English: a cut never splits a labeller reading or an unmarked run of number words
                    groups = E.number_groups(r["text"], r.get("num_chars"), r.get("edge_spans"))
                    rid = [None] * len(r["text"].split())
                    for gi, g in enumerate(groups):
                        for k in g:
                            rid[k] = gi
                    r["_reading_ids"] = rid
                    g = ctc.greedy(lp)  # English: the aligned row's own re-decode gives both edge flags
                    r["_start_ok"], r["_end_ok"] = end_close(r["text"], g, "start"), end_close(r["text"], g, "end")
                r, x = trim_row(r, x)
                st["rows_prepared"] += 1
                got.append(dict(r, _x=x))
            return got

        rows_out, kept_s, rejects = [], {"real": 0.0, "syn": 0.0}, []
        budget = {"real": hours * (1 - SYN_SHARE) * 3600 if syn else hours * 3600, "syn": hours * SYN_SHARE * 3600 if syn else 0.0}
        f = open(f"{out}/windows.jsonl", "w", encoding="utf-8")
        for kind, pool, ph in [(k, p, ph) for k, p in (("real", real), ("syn", syn)) for ph in range(PHASES)]:
            if kept_s[kind] >= budget[kind]:
                continue
            streams = group(pool, rng)  # a later phase: the same streams in a new order, new gaps and offsets
            st[f"{kind}_streams"] = len(streams)
            st[f"{kind}_phases"] = ph + 1
            for si, stream in enumerate(streams):
                if kept_s[kind] >= budget[kind]:
                    break
                stream = prepare(stream)
                if len(stream) < MIN_NUM_ROWS:
                    continue
                gaps = stream_gaps(len(stream), rng)
                x_all = stitch(stream, gaps, seed=zlib_seed(lang, kind, ph, si))
                total = len(x_all) / 16000
                quiet = lambda t0, t1: safe_quiet(x_all, t0, t1)  # noqa: E731
                flags = {}

                def edge_ok(i, side, stream=stream, flags=flags):
                    """Does row i's first / last EDGE_S s re-decode to its label's first / last letters (cached)?"""
                    r = stream[i]
                    if f"_{side}_ok" in r:
                        return r[f"_{side}_ok"]
                    if (i, side) not in flags:
                        n = int(EDGE_S * 16000)
                        g = ctc.greedy(ctc.lp(r["_x"][:n] if side == "start" else r["_x"][-n:]))
                        flags[(i, side)] = end_close(r["text"], g, side)
                        st[f"row_{side}_checked"] += 1
                        st[f"row_{side}_bad"] += not flags[(i, side)]
                    return flags[(i, side)]

                try:
                    planned = plan_windows(stream, gaps, quiet, rng, edge_ok=edge_ok)
                except Exception as e:  # noqa: BLE001  (one bad stream must not end the language)
                    st[f"{kind}_stream_error"] += 1
                    if len(rejects) < 40:
                        rejects.append({"stream_error": repr(e)[:200], "ids": [r["id"] for r in stream][:3]})
                    continue
                for j, w in enumerate(planned):
                    if kept_s[kind] >= budget[kind]:
                        break
                    st[f"{kind}_planned"] += 1
                    x = x_all[int(round(w["a"] * 16000)):int(round(min(w["b"], total) * 16000))]
                    lp = ctc.lp(x)
                    g = ctc.greedy(lp)
                    cer = C.cer(w["text"], g)
                    if cer > WIN_MAX_CER:
                        st[f"{kind}_cer"] += 1
                        continue
                    if not edges_ok(w["text"], g):
                        st[f"{kind}_edge_word"] += 1
                        if len(rejects) < 40:
                            rejects.append({"label": w["text"], "greedy": g})
                        continue
                    # a word the forced alignment puts outside its own source row: the label crosses a boundary
                    why = check_alignment(dict(w, src=[[stream[i]["id"], k] for i, k in w["src"]],
                                               seg=[[stream[i]["id"], a, b] for i, a, b in w["seg"]]),
                                          ctc.word_times(lp, w["text"]))
                    st[f"{kind}_moved_word"] += any(r.startswith("moved") for r in why)
                    if boundary_fail(why):
                        st[f"{kind}_align_boundary"] += 1
                        continue
                    wid = f"w15{tag}_{kind[0]}{ph}_{si}_{j}"
                    wav = f"{out}/wav/{wid}.wav"
                    assert not os.path.exists(wav), wav
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    row = window_row(w, stream, lang, wid, wav, len(x) / 16000, g, cer)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_out.append(row)
                    kept_s[kind] += len(x) / 16000
                    st[f"{kind}_kept"] += 1
                if si % 40 == 0:
                    f.flush()
                    vol.commit()
        f.close()
        pool_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in real + syn]
        cen = census(rows_out, pool_rows, lang)
        res = {"lang": lang, "tag": tag, "mix_dir": mix_dir, "hours_target": hours, "sources": src_st,
               "kept_h": {k: round(v / 3600, 3) for k, v in kept_s.items()}, "counts": dict(st), "census": cen,
               "edge_rejects": rejects}
        json.dump(res, open(f"{out}/stats.json", "w"), indent=1, ensure_ascii=False)
        # the source pool the census compares against (ids + spans only)
        with open(f"{out}/pool.jsonl", "w", encoding="utf-8") as pf:
            for r in pool_rows:
                pf.write(json.dumps({k: r.get(k) for k in ("id", "text", "spans", "num_chars", "src", "voice",
                                                           "audio_filepath")},
                                    ensure_ascii=False) + "\n")
        vol.commit()
        return res

    def zlib_seed(*parts) -> int:
        import zlib

        return zlib.crc32("|".join(map(str, parts)).encode())

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, max_containers=7)
    def check_lang(lang: str, tag: str, n: int, seed: int = 7) -> dict:
        """Independent label check on a random sample of windows (the build already dropped windows whose forced
        alignment put a word outside its source row):
          piece  -- each source row's stretch of the window re-decoded alone must give that row's labelled words
                    (piece_ok: CER <= MAX_CER, same first / last word): a word on the wrong side of a source boundary
                    is missing from one piece and extra in the next;
          base   -- the same check on the whole source rows (their own audio and label): the reader's error floor;
          align  -- check_alignment on the whole window (boundary = outside its row; moved = > 0.4 s inside it);
          control-- shift_control (row 1's last word given to row 2) through the piece check, swap_control (rows 1
                    and 2 swapped) through the alignment check: each check must catch its wrong label."""
        import librosa
        import soundfile as sf

        d = f"{VOL}/ml/win15{tag}/{lang}"
        dest = f"{d}/check_{n}_{seed}.json"
        assert not os.path.exists(dest), dest
        ws = _rows(f"{d}/windows.jsonl")
        pool = {r["id"]: r for r in _rows(f"{d}/pool.jsonl")}
        random.Random(f"chk-{lang}-{seed}").shuffle(ws)
        ws = ws[:n]
        ctc = _ctc(lang)

        def read(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        res = {"lang": lang, "n": len(ws), "piece_fail": 0, "pieces": 0, "pieces_bad": 0, "align_boundary_fail": 0,
               "align_moved": 0,
               "base_rows": 0, "base_fail": 0, "control_n": 0, "control_piece_caught": 0, "control_align_caught": 0,
               "synthetic": sum(bool(w.get("synthetic")) for w in ws), "fails": []}
        base_done = set()
        for w in ws:
            x = read(w["audio_filepath"])
            lp = ctc.lp(x)
            why = check_alignment(w, ctc.word_times(lp, w["text"]))
            res["align_boundary_fail"] += boundary_fail(why)
            res["align_moved"] += any(r.startswith("moved") for r in why)
            bad = []
            for pid, a, b, lab in pieces(w):
                g = ctc.greedy(ctc.lp(x[int(a * 16000):int(b * 16000)]))
                res["pieces"] += 1
                if not piece_ok(lab, g):
                    res["pieces_bad"] += 1
                    bad.append({"parent": pid, "label": lab, "greedy": g})
                if pid not in base_done and pool.get(pid, {}).get("audio_filepath"):
                    base_done.add(pid)
                    res["base_rows"] += 1
                    res["base_fail"] += not piece_ok(pool[pid]["text"], ctc.greedy(ctc.lp(read(pool[pid]["audio_filepath"]))))
            if bad:
                res["piece_fail"] += 1
                res["fails"].append({"id": w["id"], "synthetic": w.get("synthetic"), "pieces": bad[:3],
                                     "align": why[:4]})
            cw, sw = swap_control(w), shift_control(w)
            if cw is not None and sw is not None:
                res["control_n"] += 1
                res["control_align_caught"] += boundary_fail(check_alignment(cw, ctc.word_times(lp, cw["text"])))
                res["control_piece_caught"] += any(
                    not piece_ok(lab, ctc.greedy(ctc.lp(x[int(a * 16000):int(b * 16000)])))
                    for _, a, b, lab in pieces(sw))
        for k in ("piece_fail", "align_boundary_fail", "align_moved"):
            res[f"{k}_rate"] = round(res[k] / max(res["n"], 1), 4)
        res["base_fail_rate"] = round(res["base_fail"] / max(res["base_rows"], 1), 4)
        res["piece_bad_rate"] = round(res["pieces_bad"] / max(res["pieces"], 1), 4)
        for k in ("control_piece_caught", "control_align_caught"):
            res[f"{k}_rate"] = round(res[k] / max(res["control_n"], 1), 4)
        json.dump(res, open(dest, "w"), indent=1, ensure_ascii=False)
        vol.commit()
        return res

    @app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3 * 3600, max_containers=7)
    def build_edge(lang: str, tag: str, set_tag: str, hours: float, seed: int = 1, root: str = "") -> dict:
        """Edge windows (plan_edge_windows) from synthetic sets' verified rows (ml/<set_tag>/<lang>/rows.jsonl, set_tag
        "a,b" for several sets;
        clean audio, one voice per stream); rows with a letter+digit code are left out (the build re-labels codes in
        rows, not in windows). The same CTC re-decode / edge / alignment checks as build_lang. A build keeps a window
        only when every parent row is one it trains on (select_for_build), so windows are cut from the whole set."""
        from collections import Counter

        import librosa
        import soundfile as sf

        out = f"{VOL}/ml/{root or 'win15' + tag}/{lang}"   # root: another top directory (the edge pack: edgeA/win)
        assert not os.path.exists(f"{out}/windows.jsonl"), f"{out}/windows.jsonl exists: use a new tag"
        os.makedirs(f"{out}/wav", exist_ok=True)
        rng = random.Random(f"win15edge-{lang}-{seed}")
        st = Counter()
        ctc = _ctc(lang)
        src = [dict(x, src="tts", _set=t) for t in set_tag.split(",") for x in _rows(f"{VOL}/ml/{t}/{lang}/rows.jsonl")]
        syn = [x for x in src if C.has_times(x) and not any(sp.get("kind") == "alnum" for sp in x.get("spans") or [])]
        st["source_rows"], st["source_alnum_or_untimed"] = len(src), len(src) - len(syn)

        def read(p):
            x, sr = sf.read(p, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)

        rows_out, kept_s, rejects, modes = [], 0.0, [], Counter()
        f = open(f"{out}/windows.jsonl", "w", encoding="utf-8")
        for ph in range(PHASES):
            if kept_s >= hours * 3600:
                break
            streams = group(syn, rng)
            for si, stream in enumerate(streams):
                if kept_s >= hours * 3600:
                    break
                got = []
                for r in stream:
                    try:
                        x = read(r["audio_filepath"])
                    except Exception:  # noqa: BLE001
                        st["row_unreadable"] += 1
                        continue
                    r, x = trim_row(dict(r, duration=len(x) / 16000), x)
                    got.append(dict(r, _x=x))
                stream = got
                if len(stream) < MIN_NUM_ROWS:
                    continue
                gaps = stream_gaps(len(stream), rng)
                x_all = stitch(stream, gaps, seed=zlib_seed(lang, "edge", ph, si))
                total = len(x_all) / 16000
                quiet = lambda t0, t1: safe_quiet(x_all, t0, t1)  # noqa: E731
                flags = {}

                def edge_ok(i, side, stream=stream, flags=flags):
                    r = stream[i]
                    if (i, side) not in flags:
                        n = int(EDGE_S * 16000)
                        g = ctc.greedy(ctc.lp(r["_x"][:n] if side == "start" else r["_x"][-n:]))
                        flags[(i, side)] = end_close(r["text"], g, side)
                        st[f"row_{side}_bad"] += not flags[(i, side)]
                    return flags[(i, side)]

                planned = plan_edge_windows(stream, gaps, quiet, rng, edge_ok=edge_ok)
                for j, w in enumerate(planned):
                    if kept_s >= hours * 3600:
                        break
                    st["planned"] += 1
                    x = x_all[int(round(w["a"] * 16000)):int(round(min(w["b"], total) * 16000))]
                    lp = ctc.lp(x)
                    g = ctc.greedy(lp)
                    cer = C.cer(w["text"], g)
                    if cer > WIN_MAX_CER:
                        st["cer"] += 1
                        continue
                    if not edges_ok(w["text"], g):
                        st["edge_word"] += 1
                        if len(rejects) < 40:
                            rejects.append({"label": w["text"], "greedy": g})
                        continue
                    why = check_alignment(dict(w, src=[[stream[i]["id"], k] for i, k in w["src"]],
                                               seg=[[stream[i]["id"], a, b] for i, a, b in w["seg"]]),
                                          ctc.word_times(lp, w["text"]))
                    if boundary_fail(why):
                        st["align_boundary"] += 1
                        continue
                    wid = f"w15{tag}_e{ph}_{si}_{j}"
                    wav = f"{out}/wav/{wid}.wav"
                    assert not os.path.exists(wav), wav
                    sf.write(wav, x, 16000, subtype="PCM_16")
                    row = window_row(w, stream, lang, wid, wav, len(x) / 16000, g, cer)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_out.append(row)
                    kept_s += len(x) / 16000
                    st["kept"] += 1
                    modes[w["edge"]] += 1
                if si % 40 == 0:
                    f.flush()
                    vol.commit()
        f.close()
        pool_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in syn]
        gaps_s = [r["edge_gap_s"] for r in rows_out]
        res = {"lang": lang, "tag": tag, "set": set_tag, "hours_target": hours, "kept_h": round(kept_s / 3600, 3),
               "counts": dict(st), "modes": dict(modes), "position": dict(Counter(edge_bucket(r) for r in rows_out)),
               "edge_gap_s": {"min": min(gaps_s, default=None), "max": max(gaps_s, default=None),
                              "mean": round(sum(gaps_s) / max(len(gaps_s), 1), 3)},
               "census": census(rows_out, pool_rows, lang), "edge_rejects": rejects}
        json.dump(res, open(f"{out}/stats.json", "w"), indent=1, ensure_ascii=False)
        with open(f"{out}/pool.jsonl", "w", encoding="utf-8") as pf:
            for r in pool_rows:
                pf.write(json.dumps({k: r.get(k) for k in ("id", "text", "spans", "num_chars", "src", "voice",
                                                           "audio_filepath")}, ensure_ascii=False) + "\n")
        vol.commit()
        return res

    @app.local_entrypoint()
    def edge(langs: str = "de,es,fr,it,nl,pt", tag: str = "sym", set_tag: str = "ttsSYM1", hours: str = "",
             default_h: float = 0.5, seed: int = 1, root: str = ""):
        """Edge windows of one synthetic set: modal run win15.py::edge --tag sym --set-tag ttsSYM1 --hours de:0.6,nl:0.1"""
        over = {k: float(v) for k, v in (x.split(":") for x in hours.split(",") if x)}
        for r in build_edge.starmap([(l, tag, set_tag, over.get(l, default_h), seed, root) for l in langs.split(",")]):
            print(json.dumps({k: r[k] for k in ("lang", "kept_h", "counts", "modes", "edge_gap_s")}, ensure_ascii=False))

    @app.local_entrypoint()
    def main(langs: str = "de,es,fr,it,nl,pt", tag: str = "a", mix_dir: str = "mix4final8", share: float = 0.12,
             en_hours: float = 4.0, en_rows: str = "", hours: str = "", seed: int = 1):
        """share: each wave-1 language's window hours = share x its number hours in mix_dir/stats.json (number_h);
        hours ("de:0.2,en:0.1") overrides per language (smoke runs). en_rows: the en_win prep rows (local file)."""
        import subprocess
        import tempfile

        over = {k: float(v) for k, v in (x.split(":") for x in hours.split(",") if x)}
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["modal", "volume", "get", "parakeet-spoken-form", f"{mix_dir}/stats.json", f"{td}/s.json"],
                           check=True, capture_output=True)
            nh = {l: v.get("number_h", 0.0) for l, v in json.load(open(f"{td}/s.json"))["langs"].items()}
        calls = []
        for l in langs.split(","):
            h = over.get(l, en_hours if l == "en" else round(share * nh.get(l, 0.0), 2))
            er = [json.loads(x) for x in open(en_rows, encoding="utf-8") if x.strip()] if l == "en" else None
            assert l != "en" or er, "English needs --en-rows"
            print(f"{l}: {h} h of windows")
            calls.append(build_lang.spawn(l, tag, mix_dir, h, seed, er))
        for c in calls:
            try:
                r = c.get()
            except Exception as e:  # noqa: BLE001  (the other languages keep running)
                print(f"FAILED: {e!r}")
                continue
            print(json.dumps({k: r[k] for k in ("lang", "kept_h", "counts")}, ensure_ascii=False))

    @app.local_entrypoint()
    def check(langs: str = "de,es,fr,it,nl,pt,en", tag: str = "a", n: int = 120, seed: int = 7):
        for r in check_lang.starmap([(l, tag, n, seed) for l in langs.split(",")]):
            print(json.dumps({k: v for k, v in r.items() if k != "fails"}, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "census":
        census_cli(sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
