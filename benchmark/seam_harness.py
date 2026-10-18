#!/usr/bin/env python3
"""Reader-only seam harness: a plain sliding-window decode of stitched files, a word merge at the middle of each
window overlap, and the public reader as the only number parser. No streaming engine and no external parser.

Geometry ("round"): 15 s windows every 10 s (5 s overlap); each file is decoded at 5 start offsets (0, 2, 4, 6, 8 s of
leading silence, so window edges fall at different places in every sentence); adjacent windows are merged at the
midpoint of their overlap (words whose centre lies before the midpoint come from the earlier window, the rest from the
later one). Every sentence is also decoded alone ("single"). Decoding runs on Modal (modal_bench.py::seam); this file
is the stdlib side: geometry, merge, sentence assignment, the reader parse, and the seam-scorer files.

usage:
  seam_harness.py build --manifest M.json --decoded DEC.json --arm NAME --work DIR [--offsets 5]
      DEC.json = modal_bench.py output: {"long": {"<file>_o<off>": {"dur", "windows": [{"start", "words":
      [[word, t0, t1], ...]}]}}, "single": {sentence id: text}, "geometry": {...}}
      writes DIR/cases_NAME.jsonl, parse_out_NAME.jsonl, parse_in_NAME.jsonl (eval/quality/README.md formats) and the
      same three files under DIR/scorable/ with only the slots the scorer gets right on BOTH the spoken-form label
      and the written caption (oracle filter: a slot a perfect words model or a perfect digits model would miss is a
      label or reader artefact and never counts against either).
  seam_harness.py score DIR ARM [ARM ...]       eval/quality/offset_score.py on DIR and DIR/scorable (first arm = base)

Manifest: {file: [{id, lang, start, end, text, label, values: {slot: value}}]} with times in seconds in the
offset-0 file; `label` is the spoken-form text of the sentence, `values` its expected numbers by slot
(score_values.py slot types). stitch_bed.py writes one from held-out label rows.
"""
import argparse
import bisect
import functools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WIN_S, STRIDE_S, N_OFFSETS = 15.0, 10.0, 5
GAP_HALF_S = 1.5  # merge gaps: the two windows are compared within this distance of the merge point
EDGE_S = 0.5  # a number word this close to its sentence's boundary is flagged (offset_score.py --guard)


# ---------------------------------------------------------------- geometry
def offsets(n: int = N_OFFSETS, stride: float = STRIDE_S) -> list:
    """n start offsets spread evenly over one stride: 0, 2, 4, 6, 8 s for the round geometry"""
    return [round(k * stride / n, 3) for k in range(n)]


def window_starts(dur: float, win: float = WIN_S, stride: float = STRIDE_S) -> list:
    """window start times covering [0, dur]: 0, stride, 2 stride, ... until a window reaches the end"""
    starts = [0.0]
    while starts[-1] + win < dur:
        starts.append(round(starts[-1] + stride, 3))
    return starts


# ---------------------------------------------------------------- merge
def _norm(ws) -> list:
    return [t for w in ws for t in C.fold_tokens(w[0] if isinstance(w, (list, tuple)) else w)]


def merge_windows(windows: list, dur: float, win: float = WIN_S) -> tuple:
    """windows: [{"start": s, "words": [[word, t0, t1], ...]}] (times relative to the window).
    Returns (words [[word, abs t0, abs t1]], gaps [{"t", "pick", "A", "B", "R"}]): one gap per overlap where the two
    windows' words within GAP_HALF_S of the merge point differ (A = earlier window, B = later window, R = what the
    merge kept there)."""
    W = sorted(windows, key=lambda x: x["start"])
    absw = [[[w, x["start"] + a, x["start"] + b] for w, a, b in x["words"]] for x in W]
    ends = [min(x["start"] + win, dur) for x in W]
    mids = [(W[i + 1]["start"] + ends[i]) / 2 for i in range(len(W) - 1)]
    out = []
    for i, ws in enumerate(absw):
        lo = mids[i - 1] if i > 0 else float("-inf")
        hi = mids[i] if i < len(mids) else float("inf")
        out += [w for w in ws if lo <= (w[1] + w[2]) / 2 < hi]
    gaps = []
    for i, m in enumerate(mids):
        # compare the two windows around the merge point only: the words cut by either window's own edge (the first
        # and last second of the overlap) differ in almost every overlap and are never what the merge keeps
        a0, a1 = max(W[i + 1]["start"], m - GAP_HALF_S), min(ends[i], m + GAP_HALF_S)
        A = [w for w in absw[i] if a0 <= (w[1] + w[2]) / 2 < a1]
        B = [w for w in absw[i + 1] if a0 <= (w[1] + w[2]) / 2 < a1]
        if _norm(A) != _norm(B):
            R = [w for w in out if a0 <= (w[1] + w[2]) / 2 < a1]
            gaps.append({"t": round(m, 3), "pick": "mid", "A": " ".join(w[0] for w in A),
                         "B": " ".join(w[0] for w in B), "R": " ".join(w[0] for w in R)})
    return out, gaps


# ---------------------------------------------------------------- sentences
@functools.lru_cache(maxsize=None)
def is_number_word(word: str, lang: str) -> bool:
    return bool(re.search(r"\d", word)) or bool(C.N.extract_numbers(word, lang, keep_weak=False))


def assign(words: list, sents: list, off: float) -> list:
    """words of the merged long decode -> one list per sentence (sentences sorted by start; boundaries at the middle
    of the silence between two sentences, shifted by the file's offset)"""
    bounds = [(sents[i]["end"] + sents[i + 1]["start"]) / 2 + off for i in range(len(sents) - 1)]
    per = [[] for _ in sents]
    for w in words:
        per[bisect.bisect_right(bounds, (w[1] + w[2]) / 2)].append(w)
    return per


def build_cases(manifest: dict, decoded: dict, n_off: int = N_OFFSETS) -> list:
    """cases rows (eval/quality/README.md): one per sentence per offset, in manifest file order then offset"""
    geo = decoded.get("geometry", {})
    stride = geo.get("stride", STRIDE_S)
    win = geo.get("win", WIN_S)
    offs = geo.get("offsets") or offsets(n_off, stride)
    rows = []
    for f, sents in manifest.items():
        sents = sorted(sents, key=lambda s: s["start"])
        for off in offs:
            key = f"{f}_o{off:g}"
            d = decoded["long"][key]  # a missing decode is an error
            words, gaps = merge_windows(d["windows"], d["dur"], win)
            per = assign(words, sents, off)
            gtimes = [g["t"] for g in gaps]
            for i, (s, ws) in enumerate(zip(sents, per)):
                lang = s["lang"]
                lo = (sents[i - 1]["end"] + s["start"]) / 2 + off if i else float("-inf")
                hi = (s["end"] + sents[i + 1]["start"]) / 2 + off if i + 1 < len(sents) else float("inf")
                long_text = " ".join(w[0] for w in ws)
                single = decoded["single"][s["id"]]
                nts = [round((w[1] + w[2]) / 2, 3) for w in ws if is_number_word(w[0], lang)]
                rows.append({
                    "id": s["id"], "lang": lang, "values": {k: str(v) for k, v in s["values"].items()},
                    "long": long_text, "single": single, "same": C.fold_tokens(long_text) == C.fold_tokens(single),
                    "merge_gaps": [g for g in gaps if lo <= g["t"] < hi], "file": key,
                    "num_t": nts, "file_gaps": gtimes,
                    "edge_num": [bool(nts) and nts[0] - lo < EDGE_S, bool(nts) and hi - nts[-1] < EDGE_S],
                })
    return rows


def parse_rows(cases: list) -> tuple:
    """(parse_out rows, parse_in rows) from the public reader only: `values` = the reader's greedy reading, `alt` =
    its no-joiner reading with word spans (common.readings_of)."""
    out, pin = [], []
    for k, r in enumerate(cases):
        for field in ("long", "single"):
            text = r[field] or ""
            g, nj = C.readings_of(text, r["lang"])
            key = f"{k}:{field}"
            out.append({"key": key, "values": g.vals, "alt": nj.vals, "alt_spans": nj.spans, "words": nj.words})
            pin.append({"key": key, "lang": r["lang"], "text": text})
    return out, pin


def oracle_slots(manifest: dict) -> dict:
    """{sentence id: set of slots the scorer gets right BOTH on the spoken-form label (a perfect words model) and on
    the written caption (a perfect digits model)}. A slot either perfect model would miss is a label or reader
    artefact: it is wrong at every offset for that model and only moves the always-wrong count."""
    keep = {}
    for sents in manifest.values():
        for s in sents:
            vals = {k: str(v) for k, v in s["values"].items()}
            ok = set(vals)
            for field in ("label", "text"):
                if s.get(field):
                    right = C.SV.match_slots(vals, C.readings_of(s[field], s["lang"]))
                    ok &= {k for k, r in right.items() if r}
            keep[s["id"]] = ok
    return keep


def write_arm(work: Path, arm: str, cases: list, manifest: dict) -> None:
    po, pi = parse_rows(cases)
    for d, rows in ((work, cases), (work / "scorable", None)):
        if rows is None:
            keep = oracle_slots(manifest)
            rows = [{**r, "values": {k: v for k, v in r["values"].items() if k in keep[r["id"]]}} for r in cases]
        C.write_jsonl(d / f"cases_{arm}.jsonl", rows)
        C.write_jsonl(d / f"parse_out_{arm}.jsonl", po)
        C.write_jsonl(d / f"parse_in_{arm}.jsonl", pi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True, type=Path)
    b.add_argument("--decoded", required=True, type=Path)
    b.add_argument("--arm", required=True)
    b.add_argument("--work", required=True, type=Path)
    b.add_argument("--offsets", type=int, default=N_OFFSETS)
    s = sub.add_parser("score")
    s.add_argument("work", type=Path)
    s.add_argument("arms", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        manifest = json.load(open(a.manifest, encoding="utf-8"))
        decoded = json.load(open(a.decoded, encoding="utf-8"))
        manifest = {f: v for f, v in manifest.items() if any(k.startswith(f + "_o") for k in decoded["long"])}
        cases = build_cases(manifest, decoded, a.offsets)
        write_arm(a.work, a.arm, cases, manifest)
        n_gap = sum(bool(r["merge_gaps"]) for r in cases)
        print(f"{a.arm}: {len(manifest)} files, {len(cases)} sentence readings ({n_gap} merge-touched) -> {a.work}")
        return cases
    import offset_score as OS  # eval/quality (read only)

    for d, name in ((a.work, "all slots"), (a.work / "scorable", "scorable slots (scorer right on the label and on the caption)")):
        print(f"\n######## {name}: {d}")
        OS.main([str(d)] + a.arms + ["--json", str(d / "offset_score.json")])


if __name__ == "__main__":
    main()
