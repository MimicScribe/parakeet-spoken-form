"""Re-label existing rows to the letter+digit policy (2026-10-18; ml_p3_core.apply_alnum_letters): a letter
code's letters as written + its number as spoken words ("ge zwanzig" -> "G zwanzig", "ce o zwei" / "ze o zwei" ->
"CO zwei", "fünf ge" -> "fünf G"). Real rows only (wave-1 languages); English and synthetic rows are left alone.

Rows with spans (labelling passes, mix p3 parts): text, num_chars, the span's reading and word_times (the letter
names of one code merge into one token: [first start, last end]) are rewritten. Token-list rows (windows, seam
halves: "src" = [parent id, parent token index] per token) are rewritten through their parents' token maps: the window
tokens that came from one code's letter names merge into the parent's new letter token; every src index is re-pointed
at the re-labelled parent; num_tokens merges by OR; a seam half's reading / reading_part follow its parent's span.

    python3 alnum_relabel.py --lang de --parents p3_de.jsonl rows_*.jsonl --rows p3_de.jsonl \
        --tokens win_de.jsonl seam_de.jsonl seam_long_de.jsonl --out-dir out/de

Writes <out-dir>/<name> for every input file (same rows, re-labelled where needed), changed_<name> (the changed rows
only), and alnum_relabel_stats.json. Pure Python; no audio is touched.
"""

import argparse
import json
import os

import ml_p3_core as C


def _rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def relabel_row(r: dict, lang: str) -> tuple:
    """(new row, token map) for a row with spans / num_chars; (r, None) when nothing changes."""
    if r.get("synthetic") or not r.get("num_chars") or not r.get("spans"):
        return r, None
    if len(r["spans"]) != len(r["num_chars"]):
        return r, None
    lab, offs, sp, tm = C.apply_alnum_letters(r["text"], r["num_chars"], r["spans"], lang)
    if tm is None:
        return r, None
    nr = dict(r, text=lab, num_chars=offs, spans=sp, alnum_policy=True)
    wt = r.get("word_times")
    if wt and len(wt) == len(tm):
        new = [None] * (max(tm) + 1)
        for i, j in enumerate(tm):
            t = wt[i]
            if t is None:
                continue
            new[j] = [t[0], t[1]] if new[j] is None else [min(new[j][0], t[0]), max(new[j][1], t[1])]
        nr["word_times"] = new
    elif wt:
        nr["word_times"] = None  # re-align before cutting crops
    return nr, tm


def relabel_tokens(w: dict, parents: dict) -> tuple:
    """A token-list row (window / seam half) through its parents' maps: (new row, changed?)."""
    if w.get("synthetic") or not w.get("src"):
        return w, False
    toks = w["text"].split()
    if len(toks) != len(w["src"]) or not any(p in parents for p, _ in w["src"]):
        return w, False
    flags = w.get("num_tokens") or [False] * len(toks)
    out_t, out_s, out_f, last, changed = [], [], [], None, False
    for t, (pid, ti), fl in zip(toks, w["src"], flags):
        if pid not in parents:
            out_t.append(t), out_s.append([pid, ti]), out_f.append(fl)
            last = None
            continue
        tm, new_toks, old_toks = parents[pid]
        j = tm[ti] if ti < len(tm) else ti
        merged_with_prev = last == (pid, j)
        if merged_with_prev:  # a second letter name of the same code
            out_f[-1] = out_f[-1] or fl
            changed = True
            continue
        if ti < len(tm) and tm.index(j) != ti:  # the window starts inside a code's letter names: leave the row
            return w, False
        if ti < len(old_toks) and j < len(new_toks) and old_toks[ti] != new_toks[j]:
            t = new_toks[j]  # the code's letters (the parent's re-labelled token)
            changed = True
        out_t.append(t), out_s.append([pid, j]), out_f.append(fl)
        changed |= j != ti
        last = (pid, j)
    if not changed:
        return w, False
    nw = dict(w, text=" ".join(out_t), src=out_s, alnum_policy=True)
    if "num_tokens" in w:
        nw["num_tokens"] = out_f
    return nw, True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--parents", nargs="+", required=True, help="row files whose ids windows / seams point at")
    ap.add_argument("--rows", nargs="*", default=[], help="row files (spans + num_chars) to re-label")
    ap.add_argument("--tokens", nargs="*", default=[], help="token-list files (windows, seam halves) to re-label")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    parents, span_read = {}, {}
    for p in a.parents:
        for r in _rows(p):
            if r["id"] in parents:
                continue
            nr, tm = relabel_row(r, a.lang)
            if tm is not None:
                parents[r["id"]] = (tm, nr["text"].split(), r["text"].split())
                for s0, s1 in zip(r["spans"], nr["spans"]):
                    if s0.get("reading") != s1.get("reading"):
                        span_read[(r["id"], s0["reading"])] = (s1["reading"], s0.get("text", ""))
    st = {"lang": a.lang, "parents_changed": len(parents), "files": {}}
    for p in a.rows:
        rows = _rows(p)
        out, ch = [], []
        for r in rows:
            nr, tm = relabel_row(r, a.lang)
            out.append(nr)
            if tm is not None:
                ch.append(nr)
        _write(a.out_dir, p, out, ch)
        st["files"][os.path.basename(p)] = {"rows": len(rows), "changed": len(ch)}
    for p in a.tokens:
        rows = _rows(p)
        out, ch = [], []
        for w in rows:
            nw, c = relabel_tokens(w, parents)
            if c and nw.get("kind") == "alnum":
                for pid in nw.get("parents") or []:
                    hit = span_read.get((pid, nw.get("reading")))
                    if hit:
                        nw = dict(nw, reading=hit[0], reading_part=_part(nw.get("reading_part") or "", hit[1], a.lang))
                        break
            out.append(nw)
            if c:
                ch.append(nw)
        _write(a.out_dir, p, out, ch)
        st["files"][os.path.basename(p)] = {"rows": len(rows), "changed": len(ch)}
    json.dump(st, open(os.path.join(a.out_dir, "alnum_relabel_stats.json"), "w"), indent=1)
    print(json.dumps(st))
    return st


def _part(rp: str, written: str, lang: str) -> str:
    """A seam half's part of an alnum reading: letter names -> the letters (alone, or before / after the number)."""
    res = C.alnum_split(rp, written, lang)
    if res:
        return res[0]
    m = C._ALNUM_LN.fullmatch(written.strip()) or C._ALNUM_ND.fullmatch(written.strip())
    if m and [w.lower() for w in rp.split()] in C._letter_forms(lang, m["l"]):
        return m["l"]
    return rp


def _write(d, p, out, ch):
    b = os.path.basename(p)
    with open(os.path.join(d, b), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    with open(os.path.join(d, "changed_" + b), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ch))


if __name__ == "__main__":
    main()
