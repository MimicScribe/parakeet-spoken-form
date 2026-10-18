"""Repair Romanian judged rows whose chosen reading breaks gender agreement (see ro_gender).

For each span whose reading is ungrammatical here, the best-ranked grammatical candidate of its `top` list is
chosen, the span and the row text are rewritten, and the span records {"gfix": {"from": old, "rank": k}}. A span
with no grammatical same-length candidate is left as it was and its row gets "gfix_flag" (with the mechanical
repair as "suggest"; the row is kept, never dropped).
Originals stay; the repaired rows are written next to them.

    python3 ro_gfix.py rows_multi1tc.jsonl [rows_ylook1.jsonl ...]   -> rows_<pass>_gfix.jsonl in the same folder
"""

import json
import os
import re
import sys

import ro_gender as G


def _locate(text: str, reading: str, hint) -> int:
    if hint and text[hint[0]:hint[1]].lower() == reading.lower():
        return hint[0]
    hits = [m.start() for m in re.finditer(re.escape(reading), text, flags=re.I)]
    if not hits:
        return -1
    return min(hits, key=lambda h: abs(h - (hint[0] if hint else 0)))


def _case(new: str, old_in_text: str) -> str:
    return new[:1].upper() + new[1:] if old_in_text[:1].isupper() else new


def fix_row(r: dict) -> dict:
    """A repaired copy of the row and a stats dict."""
    r = json.loads(json.dumps(r))
    st = {"spans_bad": 0, "spans_fixed": 0, "spans_nogood": 0, "spans_unlocated": 0}
    ncs = r.get("num_chars") or []
    aligned = len(ncs) == len(r["spans"])
    edits = []
    flags = []
    for i, s in enumerate(r["spans"]):
        hint = ncs[i] if aligned else None
        pos = _locate(r["text"], s["reading"], hint)
        before = r["text"][:pos] if pos >= 0 else ""
        after = r["text"][pos + len(s["reading"]):] if pos >= 0 else ""
        if G.grammatical(s["reading"], before, after):
            continue
        st["spans_bad"] += 1
        n_words = len(s["reading"].split())  # word_times stay valid only when the word count is unchanged
        good = next(((k, c) for k, c in enumerate(s.get("top") or [])
                     if len(c.split()) == n_words and G.grammatical(c, before, after)), None)
        if good is None:
            st["spans_nogood"] += 1
            flags.append({"span": i, "reading": s["reading"], "why": "no grammatical same-length candidate in top",
                          "suggest": G.repair(s["reading"], before, after)})
            continue
        if pos < 0:
            st["spans_unlocated"] += 1
            flags.append({"span": i, "reading": s["reading"], "why": "span text not found in row text"})
            continue
        k, new = good
        old_text = r["text"][pos:pos + len(s["reading"])]
        edits.append((pos, len(old_text), _case(new, old_text)))
        s["gfix"] = {"from": s["reading"], "rank": k}
        s["reading"] = new
        if s.get("canonical") == s["gfix"]["from"]:
            s["canonical"] = new
        st["spans_fixed"] += 1
    if edits:
        for pos, ln, new in sorted(edits, reverse=True):
            r["text"] = r["text"][:pos] + new + r["text"][pos + ln:]
        if aligned:
            r["num_chars"] = [[a + sum(len(n) - l for p, l, n in edits if p < a),
                               b + sum(len(n) - l for p, l, n in edits if p < b)] for a, b in ncs]
    if flags:
        r["gfix_flag"] = flags
    return r, st


def main(paths):
    tot = {}
    for p in paths:
        root, ext = os.path.splitext(p)
        out = []
        rows_changed = 0
        for line in open(p, encoding="utf-8"):
            r, st = fix_row(json.loads(line))
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            rows_changed += bool(st["spans_fixed"])
            out.append(json.dumps(r, ensure_ascii=False))
        open(f"{root}_gfix{ext}", "w", encoding="utf-8").write("\n".join(out) + "\n")
        print(p, "rows with a fix:", rows_changed, "of", len(out))
    print(tot)


if __name__ == "__main__":
    main(sys.argv[1:])
