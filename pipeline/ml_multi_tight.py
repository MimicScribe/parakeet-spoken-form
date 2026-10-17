"""Tighten the multilingual parliament rows against an external transcript, without dropping clips.

A training label must never lose a meaningful spoken word, contain a word that was not spoken, or change a number
value. Each row is compared word by word with an independent transcript of the same audio. Where they disagree the
row is cut AROUND the problem: every maximal run of label words that the transcript confirms becomes a row of its
own (origin "tight"); the disagreeing words are the only audio left out. Number readings are value-checked upstream,
so they count as matching but are never cut through.

    python3 ml_multi_tight.py <rows.jsonl> <ext.jsonl> <out.jsonl> [--examples N]

Rows: rows_multi1 format. Transcript lines: {"id", "text", "words": [{"w", "s"/"start", "e"/"end" (ms)}]}.
"""
import collections
import difflib
import json
import re
import sys

import ml_multi as MM
from ml_multi_scale import MIN_PIECE_S

MIN_WORDS = 3
NUM = "\x00num"  # placeholder key of a number reading (label side) / a digit word (transcript side)
TAIL = 0.08  # seconds kept beside a cut that has no timed neighbour to bound it


# ------------------------------------------------------------------ tokens and number atoms
def tokenize(text: str) -> list:
    """[(token, start, end)] over whitespace-separated tokens."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _atoms(row: dict, toks: list):
    """Number atoms: [(first token, last token, [span indices])], or None when the spans cannot be mapped onto
    the label tokens (the row is then left as it is). An atom is the token extent of one span's reading."""
    ranges = [tuple(r) for r in row.get("num_chars") or []]
    in_num = [i for i, (_t, a, b) in enumerate(toks) if any(a < rb and b > ra for ra, rb in ranges)]
    spans = row.get("spans") or []
    keys = [MM._key(t) for t, _a, _b in toks]
    atoms, cur, covered = [], 0, set()
    for si, sp in enumerate(spans):
        words = [MM._key(w) for w in (sp.get("reading") or sp.get("canonical") or "").split()]
        words = [w for w in words if w]
        found = None
        if words:
            for s in range(cur, len(toks) - len(words) + 1):
                if keys[s:s + len(words)] == words and all(i in in_num for i in range(s, s + len(words))):
                    found = (s, s + len(words) - 1)
                    break
        if found is None:
            return None
        atoms.append((found[0], found[1], [si]))
        covered.update(range(found[0], found[1] + 1))
        cur = found[1] + 1
    if set(in_num) - covered:
        return None
    return atoms


# ------------------------------------------------------------------ alignment
def _same(x: str, y: str, abbr=frozenset()) -> bool:
    """Label word `x` and transcript word `y` agree: `_near`, or `y` abbreviates `x` ("godz." / "godziny", "ust" /
    "ustępu": a prefix of 3+ letters, or a full-stop abbreviation; "mln" / "milionów": a vowelless subsequence)."""
    if _near(x, y):
        return True
    if len(x) < len(y) + 2 or len(y) < 2:
        return False
    if x.startswith(y) and (len(y) >= 3 or y in abbr):
        return True
    if len(y) >= 3 and not re.search(r"[aeiouy]", y) and x[0] == y[0]:
        it = iter(x[1:])
        return all(c in it for c in y[1:])
    return False


def _unit(y: str, k: str) -> bool:
    """Transcript word `y` is the unit word of a number reading whose label word is `k` ("lata" / "lat", "mln" /
    "milionów"): such a reading was checked as a whole."""
    return _same(k, y) or (min(len(y), len(k)) >= 3 and y[:3] == k[:3] and abs(len(y) - len(k)) <= 2)


def _near(a: str, b: str) -> bool:
    """Same word up to a case / inflection ending (len >= 5; one is the other minus at most two letters)."""
    if a == b:
        return True
    s, l = sorted((a, b), key=len)
    return len(s) >= 5 and len(l) - len(s) <= 4 and l.startswith(s[:-2])


def _match_block(a: list, b: list, gap_end: int, forgive: bool, bad: set, lost: dict, numword: set, abbr=frozenset(), numkeys=frozenset()):
    """Pair up a differing block of (token index, key) lists: near matches and split / joined compounds agree,
    the rest disagrees. A label word not heard goes to `bad`; a transcript word with no label counterpart goes to
    `lost[gap]` (gap g sits before label token g) unless `forgive` (spelled-out numbers beside a number reading)."""
    i = j = 0

    def unit(y):  # a word of the neighbouring number reading itself ("milionów", "lat"): not a lost word
        return any(_unit(y, k) for k in numkeys)

    while i < len(a) or j < len(b):
        gap = a[i][0] if i < len(a) else gap_end
        if i >= len(a):
            if not forgive and b[j][1] != NUM and not unit(b[j][1]):
                lost.setdefault(gap, []).append(b[j][0])
            j += 1
        elif j >= len(b):
            if a[i][1] != NUM:
                bad.add(a[i][0])
            i += 1
        else:
            x, y = a[i][1], b[j][1]
            if x == NUM and y == NUM:
                i += 1
                j += 1
            elif x == NUM:
                i += 1
            elif y == NUM:
                # a digit word heard where the label has an unmarked word: a spelled number the label's
                # number readings do not cover; its value cannot be confirmed, so it is cut around
                bad.add(a[i][0])
                numword.add(a[i][0])
                i += 1
                nxt = b[j + 1][1] if j + 1 < len(b) else ""
                if i >= len(a) or a[i][1] == NUM or _same(a[i][1], nxt, abbr):
                    j += 1  # the digit word is used up (a multi-word spelled number keeps it for its next word)
            elif _near(x, y):
                i += 1
                j += 1
            elif i + 1 < len(a) and a[i + 1][1] != NUM and x + a[i + 1][1] == y:
                i += 2
                j += 1
            elif j + 1 < len(b) and b[j + 1][1] != NUM and y + b[j + 1][1] == x:
                i += 1
                j += 2
            elif _same(x, y, abbr):
                i += 1
                j += 1
            elif j + 1 < len(b) and _same(x, b[j + 1][1], abbr):
                if not forgive and not unit(b[j][1]):
                    lost.setdefault(gap, []).append(b[j][0])
                j += 1
            elif i + 1 < len(a) and _same(a[i + 1][1], y, abbr):
                bad.add(a[i][0])
                i += 1
            else:
                bad.add(a[i][0])
                if not forgive and not unit(b[j][1]):
                    lost.setdefault(gap, []).append(b[j][0])
                i += 1
                j += 1


def align(toks: list, atoms: list, ext_words: list) -> dict:
    """Compare label words with transcript words. Returns {"bad": label token indices not heard, "lost":
    {gap: [transcript word indices]}, "compared": number of non-number label words compared}."""
    atom_of = {i: n for n, (s, e, _sp) in enumerate(atoms) for i in range(s, e + 1)}
    atom_span = {s: (s, e + 1) for s, e, _sp in atoms}
    a, seen = [], set()
    for i, (t, _s, _e) in enumerate(toks):
        if i in atom_of:
            if atom_of[i] not in seen:
                seen.add(atom_of[i])
                a.append((i, NUM))
            continue
        k = MM._key(t)
        if k:
            a.append((i, k))
    b, abbr = [], set()
    for j, w in enumerate(ext_words):
        k = MM._key(w["w"])
        if not k:
            continue
        if re.search(r"\d", k):
            # "COP21", "CO2", "CO₂": the letters before the digits are a word of their own, then the number
            m = re.match(r"([^\W\d_]+)\d", k)
            if m:
                b.append((j, m.group(1)))
            if not b or b[-1][1] != NUM:
                b.append((j, NUM))
        else:
            b.append((j, k))
            if w["w"].rstrip().endswith(".") and len(k) >= 2:
                abbr.add(k)
    bad, lost, numword = set(), {}, set()
    sm = difflib.SequenceMatcher(None, [x[1] for x in a], [x[1] for x in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # transcript words spelled out where the label has a number reading belong to that (checked) reading
        near_num = any(k == NUM for _i, k in a[i1:i2])
        gap_end = a[i2][0] if i2 < len(a) else len(toks)
        numkeys = {MM._key(toks[t][0]) for _i, k in a[max(0, i1 - 1):i2 + 1] if k == NUM for t in range(*atom_span[_i])}
        _match_block(a[i1:i2], b[j1:j2], gap_end, near_num, bad, lost, numword, abbr, numkeys)
    return {"bad": bad, "numword": numword, "lost": lost, "compared": sum(1 for _i, k in a if k != NUM)}


# ------------------------------------------------------------------ timing
def _ms(w: dict, long_key: str, short_key: str):
    v = w.get(short_key, w.get(long_key))
    return None if v is None else v / 1000.0


def ext_times(w: dict):
    """(start, end) seconds of a transcript word, each None when absent."""
    return _ms(w, "start", "s"), _ms(w, "end", "e")


# ------------------------------------------------------------------ runs and rows
def _runs(n: int, bad: set, lost: dict, atoms: list) -> list:
    """Maximal [i0, i1] token ranges with no bad token and no lost gap inside; atoms are never split."""
    atom_of = {i: (s, e) for s, e, _sp in atoms for i in range(s, e + 1)}
    runs, cur = [], None
    for i in range(n):
        if i in bad:
            if cur is not None:
                runs.append(cur)
            cur = None
            continue
        if cur is not None and i in lost and not (i in atom_of and atom_of[i][0] < i):
            runs.append(cur)
            cur = None
        cur = [i, i] if cur is None else [cur[0], i]
    if cur is not None:
        runs.append(cur)
    return runs


def _lost_t(ext_words: list, lost: dict, gap: int, want_end: bool):
    """Latest end (want_end) / earliest start of the timed lost transcript words in a gap, else None."""
    ts = [ext_times(ext_words[j]) for j in lost.get(gap, [])]
    ts = [t[1] if want_end else t[0] for t in ts]
    ts = [t for t in ts if t is not None]
    return (max(ts) if want_end else min(ts)) if ts else None


def _clean_edges(run: list, lost: dict, atoms: list, wt: list, ext_words: list, strict: bool = False):
    """A number reading at a run edge stays whole and the disagreeing word beside it is cut off, since a word
    boundary separates them. It is given up only when the disagreement is glued to it: a lost transcript word
    whose audio overlaps the number's edge word (`strict` gives up every edge number beside a disagreement).
    Returns (shortened run or None, [atom indices given up])."""
    atom_at = {i: n for n, (s, e, _sp) in enumerate(atoms) for i in range(s, e + 1)}
    i0, i1, gave = run[0], run[1], []

    def glued(i, left):
        gap = i if left else i + 1
        if gap not in lost:
            return False
        if strict:
            return True
        lt = _lost_t(ext_words, lost, gap, left) if wt[i] is not None else None
        if lt is None:
            return False
        mid = (wt[i][0] + wt[i][1]) / 2  # overlapping the number's edge word by more than half
        return lt > mid if left else lt < mid

    while i0 <= i1 and i0 in atom_at and glued(i0, True):
        n = atom_at[i0]
        gave.append(n)
        i0 = atoms[n][1] + 1
    while i1 >= i0 and i1 in atom_at and glued(i1, False):
        n = atom_at[i1]
        gave.append(n)
        i1 = atoms[n][0] - 1
    return ([i0, i1] if i0 <= i1 else None), gave


def _build(row: dict, toks: list, atoms: list, run: list, k: int, rel0: float, rel1: float) -> dict:
    i0, i1 = run
    new_text, off = "", {}
    for i in range(i0, i1 + 1):
        t, a, _b = toks[i]
        if i > i0:
            new_text += " "
        off[i] = (len(new_text), a)
        new_text += t
    spans, nums = [], []
    for s, e, sps in atoms:
        if s >= i0 and e <= i1:
            spans += [row["spans"][x] for x in sps]
    for ra, rb in row.get("num_chars") or []:
        for i in range(i0, i1 + 1):
            if toks[i][1] <= ra < toks[i][2]:
                j = next(j for j in range(i, i1 + 1) if toks[j][1] < rb <= toks[j][2])
                nums.append([off[i][0] + ra - toks[i][1], off[j][0] + rb - toks[j][1]])
                break
    wt = row.get("word_times") or []
    new_wt = [None if (x is None or x is False) else [round(x[0] - rel0, 3), round(x[1] - rel0, 3)]
              for x in wt[i0:i1 + 1]]
    base_t0 = row.get("t0", 0.0)
    out = dict(row, id=f"{row['id']}_t{k}", origin="tight", parent=row["id"], duration=round(rel1 - rel0, 3),
               text=new_text, spans=spans, num_chars=nums, word_times=new_wt, t0=round(base_t0 + rel0, 3),
               t1=round(base_t0 + rel1, 3), crop_s=[round(rel0, 3), round(rel1, 3)], cer_ext=0.0,
               ext_scope="tight", plain=not spans)
    return out


def tighten_row(row: dict, ext: dict, strict: bool = False):
    """-> (rows, info). rows is [row] when it needs no change, else the confirmed runs cut out of it."""
    info = {"lang": row.get("lang", "?"), "status": "unchanged", "compared": 0, "invented": 0, "lost": 0, "number_words_unverified": 0,
            "left_out": collections.Counter(), "numbers_total": 0, "numbers_kept": 0,
            "number_reasons": collections.Counter(), "number_cases": []}
    ext_words = (ext or {}).get("words") or []
    toks = tokenize(row["text"])
    if not ext_words or not toks:
        info["status"] = "no_ext"
        return [row], info
    atoms = _atoms(row, toks)
    if atoms is None:
        info["status"] = "unmapped_numbers"
        return [row], info
    al = align(toks, atoms, ext_words)
    bad, lost = al["bad"], al["lost"]
    info["compared"] = al["compared"]
    info["numbers_total"] = len(atoms)
    info["invented"] = len(bad - al["numword"])
    info["number_words_unverified"] = len(al["numword"])
    info["lost"] = sum(len(v) for v in lost.values())
    n = len(toks)
    edge_lost = 0 in lost or n in lost
    if not bad and not lost:
        info["numbers_kept"] = len(atoms)
        return [row], info
    reason = {}
    wt = row.get("word_times") or []
    wt = list(wt) + [None] * (n - len(wt))
    dur = row["duration"]
    out, kept_s, k = [], 0.0, 0
    for run in _runs(n, bad, lost, atoms):
        whole = run
        run, gave = _clean_edges(run, lost, atoms, wt + [None], ext_words, strict)
        for ai in gave:
            reason[ai] = "glued_to_lost_word"
        if run is None:
            continue
        i0, i1 = run
        inside = [ai for ai, (s_, e_, _sp) in enumerate(atoms) if s_ >= i0 and e_ <= i1]
        words = sum(1 for t, _a, _b in toks[i0:i1 + 1] if MM._key(t))
        # boundaries: first word start, previous word end, last word end, next word start
        need = [wt[i0], wt[i1]] + ([wt[i0 - 1]] if i0 > 0 else []) + ([wt[i1 + 1]] if i1 + 1 < n else [])
        if any(x is None for x in need):
            info["left_out"]["no_word_times"] += 1
            reason.update({ai: "run_without_word_times" for ai in inside})
            continue
        prev_end = wt[i0 - 1][1] if i0 > 0 else 0.0
        next_start = wt[i1 + 1][0] if i1 + 1 < n else dur

        first_s, last_e = wt[i0][0], wt[i1][1]
        if i0 in lost:  # a lost word sits right before the run: stay clear of its audio
            lt = _lost_t(ext_words, lost, i0, True)
            if lt is not None:
                rel0 = round((first_s + min(first_s, max(prev_end, lt))) / 2, 3)
            else:
                rel0 = round(max(prev_end, first_s - min(TAIL, (first_s - prev_end) / 2)), 3)
        else:
            rel0 = round((prev_end + first_s) / 2, 3) if i0 > 0 else 0.0
        if i1 + 1 in lost:
            lt = _lost_t(ext_words, lost, i1 + 1, False)
            if lt is not None:
                hi = max(last_e, min(next_start, lt))
                rel1 = round((last_e + hi) / 2, 3)
            else:
                rel1 = round(min(next_start, last_e + min(TAIL, (next_start - last_e) / 2)), 3)
        else:
            rel1 = round((last_e + next_start) / 2, 3) if i1 + 1 < n else round(dur, 3)
        rel0, rel1 = max(0.0, min(rel0, first_s)), min(dur, max(rel1, last_e))
        if rel1 - rel0 < MIN_PIECE_S:
            info["left_out"]["too_short"] += 1
            reason.update({ai: "run_too_short" for ai in inside})
            continue
        if words < MIN_WORDS:
            info["left_out"]["too_few_words"] += 1
            reason.update({ai: "run_under_3_words" for ai in inside})
            continue
        out.append(_build(row, toks, atoms, run, k, rel0, rel1))
        kept_s += rel1 - rel0
        k += 1
        info["numbers_kept"] += len(inside)
    one_run = len(out) == 1 and len(_runs(n, bad, lost, atoms)) == 1 and not bad
    info["status"] = "trimmed" if one_run else "split"
    info["kept_s"] = kept_s
    info["number_reasons"].update(reason.values())
    for ai, why in reason.items():
        info["number_cases"].append((why, " ".join(t for t, _a, _b in toks[atoms[ai][0]:atoms[ai][1] + 1]), row["text"]))
    if not out:
        info["status"] = "emptied"
    return out, info


# ------------------------------------------------------------------ files and stats
def load_ext(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d
    return out


def tighten(rows: list, ext: dict, strict: bool = False):
    """-> (out rows, per-language stats, examples [(before, [after...])], number cases [(reason, number words, text)])."""
    stats, out, examples = {}, [], []
    number_cases = []
    for row in rows:
        lang = row.get("lang", "?")
        st = stats.setdefault(lang, collections.Counter())
        e = ext.get(row["id"])
        new, info = tighten_row(row, e, strict)
        st["rows_in"] += 1
        st["hours_in"] += row["duration"] / 3600
        st["words_compared"] += info["compared"]
        st["invented"] += info["invented"]
        st["number_words_unverified"] += info["number_words_unverified"]
        st["lost"] += info["lost"]
        if info["status"] == "emptied":  # the row is kept as it is, so its numbers stay
            info["numbers_kept"], info["number_reasons"], info["number_cases"] = info["numbers_total"], {}, []
        st["numbers_total"] += info["numbers_total"]
        st["numbers_kept"] += info["numbers_kept"]
        st["numbers_given_up"] += info["numbers_total"] - info["numbers_kept"]
        for why, c in info["number_reasons"].items():
            st["numbers_given_up_" + why] += c
        number_cases.extend(info["number_cases"])
        if info["status"] in ("unchanged", "no_ext", "unmapped_numbers"):
            st["rows_unchanged"] += 1
            if info["status"] != "unchanged":
                st["rows_" + info["status"]] += 1
            st["rows_out"] += 1
            st["hours_out"] += row["duration"] / 3600
            out.append(row)
            continue
        if info["status"] == "emptied":
            # a row with no confirmed run is kept as it is rather than dropped; it is flagged for review
            st["rows_unchanged"] += 1
            st["rows_emptied_kept"] += 1
            st["rows_out"] += 1
            st["hours_out"] += row["duration"] / 3600
            out.append(dict(row, tight_failed=True))
            continue
        st["rows_" + info["status"]] += 1
        st["rows_out"] += len(new)
        st["hours_out"] += sum(r["duration"] for r in new) / 3600
        for why, c in info["left_out"].items():
            st["segments_left_out_" + why] += c
        st["hours_cut_around_disagreements"] += (row["duration"] - info["kept_s"]) / 3600
        st["plain_rows"] += sum(1 for r in new if r["plain"])
        out += new
        if len(examples) < 1000:
            examples.append((row["text"], [r["text"] for r in new]))
    res = {}
    for lang, st in stats.items():
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(st.items())}
        d["hours_left_out"] = round(st["hours_in"] - st["hours_out"], 3)
        res[lang] = d
    return out, res, examples, number_cases


def main(argv: list):
    args = [a for a in argv if not a.startswith("--")]
    n_ex = int(argv[argv.index("--examples") + 1]) if "--examples" in argv else 0
    args = [a for a in args if a != str(n_ex) or "--examples" not in argv]
    if len(args) != 3:
        sys.exit(__doc__)
    with open(args[0], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out, stats, examples, cases = tighten(rows, load_ext(args[1]), strict="--strict" in argv)
    with open(args[2], "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    for why, num, text in cases[:n_ex]:
        print(f"\nNUMBER GIVEN UP ({why}): {num} | {text}")
    for before, after in examples[:n_ex]:
        print("\nBEFORE:", before)
        for a in after:
            print("  AFTER:", a)


if __name__ == "__main__":
    main(sys.argv[1:])
