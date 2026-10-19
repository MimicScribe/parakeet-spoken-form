"""Residual letter-name labels in rows that alnum_relabel.py cannot reach (2026-10-19).

The letter+digit policy writes a code's letters as the letters and its number as spoken words ("ge zwanzig" ->
"G zwanzig", "ce o zwei" -> "CO zwei", "fünf ge" -> "fünf G"). alnum_relabel.py does it through span records and the
parent token maps. Three kinds of row slip past it: A/B pair rows (pairs_<lang>: texts only, no spans), windows
that start inside a code's letter names (the "leave the row" case) and rows whose parent was never listed. This module
works on the text alone and is deliberately narrow:

  * a code's letter-name sequence (every spoken spelling in ml_cands.LETTERS) must sit directly before a number word
    (letters + number: "ge zwanzig") or directly after one (number + letter: "fünf ge");
  * a single-token name that is also an everyday word (es "ve", pt "dê", fr "dé", nl "se", ...) is left alone
    unless the number comes first and is not "one" ("cinq dé" is 5D, "un dé" is a die, "dê um tempo" is a verb);
  * a name written in capitals is an acronym already ("EM zwei tausend zwölf" stays).

    python3 alnum_residual.py --lang de --rows pairs_de.jsonl win_de.jsonl --out-dir out/de

Writes <out-dir>/rows_alnumB.jsonl (changed rows only: the original row with the new text, "text_old" and
"relabel": ["alnumB"]) and alnum_residual_stats.json. A relabelled token-list row keeps its parallel lists in step
(num_tokens by OR, src of the first merged token, word_times as one envelope); a pair row's token spans are
re-pointed. Pure Python; no audio is touched.
"""

import argparse
import json
import os
import re
import string

import ml_cands as M

# letter strings tried as codes: every single letter plus the multi-letter codes the policy has met
LETTER_CODES = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["CO", "SO", "COP", "MH", "DS", "SS", "BB", "BG", "PS", "MP", "DB", "AC", "FY"]

# a single-token letter name that is also an everyday word of the language: not taken before a number
AMBIGUOUS = {
    "es": {"a", "e", "o", "u", "i", "de", "te", "ve", "ese", "el"},
    "pt": {"a", "e", "o", "u", "é", "ê", "dê", "vê", "ó", "ele", "cá", "quê", "esse"},
    "fr": {"a", "e", "o", "u", "i", "dé", "elle"},
    "it": {"a", "e", "i", "o", "u", "di", "ci", "pi", "ti", "vu"},
    "de": {"a", "e", "i", "o", "u", "es", "er"},
    "nl": {"a", "e", "i", "o", "u", "es", "en", "er", "de", "je", "we"},
}
# ... except number-first ("três dê", "cinq dé") when the name ends a clause or a capitalised word follows
ND_EDGE = {"pt": {"dê"}, "fr": {"dé"}}
# "one" in front of a letter name is an article ("un dé", "ein ef"): never a code
ONE = {"un", "une", "uno", "una", "um", "uma", "een", "ein", "eine", "één", "eins"}
# a code number can also be said digit by digit or as a short compound; these join the cardinals as number words
EXTRA_NUM = {"zero", "null", "zéro", "cero", "zero", "nul"}

_CACHE: dict = {}


def _strip(tok: str) -> tuple:
    """(leading punctuation, core, trailing punctuation) of one whitespace token."""
    a = len(tok) - len(tok.lstrip(string.punctuation + "«»„“”‘’¿¡"))
    b = len(tok.rstrip(string.punctuation + "«»„“”‘’¿¡"))
    if b <= a:
        return tok, "", ""
    return tok[:a], tok[a:b], tok[b:]


def _num_words(lang: str) -> set:
    s = set(EXTRA_NUM)
    for n in range(0, 1001):
        try:
            for w in M.g(lang, "card", n):
                s.add(w.lower())
        except Exception:
            pass
    return s


def _tables(lang: str) -> tuple:
    """(names: {tuple of lower-case name tokens: letters}, longest name length, number words)."""
    if lang in _CACHE:
        return _CACHE[lang]
    names: dict = {}
    for L in LETTER_CODES:
        for seq in M._spell(lang, L, k=3):
            t = tuple(seq)
            if not t or t == (L.lower(),):
                continue  # the letter as one lower-case word is not a name
            if t in names and names[t] != L:
                names[t] = None  # two codes spell the same: ambiguous, never rewritten
            else:
                names[t] = L
    names = {t: L for t, L in names.items() if L}
    _CACHE[lang] = (names, max(len(t) for t in names), _num_words(lang))
    return _CACHE[lang]


def _name_ok(core: str) -> bool:
    return core == core.lower() or core == core.capitalize()


def _edge(parts: list, k: int) -> bool:
    """The token at k ends a clause (punctuation after it), ends the text, or a capitalised word follows."""
    if parts[k][2] or k + 1 >= len(parts):
        return True
    nxt = parts[k + 1][1]
    return bool(nxt) and nxt[0].isupper()


def fix_tokens(tokens: list, lang: str) -> tuple:
    """(new tokens, token map, n fixes). token map[i] = index of old token i in the new list (a code's name tokens
    share one new token); None when nothing changed. Tokens keep their attached punctuation."""
    if lang not in M.LETTERS:
        return tokens, None, 0
    names, longest, nums = _tables(lang)
    parts = [_strip(t) for t in tokens]
    low = [p[1].lower() for p in parts]
    out, tmap, i, fixes = [], [], 0, 0
    amb = AMBIGUOUS.get(lang, set())
    while i < len(tokens):
        hit = None
        for m in range(min(longest, len(tokens) - i), 0, -1):
            t = tuple(low[i:i + m])
            L = names.get(t)
            if not L or not all(_name_ok(parts[i + k][1]) for k in range(m)):
                continue
            nxt = low[i + m] if i + m < len(tokens) else ""
            prv = low[i - 1] if i > 0 else ""
            core = t[:-1] if m > 1 and t[-1] in ("o", "ó", "oh") else t  # "ce o zwei": the O is part of the code CO
            single_amb = m == 1 and t[0] in amb
            if t == ("se",) and prv == "per":  # Latin "per se"
                continue
            if all(w in amb for w in core):  # "de ese diecisiete": every name token is an everyday word
                single_amb = True
                if m > 1:
                    continue
            letters_first = nxt in nums and not single_amb
            number_first = prv in nums and prv not in ONE and L in ("D", "G", "K") and (
                not single_amb or (t[0] in ND_EDGE.get(lang, ()) and _edge(parts, i + m - 1)))
            if letters_first or number_first:
                hit = (m, L)
                break
        if hit is None:
            out.append(tokens[i])
            tmap.append(len(out) - 1)
            i += 1
            continue
        m, L = hit
        lead, trail = parts[i][0], parts[i + m - 1][2]
        out.append(lead + L + trail)
        tmap.extend([len(out) - 1] * m)
        i += m
        fixes += 1
    if not fixes:
        return tokens, None, 0
    return out, tmap, fixes


def fix_text(text: str, lang: str) -> tuple:
    """(new text, token map or None). Whitespace is normalised to single spaces only when a fix applies."""
    toks = text.split()
    new, tmap, n = fix_tokens(toks, lang)
    if not n:
        return text, None
    return " ".join(new), tmap


def audit(text: str, lang: str) -> list:
    """The letter-name sequences a label still holds, as 'name tokens' strings (empty = clean). For a post-build check."""
    toks = text.split()
    new, tmap, n = fix_tokens(toks, lang)
    if not n:
        return []
    found, last = [], None
    for i, j in enumerate(tmap):
        if j != last and sum(1 for k in tmap if k == j) > 1:
            found.append(" ".join(t for t, k in zip(toks, tmap) if k == j))
        last = j
    return found


def _merge(vals: list, tmap: list, how: str) -> list:
    out: list = [None] * (max(tmap) + 1)
    for v, j in zip(vals, tmap):
        cur = out[j]
        if cur is None:
            out[j] = v
        elif how == "or":
            out[j] = cur or v
        elif how == "span" and v is not None and cur is not None:
            out[j] = [min(cur[0], v[0]), max(cur[1], v[1])]
    return out


def _span(span: list, tmap: list) -> list:
    return [tmap[span[0]], tmap[span[1] - 1] + 1]


def relabel_row(r: dict, lang: str) -> tuple:
    """(new row, True) when a fix applied. Plain rows ("text"), token-list rows (num_tokens / src / word_times kept in
    step) and pair rows ({"a", "b", "a_span", "b_span"})."""
    if r.get("synthetic"):
        return r, False
    if isinstance(r.get("a"), dict) and isinstance(r.get("b"), dict):
        na, ta = fix_text(r["a"]["text"], lang)
        nb, tb = fix_text(r["b"]["text"], lang)
        if ta is None and tb is None:
            return r, False
        nr = dict(r, a=dict(r["a"], text=na), b=dict(r["b"], text=nb), alnum_policy=True, relabel=["alnumB"])
        if ta is not None and r.get("a_span"):
            nr["a_span"] = _span(r["a_span"], ta)
        if tb is not None and r.get("b_span"):
            nr["b_span"] = _span(r["b_span"], tb)
        return nr, True
    text = r.get("text")
    if not isinstance(text, str):
        return r, False
    new, tm = fix_text(text, lang)
    if tm is None:
        return r, False
    if r.get("spans") and r.get("num_chars"):
        return r, False  # span rows belong to alnum_relabel.py (their offsets are character-based)
    nr = dict(r, text=new, text_old=text, alnum_policy=True, relabel=["alnumB"])
    n_old = len(text.split())
    for k, how in (("num_tokens", "or"), ("src", "first"), ("word_times", "span")):
        v = r.get(k)
        if isinstance(v, list) and len(v) == n_old:
            nr[k] = _merge(v, tm, how)
    return nr, True


def _rows(p: str) -> list:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    st = {"lang": a.lang, "files": {}}
    changed = []
    for p in a.rows:
        rows = _rows(p)
        ch = []
        for r in rows:
            nr, c = relabel_row(r, a.lang)
            if c:
                ch.append(dict(nr, part=os.path.basename(p)))
        st["files"][os.path.basename(p)] = {"rows": len(rows), "changed": len(ch)}
        changed += ch
    with open(os.path.join(a.out_dir, "rows_alnumB.jsonl"), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in changed))
    json.dump(st, open(os.path.join(a.out_dir, "alnum_residual_stats.json"), "w"), indent=1)
    print(json.dumps(st))
    return st


if __name__ == "__main__":
    main()
