"""Spell German / Dutch / Italian compound number words as separate number morphemes in training text.

"zweihundertsiebzig" -> "zwei hundert siebzig", nl "vierentwintig" -> "vier en twintig", it "duemilaventitré" ->
"due mila venti tré". German number words average ~6.7 tokenizer pieces and the training text holds ~1,500 distinct
ones, many seen once; written as morphemes the number vocabulary is a small closed set (units, teens, tens, the
hundred / thousand words, the connector, plus ordinal endings on the last morpheme).

Conventions:
  * only a token whose WHOLE letter core parses as a number word of two or more morphemes is split (leading /
    trailing punctuation stays on the first / last morpheme); anything else, and every one-morpheme number word
    ("dreizehn", "siebzig", "tweede", "trenta"), is left as it is;
  * teens and tens stay whole; the ordinal / inflection ending stays on the last morpheme ("drei und zwanzigste",
    "een en twintigste", "venti treesimo");
  * German: "und" is its own word; "eins" only last ("hundert eins"); "ein" elsewhere ("ein hundert");
  * Dutch: the connector is written "en" also where the compound has "ën" ("tweeëntwintig" -> "twee en twintig");
    join_morphemes puts the diaeresis back;
  * Italian: the elided tens + uno / otto forms stay one word ("ventuno", "trentotto", "novantotto") and so do the
    elided hundred forms "centuno", "centotto", "centottanta": splitting them ("venti uno") would spell a vowel the
    speaker does not say; the accented final "tré" / "trè" stays as written ("venti tré"); "mille" (1000) and "mila"
    (the thousands after a multiplier) are both morphemes; a unit after the elided hundred is its own word
    ("nove centottanta nove"); an elided article stays on the first morpheme ("dell'ottanta tré");
  * a final half is its own word (de "drei einhalb", nl "twee en half"); de decade / round-number nouns keep their
    ending on the last morpheme ("neunzehn hundert neunziger", "zwei tausender");
  * whole-token parse only: a number word inside a longer word ("Dreieck", "einverstanden", "eenheid", "centro",
    "Zehntausende", "zweieinhalbjährig") or joined by a hyphen stays as it is; a period name that IS a number word
    splits like any number ("Cinquecento" -> "Cinque cento").
Case is kept (a capital stays on the first morpheme); a token whose lower case changes its length is left alone.
"""

import re

# one letter per morpheme class: u unit (1-9, any position), f unit only as the LAST morpheme, z 10-19, t tens,
# e elided tens + unit (it), h hundred, c elided hundred + rest (it), k thousand, m it "mila", a connector,
# x a final half (de "einhalb", nl "half" after "en")
_DE = dict(
    u=("ein", "zwei", "zwo", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"),
    f=("eins",),
    z=("zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"),
    t=("zwanzig", "dreißig", "dreissig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"),
    h=("hundert",), k=("tausend",), a=("und",), x=("einhalb",))
_NL = dict(
    u=("een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"),
    z=("tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien", "zeventien", "achttien", "negentien"),
    t=("twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig"),
    h=("honderd",), k=("duizend",), a=("en", "ën"), x=("half",))
_IT_TENS = ("venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta")
_IT = dict(
    u=("due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"),
    f=("uno", "tré", "trè"),
    z=("dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto",
       "diciannove"),
    t=_IT_TENS,
    e=tuple(t[:-1] + s for t in _IT_TENS for s in ("uno", "otto")),  # ventuno, ventotto, trentuno, ...
    # m before k: "millesimo" is the ordinal of "mila" too ("due millesimo"; _lexicon keeps a form's first class)
    h=("cento",), c=("centuno", "centotto", "centottanta", "centottantuno", "centottantotto"), m=("mila",),
    k=("mille",))

# the grammar over the class letters (whole sequence)
_R99 = {"de": "(?:uat|t|z|u|f)", "nl": "(?:uat|t|z|u)", "it": "(?:t[uf]?|e|z|u|f)"}
_MUL = {"de": "(?:uat|t|z|u)", "nl": "(?:z|u)"}
_B1000 = {"de": f"(?:{_MUL['de']}?ha?(?:{_R99['de']})?|{_R99['de']})",
          "nl": f"(?:{_MUL['nl']}?h(?:a?{_R99['nl']})?|{_R99['nl']})",
          "it": f"(?:u?(?:h{_R99['it']}?|c[uf]?)|{_R99['it']})"}  # "centottanta nove"
_IT_MUL = "(?:u?(?:h(?:t[uf]?|e|z|u)?|c[uf]?)|t[uf]?|e|z|u)"  # a "mila" multiplier is >= 2 (never a bare "uno")
_GRAMMAR = {
    "de": re.compile(f"(?:{_B1000['de']}?ka?{_B1000['de']}?|{_B1000['de']})x?"),  # "drei einhalb"
    "nl": re.compile(f"(?:{_B1000['nl']}?ka?{_B1000['nl']}?|{_B1000['nl']})(?:ax)?"),  # "twee en half"
    "it": re.compile(f"(?:k{_B1000['it']}?|{_IT_MUL}m{_B1000['it']}?|{_B1000['it']})"),
}
# "a" (the connector) may not end a number, "f" only ends one
_DE_ORD = {"ein": "erst", "zwei": "zweit", "zwo": "zwot", "drei": "dritt", "sieben": "siebt", "acht": "acht"}
_DE_END = ("e", "en", "er", "es", "em")
_NL_ORD = {"een": "eerste", "drie": "derde", "acht": "achtste"}


def _de_finals(base, cls):
    if cls in ("t", "h", "k"):  # ordinals, and the decade / round-number nouns ("neunziger", "zweitausender")
        return [base + "st" + e for e in _DE_END] + [base + "er", base + "ern"]
    if base == "sieben":
        stems = ["siebt", "siebent"]
    else:
        stems = [_DE_ORD.get(base, base + "t")]
    return [s + e for s in stems for e in _DE_END]


def _nl_finals(base, cls):
    if cls in ("t", "h", "k") or base == "acht":
        return [base + "ste"]
    return [_NL_ORD.get(base, base + "de")]


def _it_finals(base, cls):
    if cls == "m":
        stem = "millesim"
    elif base in ("tre", "tré", "trè", "sei"):
        stem = base.replace("é", "e").replace("è", "e") + "esim"
    else:
        stem = base[:-1] + "esim"
    return [stem + e for e in "oaie"]


_FINALS = {"de": _de_finals, "nl": _nl_finals, "it": _it_finals}
_LEX_SRC = {"de": _DE, "nl": _NL, "it": _IT}


def _lexicon(lang):
    """morpheme -> (class, final_only); ordinal / inflected forms are final-only and keep their base class."""
    lex = {}
    for cls, words in _LEX_SRC[lang].items():
        for w in words:
            lex.setdefault(w, (cls, cls in ("f", "x")))
            if cls not in ("a", "x") and not (lang == "de" and cls == "f"):  # de "eins" has no ordinal (that is "erst-")
                for fw in _FINALS[lang](w, cls):
                    lex.setdefault(fw, (cls, True))
    return lex


LANGS = tuple(_LEX_SRC)
_LEX = {lang: _lexicon(lang) for lang in LANGS}
_MAXLEN = {lang: max(map(len, _LEX[lang])) for lang in LANGS}


def parse(word: str, lang: str) -> list[str] | None:
    """The morphemes of a lower-case number word (two or more), or None: not a number word, or a one-morpheme one.
    The parse with the fewest morphemes wins (teens / tens stay whole)."""
    if lang not in _LEX or not word:
        return None
    lex, gram, n = _LEX[lang], _GRAMMAR[lang], len(word)
    best = None
    # every segmentation into lexicon morphemes (number words are short; the lexicon is ~200 entries)
    stack = [(0, [])]
    while stack:
        i, parts = stack.pop()
        if i == n:
            cls = "".join(lex[p][0] for p in parts)
            if len(parts) >= 2 and gram.fullmatch(cls) and not cls.endswith("a") \
                    and all(not lex[p][1] for p in parts[:-1]) and (best is None or len(parts) < len(best)):
                best = parts
            continue
        for j in range(i + 1, min(n, i + _MAXLEN[lang]) + 1):
            if word[i:j] in lex:
                stack.append((j, parts + [word[i:j]]))
    return best


def _split_core(core: str, lang: str) -> list[str] | None:
    low = core.lower()
    if len(low) != len(core):
        return None
    parts = parse(low, lang)
    if not parts:
        return None
    out, i = [], 0
    for p in parts:
        piece = core[i:i + len(p)]
        if lang == "nl" and p == "ën":
            piece = "en" if piece == "ën" else "EN" if piece == "ËN" else piece.replace("ë", "e").replace("Ë", "E")
        out.append(piece)
        i += len(p)
    return out


def split_token(tok: str, lang: str) -> list[str]:
    """One whitespace token -> its new tokens (the token itself when it is not a compound number word)."""
    m = re.fullmatch(r"(\W*)(.*?)(\W*)", tok)
    pre, core, post = m.group(1), m.group(2), m.group(3)
    if (e := re.fullmatch(r"(\w+['’])(\w+)", core)):  # an elided article stays on the first morpheme ("dell'otto")
        pre, core = pre + e.group(1), e.group(2)
    if not core.isalpha():
        return [tok]
    parts = _split_core(core, lang)
    if not parts:
        return [tok]
    parts[0], parts[-1] = pre + parts[0], parts[-1] + post
    return parts


def join_morphemes(parts: list[str], lang: str) -> str:
    """Inverse of split_token on a correctly spelled word: the morphemes joined (nl: "ën" after an e-final unit)."""
    out = ""
    for p in parts:
        if lang == "nl" and p.lower() == "en" and out.lower().endswith("e"):
            p = "ën" if p == "en" else "ËN"
        out += p
    return out


def split_text(text: str, lang: str) -> tuple[str, list[list[int]], list[int]]:
    """(new text, per original whitespace token the list of its new token indices, insertion points). An insertion
    point q = a space was inserted before ORIGINAL character q. Whitespace between tokens is kept as it is."""
    out, tok_map, ins, last, k = [], [], [], 0, 0
    for m in re.finditer(r"\S+", text):
        out.append(text[last:m.start()])
        parts = split_token(m.group(0), lang)
        tok_map.append(list(range(k, k + len(parts))))
        k += len(parts)
        pos = m.start()
        for j, p in enumerate(parts):
            if j:
                ins.append(pos)
                out.append(" ")
            out.append(p)
            pos += len(p)
        last = m.end()
    out.append(text[last:])
    return "".join(out), tok_map, ins


def _map_span(a: int, b: int, ins: list[int]) -> list[int]:
    return [a + sum(1 for q in ins if q <= a), b + sum(1 for q in ins if q < b)]


def _expand(xs, tok_map):
    return [xs[i] for i, ks in enumerate(tok_map) for _ in ks]


def _split_times(times, tok_map, new_toks):
    out = []
    for tt, ks in zip(times, tok_map):
        if len(ks) == 1 or not tt:  # an untimed word (None) stays untimed
            out += [tt] * len(ks)
            continue
        t0, t1 = tt
        lens = [max(1, sum(c.isalpha() for c in new_toks[k])) for k in ks]
        tot, acc = sum(lens), 0
        for n_ in lens:
            s = t0 + (t1 - t0) * acc / tot
            acc += n_
            out.append([round(s, 3), round(t0 + (t1 - t0) * acc / tot, 3)])
    return out


def split_row(row: dict, lang: str) -> dict:
    """A manifest row with its label in the split convention: `text`, the reading offsets `num_chars`, the per-token
    lists `num_tokens` / `word_times` / `src` (a split token's entries repeat; word_times split by letter count; src
    keeps the parent's UNSPLIT word index), the readings (`reading`, `reading_part`, spans[].reading). `caption` and
    spans[].canonical / text (the written value) stay. The row itself when nothing splits."""
    text = row.get("text")
    if not isinstance(text, str):
        return row
    new, tok_map, ins = split_text(text, lang)
    rd = {k: split_text(row[k], lang)[0] for k in ("reading", "reading_part") if isinstance(row.get(k), str)}
    sp_changed = any(isinstance(sp, dict) and isinstance(sp.get("reading"), str)
                     and split_text(sp["reading"], lang)[0] != sp["reading"] for sp in row.get("spans") or [])
    if new == text and all(rd[k] == row[k] for k in rd) and not sp_changed:
        return row
    out = dict(row, text=new, **rd)
    n = len(tok_map)
    new_toks = new.split()
    if "num_chars" in row:
        out["num_chars"] = [_map_span(a, b, ins) for a, b in row["num_chars"]]
    for k in ("num_tokens", "src"):
        if isinstance(row.get(k), list) and len(row[k]) == n:
            out[k] = _expand(row[k], tok_map)
    if isinstance(row.get("word_times"), list) and len(row["word_times"]) == n:
        out["word_times"] = _split_times(row["word_times"], tok_map, new_toks)
    if row.get("spans"):
        out["spans"] = [dict(sp, reading=split_text(sp["reading"], lang)[0])
                        if isinstance(sp, dict) and isinstance(sp.get("reading"), str) else sp for sp in row["spans"]]
    return out


def split_pair(p: dict, lang: str) -> dict:
    """A consistency pair (ft_train.load_pairs format) in the split convention: both texts, both token spans, the
    reading."""
    out = dict(p, reading=split_text(p["reading"], lang)[0])
    for side in ("a", "b"):
        new, tok_map, _ = split_text(p[side]["text"], lang)
        i0, i1 = p[side + "_span"]
        out[side] = dict(p[side], text=new)
        out[side + "_span"] = [tok_map[i0][0], tok_map[i1 - 1][-1] + 1]
    return out


def split_rows(rows: list[dict], lang: str, pairs: bool = False) -> tuple[list[dict], dict]:
    """(split rows, counts {rows, rows_changed, tokens_split}); the input rows are not modified."""
    f = split_pair if pairs else split_row
    out, ch, toks = [], 0, 0
    for r in rows:
        x = f(r, lang)
        if x is not r:
            ch += 1
            t0 = r["a"]["text"] if pairs else r["text"]
            toks += sum(1 for w in t0.split() if len(split_token(w, lang)) > 1)
        out.append(x)
    return out, {"rows": len(rows), "rows_changed": ch, "tokens_split": toks}
