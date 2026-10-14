"""Tier-1 multilingual spoken-form candidates (R3.11 P1): es, fr, de, pt, it, nl.

Given caption text with digits and a language, find the number spans and list the spoken readings a speaker may
have used. The CANONICAL reading of every value comes from per-language number grammars, pre-dumped into
`ml_grammar_table.json` by a grammar dumper that is not part of this repo. Variants (gender/apocope on 1, 2 and hundreds; "tausend" without "ein"; the hundreds reading of
1100-1999; ordinal case/gender endings; a decimal fraction read as one number; digit-by-digit codes) are derived from
the canonical words in the grammar's own spelling style. A value missing from the table yields NO candidates: the span
is then undecidable and never becomes a label.

Two passes:
    reqs = needed(texts, lang)          # every grammar call the spans of these texts need
    (run the Swift dumper on reqs)      # build_table(...) below writes requests.json and reads answers.json
    spans(text, lang)                   # Span(start, end, text, kind, cands) with cands[0] = canonical
"""

import json
import os
import re
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "ml_grammar_table.json")
LANGS = ("es", "fr", "de", "pt", "it", "nl")
_TABLE: dict | None = None


def _key(lang, kind, n=0, f="", s="", h=0, m=0) -> str:
    return f"{lang}|{kind}|{n}|{f}|{s}|{h}|{m}"


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.load(open(TABLE_PATH)) if os.path.exists(TABLE_PATH) else {}
    return _TABLE


class Missing(KeyError):
    pass


class _Recorder:
    """Stands in for the table while collecting requests: records every key, answers a placeholder."""

    def __init__(self):
        self.keys = set()

    def get(self, k):
        self.keys.add(k)
        return ["_"]


_REC: _Recorder | None = None


def g(lang, kind, n=0, f="", s="", h=0, m=0) -> list[str]:
    k = _key(lang, kind, n, f, s, h, m)
    if _REC is not None:
        return _REC.get(k)
    w = table().get(k)
    if not w:
        raise Missing(k)
    return list(w)


# ---------------------------------------------------------------- span finding

MONTHS = {
    "de": "januar februar märz april mai juni juli august september oktober november dezember",
    "nl": "januari februari maart april mei juni juli augustus september oktober november december",
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre",
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre",
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre",
    "pt": "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro",
}
PCT_WORDS = r"(?:%|prozent|procent|por\s+ciento|pour\s+cent|per\s+cento|por\s+cento)"
ORD_MARK = {"de": r"\.(?=\s)", "nl": r"\.(?=\s)|ste\b|de\b|e\b", "fr": r"er\b|re\b|ère\b|ème\b|e\b",
            "es": r"\.?º|\.?ª|\.?o\b|\.?a\b", "it": r"\.?º|\.?ª", "pt": r"\.?º|\.?ª"}
GROUP = {"fr": r"(?:\u00a0|\u202f| |\.)", "de": r"(?:\.|\u00a0|\u202f)", "nl": r"(?:\.|\u00a0|\u202f)",
         "es": r"(?:\.|\u00a0|\u202f)", "it": r"(?:\.|\u00a0|\u202f)", "pt": r"(?:\.|\u00a0|\u202f)"}
_NUM = {}


def _num_re(lang):
    if lang not in _NUM:
        _NUM[lang] = re.compile(
            r"(?P<pre>[€$]\s?)?(?<![\w.,/])(?P<int>\d{1,3}(?:" + GROUP[lang] + r"\d{3})+(?!\d)|\d+)"
            r"(?:(?P<dsep>[,.])(?P<frac>\d+))?(?P<ord>" + ORD_MARK[lang] + r")?(?P<deg>°)?"
            r"(?P<post>\s?(?:€|\$|eur\b|" + PCT_WORDS + r"))?(?![\w])", re.I)
    return _NUM[lang]


CLOCK = re.compile(r"(?<![\w.,:])(?P<h>\d{1,2})(?::|h)(?P<m>\d{2})(?![\d:])")


@dataclass
class Span:
    start: int
    end: int
    text: str
    kind: str
    cands: list[str] = field(default_factory=list)
    note: str = ""


def _int(s: str) -> int:
    return int(re.sub(r"[.\u00a0\u202f ]", "", s))


def _find(text: str, lang: str) -> list[dict]:
    out, taken = [], []
    for mt in CLOCK.finditer(text):
        h, m = int(mt["h"]), int(mt["m"])
        if h <= 23 and m <= 59:
            out.append({"a": mt.start(), "b": mt.end(), "kind": "clock", "h": h, "m": m})
            taken.append((mt.start(), mt.end()))
    for mt in _num_re(lang).finditer(text):
        a, b = mt.start(), mt.end()
        if any(x < b and a < y for x, y in taken):
            continue
        d = {"a": a, "b": b, "int": mt["int"], "frac": mt["frac"], "dsep": mt["dsep"]}
        ordm = mt["ord"] or ""
        pre, post = (mt["pre"] or "").strip(), (mt["post"] or "").strip().lower()
        before = text[a - 1:a] if a > 0 else ""
        after = text[b:b + 1]
        if before in ("-", "–", "+", "−") or after in ("-", "–"):
            d.update(kind="skip", note="range/sign")
        elif mt["deg"]:
            d.update(kind="skip", note="degree")
        elif mt["frac"] is not None and len(mt["frac"]) == 3 and not (lang == "fr" and mt["dsep"] == ","):
            d.update(kind="skip", note="ambiguous grouping/decimal")
        elif ordm and mt["frac"] is None and not pre and not post:
            if ordm == ".":
                nxt = text[b:].lstrip().split(" ", 1)[0].lower().strip(",.;:")
                if not (nxt in MONTHS[lang].split() or (nxt[:1].islower() and nxt.isalpha())):
                    d["b"] = b - 1  # a cardinal ending a sentence / before a capitalised word
                    ordm = ""
            if ordm:
                d.update(kind="ord", fem=ordm.lstrip(".") in ("ª", "a", "re", "ère"), mark=ordm)
        if "kind" not in d:
            if pre in ("€", "$") or post in ("€", "$", "eur"):
                d.update(kind="cur", sym="$" if "$" in (pre + post) else "€")
            elif post:
                d["kind"] = "pct"
            elif mt["frac"] is not None:
                d["kind"] = "dec"
            else:
                d["kind"] = "int"
        if d["kind"] in ("int", "dec") and re.search(r"[A-Za-zÀ-ÿ]", text[d["b"]:d["b"] + 1] + before):
            d.update(kind="skip", note="glued code")
        out.append(d)
    return sorted(out, key=lambda x: x["a"])


# ---------------------------------------------------------------- variants (grammar spelling style)

ES_APOC = {"primero": "primer", "tercero": "tercer"}


def _fem_words(ws: list[str], lang: str) -> list[str]:
    """Feminine form of an ordinal / cardinal word list (es/pt/it: -o -> -a; fr premier -> première)."""
    out = []
    for w in ws:
        if lang == "fr":
            w = {"premier": "première", "un": "une", "second": "seconde"}.get(w, w)
        elif lang in ("es", "pt", "it") and w.endswith("o") and w not in ("cento", "ciento", "zero", "cero"):
            w = w[:-1] + "a"
        out.append(w)
    return out


def _card_variants(ws: list[str], lang: str, n: int, year: bool) -> list[list[str]]:
    v = []
    last = ws[-1] if ws else ""
    if lang == "es":
        if last == "uno":
            v += [ws[:-1] + ["un"], ws[:-1] + ["una"]]
        if last == "veintiuno":
            v += [ws[:-1] + ["veintiún"], ws[:-1] + ["veintiuna"]]
        if any(w.endswith("ientos") for w in ws):  # doscientos .. novecientos, quinientos
            v.append([w[:-2] + "as" if w.endswith("ientos") else w for w in ws])
    elif lang == "pt":
        if last == "um":
            v.append(ws[:-1] + ["uma"])
        if "dois" in ws:
            v.append(["duas" if w == "dois" else w for w in ws])
        if any(w.endswith("entos") for w in ws):  # duzentos .. novecentos, quinhentos
            v.append([w[:-2] + "as" if w.endswith("entos") else w for w in ws])
        # the grammar spells Brazilian Portuguese (app 3030976f8: BR teens + short scale); European speech (VoxPopuli)
        # says/writes these. "catorze"/"quatorze" are both Brazilian.
        eu = {"dezesseis": "dezasseis", "dezessete": "dezassete", "dezenove": "dezanove", "catorze": "quatorze"}
        if any(w in eu for w in ws):
            v.append([eu.get(w, w) for w in ws])
        if any(w in ("bilhão", "bilhões") for w in ws):  # PT_PT 10^9 = "mil milhões" ("um bilhão" -> "mil milhões")
            out, i = [], 0
            while i < len(ws):
                if i + 1 < len(ws) and ws[i] == "um" and ws[i + 1] == "bilhão":
                    out += ["mil", "milhões"]
                    i += 2
                    continue
                out += ["mil", "milhões"] if ws[i] in ("bilhão", "bilhões") else [ws[i]]
                i += 1
            v.append(out)
    elif lang == "it":
        if last == "uno":
            v += [ws[:-1] + ["un"], ws[:-1] + ["una"]]
        elif last.endswith("uno") and len(last) > 3:  # ventuno -> ventun / ventuna
            v += [ws[:-1] + [last[:-1]], ws[:-1] + [last[:-1] + "a"]]
    elif lang == "fr":
        if last == "un":
            v.append(ws[:-1] + ["une"])
    elif lang == "nl":
        if ws == ["één"]:
            v.append(["een"])
    elif lang == "de":
        if ws == ["eins"]:
            v += [[x] for x in ("ein", "eine", "einen", "einem", "einer")]
        if len(ws) == 1:
            w = ws[0]
            if w.startswith("eintausend"):
                v.append([w[len("ein"):]])
            if "einhundert" in w:
                v.append([w.replace("einhundert", "hundert", 1)])
    # the hundreds reading of 1100-1999 ("fünfzehnhundert", "vijftienhonderd", "quinze cents") for non-years; years
    # read that way already in de/nl/fr, so their cardinal reading is the variant (added by the caller)
    if lang in ("de", "nl", "fr") and 1100 <= n <= 1999 and not year:
        v.append(_hundreds(lang, n))
    return v


def _hundreds(lang: str, n: int) -> list[str]:
    hi, rem = divmod(n, 100)
    head = g(lang, "card", hi)
    tail = g(lang, "card", rem) if rem else []
    if lang == "fr":
        return head + ["cents" if not rem else "cent"] + tail
    word = {"de": "hundert", "nl": "honderd"}[lang]
    return ["".join(head) + word + "".join(tail)]


def _ord_variants(ws: list[str], lang: str, fem: bool) -> list[list[str]]:
    v = []
    if lang == "de" and len(ws) == 1:
        v += [[ws[0] + e] for e in ("n", "r", "s", "m")]
    elif lang in ("es", "pt", "it"):
        f = _fem_words(ws, lang)
        if f != ws:
            v.append(f)
        if lang == "es" and ws[-1] in ES_APOC:
            v.append(ws[:-1] + [ES_APOC[ws[-1]]])
    elif lang == "fr" and ws == ["premier"]:
        v.append(["première"])
    if fem and v:  # the written marker says feminine: put the feminine reading first among the variants
        v.sort(key=lambda x: x != _fem_words(ws, lang))
    return v


def _digits(lang: str, s: str) -> list[str]:
    out = []
    for c in s:
        out += g(lang, "card", int(c))
    return out


def _dec_frac_variant(lang, n_int, frac, canonical) -> list[list[str]]:
    """Fraction read as one number ("null Komma sechsundneunzig") next to the canonical digit-by-digit reading."""
    if not frac or len(frac) > 2 or frac.startswith("0") or len(frac) < 2:
        return []
    iw = g(lang, "card", n_int)
    sep = canonical[len(iw)] if len(canonical) > len(iw) else None
    if sep is None:
        return []
    return [iw + [sep] + g(lang, "card", int(frac))]


def _uniq(xs):
    seen, out = set(), []
    for x in xs:
        s = " ".join(x)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _cands(d: dict, lang: str) -> list[str]:
    k = d["kind"]
    if k == "skip":
        return []
    if k == "clock":
        return _uniq([g(lang, "clock", h=d["h"], m=d["m"])])
    raw, frac = d["int"], d["frac"]
    n = _int(raw)
    lead0 = len(re.sub(r"\D", "", raw)) > 1 and raw.startswith("0")
    if k == "int":
        if lead0:
            return _uniq([_digits(lang, raw)])
        if n > 999_999_999_999:
            return []
        out = []
        is_year = re.fullmatch(r"1[0-9]{3}|20[0-9]{2}", raw) is not None
        if is_year:
            y = g(lang, "year", n)
            c = g(lang, "card", n)
            out += [y, c] + _card_variants(c, lang, n, year=True)
            if lang == "fr" and 1100 <= n <= 1999:
                out.append(_hundreds(lang, n))
        else:
            c = g(lang, "card", n)
            out += [c] + _card_variants(c, lang, n, year=False)
            if len(raw) >= 5 and re.fullmatch(r"\d+", raw):
                out.append(_digits(lang, raw))
        return _uniq(out)
    if k == "ord":
        if n >= 1_000_000:
            return []  # the Spanish grammar has no reading for the ordinal of 1,000,000
        o = g(lang, "ord", n)
        out = [o] + _ord_variants(o, lang, d.get("fem", False))
        if d.get("fem"):
            out = [_fem_words(o, lang)] + out if lang in ("es", "pt", "it", "fr") else out
        return _uniq(out)
    if k == "dec":
        c = g(lang, "dec", n, frac)
        return _uniq([c] + _dec_frac_variant(lang, n, frac, c))
    if k == "pct":
        f = frac or ""
        c = g(lang, "pct", n, f)
        out = [c]
        if f:
            dc = g(lang, "dec", n, f)
            tail = c[len(dc):] if c[:len(dc)] == dc else None
            if tail is not None:
                out += [v + tail for v in _dec_frac_variant(lang, n, f, dc)]
        else:
            cc = g(lang, "card", n)
            tail = c[len(cc):] if c[:len(cc)] == cc else None
            if tail is not None:
                out += [v + tail for v in _card_variants(cc, lang, n, year=False) if v]
        return _uniq(out)
    if k == "cur":
        c = g(lang, "cur", n, frac or "", d["sym"])
        return _uniq([c])
    return []


def spans(text: str, lang: str) -> list[Span]:
    out = []
    for d in _find(text, lang):
        try:
            cs = _cands(d, lang)
            note = d.get("note", "")
        except Missing as e:
            cs, note = [], f"missing {e}"
        except (ValueError, IndexError) as e:  # e.g. pt "0.011" read as a digit string with a point
            cs, note = [], f"error {e!r}"[:80]
        out.append(Span(d["a"], d["b"], text[d["a"]:d["b"]], d["kind"], cs, note))
    return out


def needed(texts, lang: str) -> set[str]:
    """Every table key the spans of `texts` need (components included)."""
    global _REC
    _REC = _Recorder()
    try:
        for t in texts:
            for d in _find(t, lang):
                try:
                    _cands(d, lang)
                except Exception:
                    pass
            for i in range(0, 131):  # small numbers: digit-by-digit, hundreds components, fractions
                _REC.get(_key(lang, "card", i))
        return set(_REC.keys)
    finally:
        _REC = None


def build_table(texts_by_lang: dict, dumper: str, workdir: str) -> int:
    """Ask a grammar dumper (not in this repo) for every needed key and merge the answers into ml_grammar_table.json."""
    import subprocess

    keys = sorted({k for lang, ts in texts_by_lang.items() for k in needed(ts, lang)})
    tab = table()
    keys = [k for k in keys if k not in tab]
    reqs = []
    for k in keys:
        lang, kind, n, f, s, h, m = k.split("|")
        reqs.append({"l": lang, "k": kind, "n": int(n), "f": f, "s": s or "€", "h": int(h), "m": int(m)})
    if reqs:
        rq, an = os.path.join(workdir, "requests.json"), os.path.join(workdir, "answers.json")
        json.dump(reqs, open(rq, "w"))
        subprocess.run([dumper, rq, an], check=True)
        for k, w in zip(keys, json.load(open(an))):
            tab[k] = w
    json.dump(tab, open(TABLE_PATH, "w"), ensure_ascii=False, sort_keys=True)
    return len(keys)


def render(text: str, sp: list[Span], choice: list[int]) -> str:
    out, pos = [], 0
    for s, c in zip(sp, choice):
        out.append(text[pos:s.start])
        out.append(s.cands[c])
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def all_digits_spanned(text: str, lang: str) -> bool:
    """True when the text has digits, every digit lies inside a number span that can have readings (not a skipped
    range/sign/degree/glued code/ambiguous grouping). P3 prefilter: a row with an unreadable number can never be kept."""
    ds = _find(text, lang)
    if not ds or any(d["kind"] == "skip" for d in ds):
        return False
    cover = [False] * len(text)
    for d in ds:
        for i in range(d["a"], d["b"]):
            cover[i] = True
    return all(cover[i] for i, c in enumerate(text) if c.isdigit())
