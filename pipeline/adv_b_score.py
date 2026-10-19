"""Adversarial-set scorer (advB): what the pinned symmetric scorer cannot say about verbatim, mixed-style number speech.
It does NOT replace or change the pinned scorer (eval/quality: number_outcomes, symscore, classread); it scores the same
hypotheses on two extra questions, per test row of `adv_b.test_plan`:

  edge_exact / edge_wer   the edge phrase (the whole stumble, the whole phone number, the whole date) is transcribed word for
                          word: the label's edge words aligned against the hypothesis (word edit distance), "oh" = "zero" = "o"
                          and "double five" = "five five" folded on both sides, accents and punctuation folded. A digit
                          written where words were said counts as an error and is reported apart (`digits`).
  value_ok (phone)        the hypothesis's phone region read as a DIGIT STRING (digits, oh / zero, pairs, hundreds, double /
                          treble / triple, country code) equals the number's digits; independent of how the speaker grouped them.

The matched normal-speech CONTROL rows (same value, same carrier, the standard reading) are scored the same way; `report` gives
the adversarial rows, the control rows and the paired difference per family / construction / quadrant, so a build can be
checked for helping the adversarial readings WITHOUT hurting normal speech.

    python adv_b_score.py --test test_en.jsonl --control control_en.jsonl --hyps hyps.jsonl [--json out.json]
    (hyps.jsonl: one {"id": ..., "hyp": ...} per test or control row)
Pure functions otherwise; tests in test_adv_b_score.py."""
import argparse
import collections
import json
import re
import unicodedata

import adv_b_words as W

REPEAT = {"double": 2, "doble": 2, "dubbel": 2, "doppel": 2, "doppio": 2, "duplo": 2, "triple": 3, "treble": 3, "tripla": 3, "triplo": 3,
          "dreifach": 3}
ZERO = {"oh", "o", "zero", "nought", "naught", "cero", "zéro", "null", "nul"}
CONNECT = {"und", "y", "e", "et", "en", "ën", "ed", "and"}
FILL = {"uh", "um", "eh", "euh", "hmm", "mm", "erm", "uhm", "ah"}


def fold(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if not unicodedata.combining(c))


def toks(s: str) -> list:
    s = s.replace("’", "'").replace("-", " ")
    return [fold(x) for x in re.findall(r"[^\W_]+(?:'[^\W_]+)?|\d+(?:[.,:/]\d+)*", s.lower(), re.U)]


def canon(t: list) -> list:
    """Fold the spellings a transcriber chooses freely: zero words, repeat words, fillers."""
    out, i = [], 0
    while i < len(t):
        w = t[i]
        if w in REPEAT and i + 1 < len(t):
            out += [("0" if t[i + 1] in ZERO else t[i + 1])] * REPEAT[w]
            i += 2
            continue
        if w in FILL:
            out.append("<f>")
        else:
            out.append("0" if w in ZERO else w)
        i += 1
    return out


def align(ref: list, hyp: list) -> list:
    """Word-level Levenshtein alignment: list of (ref index or None, hyp index or None) pairs, in order."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((i - 1, None))
            i -= 1
        else:
            ops.append((None, j - 1))
            j -= 1
    return ops[::-1]


def edge_words(row: dict) -> tuple:
    """(reference tokens of the sentence, the set of token indices inside an edge span). Tokens are produced span by span so the
    mapping from characters to tokens is exact."""
    text, spans = row["text"], sorted(row["edge_spans"])
    ref, inside, pos = [], set(), 0
    for a, b in spans:
        ref += canon(toks(text[pos:a]))
        k0 = len(ref)
        ref += canon(toks(text[a:b]))
        inside |= set(range(k0, len(ref)))
        pos = b
    ref += canon(toks(text[pos:]))
    return ref, inside


def edge_score(row: dict, hyp: str) -> dict:
    ref, inside = edge_words(row)
    h = canon(toks(hyp))
    ops = align(ref, h)
    lo, hi = min(inside), max(inside)
    err = sub = dele = ins = 0
    last_ref = None
    seen_first = False
    digits = False
    for r, j in ops:
        in_edge = r is not None and r in inside
        if r is not None and lo <= r <= hi:
            seen_first = True
        if in_edge:
            if j is None:
                dele += 1
            elif ref[r] != h[j]:
                sub += 1
                digits |= bool(re.search(r"\d", h[j]))
        elif r is None and j is not None and seen_first and last_ref is not None and last_ref < hi and last_ref >= lo:
            ins += 1                         # a hypothesis word inserted inside the edge region
            digits |= bool(re.search(r"\d", h[j]))
        if r is not None:
            last_ref = r
    n = len([r for r in inside])
    err = sub + dele + ins
    return {"edge_exact": err == 0, "edge_err": err, "edge_n": n, "edge_wer": round(err / n, 4) if n else 0.0, "sub": sub, "del": dele, "ins": ins,
            "digits": digits or bool(re.search(r"\d", hyp))}


# ------------------------------------------------------------------------------------------------ phone reader
_TUP = {}


def _table(lang: str) -> dict:
    """word tuple -> digit string for the readings of a phone group: digit words, pairs 10-99, hundreds 100-999, 'zero X' pairs."""
    if lang in _TUP:
        return _TUP[lang]
    t = {}
    for n in range(0, 1000):
        if n >= 100 and n % 100:
            continue            # 612 is "seiscientos" + the pair "doce" (see phone_digits); only round hundreds are table entries
        try:
            w = tuple(fold(x) for x in W.card(n, lang).split())
        except Exception:  # noqa: BLE001
            continue
        t.setdefault(w, str(n))
    for d in range(10):
        t[(fold(W.DIGITS[lang][d]),)] = str(d)
    z = fold(W.DIGITS[lang][0])
    for z_ in ZERO:
        t[(z_,)] = "0"
    for z_ in ZERO | {z}:
        for d in range(1, 10):
            t[(z_, fold(W.DIGITS[lang][d]))] = "0" + str(d)
    if lang == "fr":
        t[("zero", "huit", "cent")] = "0800"
        for tens, word in ((7, "septante"), (8, "huitante"), (9, "nonante")):
            for u in range(10):
                n = tens * 10 + u
                tail = () if u == 0 else ("et", "un") if u == 1 else (fold(W.DIGITS["fr"][u]),)
                t.setdefault((word,) + tail, str(n))
    _TUP[lang] = t
    return t


def phone_digits(words: list, lang: str, merge_single: bool = False) -> str:
    """Digit string of a phone reading (longest-match over the table; unknown words are skipped; 'plus' etc. add nothing)."""
    t = _table(lang)
    w = [fold(x) for x in words]
    # repeat words expand first: "double five" -> five five
    out, i = [], 0
    while i < len(w):
        if w[i] in REPEAT and i + 1 < len(w):
            j = i + 1
            best = None
            for ln in (3, 2, 1):
                if tuple(w[j:j + ln]) in t:
                    best = (ln, t[tuple(w[j:j + ln])])
                    break
            if best:
                out.append(best[1] * REPEAT[w[i]])
                i = j + best[0]
                continue
        for ln in (6, 5, 4, 3, 2, 1):
            key = tuple(w[i:i + ln])
            if len(key) == ln and key in t:
                v = t[key]
                nxt = w[i + ln] if i + ln < len(w) else ""
                if len(v) == 2 and v[0] == "0" and nxt in CONNECT:      # "null drei und dreissig": the 0 is a digit, drei und dreissig is 33
                    continue
                if len(v) == 3 and v.endswith("00") and v[0] != "0":    # a round hundred: "seiscientos" + a PAIR (doce) is 612; + a digit is 600 then that digit
                    for ln2 in (4, 3, 2, 1):
                        k2 = tuple(w[i + ln:i + ln + ln2])
                        v2 = t.get(k2) if len(k2) == ln2 else None
                        if v2 is not None and ((len(v2) == 2 and v2[0] != "0") or (merge_single and len(v2) == 1)):
                            v = v[0] + (v2 if len(v2) == 2 else "0" + v2)
                            ln += ln2
                            break
                out.append(v)
                i += ln
                break
        else:
            i += 1
    return "".join(out)


def phone_digit_set(words: list, lang: str, cap: int = 400) -> set:
    """Every digit string some parse of `words` can give (a token run is often ambiguous: "quarante quatre vingt dix sept" is 40 97 or 44 20 17;
    "seiscientos cinco" is 605 or 600 5). A hypothesis is value-right when the expected digits are among them."""
    import functools

    t = _table(lang)
    w = [fold(x) for x in words]
    n = len(w)

    @functools.lru_cache(maxsize=None)
    def go(i):
        if i >= n:
            return frozenset({""})
        res = set()
        hit = False
        if w[i] in REPEAT and i + 1 < n:
            for ln in (3, 2, 1):
                k = tuple(w[i + 1:i + 1 + ln])
                if len(k) == ln and k in t:
                    hit = True
                    for r in go(i + 1 + ln):
                        res.add(t[k] * REPEAT[w[i]] + r)
        for ln in (6, 5, 4, 3, 2, 1):
            k = tuple(w[i:i + ln])
            if len(k) == ln and k in t:
                hit = True
                v = t[k]
                for r in go(i + ln):
                    res.add(v + r)
                if len(v) == 3 and v.endswith("00") and v[0] != "0":       # "seiscientos" + "doce" = 612, + "cinco" = 605
                    for ln2 in (4, 3, 2, 1):
                        k2 = tuple(w[i + ln:i + ln + ln2])
                        v2 = t.get(k2) if len(k2) == ln2 else None
                        if v2 is not None and (len(v2) <= 2):
                            for r in go(i + ln + ln2):
                                res.add(v[0] + (v2 if len(v2) == 2 else "0" + v2) + r)
        if not hit:
            res |= set(go(i + 1))
        return frozenset(list(res)[:cap])

    return set(go(0))


def expected_digits(row: dict) -> str:
    sp = row["spans"][0]
    return re.sub(r"\D", "", sp["text"])


def phone_region(row: dict, hyp: str) -> list:
    """The hypothesis words aligned to the row's edge span (the span's neighbours anchor the region)."""
    ref, inside = edge_words(row)
    h_raw = toks(hyp)
    h = canon(h_raw)
    ops = align(ref, h)
    js = [j for r, j in ops if j is not None and (r in inside if r is not None else False)]
    if not js:
        return []
    lo, hi = min(js), max(js)
    return h_raw[lo:hi + 1]


def phone_score(row: dict, hyp: str) -> dict:
    exp = expected_digits(row)
    lang = row["lang"]
    reg = phone_region(row, hyp)
    got = phone_digits(reg, lang)
    # the label's own reading may add a country code: compare with and without the trunk zero
    cands = phone_digit_set(reg, lang)
    ok = any(g == exp or g == exp.lstrip("0") or exp == g.lstrip("0") or (g.endswith(exp.lstrip("0")) and len(g) - len(exp.lstrip("0")) <= 3) for g in cands)
    return {"value_ok": bool(ok), "digits_read": got, "digits_expected": exp}


# ------------------------------------------------------------------------------------------------ scoring + report
def score_row(row: dict, hyp: str) -> dict:
    out = edge_score(row, hyp)
    fam = row.get("family")
    if fam == "phone":
        out.update(phone_score(row, hyp))
    return out


def _cons(row):
    return row.get("control_construction") or row["construction"]


def report(test: list, control: list, hyps: dict) -> dict:
    """Aggregate tables from hypotheses {id: text}: per family / construction / quad: n, edge_exact %, mean edge_wer, value_ok % (phone);
    the same for the control rows and the paired difference on the rows that have a matched control."""
    rows = {}
    for r in test + control:
        if r["id"] in hyps:
            rows[r["id"]] = dict(score_row(r, hyps[r["id"]]), **{k: r.get(k) for k in ("lang", "family", "construction", "quad", "bed", "control_of", "control_construction")})
    def agg(rs):
        n = len(rs)
        if not n:
            return None
        o = {"n": n, "edge_exact": round(100 * sum(r["edge_exact"] for r in rs) / n, 1), "edge_wer": round(100 * sum(r["edge_wer"] for r in rs) / n, 2),
             "digits": round(100 * sum(r["digits"] for r in rs) / n, 1)}
        pv = [r for r in rs if "value_ok" in r]
        if pv:
            o["phone_value_ok"] = round(100 * sum(r["value_ok"] for r in pv) / len(pv), 1)
        return o
    adv = [r for r in rows.values() if r["bed"] == "adversarial"]
    ctl = [r for r in rows.values() if r["bed"] == "control"]
    out = {"adversarial": {}, "control": {}, "paired": {}}
    for name, group in (("adversarial", adv), ("control", ctl)):
        for key in ("family", "quad"):
            out[name][key] = {k: agg([r for r in group if r[key] == k]) for k in sorted({r[key] for r in group})}
        out[name]["construction"] = {k: agg([r for r in group if _cons(r) == k]) for k in sorted({_cons(r) for r in group})}
        out[name]["all"] = agg(group)
    byid = {r_id: r for r_id, r in rows.items()}
    pairs = [(byid[r["control_of"]], r) for r in ctl if r["control_of"] in byid]
    if pairs:
        d = [(a["edge_err"] - c["edge_err"]) for a, c in pairs]
        out["paired"] = {"n": len(pairs), "control_edge_exact": round(100 * sum(c["edge_exact"] for a, c in pairs) / len(pairs), 1),
                         "adversarial_edge_exact": round(100 * sum(a["edge_exact"] for a, c in pairs) / len(pairs), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--control", default="")
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ld = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
    test = ld(a.test)
    ctl = ld(a.control) if a.control else []
    hyps = {r["id"]: r["hyp"] for r in ld(a.hyps)}
    rep = report(test, ctl, hyps)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"adversarial": rep["adversarial"]["all"], "control": rep["control"]["all"], "paired": rep["paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
