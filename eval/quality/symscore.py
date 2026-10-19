"""Symmetric number scorer: digit form and word form are graded by the SAME procedure.

The positional scorer of score_heldout (words-right / digits-right / wrong / dropped) grades a span written in words by
the labeller's candidate readings and a span written in digits by an exact digit-group match, so the two forms differ in
what they forgive ("40 mil", "XIII", "22" for 22:00, a glued phone number and "R$150,00" were wrong; a bare year tail
such as "tausend achtzehn" counted as 1018). This module reads BOTH forms with one reader (number_outcomes.extract_numbers)
and returns independent sub-scores per caption span:
  form        digits | words | mixed | none   what stands where the number stands (from the Nums the reader finds there)
  value_ok    the VALUE is right: the reader turns the hypothesis region (digits, words, "40 mil", "2 Mio.", "XIII") into
              numbers and one of them (or, for multi-group classes, a short run of adjacent ones) equals the caption value.
              Never looks at units, decimal words, letters or symbols.
  unit        (spoken, present): the label speaks a unit (percent / euro / km ...) and the region has the unit as a word,
              a symbol or an abbreviation. "30" without "%" is unit-missing exactly like "treinta" without "por ciento".
  dec         (spoken, present): the label speaks a decimal word and the region has the word or a digit separator.
  affix       alnum only: the letter part ("G" of G20) stands in the region.
and  clean = value_ok and unit ok and dec ok  (the full reading).
Unscorable languages (no word reader) fail loudly (UnscorableLanguage).

Input: the span records of classread.score_rows (one dict per caption span, with the region, label and candidate readings).
"""
import re
from decimal import Decimal

import number_outcomes as N
import score_heldout as SH  # puts pipeline/ on the path
import census_common as CC  # noqa: E402

MULTI = {"phone", "dec", "dec_pair", "clock", "date", "range"}  # value = several numbers in a row (concat of adjacent Nums)
GLUED = {"phone", "dec_pair", "clock", "date", "range"}         # the value may also stand as ONE digit group ("997824426")
MULTI_RAW = MULTI | {"other"}                                   # digit runs may be joined for these too
CAND_KINDS = {"date", "clock", "frac", "ord", "decade", "range", "roman", "amp", "version"}
LT_CLASSES = {"int_small", "int_big", "year", "other", "pct", "cur", "unit", "dec"}  # label reading may stand in for the caption
PAIR_CLASSES = {"year", "clock", "cur", "dec_pair", "other", "date"}
SYMBOLS = set("%€$£°‰")
_RV = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
GAP_OK = re.compile(r"^[ \-–−/.]{0,2}$")
MAX_GAP = 1  # at most this many tokens between two numbers that are read as one (clock "zehn Uhr neun", price "2 Euro 80")
FLAGS = {"raw": True, "compound": True, "label_target": True, "cand": True, "neighbour": True}


class UnscorableLanguage(KeyError):
    pass


def check_lang(lang):
    if lang not in N._LEX and N._ext_reader(lang) is None:
        raise UnscorableLanguage(lang)


def roman_value(s):
    t = 0
    for a, b in zip(s, s[1:] + " "):
        v = _RV[a]
        t += -v if b in _RV and _RV[b] > v else v
    return t


def _canon_dec(d: Decimal) -> str:
    """The plain expansion with trailing fractional zeros removed and the point KEPT ('2.80' -> '2.8'; '28' stays an
    integer, never '2.8'); sign ignored on both sides."""
    return format(abs(d).normalize(), "f")


def num_strings(n):
    """(integer strings, decimal strings) a reader Num can mean."""
    ints, decs = set(), set()
    for v in n.vals:
        try:
            d = Decimal(v)
        except Exception:
            continue
        if not d.is_finite():
            continue
        if d == d.to_integral_value():
            ints.add(str(abs(int(d))))
        else:
            decs.add(_canon_dec(d))
    return ints, decs


class T:
    """Caption targets. single: one Num equal to it; dec: one non-integer Num; cat: only as adjacent Nums joined; hour: HH:00."""
    __slots__ = ("single", "dec", "cat", "hour")

    def __init__(self):
        self.single, self.dec, self.cat, self.hour = set(), set(), set(), None

    def all_int(self):
        return self.single | self.cat


def targets(text: str, kind: str = "") -> T:
    t = text.strip()
    out = T()
    rm = re.fullmatch(r"([IVXLCDM]+)(?:e|ème|eme|er|º|°|\.)?", t)
    if rm and (len(rm.group(1)) >= 2 or rm.group(1) in ("I", "V", "X")):
        out.single.add(str(roman_value(rm.group(1))))
        return out
    cm0 = re.fullmatch(r"\D*?(\d{1,2})\s*(?:[:hH]|uur|u)\s*(\d{2})\s*\D*", t) or \
        re.fullmatch(r"\D*?(\d{1,2})\.(\d{2})\s*(?:uhr|uur|h|u|ore|horas?)\b\D*", t.lower())
    if cm0 and int(cm0.group(1)) <= 24 and int(cm0.group(2)) <= 59:  # 3h25, 14:30, 10.09 Uhr: hour + minutes
        out.cat.add(cm0.group(1) + cm0.group(2))
        out.cat.add(cm0.group(1).lstrip("0") + cm0.group(2))
        if cm0.group(2) == "00":
            out.hour = cm0.group(1).lstrip("0") or "0"
        return out
    hm = re.fullmatch(r"(\d{1,2})\s*(?:uhr|h|u|horas?|heures?|ore|hrs?)", t.lower())
    if hm:
        out.hour = hm.group(1).lstrip("0") or "0"
        out.single.add(out.hour)
        return out
    rg = re.fullmatch(r"\D*?(\d+)\s*[-–—]\s*(\d+)\D*", t)
    if rg and kind in ("range", "") and not re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", t):  # 2007-2013: both numbers, in order
        out.cat.add(rg.group(1) + rg.group(2))
        return out
    m = re.search(r"\d[\d.,  ' ]*\d|\d", t)
    if not m:
        return out
    chunk = m.group(0)
    dig = re.sub(r"\D", "", chunk)
    cm = re.search(r"[.,](\d{1,2})$", chunk)  # a probable decimal tail ("2,80", "150,00", "3,5")
    if cm:
        try:
            body = re.sub(r"[.,  ' ]", "", chunk[:cm.start()])
            d = Decimal(f"{body}.{cm.group(1)}")
            if d == d.to_integral_value():
                out.single.add(str(int(d)))      # "150,00": 150 is the value
            else:
                out.dec.add(_canon_dec(d))       # "2,80": a decimal Num 2.8 ...
            out.cat.add(dig)                     # ... or "2 Euro 80" / "2, 80" as adjacent Nums, never the integer 280
            out.cat.add(dig.lstrip("0") or "0")
        except Exception:
            out.single.add(dig)
        return out
    out.single.add(dig)
    out.single.add(dig.lstrip("0") or "0")
    return out


class SNum(N.Num):
    """Reader Num + position in the whole region (gp = first token, ge = last token, clause-aware)."""
    __slots__ = ("gp", "ge")


def read(text: str, lang: str):
    """number_outcomes reader that does not run across punctuation: 'mil, dos mil' is read clause by clause, and a
    Roman numeral with an ordinal suffix ('XXe', 'IVe') as the numeral."""
    text = re.sub(r"\b([IVXLCDM]{2,7})(?:e|ème|eme|er|º|°)\b", r"\1", text)
    out, off = [], 0
    try:
        for clause in re.split(r"(?<=[.,;:!?])\s+", text):
            ns = [n for n in N.extract_numbers(clause, lang, keep_weak=True, garbles=False) if n.vals]
            for n in ns:
                s = SNum(n.vals, n.form, n.ordinal, n.weak, n.start, n.end, n.fodder)
                s.gp, s.ge = off + n.start, off + n.end
                out.append(s)
            off += len(N._TOKEN.findall(N._fold(clause)))
    except KeyError:
        raise UnscorableLanguage(lang)
    return out


def _adjacent(a, b):
    return b.gp - a.ge - 1 <= MAX_GAP


ZERO_WORD = re.compile(r"\b(?:oh|o|zero|null|nul|zéro|cero|nulla)\b", re.I)


def _match(nums, tg: T, multi: bool, pair: bool = False, cls: str = "", zone: str = ""):
    """Indices of the Nums that make the value, else None. single Num | decimal Num | (pair) two Nums read as one |
    (multi) a run of ADJACENT int Nums joined | HH:00 as the hour alone (no minutes right after it)."""
    S = [num_strings(n) for n in nums]
    for k, (a, b) in enumerate(S):
        if a & tg.single or b & tg.dec:
            return [k]
        if tg.hour and tg.hour in a:
            nxt = nums[k + 1] if k + 1 < len(nums) else None
            minute = nxt is not None and _adjacent(nums[k], nxt) and any(0 < int(x) < 60 for x in S[k + 1][0])
            if not minute and not re.search(rf"(?<!\d){tg.hour}\s*[:h.]\s*(?!00)\d\d", zone):
                return [k]
    cats = tg.all_int()
    if pair:  # two numbers read as one: "twenty twenty four" 20|24, "zehn Uhr neun" 10|09 (zero-pad only where a zero is plausible)
        for k in range(len(S) - 1):
            if not _adjacent(nums[k], nums[k + 1]):
                continue
            for a in S[k][0]:
                for b in S[k + 1][0]:
                    if not (10 <= int(a) <= 99 and 0 <= int(b) <= 99):
                        continue
                    if len(b) == 1 and cls == "year" and not ZERO_WORD.search(zone):
                        continue
                    if (a + (b if len(b) == 2 else b.zfill(2))) in cats:
                        return [k, k + 1]
    if multi:
        for s in range(len(S)):
            accs = {("", ())}
            for e in range(s, len(S)):
                if e > s and not _adjacent(nums[e - 1], nums[e]):
                    break
                new = set()
                for acc, idx in accs:
                    for x in S[e][0]:
                        z = acc + x
                        if z in cats and len(idx) >= 1:
                            return list(idx) + [e]
                        if any(t.startswith(z) for t in cats):
                            new.add((z, idx + (e,)))
                accs = new
                if not accs:
                    break
    return None


_RUN = re.compile(r"\d+(?:[   .,:/\-]\d+)*")


def raw_match(zone: str, single, cat, multi: bool) -> bool:
    """Digit runs of the region as plain strings (groups joined only through space / dot / comma / colon / slash / hyphen,
    never across a letter). One group must equal a `single` target ('2,80 EUR' is 280 only as the joined groups 2 and
    80, never as the lone group '280'); consecutive groups of one run joined may equal a `single` or `cat` target, but
    outside multi classes only a thousands grouping may join."""
    both = single | cat
    for m in _RUN.finditer(zone):
        gs = re.findall(r"\d+", m.group(0))
        for a in range(len(gs)):
            acc = ""
            for b in range(a, len(gs)):
                acc += gs[b]
                if acc in (single if b == a else both) and (
                        b == a or multi or (len(gs[a]) <= 3 and all(len(x) == 3 for x in gs[a + 1:b + 1]))):
                    return True
                if not any(t.startswith(acc) for t in both):
                    break
    return False


def form_of(nums, idx=None):
    sel = [nums[i] for i in idx] if idx is not None else nums
    fs = {n.form for n in sel if n.form in ("digits", "words")}
    if not fs:
        return "none"
    return "mixed" if len(fs) > 1 else fs.pop()


# ---------------------------------------------------------------- unit / decimal word sub-scores
_PCT_ANY = re.compile(r"(?:pour ?cent|por ?cent|por ?cien|per ?cent|prozent|procent|percent)")


def unit_sub(rec) -> tuple:
    """(spoken, present) for pct / cur / unit spans. Same test for digits and words: the unit must stand as a word, a symbol
    or the span's own written abbreviation ('km', 'kg', '%', '$'); a bare digit string or bare number words do not carry it."""
    cls, lang = rec["cls"], rec["lang"]
    if cls not in ("pct", "cur", "unit") or rec["place"] is None:
        return False, False
    lab_text = N._fold(" ".join(rec["lab_words"]))
    words = rec["reg_words"]
    hyp_text = N._fold(" ".join(words))
    toks = [w.strip(",;:!?.¿¡()\"'").lower() for w in words]
    glued = " ".join(words)
    if cls == "pct":
        rx = N._PCT.get(lang)
        spoken = bool(rx and re.search(rx, lab_text))
        present = bool(_PCT_ANY.search(hyp_text)) or "%" in glued
        return spoken, present
    lw = {SH._nw(w) for w in rec["lab_words"]}
    uw = SH._unit_words(rec["cands"], lang) & lw
    if cls == "cur":
        # a word that is also a number word ("cent" = 100 in fr) inside the amount is no unit: the unit must not match a
        # number morpheme of the span's own reading
        uw |= {w for w in lw if (w in N._CUR or w in N._MINOR) and N.parse_value(w, lang) is None}
    if not uw:
        return False, False
    hyp_norm = [SH._nw(w) for w in words]
    present = any(h == u or (len(h) >= 4 and (h.startswith(u) or u.startswith(h))) for h in hyp_norm for u in uw)
    if not present:
        present = any(ch in SYMBOLS for w in words for ch in w)
    if not present:  # the span's own abbreviation glued to a digit token ("5km", "70km/h") or standing as a token
        abbr = re.sub(r"[\d\s.,'’\-:/  ]+", "", rec["text"]).lower()
        for w in words:
            suf = re.sub(r"[\d\s.,'’\-:/]+", "", w).lower()
            if re.search(r"\d", w) and suf and (suf == abbr or (abbr and suf.startswith(abbr)) or suf in uw or any(len(u) >= 4 and u.startswith(suf) and len(suf) >= 2 for u in uw)):
                present = True
        if abbr and any(t.rstrip(".") == abbr or abbr in t.replace("/", " ").split() for t in toks):
            present = True
    return True, present


def dec_sub(rec) -> tuple:
    cls, lang = rec["cls"], rec["lang"]
    rx = CC.DEC_SAID.get(lang)
    if not rx or rec["place"] is None or not re.search(r"\d[.,]\d{1,2}(?!\d)", rec["text"]) or cls in ("phone", "clock", "date", "dec_pair"):
        return False, False
    spoken = bool(rx.search(N._fold(" ".join(rec["lab_words"]))))
    h = N._fold(" ".join(rec["reg_words"]))
    present = bool(rx.search(h)) or bool(re.search(r"\d[.,]\d", h)) or bool(CC.HALF_SAID.search(h))
    return spoken, present


def affix_sub(rec) -> tuple:
    if rec["cls"] != "alnum":
        return False, False
    letters = re.sub(r"[^A-Za-z]", "", rec["text"]).lower()
    if not letters:
        return False, False
    h = " ".join(rec["reg_words"]).lower()
    return True, bool(re.search(rf"(?<![a-z]){re.escape(letters)}(?![a-z])|(?<![a-z]){re.escape(letters)}\d", h))


# ---------------------------------------------------------------- the span scorer
def compound_groups(recs_row):
    """Chains of caption-adjacent spans (gap only hyphen / slash / dot / space): 22 14 19 44 49, 43-98."""
    recs = sorted(recs_row, key=lambda r: r["si"])
    chains, cur = [], []
    for r in recs:
        if r.get("nochain"):
            if len(cur) > 1:
                chains.append(cur)
            cur = []
            continue
        if cur and GAP_OK.match(r["cap"][cur[-1]["b"]:r["a"]]) and r["a"] >= cur[-1]["b"]:
            cur.append(r)
        else:
            if len(cur) > 1:
                chains.append(cur)
            cur = [r]
    if len(cur) > 1:
        chains.append(cur)
    return chains


def zone_variants(r):
    """The region as read, then without a first / last word that is only a NEIGHBOUR label word ('32' + 'Millionen' where the
    label span is just the 32): the +-1 word slack must not let a neighbour change the number (veinte + una = 21)."""
    w = r["reg_words"]
    if not FLAGS["neighbour"]:
        return [w]
    own = {SH._nw(x) for x in (r.get("labseg") or "").split()}
    nb = {SH._nw(x) for x in r["lab_words"]} - own
    out = [w]
    if len(w) >= 2 and SH._nw(w[0]) in nb:
        out.append(w[1:])
    if len(w) >= 2 and SH._nw(w[-1]) in nb:
        out.append(w[:-1])
    if len(w) >= 3 and SH._nw(w[0]) in nb and SH._nw(w[-1]) in nb:
        out.append(w[1:-1])
    return out


def label_targets(r, lang, kind=""):
    """Integer strings of the label's own reading when the label span is exactly one number and covers the label words
    (the audio says it: a French year said 'quatre vingt quinze' under the caption 1995). Plain numeric classes only."""
    if not r.get("labseg") or r["cls"] not in LT_CLASSES or r["kind"] in CAND_KINDS:
        return set()
    try:
        ln = read(r["labseg"], lang)
    except UnscorableLanguage:
        return set()
    if len(ln) != 1:
        return set()
    toks = len(N._TOKEN.findall(N._fold(r["labseg"])))
    if ln[0].ge - ln[0].gp + 1 < toks - 2:  # the Num must cover (almost) all words of the label span
        return set()
    return set(num_strings(ln[0])[0])


def label_multi(r, tg: T, lang):
    """The label itself speaks the target as several numbers (digit by digit 'dois quatro cinco quatro', pairs 'acht en
    twintig acht en twintig'): adjacent concatenation is then the right reading for both forms."""
    if not r.get("labseg"):
        return False
    try:
        ln = read(r["labseg"], lang)
    except UnscorableLanguage:
        return False
    return len(ln) >= 2 and _match(ln, tg, True, zone=r["labseg"]) is not None and _match(ln, tg, False, zone=r["labseg"]) is None


def sym_score_row(recs_row) -> dict:
    """recs_row: all span records of one (lang, id) row. Returns {key: result}."""
    out, comp = {}, {}
    for ch in compound_groups(recs_row):
        tgt = ch[0].get("chain_tgt") or "".join(re.sub(r"\D", "", c["text"]) for c in ch)
        if not tgt:
            continue
        lo = min(c["ireg"][0] for c in ch)
        hi = max(c["ireg"][1] for c in ch)
        for c in ch:
            comp[c["key"]] = (tgt, lo, hi)
    for r in recs_row:
        lang = r["lang"]
        check_lang(lang)
        tg = targets(r["text"], r["kind"])
        if FLAGS["label_target"]:
            tg.single |= label_targets(r, lang)  # the label (audio) reading, a second accepted value
        multi = r["cls"] in MULTI or bool(tg.dec) or label_multi(r, tg, lang)
        hyp_words = r["hyp"].split()
        idx, nums, via = None, [], None
        for variant in zone_variants(r):
            zone = " ".join(variant)
            nums_v = read(zone, lang)
            if not nums:
                nums = nums_v
            idx = _match(nums_v, tg, multi, r["cls"] in PAIR_CLASSES, r["cls"], zone)
            if idx is not None:
                nums, via = nums_v, "span"
                break
            if FLAGS["raw"] and (multi or r["cls"] in MULTI_RAW) and raw_match(zone, tg.single | (tg.cat if r["cls"] in GLUED else set()), tg.cat, True):
                idx, nums, via = [], nums_v, "raw"
                break
        if idx is None and r["key"] in comp and FLAGS["compound"]:
            tgt, lo, hi = comp[r["key"]]
            cz = " ".join(hyp_words[lo:hi])
            cn = read(cz, lang)
            ct = T()
            ct.cat |= {tgt, tgt.lstrip("0") or "0"}
            idx2 = _match(cn, ct, True, zone=cz)
            if idx2 is not None:
                idx, nums, via = idx2, cn, "compound"
            elif raw_match(cz, ct.cat, set(), True):  # the chain as one glued group, or its groups joined
                idx, nums, via = [], cn, "compound"
        if idx is None and FLAGS["cand"] and r["kind"] in CAND_KINDS and r.get("cands"):
            # classes the number reader cannot read (month names, fraction words, relative clocks, German decades): the
            # labeller's own value-preserving readings of the span, matched as whole words
            hw = [SH._nw(w) for w in r["reg_words"]]
            if SH._heard(r["cands"], hw, 0, len(hw), 0, len(hw)):
                idx, via = [], "cand"
        value_ok = idx is not None
        if via == "raw":
            form = "digits"
        elif via == "cand":
            form = "words" if not re.search(r"\d", " ".join(r["reg_words"])) else form_of(nums)
        else:
            form = form_of(nums, idx) if value_ok else form_of(nums)
        us, up = unit_sub(r)
        ds, dp = dec_sub(r)
        asp, apr = affix_sub(r)
        out[r["key"]] = dict(lang=lang, cl=r["cl"], cls=r["cls"], form=form, value_ok=value_ok, via=via,
                             unit_spoken=us, unit_present=up, dec_spoken=ds, dec_present=dp, affix_spoken=asp, affix_present=apr,
                             clean=value_ok and (not us or up) and (not ds or dp),
                             old_form=r.get("form"), old_right=r.get("right"), old_out=r.get("out"), placed=r["place"] is not None)
    return out


def sym_score_arm(recs) -> dict:
    by = {}
    for r in recs:
        by.setdefault((r["lang"], r["id"]), []).append(r)
    res = {}
    for k, rows in by.items():
        res.update(sym_score_row(rows))
    return res
