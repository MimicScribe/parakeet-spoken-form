"""Symbol-class synthetic set (2026-10-18, ttsSYM1): numbers where the frozen encoder carries a SYMBOL -- percent
with the unit word, decimals with the decimal word, prices, units, big numbers with magnitude words, phone numbers,
letter+digit codes -- in natural sentences with 2-4 numbers each (adjacent numbers, sentence-initial percent, percent
ranges, look-alike words next to numbers). Open-licence voices (ml_tts_open.py; Kokoro and Chirp 3 HD where the pilot
chose them).

  1. carriers_sym.json: per language, sentences with typed slots ({pct} {pb} {cur} {mag} {phone} ...);
  2. each slot is written by a generator (values from the grammar table's pools, so every reading exists; wide value
     spread: no value above VALUE_SHARE of its slot type), the whole sentence is realised through the ml_tts candidate
     machinery with the FILL readings (ml_cands.FILL_ALL: "por cien", "meia", bare / unit forms, ...), the reading
     rules of each span's OWN slot class (ml_tts.reading_ok: gender, halves, whole-number counts, grammar_rules slips;
     years always read in full);
  3. every planned row is checked: each span's census class (census_common.gate_class_value) is counted against a
     per-language x class TARGET (the build's synthetic room in the current mix, x a loss margin: synthetic shares
     stay inside the caps, rows a cap would refuse are not voiced), and each reading must give the written value back
     through the held-out scorer's reader (eval/quality/number_outcomes) -- a row that fails is dropped, never fixed;
  4. voices: Qwen3 (es / pt / fr) and Chatterbox (de) clones of CC0 / CC BY train-split speakers (ml_tts_open refs),
     Kokoro voices and blends (it), Chirp 3 HD personas (nl, never the held-out persona); rates 0.9-1.25.

    python ml_tts_symbol.py plan --room room.json --tag ttsSYM1 --out <dir> [--langs de,es] [--margin 1.25]
    python ml_tts_symbol.py voices --planned <dir>/planned_<lang>.jsonl --refs <refs meta.json> --vtag ttsSYM1v
    python ml_tts_symbol.py chirp --planned <dir>/voiced_nl.jsonl --vtag ttsSYM1v      # nl: cloud synthesis + upload
    modal run ml_tts.py --langs <lang> --out-tag ttsSYM1c --rows-file "ml/ttsSYM1v/{lang}/planned.jsonl"  # CTC check
    python ml_tts_symbol.py finalize --rows <ctc rows.jsonl> --out <rows.jsonl>     # exact fits only + value check

room.json: {lang: {gate class: {"room": synthetic spans the build would still admit}}} (computed from a built mix).
"""
import argparse
import collections
import decimal
import hashlib
import json
import os
import random
import re
import sys

import census_common as CC
import ml_cands as M
import ml_p3_core as C
import ml_tts as T
import ml_tts_templates as TT

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/vol"
NAMES = M.FILL_ALL
RATES = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
VALUE_SHARE = 0.03  # no written value above this share of its slot type (after VALUE_MIN uses)
VALUE_MIN = 4
MAX_ROWS = 3000
SLOT_RE = re.compile(r"\{([a-z]+)(\d?)(?:~([^}]*))?\}")  # {type}, {type2}, {type~REGEX}
MATH_OP = re.compile(r"^@([-+*/<>]) ")
# slot type -> the ml_tts class whose reading rules apply to its spans
SLOT_CLS = {"pct": "pct", "pa": "count", "pb": "pct", "dec": "dec", "cur": "price", "curbig": "price_big",
            "mag": "large", "bign": "count", "dist": "unit", "speed": "unit", "mass": "unit", "vol": "unit",
            "area": "unit", "temp": "temp", "phone": "phone", "code": "code", "year": "year", "int": "count",
            "date": "date_txt", "daten": "date_num", "ord": "ord", "roman": "roman", "frac": "frac", "ma": "count",
            "mb": "count", "mc": "count"}
# slot type -> the census class it usually lands in (template choice only; the realised class is what counts)
SLOT_GUESS = {"pct": "pct", "pa": "int_small", "pb": "pct", "dec": "dec", "cur": "cur", "curbig": "cur", "mag": None,
              "bign": "int_big", "dist": "unit", "speed": "unit", "mass": "unit", "vol": "unit", "area": "unit",
              "temp": "unit", "phone": "phone", "code": "alnum", "year": "year", "int": "int_small", "date": "date",
              "daten": "date", "ord": "ord", "roman": "roman", "frac": "frac", "ma": "int_small", "mb": "int_small",
              "mc": "int_small"}
FILLER = {"int_small", "year", "date", "ord"}  # classes a sentence may carry beside its symbol numbers
SYMBOL = ("pct", "dec", "cur", "unit", "int_big", "phone", "alnum", "range")
EXTRA = ("date", "ord", "roman", "frac")  # the second carrier file's classes (carriers_sym2.json)
CONSTRAINED = {"ord", "roman", "date"}  # span kinds a {type~REGEX} constraint applies to (never the year of a date)
MONTH_RE = {l: re.compile(r"\b(?:" + v + r")\b", re.I) for l, v in CC._M.items()}
FRACS = ["3/4", "1/4", "2/3", "1/3", "3/5", "2/5", "4/5", "3/8", "5/8", "7/10", "3/10"]
CENTURY_CTX = re.compile(r"(?:siglo|século|Jahrhundert|secolo|eeuw)\s*$|\b(?:au|du)\s*$", re.I)
SCALES = {"de": ["Millionen", "Milliarden"], "es": ["millones"], "fr": ["millions", "milliards"],
          "it": ["milioni", "miliardi"], "nl": ["miljoen", "miljard"], "pt": ["milhões", "bilhões"]}
MONEY = {"es": ["{a} €", "{a} €", "{a} $", "US$ {a}", "{a} £", "£ {a}"], "pt": ["R$ {a}", "R$ {a}", "{a} €", "US$ {a}", "£ {a}"],
         "fr": ["{a} €", "{a} €", "{a} $", "{a} £"], "it": ["{a} €", "€ {a}", "{a} €", "$ {a}", "£ {a}"],
         "de": ["{a} €", "{a} €", "{a} $", "€ {a}", "{a} £"], "nl": ["€ {a}", "{a} €", "€ {a}", "$ {a}", "£ {a}"]}
UNITS = {"dist": ["km", "km", "m"], "speed": ["km/h"], "mass": ["kg", "kg", "g", "t"], "vol": ["l", "ml"],
         "area": ["m²"]}
# an article / preposition that must elide before a vowel-initial reading ("dell'otto per cento", "d'un"): such a
# sentence is left out (the carriers write the full form)
ELISION_BAD = {"it": re.compile(r"\b(?:il|del|al|dal|nel|sul|dei)\s+(?:un|uno|una|otto|ottanta\w*|undici)\b", re.I),
               "fr": re.compile(r"\b(?:de|le)\s+(?:un|une)\b", re.I)}
CODES = {"summit": ["G20", "G7", "G8", "G77"], "gas": ["CO2", "NO2", "PM10", "PM2", "SO2"],
         "road": ["A1", "A2", "A4", "A7", "A9", "A8", "A3", "A5"],
         "ref": ["B12", "K2", "X5", "T3", "A380", "F1", "M5", "H2", "C4", "Q3", "B52", "E3", "U2", "R2", "5G", "4K", "3D"]}
CODE_CTX = (("summit", re.compile(r"(?:Gipfel|cumbre|sommet|vertice|top van|cúpula)", re.I)),
            ("gas", re.compile(r"(?:Ausstoß|emisiones|émissions|emissioni|uitstoot|emissões)", re.I)),
            ("road", re.compile(r"(?:Auf der|Sulla)\s*$")))
MONTH_CTX = re.compile(r"^\s*(?:de mayo|juin|de maio|mai|maggio)")
COUNT_BAD_END = {"de", "es", "fr", "it", "pt", "nl"}  # counts ending in 1 / 2 (un / uno / una, dois / duas) left out
_D = decimal.Decimal
# thin-cell top-up (2026-10-18; values under their gate floors in the current mix): exact values, two per sentence
THIN = {"pt": {"year": {1991: 6, 2001: 8, 2026: 10, 2028: 4, 2044: 3, 2050: 7}},
        "nl": {"year": {2030: 4}}, "it": {"int": {68: 4, 94: 4}}}
THIN_CARRIERS = {"pt": ["O plano começou em {year} e foi revisto em {year2}.",
                        "Entre {year} e {year2}, o número de sócios dobrou.",
                        "A ponte foi aberta em {year} e reformada em {year2}."],
                 "nl": ["Het plan begon in {year} en loopt nog tot {year2}.",
                        "De brug werd in {year} geopend en in {year2} vernieuwd."],
                 "it": ["Alla riunione c'erano {int} soci e {int2} ospiti.",
                        "Nel cortile abbiamo contato {int} biciclette e {int2} motorini.",
                        "In biblioteca ci sono {int} tavoli e {int2} computer.",
                        "Il pullman aveva {int} posti, ma sono saliti solo {int2} passeggeri.",
                        "Abbiamo piantato {int} alberi nel parco e {int2} lungo il fiume.",
                        "La squadra ha segnato {int} gol in casa e {int2} in trasferta."]}


def load_carriers(lang: str, name: str = "carriers_sym.json") -> list:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)[lang]


def _bad_end(lang, n) -> bool:
    return lang in COUNT_BAD_END and n % 10 in (1, 2) and n % 100 not in (11, 12)


def _grp(n: int, lang: str, p: float, rng: random.Random) -> str:
    """n with its thousands grouped (fr a space, else a dot); below 10 000 only with probability p."""
    if n < 1000 or (n < 10000 and rng.random() > p):
        return str(n)
    return f"{n:,}".replace(",", " " if lang == "fr" else ".")


class Gen:
    """Slot writers of one language. Every generator returns (written slot, value key) or None."""

    def __init__(self, lang: str, rng: random.Random):
        self.lang, self.rng, self.P = lang, rng, T.pools(lang)
        self.used = collections.defaultdict(collections.Counter)
        self.pct = [(n, f) for n, f in self.P["pct"] if n <= 100 and len(f) <= 2]
        self.dec = [(n, f) for n, f in self.P["dec"] if n <= 99 and f and len(f) <= 2]
        self.cur = [(n, f) for n, f in self.P["cur"] if n <= 5000 and len(f) <= 2]
        self.card = sorted(set(self.P["card"]))

    def ok_value(self, typ, key) -> bool:
        u = self.used[typ]
        return u[key] < max(VALUE_MIN, VALUE_SHARE * sum(u.values()))

    def take(self, typ, key):
        self.used[typ][key] += 1

    def _num(self, n, f):
        return f"{n},{f}" if f else str(n)

    def pct_(self, lo=0, hi=100):
        r = self.rng
        if lo == 0 and r.random() < 0.65:  # most percentages people say are small
            hi = 30
        xs = [x for x in self.pct if lo < x[0] <= hi and (x[1] or r.random() < 0.55)]
        if not xs:
            return None
        n, f = r.choice(xs)
        a = self._num(n, f)
        return r.choice(T.PCT_FORMS[self.lang]).format(a=a), a

    def gen(self, typ: str, ctx_before: str, ctx_after: str, state: dict):
        r, lang = self.rng, self.lang
        if typ == "pct":
            return self.pct_()
        if typ == "pa":
            xs = [n for n in self.card if 1 <= n <= 40]
            n = r.choice(xs)
            state["pa"] = n
            return str(n), n
        if typ == "pb":
            lo = state.get("pa", 1)
            return self.pct_(lo, lo + 30)
        if typ == "dec":
            n, f = r.choice(self.dec)
            a = self._num(n, f)
            return a, a
        if typ == "cur":
            if r.random() < 0.8:
                top = r.choices([99, 999, 5000], [0.6, 0.3, 0.1])[0]  # most prices are small
                n, f = r.choice([x for x in self.cur if x[0] <= top] or self.cur)
            else:  # whole prices
                n, f = r.choice([x for x in self.card if 2 <= x <= 999]), ""
            a = self._num(n, f)
            return r.choice(MONEY[lang]).format(a=a), a
        if typ == "curbig":
            n = self._big()
            if n is None:
                return None
            return r.choice(MONEY[lang]).format(a=_grp(n, lang, 0.8, r)), n
        if typ == "mag":
            sc = r.choice(SCALES[lang])
            u = r.random()
            if lang == "es" and u < 0.3:  # "1.800 millones", "3.000 millones" (= mil millones)
                n = r.choice([k * 100 for k in range(11, 100) if k % 10 != 0] + [k * 1000 for k in range(2, 10)])
                x = _grp(n, lang, 1.0, r)
            elif u < 0.6:
                xs = [x for x in self.dec if 2 <= x[0] <= 99 and len(x[1]) == 1]
                n, f = r.choice(xs)
                x = self._num(n, f)
            else:
                while True:
                    n = r.choice([x for x in self.card if 2 <= x <= 999])
                    if not _bad_end(lang, n):
                        break
                x = str(n)
            return f"{x} {sc}", f"{x}|{sc}"
        if typ == "bign":
            n = self._big()
            return (_grp(n, lang, 0.85, r), n) if n else None
        if typ in UNITS:
            u = r.choice(UNITS[typ])
            if typ == "speed":
                n = r.choice([x for x in self.card if 5 <= x <= 300])
                a = str(n)
            elif r.random() < 0.45:
                n, f = r.choice(self.dec)
                a = self._num(n, f)
            else:
                a = str(r.choice([x for x in self.card if 2 <= x <= (999 if u in ("g", "ml", "m", "cm") else 400)]))
            return f"{a} {u}", f"{a}{u}"
        if typ == "temp":
            n = r.randint(-20, 42)
            if n in (0, 1, -1):
                n = 18
            return f"{n} °C" if n >= 0 else f"-{-n} °C", n
        if typ == "phone":
            p = T.g_phone(lang, r, self.P, 0, False, None)
            return (p, p) if p else None
        if typ == "code":
            kind = next((k for k, rx in CODE_CTX if rx.search(ctx_before)), "ref")
            c = r.choice(CODES[kind])
            return c, c
        if typ == "year":
            y = r.choice(range(1985, 2036))
            return str(y), y
        if typ == "date":
            d = T.g_date_txt(lang, r, self.P, 0, False, None)
            return d, re.sub(r"\D+", "-", d).strip("-")
        if typ == "daten":
            d = T.g_date_num(lang, r, self.P, 0, False, None)
            return d, d
        if typ == "ord":
            g = "f" if state.get("re", "").endswith("a$") else "m"
            o = T.g_ord(lang, r, self.P, 0, False, g)
            return o, o
        if typ == "roman":
            if CENTURY_CTX.search(ctx_before):
                n = r.randint(12, 21)
                rn = M._to_roman(n) + ("e" if lang == "fr" else "")
            else:
                rn = f"{r.choice(TT.ROMAN_NAMES[lang])} {M._to_roman(r.randint(2, 16))}{'.' if lang == 'de' else ''}"
            return rn, rn
        if typ == "frac":
            f = r.choice(FRACS)
            return f, f
        if typ in ("ma", "mb", "mc"):
            return str(state["math"][typ]), state["math"][typ]
        if typ == "int":
            if MONTH_CTX.match(ctx_after):
                n = r.randint(2, 28)
            else:
                while True:
                    n = r.choice([x for x in self.card if 2 <= x <= (99 if r.random() < 0.8 else 999)])
                    if not _bad_end(lang, n):
                        break
            return str(n), n
        raise KeyError(typ)

    def _big(self):
        r = self.rng
        for _ in range(50):
            u = r.random()
            if u < 0.45:  # round thousands (thin in real speech for pt / nl / it)
                n = r.choice([k * 1000 for k in range(2, 1000) if not 1100 <= k * 1000 <= 2099])
            elif u < 0.6:
                n = r.choice([k * 100 for k in range(21, 1000) if k % 10])
            else:
                xs = [x for x in self.card if 2100 <= x <= 999999]
                n = r.choice(xs) if xs else None
            if n and not _bad_end(self.lang, n // 1000 if n % 1000 == 0 else n):
                return n
        return None


# ------------------------------------------------------------------------------------------------ realisation

def _year_reading(digits: str, lang: str):
    try:
        sp = M.spans(digits, lang)
    except Exception:  # noqa: BLE001
        return None
    return sp[0].cands[0].strip() if len(sp) == 1 and sp[0].cands else None


def date_piece_ok(lang, s, ok, piece, month_conn):
    """The readings of one piece of a date (0 day, 1 month, 2 year) the sentence may use (the ml_tts.realize date
    rules plus): the year as the year reading of its digits ("neunzehnhundertsiebenundneunzig", never
    "tausendneunhundert..." or a century-dropped form); a numeric date's month by its NAME; es / pt month and year with
    the "de" connector, the year's connector like the month's; es no apocope on the day ("veintiún"); nl no ordinal day
    ("vijftiende maart"); it no "di" before the month; pt "primeiro", never "um", for the 1st; no two-digit year."""
    c = lambda i: s.cands[i].strip()  # noqa: E731
    if re.search(r"\d{4}", s.text):
        yr = _year_reading(re.sub(r"\D", "", s.text), lang)
        ok = [i for i in ok if yr and re.sub(r"^(?:de|del|di|do|da)\s+", "", c(i)) == yr]
        if lang in ("es", "pt") and month_conn:
            want = any(month_conn)
            ok = [i for i in ok if bool(T.CONN.match(s.cands[i])) == want] or ok
        return ok
    if piece >= 2:  # a two-digit written year: left out (a short year reading)
        return []
    if piece == 0:
        if lang == "pt":
            ok = [i for i in ok if c(i) not in ("um", "uma")]  # "primeiro de setembro"
        if lang == "es":
            ok = [i for i in ok if not c(i).endswith("ún")]
        if lang == "nl":
            ok = [i for i in ok if not re.search(r"(?:de|ste)$", c(i))]
        return ok
    if piece == 1:
        ok = [i for i in ok if MONTH_RE[lang].search(s.cands[i])]
        if lang in ("es", "pt"):
            ok = [i for i in ok if T.CONN.match(s.cands[i])] or ok
        if lang == "it":
            ok = [i for i in ok if not c(i).startswith("di ")] or ok
    return ok


# reading conventions as real speech has them (2026-10-18 census of the mix's real rows): es "por cien" 12 of 974
# percent spans, nl "percent" 4 of 361 -> a minority (MINORITY_SHARE); a two-digit decimal fraction read digit by digit
# ("drei Komma eins vier") vs as one number ("drei Komma vierzehn"): the real digit-wise share per language
MINORITY = {"es": re.compile(r"\bpor cien\b"), "nl": re.compile(r"\bpercent\b")}
MINORITY_SHARE = 0.05
DEC2_DIGITWISE = {"de": 0.83, "es": 0.04, "fr": 0.03, "it": 0.03, "nl": 0.06, "pt": 0.03}  # real 68/82, 2/50, 1/65, 0/32, 1/16, 1/83
DEC_WORD_ANY = re.compile(r"\b(?:komma|coma|virgule|virgola|vírgula)\b", re.I)


def dec2_style(lang: str, span_text: str, reading: str):
    """"digitwise" / "number" for a reading of a two-digit decimal fraction, else None."""
    if not re.search(r"\d[,.][1-9]\d(?!\d)", span_text or ""):  # a zero-led fraction (",05") has one reading style
        return None
    m = DEC_WORD_ANY.search(reading)
    if not m:
        return None
    N = _reader()
    frac = []
    for w in reading[m.end():].split():
        got = N.extract_numbers(w, lang)
        if not got:
            break
        frac.append(min(got[0].vals))
    if not frac:
        return None
    return "digitwise" if len(frac) == 2 and all(v < 10 for v in frac) else "number"


def conv_weight(lang: str, s, cand: str) -> float:
    """A reading's weight factor so the planned conventions follow real speech (MINORITY, DEC2_DIGITWISE)."""
    f = 1.0
    if lang in MINORITY and s.kind == "pct" and MINORITY[lang].search(cand):
        f *= MINORITY_SHARE
    st = dec2_style(lang, s.text, cand) if lang in DEC2_DIGITWISE else None
    if st:
        d = DEC2_DIGITWISE[lang]
        f *= (d if st == "digitwise" else 1 - d) * 2
    return f


def convention_trim(rows: list, lang: str, rng) -> tuple:
    """Rows kept so the set's conventions stay near real speech: minority readings (es "por cien", nl "percent") at
    most MINORITY_SHARE of the percent spans, two-digit decimal styles within 0.1 of DEC2_DIGITWISE. A row is dropped
    whole when it carries an over-quota reading. Returns (rows, stats)."""
    rows = list(rows)
    rng.shuffle(rows)
    tags = []
    for r in rows:
        t = collections.Counter()
        for sp in r.get("spans") or []:
            if sp.get("kind") == "pct":
                t["pct_minority" if lang in MINORITY and MINORITY[lang].search(sp["reading"]) else "pct_major"] += 1
            st = dec2_style(lang, sp.get("text", ""), sp.get("reading", "")) if lang in DEC2_DIGITWISE else None
            if st:
                t["dec2_" + st] += 1
        tags.append(t)
    tot = collections.Counter()
    for t in tags:
        tot.update(t)
    allow = {}
    if lang in MINORITY:
        allow["pct_minority"] = int(MINORITY_SHARE * tot["pct_major"] / (1 - MINORITY_SHARE))
    if lang in DEC2_DIGITWISE and (tot["dec2_digitwise"] + tot["dec2_number"]):
        d = DEC2_DIGITWISE[lang]
        dw, nu = tot["dec2_digitwise"], tot["dec2_number"]
        if dw / (dw + nu) > d + 0.1:
            allow["dec2_digitwise"] = int((d + 0.1) * nu / (1 - d - 0.1))
        elif dw / (dw + nu) < d - 0.1:
            allow["dec2_number"] = int((1 - d + 0.1) * dw / (d - 0.1))
    got, keep, dropped = collections.Counter(), [], collections.Counter()
    for r, t in zip(rows, tags):
        over = [k for k in allow if t[k] and got[k] + t[k] > allow[k]]
        if over:
            dropped.update(over)
            continue
        got.update(t)
        keep.append(r)
    return keep, {"before": dict(tot), "allowed": allow, "dropped_rows": dict(dropped), "after": dict(got)}


def realize_mixed(lang, written, slot_ranges, rng, used):
    """ml_tts.realize for a sentence whose spans belong to slots of DIFFERENT classes: each span takes the reading
    rules (reading_ok, EXPECT) of its own slot's class. slot_ranges: [(start, end, slot type)] in `written`.
    Returns ((label, spans, offsets, slot type per span), "") or (None, reason)."""
    try:
        sp = M.spans_regional(written, lang, NAMES)
    except Exception as e:  # noqa: BLE001
        return None, f"spans_error {type(e).__name__}"
    if not sp:
        return None, "no_spans"
    if any(not s.cands for s in sp):
        return None, "missing_grammar" if any(s.note.startswith("missing") for s in sp) else "no_cands"
    if any(c.isdigit() for c in written) and not M.all_digits_spanned(written, lang):
        return None, "digits_unspanned"
    typ, slot_of = [], []
    for s in sp:
        a, b = s.start + (len(s.text) - len(s.text.lstrip())), s.end
        t = [k for k, x in enumerate(slot_ranges) if x[0] <= a and b <= x[1]]
        if len(t) != 1:
            return None, "span_crosses_slot"
        slot_of.append(t[0])
        typ.append(slot_ranges[t[0]][2])
    for k, (a, b, t, _) in enumerate(slot_ranges):
        ks = {s.kind for s, j in zip(sp, slot_of) if j == k}
        if not ks & T.EXPECT[SLOT_CLS[t]]:
            return None, f"shape_not_recognised:{t}"
    style = rng.choices(["digits", "chunks", "mix"], [0.4, 0.35, 0.25])[0]
    picks = []
    T.YEAR_QUOTA = {1: 1}  # years: the full reading only (never the clipped / century-dropped forms)
    try:
        dpos, conn = collections.Counter(), {}
        for s, t, j in zip(sp, typ, slot_of):
            cls = SLOT_CLS[t]
            grp = T.GROUP[cls]
            gender = "f" if (slot_ranges[j][3] or "").endswith("a$") else "m" if cls == "ord" else None
            ok = [i for i, c in enumerate(s.cands) if T.reading_ok(lang, grp, cls, gender, s, c, dpos[j], written)]
            rx = slot_ranges[j][3]
            if s.kind == "date":
                ok = date_piece_ok(lang, s, ok, dpos[j], conn.get(j))
                if dpos[j] == 1 and ok:
                    conn[j] = [bool(T.CONN.match(s.cands[i])) for i in ok]
            elif t in ("ma", "mb", "mc"):  # a bare number in a calculation: its canonical reading (no apocope)
                ok = [i for i in ok if i == 0]
            elif rx and s.kind in CONSTRAINED:
                ok = [i for i in ok if re.search(rx, s.cands[i].strip())]
            if s.kind == "date":
                dpos[j] += 1
            if s.kind == "phone":
                ok = [i for i in ok if T._natural_group(lang, s, s.cands[i])]
            if not ok:
                return None, f"no_reading_fits:{t}"
            if s.kind in ("phone",) and style != "mix":
                want_d = style == "digits"
                ok2 = [i for i in ok if T._digit_style(s, s.cands[i]) == want_d]
                if not want_d:
                    ok2 = [i for i in ok2 if len(s.cands[i].split()) <= 3 and len(re.sub(r"\D", "", s.text)) <= 2] or ok2
                ok = ok2 or ok
            w = [(3.0 if i == 0 else 1.5 if i <= 2 else 1.0) / (1.0 + 0.7 * used[(lang, s.kind, s.cands[i].strip())])
                 * conv_weight(lang, s, s.cands[i]) for i in ok]
            picks.append(rng.choices(ok, w)[0])
    finally:
        T.YEAR_QUOTA = {}
    offs: list = []
    label = C.render_cap(written, sp, picks, offs)
    if written.endswith(".") and not label.endswith("."):
        label += "."
    if T.FORBIDDEN.search(label) or C.WF.search(label):
        return None, "symbol_left"
    toks = label.split()
    flags = C.number_tokens(label, offs)
    for i, tk in enumerate(toks):
        if not flags[i] and C.ABBR_NEAR.match(tk.strip(",;:()")) and (
                (i > 0 and flags[i - 1]) or (i + 1 < len(toks) and flags[i + 1])):
            return None, "abbr_left"
    spans = [{"text": s.text, "kind": s.kind, "reading": s.cands[p].strip(), "canonical": s.cands[0].strip(),
              "idx": p, "n_cands": len(s.cands), "slot": t} for s, p, t in zip(sp, picks, typ)]
    for s, p in zip(sp, picks):
        used[(lang, s.kind, s.cands[p].strip())] += 1
    return (label, spans, offs), ""


# ------------------------------------------------------------------------------------------------ value check

def _caption_value(text: str):
    """The number written in a span's caption text: '3,5 %' -> 3.5, '25.000 €' -> 25000, '1 800' -> 1800."""
    t = re.sub(r"[^\d.,\s]", "", text).strip()
    t = re.sub(r"(?<=\d)[.\s  ](?=\d{3}(?!\d))", "", t)
    t = t.replace(" ", "").replace(",", ".")
    try:
        return _D(t)
    except decimal.InvalidOperation:
        return None


def value_ok(span: dict, lang: str, N) -> tuple:
    """Does the reading give the written value back through the scorer's reader (number_outcomes)?"""
    rd = span["reading"]
    if span["kind"] == "phone":  # the phone words for six and two the reader does not take: pt "meia", de "zwo"
        rd = re.sub(r"\bmeia\b", "seis", rd) if lang == "pt" else re.sub(r"\bzwo\b", "zwei", rd) if lang == "de" else rd
    got = N.extract_numbers(rd, lang)
    vals = [sorted(g.vals)[0] if len(g.vals) == 1 else None for g in got]
    if span["kind"] == "phone":
        want = re.sub(r"\D", "", span["text"])
        alts = ["".join(str(int(v)) for v in vals if v is not None)]
        # "null eins einundsiebzig": each number's digits; a two-digit group read as one number keeps its zero
        return (want in alts, f"phone {want} vs {alts}")
    if span["kind"] == "alnum":
        want = re.sub(r"\D", "", span["text"])
        return ("".join(str(int(v)) for v in vals if v is not None) == want, f"code {want} vs {vals}")
    if span["kind"] == "frac":
        return True, ""  # the grammar table's fraction words; the scorer reads no fractions
    if span["kind"] in ("date", "ord", "roman"):
        if span["kind"] == "roman":
            m = re.search(r"[IVXLC]+", span["text"])
            v = M._roman(m.group(0)) if m else None
            want = _D(v) if v else None
        else:
            dg = re.sub(r"\D", "", span["text"])
            want = _D(int(dg)) if dg else None
        if not got:  # a month name ("Mai"), no number to read back
            return (span["kind"] == "date", f"{span['text']!r} -> no number")
        if want is None:
            return True, ""
        return (any(want in {_D(x) for x in g.vals} for g in got), f"{span['text']!r} -> {vals}")
    want = _caption_value(span["text"])
    if want is None:
        return False, "no caption value"
    sets = [{_D(x) for x in g.vals} for g in got]
    if len(sets) == 1 and want in sets[0]:
        return True, ""
    if span["kind"] == "cur" and sets:
        ip, fp = int(want), int(round((want - int(want)) * 100))
        if len(sets) == 2 and ip in sets[0] and fp in sets[1]:  # "vierzehn Euro neunundneunzig"
            return True, ""
        if ip == 0 and len(sets) == 1 and (fp in sets[0] or want in sets[0]):  # "fünf Cent"
            return True, ""
    return False, f"{span['text']!r} -> {vals}"


def _reader():
    sys.path.insert(0, os.path.join(HERE, "..", "eval", "quality"))
    import number_outcomes as N  # noqa: E402
    return N


# ------------------------------------------------------------------------------------------------ planning

def _math(op: str, rng) -> dict:
    """Operands and result a sentence states: ma (op) mb = mc; > / < compare ma with mb."""
    if op == "+":
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        return {"ma": a, "mb": b, "mc": a + b}
    if op == "-":
        a = rng.randint(12, 99)
        b = rng.randint(2, a - 2)
        return {"ma": a, "mb": b, "mc": a - b}
    if op == "*":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return {"ma": a, "mb": b, "mc": a * b}
    if op == "/":
        b, c = rng.randint(2, 12), rng.randint(2, 12)
        return {"ma": b * c, "mb": b, "mc": c}
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    while a == b:
        b = rng.randint(2, 99)
    lo, hi = min(a, b), max(a, b)
    return {"ma": hi, "mb": lo} if op == ">" else {"ma": lo, "mb": hi}


def fill_template(tpl: str, gen: Gen, forced: dict | None = None):
    """(written, [(start, end, slot type, constraint regex)], value keys) or None."""
    out, ranges, keys, pos, state = "", [], [], 0, {}
    m0 = MATH_OP.match(tpl)
    if m0:
        state["math"] = _math(m0.group(1), gen.rng)
        tpl = tpl[m0.end():]
    for m in SLOT_RE.finditer(tpl):
        typ = m.group(1)
        state["re"] = m.group(3) or ""
        out += tpl[pos:m.start()]
        if forced and m.group(0) in forced:
            v = forced[m.group(0)]
            g = (str(v), v)
        else:
            g = None
            for _ in range(6):
                g = gen.gen(typ, out, tpl[m.end():], state)
                if g and gen.ok_value(typ, g[1]):
                    break
                g = None
        if g is None:
            return None
        ranges.append((len(out), len(out) + len(g[0]), typ, m.group(3)))
        keys.append((typ, g[1]))
        out += g[0]
        pos = m.end()
    out += tpl[pos:]
    return out, ranges, keys


def plan_lang(lang: str, room: dict, tag: str, seed: int, margin: float, rep: dict, max_rows: int = MAX_ROWS,
              carriers: str = "carriers_sym.json", used: dict | None = None, focus: tuple = SYMBOL, thin: bool = True,
              caps: dict | None = None) -> list:
    """used: {class: spans} an earlier plan already takes from the room; caps: {class: spans} upper limits on top of the
    room (math sentences' plain counts); focus: the classes whose targets end the plan when met."""
    N = _reader()
    rng = random.Random(f"{tag}|{seed}|{lang}|{carriers}")
    gen = Gen(lang, rng)
    tpls = load_carriers(lang, carriers)
    used = used or {}
    target = {c: int(max(0, room.get(c, {}).get("room", 0) - used.get(c, 0)) * margin)
              for c in set(SYMBOL) | FILLER | set(EXTRA) | {"clock", "dec_pair"}}
    for c, v in (caps or {}).items():
        target[c] = min(target.get(c, 0), v)
    got = collections.Counter()
    used = collections.Counter()
    seen, rows = set(), []
    drops = collections.Counter()
    bad_values = []

    def need(c):
        return (target.get(c, 0) - got[c]) / max(target.get(c, 1), 1)

    def add_row(written, spans, label, offs, keys, ckey):
        rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
        gcl = [g["gclass"] for g in spans]
        rows.append({"id": rid, "lang": lang, "cls": "sym", "ckey": ckey, "register": "sym", "caption": written,
                     "text": label, "spans": [{k: v for k, v in s.items() if k != "gclass"} for s in spans],
                     "num_chars": offs, "kind": sorted({s["kind"] for s in spans}), "split": "train",
                     "slots": [s["slot"] for s in spans], "gclasses": gcl})
        for g in gcl:
            got[g] += 1
        for typ, k in keys:
            gen.take(typ, k)

    def attempt(tpl, forced=None, ckey=None):
        f = fill_template(tpl, gen, forced)
        if f is None:
            drops["value_spread_or_gen"] += 1
            return False
        written, ranges, keys = f
        if not (6 <= len(written.split()) <= 32):
            drops["length"] += 1
            return False
        res, why = realize_mixed(lang, written, ranges, rng, used)
        if res is None:
            drops[why.split(":")[0]] += 1
            return False
        label, spans, offs = res
        if not label_ok(label):
            drops["label_symbol"] += 1
            return False
        if lang in ELISION_BAD and ELISION_BAD[lang].search(label):
            drops["elision"] += 1
            return False
        if label in seen:
            drops["duplicate"] += 1
            return False
        row = {"text": label}
        cls_n = collections.Counter()
        for s in spans:
            s["gclass"] = CC.gate_class_value(row, s, lang)[0]
            cls_n[s["gclass"]] += 1
        for s in spans:
            ok, why = value_ok(s, lang, N)
            if not ok:
                drops["value_check"] += 1
                if len(bad_values) < 60:
                    bad_values.append({"caption": written, "label": label, "why": why})
                return False
        if forced is None and any(got[c] + k > target.get(c, 0) for c, k in cls_n.items()):
            drops["class_full"] += 1
            return False
        seen.add(label)
        add_row(written, spans, label, offs, keys, ckey or "+".join(sorted(cls_n)))
        return True

    # thin-cell exact values first (two per sentence)
    for typ, vals in (THIN.get(lang, {}) if thin else {}).items():
        bag = [v for v, n in sorted(vals.items()) for _ in range(n)]
        rng.shuffle(bag)
        while bag:
            a = bag.pop()
            b = next((x for x in bag if x != a), None)
            if b is not None:
                bag.remove(b)
            else:
                b = rng.choice([y for y in range(1990, 2026) if y != a]) if typ == "year" else rng.choice([23, 37, 45, 56, 73, 86])
            for _ in range(30):
                tpl = rng.choice(THIN_CARRIERS[lang])
                if attempt(tpl, forced={"{" + typ + "}": a, "{" + typ + "2}": b}, ckey=f"thin_{typ}"):
                    break
            else:
                drops["thin_failed"] += 1

    def weight(tpl):
        ws = []
        for m in SLOT_RE.finditer(tpl):
            g = SLOT_GUESS[m.group(1)]
            if g is None:
                ws.append(0.3)
            elif g in FILLER and g not in focus:
                ws.append(0.15 if need(g) > 0 else 0.0)
            else:
                if need(g) <= 0:
                    return 0.0
                ws.append(need(g))
        return sum(ws) / max(len(ws), 1)

    tries = 0
    while len(rows) < max_rows and tries < 60 * max_rows:
        ws = [weight(t) for t in tpls]
        if not any(w > 0 for w in ws) or all(need(c) <= 0.02 for c in focus if target.get(c)):
            break
        tries += 1
        attempt(rng.choices(tpls, ws)[0])
    rep.update(rows=len(rows), target={c: v for c, v in target.items() if v}, got=dict(got), drops=dict(drops),
               bad_values=bad_values[:25], tries=tries)
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------------------------------------ replacing rows

REPL_CLASSES = {"nl": ("cur", "phone", "dec", "int_big"), "it": ("cur", "phone", "dec", "int_big", "roman"),
                "de": ("phone", "roman"), "pt": ("int_big",)}
REPLACING_PREFIX = ("ttsRV1",)  # rows of the existing replacing sets are never superseded again


def pure_rows(mix_rows: list, lang: str) -> dict:
    """class -> [(id, spans)] of the mix's synthetic rows whose every span is in that one census class."""
    out = collections.defaultdict(list)
    for r in mix_rows:
        if str(r.get("id", "")).startswith(REPLACING_PREFIX):
            continue
        cs = [CC.gate_class_value(r, sp, lang)[0] for sp in r.get("spans") or []]
        if cs and len(set(cs)) == 1:
            out[cs[0]].append((r["id"], len(cs)))
    return out


def assign_supersedes(rows: list, pool: dict, rng) -> dict:
    """Each new row supersedes old single-class rows of its class until their spans reach the row's spans of that
    class (a row that cannot be covered is dropped). Returns stats; rows get "supersedes"."""
    st = collections.Counter()
    for c in pool:
        rng.shuffle(pool[c])
    keep = []
    for r in rows:
        n = collections.Counter(r["gclasses"])
        c = next((k for k in n if k in pool), None)
        if c is None:
            st["no_class"] += 1
            continue
        got, ids = 0, []
        while got < n[c] and pool[c]:
            i, k = pool[c].pop()
            ids.append(i)
            got += k
        if got < n[c]:
            st[f"uncovered_{c}"] += 1
            for i in ids:  # nothing is superseded for a row that is dropped
                pool[c].append((i, 1))
            continue
        r["supersedes"] = ids
        st[f"rows_{c}"] += 1
        st[f"new_spans_{c}"] += n[c]
        st[f"old_spans_{c}"] += got
        keep.append(r)
    rows[:] = keep
    return dict(st)


# ------------------------------------------------------------------------------------------------ English

EN_KOKORO = ("af_heart", "af_bella", "af_sarah", "af_sky", "am_adam", "am_michael", "am_onyx", "bf_emma",
             "bf_isabella", "bf_lily", "bm_george", "bm_daniel")  # ml_tts_fill.EN_KOKORO_TRAIN (dev voices left out)
EN_SLOT_RE = re.compile(r"\{([a-z_0-9]+?)(\d?)\}")
# English readings (readings.KINDS, the English label rules) a sentence may not use: generator slips ("a eighth",
# "three halves", "two quarters") and quarter forms that do not fit after "Our"
EN_BAD = {"fraction": re.compile(r"\ba [aeiou]|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven) (?:halves|quarters|thirds)\b"
                                 r"|\bthree thirds|\bfour quarters|\btwo halves"),
          "quarter": re.compile(r"^the\b"),
          # a price with no currency word ("four thousand twenty nine": readings.k_money's bare branch)
          "money": re.compile(r"^(?!.*\b(?:dollars?|euros?|pounds?|pence|penny|cents?|yen|yuan|rupees?|francs?|kronor|krona|"
                              r"zlotys?|pesos?|centavos?|bucks|rand)\b)")}
# no label may hold a digit, a symbol or a slot left unfilled (2026-10-18 audit: "{int_4digit}" rows)
LABEL_BAD = re.compile(r"[{}0-9$%€£#@+_/]|[−×=]")


def label_ok(text: str) -> bool:
    return not LABEL_BAD.search(text)


def _en_year_fwd(r, R):
    y = r.randint(2026, 2060)
    return R.year(y) if r.random() < 0.75 else R.cardinal(y)


def _en_money_x(r, R):
    """Prices in other English-speaking currencies: Australian / Canadian dollars, pounds and pence, rupees, euros."""
    n = r.randint(2, 9999) if r.random() < 0.7 else r.randint(10, 999)
    unit = r.choice(["Australian dollars", "Canadian dollars", "pounds", "pounds", "rupees", "euros", "rand"])
    if unit == "pounds" and r.random() < 0.4:
        return f"{R.cardinal(n)} pounds {R.cardinal(r.randint(1, 99))}" if r.random() < 0.5 else \
            f"{R.cardinal(n)} pounds and {R.cardinal(r.randint(2, 99))} pence"
    return f"{R.cardinal(n)} {unit}"


def _en_decimal_uk(r, R):
    """UK "nought point" decimals below one."""
    return "nought point " + R.digits(str(r.randint(1, 99)))


EN_CUSTOM = {"year_fwd": _en_year_fwd, "money_x": _en_money_x, "decimal_uk": _en_decimal_uk}


def plan_en(n: int, tag: str, seed: int, rep: dict, carriers=("en",), avoid=frozenset()) -> list:
    """English rows: carriers_sym2.json "en" sentences, each slot a readings.KINDS reading (the English label rules:
    spoken form, years as words); math slots ma / mb / mc are readings.cardinal of the computed values. The label is the
    readings; the synthesiser gets readings.tts_form (letter names, "ex")."""
    import readings as R

    rng = random.Random(f"{tag}|{seed}|en")
    tpls = [t for c in carriers for t in load_carriers(c, "carriers_sym2.json")]
    used = collections.defaultdict(collections.Counter)
    rows, seen, drops = [], set(), collections.Counter()
    tries = 0
    while len(rows) < n and tries < 40 * n:
        tries += 1
        tpl = rng.choice(tpls)
        m0 = MATH_OP.match(tpl)
        mv = _math(m0.group(1), rng) if m0 else {}
        body = tpl[m0.end():] if m0 else tpl
        text, tts, offs, kinds, ok = "", "", [], [], True
        pos = 0
        for m in EN_SLOT_RE.finditer(body):
            k = m.group(1)
            text += body[pos:m.start()]
            tts += body[pos:m.start()]
            if k in ("ma", "mb", "mc"):
                slot = R.cardinal(mv[k])
            elif k in EN_CUSTOM:
                slot = None
                for _ in range(8):
                    x = EN_CUSTOM[k](rng, R)
                    u = used[k]
                    if u[x.lower()] < max(VALUE_MIN, VALUE_SHARE * sum(u.values())):
                        slot = x
                        break
                if slot is None:
                    ok = False
                    break
            else:
                slot = None
                for _ in range(8):
                    x = R.KINDS[k](rng)
                    if EN_BAD.get(k) and EN_BAD[k].search(x.replace("|", " ")):
                        continue
                    u = used[k]
                    if u[x.replace("|", " ").lower()] < max(VALUE_MIN, VALUE_SHARE * sum(u.values())):
                        slot = x
                        break
                if slot is None:
                    ok = False
                    break
            w = slot.replace("|", " ")
            if not text:  # a sentence that opens on the number
                w = w[:1].upper() + w[1:]
            offs.append([len(text), len(text) + len(w)])
            text += w
            tts += (R.tts_form(slot).replace("|", ", ") if k not in ("ma", "mb", "mc") else slot)
            if not tts.strip() or tts.strip() == w:
                pass
            kinds.append(k)
            pos = m.end()
        if not ok:
            drops["value_spread"] += 1
            continue
        text += body[pos:]
        tts += body[pos:]
        tts = tts[:1].upper() + tts[1:]
        if text in seen or text in avoid:
            drops["duplicate"] += 1
            continue
        if not label_ok(text):
            drops["label_symbol"] += 1
            continue
        seen.add(text)
        for k, (a, b) in zip(kinds, offs):
            if k not in ("ma", "mb", "mc"):
                used[k][text[a:b].lower()] += 1
        rid = f"{tag}_en_{hashlib.md5(f'{seed}|{text}'.encode()).hexdigest()[:10]}"
        rows.append({"id": rid, "lang": "en", "cls": "sym", "ckey": "+".join(sorted(set(kinds))), "template": tpl,
                     "kinds": kinds, "text": text, "synth_text": tts, "num_chars": offs,
                     "spans": [{"kind": k, "reading": text[a:b], "slot": k} for k, (a, b) in zip(kinds, offs)],
                     "gclasses": kinds, "split": "train"})
    rep.update(rows=len(rows), drops=dict(drops), tries=tries, kinds=dict(collections.Counter(k for r in rows for k in r["kinds"])))
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------------------------------------ voices

KOKORO_IT = ["if_sara", "im_nicola", "blend:i:if_sara*0.6+ef_dora*0.4", "blend:i:if_sara*0.6+pf_dora*0.4",
             "blend:i:im_nicola*0.6+em_alex*0.4", "blend:i:if_sara*0.7+ff_siwis*0.3",
             "blend:i:im_nicola*0.7+em_santa*0.3", "blend:i:im_nicola*0.7+pm_santa*0.3",
             "blend:i:if_sara*0.5+pf_dora*0.25+ef_dora*0.25", "blend:i:im_nicola*0.5+em_alex*0.25+pm_santa*0.25"]
ENGINE = {"es": "qwen3", "pt": "qwen3", "fr": "qwen3", "de": "chatterbox", "it": "kokoro", "nl": "chirp", "en": "kokoro"}
LICENCE = {"qwen3": "Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0); reference speaker {src} ({lic})",
           "chatterbox": "Chatterbox Multilingual V3 (MIT); reference speaker {src} ({lic})"}


def assign_voices(rows: list, lang: str, vtag: str, refs: list | None = None, cloud: list | None = None) -> None:
    """Voice x rate per row, round robin over the voices in a seeded order (equal rows per voice)."""
    from mix_build import heldout_persona, is_heldout_voice

    eng = ENGINE[lang]
    rng = random.Random(f"{vtag}|voices|{lang}")
    if eng == "kokoro":
        pool = list(EN_KOKORO if lang == "en" else KOKORO_IT)
    elif eng == "chirp":
        pool = [v for v in cloud if not is_heldout_voice(lang, v[0]) and heldout_persona(v[0]) != "Charon"]
    else:
        pool = list(refs)
    assert len(pool) >= 10 or eng in ("kokoro",), f"{lang}: only {len(pool)} voices"
    order = list(range(len(rows)))
    rng.shuffle(order)
    for k, i in enumerate(order):
        r, v, sp = rows[i], pool[k % len(pool)], rng.choice(RATES)
        if eng == "kokoro":
            r.update(engine="kokoro", engine_src="kokoro", voice=v, voice_license=T.KOKORO, voice_path=None,
                     speaker=None, speed=sp, pitch=1.0)
            if lang == "en":  # English: synthesised + CTC-checked in ml_tts_open.en_verify (presynth path filled there)
                r["presynth"] = f"{VOL}/ml/{vtag}/{lang}/presynth/{r['id']}.wav"
        elif eng == "chirp":
            r.update(engine="presynth", engine_src="chirp3hd", voice=v[0], locale=v[1],
                     voice_license="Google Cloud Text-to-Speech output", voice_path=None, speaker=None, speed=sp,
                     pitch=1.0, presynth=f"{VOL}/ml/{vtag}/{lang}/presynth/{r['id']}.wav")
        else:
            r.update(engine="presynth", engine_src=eng, voice=f"{eng}:{lang}:{v['spk']}", ref_wav=v["wav"],
                     ref_text=v["text"], voice_license=LICENCE[eng].format(src=v["source"], lic=v["license"]),
                     voice_path=None, speaker=None, speed=sp, pitch=1.0,
                     presynth=f"{VOL}/ml/{vtag}/{lang}/presynth/{r['id']}.wav")
    assert not any(is_heldout_voice(lang, r["voice"]) for r in rows)


def assign_voices_en2(rows: list, refs: list, vtag: str) -> dict:
    """English cloned voices: within each accent the references alternate Qwen3 / Chatterbox (both engines on every
    accent); rows round robin over the references in a seeded order, rates 0.9-1.25."""
    rng = random.Random(f"{vtag}|voices|en2")
    by = collections.defaultdict(list)
    for v in sorted(refs, key=lambda v: v["spk"]):
        by[v["accent"]].append(v)
    pool, k = [], 0
    for acc in sorted(by):  # one running toggle: each accent gets both engines, the totals stay even
        for v in by[acc]:
            pool.append(dict(v, engine=("qwen3", "chatterbox")[k % 2]))
            k += 1
    order = list(range(len(rows)))
    rng.shuffle(order)
    for k, i in enumerate(order):
        r, v = rows[i], pool[k % len(pool)]
        r.update(engine="presynth", engine_src=v["engine"], voice=f"{v['engine']}:en:{v['spk']}", ref_wav=v["wav"],
                 ref_text=v["text"], accent=v["accent"],
                 voice_license=LICENCE[v["engine"]].format(src=v["source"], lic=v["license"]), voice_path=None,
                 speaker=None, speed=rng.choice(RATES), pitch=1.0, presynth=f"{VOL}/ml/{vtag}/en/presynth/{r['id']}.wav")
    return dict(collections.Counter((r["engine_src"], r["accent"]) for r in rows))


# ------------------------------------------------------------------------------------------------ finalize

def _fold(w: str) -> str:
    import unicodedata
    w = unicodedata.normalize("NFKD", w.lower())
    return "".join(c for c in w if c.isalpha())


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def respell(label: str, greedy: str, lang: str, N) -> str:
    """The greedy text with its reader-spelling slips undone: a greedy word that is NO number word and lies within 1
    letter (2 for >= 7 letters) of exactly one number word / decimal / percent word of the label is written as that
    label word ("vírgola" -> "vírgula", "ero" -> "zero"). A greedy number word is never touched, so a different
    number ("milhões" for "bilhões") still disagrees."""
    num = lambda w: bool(N.extract_numbers(w, lang))  # noqa: E731
    lw = {}
    for w in label.split():
        f = _fold(w)
        if f and (num(w) or f in DEC_WORDS.get(lang, set())):
            lw[f] = w.strip(".,;:!?").lower()
    every = {_fold(w) for w in label.split()}
    out = []
    for g in greedy.split():
        f = _fold(g)
        if f and f not in every and not num(g):  # a word the label has as written (pt "dos" of) stays
            near = {v for k, v in lw.items() if _lev(f, k) <= (2 if len(k) >= 7 else 1)}
            if len(near) == 1:
                g = near.pop()
        out.append(g)
    return " ".join(out)


DEC_WORDS = {"de": {"komma", "prozent"}, "nl": {"komma", "procent", "percent"}, "es": {"coma", "por", "ciento", "cien"},
             "fr": {"virgule", "pour", "cent"}, "it": {"virgola", "per", "cento"}, "pt": {"virgula", "por", "cento"},
             "en": {"point", "percent"}}


def numbers_agree(label: str, greedy: str, lang: str, N, readings=()) -> bool:
    """The CTC reader's greedy text holds the label's numbers, in order, each with a value of the label's number
    (number_outcomes on both sides, after respell), and every word of each reading that is no number word (the unit,
    decimal, percent, currency word: "Prozent", "coma", "por cien", "Euro") as a greedy word: a spelling slip in a
    long number word passes; a different number or a dropped symbol word does not."""
    g = respell(label, greedy, lang, N)
    a, b = N.extract_numbers(label, lang), N.extract_numbers(g, lang)
    if not (len(a) == len(b) and all(x.vals & y.vals for x, y in zip(a, b))):
        return False
    have = {_fold(w) for w in g.replace("-", " ").split()}
    for rd in readings:
        for w in rd.split():
            if _fold(w) and not N.extract_numbers(w, lang) and _fold(w) not in have:
                return False
    return True


def spread_trim(rows: list, rng, share: float = VALUE_SHARE, min_n: int = VALUE_MIN, skip=("ma", "mb", "mc")) -> tuple:
    """Rows kept so no reading of a slot kind passes `share` of that kind's spans (after min_n); a row carrying an
    over-share reading is dropped whole. English rows (spans carry their slot kind and reading). Math operands are
    exempt (their range is the arithmetic's)."""
    rows = list(rows)
    rng.shuffle(rows)
    tot = collections.Counter(sp["kind"] for r in rows for sp in r["spans"] if sp["kind"] not in skip)
    got, keep, dropped = collections.defaultdict(collections.Counter), [], collections.Counter()
    for r in rows:
        ks = [(sp["kind"], sp["reading"].lower()) for sp in r["spans"] if sp["kind"] not in skip]
        if any(got[k][v] + 1 > max(min_n, share * tot[k]) for k, v in ks):
            dropped[next(k for k, v in ks if got[k][v] + 1 > max(min_n, share * tot[k]))] += 1
            continue
        for k, v in ks:
            got[k][v] += 1
        keep.append(r)
    return keep, {"dropped_by_kind": dict(dropped), "kept": len(keep), "in": len(rows)}


def finalize(rows: list, lang: str) -> tuple:
    """CTC-verified rows -> the set's rows. An exact fit of every reading is kept; a fuzzy fit (ml_tts.check_greedy
    allows a few letters off in a long reading, which could hide a misread digit) only when numbers_agree on the
    greedy text the verification kept (keep_greedy); rows without it are dropped."""
    N = _reader()
    keep, st = [], collections.Counter(n_in=len(rows))
    for r in rows:
        if not label_ok(r["text"]) or (lang == "en" and any(sp.get("kind") == "money" and EN_BAD["money"].search(sp["reading"])
                                                            for sp in r.get("spans") or [])):
            st["label_rejected"] += 1
            continue
        if r.get("fit") == "exact":
            st["exact"] += 1
        elif r.get("greedy") and numbers_agree(r["text"], r["greedy"], lang, N, [sp["reading"] for sp in r["spans"]]):
            st["fuzzy_numbers_agree"] += 1
        else:
            st["fuzzy_dropped"] += 1
            continue
        keep.append({k: v for k, v in r.items() if k != "greedy"} | {"src": "tts"})
    st["kept"] = len(keep)
    return keep, dict(st)


# ------------------------------------------------------------------------------------------------ CLI

def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--room", required=True)
    p.add_argument("--tag", default="ttsSYM1")
    p.add_argument("--out", required=True)
    p.add_argument("--langs", default="de,es,fr,it,nl,pt")
    p.add_argument("--margin", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--max-rows", type=int, default=MAX_ROWS)
    p.add_argument("--carriers", default="carriers_sym.json")
    p.add_argument("--used-dir", default="", help="an earlier plan's planned_<lang>.jsonl: its spans leave the room")
    p.add_argument("--focus", default=",".join(SYMBOL))
    p.add_argument("--no-thin", action="store_true")
    p.add_argument("--caps", default="", help='json {lang: {class: spans}} limits on top of the room')
    v = sub.add_parser("voices")
    v.add_argument("--planned", required=True)
    v.add_argument("--refs", default="")
    v.add_argument("--vtag", default="ttsSYM1v")
    v.add_argument("--out", required=True)
    c = sub.add_parser("chirp")
    c.add_argument("--planned", required=True)
    c.add_argument("--vtag", default="ttsSYM1v")
    c.add_argument("--work", default="/tmp/tts_sym")
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--dry", action="store_true")
    q = sub.add_parser("plan-repl")
    q.add_argument("--mix", required=True, help="dir with the built mix's tts_<lang>.jsonl")
    q.add_argument("--tag", default="ttsSYM1r")
    q.add_argument("--seed", type=int, default=20)
    q.add_argument("--out", required=True)
    q.add_argument("--langs", default="nl,it,de,pt")
    e = sub.add_parser("plan-en")
    e.add_argument("--n", type=int, default=3000)
    e.add_argument("--tag", default="ttsSYM1")
    e.add_argument("--seed", type=int, default=18)
    e.add_argument("--out", required=True)
    e.add_argument("--carriers", default="en", help="carrier keys in carriers_sym2.json (en,en2)")
    e.add_argument("--avoid", default="", help="planned rows of an earlier English set (no repeated sentence)")
    f = sub.add_parser("finalize")
    f.add_argument("--rows", required=True, help="CTC-verified rows.jsonl (several: comma-separated)")
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        room = json.load(open(a.room))
        os.makedirs(a.out, exist_ok=True)
        summary = {}
        for lang in a.langs.split(","):
            rep: dict = {}
            used = collections.Counter(g for r in _rows(os.path.join(a.used_dir, f"planned_{lang}.jsonl"))
                                       for g in r["gclasses"]) if a.used_dir else None
            caps = json.loads(a.caps).get(lang) if a.caps else None
            rows = plan_lang(lang, room[lang], a.tag, a.seed, a.margin, rep, a.max_rows, a.carriers, used,
                             tuple(a.focus.split(",")), not a.no_thin, caps)
            rep["chars"] = sum(len(r["text"]) for r in rows)
            summary[lang] = rep
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            print(lang, json.dumps({k: rep[k] for k in ("rows", "chars", "got", "drops")}, ensure_ascii=False))
        json.dump(summary, open(os.path.join(a.out, "plan_report.json"), "w"), ensure_ascii=False, indent=1)
    elif a.cmd == "voices":
        rows = _rows(a.planned)
        lang = rows[0]["lang"]
        refs = json.load(open(a.refs))["refs"] if a.refs else None
        cloud = None
        if ENGINE[lang] == "chirp":
            import ml_tts_cloud as CL
            cloud = list(CL.voices(CL.Auth(), lang))
        assign_voices(rows, lang, a.vtag, refs, cloud)
        _write(a.out, rows)
        print(lang, len(rows), collections.Counter(r["voice"] for r in rows).most_common(3), len({r["voice"] for r in rows}))
    elif a.cmd == "chirp":
        _chirp(a)
    elif a.cmd == "plan-repl":
        os.makedirs(a.out, exist_ok=True)
        summary = {}
        for lang in a.langs.split(","):
            pool = pure_rows(_rows(os.path.join(a.mix, f"tts_{lang}.jsonl")), lang)
            room = {c: {"room": sum(k for _, k in pool[c])} for c in REPL_CLASSES[lang]}
            rep: dict = {}
            rows = plan_lang(lang, room, a.tag, a.seed, 1.0, rep, MAX_ROWS, "carriers_sym3.json", None,
                             REPL_CLASSES[lang], False, {c: 0 for c in ("int_small", "year", "date", "ord")})
            rows = [r for r in rows if set(r["gclasses"]) <= set(REPL_CLASSES[lang])]
            rep["supersede"] = assign_supersedes(rows, {c: pool[c] for c in REPL_CLASSES[lang]}, random.Random(a.seed))
            rep["rows_final"] = len(rows)
            rep["chars"] = sum(len(r["text"]) for r in rows)
            summary[lang] = rep
            _write(os.path.join(a.out, f"planned_{lang}.jsonl"), rows)
            print(lang, json.dumps({k: rep[k] for k in ("rows_final", "chars", "got", "supersede")}))
        json.dump(summary, open(os.path.join(a.out, "plan_report.json"), "w"), ensure_ascii=False, indent=1)
    elif a.cmd == "plan-en":
        rep: dict = {}
        rows = plan_en(a.n, a.tag, a.seed, rep, tuple(a.carriers.split(",")),
                       frozenset(r["text"] for r in _rows(a.avoid)) if a.avoid else frozenset())
        os.makedirs(a.out, exist_ok=True)
        _write(os.path.join(a.out, "planned_en.jsonl"), rows)
        rep["chars"] = sum(len(r["synth_text"]) for r in rows)
        print(json.dumps(rep))
    elif a.cmd == "finalize":
        rows = [x for p in a.rows.split(",") for x in _rows(p)]
        keep, st = finalize(rows, rows[0]["lang"])
        _write(a.out, keep)
        print(json.dumps(st))


def _chirp(a):
    """nl: Chirp 3 HD synthesis of voiced rows (16 kHz LINEAR16 at the row's rate), then upload of the clips + the
    planned rows to NEW volume paths (an existing path is an error)."""
    import base64
    import subprocess
    import time
    from concurrent.futures import ThreadPoolExecutor

    import ml_tts_cloud as CL

    rows = _rows(a.planned)
    lang = rows[0]["lang"]
    chars = sum(len(r["text"]) for r in rows)
    print(json.dumps({"rows": len(rows), "chars": chars, "usd_list": round(chars / 1e6 * 30, 2)}))
    if a.dry:
        return
    auth = CL.Auth()
    w = os.path.join(a.work, a.vtag, lang)
    os.makedirs(w, exist_ok=True)

    def synth(r):
        p = os.path.join(w, r["id"] + ".wav")
        if os.path.exists(p):
            return True
        body = {"input": {"text": r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
        for attempt in range(4):
            try:
                d = CL._req(auth, f"{CL.API}/text:synthesize", body)
                open(p, "wb").write(base64.b64decode(d["audioContent"]))
                return True
            except Exception:  # noqa: BLE001
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(a.threads) as ex:
        ok = list(ex.map(synth, rows))
    rows = [r for r, k in zip(rows, ok) if k]
    _write(os.path.join(w, "planned.jsonl"), rows)
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", w + "/", f"ml/{a.vtag}/{lang}/presynth/"],
                   cwd=HERE, check=True, capture_output=True)
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                    f"ml/{a.vtag}/{lang}/planned.jsonl"], cwd=HERE, check=True, capture_output=True)
    print(json.dumps({"uploaded_rows": len(rows), "cloud_failed": ok.count(False), "chars": chars}))


if __name__ == "__main__":
    main()
