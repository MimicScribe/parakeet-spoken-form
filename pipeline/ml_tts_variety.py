"""Reading-variety fill (2026-10-18): the synthetic numbers read the other ways people say them.

The census of the built mix (reading_conv) found that the synthetic rows use one reading convention where real speech
uses several ("vierzehn Uhr dreißig" but never "halb drei", years never in pairs in Dutch, phones split 2+1, prices
always with the unit word, ...). A plan (variety_plan_rv1.json) gives per language x cell the target share of each
convention. This module:

  1. counts the conventions of the mix's synthetic units per cell (reading_conv.units / classify, the census's code);
  2. "replace" cells: picks rows of over-represented conventions to drop (rows whose every number is in the cell) and
     re-reads their values with the conventions the cell lacks (same values where the convention allows: a year
     keeps its value, a clock time keeps its minutes), so class sizes and value spread stay; "grow" cells add rows;
  3. writes two numbers of one cell per sentence (two carrier clauses of ml_tts_templates joined by "and": half the
     characters per number of a one-slot sentence), realised through ml_tts.realize with the fill readings
     (ml_cands.FILL_ALL + the regional readings) and a pick that forces each number's convention;
  4. checks every planned row with reading_conv (each number has the convention it was planned with) and records the
     ids it replaces ("supersedes"; mix_build.TTS_REPLACING drops them at build time when the row survives);
  5. voices rows with Chirp 3 HD personas x regional locales x speaking rates 0.9-1.25 (never the held-out persona,
     never the seam bed's second voice), uploads clips + planned rows to NEW volume paths (an existing path is an
     error, never --force). Verification as for every cloud set: ml_tts.tts_lang --rows-file (CTC check), then the
     second-transcript value judge. Degradation (noise / reverb / phone band) is applied by mix_build like every set.

    python ml_tts_variety.py plan --mix <dir with tts_<lang>.jsonl> --tag ttsRV1 --out <dir> [--langs de,nl]
    python ml_tts_variety.py synth --planned <dir>/planned_<lang>.jsonl --vtag ttsRV1p --pilot 30
    modal run ml_tts.py --langs <lang> --out-tag <vtag> --rows-file "ml/<vtag>/{lang}/planned.jsonl"
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import re
import subprocess

import ml_cands as M
import ml_tts as T
import ml_tts_templates as TT
import reading_conv as RC

HERE = os.path.dirname(os.path.abspath(__file__))
VOL = "/vol"
# the readings offered: relative clock times + the fill readings (not the Belgian / Swiss / European Portuguese /
# Austrian regional words: the voices are fr-FR / fr-CA and mostly pt-BR)
NAMES = frozenset({"clock_rel"}) | M.FILL_ALL
RATES = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
CONN = {"de": "und", "nl": "en", "es": "y", "fr": "et", "it": "e", "pt": "e"}
WORDS_MAX = 30  # written words of a two-clause sentence
# the conventions counted for prices are the ones with cents (whole / scale prices are not part of the plan)
CENT_CONVS = {"unit+cents", "unit+join+cents", "unit+join+cents+centword", "unit+cents+centword", "decimal_word+unit",
              "decimal_word", "bare_joiner", "bare_no_unit", "cents_only"}
# the seam bed's second voice (an evaluation voice; the held-out persona is mix_build.HELDOUT_VOICES)
EVAL_PERSONAS = {"Charon"}
EXTRA_LOCALES = {"pt": ["pt-PT"], "de": ["de-AT"]}  # asked for too; used where the service has Chirp 3 HD voices
GROUP = {"clock": "TIME", "cur": "MONEY", "phone": "PHONE", "year": "YEAR", "dec": "QTY", "pct": "QTY",
         "int_big": "QTY", "range": "QTY"}
TTS_CLS = {"clock": "clock24", "cur": "price", "phone": "phone", "year": "year", "dec": "dec", "pct": "pct",
           "int_big": "large", "range": "range_between"}
REL_CONVS = {"half_next", "half_past", "quarter_past", "quarter_to", "near_half", "min_past", "min_to", "oclock",
             "oclock_hourword"}


def cell_of(u: dict) -> str | None:
    return RC.cell(None, u)


# ------------------------------------------------------------------------------------------------ census of a mix

def census(rows: list, lang: str) -> dict:
    """cell -> {"convs": Counter, "units": [(row id, unit index, conv, unit)], "pure": {row id}}: the synthetic units
    of each cell and the rows whose every number is in that one cell."""
    out = collections.defaultdict(lambda: {"convs": collections.Counter(), "units": [], "pure": set()})
    for r in rows:
        us = RC.units(r, lang)
        cells = []
        for k, u in enumerate(us):
            c = cell_of(u)
            cells.append(c)
            if c is None:
                continue
            conv, _ = RC.classify(lang, u)
            if conv is None:
                continue
            if c == "cur" and conv not in CENT_CONVS:
                continue
            out[c]["convs"][conv] += 1
            out[c]["units"].append((r["id"], k, conv, u))
        if cells and len(set(cells)) == 1 and cells[0] is not None:
            out[cells[0]]["pure"].add(r["id"])
    return out


def plan_counts(cell: dict, cen: dict) -> tuple:
    """(need {conv: n}, excess {conv: n}) of a plan cell against the mix census of that cell."""
    have = cen["convs"]
    tgt = cell["target"]
    tot = sum(tgt.values())
    if cell["mode"] == "grow":
        return {c: round(cell["add"] * s / tot) for c, s in tgt.items() if round(cell["add"] * s / tot)}, {}
    n = sum(have.values())
    want = {c: round(n * s / tot) for c, s in tgt.items()}
    need = {c: w - have.get(c, 0) for c, w in want.items() if w > have.get(c, 0)}
    excess = {c: have[c] - want.get(c, 0) for c in have if have[c] > want.get(c, 0)}
    return need, excess


# ------------------------------------------------------------------------------------------------ values per convention

def _clock_minutes(lang: str, conv: str) -> list:
    return {"half_next": [30], "half_past": [30], "quarter_past": [15], "quarter_to": [45], "oclock": [0],
            "oclock_hourword": [0],
            "near_half": [25, 35] if lang == "de" else [20, 25, 35, 40],
            "min_past": [5, 10, 20] if lang == "de" else [5, 10],
            "min_to": ([40, 50, 55] if lang == "de" else [50, 55] if lang == "nl" else [35, 40, 50, 55]),
            "noon_midnight": [0, 15, 30], "digital_hourword": [5, 10, 20, 25, 35, 40, 50, 55],
            "digital_joiner": [5, 10, 20, 25, 35, 40, 50, 55], "digital_bare": [5, 10, 20, 25, 35, 40, 50, 55]}[conv]


TO_CONVS = {"half_next", "quarter_to", "near_half", "min_to"}  # said with the NEXT hour


def _clock_hour_ok(lang: str, conv: str, h: int, m: int) -> bool:
    """Hours 6-22. es / pt / it carriers say "a las" / "às" / "alle": never an hour said "one" (la una, a uma);
    fr / it / pt 0:xx and 12:xx are noon / midnight readings (their own convention)."""
    if conv == "noon_midnight":
        return h == 12 or (h == 0 and lang in ("fr", "it", "pt"))
    if h in (0, 12) and lang in ("fr", "it", "pt", "es"):
        return False
    if lang in ("es", "pt", "it") and (h % 12 == 1 or (conv in TO_CONVS and (h + 1) % 12 == 1)):
        return False
    return 6 <= h <= 22


def _clock_value(u):
    m = re.search(r"(\d{1,2})\s*[:h.]\s*(\d{2})", u["written"])
    return (int(m.group(1)), int(m.group(2))) if m else None


def compatible(lang: str, cell: str, conv: str, u: dict) -> bool:
    """Can the dropped unit u's value be read with conv?"""
    cls = cell.split(":")[0]
    if cls == "year":
        v = u.get("val")
        if not isinstance(v, int):
            return False
        lo = v % 100
        return {"pairs": lo >= 10, "pairs_zero": 1 <= lo <= 9, "two_digit": lo >= 10 and v >= 1900,
                "hundreds": 1100 <= v <= 1999 and lang in ("de", "nl", "fr"), "thousands": True}.get(conv, False)
    if cls == "clock":
        hm = _clock_value(u)
        return bool(hm) and hm[1] in _clock_minutes(lang, conv) and _clock_hour_ok(lang, conv, *hm)
    if cls == "cur":
        m = re.search(r"(\d+)[.,](\d{2})(?!\d)", u["written"])
        if not m or conv == "cents_only":
            return False
        # a price said without its unit is a small price ("zwei fünfzig", "vinte e três e noventa")
        small = int(m.group(1)) <= 99 or conv not in ("bare_no_unit", "bare_joiner")
        return int(m.group(2)) >= 10 and int(m.group(1)) >= 1 and small
    if cls == "dec":
        m = re.search(r"(\d+)[.,](\d+)", u["written"])
        if not m:
            return False
        f = m.group(2)
        return {"comma_1digit": len(f) == 1, "point_1digit": len(f) == 1, "half_word": f == "5",
                "comma_as_number": len(f) == 2 and f[0] != "0" and f[1] != "0", "con": len(f) <= 2,
                "comma_digitwise": len(f) == 2}.get(conv, False)
    return False


# ------------------------------------------------------------------------------------------------ written slots

def _phone_written(lang: str, conv: str, rng) -> str:
    """A written phone number whose groups can be read with conv (the pick does the reading)."""
    d = lambda k: "".join(str(rng.randint(0, 9)) for _ in range(k))  # noqa: E731
    nz = lambda k: str(rng.randint(1, 9)) + d(k - 1)  # noqa: E731  (a group without a leading zero)
    pats = {
        "de": {"pairs": [lambda: f"0{nz(3)} {nz(2)} {nz(2)} {nz(2)}", lambda: f"0{nz(2)} {nz(2)} {nz(2)} {nz(2)}"],
               "hundreds": [lambda: f"0{nz(2)} {nz(3)} {nz(3)}", lambda: f"0{nz(3)} {nz(3)} {nz(3)}"],
               "mixed_natural": [lambda: f"0{nz(3)} {nz(3)} {nz(2)} {nz(2)}"],
               "digits": [lambda: f"0{nz(3)} {nz(3)} {d(4)}"], "1+2_splits": [lambda: f"0{nz(2)} {nz(3)} {nz(3)}"]},
        "nl": {"pairs": [lambda: f"0{nz(1)} {nz(2)} {nz(2)} {nz(2)} {nz(2)}", lambda: f"0{nz(2)} {nz(2)} {nz(2)} {nz(2)}"],
               "mixed_natural": [lambda: f"0{nz(2)} {nz(3)} {nz(2)} {nz(2)}"], "1+2_splits": [lambda: f"0{nz(2)} {nz(3)} {nz(3)}"],
               "hundreds+pairs": [lambda: f"0{nz(2)} {nz(3)} {nz(2)} {nz(2)}"], "digits": [lambda: f"0{nz(2)} {d(3)} {d(4)}"]},
        "es": {"1+2_splits": [lambda: f"{rng.choice('69')}{nz(2)} {nz(3)} {nz(3)}", lambda: f"{rng.choice('69')}{nz(2)} {nz(2)} {nz(2)} {nz(2)}"],
               "pairs": [lambda: f"{rng.choice('69')} {nz(2)} {nz(2)} {nz(2)} {nz(2)}", lambda: f"{rng.choice('69')}{nz(1)} {nz(2)} {nz(2)} {nz(2)} {d(1)}"],
               "mixed_natural": [lambda: f"{rng.choice('69')}{nz(2)} {nz(3)} {nz(3)}"],
               "hundreds+pairs": [lambda: f"{rng.choice('69')}{nz(2)} {nz(2)} {nz(2)} {nz(2)}"], "digits": [lambda: f"{rng.choice('69')}{d(2)} {d(3)} {d(3)}"],
               "hundreds": [lambda: f"{rng.choice('69')}{nz(2)} {nz(3)} {nz(3)}"]},
        "fr": {"pairs": [lambda: f"0{nz(1)} {nz(2)} {nz(2)} {nz(2)} {nz(2)}"],
               "hundreds+pairs": [lambda: f"0{nz(1)} {nz(2)} {nz(3)} {nz(3)}", lambda: f"0{nz(3)} {nz(2)} {nz(2)} {nz(2)}"],
               "digits": [lambda: f"0{nz(1)} {d(2)} {d(2)} {d(2)} {d(2)}"], "1+2_splits": [lambda: f"0{nz(2)} {nz(3)} {nz(3)}"],
               "hundreds": [lambda: f"0{nz(2)} {nz(3)} {nz(3)}"], "mixed_natural": [lambda: f"0{nz(2)} {nz(3)} {nz(2)} {nz(2)}"]},
        # Italian mobile numbers (3xx): the carriers say "il {n}", and "il zero" is not Italian
        "it": {"digits": [lambda: f"3{d(2)} {d(3)} {d(4)}"],
               "pairs": [lambda: f"3{nz(1)} {nz(2)} {nz(2)} {nz(2)} {nz(2)}"],
               "hundreds+pairs": [lambda: f"3{nz(2)} {nz(3)} {nz(2)} {nz(2)}"], "1+2_splits": [lambda: f"3{nz(2)} {nz(3)} {nz(3)}"],
               "mixed_natural": [lambda: f"3{nz(2)} {nz(3)} {nz(4)}"]},
        "pt": {"pairs": [lambda: f"{nz(2)} {nz(2)} {nz(2)} {nz(2)} {nz(2)} {d(1)}", lambda: f"9{nz(1)} {nz(2)} {nz(2)} {nz(2)} {d(1)}"],
               "1+2_splits": [lambda: f"9{nz(2)} {nz(3)} {nz(3)}", lambda: f"2{nz(2)} {nz(3)} {nz(3)}"],
               "digits": [lambda: f"9{d(2)} {d(3)} {d(3)}"], "hundreds": [lambda: f"2{nz(2)} {nz(3)} {nz(3)}"],
               "mixed_natural": [lambda: f"9{nz(2)} {nz(3)} {nz(3)}"]},
    }
    return rng.choice(pats[lang][conv])()


JOINERS = {l: {w.split()[0].lower() for w in M.RANGE_JOIN[l] + M.RANGE_OR[l]} | {"à"} for l in M.LANGS}


def _range_group(conv: str) -> str:
    if conv.startswith("between.."):
        return "RANGE_BETWEEN"
    if conv.split("..")[0] in ("von", "de", "da", "van"):
        return "RANGE_FROM"
    return "QTY"


def slot_text(lang: str, cell: str, conv: str, val, rng, P) -> str | None:
    """The written slot for one number of `cell` read with `conv`; val = a dropped unit to keep the value of."""
    cls = cell.split(":")[0]
    if cls == "year":
        return str(val["val"])
    if cls == "clock":
        if val is not None:
            h, m = _clock_value(val)
        else:  # a time the grammar table knows (pools), with the convention's minutes
            ok = [(h, m) for h, m in P["clock"] if m in _clock_minutes(lang, conv) and _clock_hour_ok(lang, conv, h, m)]
            if not ok:
                return None
            h, m = rng.choice(ok)
        return f"{h}:{m:02d}"
    if cls == "phone":
        return _phone_written(lang, conv, rng)
    if cls == "cur":
        if conv == "cents_only":
            n, f = 0, rng.choice(["10", "20", "25", "30", "40", "50", "60", "75", "80", "90", "95", "99", "15", "45"])
        elif val is not None:
            mm = re.search(r"(\d+)[.,](\d{2})(?!\d)", val["written"])
            n, f = int(mm.group(1)), mm.group(2)
        else:
            n = rng.randint(1, 60) if conv in ("bare_no_unit", "bare_joiner") else rng.randint(1, 99)
            f = rng.choice(["50", "90", "99", "95", "75", "10", "25", "45", "20", "80", "60", "35"])
        sym = rng.choice(T._money_forms(lang))
        if lang == "pt" and "R$" not in sym and rng.random() < 0.6:
            sym = "R$ {a}"
        return sym.format(a=f"{n},{f}")
    if cls == "dec":
        if val is not None:
            mm = re.search(r"(\d+)[.,](\d+)", val["written"])
            n, f = int(mm.group(1)), mm.group(2)
        elif conv in ("comma_as_number", "comma_digitwise") or (conv == "con" and rng.random() < 0.5):
            # a two-digit fraction the grammar table can read ("tres coma veinticinco"), not "x,y0" / "x,0y"
            two = [(a, b) for a, b in P["dec"] if len(b) == 2 and b[0] != "0" and b[1] != "0" and a <= 999]
            n, f = rng.choice(two) if two else (rng.randint(0, 99), str(rng.choice(range(11, 100))))
        else:
            n = rng.randint(1, 19) if conv == "half_word" else rng.randint(0, 49)
            f = {"half_word": "5", "comma_1digit": str(rng.randint(1, 9)), "point_1digit": str(rng.randint(1, 9)),
                 "con": rng.choice(["5", "2", "8", "3", "4", "6"])}.get(conv, str(rng.randint(1, 9)))
        sep = "." if conv == "point_1digit" else ","
        if conv == "half_word" and lang in ("es", "fr", "it", "pt"):
            return f"{n}{sep}{f}"  # "dos litros y medio": the half follows the noun, so no noun ("dos y medio")
        return f"{n}{sep}{f} {rng.choice(TT.DEC_NOUNS[lang])}"
    if cls == "pct":
        n = rng.choice([100] * 3 + list(range(2, 100))) if conv == "por cien" else rng.randint(2, 95)
        return f"{n} %" if lang in ("fr", "es") or rng.random() < 0.5 else f"{n}%"
    if cls == "int_big":
        n = val
        sep = "." if lang in ("de", "nl", "es", "it", "pt") else " "
        noun = rng.choice({"de": ["Euro", "Mitarbeiter", "Besucher", "Kunden"], "nl": ["euro", "bezoekers", "klanten", "leerlingen"]}.get(lang, TT.NOUNS[lang]))
        return f"{n // 1000}{sep}{n % 1000:03d} {noun}"
    if cls == "range":
        g = _range_group(conv)
        if g == "RANGE_FROM":
            l, r = T._pair(rng, P, False)
            u = rng.choice(["h", "km", "kg", "min", "h"])
            return f"{l}–{r} {u}"
        l, r = T._pair(rng, P, False, nounish=True)
        return f"{l}–{r} {rng.choice(TT.NOUNS[lang])}"
    return None


def _fresh_big(lang: str, rng, avoid: set) -> int:
    """A count 1100-1999 the grammar table can read (pools), not yet in the class, not ending in 1 / 2 ("ein" / "één"
    agree with the noun), not round."""
    pool = [n for n in T.pools(lang)["card"] if 1100 <= n <= 1999 and n % 100 and n % 10 not in (1, 2)]
    fresh = [n for n in pool if n not in avoid]
    return rng.choice(fresh or pool)


# ------------------------------------------------------------------------------------------------ the pick

def _unit_of(lang, cell, s_list, picks, text_after):
    """A reading_conv unit for one slot's spans and chosen readings."""
    cls = cell.split(":")[0]
    w = "".join(s.text for s, _ in zip(s_list, picks))
    rd = " ".join(c.strip() for c in picks)
    u = {"cls": cls if cls != "year" else "year", "kind": s_list[0].kind, "written": w, "reading": rd,
         "after": text_after, "before": "", "val": None}
    if cls == "year":
        u["val"] = int(re.sub(r"\D", "", w))
    if cls == "int_big":
        u["val"] = int(re.sub(r"\D", "", w))
    if cls == "phone":
        u["pieces"] = [(s.text, c) for s, c in zip(s_list, picks)]
    if cls == "range":
        u["right"] = picks[-1]
    return u


def _weights(lang, conv, s, idxs, used):
    out = []
    for i in idxs:
        c = s.cands[i]
        w = 3.0 if i == 0 else 1.5 if i <= 2 else 1.0
        if conv in REL_CONVS | {"digital_joiner", "noon_midnight"} and s.kind == "clock":
            hw = c.split()[0].lower()
            try:  # the 12-hour word is how people say a relative / "y media" time
                h24 = int(re.match(r"\d+", s.text).group(0))
                if h24 > 12 and hw in (M.g(lang, "card", h24) + [""])[:1]:
                    w *= 0.25
            except (M.Missing, AttributeError):
                pass
        out.append(w / (1.0 + 0.7 * used[(lang, s.kind, c.strip())]))
    return out


HALF_MASC = {"es": "medio", "fr": "demi", "it": "mezzo", "pt": "meio"}


def natural(lang: str, cls: str, conv: str, s, c: str) -> bool:
    """Readings a convention allows but people do not say in these carriers."""
    ws = c.split()
    if lang == "pt" and re.search(r"\b(?:dezasseis|dezassete|dezanove)\b", c):
        return False  # European teens: the Chirp Portuguese voices are Brazilian
    if cls == "clock":
        if lang in ("es", "pt") and "para" in ws:
            return False  # "a las diez para las cuatro": the carrier already says "a las"
        h24 = int(re.match(r"\d+", s.text).group(0))
        if conv in REL_CONVS - {"oclock_hourword"} and h24 > 12:
            try:
                if ws[:len(M.g(lang, "card", h24))] == M.g(lang, "card", h24):
                    return False  # "dix-huit heures moins le quart", "quince y media": the 12-hour word is said
            except M.Missing:
                return False
        if lang == "fr" and conv in REL_CONVS and not re.search(r"\b(?:heures?|midi|minuit)\b", c):
            return False  # "à six et demie" without "heures"
    if cls == "cur" and lang == "it":
        if "virgola" in ws and conv != "decimal_word+unit":
            return False
        m = re.search(r"[.,](\d{2})", s.text)
        if "centesimo" in ws and not (m and int(m.group(1)) == 1):
            return False  # "novantanove centesimo" (a grammar-table slip)
    if cls == "dec":
        m = re.search(r"[.,](\d+)", s.text)
        frac = m.group(1) if m else ""
        if conv == "half_word" and lang in HALF_MASC and HALF_MASC[lang] not in ws:
            return False  # no noun: the masculine half ("dos y medio")
        if conv in ("con", "comma_as_number") and len(frac) == 2:
            sep = next((k for k, w in enumerate(ws) if w in ("con", "komma", "coma", "virgule", "virgola", "vírgula")), None)
            segs = RC.segment(lang, RC.joined(" ".join(ws[sep + 1:])), frac) if sep is not None else []
            if not segs or min(len(x) for x in segs) != 1:
                return False  # "cuarenta y nueve con setenta y cinco", not "con siete cinco"
    return True


def make_pick(lang: str, slots: list, rng, used, before_ctx: dict, zwo: bool, meia: bool, written: str):
    """pick(spans, ok) for ml_tts.realize: every slot (a char range of the written sentence + cell + conv) gets readings
    whose reading_conv convention is the slot's; None when a slot has no such reading."""
    def pick(sp, oks):
        picks = [ok[0] for ok in oks]
        for (a, b, cell, conv) in slots:
            idx = [k for k, s in enumerate(sp) if s.start >= a - 1 and s.end <= b + 8 and s.start < b]
            if not idx:
                return None
            after = written[sp[idx[-1]].end:sp[idx[-1]].end + 40]
            cls = cell.split(":")[0]
            ctx = before_ctx.get((a, b), "")
            if cls == "phone":
                got = _pick_phone(lang, conv, [sp[k] for k in idx], [oks[k] for k in idx], rng, zwo, meia)
                if got is None:
                    return None
                for k, c in zip(idx, got):
                    picks[k] = c
                continue
            if cls == "range":
                lk, rk = idx[0], idx[-1]
                good = []
                for i in oks[rk]:
                    if conv in ("bare", "between_bare") and sp[rk].cands[i].split()[0].lower() in JOINERS[lang]:
                        continue  # "zu acht" / "auf acht" are no bare range
                    u = _unit_of(lang, cell, [sp[lk], sp[rk]], [sp[lk].cands[oks[lk][0]], sp[rk].cands[i]], after)
                    u["before"] = ctx
                    if RC.classify(lang, u)[0] == conv:
                        good.append(i)
                if not good:
                    return None
                picks[lk] = oks[lk][0]
                picks[rk] = rng.choices(good, _weights(lang, conv, sp[rk], good, used))[0]
                continue
            k = idx[0]
            s = sp[k]
            good = []
            for i in oks[k]:
                c = s.cands[i]
                if not natural(lang, cls, conv, s, c):
                    continue
                if cls == "cur" and conv == "unit+join+cents" and lang == "it" and "virgola" in c:
                    continue
                if RC.classify(lang, _unit_of(lang, cell, [s], [c], after))[0] == conv:
                    good.append(i)
            if not good:
                return None
            picks[k] = rng.choices(good, _weights(lang, conv, s, good, used))[0]
        return picks
    return pick


_KINDS = ("d", "pairs", "hundreds", "thousands", "1+2", "2/3mix")


def _gkind(lang, s, c):
    sh = RC.group_shape(lang, re.sub(r"\D", "", s.text), c)
    n = len(re.sub(r"\D", "", s.text))
    if sh.startswith("X") or sh == "?":
        return None
    core = sh.lstrip("z")
    if sh == "d" or n == 1:
        return "d"
    if core in ("n2", "pairs"):
        return "pairs"
    if core.startswith("n"):
        return "hundreds" if core == "n3" else "thousands"
    if core in ("1+2", "2/d"):
        return "1+2"
    return "2/3mix"


# the group kinds a convention reads its groups with, after the first group (the area code is said any natural way):
# 2-digit groups always as one number ("zweiundsiebzig") except in "digits", 3-digit groups by the convention
_GROUP_KINDS = {"pairs": {2: {"pairs"}, 4: {"pairs"}, 1: {"d"}},
                "hundreds+pairs": {2: {"pairs"}, 3: {"hundreds"}, 4: {"pairs"}, 1: {"d"}},
                "hundreds": {3: {"hundreds"}, 2: {"pairs"}, 1: {"d"}},
                "1+2_splits": {2: {"pairs"}, 3: {"1+2"}, 4: {"pairs"}, 1: {"d"}},
                "mixed_natural": {2: {"pairs"}, 1: {"d"}},
                "digits": {}}


def _pick_phone(lang, conv, sps, oks, rng, zwo, meia):
    by = []
    for j, (s, ok) in enumerate(zip(sps, oks)):
        g = collections.defaultdict(list)
        allow = None if j == 0 else _GROUP_KINDS[conv].get(len(re.sub(r"\D", "", s.text)))
        for i in ok:
            k = _gkind(lang, s, s.cands[i])
            if k and (allow is None or k in allow):
                g[k].append(i)
        if not g:
            return None
        by.append(g)
    combos = list(itertools.product(*[sorted(g) for g in by]))
    rng.shuffle(combos)
    for combo in combos[:4000]:
        picks = []
        for g, k, s in zip(by, combo, sps):
            cand = g[k]
            pref = [i for i in cand if ("zwo" in s.cands[i].split()) == zwo] if lang == "de" else \
                [i for i in cand if ("meia" in s.cands[i].split()) == meia] if lang == "pt" else cand
            picks.append(rng.choice(pref or cand))
        conv_got, _ = RC.phone_groups(lang, [(s.text, s.cands[i]) for s, i in zip(sps, picks)])
        if conv_got == conv:
            if lang == "de" and zwo and not any("zwo" in s.cands[i].split() for s, i in zip(sps, picks)):
                continue
            if lang == "pt" and meia and not any("meia" in s.cands[i].split() for s, i in zip(sps, picks)):
                continue
            return picks
    return None


# ------------------------------------------------------------------------------------------------ sentences

def _clause(lang: str, group: str, rng, conv: str, avoid: int | None):
    cs = TT.CLAUSES[lang][group]
    ci = rng.choice([i for i in range(len(cs)) if i != avoid] or [0])
    c = cs[ci]
    if conv == "noon_midnight":  # "a mezzogiorno", "ao meio-dia", "à meia-noite" (not "alle" / "às")
        c = c.replace("alle {n}", "a {n}") if lang == "it" else c
        c = c.replace("às {n}", "ao {n}") if lang == "pt" else c
    return ci, c


def sentence(lang: str, items: list, rng, P):
    """items: [(cell, conv, val)] (one or two numbers of one cell) -> (written, slots, before contexts) or None."""
    parts, slots, ctx, avoid = [], [], {}, None
    for cell, conv, val in items:
        cls = cell.split(":")[0]
        grp = _range_group(conv) if cls == "range" else GROUP[cls]
        ci, cl = _clause(lang, grp, rng, conv, avoid)
        avoid = ci
        st = slot_text(lang, cell, conv, val, rng, P)
        if st is None:
            return None
        parts.append((cl, st, cell, conv))
    text, pos = "", 0
    for j, (cl, st, cell, conv) in enumerate(parts):
        pre, post = cl.split("{n}")
        if j == 0:
            seg = pre
            seg = seg[:1].upper() + seg[1:] if seg else seg
        else:
            seg = f", {CONN[lang]} {pre}"
        a = len(text) + len(seg)
        text += seg + st
        b = len(text)
        text += post
        slots.append((a, b, cell, conv))
        ctx[(a, b)] = text[max(0, a - 60):a]
    text += "."
    if len(text.split()) > WORDS_MAX:
        return None
    return text, slots, ctx


# ------------------------------------------------------------------------------------------------ planning

def plan_lang(lang: str, cells: list, mix_rows: list, tag: str, seed: int, rep: dict) -> list:
    """The planned rows of one language (no network)."""
    rng = random.Random(f"{tag}|{seed}|{lang}")
    P = T.pools(lang)
    cen = census(mix_rows, lang)
    by_id = {r["id"]: r for r in mix_rows}
    used = collections.Counter()
    rows, seen, taken = [], set(), set()
    for cell in [c for c in cells if c["lang"] == lang]:
        name = cell["cls"]
        cc = cen.get(name, {"convs": collections.Counter(), "units": [], "pure": set()})
        need, excess = plan_counts(cell, cc)
        if "mixed_natural" in need:  # "nueve cinco cuatro siete noventa y tres ...": the few mixed phones become pairs
            need["pairs"] = need.get("pairs", 0) + need.pop("mixed_natural")
        crep = rep.setdefault(name, {"need": dict(need), "excess": dict(excess), "dropped": 0, "kept_value": 0,
                                     "fresh_value": 0, "rows": 0, "units": 0, "fail": collections.Counter()})
        # drop candidates: units of over-represented conventions in rows whose numbers are all in this cell
        cand = [(rid, k, conv, u) for rid, k, conv, u in cc["units"] if conv in excess and rid in cc["pure"]]
        if name == "cur" and lang == "it":  # the "euro virgola .. centesimi" hybrids go first
            cand.sort(key=lambda x: not (x[2] == "decimal_word+unit" and re.search(r"centesim", x[3]["reading"])))
        else:
            rng.shuffle(cand)
        left = dict(excess)
        # every row a drop touches leaves whole: count its units (2-number rows count twice)
        units_of_row = collections.Counter(rid for rid, *_ in cc["units"])
        jobs = []  # (conv, value unit or None, superseded ids)
        pending = []  # the other numbers of a dropped 2-3 number row: replaced by the next jobs (no new drop)
        cand.sort(key=lambda x: units_of_row[x[0]] > 1)  # one-number rows first (a stable sort keeps the shuffle)
        order = sorted(need, key=lambda c: len([x for x in cand if compatible(lang, name, c, x[3])]))
        kind = name.split(":")[0]
        for conv in order:
            for _ in range(need[conv]):
                val, sup = None, []
                if cell["mode"] == "replace":
                    pu = next((u for u in pending if compatible(lang, name, conv, u)), pending[0] if pending else None)
                    if pu is not None:
                        pending.remove(pu)
                        val = pu if compatible(lang, name, conv, pu) else None
                    else:
                        hit = next((x for x in cand if x[0] not in taken and left.get(x[2], 0) > 0
                                    and compatible(lang, name, conv, x[3])), None)
                        if hit is None:  # no compatible value left: drop an over-represented row anyway, fresh value
                            hit = next((x for x in cand if x[0] not in taken and left.get(x[2], 0) > 0), None)
                        if hit is None:
                            crep["fail"]["no_row_to_replace"] += 1
                            continue
                        val = hit[3] if compatible(lang, name, conv, hit[3]) else None
                        taken.add(hit[0])
                        for rid2, k2, conv2, u2 in cc["units"]:
                            if rid2 == hit[0]:
                                left[conv2] = left.get(conv2, 0) - 1
                                if k2 != hit[1]:
                                    pending.append(u2)
                        crep["dropped"] += units_of_row[hit[0]]
                        sup = [hit[0]]
                if kind == "int_big":
                    val = _fresh_big(lang, rng, {u2.get("val") for _, _, _, u2 in cc["units"]})
                elif kind in ("phone", "pct", "range"):
                    val = None
                if kind == "year" and val is None:  # a year value of this cell's spread that the convention can read
                    pool = [u2 for *_, u2 in cc["units"] if compatible(lang, name, conv, u2)]
                    if not pool:
                        crep["fail"]["no_year_value"] += 1
                        continue
                    val = dict(rng.choice(pool))
                    crep["fresh_value"] += 1
                else:
                    crep["kept_value" if val is not None and kind in ("year", "clock", "cur", "dec") else "fresh_value"] += 1
                jobs.append((conv, val, sup))
        crep["dropped_not_replaced"] = len(pending)
        rng.shuffle(jobs)
        per = 1 if name == "phone" else 2
        groups = [jobs[i:i + per] for i in range(0, len(jobs), per)]
        def attempt_group(g, tries):
            for attempt in range(tries):
                fresh = attempt >= tries // 2  # a kept value that keeps failing (a grammar gap): a fresh one
                items = [(name, conv, None if fresh and kind not in ("year",) else val) for conv, val, _ in g]
                st = sentence(lang, items, rng, P)
                if st is None:
                    crep["fail"]["sentence"] += 1
                    continue
                written, slots, ctx = st
                zwo = lang == "de" and kind == "phone" and rng.random() < 0.3
                meia = lang == "pt" and kind == "phone" and rng.random() < 0.5
                pick = make_pick(lang, slots, rng, used, ctx, zwo, meia, written)
                T.VARIETY_FILL = True
                try:  # ranges: the clause's joiner rule (WANT) is the pick's convention check instead
                    res, why = T.realize(lang, TTS_CLS[kind], written, GROUP[kind], None, rng, used, names=NAMES,
                                         pick=pick)
                finally:
                    T.VARIETY_FILL = False
                if res is None:
                    crep["fail"][why + ":" + "+".join(c for c, _, _ in g)] += 1
                    continue
                label, spans, offs = res
                if label in seen:
                    crep["fail"]["duplicate"] += 1
                    continue
                # the census's own reading of the planned row: every number has its planned convention
                r = {"id": "x", "text": label, "caption": written, "spans": spans, "num_chars": offs, "src": "tts"}
                us = RC.units(r, lang)
                got = [RC.classify(lang, u)[0] for u in us if (cell_of(u) or "").split(":")[0] == kind]
                want = [conv for conv, _, _ in g]
                if sorted(got) != sorted(want) or len(us) != len(want):
                    crep["fail"]["census_disagrees"] += 1
                    continue
                for sp_ in spans:
                    used[(lang, sp_["kind"], sp_["reading"])] += 1
                return written, label, spans, offs, want, [i for _, _, sup in g for i in sup], meia
            return None

        for g in groups:
            got_rows = [attempt_group(g, 12)]
            if got_rows[0] is None and len(g) > 1:  # two numbers that do not go together: one sentence each
                got_rows = [attempt_group([x], 8) for x in g]
            for row in got_rows:
                if row is None:
                    crep["fail"]["gave_up"] += 1
                    continue
                written, label, spans, offs, convs, sup, meia = row
                seen.add(label)
                rid = f"{tag}_{lang}_{hashlib.md5(f'{seed}|{label}'.encode()).hexdigest()[:10]}"
                rows.append({"id": rid, "lang": lang, "cls": TTS_CLS[kind], "ckey": name, "convs": convs,
                             "supersedes": sup, "caption": written, "text": label, "spans": spans, "num_chars": offs,
                             "kind": sorted({s_["kind"] for s_ in spans}), "split": "train", "meia": meia})
                crep["rows"] += 1
                crep["units"] += len(convs)
        crep["fail"] = dict(crep["fail"])
    # a superseded row is replaced only once
    assert len({i for r in rows for i in r["supersedes"]}) == sum(len(r["supersedes"]) for r in rows)
    for r in rows:
        assert all(i in by_id for i in r["supersedes"])
    return rows


# ------------------------------------------------------------------------------------------------ voices

def assign_voices(rows: list, voices: list, lang: str, rng) -> None:
    """Chirp 3 HD persona x locale x rate per row; Brazilian "meia" only with pt-BR, European teens only with pt-PT."""
    from mix_build import heldout_persona, is_heldout_voice

    ok = [(n, loc) for n, loc in voices if not is_heldout_voice(lang, n) and heldout_persona(n) not in EVAL_PERSONAS]
    assert ok, f"{lang}: no voice left"
    for r in rows:
        pool = ok
        if lang == "pt":
            eu = bool(re.search(r"\b(?:dezasseis|dezassete|dezanove)\b", r["text"]))
            want = "pt-PT" if eu else "pt-BR" if r.get("meia") else None
            pool = [v for v in ok if v[1] == want] or ok if want else ok
        if lang == "es" and re.search(r"\bpor cien\b|\bcon\b", r["text"]) and r["ckey"] in ("pct", "dec"):
            pool = [v for v in ok if v[1] == "es-ES"] or ok  # "por cien" / "tres con cinco": Spain
        name, loc = rng.choice(pool)
        r.update(engine="presynth", voice=name, locale=loc, voice_license="Google Cloud Text-to-Speech output",
                 voice_path=None, speaker=None, speed=rng.choice(RATES), pitch=1.0)


def cloud_voices(auth, lang: str) -> list:
    import ml_tts_cloud as CL

    out = list(CL.voices(auth, lang))
    for loc in EXTRA_LOCALES.get(lang, []):
        try:
            d = CL._req(auth, f"{CL.API}/voices?languageCode={loc}")
        except Exception:  # noqa: BLE001  (a locale the service does not have)
            continue
        out += sorted((v["name"], loc) for v in d.get("voices", []) if "Chirp3-HD" in v["name"]
                      and loc in v.get("languageCodes", [loc]))
    return sorted(set(out))


# ------------------------------------------------------------------------------------------------ CLI

def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--plan", default=os.path.join(HERE, "variety_plan_rv1.json"))
    p.add_argument("--mix", required=True, help="dir with the built mix's tts_<lang>.jsonl")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--langs", default="de,nl,es,fr,it,pt")
    p.add_argument("--seed", type=int, default=18)
    s = sub.add_parser("synth")
    s.add_argument("--planned", required=True)
    s.add_argument("--vtag", required=True, help="clips + planned rows go to ml/<vtag>/<lang>/ (NEW paths only)")
    s.add_argument("--skip", type=int, default=0)
    s.add_argument("--pilot", type=int, default=0)
    s.add_argument("--work", default="/tmp/tts_variety")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--seed", type=int, default=18)
    a = ap.parse_args()
    if a.cmd == "plan":
        cells = json.load(open(a.plan))["cells"]
        os.makedirs(a.out, exist_ok=True)
        summary = {}
        for lang in a.langs.split(","):
            rep: dict = {}
            rows = plan_lang(lang, cells, _rows(os.path.join(a.mix, f"tts_{lang}.jsonl")), a.tag, a.seed, rep)
            with open(os.path.join(a.out, f"planned_{lang}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary[lang] = {"rows": len(rows), "units": sum(len(r["convs"]) for r in rows),
                             "chars": sum(len(r["text"]) for r in rows), "cells": rep}
            print(lang, json.dumps({k: v for k, v in summary[lang].items() if k != "cells"}))
        json.dump(summary, open(os.path.join(a.out, "plan_report.json"), "w"), ensure_ascii=False, indent=1)
        tot = sum(v["chars"] for v in summary.values())
        print(json.dumps({"rows": sum(v["rows"] for v in summary.values()), "units": sum(v["units"] for v in summary.values()),
                          "chars": tot, "usd_list": round(tot / 1e6 * 30, 2)}))
        return
    # synth
    import base64
    import time
    from concurrent.futures import ThreadPoolExecutor

    import ml_tts_cloud as CL

    rows = _rows(a.planned)[a.skip:]
    if a.pilot:
        rows = rows[:a.pilot]
    lang = rows[0]["lang"]
    auth = CL.Auth()
    vs = cloud_voices(auth, lang)
    assign_voices(rows, vs, lang, random.Random(f"{a.vtag}|{a.seed}|{lang}"))
    for r in rows:
        r["presynth"] = f"{VOL}/ml/{a.vtag}/{lang}/presynth/{r['id']}.wav"
    w = os.path.join(a.work, a.vtag, lang)
    os.makedirs(w, exist_ok=True)

    def synth(r):
        out = os.path.join(w, r["id"] + ".wav")
        if os.path.exists(out):
            return True
        body = {"input": {"text": r["text"]}, "voice": {"languageCode": r["locale"], "name": r["voice"]},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000, "speakingRate": r["speed"]}}
        for attempt in range(5):
            try:
                d = CL._req(auth, f"{CL.API}/text:synthesize", body)
                open(out, "wb").write(base64.b64decode(d["audioContent"]))
                return True
            except Exception:  # noqa: BLE001  (rate limits / transient errors: back off)
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(a.threads) as ex:
        ok = list(ex.map(synth, rows))
    rows = [r for r, k in zip(rows, ok) if k]
    with open(os.path.join(w, "planned.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(r["text"]) for r in rows)
    # new paths only: no --force, an existing path is an error
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", w + "/", f"ml/{a.vtag}/{lang}/presynth/"],
                   cwd=HERE, check=True, capture_output=True)
    subprocess.run(["modal", "volume", "put", "parakeet-spoken-form", os.path.join(w, "planned.jsonl"),
                    f"ml/{a.vtag}/{lang}/planned.jsonl"], cwd=HERE, check=True, capture_output=True)
    print(json.dumps({"lang": lang, "uploaded_rows": len(rows), "cloud_failed": ok.count(False), "chars": chars,
                      "usd_list": round(chars / 1e6 * 30, 2),
                      "voices": collections.Counter(r["voice"] for r in rows).most_common(40)}))


if __name__ == "__main__":
    main()
