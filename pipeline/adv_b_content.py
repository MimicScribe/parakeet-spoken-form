"""Adversarial pack (adv_b): the constructions (inventory) per language. A construction is one way of saying a thing; its values
are the keys the readout splits (V_a / V_b), the construction itself is what C_a / C_b hold out. Families (the tag the next build
includes or excludes): `phone` (mixed phone styles), `stumble` (restarts, fillers, corrections inside numbers), `phrasing`
(unusual but common phrasings). Every construction can also build a CONTROL piece: the standard reading of the same value
(`ctrl`), used by the matched normal-speech slice of the test set."""
import random
from dataclasses import dataclass
from typing import Callable

import adv_b_phone as PH
import adv_b_phrasing as PR
import adv_b_stumble as ST
import adv_b_words as W
from edge_a_content import Cons, Piece


@dataclass
class AC(Cons):
    split_group: str = ""            # constructions sharing a group share ONE value split (a phone number held out is held out of every style)
    ctrl: Callable = None            # (key, rng) -> Piece: the standard reading of the same value (normal-speech control)
    sub: str = ""                    # sub-area inside the family (phone kind, stumble type, phrasing area)
    min_rows: int = 0


def _piece(label, synth, nums, ambig=()):
    p = Piece(label, "", list(nums), tuple(ambig))
    p.synth = synth
    return p


# ------------------------------------------------------------------------------------------------ phone
PHONE_AMB = ("phone_style",)


def _phone_piece(lang, kind, style, key, rng, fn):
    """fn(style, key, rng) -> (groups, plus, prefix)."""
    groups, plus, prefix = fn(style, key, rng)
    reading = " ".join(groups)
    label = (prefix + " " if prefix else "") + reading
    synth = (prefix + ", " if prefix else "") + ", ".join(groups)
    amb = set(PHONE_AMB)
    if " oh" in " " + reading or reading.startswith("oh"):
        amb.add("oh_vs_zero")
    if any(w in reading.split() for w in ("double", "doble", "triple", "treble")):
        amb.add("double_triple")
    if "septante" in reading or "nonante" in reading or "huitante" in reading:
        amb.add("regional_numerals")
    return _piece(label, synth, [("phone", PH.caption(kind, key, plus), reading)], sorted(amb))


def _pcons(lang, id, kind, style, fn, group, values, ctrl_style=None, ctrl_fn=None, sub=None, weight=1.0):
    ctrl_fn = ctrl_fn or fn

    def make(k, rng):
        return _phone_piece(lang, kind, style, k, rng, fn)

    ctrl = None
    if ctrl_style:
        ctrl = lambda k, rng: _phone_piece(lang, kind, ctrl_style, k, rng, ctrl_fn)  # noqa: E731
    return AC(id=id, lang=lang, family="phone", cat="phone", values=list(values), make=make, split_group=group, ctrl=ctrl, sub=sub or kind,
              weight=weight, ambig=PHONE_AMB)


def build_phone(lang: str) -> list:
    U = {k: PH.universe(k, n) for k, n in [("us", 300), ("tf_us", 120), ("uk_mob", 260), ("uk_lon", 160), ("es_mob", 260), ("mx", 160), ("tf_mx", 80),
                                          ("fr", 300), ("fr_be", 140), ("tf_fr", 80), ("de", 160), ("pt", 160), ("it", 160), ("nl", 160)]}
    out = []
    if lang == "en":
        us = lambda st, k, rng: PH.us_style(st, k, rng)  # noqa: E731
        uk = lambda st, k, rng: PH.uk_style(st, k, rng)  # noqa: E731
        for st, w in [("digits_zero", 1.0), ("digits_oh", 1.0), ("mixed", 1.6), ("triplet", 1.0), ("double", 1.2), ("area_code", 1.0), ("country_us", 1.0)]:
            dbl = st == "double"
            vals = [v for v in U["us"] if (not dbl) or any(a == b for a, b in zip(v[3:], v[4:]))]
            out.append(_pcons(lang, f"en.phone.us_{st}", "us", st, us, "us", vals, ctrl_style="digits_oh", sub="us", weight=w))
        for st in ("tollfree_hundred", "tollfree_ohoh"):
            out.append(_pcons(lang, f"en.phone.{st}", "tf_us", st, lambda s, k, rng: (PH.tf_us_style(s, k, rng), "", ""), "tf_us",
                              [v for v in U["tf_us"] if st == "tollfree_ohoh" or v[1:4] == "800"],
                              ctrl_style="tollfree_ohoh", sub="tollfree"))
        for st, w in [("digits", 1.0), ("double", 1.4), ("mobile_pairs", 1.2), ("country_uk", 1.0)]:
            vals = [v for v in U["uk_mob"] if st != "double" or any(a == b for a, b in zip(v[2:], v[3:]))]
            out.append(_pcons(lang, f"en.phone.uk_{st}", "uk_mob", st, uk, "uk_mob", vals, ctrl_style="digits", sub="uk", weight=w))
        out.append(_pcons(lang, "en.phone.uk_london_pairs", "uk_lon", "london_pairs", uk, "uk_lon", U["uk_lon"], ctrl_style="london_pairs", sub="uk"))
        # the control of a London pairs row is the same pairs reading: London numbers are rarely digit by digit; see ctrl in the doc
    elif lang == "es":
        es = lambda kind: (lambda st, k, rng: es_fn(st, k, rng, kind))  # noqa: E731
        def es_fn(st, k, rng, kind):
            return PH.es_style(st, k, rng, kind)
        for st, w in [("digits", 1.2), ("hundreds_pairs", 1.6), ("mixed", 1.4), ("country", 1.0), ("doble", 1.0)]:
            vals = [v for v in U["es_mob"] if st != "doble" or any(a == b for a, b in zip(v, v[1:]))]
            out.append(_pcons(lang, f"es.phone.{st}", "es_mob", st, es("es_mob"), "es_mob", vals, ctrl_style="digits", sub="es", weight=w))
        out.append(_pcons(lang, "es.phone.mx_pairs", "mx", "pairs", es("mx"), "mx", U["mx"], ctrl_style="pairs", sub="mx"))
        out.append(_pcons(lang, "es.phone.tollfree_mx", "tf_mx", "x", es("tf_mx"), "tf_mx", U["tf_mx"], ctrl_style="x", sub="tollfree"))
    elif lang == "fr":
        fr = lambda kind: (lambda st, k, rng: PH.fr_style(st, k, rng, kind))  # noqa: E731
        for st, w in [("digits", 1.2), ("pairs", 1.6), ("mixed", 1.4), ("country", 1.0), ("double", 1.0)]:
            vals = [v for v in U["fr"] if st != "double" or any(v[i] == v[i + 1] for i in range(0, 10, 2))]
            out.append(_pcons(lang, f"fr.phone.{st}", "fr", st, fr("fr"), "fr", vals, ctrl_style="pairs", sub="fr", weight=w))
        out.append(_pcons(lang, "fr.phone.be_septante", "fr_be", "x", fr("fr_be"), "fr_be", U["fr_be"], ctrl_style="x", ctrl_fn=lambda st, k, rng: ([W.digits(k[:2], "fr")] + [W.pair(k[i:i + 2], "fr") for i in range(2, 10, 2)], "", ""), sub="regional"))
        out.append(_pcons(lang, "fr.phone.tollfree", "tf_fr", "x", fr("tf_fr"), "tf_fr", U["tf_fr"], ctrl_style="x", sub="tollfree"))
    else:
        kind = {"de": "de", "pt": "pt", "it": "it", "nl": "nl"}[lang]
        sm = lambda st, k, rng: PH.small_style(st, k, rng, lang, kind)  # noqa: E731
        for st in ("digits", "pairs", "mixed", "country"):
            out.append(_pcons(lang, f"{lang}.phone.{st}", kind, st, sm, kind, U[kind], ctrl_style="digits", sub=lang))
    return out


# ------------------------------------------------------------------------------------------------ stumble
def build_stumble(lang: str) -> list:
    out = []

    def add(id, cat, values, make, ctrl=None, sub=""):
        out.append(AC(id=f"{lang}.stum.{id}", lang=lang, family="stumble", cat=cat, values=list(values), make=make, ambig=ST.AMB, ctrl=ctrl, sub=sub or id))

    add("restart_prefix", "cnt", ST._num_keys(lang, "prefix"), ST.restart_prefix(lang), ST.c_prefix(lang))
    add("repeat_det", "cnt", ST._num_keys(lang, "any"), ST.repeat_det(lang), ST.c_prefix(lang))
    add("filler_mid", "cnt", ST._num_keys(lang, "split"), ST.filler_mid(lang), ST.c_prefix(lang))
    add("self_fix", "cnt", [f"{a}-{b}" for a, b in ST.near_pairs(lang)], ST.self_fix(lang), ST.c_selffix(lang))
    add("trail_resume", "cnt", ST._num_keys(lang, "prefix"), ST.trail_resume(lang), ST.c_prefix(lang))
    add("unit_false_start", "ufs", [str(n) for n in (2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90) if not (lang != "en" and n % 10 == 1)],
        ST.unit_false_start(lang), ST.c_unit(lang))
    add("digit_fix", "code", ST.digit_pairs(), ST.digit_fix(lang), ST.c_digits(lang))
    if lang in ("en", "es"):
        add("restart_scale", "cnt", [str(n) for n in (100, 200, 300, 400, 500, 600, 700, 800, 900, 150, 250, 350, 450, 550, 650, 750, 850, 950, 120, 125, 175, 225, 275, 325, 375)],
            ST.restart_scale(lang), ST.c_scale(lang))
    return out


# ------------------------------------------------------------------------------------------------ phrasing
def _plain_ctrl(label_fn):
    def ctrl(k, rng):
        label, nums = label_fn(k, rng)
        return _piece(label, label, nums)
    return ctrl


def build_phrasing(lang: str) -> list:
    out = []

    def add(id, cat, values, make, ctrl=None, sub="", ambig=()):
        out.append(AC(id=f"{lang}.phr.{id}", lang=lang, family="phrasing", cat=cat, values=list(values), make=make, ctrl=ctrl, sub=sub or id, ambig=tuple(ambig)))

    dk = PR.date_keys()
    if lang == "en":
        E = lambda f: (lambda k, rng: f(k, rng))  # noqa: E731
        c = lambda n: W.card(int(n), "en")  # noqa: E731
        add("emerg", "plain", ["911", "999", "112", "404", "311", "411", "211", "101", "505", "707", "808", "303", "606"], PR.en_emerg, sub="regional_digits")
        add("nought_point", "plain", [f"0.{d}" for d in "123456789"] + ["0.05", "0.25", "0.75", "0.15", "0.35", "0.45", "0.55", "0.65", "0.85", "0.95", "0.02", "0.08"],
            PR.en_nought, lambda k, rng: _piece("zero point " + " ".join("oh" if ch == "0" else W.DIGITS["en"][int(ch)] for ch in k.split(".")[1]),
                                                "zero point " + " ".join("oh" if ch == "0" else W.DIGITS["en"][int(ch)] for ch in k.split(".")[1]),
                                                [("dec", k, "zero point " + " ".join("oh" if ch == "0" else W.DIGITS["en"][int(ch)] for ch in k.split(".")[1]))]),
            sub="regional_digits")
        add("hundred_and", "plain", [str(n) for n in list(range(101, 110)) + list(range(112, 200, 7)) + list(range(201, 910, 29))], PR.en_hundred_and,
            lambda k, rng: _piece(f"{c(int(k) // 100 * 100)} {c(int(k) % 100)}", f"{c(int(k) // 100 * 100)} {c(int(k) % 100)}", [("int", k, f"{c(int(k) // 100 * 100)} {c(int(k) % 100)}")]),
            sub="regional_digits")
        add("version", "ver", PR.versions(), PR.en_version, sub="version")
        add("regnal", "reg", PR.regnal_keys("en"), PR.en_regnal, sub="regnal")
        add("address", "addr", PR.addresses(), PR.en_address,
            lambda k, rng: _piece(*(lambda w: (w, w, [("alnum" if k[-1].isalpha() else "int", k, w)]))(c(int(k.rstrip("ABCDE"))) + (f" {k[-1]}" if k[-1].isalpha() else ""))), sub="address")
        add("height", "h", [f"{f}-{i}" for f in (5, 6) for i in range(0, 12)] + ["4-11", "7-0", "7-1"], PR.en_height,
            lambda k, rng: _piece(*(lambda f, i: (lambda w: (w, w, [("unit", f"{f}'{i}\"", w)]))(f"{c(f)} feet {c(i)} inches" if i > 1 else f"{c(f)} feet"))(*map(int, k.split("-")))),
            sub="measurement")
        add("bp", "bp", [f"{s}-{d}" for s in (110, 115, 120, 125, 130, 135, 140, 145, 150, 160) for d in (70, 75, 80, 85, 90)], PR.en_bp,
            lambda k, rng: _piece(*(lambda s, d: (lambda w: (w, w, [("int", str(s), c(s)), ("int", str(d), c(d))]))(f"{c(s)} over {c(d)}"))(*map(int, k.split("-")))), sub="measurement")
        add("temp", "deg", ["-1", "-2", "-3", "-4", "-5", "-6", "-8", "-10", "-12", "-15", "-20", "-25", "-30", "-40"] + [f"{a}.{b}" for a in (36, 37, 38, 39, 40) for b in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
            PR.en_temp, sub="measurement")
        add("fraction", "qty", ["2+1/2", "1+1/2", "3+1/2", "4+1/2", "0+3/4", "0+1/4", "1+3/4", "2+1/4", "0+2/3", "0+1/3", "5+1/2", "2+3/4", "0+5/8", "0+7/8", "0+1/2", "1+1/4", "3+1/4", "0+1/8"],
            PR.en_frac, sub="measurement")
        add("power", "sci", ["10^6", "10^3", "10^9", "10^2", "2^8", "2^10", "x^2", "x^3", "10^-3", "10^-6", "2^3", "3^2", "5^4", "10^12", "2^16", "x^4", "10^-2", "2^5"], PR.en_power, sub="math")
        add("ratio", "ratio", ["3:1", "2:1", "4:3", "5:2", "7:2", "3:2", "10:1", "5:1", "9:2", "11:4", "6:1", "8:5", "4:1", "13:8", "7:3", "5:3"], PR.en_ratio, sub="math")
        mins_rel = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        add("clock_rel", "time", PR._clock_keys([m for m in mins_rel if m]), PR.en_clock_rel,
            lambda k, rng: PR.en_clock_digital(k, random.Random(1)), sub="clock")
        add("clock_digital", "time", PR._clock_keys([0, 5, 8, 15, 20, 30, 40, 45, 50, 55]), PR.en_clock_digital, sub="clock")
        add("clock_mil", "time", [f"{h}:{m:02d}" for h in (6, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23) for m in (0, 15, 30, 45, 5)], PR.en_clock_mil,
            lambda k, rng: PR.en_clock_digital(f"{int(k.split(':')[0]) % 12 or 12}:{k.split(':')[1]}", random.Random(1)), sub="clock")
        add("date_dmy", "date", dk, PR.en_date_dmy, lambda k, rng: PR.en_date_mdy(k, random.Random(1)), sub="date", ambig=PR.AM_DATE)
        add("date_mdy", "date", dk, PR.en_date_mdy, sub="date", ambig=PR.AM_DATE)
        add("date_numeric", "date", dk, PR.en_date_numeric, lambda k, rng: PR.en_date_mdy(k, random.Random(1)), sub="date", ambig=PR.AM_DATE)
        add("date_iso", "date", dk, PR.en_date_iso, lambda k, rng: PR.en_date_mdy(k, random.Random(1)), sub="date", ambig=PR.AM_DATE)
        add("range_to", "range", PR.ranges_en(), PR.en_range_to, sub="range")
        add("range_fast", "range", PR.ranges_en(), PR.en_range_fast, PR.en_range_to, sub="range", ambig=PR.AM_RANGE)
    elif lang == "es":
        c = lambda n: W.card(int(n), "es")  # noqa: E731
        add("version", "ver", PR.versions(), PR.es_version, sub="version")
        add("regnal", "reg", [f"{n}:{o}" for n, os_ in PR.ES_REG.items() for o in os_ if o in PR.ES_ORD], PR.es_regnal, sub="regnal")
        add("address", "addr", [a for a in PR.addresses() if not a.rstrip("ABCDE").endswith("1") or a.rstrip("ABCDE") in ("101",)], PR.es_address, sub="address")
        add("height", "h", [f"1-{c_}" for c_ in (55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95)], PR.es_height, sub="measurement")
        add("fever", "fever", [f"{a}.{b}" for a in (36, 37, 38, 39, 40) for b in (2, 3, 4, 5, 6, 7, 8, 9)], PR.es_fever, sub="measurement")
        add("fraction", "qty", ["2+1/2", "1+1/2", "3+1/2", "4+1/2", "0+3/4", "0+1/4", "0+2/3", "0+1/3", "5+1/2", "2+3/4"], PR.es_frac, sub="measurement")
        add("bp", "bp", [f"{s}-{d}" for s in (110, 115, 120, 125, 130, 135, 140, 145, 150, 160) for d in (70, 75, 80, 85, 90)], PR.es_bp, sub="measurement")
        add("clock_rel", "time", PR._clock_keys([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]), PR.es_clock_rel, sub="clock")
        add("clock_lat", "time", PR._clock_keys([35, 40, 45, 50, 55, 5, 10, 20]), PR.es_clock_lat, sub="clock")
        add("date_words", "date", dk, PR.es_date_words, sub="date", ambig=PR.AM_DATE)
        add("range_to", "range", PR.ranges_en(), PR.es_range_to, sub="range")
        add("range_fast", "range", [r for r in PR.ranges_en() if not any(int(x) % 10 == 1 for x in r.split("-"))], PR.es_range_fast, PR.es_range_to, sub="range", ambig=PR.AM_RANGE)
    elif lang == "fr":
        add("version", "ver", PR.versions(), PR.fr_version, sub="version")
        add("regnal", "reg", [f"{n}:{o}" for n, os_ in PR.FR_REG.items() for o in os_ if o in PR.FR_ORD], PR.fr_regnal, sub="regnal")
        add("address", "addr", [a for a in PR.addresses() if not a.rstrip("ABCDE").endswith("1") or a.rstrip("ABCDE") in ("101",)], PR.fr_address, sub="address")
        add("height", "h", [f"1-{c_}" for c_ in (55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95)], PR.fr_height, sub="measurement")
        add("fever", "fever", [f"{a}.{b}" for a in (36, 37, 38, 39, 40) for b in (2, 3, 4, 5, 6, 7, 8, 9)], PR.fr_fever, sub="measurement")
        add("fraction", "qty", ["2+1/2", "1+1/2", "3+1/2", "4+1/2", "0+3/4", "0+1/4", "0+2/3", "0+1/3", "5+1/2", "2+3/4"], PR.fr_frac, sub="measurement")
        add("bp", "bp", [f"{s}-{d}" for s in (110, 120, 130, 140, 150, 160) for d in (70, 80, 90)], PR.fr_bp, sub="measurement")
        add("clock_rel", "time", PR._clock_keys([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]), PR.fr_clock_rel, sub="clock")
        add("clock_24", "time", [f"{h}:{m:02d}" for h in (13, 14, 15, 16, 17, 18, 19, 20, 22, 23) for m in (0, 15, 30, 45, 10, 20)], PR.fr_clock_24, sub="clock")
        add("date_words", "date", dk, PR.fr_date_words, sub="date", ambig=PR.AM_DATE)
        add("range_to", "range", PR.ranges_en(), PR.fr_range_to, sub="range")
        add("range_fast", "range", [r for r in PR.ranges_en() if not any(int(x) % 10 == 1 for x in r.split("-"))], PR.fr_range_fast, PR.fr_range_to, sub="range", ambig=PR.AM_RANGE)
    return out


def build(lang: str) -> list:
    out = build_phone(lang)
    if lang in ("en", "es", "fr"):
        out += build_stumble(lang) + build_phrasing(lang)
    return out


BUILD = {l: (lambda l=l: build(l)) for l in W.LANGS}
