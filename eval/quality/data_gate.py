#!/usr/bin/env python3
"""RULE #1 data gate: does a built mix meet the "enough data" bar (plus variety rules) per language?

usage: data_gate.py <dir with mix4 copies> [--langs de,es,fr,it,nl,pt] [--tier full|next|moderate|light] [--json out.json]
       (--tier omitted = the language's default tier: TIER_LANGS below; full = the first bar, unchanged)
       (files: <dir>/[mix4_]p3_<lang>.jsonl, tts_<lang>, win_<lang>, heldout_<lang>; `modal volume get` copies work)

mix_report.py describes a mix; this tool JUDGES it against the bar, from what the trainer will read:
  * judged rows count toward the bar; synthetic spans count too but are <= 30% of each class (years and
    counts included; 50% for clock / price / phone / date, and for every class in en / es / fr);
  * a spoken number + scale word ("cinco millones") counts toward the million / billion scale; an es / fr day number +
    month is a date and an es "a las siete de la tarde" a clock time (SCALE_WORDS, SPOKEN_DATE, SPOKEN_CLOCK); real number-row hours count real speech only; second-recognizer spans (census_common.EXT_BY) count but their share per
    class must be <= 25%);
  * a YEAR is a bare 4-digit integer 1800-2099 written as one token (fixes the census bug that counted "1000"-"1099"
    as years: those are counts); 1100-1799 is reported apart ("early"), never as a year;
  * per class: spans, hours, distinct values, singleton share, top-1 / top-5 share, synthetic and ext share;
  * years: every 1990-2030 value >= 40 (variety rule; a floor of 10 is the hard minimum), forward 2031-2060 >= 10
    each, 2030/2035/2040/2050 >= 40, >= 110 of 1900-2030 seen, >= 60 values seen >= 10x, singletons <= 20%,
    top value <= 3%;
  * int_small (0-999): >= 2500 spans, each 0-100 >= 10x, each hundred >= 5x, top-5 <= 30%; int_big (>= 1000, not a
    year): >= 400 spans, >= 200 distinct, each scale (thousand / million / billion) >= 50;
  * dec pct cur clock phone >= 300; ord >= 200; date range decade unit >= 150;
  * per language: >= 10 h real number rows, >= 2 h windows, >= 500 distinct videos / sessions, <= 13 rows per
    session, <= 3 rows per digit-masked caption template (a masked caption of < 3 letter words is no template);
  * the decision (held-out) set: >= 150 year spans and >= 30 per claimed class.
Prints PASS / FAIL per check and a one-line verdict per language; exit 1 when any language fails.
"""
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
# the census rules the mix build applies too (one definition: the build's caps and these checks count the same things)
from census_common import (CLOCK_CTX, DOT_CLOCK, EXT_BY, GROUP_CAP, GROUP_CAP_LANG, SPOKEN_CLOCK, SPOKEN_CUR,  # noqa: E402,F401
                           SPOKEN_DATE, TEMPLATE_CAP, by_value, ext_span, gate_class, group_cap, group_id, morph_class,
                           span_class, spoken_shape, synthetic, template_key)

BAR_YEARS = {"de": 2000, "it": 2000, "nl": 2000}
CLASS_MIN = {"dec": 300, "pct": 300, "cur": 300, "clock": 300, "phone": 300, "ord": 200, "date": 150, "range": 150,
             "decade": 150, "unit": 150}
FORWARD_KEY = (2030, 2035, 2040, 2050)


def rows(d, name):
    for p in (f"{d}/{name}.jsonl", f"{d}/mix4_{name}.jsonl", f"{d}/mix4/{name}.jsonl"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
    return []


def scale_of(v):
    return "billion" if v >= 10**9 else "million" if v >= 10**6 else "thousand"


# shapes real speech rarely has may be up to half synthetic; every other class <= 30%
SYN_SHARE_KIND = {"clock": 0.5, "cur": 0.5, "phone": 0.5, "date": 0.5}
# English first, Spanish and French next, filled to the full bar "even if it means using more
# synthetic speech": in these three languages ANY class may be up to half synthetic (mix_build.TTS_SHARE_LANG mirrors it);
# every other language keeps the caps above
SYN_SHARE_LANG = {"en": 0.5, "es": 0.5, "fr": 0.5}
# English phone numbers may be mostly synthetic (real English speech has ~8): the share may
# pass 50% while the synthetic spans stay within the class floor (CLASS_MIN x tier factor, 300 at full tier)
# es/fr fractions and "&" have no real source (their captions hold none), so they get
# the same exemption, with an explicit floor because they have no class minimum
SYN_FLOOR_EXEMPT = {("en", "phone"), ("es", "frac"), ("es", "amp"), ("fr", "frac"), ("fr", "amp")}
SYN_FLOOR_SPANS = {"frac": 150, "amp": 150}  # floor for exempt classes without a CLASS_MIN entry (scaled by the tier factor)


def syn_limit(lang, k):
    return max(SYN_SHARE_KIND.get(k, 0.30), SYN_SHARE_LANG.get(lang, 0.0))


# a spoken number followed by a scale word ("cinco millones", "deux milliards", "3,5 millones") is a million / billion
# figure whatever the caption wrote ("5 millones" or "5.000.000"): the spoken-form label says the scale word either
# way. (word, scale) in the LABEL text right after the span's reading; a reading ending in "mil" + "millones" (es
# "dos mil millones") is a billion. Counted toward "int_big each scale" (2026-10-17).
SCALE_WORDS = {
    "en": (("millions", "million"), ("million", "million"), ("billions", "billion"), ("billion", "billion"),
           ("trillion", "billion")),
    "es": (("mil millones", "billion"), ("millones", "million"), ("millón", "million"), ("billones", "billion"),
           ("billón", "billion")),
    "fr": (("millions", "million"), ("million", "million"), ("milliards", "billion"), ("milliard", "billion")),
    "de": (("millionen", "million"), ("million", "million"), ("milliarden", "billion"), ("milliarde", "billion")),
    "it": (("milioni", "million"), ("milione", "million"), ("miliardi", "billion"), ("miliardo", "billion")),
    "nl": (("miljoen", "million"), ("miljard", "billion")),
    "pt": (("mil milhões", "billion"), ("milhões", "million"), ("milhão", "million"), ("bilhões", "billion"),
           ("bilhão", "billion")),
}



def word_scale(text, reading, lang):
    """'million' / 'billion' when the reading is followed by a scale word in the label text, else None."""
    if not reading or lang not in SCALE_WORDS:
        return None
    t, rd = text.lower(), reading.lower()
    i = t.find(rd)
    if i < 0:
        return None
    nxt = re.sub(r"^[\s,]+", "", t[i + len(rd):])
    for w, sc in SCALE_WORDS[lang]:
        if re.match(re.escape(w) + r"\b", nxt):
            if sc == "million" and lang == "es" and rd.split()[-1] == "mil":
                return "billion"
            return sc
    return None


def check(name, ok, got, want):
    return {"check": name, "pass": bool(ok), "got": got, "want": want}


TIER_LANGS = {"full": "en de es fr it nl pt", "next": "ru pl ro", "moderate": "uk sk fi bg cs hr sv hu",  # uk: (no real word-form source for years / prices / phones)
              "light": "et da lt lv mt sl el"}
TIER_OF = {l: t for t, ls in TIER_LANGS.items() for l in ls.split()}
FWD_KEYS = FORWARD_KEY


def tier_of(lang):
    return TIER_OF.get(lang, "full")


# full = the first bar; next = same checks, counts ~55% (years spread 1990-2030 >= 20, 2031-2060 >= 5, key years >= 20)
TIER_P = {
    "full": dict(f=1.0, yh=3, seen=110, v10=60, each=40, floor=10, fwd=10, key=40, ints=2500, i100=10, hund=5, ih=3,
                 big=400, bigd=200, bigs=50, rh=10, wh=2, grp=500, ho_year=150, ho_cls=30, ext=True),
    "next": dict(f=0.55, yh=1.65, seen=110, v10=60, each=20, floor=10, fwd=5, key=20, ints=1375, i100=6, hund=3, ih=1.65,
                 big=220, bigd=110, bigs=28, rh=5.5, wh=1.1, grp=275, ho_year=80, ho_cls=16, ext=True),
}
# moderate: ~25% counts, the main classes present with value spread; light: no-damage coverage only
MOD = dict(year=375, int_small=625, int_big=100, dec=75, pct=75, clock=75, cur=75, each=8, fwd_key=5, fwd_seen=10,
           top=0.05, rh=2.5, wh=0.5, grp=125, ho_year=40, ho_cls=10, ext=True)
LIGHT = dict(spans=20, distinct=10, top=0.10)


def checks_wave(ctx, P, bar_years):
    cls, f = ctx["cls"], P["f"]
    out = []
    y = cls["year"]
    yv = y["vals"]
    ny = y["spans"]
    ys = round(bar_years * f)
    out.append(check("year spans", ny >= ys, ny, ys))
    out.append(check("year hours", y["hours"] >= P["yh"], round(y["hours"], 2), P["yh"]))
    seen = sum(1 for v in range(1900, 2031) if yv[v])
    out.append(check("years 1900-2030 seen", seen >= P["seen"], seen, P["seen"]))
    n10 = sum(1 for c in yv.values() if c >= 10)
    out.append(check("year values seen >=10x", n10 >= P["v10"], n10, P["v10"]))
    e = P["each"]
    low = [v for v in range(1990, 2031) if yv[v] < e]
    out.append(check(f"every 1990-2030 >= {e}", not low, f"{41 - len(low)}/41 ok", "41/41"))
    fl = P["floor"]
    out.append(check(f"every 1990-2030 >= {fl} (hard floor)", all(yv[v] >= fl for v in range(1990, 2031)),
                     f"{sum(yv[v] >= fl for v in range(1990, 2031))}/41", "41/41"))
    fw = [v for v in range(2031, 2061) if yv[v] < P["fwd"]]
    out.append(check(f"every 2031-2060 >= {P['fwd']}", not fw, f"{30 - len(fw)}/30 ok", "30/30"))
    out.append(check(f"2030/2035/2040/2050 >= {P['key']}", all(yv[v] >= P["key"] for v in FORWARD_KEY),
                     {v: yv[v] for v in FORWARD_KEY}, P["key"]))
    single = sum(1 for c in yv.values() if c == 1) / max(1, len(yv))
    out.append(check("year singleton share <= 20%", single <= 0.20, round(single, 3), 0.20))
    top1 = max(yv.values(), default=0) / max(1, ny)
    out.append(check("top year value <= 3%", top1 <= 0.03, round(top1, 3), 0.03))
    s = cls["int_small"]
    out.append(check("int_small spans", s["spans"] >= P["ints"], s["spans"], P["ints"]))
    t = P["i100"]
    out.append(check(f"each 0-100 >= {t}x", all(s["vals"][v] >= t for v in range(0, 101)),
                     f"{sum(s['vals'][v] >= t for v in range(0, 101))}/101", "101/101"))
    h = P["hund"]
    out.append(check(f"each hundred >= {h}x", all(s["vals"][v] >= h for v in range(100, 1000, 100)),
                     f"{sum(s['vals'][v] >= h for v in range(100, 1000, 100))}/9", "9/9"))
    top5 = sum(c for _, c in s["vals"].most_common(5)) / max(1, s["spans"])
    out.append(check("int_small top-5 <= 30%", top5 <= 0.30, round(top5, 3), 0.30))
    out.append(check("int_small hours", s["hours"] >= P["ih"], round(s["hours"], 2), P["ih"]))
    b = cls["int_big"]
    sc = collections.Counter(scale_of(v) for v, c in b["vals"].items() for _ in range(c))
    out.append(check("int_big spans", b["spans"] >= P["big"], b["spans"], P["big"]))
    out.append(check("int_big distinct", len(b["vals"]) >= P["bigd"], len(b["vals"]), P["bigd"]))
    sc.update(ctx.get("word_scales", {}))  # spoken number + scale word (SCALE_WORDS)
    out.append(check(f"int_big each scale >= {P['bigs']}", all(sc[k] >= P["bigs"] for k in ("thousand", "million", "billion")),
                     dict(sc), P["bigs"]))
    mins = {k: round(m * f) for k, m in CLASS_MIN.items()}
    for k, m in mins.items():
        out.append(check(f"{k} spans", cls[k]["spans"] >= m, cls[k]["spans"], m))
    out.append(check("real number-row hours", ctx["real_h"] >= P["rh"], round(ctx["real_h"], 2), P["rh"]))
    out.append(check("real window hours", ctx["wh"] >= P["wh"], round(ctx["wh"], 2), P["wh"]))
    out.append(check("distinct videos/sessions", len(ctx["groups"]) >= P["grp"], len(ctx["groups"]), P["grp"]))
    hc = ctx["hc"]
    out.append(check("decision-set year spans", hc["year"] >= P["ho_year"], hc["year"], P["ho_year"]))
    for k in [k for k in mins if cls[k]["spans"] >= mins[k]]:  # claimed = meets its training bar
        out.append(check(f"decision-set {k} spans", hc[k] >= P["ho_cls"], hc[k], P["ho_cls"]))
    return out


def checks_moderate(ctx):
    cls, M = ctx["cls"], MOD
    out = []
    for k in ("year", "int_small", "int_big", "dec", "pct", "clock", "cur"):
        out.append(check(f"{k} spans", cls[k]["spans"] >= M[k], cls[k]["spans"], M[k]))
    yv = cls["year"]["vals"]
    e = M["each"]
    low = [v for v in range(1990, 2031) if yv[v] < e]
    out.append(check(f"every 1990-2030 >= {e}", not low, f"{41 - len(low)}/41 ok", "41/41"))
    fwd_seen = sum(1 for v in range(2031, 2061) if yv[v])
    out.append(check(f"forward years present ({M['fwd_seen']}+ of 2031-2060)", fwd_seen >= M["fwd_seen"], fwd_seen, M["fwd_seen"]))
    out.append(check(f"2030/2035/2040/2050 >= {M['fwd_key']}", all(yv[v] >= M["fwd_key"] for v in FORWARD_KEY),
                     {v: yv[v] for v in FORWARD_KEY}, M["fwd_key"]))
    for k in ("year", "int_small", "int_big"):
        x = cls[k]
        t = max(x["vals"].values(), default=0) / max(1, x["spans"])
        out.append(check(f"top {k} value <= 5%", t <= M["top"], round(t, 3), M["top"]))
    out.append(check("real number-row hours", ctx["real_h"] >= M["rh"], round(ctx["real_h"], 2), M["rh"]))
    out.append(check("real window hours", ctx["wh"] >= M["wh"], round(ctx["wh"], 2), M["wh"]))
    out.append(check("distinct videos/sessions", len(ctx["groups"]) >= M["grp"], len(ctx["groups"]), M["grp"]))
    hc = ctx["hc"]
    out.append(check("decision-set year spans", hc["year"] >= M["ho_year"], hc["year"], M["ho_year"]))
    for k in ("dec", "pct", "clock", "cur"):
        if cls[k]["spans"] >= M[k]:
            out.append(check(f"decision-set {k} spans", hc[k] >= M["ho_cls"], hc[k], M["ho_cls"]))
    return out


def checks_light(ctx):
    out = []
    L = LIGHT
    for k, x in sorted(ctx["cls"].items()):
        if k in ("year_early", "int_other") or not x["spans"]:
            continue  # reported apart / not a class the trainer sees
        out.append(check(f"{k} spans >= {L['spans']}", x["spans"] >= L["spans"], x["spans"], L["spans"]))
        out.append(check(f"{k} distinct >= {L['distinct']}", len(x["vals"]) >= L["distinct"], len(x["vals"]), L["distinct"]))
        t = max(x["vals"].values(), default=0) / max(1, x["spans"])
        out.append(check(f"top {k} value <= 10%", t <= L["top"], round(t, 3), L["top"]))
    return out


def common_checks(ctx, tier):
    """Every tier: synthetic share <= 30% per class (syn_limit: 50% for clock / price / phone / date, and for every
    class of SYN_SHARE_LANG), session / template caps; ext share <= 25% except light."""
    out = []
    cls = ctx["cls"]
    if tier != "light":
        for k, x in cls.items():
            if x["spans"] >= 20:
                e = x["ext"] / x["spans"]
                out.append(check(f"{k} ext share <= 25%", e <= 0.25, round(e, 3), 0.25))
    for k, x in cls.items():  # synthetic is fine for every class, years and counts too, at <= 30% of it
        if x["syn"]:
            sh = x["syn"] / x["spans"]
            lim = syn_limit(ctx["lang"], k)
            if (ctx["lang"], k) in SYN_FLOOR_EXEMPT and (k in CLASS_MIN or k in SYN_FLOOR_SPANS):
                floor = round(CLASS_MIN.get(k, SYN_FLOOR_SPANS.get(k, 0)) * TIER_P.get(tier, {"f": 1.0})["f"])
                out.append(check(f"{k} synthetic share <= {lim:.0%} or synthetic spans <= floor {floor}",
                                 sh <= lim or x["syn"] <= floor, {"share": round(sh, 3), "synthetic": x["syn"]},
                                 {"share": lim, "floor": floor}))
                continue
            out.append(check(f"{k} synthetic share <= {lim:.0%}", sh <= lim, round(sh, 3), lim))
    groups = ctx["groups"]
    fb = sum(c for g, c in groups.items() if g.startswith("id:")) / max(sum(groups.values()), 1)
    out.append(check("rows with a known video/session id", fb <= 0.05, f"{1 - fb:.1%}", ">= 95%"))  # else variety is inflated
    big = sum(1 for g, c in groups.items() if c > group_cap(ctx["lang"], g))
    caps = GROUP_CAP_LANG.get(ctx["lang"])
    out.append(check("groups with > 13 rows" + (f" ({', '.join(f'{k} {v}' for k, v in caps.items())})" if caps else ""),
                     big == 0, big, 0))
    tmpl = sum(c - TEMPLATE_CAP for c in ctx["templates"].values() if c > TEMPLATE_CAP)
    out.append(check("rows over the 3-per-template cap", tmpl == 0, tmpl, 0))
    return out


def gate_lang(d, lang, tier=None):
    # every row of the synthetic part is synthetic, whatever the row's own fields say
    train = rows(d, f"p3_{lang}") + [dict(r, synthetic=True) for r in rows(d, f"tts_{lang}")]
    win = rows(d, f"win_{lang}")
    ho = [r for r in rows(d, f"heldout_{lang}") if not synthetic(r)]
    cls = collections.defaultdict(lambda: {"spans": 0, "hours": 0.0, "vals": collections.Counter(), "syn": 0, "ext": 0})
    real_h = 0.0
    groups = collections.Counter()
    templates = collections.Counter()
    word_scales = collections.Counter()  # (scale) -> spans: spoken number + scale word
    for r in train:
        sps = r.get("spans") or []
        syn = synthetic(r)
        if not syn and sps:
            real_h += r.get("duration", 0) / 3600
            groups[group_id(r)] += 1
            t = template_key(r)  # a short masked caption ("# euro.") is no template (mix_build caps the same way)
            if t:
                templates[t] += 1
        for sp in sps:
            c, v = morph_class(r.get("text", ""), sp, *span_class(sp), lang)  # hu / fi suffixes, span scale words
            c0, c = c, spoken_shape(r.get("text", ""), sp, c, v, lang)  # c0: the caption class (scale words below)
            x = cls[c]
            if syn:  # synthetic counts toward the bar, capped per class (check below)
                x["syn"] += 1
            x["spans"] += 1
            x["hours"] += r.get("duration", 0) / 3600 / max(1, len(sps))
            x["vals"][v] += 1
            x["ext"] += ext_span(sp)  # census_common.EXT_BY: decided mainly by the second transcript
            if c0 in ("int_small", "int_big", "int_other", "dec") and not (c0 == "int_big" and isinstance(v, int) and v >= 10**6):
                ws = word_scale(r.get("text", ""), sp.get("reading", ""), lang)
                if ws:
                    word_scales[ws] += 1
    tier = tier or tier_of(lang)
    y = cls["year"]
    detail = {"year_values": {str(v): c for v, c in sorted(y["vals"].items()) if isinstance(v, int) and 1900 <= v <= 2099},
              "int_small_values": {str(v): c for v, c in sorted(cls["int_small"]["vals"].items()) if v <= 100 or v % 100 == 0},
              "scales": dict(collections.Counter(scale_of(v) for v, c in cls["int_big"]["vals"].items() for _ in range(c))),
              "word_scales": dict(word_scales), "real_hours": round(real_h, 2), "groups": len(groups),
              "mean_row_s": round(sum(r.get("duration", 0) for r in train if not synthetic(r) and r.get("spans"))
                                  / max(1, sum(1 for r in train if not synthetic(r) and r.get("spans"))), 2),
              "class_spans_total": {k: x["spans"] for k, x in cls.items()}}
    wh = sum(w.get("duration", 0) for w in win if not synthetic(w)) / 3600
    hc = collections.Counter(spoken_shape(r.get("text", ""), sp, *morph_class(r.get("text", ""), sp, *span_class(sp), lang), lang)
                             for r in ho for sp in r.get("spans") or [])
    ctx = {"cls": cls, "real_h": real_h, "wh": wh, "groups": groups, "templates": templates, "hc": hc, "lang": lang,
           "word_scales": word_scales}
    if tier in ("full", "next"):
        out = checks_wave(ctx, TIER_P[tier], BAR_YEARS.get(lang, 1500))
    elif tier == "moderate":
        out = checks_moderate(ctx)
    else:
        out = checks_light(ctx)
    out += common_checks(ctx, tier)
    summary = {k: {"spans": x["spans"], "hours": round(x["hours"], 2), "distinct": len(x["vals"]),
                   "singleton_share": round(sum(1 for c in x["vals"].values() if c == 1) / max(1, len(x["vals"])), 3),
                   "top5_share": round(sum(c for _, c in x["vals"].most_common(5)) / max(1, x["spans"]), 3),
                   "synthetic_spans": x["syn"], "ext_share": round(x["ext"] / max(1, x["spans"]), 3)}
               for k, x in sorted(cls.items(), key=lambda t: -t[1]["spans"])}
    return {"tier": tier, "checks": out, "detail": detail, "classes": summary, "year_early_spans": cls["year_early"]["spans"],
            "heldout_classes": dict(hc)}


def main(argv):
    d = argv[0]
    langs = (argv[argv.index("--langs") + 1] if "--langs" in argv else "de,es,fr,it,nl,pt").split(",")
    tier_arg = argv[argv.index("--tier") + 1] if "--tier" in argv else None
    if tier_arg not in (None, *TIER_P, "moderate", "light"):
        print(f"unknown --tier {tier_arg}")
        return 2
    res, bad = {}, 0
    for lang in langs:
        if not rows(d, f"p3_{lang}"):
            print(f"{lang}: FAIL, no p3_{lang}.jsonl under {d}")
            bad += 1
            continue
        g = res[lang] = gate_lang(d, lang, tier_arg)
        fails = [c for c in g["checks"] if not c["pass"]]
        bad += bool(fails)
        verdict = "PASS" if not fails else f"FAIL ({len(fails)} of {len(g['checks'])} checks)"
        print(f"\n== {lang} [{g['tier']}]: {verdict}"
              f"  (1100-1799 'early' int spans kept out of years: {g['year_early_spans']})")
        for c in g["checks"]:
            print(f"  {'ok  ' if c['pass'] else 'FAIL'} {c['check']:38s} got {c['got']}  want {c['want']}")
        print(f"  {'class':12s} {'spans':>6s} {'h':>6s} {'dist':>5s} {'single':>6s} {'top5':>5s} {'syn':>5s} {'ext':>5s}")
        for k, x in g["classes"].items():
            print(f"  {k:12s} {x['spans']:6d} {x['hours']:6.2f} {x['distinct']:5d} {x['singleton_share']:6.2f} "
                  f"{x['top5_share']:5.2f} {x['synthetic_spans']:5d} {x['ext_share']:5.2f}")
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
