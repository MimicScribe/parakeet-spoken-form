"""Census rules shared by the mix build (mix_build, ml_balance, ml_p3) and the data gate / held-out carve
(eval/quality/data_gate, carve_heldout), so the build's caps and the gate's checks count the same things:

  * EXT_BY / ext_span / ext_row: a span decided mainly by the second transcript;
  * synthetic(row): a synthetic-speech row;
  * group_id / group_cap: the unit of speaker / topic variety and its row cap;
  * template_key: the digit-masked caption template the template cap counts;
  * held_unit: the unit a held-out id takes with it (a parliament day, a video);
  * span_class / morph_class / spoken_shape / gate_class: the census class of a span.

Pure Python (re only): importable from the build containers and from the local gate alike.
"""
import re

# Spans whose reading (or whose acceptance) rests mainly on the second transcript rather than on the CTC reader:
#   ext           the CTC margin cleared the bar but the greedy text did not fit; the transcript's agreement decided
#   ext_pick      the transcript chose between readings the CTC could not tell apart
#   ext_ctx       the CTC margin / greedy fit could not settle the span; the transcript holds the reading in its caption
#                 place. The CTC best must still be that reading, but only by EXT_CTX_MARGIN (1 nat) over rivals, a
#                 third of the normal 3-nat bar: that is not an independent CTC decision, so it counts here
#   ext_value     the transcript's digits in the caption place fix the value and stand in for the greedy fit check
#                 (Dutch); the same role as "ext", so it counts here
#   ext_regional  the transcript chose a regional reading the CTC pass had not decided
# Not counted: by "ctc" spans that carry the ext_value FLAG (the 16-language parliament path). There the CTC picks the
# reading (fit + margin over rivals, or a single reading that follows from the value), and the transcript's value check
# is an extra filter on top: the CTC still decides. Counting them would put every such span under the ext caps.
EXT_BY = ("ext", "ext_pick", "ext_ctx", "ext_value", "ext_regional")


def ext_span(sp: dict) -> bool:
    """Was this span decided mainly by the second transcript (EXT_BY)?"""
    return sp.get("by") in EXT_BY


def ext_row(r: dict) -> bool:
    """Does any span of this row rest mainly on the second transcript?"""
    return any(ext_span(sp) for sp in r.get("spans") or [])


def synthetic(r):
    return r.get("src") == "tts" or str(r.get("origin", "")).startswith("tts") or bool(r.get("synthetic"))


def group_id(r):
    """Video (YODAS), plenary session (VoxPopuli) or sentence (FLEURS): the unit of speaker/topic variety."""
    i = str(r.get("id", ""))
    m = re.match(r"^[a-z]{2}\d{3}_\d+_(.{11})_", i)
    if m:
        return "yt:" + m.group(1)
    if "PLENARY" in i or re.match(r"^\d{8}-", i):
        return "ep:" + i.split("_")[0]
    m = re.match(r"^fleurs\w*?_(\d+)_", i)
    if m:
        return "fl:" + m.group(1)
    if i.startswith("cv22_"):  # Common Voice: up to 3 speakers read one sentence; the sentence is the unit
        t = str(r.get("caption") or r.get("text") or "")
        return "cv:" + " ".join(re.findall(r"[^\W_]+", t.lower().replace("’", "'")))
    return "id:" + i


def span_class(sp):
    """(class, value or None). Years are decided on the WRITTEN caption token, never on the value alone."""
    k, t = sp.get("kind", "?"), sp.get("text", "").strip()
    if k != "int":
        return k, t
    if re.fullmatch(r"\d{4}", t):
        v = int(t)
        if 1800 <= v <= 2099:
            return "year", v
        if 1100 <= v <= 1799:
            return "year_early", v
    digits = re.sub(r"[\s.,'  ’]", "", t)
    if digits.isdigit():
        v = int(digits)
        return ("int_small" if v < 1000 else "int_big"), v
    return "int_other", t



# the English YouTube top-up keeps its 30-rows-per-video cap; every other group keeps 13
GROUP_CAP = 13
GROUP_CAP_LANG = {"en": {"yt": 30}}


def group_cap(lang, g):
    return GROUP_CAP_LANG.get(lang, {}).get(g.split(":", 1)[0], GROUP_CAP)


# a day number said with its month ("quince de marzo", "le dix-huit juillet") is a date, and a Spanish hour said with
# "a las ... de la tarde / horas / y media" is a clock time, whatever the caption wrote ("15 de marzo", "a las 7"): the
# labeller marks them int, the spoken-form label is the same words either way (2026-10-17; English: en_adapter does the
# same from the caption). Only the languages listed here; the rest keep the caption kinds.
_M = {"es": "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre",
      "fr": "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"}
SPOKEN_DATE = {"es": re.compile(r"^\s*de\s+(?:" + _M["es"] + r")\b"), "fr": re.compile(r"^\s*(?:" + _M["fr"] + r")\b")}
# de / it / nl / pt (2026-10-17, parliament fill): the day is often an ORDINAL span (de "am 6. Dezember" -> "sechsten
# Dezember", pt "1.º de maio"), so ord spans with a day value count too (spoken_shape)
_M.update({"de": "januar|jänner|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember",
           "it": "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre",
           "nl": "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december",
           "pt": "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"})
SPOKEN_DATE.update({"de": re.compile(r"^\s*(?:" + _M["de"] + r")\b"), "it": re.compile(r"^\s*(?:" + _M["it"] + r")\b"),
                    "nl": re.compile(r"^\s*(?:" + _M["nl"] + r")\b"),
                    "pt": re.compile(r"^\s*(?:de\s+)?(?:" + _M["pt"] + r")\b")})
# Italian "alle 18.30" is labelled a decimal span whose reading is the time ("diciotto e trenta"): a dec span written
# H.MM (H <= 23, MM <= 59) without the decimal word in its reading is a clock time
DOT_CLOCK = {"it": "virgola"}
SPOKEN_CLOCK = {"es": (re.compile(r"\b(?:a|desde|hasta|de|para|sobre)\s+las?\s+$"),
                       re.compile(r"^\s*(?:horas?|y media|y cuarto|menos cuarto|de la mañana|de la tarde|de la noche|"
                                  r"de la madrugada|en punto)\b"))}


# (English census rule, extended to es / fr): a number followed by a currency word is a
# price ("treinta euros", "deux millions d'euros"); parliament captions write "30 euros", which the labeller marks int
SPOKEN_CUR = {"es": re.compile(r"^\s*(?:(?:mil\s+)?mill(?:ones|ón)\s+)?(?:de\s+)?(?:euros?|dólares|dolares|libras|"
                               r"pesos|céntimos|centimos)\b"),
              "fr": re.compile(r"^\s*(?:(?:millions?|milliards?)\s+)?(?:d'|de\s+)?(?:euros?|dollars?|livres|francs?|"
                               r"centimes)\b")}

# moderate / next tiers (2026-10-17 census fix, counts only): a spoken clock time said with the hour word or a
# clock preposition ("klockan tjugoett", "kello 17", "16 órakor", "v 15 hodin", "o godzinie 15", "la ora 15"):
# (before-the-reading regex or None, after-the-reading regex or None); every regex given must match
CLOCK_CTX = {"sv": (re.compile(r"\b(?:klockan|kl\.?)\s+$"), None),
             "fi": (re.compile(r"\b(?:kello|klo)\s+$"), None),
             "hu": (None, re.compile(r"^\s*(?:órakor|óráig|órától|órára)(?!\w)")),
             "cs": (re.compile(r"\b(?:v|ve|od|do|kolem|okolo)\s+$"), re.compile(r"^\s*hod(?:in|iny|ině)(?!\w)")),
             "sk": (re.compile(r"\b(?:o|od|do|okolo)\s+$"), re.compile(r"^\s*hod(?:ine|iny|ín|in)(?!\w)")),
             "pl": (re.compile(r"\b(?:godzinie|godziny|godzinę|godz\.)\s+$"), None),
             "ro": (re.compile(r"\b(?:ora|orele|orei)\s+$"), None)}
# a scale word inside an int span's own text ("12 millió", "200 miljoonaa", "450 miljoner", "45 tisíc"): the number
# part times the scale is an int_big figure (stems, first match wins)
SPAN_SCALE = {"hu": (("milliárd", 10**9), ("millió", 10**6), ("ezer", 10**3)),
              "fi": (("miljard", 10**9), ("miljoon", 10**6), ("tuha", 10**3)),
              "sv": (("miljard", 10**9), ("miljon", 10**6), ("tusen", 10**3)),
              "cs": (("miliard", 10**9), ("milion", 10**6), ("tisíc", 10**3)),
              "sk": (("miliard", 10**9), ("milión", 10**6), ("tisíc", 10**3)),
              "pl": (("miliard", 10**9), ("milion", 10**6), ("tysi", 10**3)),
              "ro": (("miliard", 10**9), ("milion", 10**6), ("mii", 10**3))}
# Hungarian case ending / postposition on a number ("2015-ben", "15-én", "1990-es", "2001 óta"), Finnish case ending
# ("2015:ssä") or year word ("vuonna 2015"): the number part is classified by its value and shape
HU_SUFFIX = re.compile(r"^(\d{1,4})(?:-([a-záéíóöőúüű]{1,8})(?:\s+\S+)?|\s+(?:óta|előtt|után|során|végéig|végén))$")
HU_DAY = {"án", "én", "ától", "étől", "áig", "éig", "ai", "ei", "ára", "ére", "ika", "ike", "ikán", "ikén"}
FI_SUFFIX = re.compile(r"^(?:(?:vuonna|vuodesta|vuoteen|vuoden)\s+)?(\d{1,4})(?::([a-zåäö]{1,6}))?$")


def by_value(v, day=False):
    """(class, value) of a bare integer by its value and digit shape (span_class rules; day: a day-of-month ending)."""
    if day and 1 <= v <= 31:
        return "date", v
    if 1800 <= v <= 2099 and len(str(v)) == 4:
        return "year", v
    if 1100 <= v <= 1799 and len(str(v)) == 4:
        return "year_early", v
    return ("int_small" if v < 1000 else "int_big"), v


def morph_class(text, sp, c, v, lang):
    """An int_other span of a suffixing language (hu / fi case endings, a scale word or clock word inside the span
    text) reclassified by its number part; any other span unchanged. Counts only: the label is not touched."""
    if c != "int_other" or not isinstance(v, str):
        return c, v
    t = v.strip()
    if lang == "hu" and (m := HU_SUFFIX.match(t)):
        return by_value(int(m.group(1)), day=(m.group(2) or "") in HU_DAY)
    if lang == "fi" and (m := FI_SUFFIX.match(t)) and t != m.group(1):
        return by_value(int(m.group(1)))
    m = re.match(r"^(\d{1,3}(?:[  .]\d{3})*|\d+)\s+(\S+)", t)
    if not m:
        return c, v
    n, word = int(re.sub(r"\D", "", m.group(1))), m.group(2).lower()
    for stem, mult in SPAN_SCALE.get(lang, ()):
        if word.startswith(stem):
            return by_value(n * mult)
    if lang in CLOCK_CTX and 0 <= n <= 24:
        b, a = CLOCK_CTX[lang]
        rd = (sp.get("reading") or "").lower()
        tl = text.lower()
        i = tl.find(rd) if rd else -1
        before = tl[:i] if i >= 0 else ""
        if a is not None and a.match(t[len(m.group(1)):].lower()) and (b is None or b.search(before)):
            return "clock", n
    return c, v


# ordinal word endings: de "zehnte(n|r|s|m)", "erste", "dritte", "zwanzigste"; nl "tiende", "eerste", "twintigste"
ORD_WORD = {"de": re.compile(r"(?:te|ten|ter|tes|tem)$"), "nl": re.compile(r"(?:de|ste)$")}


def spoken_shape(text, sp, c, v, lang):
    """The class of an int span from the label words around it (SPOKEN_DATE / SPOKEN_CLOCK / SPOKEN_CUR), else c."""
    rd0 = (sp.get("reading") or "").lower()
    if lang in SPOKEN_CUR and c in ("int_small", "int_big", "dec") and rd0 and rd0 in text.lower():
        t0 = text.lower()
        if SPOKEN_CUR[lang].match(t0[t0.find(rd0) + len(rd0):]):
            return "cur"
    if c == "dec" and lang in DOT_CLOCK and re.fullmatch(r"\s*([01]?\d|2[0-3])\.[0-5]\d\s*", str(v)) and rd0 \
            and DOT_CLOCK[lang] not in rd0:
        return "clock"
    if c == "ord" and lang in ("de", "it", "nl", "pt") and re.fullmatch(r"\s*\d{1,2}\s*(?:\.|º|\.º|°)?\s*", str(v)):
        c, v = "int_small", int(re.sub(r"\D", "", str(v)))  # a day number written as an ordinal; reverts below if no month
        if not (1 <= v <= 31 and SPOKEN_DATE[lang].match((lambda t, r: t[t.find(r) + len(r):] if r and r in t else "")(
                text.lower(), rd0))):
            return "ord"
    if lang in CLOCK_CTX and c in ("int_small", "ord") and re.fullmatch(r"\s*\d{1,2}\s*\.?\s*", str(v)) \
            and int(re.sub(r"\D", "", str(v))) <= 24 and rd0 and rd0 in text.lower():
        b, a = CLOCK_CTX[lang]
        t0 = text.lower()
        i0 = t0.find(rd0)
        if (b is None or b.search(t0[:i0])) and (a is None or a.match(t0[i0 + len(rd0):])):
            return "clock"
    if c != "int_small" or not isinstance(v, int) or not (lang in SPOKEN_DATE or lang in SPOKEN_CLOCK):
        return c
    t, rd = text.lower(), rd0
    i = t.find(rd) if rd else -1
    if i < 0:
        return c
    after, before = t[i + len(rd):], t[:i]
    if 1 <= v <= 31 and lang in SPOKEN_DATE and SPOKEN_DATE[lang].match(after):
        return "date"
    if 1 <= v <= 24 and lang in SPOKEN_CLOCK and SPOKEN_CLOCK[lang][0].search(before) and SPOKEN_CLOCK[lang][1].match(after):
        return "clock"
    return c



TEMPLATE_CAP = 3  # real number rows per digit-masked caption template
TEMPLATE_MIN_WORDS = 3  # a masked caption with fewer letter words ("#.", "# euro.") is a short answer, not a template


def template_key(r: dict) -> str:
    """The digit-masked caption template; "" (not capped, not counted) when it has < TEMPLATE_MIN_WORDS letter words."""
    t = re.sub(r"\d+", "#", str(r.get("caption", "")).lower()).strip()
    return t if len(re.findall(r"[^\W\d_]+", t)) >= TEMPLATE_MIN_WORDS else ""


def held_unit(rid: str) -> str | None:
    """The held-out unit of a row id: the parliament day (plenary AND committee sessions of that day:
    "20180618-0900-PLENARY-...", "20180618-1500-COMMITTEE-...") or the YODAS video; None for other ids."""
    rid = str(rid)
    m = re.match(r"(\d{8})-\d{4}-[A-Z]", rid)
    if m:
        return "day:" + m.group(1)
    m = re.match(r"[a-z]{2}\d+_\d+_(.{11})_", rid)
    return "vid:" + m.group(1) if m else None


_GROUP_LINE = re.compile(r"^#\s*group\s+(vox|yodas):(\S+)")


def read_held_list(path: str) -> tuple:
    """(ids, units) of a held-out list (heldout_<lang>.txt). "#" lines are comments, never ids; a carve's
    "# group vox:<day>" / "# group yodas:<video>" line adds that unit. units = held_unit of every id + the group lines."""
    ids, units = set(), set()
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if not t:
            continue
        if t.startswith("#"):
            m = _GROUP_LINE.match(t)
            if m:
                units.add(("day:" if m.group(1) == "vox" else "vid:") + m.group(2))
            continue
        ids.add(t)
    return ids, units | ({held_unit(i) for i in ids} - {None})


def gate_class_value(r: dict, sp: dict, lang: str) -> tuple:
    """(class, value) of span `sp` in row `r` exactly as the data gate counts it: span_class, then the suffix / scale
    rules of morph_class (hu "2015-ben", fi "2015:ssä" are years by their number part), then the spoken shapes of
    spoken_shape. The one year definition: a span is a year of value v iff this returns ("year", v)."""
    t = r.get("text", "")
    c, v = morph_class(t, sp, *span_class(sp), lang)
    c, v = cardinal_ord(sp, c, v, lang)
    return spoken_shape(t, sp, c, v, lang), v


def cardinal_ord(sp, c, v, lang):
    """A de / nl "ord" span that is a sentence-final cardinal ("Punkt 10." read "zehn", "2008." read
    "zweitausendacht"): the period made it look like an ordinal, the reading says it is not one, so it is classified
    by its value. Counts only: the label is not touched. Any other span unchanged."""
    rd = (sp.get("reading") or "").split()
    if c != "ord" or lang not in ORD_WORD or not rd or not isinstance(v, str):
        return c, v
    m = re.fullmatch(r"\s*(\d{1,4})\s*\.\s*", v)
    last = re.sub(r"[^\w]+$", "", rd[-1].lower())
    if not m or not last or ORD_WORD[lang].search(last):
        return c, v
    return by_value(int(m.group(1)))


def gate_class(r: dict, sp: dict, lang: str) -> str:
    """The census class of span `sp` in row `r` as the data gate counts it (gate_class_value without the value)."""
    return gate_class_value(r, sp, lang)[0]


def year_of(r: dict, sp: dict, lang: str = "") -> int | None:
    """The year value of a span as the data gate counts years (gate_class_value), else None."""
    c, v = gate_class_value(r, sp, lang)
    return v if c == "year" and isinstance(v, int) else None


# ---------------------------------------------------------------- tiers and synthetic-share limits (data gate rules)
CLASS_MIN = {"dec": 300, "pct": 300, "cur": 300, "clock": 300, "phone": 300, "ord": 200, "date": 150, "range": 150,
             "decade": 150, "unit": 150}
TIER_LANGS = {"full": "en de es fr it nl pt", "next": "ru pl ro", "moderate": "uk sk fi bg cs hr sv hu",
              "light": "et da lt lv mt sl el"}  # uk moderate: no real word-form source for years / prices / phones
TIER_OF = {l: t for t, ls in TIER_LANGS.items() for l in ls.split()}
TIER_FACTOR = {"full": 1.0, "next": 0.55}  # count factor of a tier's bar (the gate's TIER_P "f"); others 1.0


def tier_of(lang):
    return TIER_OF.get(lang, "full")


# shapes real speech rarely has may be up to half synthetic; every other class <= 30%
SYN_SHARE_KIND = {"clock": 0.5, "cur": 0.5, "phone": 0.5, "date": 0.5}
# English first, Spanish and French next, filled to the full bar even with more synthetic speech: in these three
# languages ANY class may be up to half synthetic; every other language keeps the caps above
SYN_SHARE_LANG = {"en": 0.5, "es": 0.5, "fr": 0.5}
# English phone numbers may be mostly synthetic (real English speech has ~8): the share may pass 50% while the
# synthetic spans stay within the class floor (CLASS_MIN x tier factor); es / fr fractions and "&" have no real source,
# so they get the same exemption, with an explicit floor because they have no class minimum
SYN_FLOOR_EXEMPT = {("en", "phone"), ("es", "frac"), ("es", "amp"), ("fr", "frac"), ("fr", "amp")}
SYN_FLOOR_SPANS = {"frac": 150, "amp": 150}


def syn_limit(lang, k):
    """Max synthetic share of a class's spans in a language."""
    return max(SYN_SHARE_KIND.get(k, 0.30), SYN_SHARE_LANG.get(lang, 0.0))


def syn_floor(lang, k, tier=None):
    """The synthetic span count an exempt (lang, class) may hold whatever its share (SYN_FLOOR_EXEMPT), else None."""
    if (lang, k) not in SYN_FLOOR_EXEMPT or not (k in CLASS_MIN or k in SYN_FLOOR_SPANS):
        return None
    return round(CLASS_MIN.get(k, SYN_FLOOR_SPANS.get(k, 0)) * TIER_FACTOR.get(tier or tier_of(lang), 1.0))


def syn_ok(lang, k, syn, spans, tier=None) -> bool:
    """The gate's synthetic check of one class: share <= syn_limit, or (exempt classes) synthetic spans <= syn_floor."""
    if not syn:
        return True
    fl = syn_floor(lang, k, tier)
    return syn / max(spans, 1) <= syn_limit(lang, k) or (fl is not None and syn <= fl)
