"""Deterministic grammar checks for SYNTHETIC readings (the spoken words a generator puts in a label and voices).

Each check finds one known generator slip (code: what it catches):

  date_ord_gender     es / pt / it / fr "1º" / "1er" before a month in a feminine form ("el primera de octubre",
                      "il prima febbraio", "le première octobre")
  date_ord_apocope    es "el primer de octubre" (the apocope belongs before a masculine noun; dates say "primero")
  pt_clock_hour_masc  pt clock hours 1, 2, 21, 22 are feminine ("às duas e cinco", never "dois e cinco")
  es_clock_hour_masc  es clock hours 1 / 21 are feminine ("la una y cinco", "las veintiuna horas")
  fr_clock_et         fr "et" between the hour and the minutes ("seize heures et quinze"); "et (un) quart", "et demie"
                      and the "et" inside "vingt et un" are right
  fr_clock_hour_un    fr hour 1 / 21 masculine ("vingt et un heures": the hour is feminine, "vingt et une heures")
  fr_12am_douze       fr 12:xx a.m. read "douze ..." (that is noon; midnight is "minuit" / "zéro heure")
  it_clock_ore        it "<hour> ore ..." for a clock time ("sedici ore e un quarto" is a duration)
  clock_midnight_noon it 0:xx read "dodici ..." (that is 12:xx)
  count_digit_chunks  a count read as digit chunks ("51000 funcionários" -> "cinco dez zero zero")
  hour_noun_number    hour noun in the wrong number ("vingt heure", "dezenove hora", "une heures", "vingt et une
                      heure", fr "zéro heures")
  cents_word          es "dólares / pesos ... céntimos", pt "reais / dólares ... cêntimos" (those cents are centavos)
  price_two_numbers   de / nl price with cents and no unit word joined by "und" / "en" ("achtundvierzig und zehn":
                      heard as two numbers)
  frac_half_unit      "½" before a unit read as a noun ("medio de litro", "un mezzo di chilo", "één half liter")
  nl_een_half         nl "en één half" (the stressed numeral; "en een half" / "-enhalf" are the spoken forms)
  it_spelling         it "ventiseesimo", unaccented final "-tre" in compounds ("duemilatre"; standard "duemilatré")
  slav_cost_acc       uk / ru feminine "одна" after "коштує" / "стоит" (the verb takes the accusative "одну")
  cur_word_mismatch   ro / pl reading names another currency or its cents ("3,50 €" -> "trei euro și cincizeci de
                      bani", "74,11 €" -> "siedemdziesiąt cztery złote ...")
  ro_de_linker        ro "de" before the noun iff the last two digits are 00 or 20-99 ("douăzeci de euro",
                      "nouăsprezece euro", "o sută cinci lei")
  ro_noun_gender      ro feminine "două" / "una" before a masculine / neuter currency noun ("două de euro")
  ro_gender           ro masculine "doi" before a feminine scale / count noun or clock hour (ro_gender.violations)
  ro_o_mie            ro bare "mie" for one thousand ("o mie nouă sute")
  pl_clock_minute     pl clock minutes inflected as an ordinal ("osiemnastej pięćdziesiątej piątej"; the minutes
                      are a cardinal: "osiemnastej pięćdziesiąt pięć")
  pl_gen_case         pl nominative cardinal right after a word that takes the genitive ("zabrakło dwa tysiące")
  pl_dwie_scale       pl "dwie tysiące / miliony / miliardy" (the scale nouns are masculine: "dwa tysiące")

`span_errors` checks one span of a row; `row_errors` all of them. A row is a synthetic label row: "text" (the spoken
label), "spans" (each with "text" = written piece, "kind", "reading"), the written sentence in "caption" (ml_tts
rows) or "text_written" (LLM-written rows). The checks never look at audio: a row they flag speaks the wrong form and
is taken out, not relabelled.
"""

import re

MONTHS = {
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre",
    "pt": "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro",
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre",
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre",
}
MONTH_SET = {lang: set(v.split()) for lang, v in MONTHS.items()}
FIRST_MASC = {"es": "primero", "pt": "primeiro", "it": "primo", "fr": "premier"}

FR_ET_BAD = re.compile(r"(?<!vingt )(?<!trente )(?<!quarante )(?<!cinquante )(?<!soixante )\bet "
                       r"(?!quart\b|un quart\b|demie?\b)")
# a masculine first word is the hour only when it is not the minutes / quarter of "<minutes> para as <hour>" ("um
# quarto para as três", "dois minutos para as três", "un cuarto para las tres")
PT_HOUR_MASC = re.compile(r"^(?:vinte e )?(?:um|dois)\b(?! quarto\b| minutos?\b)")
ES_HOUR_MASC = re.compile(r"^(?:veintiuno|veintiún|uno|un)\b(?! cuarto\b| minutos?\b)")
# it "alle ore sedici e trenta": "ore" after an article / preposition is the clock noun, not a duration
IT_ORE_ARTICLE = re.compile(r"^(?:\S+ )?(?:le|alle|dalle|delle|nelle|sulle|verso le) ore\b")
# generator class names that mean the same census class ("years": ml_tts_multi's multi-year sentences)
CLS_ALIAS = {"years": "year", "dates": "date"}


def _cls(row: dict) -> str:
    c = row.get("cls") or ""
    return CLS_ALIAS.get(c, c)


CENTS_BAD = {"es": (re.compile(r"\b(?:dólar|dólares|peso|pesos)\b"), re.compile(r"\bcéntimos?\b")),
             "pt": (re.compile(r"\b(?:real|reais|dólar|dólares)\b"), re.compile(r"\bcêntimos?\b"))}
UNIT_WORDS = {"de": re.compile(r"\b(?:Euro|Dollar|Pfund|Cent|Franken)\b", re.I),
              "nl": re.compile(r"\b(?:euro|dollar|pond|cent|frank)\b", re.I)}
PRICE_JOIN = {"de": "und", "nl": "en"}
HALF_NOUN = {"medio", "media", "meio", "meia", "mezzo", "mezza", "demi", "demie", "half", "halb", "halbe"}
OF_WORDS = {"de", "di", "d'", "do", "da", "del"}
IT_TRE = re.compile(r"^\w+(?:ta|ti|la|le|to)tre$")
SLAV_COST = {"uk": {"коштує", "коштують", "коштувала", "коштував"}, "ru": {"стоит", "стоят", "стоила", "стоил"}}

# written currency (regex) -> (its own words, words of the OTHER currency its reading must not have)
CUR_WORDS = {"ro": {r"€|\beur": (r"\beuro\b", r"\b(?:lei|leu|bani|ban)\b"),
                    r"\blei\b|\bron\b": (r"\b(?:lei|leu)\b", r"\b(?:euro|cenți|cent)\b")},
             "pl": {r"€|\beur": (r"\beur", r"złot|grosz"), r"zł": (r"złot", r"\beur|\bcent")}}
RO_CUR_NOUNS = {"euro", "lei", "leu", "bani", "ban", "cenți", "cent", "dolari", "dolar", "ron"}
RO_FEM_END = {"două", "una"}
PL_NOM = set("""jeden jedna jedno dwa dwie trzy cztery pięć sześć siedem osiem dziewięć dziesięć jedenaście dwanaście
trzynaście czternaście piętnaście szesnaście siedemnaście osiemnaście dziewiętnaście dwadzieścia trzydzieści czterdzieści
pięćdziesiąt sześćdziesiąt siedemdziesiąt osiemdziesiąt dziewięćdziesiąt sto dwieście trzysta czterysta pięćset sześćset
siedemset osiemset dziewięćset tysiąc""".split())
# words after which a Polish amount takes the genitive (prepositions and verbs of lacking / needing / reaching)
PL_GEN_GOV = {"około", "od", "do", "rzędu", "bez", "dla", "powyżej", "poniżej", "niespełna", "brak", "brakuje",
              "brakowało", "zabrakło", "zabraknie", "potrzeba", "wymaga", "wymagało", "sięga", "sięgają", "sięgała",
              "sięgnęła", "sięgnął", "sięgnęły"}
PL_DAT_PRON = {"nam", "mi", "mu", "im", "jej", "wam", "ci", "nim"}
PL_ORD_END = re.compile(r"(?:ego|ej|ym|ą|emu)$")


def _tok(t: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", t.lower())


def _written(row: dict) -> str:
    return row.get("caption") or row.get("text_written") or ""


def _around(text: str, piece: str) -> tuple:
    """(before, after) of the first occurrence of `piece` in `text` ("", "") if absent."""
    if not piece:
        return "", ""
    i = text.lower().find(piece.lower())
    if i < 0:
        return "", ""
    return text[:i], text[i + len(piece):]


def _hour(sp_text: str):
    m = re.match(r"\s*(\d{1,2})(?:\s*[:.h]\s*(\d{2}))?", sp_text)
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else None) if m else (None, None)


def span_errors(lang: str, sp: dict, row: dict) -> list:
    """Error codes of one span (see the module docstring); [] = no known slip."""
    rd = (sp.get("reading") or "").strip()
    st = sp.get("text") or ""
    kind = sp.get("kind")
    cls = _cls(row)
    if not rd:
        return []
    rl = rd.lower()
    toks = [_tok(t) for t in rd.split()]
    toks = [t for t in toks if t]
    w_before, w_after = _around(_written(row), st)
    s_before, s_after = _around(row.get("text", ""), rd)
    out = []
    clock = kind == "clock" or cls in ("clock12", "clock24", "clock")

    if lang in FIRST_MASC and kind == "ord" and re.fullmatch(r"\s*1\s*(?:º|ª|o|er|re|ère)\W*", st):
        nxt = [_tok(t) for t in w_after.split()[:2]]
        if cls == "date_txt" or (nxt and (nxt[0] in MONTH_SET[lang] or (nxt[0] in ("de", "di") and len(nxt) > 1
                                                                          and nxt[1] in MONTH_SET[lang]))):
            if lang == "es" and toks == ["primer"]:
                out.append("date_ord_apocope")
            elif toks != [FIRST_MASC[lang]] and toks != ["uno"] and toks != ["um"]:
                out.append("date_ord_gender")

    if clock and lang == "pt" and PT_HOUR_MASC.search(rl):
        out.append("pt_clock_hour_masc")
    if clock and lang == "es" and ES_HOUR_MASC.search(rl):
        out.append("es_clock_hour_masc")
    if clock and lang == "fr":
        if FR_ET_BAD.search(rl):
            out.append("fr_clock_et")
        if re.search(r"\b(?:vingt et )?un heures?\b", rl) or re.match(r"(?:vingt et )?un\b", rl):
            out.append("fr_clock_hour_un")
        if re.match(r"\s*12\s*[:h.]\s*\d{2}\s*a\.?\s?m", st, re.I) and rl.startswith("douze"):
            out.append("fr_12am_douze")
    if clock and lang == "it":
        if re.match(r"\S+(?: \S+)? ore\b", rl) and not IT_ORE_ARTICLE.match(rl):
            out.append("it_clock_ore")  # "sedici ore e un quarto" is a duration
        if re.match(r"\s*0?0\s*[:.]", st) and rl.startswith("dodici"):
            out.append("clock_midnight_noon")  # 0:55 read as 12:55
    if clock and lang in ("fr", "pt", "es"):
        for i, t in enumerate(toks):
            if t in ("heure", "hora"):
                prev = toks[i - 1] if i else ""
                prev2 = toks[i - 2] if i > 1 else ""
                if prev not in ("un", "une", "um", "uma", "zéro", "zero"):
                    out.append("hour_noun_number")
                elif prev in ("un", "une", "um", "uma") and prev2 in ("et", "e", "y"):
                    out.append("hour_noun_number")  # "vingt et une heure": 21 takes the plural
            elif t in ("heures", "horas") and i:
                prev, prev2 = toks[i - 1], toks[i - 2] if i > 1 else ""
                if prev in ("un", "une", "um", "uma") and prev2 not in ("et", "e", "y"):
                    out.append("hour_noun_number")
                elif lang == "fr" and prev == "zéro":
                    out.append("hour_noun_number")  # "zéro heures": French 0 takes the singular ("zéro heure")

    if kind == "int" and cls in ("count", "large") and re.search(r"\b(?:zero|cero|zéro|null|nul|nulla)\b", rl) \
            and not re.match(r"\s*0", st):
        out.append("count_digit_chunks")  # "51000 funcionários" read "cinco dez zero zero"
    if lang in CENTS_BAD and (kind == "cur" or cls.startswith("price")):
        u, c = CENTS_BAD[lang]
        if u.search(rl) and c.search(rl):
            out.append("cents_word")
    if lang in PRICE_JOIN and (kind == "cur" or cls.startswith("price")) and re.search(r"\d[.,]\d{2}(?!\d)", st):
        if not UNIT_WORDS[lang].search(rd) and PRICE_JOIN[lang] in toks:
            out.append("price_two_numbers")

    if kind == "frac" and st.strip() == "½":
        nxt = [_tok(t) for t in w_after.split()[:1]]
        if lang in ("es", "pt", "it", "fr"):
            if (nxt and nxt[0] in OF_WORDS) or re.match(r"\s*d'", w_after):
                out.append("frac_half_unit")  # "½ de litro": no reading of it is natural ("medio litro" has no "de")
        elif nxt and nxt[0] and set(toks) & HALF_NOUN and not re.search(r"\bhalve[rn]?\b", rl):
            out.append("frac_half_unit")  # "½ liter" read "één half" / "half": the unit takes "halve" / "halber"
    if lang == "nl" and re.search(r"\ben één half\b", rl):
        out.append("nl_een_half")
    if lang == "it" and ("ventiseesimo" in rl or any(IT_TRE.match(t) for t in toks)):
        out.append("it_spelling")

    if lang in SLAV_COST and "одна" in toks:
        prev = [_tok(t) for t in s_before.split()[-2:]]
        if set(prev) & SLAV_COST[lang]:
            out.append("slav_cost_acc")

    if lang in CUR_WORDS and kind == "cur" and _cur_mismatch(lang, st, rl):
        out.append("cur_word_mismatch")
    if lang == "ro":
        out += _ro_errors(rd, s_before, s_after)
    if lang == "pl":
        out += _pl_errors(sp, row, rd, toks, s_before, clock)
    return out


def _cur_mismatch(lang: str, written: str, rl: str) -> bool:
    """The reading names another currency than the written symbol, or the other currency's cents ("3,50 €" read
    "trei euro și cincizeci de bani", "74,11 €" read "... złote ... eurocentów")."""
    w = written.lower()
    for sym, (mine, other) in CUR_WORDS[lang].items():
        if re.search(sym, w):
            return bool(re.search(other, rl))
    return False


def _ro_errors(rd: str, before: str, after: str) -> list:
    import ml_multi as MM
    import ro_gender

    out = []
    toks = rd.split()
    keys = [_tok(t) for t in toks]
    tail = [_tok(t) for t in after.split()[:2]]
    nums = MM.parse_numbers(rd, "ro")
    for n in nums:
        j = n.end + 1  # the token after the number
        has_de = j < len(keys) and keys[j] == "de"
        noun = keys[j + 1] if has_de and j + 1 < len(keys) else keys[j] if j < len(keys) else ""
        if not noun and j >= len(keys):  # the noun is outside the span ("12,50" + " lei")
            has_de = bool(tail) and tail[0] == "de"
            noun = (tail[1] if has_de and len(tail) > 1 else tail[0]) if tail else ""
        if noun not in RO_CUR_NOUNS:
            continue
        piece = keys[n.start:n.end + 1]
        if "virgulă" in piece or n.value != int(n.value):
            continue
        v = int(n.value)
        want_de = v >= 20 and (v % 100 == 0 or v % 100 >= 20)
        if v > 0 and want_de != has_de:
            out.append("ro_de_linker")
        if piece and piece[-1] in RO_FEM_END:
            out.append("ro_noun_gender")
    if not ro_gender.grammatical(rd, before, after):
        out.append("ro_gender")
    for i, k in enumerate(keys):
        if k == "mie" and (i == 0 and _tok((before.split() or [""])[-1]) != "o" or i > 0 and keys[i - 1] != "o"):
            out.append("ro_o_mie")
            break
    return out


def _pl_errors(sp, row, rd, toks, before, clock) -> list:
    out = []
    st = sp.get("text") or ""
    if clock:
        h, m = _hour(st)
        if h is not None and m:
            nh = 1 if toks and toks[0] == "zero" else 2 if 21 <= h <= 23 else 1
            if any(PL_ORD_END.search(t) for t in toks[nh:]):
                out.append("pl_clock_minute")
    for i, t in enumerate(toks[:-1]):
        if t == "dwie" and re.match(r"(?:tysiące|tysięcy|miliony|milionów|miliardy|miliardów)$", toks[i + 1]):
            out.append("pl_dwie_scale")
            break
    kind, cls = sp.get("kind"), _cls(row)
    if kind in ("int", "cur", "range") and cls not in ("year", "date") and toks and toks[0] in PL_NOM \
            and not PL_ORD_END.search(toks[-1]):
        prev = [_tok(t) for t in before.split()[-2:]]
        if prev and (prev[-1] in PL_GEN_GOV or (len(prev) == 2 and prev[-1] in PL_DAT_PRON and prev[0] in PL_GEN_GOV)):
            out.append("pl_gen_case")  # "zabrakło (nam) dwa tysiące" -> "dwóch tysięcy"
    return out


def row_errors(lang: str, row: dict) -> list:
    """Sorted unique error codes of every span of a row."""
    out = set()
    for sp in row.get("spans") or []:
        out.update(span_errors(lang, sp, row))
    return sorted(out)
