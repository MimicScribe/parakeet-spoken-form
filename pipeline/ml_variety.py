"""Variety and adversarial selection for the spoken-form data (2026-10-16). Text only: finds the captions that carry a
shape the data is short of, so a selection can ask for them by class. Never labels anything (labels come from the
usual chains: ml_p3.label2 for rows with digits, the plain-row check for number-free look-alikes).

Classes (each a regex family per language; a caption may hit several):
    biz_*   business shapes: quarter / half-year / fiscal year, basis points, percentage points, scale abbreviations
            ("2,5 Mrd. €"), "Art. 12" / "Artikel 12", "Nr. 5", "§ 3", km/h, m², °C, multipliers ("x 3"), percent ranges
    look_*  look-alikes: a number-sounding word that must stay a word (homophones: it "sei", pt "sem"/"cem", fr
            "sans"/"cent", en "for"/"four"), number words inside idioms ("zu zweit", "à deux", "uno de ellos"), and
            names / codes with digits ("Louis XIV", "COVID-19", "Windows 11", "G7", "MP3", "A4")
    messy_* hesitations and self-corrections next to numbers ("zwei … äh zweitausend", "twenty, no, thirty"),
            number lists ("3, 4 und 5"), alternatives ("2 oder 3"), repeats
    cs_*    code-switching: English words / number words / brands inside non-English captions

    python3 ml_variety.py census --lang de file.jsonl ...      # counts + examples per class
    python3 ml_variety.py select --lang de --out ids.json file.jsonl ...
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

LANGS = ("en", "de", "es", "fr", "it", "nl", "pt", "ru", "uk", "pl", "ro")
I = re.I
N = r"\d+(?:[.,]\d+)?"  # a written number


def _rx(p, flags=I):
    return re.compile(p, flags)


# ------------------------------------------------------------------------------------------------- business shapes
_QUARTER = {
    "en": r"\bQ[1-4]\b|\b(?:first|second|third|fourth|1st|2nd|3rd|4th) quarter\b",
    "de": r"\bQ[1-4]\b|\b[1-4]\.\s?Quartal\b",
    "fr": r"\b[TQ][1-4]\b|\b[1-4](?:er|e|ème)\s?trimestre\b",
    "es": r"\b[TQ][1-4]\b|\b[1-4]T\b|\b[1-4](?:º|°|er|o)?\s?trimestre\b",
    "it": r"\b[TQ][1-4]\b|\b[1-4](?:º|°)?\s?trimestre\b",
    "nl": r"\bQ[1-4]\b|\b[1-4](?:e|de|ste)\s?kwartaal\b",
    "pt": r"\b[TQ][1-4]\b|\b[1-4]T\b|\b[1-4](?:º|°)?\s?trimestre\b",
    "ru": r"\b[1-4]\s?(?:-?(?:й|м|го))?\s?квартал\w*|\bQ[1-4]\b", "uk": r"\b[1-4]\s?(?:-?(?:й|му|го))?\s?квартал\w*|\bQ[1-4]\b",
    "pl": r"\b[1-4]\.?\s?kwarta\w*|\bQ[1-4]\b", "ro": r"\btrimestrul\s+[1-4]\b|\bT[1-4]\b|\bQ[1-4]\b",
}
_HALF = {
    "en": r"\bH[12]\b", "de": r"\bH[12]\b|\b[12]\.\s?Halbjahr\b", "fr": r"\b[SH][12]\b|\b[12](?:er|e|nd)\s?semestre\b",
    "es": r"\b[SH][12]\b|\b[12](?:º|°|er)?\s?semestre\b", "it": r"\b[SH][12]\b|\b[12](?:º|°)?\s?semestre\b",
    "nl": r"\bH[12]\b|\b[12](?:e|ste)\s?(?:halfjaar|semester)\b", "pt": r"\b[SH][12]\b|\b[12](?:º|°)?\s?semestre\b",
    "ru": r"\b[12]\s?(?:-?(?:е|м|го))?\s?полугоди\w*", "uk": r"\b[12]\s?(?:-?(?:е|му|го))?\s?півріч\w*",
    "pl": r"\b[12]\.?\s?półrocz\w*|\bH[12]\b", "ro": r"\bsemestrul\s+[12]\b|\bS[12]\b",
}
_FY = {
    "en": r"\bFY\s?'?\d{2,4}\b|\bfiscal (?:year )?'?\d{2,4}\b",
    "de": r"\bFY\s?\d{2,4}\b|\bGJ\s?\d{2,4}\b|\bGeschäftsjahr(?:es)?\s+\d{2,4}(?:/\d{2,4})?",
    "fr": r"\bFY\s?\d{2,4}\b|\bexercice\s+\d{4}", "es": r"\bFY\s?\d{2,4}\b|\bejercicio\s+(?:fiscal\s+)?\d{4}",
    "it": r"\bFY\s?\d{2,4}\b|\besercizio\s+\d{4}", "nl": r"\bFY\s?\d{2,4}\b|\bboekjaar\s+\d{4}",
    "pt": r"\bFY\s?\d{2,4}\b|\bexercício\s+(?:fiscal\s+)?(?:de\s+)?\d{4}", "ru": r"\bFY\s?\d{2,4}\b|финансов\w+ год\w*",
    "uk": r"\bFY\s?\d{2,4}\b|фінансов\w+ р[іо]к\w*", "pl": r"\bFY\s?\d{2,4}\b|rok\w* obrotow\w*", "ro": r"\bFY\s?\d{2,4}\b|anul fiscal",
}
_BP = {
    "en": N + r"\s?(?:bps?|basis points?)\b", "de": N + r"\s?(?:Basispunkte?n?|bp)\b",
    "fr": N + r"\s?(?:points? de base|pb)\b", "es": N + r"\s?(?:puntos? básicos|pb)\b",
    "it": N + r"\s?(?:punti base|pb|bp)\b", "nl": N + r"\s?(?:basispunten|bp)\b", "pt": N + r"\s?(?:pontos?[- ]base|pb)\b",
    "ru": N + r"\s?(?:базисн\w+ пункт\w*|б\.\s?п\.)", "uk": N + r"\s?(?:базисн\w+ пункт\w*|б\.\s?п\.)",
    "pl": N + r"\s?(?:punkt\w* bazow\w*|pb)\b", "ro": N + r"\s?(?:puncte de bază|pb)\b",
}
_PCTPT = {
    "en": N + r"\s?(?:percentage points?|pp)\b", "de": N + r"\s?(?:Prozentpunkte?n?|pp)\b",
    "fr": N + r"\s?points? de pourcentage", "es": N + r"\s?puntos? porcentuales?", "it": N + r"\s?punti percentuali",
    "nl": N + r"\s?procentpunt(?:en)?", "pt": N + r"\s?pontos? percentuais", "ru": N + r"\s?процентн\w+ пункт\w*",
    "uk": N + r"\s?відсотков\w+ пункт\w*", "pl": N + r"\s?punkt\w* procentow\w*", "ro": N + r"\s?puncte procentuale",
}
_SCALE = r"\d\s?(?:Mrd|Mio|Mld|mld|mln|Mln|bn|mrd|Md|mds|M€|k€)\.?(?=[\s€$£,;:)]|$)|[€$£]\s?" + N + r"\s?(?:M|B|bn|k)\b|\d\s?[kKMB]\s?[€$£]"
_ART = {
    "en": r"\b(?:Art\.|Article)\s?\d+", "de": r"\bArt\.\s?\d+|\bArtikel[ns]?\s+\d+", "fr": r"\b[Aa]rt\.\s?\d+|\barticles?\s+\d+",
    "es": r"\b[Aa]rt\.\s?\d+|\bartículos?\s+\d+", "it": r"\b[Aa]rt\.\s?\d+|\barticol[oi]\s+\d+",
    "nl": r"\b[Aa]rt\.\s?\d+|\bartikel(?:en)?\s+\d+", "pt": r"\b[Aa]rt\.\s?\d+|\bartigos?\s+\d+",
    "ru": r"\bст\.\s?\d+|\bстать\w+\s+\d+", "uk": r"\bст\.\s?\d+|\bстатт\w+\s+\d+", "pl": r"\bart\.\s?\d+|\bartykuł\w*\s+\d+",
    "ro": r"\bart\.\s?\d+|\barticolul(?:ui)?\s+\d+",
}
_NR = r"(?<!\w)(?:[Nn]r|[Nn]ro|N\.?º|n\.?º|[Nn]úm|No|n°|N°|№)\.?\s?\d+"
_PARA = r"§\s?\d+"
_SPEED = r"\d\s?(?:km/h|kmh|mph|km/u|km/ч|км/ч|км/год)(?!\w)"
_AREA = r"\d\s?(?:m²|m2|km²|km2|qm|м²|м2|кв\.\s?м)(?!\w)|\d\s?ha\b(?!\s+(?:\w+(?:do|to|ido|ato|uto)|sido|dicho|hecho|estado))"
_TEMP = r"-?\d+(?:[.,]\d+)?\s?°\s?[CF]?(?!\w)|\d\s?(?:Grad Celsius|degrés Celsius|grados centígrados|gradi centigradi)"
_MULT = r"\b\d+\s?[x×]\s?\d+\b|(?<![\w.])[x×]\s?\d+\b|\b\d+\s?[x×](?![\w\d])"
_PCT_RANGE = {
    "en": N + r"\s?(?:%\s?)?(?:-|–|to)\s?" + N + r"\s?(?:%|percent)",
    "de": N + r"\s?(?:%\s?)?(?:-|–|bis)\s?" + N + r"\s?(?:%|Prozent)", "fr": N + r"\s?(?:%\s?)?(?:-|–|à)\s?" + N + r"\s?(?:%|pour ?cent)",
    "es": N + r"\s?(?:%\s?)?(?:-|–|a|y)\s?" + N + r"\s?(?:%|por ciento)", "it": N + r"\s?(?:%\s?)?(?:-|–|al|a|e)\s?" + N + r"\s?(?:%|per cento)",
    "nl": N + r"\s?(?:%\s?)?(?:-|–|tot)\s?" + N + r"\s?(?:%|procent)", "pt": N + r"\s?(?:%\s?)?(?:-|–|a|e)\s?" + N + r"\s?(?:%|por cento)",
    "ru": N + r"\s?(?:%\s?)?(?:-|–|до)\s?" + N + r"\s?(?:%|процент\w*)", "uk": N + r"\s?(?:%\s?)?(?:-|–|до)\s?" + N + r"\s?(?:%|відс\w*)",
    "pl": N + r"\s?(?:%\s?)?(?:-|–|do)\s?" + N + r"\s?(?:%|proc\w*)", "ro": N + r"\s?(?:%\s?)?(?:-|–|la)\s?" + N + r"\s?(?:%|la sută)",
}


def _biz(lang):
    return {"biz_quarter": _rx(_QUARTER[lang], 0), "biz_half": _rx(_HALF[lang], 0), "biz_fy": _rx(_FY[lang]),
            "biz_bp": _rx(_BP[lang], 0), "biz_pctpt": _rx(_PCTPT[lang]), "biz_scale": _rx(_SCALE, 0),
            "biz_art": _rx(_ART[lang], 0), "biz_nr": _rx(_NR, 0), "biz_para": _rx(_PARA, 0), "biz_speed": _rx(_SPEED),
            "biz_area": _rx(_AREA, 0), "biz_temp": _rx(_TEMP), "biz_mult": _rx(_MULT, 0), "biz_pctrange": _rx(_PCT_RANGE[lang])}


# ------------------------------------------------------------------------------------------------- look-alikes
# homophones / near-homophones of a number word that are NOT numbers (or number words used as plain words)
_HOMO = {
    "en": r"\b(?:for|to|too|won|ate)\b(?=\s+(?:\d|one|two|three|four|five|six|seven|eight|nine|ten|a)\b)"
          r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:for|to|won|ate)\b",
    "de": r"\b(?:achten|achtet|achtete|beachten|beachtet|außer Acht|in Acht|Acht geben|Achtung|achtsam\w*)\b",
    "nl": r"\b(?:in acht (?:nemen|genomen|neemt)|acht slaan|achten|achting|eens|eenmaal|eenheid)\b",
    "fr": r"\b(?:sans|sang|s'en|sent|vin|vins|vain|vint|scie|cette|neuf|neuve|neufs|dis|dit)\b",
    "es": r"\b(?:sien|sienes|tras|atrás|siento|sientes|nuevo|nueva)\b",
    "it": r"\b(?:(?:tu|ci|non|che|lo|ne|se|dove|come|chi|cosa|ma|e|perché) sei|sei (?:stat[oaie]|un|una|il|la|qui|lì|sicur[oa]|tu|così|bravo|brava|molto))\b",
    "pt": r"\b(?:sem|sei|trás|atrás|cento? e poucos)\b",
    "ro": r"\b(?:nouă|noua)\b(?!\s+(?:milioane|mii|sute|ani|zile|ore|euro|lei|la sută|procente))",
    "pl": r"\b(?:raz|razem|sto lat|stoi|stoją)\b",
    "ru": r"\b(?:раз|разом|стоит|стоят|сорока|семья|семьи|опять)\b",
    "uk": r"\b(?:раз|разом|стоїть|стоять|сорока|сім'я|сім’я|знову)\b",
}
# number words inside idioms / fixed phrases / pronouns
_IDIOM = {
    "en": r"\b(?:one of (?:them|us|the|those|these)|someone|no one|anyone|each one|the one|once|at once|first of all|second-hand|a hundred percent|a million times|on all fours|forty winks|nine to five|a couple of)\b",
    "de": r"\b(?:zu (?:zweit|dritt|viert)|Tausendsassa|Elfmeter\w*|Einheit|einmal|einander|einig\w*|vereint|Zweifel\w*|zweifel\w*|dreist|Viertel\w*|tausendmal|hundertprozentig|erstens|zweitens|drittens|einerseits|beide[nrs]?)\b",
    "nl": r"\b(?:met z'n (?:tweeën|drieën|vieren)|honderduit|duizendpoot|eerst|tweede kamer|Tweede Kamer|honderd procent|allebei|beide[n]?|een paar|enkele)\b",
    "fr": r"\b(?:à deux|tous les deux|toutes les deux|l'un des|l'une des|quelqu'un|chacun|chacune|aucun|aucune|mille fois|cent fois|deux-roues|mille-pattes|quatre-quatre|vingt-quatre heures sur vingt-quatre|une fois|à la fois|d'une part|premièrement|deuxièmement|dizaine|centaine|quinzaine|huitaine)\b",
    "es": r"\b(?:uno de (?:ellos|ellas|los|las)|una de (?:ellas|las)|cada uno|cada una|ninguno|ninguna|alguno|alguna|uno mismo|los dos|las dos|ambos|mil veces|cien por cien|a la vez|una vez|primero|en primer lugar|decena|centenar|docena)\b",
    "it": r"\b(?:uno di (?:loro|noi|voi|questi)|una di (?:loro|queste)|ognuno|ognuna|ciascuno|nessuno|qualcuno|qualcuna|mille grazie|cento per cento|una volta|tutti e due|tutte e due|entrambi|decina|centinaio|migliaio|dozzina|innanzitutto|in primo luogo)\b",
    "pt": r"\b(?:um dos|uma das|cada um|cada uma|nenhum|nenhuma|algum|alguma|os dois|as duas|ambos|às duas por três|mil vezes|cem por cento|uma vez|primeiro|em primeiro lugar|dezena|centena|dúzia)\b",
    "ro": r"\b(?:unul dintre|una dintre|fiecare|niciunul|amândoi|amândouă|o dată|odată|în primul rând|o sută la sută)\b",
    "pl": r"\b(?:jeden z|jedna z|jedno z|każdy|żaden|obaj|oboje|obie|sto procent|po pierwsze|raz na zawsze|dwójka|trójka|tysiąc razy)\b",
    "ru": r"\b(?:один из|одна из|одно из|каждый|ни один|оба|обе|сто процентов|во-первых|во-вторых|двойка|тройка|пятёрка|тысячу раз)\b",
    "uk": r"\b(?:один з|одна з|одне з|кожен|кожний|жоден|обидва|обидві|сто відсотків|по-перше|по-друге|двійка|трійка|п'ятірка|тисячу разів)\b",
}
# names / codes / products with a digit or a roman numeral (labelled by ml_cands alnum / roman / version kinds)
_NAMED = (r"\b(?:COVID|Covid|covid|SARS-CoV)-?\s?\d+|\b(?:Windows|iOS|Android|macOS|PlayStation|PS|Xbox|iPhone|Galaxy|Pixel|"
          r"Formula|Formel|Fórmula|Boeing|Airbus|Golf|Euro|Web|Industrie|Industria|Industry|Office|Office)\s?-?[A-Z]?\d+(?:\.\d)?\b"
          r"|\b[GE]\s?(?:7|8|20|77)\b|\b[2-6]G\b|\bMP[34]\b|\bA[0-6]\b|\b(?:F|Su|MiG|A|B|C|T)-\d{2,3}\b|\b\d{1,2}D\b|\b24/7\b"
          r"|\b[A-Z][a-zà-ÿ]+\s+(?:I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI)\b")

# ------------------------------------------------------------------------------------------------- messy number speech
_FILL = {
    "en": r"uh|um|uhm|er|erm|hmm", "de": r"äh|ähm|öh|ehm|hm|hmm", "fr": r"euh|heu|hum|bah|ben",
    "es": r"eh|ehm|mmm|em", "it": r"ehm|eh|uhm|mmm|cioè", "nl": r"eh|ehm|uh|uhm|hè", "pt": r"é+h|ahn|hum|tipo|né",
    "ru": r"э+|эм|ну|мм", "uk": r"е+|ем|ну|мм", "pl": r"y+|yyy|eee|eh|no", "ro": r"ă+|ăă|ehm|deci",
}
_CORR = {
    "en": r"no|sorry|I mean|or rather|actually|correction", "de": r"nein|nee|Entschuldigung|ich meine|äh nein|pardon|genauer gesagt|Quatsch",
    "fr": r"non|pardon|je veux dire|ou plutôt|excusez-moi|euh non", "es": r"no|perdón|digo|o sea|mejor dicho|quiero decir|perdona",
    "it": r"no|anzi|scusate|scusa|cioè|volevo dire|pardon", "nl": r"nee|sorry|ik bedoel|pardon|of eigenlijk|eigenlijk",
    "pt": r"não|perdão|desculpe|digo|ou melhor|quer dizer|aliás", "ru": r"нет|точнее|вернее|простите|извините|то есть",
    "uk": r"ні|точніше|вірніше|вибачте|тобто", "pl": r"nie|przepraszam|to znaczy|właściwie|a raczej|czyli", "ro": r"nu|pardon|adică|mai exact|scuze",
}
_AND = {"en": "and|or", "de": "und|oder|bis", "fr": "et|ou|à", "es": "y|o|u|a", "it": "e|o|a", "nl": "en|of|tot", "pt": "e|ou|a",
        "ru": "и|или|до", "uk": "і|й|та|або|до", "pl": "i|lub|albo|do", "ro": "și|sau|la"}
_NUMW = {  # small number words (to catch "zwei … äh zweitausend" where captions write words)
    "en": "one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred|thousand",
    "de": "eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|zwanzig|dreißig|hundert|tausend",
    "fr": "deux|trois|quatre|cinq|six|sept|huit|neuf|dix|vingt|trente|cent|mille",
    "es": "uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|veinte|treinta|cien|mil",
    "it": "uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|venti|trenta|cento|mille",
    "nl": "twee|drie|vier|vijf|zes|zeven|acht|negen|tien|twintig|dertig|honderd|duizend",
    "pt": "dois|três|quatro|cinco|seis|sete|oito|nove|dez|vinte|trinta|cem|mil",
    "ru": "один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|двадцать|сто|тысяч\\w*",
    "uk": "один|два|три|чотири|п'ять|шість|сім|вісім|дев'ять|десять|двадцять|сто|тисяч\\w*",
    "pl": "jeden|dwa|trzy|cztery|pięć|sześć|siedem|osiem|dziewięć|dziesięć|dwadzieścia|sto|tysiąc\\w*",
    "ro": "unu|doi|două|trei|patru|cinci|șase|șapte|opt|nouă|zece|douăzeci|sută|mie|mii",
}
_P = r"[\s,.…;:–—-]*"  # pause punctuation between two parts


def _messy(lang):
    num = r"(?:\d[\d.,]*|\b(?:" + _NUMW[lang] + r")\b)"
    return {
        "messy_filler": _rx(num + _P + r"\b(?:" + _FILL[lang] + r")\b" + _P + num + "|" + r"\b(?:" + _FILL[lang] + r")\b" + _P + r"\d"),
        "messy_corr": _rx(num + r"(?:\s?(?:%|€|\w{1,12}))?" + _P + r"\b(?:" + _CORR[lang] + r")\b" + _P + num),
        "messy_repeat": _rx(r"(?<![\d.,])(\d+)(?:\s*(?:\.\.\.|…)\s*|,\s+|\s+|\s[-–]\s)\1(?![\d.,])"
                            r"|(?<![\d.,])(\d)(?:\s*(?:\.\.\.|…)\s*|,\s+|\s+|\s[-–]\s)\2\d{2,}(?![\d.,])"),
        "messy_list": _rx(r"(?<![\d.,])\d+(?:\s?,\s?\d+){1,}\s?,?\s(?:" + _AND[lang] + r")\s\d+|(?<![\d.,])\d+(?:\s?,\s\d+){3,}"),
        "messy_alt": _rx(r"(?<![\d.,])\d+(?:[.,]\d+)?\s(?:" + _AND[lang].split("|", 2)[1] + r")\s\d+(?:[.,]\d+)?(?![\d.,])"),
    }


# ------------------------------------------------------------------------------------------------- code-switching
# English function words that never mean anything else in the language ("so" is German / Italian, "we" Dutch)
_EN_FUNC = r"the|and|with|you|your|this|that|what|it's|i'm|don't|please|thank you|thanks|is|are|will|can|of the|in the"
_EN_FUNC_EXCL = {"nl": {"we", "is"}, "de": {"will"}, "ro": {"are"}, "pt": {"is"}}
# English loanwords / business words used inside other languages
_EN_LOAN = (r"business|meeting|deal|update|feature|release|sale|free|shop|online|download|level|game|team|live|show|"
            r"best|top|price|cashback|black friday|weekend|cool|nice|okay|yeah|sorry|guys|performance|budget|marketing|"
            r"start-?up|follower|like|subscribe|link|tutorial|review|unboxing|gaming|stream|streamer")
_EN_NUM = r"one|two|three|four|five|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|billion|first|third|percent"
_EN_NUM_EXCL = {"fr": {"six", "second", "million"}, "nl": {"ten"}, "pl": {"ten"}}


def _cs(lang):
    if lang == "en":  # the reverse direction: non-English number / currency words in English captions
        return {"cs_foreign_num": _rx(r"\b(?:euros?|reais|pesos|zł|złoty|rubles?|hryvnia|eins|zwei|drei|deux|trois|uno|dos|tres|cuatro|"
                                       r"cinco|seis|siete|ocho|nueve|diez|cento|mille|mil)\b(?=.*\d)|\d.*\b(?:euros?|reais|pesos|"
                                       r"uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|zwei|drei|deux|trois)\b")}
    func = "|".join(w for w in _EN_FUNC.split("|") if w not in _EN_FUNC_EXCL.get(lang, set()))
    func = func + "|" + "|".join(w[:1].upper() + w[1:] for w in func.split("|"))
    return {"cs_en_sentence": _rx(r"(?:\b(?:" + func + r")\b.*){2,}", 0),
            "cs_en_loan": _rx(r"\b(?:" + _EN_LOAN + r")\b"),
            "cs_en_num": _rx(r"\b(?:" + _EN_NUM + r")\b"),
            "cs_brand": _rx(_NAMED, 0)}


_CACHE = {}


def patterns(lang: str) -> dict:
    if lang not in _CACHE:
        p = _biz(lang)
        p["look_homo"] = _rx(_HOMO[lang])
        p["look_idiom"] = _rx(_IDIOM[lang])
        p["look_named"] = _rx(_NAMED, 0)
        p.update(_messy(lang))
        p.update(_cs(lang))
        _CACHE[lang] = p
    return _CACHE[lang]


def classify(text: str, lang: str) -> dict:
    """{class: [matched text, ...]} for one caption."""
    out = {}
    for k, rx in patterns(lang).items():
        ms = [m.group(0) for m in rx.finditer(text)]
        if k == "cs_en_num":
            ms = [m for m in ms if m.lower() not in _EN_NUM_EXCL.get(lang, set())]
        if ms:
            out[k] = ms
    return out


def template(text: str) -> str:
    """The sentence pattern a template cap counts (digits masked, case and spacing folded)."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def group_of(row: dict) -> str:
    """The unit of the per-session / per-video cap: a parliament session or a YouTube video."""
    rid = str(row.get("id") or row.get("utt_id") or "")
    m = re.match(r"(\d{8}-\d{4}-PLENARY)", rid)
    if m:
        return m.group(1)
    m = re.match(r"[a-z]{2}\d{3}_\d{8}_(.{11})_", rid)  # YODAS: <lang><n>_<shard>_<video id>_...
    return m.group(1) if m else rid


def select(rows: list, lang: str, per_class: int, per_template: int = 3, per_group: int = 8, seed: int = 1,
           skip: set | None = None) -> tuple[list, dict]:
    """Rows chosen per class (at most per_class each), with the template and session/video caps across the whole
    selection. Rows hitting several classes count for each. Returns (rows, {class: n})."""
    rng = random.Random(f"{seed}|{lang}")
    by = defaultdict(list)
    for r in rows:
        if skip and (r.get("id") or r.get("utt_id")) in skip:
            continue
        for k in classify(r["text"], lang):
            by[k].append(r)
    tpl, grp, chosen, got = Counter(), Counter(), {}, Counter()
    for k in sorted(by, key=lambda c: len(by[c])):  # rarest class first
        pool = by[k][:]
        rng.shuffle(pool)
        for r in pool:
            if got[k] >= per_class:
                break
            rid = r.get("id") or r.get("utt_id")
            if rid in chosen:
                got[k] += 1
                continue
            t, gk = template(r["text"]), group_of(r)
            if tpl[t] >= per_template or grp[gk] >= per_group:
                continue
            tpl[t] += 1
            grp[gk] += 1
            chosen[rid] = dict(r, variety=sorted(classify(r["text"], lang)))
            got[k] += 1
    return list(chosen.values()), dict(got)


def _read(paths):
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "select"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = list(_read(a.files))
    if a.cmd == "census":
        c, ex = Counter(), defaultdict(list)
        for r in rows:
            for k, ms in classify(r["text"], a.lang).items():
                c[k] += 1
                if len(ex[k]) < a.examples:
                    ex[k].append(r["text"][:160])
        print(json.dumps({"lang": a.lang, "rows": len(rows), "classes": dict(sorted(c.items())), "examples": ex},
                         ensure_ascii=False, indent=1))
    else:
        sel, got = select(rows, a.lang, a.per_class)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({"lang": a.lang, "selected": len(sel), "per_class": got}, ensure_ascii=False))




# ------------------------------------------------------------------------------------------------- regional relabel
# Regional readings (ml_cands VARIETY: Belgian / Swiss "septante" / "huitante" / "nonante", European Portuguese
# "mil milhões" / "bilião", Austrian "Jänner", relative clock readings) are decided from the EXTERNAL transcript, never
# by the CTC: the xlsr CTC prefers the shorter "septante" even where the audio says "soixante-dix" (measured on a dry run).
# After label2: (1) a kept row's reading is swapped for a regional reading of the same value when the transcript has
# that reading word for word and not the kept one; (2) a row label2 left undecided is recovered when each undecided span
# has exactly one reading the transcript has word for word and it is a regional one (new information label2 lacked);
# the row guards (CER, caption gap, symbols, abbreviations) then run as in label2 (ml_p3_core.make_row).
# Portuguese spelling: "dezasseis / dezassete / dezanove" (European) where the transcript writes them, and always for
# European Parliament rows (src "vox": Portuguese MEPs and interpreters are European); the number grammar spells BR.
def _value_mismatch(caption: str, ext: str) -> bool:
    """A digit group the recognizer wrote that is no caption group (separators inside a number joined: 1.000 = 1000,
    8:00 = 800) nor its start ("8 Uhr" for "8:00 Uhr" is fine): it heard another value."""
    def groups(t):
        return re.findall(r"\d+", re.sub(r"(?<=\d)[.,:\s](?=\d)", "", t))
    cap = groups(caption)
    return any(not any(c.startswith(g) for c in cap) for g in groups(ext))


PT_EU_SPELL = {"dezesseis": "dezasseis", "dezessete": "dezassete", "dezenove": "dezanove"}


def _words(t: str) -> list[str]:
    import ml_decide as D

    return [w for w in (D.norm_word(x) for x in t.split()) if w]


def _has_run(reading: str, ext: str) -> bool:
    """reading's letters equal a run of whole transcript words (compound spacing / hyphens free: "huitante un" =
    "huitante-un"), spelling-sensitive (no dialect folding)."""
    k, ws = "".join(_words(reading)), _words(ext)
    for i in range(len(ws)):
        acc = ""
        for j in range(i, len(ws)):
            acc += ws[j]
            if acc == k:
                return True
            if len(acc) >= len(k) or not k.startswith(acc):
                break
    return False


def _pt_eu(reading: str) -> str:
    return " ".join(PT_EU_SPELL.get(w, w) for w in reading.split(" "))


def regional_relabel(lang: str, rows: list, diag: list, ext: dict, cands: dict) -> tuple[list, dict]:
    """rows: label2 kept rows; diag: label2 diag rows; ext: {id: transcript text}; cands: {id: candidate row}.
    Returns (rows to use instead of `rows`, with recovered rows appended; stats)."""
    import ml_cands as M
    import ml_decide as D
    import ml_p3_core as C

    D.set_script(lang)
    st = Counter()
    out = []
    for row in rows:
        cap, e = row.get("caption"), ext.get(row["id"])
        sp = M.spans_regional(cap, lang) if cap else []
        now = M.spans(cap, lang) if cap else []
        if not cap or len(sp) != len(row.get("spans", [])) or any(not s.cands for s in sp):
            out.append(row)
            st["kept_unaligned"] += 1
            continue
        picks, changed = [], []
        for i, (s, rs) in enumerate(zip(sp, row["spans"])):
            cs = [c.strip() for c in s.cands]
            low = [c.lower() for c in cs]
            if rs["reading"].strip().lower() not in low:  # render_cap capitalises a reading that opens a sentence
                picks = None
                break
            p = low.index(rs["reading"].strip().lower())
            n_base = len(now[i].cands)  # the readings after it are the regional ones
            if e and not _has_run(rs["reading"].strip(), e):
                alt = [j for j in range(n_base, len(cs)) if _has_run(cs[j], e)]
                if alt:
                    p = max(alt, key=lambda j: len(cs[j]))
                    changed.append(i)
            picks.append(p)
        if picks is None:
            out.append(row)
            st["kept_unaligned"] += 1
            continue
        offs = []
        label = C.render_cap(cap, sp, picks, offs)
        if lang == "pt" and (row.get("src") == "vox" or (e and any(w in _words(e) for w in PT_EU_SPELL.values()))):
            for i, (a, b) in enumerate(offs):
                pe = _pt_eu(label[a:b])
                br_heard = bool(e) and any(w in _words(e) for w in _words(label[a:b]) if w in PT_EU_SPELL)
                if pe != label[a:b] and ((row.get("src") == "vox" and not br_heard) or _has_run(pe, e or "")):
                    label = label[:a] + pe + label[b:]  # same length: the teen words differ in one letter
                    changed.append(i)
        if not changed:
            out.append(row)
            continue
        nr = dict(row, text=label, num_chars=offs, regional=sorted(set(changed)))
        nr["spans"] = [dict(rs, reading=label[a:b].strip(), by="ext_regional") if i in changed else rs
                       for i, (rs, (a, b)) in enumerate(zip(row["spans"], offs))]
        nr["word_times"] = None  # the word count may change: re-align before cutting crops
        out.append(nr)
        st["kept_relabelled"] += 1
    kept_ids = {r["id"] for r in rows}
    for dg in diag:
        if dg.get("reason") != "undecided" or dg["id"] in kept_ids:
            continue
        row, e = cands.get(dg["id"]), ext.get(dg["id"])
        if not row or not e:
            st["undecided_no_row_or_ext"] += 1
            continue
        if dg.get("trim") or row["duration"] > 20.0:
            st["undecided_trimmed_or_long"] += 1  # the audio label2 decided on is not the candidate row's
            continue
        sp, now = M.spans_regional(row["text"], lang), M.spans(row["text"], lang)
        if len(sp) != len(dg["spans"]):
            st["undecided_unaligned"] += 1
            continue
        if any(d.get("cands") is not None and d["cands"] != s.cands for s, d in zip(now, dg["spans"])):
            st["undecided_stale_diag"] += 1  # decided on older readings (an earlier ml_cands): relabel that row anew
            continue
        ds, ok = [], True
        for s, d in zip(sp, dg["spans"]):
            if d["decided"]:  # diag rows hold the picked reading; make_row wants its index
                pk = d["pick"] if isinstance(d["pick"], int) else next(
                    (j for j, c in enumerate(d["cands"]) if c.strip() == str(d["pick"]).strip()), None)
                if pk is None:
                    ok = False
                    break
                ds.append(dict(d, pick=pk))
                continue
            base = d.get("cands") or []
            extra = [c for c in s.cands if c not in base]
            hits = [c for c in extra if _has_run(c, e)]
            if not base or len({D.key(c) for c in hits}) != 1 or any(_has_run(c, e) for c in base):
                ok = False
                break
            cs = base + extra
            ds.append(dict(d, cands=cs, pick=cs.index(max(hits, key=len)), decided=True, by="ext_regional"))
        if not ok:
            st["undecided_not_recovered"] += 1
            continue
        res = C.make_row(row, {"spans": ds, "greedy": dg.get("greedy", "")}, None, ext=e)
        if not res.get("row"):
            st["recover_rejected_" + res.get("reason", "?")] += 1
            continue
        nr = dict(res["row"], regional=[i for i, d in enumerate(ds) if d.get("by") == "ext_regional"], word_times=None)
        out.append(nr)
        st["undecided_recovered"] += 1
    return out, dict(st)


def regional_main(argv=None):
    ap = argparse.ArgumentParser(description="ext-decided regional readings after label2 (see regional_relabel)")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = list(_read([a.rows]))
    diag = list(_read([a.diag]))
    ext = {r["id"]: r.get("text") or "" for r in _read([a.ext]) if r.get("status", "completed") == "completed"}
    cands = {r["id"]: r for r in _read(a.cands)}
    out, st = regional_relabel(a.lang, rows, diag, ext, cands)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"lang": a.lang, "rows_in": len(rows), "rows_out": len(out), **st}, ensure_ascii=False))
    for r in out:
        if r.get("regional"):
            print("   ", r["id"][:40], "|", r["text"][:150])


# ------------------------------------------------------------------------------------------------- number-free rows
PLAIN_MAX_CER = {"yodas": 0.12, "vox": 0.15, "tts": 0.15}


def plain_check(lang: str, rows: list, ext: dict) -> tuple[list, dict]:
    """Number-free look-alike / control rows: the caption is the label when the external transcript agrees (CER bound
    of label2 for the source, no 2-word caption gap), the transcript has no digit and no number word the caption lacks,
    and every look-alike word the caption was picked for is in the transcript as that word."""
    import ml_decide as D
    import ml_p3_core as C

    D.set_script(lang)
    C.set_script(lang)
    numw = re.compile(r"\b(?:" + _NUMW[lang] + r")\b", re.I)
    st, out = Counter(), []
    for r in rows:
        e = ext.get(r["id"])
        if e is None:
            st["no_ext"] += 1
            continue
        if re.search(r"\d", e) or len(numw.findall(e)) > len(numw.findall(r["text"])):
            st["ext_has_number"] += 1  # the audio may hold a number the caption left out
            continue
        c = C.cer(r["text"], e)
        if c > PLAIN_MAX_CER.get(r.get("src"), 0.12):
            st["cer"] += 1
            continue
        if C.word_gap(r["text"], e) >= 2:
            st["caption_gap"] += 1
            continue
        hits = [m for k, ms in classify(r["text"], lang).items() if k in ("look_homo", "look_idiom") for m in ms]
        ew = " ".join(_words(e))
        if any(" ".join(_words(h)) not in ew for h in hits):
            st["lookalike_not_heard"] += 1
            continue
        out.append(dict(r, cer_ext=round(c, 3), spans=[], num_chars=[], caption=r["text"]))
        st["kept_" + (r.get("kinds") or ["?"])[0]] += 1
    return out, dict(st)


def _spoken(text: str, lang: str) -> str:
    """A transcript with its digits read out (canonical ml_cands readings; wave 1 only): some recognizer models write
    digits ("6 keer 7", "8,5 miljard"), the labels never do."""
    import ml_cands as M

    if lang not in M.LANGS or not re.search(r"\d", text):
        return text
    try:
        sp = [s for s in M.spans(text, lang) if s.cands]
        return M.render(text, sp, [0] * len(sp))
    except Exception:  # noqa: BLE001
        return text


def _look_words(hits: list, lang: str) -> list:
    """The look-alike words of a classify() hit ("for 4" -> "for"): letters only, number words left out."""
    numw = re.compile(r"^(?:" + _NUMW[lang] + r")$", re.I)
    return [w for h in hits for w in re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", h) if not numw.match(w)]


def synth_check(lang: str, rows: list, ext: dict, max_cer: float = 0.15) -> tuple[list, dict]:
    """Synthetic look-alike / business rows against an external transcript of the clip: the transcript (its digits read
    out for wave 1) matches the spoken label, or the written caption for a recognizer that writes digits, within
    max_cer, and every look-alike word of the caption is in it as that word (a "nouă" heard as "9" fails)."""
    import ml_decide as D
    import ml_p3_core as C

    D.set_script(lang)
    C.set_script(lang)  # the CER keeps Cyrillic letters only after this

    def norm(t):
        return " ".join(re.sub(r"[^\w\s]", "", t.lower().replace("-", " ").replace("ё", "е")).split())

    st, out = Counter(), []
    for r in rows:
        e = ext.get(r["id"])
        if not e:
            st["no_ext"] += 1
            continue
        c = min(C.cer(norm(r["text"]), norm(_spoken(e, lang))), C.cer(norm(r["caption"]), norm(e)))
        if c > max_cer:
            st[f"{r['cls']}:cer"] += 1
            continue
        if _value_mismatch(r["caption"], e):
            st[f"{r['cls']}:value_mismatch"] += 1  # a digit-writing recognizer heard another value (15 vs 25)
            continue
        hits = [m for k, ms in classify(r["caption"], lang).items() if k in ("look_homo", "look_idiom") for m in ms]
        ew = set(norm(e).split())
        if any(norm(w) not in ew for w in _look_words(hits, lang)):
            st[f"{r['cls']}:lookalike_not_heard"] += 1
            continue
        out.append(dict(r, cer_ext=round(c, 3)))
        st[f"{r['cls']}:kept"] += 1
    return out, dict(sorted(st.items()))

if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["regional"]:
        regional_main(sys.argv[2:])
    else:
        main()
