#!/usr/bin/env python3
"""Builds fixtures/known_answer_rows.json for test_known_answers.py: SYNTHETIC rows (invented carrier sentences with one
number each), 30 per language for de es fr it nl pt. No real held-out text is used.

    pip install num2words
    python3 make_known_answer_fixture.py            # rewrites fixtures/known_answer_rows.json

The spoken form comes from num2words (a number-to-words library independent of the scorer's own lexicon). Per row:
  caption  the written form (digits, symbol); label  the spoken form; seg  the label words of the number;
  unit / dec  the spoken unit / decimal word inside seg ("" when none);
  hyps     hypotheses derived from the label by one deliberate corruption each (only where it applies):
    drop_unit   the unit word is removed          drop_dec  the decimal word is removed
    swap        one number morpheme is another (a different value)
    trunc       the final scale word is cut (thousand / million)
    respace     the same words with the spaces / hyphens moved (value unchanged)
    fuse        the spoken number's separate words glued into one invented word (es fr pt, two or more number words)
    misplit     a space inside the last number word, in the middle of a morpheme (a garbled word)
    en_words    the number spoken in English words (an English leak)
"""
import json
import os
import re
import sys

from num2words import num2words as n2w

import number_outcomes as N

LANGS = ["de", "es", "fr", "it", "nl", "pt"]
CARRIERS = {
    "de": ["Laut dem Bericht liegt der Wert im Moment bei {X} und steigt weiter.", "Am Montag nannte die Firma {X} als Ziel.",
           "Im Archiv steht {X} neben dem alten Eintrag.", "Die Schule meldete {X} für das kommende Jahr.",
           "Der Vertrag sieht {X} vor, sagte der Anwalt."],
    "es": ["Según el informe, el valor actual es {X} y sigue subiendo.", "El lunes la empresa dio {X} como objetivo.",
           "En el archivo figura {X} junto a la entrada antigua.", "La escuela anunció {X} para el próximo año.",
           "El contrato prevé {X}, dijo el abogado."],
    "fr": ["Selon le rapport, la valeur actuelle est de {X} et continue de monter.", "Lundi, l'entreprise a donné {X} comme objectif.",
           "Dans les archives, on trouve {X} à côté de l'ancienne entrée.", "L'école a annoncé {X} pour l'année prochaine.",
           "Le contrat prévoit {X}, a dit l'avocat."],
    "it": ["Secondo il rapporto, il valore attuale è {X} e continua a salire.", "Lunedì l'azienda ha indicato {X} come obiettivo.",
           "Nell'archivio compare {X} accanto alla vecchia voce.", "La scuola ha annunciato {X} per il prossimo anno.",
           "Il contratto prevede {X}, ha detto l'avvocato."],
    "nl": ["Volgens het rapport ligt de huidige waarde op {X} en dat blijft stijgen.", "Maandag noemde het bedrijf {X} als doel.",
           "In het archief staat {X} naast de oude vermelding.", "De school meldde {X} voor volgend jaar.",
           "Het contract voorziet in {X}, zei de advocaat."],
    "pt": ["Segundo o relatório, o valor atual é {X} e continua a subir.", "Na segunda-feira a empresa indicou {X} como objetivo.",
           "No arquivo consta {X} ao lado da entrada antiga.", "A escola anunciou {X} para o próximo ano.",
           "O contrato prevê {X}, disse o advogado."],
}
UNITS = {  # spoken unit words (pct, euro, km, kg) and the decimal word
    "de": {"pct": "Prozent", "eur": "Euro", "km": "Kilometer", "kg": "Kilogramm", "dec": "Komma"},
    "es": {"pct": "por ciento", "eur": "euros", "km": "kilómetros", "kg": "kilogramos", "dec": "coma"},
    "fr": {"pct": "pour cent", "eur": "euros", "km": "kilomètres", "kg": "kilogrammes", "dec": "virgule"},
    "it": {"pct": "per cento", "eur": "euro", "km": "chilometri", "kg": "chilogrammi", "dec": "virgola"},
    "nl": {"pct": "procent", "eur": "euro", "km": "kilometer", "kg": "kilogram", "dec": "komma"},
    "pt": {"pct": "por cento", "eur": "euros", "km": "quilómetros", "kg": "quilogramas", "dec": "vírgula"},
}
CENTS_JOIN = {"de": " ", "es": " con ", "fr": " ", "it": " e ", "nl": " ", "pt": " e "}
SCALE = {"de": r"(?:tausend|Millionen)", "es": r"(?:mil|millones)", "fr": r"(?:mille|millions)", "it": r"(?:mila|milioni)",
         "nl": r"(?:duizend|miljoen)", "pt": r"(?:mil|milhões)"}
# (kind, value or text, written form)
ROWS = (
    [("int", v) for v in (7, 13, 21, 28, 45, 96, 99)]
    + [("int", v) for v in (250, 4500, 12000, 25000, 400000, 600000)]
    + [("year", v) for v in (1998, 2005, 2018, 2024)]
    + [("pct", v) for v in (28, 5, 40, 12)]
    + [("eur", v) for v in (250, 40)] + [("price", (2, 80))]
    + [("km", 30), ("kg", 5), ("km", 12)]
    + [("dec", v) for v in ("3,5", "2,5", "12,5")]
)
assert len(ROWS) == 30


def thousands(v: int, lang: str) -> str:
    if v < 10000:
        return str(v)
    sep = " " if lang == "fr" else "."
    return f"{v:,}".replace(",", sep)


def words(v: int, lang: str, year: bool = False) -> str:
    return n2w(v, lang=lang, to="year") if year else n2w(v, lang=lang)


def lab_of(sentence: str) -> str:
    s = re.sub(r"[.,;!?]", "", sentence)
    return s[0].lower() + s[1:]


def build_row(lang: str, i: int, kind: str, val) -> dict:
    u, carrier = UNITS[lang], CARRIERS[lang][i % 5]
    unit = dec = ""
    cur_hyp = {}  # variant -> seg

    def swap_int(v: int) -> int:
        s = str(v)
        return int(str(int(s[0]) + 1 if s[0] != "9" else 8) + s[1:])

    if kind == "int":
        cap, seg = thousands(val, lang), words(val, lang)
        sw = words(swap_int(val), lang)
        en = n2w(val, lang="en")
    elif kind == "year":
        cap, seg = str(val), words(val, lang, year=True)
        sw = words(val - 10 if val % 100 >= 10 else val + 10, lang, year=True)
        en = n2w(val, lang="en", to="year")
    elif kind == "pct":
        unit = u["pct"]
        cap, seg = f"{val} %", f"{words(val, lang)} {unit}"
        sw = f"{words(swap_int(val), lang)} {unit}"
        en = f"{n2w(val, lang='en')} {unit}"
    elif kind == "eur":
        unit = u["eur"]
        cap, seg = f"{thousands(val, lang)} €", f"{words(val, lang)} {unit}"
        sw = f"{words(swap_int(val), lang)} {unit}"
        en = f"{n2w(val, lang='en')} {unit}"
    elif kind == "price":
        unit = u["eur"]
        a, b = val
        cap, seg = f"{a},{b} €", f"{words(a, lang)} {unit}{CENTS_JOIN[lang]}{words(b, lang)}"
        sw = f"{words(a, lang)} {unit}{CENTS_JOIN[lang]}{words(b + 10, lang)}"
        en = f"{n2w(a, lang='en')} {unit}{CENTS_JOIN[lang]}{n2w(b, lang='en')}"
    elif kind in ("km", "kg"):
        unit = u[kind]
        cap, seg = f"{val} {kind}", f"{words(val, lang)} {unit}"
        sw = f"{words(swap_int(val), lang)} {unit}"
        en = f"{n2w(val, lang='en')} {unit}"
    else:  # dec
        dec = u["dec"]
        whole, frac = val.split(",")
        spoken = lambda w, f, lg: f"{n2w(int(w), lang=lg)} {{D}} " + " ".join(n2w(int(c), lang=lg) for c in f)  # noqa: E731
        cap = val
        seg = spoken(whole, frac, lang).format(D=dec)
        frac2 = frac[:-1] + str((int(frac[-1]) + 1) % 10)
        sw = spoken(whole, frac2, lang).format(D=dec)
        en = f"{n2w(int(whole), lang='en')} point " + " ".join(n2w(int(c), lang="en") for c in frac)
    caption = carrier.format(X=cap)
    label = lab_of(carrier.format(X=seg))
    row = {"lang": lang, "id": f"ka_{lang}_{i:02d}", "kind": kind, "caption": caption, "label": label, "seg": seg,
           "unit": unit, "dec": dec, "hyps": {}}

    def put(name, new_seg):
        if new_seg != seg:
            row["hyps"][name] = lab_of(carrier.format(X=new_seg))

    put("swap", sw)
    put("en_words", en)
    if unit and kind != "price":
        put("drop_unit", re.sub(rf"\s*{re.escape(unit)}$", "", seg))
    if unit and kind == "price":
        put("drop_unit", seg.replace(f" {unit}", "", 1))
    if dec:
        put("drop_dec", seg.replace(f" {dec} ", " ", 1))
    if kind in ("int", "eur", "pct", "km", "kg") and isinstance(val, int) and re.search(SCALE[lang] + r"(?:\s+\w+)?$", seg):
        body = re.sub(rf"\s*{SCALE[lang]}(?=(?:\s+{re.escape(unit)})?$)", "", seg) if unit else re.sub(rf"\s*{SCALE[lang]}$", "", seg)
        put("trunc", body)
    # respace: the value is unchanged. Fused compounds (de nl it) are cut at their morpheme joins, hyphenated ones
    # (fr) lose the hyphen; es / pt are already spelled in separate words, so they have none.
    if lang in ("de", "nl", "it") and kind in ("int", "year", "pct", "eur", "km", "kg"):
        num_only = re.sub(rf"\s*{re.escape(unit)}$", "", seg) if unit else seg
        cut = re.sub(r"(hundert|honderd|cento|tausend|duizend|mila)(?=[a-zäöü])", r"\1 ", num_only)
        join = {"de": "und", "nl": "en"}.get(lang)
        if join and isinstance(val, int) and val % 100 > 20 and val % 10:  # "achtundzwanzig" -> "acht und zwanzig"
            tens = words(val % 100 - val % 10, lang)
            if cut.endswith(tens) and cut[: -len(tens)].endswith(join):
                cut = cut[: -len(tens) - len(join)] + f" {join} " + tens
        put("respace", cut + (f" {unit}" if unit else ""))
    num_part = re.sub(rf"\s*{re.escape(unit)}$", "", seg) if unit and kind != "price" else seg
    if kind in ("int", "year", "pct", "eur", "km", "kg"):
        toks = re.split(r"[\s-]+", num_part)
        if lang in ("es", "fr", "pt") and len(toks) >= 2:
            put("fuse", "".join(toks) + (f" {unit}" if unit else ""))
        last = toks[-1]
        if len(last) >= 4:
            h = len(last) // 2
            bad = num_part[: len(num_part) - len(last)] + last[:h] + " " + last[h:] + (f" {unit}" if unit else "")
            reads = lambda t: bool(N.extract_numbers(t, lang, keep_weak=True, garbles=False))  # noqa: E731
            if bad != row["hyps"].get("respace") and not (reads(last[:h]) or reads(last[h:])):
                # kept only when neither piece is itself a number word: a cut on a morpheme join is the benign respace
                put("misplit", bad)
    if lang == "fr" and "-" in seg:
        put("respace", seg.replace("-", " "))
    return row


def main():
    rows = []
    for lang in LANGS:
        for i, (kind, val) in enumerate(ROWS):
            rows.append(build_row(lang, i, kind, val))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "known_answer_rows.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
