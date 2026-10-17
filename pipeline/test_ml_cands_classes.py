"""Price / phone / code / clock / "and a half" candidate checks (wave-1 data, 2026-10-15), plus the wave-2 classes:
units, signs, dates, ranges, decades, letter+number codes, roman numerals, fractions, versions / IPs, "&", per mille.

    python3 test_ml_cands_classes.py      # needs ml_grammar_table.json (build_table over these texts)
"""

import re
import sys

import ml_cands as M

TEXTS = {
    "pt": ["Custa R$ 200.", "Custa R$ 1,50.", "Custa R$19,90 hoje.", "Ligue 11 9 4704 8697 agora.", "O valor é 2,5 %.",
           "Às 18h30 começa.", "Custa 1 £.", "Custa R$3.600.",
           "Llama al +34 612 345 678.", "Ligue 11 2094 4214 agora.", "O WhatsApp é o 991 85 54 18.", "Custa A$ 50.",
           "Custa R$ 5.", "em 2020-2021-2022 houve"],
    "es": ["Cuesta 1 £.", "Cuesta $30.", "Llama al 612 345 678.", "Son las 1:15.", "Creció 2,5 puntos."],
    "fr": ["Ça coûte 1 £.", "Appelez le 06 12 34 56 78.", "Il est 7:01.", "Un million: 1.000.000 $.",
           "Un million: 1.000.000 €.",
           "Ils sont 12 345 678 personnes."],
    "it": ["Costa 1 €.", "Il numero è 02 1234 5678.", "Sono le 11:20."],
    "de": ["Es kostet 1,50 €.", "Es sind 2,5 Prozent.", "Die Nummer ist 0612345678."],
    "nl": ["Het is 1,5 keer zo veel.", "Het kost 7 €."],
}


def sp(text, lang):
    return M.spans(text, lang)


def has(text, lang, i, reading):
    s = sp(text, lang)[i]
    return any(c.strip() == reading for c in s.cands), (text, s.kind, s.cands[:12])


def _hour_word_agrees():
    """The hour word agrees with the hour (judge audit 2026-10-17: "quinze hora e meia", "une heures" were generated)."""
    bad = []
    for lang, t in (("pt", "Às 15:30 sai."), ("pt", "Às 01:30 sai."), ("pt", "Às 02:10 sai."), ("fr", "À 13:34 il part."),
                    ("fr", "À 1:34 il part.")):
        for c in M.spans(t, lang)[0].cands:
            w = c.split()
            for i, u in enumerate(w):
                one = i > 0 and w[i - 1] in ("uma", "um", "une", "un", "zero", "zéro")
                if (u in ("hora", "heure") and not one) or (u in ("horas", "heures") and i > 0 and w[i - 1] in ("uma", "um", "une", "un")):
                    bad.append(c)
    return not bad, bad


CHECKS = [
    # currency: R$ = reais, unit gender, cents with/without the cent word, pesos for "$" in Spanish
    lambda: has("Custa R$ 200.", "pt", 0, "duzentos reais"),
    lambda: (all("dólares" not in c for c in sp("Custa R$ 200.", "pt")[0].cands), sp("Custa R$ 200.", "pt")[0].cands),
    lambda: has("Custa R$ 1,50.", "pt", 0, "um real e cinquenta"),
    lambda: has("Custa R$ 1,50.", "pt", 0, "um real e cinquenta centavos"),
    lambda: has("Custa R$19,90 hoje.", "pt", 0, "dezenove e noventa"),
    lambda: (all("seiscentas" not in c for c in sp("Custa R$3.600.", "pt")[0].cands), sp("Custa R$3.600.", "pt")[0].cands),
    lambda: has("Custa 1 £.", "pt", 0, "uma libra"),
    lambda: has("Cuesta 1 £.", "es", 0, "una libra"),
    lambda: has("Ça coûte 1 £.", "fr", 0, "une livre"),
    lambda: has("Costa 1 €.", "it", 0, "un euro"),
    lambda: (all(c.split()[0] != "una" for c in sp("Costa 1 €.", "it")[0].cands), sp("Costa 1 €.", "it")[0].cands),
    lambda: has("Cuesta $30.", "es", 0, "treinta pesos"),
    lambda: has("Es kostet 1,50 €.", "de", 0, "ein Euro fünfzig"),
    lambda: has("Un million: 1.000.000 $.", "fr", 0, "un million de dollars"),
    lambda: has("Un million: 1.000.000 €.", "fr", 0, "un million d'euros"),
    # phone numbers: one span per group, separators dropped, groupings as chunks
    lambda: (len(sp("Appelez le 06 12 34 56 78.", "fr")) == 5, [s.text for s in sp("Appelez le 06 12 34 56 78.", "fr")]),
    lambda: has("Appelez le 06 12 34 56 78.", "fr", 0, "zéro six"),
    lambda: has("Appelez le 06 12 34 56 78.", "fr", 1, "douze"),
    lambda: has("Llama al 612 345 678.", "es", 0, "seiscientos doce"),
    lambda: has("Llama al 612 345 678.", "es", 0, "seis uno dos"),
    lambda: has("Ligue 11 9 4704 8697 agora.", "pt", 2, "quarenta e sete zero quatro"),
    lambda: has("Il numero è 02 1234 5678.", "it", 1, "dodici trentaquattro"),
    # French space-grouped thousands stay one figure, not a phone number
    lambda: (all(s.kind != "phone" for s in sp("Ils sont 12 345 678 personnes.", "fr")),
             [(s.text, s.kind) for s in sp("Ils sont 12 345 678 personnes.", "fr")]),
    # not phone numbers (review H2): dates, year runs, score / round-number lists; foreign dollar signs are not "$"
    *[(lambda t=t, l=l: (all(x.kind != "phone" for x in sp(t, l)), [(x.text, x.kind) for x in sp(t, l)]))
      for t, l in [("Le 12-05-2020 à midi.", "fr"), ("le 2020-10-15 matin", "fr"), ("em 2020-2021-2022 houve", "pt"),
                   ("de 10 à 12 15 18 20 heures", "fr"), ("Sono 100 200 300 persone", "it"),
                   ("scores 2 1 0 3 4 2 1 ok", "fr"), ("son 1 2 3 4 5 6 7 ya", "es")]],
    *[(lambda t=t, l=l: (sp(t, l)[0].kind == "phone", [(x.text, x.kind) for x in sp(t, l)]))
      for t, l in [("Llama al +34 612 345 678.", "es"), ("Ligue 11 2094 4214 agora.", "pt"),
                   ("O WhatsApp é o 991 85 54 18.", "pt")]],
    lambda: (all(x.kind != "cur" for x in sp("Custa A$ 50.", "pt")), [(x.text, x.kind) for x in sp("Custa A$ 50.", "pt")]),
    lambda: (sp("Custa R$ 5.", "pt")[0].kind == "cur", sp("Custa R$ 5.", "pt")[0].kind),
    # unbroken code with a leading zero: digit by digit and chunked readings
    lambda: (sp("Die Nummer ist 0612345678.", "de")[0].kind == "code", sp("Die Nummer ist 0612345678.", "de")[0].kind),
    lambda: has("Die Nummer ist 0612345678.", "de", 0, "null sechs eins zwei drei vier fünf sechs sieben acht"),
    # clock: 12-hour reading, half / quarter past, feminine hour
    lambda: has("Às 18h30 começa.", "pt", 0, "seis e meia"),
    lambda: has("Son las 1:15.", "es", 0, "una y cuarto"),
    lambda: has("Sono le 11:20.", "it", 0, "undici e venti"),
    lambda: has("Il est 7:01.", "fr", 0, "sept heures un"),
    # "and a half"
    lambda: has("Creció 2,5 puntos.", "es", 0, "dos y medio"),
    lambda: has("Es sind 2,5 Prozent.", "de", 0, "zweieinhalb Prozent"),
    lambda: has("Het is 1,5 keer zo veel.", "nl", 0, "anderhalf"),
    lambda: has("O valor é 2,5 %.", "pt", 0, "dois e meio por cento"),
]


# ---------------------------------------------------------------- wave-2 classes
# (lang, text, span index, a reading that must be among that span's candidates; whitespace around it ignored)
POS = [
    # units: number + unit words, singular / plural by value, decimals, feminine units, "and a half"
    ("es", "Corrió 20 km hoy.", 0, "veinte kilómetros"), ("es", "Dura 1 h.", 0, "una hora"),
    ("es", "Son 2,5 kg de harina.", 0, "dos kilogramos y medio"), ("es", "Son 2,5 kg de harina.", 0, "dos coma cinco kilos"),
    ("es", "Hace -3 °C hoy.", 0, "menos tres grados centígrados"), ("es", "Va a 120 km/h.", 0, "ciento veinte kilómetros por hora"),
    ("es", "Mide 50 m² ya.", 0, "cincuenta metros cuadrados"), ("es", "Tiene 1 GB.", 0, "un gigabyte"),
    ("es", "Tiene 500 MB.", 0, "quinientos megas"), ("es", "Pesa 1 t.", 0, "una tonelada"), ("es", "Hay 30° hoy.", 0, "treinta grados"),
    ("es", "Subió 5‰ ayer.", 0, "cinco por mil"), ("es", "Dura 2,5 h.", 0, "dos horas y media"),
    ("pt", "Correu 20 km hoje.", 0, "vinte quilômetros"), ("pt", "Pesa 1,5 kg.", 0, "um quilograma e meio"),
    ("pt", "Dura 1 h.", 0, "uma hora"), ("pt", "Dura 2,5 h.", 0, "duas horas e meia"), ("pt", "Faz 30 °C.", 0, "trinta graus Celsius"),
    ("pt", "Subiu 5‰ ontem.", 0, "cinco por mil"),
    ("fr", "Il roule à 90 km/h.", 0, "quatre vingt dix kilomètres par heure"), ("fr", "Il pèse 1 kg.", 0, "un kilogramme"),
    ("fr", "Il fait -3 °C.", 0, "moins trois degrés Celsius"), ("fr", "Il pèse 1,5 kg.", 0, "un kilogramme et demi"),
    ("fr", "Ça dure 1 h.", 0, "une heure"), ("fr", "Il fait 20 cm.", 0, "vingt centimètres"),
    ("it", "Dura 1 h.", 0, "un'ora"), ("it", "Pesa 1,5 kg.", 0, "un chilo e mezzo"), ("it", "Va a 90 km/h.", 0, "novanta chilometri orari"),
    ("it", "Pesa 450 g.", 0, "quattrocentocinquanta grammi"),
    ("nl", "Het weegt 1,5 kg.", 0, "anderhalf kilogram"), ("nl", "Hij rijdt 90 km/h.", 0, "negentig kilometer per uur"),
    ("nl", "Het is -3 °C.", 0, "min drie graden Celsius"), ("nl", "Het is 20 cm.", 0, "twintig centimeter"),
    ("de", "Es wiegt 1,5 kg.", 0, "eineinhalb Kilogramm"), ("de", "Er lief 1 h.", 0, "eine Stunde"),
    ("de", "Es hat -3 °C.", 0, "minus drei Grad Celsius"), ("de", "Er fährt 90 km/h.", 0, "neunzig Kilometer pro Stunde"),
    ("de", "Er lief 20 km hin.", 0, "zwanzig Kilometer"), ("de", "Sie hat 5‰ drin.", 0, "fünf Promille"),
    ("fr", "Il a 5‰ dans le sang.", 0, "cinq pour mille"), ("it", "Ha il 5‰ di alcol.", 0, "cinque per mille"),
    ("nl", "Hij heeft 5‰ drank.", 0, "vijf promille"),
    # signs
    ("es", "La cifra es +2 ahora.", 0, "más dos"), ("de", "Der Wert ist -5 jetzt.", 0, "minus fünf"),
    ("it", "Il saldo è -5 % oggi.", 0, "meno cinque per cento"), ("pt", "O saldo é +3 hoje.", 0, "mais três"),
    ("fr", "Le solde est −5 aujourd'hui.", 0, "moins cinq"), ("nl", "De stand is -5 nu.", 0, "min vijf"),
    # dates: day / month / year spans
    ("es", "Es el 12/05/2020 hoy.", 0, "doce"), ("es", "Es el 12/05/2020 hoy.", 1, "de mayo"),
    ("es", "Es el 12/05/2020 hoy.", 1, "del cinco"), ("es", "Es el 12/05/2020 hoy.", 2, "de dos mil veinte"),
    ("es", "Es el 1/5/2020 hoy.", 0, "primero"),
    ("pt", "Em 12/05/2020 foi.", 0, "doze"), ("pt", "Em 12/05/2020 foi.", 1, "de maio"), ("pt", "Em 12/05/2020 foi.", 2, "de dois mil e vinte"),
    ("fr", "Le 12/05/2020 était mardi.", 1, "mai"), ("fr", "Le 1/5/2020 était mardi.", 0, "premier"),
    ("fr", "Le 12/05/2020 était mardi.", 2, "deux mille vingt"),
    ("it", "Il 12/05/2020 era martedì.", 1, "maggio"), ("it", "Il 12/05/2020 era martedì.", 2, "duemilaventi"),
    ("nl", "Op 12-05-2020 was het.", 1, "mei"), ("nl", "Op 12-05-2020 was het.", 2, "tweeduizendtwintig"),
    ("de", "Am 12.05.2020 war es so.", 0, "zwölften"), ("de", "Am 12.05.2020 war es so.", 1, "Mai"),
    ("de", "Am 12.05.2020 war es so.", 2, "zweitausendzwanzig"), ("de", "Am 12.05. kommt er.", 1, "Mai"),
    ("de", "Am 12.05. kommt er.", 1, "fünften"), ("es", "Fue el 3/9/98 aquí.", 2, "noventa y ocho"),
    # ranges: left end, then joiner + right end
    ("es", "Del 10-12 de mayo.", 0, "diez"), ("es", "Del 10-12 de mayo.", 1, "a doce"), ("es", "Del 10-12 de mayo.", 1, "al doce"),
    ("es", "Entre 2010-2014 subió.", 1, "a dos mil catorce"), ("es", "Entre 2010-14 subió.", 1, "a dos mil catorce"),
    ("es", "Ganó 26-0 ayer.", 1, "a cero"), ("es", "Son de 10-12 km hoy.", 1, "a doce kilómetros"),
    ("es", "Subió 5-10 % hoy.", 1, "a diez por ciento"), ("es", "Del 10 – 12 de mayo.", 1, "a doce"),
    ("pt", "Entre 10-12 anos.", 1, "até doze"), ("pt", "Desde 1995-96 foi.", 1, "a mil novecentos e noventa e seis"),
    ("fr", "Entre 10-12 ans.", 1, "à douze"), ("it", "Tra 10-12 anni.", 1, "fino a dodici"),
    ("nl", "Tussen 10-12 jaar.", 1, "tot en met twaalf"), ("de", "Zwischen 10-12 Jahren.", 1, "bis zwölf"),
    ("de", "Zwischen 2010-14 stieg es.", 1, "bis zweitausendvierzehn"),
    # decades
    ("es", "En los años 90 todo cambió.", 0, "noventa"), ("it", "Negli anni '90 tutto cambiò.", 0, "novanta"),
    ("it", "Nel '90 tutto cambiò.", 0, "novanta"), ("fr", "Dans les années 80 tout a changé.", 0, "quatre vingts"),
    ("de", "In den 90ern war das anders.", 0, "Neunzigern"), ("de", "Die 90er-Jahre waren toll.", 0, "neunziger"),
    ("de", "In den 1980er Jahren war es so.", 0, "neunzehnhundertachtziger"),
    ("nl", "In de jaren 90 was het anders.", 0, "negentig"), ("pt", "Nos anos 90 foi bom.", 0, "noventa"),
    # letter + number codes
    ("es", "Tiene 5G ya.", 0, "cinco ge"), ("es", "Un papel A4 aquí.", 0, "a cuatro"), ("es", "La COVID-19 llegó.", 0, "diecinueve"),
    ("de", "Das 3D Kino.", 0, "drei de"), ("de", "Der CO2 Gehalt.", 0, "ze o zwei"), ("fr", "La COVID-19 est là.", 0, "dix neuf"),
    ("it", "Il G7 sull'energia.", 0, "gi sette"), ("nl", "Een 4K scherm.", 0, "vier ka"), ("pt", "O G20 reuniu.", 0, "gê vinte"),
    ("de", "Die COVID-19-Pandemie war da.", 0, "neunzehn"),
    # roman numerals
    ("es", "Luis XIV reinó.", 0, "catorce"), ("es", "En el siglo XXI hay más.", 0, "veintiuno"),
    ("es", "Felipe II reinó.", 0, "segundo"), ("fr", "Au XXIe siècle tout va vite.", 0, "vingt et unième"),
    ("fr", "Louis XIV régnait.", 0, "quatorze"), ("de", "Ludwig XIV. regierte lange.", 0, "der Vierzehnte"),
    ("de", "Im XIX. Jahrhundert war das so.", 0, "neunzehnten"), ("it", "Nel secolo XXI c'è più.", 0, "ventunesimo"),
    ("it", "Luigi XIV regnava.", 0, "quattordicesimo"), ("nl", "Lodewijk XIV regeerde.", 0, "de veertiende"),
    ("pt", "No século XXI há mais.", 0, "vinte e um"),
    # fractions, mixed numbers, 24/7
    ("es", "Falta 3/4 de litro.", 0, "tres cuartos"), ("es", "Falta 1/3 de litro.", 0, "un tercio"), ("es", "Falta ½ litro.", 0, "la mitad"),
    ("es", "Mide 2 1/2 metros.", 0, "dos y medio"), ("es", "Trabajan 24/7 aquí.", 0, "veinticuatro siete"),
    ("pt", "Falta 3/4 do total.", 0, "três quartos"), ("fr", "Il reste 1/3 du total.", 0, "un tiers"),
    ("fr", "Il reste 3/4 du total.", 0, "trois quarts"), ("fr", "Ouvert 24/7 ici.", 0, "vingt quatre sur sept"),
    ("it", "Resta 3/4 del totale.", 0, "tre quarti"), ("it", "Resta 1/3 del totale.", 0, "un terzo"),
    ("nl", "Er is 3/4 van de tijd.", 0, "drie kwart"), ("de", "Es sind 3/4 davon.", 0, "drei Viertel"),
    ("de", "Es ist 1/2 davon.", 0, "ein halb"), ("de", "Es sind 2½ Liter.", 0, "zweieinhalb"),
    # versions, IPs
    ("es", "Versión 2.0 nueva.", 0, "dos punto cero"), ("de", "Version 2.0 neu.", 0, "zwei Punkt null"),
    ("fr", "Version 2.0 neuve.", 0, "deux point zéro"), ("it", "Versione 2.0 nuova.", 0, "due punto zero"),
    ("nl", "Versie 2.0 nieuw.", 0, "twee punt nul"), ("pt", "Versão 2.0 nova.", 0, "dois ponto zero"),
    ("es", "Es la 5.1.3 ya.", 1, "punto uno"), ("es", "La IP 192.168.0.1 sirve.", 1, "punto ciento sesenta y ocho"),
    # ampersand
    ("es", "Una sesión Q&A hoy.", 0, "cu y a"), ("de", "Die M&A Abteilung.", 0, "em und a"), ("fr", "Le service R&D ici.", 0, "erre et dé"),
    ("it", "Il Q&A di oggi.", 0, "cu e a"), ("es", "Smith & Wesson hoy.", 0, "y"), ("de", "Smith & Wesson hier.", 0, "and"),
    ("nl", "Een Q&A nu.", 0, "kuu en a"),
    # decimals with a point and a clock written with a dot
    ("es", "Mide 3.14 ya.", 0, "tres punto catorce"), ("de", "Um 15.45 Uhr gehen wir.", 0, "fünfzehn Uhr fünfundvierzig"),
]
# texts whose digits must stay unspanned-or-skipped, or must NOT take the named kind: (lang, text, predicate over [(text, kind)])
def _kinds(text, lang):
    return [(s.text, s.kind) for s in M.spans(text, lang)]


NEG = [
    # hyphen compounds and chains are not ranges / signs; ISO dates, phone-like triples stay unspanned
    ("de", "Ein 13-jähriger Junge.", lambda k: [x[1] for x in k] == ["skip"]),
    ("de", "Die Il-76 flog.", lambda k: all(x[1] == "skip" for x in k) or not k),
    ("fr", "C'est 132-250-84.", lambda k: all(x[1] == "skip" for x in k)),
    ("es", "Fue el 2020-10-15 ya.", lambda k: all(x[1] == "skip" for x in k)),
    ("es", "Son 2-12-5 ya.", lambda k: all(x[1] == "skip" for x in k)),
    ("nl", "Het 54-jarige type.", lambda k: [x[1] for x in k] == ["skip"]),
    # "ha" is a verb in Spanish / Italian, never hectares; wrong-case units are not units
    ("es", "Europa 2020 ha crecido.", lambda k: k == [("2020", "int")]),
    ("it", "Lui ha 2020 cose.", lambda k: k == [("2020", "int")]),
    ("es", "Pesa 5 S aquí.", lambda k: all(x[1] != "unit" for x in k)), ("es", "Pesa 5 kms aquí.", lambda k: k == [("5", "int")]),
    ("fr", "Il a 2 l'ordinateur.", lambda k: all(x[1] != "unit" for x in k)),
    ("es", "Cuesta 20 km/ día.", lambda k: all(x[1] != "unit" for x in k)),
    # fractions: not dates, not units, not 3-part slashes
    ("es", "El 12/05/2020 fue.", lambda k: all(x[1] != "frac" for x in k)),
    ("es", "Va a 3 km/h hoy.", lambda k: [x[1] for x in k] == ["unit"]),
    ("es", "Son 1/2/3 casos.", lambda k: all(x[1] == "skip" or x[1] != "frac" for x in k)),
    # dates need a real day and month; two-digit years with dots are versions
    ("es", "Fue el 32/05/2020 ya.", lambda k: all(x[1] != "date" for x in k)),
    ("es", "Fue el 12/13/2020 ya.", lambda k: all(x[1] != "date" for x in k)),
    ("es", "Fue el 12/05/1850 ya.", lambda k: all(x[1] != "date" for x in k)),
    ("nl", "Zie versie 4.2.", lambda k: all(x[1] != "date" for x in k)),
    ("de", "Es ist 1.2.20 hier.", lambda k: all(x[1] != "date" for x in k)),
    ("es", "Son 1.000.000 casos.", lambda k: k == [("1.000.000", "int")]),
    # decades need the context word: a count followed by a number is not a decade
    ("es", "Tiene 5 años 90 casas.", lambda k: all(x[1] != "decade" for x in k)),
    ("es", "Hay 90 casos.", lambda k: all(x[1] != "decade" for x in k)),
    ("de", "Es sind 90 Euro.", lambda k: all(x[1] != "decade" for x in k)),
    # roman numerals: strong context only, II..XXX, uppercase, single V / X only after a strong word
    ("es", "Malcolm X habló.", lambda k: k == []), ("es", "Luis V habló.", lambda k: k == []),
    ("es", "El siglo xxi es largo.", lambda k: k == []), ("es", "Dije I y luego II.", lambda k: k == []),
    ("es", "Vitamina C es buena.", lambda k: k == []), ("es", "El siglo XL es lejos.", lambda k: k == []),
    ("es", "El siglo X fue largo.", lambda k: [x[1] for x in k] == ["roman"]),
    # codes: 4-digit tails and letter groups after digits are not codes; lowercase short codes are left alone
    ("es", "Es el H1N1 aquí.", lambda k: k == []), ("es", "Es un mp3 aquí.", lambda k: k == []),
    ("es", "Es 20M de euros.", lambda k: all(x[1] != "alnum" for x in k)),
    # sign is not a range or a glued code
    ("es", "Son 10-5 casos.", lambda k: [x[1] for x in k] == ["range", "range"]),
    ("es", "Son 5 - 3 casos.", lambda k: all(x[1] == "int" for x in k)),
    # "&" in URLs between digits is untouched
    ("es", "Ve a x?a=1&b=2 ya.", lambda k: all(x[1] != "amp" for x in k)),
    # a currency next to a range is left alone (the digits stay unspanned)
    ("es", "Cuesta 10-12 € hoy.", lambda k: all(x[1] == "skip" for x in k)),
]
# spans that absorb a written hour word / abbreviation: (lang, text, span text, kind, reading that must be a candidate)
ABBR = [
    ("de", "Um 22:08 Uhr beginnt es.", "22:08 Uhr", "clock", "zweiundzwanzig Uhr acht"),
    ("de", "Gegen 9.30 Uhr geht es los.", "9.30 Uhr", "clock", "neun Uhr dreißig"),
    ("nl", "Om 11:20 uur begint het.", "11:20 uur", "clock", "elf uur twintig"),
    ("es", "Son las 11:00 h ya.", "11:00 h", "clock", "once horas"),
    ("es", "Llegó a las 10:30 a. m. ayer.", "10:30 a. m.", "clock", "diez y media de la mañana"),
    ("pt", "Chegou às 3:15 p.m. ontem.", "3:15 p.m.", "clock", "três e quinze da tarde"),
    ("nl", "Van 18.00 tot 20.00 uur open.", "18.00", "clock", "achttien uur"),
    ("nl", "Van 18.00 tot 20.00 uur open.", "20.00 uur", "clock", "twintig uur"),
    ("de", "Das kostet 250 Mrd. Euro.", "250 Mrd.", "int", "zweihundertfünfzig Milliarden"),
    ("de", "Das sind 5 Mio. EUR heute.", "5 Mio. EUR", "int", "fünf Millionen Euro"),
    ("de", "Das ist 1 Mio. € wert.", "1 Mio. €", "int", "eine Million Euro"),
    ("es", "Son 5 Mio. EUR hoy.", "5 Mio. EUR", "int", "cinco millones de euros"),
    ("fr", "C'est 5 M€ cette année.", "5 M€", "int", "cinq millions d'euros"),
    ("it", "Sono 2 mln di persone.", "2 mln", "int", "due milioni"),
    ("de", "Das ist Nr. 11 hier.", "Nr. 11", "int", "Nummer elf"),
    ("de", "Das sind ca. 3000 Leute.", "ca. 3000", "int", "circa dreitausend"),
    ("it", "Vedi il n. 11 qui.", "n. 11", "int", "numero undici"),
    ("pt", "Pela Lei nº 13 de hoje.", "nº 13", "int", "número treze"),
    ("nl", "Het kost 15 euro,- hier.", "15 euro,-", "int", "vijftien euro"),
    ("de", "Das ist 3 x 4 Meter.", "x 4", "int", "mal vier"),
    ("es", "Es 3 x 4 metros.", "x 4", "int", "por cuatro"),
    ("de", "Das sind 100 USD heute.", "100 USD", "cur", "einhundert Dollar"),
    ("it", "Arrivò il 1° posto.", "1°", "ord", "primo"),
    ("es", "Quedó en 2º lugar.", "2º", "ord", "segundo"),
    ("nl", "In 1839 begon het.", "1839", "int", "achttien negenendertig"),
    ("nl", "In 1469 begon het.", "1469", "int", "veertien negenenzestig"),
    ("nl", "In 1905 begon het.", "1905", "int", "negentien nul vijf"),
]
# the written abbreviation must not survive next to the span, nor a degree sign become an ordinal
ABBR_NEG = [
    ("es", "Hacen 20 °C hoy.", lambda k: [x[1] for x in k] == ["unit"]),
    ("it", "Ci sono 20° C oggi.", lambda k: all(x[1] != "ord" for x in k)),
    ("it", "Il 3° grado è quello.", lambda k: all(x[1] != "ord" for x in k)),
    ("de", "Am 10:30 am Montag.", lambda k: k and k[0][0] == "10:30"),
    ("de", "Er kam um 10:30 Uhr.", lambda k: k[0][0] == "10:30 Uhr"),
    ("de", "Es gibt 3x4 Optionen.", lambda k: all(not x[0].startswith("x") for x in k)),
]
for _l, _t, *_ in ABBR:
    TEXTS.setdefault(_l, []).append(_t)
for _l, _t, _ in ABBR_NEG:
    TEXTS.setdefault(_l, []).append(_t)
for _l, _t, *_ in POS:
    TEXTS.setdefault(_l, []).append(_t)
for _l, _t, _ in NEG:
    TEXTS.setdefault(_l, []).append(_t)

FORBIDDEN = set("0123456789$€£¥¢₹₩₽%#/+°⁇²³¹½¼¾⅓⅔&ºª‰")
ABBR_WORD = re.compile(r"[A-Za-zÀ-ÿ]\.")  # a letter abbreviation with its period


def _no_forbidden() -> tuple[bool, object]:
    """No candidate of any span of any test text contains a digit or a character the label must not keep."""
    bad = []
    for lang, ts in TEXTS.items():
        for t in ts:
            for s in M.spans(t, lang):
                for c in s.cands:
                    if FORBIDDEN & set(c) or ABBR_WORD.search(c):
                        bad.append((lang, t, c))
    return not bad, bad[:5]


def _span_has(lang, text, span_text, kind, reading):
    hit = [x for x in M.spans(text, lang) if x.text == span_text]
    ok = bool(hit) and hit[0].kind == kind and reading in [c.strip() for c in hit[0].cands]
    return ok, (text, [(x.text, x.kind, x.cands[:6], x.note) for x in M.spans(text, lang)])


def _all_have_cands() -> tuple[bool, object]:
    """Every span of every POS text has candidates (the table holds every key `needed` asked for)."""
    bad = [(l, t, s.text, s.note) for l, t, *_ in POS for s in M.spans(t, l) if s.kind != "skip" and not s.cands]
    return not bad, bad[:5]


def _needed_keys() -> tuple[bool, object]:
    ks = M.needed(["Son 2,5 kg hoy."], "es") | M.needed(["Am 12.05.2020 war es."], "de") | M.needed(["Luis XIV"], "es")
    want = {"es|dec|2|5||0|0", "de|ord|12|||0|0", "de|year|2020|||0|0", "es|ord|14|||0|0", "es|card|14|||0|0"}
    return want <= ks, want - ks


CHECKS += [
    *[(lambda l=l, t=t, i=i, r=r: has(t, l, i, r)) for l, t, i, r in POS],
    *[(lambda l=l, t=t, p=p: (p(_kinds(t, l)), (t, _kinds(t, l)))) for l, t, p in NEG],
    *[(lambda a=a: _span_has(*a)) for a in ABBR],
    *[(lambda l=l, t=t, p=p: (bool(p(_kinds(t, l))), (t, _kinds(t, l)))) for l, t, p in ABBR_NEG],
    _no_forbidden, _all_have_cands, _needed_keys,
    _hour_word_agrees,
]


def run() -> int:
    bad = 0
    for i, c in enumerate(CHECKS):
        try:
            ok, info = c()
        except Exception as e:  # noqa: BLE001
            ok, info = False, repr(e)
        if not ok:
            bad += 1
            print(f"FAIL #{i}: {info}")
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} pass")
    return bad


if __name__ == "__main__":
    if "--build" in sys.argv:  # extend the table with every key these texts need (private grammar dumper)
        d = sys.argv[sys.argv.index("--build") + 1]
        print("added", M.build_table(TEXTS, f"{d}/dump", d))
    sys.exit(1 if run() else 0)
