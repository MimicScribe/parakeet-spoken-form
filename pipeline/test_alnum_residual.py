"""Residual letter-name labels (alnum_residual.py): python3 test_alnum_residual.py   (no GPU; the module is new)."""
import alnum_residual as R

FIX = [  # (lang, label, expected)
    ("de", "Dann künstliche Intelligenz, fünf ge. Es gibt", "Dann künstliche Intelligenz, fünf G. Es gibt"),
    ("de", "Wir drücken ef neun und schauen", "Wir drücken F neun und schauen"),
    ("de", "Im ku zwei ist der Umsatz gestiegen.", "Im Q zwei ist der Umsatz gestiegen."),
    ("de", "die Rebellen der em drei und zwanzig keine", "die Rebellen der M drei und zwanzig keine"),
    ("es", "la reunión del ge siete, es dudoso", "la reunión del G siete, es dudoso"),
    ("es", "Primero, la ce uno. Faltan", "Primero, la C uno. Faltan"),
    ("es", "el despliegue del cinco ge, cruciales", "el despliegue del cinco G, cruciales"),
    ("fr", "dont le cé o deux. C'est un coût", "dont le CO deux. C'est un coût"),
    ("fr", "cette lecture cinq dé.", "cette lecture cinq D."),
    ("it", "è tornato ad essere un acca due, quindi", "è tornato ad essere un H due, quindi"),
    ("nl", "staat bee twee en vijftig als nieuwtje", "staat B twee en vijftig als nieuwtje"),
    ("nl", "naast se o twee ook nog allerlei", "naast CO twee ook nog allerlei"),
    ("nl", "(vijf gee) en internet", "(vijf G) en internet"),
    ("pt", "na seis dê Mark dois e na cinco dê Mark quatro", "na seis D Mark dois e na cinco D Mark quatro"),
    ("pt", "menos o gê oito, pegar", "menos o G oito, pegar"),
]
KEEP = [  # prose and acronyms that look like a letter name before a number
    ("es", "alguien ve quinientos"), ("es", "Ve y disfruta de las ofertas"), ("fr", "un dé sur une nouvelle"),
    ("nl", "niet per se vijf dagen per week"), ("pt", "Dê um tempo, apenas por vinte e quatro horas"),
    ("pt", "o carro dê cento e dez cavalos"), ("de", "bei der Fußball-EM zwei tausend zwölf"),
    ("de", "keine G zwanzig, es"), ("en", "ge zwanzig"),  # already the policy form / out of scope
    ("de", "ein ef und dann nichts"), ("fr", "il y a un dé"),
]
for lang, t, want in FIX:
    got = R.fix_text(t, lang)[0]
    assert got == want, (lang, t, got)
for lang, t in KEEP:
    assert R.fix_text(t, lang) == (t, None), (lang, t, R.fix_text(t, lang))

# token map: the letter names of one code merge into one token
new, tm, n = R.fix_tokens("im ce o zwei Wert".split(), "de")
assert new == ["im", "CO", "zwei", "Wert"] and tm == [0, 1, 1, 2, 3] and n == 1, (new, tm)

# a token-list row keeps its parallel lists in step
row = {"text": "der ce o zwei Wert", "num_tokens": [False, True, True, True, False], "src": [["p", 0], ["p", 1], ["p", 2], ["p", 3], ["p", 4]],
       "word_times": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}
nr, ch = R.relabel_row(row, "de")
assert ch and nr["text"] == "der CO zwei Wert" and nr["text_old"] == row["text"] and nr["relabel"] == ["alnumB"]
assert nr["num_tokens"] == [False, True, True, False] and nr["src"] == [["p", 0], ["p", 1], ["p", 3], ["p", 4]]
assert nr["word_times"] == [[0, 1], [1, 3], [3, 4], [4, 5]], nr["word_times"]
assert R.relabel_row({"text": "nichts zu tun"}, "de") == ({"text": "nichts zu tun"}, False)
assert R.relabel_row({"text": "ce o zwei", "spans": [{}], "num_chars": [[0, 9]]}, "de")[1] is False  # span rows: alnum_relabel.py

# a pair row: token spans are re-pointed
pair = {"a": {"text": "Drei plus neun ergibt zwölf."},
        "b": {"text": "Der ce o zwei tagt. Drei plus neun ergibt zwölf."}, "a_span": [0, 1], "b_span": [5, 6], "reading": "Drei"}
nr, ch = R.relabel_row(pair, "de")
assert ch and nr["b"]["text"] == "Der CO zwei tagt. Drei plus neun ergibt zwölf." and nr["b_span"] == [4, 5] and nr["a_span"] == [0, 1]
assert nr["b"]["text"].split()[nr["b_span"][0]] == "Drei"

# audit lists what is left
assert R.audit("im ce o zwei Wert", "de") == ["ce o"], R.audit("im ce o zwei Wert", "de")
assert R.audit("im CO zwei Wert", "de") == []
print("ok")
