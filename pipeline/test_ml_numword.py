"""Local tests for ml_numword (no audio, no Modal): python3 test_ml_numword.py"""
import ml_numword as W


def got(text, lang):
    r = W.analyse(text, lang)
    return r["ok"], [(text[n.a:n.b], str(n.value), n.cls) for n in r["spans"] if not n.weak], r["classes"], r["why"]


# ---- 1. compose: scale words, place rule, splits ------------------------------------------------------------------
assert W.compose([19, 100, 80]) == [1980]
assert W.compose([2, 1000, 20]) == [2020]
assert W.compose([1000, 900, 84]) == [1984]
assert W.compose([3, 10**6, 300, 1000]) == [3300000]
assert W.compose([2, 3]) == [2, 3]  # two numbers said in a row
assert W.compose([20, 20]) == [20, 20]
assert W.compose([100, 1]) == [101]
assert W.split_compound("de", "neunzehnhundertachtzig") == [19, 100, 80]
assert W.split_compound("de", "dreieck") is None and W.split_compound("it", "settembre") is None
assert W.split_compound("nl", "achtergrond") is None and W.split_compound("de", "zweifel") is None

# ---- 2. classes and spans per language ----------------------------------------------------------------------------
ok, sp, cls, _ = got("Um halb drei treffen wir uns im Jahr neunzehnhundertachtzig, am zwölften Mai.", "de")
assert ok and sp == [("halb drei", "3", "clock"), ("neunzehnhundertachtzig", "1980", "year"), ("zwölften", "12", "date")]
ok, sp, cls, _ = got("Das kostet fünf Euro, zwanzig Prozent mehr, um drei Uhr dreißig.", "de")
assert sp == [("fünf Euro", "5", "cur"), ("zwanzig Prozent", "20", "pct"), ("drei Uhr dreißig", "3", "clock")]
ok, sp, cls, _ = got("A las tres y media, cinco millones de euros, en mil novecientos ochenta y cuatro.", "es")
assert sp == [("tres y media", "3", "clock"), ("cinco millones de euros", "5000000", "cur"),
              ("mil novecientos ochenta y cuatro", "1984", "year")], sp
ok, sp, cls, _ = got("Le dix-huit juin mil neuf cent quarante, à huit heures et demie, deux virgule cinq kilos.", "fr")
assert sp == [("dix-huit", "18", "date"), ("mil neuf cent quarante", "1940", "year"),
              ("huit heures et demie", "8", "clock"), ("deux virgule cinq", "2.5", "dec")], sp
ok, sp, cls, _ = got("Alle otto e mezza, il trenta per cento, nel millenovecentottanta, il ventitreesimo giorno.", "it")
assert sp == [("otto e mezza", "8", "clock"), ("trenta per cento", "30", "pct"), ("millenovecentottanta", "1980", "year"),
              ("ventitreesimo", "23", "ord")], sp
ok, sp, cls, _ = got("Om drie uur, drieëntwintig appels, de tweede keer, in tweeduizendtwintig.", "nl")
assert sp == [("drie uur", "3", "clock"), ("drieëntwintig", "23", "int"), ("tweede", "2", "ord"),
              ("tweeduizendtwintig", "2020", "year")], sp
ok, sp, cls, _ = got("Às duas e meia, mil e quinhentos reais, o terceiro dia.", "pt")
assert sp == [("duas e meia", "2", "clock"), ("mil e quinhentos reais", "1500", "cur"), ("terceiro", "3", "ord")], sp

# ---- 3. look-alikes: articles, idioms, words that are numbers only sometimes ------------------------------------
assert got("Ich habe ein Haus und eine Katze.", "de")[0] is False  # articles
assert got("Un jour, je suis parti.", "fr")[0] is False  # "un jour" = "one day" / "someday": left out
assert got("Grazie mille a tutti.", "it")[0] is False  # bare scale word: an idiom
assert got("Secondo me è vero.", "it")[0] is False and got("Según el segundo informe.", "es")[0] is False
assert got("Ne abbiamo parlato prima.", "it")[0] is False  # "prima" = before
assert got("Tu sei bravo.", "it")[0] is False and got("Tu sei bravo.", "it")[3] == "no_number"
assert got("Ci sono sei anni di differenza.", "it")[1] == [("sei", "6", "unit")]
assert got("Sei andato via.", "it")[3] == "ambiguous"  # cannot tell: the caption is dropped
assert got("Un appartement tout neuf.", "fr")[0] is False
assert got("Das dürfen wir nicht außer Acht lassen.", "de")[0] is False
assert got("Dat moet je in acht nemen.", "nl")[0] is False
assert got("Es un cuarto pequeño.", "es")[3] == "ambiguous"
assert got("Wir haben 3 Kinder und zwei Hunde.", "de")[3] == "digits"
assert got("Zwei, drei Leute.", "de")[1] == [("Zwei", "2", "int"), ("drei", "3", "int")]  # punctuation splits

# ---- 4. template / select caps ------------------------------------------------------------------------------------
r = W.analyse("Wir kommen um drei Uhr.", "de")
assert W.template("Wir kommen um drei Uhr.", r["spans"]) == "wir kommen um #"


def cand(i, video, text, classes, years=(), values=("2",)):
    return {"id": f"r{i}", "video": video, "text": text, "classes": list(classes), "years": list(years),
            "values": list(values), "template": text.lower()}


cs = [cand(i, "v1", f"t{i}", ["int"]) for i in range(20)] + [cand(100 + i, f"w{i}", "same", ["int"]) for i in range(5)]
cs += [cand(200 + i, f"y{i}", f"year{i}", ["year"], years=[1980 if i < 6 else 2031 + i]) for i in range(10)]
pick = W.select(cs, 50, per_video=8, per_template=3, per_year=3, shares={"int": 0.5, "year": 0.5})
assert sum(c["video"] == "v1" for c in pick) <= 8
assert sum(c["text"] == "same" for c in pick) == 1  # one row per caption text
assert sum(1980 in c["years"] for c in pick) <= 3
years = [c for c in pick if c["why"] == "year"]
assert years and years[0]["years"][0] >= 2025  # forward years first

# ---- 5. make_row: label2 shape, no weak article spans ------------------------------------------------------------
row = W.make_row("Ein Mann kam um drei Uhr.", "de", "x1", "/vol/x.wav", 2.0)
assert row["text"] == row["caption"] and row["num_chars"] == [[16, 24]] and row["spans"][0]["kind"] == "clock"
assert row["spans"][0]["reading"] == "drei Uhr" and row["spans"][0]["by"] == "caption"
assert W.make_row("Ein Mann kam.", "de", "x2", "/vol/x.wav", 2.0) is None
# ---- 6. review of the first selection (2026-10-16) -----------------------------------------------------------------
assert got("Vous avez un livre entre les mains.", "fr")[0] is False  # a book, not a pound
assert got("Para dar um peso maior.", "pt")[0] is False
assert got("Ein Pfund Mehl.", "de")[0] is False
assert got("Nei confronti di terzi.", "it")[0] is False and got("Su due ottave.", "it")[1] == [("due", "2", "int")]
assert got("De segunda a sexta.", "pt")[3] == "ambiguous"
assert got("Dos mil veces presentido.", "es")[1] == [("Dos mil", "2000", "big")]
assert got("En dos mil fuimos.", "es")[1] == [("dos mil", "2000", "year")]
assert got("Wir haben fünf Pfund Dividende bekommen.", "de")[1] == [("fünf Pfund", "5", "cur")]
print("ok")

# French hyphenated 70-99 / "vingt-et-un" and digit-by-digit decimals keep their value
for t, l, want in [("quatre-vingt-dix personnes", "fr", "90"), ("soixante-dix-sept euros", "fr", "77"),
                   ("vingt-et-un jours", "fr", "21"), ("zwei Komma null fünf Prozent", "de", "2.05"),
                   ("drei Komma zwei fünf Meter", "de", "3.25")]:
    got = [str(n.value) for n in W.nums(t, l)]
    assert got == [want], (t, got)
print("hyphen / decimal digits ok")
