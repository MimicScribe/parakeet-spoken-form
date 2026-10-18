"""compound_split: de / nl / it compound number words spelled as separate morphemes (training-text convention);
non-number look-alikes untouched; idempotent; joining the morphemes gives the token back; row / pair offsets follow."""
import inspect

import compound_split as C
import mix_build

SPLIT = {
    "de": {"zweihundertsiebzig": "zwei hundert siebzig", "vierundzwanzig": "vier und zwanzig",
           "neunzehnhundertneunzig": "neunzehn hundert neunzig", "zweitausenddreihundert": "zwei tausend drei hundert",
           "einhundertfünfundzwanzigtausendfünfhundert": "ein hundert fünf und zwanzig tausend fünf hundert",
           "dreiundzwanzigste": "drei und zwanzigste", "zweitausendachtundzwanzig": "zwei tausend acht und zwanzig",
           "hunderteins": "hundert eins", "tausendundeins": "tausend und eins", "dreieinhalb": "drei einhalb",
           "neunzehnhundertneunziger": "neunzehn hundert neunziger", "Zweihundert,": "Zwei hundert,",
           "zweihundertsiebzigsten.": "zwei hundert siebzigsten."},
    "nl": {"vierentwintig": "vier en twintig", "tweehonderdvijftig": "twee honderd vijftig",
           "negentienhonderdnegentig": "negentien honderd negentig", "drieduizend": "drie duizend",
           "tweeëntwintig": "twee en twintig", "eenentwintigste": "een en twintigste",
           "tweeduizendnegentien": "twee duizend negentien", "tweeënhalf": "twee en half"},
    "it": {"duemilaventitré": "due mila venti tré", "centoventicinque": "cento venti cinque",
           "millenovecentonovanta": "mille nove cento novanta", "duemila": "due mila", "centouno": "cento uno",
           "milleottocentottanta": "mille otto centottanta", "novecentottantanove": "nove centottanta nove",
           "ventunomila": "ventuno mila", "duemillesimo": "due millesimo", "dell'ottantatré": "dell'ottanta tré",
           "trentaquattromila": "trenta quattro mila"},
}
# never split: one-morpheme number words (teens, tens, ordinals, elided it forms) and non-number look-alikes
KEEP = {
    "de": ["dreizehn", "siebzig", "zwanzig", "zwanzigsten", "einem", "eins", "vierte", "einverstanden", "Dreieck",
           "Achtung", "Einheit", "Zehntausende", "zweieinhalbjährig", "Hunderter", "einer", "Elfmeter", "zwei-hundert"],
    "nl": ["tweede", "eenheid", "negentien", "honderden", "tweeën", "elfen", "achten", "vieren", "anderhalf",
           "Driehoek", "achter"],
    "it": ["ventuno", "trentotto", "centottanta", "ventunesimo", "trenta", "centro", "milano", "novella", "settembre",
           "ottobre", "tredici", "centinaia", "sei", "mille"],
}

for lang, cases in SPLIT.items():
    for tok, want in cases.items():
        got = C.split_token(tok, lang)
        assert " ".join(got) == want, (lang, tok, got)
        assert C.join_morphemes(got, lang) == tok, (lang, tok, got)  # round trip
        assert all(C.split_token(p, lang) == [p] for p in got), (lang, tok, got)  # idempotent
        assert C.split_text(" ".join(got), lang)[0] == want
for lang, toks in KEEP.items():
    for tok in toks:
        assert C.split_token(tok, lang) == [tok], (lang, tok, C.split_token(tok, lang))
# another language's rules never apply (es / fr / ... untouched)
assert C.split_token("veinticinco", "es") == ["veinticinco"]

# a P3 row: text, reading offsets, word times, span readings follow; caption / canonical stay
txt = "Es waren zweihundertsiebzig Leute, am vierundzwanzigsten Mai."
a, b = txt.index("zweihundertsiebzig"), txt.index("zweihundertsiebzig") + len("zweihundertsiebzig")
c, d = txt.index("vierundzwanzigsten"), txt.index("vierundzwanzigsten") + len("vierundzwanzigsten")
row = {"text": txt, "caption": "Es waren 270 Leute, am 24. Mai.", "num_chars": [[a, b], [c, d]],
       "word_times": [[0, 1], [1, 2], [2, 5], [5, 6], [6, 7], [7, 10], [10, 11]],
       "spans": [{"text": "270", "reading": "zweihundertsiebzig", "canonical": "zweihundertsiebzig"},
                 {"text": "24.", "reading": "vierundzwanzigsten", "canonical": "vierundzwanzigsten"}]}
out = C.split_row(row, "de")
assert out["text"] == "Es waren zwei hundert siebzig Leute, am vier und zwanzigsten Mai.", out["text"]
assert [out["text"][x:y] for x, y in out["num_chars"]] == ["zwei hundert siebzig", "vier und zwanzigsten"]
assert [sp["reading"] for sp in out["spans"]] == ["zwei hundert siebzig", "vier und zwanzigsten"]
assert [sp["canonical"] for sp in out["spans"]] == ["zweihundertsiebzig", "vierundzwanzigsten"]
assert out["caption"] == row["caption"] and row["text"] == txt  # the input row is not modified
assert len(out["word_times"]) == len(out["text"].split()) and out["word_times"][2][0] == 2 \
    and out["word_times"][4][1] == 5 and out["word_times"][3][0] == out["word_times"][2][1]
assert C.split_row(out, "de") == out  # idempotent on rows
assert C.split_row({"text": "a vierundzwanzig", "word_times": [[0, 1], None]}, "de")["word_times"] == [[0, 1], None, None, None]
# a mid-token span start (a cut word) maps to the same character
r2 = C.split_row({"text": "x zweihundertsiebzig", "num_chars": [[6, 20]]}, "de")
assert r2["text"][slice(*r2["num_chars"][0])] == "hundert siebzig", r2

# a window: per-token flags and src repeat for each morpheme; seam readings split
win = {"text": "um neunzehnhundertneunzig dann", "num_tokens": [False, True, False],
       "src": [["p", 4], ["p", 5], ["p", 6]], "reading": "neunzehnhundertneunzig", "reading_part": "neunzehnhundert"}
w = C.split_row(win, "de")
assert w["num_tokens"] == [False, True, True, True, False] and w["src"][1:4] == [["p", 5]] * 3
assert w["reading"] == "neunzehn hundert neunzig" and w["reading_part"] == "neunzehn hundert"
# a row without any compound is returned as is
plain = {"text": "zwanzig Leute", "num_tokens": [True, False]}
assert C.split_row(plain, "de") is plain

# a consistency pair: both token spans still cover the (split) reading, as ft_train.load_pairs checks
pair = {"a": {"audio_filepath": "a", "text": "Punkt zweiundsiebzig und zweiundvierzig."}, "a_span": [3, 4],
        "b": {"audio_filepath": "b", "text": "zweiundvierzig."}, "b_span": [0, 1], "reading": "zweiundvierzig."}
p = C.split_pair(pair, "de")
for side in ("a", "b"):
    i0, i1 = p[side + "_span"]
    assert p[side]["text"].split()[i0:i1] == p["reading"].split(), (side, p)
assert p["a_span"] == [5, 8] and p["reading"] == "zwei und vierzig."

rows, st = C.split_rows([row, plain], "de")
assert st == {"rows": 2, "rows_changed": 1, "tokens_split": 2}, st

# the build flag reaches build4
assert "split_langs" in inspect.signature(mix_build.build4).parameters
print("compound split ok")
