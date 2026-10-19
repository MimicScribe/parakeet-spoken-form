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
    "it": {"duemilaventitré": "due mila venti tre", "centoventicinque": "cento venti cinque",
           "millenovecentonovanta": "mille nove cento novanta", "duemila": "due mila", "centouno": "cento uno",
           "milleottocentottanta": "mille otto centottanta", "novecentottantanove": "nove centottanta nove",
           "ventunomila": "ventuno mila", "duemillesimo": "due millesimo", "dell'ottantatré": "dell'ottanta tre",
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

OLD_IT = {"it_elided": False}  # the form before 2026-10-18 (elided compounds whole); the default splits them too
for lang, cases in SPLIT.items():
    for tok, want in cases.items():
        got = C.split_token(tok, lang, **(OLD_IT if lang == "it" else {}))
        assert " ".join(got) == want, (lang, tok, got)
        assert C.join_morphemes(got, lang) == tok, (lang, tok, got)  # round trip
        assert all(C.split_token(p, lang, **(OLD_IT if lang == "it" else {})) == [p] for p in got), (lang, tok, got)
        assert C.split_text(" ".join(got), lang, **(OLD_IT if lang == "it" else {}))[0] == want
for lang, toks in KEEP.items():
    for tok in toks:
        assert C.split_token(tok, lang, **(OLD_IT if lang == "it" else {})) == [tok], (lang, tok)
# Italian with the elided forms split (IT_SPLIT_ELIDED / it_elided=True, 2026-10-18): the full morphemes, the elided
# vowel written back; the default above is unchanged
IT_ELIDED = {"ventuno": "venti uno", "trentotto": "trenta otto", "centottanta": "cento ottanta",
             "ventunesimo": "venti unesimo", "novantotto": "novanta otto", "centottantuno": "cento ottanta uno",
             "duecentottantotto": "due cento ottanta otto", "ventun": "venti un", "ventuna": "venti una",
             "Centottanta,": "Cento ottanta,", "l'ottantuno": "l'ottanta uno", "centottantesimo": "cento ottantesimo",
             "milleottocentottanta": "mille otto cento ottanta", "novecentottantanove": "nove cento ottanta nove",
             "ventunomila": "venti uno mila", "duemilaventitré": "due mila venti tre", "centouno": "cento uno"}
for tok, want in IT_ELIDED.items():
    got = C.split_token(tok, "it", it_elided=True)
    assert " ".join(got) == want, (tok, got)
    assert C.join_morphemes(got, "it") == tok, (tok, got)
    assert all(C.split_token(p, "it", it_elided=True) == [p] for p in got), (tok, got)
for tok, (want, back) in {"centuno": ("cento uno", "centouno"), "centotto": ("cento otto", "centootto"),
                          "centunomila": ("cento uno mila", "centounomila")}.items():
    got = C.split_token(tok, "it", it_elided=True)  # the elided hundred before uno / otto joins back unelided
    assert " ".join(got) == want and C.join_morphemes(got, "it") == back, (tok, got)
    assert C.split_token(back, "it", it_elided=True) == got
for tok in ("trenta", "centro", "ottanta", "otto", "uno", "una", "un", "Trento", "quarantena", "centrale", "venti"):
    assert C.split_token(tok, "it", it_elided=True) == [tok], tok
assert C.split_token("ventuno", "it") == ["venti", "uno"] and C.IT_SPLIT_ELIDED  # the default: always split
assert C.split_token("ventuno", "it", it_elided=False) == ["ventuno"]
assert C.split_text("costa ventuno euro", "it", it_elided=True)[0] == "costa venti uno euro"
r_it = {"text": "ne arrivarono ventotto", "num_chars": [[14, 22]], "spans": [{"reading": "ventotto"}]}
assert C.split_row(r_it, "it", it_elided=False) is r_it
assert C.split_row(r_it, "it", it_elided=True)["spans"][0]["reading"] == "venti otto"
assert C.split_rows([r_it], "it", it_elided=True)[1]["rows_changed"] == 1
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


def test_it_final_tre_loses_its_accent_when_split_and_regains_it_when_joined():
    # 2026-10-18: one spelling for one sound ("venti tre" as the standalone "tre"); the fused word keeps "tré"
    assert C.split_token("ventitré", "it") == ["venti", "tre"]
    assert C.split_token("Trentatrè,", "it") == ["Trenta", "tre,"]
    assert C.split_token("tre", "it") == ["tre"] and C.split_token("tremila", "it") == ["tre", "mila"]
    assert C.join_morphemes(["due", "mila", "venti", "tre"], "it") == "duemilaventitré"
    assert C.join_morphemes(["tre", "mila"], "it") == "tremila"

# ---- Italian restored-vowel form ("it+", also plain "it" while IT_SPLIT_ELIDED is on): num_chars offsets (2026-10-18)
import tempfile  # noqa: E402

for bad in (["it"], ["it+"], ["de", "nl", "it"]):
    try:
        C.check_split_langs(bad)
    except ValueError as e:
        assert "it-" in str(e) and "num_chars" in str(e), str(e)
    else:
        raise AssertionError(f"{bad} must be refused while the Italian offsets are misaligned")
C.check_split_langs(["de", "nl", "it-"])  # the whole-elided form is exact (red-on-base: control)
C.check_split_langs([])
with tempfile.TemporaryDirectory() as d:  # build4 refuses before it reads or writes anything
    try:
        mix_build.build4(d, ("it",), split_langs=("it",))
    except ValueError as e:
        assert "num_chars" in str(e)
    else:
        raise AssertionError("build4 must refuse the restored-vowel Italian split")
    assert not __import__("os").listdir(d), "refused before creating the mix dir"
# the bug itself, expected to fail until compound_split counts the restored vowel (then set IT_OFFSETS_FIXED = True)
_r = C.split_row({"text": "ne arrivarono ventotto", "num_chars": [[14, 22]]}, "it", it_elided=True)
_still_broken = _r["text"][slice(*_r["num_chars"][0])] != "venti otto"  # today: "venti ott"
assert _still_broken == (not C.IT_OFFSETS_FIXED), "IT_OFFSETS_FIXED must be True exactly when the offsets are exact"

# ---- Italian "it-" leftovers found by the single-convention assert (2026-10-18; red-on-base: both stayed fused)
# (1) "uno" after a hundred inside a mila multiplier: 301,700 "trecentounomilasettecento"
assert C.split_token("trecentounomilasettecento", "it", False) == ["tre", "cento", "uno", "mila", "sette", "cento"]
assert C.split_token("centounomila", "it", False) == ["cento", "uno", "mila"]
assert C.split_token("unomila", "it", False) == ["unomila"] and C.split_token("ventunomila", "it", False) == ["ventuno", "mila"]
# (2) an identifier suffix after the number word: directive "96/53/C" read "cinquantatré-C" (accent dropped, suffix kept)
assert C.split_token("cinquantatré-C,", "it", False) == ["cinquanta", "tre-C,"]
assert C.split_token("zweihundert-Euro", "de") == ["zweihundert-Euro"]  # only a one-capital suffix counts
new, tok_map, ins = C.split_text("direttiva novanta sei cinquantatré-C del", "it", False)
assert new == "direttiva novanta sei cinquanta tre-C del" and "tré" not in new
print("test_compound_split: it- fused leftovers ok")
