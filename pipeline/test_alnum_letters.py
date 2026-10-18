"""Letter+digit label policy (2026-10-18): letters as written + the number as spoken words.

python3 test_alnum_letters.py   (no GPU; red on the base: make_row wrote the letter names)
"""
import ml_cands as M
import ml_p3_core as C


def fake(cap, lang, picks_text, greedy=None):
    sp = M.spans(cap, lang)
    ds = []
    for s, want in zip(sp, picks_text):
        assert want in s.cands, (cap, want, s.cands)
        ds.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": s.cands.index(want), "margin": 9.0,
                   "fit": True, "decided": True})
    return {"spans": ds, "greedy": greedy if greedy is not None else M.render(cap, sp, [d["pick"] for d in ds]).lower()}


def label(cap, lang, picks):
    row = {"audio_filepath": "x.wav", "duration": 3.0, "text": cap, "lang": lang, "src": "vox", "id": "a",
           "origin": "new"}
    r = C.make_row(row, fake(cap, lang, picks))
    assert "row" in r, (cap, r)
    return r["row"]


# make_row (the labeller for new data): the CTC decided a letter-name reading, the label keeps the letters
r = label("Die G20 trifft sich.", "de", ["ge zwanzig"])
assert r["text"] == "Die G zwanzig trifft sich.", r["text"]
assert r["spans"][0]["reading"] == "G zwanzig" and r["text"][slice(*r["num_chars"][0])] == "G zwanzig"
# both spoken German spellings of C give one label (the CO2 variants unify)
assert label("Weniger CO2 bitte.", "de", ["ce o zwei"])["text"] == "Weniger CO zwei bitte."
assert label("Weniger CO2 bitte.", "de", ["ze o zwei"])["text"] == "Weniger CO zwei bitte."
# Dutch se / cee
assert label("Minder CO2 graag.", "nl", ["se o twee"])["text"] == "Minder CO twee graag."
assert label("Minder CO2 graag.", "nl", ["cee o twee"])["text"] == "Minder CO twee graag."
# number before the letter ("5G"), a sentence-initial capitalised reading, a later span's offsets shift
r = label("Netz mit 5G und 20 Masten.", "de", ["fünf ge", "zwanzig"])
assert r["text"] == "Netz mit fünf G und zwanzig Masten.", r["text"]
assert [r["text"][a:b] for a, b in r["num_chars"]] == ["fünf G", "zwanzig"]
assert label("Q4 war gut.", "de", ["Ku vier"] if "Ku vier" in M.spans("Q4 war gut.", "de")[0].cands
             else ["ku vier"])["text"] == "Q vier war gut."
# es / fr / it / pt
assert label("El G20 se reúne.", "es", ["ge veinte"])["text"] == "El G veinte se reúne."
assert label("Le G7 a parlé.", "fr", ["gé sept"])["text"] == "Le G sept a parlé."
assert label("Il G7 si riunisce.", "it", ["gi sette"])["text"] == "Il G sette si riunisce."
assert label("O G20 reuniu.", "pt", ["gê vinte"])["text"] == "O G vinte reuniu."

# alnum_split: an acronym said as a word keeps its letters; a reading not starting with the letters stays None
assert C.alnum_split("COP ventuno", "COP21", "it") == ("COP ventuno", 1, True)
assert C.alnum_split("ci o pi ventuno", "COP21", "it") == ("COP ventuno", 3, True)
assert C.alnum_split("zwanzig", "G20", "de") is None
assert C.alnum_split("ge zwanzig", "Corona-19", "de") is None  # not a letter+digit code
assert C.alnum_split("ge zwanzig", "G20", "en") is None  # English out of scope

# token map (for re-labelling rows that carry word times / window token lists)
lab, offs, sp, tm = C.apply_alnum_letters("weniger ce o zwei bitte", [[8, 17]],
                                          [{"text": "CO2", "kind": "alnum", "reading": "ce o zwei"}], "de")
assert lab == "weniger CO zwei bitte" and offs == [[8, 15]] and tm == [0, 1, 1, 2, 3], (lab, offs, tm)
lab, offs, sp, tm = C.apply_alnum_letters("(ge zwanzig) Gipfel", [[1, 11]],
                                          [{"text": "G20", "kind": "alnum", "reading": "ge zwanzig"}], "de")
assert lab == "(G zwanzig) Gipfel" and tm == [0, 1, 2], (lab, tm)
assert C.apply_alnum_letters("drei Leute", [[0, 4]], [{"text": "3", "kind": "int", "reading": "drei"}], "de")[3] is None

# re-labelling stored rows: word times merge, windows follow their parent's token map
import alnum_relabel as R  # noqa: E402

par = {"id": "p", "lang": "de", "text": "weniger ce o zwei bitte", "num_chars": [[8, 17]],
       "spans": [{"text": "CO2", "kind": "alnum", "reading": "ce o zwei"}],
       "word_times": [[0.0, 0.4], [0.5, 0.6], [0.6, 0.7], [0.7, 1.0], [1.1, 1.4]]}
nr, tm = R.relabel_row(par, "de")
assert nr["text"] == "weniger CO zwei bitte" and nr["word_times"] == [[0.0, 0.4], [0.5, 0.7], [0.7, 1.0], [1.1, 1.4]]
parents = {"p": (tm, nr["text"].split(), par["text"].split())}
win = {"id": "w", "text": "x ce o zwei bitte", "src": [["q", 3], ["p", 1], ["p", 2], ["p", 3], ["p", 4]],
       "num_tokens": [False, True, True, True, False]}
nw, changed = R.relabel_tokens(win, parents)
assert changed and nw["text"] == "x CO zwei bitte", nw
assert nw["src"] == [["q", 3], ["p", 1], ["p", 2], ["p", 3]] and nw["num_tokens"] == [False, True, True, False]
# a window that starts inside the code's letter names is left as it is
assert R.relabel_tokens({"id": "w2", "text": "o zwei bitte", "src": [["p", 2], ["p", 3], ["p", 4]]}, parents)[1] is False
assert R.relabel_row(dict(par, synthetic=True), "de")[1] is None  # synthetic rows: the fill's own labels
print("ok")
