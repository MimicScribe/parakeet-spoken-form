"""Parliament-speech labelling fixes (2026-10-16): candidates, decisions and row edges for European Parliament speech
with partial pseudo-label captions and an external verbatim transcript. Local, no GPU: python3 test_ml_parl.py

Each block also runs with the fix switched off and asserts the OLD result, so the test is red on the old code."""
import numpy as np

import ml_cands as M
import ml_decide as D
import ml_p3_core as C


def cands(cap, lang, text):
    return [s.cands for s in M.spans(cap, lang) if s.text == text][0]


def without(mod, name, fn):
    attr = "CAND_FIXES" if mod is M else "FIXES"
    old = set(getattr(mod, attr))
    getattr(mod, attr).discard(name)
    try:
        return fn()
    finally:
        getattr(mod, attr).clear()
        getattr(mod, attr).update(old)


# ---------------------------------------------------------------- candidates, every language
def has(cap, lang, text, want):
    return want in cands(cap, lang, text)


RANGE_BARE = [("Il periodo 2021-2027 è lungo.", "it", "-2027", " duemilaventisette"),
              ("El periodo 2021-2027 es largo.", "es", "-2027", " dos mil veintisiete"),
              ("La période 2021-2027 est longue.", "fr", "-2027", " deux mille vingt sept"),
              ("Der Zeitraum 2021-2027 ist lang.", "de", "-2027", " zweitausendsiebenundzwanzig"),
              ("De periode 2021-2027 is lang.", "nl", "-2027", " tweeduizendzevenentwintig"),
              ("O período 2021-2027 é longo.", "pt", "-2027", " dois mil e vinte e sete")]
for cap, lang, t, want in RANGE_BARE:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "range_bare", lambda: has(cap, lang, t, want)), lang
RANGE_OR = [("Ci vogliono 1-2 milioni.", "it", "-2", " o due"), ("Hacen falta 1-2 millones.", "es", "-2", " o dos"),
            ("Il faut 1-2 millions.", "fr", "-2", " ou deux"), ("Es braucht 1-2 Millionen.", "de", "-2", " oder zwei"),
            ("Er is 1-2 miljoen nodig.", "nl", "-2", " of twee"), ("São precisos 1-2 milhões.", "pt", "-2", " ou dois")]
for cap, lang, t, want in RANGE_OR:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "range_or", lambda: has(cap, lang, t, want)), lang
ACR = [("Alla COP21 di Parigi.", "it", "COP21", "COP ventuno"), ("En la COP21 de París.", "es", "COP21", "COP veintiuno"),
       ("À la COP21 de Paris.", "fr", "COP21", "COP vingt et un"), ("Auf der COP21 in Paris.", "de", "COP21", "COP einundzwanzig"),
       ("Op de COP21 in Parijs.", "nl", "COP21", "COP eenentwintig"), ("Na COP21 de Paris.", "pt", "COP21", "COP vinte e um")]
for cap, lang, t, want in ACR:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "acr_word", lambda: has(cap, lang, t, want)), lang
assert not any(c.startswith("FP ") for c in cands("Il programma FP7 è finito.", "it", "FP7"))  # no vowel: letters only
CLOCK = [("Ci vediamo alle 10.30 in aula.", "it", "10.30", "dieci e trenta"),
         ("Nos vemos a las 10.30 en la sala.", "es", "10.30", "diez y treinta"),
         ("On se voit à 10.30 dans la salle.", "fr", "10.30", "dix heures trente"),
         ("Wir sehen uns um 10.30 im Saal.", "de", "10.30", "zehn Uhr dreißig"),
         ("We zien elkaar om 10.30 in de zaal.", "nl", "10.30", "tien uur dertig"),
         ("Vemo-nos às 10.30 na sala.", "pt", "10.30", "dez e trinta")]
for cap, lang, t, want in CLOCK:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "dot_clock", lambda: has(cap, lang, t, want)), lang
assert not any("ore" in c for c in cands("Il valore è 3.75 qui.", "it", "3.75"))  # minutes > 59: no time reading
assert has("Secondo l'articolo 5.2 del trattato.", "it", "5.2", "cinque due")
assert has("Según el artículo 5.2 del tratado.", "es", "5.2", "cinco dos")
assert has("Nach Artikel 47.1 des Vertrags.", "de", "47.1", "siebenundvierzig eins")
assert not without(M, "dot_bare", lambda: has("Secondo l'articolo 5.2 del trattato.", "it", "5.2", "cinque due"))
assert not has("Rund 3,5 % mehr.", "de", "3,5 %", "drei fünf")  # a comma decimal gets no bare reading
assert has("La regola 75,3 delle proposte.", "it", "75,3", "settantacinque comma tre")
assert not without(M, "it_comma", lambda: has("La regola 75,3 delle proposte.", "it", "75,3", "settantacinque comma tre"))
assert not any("comma" in c for c in cands("La regla 75,3 de las propuestas.", "es", "75,3"))
PAIRS = [("Emendamento 5707 approvato.", "it", "5707", "cinquantasette zero sette"),
         ("Enmienda 5707 aprobada.", "es", "5707", "cincuenta y siete cero siete"),
         ("Änderungsantrag 4762 angenommen.", "de", "4762", "siebenundvierzig zweiundsechzig")]
for cap, lang, t, want in PAIRS:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "int_pairs", lambda: has(cap, lang, t, want)), lang
# int_pairs leaves years alone; year_spoken (with a transcript only) adds the two-pair and two-digit readings of a year
assert without(M, "year_spoken", lambda: not any(c == "venti dodici" for c in cands("Nel 2012 era diverso.", "it", "2012")))
assert has("Nel 2012 era diverso.", "it", "2012", "venti dodici")
assert has("Seit 1991 nicht.", "de", "1991", "einundneunzig") and has("Bis 2060 also.", "de", "2060", "zwanzig sechzig")
assert has("Desde 1904 hay.", "es", "1904", "diecinueve cero cuatro")
assert cands("Seit 1991 nicht.", "de", "1991")[0] == "neunzehnhunderteinundneunzig"  # the full reading stays first
assert not any(c == "fünf" for c in cands("Seit 2005 nicht.", "de", "2005"))  # no one-digit short year
print("candidates ok")


# ---------------------------------------------------------------- decisions (a fake CTC: nll = 3 x letter edits to the audio)
class FakeCTC:
    def __init__(self, lang, spoken, greedy):
        self.lang, self.spoken, self.g = lang, C.letters(spoken), greedy

    def greedy(self, lp):
        return self.g

    def nll(self, lp, text):
        return 3.0 * D._lev(C.letters(text), self.spoken)  # ~3 nats per letter the audio lacks or has extra


def label(cap, lang, spoken, greedy, ext):
    r = D.decide(FakeCTC(lang, spoken, greedy), cap, None, ext=ext)
    if not all(s["decided"] for s in r["spans"]):
        return None
    return M.render(cap, M.spans(cap, lang), [s["pick"] for s in r["spans"]])


# elision: the neighbour word is "dell'impatto", not "dell"
cap = "È pari al 22% dell'impatto finanziario."
sp = "è pari al ventidue per cento dell'impatto finanziario"
assert label(cap, "it", sp, sp, None) == "È pari al ventidue per cento dell'impatto finanziario.", label(cap, "it", sp, sp, None)
assert without(D, "elision", lambda: label(cap, "it", sp, sp, None)) is None
cap = "Il faut 22 % d'augmentation."
sp = "il faut vingt deux pour cent d'augmentation"
assert label(cap, "fr", sp, sp, None) is not None  # red-on-base: control (a 1-letter elision fit fuzzily before)
# letters: the transcript writes "co due", the CTC greedy garbles it
cap, sp = "Le emissioni di CO2 per lottare.", "le emissioni di ci o due per lottare"
g, ext = "le emissioni dicio dueper lotare", "le emissioni di co due per lottare"
assert label(cap, "it", sp, g, ext) == "Le emissioni di ci o due per lottare.", label(cap, "it", sp, g, ext)
assert without(D, "letters_ext", lambda: without(D, "ext_ctx", lambda: label(cap, "it", sp, g, ext))) is None
cap, sp = "Die Emissionen von CO2 steigen.", "die emissionen von ce o zwei steigen"
# the German C has two names ("ze", "ce"): the same letters, so the label keeps the grammar's first spelling
assert label(cap, "de", sp, "die emisionen von zeo zwai steigen", "die emissionen von c o zwei steigen") == \
    "Die Emissionen von ze o zwei steigen."
# a range said with no connector; base_iter + ext_ctx settle both ends together
cap = "Per il periodo 2021-2027 affinché si possa."
sp = "per il periodo duemilaventuno duemilaventisette affinché si possa"
g = "per il periodo duemila ventuno o duemila ventisette affinche si possa"
assert label(cap, "it", sp, g, sp) == "Per il periodo duemilaventuno duemilaventisette affinché si possa.", label(cap, "it", sp, g, sp)
assert without(M, "range_bare", lambda: label(cap, "it", sp, g, sp)) is None
# "1-2 milioni" said "uno o due milioni": the "or" reading, never the bare one (a lost word)
cap, sp = "Parliamo di 1-2 milioni di euro.", "parliamo di uno o due milioni di euro"
assert label(cap, "it", sp, "parliamo di uno du milioni di euro", sp) == "Parliamo di uno o due milioni di euro."
assert without(M, "range_or", lambda: label(cap, "it", sp, "parliamo di uno du milioni di euro", sp)) is None
# a dotted time
cap, sp = "Siate puntuali alle 10.30 per favore.", "siate puntuali alle dieci e trenta per favore"
assert label(cap, "it", sp, "siate puntuali alle diecietrenta per favore", sp) == "Siate puntuali alle dieci e trenta per favore."
# a word acronym the transcript writes as a word
cap, sp = "Raggiunto dalla COP21 è importante.", "raggiunto dalla cop ventuno è importante"
assert label(cap, "it", sp, "raggiunto dalla coppa ventuno e importante", sp) == "Raggiunto dalla COP ventuno è importante."
# SAFETY: the caption repeats a number the speaker said once ("70-70" spoken "settanta") -> never decided
cap, sp = "Abbiamo circa 70-70 partecipanti.", "abbiamo circa settanta partecipanti"
assert label(cap, "it", sp, "abbiamo circa settanta-sette zero partecipanti", sp) is None
# SAFETY: a caption that dropped "mil" stays undecided even when ext holds "ocho" elsewhere
cap, sp = "Son 8 puestos.", "son ocho mil puestos"
assert label(cap, "es", sp, "son ocho mil puestos", "son ocho mil puestos ocho") is None
# SAFETY: the transcript disagrees on the value -> undecided (no reading of "26" matches "ventisette")
cap, sp = "Ci metteranno magari 26-30 anni.", "ci metteranno magari ventisette trenta anni"
assert label(cap, "it", sp, "ci metteranno magari ventiseste trent'anni", sp) is None
print("decisions ok")


# ---------------------------------------------------------------- row edges by the transcript's word times
def speechy(dur, quiet):
    x = np.random.RandomState(0).randn(int(dur * 16000)).astype(np.float32) * 0.3
    for a, b in quiet:
        x[int(a * 16000):int(b * 16000)] *= 0.01
    return x


def words_at(text, t0=0.2, step=0.5, gap=0.15):
    out, t = [], t0
    for w in text.split():
        out.append({"w": w, "s": int(t * 1000), "e": int((t + step - gap) * 1000)})
        t += step
    return out


ext = "che dice l'onorevole ferber il pacchetto è un elemento chiave delle riforme e anche per rafforzare"
ws = words_at(ext)
x = speechy(ws[-1]["e"] / 1000 + 0.3, [(w["e"] / 1000, (ws[i + 1]["s"] if i + 1 < len(ws) else w["e"] + 300) / 1000)
                                     for i, w in enumerate(ws)])
cap = "Il pacchetto è un elemento chiave delle riforme."
tr = C.ext_trim(cap, ws, x)
assert tr and tr["lead"] == 4 and tr["trail"] == 4, tr
assert ws[3]["e"] / 1000 <= tr["a"] + C.TRIM_SLACK and tr["a"] - C.TRIM_SLACK <= ws[4]["s"] / 1000, tr
assert C.ext_words_in(ws, tr["a"], tr["b"])[0] == "il pacchetto è un elemento chiave delle riforme", tr
sh = C.shift_words(ws, tr["a"], tr["b"])
assert [w["w"] for w in sh][0] == "il" and sh[0]["s"] >= 0
# no extra words around the caption -> nothing to cut
assert C.ext_trim(ext, ws, x) is None
# a caption whose first word the transcript lacks -> the lead side stays
assert C.ext_trim("Ecco il pacchetto è un elemento chiave delle riforme.", ws, x)["lead"] == 0
# no quiet frame between the words -> no cut
assert C.ext_trim(cap, ws, speechy(len(x) / 16000, [])) is None
# a caption that ends with a number keeps its tail
assert C.ext_trim("Il pacchetto è un elemento chiave delle 3", ws, x)["trail"] == 0
print("ext_trim ok")

# a piece passes on the transcript words inside it when its own CTC decode is noisy
pws = words_at("il pacchetto vale venti milioni di euro")
ok, c, ce = C.piece_ok("il pacchetto vale venti milioni di euro", "ul pachetto valle vnti miglioni deuro", 0.15, pws, 0.0, 3.6, 3.6)
assert ok and ce == 0.0 and c > 0.15, (ok, c, ce)
assert not C.piece_ok("il pacchetto vale venti milioni di euro", "ul pachetto valle vnti miglioni deuro", 0.15, None, 0, 3.6, 3.6)[0]
# a transcript word cut through at the piece's end -> refused
assert not C.piece_ok("il pacchetto vale venti milioni di", "x", 0.15, pws, 0.0, pws[-1]["s"] / 1000 + 0.1, 3.6)[0]
print("piece_ok ok")

# salvage: make_row keeps an undecided span's mark, so its piece can be left out
row = {"audio_filepath": "x.wav", "duration": 30.0, "text": "Nel 2014 abbiamo 26 casi.", "lang": "it", "src": "vox", "id": "s",
       "origin": "new"}
spn = M.spans(row["text"], "it")
ds = [{"text": s.text, "kind": s.kind, "cands": s.cands, "pick": 0, "margin": 9.0, "fit": True, "decided": True} for s in spn]
ds[1] = dict(ds[1], salvage=True)
nr = C.make_row(row, {"spans": ds, "greedy": "nel duemilaquattordici abbiamo ventisei casi"})["row"]
assert [s.get("undecided", False) for s in nr["spans"]] == [False, True], nr["spans"]
print("salvage marks ok")

# SAFETY (ext_veto): the transcript holds "tre punto uno" where the CTC best is "tre uno" -> refused
cap = "Se si guarda l'articolo 3.1 per esempio."
sp, ext = "se si guarda l'articolo tre uno per esempio", "se si guarda l'articolo tre punto uno per esempio"
assert without(D, "ext_pick", lambda: label(cap, "it", sp, sp, ext)) is None
assert without(D, "ext_pick", lambda: without(D, "ext_veto", lambda: label(cap, "it", sp, sp, ext))) == \
    "Se si guarda l'articolo tre uno per esempio."
# ext_pick (wave-1): the transcript's reading only adds a whole word to the CTC best ("punto", which the xlsr CTC drops
# with 10-90 nats to spare on the parliament batch): the transcript decides
assert label(cap, "it", sp, sp, ext) == "Se si guarda l'articolo tre punto uno per esempio.", label(cap, "it", sp, sp, ext)
assert label(cap, "it", sp, sp, sp) == "Se si guarda l'articolo tre uno per esempio."  # both say "tre uno": kept
print("ext veto ok")

# without an external transcript decide() weighs the old readings only; make_row renders from the decision's readings
cap, sp = "Per il periodo 2021-2027 affinché si possa.", "per il periodo duemilaventuno duemilaventisette affinché si possa"
r = D.decide(FakeCTC("it", sp, sp), cap, None, ext=None)
assert " duemilaventisette" not in r["spans"][1]["cands"], r["spans"][1]["cands"]
r = D.decide(FakeCTC("it", sp, sp), cap, None, ext=sp)
assert " duemilaventisette" in r["spans"][1]["cands"]
row = {"audio_filepath": "x.wav", "duration": 5.0, "text": cap, "lang": "it", "src": "vox", "id": "q", "origin": "new"}
old_c = r["spans"][1]["cands"]
r["spans"][1] = dict(r["spans"][1], cands=[c for c in old_c if c != " a duemilaventisette"],
                     pick=[c for c in old_c if c != " a duemilaventisette"].index(" duemilaventisette"))
assert C.make_row(row, r, ext=sp)["row"]["text"] == "Per il periodo duemilaventuno duemilaventisette affinché si possa."
print("ext-only candidates ok")

# ext_veto with one neighbour: the transcript words the left neighbour differently ("dell'articolo")
cap = "Tra queste rientra l'articolo 2.2, la concessione."
sp, ext = "tra queste rientra l'articolo due due la concessione", "tra queste rientra dell'articolo due punto due la concessione"
assert label(cap, "it", sp, sp, ext) is None and without(D, "ext_veto", lambda: label(cap, "it", sp, sp, ext)) is not None
# base_iter settles a chain of groups over passes: "1.1.1" said "uno uno uno"
cap, sp = "Il rispetto della Convenzione 1.1.1 sull'eliminazione.", "il rispetto della convenzione uno uno uno sull'eliminazione"
assert label(cap, "it", sp, sp, sp) == "Il rispetto della Convenzione uno uno uno sull'eliminazione.", label(cap, "it", sp, sp, sp)
print("veto one side / base passes ok")

# ord_card: "Nummer 68. Abstimmung" said as a cardinal; the period stays in the label
for cap, lang, t, want in [("Nummer 68. Abstimmung endet.", "de", "68.", "achtundsechzig."),
                           ("Nummer 68. Stemming eindigt.", "nl", "68.", "achtenzestig.")]:
    assert has(cap, lang, t, want), (lang, cands(cap, lang, t))
    assert not without(M, "ord_card", lambda: has(cap, lang, t, want)), lang
cap, sp = "Das ist die Nummer 68. Abstimmung endet.", "das ist die nummer achtundsechzig abstimmung endet"
assert label(cap, "de", sp, sp, sp) == "Das ist die Nummer achtundsechzig. Abstimmung endet.", label(cap, "de", sp, sp, sp)
print("ord_card ok")


# ---------------------------------------------------------------- wave-1 ext_pick, year_spoken, ext_value (2026-10-16)
class FakeCTC2(FakeCTC):
    def __init__(self, lang, spoken, greedy, per_letter):
        super().__init__(lang, spoken, greedy)
        self.k = per_letter

    def nll(self, lp, text):
        return self.k * D._lev(C.letters(text), self.spoken)


def label2(cap, lang, spoken, greedy, ext, per_letter=3.0):
    r = D.decide(FakeCTC2(lang, spoken, greedy, per_letter), cap, None, ext=ext)
    if not all(s["decided"] for s in r["spans"]):
        return None
    return M.render(cap, M.spans(cap, lang), [s["pick"] for s in r["spans"]]), [s["by"] for s in r["spans"]]


# the CTC drops a short spoken word with many nats to spare; the transcript's reading only adds that word: it decides
cap = "Wir treffen uns um 16.30 Uhr hier."
sp, ext = "wir treffen uns um sechzehn dreißig hier", "wir treffen uns um sechzehn uhr dreißig hier"
assert label2(cap, "de", sp, sp, ext, 8.0) == ("Wir treffen uns um sechzehn Uhr dreißig hier.", ["ext_pick"]), \
    label2(cap, "de", sp, sp, ext, 8.0)
assert without(D, "ext_pick", lambda: label2(cap, "de", sp, sp, ext, 8.0)) is None
# a different word form far from the CTC best (16 nats, not an added word): refused
cap = "Em 2015 houve uma crise."
sp, ext = "em duas mil e quinze houve uma crise", "em dois mil e quinze houve uma crise"
assert label2(cap, "pt", sp, sp, ext, 8.0) is None
# ... and decided by the transcript when the CTC barely prefers the other form (gap <= EXT_PICK_GAP)
assert label2(cap, "pt", sp, sp, ext, 1.0) == ("Em dois mil e quinze houve uma crise.", ["ext_pick"]), label2(cap, "pt", sp, sp, ext, 1.0)
# "hundertdrei" / "einhundertdrei": the transcript's own spelling habit, never decided by it
cap = "Es kostet 103 Millionen Euro."
sp, ext = "es kostet hundertdrei millionen euro", "es kostet einhundertdrei millionen euro"
assert label2(cap, "de", sp, sp, ext, 1.0) is None
assert D._one_scale_only("hundertdrei", "einhundertdrei") and not D._one_scale_only("zwölf dreißig", "zwölf Uhr dreißig")
assert D._adds_words("otto quattro", "otto punto quattro") and not D._adds_words("dois mil", "duas mil")
# a caption number the audio lacks must not borrow the transcript's second occurrence of it
cap = "2015 begann es und 2015 endete es."
spn = D.spans_for(cap, "pt", "x")
assert not D._ext_once_each(spn, spn[0], "dois mil e quinze", "e dois mil e quinze endete es")
assert D._ext_once_each(spn, spn[0], "dois mil e quinze", "dois mil e quinze begann es und dois mil e quinze endete es")
print("wave-1 ext_pick ok")

# year_spoken: a year said by its last two digits / as two pairs is read so when the transcript and the CTC agree
cap = "Das war 1991 so."
sp = "das war einundneunzig so"
assert label2(cap, "de", sp, sp, sp) == ("Das war einundneunzig so.", ["ctc"]), label2(cap, "de", sp, sp, sp)
assert without(M, "year_spoken", lambda: label2(cap, "de", sp, sp, sp)) is None
# the CTC alone (the transcript lacks it in place) never takes the last-two-digits reading: it may have dropped "neunzehnhundert"
assert label2(cap, "de", sp, sp, "das war so") is None
cap, sp = "Bis 2060 also nicht.", "bis zwanzig sechzig also nicht"
assert label2(cap, "de", sp, sp, sp)[0] == "Bis zwanzig sechzig also nicht."
# the full reading said: the short readings are far behind and change nothing
cap, sp = "Das war 1991 so.", "das war neunzehnhunderteinundneunzig so"
assert label2(cap, "de", sp, sp, sp)[0] == "Das war neunzehnhunderteinundneunzig so."
print("year_spoken ok")

# ext_value (nl): the transcript writes digits; its digits in place confirm the value, the CTC must clearly prefer one
# wording (the xlsr greedy text garbles Dutch number words, so `fits` fails)
cap = "Dat is sinds 1993 zo gebleven."
sp, g = "dat is sinds negentienhonderddrieënnegentig zo gebleven", "dat is sinds negentienonderdrieenegentig zo gebleven"
assert label2(cap, "nl", sp, g, "dat is sinds 1993 zo gebleven", 8.0) == (
    "Dat is sinds negentienhonderddrieënnegentig zo gebleven.", ["ext_value"]), label2(cap, "nl", sp, g, "dat is sinds 1993 zo gebleven", 8.0)
assert without(D, "ext_value", lambda: label2(cap, "nl", sp, g, "dat is sinds 1993 zo gebleven", 8.0)) is None
assert label2(cap, "nl", sp, g, "dat is sinds 1983 zo gebleven", 8.0) is None  # another value
assert label2(cap, "nl", sp, g, "dat is 1993 zo gebleven", 8.0) is None  # not in the caption's place
assert label2(cap, "nl", sp, g, "dat is sinds 1993 zo gebleven", 1.0) is None  # the CTC does not clearly prefer one
cap = "Het gaat om 8 banen."
sp, g = "het gaat om acht banen", "het gaat om ach banen"
s0 = M.spans(cap, "nl")[0]
assert D.ext_value(cap, s0, "het gaat om 8 banen", "nl") == {"ord": False}
assert D.ext_value(cap, s0, "het gaat om 8000 banen", "nl") is None and D.ext_value(cap, s0, "het gaat om 8 duizend banen", "nl") is None
cap = "Ruim 15% meer."
s0 = M.spans(cap, "nl")[0]
assert D.ext_value(cap, s0, "ruim 15 procent meer", "nl") and D.ext_value(cap, s0, "ruim 15% meer", "nl")
assert D.ext_value(cap, s0, "ruim 15 meer", "nl") is None
cap = "Dat was op de 21. dag."
s0 = M.spans(cap, "nl")[0]
assert D.ext_value(cap, s0, "dat was op de 21e dag", "nl") == {"ord": True}
assert D.ext_value("Het is 1,7 miljard.", M.spans("Het is 1,7 miljard.", "nl")[0], "het is 17 miljard", "nl") is None
print("ext_value ok")
