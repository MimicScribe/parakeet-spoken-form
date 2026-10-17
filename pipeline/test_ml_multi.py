"""python3 -m unittest test_ml_multi  (needs ml_multi_lex.json; the reader is built from it)."""
import unittest
from decimal import Decimal as D

import ml_multi as M
import ml_tts_multi as B


def vals(text, lang):
    return [n.value for n in M.parse_numbers(text, lang)]


def words(text, step=400):
    return [{"w": w, "s": i * step, "e": i * step + step - 50} for i, w in enumerate(text.split())]


class Reader(unittest.TestCase):
    def test_polish_cases(self):
        self.assertEqual(vals("w dwa tysiące dwudziestym piątym roku", "pl"), [D(2025)])
        self.assertEqual(vals("do pięciu milionów euro", "pl"), [D(5000000)])
        self.assertEqual(vals("tysiąc dziewięćset osiemdziesiątego dziewiątego roku", "pl"), [D(1989)])
        self.assertEqual(vals("dwieście pięćdziesiąt tysięcy", "pl"), [D(250000)])
        self.assertEqual(vals("o piętnastej trzydzieści", "pl"), [D(15), D(30)])
        self.assertEqual(vals("osiemnastego", "pl"), [D(18)])

    def test_polish_decimals_halves(self):
        self.assertEqual(vals("trzy przecinek zero pięć", "pl"), [D("3.05")])
        self.assertEqual(vals("trzy przecinek pięć procent", "pl"), [D("3.5")])
        self.assertEqual(vals("półtora miliona", "pl"), [D(1500000)])
        self.assertEqual(vals("dwa i pół", "pl"), [D("2.5")])
        self.assertEqual(vals("pół roku temu", "pl"), [])  # "pół roku" is not a number value

    def test_not_a_number_word(self):
        self.assertEqual(vals("trzeba to zrobić", "pl"), [])

    def test_bulgarian_cents_word_is_not_hundred(self):
        # "стотинки" (cents) starts like "сто" (100): a price "деветдесет и пет лева и деветдесет стотинки" is 95 + 90
        self.assertEqual(vals("деветдесет и пет лева и деветдесет стотинки", "bg"), [D(95), D(90)])
        self.assertEqual(vals("сто лева", "bg"), [D(100)])

    def test_romanian(self):
        self.assertEqual(vals("în anul două mii douăzeci și cinci", "ro"), [D(2025)])
        self.assertEqual(vals("o mie nouă sute optzeci și nouă", "ro"), [D(1989)])
        self.assertEqual(vals("trei virgulă cinci la sută", "ro"), [D("3.5")])  # "la sută" = percent, not 100
        self.assertEqual(vals("al doilea", "ro"), [D(2)])  # the article is not a number
        self.assertEqual(vals("cinci milioane de euro", "ro"), [D(5000000)])

    def test_swedish_compounds(self):
        self.assertEqual(vals("tvåtusentjugofem", "sv"), [D(2025)])
        self.assertEqual(vals("nittonhundranittiotre", "sv"), [D(1993)])
        self.assertEqual(vals("tjugoförsta", "sv"), [D(21)])
        self.assertEqual(vals("tre komma fem procent", "sv"), [D("3.5")])

    def test_two_numbers_not_glued(self):
        self.assertEqual(vals("pięć dwadzieścia", "pl"), [D(5), D(20)])
        self.assertEqual(vals("zero pięć", "pl"), [D(0), D(5)])


class Written(unittest.TestCase):
    def test_grouping_and_scale(self):
        self.assertIn([D(1500000)], M.written_values(["1", "500", "000"]))
        self.assertIn([D(5000000)], M.written_values(["5", "mln"]))
        self.assertIn([D("3.5")], M.written_values(["3,5%"]))
        self.assertIn([D(15), D(30)], M.written_values(["15.30"]))
        self.assertIn([D(15)], M.written_values(["15:00"]))


class LabelA(unittest.TestCase):
    def test_accept_inflected_words(self):
        cap = "W 2025 roku wydamy 5 mln zł."
        ext = words("w dwa tysiące dwudziestym piątym roku wydamy pięć milionów złotych")
        r = M.label_row(cap, ext, "pl")
        self.assertEqual(r["text"], "W dwa tysiące dwudziestym piątym roku wydamy pięć milionów złotych.")
        self.assertFalse(r["masked"])

    def test_value_mismatch_is_masked_and_cut(self):
        cap = "Mamy 2024 projekty i 15 wniosków."
        ext = words("mamy dwa tysiące dwadzieścia trzy projekty i piętnaście wniosków")
        r = M.label_row(cap, ext, "pl")
        self.assertEqual(len(r["masked"]), 1)
        self.assertIn("piętnaście", r["text"])
        self.assertEqual(len(r["pieces"]), 2)
        self.assertTrue(r["pieces"][1][2].startswith("projekty"))

    def test_sentence_start_capital(self):
        r = M.label_row("15 posłów głosowało.", words("piętnastu posłów głosowało"), "pl")
        self.assertEqual(r["text"], "Piętnastu posłów głosowało.")


class SyntheticChecks(unittest.TestCase):
    def test_row_ok(self):
        r = {"written": "12,49 zł", "check": "values", "text_written": "Bilet kosztuje 12,49 zł.",
             "text": "Bilet kosztuje dwanaście złotych czterdzieści dziewięć groszy."}
        ok, why = B.check_row(r, "pl")
        self.assertTrue(ok, why)

    def test_row_wrong_value(self):
        r = {"written": "12,49 zł", "check": "values", "text_written": "Bilet kosztuje 12,49 zł.",
             "text": "Bilet kosztuje dwanaście złotych czterdzieści osiem groszy."}
        self.assertFalse(B.check_row(r, "pl")[0])

    def test_bare_cents_rejected(self):
        self.assertFalse(B.cents_ok("912,05 €", "nouă sute doisprezece, cinci euro", "ro"))
        self.assertTrue(B.cents_ok("912,05 €", "nouă sute doisprezece euro și cinci cenți", "ro"))
        self.assertTrue(B.cents_ok("89,15 €", "osiemdziesiąt dziewięć euro piętnaście centów", "pl"))

    def test_phone_digits(self):
        self.assertTrue(B.piece_ok("512 345 678", "pięćset dwanaście trzysta czterdzieści pięć sześćset siedemdziesiąt osiem",
                                   "digits", "pl")[0])
        self.assertFalse(B.piece_ok("512 345 678", "pięćset dwanaście trzysta czterdzieści pięć", "digits", "pl")[0])


class CaseExpansion(unittest.TestCase):
    """2026-10-16 fix 1: the candidate set always holds the lexicon-driven inflections of every reading."""

    def test_polish_year_locative_always_offered(self):
        import ml_multi_ctc as C

        # pilot miss: the LLM offered only these two for "w 2013 roku"; the speaker said "trzynastym"
        c = C.expand_cands("2013", ["dwa tysiące trzynaście", "dwa tysiące trzynastego"], "pl", " roku", "w ")
        self.assertIn("dwa tysiące trzynastym", c)
        self.assertEqual(c[:2], ["dwa tysiące trzynaście", "dwa tysiące trzynastego"])  # the LLM's order first

    def test_range_read_without_connector_is_offered(self):
        import ml_multi_ctc as C

        # pilot hand check: the speaker said "dwadzieścia dwadzieścia pięć procent" for "20-25%", without "do"
        c = C.expand_cands("20-25%", ["dwadzieścia do dwadzieścia pięć procent"], "pl")
        self.assertIn("dwadzieścia dwadzieścia pięć procent", c)

    def test_romanian_de_and_decimal_word(self):
        import ml_multi_ctc as C

        c = C.expand_cands("2 miliarde euro", ["două miliarde euro"], "ro")
        self.assertIn("două miliarde de euro", c)
        c = C.expand_cands("6,3%", ["șase virgula trei la sută"], "ro")
        self.assertIn("șase virgulă trei la sută", c)
        self.assertNotIn("șase virgula trei la sută", c)

    def test_polish_clock_hour_as_ordinal(self):
        import ml_multi_ctc as C

        # final hand check: "o godzinie 11.30" got "jedenastu trzydzieści"; the LLM offered only cardinal hours
        c = C.expand_cands("11.30", ["jedenaście trzydzieści"], "pl")
        self.assertIn("jedenastej trzydzieści", c)
        self.assertIn("jedenasta trzydzieści", c)

    def test_polish_cases_of_a_scaled_amount(self):
        import ml_multi_ctc as C

        c = C.expand_cands("5 mln zł", ["pięć milionów złotych"], "pl")
        self.assertIn("pięciu milionów złotych", c)
        self.assertIn("pięcioma milionami złotych", c)

    def test_romanian_ordinal_gender(self):
        import ml_multi_cases as MC

        self.assertIn("a doua", MC.expand("al doilea", "ro"))
        self.assertIn("două mii douăzeci și două", [w for _k, w in MC._variants(2022, False, "ro")])

    def test_paradigms_parse_back(self):
        import ml_multi_cases as MC

        for lang in ("pl", "ro"):
            for n in list(range(0, 2101)) + [22000, 250000, 1500000, 5000000, 10**9, 3 * 10**9]:
                for _k, w in MC._variants(n, False, lang) + MC._variants(n, True, lang):
                    self.assertTrue(M.values_match([str(n)], w, lang)[0], (lang, n, w))

    def test_paradigm_forms(self):
        import ml_multi_cases as MC

        self.assertEqual(MC.pl_ordinal(1989, "ym"), "tysiąc dziewięćset osiemdziesiątym dziewiątym")
        self.assertEqual(MC.pl_ordinal(2000, "ego"), "dwutysięcznego")
        self.assertEqual(MC.pl_cardinal(22000, "gen"), "dwudziestu dwóch tysięcy")
        self.assertEqual(MC.pl_cardinal(2013, "nom_mp"), "dwóch tysięcy trzynastu")
        self.assertEqual(MC.ro_cardinal(25 * 10**6), "douăzeci și cinci de milioane")
        self.assertEqual(MC.ro_ordinal(20, "f"), "a douăzecea")

    def test_bad_drafted_form_no_longer_misreads(self):
        # the pl draft listed "trzydziestoma" under 13; the curated paradigm owns it as 30
        self.assertEqual(vals("trzydziestoma", "pl"), [D(30)])


class ReadingAllowlist(unittest.TestCase):
    """2026-10-16 fix 2: a reading holds only number words, the number's own unit / currency / percent word, and
    connectors."""

    def test_swedish_punk_rejected(self):
        import ml_multi_ctc as C

        self.assertFalse(C.reading_ok("1.1", "ett punk ett", "sv"))
        self.assertTrue(C.reading_ok("1.1", "ett komma ett", "sv"))

    def test_invented_word_rejected(self):  # red-on-base: control (the old extra-word count caught these)
        import ml_multi_ctc as C

        self.assertFalse(C.reading_ok("2013", "dwa tysiące trzynastego wieku", "pl"))
        self.assertFalse(C.reading_ok("15", "piętnastu posłów", "pl"))

    def test_written_name_not_lost(self):
        import ml_multi_ctc as C

        # pilot 2026-10-16: "COVID-19" read "dziewiętnaste" passed the value check and dropped the name
        self.assertFalse(C.reading_ok("COVID-19", "dziewiętnaste", "pl"))
        self.assertFalse(C.reading_ok("Horyzont 2020", "dwa tysiące dwadzieścia", "pl"))
        self.assertTrue(C.reading_ok("Horyzont 2020", "Horyzont dwa tysiące dwadzieścia", "pl"))
        self.assertTrue(C.reading_ok("5-tego", "piątego", "pl"))

    def test_percent_word_after_number(self):
        import ml_multi_ctc as C

        self.assertFalse(C.reading_ok("0,1%", "zero procent i jeden", "pl"))  # pilot: value [0, 1] passed as a pair
        self.assertTrue(C.reading_ok("0,1%", "zero przecinek jeden procent", "pl"))

    def test_units_and_connectors_kept(self):
        import ml_multi_ctc as C

        self.assertTrue(C.reading_ok("2013 r.", "dwa tysiące trzynastego roku", "pl"))
        self.assertTrue(C.reading_ok("5 mln zł", "pięć milionów złotych", "pl"))
        self.assertTrue(C.reading_ok("3,5%", "trzy i pół procent", "pl"))
        self.assertTrue(C.reading_ok("5-10", "pięć do dziesięciu", "pl"))
        self.assertTrue(C.reading_ok("17.08.2020", "siedemnastego sierpnia dwa tysiące dwudziestego", "pl"))
        self.assertTrue(C.reading_ok("al 2-lea", "al doilea", "ro"))
        self.assertTrue(C.reading_ok("25%", "douăzeci și cinci la sută", "ro"))
        self.assertTrue(C.reading_ok("20 de milioane de euro", "douăzeci de milioane de euro", "ro"))


class TranscriptDigits(unittest.TestCase):
    """The external transcript writes pl / ro numbers as digits, sometimes in additive groups or with month names."""

    def test_agree(self):
        import ml_multi_ctc as C

        self.assertTrue(C.digits_agree("2.10.1998", "2 października 1998", "pl"))
        self.assertTrue(C.digits_agree("+40 721 158 586", "40 720 1 150 8 580 6", "ro"))
        self.assertTrue(C.digits_agree("0786 341 528", "0 786 341 520 8", "ro"))
        self.assertTrue(C.digits_agree("22 942 44 47", "22 942 4447", "pl"))
        self.assertTrue(C.digits_agree("2024", "2020 4", "ro"))

    def test_disagree(self):
        import ml_multi_ctc as C

        self.assertFalse(C.digits_agree("2024", "2020 5", "ro"))
        self.assertFalse(C.digits_agree("203", "20 3", "pl"))
        self.assertFalse(C.digits_agree("3,5%", "35%", "pl"))
        self.assertFalse(C.digits_agree("72860", "70-2860", "pl"))


# ---------------------------------------------------------------- completeness
# Hand-written nominative forms per language: cardinals 0-20, tens 20-90, 100, 1000 and ordinals 1-10 (None: the
# language's table has no plain ordinal list to test); ";" separates multi-word entries.
COMPLETE = {
    "bg": dict(
        card="нула един два три четири пет шест седем осем девет десет единадесет дванадесет тринадесет четиринадесет петнадесет шестнадесет седемнадесет осемнадесет деветнадесет двадесет",
        tens="двадесет тридесет четиридесет петдесет шестдесет седемдесет осемдесет деветдесет",
        hundred="сто",
        thousand="хиляда",
        ordinal="първи втори трети четвърти пети шести седми осми девети десети"),
    "hr": dict(
        card="nula jedan dva tri četiri pet šest sedam osam devet deset jedanaest dvanaest trinaest četrnaest petnaest šesnaest sedamnaest osamnaest devetnaest dvadeset",
        tens="dvadeset trideset četrdeset pedeset šezdeset sedamdeset osamdeset devedeset",
        hundred="sto",
        thousand="tisuću",
        ordinal="prvi drugi treći četvrti peti šesti sedmi osmi deveti deseti"),
    "cs": dict(
        card="nula jedna dva tři čtyři pět šest sedm osm devět deset jedenáct dvanáct třináct čtrnáct patnáct šestnáct sedmnáct osmnáct devatenáct dvacet",
        tens="dvacet třicet čtyřicet padesát šedesát sedmdesát osmdesát devadesát",
        hundred="sto",
        thousand="tisíc",
        ordinal="první druhý třetí čtvrtý pátý šestý sedmý osmý devátý desátý"),
    "da": dict(
        card="nul en to tre fire fem seks syv otte ni ti elleve tolv tretten fjorten femten seksten sytten atten nitten tyve",
        tens="tyve tredive fyrre halvtreds treds halvfjerds firs halvfems",
        hundred="hundrede",
        thousand="tusind",
        ordinal="første anden tredje fjerde femte sjette syvende ottende niende tiende"),
    "et": dict(
        card="null üks kaks kolm neli viis kuus seitse kaheksa üheksa kümme üksteist kaksteist kolmteist neliteist viisteist kuusteist seitseteist kaheksateist üheksateist kakskümmend",
        tens="kakskümmend kolmkümmend nelikümmend viiskümmend kuuskümmend seitsekümmend kaheksakümmend üheksakümmend",
        hundred="sada",
        thousand="tuhat",
        ordinal="esimene teine kolmas neljas viies kuues seitsmes kaheksas üheksas kümnes"),
    "fi": dict(
        card="nolla yksi kaksi kolme neljä viisi kuusi seitsemän kahdeksan yhdeksän kymmenen yksitoista kaksitoista kolmetoista neljätoista viisitoista kuusitoista seitsemäntoista kahdeksantoista yhdeksäntoista kaksikymmentä",
        tens="kaksikymmentä kolmekymmentä neljäkymmentä viisikymmentä kuusikymmentä seitsemänkymmentä kahdeksankymmentä yhdeksänkymmentä",
        hundred="sata",
        thousand="tuhat",
        ordinal="ensimmäinen toinen kolmas neljäs viides kuudes seitsemäs kahdeksas yhdeksäs kymmenes"),
    "el": dict(
        card="μηδέν ένα δύο τρία τέσσερα πέντε έξι επτά οκτώ εννέα δέκα έντεκα δώδεκα δεκατρία δεκατέσσερα δεκαπέντε δεκαέξι δεκαεπτά δεκαοκτώ δεκαεννέα είκοσι",
        tens="είκοσι τριάντα σαράντα πενήντα εξήντα εβδομήντα ογδόντα ενενήντα",
        hundred="εκατό",
        thousand="χίλια",
        ordinal="πρώτος δεύτερος τρίτος τέταρτος πέμπτος έκτος έβδομος όγδοος ένατος δέκατος"),
    "hu": dict(
        card="nulla egy kettő három négy öt hat hét nyolc kilenc tíz tizenegy tizenkettő tizenhárom tizennégy tizenöt tizenhat tizenhét tizennyolc tizenkilenc húsz",
        tens="húsz harminc negyven ötven hatvan hetven nyolcvan kilencven",
        hundred="száz",
        thousand="ezer",
        ordinal="első második harmadik negyedik ötödik hatodik hetedik nyolcadik kilencedik tizedik"),
    "lv": dict(
        card="nulle viens divi trīs četri pieci seši septiņi astoņi deviņi desmit vienpadsmit divpadsmit trīspadsmit četrpadsmit piecpadsmit sešpadsmit septiņpadsmit astoņpadsmit deviņpadsmit divdesmit",
        tens="divdesmit trīsdesmit četrdesmit piecdesmit sešdesmit septiņdesmit astoņdesmit deviņdesmit",
        hundred="simts",
        thousand="tūkstotis",
        ordinal="pirmais otrais trešais ceturtais piektais sestais septītais astotais devītais desmitais"),
    "lt": dict(
        card="nulis vienas du trys keturi penki šeši septyni aštuoni devyni dešimt vienuolika dvylika trylika keturiolika penkiolika šešiolika septyniolika aštuoniolika devyniolika dvidešimt",
        tens="dvidešimt trisdešimt keturiasdešimt penkiasdešimt šešiasdešimt septyniasdešimt aštuoniasdešimt devyniasdešimt",
        hundred="šimtas",
        thousand="tūkstantis",
        ordinal="pirmas antras trečias ketvirtas penktas šeštas septintas aštuntas devintas dešimtas"),
    "mt": dict(
        card="żero wieħed tnejn tlieta erbgħa ħamsa sitta sebgħa tmienja disgħa għaxra ħdax tnax tlettax erbatax ħmistax sittax sbatax tmintax dsatax għoxrin",
        tens="għoxrin tletin erbgħin ħamsin sittin sebgħin tmenin disgħin",
        hundred="mija",
        thousand="elf",
        ordinal="l-ewwel;it-tieni;it-tielet;ir-raba';il-ħames;is-sitt;is-seba';it-tmien;id-disa';l-għaxar"),
    "pl": dict(
        card="zero jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć jedenaście dwanaście trzynaście czternaście piętnaście szesnaście siedemnaście osiemnaście dziewiętnaście dwadzieścia",
        tens="dwadzieścia trzydzieści czterdzieści pięćdziesiąt sześćdziesiąt siedemdziesiąt osiemdziesiąt dziewięćdziesiąt",
        hundred="sto",
        thousand="tysiąc",
        ordinal="pierwszy drugi trzeci czwarty piąty szósty siódmy ósmy dziewiąty dziesiąty"),
    "ro": dict(
        card="zero unu doi trei patru cinci șase șapte opt nouă zece unsprezece doisprezece treisprezece paisprezece cincisprezece șaisprezece șaptesprezece optsprezece nouăsprezece douăzeci",
        tens="douăzeci treizeci patruzeci cincizeci șaizeci șaptezeci optzeci nouăzeci",
        hundred="o sută",
        thousand="o mie",
        ordinal="primul;al doilea;al treilea;al patrulea;al cincilea;al șaselea;al șaptelea;al optulea;al nouălea;al zecelea"),
    "sk": dict(
        card="nula jeden dva tri štyri päť šesť sedem osem deväť desať jedenásť dvanásť trinásť štrnásť pätnásť šestnásť sedemnásť osemnásť devätnásť dvadsať",
        tens="dvadsať tridsať štyridsať päťdesiat šesťdesiat sedemdesiat osemdesiat deväťdesiat",
        hundred="sto",
        thousand="tisíc",
        ordinal="prvý druhý tretí štvrtý piaty šiesty siedmy ôsmy deviaty desiaty"),
    "sl": dict(
        card="nič ena dva tri štiri pet šest sedem osem devet deset enajst dvanajst trinajst štirinajst petnajst šestnajst sedemnajst osemnajst devetnajst dvajset",
        tens="dvajset trideset štirideset petdeset šestdeset sedemdeset osemdeset devetdeset",
        hundred="sto",
        thousand="tisoč",
        ordinal="prvi drugi tretji četrti peti šesti sedmi osmi deveti deseti"),
    "sv": dict(
        card="noll en två tre fyra fem sex sju åtta nio tio elva tolv tretton fjorton femton sexton sjutton arton nitton tjugo",
        tens="tjugo trettio fyrtio femtio sextio sjuttio åttio nittio",
        hundred="hundra",
        thousand="tusen",
        ordinal="första andra tredje fjärde femte sjätte sjunde åttonde nionde tionde"),
}


class Completeness(unittest.TestCase):
    """Every language's plain nominative numerals read back as exactly their value (a hole in a table is a number
    the pipeline silently cannot read)."""

    @staticmethod
    def _split(s):
        return s.split(";") if ";" in s else s.split()

    def test_all_languages_listed(self):
        self.assertEqual(set(COMPLETE), set(M.L16))

    def test_cardinals_tens_hundred_thousand(self):
        for lang, d in COMPLETE.items():
            exp = list(enumerate(self._split(d["card"]))) + [(20 + 10 * i, w) for i, w in enumerate(self._split(d["tens"]))]
            exp += [(100, d["hundred"]), (1000, d["thousand"])]
            self.assertEqual([v for v, _ in exp[:21]], list(range(21)))
            for v, w in exp:
                self.assertEqual(vals(w, lang), [D(v)], (lang, w, v))

    def test_ordinals_1_to_10(self):
        for lang, d in COMPLETE.items():
            if d["ordinal"] is None:  # no plain ordinal list to test
                continue
            ws = self._split(d["ordinal"])
            self.assertEqual(len(ws), 10, lang)
            for v, w in enumerate(ws, 1):
                self.assertEqual(vals(w, lang), [D(v)], (lang, w, v))


class TableHoles(unittest.TestCase):
    def test_slovene_small_numbers_and_compounds(self):
        for w, v in (("nič", 0), ("ena", 1), ("dva", 2), ("tri", 3), ("štiri", 4), ("pet", 5), ("šest", 6), ("sedem", 7),
                     ("šestdeset", 60), ("petindvajset", 25), ("dvaindvajset", 22), ("šestdeseti", 60), ("prvi", 1),
                     ("sedmi", 7), ("tretji", 3)):
            self.assertEqual(vals(w, "sl"), [D(v)], w)

    def test_base_forms_added_by_hand(self):
        self.assertEqual(vals("trys šimtai", "lt"), [D(300)])
        self.assertEqual(vals("divi simti", "lv"), [D(200)])
        self.assertEqual(vals("elfejn", "mt"), [D(2000)])
        self.assertEqual(vals("tliet elef", "mt"), [D(3000)])
        self.assertEqual(vals("förtio", "sv"), [D(40)])
        self.assertEqual(vals("sittax", "mt"), [D(16)])  # was read as 6

    def test_maltese_and_is_not_a_denominator(self):
        self.assertEqual(vals("ħamsa u għoxrin", "mt"), [D(25)])
        self.assertEqual(vals("mija u ħamsa", "mt"), [D(105)])
        self.assertNotIn("u", M.lexicon("mt").frac)

    def test_finnish_teen_ordinals(self):
        for w, v in (("yhdestoista", 11), ("kahdestoista", 12), ("kolmastoista", 13), ("neljästoista", 14),
                     ("viidestoista", 15), ("kuudestoista", 16), ("seitsemästoista", 17), ("kahdeksastoista", 18),
                     ("yhdeksästoista", 19), ("viidennentoista", 15), ("kahdennessatoista", 12)):
            ns = M.parse_numbers(w, "fi")
            self.assertEqual([n.value for n in ns], [D(v)], w)
            self.assertTrue(ns[0].ordinal, w)
        self.assertEqual(vals("yksitoista", "fi"), [D(11)])  # the cardinal stays a cardinal
        self.assertFalse(M.parse_numbers("yksitoista", "fi")[0].ordinal)
        self.assertEqual(vals("toista", "fi"), [])


class UnreadableCount(unittest.TestCase):
    """A scale word after an unknown alphabetic word is not read alone: refusing masks the span, a value must not
    change."""

    def test_unknown_word_before_scale_word(self):
        self.assertEqual(vals("kelios šimtai", "lt"), [])
        self.assertEqual(vals("nekaj tisoč", "sl"), [])
        self.assertEqual(vals("kilka tysięcy", "pl"), [])  # "a few thousand" is no value
        self.assertEqual(vals("dvi tisoč", "sl"), [])  # a count-like word the tables lack

    def test_function_word_before_scale_word_reads(self):
        self.assertEqual(vals("w roku tysiąc dziewięćset", "pl"), [D(1900)])  # "in the year 1900"
        self.assertEqual(vals("v roku tisícdeväťstotridsaťdeväť", "sk"), [D(1939)])
        self.assertEqual(vals("več kot tisoč", "sl"), [D(1000)])  # "more than a thousand"

    def test_known_counts_still_read(self):
        self.assertEqual(vals("du šimtai", "lt"), [D(200)])
        self.assertEqual(vals("dva tisoč", "sl"), [D(2000)])
        self.assertEqual(vals("pięć tysięcy", "pl"), [D(5000)])

    def test_break_or_one_letter_word_before_scale_word(self):
        self.assertEqual(vals("kot, tisoč", "sl"), [D(1000)])
        self.assertEqual(vals("o sută", "ro"), [D(100)])
        self.assertEqual(vals("w tysiąc", "pl"), [D(1000)])
        self.assertEqual(vals("tisoč", "sl"), [D(1000)])


class HalfConstruction(unittest.TestCase):
    def test_side_words_are_not_halves(self):
        for lang, text, want in (("et", "viis poolt kolm vastu", [5, 3]), ("fi", "kolme puolella", [3]),
                                 ("hu", "három fel", [3]), ("sv", "tre halv", [3]), ("da", "to halv", [2]),
                                 ("lv", "trīs pus", [3]), ("lt", "du puse", [2]), ("pl", "pół roku", [])):
            self.assertEqual(vals(text, lang), [D(x) for x in want], (lang, text))

    def test_proper_constructions_read(self):
        for lang, text, want in (("pl", "dwa i pół", "2.5"), ("sv", "tre och en halv", "3.5"), ("da", "to og en halv", "2.5"),
                                 ("fi", "kolme ja puoli", "3.5"), ("et", "kolm ja pool", "3.5"), ("hu", "három és fél", "3.5"),
                                 ("lv", "trīs un puse", "3.5"), ("lt", "du su puse", "2.5"), ("mt", "tlieta u nofs", "3.5"),
                                 ("sl", "tri in pol", "3.5"), ("ro", "trei și jumătate", "3.5")):
            self.assertEqual(vals(text, lang), [D(want)], (lang, text))


class NotNumbers(unittest.TestCase):
    def test_words_that_fold_or_stem_onto_numbers(self):
        for lang, ws in (("hr", ["što", "petak", "četvrtak", "petka"]), ("cs", ["ste", "set", "pátek", "čtvrtek", "čtvrtku"]),
                         ("sk", ["ste", "piatok", "štvrtok"]), ("hu", ["egyetem", "hatóság", "hétfő", "hétvégén"]),
                         ("da", ["treenighed"]), ("pl", ["czwartek", "piątek", "piątku"]), ("sl", ["petek"]),
                         ("bg", ["четвъртък", "петък"]), ("et", ["kolmapäev", "neljapäev"]),
                         ("lt", ["pirmadienis", "trečiadienį"]), ("lv", ["otrdiena"])):
            for w in ws:
                self.assertEqual(vals(w, lang), [], (lang, w))

    def test_capitalised_weekdays(self):
        self.assertEqual(vals("τη Τρίτη το πρωί", "el"), [])
        self.assertEqual(vals("την Πέμπτη", "el"), [])
        self.assertEqual(vals("την τρίτη θέση", "el"), [D(3)])
        self.assertEqual(vals("it-Tlieta", "mt"), [])
        self.assertEqual(vals("tlieta", "mt"), [D(3)])

    def test_legit_numbers_and_compounds_survive(self):
        for lang, w, v in (("cs", "sto", 100), ("cs", "stě", 100), ("hr", "sto", 100), ("pl", "dwudziestego", 20),
                           ("pl", "trzydziestu", 30), ("pl", "siedemdziesięciu", 70), ("hu", "huszonötben", 25),
                           ("sv", "tjugofem", 25)):
            self.assertEqual(vals(w, lang), [D(v)], (lang, w))
        self.assertEqual(vals("pět set", "cs"), [D(500)])
        self.assertEqual(vals("dvě stě", "cs"), [D(200)])

    def test_unlisted_ending_must_be_a_number_ending(self):
        lx = M.lexicon("hu")
        self.assertNotIn("tem", lx.endings)


class MonthForms(unittest.TestCase):
    def test_no_common_word_or_spaced_abbreviation(self):
        pl = M.month_forms("pl")
        for w in ("sie", "lis", "lip", "lut"):
            self.assertNotIn(w, pl)
        self.assertEqual(pl["sierpnia"], 8)
        for lang in M.L16:
            for w in M.month_forms(lang):
                self.assertNotIn(" ", w, (lang, w))


class TableHygiene(unittest.TestCase):
    def test_no_junk_forms(self):
        import json
        import re

        with open(M.LEX_PATH) as f:
            txt = f.read()
        self.assertEqual(re.findall("[\x00-\x08\x0b-\x1f\x7f\u00ad\u200b-\u200f]|<[a-z]+>", txt), [])

    def test_check_filters_and_keeps_overrides(self):
        import ml_multi_lexgen as G

        lex = {"cardinal": {"2": ["du"], "3": ["tris", "trys", "bad\x01"]}, "ordinal": {"3": ["trys", "tre<case>"]},
               "decimal": ["kablelis", "x<case>"], "connector": [], "half": ["pusė\xad"], "minus": [],
               "fraction": {"10": ["dešimtoji"]}}
        out = G.check(lex, "lt")
        self.assertIn("trys", out["cardinal"]["3"])
        self.assertNotIn("trys", out["ordinal"]["3"])
        self.assertEqual(out["cardinal"]["3"], ["tris", "trys"])
        self.assertEqual(out["ordinal"]["3"], [])
        self.assertEqual(out["decimal"], ["kablelis"])
        self.assertEqual(out["half"], [])

    def test_clean_form(self):
        import ml_multi_lexgen as G

        self.assertEqual(G.clean_form("o\xa0sută"), "o sută")
        for bad in ("a\x01", "tisoč<case>", "mija\"", "deset\xad", "12", ""):
            self.assertIsNone(G.clean_form(bad), bad)

    def test_failed_chunk_raises(self):
        from unittest import mock

        import ml_multi_lexgen as G

        with mock.patch.object(G.ml_llm, "call", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                G.gen("pl", samples=1)
        with mock.patch.object(G.ml_llm, "call", return_value="{}"):  # parses, but drafts no forms for 0-100
            with self.assertRaises(RuntimeError):
                G.gen("pl", samples=1)


class CuratedModuleImport(unittest.TestCase):
    def test_import_error_inside_cases_module_propagates(self):
        import builtins
        from unittest import mock

        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "ml_multi_cases":
                raise ImportError("broken dependency", name="some_dependency")
            return real(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake):
            with self.assertRaises(ImportError):
                M.Lex("pl")

    def test_missing_module_is_ignored(self):
        import builtins
        from unittest import mock

        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "ml_multi_cases":
                raise ModuleNotFoundError("No module named 'ml_multi_cases'", name="ml_multi_cases")
            return real(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake):
            self.assertTrue(M.Lex("sv").forms)


class HeldDays(unittest.TestCase):
    @staticmethod
    def _vox():
        import importlib
        import sys
        from unittest import mock

        with mock.patch.dict(sys.modules, {"modal": mock.MagicMock()}):
            sys.modules.pop("ml_multi_vox", None)
            return importlib.import_module("ml_multi_vox")

    def test_only_403_404_mean_no_set(self):
        import urllib.error
        from unittest import mock

        V = self._vox()
        for code in (403, 404):
            err = urllib.error.HTTPError("u", code, "x", {}, None)
            with mock.patch("urllib.request.urlretrieve", side_effect=err):
                self.assertEqual(V.held_days("xx"), set())

    def test_other_errors_retry_then_raise(self):
        import urllib.error
        from unittest import mock

        V = self._vox()
        for err in (urllib.error.HTTPError("u", 500, "x", {}, None), urllib.error.URLError("net"), OSError("reset")):
            with mock.patch("urllib.request.urlretrieve", side_effect=err) as m:
                with self.assertRaises(RuntimeError):
                    V.held_days("xx", tries=3, wait=0)
                self.assertEqual(m.call_count, 3)

    def test_transient_error_is_retried(self):
        import urllib.error
        from unittest import mock

        V = self._vox()
        calls = []

        def flaky(url, path):
            calls.append(url)
            if len(calls) < 2:
                raise urllib.error.URLError("blip")
            import gzip

            with gzip.open(path, "wt") as f:
                f.write("session_id|split\n20200101xx|dev\n20190101xx|train\n")

        with mock.patch("urllib.request.urlretrieve", side_effect=flaky):
            self.assertEqual(V.held_days("xx", tries=3, wait=0), {"20200101"})


if __name__ == "__main__":
    unittest.main()
