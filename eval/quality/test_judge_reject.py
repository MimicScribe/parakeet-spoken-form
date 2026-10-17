"""Run: python3 -m pytest eval/quality  (or python3 test_judge_reject.py). Covers review flaws (a)-(e)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import judge_reject as J  # noqa: E402

M, D, C = J.M, J.D, J.C


def row(lang, caption, *readings, rid="r1"):
    """A row as the labeller writes it: spans from the current grammar, our pick = the candidate with that reading."""
    sp = M.spans(caption, lang)
    assert len(sp) == len(readings), [s.text for s in sp]
    picks, info = [], []
    for s, r in zip(sp, readings):
        picks.append([D.key(c) for c in s.cands].index(D.key(r)))
        info.append({"text": s.text, "kind": s.kind, "reading": r})
    return {"id": rid, "caption": caption, "text": C.render_cap(caption, sp, picks), "spans": info}


def ext(text, status="completed"):
    return {"id": "r1", "text": text, "status": status}


def verdict(lang, r, judge_text, **kw):
    return J.judge_row(r, ext(judge_text), lang, **kw)


class RomanceTests(unittest.TestCase):
    # (a) substring matches and invisible scale words
    def test_dropped_scale_word_is_rejected(self):
        r = row("es", "En 2020 hubo 8 puestos más", "dos mil veinte", "ocho")
        self.assertIsNone(verdict("es", r, "en dos mil veinte hubo ocho puestos más")["reason"])
        v = verdict("es", r, "en dos mil veinte hubo ocho mil puestos más")
        self.assertEqual(v["reason"], "span_disagree")

    def test_short_reading_does_not_match_inside_other_words(self):
        r = row("es", "Hay 2 perros aquí", "dos")
        self.assertIsNone(verdict("es", r, "hay dos perros aquí")["reason"])
        # old letter-key substring test: "dos" is inside "adios"
        self.assertEqual(verdict("es", r, "hay tres perros adios aquí")["reason"], "span_disagree")
        self.assertEqual(verdict("es", r, "hay tres perros aquí adios")["reason"], "span_disagree")

    def test_reading_elsewhere_in_sentence_does_not_count(self):
        r = row("es", "Hay 2 perros y dos gatos", "dos")  # the second "dos" is a word, not a span
        self.assertEqual(verdict("es", r, "hay tres perros y dos gatos")["reason"], "span_disagree")

    # (b) "both"
    def test_compound_variant_is_a_disagreement(self):
        r = row("de", "Auf Rang 190 aufgestellt", "hundertneunzig")
        self.assertIsNone(verdict("de", r, "auf rang hundertneunzig aufgestellt")["reason"])
        self.assertEqual(verdict("de", r, "auf rang einhundertneunzig aufgestellt")["reason"], "span_disagree")

    def test_extra_unit_word_after_reading(self):
        r = row("es", "Tengo 11 gatos", "once")
        self.assertIsNone(verdict("es", r, "tengo once gatos")["reason"])
        self.assertEqual(verdict("es", r, "tengo once horas gatos")["reason"], "span_disagree")

    def test_dropped_percent_is_a_disagreement(self):
        r = row("es", "Subió un 30 de nada", "treinta")
        self.assertEqual(verdict("es", r, "subió un treinta por ciento de nada")["reason"], "span_disagree")

    def test_accents_and_spaces_are_folded(self):
        r = row("de", "Es sind 55 Leute", "fünfundfünfzig")
        self.assertIsNone(verdict("de", r, "es sind funfundfunfzig leute")["reason"])
        self.assertIsNone(verdict("de", r, "es sind fünfundfünfzig leute")["reason"])

    # (c) missing / failed judge
    def test_missing_or_failed_judge_rejects(self):
        r = row("es", "Hay 2 perros aquí", "dos")
        for a in (None, ext("hay dos perros aquí", "client_error"), ext("hay dos perros aquí", "error"), ext("")):
            self.assertEqual(J.judge_row(r, a, "es")["reason"], "no_judge")

    def test_retry_replaces_failure(self):
        # a retry appended later replaces the failure for the judge
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "o.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in [
                {"id": "b", "status": "client_error"}, {"id": "b", "status": "completed", "text": "y"}]) + "\n")
            self.assertEqual(J.load_ext(str(p))["b"]["status"], "completed")

    # (d) stale spans
    def test_stale_spans_are_span_mismatch(self):
        r = row("es", "Hay 2 perros aquí", "dos")
        r["spans"][0]["text"] = "2 perros"
        self.assertEqual(verdict("es", r, "hay dos perros aquí")["reason"], "span_mismatch")
        r2 = row("es", "Hay 2 perros aquí", "dos")
        r2["spans"] = []
        self.assertEqual(verdict("es", r2, "hay dos perros aquí")["reason"], "span_mismatch")

    # non-number words
    def test_lost_or_invented_words_are_rejected(self):
        r = row("es", "Hay 2 perros grandes en la casa vieja de mi tío", "dos")
        self.assertIsNone(verdict("es", r, "hay dos perros grandes en la casa vieja de mi tío")["reason"])
        v = verdict("es", r, "hay dos perros que viven lejos de aquí también")
        self.assertEqual(v["reason"], "wer")
        self.assertGreater(v["wer"], 0.25)

    def test_text_only_and_word_lists_agree(self):
        r = row("es", "Hay 2 perros aquí", "dos")
        words = [{"w": w, "c": 1, "s": 0, "e": 1} for w in "hay dos perros aquí".split()]
        self.assertIsNone(J.judge_row(r, {"id": "r1", "status": "completed", "words": words}, "es")["reason"])


class DutchTests(unittest.TestCase):
    # (e)
    def test_value_is_positional(self):
        r = row("nl", "Hij had 30 appels en 20 peren bij zich", "dertig", "twintig")
        self.assertIsNone(verdict("nl", r, "hij had 30 appels en 20 peren bij zich")["reason"])
        # same figures, swapped positions: the old multiset test passed this
        self.assertEqual(verdict("nl", r, "hij had 20 appels en 30 peren bij zich")["reason"], "span_disagree")

    def test_no_digits_is_not_a_pass(self):
        r = row("nl", "Hij had 30 appels en 20 peren bij zich", "dertig", "twintig")
        self.assertEqual(verdict("nl", r, "hij had appels en peren bij zich")["reason"], "span_disagree")
        self.assertIsNone(verdict("nl", r, "hij had dertig appels en twintig peren bij zich")["reason"])
        self.assertEqual(verdict("nl", r, "hij had vijftig appels en twintig peren bij zich")["reason"], "span_disagree")

    def test_extra_zeros_and_scale(self):
        r = row("nl", "Het kost 30 euro per stuk hier", "dertig")
        self.assertEqual(verdict("nl", r, "het kost 30.000 euro per stuk hier")["reason"], "span_disagree")
        r2 = row("nl", "Er waren 2000 mensen daar", "tweeduizend")
        self.assertIsNone(verdict("nl", r2, "er waren 2 duizend mensen daar")["reason"])
        self.assertIsNone(verdict("nl", r2, "er waren 2000 mensen daar")["reason"])

    def test_spoken_numbers_written_as_digits_elsewhere_are_ignored(self):
        r = row("nl", "Hij had 30 appels en twintig peren bij zich", "dertig")
        self.assertIsNone(verdict("nl", r, "hij had 30 appels en 20 peren bij zich")["reason"])

    def test_missing_judge(self):
        r = row("nl", "Hij had 30 appels", "dertig")
        self.assertEqual(J.judge_row(r, None, "nl")["reason"], "no_judge")


class HelperTests(unittest.TestCase):
    def test_numwords_come_from_the_grammar_table(self):
        self.assertTrue({"dos", "mil", "ocho"} <= J.numwords("es"))
        self.assertTrue({"tausend", "null", "zwei"} <= J.numwords("de"))
        self.assertNotIn("y", J.numwords("es"))

    def test_align_maps_positions(self):
        mp = J.align(list("abcd"), list("abxcd"))
        self.assertEqual([m for m in mp], [0, 1, 3, 4])


class SpacingTests(unittest.TestCase):
    # yield audit 2026-10-16: 58 rejected rows differed from the judge text only in compound spacing
    def test_compound_spacing_is_not_a_word_error(self):
        r = row("de", "Nochmal die 4.", "vier")
        self.assertIsNone(verdict("de", r, "noch mal die vier")["reason"])
        r = row("it", "20 punti. Indestrezza.", "venti")
        self.assertIsNone(verdict("it", r, "venti punti in destrezza")["reason"])

    def test_real_word_changes_still_count(self):  # control: passes before and after the fix
        r = row("de", "Nochmal die 4.", "vier")
        self.assertEqual(verdict("de", r, "noch einmal der vier")["reason"], "wer")
        r = row("fr", "Nous avons accepté plus de 250 amendements venus de ce groupe.", "deux cent cinquante")
        self.assertIsNotNone(verdict("fr", r, "nous avons accepté plus de deux cent cinquante amendements")["reason"])


class AuditFixTests(unittest.TestCase):
    """Judge-reject audit 2026-10-17: formatting the second transcript does differently is not a label error."""

    def test_letter_name_vs_bare_letter(self):
        r = row("de", "Die G20 tagt.", "ge zwanzig")
        self.assertIsNone(verdict("de", r, "die g zwanzig tagt")["reason"])
        r = row("fr", "Le CO2 monte.", "cé o deux")
        self.assertIsNone(verdict("fr", r, "le co deux monte")["reason"])

    def test_letter_or_number_change_still_rejected(self):  # red-on-base: control
        r = row("de", "Die G20 tagt.", "ge zwanzig")
        self.assertEqual(verdict("de", r, "die g sieben tagt")["reason"], "span_disagree")
        self.assertEqual(verdict("de", r, "die b zwanzig tagt")["reason"], "span_disagree")
        self.assertEqual(verdict("de", r, "die zwanzig tagt")["reason"], "span_disagree")

    def test_synthetic_joiner_and_hour_word(self):
        r = row("pt", "Às 06:19 sai o trem.", "seis dezenove", rid="tts2_pt_1")
        self.assertIsNone(verdict("pt", r, "às seis e dezenove sai o trem")["reason"])
        self.assertIsNone(verdict("pt", r, "às seis horas e dezenove sai o trem")["reason"])
        self.assertEqual(verdict("pt", r, "às seis e vinte sai o trem")["reason"], "span_disagree")  # value still checked

    def test_synthetic_label_hour_word_is_compared(self):  # red-on-base: control
        r = row("pt", "O trem sai às 15:30 sem falta.", "quinze horas e meia", rid="tts2_pt_2")
        self.assertEqual(verdict("pt", r, "o trem sai às quinze hora e meia sem falta")["reason"], "span_disagree")

    def test_real_row_joiner_stays_strict(self):  # red-on-base: control
        r = row("pt", "Às 06:19 sai o trem.", "seis dezenove")
        self.assertEqual(verdict("pt", r, "às seis e dezenove sai o trem")["reason"], "span_disagree")

    def test_synthetic_german_ein_prefix(self):
        r = row("de", "Auf Rang 190 aufgestellt", "hundertneunzig", rid="tts1_de_1")
        self.assertIsNone(verdict("de", r, "auf rang einhundertneunzig aufgestellt")["reason"])
        r = row("de", "Auf Rang 190 aufgestellt", "hundertneunzig")  # real speech: the two hearings disagree
        self.assertEqual(verdict("de", r, "auf rang einhundertneunzig aufgestellt")["reason"], "span_disagree")

    def test_repeat_is_not_a_lost_word(self):
        r = row("fr", "Il faut agir structurellement sur les 3 marchés.", "trois")
        self.assertIsNone(verdict("fr", r, "il faut agir structurellement structurellement sur les trois marchés")["reason"])
        self.assertEqual(verdict("fr", r, "il faut agir vraiment structurellement sur les trois marchés")["reason"],
                         "lost_word")  # control: a new word still counts
        r = row("de", "Wir sparen 20 Prozent weniger als 2020 ein.", "zwanzig Prozent", "zweitausendzwanzig")
        self.assertEqual(verdict("de", r, "wir sparen weniger als zwanzig prozent weniger als zweitausendzwanzig ein")[
            "reason"], "wer")  # control: words around a number span are not a stutter
        r = row("fr", "Il faut agir structurellement structurellement sur les 3 marchés.", "trois")
        self.assertEqual(verdict("fr", r, "il faut agir structurellement sur les trois marchés")["reason"],
                         "extra_word")  # control: a repeat only the label has is an invented word


class DutchAuditFixTests(unittest.TestCase):
    def test_date_parts(self):
        r = row("nl", "Het contract is op 9.12.2003 getekend.", "negen", " december", " tweeduizenddrie")
        self.assertIsNone(verdict("nl", r, "het contract is op 9 december 2003 getekend")["reason"])
        self.assertEqual(verdict("nl", r, "het contract is op 9 december 2013 getekend")["reason"], "span_disagree")
        self.assertEqual(verdict("nl", r, "het contract is op 8 december 2003 getekend")["reason"], "span_disagree")

    def test_ratio_is_not_split(self):  # red-on-base: control (the split must not reach a ratio)
        r = row("nl", "Dat zou 3:2 zijn.", "drie", "twee")
        self.assertEqual(verdict("nl", r, "dat zou 3 tot 2 zijn")["reason"], "span_disagree")

    def test_judge_clock_is_not_split_for_a_plain_number(self):  # red-on-base: control
        r = row("nl", "We ronden af voor 13 uur.", "dertien")
        self.assertEqual(verdict("nl", r, "we ronden af voor 13.15")["reason"], "span_disagree")

    def test_phone_digit_run(self):
        r = row("nl", "Bel 08-276-05-92 nu.", "nul acht", " twee zeven zes", " nul vijf", " negen twee")
        self.assertIsNone(verdict("nl", r, "bel 082760592 nu")["reason"])
        self.assertEqual(verdict("nl", r, "bel 082760593 nu")["reason"], "span_disagree")
        self.assertIsNone(verdict("nl", r, "bel 08 276 05 92 nu")["reason"])
        self.assertEqual(verdict("nl", r, "bel 0827605921 nu")["reason"], "span_disagree")  # judge heard one more digit

    def test_price_unit_words(self):
        r = row("nl", "De klant betaalde € 21,25 voor de dienst.", "eenentwintig euro en vijfentwintig cent")
        self.assertIsNone(verdict("nl", r, "de klant betaalde 21,25 euro voor de dienst")["reason"])
        self.assertEqual(verdict("nl", r, "de klant betaalde 21,52 euro voor de dienst")["reason"], "span_disagree")


class NegationTests(unittest.TestCase):
    """Quality bar: a lost or invented short negation must not pass."""

    def test_lost_negation(self):
        r = row("pt", "Talvez o saibam, a lei de 1974 é clara.", "mil novecentos e setenta e quatro")
        self.assertEqual(verdict("pt", r, "talvez não o saibam a lei de mil novecentos e setenta e quatro é clara")[
            "reason"], "negation")
        r = row("fr", "Il y a eu 3 votes ce matin.", "trois")
        self.assertEqual(verdict("fr", r, "il n'y a pas eu trois votes ce matin")["reason"], "negation")

    def test_invented_negation(self):
        r = row("de", "Das ist nicht gut für 3 Leute.", "drei")
        self.assertEqual(verdict("de", r, "das ist gut für drei leute")["reason"], "negation")

    def test_same_negations_pass(self):  # red-on-base: control
        r = row("es", "No hay 3 perros, no.", "tres")
        self.assertIsNone(verdict("es", r, "no hay tres perros no")["reason"])
        r = row("nl", "Er staat 100 euro tegenover en niets anders.", "honderd")
        self.assertIsNone(verdict("nl", r, "er staat 100 euro tegenover en niks anders")["reason"])
        r = row("fr", "On n'a prévu que 3 lignes.", "trois")
        self.assertIsNone(verdict("fr", r, "on a prévu que trois lignes")["reason"])  # liaison: not hearable
        r = row("fr", "Il n'y a pas 3 chats.", "trois")
        self.assertIsNone(verdict("fr", r, "il n'y a pas trois chats")["reason"])


class SyntheticReadingTests(unittest.TestCase):
    def test_feminine_year_in_synthetic_row_is_rejected(self):
        r = {"id": "tts2_pt_x", "cls": "year", "caption": "Ficou pronto em 1932.", "text": "Ficou pronto em mil novecentos e trinta e duas.",
             "spans": [{"text": "1932", "kind": "int", "reading": "mil novecentos e trinta e duas"}]}
        self.assertEqual(J.judge_row(r, ext("ficou pronto em mil novecentos e trinta e duas"), "pt")["reason"], "bad_reading")

    def test_feminine_count_is_kept(self):  # red-on-base: control
        r = row("pt", "Comprei 2 casas.", "duas", rid="tts2_pt_y")
        r["cls"] = "count"
        self.assertIsNone(verdict("pt", r, "comprei duas casas")["reason"])


if __name__ == "__main__":
    unittest.main()
