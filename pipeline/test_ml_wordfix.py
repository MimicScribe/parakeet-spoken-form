"""python3 test_ml_wordfix.py  (word repair of judged rows: ml_wordfix)."""
import unittest

import ml_wordfix as W


def lex(*texts):
    lx = W.Lexicon()
    for t in texts:
        lx.add(t)
    return lx


def row(text, reading, raw, lang="de", rid="20141013-1500-COMMITTEE-CONT_de_88_p0"):
    a = text.index(reading)
    return {"id": rid, "lang": lang, "text": text, "caption": text.replace(reading, raw), "duration": 5.0,
            "num_chars": [[a, a + len(reading)]], "spans": [{"text": raw, "reading": reading, "kind": "int"}],
            "audio_filepath": "/vol/x.wav", "word_times": [[0, 1]]}


def ext(text):
    return {"id": "x", "words": [{"w": w} for w in text.split()]}


DE = lex("wir haben das neue Parlament gesehen und das neue Parlament gewählt",
         "dann haben wir eben gesagt und eben nicht gemacht",
         "dass der Präsident Barroso kommt und Präsident Barroso spricht",
         "das machen wir und das macht er und das machen sie und das macht sie")


def scorer(prefer):
    """A fake CTC: texts containing every word of `prefer` score 10 nats better than the caption."""
    return lambda t: -10.0 if all(w in t.lower().split() for w in prefer.split()) else 0.0


class Plan(unittest.TestCase):
    def test_inserts_a_dropped_interior_word(self):
        r = row("Im Jahr zweitausendvierzehn wurden wir in das Parlament gewählt.", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("im jahr zweitausendvierzehn wurden wir in das neue parlament gewählt"), DE, "de")
        self.assertEqual(p["status"], "candidate")
        res = W.decide_row(r, p, DE, "de", scorer("neue"))
        self.assertEqual(res["status"], "repaired")
        nr = res["row"]
        self.assertEqual(nr["text"], "Im Jahr zweitausendvierzehn wurden wir in das neue Parlament gewählt.")
        a, b = nr["num_chars"][0]
        self.assertEqual(nr["text"][a:b], "zweitausendvierzehn")
        self.assertEqual(nr["id"], "20141013-1500-COMMITTEE-CONT_de_88_p0_w")
        self.assertEqual(nr["wordfix"]["of"], r["id"])
        self.assertNotIn("word_times", nr)
        self.assertEqual(nr["caption"], "Im Jahr 2014 wurden wir in das neue Parlament gewählt.")
        self.assertEqual(nr["caption_orig"], r["caption"])

    def test_audio_must_prefer_the_repair_by_the_margin(self):
        r = row("Im Jahr zweitausendvierzehn wurden wir in das Parlament gewählt.", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("im jahr zweitausendvierzehn wurden wir in das neue parlament gewählt"), DE, "de")
        res = W.decide_row(r, p, DE, "de", lambda t: -8.0 if "neue" in t else 0.0)
        self.assertEqual(res["status"], "audio_disagrees")  # 8 nats < MARGIN
        self.assertEqual(res["tried"][0]["gain"], 8.0)

    def test_deletes_an_added_word_keeping_punctuation(self):
        r = row("Wir haben zweitausendvierzehn eben nicht gemacht, und das Parlament gewählt.",
                "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn nicht gemacht und das parlament gewählt"), DE, "de")
        res = W.decide_row(r, p, DE, "de", lambda t: -15.0 if "eben" not in t else 0.0)
        self.assertEqual(res["row"]["text"], "Wir haben zweitausendvierzehn nicht gemacht, und das Parlament gewählt.")

    def test_false_start_inserted_as_words(self):
        r = row("Und das bedeutet zweitausendvierzehn neue Jobs und das Parlament gewählt.", "zweitausendvierzehn",
                "2014")
        p = W.plan_row(r, ext("und das machen— macht— bedeutet zweitausendvierzehn neue jobs und das parlament gewählt"),
                       DE, "de")
        res = W.decide_row(r, p, DE, "de", scorer("machen macht"))
        self.assertEqual(res["row"]["text"],
                         "Und das machen macht bedeutet zweitausendvierzehn neue Jobs und das Parlament gewählt.")

    def test_sentence_start_casing(self):
        r = row("Wir haben zweitausendvierzehn gewählt. Neue Parlament gewählt und das gesehen.", "zweitausendvierzehn",
                "2014")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn gewählt und neue parlament gewählt und das gesehen"), DE,
                       "de")
        res = W.decide_row(r, p, DE, "de", lambda t: 0.0 if t == r["text"] else -10.0)
        self.assertEqual(res["row"]["text"],
                         "Wir haben zweitausendvierzehn gewählt. Und neue Parlament gewählt und das gesehen.")

    def test_edge_differences_are_left_alone(self):
        r = row("Wir haben zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("eben wir haben zweitausendvierzehn das parlament gewählt und gesehen dann"), DE, "de")
        self.assertEqual(p["status"], "edge_only")

    def test_hesitation_is_not_a_difference(self):
        r = row("Wir haben zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("wir haben äh zweitausendvierzehn das ähm parlament gewählt und gesehen"), DE, "de")
        self.assertEqual(p["status"], "no_difference")

    def test_number_words_are_never_brought_in(self):
        r = row("Wir haben zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn das parlament zwei mal gewählt und gesehen"), DE, "de")
        self.assertEqual(p["stretches"][0]["guard"], "number")

    def test_no_joiner_between_two_number_readings(self):
        text = "Es waren zweitausend dreihundert Leute und das neue Parlament gewählt."
        a, b = text.index("zweitausend"), text.index("dreihundert")
        r = {"id": "20141013-1500-COMMITTEE-CONT_de_88_p0", "lang": "de", "text": text,
             "caption": "Es waren 2000 300 Leute und das neue Parlament gewählt.", "duration": 5.0,
             "num_chars": [[a, a + 11], [b, b + 11]],
             "spans": [{"text": "2000", "reading": "zweitausend", "kind": "int"},
                       {"text": "300", "reading": "dreihundert", "kind": "int"}],
             "audio_filepath": "/vol/x.wav", "word_times": [[0, 1]]}
        p = W.plan_row(r, ext("es waren zweitausend und dreihundert leute und das neue parlament gewählt"), DE, "de")
        self.assertTrue(all(s["guard"] for s in p["stretches"] if "und" in s["keys"]), p["stretches"])

    def test_unknown_spelling_is_not_brought_in(self):
        r = row("Wir haben zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn das parlamend gewählt und gesehen"), DE, "de")
        self.assertEqual(p["status"], "guarded")

    def test_names_keep_the_caption_spelling(self):
        r = row("dass der Präsident Waletzkis zweitausendvierzehn kommt und spricht", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("dass der präsident barroso zweitausendvierzehn kommt und spricht"), DE, "de")
        self.assertEqual(p["stretches"][0]["guard"], "name")

    def test_stretch_next_to_an_edge_is_left_alone(self):
        r = row("Dann ich noch zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("dann eben ich noch zweitausendvierzehn das parlament gewählt und gesehen"), DE, "de")
        self.assertEqual(p["stretches"][0]["guard"], "near_edge")

    def test_too_long_or_too_many(self):
        r = row("Wir haben zweitausendvierzehn das Parlament gewählt und gesehen", "zweitausendvierzehn", "2014")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn eben das neue parlament eben gewählt eben und gesehen"),
                       DE, "de")
        self.assertEqual(p["status"], "many")
        p = W.plan_row(r, ext("wir haben zweitausendvierzehn das eben neue eben dann parlament gewählt und gesehen"),
                       DE, "de")
        self.assertEqual(p["stretches"][0]["guard"], "long")


class French(unittest.TestCase):
    FR = lex("je vous rappelle mais nous sommes en avance et mais nous sommes là",
             "je remercie monsieur le président et monsieur le rapporteur")

    def test_compound_spacing_peeled(self):
        r = row("Nous sommes en deux mille dix neuf. Je vous rappelle que nous sommes là.", "deux mille dix neuf",
                "2019", "fr")
        p = W.plan_row(r, ext("nous sommes en deux mille dix-neuf mais je vous rappelle que nous sommes là"), self.FR,
                       "fr")
        self.assertEqual([s["words"] for s in p["stretches"]], [["mais"]])
        res = W.decide_row(r, p, self.FR, "fr", scorer("mais"))
        self.assertEqual(res["row"]["text"], "Nous sommes en deux mille dix neuf. Mais je vous rappelle que nous sommes là.")

    def test_name_spelling_kept(self):
        r = row("Je vous rappelle, Chandeler et nous, que nous sommes en deux mille dix neuf là", "deux mille dix neuf",
                "2019", "fr")
        p = W.plan_row(r, ext("je vous rappelle chandelier et nous que nous sommes en deux mille dix-neuf là"),
                       lex("le chandelier est là et le chandelier aussi"), "fr")
        self.assertEqual(p["stretches"][0]["guard"], "name")

    def test_abbreviation_spoken(self):
        r = row("Je remercie M. le président pour les deux mille dix neuf jours", "deux mille dix neuf", "2019", "fr")
        p = W.plan_row(r, ext("je remercie monsieur le président pour les deux mille dix-neuf jours"), self.FR, "fr")
        res = W.decide_row(r, p, self.FR, "fr", scorer("monsieur"))
        self.assertEqual(res["row"]["text"], "Je remercie monsieur le président pour les deux mille dix neuf jours")


class SpotCheckFailures(unittest.TestCase):
    def test_eh_bien_is_one_phrase(self):
        fr = lex("et eh bien nous sommes là et eh bien voilà", "les revenus à publier ne sont pas là")
        r = row("les revenus à publier ne sont pas là en deux mille dix neuf", "deux mille dix neuf", "2019", "fr")
        p = W.plan_row(r, ext("les revenus à publier eh bien ne sont pas là en deux mille dix-neuf"), fr, "fr")
        self.assertEqual(p["stretches"][0]["words"], ["eh", "bien"])

    def test_moved_word_is_not_deleted(self):
        nl = lex("de uiterste termijn is morgen en de uiterste termijn is vandaag", "uiteraard is de termijn")
        r = row("dank u wel. Uiteraard is de termijn dertig maart en dan zijn we klaar.", "dertig", "30", "nl")
        p = W.plan_row(r, ext("dank u wel uiterste termijn is 30 maart en dan zijn we klaar"), nl, "nl")
        self.assertTrue(all(s["guard"] for s in p["stretches"]), p["stretches"])


class Punctuation(unittest.TestCase):
    def test_doubled_word_deleted_without_its_full_stop(self):
        fr = lex("cinquante milliards d'euros par an et d'euros par mois", "nous avons cinquante milliards et cela")
        r = row("nous avons cinquante milliards d'euros. d'euros par an et cela nous suffit", "cinquante", "50", "fr")
        p = W.plan_row(r, ext("nous avons cinquante milliards d'euros par an et cela nous suffit"), fr, "fr")
        res = W.decide_row(r, p, fr, "fr", lambda t: 0.0 if t == r["text"] else -20.0)
        self.assertEqual(res["row"]["text"], "nous avons cinquante milliards d'euros par an et cela nous suffit")


class Dutch(unittest.TestCase):
    NL = lex("en dan zou je in het bedrijf kunnen blijven en dan zou je hier zijn")

    def test_digits_align_with_readings(self):
        r = row("Het kost honderd miljoen euro in het bedrijf kunnen blijven.", "honderd miljoen", "100 miljoen", "nl")
        p = W.plan_row(r, ext("het kost 100 miljoen euro zou je in het bedrijf kunnen blijven"), self.NL, "nl")
        self.assertEqual([s["words"] for s in p["stretches"]], [["zou", "je"]])
        self.assertIsNone(p["stretches"][0]["guard"])


class Ids(unittest.TestCase):
    def test_time_suffix_kept_last(self):
        self.assertEqual(W.wid("20180704-0900-PLENARY-18-de_20180704-18:01:49_9"),
                         "20180704-0900-PLENARY-18-de_w_20180704-18:01:49_9")
        self.assertEqual(W.wid("abc_p1"), "abc_p1_w")

    def test_reading_must_be_found_at_its_place(self):
        r = row("Wir haben eine Frage und eine Antwort", "eine", "1")
        r["num_chars"] = [[25, 29]]
        self.assertFalse(W.readings_findable(r, r["text"], r["num_chars"]))
        self.assertTrue(W.readings_findable(r, r["text"], [[10, 14]]))


if __name__ == "__main__":
    unittest.main()
