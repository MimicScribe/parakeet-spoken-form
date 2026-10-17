"""English rare-shape caption scan (no Modal): python3 test_en_hits.py"""
import en_hits as H

assert H.hits_of("call us on 555-1234 today") == ["phone"]
assert "phone" in H.hits_of("ring 0161 496 0000") and "phone" in H.hits_of("dial 1-800-555-0199")
assert "phone" in H.hits_of("at (212) 555-1234")
for t in ("in 2019 we", "it cost $1,000", "about 3.5 million", "from 1990-2000", "page 12"):
    assert "phone" not in H.hits_of(t), t
assert "clock" in H.hits_of("see you at 3:45") and "clock" in H.hits_of("at 7pm") and "clock" in H.hits_of("10 a.m. sharp")
assert "clock" in H.hits_of("at four o'clock") and "clock" not in H.hits_of("a 3.5 rating")
assert "clock_words" in H.hits_of("half past three") and "clock_words" in H.hits_of("quarter to four")
assert "clock_words" in H.hits_of("twenty to six") and "clock_words" not in H.hits_of("twenty to thirty people")
assert H.hits_of("oh two nine double four seven") == ["double"] and H.hits_of("nought point five") == ["nought"]
rows = [{"utt_id": f"u{i}", "duration": 5.0, "text": t, "original_audio_id": v} for i, (t, v) in
        enumerate([("at 3:45", "a"), ("at 4:15", "a"), ("at 5:15", "a"), ("at 6:15", "a"), ("double two", "b"),
                   ("nothing", "c")])]
hs = H.scan_rows(rows, "s")
assert len(hs) == 5 and all(h["shard"] == "s" for h in hs)
sel = H.pick(hs, per_video=2)
assert [h["utt_id"] for h in sel] == ["u4", "u0", "u1"], [h["utt_id"] for h in sel]  # rarest shape first, 2 per video
print("ok")
assert "clock_words" not in H.hits_of("for five to six years") and "clock_words" not in H.hits_of("from quarter to quarter")
assert "clock_words" in H.hits_of("By ten past eight, we") and "clock_words" in H.hits_of("leave at twenty to six.")
print("ok2")
# rare shapes (2026-10-17)
assert "roman" in H.hits_of("World War II ended") and "roman" not in H.hits_of("The XL size")
assert "amp" in H.hits_of("AT&T and R&D") and "amp" in H.hits_of("Johnson & Johnson")
assert "frac" in H.hits_of("add 3/4 cup") and "frac" in H.hits_of("½ cup") and "frac" not in H.hits_of("on 12/25 we")
assert "version" in H.hits_of("iOS 17.4 is out") and "version" in H.hits_of("v1.2.3") and "version" not in H.hits_of("3.5 stars")
assert H.hits_of("añade 1/2 taza", "es") == ["frac"] and H.hits_of("AT&T", "fr") == ["amp"]
assert H.hits_of("World War II", "es") == []  # Roman numerals are an English-only shape here
assert H.hits_of("add 3/4 cup and call 555-1234", kinds=("frac",)) == ["frac"]
print("ok3")
# es / fr phone, clock and price shapes (2026-10-17)
for t in ("llame al 912 345 678", "mi móvil es 612345678", "al +34 612 345 678 ya", "ring 55 1234 5678"):
    assert "phone" in H.hits_of(t, "es"), t
for t in ("el 12-03-2020 fue", "pagó 3,5 millones", "en 2019 hubo", "el 15 de marzo", "son 1.234.567 personas"):
    assert "phone" not in H.hits_of(t, "es"), t
for t in ("appelez le 01 23 45 67 89", "le 06.12.34.56.78", "au +33 6 12 34 56 78", "le 0800 123 456"):
    assert "phone" in H.hits_of(t, "fr"), t
for t in ("le 12/03/2020", "il a 3,50 de moyenne", "en 2019"):
    assert "phone" not in H.hits_of(t, "fr"), t
assert "cur" in H.hits_of("cuesta 3,50 euros", "es") and "cur" in H.hits_of("son 12,99 €", "es") and "cur" not in H.hits_of("3,5 millones", "es")
assert "cur" in H.hits_of("ça coûte 4,99 €", "fr") and "cur" in H.hits_of("12 € 50", "fr") and "cur" not in H.hits_of("le 3,5 pour cent", "fr")
assert "clock" in H.hits_of("a las 10:30", "es") and "clock" in H.hits_of("a las 8", "es") and "clock" not in H.hits_of("3,5 de media", "es")
assert "clock" in H.hits_of("à 14h30", "fr") and "clock" in H.hits_of("vers 9:15 ce soir", "fr") and "clock" not in H.hits_of("la page 12", "fr")
print("ok4")
assert H.hits_of("hit us up at 866-846-5278 now", kinds=("phone10",)) == ["phone10"] and H.hits_of("ISBN 81-208-0223-3", kinds=("phone10",)) == []
