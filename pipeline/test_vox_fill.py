"""Tests for vox_fill (targeted parliament selection)."""

import vox_fill as V


def test_scale_words():
    assert V.classes("Wir geben 2 Milliarden Euro aus.", "de")["scale_b"]
    assert V.classes("Es sind 3,5 Millionen Menschen.", "de")["scale_m"]
    assert V.classes("Sono 1,5 miliardi di euro.", "it")["scale_b"]
    assert V.classes("Ci sono 20 milioni di persone.", "it")["scale_m"]
    assert V.classes("Het gaat om 2 miljard euro.", "nl")["scale_b"]
    assert V.classes("Ruim 5 miljoen mensen.", "nl")["scale_m"]
    c = V.classes("São 3 mil milhões de euros.", "pt")
    assert c["scale_b"] and not c["scale_m"]
    assert V.classes("São 3 milhões de pessoas.", "pt")["scale_m"]
    assert not V.classes("Es gab 3000 Fälle.", "de")["scale_m"]


def test_dates():
    assert V.classes("Wir treffen uns am 6. Dezember in Brüssel.", "de")["date"]
    assert not V.classes("Das ist der 6. Bericht.", "de")["date"]
    assert V.classes("La riunione del 3 maggio.", "it")["date"]
    assert V.classes("Il 1° maggio.", "it")["date"]
    assert V.classes("Op 12 maart stemmen we.", "nl")["date"]
    assert V.classes("A reunião de 3 de maio.", "pt")["date"]
    assert not V.classes("Temos 3 maiores problemas.", "pt")["date"]


def test_it_clock():
    assert V.classes("La votazione si terrà alle 15.30.", "it")["clock"]
    assert V.classes("Dalle 9 alle 12.30.", "it")["clock"]
    assert not V.classes("Siamo arrivati alle 15 per cento.", "it")["clock"]
    assert not V.classes("Alle 15.30 Uhr.", "de")["clock"]


def _row(i, text, day="20150101", sess="20150101-0900-PLENARY"):
    return {"utt_id": f"{sess}_{i}", "text": text, "day": day}


def test_select_quotas_caps_and_exclusions():
    rows = [_row(i, f"Wir geben {i} Milliarden Euro aus und das ist viel Geld für alle.", sess=f"20150101-S{i % 50}")
            for i in range(300)]
    rows += [_row(1000 + i, f"Am {i % 28 + 1}. Mai {i} reden wir hier im Haus über das wichtige Thema.", day="20160101",
                  sess=f"20160101-S{i}") for i in range(100)]
    rows.append(_row(5000, "Am 3. Mai 2015 reden wir über ein ganz anderes Thema.", day="20990101", sess="20990101-X"))
    held = {"20990101"}
    exclude = {rows[0]["utt_id"]}
    sel, st = V.select(rows, "de", {"scale_b": 40, "date": 30, "scale_m": 0}, held, exclude, per_template=99,
                       per_session=2)
    ids = {r["utt_id"] for r in sel}
    assert rows[0]["utt_id"] not in ids and "20990101-X_5000" not in ids
    assert st["classes"]["date"] >= 30 and st["classes"]["scale_b"] >= 40
    assert "scale_m" not in st["quotas"]
    per_sess = {}
    for r in sel:
        per_sess[V.session(r["utt_id"])] = per_sess.get(V.session(r["utt_id"]), 0) + 1
    assert max(per_sess.values()) <= 2


def test_select_years_per_lang():
    rows = [_row(i, f"Es geht um {i} Millionen Menschen in ganz Europa heute.", day=f"20{10 + i % 6}0101",
                 sess=f"20{10 + i % 6}0101-S{i}") for i in range(120)]
    sel, st = V.select(rows, "de", {"scale_m": 100}, set(), set(), years_per_lang=2, per_template=99)
    assert len({r["day"][:4] for r in sel}) <= 2 and len(st["debate_years"]) == 2


def test_template_cap():
    rows = [_row(i, "Wir geben 5 Milliarden Euro aus und das ist viel Geld.", sess=f"S{i}") for i in range(20)]
    sel, _ = V.select(rows, "de", {"scale_b": 20}, set(), set(), per_template=3)
    assert len(sel) == 3


def test_pilot_round_robin():
    sel = [{"utt_id": str(i), "why": "date" if i % 3 else "scale_b"} for i in range(30)]
    p = V.pilot(sel, 10)
    assert len(p) == 10 and sum(r["why"] == "scale_b" for r in p) == 5


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
