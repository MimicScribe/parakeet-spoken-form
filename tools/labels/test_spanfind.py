import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spanfind as S


def one(t, i=0):
    sp = S.find_spans(t)
    return sp[i]


def test_cardinal_digits_pairs_decimal_ordinal_decade():
    a = one("held forty four thousand two hundred books"); assert a.kind == "int" and a.value == "44200"
    b = one("call five five five oh one four seven now"); assert b.kind == "digits" and b.value == "5550147"
    c = one("in twenty fifteen"); assert c.kind == "pairs" and c.value == "2015"
    d = one("ten thirty AM"); assert d.value == "1030" and d.suffix == ["AM"]
    e = one("five point two percent"); assert e.kind == "decimal" and e.value == "5.2" and e.suffix == ["percent"]
    f = one("the thirty first"); assert f.kind == "ordinal" and f.value == "31"
    g = one("in the nineties"); assert g.kind == "decade"
    h = one("twelve hundred and five"); assert h.value == "1205"


def test_codes_and_letters():
    s = one("the B two B deal"); assert s.reading == "B two B" and s.prefix == ["B"] and s.suffix == ["B"]
    assert "A" not in one("A two story house").prefix                        # the article is never a code letter


def test_commas_are_stripped_from_readings_and_candidates():
    s = one("Dial two three nine, two one eight, please.")
    assert "," not in s.reading and s.reading == "two three nine two one eight"
    assert all("," not in c for c in S.candidates(s))
    lab = "Dial two three nine, two one eight, please."
    assert S.apply_candidate(lab, s, s.reading) == "Dial two three nine two one eight, please."
    assert S.strip_inner_commas(lab, S.find_spans(lab)) == "Dial two three nine two one eight, please."


def test_candidates_current_first_oh_zero_double_and_repeat_variants():
    s = one("the pin is one oh five")
    c = S.candidates(s)
    assert c[0] == "one oh five" and "one zero five" in c and "one hundred five" in c and "one hundred and five" in c
    r = one("ring five five oh one"); cr = S.candidates(r)
    assert "five five five oh one" in cr and "five oh one" in cr and "double five oh one" in cr


def test_case_variants_are_deduplicated_and_capitalisation_follows_the_current_reading():
    c = S.candidates(one("And Forty Five of them"))
    assert c[0] == "Forty Five" and len([x for x in c if x.lower() == "forty five"]) == 1
    c2 = S.candidates(one("A thousand experiments"))
    assert all(x[:1].isupper() for x in c2)


def test_generated_flags_compositional_readings():
    s = one("call eight hundred nine two seven eight nine two zero now")
    assert S.generated(s)
    assert S.generated(one("two thousand and five"))


def test_weak_flag_and_unparsed():
    assert one("one of the best").weak
    assert not one("fifteen miles").weak
    assert S.find_spans("point of view") == []


def test_disfluent_repeat_does_not_crash():
    for s in S.find_spans("found ten million ten million thirty one thousand votes"):
        S.candidates(s); S.generated(s)


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
