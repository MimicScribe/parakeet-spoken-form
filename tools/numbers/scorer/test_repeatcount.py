#!/usr/bin/env python3
"""golden pairs for repeatcount.py. Each row: (ref canonical string, hypothesis span, expected class). Negatives are the point: a hypothesis that DROPS a copy
('two two' vs 'two') must FAIL, 'double two' must equal 'two two', and the reader must not invent value ('in twenty five' != 2005, 'sixty si' != 66, 'twenty' != '2').
  python3 -I test_repeatcount.py         # prints the table and exits 1 on any miss"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repeatcount as RC

E = "exact"
GOLD = [
    # --- the brief's anchors
    ("22", "two two", E), ("22", "double two", E), ("22", "22", E), ("22", "two", "repeat_drop"), ("2", "two two", "repeat_add"), ("22", "double two two", "repeat_add"),
    ("222", "triple two", E), ("222", "two two two", E), ("222", "double two", "repeat_drop"), ("222", "two two two two", "repeat_add"), ("222", "222", E), ("222", "22", "repeat_drop"),
    ("2222", "quadruple two", E), ("2222", "double two double two", E), ("2222", "triple two two", E), ("2222", "two two two", "repeat_drop"), ("2222", "triple two", "repeat_drop"),
    ("55555", "five five five five five", E), ("55555", "triple five double five", E), ("55555", "double five triple five", E), ("55555", "five five five five", "repeat_drop"),
    ("000", "triple oh", E), ("000", "zero zero zero", E), ("000", "oh oh oh", E), ("000", "triple zero", E), ("000", "nought nought nought", E), ("000", "double oh", "repeat_drop"),
    ("007", "double oh seven", E), ("007", "double-oh-seven", E), ("007", "zero zero seven", E), ("007", "oh oh seven", E), ("007", "oh seven", "repeat_drop"), ("007", "7", "repeat_drop"),
    # --- grouping / form blindness
    ("2233", "twenty two thirty three", E), ("2233", "double two double three", E), ("2233", "22 33", E), ("2233", "22-33", E), ("2233", "2,233", E), ("2233", "two two three three", E),
    ("2233", "2233", E), ("2233", "twenty two thirty", "other_value"), ("2233", "two three", "repeat_drop"), ("2233", "twenty three", "repeat_drop"),
    ("3232", "thirty two thirty two", E), ("3232", "three two three two", E), ("3232", "thirty two", "period_drop"), ("3232", "3232", E), ("3232", "32", "period_drop"),
    ("2020", "twenty twenty", E), ("2020", "two zero two zero", E), ("2020", "twenty", "period_drop"), ("2020", "two thousand twenty", E), ("2020", "2020", E),
    ("1111", "eleven eleven", E), ("1111", "one one one one", E), ("1111", "eleven", "repeat_drop"), ("11", "eleven", E), ("11", "one one", E),
    ("8776642233", "eight seven seven six six four two two three three", E), ("8776642233", "eight double seven double six four double two double three", E),
    ("8776642233", "877 664 2233", E), ("8776642233", "877-664-2233", E), ("8776642233", "eight seven six six four two two three three", "repeat_drop"),
    ("8776642233", "eight seven seven six six four two two three", "repeat_drop"), ("8776642233", "eight seven seven six six four two two two three three", "repeat_add"),
    # --- abab / abcabc
    ("2121", "two one two one", E), ("2121", "twenty one twenty one", E), ("2121", "two one", "period_drop"), ("2121", "twenty one", "period_drop"), ("212121", "two one two one", "period_drop"),
    ("123123", "one two three one two three", E), ("123123", "one two three", "period_drop"), ("123123", "one two three one two three one two three", "period_add"),
    # --- near repeats (minimal pairs must not be conflated)
    ("2233", "two two three", "repeat_drop"), ("233", "two two three three", "repeat_add"), ("2233", "two three three", "repeat_drop"), ("2223", "two two three", "repeat_drop"),
    ("2223", "two two two three", E), ("2233", "two two two three", "repeat_mixed"), ("233", "two two three", "repeat_mixed"), ("22", "two three", "other_value"),
    # --- homophones only inside a run
    ("2244", "two two for four", E), ("2244", "to two four four", E), ("4444", "four four for for", E), ("22", "to two", E), ("42", "for two", E),
    ("5", "for five", "other_value"),
    # --- reader must not invent value
    ("2005", "in twenty five", "repeat_drop"), ("66", "sixty si", "other_value"), ("2", "twenty", "other_value"), ("20", "two", "other_value"), ("17", "seven", "other_value"),
    ("5", "fifty", "other_value"), ("50", "five", "other_value"), ("500", "five", "repeat_drop"), ("202", "twenty two", "other_value"), ("22", "two hundred two", "other_value"),
    ("0", "o", "slot_dropped"), ("905", "nine o five", E), ("905", "nine oh five", E), ("905", "nine zero five", E), ("905", "nine five", "other_value"),
    # --- dropped slot / empty
    ("22", "", "slot_dropped"), ("22", "the meter is fine", "slot_dropped"), ("22", "oh I see", "slot_dropped"),
    # --- decimals / punctuation are separators inside a run
    ("2929", "$29.29", E), ("2929", "twenty nine twenty nine", E), ("2929", "twenty nine", "period_drop"), ("5501", "55 01", E),
    # --- digits form vs words form are the same (form blind)
    ("4477", "4477", E), ("4477", "44 77", E), ("4477", "447", "repeat_drop"), ("4477", "44477", "repeat_add"), ("4477", "4 4 7 7", E), ("4477", "four four seven seven", E),
    ("4477", "forty four seventy seven", E), ("4477", "forty four seven seven", E), ("4477", "four seven", "repeat_drop"),
]

ALNUM = [  # letter + digit codes (class H): mode alnum
    ("TT447", "double T four four seven", E), ("TT447", "TT447", E), ("TT447", "T T 4 4 7", E), ("TT447", "tee tee four four seven", E), ("TT447", "double tee four double seven", "repeat_mixed"),
    ("TT447", "T 447", "repeat_drop"), ("TT447", "double t four four seven", E), ("AAA12", "triple A one two", E), ("AAA12", "double A one two", "repeat_drop"), ("AAA12", "AAA12", E),
    ("KK77", "kilo kilo seven seven", E), ("KK77", "kilo seven seven", "repeat_drop"), ("KK77", "double kilo double seven", E), ("KK77", "K K 7 7", E), ("KK77", "kk77", E),
    ("KK77", "kilo kilo seven", "repeat_drop"), ("KK5009", "KK-5009", E), ("KK5009", "KK 5009", E), ("NT770", "NT-770", E), ("TT447", "TT-447", E), ("KK5009", "KK-509", "repeat_drop"), ("BDF99", "bee dee eff nine nine", E), ("BDF99", "B D F 9 9", E), ("BDF99", "bdf99", E),
]

SEQ = [  # self-correction: [retracted, corrected], both required in order
    (["447", "4447"], "four four seven sorry four four four seven", E), (["447", "4447"], "447 sorry 4447", E), (["447", "4447"], "four four four seven", "retraction_dropped"),
    (["447", "4447"], "four four seven", "correction_dropped"), (["447", "4447"], "four four seven no wait four four seven", "repeat_drop"),
    (["447", "4447"], "four seven sorry four four four seven", "repeat_drop"), (["2200", "220"], "two two zero zero I mean two two zero", E), (["2200", "220"], "two two zero zero I mean two zero", "repeat_drop"),
]

SPAN = [  # (ref chunk text, slot text inside it, hyp chunk, expected located span as canonical string)
    ("The meter reads two two three three and it is stable.", "two two three three", "The meter reads 2233 and it is stable.", "2233"),
    ("The meter reads two two three three and it is stable.", "two two three three", "The meter reads two three and it is stable.", "23"),
    ("The meter reads two two three three and it is stable.", "two two three three", "The meter reads and it is stable.", ""),
    ("Call me on eight seven seven, six six four, two two three three tomorrow.", "eight seven seven, six six four, two two three three", "Call me on 877-664-2233 tomorrow.", "8776642233"),
    ("The code is four four seven, sorry, four four four seven, please.", "four four seven, sorry, four four four seven", "The code is 447, sorry, 4447, please.", "4474447"),
    ("Then oh I see the meter shows double two today.", "double two", "Then oh I see the meter shows 22 today.", "22"),
]

OTHER = [
    (["fife fife zero", "five five zero"], "Over the radio fife\u00a1 fife zero now.", True), (["nine one one", "911"], "Dial nine\u00a1 one\u00bf one now.", True), (["four by eight", "4x8"], "\u00bfCut 4 by 8? ok", True),  # (accept forms, hyp chunk, expected match)
    (["twenty four seven", "24/7", "24 7"], "We run it twenty four seven for them.", True), (["twenty four seven", "24/7"], "We run it 24/7 for them.", True),
    (["twenty four seven", "24/7"], "We run it twenty four for them.", False), (["nine to five", "9 to 5", "9-5"], "He works 9-5 at the depot.", True), (["nine to five", "9 to 5"], "He works nine to five at the depot.", True),
    (["nine to five", "9 to 5", "9-5"], "He works nine at the depot.", False), (["nine one one", "911"], "Dial 911 now.", True), (["nine one one", "911"], "Dial nine one now.", False),
    (["two point five lakh", "250000", "2.5 lakh"], "It was about two point five lakh rupees.", True), (["two point five lakh", "250000", "2.5 lakh"], "It was about 250,000 rupees.", True),
    (["two point five lakh", "250000", "2.5 lakh"], "It was about twenty five lakh rupees.", False), (["two by four", "2x4", "2 by 4"], "Cut a 2x4 for the frame.", True),
    (["two by four", "2x4"], "Cut a two by four for the frame.", True), (["two by four", "2x4"], "Cut a two by six for the frame.", False),
    (["one nine two decimal one six eight", "192.168"], "Go to 192.168 first.", True), (["four stroke seven", "4/7"], "Item 4/7 is late.", True), (["four stroke seven", "4/7"], "Item 47 is late.", False),
]


CROSS = [  # all surfaces of one expression must canonicalise to ONE string (other forms: a system may write digits where the reference has words)
    ["three point two times ten to the minus five", "3.2e-5", "3.2 x 10^-5", "3.2\u00d710\u207b\u2075", "3.2 times 10 to the minus 5"],
    ["three point two times ten to the sixth", "3.2e6", "3.2 x 10^6", "3.2 times 10 to the 6th"], ["ten to the sixth", "10^6", "10 to the 6th"], ["two to the power of ten", "2^10", "2**10", "2 to the power of 10"],
    ["two point five lakh", "2.5 lakh", "250000", "250,000"], ["one crore twenty lakh", "1 crore 20 lakh", "12000000", "1,20,00,000", "1.2 crore"], ["forty seven lakh", "47 lakh", "4700000", "47,00,000"],
    ["one nine two decimal one six eight decimal oh decimal one", "192.168.0.1"], ["two by four", "2x4", "2 x 4", "2 by 4"], ["channel one niner", "channel 19"], ["four stroke seven", "4/7", "4 stroke 7"],
    ["nought point five", "0.5"]]
CROSS_NEG = [("2.5 lakh", "25 lakh"), ("3.2e-5", "3.2e-6"), ("2x4", "2x6"), ("12000000", "1200000"), ("10^6", "10^5"), ("192.168.0.1", "192.168.1.0")]


MARKED = [  # hypotheses carrying counted repeat markers: stripped before reading, reported apart (reference stays UNMARKED). Final set: copy1 '\u00a1', copy2 '\u00bf', copy3+ '\u00a1\u00a1'
    ("22", "two\u00a1 two", E), ("222", "two\u00a1 two\u00bf two", E), ("2222", "two\u00a1 two\u00bf two\u00a1\u00a1 two", E), ("22222", "two\u00a1 two\u00bf two\u00a1\u00a1 two\u00a1\u00a1 two", E),
    ("222", "two\u00a1 two", "repeat_drop"), ("22", "two\u00a1 two\u00bf two", "repeat_add"), ("22", "2\u00a12", E), ("2233", "two\u00a1 two three\u00a1 three", E), ("22", "double two", E),
    ("2222", "two\u00a1 two\u00bf two", "repeat_drop"), ("22", "two\u00a1two", E), ("22", "two \u00a1 two", E), ("0", "\u00bfzero", E), ("5", "\u00bfcinco? five", E)]
CHUNK = [  # (ref chunk, slot text, truth, hyp chunk, expect exact): neighbours of a slot must never be read as digits; a misheard digit word INSIDE the slot may be rescued
    ("The code is five five two to the left.", "five five two", "552", "The code is five five two to the left.", True),
    ("The code is five five two to the left.", "five five two", "552", "The code is 552 to the left.", True),
    ("Go up to two two now.", "two two", "22", "Go up to two two now.", True),
    ("Go up to two two now.", "two two", "22", "Go up to 22 now.", True),
    ("The code is five five two, then stop.", "five five two", "552", "The code is five five two, oh, then stop.", True),
    ("Thanks. The code is five five two.", "five five two", "552", "Oh. The code is five five two.", True),
    ("The code is five five two and that is all.", "five five two", "552", "The code is five five two and that is all. Oh thanks", True),
    ("The pin is four four four now.", "four four four", "444", "The pin is four four for now.", True),
    ("The pin is four four four now.", "four four four", "444", "The pin is four four now.", False),
    ("The pin is four four four now.", "four four four", "444", "The pin is four four for for now.", False),
    ("The pin is five and the room is five.", "five", "5", "The pin is five and the room is five.", True),
    ("Call two two now.", "two two", "22", "\u00bfCall? Call two\u00a1 two now.", True),
    ("Call me on five five two now.", "five five two", "552", "Call me on five and five two now.", False),
    ("Call me on two two now.", "two two", "22", "Call me on two uh two now.", True),
    ("The code is four oh four now.", "four oh four", "404", "The code is 404 now.", True),
    ("Open at nine now.", "nine", "9", "Oh, open at nine now.", True)]
UNIT = [  # marker edge cases
    ("22", "two\u00a1two", E), ("22", "two\u2081 two\u2082", E), ("22", "two\u00a1 two", E), ("222", "two\u00a1 two\u00bf two", E), ("222", "two\u00a1\u00a1 two\u00a1\u00a1 two", E)]


def main():
    bad = 0
    for ref_text, slot_text, truth, hyp_text, want in CHUNK:
        a = ref_text.index(slot_text)
        r = RC.score_chunk(ref_text, (a, a + len(slot_text)), truth, hyp_text, "digits")
        if r["exact"] != want:
            bad += 1
            print("MISS [chunk]", repr(hyp_text), "want exact", want, "got", r["cls"], repr(r.get("span")))
    for ref, hyp, want in UNIT:
        r = RC.score_value(ref, hyp)
        if r["cls"] != want:
            bad += 1
            print("MISS [marker unit]", ref, repr(hyp), want, r["cls"])
    for ref, hyp, want in MARKED:
        r = RC.score_value(ref, hyp)
        if r["cls"] != want:
            bad += 1
            print("MISS [marked]", ref, repr(hyp), want, r["cls"])
    m = RC.marker_report("two\u00a1 two\u00bf two\u00a1\u00a1 two", 3, copies=4)
    if not (m["n_markers"] == 3 and m["as_expected"] and m["seq_ok"] and m["marks"] == ["\u00a1", "\u00bf", "\u00a1\u00a1"]):
        bad += 1
        print("MISS [marker_report]", m)
    r = RC.score_chunk("The pin is two two two now.", (11, 24), "222", "The pin is two\u00a1 two\u00bf two now.", "digits", 2)
    if not (r["exact"] and r["markers"]["n_markers"] == 2):
        bad += 1
        print("MISS [marked chunk]", r)
    for g in CROSS:
        if len({RC.canon_other(x) for x in g}) != 1:
            bad += 1
            print("MISS [cross]", [(x, RC.canon_other(x)) for x in g])
    for a, b in CROSS_NEG:
        if RC.canon_other(a) == RC.canon_other(b):
            bad += 1
            print("MISS [cross-neg] must differ:", a, b)
    for mode, table in (("digits", GOLD), ("alnum", ALNUM)):
        for ref, hyp, want in table:
            r = RC.score_value(ref, hyp, mode)
            if r["cls"] != want:
                bad += 1
                print(f"MISS [{mode}] ref={ref!r} hyp={hyp!r} want={want} got={r['cls']} (hyp_best={r['hyp']!r})")
    for refs, hyp, want in SEQ:
        r = RC.score_sequence(refs, hyp)
        if r["cls"] != want:
            bad += 1
            print(f"MISS [seq] refs={refs} hyp={hyp!r} want={want} got={r['cls']} {r}")
    for ref_text, slot_text, hyp_text, want in SPAN:
        a = ref_text.index(slot_text)
        span, info = RC.locate_span(ref_text, (a, a + len(slot_text)), hyp_text)
        got = "".join(sorted(RC.joined_candidates(span))[:1])
        if got != want:
            bad += 1
            print(f"MISS [span] {hyp_text!r} want={want!r} got={got!r} span={span!r}")
    for acc, hyp, want in OTHER:
        r = RC.score_other(acc, hyp)
        if r["exact"] != want:
            bad += 1
            print(f"MISS [other] accept={acc} hyp={hyp!r} want={want} got={r}")
    n = len(GOLD) + len(ALNUM) + len(SEQ) + len(SPAN) + len(OTHER) + len(CROSS) + len(CROSS_NEG) + len(MARKED) + 2 + len(CHUNK) + len(UNIT)
    # selftest: every reference string scores exact against its own canonical spoken forms (plain digits)
    for ref in ("22", "222", "2233", "8776642233", "007", "000", "123123"):
        assert RC.score_value(ref, " ".join(ref), "digits")["exact"], ref
    print(f"reader={RC.reader_id()} golden pairs={n} misses={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
