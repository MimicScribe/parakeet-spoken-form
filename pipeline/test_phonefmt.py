"""phonefmt: separators only, groups spoken-style (plain python, no deps): python test_phonefmt.py"""
import re

from phonefmt import phone_span, reformat, rewrite_text

for src, want in [("08,72, 960", "08 72 960"), ("0,498,250", "0 498 250"), ("0,137610", "0 137610"),
                  ("+421 905 123 456", "+421 905 123 456"), ("(02) 5551 2345", "02 5551 2345"),
                  ("0905-123-456", "0905 123 456"), ("0905.123.456", "0905 123 456"), ("6116232473", "6116232473")]:
    assert reformat(src) == want, (src, reformat(src))
assert reformat("6116232473", [2, 2, 2, 2, 2]) == "61 16 23 24 73" and reformat("0905123456", [4, 3, 3]) == "0905 123 456"
assert reformat("0905123456", [4, 3]) == "0905123456"  # sizes that do not add up: no change
for s in ["5 000 000", "130 000", "1,500,000", "2014-2020", "5. 9. 2021", "5.9.2021", "100-250", "1644. - 1912", "12 345", "24.8.\u20135.9.2021", "24.8.-5.9.2021", "1418. \u2013 1450.", "11 000\u201322 500", "1995./1996", "1469 - 1539"]:
    assert not phone_span(s), s
for s in ["08,72, 960", "0,498,250", "+421 905 123 456", "0905 123 456"]:
    assert phone_span(s), s
t, n = rewrite_text("Zavolajte 0905,123,456 alebo 5 000 000 eur v 2014-2020")
t2, n2 = rewrite_text("Sejong (1418\u20131450) a (1894-1895)")
assert n2 == 0 and t2 == "Sejong (1418\u20131450) a (1894-1895)"
assert t == "Zavolajte 0905 123 456 alebo 5 000 000 eur v 2014-2020" and n == 1, t
for s in ["08,72, 960", "+421 (905) 123-456", "0,137610"]:
    assert re.sub(r"\D", "", reformat(s)) == re.sub(r"\D", "", s)
t3, n3 = rewrite_text("tel 0905,123,456 a 08,72, 960", accept=lambda b, a: b.startswith("08"))
assert t3 == "tel 0905,123,456 a 08 72 960" and n3 == 1, t3
print("test_phonefmt: ok")
