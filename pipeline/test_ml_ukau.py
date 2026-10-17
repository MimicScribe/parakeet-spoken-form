"""UK / AU spoken forms: every label parses back to the value it means (number reader: score_retrain's CORE block).
python3 test_ml_ukau.py"""
import decimal
import re

import ml_ukau as U

import score_retrain as NO  # noqa: E402  (same CORE block as the evaluation copy)

rows = U.rows({f: 60 for f in U.FORMS}, 3)
assert {f: sum(r["cls"] == f for r in rows) for f in U.FORMS} == {f: 60 for f in U.FORMS}
assert len({r["text"] for r in rows}) == len(rows)
assert {"train", "dev"} == {r["split"] for r in rows}
for r in rows:
    assert not re.search(r"\d", r["text"]), r["text"]
    vals = [sorted(n.vals)[0] for n in NO.extract_numbers(r["text"], "en")]
    if r["cls"] == "phone":
        assert [int(v) for v in vals] == [int(c) for c in r["value"]], (r["text"], vals, r["value"])
    elif r["cls"] == "nought":
        assert decimal.Decimal(r["value"]) in vals, (r["text"], vals)
    elif r["cls"] == "money":
        v = r["value"]
        want = decimal.Decimal(v[1:]) if v[0] in "£$" else decimal.Decimal(v[:-1])
        assert want in vals, (r["text"], vals, v)
    elif r["cls"] == "clock":
        h = int(r["value"].split(":")[0])
        hh = h % 12 + 1 if " to " in r["text"] else h
        assert decimal.Decimal(hh) in vals, (r["text"], vals)
    elif r["cls"] == "date":
        d = int(r["value"].split()[0])
        assert decimal.Decimal(d) in vals and r["value"].split()[1] in r["text"], (r["text"], vals)
assert any("double" in r["text"] for r in rows if r["cls"] == "phone")
assert any(" oh " in r["text"] for r in rows if r["cls"] == "phone")
assert "one pounds" not in " ".join(r["text"] for r in rows)
print("ok")
