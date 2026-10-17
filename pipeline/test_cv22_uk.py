"""cv22_uk span-local check (local, no GPU): python3 test_cv22_uk.py"""
import cv22_uk as V


def row(text):
    return {"id": "cv22_x", "text": text, "caption": text, "spans": [], "num_chars": [], "kinds": ["control"]}


def test_span_masked_when_not_heard():
    # red-on-base: rows were kept with every span after a whole-row CER <= 25%, a misheard number included
    r = V.span_check(row("А вони понабирали по двадцять, по тридцять!"), "а вони понабирали по двадцять по сорок")
    assert [s["text"] for s in r["spans"]] == ["20"] and r["spans_masked"] == ["тридцять"]
    a, b = r["num_chars"][0]
    assert r["text"][a:b] == "двадцять"
    assert r["kinds"] == ["int"]


def test_all_masked_row_is_not_a_control():
    r = V.span_check(row("Професор знав, що їх не вистачить на два місяці."), "професор знав що їх не вистачить на місяці")
    assert r["spans"] == [] and r["kinds"] == ["masked"]


def test_heard_spans_kept_and_retagged():
    r = V.span_check(row("Це тривало близько двадцяти чотирьох секунд."), "це тривало близько двадцяти чотирьох секунд")
    assert [(s["text"], s["kind"]) for s in r["spans"]] == [("24", "int")] and r["spans_masked"] == []


if __name__ == "__main__":
    test_span_masked_when_not_heard()
    test_all_masked_row_is_not_a_control()
    test_heard_spans_kept_and_retagged()
    print("ok")
