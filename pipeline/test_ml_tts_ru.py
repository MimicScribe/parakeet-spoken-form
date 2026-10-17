"""Local tests for ml_tts_ru (no GPU, no network): python3 test_ml_tts_ru.py"""
import collections
import re

import ml_cands as M
import ml_p3_core as C
import ml_tts_ru as U

ROWS = U.plan(700, 1)


def test_every_class_filled():
    have = collections.Counter(r["cls"] for r in ROWS)
    tot = sum(U.CLASSES.values())
    for c, w in U.CLASSES.items():
        assert have[c] >= round(700 * w / tot) - 1, (c, have[c])


def test_labels_spoken_only():
    for r in ROWS:
        assert not U.FORBIDDEN.search(r["text"]) and not C.WF.search(r["text"]), r["text"]
        assert all(re.search(r"\w", t) for t in r["text"].split()), r["text"]  # no "—" token (CTC alignment)
        assert r["text"][-1] in ".!?", r["text"]
        assert U.WORDS_MIN <= len(r["text"].split()) <= U.WORDS_MAX, r["text"]


def test_offsets_and_spans_match_caption():
    for r in ROWS:
        sp = M.spans(r["caption"], "ru")
        assert [s.kind for s in sp] == [s["kind"] for s in r["spans"]], r["caption"]
        for (a, b), s in zip(r["num_chars"], r["spans"]):
            assert r["text"][a:b].lower() == s["reading"].lower(), (r["text"], s)
            assert s["reading"] in [c.strip() for c in M.spans(r["caption"], "ru")[r["spans"].index(s)].cands]


def test_case_forms():
    # red-on-base: control (new module)
    assert U.ordw(2025, "m", "loc") == "две тысячи двадцать пятом"
    assert U.ordw(1998, "m", "gen") == "тысяча девятьсот девяносто восьмого"
    assert U.ordw(25, "n", "gen") == "двадцать пятого"
    assert U._amount("рубль", 1, "nom") == "один рубль"
    assert U._amount("рубль", 2, "nom") == "два рубля"
    assert U._amount("рубль", 240, "nom") == "двести сорок рублей"
    assert U._amount("копейка", 21, "nom") == "двадцать одна копейка"
    assert U._amount("копейка", 2, "nom") == "две копейки"
    assert U._amount("доллар", 3, "nom") == "три доллара"
    assert U._amount("цент", 50, "nom") == "пятьдесят центов"
    assert U._amount("евро", 5, "nom") == "пять евро"
    assert U._amount("процент", 1, "gen") == "одного процента"
    assert U.card(5, "gen") == "пяти"
    assert U.card(1500, "acc") == "тысячу пятьсот"
    assert U.card(2000000, "acc") == "два миллиона"
    assert U.card(10, "acc", "f") == "десять" and U.card(1, "acc", "f") == "одну"
    assert U._dec_reading(_R(0.9), 3, "5") == "три целых пять десятых"
    assert U._dec_reading(_R(0.9), 1, "1") == "одна целая одна десятая"
    assert U._dec_reading(_R(0.9), 2, "25") == "две целых двадцать пять сотых"
    assert U._dec_reading(_R(0.1), 3, "5") == "три запятая пять"
    assert U._suffix("m", "loc") == "м" and U._suffix("n", "nom") == "е" and U._suffix("n", "gen") == "го"


class _R:
    """A stand-in rng whose random() is fixed (picks the decimal style)."""

    def __init__(self, v):
        self.v = v

    def random(self):
        return self.v


def test_year_spread():
    ys = collections.Counter(int(s["text"]) for r in ROWS if r["cls"] == "year" for s in r["spans"])
    assert sum(1 for y in range(1990, 2031) if ys[y]) >= 38, sorted(ys)
    assert sum(1 for y in range(2031, 2061) if ys[y]) >= 15, sorted(ys)
    assert max(ys.values()) <= 3


def test_phone_values_distinct():
    ph = [r["caption"] for r in ROWS if r["cls"] == "phone"]
    nums = {re.search(r"8-[\d-]+|\(\d{3}\) [\d-]+", c).group(0) for c in ph}
    assert len(nums) == len(ph)


def test_price_currencies_and_cents():
    pr = [r for r in ROWS if r["cls"] == "price"]
    for sym in ("₽", "$", "€"):
        assert sum(sym in r["caption"] for r in pr) >= 5, sym
    assert sum("копе" in r["text"] for r in pr) >= 5 and sum("цент" in r["text"] for r in pr) >= 3


def test_dev_split_and_determinism():
    dev = sum(r["split"] == "dev" for r in ROWS)
    assert 0.04 * len(ROWS) <= dev <= 0.18 * len(ROWS), dev
    again = U.plan(700, 1)
    assert [r["id"] for r in again] == [r["id"] for r in ROWS]


def test_lead_never_before_imperative():
    import random

    rng = random.Random(0)
    for _ in range(50):
        assert U._lead(rng, "Звоните по номеру 8-916-123-45-67.").startswith("Звоните")
        t = U._lead(rng, "Мой сын учится в 7-м классе.")
        assert not t.startswith(tuple(U.LEADS_REPORT)), t


def test_strip_parens():
    lab, offs = U._strip_parens("по номеру (четыреста девяносто пять) сто два", [[11, 35], [37, 44]])
    assert lab == "по номеру четыреста девяносто пять сто два" and lab[offs[0][0]:offs[0][1]] == "четыреста девяносто пять"
    assert lab[offs[1][0]:offs[1][1]] == "сто два"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
