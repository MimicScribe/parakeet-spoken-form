"""Local tests for ml_tts_uk (no GPU, no network): python3 test_ml_tts_uk.py"""
import collections
import re

import ml_cands as M
import ml_p3_core as C
import ml_tts_uk as U

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
        sp = M.spans(r["caption"], "uk")
        assert [s.kind for s in sp] == [s["kind"] for s in r["spans"]], r["caption"]
        for (a, b), s in zip(r["num_chars"], r["spans"]):
            assert r["text"][a:b].lower() == s["reading"].lower(), (r["text"], s)
            assert s["reading"] in [c.strip() for c in M.spans(r["caption"], "uk")[r["spans"].index(s)].cands]


def test_case_forms():
    # red-on-base: control (new module)
    assert U.ordw(2025, "m", "loc") == "дві тисячі двадцять п'ятому"
    assert U.ordw(1998, "m", "gen") == "тисяча дев'ятсот дев'яносто восьмого"
    assert U.ordw(2000, "m", "loc") == "двохтисячному"
    assert U.ordw(14, "f", "loc") == "чотирнадцятій"
    assert U.ordw(25, "n", "gen") == "двадцять п'ятого"
    assert U._amount("гривня", 1, "nom") == "одна гривня"
    assert U._amount("гривня", 1, "acc") == "одну гривню"
    assert U._amount("гривня", 2, "nom") == "дві гривні"
    assert U._amount("гривня", 240, "nom") == "двісті сорок гривень"
    assert U._amount("копійка", 21, "nom") == "двадцять одна копійка"
    assert U._amount("долар", 3, "nom") == "три долари"
    assert U._amount("відсоток", 1, "gen") == "одного відсотка"
    assert f"{U.card(40)} {U.noun('відсоток', 40, 'nom')}" == "сорок відсотків"
    assert f"{U.card(2)} {U.noun('відсоток', 2, 'nom')}" == "два відсотки"
    assert U.card(5, "gen") == "п'яти"
    assert U.card(1500, "acc") == "тисячу п'ятсот"  # "виділила тисячу п'ятсот гривень", never "тисяча"
    assert U.card(21000, "acc") == "двадцять одну тисячу"
    assert U.card(2000000, "acc") == "два мільйони"
    assert U._dec_reading(_R(0.9), 3, "5") == "три цілих п'ять десятих"
    assert U._dec_reading(_R(0.9), 1, "1") == "одна ціла одна десята"
    assert U._dec_reading(_R(0.1), 3, "5") == "три кома п'ять"


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
    nums = {re.search(r"\(\d{3}\) [\d-]+", c).group(0) for c in ph}
    assert len(nums) == len(ph)


def test_dev_split_by_template():
    # red-on-base: the per-row split put most carrier templates in both train and dev
    by = collections.defaultdict(set)
    for r in ROWS:
        by[U.template_key(r["caption"])].add(r["split"])
    assert all(len(v) == 1 for v in by.values()), [k for k, v in by.items() if len(v) > 1][:3]
    for c in U.CLASSES:
        sp = {r["split"] for r in ROWS if r["cls"] == c}
        assert sp == {"train", "dev"}, (c, sp)
    dev = sum(r["split"] == "dev" for r in ROWS)
    assert 0.04 * len(ROWS) <= dev <= 0.25 * len(ROWS), dev
    # the lead and the value never change the template; the counted noun after the value is part of the slot
    assert U.template_key("Кажуть, що ціни на продукти зросли на 7 % за місяць.") == U.template_key(
        "Ціни на продукти зросли на 21% за місяць.")
    assert U.template_key("Ми доїхали до моря за 2 години.") == U.template_key("Ми доїхали до моря за 25 хвилин.")


def test_resplit_keeps_dev_voices_out_of_train():
    rows = [dict(r, voice="dv" if i % 5 == 0 else "tv") for i, r in enumerate(ROWS)]
    new = U.resplit(rows, {"dv"})
    assert not any(r["split"] == "train" and r["voice"] == "dv" for r in new)
    assert {r["split"] for r in new} == {"train", "dev", "unused"}


def test_determinism():
    again = U.plan(700, 1)
    assert [r["id"] for r in again] == [r["id"] for r in ROWS]


def test_lead_never_before_imperative():
    import random

    rng = random.Random(0)
    for _ in range(50):
        assert U._lead(rng, "Телефонуйте за номером (050) 123-45-67.").startswith("Телефонуйте")
        t = U._lead(rng, "Мій син навчається у 7-му класі.")
        assert not t.startswith(tuple(U.LEADS_REPORT)), t


def test_strip_parens():
    lab, offs = U._strip_parens("за номером (нуль п'ятдесят) сто два", [[12, 26], [28, 35]])
    assert lab == "за номером нуль п'ятдесят сто два" and lab[offs[0][0]:offs[0][1]] == "нуль п'ятдесят"
    assert lab[offs[1][0]:offs[1][1]] == "сто два"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
