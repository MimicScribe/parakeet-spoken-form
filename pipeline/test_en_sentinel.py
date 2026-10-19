"""English sentinel fixes (score_retrain; no GPU, no volume): python -m pytest test_en_sentinel.py -q"""
import score_retrain as S


# ---- ref_years / year_value_kept include 19xx
def test_ref_years_reads_19xx_in_digits_and_words():
    assert S.ref_years("in 1995 and 2020") == [1995, 2020]
    assert S.ref_years("in nineteen ninety five we") == [1995]
    assert S.ref_years("nineteen oh five and nineteen eighty") == [1905, 1980]
    assert S.ref_years("one thousand nine hundred and ninety five") == [1995]
    assert S.ref_years("nineteen eighty five") == [1985]  # not 1980 plus a stray "five"


def test_year_value_kept_scores_a_19xx_year_by_value():
    assert S.year_value_kept("in 1995 we", "in nineteen ninety five we") == (1, 1)
    assert S.year_value_kept("in 1995 we", "in 1995 we") == (1, 1)
    assert S.year_value_kept("in 1995 we", "in nineteen ninety we") == (1, 0)  # tail lost
    assert S.year_value_kept("in nineteen oh five we", "in 1905 we") == (1, 1)
    assert S.year_value_kept("in 1900 we", "in 1900 we") == (1, 1)


def test_20xx_years_are_scored_exactly_as_before():  # red-on-base: control
    assert S.ref_years("in 2020") == [2020]
    assert S.year_value_kept("2020", "twenty twenty") == (1, 1)
    assert S.year_value_kept("2020", "two thousand twenty one") == (1, 0)
    assert S.year_readings(2005) == ["2005", "two thousand five", "two thousand and five", "twenty oh five"]
    assert S.ref_years("two thousand seven hundred fifty") == []  # the head of a bigger number


def test_a_lost_19xx_year_is_not_a_two_thousand_truncation(tmp_path):
    # year_trunc counts a year lost while the hyp holds a bare "two thousand": only 20xx years can be cut that way
    import json

    run = tmp_path / "r"
    run.mkdir()
    rows = [{"id": "a", "ref": "in 1995 we grew\t", "hyp": "in two thousand we grew"}]
    for name in ("d1_dev", "e22_numbers", "fast", "d9_dev", "biz_dev"):
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) if name == "e22_numbers" else "")
    for name in ("e22_other", "lspc_dev", "fleurs_wer", "vox_dev", "fleurs_num", "e21_dev", "adv", "noise"):
        (run / f"{name}.jsonl").write_text("")
    s = S.score(str(tmp_path), "r", stock_run="r")
    assert s["year_value_n"] == 1 and s["year_value_kept"] == 0 and s["year_trunc"] == 0
