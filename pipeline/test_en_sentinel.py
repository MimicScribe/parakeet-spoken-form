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


# ---- en_wf and fast_d1_ok are report-only in the checkpoint selection
BASE = {"blend_twenty_thousand": 0, "digits_e22_numbers": 5, "digits_d1_dev": 5, "chimeras": 0, "adv_en_ok": 50,
        "fast_d1_ok": 40, "leak_vox_dev_total": 0, "leak_fleurs_num_total": 0, "wer_lspc_dev": 5, "wer_d1_dev": 5,
        "wer_fleurs": {}}


def test_a_value_blind_digit_count_and_a_say_clip_match_do_not_decide_a_selection():
    res = {"a": dict(BASE), "b": dict(BASE, digits_e22_numbers=0, digits_d1_dev=0, fast_d1_ok=90)}  # b: fewer digits, more say hits
    sel = S.select(res, ["a", "b"])
    assert sel["tied"] == ["a", "b"] and sel["run"] == "a"  # nothing else differs: a tie, first listed
    assert sel["metrics"]["b"]["en_wf"] == 0 and sel["metrics"]["b"]["fast_d1_ok"] == 90  # still reported
    assert S.select(res, ["a", "b"], report_only=())["run"] == "b"  # the retrain-3 order, still available
    assert S.SELECT_REPORT_ONLY == ("en_wf", "fast_d1_ok")


def test_the_remaining_keys_still_decide():  # red-on-base: control
    res = {"a": dict(BASE), "b": dict(BASE, adv_en_ok=60), "c": dict(BASE, chimeras=3)}
    assert S.select(res, ["a", "b", "c"])["run"] == "b"


# ---- English held-out VALUE scorer (en_value)
def test_en_value_scores_value_in_either_form_by_labeller_kind():
    rows = [{"ref": "In 2020 we grew 5% to $105 million at 3:30 pm.\tx",
             "hyp": "in twenty twenty we grew five percent to one hundred five million at three thirty pm"},
            {"ref": "In 2020 we grew 5%.", "hyp": "in two thousand twenty we grew 5%"}]
    v = S.en_value(rows)
    assert v["n"] == 7 and v["WRONG_VALUE"] == 0 and v["DROPPED"] == 0
    assert v["WORDS_RIGHT_VALUE"] == 6 and v["DIGITS_RIGHT_VALUE"] == 1
    assert v["by_kind"]["year"]["WORDS_RIGHT_VALUE"] == 2  # "twenty twenty" and "two thousand twenty" are both 2020
    assert v["by_kind"]["percent"]["DIGITS_RIGHT_VALUE"] == 1 and v["by_kind"]["money"]["WORDS_RIGHT_VALUE"] == 1


def test_en_value_sees_a_wrong_number_written_in_words():
    hyp = "in two thousand twenty one we grew five percent"  # no digit and no symbol: en_wf cannot see it
    assert not S.WF.search(hyp)
    v = S.en_value([{"ref": "In 2020 we grew 5%.", "hyp": hyp}])
    assert v["WRONG_VALUE"] == 1 and v["words"] == 2 and v["words_wrong"] == 1 and v["digits_wrong"] == 0
    assert v["by_kind"]["year"]["WRONG_VALUE"] == 1 and v["by_kind"]["percent"]["WORDS_RIGHT_VALUE"] == 1


def test_en_value_counts_drops_and_wrong_digits():
    v = S.en_value([{"ref": "Revenue was 2.5 billion.", "hyp": "revenue was 2.6 billion"},
                    {"ref": "Call 555 0142 now.", "hyp": "call now"}])
    assert v["WRONG_VALUE"] == 1 and v["digits"] == 1 and v["digits_wrong"] == 1 and v["DROPPED"] == 2
    assert S.en_value([])["n"] == 0


def test_the_sentinel_reports_it_for_the_real_beds(tmp_path):
    import json

    run = tmp_path / "r"
    run.mkdir()
    rows = [{"id": "a", "ref": "in 2020 we grew\t", "hyp": "in two thousand twenty one we grew"}]
    for name in ("d1_dev", "e22_numbers", "e22_other", "lspc_dev", "fast", "d9_dev", "biz_dev", "fleurs_wer", "vox_dev",
                 "fleurs_num", "e21_dev", "adv", "noise"):
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) if name == "e22_numbers" else "")
    s = S.score(str(tmp_path), "r", stock_run="r")
    assert s["en_value_e22_numbers"]["WRONG_VALUE"] == 1 and s["en_value_d1_dev"]["n"] == 0
    assert s["en_value_wrong"] == 1 and s["en_value_words_wrong"] == 1 and s["en_value_words"] == 1
    assert S.sel_metrics(s | {"digits_e22_numbers": 0, "digits_d1_dev": 0, "adv_en_ok": 0, "fast_d1_ok": 0,
                              "leak_vox_dev_total": 0, "leak_fleurs_num_total": 0, "wer_lspc_dev": 0,
                              "wer_d1_dev": 0})["en_value_wrong"] == 1
