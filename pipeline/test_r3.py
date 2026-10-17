"""Retrain-3 unit tests (plan R3.13 #16). python -m pytest test_r3.py -q"""
import labeller as L
import filter3 as F
import mix_build as M
import score_retrain as S
import restore_fillers as R


def test_render_ranges_a_hundred_offset():
    cap = "about a 100 people and 25 more"
    sp = L.spans(cap)
    label, ranges = L.render_ranges(cap, sp, [0] * len(sp))
    for (a, b), s in zip(ranges, sp):
        assert label[a:b].strip() and not label[a:b].startswith(" ")
    assert label[ranges[-1][0]:ranges[-1][1]].startswith("twenty")


def test_edge_units_by_value():
    assert F.edge_ok("In twenty eleven we grew five percent.", "In 2011, we grew 5%.")
    assert F.edge_ok("It cost five dollars", "It cost $5")
    assert not F.edge_ok("So two thousand eleven was big", "2011 was big")
    assert not F.edge_ok("we grew five percent", "we grew six percent")


def test_unit_abbreviation_drop():
    assert F.UNIT_ABBR.search("three thousand seven hundred m away")
    assert not F.UNIT_ABBR.search("four oh one K plan")


def test_blend_counter_excludes_scales():
    assert S.BLEND.search("in twenty thousand fourteen we")
    assert not S.BLEND.search("twenty thousand nine hundred people")
    assert not S.BLEND.search("twenty thousand people")


def test_select_without_stock_init():
    base = {"blend_twenty_thousand": 0, "digits_e22_numbers": 1, "digits_d1_dev": 1, "chimeras": 0, "adv_en_ok": 50,
            "fast_d1_ok": 40, "leak_vox_dev_total": 0, "leak_fleurs_num_total": 0, "wer_lspc_dev": 0.05,
            "wer_d1_dev": 0.1, "wer_fleurs": {"de_de": 0.05}}
    res = {"a": dict(base), "b": {**base, "adv_en_ok": 60}, "v10": dict(base)}
    assert S.select(res, ["a", "b"], baseline="v10")["run"] == "b"


def test_year_ratio_regexes():
    assert M.TWO_K_0X.search("in two thousand nine we") and not M.TWO_K_0X.search("two thousand nineteen")
    assert not M.TWO_K_LOW.search("two thousand five hundred people")
    assert M.TWENTY_YR.search("twenty fourteen was")


def test_restore_fillers_no_number_repeats():
    t, n = R.merge("So we grew five percent.", "So, uh, we we grew five five percent.")
    assert t == "So uh we we grew five percent." and n == 2


def test_leak_words_complete():  # review: [5:-5] slicing dropped "two"/"percent", added "o"
    de = S.leak_re("de")
    assert de.search("zwei two") and de.search("percent") and not S.leak_re("pt").search("o carro")
    assert not S.leak_re("pl").search("ten dom")


def test_blend_no_backtrack():
    assert not S.BLEND.search("twenty thousand fifty five hundred")


def test_scaled_decimal_units():
    assert F.units("we made two point five million dollars") == F.units("we made $2.5 million")


def test_restore_no_link_word_repeats():
    t, _ = R.merge("so we made a hundred and five sales", "so we made a a hundred and and five sales")
    assert t == "so we made a hundred and five sales"


def test_uk_tts_set_listed_and_targeted():
    # red-on-base: fails before tts6uk was added (build4 would never read the Ukrainian synthetic set)
    assert ("tts6uk", "tts6uk_judge") in M.TTS_SETS
    assert "tts6uk" in M.TTS_TARGETED
    assert len({tag for tag, _ in M.TTS_SETS}) == len(M.TTS_SETS)  # one judge marker per set


def test_select_veto():  # run 2: a seam-gate veto removes a run before the walk; all vetoed -> nothing selected
    base = {"blend_twenty_thousand": 0, "digits_e22_numbers": 0, "digits_d1_dev": 0, "chimeras": 0, "adv_en_ok": 10,
            "fast_d1_ok": 10, "leak_vox_dev_total": 0, "leak_fleurs_num_total": 0, "wer_lspc_dev": 5, "wer_d1_dev": 5,
            "wer_fleurs": {}}
    res = {"a": dict(base, adv_en_ok=20), "b": dict(base)}
    assert S.select(res, ["a", "b"])["run"] == "a"
    sel = S.select(res, ["a", "b"], veto={"a": ["wave-1 merge-made wrong 9 > 0.5 x B 4"]})
    assert sel["run"] == "b" and "a" in sel["vetoed"]
    assert S.select(res, ["a", "b"], veto={"a": ["x"], "b": ["y"]})["run"] is None
