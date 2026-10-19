"""Per-part listing weights of the v4 mix (2026-10-18; no GPU, no volume). Needs the modal package:
    uv run --no-project --with modal --with numpy --with pytest python -m pytest test_part_weights.py -q
A part listed n times is read n times; default (empty) weights leave the listing exactly as it was."""
import inspect
from collections import Counter

import pytest

MA = pytest.importorskip("modal_app")

LIST = [("a.jsonl", 1.5, "span"), ("b.jsonl", 0, "nonen_span"), ("b.jsonl", 0, "nonen_span"), ("c.jsonl", 0, "free")]


def test_empty_weights_return_the_listing_unchanged():
    assert MA.apply_part_weights(LIST) is LIST
    assert MA.apply_part_weights(LIST, {}, []) is LIST
    assert MA.apply_part_weights(LIST, "", "") is LIST


def test_weight_sets_the_total_count_and_keeps_hours_tag_and_position():
    out = MA.apply_part_weights(LIST, {"a.jsonl": 3, "b.jsonl": 1})
    assert out == [("a.jsonl", 1.5, "span")] * 3 + [("b.jsonl", 0, "nonen_span"), ("c.jsonl", 0, "free")]
    assert MA.apply_part_weights(LIST, "c.jsonl=2").count(("c.jsonl", 0, "free")) == 2


def test_unlisted_part_and_bad_counts_are_errors():
    for bad in ({"typo.jsonl": 2}, {"a.jsonl": 0}):
        with pytest.raises(AssertionError):
            MA.apply_part_weights(LIST, bad)


def test_extra_parts_are_appended_n_times():
    out = MA.apply_part_weights(LIST, None, "ml/new/rows.jsonl:nonen_span:2")
    assert out[:4] == LIST and out[4:] == [("ml/new/rows.jsonl", 0, "nonen_span")] * 2
    with pytest.raises(AssertionError):
        MA.apply_part_weights(LIST, None, "a.jsonl:span:2")  # already listed: use part_weights


def test_parsers():
    assert MA.parse_part_weights("x=2, y/z.jsonl=10") == {"x": 2, "y/z.jsonl": 10}
    assert MA.parse_part_weights("") == {} and MA.parse_part_weights(None) == {}
    assert MA.parse_extra_parts("p:q:3") == [("p", 0, "q", 3)]
    with pytest.raises(AssertionError):
        MA.parse_part_weights("novalue")


def test_v4_listing_default_is_unchanged_and_a_weight_lists_the_part_n_times():
    kw = dict(repeat=1, anchor18=2, win_repeat=3, en_bal=False, en_plain_h=80.0)
    base = MA.retrain3_mix_v4(MA.RUN2_LANGS, **kw)
    assert MA.retrain3_mix_v4(MA.RUN2_LANGS, **kw, part_weights={}, extra_parts=[]) == base
    key = "mix4/tts_de.jsonl"
    w = MA.retrain3_mix_v4(MA.RUN2_LANGS, **kw, part_weights={key: 3})
    assert Counter(w)[(key, 0, "nonen_span")] == 3 and Counter(base)[(key, 0, "nonen_span")] == 1
    assert len(w) == len(base) + 2
    dry = MA.retrain3_mix_v4(MA.RUN2_LANGS, **kw, part_weights={key: 3}, mix_dir="mix4final10")
    assert [m for m in dry if m[0] == "mix4final10/tts_de.jsonl"] == [("mix4final10/tts_de.jsonl", 0, "nonen_span")] * 3


def test_entrypoints_take_the_weights():
    for f in (MA.retrain3, MA.retrain3_80, MA.share_report, MA.build_mix4):
        p = inspect.signature(f._raw_f_).parameters
        assert p["part_weights"].default == "" and p["extra_parts"].default == "", f
    assert inspect.signature(MA.retrain3_80._raw_f_) == inspect.signature(MA.retrain3._raw_f_)
