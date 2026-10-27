import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiering as T


def test_span_risk_classes():
    assert "digits_long" in T.span_risk("digits", "5550147", [], [], "five five five oh one four seven")
    assert "repeat" in T.span_risk("digits", "55", [], [], "five five")
    assert "code" in T.span_risk("int", "2", ["B"], ["B"], "B two B")
    assert T.span_risk("int", "15", [], [], "fifteen") == []


def test_tier_rule():
    ok = dict(status="scored", min_margin=9.0, any_tie=False)
    assert T.tier_of(ok, 3.0, True, True, False, "caption") == 1
    assert T.tier_of(ok, 3.0, True, True, True, "caption") == 2             # high-risk class
    assert T.tier_of(dict(ok, min_margin=1.0), 3.0, True, True, False, "caption") == 2
    assert T.tier_of(dict(ok, status="cer_high"), 3.0, True, True, False, "synthetic_script") == 2
    assert T.tier_of(ok, 3.0, False, True, False, "caption") == 2           # free greedy value differs
    assert T.tier_of(dict(ok, status="error"), 3.0, False, False, True, "human_transcript") == 0


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
