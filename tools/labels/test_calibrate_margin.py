import sys, os, random
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_margin as K


def rec(i, margin, chosen_is_current=True, status="scored", cur="one hundred dollars", alt="a hundred dollars", tie=False):
    ch = cur if chosen_is_current else alt
    return dict(id="r%d" % i, status=status, min_margin=margin, any_tie=tie, changed=not chosen_is_current, chosen_label=ch,
                spans=[dict(ord=0, current=cur, chosen=ch, chosen_idx=0 if chosen_is_current else 1, ncand=2, margin=margin, margin_x=margin, tie=tie, same_key_alt=False,
                            cur_loss=1.0, best_loss=1.0, cur_rank=0 if chosen_is_current else 1, top2=[[ch, 1.0], [alt if chosen_is_current else cur, 1.0 + margin]], kind="int", w=[0, 2])])


def test_wilson_and_alt_types():
    lo, hi = K.wilson(90, 100)
    assert 0.82 < lo < 0.86 < 0.94 < hi
    assert K.alt_type("one oh five", "one zero five") == "oh_zero_double"
    assert K.alt_type("a hundred dollars", "one hundred dollars") == "a_vs_one"
    assert K.alt_type("one hundred five", "one hundred and five") == "and"


def test_curve_and_threshold_on_synthetic_gold():
    R = random.Random(1)
    scored = []
    for i in range(400):
        if i < 300:
            scored.append(rec(i, R.uniform(4, 30), True))                    # confident and right
        else:
            m = R.uniform(0, 3); scored.append(rec(i, m, R.random() < 0.5))   # low margin: coin flip
    tab = K.span_table(scored, {r["id"] for r in scored})
    cv = K.curve(tab)
    thr = K.pick_threshold(cv)
    assert thr is not None and thr >= 1 and cv[0]["coverage"] == 1.0
    hi = [c for c in cv if c["threshold"] == thr][0]
    assert hi["accuracy"] >= 0.9 and hi["coverage"] >= 0.7


def test_fallback_never_drops():
    assert K.decide(rec(1, 0.3), 1.0)[0] == "unconfirmed"
    assert K.decide(rec(2, 5.0, False), 1.0) == ("relabel", "audio-scored form differs from the current label (margin 5.00 >= 1.00)", "a hundred dollars")
    assert K.decide(rec(3, 5.0, True), 1.0)[0] == "keep"
    for st in ("error", "align_fail", "cer_high"):
        a, why, new = K.decide(rec(4, 9.0, False, status=st), 1.0)
        assert a == "unconfirmed" and new is None


def test_ledger_writes_decisions_and_evidence():
    import ledger as LG, tempfile
    d = tempfile.mkdtemp()
    L = LG.Ledger(os.path.join(d, "t.db"))
    L.add_row(row_id="p#0", part="p", idx=0, key="a:x", variant="clean", audio_path="/a/x.wav", path_sha1="s", source_class="human_transcript", real_synth="real", label="one hundred dollars", raw_line='{"text":"one hundred dollars"}')
    L.add_part("p", "f", "t", 1)
    out = K.apply_to_ledger(L, [rec(0, 5.0, False)], 1.0, {"r0": "p#0"})
    assert out == {"relabel": 1}
    st = L.db.execute("SELECT action, decided_by, new_label FROM latest_decision").fetchone()
    assert tuple(st) == ("relabel", "rule", "a hundred dollars")
    assert L.db.execute("SELECT COUNT(*) FROM evidence WHERE source='ctc_cand_scored'").fetchone()[0] == 1
    try:
        L.db.execute("UPDATE decisions SET action='keep'"); assert False
    except Exception as e:
        assert "append-only" in str(e)


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
