import sys, os, json, random, tempfile, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dupcheck as D

TRAIN = ["The quarterly report shows revenue grew by twelve percent across the northern region this year.",
         "She walked to the station and bought a ticket for the morning train to the coast.",
         "Our team shipped the new release on Thursday after fixing the last three regressions.",
         "Please confirm the delivery address before the courier leaves the warehouse tomorrow.",
         "The committee will review all of the applications received before the end of the month."]
EVAL = ["He gave the reference as five five two one nine on the second call.",
        "The invoice total was nine hundred ninety nine thousand and seventy five dollars exactly.",
        "Yesterday the library opened early so that the school groups could visit the rare book room."]


def test_unit_folding_makes_written_and_spoken_units_equal():
    assert D.unitnorm("it runs at 153.7 GHz", {"ghz", "gigahertz"}) == "it runs at 153.7 unit"
    assert D.unitnorm("it runs at 153.7 GHz") == "it runs at 153.7 GHz"            # only the default units are folded
    assert D.unitnorm("it was 9.84 °F today") == "it was 9.84 F today"
    assert D.unitnorm("it was 9.84 degrees Fahrenheit") == "it was 9.84 unit unit"
    assert D.unitnorm("up 5%") == "up 5 unit"
    assert D.unitnorm("up five percent") == "up five unit"
    assert D.unitnorm("a gigahertz band", {"gigahertz"}) == "a unit band"


def test_digit_strings_written_and_spoken():
    assert D.digit_strings("code 55,213 ok", 5) == {"55213"}
    assert D.digit_strings("code five five double two one nine", 5) == {"552219"}
    assert D.digit_strings("call 12 or 345", 5) == set()
    assert D.spoken_digit_strings("oh seven then double five") == ["07", "55"]


def test_number_runs_need_min_words_and_trim_a_trailing_and():
    assert D.number_runs("it cost nine hundred ninety nine thousand and seventy five dollars", 4) == ["nine hundred ninety nine thousand and seventy five"]
    assert D.number_runs("we saw twenty two cars and ten dogs", 4) == []
    assert D.number_runs("one two three four and then", 4) == ["one two three four"]


def test_planted_near_duplicates_are_caught_and_shuffles_are_not():
    R = random.Random(7)
    swap = lambda t: __import__("re").sub(r"\b(twelve|three|one|two)\b", lambda m: R.choice(["four", "seven", "eighty"]), t)
    train_items = [("t.jsonl", t) for t in TRAIN]
    ix = D.build(train_items, lambda t: D.N.shingles(D.unitnorm(t)))
    for t in TRAIN:
        v, k = ix.best(D.N.shingles(D.unitnorm(swap(t))))
        assert v >= 0.6, (t, v)                                    # same sentence, other numbers
        w = t.split(); R.shuffle(w)
        assert ix.best(D.N.shingles(" ".join(w)))[0] < 0.6


def test_run_reports_hits_leakage_and_gates():
    rows = [dict(n="a1", said="The quarterly report shows revenue grew by eight percent across the northern region this year."),
            dict(n="a2", said="Write down the reference as five five two one nine before you hang up."),
            dict(n="a3", said="Nobody expected the ferry to leave quite so early on a calm grey morning."),
            dict(n="a3", said="Nobody expected the ferry to leave quite so early on a calm grey morning.")]
    res = D.run(rows, [("t", t) for t in TRAIN], [("e", t) for t in EVAL], ("said",))
    txt = res["report"]
    assert "HITS >= 0.5: 1" in txt and "NEW: The quarterly report" in txt
    assert res["leakage"]["digit_hits"] == [(1, "55219")], res["leakage"]
    assert "exact duplicate sentences in pack: 1" in txt and not res["ok"]
    clean = D.run(rows[2:3], [("t", t) for t in TRAIN], [("e", t) for t in EVAL])
    assert clean["ok"] and "GATE leakage 0 and exact duplicates 0 -> PASS" in clean["report"]


def test_verbatim_number_run_from_eval_is_leakage():
    rows = [dict(said="The cheque said nine hundred ninety nine thousand and seventy five in the box.")]
    res = D.run(rows, [], [("e", t) for t in EVAL])
    assert res["leakage"]["run_hits"] and not res["ok"]


def test_cli_with_files_and_gate_exit_code():
    d = tempfile.mkdtemp()
    w = lambda name, rows: [open(os.path.join(d, name), "w").write("".join(json.dumps(r) + "\n" for r in rows)), os.path.join(d, name)][1]
    rows = w("rows.jsonl", [dict(said="Nobody expected the ferry to leave quite so early on a calm grey morning.")])
    tr = w("train.jsonl", [dict(text=t) for t in TRAIN])
    ev = w("eval.jsonl", [dict(said=t) for t in EVAL])
    out = os.path.join(d, "r.txt")
    with contextlib.redirect_stdout(io.StringIO()):
        assert D.main(["--rows", rows, "--train", tr, "--eval", ev, "--out", out, "--gate"]) == 0
    assert "[max similarity vs TRAIN]" in open(out).read()
    bad = w("bad.jsonl", [dict(said=TRAIN[0]), dict(said=TRAIN[0])])
    with contextlib.redirect_stdout(io.StringIO()):
        assert D.main(["--rows", bad, "--train", tr, "--eval", ev, "--gate"]) == 1
        assert D.main(["--rows", bad, "--train", tr, "--eval", ev]) == 0


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
