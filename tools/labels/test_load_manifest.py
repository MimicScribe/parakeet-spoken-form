import sys, os, json, tempfile
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger as LG, load_listing as LL, manifest as MF, tiering as T, label_local as LC


def mk_listing(d):
    p = os.path.join(d, "talks_a.jsonl")
    rows = [dict(audio_filepath="/data/a/x0.wav", text="Call five five oh one now.", duration=4.0),
            dict(audio_filepath="/data/a/x1.wav", text="It cost 20 dollars.", text_v2="It cost twenty dollars.", duration=3.0),
            dict(audio_filepath="/data/a/x2.wav", text="No numbers here at all.", duration=2.0),
            dict(audio_filepath="/data/a/x3__aug.wav", text="Dial two three nine two one eight please.", duration=5.0, synthetic=True)]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rules = os.path.join(d, "rules.json")
    json.dump([{"part_regex": "^talks_", "source_class": "human_transcript", "dataset": "talks"}], open(rules, "w"))
    return p, rules


def test_load_classifies_by_rules_and_default_and_stores_spans():
    d = tempfile.mkdtemp()
    p, rules = mk_listing(d)
    L = LL.load(os.path.join(d, "t.db"), [p], LL.load_rules(rules), with_spans=True, log=None)
    got = {r["row_id"]: (r["source_class"], r["variant"], r["digits_in_label"]) for r in L.db.execute("SELECT * FROM rows")}
    assert got["talks_a#0"] == ("human_transcript", "clean", 0) and got["talks_a#1"][2] == 1 and got["talks_a#3"][1] == "aug"
    L2 = LL.load(os.path.join(d, "t2.db"), [p], [], with_spans=True, log=None)               # no rules: caption / real, synthetic flag honoured
    got = {r["row_id"]: r["source_class"] for r in L2.db.execute("SELECT row_id, source_class FROM rows")}
    assert got["talks_a#0"] == "caption" and got["talks_a#3"] == "synthetic_script"
    n_spans = {r["row_id"]: r["n"] for r in L.db.execute("SELECT row_id, COUNT(*) n FROM spans GROUP BY row_id")}
    assert n_spans.get("talks_a#0") == 1 and "talks_a#1" not in n_spans and "talks_a#2" not in n_spans          # digit labels get no spans
    assert [r["label"] for r in L.db.execute("SELECT label FROM candidates WHERE origin='text_v2'")] == ["It cost twenty dollars."]
    assert L.export_mix(os.path.join(d, "out"))["talks_a"]["rows_out"] == 4


def test_rebuild_refuses_when_human_answers_exist():
    d = tempfile.mkdtemp()
    p, _ = mk_listing(d)
    db = os.path.join(d, "t.db")
    L = LL.load(db, [p], [], log=None)
    L.add_gold("set", "i1", "CORRECT", row_id="talks_a#0")
    L.commit(); L.close()
    try:
        LL.load(db, [p], [], log=None); assert False
    except SystemExit as e:
        assert "human answers" in str(e)
    LL.load(db, [p], [], force=True, log=None).close()


def test_manifest_scores_the_digit_free_label_and_skips_rows_without_choices():
    d = tempfile.mkdtemp()
    p, _ = mk_listing(d)
    L = LL.load(os.path.join(d, "t.db"), [p], [], log=None)
    m0 = MF.manifest_row(L, "talks_a#0")
    assert m0["label"] == "Call five five oh one now." and m0["spans"][0]["cands"][0] == "five five oh one" and len(m0["spans"][0]["cands"]) > 3
    m1 = MF.manifest_row(L, "talks_a#1")
    assert m1["label"] == "It cost twenty dollars."          # digits in the label: the planned all-words label is scored
    L.add_candidate("talks_a#0", "Call five five oh uh one now.", "filler")
    assert MF.manifest_row(L, "talks_a#0", alt_origins=("filler",))["alt_labels"] == ["Call five five oh uh one now."]
    assert MF.manifest_row(L, "talks_a#0")["alt_labels"] == []
    L.db.execute("UPDATE rows SET text_v2=NULL WHERE row_id='talks_a#1'")
    L.commit()
    assert MF.manifest_row(L, "talks_a#1") is None                  # digits and no all-words alternative
    out = os.path.join(d, "m.jsonl")
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        MF.main(["--db", os.path.join(d, "t.db"), "--out", out])
    ids = [json.loads(l)["id"] for l in open(out)]
    assert ids == ["talks_a#0", "talks_a#3"], ids              # only rows with at least one span that has several readings


def test_record_verdict_relabels_only_with_a_typed_reading_and_refuses_digits():
    d = tempfile.mkdtemp()
    p, rules = mk_listing(d)
    L = LL.load(os.path.join(d, "t.db"), [p], LL.load_rules(rules), log=None)
    L.add_queue("q1", "Check", "Is the reading right?", scope="span")
    L.add_queue_item("q1", "i0", "Call five five oh one now.", row_id="talks_a#0", span=[5, 21], candidates=["five five oh one"])
    r = L.record_verdict("q1", "i0", "wrong", heard="five five zero one")
    assert r["action"] == "relabel"
    row = L.db.execute("SELECT * FROM row_state WHERE row_id='talks_a#0'").fetchone()
    assert row["final_label"] == "Call five five zero one now." and row["decided_by"] == "human"
    try:
        L.record_verdict("q1", "i0", "wrong", heard="5501"); assert False
    except ValueError as e:
        assert "no digits" in str(e)
    assert L.record_verdict("q1", "i0", "unsure")["action"] == "unconfirmed"
    assert [x["row_id"] for x in L.db.execute("SELECT row_id FROM review_page")] == ["talks_a#0"]
    assert L.queue_summary()[0]["done"] == 1


def test_tiering_estimate_on_a_loaded_ledger():
    d = tempfile.mkdtemp()
    p, rules = mk_listing(d)
    L = LL.load(os.path.join(d, "t.db"), [p], [], with_spans=True, log=None); L.close()          # no rules: caption + synthetic rows (human transcripts need no paid pass)
    rep = T.estimate(os.path.join(d, "t.db"), usd_per_min=0.01)
    assert rep["rows_with_number_spans"] == 2 and rep["usd_per_min"] == 0.01
    assert rep["scenarios_tiered"]["planning"]["referee_rows"] >= 1


def test_label_local_resumes_after_a_torn_line():
    d = tempfile.mkdtemp()
    f = os.path.join(d, "o.jsonl")
    open(f, "w").write(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n" + '{"id": "c", "sta')
    assert LC.done_ids(f) == {"a", "b"} and LC.done_ids(os.path.join(d, "none")) == set()


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
