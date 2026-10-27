import sys, os, json, tempfile
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger as LG


def mk():
    d = tempfile.mkdtemp()
    L = LG.Ledger(os.path.join(d, "t.db"))
    lines = [json.dumps({"audio_filepath": "/data/a/x%d.wav" % i, "text": "call five five oh %d" % i, "duration": 1.5}) for i in range(3)]
    L.add_part("p1", "f", "listing", 3)
    for i, ln in enumerate(lines):
        L.add_row(row_id="p1#%d" % i, part="p1", idx=i, key=LG.stable_key("/data/a/x%d.wav" % i), variant="clean", audio_path="/data/a/x%d.wav" % i, path_sha1=LG.sha1("/data/a/x%d.wav" % i),
                  source_class="caption", real_synth="real", label=json.loads(ln)["text"], raw_line=ln)
    L.commit()
    return L, d, lines


def test_stable_keys():
    assert LG.stable_key("/data/aug/wav/clip121_x_391_1778__aug.wav") == LG.stable_key("/data/dom/wav/clip121_x_391_1778.wav") == "a:clip121_x_391_1778"
    assert LG.variant_of("/a/b__aug.wav") == "aug" and LG.variant_of("/data/mix/wav/spd_p_0.wav") == "spd" and LG.variant_of("/a/b.wav") == "clean"
    k1 = LG.stable_key("/data/mix/wav/spd_p_0.wav", "Oh, that's a hard question.", "vid")
    k2 = LG.stable_key("/data/mix/wav/spd_p_9.wav", "oh that's a hard   question", "vid")
    assert k1 == k2 and k1.startswith("vt:vid|")
    assert LG.span_key("a:x", 2) == "a:x#s2"


def test_export_without_decisions_is_byte_identical_and_drop_relabel_apply():
    L, d, lines = mk()
    out = os.path.join(d, "out")
    c = L.export_mix(out)
    assert c["p1"] == dict(rows_in=3, rows_out=3, relabelled=0, dropped=0)
    assert open(os.path.join(out, "p1.jsonl")).read() == "".join(l + "\n" for l in lines)
    L.add_decision("p1#0", "relabel", "form from audio", "rule", new_label="call five five oh zero")
    L.add_decision("p1#1", "drop", "garbled", "human")
    L.add_decision("p1#2", "unconfirmed", "below margin", "rule")
    c = L.export_mix(out)["p1"]
    assert c == dict(rows_in=3, rows_out=2, relabelled=1, dropped=1)
    rows = [json.loads(l) for l in open(os.path.join(out, "p1.jsonl"))]
    assert rows[0]["text"] == "call five five oh zero" and rows[0]["label_before_ledger"] == "call five five oh 0" and rows[1]["text"].endswith("2")
    L.add_decision("p1#1", "keep", "changed mind", "human")            # latest decision wins
    assert L.export_mix(out)["p1"]["rows_out"] == 3
    st = {r["row_id"]: r["action"] for r in L.db.execute("SELECT row_id, action FROM row_state")}
    assert st == {"p1#0": "relabel", "p1#1": "keep", "p1#2": "unconfirmed"}


def test_append_only_and_validation():
    L, d, _ = mk()
    eid = L.add_evidence("p1#0", "ctc", "tool", value="x", margin=1.5, sha="abc")
    L.add_decision("p1#0", "keep", "r", "rule")
    for sql in ("UPDATE evidence SET value='y'", "DELETE FROM evidence", "UPDATE decisions SET action='drop'", "DELETE FROM decisions"):
        try:
            L.db.execute(sql); assert False, sql
        except Exception as e:
            assert "append-only" in str(e)
    for bad in (("p1#0", "maybe", "r", "rule"), ("p1#0", "keep", "r", "bot"), ("p1#0", "relabel", "r", "rule")):
        try:
            L.add_decision(*bad); assert False
        except ValueError:
            pass
    assert eid == 1


def test_gold_is_append_only_and_testbed_rows_have_no_row_id():
    L, d, _ = mk()
    L.add_gold("heldout_set", "c01", "NATURAL", question="natural?")
    L.add_gold("calibration_set", "c08", "WORD MISSING FROM LABEL", row_id="p1#0")
    try:
        L.db.execute("DELETE FROM gold"); assert False
    except Exception as e:
        assert "append-only" in str(e)
    assert [tuple(r) for r in L.db.execute("SELECT bed, row_id FROM gold ORDER BY id")] == [("heldout_set", None), ("calibration_set", "p1#0")]


def test_cli_stats_and_query():
    L, d, _ = mk(); L.close()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        LG.main(["--db", os.path.join(d, "t.db"), "stats"])
    assert json.loads(buf.getvalue())["rows"] == 3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        LG.main(["--db", os.path.join(d, "t.db"), "query", "select count(*) n from rows"])
    assert buf.getvalue().split()[-1] == "3"


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
