import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeller_core as C, spanfind as SF


def mk_row(label, extra=None):
    spans = []
    for k, s in enumerate(SF.find_spans(label)):
        spans.append(dict(ord=k, char_start=s.char_start, char_end=s.char_end, cands=SF.candidates(s), kind=s.kind, w_start=s.w_start, w_end=s.w_end))
    return dict(id="r", label=label, spans=spans, **(extra or {}))


def fake_loss(truth):
    """loss = edit distance (words) between a candidate label and the 'audio truth' sentence"""
    t = truth.lower().split()
    return lambda labs: [float(C.edit(l.lower().split(), t)) * 3.0 for l in labs]


def test_never_rejects_and_keeps_all_statuses():
    row = mk_row("Call two three nine two one eight today.")
    r = C.score_row(row, fake_loss("call two three nine two one eight today"), free_hyp="CALL TWO THREE NINE TWO ONE EIGHT TODAY")
    assert r["status"] in C.STATUSES and r["id"] == "r"
    r2 = C.score_row(row, fake_loss("completely different sentence here"), free_hyp="XYZ")
    assert r2["status"] in ("cer_high", "undecided") and "spans" in r2          # not dropped


def test_top2_includes_same_key_alternates():
    row = mk_row("the pin is one oh five")
    # audio says "zero": oh and zero are the same canon_key -> old margin_x would ignore it, new margin sees it
    r = C.score_row(row, fake_loss("the pin is one zero five"))
    sp = r["spans"][0]
    assert sp["chosen"] == "one zero five" and sp["current"] == "one oh five"
    assert sp["same_key_alt"] is False or sp["same_key_alt"] is True
    assert sp["cur_rank"] >= 1 and len(sp["top2"]) == 2
    assert sp["margin"] > 0 and sp["margin_x"] >= sp["margin"]


def test_tie_flag_when_two_candidates_equal():
    row = mk_row("the code is one oh five")
    r = C.score_row(row, lambda labs: [1.0] * len(labs))
    assert r["spans"][0]["tie"] is True and r["status"] == "undecided"


def test_repeat_count_candidates_present_and_chosen():
    row = mk_row("ring five five oh one now")
    cands = row["spans"][0]["cands"]
    assert "five five five oh one" in cands and "five oh one" in cands
    r = C.score_row(row, fake_loss("ring five five five oh one now"))
    assert r["spans"][0]["chosen"] == "five five five oh one" and r["changed"] is True


def test_commas_never_in_candidates_and_trailing_punct_kept():
    row = mk_row("Dial two three nine, two one eight, please.")
    assert all("," not in c for c in row["spans"][0]["cands"])
    lab = C.build_label(row["label"], row["spans"], [0])
    assert lab == "Dial two three nine two one eight, please."


def test_row_alternatives_scored():
    row = mk_row("it was forty four thousand two hundred ok", dict(alt_labels=["it was forty four thousand um two hundred ok"]))
    r = C.score_row(row, fake_loss("it was forty four thousand um two hundred ok"))
    assert r["row_alts"] and r["row_alts"][0]["delta_vs_chosen"] < 0          # the filler version fits better


def test_align_failure_becomes_status():
    row = mk_row("call five five oh")
    def bad(_): raise RuntimeError("len mismatch")
    r = C.score_row(row, fake_loss("call five five oh"), free_hyp="CALL FIVE FIVE O", align_fn=bad)
    assert r["status"] == "align_fail" and r["words"] is None


def test_error_is_a_status_not_an_exception():
    r = C.score_row(dict(id="x", label="a b", spans=[dict(char_start=0, char_end=1, cands=None)]), lambda l: [0.0])
    assert r["status"] == "error"


def test_ids_cover_vocab():
    ids = C.label_ids("call five oh one AM")
    assert all(i in C.VOCAB.values() for i in ids)


def test_edit_bitvector_equals_the_reference_dp():
    import random
    random.seed(7)
    for _ in range(3000):
        al = random.choice(["ab", "abc", "ABCDEFGH |'"])
        a = "".join(random.choice(al) for _ in range(random.randint(0, 60))); b = "".join(random.choice(al) for _ in range(random.randint(0, 60)))
        assert C.edit(a, b) == C.edit_dp(a, b), (a, b)
    a, b = "the meeting had started a little late " * 6, "the meting had stared a litle late and " * 6          # long, ~230 chars
    assert C.edit(a, b) == C.edit_dp(a, b)
    assert C.edit("a b c".split(), "a c d".split()) == C.edit_dp("a b c".split(), "a c d".split()) == 2          # word lists (fake_loss in these tests)
    assert C.edit("", "abc") == 3 and C.edit("abc", "") == 3 and C.edit("", "") == 0


def test_row_loss_cache_scores_each_distinct_label_once_and_changes_nothing():
    row = mk_row("call two three nine two one eight today at five thirty and the pin is one oh five")
    assert sum(1 for sp in row["spans"] if len(sp["cands"]) > 1) >= 2            # two passes happen
    base = fake_loss("call two three nine two one eight today at five thirty and the pin is one zero five")
    seen = []
    def counting(labs):
        seen.extend(labs); return base(labs)
    r_cached = C.score_row(row, counting, free_hyp="X")
    # reference: the same row scored with a loss_fn that sees every call (no cache) = the cached result must equal it
    r_plain = C.score_row(row, lambda labs: base(labs), free_hyp="X")
    assert r_cached == r_plain
    assert len(seen) == len(set(seen)), "a label was scored twice"
    uncached_calls = []
    # count what the old code asked for: replay the passes without the cache
    spans = row["spans"]; ch = [0] * len(spans)
    multi = [k for k, sp in enumerate(spans) if len(sp["cands"]) > 1]
    for ps in range(2):
        for k in multi:
            labs = [C.build_label(row["label"], spans, ch[:k] + [c] + ch[k + 1:]) for c in range(len(spans[k]["cands"]))]
            uncached_calls += labs
            ch[k] = C.span_result(spans[k], base(labs), ch[k])["chosen_idx"]
    assert len(seen) < len(uncached_calls)


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
