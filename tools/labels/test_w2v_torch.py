import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeller_core as C


def _lp_for(words, T=200, V=32):
    """log-probs that spell `words` (CTC letters, '|' between words) with some blanks; deterministic"""
    import torch
    ids = C.word_ids([w for _, w in C.norm_words(words)])
    lp = torch.full((T, V), -8.0)
    lp[:, 0] = -0.05                                   # blank everywhere
    step = T // (len(ids) + 1)
    for k, i in enumerate(ids):
        lp[(k + 1) * step, :] = -8.0; lp[(k + 1) * step, i] = -0.01
    return torch.log_softmax(lp, -1)


def test_loss_prefers_the_spoken_reading():
    import w2v_torch as W
    lp = _lp_for("call five oh one now")
    f = W.losses_fn(lp, 0)
    ls = f(["call five oh one now", "call five zero one now", "call five five oh one now"])
    assert ls[0] < ls[1] and ls[0] < ls[2]


def test_greedy_and_align_roundtrip():
    import w2v_torch as W
    lp = _lp_for("call five oh one now")
    inv = {v: k for k, v in C.VOCAB.items()}
    assert W.greedy_text(lp, 0, inv).lower() == "call five o one now"
    t = W.align_fn(lp, 0)("call five oh one now")
    assert len(t) == 5 and all(a < b for a, b, _ in t)


def test_score_row_end_to_end_on_fake_audio():
    import w2v_torch as W, spanfind as SF
    lp = _lp_for("ring five five five oh one now")
    label = "ring five five oh one now"
    sp = [dict(ord=k, char_start=s.char_start, char_end=s.char_end, cands=SF.candidates(s), kind=s.kind, w_start=s.w_start, w_end=s.w_end) for k, s in enumerate(SF.find_spans(label))]
    inv = {v: k for k, v in C.VOCAB.items()}
    r = C.score_row(dict(id="a", label=label, spans=sp), W.losses_fn(lp, 0), free_hyp=W.greedy_text(lp, 0, inv), align_fn=W.align_fn(lp, 0))
    assert r["chosen_label"] == "ring five five five oh one now" and r["status"] in C.STATUSES and r["words"]
    assert r["spans"][0]["cur_rank"] >= 1


def test_rms_onset():
    import numpy as np, w2v_torch as W
    rng = np.random.default_rng(0)
    x = np.concatenate([0.001 * rng.standard_normal(8000), 0.3 * rng.standard_normal(16000)]).astype("float32")
    o = W.rms_onset(x)
    assert 0.45 <= o <= 0.52


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skipped: torch not installed (0 tests, 0 failed)")
        sys.exit(0)
    import tinytest; tinytest.run(globals())
