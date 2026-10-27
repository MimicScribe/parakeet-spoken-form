import sys, os, json, tempfile, io, contextlib, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import semantic as S, variety as V


class HashEmbedder:
    """stand-in embedder: normalised bag of hashed words (deterministic, no weights)"""
    dim = 64

    def encode(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                X[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-9)


TRAIN = ["The council approved the budget after a long debate about the roads.", "Please bring the signed form to the front desk before noon on Friday.",
         "The ferry crosses the bay twice a day when the weather is calm."]


def test_frames_mask_numbers_dedupe_and_drop_short():
    fr = S.usable_frames(["We shipped twelve crates on Monday.", "We shipped nine crates on Tuesday.", "Seven. Ten.", "Go now"])
    assert fr == ["we shipped # crates on #"], fr


def test_best_matches_is_blockwise_exact():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 16)).astype(np.float32); T = rng.standard_normal((50, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True); T /= np.linalg.norm(T, axis=1, keepdims=True)
    b1, a1 = S.best_matches(X, T, block=7)
    full = X @ T.T
    assert np.allclose(b1, full.max(1)) and (a1 == full.argmax(1)).all()


def test_index_check_and_resume():
    d = tempfile.mkdtemp(); E = HashEmbedder()
    fr, emb = S.build_index("train", TRAIN * 2 + ["A completely different sentence about trains and timetables today."], d, E, step=2, log=None)
    assert len(fr) == 4 and emb.shape == (4, 64) and os.path.exists(os.path.join(d, "train_emb.npy"))
    np.save(os.path.join(d, "train_emb_part.npy"), emb[:2])                       # an interrupted run: 2 of 4 frames done
    fr2, emb2 = S.build_index("train", [], d, E, step=2, log=None)
    assert fr2 == fr and np.allclose(emb2.astype(np.float32), emb.astype(np.float32))
    rows = [dict(n="a", said="The council approved the budget after a long debate about the roads."), dict(n="b", said="Dolphins are mammals that live in warm shallow seas."),
            dict(n="c", said="Dolphins are mammals that live in warm shallow seas.")]
    res = S.check(rows, E, {"train": (fr, emb)})
    assert res["train_max"] > 0.99 and res["train_ge_0.90"] == 1 and res["within_max"] > 0.99 and res["within_ge_0.80"] == 2 and res["max_any"] > 0.99 and res["rows"] == 3
    assert res["hits_top"][0][0] == "train"


def test_calibrate_reports_cosine_vs_lexical():
    pairs = [(a, b) for a, b in [("The council approved the budget after a long debate.", "The council approved the budget after a long debate about roads."),
                                 ("We shipped twelve crates on Monday.", "Cargo of nine boxes went out Tuesday.")]]
    out = S.calibrate(pairs, HashEmbedder())
    assert out["pairs"] == 2 and out["paraphrase_cos"][0] > out["paraphrase_cos"][1] and len(out["paraphrase_lexical"]) == 2


def test_cli_roundtrip_with_a_fake_embedder_and_variety_gate():
    d = tempfile.mkdtemp(); E = HashEmbedder()
    tr = os.path.join(d, "train.jsonl"); open(tr, "w").write("".join(json.dumps(dict(text=t)) + "\n" for t in TRAIN))
    ev = os.path.join(d, "eval.jsonl"); open(ev, "w").write(json.dumps(dict(said="Nothing here resembles anything else in the whole collection.")) + "\n")
    rows = os.path.join(d, "rows.jsonl"); open(rows, "w").write("".join(json.dumps(dict(n=i, said=t)) + "\n" for i, t in enumerate(["The ferry crosses the bay twice a day when the weather is calm.", "Bring your own towels to the pool on Saturday afternoon."])))
    idx = os.path.join(d, "cache")
    with contextlib.redirect_stdout(io.StringIO()):
        assert S.main(["index", "--texts", tr, "--name", "train", "--out", idx], E) == 0
        assert S.main(["index", "--texts", ev, "--name", "eval", "--out", idx], E) == 0
        assert S.main(["check", "--rows", rows, "--index", idx, "--out", os.path.join(d, "sem.json")], E) == 0
    sem = json.load(open(os.path.join(d, "sem.json")))
    assert sem["train_max"] > 0.99 and sem["rows"] == 2
    gate = V.semantic_gate(sem, V.merged(None))
    assert gate[0][1] is False                                        # a row at cosine ~1 against the index is a hard hit
    clean = dict(sem, train_max=0.5, max_any=0.5, train_p99=0.5, within_p99=0.4, eval_p99=0.1)
    assert V.semantic_gate(clean, V.merged(None))[0][1] is True


def test_real_minilm_when_the_weights_are_present():
    try:
        from minilm_np import Embedder
        E = Embedder()
    except FileNotFoundError:
        print("  (skipped: MiniLM weights not found)")
        return
    X = E.encode(["The meter reads forty two units today.", "The gauge shows forty two units right now.", "A ship left the harbour at dawn."])
    assert X.shape == (3, 384) and abs(float(np.linalg.norm(X[0])) - 1) < 1e-4
    assert float(X[0] @ X[1]) > 0.6 > float(X[0] @ X[2])
    assert np.allclose(E.encode(["The meter reads forty two units today."])[0], X[0], atol=1e-5)          # batch size does not matter


if __name__ == "__main__":
    import tinytest; tinytest.run(globals())
