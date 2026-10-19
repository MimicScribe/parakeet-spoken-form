"""Measured exposure (exposure.py, ft_train.SeenRows, exposure_report.py) against a fake dataloader; no GPU, no NeMo.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    python -m pytest test_exposure.py -q
"""
import json
import os
import random

import torch

import exposure as E
import exposure_report as R
import ft_train as F


def tok(text):
    return [ord(c) % 50 + 1 for c in text]


def make_mix(tmp_path, texts_by_part):
    """A tiny volume: one manifest per part, rows carrying lang/spans like the p3 manifests."""
    vol = tmp_path / "vol"
    for part, rows in texts_by_part.items():
        os.makedirs(os.path.dirname(f"{vol}/{part}"), exist_ok=True)
        with open(f"{vol}/{part}", "w") as f:
            for i, (text, dur, lang, kinds) in enumerate(rows):
                f.write(json.dumps({"audio_filepath": f"/a/{part}/{i}.wav", "duration": dur, "text": text, "lang": lang,
                                    "spans": [{"kind": k} for k in kinds]}) + "\n")
    return str(vol)


PARTS = {
    "mix4/p3_de.jsonl": [(f"zwei und {'x' * i} dreissig", 2.0 + i / 10, "de", ["cardinal"]) for i in range(8)],
    "mix4/p3_fr.jsonl": [(f"il est {'y' * i} cinq heures", 3.0, "fr", ["clock"]) for i in range(6)],
    "replay/lspc.jsonl": [(f"hello {'z' * i} world", 1.5, None, []) for i in range(5)],
}


def build(tmp_path, tags=None):
    vol = make_mix(tmp_path, PARTS)
    meta = []
    out = str(tmp_path / "train_manifest.jsonl")
    mix = [(p, 0, (tags or {}).get(p)) if tags else (p, 0) for p in PARTS]
    F.write_mix(vol, out, mix, seed=3, meta_out=meta)
    rows = [json.loads(l) for l in open(out)]
    return rows, meta


class FakeLoader:
    """What the trainer's dataloader yields: (signal, signal_len, tokens, tokens_len), lhotse-style batches drawn
    from the manifest rows (with repeats, as a sampler that cycles)."""

    def __init__(self, rows, per_batch, n_batches, seed=0):
        r = random.Random(seed)
        self.drawn = [[r.randrange(len(rows)) for _ in range(per_batch)] for _ in range(n_batches)]
        self.rows = rows

    def __iter__(self):
        for idxs in self.drawn:
            toks = [tok(self.rows[i]["text"]) for i in idxs]
            w = max(map(len, toks))
            yield (torch.zeros(len(idxs), 10),
                   torch.tensor([round(self.rows[i]["duration"] * 16000) for i in idxs]),
                   torch.tensor([t + [0] * (w - len(t)) for t in toks]), torch.tensor([len(t) for t in toks]))


class FakeTrainer:
    global_step = 0


def run(exp, rows, meta, loader, accumulate, step0=0, sha="abc123def456", crash_after=None):
    E.write_row_index(exp, sha, rows, meta, tok)
    cb = F.SeenRows(exp, sha)
    t = FakeTrainer()
    t.global_step = step0
    cb.on_train_start(t, None)
    for k, batch in enumerate(loader):
        if crash_after is not None and k == crash_after:
            cb.on_exception(t, None, RuntimeError("preempted"))
            return t
        cb.on_train_batch_start(t, None, batch, k)
        if (k + 1) % accumulate == 0:
            t.global_step += 1  # the optimizer step completes before on_train_batch_end
        cb.on_train_batch_end(t, None, None, batch, k)
    cb.on_train_end(t, None)
    return t


def test_write_mix_meta_matches_manifest_order_and_changes_nothing(tmp_path):
    rows, meta = build(tmp_path, tags={"mix4/p3_de.jsonl": "nonen_span"})
    assert len(rows) == len(meta) == 19
    for x, m in zip(rows, meta):  # the part of each written row is the part its label came from
        part = m["part"]
        assert any(x["text"] == y[0] for y in PARTS[part]), (x, m)
    assert {m["tag"] for m in meta} == {"nonen_span", ""}
    assert {(m["lang"], tuple(m["kinds"])) for m in meta if m["part"] == "mix4/p3_fr.jsonl"} == {("fr", ("clock",))}
    assert {m["lang"] for m in meta if m["part"] == "replay/lspc.jsonl"} == {"en"}
    # the manifest is byte-identical with and without meta_out
    vol = str(tmp_path / "vol")
    F.write_mix(vol, str(tmp_path / "plain.jsonl"), [(p, 0, "nonen_span") if "de" in p else (p, 0) for p in PARTS], seed=3)
    assert open(tmp_path / "plain.jsonl").read() == open(tmp_path / "train_manifest.jsonl").read()


def test_exposure_counts_equal_what_the_loader_yielded(tmp_path):
    rows, meta = build(tmp_path)
    loader = FakeLoader(rows, per_batch=4, n_batches=30)
    run(str(tmp_path), rows, meta, loader, accumulate=2)
    e = E.exposure(str(tmp_path))
    assert e["steps"] == 15 and e["rows_drawn"] == 120 and e["unmatched"] == 0
    want = {}
    for idxs in loader.drawn:
        for i in idxs:
            want[meta[i]["part"]] = want.get(meta[i]["part"], 0) + 1
    assert {k: round(v["draws"]) for k, v in e["parts"].items()} == want
    assert {k: v["listed_rows"] for k, v in e["parts"].items()} == {"mix4/p3_de.jsonl": 8, "mix4/p3_fr.jsonl": 6,
                                                                     "replay/lspc.jsonl": 5}
    assert round(e["cells"]["de/cardinal"]["draws"]) == want["mix4/p3_de.jsonl"]
    assert round(e["cells"]["fr/clock"]["draws"]) == want["mix4/p3_fr.jsonl"]
    assert abs(e["seconds_drawn"] - sum(rows[i]["duration"] for b in loader.drawn for i in b)) < 0.5
    txt = E.report(str(tmp_path))
    assert "de/cardinal" in txt and "15 optimizer steps" in txt


def test_one_line_per_optimizer_step_with_all_micro_batches(tmp_path):
    rows, meta = build(tmp_path)
    run(str(tmp_path), rows, meta, FakeLoader(rows, 3, 8), accumulate=4)
    recs = [json.loads(l) for l in open(tmp_path / "seen_rows.jsonl")]
    assert recs[0] == {"start": 0, "manifest": "abc123def456"}
    assert [(r["step"], len(r["h"])) for r in recs[1:]] == [(0, 12), (1, 12)]


def test_resume_drops_the_redone_steps(tmp_path):
    rows, meta = build(tmp_path)
    loader = FakeLoader(rows, 2, 20, seed=1)
    run(str(tmp_path), rows, meta, loader, accumulate=1, crash_after=10)  # steps 0..9 logged, then killed
    # the checkpoint was at step 6: the resumed attempt redoes steps 6.. with other batches
    resumed = FakeLoader(rows, 2, 6, seed=2)
    run(str(tmp_path), rows, meta, resumed, accumulate=1, step0=6)
    steps = [r["step"] for _, r in E.effective_steps(open(tmp_path / "seen_rows.jsonl").read().splitlines())]
    assert steps == list(range(12))  # 0..5 from attempt 1, 6..11 from attempt 2, none twice
    e = E.exposure(str(tmp_path))
    assert e["rows_drawn"] == 24
    want = sum(1 for b in loader.drawn[:6] + resumed.drawn for _ in b)
    assert want == 24


def test_duplicate_labels_are_split_by_duration_then_evenly(tmp_path):
    vol = make_mix(tmp_path, {"mix4/p3_de.jsonl": [("gleich", 2.0, "de", ["cardinal"]), ("gleich", 5.0, "de", ["cardinal"])],
                              "mix4/p3_fr.jsonl": [("gleich", 2.0, "fr", ["clock"]), ("andere", 1.0, "fr", ["clock"])]})
    meta, out = [], str(tmp_path / "train_manifest.jsonl")
    F.write_mix(vol, out, [("mix4/p3_de.jsonl", 0), ("mix4/p3_fr.jsonl", 0)], seed=1, meta_out=meta)
    rows = [json.loads(l) for l in open(out)]
    five = next(i for i, x in enumerate(rows) if x["duration"] == 5.0)
    two = next(i for i, x in enumerate(rows) if x["text"] == "gleich" and x["duration"] == 2.0)
    other = next(i for i, x in enumerate(rows) if x["text"] == "andere")
    loader = FakeLoader(rows, 1, 3)
    loader.drawn = [[five], [five], [other]]
    run(str(tmp_path), rows, meta, loader, accumulate=1)
    e = E.exposure(str(tmp_path))
    assert e["ambiguous"] == 0 and round(e["parts"]["mix4/p3_de.jsonl"]["draws"]) == 2  # 5.0 s picks the de row alone
    loader.drawn = [[two]]
    os.remove(tmp_path / "seen_rows.jsonl")
    run(str(tmp_path), rows, meta, loader, accumulate=1)
    e = E.exposure(str(tmp_path))
    assert e["ambiguous"] == 1  # a 2.0 s "gleich" fits a de and a fr row: half a draw each
    assert e["parts"]["mix4/p3_de.jsonl"]["draws"] == 0.5 and e["parts"]["mix4/p3_fr.jsonl"]["draws"] == 0.5


def test_a_failing_log_never_stops_the_run(tmp_path, capsys):
    cb = F.SeenRows(str(tmp_path / "no" / "such" / "dir"), "x")
    t = FakeTrainer()
    cb.on_train_start(t, None)
    cb.on_train_batch_start(t, None, (torch.zeros(1, 1), torch.tensor([16000]), torch.tensor([[1]]), torch.tensor([1])), 0)
    cb.on_train_end(t, None)
    assert "SEEN ROWS LOG FAILED" in capsys.readouterr().out


def test_report_cli_warns_on_a_manifest_that_is_not_the_logged_one(tmp_path, capsys):
    import hashlib

    rows, meta = build(tmp_path)
    sha = hashlib.sha256(open(tmp_path / "train_manifest.jsonl", "rb").read()).hexdigest()[:12]
    run(str(tmp_path), rows, meta, FakeLoader(rows, 2, 4), accumulate=1, sha=sha)
    R.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "WARNING" not in out and "PART" in out
    with open(tmp_path / "train_manifest.jsonl", "a") as f:
        f.write(json.dumps({"audio_filepath": "x", "duration": 1.0, "text": "extra"}) + "\n")
    R.main([str(tmp_path)])
    assert "WARNING: train_manifest.jsonl sha" in capsys.readouterr().out
