"""R4 rehearsal losses (arm b: suffix anchoring + gate; arm c: cross-window consistency) on synthetic tensors.
NeMo is not needed: the loss pieces are pure-torch helpers in ft_train.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    python -m pytest test_r4_losses.py -q
"""
import json

import pytest
import torch

import ft_train as F


class FakeSPM:
    """Word -> pieces: "▁" + the first 5 letters, then 5-letter continuation pieces (so long words span pieces)."""

    def __init__(self):
        self.pieces = ["<unk>"]
        self.index = {}

    def _id(self, piece):
        if piece not in self.index:
            self.index[piece] = len(self.pieces)
            self.pieces.append(piece)
        return self.index[piece]

    def encode(self, text):
        ids = []
        for w in text.split():
            ids.append(self._id("▁" + w[:5]))
            for k in range(5, len(w), 5):
                ids.append(self._id(w[k:k + 5]))
        return ids

    def id_to_piece(self, i):
        return self.pieces[i]

    def get_piece_size(self):
        return len(self.pieces)


@pytest.fixture
def spm():
    return FakeSPM()


def positions_by_word(spm, text, keep):
    """keep (n_ids + 1) -> (per-word True/False when all of the word's positions agree, trailing flag)."""
    ids = spm.encode(text)
    words = F.row_words(spm, ids)
    out = []
    for _, a, b in words:
        vals = set(keep[a:b + 1])
        assert len(vals) == 1, "a word's pieces must share one anchoring decision"
        out.append(vals.pop())
    return out, keep[len(ids)]


# ---------------------------------------------------------------- arm b: masks per tag
def test_span_arm_a_prefix_only(spm):
    text = "so in the year twenty nineteen we grew a lot"
    ids = spm.encode(text)
    keep = F.klu_row(spm, "span", ids)
    words, trail = positions_by_word(spm, text, keep)
    # prefix before the first number word minus one word ("year"), nothing after it
    assert words == [True, True, True, False, False, False, False, False, False, False]
    assert not trail


def test_span_arm_b_suffix(spm):
    text = "so in the year twenty nineteen we grew a lot"
    ids = spm.encode(text)
    keep = F.klu_row(spm, "span", ids, suffix_anchor=True)
    words, trail = positions_by_word(spm, text, keep)
    # number words free, one word before ("year") and one after ("we") free, everything else anchored + the end
    assert words == [True, True, True, False, False, False, False, True, True, True]
    assert trail


def test_span_arm_b_after_guard_param(spm):
    text = "we sold twenty units to them"
    ids = spm.encode(text)
    w0, _ = positions_by_word(spm, text, F.klu_row(spm, "span", ids, suffix_anchor=True, anchor_after=0))
    w2, _ = positions_by_word(spm, text, F.klu_row(spm, "span", ids, suffix_anchor=True, anchor_after=2))
    assert w0 == [True, False, False, True, True, True]
    assert w2 == [True, False, False, False, False, True]


def test_number_last_word_leaves_end_free(spm):
    text = "the total was forty"
    ids = spm.encode(text)
    words, trail = positions_by_word(spm, text, F.klu_row(spm, "span", ids, suffix_anchor=True))
    assert words == [True, True, False, False]
    assert not trail  # the end position follows the number directly: inside the after-guard


def test_two_number_runs(spm):
    text = "about five people and then more than thirty came home"
    ids = spm.encode(text)
    words, trail = positions_by_word(spm, text, F.klu_row(spm, "span", ids, suffix_anchor=True))
    assert words == [False, False, False, True, True, True, False, False, False, True]
    assert trail


def test_anchor_num_arm_a_free_arm_b_anchored(spm):
    text = "World War II changed everything for the country"
    ids = spm.encode(text)
    assert not any(F.klu_row(spm, "anchor_num", ids))
    words, trail = positions_by_word(spm, text, F.klu_row(spm, "anchor_num", ids, suffix_anchor=True))
    assert words == [True, False, False, False, True, True, True, True]  # "War" before, "changed" after
    assert trail


def test_anchor_and_nonen_full_free_none(spm):
    ids = spm.encode("twenty people came")
    assert all(F.klu_row(spm, "anchor", ids)) and all(F.klu_row(spm, "nonen", ids))
    assert not any(F.klu_row(spm, "free", ids))
    assert not any(F.klu_row(spm, "free", ids, suffix_anchor=True))


def test_en_number_words_wider_than_number_span():
    w = "I came second and one of them said Q one in World War I".split()
    f = F.en_number_words(w)
    assert f[2] and f[4]  # numberish: "second", "one" (over-flagging only loses anchoring)
    assert f[-1]  # "War I"
    assert not f[0]  # a plain "I"


def test_word_flags_num_chars_and_num_tokens():
    text = "wir haben zweiundzwanzig Euro bezahlt"
    a = text.index("zweiundzwanzig")
    assert F.word_flags({"text": text, "num_chars": [[a, a + len("zweiundzwanzig Euro")]]}) == [False, False, True, True, False]
    assert F.word_flags({"text": text, "num_tokens": [0, 0, 1, 0, 0]}) == [False, False, True, False, False]
    with pytest.raises(ValueError):
        F.word_flags({"text": text, "num_tokens": [0, 1]})
    with pytest.raises(ValueError):
        F.word_flags({"text": text})


def nspan_inputs(spm, text, flags):
    """What train() builds for one nonen_span label: the arm-a prefix count and the arm-b positions."""
    ids = spm.encode(text)
    pre = F.nspan_prefix({"text": text, "num_tokens": flags})
    pw = pre.split() if pre == text else pre.split()[:-1]
    span_prefix = {tuple(ids): len(spm.encode(" ".join(pw))) if pw else 0}
    pos = F.anchor_positions(F.row_words(spm, ids), flags, len(ids))
    return ids, span_prefix, {tuple(ids): pos}


def test_nonen_span_arm_a_prefix_arm_b_suffix(spm):
    text = "nous avons payé vingt-deux euros hier soir pour cela"
    flags = [False, False, False, True, True, False, False, False, False]
    ids, span_prefix, nspan_pos = nspan_inputs(spm, text, flags)
    a, _ = positions_by_word(spm, text, F.klu_row(spm, "nonen_span", ids, span_prefix=span_prefix, nspan_pos=nspan_pos))
    assert a == [True, True, False, False, False, False, False, False, False]  # arm a ignores nspan_pos
    b, trail = positions_by_word(spm, text, F.klu_row(spm, "nonen_span", ids, suffix_anchor=True,
                                                      span_prefix=span_prefix, nspan_pos=nspan_pos))
    assert b == [True, True, False, False, False, False, True, True, True]
    assert trail
    # arm b on the prefix part matches arm a exactly (the prefix rule is a special case)
    pa = F.klu_row(spm, "nonen_span", ids, span_prefix=span_prefix)
    pb = F.klu_row(spm, "nonen_span", ids, suffix_anchor=True, span_prefix=span_prefix, nspan_pos=nspan_pos)
    k = sum(pa)
    assert pb[:k] == pa[:k]


def test_nonen_span_missing_positions_falls_back_to_prefix(spm):
    text = "nous avons payé vingt euros"
    ids, span_prefix, _ = nspan_inputs(spm, text, [False, False, False, True, True])
    assert F.klu_row(spm, "nonen_span", ids, suffix_anchor=True, span_prefix=span_prefix, nspan_pos={}) == \
        F.klu_row(spm, "nonen_span", ids, span_prefix=span_prefix)


def test_klu_cache(spm):
    ids = spm.encode("we paid twenty dollars")
    cache = {}
    first = F.klu_row(spm, "span", ids, suffix_anchor=True, cache=cache)
    assert tuple(ids) in cache and F.klu_row(spm, "span", ids, suffix_anchor=True, cache=cache) == first


# ---------------------------------------------------------------- arm b: gate threshold
def test_row_gate():
    tags = ["anchor", "anchor_num", "nonen_span", "nonen_span", "span", "nonen", "free"]
    keys = [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    assert F.row_gate(tags, keys, 0.9, 0.0, set()) == [0.9] * 7
    assert F.row_gate(tags, keys, 0.9, 0.5, {(4,)}) == [0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]


def test_blank_gate_threshold():
    p_nb = torch.tensor([[[0.95, 0.7, 0.3]], [[0.95, 0.7, 0.3]]])  # (B=2, T=1, U=3)
    t_lnb = p_nb.log()
    kmask = torch.ones(2, 1, 3, dtype=torch.bool)
    g = F.blank_gate(kmask, t_lnb, torch.tensor([0.9, 0.5]))
    assert g[0, 0].tolist() == [True, False, False]
    assert g[1, 0].tolist() == [True, True, False]
    kmask[1, 0, 0] = False
    assert F.blank_gate(kmask, t_lnb, torch.tensor([0.9, 0.5]))[1, 0].tolist() == [False, True, False]


def test_per_tag_summary():
    tags = ["anchor", "nonen_span", "anchor"]
    acc = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 2.0])] + [torch.zeros(3)] * 12
    s = F.per_tag_summary(tags, acc)
    assert s["anchor"]["rows"] == 2 and s["anchor"]["kl_b"] == pytest.approx(1.0) and s["anchor"]["kl_b_n"] == 4
    assert s["nonen_span"]["kl_b"] == pytest.approx(0.5)
    assert s["nonen_span"]["kl_t"] == 0.0 and s["nonen_span"]["kl_t_n"] == 0


# ---------------------------------------------------------------- arm c: consistency loss
NB = 9  # 8 tokens + blank; logits carry 3 extra duration outputs


def logits_for(seed, T=6, P=3, V=NB + 3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, P, V, generator=g)


def test_consistency_zero_for_identical():
    lg = logits_for(1)
    labels = torch.tensor([2, 5, 1])
    la = F.emission_logdist(lg, labels, 6, NB)
    assert la.shape == (3, NB - 1)
    assert torch.allclose(la.exp().sum(-1), torch.ones(3), atol=1e-5)
    assert F.sym_kl(la, F.emission_logdist(lg.clone(), labels, 6, NB)).item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_positive_and_symmetric():
    labels = torch.tensor([2, 5, 1])
    la = F.emission_logdist(logits_for(1), labels, 6, NB)
    lb = F.emission_logdist(logits_for(2, T=8), labels, 5, NB)  # B: another window, other length
    ab, ba = F.sym_kl(la, lb), F.sym_kl(lb, la)
    assert ab.item() > 1e-3
    assert ab.item() == pytest.approx(ba.item(), rel=1e-6)


def test_consistency_gradient_flows_to_both_sides():
    labels = torch.tensor([2, 5, 1])
    a = logits_for(1).requires_grad_(True)
    b = logits_for(2).requires_grad_(True)
    F.sym_kl(F.emission_logdist(a, labels, 6, NB), F.emission_logdist(b, labels, 6, NB)).backward()
    assert a.grad is not None and a.grad.abs().sum() > 0 and torch.isfinite(a.grad).all()
    assert b.grad is not None and b.grad.abs().sum() > 0 and torch.isfinite(b.grad).all()
    assert a.grad[..., NB:].abs().sum() == 0  # the duration head is not compared


def test_emission_weights_ignore_padded_frames():
    labels = torch.tensor([2, 5, 1])
    lg = logits_for(3)
    lg2 = lg.clone()
    lg2[4:] = torch.randn(2, 3, NB + 3) * 10  # frames past t_len = 4
    assert torch.allclose(F.emission_logdist(lg, labels, 4, NB), F.emission_logdist(lg2, labels, 4, NB), atol=1e-6)


def test_emission_point_follows_label_posterior():
    """The compared distribution is the token distribution where the label is most likely emitted."""
    T, P, V = 5, 1, NB + 3
    lg = torch.zeros(T, P, V)
    lg[2, 0, 3] = 20.0  # frame 2 emits token 3 with near certainty
    lg[0, 0, 6] = 5.0  # frame 0 prefers token 6 but the label is 3
    d = F.emission_logdist(lg, torch.tensor([3]), T, NB).exp()
    assert d[0].argmax().item() == 3
    # w_t is proportional to P(y_u | t, u) (full softmax incl. blank), the mixture is over P(. | t, u, non-blank)
    p_y = lg[:, 0, :NB].softmax(-1)[:, 3]
    w = p_y / p_y.sum()
    want = (w[:, None] * lg[:, 0, :NB - 1].softmax(-1)).sum(0)
    assert torch.allclose(d[0], want, atol=1e-5)


# ---------------------------------------------------------------- arm c: pair list format + sampler
def pair(a_text, b_text, a_span, b_span, reading, **kw):
    return json.dumps({"a": {"audio_filepath": "/vol/a.wav", "duration": 9.0, "text": a_text},
                       "b": {"audio_filepath": "/vol/b.wav", "duration": 14.96, "text": b_text},
                       "a_span": a_span, "b_span": b_span, "reading": reading, "parent": "row1", "lang": "de", **kw})


def test_load_pairs_valid_and_skips_unknown():
    a, b = "wir zahlen zwei hundert Euro", "zwei hundert Euro im Jahr"
    lines = [pair(a, b, [2, 4], [0, 2], "zwei hundert"), "", pair(a, "other text zwei hundert", [2, 4], [2, 4], "zwei hundert")]
    pairs, st = F.load_pairs(lines, known_texts={a, b})
    assert len(pairs) == 1 and st == {"kept": 1, "not_in_mix": 1}
    pairs, st = F.load_pairs(lines)
    assert len(pairs) == 2


def test_load_pairs_rejects_bad_span():
    with pytest.raises(ValueError):
        F.load_pairs([pair("wir zahlen zwei hundert Euro", "zwei hundert Euro", [1, 3], [0, 2], "zwei hundert")])
    with pytest.raises(ValueError):
        F.load_pairs([pair("wir zahlen zwei", "zwei Euro", [2, 4], [0, 1], "zwei")])  # out of range
    bad = json.loads(pair("zwei Euro", "zwei Euro", [0, 1], [0, 1], "zwei"))
    del bad["b"]["audio_filepath"]
    with pytest.raises(ValueError):
        F.load_pairs([json.dumps(bad)])


def test_span_label_positions(spm):
    text = "wir zahlen zweiundzwanzig Euro"
    ids = spm.encode(text)
    pos = F.span_label_positions(spm, ids, text, [2, 3])
    words = F.row_words(spm, ids)
    assert pos == list(range(words[2][1], words[2][2] + 1)) and len(pos) == 3  # a 3-piece word
    assert F.span_label_positions(spm, ids, text + " extra", [2, 3]) is None  # word split does not line up


def test_pair_bank(tmp_path):
    staged = tmp_path / "stage"
    (staged / "vol").mkdir(parents=True)
    (staged / "vol" / "a.wav").write_bytes(b"")
    bank = F.PairBank([{"x": 1}, {"x": 2}], per_batch=2, seed=3, stage_dir=str(staged))
    s = bank.sample()
    assert len(s) == 2 and all(x in bank.pairs for x in s)
    assert bank.path("/vol/a.wav") == str(staged) + "/vol/a.wav"
    assert bank.path("/vol/b.wav") == "/vol/b.wav"  # not staged: the volume path
    assert F.PairBank([], per_batch=2).sample() == []
    s1 = F.PairBank(list(range(50)), per_batch=4, seed=7).sample()
    assert s1 == F.PairBank(list(range(50)), per_batch=4, seed=7).sample()  # seeded


# ---------------------------------------------------------------- smoke: add_r3_losses' training step on a tiny fake model
class TinySPM(FakeSPM):
    """FakeSPM inside a full-size vocabulary (8192 pieces + blank), as add_r3_losses indexes it."""

    def __init__(self):
        super().__init__()
        for p in (".", "▁twenty", "▁thousand", "5", "▁two"):
            self._id(p)

    def id_to_piece(self, i):
        return self.pieces[i] if i < len(self.pieces) else f"<|x{i}|>"

    def decode(self, ids):
        return "".join(self.id_to_piece(int(i)) for i in ids).replace("\u2581", " ").strip()

    def get_piece_size(self):
        return 8192


class Tok:
    def __init__(self):
        self.tokenizer = TinySPM()

    def text_to_ids(self, t):
        return self.tokenizer.encode(t)


class Pre(torch.nn.Module):
    def forward(self, input_signal, length):
        n = input_signal.shape[1] // 160
        x = input_signal[:, :n * 160].reshape(input_signal.shape[0], n, 160)[..., :4].transpose(1, 2)
        return x, length // 160


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 8)

    def forward(self, audio_signal, length):
        return self.lin(audio_signal.transpose(1, 2)).transpose(1, 2), length


class Dec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8193, 8)
        self.lstm = torch.nn.LSTM(8, 8, batch_first=True)

    def forward(self, targets, target_length):
        x = torch.cat([torch.zeros(targets.shape[0], 1, 8), self.emb(targets)], 1)
        y, _ = self.lstm(x)
        return y.transpose(1, 2), target_length, None


class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_net = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(8, 8198))

    def joint(self, f, g):
        return self.joint_net(f[:, :, None, :] + g[:, None, :, :])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.preprocessor, self.encoder, self.decoder, self.joint = Pre(), Enc(), Dec(), Joint()
        self.tokenizer = Tok()

    def training_step(self, batch, batch_idx):
        return {"loss": (self.joint.joint_net[-1].bias ** 2).mean()}


def fake_batch(m, texts):
    ids = [m.tokenizer.text_to_ids(t) for t in texts]
    tok = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
    for i, x in enumerate(ids):
        tok[i, :len(x)] = torch.tensor(x)
    sig = torch.randn(len(ids), 160 * 12, generator=torch.Generator().manual_seed(1))
    return (sig, torch.tensor([160 * 12, 160 * 10, 160 * 12][:len(ids)]), tok, torch.tensor([len(x) for x in ids])), ids


def test_training_step_smoke_arms(tmp_path, capsys):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    texts = ["so in the year twenty nineteen we grew", "nous avons payé vingt euros hier", "World War II changed it"]
    for arm in ("a", "b", "c"):
        m = FakeModel()
        batch, ids = fake_batch(m, texts)
        tags = {tuple(ids[0]): "span", tuple(ids[1]): "nonen_span", tuple(ids[2]): "anchor_num"}
        kw = {}
        if arm == "b":
            words = F.row_words(m.tokenizer.tokenizer, ids[1])
            kw = dict(suffix_anchor=True, gate_human=0.5, nspan_pos={
                tuple(ids[1]): F.anchor_positions(words, [False, False, False, True, True, False], len(ids[1]))})
        if arm == "c":
            for name, n in (("a", 160 * 9), ("b", 160 * 11)):
                sf.write(str(tmp_path / f"{name}.wav"), np.random.RandomState(n).randn(n).astype("float32") * 0.1, 16000)
            p = json.loads(pair("wir zahlen zwei hundert Euro", "zwei hundert Euro im Jahr", [2, 4], [0, 2], "zwei hundert"))
            p["a"]["audio_filepath"], p["b"]["audio_filepath"] = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
            kw = dict(cons_weight=1.0, pair_bank=F.PairBank([p], per_batch=2, seed=1))
        F.add_r3_losses(m, tags, span_prefix={tuple(ids[1]): 3}, **kw)
        out = m.training_step(batch, 0)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert m.decoder.lstm.weight_hh_l0.grad is not None and m.decoder.lstm.weight_hh_l0.grad.abs().sum() > 0
        log = capsys.readouterr().out
        assert "r3tag span" in log and "r3tag nonen_span" in log and "r3tag anchor_num" in log
        if arm == "c":
            assert m._r3_parts["cons"].item() > 0 and "pairs 2" in log
        else:
            assert m._r3_parts["cons"].item() == 0
