"""unfreeze_top (the last N encoder blocks trainable) on a small fake model. NeMo is not needed.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    python -m pytest test_unfreeze_top.py -q
"""
import copy

import torch

import ft_eval
import ft_train as F
import test_r4_losses as T


class Block(torch.nn.Module):
    """A conformer-block stand-in: a linear map, BatchNorm (running stats) and dropout."""

    def __init__(self, d=8):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.bn = torch.nn.BatchNorm1d(d)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):  # [B, D, T]
        return x + self.drop(self.bn(self.lin(x.transpose(1, 2)).transpose(1, 2)))


class Enc(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.pre_encode = torch.nn.Linear(4, 8)
        self.layers = torch.nn.ModuleList([Block() for _ in range(n)])

    def forward(self, audio_signal, length):
        x = self.pre_encode(audio_signal.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x, length

    def freeze(self):  # NeMo's NeuralModule.freeze
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


class Model(T.FakeModel):
    """The main loss reads the encoder through the model's forward (as NeMo's training_step does)."""

    def __init__(self, main_scale=1.0):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = Enc()
        self.main_scale = main_scale  # 0: the main loss keeps its graph but sends no gradient

    def training_step(self, batch, batch_idx):
        signal, signal_len, tokens, tokens_len = batch[:4]
        feats, flen = self.preprocessor(input_signal=signal, length=signal_len)
        enc, _ = self.encoder(audio_signal=feats, length=flen)
        g, _, _ = self.decoder(targets=tokens, target_length=tokens_len)
        lg = self.joint.joint(enc.transpose(1, 2), g.transpose(1, 2))
        return {"loss": -self.main_scale * lg.log_softmax(-1)[..., 8192].mean()}


TEXTS = ["wir zahlen sieben und dreißig euro", "nous avons payé vingt euros hier", "so in the year twenty nineteen"]


def setup(n_top, perturb=False, main_scale=1.0):
    m = Model(main_scale)
    batch, ids = T.fake_batch(m, TEXTS)
    m.encoder.freeze()
    stock = F.unfreeze_top_layers(m, n_top) if n_top else None
    if perturb:  # the top blocks as after some training
        with torch.no_grad():
            for p in m.encoder.layers[-1].parameters():
                p.add_(0.3)
    tags = {tuple(ids[0]): "nonen_span", tuple(ids[1]): "nonen", tuple(ids[2]): "span"}
    F.add_r3_losses(m, tags, span_prefix={tuple(ids[0]): 3}, sub_batch=1)
    m.train()
    m.encoder.eval()  # KeepEncoderFrozen
    return m, batch, stock


def names(m, cond):
    return {n for n, p in m.named_parameters() if cond(p)}


def test_default_freezes_the_whole_encoder():
    # red-on-base: control (unfreeze_top 0 must leave the frozen-encoder behaviour as it was)
    m, batch, stock = setup(0)
    assert stock is None and not getattr(m, "_enc_unfrozen", ())
    assert not [n for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    m.training_step(batch, 0)["loss"].backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    assert not [k for k in F.trainable_state(m) if k.startswith("encoder.")]


def test_exactly_the_top_blocks_train():
    m, batch, stock = setup(2)
    want = {n for n, _ in m.named_parameters() if n.startswith(("encoder.layers.2.", "encoder.layers.3."))}
    assert names(m, lambda p: p.requires_grad) & {n for n, _ in m.named_parameters() if n.startswith("encoder.")} == want
    assert set(stock) == {k for k in m.state_dict() if k.startswith(("encoder.layers.2.", "encoder.layers.3."))}
    assert "encoder.layers.3.bn.running_mean" in stock  # buffers ride along (fixed: eval mode)
    before = copy.deepcopy(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)  # every parameter, as NeMo builds its groups
    out = m.training_step(batch, 0)
    out["loss"].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n in want)
    opt.step()
    after = m.state_dict()
    for k, v in before.items():
        moved = not torch.equal(v, after[k])
        if k in want:
            assert moved, k
        elif k.startswith("encoder."):  # frozen tensors and every BatchNorm statistic stay put
            assert not moved, k
    assert not m.encoder.training and all(not p.requires_grad for p in m.encoder.layers[1].parameters())


def test_teacher_keeps_the_stock_blocks():
    # at stock the teacher equals the student: every KL term equals the frozen-encoder run's, bit for bit (float noise
    # of ~1e-8, not 0, in both)
    m, batch, _ = setup(2)
    m.training_step(batch, 0)
    m0, batch0, _ = setup(0)
    m0.training_step(batch0, 0)
    for k in ("kl_b", "kl_p", "kl_t", "kl_d"):
        assert torch.equal(m._r3_parts[k], m0._r3_parts[k]), k
    assert m._r3_parts["kl_b"].item() < 1e-6
    # after the top blocks moved, the teacher still reads the stock blocks: the KL terms see the drift
    m2, batch2, _ = setup(2, perturb=True)
    ref = Model()
    ref.encoder.eval()
    feats, flen = m2.preprocessor(input_signal=batch2[0], length=batch2[1])
    enc_t, _ = F.stock_top_encoder(m2, feats, flen)
    assert torch.allclose(enc_t, ref.encoder(audio_signal=feats, length=flen)[0])
    assert not torch.allclose(enc_t, m2.encoder(audio_signal=feats, length=flen)[0])
    m2.training_step(batch2, 0)["loss"].backward()
    assert m2._r3_parts["kl_b"].item() > 100 * m0._r3_parts["kl_b"].item()
    # the auxiliary losses alone reach the unfrozen blocks (the student side keeps the training pass's graph)
    m3, batch3, _ = setup(2, perturb=True, main_scale=0.0)
    m3.training_step(batch3, 0)["loss"].backward()
    g = m3.encoder.layers[-1].lin.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_checkpoint_round_trip_and_loader():
    m, batch, _ = setup(2, perturb=True)
    sd = F.trainable_state(m)
    enc_keys = {k for k in sd if k.startswith("encoder.")}
    assert enc_keys == {k for k in m.state_dict() if k.startswith(("encoder.layers.2.", "encoder.layers.3."))}
    assert any(k.startswith("decoder.") for k in sd) and any(k.startswith("joint.") for k in sd)
    # a reader: a fresh stock model, stock = decoder + joint only (as _stock_model)
    r = Model()
    r.eval()
    stock = {k: v.clone() for k, v in r.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
    feats, flen = r.preprocessor(input_signal=batch[0], length=batch[1])
    with torch.no_grad():
        e_stock = r.encoder(audio_signal=feats, length=flen)[0].clone()
        assert ft_eval.load_over_stock(r, stock, sd) == len(enc_keys)
        for k in enc_keys:
            assert torch.equal(r.state_dict()[k], sd[k])
        e_run = r.encoder(audio_signal=feats, length=flen)[0]
        assert not torch.allclose(e_run, e_stock)  # decoding the checkpoint reads its own encoder
        # the next run ("ultra" here) gets the stock blocks back
        assert ft_eval.load_over_stock(r, stock, {}) == 0
        assert torch.equal(r.encoder(audio_signal=feats, length=flen)[0], e_stock)


def test_l2sp_anchors_the_encoder_blocks_to_stock(tmp_path):
    m, batch, stock = setup(2, perturb=True)
    anchor = tmp_path / "state.pt"
    torch.save({k: v for k, v in m.state_dict().items() if k.startswith(("decoder.", "joint."))}, anchor)

    class Tr:
        accumulate_grad_batches = 1
        optimizers = [type("O", (), {"param_groups": [{"lr": 0.1}]})()]

    cb = F.L2SP(str(anchor), 1.0, [], extra_anchor=stock)
    w = m.encoder.layers[-1].lin.weight
    d0 = (w - stock["encoder.layers.3.lin.weight"]).norm().item()
    cb.on_train_batch_end(Tr(), m, None, None, 0)
    trainable = names(m, lambda p: p.requires_grad)
    assert set(cb.anchor) == trainable  # every trainable tensor has an anchor
    assert (w - stock["encoder.layers.3.lin.weight"]).norm().item() < d0  # pulled toward stock
    cb0 = F.L2SP(str(anchor), 1.0, [])  # default: the anchor file only
    cb0.on_train_batch_end(Tr(), m, None, None, 0)
    assert not [n for n in cb0.anchor if n.startswith("encoder.")]


def test_retrain3_80_takes_retrain3s_arguments():
    import inspect

    import pytest

    MA = pytest.importorskip("modal_app")
    assert inspect.signature(MA.retrain3_80._raw_f_) == inspect.signature(MA.retrain3._raw_f_)
