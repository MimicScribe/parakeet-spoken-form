"""Gentler top-block encoder training (unfreeze_top + a separate encoder lr / a stronger encoder L2-SP pull / LoRA
adapters), on small fake models. NeMo is not needed (the scheduler check is skipped without it).

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    --with modal python -m pytest test_enc_gentle.py -q
"""
import copy

import pytest
import torch

import ft_train as F
import test_r4_losses as T
import test_unfreeze_top as U


# ---- a NeMo-named conformer block stand-in: self_attn.linear_{q,k,v,out,pos}, feed_forward{1,2}.linear{1,2}, conv ------
class FF(torch.nn.Module):
    def __init__(self, d=8, f=16):
        super().__init__()
        self.linear1 = torch.nn.Linear(d, f)
        self.linear2 = torch.nn.Linear(f, d)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class Attn(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = (torch.nn.Linear(d, d) for _ in range(4))
        self.linear_pos = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):  # one head, no mask: softmax attention over time
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        a = torch.softmax(q @ k.transpose(1, 2) / 8 ** 0.5, -1)
        return self.linear_out(a @ v) + self.linear_pos(x) * 0.1


class Layer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.feed_forward1, self.feed_forward2, self.self_attn = FF(), FF(), Attn()
        self.conv = torch.nn.Conv1d(d, d, 3, padding=1)
        self.norm_out = torch.nn.LayerNorm(d)

    def forward(self, x):  # [B, T, D]
        x = x + 0.5 * self.feed_forward1(x)
        x = x + self.self_attn(x)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.feed_forward2(x)
        return self.norm_out(x)


class NEnc(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.pre_encode = torch.nn.Linear(4, 8)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(n)])

    def forward(self, audio_signal, length):
        x = self.pre_encode(audio_signal.transpose(1, 2))
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2), length

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


class NModel(U.Model):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = NEnc()


TEXTS = U.TEXTS


def lora_setup(rank=8, alpha=16.0, n_top=2):
    m = NModel()
    batch, ids = T.fake_batch(m, TEXTS)
    m.encoder.freeze()
    plain = copy.deepcopy(m)  # the same model without adapters
    stock = F.unfreeze_top_layers(m, n_top, train_base=False)
    torch.manual_seed(5)
    F.add_lora(m, n_top, rank, alpha)
    tags = {tuple(ids[0]): "nonen_span", tuple(ids[1]): "nonen", tuple(ids[2]): "span"}
    F.add_r3_losses(m, tags, span_prefix={tuple(ids[0]): 3}, sub_batch=1)
    m.train()
    m.encoder.eval()
    return m, plain, batch, stock


def move_adapters(m, scale=0.3):
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for _, mod in m._lora["targets"]:
            mod.lora_B.copy_(torch.randn(mod.lora_B.shape, generator=g) * scale)
            mod.lora_A.copy_(torch.randn(mod.lora_A.shape, generator=g) * scale)


def enc_out(m, batch):
    feats, flen = m.preprocessor(input_signal=batch[0], length=batch[1])
    return m.encoder(audio_signal=feats, length=flen)[0]


# ---- S-encA: parameter groups ---------------------------------------------------------------------------------------
def test_param_groups_give_the_encoder_its_own_lr():
    m, _, _ = U.setup(2)
    groups = F.encoder_param_groups(m, 1e-4, 0.2)
    assert [g["lr"] for g in groups] == [1e-4, pytest.approx(2e-5)]
    opt = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.0)
    assert [g["lr"] for g in opt.param_groups] == [1e-4, pytest.approx(2e-5)]
    want_enc = {n for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")}
    assert want_enc and all(n.startswith(("encoder.layers.2.", "encoder.layers.3.")) for n in want_enc)
    by_id = {id(p): n for n, p in m.named_parameters()}
    in_enc = [by_id[id(p)] for p in opt.param_groups[1]["params"]]
    assert set(in_enc) == want_enc and len(in_enc) == len(want_enc)
    rest = [by_id[id(p)] for p in opt.param_groups[0]["params"]]
    assert not set(rest) & want_enc
    assert set(rest) | want_enc == {n for n, _ in m.named_parameters()}  # every tensor held exactly once
    assert len(rest) + len(in_enc) == len(list(m.parameters()))
    # a step moves each group by its own lr (Adam's first step is ~ lr * sign(grad))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()
    d_enc = max((p - before[by_id[id(p)]]).abs().max().item() for p in opt.param_groups[1]["params"])
    d_dec = max((p - before[by_id[id(p)]]).abs().max().item() for p in opt.param_groups[0]["params"] if p.requires_grad)
    assert d_enc == pytest.approx(2e-5, rel=1e-2) and d_dec == pytest.approx(1e-4, rel=1e-2)


def test_param_groups_need_trainable_encoder_tensors():
    m, _, _ = U.setup(0)
    with pytest.raises(AssertionError):
        F.encoder_param_groups(m, 1e-4, 0.2)


def test_param_group_log_counts_trainable_tensors_only(capsys):
    m, _, _ = U.setup(2)
    trainer = Tr(F.encoder_param_groups(m, 1e-4, 0.2))
    F.ParamGroupLog().on_train_start(trainer, m)
    out = capsys.readouterr().out
    n_enc = len([n for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")])
    assert f"param group 1: {n_enc} trainable tensors" in out and f"{n_enc} encoder tensors) peak lr 2e-05" in out
    assert "param group 0:" in out and " 0 encoder tensors) peak lr 0.0001" in out  # the frozen encoder tensors are not counted


def test_param_groups_refuse_a_peak_below_the_cosine_floor():
    m, _, _ = U.setup(2)
    with pytest.raises(AssertionError):
        F.encoder_param_groups(m, 1e-4, 0.05)


def test_nemo_cosine_scheduler_gives_each_group_its_own_peak():
    # NeMo's CosineAnnealing (what setup_optimization builds): peak lr per group, one shared floor (lr / 10)
    try:
        from nemo.core.optim.lr_scheduler import CosineAnnealing
    except Exception as e:  # noqa: BLE001  (NeMo's import chain pulls in many optional packages)
        pytest.skip(f"NeMo not importable here: {e!r}")

    m, _, _ = U.setup(2)
    opt = torch.optim.AdamW(F.encoder_param_groups(m, 1e-4, 0.2), lr=1e-4)
    sch = CosineAnnealing(opt, max_steps=500, warmup_steps=20, min_lr=1e-5)
    lrs = []
    for _ in range(500):
        lrs.append([g["lr"] for g in opt.param_groups])
        opt.step()
        sch.step()
    peak = [max(x[i] for x in lrs) for i in (0, 1)]
    assert peak == [pytest.approx(1e-4, rel=0.01), pytest.approx(2e-5, rel=0.01)]
    assert lrs[-1][0] == pytest.approx(1e-5, rel=0.05) and lrs[-1][1] == pytest.approx(1e-5, rel=0.05)


# ---- S-encB: L2-SP with its own lr per group and its own c for the encoder ------------------------------------------
class Tr:
    accumulate_grad_batches = 1

    def __init__(self, groups):
        self.optimizers = [torch.optim.AdamW(groups, lr=0.1)]


def l2sp_case(tmp_path, groups_fn, c, enc_c):
    m, batch, stock = U.setup(2, perturb=True)
    anchor = tmp_path / "state.pt"
    dec_anchor = {k: v + 0.5 for k, v in m.state_dict().items() if k.startswith(("decoder.", "joint."))
                  and v.is_floating_point()}
    torch.save(dec_anchor, anchor)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    cb = F.L2SP(str(anchor), c, [], extra_anchor=stock, enc_c=enc_c)
    cb.on_train_batch_end(Tr(groups_fn(m)), m, None, None, 0)
    return m, before, dec_anchor, stock


def test_l2sp_uses_each_groups_lr_and_c(tmp_path):
    m, before, dec_anchor, stock = l2sp_case(tmp_path, lambda m: F.encoder_param_groups(m, 0.1, 0.2), 10.0, 100.0)
    n_enc = n_dec = 0
    for n, p in m.named_parameters():
        if n in dec_anchor:  # decoder / joint: lr 0.1, c 10 -> p - 1.0 * (p - a) = the anchor (rate 1)
            want = before[n] - 0.1 * 10.0 * (before[n] - dec_anchor[n])
            n_dec += 1
        elif n in stock:  # encoder: lr 0.02, c 100 -> rate 2.0 (an overshoot past the anchor: p - 2 (p - a))
            want = before[n] - 0.02 * 100.0 * (before[n] - stock[n])
            n_enc += 1
        else:
            continue
        assert torch.allclose(p, want, atol=1e-6), n
    assert n_enc > 0 and n_dec > 0


def test_l2sp_encoder_c_defaults_to_c_and_one_group_is_the_old_rule(tmp_path):
    # one group, enc_c None: every anchored parameter is pulled with p - lr * c * (p - anchor), as before the change
    m, before, dec_anchor, stock = l2sp_case(
        tmp_path, lambda m: [{"params": list(m.parameters()), "lr": 0.1}], 1.0, None)
    for n, p in m.named_parameters():
        a = dec_anchor.get(n, stock.get(n))
        if a is not None:
            assert torch.allclose(p, before[n] - 0.1 * 1.0 * (before[n] - a), atol=1e-7), n


def test_l2sp_encoder_only_pull(tmp_path):
    # c 0 for the decoder + joint, encoder c 50: only the encoder moves
    m, before, dec_anchor, stock = l2sp_case(tmp_path, lambda m: F.encoder_param_groups(m, 0.1, 0.5), 0.0, 50.0)
    for n, p in m.named_parameters():
        if n in dec_anchor:
            assert torch.equal(p, before[n]), n
        elif n in stock:
            assert torch.allclose(p, before[n] - 0.05 * 50.0 * (before[n] - stock[n]), atol=1e-6), n


# ---- S-encC: LoRA ---------------------------------------------------------------------------------------------------
def test_lora_targets_and_trainability():
    m, plain, batch, stock = lora_setup()
    names_ = [n for n, _ in m._lora["targets"]]
    want = []
    for i in (2, 3):
        want += [f"encoder.layers.{i}.{x}" for x in (
            "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
            "feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2")]
    assert sorted(names_) == sorted(want)  # not linear_pos, not the convolution, not the lower blocks
    trainable_enc = {n for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")}
    assert trainable_enc == {f"{t}.lora_{ab}" for t in want for ab in "AB"}  # the base weights stay frozen
    assert m.encoder.layers[2].self_attn.linear_q.lora_A.shape == (8, 8)
    assert m.encoder.layers[2].feed_forward1.linear1.lora_B.shape == (16, 8)
    # B = 0: the student starts exactly at stock
    assert torch.equal(enc_out(m, batch), enc_out(plain, batch))


def test_lora_merge_equals_the_unmerged_forward():
    m, plain, batch, stock = lora_setup()
    move_adapters(m)
    with torch.no_grad():
        y_lora = enc_out(m, batch)
        assert not torch.allclose(y_lora, enc_out(plain, batch), atol=1e-3)  # the adapters do something
        sd = F.trainable_state(m)
        assert not [k for k in sd if "lora" in k]  # never written as adapters
        assert {k for k in sd if k.startswith("encoder.")} == {k for k in m.state_dict() if k.startswith(
            ("encoder.layers.2.", "encoder.layers.3.")) and "lora" not in k}  # the same keys as a full top-2 fine-tune
        reader = copy.deepcopy(plain)  # a plain model, no hooks, no adapters: what every reader builds
        missing, unexpected = reader.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(("encoder.", "preprocessor.")) for k in missing)
        y_merged = enc_out(reader, batch)
    assert (y_lora - y_merged).abs().max().item() < 1e-5
    # untouched tensors of the top blocks are the stock values, the adapted weights are not
    assert torch.equal(sd["encoder.layers.3.conv.weight"], stock["encoder.layers.3.conv.weight"])
    assert torch.equal(sd["encoder.layers.3.self_attn.linear_pos.weight"], stock["encoder.layers.3.self_attn.linear_pos.weight"])
    assert not torch.equal(sd["encoder.layers.3.self_attn.linear_q.weight"], stock["encoder.layers.3.self_attn.linear_q.weight"])
    # the live model's own weights were not rewritten by the merge
    assert torch.equal(m.encoder.layers[3].self_attn.linear_q.weight, stock["encoder.layers.3.self_attn.linear_q.weight"])


def test_lora_teacher_reads_stock_and_the_losses_reach_the_adapters():
    m, plain, batch, stock = lora_setup()
    m.training_step(batch, 0)
    assert m._r3_parts["kl_b"].item() < 1e-6  # at B = 0 student == teacher
    move_adapters(m)
    feats, flen = m.preprocessor(input_signal=batch[0], length=batch[1])
    enc_t, _ = F.stock_top_encoder(m, feats, flen)
    assert torch.allclose(enc_t, enc_out(plain, batch), atol=1e-6)  # the teacher: stock, adapters off
    assert not torch.allclose(enc_t, enc_out(m, batch), atol=1e-3)
    assert m._lora["state"]["on"]  # the switch is back on after the teacher pass
    m.zero_grad()
    m.training_step(batch, 0)["loss"].backward()
    assert m._r3_parts["kl_b"].item() > 1e-6  # the KL sees the drift
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for _, mod in m._lora["targets"]
               for p in (mod.lora_A, mod.lora_B))
    assert all(p.grad is None for n, p in m.named_parameters() if n.startswith("encoder.") and "lora" not in n)


def test_lora_checkpoint_loads_with_the_reader():
    import ft_eval

    m, plain, batch, stock = lora_setup()
    move_adapters(m)
    sd = F.trainable_state(m)
    r = copy.deepcopy(plain)
    base = {k: v.clone() for k, v in r.state_dict().items() if not k.startswith(("encoder.", "preprocessor."))}
    n = ft_eval.load_over_stock(r, base, sd)
    assert n == len([k for k in sd if k.startswith("encoder.")])
    with torch.no_grad():
        assert (enc_out(r, batch) - enc_out(m, batch)).abs().max().item() < 1e-5


def test_defaults_unchanged_without_lora():
    # red-on-base: control (no adapters: trainable_state and the teacher are what they were)
    m, batch, stock = U.setup(2, perturb=True)
    assert getattr(m, "_lora", None) is None
    sd = F.trainable_state(m)
    assert {k for k in sd if k.startswith("encoder.")} == {k for k in m.state_dict() if k.startswith(
        ("encoder.layers.2.", "encoder.layers.3."))}
    with F.lora_disabled(m):
        pass


# ---- arms ----------------------------------------------------------------------------------------------------------
def test_arms_differ_from_r8s_by_one_encoder_knob():
    import inspect

    MA = pytest.importorskip("modal_app")
    base = MA.R3_ARMS["R8-s"]
    one = {"R8-s-encA": ("enc_lr_scale", 0.2), "R8-s-encB": ("enc_l2sp_c", 100.0), "R8-s-encC": ("lora_rank", 8)}
    for arm, (k, v) in one.items():
        cfg = MA.R3_ARMS[arm]
        diff = {x: cfg[x] for x in cfg if cfg.get(x) != base.get(x)}
        assert diff == {k: v} or (arm == "R8-s-encC" and diff == {"lora_rank": 8, "lora_alpha": 16.0}), (arm, diff)
        assert not cfg.get("kl_mask_space")
        assert cfg["recipe"] == base["recipe"]
    params = inspect.signature(F.train).parameters
    for k in ("enc_lr_scale", "enc_l2sp_c", "lora_rank", "lora_alpha"):
        assert k in params
    for arm in ("R8-s", "R8-s-enk0", "R8-s-msk"):  # existing arms: the new knobs stay off
        assert not {"enc_lr_scale", "enc_l2sp_c", "lora_rank"} & set(MA.R3_ARMS[arm])
