"""The encoder branch of the main run (R10-brC / R10-brE / R10-brEr): a warm start off the main run's step-1000 weights with
the main run's lr tail, its own sampler seed, a stock L2-SP anchor, and (E) the top 2 encoder blocks on an absolute lr
schedule with per-group gradient clipping, an L2-delta log and an optional real-rows-only encoder gradient. Small fake
models; NeMo is not needed (its scheduler is compared when importable).

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    --with modal python -m pytest test_enc_branch.py -q
"""
import ast
import inspect
import json
import math

import pytest
import torch

import ft_train as F
import test_r4_losses as T
import test_unfreeze_top as U


class FakeTrainer:
    def __init__(self, opt=None, step=0):
        self.optimizers = [opt]
        self.global_step = step


def nemo_cosine(step, peak, floor, warmup, total):
    """NeMo CosineAnnealing, written out from its source (WarmupPolicy.get_lr + _cosine_annealing)."""
    if warmup > 0 and step <= warmup:
        return peak * (step + 1) / (warmup + 1)
    if step > total:
        return floor
    return (peak - floor) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) + floor


# ---- lr schedules -----------------------------------------------------------------------------------------------------
def test_cosine_matches_the_nemo_scheduler_and_hand_values():
    for k in (0, 5, 20, 21, 100, 2500, 4999, 5000, 5001):
        assert F.cosine_lr_at(k, 1e-4, 1e-5, 20, 5000) == pytest.approx(nemo_cosine(k, 1e-4, 1e-5, 20, 5000), rel=1e-12)
    assert F.cosine_lr_at(0, 1e-4, 1e-5, 20, 5000) == pytest.approx(1e-4 / 21)
    assert F.cosine_lr_at(20, 1e-4, 1e-5, 20, 5000) == pytest.approx(1e-4)
    assert F.cosine_lr_at(1000, 1e-4, 1e-5, 20, 5000) == pytest.approx(9.1625e-5, rel=1e-3)  # not near the 1e-5 floor
    assert F.cosine_lr_at(5000, 1e-4, 1e-5, 20, 5000) == pytest.approx(1e-5)
    assert F.cosine_lr_at(6000, 1e-4, 1e-5, 20, 5000) == 1e-5


def test_cosine_equals_nemo_when_importable():
    try:
        from nemo.core.optim.lr_scheduler import CosineAnnealing
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"NeMo not importable here: {e!r}")
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-4)
    sch = CosineAnnealing(opt, max_steps=5000, warmup_steps=20, min_lr=1e-5)
    for k in range(5001):
        assert opt.param_groups[0]["lr"] == pytest.approx(F.cosine_lr_at(k, 1e-4, 1e-5, 20, 5000), rel=1e-9), k
        opt.step()
        sch.step()


def test_decoder_lr_continues_the_main_runs_tail_not_a_full_restart():
    lr, w, total, off = 1e-4, 20, 5000, 1000
    first = F.decoder_lr_at(0, lr, w, total, off)
    assert first == pytest.approx(nemo_cosine(1000, lr, lr / 10, w, total) / 21)  # tail value x the restart ramp
    assert F.decoder_lr_at(w, lr, w, total, off) == pytest.approx(nemo_cosine(1000 + w, lr, lr / 10, w, total))
    for k in (21, 100, 250, 499):
        assert F.decoder_lr_at(k, lr, w, total, off) == pytest.approx(nemo_cosine(off + k, lr, lr / 10, w, total))
    # the restart a plain 500-step run would make: back to the full lr, ending at the floor
    plain_end = nemo_cosine(499, lr, lr / 10, w, 500)
    assert F.decoder_lr_at(499, lr, w, total, off) > 5 * plain_end
    assert F.decoder_lr_at(250, lr, w, total, off) < lr  # never above the main run's own lr at that step
    # offset 0 is the plain schedule (NeMo's own warmup, no second ramp)
    assert F.decoder_lr_at(3, lr, w, total, 0) == pytest.approx(nemo_cosine(3, lr, lr / 10, w, total))


def test_branch_parts_off_by_default_and_group_lrs_at_several_steps():
    # red-on-base: control (every option off = no callback and Lightning's global clip 1.0, as before)
    assert F.branch_trainer_parts(1e-4, 20) == ([], 1.0)
    m, _, _ = U.setup(2)
    groups = F.encoder_param_groups(m, 1e-4, 1.0, peak=2e-5)
    assert [g["lr"] for g in groups] == [1e-4, 2e-5]
    opt = torch.optim.AdamW(groups, lr=1e-4)
    cbs, clip = F.branch_trainer_parts(1e-4, 20, lr_offset=1000, lr_total=5000, enc_lr_peak=2e-5, enc_lr_floor=2e-6,
                                       enc_lr_warmup=50, enc_lr_steps=500, enc_clip=1.0, max_steps=500, has_encoder=True)
    assert clip is None and [type(c).__name__ for c in cbs] == ["LrSchedule", "GroupClip"]
    sched = cbs[0]
    seen = {}
    for k in range(500):
        sched.on_before_optimizer_step(FakeTrainer(opt, k), m, opt)
        seen[k] = [g["lr"] for g in opt.param_groups]
        for g in opt.param_groups:  # NeMo's scheduler steps after the optimizer: its shared floor must not leak back
            g["lr"] = 1e-5
    dec = lambda k: nemo_cosine(1000 + k, 1e-4, 1e-5, 20, 5000) * ((k + 1) / 21 if k <= 20 else 1)  # noqa: E731
    enc = lambda k: nemo_cosine(k, 2e-5, 2e-6, 50, 500)  # noqa: E731
    for k in (0, 1, 19, 20, 21, 49, 50, 51, 100, 250, 400, 499):
        assert seen[k] == [pytest.approx(dec(k)), pytest.approx(enc(k))], k
    assert seen[0][1] == pytest.approx(2e-5 / 51) and seen[50][1] == pytest.approx(2e-5)  # own warmup, own peak
    assert seen[499][1] == pytest.approx(2e-6, rel=2e-3)  # its own floor, not NeMo's shared 1e-5
    assert seen[499][1] / seen[499][0] < 0.03  # the ratio to the decoder keeps falling (NeMo's shared floor ends at 1.0)
    # C and E: the same decoder schedule
    optc = torch.optim.AdamW([{"params": list(m.parameters()), "lr": 1e-4}], lr=1e-4)
    cbc, clipc = F.branch_trainer_parts(1e-4, 20, lr_offset=1000, lr_total=5000, max_steps=500)
    assert clipc == 1.0 and [type(c).__name__ for c in cbc] == ["LrSchedule"]
    for k in (0, 20, 250, 499):
        cbc[0].on_before_optimizer_step(FakeTrainer(optc, k), m, optc)
        assert optc.param_groups[0]["lr"] == seen[k][0]


def test_full_continuation_schedule_shape_over_4000_steps():
    """R10-encF: global steps 1,000 -> 5,000 of the main run's schedule; the encoder has its own 4,000-step schedule."""
    n = 4000
    cbs, clip = F.branch_trainer_parts(1e-4, 20, lr_offset=1000, lr_total=5000, enc_lr_peak=2e-5, enc_lr_floor=2e-6,
                                       enc_lr_warmup=50, enc_lr_steps=n, enc_clip=1.0, max_steps=n, has_encoder=True)
    m, _, _ = U.setup(2)
    opt = torch.optim.AdamW(F.encoder_param_groups(m, 1e-4, 1.0, peak=2e-5), lr=1e-4)
    dec, enc = [], []
    for k in range(n):
        cbs[0].on_before_optimizer_step(FakeTrainer(opt, k), m, opt)
        dec.append(opt.param_groups[0]["lr"])
        enc.append(opt.param_groups[1]["lr"])
        for g in opt.param_groups:
            g["lr"] = 1e-5  # NeMo's shared floor, written back after the step
    # decoder: the main run's tail to its floor at global step 5,000 (k = 3,999 is global 4,999)
    assert dec[20] == pytest.approx(nemo_cosine(1020, 1e-4, 1e-5, 20, 5000))
    assert dec[2000] == pytest.approx(nemo_cosine(3000, 1e-4, 1e-5, 20, 5000))
    assert dec[3999] == pytest.approx(1e-5, rel=1e-3) and dec[0] < dec[20] and max(dec) == dec[20] < 1e-4
    assert all(a >= b for a, b in zip(dec[20:], dec[21:]))  # monotone down the tail
    # encoder: warmup 50, peak 2e-5, half-way at the middle, 2e-6 at the end, never NeMo's 1e-5 floor
    assert enc[0] == pytest.approx(2e-5 / 51) and enc[50] == pytest.approx(2e-5) == max(enc)
    assert enc[2025] == pytest.approx(nemo_cosine(2025, 2e-5, 2e-6, 50, n))
    assert enc[1000] > 1.5e-5 and 0.9e-5 < enc[2025] < 1.2e-5  # a 4,000-step cosine, not the 500-step one
    assert enc[3999] == pytest.approx(2e-6, rel=1e-3) and min(enc[50:]) == enc[3999]
    # a schedule length that does not span the run is refused (R10-brE's 500 steps launched with 4,000)
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, lr_offset=1000, lr_total=5000, enc_lr_peak=2e-5, enc_lr_floor=2e-6,
                               enc_lr_warmup=50, enc_lr_steps=500, max_steps=n, has_encoder=True)


def test_encf_arm_is_brE_over_the_rest_of_the_main_schedule():
    MA = pytest.importorskip("modal_app")
    E, Fa = MA.R3_ARMS["R10-brE"], MA.R3_ARMS["R10-encF"]
    assert {k for k in set(E) | set(Fa) if E.get(k) != Fa.get(k)} == {"enc_lr_steps"}
    assert Fa["enc_lr_steps"] == MA.MAIN_RUN_STEPS - MA.BRANCH_STEP == 4000
    assert Fa["lr_offset"] + Fa["enc_lr_steps"] == Fa["lr_total"]  # ends exactly where the main run's schedule ends


def test_branch_parts_refuse_inconsistent_options():
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, lr_offset=1000, lr_total=1200, max_steps=500)  # runs past the schedule
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, lr_offset=1000, max_steps=500)  # an offset needs the total
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, enc_lr_peak=2e-5, enc_lr_floor=2e-6, max_steps=500, has_encoder=False)
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, enc_clip=1.0, max_steps=500, has_encoder=False)
    with pytest.raises(AssertionError):
        F.branch_trainer_parts(1e-4, 20, enc_lr_warmup=50, max_steps=500, has_encoder=True)  # a knob without its peak


def test_absolute_peak_group_is_its_own_group_and_keeps_the_scale_path():
    m, _, _ = U.setup(2)
    with pytest.raises(AssertionError):
        F.encoder_param_groups(m, 1e-4, 0.2, peak=2e-5)  # two ways to set one lr
    with pytest.raises(AssertionError):
        F.encoder_param_groups(m, 1e-4, 1.0, peak=5e-6)  # below NeMo's min_lr (lr / 10): its constructor would refuse
    old = F.encoder_param_groups(m, 1e-4, 0.2)
    assert [g["lr"] for g in old] == [1e-4, pytest.approx(2e-5)]  # the enc_lr_scale path is as it was


# ---- per-group clipping ------------------------------------------------------------------------------------------------
def grad_case(dec_scale, enc_scale):
    m, batch, _ = U.setup(2, perturb=True)
    m.training_step(batch, 0)["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            p.grad.mul_(enc_scale if n.startswith("encoder.") else dec_scale)
    return m


def group_norm(m, enc):
    return torch.linalg.vector_norm(torch.stack([p.grad.norm() for n, p in m.named_parameters()
                                                 if p.grad is not None and n.startswith("encoder.") == enc])).item()


def test_group_clip_clips_each_group_by_its_own_norm_and_logs_pre_clip(capsys):
    m = grad_case(1.0, 1.0)
    d0, e0 = group_norm(m, False), group_norm(m, True)
    for n, p in m.named_parameters():  # scale the groups to norms 5 (decoder + joint) and 7 (encoder)
        if p.grad is not None:
            p.grad.mul_((5.0 / d0) if not n.startswith("encoder.") else (7.0 / e0))
    cb = F.GroupClip(enc_clip=0.5)
    cb.on_before_optimizer_step(FakeTrainer(step=0), m, None)
    assert group_norm(m, False) == pytest.approx(1.0, rel=1e-5) and group_norm(m, True) == pytest.approx(0.5, rel=1e-5)
    assert cb.last == {"dec": pytest.approx(5.0, rel=1e-5), "enc": pytest.approx(7.0, rel=1e-5)}  # pre-clip norms
    out = capsys.readouterr().out
    assert "grad_norm pre-clip dec 5.000 (clip 1, clipped 1/1) enc 7.000 (clip 0.5, clipped 1/1)" in out


def test_group_clip_leaves_small_groups_alone_and_the_decoder_clip_is_the_controls():
    m = grad_case(0.1, 1.0)
    before = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    d0, e0 = group_norm(m, False), group_norm(m, True)
    cb = F.GroupClip(enc_clip=10 * e0)
    cb.on_before_optimizer_step(FakeTrainer(step=1), m, None)
    for n, p in m.named_parameters():
        if p.grad is not None and n.startswith("encoder."):
            assert torch.equal(p.grad, before[n])  # under its threshold: untouched
    assert d0 < 1.0 and all(torch.equal(p.grad, before[n]) for n, p in m.named_parameters() if p.grad is not None)
    # decoder + joint clipped exactly as the control's global clip does (the control has no encoder gradient)
    mc = grad_case(30.0, 1.0)
    for p in mc.encoder.parameters():
        p.grad = None
    ref = {n: p.grad.clone() for n, p in mc.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_([p for p in mc.parameters() if p.grad is not None], 1.0)  # Lightning's global clip
    want = {n: p.grad.clone() for n, p in mc.named_parameters() if p.grad is not None}
    for n, p in mc.named_parameters():
        if n in ref:
            p.grad = ref[n].clone()
    F.GroupClip(enc_clip=0.5).on_before_optimizer_step(FakeTrainer(step=1), mc, None)
    assert all(torch.equal(p.grad, want[n]) for n, p in mc.named_parameters() if n in want)


# ---- sampler seed ------------------------------------------------------------------------------------------------------
def test_sampler_seed_default_is_the_run_seed_and_the_explicit_one_wins():
    # red-on-base: control (the default is what train() did: seed, or seed + 1000 * step on a resume)
    assert F.resolve_sampler_seed(1) == 1 and F.resolve_sampler_seed(1, 0, True, 250) == 1 + 1000 * 250
    assert F.resolve_sampler_seed(1, 4242) == 4242 != F.resolve_sampler_seed(1)
    assert F.resolve_sampler_seed(1, 4242, True, 250) == 4242 + 1000 * 250  # a resume of the branch itself still reseeds
    assert F.resolve_sampler_seed(1, 4242) == F.resolve_sampler_seed(1, 4242)  # C and E draw the same order


# ---- encoder L2 delta at each save --------------------------------------------------------------------------------------
def test_encoder_l2_delta_is_the_distance_from_stock(tmp_path, capsys):
    m, _, _ = U.setup(2)
    assert F.encoder_l2_delta(m) == 0.0
    n_moved = sum(p.numel() for p in m.encoder.layers[-1].parameters())
    m2, _, _ = U.setup(2, perturb=True)  # the last block moved by +0.3 on every parameter
    assert F.encoder_l2_delta(m2) == pytest.approx(0.3 * math.sqrt(n_moved), rel=1e-5)
    d = F.log_encoder_delta(m2, str(tmp_path), 500)
    assert f"encoder L2 delta from stock at step 500: {d:.4f}" in capsys.readouterr().out
    assert json.loads((tmp_path / "enc_l2_delta.jsonl").read_text()) == {"step": 500, "enc_l2_delta": d}
    m0, _, _ = U.setup(0)  # frozen encoder: nothing logged, nothing written (the control is untouched)
    assert F.encoder_l2_delta(m0) is None and F.log_encoder_delta(m0, str(tmp_path / "none"), 250) is None
    assert not (tmp_path / "none").exists()


def test_save_every_logs_the_encoder_delta_at_each_save(tmp_path, capsys):
    m, _, _ = U.setup(2, perturb=True)
    saved = []
    cb = F.SaveEvery(str(tmp_path), 250, lambda: saved.append(1))
    cb.on_train_batch_end(FakeTrainer(step=250), m, None, None, 0)
    cb.on_train_batch_end(FakeTrainer(step=500), m, None, None, 0)
    lines = [json.loads(l) for l in (tmp_path / "enc_l2_delta.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [250, 500] and (tmp_path / "step500.pt.done").exists() and len(saved) == 2
    assert "encoder L2 delta from stock at step 250" in capsys.readouterr().out
    m0, _, _ = U.setup(0)
    d0 = tmp_path / "ctl"
    d0.mkdir()
    F.SaveEvery(str(d0), 250, lambda: None).on_train_batch_end(FakeTrainer(step=250), m0, None, None, 0)
    assert sorted(p.name for p in d0.iterdir()) == ["step250.pt", "step250.pt.done"]


# ---- encoder real-only --------------------------------------------------------------------------------------------------
def real_only_model(synth_rows, real_only=True):
    m = U.Model()
    batch, ids = T.fake_batch(m, U.TEXTS)
    m.encoder.freeze()
    F.unfreeze_top_layers(m, 2)
    tags = {tuple(ids[0]): "nonen_span", tuple(ids[1]): "nonen", tuple(ids[2]): "span"}
    syn = {tuple(ids[i]) for i in synth_rows}
    F.add_r3_losses(m, tags, span_prefix={tuple(ids[0]): 3}, sub_batch=1, synthetic=syn, enc_real_only=real_only)
    m.train()
    m.encoder.eval()
    return m, batch, ids


def enc_grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if n.startswith("encoder.") and p.grad is not None}


def dec_grad_sum(m):
    return sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if n.startswith(("decoder.", "joint.")) and p.grad is not None)


def test_real_only_masks_the_encoder_gradient_of_the_chosen_rows_only():
    a, _, _ = U.setup(2)
    b, _, _ = U.setup(2)
    feats = lambda m: m.preprocessor(input_signal=T.fake_batch(m, U.TEXTS)[0][0], length=T.fake_batch(m, U.TEXTS)[0][1])  # noqa: E731
    # the hook exists only after unfreeze_top; set the row mask by hand: rows 0 and 2 real, row 1 synthetic
    b._enc_row_keep = torch.tensor([True, False, True])
    fa, la = feats(a)
    fb, lb = feats(b)
    ya, yb = a.encoder(audio_signal=fa, length=la)[0], b.encoder(audio_signal=fb, length=lb)[0]
    assert torch.equal(ya, yb)  # the values are unchanged
    ya[1].sum().backward()
    yb[1].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.encoder.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in b.encoder.parameters())  # row 1: no encoder gradient
    a.zero_grad(set_to_none=True)
    b.zero_grad(set_to_none=True)
    fa, la = feats(a)
    fb, lb = feats(b)
    a.encoder(audio_signal=fa, length=la)[0][0].sum().backward()
    b.encoder(audio_signal=fb, length=lb)[0][0].sum().backward()
    ga, gb = enc_grads(a), enc_grads(b)
    assert ga and all(torch.equal(ga[n], gb[n]) for n in ga)  # a real row: the same gradient as without the mask


def test_real_only_training_step_zeroes_synthetic_rows_and_nothing_else():
    keeps = []
    m, batch, ids = real_only_model({0, 1, 2})  # every row synthetic: no encoder gradient at all, decoder + joint still train
    m.encoder.register_forward_hook(lambda mod, a, k, out: keeps.append(m._enc_row_keep.clone()), with_kwargs=True)
    m.training_step(batch, 0)["loss"].backward()
    assert keeps[0].tolist() == [False, False, False]
    assert all(g.abs().sum() == 0 for g in enc_grads(m).values()) and dec_grad_sum(m) > 0
    # one synthetic row: the encoder gradient is the real rows' share, not the full one
    part, batch, ids = real_only_model({1})
    part.training_step(batch, 0)["loss"].backward()
    full, batch, ids = real_only_model({1}, real_only=False)
    full.training_step(batch, 0)["loss"].backward()
    gp, gf = enc_grads(part), enc_grads(full)
    assert any(gp[n].abs().sum() > 0 for n in gp) and any(not torch.allclose(gp[n], gf[n]) for n in gp)
    assert getattr(full, "_enc_row_keep", None) is None  # off: no mask is ever set


def test_real_only_off_is_the_old_encoder_gradient_bit_for_bit():
    # red-on-base: control (same synthetic set, flag off = the run without the option; the model of test_unfreeze_top)
    off, batch, _ = real_only_model({1}, real_only=False)
    off.training_step(batch, 0)["loss"].backward()
    ref, batch2, _ = U.setup(2)
    ref.training_step(batch2, 0)["loss"].backward()
    goff, gref = enc_grads(off), enc_grads(ref)
    assert goff and all(torch.equal(goff[n], gref[n]) for n in gref)


def test_real_only_needs_unfreeze_top_and_synthetic_labels():
    m = U.Model()
    batch, ids = T.fake_batch(m, U.TEXTS)
    m.encoder.freeze()
    with pytest.raises(AssertionError):
        F.add_r3_losses(m, {tuple(ids[0]): "nonen_span"}, synthetic={tuple(ids[0])}, enc_real_only=True)  # frozen encoder
    F.unfreeze_top_layers(m, 2)
    with pytest.raises(AssertionError):
        F.add_r3_losses(m, {tuple(ids[0]): "nonen_span"}, synthetic=set(), enc_real_only=True)  # nothing to mask


# ---- train() options, arms, launch ---------------------------------------------------------------------------------------
def test_train_takes_the_branch_options_with_off_defaults():
    MA = pytest.importorskip("modal_app")
    params = inspect.signature(F.train).parameters
    for k, v in MA._BRANCH_DEFAULTS.items():
        if k == "anchor_from":
            assert k not in params  # modal_app's: the anchor file is resolved before train()
        else:
            assert params[k].default == v, k


def test_anchor_stays_stock_for_a_warm_start_with_anchor_from():
    MA = pytest.importorskip("modal_app")
    stock = lambda: "STOCK"  # noqa: E731
    r = MA.resolve_l2sp_anchor
    assert r("ultra", True, stock=stock) == "STOCK"  # unchanged paths
    assert r("main@1000", True, stock=stock) == f"{MA.VOL}/exp/main/step1000.pt"  # the old warm-start rule: the init's own
    assert r("main@1000", False, stock=stock) == f"{MA.VOL}/exp/v10pt/step1500.pt"
    assert r("main@1000", True, "ultra", stock=stock) == "STOCK"  # a branch: stock, not the init checkpoint
    assert r("main@1000", False, "ultra", stock=stock) == "STOCK"
    with pytest.raises(AssertionError):
        r("ultra", False, stock=stock)
    with pytest.raises(AssertionError):
        r("main@1000", True, "main@1000", stock=stock)


def test_branch_start_must_be_the_main_runs_step_of_the_arm():
    MA = pytest.importorskip("modal_app")
    cfg = MA.R3_ARMS["R10-brE"]
    ok = lambda p: True  # noqa: E731
    MA.check_branch_start(cfg, "br1", "mainrun@1000", exists=ok)
    for bad in ("ultra", "mainrun@750", "mainrun", "br1@1000", "@1000"):
        with pytest.raises(AssertionError):
            MA.check_branch_start(cfg, "br1", bad, exists=ok)
    with pytest.raises(AssertionError):
        MA.check_branch_start(cfg, "br1", "mainrun@1000", exists=lambda p: False)  # not on the volume
    MA.check_branch_start(MA.R3_ARMS["R10-main"], "main", "ultra", exists=lambda p: False)  # a stock-start arm: no check


def test_branch_arms_differ_from_the_main_arm_and_from_each_other_by_named_knobs():
    MA = pytest.importorskip("modal_app")
    main, C, E, Er = (MA.R3_ARMS[a] for a in ("R10-main", "R10-brC", "R10-brE", "R10-brEr"))

    def diff(a, b):
        return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}

    assert set(diff(main, C)) == {"recipe", "anchor_from", "sampler_seed", "lr_offset", "lr_total"}
    assert C["lr"] == main["lr"] == 1e-4 and C["l2sp_c"] == main["l2sp_c"] == 10.0
    assert C["recipe"] == {k: v for k, v in main["recipe"].items() if k != "init_from"}  # init_from is given at launch
    assert C["anchor_from"] == "ultra" and C["lr_offset"] == MA.BRANCH_STEP == 1000 and C["lr_total"] == MA.MAIN_RUN_STEPS == 5000
    assert set(diff(C, E)) == {"unfreeze_top", "enc_lr_peak", "enc_lr_floor", "enc_lr_warmup", "enc_lr_steps", "enc_clip"}
    assert (E["unfreeze_top"], E["enc_lr_peak"], E["enc_lr_floor"], E["enc_lr_steps"]) == (2, 2e-5, 2e-6, 500)
    assert 20 <= E["enc_lr_warmup"] <= 50 and E["enc_clip"] > 0
    assert set(diff(E, Er)) == {"enc_real_only"} and Er["enc_real_only"] is True
    assert "enc_lr_scale" not in E and "enc_l2sp_c" not in E and not E.get("kl_mask_space")


def test_no_other_arm_sets_a_branch_option_and_defaults_are_off():
    MA = pytest.importorskip("modal_app")
    keys = set(MA._BRANCH_DEFAULTS) | {"unfreeze_top"}
    new = {"R10-brC", "R10-brE", "R10-brEr", "R10-encF", "R10-anc1"}
    assert not [a for a, cfg in MA.R3_ARMS.items() if a not in new and keys & set(cfg)]
    assert all(not v for v in MA._BRANCH_DEFAULTS.values())  # every option off


def test_retrain3_forwards_every_branch_option_and_the_three_signatures_agree():
    MA = pytest.importorskip("modal_app")
    tree = ast.parse(inspect.getsource(MA))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "train" and getattr(n.func.value, "id", "") == "ft_train"
                and any(k.arg == "enc_lr_scale" for k in n.keywords))
    passed = {k.arg for k in call.keywords}
    want = set(MA._BRANCH_DEFAULTS) - {"anchor_from"}
    assert want <= passed and want <= set(inspect.signature(F.train).parameters)
    sig = inspect.signature(MA.retrain3._raw_f_)
    assert inspect.signature(MA.retrain3_80._raw_f_) == sig == inspect.signature(MA.retrain3_80_main._raw_f_)
    src = inspect.getsource(MA.retrain3._raw_f_)
    assert "resolve_l2sp_anchor(" in src and "check_branch_start(" in src and "_BRANCH_DEFAULTS" in src


def test_anchor_test_arm_is_the_main_arm_on_the_main_schedule_and_the_seed_flag_is_forwarded():
    MA = pytest.importorskip("modal_app")
    a, m = MA.R3_ARMS["R10-anc1"], MA.R3_ARMS["R10-main"]
    assert {k: (m.get(k), a.get(k)) for k in set(a) | set(m) if a.get(k) != m.get(k)} == {"lr_total": (None, 5000)}
    assert a["recipe"] == m["recipe"] and a.get("lr_offset", 0) == 0  # a stock start, the plain (offset 0) schedule
    for k in (0, 5, 20, 250, 499):  # the 500-step run's lr is the 5,000-step main's lr at the same step
        assert F.decoder_lr_at(k, a["lr"], 20, a["lr_total"], 0) == pytest.approx(F.cosine_lr_at(k, m["lr"], m["lr"] / 10, 20, 5000))
    assert "sampler_seed" in inspect.signature(MA.retrain3._raw_f_).parameters
    assert 'cfg["sampler_seed"] = sampler_seed or cfg["sampler_seed"]' in inspect.getsource(MA.retrain3._raw_f_)
    assert F.resolve_sampler_seed(1, 777) == 777 and F.resolve_sampler_seed(1, 0) == 1  # an explicit seed beats `seed`
