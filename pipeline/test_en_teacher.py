"""English KL options of add_r3_losses (en_teacher: a second frozen teacher on the English rows; en_kl_scale: the English
rows' KL x s) on the tiny fake model of test_r4_losses. NeMo is not needed.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    python -m pytest test_en_teacher.py -q
"""
import re

import pytest
import torch

import ft_train as F
import test_r4_losses as T

ROWS = [("nous avons payé vingt euros hier", "nonen_span"), ("so in the year twenty nineteen we grew", "span"),
        ("wir haben two Hunde", "nonen"), ("the cat sat on the mat today", "anchor")]
EN_TAGS = {"span", "anchor", "anchor_num"}


def b_state(m, bias=0.3, piece=None):
    """A "model B" decoder + joint state: m's own tensors with the joint's output bias moved (all pieces by `bias`
    times a fixed random pattern, or only `piece` by `bias`)."""
    sd = {k: v.clone() for k, v in m.state_dict().items() if k.startswith(("decoder.", "joint."))}
    b = sd["joint.joint_net.1.bias"]
    if piece is None:
        b += bias * torch.randn(b.shape, generator=torch.Generator().manual_seed(7))
    else:
        b[piece] += bias
    return sd


def run(rows=ROWS, perturb_student=0.0, **kw):
    """(model, loss, {tag: {term: (value, n)}} from the r3tag log line) for one training step."""
    m = T.FakeModel()
    batch, ids = T.fake_batch(m, [t for t, _ in rows]) if len(rows) <= 3 else fake_batch4(m, [t for t, _ in rows])
    tags = {tuple(i): tg for i, (_, tg) in zip(ids, rows)}
    if callable(kw.get("en_teacher_state")):
        kw["en_teacher_state"] = kw["en_teacher_state"](m)
    F.add_r3_losses(m, tags, span_prefix={tuple(i): 3 for i, (_, tg) in zip(ids, rows) if tg == "nonen_span"},
                    suffix_anchor=True, sub_batch=1, **kw)
    if perturb_student:  # the student moved away from both teachers (the teachers were copied at add_r3_losses)
        with torch.no_grad():
            m.joint.joint_net[1].bias.add_(perturb_student * torch.randn(8198, generator=torch.Generator().manual_seed(3)))
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = m.training_step(batch, 0)
    per_tag = {}
    for line in buf.getvalue().splitlines():
        if line.startswith("r3tag "):
            tg = line.split()[1]
            per_tag[tg] = {k: (float(v), int(n)) for k, v, n in re.findall(r"(kl_\w+|wf|en) (\S+) \((\d+)\)", line)}
    return m, out["loss"], per_tag


def fake_batch4(m, texts):
    ids = [m.tokenizer.text_to_ids(t) for t in texts]
    tok = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
    for i, x in enumerate(ids):
        tok[i, :len(x)] = torch.tensor(x)
    sig = torch.randn(len(ids), 160 * 12, generator=torch.Generator().manual_seed(1))
    lens = torch.tensor([160 * 12, 160 * 10, 160 * 12, 160 * 11][:len(ids)])
    return (sig, lens, tok, torch.tensor([len(x) for x in ids])), ids


# ---------------------------------------------------------------- routing helper
def test_routed_teacher_logits_per_row():
    torch.manual_seed(0)
    js, je = T.Joint(), T.Joint()
    f, gs, ge = torch.randn(3, 5, 8), torch.randn(3, 4, 8), torch.randn(3, 4, 8)
    rows = torch.tensor([True, False, True])
    out = F.routed_teacher_logits(js, je, f, gs, ge, rows)
    for i, en in enumerate(rows.tolist()):
        want = (je.joint(f[i:i + 1], ge[i:i + 1]) if en else js.joint(f[i:i + 1], gs[i:i + 1])).float()
        assert torch.allclose(out[i:i + 1], want, atol=1e-6)
    # no English row / no English teacher: the stock call itself; all English rows: the English joint only
    assert torch.equal(F.routed_teacher_logits(js, je, f, gs, ge, torch.zeros(3, dtype=torch.bool)), js.joint(f, gs).float())
    assert torch.equal(F.routed_teacher_logits(js, None, f, gs, None, rows), js.joint(f, gs).float())
    assert torch.equal(F.routed_teacher_logits(js, je, f, gs, ge, torch.ones(3, dtype=torch.bool)), je.joint(f, ge).float())


def test_build_en_teacher_loads_and_freezes():
    m = T.FakeModel()
    sd = b_state(m)
    dec, joint = F.build_en_teacher(m, sd)
    assert torch.equal(joint.joint_net[1].bias, sd["joint.joint_net.1.bias"])
    assert not torch.equal(joint.joint_net[1].bias, m.joint.joint_net[1].bias)  # a copy, the student untouched
    assert not dec.training and not joint.training
    assert all(not p.requires_grad for p in list(dec.parameters()) + list(joint.parameters()))
    with pytest.raises(AssertionError):  # an encoder fine-tune cannot share the stock encoder output
        F.build_en_teacher(m, {**sd, "encoder.lin.weight": torch.zeros(8, 4)})
    with pytest.raises(AssertionError):  # a joint tensor left unset
        F.build_en_teacher(m, {k: v for k, v in sd.items() if k != "joint.joint_net.1.bias"})


# ---------------------------------------------------------------- masks
def test_en_full_anchors_english_number_cells():
    # red-on-base: klu_row had no en_full; span / anchor_num number words were always free (stock writes digits)
    spm = T.FakeSPM()
    ids = spm.encode("so in the year twenty nineteen we grew")
    assert not all(F.klu_row(spm, "span", ids, suffix_anchor=True))
    for tag in ("span", "anchor_num"):
        assert all(F.klu_row(spm, tag, ids, suffix_anchor=True, en_full=True))
        assert all(F.klu_row(spm, tag, ids, en_full=True))  # arm a too
    # non-English and free rows are not touched by en_full
    pos = F.klu_row(spm, "nonen_span", ids, span_prefix={tuple(ids): 2}, en_full=True)
    assert pos == F.klu_row(spm, "nonen_span", ids, span_prefix={tuple(ids): 2})
    assert not any(F.klu_row(spm, "free", ids, en_full=True))


# ---------------------------------------------------------------- off = unchanged
def test_defaults_bit_identical():
    # red-on-base: control (en_teacher off, en_kl_scale 1 must leave every term as it was)
    _, l0, t0 = run()
    m1, l1, t1 = run(en_teacher_state=None, en_kl_scale=1.0, en_teacher_num="anchor")
    assert torch.equal(l0, l1) and t0 == t1
    assert "kl_te" not in m1._r3_parts


# ---------------------------------------------------------------- en_teacher routing on a mixed batch
def test_en_teacher_routes_english_rows_only():
    # stock-teacher run vs B-teacher run (mask mode: same positions), student moved away from both
    _, _, base = run(perturb_student=0.5)
    m, loss, ent = run(perturb_student=0.5, en_teacher_state=lambda m: b_state(m, bias=2.0), en_teacher_num="mask")
    assert torch.isfinite(loss)
    for tg in ("nonen", "nonen_span"):  # stock teacher kept: the same values, the same cells
        for k in ("kl_b", "kl_p", "kl_t"):
            assert ent[tg][k] == pytest.approx(base[tg][k], rel=1e-5), (tg, k)
    for tg in ("span", "anchor"):  # B teacher: the KL values move; English token KL now on
        assert ent[tg]["kl_p"][0] != pytest.approx(base[tg]["kl_p"][0], rel=1e-3), tg
        assert base[tg]["kl_t"][1] == 0 and ent[tg]["kl_te"][1] > 0 and ent[tg]["kl_te"][0] > 0, tg
    assert m._r3_parts["kl_te"].item() > 0


def test_en_teacher_equal_to_stock_matches_the_stock_terms():
    # B = an exact copy of stock: blank / punct / duration terms equal the stock run's; English token KL (teacher =
    # student at the start) ~ 0
    _, _, base = run()
    m, _, ent = run(en_teacher_state=lambda m: b_state(m, bias=0.0), en_teacher_num="mask")
    for tg in base:
        for k in ("kl_b", "kl_p", "kl_t"):
            assert ent[tg][k][1] == base[tg][k][1]
            assert ent[tg][k][0] == pytest.approx(base[tg][k][0], abs=1e-6)
    assert m._r3_parts["kl_te"].item() < 1e-6


def test_en_teacher_anchor_mode_opens_number_cells():
    _, _, mask = run(perturb_student=0.5, en_teacher_state=lambda m: b_state(m), en_teacher_num="mask")
    _, _, anchor = run(perturb_student=0.5, en_teacher_state=lambda m: b_state(m), en_teacher_num="anchor")
    assert anchor["span"]["kl_p"][1] > mask["span"]["kl_p"][1]  # number words + guard words now anchored
    assert anchor["span"]["kl_te"][1] > mask["span"]["kl_te"][1]
    assert anchor["anchor"]["kl_p"] == mask["anchor"]["kl_p"]  # a full-anchor row: unchanged
    assert anchor["nonen_span"] == mask["nonen_span"]


def test_en_teacher_digit_cells_dropped():
    # a B that writes the digit piece "5" everywhere: every English cell is a digit cell, so no English KL remains
    def digit_b(m):
        return b_state(m, bias=40.0, piece=m.tokenizer.tokenizer.index["5"])

    _, _, ent = run(perturb_student=0.5, en_teacher_state=digit_b)
    for tg in ("span", "anchor"):
        assert ent[tg]["kl_b"][1] == ent[tg]["kl_p"][1] == ent[tg]["kl_te"][1] == 0, tg
    assert ent["nonen"]["kl_p"][1] > 0  # stock rows unaffected


def test_en_teacher_gradient_and_sub_batch():
    losses = []
    for sb in (1, 2, 4):
        m = T.FakeModel()
        batch, ids = fake_batch4(m, [t for t, _ in ROWS])
        F.add_r3_losses(m, {tuple(i): tg for i, (_, tg) in zip(ids, ROWS)}, span_prefix={tuple(ids[0]): 3},
                        sub_batch=sb, en_teacher_state=b_state(m))
        with torch.no_grad():
            m.joint.joint_net[1].bias.add_(0.5)
        out = m.training_step(batch, 0)
        out["loss"].backward()
        assert m.decoder.lstm.weight_hh_l0.grad.abs().sum() > 0
        e_dec, e_joint = m._kl_en_teacher
        assert all(p.grad is None for p in list(e_dec.parameters()) + list(e_joint.parameters()))
        losses.append(out["loss"].item())
    assert losses[0] == pytest.approx(losses[1], rel=1e-5) == pytest.approx(losses[2], rel=1e-5)


# ---------------------------------------------------------------- en_kl_scale
EN_ONLY = [("so in the year twenty nineteen we grew", "span"), ("the cat sat on the mat today", "anchor")]


def test_en_kl_scale_zero_no_english_kl():
    # red-on-base: add_r3_losses had no en_kl_scale; the English rows were always KL-anchored to stock
    m1, _, _ = run(EN_ONLY, perturb_student=0.5)
    assert m1._r3_parts["kl_b"].item() > 0 and m1._r3_parts["kl_p"].item() > 0
    m0, _, _ = run(EN_ONLY, perturb_student=0.5, en_kl_scale=0.0)
    assert m0._r3_parts["kl_b"].item() == 0 and m0._r3_parts["kl_p"].item() == 0
    mh, _, _ = run(EN_ONLY, perturb_student=0.5, en_kl_scale=0.5)
    for k in ("kl_b", "kl_p"):
        assert mh._r3_parts[k].item() == pytest.approx(0.5 * m1._r3_parts[k].item(), rel=1e-6)


def test_en_kl_scale_zero_keeps_non_english_contribution(monkeypatch):
    seen = []
    orig = F.per_tag_summary
    monkeypatch.setattr(F, "per_tag_summary", lambda tags, acc: seen.append((tags, acc)) or orig(tags, acc))
    m1, _, t1 = run(perturb_student=0.5)
    m0, _, t0 = run(perturb_student=0.5, en_kl_scale=0.0)
    tags, acc = seen[0]  # the scale-1 run's per-row numerators / counts
    ne = torch.tensor([tg not in EN_TAGS for tg in tags])
    for k, part in ((0, "kl_b"), (2, "kl_p"), (12, "kl_d")):  # the same counts, English numerators gone
        want = acc[k][ne].sum() / acc[k + 1].sum().clamp(min=1)
        assert m0._r3_parts[part].item() == pytest.approx(want.item(), rel=1e-5, abs=1e-12), part
        assert bool((seen[1][1][k][~ne] == 0).all()) and torch.equal(seen[1][1][k + 1], acc[k + 1])
    assert acc[2][~ne].sum() > 0 and acc[2][ne].sum() > 0  # both kinds of rows had punct KL to lose / keep
    assert m0._r3_parts["kl_t"].item() == pytest.approx(m1._r3_parts["kl_t"].item(), rel=1e-6)  # non-English only
    for tg in ("nonen", "nonen_span"):
        assert t0[tg] == t1[tg]


def test_en_kl_scale_with_en_teacher():
    m1, _, _ = run(EN_ONLY, perturb_student=0.5, en_teacher_state=lambda m: b_state(m))
    m0, _, _ = run(EN_ONLY, perturb_student=0.5, en_teacher_state=lambda m: b_state(m), en_kl_scale=0.0)
    assert m1._r3_parts["kl_te"].item() > 0 and m0._r3_parts["kl_te"].item() == 0


def test_retrain3_passes_the_english_kl_arms():
    import inspect

    MA = pytest.importorskip("modal_app")
    assert MA.R3_ARMS["R8-s-ent"]["en_teacher"] == "w3soup_a0.5@0"
    assert MA.R3_ARMS["R8-s-enk0"]["en_kl_scale"] == 0.0
    for arm in ("R8-s-ent", "R8-s-enk0"):  # one change against R8-s
        diff = {k for k in set(MA.R3_ARMS[arm]) | set(MA.R3_ARMS["R8-s"])
                if MA.R3_ARMS[arm].get(k) != MA.R3_ARMS["R8-s"].get(k)}
        assert diff in ({"en_teacher"}, {"en_kl_scale"}), (arm, diff)
    src = inspect.getsource(MA.retrain3._raw_f_)
    assert 'en_teacher=cfg["en_teacher"]' in src and 'en_kl_scale=cfg["en_kl_scale"]' in src
    assert {"en_teacher", "en_teacher_num", "en_kl_scale"} <= set(inspect.signature(F.train).parameters)


# ---------------------------------------------------------------- en_free_teacher / en_span_gate (spike 2026-10-18)
FREE_ROWS = ROWS[:3] + [("the code is queue four seven nine", "free")]  # (the 4-row fake batch)


def test_klu_row_free_anchors_only_with_the_flag():
    # red-on-base: klu_row had no en_free; free rows never had a KL position
    spm = T.FakeSPM()
    ids = spm.encode("the code is queue four seven nine")
    assert not any(F.klu_row(spm, "free", ids, en_full=True))
    assert not any(F.klu_row(spm, "free", ids, suffix_anchor=True))
    assert all(F.klu_row(spm, "free", ids, en_full=True, en_free=True))
    assert F.klu_row(spm, "nonen_span", ids, span_prefix={tuple(ids): 2}, en_free=True) == \
        F.klu_row(spm, "nonen_span", ids, span_prefix={tuple(ids): 2})  # no other tag changes


def test_row_gate_span_only_with_the_flag():
    # red-on-base: row_gate had no span_gate; English span rows always took `gate`
    tags = ["span", "anchor", "anchor_num", "free", "nonen", "nonen_span", "nonen_span"]
    keys = [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    base = F.row_gate(tags, keys, 0.9, 0.5, {(7,)})
    assert base == [0.9, 0.5, 0.5, 0.9, 0.9, 0.5, 0.9]
    assert F.row_gate(tags, keys, 0.9, 0.5, {(7,)}, span_gate=0.0) == base  # off = unchanged
    on = F.row_gate(tags, keys, 0.9, 0.5, {(7,)}, span_gate=0.5)
    assert on[0] == 0.5 and on[1:] == base[1:]  # the span rows only
    assert F.row_gate(tags, keys, 0.9, 0.0, set(), span_gate=0.4) == [0.4] + [0.9] * 6  # also without gate_human
    assert F.row_gate(tags, keys, 0.9, 0.0, set()) == [0.9] * 7


def test_en_free_teacher_gives_free_rows_a_b_kl_only_with_the_flag():
    # red-on-base: free rows got no KL at all (klu False, no teacher), whatever en_teacher was
    st = lambda m: b_state(m, bias=2.0)  # noqa: E731
    _, _, off = run(FREE_ROWS, perturb_student=0.5, en_teacher_state=st)
    m, loss, on = run(FREE_ROWS, perturb_student=0.5, en_teacher_state=st, en_free_teacher=True)
    for k in ("kl_b", "kl_p", "kl_t"):  # flag off: nothing on the free row
        assert off["free"][k][1] == 0, k
    assert off["free"].get("kl_te", (0.0, 0))[1] == 0
    assert on["free"]["kl_p"][1] > 0 and on["free"]["kl_p"][0] > 0  # punct KL against B, student moved away
    assert on["free"]["kl_te"][1] > 0 and on["free"]["kl_te"][0] > 0  # the English token KL against B: nonzero
    assert on["free"]["kl_b"][1] > 0
    assert torch.isfinite(loss) and m._r3_parts["kl_te"].item() > 0
    for tg in ("nonen", "nonen_span", "span"):  # every other row: the same values, the same cells
        for k in on[tg]:
            assert on[tg][k] == pytest.approx(off[tg][k], rel=1e-5), (tg, k)
    # the free row reads B, not stock: a B equal to stock gives the stock-teacher values on it
    _, _, same = run(FREE_ROWS, perturb_student=0.5, en_teacher_state=lambda m: b_state(m, bias=0.0), en_free_teacher=True)
    assert same["free"]["kl_p"][0] != pytest.approx(on["free"]["kl_p"][0], rel=1e-3)


def test_en_free_teacher_needs_the_english_teacher():
    with pytest.raises(AssertionError):
        run(FREE_ROWS, en_free_teacher=True)  # no en_teacher
    with pytest.raises(AssertionError):
        run(FREE_ROWS, en_free_teacher=True, en_teacher_state=lambda m: b_state(m), en_teacher_num="mask")


def test_en_free_teacher_follows_en_kl_scale():
    st = lambda m: b_state(m, bias=2.0)  # noqa: E731
    only_free = [("the code is queue four seven nine", "free"), ("wir haben two Hunde", "nonen")]
    m1, _, _ = run(only_free, perturb_student=0.5, en_teacher_state=st, en_free_teacher=True)
    m3, _, _ = run(only_free, perturb_student=0.5, en_teacher_state=st, en_free_teacher=True, en_kl_scale=3.0)
    assert m3._r3_parts["kl_te"].item() == pytest.approx(3.0 * m1._r3_parts["kl_te"].item(), rel=1e-5)


def test_en_span_gate_lowers_the_span_gate_only_with_the_flag():
    # red-on-base: the span rows' blank-KL gate was `gate` (0.9) in add_r3_losses whatever the arm. A B whose blank logit is
    # raised sits at P(non-blank) between 0.5 and 0.9 on these cells: the 0.9 gate drops them all, the 0.5 gate keeps them
    st = lambda m: b_state(m, bias=8.2, piece=8192)  # noqa: E731
    _, _, base = run(perturb_student=0.5, en_teacher_state=st)
    _, _, same = run(perturb_student=0.5, en_teacher_state=st, en_span_gate=0.0)
    assert base == same  # off = unchanged
    assert base["span"]["kl_b"][1] == 0
    _, _, g = run(perturb_student=0.5, en_teacher_state=st, en_span_gate=0.5)
    assert g["span"]["kl_b"][1] > 0 and g["span"]["kl_b"][0] > 0  # the blank KL now reaches the span rows
    for tg in ("nonen", "nonen_span", "anchor"):  # every other row keeps its gate
        assert g[tg] == base[tg], tg


def test_enfree_arm_is_the_en3_arm_plus_the_two_flags():
    import inspect

    MA = pytest.importorskip("modal_app")
    arm, base = MA.R3_ARMS["R8-s-enfree"], MA.R3_ARMS["R8-s-en3"]
    diff = {k: (base.get(k), arm.get(k)) for k in set(arm) | set(base) if arm.get(k) != base.get(k)}
    assert diff == {"en_free_teacher": (None, True), "en_span_gate": (None, 0.5)}, diff
    for name, a in MA.R3_ARMS.items():  # every other arm leaves both off
        if name != "R8-s-enfree":
            assert not a.get("en_free_teacher") and not a.get("en_span_gate"), name
    src = inspect.getsource(MA.retrain3._raw_f_)
    assert 'en_free_teacher=cfg["en_free_teacher"]' in src and 'en_span_gate=cfg["en_span_gate"]' in src
    assert {"en_free_teacher", "en_span_gate"} <= set(inspect.signature(F.train).parameters)
    assert inspect.signature(F.train).parameters["en_free_teacher"].default is False
    assert inspect.signature(F.train).parameters["en_span_gate"].default == 0.0
