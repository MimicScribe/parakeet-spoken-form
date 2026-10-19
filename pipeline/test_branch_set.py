"""The matched encoder / decoder branch set (R10-brX0 / X1 / X2 / X2b / X3 / X4 / X5): seven 500-step branches off the main run's
step-1000 weights, each differing from its parent by the keys named in modal_app.py and no others; the summed encoder lr of
each; top-4 = encoder blocks 20-23 of 24; X4 = real rows only; X5 = a 4x weaker L2-SP pull (exempt rows unchanged); the branch start check; the launcher's view of them.
No GPU, no volume.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    --with modal python -m pytest test_branch_set.py -q
"""
import inspect
import os
import re
import subprocess

import pytest
import torch

import ft_train as F
import run_manifest as RM
import test_enc_branch as B
import test_unfreeze_top as U

MA = pytest.importorskip("modal_app")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "launch_run.sh")
ARMS = ("R10-brX0", "R10-brX1", "R10-brX2", "R10-brX2b", "R10-brX3", "R10-brX4", "R10-brX5")
# arm -> (parent arm, the keys it changes, with their values)
SPEC = {
    "R10-brX0": ("R10-brC", {}),
    "R10-brX1": ("R10-brE", {}),
    "R10-brX2": ("R10-brX1", {"enc_lr_peak": 6e-5, "enc_lr_floor": 6e-6}),
    "R10-brX2b": ("R10-brX1", {"enc_lr_peak": 1.5e-5, "enc_lr_floor": 7.5e-6}),
    "R10-brX3": ("R10-brX1", {"unfreeze_top": 4}),
    "R10-brX4": ("R10-brX1", {"enc_real_only": True}),
    "R10-brX5": ("R10-brX0", {"l2sp_c": 2.5}),
}


def diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def enc_lr_series(arm, steps=500):
    cfg = MA.R3_ARMS[arm]
    return [F.cosine_lr_at(k, cfg["enc_lr_peak"], cfg["enc_lr_floor"], cfg["enc_lr_warmup"], cfg["enc_lr_steps"]) for k in range(steps)]


# ---- the arms differ from their parents by exactly the stated keys ----------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_arm_differs_from_its_parent_only_by_the_stated_keys(arm):
    parent, changes = SPEC[arm]
    a, p = MA.R3_ARMS[arm], MA.R3_ARMS[parent]
    d = diff(p, a)
    assert set(d) == set(changes), d
    for k, v in changes.items():
        assert a[k] == v, k


def test_set_shares_the_branch_mechanics_with_the_original_arms():
    C, E = MA.R3_ARMS["R10-brC"], MA.R3_ARMS["R10-brE"]
    assert diff(C, MA.R3_ARMS["R10-brX0"]) == {} and diff(E, MA.R3_ARMS["R10-brX1"]) == {}
    assert diff(MA.R3_ARMS["R10-brEr"], MA.R3_ARMS["R10-brX4"]) == {}  # X4 is the existing real-rows-only arm, on the fixed engine
    for arm in ARMS:
        a = MA.R3_ARMS[arm]
        assert (a["sampler_seed"], a["lr_offset"], a["lr_total"], a["anchor_from"]) == (4242, 1000, 5000, "ultra")
        assert a["lr"] == 1e-4 and a["l2sp_c"] == (2.5 if arm == "R10-brX5" else 10.0) and a["recipe"] == C["recipe"]
        assert not (set(a) & {"kl_mask_space", "enc_lr_scale", "enc_l2sp_c", "lora_rank", "l2sp_exempt_num"})
        assert bool(a.get("enc_real_only")) == (arm == "R10-brX4")
        assert a.get("enc_clip", 0) in (0, 1.0)  # the clip is not a lever: X1's 1.0 wherever there is an encoder
    for arm in ("R10-brX0", "R10-brX5"):  # decoder-only arms: no encoder option at all
        assert not (set(MA.R3_ARMS[arm]) & (set(MA._BRANCH_DEFAULTS) - {"anchor_from", "sampler_seed", "lr_offset", "lr_total"}) | {"unfreeze_top"} & set(MA.R3_ARMS[arm]))
    assert not [a for a in MA.R3_ARMS if a.startswith("R10-brX6")]  # the raised-clip arm is not defined


# ---- encoder lr totals -----------------------------------------------------------------------------------------------
def test_encoder_lr_totals_and_realised_ratios():
    t = {a: sum(enc_lr_series(a)) for a in ("R10-brX1", "R10-brX2", "R10-brX2b")}
    assert t["R10-brX1"] == pytest.approx(5.459e-3, rel=1e-4)
    assert t["R10-brX2"] / t["R10-brX1"] == pytest.approx(3.0, rel=1e-12)  # same shape, 3x
    assert t["R10-brX2b"] / t["R10-brX1"] == pytest.approx(1.0, abs=0.005)  # X1's total (within 0.5%) ...
    first = {a: sum(enc_lr_series(a)[:250]) / t[a] for a in t}  # ... on a flatter schedule
    assert first["R10-brX1"] == pytest.approx(0.728, abs=1e-3)
    assert first["R10-brX2"] == pytest.approx(first["R10-brX1"], abs=1e-9)
    assert first["R10-brX2b"] == pytest.approx(0.580, abs=1e-3)
    print({a: (v, first[a]) for a, v in t.items()})


def test_cumulative_l2sp_pull_per_arm():
    """The encoder pull is p -= lr * c * (p - stock) with the group's own lr, c = the arm's l2sp_c (no enc_l2sp_c): the
    cumulative pull is sum(lr) * c; a displacement decays by exp(-pull) at most. X5 pulls the decoder + joint at a quarter."""
    enc = {a: sum(enc_lr_series(a)) * MA.R3_ARMS[a]["l2sp_c"] for a in ("R10-brX1", "R10-brX2", "R10-brX2b", "R10-brX3", "R10-brX4")}
    assert enc["R10-brX1"] == pytest.approx(0.05459, rel=1e-4) == pytest.approx(enc["R10-brX3"]) == pytest.approx(enc["R10-brX4"])
    assert enc["R10-brX2"] == pytest.approx(0.16377, rel=1e-4) and enc["R10-brX2b"] == pytest.approx(0.05441, rel=1e-3)
    dec = {a: sum(F.decoder_lr_at(k, 1e-4, 20, 5000, 1000) for k in range(500)) * MA.R3_ARMS[a]["l2sp_c"] for a in ("R10-brX0", "R10-brX5")}
    assert dec["R10-brX5"] == pytest.approx(0.25 * dec["R10-brX0"], rel=1e-12) and dec["R10-brX0"] == pytest.approx(0.42, abs=0.02)
    print("pull", enc, dec)


@pytest.mark.parametrize("arm", ("R10-brX1", "R10-brX2", "R10-brX2b", "R10-brX3"))
def test_encoder_schedule_is_accepted_by_the_trainer_parts_and_nemos_floor(arm):
    cfg = MA.R3_ARMS[arm]
    m, _, _ = U.setup(2)
    # NeMo's scheduler is built with the group's starting lr and min_lr lr / 10: the peak may not sit below it
    assert cfg["enc_lr_peak"] >= cfg["lr"] / 10 and 0 < cfg["enc_lr_floor"] < cfg["enc_lr_peak"]
    cbs, clip = F.branch_trainer_parts(cfg["lr"], 20, lr_offset=cfg["lr_offset"], lr_total=cfg["lr_total"],
                                       enc_lr_peak=cfg["enc_lr_peak"], enc_lr_floor=cfg["enc_lr_floor"],
                                       enc_lr_warmup=cfg["enc_lr_warmup"], enc_lr_steps=cfg["enc_lr_steps"],
                                       enc_clip=cfg["enc_clip"], max_steps=500, has_encoder=True)
    assert [type(c).__name__ for c in cbs] == ["LrSchedule", "GroupClip"] and clip is None
    groups = F.encoder_param_groups(m, cfg["lr"], 1.0, peak=cfg["enc_lr_peak"])
    assert [g["lr"] for g in groups] == [cfg["lr"], cfg["enc_lr_peak"]]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    seen = []
    for k in range(500):  # what the optimizer steps actually take (LrSchedule), not NeMo's shared cosine
        cbs[0].on_before_optimizer_step(B.FakeTrainer(opt, k), m, opt)
        seen.append(opt.param_groups[1]["lr"])
        for g in opt.param_groups:
            g["lr"] = 1e-5
    assert seen == enc_lr_series(arm)


# ---- top-4 = encoder blocks 20-23 -------------------------------------------------------------------------------------
def real_depth_model(n_top):
    m = U.Model()
    m.encoder = U.Enc(24)  # the Parakeet encoder has 24 conformer blocks
    m.encoder.freeze()
    stock = F.unfreeze_top_layers(m, n_top)
    return m, stock


@pytest.mark.parametrize("n_top,layers", ((4, range(20, 24)), (2, range(22, 24))))
def test_top_n_unfreezes_exactly_those_blocks_of_24(n_top, layers):
    m, stock = real_depth_model(n_top)
    train = {n for n, p in m.named_parameters() if n.startswith("encoder.") and p.requires_grad}
    want = {n for n, _ in m.named_parameters() if re.match(r"encoder\.layers\.(\d+)\.", n) and int(n.split(".")[2]) in layers}
    assert train == want and len(train) == len(layers) * 4  # 4 tensors in the fake block
    frozen = {n for n, p in m.named_parameters() if n.startswith("encoder.") and not p.requires_grad}
    assert frozen and all(not n.startswith("encoder.layers.") or int(n.split(".")[2]) < min(layers) for n in frozen)
    assert "encoder.pre_encode.weight" in frozen
    assert m._enc_unfrozen == tuple(f"encoder.layers.{i}." for i in layers)
    assert {int(k.split(".")[2]) for k in stock} == set(layers)  # the L2-SP anchor and the saved blocks
    saved = {int(k.split(".")[2]) for k in F.trainable_state(m) if k.startswith("encoder.layers.")}
    assert saved == set(layers)
    with pytest.raises(AssertionError):
        F.unfreeze_top_layers(U.Model(), 5)  # a 4-block fake encoder has no 5th block from the top


def test_x3_is_the_only_arm_with_four_blocks_and_retrain3_applies_it():
    assert MA.R3_ARMS["R10-brX3"]["unfreeze_top"] == 4
    assert [a for a in MA.R3_ARMS if MA.R3_ARMS[a].get("unfreeze_top") == 4] == ["R10-brX3"]
    assert 'arm_top = cfg.pop("unfreeze_top", 0)' in inspect.getsource(MA.retrain3._raw_f_)


# ---- X5: a weaker L2-SP pull on the decoder + joint; the exempt rows are the same as in every arm ------------------------
def pull_case(c, exempt):
    """One L2SP step (lr 0.01) on a stand-in with the real tensor names: weights 1, anchor 0."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)  # the prediction network
            self.joint = torch.nn.Module()
            self.joint.joint_net = torch.nn.Sequential(torch.nn.Linear(2, 6))  # the output layer: 6 rows

    m = M()
    cb = F.L2SP("unused", c, exempt)
    cb.anchor = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for p in m.parameters():
        p.data.fill_(1.0)

    class Opt:
        param_groups = [{"lr": 0.01, "params": list(m.parameters())}]

    class Tr:
        optimizers = [Opt]
        accumulate_grad_batches = 1

    cb.on_train_batch_end(Tr, m, None, None, 0)
    return m


def test_x5_pulls_decoder_and_joint_rows_at_a_quarter_and_leaves_the_exempt_rows_alone():
    exempt = [1, 4]  # stand-in for the written-form rows (digit / symbol pieces)
    full, weak = pull_case(MA.R3_ARMS["R10-brX0"]["l2sp_c"], exempt), pull_case(MA.R3_ARMS["R10-brX5"]["l2sp_c"], exempt)
    for m, keep in ((full, 0.9), (weak, 0.975)):
        w, b = m.joint.joint_net[0].weight.detach(), m.joint.joint_net[0].bias.detach()
        for r in range(6):  # blank / duration / word rows (and every other non-exempt row) are pulled; exempt rows are not
            assert torch.allclose(w[r], torch.full_like(w[r], 1.0 if r in exempt else keep)), r
        assert [round(float(x), 6) for x in b] == [keep if r not in exempt else 1.0 for r in range(6)]
        assert torch.allclose(m.decoder.weight, torch.full_like(m.decoder.weight, keep))  # prediction network: pulled in full
    # the exempt list is built from the written-form pieces alone (no l2sp_exempt_num): the same in X0 and X5
    assert 'exempt = [i for i in wf_piece_ids(m.tokenizer.tokenizer) if i < last_layer]' in inspect.getsource(F.train)
    assert not [a for a in MA.R3_ARMS if MA.R3_ARMS[a].get("l2sp_exempt_num") and a.startswith("R10-brX")]
    assert MA.R3_ARMS["R10-brX5"]["l2sp_c"] == 0.25 * MA.R3_ARMS["R10-brX0"]["l2sp_c"]


def test_x4_real_rows_only_needs_the_top_blocks_and_is_forwarded():
    assert 'enc_real_only=enc_real_only' in inspect.getsource(F.train) and "enc_real_only" in MA._BRANCH_DEFAULTS
    assert MA.R3_ARMS["R10-brX4"]["unfreeze_top"] == 2 and MA.R3_ARMS["R10-brX4"]["enc_real_only"] is True


# ---- branch start checks still enforce init-from main@1000 --------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_branch_start_is_still_main_at_1000_and_stock_anchor(arm):
    cfg = MA.R3_ARMS[arm]
    ok = lambda p: True  # noqa: E731
    MA.check_branch_start(cfg, "x", "wave1_m10_main@1000", exists=ok)
    for bad in ("ultra", "wave1_m10_main@500", "wave1_m10_main@2500", "wave1_m10_main", "x@1000", "@1000"):
        with pytest.raises(AssertionError):
            MA.check_branch_start(cfg, "x", bad, exists=ok)
    with pytest.raises(AssertionError):
        MA.check_branch_start(cfg, "x", "wave1_m10_main@1000", exists=lambda p: False)
    assert cfg["anchor_from"] == "ultra" and MA.resolve_l2sp_anchor("wave1_m10_main@1000", True, "ultra", stock=lambda: "S") == "S"
    assert cfg["lr_offset"] == 1000 == MA.BRANCH_STEP


# ---- the launcher ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_launcher_knows_the_arm_as_a_branch_arm(arm):
    assert RM.has_arm(HERE, arm)
    glob_ok = subprocess.run(["bash", "-c", f'[[ {arm} == R10-br* ]]'], capture_output=True).returncode == 0
    assert glob_ok  # "a branch arm needs --init-from <other run>@<step>" applies


def test_launcher_expects_encoder_speed_for_the_encoder_arms_only():
    line = next(l for l in open(SCRIPT).read().splitlines() if "EXPECT=1.9" in l)
    cond = re.search(r"\[\[ (.*?) \]\] && EXPECT=1.9", line).group(1)
    for arm in ARMS + ("R10-brC", "R10-brE", "R10-encF", "R10-main"):
        hit = subprocess.run(["bash", "-c", f'ARM={arm}; [[ {cond} ]]'], capture_output=True).returncode == 0
        assert hit == (arm in ("R10-brX1", "R10-brX2", "R10-brX2b", "R10-brX3", "R10-brX4", "R10-brE", "R10-brEr", "R10-encF")), arm
