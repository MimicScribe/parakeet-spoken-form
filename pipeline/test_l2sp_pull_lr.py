"""The L2-SP pull must use the lr the optimizer step actually took. Lightning steps the run's (NeMo) scheduler BEFORE
on_train_batch_end, and that scheduler (max_steps = this run's steps) rewrites every group's lr; with LrSchedule
(lr_total / enc_lr_peak) the step itself followed LrSchedule's value, so the pull has to as well (2026-10-19: a 500-step
lr_total spike was pulled on a 500-step cosine, 0.56x the strength of the 5,000-step run it was matched to).

A real Lightning fit on CPU: zero gradients (AdamW then moves nothing), so the only change per step is the pull
p <- p - lr * c * (p - anchor), and the lr each step pulled with is read back from the weights alone.

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    --with modal python -m pytest test_l2sp_pull_lr.py -q
"""
import math

import pytest
import torch

pl = pytest.importorskip("lightning.pytorch")

import ft_train as F

PEAK, WARMUP, C = 1e-4, 20, 10.0


class Tiny(pl.LightningModule):
    """A decoder-like tensor and a joint whose names L2SP knows; the loss has zero gradient everywhere."""

    def __init__(self, sched_steps: int, two_groups: bool = False):
        super().__init__()
        self.decoder = torch.nn.Linear(3, 3)
        self.joint = torch.nn.Module()
        self.joint.joint_net = torch.nn.Sequential(torch.nn.Linear(3, 4))
        self.enc = torch.nn.Linear(2, 2) if two_groups else None
        self.sched_steps = sched_steps
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(1.0)

    def training_step(self, batch, batch_idx):
        return sum((p * 0.0).sum() for p in self.parameters())

    def configure_optimizers(self):
        groups = [{"params": list(self.decoder.parameters()) + list(self.joint.parameters()), "lr": PEAK}]
        if self.enc is not None:
            groups.append({"params": list(self.enc.parameters()), "lr": 2e-5})
        opt = torch.optim.AdamW(groups, lr=PEAK, betas=(0.9, 0.98), weight_decay=0.0)
        # NeMo's CosineAnnealing (warmup 20, floor PEAK / 10, max_steps = this run's): rewrites each group's lr from its
        # base lr every step, like NeMo's _get_lr (cosine_lr_at is checked against NeMo in test_enc_branch.py)
        n, floor = self.sched_steps, PEAK / 10
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: F.cosine_lr_at(k, PEAK, floor, WARMUP, n) / PEAK)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def train_dataloader(self):
        return torch.utils.data.DataLoader([torch.zeros(1)] * 400, batch_size=1)


class Watch(pl.Callback):
    """After each optimizer step's pull: a copy of one decoder weight (anchor 0, so p_{k+1} = p_k * (1 - lr_k * c))."""

    def __init__(self, name="decoder.weight"):
        self.name, self.trace = name, []

    def on_train_batch_end(self, trainer, m, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.trace.append(dict(m.named_parameters())[self.name].detach().clone())


def fit(tmp_path, steps, cbs, two_groups=False, l2sp_cls=F.L2SP, watch="decoder.weight"):
    m = Tiny(steps, two_groups)
    anchor = tmp_path / "anchor.pt"
    torch.save({n: torch.zeros_like(p) for n, p in m.named_parameters()}, anchor)
    pull = l2sp_cls(str(anchor), C, [])
    w = Watch(watch)
    cbs = [pull] + cbs + [w]  # the order the training code uses: L2SP (extra_cb) before LrSchedule (branch_cb)
    F.attach_lr_schedule(cbs)
    tr = pl.Trainer(accelerator="cpu", devices=1, precision="64-true", max_steps=steps, accumulate_grad_batches=2,
                    logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                    num_sanity_val_steps=0, limit_val_batches=0, callbacks=cbs)
    tr.fit(m)
    first = torch.ones(1).double()
    vals = [first] + [t.flatten()[0:1].double() for t in w.trace]
    return [float((1 - vals[k + 1] / vals[k]) / C) for k in range(len(w.trace))], m  # the lr each step pulled with


def test_a_lr_total_run_pulls_with_the_lr_its_steps_took(tmp_path):
    """max_steps 50 on a 500-step schedule (as the 500-step spikes on the 5,000-step one): the pull follows that
    schedule (~PEAK after the warmup), not the run's own 50-step cosine that ends at the floor. Red on base."""
    sched = F.LrSchedule({0: lambda k: F.decoder_lr_at(k, PEAK, WARMUP, 500, 0)})
    got, _ = fit(tmp_path, 50, [sched])
    want = [F.decoder_lr_at(k, PEAK, WARMUP, 500, 0) for k in range(50)]
    assert len(got) == 50
    for k in range(50):
        assert got[k] == pytest.approx(want[k], rel=1e-4), (k, got[k], want[k])
    assert got[45] > 0.9 * PEAK  # the 50-step cosine would be near PEAK / 10 here


def test_a_branch_pulls_each_group_with_its_own_schedule(tmp_path):
    """Branch arms: decoder on the main run's tail (offset), encoder group on its own cosine; both pulls follow them."""
    fns = {0: lambda k: F.decoder_lr_at(k, PEAK, WARMUP, 500, 100),
           1: lambda k: F.cosine_lr_at(k, 2e-5, 2e-6, 10, 40)}
    got_dec, _ = fit(tmp_path, 40, [F.LrSchedule(fns)], two_groups=True)
    got_enc, _ = fit(tmp_path, 40, [F.LrSchedule(fns)], two_groups=True, watch="enc.weight")
    for k in range(40):
        assert got_dec[k] == pytest.approx(fns[0](k), rel=1e-4), k
        assert got_enc[k] == pytest.approx(fns[1](k), rel=1e-3), k


class BaseL2SP(F.L2SP):
    """The pull exactly as it was before the fix (the group lr after the scheduler stepped)."""

    def pull_lrs(self, groups):
        return [g["lr"] for g in groups]


def test_without_lr_schedule_nothing_changes_bit_for_bit(tmp_path):
    """The main run's case (no LrSchedule): the same weights, bit for bit, as the pre-fix pull; and the pull reads the
    scheduler's group lr (the next step's value, as before)."""
    got, m_new = fit(tmp_path, 30, [])
    _, m_old = fit(tmp_path, 30, [], l2sp_cls=BaseL2SP)
    for (n, a), (_, b) in zip(m_new.named_parameters(), m_old.named_parameters()):
        assert torch.equal(a, b), n
    for k in range(30):
        assert got[k] == pytest.approx(F.cosine_lr_at(k + 1, PEAK, PEAK / 10, WARMUP, 30), rel=1e-4), k


def test_attach_finds_the_schedule_and_refuses_two():
    pull = F.L2SP("x", C, [])
    sched = F.LrSchedule({0: lambda k: 1.0})
    assert F.attach_lr_schedule([pull, sched]) and pull.sched is sched
    assert not F.attach_lr_schedule([pull]) and pull.sched is None
    with pytest.raises(AssertionError):
        F.attach_lr_schedule([pull, sched, F.LrSchedule({0: lambda k: 1.0})])
    groups = [{"lr": 0.5}, {"lr": 0.25}]
    assert pull.pull_lrs(groups) == [0.5, 0.25]
    pull.sched = sched
    sched.last = {0: 0.125}
    assert pull.pull_lrs(groups) == [0.125, 0.25]  # a group LrSchedule does not drive keeps its own lr
