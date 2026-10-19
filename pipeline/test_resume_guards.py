"""Resume safety of ft_train (no GPU, no NeMo): the atomic `.step` marker. python -m pytest test_resume_guards.py -q

uv run --no-project --python 3.11 --with torch==2.4.1 --with "numpy<2" --with lightning --with omegaconf --with pytest \
    python -m pytest test_resume_guards.py -q
"""
import os

import ft_train as F


class FakeTrainer:
    def __init__(self, step, fail=False):
        self.global_step, self.fail, self.saved = step, fail, []

    def save_checkpoint(self, path):
        if self.fail:
            raise OSError("disk full")
        self.saved.append(path)
        open(path, "w").write("ckpt")


def run_save(tmp_path, trainer, monkeypatch=None):
    exp = str(tmp_path)
    cb = F.SaveEvery(exp, 250, lambda: None, last=f"{exp}/last.ckpt")
    monkeypatch.setattr(F.torch, "save", lambda obj, path: open(path, "w").write("w"))
    monkeypatch.setattr(F, "trainable_state", lambda m: {})
    cb.on_train_batch_end(trainer, object(), None, None, 0)
    return exp


def test_step_marker_swapped_atomically_before_the_checkpoint(tmp_path, monkeypatch):
    order = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append(os.path.basename(b)), real(a, b))[1])
    exp = run_save(tmp_path, FakeTrainer(500), monkeypatch)
    assert open(f"{exp}/last.ckpt.step").read() == "500"
    assert order == ["last.ckpt.step", "last.ckpt"], order  # the marker goes through os.replace, ahead of the swap
    assert not [n for n in os.listdir(exp) if n.endswith(".tmp")]
    assert os.path.exists(f"{exp}/step500.pt.done")


def test_failed_checkpoint_leaves_the_old_marker(tmp_path, monkeypatch):
    # red-on-base: control (the base also writes the marker only after a good checkpoint)
    open(tmp_path / "last.ckpt.step", "w").write("250")
    open(tmp_path / "last.ckpt", "w").write("old")
    try:
        run_save(tmp_path, FakeTrainer(500, fail=True), monkeypatch)
    except OSError:
        pass
    assert open(tmp_path / "last.ckpt.step").read() == "250"
    assert open(tmp_path / "last.ckpt").read() == "old"


def test_read_resume_step_missing_file_falls_back_to_done_markers(tmp_path):
    exp = str(tmp_path)
    last = f"{exp}/last.ckpt"
    open(f"{exp}/step250.pt.done", "w").close()
    open(f"{exp}/step750.pt.done", "w").close()
    open(f"{exp}/step500.pt.done", "w").close()
    assert F.read_resume_step(last, exp) == 750  # no .step at all
    open(last + ".step", "w").write("")  # a truncated marker
    assert F.read_resume_step(last, exp) == 750
    open(last + ".step", "w").write("500")
    assert F.read_resume_step(last, exp) == 500  # a good marker wins
    assert F.read_resume_step(f"{exp}/nope/last.ckpt", f"{exp}/nope") == 0
