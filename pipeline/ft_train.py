"""Fine-tune Parakeet Ultra's prediction network and joint with the encoder frozen.

Saves only the trainable weights (decoder + joint, ~18M params) to /vol/exp/<run>/decoder_joint.pt.
"""

import json
import os
import random
import time

import lightning.pytorch as pl
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf, open_dict


class KeepEncoderFrozen(pl.Callback):
    """Lightning calls model.train() at epoch start; keep the frozen encoder's BatchNorm and
    dropout in eval mode."""

    def on_train_epoch_start(self, trainer, m):
        m.encoder.eval()

    def on_train_batch_start(self, trainer, m, batch, idx):
        m.encoder.eval()


class SaveEvery(pl.Callback):
    """Keep the trainable weights every `every` steps (no full checkpoints: 18M params, ~70 MB)."""

    def __init__(self, exp: str, every: int, on_save):
        self.exp, self.every, self.on_save = exp, every, on_save

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        step = trainer.global_step
        if step and step % self.every == 0:
            torch.save(trainable_state(m), f"{self.exp}/step{step}.pt")
            self.on_save()


class HoldRows(pl.Callback):
    """Hold the joint's output rows `rows` at their original values: gradients are zeroed and the
    rows are copied back after every step (AdamW's decoupled weight decay would still move them).
    Rows 8192-8197 are blank + the five TDT durations: keeping them anchors when the decoder emits
    nothing and how far it skips, which the app's blank-gap and duration heuristics are tuned on.
    The shared hidden layers still train, so this limits the drift rather than removing it."""

    def __init__(self, m, rows: range):
        self.lin = m.joint.joint_net[-1]
        self.idx = torch.tensor(list(rows))
        self.w = self.lin.weight.detach()[self.idx].clone()
        self.b = self.lin.bias.detach()[self.idx].clone()

        def hook(g):
            g = g.clone()
            g[self.idx.to(g.device)] = 0
            return g

        self.lin.weight.register_hook(hook)
        self.lin.bias.register_hook(hook)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        dev = self.lin.weight.device
        self.lin.weight[self.idx.to(dev)] = self.w.to(dev, self.lin.weight.dtype)
        self.lin.bias[self.idx.to(dev)] = self.b.to(dev, self.lin.bias.dtype)


def add_blank_duration_kl(m, weight: float, sub_batch: int = 2):
    """Keep the model's emit-or-wait behaviour close to stock Ultra's.

    Extra loss: KL(teacher || student) over (a) blank vs not-blank and (b) the five TDT durations,
    at every (frame, label) point of the lattice, teacher = a frozen copy of the stock prediction
    network + joint on the same frozen encoder output. Both terms ignore WHICH token is emitted, so
    the student can still learn spoken forms (stock would emit a digit, the student a word: both
    "not blank"). Fine-tuning without it lowered blank confidence and the app's windowed path then
    dropped more words."""
    import copy

    teacher_dec, teacher_joint = copy.deepcopy(m.decoder).eval(), copy.deepcopy(m.joint).eval()
    for p in list(teacher_dec.parameters()) + list(teacher_joint.parameters()):
        p.requires_grad_(False)
    m._kl_teacher = (teacher_dec, teacher_joint)  # attribute (not a submodule): not saved
    orig_step = m.training_step

    def training_step(batch, batch_idx):
        out = orig_step(batch, batch_idx)
        signal, signal_len, tokens, tokens_len = batch[:4]
        tdec, tjoint = m._kl_teacher
        tdec.to(signal.device), tjoint.to(signal.device)
        with torch.no_grad():
            feats, feats_len = m.preprocessor(input_signal=signal, length=signal_len)  # no SpecAugment
            enc, enc_len = m.encoder(audio_signal=feats, length=feats_len)
            t_g, _, _ = tdec(targets=tokens, target_length=tokens_len)
        # Dropout off for this term (gradients still flow): with it on, the student differs from the
        # teacher even at step 0 (KL ~0.35) and the term would chase noise, not drift.
        dec_was = m.decoder.training
        m.decoder.eval()
        s_g, _, _ = m.decoder(targets=tokens, target_length=tokens_len)
        m.decoder.train(dec_was)
        f = enc.transpose(1, 2)
        nb = 8193  # 8192 tokens + blank; the last 5 outputs are durations

        def chunk_kl(f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl):
            # eval() inside the checkpointed function, so the backward recompute also runs it
            # without dropout.
            joint_was = m.joint.training
            m.joint.eval()
            s_logits = m.joint.joint(f_sl, s_sl).float()
            m.joint.train(joint_was)
            with torch.no_grad():
                t_logits = tjoint.joint(f_sl, t_sl).float()
            T, U = s_logits.shape[1], s_logits.shape[2]
            mask = ((torch.arange(T, device=f_sl.device)[None, :, None] < enc_len_sl[:, None, None])
                    & (torch.arange(U, device=f_sl.device)[None, None, :] <= tok_len_sl[:, None, None]))
            s_lb = s_logits[..., nb - 1] - s_logits[..., :nb].logsumexp(-1)
            t_lb = t_logits[..., nb - 1] - t_logits[..., :nb].logsumexp(-1)
            s_lnb = torch.log1p(-s_lb.exp().clamp(max=1 - 1e-6))
            t_lnb = torch.log1p(-t_lb.exp().clamp(max=1 - 1e-6))
            kl_blank = t_lb.exp() * (t_lb - s_lb) + t_lnb.exp() * (t_lnb - s_lnb)
            s_dur, t_dur = s_logits[..., nb:].log_softmax(-1), t_logits[..., nb:].log_softmax(-1)
            kl_dur = (t_dur.exp() * (t_dur - s_dur)).sum(-1)
            return ((kl_blank + kl_dur) * mask).sum(), mask.sum()

        total, count = 0.0, 0
        for i in range(0, f.shape[0], sub_batch):
            sl = slice(i, i + sub_batch)
            u = int(tokens_len[sl].max()) + 1
            t = int(enc_len[sl].max())
            # Checkpointed: the [b, T, U, 8198] logits are rebuilt during backward instead of
            # being held for every sub-batch at once (that ran out of memory at batch 600 s).
            part, n = torch.utils.checkpoint.checkpoint(
                chunk_kl, f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                enc_len[sl], tokens_len[sl], use_reentrant=False)
            total = total + part
            count += int(n)
        kl = total / max(count, 1)
        m.log("kl_blank_dur", kl.detach(), prog_bar=False)
        out["loss"] = out["loss"] + weight * kl
        if batch_idx % 10 == 0:
            print(f"kl_blank_dur {kl.item():.4f}", flush=True)
        return out

    m.training_step = training_step


class StepLog(pl.Callback):
    def __init__(self):
        self.t0 = time.time()

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        if trainer.global_step % 10 == 0:
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
            print(f"step {trainer.global_step} loss {loss:.4f} {time.time() - self.t0:.0f}s", flush=True)


def write_mix(vol: str, out: str, parts: list[tuple[str, float]], seed: int = 1) -> float:
    """Concatenate manifests, taking `hours` from each (all if hours <= 0)."""
    r = random.Random(seed)
    rows = []
    for path, hours in parts:
        src = [json.loads(l) for l in open(f"{vol}/{path}")]
        r.shuffle(src)
        if hours > 0:
            acc, keep = 0.0, []
            for x in src:
                if acc >= hours * 3600:
                    break
                keep.append(x)
                acc += x["duration"]
            src = keep
        print(f"mix {path}: {len(src)} rows, {sum(x['duration'] for x in src) / 3600:.2f} h")
        rows += src
    r.shuffle(rows)
    with open(out, "w") as f:
        for x in rows:
            f.write(json.dumps({k: x[k] for k in ("audio_filepath", "duration", "text")}) + "\n")
    return sum(x["duration"] for x in rows) / 3600


def trainable_state(m) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()
            if not (k.startswith("encoder.") or k.startswith("preprocessor."))}


def train(vol: str, ultra: str, run: str, mix: list[tuple[str, float]], max_steps: int,
          lr: float = 1e-4, batch_duration: float = 600, warmup: int = 20, freeze_blank_duration: bool = False,
          save_every: int = 0, on_save=lambda: None, weight_decay: float = 1e-3, spec_augment: bool = True,
          kl_weight: float = 0.0):
    import nemo.collections.asr as nemo_asr

    exp = f"{vol}/exp/{run}"
    os.makedirs(exp, exist_ok=True)
    hours = write_mix(vol, f"{exp}/train_manifest.jsonl", mix)
    print(f"train mix: {hours:.2f} h")

    m = nemo_asr.models.ASRModel.restore_from(ultra, map_location="cpu")
    # A fresh train_ds: the checkpoint's own carries its original corpus settings (tarred inputs,
    # bucket bins, a null max_tps) that do not apply here.
    with open_dict(m.cfg):
        print("checkpoint train_ds keys:", sorted(m.cfg.train_ds.keys()))
        m.cfg.train_ds = OmegaConf.create({
            "manifest_filepath": f"{exp}/train_manifest.jsonl", "sample_rate": 16000,
            "use_lhotse": True, "text_field": "text", "batch_duration": batch_duration,
            "max_duration": 20.0, "min_duration": 0.5, "use_bucketing": True, "num_buckets": 10,
            "bucket_buffer_size": 20000, "shuffle": True, "shuffle_buffer_size": 10000, "seed": 1,
            "num_workers": 8, "pin_memory": True, "skip_missing_manifest_entries": False})
    m.setup_training_data(m.cfg.train_ds)
    if not spec_augment:
        # With the encoder frozen, SpecAugment masks the encoder's INPUT while the label keeps the
        # masked words, which rewards skipping ahead and guessing (suspected cause of the extra
        # deletions in v3).
        m.spec_augmentation = None
        print("SpecAugment off")
    m.encoder.freeze()
    if kl_weight:
        add_blank_duration_kl(m, kl_weight)
        print(f"blank/duration KL to stock, weight {kl_weight}")
    extra = []
    if freeze_blank_duration:
        n = m.joint.joint_net[-1].weight.shape[0]
        extra.append(HoldRows(m, range(n - 6, n)))
        print(f"holding joint rows {n - 6}..{n - 1} (blank + durations)")
    groups = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            k = ".".join(n.split(".")[:2])
            groups[k] = groups.get(k, 0) + p.numel()
    print(f"trainable params: {sum(groups.values()):,} {groups}")

    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=max_steps,
                         limit_train_batches=max_steps, limit_val_batches=0, num_sanity_val_steps=0,
                         use_distributed_sampler=False, gradient_clip_val=1.0, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         callbacks=[KeepEncoderFrozen(), StepLog()] + extra
                         + ([SaveEvery(exp, save_every, on_save)] if save_every else []))
    m.set_trainer(trainer)
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": lr, "betas": [0.9, 0.98], "weight_decay": weight_decay,
        "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": lr / 10,
                  "max_steps": max_steps}}))
    t0 = time.time()
    trainer.fit(m)
    print(f"trained {max_steps} steps in {time.time() - t0:.0f}s")
    torch.save(trainable_state(m), f"{exp}/decoder_joint.pt")
    return m
