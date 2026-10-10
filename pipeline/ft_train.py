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

from common import number_span


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


class NoDropout:
    """Zero every dropout rate in `module` while staying in train mode (cuDNN's LSTM backward
    refuses eval mode)."""

    def __init__(self, module):
        self.module, self.saved = module, []

    def __enter__(self):
        for mod in self.module.modules():
            if isinstance(mod, torch.nn.Dropout) or isinstance(mod, torch.nn.LSTM):
                self.saved.append((mod, "p" if isinstance(mod, torch.nn.Dropout) else "dropout"))
        self.values = [getattr(mod, attr) for mod, attr in self.saved]
        for mod, attr in self.saved:
            setattr(mod, attr, 0.0)

    def __exit__(self, *exc):
        for (mod, attr), v in zip(self.saved, self.values):
            setattr(mod, attr, v)
        self.saved = []


def add_blank_duration_kl(m, weight: float, sub_batch: int = 2, only_texts=None, terms: str = "blank,dur",
                          blank_scale: float = 1.0, punct_weight: float = 0.0, punct_rows: str = "all"):
    """Keep the model's emit-or-wait behaviour close to stock Ultra's.

    Extra loss: KL(teacher || student) over (a) blank vs not-blank and (b) the five TDT durations,
    at every (frame, label) point of the lattice, teacher = a frozen copy of the stock prediction
    network + joint on the same frozen encoder output. Both terms ignore WHICH token is emitted, so
    the student can still learn spoken forms (stock would emit a digit, the student a word: both
    "not blank"). Fine-tuning without it lowered blank confidence and the app's windowed path then
    dropped more words.

    `only_texts`: apply the term only to utterances whose label is in this set (the real-speech
    replay rows), so TTS rows learn spoken form freely. The batch carries no source tag, so rows
    are matched by their token ids.

    `terms`: "dur" anchors only the duration distribution. Measured 2026-10-08: duration-only does
    NOT hold the app's streaming deletions (worse than no anchor) although it keeps emission timing
    identical to stock; the blank term is what holds them."""
    import copy

    teacher_dec, teacher_joint = copy.deepcopy(m.decoder).eval(), copy.deepcopy(m.joint).eval()
    for p in list(teacher_dec.parameters()) + list(teacher_joint.parameters()):
        p.requires_grad_(False)
    m._kl_teacher = (teacher_dec, teacher_joint)  # attribute (not a submodule): not saved
    tset0 = {x.strip() for x in terms.split(",")}
    assert "blank" in tset0 or not tset0 & {"gated", "gnorm", "g2", "tri"}, f"{terms}: gate/tri modes need 'blank'"
    assert len(tset0 & {"gated", "gnorm", "g2"}) <= 1, f"{terms}: gated, gnorm and g2 are mutually exclusive"
    assert len(tset0 & {"dur", "durgate", "durspan"}) <= 1, \
        f"{terms}: dur, durgate and durspan pick one duration mask"
    assert "durspan" not in tset0 or only_texts is not None, \
        "durspan needs kl_replay_only (the TTS rows must stay free)"
    keep = {tuple(m.tokenizer.text_to_ids(t)) for t in only_texts} if only_texts is not None else None
    # Sentence-terminal pieces (". ? !" with or without the word-boundary marker). `punct_weight` adds a binary KL on
    # {terminal punctuation vs any other non-blank token} to stock at every lattice point of EVERY row (TTS included):
    # fine-tunes over-split sentences (E22 sentence-end precision 89.5 -> 84, INV-PUNCT +29 false terminals, 2026-10-08)
    # while number words — "other non-blank" — stay free.
    spm = m.tokenizer.tokenizer
    term_ids = [i for i in range(spm.get_piece_size())
                if spm.id_to_piece(i).lstrip("\u2581") in {".", "?", "!", "...", "\u2026"}]
    term_t = torch.tensor(term_ids)  # moved to the device once, in the first step
    if punct_weight:
        assert term_ids, "no terminal punctuation pieces found"
        print(f"punctuation anchor on {len(term_ids)} pieces: {[spm.id_to_piece(i) for i in term_ids]}, weight {punct_weight}")
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
        # Dropout off for this term: with it on, the student differs from the teacher even at
        # step 0 (KL ~0.35) and the term would chase noise, not drift.
        with NoDropout(m.decoder):
            s_g, _, _ = m.decoder(targets=tokens, target_length=tokens_len)
        f = enc.transpose(1, 2)
        nb = 8193  # 8192 tokens + blank; the last 5 outputs are durations

        def chunk_kl(f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl, lab_sl, span_sl):
            # Dropout zeroed inside the checkpointed function, so the backward recompute matches.
            with NoDropout(m.joint):
                s_logits = m.joint.joint(f_sl, s_sl).float()
            with torch.no_grad():
                t_logits = tjoint.joint(f_sl, t_sl).float()
            T, U = s_logits.shape[1], s_logits.shape[2]
            mask = ((torch.arange(T, device=f_sl.device)[None, :, None] < enc_len_sl[:, None, None])
                    & (torch.arange(U, device=f_sl.device)[None, None, :] <= tok_len_sl[:, None, None]))
            s_lb = s_logits[..., nb - 1] - s_logits[..., :nb].logsumexp(-1)
            t_lb = t_logits[..., nb - 1] - t_logits[..., :nb].logsumexp(-1)
            s_lnb = s_logits[..., :nb - 1].logsumexp(-1) - s_logits[..., :nb].logsumexp(-1)
            t_lnb = t_logits[..., :nb - 1].logsumexp(-1) - t_logits[..., :nb].logsumexp(-1)
            kl_blank = t_lb.exp() * (t_lb - s_lb) + t_lnb.exp() * (t_lnb - s_lnb)
            tset = {x.strip() for x in terms.split(",")}
            if "tri" in tset:
                lab = torch.zeros(s_logits.shape[0], U, dtype=torch.long, device=f_sl.device)
                lab[:, :U - 1] = lab_sl[:, :U - 1]
                idx = lab[:, None, :, None].expand(-1, T, -1, 1)
                s_ll = s_logits[..., :nb].gather(-1, idx).squeeze(-1) - s_logits[..., :nb].logsumexp(-1)
                t_ll = t_logits[..., :nb].gather(-1, idx).squeeze(-1) - t_logits[..., :nb].logsumexp(-1)
                s_lr = torch.log1p(-torch.logsumexp(torch.stack([s_lb, s_ll]), 0).exp().clamp(max=1 - 1e-4))
                t_lr = torch.log1p(-torch.logsumexp(torch.stack([t_lb, t_ll]), 0).exp().clamp(max=1 - 1e-4))
                kl3 = (t_lb.exp() * (t_lb - s_lb) + t_ll.exp() * (t_ll - s_ll) + t_lr.exp() * (t_lr - s_lr))
                valid = (mask & (torch.arange(U, device=f_sl.device)[None, None, :] < tok_len_sl[:, None, None])
                         & (t_ll.exp() > 0.5))
                kl_blank = torch.where(valid, kl3, kl_blank)
            s_dur, t_dur = s_logits[..., nb:].log_softmax(-1), t_logits[..., nb:].log_softmax(-1)
            kl_dur = (t_dur.exp() * (t_dur - s_dur)).sum(-1)
            gate = mask & (t_lnb.exp() > 0.9)  # computed in every mode: the pass rate is logged
            dur_on = "dur" in tset or "durgate" in tset or "durspan" in tset
            if "durgate" in tset:
                dmask = gate
            elif "durspan" in tset:
                dmask = mask & ~span_sl[:, None, :]
            else:
                dmask = mask
            bmask = mask
            if "g2" in tset:
                wait = mask & (t_lb.exp() > 0.95)
                zero = torch.zeros((), device=f_sl.device)
                w_sum = (kl_blank * wait).sum() if "blank" in tset else zero
                d_n = dmask.sum() if dur_on else zero
                return ((kl_blank * gate).sum(), gate.sum(), (kl_dur * dmask).sum() if dur_on else zero,
                        d_n, mask.sum(), gate.sum(), w_sum, wait.sum())
            if "gated" in tset or "gnorm" in tset:
                bmask = gate
            zero = torch.zeros((), device=f_sl.device)
            b_sum = (kl_blank * bmask).sum() if "blank" in tset else zero
            d_sum = (kl_dur * dmask).sum() if dur_on else zero
            b_n = (mask if "gated" in tset else bmask).sum()
            d_n = dmask.sum() if dur_on else zero
            return b_sum, b_n, d_sum, d_n, mask.sum(), gate.sum(), zero, zero

        if punct_weight:
            tid = term_t.to(f.device)

            def chunk_punct(f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl):
                with NoDropout(m.joint):
                    s_logits = m.joint.joint(f_sl, s_sl).float()
                with torch.no_grad():
                    t_logits = tjoint.joint(f_sl, t_sl).float()
                T, U = s_logits.shape[1], s_logits.shape[2]
                mask = ((torch.arange(T, device=f_sl.device)[None, :, None] < enc_len_sl[:, None, None])
                        & (torch.arange(U, device=f_sl.device)[None, None, :] <= tok_len_sl[:, None, None]))
                s_nb, t_nb = s_logits[..., :nb - 1], t_logits[..., :nb - 1]
                s_lt = s_nb[..., tid].logsumexp(-1) - s_nb.logsumexp(-1)
                t_lt = t_nb[..., tid].logsumexp(-1) - t_nb.logsumexp(-1)
                s_lo = torch.log((1 - s_lt.exp()).clamp(min=1e-7))
                t_lo = torch.log((1 - t_lt.exp()).clamp(min=1e-7))
                kl = t_lt.exp() * (t_lt - s_lt) + t_lo.exp() * (t_lo - s_lo)
                return (kl * mask).sum(), mask.sum()

            p_tot = p_cnt = torch.zeros((), device=f.device)
            p_rows = list(range(f.shape[0]))
            if punct_rows == "replay" and keep is not None:
                p_rows = [i for i in p_rows if tuple(tokens[i, :int(tokens_len[i])].tolist()) in keep]
            for j in range(0, len(p_rows), sub_batch):
                sl = torch.tensor(p_rows[j:j + sub_batch], device=f.device)
                u = int(tokens_len[sl].max()) + 1
                t = int(enc_len[sl].max())
                ps, pn = torch.utils.checkpoint.checkpoint(
                    chunk_punct, f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                    enc_len[sl], tokens_len[sl], use_reentrant=False)
                p_tot, p_cnt = p_tot + ps, p_cnt + pn
            kl_p = p_tot / p_cnt.clamp(min=1)
            out["loss"] = out["loss"] + punct_weight * kl_p
            m.log("kl_punct", kl_p.detach(), prog_bar=False)
            if batch_idx % 10 == 0:
                print(f"kl_punct {kl_p.item():.5f}", flush=True)
        rows = list(range(f.shape[0]))
        spans = None
        if keep is not None:
            row_toks = [tokens[i, :int(tokens_len[i])].tolist() for i in rows]
            rows = [i for i, t in zip(rows, row_toks) if tuple(t) in keep]
            if "durspan" in tset0:
                spans = span_mask(m.tokenizer.tokenizer, [(i, row_toks[i]) for i in rows],
                                  tokens.shape[0], int(tokens_len.max()), f.device)
        zero = torch.zeros((), device=f.device)
        b_tot = d_tot = b_cnt = d_cnt = count = passed = w_tot = w_cnt = zero
        for j in range(0, len(rows), sub_batch):
            sl = torch.tensor(rows[j:j + sub_batch], device=f.device)
            u = int(tokens_len[sl].max()) + 1
            t = int(enc_len[sl].max())
            lab_sl = torch.zeros(len(sl), u, dtype=torch.long, device=f.device)
            lab_sl[:, :u - 1] = tokens[sl, :u - 1]
            span_sl = spans[sl, :u] if spans is not None else None
            b_sum, b_n, d_sum, d_n, n, n_pass, w_sum, w_n = torch.utils.checkpoint.checkpoint(
                chunk_kl, f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                enc_len[sl], tokens_len[sl], lab_sl, span_sl, use_reentrant=False)
            b_tot, d_tot = b_tot + b_sum, d_tot + d_sum
            b_cnt, d_cnt, count, passed = b_cnt + b_n, d_cnt + d_n, count + n, passed + n_pass
            w_tot, w_cnt = w_tot + w_sum, w_cnt + w_n
        kl_b = blank_scale * (b_tot / b_cnt.clamp(min=1) + w_tot / w_cnt.clamp(min=1))
        kl_d = d_tot / (d_cnt if "durspan" in tset0 else (passed if "durgate" in tset0 else count)).clamp(min=1)
        kl = kl_b + kl_d
        m.log("kl_blank_dur", kl.detach(), prog_bar=False)
        out["loss"] = out["loss"] + weight * kl
        if batch_idx % 10 == 0:
            print(f"kl_blank_dur {kl.item():.4f} (blank {kl_b.item():.4f} dur {kl_d.item():.4f}) "
                  f"rows {len(rows)}/{f.shape[0]} gate-pass {int(passed)}/{int(count)} wait-pass {int(w_cnt)}"
                  + (f" span-free {int(count - d_cnt)}/{int(count)}" if "durspan" in tset0 else ""),
                  flush=True)
        return out

    m.training_step = training_step


def row_words(spm, ids) -> list[tuple[str, int, int]]:
    """The row's words as (text, first token index, last token index). A piece without the
    word-boundary marker continues the previous word, so a word may span several pieces."""
    out = []
    for j, tid in enumerate(ids):
        piece = spm.id_to_piece(int(tid))
        if piece.startswith("\u2581") or not out:
            out.append([piece.lstrip("\u2581"), j, j])
        else:
            out[-1][0] += piece
            out[-1][2] = j
    return [(w, a, b) for w, a, b in out]


def span_mask(spm, pairs, batch, width, device):
    """(batch, width + 1) bool: True at (i, j) when label position j of row i sits inside a
    spoken number (common.number_span over row_words). `pairs`: (row index, token ids).
    Position len(ids) — the trailing wait — and the padding stay False: those are anchored."""
    span = torch.zeros(batch, width + 1, dtype=torch.bool)
    for i, ids in pairs:
        words = row_words(spm, ids)
        for (w, a, b), flag in zip(words, number_span([w for w, _, _ in words])):
            if flag:
                span[i, a:b + 1] = True
    return span.to(device)


class StepLog(pl.Callback):
    def __init__(self):
        self.t0 = time.time()
        self.clipped = 0

    def on_before_optimizer_step(self, trainer, m, optimizer):
        # Gradient norm before clipping (clip 1.0): a large KL term can scale the whole update down.
        norm = torch.linalg.vector_norm(torch.stack(
            [p.grad.detach().float().norm() for p in m.parameters() if p.grad is not None]))
        self.clipped += int(norm > 1.0)
        if trainer.global_step % 10 == 0:
            print(f"grad_norm {norm.item():.3f} clipped {self.clipped}/{trainer.global_step + 1}", flush=True)

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        if trainer.global_step % 10 == 0:
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
            print(f"step {trainer.global_step} loss {loss:.4f} {time.time() - self.t0:.0f}s", flush=True)


def write_mix(vol: str, out: str, parts: list[tuple[str, float]], seed: int = 1, replay_texts=None,
              kl_exempt: str = "") -> float:
    """Concatenate manifests, taking `hours` from each (all if hours <= 0). Labels of rows from
    `replay/` manifests are added to `replay_texts` when given."""
    r = random.Random(seed)
    rows, exempt = [], set()
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
        if replay_texts is not None and path.startswith("replay/"):
            if kl_exempt and kl_exempt in path:
                exempt.update(x["text"] for x in src)
            else:
                replay_texts.update(x["text"] for x in src)
        rows += src
    if replay_texts is not None and kl_exempt == "numwords":
        # Real speech that says numbers is where spoken form is learned on real audio: no anchor there,
        # whichever manifest the row came from (LSPC also holds number-word rows).
        from common import NUMBER_WORDS
        exempt |= {t for t in replay_texts if NUMBER_WORDS.search(t)}
    if replay_texts is not None and exempt:
        replay_texts.difference_update(exempt)
        print(f"KL-exempt labels ({kl_exempt}): {len(exempt)}, anchored: {len(replay_texts)}")
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
          kl_weight: float = 0.0, kl_replay_only: bool = False, init_from: str = "", kl_terms: str = "blank,dur",
          seed: int = 1, kl_exempt: str = "", kl_blank_scale: float = 1.0, kl_punct: float = 0.0,
          kl_punct_rows: str = "all"):
    """`init_from` = "<run>@<step>": start from that run's checkpoint (the KL teacher stays stock)."""
    import nemo.collections.asr as nemo_asr

    exp = f"{vol}/exp/{run}"
    os.makedirs(exp, exist_ok=True)
    replay_texts = set()
    hours = write_mix(vol, f"{exp}/train_manifest.jsonl", mix, seed=seed, replay_texts=replay_texts,
                      kl_exempt=kl_exempt)
    pl.seed_everything(seed)
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
            "bucket_buffer_size": 20000, "shuffle": True, "shuffle_buffer_size": 10000, "seed": seed,
            "num_workers": 8, "pin_memory": True, "skip_missing_manifest_entries": False})
    m.setup_training_data(m.cfg.train_ds)
    if not spec_augment:
        # With the encoder frozen, SpecAugment masks the encoder's INPUT while the label keeps the
        # masked words, which rewards skipping ahead and guessing (suspected cause of the extra
        # deletions in v3).
        m.spec_augmentation = None
        print("SpecAugment off")
    m.encoder.freeze()
    if kl_weight or kl_punct:
        add_blank_duration_kl(m, kl_weight, only_texts=replay_texts if kl_replay_only else None, terms=kl_terms,
                              blank_scale=kl_blank_scale, punct_weight=kl_punct, punct_rows=kl_punct_rows)
        print(f"KL to stock ({kl_terms}), weight {kl_weight}"
              + (f", replay rows only ({len(replay_texts)} labels)" if kl_replay_only else ""))
    if init_from:
        src, _, step = init_from.partition("@")
        path = f"{vol}/exp/{src}/step{step}.pt" if step else f"{vol}/exp/{src}/decoder_joint.pt"
        missing, unexpected = m.load_state_dict(
            torch.load(path, map_location="cpu"), strict=False)
        assert not unexpected and not [k for k in missing if k.startswith(("decoder.", "joint."))], missing
        print(f"initialized from {init_from} ({path})")
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
