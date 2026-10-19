"""Fine-tune Parakeet Ultra's prediction network and joint with the encoder frozen (optionally its top conformer blocks
trainable too: train(unfreeze_top=N)).

Saves only the trainable weights (decoder + joint, ~18M params, + the unfrozen encoder blocks) to
/vol/exp/<run>/decoder_joint.pt.
"""

import collections
import json
import math
import re
import os
import random
import time

import lightning.pytorch as pl
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf, open_dict

from common import number_span
import exposure


class KeepEncoderFrozen(pl.Callback):
    """Lightning calls model.train() at epoch start; keep the frozen encoder's BatchNorm and
    dropout in eval mode. With unfreeze_top the trainable top blocks stay in eval mode too (unfreeze_top_layers)."""

    def on_train_epoch_start(self, trainer, m):
        m.encoder.eval()

    def on_train_batch_start(self, trainer, m, batch, idx):
        m.encoder.eval()


def write_atomic(path: str, text: str) -> None:
    """Write `text` to `path` through a temp file and os.replace: a reader sees the old or the new content, never a
    truncated file (a crash between open and write used to leave an empty `.step`)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def read_resume_step(last: str, exp: str) -> int:
    """The step a resume continues from: `<last>.step`, written by SaveEvery. A missing, empty or unreadable file falls
    back to the highest `step<N>.pt.done` in `exp` (SaveEvery writes both at the same boundary), then to 0, so a run
    never fails to resume (or resumes with a nonsense sampler seed) over a lost marker."""
    try:
        return int(open(last + ".step").read().strip())
    except (OSError, ValueError):
        pass
    done = [int(m.group(1)) for n in (os.listdir(exp) if os.path.isdir(exp) else [])
            if (m := re.fullmatch(r"step(\d+)\.pt\.done", n))]
    step = max(done, default=0)
    print(f"resume: {last}.step missing or unreadable -> step {step} from the .done markers", flush=True)
    return step


def manifest_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_mix_drift(exp: str, manifest: str, resuming: bool, strict: bool = True) -> str:
    """Mix-drift guard. Every attempt rebuilds the training manifest from the live volume. The first start of a run
    stores its sha256 in `<exp>/train_manifest.sha256`; a resume (strict: same seed, so the same mix and order) must
    write the identical manifest, otherwise a part changed under the run and the resumed steps would train on another
    mix than the first ones. A fresh start (no checkpoint to resume) replaces the stored hash. A resume of a run
    that predates the marker records the current hash and says so. strict=False (a resume that reseeds the shuffle on
    purpose) only records."""
    marker = f"{exp}/train_manifest.sha256"
    now = manifest_sha256(manifest)
    first = open(marker).read().strip() if resuming and os.path.exists(marker) else ""
    if first and strict:
        assert first == now, (f"mix drift: the manifest written for this resume ({now[:12]}) differs from the run's first "
                              f"({first[:12]}, {marker}); a mix part changed on the volume since the run started. "
                              f"Restore it, or start a new run name")
        return now
    if resuming and not first:
        print(f"resume: no stored manifest hash for this run; recording {now[:12]}", flush=True)
    if not (resuming and first):  # keep the first hash of a non-strict resume
        write_atomic(marker, now)
    print(f"manifest sha256 {now[:12]} recorded", flush=True)
    return now


class SaveEvery(pl.Callback):
    """Keep the trainable weights every `every` steps (no full checkpoints: 18M params, ~70 MB)."""

    def __init__(self, exp: str, every: int, on_save, last: str = ""):
        self.exp, self.every, self.on_save, self.last = exp, every, on_save, last

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        step = trainer.global_step
        if step and step % self.every == 0 and not os.path.exists(f"{self.exp}/step{step}.pt.done"):
            torch.save(trainable_state(m), f"{self.exp}/step{step}.pt")
            if self.last:  # full state (optimizer, scheduler, step) for a resume after preemption
                trainer.save_checkpoint(self.last + ".tmp")
                write_atomic(self.last + ".step", str(step))  # before the checkpoint swap: never behind it
                os.replace(self.last + ".tmp", self.last)
            log_encoder_delta(m, self.exp, step)  # nothing without trainable encoder blocks
            open(f"{self.exp}/step{step}.pt.done", "w").close()
            self.on_save()


class HoldRows(pl.Callback):
    """Hold the joint's output rows `rows` at their original values: gradients are zeroed and the
    rows are copied back after every step (AdamW's decoupled weight decay would still move them).
    Rows 8192-8197 are blank + the five TDT durations: keeping them anchors when the decoder emits
    nothing and how far it skips, which downstream blank and duration handling relies on.
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
                          blank_scale: float = 1.0, punct_weight: float = 0.0, punct_rows: str = "all",
                          span_texts=None, nonen_texts=None, tok_weight: float = 0.0):
    """Keep the model's emit-or-wait behaviour close to stock Ultra's.

    Extra loss: KL(teacher || student) over (a) blank vs not-blank and (b) the five TDT durations,
    at every (frame, label) point of the lattice, teacher = a frozen copy of the stock prediction
    network + joint on the same frozen encoder output. Both terms ignore WHICH token is emitted, so
    the student can still learn spoken forms (stock would emit a digit, the student a word: both
    "not blank"). Fine-tuning without it lowered blank confidence and windowed streaming decode then
    dropped more words.

    `only_texts`: apply the term only to utterances whose label is in this set (the real-speech
    replay rows), so TTS rows learn spoken form freely. The batch carries no source tag, so rows
    are matched by their token ids.

    `terms`: "dur" anchors only the duration distribution. Measured 2026-10-08: duration-only does
    NOT hold the streaming-decode deletions (worse than no anchor) although it keeps emission timing
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
    # Real-speech retrain (2026-10-12): `span_texts` = English real number rows: anchored like replay rows except INSIDE
    # the number span when terms has "blankspan" (the labels change there by design). `nonen_texts` = non-English rows:
    # anchored like replay rows, plus (tok_weight) a full-distribution token KL to stock, the anti-leak term (stock
    # writes digits there; the student must not drift to English number words).
    ids = lambda ts: {tuple(m.tokenizer.text_to_ids(t)) for t in ts}  # noqa: E731
    span_keep = ids(span_texts) if span_texts else set()
    nonen_keep = ids(nonen_texts) if nonen_texts else set()
    keep = ids(only_texts) | span_keep | nonen_keep if only_texts is not None else None
    assert "blankspan" not in tset0 or span_keep, "blankspan needs span_texts"
    if keep is not None:
        print(f"KL rows: anchor {len(only_texts)} span {len(span_keep)} non-en {len(nonen_keep)} labels")
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
            if "blankspan" in tset and span_sl is not None:
                bmask = bmask & ~span_sl[:, None, :]
            zero = torch.zeros((), device=f_sl.device)
            b_sum = (kl_blank * bmask).sum() if "blank" in tset else zero
            d_sum = (kl_dur * dmask).sum() if dur_on else zero
            b_n = (mask if "gated" in tset else bmask).sum()
            if "blankspan" in tset and span_sl is not None and "gated" in tset:
                b_n = (mask & ~span_sl[:, None, :]).sum()
            d_n = dmask.sum() if dur_on else zero
            return b_sum, b_n, d_sum, d_n, mask.sum(), gate.sum(), zero, zero

        merged = (MERGED_KL and keep is not None and punct_rows == "replay"
                  and not tset0 & {"g2", "tri", "durspan", "durgate", "dur"})
        if merged:
            # ONE joint evaluation per sub-batch for the blank/gnorm(+blankspan), punctuation and token terms (they
            # used three, each recomputed in backward: the dress rehearsal ran ~2x slower than v12y). Same rows (keep;
            # punct rows == keep with punct_rows="replay"), same masks and normalisations as the separate passes.
            tid = term_t.to(f.device)
            row_toks = [tokens[i, :int(tokens_len[i])].tolist() for i in range(f.shape[0])]
            rows = [i for i, t in enumerate(row_toks) if tuple(t) in keep]
            is_nonen = torch.tensor([tuple(t) in nonen_keep for t in row_toks], device=f.device)
            spans = None
            if "blankspan" in tset0:
                spans = span_mask(m.tokenizer.tokenizer, [(i, row_toks[i]) for i in rows if tuple(row_toks[i]) in span_keep],
                                  tokens.shape[0], int(tokens_len.max()), f.device)

            def chunk_all(f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl, span_sl, nonen_sl):
                with NoDropout(m.joint):
                    s_logits = m.joint.joint(f_sl, s_sl).float()
                with torch.no_grad():
                    t_logits = tjoint.joint(f_sl, t_sl).float()
                T, U = s_logits.shape[1], s_logits.shape[2]
                mask = ((torch.arange(T, device=f_sl.device)[None, :, None] < enc_len_sl[:, None, None])
                        & (torch.arange(U, device=f_sl.device)[None, None, :] <= tok_len_sl[:, None, None]))
                s_all = s_logits[..., :nb].logsumexp(-1)
                t_all = t_logits[..., :nb].logsumexp(-1)
                s_lb = s_logits[..., nb - 1] - s_all
                t_lb = t_logits[..., nb - 1] - t_all
                s_nbz = s_logits[..., :nb - 1].logsumexp(-1)
                t_nbz = t_logits[..., :nb - 1].logsumexp(-1)
                s_lnb, t_lnb = s_nbz - s_all, t_nbz - t_all
                kl_blank = t_lb.exp() * (t_lb - s_lb) + t_lnb.exp() * (t_lnb - s_lnb)
                gate = mask & (t_lnb.exp() > 0.9)
                bmask = gate if ("gated" in tset0 or "gnorm" in tset0) else mask
                if span_sl is not None:
                    bmask = bmask & ~span_sl[:, None, :]
                zero = torch.zeros((), device=f_sl.device)
                b_sum = (kl_blank * bmask).sum() if "blank" in tset0 else zero
                b_n = (mask if "gated" in tset0 else bmask).sum()
                if span_sl is not None and "gated" in tset0:
                    b_n = (mask & ~span_sl[:, None, :]).sum()
                # punctuation: {terminal vs other non-blank}
                s_lt = s_logits[..., tid].logsumexp(-1) - s_nbz
                t_lt = t_logits[..., tid].logsumexp(-1) - t_nbz
                s_lo = torch.log((1 - s_lt.exp()).clamp(min=1e-7))
                t_lo = torch.log((1 - t_lt.exp()).clamp(min=1e-7))
                kl_p = t_lt.exp() * (t_lt - s_lt) + t_lo.exp() * (t_lo - s_lo)
                p_sum, p_n = (kl_p * mask).sum(), mask.sum()
                # token-level KL on non-English rows
                tmask = mask & nonen_sl[:, None, None]
                s_lp = s_logits[..., :nb] - s_all[..., None]
                t_lp = t_logits[..., :nb] - t_all[..., None]
                kl_t = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
                k_sum, k_n = (kl_t * tmask).sum(), tmask.sum()
                return b_sum, b_n, mask.sum(), gate.sum(), p_sum, p_n, k_sum, k_n

            zero = torch.zeros((), device=f.device)
            b_tot = b_cnt = count = passed = p_tot = p_cnt = k_tot = k_cnt = zero
            for j in range(0, len(rows), sub_batch):
                sl = torch.tensor(rows[j:j + sub_batch], device=f.device)
                u = int(tokens_len[sl].max()) + 1
                t = int(enc_len[sl].max())
                span_sl = spans[sl, :u] if spans is not None else None
                r_ = torch.utils.checkpoint.checkpoint(
                    chunk_all, f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                    enc_len[sl], tokens_len[sl], span_sl, is_nonen[sl], use_reentrant=False)
                b_tot, b_cnt, count, passed = b_tot + r_[0], b_cnt + r_[1], count + r_[2], passed + r_[3]
                p_tot, p_cnt, k_tot, k_cnt = p_tot + r_[4], p_cnt + r_[5], k_tot + r_[6], k_cnt + r_[7]
            kl_b = blank_scale * (b_tot / b_cnt.clamp(min=1))
            kl_p = p_tot / p_cnt.clamp(min=1)
            kl_t = k_tot / k_cnt.clamp(min=1)
            out["loss"] = out["loss"] + weight * kl_b + punct_weight * kl_p + (tok_weight * kl_t if tok_weight else 0)
            m.log("kl_blank_dur", kl_b.detach(), prog_bar=False)
            if batch_idx % 10 == 0:
                n_non = int(is_nonen.sum())
                print(f"kl_punct {kl_p.item():.5f}\nkl_tok {kl_t.item():.5f} rows {n_non}\n"
                      f"kl_blank_dur {kl_b.item():.4f} (blank {kl_b.item():.4f} dur 0.0000) rows {len(rows)}/{f.shape[0]} "
                      f"gate-pass {int(passed)}/{int(count)} wait-pass 0", flush=True)
            if not (KL_EQ_CHECK and batch_idx < 3):
                return out
            merged_vals = (kl_b.item(), kl_p.item(), kl_t.item())
            out_loss = out["loss"]
            out["loss"] = orig_loss = out_loss.detach() * 0  # the separate passes below only report; no 2nd gradient
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
        if tok_weight and nonen_keep:
            def chunk_tok(f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl):
                with NoDropout(m.joint):
                    s_logits = m.joint.joint(f_sl, s_sl).float()
                with torch.no_grad():
                    t_logits = tjoint.joint(f_sl, t_sl).float()
                T, U = s_logits.shape[1], s_logits.shape[2]
                mask = ((torch.arange(T, device=f_sl.device)[None, :, None] < enc_len_sl[:, None, None])
                        & (torch.arange(U, device=f_sl.device)[None, None, :] <= tok_len_sl[:, None, None]))
                s_lp = s_logits[..., :nb].log_softmax(-1)
                t_lp = t_logits[..., :nb].log_softmax(-1)
                kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
                return (kl * mask).sum(), mask.sum()

            n_rows = [i for i in range(f.shape[0]) if tuple(tokens[i, :int(tokens_len[i])].tolist()) in nonen_keep]
            k_tot = k_cnt = torch.zeros((), device=f.device)
            for j in range(0, len(n_rows), sub_batch):
                sl = torch.tensor(n_rows[j:j + sub_batch], device=f.device)
                u = int(tokens_len[sl].max()) + 1
                t = int(enc_len[sl].max())
                ks, kn = torch.utils.checkpoint.checkpoint(
                    chunk_tok, f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                    enc_len[sl], tokens_len[sl], use_reentrant=False)
                k_tot, k_cnt = k_tot + ks, k_cnt + kn
            kl_t = k_tot / k_cnt.clamp(min=1)
            out["loss"] = out["loss"] + tok_weight * kl_t
            if batch_idx % 10 == 0:
                print(f"kl_tok {kl_t.item():.5f} rows {len(n_rows)}", flush=True)
        rows = list(range(f.shape[0]))
        spans = None
        if keep is not None:
            row_toks = [tokens[i, :int(tokens_len[i])].tolist() for i in rows]
            rows = [i for i, t in zip(rows, row_toks) if tuple(t) in keep]
            if "durspan" in tset0:
                spans = span_mask(m.tokenizer.tokenizer, [(i, row_toks[i]) for i in rows],
                                  tokens.shape[0], int(tokens_len.max()), f.device)
            elif "blankspan" in tset0:
                spans = span_mask(m.tokenizer.tokenizer, [(i, row_toks[i]) for i in rows if tuple(row_toks[i]) in span_keep],
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
        if merged:  # KL_EQ_CHECK: compare the merged values with the separate passes, keep the merged gradient
            sep = (kl_b.item(), kl_p.item() if punct_weight else 0.0, kl_t.item() if (tok_weight and nonen_keep) else 0.0)
            print(f"KL_EQ_CHECK batch {batch_idx}: merged {merged_vals} separate {sep}", flush=True)
            out["loss"] = out_loss
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


# ---------------------------------------------------------------------------------------------------------------
# RETRAIN-3 auxiliary losses. One joint pass per sub-batch over EVERY row:
#   tag      English  blank+gnorm KL      punct KL         token KL  wf_pen  en_pen
#   anchor   yes      full                full             -         yes     -
#   anchor_num yes    -  (v10pt numwords exemption)  -     -         yes     -
#   span     yes      prefix only (u < first number word - 1 word: the teacher, forced on the student's spoken
#                     number tokens, is off-distribution after them)   prefix only  -  yes  -
#   nonen    no       full                full             yes       -       yes
#   nonen_span no     prefix only (span_prefix: before the first number reading - 1 word; wave-1 spoken-form rows and
#                     their window-edge crops)  prefix only  prefix only  yes  yes
#   free     yes      -                   -                -         yes     -
# wf_pen / en_pen: -log(1 - P(set | non-blank)) at the student's emitting cells (P(non-blank) > 0.1, detached),
# averaged per row over that row's emitting cells, then over rows: conditioned on non-blank so it never moves the blank
# logit, per-row so it does not scale with T x U.
WF_CHARS = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°⁇²³¹½¼¾⅓⅔]")
EN_NUM_WORDS = ("two three four five seven eight eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
                "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand first third").split()
# no "one" (pl "they"), "second" (fr), six, ten (pl/cs/sk/nl), nine. en_set "whole" (the default): only whole-word
# pieces count (en_num_piece_ids). The tokenizer splits most of these words, so only "two" and "first" survive there.
# The token KL does not guard the multi-piece ones on nonen_span number positions either (it skips those columns).
# en_set "first" (en_num_first_piece_ids): the first piece of each word in EN_NUM_WORDS_FIRST. The data decides which
# pieces are shared with other languages, so the list can be wider: a piece common as a word start in the
# non-English labels ("▁six", "▁sette", "▁mil", "▁zero", ...) is dropped there, whatever word it came from.
EN_NUM_WORDS_FIRST = tuple(EN_NUM_WORDS) + tuple(
    "zero one six nine ten million billion point percent second fourth fifth twelfth hundredth thousandth".split())
EN_FIRST_MIN_LETTERS = 3  # letters after "▁": shorter first pieces ("▁t" of thousand, "▁se") start too many words
EN_FIRST_MAX_NONEN = 20  # a first piece seen this often as a word start in non-English labels is shared: not penalised


TWENTY_PAIR = re.compile(r"\btwenty (?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                         r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b")


def wf_piece_ids(spm) -> list[int]:
    return [i for i in range(spm.get_piece_size()) if WF_CHARS.search(spm.id_to_piece(i))
            and not re.fullmatch(r"<\|.*\|>", spm.id_to_piece(i))]


def en_num_piece_ids(spm) -> list[int]:
    want = {"▁" + w for w in EN_NUM_WORDS}
    return [i for i in range(spm.get_piece_size()) if spm.id_to_piece(i) in want]


def num_piece_ids(spm, labels_flags, min_count: int = 20, min_share: float = 0.5) -> tuple[list[int], int]:
    """Pieces that mostly spell spoken numbers in non-English rows: `labels_flags` = [(token ids, per-word number flags)]
    (nonen_span rows, word_flags). A piece qualifies with >= min_count occurrences inside number words and at least
    min_share of all its occurrences in those rows inside them. Returns (ids, rows skipped on a word-split mismatch, ids
    with a share in [min_share / 2, min_share): logged so the boundary is visible)."""
    num, other, mis = collections.Counter(), collections.Counter(), 0
    for ids, flags in labels_flags:
        words = row_words(spm, ids)
        if len(words) != len(flags):
            mis += 1
            continue
        for (_, a, b), f in zip(words, flags):
            for j in range(a, b + 1):
                (num if f else other)[int(ids[j])] += 1
    keep = [i for i, c in num.items() if c >= min_count and c / (c + other[i]) >= min_share]
    near = [i for i, c in num.items() if c >= min_count and min_share / 2 <= c / (c + other[i]) < min_share]
    return sorted(keep), mis, sorted(near)


def word_start_counts(spm, labels, only=None) -> collections.Counter:
    """How often each word-initial piece ("▁...") occurs over `labels` (token-id sequences), counting only ids in
    `only` when given."""
    c = collections.Counter()
    for ids in labels:
        for i in ids:
            if (only is None or i in only) and spm.id_to_piece(int(i)).startswith("▁"):
                c[int(i)] += 1
    return c


def en_num_first_piece_ids(spm, encode, nonen_labels, words=EN_NUM_WORDS_FIRST, min_letters=EN_FIRST_MIN_LETTERS,
                           max_nonen=EN_FIRST_MAX_NONEN) -> tuple[list[int], dict]:
    """en_set "first": per English number word, the first piece of its tokenization (`encode(word)`: the word-initial
    form, as mid-sentence); kept when it starts with "▁", has >= min_letters letters after it, and occurs fewer than
    max_nonen times as a word start over `nonen_labels` (the distinct non-English label token sequences). Returns
    (kept ids, report): report = {"kept": {piece: [words]}, "excluded": {piece: (count, [words])},
    "short": {word: piece}, "covered": n words with a kept piece, "words": n words}."""
    first = {}
    for w in words:
        ids = encode(w)
        if ids:
            first.setdefault(int(ids[0]), []).append(w)
    counts = word_start_counts(spm, nonen_labels, only=set(first))
    kept, rep = [], {"kept": {}, "excluded": {}, "short": {}, "words": len(words)}
    for i, ws in first.items():
        p = spm.id_to_piece(i)
        if not p.startswith("▁") or sum(ch.isalpha() for ch in p[1:]) < min_letters:
            rep["short"].update({w: p for w in ws})
        elif counts[i] >= max_nonen:
            rep["excluded"][p] = (counts[i], ws)
        else:
            kept.append(i)
            rep["kept"][p] = ws
    rep["covered"] = sum(len(ws) for ws in rep["kept"].values())
    return sorted(kept), rep


# decimal-comma penalty (comma_weight, try 3): the stock prior writes "3,5" where a spoken-form label says the decimal
# word (komma / coma / virgule / virgola); "," is no written-form piece and the punctuation KL anchors terminal marks
# only, so no other term covers it. Exact pieces only (a substring match would reach every piece holding a comma); the
# Ultra vocabulary has ",", ",-" and ",." ("▁," listed in case a tokenizer has it).
COMMA_PIECES = (",", "\u2581,", ",-", ",.")


def comma_piece_ids(spm) -> list[int]:
    """The ids of the pieces that are exactly one of COMMA_PIECES."""
    return [i for i in range(spm.get_piece_size()) if spm.id_to_piece(i) in COMMA_PIECES]


def comma_target_cols(row_toks, comma_ids, W: int) -> "torch.Tensor":
    """(B, W) True at label column u where the row's own target piece row_toks[i][u] is a comma piece: the comma
    penalty skips those columns (a list or clause comma in the label is right there, and penalising it would fight the
    RNNT loss); the rest of the row keeps the penalty."""
    cs = {int(c) for c in comma_ids}
    out = torch.zeros(len(row_toks), W, dtype=torch.bool)
    for i, toks in enumerate(row_toks):
        for u, t in enumerate(toks[:W]):
            if int(t) in cs:
                out[i, u] = True
    return out


def wf_row_weights(is_nspan, wf_en: float) -> "torch.Tensor":
    """Per-row factor on wf_pen (wf_en_weight): wf_en on the rows wf_pen covers that are not nonen_span (the English
    spoken-form rows), 1 on nonen_span rows, so their digit fallback stays as it is."""
    return torch.where(is_nspan, torch.ones((), device=is_nspan.device),
                       torch.full((), float(wf_en), device=is_nspan.device))


def en_pen_cols(klu, nonen, nspan, en_set: str):
    """Label columns where en_pen applies, per row: "whole" = every column (rows: nonen + nonen_span); "first" = every
    column of nonen rows, and on nonen_span rows only the number columns (klu False: where the token KL to stock is
    off), so one English word inside a number is not diluted by the rest of the row."""
    if en_set == "whole":
        return None
    return nonen[:, None] | (nspan[:, None] & ~klu)


def numberish(word: str) -> bool:
    from common import NUMBER_WORDS

    w = word.lower().strip(".,;:!?\"'")
    return bool(NUMBER_WORDS.search(w)) or w in {"one", "oh", "zero", "first", "second", "third", "fourth", "fifth",
                                                 "sixth", "seventh", "eighth", "ninth", "tenth", "hundredth"}


def prefix_len(spm, ids) -> int:
    """Label positions [0, k) before the first number word, minus one word (number_span misses "Q one", "COVID")."""
    words = row_words(spm, ids)
    for wi, (w, a, b) in enumerate(words):
        if numberish(w):
            return words[wi - 1][1] if wi > 0 else 0
    return len(ids) + 1


def nspan_prefix(row: dict) -> str:
    """The label text before the row's first spoken number: P3 rows carry `num_chars` (reading offsets), window-edge
    crops `num_tokens` (a flag per whitespace token). A row with no number keeps its whole label."""
    if "num_chars" in row:
        return row["text"][:row["num_chars"][0][0]] if row["num_chars"] else row["text"]
    if "num_tokens" in row:
        toks = row["text"].split()
        k = row["num_tokens"].index(True) if True in row["num_tokens"] else len(toks)
        return " ".join(toks[:k])
    # a row without offsets would be fully anchored to the digit-writing teacher
    raise ValueError(f"nonen_span row without num_chars / num_tokens: {row.get('id') or row['text'][:60]}")


# ---------------------------------------------------------------------------------------------------------------
# R4 rehearsal arm (b) "drop fix": anchor every NON-number label position (prefix AND suffix) instead of the prefix only.
# Label position j = the decoder state after tokens[:j], predicting tokens[j]; position len(ids) predicts the end.
# Unanchored positions: (1) the number words themselves (the stock teacher writes digits there); (2) `before` words before
# each number run (the prefix rule's margin: number_span misses "Q one", "COVID"); (3) `after` words after each number
# run, and the trailing position when it falls inside that guard: the teacher is teacher-forced on the student's SPOKEN
# number tokens, a prediction-network state stock never produced, so its next-word distribution right after a number is
# off-distribution. Default after = 1: one word lets the LSTM state settle on ordinary text again; the blank-KL gate
# (teacher P(non-blank) > threshold) filters further positions where the teacher is unsure.
ROMAN_WORD = re.compile(r"(?:II|III|IV|VI|VII|VIII|IX|XI|XII)[.,;:!?]*")


def en_number_words(words: list[str]) -> list[bool]:
    """English number flags per word: number_span OR numberish OR a Roman numeral (write_mix's says_number set). Wider
    than any single rule: an over-flagged word only loses anchoring, an under-flagged one is anchored to digits."""
    span = number_span(words)
    return [bool(s) or numberish(w) or bool(ROMAN_WORD.fullmatch(w))
            or (w.rstrip(".,;:!?") == "I" and k > 0 and words[k - 1].lower() == "war")
            for k, (w, s) in enumerate(zip(words, span))]


def word_flags(row: dict) -> list[bool]:
    """Per whitespace token of a nonen_span row's label: True inside a number reading. `num_tokens` (windows, crops) is
    used as is; `num_chars` (P3 / synthetic rows: reading character offsets) by overlap, as ml_p3_core.number_tokens."""
    toks = row["text"].split()
    if "num_tokens" in row:
        flags = [bool(x) for x in row["num_tokens"]]
        if len(flags) != len(toks):
            raise ValueError(f"num_tokens length {len(flags)} != {len(toks)} tokens: {row.get('id') or row['text'][:60]}")
        return flags
    if "num_chars" in row:
        flags, pos = [], 0
        for tok in toks:
            a = row["text"].index(tok, pos)
            pos = a + len(tok)
            flags.append(any(x < pos and a < y for x, y in row["num_chars"]))
        return flags
    raise ValueError(f"nonen_span row without num_chars / num_tokens: {row.get('id') or row['text'][:60]}")


def anchor_positions(words: list[tuple[str, int, int]], flags: list[bool], n_ids: int, before: int = 1,
                     after: int = 1) -> list[bool]:
    """(n_ids + 1) flags, True where the anchor KL applies. `words` = row_words(...) (text, first, last label position),
    `flags` = per-word number flags (same length). See the block comment above for what stays unanchored."""
    assert len(words) == len(flags), (len(words), len(flags))
    nw = len(words)
    blocked = [False] * (nw + 1)  # index nw = the trailing (end) position
    for k, f in enumerate(flags):
        if f:
            for q in range(max(0, k - before), min(nw, k + after) + 1):
                blocked[q] = True
    keep = [True] * (n_ids + 1)
    for (_, a, b), bl in zip(words, blocked):
        if bl:
            keep[a:b + 1] = [False] * (b + 1 - a)
    keep[n_ids] = not blocked[nw]
    return keep


def klu_row(spm, tag: str, ids, suffix_anchor: bool = False, anchor_after: int = 1, nspan_pos: dict | None = None,
            span_prefix: dict | None = None, cache: dict | None = None, en_full: bool = False,
            en_free: bool = False) -> list[bool]:
    """(len(ids) + 1) flags: the label positions where the anchor KL applies for one row (add_r3_losses' table; padding
    beyond len(ids) is masked there anyway). R4 arm b (suffix_anchor): span and anchor_num rows anchor every non-number
    position (anchor_positions over en_number_words); nonen_span rows use nspan_pos (built in train() from the row's
    number tokens) and fall back to the prefix rule when it has no entry. en_full (en_teacher with en_teacher_num
    "anchor"): span and anchor_num rows anchor EVERY position, number words included (EN_TEACHER block comment).
    en_free (en_free_teacher): the English "free" rows anchor every position too, to the English teacher."""
    n, key = len(ids), tuple(ids)
    if tag in ("anchor", "nonen"):
        return [True] * (n + 1)
    if en_free and tag == "free":
        return [True] * (n + 1)
    if en_full and tag in ("span", "anchor_num"):
        return [True] * (n + 1)
    if suffix_anchor and tag in ("span", "anchor_num"):
        if cache is None or key not in cache:
            words = row_words(spm, ids)
            pos = anchor_positions(words, en_number_words([w for w, _, _ in words]), n, after=anchor_after)
            if cache is None:
                return pos
            cache[key] = pos
        return cache[key]
    if tag == "span":
        k = prefix_len(spm, ids)
        return [j < k for j in range(n + 1)]
    if tag == "nonen_span":  # non-English spoken-form numbers: the stock teacher writes digits there
        pos = (nspan_pos or {}).get(key) if suffix_anchor else None
        if pos is not None:
            return list(pos)
        k = (span_prefix or {}).get(key, 0)  # arm a, or a row whose word split did not line up (counted in train())
        return [j < k for j in range(n + 1)]
    return [False] * (n + 1)  # free, and anchor_num without suffix anchoring (v10pt's numwords exemption)


# ---------------------------------------------------------------------------------------------------------------
# English KL options (2026-10-18). Stock-init arms never relearned B's English spoken form while
# the English rows were KL-anchored to stock (a digit writer); the anchors-off arm did. Two arms, both off by default:
# en_kl_scale s (!= 1): the KL terms (blank, punct, duration, English token) of the English KL rows (EN_TEACHER_TAGS)
#   are multiplied by s per row. Only their numerators: the cell counts the KL averages divide by are unchanged, so
#   the non-English rows' contribution stays exactly what it is at s = 1 (s = 0: no KL on English, the rest as stock).
# en_teacher (a decoder + joint checkpoint, e.g. B = w3soup_a0.5@0): a second frozen teacher. English KL rows read ITS
#   logits for every KL term; every other row keeps stock. The teacher reads the same encoder output as the stock
#   teacher (B was trained with the encoder frozen: its encoder IS stock; a checkpoint holding encoder tensors is
#   refused). Masks under this teacher:
#   - token KL: English rows had none (stock writes digits, the RNNT term owned the tokens); with a spoken-form teacher
#     they get one, same rule as the non-English rows (non-blank distribution at the teacher's emitting cells), as its
#     own term (kl_te, weight tok_weight) so the non-English token-KL average is not diluted.
#   - number cells, en_teacher_num "anchor" (default): span / anchor_num rows anchor every label position (klu_row
#     en_full). Both reasons for the stock masks are gone: the teacher writes the number words the label holds, and
#     teacher forcing on spoken number tokens is its own distribution, so no guard word is needed after them.
#     en_teacher_num "mask": the stock-era positions (klu_row as without the teacher); the teacher only swaps logits.
#   - digit cells: wherever the English teacher still puts > 0.1 of its non-blank mass on written-form pieces (B writes
#     digits on ~11 English sentinel items, alnum codes "Q4"), the cell is dropped from every KL term (the nonen_span
#     wf_cell rule; with kl_mask_space the bare "▁" counts too).
EN_TEACHER_TAGS = ("anchor", "anchor_num", "span")
# en_free_teacher (default off, needs en_teacher with en_teacher_num "anchor"): the "free" rows (English synthetic number
# rows: they carry the Q-codes and the year minimal pairs) join the English KL rows. Today they get no KL at all (klu
# False, no teacher), so nothing held their English spoken form while the shared digit-entry cells were pulled to digits
# by the non-English rows. With the flag they read the English teacher for every KL term, like the other English rows
# (en_kl_scale applies to them too). en_span_gate g > 0 (default 0 = off): the blank-KL teacher gate of the English
# "span" rows is g instead of `gate` (row_gate).


def en_kl_tags(en_free: bool) -> tuple:
    """The tags whose rows read the English teacher / take en_kl_scale."""
    return EN_TEACHER_TAGS + (("free",) if en_free else ())


def routed_teacher_logits(stock_joint, en_joint, f, g_stock, g_en, en_rows):
    """Teacher joint logits (float) for one sub-batch: rows where en_rows is True from en_joint on g_en, the others from
    stock_joint on g_stock, both on the same encoder features f (B, T, D). en_joint None = the stock call, unchanged.
    Each joint runs only on its own rows."""
    if en_joint is None or not bool(en_rows.any()):
        return stock_joint.joint(f, g_stock).float()
    if bool(en_rows.all()):
        return en_joint.joint(f, g_en).float()
    a = stock_joint.joint(f[~en_rows], g_stock[~en_rows]).float()
    b = en_joint.joint(f[en_rows], g_en[en_rows]).float()
    out = a.new_empty((f.shape[0],) + tuple(a.shape[1:]))
    out[~en_rows] = a
    out[en_rows] = b
    return out


def build_en_teacher(m, state: dict):
    """(decoder, joint): frozen eval copies of m's decoder / joint with `state`'s tensors ("decoder." / "joint." names,
    a train() step checkpoint). Refuses a checkpoint with any other tensor (an encoder fine-tune cannot share the stock
    teacher's encoder output) or one that leaves a decoder / joint tensor unset."""
    import copy

    other = [k for k in state if not k.startswith(("decoder.", "joint."))]
    assert not other, f"en_teacher checkpoint holds {len(other)} non decoder / joint tensors, e.g. {other[:3]}"
    dec, joint = copy.deepcopy(m.decoder), copy.deepcopy(m.joint)
    for mod, pre in ((dec, "decoder."), (joint, "joint.")):
        missing, unexpected = mod.load_state_dict({k[len(pre):]: v for k, v in state.items() if k.startswith(pre)},
                                                  strict=False)
        assert not unexpected and not missing, (pre, missing[:3], unexpected[:3])
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    return dec, joint


def blank_gate(kmask, t_lnb, thr):
    """Cells where the blank KL applies: kmask and teacher P(non-blank) > the row's threshold. kmask / t_lnb (B, T, U),
    thr (B,)."""
    return kmask & (t_lnb.exp() > thr[:, None, None])


def row_gate(tags: list[str], keys: list, gate: float, gate_human: float, synthetic: set,
             span_gate: float = 0.0) -> list[float]:
    """Per-row blank-KL teacher gate (teacher P(non-blank) must exceed it). gate_human (when > 0) replaces `gate` on the
    human-verified rows: anchor, anchor_num, and nonen_span rows whose label is not synthetic (TTS rows, TTS windows).
    span_gate (when > 0, en_span_gate) replaces it on the English "span" rows; 0 = unchanged."""
    out = [gate] * len(tags)
    if gate_human:
        out = [gate_human if tg in ("anchor", "anchor_num") or (tg == "nonen_span" and k not in synthetic) else g
               for g, tg, k in zip(out, tags, keys)]
    if span_gate:
        out = [span_gate if tg == "span" else g for g, tg in zip(out, tags)]
    return out


def per_tag_summary(tags: list[str], acc: list) -> dict:
    """{tag: {term: value, term_n: count}} from the per-row (numerator, denominator) vectors of add_r3_losses' chunk:
    acc[0/1] blank KL, [2/3] punct KL, [4/5] token KL, [6/7] wf_pen, [8/9] en_pen (numerator / count per row), and
    [16/17] the English token KL (en_teacher) when acc holds it."""
    out = {}
    for tg in sorted(set(tags)):
        ix = torch.tensor([i for i, t in enumerate(tags) if t == tg], device=acc[0].device)
        d = {"rows": len(ix)}
        for name, k in (("kl_b", 0), ("kl_p", 2), ("kl_t", 4), ("wf", 6), ("en", 8)) + ((("kl_te", 16),) if len(acc) > 17 else ()):
            num, den = acc[k][ix].sum(), acc[k + 1][ix].sum()
            d[name], d[name + "_n"] = float(num / den.clamp(min=1)), int(den)
        out[tg] = d
    return out


# ---------------------------------------------------------------------------------------------------------------
# R4 rehearsal arm (c) "cross-window consistency". The same number reading heard in two examples (two streaming windows
# of one stream at different phases, or a window and its parent row) should get the same token distribution at its
# emission point. Pair list (one JSON object per line, built by the mix builder):
#   {"a": {"audio_filepath", "duration", "text"}, "b": {...}, "a_span": [i0, i1], "b_span": [j0, j1],
#    "reading": "zweiundzwanzig", "parent": "<row id>", "lang": "de"}
# a_span / b_span: whitespace-token index range [start, end) of the shared reading in a.text / b.text; both must equal
# reading.split(). Per label position u of the reading (piece level), the emission point is a proxy for the RNNT
# alignment without forward-backward: w_t = softmax over valid frames t of log P(y_u | t, u) (detached), and the
# compared distribution is the w-weighted mixture of the student's non-blank token distribution P(. | t, u, non-blank).
# Loss = symmetric KL 0.5 (KL(A||B) + KL(B||A)) averaged over the reading's positions and pairs; gradient flows into
# both sides (neither is a teacher: the point is agreement, the RNNT term decides which reading).
def load_pairs(lines, known_texts: set | None = None) -> tuple[list[dict], collections.Counter]:
    """Parse + validate pair lines. A pair whose span does not match its reading is a builder bug (raise); a pair whose
    members are not training rows of this mix (known_texts) is skipped and counted."""
    pairs, st = [], collections.Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        for side in ("a", "b"):
            x = p[side]
            for k in ("audio_filepath", "text"):
                if k not in x:
                    raise ValueError(f"pair {p.get('parent')}: {side} without {k}")
            i0, i1 = p[side + "_span"]
            if not (0 <= i0 < i1 <= len(x["text"].split())) or x["text"].split()[i0:i1] != p["reading"].split():
                raise ValueError(f"pair {p.get('parent')}: {side}_span {p[side + '_span']} is not {p['reading']!r} in "
                                 f"{x['text'][:80]!r}")
        if known_texts is not None and not (p["a"]["text"] in known_texts and p["b"]["text"] in known_texts):
            st["not_in_mix"] += 1
            continue
        pairs.append(p)
        st["kept"] += 1
    return pairs, st


def span_label_positions(spm, ids, text: str, span) -> list[int] | None:
    """Label positions (piece indices) of the whitespace-token range `span` of `text`; None when the tokenizer's word
    split (row_words) does not line up with the whitespace tokens."""
    words = row_words(spm, ids)
    if len(words) != len(text.split()):
        return None
    i0, i1 = span
    return list(range(words[i0][1], words[i1 - 1][2] + 1))


def emission_logdist(logits, labels, t_len: int, nb: int = 8193):
    """logits (T, P, V') of the joint at the reading's P label positions, labels (P,) the reading's token ids, t_len valid
    frames. Returns (P, nb - 1) log of the emission-weighted non-blank token distribution (block comment above)."""
    T = logits.shape[0]
    lp = logits[..., :nb].float().log_softmax(-1)
    ly = lp.gather(-1, labels.view(1, -1, 1).expand(T, -1, 1)).squeeze(-1)  # (T, P) log P(y_u | t, u)
    valid = (torch.arange(T, device=logits.device) < t_len)[:, None]
    lw = ly.detach().masked_fill(~valid, float("-inf")).log_softmax(0)
    ltok = logits[..., :nb - 1].float().log_softmax(-1)
    return torch.logsumexp(ltok + lw[..., None], dim=0)


def sym_kl(la, lb):
    """Mean over positions of 0.5 (KL(a||b) + KL(b||a)); la, lb (P, V) log-probabilities."""
    pa, pb = la.exp(), lb.exp()
    return (0.5 * ((pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1))).mean()


class PairBank:
    """Draws `per_batch` pairs per training batch (seeded; with replacement across batches)."""

    def __init__(self, pairs: list[dict], per_batch: int = 1, seed: int = 1, stage_dir: str = ""):
        self.pairs, self.per_batch, self.rng, self.stage_dir = pairs, per_batch, random.Random(seed), stage_dir

    def path(self, p: str) -> str:
        """The staged copy when write_mix staged this file (pair members are mix rows)."""
        s = self.stage_dir + p if self.stage_dir else ""
        return s if s and os.path.exists(s) else p

    def sample(self) -> list[dict]:
        return [self.rng.choice(self.pairs) for _ in range(self.per_batch)] if self.pairs else []


def consistency_term(m, bank: PairBank, dev, nb: int = 8193):
    """(loss, number of pairs used) for one batch: one frozen-encoder pass over the 2 x per_batch clips, a teacher-forced
    student decoder pass, and the joint only at each reading's label positions."""
    import soundfile as sf

    spm = m.tokenizer.tokenizer
    items, sides = [], []
    for p in bank.sample():
        side = []
        for s in ("a", "b"):
            ids = list(m.tokenizer.text_to_ids(p[s]["text"]))
            pos = span_label_positions(spm, ids, p[s]["text"], p[s + "_span"])
            side.append((ids, pos))
        if side[0][1] is None or side[1][1] is None or [side[0][0][j] for j in side[0][1]] != [side[1][0][j] for j in side[1][1]]:
            continue
        for s, (ids, pos) in zip(("a", "b"), side):
            x, sr = sf.read(bank.path(p[s]["audio_filepath"]), dtype="float32")
            assert sr == 16000, (p[s]["audio_filepath"], sr)
            items.append((torch.from_numpy(x.mean(1) if x.ndim > 1 else x), ids, pos))
        sides.append(len(items) - 2)
    if not sides:
        return torch.zeros((), device=dev), 0
    n = len(items)
    sig = torch.zeros(n, max(len(x) for x, _, _ in items))
    for i, (x, _, _) in enumerate(items):
        sig[i, :len(x)] = x
    sig_len = torch.tensor([len(x) for x, _, _ in items])
    toks = torch.zeros(n, max(len(ids) for _, ids, _ in items), dtype=torch.long)
    for i, (_, ids, _) in enumerate(items):
        toks[i, :len(ids)] = torch.tensor(ids)
    tok_len = torch.tensor([len(ids) for _, ids, _ in items])
    sig, sig_len, toks, tok_len = sig.to(dev), sig_len.to(dev), toks.to(dev), tok_len.to(dev)
    with torch.no_grad():
        feats, feats_len = m.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = m.encoder(audio_signal=feats, length=feats_len)
    with NoDropout(m.decoder):
        g, _, _ = m.decoder(targets=toks, target_length=tok_len)
    f = enc.transpose(1, 2)
    dists = []
    for i, (_, ids, pos) in enumerate(items):
        t = int(enc_len[i])
        pos_t = torch.tensor(pos, device=dev)
        with NoDropout(m.joint):
            lg = m.joint.joint(f[i:i + 1, :t], g[i:i + 1, :, pos_t].transpose(1, 2))[0]
        dists.append(emission_logdist(lg, toks[i, pos_t], t, nb))
    loss = sum(sym_kl(dists[k], dists[k + 1]) for k in sides) / len(sides)
    return loss, len(sides)


def add_r3_losses(m, row_tags: dict, kl_weight: float = 10.0, punct_weight: float = 30.0, tok_weight: float = 1.0,
                  wf_weight: float = 1.0, en_weight: float = 1.0, sub_batch: int = 2, emit_p: float = 0.1,
                  dur_weight: float = 0.0, span_prefix: dict | None = None, suffix_anchor: bool = False,
                  anchor_after: int = 1, nspan_pos: dict | None = None, gate: float = 0.9, gate_human: float = 0.0,
                  synthetic: set | None = None, cons_weight: float = 0.0, pair_bank: "PairBank | None" = None,
                  en_set: str = "whole", comma_weight: float = 0.0, wf_en_weight: float = 1.0,
                  kl_mask_space: bool = False, en_teacher_state: dict | None = None, en_teacher_num: str = "anchor",
                  en_kl_scale: float = 1.0, wf_pen_digits_only: bool = False,
                  kl_mask_space_en: bool = False, enc_real_only: bool = False, en_free_teacher: bool = False,
                  en_span_gate: float = 0.0):
    """en_free_teacher / en_span_gate (both off by default, bit for bit): the "free" rows join the English KL rows, and the
    English "span" rows get blank-KL gate en_span_gate (block comment at EN_TEACHER_TAGS).
    enc_real_only (unfreeze_top only): the rows whose label is in `synthetic` send no gradient into the encoder (every
    loss, main and auxiliary, reads the same encoder output, whose value is unchanged; see unfreeze_top_layers). The decoder
    and joint train on every row as before. `synthetic` holds the nonen_span labels of TTS rows / TTS windows (write_mix);
    False = nothing changes, bit for bit.
    wf_pen_digits_only: wf_pen counts only the pieces made of digits (0-9), not the symbol pieces (% $ € £ / …) that the
    frozen encoder carries at spoken percent / currency / unit words; False = unchanged bit for bit.
    R3 losses (table above). R4 options, all off by default: suffix_anchor (arm b: anchor every non-number position of
    span / anchor_num rows, and of nonen_span rows via nspan_pos {label ids: anchor_positions}), gate_human (arm b: the
    blank-KL teacher gate on human-verified rows, `gate` elsewhere), cons_weight + pair_bank (arm c: cross-window
    consistency, consistency_term). en_set: the en_pen piece set and columns ("whole" | "first"; EN_NUM_WORDS_FIRST,
    en_pen_cols). comma_weight > 0: the decimal-comma penalty (comma_piece_ids) at the number columns of nonen_span rows
    (klu False, as en_set "first"), except columns whose own target is a comma (comma_target_cols); 0 = the loss is
    unchanged bit for bit. wf_en_weight != 1: wf_pen per row x wf_row_weights (English rows x wf_en_weight, nonen_span
    rows x 1); 1 = unchanged bit for bit. kl_mask_space: the nonen_span "teacher is writing digits" cell mask also counts
    the bare word-boundary piece "▁" (the only entry into digit mode: the vocabulary has no "▁2"-style pieces), so the
    KL terms stop teaching digit entry over the number's audio at units-first onsets; False = unchanged bit for bit.
    en_teacher_state (a decoder + joint state dict) / en_teacher_num / en_kl_scale: the English KL options (block comment
    at EN_TEACHER_TAGS); None / 1 = unchanged bit for bit."""
    import copy

    assert en_set in ("whole", "first"), en_set
    assert en_teacher_num in ("anchor", "mask"), en_teacher_num
    assert en_kl_scale >= 0, en_kl_scale
    assert 0.0 <= en_span_gate < 1.0, en_span_gate
    assert not en_free_teacher or (en_teacher_state is not None and en_teacher_num == "anchor"), \
        "en_free_teacher needs en_teacher with en_teacher_num anchor"
    en_tags = en_kl_tags(en_free_teacher)

    teacher_dec, teacher_joint = copy.deepcopy(m.decoder).eval(), copy.deepcopy(m.joint).eval()
    for p in list(teacher_dec.parameters()) + list(teacher_joint.parameters()):
        p.requires_grad_(False)
    m._kl_teacher = (teacher_dec, teacher_joint)
    m._kl_en_teacher = build_en_teacher(m, en_teacher_state) if en_teacher_state is not None else None
    en_full = m._kl_en_teacher is not None and en_teacher_num == "anchor"
    if m._kl_en_teacher is not None or en_kl_scale != 1.0:
        print(f"English KL rows {en_tags}: teacher {'second checkpoint' if m._kl_en_teacher else 'stock'}"
              + (f" (number cells: {en_teacher_num}; English token KL on)" if m._kl_en_teacher else "")
              + f", KL x {en_kl_scale}; every other row: stock teacher, KL x 1", flush=True)
    spm = m.tokenizer.tokenizer
    term_t = torch.tensor([i for i in range(spm.get_piece_size())
                           if spm.id_to_piece(i).lstrip("▁") in {".", "?", "!", "...", "…"}])
    if en_set == "first":
        nonen_labels = [k for k, tg in row_tags.items() if tg in ("nonen", "nonen_span")]
        en_ids, rep = en_num_first_piece_ids(spm, lambda w: m.tokenizer.text_to_ids(w), nonen_labels)
        print(f"en_set first: kept {len(rep['kept'])} pieces covering {rep['covered']}/{rep['words']} English number "
              f"words {rep['kept']}; excluded (>= {EN_FIRST_MAX_NONEN} word starts in {len(nonen_labels)} non-English "
              f"labels) {rep['excluded']}; too short {rep['short']}", flush=True)
    else:
        en_ids = en_num_piece_ids(spm)
    wf_t, en_t = torch.tensor(wf_piece_ids(spm)), torch.tensor(en_ids, dtype=torch.long)
    space_ids = [i for i in range(spm.get_piece_size()) if spm.id_to_piece(i) == "▁"]
    wfm_t = torch.tensor(wf_piece_ids(spm) + space_ids) if kl_mask_space or kl_mask_space_en else wf_t
    wfp_t = wf_t
    if wf_pen_digits_only:
        wfp_t = torch.tensor([i for i in wf_piece_ids(spm) if re.fullmatch(r"▁?[0-9]+", spm.id_to_piece(i))])
        assert len(wfp_t), "wf_pen_digits_only but no digit piece"
        print(f"wf_pen_digits_only: wf_pen on {len(wfp_t)} digit pieces (of {len(wf_t)} written-form pieces)", flush=True)
    if kl_mask_space:
        assert space_ids, "kl_mask_space but no bare '▁' piece in the tokenizer"
        print(f"kl_mask_space: nonen_span digit-cell mask counts wf pieces + '▁' {space_ids}", flush=True)
    if kl_mask_space_en:  # the '▁' count on the English teacher's rows only; nonen_span rows keep the wf-only mask
        assert not kl_mask_space and en_teacher_state is not None and space_ids, "kl_mask_space_en needs en_teacher, not kl_mask_space"
        print(f"kl_mask_space_en: English-teacher rows' digit-cell mask counts wf pieces + '▁' {space_ids}; nonen_span rows wf only",
              flush=True)
    th_first = int(m.tokenizer.text_to_ids("thousand")[0])
    th_t = torch.tensor([th_first])
    cm_t = torch.tensor(comma_piece_ids(spm) if comma_weight else [], dtype=torch.long)
    if comma_weight:
        assert len(cm_t), "comma_weight > 0 but no comma piece in the tokenizer"
        print(f"comma pen: weight {comma_weight} pieces {[(int(i), spm.id_to_piece(int(i))) for i in cm_t]}", flush=True)
    if en_span_gate:
        print(f"en_span_gate {en_span_gate}: blank-KL gate of the English span rows (others: gate {gate}, "
              f"gate_human {gate_human or '-'})", flush=True)
    if wf_en_weight != 1.0:
        print(f"wf_en_weight {wf_en_weight}: wf_pen x {wf_en_weight} on English rows, x 1 on nonen_span rows", flush=True)
    print(f"R3 losses: wf pieces {len(wf_t)} e.g. {[spm.id_to_piece(int(i)) for i in wf_t[:12]]}; "
          f"en_num pieces ({en_set}) {[spm.id_to_piece(int(i)) for i in en_t]}; thousand first piece {spm.id_to_piece(th_first)!r}; tags {dict(collections.Counter(row_tags.values()))}",
          flush=True)
    print(f"R4 options: suffix_anchor {suffix_anchor} (after {anchor_after} word, nonen_span rows with positions "
          f"{len(nspan_pos or {})}) gate {gate} gate_human {gate_human or '-'} (synthetic labels {len(synthetic or ())}) "
          f"consistency {cons_weight} (pairs {len(pair_bank.pairs) if pair_bank else 0})", flush=True)
    orig_step = m.training_step
    nb = 8193
    en_cache: dict = {}
    if enc_real_only:
        assert getattr(m, "_enc_unfrozen", ()), "enc_real_only needs unfreeze_top"
        assert synthetic, "enc_real_only but no synthetic label (nothing to mask)"
        syn_rows = {"seen": 0, "masked": 0}

    def training_step(batch, batch_idx):
        if enc_real_only:  # before the forward: the encoder hook reads the row mask
            tk, tl = batch[2], batch[3]
            keep = torch.tensor([tuple(tk[i, :int(tl[i])].tolist()) not in synthetic for i in range(tk.shape[0])])
            m._enc_row_keep = keep
            syn_rows["seen"] += len(keep)
            syn_rows["masked"] += int((~keep).sum())
            if batch_idx % 10 == 0:
                print(f"enc_real_only: rows without encoder gradient {syn_rows['masked']}/{syn_rows['seen']}", flush=True)
        out = orig_step(batch, batch_idx)
        signal, signal_len, tokens, tokens_len = batch[:4]
        tdec, tjoint = m._kl_teacher
        tdec.to(signal.device), tjoint.to(signal.device)
        edec, ejoint = m._kl_en_teacher or (None, None)
        if edec is not None:
            edec.to(signal.device), ejoint.to(signal.device)
        dev = signal.device
        if getattr(m, "_enc_unfrozen", ()):
            # unfreeze_top: the student side takes the training pass's own encoder output (with its graph, so these
            # losses reach the unfrozen blocks); the teacher side = stock blocks on the same encoder input
            feats, feats_len, enc_s, enc_s_len = m._enc_main
            m._enc_main = None
            enc, enc_len = stock_top_encoder(m, feats.detach(), feats_len)
            assert enc.shape == enc_s.shape and torch.equal(enc_len, enc_s_len), (enc.shape, enc_s.shape)
            if not getattr(m, "_enc_checked", False):
                m._enc_checked = True
                print(f"unfreeze_top check (first step): max |student - stock| encoder output "
                      f"{(enc_s.detach().float() - enc.float()).abs().max().item():.3g}", flush=True)
            with torch.no_grad():
                t_g, _, _ = tdec(targets=tokens, target_length=tokens_len)
            f_s = enc_s.transpose(1, 2)
        else:
            with torch.no_grad():
                feats, feats_len = m.preprocessor(input_signal=signal, length=signal_len)
                enc, enc_len = m.encoder(audio_signal=feats, length=feats_len)
                t_g, _, _ = tdec(targets=tokens, target_length=tokens_len)
            f_s = None
        te_g = None
        if edec is not None:  # the English teacher's prediction net on the same labels (it shares `enc`, stock)
            with torch.no_grad():
                te_g, _, _ = edec(targets=tokens, target_length=tokens_len)
        with NoDropout(m.decoder):
            s_g, _, _ = m.decoder(targets=tokens, target_length=tokens_len)
        f = enc.transpose(1, 2)
        if f_s is None:  # frozen encoder: student and teacher read the same features (the default, unchanged)
            f_s = f
        B = f.shape[0]
        row_toks = [tokens[i, :int(tokens_len[i])].tolist() for i in range(B)]
        tags = [row_tags.get(tuple(t), "free") for t in row_toks]
        W = int(tokens_len.max()) + 1
        klu = torch.zeros(B, W, dtype=torch.bool)  # label positions where KL applies
        for i, tg in enumerate(tags):
            klu[i, :len(row_toks[i]) + 1] = torch.tensor(klu_row(spm, tg, row_toks[i], suffix_anchor, anchor_after,
                                                                 nspan_pos, span_prefix, en_cache, en_full,
                                                                 en_free_teacher))
        klu = klu.to(dev)
        is_en = torch.tensor([tg in en_tags for tg in tags], device=dev)
        is_ent = is_en & (edec is not None)  # rows read from the English teacher
        thr = torch.tensor(row_gate(tags, [tuple(t) for t in row_toks], gate, gate_human, synthetic or set(), en_span_gate),
                           device=dev)
        is_nonen = torch.tensor([tg == "nonen" for tg in tags], device=dev)
        is_nspan = torch.tensor([tg == "nonen_span" for tg in tags], device=dev)
        # anti-blend (review #6): English rows saying "twenty <10-99>" with no "thousand" anywhere in the label
        texts = [spm.decode(t).lower() for t in row_toks]
        is_yr = torch.tensor([tg not in ("nonen", "nonen_span") and bool(TWENTY_PAIR.search(tx)) and "thousand" not in tx
                              for tg, tx in zip(tags, texts)], device=dev)
        yrcol = torch.zeros(B, W, dtype=torch.bool)  # label columns right after a "twenty" of a year pair
        for i in range(B):
            if not is_yr[i]:
                continue
            ws = row_words(spm, row_toks[i])
            for k in range(len(ws) - 1):
                if ws[k][0].lower() == "twenty" and TWENTY_PAIR.match("twenty " + ws[k + 1][0].lower()):
                    u = ws[k + 1][1]
                    if row_toks[i][u] != th_first:
                        yrcol[i, u] = True
        yrcol = yrcol.to(dev)
        tid, wf, en, th, cm = term_t.to(dev), wf_t.to(dev), en_t.to(dev), th_t.to(dev), cm_t.to(dev)
        wfm = wfm_t.to(dev)
        wfp = wfp_t.to(dev)
        # comma penalty: columns open to it (not a comma target of the row's own label); unused when comma_weight 0
        cm_ok = (~comma_target_cols(row_toks, cm_t.tolist(), W)).to(dev) if comma_weight else torch.ones_like(klu)

        def chunk(fs_sl, f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl, klu_sl, nonen_sl, nspan_sl, yr_sl, yrcol_sl, thr_sl,
                  cmok_sl, te_sl, ent_sl):
            with NoDropout(m.joint):
                s_logits = m.joint.joint(fs_sl, s_sl).float()
            with torch.no_grad():
                t_logits = routed_teacher_logits(tjoint, ejoint, f_sl, t_sl, te_sl, ent_sl)
            T, U = s_logits.shape[1], s_logits.shape[2]
            mask = ((torch.arange(T, device=dev)[None, :, None] < enc_len_sl[:, None, None])
                    & (torch.arange(U, device=dev)[None, None, :] <= tok_len_sl[:, None, None]))
            kmask = mask & klu_sl[:, None, :U]
            s_all, t_all = s_logits[..., :nb].logsumexp(-1), t_logits[..., :nb].logsumexp(-1)
            s_nbz, t_nbz = s_logits[..., :nb - 1].logsumexp(-1), t_logits[..., :nb - 1].logsumexp(-1)
            s_lb, t_lb = s_logits[..., nb - 1] - s_all, t_logits[..., nb - 1] - t_all
            s_lnb, t_lnb = s_nbz - s_all, t_nbz - t_all
            kl_blank = t_lb.exp() * (t_lb - s_lb) + t_lnb.exp() * (t_lnb - s_lnb)
            bmask = blank_gate(kmask, t_lnb, thr_sl)  # gnorm (0.9; R4 arm b: 0.5 on human-verified rows)
            s_lt, t_lt = s_logits[..., tid].logsumexp(-1) - s_nbz, t_logits[..., tid].logsumexp(-1) - t_nbz
            s_lo = torch.log((1 - s_lt.exp()).clamp(min=1e-7))
            t_lo = torch.log((1 - t_lt.exp()).clamp(min=1e-7))
            kl_p = t_lt.exp() * (t_lt - s_lt) + t_lo.exp() * (t_lo - s_lo)
            # TDT duration head (the outputs after the 8193 token+blank logits): KL to stock where KL applies
            s_ld, t_ld = s_logits[..., nb:].log_softmax(-1), t_logits[..., nb:].log_softmax(-1)
            kl_d = (t_ld.exp() * (t_ld - s_ld)).sum(-1)
            dmask = kmask & (t_lnb.exp() > emit_p)
            # token KL (non-English rows): over the non-blank distribution at the teacher's emitting cells; averaged over
            # all T x U cells it was ~95% blank and inert (retrain-2 diagnosis; both retrain-3 reviews)
            # nonen_span rows: token KL on the prefix only (klu), never on or after the spoken number
            tok_rows = nonen_sl[:, None, None] | (nspan_sl[:, None, None] & klu_sl[:, None, :U])
            tmask = mask & tok_rows & (t_lnb.exp() > emit_p)
            # on nonen_span rows the prefix columns still span every frame, including the number's audio,
            # where stock writes digits; drop cells where the teacher's written-form mass (given non-blank) is > 0.1
            # English token KL (en_teacher only): the English teacher's rows at their klu positions, own term (kl_te)
            temask = mask & ent_sl[:, None, None] & klu_sl[:, None, :U] & (t_lnb.exp() > emit_p)
            if bool(nspan_sl.any()) or bool(ent_sl.any()):
                # the English teacher's digit cells drop out the same way (where it still writes digits)
                t_wf = (t_logits[..., wfm].logsumexp(-1) - t_nbz).exp()
                wf_cell = (nspan_sl | ent_sl)[:, None, None] & (t_wf > 0.1)
                if kl_mask_space_en:  # nonen_span rows: the written-form pieces only (no '▁'), as without the mask
                    t_wf0 = (t_logits[..., wf].logsumexp(-1) - t_nbz).exp()
                    wf_cell = (nspan_sl[:, None, None] & (t_wf0 > 0.1)) | (ent_sl[:, None, None] & (t_wf > 0.1))
                tmask = tmask & ~wf_cell
                bmask = bmask & ~wf_cell
                kmask = kmask & ~wf_cell
                dmask = dmask & ~wf_cell
                temask = temask & ~wf_cell
            if bool((nonen_sl | nspan_sl | ent_sl).any()):
                s_lp = s_logits[..., :nb - 1] - s_nbz[..., None]
                t_lp = t_logits[..., :nb - 1] - t_nbz[..., None]
                kl_t = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
                del s_lp, t_lp
            else:
                kl_t = torch.zeros_like(s_nbz)
            emit = mask & (s_lnb.detach().exp() > emit_p)

            def pen(ids, rows_sl, cols=None):
                p = 0.99 * (s_logits[..., ids].logsumexp(-1) - s_nbz).exp()  # bounded (-log <= 4.6) and still graded at p -> 1
                e = emit & rows_sl[:, None, None]
                if cols is not None:
                    e = e & cols[:, None, :e.shape[2]]
                per_row = (-torch.log1p(-p) * e).sum((1, 2)) / e.sum((1, 2)).clamp(min=1)
                return per_row, rows_sl.float()  # per row: 0 off rows_sl (no emitting cell counted there)

            wf_s, wf_n = pen(wfp, ~nonen_sl)
            if wf_en_weight != 1.0:  # off: wf_s untouched, bit for bit
                wf_s = wf_s * wf_row_weights(nspan_sl, wf_en_weight)
            en_s, en_n = (pen(en, nonen_sl | nspan_sl, en_pen_cols(klu_sl, nonen_sl, nspan_sl, en_set)) if len(en)
                          else (torch.zeros_like(nonen_sl, dtype=torch.float),) * 2)
            th_s, th_n = pen(th, yr_sl, yrcol_sl)
            # decimal comma: nonen_span rows, number columns only (klu False), not where the label itself has a comma
            cm_s, cm_n = (pen(cm, nspan_sl, nspan_sl[:, None] & ~klu_sl & cmok_sl) if comma_weight
                          else (torch.zeros_like(nonen_sl, dtype=torch.float),) * 2)
            # per-row sums (summed over the batch below; per tag for the r3tag log line)
            return ((kl_blank * bmask).sum((1, 2)), bmask.sum((1, 2)).float(), (kl_p * kmask).sum((1, 2)),
                    kmask.sum((1, 2)).float(), (kl_t * tmask).sum((1, 2)), tmask.sum((1, 2)).float(), wf_s, wf_n, en_s,
                    en_n, th_s, th_n, (kl_d * dmask).sum((1, 2)), dmask.sum((1, 2)).float(), cm_s, cm_n,
                    (kl_t * temask).sum((1, 2)), temask.sum((1, 2)).float())

        parts = []
        for j in range(0, B, sub_batch):
            sl = torch.arange(j, min(B, j + sub_batch), device=dev)
            u = int(tokens_len[sl].max()) + 1
            t = int(enc_len[sl].max())
            parts.append(torch.utils.checkpoint.checkpoint(
                chunk, f_s[sl, :t], f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                enc_len[sl], tokens_len[sl], klu[sl], is_nonen[sl], is_nspan[sl], is_yr[sl], yrcol[sl], thr[sl],
                cm_ok[sl], None if te_g is None else te_g[sl, :, :u].transpose(1, 2), is_ent[sl], use_reentrant=False))
        acc = [torch.cat(v) for v in zip(*parts)]  # 18 per-row vectors (B,)
        if en_kl_scale != 1.0:  # numerators only: the KL averages keep their cell counts (block comment)
            w_en = torch.where(is_en, torch.full((), float(en_kl_scale), device=dev), torch.ones((), device=dev))
            for k in (0, 2, 12, 16):  # blank, punct, duration, English token KL
                acc[k] = acc[k] * w_en
        tot = [a.sum() for a in acc]
        kl_b = tot[0] / tot[1].clamp(min=1)
        kl_p = tot[2] / tot[3].clamp(min=1)
        kl_t = tot[4] / tot[5].clamp(min=1)
        wf_pen = tot[6] / tot[7].clamp(min=1)
        en_pen = tot[8] / tot[9].clamp(min=1)
        th_pen = tot[10] / tot[11].clamp(min=1)
        kl_d = tot[12] / tot[13].clamp(min=1)
        cm_pen = tot[14] / tot[15].clamp(min=1)
        aux = (kl_weight * kl_b + punct_weight * kl_p + tok_weight * kl_t + wf_weight * wf_pen + en_weight * en_pen
               + wf_weight * th_pen + dur_weight * kl_d)
        if comma_weight:  # off: the sum above, unchanged bit for bit
            aux = aux + comma_weight * cm_pen
        kl_te = tot[16] / tot[17].clamp(min=1)
        if edec is not None:  # off: no English token KL, unchanged bit for bit
            aux = aux + tok_weight * kl_te
        cons, n_pairs = torch.zeros((), device=dev), 0
        if cons_weight and pair_bank is not None:
            cons, n_pairs = consistency_term(m, pair_bank, dev, nb)
            aux = aux + cons_weight * cons
        m._r3_parts = {"rnnt": out["loss"].detach(), "kl_b": kl_b.detach(), "kl_p": kl_p.detach(),
                       "kl_t": kl_t.detach(), "wf": wf_pen.detach(), "en": en_pen.detach(),
                       "th": th_pen.detach(), "kl_d": kl_d.detach(), "cons": cons.detach(),
                       "comma": cm_pen.detach()}
        if edec is not None:
            m._r3_parts["kl_te"] = kl_te.detach()
        if os.environ.get("R3_GRADSHARE") and batch_idx % 50 == 0:
            # acceptance check: each term's gradient on the joint's last layer, overall and on the written-form
            # output rows (wf_pen must reach >= 10% of the RNNT gradient there)
            W = m.joint.joint_net[-1].weight
            wf_rows = wf.clamp(max=W.shape[0] - 1)
            gs = {}
            # rnnt LAST: NeMo's TDT loss writes its gradient in place, so it cannot be backpropagated twice; on these
            # measurement batches (E2 only, R3_GRADSHARE) the RNNT term is detached after being measured
            for k, term in (("wf", wf_weight * wf_pen), ("th", wf_weight * th_pen), ("kl_t", tok_weight * kl_t),
                            ("kl_b", kl_weight * kl_b), ("cons", cons_weight * cons), ("rnnt", out["loss"])):
                torch.cuda.empty_cache()
                if not term.requires_grad:
                    gs[k] = (0.0, 0.0)
                    continue
                g = torch.autograd.grad(term, W, retain_graph=True, allow_unused=True)[0]
                gs[k] = (0.0, 0.0) if g is None else (g.norm().item(), g[wf_rows].norm().item())
            print("r3grad " + " ".join(f"{k} {a:.4g}/{b:.4g}" for k, (a, b) in gs.items())
                  + f" wf_share_at_wf_rows {gs['wf'][1] / max(gs['rnnt'][1], 1e-12):.3f}", flush=True)
            out["loss"] = out["loss"].detach()
        out["loss"] = out["loss"] + aux
        if batch_idx % 10 == 0:
            print("r3 " + " ".join(f"{k} {v.item():.5f}" for k, v in m._r3_parts.items())
                  + f" pairs {n_pairs} tags {dict(collections.Counter(tags))}", flush=True)
            # per tag (rehearsal log): value (cells or rows it averages over)
            for tg, d in per_tag_summary(tags, acc).items():
                print(f"r3tag {tg} rows {d['rows']} " + " ".join(f"{k} {d[k]:.5f} ({d[k + '_n']})"
                                                               for k in ("kl_b", "kl_p", "kl_t", "wf", "en")
                                                               + (("kl_te",) if edec is not None else ())), flush=True)
        return out

    m.training_step = training_step


class L2SP(pl.Callback):
    """Decoupled pull toward v10pt after every optimizer step: p <- p - lr * c * (p - p_v10pt). The anchor is loaded
    from the v10pt FILE (also on resume: init_from is skipped then). The written-form output rows of the joint's last
    layer are exempt (N2: they must stay free to keep non-English digits at stock strength)."""

    def __init__(self, anchor_path: str, c: float, exempt_rows: list[int], extra_anchor: dict | None = None,
                 enc_c: float | None = None):
        """extra_anchor: more anchor tensors by parameter name (unfreeze_top: the stock encoder blocks), for trainable
        parameters the anchor file does not hold. enc_c: the pull strength c of the encoder parameters ("encoder."
        names); None = the same c as everything else (unchanged). Each parameter is pulled with the learning rate of
        its own optimizer parameter group (one group: every parameter has the group's lr, as before)."""
        self.path, self.c, self.exempt, self.extra = anchor_path, c, exempt_rows, extra_anchor
        self.enc_c = c if enc_c is None else enc_c
        self.anchor = None

    def on_train_batch_end(self, trainer, m, outputs, batch, batch_idx):
        if not (self.c or self.enc_c) or (batch_idx + 1) % trainer.accumulate_grad_batches:
            return  # once per OPTIMIZER step (gradient accumulation: several batches per step)
        if self.anchor is None:
            sd = torch.load(self.path, map_location="cpu")
            self.anchor = {n: sd[n].to(p.device, p.dtype) for n, p in m.named_parameters() if p.requires_grad and n in sd}
            print(f"L2SP c={self.c}: {len(self.anchor)} tensors anchored to {self.path}", flush=True)
            if self.extra:
                more = {n: self.extra[n].to(p.device, p.dtype) for n, p in m.named_parameters()
                        if p.requires_grad and n in self.extra and n not in self.anchor}
                self.anchor.update(more)
                free = [n for n, p in m.named_parameters() if p.requires_grad and n not in self.anchor]
                print(f"L2SP c={self.c}: + {len(more)} encoder tensors anchored to stock "
                      f"({sum(v.numel() for v in more.values()):,} params); trainable tensors without an anchor: "
                      f"{len(free)} {free[:4]}", flush=True)
            if self.enc_c != self.c:
                print(f"L2SP per-group: encoder tensors c={self.enc_c}, all others c={self.c}", flush=True)
        groups = trainer.optimizers[0].param_groups
        lr0 = groups[0]["lr"]
        lr_of = {id(p): g["lr"] for g in groups for p in g.get("params", ())}  # each parameter: its own group's lr
        last = "joint.joint_net." + str(len(m.joint.joint_net) - 1)
        with torch.no_grad():
            drift = 0.0
            drift_enc = 0.0
            for n, p in m.named_parameters():
                a = self.anchor.get(n)
                if a is None:
                    continue
                d = p.data - a
                if n.startswith(last) and self.exempt:
                    d[self.exempt] = 0
                is_enc = n.startswith("encoder.")
                c = self.enc_c if is_enc else self.c
                if c:
                    p.data.sub_(d, alpha=lr_of.get(id(p), lr0) * c)
                sq = float((d.float() ** 2).sum())
                drift += sq
                drift_enc += sq if is_enc else 0.0
        if (batch_idx + 1) % (50 * trainer.accumulate_grad_batches) == 0:
            print(f"l2sp drift {drift ** 0.5:.4f}" + (f" (encoder {drift_enc ** 0.5:.4f})" if drift_enc else ""), flush=True)


MERGED_KL = True  # one joint pass for all KL terms (2026-10-13 speed fix)
KL_EQ_CHECK = os.environ.get("KL_EQ_CHECK") == "1"  # first 3 batches: also run the separate passes and print both


class SeenRows(pl.Callback):
    """Measured exposure: appends the rows every optimizer step's micro-batches actually contained to
    `<exp>/seen_rows.jsonl` (exposure.py has the format and the reader, exposure_report.py the report). Purely additive:
    no RNG, no model or batch change, and a failing write only prints a warning."""

    def __init__(self, exp: str, sha12: str):
        self.exp, self.sha12, self.log = exp, sha12, None

    def _guard(self, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # never let the bookkeeping stop a run
            print(f"SEEN ROWS LOG FAILED ({type(e).__name__}: {e}); exposure for later steps is not recorded", flush=True)

    def on_train_start(self, trainer, m):
        self.log = self._guard(exposure.SeenLog, self.exp, self.sha12, trainer.global_step)

    def on_train_batch_start(self, trainer, m, batch, idx):
        if self.log is not None:
            self._guard(self.log.add, trainer.global_step, batch)

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        if self.log is not None:
            self._guard(self.log.end_batch, trainer.global_step)

    def on_train_end(self, trainer, m):
        if self.log is not None:
            self._guard(self.log.flush)

    def on_exception(self, trainer, m, exception):
        self.on_train_end(trainer, m)


class StepLog(pl.Callback):
    def __init__(self, mem: bool = False):
        self.t0 = time.time()
        self.clipped = 0
        self.mem = mem  # also print the peak GPU memory every 10 steps (unfreeze_top runs)

    def on_before_optimizer_step(self, trainer, m, optimizer):
        # Gradient norm before clipping (clip 1.0): a large KL term can scale the whole update down.
        norm = torch.linalg.vector_norm(torch.stack(
            [p.grad.detach().float().norm() for p in m.parameters() if p.grad is not None]))
        self.clipped += int(norm > 1.0)
        if trainer.global_step % 10 == 0:
            print(f"grad_norm {norm.item():.3f} clipped {self.clipped}/{trainer.global_step + 1}", flush=True)
            if self.mem:  # the unfrozen encoder blocks' share of it
                enc = [p.grad.detach().float().norm() for n, p in m.named_parameters()
                       if n.startswith("encoder.") and p.grad is not None]
                enc_norm = torch.linalg.vector_norm(torch.stack(enc)).item() if enc else 0.0
                print(f"grad_norm encoder {enc_norm:.3f} ({len(enc)} tensors with a gradient)", flush=True)

    def on_train_batch_end(self, trainer, m, outputs, batch, idx):
        if trainer.global_step % 10 == 0:
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
            print(f"step {trainer.global_step} loss {loss:.4f} {time.time() - self.t0:.0f}s", flush=True)
            if self.mem and torch.cuda.is_available():
                print(f"gpu_mem peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)


def write_mix(vol: str, out: str, parts: list[tuple], seed: int = 1, replay_texts=None,
              kl_exempt: str = "", tagged: dict | None = None, stage_dir: str = "", meta_out: list | None = None) -> float:
    """Concatenate manifests, taking `hours` from each (all if hours <= 0). Labels of rows from
    `replay/` manifests are added to `replay_texts` when given. `meta_out` (a list) receives one exposure.row_meta
    (part, tag, language, number kinds) per written row, in manifest order; it changes nothing else."""
    r = random.Random(seed)
    rows, exempt = [], set()
    part_of = {}
    for part in parts:
        path, hours = part[0], part[1]
        tag = part[2] if len(part) > 2 else None
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
        if tag is not None and tagged is not None:
            # explicit KL treatment: "anchor" | "anchor_num" | "span" | "nonen" | "nonen_span" | "free"
            if tag != "free":
                tagged.setdefault(tag, set()).update(x["text"] for x in src)
            if tag == "nonen_span":
                pre = tagged.setdefault("_nspan_prefix", {})
                nw = tagged.setdefault("_nspan_words", {})  # R4 arm b: per-word number flags (suffix anchoring)
                syn = tagged.setdefault("_synthetic", set())  # R4 arm b: TTS rows / TTS windows keep the 0.9 gate
                for x in src:
                    pre[x["text"]] = nspan_prefix(x)
                    try:
                        nw[x["text"]] = word_flags(x)
                    except ValueError as e:  # num_tokens / label length mismatch: this row keeps the prefix rule
                        print(f"R4 word flags skipped: {e}")
                    if (x.get("synthetic") or str(x.get("origin", "")).startswith("tts")
                            or re.match(r"mix4\w*/tts_", path)):  # a candidate mix dir (mix4final/...) too
                        syn.add(x["text"])
            if tag in ("anchor", "anchor_num"):
                # rows that say a number (numberish words, Roman numerals) are KL-exempt (v10pt's numwords rule): the
                # stock teacher forced on spoken numbers derails (2,666 such rows were full-anchored)
                from common import NUMBER_WORDS

                def says_number(t):  # NUMBER_WORDS (not numberish: "one"/"second" are mostly ordinary words) + Roman
                    return NUMBER_WORDS.search(t) is not None or re.search(r"\b(?:II|III|IV|VI|VII|VIII|IX|XI|XII)\b|"
                                                                           r"\bWorld War I\b", t) is not None

                num = {x["text"] for x in src if says_number(x["text"])}
                plain = {x["text"] for x in src} - num
                tagged.setdefault("anchor_num", set()).update(num)
                tagged.setdefault("anchor", set()).update(plain)
                tagged[tag] -= (plain if tag == "anchor_num" else num)
        elif replay_texts is not None and path.startswith("replay/"):
            if kl_exempt and kl_exempt in path:
                exempt.update(x["text"] for x in src)
            else:
                replay_texts.update(x["text"] for x in src)
        rows += src
        if meta_out is not None:
            part_of.update({id(x): (path, tag) for x in src})
    if replay_texts is not None and kl_exempt == "numwords":
        # Real speech that says numbers is where spoken form is learned on real audio: no anchor there,
        # whichever manifest the row came from (LSPC also holds number-word rows).
        from common import NUMBER_WORDS
        exempt |= {t for t in replay_texts if NUMBER_WORDS.search(t)}
    if replay_texts is not None and exempt:
        replay_texts.difference_update(exempt)
        print(f"KL-exempt labels ({kl_exempt}): {len(exempt)}, anchored: {len(replay_texts)}")
    r.shuffle(rows)
    if meta_out is not None:
        meta_out.extend(exposure.row_meta(x, *part_of[id(x)]) for x in rows)
    if stage_dir:
        # Copy the selected audio to the container's local disk: ~60k small files read from the network volume made
        # each step ~3x slower than v12y's (dress rehearsal 2026-10-13).
        from common import stage_files

        t0 = time.time()
        rows = stage_files(rows, stage_dir)
        print(f"staged {len(rows)} files to {stage_dir} in {time.time() - t0:.0f}s", flush=True)
    with open(out, "w") as f:
        for x in rows:
            f.write(json.dumps({k: x[k] for k in ("audio_filepath", "duration", "text")}) + "\n")
    return sum(x["duration"] for x in rows) / 3600


def trainable_state(m) -> dict:
    """Decoder + joint, plus the encoder blocks unfreeze_top_layers made trainable (parameters and buffers). With LoRA
    adapters (add_lora) the adapters are MERGED into the base weights and no adapter tensor is written: the checkpoint
    has the same keys and shapes as a full top-block fine-tune, so every reader and the int8 patch path are unchanged."""
    keep = getattr(m, "_enc_unfrozen", ())
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()
          if (not (k.startswith("encoder.") or k.startswith("preprocessor.")) or k.startswith(keep))
          and not (getattr(m, "_lora", None) and ".lora_" in k)}
    if getattr(m, "_lora", None):
        sd.update(lora_merged_weights(m))
    return sd


def unfreeze_top_layers(m, n: int, train_base: bool = True) -> dict:
    """Make the last `n` conformer blocks of the frozen encoder trainable: every parameter under encoder.layers.<L-n> ..
    encoder.layers.<L-1> (both feed-forward modules, self-attention, the convolution module with its BatchNorm affine,
    and the block's layer norms). pre_encode (subsampling), the positional encoding and blocks 0 .. L-n-1 stay frozen.
    The whole encoder stays in eval mode (KeepEncoderFrozen): dropout off and BatchNorm on its stock running statistics,
    which are not re-estimated (batches skewed to the trained languages would shift them for every language); the
    student and the KL teacher then see the same deterministic blocks.
    Side effects on `m`: `_enc_unfrozen` (the blocks' state-dict prefixes, kept by trainable_state), `_enc_stock_top`
    (their stock tensors, an attribute so neither saved nor trained: the KL teacher's top blocks in add_r3_losses), and
    a forward hook that keeps the training pass's encoder input and output (`_enc_main`) for add_r3_losses.
    train_base False (LoRA, add_lora): the blocks' own parameters stay frozen; everything else here is unchanged (the
    teacher, the hook, the saved prefixes) and the adapters are the only trainable encoder tensors.
    Returns the stock tensors ("encoder.layers.<i>.<name>" -> CPU clone, parameters and buffers): the L2-SP anchor."""
    layers = m.encoder.layers
    n_layers = len(layers)
    assert 0 < n <= n_layers, f"unfreeze_top {n}: the encoder has {n_layers} blocks"
    idx = list(range(n_layers - n, n_layers))
    prefixes = tuple(f"encoder.layers.{i}." for i in idx)
    for i in idx:
        for p in layers[i].parameters():
            p.requires_grad_(train_base)
    stock = {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.startswith(prefixes)}
    m._enc_unfrozen = prefixes
    m._enc_stock_top = stock
    m._enc_main = None

    def keep_main(mod, args, kwargs, out):
        if torch.is_grad_enabled():  # the training pass only (the teacher and consistency passes run under no_grad)
            keep = getattr(m, "_enc_row_keep", None)
            if keep is not None:  # enc_real_only: same values, but no gradient reaches the encoder from the other rows
                out = (torch.where(keep.to(out[0].device)[:, None, None], out[0], out[0].detach()),) + tuple(out[1:])
            m._enc_main = (kwargs["audio_signal"], kwargs["length"], out[0], out[1])
            if keep is not None:
                return out

    m.encoder.register_forward_hook(keep_main, with_kwargs=True)
    n_par = sum(p.numel() for i in idx for p in layers[i].parameters())
    n_ten = sum(1 for i in idx for _ in layers[i].parameters())
    subs = sorted({k[len(f"encoder.layers.{idx[0]}."):].split(".")[0] for k in stock if k.startswith(prefixes[0])})
    print(f"unfreeze_top {n}: encoder.layers.{idx[0]}..{idx[-1]} of {n_layers} {'trainable' if train_base else 'FROZEN (LoRA)'}"
          f", {n_par:,} params in {n_ten} "
          f"tensors (per block: {', '.join(subs)}); pre_encode + layers 0..{idx[0] - 1} frozen; encoder kept in eval mode "
          f"(dropout off, BatchNorm running stats fixed)", flush=True)
    return stock


# --- LoRA adapters on the top encoder blocks (S-encC) -------------------------------------------------------------
# The adapter is a forward hook on the target nn.Linear (out = W x + b + scale * B(A x)); the base weight keeps its name,
# shape and value, so the teacher's functional_call over the stock tensors and the checkpoint keys do not change. The
# adapters live on the Linear as parameters `lora_A` [r, in] and `lora_B` [out, r] (B = 0 at the start: the student
# equals stock). They are merged into `weight` when a checkpoint is written (trainable_state) and never saved as such.
LORA_TARGET = re.compile(r"^(self_attn\.linear_(q|k|v|out)|feed_forward[12]\.linear[12])$")


def add_lora(m, n_top: int, rank: int, alpha: float) -> dict:
    """LoRA of rank `rank` on the attention q/k/v/out and both feed-forward linears (not linear_pos, not the convolution
    module) of the last `n_top` conformer blocks. Returns the bookkeeping stored on `m._lora`."""
    assert rank > 0 and n_top > 0
    layers = m.encoder.layers
    n_layers = len(layers)
    state = {"on": True}
    targets = []
    for i in range(n_layers - n_top, n_layers):
        for name, mod in layers[i].named_modules():
            if LORA_TARGET.match(name) and isinstance(mod, torch.nn.Linear):
                targets.append((f"encoder.layers.{i}.{name}", mod))
    assert len(targets) == 8 * n_top, f"LoRA: {len(targets)} target linears in the top {n_top} blocks (expected {8 * n_top})"
    scale = alpha / rank
    n_par = 0
    for full, mod in targets:
        mod.register_parameter("lora_A", torch.nn.Parameter(torch.empty(rank, mod.in_features)))
        mod.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(mod.out_features, rank)))
        torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)  # the nn.Linear / PEFT init; B = 0
        n_par += mod.lora_A.numel() + mod.lora_B.numel()

        def hook(mod, args, out, state=state, scale=scale):
            if not state["on"]:
                return None
            x = args[0]
            return out + torch.nn.functional.linear(torch.nn.functional.linear(x, mod.lora_A), mod.lora_B) * scale

        mod.register_forward_hook(hook)
    m._lora = {"state": state, "targets": targets, "scale": scale, "rank": rank}
    print(f"LoRA rank {rank} alpha {alpha} (scale {scale}): {len(targets)} linears in encoder.layers."
          f"{n_layers - n_top}..{n_layers - 1} (attention q/k/v/out + both feed-forwards), {n_par:,} adapter params "
          f"(base weights frozen, B = 0 at the start); merged into the weights at every checkpoint", flush=True)
    return m._lora


class lora_disabled:
    """Context manager: the adapters contribute nothing inside it (the stock-encoder teacher pass)."""

    def __init__(self, m):
        self.state = getattr(m, "_lora", None) and m._lora["state"]

    def __enter__(self):
        if self.state:
            self.prev, self.state["on"] = self.state["on"], False

    def __exit__(self, *exc):
        if self.state:
            self.state["on"] = self.prev


def lora_merged_weights(m) -> dict:
    """{"encoder.layers.<i>.<name>.weight": W + scale * B @ A} (fp32, CPU) for every adapted linear."""
    out = {}
    scale = m._lora["scale"]
    for full, mod in m._lora["targets"]:
        w = mod.weight.detach().float().cpu()
        out[f"{full}.weight"] = w + scale * (mod.lora_B.detach().float().cpu() @ mod.lora_A.detach().float().cpu())
    return out


def encoder_param_groups(m, lr: float, enc_lr_scale: float, peak: float | None = None) -> list[dict]:
    """Two optimizer parameter groups: the trainable encoder tensors at lr * enc_lr_scale, everything else (decoder, joint,
    the frozen encoder tensors, which never get a gradient) at lr. NeMo's scheduler gives every group its own base lr; its
    min_lr (lr / 10) is shared, so the encoder group decays from lr * enc_lr_scale to that same floor.
    peak (enc_lr_peak): the encoder group's absolute starting lr; its schedule is then LrSchedule's, not NeMo's (NeMo only
    needs the starting lr to be >= its min_lr, lr / 10)."""
    enc = [p for n, p in m.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    ids = {id(p) for p in enc}
    assert enc, "enc_lr_scale needs trainable encoder tensors (unfreeze_top)"
    if peak is None:
        assert enc_lr_scale >= 0.1, "enc_lr_scale < 0.1: the group's peak would sit below the shared cosine floor (lr / 10)"
        enc_lr = lr * enc_lr_scale
    else:
        assert enc_lr_scale == 1.0 and peak >= lr / 10, f"enc_lr_peak {peak}: not with enc_lr_scale, and >= lr / 10 (NeMo's floor)"
        enc_lr = peak
    return [{"params": [p for p in m.parameters() if id(p) not in ids], "lr": lr},
            {"params": enc, "lr": enc_lr}]


class ParamGroupLog(pl.Callback):
    """Print the optimizer's real parameter groups at the start of training (the optimizer is rebuilt by Lightning).
    Counts the trainable tensors only (group 0 also holds the frozen ones, which never get a gradient)."""

    def on_train_start(self, trainer, m):
        enc_ids = {id(p) for n, p in m.named_parameters() if n.startswith("encoder.")}
        for i, g in enumerate(trainer.optimizers[0].param_groups):
            ps = [p for p in g["params"] if p.requires_grad]
            print(f"param group {i}: {len(ps)} trainable tensors ({sum(p.numel() for p in ps):,} params, "
                  f"{sum(1 for p in ps if id(p) in enc_ids)} encoder tensors) peak lr {g.get('initial_lr', g['lr']):.3g}",
                  flush=True)


def cosine_lr_at(step: int, peak: float, floor: float, warmup: int, total: int) -> float:
    """NeMo's CosineAnnealing (warmup_steps=warmup, max_steps=total, min_lr=floor) at scheduler step `step`, without NeMo:
    linear warmup peak * (step + 1) / (warmup + 1) for step <= warmup, then the cosine from peak to floor over
    total - warmup steps, floor after total."""
    if warmup > 0 and step <= warmup:
        return peak * (step + 1) / (warmup + 1)
    if step > total:
        return floor
    return (peak - floor) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) + floor


def decoder_lr_at(k: int, lr: float, warmup: int, total: int, offset: int) -> float:
    """Decoder + joint lr of optimizer step k of a run that continues a `total`-step schedule from step `offset` (a branch
    off the main run's checkpoint): the main run's cosine at offset + k, so the tail goes on where the main run was, times a
    restart ramp (k + 1) / (warmup + 1) over the first `warmup` steps (the optimizer's moments start empty). offset 0 =
    the plain schedule (NeMo's own warmup), no extra ramp."""
    if offset == 0:
        return cosine_lr_at(k, lr, lr / 10, warmup, total)
    ramp = (k + 1) / (warmup + 1) if k <= warmup else 1.0
    return cosine_lr_at(offset + k, lr, lr / 10, warmup, total) * ramp


class LrSchedule(pl.Callback):
    """Set the learning rate of chosen optimizer parameter groups from an absolute schedule f(k) just before every
    optimizer step (k = the step about to be taken, from 0; trainer.global_step also after a resume). It overrides what
    NeMo's scheduler wrote, so a group follows its own peak, warmup and floor instead of NeMo's shared cosine floor. The
    L2-SP pull (on_train_batch_end, before the scheduler steps) reads the same lr."""

    def __init__(self, fns: dict):
        self.fns = fns  # {param group index: step -> lr}
        self.last = {}

    def on_before_optimizer_step(self, trainer, m, optimizer):
        k = trainer.global_step
        groups = trainer.optimizers[0].param_groups
        for i, fn in self.fns.items():
            groups[i]["lr"] = self.last[i] = fn(k)
        if k % 50 == 0:
            print(f"lr step {k}: " + " ".join(f"group{i} {v:.4g}" for i, v in sorted(self.last.items())), flush=True)


class GroupClip(pl.Callback):
    """Gradient clipping per group instead of Lightning's one global norm (set gradient_clip_val None when this runs):
    the decoder + joint tensors to `dec_clip` (1.0: exactly what the decoder-only control does), the encoder tensors to
    their own fixed `enc_clip`. Runs before the optimizer step, after StepLog has read the global norm; the pre-clip norm of
    each group is printed every 10 steps with how often it clipped."""

    def __init__(self, enc_clip: float, dec_clip: float = 1.0):
        self.enc_clip, self.dec_clip = enc_clip, dec_clip
        self.clipped = {"dec": 0, "enc": 0}
        self.last = {}

    def on_before_optimizer_step(self, trainer, m, optimizer):
        parts = {"dec": ([], self.dec_clip), "enc": ([], self.enc_clip)}
        for n, p in m.named_parameters():
            if p.grad is not None:
                parts["enc" if n.startswith("encoder.") else "dec"][0].append(p)
        for g, (ps, clip) in parts.items():
            if ps:
                norm = float(torch.nn.utils.clip_grad_norm_(ps, clip))  # the norm before clipping
                self.last[g] = norm
                self.clipped[g] += int(norm > clip)
        k = trainer.global_step
        if k % 10 == 0:
            print("grad_norm pre-clip " + " ".join(f"{g} {v:.3f} (clip {parts[g][1]:g}, clipped {self.clipped[g]}/{k + 1})"
                                                       for g, v in self.last.items()), flush=True)


def branch_trainer_parts(lr: float, warmup: int, lr_offset: int = 0, lr_total: int = 0, enc_lr_peak: float = 0.0,
                         enc_lr_floor: float = 0.0, enc_lr_warmup: int = 0, enc_lr_steps: int = 0, enc_clip: float = 0.0,
                         max_steps: int = 0, has_encoder: bool = False):
    """(callbacks, gradient_clip_val) for the warm-start / encoder-schedule options. Every option off (the defaults) =
    ([], 1.0): the Trainer is what it always was. lr_total > 0: the decoder + joint group (index 0) follows decoder_lr_at
    (offset lr_offset of a lr_total-step schedule). enc_lr_peak > 0: the encoder group (index 1, a separate group:
    encoder_param_groups) follows its own schedule, cosine peak -> enc_lr_floor over enc_lr_steps (default max_steps) after
    enc_lr_warmup warmup steps. enc_clip > 0: GroupClip replaces the global clip."""
    cbs, fns = [], {}
    if lr_total:
        assert lr_offset == 0 or lr_offset > warmup, f"lr_offset {lr_offset} must be 0 or past the warmup {warmup}"
        assert lr_offset + max_steps <= lr_total, f"lr_offset {lr_offset} + max_steps {max_steps} > lr_total {lr_total}"
        fns[0] = lambda k: decoder_lr_at(k, lr, warmup, lr_total, lr_offset)
    else:
        assert not lr_offset, "lr_offset needs lr_total"
    if enc_lr_peak:
        assert has_encoder and 0 < enc_lr_floor < enc_lr_peak, "enc_lr_peak needs trainable encoder blocks and 0 < floor < peak"
        steps = enc_lr_steps or max_steps
        assert steps == max_steps, f"enc_lr_steps {steps} != max_steps {max_steps}: the encoder schedule must span the run"
        fns[1] = lambda k: cosine_lr_at(k, enc_lr_peak, enc_lr_floor, enc_lr_warmup, steps)
    else:
        assert not (enc_lr_floor or enc_lr_warmup or enc_lr_steps), "enc_lr_floor / warmup / steps need enc_lr_peak"
    if fns:
        cbs.append(LrSchedule(fns))
    if enc_clip:
        assert has_encoder, "enc_clip needs trainable encoder blocks (unfreeze_top)"
        cbs.append(GroupClip(enc_clip))
    return cbs, (None if enc_clip else 1.0)


def resolve_sampler_seed(seed: int, explicit: int = 0, resuming: bool = False, step0: int = 0) -> int:
    """The lhotse sampler seed: `seed` (default) or the explicit one (a branch off another run's checkpoint draws its own
    batch order; the mix, written with `seed`, stays the same); a resume offsets it by 1000 * the step it resumes from."""
    base = explicit or seed
    return base + 1000 * step0 if resuming else base


def encoder_l2_delta(m) -> float | None:
    """sqrt(sum of squared differences) between the checkpoint's top encoder tensors (trainable_state, LoRA merged) and
    their stock values; None when no encoder block is unfrozen. Buffers (BatchNorm statistics) stay stock: zero."""
    stock = getattr(m, "_enc_stock_top", None)
    if not stock:
        return None
    sd = trainable_state(m)
    return math.sqrt(sum(float(((sd[k].float() - v.float()) ** 2).sum()) for k, v in stock.items() if k in sd))


def log_encoder_delta(m, exp: str, step: int):
    """At a save: print the encoder's L2 distance from stock and append it to <exp>/enc_l2_delta.jsonl."""
    d = encoder_l2_delta(m)
    if d is None:
        return None
    print(f"encoder L2 delta from stock at step {step}: {d:.4f}", flush=True)
    with open(f"{exp}/enc_l2_delta.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "enc_l2_delta": d}) + "\n")
    return d


def stock_top_encoder(m, feats, feats_len):
    """The encoder output with the stock values of the unfrozen top blocks (the KL teacher's encoder), no gradient."""
    dev = feats.device
    cache = getattr(m, "_enc_stock_dev", None)
    if cache is None or next(iter(cache.values())).device != dev:
        cache = {k[len("encoder."):]: v.to(dev) for k, v in m._enc_stock_top.items()}
        m._enc_stock_dev = cache
    with torch.no_grad(), lora_disabled(m):  # the stock values: no adapter contribution either
        return torch.func.functional_call(m.encoder, cache, args=(), kwargs=dict(audio_signal=feats, length=feats_len))


def train(vol: str, ultra: str, run: str, mix: list[tuple[str, float]], max_steps: int,
          lr: float = 1e-4, batch_duration: float = 600, warmup: int = 20, freeze_blank_duration: bool = False,
          save_every: int = 0, on_save=lambda: None, weight_decay: float = 1e-3, spec_augment: bool = True,
          kl_weight: float = 0.0, kl_replay_only: bool = False, init_from: str = "", kl_terms: str = "blank,dur",
          seed: int = 1, kl_exempt: str = "", kl_blank_scale: float = 1.0, kl_punct: float = 0.0,
          kl_punct_rows: str = "all", kl_tok: float = 0.0, resume: bool = True, stage_dir: str = "",
          r3: bool = False, wf_weight: float = 1.0, en_weight: float = 1.0, l2sp_c: float = 0.0,
          l2sp_anchor: str = "", accumulate: int = 1, dur_weight: float = 0.0, suffix_anchor: bool = False,
          anchor_after: int = 1, gate: float = 0.9, gate_human: float = 0.0, cons_weight: float = 0.0,
          cons_pairs: str = "", cons_per_batch: int = 1, en_set: str = "whole", r3_sub_batch: int = 2,
          comma_weight: float = 0.0, wf_en_weight: float = 1.0, unfreeze_top: int = 0, kl_mask_space: bool = False,
          l2sp_exempt_num: bool = False, en_teacher: str = "", en_teacher_num: str = "anchor", en_kl_scale: float = 1.0,
          wf_pen_digits_only: bool = False, kl_mask_space_en: bool = False, enc_lr_scale: float = 1.0, enc_l2sp_c: float | None = None,
          lora_rank: int = 0, lora_alpha: float = 16.0, sampler_seed: int = 0, lr_offset: int = 0, lr_total: int = 0,
          enc_lr_peak: float = 0.0, enc_lr_floor: float = 0.0, enc_lr_warmup: int = 0, enc_lr_steps: int = 0,
          enc_clip: float = 0.0, enc_real_only: bool = False, en_free_teacher: bool = False, en_span_gate: float = 0.0):
    """`init_from` = "<run>@<step>": start from that run's checkpoint (the KL teacher stays stock).
    Mix entries may carry a third element, the KL treatment tag ("anchor" | "span" | "nonen" | "free"); tagged
    entries replace the path-based "replay/" rule. `resume`: continue from exp/<run>/last.ckpt when it exists.
    R4 (r3 only, all off by default): suffix_anchor / anchor_after / gate / gate_human (arm b), cons_weight + cons_pairs
    (comma-separated volume-relative pair lists; missing files are skipped) + cons_per_batch (arm c).
    en_set (r3): the en_pen piece set ("whole" | "first", add_r3_losses). r3_sub_batch: rows per auxiliary joint pass
    (add_r3_losses sub_batch): 1 halves that pass's peak memory, same batch and loss values, slower step.
    comma_weight (r3): the decimal-comma penalty's weight (add_r3_losses; 0 = off). wf_en_weight (r3): wf_pen's factor
    on English rows (add_r3_losses; 1 = off). unfreeze_top > 0 (r3 only): the last unfreeze_top conformer blocks of the
    encoder train too (unfreeze_top_layers: eval mode, L2-SP to their stock values, saved in every step checkpoint; the r3
    KL teacher keeps the stock blocks); 0 = the frozen encoder, unchanged. l2sp_exempt_num (r3 + l2sp_c): the joint's
    output rows of the pieces that mostly spell non-English spoken numbers (num_piece_ids over the nonen_span labels) and
    of the bare "▁" piece are exempt from the L2-SP pull too, like the written-form rows; False = unchanged.
    en_teacher (r3) = "<run>@<step>": the English KL rows (anchor / anchor_num / span) take that decoder + joint checkpoint
    as their KL teacher, every other row keeps stock; en_teacher_num "anchor" | "mask": their number cells anchored to it
    or masked as with stock; "" = stock everywhere, unchanged. en_kl_scale (r3): the English KL rows' KL terms x this
    (0 = no KL on English, non-English unchanged); 1 = unchanged. add_r3_losses' EN_TEACHER block comment has the masks.
    en_free_teacher / en_span_gate (r3, both off by default): the English "free" rows read the English teacher too; the English
    span rows' blank-KL gate (EN_TEACHER_TAGS block comment).
    Gentler encoder training (unfreeze_top only; all off by default, nothing changes when off): enc_lr_scale != 1: the
    trainable encoder tensors form their own optimizer parameter group at lr * enc_lr_scale (encoder_param_groups; the
    L2-SP pull uses each parameter's own group lr). enc_l2sp_c: the L2-SP strength c of the encoder tensors (None = l2sp_c;
    the decoder and joint keep l2sp_c). lora_rank > 0: the top blocks' own weights stay frozen and LoRA adapters of that
    rank (scale lora_alpha / lora_rank) on the attention q/k/v/out and feed-forward linears train instead (add_lora); the
    checkpoints hold the adapters merged into the weights.
    Branch options (a warm start off another run's checkpoint, optionally with the encoder; all off by default and then
    nothing changes, bit for bit): sampler_seed != 0: the lhotse sampler's seed (batch order); the mix is still written with
    `seed`. lr_total > 0: the decoder + joint lr follows a lr_total-step cosine (peak lr, floor lr / 10, warmup steps) from
    step lr_offset on (decoder_lr_at: the main run's tail, with a restart ramp), through LrSchedule. enc_lr_peak > 0
    (unfreeze_top): the encoder tensors are their own optimizer group with an ABSOLUTE schedule: warmup enc_lr_warmup
    steps to enc_lr_peak, cosine to enc_lr_floor over enc_lr_steps (0 = max_steps); NeMo's shared floor does not apply.
    enc_clip > 0 (unfreeze_top): gradient clipping per group (GroupClip): decoder + joint at 1.0 as in the control, the
    encoder at enc_clip; pre-clip norms are logged. enc_real_only (unfreeze_top, r3): rows tagged synthetic send no gradient
    into the encoder (add_r3_losses). With unfreeze_top, every save logs the encoder's L2 distance from stock
    (enc_l2_delta.jsonl in the run's folder)."""
    import nemo.collections.asr as nemo_asr

    exp = f"{vol}/exp/{run}"
    os.makedirs(exp, exist_ok=True)
    last = f"{exp}/last.ckpt"
    resuming = resume and os.path.exists(last)
    sampler_seed_arg = sampler_seed
    sampler_seed = resolve_sampler_seed(seed, sampler_seed_arg)
    if resuming:
        # lhotse restarts its sampler on resume: a NEW sampler seed so the resumed run does not replay the batches it
        # already saw (NeMo lhotse docs). R3: the MIX (which rows) stays the same — only the order
        # changes (retrain-3 review: reseeding write_mix changed the data mid-run).
        step0 = read_resume_step(last, exp)
        sampler_seed = resolve_sampler_seed(seed, sampler_seed_arg, True, step0)
        if not r3:
            seed = sampler_seed
        print(f"resume: sampler seed -> {sampler_seed}")
    replay_texts = set()
    tagged: dict = {}
    manifest = f"{stage_dir}/train_manifest.jsonl" if stage_dir else f"{exp}/train_manifest.jsonl"
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
    row_meta = []
    hours = write_mix(vol, manifest, mix, seed=seed, replay_texts=replay_texts,
                      kl_exempt=kl_exempt, tagged=tagged, stage_dir=stage_dir, meta_out=row_meta)
    if stage_dir:
        import shutil

        shutil.copyfile(manifest, f"{exp}/train_manifest.jsonl")  # the record of what was trained on
    check_mix_drift(exp, manifest, resuming, strict=r3)  # r3 keeps the seed on resume: the same mix, same order
    if tagged and not r3:
        replay_texts |= tagged.get("anchor", set())
        # a label in two tags keeps the stricter one: anchor > nonen > span
        span_texts = tagged.get("span", set()) - replay_texts
        nonen_texts = tagged.get("nonen", set()) - replay_texts
        print(f"tagged KL labels: anchor {len(replay_texts)} span {len(span_texts)} non-en {len(nonen_texts)}")
    else:
        span_texts = nonen_texts = None
    pl.seed_everything(seed)
    print(f"train mix: {hours:.2f} h")

    m = nemo_asr.models.ASRModel.restore_from(ultra, map_location="cpu")
    manifest_sha12 = manifest_sha256(manifest)[:12]
    try:  # measured exposure (exposure.py): what each manifest line is, so the seen-rows log can be read back
        exposure.write_row_index(exp, manifest_sha12, [json.loads(l) for l in open(manifest, encoding="utf-8")],
                                 row_meta, lambda t: m.tokenizer.text_to_ids(t))
    except Exception as e:
        print(f"row index NOT written ({type(e).__name__}: {e}); exposure_report will not work for this run", flush=True)
    # A fresh train_ds: the checkpoint's own carries its original corpus settings (tarred inputs,
    # bucket bins, a null max_tps) that do not apply here.
    with open_dict(m.cfg):
        print("checkpoint train_ds keys:", sorted(m.cfg.train_ds.keys()))
        m.cfg.train_ds = OmegaConf.create({
            "manifest_filepath": manifest, "sample_rate": 16000,
            "use_lhotse": True, "text_field": "text", "batch_duration": batch_duration,
            "max_duration": 20.0, "min_duration": 0.5, "use_bucketing": True, "num_buckets": 10,
            "bucket_buffer_size": 20000, "shuffle": True, "shuffle_buffer_size": 10000, "seed": sampler_seed,
            "num_workers": 8, "pin_memory": True, "skip_missing_manifest_entries": False})
    m.setup_training_data(m.cfg.train_ds)
    if not spec_augment:
        # With the encoder frozen, SpecAugment masks the encoder's INPUT while the label keeps the
        # masked words, which rewards skipping ahead and guessing (suspected cause of the extra
        # deletions in v3).
        m.spec_augmentation = None
        print("SpecAugment off")
    m.encoder.freeze()
    enc_stock = None
    if unfreeze_top:
        # the r3 step is the one that reads the training pass's encoder output; SpecAugment would hand the teacher masked
        # input; consistency_term encodes its pairs without a gradient
        assert r3 and not spec_augment and not cons_weight, "unfreeze_top needs r3, SpecAugment off and no consistency term"
        enc_stock = unfreeze_top_layers(m, unfreeze_top, train_base=not lora_rank)
        if lora_rank:
            add_lora(m, unfreeze_top, lora_rank, lora_alpha)
    assert unfreeze_top or not (lora_rank or enc_lr_scale != 1.0 or enc_l2sp_c is not None), \
        "enc_lr_scale / enc_l2sp_c / lora_rank need unfreeze_top"
    assert unfreeze_top or not (enc_lr_peak or enc_clip or enc_real_only), \
        "enc_lr_peak / enc_clip / enc_real_only need unfreeze_top"
    assert not (enc_lr_peak and enc_lr_scale != 1.0), "enc_lr_peak (absolute schedule) and enc_lr_scale are two ways to set one lr"
    assert not (lora_rank and enc_l2sp_c is not None), "enc_l2sp_c: the LoRA base weights are frozen (no encoder pull)"
    if "blankspan" in kl_terms:
        assert kl_replay_only, "blankspan only works with kl_replay_only (span rows are matched by label)"
    assert r3 or (not en_teacher and en_kl_scale == 1.0 and not en_free_teacher and not en_span_gate), \
        "en_teacher / en_kl_scale / en_free_teacher / en_span_gate need r3"
    en_state = None
    if en_teacher:  # loaded on every start (resume too): the teacher is an attribute, never in a checkpoint
        assert "@" in en_teacher and not en_teacher.startswith("ultra"), f"en_teacher {en_teacher!r}: '<run>@<step>'"
        src, _, step = en_teacher.partition("@")
        en_path = f"{vol}/exp/{src}/step{step}.pt"
        en_state = torch.load(en_path, map_location="cpu")
        print(f"en_teacher {en_teacher} ({en_path}): {len(en_state)} tensors, number cells {en_teacher_num}", flush=True)
    extra_cb = []
    if r3:
        ids = lambda t: tuple(m.tokenizer.text_to_ids(t))  # noqa: E731
        row_tags = {}
        ns = tagged.get("nonen_span", set())
        for other in ("free", "span", "anchor_num", "nonen", "anchor"):  # a stricter tag would win silently
            both = ns & tagged.get(other, set())
            assert not both, f"{len(both)} nonen_span labels also tagged {other}, e.g. {next(iter(both))[:80]!r}"
        for tag in ("free", "span", "nonen_span", "anchor_num", "nonen", "anchor"):  # later = stricter wins
            for t in tagged.get(tag, ()):
                row_tags[ids(t)] = tag
        span_prefix = {}
        for t, pre in tagged.get("_nspan_prefix", {}).items():  # anchored label positions: prefix minus its last word
            pw = pre.split() if pre == t else pre.split()[:-1]  # a row with no number keeps every word (review M2)
            span_prefix[ids(t)] = len(ids(" ".join(pw))) if pw else 0
        nspan_pos, mis = {}, 0
        if suffix_anchor:
            spm = m.tokenizer.tokenizer
            for t, flags in tagged.get("_nspan_words", {}).items():
                k = ids(t)
                words = row_words(spm, k)
                if len(words) != len(flags):  # the tokenizer's word split differs: this row keeps the prefix rule
                    mis += 1
                    continue
                nspan_pos[k] = anchor_positions(words, flags, len(k), after=anchor_after)
            print(f"R4 suffix anchoring: {len(nspan_pos)} nonen_span labels, {mis} with a word-split mismatch (prefix only)")
        synthetic = {ids(t) for t in tagged.get("_synthetic", ())}
        bank = None
        if cons_weight:
            lines = []
            for pth in [x for x in cons_pairs.split(",") if x]:
                if os.path.exists(f"{vol}/{pth}"):
                    lines += open(f"{vol}/{pth}").read().splitlines()
                else:
                    print(f"R4 consistency: {pth} missing, skipped")
            pairs, st = load_pairs(lines, known_texts=set(tagged.get("nonen_span", ())) | set(tagged.get("span", ())))
            print(f"R4 consistency pairs: {dict(st)}")
            assert pairs, "cons_weight > 0 but no usable pair (cons_pairs)"
            bank = PairBank(pairs, per_batch=cons_per_batch, seed=sampler_seed, stage_dir=stage_dir)
        add_r3_losses(m, row_tags, kl_weight=kl_weight, punct_weight=kl_punct, tok_weight=kl_tok,
                      wf_weight=wf_weight, en_weight=en_weight, dur_weight=dur_weight, span_prefix=span_prefix,
                      suffix_anchor=suffix_anchor, anchor_after=anchor_after, nspan_pos=nspan_pos, gate=gate,
                      gate_human=gate_human, synthetic=synthetic, cons_weight=cons_weight, pair_bank=bank,
                      en_set=en_set, sub_batch=r3_sub_batch, comma_weight=comma_weight,
                      wf_en_weight=wf_en_weight, kl_mask_space=kl_mask_space, en_teacher_state=en_state,
                      en_teacher_num=en_teacher_num, en_kl_scale=en_kl_scale, wf_pen_digits_only=wf_pen_digits_only,
                      kl_mask_space_en=kl_mask_space_en, enc_real_only=enc_real_only,
                      en_free_teacher=en_free_teacher, en_span_gate=en_span_gate)
        if l2sp_c or enc_l2sp_c:
            last_layer = m.joint.joint_net[-1].weight.shape[0]
            exempt = [i for i in wf_piece_ids(m.tokenizer.tokenizer) if i < last_layer]
            if l2sp_exempt_num:
                spm = m.tokenizer.tokenizer
                lf = [(ids(t), fl) for t, fl in tagged.get("_nspan_words", {}).items()]
                num_ids, nmis, near = num_piece_ids(spm, lf)
                assert num_ids, "l2sp_exempt_num but no number piece found in the nonen_span labels"
                assert nmis <= 0.2 * len(lf), f"l2sp_exempt_num: {nmis}/{len(lf)} labels with a word-split mismatch"
                # the bare "▁" (digit-mode entry) row too: kl_mask_space frees it from the KL, this from the pull
                space = [i for i in range(spm.get_piece_size()) if spm.id_to_piece(i) == "▁"]
                exempt = sorted(set(exempt) | {i for i in num_ids + space if i < last_layer})
                print(f"l2sp_exempt_num: {len(num_ids)} number pieces + '▁' {space} exempt from {len(lf)} nonen_span "
                      f"labels ({nmis} word-split mismatches): {[spm.id_to_piece(i) for i in num_ids]}; below the share "
                      f"cut (not exempt): {[spm.id_to_piece(i) for i in near]}", flush=True)
            extra_cb.append(L2SP(l2sp_anchor, l2sp_c, exempt, extra_anchor=None if lora_rank else enc_stock,
                                 enc_c=enc_l2sp_c))
            weight_decay = 0.0
            print(f"L2SP toward {l2sp_anchor} c={l2sp_c}"
                  + (f" (encoder c={enc_l2sp_c})" if enc_l2sp_c is not None else "")
                  + f", {len(exempt)} written-form rows exempt; AdamW decay 0")
    elif kl_weight or kl_punct:
        add_blank_duration_kl(m, kl_weight, only_texts=replay_texts if kl_replay_only else None, terms=kl_terms,
                              blank_scale=kl_blank_scale, punct_weight=kl_punct, punct_rows=kl_punct_rows,
                              span_texts=span_texts, nonen_texts=nonen_texts, tok_weight=kl_tok)
        print(f"KL to stock ({kl_terms}), weight {kl_weight}"
              + (f", replay rows only ({len(replay_texts)} labels)" if kl_replay_only else ""))
    if init_from and not resuming:
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

    if enc_lr_scale != 1.0 or enc_lr_peak:
        # NeMo calls setup_optimizer_param_groups on every (re)build of the optimizer, Lightning's included
        def split_groups():
            m._optimizer_param_groups = encoder_param_groups(m, lr, enc_lr_scale, peak=enc_lr_peak or None)

        m.setup_optimizer_param_groups = split_groups
        print(f"enc_lr_scale {enc_lr_scale}: encoder tensors at lr {enc_lr_peak or lr * enc_lr_scale:g}, decoder + joint at "
              f"lr {lr:g}", flush=True)
    branch_cb, clip_val = branch_trainer_parts(
        lr, warmup, lr_offset=lr_offset, lr_total=lr_total, enc_lr_peak=enc_lr_peak, enc_lr_floor=enc_lr_floor,
        enc_lr_warmup=enc_lr_warmup, enc_lr_steps=enc_lr_steps, enc_clip=enc_clip, max_steps=max_steps,
        has_encoder=bool(unfreeze_top))
    if branch_cb:
        print(f"branch options: sampler_seed {sampler_seed} lr_offset {lr_offset} lr_total {lr_total} enc_lr_peak "
              f"{enc_lr_peak:g} floor {enc_lr_floor:g} warmup {enc_lr_warmup} steps {enc_lr_steps or max_steps} enc_clip "
              f"{enc_clip:g} enc_real_only {enc_real_only}", flush=True)

    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=max_steps,
                         limit_train_batches=max_steps * accumulate, limit_val_batches=0, num_sanity_val_steps=0,
                         use_distributed_sampler=False, gradient_clip_val=clip_val, logger=False,
                         accumulate_grad_batches=accumulate,
                         enable_checkpointing=False, enable_progress_bar=False,
                         callbacks=[KeepEncoderFrozen(), StepLog(mem=bool(unfreeze_top)), SeenRows(exp, manifest_sha12)] + extra + extra_cb
                         + ([ParamGroupLog()] if enc_lr_scale != 1.0 or enc_lr_peak else [])
                         + branch_cb
                         + ([SaveEvery(exp, save_every, on_save, last)] if save_every else []))
    m.set_trainer(trainer)
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": lr, "betas": [0.9, 0.98], "weight_decay": weight_decay,
        "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": lr / 10,
                  "max_steps": max_steps}}))
    if unfreeze_top:  # every trainable tensor must reach the optimizer (NeMo builds its parameter groups here)
        held = {id(p) for g in m._optimizer.param_groups for p in g["params"]}
        trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        missing = [n for n, p in trainable if id(p) not in held]
        assert not missing, f"{len(missing)} trainable tensors not in the optimizer, e.g. {missing[:3]}"
        n_enc = sum(1 for n, _ in trainable if n.startswith("encoder."))
        print(f"optimizer: all {len(trainable)} trainable tensors held ({n_enc} encoder)", flush=True)
    t0 = time.time()
    if resuming:
        print(f"RESUMING from {last}")
    trainer.fit(m, ckpt_path=last if resuming else None)
    print(f"trained {max_steps} steps in {time.time() - t0:.0f}s")
    torch.save(trainable_state(m), f"{exp}/decoder_joint.pt")
    return m
