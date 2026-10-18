"""Fine-tune Parakeet Ultra's prediction network and joint with the encoder frozen (optionally its top conformer blocks
trainable too: train(unfreeze_top=N)).

Saves only the trainable weights (decoder + joint, ~18M params, + the unfrozen encoder blocks) to
/vol/exp/<run>/decoder_joint.pt.
"""

import collections
import json
import re
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
    dropout in eval mode. With unfreeze_top the trainable top blocks stay in eval mode too (unfreeze_top_layers)."""

    def on_train_epoch_start(self, trainer, m):
        m.encoder.eval()

    def on_train_batch_start(self, trainer, m, batch, idx):
        m.encoder.eval()


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
                os.replace(self.last + ".tmp", self.last)
                open(self.last + ".step", "w").write(str(step))
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
            span_prefix: dict | None = None, cache: dict | None = None) -> list[bool]:
    """(len(ids) + 1) flags: the label positions where the anchor KL applies for one row (add_r3_losses' table; padding
    beyond len(ids) is masked there anyway). R4 arm b (suffix_anchor): span and anchor_num rows anchor every non-number
    position (anchor_positions over en_number_words); nonen_span rows use nspan_pos (built in train() from the row's
    number tokens) and fall back to the prefix rule when it has no entry."""
    n, key = len(ids), tuple(ids)
    if tag in ("anchor", "nonen"):
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


def blank_gate(kmask, t_lnb, thr):
    """Cells where the blank KL applies: kmask and teacher P(non-blank) > the row's threshold. kmask / t_lnb (B, T, U),
    thr (B,)."""
    return kmask & (t_lnb.exp() > thr[:, None, None])


def row_gate(tags: list[str], keys: list, gate: float, gate_human: float, synthetic: set) -> list[float]:
    """Per-row blank-KL teacher gate (teacher P(non-blank) must exceed it). gate_human (when > 0) replaces `gate` on the
    human-verified rows: anchor, anchor_num, and nonen_span rows whose label is not synthetic (TTS rows, TTS windows)."""
    if not gate_human:
        return [gate] * len(tags)
    return [gate_human if tg in ("anchor", "anchor_num") or (tg == "nonen_span" and k not in synthetic) else gate
            for tg, k in zip(tags, keys)]


def per_tag_summary(tags: list[str], acc: list) -> dict:
    """{tag: {term: value, term_n: count}} from the per-row (numerator, denominator) vectors of add_r3_losses' chunk:
    acc[0/1] blank KL, [2/3] punct KL, [4/5] token KL, [6/7] wf_pen, [8/9] en_pen (numerator / count per row)."""
    out = {}
    for tg in sorted(set(tags)):
        ix = torch.tensor([i for i, t in enumerate(tags) if t == tg], device=acc[0].device)
        d = {"rows": len(ix)}
        for name, k in (("kl_b", 0), ("kl_p", 2), ("kl_t", 4), ("wf", 6), ("en", 8)):
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
                  kl_mask_space: bool = False):
    """R3 losses (table above). R4 options, all off by default: suffix_anchor (arm b: anchor every non-number position of
    span / anchor_num rows, and of nonen_span rows via nspan_pos {label ids: anchor_positions}), gate_human (arm b: the
    blank-KL teacher gate on human-verified rows, `gate` elsewhere), cons_weight + pair_bank (arm c: cross-window
    consistency, consistency_term). en_set: the en_pen piece set and columns ("whole" | "first"; EN_NUM_WORDS_FIRST,
    en_pen_cols). comma_weight > 0: the decimal-comma penalty (comma_piece_ids) at the number columns of nonen_span rows
    (klu False, as en_set "first"), except columns whose own target is a comma (comma_target_cols); 0 = the loss is
    unchanged bit for bit. wf_en_weight != 1: wf_pen per row x wf_row_weights (English rows x wf_en_weight, nonen_span
    rows x 1); 1 = unchanged bit for bit. kl_mask_space: the nonen_span "teacher is writing digits" cell mask also counts
    the bare word-boundary piece "▁" (the only entry into digit mode: the vocabulary has no "▁2"-style pieces), so the
    KL terms stop teaching digit entry over the number's audio at units-first onsets; False = unchanged bit for bit."""
    import copy

    assert en_set in ("whole", "first"), en_set

    teacher_dec, teacher_joint = copy.deepcopy(m.decoder).eval(), copy.deepcopy(m.joint).eval()
    for p in list(teacher_dec.parameters()) + list(teacher_joint.parameters()):
        p.requires_grad_(False)
    m._kl_teacher = (teacher_dec, teacher_joint)
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
    wfm_t = torch.tensor(wf_piece_ids(spm) + space_ids) if kl_mask_space else wf_t
    if kl_mask_space:
        assert space_ids, "kl_mask_space but no bare '▁' piece in the tokenizer"
        print(f"kl_mask_space: nonen_span digit-cell mask counts wf pieces + '▁' {space_ids}", flush=True)
    th_first = int(m.tokenizer.text_to_ids("thousand")[0])
    th_t = torch.tensor([th_first])
    cm_t = torch.tensor(comma_piece_ids(spm) if comma_weight else [], dtype=torch.long)
    if comma_weight:
        assert len(cm_t), "comma_weight > 0 but no comma piece in the tokenizer"
        print(f"comma pen: weight {comma_weight} pieces {[(int(i), spm.id_to_piece(int(i))) for i in cm_t]}", flush=True)
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

    def training_step(batch, batch_idx):
        out = orig_step(batch, batch_idx)
        signal, signal_len, tokens, tokens_len = batch[:4]
        tdec, tjoint = m._kl_teacher
        tdec.to(signal.device), tjoint.to(signal.device)
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
                                                                 nspan_pos, span_prefix, en_cache))
        klu = klu.to(dev)
        thr = torch.tensor(row_gate(tags, [tuple(t) for t in row_toks], gate, gate_human, synthetic or set()), device=dev)
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
        # comma penalty: columns open to it (not a comma target of the row's own label); unused when comma_weight 0
        cm_ok = (~comma_target_cols(row_toks, cm_t.tolist(), W)).to(dev) if comma_weight else torch.ones_like(klu)

        def chunk(fs_sl, f_sl, s_sl, t_sl, enc_len_sl, tok_len_sl, klu_sl, nonen_sl, nspan_sl, yr_sl, yrcol_sl, thr_sl,
                  cmok_sl):
            with NoDropout(m.joint):
                s_logits = m.joint.joint(fs_sl, s_sl).float()
            with torch.no_grad():
                t_logits = tjoint.joint(f_sl, t_sl).float()
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
            if bool(nspan_sl.any()):
                t_wf = (t_logits[..., wfm].logsumexp(-1) - t_nbz).exp()
                wf_cell = nspan_sl[:, None, None] & (t_wf > 0.1)
                tmask = tmask & ~wf_cell
                bmask = bmask & ~wf_cell
                kmask = kmask & ~wf_cell
                dmask = dmask & ~wf_cell
            if bool((nonen_sl | nspan_sl).any()):
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

            wf_s, wf_n = pen(wf, ~nonen_sl)
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
                    en_n, th_s, th_n, (kl_d * dmask).sum((1, 2)), dmask.sum((1, 2)).float(), cm_s, cm_n)

        parts = []
        for j in range(0, B, sub_batch):
            sl = torch.arange(j, min(B, j + sub_batch), device=dev)
            u = int(tokens_len[sl].max()) + 1
            t = int(enc_len[sl].max())
            parts.append(torch.utils.checkpoint.checkpoint(
                chunk, f_s[sl, :t], f[sl, :t], s_g[sl, :, :u].transpose(1, 2), t_g[sl, :, :u].transpose(1, 2),
                enc_len[sl], tokens_len[sl], klu[sl], is_nonen[sl], is_nspan[sl], is_yr[sl], yrcol[sl], thr[sl],
                cm_ok[sl], use_reentrant=False))
        acc = [torch.cat(v) for v in zip(*parts)]  # 16 per-row vectors (B,)
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
        cons, n_pairs = torch.zeros((), device=dev), 0
        if cons_weight and pair_bank is not None:
            cons, n_pairs = consistency_term(m, pair_bank, dev, nb)
            aux = aux + cons_weight * cons
        m._r3_parts = {"rnnt": out["loss"].detach(), "kl_b": kl_b.detach(), "kl_p": kl_p.detach(),
                       "kl_t": kl_t.detach(), "wf": wf_pen.detach(), "en": en_pen.detach(),
                       "th": th_pen.detach(), "kl_d": kl_d.detach(), "cons": cons.detach(),
                       "comma": cm_pen.detach()}
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
                                                               for k in ("kl_b", "kl_p", "kl_t", "wf", "en")), flush=True)
        return out

    m.training_step = training_step


class L2SP(pl.Callback):
    """Decoupled pull toward v10pt after every optimizer step: p <- p - lr * c * (p - p_v10pt). The anchor is loaded
    from the v10pt FILE (also on resume: init_from is skipped then). The written-form output rows of the joint's last
    layer are exempt (N2: they must stay free to keep non-English digits at stock strength)."""

    def __init__(self, anchor_path: str, c: float, exempt_rows: list[int], extra_anchor: dict | None = None):
        """extra_anchor: more anchor tensors by parameter name (unfreeze_top: the stock encoder blocks), for trainable
        parameters the anchor file does not hold."""
        self.path, self.c, self.exempt, self.extra = anchor_path, c, exempt_rows, extra_anchor
        self.anchor = None

    def on_train_batch_end(self, trainer, m, outputs, batch, batch_idx):
        if not self.c or (batch_idx + 1) % trainer.accumulate_grad_batches:
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
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        last = "joint.joint_net." + str(len(m.joint.joint_net) - 1)
        with torch.no_grad():
            drift = 0.0
            for n, p in m.named_parameters():
                a = self.anchor.get(n)
                if a is None:
                    continue
                d = p.data - a
                if n.startswith(last) and self.exempt:
                    d[self.exempt] = 0
                p.data.sub_(d, alpha=lr * self.c)
                drift += float((d.float() ** 2).sum())
        if (batch_idx + 1) % (50 * trainer.accumulate_grad_batches) == 0:
            print(f"l2sp drift {drift ** 0.5:.4f}", flush=True)


MERGED_KL = True  # one joint pass for all KL terms (2026-10-13 speed fix)
KL_EQ_CHECK = os.environ.get("KL_EQ_CHECK") == "1"  # first 3 batches: also run the separate passes and print both


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
              kl_exempt: str = "", tagged: dict | None = None, stage_dir: str = "") -> float:
    """Concatenate manifests, taking `hours` from each (all if hours <= 0). Labels of rows from
    `replay/` manifests are added to `replay_texts` when given."""
    r = random.Random(seed)
    rows, exempt = [], set()
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
    if replay_texts is not None and kl_exempt == "numwords":
        # Real speech that says numbers is where spoken form is learned on real audio: no anchor there,
        # whichever manifest the row came from (LSPC also holds number-word rows).
        from common import NUMBER_WORDS
        exempt |= {t for t in replay_texts if NUMBER_WORDS.search(t)}
    if replay_texts is not None and exempt:
        replay_texts.difference_update(exempt)
        print(f"KL-exempt labels ({kl_exempt}): {len(exempt)}, anchored: {len(replay_texts)}")
    r.shuffle(rows)
    if stage_dir:
        # Copy the selected audio to the container's local disk: ~60k small files read from the network volume made
        # each step ~3x slower than v12y's (dress rehearsal 2026-10-13).
        from concurrent.futures import ThreadPoolExecutor
        import shutil

        t0 = time.time()

        def stage(x):
            dst = stage_dir + x["audio_filepath"]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copyfile(x["audio_filepath"], dst)
            return {**x, "audio_filepath": dst}

        with ThreadPoolExecutor(64) as ex:
            rows = list(ex.map(stage, rows))
        print(f"staged {len(rows)} files to {stage_dir} in {time.time() - t0:.0f}s", flush=True)
    with open(out, "w") as f:
        for x in rows:
            f.write(json.dumps({k: x[k] for k in ("audio_filepath", "duration", "text")}) + "\n")
    return sum(x["duration"] for x in rows) / 3600


def trainable_state(m) -> dict:
    """Decoder + joint, plus the encoder blocks unfreeze_top_layers made trainable (parameters and buffers)."""
    keep = getattr(m, "_enc_unfrozen", ())
    return {k: v.detach().cpu() for k, v in m.state_dict().items()
            if not (k.startswith("encoder.") or k.startswith("preprocessor.")) or k.startswith(keep)}


def unfreeze_top_layers(m, n: int) -> dict:
    """Make the last `n` conformer blocks of the frozen encoder trainable: every parameter under encoder.layers.<L-n> ..
    encoder.layers.<L-1> (both feed-forward modules, self-attention, the convolution module with its BatchNorm affine,
    and the block's layer norms). pre_encode (subsampling), the positional encoding and blocks 0 .. L-n-1 stay frozen.
    The whole encoder stays in eval mode (KeepEncoderFrozen): dropout off and BatchNorm on its stock running statistics,
    which are not re-estimated (batches skewed to the trained languages would shift them for every language); the
    student and the KL teacher then see the same deterministic blocks.
    Side effects on `m`: `_enc_unfrozen` (the blocks' state-dict prefixes, kept by trainable_state), `_enc_stock_top`
    (their stock tensors, an attribute so neither saved nor trained: the KL teacher's top blocks in add_r3_losses), and
    a forward hook that keeps the training pass's encoder input and output (`_enc_main`) for add_r3_losses.
    Returns the stock tensors ("encoder.layers.<i>.<name>" -> CPU clone, parameters and buffers): the L2-SP anchor."""
    layers = m.encoder.layers
    n_layers = len(layers)
    assert 0 < n <= n_layers, f"unfreeze_top {n}: the encoder has {n_layers} blocks"
    idx = list(range(n_layers - n, n_layers))
    prefixes = tuple(f"encoder.layers.{i}." for i in idx)
    for i in idx:
        for p in layers[i].parameters():
            p.requires_grad_(True)
    stock = {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.startswith(prefixes)}
    m._enc_unfrozen = prefixes
    m._enc_stock_top = stock
    m._enc_main = None

    def keep_main(mod, args, kwargs, out):
        if torch.is_grad_enabled():  # the training pass only (the teacher and consistency passes run under no_grad)
            m._enc_main = (kwargs["audio_signal"], kwargs["length"], out[0], out[1])

    m.encoder.register_forward_hook(keep_main, with_kwargs=True)
    n_par = sum(p.numel() for i in idx for p in layers[i].parameters())
    n_ten = sum(1 for i in idx for _ in layers[i].parameters())
    subs = sorted({k[len(f"encoder.layers.{idx[0]}."):].split(".")[0] for k in stock if k.startswith(prefixes[0])})
    print(f"unfreeze_top {n}: encoder.layers.{idx[0]}..{idx[-1]} of {n_layers} trainable, {n_par:,} params in {n_ten} "
          f"tensors (per block: {', '.join(subs)}); pre_encode + layers 0..{idx[0] - 1} frozen; encoder kept in eval mode "
          f"(dropout off, BatchNorm running stats fixed)", flush=True)
    return stock


def stock_top_encoder(m, feats, feats_len):
    """The encoder output with the stock values of the unfrozen top blocks (the KL teacher's encoder), no gradient."""
    dev = feats.device
    cache = getattr(m, "_enc_stock_dev", None)
    if cache is None or next(iter(cache.values())).device != dev:
        cache = {k[len("encoder."):]: v.to(dev) for k, v in m._enc_stock_top.items()}
        m._enc_stock_dev = cache
    with torch.no_grad():
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
          comma_weight: float = 0.0, wf_en_weight: float = 1.0, unfreeze_top: int = 0, kl_mask_space: bool = False):
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
    KL teacher keeps the stock blocks); 0 = the frozen encoder, unchanged."""
    import nemo.collections.asr as nemo_asr

    exp = f"{vol}/exp/{run}"
    os.makedirs(exp, exist_ok=True)
    last = f"{exp}/last.ckpt"
    resuming = resume and os.path.exists(last)
    sampler_seed = seed
    if resuming:
        # lhotse restarts its sampler on resume: a NEW sampler seed so the resumed run does not replay the batches it
        # already saw (NeMo lhotse docs). R3: the MIX (which rows) stays the same — only the order
        # changes (retrain-3 review: reseeding write_mix changed the data mid-run).
        step0 = int(open(last + ".step").read().strip() or 0)
        sampler_seed = seed + 1000 * step0
        if not r3:
            seed = sampler_seed
        print(f"resume: sampler seed -> {sampler_seed}")
    replay_texts = set()
    tagged: dict = {}
    manifest = f"{stage_dir}/train_manifest.jsonl" if stage_dir else f"{exp}/train_manifest.jsonl"
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
    hours = write_mix(vol, manifest, mix, seed=seed, replay_texts=replay_texts,
                      kl_exempt=kl_exempt, tagged=tagged, stage_dir=stage_dir)
    if stage_dir:
        import shutil

        shutil.copyfile(manifest, f"{exp}/train_manifest.jsonl")  # the record of what was trained on
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
        enc_stock = unfreeze_top_layers(m, unfreeze_top)
    if "blankspan" in kl_terms:
        assert kl_replay_only, "blankspan only works with kl_replay_only (span rows are matched by label)"
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
                      wf_en_weight=wf_en_weight, kl_mask_space=kl_mask_space)
        if l2sp_c:
            last_layer = m.joint.joint_net[-1].weight.shape[0]
            exempt = [i for i in wf_piece_ids(m.tokenizer.tokenizer) if i < last_layer]
            extra_cb.append(L2SP(l2sp_anchor, l2sp_c, exempt, extra_anchor=enc_stock))
            weight_decay = 0.0
            print(f"L2SP toward {l2sp_anchor} c={l2sp_c}, {len(exempt)} written-form rows exempt; AdamW decay 0")
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

    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=max_steps,
                         limit_train_batches=max_steps * accumulate, limit_val_batches=0, num_sanity_val_steps=0,
                         use_distributed_sampler=False, gradient_clip_val=1.0, logger=False,
                         accumulate_grad_batches=accumulate,
                         enable_checkpointing=False, enable_progress_bar=False,
                         callbacks=[KeepEncoderFrozen(), StepLog(mem=bool(unfreeze_top))] + extra + extra_cb
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
