"""Pre-flight for the span-selective duration anchor (kl_terms durspan, v10b): the span-mask
audit, the blank/duration drift of an existing checkpoint against stock, and, optionally, a
short micro-train with the v10b loss plus year/money decodes. No full training run.

Stage A mirrors ft_train.add_blank_duration_kl's chunk_kl exactly — the same clean-feats path
(preprocessor + frozen encoder, no SpecAugment), the same blank and 5-bin duration formulas,
the same gate (stock P(non-blank) > 0.9) — and splits every lattice point by row class
(number-word row vs clean) and span membership, using ft_train.span_mask: the mask the
training will use, not a reimplementation of it.

Budget on an L4 (steps=0, the default): load ~90 s, stage A ~60 s, one decode ~60 s — about
3-4 min; every stage prints its wall clock. steps>0 adds ~30 s of loader setup plus ~9 s a
step (batch 200 s) and a second decode.

    modal run pipeline/modal_app.py::preflight --run v10a --kl-weight <the run's kl_weight>
"""

import json
import os
import re
import time

from common import NUMBER_WORDS, WRITTEN_FORM, number_span
from ft_train import row_words, span_mask, write_mix

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MONEY = re.compile(r"[\u0024\u20ac\u00a3]\s?\d")  # a currency sign before a digit
GERMAN = re.compile(r"\b(eine?|eins|zwei|drei|vier|f\u00fcnf|sechs|sieben|acht|neun|zehn|hundert|tausend)\b")
UNITS = set("two three four five six seven eight nine".split())
TENS = set("twenty thirty forty fifty sixty seventy eighty ninety".split())


def probes(vol):
    """E22 utterances with a year and with a leading-sign money amount: the collapse and drop
    probes (short clips, so the decode stays inside the budget)."""
    path = f"{vol}/e22/numbers.jsonl"
    if not os.path.exists(path):
        print("no e22/numbers.jsonl: decode probes skipped")
        return [], []
    rows = [json.loads(l) for l in open(path)]
    years = [r for r in rows if YEAR.search(r["text"])][:120]
    money = [r for r in rows if MONEY.search(r["text"])][:40]
    print(f"probes: {len(years)} year rows, {len(money)} money rows")
    return years, money


def collapse_signs(text):
    """A bare unit word right before a tens word ('two twenty one' for 'twenty twenty one'):
    the repeated-number-word collapse signature; rare as an honest year reading."""
    w = re.sub(r"[^\w' ]", " ", text.lower()).split()
    return sum(1 for a, b in zip(w, w[1:]) if a in UNITS and b in TENS)


def decode_summary(m, years, money, label):
    import ft_eval

    paths = [r["audio_filepath"] for r in years] + [r["audio_filepath"] for r in money]
    if not paths:
        return
    t0 = time.time()
    hyps = ft_eval.decode(m, paths)
    yh, mh = hyps[:len(years)], hyps[len(years):]
    print(f"[{label}] decoded {len(paths)} clips in {time.time() - t0:.0f}s")
    for name, hs in (("years", yh), ("money", mh)):
        print(f"[{label}] {name}: collapse-sign {sum(collapse_signs(h['text']) for h in hs)}, "
              f"german-subword {sum(bool(GERMAN.search(h['text'])) for h in hs)}, "
              f"written-form {sum(bool(WRITTEN_FORM.search(h['text'])) for h in hs)} of {len(hs)}")
    for r, h in list(zip(years, yh))[:8]:
        print(f"  ref {r['text'][:70]!r}\n  hyp {h['text'][:70]!r}")
    for r, h in list(zip(money, mh))[:5]:
        print(f"  ref {r['text'][:70]!r}\n  hyp {h['text'][:70]!r}")


def measure(m, stock, vol, parts, terms="blank,gnorm,durspan", max_batches=8):
    """Stage A: the checkpoint against stock, per lattice point, split by row class and span
    membership (anchored vs span). Mirrors chunk_kl; see the module docstring."""
    import copy
    import torch
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    try:
        import librosa
    except ImportError:
        librosa = None

    tdec, tjoint = copy.deepcopy(stock.decoder).eval(), copy.deepcopy(stock.joint).eval()
    tdec, tjoint = tdec.cuda(), tjoint.cuda()
    spm = m.tokenizer.tokenizer
    nb = 8193  # 8192 tokens + blank (index 8192); indices 8193:8198 are the 5 durations

    exp = f"{vol}/exp/preflight_probe"
    os.makedirs(exp, exist_ok=True)
    write_mix(vol, f"{exp}/manifest.jsonl", parts, seed=1)
    rows = [json.loads(l) for l in open(f"{exp}/manifest.jsonl")][:max_batches * 4]
    if not rows:
        print("no rows: stage A skipped")
        return
    classes = ["numword" if NUMBER_WORDS.search(r["text"]) else "clean" for r in rows]
    print(f"probe rows: {len(rows)} ({classes.count('numword')} number-word, "
          f"{classes.count('clean')} clean)")
    stats = {c: {"b": 0.0, "b_n": 0, "da": 0.0, "da_n": 0, "ds": 0.0, "ds_n": 0}
             for c in ("clean", "numword")}
    shown = 0
    for i in range(0, len(rows), 4):
        batch, cls = rows[i:i + 4], classes[i:i + 4]
        sigs = []
        for r in batch:
            if sf is not None:
                x, _ = sf.read(r["audio_filepath"])
            elif librosa is not None:
                x, _ = librosa.load(r["audio_filepath"], sr=16000)
            else:
                raise RuntimeError("Neither soundfile nor librosa available")
            if hasattr(x, "ndim") and x.ndim > 1:
                x = x.mean(axis=-1)
            sigs.append(torch.tensor(x, dtype=torch.float32))
        T = max(len(x) for x in sigs)
        signal = torch.zeros(len(batch), T).cuda()
        for k, x in enumerate(sigs):
            signal[k, :len(x)] = x.cuda()
        signal_len = torch.tensor([len(x) for x in sigs]).cuda()
        ids = [m.tokenizer.text_to_ids(r["text"]) for r in batch]
        U = max(len(t) for t in ids)
        tokens = torch.zeros(len(batch), U, dtype=torch.long).cuda()
        for k, t in enumerate(ids):
            tokens[k, :len(t)] = torch.tensor(t).cuda()
        tokens_len = torch.tensor([len(t) for t in ids]).cuda()
        with torch.no_grad():
            feats, feats_len = m.preprocessor(input_signal=signal, length=signal_len)
            enc, enc_len = m.encoder(audio_signal=feats, length=feats_len)
            t_g, _, _ = tdec(targets=tokens, target_length=tokens_len)
            s_g, _, _ = m.decoder(targets=tokens, target_length=tokens_len)
        f = enc.transpose(1, 2)
        span = span_mask(spm, list(enumerate(ids)), len(batch), U, f.device)
        for k, c in enumerate(cls):
            if c == "numword" and shown < 5:
                shown += 1
                texts = [w for w, _, _ in row_words(spm, ids[k])]
                flags = number_span(texts)
                mark = " ".join(("[" + w + "]") if fl else w for w, fl in zip(texts, flags))
                print(f"  span audit [{mark[:110]}]")
        for j in range(0, len(batch), 2):
            idx = list(range(j, min(j + 2, len(batch))))
            sub = torch.tensor(idx, device=f.device)
            with torch.no_grad():
                s_logits = m.joint.joint(f[sub], s_g[sub].transpose(1, 2)).float()
                t_logits = tjoint.joint(f[sub], t_g[sub].transpose(1, 2)).float()
            Tl, Ul = s_logits.shape[1], s_logits.shape[2]
            mask = ((torch.arange(Tl, device=f.device)[None, :, None] < enc_len[sub][:, None, None])
                    & (torch.arange(Ul, device=f.device)[None, None, :] <= tokens_len[sub][:, None, None]))
            s_lb = s_logits[..., nb - 1] - s_logits[..., :nb].logsumexp(-1)
            t_lb = t_logits[..., nb - 1] - t_logits[..., :nb].logsumexp(-1)
            s_lnb = s_logits[..., :nb - 1].logsumexp(-1) - s_logits[..., :nb].logsumexp(-1)
            t_lnb = t_logits[..., :nb - 1].logsumexp(-1) - t_logits[..., :nb].logsumexp(-1)
            kl_blank = t_lb.exp() * (t_lb - s_lb) + t_lnb.exp() * (t_lnb - s_lnb)
            s_dur, t_dur = s_logits[..., nb:].log_softmax(-1), t_logits[..., nb:].log_softmax(-1)
            kl_dur = (t_dur.exp() * (t_dur - s_dur)).sum(-1)
            gate = mask & (t_lnb.exp() > 0.9)
            sp = span[sub][:, :Ul]
            for lk, gk in enumerate(idx):
                st = stats[cls[gk]]
                st["b"] += float((kl_blank * gate)[lk].sum())
                st["b_n"] += int(gate[lk].sum())
                anchored = mask[lk] & ~sp[lk]
                inspan = mask[lk] & sp[lk]
                st["da"] += float((kl_dur * anchored).sum())
                st["da_n"] += int(anchored.sum())
                st["ds"] += float((kl_dur * inspan).sum())
                st["ds_n"] += int(inspan.sum())
    for c in ("clean", "numword"):
        s = stats[c]
        if s["b_n"] or s["da_n"] or s["ds_n"]:
            print(f"{c} rows: blank KL (gate points) {s['b'] / max(s['b_n'], 1):.4f} over {s['b_n']} pts; "
                  f"duration KL anchored {s['da'] / max(s['da_n'], 1):.4f} over {s['da_n']} pts; "
                  f"duration KL span {s['ds'] / max(s['ds_n'], 1):.4f} over {s['ds_n']} pts")
    print("read: 'duration KL anchored' is what durspan pulls back to stock (v10a left it free — "
          "the 0.2M-to-2M drops); 'duration KL span' stays free by design; blank KL is the "
          "trained-in value. numword blank much above clean blank means the checkpoint's runs "
          "exempted the number-word rows (kl_exempt numwords).")


def microtrain(vol, ultra, run, steps, kl_weight, lr, parts, terms, seed=1):
    import ft_train

    t0 = time.time()
    m = ft_train.train(vol, ultra, f"{run}_pf", mix=parts, max_steps=steps, lr=lr, warmup=2,
                       spec_augment=False, kl_weight=kl_weight,
                       kl_replay_only=True, kl_terms=terms, kl_exempt="", init_from=run,
                       batch_duration=200, seed=seed)
    print(f"micro-train: {steps} steps in {time.time() - t0:.0f}s (exp/{run}_pf)")
    return m.cuda().eval()


def main(vol, ultra, run, kl_weight, tts="", steps=0, lr=1e-4, terms="blank,gnorm,durspan",
         lspc_h=0.25, numwords_h=0.5, fleurs_h=0.25, tts_h=0.25, seed=1):
    import nemo.collections.asr as nemo_asr
    import ft_eval

    t0 = time.time()
    m = ft_eval.load_model(vol, run, ultra)
    stock = nemo_asr.models.ASRModel.restore_from(ultra, map_location="cpu")
    print(f"loaded {run} and stock in {time.time() - t0:.0f}s")

    years, money = probes(vol)
    candidate_parts = [("replay/lspc_train.jsonl", lspc_h), ("replay/lspc_train_numwords.jsonl", numwords_h),
                       ("replay/fleurs_train_multi.jsonl", fleurs_h)]
    parts = []
    for path, h in candidate_parts:
        if os.path.exists(f"{vol}/{path}"):
            parts.append((path, h))
        else:
            print(f"part {path!r} not found on volume, skipped")
    if tts and os.path.exists(f"{vol}/tts/{tts}/train_ok.jsonl"):
        parts.append((f"tts/{tts}/train_ok.jsonl", tts_h))
    else:
        print(f"tts {tts!r}: no train_ok manifest, TTS rows left out")

    decode_summary(m, years, money, f"{run} before")
    t = time.time()
    measure(m, stock, vol, parts, terms=terms)
    print(f"stage A in {time.time() - t:.0f}s")
    if steps:
        t = time.time()
        m2 = microtrain(vol, ultra, run, steps, kl_weight, lr, parts, terms, seed)
        decode_summary(m2, years, money, f"{run}_pf after {steps} steps")
        measure(m2, stock, vol, parts, terms=terms)
        print(f"stage B in {time.time() - t:.0f}s")
