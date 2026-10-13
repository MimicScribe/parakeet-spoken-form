"""S3 English relabel on GPU: per row, choose the spoken reading of every caption number, then cut window crops.

Checks, all must pass for a number span (else every crop containing it is dropped):
1. stock-value precheck: stock Ultra's decode contains the caption's digits (spans worth >= 10);
2. two scorers agree: full-row TDT NLL under v10pt and CTC NLL under wav2vec2-large-960h-lv60-self pick the same
   candidate, each by >= MARGIN nats (other spans fixed at their current choice; one coordinate pass);
3. absolute fit: the chosen words appear in wav2vec2's own greedy transcript (<= 1 word edit per 3 words).
Rows also need caption-vs-stock CER <= 15%. Crops: wav2vec2 forced alignment of the final label, <= 15 s windows cut in
word gaps, never inside a number span; some preview-shaped 2-8 s crops.

Writes /vol/d1/rows/<shard>.jsonl (every row: decisions, decodes, reasons) and /vol/d1/crops/<shard>.jsonl + wavs.
"""

import json
import os
import random
import re

import labeller as L

MARGIN = 3.0
RECORD_STOCK = True  # validation: also score every candidate under STOCK's decoder+joint
W2V = "facebook/wav2vec2-large-960h-lv60-self"
FILLERS = re.compile(r"\b(?:uh|um|uhm|er|erm|hmm|mm|mhm|uh-huh|mm-hmm)\b", re.I)


def norm_ctc(text: str) -> str:
    """wav2vec2-960h spelling (LibriSpeech): capitals, apostrophes; the digit "oh" is written "O"; AM/PM are letters."""
    t = text.upper().replace("-", " ")
    t = re.sub(r"[^A-Z' ]+", " ", t)
    t = re.sub(r"\bOH\b", "O", t)
    t = re.sub(r"\b([AP])M\b", r"\1 M", t)
    return re.sub(r"\s+", " ", t).strip()


def digit_runs(text: str) -> set[str]:
    return {re.sub(r"[,.]", "", m) for m in re.findall(r"\d[\d,.]*\d|\d", text)}


def cer(a: str, b: str) -> float:
    a = re.sub(r"[^a-z0-9]", "", a.lower())
    b = re.sub(r"[^a-z0-9]", "", b.lower())
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def word_edit(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def fits(span_words: list[str], hyp_words: list[str]) -> bool:
    n = len(span_words)
    if not n:
        return True
    tol = 0  # exact: a caption missing words pushes both scorers toward longer readings (validation 2026-10-13)
    for k in range(max(1, n - tol), n + tol + 1):
        for i in range(0, max(1, len(hyp_words) - k + 1)):
            if word_edit(span_words, hyp_words[i:i + k]) <= tol:
                return True
    return False


def precheck(span: L.Span, stock_digits: set[str], stock_text: str) -> bool:
    if span.kind == "spelled":
        sv = L.spelled_value(span.gold)
        if sv is None:
            return True
        val = sv[0]
        if isinstance(val, int) and val < 10:
            return True
        v = str(val).replace(".", "")
        return v in stock_digits or span.gold.lower() in stock_text.lower()
    ds = [re.sub(r"[,.]", "", d) for d in re.findall(r"\d[\d,.]*\d|\d", span.text)]
    big = [d for d in ds if len(d.lstrip("0")) >= 2]
    return all(d in stock_digits for d in big)


class Scorer:
    def __init__(self, vol: str, ultra: str):
        import torch
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        import ft_eval

        self.torch = torch
        self.stock = ft_eval.load_model(vol, "ultra", ultra)
        self.v10 = ft_eval.load_model(vol, "ultra", ultra)
        sd = torch.load(f"{vol}/exp/v10pt/step1500.pt", map_location="cpu")
        missing, unexpected = self.v10.load_state_dict(sd, strict=False)
        assert not unexpected
        self.v10 = self.v10.cuda().eval()
        durations = list(self.v10.cfg.model_defaults.tdt_durations)
        self.tdt = RNNTLoss(num_classes=self.v10.joint.num_classes_with_blank - 1, reduction="sum", loss_name="tdt",
                            loss_kwargs={"durations": durations, "sigma": 0.0, "omega": 0.0})
        self.proc = Wav2Vec2Processor.from_pretrained(W2V, cache_dir="/tmp/w2v")
        self.w2v = Wav2Vec2ForCTC.from_pretrained(W2V, cache_dir="/tmp/w2v").cuda().eval()
        self.vocab = self.proc.tokenizer.get_vocab()

    def decode(self, model, paths):
        import ft_eval

        return [h["text"] for h in ft_eval.decode(model, paths, batch_size=16)]

    def encode(self, x):
        torch = self.torch
        with torch.no_grad():
            sig = torch.tensor(x, device="cuda")[None]
            feats, fl = self.v10.preprocessor(input_signal=sig, length=torch.tensor([sig.shape[1]], device="cuda"))
            enc, el = self.v10.encoder(audio_signal=feats, length=fl)
            iv = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.cuda()
            lp = self.w2v(iv).logits.float().log_softmax(-1)[0]
        return enc.transpose(1, 2), el, lp

    def tdt_nll(self, f, el, text, model=None):
        torch = self.torch
        model = model or self.v10
        ids = model.tokenizer.text_to_ids(text)
        with torch.no_grad():
            tok = torch.tensor([ids], device="cuda")
            tl = torch.tensor([len(ids)], device="cuda")
            g, _, _ = model.decoder(targets=tok, target_length=tl)
            logits = model.joint.joint(f, g.transpose(1, 2)).float()
            return float(self.tdt(log_probs=logits, targets=tok, input_lengths=el, target_lengths=tl))

    def ctc_ids(self, text):
        t = norm_ctc(text).replace(" ", "|")
        return [self.vocab[c] for c in t if c in self.vocab]

    def ctc_nll(self, lp, text):
        torch = self.torch
        ids = self.ctc_ids(text)
        if not ids or len(ids) > lp.shape[0]:
            return float("inf")
        return float(torch.nn.functional.ctc_loss(lp[:, None], torch.tensor([ids], device="cuda"),
                                                  torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                                                  blank=self.proc.tokenizer.pad_token_id, reduction="sum",
                                                  zero_infinity=False))

    def greedy_w2v(self, lp):
        ids = lp.argmax(-1).tolist()
        return self.proc.decode(ids).replace("|", " ")

    def align(self, lp, text):
        """Word (start s, end s) for each whitespace token of `text` (None when unalignable)."""
        import torchaudio.functional as AF

        torch = self.torch
        toks = text.split()
        chars, owner = [], []
        for i, w in enumerate(toks):
            n = norm_ctc(w).replace(" ", "")
            ids = [self.vocab[c] for c in n if c in self.vocab]
            if chars and ids:
                chars.append(self.vocab["|"])
                owner.append(-1)
            chars += ids
            owner += [i] * len(ids)
        if not chars or len(chars) > lp.shape[0]:
            return None
        ali, scores = AF.forced_align(lp[None].cpu(), torch.tensor([chars]), blank=self.proc.tokenizer.pad_token_id)
        ali, scores = ali[0].tolist(), scores[0].exp().tolist()
        frame_s = 0.02
        times = [[None, None, []] for _ in toks]
        k = -1
        prev = None
        for t, a in enumerate(ali):
            if a == self.proc.tokenizer.pad_token_id:
                prev = None
                continue
            if a != prev:
                k += 1
            prev = a
            if k >= len(owner):
                break
            o = owner[k]
            if o >= 0:
                if times[o][0] is None:
                    times[o][0] = t * frame_s
                times[o][1] = (t + 1) * frame_s
                times[o][2].append(scores[t])
        out = []
        for i, (s, e, sc) in enumerate(times):
            out.append((s, e, sum(sc) / len(sc) if sc else None))
        # tokens with no letters (pure punctuation) borrow their neighbour's time
        for i in range(len(out)):
            if out[i][0] is None:
                j = i - 1
                while j >= 0 and out[j][0] is None:
                    j -= 1
                out[i] = (out[j][1], out[j][1], 1.0) if j >= 0 else (0.0, 0.0, 1.0)
        return out


def choose(sc: Scorer, f, el, lp, text, sp):
    """Coordinate pass over spans; returns (choice, per-span info)."""
    choice = [0] * len(sp)
    info = []
    for i, s in enumerate(sp):
        if len(s.cands) == 1:
            info.append({"n": 1})
            continue
        tdt, ctc, stk = [], [], []
        for c in range(len(s.cands)):
            ch = choice[:]
            ch[i] = c
            t = L.render(text, sp, ch)
            tdt.append(sc.tdt_nll(f, el, t))
            ctc.append(sc.ctc_nll(lp, t))
            stk.append(sc.tdt_nll(f, el, t, sc.stock) if RECORD_STOCK else 0.0)
        bt = min(range(len(tdt)), key=tdt.__getitem__)
        bc = min(range(len(ctc)), key=ctc.__getitem__)
        # Orthographic siblings (same words for the CTC scorer) are not competitors: margins are taken against the best
        # candidate with DIFFERENT words; the TDT pick supplies the orthography.
        key = [norm_ctc(c) for c in s.cands]
        other_t = [tdt[c] for c in range(len(tdt)) if key[c] != key[bt]]
        other_c = [ctc[c] for c in range(len(ctc)) if key[c] != key[bc]]
        mt = (min(other_t) - tdt[bt]) if other_t else float("inf")
        mc = (min(other_c) - ctc[bc]) if other_c else float("inf")
        ok = key[bt] == key[bc] and mt >= MARGIN and mc >= MARGIN
        choice[i] = bt if ok else 0
        info.append({"n": len(s.cands), "tdt": bt, "ctc": bc, "mt": round(mt, 2), "mc": round(mc, 2), "ok": ok,
                     "s_tdt": [round(x, 2) for x in tdt], "s_ctc": [round(x, 2) for x in ctc],
                     "s_stk": [round(x, 2) for x in stk], "keys": [norm_ctc(c) for c in s.cands]})
    return choice, info


def crops(words, times, bad_tok, num_tok, dur, rnd, max_s=15.0):
    """Token index ranges [a, b) to keep: contiguous, <= max_s, no bad token inside, edges in word gaps."""
    n = len(words)
    out = []
    a = 0
    while a < n:
        if bad_tok[a]:
            a += 1
            continue
        start = max(0.0, times[a][0] - 0.15)
        b = a
        while b < n and not bad_tok[b] and times[b][1] - start <= max_s:
            b += 1
        # do not end inside a number span
        while b > a and b < n and num_tok[b] and num_tok[b - 1] and num_tok[b] == num_tok[b - 1]:
            b -= 1
        if b <= a:
            a += 1
            continue
        out.append((a, b))
        a = b
    # preview-shaped 2-8 s crops (~20%): a random sub-window of a kept crop, widened so no number span is split
    extra = []
    for a, b in out:
        if rnd.random() < 0.2 and b - a > 4:
            k = rnd.randint(a, b - 2)
            while k > a and num_tok[k] and num_tok[k] == num_tok[k - 1]:
                k -= 1
            e, want = k + 1, rnd.uniform(2, 8)
            while e < b and times[e][1] - times[k][0] < want:
                e += 1
            while e < b and num_tok[e] and num_tok[e] == num_tok[e - 1]:
                e += 1
            if (k, e) not in out:
                extra.append((k, e))
    return out + extra


def label_shard(vol: str, sc: Scorer, man: str, out_tag: str, keep_audio=True):
    import soundfile as sf

    rows = [json.loads(l) for l in open(man)]
    num = [r for r in rows if r["kind"] != "plain"]
    plain = [r for r in rows if r["kind"] == "plain"]
    os.makedirs(f"{vol}/d1/rows", exist_ok=True)
    os.makedirs(f"{vol}/d1/crops", exist_ok=True)
    wav_dir = f"{vol}/d1/wav/{out_tag}"
    os.makedirs(wav_dir, exist_ok=True)
    paths = [r["audio_filepath"] for r in rows]
    stock_txt = sc.decode(sc.stock, paths)
    v10_txt = sc.decode(sc.v10, paths)
    rnd = random.Random(out_tag)
    out_rows, out_crops = [], []
    for r, st, vt in zip(rows, stock_txt, v10_txt):
        rec = {"utt_id": r["utt_id"], "kind": r["kind"], "dur": r["duration"], "caption": r["text"], "stock": st,
               "v10": vt, "fill_cap": len(FILLERS.findall(r["text"])), "fill_stock": len(FILLERS.findall(st)),
               "fill_v10": len(FILLERS.findall(vt))}
        out_rows.append(rec)
        if r["kind"] == "plain":
            continue
        text = r["text"]
        sp = L.spans(text)
        rec["cer"] = round(cer(text, st), 3)
        if rec["cer"] > 0.15:
            rec["drop"] = "cer"
            continue
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        f, el, lp = sc.encode(x)
        sdig = digit_runs(st)
        pre = [precheck(s, sdig, st) for s in sp]
        choice, info = choose(sc, f, el, lp, text, sp)
        w2v = norm_ctc(sc.greedy_w2v(lp)).split()
        rec["w2v"] = " ".join(w2v)
        spans_out, decided = [], []
        for s, c, inf, p in zip(sp, choice, info, pre):
            fit = fits(norm_ctc(s.cands[c]).split(), w2v)
            ok = p and fit and inf.get("ok", True)
            decided.append(ok)
            spans_out.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": c, "gold": s.gold,
                              "pre": p, "fit": fit, **inf, "decided": ok})
        rec["spans"] = spans_out
        label = L.render(text, sp, choice)
        rec["label"] = label
        if not any(decided):
            rec["drop"] = "no_decided_span"
            continue
        # token-level bookkeeping on the rendered label
        toks = label.split()
        tok_span = [0] * len(toks)
        bad = [False] * len(toks)
        pos = 0
        starts = []
        for t in toks:
            k = label.find(t, pos)
            starts.append(k)
            pos = k + len(t)
        # map span char ranges into the rendered label
        shift, ranges = 0, []
        for s, c in zip(sp, choice):
            a = s.start + shift
            b = a + len(s.cands[c])
            ranges.append((a, b))
            shift += len(s.cands[c]) - (s.end - s.start)
        for j, (a, b) in enumerate(ranges, 1):
            for ti, st0 in enumerate(starts):
                if a <= st0 < b or (st0 < a < st0 + len(toks[ti])):
                    tok_span[ti] = j
                    if not decided[j - 1]:
                        bad[ti] = True
        times = sc.align(lp, label)
        if times is None:
            rec["drop"] = "align"
            continue
        low = [tm[2] is not None and tm[2] < 0.05 for tm in times]
        bad = [b or lw for b, lw in zip(bad, low)]
        for ci, (a, b) in enumerate(crops(toks, times, bad, tok_span, r["duration"], rnd)):
            if not any(tok_span[a:b]):
                continue
            t0 = (times[a - 1][1] + times[a][0]) / 2 if a > 0 else max(0.0, times[a][0] - 0.15)
            t1 = (times[b - 1][1] + times[b][0]) / 2 if b < len(toks) else min(r["duration"], times[b - 1][1] + 0.15)
            if t1 - t0 < 1.0:
                continue
            txt = " ".join(toks[a:b])
            txt = txt[0].upper() + txt[1:]
            seg = x[int(t0 * 16000):int(t1 * 16000)]
            wav = f"{wav_dir}/{r['utt_id']}_{ci}.wav"
            if keep_audio:
                sf.write(wav, seg, 16000, subtype="PCM_16")
            nw = len(txt.split())
            out_crops.append({"audio_filepath": wav, "duration": round(t1 - t0, 3), "text": txt,
                              "src": r["utt_id"], "wps": round(nw / max(t1 - t0, 0.1), 2),
                              "kinds": sorted({sp[j - 1].kind for j in set(tok_span[a:b]) if j}),
                              "video": r["original_audio_id"]})
    with open(f"{vol}/d1/rows/{out_tag}.jsonl", "w") as fo:
        for rec in out_rows:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{vol}/d1/crops/{out_tag}.jsonl", "w") as fo:
        for c in out_crops:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    return {"shard": out_tag, "rows": len(num), "plain": len(plain), "crops": len(out_crops),
            "crop_h": sum(c["duration"] for c in out_crops) / 3600,
            "dropped": sum(1 for r in out_rows if r.get("drop"))}


def decide_text(sc: Scorer, x, text: str, stock_text: str) -> dict:
    """The per-row decision without crops (validation): spans, picks, and why each span is (un)decided."""
    sp = L.spans(text)
    f, el, lp = sc.encode(x)
    pre = [precheck(s, digit_runs(stock_text), stock_text) for s in sp]
    choice, info = choose(sc, f, el, lp, text, sp)
    w2v = norm_ctc(sc.greedy_w2v(lp)).split()
    out = []
    for s, c, inf, p in zip(sp, choice, info, pre):
        fit = fits(norm_ctc(s.cands[c]).split(), w2v)
        out.append({"text": s.text, "kind": s.kind, "pick": s.cands[c], "cands": s.cands, "gold": s.gold, "n": len(s.cands),
                    "pre": p, "fit": fit, **inf, "decided": bool(p and fit and inf.get("ok", True))})
    return {"spans": out, "label": L.render(text, sp, choice), "w2v": " ".join(w2v)}


def validate(vol: str, sc: Scorer, items_path: str, out_path: str):
    """items: [{"id", "wav" (volume-relative), "caption"}] -> per-item decisions."""
    import soundfile as sf

    items = json.load(open(items_path))
    stock = sc.decode(sc.stock, [f"{vol}/{it['wav']}" for it in items])
    res = []
    for it, st in zip(items, stock):
        x, _ = sf.read(f"{vol}/{it['wav']}", dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        r = decide_text(sc, x, it["caption"], st)
        r.update({"id": it["id"], "stock": st, "caption": it["caption"]})
        res.append(r)
    json.dump(res, open(out_path, "w"), indent=0)
    return len(res)
