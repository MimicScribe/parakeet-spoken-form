"""torch helpers for the never-reject labeller: losses of whole-row labels, greedy CTC, forced alignment. Tested on random log-probs."""
import labeller_core as C


def losses_fn(lp, blank, vocab=C.VOCAB, batch=64):
    """lp: (T, V) log-probs on CPU/GPU. returns loss_fn(list[str]) -> list[float] (CTC loss per label, sum reduction, zero_infinity)."""
    import torch, torch.nn.functional as F
    T = lp.shape[0]

    def fn(labels):
        out = []
        for i in range(0, len(labels), batch):
            chunk = labels[i:i + batch]
            tg = [C.label_ids(l, vocab) for l in chunk]
            for t in tg:
                if not t:
                    t.append(vocab["|"])
            lps = lp.unsqueeze(1).expand(T, len(tg), lp.shape[1])
            ls = F.ctc_loss(lps, torch.tensor([x for t in tg for x in t], device=lp.device), torch.full((len(tg),), T, dtype=torch.long),
                            torch.tensor([len(t) for t in tg], dtype=torch.long), blank=blank, reduction="none", zero_infinity=True)
            out += [float(x) for x in ls.cpu().tolist()]
        return out
    return fn


def greedy_text(lp, blank, inv):
    g = lp.argmax(-1).tolist(); prev = None; chars = []
    for i in g:
        if i != prev and i != blank:
            chars.append(inv[i])
        prev = i
    return "".join(chars).replace("|", " ").strip()


def align_fn(lp, blank, vocab=C.VOCAB, frame_s=0.02):
    """label -> [[start, end, mean_prob], ...] per CTC word (torchaudio forced alignment); raises if the word count does not match"""
    import numpy as np, torch, torchaudio
    sep = vocab["|"]

    def fn(label):
        nw = C.norm_words(label)
        ids = C.word_ids([w for _, w in nw], vocab)
        tgt = torch.tensor([ids])
        ali, sc = torchaudio.functional.forced_align(lp.unsqueeze(0).cpu(), tgt, blank=blank)
        spans = torchaudio.functional.merge_tokens(ali[0], sc[0].exp(), blank=blank)
        words, cur = [], []
        for s in spans:
            if s.token == sep:
                if cur:
                    words.append(cur); cur = []
            else:
                cur.append(s)
        if cur:
            words.append(cur)
        if len(words) != len(nw):
            raise ValueError("align_len %d != %d" % (len(words), len(nw)))
        return [[round(ch[0].start * frame_s, 3), round(ch[-1].end * frame_s, 3), round(float(np.mean([c.score for c in ch])), 3)] for ch in words]
    return fn


def rms_onset(x, sr=16000):
    """first time (s) the 20 ms frame level stays >= floor + 12 dB for 3 frames (floor = 5th percentile of frame dB); numpy only."""
    import numpy as np
    n = int(0.02 * sr); nf = len(x) // n
    if nf < 5:
        return None
    db = 20 * np.log10(np.maximum(np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1)), 1e-6))
    thr = np.percentile(db, 5) + 12
    above = db >= thr
    for i in range(nf - 2):
        if above[i] and above[i + 1] and above[i + 2]:
            return round(i * 0.02, 3)
    return None
