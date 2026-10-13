"""Tier-1 reading decision (R3.11 P2): which candidate reading of each number span was spoken.

Same rule as English (label_gpu.choose + fits): a per-language wav2vec2 CTC reader scores the whole row with each
candidate in place; the span is DECIDED when the best candidate beats every candidate with different words by
>= MARGIN nats AND its words appear verbatim (accents, case, ß/ss and compound spacing normalised) in the CTC greedy
transcript. Candidates that differ only in spelling the audio cannot show ("één"/"een") form one group.
"""

import re
import unicodedata

import ml_cands as M

MARGIN = 3.0
MODELS = {"de": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
          "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french", "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
          "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"}


def norm_word(w: str) -> str:
    w = w.lower().replace("ß", "ss")
    w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", w)


# European vs Brazilian Portuguese spellings of the same spoken number: one group (the label keeps the app grammar's
# European spelling, the canonical candidate)
DIALECT = {"dezesseis": "dezasseis", "dezessete": "dezassete", "dezenove": "dezanove", "quatorze": "catorze"}


def key(reading: str) -> str:
    """What the audio can distinguish: letters only, no accents, no spaces (compounds may be split either way)."""
    return "".join(DIALECT.get(norm_word(w), norm_word(w)) for w in reading.split())


def ending_only(a: str, b: str) -> bool:
    """Readings that differ only in gender/case endings ("eine"/"einer", "primero"/"primer", "uno"/"una")."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return False
    for x, y in zip(wa, wb):
        x, y = norm_word(x), norm_word(y)
        if x == y:
            continue
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        if n < 2 or max(len(x), len(y)) - n > 2:
            return False
    return True


def _approx_in(k: str, g: str) -> int:
    """Smallest edit distance between k and any substring of g (Sellers)."""
    prev = [0] * (len(g) + 1)
    for i, c in enumerate(k, 1):
        cur = [i] + [0] * len(g)
        for j, d in enumerate(g, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d))
        prev = cur
    return min(prev)


def fits(reading: str, greedy: str, fuzzy: bool = False) -> bool:
    """Exact: the reading's words are a run of greedy words (compound spacing free). Fuzzy (long readings only, no
    ending-only rival): the reading occurs in the space-free greedy text within 1 edit (>= 6 letters) or 2 (>= 12),
    which absorbs the CTC's word-boundary slips ("jahreintausend...") and one-letter spelling noise ("soisante")."""
    if _exact_fit(reading, greedy):
        return True
    k = key(reading)
    if not fuzzy or len(k) < 6:
        return False
    tol = 2 if len(k) >= 12 else 1
    g = "".join(DIALECT.get(norm_word(w), norm_word(w)) for w in greedy.split())
    return _approx_in(k, g) <= tol


def _exact_fit(reading: str, greedy: str) -> bool:
    k = key(reading)
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in greedy.split()]
    ws = [w for w in ws if w]
    for i in range(len(ws)):
        acc = ""
        for j in range(i, len(ws)):
            acc += ws[j]
            if acc == k:
                return True
            if len(acc) >= len(k) or not k.startswith(acc):
                break
    return False


class LangCTC:
    def __init__(self, lang: str, device: str = "cuda"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch, self.lang, self.device = torch, lang, device
        self.proc = Wav2Vec2Processor.from_pretrained(MODELS[lang])
        self.model = Wav2Vec2ForCTC.from_pretrained(MODELS[lang]).to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.delim = self.vocab.get("|")

    def lp(self, x):
        torch = self.torch
        with torch.no_grad():
            iv = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            return self.model(iv).logits.float().log_softmax(-1)[0]

    def greedy(self, lp) -> str:
        return self.proc.decode(lp.argmax(-1).tolist()).lower()

    def ids(self, text: str) -> list[int]:
        out = []
        for w in re.sub(r"[^\w' ]+", " ", text.lower()).split():
            cs = []
            for c in w.replace("ß", "ss" if "ß" not in self.vocab else "ß"):
                if c in self.vocab:
                    cs.append(self.vocab[c])
                else:
                    b = "".join(d for d in unicodedata.normalize("NFD", c) if not unicodedata.combining(d))
                    cs += [self.vocab[d] for d in b if d in self.vocab]
            if cs:
                if out and self.delim is not None:
                    out.append(self.delim)
                out += cs
        return out

    def nll(self, lp, text: str) -> float:
        torch = self.torch
        ids = self.ids(text)
        if not ids or len(ids) > lp.shape[0]:
            return float("inf")
        return float(torch.nn.functional.ctc_loss(
            lp[:, None, :], torch.tensor([ids], device=lp.device), torch.tensor([lp.shape[0]]),
            torch.tensor([len(ids)]), blank=self.blank, reduction="sum", zero_infinity=True))


def decide(ctc: LangCTC, caption: str, lp) -> dict:
    sp = M.spans(caption, ctc.lang)
    greedy = ctc.greedy(lp)
    base = [0] * len(sp)
    out = []
    for j, s in enumerate(sp):
        if not s.cands:
            out.append({"text": s.text, "kind": s.kind, "decided": False, "why": s.note or "no candidates"})
            continue
        scores = []
        for ci in range(len(s.cands)):
            ch = list(base)
            ch[j] = ci
            # an undecidable span stays as its caption digits (the CTC alphabet drops them), the same for every candidate
            text = M.render(caption, [t for t in sp if t.cands], [c for t, c in zip(sp, ch) if t.cands])
            scores.append(ctc.nll(lp, text))
        best = min(range(len(scores)), key=lambda i: (scores[i], i))
        bk = key(s.cands[best])
        best = min(i for i in range(len(s.cands)) if key(s.cands[i]) == bk)  # same audio: keep the grammar's spelling
        rivals = [scores[i] for i in range(len(scores)) if key(s.cands[i]) != bk]
        margin = (min(rivals) - scores[best]) if rivals else float("inf")
        has_end_rival = any(ending_only(s.cands[best], c) and key(c) != bk for c in s.cands)
        fit = fits(s.cands[best], greedy, fuzzy=not has_end_rival)
        # a gender/case-ending rival ("eine"/"einen") can hide in liaison ("einen Euro" ~ "eine neuro"): then the
        # reading must also fit together with the caption's next word
        if fit and has_end_rival:
            nxt = re.match(r"\s+([^\W\d_]+)", caption[s.end:])
            if nxt:
                fit = fits(s.cands[best] + " " + nxt.group(1), greedy)
        out.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": best, "margin": round(margin, 2),
                    "fit": fit, "decided": margin >= MARGIN and fit})
    return {"spans": out, "greedy": greedy}
