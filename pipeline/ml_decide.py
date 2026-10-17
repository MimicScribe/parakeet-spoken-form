"""Tier-1 reading decision (R3.11 P2): which candidate reading of each number span was spoken.

Same rule as English (label_gpu.choose + fits): a per-language wav2vec2 CTC reader scores the whole row with each
candidate in place; the span is DECIDED when the best candidate beats every candidate with different words by
>= MARGIN nats AND its words appear verbatim (accents, case, ß/ss and compound spacing normalised) in the CTC greedy
transcript. Candidates that differ only in spelling the audio cannot show ("één"/"een") form one group.
"""

import os
import re
import unicodedata
from dataclasses import replace

import ml_cands as M

MARGIN = 3.0
# Parliament-speech fixes (2026-10-16; old path: ML_DECIDE_OFF=elision,letters_ext,base_iter,ext_ctx,ext_veto):
#   elision     the caption's neighbour word keeps an apostrophe-joined part ("dell'impatto", "l'article"), so the
#               reading + neighbour test compares the whole spoken word, not "dell"
#   letters_ext a letter-code reading ("ci o due", "esse e di") also matches a transcript that writes the letters
#               ("co due", "s e d")
#   base_iter   every span is scored a second time with the other spans at their first-pass best reading (a range's
#               right end or a version's next group no longer sits at its first, often wrong, connector)
#   ext_ctx     a span the CTC margin / greedy cannot settle is decided when the external transcript holds the reading
#               WITH its caption neighbour(s) and no rival reading in that place, and the CTC best is that same reading
#   ext_veto    a CTC decision is refused when the external transcript holds a different reading in the span's place
#   ext_value   (nl) the transcript writes Dutch numbers as digits: when its digits in the span's place (between the
#               caption neighbours) have the span's value, the CTC alone chooses the wording, with the margins that
#               agreed with a word transcript >= 96% of the time on the other wave-1 languages (VALUE_MARGIN_*)
#   ext_pick    (wave-1; ru / uk always) the transcript decides between readings the CTC cannot tell apart: see
#               EXT_PICK_GAP below
FIXES = ({"elision", "letters_ext", "base_iter", "ext_ctx", "ext_veto", "ext_pick", "ext_value"}
         - set(filter(None, os.environ.get("ML_DECIDE_OFF", "").split(","))))
ALL_FIXES = frozenset(FIXES)  # the start-up set (env switches applied); label_lang resets to it on every call
BASE_PASSES = 3  # base_iter: at most this many scoring passes per row
EXT_CTX_MARGIN = 1.0  # nats: the CTC best must still lead every reading the transcript tells apart
SHORT_MARGIN = 6.0  # rivals whose readings differ by <= 3 letters ("un euro y cincuenta" / "un euro cincuenta"): a short
# word the CTC greedy often drops must win by more (review M5)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]
MODELS = {"de": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
          "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french", "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
          "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
          # ru / uk (2026-10-16): Apache-2.0 wav2vec2 CTC readers (XLSR-53 ru; XLS-R 300M uk, Common Voice 10)
          "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian", "uk": "Yehor/w2v-xls-r-uk"}
# A stronger Dutch reader (Apache-2.0, XLS-R 1B on Common Voice 8: test WER 10.4 / CER 3.0 against 15.7 / 5.4 for the
# xlsr-53 reader above); it writes number words. Try it with label2 --ctc-model (the transcript writes Dutch numbers as
# digits, so the CTC alone chooses the wording there).
STRONG = {"nl": "jonatasgrosman/wav2vec2-xls-r-1b-dutch"}
# Cyrillic letters count as letters only while a ru / uk row is decided (set by decide() and LangCTC); the wave-1
# languages keep the Latin-only keys
CYR = False
_NON_LETTER = {False: re.compile(r"[^a-z]"), True: re.compile(r"[^a-zа-яієґ]")}


def set_script(lang: str) -> None:
    global CYR
    CYR = lang in M.SLAV


def norm_word(w: str) -> str:
    w = w.lower().replace("ß", "ss")
    w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
    return _NON_LETTER[CYR].sub("", w)


# European vs Brazilian Portuguese spellings of the same spoken number: one group (the label keeps the number grammar's
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


def _approx_in(k: str, g: str, ends: set | None = None) -> int:
    """Smallest edit distance between k and any substring of g (Sellers); with `ends`, only substrings that END at one
    of those positions (word boundaries of the greedy text) count."""
    prev = [0] * (len(g) + 1)
    for i, c in enumerate(k, 1):
        cur = [i] + [0] * len(g)
        for j, d in enumerate(g, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d))
        prev = cur
    return min(prev) if ends is None else min(prev[j] for j in ends)


def fits(reading: str, greedy: str, fuzzy: bool = False) -> bool:
    """Exact: the reading's words are a run of greedy words (compound spacing free). Fuzzy (long readings only, no
    ending-only rival): the reading occurs in the space-free greedy text within 1 edit (>= 6 letters) or 2 (>= 12),
    ending at a greedy word boundary; absorbs left word-boundary slips ("jahreintausend") and one-letter noise
    ("soisante") but never a longer spoken number ("twintig" inside "twintigduizend")."""
    if _exact_fit(reading, greedy):
        return True
    k = key(reading)
    if not fuzzy or len(k) < 6:
        return False
    tol = 2 if len(k) >= 12 else 1
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in greedy.split()]
    g, ends, pos = "", set(), 0
    for w in ws:
        g += w
        pos += len(w)
        ends.add(pos)
    # the match may START inside a greedy word (the CTC glues a number to the word before it) but must END at a word
    # boundary: "twintig" must not fit inside a spoken "twintigduizend" (the caption had dropped "duizend": 2026-10-15)
    return bool(ends) and _approx_in(k, g, ends) <= tol


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
    def __init__(self, lang: str, device: str = "cuda", model: str | None = None):
        """model: another wav2vec2 CTC reader for the language than MODELS[lang] (label2 --ctc-model)."""
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch, self.lang, self.device = torch, lang, device
        set_script(lang)
        name = model or MODELS[lang]
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
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

    def word_times(self, lp, text: str) -> list:
        """CTC forced alignment of `text` (a label) against the row: per whitespace token, (t0, t1) seconds from the
        first to the last frame of its letters, or None for a token with no letters in this model's alphabet."""
        import torchaudio.functional as AF

        torch = self.torch
        toks = text.split()
        ids, owner = [], []
        for ti, tok in enumerate(toks):
            for sub in re.sub(r"[^\w' ]+", " ", tok.lower()).split():
                cs = self.ids(sub)
                if not cs:
                    continue
                if ids and self.delim is not None:
                    ids.append(self.delim)
                    owner.append(-1)
                ids += cs
                owner += [ti] * len(cs)
        out = [None] * len(toks)
        # a repeated letter ("ll") needs a blank frame between its two frames (review H1)
        need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        if not ids or need > lp.shape[0]:
            return out
        try:
            al, sc = AF.forced_align(lp[None].float().cpu(), torch.tensor([ids], dtype=torch.int32), blank=self.blank)
            spans = AF.merge_tokens(al[0], sc[0].exp(), blank=self.blank)
        except (RuntimeError, ValueError):
            return out
        if len(spans) != len(ids):
            return out
        sec = 320 / 16000  # wav2vec2 frame stride
        for sp, ti in zip(spans, owner):
            if ti < 0:
                continue
            t0, t1 = sp.start * sec, sp.end * sec
            out[ti] = (t0, t1) if out[ti] is None else (min(out[ti][0], t0), max(out[ti][1], t1))
        return out

    def nll(self, lp, text: str) -> float:
        torch = self.torch
        ids = self.ids(text)
        if not ids or len(ids) > lp.shape[0]:
            return float("inf")
        return float(torch.nn.functional.ctc_loss(
            lp[:, None, :], torch.tensor([ids], device=lp.device), torch.tensor([lp.shape[0]]),
            torch.tensor([len(ids)]), blank=self.blank, reduction="sum", zero_infinity=True))


_SCALE: dict = {}
SCALE_EXTRA = {"es": ("mil", "millón", "millones", "mil millones"), "pt": ("mil", "milhão", "milhões", "bilhão", "bilhões"),
               "fr": ("mille", "million", "millions", "milliard", "milliards"),
               "it": ("mille", "mila", "milione", "milioni", "miliardo", "miliardi"),
               "de": ("tausend", "Million", "Millionen", "Milliarde", "Milliarden"),
               "nl": ("duizend", "miljoen", "miljard"),
               "ru": ("тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов", "миллиард", "миллиарда",
                      "миллиардов", "процент", "процента", "процентов"),
               "uk": ("тисяча", "тисячі", "тисяч", "мільйон", "мільйони", "мільйонів", "мільярд", "мільярди",
                      "мільярдів", "відсоток", "відсотки", "відсотків", "відсотка", "процент", "проценти", "процентів")}
PCT_SPOKEN = ("por ciento", "por cento", "pour cent", "per cento", "Prozent", "procent", "percento")


def scale_words(lang: str) -> list[str]:
    """Words that turn "ocho" into a bigger number or a percentage when they follow it: the grammar's own words for
    1,000 / 10^6 / 10^9 (last word, so "mil", "mille", "tausend", "duizend", "milioni", ...) and the percent words."""
    if lang not in _SCALE:
        ws = set()
        for n in (1000, 10**6, 2 * 10**6, 10**9, 2 * 10**9):
            try:
                w = M.g(lang, "card", n)
            except M.Missing:
                continue
            last = w[-1]
            ws.add(re.sub(r"^(?:ein|eine|un|uno|um|één|een)", "", last) or last)
        # plural / bound forms the grammar's 1,000 and 10^6 entries do not show on their own ("duemila", "millones")
        ws |= set(SCALE_EXTRA.get(lang, ()))
        _SCALE[lang] = sorted(ws) + list(PCT_SPOKEN)
    return _SCALE[lang]


def context_fits(caption: str, s, reading: str, greedy: str, lang: str) -> bool:
    """Review B1: a reading must also fit together with the caption's NEXT word (fuzzy), so a caption that dropped
    "mil" / "tausend" / "per cento" ("8 puestos" spoken "ocho mil puestos") is not decided; with no next word, the
    previous word must fit with it instead. And the greedy text must not continue the reading with a scale / percent
    word the caption does not have."""
    r = reading.strip()
    nw, pw = next_word(caption, s), prev_word(caption, s)
    if nw:
        if not fits(r + " " + nw, greedy, fuzzy=True):
            return False
    elif pw and not fits(pw + " " + r, greedy, fuzzy=True):
        return False
    return not _scale_follows(nw, r, greedy, lang)


def _scale_follows(nw, r: str, text: str, lang: str) -> bool:
    """`text` continues the reading with a scale / percent word the caption does not have ("ocho" + "mil")."""
    for w in scale_words(lang):
        if (nw or "").lower() == w.split()[0].lower() or key(w) in key(r)[-len(key(w)):]:
            continue
        if fits(r + " " + w, text):
            return True
    return False


_APOS = "['’]"


def next_word(caption: str, s) -> str | None:
    """The caption word right after span s (letters only; None when a digit / nothing follows). With the elision fix
    the word keeps an apostrophe-joined part ("22% dell'impatto" -> "dell'impatto", not "dell")."""
    return _next_in(caption[s.end:])


def prev_word(caption: str, s) -> str | None:
    """The caption word right before span s; with the elision fix an elided article glued to the number counts
    ("all'80%" -> "all'", "l'NO2" -> "l'")."""
    return _prev_in(caption[:s.start])


def _next_in(right: str) -> str | None:
    pat = r"[\s,;:]*([^\W\d_]+(?:" + _APOS + r"[^\W\d_]+)?)" if "elision" in FIXES else r"[\s,;:]*([^\W\d_]+)"
    m = re.match(pat, right)
    return m.group(1) if m else None


def _prev_in(left: str) -> str | None:
    if "elision" in FIXES:
        m = re.search(r"((?:[^\W\d_]+" + _APOS + r")?[^\W\d_]+|[^\W\d_]+" + _APOS + r")[\s,;:]*$", left)
    else:
        m = re.search(r"([^\W\d_]+)[\s,;:]*$", left)
    return m.group(1) if m else None


def ext_agrees(cands: list, best: int, ext: str, kind: str = "", lang: str = "") -> bool:
    """The second recognizer heard the best reading as WHOLE words (so "twintig" never matches inside a spoken
    "twintigduizend": the caption-value error ml_decide.fits guards against) and no rival anywhere. With the
    letters_ext fix (kind alnum / amp, lang given) a reading also matches the transcript's letter spelling ("ci o due"
    = "co due"); rivals the transcript writes the same way cannot refuse it."""
    pk = key(cands[best])
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in (ext or "").split()]
    g = "".join(ws)
    pf = ext_forms(cands[best], kind, lang)
    if pf == {pk}:
        whole = any("".join(ws[i:j]) == pk for i in range(len(ws)) for j in range(i + 1, min(len(ws), i + 12) + 1))
        return whole and not any(key(c) in g and key(c) not in pk for c in cands if key(c) != pk)
    if not any(_whole_in(f, ws) for f in pf):
        return False
    for c in cands:
        cf = ext_forms(c, kind, lang)
        if key(c) == pk or cf & pf:
            continue
        if any(f in g and not any(f in p for p in pf) for f in cf):
            return False
    return True


def _whole_in(k: str, ws: list) -> bool:
    """k is the letters of a run of consecutive whole words of ws (compound spacing free)."""
    if not k:
        return False
    for i in range(len(ws)):
        acc = ""
        for j in range(i, min(len(ws), i + 16)):
            acc += ws[j]
            if acc == k:
                return True
            if len(acc) >= len(k) or not k.startswith(acc):
                break
    return False


def _seq_in(parts: list, ws: list, tail: str = "") -> bool:
    """parts (then tail) are, in order, the letters of consecutive runs of whole words of ws: every part starts and
    ends at a word boundary (a part's own words may be spaced either way)."""
    parts = [p for p in parts + [tail] if p]
    for i in range(len(ws)):
        j, ok = i, True
        for p in parts:
            acc = ""
            while j < len(ws) and len(acc) < len(p) and p.startswith(acc + ws[j]):
                acc += ws[j]
                j += 1
            if acc != p:
                ok = False
                break
        if ok:
            return True
    return False


_LNAMES: dict = {}


def _letter_names(lang: str) -> dict:
    """Spoken letter name (word tuple, key form) -> letter, the language's names and the English ones."""
    if lang not in _LNAMES:
        m = {}
        for src in (M.LETTERS.get(lang, {}), {k: [v] for k, v in M.EN_LETTERS.items()}):
            for let, names in src.items():
                for n in names:
                    m.setdefault(tuple(norm_word(w) for w in n.split()), let)
        _LNAMES[lang] = m
    return _LNAMES[lang]


def ext_forms(cand: str, kind: str = "", lang: str = "") -> set:
    """key() forms a verbatim transcript may write this reading as: the reading itself and, for a letter code
    (letters_ext; kinds alnum / amp), the reading with every letter name written as its letter ("ci o due" -> "codue",
    "esse and di" -> "sandd")."""
    out = {key(cand)}
    if "letters_ext" in FIXES and kind in ("alnum", "amp") and lang:
        names = _letter_names(lang)
        ws = [norm_word(w) for w in cand.split()]
        res, i = [], 0
        while i < len(ws):
            for n in (2, 1):
                t = tuple(ws[i:i + n])
                if len(t) == n and t in names:
                    res.append(names[t])
                    i += n
                    break
            else:
                res.append(DIALECT.get(ws[i], ws[i]))
                i += 1
        out.add("".join(res))
    return out


def letter_seq(cand: str, lang: str) -> tuple:
    """The reading as letters and words: ("L", letter) for a spoken letter name, ("W", word) otherwise. Two readings
    with the same sequence spell the same letters with different names ("ze o zwei" / "ce o zwei", "bee" / "be"): a
    spelling choice, not a different code ("COP ventuno" is a word, "ci o pi ventuno" three letters)."""
    names = _letter_names(lang)
    ws = [norm_word(w) for w in cand.split()]
    res, i = [], 0
    while i < len(ws):
        for n in (2, 1):
            t = tuple(ws[i:i + n])
            if len(t) == n and t in names:
                res.append(("L", names[t]))
                i += n
                break
        else:
            res.append(("W", DIALECT.get(ws[i], ws[i])))
            i += 1
    return tuple(res)


def ext_context(left: str, right: str, s, best: int, ext: str, lang: str) -> bool:
    """ext_ctx fix: the transcript holds the best reading together with its neighbour word(s) as whole words
    ("ventidue per cento dell'impatto", "dieci e trenta dobbiamo", "cop ventuno la"), no reading it tells apart from
    the best sits in that place, and no scale / percent word follows a reading that ends the caption. left / right =
    the row text before / after the span with the OTHER spans rendered at their current readings, so a range's two
    ends are checked together ("uno due milioni" is not "uno o due milioni")."""
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in (ext or "").split()]
    pw, nw = _prev_in(left), _next_in(right)
    if not ws or not (pw or nw):
        return False
    pk, nk = (key(pw) if pw else ""), (key(nw) if nw else "")
    glue = bool(pw) and pw[-1] in "'’"  # "all'" + "ottanta": the transcript writes one word

    def heard(c, strict):
        for f in ext_forms(c, s.kind, lang):
            if strict and _seq_in([pk + f] if glue else [pk, f], ws, nk):
                return True
            if not strict and _whole_in(pk + f + nk, ws):
                return True
        return False

    if not heard(s.cands[best], False):
        return False
    strict = heard(s.cands[best], True)
    bf = ext_forms(s.cands[best], s.kind, lang)
    for c in s.cands:
        # a rival refuses when it is heard as firmly as the best: word for word, or (best only by letters, word
        # boundaries free) at all ("di un o" / "di uno" read the same letters)
        if not (ext_forms(c, s.kind, lang) & bf) and heard(c, strict):
            return False
    return bool(nw) or not _scale_follows(None, s.cands[best].strip(), (ext or "").lower(), lang)


def ext_contradicts(left: str, right: str, s, best: int, ext: str, lang: str) -> bool:
    """ext_veto fix: the transcript holds, word for word in the span's place, a reading it tells apart from the CTC
    best, and not the best ("articolo tre punto uno" where the CTC best is "tre uno"): the CTC decision is refused.
    2026-10-16 Italian probe: xlsr drops a short spoken "punto" / "virgola" with 9-80 nats to spare."""
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in (ext or "").split()]
    pw, nw = _prev_in(left), _next_in(right)
    if not ws or not (pw or nw):
        return False
    pk, nk = (key(pw) if pw else ""), (key(nw) if nw else "")
    glue = bool(pw) and pw[-1] in "'’"
    # with both neighbours, and with each one alone (the transcript may word the other differently: "dell'articolo"
    # for the caption's "l'articolo")
    sides = [(a, b) for a, b in ((pk, nk), (pk, ""), ("", nk)) if a or b]
    bf = ext_forms(s.cands[best], s.kind, lang)
    if any(_whole_in(a + f + b, ws) for a, b in sides for f in bf):
        return False
    for c in s.cands:
        cf = ext_forms(c, s.kind, lang)
        if not (cf & bf) and any(_seq_in([a + f] if glue and a else [a, f], ws, b) for a, b in sides for f in cf):
            return True
    return False


def _sides(caption: str, sp: list, base: list, j: int) -> tuple[str, str]:
    """The row text left and right of span j, every other span with candidates at its base reading."""
    left, right, pos = [], [], 0
    for k, (t, c) in enumerate(zip(sp, base)):
        if k == j:
            left.append(caption[pos:t.start])
            pos = t.end
            continue
        seg = (left if k < j else right)
        seg.append(caption[pos:t.start])
        seg.append(t.cands[c] if t.cands else caption[t.start:t.end])
        pos = t.end
    right.append(caption[pos:])
    return "".join(left), "".join(right)


def decide(ctc: LangCTC, caption: str, lp, ext: str | None = None) -> dict:
    """ext (optional): a second recognizer's verbatim transcript of the same audio. A span whose margin clears the bar
    but whose reading the noisy CTC greedy text misspells is decided when ext agrees (ext_agrees); marked by="ext".
    With the ext_ctx fix a span is also decided when ext holds its CTC-best reading in its caption place
    (ext_context) and the CTC best leads every reading ext tells apart by EXT_CTX_MARGIN; marked by="ext_ctx".
    base_iter: a second pass scores every span with the other spans at their first-pass best."""
    set_script(ctc.lang)
    sp = spans_for(caption, ctc.lang, ext)
    greedy = ctc.greedy(lp)
    base = [0] * len(sp)
    out = _decide_pass(ctc, caption, lp, ext, sp, greedy, base)
    if "base_iter" in FIXES and sum(1 for t in sp if t.cands) > 1:
        for _ in range(BASE_PASSES - 1):  # until the picks stop moving ("1.1.1": the groups settle in turn)
            base2 = [d.get("pick", 0) for d in out]
            if base2 == base:
                break
            base = base2
            out = _decide_pass(ctc, caption, lp, ext, sp, greedy, base)
    return {"spans": out, "greedy": greedy}


def spans_for(caption: str, lang: str, ext: str | None) -> list:
    """The spans and readings decide() weighs. The parliament candidate additions (ml_cands.CAND_FIXES) are offered only
    with an external transcript: the xlsr CTC prefers a reading without a short spoken word ("tre uno" for a spoken
    "tre punto uno", "uno comma sette" for "uno virgola sette"), and only the transcript (ext_veto, ext_ctx) catches
    that. Without one the readings are the old ones."""
    if ext or not M.CAND_FIXES:  # an empty transcript (silence) is no transcript
        return M.spans(caption, lang)
    saved = set(M.CAND_FIXES)
    M.CAND_FIXES.clear()
    try:
        return M.spans(caption, lang)
    finally:
        M.CAND_FIXES.update(saved)


EXT_PICK_GAP = 10.0  # ext_pick: the CTC's score for the transcript's reading at most this far from its best
# ext_pick, wave-1 languages (2026-10-16 parliament batch): the xlsr CTC drops a short spoken word with 10-90 nats to
# spare ("otto quattro" for a spoken "otto punto quattro", "sechzehn dreißig" for "sechzehn Uhr dreißig", "once
# treinta" for "once y treinta"). When the transcript's reading is the CTC best plus whole words, it may trail by up to
# this much.
EXT_PICK_WORD_GAP = 100.0


def _adds_words(a: str, b: str) -> bool:
    """b is a with one or more whole words added (a's words, in order, are a subsequence of b's, and b is longer)."""
    wa, wb = [key(w) for w in a.split()], [key(w) for w in b.split()]
    wa, wb = [w for w in wa if w], [w for w in wb if w]
    if not wa or len(wb) <= len(wa):
        return False
    it = iter(wb)
    return all(any(w == x for x in it) for w in wa)


_ONE_SCALE = re.compile(r"(?:ein|een)(hundert|tausend|honderd|duizend)")


def _one_scale_only(a: str, b: str) -> bool:
    """The readings differ only in a spoken "one" before hundred / thousand ("hundertdrei" / "einhundertdrei"). The
    transcript writes "einhundert" 148 times to "hundert" 43 on the German parliament batch, the CTC greedy text the
    reverse (38 / 151): neither tells the two apart reliably, so ext_pick leaves them alone."""
    return key(a) != key(b) and _ONE_SCALE.sub(r"\1", key(a)) == _ONE_SCALE.sub(r"\1", key(b))


def _count_whole(k: str, ws: list) -> int:
    """How many times k is the letters of a run of consecutive whole words of ws."""
    n = 0
    for i in range(len(ws)):
        acc = ""
        for j in range(i, min(len(ws), i + 16)):
            acc += ws[j]
            if acc == k:
                n += 1
                break
            if len(acc) >= len(k) or not k.startswith(acc):
                break
    return n


def _ext_once_each(sp: list, s, reading: str, ext: str) -> bool:
    """The transcript holds the reading at least as often as the caption has spans that can be read so: ext_context
    looks for the reading anywhere in the transcript, so a caption number the audio lacks (a caption that runs past the
    audio's edge) must not borrow a second occurrence of the same number."""
    k = key(reading)
    ws = [DIALECT.get(norm_word(w), norm_word(w)) for w in (ext or "").split()]
    need = sum(1 for t in sp if t.cands and any(key(c) == k for c in t.cands))
    return _count_whole(k, ws) >= max(need, 1)


# ---------------------------------------------------------------- ext_value (Dutch: a digit transcript)
VALUE_LANGS = ("nl",)
# CTC margins (nats) for a wording the transcript cannot show. Calibrated on de / es / fr / it / pt spans where the word
# transcript held exactly one reading in place (2026-10-16 parliament batch): the CTC best agreed with it 96-100% when
# it was the LONGER reading and led a rival within 3 letters by >= 10 nats, or any other rival by >= 40; when the CTC
# best was the SHORTER reading (the xlsr drops short words and endings) it agreed only 73-88% below 40 nats.
VALUE_MARGIN_NEAR = 10.0
VALUE_MARGIN = 40.0
_ORD_SFX = re.compile(r"(?<=\d)(?:ste|de|e)$")


def _vtokens(text: str) -> list:
    """Tokens for the value test: ("N", digits, pct, ordinal suffix) for a number token (thousands dots removed,
    ":" and "u" clock separators read as "."), ("W", key) for a word; digit runs joined by "-" split apart."""
    out = []
    for raw in re.split(r"\s+", text or ""):
        for t in re.split(r"(?<=\d)[-–](?=\d)|(?<=\d)-(?=[^\W\d_])|(?<=[^\W\d_])-(?=\d)", raw):
            t = t.strip(".,;:!?()[]\"'«»“”‘’").lower()
            if not t:
                continue
            if re.search(r"\d", t):
                pct = t.endswith("%")
                t = t.rstrip("%").strip()
                sfx = bool(_ORD_SFX.search(t))
                t = _ORD_SFX.sub("", t)
                t = re.sub(r"(?<=\d)[:u](?=\d\d\b)", ".", t)
                t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", t)
                out.append(("N", t, pct, sfx))
            else:
                k = norm_word(t)
                if k:
                    out.append(("W", k))
    return out


def _vsame(a, b) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def ext_value(caption: str, s, ext: str, lang: str) -> dict | None:
    """ext_value fix: the transcript holds the span's number as digits in its caption place: the caption word / number
    before it and the one after it (when the caption has them; at least one) sit right beside it, after at most two
    words of the reading itself ("15 procent", "12 uur"). Returns {"ord": True when the transcript marks an ordinal}
    or None. The value must be the caption's: same digits, same decimal separator, a percent sign or word when the
    caption has one; and the transcript must hold that number at least as often as the caption."""
    sv = _vtokens(s.text)
    sv = [t for t in sv if t[0] == "N"]
    if len(sv) != 1:
        return None
    sv = sv[0]
    left, right = _vtokens(caption[:s.start]), _vtokens(caption[s.end:])
    pw, nw = (left[-1] if left else None), (right[0] if right else None)
    if not (pw or nw):
        return None
    ev = _vtokens(ext)
    words = {norm_word(w) for c in s.cands for w in c.split()}
    pct_word = {norm_word(w) for p in PCT_SPOKEN for w in p.split()}
    hits = []
    for i, t in enumerate(ev):
        if t[0] != "N" or t[1] != sv[1]:
            continue
        if pw and not (i > 0 and _vsame(ev[i - 1], pw)):
            continue
        j, said_pct = i + 1, t[2]
        while j < len(ev) and j - i <= 2 and ev[j][0] == "W" and ev[j][1] in words and not (nw and _vsame(ev[j], nw)):
            said_pct = said_pct or ev[j][1] in pct_word
            j += 1
        if nw and not (j < len(ev) and _vsame(ev[j], nw)):
            continue
        if not nw and j < len(ev) and ev[j][0] == "W" and any(ev[j][1] == key(w) for w in scale_words(lang)):
            continue  # the audio goes on with a scale / percent word the caption lacks
        if said_pct != (s.kind == "pct" or sv[2]):
            continue
        hits.append(t)
    if not hits:
        return None
    need = sum(1 for t in _vtokens(caption) if t[0] == "N" and t[1] == sv[1])
    if sum(1 for t in ev if t[0] == "N" and t[1] == sv[1]) < need:
        return None
    return {"ord": any(h[3] for h in hits)}


def _value_pick(s, scores, margin_del: float, info: dict, lang: str):
    """The reading ext_value decides, or None: among the readings the transcript's digits allow (an ordinal suffix
    rules out the cardinal readings of an ordinal span; a clock keeps its caption hour), the CTC best must lead every
    other wording by VALUE_MARGIN_NEAR (a rival within 3 letters that is shorter than the best) or VALUE_MARGIN; a span
    with one reading needs the CTC to prefer it to the row without it (MARGIN)."""
    idx = list(range(len(s.cands)))
    if s.kind == "ord" and info["ord"]:
        idx = [i for i in idx if not s.cands[i].rstrip().endswith(".")]
    if s.kind == "clock":
        m = re.match(r"\s*(\d{1,2})", s.text)
        try:
            hw = key(" ".join(M.g(lang, "card", int(m.group(1))))) if m else None
        except M.Missing:
            hw = None
        idx = [i for i in idx if hw and key(s.cands[i].split()[0]) == hw]
    if not idx:
        return None
    b = min(idx, key=lambda i: (scores[i], i))
    bk = key(s.cands[b])
    b = min(i for i in idx if key(s.cands[i]) == bk)
    riv = [i for i in idx if key(s.cands[i]) != bk]
    if not riv:
        return b if (len(set(map(key, s.cands))) > 1 or margin_del >= MARGIN) else None
    for i in riv:
        rk = key(s.cands[i])
        near = _lev(rk, bk) <= 3 and len(rk) < len(bk)
        if scores[i] - scores[b] < (VALUE_MARGIN_NEAR if near else VALUE_MARGIN):
            return None
    return b


def _decide_pass(ctc, caption, lp, ext, sp, greedy, base) -> list:
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
        rivals = [i for i in range(len(scores)) if key(s.cands[i]) != bk]
        margin = (min(scores[i] for i in rivals) - scores[best]) if rivals else float("inf")
        margin_ok = all(scores[i] - scores[best] >= (SHORT_MARGIN if _lev(key(s.cands[i]), bk) <= 3 else MARGIN)
                        for i in rivals)
        if not rivals:
            # a single reading (one phone digit, a fixed currency phrase) is otherwise decided by `fit` alone, which a
            # word elsewhere in the greedy text satisfies (review M3): it must beat the same row WITHOUT the span, and
            # sit next to its caption neighbour in the greedy text
            sp_del = [replace(t, cands=[""]) if t is s else t for t in sp if t.cands]
            ch = [0 if t is s else c for t, c in zip(sp, base) if t.cands]
            text_del = re.sub(r"  +", " ", M.render(caption, sp_del, ch))
            margin = ctc.nll(lp, text_del) - scores[best]
            margin_ok = margin >= MARGIN
        has_end_rival = any(ending_only(s.cands[best], c) and key(c) != bk for c in s.cands)
        fit = fits(s.cands[best], greedy, fuzzy=not has_end_rival)
        # a gender/case-ending rival ("eine"/"einen") can hide in liaison ("einen Euro" ~ "eine neuro"): then the
        # reading must also fit together with the caption's next word
        if fit and has_end_rival:
            nxt = re.match(r"\s+([^\W\d_]+)", caption[s.end:])
            nw = next_word(caption, s) if "elision" in FIXES and nxt else (nxt.group(1) if nxt else None)
            if nw:
                fit = fits(s.cands[best] + " " + nw, greedy)
        if fit:
            fit = context_fits(caption, s, s.cands[best], greedy, ctc.lang)
        by = "ctc"
        if (margin_ok and fit and ext and "ext_veto" in FIXES
                and ext_contradicts(*_sides(caption, sp, base, j), s, best, ext, ctc.lang)):
            fit, by = False, "ext_veto"
        margin_x = margin_ok
        if rivals and "letters_ext" in FIXES and s.kind in ("alnum", "amp") and ext:
            # with a transcript, letter-name spelling variants of the best ("ze"/"ce") need no margin over it
            ls = letter_seq(s.cands[best], ctc.lang)
            margin_x = all(scores[i] - scores[best] >= (SHORT_MARGIN if _lev(key(s.cands[i]), bk) <= 3 else MARGIN)
                           for i in rivals if letter_seq(s.cands[i], ctc.lang) != ls)
        # the scale/percent guard (context_fits) is re-run against the second transcript, so "8 puestos"
        # spoken "ocho mil puestos" is not decided "ocho"
        sides = _sides(caption, sp, base, j) if ext else None
        if (margin_x and not fit and by != "ext_veto" and ext is not None
                and ext_agrees(s.cands, best, ext, s.kind, ctc.lang)
                and (context_fits(caption, s, s.cands[best], ext.lower(), ctc.lang)
                     or ("letters_ext" in FIXES and s.kind in ("alnum", "amp")
                         and ext_context(*sides, s, best, ext, ctc.lang)))):
            fit, margin_ok, by = True, True, "ext"
        if (not (margin_ok and fit) and by != "ext_veto" and "ext_ctx" in FIXES and ext
                and ext_context(*sides, s, best, ext, ctc.lang)):
            bf = ext_forms(s.cands[best], s.kind, ctc.lang)
            if rivals:
                ls = letter_seq(s.cands[best], ctc.lang) if s.kind in ("alnum", "amp") else None
                ok = all(scores[i] - scores[best] >= (
                    EXT_CTX_MARGIN if not (ext_forms(s.cands[i], s.kind, ctc.lang) & bf)
                    else 0.0 if ls is not None and letter_seq(s.cands[i], ctc.lang) == ls
                    else SHORT_MARGIN if _lev(key(s.cands[i]), bk) <= 3 else MARGIN) for i in rivals)
            else:
                ok = margin >= EXT_CTX_MARGIN
            if ok:
                fit, margin_ok, by = True, True, "ext_ctx"
        slav = ctc.lang in M.SLAV
        if (not (margin_ok and fit) and ext and sides is not None
                and (by != "ext_veto" if slav else ("ext_pick" in FIXES and rivals))):
            # ru / uk: the xlsr CTC is weak on endings ("восемнадцать" for a spoken "восемнадцатый стих"). The second
            # transcript decides when exactly one reading sits in its place as whole words with the neighbours, and the
            # CTC does not strongly prefer another (EXT_PICK_GAP); the ending guard below still needs its word forms.
            # Wave-1 (ext_pick fix): the same, also over an ext_veto, and up to EXT_PICK_WORD_GAP when the transcript's
            # reading only adds whole words to the CTC best; the transcript must hold that reading once per caption span
            # that can be read so (_ext_once_each). A wave-1 span with no rival reading is left to ext_ctx: its CTC
            # test is the row without it, and a negative margin there meant a caption number the trimmed audio lacks.
            fit_keys = {}
            for i in range(len(s.cands)):
                if ext_context(*sides, s, i, ext, ctc.lang):
                    fit_keys.setdefault(key(s.cands[i]), i)
            if len(fit_keys) == 1:
                i = next(iter(fit_keys.values()))
                gap = scores[i] - scores[best]
                ok = gap <= EXT_PICK_GAP
                if not slav:
                    ok = ((ok or (gap <= EXT_PICK_WORD_GAP and _adds_words(s.cands[best], s.cands[i])))
                          and not _one_scale_only(s.cands[best], s.cands[i]) and _ext_once_each(sp, s, s.cands[i], ext))
                if ok:
                    best, bk = i, key(s.cands[i])
                    fit, margin_ok, by = True, True, "ext_pick"
        if (not (margin_ok and fit) and by != "ext_veto" and ext and "ext_value" in FIXES and ctc.lang in VALUE_LANGS
                and s.kind not in ("alnum", "amp", "code", "phone", "roman")):
            info = ext_value(caption, s, ext, ctc.lang)
            v = _value_pick(s, scores, margin, info, ctc.lang) if info is not None else None
            if v is not None:
                best, bk = v, key(s.cands[v])
                fit, margin_ok, by = True, True, "ext_value"
        if by in ("ext", "ext_ctx") and "letters_ext" in FIXES and s.kind in ("alnum", "amp"):
            # letter-name spelling variants were not told apart: the label keeps the grammar's first spelling
            ls = letter_seq(s.cands[best], ctc.lang)
            best = min(i for i in range(len(s.cands)) if letter_seq(s.cands[i], ctc.lang) == ls)
        if (ctc.lang in M.SLAV and margin_ok and fit
                and any(ending_only(s.cands[best], c) and key(c) != bk for c in s.cands)
                and (ext is None or not fits(s.cands[best], ext.lower()))):
            # ru / uk: a case / gender ending is decided only when the external transcript has the same word forms
            # (the xlsr greedy text cannot be trusted on a one-letter ending: "восьмі" / "восьми")
            fit, by = False, "ext_missing"
        if (fit and margin_ok and by not in ("ext_ctx", "ext_pick") and _year_last_two(ctc.lang, s.text, s.cands[best])
                and not (ext and ext_context(*sides, s, best, ext, ctc.lang))):
            # "einundneunzig" for a caption 1991 drops "neunzehnhundert": the xlsr reader drops short words, so the
            # short reading is taken only where the transcript holds it in place
            fit, by = False, "year_short_unconfirmed"
        out.append({"text": s.text, "kind": s.kind, "cands": s.cands, "pick": best, "margin": round(margin, 2),
                    "fit": fit, "decided": margin_ok and fit, "by": by, "scores": [round(x, 2) for x in scores]})
    return out


def _year_last_two(lang: str, text: str, reading: str) -> bool:
    """Is `reading` the last-two-digits reading of a 1900-2099 year caption?"""
    if not re.fullmatch(r"(19|20)\d\d", text.strip()):
        return False
    lo = int(text.strip()) % 100
    try:
        short = " ".join(M.g(lang, "card", lo))
    except (KeyError, TypeError):
        return False
    return lo >= 10 and key(reading) == key(short)
