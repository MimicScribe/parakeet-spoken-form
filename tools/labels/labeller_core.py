#!/usr/bin/env python3
"""Pure logic of the NEVER-REJECT candidate labeller (no torch import, so it is testable offline).
A row is a label plus its number spans; each span carries candidate readings of the same value (spanfind.py). The labeller scores every
candidate reading against the audio with a CTC model and picks the best one; it never drops a row:
  * every input row comes back with status in {scored, undecided, cer_high, align_fail, error}
  * per span the TOP-2 candidate losses are kept, and the margin is computed over ALL candidates, including same-key alternates (oh / zero, double X): no canon_key collapse
    (the collapsed margin is kept as margin_x for comparison); tie flag; current-reading loss and rank
  * row-level alternative labels (e.g. the label with a spoken filler inside the number) are scored against the chosen label
  * candidates come from words -> value -> readings (spanfind.py: commas stripped, +-1 repeat-count variants), so no digit caption is needed
Status priority: error > align_fail > cer_high > undecided > scored.  Downstream rule (calibrate_margin.py): below the calibrated margin keep the current label, mark unconfirmed."""
import re, unicodedata

VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17,
         "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}
MARGIN = 1.0          # old rejection threshold; here it only sets status 'undecided' (nothing is dropped)
TIE_EPS = 0.25        # nats: top-2 closer than this = tie
MAX_CER = 0.22        # status 'cer_high' (0.6 when a span is a coded cell)


def norm_words(label):
    """label tokens -> [(token_index, CTC word)]; CTC spelling rules: oh -> O, AM / PM -> A M (two CTC words, one token)."""
    out = []
    for i, tok in enumerate(label.split()):
        t = unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode().upper()
        t = re.sub(r"[^A-Z']", "", t)
        if not t:
            continue
        if t == "OH":
            t = "O"
        if t in ("AM", "PM"):
            out.append((i, t[0])); out.append((i, "M")); continue
        out.append((i, t))
    return out


def canon_key(c):
    """the OLD collapse (oh == zero, 'double x' == 'x x', spelled letters joined). Only used for margin_x and the same-key flag."""
    c = re.sub(r"\b(?:oh|zero)\b", "0", c); c = re.sub(r"\bdouble (\w+)", r"\1 \1", c)
    c = re.sub(r"\b([A-Z]) (?=[A-Z]\b)", r"\1", c); c = re.sub(r"\b([A-Z]) ([A-Z])$", r"\1\2", c)
    return c


def word_ids(words, vocab=VOCAB):
    s = []
    for k, w in enumerate(words):
        if k:
            s.append(vocab["|"])
        s += [vocab[c] for c in w]
    return s


def label_ids(label, vocab=VOCAB):
    return word_ids([w for _, w in norm_words(label)], vocab)


def build_label(label, spans, choice):
    """replace each span's [char_start, char_end) by its chosen candidate, right to left so offsets stay valid. Punctuation after the span survives."""
    out = label
    for k in sorted(range(len(spans)), key=lambda k: -spans[k]["char_start"]):
        sp = spans[k]
        out = out[:sp["char_start"]] + sp["cands"][choice[k]] + out[sp["char_end"]:]
    return out


def edit_dp(a, b):
    """reference Levenshtein (O(n*m) Python loops); kept for the tests"""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit(a, b):
    """Levenshtein distance of two sequences of hashables (strings or word lists): Myers / Hyyro bit-vector, one big-int op chain per symbol
    of b instead of an O(n*m) Python loop (~10 ms -> <1 ms per row; the GIL is shared with the GPU feeding thread). Same integer as edit_dp."""
    m = len(a)
    if m == 0:
        return len(b)
    if len(b) == 0:
        return m
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    top = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def cer_of(label, hyp):
    ref = " ".join(w for _, w in norm_words(label))
    return edit(ref, hyp) / max(1, len(ref))


def span_result(sp, losses, choice_idx):
    """losses aligned to sp['cands'] -> result dict. margin = 2nd-best minus best over ALL candidates; margin_x = best other canon_key minus chosen (old)."""
    cands = sp["cands"]
    order = sorted(range(len(cands)), key=lambda i: losses[i])
    b = order[0]
    second = order[1] if len(order) > 1 else None
    margin = (losses[second] - losses[b]) if second is not None else 99.0
    kb = canon_key(cands[b])
    other = [losses[i] for i in order if canon_key(cands[i]) != kb]
    margin_x = (min(other) - losses[b]) if other else 99.0
    return dict(ord=sp.get("ord"), current=cands[0], chosen=cands[b], chosen_idx=b, ncand=len(cands), margin=round(margin, 3), margin_x=round(margin_x, 3),
                tie=bool(second is not None and margin < TIE_EPS), same_key_alt=bool(second is not None and canon_key(cands[second]) == kb),
                cur_loss=round(losses[0], 3), best_loss=round(losses[b], 3), cur_rank=order.index(0),
                top2=[[cands[i], round(losses[i], 3)] for i in order[:2]], kind=sp.get("kind"), w=[sp.get("w_start"), sp.get("w_end")])


def score_row(row, loss_fn, free_hyp=None, align_fn=None, passes=2):
    """row: {id, label, spans:[{ord, char_start, char_end, cands (cands[0] = current reading), kind, w_start, w_end}], alt_labels:[str]?, coded_cell?:bool}
    loss_fn(list[str]) -> list[float] CTC losses of whole-row labels; free_hyp: greedy CTC transcript or None; align_fn(label) -> word times or None (raises on failure).
    Never raises for data problems: returns a record with status."""
    rec = dict(id=row["id"], label=row["label"], status="scored", spans=[], row_alts=[])
    raw_loss_fn, memo = loss_fn, {}

    def loss_fn(labs):
        """same losses, each distinct whole-row label scored ONCE per row: pass 2 re-asks most of pass 1's labels (same context) and duplicate candidate strings repeat inside a batch.
        A CTC loss is per sequence (independent of what else is in the batch), so the values are the ones the uncached calls return."""
        need = [l for l in dict.fromkeys(labs) if l not in memo]
        if need:
            memo.update(zip(need, raw_loss_fn(need)))
        return [memo[l] for l in labs]
    try:
        label = row["label"]; spans = row.get("spans") or []
        choice = [0] * len(spans); res = [None] * len(spans)
        multi = [k for k, sp in enumerate(spans) if len(sp["cands"]) > 1]
        for ps in range(passes if len(multi) > 1 else 1):
            for k in multi:
                sp = spans[k]
                labs = [build_label(label, spans, choice[:k] + [c] + choice[k + 1:]) for c in range(len(sp["cands"]))]
                ls = loss_fn(labs)
                r = span_result(sp, ls, choice[k]); choice[k] = r["chosen_idx"]; res[k] = r
        for k, sp in enumerate(spans):
            if res[k] is None:
                res[k] = dict(ord=sp.get("ord"), current=sp["cands"][0], chosen=sp["cands"][0], chosen_idx=0, ncand=len(sp["cands"]), margin=99.0, margin_x=99.0, tie=False,
                              same_key_alt=False, cur_loss=None, best_loss=None, cur_rank=0, top2=[[sp["cands"][0], None]], kind=sp.get("kind"), w=[sp.get("w_start"), sp.get("w_end")])
        chosen_label = build_label(label, spans, choice)
        rec["spans"] = res; rec["chosen_label"] = chosen_label; rec["changed"] = chosen_label != label
        alts = [a for a in (row.get("alt_labels") or []) if a and a != chosen_label]
        if alts:
            ls = loss_fn([chosen_label] + alts)
            rec["row_alts"] = [dict(label=a, loss=round(l, 3), delta_vs_chosen=round(l - ls[0], 3)) for a, l in zip(alts, ls[1:])]
            rec["chosen_loss"] = round(ls[0], 3)
        rec["min_margin"] = min([r["margin"] for r in res if r["ncand"] > 1], default=99.0)
        rec["any_tie"] = any(r["tie"] for r in res); rec["any_same_key_alt"] = any(r["same_key_alt"] for r in res if r["ncand"] > 1)
        rec["n_multi"] = len(multi)
        if free_hyp is not None:
            rec["hyp"] = free_hyp; rec["cer"] = round(cer_of(chosen_label, free_hyp), 3)
        coded = bool(row.get("coded_cell"))
        status = "scored"
        if rec["min_margin"] < MARGIN or rec["any_tie"]:
            status = "undecided"
        if rec.get("cer") is not None and rec["cer"] > (0.6 if coded else MAX_CER):
            status = "cer_high"
        if align_fn is not None:
            try:
                rec["words"] = align_fn(chosen_label)
            except Exception as e:
                rec["words"] = None; rec["align_error"] = repr(e)[:120]; status = "align_fail"
        rec["status"] = status
    except Exception as e:  # never reject: a data problem becomes a status
        rec["status"] = "error"; rec["error"] = repr(e)[:200]
    return rec


STATUSES = ("scored", "undecided", "cer_high", "align_fail", "error")
