"""Word repair for judged rows whose only fault is a NON-number word (2026-10-16).

Parliament captions are edited text: they drop false starts and small words ("das machen macht bedeutet" ->
"das bedeutet"), add words the speaker did not say, and normalise wording. When an external verbatim transcript of a
row's audio rejected the row for such a word while every number span agreed, the row is repaired only where two of
three sources agree:

  1. the caption label and the transcript are aligned word by word (letters only, accents and compound spacing
     ignored); each INTERIOR stretch where they differ (not touching the row's first or last word: clip edges are a
     matter for the partial-rows edge retry) is a candidate;
  2. a candidate is refused (the caption stays) when it touches a number reading, brings in a number / percent /
     hour word, is longer than MAX_WORDS words, changes a name-like caption word (capitalised mid-sentence and rare,
     or all capitals: the transcript misspells names), brings in a word the language's captions never write
     mid-sentence at least MIN_LEX times (unknown spelling or casing), or crosses sentence punctuation;
  3. the CTC reader scores the whole row with the caption wording and with each candidate in place: a candidate is
     applied only when the audio prefers it by >= MARGIN nats (hesitations "äh" / "euh" are never inserted).

The repaired row keeps its caption, spans and readings; its text (and num_chars, word_times) change, and it gets a new
id (wid) so the original row's reject stays in force. The new pass must then be judged like any other.

    modal run ml_wordfix.py::wordfix --langs de,fr --src rows_granvox2 --ext wordfix_granvox2.jsonl --name rows_granvox2w
    modal run ml_wordfix.py::wordfix ... --dry --calib 150      (no pass written; margin calibration on kept rows)
    python3 ml_wordfix.py plan <lang> <rows.jsonl> <ext.jsonl> <lexicon rows.jsonl ...>   (no CTC: the candidates)

ext lines: {"id", "words": [{"w", ...}]} (or "text"), one per candidate row, the transcript of the row's own audio.
"""

import collections
import difflib
import json
import os
import random
import re
import sys

import ml_decide as D

# nats the audio must prefer the transcript wording by. Calibrated 2026-10-16 (de fr es nl, granvox2 rejects): real
# repairs score 20-200 nats, the doubtful ones of the spot check sat at 3-9; on kept rows, wrong single-word edits
# reach >= 10 nats 0-2% of the time (vs 0-3% at ml_decide.MARGIN = 3)
MARGIN = 10.0
MAX_WORDS = 3  # words per stretch (either side)
MAX_STRETCHES = 2  # interior stretches per row
EDGE_WORDS = 2  # aligned words a stretch needs on each side: the CTC misreads clipped onsets / tails (calibration
# 2026-10-16: deleting the 2nd word of a row whose first word was clipped scored 160 nats "better")
MIN_LEX = 2  # mid-sentence caption occurrences an inserted word needs (its spelling and casing come from there)
NAME_RARE = 3  # a capitalised mid-sentence caption word seen fewer times than this is treated as a name
# hesitations are never inserted ("um" is a German / Portuguese word, so it is not one here)
HESITATIONS = {"äh", "ähm", "öh", "ehm", "eh", "euh", "heu", "hum", "hm", "hmm", "mm", "mmm", "uh", "uhm", "ahm", "eeh",
               "ehh", "uhh", "mhm", "ehmm"}
# articles that are also number words: they may be inserted / replaced (any other number word may not)
ARTICLES = {"de": {"ein", "eine", "einen", "einem", "einer", "eines"}, "fr": {"un", "une"},
            "es": {"un", "una", "uno"}, "it": {"un", "una", "uno"}, "pt": {"um", "uma"}, "nl": {"een"}}
QUALIFIERS = {"de": {"prozent", "komma", "uhr"}, "es": {"por", "ciento", "coma", "horas", "hora"},
              "fr": {"pour", "cent", "virgule", "heures", "heure"}, "it": {"per", "cento", "virgola", "ore", "ora"},
              "nl": {"procent", "komma", "uur"}, "pt": {"por", "cento", "virgula", "horas", "hora"}}
SCALES = {"de": {"hundert", "tausend", "million", "millionen", "milliarde", "milliarden"},
          "es": {"cien", "ciento", "cientos", "mil", "millon", "millones", "billon", "billones"},
          "fr": {"cent", "cents", "mille", "million", "millions", "milliard", "milliards"},
          "it": {"cento", "mille", "mila", "milione", "milioni", "miliardo", "miliardi"},
          "nl": {"honderd", "duizend", "miljoen", "miljard", "miljoenen", "miljarden"},
          "pt": {"cem", "cento", "centos", "mil", "milhao", "milhoes", "bilhao", "bilhoes"}}
STEMS = {"de": ("tausend", "hundert", "million", "milliard"), "nl": ("duizend", "honderd", "miljoen", "miljard"),
         "it": ("mila", "cento", "milion", "miliard")}
JOINERS = {"und", "y", "e", "et", "en", "con", "de", "di", "d", "of", "and"}  # inside number readings, not numbers
TOK = re.compile(r"\S+")
EDGE_PUNCT = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
SENT_END = re.compile(r"[.!?…][\"'»”)\]]*$")
APOS = str.maketrans({"’": "'", "‘": "'"})

_NUMWORDS: dict = {}


def numwords(lang: str) -> set:
    """Number words of `lang` (keys) from the grammar table's cardinal entries + scale words (articles excluded)."""
    if lang not in _NUMWORDS:
        here = os.path.dirname(os.path.abspath(__file__))
        p = next(x for x in (os.path.join(here, "ml_grammar_table.json"), "/root/ml_grammar_table.json")
                 if os.path.exists(x))
        ws = set()
        with open(p) as f:
            tab = json.load(f)
        for k, v in tab.items():
            kl, kind = k.split("|")[:2]
            if kl == lang and kind == "card":
                for c in v:
                    ws.update(D.key(w) for w in c.split())
        _NUMWORDS[lang] = (ws | SCALES.get(lang, set())) - JOINERS - {""}
    return _NUMWORDS[lang]


def is_numberish(k: str, lang: str) -> bool:
    if k in numwords(lang):
        return True
    return len(k) >= 6 and any(k.startswith(s) or k.endswith(s) for s in STEMS.get(lang, ()))


def split_tok(t: str):
    """(leading punctuation, core, trailing punctuation) of a whitespace token."""
    m = EDGE_PUNCT.match(t)
    return m.group(1), m.group(2), m.group(3)


def wkey(core: str, lang: str) -> str:
    if lang == "nl" and re.search(r"\d", core):
        return "#"
    return D.key(core)


# ------------------------------------------------------------------ lexicon (spelling + casing from the captions)
class Lexicon:
    """Mid-sentence surface forms of each word in the language's captions (lower-cased core -> Counter of forms)."""

    def __init__(self):
        self.forms = collections.defaultdict(collections.Counter)

    def add(self, text: str):
        prev = None
        for m in TOK.finditer(text or ""):
            lead, core, trail = split_tok(m.group().translate(APOS))
            if core and prev is not None and not SENT_END.search(prev) and not re.search(r"\d", core):
                self.forms[core.lower()][core] += 1
            prev = m.group()

    def count(self, w: str) -> int:
        return sum(self.forms.get(w.translate(APOS).lower(), {}).values())

    def form(self, w: str) -> str | None:
        c = self.forms.get(w.translate(APOS).lower())
        if not c or sum(c.values()) < MIN_LEX:
            return None
        return c.most_common(1)[0][0]

    def to_json(self) -> dict:
        return {k: dict(v) for k, v in self.forms.items()}

    @classmethod
    def from_json(cls, d: dict) -> "Lexicon":
        lx = cls()
        for k, v in d.items():
            lx.forms[k] = collections.Counter(v)
        return lx


# ------------------------------------------------------------------ alignment
def label_tokens(text: str, num_chars, lang: str) -> list:
    out = []
    for m in TOK.finditer(text):
        lead, core, trail = split_tok(m.group())
        num = next((k for k, (a, b) in enumerate(num_chars or []) if a < m.end() and m.start() < b), None)
        out.append({"s": m.group(), "lead": lead, "core": core, "trail": trail, "a": m.start(), "b": m.end(),
                    "num": num, "key": wkey(core, lang) if num is None or lang != "nl" else "#"})
    return out


def ext_words(ext: dict) -> list:
    ws = ext.get("words")
    return [w["w"] for w in ws] if ws else (ext.get("text") or "").split()


def judge_seq(words: list, lang: str) -> list:
    """[(key, surface)] of the transcript; hesitations dropped. Dutch: a digit run and the number / percent / hour
    words glued to it are one "#" item (the transcript writes numbers as digits)."""
    seq = []
    words = [w.translate(APOS).strip("—–-…") for w in words]  # the transcript marks a cut-off word ("machen—")
    for n, w in enumerate(words):
        k = wkey(w, lang)
        nxt = words[n + 1].lower() if n + 1 < len(words) else ""
        # "eh bien" (fr) is a phrase, not a hesitation + "bien" (spot check 2026-10-16)
        if k and (w.lower().strip(".,;:!?") not in HESITATIONS or (k == "eh" and nxt == "bien")):
            seq.append([k, w.lower()])
    if lang == "nl":
        arts = ARTICLES[lang]
        for x in seq:  # a number the transcript wrote as words ("twaalf uur") is a number item too
            if x[0] != "#" and is_numberish(x[0], lang) and x[0] not in arts:
                x[0] = "#"
        for _ in range(2):
            for i, (k, _w) in enumerate(seq):
                near = (i > 0 and seq[i - 1][0] == "#") or (i + 1 < len(seq) and seq[i + 1][0] == "#")
                if k != "#" and near and (is_numberish(k, lang) or k in QUALIFIERS[lang]):
                    seq[i][0] = "#"
        out = []
        for k, w in seq:
            if k == "#" and out and out[-1][0] == "#":
                continue
            out.append([k, w])
        seq = out
    return [tuple(x) for x in seq]


def label_seq(toks: list, lang: str) -> list:
    """[(key, [token indices])]: punctuation-only tokens skipped; Dutch number readings collapse to one "#" item."""
    seq = []
    for i, t in enumerate(toks):
        if not t["key"]:
            continue
        if lang == "nl" and t["key"] == "#" and seq and seq[-1][0] == "#":
            seq[-1][1].append(i)
            continue
        seq.append((t["key"], [i]))
    if lang == "nl":
        # as on the transcript side: number / percent / hour words next to a reading go with it ("vijftien uur")
        for _ in range(2):
            for n, (k, ix) in enumerate(seq):
                near = (n > 0 and seq[n - 1][0] == "#") or (n + 1 < len(seq) and seq[n + 1][0] == "#")
                if k != "#" and near and (is_numberish(k, lang) or k in QUALIFIERS[lang]):
                    seq[n] = ("#", ix)
        out = []
        for k, ix in seq:
            if k == "#" and out and out[-1][0] == "#":
                out[-1] = ("#", out[-1][1] + ix)
                continue
            out.append((k, list(ix)))
        seq = out
    return seq


def stretches(toks: list, jseq: list, lang: str) -> list:
    """Differing stretches between label and transcript: [{op, i1, i2, j1, j2, toks, words, edge}]."""
    A = [k for k, _ in label_seq(toks, lang)]
    lseq = label_seq(toks, lang)
    B = [k for k, _ in jseq]
    out = []
    for op, i1, i2, j1, j2 in _peeled(A, B):
        if op == "equal" or "".join(A[i1:i2]) == "".join(B[j1:j2]):
            continue  # compound spacing ("nochmal" / "noch mal") is not a word difference
        ti = [x for _, ix in lseq[i1:i2] for x in ix]
        if ti:
            ti = list(range(ti[0], ti[-1] + 1))  # punctuation-only tokens inside the stretch go with it
        out.append({"op": op, "i1": i1, "i2": i2, "j1": j1, "j2": j2, "toks": ti,
                    "after": lseq[i1 - 1][1][-1] if i1 > 0 else -1, "words": [w for _, w in jseq[j1:j2]],
                    "keys": B[j1:j2], "edge": (i1 == 0 and j1 == 0) or (i2 == len(A) and j2 == len(B)),
                    "near_edge": i1 < EDGE_WORDS or i2 > len(A) - EDGE_WORDS})
    return out


def _peeled(A: list, B: list) -> list:
    """difflib opcodes, with the same-letter words at either end of a replace peeled off as equal ("dix neuf" /
    "dix-neuf mais" -> the insertion of "mais" alone)."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op != "replace":
            out.append((op, i1, i2, j1, j2))
            continue
        changed = True
        while changed and i1 < i2 and j1 < j2:
            changed = False
            for m in range(1, i2 - i1 + 1):
                k = next((k for k in range(1, j2 - j1 + 1) if "".join(A[i1:i1 + m]) == "".join(B[j1:j1 + k])), None)
                if k and (m, k) != (i2 - i1, j2 - j1):
                    i1, j1, changed = i1 + m, j1 + k, True
                    break
            for m in range(1, i2 - i1 + 1):
                k = next((k for k in range(1, j2 - j1 + 1) if "".join(A[i2 - m:i2]) == "".join(B[j2 - k:j2])), None)
                if k and (m, k) != (i2 - i1, j2 - j1):
                    i2, j2, changed = i2 - m, j2 - k, True
                    break
        if i1 == i2 and j1 == j2:
            continue
        out.append(("replace" if i1 < i2 and j1 < j2 else "delete" if i1 < i2 else "insert", i1, i2, j1, j2))
    return out


def abbreviation(t: dict) -> bool:
    """A written abbreviation in the caption ("M.", "Dr.", "Mme"): the transcript's spoken word may replace it."""
    return bool(re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ]{0,2}", t["core"]) and t["trail"].startswith(".")) or t["core"] in (
        "Mme", "Mmes", "Mlle", "MM", "Sr", "Sra", "Dra")


def sentence_start(toks_s: list, k: int) -> bool:
    if k <= 0 or re.fullmatch(r"[\"'«„(]*[A-ZÀ-Ý][a-zà-ÿ]{0,2}\.", toks_s[k - 1]):
        return False  # "M. Barnier": an abbreviation, not a sentence end
    return bool(SENT_END.search(toks_s[k - 1]))


def name_like(toks: list, i: int, lex: Lexicon, lang: str) -> bool:
    core = toks[i]["core"]
    if abbreviation(toks[i]):
        return False
    if not core or not core[:1].isupper():
        return False
    if len(core) > 1 and core.isupper():
        return True  # an acronym
    if i == 0 or sentence_start([t["s"] for t in toks], i):
        return False  # capitalised as a sentence start
    if lang != "de":
        return True  # Romance / Dutch: a capitalised mid-sentence word is a name
    return lex.count(core) < NAME_RARE  # German nouns are capitalised: only a rare one counts as a name


def guard(st: dict, toks: list, lex: Lexicon, lang: str) -> str | None:
    """Why this interior stretch may not be repaired (None = it may, if the audio agrees)."""
    if st.get("near_edge"):
        return "near_edge"
    if max(len([i for i in st["toks"] if toks[i]["key"]]), len(st["keys"])) > MAX_WORDS:
        return "long"
    arts = ARTICLES.get(lang, set())
    for i in st["toks"]:
        t = toks[i]
        if t["num"] is not None or t["key"] == "#":
            return "number"
        if t["key"] and ((is_numberish(t["key"], lang) and t["key"] not in arts) or t["key"] in QUALIFIERS[lang]):
            return "number"
        if name_like(toks, i, lex, lang):
            return "name"
    for k in st["keys"]:
        if k == "#" or (is_numberish(k, lang) and k not in arts) or k in QUALIFIERS[lang]:
            return "number"
    # a joiner put in or taken out between two number readings changes how the number is said ("novecentosessantadue
    # novantasei" -> "... e novantasei"): never (spot check 2026-10-17)
    ts_ = st["toks"]
    before = ts_[0] - 1 if ts_ else st["after"]
    after = ts_[-1] + 1 if ts_ else st["after"] + 1
    isnum = lambda i: 0 <= i < len(toks) and (toks[i]["num"] is not None or toks[i]["key"] == "#")  # noqa: E731
    if (set(st["keys"]) | {toks[i]["key"] for i in ts_}) & JOINERS and isnum(before) and isnum(after):
        return "number"
    for w in st["words"]:
        if re.search(r"[^\w'\-]", w) or lex.form(w) is None:
            return "unknown"
    ts = st["toks"]
    for n, i in enumerate(ts):
        t = toks[i]
        if (n > 0 and t["lead"]) or (n < len(ts) - 1 and re.search(r"[^,\s]", t["trail"])):
            return "punct"
    if st["op"] == "delete" and ts and toks[ts[0]]["lead"]:
        return "punct"
    return None


# ------------------------------------------------------------------ applying repairs
def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def apply(text: str, num_chars, toks: list, sts: list, lex: Lexicon, lang: str):
    """(new text, new num_chars) with the stretches `sts` replaced by the transcript wording."""
    out = []  # [surface, original token index or None]
    by_first = {st["toks"][0]: st for st in sts if st["toks"]}
    ins_after = {st["after"]: st for st in sts if not st["toks"]}
    skip = set()
    for i, t in enumerate(toks):
        if i in skip:
            continue
        st = by_first.get(i)
        if st is not None:
            first, last = toks[st["toks"][0]], toks[st["toks"][-1]]
            new = [lex.form(w) for w in st["words"]]
            if new:
                new[0] = first["lead"] + new[0]
                new[-1] = new[-1] + (last["trail"][1:] if abbreviation(last) else last["trail"])
                out += [[w, None] for w in new]
            elif last["trail"] and out and not split_tok(out[-1][0])[2]:
                nxt = toks[st["toks"][-1] + 1]["core"] if st["toks"][-1] + 1 < len(toks) else ""
                if not (SENT_END.search(last["trail"]) and nxt[:1].islower()):
                    # a deleted word's trailing punctuation stays on the word before, unless it ended a sentence
                    # that the next word continues ("milliards d'euros. d'euros par an": a doubled caption word)
                    out[-1][0] += last["trail"]
            skip.update(st["toks"])
        else:
            out.append([t["s"], i])
        last_i = st["toks"][-1] if st is not None else i
        if last_i in ins_after:
            out += [[lex.form(w), None] for w in ins_after[last_i]["words"]]
    # casing: inserted words at a sentence start are capitalised; an original word that stops / starts being a
    # sentence start takes its mid-sentence casing / a capital
    surf = [s for s, _ in out]
    orig_s = [t["s"] for t in toks]
    for k, (s, oi) in enumerate(out):
        now = k == 0 or sentence_start(surf, k)
        if oi is None:
            if now and k > 0:
                out[k][0] = surf[k] = _lead_cap(s)
            continue
        was = oi == 0 or sentence_start(orig_s, oi)
        if now and not was and k > 0:
            out[k][0] = surf[k] = _lead_cap(s)
        elif was and not now:
            lead, core, trail = split_tok(s)
            f = lex.forms.get(core.lower())
            low = f.most_common(1)[0][0][:1].islower() if f else lang != "de"  # unknown: German may be a noun
            if toks[oi]["num"] is None and low and not name_like(toks, oi, lex, lang):
                out[k][0] = surf[k] = lead + core[:1].lower() + core[1:] + trail
            elif toks[oi]["num"] is not None and lang != "de":
                out[k][0] = surf[k] = lead + core[:1].lower() + core[1:] + trail
    new_text = " ".join(surf)
    starts, pos = {}, 0
    for s, oi in out:
        if oi is not None:
            starts[oi] = pos
        pos += len(s) + 1
    nc = []
    for a, b in num_chars or []:
        ta = next(i for i, t in enumerate(toks) if t["a"] <= a < t["b"])
        tb = next(i for i, t in enumerate(toks) if t["a"] < b <= t["b"])
        if ta not in starts or tb not in starts:
            raise ValueError("a number token was dropped")
        na, nb = starts[ta] + a - toks[ta]["a"], starts[tb] + b - toks[tb]["a"]
        if new_text[na:nb].lower() != text[a:b].lower():
            raise ValueError("number offsets moved")
        nc.append([na, nb])
    return new_text, nc


def _lead_cap(s: str) -> str:
    lead, core, trail = split_tok(s)
    return lead + _cap(core) + trail


def apply_row(row: dict, toks: list, sts: list, lex: Lexicon, lang: str):
    return apply(row["text"], row.get("num_chars"), toks, sts, lex, lang)


def _segments(text: str, ranges) -> list:
    out, pos = [], 0
    for a, b in ranges:
        out.append(text[pos:a])
        pos = b
    return out + [text[pos:]]


def repair_caption(row: dict, text: str, nc: list, lang: str) -> str | None:
    """The caption with the same word repairs: the stretches between number spans are the same in caption and label,
    so each edited stretch is copied over (None when a stretch differs or the grammar would find other spans; the Dutch
    judge aligns the CAPTION with the transcript, and render_cap(caption) should give the label)."""
    import ml_cands as M

    cap = row.get("caption") or ""
    sp = M.spans(cap, lang)
    if [s.text for s in sp] != [x["text"] for x in row.get("spans") or []]:
        return None
    old_l, new_l = _segments(row["text"], row.get("num_chars") or []), _segments(text, nc)
    old_c = _segments(cap, [(s.start, s.end) for s in sp])
    if not (len(old_l) == len(new_l) == len(old_c)):
        return None
    segs = []
    for ol, nl_, oc in zip(old_l, new_l, old_c):
        if ol == nl_:
            segs.append(oc)
        elif ol == oc:
            segs.append(nl_)
        else:
            return None
    out = segs[0]
    for s, seg in zip(sp, segs[1:]):
        out += s.text + seg
    if [s.text for s in M.spans(out, lang)] != [s.text for s in sp]:
        return None
    return out


def readings_findable(row: dict, text: str, nc: list) -> bool:
    """The judge locates each stored reading in a post-processed label by searching left to right: every reading must
    be found exactly at its own place (an inserted article "eine" before a reading "eine" would mislead it)."""
    pos = 0
    for sp, (a, b) in zip(row.get("spans") or [], nc):
        r = sp["reading"].strip().lower()
        f = text.lower().find(r, pos)
        if f != a:
            return False
        pos = f + len(r)
    return True


# ------------------------------------------------------------------ one row: candidates and decision
def plan_row(row: dict, ext: dict, lex: Lexicon, lang: str) -> dict:
    """{"status", "toks", "stretches" (interior, with "guard")} without the audio."""
    D.set_script(lang)
    toks = label_tokens(row["text"], row.get("num_chars"), lang)
    sts = stretches(toks, judge_seq(ext_words(ext), lang), lang)
    inner = [s for s in sts if not s["edge"]]
    for s in inner:
        s["guard"] = guard(s, toks, lex, lang)
    # a word the caption has in one stretch and the transcript in another moved, it was not dropped or invented
    # ("Uiteraard is de termijn" / "uiterste termijn is": deleting "de termijn" alone scored 19 nats)
    for s in sts:
        lk = {toks[i]["key"] for i in s["toks"] if toks[i]["key"]}
        for o in sts:
            if o is not s and (lk & set(o["keys"]) or set(s["keys"]) & {toks[i]["key"] for i in o["toks"]}):
                if not s["edge"] and not s["guard"]:
                    s["guard"] = "moved"
    res = {"toks": toks, "stretches": inner, "n_edge": len(sts) - len(inner)}
    if not inner:
        res["status"] = "edge_only" if sts else "no_difference"
    elif len(inner) > MAX_STRETCHES:
        res["status"] = "many"
    elif all(s["guard"] for s in inner):
        res["status"] = "guarded"
    else:
        res["status"] = "candidate"
    return res


def surface(toks: list, st: dict) -> str:
    return " ".join(toks[i]["s"] for i in st["toks"])


def decide_row(row: dict, plan: dict, lex: Lexicon, lang: str, nll, margin: float = MARGIN) -> dict:
    """nll(text) -> CTC negative log-likelihood of the row audio. -> {"status", "row"?, "fixed", "tried"}."""
    toks = plan["toks"]
    base = nll(row["text"])
    tried, ok = [], []
    for st in plan["stretches"]:
        info = {"label": surface(toks, st), "ext": " ".join(st["words"]), "op": st["op"], "guard": st["guard"]}
        if not st["guard"]:
            try:
                t1, _ = apply_row(row, toks, [st], lex, lang)
                info["gain"] = round(base - nll(t1), 2)
                if info["gain"] >= margin:
                    ok.append(st)
            except (ValueError, StopIteration) as e:
                info["guard"] = f"apply: {e}"
        tried.append(info)
    res = {"tried": tried, "base_nll": round(base, 2)}
    if not ok:
        res["status"] = "audio_disagrees"
        return res
    try:
        text, nc = apply_row(row, toks, ok, lex, lang)
    except (ValueError, StopIteration) as e:
        res["status"] = f"apply: {e}"
        return res
    if re.search(r"\d", text) or not readings_findable(row, text, nc):
        res["status"] = "readings_not_findable"
        return res
    cap = repair_caption(row, text, nc, lang)
    if cap is None:
        res["status"] = "caption_not_repairable"
        return res
    res["status"] = "repaired"
    res["fixed"] = [t for t in tried if t.get("gain", -1e9) >= margin and not t["guard"]]
    res["row"] = dict(row, text=text, num_chars=nc, id=wid(row["id"]), caption=cap, caption_orig=row.get("caption"),
                      wordfix={"of": row["id"], "margin": margin,
                               "fixed": [{"label": t["label"], "ext": t["ext"], "gain": t["gain"]} for t in res["fixed"]]})
    res["row"].pop("word_times", None)
    return res


def wid(rid: str) -> str:
    """The repaired row's id: "_w" before a VoxPopuli time suffix (the held-out rules key on it), else appended."""
    m = re.match(r"^(.*)_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$", str(rid))
    return f"{m.group(1)}_w_{m.group(2)}" if m else f"{rid}_w"


# ------------------------------------------------------------------ margin calibration (kept rows, wrong edits)
def perturbations(row: dict, lex: Lexicon, lang: str, rng) -> list:
    """Wrong edits of a correct label: [(kind, text)] for one deleted, one inserted and one substituted interior
    non-number word. The audio must NOT prefer them (false repairs); the reverse direction is the recall check."""
    toks = label_tokens(row["text"], row.get("num_chars"), lang)
    ok = [i for i, t in enumerate(toks) if EDGE_WORDS <= i < len(toks) - EDGE_WORDS and t["num"] is None and len(t["key"]) >= 3
          and not is_numberish(t["key"], lang) and not t["lead"] and not t["trail"]]
    if not ok:
        return []
    out = []
    common = getattr(lex, "_common", None)
    if common is None:
        common = lex._common = [w for w, c in sorted(lex.forms.items(), key=lambda x: -sum(x[1].values()))[:3000]
                                if len(w) >= 4 and w.isalpha()]
    i = rng.choice(ok)
    out.append(("del", " ".join(t["s"] for k, t in enumerate(toks) if k != i)))
    j = rng.choice(ok)
    ins = rng.choice(common)
    out.append(("ins", " ".join(sum(([t["s"], ins] if k == j else [t["s"]] for k, t in enumerate(toks)), []))))
    w = toks[rng.choice(ok)]
    near = [x for x in difflib.get_close_matches(w["core"].lower(), common, n=5, cutoff=0.75) if D.key(x) != w["key"]]
    if near:
        out.append(("sub", " ".join(near[0] if t is w else t["s"] for t in toks)))
    return out


# ------------------------------------------------------------------ Modal step
try:
    import modal
except ImportError:  # the pure functions above run (and are tested) without Modal
    modal = None

if modal is not None:
    VOL = "/vol"
    OUT2 = f"{VOL}/ml/p3v2"
    vol = modal.Volume.from_name("parakeet-spoken-form")
    gpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .apt_install("ffmpeg", "libsndfile1")
                 .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.44.2", "soundfile", "librosa",
                              "numpy<2")
                 .add_local_python_source("ml_cands", "ml_decide", "ml_slav", "ml_wordfix")
                 .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
    app = modal.App("parakeet-ml-wordfix")

    def _jl(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def build_lexicon(lang: str) -> Lexicon:
        import glob

        lx = Lexicon()
        for p in sorted(glob.glob(f"{OUT2}/{lang}/rows*.jsonl")):
            for r in _jl(p):
                lx.add(r.get("caption") or "")
        for p in sorted(glob.glob(f"{OUT2}/{lang}/cands_*.jsonl")):
            for r in _jl(p):
                lx.add(r.get("text") or "")
        return lx

    @app.function(image=gpu_image, volumes={VOL: vol}, gpu="L4", memory=16384, timeout=2 * 3600, max_containers=4)
    def wordfix_lang(lang: str, src: str, ext: str, name: str, margin: float = MARGIN, dry: bool = False,
                     ctc_model: str = "", calib: int = 0) -> dict:
        import librosa
        import soundfile as sf

        ctc = D.LangCTC(lang, model=(D.STRONG.get(lang) if ctc_model == "strong" else ctc_model) or None)
        lx = build_lexicon(lang)
        rows = {r["id"]: r for r in _jl(f"{OUT2}/{lang}/{src}.jsonl")}
        exts = {e["id"]: e for e in _jl(f"{OUT2}/{lang}/{ext}")}
        out_dir = f"{OUT2}/{lang}" + ("/dry" if dry else "")
        os.makedirs(out_dir, exist_ok=True)

        def audio(r):
            # later labelling passes may re-cut a piece under the same id (same wav path): the transcript was made of
            # the audio as it was, so a wav whose length no longer matches the row is not scored
            if abs(sf.info(r["audio_filepath"]).duration - r["duration"]) > 0.05:
                raise RuntimeError("audio_changed")
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x

        st = collections.Counter()
        kept, diag = [], []
        for rid, e in exts.items():
            row = rows.get(rid)
            if row is None:
                st["not_in_src"] += 1
                continue
            plan = plan_row(row, e, lx, lang)
            d = {"id": rid, "status": plan["status"], "n_edge": plan["n_edge"],
                 "stretches": [{"label": surface(plan["toks"], s), "ext": " ".join(s["words"]), "guard": s["guard"]}
                               for s in plan["stretches"]]}
            if plan["status"] == "candidate":
                try:
                    lp = ctc.lp(audio(row))
                    res = decide_row(row, plan, lx, lang, lambda t: ctc.nll(lp, t), margin)
                except Exception as ex:  # noqa: BLE001  one bad file must not lose the language
                    res = {"status": str(ex) if str(ex) == "audio_changed" else f"error {type(ex).__name__}",
                           "tried": []}
                d.update(status=res["status"], tried=res["tried"], base_nll=res.get("base_nll"))
                if res.get("tried"):
                    d["greedy"] = ctc.greedy(lp)
                if res.get("row"):
                    nr = res["row"]
                    times = ctc.word_times(lp, nr["text"])
                    nr["word_times"] = [None if t is None else [round(t[0], 3), round(t[1], 3)] for t in times]
                    d["text"] = nr["text"]
                    kept.append(nr)
            st[d["status"]] += 1
            diag.append(d)
        cal = []
        if calib:
            rej = set()
            rp = f"{OUT2}/{lang}/reject_ids.txt"
            if os.path.exists(rp):
                rej = {l.strip() for l in open(rp) if l.strip()}
            pool = sorted(i for i in rows if i not in rej and i not in exts)
            rng = random.Random(f"wordfix-{lang}")
            rng.shuffle(pool)
            for rid in pool[:calib]:
                row = rows[rid]
                try:
                    lp = ctc.lp(audio(row))
                    b = ctc.nll(lp, row["text"])
                    cal.append({"id": rid, "kind": "base", "nll": round(b, 2), "greedy": ctc.greedy(lp)})
                    for kind, t in perturbations(row, lx, lang, rng):
                        cal.append({"id": rid, "kind": kind, "gain_wrong": round(b - ctc.nll(lp, t), 2), "text": t})
                except Exception:  # noqa: BLE001
                    continue
        with open(f"{out_dir}/wordfix_{name}.jsonl", "w") as f:
            for d in diag:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if cal:
            with open(f"{out_dir}/wordfix_calib_{name}.jsonl", "w") as f:
                for c in cal:
                    f.write(json.dumps(c) + "\n")
        with open(f"{out_dir}/{name}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = {"lang": lang, "src": src, "name": name, "margin": margin, "dry": dry, "status": dict(st),
                 "repaired": len(kept), "hours": round(sum(r["duration"] for r in kept) / 3600, 3), "calib": len(cal)}
        json.dump(stats, open(f"{out_dir}/stats_{name}.json", "w"), indent=1)
        vol.commit()
        return stats

    @app.local_entrypoint()
    def wordfix(langs: str, src: str, ext: str, name: str, margin: float = MARGIN, dry: bool = False,
                ctc_model: str = "", calib: int = 0):
        """ctc_model "strong" uses ml_decide.STRONG for languages that have one (nl)."""
        ls = langs.split(",")
        args = [(lg, src, ext, name, margin, dry, ctc_model if ctc_model != "strong" or lg in D.STRONG else "", calib)
                for lg in ls]
        for s in wordfix_lang.starmap(args):
            print(json.dumps(s, ensure_ascii=False))


def _main(argv):
    """plan <lang> <rows.jsonl> <ext.jsonl> <lexicon files...>: the candidates and their guards, no audio."""
    if argv[:1] != ["plan"]:
        print(__doc__)
        return
    lang, rows_p, ext_p, lex_ps = argv[1], argv[2], argv[3], argv[4:]
    lx = Lexicon()
    for p in lex_ps:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                lx.add(r.get("caption") or r.get("text") or "")
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(rows_p, encoding="utf-8") if l.strip()}
    st = collections.Counter()
    for l in open(ext_p, encoding="utf-8"):
        e = json.loads(l)
        r = rows.get(e["id"])
        if r is None:
            continue
        p = plan_row(r, e, lx, lang)
        st[p["status"]] += 1
        print(json.dumps({"id": e["id"], "status": p["status"], "stretches": [
            {"label": surface(p["toks"], s), "ext": " ".join(s["words"]), "guard": s["guard"]} for s in p["stretches"]]},
            ensure_ascii=False))
    print(json.dumps(dict(st)), file=sys.stderr)


if __name__ == "__main__":
    _main(sys.argv[1:])
