"""Data-balance pieces for the spoken-form mix (pure Python: no Modal, no torch). Three parts:

1. Cut-number crops at every window offset (a GENERATOR). A streaming decoder reads 14.96 s windows every 9.44 s; a number that a
   window edge cuts must come out as exactly the words audible on each side. From labelled rows with word times
   (stitched into streams like ml_p3_core.build_windows), each number reading gets cut positions before it, in every gap
   between its words, inside each of its words (at 1/4 and 3/4 of the word's sound) and after it. Each position gives a
   MIRRORED pair: the window ending at the cut (left) and the window starting there (right), both WIN long where the
   stream allows. Labels use the build_windows edge rule unchanged (crop_label): a word with >= KEEP_FRAC (2/3) of its
   sound span [t0 - LEAD, t1 + TAIL] inside is labelled whole, <= DROP_FRAC (1/3) is left out, anything in between makes
   the position unusable. The two fractions of one word add up to 1, so the left and right labels of a pair partition
   the reading: together they are the full reading, each word exactly once (seam_pair checks this).

2. Number-free share + look-alike controls. number_free(text, lang): no digit / symbol and no number word of the
   language (ml_cands grammar table), except the look-alike tokens that are articles / pronouns / homophones in
   number-free speech. lookalikes(text, lang) tags the controls (articles ein/un/una/um/een, pronoun "uno de", homophones
   sei / sem / sans / sang / vin / Acht / ...). pick_plain fills a language's number-free hours up to a target share,
   homophone controls first, then pronoun, then article rows, then the rest, per source at most SRC_CAP of the hours.

3. Cross-source / cross-pass dedupe. One row per utterance: audio identity (VoxPopuli utterance time suffix, YODAS video +
   overlapping time range, Granary-VoxPopuli segment vs VoxPopuli-labelled utterance of the same session with
   overlapping text), normalised caption text (>= MIN_KEY_CHARS), near-duplicate captions (word 3-shingle Jaccard >=
   NEAR_JACCARD: ad jingles, boilerplate). A duplicate of a protected item (held-out, validation, decision or gate set)
   is removed from TRAINING; protected sets are never touched.
"""

import re
import unicodedata
from collections import Counter, defaultdict

EXT_BY = ("ext", "ext_pick")  # spans a second recognizer decided where the CTC did not agree

import ml_p3_core as C

# ------------------------------------------------------------------------------------------------ 1. seam crops
IN_WORD = (0.25, 0.75)  # cut points inside a word's sound span (1/4 -> right side keeps it, 3/4 -> left side keeps it)
MIN_HALF = 1.0  # s: each half of a pair must be at least this long and hold >= 1 word


def crop_label(words: list, a: float, b: float, num_keep: float | None = None, total: float | None = None) -> dict | None:
    """The build_windows label of [a, b] over stream words [(tok, t0, t1, is_num, row_i, k)]: None when a word near the
    window has no time or sits between DROP_FRAC and KEEP_FRAC inside (no label is what the audio holds). Otherwise
    {"inside": labelled words in time order, "kept_cut", "cut", "fracs"}. Same tests, same order as
    ml_p3_core.build_windows (test_ml_balance checks the two agree); num_keep = the compound number-word rule
    (ml_p3_core.num_keep_frac)."""
    near = [w for w in words if w[1] is None or (w[1] < b and w[2] > a) or (w[2] is not None and a <= w[2] <= b)]
    if any(w[1] is None for w in near):
        return None
    t1 = float("inf") if total is None else total  # sound spans are clipped to the stream [0, total]
    whole = [w for w in near if max(0.0, w[1] - C.LEAD) >= a and min(t1, w[2] + C.TAIL) <= b]
    part = [w for w in near if w not in whole]
    fr = {id(w): C.inside_frac(w, a, b, 0.0, total) for w in part}
    if any(C.DROP_FRAC < fr[id(w)] < C.KEEP_FRAC or C.num_edge_ambiguous(w, fr[id(w)], num_keep) for w in part):
        return None
    kept = [w for w in part if fr[id(w)] >= C.KEEP_FRAC]
    cut = [w for w in part if fr[id(w)] <= C.DROP_FRAC]
    return {"inside": sorted(whole + kept, key=lambda w: w[1]), "kept_cut": kept, "cut": cut,
            "fracs": [[w[0], round(fr[id(w)], 2), "kept" if w in kept else "dropped"] for w in part]}


def readings(words: list, rows: list[dict]) -> list[dict]:
    """Number readings of a stream: [{"idx": stream word indices, "kind", "row"}] from each row's num_chars / spans."""
    out = []
    by_row = defaultdict(list)
    for j, w in enumerate(words):
        by_row[w[4]].append(j)
    for i, r in enumerate(rows):
        starts, pos = [], 0
        for tok in r["text"].split():
            p = r["text"].index(tok, pos)
            starts.append((p, p + len(tok)))
            pos = p + len(tok)
        idx = by_row[i]
        for n, (x, y) in enumerate(r.get("num_chars") or []):
            ks = [k for k, (p, q) in enumerate(starts) if p < y and x < q]
            if not ks:
                continue
            sp = (r.get("spans") or [{}] * (n + 1))[n] if n < len(r.get("spans") or []) else {}
            out.append({"idx": [idx[k] for k in ks], "kind": sp.get("kind", "?"), "row": i})
    return out


def cut_positions(words: list, rd: dict) -> list[tuple[str, float]]:
    """[(position name, cut second)] for one reading: before, gap<j> between its words, in<j>_lo / in<j>_hi inside each
    word (IN_WORD), after. Positions whose word has no time are skipped."""
    ws = [words[j] for j in rd["idx"]]
    if any(w[1] is None for w in ws):
        return []
    lo = lambda w: w[1] - C.LEAD  # noqa: E731
    hi = lambda w: w[2] + C.TAIL  # noqa: E731
    first, last = rd["idx"][0], rd["idx"][-1]
    out = []
    prev = words[first - 1] if first > 0 and words[first - 1][2] is not None else None
    out.append(("before", max(lo(ws[0]) - 0.01, prev[2] if prev else 0.0)))
    for j, w in enumerate(ws):
        span = hi(w) - lo(w)
        out.append((f"in{j}_lo", lo(w) + IN_WORD[0] * span))
        out.append((f"in{j}_hi", lo(w) + IN_WORD[1] * span))
        if j + 1 < len(ws):
            out.append((f"gap{j}", (w[2] + ws[j + 1][1]) / 2))
    nxt = words[last + 1] if last + 1 < len(words) and words[last + 1][1] is not None else None
    out.append(("after", min(hi(ws[-1]) + 0.01, nxt[1] if nxt else hi(ws[-1]) + 0.01)))
    # no cut next to a word of <= SHORT_LETTERS letters ("e", "y", "et"): CTC times of such words are the least
    # reliable (seam check 2026-10-16: the external transcript put "e" of "cinquenta e quatro" 0.4 s inside the
    # left half the times gave it to the right half)
    return [(p, c) for p, c in out if not any(len(C.letters(w[0])) <= SHORT_LETTERS and w[1] is not None
                                              and w[1] - C.LEAD - SHORT_PAD < c < w[2] + C.TAIL + SHORT_PAD for w in words)]


SHORT_LETTERS, SHORT_PAD = 2, 0.15


def _far_edge(words: list, a: float, b: float, side: str, tries: int = 4, num_keep: float | None = None,
              total: float | None = None):
    """Move the FAR edge of a half (a for a left half, b for a right half) inward to a word's sound boundary until the
    label is defined (the cut edge never moves). Returns (a, b, label) or None."""
    for _ in range(tries):
        lab = crop_label(words, a, b, num_keep, total)
        if lab is not None:
            return a, b, lab
        far = a if side == "left" else b
        amb = [w for w in words if w[1] is not None and w[1] - C.LEAD < far < w[2] + C.TAIL]
        if not amb:
            return None
        w = amb[0] if side == "left" else amb[-1]
        if side == "left":
            a = w[2] + C.TAIL  # the word falls wholly outside
        else:
            b = w[1] - C.LEAD
        if b - a < MIN_HALF:
            return None
    lab = crop_label(words, a, b, num_keep, total)
    return (a, b, lab) if lab is not None else None


def seam_pair(words: list, total: float, c: float, win: float = C.WIN, num_keep: float | None = None) -> dict | None:
    """The mirrored pair at cut c: left [c - win, c], right [c, c + win] (clamped to the stream; the far edges move inward
    to a word boundary when needed). None when the cut itself is ambiguous or a half is too short / empty."""
    L = _far_edge(words, max(0.0, c - win), c, "left", num_keep=num_keep, total=total)
    R = _far_edge(words, c, min(total, c + win), "right", num_keep=num_keep, total=total)
    if L is None or R is None:
        return None
    (la, lb, ll), (ra, rb, rl) = L, R
    if lb - la < MIN_HALF or rb - ra < MIN_HALF or not ll["inside"] or not rl["inside"]:
        return None
    return {"c": round(c, 3), "left": {"a": round(la, 3), "b": round(lb, 3), "lab": ll},
            "right": {"a": round(ra, 3), "b": round(rb, 3), "lab": rl}}


def pair_partitions(words: list, pair: dict, rd: dict) -> bool:
    """Invariant: the reading's words in the left label + those in the right label = the reading, in order, each once."""
    want = [words[j] for j in rd["idx"]]
    got = [w for w in pair["left"]["lab"]["inside"] if w in want] + [w for w in pair["right"]["lab"]["inside"] if w in want]
    return [id(w) for w in got] == [id(w) for w in want]


def _half_row(words: list, h: dict, side: str, rd: dict, pos: str) -> dict:
    ins = h["lab"]["inside"]
    num = {id(words[j]) for j in rd["idx"]}
    cut = h["lab"]["cut"]
    cl = [w[0] for w in cut if w[1] < h["a"] + 1.0]
    cr = [w[0] for w in cut if w[2] > h["b"] - 1.0]
    # the cut edge is the right edge of a left half and the left edge of a right half
    at_cut = [w for w in cut if (w[2] + C.TAIL > h["b"] - 0.01 if side == "left" else w[1] - C.LEAD < h["a"] + 0.01)]
    edge_w = (ins[-1] if side == "left" else ins[0]) if ins else None
    return {"a": h["a"], "b": h["b"], "text": " ".join(w[0] for w in ins), "num_tokens": [w[3] for w in ins],
            "cut_edge_dropped": at_cut[0][0] if at_cut else None,
            "cut_edge_kept": bool(edge_w is not None and edge_w in h["lab"]["kept_cut"]),
            "side": side, "pos": pos, "kind": rd["kind"],
            "reading_part": " ".join(w[0] for w in ins if id(w) in num),
            "cut_words": len(cut), "kept_cut_words": len(h["lab"]["kept_cut"]), "edge_fracs": h["lab"]["fracs"],
            "cut_left": cl[-1] if cl else None, "cut_right": cr[0] if cr else None,
            "rows": sorted({w[4] for w in ins} | {w[4] for w in cut}), "src": [[w[4], w[5]] for w in ins]}


def _edge_ok(lw: list, gw: list, end: bool, dropped: str | None, kept: bool) -> bool:
    """One edge of a re-decoded half (end=True: the last word). A greedy edge token that is a piece of the dropped
    word there (its head at an end edge, its tail at a start edge; any length, not the whole word) is skipped; then
    the edge words must match, or, when the label's edge word is a kept cut word, the greedy token may be its
    head / tail with >= half its letters."""
    L = C.letters
    if not lw or not gw:
        return False
    i = -1 if end else 0
    d = L(dropped or "")
    piece = (lambda t: d.startswith(t)) if end else (lambda t: d.endswith(t))
    if d and gw[i] != lw[i] and gw[i] != d and piece(gw[i]):
        gw = gw[:-1] if end else gw[1:]
    if not gw:
        return False
    e, g = lw[i], gw[i]
    part = e.startswith(g) if end else e.endswith(g)
    return g == e or (kept and part and len(g) >= max(2, len(e) / 2))


def seam_edges_ok(h: dict, greedy: str) -> bool:
    """Re-decode check of one half: the windows2 edge check (ml_p3_core.edges_match) with the cut edge read as a cut
    -- a piece of the dropped word may show, a kept cut word may come out shortened; hearing the WHOLE dropped word
    still fails (the word times are off). Far edge: the same rule with that edge's own cut word."""
    L = C.letters
    lw = [L(w) for w in h["text"].split() if L(w)]
    gw = [L(w) for w in greedy.split() if L(w)]
    kept = {x[0] for x in h.get("edge_fracs") or [] if x[2] == "kept"}
    first_kept = bool(h["text"].split() and h["text"].split()[0] in kept)
    last_kept = bool(h["text"].split() and h["text"].split()[-1] in kept)
    if h["side"] == "left":  # far edge = start (cut_left), cut edge = end
        return (_edge_ok(lw, gw, False, h.get("cut_left"), first_kept)
                and _edge_ok(lw, gw, True, h.get("cut_edge_dropped"), h.get("cut_edge_kept", last_kept)))
    return (_edge_ok(lw, gw, False, h.get("cut_edge_dropped"), h.get("cut_edge_kept", first_kept))
            and _edge_ok(lw, gw, True, h.get("cut_right"), last_kept))


SEAM_SKIPS = Counter()  # pairs seam_crops skipped (strict=False), per reason


def seam_crops(rows: list[dict], rng=None, positions: int = 0, win: float = C.WIN,
               num_keep: float | None = None, strict: bool = False) -> list[dict]:
    """Mirrored seam pairs over one stream of labelled rows (text, duration, word_times, num_chars, spans).
    positions = 0: every cut position of every reading; > 0: that many per reading, chosen by rng (balanced over the
    position names by the caller's rotation). Returns [{"pair": k, "reading": text, "kind", "pos", "left", "right"}].
    num_keep (ml_p3_core.num_keep_frac: compound number words): no cut inside a number word (IN_WORD puts 1/4 or 3/4 of
    it on a side, which that rule never labels), so those positions are not offered."""
    words_ = C.stream_words(rows)
    _, words, total = words_
    out = []
    for rd in readings(words, rows):
        cands = cut_positions(words, rd)
        if num_keep:
            cands = [(p, c) for p, c in cands if not p.startswith("in")]
        if positions and len(cands) > positions:
            cands = rng.sample(cands, positions) if rng is not None else cands[:positions]
        reading = " ".join(words[j][0] for j in rd["idx"])
        for pos, c in cands:
            p = seam_pair(words, total, c, win, num_keep)
            if p is None:
                continue
            if not pair_partitions(words, p, rd):  # a far edge moved past a reading word: the halves do not hold it
                assert not strict, (reading, pos)  # (a crash here cost a whole seam_crops run, 2026-10-17: skip it)
                SEAM_SKIPS["partition"] += 1
                continue
            out.append({"reading": reading, "kind": rd["kind"], "pos": pos, "n_words": len(rd["idx"]),
                        "left": _half_row(words, p["left"], "left", rd, pos),
                        "right": _half_row(words, p["right"], "right", rd, pos)})
    return out


def pos_class(pos: str, n_words: int) -> str:
    """before / after / gap (between two words of a reading) / in_word (a cut through a word of the reading)."""
    return "gap" if pos.startswith("gap") else "in_word" if pos.startswith("in") else pos


# --------------------------------------------------------------------------------- 2. number-free + look-alikes
SYM = re.compile(r"[0-9$€£¥¢₹₩₽%#/+°²³¹½¼¾⅓⅔&‰⁇]")
_TOK = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.U)

# articles / pronouns that are number-word strings (the spoken form of "1" is the same word: never a number to invent)
ARTICLES = {
    "de": {"ein", "eine", "einen", "einem", "einer", "eines"},
    "fr": {"un", "une"},
    "es": {"un", "una", "uno", "unos", "unas"},
    "it": {"un", "una", "uno"},
    "pt": {"um", "uma", "uns", "umas"},
    "nl": {"een"},
    "en": {"one"},
}
# pronoun uses ("one of them") -- regexes over the lowercased text
PRONOUN = {
    "de": r"\b(?:einer|eine|eines|einen) (?:der|des|von|unserer|ihrer|meiner)\b|\bder eine\b|\bdie eine\b",
    "fr": r"\bl'une?\b|\b(?:un|une) (?:des|de ces|d'entre)\b",
    "es": r"\b(?:uno|una) (?:de|del)\b|\bcada uno\b|\bel uno\b|\blos unos\b",
    "it": r"\b(?:uno|una) (?:dei|degli|delle|di)\b|\bl'uno\b|\bciascuno\b|\bognuno\b",
    "pt": r"\b(?:um|uma) (?:dos|das|de)\b|\bcada um\b",
    "nl": r"\been van\b|\bde een\b",
    "en": r"\bone of\b|\bno one\b|\bsomeone\b|\bthe one\b|\banyone\b|\beveryone\b",
}
# homophones / homographs of number words in number-free speech (the non-number spelling or sense)
HOMOPHONE = {
    "de": r"\b(?:außer acht|in acht (?:nehmen|genommen|zu nehmen)|acht (?:geben|gegeben|zu geben)|achtgeben|achten|"
          r"achtet|beachten|beachtet|zehen)\b",
    "fr": r"\b(?:sans|sang|sent|s'en|vin|vins|vain|d'eux|cette|cet)\b",  # cent / vingt / deux / sept
    "es": r"\b(?:siento|siente|sientes)\b",  # ciento / siete
    "it": r"\b(?:tu|non|se|ne|ci|lo|la|li|le|mi|ti|ma|che|dove|come|chi|quando|perché|già) sei\b|"
          r"\bsei (?:stato|stata|un|una|il|la|qui|qua|sicuro|sicura|tu|davvero|bravo|brava|pronto|pronta|d'accordo)\b",
    "pt": r"\b(?:sem|cento e|trás|atrás)\b",  # cem / três
    "nl": r"\b(?:ik acht|acht ik|geacht|achten|achtte|vieren|viert|gevierd)\b",  # acht / vier
    "en": r"\b(?:for|to|too|ate|won|fore)\b",  # four / two / eight / one
}
# homophone tokens that are themselves number-word strings: a row holding one is number-free only when HOMOPHONE
# matches it (it "sei" = you are, de "außer Acht", nl "ik acht") and no number context does
HOMO_TOKENS = {"de": {"acht", "achten"}, "it": {"sei"}, "nl": {"acht"}}
NUM_CONTEXT = {
    "it": r"\bsei (?:anni|mesi|giorni|ore|volte|milioni|mila|per cento|euro|punti|settimane|minuti|persone|paesi|gruppi|"
          r"stati|membri)\b",
    "de": r"\bacht (?:jahre|monate|tage|stunden|prozent|millionen|euro|mal|wochen|minuten|uhr|länder|mitgliedstaaten)\b|"
          r"\b(?:am|zum|vom|im|des|dem|den|der|die|das) achten\b",  # "am achten Mai" is the ordinal
    "nl": r"\bacht (?:jaar|maanden|dagen|uur|procent|miljoen|euro|keer|weken|minuten|landen|lidstaten)\b",
}


def norm_tokens(text: str) -> list[str]:
    s = text.lower().replace("’", "'")
    return _TOK.findall(s)


_LEX = {}
# basic number words (0-20, tens, hundreds, scales, ordinals 1-10) on top of the grammar table, whose entries cover only
# the values seen in captions; connectors inside a reading ("treinta y uno", "vingt et un") are not number words
BASE = {
    "de": "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn "
          "siebzehn achtzehn neunzehn zwanzig dreißig vierzig fünfzig sechzig siebzig achtzig neunzig hundert tausend "
          "million millionen milliarde milliarden erste erster ersten zweite zweiten dritte dritten vierte fünfte "
          "sechste siebte achte neunte zehnte hundertste",
    "fr": "zéro un une deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize vingt "
          "trente quarante cinquante soixante septante huitante octante nonante cent cents mille million millions "
          "milliard milliards premier première deuxième second seconde troisième quatrième cinquième sixième "
          "septième huitième neuvième dixième",
    "es": "cero uno una un dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis "
          "dieciseis diecisiete dieciocho diecinueve veinte treinta cuarenta cincuenta sesenta setenta ochenta noventa "
          "cien ciento cientos doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos "
          "novecientos mil millón millones primero primera segundo segunda tercero tercera cuarto cuarta quinto "
          "quinta sexto séptimo octavo noveno décimo",
    "it": "zero uno una un due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici "
          "quindici sedici diciassette diciotto diciannove venti trenta quaranta cinquanta sessanta settanta ottanta "
          "novanta cento mille mila milione milioni miliardo miliardi primo prima secondo seconda terzo terza quarto "
          "quinto sesto settimo ottavo nono decimo",
    "pt": "zero um uma dois duas três quatro cinco seis sete oito nove dez onze doze treze catorze quatorze quinze "
          "dezesseis dezasseis dezessete dezassete dezoito dezenove dezanove vinte trinta quarenta cinquenta sessenta "
          "setenta oitenta noventa cem cento duzentos trezentos quatrocentos quinhentos seiscentos setecentos "
          "oitocentos novecentos mil milhão milhões milhar bilhão primeiro primeira segundo segunda terceiro "
          "terceira quarto quinto sexto sétimo oitavo nono décimo",
    "nl": "nul een één twee drie vier vijf zes zeven acht negen tien elf twaalf dertien veertien vijftien zestien "
          "zeventien achttien negentien twintig dertig veertig vijftig zestig zeventig tachtig negentig honderd "
          "duizend miljoen miljard eerste tweede derde vierde vijfde zesde zevende achtste negende tiende",
    "en": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
          "billion first second third fourth fifth sixth seventh eighth ninth tenth",
}
CONNECTORS = {"de": {"und"}, "fr": {"et"}, "es": {"y"}, "it": {"e"}, "pt": {"e"}, "nl": {"en"}, "en": {"and"}}


def number_lexicon(lang: str) -> set:
    """Number-word tokens of a language: every whitespace / hyphen token of its card / ord / year readings in the
    ml_cands grammar table, plus BASE, minus CONNECTORS."""
    if lang not in _LEX:
        lex = set(BASE.get(lang, "").split())
        try:
            import ml_cands as M

            for k, v in M.table().items():
                if k.startswith(f"{lang}|") and k.split("|")[1] in ("card", "ord", "year"):
                    for w in v:
                        lex |= {t for t in re.split(r"[\s\-]+", w.lower()) if t}
        except Exception:  # noqa: BLE001  (no grammar table: the BASE words only)
            pass
        base = set(BASE.get(lang, "").split())
        if lang == "de":  # ordinal inflections: zehnte -> zehnten / zehnter / zehntes / zehntem
            lex |= {w + e for w in base if w.endswith("te") for e in ("n", "r", "s", "m")}
        if lang in ("it", "es", "pt"):  # gender / number: primo -> prima / primi / prime / primos / primas
            lex |= {w[:-1] + e for w in base if w.endswith(("o", "a")) for e in ("o", "a", "i", "e", "os", "as")}
        _LEX[lang] = lex - CONNECTORS.get(lang, set())
    return _LEX[lang]


# compounding: a token made only of number stems (+ an inner connector, + a number suffix) is a number word too
# ("zweieinhalb", "dreimal", "zwanzigjährigen", "tienduizenden", "duemilaventi", "veintitrés")
STEMS = {
    "de": "ein eins zwei drei vier fünf sechs sieben sieb acht neun zehn elf zwölf zwanzig dreißig vierzig fünfzig "
          "sechzig siebzig achtzig neunzig hundert tausend million millionen milliarde milliarden zig",
    "nl": "een twee drie vier vijf zes zeven acht negen tien elf twaalf der veer vijf zes zeven tach negen tig twintig "
          "dertig veertig vijftig zestig zeventig tachtig negentig honderd duizend",
    "it": "un uno una due tre tré quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici "
          "sedici diciassette diciotto diciannove venti vent trenta trent quaranta quarant cinquanta cinquant sessanta "
          "sessant settanta settant ottanta ottant novanta novant cento mille mila",
    "es": "veinti dieci uno una un dos tres trés cuatro cinco seis séis siete ocho nueve cientos ciento",
    "pt": "",
    "fr": "",
}
INFIX = {"de": ("und",), "nl": ("en", "ën"), "it": (), "es": (), "pt": (), "fr": ()}
SUFFIX = {"de": ("mal", "fach", "fache", "fachen", "jährig", "jährige", "jährigen", "jähriger", "halb", "einhalb",
                 "erjahre", "erjahren", "erjahrenden",
                 "tel", "tels", "stel", "steln", "ten", "te", "ter", "tes", "sten", "ste", "ster", "er", "ern", "s"),
          "nl": ("maal", "voudig", "jarig", "jarige", "de", "ste", "e", "en", "tal"),
          "it": ("esimo", "esima", "esimi", "esime", "enne", "enni"), "es": (), "pt": (), "fr": ()}


def _compound_num(tok: str, lang: str) -> bool:
    stems = set(STEMS.get(lang, "").split())
    if not stems or len(tok) < 5:
        return False
    pieces = stems | set(INFIX.get(lang, ()))
    suf = SUFFIX.get(lang, ())
    n = len(tok)
    ok = [False] * (n + 1)  # ok[i]: tok[:i] splits into >= 1 stems / infixes
    cnt = [0] * (n + 1)
    ok[0] = True
    for i in range(n):
        if not ok[i]:
            continue
        for j in range(i + 1, n + 1):
            p = tok[i:j]
            if p in pieces and (not ok[j] or cnt[j] < cnt[i] + (p in stems)):
                ok[j], cnt[j] = True, cnt[i] + (p in stems)
    if ok[n] and cnt[n] >= 2:
        return True
    # one stem + a suffix: not for the "one" stems ("einmal", "eenmaal", "einer" are everyday words)
    # one stem + a short inflection ("vieren" = to celebrate, "achter") is no number: those need >= 2 stems
    return any(tok.endswith(x) and ok[n - len(x)] and n - len(x) >= 3 and tok[:n - len(x)] not in ONE_STEMS
               and cnt[n - len(x)] >= (2 if x in WEAK_SUFFIX.get(lang, ()) else 1) for x in suf)


ONE_STEMS = {"ein", "eins", "een", "un", "uno", "una"}
WEAK_SUFFIX = {"de": {"en", "e", "s", "te", "ten", "ter", "tes", "ste", "sten", "ster"},  # "zwanziger" stays a number
               "nl": {"en", "e", "de", "ste", "er"}}  # "vieren" (to celebrate), "achter" (behind)


def lookalikes(text: str, lang: str) -> list[str]:
    """Control tags of a number-free text: "homophone", "pronoun", "article" (article = an ARTICLES token)."""
    t = text.lower().replace("’", "'")
    out = []
    if re.search(HOMOPHONE.get(lang, r"$^"), t):
        out.append("homophone")
    if re.search(PRONOUN.get(lang, r"$^"), t):
        out.append("pronoun")
    if set(norm_tokens(text)) & ARTICLES.get(lang, set()):
        out.append("article")
    return out


def number_free(text: str, lang: str, lex: set | None = None) -> bool:
    """No digit / symbol and no number word except the look-alikes (articles, and homophone tokens in a homophone
    context that is not a number context)."""
    if SYM.search(text):
        return False
    lex = number_lexicon(lang) if lex is None else lex
    toks = set(norm_tokens(text))
    toks |= {p for t in toks if "-" in t for p in t.split("-")}  # "dix-sept", "vingt-et-un"
    hits = (toks & lex) - ARTICLES.get(lang, set())
    hits |= {t for t in toks - lex - ARTICLES.get(lang, set()) if _compound_num(t, lang)}
    if not hits:
        return True
    t = text.lower().replace("’", "'")
    if not hits <= HOMO_TOKENS.get(lang, set()) or re.search(NUM_CONTEXT.get(lang, r"$^"), t):
        return False
    # every occurrence of the homophone token must sit inside a homophone-context match
    spans = [m.span() for m in re.finditer(HOMOPHONE.get(lang, r"$^"), t)]
    for tok in hits:
        for m in re.finditer(rf"(?<![^\W\d_]){re.escape(tok)}(?![^\W\d_])", t):
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                return False
    return True


SRC_CAP = 0.6  # at most this share of a language's picked plain hours from one source (the harvest rule)
CONTROL_ORDER = ("homophone", "pronoun", "article", "")


def plain_target_h(number_h: float, share: float, have_h: float = 0.0) -> float:
    """Hours of number-free speech still needed so number-free / (number-free + number) = share."""
    return max(0.0, share / (1 - share) * number_h - have_h)


def pick_plain(rows: list[dict], lang: str, hours: float, rng, lex: set | None = None) -> tuple[list[dict], dict]:
    """Up to `hours` of number-free rows of one language: rows failing number_free are skipped (counted), controls first
    (homophone, pronoun, article), each source <= SRC_CAP of the hours. Returns (picked, stats)."""
    lex = number_lexicon(lang) if lex is None else lex
    st = Counter()
    pool = defaultdict(list)
    for r in rows:
        if not number_free(r["text"], lang, lex):
            st["not_number_free"] += 1
            continue
        tags = lookalikes(r["text"], lang)
        r = dict(r, controls=tags)
        pool[next((t for t in CONTROL_ORDER if t in tags or t == ""), "")].append(r)
    for v in pool.values():
        rng.shuffle(v)
    out, h, by_src = [], 0.0, Counter()
    for tag in CONTROL_ORDER:
        for r in pool.get(tag, []):
            if h >= hours * 3600:
                break
            s = r.get("src") or "?"
            if by_src[s] + r["duration"] > SRC_CAP * hours * 3600 and len({x.get("src") for x in rows}) > 1:
                continue
            out.append(r)
            by_src[s] += r["duration"]
            h += r["duration"]
            st[f"picked_{tag or 'plain'}"] += 1
    st["picked_h"] = round(h / 3600, 3)
    for t in ("homophone", "pronoun", "article"):
        st[f"rows_with_{t}"] = sum(t in r["controls"] for r in out)
    return out, dict(st)


# ----------------------------------------------------------------------------------------------- 3. dedupe
MIN_KEY_CHARS = 20  # a normalised caption shorter than this ("Danke.", "2009.") is not an utterance identity
MIN_KEY_WORDS = 4  # ... nor one of fewer words ("150 reais.", "Vielen Dank, Herr Präsident.")
NEAR_JACCARD = 0.8
_VOX_SFX = re.compile(r"_(\d{8}-\d{2}:\d{2}:\d{2}_\d+)$")
_YT = re.compile(r"[a-z]{2}\d+_\d+_([A-Za-z0-9_-]{11})_(\d+)_(\d+)_(\d+)_(\d+)$")
_PIECE = re.compile(r"_p(\d+)$")


def norm_text(t: str) -> str:
    """Lowercase, accents folded, letters + digits only (digits kept: '21 Leute' != '22 Leute')."""
    s = unicodedata.normalize("NFKD", (t or "").lower().replace("ß", "ss"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9а-яієґ]", "", s)


def _base_id(rid: str) -> tuple[str, int | None]:
    m = _PIECE.search(str(rid))
    return (str(rid)[:m.start()], int(m.group(1))) if m else (str(rid), None)


def session_key(rid: str) -> str | None:
    """VoxPopuli session (event without the language tag): '20090113-0900-PLENARY-12-de_...' and Granary
    '20090113-0900-PLENARY-12_de_104' -> '20090113-0900-PLENARY-12'."""
    m = re.match(r"(\d{8}-\d{4}-[A-Z]+(?:-[A-Z0-9]+)*?)[-_](?:de|es|fr|it|nl|pt|en|[a-z]{2})(?:_|$)", str(rid))
    return m.group(1) if m else None


def audio_key(row: dict):
    """(kind, key, (start, end) or None): the recording a row comes from. VoxPopuli labelled: the utterance time suffix
    (one utterance can sit under two agenda ids). YODAS: (video, time range from the id; a piece shifts it).
    Granary-VoxPopuli: ('gv', session, None), matched to labelled rows of the same session by text. Other: id."""
    rid, _ = _base_id(row.get("id", ""))
    m = _VOX_SFX.search(rid)
    if m:
        return ("vox", m.group(1), None)
    m = _YT.search(rid)
    if m:
        s = int(m.group(2)) + int(m.group(3)) / 100
        e = s + int(m.group(4)) + int(m.group(5)) / 100
        if row.get("piece"):
            s, e = s + row["piece"][0], s + row["piece"][1]
        return ("yt", m.group(1), (round(s, 2), round(e, 2)))
    sk = session_key(rid)
    if sk:
        return ("gv", sk, None)
    return ("id", rid, None)


def _overlap(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0])) if inter > 0 else 0.0


def shingles(t: str, n: int = 3) -> set:
    w = [norm_text(x) for x in (t or "").split()]
    w = [x for x in w if x]
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _nums(r: dict) -> tuple:
    """The digit runs of a row's caption, as a sorted tuple (training near-duplicates must carry the same numbers)."""
    return tuple(sorted(re.findall(r"\d+", _caption(r))))


def _caption(r: dict) -> str:
    """The text that identifies an utterance: the source caption when the row has one, else its label."""
    return r.get("caption") or r.get("text") or ""


def text_key(r: dict) -> str | None:
    t = _caption(r)
    k = norm_text(t)
    return k if len(k) >= MIN_KEY_CHARS and len(t.split()) >= MIN_KEY_WORDS else None


def _read_prompt(r: dict) -> bool:
    """FLEURS rows: several speakers read the same prompt; the same text there is NOT the same recording."""
    return str(r.get("id", "")).startswith(("fleurs", "fleursval", "fleursdev")) or r.get("src") == "fleurs"


def _rank(r: dict) -> tuple:
    """Which copy of a duplicate group training keeps: not decided by a second recognizer, then lower CTC CER, then id."""
    ext = any(sp.get("by") in EXT_BY for sp in r.get("spans") or [])
    return (ext, r.get("cer_ctc", 1.0) if r.get("cer_ctc") is not None else 1.0, str(r.get("id")))


def dedupe(train: list[dict], protected: list[dict] = (), near: bool = True, within: bool = True,
           part_of=None) -> tuple[list[dict], dict]:
    """One row per utterance. `train` = every training row of one language (all passes / sources / parts);
    `protected` = held-out / validation / decision / gate rows: never removed, and any training row that duplicates one
    is removed. within=False: only the protected check (windows, which overlap by design). part_of(row) -> part name for
    the report (default row["part"]). Returns (kept training rows in input order, report)."""
    part_of = part_of or (lambda r: r.get("part", "?"))
    rep = Counter()
    why = {}
    prot_text = {text_key(p) for p in protected} - {None}
    prot_vox, prot_yt, prot_gv = set(), defaultdict(list), defaultdict(list)
    for p in protected:
        k, key, rng_ = audio_key(p)
        if k == "vox":
            prot_vox.add(key)
        elif k == "yt":
            prot_yt[key].append(rng_)
        sk = session_key(_base_id(p.get("id", ""))[0])
        if sk:
            prot_gv[sk].append(shingles(_caption(p)))
    prot_sh = [shingles(_caption(p)) for p in protected if text_key(p)]
    sh_index = defaultdict(list)
    for i, s in enumerate(prot_sh):
        for g in s:
            sh_index[g].append(i)

    def hit_protected(r):
        k, key, rg = audio_key(r)
        if k == "vox" and key in prot_vox:
            return "heldout_audio_vox"
        if k == "yt" and any(_overlap(rg, q) > 0.5 for q in prot_yt.get(key, [])):
            return "heldout_audio_yt"
        sk = session_key(_base_id(r.get("id", ""))[0])
        if sk and sk in prot_gv:
            s = shingles(_caption(r))
            if any(len(s & q) >= 0.5 * min(len(s), len(q)) and min(len(s), len(q)) >= 3 for q in prot_gv[sk]):
                return "heldout_session_text"
        t1 = text_key(r)
        if t1 and t1 in prot_text:
            return "heldout_text"
        if near and t1:
            s = shingles(_caption(r))
            cand = Counter(i for g in s for i in sh_index.get(g, ()))
            if any(_jac(s, prot_sh[i]) >= NEAR_JACCARD for i, n in cand.items() if n >= 0.5 * len(s)):
                return "heldout_near_text"
        return None

    kept = []
    for r in train:
        h = hit_protected(r)
        if h:
            rep[f"{part_of(r)}:{h}"] += 1
            rep[h] += 1
            why[r["id"]] = h
        else:
            kept.append(r)
    if not within:
        rep["train_in"], rep["train_out"] = len(train), len(kept)
        return kept, {"counts": dict(rep), "removed": why}
    # training <-> training: group by audio / text keys, keep the best-ranked copy
    order = {id(r): i for i, r in enumerate(kept)}
    kept = sorted(kept, key=_rank)
    seen_vox, seen_yt, seen_text, seen_gv = set(), defaultdict(list), set(), defaultdict(list)
    out = []
    sh_out = defaultdict(list)
    out_sh, out_nums = [], []
    for r in kept:
        k, key, rg = audio_key(r)
        t1 = None if _read_prompt(r) else text_key(r)
        reason = None
        if k == "vox" and key in seen_vox:
            reason = "dup_audio_vox"
        elif k == "yt" and any(_overlap(rg, q) > 0.5 for q in seen_yt[key]):
            reason = "dup_audio_yt"
        elif t1 and t1 in seen_text:
            reason = "dup_text"
        else:
            sk = session_key(_base_id(r.get("id", ""))[0])
            s = shingles(_caption(r))
            if sk and any(len(s & q) >= 0.5 * min(len(s), len(q)) and min(len(s), len(q)) >= 3 for q in seen_gv[sk]):
                reason = "dup_session_text"
            elif near and t1:
                cand = Counter(i for g in s for i in sh_out.get(g, ()))
                nums = _nums(r)  # a changed number ("2030" vs "2050") is a different utterance, not a near-copy
                if any(_jac(s, out_sh[i]) >= NEAR_JACCARD and out_nums[i] == nums for i, n in cand.items()
                       if n >= 0.5 * len(s)):
                    reason = "dup_near_text"
        if reason:
            rep[f"{part_of(r)}:{reason}"] += 1
            rep[reason] += 1
            why[r["id"]] = reason
            continue
        out.append(r)
        if k == "vox":
            seen_vox.add(key)
        elif k == "yt":
            seen_yt[key].append(rg)
        if t1:
            seen_text.add(t1)
        sk = session_key(_base_id(r.get("id", ""))[0])
        s = shingles(_caption(r))
        if sk:
            seen_gv[sk].append(s)
        if t1:
            for g in s:
                sh_out[g].append(len(out_sh))
            out_sh.append(s)
            out_nums.append(_nums(r))
    out.sort(key=lambda r: order[id(r)])
    rep["train_in"], rep["train_out"] = len(train), len(out)
    return out, {"counts": dict(rep), "removed": why}


def held_unit(rid: str) -> str | None:
    """The held-out unit of a row id (build4 / windows2): the VoxPopuli plenary day or the YODAS video."""
    rid = str(rid)
    m = re.match(r"(\d{8})-\d{4}-PLENARY", rid)
    if m:
        return "day:" + m.group(1)
    m = re.match(r"[a-z]{2}\d+_\d+_(.{11})_", rid)
    return "vid:" + m.group(1) if m else None


def take_seam(rows: list[dict], hours: float, rng) -> list[dict]:
    """Whole pairs (both halves) up to `hours`, round-robin over (position class, kind) so every cut position and every
    number class gets its share before any cell gets more."""
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair"]].append(r)
    cells = defaultdict(list)
    for pid, hs in sorted(pairs.items()):
        if len(hs) == 2:
            cells[(hs[0].get("pos_class"), hs[0].get("kind"))].append(hs)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells, key=str)
    out, h = [], 0.0
    while keys and h < hours * 3600:
        nxt = []
        for k in keys:
            if h >= hours * 3600:
                break
            if cells[k]:
                hs = cells[k].pop()
                out += hs
                h += sum(x["duration"] for x in hs)
            if cells[k]:
                nxt.append(k)
        keys = nxt
    return out
